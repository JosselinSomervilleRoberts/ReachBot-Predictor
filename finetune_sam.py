from generate_finetuning_dataset import load_bbox_coords, load_gt_masks, check_if_finetuning_dataset_exists
from segment_anything import SamPredictor, sam_model_registry
import configparser
from collections import defaultdict
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import cv2
import os
from statistics import mean
from torch.nn.functional import threshold, normalize

config = configparser.ConfigParser()
config.read("config.ini")
FINETUNE_DATA_FOLDER = config["PATHS"]["FINETUNE_DATASET"]

def finetune(lr: float = 1e-4, weight_decay:float = 0.0, num_epochs: int = 100, batch_size: int = 4, num_workers: int = 4, save_model: bool = True, n_train: int = 10):
    if not check_if_finetuning_dataset_exists():
        raise Exception("Finetuning dataset does not exist. Please generate it first.")
    
    # Load the dataset
    bbox_coords: dict = load_bbox_coords()
    ground_truth_masks: dict = load_gt_masks()

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)
    MODEL_TYPE = config["SAM"]["MODEL_TYPE"]
    MODEL_NAME = config["SAM"]["MODEL_NAME"]
    sam_model = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_NAME)
    sam_model = sam_model.to(device)
    sam_model.train()
    print("SAM model loaded")

    # Preprocess data
    transformed_data = defaultdict(dict)
    for k in tqdm(bbox_coords.keys(), desc="Preprocessing data"):
        img_folder = os.path.join(FINETUNE_DATA_FOLDER, "images", str(k) + ".png")
        image = cv2.imread(img_folder)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        input_image = transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=device)
        transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
        
        input_image = sam_model.preprocess(transformed_image)
        original_image_size = image.shape[:2]
        input_size = tuple(transformed_image.shape[-2:])

        transformed_data[k]['image'] = input_image
        transformed_data[k]['input_size'] = input_size
        transformed_data[k]['original_image_size'] = original_image_size

    # Setup optimizer, loss function, and data loader
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.MSELoss()
    # loss_fn = torch.nn.BCELoss()
    keys = list(bbox_coords.keys())

    losses = []
    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_losses = []
        # Just train on the first x examples
        for k in keys[:n_train]:
            input_image = transformed_data[k]['image'].to(device)
            input_size = transformed_data[k]['input_size']
            original_image_size = transformed_data[k]['original_image_size']
            
            # No grad here as we don't want to optimise the encoders
            with torch.no_grad():
                image_embedding = sam_model.image_encoder(input_image)
                
                prompt_box = bbox_coords[k]
                box = transform.apply_boxes(prompt_box, original_image_size)
                box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
                box_torch = box_torch[None, :]
                
                sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
                    points=None,
                    boxes=box_torch,
                    masks=None,
                )
            low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            )

            upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
            binary_mask = normalize(threshold(upscaled_masks, 0.0, 0))

            gt_mask_resized = torch.from_numpy(np.resize(ground_truth_masks[k], (1, 1, ground_truth_masks[k].shape[0], ground_truth_masks[k].shape[1]))).to(device)
            gt_binary_mask = torch.as_tensor(gt_mask_resized > 0, dtype=torch.float32)
            
            loss = loss_fn(binary_mask, gt_binary_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(epoch_losses)
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')

        if mean(epoch_losses) < best_loss:
            best_loss = mean(epoch_losses)
            if save_model:
                SAVE_FOLDER = config["PATHS"]["SAVE_MDOEL"]
                torch.save(sam_model.state_dict(), os.path.join(SAVE_FOLDER, f"{MODEL_TYPE}_{MODEL_NAME}_finetuned.pth"))

if __name__ == "__main__":
    finetune()