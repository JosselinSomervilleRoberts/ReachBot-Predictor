from generate_finetuning_dataset import load_bbox_coords, load_gt_masks, check_if_finetuning_dataset_exists, show_box, show_mask, generate_finetuning_dataset
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
import numpy as np
import matplotlib.pyplot as plt
from toolbox.log import Logger, print_color, sdebug, warn
from toolbox.aws import shutdown
import argparse
from PIL import Image
import io

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune SAM")

    # For learning
    parser.add_argument("--class_name", type=str, default="boulder", help="Class to finetune")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--save_model", type=bool, default=True, help="Whether to save the model")
    parser.add_argument("--n_train", type=int, default=200, help="Number of training images")
    parser.add_argument("--n_test", type=int, default=50, help="Number of test images")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of gradient accumulation steps")

    # For Logger
    parser.add_argument("--verbose", type=bool, default=False, help="Whether to print to console")
    parser.add_argument("--save", type=bool, default=False, help="Whether to save to file")
    parser.add_argument("--save_path", type=str, default="logs", help="Directory to save logs to")
    parser.add_argument("--tensorboard", type=bool, default=True, help="Whether to use tensorboard")

    # Usefull for AWS
    parser.add_argument("--shutdown", action="store_true", help="Whether to shutdown the instance after training")

    return parser.parse_args()

l = None
config = configparser.ConfigParser()
config.read("config.ini")
FINETUNE_DATA_FOLDER = config["PATHS"]["FINETUNE_DATASET"]

def finetune(class_name: str, lr: float = 1e-4, weight_decay:float = 0.0, num_epochs: int = 100, save_model: bool = True, n_train: int = 10, grad_accumulations: int = 1):
    if not check_if_finetuning_dataset_exists():
        warn("Finetuning dataset not found. Generating it now.")
        generate_finetuning_dataset()
    
    # Load the dataset
    bbox_coords: dict = load_bbox_coords(class_name=class_name)
    ground_truth_masks: dict = load_gt_masks(class_name=class_name)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print_color(f"Using device: {device}", color="bold")
    MODEL_TYPE = config["SAM"]["MODEL_TYPE"]
    MODEL_NAME = config["SAM"]["MODEL_NAME"]
    # Check if checkpoints is already downloaded
    if not os.path.exists(MODEL_NAME):
        print_color("Downloading SAM model", color="yellow")
        try:
            os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/{MODEL_NAME}")
        except Exception as e:
            raise Exception(f"Failed to download SAM model. Please check your internet connection or SAM model name. Error: {e}")
    else:
        print_color("SAM model already downloaded", color="green")
    sam_model = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_NAME)
    sam_model = sam_model.to(device)
    sam_model.train()
    print_color("SAM model loaded", color="green")

    # Preprocess data
    transformed_data = defaultdict(dict)
    keys = list(bbox_coords.keys())
    for k in tqdm(keys[:n_train], desc="Preprocessing data"):
        img_folder = os.path.join(FINETUNE_DATA_FOLDER, class_name, "images", str(k) + ".png")
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

    losses = []
    best_loss = float('inf')

    loss_of_batch = 0
    optimizer.zero_grad()
    for epoch in range(num_epochs):
        epoch_losses = []
        # Just train on the first x examples
        for k in tqdm(keys[:n_train]):
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

            # Plot the gt_binary_mask and binary_mask
            if k % 100 == 0:
                l.log_image(f"Input image {k}", input_image[0])
                l.log_image(f"Ground truth mask {k}", gt_binary_mask[0][0])
                l.log_image(f"Predicted mask {k}", binary_mask[0][0])
            
            loss = loss_fn(binary_mask, gt_binary_mask)
            loss_of_batch += loss
            loss.backward()
            l.log_value("Loss", loss.item())

            if k % grad_accumulations == 0 or k == keys[n_train - 1]:
                optimizer.step()
                optimizer.zero_grad()

            epoch_losses.append(loss.item())
        losses.append(epoch_losses)
        print(f'EPOCH: {epoch}')
        print(f'Mean loss: {mean(epoch_losses)}')
        l.log_value("Mean epoch loss", mean(epoch_losses), index=epoch)

        if mean(epoch_losses) < best_loss:
            best_loss = mean(epoch_losses)
            if save_model:
                l.save_model(sam_model, f"{MODEL_TYPE}_{MODEL_NAME}_best.pth")
    return losses, sam_model

def compare_untrained_and_trained(class_name: str, trained_model, index: int):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_TYPE = config["SAM"]["MODEL_TYPE"]
    MODEL_NAME = config["SAM"]["MODEL_NAME"]
    SAVE_FOLDER = config["PATHS"]["SAVE_MDOEL"]
    sam_model_orig = sam_model_registry[MODEL_TYPE](checkpoint=MODEL_NAME)
    sam_model_orig = sam_model_orig.to(device)

    # Set the two models
    predictor_tuned = SamPredictor(trained_model)
    predictor_original = SamPredictor(sam_model_orig)

    # Load data
    bbox_coords: dict = load_bbox_coords(class_name)
    ground_truth_masks: dict = load_gt_masks(class_name)
    keys = list(bbox_coords.keys())

    # Display result on new data
    k = keys[index]
    img_folder = os.path.join(FINETUNE_DATA_FOLDER, class_name, "images", str(k) + ".png")
    image = cv2.imread(img_folder)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor_tuned.set_image(image)
    predictor_original.set_image(image)

    input_bbox = np.array(bbox_coords[k])

    masks_tuned, _, _ = predictor_tuned.predict(
        point_coords=None,
        box=input_bbox,
        multimask_output=False,
    )

    masks_orig, _, _ = predictor_original.predict(
        point_coords=None,
        box=input_bbox,
        multimask_output=False,
    )

    _, axs = plt.subplots(1, 3, figsize=(16, 4))

    axs[0].imshow(image)
    show_mask(masks_tuned, axs[0])
    show_box(input_bbox, axs[0])
    axs[0].set_title('Mask with Tuned Model', fontsize=26)
    axs[0].axis('off')

    axs[1].imshow(image)
    show_mask(masks_orig, axs[1])
    show_box(input_bbox, axs[1])
    axs[1].set_title('Mask with Untuned Model', fontsize=26)
    axs[1].axis('off')

    axs[2].imshow(image)
    show_mask(ground_truth_masks[k], axs[2])
    show_box(input_bbox, axs[2])
    axs[2].set_title('Ground Truth Mask', fontsize=26)
    axs[2].axis('off')

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    l.log_image(f"Comparison", im)
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    l = Logger(args.verbose, args.save, args.save_path, args.tensorboard)
    losses, trained_model = finetune(args.class_name, args.lr, args.weight_decay, args.num_epochs, args.save_model, args.n_train, args.gradient_accumulation_steps)

    # Compare on training data
    for index in range(min(args.n_train, args.n_test)):
        compare_untrained_and_trained(args.class_name, trained_model, index)

    # Compare on test data
    for index in range(args.n_train, args.n_train + args.n_test):
        compare_untrained_and_trained(args.class_name, trained_model, index)

    # Shutdown EC2 instance
    if args.shutdown:
        shutdown()