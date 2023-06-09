from segment_anything import sam_model_registry
import torch
from segment_anything.utils.transforms import ResizeLongestSide
from tqdm import tqdm
import cv2
import os
from torch.nn.functional import threshold, normalize
import numpy as np
import matplotlib.pyplot as plt
from toolbox.log import print_color
from toolbox.aws import shutdown
import argparse

from typing import Optional
import glob
import nvidia_smi
from torch.utils.data import Dataset, DataLoader
from evaluation.compute_metrics import compute_all_metrics, log_metrics
import wandb
from datetime import datetime
import gc
gc.enable()

# This is to solve a memory leak
# See: https://stackoverflow.com/questions/31156578/matplotlib-doesnt-release-memory-after-savefig-and-close
import matplotlib
matplotlib.use('Agg')

from data_utils.datasets import FullImageDataset, CroppedImageDataset
from data_utils.utils import get_device, set_description


def process_batch(sam_model, data, keep_grad: bool, device: str) -> torch.Tensor:
    input_image, gt_binary_masks, original_image_size = data
    gt_binary_masks = gt_binary_masks.to(device)
    # Reshape gt_binary_masks to (batch_size, H, W)
    gt_binary_masks = torch.squeeze(gt_binary_masks)

    # 1. Preprocess the image
    input_image_torch = torch.as_tensor(input_image, device=device)
    transformed_image = input_image_torch.permute(0, 2+1, 0+1, 1+1).contiguous()#[None, :, :, :]
    
    input_image = sam_model.preprocess(transformed_image).to(device)
    input_size = tuple(transformed_image.shape[-2:])


    # 2. Get prompt embeddings
    with torch.no_grad():
        image_embedding = sam_model.image_encoder(input_image)
        
        box = torch.Tensor([0,0,input_size[0], input_size[1]])
        box_torch = torch.as_tensor(box, dtype=torch.float, device=device)
        box_torch = box_torch[None, :]
        
        sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(
            points=None,
            boxes=box_torch,
            masks=None,
        )


    # 3. Apply decoder
    with torch.set_grad_enabled(keep_grad):
        low_res_masks, iou_predictions = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        # Postprocess each mask individually and then concatenate then to [batch_size, 1, H, W]
        binary_masks = torch.zeros((low_res_masks.shape[0], original_image_size[0][0], original_image_size[1][0]), device=device)
        # upscaled_masks = sam_model.postprocess_masks(low_res_masks, input_size, original_image_size).to(device)
        for i in range(low_res_masks.shape[0]):
            upscaled_masks_one_image = sam_model.postprocess_masks(low_res_masks[i].unsqueeze(0), input_size, [original_image_size[0][i], original_image_size[1][i]]).squeeze(0)
            binary_mask = normalize(threshold(upscaled_masks_one_image, 0.0, 0))
            # binary_mask has shape (num_masks, H, W)
            # Collapse all the masks into one and squeeze to get (H, W)
            binary_mask = torch.squeeze(torch.max(binary_mask, dim=0, keepdim=True)[0])
            binary_masks[i] = binary_mask

    return binary_masks, gt_binary_masks


def show_predictions(input_image, binary_masks, gt_binary_masks, save_path):
    plt.figure(figsize=(20,20))
    plt.subplot(1, 3, 1)
    plt.imshow(input_image[0].permute(0, 1, 2).detach().cpu().numpy())
    plt.axis('off')
    plt.subplot(1, 3, 2)
    if len(gt_binary_masks.shape) > 2:
        gt_binary_masks = gt_binary_masks[0]
    plt.imshow(gt_binary_masks.detach().cpu().numpy())
    plt.axis('off')
    plt.subplot(1, 3, 3)
    if len(binary_masks.shape) > 2:
        binary_masks = binary_masks[0]
    binary_mask = binary_masks.detach()
    binary_mask = (binary_mask / torch.max(binary_mask)) > 0.5
    plt.imshow(binary_mask.cpu().numpy())
    plt.axis('off')
    plt.savefig(save_path)

    # This is to empty matplotlib's memory (otherwise we get a memory leak)
    plt.cla() 
    plt.clf() 
    plt.close('all')

    # Garbage collector to prevent memory leaks
    gc.collect()


def train(sam_model, args, train_dataloader, val_dataloader):
    # Setup optimizer, loss function
    optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    device = sam_model.device

    # keep track of losses
    losses = []
    best_avg_metric = 0
    loss_of_batch = 0
    step = 0

    # Create directory to save plots and checkpoints
    # Adds the date and time to the run_name
    run_name = args.run_name + "_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    save_dir = os.path.join(args.save_dir, run_name)
    plot_dir = os.path.join(save_dir, "plots")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    optimizer.zero_grad()
    for epoch in range(args.num_epochs):

        epoch_losses = []

        description = f"Epoch {epoch} (train)"
        with tqdm(enumerate(train_dataloader), total=len(train_dataloader), unit="image", unit_scale=args.batch_size) as pbar:
            for k, data in pbar:
                set_description(pbar, description, k, frequency=1)
                binary_masks, gt_binary_masks = process_batch(sam_model, data, keep_grad=True, device=device)
                step += args.batch_size

                # Plot the gt_binary_mask and binary_mask using matplotlib
                if args.plot_every_batch > 0 and k % args.plot_every_batch == 0:
                    img_save_path = os.path.join(plot_dir, f"train_epoch_{epoch}_batch_{k}.png")
                    show_predictions(data[0], binary_masks, gt_binary_masks, img_save_path)

                    # Compute metrics
                    binary_masks = binary_masks.reshape(gt_binary_masks.shape)
                    if len(binary_masks.shape) == 2:
                        binary_masks = binary_masks.unsqueeze(0)
                        gt_binary_masks = gt_binary_masks.unsqueeze(0)
                    list_metrics = []
                    for i in range(len(gt_binary_masks)):
                        metrics = compute_all_metrics(ground_truth=1-gt_binary_masks[i], prediction_binary=(binary_masks[i] / torch.max(binary_masks[i])) < 0.5, separate_images=args.use_full_images)
                        list_metrics.append(metrics)
                    log_metrics(list_metrics, prefix="Train ", step=step)
                
                # Compute loss
                loss = loss_fn(binary_masks, gt_binary_masks.reshape(binary_masks.shape))
                loss_of_batch += loss
                loss.backward()
                wandb.log({"Training/loss": loss.item()}, step=step)

                if k % args.grad_accumulations == 0 or k == args.n_train - 1:
                    optimizer.step()
                    optimizer.zero_grad()

                epoch_losses.append(loss.item())
            losses.append(epoch_losses)
            
        description = f"Epoch {epoch} (val)  "
        list_metrics = []
        with tqdm(enumerate(val_dataloader), total=len(val_dataloader), unit="image", unit_scale=args.batch_size) as pbar:
            for k, data in pbar:
                set_description(pbar, description, k, frequency=1)
                binary_masks, gt_binary_masks = process_batch(sam_model, data, keep_grad=False, device=device)

                # Plot the gt_binary_mask and binary_mask using matplotlib
                if args.plot_every_batch > 0 and k % args.plot_every_batch == 0:
                    img_save_path = os.path.join(plot_dir, f"val_epoch_{epoch}_batch_{k}.png")
                    show_predictions(data[0], binary_masks, gt_binary_masks, img_save_path)

                binary_masks = binary_masks.reshape(gt_binary_masks.shape)
                if len(binary_masks.shape) == 2:
                    binary_masks = binary_masks.unsqueeze(0)
                    gt_binary_masks = gt_binary_masks.unsqueeze(0)
                for i in range(len(gt_binary_masks)):
                    metrics = compute_all_metrics(ground_truth=1-gt_binary_masks[i], prediction_binary=(binary_masks[i] / torch.max(binary_masks[i])) < 0.5, separate_images=args.use_full_images)
                    list_metrics.append(metrics)
        print("")
        avg_metric: float = log_metrics(list_metrics)
        print("")    

        # Creates dir checkpoints in save_dir if it doesn't exist
        checkpoint_dir = os.path.join(save_dir, "checkpoints")
        if args.save_model:
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(sam_model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"))
        if avg_metric > best_avg_metric:
            best_avg_metric = avg_metric
            if args.save_model:
                torch.save(sam_model.state_dict(), os.path.join(checkpoint_dir, "best.pth"))


def get_sam_model(model_type: str, checkpoint: Optional[str] = None, device: Optional[str] = None):
    assert model_type in ["vit_h", "vit_l", "vit_b"], "The model type provided {model_type} is not supported."
    if device is None: device = get_device()
    if checkpoint is None:
        if model_type == "vit_h": checkpoint = "sam_vit_h_4b8939.pth"
        if model_type == "vit_l": checkpoint = "sam_vit_l_0b3195.pth"
        if model_type == "vit_b": checkpoint = "sam_vit_b_01ec64.pth"

    if not os.path.exists(checkpoint): # os.path.join("sam_models", checkpoint)):
        print_color("Downloading SAM model", color="yellow")
        try:
            os.system(f"wget https://dl.fbaipublicfiles.com/segment_anything/{checkpoint}")
        except Exception as e:
            raise Exception(f"Failed to download SAM model. Please check your internet connection or SAM model name. Error: {e}")
    else:
        print_color("SAM model already downloaded", color="green")
    sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
    sam_model = sam_model.to(device)
    sam_model.train()
    print_color(f"SAM model successfully loaded on {device}", color="green")
    return sam_model


def finetune_new(args):
    nvidia_smi.nvmlInit()
    wandb.init(project=f"Reachbot-{args.class_name}", name=args.run_name)

    # Load model
    print_color("\nLoading model...", color="bold")
    sam_model = get_sam_model(args.model_type)
    transform = ResizeLongestSide(sam_model.image_encoder.img_size)

    # Load data
    print_color("\nLoading data...", color="bold")
    if args.use_full_images:
        training_data = FullImageDataset(class_name=args.class_name, train=True, n=args.n_train, device="cpu", transform=transform, collapse=True)
        val_data = FullImageDataset(class_name=args.class_name, train=False, n=args.n_val, device="cpu", transform=transform, collapse=True)
    else:
        training_data = CroppedImageDataset(class_name=args.class_name, train=True, n=args.n_train, device="cpu", transform=transform)
        val_data = CroppedImageDataset(class_name=args.class_name, train=False, n=args.n_val, device="cpu", transform=transform)
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=True)

    # Train
    print_color("\nTraining model...", color="bold")
    train(sam_model, args, train_dataloader, val_dataloader)

    wandb.finish()
    nvidia_smi.nvmlShutdown()


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune SAM")

    # For learning
    parser.add_argument("--class_name", type=str, default="cracks", help="Class to finetune")
    parser.add_argument("--model_type", type=str, default="vit_b", help="Type of SAM.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--save_model", type=bool, default=True, help="Whether to save the model")
    parser.add_argument("--n_train", type=int, default=-1, help="Number of training images")
    parser.add_argument("--n_val", type=int, default=-1, help="Number of test images")
    parser.add_argument("--grad_accumulations", type=int, default=32, help="Number of gradient accumulation steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--plot_every_batch", type=int, default=50, help="Plot predictions every")
    parser.add_argument("--use_full_images", type=bool, default=False, help="Whether to use full images or copped images")

    # For Logger
    parser.add_argument("--run_name", type=str, default="SAM", help="Name of the run")
    parser.add_argument("--save", type=bool, default=False, help="Whether to save to file")
    parser.add_argument("--save_dir", type=str, default="saves", help="Directory to save models and plots")

    # Usefull for AWS
    parser.add_argument("--shutdown", action="store_true", help="Whether to shutdown the instance after training")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    finetune_new(args)

    if args.shutdown:
        shutdown()