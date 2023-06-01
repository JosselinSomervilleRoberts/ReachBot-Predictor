# The purpose of this code is to finetune the Mask R-CNN model on the custom Labelbox dataset

# Custom function to load the dataset.
# The dataset is stored in the following format:
# a subfolder for each class
# each subfolder contains images, masks, and bboxes for that class
# the images are stored as .png files
# the masks are stored as .png files
# the bboxes are stored as .txt files containing the coordinates of the bboxes, each row being a pixel index

# Imports
import os
import torch
import torchvision
import cv2
from  torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import maskrcnn_resnet50_fpn
import pytorch_lightning as pl
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.models.detection.mask_rcnn import MaskRCNN
import torchvision.models as models
import numpy as np
import wandb
import matplotlib.pyplot as plt



def load_bbox_coords(class_name: str):
    """Loads the bounding box coordinates from the FINETUNE_DATASET_FOLDER folder."""
    bbox_coords = {}
    bboxes_path = os.path.join(f"./datasets/finetune/", class_name, "bboxes")
    for k in range(len(os.listdir(bboxes_path))):
        bbox_path = os.path.join(bboxes_path, str(k) + ".txt")
        with open(bbox_path, "r") as f:
            bbox = np.array([int(x) for x in f.read().split()])
        bbox_coords[int(k)] = bbox
    return bbox_coords

def load_mask(class_name: str, key: str):
    # Load the mask
    return Image.open(f"./datasets/finetune/{class_name}/masks/{key}.png").convert("L")

def load_image(class_name: str, key: str):
    # Load the image
    return Image.open(f"./datasets/finetune/{class_name}/images/{key}.png").convert("RGB")

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_folder, class_name):
        self.data_folder = data_folder
        self.class_name = class_name
        self.keys = list(load_bbox_coords(class_name).keys())
        self.bboxes = load_bbox_coords(class_name)

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        key = self.keys[index]
        img_path = os.path.join(self.data_folder, self.class_name, 'images', f"{key}.png")
        mask_path = os.path.join(self.data_folder, self.class_name, 'masks', f"{key}.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Get the bounding box
        bbox = self.bboxes[key]

        # Add some padding to the bounding box
        bbox = (bbox[0] - 10, bbox[1] - 10, bbox[2] + 10, bbox[3] + 10)

        # Crop the bounding box so that it is within the image
        bbox = (max(bbox[0], 0), max(bbox[1], 0), min(bbox[2], mask.width), min(bbox[3], mask.height))

        # Convert the bounding box to a numpy array
        bbox = np.array(bbox)

        # Convert PIL image to tensors
        image = np.array(image)
        # image = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

        gt_grayscale = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (gt_grayscale == 0).astype(int)
        # mask = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]


        # Resize
        image = cv2.resize(image.astype('float32'), (256,256))
        mask = cv2.resize(mask.astype('float32'), (256,256))
        mask = torch.tensor(mask).unsqueeze(0)
        image = torch.tensor(image)
        image = image.permute((2, 0, 1))

        # Normalize image
        # image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Convert the bbox to tensor
        bbox = torch.tensor(bbox).unsqueeze(0)
    
        target = {'boxes': bbox,
                  'masks': mask,
                  'labels': torch.tensor([1])
                  }
        return image, target
    



# Main function
def finetune_mask_rcnn(class_name: str = "boulder", batch_size: int = 8, lr: float = 0.001, weight_decay: float = 0.0005, num_epochs: int = 10, log_wandb: bool = False):
    # Use GPU if available
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('device:', device)

    # Load the dataset
    dataset = CustomDataset("./datasets/finetune", class_name)

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create the dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # Load the pretrainded Mask RCNN model
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Setup optimizer, loss function, and data loader
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Define lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Initialize Weights & Biases
    if log_wandb:
        wandb.init(
            # set the wandb project where this run will be logged
            project="NASA-ReachBot",

            name=f"Mask-RCNN_Finetuning_class={class_name}_batch_size={batch_size}_lr={lr}_weight_decay={weight_decay}_epochs={num_epochs}",
            
            # track hyperparameters and run metadata
            config={
            "classes": class_name,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "architecture": "Mask-RCNN",
            "dataset": "Custom Labelbox Dataset",
            "train_size": train_size,
            "test_size": test_size,
            "epochs": num_epochs,
            }
        )

    for epoch in range(num_epochs):
        print(f"-------- Epoch {epoch} --------")
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_dataloader, epoch, loss_fn, log_wandb)
        # update the learning rate
        lr_scheduler.step()
        # Path to saved model
        model_path = f"./models/mask_rcnn_finetuned_{class_name}_epochs_{epoch}_lr_{lr}_weightdecay_{weight_decay}.pt"
        # Save the model
        torch.save(model.state_dict(), model_path)
        # evaluate on the test dataset
        evaluate(model, epoch, model_path, test_dataloader, log_wandb)


    wandb.finish()




def train_one_epoch(model, optimizer, train_dataloader, epoch, criterion, log_wandb):
    # Set model to training mode
    model.train()
    # Iterate over the dataset
    iteration = 0
    for images, targets in train_dataloader:
        # Move images and targets to device
        #images = list(image.to(device) for image in images)
        #print(targets)
        #targets = list(target.to(device) for target in targets)


        # Zero out the gradients
        optimizer.zero_grad()

        # Forward pass
        targets = [{'boxes': targets['boxes'][i], 'labels': targets['labels'][i], 'masks': targets['masks'][i]} for i in range(len(targets['labels']))]
        loss_dict = model(images, targets)
        # Backward pass
        # loss = criterion(predictions, targets)
        loss = loss_dict['loss_mask']
        loss.backward()
        optimizer.step()

        # log metrics to Weights & Biases
        if log_wandb:
            wandb.log({"Epoch": epoch,
                       "loss_mask": loss_dict['loss_mask'],
                       "loss_box_reg": loss_dict['loss_box_reg'],
                       "loss_classifier": loss_dict['loss_classifier'],
                       "loss_objectness": loss_dict['loss_objectness'],
                       "loss_rpn_box_reg": loss_dict['loss_rpn_box_reg']})

        # Print training loss
        print(f"Epoch {epoch} Iteration #{iteration} loss_mask: {loss_dict['loss_mask']}")

        iteration += 1


def evaluate(model, epoch, model_path, test_dataloader, log_wandb):
    # Load the model
    # model.load_state_dict(torch.load(model_path))
    # Set model to evaluation mode
    model.eval()
    # Iterate over the dataset
    iteration = 0
    for images, targets in test_dataloader:

        predictions = model(images)

        for i in range(len(predictions)):

            # Combine the original mask with the original image
            mask = targets['masks'][i,0].detach().cpu().numpy()
            mask = np.where(mask > 0.5, 1, 0)
            mask = np.where(mask == 1, 255, 0)
            mask = np.array(mask, dtype=np.uint8)
            mask = np.stack((mask,)*3, axis=-1)
            # mask is green
            mask[:,:,0] = 0
            mask[:,:,2] = 0

            image = images[i].detach().cpu().permute((1, 2, 0))
            image = np.array(image, dtype=np.uint8)
            mask = cv2.addWeighted(mask, 0.5, image, 1, 0)

            # Save the original mask + original image combined
            plt.imsave(f"./results/mask_rcnn_finetuned_epoch_{epoch}_iteration_{iteration}_i_{i}_original.png", mask, cmap='gray')


            for j in range(len(predictions[i]['masks'])):
                mask=predictions[i]['masks'][j,0].detach().cpu().numpy()
                score=predictions[i]['scores'][j].detach().cpu().numpy()

                # log metrics to Weights & Biases
                if log_wandb:
                    wandb.log({"score": score})

                # Combine predicted mask and original image in one image
                mask = np.where(mask > 0.5, 1, 0)
                mask = np.where(mask == 1, 255, 0)
                mask = np.array(mask, dtype=np.uint8)
                mask = np.stack((mask,)*3, axis=-1)
                # mask is red
                mask[:,:,1] = 0
                mask[:,:,2] = 0

                image = images[i].detach().cpu().permute((1, 2, 0))
                image = np.array(image, dtype=np.uint8)
                mask = cv2.addWeighted(mask, 0.5, image, 1, 0)

                # Save the predicted nask + original image combined
                plt.imsave(f"./results/mask_rcnn_finetuned_epoch_{epoch}_iteration_{iteration}_i_{i}_j__{j}.png", mask, cmap='gray')


                # Print test score
                print(f"Test Iteration #{iteration} score: {score}")

        iteration += 1
    



if __name__ == "__main__":
    finetune_mask_rcnn(class_name="boulder",
         batch_size=8,
         lr=0.001,
         weight_decay=0.0005,
         num_epochs=1,
         log_wandb=False)
    

    
    # batch_size = 8

    # # Use GPU if available
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print('device:', device)

    # # Load the dataset
    # dataset = CustomDataset(data_folder="./datasets/finetune", class_name="boulder")

    # # Split the dataset into train and test sets
    # train_size = int(0.8 * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # # Create the dataloaders
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # # Load the pretrainded Mask RCNN model
    # model = maskrcnn_resnet50_fpn()
    # model.load_state_dict(torch.load("./models/mask_rcnn_finetuned_boulder_epochs_9_lr_0.001_weightdecay_0.0005.pt"))

    # evaluate(model, 9, test_dataloader, False)