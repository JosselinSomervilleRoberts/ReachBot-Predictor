from generate_maskrcnn_dataset import generate_maskrcnn_dataset
import configparser
from PIL import Image
import os
import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import warnings
warnings.filterwarnings('ignore')
import torchvision.models.detection.mask_rcnn
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import wandb
from  evaluation.compute_metrics import compute_all_metrics, log_metrics
from tqdm import tqdm
from toolbox.printing import print_color

# config
config = configparser.ConfigParser()
config.read('./config.ini')
LABELBOX_DATASET_FOLDER = config["PATHS"]["LABELBOX_DATASET"]
MASKRCNN_DATASET_FOLDER = config["PATHS"]["MASKRCNN_DATASET"]
CLASS_NAMES = ["__background__", "boulder"]




# Generate a new appropriate dataset
# generate_maskrcnn_dataset()




# Custom Dataset
class RocksDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms

        # Load all image files, sorting them to ensure that they are aligned
        self.imgs = os.listdir(os.path.join(root, "images"))
        # Get the index of each image contained in the file name ('image_0.png', etc.)
        indices = [int(file_name.split("_")[1].split(".")[0]) for file_name in self.imgs if file_name.startswith("image_") and file_name.endswith(".png")]
        # Sort the image files by index
        self.imgs = ['image_' + str(i) + '.png' for i in sorted(indices)]


        # Do the same for the masks
        self.masks = os.listdir(os.path.join(root, "masks"))
        indices = [int(file_name.split("_")[1].split(".")[0]) for file_name in self.masks if file_name.startswith("mask_") and file_name.endswith(".png")]
        self.masks = ['mask_' + str(i) + '.png' for i in sorted(indices)]

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # print('obj_ids:', obj_ids, 'for index ', idx,'\n')
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# Build the model      
def build_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Stop here if you are fine-tunning Faster-RCNN

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

# ## In case the required helper functions are not installed, uncomment these lines
# !git clone https://github.com/pytorch/vision.git
# %cd vision
# !git checkout v0.3.0

# !cp references/detection/utils.py ../
# !cp references/detection/transforms.py ../
# !cp references/detection/coco_eval.py ../
# !cp references/detection/engine.py ../
# !cp references/detection/coco_utils.py ../


# Create train and tests datasets, and data loaders
def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def datasets_and_dataloaders(batch_size):
    # use our dataset and defined transformations
    dataset = RocksDataset(root=MASKRCNN_DATASET_FOLDER, transforms=get_transform(train=True))
    dataset_test = RocksDataset(root=MASKRCNN_DATASET_FOLDER, transforms=get_transform(train=False))

    # split the dataset in train and test set
    torch.manual_seed(10)
    np.random.seed(10)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    indices = np.random.permutation(range(len(dataset)))
    dataset = torch.utils.data.Subset(dataset, indices[:-test_size])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-test_size:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn)
    
    return dataset, dataset_test, data_loader, data_loader_test



def get_coloured_mask(mask):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[0, 255, 0],[0, 0, 255],[255, 0, 0],[0, 255, 255],[255, 255, 0],[255, 0, 255],[80, 70, 180],[250, 80, 190],[245, 145, 50],[70, 150, 250],[50, 190, 190]]
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[random.randrange(0,10)]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask

def get_prediction(img_path, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0
    
    """
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)

    img = img.to(device)
    pred = model([img])
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x>confidence]
    if pred_t == []:
        return [], [], []
    else:
        pred_t = pred_t[-1]
        masks = (pred[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        # print(pred[0]['labels'].numpy().max())
        pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
        masks = masks[:pred_t+1]
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        return masks, pred_boxes, pred_class

def segment_instance(img_path, confidence=0.5, rect_th=2, text_size=2, text_th=2):
    """
    segment_instance
      parameters:
        - img_path - path to input image
        - confidence- confidence to keep the prediction or not
        - rect_th - rect thickness
        - text_size
        - text_th - text thickness
      method:
        - prediction is obtained by get_prediction
        - each mask is given random color
        - each mask is added to the image in the ration 1:0.8 with opencv
        - final output is displayed
    """
    masks, boxes, pred_cls = get_prediction(img_path, confidence)
    if masks == [] and boxes == [] and pred_cls == []:
        print('No boulders detected for this confidence threshold')
        return
    else:
      img = cv2.imread(img_path)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      for i in range(len(masks)):
        rgb_mask = get_coloured_mask(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, (int(boxes[i][0][0]), int(boxes[i][0][1])), (int(boxes[i][1][0]), int(boxes[i][1][1])), color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img,pred_cls[i], (int(boxes[i][0][0]), int(boxes[i][0][1])), cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=text_th)
      plt.figure(figsize=(20,30))
      plt.imshow(img)
      plt.xticks([])
      plt.yticks([])
      plt.show()


# Inference
def visualize_results(num_test_images):
    # load the model
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=2)
    model.load_state_dict(torch.load(f"./models/maskrcnn_finetuned_model_hyper_1_epochs_20_batch_size_1.pt"))
    # model.load_state_dict(torch.load(f"./models/maskrcnn_finetuned_model_hyper_1_epochs_20_batch_size_12.pt"))
    # set to evaluation mode
    model.eval()
    CLASS_NAMES = ['__background__', 'boulder']
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    test_set_indices = indices[-test_size:]
    imgs = os.listdir(os.path.join(MASKRCNN_DATASET_FOLDER, "images"))
    images_file_indices = [int(file_name.split("_")[1].split(".")[0]) for file_name in imgs if file_name.startswith("image_") and file_name.endswith(".png")]
    images_file_indices = images_file_indices[-test_size:]

    for i in range(num_test_images):

        image_file_index = images_file_indices[i]

        if image_file_index not in images_file_indices:
            print('The image corresponding to the index is not present in the test dataset')
        else:
            test_image_path = f'./datasets/maskrcnn/images/image_{image_file_index}.png'
            segment_instance(test_image_path, confidence=0.6, rect_th=2, text_size=2, text_th=2)




# Parameters:
# - learning rate
# - momentum
# - weight decay
# - number of epochs
# - batch size
# - optimizer
# - scheduler
# - hidden layer size



# Custom functions for hyperparameter tuning

def build_model_hyper(num_classes, hidden_layer_size):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Stop here if you are fine-tunning Faster-RCNN

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = hidden_layer_size
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

@torch.no_grad()
def evaluate_hyper(model, data_loader, device, log_wandb):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # Average the loss over the test dataset
    loss = 0
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 100, header):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()

            # get the loss_mask
            loss += loss_value

            if log_wandb:
                wandb.log({
                        "eval_loss_classifier": loss_dict_reduced['loss_classifier'],
                        "eval_loss_box_reg": loss_dict_reduced['loss_box_reg'],
                        "eval_loss_mask": loss_dict_reduced['loss_mask'],
                        "eval_loss_objectness": loss_dict_reduced['loss_objectness'],
                        "eval_loss_rpn_box_reg": loss_dict_reduced['loss_rpn_box_reg']})

    loss = loss / len(data_loader)

    if log_wandb:
        wandb.log({"eval_loss": loss})

    torch.set_num_threads(n_threads)
    return loss

def evaluate_hyper_custom(model, data_loader_test, device, log_wandb):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    # metric_logger = utils.MetricLogger(delimiter="  ")
    # header = 'Test:'

    model.eval()

    metrics_custom_list = []

    for images, targets in tqdm(data_loader_test, desc="Evaluating (image)", total=len(data_loader_test)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        predictions = model(images)

        pred_masks = (predictions[0]['masks']).squeeze().detach().cpu().numpy()
        pred_masks = np.array(pred_masks)

        pred_masks_binary = (predictions[0]['masks']>0.5).squeeze().detach().cpu().numpy()
        pred_masks_binary = np.array(pred_masks_binary)

        ground_truth_mask = targets[0]['masks'].squeeze().detach().cpu().numpy()
        ground_truth_mask = np.array(ground_truth_mask)

        if pred_masks.shape[0] == 0:
            pred_mask = np.zeros(pred_masks.shape[1], pred_masks.shape[2])
            pred_mask_binary = np.zeros(pred_masks.shape[1], pred_masks.shape[2])
        else:
            # merge all pred_masks into one pred_mask
            pred_mask = np.max(pred_masks, axis=0)
            
            # merge all pred_masks_binary into one pred_mask_binary
            pred_mask_binary = np.zeros((pred_masks_binary.shape[1], pred_masks_binary.shape[2]))
            for i in range(pred_masks_binary.shape[0]):
                pred_mask_binary += pred_masks_binary[i,:,:]
            # convert boolean values to integers
            pred_mask_binary = pred_mask_binary.astype(int)

        # merge all ground_truth_mask channels into one ground_truth_mask
        if len(ground_truth_mask.shape) > 2:
            ground_truth_mask = np.max(ground_truth_mask, axis=0)

        metrics_custom_list.append(compute_all_metrics(ground_truth=ground_truth_mask,
                                                    prediction=pred_mask,
                                                    prediction_binary=pred_mask_binary,
                                                    threshold=0.5))

    log_metrics(metrics_custom_list, log_wandb)

    torch.set_num_threads(n_threads)
    # return the average of metrics_custom['average'] over all metrics_custom dictionaries in metrics_custom_list
    return np.mean([metrics_custom['full_mask']['average'] for metrics_custom in metrics_custom_list])

            

# Hyperparaneter tuning: for number of epochs and learning rate
def hyperparameter_tuning(params, log_wandb):
    # from toolbox.aws import shutdown

    # get the parameters
    learning_rates = params['learning_rates']
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    weight_decay = params['weight_decay']
    momentum = params['momentum']
    gamma = params['gamma']
    step_size = params['step_size']

    # get the data loaders
    dataset, dataset_test, data_loader, data_loader_test = datasets_and_dataloaders(batch_size)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and boulder
    num_classes = 2

    best_loss = 0

    for learning_rate in learning_rates:

        # get the model using our helper function
        model = build_model(num_classes)
        # move model to the right device
        model.to(device)

        # construct an optimizer with the default parameters
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=learning_rate,
                                    momentum=momentum, weight_decay=weight_decay)

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        
        # Initialize Weights & Biases
        if log_wandb:
            wandb.init(
                # set the wandb project where this run will be logged
                project="NASA-ReachBot",

                name=f"Mask-RCNN_Finetuning_class=boulder_hyper_tuning_2_learning_rate={learning_rate}",
                
                # track hyperparameters and run metadata
                config={
                "architecture": "Mask-RCNN",
                "dataset": "Custom Labelbox Dataset",
                "classes": "boulder",
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "momentum": momentum,
                "step_size": step_size,
                "gamma": gamma,
                "batch_size": batch_size,
                "train_size": len(dataset),
                "test_size": len(dataset_test),
                "epochs": num_epochs,
                }
            )

        print_color(f'\n----------Tuning for learning_rate={learning_rate}', color='purple')
        
        # train for num_epochs epochs
        for epoch in range(num_epochs):
            print_color(f"\n----------Epoch {epoch+1}/{num_epochs}", color="blue")
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, log_wandb=log_wandb)
            # update the learning rate
            # lr_scheduler.step()
            # evaluate on the test dataset
            loss = evaluate_hyper_custom(model, data_loader_test, device, log_wandb)
            # save the best model
            if loss > best_loss:
                best_loss = loss
                best_model = model
                best_learning_rate = learning_rate

        wandb.finish()

    print(f'--------The best model is the one with learning_rate={best_learning_rate}')
    # Save the best model
    torch.save(best_model.state_dict(), f"./models/maskrcnn_finetuned_model_hyper_2_epochs_{num_epochs}_learning_rate_{best_learning_rate}.pt")

    # shutdown()




if __name__ == '__main__':
    log_wandb = True

    params = {}
    params['num_epochs'] = 20
    params['learning_rates'] = [1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    params['batch_size'] = 8
    params['weight_decay'] = 5e-4
    params['momentum'] = 0.9
    params['step_size'] = 3
    params['gamma'] = 0.1

    hyperparameter_tuning(params, log_wandb)