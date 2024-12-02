import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import torch
import cv2
import random
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision

import segment_anything.utils.misc as utils
from dataloader_coco import build_dataset
from segment_anything import sam_model_registry
import segment_anything.utils.misc as utils
from segment_anything.network import MedSAM



# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# %% set up parser
parser = argparse.ArgumentParser()

parser.add_argument('--device', default='cuda', help='device to use for training / testing')
parser.add_argument("--task_name", type=str, default="MedSAM-ViT-L")
parser.add_argument("--model_type", type=str, default="vit_l")
parser.add_argument("--checkpoint", type=str, default="./exp/checkpoint/sam_vit_l_0b3195.pth")
parser.add_argument("--pretrain_model_path", type=str, default="./exp/box_model/checkpoint_epoch_240.pth")
parser.add_argument("--work_dir", type=str, default="./exp/box_model")
parser.add_argument("--trained_model", type=str, default="./exp/box_model")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--threshold", type=int, default=0.5)


#dataset parameters
parser.add_argument('--dataset_file', default='coco')
parser.add_argument('--coco_path', type=str)
parser.add_argument('--remove_difficult', action='store_true')
parser.add_argument("--batch_size", type=int, default=1)

# Loss
parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                help="Disables auxiliary decoding losses (loss at each layer)")
parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
# Matcher
parser.add_argument('--set_cost_class', default=1, type=float,
                help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', default=5, type=float,
                help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', default=2, type=float,
                help="giou box coefficient in the matching cost")

# * Loss coefficients
parser.add_argument('--mask_loss_coef', default=1, type=float)
parser.add_argument('--dice_loss_coef', default=1, type=float)
parser.add_argument('--bbox_loss_coef', default=5, type=float)
parser.add_argument('--giou_loss_coef', default=2, type=float)
parser.add_argument('--eos_coef', default=0.1, type=float,
                help="Relative classification weight of the no-object class")

args = parser.parse_args()

device = args.device
#build dataloader
dataset_train = build_dataset(image_set='train', args=args)
dataset_val = build_dataset(image_set='val', args=args)
data_loader_train = DataLoader(dataset_train, batch_size=args.batch_size, 
                               collate_fn=utils.custom_collate, num_workers=args.num_workers)
data_loader_val = DataLoader(dataset_val, batch_size=args.batch_size, 
                             collate_fn=utils.custom_collate, num_workers=args.num_workers)

import pdb; pdb.set_trace()

#bbox coordinates are provided as XYXY : left top right bottom 

#color and category dictionary
color_dictionary = {
    1: (255, 105, 180), # Hot Pink - Shoats
    2: (255, 140, 0),   # Dark Orange - Cattle
    3: (0, 191, 255),   # Deep Sky Blue - Impala
    4: (255, 255, 224), # Light Yellow - Zebra
    5: (199, 21, 133),  # Medium Violet Red - Wildebeest
    6: (72, 61, 139),   # Dark Slate Blue - Buffalo
    7: (255, 69, 0),    # Red Orange - Topi
    8: (218, 112, 214)  # Orchid - Other
}


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2]-box[0], box[3]-box[1]
    # x0, y0 = box[0]-(w/2), box[1]-(h/2)
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="red", facecolor=(0, 0, 0, 0), lw=3)
    )

#load model
sam_model, criterion, postprocessors = sam_model_registry[args.model_type](checkpoint=None, args=args)
model = MedSAM(
    image_encoder=sam_model.image_encoder,
    mask_decoder=sam_model.mask_decoder, #100 objects per image
    prompt_encoder=sam_model.prompt_encoder,
).to(device)
model.eval()

#load trained-weights
if os.path.isfile(args.pretrain_model_path):
    ## Map model to be loaded to specified single GPU
    checkpoint = torch.load(args.pretrain_model_path, map_location=device)
    model.load_state_dict(checkpoint["model"])


def plot_points(image, labels, boxes, image_id):
    # import pdb; pdb.set_trace()
    image = np.transpose(image, (1,2,0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image -= image.min()
    image /= image.max()
    image = np.int32(image*255)
    print(image.shape, boxes.shape)
    # color = (0, 0, 255)
    for box, label in zip(boxes, labels):
        color = color_dictionary[label]
        l, t = int(box[0]), int(box[1])
        r, b = int(box[2]), int(box[3])
        image = cv2.rectangle(image, (l, t), (r, b), color, 2)
    img_name = f"./prediction_plots/{image_id}.jpg"
    cv2.imwrite(img_name, image)


#plot boxes and classes on the image
for step, data in enumerate(data_loader_val):
    image = data[0]
    targets = data[1]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    b,c,h,w = image.tensors.shape
    boxes_np = np.repeat(np.array([[0,0,h,w]]), args.batch_size, axis=0)
    image = image.to(device)
    outputs = model(image, boxes_np)

    #convert into boxes
    orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
    results = postprocessors['bbox'](outputs, orig_target_sizes)

    keep = results[0]['scores'] > args.threshold
    scores = results[0]['scores'][keep]
    boxes = results[0]['boxes'][keep]
    labels = results[0]['labels'][keep]
    bx = torchvision.ops.nms(boxes, scores, iou_threshold=0.4)
    boxes = boxes[bx].detach().cpu().numpy()
    scores = scores[bx].detach().cpu().numpy()
    labels = labels[bx].detach().cpu().numpy()

    #get the highest confidence scores
    # scores = results[0]['scores'][keep].detach().cpu().numpy()
    # boxes = results[0]['boxes'][keep].detach().cpu().numpy()
    # labels = results[0]['labels'][keep].detach().cpu().numpy()
    image = image.tensors[0].detach().cpu().numpy()
    image_id = targets[0]['image_id'].detach().cpu().numpy()
    print("targets", targets[0]['boxes'].shape, labels)
    plot_points(image, labels, boxes, image_id)

    if step == 240:
        break