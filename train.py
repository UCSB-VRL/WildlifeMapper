# -*- coding: utf-8 -*-
"""
train the image encoder and mask decoder
"""

# %% setup environment
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math

join = os.path.join
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler


from dataloader_coco import build_dataset
from segment_anything import sam_model_registry
import torch.nn.functional as F
import argparse
import random
from datetime import datetime
import segment_anything.utils.misc as utils
from segment_anything.network import MedSAM
from inference import evaluate, get_coco_api_from_dataset

#bowen 
import train_utils
# set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()

# torch.distributed.init_process_group(backend="gloo")

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


# %% set up parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tr_npy_path",
    type=str,
    default="data/npy/CT_Abd",
    help="path to training npy files; two subfolders: gts and imgs",
)
parser.add_argument("--task_name", type=str, default="MedSAM-ViT-L")
parser.add_argument("--model_type", type=str, default="vit_l")
parser.add_argument("--checkpoint", type=str, default="./exp/checkpoint/sam_vit_l_0b3195.pth")
parser.add_argument("--pretrain_model_path", type=str, default="")
parser.add_argument("--work_dir", type=str, default="./exp/box_model")
parser.add_argument("--trained_model", type=str, default="./exp/box_model")

# train
parser.add_argument("--num_epochs", type=int, default=550)
parser.add_argument("--batch_size", type=int, default=6)
parser.add_argument("--num_workers", type=int, default=8)

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

#dataset parameters
parser.add_argument('--dataset_file', default='coco')
parser.add_argument('--coco_path', type=str)
parser.add_argument('--remove_difficult', action='store_true')

parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
parser.add_argument('--device', default='cuda', help='device to use for training / testing')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
parser.add_argument('--eval', action='store_true')

# Optimizer parameters
parser.add_argument("--weight_decay", type=float, default=0.001, help="weight decay (default: 0.01)")
parser.add_argument("--lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
parser.add_argument('--lr_drop', default=40, type=int)
parser.add_argument("--use_wandb", type=bool, default=False, help="use wandb to monitor training")
parser.add_argument("--use_amp", action="store_true", default=False, help="use amp")
parser.add_argument("--resume", type=str, default="", help="Resuming training from checkpoint")

#distributed training
parser.add_argument("--distributed", type=str, default=False, help="Distributed training")

# bowen
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')


args = parser.parse_args()

# bowen need to put it here since code starts before main function. 
train_utils.init_distributed_mode(args)

#bbox coordinates are provided as XYXY : left top right bottom 
def show_box(box, ax):
    w, h = box[2], box[3]
    x0, y0 = box[0]-(w/2), box[1]-(h/2)
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )

# %% sanity test of dataset class
# tr_dataset = NpyDataset("data/npy/CT_Abd")
# tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
dataset_train = build_dataset(image_set='train', args=args)
dataset_val = build_dataset(image_set='val', args=args)
import cv2
for step, data in enumerate(dataset_train):
    break
    image = np.transpose(np.asarray(data['image']), (1,2,0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = data['target']['boxes']
    bboxes = bboxes*1024
    print(image.shape, bboxes.shape)
    # show the example
    _, axs = plt.subplots(1, figsize=(50, 50))
    idx = random.randint(0, 7)
    axs.imshow(image)
    for bboxe in bboxes:
        show_box(bboxe.numpy(), axs)
    axs.axis("off")
    plt.subplots_adjust(wspace=0.01, hspace=0)
    plt.savefig("./data_sanitycheck.png", bbox_inches="tight", dpi=300)
    plt.close()
    break

# distributed training dataset setup
if args.distributed:
    sampler_train = DistributedSampler(dataset_train)
    sampler_val = DistributedSampler(dataset_val, shuffle=False)
else:
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

batch_sampler_train = torch.utils.data.BatchSampler(
    sampler_train, args.batch_size, drop_last=True)

data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                            collate_fn=utils.custom_collate, num_workers=args.num_workers)
data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, 
                            drop_last=False, collate_fn=utils.custom_collate, num_workers=args.num_workers)

#for evaluation, coco_api
base_ds = get_coco_api_from_dataset(dataset_val)

# bowen
if args.use_wandb and train_utils.is_main_process():
    import wandb

    wandb.login()
    wandb.init(
        project="aerial_detection_project",
        name='no_prompt',
        config={
            "lr": args.lr,
            "batch_size": args.batch_size,
            "data_path": args.tr_npy_path,
            "model_type": args.model_type,
        },
    )

# %% set up model for training
device = args.device
run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, "sam-" + run_id)
device = torch.device(args.device)
# %% set up model

def main():

    seed = args.seed + train_utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    #initialize network
    sam_model, criterion, postprocessors = sam_model_registry[args.model_type](checkpoint=args.checkpoint, args=args)
    medsam_model = MedSAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder, #50 objects per image
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    # bowen
    model_without_ddp = medsam_model
    if args.distributed:
        medsam_model = torch.nn.parallel.DistributedDataParallel(medsam_model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = medsam_model.module

    medsam_model.train()
    criterion.train()

    # img_mask_encdec_params = list(medsam_model.image_encoder.parameters()) + 
    #                         list(medsam_model.mask_decoder.parameters())
    # bowen
    mask_prompt_params =  list(model_without_ddp.mask_decoder.parameters()) \
                        + list(model_without_ddp.prompt_encoder.parameters()) 
    hfc_adaptor_params = list(model_without_ddp.image_encoder.hfc_embed.parameters()) \
                        + list(model_without_ddp.image_encoder.patch_embed.parameters()) \
                        + list(model_without_ddp.image_encoder.hfc_attn.parameters())
    optimizer = torch.optim.AdamW([{"params" : mask_prompt_params},
                                   {"params": hfc_adaptor_params, "lr": 0.0001}], lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    #Num6er mask decoder parameters:
    train_params_list = [n for n, p in model_without_ddp.named_parameters() if p.requires_grad]
    print(train_params_list)

    # %% train
    num_epochs = args.num_epochs
    iter_num = 0
    best_loss = 1e10

    print("Number of training samples: ", dataset_train.__len__())

    start_epoch = 0
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("RESUMING TRAINING")
            ## Map model to be loaded to specified single GPU
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            # bowen
            # medsam_model.load_state_dict(checkpoint["model"])
            model_without_ddp.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint["optimizer"])
    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        #metric logging imported from DETR codebase
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 10

        if args.distributed:
            sampler_train.set_epoch(epoch)
        
        for data in metric_logger.log_every(data_loader_train, print_freq, header):
            optimizer.zero_grad()
            image = data[0]
            targets = data[1]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            #passing the whole image as prompt
            b,c,h,w = image.tensors.shape
            boxes_np = np.repeat(np.array([[0,0,h,w]]), args.batch_size, axis=0)
            image = image.to(device)
            outputs = medsam_model(image, boxes_np)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                            for k, v in loss_dict_reduced.items()}
            loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

            loss_value = losses_reduced_scaled.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            optimizer.zero_grad()            
            losses.backward()

            if args.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(medsam_model.parameters(), args.clip_max_norm)
            optimizer.step()

            metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            
            epoch_loss += losses.item()
            iter_num += 1

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        #learning rate schedular
        lr_scheduler.step()


        #Evaluation after each epoch
        test_stats, coco_evaluator = evaluate(
            medsam_model, criterion, postprocessors, data_loader_val, base_ds, device, args)

        
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                       'epoch': epoch}
        # bowen
        if args.use_wandb and train_utils.is_main_process():
            wandb.log(log_stats)
        print(
            f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}'
        )
        ## save the latest model
        checkpoint = {
            # bowen
            "model": model_without_ddp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        if epoch%40 == 0:
            checkpoint_path = f"{args.trained_model}/checkpoint_epoch_{epoch}.pth"
            # bowen
            # torch.save(checkpoint, checkpoint_path)
            train_utils.save_on_master(checkpoint, checkpoint_path)
        
        ## save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint = {
                # bowen
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            best_checkpoint_path = f"{args.trained_model}/best_checkpoint.pth"
            # bowen
            # torch.save(checkpoint, best_checkpoint_path)
            train_utils.save_on_master(checkpoint, best_checkpoint_path)

if __name__ == "__main__":
    main()
