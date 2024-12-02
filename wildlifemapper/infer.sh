#/bin/bash
CUDA_VISIBLE_DEVICES=5 python visualize_prediction.py --coco_path /mnt/mara/coco_1024_fixed --pretrain_model_path ./exp/box_model/best_checkpoint.pth  --num_workers 2
