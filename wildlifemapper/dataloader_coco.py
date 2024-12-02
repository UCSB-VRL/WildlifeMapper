

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import numpy as np
import cv2
import random

import torch
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt

import segment_anything.utils.augmentation as T
from segment_anything.utils.augmentation_yolo import random_perspective 


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, image_set, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self.indices = range(self.__len__())
        self.img_size = 1024
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.mosaic = image_set
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]

    def __getitem__(self, idx):
        if self.mosaic == '_train':
            img, target = self.load_mosaic(idx)
            # self.sanity_test(img, target['boxes'])
            # import pdb; pdb.set_trace()
        else:
            # import pdb; pdb.set_trace()
            img, target = super(CocoDetection, self).__getitem__(idx)
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            img, target = self.prepare(img, target)
        if self._transforms is not None:
            #target boxes "ltrb" are normalized as cx/w,cy/h, bw/w, bh/h
            img, target = self._transforms(img, target=target)
            # self.fliplr_test(img, target)
        return {"image": img, "target": target}
    
    def fliplr_test(self, img, target):
        import pdb; pdb.set_trace()
        img = np.transpose(np.array(img), (1,2,0))*1024
        img-=img.min()
        img/=img.max()
        image = img*255
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes = np.array(target['boxes'])
        bboxes = bboxes*1024
        print(image.shape, bboxes.shape)
        # show the example
        _, axs = plt.subplots(1, figsize=(50, 50))
        axs.imshow(np.uint8(image))
        for box in bboxes:
            w, h = box[2], box[3]
            x0, y0 = box[0]-(w/2), box[1]-(h/2)
            axs.add_patch(
                plt.Rectangle((x0, y0), w, h, edgecolor="red", facecolor=(0, 0, 0, 0), lw=2)
            )
        axs.axis("off")
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig("./flip_check.png", bbox_inches="tight", dpi=300)
        plt.close()


    def sanity_test(self, img, boxes):
        import pdb; pdb.set_trace()
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(image.shape, boxes.shape)
        color = (0, 0, 255)
        for box in boxes.numpy():
            l, t = int(box[0]), int(box[1])
            r, b = int(box[2]), int(box[3])
            image = cv2.rectangle(image, (l, t), (r, b), color, 2)
        cv2.imwrite("./mosaic_check2.png", image)
        # _, axs = plt.subplots(1, figsize=(50, 50))
        # axs.imshow(image)
        # for box in boxes.numpy():
        #     w = box[2] - box[0]
        #     h = box[3] - box[1]
        #     axs.add_patch(
        #         plt.Rectangle((box[0], box[1]), w, h, edgecolor="red", facecolor=(0, 0, 0, 0), lw=2)
        #     )
        # axs.axis("off")
        # plt.subplots_adjust(wspace=0.01, hspace=0)
        # plt.savefig("./mosaic_check.png", bbox_inches="tight", dpi=300)
        # plt.close()
    
    def load_mosaic(self, index):
        # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
        boxes4, labels4, center4, image_id4, area4, iscrowd4 = [], [], [], [], [], []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            # img, _, (h, w) = self.load_image(index)
            img, target = super(CocoDetection, self).__getitem__(index)
            image_id = self.ids[index]
            target = {'image_id': image_id, 'annotations': target}
            img, target = self.prepare(img, target)
            h, w = np.asarray(target['size'])
            img = np.array(img)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # targes are in COCO format
            boxes = np.array(target['boxes'])
            labels = np.array(target['labels'])
            center = np.array(target['center'])
            image_id = np.array(target['image_id'])
            area = np.array(target['area'])
            iscrowd = np.array(target['iscrowd'])
            orig_size = np.array(target['orig_size'])
            size = np.array(target['size'])

            if boxes.size:
                boxes = self.ltrb2cxcywh(boxes)
                boxes = self.xywhn2xyxy(boxes, padw, padh)  # normalized xywh to pixel xyxy format
            boxes4.append(boxes)
            labels4.append(labels)
            center4.append(center)
            image_id4.append(image_id)
            area4.append(area)
            iscrowd4.append(iscrowd)

        # Concat/clip labels
        boxes4 = np.concatenate(boxes4, 0)
        labels4 = np.concatenate(labels4, 0)
        # center4 = np.concatenate(center4, 0)
        image_id4 = np.concatenate(image_id4, 0)
        # area4 = np.concatenate(area4, 0)
        iscrowd4 = np.concatenate(iscrowd4, 0)
        
        np.clip(boxes4, 0, 2 * s, out=boxes4)  # clip when using random_perspective()
        #make yolo-format to use random perspective augmentation
        boxes4 = np.concatenate((np.expand_dims(labels4, axis=1), boxes4), axis=1)

        # import pdb; pdb.set_trace()
        # Augment
        # both img and boxes are absolute values at this point of time
        img4, boxes4 = random_perspective(img4,
                                           boxes4,
                                           degrees=0.0,
                                           translate=0.1,
                                           scale=0.9,
                                           shear=0.0,
                                           perspective=0.0,
                                           border=self.mosaic_border)  # border to remove

        #make coco format targets
        new_target = {}
        new_target['boxes'] = torch.tensor(boxes4[:, 1:], dtype=torch.float32)
        new_target['labels'] = torch.tensor(boxes4[:, 0], dtype=torch.int64)
        new_target['image_id'] = torch.tensor(image_id4, dtype=torch.int64)
        new_target['orig_size'] = torch.tensor(orig_size, dtype=torch.int64)
        new_target['size'] = torch.tensor(list(img4.shape[:2]), dtype=torch.int64)

        #calculate center points of new boxes
        new_boxes = torch.as_tensor(boxes4[:, 1:], dtype=torch.float32)
        centre_points = torch.cat((new_boxes[:, ::2].mean(1, True), new_boxes[:, 1::2].mean(1, True)), 1)
        new_target["center"] = centre_points

        return img4, new_target
    
    def ltrb2cxcywh(self, boxes):
        # convert ltrb to cxcywh
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        boxes[:, 0] = boxes[:, 0] + (boxes[:, 2]//2)
        boxes[:, 1] = boxes[:, 1] + (boxes[:, 3]//2)
        return boxes
    
    def xywhn2xyxy(self, x, padw=0, padh=0):
        # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[..., 0] = (x[..., 0] - x[..., 2] / 2) + padw  # top left x
        y[..., 1] = (x[..., 1] - x[..., 3] / 2) + padh  # top left y
        y[..., 2] = (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
        y[..., 3] = (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
        return y

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id], dtype=torch.int32)

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        #center-point of the box for Bobby, yo!!
        centre_points = torch.cat((boxes[:, ::2].mean(1, True), boxes[:, 1::2].mean(1, True)), 1)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["center"] = centre_points
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

#TODO : Add data augmentation later, transforms.py file from DETR
def make_coco_transforms(image_set):

    if image_set == 'train':
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.FlipLR(fliplr=0.5)
        ])
        return normalize
    
    if image_set == 'val':
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return normalize
    
    raise ValueError(f'unknown {image_set}')
    

    # #scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    # if image_set == 'train':
    #     return T.Compose([
    #         T.RandomHorizontalFlip(),
    #         T.RandomSelect(
    #             T.RandomResize(scales, max_size=1333),
    #             T.Compose([
    #                 T.RandomResize([400, 500, 600]),
    #                 T.RandomSizeCrop(384, 600),
    #                 T.RandomResize(scales, max_size=1333),
    #             ])
    #         ),
    #         normalize,
    #     ])

    # if image_set == 'val':
    #     return T.Compose([
    #         T.RandomResize([800], max_size=1333),
    #         normalize,
    #     ])

    # raise ValueError(f'unknown {image_set}')


def build_dataset(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        # "train": (root / "train2017", root / "annotations" / "instances_train2017.json"),
        "train": (root / "val2017", root / "annotations" / "instances_val2017.json"),
        "val": (root / "val2017", root / "annotations" / "instances_val2017.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    #dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    dataset = CocoDetection(img_folder, ann_file, image_set, transforms=make_coco_transforms(image_set), return_masks=False)
    return dataset