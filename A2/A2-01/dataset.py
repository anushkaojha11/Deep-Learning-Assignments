# dataset.py
import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision.datasets import CocoDetection
from typing import Optional, Callable, Tuple, Any
from loss import iou_xywh_numpy

ANCHORS_V3 = [
    [[10, 13],  [16, 30],   [33, 23]],
    [[30, 61],  [62, 45],   [59, 119]],
    [[116, 90], [156, 198], [373, 326]],
]

ANCHORS_V4 = [
    [[12, 16],   [19, 36],   [40, 28]],
    [[36, 75],   [76, 55],   [72, 146]],
    [[142, 110], [192, 243], [459, 401]],
]

STRIDES     = [8, 16, 32]
NUM_ANCHORS = 3
NUM_CLASSES = 80
MAX_BOXES   = 150


def build_cats_dict(path2json):
    with open(path2json) as f:
        categories = json.load(f)['categories']
    cats_dict = {}
    for i, cat in enumerate(categories[:80]):
        cats_dict[str(cat['id'])] = i
    return cats_dict


class CustomCoco(CocoDetection):
    def __init__(
        self,
        root      : str,
        annFile   : str,
        transform : Optional[Callable] = None,
        img_size  : int = 416,
        model_ver : str = 'v3',
    ) -> None:
        super(CocoDetection, self).__init__(root)
        from pycocotools.coco import COCO
        self.coco      = COCO(annFile)
        self.ids       = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.img_size  = img_size
        self.anchors   = ANCHORS_V4 if model_ver == 'v4' else ANCHORS_V3
        self.cats_dict = build_cats_dict(annFile)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        coco   = self.coco
        img_id = self.ids[index]

        ann_ids = coco.getAnnIds(imgIds=img_id)
        target  = coco.loadAnns(ann_ids)
        path    = coco.loadImgs(img_id)[0]['file_name']
        img     = np.array(
            Image.open(os.path.join(self.root, path)).convert('RGB')
        )
        orig_h, orig_w = img.shape[:2]

        # collect valid boxes in coco format [x1, y1, w, h]
        raw_bboxes = []
        raw_cats   = []
        for obj in target:
            x, y, w, h = obj['bbox']
            # skip degenerate boxes
            if w < 1 or h < 1:
                continue
            # clamp to image bounds
            x = np.clip(x, 0, orig_w)
            y = np.clip(y, 0, orig_h)
            w = np.clip(w, 0, orig_w - x)
            h = np.clip(h, 0, orig_h - y)
            if w < 1 or h < 1:
                continue
            raw_bboxes.append([float(x), float(y), float(w), float(h)])
            raw_cats.append(obj['category_id'])

        # apply transform — albumentations handles bbox scaling automatically
        if self.transform is not None and len(raw_bboxes) > 0:
            transformed  = self.transform(
                image        = img,
                bboxes       = raw_bboxes,
                category_ids = raw_cats,
            )
            img      = transformed['image']           # now 608x608
            bboxes   = transformed['bboxes']          # scaled to 608x608
            cat_ids  = transformed['category_ids']

        elif self.transform is not None:
            transformed = self.transform(
                image=img, bboxes=[], category_ids=[]
            )
            img     = transformed['image']
            bboxes  = []
            cat_ids = []

        else:
            # manually scale boxes to img_size
            scale_x = self.img_size / orig_w
            scale_y = self.img_size / orig_h
            bboxes  = [
                [b[0]*scale_x, b[1]*scale_y, b[2]*scale_x, b[3]*scale_y]
                for b in raw_bboxes
            ]
            cat_ids = raw_cats

        # convert to numpy arrays
        if len(bboxes) > 0:
            bboxes_np  = np.array(bboxes,  dtype=np.float32)   # [N, 4] coco xyWH
            cat_ids_np = np.array(cat_ids, dtype=np.int32)
        else:
            bboxes_np  = np.zeros((0, 4), dtype=np.float32)
            cat_ids_np = np.zeros(0,      dtype=np.int32)

        labels, bboxes_out = self._create_label(bboxes_np, cat_ids_np)
        return img, labels, bboxes_out

    def _create_label(self, bboxes, class_inds):
        """
        bboxes    : np.array [N, 4] in COCO format [x1, y1, w, h]
                    coordinates are in img_size pixel space (0 to img_size)
        class_inds: np.array [N] of COCO category ids
        """
        strides           = np.array(STRIDES)
        train_output_size = self.img_size / strides  # e.g. [76, 38, 19] for v4

        label = [
            np.zeros((
                int(train_output_size[i]),
                int(train_output_size[i]),
                NUM_ANCHORS,
                5 + NUM_CLASSES,
            ))
            for i in range(3)
        ]
        bboxes_xywh = [np.zeros((MAX_BOXES, 4)) for _ in range(3)]
        bbox_count  = np.zeros((3,))

        for i in range(len(bboxes)):
            # bboxes[i] is [x1, y1, w, h] in img_size space
            x1, y1, w, h = bboxes[i]

            # skip invalid
            if w < 1 or h < 1:
                continue

            cat_key = str(int(class_inds[i]))
            if cat_key not in self.cats_dict:
                continue
            bbox_class_ind = self.cats_dict[cat_key]

            one_hot = np.zeros(NUM_CLASSES, dtype=np.float32)
            one_hot[bbox_class_ind] = 1.0

            # convert to center xywh — still in img_size pixel space
            cx = x1 + w * 0.5
            cy = y1 + h * 0.5
            cx = np.clip(cx, 0, self.img_size)
            cy = np.clip(cy, 0, self.img_size)
            w  = np.clip(w,  0, self.img_size)
            h  = np.clip(h,  0, self.img_size)

            bbox_xywh = np.array([cx, cy, w, h], dtype=np.float32)

            # scale to each feature map
            bbox_xywh_scaled = bbox_xywh[np.newaxis, :] / strides[:, np.newaxis]

            iou_list       = []
            exist_positive = False

            for scale_i in range(3):
                anchors_xywh         = np.zeros((NUM_ANCHORS, 4))
                anchors_xywh[:, 0:2] = (
                    np.floor(bbox_xywh_scaled[scale_i, 0:2]).astype(np.int32) + 0.5
                )
                anchors_xywh[:, 2:4] = self.anchors[scale_i]

                iou_scale = iou_xywh_numpy(
                    bbox_xywh_scaled[scale_i][np.newaxis, :], anchors_xywh
                )
                iou_list.append(iou_scale)
                iou_mask = iou_scale > 0.3

                if np.any(iou_mask):
                    xind, yind = np.floor(
                        bbox_xywh_scaled[scale_i, 0:2]
                    ).astype(np.int32)

                    grid_size = int(train_output_size[scale_i])
                    xind = np.clip(xind, 0, grid_size - 1)
                    yind = np.clip(yind, 0, grid_size - 1)

                    # store in img_size pixel space
                    label[scale_i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[scale_i][yind, xind, iou_mask, 4:5] = 1.0
                    label[scale_i][yind, xind, iou_mask, 5:]  = one_hot

                    bbox_ind = int(bbox_count[scale_i] % MAX_BOXES)
                    bboxes_xywh[scale_i][bbox_ind, :4] = bbox_xywh
                    bbox_count[scale_i] += 1
                    exist_positive = True

            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou_list).reshape(-1))
                best_detect     = int(best_anchor_ind / NUM_ANCHORS)
                best_anchor     = int(best_anchor_ind % NUM_ANCHORS)

                xind, yind = np.floor(
                    bbox_xywh_scaled[best_detect, 0:2]
                ).astype(np.int32)

                grid_size = int(train_output_size[best_detect])
                xind = np.clip(xind, 0, grid_size - 1)
                yind = np.clip(yind, 0, grid_size - 1)

                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:]  = one_hot

                bbox_ind = int(bbox_count[best_detect] % MAX_BOXES)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        flatten_sizes = [
            int(train_output_size[2]) ** 2 * NUM_ANCHORS,
            int(train_output_size[1]) ** 2 * NUM_ANCHORS,
            int(train_output_size[0]) ** 2 * NUM_ANCHORS,
        ]

        label_s = torch.tensor(label[2], dtype=torch.float32).view(flatten_sizes[0], 5 + NUM_CLASSES)
        label_m = torch.tensor(label[1], dtype=torch.float32).view(flatten_sizes[1], 5 + NUM_CLASSES)
        label_l = torch.tensor(label[0], dtype=torch.float32).view(flatten_sizes[2], 5 + NUM_CLASSES)

        bboxes_s = torch.tensor(bboxes_xywh[2], dtype=torch.float32)
        bboxes_m = torch.tensor(bboxes_xywh[1], dtype=torch.float32)
        bboxes_l = torch.tensor(bboxes_xywh[0], dtype=torch.float32)

        labels     = torch.cat([label_l, label_m, label_s], dim=0)
        bboxes_out = torch.cat([bboxes_l, bboxes_m, bboxes_s], dim=0)

        return labels, bboxes_out