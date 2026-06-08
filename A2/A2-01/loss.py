import torch
import torch.nn as nn
import numpy as np
import math


def iou_xywh_numpy(boxes1, boxes2):
    """
    Calculate IoU between two sets of boxes in xywh format (numpy).
    Used during label assignment in the dataset class.
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # convert xywh -> xyxy
    boxes1_xyxy = np.concatenate([
        boxes1[..., :2] - boxes1[..., 2:] * 0.5,
        boxes1[..., :2] + boxes1[..., 2:] * 0.5
    ], axis=-1)
    boxes2_xyxy = np.concatenate([
        boxes2[..., :2] - boxes2[..., 2:] * 0.5,
        boxes2[..., :2] + boxes2[..., 2:] * 0.5
    ], axis=-1)

    left_up    = np.maximum(boxes1_xyxy[..., :2], boxes2_xyxy[..., :2])
    right_down = np.minimum(boxes1_xyxy[..., 2:], boxes2_xyxy[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou


def ciou_xywh_torch(boxes1, boxes2):
    """
    Calculate CIoU between two sets of boxes in xywh format (torch).
    CIoU = IoU - (center distance penalty) - (aspect ratio penalty)
    Used as the bounding box loss during training.
    """
    # xywh -> xyxy
    boxes1_xyxy = torch.cat([
        boxes1[..., :2] - boxes1[..., 2:] * 0.5,
        boxes1[..., :2] + boxes1[..., 2:] * 0.5
    ], dim=-1)
    boxes2_xyxy = torch.cat([
        boxes2[..., :2] - boxes2[..., 2:] * 0.5,
        boxes2[..., :2] + boxes2[..., 2:] * 0.5
    ], dim=-1)

    # ensure x1y1 < x2y2
    boxes1_xyxy = torch.cat([
        torch.min(boxes1_xyxy[..., :2], boxes1_xyxy[..., 2:]),
        torch.max(boxes1_xyxy[..., :2], boxes1_xyxy[..., 2:])
    ], dim=-1)
    boxes2_xyxy = torch.cat([
        torch.min(boxes2_xyxy[..., :2], boxes2_xyxy[..., 2:]),
        torch.max(boxes2_xyxy[..., :2], boxes2_xyxy[..., 2:])
    ], dim=-1)

    # areas
    boxes1_area = (boxes1_xyxy[..., 2] - boxes1_xyxy[..., 0]) * \
                  (boxes1_xyxy[..., 3] - boxes1_xyxy[..., 1])
    boxes2_area = (boxes2_xyxy[..., 2] - boxes2_xyxy[..., 0]) * \
                  (boxes2_xyxy[..., 3] - boxes2_xyxy[..., 1])

    # intersection
    inter_left_up    = torch.max(boxes1_xyxy[..., :2], boxes2_xyxy[..., :2])
    inter_right_down = torch.min(boxes1_xyxy[..., 2:], boxes2_xyxy[..., 2:])
    inter_section    = torch.clamp(inter_right_down - inter_left_up, min=0)
    inter_area       = inter_section[..., 0] * inter_section[..., 1]

    # iou
    union_area = boxes1_area + boxes2_area - inter_area
    iou        = inter_area / (union_area + 1e-6)

    # outer box diagonal
    outer_left_up    = torch.min(boxes1_xyxy[..., :2], boxes2_xyxy[..., :2])
    outer_right_down = torch.max(boxes1_xyxy[..., 2:], boxes2_xyxy[..., 2:])
    outer            = torch.clamp(outer_right_down - outer_left_up, min=0)
    outer_diagonal   = outer[..., 0] ** 2 + outer[..., 1] ** 2 + 1e-6

    # center distance
    boxes1_center = (boxes1_xyxy[..., :2] + boxes1_xyxy[..., 2:]) * 0.5
    boxes2_center = (boxes2_xyxy[..., :2] + boxes2_xyxy[..., 2:]) * 0.5
    center_dist   = (boxes1_center[..., 0] - boxes2_center[..., 0]) ** 2 + \
                    (boxes1_center[..., 1] - boxes2_center[..., 1]) ** 2

    # aspect ratio penalty
    boxes1_size = torch.clamp(boxes1_xyxy[..., 2:] - boxes1_xyxy[..., :2], min=1e-6)
    boxes2_size = torch.clamp(boxes2_xyxy[..., 2:] - boxes2_xyxy[..., :2], min=1e-6)
    v     = (4 / (math.pi ** 2)) * torch.pow(
                torch.atan(boxes1_size[..., 0] / boxes1_size[..., 1]) -
                torch.atan(boxes2_size[..., 0] / boxes2_size[..., 1]), 2)
    alpha = v / (1 - iou + v + 1e-6)

    ciou = iou - (center_dist / outer_diagonal) - (alpha * v)
    return ciou


class YOLOLoss(nn.Module):
    """
    YOLOv3/v4 loss function.
    Supports two bbox loss modes:
      - 'iou' : MSE on normalized tx,ty,tw,th  (YOLOv3 paper)
      - 'ciou': CIoU loss                       (YOLOv4 improvement)
    """
    def __init__(self, img_size=416, num_classes=80,
                 lambda_coord=1.0, lambda_noobj=0.5,
                 bbox_loss='iou'):
        super(YOLOLoss, self).__init__()
        assert bbox_loss in ('iou', 'ciou'), "bbox_loss must be 'iou' or 'ciou'"
        self.img_size     = img_size
        self.num_classes  = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.bbox_loss    = bbox_loss
        self.bce          = nn.BCEWithLogitsLoss(reduction='none')
        self.mse          = nn.MSELoss(reduction='none')

    def forward(self, outputs, labels):
        """
        Args:
            outputs : raw model output  [B, N, 5+C]
            labels  : ground truth      [B, N, 5+C]
                      format: [cx, cy, w, h, obj, cls...]
        Returns:
            loss       : total scalar loss
            loss_box   : bbox component
            loss_conf  : confidence component
            loss_cls   : class component
        """
        outputs = outputs.float()
        labels  = labels.float()

        B = outputs.size(0)

        pred_xywh = outputs[..., 0:4]
        raw_conf  = outputs[..., 4:5]
        raw_cls   = outputs[..., 5:]
    
        label_xywh       = labels[..., :4]
        label_obj_mask   = labels[..., 4:5].clamp(0, 1)
        label_noobj_mask = 1.0 - label_obj_mask
        label_cls        = labels[..., 5:].clamp(0, 1)

        #Bounding box loss
        if self.bbox_loss == 'ciou':
            ciou = ciou_xywh_torch(pred_xywh, label_xywh)
            loss_box = torch.sum(
                label_obj_mask.squeeze(-1) * (1.0 - ciou)
            ) / B
        
        else:  # 'iou' — MSE on normalized coords
            # normalize by image size to keep values in [0,1] range
            pred_norm  = pred_xywh  / self.img_size
            label_norm = label_xywh / self.img_size
            # only compute loss on foreground (obj) anchors
            loss_box = self.lambda_coord * torch.sum(
                label_obj_mask * self.mse(pred_norm, label_norm)
            ) / (label_obj_mask.sum() + 1e-6)

        #Confidence loss
        loss_conf = (
            torch.sum(label_obj_mask   * self.bce(raw_conf, label_obj_mask)) +
            self.lambda_noobj *
            torch.sum(label_noobj_mask * self.bce(raw_conf, label_obj_mask))
        ) / B

        #Class loss
        loss_cls = torch.sum(
            label_obj_mask * self.bce(raw_cls, label_cls)
        ) / B

        loss = loss_box + loss_conf + loss_cls

        return loss, loss_box, loss_conf, loss_cls