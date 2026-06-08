import os
import time
import warnings
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import albumentations as A

from model import MyDarknet
from dataset import CustomCoco
from loss import YOLOLoss


def get_transforms(img_size):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['category_ids'],
        min_visibility=0.1,
    ))

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
    ], bbox_params=A.BboxParams(
        format='coco',
        label_fields=['category_ids'],
        min_visibility=0.1,
    ))

    return train_transform, val_transform


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloaders(path2data, path2json, img_size, model_ver,
                    batch_size, total_samples=5000, val_samples=1000):
    """
    Load COCO dataset and split into train/val.
    """
    train_transform, val_transform = get_transforms(img_size)

    full_dataset = CustomCoco(
        root      = path2data,
        annFile   = path2json,
        transform = train_transform,
        img_size  = img_size,
        model_ver = model_ver,
    )

    train_indices = list(range(0, total_samples - val_samples))
    val_indices   = list(range(total_samples - val_samples, total_samples))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset   = Subset(full_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,
        collate_fn  = collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = collate_fn,
    )

    print(f'Train : {len(train_dataset)} samples ({len(train_loader)} batches)')
    print(f'Val   : {len(val_dataset)} samples ({len(val_loader)} batches)')

    return train_loader, val_loader


def train(
    cfg_path,
    weights_path,
    path2data,
    path2json,
    model_ver   = 'v3',
    bbox_loss   = 'iou',
    n_epoch     = 10,
    batch_size  = 8,
    lr          = 1e-5,
    img_size    = 416,
    ckpt_dir    = './checkpoints',
    device      = None,
):
    """
    Main training function.

    Args:
        cfg_path    : path to .cfg file
        weights_path: path to pretrained .weights file
        path2data   : path to COCO images folder
        path2json   : path to COCO annotation json
        model_ver   : 'v3' or 'v4'
        bbox_loss   : 'iou' or 'ciou'
        n_epoch     : number of epochs
        batch_size  : batch size
        lr          : learning rate
        img_size    : input image size
        ckpt_dir    : folder to save checkpoints
        device      : torch device (auto-detected if None)
    """
    #Setup 
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device : {device}')
    print(f'Model        : YOLO{model_ver}')
    print(f'BBox loss    : {bbox_loss}')
    print(f'Epochs       : {n_epoch}')
    print(f'Batch size   : {batch_size}')
    print(f'Image size   : {img_size}')

    os.makedirs(ckpt_dir, exist_ok=True)

    #Model
    print('\nLoading model...')
    model = MyDarknet(cfg_path)
    model.load_weights(weights_path)
    model.to(device)
    print('Model loaded successfully.')

    #Data
    print('\nLoading dataset...')
    train_loader, val_loader = get_dataloaders(
        path2data, path2json, img_size, model_ver, batch_size
    )

    #Loss and Optimizer
    criterion = YOLOLoss(
        img_size   = img_size,
        num_classes = 80,
        bbox_loss  = bbox_loss,
    ).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr           = lr,
        momentum     = 0.9,
        weight_decay = 5e-4,
    )

    #Training Loop
    history = {
        'loss' : [],
        'box'  : [],
        'conf' : [],
        'cls'  : [],
        'map50': [],
    }

    print('\nStarting training...\n')

    for epoch in range(n_epoch):
        model.train()
        t0 = time.time()

        running_loss = running_box = running_conf = running_cls = 0.0
        n_batches  = 0
        n_skipped  = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epoch}')

        for inputs, labels, bboxes in pbar:
            # stack inputs: list of numpy arrays -> tensor [B, C, H, W]
            inputs = torch.from_numpy(
                np.array(inputs)
            ).squeeze(1).permute(0, 3, 1, 2).float().to(device) / 255.0

            labels = torch.stack(labels).to(device).float()

            optimizer.zero_grad()

            use_cuda = (device.type == 'cuda')
            outputs  = model(inputs, use_cuda)

            loss, loss_box, loss_conf, loss_cls = criterion(outputs, labels)

            # skip bad batches
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                n_skipped += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            torch.cuda.empty_cache()

            B = inputs.size(0)
            running_loss += loss.item()      * B
            running_box  += loss_box.item()  * B
            running_conf += loss_conf.item() * B
            running_cls  += loss_cls.item()  * B
            n_batches    += 1

            avg_loss = running_loss / (n_batches * B)
            pbar.set_postfix(
                loss = f'{avg_loss:.3f}',
                box  = f'{loss_box.item():.3f}',
                conf = f'{loss_conf.item():.3f}',
            )

        if n_skipped:
            print(f'  [warn] skipped {n_skipped} batches due to nan/inf loss')

        denom      = max(n_batches, 1)
        epoch_loss = running_loss / (denom * batch_size)
        epoch_box  = running_box  / (denom * batch_size)
        epoch_conf = running_conf / (denom * batch_size)
        epoch_cls  = running_cls  / (denom * batch_size)
        elapsed    = time.time() - t0

        #Validation mAP
        map50 = compute_map(model, val_loader, device, img_size)

        history['loss' ].append(epoch_loss)
        history['box'  ].append(epoch_box)
        history['conf' ].append(epoch_conf)
        history['cls'  ].append(epoch_cls)
        history['map50'].append(map50 if map50 is not None else 0.0)

        print(
            f'Epoch {epoch+1:02d}/{n_epoch} | '
            f'Loss: {epoch_loss:.4f} | '
            f'Box: {epoch_box:.4f} | '
            f'Conf: {epoch_conf:.4f} | '
            f'Cls: {epoch_cls:.4f} | '
            f'mAP@50: {map50:.4f} | '
            f'Time: {elapsed:.1f}s'
        )

        # save checkpoint every epoch
        ckpt_path = os.path.join(ckpt_dir, f'yolo{model_ver}_{bbox_loss}_epoch{epoch+1}.pt')
        torch.save(model.state_dict(), ckpt_path)
        print(f'  └─ saved → {ckpt_path}')

    print('\nTraining complete.')
    return history, model


def compute_map(model, dataloader, device, img_size,
                conf_thresh=0.05, nms_thresh=0.4):
    """Compute mAP@0.5 on a dataloader."""
    try:
        from torchmetrics.detection import MeanAveragePrecision
        from torchvision.ops import nms as tv_nms
    except ImportError:
        print('torchmetrics not installed — skipping mAP')
        return 0.0

    metric = MeanAveragePrecision(
        iou_type             = 'bbox',
        iou_thresholds       = [0.5],
        max_detection_thresholds = [1, 10, 100],
    )

    model.eval()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter('ignore')

        for inputs, labels, bboxes in tqdm(dataloader, desc='  mAP eval', leave=False):
            inputs = torch.from_numpy(
                np.array(inputs)
            ).squeeze(1).permute(0, 3, 1, 2).float().to(device) / 255.0

            use_cuda = (device.type == 'cuda')
            outputs  = model(inputs, use_cuda)

            # predicted boxes
            pred_conf = outputs[..., 4]
            pred_cls  = outputs[..., 5:]
            pred_xy   = outputs[..., 0:2]
            pred_wh   = outputs[..., 2:4]

            pred_x1 = (pred_xy[..., 0] - pred_wh[..., 0] / 2).clamp(0, img_size)
            pred_y1 = (pred_xy[..., 1] - pred_wh[..., 1] / 2).clamp(0, img_size)
            pred_x2 = (pred_xy[..., 0] + pred_wh[..., 0] / 2).clamp(0, img_size)
            pred_y2 = (pred_xy[..., 1] + pred_wh[..., 1] / 2).clamp(0, img_size)
            pred_boxes = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)

            cls_scores, cls_ids = pred_cls.max(dim=-1)
            scores = pred_conf * cls_scores
            scores = scores.clamp(0, 1)

            preds        = []
            targets_list = []

            for i in range(inputs.size(0)):
                mask = scores[i] > conf_thresh
                if mask.sum() == 0:
                    preds.append(dict(
                        boxes  = torch.zeros((0, 4)),
                        scores = torch.zeros(0),
                        labels = torch.zeros(0, dtype=torch.long),
                    ))
                else:
                    b    = pred_boxes[i][mask].cpu()
                    s    = scores[i][mask].cpu()
                    l    = cls_ids[i][mask].cpu().long()
                    keep = tv_nms(b, s, nms_thresh)[:100]
                    preds.append(dict(boxes=b[keep], scores=s[keep], labels=l[keep]))

                # ground truth
                label_i  = labels[i] if isinstance(labels, (list, tuple)) \
                           else torch.stack(labels)[i]
                obj_mask = label_i[..., 4] > 0
                gt_xywh  = label_i[obj_mask][..., :4]
                gt_cls   = label_i[obj_mask][..., 5:].argmax(dim=-1).long()

                if gt_xywh.shape[0] == 0:
                    targets_list.append(dict(
                        boxes  = torch.zeros((0, 4)),
                        labels = torch.zeros(0, dtype=torch.long),
                    ))
                    continue

                gt_x1 = (gt_xywh[..., 0] - gt_xywh[..., 2] / 2).clamp(0)
                gt_y1 = (gt_xywh[..., 1] - gt_xywh[..., 3] / 2).clamp(0)
                gt_x2 =  gt_xywh[..., 0] + gt_xywh[..., 2] / 2
                gt_y2 =  gt_xywh[..., 1] + gt_xywh[..., 3] / 2
                targets_list.append(dict(
                    boxes  = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1).cpu(),
                    labels = gt_cls.cpu(),
                ))

            metric.update(preds, targets_list)

    try:
        result = metric.compute()
        return result['map_50'].item()
    except Exception:
        return 0.0