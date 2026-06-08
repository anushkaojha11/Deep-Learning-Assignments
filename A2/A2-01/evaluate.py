import os
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

import albumentations as A

from model import MyDarknet
from dataset import CustomCoco
from train import collate_fn, compute_map


def evaluate(
    cfg_path,
    weights_path,
    path2data,
    path2json,
    model_ver  = 'v3',
    img_size   = 416,
    batch_size = 8,
    device     = None,
):
    """
    Evaluate a YOLO model on COCO validation set and print mAP@0.5.

    Args:
        cfg_path     : path to .cfg file
        weights_path : path to .weights or .pt checkpoint
        path2data    : path to COCO images folder
        path2json    : path to COCO annotation json
        model_ver    : 'v3' or 'v4'
        img_size     : input image size
        batch_size   : batch size
        device       : torch device (auto-detected if None)
    """
    #Setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device : {device}')
    print(f'Model        : YOLO{model_ver}')
    print(f'Image size   : {img_size}')

    #Model
    print('\nLoading model...')
    model = MyDarknet(cfg_path)

    # support both darknet .weights and pytorch .pt checkpoints
    if weights_path.endswith('.weights'):
        model.load_weights(weights_path)
    else:
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    print('Model loaded successfully.')

    #Data
    print('\nLoading dataset...')
    val_transform = A.Compose([
        A.Resize(img_size, img_size),
    ], bbox_params=A.BboxParams(
        format       = 'coco',
        label_fields = ['category_ids'],
        min_visibility = 0.1,
    ))

    full_dataset = CustomCoco(
        root      = path2data,
        annFile   = path2json,
        transform = val_transform,
        img_size  = img_size,
        model_ver = model_ver,
    )

    # use last 1000 samples as validation set
    total_samples = 5000
    val_samples   = 1000
    val_indices   = list(range(total_samples - val_samples, total_samples))
    val_dataset   = Subset(full_dataset, val_indices)

    val_loader = DataLoader(
        val_dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = collate_fn,
    )
    print(f'Val : {len(val_dataset)} samples ({len(val_loader)} batches)')

    #Evaluate
    print('\nComputing mAP@0.5...')
    map50 = compute_map(model, val_loader, device, img_size)

    print('\n' + '='*40)
    print(f'  YOLO{model_ver} mAP@0.5 : {map50:.4f}')
    print('='*40)

    return map50