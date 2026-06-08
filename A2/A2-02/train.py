# train.py

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from model import get_model
from dataset import get_dataloaders, CLASS_COLORS, denormalize, NUM_CLASSES


# ── Metric ─────────────────────────────────────────────────────────────────────

def compute_miou(pred, target, n_classes=NUM_CLASSES):
    """
    Compute mean Intersection over Union (mIoU).
    pred   : raw logits [B, C, H, W]
    target : ground truth [B, H, W] with values in {0..n_classes-1}
    """
    pred = pred.argmax(dim=1)   # [B, H, W]
    ious = []
    for cls in range(n_classes):
        inter = ((pred == cls) & (target == cls)).sum().float()
        union = ((pred == cls) | (target == cls)).sum().float()
        if union > 0:
            ious.append((inter / union).item())
    return np.mean(ious) if ious else 0.0


# ── Training Loop ──────────────────────────────────────────────────────────────

def train(
    model_name  = 'unet_resnet18',
    n_epochs    = 20,
    batch_size  = 16,
    lr          = 1e-3,
    img_size    = 128,
    data_dir    = './data',
    ckpt_dir    = './checkpoints',
    device      = None,
):
    """
    Train a U-Net model on Oxford Pet dataset.

    Args:
        model_name : 'unet_resnet18' or 'unet_resnet18_no_skip'
        n_epochs   : number of training epochs
        batch_size : batch size
        lr         : learning rate
        img_size   : input image size
        data_dir   : path to dataset
        ckpt_dir   : folder to save checkpoints
        device     : torch device (auto if None)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(ckpt_dir, exist_ok=True)

    logging.info(f'Model      : {model_name}')
    logging.info(f'Epochs     : {n_epochs}')
    logging.info(f'Batch size : {batch_size}')
    logging.info(f'LR         : {lr}')
    logging.info(f'Image size : {img_size}')
    logging.info(f'Device     : {device}')

    # ── Data ───────────────────────────────────────────────────────────────
    logging.info('Loading dataset...')
    train_loader, test_loader = get_dataloaders(
        img_size     = img_size,
        batch_size   = batch_size,
        data_dir     = data_dir,
        num_workers  = 2,
    )

    # ── Model ──────────────────────────────────────────────────────────────
    logging.info('Building model...')
    model = get_model(model_name, n_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    logging.info(f'Params: {sum(p.numel() for p in model.parameters()):,}')

    # ── Loss + Optimizer + Scheduler ───────────────────────────────────────
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    # ── History ────────────────────────────────────────────────────────────
    history = {
        'train_loss' : [],
        'val_miou'   : [],
    }

    best_miou   = 0.0
    save_name   = os.path.join(ckpt_dir, f'{model_name}_pet.pt')

    logging.info('Starting training...')

    for epoch in range(n_epochs):
        t0 = time.time()

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        for imgs, masks in pbar:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        epoch_iou = []

        with torch.no_grad():
            for imgs, masks in test_loader:
                imgs  = imgs.to(device)
                masks = masks.to(device)
                outputs = model(imgs)
                epoch_iou.append(compute_miou(outputs, masks))

        scheduler.step()

        mean_loss = np.mean(epoch_loss)
        mean_iou  = np.mean(epoch_iou)
        elapsed   = time.time() - t0

        history['train_loss'].append(mean_loss)
        history['val_miou'].append(mean_iou)

        logging.info(
            f'Epoch {epoch+1:02d}/{n_epochs} | '
            f'Loss: {mean_loss:.4f} | '
            f'mIoU: {mean_iou:.4f} | '
            f'Time: {elapsed:.1f}s'
        )

        # save best model
        if mean_iou > best_miou:
            best_miou = mean_iou
            torch.save(model.state_dict(), save_name)
            logging.info(f'  └─ New best mIoU: {best_miou:.4f} → saved to {save_name}')

    logging.info(f'Training complete. Best mIoU: {best_miou:.4f}')
    return history, model, save_name


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(
    model_name   = 'unet_resnet18',
    weights_path = None,
    img_size     = 128,
    batch_size   = 16,
    data_dir     = './data',
    device       = None,
):
    """
    Evaluate a trained U-Net on the Oxford Pet test set.
    Returns mIoU score.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info(f'Evaluating : {model_name}')
    logging.info(f'Weights    : {weights_path}')

    # load model
    model = get_model(model_name, n_classes=NUM_CLASSES, pretrained=False)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # load data
    _, test_loader = get_dataloaders(
        img_size    = img_size,
        batch_size  = batch_size,
        data_dir    = data_dir,
        num_workers = 2,
    )

    ious = []
    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc='Evaluating'):
            imgs  = imgs.to(device)
            masks = masks.to(device)
            outputs = model(imgs)
            ious.append(compute_miou(outputs, masks))

    miou = np.mean(ious)
    logging.info(f'mIoU@test : {miou:.4f}')

    print('\n' + '='*40)
    print(f'  {model_name}')
    print(f'  mIoU@test : {miou:.4f}')
    print('='*40)

    return miou


# ── Visualization ──────────────────────────────────────────────────────────────

def visualize_predictions(
    model_name   = 'unet_resnet18',
    weights_path = None,
    img_size     = 128,
    data_dir     = './data',
    output_dir   = './output',
    n_samples    = 5,
    device       = None,
):
    """
    Save a visualization of model predictions vs ground truth.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_dir, exist_ok=True)

    # load model
    model = get_model(model_name, n_classes=NUM_CLASSES, pretrained=False)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # load dataset
    from dataset import PetSegDataset
    test_data = PetSegDataset(split='test', img_size=img_size, data_dir=data_dir)

    CLASS_NAMES = ['Pet', 'Background', 'Border']

    fig, axes = plt.subplots(n_samples, 4, figsize=(14, n_samples * 3))
    for ax, t in zip(axes[0], ['Input', 'Ground Truth', 'Prediction', 'Overlay']):
        ax.set_title(t, fontsize=11, fontweight='bold')

    for row in range(n_samples):
        img, mask = test_data[row * 50]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))
            pred = pred.argmax(1).squeeze().cpu().numpy()

        img_d = denormalize(img)

        axes[row][0].imshow(img_d)
        axes[row][1].imshow(CLASS_COLORS[mask.numpy()])
        axes[row][2].imshow(CLASS_COLORS[pred])
        axes[row][3].imshow(img_d)
        axes[row][3].imshow(CLASS_COLORS[pred], alpha=0.5)

        for ax in axes[row]:
            ax.axis('off')

    patches = [
        mpatches.Patch(color=CLASS_COLORS[i]/255, label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=10)
    plt.suptitle(f'U-Net Results — {model_name}', fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'predictions_{model_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logging.info(f'Predictions saved → {save_path}')
    plt.close()