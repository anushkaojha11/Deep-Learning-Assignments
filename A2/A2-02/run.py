# run.py

import argparse
import os
import logging
import torch
from datetime import datetime


# ── Logger ─────────────────────────────────────────────────────────────────────

def setup_logger(log_dir, mode, model_name):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name  = f'{mode}_{model_name}_{timestamp}.log'
    log_path  = os.path.join(log_dir, log_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()

    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logging.info(f'Log file: {log_path}')
    return log_path


# ── Argument Parser ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='U-Net Image Segmentation')

    # mode
    parser.add_argument('--train',    action='store_true', help='Run training')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')
    parser.add_argument('--visualize',action='store_true', help='Visualize predictions')

    # model
    parser.add_argument('--model', type=str, default='unet_resnet18',
                        choices=[
                            'unet_scratch',
                            'unet_resnet18',
                            'unet_resnet18_no_skip',
                        ],
                        help='Model to use (default: unet_resnet18)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to saved weights (.pt) for evaluate/visualize')

    # data
    parser.add_argument('--dataset',  type=str, default='oxford_pet',
                        help='Dataset name (default: oxford_pet)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to dataset folder (default: ./data)')

    # training
    parser.add_argument('--epochs',     type=int,   default=20)
    parser.add_argument('--batch_size', type=int,   default=16)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--img_size',   type=int,   default=128)

    # output
    parser.add_argument('--output', type=str, default='./output',
                        help='Folder for output plots')
    parser.add_argument('--ckpt',   type=str, default='./checkpoints',
                        help='Folder for model checkpoints')
    parser.add_argument('--logs',   type=str, default='./logs',
                        help='Folder for log files')

    return parser.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def resolve_weights(args):
    """Auto-resolve weights path if not provided."""
    if args.weights is not None:
        return args.weights
    return os.path.join(args.ckpt, f'{args.model}_pet.pt')


# ── Run Training ───────────────────────────────────────────────────────────────

def run_train(args):
    import numpy as np
    import matplotlib.pyplot as plt
    from train import train

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    history, model, save_name = train(
        model_name = args.model,
        n_epochs   = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        img_size   = args.img_size,
        data_dir   = args.data_dir,
        ckpt_dir   = args.ckpt,
        device     = device,
    )

    # log summary
    best_epoch = int(np.argmax(history['val_miou'])) + 1
    best_miou  = max(history['val_miou'])
    logging.info(f'Best mIoU : {best_miou:.4f} at epoch {best_epoch}')
    logging.info(f'Final Loss: {history["train_loss"][-1]:.4f}')

    # plot history
    epochs = list(range(1, len(history['train_loss']) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'Training History — {args.model}', fontsize=13, fontweight='bold')

    axes[0].plot(epochs, history['train_loss'], marker='o', color='steelblue')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].grid(True)

    axes[1].plot(epochs, history['val_miou'], marker='s', color='darkorange')
    axes[1].set_title('Validation mIoU')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim(0, 1)
    axes[1].grid(True)

    plt.tight_layout()
    os.makedirs(args.output, exist_ok=True)
    plot_path = os.path.join(args.output, f'history_{args.model}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logging.info(f'Training plot saved → {plot_path}')
    plt.close()


# ── Run Evaluation ─────────────────────────────────────────────────────────────

def run_evaluate(args):
    from train import evaluate

    weights_path = resolve_weights(args)

    if not os.path.exists(weights_path):
        logging.error(f'Weights not found: {weights_path}')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    evaluate(
        model_name   = args.model,
        weights_path = weights_path,
        img_size     = args.img_size,
        batch_size   = args.batch_size,
        data_dir     = args.data_dir,
        device       = device,
    )


# ── Run Visualization ──────────────────────────────────────────────────────────

def run_visualize(args):
    from train import visualize_predictions

    weights_path = resolve_weights(args)

    if not os.path.exists(weights_path):
        logging.error(f'Weights not found: {weights_path}')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    visualize_predictions(
        model_name   = args.model,
        weights_path = weights_path,
        img_size     = args.img_size,
        data_dir     = args.data_dir,
        output_dir   = args.output,
        device       = device,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # determine mode for logger
    if args.train:
        mode = 'train'
    elif args.evaluate:
        mode = 'evaluate'
    elif args.visualize:
        mode = 'visualize'
    else:
        mode = 'unknown'

    setup_logger(args.logs, mode, args.model)

    if args.train:
        run_train(args)

    elif args.evaluate:
        run_evaluate(args)

    elif args.visualize:
        run_visualize(args)

    else:
        logging.info('Please specify a mode: --train, --evaluate, or --visualize')
        logging.info('Examples:')
        logging.info('  python3 run.py --model unet_resnet18         --epochs 20 --train')
        logging.info('  python3 run.py --model unet_resnet18_no_skip --epochs 20 --train')
        logging.info('  python3 run.py --model unet_resnet18         --evaluate')
        logging.info('  python3 run.py --model unet_resnet18         --visualize')


if __name__ == '__main__':
    main()