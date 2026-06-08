# run.py

import argparse
import os
import logging
import torch
from datetime import datetime


#Logger Setup

def setup_logger(log_dir, mode, model_name, bbox_loss=None):
    """
    Sets up a logger that writes to both terminal and a log file.
    Log file name includes mode, model, and timestamp.
    """
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if bbox_loss:
        log_name = f'{mode}_{model_name}_{bbox_loss}_{timestamp}.log'
    else:
        log_name = f'{mode}_{model_name}_{timestamp}.log'

    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # remove existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # file handler — writes to log file
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)

    # terminal handler — prints to screen
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logging.info(f'Log file: {log_path}')
    return log_path


#Argument Parser

def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3/v4 Object Detection')

    # mode
    parser.add_argument('--infer',    action='store_true', help='Run inference')
    parser.add_argument('--train',    action='store_true', help='Run training')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation')

    # model
    parser.add_argument('--model',   type=str, default='yolov3',
                        choices=['yolov3', 'yolov4'],
                        help='Model version (default: yolov3)')
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to weights file (.weights or .pt)')

    # data
    parser.add_argument('--image',   type=str, default=None,
                        help='Path to image for inference')
    parser.add_argument('--data',    type=str,
                        default=os.path.expanduser(
                            '~/fiftyone/coco-2017/validation/data'),
                        help='Path to COCO images folder')
    parser.add_argument('--json',    type=str,
                        default=os.path.expanduser(
                            '~/fiftyone/coco-2017/raw/instances_val2017.json'),
                        help='Path to COCO annotation json')

    # training
    parser.add_argument('--epochs',     type=int,   default=10)
    parser.add_argument('--batch_size', type=int,   default=8)
    parser.add_argument('--lr',         type=float, default=1e-5)
    parser.add_argument('--bbox_loss',  type=str,   default='iou',
                        choices=['iou', 'ciou'],
                        help='Bounding box loss type (default: iou)')

    # output
    parser.add_argument('--output', type=str, default='./output',
                        help='Folder for inference result images')
    parser.add_argument('--ckpt',   type=str, default='./checkpoints',
                        help='Folder to save training checkpoints')
    parser.add_argument('--logs',   type=str, default='./logs',
                        help='Folder to save log files')
    parser.add_argument('--conf',   type=float, default=0.5,
                        help='Confidence threshold for inference')
    parser.add_argument('--nms',    type=float, default=0.4,
                        help='NMS threshold for inference')

    return parser.parse_args()


#Path Helper

def get_paths(args):
    """Resolve cfg and weights paths from model name."""
    base     = os.path.dirname(os.path.abspath(__file__))
    ver      = 'v3' if args.model == 'yolov3' else 'v4'
    cfg_path = os.path.join(base, 'cfg', f'yolo{ver}.cfg')

    if args.weights is None:
        weights_path = os.path.join(base, 'weights', f'{args.model}.weights')
    else:
        weights_path = args.weights

    img_size = 416 if ver == 'v3' else 608

    return cfg_path, weights_path, ver, img_size


#Inference

def run_infer(args, cfg_path, weights_path, ver, img_size):
    """Run inference on a single image."""
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from model import MyDarknet
    from util import write_results, load_classes

    if args.image is None:
        logging.error('--image is required for inference.')
        return

    if not os.path.exists(args.image):
        logging.error(f'Image not found: {args.image}')
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Device    : {device}')
    logging.info(f'Model     : {args.model}')
    logging.info(f'Image     : {args.image}')
    logging.info(f'Image size: {img_size}')

    # load model
    logging.info('Loading model...')
    model = MyDarknet(cfg_path)
    model.load_weights(weights_path)
    model.to(device)
    model.eval()
    logging.info('Model loaded.')

    # preprocess image
    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img_rgb.shape[:2]

    img_resized = cv2.resize(img_rgb, (img_size, img_size))
    img_tensor  = torch.from_numpy(img_resized).permute(2, 0, 1).float()
    img_tensor  = img_tensor.unsqueeze(0) / 255.0
    img_tensor  = img_tensor.to(device)

    # forward pass
    logging.info('Running inference...')
    use_cuda = (device.type == 'cuda')
    with torch.no_grad():
        pred = model(img_tensor, use_cuda)

    detections = write_results(pred, args.conf, 80, nms_conf=args.nms)

    if isinstance(detections, int):
        logging.info('No detections found.')
        return

    # load class names
    names_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'data', 'coco.names'
    )
    classes = load_classes(names_path)

    # scale boxes back to original size
    scale_x = orig_w / img_size
    scale_y = orig_h / img_size
    detections = detections.cpu()

    logging.info(f'Detections: {len(detections)}')
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(img_rgb)
    colors = plt.cm.Set1(range(len(detections)))

    for idx, det in enumerate(detections):
        x1   = int(det[1].item() * scale_x)
        y1   = int(det[2].item() * scale_y)
        x2   = int(det[3].item() * scale_x)
        y2   = int(det[4].item() * scale_y)
        cls  = int(det[-1].item())
        conf = det[5].item()
        label = f'{classes[cls]} {conf:.2f}'

        logging.info(
            f'  {classes[cls]:20s} conf={conf:.3f}  '
            f'box=[{x1},{y1},{x2},{y2}]'
        )

        color = colors[idx % len(colors)]
        rect  = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, label, color=color,
                fontsize=10, fontweight='bold',
                bbox=dict(facecolor='black', alpha=0.4, pad=2))

    ax.axis('off')
    plt.title(
        f'{args.model} — {len(detections)} object(s) detected',
        fontsize=13
    )
    plt.tight_layout()

    os.makedirs(args.output, exist_ok=True)
    img_name  = os.path.splitext(os.path.basename(args.image))[0]
    save_path = os.path.join(
        args.output, f'det_{args.model}_{img_name}.png'
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logging.info(f'Result saved → {save_path}')


#Training

def run_train(args, cfg_path, weights_path, ver, img_size):
    """Run training."""
    import numpy as np
    import matplotlib.pyplot as plt
    from train import train

    logging.info(f'Starting training — YOLO{ver} | loss={args.bbox_loss}')
    logging.info(f'Epochs     : {args.epochs}')
    logging.info(f'Batch size : {args.batch_size}')
    logging.info(f'LR         : {args.lr}')
    logging.info(f'Image size : {img_size}')
    logging.info(f'Data       : {args.data}')
    logging.info(f'JSON       : {args.json}')

    history, model = train(
        cfg_path     = cfg_path,
        weights_path = weights_path,
        path2data    = args.data,
        path2json    = args.json,
        model_ver    = ver,
        bbox_loss    = args.bbox_loss,
        n_epoch      = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        img_size     = img_size,
        ckpt_dir     = args.ckpt,
    )

    # log final results
    best_epoch = int(np.argmax(history['map50'])) + 1
    best_map   = max(history['map50'])
    logging.info(f'Training complete.')
    logging.info(f'Best mAP@50 : {best_map:.4f} at epoch {best_epoch}')
    logging.info(f'Final loss  : {history["loss"][-1]:.4f}')

    # plot and save
    epochs = list(range(1, len(history['loss']) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(
        f'YOLO{ver} Training History ({args.bbox_loss} loss)',
        fontsize=14, fontweight='bold'
    )

    axes[0,0].plot(epochs, history['loss'],  marker='o', color='steelblue')
    axes[0,0].set_title('Total Loss');  axes[0,0].grid(True)

    axes[0,1].plot(epochs, history['box'],   marker='o', color='darkorange')
    axes[0,1].set_title('Box Loss');    axes[0,1].grid(True)

    axes[1,0].plot(epochs, history['conf'],  marker='o', color='tomato')
    axes[1,0].set_title('Conf Loss');   axes[1,0].grid(True)

    axes[1,1].plot(epochs, history['map50'], marker='s', color='green')
    axes[1,1].set_title('mAP@50 (val)'); axes[1,1].grid(True)
    axes[1,1].set_ylim(0, max(best_map * 1.2, 0.1))

    for ax in axes.flat:
        ax.set_xlabel('Epoch')

    plt.tight_layout()
    os.makedirs(args.output, exist_ok=True)
    plot_path = os.path.join(
        args.output, f'history_yolo{ver}_{args.bbox_loss}.png'
    )
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logging.info(f'Training plot saved → {plot_path}')


#Evaluation

def run_evaluate(args, cfg_path, weights_path, ver, img_size):
    """Run evaluation."""
    from evaluate import evaluate

    logging.info(f'Starting evaluation — YOLO{ver}')
    logging.info(f'Weights    : {weights_path}')
    logging.info(f'Image size : {img_size}')
    logging.info(f'Data       : {args.data}')
    logging.info(f'JSON       : {args.json}')

    map50 = evaluate(
        cfg_path     = cfg_path,
        weights_path = weights_path,
        path2data    = args.data,
        path2json    = args.json,
        model_ver    = ver,
        img_size     = img_size,
        batch_size   = args.batch_size,
    )

    logging.info(f'mAP@0.5 : {map50:.4f}')


#Main

def main():
    args = parse_args()

    # determine mode string for logger
    if args.infer:
        mode = 'infer'
    elif args.train:
        mode = 'train'
    elif args.evaluate:
        mode = 'evaluate'
    else:
        mode = 'unknown'

    # setup logger first so everything is captured
    setup_logger(
        log_dir    = args.logs,
        mode       = mode,
        model_name = args.model,
        bbox_loss  = args.bbox_loss if args.train else None,
    )

    # resolve paths
    cfg_path, weights_path, ver, img_size = get_paths(args)

    # verify files exist
    if not os.path.exists(cfg_path):
        logging.error(f'cfg not found: {cfg_path}')
        return

    if not os.path.exists(weights_path):
        logging.error(f'weights not found: {weights_path}')
        return

    # run selected mode
    if args.infer:
        run_infer(args, cfg_path, weights_path, ver, img_size)

    elif args.train:
        run_train(args, cfg_path, weights_path, ver, img_size)

    elif args.evaluate:
        run_evaluate(args, cfg_path, weights_path, ver, img_size)

    else:
        logging.info('Please specify a mode: --infer, --train, or --evaluate')
        logging.info('Examples:')
        logging.info('  python3 run.py --model yolov3 --weights weights/yolov3.weights --image images/dog-cycle-car.png --infer')
        logging.info('  python3 run.py --model yolov4 --bbox_loss ciou --epochs 5 --train')
        logging.info('  python3 run.py --model yolov4 --weights weights/yolov4.weights --evaluate')


if __name__ == '__main__':
    main()