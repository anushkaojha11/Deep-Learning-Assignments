import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from sklearn.manifold import TSNE
import random, os, math, time, argparse, logging
import timm as timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)
os.makedirs('saved', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('data',  exist_ok=True)
os.makedirs('logs',  exist_ok=True)

CLASSES = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

def setup_logger(name):
    """Logs to both console and logs/<name>.log"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file
    fh = logging.FileHandler(f'logs/{name}.log')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

def get_args():
    parser = argparse.ArgumentParser(description='A3: Self-Supervised Learning')
    parser.add_argument('--model',        type=str,   required=True, choices=['dino','mae'])
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--train',        action='store_true')
    parser.add_argument('--evaluate',     action='store_true')
    parser.add_argument('--linear',       action='store_true')
    parser.add_argument('--weights',      type=str,   default=None)
    parser.add_argument('--no-centering', action='store_true')
    parser.add_argument('--n-local',      type=int,   default=4)
    parser.add_argument('--mask-ratio',   type=float, default=0.75)
    return parser.parse_args()

# ─── Constants ────────────────────────────────────────────────────────────────
EVAL_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

MAE_MEAN = [0.4914, 0.4822, 0.4465]
MAE_STD  = [0.247,  0.243,  0.261]

# ─── DINO Augmentation ────────────────────────────────────────────────────────
class DINOAugmentation:
    def __init__(self, image_size=32, n_local=4):
        normalize = transforms.Normalize([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010])
        flip_jitter = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.4, 1.0)),
            *flip_jitter,
            transforms.ToTensor(), normalize
        ])
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.05, 0.4)),
            *flip_jitter,
            transforms.ToTensor(), normalize
        ])
        self.n_local = n_local

    def __call__(self, img):
        global1 = self.global_transform(img)
        global2 = self.global_transform(img)
        locals_ = [self.local_transform(img) for _ in range(self.n_local)]
        return [global1, global2] + locals_


class CIFAR10DINO(Dataset):
    def __init__(self, root='./data', train=True, n_local=4):
        self.dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=True)
        self.augment = DINOAugmentation(n_local=n_local)
    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.augment(img), label

def dino_collate(batch):
    crops_list, labels = zip(*batch)
    n_views = len(crops_list[0])
    stacked = [torch.stack([crops_list[i][v] for i in range(len(crops_list))]) for v in range(n_views)]
    return stacked, torch.tensor(labels)

# ─── DINO Model ───────────────────────────────────────────────────────────────
class DINOHead(nn.Module):
    def __init__(self, in_dim=192, hidden_dim=512, out_dim=256, n_layers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 2):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, out_dim, bias=False))
        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(nn.Linear(out_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)

def build_dino_model(out_dim=256):
    vit = timm.create_model('vit_tiny_patch16_224', pretrained=False,
                             img_size=32, patch_size=4, num_classes=0)
    head = DINOHead(in_dim=vit.embed_dim, out_dim=out_dim)
    return vit, head

# ─── DINO Loss ────────────────────────────────────────────────────────────────
class DINOLoss(nn.Module):
    def __init__(self, out_dim=256, teacher_temp=0.04, student_temp=0.1,
                 center_momentum=0.9, use_centering=True):
        super().__init__()
        self.student_temp    = student_temp
        self.teacher_temp    = teacher_temp
        self.center_momentum = center_momentum
        self.use_centering   = use_centering
        self.register_buffer('center', torch.zeros(1, out_dim))
        self.center_norms = []  # track across epochs

    def forward(self, student_out, teacher_out):
        s_probs = [F.log_softmax(s / self.student_temp, dim=-1) for s in student_out]

        if self.use_centering:
            t_probs = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach()
                       for t in teacher_out]
        else:
            t_probs = [F.softmax(t / self.teacher_temp, dim=-1).detach()
                       for t in teacher_out]

        total_loss, n_loss_terms = 0, 0
        for t_idx, t_prob in enumerate(t_probs):
            for s_idx, s_log_prob in enumerate(s_probs):
                if s_idx == t_idx:
                    continue
                total_loss += -(t_prob * s_log_prob).sum(dim=-1).mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms
        self.update_center(torch.stack(teacher_out).mean(dim=0))
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_mean):
        self.center = self.center * self.center_momentum + teacher_mean * (1 - self.center_momentum)

# ─── MAE Components ───────────────────────────────────────────────────────────
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, embed_dim=192):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def get_2d_sincos_pos_embed(embed_dim, grid_size):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)

    def sincos_1d(pos, dim):
        omega = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))
        out = pos.reshape(-1, 1) * omega.reshape(1, -1)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)

    half = embed_dim // 2
    emb = np.concatenate([sincos_1d(grid_h.flatten(), half),
                           sincos_1d(grid_w.flatten(), half)], axis=1)
    return torch.tensor(emb, dtype=torch.float32)

class MAEEncoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3,
                 embed_dim=192, depth=6, num_heads=3, mlp_ratio=4.0, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio  = mask_ratio
        self.patch_embed = PatchEmbed(img_size, patch_size, in_ch, embed_dim)
        self.embed_dim   = embed_dim

        pos_embed = get_2d_sincos_pos_embed(embed_dim, img_size // patch_size)
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.0, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def random_masking(self, x):
        N, L, D = x.shape
        n_keep  = int(L * (1 - self.mask_ratio))
        noise   = torch.rand(N, L, device=x.device)
        ids_shuffle  = noise.argsort(dim=1)
        ids_restore  = ids_shuffle.argsort(dim=1)
        ids_keep     = ids_shuffle[:, :n_keep]
        x_visible    = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask         = torch.ones(N, L, device=x.device)
        mask[:, :n_keep] = 0
        mask         = torch.gather(mask, 1, ids_restore)
        return x_visible, mask, ids_restore

    def forward(self, x):
        x = self.patch_embed(x) + self.pos_embed
        x_vis, mask, ids_restore = self.random_masking(x)
        x_vis = self.norm(self.transformer(x_vis))
        return x_vis, mask, ids_restore

class MAEDecoder(nn.Module):
    def __init__(self, n_patches, patch_size=4, in_ch=3,
                 encoder_dim=192, decoder_dim=128, depth=4, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        patch_pixels = patch_size * patch_size * in_ch
        grid_size    = int(math.sqrt(n_patches))

        self.embed      = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        pos_embed = get_2d_sincos_pos_embed(decoder_dim, grid_size)
        self.register_buffer('pos_embed', pos_embed.unsqueeze(0))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=num_heads,
            dim_feedforward=int(decoder_dim * mlp_ratio),
            dropout=0.0, activation='gelu',
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(decoder_dim)
        self.pred = nn.Linear(decoder_dim, patch_pixels)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x_vis, ids_restore):
        N        = x_vis.size(0)
        x        = self.embed(x_vis)
        n_masked = ids_restore.size(1) - x.size(1)
        x_full   = torch.cat([x, self.mask_token.expand(N, n_masked, -1)], dim=1)
        x_full   = torch.gather(x_full, 1,
                       ids_restore.unsqueeze(-1).expand(-1, -1, x_full.size(-1)))
        x_full   = self.norm(self.transformer(x_full + self.pos_embed))
        return self.pred(x_full)

class MAE(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3,
                 encoder_dim=192, encoder_depth=6, encoder_heads=3,
                 decoder_dim=128, decoder_depth=4, decoder_heads=4,
                 mask_ratio=0.75, norm_pix_loss=True):
        super().__init__()
        self.patch_size    = patch_size
        self.in_ch         = in_ch
        self.norm_pix_loss = norm_pix_loss
        self.encoder = MAEEncoder(img_size, patch_size, in_ch,
                                   encoder_dim, encoder_depth, encoder_heads,
                                   mask_ratio=mask_ratio)
        n_patches = self.encoder.patch_embed.n_patches
        self.decoder = MAEDecoder(n_patches, patch_size, in_ch,
                                   encoder_dim, decoder_dim, decoder_depth, decoder_heads)

    def patchify(self, imgs):
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], self.in_ch, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)
        return x.reshape(imgs.shape[0], h * w, p * p * self.in_ch)

    def forward(self, imgs):
        x_vis, mask, ids_restore = self.encoder(imgs)
        pred   = self.decoder(x_vis, ids_restore)
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean   = target.mean(dim=-1, keepdim=True)
            var    = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred, mask

# ─── DINO Training ────────────────────────────────────────────────────────────
def train_dino(args, logger):
    OUT_DIM  = 256
    EMA_M    = 0.996

    student_vit, student_head = build_dino_model(OUT_DIM)
    teacher_vit, teacher_head = build_dino_model(OUT_DIM)
    student_vit, student_head = student_vit.to(device), student_head.to(device)
    teacher_vit, teacher_head = teacher_vit.to(device), teacher_head.to(device)

    teacher_vit.load_state_dict(student_vit.state_dict())
    teacher_head.load_state_dict(student_head.state_dict())
    for p in teacher_vit.parameters():  p.requires_grad = False
    for p in teacher_head.parameters(): p.requires_grad = False

    dataset    = CIFAR10DINO(n_local=args.n_local)
    loader     = DataLoader(dataset, batch_size=64, shuffle=True,
                            num_workers=2, drop_last=True, collate_fn=dino_collate)
    loss_fn    = DINOLoss(out_dim=OUT_DIM, use_centering=not args.no_centering).to(device)
    optimizer  = torch.optim.AdamW(
        list(student_vit.parameters()) + list(student_head.parameters()),
        lr=5e-4, weight_decay=0.04
    )

    losses       = []
    center_norms = []
    epoch_times  = []

    for epoch in range(args.epochs):
        student_vit.train(); student_head.train()
        ep = []
        t0 = time.time()

        for crops, _ in tqdm(loader, desc=f'DINO {epoch+1}/{args.epochs}'):
            crops       = [c.to(device) for c in crops]
            student_out = [student_head(student_vit(c)) for c in crops]
            with torch.no_grad():
                teacher_out = [teacher_head(teacher_vit(crops[0])),
                               teacher_head(teacher_vit(crops[1]))]
            loss = loss_fn(student_out, teacher_out)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            with torch.no_grad():
                for s_p, t_p in zip(student_vit.parameters(), teacher_vit.parameters()):
                    t_p.data = EMA_M * t_p.data + (1 - EMA_M) * s_p.data
                for s_p, t_p in zip(student_head.parameters(), teacher_head.parameters()):
                    t_p.data = EMA_M * t_p.data + (1 - EMA_M) * s_p.data
            ep.append(loss.item())

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_loss    = np.mean(ep)
        cnorm       = loss_fn.center.norm().item()
        losses.append(avg_loss)
        center_norms.append(cnorm)
        logger.info(f'Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Center norm: {cnorm:.4f} | Time: {elapsed:.1f}s')

    # save checkpoint
    run_name = f"dino_nlocal{args.n_local}{'_nocentering' if args.no_centering else ''}"
    ckpt_path = f'saved/{run_name}.pt'
    torch.save({'student_vit': student_vit.state_dict(),
                'student_head': student_head.state_dict()}, ckpt_path)
    logger.info(f'Saved checkpoint: {ckpt_path}')

    # plot loss + center norm
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
    ax1.plot(losses, marker='o', color='darkorange')
    ax1.set_title(f'DINO Loss ({run_name})'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-Entropy'); ax1.grid(True)
    ax2.plot(center_norms, marker='s', color='purple')
    ax2.set_title('Center Norm across Epochs'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('||center||'); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/{run_name}_loss.png', dpi=120, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved plot: plots/{run_name}_loss.png')

    return student_vit, student_head


# ─── MAE Training ─────────────────────────────────────────────────────────────
def train_mae(args, logger):
    mae_model = MAE(
        img_size=32, patch_size=4, in_ch=3,
        encoder_dim=192, encoder_depth=6, encoder_heads=3,
        decoder_dim=128, decoder_depth=4, decoder_heads=4,
        mask_ratio=args.mask_ratio, norm_pix_loss=True,
    ).to(device)

    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MAE_MEAN, MAE_STD),
    ])
    dataset   = torchvision.datasets.CIFAR10('./data', train=True, transform=train_tf, download=True)
    loader    = DataLoader(dataset, batch_size=128, shuffle=True,
                           num_workers=2, pin_memory=True, drop_last=True)
    optimizer = torch.optim.AdamW(mae_model.parameters(), lr=1.5e-4,
                                   weight_decay=0.05, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    losses      = []
    epoch_times = []
    mae_model.train()

    for epoch in range(args.epochs):
        ep = []
        t0 = time.time()
        for imgs, _ in tqdm(loader, desc=f'MAE {epoch+1}/{args.epochs}'):
            imgs = imgs.to(device)
            loss, _, _ = mae_model(imgs)
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(mae_model.parameters(), max_norm=1.0)
            optimizer.step()
            ep.append(loss.item())
        scheduler.step()

        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_loss = np.mean(ep)
        losses.append(avg_loss)
        logger.info(f'Epoch {epoch+1:03d} | Recon Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s')

    # save checkpoint
    run_name  = f"mae_mask{int(args.mask_ratio*100)}"
    ckpt_path = f'saved/{run_name}_encoder.pt'
    torch.save(mae_model.encoder.state_dict(), ckpt_path)
    logger.info(f'Saved checkpoint: {ckpt_path}')

    # plot loss
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(losses, marker='o', color='steelblue')
    ax.set_title(f'MAE Loss (mask={args.mask_ratio})'); ax.set_xlabel('Epoch'); ax.set_ylabel('MSE (masked patches)');
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/{run_name}_loss.png', dpi=120, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved plot: plots/{run_name}_loss.png')

    return mae_model

# ─── Linear Evaluation: DINO ──────────────────────────────────────────────────
def linear_eval_dino(student_vit, logger, run_name):
    for p in student_vit.parameters(): p.requires_grad = False
    student_vit.eval()

    embed_dim = student_vit.embed_dim
    clf       = nn.Linear(embed_dim, 10).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)

    train_ds = torchvision.datasets.CIFAR10('./data', train=True,  transform=EVAL_TF, download=True)
    test_ds  = torchvision.datasets.CIFAR10('./data', train=False, transform=EVAL_TF, download=True)
    trl = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2)
    tel = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)

    for epoch in range(10):
        clf.train(); correct = total = 0
        for imgs, labels in tqdm(trl, desc=f'DINO Linear Eval {epoch+1}/10'):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad(): h = student_vit(imgs)
            loss = F.cross_entropy(clf(h), labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            correct += (clf(h).argmax(1) == labels).sum().item()
            total   += labels.size(0)
        logger.info(f'  Linear Eval Epoch {epoch+1}/10 | Train Acc: {correct/total*100:.2f}%')

    clf.eval(); correct = total = 0
    embeddings, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tel:
            imgs, labels = imgs.to(device), labels.to(device)
            h = student_vit(imgs)
            correct    += (clf(h).argmax(1) == labels).sum().item()
            total      += labels.size(0)
            embeddings.append(h.cpu())
            all_labels.append(labels.cpu())

    acc = correct / total * 100
    logger.info(f'DINO Linear Eval Test Accuracy: {acc:.2f}%')

    embeddings = torch.cat(embeddings)
    all_labels = torch.cat(all_labels)
    plot_tsne(embeddings, all_labels, f'DINO ({run_name})', f'plots/{run_name}_tsne.png', logger)
    return acc


# ─── Linear Evaluation: MAE ───────────────────────────────────────────────────
def linear_eval_mae(mae_model, logger, run_name):
    mae_model.encoder.mask_ratio = 0.0
    mae_model.encoder.eval()
    for p in mae_model.encoder.parameters(): p.requires_grad = False

    clf       = nn.Linear(mae_model.encoder.embed_dim, 10).to(device)
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-3)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(MAE_MEAN, MAE_STD)
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(MAE_MEAN, MAE_STD)
    ])
    train_ds = torchvision.datasets.CIFAR10('./data', train=True,  transform=train_tf)
    test_ds  = torchvision.datasets.CIFAR10('./data', train=False, transform=test_tf)
    trl = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=2)
    tel = DataLoader(test_ds,  batch_size=256, shuffle=False, num_workers=2)

    for epoch in range(10):
        clf.train(); correct = total = 0
        for imgs, labels in tqdm(trl, desc=f'MAE Linear Eval {epoch+1}/10'):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                x_vis, _, _ = mae_model.encoder(imgs)
                feats = x_vis.mean(dim=1)
            logits = clf(feats)
            loss   = F.cross_entropy(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
        logger.info(f'  Linear Eval Epoch {epoch+1}/10 | Train Acc: {correct/total*100:.2f}%')

    clf.eval(); correct = total = 0
    embeddings, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in tel:
            imgs, labels = imgs.to(device), labels.to(device)
            x_vis, _, _ = mae_model.encoder(imgs)
            feats = x_vis.mean(dim=1)
            correct    += (clf(feats).argmax(1) == labels).sum().item()
            total      += labels.size(0)
            embeddings.append(feats.cpu())
            all_labels.append(labels.cpu())

    acc = correct / total * 100
    logger.info(f'MAE Linear Eval Test Accuracy: {acc:.2f}%')

    embeddings = torch.cat(embeddings)
    all_labels = torch.cat(all_labels)
    plot_tsne(embeddings, all_labels, f'MAE ({run_name})', f'plots/{run_name}_tsne.png', logger)
    return acc


# ─── t-SNE Plot ───────────────────────────────────────────────────────────────
def plot_tsne(embeddings, labels, title, save_path, logger):
    logger.info(f'Running t-SNE for {title} ...')
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    idx    = np.random.choice(len(embeddings), min(2000, len(embeddings)), replace=False)
    proj   = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(
                 embeddings[idx].numpy())
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(10):
        mask_c = labels[idx].numpy() == c
        ax.scatter(proj[mask_c,0], proj[mask_c,1], c=[colors[c]],
                   label=CLASSES[c], alpha=0.6, s=10)
    ax.set_title(f't-SNE: {title}', fontsize=12)
    ax.legend(fontsize=7, markerscale=2)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved t-SNE: {save_path}')


# ─── MAE Reconstruction Visualization ────────────────────────────────────────
def visualize_mae(mae_model, logger, run_name):
    mae_model.encoder.mask_ratio = 0.75
    mae_model.eval()

    test_tf  = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(MAE_MEAN, MAE_STD)
    ])
    imgs_viz, _ = next(iter(DataLoader(
        torchvision.datasets.CIFAR10('./data', train=False, transform=test_tf),
        batch_size=8, shuffle=True
    )))
    imgs_viz = imgs_viz.to(device)

    with torch.no_grad():
        _, pred, mask = mae_model(imgs_viz)

    p    = mae_model.patch_size
    h_g  = w_g = 32 // p

    def unpatchify(patches):
        N = patches.size(0)
        x = patches.reshape(N, h_g, w_g, p, p, 3)
        x = x.permute(0, 5, 1, 3, 2, 4)
        return x.reshape(N, 3, h_g*p, w_g*p)

    pred_imgs = unpatchify(pred.cpu())
    mean_t    = torch.tensor(MAE_MEAN).view(3,1,1)
    std_t     = torch.tensor(MAE_STD).view(3,1,1)

    orig_np   = (imgs_viz.cpu() * std_t + mean_t).clamp(0,1).permute(0,2,3,1).numpy()
    pred_np   = (pred_imgs       * std_t + mean_t).clamp(0,1).permute(0,2,3,1).numpy()

    mask_exp  = mask.cpu().view(-1, h_g, w_g).unsqueeze(1)
    mask_exp  = mask_exp.repeat_interleave(p, dim=2).repeat_interleave(p, dim=3)
    mask_np   = mask_exp.expand(-1,3,-1,-1).permute(0,2,3,1).numpy()
    masked_np = orig_np.copy()
    masked_np[mask_np.astype(bool)] = 0.5

    N_show = 4
    fig, axes = plt.subplots(3, N_show, figsize=(2*N_show, 6))
    for row, (imgs_row, title) in enumerate(zip(
        [orig_np, masked_np, pred_np],
        ['Original', 'Masked (75%)', 'Reconstructed']
    )):
        axes[row, 0].set_ylabel(title, fontsize=10)
        for col in range(N_show):
            axes[row, col].imshow(imgs_row[col])
            axes[row, col].axis('off')
    plt.suptitle(f'MAE Reconstruction ({run_name})', fontsize=13, y=1.02)
    plt.tight_layout()
    save_path = f'plots/{run_name}_reconstruction.png'
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    logger.info(f'Saved reconstruction: {save_path}')
    
if __name__ == '__main__':
    args = get_args()

    # build run name
    if args.model == 'dino':
        run_name = f"dino_nlocal{args.n_local}{'_nocentering' if args.no_centering else ''}"
    else:
        run_name = f"mae_mask{int(args.mask_ratio*100)}"

    logger = setup_logger(run_name)
    logger.info(f'Device  : {device}')
    logger.info(f'Model   : {args.model}')
    logger.info(f'Epochs  : {args.epochs}')
    logger.info(f'Train   : {args.train}')
    logger.info(f'Eval    : {args.evaluate}')
    logger.info(f'Run name: {run_name}')

    # ── DINO ──────────────────────────────────────────────────────────────────
    if args.model == 'dino':

        if args.train:
            logger.info('Starting DINO training ...')
            student_vit, student_head = train_dino(args, logger)

        if args.evaluate and args.linear:
            logger.info('Starting DINO linear evaluation ...')

            # load weights if not just trained
            if not args.train:
                student_vit, student_head = build_dino_model(out_dim=256)
                student_vit  = student_vit.to(device)
                student_head = student_head.to(device)
                ckpt_path = args.weights if args.weights else f'saved/{run_name}.pt'
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                student_vit.load_state_dict(ckpt['student_vit'])
                logger.info(f'Loaded weights from {ckpt_path}')

            acc = linear_eval_dino(student_vit, logger, run_name)
            logger.info(f'Final DINO Linear Eval Accuracy: {acc:.2f}%')

    # ── MAE ───────────────────────────────────────────────────────────────────
    elif args.model == 'mae':

        if args.train:
            logger.info('Starting MAE training ...')
            mae_model = train_mae(args, logger)
            logger.info('Generating MAE reconstruction visualization ...')
            visualize_mae(mae_model, logger, run_name)

        if args.evaluate and args.linear:
            logger.info('Starting MAE linear evaluation ...')

            # load weights if not just trained
            if not args.train:
                mae_model = MAE(
                    img_size=32, patch_size=4, in_ch=3,
                    encoder_dim=192, encoder_depth=6, encoder_heads=3,
                    decoder_dim=128, decoder_depth=4, decoder_heads=4,
                    mask_ratio=args.mask_ratio, norm_pix_loss=True,
                ).to(device)
                ckpt_path = args.weights if args.weights else f'saved/{run_name}_encoder.pt'
                mae_model.encoder.load_state_dict(
                    torch.load(ckpt_path, map_location=device, weights_only=False)
                )
                logger.info(f'Loaded weights from {ckpt_path}')

            acc = linear_eval_mae(mae_model, logger, run_name)
            logger.info(f'Final MAE Linear Eval Accuracy: {acc:.2f}%')

    logger.info('Done.')