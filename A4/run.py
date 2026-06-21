import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import os, random, time, argparse
from PIL import Image

#  Argument Parser

def get_args():
    parser = argparse.ArgumentParser(description='A4: Generative Models')
    parser.add_argument('--model',    type=str, choices=['gan', 'cyclegan', 'ddpm'], required=True)
    parser.add_argument('--dataset',  type=str, choices=['mnist', 'celeba'], default=None)
    parser.add_argument('--epochs',   type=int, default=20)
    parser.add_argument('--train',    action='store_true')
    parser.add_argument('--weights',  type=str, default=None)
    parser.add_argument('--test-image', type=str, default=None, dest='test_image')
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--n',        type=int, default=64)
    parser.add_argument('--schedule', type=str, choices=['linear', 'cosine'], default='linear')
    parser.add_argument('--lambda-cyc', type=float, default=10.0, dest='lambda_cyc')
    parser.add_argument('--d-lr',     type=float, default=2e-4, dest='d_lr')
    return parser.parse_args()

#  Utilities

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def denorm(t):
    return (t * 0.5 + 0.5).clamp(0, 1)

if __name__ == '__main__':
    args = get_args()
    set_seed(42)
    os.makedirs('saved', exist_ok=True)
    os.makedirs('logs',  exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    print(f'Device: {device}')
    print(f'Model:  {args.model}')
    print(f'Args:   {args}')

#  Part 1: GAN Architecture

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256),  nn.LeakyReLU(0.2),
            nn.Linear(256, 512),    nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),   nn.LeakyReLU(0.2),
            nn.Linear(1024, img_dim), nn.Tanh()
        )
    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 1024), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(1024, 512),     nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256),      nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1),        nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

#  Part 2: CycleGAN Architecture

class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3),
            nn.InstanceNorm2d(ch),
        )
    def forward(self, x):
        return x + self.block(x)

class CycleGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_res=6):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.Conv2d(ngf,   ngf*2, 3, stride=2, padding=1), nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1), nn.InstanceNorm2d(ngf*4), nn.ReLU(True),
        ]
        for _ in range(n_res):
            layers.append(ResidualBlock(ngf * 4))
        layers += [
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf,   3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf), nn.ReLU(True),
            nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7), nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)
    def forward(self, x):
        return self.model(x)

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=64):
        super().__init__()
        def block(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=2, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        self.model = nn.Sequential(
            *block(in_ch, ndf, norm=False),
            *block(ndf,   ndf*2),
            *block(ndf*2, ndf*4),
            nn.ZeroPad2d(1),
            nn.Conv2d(ndf*4, 1, 4, padding=1),
        )
    def forward(self, x):
        return self.model(x)

#  Part 3: DDPM Architecture

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -torch.arange(half, device=t.device).float() * (torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.conv1    = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2    = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_ch))
        self.residual = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.norm1    = nn.GroupNorm(8, out_ch)
        self.norm2    = nn.GroupNorm(8, out_ch)
    def forward(self, x, t_emb):
        out1 = self.conv1(x)
        h = self.norm1(out1 * torch.sigmoid(out1))
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        out2 = self.conv2(h)
        h = self.norm2(out2 * torch.sigmoid(out2))
        return h + self.residual(x)

class SimpleUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_dim=256):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.enc1 = ResBlock(in_ch,     base_ch,   time_dim)
        self.enc2 = ResBlock(base_ch,   base_ch*2, time_dim)
        self.down  = nn.MaxPool2d(2)
        self.bot   = ResBlock(base_ch*2, base_ch*4, time_dim)
        self.up    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2  = ResBlock(base_ch*4 + base_ch*2, base_ch*2, time_dim)
        self.dec1  = ResBlock(base_ch*2 + base_ch,   base_ch,   time_dim)
        self.out   = nn.Conv2d(base_ch, in_ch, 1)
    def forward(self, x, t):
        t_emb = self.time_embed(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down(e1), t_emb)
        b  = self.bot(self.down(e2), t_emb)
        d2 = self.dec2(torch.cat([self.up(b), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1), t_emb)
        return self.out(d1)

#  Data Loaders

def get_mnist_loader(batch_size=128):
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    ds = torchvision.datasets.MNIST('./data', train=True, download=True, transform=t)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

def get_celeba_loaders(batch_size=16, n_dark=30000, n_blonde=30000):
    t = transforms.Compose([
        transforms.CenterCrop(178),
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    torchvision.datasets.CelebA._check_integrity = lambda self: True
    full = torchvision.datasets.CelebA('./data', split='train', target_type='attr',
                                        download=False, transform=t)
    BLONDE_ATTR = 9
    dark_idx   = [i for i,(_, a) in enumerate(full) if a[BLONDE_ATTR] == 0][:n_dark]
    blonde_idx = [i for i,(_, a) in enumerate(full) if a[BLONDE_ATTR] == 1][:n_blonde]
    dark_ds   = torch.utils.data.Subset(full, dark_idx)
    blonde_ds = torch.utils.data.Subset(full, blonde_idx)
    loader_dark   = DataLoader(dark_ds,   batch_size=batch_size, shuffle=True,  num_workers=2, drop_last=True)
    loader_blonde = DataLoader(blonde_ds, batch_size=batch_size, shuffle=True,  num_workers=2, drop_last=True)
    print(f'Domain X (dark hair):   {len(dark_ds)} images')
    print(f'Domain Y (blonde hair): {len(blonde_ds)} images')
    return loader_dark, loader_blonde

#  DDPM Noise Schedules

T = 1000

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    t = torch.linspace(0, timesteps, timesteps + 1)
    alphas_bar = torch.cos(((t / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - (alphas_bar[1:] / alphas_bar[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

def get_noise_schedule(schedule='linear'):
    if schedule == 'cosine':
        betas = cosine_beta_schedule(T).to(device)
    else:
        betas = linear_beta_schedule(T).to(device)
    alphas       = 1.0 - betas
    alpha_bar    = torch.cumprod(alphas, dim=0)
    sqrt_ab      = torch.sqrt(alpha_bar)
    sqrt_one_ab  = torch.sqrt(1.0 - alpha_bar)
    sqrt_recip_a = torch.sqrt(1.0 / alphas)
    prev_ab      = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
    post_var     = betas * (1.0 - prev_ab) / (1.0 - alpha_bar)
    return betas, alphas, alpha_bar, sqrt_ab, sqrt_one_ab, sqrt_recip_a, post_var

def q_sample(x0, t, sqrt_ab, sqrt_one_ab, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    return sqrt_ab[t][:,None,None,None] * x0 + sqrt_one_ab[t][:,None,None,None] * noise

#  Training: GAN

def train_gan(args):
    print('\n--- Training Vanilla GAN on MNIST ---')
    loader = get_mnist_loader()
    Z_DIM  = 100
    G_gan  = Generator(Z_DIM).to(device)
    D_gan  = Discriminator().to(device)
    opt_G  = torch.optim.Adam(G_gan.parameters(), lr=2e-4,    betas=(0.5, 0.999))
    opt_D  = torch.optim.Adam(D_gan.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    criterion   = nn.BCELoss()
    fixed_noise = torch.randn(64, Z_DIM).to(device)

    g_losses, d_losses, epoch_times = [], [], []

    for epoch in range(args.epochs):
        t0 = time.time()
        g_ep, d_ep = [], []
        for real_imgs, _ in tqdm(loader, desc=f'GAN Epoch {epoch+1}/{args.epochs}'):
            B = real_imgs.size(0)
            real_imgs   = real_imgs.view(B, -1).to(device)
            real_labels = torch.ones(B, 1).to(device)
            fake_labels = torch.zeros(B, 1).to(device)

            # Train Discriminator
            z        = torch.randn(B, Z_DIM).to(device)
            fake_imgs = G_gan(z).detach()
            d_loss   = criterion(D_gan(real_imgs), real_labels) + \
                       criterion(D_gan(fake_imgs), fake_labels)
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # Train Generator
            z      = torch.randn(B, Z_DIM).to(device)
            g_loss = criterion(D_gan(G_gan(z)), real_labels)
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            g_ep.append(g_loss.item())
            d_ep.append(d_loss.item())

        ep_time = time.time() - t0
        epoch_times.append(ep_time)
        g_losses.append(np.mean(g_ep))
        d_losses.append(np.mean(d_ep))
        print(f'Epoch {epoch+1:02d} | G: {np.mean(g_ep):.3f} | D: {np.mean(d_ep):.3f} | {ep_time:.1f}s')

    # Save model
    torch.save({'G': G_gan.state_dict(), 'D': D_gan.state_dict()}, 'saved/gan_mnist.pt')
    print('Saved -> saved/gan_mnist.pt')

    # Plot losses
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(g_losses, label='Generator',     color='steelblue')
    axes[0].plot(d_losses, label='Discriminator', color='coral')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Vanilla GAN Training Losses')
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epoch_times, marker='o', color='green')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Seconds')
    axes[1].set_title(f'Epoch Time (avg: {np.mean(epoch_times):.1f}s)')
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig('plots/gan_training.png', dpi=150)
    print('Saved -> plots/gan_training.png')

    # Save generated grid
    G_gan.eval()
    with torch.no_grad():
        fake = G_gan(fixed_noise).view(-1, 1, 28, 28).cpu()
    grid = torchvision.utils.make_grid(fake, nrow=8, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f'GAN Generated MNIST (Epoch {args.epochs})')
    plt.axis('off')
    plt.savefig('plots/gan_generated.png', dpi=150, bbox_inches='tight')
    print('Saved -> plots/gan_generated.png')

#  Training: CycleGAN

def train_cyclegan(args):
    print('\n--- Training CycleGAN on CelebA ---')
    loader_dark, loader_blonde = get_celeba_loaders()

    G_net  = CycleGenerator().to(device)
    F_net  = CycleGenerator().to(device)
    D_X    = PatchDiscriminator().to(device)
    D_Y    = PatchDiscriminator().to(device)

    opt_G_all = torch.optim.Adam(
        list(G_net.parameters()) + list(F_net.parameters()), lr=2e-4, betas=(0.5, 0.999))
    opt_D_all = torch.optim.Adam(
        list(D_X.parameters()) + list(D_Y.parameters()), lr=2e-4, betas=(0.5, 0.999))

    adv_loss = nn.MSELoss()
    cyc_loss = nn.L1Loss()
    LAMBDA_CYC = args.lambda_cyc
    LAMBDA_IDT = 5.0

    g_losses, d_losses, epoch_times = [], [], []

    for epoch in range(args.epochs):
        t0 = time.time()
        g_ep, d_ep = [], []
        dark_iter   = iter(loader_dark)
        blonde_iter = iter(loader_blonde)
        n_batches   = min(len(loader_dark), len(loader_blonde))

        for _ in tqdm(range(n_batches), desc=f'CycleGAN Epoch {epoch+1}/{args.epochs}', mininterval=5.0):
            real_x, _ = next(dark_iter)
            real_y, _ = next(blonde_iter)
            real_x, real_y = real_x.to(device), real_y.to(device)

            # Train Generators
            opt_G_all.zero_grad()
            fake_y  = G_net(real_x)
            fake_x  = F_net(real_y)
            cycle_x = F_net(fake_y)
            cycle_y = G_net(fake_x)
            idt_x   = F_net(real_x)
            idt_y   = G_net(real_y)

            patch_shape = D_Y(fake_y).shape
            real_label  = torch.ones(patch_shape,  device=device)
            fake_label  = torch.zeros(patch_shape, device=device)

            loss_G_adv = adv_loss(D_Y(fake_y), real_label) + adv_loss(D_X(fake_x), real_label)
            loss_cyc   = cyc_loss(cycle_x, real_x) + cyc_loss(cycle_y, real_y)
            loss_idt   = cyc_loss(idt_x, real_x)   + cyc_loss(idt_y, real_y)
            loss_G     = loss_G_adv + LAMBDA_CYC * loss_cyc + LAMBDA_IDT * loss_idt
            loss_G.backward(); opt_G_all.step()

            # Train Discriminators
            opt_D_all.zero_grad()
            loss_DX = adv_loss(D_X(real_x), real_label) + adv_loss(D_X(fake_x.detach()), fake_label)
            loss_DY = adv_loss(D_Y(real_y), real_label) + adv_loss(D_Y(fake_y.detach()), fake_label)
            loss_D  = (loss_DX + loss_DY) * 0.5
            loss_D.backward(); opt_D_all.step()

            g_ep.append(loss_G.item())
            d_ep.append(loss_D.item())

        ep_time = time.time() - t0
        epoch_times.append(ep_time)
        g_losses.append(np.mean(g_ep))
        d_losses.append(np.mean(d_ep))
        print(f'Epoch {epoch+1:02d} | G: {np.mean(g_ep):.3f} | D: {np.mean(d_ep):.3f} | {ep_time:.1f}s')

    # Save
    suffix = 'cyc0' if LAMBDA_CYC == 0 else 'celeba'
    ckpt_path = f'saved/cyclegan_{suffix}.pt'
    torch.save({'G': G_net.state_dict(), 'F': F_net.state_dict()}, ckpt_path)
    print(f'Saved -> {ckpt_path}')

    # Plot losses
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(g_losses, label='Generator',     color='steelblue')
    axes[0].plot(d_losses, label='Discriminator', color='coral')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title(f'CycleGAN Training Losses (λ_cyc={LAMBDA_CYC})')
    axes[0].legend(); axes[0].grid(True)
    axes[1].plot(epoch_times, marker='o', color='green')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Seconds')
    axes[1].set_title(f'Epoch Time (avg: {np.mean(epoch_times):.1f}s)')
    axes[1].grid(True)
    plt.tight_layout()
    plot_path = f'plots/cyclegan_training_{suffix}.png'
    plt.savefig(plot_path, dpi=150)
    print(f'Saved -> {plot_path}')

    # Translation grid
    G_net.eval(); F_net.eval()
    with torch.no_grad():
        batch_x, _ = next(iter(DataLoader(torch.utils.data.Subset(
            torchvision.datasets.CelebA('./data', split='train', target_type='attr',
                download=False, transform=transforms.Compose([
                    transforms.CenterCrop(178), transforms.Resize(64),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])),
            [0,1,2,3]), batch_size=4)))
        fake_y = G_net(batch_x.to(device)).cpu()
        fake_x = F_net(batch_x.to(device)).cpu()

    fig, axes = plt.subplots(3, 4, figsize=(10, 8))
    for col in range(4):
        axes[0, col].imshow(denorm(batch_x[col]).permute(1,2,0)); axes[0, col].axis('off')
        axes[1, col].imshow(denorm(fake_y[col]).permute(1,2,0));  axes[1, col].axis('off')
        axes[2, col].imshow(denorm(fake_x[col]).permute(1,2,0));  axes[2, col].axis('off')
    axes[0,0].set_ylabel('Real Dark');      axes[1,0].set_ylabel('→ Blonde')
    axes[2,0].set_ylabel('Dark → Dark')
    plt.suptitle(f'CycleGAN Translations (λ_cyc={LAMBDA_CYC})', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'plots/cyclegan_grid_{suffix}.png', dpi=150)
    print(f'Saved -> plots/cyclegan_grid_{suffix}.png')

#  Training: DDPM

def train_ddpm(args):
    print(f'\n--- Training DDPM on MNIST (schedule={args.schedule}) ---')
    loader = get_mnist_loader()
    betas, alphas, alpha_bar, sqrt_ab, sqrt_one_ab, sqrt_recip_a, post_var = \
        get_noise_schedule(args.schedule)

    unet     = SimpleUNet().to(device)
    opt_ddpm = torch.optim.Adam(unet.parameters(), lr=2e-4)
    print(f'U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}')

    ddpm_losses, epoch_times = [], []

    for epoch in range(args.epochs):
        t0 = time.time()
        unet.train()
        ep_loss = []
        for x0, _ in tqdm(loader, desc=f'DDPM Epoch {epoch+1}/{args.epochs}'):
            x0   = x0.to(device)
            B    = x0.size(0)
            t    = torch.randint(0, T, (B,), device=device)
            noise  = torch.randn_like(x0)
            x_t    = q_sample(x0, t, sqrt_ab, sqrt_one_ab, noise)
            pred   = unet(x_t, t)
            loss   = F.mse_loss(pred, noise)
            opt_ddpm.zero_grad(); loss.backward(); opt_ddpm.step()
            ep_loss.append(loss.item())

        ep_time = time.time() - t0
        epoch_times.append(ep_time)
        ddpm_losses.append(np.mean(ep_loss))
        print(f'Epoch {epoch+1:03d} | Loss: {np.mean(ep_loss):.4f} | {ep_time:.1f}s')

    ckpt_path = f'saved/ddpm_mnist_{args.schedule}.pt'
    torch.save(unet.state_dict(), ckpt_path)
    print(f'Saved -> {ckpt_path}')

    plt.figure(figsize=(8, 4))
    plt.plot(ddpm_losses, marker='o', color='orange')
    plt.title(f'DDPM Training Loss ({args.schedule} schedule)')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig(f'plots/ddpm_loss_{args.schedule}.png', dpi=150)
    print(f'Saved -> plots/ddpm_loss_{args.schedule}.png')

    return unet, betas, sqrt_ab, sqrt_one_ab, sqrt_recip_a, post_var

#  DDPM Sampling

@torch.no_grad()
def p_sample(unet, x_t, t_scalar, betas, sqrt_one_ab, sqrt_recip_a, post_var):
    t_batch    = torch.full((x_t.size(0),), t_scalar, device=device, dtype=torch.long)
    pred_noise = unet(x_t, t_batch)
    coeff      = betas[t_scalar] / sqrt_one_ab[t_scalar]
    mean       = sqrt_recip_a[t_scalar] * (x_t - coeff * pred_noise)
    if t_scalar == 0:
        return mean
    return mean + torch.sqrt(post_var[t_scalar]) * torch.randn_like(x_t)

@torch.no_grad()
def generate_samples(unet, betas, sqrt_one_ab, sqrt_recip_a, post_var, n=64):
    unet.eval()
    x = torch.randn(n, 1, 28, 28).to(device)
    for t in tqdm(reversed(range(T)), total=T, desc='Sampling'):
        x = p_sample(unet, x, t, betas, sqrt_one_ab, sqrt_recip_a, post_var)
    return x

@torch.no_grad()
def generate_trajectory(unet, betas, sqrt_one_ab, sqrt_recip_a, post_var, n=8):
    unet.eval()
    x = torch.randn(n, 1, 28, 28).to(device)
    snapshots = []
    show_at   = {999, 800, 600, 400, 200, 100, 50, 0}
    for t in tqdm(reversed(range(T)), total=T, desc='Trajectory'):
        x = p_sample(unet, x, t, betas, sqrt_one_ab, sqrt_recip_a, post_var)
        if t in show_at:
            snapshots.append((t, x.cpu().clone()))
    return snapshots

#  CycleGAN: Test with own face

def test_face(args):
    print(f'\n--- Testing face image: {args.test_image} ---')
    if not os.path.exists(args.test_image):
        print(f'File not found: {args.test_image}'); return

    IMG_SIZE = 64
    face_transform = transforms.Compose([
        transforms.CenterCrop(min(Image.open(args.test_image).size)),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    img_tensor = face_transform(Image.open(args.test_image).convert('RGB')).unsqueeze(0).to(device)

    G_net = CycleGenerator().to(device)
    F_net = CycleGenerator().to(device)
    ckpt  = torch.load(args.weights, map_location=device)
    G_net.load_state_dict(ckpt['G']); F_net.load_state_dict(ckpt['F'])
    G_net.eval(); F_net.eval()

    with torch.no_grad():
        to_blonde = G_net(img_tensor).squeeze(0).cpu()
        to_dark   = F_net(img_tensor).squeeze(0).cpu()

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, title, img in zip(axes,
        ['Original', 'G: → Blonde Hair', 'F: → Dark Hair'],
        [img_tensor.squeeze(0).cpu(), to_blonde, to_dark]):
        ax.imshow(denorm(img).permute(1,2,0)); ax.set_title(title); ax.axis('off')
    plt.suptitle('CycleGAN — My Face', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/my_face_result.png', dpi=150, bbox_inches='tight')
    print('Saved -> plots/my_face_result.png')

#  Main Dispatcher

if __name__ == '__main__':
    args = get_args()
    set_seed(42)
    os.makedirs('saved', exist_ok=True)
    os.makedirs('logs',  exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    print(f'Device: {device}')
    print(f'Model:  {args.model}')
    print(f'Args:   {args}')

    if args.model == 'gan':
        if args.train:
            train_gan(args)

    elif args.model == 'cyclegan':
        if args.test_image and args.weights:
            test_face(args)
        elif args.train:
            train_cyclegan(args)

    elif args.model == 'ddpm':
        if args.train:
            unet, betas, sqrt_ab, sqrt_one_ab, sqrt_recip_a, post_var = train_ddpm(args)
            # Generate samples right after training
            samples = generate_samples(unet, betas, sqrt_one_ab, sqrt_recip_a, post_var, n=64)
            grid = torchvision.utils.make_grid(samples, nrow=8, normalize=True)
            plt.figure(figsize=(10,10))
            plt.imshow(grid.permute(1,2,0).cpu()); plt.axis('off')
            plt.title(f'DDPM Generated MNIST ({args.schedule} schedule)')
            plt.savefig(f'plots/ddpm_generated_{args.schedule}.png', dpi=150, bbox_inches='tight')
            print(f'Saved -> plots/ddpm_generated_{args.schedule}.png')
            # Trajectory
            snaps = generate_trajectory(unet, betas, sqrt_one_ab, sqrt_recip_a, post_var)
            fig, axes = plt.subplots(8, len(snaps), figsize=(len(snaps)*1.5, 12))
            for col, (t, imgs) in enumerate(snaps):
                for row in range(8):
                    axes[row][col].imshow(imgs[row].squeeze().numpy(), cmap='gray')
                    axes[row][col].axis('off')
                    if row == 0: axes[row][col].set_title(f't={t}', fontsize=9)
            plt.suptitle(f'Reverse Diffusion ({args.schedule})', fontsize=13)
            plt.tight_layout()
            plt.savefig(f'plots/ddpm_trajectory_{args.schedule}.png', dpi=150)
            print(f'Saved -> plots/ddpm_trajectory_{args.schedule}.png')

        elif args.generate and args.weights:
            betas, alphas, alpha_bar, sqrt_ab, sqrt_one_ab, sqrt_recip_a, post_var = \
                get_noise_schedule(args.schedule)
            unet = SimpleUNet().to(device)
            unet.load_state_dict(torch.load(args.weights, map_location=device))
            samples = generate_samples(unet, betas, sqrt_one_ab, sqrt_recip_a, post_var, n=args.n)
            grid = torchvision.utils.make_grid(samples, nrow=8, normalize=True)
            plt.figure(figsize=(10,10))
            plt.imshow(grid.permute(1,2,0).cpu()); plt.axis('off')
            plt.title(f'DDPM Generated MNIST ({args.schedule})')
            plt.savefig(f'plots/ddpm_generated_{args.schedule}.png', dpi=150, bbox_inches='tight')
            print(f'Saved -> plots/ddpm_generated_{args.schedule}.png')