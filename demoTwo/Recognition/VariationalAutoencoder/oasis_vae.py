# oasis_vae.py
# A compact, fully-commented Variational Autoencoder (VAE) for OASIS PNG slices.
# This version:
# - Keeps the original architecture and BCE-with-logits objective you were using.
# - Fixes the AMP deprecation warnings by switching from torch.cuda.amp.* to torch.amp.*.
# - Explains, line-by-line, what each part does so you can learn the flow.

import os, glob, time, math, argparse, random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob, os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import umap


import shutil


# -----------------------------
# Device setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

# -----------------------------
# Arguments
# -----------------------------
parser = argparse.ArgumentParser()
# Root of the OASIS folder that contains the predefined splits:
#   keras_png_slices_train / keras_png_slices_validate / keras_png_slices_test
parser.add_argument("--data_root", type=str, default="/home/groups/comp3710/OASIS")
# Images are resized to img_size x img_size, and the conv stack expects size divisible by 16
parser.add_argument("--img_size", type=int, default=128)
# Batch size per step
parser.add_argument("--batch_size", type=int, default=128)
# Number of training epochs
parser.add_argument("--epochs", type=int, default=35)
# Adam learning rate
parser.add_argument("--lr", type=float, default=1e-3)
# Size of the latent vector (z). Keep 2 to visualize the manifold grid.
parser.add_argument("--latent_dim", type=int, default=64)
# DataLoader workers: on HPCs, keep this low (often 0–2). Default 1 to avoid warnings.
parser.add_argument("--num_workers", type=int, default=1)
# Simple augmentation (horizontal flip). Applies only to train set when enabled.
parser.add_argument("--augment", action="store_true")
# Optional: seed for reproducibility (set to None to skip)
parser.add_argument("--seed", type=int, default=42)

# This trick supports both: running as a script and from a notebook.
args = parser.parse_args([]) if "__file__" not in globals() else parser.parse_args()

# -----------------------------
# Reproducibility (optional)
# -----------------------------
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

# -----------------------------
# Paths for the predefined splits
# -----------------------------
train_dir = os.path.join(args.data_root, "keras_png_slices_train")
val_dir   = os.path.join(args.data_root, "keras_png_slices_validate")
test_dir  = os.path.join(args.data_root, "keras_png_slices_test")

# Output directory for images and weights (overwrite)

if os.path.exists("vae_outputs"):
    shutil.rmtree("vae_outputs")
os.makedirs("vae_outputs")


# -----------------------------
# Dataset: loads PNG/JPG files into grayscale tensors in [0,1]
# with optional augmentation for training
# -----------------------------

class AddGaussianNoise:
    def __init__(self, std=0.01):
        self.std = std
    def __call__(self, x):
        if self.std <= 0:
            return x
        return (x + torch.randn_like(x) * self.std).clamp(0.0, 1.0)
    

class PNGFolderDataset(Dataset):
    def __init__(self, root, img_size=128, train_mode=False, augment=False):
        # Collect PNG/JPG files
        self.paths = sorted(
            glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
            + glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
            + glob.glob(os.path.join(root, "**", "*.jpeg"), recursive=True)
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {root}")
        self.train_mode = train_mode
        self.augment_flag = augment

        # Base pipeline: grayscale → resize → tensor
        self.base_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        # Augmentation pipeline: applied instead of base_tf if training+augment
        self.aug_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR, fill=0),
            transforms.RandomResizedCrop(img_size, scale=(0.9, 1.1), ratio=(1.0, 1.0), interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            AddGaussianNoise(std=0.01),
        ])

    def __len__(self):
        return len(self.paths)

    def augment(self, img):
        # Explicit hook for augmentation
        return self.aug_tf(img)

    def _load_image(self, path):
        img = Image.open(path)
        if self.train_mode and self.augment_flag:
            return self.augment(img)
        return self.base_tf(img)

    def __getitem__(self, idx):
        return self._load_image(self.paths[idx])



# Instantiate datasets for the predefined splits
train_set = PNGFolderDataset(train_dir, img_size=args.img_size, train_mode=True,  augment=args.augment)
val_set   = PNGFolderDataset(val_dir,   img_size=args.img_size, train_mode=False, augment=False)
test_set  = PNGFolderDataset(test_dir,  img_size=args.img_size, train_mode=False, augment=False)

# DataLoaders: pin_memory=True helps GPU transfers; keep workers modest on clusters
pin = (device.type == "cuda")
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

# -----------------------------
# Model: Convolutional VAE
#   Encoder: 128x128 -> 64 -> 32 -> 16 -> 8 spatial, channels 1->32->64->128->256
#   Latent:  flatten 8*8*256 -> fc_mu/fc_logvar -> reparameterize z
#   Decoder:  z -> fc -> 8*8*256 -> deconvs -> 128x128x1 (logits)
# -----------------------------
class ConvVAE(nn.Module):
    def __init__(self, img_size=128, latent_dim=2):
        super().__init__()
        # Base channel multiplier
        c = 32

        # Encoder: stride-2 convs halve spatial dims each time, while increasing channels
        self.enc = nn.Sequential(
            nn.Conv2d(1,   c,   4, 2, 1), nn.ReLU(True),   # 128 -> 64,   1 -> 32
            nn.Conv2d(c,  c*2, 4, 2, 1), nn.ReLU(True),   # 64  -> 32,  32 -> 64
            nn.Conv2d(c*2,c*4, 4, 2, 1), nn.ReLU(True),   # 32  -> 16,  64 -> 128
            nn.Conv2d(c*4,c*8, 4, 2, 1), nn.ReLU(True),   # 16  -> 8,  128 -> 256
        )
        # Spatial size after 4 stride-2 downsamples = img_size / 16
        feat_spatial = img_size // 16
        self.feat_h = self.feat_w = feat_spatial
        self.feat_c = c * 8

        # Flattened feature size at the encoder output
        enc_out = self.feat_c * self.feat_h * self.feat_w

        # Latent heads: mean and log-variance for a diagonal Gaussian, they are dif samples of the same vector at this stage. d
        self.fc_mu     = nn.Linear(enc_out, latent_dim)
        self.fc_logvar = nn.Linear(enc_out, latent_dim)

        # Decoder prelayer: z -> flattened conv feature map
        self.fc_dec = nn.Linear(latent_dim, enc_out)

        # Decoder: transpose-convs upsample back to 128x128x1 (logits)
        # No final activation here because we use BCEWithLogits loss (it applies sigmoid internally)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c*8, c*4, 4, 2, 1), nn.ReLU(True),  # 8  -> 16, 256 -> 128
            nn.ConvTranspose2d(c*4, c*2, 4, 2, 1), nn.ReLU(True),  # 16 -> 32, 128 -> 64
            nn.ConvTranspose2d(c*2, c,   4, 2, 1), nn.ReLU(True),  # 32 -> 64,   64 -> 32
            nn.ConvTranspose2d(c,   1,   4, 2, 1)                  # 64 -> 128,  32 -> 1  (logits)
        )

    def encode(self, x):
        # Pass through encoder conv stack and flatten
        h = self.enc(x)
        h = h.view(x.size(0), -1)
        # Predict μ and logσ² (log-variance)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # z = μ + σ * ε, with ε ~ N(0, I)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # Expand z back to a conv feature map and run deconvs to image logits
        h = self.fc_dec(z)
        h = h.view(z.size(0), self.feat_c, self.feat_h, self.feat_w)
        x_hat = self.dec(h)
        return x_hat

    def forward(self, x):
        # Full VAE pass: encode -> sample -> decode
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

# Build model and show summary
model = ConvVAE(img_size=args.img_size, latent_dim=args.latent_dim).to(device)
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
print("Model No. of Parameters:", sum(p.numel() for p in model.parameters()))
print(model)

# -----------------------------
# VAE loss:
#  - Reconstruction: BCEWithLogits between logits and targets in [0,1]
#  - KL: encourage q(z|x) ~ N(0,I) for a smooth, generative latent space
# -----------------------------
def vae_loss(x_hat, x, mu, logvar):
    # Reconstruction term: we pass raw logits and let BCEWithLogits apply sigmoid internally
    bce = F.binary_cross_entropy_with_logits(x_hat, x, reduction="mean")
    # KL term: analytical KL between q(z|x) = N(μ,σ²) and p(z) = N(0, I)
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld, bce, kld

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Mixed precision (AMP): new API to avoid deprecation warnings
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))





def save_manifold_grid(model, loader, device, fname="vae_outputs/latent_umap.png", n_neighbors=15, min_dist=0.1, max_points=None, random_state=42):
    model.eval()
    zs = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device, non_blocking=True)
            mu, _ = model.encode(x)
            zs.append(mu.cpu())
    z = torch.cat(zs, dim=0).numpy()
    if max_points is not None and z.shape[0] > max_points:
        idx = np.random.RandomState(random_state).choice(z.shape[0], size=max_points, replace=False)
        z = z[idx]
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    emb = reducer.fit_transform(z)
    plt.figure(figsize=(5,5))
    plt.scatter(emb[:,0], emb[:,1], s=6, alpha=0.6)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()


# -----------------------------
# Training loop
# -----------------------------
print("> Training")
start = time.time()
best_val = float("inf")

# Take a fixed test batch for consistent recon snapshots
fixed_test_batch = next(iter(test_loader))
fixed_test_batch = fixed_test_batch.to(device)

for epoch in range(1, args.epochs + 1):
    model.train()
    running_loss, running_bce, running_kld = 0.0, 0.0, 0.0

    for x in train_loader:
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        # AMP autocast context with the new API (prevents the deprecation warning)
        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            x_hat, mu, logvar = model(x)
            loss, bce, kld = vae_loss(x_hat, x, mu, logvar)

        # Scale, backward, and step with AMP
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track sums (multiply by batch size) to average later
        running_loss += loss.item() * x.size(0)
        running_bce  += bce.item()  * x.size(0)
        running_kld  += kld.item()  * x.size(0)

    # Epoch averages over the whole training set
    train_loss = running_loss / len(train_set)
    train_bce  = running_bce  / len(train_set)
    train_kld  = running_kld  / len(train_set)

    # Validation pass (no grad)
    model.eval()
    val_loss, val_bce, val_kld = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x in val_loader:
            x = x.to(device, non_blocking=True)
            x_hat, mu, logvar = model(x)
            loss, bce, kld = vae_loss(x_hat, x, mu, logvar)
            val_loss += loss.item() * x.size(0)
            val_bce  += bce.item()  * x.size(0)
            val_kld  += kld.item()  * x.size(0)

    val_loss /= len(val_set)
    val_bce  /= len(val_set)
    val_kld  /= len(val_set)

    # Epoch log: "bce" is the reconstruction term here (BCE-with-logits)
    print(f"Epoch [{epoch}/{args.epochs}] "
          f"Train: total={train_loss:.4f} bce={train_bce:.4f} kld={train_kld:.4f} | "
          f"Val: total={val_loss:.4f} bce={val_bce:.4f} kld={val_kld:.4f}")

    # Save a consistent reconstruction panel each epoch for qualitative progress
    with torch.no_grad():
        x = fixed_test_batch
        x_hat, _, _ = model(x)

    # Keep the best validation model
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "vae_outputs/vae_oasis_best.pth")

end = time.time()
print("Training took " + str(end - start) + " secs or " + str((end - start)/60) + " mins in total")

# -----------------------------
# Testing loop (held-out split)
# -----------------------------
print("> Testing")
start = time.time()
model.eval()
test_loss, test_bce, test_kld = 0.0, 0.0, 0.0
with torch.no_grad():
    for x in test_loader:
        x = x.to(device, non_blocking=True)
        x_hat, mu, logvar = model(x)
        loss, bce, kld = vae_loss(x_hat, x, mu, logvar)
        test_loss += loss.item() * x.size(0)
        test_bce  += bce.item()  * x.size(0)
        test_kld  += kld.item()  * x.size(0)

test_loss /= len(test_set)
test_bce  /= len(test_set)
test_kld  /= len(test_set)
print(f"Test: total={test_loss:.4f} bce={test_bce:.4f} kld={test_kld:.4f}")

end = time.time()
print("Testing took " + str(end - start) + " secs or " + str((end - start)/60) + " mins in total")

# Save final weights
torch.save(model.state_dict(), "vae_outputs/vae_oasis_final.pth")
print("Model saved to vae_outputs/vae_oasis_final.pth")

# Save a 2-D latent manifold grid if latent_dim=2
save_manifold_grid(model, test_loader, device, fname="vae_outputs/latent_umap.png")
print("Saved manifold grid to vae_outputs/manifold_grid.png")
