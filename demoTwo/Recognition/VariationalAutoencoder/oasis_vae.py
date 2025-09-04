# oasis_vae.py
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("Warning CUDA not found. Using CPU")

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="/home/groups/comp3710/OASIS")
parser.add_argument("--img_size", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=35)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--latent_dim", type=int, default=2)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--augment", action="store_true")
args = parser.parse_args([]) if "__file__" not in globals() else parser.parse_args()

train_dir = os.path.join(args.data_root, "keras_png_slices_train")
val_dir   = os.path.join(args.data_root, "keras_png_slices_validate")
test_dir  = os.path.join(args.data_root, "keras_png_slices_test")
os.makedirs("vae_outputs", exist_ok=True)

class PNGFolderDataset(Dataset):
    def __init__(self, root, img_size=128, train_mode=False, augment=False):
        self.paths = sorted(
            glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
            + glob.glob(os.path.join(root, "**", "*.jpg"), recursive=True)
            + glob.glob(os.path.join(root, "**", "*.jpeg"), recursive=True)
        )
        if len(self.paths) == 0:
            raise RuntimeError(f"No images found under {root}")
        self.img_size = img_size
        self.train_mode = train_mode
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def _load_image(self, path):
        img = Image.open(path).convert("L")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        if self.train_mode and self.augment:
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None, ...]
        return tensor

    def __getitem__(self, idx):
        x = self._load_image(self.paths[idx])
        return x

train_set = PNGFolderDataset(train_dir, img_size=args.img_size, train_mode=True,  augment=args.augment)
val_set   = PNGFolderDataset(val_dir,   img_size=args.img_size, train_mode=False, augment=False)
test_set  = PNGFolderDataset(test_dir,  img_size=args.img_size, train_mode=False, augment=False)

pin = (device.type == "cuda")
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=pin)
val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

class ConvVAE(nn.Module):
    def __init__(self, img_size=128, latent_dim=2):
        super().__init__()
        c = 32
        self.enc = nn.Sequential(
            nn.Conv2d(1,   c, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(c,  c*2, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(c*2,c*4, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(c*4,c*8, 4, 2, 1), nn.ReLU(True),
        )
        feat_spatial = img_size // 16
        self.feat_h = self.feat_w = feat_spatial
        self.feat_c = c*8
        enc_out = self.feat_c * self.feat_h * self.feat_w
        self.fc_mu     = nn.Linear(enc_out, latent_dim)
        self.fc_logvar = nn.Linear(enc_out, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, enc_out)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c*8, c*4, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(c*4, c*2, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(c*2, c,   4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(c,   1,   4, 2, 1)
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), self.feat_c, self.feat_h, self.feat_w)
        x_hat = self.dec(h)
        return x_hat

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

model = ConvVAE(img_size=args.img_size, latent_dim=args.latent_dim).to(device)
if device.type == "cuda":
    print(torch.cuda.get_device_name(0))
print("Model No. of Parameters:", sum(p.numel() for p in model.parameters()))
print(model)

def vae_loss(x_hat, x, mu, logvar):
    bce = F.binary_cross_entropy_with_logits(x_hat, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + kld, bce, kld

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

def save_reconstructions(epoch, x, x_hat, max_n=16, fname_prefix="vae_outputs/recon"):
    n = min(x.size(0), max_n)
    x = x[:n].detach().cpu().numpy()
    x_hat = torch.sigmoid(x_hat[:n]).detach().cpu().numpy()
    fig, axes = plt.subplots(2, n, figsize=(n*1.2, 2.4))
    for i in range(n):
        axes[0, i].imshow(x[i,0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(x_hat[i,0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    plt.tight_layout()
    fig.savefig(f"{fname_prefix}_epoch{epoch:03d}.png", dpi=150)
    plt.close(fig)

def save_manifold_grid(decoder, latent_dim=2, img_size=128, grid_size=20, span=3.0, fname="vae_outputs/manifold_grid.png"):
    if latent_dim != 2:
        return
    with torch.no_grad():
        lin = np.linspace(-span, span, grid_size)
        grid = []
        for yi in lin:
            row = []
            for xi in lin:
                z = torch.tensor([[xi, yi]], dtype=torch.float32, device=device)
                x_hat = torch.sigmoid(decoder(z)).cpu().numpy()[0,0]
                row.append(x_hat)
            grid.append(np.concatenate(row, axis=1))
        full = np.concatenate(grid, axis=0)
        plt.figure(figsize=(6,6))
        plt.imshow(full, cmap="gray", vmin=0, vmax=1)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(fname, dpi=200)
        plt.close()

print("> Training")
start = time.time()
best_val = float("inf")

fixed_test_batch = next(iter(test_loader))
fixed_test_batch = fixed_test_batch.to(device)

for epoch in range(1, args.epochs + 1):
    model.train()
    running_loss, running_bce, running_kld = 0.0, 0.0, 0.0
    for x in train_loader:
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
            x_hat, mu, logvar = model(x)
            loss, bce, kld = vae_loss(x_hat, x, mu, logvar)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * x.size(0)
        running_bce  += bce.item()  * x.size(0)
        running_kld  += kld.item()  * x.size(0)

    train_loss = running_loss / len(train_set)
    train_bce  = running_bce  / len(train_set)
    train_kld  = running_kld  / len(train_set)

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

    print(f"Epoch [{epoch}/{args.epochs}] "
          f"Train: total={train_loss:.4f} bce={train_bce:.4f} kld={train_kld:.4f} | "
          f"Val: total={val_loss:.4f} bce={val_bce:.4f} kld={val_kld:.4f}")

    with torch.no_grad():
        x = fixed_test_batch
        x_hat, _, _ = model(x)
        save_reconstructions(epoch, x, x_hat, max_n=12)

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "vae_outputs/vae_oasis_best.pth")

end = time.time()
print("Training took " + str(end - start) + " secs or " + str((end - start)/60) + " mins in total")

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

torch.save(model.state_dict(), "vae_outputs/vae_oasis_final.pth")
print("Model saved to vae_outputs/vae_oasis_final.pth")

save_manifold_grid(model.decode, latent_dim=args.latent_dim, img_size=args.img_size, grid_size=20, span=3.0,
                   fname="vae_outputs/manifold_grid.png")
print("Saved manifold grid to vae_outputs/manifold_grid.png")
```
