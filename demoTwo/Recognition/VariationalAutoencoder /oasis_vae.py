# oasis_vae.py
import os, glob, random, math, time, argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils as vutils
import matplotlib.pyplot as plt

try:
    import nibabel as nib
    HAS_NIB = True
except Exception:
    HAS_NIB = False

class OasisMRIDataset(Dataset):
    def __init__(self, root, img_size=128, slice_stride=4, max_slices_per_vol=None):
        self.items = []
        img_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
        for ext in img_exts:
            self.items.extend([("img", p, None) for p in glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True)])
        npy_paths = glob.glob(os.path.join(root, "**", "*.npy"), recursive=True)
        for p in npy_paths:
            self.items.append(("npy", p, None))
        nii_paths = glob.glob(os.path.join(root, "**", "*.nii*"), recursive=True)
        if nii_paths and not HAS_NIB:
            raise RuntimeError("Found NIfTI files but nibabel is not installed. Install with: pip install nibabel")
        if nii_paths and HAS_NIB:
            for p in nii_paths:
                try:
                    vol = nib.load(p).get_fdata()
                    if vol.ndim == 4:
                        vol = vol[..., 0]
                    axis = np.argmax(vol.shape)
                    n = vol.shape[axis]
                    idxs = list(range(0, n, max(1, slice_stride)))
                    if max_slices_per_vol is not None:
                        idxs = idxs[:max_slices_per_vol]
                    for s in idxs:
                        self.items.append(("nii", p, (axis, s)))
                except Exception:
                    continue
        if not self.items:
            raise RuntimeError(f"No files found under {root}")
        self.tf = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def __len__(self):
        return len(self.items)

    def _to_pil(self, arr):
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[..., 0]
        arr = arr.astype(np.float32)
        if np.nanmax(arr) > 1.0 or np.nanmin(arr) < 0.0:
            mn, mx = np.nanmin(arr), np.nanmax(arr)
            if mx > mn:
                arr = (arr - mn) / (mx - mn)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    def __getitem__(self, idx):
        t, p, meta = self.items[idx]
        if t == "img":
            im = Image.open(p).convert("L")
        elif t == "npy":
            arr = np.load(p)
            im = self._to_pil(arr)
        else:
            axis, s = meta
            vol = nib.load(p).get_fdata()
            if vol.ndim == 4:
                vol = vol[..., 0]
            if axis == 0:
                arr = vol[s, :, :]
            elif axis == 1:
                arr = vol[:, s, :]
            else:
                arr = vol[:, :, s]
            im = self._to_pil(arr)
        x = self.tf(im)
        return x

class VAE(nn.Module):
    def __init__(self, z_dim=2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.enc_out = 256 * 8 * 8
        self.fc_mu = nn.Linear(self.enc_out, z_dim)
        self.fc_logvar = nn.Linear(self.enc_out, z_dim)
        self.fc_dec = nn.Linear(z_dim, self.enc_out)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Tanh(),
        )

    def encode(self, x):
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(z.size(0), 256, 8, 8)
        x = self.dec(h)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar

def loss_fn(x, xhat, mu, logvar, beta):
    rec = F.mse_loss(xhat, x, reduction="sum") / x.size(0)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return rec + beta * kld, rec, kld

def save_reconstructions(model, loader, device, out_path, n=16):
    model.eval()
    xs = []
    xhs = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            xhat, _, _ = model(x)
            xs.append(x[: n].cpu())
            xhs.append(xhat[: n].cpu())
            break
    x = torch.cat(xs, dim=0)
    xhat = torch.cat(xhs, dim=0)
    grid = torch.cat([x, xhat], dim=0)
    grid = (grid + 1) / 2
    vutils.save_image(grid, out_path, nrow=n, padding=2)

def save_prior_samples(model, device, out_path, z_dim=2, n=64):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n, z_dim, device=device)
        x = model.decode(z).cpu()
        x = (x + 1) / 2
        vutils.save_image(x, out_path, nrow=int(math.sqrt(n)), padding=2)

def save_manifold_grid(model, device, out_path, z_dim=2, grid_size=20, lim=3.0):
    if z_dim != 2:
        return
    model.eval()
    lin = torch.linspace(-lim, lim, grid_size)
    z = torch.stack(torch.meshgrid(lin, lin, indexing="xy"), dim=-1).view(-1, 2).to(device)
    with torch.no_grad():
        x = model.decode(z).cpu()
        x = (x + 1) / 2
        vutils.save_image(x, out_path, nrow=grid_size, padding=1)

def encode_dataset(model, loader, device):
    model.eval()
    zs = []
    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            mu, logvar = model.encode(x)
            zs.append(mu.cpu())
    return torch.cat(zs, dim=0).numpy()

def save_latent_scatter(z, out_path):
    if z.shape[1] != 2:
        return
    plt.figure(figsize=(5,5))
    plt.scatter(z[:,0], z[:,1], s=4, alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/home/groups/comp3710/", type=str)
    p.add_argument("--img_size", default=128, type=int)
    p.add_argument("--batch_size", default=128, type=int)
    p.add_argument("--epochs", default=30, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--beta", default=1.0, type=float)
    p.add_argument("--z_dim", default=2, type=int)
    p.add_argument("--workers", default=4, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--slice_stride", default=4, type=int)
    p.add_argument("--max_slices_per_vol", default=None, type=int)
    p.add_argument("--val_ratio", default=0.1, type=float)
    p.add_argument("--outdir", default="oasis_vae_runs", type=str)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = OasisMRIDataset(
        args.data_root, img_size=args.img_size, slice_stride=args.slice_stride, max_slices_per_vol=args.max_slices_per_vol
    )
    n_val = max(1, int(len(dataset) * args.val_ratio))
    n_train = len(dataset) - n_val
    g = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=g)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(z_dim=args.z_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_loss = float("inf")
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for x in train_loader:
            x = x.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                xhat, mu, logvar = model(x)
                loss, rec, kld = loss_fn(x, xhat, mu, logvar, args.beta)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run_loss += loss.item() * x.size(0)
        run_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device, non_blocking=True)
                xhat, mu, logvar = model(x)
                loss, rec, kld = loss_fn(x, xhat, mu, logvar, args.beta)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"epoch {epoch}/{args.epochs} train_loss {run_loss:.4f} val_loss {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.outdir, "oasis_vae.pt"))

        if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
            save_reconstructions(model, val_loader, device, os.path.join(args.outdir, f"recon_epoch_{epoch}.png"))
            save_prior_samples(model, device, os.path.join(args.outdir, f"samples_epoch_{epoch}.png"), z_dim=args.z_dim)
    elapsed = time.time() - start
    print(f"done {elapsed:.1f}s")

    model.load_state_dict(torch.load(os.path.join(args.outdir, "oasis_vae.pt"), map_location=device))
    save_reconstructions(model, val_loader, device, os.path.join(args.outdir, "recon_best.png"))
    save_prior_samples(model, device, os.path.join(args.outdir, "samples_best.png"), z_dim=args.z_dim)
    save_manifold_grid(model, device, os.path.join(args.outdir, "manifold_grid.png"), z_dim=args.z_dim, grid_size=20, lim=3.0)
    z = encode_dataset(model, val_loader, device)
    save_latent_scatter(z, os.path.join(args.outdir, "latent_scatter.png"))

if __name__ == "__main__":
    main()
