# ================================== train.py ===================================
# Place this file at: train.py
from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet
from pyimagesearch import config as C

from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os

# ------------------------- Metrics & loss utilities ----------------------------
def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    # labels: [B,H,W] long -> [B,C,H,W] float
    return F.one_hot(labels, num_classes=num_classes).permute(0,3,1,2).float()

def soft_dice_per_class(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, smooth: float):
    """
    Compute per-class soft Dice between logits (pre-softmax) and integer labels.
    Returns: tensor [C] with dice per class (ignores absent-class batches gracefully).
    """
    probs = F.softmax(logits, dim=1)               # [B,C,H,W]
    target = one_hot(labels, num_classes)          # [B,C,H,W]

    dims = (0,2,3)  # sum over batch and spatial dims
    intersect = torch.sum(probs * target, dims)
    denom = torch.sum(probs, dims) + torch.sum(target, dims)
    dice = (2.0 * intersect + smooth) / (denom + smooth)  # [C]
    return dice

def combined_loss(logits, labels, ce_weight=1.0, dice_weight=C.DICE_WEIGHT):
    ce = F.cross_entropy(logits, labels)
    dice_c = soft_dice_per_class(logits, labels, C.NUM_CLASSES, C.SMOOTH)
    mean_dice = dice_c.mean()
    # Maximise dice => minimise (1 - dice)
    loss = ce_weight * ce + dice_weight * (1.0 - mean_dice)
    return loss, ce.detach(), mean_dice.detach(), dice_c.detach()

# ------------------------------- Visualisation ---------------------------------
def colorize_mask(mask_hw: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert HxW integer mask to color RGB for quick visualisation.
    Uses a fixed palette; safe for up to ~10 classes.
    """
    palette = np.array([
        [0, 0, 0],
        [220, 20, 60],
        [65, 105, 225],
        [34, 139, 34],
        [255, 140, 0],
        [128, 0, 128],
        [0, 206, 209],
        [255, 215, 0],
        [199, 21, 133],
        [139, 69, 19],
    ], dtype=np.uint8)
    rgb = np.zeros((mask_hw.shape[0], mask_hw.shape[1], 3), dtype=np.uint8)
    for c in range(num_classes):
        rgb[mask_hw == c] = palette[c % len(palette)]
    return rgb

def save_qualitative_batch(img_bchw, gt_bhw, pred_bhw, filenames, out_dir, tag, max_items=8):
    os.makedirs(out_dir, exist_ok=True)
    b = min(img_bchw.size(0), max_items)
    rows = []
    for i in range(b):
        img = img_bchw[i].cpu().numpy().transpose(1,2,0)  # [H,W,1]
        img = (img * 255.0).clip(0,255).astype(np.uint8)
        img = np.repeat(img, 3, axis=2)                   # grayscale->RGB

        gt  = gt_bhw[i].cpu().numpy().astype(np.uint8)
        pr  = pred_bhw[i].cpu().numpy().astype(np.uint8)

        gt_rgb = colorize_mask(gt, C.NUM_CLASSES)
        pr_rgb = colorize_mask(pr, C.NUM_CLASSES)

        # side-by-side: image | GT | Pred
        trip = np.concatenate([img, gt_rgb, pr_rgb], axis=1)
        rows.append(trip)

    grid = np.concatenate(rows, axis=0)
    out_path = os.path.join(out_dir, f"{tag}.png")
    cv2 = __import__("cv2")  # lazy import to avoid headless issues if unused
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

# ---------------------------------- Training -----------------------------------
def main():
    print(f"[INFO] Device: {C.DEVICE} | Classes: {C.NUM_CLASSES} | ImgSize: {C.INPUT_IMAGE_HEIGHT}x{C.INPUT_IMAGE_WIDTH}")
    # Datasets & loaders
    train_ds = SegmentationDataset(C.IMG_TRAIN_DIR, C.MASK_TRAIN_DIR, (C.INPUT_IMAGE_HEIGHT, C.INPUT_IMAGE_WIDTH), augment=True)
    val_ds   = SegmentationDataset(C.IMG_VAL_DIR,   C.MASK_VAL_DIR,   (C.INPUT_IMAGE_HEIGHT, C.INPUT_IMAGE_WIDTH), augment=False)

    train_loader = DataLoader(train_ds, batch_size=C.BATCH_SIZE, shuffle=True,  pin_memory=C.PIN_MEMORY, num_workers=os.cpu_count())
    val_loader   = DataLoader(val_ds,   batch_size=C.BATCH_SIZE, shuffle=False, pin_memory=C.PIN_MEMORY, num_workers=os.cpu_count())

    print(f"[INFO] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Model / optimiser
    model = UNet(in_channels=C.IN_CHANNELS, num_classes=C.NUM_CLASSES, base=64).to(C.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=C.INIT_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # Trackers
    history = {"train_loss": [], "val_loss": [], "val_mean_dice": []}
    best_mean_dice = -1.0

    # Train epochs
    for epoch in range(1, C.NUM_EPOCHS+1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"[E{epoch}/{C.NUM_EPOCHS}] Train")
        for imgs, masks, _ in pbar:
            imgs  = imgs.to(C.DEVICE, non_blocking=True)       # [B,1,H,W]
            masks = masks.to(C.DEVICE, non_blocking=True)      # [B,H,W]

            logits = model(imgs)                                # [B,C,H,W]
            loss, ce, mean_dice, _ = combined_loss(logits, masks)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.3f}", dice=f"{mean_dice.item():.3f}")

        avg_train_loss = running_loss / len(train_ds)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        val_loss_sum = 0.0
        dice_sum = torch.zeros(C.NUM_CLASSES, device=C.DEVICE)
        count_batches = 0

        with torch.no_grad():
            for imgs, masks, filenames in tqdm(val_loader, desc=f"[E{epoch}] Val"):
                imgs  = imgs.to(C.DEVICE, non_blocking=True)
                masks = masks.to(C.DEVICE, non_blocking=True)

                logits = model(imgs)
                loss, _, _, dice_c = combined_loss(logits, masks)
                val_loss_sum += loss.item() * imgs.size(0)

                dice_sum += dice_c
                count_batches += 1

            per_class_dice = (dice_sum / max(count_batches,1)).detach().cpu().numpy()
            mean_dice = float(np.mean(per_class_dice))
            avg_val_loss = val_loss_sum / len(val_ds)

            history["val_loss"].append(avg_val_loss)
            history["val_mean_dice"].append(mean_dice)

            # Save a small qualitative batch each epoch (argmax)
            preds = torch.argmax(logits, dim=1)  # [B,H,W]
            save_qualitative_batch(
                imgs, masks, preds, filenames,
                out_dir=os.path.join(C.VIZ_DIR, "val"),
                tag=f"epoch_{epoch:03d}",
                max_items=C.SAVE_N_VAL_VIZ,
            )

            print(f"[VAL] loss={avg_val_loss:.4f} | meanDSC={mean_dice:.4f} | per-class={np.round(per_class_dice,4)}")

        # Scheduler on mean DSC (maximize)
        scheduler.step(mean_dice)

        # Checkpointing
        torch.save(model.state_dict(), C.LAST_MODEL_PATH)
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save(model.state_dict(), C.BEST_MODEL_PATH)
            print(f"[CKPT] New best mean DSC {best_mean_dice:.4f} — saved to {C.BEST_MODEL_PATH}")

        # Early note on DSC requirement
        if mean_dice >= 0.90:
            print("[INFO] Mean DSC ≥ 0.90 — requirement threshold hit (ensure per-label also ≥ 0.90 in final test).")

    # Plots
    plt.figure(); plt.plot(history["train_loss"], label="train"); plt.plot(history["val_loss"], label="val")
    plt.title("Loss"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.savefig(C.LOSS_PLOT_PATH); plt.close()

    plt.figure(); plt.plot(history["val_mean_dice"], label="val mean DSC")
    plt.title("Validation Mean DSC"); plt.xlabel("epoch"); plt.ylabel("DSC"); plt.legend()
    plt.ylim([0,1]); plt.savefig(C.DSC_PLOT_PATH); plt.close()

    print(f"[DONE] Best val mean DSC: {best_mean_dice:.4f}")

if __name__ == "__main__":
    main()
