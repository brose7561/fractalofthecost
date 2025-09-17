
# ================================= predict.py ==================================
# Place this file at: predict.py
from pyimagesearch.dataset import SegmentationDataset
from pyimagesearch.model import UNet
from pyimagesearch import config as C

from torch.utils.data import DataLoader
import torch
import numpy as np
import os
from tqdm import tqdm

# Reuse helpers from train (quick local import without circulars)
from train import colorize_mask, save_qualitative_batch, soft_dice_per_class

def load_model(path):
    model = UNet(in_channels=C.IN_CHANNELS, num_classes=C.NUM_CLASSES, base=64)
    sd = torch.load(path, map_location=C.DEVICE)
    model.load_state_dict(sd)
    model.to(C.DEVICE)
    model.eval()
    return model

def evaluate_and_visualise(model_path, split="test"):
    assert split in ("test", "validate")
    if split == "test":
        img_dir, mask_dir = C.IMG_TEST_DIR, C.MASK_TEST_DIR
        out_dir = os.path.join(C.VIZ_DIR, "test")
    else:
        img_dir, mask_dir = C.IMG_VAL_DIR, C.MASK_VAL_DIR
        out_dir = os.path.join(C.VIZ_DIR, "val_final")

    ds = SegmentationDataset(img_dir, mask_dir, (C.INPUT_IMAGE_HEIGHT, C.INPUT_IMAGE_WIDTH), augment=False)
    loader = DataLoader(ds, batch_size=C.BATCH_SIZE, shuffle=False, pin_memory=C.PIN_MEMORY, num_workers=os.cpu_count())

    model = load_model(model_path)

    dice_sum = torch.zeros(C.NUM_CLASSES, device=C.DEVICE)
    count_batches = 0

    with torch.no_grad():
        for imgs, masks, filenames in tqdm(loader, desc=f"[Infer:{split}]"):
            imgs  = imgs.to(C.DEVICE, non_blocking=True)
            masks = masks.to(C.DEVICE, non_blocking=True)

            logits = model(imgs)
            dice_c = soft_dice_per_class(logits, masks, C.NUM_CLASSES, C.SMOOTH)
            dice_sum += dice_c
            count_batches += 1

            preds = torch.argmax(logits, dim=1)
            save_qualitative_batch(
                imgs, masks, preds, filenames,
                out_dir=out_dir, tag=f"batch_{count_batches:04d}", max_items=C.SAVE_N_TEST_VIZ
            )

    per_class_dice = (dice_sum / max(count_batches,1)).detach().cpu().numpy()
    mean_dice = float(np.mean(per_class_dice))
    print(f"[RESULTS:{split}] mean DSC={mean_dice:.4f} | per-class={np.round(per_class_dice,4)}")
    print("Requirement reminder: > 0.90 DSC for all labels (not just mean).")

if __name__ == "__main__":
    # By default, evaluate the best checkpoint on the test set
    evaluate_and_visualise(C.BEST_MODEL_PATH, split="test")