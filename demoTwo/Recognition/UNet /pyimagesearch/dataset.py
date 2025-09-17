# =========================== pyimagesearch/dataset.py ===========================

from torch.utils.data import Dataset
from torchvision import transforms as T
import torch
import cv2
import os
from typing import List, Tuple, Optional

class SegmentationDataset(Dataset):
    """
    Segmentation dataset for OASIS PNG slices. Assumes a *file-name match*
    between image and mask folders (e.g., 'case_00001.png' in both).
    - Images: grayscale [H, W] -> tensor [1, H, W], float in [0,1]
    - Masks: per-pixel class indices [H, W], dtype=torch.long (not one-hot)
    """
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        img_size: Tuple[int, int],
        augment: bool = False
    ):
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_h, self.img_w = img_size
        self.augment = augment

        self.image_paths = sorted([
            os.path.join(self.image_dir, f) for f in os.listdir(self.image_dir)
            if f.lower().endswith(".png")
        ])
        # mask paths resolved by file name
        self.mask_paths = [os.path.join(self.mask_dir, os.path.basename(p)) for p in self.image_paths]

        assert len(self.image_paths) == len(self.mask_paths) and len(self.image_paths) > 0, \
            f"No matching image/mask pairs found in {image_dir} and {mask_dir}."

        # Basic transforms (same resize for img and mask)
        self.to_tensor = T.ToTensor()  # converts HxW (uint8) -> 1xHxW float [0,1]
        self.resize = (self.img_w, self.img_h)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_image(self, p: str):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {p}")
        img = cv2.resize(img, self.resize, interpolation=cv2.INTER_AREA)
        return img

    def _load_mask(self, p: str):
        # Expect mask PNG with integer labels [0..C-1]
        mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {p}")
        mask = cv2.resize(mask, self.resize, interpolation=cv2.INTER_NEAREST)
        return mask

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        img = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        # Optional lightweight augmentation (same geometry must apply to both)
        if self.augment:
            # Horizontal flip
            if torch.rand(1).item() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            # Vertical flip
            if torch.rand(1).item() < 0.2:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)

        # To tensors
        img_t = self.to_tensor(img)           # [1, H, W] float
        mask_t = torch.from_numpy(mask).long()  # [H, W] long indices

        return img_t, mask_t, os.path.basename(img_path)  # return filename for viz
