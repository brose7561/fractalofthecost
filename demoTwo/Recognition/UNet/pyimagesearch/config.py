# =========================== pyimagesearch/config.py ============================


import os
import torch

# ---- Data roots (customise if needed) -----------------------------------------
# Rangpur cluster OASIS path and split folders (images + masks)
DATA_ROOT = os.environ.get("OASIS_ROOT", "/home/groups/comp3710/OASIS")

IMG_TRAIN_DIR = os.path.join(DATA_ROOT, "keras_png_slices_train")
IMG_VAL_DIR   = os.path.join(DATA_ROOT, "keras_png_slices_validate")
IMG_TEST_DIR  = os.path.join(DATA_ROOT, "keras_png_slices_test")

MASK_TRAIN_DIR = os.path.join(DATA_ROOT, "keras_png_slices_seg_train")
MASK_VAL_DIR   = os.path.join(DATA_ROOT, "keras_png_slices_seg_validate")
MASK_TEST_DIR  = os.path.join(DATA_ROOT, "keras_png_slices_seg_test")

# ---- Model / task config ------------------------------------------------------
# OASIS preprocessed commonly has 4 classes (background, GM, WM, CSF),
# but leave configurable. If unsure, set via env: OASIS_NUM_CLASSES.
NUM_CLASSES = int(os.environ.get("OASIS_NUM_CLASSES", "4"))
IN_CHANNELS = 1  # OASIS PNG slices are grayscale

# Input size: UNet uses "same" padding, but keep a consistent training size.
# 256x256 is typical for OASIS; adjust if needed (must be divisible by 2^depth).
INPUT_IMAGE_HEIGHT = int(os.environ.get("OASIS_IMG_H", "256"))
INPUT_IMAGE_WIDTH  = int(os.environ.get("OASIS_IMG_W", "256"))

# ---- Training config ----------------------------------------------------------
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False

INIT_LR   = float(os.environ.get("OASIS_LR", "1e-3"))
NUM_EPOCHS = int(os.environ.get("OASIS_EPOCHS", "50"))
BATCH_SIZE = int(os.environ.get("OASIS_BATCH", "16"))

# Loss mixing: total = CE + (DICE_WEIGHT * (1 - mean_dice))
DICE_WEIGHT = float(os.environ.get("OASIS_DICE_W", "1.0"))
SMOOTH      = 1.0  # dice smoothing to avoid div-by-zero

# ---- Thresholds / visualisation ----------------------------------------------
# Argmax is used for categorical prediction; THRESHOLD not required for softmax.
SAVE_N_VAL_VIZ = int(os.environ.get("OASIS_VIZ_PER_EPOCH", "8"))
SAVE_N_TEST_VIZ = int(os.environ.get("OASIS_VIZ_TEST", "12"))

# ---- Output paths -------------------------------------------------------------
BASE_OUTPUT = os.environ.get("OASIS_OUT", "output")
MODEL_DIR   = os.path.join(BASE_OUTPUT, "models")
PLOTS_DIR   = os.path.join(BASE_OUTPUT, "plots")
VIZ_DIR     = os.path.join(BASE_OUTPUT, "viz")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "unet_oasis_best.pth")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "unet_oasis_last.pth")
LOSS_PLOT_PATH  = os.path.join(PLOTS_DIR, "loss_curve.png")
DSC_PLOT_PATH   = os.path.join(PLOTS_DIR, "dsc_curve.png")

# Ensure output directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(VIZ_DIR, exist_ok=True)