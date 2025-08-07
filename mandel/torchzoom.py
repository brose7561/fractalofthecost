import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Device configuration: use MPS on macOS if available
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Using device: {device}")

# Parameters
WIDTH, HEIGHT = 600, 600
MAX_ITERS = 100
ZOOM_SCALE = 0.94
MIN_CONTRAST = 3.0

# Initial view window
x_center = -0.743643887037151
y_center =  0.131825904205330
x_width  =  2.8

frames = []
extents = []

# Sobel kernels for contrast detection
_kx = torch.tensor([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]], dtype=torch.float32, device=device)
_ky = torch.tensor([[1,  2,  1],
                    [0,  0,  0],
                    [-1, -2, -1]], dtype=torch.float32, device=device)
kernel_x = _kx.view(1,1,3,3)
kernel_y = _ky.view(1,1,3,3)

def mandelbrot_torch(h, w, maxit, x_min, x_max, y_min, y_max):
    # Create coordinate grids
    xs = torch.linspace(x_min, x_max, w, device=device)
    ys = torch.linspace(y_min, y_max, h, device=device)
    xx, yy = torch.meshgrid(xs, ys, indexing='xy')
    # Initialize z = 0, c = x+iy
    zr = torch.zeros_like(xx)
    zi = torch.zeros_like(yy)
    cr, ci = xx, yy

    # Iteration count tensor
    divtime = torch.full((h, w), maxit, dtype=torch.int32, device=device)

    for i in range(maxit):
        zr2 = zr*zr
        zi2 = zi*zi
        mag2 = zr2 + zi2

        # Mask of points still in the set
        mask = mag2 <= 4.0

        # Compute next z on masked points
        new_zr = zr2 - zi2 + cr
        new_zi = 2*zr*zi + ci
        zr = torch.where(mask, new_zr, zr)
        zi = torch.where(mask, new_zi, zi)

        # Record divergence time
        diverged = (mag2 > 4.0)
        just_diverged = diverged & (divtime == maxit)
        divtime[just_diverged] = i

    # Move to CPU for plotting/contrast
    return divtime.cpu().float()

def get_contrast_torch(data, sigma_frac=0.4):
    """
    Compute Sobel contrast and return:
      - raw max contrast (to compare against MIN_CONTRAST)
      - the unweighted contrast map (numpy)
      - the (i,j) index of the peak in a centre-weighted map
    """
    # Bring to device & add batch/channel dims
    d = data.to(device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    gx = F.conv2d(d, kernel_x, padding=1)
    gy = F.conv2d(d, kernel_y, padding=1)
    contrast = torch.hypot(gx, gy).squeeze(0).squeeze(0)  # [H,W]

    # Convert raw contrast to CPU numpy for potential plotting
    contrast_cpu = contrast.cpu().numpy()
    raw_max = float(contrast.max().item())

    # Build a Gaussian centre-weight mask
    H, W = contrast.shape
    ys = torch.arange(H, device=device).view(H,1).float()
    xs = torch.arange(W, device=device).view(1,W).float()
    cy, cx = (H-1)/2, (W-1)/2
    sigma = sigma_frac * max(H, W)
    gauss = torch.exp(-((ys-cy)**2 + (xs-cx)**2) / (2*sigma*sigma))

    # Centre-weighted contrast
    weighted = contrast * gauss
    # Find peak in weighted map
    idx_flat = torch.argmax(weighted)
    peak_i = int(idx_flat // W)
    peak_j = int(idx_flat % W)

    return raw_max, contrast_cpu, (peak_i, peak_j)

def generate_frame():
    global x_center, y_center, x_width

    # Compute current window
    y_height = x_width * HEIGHT / WIDTH
    x_min = x_center - x_width/2
    x_max = x_center + x_width/2
    y_min = y_center - y_height/2
    y_max = y_center + y_height/2

    # Mandelbrot + contrast
    mandel = mandelbrot_torch(HEIGHT, WIDTH, MAX_ITERS, x_min, x_max, y_min, y_max)
    contrast_score, contrast_map, (i, j) = get_contrast_torch(mandel)

    if contrast_score >= MIN_CONTRAST:
        # Recenter at the centre-biased peak
        x_center = x_min + (x_max - x_min) * (j / WIDTH)
        y_center = y_min + (y_max - y_min) * (i / HEIGHT)
        x_width *= ZOOM_SCALE

        frames.append(mandel.numpy())
        extents.append((x_min, x_max, y_min, y_max))


# Bootstrap with two frames
generate_frame()
generate_frame()

# Plot setup
fig, ax = plt.subplots(figsize=(6,6))
img = ax.imshow(frames[0], cmap='inferno', extent=extents[0])
ax.axis('off')
fig.suptitle("Mandelbrot Zoom – PyTorch + MPS", fontsize=12)

def update(frame_idx):
    if frame_idx >= len(frames) - 1:
        generate_frame()
    img.set_data(frames[frame_idx])
    img.set_extent(extents[frame_idx])
    return [img]

ani = FuncAnimation(fig, update, frames=200, interval=30, blit=True)
plt.show()
