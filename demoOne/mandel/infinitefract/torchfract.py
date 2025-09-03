# fractal_zoom_api.py

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fractals import (
    generate_mandelbrot,
    generate_julia,
    generate_burning_ship,
    generate_newton,
    generate_cantor_set,
    generate_takagi_curve,
    generate_sierpinski_triangle,
    generate_feigenbaum_attractor,
)

# ——————————————————————————————————————————————
# Configurable parameters
# ——————————————————————————————————————————————
FRACTAL = "burning_ship"
# options: "mandelbrot","julia","burning_ship","newton",
#          "cantor","takagi","sierpinski","feigenbaum"

WIDTH, HEIGHT = 600, 600
MAX_ITERS    = 200
ZOOM_SCALE   = 0.94
MIN_CONTRAST = 3.0

# Starting window
x_center, y_center = -0.743643887037151, 0.131825904205330
x_width = 2.8
# ——————————————————————————————————————————————

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f"Using device: {device}")

frames = []
extents = []
click_xy = None   # will hold (fract_x, fract_y) on click

ESCAPE_TIME = {"mandelbrot","julia","burning_ship","newton"}
STATIC      = {"cantor","takagi","sierpinski","feigenbaum"}

# Sobel kernels
_kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=device)
_ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=device)
kernel_x = _kx.view(1,1,3,3)
kernel_y = _ky.view(1,1,3,3)

def get_contrast_torch(data, sigma_frac=0.4):
    if isinstance(data, np.ndarray):
        t = torch.from_numpy(data).float().to(device)
    else:
        t = data.to(device).float()
    d = t.unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(d, kernel_x, padding=1)
    gy = F.conv2d(d, kernel_y, padding=1)
    contrast = torch.hypot(gx, gy).squeeze(0).squeeze(0)
    raw_map = contrast.cpu().numpy()
    raw_max = float(contrast.max().item())
    H, W = contrast.shape
    ys = torch.arange(H, device=device).view(H,1).float()
    xs = torch.arange(W, device=device).view(1,W).float()
    cy, cx = (H-1)/2, (W-1)/2
    sigma = sigma_frac * max(H, W)
    gauss = torch.exp(-((ys-cy)**2 + (xs-cx)**2)/(2*sigma*sigma))
    weighted = contrast * gauss
    idx = torch.argmax(weighted)
    return raw_max, raw_map, (int(idx//W), int(idx%W))

def generate_frame():
    global x_center, y_center, x_width, click_xy

    # 1) If the user clicked, steer there immediately
    if click_xy is not None:
        # 1a) apply the click‐zoom
        x_center, y_center = click_xy
        x_width *= ZOOM_SCALE
        click_xy = None   # clear for next time

        # 1b) compute the new window
        y_height = x_width * HEIGHT / WIDTH
        x_min = x_center - x_width/2
        x_max = x_center + x_width/2
        y_min = y_center - y_height/2
        y_max = y_center + y_height/2

        # 1c) generate and record the frame
        grid = _dispatch_fractal(x_min, x_max, y_min, y_max)
        frames.append(grid)
        extents.append((x_min, x_max, y_min, y_max))
        return

    # 2) No click: do normal contrast‐driven zoom for escape‐time, or static render
    # 2a) compute current window
    y_height = x_width * HEIGHT / WIDTH
    x_min = x_center - x_width/2
    x_max = x_center + x_width/2
    y_min = y_center - y_height/2
    y_max = y_center + y_height/2

    # 2b) generate the fractal grid
    grid = _dispatch_fractal(x_min, x_max, y_min, y_max)

    # 2c) handle escape‐time vs static
    if FRACTAL in ESCAPE_TIME:
        raw_max, _, (i, j) = get_contrast_torch(grid)
        if raw_max >= MIN_CONTRAST:
            # recenter based on contrast peak (for next call)
            x_center = x_min + (x_max - x_min) * (j / WIDTH)
            y_center = y_min + (y_max - y_min) * (i / HEIGHT)
            x_width  *= ZOOM_SCALE

            frames.append(grid)
            extents.append((x_min, x_max, y_min, y_max))
    else:
        # static fractals: one shot
        frames.append(grid)
        extents.append((x_min, x_max, y_min, y_max))


def _dispatch_fractal(x_min, x_max, y_min, y_max):
    """Helper to pick and run the right generator given window bounds."""
    if   FRACTAL == "mandelbrot":
        return generate_mandelbrot(HEIGHT, WIDTH, MAX_ITERS, x_min, x_max, y_min, y_max, device)
    elif FRACTAL == "julia":
        return generate_julia(HEIGHT, WIDTH, MAX_ITERS, x_min, x_max, y_min, y_max, device)
    elif FRACTAL == "burning_ship":
        return generate_burning_ship(HEIGHT, WIDTH, MAX_ITERS, x_min, x_max, y_min, y_max, device)
    elif FRACTAL == "newton":
        return generate_newton(HEIGHT, WIDTH, MAX_ITERS, x_min, x_max, y_min, y_max, device)
    elif FRACTAL == "cantor":
        return generate_cantor_set(HEIGHT, WIDTH)
    elif FRACTAL == "takagi":
        return generate_takagi_curve(HEIGHT, WIDTH)
    elif FRACTAL == "sierpinski":
        return generate_sierpinski_triangle(HEIGHT, WIDTH)
    elif FRACTAL == "feigenbaum":
        return generate_feigenbaum_attractor(HEIGHT, WIDTH)
    else:
        raise ValueError(f"Unsupported FRACTAL: {FRACTAL}")


# bootstrap
if FRACTAL in ESCAPE_TIME:
    generate_frame(); generate_frame()
else:
    generate_frame()

# setup plot
fig, ax = plt.subplots(figsize=(6,6))
img = ax.imshow(
    frames[0],
    cmap='inferno',
    extent=extents[0],
    interpolation='bilinear',
    origin='lower'
)

ax.axis('off')
mode = "Zoom" if FRACTAL in ESCAPE_TIME else "Static"
fig.suptitle(f"{FRACTAL.title()} {mode} — click to steer", fontsize=12)

# handle mouse clicks
def on_click(event):
    global click_xy
    if event.inaxes is not ax: 
        return
    # event.xdata, event.ydata are in fractal coords because of extent
    click_xy = (event.xdata, event.ydata)

fig.canvas.mpl_connect('button_press_event', on_click)

# animation for escape-time fractals
if FRACTAL in ESCAPE_TIME:
    def update(idx):
        # ensure enough frames
        while idx >= len(frames):
            generate_frame()
        img.set_data(frames[idx])
        img.set_extent(extents[idx])
        return [img]

    ani = FuncAnimation(fig, update, frames=200, interval=30, blit=True)

plt.show()
