import itertools
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
)

# ——————————————————————————————————————————————
# Configurable parameters
# ——————————————————————————————————————————————
WIDTH, HEIGHT = 800, 800
MAX_ITERS    = 200
ZOOM_SCALE   = 0.94
MIN_CONTRAST = 3.0

# Initial view parameters
INIT_CENTER = (-0.743643887037151,  0.131825904205330)
INIT_WIDTH  = 2.8

# Fractal list and fade length
ALL_FRACTALS = ["mandelbrot", "julia", "burning_ship", "newton"]
FADE_FRAMES  = 15

# “Dot” start width (one pixel)
SMALL_WIDTH = 1.0 / WIDTH

# Stop zoom‐in below this width
MIN_WIDTH = 1e-6

# Device
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print("Using device:", device)

# Precompute Sobel kernels
_kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], device=device, dtype=torch.float32)
_ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], device=device, dtype=torch.float32)
kernel_x = _kx.view(1,1,3,3)
kernel_y = _ky.view(1,1,3,3)

# ——————————————————————————————————————————————
# State variables
# ——————————————————————————————————————————————
fractal_idx       = 0
current_fractal   = ALL_FRACTALS[fractal_idx]
x_center, y_center = INIT_CENTER
x_width           = INIT_WIDTH

# Bootstrap, fade, and intro flags/counters
bootstrap_left    = 2
fade_in_progress  = False
fading_counter    = 0
intro_zooming     = False

click_xy          = None
last_extent       = (0,1,0,1)
black_frame       = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

# ——————————————————————————————————————————————
# Helper functions
# ——————————————————————————————————————————————
def get_contrast_torch(data, sigma_frac=0.4):
    t = torch.from_numpy(data).float().to(device) if isinstance(data, np.ndarray) else data.to(device).float()
    d = t.unsqueeze(0).unsqueeze(0)
    gx = F.conv2d(d, kernel_x, padding=1)
    gy = F.conv2d(d, kernel_y, padding=1)
    c = torch.hypot(gx, gy).squeeze()
    raw_max = float(c.max().item())
    H, W = c.shape
    ys = torch.arange(H, device=device).view(H,1).float()
    xs = torch.arange(W, device=device).view(1,W).float()
    cy, cx = (H-1)/2, (W-1)/2
    sigma = sigma_frac * max(H, W)
    gauss = torch.exp(-((ys-cy)**2 + (xs-cx)**2)/(2*sigma*sigma))
    weighted = c * gauss
    idx = int(torch.argmax(weighted))
    return raw_max, (idx//W, idx%W)

def dispatch(fractal, x_c, y_c, w):
    y_h = w * HEIGHT / WIDTH
    xmin, xmax = x_c - w/2, x_c + w/2
    ymin, ymax = y_c - y_h/2, y_c + y_h/2
    if fractal == "mandelbrot":
        g = generate_mandelbrot(HEIGHT, WIDTH, MAX_ITERS, xmin, xmax, ymin, ymax, device)
    elif fractal == "julia":
        g = generate_julia(HEIGHT, WIDTH, MAX_ITERS, xmin, xmax, ymin, ymax, device)
    elif fractal == "burning_ship":
        g = generate_burning_ship(HEIGHT, WIDTH, MAX_ITERS, xmin, xmax, ymin, ymax, device)
    elif fractal == "newton":
        g = generate_newton(HEIGHT, WIDTH, MAX_ITERS, xmin, xmax, ymin, ymax, device)
    else:
        raise ValueError(f"Unknown fractal: {fractal}")
    return g, (xmin, xmax, ymin, ymax)

def switch_to_next():
    global fractal_idx, current_fractal
    global x_center, y_center, x_width
    global bootstrap_left, fade_in_progress, fading_counter, intro_zooming, click_xy

    fractal_idx = (fractal_idx + 1) % len(ALL_FRACTALS)
    current_fractal = ALL_FRACTALS[fractal_idx]
    x_center, y_center = INIT_CENTER
    x_width = SMALL_WIDTH         # start as a “dot”
    bootstrap_left = 2
    fade_in_progress = False
    fading_counter = 0
    intro_zooming = True
    click_xy = None

def next_frame():
    global x_center, y_center, x_width
    global bootstrap_left, fade_in_progress, fading_counter, intro_zooming
    global click_xy, last_extent

    # 1) Fade‐to‐black handling
    if fade_in_progress:
        if fading_counter > 0:
            fading_counter -= 1
            return black_frame, last_extent
        fade_in_progress = False
        switch_to_next()
        return next_frame()

    # 2) Intro zoom‐out sequence
    if intro_zooming:
        grid, ext = dispatch(current_fractal, x_center, y_center, x_width)
        last_extent = ext
        # grow window exponentially
        if x_width < INIT_WIDTH:
            x_width = min(x_width / ZOOM_SCALE, INIT_WIDTH)
        else:
            intro_zooming = False
        return grid, ext

    # 3) Bootstrap frames
    if bootstrap_left > 0:
        grid, ext = dispatch(current_fractal, x_center, y_center, x_width)
        last_extent = ext
        bootstrap_left -= 1
        return grid, ext

    # 4) Click override
    if click_xy is not None:
        x_center, y_center = click_xy
        x_width *= ZOOM_SCALE
        click_xy = None
        grid, ext = dispatch(current_fractal, x_center, y_center, x_width)
        last_extent = ext
        return grid, ext

    # 5) Contrast‐driven zoom
    grid, ext = dispatch(current_fractal, x_center, y_center, x_width)
    last_extent = ext
    raw_max, (i, j) = get_contrast_torch(grid)
    if raw_max >= MIN_CONTRAST and x_width > MIN_WIDTH:
        xmin, xmax, ymin, ymax = ext
        x_center = xmin + (xmax - xmin)*(j/WIDTH)
        y_center = ymin + (ymax - ymin)*(i/HEIGHT)
        x_width  *= ZOOM_SCALE
        return grid, ext

    # 6) End of zoom → start fade
    fade_in_progress = True
    fading_counter = FADE_FRAMES
    return black_frame, last_extent

# ——————————————————————————————————————————————
# Set up figure & start animation
# ——————————————————————————————————————————————
fig, ax = plt.subplots(figsize=(6,6))
# initialize first fractal
intro_zooming = True
bootstrap_left = 0
switch_to_next()  # sets up first fractal into intro mode
grid0, ext0 = next_frame()
img = ax.imshow(
    grid0,
    cmap='inferno',
    extent=ext0,
    interpolation='bilinear',
    origin='lower'
)
ax.axis('off')
fig.suptitle("Click to steer; cycling fractals with intro zoom…", fontsize=12)

def on_click(event):
    global click_xy
    if event.inaxes is not ax:
        return
    click_xy = (event.xdata, event.ydata)

fig.canvas.mpl_connect('button_press_event', on_click)

def update(_):
    grid, ext = next_frame()
    img.set_data(grid)
    img.set_extent(ext)
    return [img]

ani = FuncAnimation(
    fig, update,
    frames=itertools.count(),
    interval=30,
    blit=True
)

plt.show()
