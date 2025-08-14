import itertools
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# =========================================================
# Config
# =========================================================
WIDTH, HEIGHT = 800, 800
MAX_ITERS    = 200
ZOOM_SCALE   = 0.94          # per-frame zoom-in factor (0<ZOOM_SCALE<1)
MIN_CONTRAST = 3.0           # Sobel magnitude threshold to keep zooming
FADE_FRAMES  = 15            # frames of black between fractals
ALL_FRACTALS = ["mandelbrot", "julia", "burning_ship", "newton"]

# Start location for Mandelbrot/Newton; Julia ignores center except for view
INIT_CENTER  = (-0.743643887037151, 0.131825904205330)
INIT_WIDTH   = 2.8           # starting view width for regular zooming
SMALL_WIDTH  = 1.0 / WIDTH   # “dot” width for intro (≈ one pixel)
MIN_WIDTH    = 1e-6          # stop when width is smaller than this

# Julia parameter (nice classic)
JULIA_C = (-0.8, 0.156)

# Device (MPS if available)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =========================================================
# Utilities: Sobel-based contrast picker (fast, torch)
# =========================================================
_kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=device)
_ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=device)
kernel_x = _kx.view(1,1,3,3)
kernel_y = _ky.view(1,1,3,3)

def get_contrast_torch(data: np.ndarray, sigma_frac: float = 0.4):
    """
    Returns (raw_max_contrast, (row, col)) using Sobel + center weighting.
    """
    t = torch.from_numpy(data).to(device=device, dtype=torch.float32)
    d = t.unsqueeze(0).unsqueeze(0)                  # (1,1,H,W)
    gx = F.conv2d(d, kernel_x, padding=1)
    gy = F.conv2d(d, kernel_y, padding=1)
    c  = torch.hypot(gx, gy).squeeze(0).squeeze(0)   # (H,W)
    raw_max = float(c.max().item())

    H, W = c.shape
    ys = torch.arange(H, device=device).view(H,1).float()
    xs = torch.arange(W, device=device).view(1,W).float()
    cy, cx = (H-1)/2, (W-1)/2
    sigma = sigma_frac * max(H, W)
    gauss = torch.exp(-((ys-cy)**2 + (xs-cx)**2)/(2*sigma*sigma))
    weighted = c * gauss
    idx = int(torch.argmax(weighted))
    return raw_max, (idx // W, idx % W)

# =========================================================
# Fractal generators (torch, returns float32 numpy in [0,1])
# =========================================================
def _linspace2d(xmin, xmax, ymin, ymax, W, H, device):
    xs = torch.linspace(xmin, xmax, W, dtype=torch.float32, device=device)
    ys = torch.linspace(ymin, ymax, H, dtype=torch.float32, device=device)
    X, Y = torch.meshgrid(ys, xs, indexing="ij")  # Y=row, X=col; shape (H,W)
    return X, Y

def generate_mandelbrot(H, W, max_iters, xmin, xmax, ymin, ymax, device):
    Y, X = _linspace2d(xmin, xmax, ymin, ymax, W, H, device)  # careful ordering
    cr, ci = X, Y
    zr = torch.zeros_like(cr)
    zi = torch.zeros_like(ci)
    count = torch.zeros_like(cr, dtype=torch.int32)

    for _ in range(max_iters):
        # z = z^2 + c
        zr2 = zr*zr - zi*zi + cr
        zi2 = 2*zr*zi + ci
        zr, zi = zr2, zi2
        mag2 = zr*zr + zi*zi
        inside = mag2 <= 4.0
        count = count + inside.int()
        # stop updating escaped points by masking
        zr = zr * inside + zr * (~inside)
        zi = zi * inside + zi * (~inside)

    img = count.clamp(min=0).to(torch.float32) / float(max_iters)
    return img.detach().cpu().numpy()

def generate_julia(H, W, max_iters, xmin, xmax, ymin, ymax, device, c=JULIA_C):
    Y, X = _linspace2d(xmin, xmax, ymin, ymax, W, H, device)
    zr, zi = X, Y
    cr = torch.full_like(X, c[0])
    ci = torch.full_like(Y, c[1])
    count = torch.zeros_like(X, dtype=torch.int32)

    for _ in range(max_iters):
        zr2 = zr*zr - zi*zi + cr
        zi2 = 2*zr*zi + ci
        zr, zi = zr2, zi2
        mag2 = zr*zr + zi*zi
        inside = mag2 <= 4.0
        count = count + inside.int()
        zr = zr * inside + zr * (~inside)
        zi = zi * inside + zi * (~inside)

    img = count.to(torch.float32) / float(max_iters)
    return img.detach().cpu().numpy()

def generate_burning_ship(H, W, max_iters, xmin, xmax, ymin, ymax, device):
    Y, X = _linspace2d(xmin, xmax, ymin, ymax, W, H, device)
    cr, ci = X, Y
    zr = torch.zeros_like(cr)
    zi = torch.zeros_like(ci)
    count = torch.zeros_like(cr, dtype=torch.int32)

    for _ in range(max_iters):
        ar = zr.abs()
        ai = zi.abs()
        zr2 = ar*ar - ai*ai + cr
        zi2 = 2*ar*ai + ci
        zr, zi = zr2, zi2
        mag2 = zr*zr + zi*zi
        inside = mag2 <= 4.0
        count = count + inside.int()
        zr = zr * inside + zr * (~inside)
        zi = zi * inside + zi * (~inside)

    img = count.to(torch.float32) / float(max_iters)
    return img.detach().cpu().numpy()

def generate_newton(H, W, max_iters, xmin, xmax, ymin, ymax, device):
    # Newton on f(z) = z^3 - 1
    Y, X = _linspace2d(xmin, xmax, ymin, ymax, W, H, device)
    zr, zi = X, Y
    count = torch.zeros_like(X, dtype=torch.int32)

    eps = 1e-8
    for _ in range(max_iters):
        # f(z) = z^3 - 1
        zr2 = zr*zr - zi*zi
        zi2 = 2*zr*zi
        fr  = zr*(zr2 - 3*zi*zi) - 1.0                 # Re(z^3 - 1)
        fi  = zi*(3*zr*zr - zi*zi)                     # Im(z^3 - 1)

        # f'(z) = 3 z^2
        d_r = 3*(zr2 - zi*zi)                          # Re(3 z^2)
        d_i = 6*(zr*zi)                                # Im(3 z^2)

        denom = d_r*d_r + d_i*d_i + eps
        # delta = f / f'
        dr = (fr*d_r + fi*d_i) / denom
        di = (fi*d_r - fr*d_i) / denom

        zr = zr - dr
        zi = zi - di

        # convergence by |f(z)| < tol
        magf = torch.hypot(fr, fi)
        still = magf > 1e-6
        count = count + still.int()

    img = count.to(torch.float32) / float(max_iters)
    return img.detach().cpu().numpy()

# =========================================================
# Dispatch + Navigation
# =========================================================
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

# =========================================================
# Global state machine
# =========================================================
fractal_idx = -1             # so first switch_to_next() selects index 0
current_fractal = None
x_center, y_center = INIT_CENTER
x_width = INIT_WIDTH

intro_zooming = False        # when True: start from SMALL_WIDTH and zoom out to INIT_WIDTH
fade_in_progress = False
fading_counter = 0

click_xy = None
last_extent = (0, 1, 0, 1)
black_frame = np.zeros((HEIGHT, WIDTH), dtype=np.float32)

mode = "compute"             # "compute" then "playback"
remaining_fractals_to_compute = len(ALL_FRACTALS)

# store all frames for later playback (data + extent)
recorded_frames = []
recorded_extents = []
play_idx = 0                 # playback cursor

def start_intro_for(fractal_name):
    global current_fractal, x_center, y_center, x_width
    global intro_zooming
    current_fractal = fractal_name
    x_center, y_center = INIT_CENTER
    x_width = SMALL_WIDTH
    intro_zooming = True

def switch_to_next():
    global fractal_idx
    fractal_idx = (fractal_idx + 1) % len(ALL_FRACTALS)
    start_intro_for(ALL_FRACTALS[fractal_idx])

def append_frame(grid, ext):
    # Store every shown frame
    recorded_frames.append(grid.copy())
    recorded_extents.append(ext)

def next_frame_compute():
    """
    Compute-mode frame generator:
    - Handles fade to black
    - Handles intro zoom (dot emerging)
    - Bootstrap/normal frames with click override + contrast guidance
    - Switches to playback after last fractal finishes its fade
    """
    global intro_zooming, fade_in_progress, fading_counter
    global x_center, y_center, x_width, last_extent
    global remaining_fractals_to_compute, mode

    # 1) Fade-to-black phase
    if fade_in_progress:
        if fading_counter > 0:
            fading_counter -= 1
            return black_frame, last_extent
        # Fade finished. If more fractals to compute, start next; else go playback.
        fade_in_progress = False
        if remaining_fractals_to_compute > 0:
            switch_to_next()
            # Show a black frame right before dot emerges (nice theatrical beat)
            return black_frame, last_extent
        else:
            # No more fractures to compute → switch to playback
            mode = "playback"
            # Start playback at the beginning
            return recorded_frames[0], recorded_extents[0]

    # 2) Intro zoom: grow from dot (SMALL_WIDTH) up to INIT_WIDTH
    if intro_zooming:
        grid, ext = dispatch(current_fractal, x_center, y_center, x_width)
        last_extent = ext
        if x_width < INIT_WIDTH:
            x_width = min(x_width / ZOOM_SCALE, INIT_WIDTH)  # exponential growth
        else:
            intro_zooming = False
        return grid, ext

    # 3) Normal navigation frame
    grid, ext = dispatch(current_fractal, x_center, y_center, x_width)
    last_extent = ext

    # 3a) Click override (one-step steer)
    global click_xy
    if click_xy is not None and ext[0] <= click_xy[0] <= ext[1] and ext[2] <= click_xy[1] <= ext[3]:
        x_center, y_center = click_xy
        x_width *= ZOOM_SCALE
        click_xy = None
        return grid, ext

    # 3b) Contrast-guided zoom
    raw_max, (ri, rj) = get_contrast_torch(grid)
    if raw_max >= MIN_CONTRAST and x_width > MIN_WIDTH:
        xmin, xmax, ymin, ymax = ext
        x_center = xmin + (xmax - xmin) * (rj / WIDTH)
        y_center = ymin + (ymax - ymin) * (ri / HEIGHT)
        x_width *= ZOOM_SCALE
        return grid, ext

    # 4) Bottomed out → trigger fade and count this fractal as completed
    fade_in_progress = True
    fading_counter = FADE_FRAMES
    remaining_fractals_to_compute -= 1
    return black_frame, last_extent

def next_frame_playback():
    """Return next recorded frame in an infinite loop."""
    global play_idx
    if not recorded_frames:
        return black_frame, last_extent
    frame = recorded_frames[play_idx]
    ext   = recorded_extents[play_idx]
    play_idx = (play_idx + 1) % len(recorded_frames)
    return frame, ext

def next_frame():
    if mode == "compute":
        grid, ext = next_frame_compute()
        # Store everything shown during compute (including black and intros)
        append_frame(grid, ext)
        return grid, ext
    else:
        return next_frame_playback()

# =========================================================
# Matplotlib wiring
# =========================================================
fig, ax = plt.subplots(figsize=(6,6))
ax.axis("off")
fig.suptitle("Click to steer • Contrast-guided zoom • Auto-fade & cycle → playback loop", fontsize=11)

# initialize: start first fractal immediately as a dot emerging
switch_to_next()
grid0, ext0 = next_frame()

img = ax.imshow(
    grid0,
    cmap="inferno",
    extent=ext0,
    interpolation="bilinear",
    origin="lower",
)

def on_click(event):
    global click_xy
    if event.inaxes != ax: 
        return
    if event.xdata is None or event.ydata is None:
        return
    click_xy = (event.xdata, event.ydata)

fig.canvas.mpl_connect("button_press_event", on_click)

def update(_):
    grid, ext = next_frame()
    img.set_data(grid)
    img.set_extent(ext)
    return [img]

ani = FuncAnimation(
    fig, update,
    frames=itertools.count(),
    interval=30,   # ms
    blit=True
)

plt.show()
