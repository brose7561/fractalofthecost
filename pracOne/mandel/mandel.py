import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = pick_device()
print(f"Using device: {device}")

def coords_for_resolution(xmin, xmax, ymin, ymax, pixels):
    width = xmax - xmin
    height = ymax - ymin
    step_x = width / pixels
    step_y = height / pixels
    return min(step_x, step_y)

PRESETS = {
    # Mandelbrot presets
    "mandel_full": {
        "fractal": "mandelbrot",
        "coords": (-2.0, 1.0, -1.3, 1.3),
        "pixels": 2500,
        "max_iters": 400,
        "julia_c": None
    },
    "seahorse": {
        "fractal": "mandelbrot",
        "coords": (-0.8, -0.6, 0.0, 0.2),
        "pixels": 2500,
        "max_iters": 400,
        "julia_c": None
    },
    "elephant": {
        "fractal": "mandelbrot",
        "coords": (0.25, 0.35, -0.65, -0.55),
        "pixels": 2500,
        "max_iters": 400,
        "julia_c": None
    },
    "spiral": {
        "fractal": "mandelbrot",
        "coords": (-0.75, -0.74, 0.1, 0.11),
        "pixels": 2500,
        "max_iters": 400,
        "julia_c": None
    },
    "triple_spiral": {
        "fractal": "mandelbrot",
        "coords": (-0.17, -0.15, 1.04, 1.06),
        "pixels": 2500,
        "max_iters": 600,
        "julia_c": None
    },
    # Julia presets
    "julia_full": {
        "fractal": "julia",
        "coords": (-1.5, 1.5, -1.5, 1.5),
        "pixels": 2500,
        "max_iters": 400,
        "julia_c": complex(-0.8, 0.156)
    },
    "julia_classic": {
        "fractal": "julia",
        "coords": (-1.5, 1.5, -1.5, 1.5),
        "pixels": 2500,
        "max_iters": 400,
        "julia_c": complex(-0.70176, -0.3842)
    },
    "julia_flower": {
        "fractal": "julia",
        "coords": (-1.5, 1.5, -1.5, 1.5),
        "pixels": 2500,
        "max_iters": 400,
        "julia_c": complex(0.285, 0.01)
    },
    "julia_dust": {
        "fractal": "julia",
        "coords": (-1.5, 1.5, -1.5, 1.5),
        "pixels": 2500,
        "max_iters": 400,
        "julia_c": complex(-0.4, 0.6)
    },
    "julia_seahorse": {
        "fractal": "julia",
        "coords": (-1.5, 1.5, -1.5, 1.5),
        "pixels": 2500,
        "max_iters": 400,
        "julia_c": complex(-0.75, 0.11)
    }
}

def processFractal(a):
    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([
        10 + 20 * np.cos(a_cyclic),
        30 + 50 * np.sin(a_cyclic),
        155 - 80 * np.cos(a_cyclic)], 2)
    img[a == a.max()] = 0
    return np.uint8(np.clip(img, 0, 255))

def make_grid(xmin, xmax, ymin, ymax, step):
    Y, X = np.mgrid[ymin:ymax:step, xmin:xmax:step]
    return X, Y

@torch.no_grad()
def mandelbrot_torch(X, Y, max_iters):
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)
    z = torch.complex(x, y)
    zs = z.clone()
    ns = torch.zeros_like(z, dtype=torch.float32)
    z, zs, ns = z.to(device), zs.to(device), ns.to(device)
    for _ in range(max_iters):
        zs = zs * zs + z
        not_diverged = torch.abs(zs) < 4.0
        ns += not_diverged.float()
    return ns.cpu().numpy()

@torch.no_grad()
def julia_torch(X, Y, c, max_iters):
    x = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(Y, dtype=torch.float32)
    zs = torch.complex(x, y)
    c_t = torch.full_like(zs, fill_value=complex(c.real, c.imag))
    ns = torch.zeros_like(zs, dtype=torch.float32)
    zs, c_t, ns = zs.to(device), c_t.to(device), ns.to(device)
    for _ in range(max_iters):
        zs = zs * zs + c_t
        not_diverged = torch.abs(zs) < 4.0
        ns += not_diverged.float()
    return ns.cpu().numpy()

def compute_fractal(name):
    preset = PRESETS[name]
    xmin, xmax, ymin, ymax = preset["coords"]
    step = coords_for_resolution(xmin, xmax, ymin, ymax, preset["pixels"])
    X, Y = make_grid(xmin, xmax, ymin, ymax, step)
    print(f"Computing {name}: {X.shape}, step={step}, iters={preset['max_iters']}, fractal={preset['fractal']}")
    t0 = time.time()
    if preset["fractal"] == "julia":
        ns = julia_torch(X, Y, preset["julia_c"], preset["max_iters"])
    else:
        ns = mandelbrot_torch(X, Y, preset["max_iters"])
    t1 = time.time()
    print(f"Done in {t1 - t0:.2f}s")
    return processFractal(ns), (xmin, xmax, ymin, ymax)

# --- GUI setup ---
fig, ax = plt.subplots()
try:
    plt.get_current_fig_manager().full_screen_toggle()
except:
    pass
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
ax.set_position([0, 0, 1, 1])
ax.axis("off")

# Show startup loader text
loading_text = ax.text(0.5, 0.5, "Loading initial fractal...", ha='center', va='center', fontsize=20, color='orange')
fig.canvas.draw()
plt.pause(0.001)

# Compute initial fractal
initial = "mandel_full"
img, extent = compute_fractal(initial)
loading_text.remove()
img_disp = ax.imshow(img, origin="lower", extent=extent)

# Separate presets into two columns
mandel_presets = [k for k, v in PRESETS.items() if v["fractal"] == "mandelbrot"]
julia_presets = [k for k, v in PRESETS.items() if v["fractal"] == "julia"]

buttons = {}

def make_callback(preset_name, btn_ax_ref, button_ref):
    def callback(event):
        # Indicate loading
        btn_ax_ref.set_facecolor("orange")
        button_ref.label.set_text("Loading…")
        fig.canvas.draw()
        plt.pause(0.001)  # allow UI to update

        # Compute fractal
        new_img, new_extent = compute_fractal(preset_name)
        img_disp.set_data(new_img)
        img_disp.set_extent(new_extent)

        # Restore button state
        btn_ax_ref.set_facecolor("lightgrey")
        button_ref.label.set_text(preset_name)
        fig.canvas.draw_idle()
    return callback

# Create left column for Mandelbrot
for i, name in enumerate(mandel_presets):
    btn_ax = plt.axes([0.01, 0.95 - i*0.05, 0.12, 0.04])
    btn_ax.set_facecolor("lightgrey")
    button = Button(btn_ax, name, hovercolor="0.85")
    button.on_clicked(make_callback(name, btn_ax, button))
    buttons[name] = button

# Create right column for Julia
for i, name in enumerate(julia_presets):
    btn_ax = plt.axes([0.87, 0.95 - i*0.05, 0.12, 0.04])
    btn_ax.set_facecolor("lightgrey")
    button = Button(btn_ax, name, hovercolor="0.85")
    button.on_clicked(make_callback(name, btn_ax, button))
    buttons[name] = button

plt.show()
