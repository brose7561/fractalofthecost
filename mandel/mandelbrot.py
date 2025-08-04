import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.ndimage
from mpmath import mp, mpf
from tqdm import tqdm  # pip install tqdm

# ─── Settings ───────────────────────────────────────────────
mp.dps = 50  # High precision zoom

WIDTH = HEIGHT = 500  
MAX_ITERS = 300
PRELOAD_FRAMES = 1000
FPS = 60
INTERVAL = 1000 // FPS

ZOOM_RATE = mpf("0.95")
PAN_RATE = mpf("0.1")
CENTER_TOLERANCE = 0.05
MIN_CONTRAST = 5.0
CONTRAST_TOL = 0.5
STABILITY_THRESHOLD = 5

X_CENTER = mpf("-0.743643887037151")
Y_CENTER = mpf("0.131825904205330")
X_WIDTH = mpf("3.5")
# ────────────────────────────────────────────────────────────

frames = []
extents = []
x_center = X_CENTER
y_center = Y_CENTER
x_width = X_WIDTH
stability_counter = 0
previous_contrast_score = 0

def mandelbrot(h, w, maxit, x_min, x_max, y_min, y_max):
    y, x = np.ogrid[float(y_min):float(y_max):h*1j, float(x_min):float(x_max):w*1j]
    c = x + y * 1j
    z = np.zeros_like(c)
    divtime = np.full(c.shape, maxit, dtype=int)
    for i in range(maxit):
        z = z**2 + c
        diverge = np.abs(z) > 2
        div_now = diverge & (divtime == maxit)
        divtime[div_now] = i
        z[diverge] = 2
    return divtime

def get_contrast(data):
    sx = scipy.ndimage.sobel(data, axis=1)
    sy = scipy.ndimage.sobel(data, axis=0)
    sobel = np.hypot(sx, sy)
    entropy = np.abs(scipy.ndimage.gaussian_laplace(data, sigma=1.0))
    yy, xx = np.meshgrid(np.linspace(-1, 1, data.shape[0]), np.linspace(-1, 1, data.shape[1]), indexing='ij')
    radial = 1 - np.exp(-(xx**2 + yy**2) / 0.3)
    interesting = 0.7 * sobel + 0.3 * entropy
    return np.max(interesting), interesting * radial

def generate_frame():
    global x_center, y_center, x_width
    global previous_contrast_score, stability_counter

    y_height = x_width * HEIGHT / WIDTH
    x_min = x_center - x_width / 2
    x_max = x_center + x_width / 2
    y_min = y_center - y_height / 2
    y_max = y_center + y_height / 2

    data = mandelbrot(HEIGHT, WIDTH, MAX_ITERS, x_min, x_max, y_min, y_max)
    contrast_score, contrast_map = get_contrast(data)

    if contrast_score < MIN_CONTRAST:
        return None, None

    mask = contrast_map > np.percentile(contrast_map, 99.5)
    if not np.any(mask):
        return None, None

    y_idx, x_idx = np.nonzero(mask)
    weights = contrast_map[y_idx, x_idx]
    cx = np.average(x_idx, weights=weights)
    cy = np.average(y_idx, weights=weights)

    target_x = x_min + (x_max - x_min) * mpf(cx) / WIDTH
    target_y = y_min + (y_max - y_min) * mpf(cy) / HEIGHT

    dx = target_x - x_center
    dy = target_y - y_center

    if abs(dx) > CENTER_TOLERANCE * x_width or abs(dy) > CENTER_TOLERANCE * x_width:
        x_center += dx * PAN_RATE
        y_center += dy * PAN_RATE
    else:
        if abs(contrast_score - previous_contrast_score) < CONTRAST_TOL:
            stability_counter += 1
            if stability_counter > STABILITY_THRESHOLD:
                return None, None
        else:
            stability_counter = 0
        x_center = target_x
        y_center = target_y
        x_width *= ZOOM_RATE
        previous_contrast_score = contrast_score

    return data, (float(x_min), float(x_max), float(y_min), float(y_max))


# ─── Preload Phase ───────────────────────────────────────────
print(f" Preloading {PRELOAD_FRAMES} frames...")
for _ in tqdm(range(PRELOAD_FRAMES)):
    data, extent = generate_frame()
    if data is not None:
        frames.append(data)
        extents.append(extent)
    else:
        continue
print("Preloading done. Launching viewer...")

# ─── Viewer ─────────────────────────────────────────────────
fig, ax = plt.subplots()
img = ax.imshow(frames[0], cmap='inferno', extent=extents[0])
ax.axis('off')
fig.suptitle("Mandelbrot Zoom – Fast Preloaded", fontsize=14)

def update(i):
    idx = i % len(frames)
    img.set_data(frames[idx])
    img.set_extent(extents[idx])
    return [img]

ani = FuncAnimation(fig, update, interval=INTERVAL, blit=True)
plt.show()
