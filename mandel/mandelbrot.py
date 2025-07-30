import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.ndimage
from mpmath import mp, mpf

# High-precision setup
mp.dps = 50

# Constants
WIDTH = HEIGHT = 300
MAX_ITERS = 100
ZOOM_SCALE = mpf("0.94")
MIN_CONTRAST = 10.0
CENTER_TOLERANCE = 0.1  # as a fraction of x_width
PAN_SCALE = mpf("0.2")  # how fast to pan toward center
MAX_STABILITY = 10      # stop zooming if no contrast gain for this many frames

# Initial center and zoom
x_center = mpf("-0.743643887037151")
y_center = mpf("0.131825904205330")
x_width = mpf("2.8")

frames = []
extents = []
previous_contrast_score = 0
stability_counter = 0

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
    contrast = np.hypot(sx, sy)
    return np.max(contrast), contrast

def generate_frame():
    global x_center, y_center, x_width
    global previous_contrast_score, stability_counter

    y_height = x_width * HEIGHT / WIDTH
    x_min = x_center - x_width / 2
    x_max = x_center + x_width / 2
    y_min = y_center - y_height / 2
    y_max = y_center + y_height / 2

    data = mandelbrot(HEIGHT, WIDTH, MAX_ITERS, x_min, x_max, y_min, y_max)
    contrast_score, contrast = get_contrast(data)

    if contrast_score < MIN_CONTRAST:
        print("Low contrast, skipping frame")
        return

    threshold = np.percentile(contrast, 99.5)
    mask = contrast > threshold
    if not np.any(mask):
        print("No significant contrast zone found")
        return

    y_idx, x_idx = np.nonzero(mask)
    weights = contrast[y_idx, x_idx]
    cx = np.average(x_idx, weights=weights)
    cy = np.average(y_idx, weights=weights)

    contrast_x = x_min + (x_max - x_min) * mpf(cx) / mpf(WIDTH)
    contrast_y = y_min + (y_max - y_min) * mpf(cy) / mpf(HEIGHT)

    dx = contrast_x - x_center
    dy = contrast_y - y_center

    if abs(dx) > CENTER_TOLERANCE * x_width or abs(dy) > CENTER_TOLERANCE * x_width:
        x_center += dx * PAN_SCALE
        y_center += dy * PAN_SCALE
        print("Panning toward contrast zone")
    else:
        if abs(contrast_score - previous_contrast_score) < 0.5:
            stability_counter += 1
            if stability_counter > MAX_STABILITY:
                print("No improvement in contrast — holding zoom")
                return
        else:
            stability_counter = 0

        x_center = contrast_x
        y_center = contrast_y
        x_width *= ZOOM_SCALE
        previous_contrast_score = contrast_score
        print("Zooming in")

    frames.append(data)
    extents.append((float(x_min), float(x_max), float(y_min), float(y_max)))

# Bootstrap with at least one frame
generate_frame()
if not frames:
    raise RuntimeError("Failed to generate initial frame. Check contrast threshold or bounds.")
generate_frame()

# Plot setup
fig, ax = plt.subplots()
img = ax.imshow(frames[0], cmap='inferno', extent=extents[0])
ax.axis('off')
fig.suptitle("Mandelbrot Zoom – Stabilized Contrast Tracking", fontsize=12)

def update(i):
    idx = i
    if idx >= len(frames) - 1:
        generate_frame()

    if idx < len(frames):
        img.set_data(frames[idx])
        img.set_extent(extents[idx])
    return [img]

ani = FuncAnimation(fig, update, interval=30, blit=True)
plt.show()
