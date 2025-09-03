
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.ndimage
from mpmath import mp, mpf

# High-precision setup
mp.dps = 50

# Constants
WIDTH = HEIGHT = 600
MAX_ITERS = 100
ZOOM_SCALE = mpf("0.94")
MIN_CONTRAST = 3.0

# Initial center and zoom
x_center = mpf("-0.743643887037151")
y_center = mpf("0.131825904205330")
x_width = mpf("2.8")

frames = []
extents = []

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

    # Current zoom window
    y_height = x_width * HEIGHT / WIDTH
    x_min = x_center - x_width / 2
    x_max = x_center + x_width / 2
    y_min = y_center - y_height / 2
    y_max = y_center + y_height / 2

    data = mandelbrot(HEIGHT, WIDTH, MAX_ITERS, x_min, x_max, y_min, y_max)
    contrast_score, contrast = get_contrast(data)

    if contrast_score >= MIN_CONTRAST:
        i, j = np.unravel_index(np.argmax(contrast), contrast.shape)
        i, j = int(i), int(j)
        x_center = x_min + (x_max - x_min) * mpf(j) / mpf(WIDTH)
        y_center = y_min + (y_max - y_min) * mpf(i) / mpf(HEIGHT)
        x_width *= ZOOM_SCALE
        frames.append(data)
        extents.append((float(x_min), float(x_max), float(y_min), float(y_max)))

# Bootstrap with two frames
generate_frame()
generate_frame()

# Plot setup
fig, ax = plt.subplots()
img = ax.imshow(frames[0], cmap='inferno', extent=extents[0])
ax.axis('off')
fig.suptitle("Mandelbrot Zoom – Working Contrast Tracker", fontsize=12)

def update(i):
    idx = i
    if idx >= len(frames) - 1:
        generate_frame()

    img.set_data(frames[idx])
    img.set_extent(extents[idx])
    return [img]

ani = FuncAnimation(fig, update, interval=30, blit=True)
plt.show()