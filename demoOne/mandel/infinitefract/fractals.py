import torch
import numpy as np
from mpmath import mp, mpf

# Default device selection for PyTorch-based generators
default_device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

# -------------------- Existing Generators --------------------

def generate_mandelbrot(h, w, maxit, x_min, x_max, y_min, y_max, device=default_device):
    xs = torch.linspace(x_min, x_max, w, device=device)
    ys = torch.linspace(y_min, y_max, h, device=device)
    xx, yy = torch.meshgrid(xs, ys, indexing='xy')
    zr = torch.zeros_like(xx)
    zi = torch.zeros_like(yy)
    cr, ci = xx, yy
    divtime = torch.full((h, w), maxit, dtype=torch.int32, device=device)
    for i in range(maxit):
        zr2 = zr*zr
        zi2 = zi*zi
        mag2 = zr2 + zi2
        mask = mag2 <= 4.0
        new_zr = zr2 - zi2 + cr
        new_zi = 2*zr*zi + ci
        zr = torch.where(mask, new_zr, zr)
        zi = torch.where(mask, new_zi, zi)
        diverged = (mag2 > 4.0) & (divtime == maxit)
        divtime[diverged] = i
    return divtime.float().cpu().numpy()


def generate_julia(h, w, maxit, x_min, x_max, y_min, y_max, device=default_device, c=-0.8+0.156j):
    xs = torch.linspace(x_min, x_max, w, device=device)
    ys = torch.linspace(y_min, y_max, h, device=device)
    xx, yy = torch.meshgrid(xs, ys, indexing='xy')
    zr = xx
    zi = yy
    cr = torch.full_like(zr, c.real)
    ci = torch.full_like(zi, c.imag)
    divtime = torch.full((h, w), maxit, dtype=torch.int32, device=device)
    for i in range(maxit):
        zr2 = zr*zr
        zi2 = zi*zi
        mag2 = zr2 + zi2
        mask = mag2 <= 4.0
        new_zr = zr2 - zi2 + cr
        new_zi = 2*zr*zi + ci
        zr = torch.where(mask, new_zr, zr)
        zi = torch.where(mask, new_zi, zi)
        diverged = (mag2 > 4.0) & (divtime == maxit)
        divtime[diverged] = i
    return divtime.float().cpu().numpy()


def generate_burning_ship(h, w, maxit, x_min, x_max, y_min, y_max, device=default_device):
    xs = torch.linspace(x_min, x_max, w, device=device)
    ys = torch.linspace(y_min, y_max, h, device=device)
    xx, yy = torch.meshgrid(xs, ys, indexing='xy')
    zr = torch.zeros_like(xx)
    zi = torch.zeros_like(yy)
    cr, ci = xx, yy
    divtime = torch.full((h, w), maxit, dtype=torch.int32, device=device)
    for i in range(maxit):
        zr = zr.abs()
        zi = zi.abs()
        zr2 = zr*zr
        zi2 = zi*zi
        mag2 = zr2 + zi2
        mask = mag2 <= 4.0
        new_zr = zr2 - zi2 + cr
        new_zi = 2*zr*zi + ci
        zr = torch.where(mask, new_zr, zr)
        zi = torch.where(mask, new_zi, zi)
        diverged = (mag2 > 4.0) & (divtime == maxit)
        divtime[diverged] = i
    return divtime.float().cpu().numpy()


def generate_newton(h, w, maxit, x_min, x_max, y_min, y_max, device=default_device):
    xs = torch.linspace(x_min, x_max, w, device=device)
    ys = torch.linspace(y_min, y_max, h, device=device)
    zr = xs.view(1, w).repeat(h, 1)
    zi = ys.view(h, 1).repeat(1, w)
    divtime = torch.zeros((h, w), dtype=torch.int32, device=device)
    tol = 1e-6
    for i in range(maxit):
        z = zr + 1j*zi
        f = z**3 - 1
        df = 3*z**2
        z_next = z - f/df
        mask = (torch.abs(z_next - z) > tol)
        divtime[mask] = i
        zr = z_next.real
        zi = z_next.imag
    return divtime.float().cpu().numpy()

# -------------------- New Generators (every ~3rd interesting fractal) --------------------

def generate_cantor_set(h, w, maxit=10, **kwargs):
    """
    Classic 1D Cantor set rasterized into a 2D image: columns in the set are white.
    maxit: number of removal iterations.
    """
    img = np.zeros((h, w), dtype=np.float32)
    for x in range(w):
        pos = x / float(w-1)
        in_set = True
        for _ in range(maxit):
            pos3 = pos * 3
            if 1 <= pos3 <= 2:
                in_set = False
                break
            pos = pos3 % 1
        if in_set:
            img[:, x] = 1.0
    return img


def generate_takagi_curve(h, w, maxit=10, **kwargs):
    """
    Blancmange (Takagi) curve: f(x) = sum_{n=0..maxit-1} 2^-n * triangle_wave(2^n x)
    Rastered as a thin white curve on black background.
    """
    xs = np.linspace(0, 1, w)
    def triangle_wave(x): return 2 * np.abs(x - np.floor(x + 0.5))
    f = np.zeros(w, dtype=np.float32)
    for n in range(maxit):
        f += (0.5**n) * triangle_wave((2**n) * xs)
    # normalize
    f = (f - f.min()) / f.ptp()
    img = np.zeros((h, w), dtype=np.float32)
    ys = np.round((1 - f) * (h-1)).astype(int)
    img[ys, np.arange(w)] = 1.0
    return img


def generate_sierpinski_triangle(h, w, maxit=100000, **kwargs):
    """
    Chaos-game Sierpinski triangle: plot maxit points.
    """
    img = np.zeros((h, w), dtype=np.float32)
    pts = np.array([[0, 0], [w-1, 0], [w/2, h-1]], dtype=np.float32)
    pt = np.random.rand(2) * [w-1, h-1]
    for _ in range(maxit):
        vertex = pts[np.random.randint(0, 3)]
        pt = (pt + vertex) / 2
        img[int(pt[1]), int(pt[0])] = 1.0
    return img


def generate_feigenbaum_attractor(h, w, maxit=50000, r=3.570, **kwargs):
    """
    Feigenbaum attractor via logistic map (x_{n+1} = r x_n (1-x_n)).
    Plots (x_n, x_{n+1}) density on a 2D grid.
    """
    x = 0.5
    arr = np.zeros((h, w), dtype=np.float32)
    # warm-up to reach attractor
    for _ in range(1000):
        x = r * x * (1 - x)
    # sample points
    for _ in range(maxit):
        x = r * x * (1 - x)
        y = r * x * (1 - x)
        xi = int(x * (w-1))
        yi = int(y * (h-1))
        arr[yi, xi] += 1
    # normalize
    if arr.max() > 0:
        arr /= arr.max()
    return arr
