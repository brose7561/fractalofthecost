import torch
import numpy as np
import matplotlib.pyplot as plt

# Device setup
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"[✅] Running on: {device}")

def compute_mandelbrot(settings):
    """Return a plottable RGB image of the Mandelbrot set."""
    zoom = settings["zoom"]
    cx, cy = settings["center_x"], settings["center_y"]
    max_iter = settings["max_iter"]
    device = settings["device"]
    img_width = settings["img_width"]
    img_height = settings["img_height"]
    base_width = settings["base_width"]
    base_height = settings["base_height"]

    # Calculate view window size from zoom
    view_width = base_width / zoom
    view_height = base_height / zoom

    print(f"[ℹ️] Viewport: center=({cx:.4f}, {cy:.4f}), zoom={zoom:.2f}")
    print(f"[ℹ️] Generating grid of size {img_width}x{img_height}")

    try:
        x_vals = np.linspace(cx - view_width / 2, cx + view_width / 2, img_width)
        y_vals = np.linspace(cy - view_height / 2, cy + view_height / 2, img_height)
        X, Y = np.meshgrid(x_vals, y_vals)  # shape (H, W)
        print(f"[📐] Grid shape: {X.shape}")
    except Exception as e:
        raise RuntimeError(f"[❌] Grid creation failed: {e}")

    try:
        x = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(Y, dtype=torch.float32)
        z = torch.complex(x, y).to(dtype=torch.complex64)
    except Exception as e:
        raise RuntimeError(f"[❌] Tensor conversion error: {e}")

    zs = z.clone()
    ns = torch.zeros_like(z, dtype=torch.float32)

    try:
        z = z.to(device)
        zs = zs.to(device)
        ns = ns.to(device)
    except Exception as e:
        raise RuntimeError(f"[❌] Failed to move tensors to device '{device}': {e}")

    print(f"[⚙️] Computing Mandelbrot with max_iter={max_iter}...")

    for _ in range(max_iter):
        zs_ = zs * zs + z
        not_diverged = torch.abs(zs_) < 4.0
        ns += not_diverged
        zs = zs_

    ns_np = ns.cpu().numpy()
    print(f"[✅] Computation complete.")
    print(f"[🧪] Divergence map range: min={ns_np.min()}, max={ns_np.max()}")
    return render_fractal(ns_np)

def render_fractal(a):
    """Convert Mandelbrot divergence data into a colorful RGB image."""
    if a.max() == a.min():
        print("[⚠️] Flat image — all values are", a.max())
        a = a * 0  # force flat color

    a_cyclic = (6.28 * a / 20.0).reshape(list(a.shape) + [1])
    img = np.concatenate([
        10 + 20 * np.cos(a_cyclic),
        30 + 50 * np.sin(a_cyclic),
        155 - 80 * np.cos(a_cyclic)
    ], axis=2)

    if a.max() != a.min():
        img[a == a.max()] = 0

    img = np.real(img)
    return np.uint8(np.clip(img, 0, 255))


# ===========================
# 🚀 Test as standalone script
# ===========================
import time

if __name__ == "__main__":
    presets = [
        {
            "name": "Full Set",
            "center_x": -0.5,
            "center_y": 0.0,
            "zoom": 1.0
        },
        {
            "name": "Zoomed In (Main Bulb)",
            "center_x": -0.5,
            "center_y": 0.0,
            "zoom": 5.0
        },
        {
            "name": "Seahorse Valley",
            "center_x": -0.745,
            "center_y": 0.115,
            "zoom": 40.0
        },
        {
            "name": "Elephant Valley",
            "center_x": 0.282,
            "center_y": 0.01,
            "zoom": 100.0
        },
        {
            "name": "Tiny Curl",
            "center_x": -0.743643887037151,
            "center_y": 0.13182590420533,
            "zoom": 1500.0
        }
    ]

    for preset in presets:
        settings = {
            "center_x": preset["center_x"],
            "center_y": preset["center_y"],
            "zoom": preset["zoom"],
            "base_width": 3.0,
            "base_height": 2.6,
            "img_width": 800,
            "img_height": 600,
            "max_iter": 500,
            "device": device
        }

        print(f"\n🎯 Showing: {preset['name']}")
        img = compute_mandelbrot(settings)
        plt.imshow(img)
        plt.title(preset['name'])
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.pause(2.0)  # Show for 2 seconds

    print("\n✅ All presets complete.")
    plt.close()

