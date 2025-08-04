import numpy as np
import torch
import matplotlib.pyplot as plt
from mandelbrot import compute_mandelbrot

# === View State ===
_view_settings = {
    "center_x": -0.5,
    "center_y": 0.0,
    "zoom": 1.0,
    "base_width": 3.0,
    "base_height": 2.6,
    "img_width": 2000,
    "img_height": 1800,
    "max_iter": 200,
    "device": torch.device("mps" if torch.backends.mps.is_available() else "cpu")
}

# === Frame Buffer ===
_frame_cache = {}

def set_view_setting(key, value):
    _view_settings[key] = value

def get_view_setting(key):
    return _view_settings[key]

def _key():
    """Unique cache key for current view settings"""
    return (round(_view_settings['center_x'], 5),
            round(_view_settings['center_y'], 5),
            round(np.log10(_view_settings['zoom']), 5),
            _view_settings['img_width'],
            _view_settings['img_height'],
            _view_settings['max_iter'])

def get_frame():
    """Return cached image or compute new one if needed."""
    k = _key()
    if k not in _frame_cache:
        _frame_cache[k] = compute_mandelbrot(_view_settings)
    return _frame_cache[k]


def interpolate_frames(frame1, frame2, alpha):
    return (1 - alpha) * frame1 + alpha * frame2

def zoom_to(target_zoom, speed=300, keyframe_every=5):
    current_zoom = _view_settings["zoom"]
    diff = abs(np.log(target_zoom) - np.log(current_zoom))
    steps = max(1, int(diff * speed))

    # Precompute all keyframes only at coarse intervals
    keyframes = {}
    for i in range(0, steps + 1, keyframe_every):
        interp_zoom = current_zoom * ((target_zoom / current_zoom) ** (i / steps))
        _view_settings["zoom"] = interp_zoom
        keyframes[i] = get_frame()

    for i in range(steps + 1):
        interp_zoom = current_zoom * ((target_zoom / current_zoom) ** (i / steps))
        _view_settings["zoom"] = interp_zoom

        k = (i // keyframe_every) * keyframe_every
        k_next = min(k + keyframe_every, steps)

        if i == k or i == steps:
            frame = keyframes[k]
        else:
            alpha = (i - k) / (k_next - k)
            frame = interpolate_frames(keyframes[k], keyframes[k_next], alpha)

        yield frame




if __name__ == "__main__":
    
    render_stream(zoom_to(get_view_setting("zoom") * 5, speed=200, keyframe_every=10))

