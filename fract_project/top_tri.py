import math, random, time
import turtle as T
import tkinter as TK
import colorsys
import numpy as np
from PIL import Image, ImageTk
from collections import deque

try:
    import torch
    TORCH_OK = True
    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
except Exception:
    TORCH_OK = False
    DEVICE = None

SCREEN_W, SCREEN_H = 1200, 900
SIDE = 800.0
DOT_BASE = 3
START_SCALE = 0.001
SCALE_GROWTH = 1.022
BATCH_MIN = 400
BATCH_MAX = 9000

VIEW_START = START_SCALE
VIEW_MAX = 1.0

ZOOM_TARGET_SCALE = 0.06
ZOOM_STEP_FACTOR = 1.2
ZOOM_MAX_STEPS = 30

LAYER_GROWTH = 2.0
MAX_LAYERS = 10
SAT = 0.85
VAL = 1.0
HUE_RATE = 0.45
RESERVOIR_MAX = 50000
NEST_COPY_PER_FRAME = 5000

h = SIDE * math.sqrt(3) / 2.0
V0 = (0.0, 0.0)
V1 = (SIDE, 0.0)
V2 = (SIDE / 2.0, h)
BASE_VERTS = (V0, V1, V2)
CENTER = (SIDE / 2.0, h / 3.0)

def scale_verts(verts, s, center=CENTER):
    cx, cy = center
    return tuple((cx + (x - cx) * s, cy + (y - cy) * s) for x, y in verts)

def bounds_of(verts, margin=0.04):
    xs = [x for x, _ in verts]
    ys = [y for _, y in verts]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    dx, dy = xmax - xmin, ymax - ymin
    return (xmin - dx * margin, xmax + dx * margin, ymin - dy * margin, ymax + dy * margin)

def set_view_for(verts):
    xmin, xmax, ymin, ymax = bounds_of(verts)
    T.setworldcoordinates(xmin, ymin, xmax, ymax)
    return xmin, xmax, ymin, ymax

def choose_indices(n):
    if TORCH_OK:
        idx = torch.randint(0, 3, (n,), device=DEVICE)
        return idx.tolist()
    return [random.randrange(3) for _ in range(n)]

def hsv_rgb(h, s=SAT, v=VAL):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r*255), int(g*255), int(b*255)

def top_subtriangle(verts):
    v0, v1, v2 = verts
    a = ((v2[0] + v0[0]) * 0.5, (v2[1] + v0[1]) * 0.5)
    b = v2
    c = ((v2[0] + v1[0]) * 0.5, (v2[1] + v1[1]) * 0.5)
    return (a, b, c)

def affine_from_triangles(src, dst):
    sa, sb, sc = src
    da, db, dc = dst
    S = np.array([[sa[0], sb[0], sc[0]],
                  [sa[1], sb[1], sc[1]],
                  [1.0,  1.0,  1.0 ]], dtype=np.float64)
    Dx = np.array([da[0], db[0], dc[0]], dtype=np.float64)
    Dy = np.array([da[1], db[1], dc[1]], dtype=np.float64)
    coeff_x = np.linalg.solve(S, Dx)
    coeff_y = np.linalg.solve(S, Dy)
    M = np.array([[coeff_x[0], coeff_x[1]],
                  [coeff_y[0], coeff_y[1]]], dtype=np.float64)
    t = np.array([coeff_x[2], coeff_y[2]], dtype=np.float64)
    return M, t

def apply_affine(M, t, xs, ys):
    P = np.vstack((xs, ys))  # 2 x N
    Q = (M @ P) + t[:, None]
    return Q[0], Q[1]

def build_layers():
    layers = []
    s = START_SCALE
    base_hue = 0.55
    for i in range(MAX_LAYERS):
        verts = scale_verts(BASE_VERTS, s)
        cx = sum(v[0] for v in verts) / 3.0
        cy = sum(v[1] for v in verts) / 3.0
        dot = max(1, int(DOT_BASE * (1.0 - 0.7 * min(1.0, s))))
        weight = min(1.0, 0.15 + s)
        color = hsv_rgb(base_hue + i * 0.06)
        layer = {
            "s": s,
            "verts": verts,
            "x": cx,
            "y": cy,
            "dot": dot,
            "w": weight,
            "xs": deque(maxlen=RESERVOIR_MAX),
            "ys": deque(maxlen=RESERVOIR_MAX),
            "color": color,
            "M": None,
            "t": None
        }
        if i > 0:
            dst = top_subtriangle(verts)
            src = layers[-1]["verts"]
            M, t = affine_from_triangles(src, dst)
            layer["M"] = M
            layer["t"] = t
            x0, y0 = apply_affine(M, t, np.array([layers[-1]["x"]]), np.array([layers[-1]["y"]]))
            layer["x"] = float(x0[0])
            layer["y"] = float(y0[0])
        layers.append(layer)
        s *= LAYER_GROWTH
        if s >= VIEW_MAX:
            break
    return layers

def world_to_px(xs, ys, bounds):
    xmin, xmax, ymin, ymax = bounds
    sx = (SCREEN_W - 1) / (xmax - xmin)
    sy = (SCREEN_H - 1) / (ymax - ymin)
    px = ((xs - xmin) * sx).astype(np.int32)
    py = (SCREEN_H - 1 - (ys - ymin) * sy).astype(np.int32)
    return px, py

def sierpinski_turtle():
    screen = T.Screen()
    screen.setup(SCREEN_W, SCREEN_H)
    screen.screensize(SCREEN_W, SCREEN_H)
    screen.title("Sierpiński Triangle — Nested Top Embedding")

    T.tracer(0, 0)
    view = VIEW_START
    bounds = set_view_for(scale_verts(BASE_VERTS, view))

    canvas = screen.getcanvas()
    buf = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    pil_img = Image.fromarray(buf)
    tk_img = ImageTk.PhotoImage(pil_img)
    img_id = canvas.create_image(0, 0, anchor="nw", image=tk_img)

    zoom_request = {"go": False, "s": view}

    def fast_zoom():
        cur = zoom_request["s"]
        steps = 0
        while cur < ZOOM_TARGET_SCALE and steps < ZOOM_MAX_STEPS:
            cur = min(ZOOM_TARGET_SCALE, cur * ZOOM_STEP_FACTOR)
            nonlocal bounds
            bounds = set_view_for(scale_verts(BASE_VERTS, cur))
            steps += 1

    def on_zoom_click():
        zoom_request["go"] = True
        zoom_request["s"] = view
        fast_zoom()

    root = screen.getcanvas().winfo_toplevel()
    btn = TK.Button(root, text="Zoom to Centre", command=on_zoom_click)
    btn.place(relx=1.0, x=-20, y=20, anchor="ne")

    layers = build_layers()
    prev_view = view
    base_hue = 0.55

    while True:
        if zoom_request["go"]:
            zoom_request["go"] = False

        view = min(VIEW_MAX, view * SCALE_GROWTH)
        dv = max(0.0, view - prev_view)
        prev_view = view
        base_hue = (base_hue + HUE_RATE * dv) % 1.0
        bounds = set_view_for(scale_verts(BASE_VERTS, view))

        for L in layers:
            alpha = min(1.0, L["s"])
            batch = int((BATCH_MIN + (BATCH_MAX - BATCH_MIN) * alpha) * L["w"])
            idxs = choose_indices(batch)
            v0, v1, v2 = L["verts"]
            for k in idxs:
                if k == 0:
                    vx, vy = v0
                elif k == 1:
                    vx, vy = v1
                else:
                    vx, vy = v2
                L["x"] = 0.5 * (L["x"] + vx)
                L["y"] = 0.5 * (L["y"] + vy)
                L["xs"].append(L["x"])
                L["ys"].append(L["y"])

        for i in range(1, len(layers)):
            prev = layers[i-1]
            cur = layers[i]
            if cur["M"] is None:
                continue
            if len(prev["xs"]) == 0:
                continue
            xs = np.fromiter(prev["xs"], dtype=np.float64)
            ys = np.fromiter(prev["ys"], dtype=np.float64)
            if xs.size > NEST_COPY_PER_FRAME:
                xs = xs[-NEST_COPY_PER_FRAME:]
                ys = ys[-NEST_COPY_PER_FRAME:]
            tx, ty = apply_affine(cur["M"], cur["t"], xs, ys)
            for a, b in zip(tx, ty):
                cur["xs"].append(a)
                cur["ys"].append(b)

        buf[:] = 0
        for i, L in enumerate(layers):
            if not L["xs"]:
                continue
            xs = np.fromiter(L["xs"], dtype=np.float64)
            ys = np.fromiter(L["ys"], dtype=np.float64)
            px, py = world_to_px(xs, ys, bounds)
            mask = (px >= 0) & (px < SCREEN_W) & (py >= 0) & (py < SCREEN_H)
            px = px[mask]
            py = py[mask]
            color = hsv_rgb(base_hue + i * 0.06)
            if L["dot"] == 1:
                buf[py, px] = color
            else:
                r = L["dot"] // 2
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        px2 = px + dx
                        py2 = py + dy
                        m2 = (px2 >= 0) & (px2 < SCREEN_W) & (py2 >= 0) & (py2 < SCREEN_H)
                        buf[py2[m2], px2[m2]] = color

        pil_img = Image.fromarray(buf)
        tk_img = ImageTk.PhotoImage(pil_img)
        canvas.itemconfig(img_id, image=tk_img)
        canvas.image = tk_img
        T.update()

if __name__ == "__main__":
    sierpinski_turtle()
