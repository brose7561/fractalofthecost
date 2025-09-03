import math, random, time
import turtle as T
import tkinter as TK
import colorsys

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
DOT_BASE = 4
START_SCALE = 0.001
SCALE_GROWTH = 1.022
BATCH_MIN = 200
BATCH_MAX = 6000

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

def choose_indices(n):
    if TORCH_OK:
        idx = torch.randint(0, 3, (n,), device=DEVICE)
        return idx.tolist()
    return [random.randrange(3) for _ in range(n)]

def hsv_hex(h, s=SAT, v=VAL):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def build_layers():
    layers = []
    s = START_SCALE
    for i in range(MAX_LAYERS):
        verts = scale_verts(BASE_VERTS, s)
        cx = sum(v[0] for v in verts) / 3.0
        cy = sum(v[1] for v in verts) / 3.0
        dot = max(1, int(DOT_BASE * (1.0 - 0.7 * min(1.0, s))))
        weight = min(1.0, 0.15 + s)  # larger triangles draw a bit more per frame
        layers.append({"s": s, "verts": verts, "x": cx, "y": cy, "dot": dot, "w": weight})
        s *= LAYER_GROWTH
        if s >= VIEW_MAX:
            break
    return layers

def sierpinski_turtle():
    screen = T.Screen()
    screen.setup(SCREEN_W, SCREEN_H)
    screen.screensize(SCREEN_W, SCREEN_H)
    screen.title("Sierpiński Triangle — Nested Layers")

    pen = T.Turtle(visible=False)
    pen.penup()
    pen.hideturtle()
    pen.speed(0)
    T.tracer(0, 0)

    layers = build_layers()

    view = VIEW_START
    prev_view = view
    set_view_for(scale_verts(BASE_VERTS, view))

    zoom_request = {"go": False, "s": view}
    base_hue = 0.55

    def fast_zoom():
        cur = zoom_request["s"]
        steps = 0
        T.tracer(0, 0)
        while cur < ZOOM_TARGET_SCALE and steps < ZOOM_MAX_STEPS:
            cur = min(ZOOM_TARGET_SCALE, cur * ZOOM_STEP_FACTOR)
            set_view_for(scale_verts(BASE_VERTS, cur))
            T.update()
            steps += 1

    def on_zoom_click():
        zoom_request["go"] = True
        zoom_request["s"] = view
        fast_zoom()

    root = screen.getcanvas().winfo_toplevel()
    btn = TK.Button(root, text="Zoom to Centre", command=on_zoom_click)
    btn.place(relx=1.0, x=-20, y=20, anchor="ne")

    while True:
        if zoom_request["go"]:
            zoom_request["go"] = False

        view = min(VIEW_MAX, view * SCALE_GROWTH)
        dv = max(0.0, view - prev_view)
        prev_view = view
        base_hue = (base_hue + HUE_RATE * dv) % 1.0

        for i, L in enumerate(layers):
            alpha = min(1.0, L["s"])
            batch = int((BATCH_MIN + (BATCH_MAX - BATCH_MIN) * alpha) * L["w"])
            color = hsv_hex(base_hue + i * 0.06)
            idxs = choose_indices(batch)
            for k in idxs:
                vx, vy = L["verts"][k]
                L["x"] = 0.5 * (L["x"] + vx)
                L["y"] = 0.5 * (L["y"] + vy)
                T.penup()
                T.goto(L["x"], L["y"])
                T.dot(L["dot"], color)

        set_view_for(scale_verts(BASE_VERTS, view))
        T.update()

if __name__ == "__main__":
    sierpinski_turtle()
