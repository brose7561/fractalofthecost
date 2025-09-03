import math, random, time
import turtle as T
import tkinter as TK

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
COLOR = "#20d0ff"

ZOOM_TARGET_SCALE = 0.06
ZOOM_STEP_FACTOR = 1.2
ZOOM_MAX_STEPS = 30

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

def sierpinski_turtle():
    screen = T.Screen()
    screen.setup(SCREEN_W, SCREEN_H)
    screen.screensize(SCREEN_W, SCREEN_H)
    screen.title("Sierpiński Triangle — Turtle + Zoom Button")

    pen = T.Turtle(visible=False)
    pen.penup()
    pen.hideturtle()
    pen.speed(0)
    pen.color(COLOR)
    T.tracer(0, 0)

    s = START_SCALE
    verts = scale_verts(BASE_VERTS, s)
    set_view_for(verts)
    x, y = sum(v[0] for v in verts) / 3.0, sum(v[1] for v in verts) / 3.0

    zoom_request = {"go": False, "s": s}

    def fast_zoom():
        cur = zoom_request["s"]
        steps = 0
        T.tracer(0, 0)
        while cur < ZOOM_TARGET_SCALE and steps < ZOOM_MAX_STEPS:
            cur = min(ZOOM_TARGET_SCALE, cur / ZOOM_STEP_FACTOR if cur < 1e-12 else cur * (1.0 / ZOOM_STEP_FACTOR))
            v = scale_verts(BASE_VERTS, cur)
            set_view_for(v)
            T.update()
            steps += 1

    def on_zoom_click():
        zoom_request["go"] = True
        zoom_request["s"] = s
        fast_zoom()

    root = screen.getcanvas().winfo_toplevel()
    btn = TK.Button(root, text="Zoom to Centre", command=on_zoom_click)
    btn.place(relx=1.0, x=-20, y=20, anchor="ne")

    t0 = time.time()
    while True:
        if zoom_request["go"]:
            zoom_request["go"] = False

        s = min(1.0, s * SCALE_GROWTH)
        verts = scale_verts(BASE_VERTS, s)
        alpha = s
        batch = int(BATCH_MIN + (BATCH_MAX - BATCH_MIN) * alpha)
        dot_size = max(1, int(DOT_BASE * (1.0 - 0.7 * alpha)))
        idxs = choose_indices(batch)
        for k in idxs:
            vx, vy = verts[k]
            x = 0.5 * (x + vx)
            y = 0.5 * (y + vy)
            T.penup()
            T.goto(x, y)
            T.dot(dot_size, COLOR)
        set_view_for(verts)
        T.update()

if __name__ == "__main__":
    sierpinski_turtle()
