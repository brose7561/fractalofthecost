import viewbrot
import matplotlib.pyplot as plt
import argparse

fig, ax = plt.subplots()
img_handle = None

instructions = """
🌀 Mandelbrot Explorer - Interactive Controls:
  ▸ W = pan up
  ▸ A = pan left
  ▸ S = pan down
  ▸ D = pan right
  ▸ Z = zoom in
  ▸ X = zoom out
  ▸ Q = quit
Click on the image window first, then press keys.
"""

def on_key(event):
    global img_handle
    if event.key == 'q':
        print("👋 Exiting viewer.")
        plt.close(fig)
        return
    elif event.key == 'w':
        viewbrot.pan(0, -1)
    elif event.key == 's':
        viewbrot.pan(0, 1)
    elif event.key == 'a':
        viewbrot.pan(-1, 0)
    elif event.key == 'd':
        viewbrot.pan(1, 0)
    elif event.key == 'z':
        viewbrot.zoom(1.5)
    elif event.key == 'x':
        viewbrot.zoom(0.67)
    else:
        print(f"⚠️ Unmapped key: '{event.key}'")
        return

    print(f"🔍 Re-rendering view (zoom={viewbrot.get_view_setting('zoom'):.2f})...")
    data = viewbrot.compute_mandelbrot()
    img = viewbrot.processFractal(data)
    img_handle.set_data(img)
    fig.canvas.draw_idle()

def interactive_mode():
    global img_handle
    print(instructions)
    data = viewbrot.compute_mandelbrot()
    img = viewbrot.processFractal(data)
    img_handle = ax.imshow(img)
    fig.canvas.mpl_connect('key_press_event', on_key)
    plt.tight_layout(pad=0)
    plt.show()

def preset_mode():
    print("🎯 Mandelbrot Preset Zoom")
    viewbrot.set_view_setting("center_x", -0.75)
    viewbrot.set_view_setting("center_y", 0.1)
    viewbrot.set_view_setting("zoom", 10.0)
    viewbrot.render()

def main():
    parser = argparse.ArgumentParser(description="Explore the Mandelbrot Set")
    parser.add_argument('--mode', choices=['interactive', 'preset'], default='interactive')
    args = parser.parse_args()

    print(f"Running on: {viewbrot.get_view_setting('device')}")

    if args.mode == 'interactive':
        interactive_mode()
    elif args.mode == 'preset':
        preset_mode()

if __name__ == "__main__":
    main()