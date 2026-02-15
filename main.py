import os
import sys
import argparse
import subprocess
import shutil

os.environ['GL_SILENCE_DEPRECATION'] = '1'

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import glfw
import numpy as np
from OpenGL import GL as gl
from PIL import Image

from core.window import AppWindow
from core.shader import ShaderProgram
from core.text_renderer import TextRenderer
from transformer.parameters import TransformerConfig
from transformer.computation import TransformerBlock
from visualization.scene import Scene


def parse_args():
    parser = argparse.ArgumentParser(description="Transformer Block Visualizer")
    parser.add_argument("--record", action="store_true",
                        help="Record animation as image sequence")
    parser.add_argument("--fps", type=int, default=30,
                        help="Recording FPS (default: 30)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Animation speed multiplier for recording (default: 1.0)")
    parser.add_argument("--width", type=int, default=1920,
                        help="Recording width (default: 1920)")
    parser.add_argument("--height", type=int, default=1080,
                        help="Recording height (default: 1080)")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg",
                        help="Image format (default: jpg)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: recordings/)")
    return parser.parse_args()


def setup(args):
    """Create window, shaders, scene. Returns (app, scene, box_shader, text_renderer, label_renderer)."""
    if args.record:
        w, h = args.width, args.height
        title = f"Recording {w}x{h} @ {args.fps}fps..."
    else:
        w, h = 1600, 900
        title = "Transformer Block Visualizer"

    app = AppWindow(width=w, height=h, title=title)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    box_shader = ShaderProgram(
        os.path.join(base_dir, "shaders", "box_instanced.vert"),
        os.path.join(base_dir, "shaders", "box_instanced.frag"),
    )
    text_renderer = TextRenderer(
        os.path.join(base_dir, "shaders", "text.vert"),
        os.path.join(base_dir, "shaders", "text.frag"),
        font_size=54,
    )
    label_renderer = TextRenderer(
        os.path.join(base_dir, "shaders", "text.vert"),
        os.path.join(base_dir, "shaders", "text.frag"),
        font_size=40,
    )

    config = TransformerConfig()
    transformer = TransformerBlock(config)
    x = np.random.RandomState(123).randn(config.seq_len, config.d_model).astype(np.float32) * 0.5
    results = transformer.forward(x)
    scene = Scene(results, config, box_shader)

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_MULTISAMPLE)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)

    return app, scene, box_shader, text_renderer, label_renderer


def render_frame(scene, text_renderer, label_renderer, fb_w, fb_h):
    """Clear, render scene + labels + stage name overlay."""
    aspect = fb_w / max(fb_h, 1)
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    scene.render(aspect)
    scene.render_labels(label_renderer, fb_w, fb_h)
    stage_idx = scene.timeline.get_current_stage_index()
    stage = scene.timeline.stages[stage_idx]
    text_renderer.render_stage_name(stage.stage_name, fb_w, fb_h)
    return stage


def read_pixels(fb_w, fb_h, target_w, target_h):
    """Read framebuffer and return a PIL Image, downsampled if retina."""
    gl.glFinish()
    pixels = gl.glReadPixels(0, 0, fb_w, fb_h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    img = Image.frombytes("RGB", (fb_w, fb_h), pixels)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if fb_w > target_w:
        img = img.resize((target_w, target_h), Image.LANCZOS)
    return img


def run_interactive(args):
    app, scene, box_shader, text_renderer, label_renderer = setup(args)

    def on_key(key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(app.window, True)
        elif key == glfw.KEY_SPACE:
            scene.timeline.toggle_play()
        elif key == glfw.KEY_RIGHT:
            idx = scene.timeline.get_current_stage_index()
            scene.timeline.jump_to_stage(idx + 1)
        elif key == glfw.KEY_LEFT:
            idx = scene.timeline.get_current_stage_index()
            scene.timeline.jump_to_stage(idx - 1)
        elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:
            scene.timeline.speed = min(5.0, scene.timeline.speed + 0.25)
            print(f"Speed: {scene.timeline.speed:.2f}x")
        elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:
            scene.timeline.speed = max(0.25, scene.timeline.speed - 0.25)
            print(f"Speed: {scene.timeline.speed:.2f}x")
        elif key == glfw.KEY_R:
            scene.timeline.current_time = 0.0
            scene.timeline.playing = True
            print("Reset to beginning")

    app.add_key_callback(on_key)
    app.add_mouse_callback(lambda dx, dy: scene.camera.handle_mouse(dx, dy))
    app.add_scroll_callback(lambda xo, yo: scene.camera.handle_scroll(xo, yo))

    last_time = glfw.get_time()
    last_stage = ""

    print("=== Transformer Block Visualizer ===")
    print("Controls:")
    print("  Space      : Play / Pause")
    print("  Left/Right : Previous / Next stage")
    print("  +/-        : Speed up / Slow down")
    print("  R          : Reset to beginning")
    print("  Mouse drag : Orbit camera")
    print("  Scroll     : Zoom")
    print("  Esc        : Quit")
    print()
    sys.stdout.flush()

    while not app.should_close():
        current_time = glfw.get_time()
        dt = min(current_time - last_time, 0.05)
        last_time = current_time

        scene.update(dt)

        fb_w, fb_h = app.get_framebuffer_size()
        stage = render_frame(scene, text_renderer, label_renderer, fb_w, fb_h)

        phase, phase_t = stage.get_phase(scene.timeline.current_time)
        stage_id = f"{stage.stage_name}:{phase}"
        if stage_id != last_stage:
            last_stage = stage_id
            print(f"  [{stage.stage_name}] {phase} (t={scene.timeline.current_time:.1f}s)")
            sys.stdout.flush()

        app.swap_and_poll()

    scene.renderer.destroy()
    box_shader.destroy()
    text_renderer.destroy()
    label_renderer.destroy()
    app.terminate()


def run_record(args):
    app, scene, box_shader, text_renderer, label_renderer = setup(args)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.output_dir or os.path.join(base_dir, "recordings")
    os.makedirs(out_dir, exist_ok=True)

    ext = args.format
    save_kwargs = {"quality": 95} if ext == "jpg" else {}

    scene.timeline.playing = False
    loop_duration = scene.timeline.total_duration + scene.timeline.return_duration
    sim_dt = args.speed / args.fps
    total_frames = int(loop_duration * args.fps / args.speed)

    print(f"=== Recording: {total_frames} frames @ {args.fps}fps (speed={args.speed}x) ===")
    print(f"  Duration : {loop_duration:.1f}s (animation {scene.timeline.total_duration:.1f}s + return {scene.timeline.return_duration:.1f}s)")
    print(f"  Output   : {out_dir}/frame_XXXXX.{ext}")

    fb_w, fb_h = app.get_framebuffer_size()
    print(f"  Size     : {args.width}x{args.height} (fb {fb_w}x{fb_h})")
    print()

    # Warm up: advance scene to t=0 so camera/visuals initialize
    scene.timeline.current_time = 0.0
    scene.update(1.0 / 60.0)

    for i in range(total_frames):
        scene.timeline.current_time = (i * sim_dt) % loop_duration
        scene.update(0.0)

        stage = render_frame(scene, text_renderer, label_renderer, fb_w, fb_h)
        img = read_pixels(fb_w, fb_h, args.width, args.height)

        frame_path = os.path.join(out_dir, f"frame_{i:05d}.{ext}")
        img.save(frame_path, **save_kwargs)

        app.swap_and_poll()
        if app.should_close():
            print("\nWindow closed, stopping.")
            break

        if (i + 1) % args.fps == 0 or i == total_frames - 1:
            pct = (i + 1) / total_frames * 100
            print(f"  [{i+1}/{total_frames}] {pct:.0f}%  t={scene.timeline.current_time:.1f}s  {stage.stage_name}")
            sys.stdout.flush()

    num_saved = i + 1
    print(f"\nDone. {num_saved} frames saved to {out_dir}/")

    # Get transformer results for sound generation
    config = TransformerConfig()
    transformer = TransformerBlock(config)
    x = np.random.RandomState(123).randn(config.seq_len, config.d_model).astype(np.float32) * 0.5
    results = transformer.forward(x)

    scene.renderer.destroy()
    box_shader.destroy()
    text_renderer.destroy()
    label_renderer.destroy()
    app.terminate()

    # Generate audio
    print("\n=== Generating soundtrack ===")
    from audio import build_soundtrack, write_wav
    audio = build_soundtrack(results, config)
    wav_path = os.path.join(out_dir, "soundtrack.wav")
    write_wav(wav_path, audio)
    print(f"  Audio saved to {wav_path}")

    # Encode video+audio in single ffmpeg pass
    if shutil.which("ffmpeg"):
        mp4_path = os.path.join(base_dir, "recordings", "transformer.mp4")
        frame_pattern = os.path.join(out_dir, f"frame_%05d.{ext}")
        cmd = [
            "ffmpeg", "-y",
            "-framerate", str(args.fps),
            "-i", frame_pattern,
            "-i", wav_path,
            "-c:v", "libx264",
            "-crf", "18",
            "-preset", "slow",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            mp4_path,
        ]
        print(f"\n=== Encoding video + audio ===")
        print(f"  {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        size_mb = os.path.getsize(mp4_path) / (1024 * 1024)
        print(f"\n  Video saved: {mp4_path} ({size_mb:.1f} MB)")
    else:
        print("\n  ffmpeg not found — skipping video encoding.")
        print(f"  Frames at: {out_dir}/")
        print(f"  Audio at: {wav_path}")


def main():
    args = parse_args()
    if args.record:
        run_record(args)
    else:
        run_interactive(args)


if __name__ == '__main__':
    main()
