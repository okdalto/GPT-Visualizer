"""Capture a single high-res JPG screenshot from the Transformer Block Visualizer.

Usage:
    python capture_screenshot.py [--time TIME_SEC] [--output PATH]
"""
import os
import sys
import argparse

os.environ['GL_SILENCE_DEPRECATION'] = '1'
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


def capture_screenshot(target_time: float, out_path: str):
    width, height = 1920, 1080
    app = AppWindow(width=width, height=height, title="Capturing screenshot...")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    box_shader = ShaderProgram(
        os.path.join(base_dir, "shaders", "box_instanced.vert"),
        os.path.join(base_dir, "shaders", "box_instanced.frag"),
    )
    text_renderer = TextRenderer(
        os.path.join(base_dir, "shaders", "text.vert"),
        os.path.join(base_dir, "shaders", "text.frag"),
        font_size=42,
    )
    label_renderer = TextRenderer(
        os.path.join(base_dir, "shaders", "text.vert"),
        os.path.join(base_dir, "shaders", "text.frag"),
        font_size=27,
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
    gl.glClearColor(0.08, 0.08, 0.12, 1.0)

    # Advance timeline to target time
    scene.timeline.playing = False
    dt_step = 1.0 / 60.0
    t = 0.0
    while t < target_time:
        scene.timeline.current_time = t
        scene.update(dt_step)
        t += dt_step

    # Set exact time and do final update
    scene.timeline.current_time = target_time
    scene.update(0.0)

    fb_w, fb_h = app.get_framebuffer_size()
    aspect = fb_w / max(fb_h, 1)

    # Render the frame
    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    scene.render(aspect)
    scene.render_labels(label_renderer, fb_w, fb_h)

    stage_idx = scene.timeline.get_current_stage_index()
    stage = scene.timeline.stages[stage_idx]
    text_renderer.render_stage_name(stage.stage_name, fb_w, fb_h)

    gl.glFinish()

    # Read pixels
    pixels = gl.glReadPixels(0, 0, fb_w, fb_h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    img = Image.frombytes("RGB", (fb_w, fb_h), pixels)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Downsample if retina
    if fb_w > width:
        img = img.resize((width, height), Image.LANCZOS)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, "JPEG", quality=95)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Saved {out_path} ({img.size[0]}x{img.size[1]}, {size_kb:.0f}KB)")
    print(f"  Stage: {stage.stage_name}, t={target_time:.1f}s")

    scene.renderer.destroy()
    box_shader.destroy()
    text_renderer.destroy()
    label_renderer.destroy()
    app.terminate()


def main():
    parser = argparse.ArgumentParser(description="Capture a screenshot from Transformer Visualizer")
    parser.add_argument("--time", type=float, default=None,
                        help="Time in seconds to capture (default: auto-pick multi-head attention)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path (default: assets/hero.jpg)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    if args.output is None:
        args.output = os.path.join(base_dir, "assets", "hero.jpg")

    if args.time is None:
        # Auto-compute: target the multi-head attention stage during
        # the Weights x V computation (all 4 heads visible, most impactful)
        from animation.timeline import AnimationTimeline
        tl = AnimationTimeline()
        mha_stage = tl.get_stage('multi_head_attn')
        # 75% through the compute phase → Weights×V sub-op in progress
        target = mha_stage.start_time + mha_stage.appear_duration + mha_stage.compute_duration * 0.75
        args.time = target
        print(f"Auto-selected time: {args.time:.1f}s (multi_head_attn compute)")

    capture_screenshot(args.time, args.output)


if __name__ == "__main__":
    main()
