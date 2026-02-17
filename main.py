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
    parser = argparse.ArgumentParser(description="GPT Visualizer")
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
    parser.add_argument("--text", type=str, default=None,
                        help="Input text prompt for autoregressive generation")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to trained weights (.npz)")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (0=greedy, >0=random, e.g. 0.8)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for sampling (default: deterministic from input)")
    return parser.parse_args()


def setup(args):
    """Create window, shaders, scene. Returns (app, scene, box_shader, text_renderer, label_renderer, config, generation_steps)."""
    if args.record:
        w, h = 640, 360  # small context window; actual rendering via offscreen FBO
        title = f"Recording {args.width}x{args.height} @ {args.fps}fps..."
    else:
        w, h = 1600, 900
        title = "GPT Visualizer"

    app = AppWindow(width=w, height=h, title=title)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    box_shader = ShaderProgram(
        os.path.join(base_dir, "shaders", "box_instanced.vert"),
        os.path.join(base_dir, "shaders", "box_instanced.frag"),
    )
    text_renderer = TextRenderer(
        os.path.join(base_dir, "shaders", "text.vert"),
        os.path.join(base_dir, "shaders", "text.frag"),
        font_size=44,
    )
    label_renderer = TextRenderer(
        os.path.join(base_dir, "shaders", "text.vert"),
        os.path.join(base_dir, "shaders", "text.frag"),
        font_size=96,
    )

    config = TransformerConfig()
    input_labels = None
    output_labels = None

    weights_path = args.weights
    if weights_path is None:
        default_path = os.path.join(base_dir, "transformer", "weights", "charlm.npz")
        if args.text and os.path.exists(default_path):
            weights_path = default_path

    generation_steps = None

    if weights_path and os.path.exists(weights_path) and args.text:
        from transformer.model import CharLM
        from transformer.vocab import (
            encode_text, make_labels, ID_TO_CHAR, CHAR_TO_ID, PAD_ID,
        )

        model = CharLM(config)
        model.load(weights_path)

        # Pre-compute all autoregressive generation steps
        generation_steps = []
        current_chars = [c for c in args.text.lower() if c in CHAR_TO_ID]
        current_chars = current_chars[:config.seq_len]

        print("=== Autoregressive Generation ===")
        for step in range(config.seq_len):
            token_ids = encode_text(
                ''.join(current_chars), config.seq_len)
            results = model.forward(token_ids, temperature=args.temperature,
                                    seed=args.seed)
            pred_pos = len(current_chars) - 1
            pred_id = int(results['predicted_ids'][pred_pos])
            pred_prob = float(results['probs'][pred_pos, pred_id]) * 100

            in_labels = make_labels(token_ids)
            out_labels = [''] * config.seq_len
            pred_char = make_labels(np.array([pred_id]))[0]
            out_labels[pred_pos] = f"{pred_char} {pred_prob:.0f}%"

            results['pred_pos'] = pred_pos
            generation_steps.append((results, in_labels, out_labels))
            print(f"  Step {step + 1}: "
                  f"{''.join(current_chars)} -> {pred_char} ({pred_prob:.0f}%)")

            if pred_id == PAD_ID or len(current_chars) >= config.seq_len:
                break
            current_chars.append(ID_TO_CHAR[pred_id])

        # Final step: show the completed text
        final_text = ''.join(current_chars)
        token_ids = encode_text(final_text, config.seq_len)
        results = model.forward(token_ids, temperature=args.temperature,
                                seed=args.seed)
        in_labels = make_labels(token_ids)
        out_labels = [''] * config.seq_len
        results['pred_pos'] = len(current_chars) - 1
        generation_steps.append((results, in_labels, out_labels))

        print(f"  Result: {repr(final_text)}")
        print()

        # Use first step for initial scene
        results, input_labels, output_labels = generation_steps[0]
    else:
        blocks = [TransformerBlock(config, seed=42 + i) for i in range(config.num_blocks)]
        x = np.random.RandomState(123).randn(config.seq_len, config.d_model).astype(np.float32) * 0.5
        block_results = []
        inp = x
        for block in blocks:
            r = block.forward(inp, causal=True)
            block_results.append(r)
            inp = r['output']
        results = dict(block_results[0])
        for i, r in enumerate(block_results):
            results[f'block_{i}'] = r
        results['output'] = block_results[-1]['output']

    if args.record:
        aspect = args.width / max(args.height, 1)
    else:
        fb_w, fb_h = app.get_framebuffer_size()
        aspect = fb_w / max(fb_h, 1)
    scene = Scene(results, config, box_shader, aspect=aspect,
                  input_labels=input_labels, output_labels=output_labels)

    # For generation mode: first step plays once, last step loops
    if generation_steps and len(generation_steps) > 1:
        scene.timeline.loop = False

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_BLEND)
    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
    gl.glEnable(gl.GL_MULTISAMPLE)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)

    return app, scene, box_shader, text_renderer, label_renderer, config, generation_steps


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


def create_offscreen_fbo(width, height, samples=4):
    """Create MSAA offscreen framebuffer for exact-resolution recording."""
    msaa_fbo = gl.glGenFramebuffers(1)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, msaa_fbo)

    msaa_color = gl.glGenRenderbuffers(1)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, msaa_color)
    gl.glRenderbufferStorageMultisample(
        gl.GL_RENDERBUFFER, samples, gl.GL_RGBA8, width, height)
    gl.glFramebufferRenderbuffer(
        gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
        gl.GL_RENDERBUFFER, msaa_color)

    msaa_depth = gl.glGenRenderbuffers(1)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, msaa_depth)
    gl.glRenderbufferStorageMultisample(
        gl.GL_RENDERBUFFER, samples, gl.GL_DEPTH_COMPONENT24, width, height)
    gl.glFramebufferRenderbuffer(
        gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT,
        gl.GL_RENDERBUFFER, msaa_depth)

    status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
    if status != gl.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"MSAA FBO incomplete: {status:#x}")

    resolve_fbo = gl.glGenFramebuffers(1)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, resolve_fbo)

    resolve_color = gl.glGenRenderbuffers(1)
    gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, resolve_color)
    gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_RGBA8, width, height)
    gl.glFramebufferRenderbuffer(
        gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
        gl.GL_RENDERBUFFER, resolve_color)

    status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
    if status != gl.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"Resolve FBO incomplete: {status:#x}")

    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    return {
        'msaa': msaa_fbo, 'resolve': resolve_fbo,
        'w': width, 'h': height,
        '_rbs': (msaa_color, msaa_depth, resolve_color),
    }


def fbo_read_pixels(fbo):
    """Resolve MSAA and read pixels from offscreen FBO."""
    w, h = fbo['w'], fbo['h']
    gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, fbo['msaa'])
    gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, fbo['resolve'])
    gl.glBlitFramebuffer(0, 0, w, h, 0, 0, w, h,
                         gl.GL_COLOR_BUFFER_BIT, gl.GL_NEAREST)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo['resolve'])
    gl.glFinish()
    pixels = gl.glReadPixels(0, 0, w, h, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    img = Image.frombytes("RGB", (w, h), pixels)
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def destroy_fbo(fbo):
    """Clean up offscreen FBO resources."""
    for fb in (fbo['msaa'], fbo['resolve']):
        gl.glDeleteFramebuffers(1, [fb])
    for rb in fbo['_rbs']:
        gl.glDeleteRenderbuffers(1, [rb])


def run_interactive(args):
    app, scene, box_shader, text_renderer, label_renderer, config, gen_steps = setup(args)

    # Mutable references for closures
    scene_ref = [scene]
    step_idx = [0]

    def advance_generation():
        """Advance to next generation step if available."""
        if not gen_steps or step_idx[0] >= len(gen_steps) - 1:
            return
        step_idx[0] += 1
        results, in_lbl, out_lbl = gen_steps[step_idx[0]]
        fb_w, fb_h = app.get_framebuffer_size()
        aspect = fb_w / max(fb_h, 1)
        is_last = step_idx[0] == len(gen_steps) - 1
        new_scene = Scene(results, config, box_shader, aspect=aspect,
                          input_labels=in_lbl, output_labels=out_lbl,
                          renderer=scene_ref[0].renderer,
                          final_display=is_last,
                          logits_only=(not is_last))
        new_scene.timeline.loop = False
        scene_ref[0] = new_scene
        print(f"  Generation step {step_idx[0] + 1}/{len(gen_steps)}")
        sys.stdout.flush()

    def on_key(key, scancode, action, mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(app.window, True)
        elif key == glfw.KEY_SPACE:
            scene_ref[0].timeline.toggle_play()
        elif key == glfw.KEY_RIGHT:
            idx = scene_ref[0].timeline.get_current_stage_index()
            scene_ref[0].timeline.jump_to_stage(idx + 1)
        elif key == glfw.KEY_LEFT:
            idx = scene_ref[0].timeline.get_current_stage_index()
            scene_ref[0].timeline.jump_to_stage(idx - 1)
        elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:
            scene_ref[0].timeline.speed = min(5.0, scene_ref[0].timeline.speed + 0.25)
            print(f"Speed: {scene_ref[0].timeline.speed:.2f}x")
        elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:
            scene_ref[0].timeline.speed = max(0.25, scene_ref[0].timeline.speed - 0.25)
            print(f"Speed: {scene_ref[0].timeline.speed:.2f}x")
        elif key == glfw.KEY_R:
            scene_ref[0].timeline.current_time = 0.0
            scene_ref[0].timeline.playing = True
            print("Reset to beginning")

    app.add_key_callback(on_key)
    app.add_mouse_callback(lambda dx, dy: scene_ref[0].camera.handle_mouse(dx, dy))
    app.add_scroll_callback(lambda xo, yo: scene_ref[0].camera.handle_scroll(xo, yo))

    last_time = glfw.get_time()
    last_stage = ""

    print("=== GPT Visualizer ===")
    print("Controls:")
    print("  Space      : Play / Pause")
    print("  Left/Right : Previous / Next stage")
    print("  +/-        : Speed up / Slow down")
    print("  R          : Reset to beginning")
    print("  Mouse drag : Orbit camera")
    print("  Scroll     : Zoom")
    print("  Esc        : Quit")
    if gen_steps:
        print(f"\n  Generation mode: {len(gen_steps)} steps")
    print()
    sys.stdout.flush()

    while not app.should_close():
        current_time = glfw.get_time()
        dt = min(current_time - last_time, 0.05)
        last_time = current_time

        scene_ref[0].update(dt)

        # Advance generation when current cycle completes
        if gen_steps and scene_ref[0].timeline.completed:
            advance_generation()

        fb_w, fb_h = app.get_framebuffer_size()
        stage = render_frame(scene_ref[0], text_renderer, label_renderer, fb_w, fb_h)

        phase, phase_t = stage.get_phase(scene_ref[0].timeline.current_time)
        stage_id = f"{stage.stage_name}:{phase}"
        if stage_id != last_stage:
            last_stage = stage_id
            print(f"  [{stage.stage_name}] {phase} (t={scene_ref[0].timeline.current_time:.1f}s)")
            sys.stdout.flush()

        app.swap_and_poll()

    scene_ref[0].renderer.destroy()
    box_shader.destroy()
    text_renderer.destroy()
    label_renderer.destroy()
    app.terminate()


def run_record(args):
    app, scene, box_shader, text_renderer, label_renderer, config, gen_steps = setup(args)
    fbo = create_offscreen_fbo(args.width, args.height)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.output_dir or os.path.join(base_dir, "recordings")
    os.makedirs(out_dir, exist_ok=True)

    ext = args.format
    save_kwargs = {"quality": 95} if ext == "jpg" else {}
    fb_w, fb_h = fbo['w'], fbo['h']
    sim_dt = args.speed / args.fps
    aspect = args.width / max(args.height, 1)
    renderer = scene.renderer

    # Build list of scenes for each generation step
    scenes = []
    if gen_steps and len(gen_steps) > 1:
        for si, (results, in_lbl, out_lbl) in enumerate(gen_steps):
            is_last = si == len(gen_steps) - 1
            is_first = si == 0
            s = Scene(results, config, box_shader, aspect=aspect,
                      input_labels=in_lbl, output_labels=out_lbl,
                      renderer=renderer,
                      final_display=is_last,
                      logits_only=(not is_last and not is_first))
            s.timeline.playing = False
            s.timeline.loop = False
            scenes.append(s)
    else:
        scene.timeline.playing = False
        scenes.append(scene)

    # Calculate per-step frame counts
    step_infos = []
    total_frames = 0
    for s in scenes:
        dur = s.timeline.total_duration + s.timeline.return_duration
        nf = int(dur * args.fps / args.speed)
        step_infos.append((s, dur, nf))
        total_frames += nf

    total_duration = sum(dur for _, dur, _ in step_infos)
    print(f"=== Recording: {total_frames} frames @ {args.fps}fps (speed={args.speed}x) ===")
    print(f"  Steps    : {len(scenes)}")
    print(f"  Duration : {total_duration:.1f}s total")
    for si, (s, dur, nf) in enumerate(step_infos):
        mode = "full" if si == 0 else ("final" if si == len(scenes) - 1 else "logits")
        print(f"    Step {si+1}: {dur:.1f}s ({nf} frames, {mode})")
    print(f"  Output   : {out_dir}/frame_XXXXX.{ext}")
    print(f"  Size     : {fb_w}x{fb_h} (offscreen FBO)")
    print()

    # Record all frames across all steps
    num_saved = 0
    frame_idx = 0
    aborted = False
    for step_i, (s, dur, nf) in enumerate(step_infos):
        print(f"--- Step {step_i+1}/{len(scenes)} ({nf} frames) ---")
        sys.stdout.flush()

        # Warm up
        s.timeline.current_time = 0.0
        s.update(1.0 / 60.0)

        for i in range(nf):
            s.timeline.current_time = i * sim_dt
            s.update(0.0)

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo['msaa'])
            gl.glViewport(0, 0, fb_w, fb_h)
            stage = render_frame(s, text_renderer, label_renderer, fb_w, fb_h)
            img = fbo_read_pixels(fbo)

            frame_path = os.path.join(out_dir, f"frame_{frame_idx:05d}.{ext}")
            img.save(frame_path, **save_kwargs)
            frame_idx += 1
            num_saved += 1

            app.swap_and_poll()
            if app.should_close():
                print("\nWindow closed, stopping.")
                aborted = True
                break

            if (i + 1) % args.fps == 0 or i == nf - 1:
                pct = frame_idx / total_frames * 100
                print(f"  [{frame_idx}/{total_frames}] {pct:.0f}%  t={s.timeline.current_time:.1f}s  {stage.stage_name}")
                sys.stdout.flush()

        if aborted:
            break

    print(f"\nDone. {num_saved} frames saved to {out_dir}/")

    destroy_fbo(fbo)
    renderer.destroy()
    box_shader.destroy()
    text_renderer.destroy()
    label_renderer.destroy()
    app.terminate()

    # Generate audio for each step and concatenate
    print("\n=== Generating soundtrack ===")
    from audio import build_soundtrack, write_wav
    audio_parts = []
    for si, (s, dur, nf) in enumerate(step_infos):
        print(f"  Step {si+1}/{len(scenes)}:")
        part = build_soundtrack(s.results, config, timeline=s.timeline,
                               logits_only=s.logits_only)
        audio_parts.append(part)
    audio = np.concatenate(audio_parts, axis=0)
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
