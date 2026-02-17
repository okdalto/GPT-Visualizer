# GPT Visualizer

An interactive 3D visualization of a 4-block GPT (decoder-only transformer), built with Python and OpenGL. Every matrix operation — from QKV projection to multi-head attention to feed-forward layers — is animated step by step, showing how data flows through the architecture. Includes a trained character-level language model that generates text autoregressively on screen.

![OpenGL 4.1](https://img.shields.io/badge/OpenGL-4.1%20Core-blue)
![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green)

<p align="center">
  <video src="https://github.com/user-attachments/assets/1f1d0d02-8fe7-4012-a836-3d8b62d2ef99" width="720" autoplay loop muted>
    Your browser does not support the video tag.
  </video>
</p>

## What It Shows

The visualizer renders each matrix as a grid of colored cubes (blue = negative, red = positive) and animates the full pipeline of a 4-block GPT:

**Block 1 (fully animated):**

| Stage | Operation | Dimensions |
|-------|-----------|------------|
| **Char Display** | Animated character fly-in | — |
| **Input** | Token embeddings + positional encoding | 16 × 16 |
| **QKV Projection** | X · W_Q, X · W_K, X · W_V | 16×16 · 16×16 → 16×16 |
| **Multi-Head Attention** | Q_h · K_h^T → softmax → Weights · V_h (×4 heads) | 16×4 per head |
| **Concat + Output Proj** | Concat(heads) · W_O | 16×16 · 16×16 → 16×16 |
| **Residual + LayerNorm 1** | X + Attn → LayerNorm | 16×16 |
| **FFN** | W1 → ReLU → W2 | 16→128→16 |
| **Residual + LayerNorm 2** | LN1 + FFN → LayerNorm | 16×16 |
| **Output** | Block 1 result | 16 × 16 |

**Blocks 2–4 (simplified):** Each block shows an in-place color transition from input to output.

**LM Head (with `--text`):**

| Stage | Operation |
|-------|-----------|
| **Output Projection** | Final output · W_embed^T → logits |
| **Token Probs** | Softmax → row selection → argmax |

### Animation Details

- **Matrix multiplication**: Weight columns descend onto input rows, dissolve in (element-wise multiply), then collapse into result positions (summation)
- **Softmax / LayerNorm**: In-place color transitions with diagonal wave propagation
- **ReLU**: Color shift showing zero-suppression
- **Residual connections**: Skip-connection matrices fly back from earlier stages
- **Transpose**: K_h elements animate from (col, row) → (row, col) positions
- **Autoregressive generation**: Each generation step plays the full pipeline, then the next token is appended and the pipeline runs again

## Getting Started

### Requirements

- Python 3.9+
- OpenGL 4.1 compatible GPU
- macOS / Linux / Windows

### Install

```bash
pip install -r requirements.txt
```

Dependencies: `PyOpenGL`, `glfw`, `numpy`, `Pillow`

### Run

**Interactive mode** (random weights):

```bash
python main.py
```

**With trained model** (autoregressive text generation):

```bash
python main.py --text hello
```

The model autoregressively generates characters one at a time, animating the full transformer pipeline for each step.

### Train

Train a character-level language model on Tiny Shakespeare:

```bash
python -m transformer.train --epochs 150
```

Weights are saved to `transformer/weights/charlm.npz`. Training options:

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 150 | Number of training epochs |
| `--batch-size` | 64 | Batch size |
| `--lr` | 3e-3 | Initial learning rate |
| `--lr-min` | 1e-4 | Minimum learning rate (cosine schedule) |
| `--max-norm` | 1.0 | Gradient clipping max norm |
| `--check-grad` | off | Run numerical gradient check |

### Record

Record the animation as a video with synchronized audio:

```bash
python main.py --record --text hello --width 1920 --height 1024 --fps 30
```

Outputs `recordings/transformer.mp4` (requires `ffmpeg`). Recording options:

| Flag | Default | Description |
|------|---------|-------------|
| `--record` | off | Enable recording mode |
| `--fps` | 30 | Frames per second |
| `--speed` | 1.0 | Animation speed multiplier |
| `--width` | 1920 | Output width |
| `--height` | 1080 | Output height |
| `--format` | jpg | Frame format (jpg/png) |
| `--output-dir` | recordings/ | Output directory |
| `--temperature` | 0.3 | Sampling temperature (0=greedy) |
| `--seed` | auto | Random seed for sampling |
| `--weights` | auto | Path to trained weights (.npz) |

## Controls

| Key | Action |
|-----|--------|
| **Space** | Play / Pause |
| **← →** | Previous / Next stage |
| **+ / -** | Speed up / Slow down (0.25x – 5.0x) |
| **R** | Reset to beginning |
| **Mouse drag** | Orbit camera |
| **Scroll** | Zoom |
| **Esc** | Quit |

## Architecture

```
main.py                          Entry point, interactive & recording modes
├── transformer/
│   ├── parameters.py            GPT config (d_model=16, seq_len=16, heads=4, blocks=4)
│   ├── computation.py           Single transformer block (NumPy forward pass)
│   ├── model.py                 CharLM: 4-block GPT with embedding & LM head
│   ├── train.py                 Training loop (Tiny Shakespeare, cosine LR, gradient clipping)
│   └── vocab.py                 Character vocabulary (32 tokens: a-z, space, punctuation, PAD)
├── visualization/
│   ├── scene.py                 Stage layout, camera path, flow connections
│   ├── operation_visual.py      MatmulVisual, AddVisual, ActivationVisual, StaticMatrixVisual
│   ├── layout.py                3D positioning constants
│   └── colormap.py              Matrix values → RGBA
├── animation/
│   ├── timeline.py              Per-stage timing & per-group speed config
│   └── easing.py                Easing functions
├── core/
│   ├── window.py                GLFW window (1600×900, 4x MSAA)
│   ├── renderer.py              Instanced cube renderer (up to 8192 instances)
│   ├── shader.py                Shader program loader
│   ├── camera.py                Waypoint camera with smoothstep interpolation
│   └── text_renderer.py         Font atlas text rendering
├── shaders/
│   ├── box_instanced.vert/frag  Phong-lit instanced cubes
│   └── text.vert/frag           2D/3D text overlay
└── audio/
    ├── common.py                Shared constants (SR=44100) and WAV I/O
    ├── music.py                 Generative industrial electronic music (numpy → WAV)
    ├── sonification.py          Data sonification synced to animation timeline
    └── preview_sounds.py        Individual sound sample generator
```

### Rendering

- **Instanced rendering**: Each matrix element is a unit cube with per-instance position, color (RGBA), and scale, batched into one draw call per visible stage
- **Phong lighting**: Ambient + diffuse + specular shading on every cube
- **Orthographic camera**: Auto-frames each stage with consistent margins; smooth waypoint interpolation between stages
- **Text**: 3D world-space labels projected to screen, plus 2D stage name overlay

### Animation System

- **Timeline**: Multi-stage pipeline with `appear → compute → settle` phases; loops automatically
- **Per-group speed control**: `STAGE_SPEED` config in `timeline.py` lets you independently adjust the speed of each sub-operation (e.g., slow down the FFN matmuls, speed up softmax)
- **Staggered timing**: Diagonal wave propagation so elements don't all move at once
- **Perlin noise variation**: Pre-computed noise table gives each element slightly different movement speed for organic feel
- **Seamless transitions**: Output of one stage flies directly into the next stage's input position with alpha crossfade

### Audio

- **audio/sonification.py**: Generates micro-sounds (needle tones, clicks, sine blips, white noise bursts, data tones, frequency sweeps) synchronized to animation timeline events — matrix arrivals, multiplications, softmax, ReLU, and residual additions. Weighted distribution favors needle (45%) and click (35%) for a crisp, precise feel. Dynamic range compressor evens out volume across stages. Covers all stages including blocks 2–4, output projection, and token probabilities.
- **audio/music.py**: Pure numpy synthesis of industrial electronic music (155 BPM, stereo WAV). Includes FM bass, metallic percussion, data sonification streams, micro-click patterns, electrical impulse sequences, Shepard tones, and cellular automata glitch textures.
- **audio/common.py**: Shared WAV writer and sample rate constant.
- **audio/preview_sounds.py**: Generate individual WAV samples for each micro-sound type (`python -m audio.preview_sounds`).

## Configuration

### GPT Parameters

Edit `transformer/parameters.py`:

```python
@dataclass
class TransformerConfig:
    d_model: int = 16
    seq_len: int = 16
    num_heads: int = 4
    d_k: int = 4        # d_model // num_heads
    d_ff: int = 128
    num_blocks: int = 4
```

### Animation Speed

Edit `STAGE_SPEED` in `animation/timeline.py` to control per-stage, per-operation timing:

```python
STAGE_SPEED = {
    'qkv_projection':     {'appear': 1.0, 'settle': 1.0, 'compute': [0.6]},
    'multi_head_attn':    {'appear': 1.0, 'settle': 1.0, 'compute': [1.0, 1.0, 0.7]},
    'ffn':                {'appear': 1.0, 'settle': 1.0, 'compute': [0.25, 1.0, 0.25]},
    'block_2':            {'appear': 1.5, 'settle': 1.5, 'compute': [1.5]},
    # ... higher value = faster, lower = slower
}
```

## License

GPL-3.0
