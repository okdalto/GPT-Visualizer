# Transformer Block Visualizer

An interactive 3D visualization of a Transformer encoder block, built with Python and OpenGL. Every matrix operation — from QKV projection to multi-head attention to feed-forward layers — is animated step by step, showing how data flows through the architecture.

![OpenGL 4.1](https://img.shields.io/badge/OpenGL-4.1%20Core-blue)
![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green)

<p align="center">
  <img src="assets/hero.jpg" alt="Transformer Block Visualizer — Multi-Head Attention" width="720">
</p>

## What It Shows

The visualizer renders each matrix as a grid of colored cubes (blue = negative, red = positive) and animates the 8 stages of a single Transformer encoder block:

| Stage | Operation | Dimensions |
|-------|-----------|------------|
| **Input** | Token embeddings | 8 × 16 |
| **QKV Projection** | X · W_Q, X · W_K, X · W_V | 8×16 · 16×16 → 8×16 |
| **Multi-Head Attention** | Q_h · K_h^T → softmax → Weights · V_h (×4 heads) | 8×4 per head |
| **Concat + Output Proj** | Concat(heads) · W_O | 8×16 · 16×16 → 8×16 |
| **Residual + LayerNorm 1** | X + Attn → LayerNorm | 8×16 |
| **FFN** | W1 → ReLU → W2 | 16→64→16 |
| **Residual + LayerNorm 2** | LN1 + FFN → LayerNorm | 8×16 |
| **Output** | Final result | 8 × 16 |

### Animation Details

- **Matrix multiplication**: Weight columns descend onto input rows, dissolve in (element-wise multiply), then collapse into result positions (summation)
- **Softmax / LayerNorm**: In-place color transitions with diagonal wave propagation
- **ReLU**: Color shift showing zero-suppression
- **Residual connections**: Skip-connection matrices fly back from earlier stages
- **Transpose**: K_h elements animate from (col, row) → (row, col) positions

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

```bash
python main.py
```

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
main.py                          Entry point & render loop
├── transformer/
│   ├── parameters.py            TransformerConfig (d_model=16, seq_len=8, heads=4)
│   └── computation.py           NumPy forward pass → result matrices
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
├── audio/
│   ├── common.py                Shared constants (SR=44100) and WAV I/O
│   ├── music.py                 Generative industrial electronic music (numpy → WAV)
│   └── sonification.py          Data sonification synced to animation timeline
└── capture_screenshot.py        Screenshot capture utility
```

### Rendering

- **Instanced rendering**: One draw call per frame — each matrix element is a unit cube with per-instance position, color (RGBA), and scale
- **Phong lighting**: Ambient + diffuse + specular shading on every cube
- **Orthographic camera**: Auto-frames each stage with consistent margins; smooth waypoint interpolation between stages
- **Text**: 3D world-space labels projected to screen, plus 2D stage name overlay

### Animation System

- **Timeline**: 8 stages with `appear → compute → settle` phases; loops automatically
- **Per-group speed control**: `STAGE_SPEED` config in `timeline.py` lets you independently adjust the speed of each sub-operation (e.g., slow down the FFN matmuls, speed up softmax)
- **Staggered timing**: Diagonal wave propagation so elements don't all move at once
- **Perlin noise variation**: Pre-computed noise table gives each element slightly different movement speed for organic feel
- **Seamless transitions**: Output of one stage flies directly into the next stage's input position with alpha crossfade

### Audio

- **audio/music.py**: Pure numpy synthesis of industrial electronic music (155 BPM, 2 min stereo WAV). Includes FM bass, metallic percussion, data sonification streams, micro-click patterns, electrical impulse sequences, Shepard tones, and cellular automata glitch textures.
- **audio/sonification.py**: Generates micro-sounds synchronized to animation timeline events (matrix arrivals, multiplications, activations).
- **audio/common.py**: Shared WAV writer and sample rate constant.

## Configuration

### Transformer Parameters

Edit `transformer/parameters.py`:

```python
@dataclass
class TransformerConfig:
    d_model: int = 16
    seq_len: int = 8
    num_heads: int = 4
    d_k: int = 4       # d_model // num_heads
    d_ff: int = 64
```

### Animation Speed

Edit `STAGE_SPEED` in `animation/timeline.py` to control per-stage, per-operation timing:

```python
STAGE_SPEED = {
    'qkv_projection':     {'appear': 1.0, 'settle': 1.0, 'compute': [0.7]},
    'multi_head_attn':    {'appear': 1.0, 'settle': 1.0, 'compute': [1.0, 1.0, 0.7]},
    'ffn':                {'appear': 1.0, 'settle': 1.0, 'compute': [0.25, 1.0, 0.25]},
    # ... higher value = faster, lower = slower
}
```

## License

MIT
