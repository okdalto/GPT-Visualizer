import numpy as np


# Spacing constant for box layout
SPACING = 0.5  # box_size(0.4) + gap(0.1)

# Layout gap constants (horizontal/vertical gaps between matrices)
MATRIX_X_GAP = 4.0        # default horizontal gap between side-by-side matrices
MHA_MATMUL_X_GAP = 4.0    # horizontal gap for MHA matmul sub-ops
QKV_STACK_Y_GAP = 4.0     # vertical gap for Q, K, V projection stacking
MHA_HEAD_Y_GAP = 3.5      # vertical gap between attention heads

# Z-axis offsets within a matmul group (A at base, C in front, B behind)
MATMUL_Z_C = 3.0          # result matrix Z offset from A
MATMUL_Z_B = 6.0          # weight matrix Z offset from A

# Z-axis offsets within residual+layernorm stages
RESIDUAL_ADD_Z = 6.0      # Add result Z offset

# Z-axis offsets within FFN stage
FFN_W1_Z = 4.0            # W1 weight Z offset
FFN_PRERELU_Z = 8.0       # pre-relu / ReLU / Matmul 2 A position

# Stage Z positions - each stage is laid out along the Z axis
STAGE_Z = {
    'char_display': -2.0,
    'input': 0.0,
    'qkv_projection': 18.0,
    'multi_head_attn': 42.0,
    'concat_output_proj': 60.0,
    'residual_ln1': 70.0,
    'ffn': 76.0,            # = residual_ln1 + RESIDUAL_ADD_Z (seamless from LN1)
    'residual_ln2': 100.0,
    'output': 106.0,         # = residual_ln2 + RESIDUAL_ADD_Z (seamless from LN2)
    'block_1': 106.0,       # alias for output Z (used in logits_only abbreviated mode)
    'block_2': 114.0,
    'block_3': 122.0,
    'block_4': 130.0,
    'output_projection': 130.0,   # A at final block Z (seamless), B at +4, C at +2
    'token_probs': 144.0,
}


def matrix_origin(z: float, x_offset: float = 0.0, y_offset: float = 0.0):
    """Return origin position for a matrix visual at given stage Z with offsets."""
    return np.array([x_offset, y_offset, z], dtype=np.float32)


def side_by_side_x(widths: list[float], gap: float = 2.0) -> list[float]:
    """Calculate X offsets for matrices placed side by side with gaps.
    Centers the group around x=0."""
    total = sum(widths) + gap * (len(widths) - 1)
    x = -total / 2
    offsets = []
    for w in widths:
        offsets.append(x)
        x += w + gap
    return offsets


def stacked_y(heights: list[float], gap: float = 2.0) -> list[float]:
    """Calculate Y offsets for matrices stacked vertically.
    First matrix at top, subsequent ones below."""
    y = 0.0
    offsets = []
    for h in heights:
        offsets.append(y)
        y -= h + gap
    return offsets
