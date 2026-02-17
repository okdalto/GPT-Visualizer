import numpy as np


def matrix_to_colors(matrix: np.ndarray, alpha: float = 1.0,
                     vmax: float = None) -> np.ndarray:
    """Convert a matrix of values to RGBA colors. Returns (rows, cols, 4).
    If vmax is provided, use symmetric range [-vmax, vmax] for normalization.
    Otherwise, compute from matrix."""
    if vmax is None:
        vmax = max(abs(matrix.max()), abs(matrix.min()), 0.01)
    vmin = -vmax

    rows, cols = matrix.shape
    colors = np.zeros((rows, cols, 4), dtype=np.float32)

    # Vectorized colormap
    t = np.clip((matrix - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)

    mask_low = t < 0.5
    s_low = t * 2.0
    s_high = (t - 0.5) * 2.0

    colors[:, :, 0] = np.where(mask_low, s_low, 1.0)
    colors[:, :, 1] = np.where(mask_low, s_low, 1.0 - s_high)
    colors[:, :, 2] = np.where(mask_low, 1.0, 1.0 - s_high)
    colors[:, :, 3] = alpha

    return colors


def matrix_to_colors_sequential(matrix: np.ndarray, alpha: float = 1.0,
                                 vmin: float = 0.0,
                                 vmax: float = None) -> np.ndarray:
    """Convert a matrix of non-negative values to RGBA using a sequential colormap.

    Maps [vmin, vmax] through dark purple → blue → bright yellow.
    Ideal for probability distributions. Returns (rows, cols, 4).
    """
    if vmax is None:
        vmax = max(float(matrix.max()), 0.01)

    rows, cols = matrix.shape
    colors = np.zeros((rows, cols, 4), dtype=np.float32)

    t = np.clip((matrix - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)

    mask_low = t < 0.5
    s_low = t * 2.0
    s_high = (t - 0.5) * 2.0

    # Dark purple (0.05, 0.0, 0.15) → Blue (0.2, 0.4, 0.9) → Bright yellow (1.0, 0.95, 0.3)
    colors[:, :, 0] = np.where(mask_low, 0.05 + s_low * 0.15, 0.2 + s_high * 0.8)
    colors[:, :, 1] = np.where(mask_low, s_low * 0.4, 0.4 + s_high * 0.55)
    colors[:, :, 2] = np.where(mask_low, 0.15 + s_low * 0.75, 0.9 - s_high * 0.6)
    colors[:, :, 3] = alpha

    return colors


def matrix_to_colors_sequential_per_row(matrix: np.ndarray,
                                         alpha: float = 1.0) -> np.ndarray:
    """Sequential colormap with per-row normalization.

    Each row is normalized independently so its max value maps to bright yellow
    and its min maps to dark purple. Useful for probability matrices where
    different rows have vastly different magnitudes.
    """
    rows, cols = matrix.shape
    colors = np.zeros((rows, cols, 4), dtype=np.float32)
    for i in range(rows):
        row = matrix[i:i+1, :]
        colors[i:i+1] = matrix_to_colors_sequential(row, alpha)
    return colors


def matrix_to_colors_sequential_per_row_fade(matrix: np.ndarray,
                                              alpha: float = 1.0) -> np.ndarray:
    """Per-row sequential colormap; all-zero rows become fully transparent."""
    colors = matrix_to_colors_sequential_per_row(matrix, alpha)
    row_max = np.abs(matrix).max(axis=1)
    colors[row_max < 1e-8, :, 3] = 0.0
    return colors


def matrix_to_colors_sequential_fade(matrix: np.ndarray,
                                      alpha: float = 1.0) -> np.ndarray:
    """Sequential colormap; near-zero elements become fully transparent."""
    colors = matrix_to_colors_sequential(matrix, alpha)
    colors[np.abs(matrix) < 1e-8, 3] = 0.0
    return colors
