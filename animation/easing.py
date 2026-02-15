import numpy as np


def ease_in_out_cubic(t: float) -> float:
    t = np.clip(t, 0.0, 1.0)
    if t < 0.5:
        return 4.0 * t * t * t
    else:
        return 1.0 - (-2.0 * t + 2.0) ** 3 / 2.0


def ease_out_cubic(t: float) -> float:
    t = np.clip(t, 0.0, 1.0)
    return 1.0 - (1.0 - t) ** 3


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)
