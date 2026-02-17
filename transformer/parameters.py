from dataclasses import dataclass


@dataclass
class TransformerConfig:
    d_model: int = 16
    seq_len: int = 16
    num_heads: int = 4
    d_k: int = 4        # d_model // num_heads
    d_ff: int = 128      # Feed-forward hidden dimension
    num_blocks: int = 4  # Number of transformer blocks

    def __post_init__(self):
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")
        expected_d_k = self.d_model // self.num_heads
        if self.d_k != expected_d_k:
            raise ValueError(
                f"d_k ({self.d_k}) must equal d_model // num_heads ({expected_d_k})")
