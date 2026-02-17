import os
import zlib
import numpy as np
from .parameters import TransformerConfig
from .computation import TransformerBlock, softmax
from .vocab import VOCAB_SIZE


# Per-block parameter names (in order)
_BLOCK_PARAM_NAMES = [
    'W_Q', 'W_K', 'W_V', 'W_O',
    'W1', 'b1', 'W2', 'b2',
    'gamma1', 'beta1', 'gamma2', 'beta2',
]


class CharLM:
    """Character-level language model: Embedding + N TransformerBlocks + Output head."""

    def __init__(self, config: TransformerConfig, vocab_size: int = VOCAB_SIZE):
        self.config = config
        self.vocab_size = vocab_size
        rng = np.random.RandomState(42)

        d = config.d_model
        # Embedding: (vocab_size, d_model)
        self.embedding = rng.randn(vocab_size, d).astype(np.float32) * 0.1
        # Learned positional encoding: (seq_len, d_model)
        self.pos_encoding = rng.randn(config.seq_len, d).astype(np.float32) * 0.1
        # Transformer blocks (different seed per block for distinct weights)
        self.blocks = [TransformerBlock(config, seed=42 + i) for i in range(config.num_blocks)]

    def forward(self, token_ids: np.ndarray, temperature: float = 0.0,
                seed: int = None) -> dict:
        """Full forward pass. Returns results dict compatible with Scene.

        Block 0's results are stored at the top level (backward compat with Scene).
        All blocks' results are also stored as results['block_0'], ['block_1'], etc.
        """
        embedded = self.embedding[token_ids]    # (seq_len, d_model)
        x = embedded + self.pos_encoding        # (seq_len, d_model)

        # Run through all blocks
        block_results = []
        for i, block in enumerate(self.blocks):
            r = block.forward(x, causal=True)
            block_results.append(r)
            x = r['output']

        # Block 0's results at top level (for Scene backward compat)
        results = dict(block_results[0])
        # Store all blocks' results
        for i, r in enumerate(block_results):
            results[f'block_{i}'] = r
        # Final output is last block's output
        results['output'] = block_results[-1]['output']

        output = results['output']              # (seq_len, d_model)
        logits = output @ self.embedding.T      # (seq_len, vocab_size) — weight tying

        if temperature > 0:
            probs = softmax(logits / temperature, axis=-1)
        else:
            probs = softmax(logits, axis=-1)

        results['token_ids'] = token_ids.copy()
        results['embedded'] = embedded.copy()
        results['logits'] = logits.copy()
        results['probs'] = probs.copy()
        results['W_out'] = self.embedding.T.copy()

        if temperature > 0:
            base_seed = zlib.crc32(token_ids.tobytes()) & 0xFFFFFFFF
            if seed is not None:
                base_seed = (base_seed ^ seed) & 0xFFFFFFFF
            sample_rng = np.random.RandomState(base_seed)
            predicted_ids = np.array([
                sample_rng.choice(self.vocab_size, p=probs[i])
                for i in range(self.config.seq_len)
            ], dtype=np.int32)
        else:
            predicted_ids = np.argmax(logits, axis=-1)
        results['predicted_ids'] = predicted_ids

        # Selection matrix: 1.0 for selected token per row
        selection = np.zeros_like(probs)
        for i in range(self.config.seq_len):
            selection[i, int(predicted_ids[i])] = 1.0
        results['selection'] = selection
        return results

    def get_params(self) -> list[np.ndarray]:
        """Return all trainable parameters as a flat list.
        Order: embedding, pos_encoding, then per-block params × num_blocks.
        """
        params = [self.embedding, self.pos_encoding]
        for block in self.blocks:
            params.extend([
                block.W_Q, block.W_K, block.W_V, block.W_O,
                block.W1, block.b1, block.W2, block.b2,
                block.gamma1, block.beta1, block.gamma2, block.beta2,
            ])
        return params

    def set_params(self, params: list[np.ndarray]):
        """Set all trainable parameters from a flat list."""
        self.embedding = params[0]
        self.pos_encoding = params[1]
        idx = 2
        for block in self.blocks:
            (block.W_Q, block.W_K, block.W_V, block.W_O,
             block.W1, block.b1, block.W2, block.b2,
             block.gamma1, block.beta1, block.gamma2, block.beta2) = params[idx:idx+12]
            idx += 12

    def save(self, path: str):
        """Save all parameters to .npz file."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            'embedding': self.embedding,
            'pos_encoding': self.pos_encoding,
        }
        for i, block in enumerate(self.blocks):
            for name in _BLOCK_PARAM_NAMES:
                data[f'block{i}_{name}'] = getattr(block, name)
        np.savez(path, **data)

    def load(self, path: str):
        """Load parameters from .npz file."""
        data = np.load(path)
        self.embedding = data['embedding']
        self.pos_encoding = data['pos_encoding']
        for i, block in enumerate(self.blocks):
            for name in _BLOCK_PARAM_NAMES:
                setattr(block, name, data[f'block{i}_{name}'])
