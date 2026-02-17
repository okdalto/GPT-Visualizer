import numpy as np
from .parameters import TransformerConfig


def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(var + eps) + beta


class TransformerBlock:
    def __init__(self, config: TransformerConfig, seed: int = 42):
        self.config = config
        rng = np.random.RandomState(seed)

        d = config.d_model
        # Xavier initialization: scale = sqrt(2 / (fan_in + fan_out))
        attn_scale = np.sqrt(2.0 / (d + d))
        self.W_Q = rng.randn(d, d).astype(np.float32) * attn_scale
        self.W_K = rng.randn(d, d).astype(np.float32) * attn_scale
        self.W_V = rng.randn(d, d).astype(np.float32) * attn_scale
        self.W_O = rng.randn(d, d).astype(np.float32) * attn_scale

        ffn_scale1 = np.sqrt(2.0 / (d + config.d_ff))
        ffn_scale2 = np.sqrt(2.0 / (config.d_ff + d))
        self.W1 = rng.randn(d, config.d_ff).astype(np.float32) * ffn_scale1
        self.b1 = np.zeros(config.d_ff, dtype=np.float32)
        self.W2 = rng.randn(config.d_ff, d).astype(np.float32) * ffn_scale2
        self.b2 = np.zeros(d, dtype=np.float32)

        self.gamma1 = np.ones(d, dtype=np.float32)
        self.beta1 = np.zeros(d, dtype=np.float32)
        self.gamma2 = np.ones(d, dtype=np.float32)
        self.beta2 = np.zeros(d, dtype=np.float32)

    def forward(self, x: np.ndarray, causal: bool = False) -> dict:
        results = {}
        results['input'] = x.copy()

        # QKV projections
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V
        results['Q'] = Q.copy()
        results['K'] = K.copy()
        results['V'] = V.copy()
        results['W_Q'] = self.W_Q.copy()
        results['W_K'] = self.W_K.copy()
        results['W_V'] = self.W_V.copy()

        nh = self.config.num_heads
        dk = self.config.d_k

        # Split into heads
        Q_heads = Q.reshape(self.config.seq_len, nh, dk)
        K_heads = K.reshape(self.config.seq_len, nh, dk)
        V_heads = V.reshape(self.config.seq_len, nh, dk)

        attention_scores = []
        attention_weights = []
        head_outputs = []

        for h in range(nh):
            q_h = Q_heads[:, h, :]  # (seq, dk)
            k_h = K_heads[:, h, :]
            v_h = V_heads[:, h, :]

            score = q_h @ k_h.T / np.sqrt(float(dk))
            if causal:
                causal_mask = np.triu(
                    np.ones((self.config.seq_len, self.config.seq_len),
                            dtype=np.float32), k=1)
                score = score - causal_mask * 1e9
            weight = softmax(score, axis=-1)
            out = weight @ v_h

            attention_scores.append(score.copy())
            attention_weights.append(weight.copy())
            head_outputs.append(out.copy())

        results['attention_scores'] = attention_scores
        results['attention_weights'] = attention_weights
        results['head_outputs'] = head_outputs
        if causal:
            results['causal_mask'] = np.triu(
                np.ones((self.config.seq_len, self.config.seq_len),
                        dtype=np.float32), k=1)

        # Concatenate heads
        concat = np.concatenate(head_outputs, axis=-1)
        results['concat'] = concat.copy()

        # Output projection
        attn_out = concat @ self.W_O
        results['attn_output'] = attn_out.copy()
        results['W_O'] = self.W_O.copy()

        # Residual + LayerNorm 1
        residual1 = x + attn_out
        ln1 = layer_norm(residual1, self.gamma1, self.beta1)
        results['residual1'] = residual1.copy()
        results['layernorm1'] = ln1.copy()

        # FFN
        ffn_hidden = ln1 @ self.W1 + self.b1
        results['ffn_pre_relu'] = ffn_hidden.copy()
        ffn_hidden = np.maximum(0, ffn_hidden)
        results['ffn_hidden'] = ffn_hidden.copy()
        ffn_out = ffn_hidden @ self.W2 + self.b2
        results['ffn_output'] = ffn_out.copy()
        results['W1'] = self.W1.copy()
        results['W2'] = self.W2.copy()

        # Residual + LayerNorm 2
        residual2 = ln1 + ffn_out
        ln2 = layer_norm(residual2, self.gamma2, self.beta2)
        results['residual2'] = residual2.copy()
        results['layernorm2'] = ln2.copy()
        results['output'] = ln2.copy()

        return results
