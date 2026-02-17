"""Training script for character-level language model (Tiny Shakespeare).

Usage:
    python -m transformer.train
    python -m transformer.train --epochs 50 --lr 3e-3
    python -m transformer.train --check-grad   # verify gradient correctness
"""
import os
import sys
import argparse
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer.parameters import TransformerConfig
from transformer.computation import softmax, layer_norm
from transformer.model import CharLM
from transformer.vocab import (
    VOCAB_SIZE, PAD_ID, CHAR_TO_ID,
    encode_text, decode_ids, make_labels,
)


# ── Data loading ──────────────────────────────────────────────────────

def load_shakespeare() -> str:
    """Load and lowercase Tiny Shakespeare text."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "tiny_shakespeare.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Download tiny_shakespeare.txt first.")
    with open(path) as f:
        return f.read().lower().replace('\n', ' ')


def make_dataset(text: str, seq_len: int):
    """Create input/target pairs from continuous text.

    Splits text into non-overlapping chunks of (seq_len + 1) characters.
    Input:  chunk[:-1]  (first seq_len chars)
    Target: chunk[1:]   (next seq_len chars, shifted by 1)
    Mask:   all ones (every position is valid)
    """
    # Encode entire text (skip unknown characters)
    ids = np.array([CHAR_TO_ID[c] for c in text if c in CHAR_TO_ID], dtype=np.int32)
    n_chunks = len(ids) // (seq_len + 1)
    ids = ids[:n_chunks * (seq_len + 1)]
    chunks = ids.reshape(n_chunks, seq_len + 1)

    inputs = chunks[:, :-1].copy()
    targets = chunks[:, 1:].copy()
    masks = np.ones((n_chunks, seq_len), dtype=np.float32)
    return inputs, targets, masks


# ── Backward pass helpers ─────────────────────────────────────────────

def softmax_backward(dout, sm_out):
    """Backward through softmax. dout, sm_out: (seq, seq) or (seq, vocab)."""
    dot = np.sum(dout * sm_out, axis=-1, keepdims=True)
    return sm_out * (dout - dot)


def layer_norm_backward(dy, x, gamma, eps=1e-5):
    """Backward through layer_norm. Returns dx, dgamma, dbeta."""
    N = x.shape[-1]
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x - mu) * std_inv

    dgamma = np.sum(dy * x_hat, axis=0)
    dbeta = np.sum(dy, axis=0)

    dx_hat = dy * gamma
    dvar = np.sum(dx_hat * (x - mu) * (-0.5) * std_inv**3, axis=-1, keepdims=True)
    dmu = np.sum(dx_hat * (-std_inv), axis=-1, keepdims=True) + \
          dvar * np.mean(-2.0 * (x - mu), axis=-1, keepdims=True)
    dx = dx_hat * std_inv + dvar * 2.0 * (x - mu) / N + dmu / N
    return dx, dgamma, dbeta


def relu_backward(dout, pre_relu):
    """Backward through ReLU."""
    return dout * (pre_relu > 0).astype(dout.dtype)


# ── Per-block forward / backward ─────────────────────────────────────

def _f64(a):
    """Cast to float64 for numerical stability."""
    return np.asarray(a, dtype=np.float64)


def _block_forward(x, W_Q, W_K, W_V, W_O, W1, b1, W2, b2,
                   gamma1, beta1, gamma2, beta2,
                   seq_len, nh, dk, causal_mask):
    """Forward pass through a single transformer block.
    Returns (output, intermediates_dict)."""
    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V
    Q_h = Q.reshape(seq_len, nh, dk)
    K_h = K.reshape(seq_len, nh, dk)
    V_h = V.reshape(seq_len, nh, dk)

    weights_list = []
    head_outs = []
    for h in range(nh):
        q = Q_h[:, h, :]
        k = K_h[:, h, :]
        v = V_h[:, h, :]
        score = q @ k.T / np.sqrt(float(dk))
        score = score - causal_mask * 1e9
        w = softmax(score, axis=-1)
        out = w @ v
        weights_list.append(w)
        head_outs.append(out)

    concat = np.concatenate(head_outs, axis=-1)
    attn_out = concat @ W_O
    residual1 = x + attn_out
    ln1 = layer_norm(residual1, gamma1, beta1)
    ffn_pre = ln1 @ W1 + b1
    ffn_relu = np.maximum(0, ffn_pre)
    ffn_out = ffn_relu @ W2 + b2
    residual2 = ln1 + ffn_out
    ln2 = layer_norm(residual2, gamma2, beta2)

    intermediates = {
        'x': x, 'Q': Q, 'K': K, 'V': V,
        'Q_h': Q_h, 'K_h': K_h, 'V_h': V_h,
        'weights_list': weights_list, 'concat': concat,
        'residual1': residual1, 'ln1': ln1,
        'ffn_pre': ffn_pre, 'ffn_relu': ffn_relu,
        'residual2': residual2,
    }
    return ln2, intermediates


def _block_backward(d_output, intermediates, W_Q, W_K, W_V, W_O,
                    W1, b1, W2, b2, gamma1, beta1, gamma2, beta2,
                    seq_len, nh, dk, causal_mask):
    """Backward pass through a single transformer block.
    Returns (dx, block_grads_list)."""
    x = intermediates['x']
    Q, K, V = intermediates['Q'], intermediates['K'], intermediates['V']
    Q_h, K_h, V_h = intermediates['Q_h'], intermediates['K_h'], intermediates['V_h']
    weights_list = intermediates['weights_list']
    concat = intermediates['concat']
    residual1 = intermediates['residual1']
    ln1 = intermediates['ln1']
    ffn_pre = intermediates['ffn_pre']
    ffn_relu = intermediates['ffn_relu']
    residual2 = intermediates['residual2']

    # LayerNorm 2 backward
    dresidual2, dgamma2, dbeta2 = layer_norm_backward(d_output, residual2, gamma2)

    # Residual 2: residual2 = ln1 + ffn_out
    dln1_from_res2 = dresidual2.copy()
    dffn_out = dresidual2.copy()

    # FFN backward
    dW2 = ffn_relu.T @ dffn_out
    db2 = dffn_out.sum(axis=0)
    dffn_relu = dffn_out @ W2.T
    dffn_pre = relu_backward(dffn_relu, ffn_pre)
    dW1 = ln1.T @ dffn_pre
    db1 = dffn_pre.sum(axis=0)
    dln1_from_ffn = dffn_pre @ W1.T

    dln1 = dln1_from_res2 + dln1_from_ffn

    # LayerNorm 1 backward
    dresidual1, dgamma1, dbeta1 = layer_norm_backward(dln1, residual1, gamma1)

    dx_from_res1 = dresidual1.copy()
    dattn_out = dresidual1.copy()

    # Output projection: attn_out = concat @ W_O
    dW_O = concat.T @ dattn_out
    dconcat = dattn_out @ W_O.T
    dconcat_heads = dconcat.reshape(seq_len, nh, dk)

    dQ = np.zeros_like(Q)
    dK = np.zeros_like(K)
    dV = np.zeros_like(V)

    for h in range(nh):
        dout_h = dconcat_heads[:, h, :]
        w_h = weights_list[h]
        v_h = V_h[:, h, :]
        q_h = Q_h[:, h, :]
        k_h = K_h[:, h, :]

        dw_h = dout_h @ v_h.T
        dv_h = w_h.T @ dout_h
        dscore_h = softmax_backward(dw_h, w_h)
        dscore_h *= (1.0 - causal_mask)

        scale = 1.0 / np.sqrt(float(dk))
        dq_h = dscore_h @ k_h * scale
        dk_h = dscore_h.T @ q_h * scale

        dQ[:, h * dk:(h + 1) * dk] = dq_h
        dK[:, h * dk:(h + 1) * dk] = dk_h
        dV[:, h * dk:(h + 1) * dk] = dv_h

    dW_Q = x.T @ dQ
    dW_K = x.T @ dK
    dW_V = x.T @ dV
    dx_from_qkv = dQ @ W_Q.T + dK @ W_K.T + dV @ W_V.T

    dx = dx_from_res1 + dx_from_qkv

    block_grads = [
        dW_Q, dW_K, dW_V, dW_O,
        dW1, db1, dW2, db2,
        dgamma1, dbeta1, dgamma2, dbeta2,
    ]
    return dx, block_grads


# ── Full forward + backward ──────────────────────────────────────────

def forward_backward(model: CharLM, token_ids: np.ndarray,
                     targets: np.ndarray, mask: np.ndarray):
    """Run forward + backward pass for a single example.

    All computation is done in float64 for gradient precision.
    """
    config = model.config
    seq_len = config.seq_len
    d = config.d_model
    nh = config.num_heads
    dk = config.d_k

    embedding = _f64(model.embedding)
    pos_enc = _f64(model.pos_encoding)
    mask64 = _f64(mask)

    # Upcast per-block parameters
    block_params = []
    for block in model.blocks:
        bp = (
            _f64(block.W_Q), _f64(block.W_K), _f64(block.W_V), _f64(block.W_O),
            _f64(block.W1), _f64(block.b1), _f64(block.W2), _f64(block.b2),
            _f64(block.gamma1), _f64(block.beta1), _f64(block.gamma2), _f64(block.beta2),
        )
        block_params.append(bp)

    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float64), k=1)

    # ── Forward pass ──

    embedded = embedding[token_ids]
    x = embedded + pos_enc

    block_intermediates = []
    for bp in block_params:
        x, intermediates = _block_forward(x, *bp, seq_len, nh, dk, causal_mask)
        block_intermediates.append(intermediates)

    # Output head (weight tying: logits = last_output @ embedding.T)
    final_output = x  # last block's ln2
    logits = final_output @ embedding.T
    probs = softmax(logits, axis=-1)

    # ── Loss (cross-entropy, masked) ──
    num_valid = max(mask64.sum(), 1.0)
    log_probs = np.log(probs[np.arange(seq_len), targets] + 1e-10)
    loss = float(-np.sum(log_probs * mask64) / num_valid)

    # ── Backward pass ──

    # dlogits
    dlogits = probs.copy()
    dlogits[np.arange(seq_len), targets] -= 1.0
    dlogits *= mask64[:, None] / num_valid

    # Output head backward (weight tying)
    demb_from_output = (final_output.T @ dlogits).T   # (vocab, d)
    dx = dlogits @ embedding                           # (seq, d)

    # Backward through blocks in reverse
    all_block_grads = []
    for i in range(len(block_params) - 1, -1, -1):
        dx, block_grads = _block_backward(
            dx, block_intermediates[i], *block_params[i],
            seq_len, nh, dk, causal_mask)
        all_block_grads.insert(0, block_grads)

    # Embedding + pos encoding gradients
    dpos = dx.copy()
    demb = np.zeros((model.vocab_size, d), dtype=np.float64)
    np.add.at(demb, token_ids, dx)
    demb += demb_from_output

    # Collect gradients: embedding, pos_encoding, then per-block × num_blocks
    grads = [demb, dpos]
    for bg in all_block_grads:
        grads.extend(bg)

    return loss, grads


# ── Batched training step ─────────────────────────────────────────────

def train_step(model: CharLM, batch_ids: np.ndarray,
               batch_targets: np.ndarray, batch_masks: np.ndarray):
    """Compute average loss and gradients over a batch."""
    batch_size = batch_ids.shape[0]
    total_loss = 0.0
    accum_grads = None

    for i in range(batch_size):
        loss, grads = forward_backward(
            model, batch_ids[i], batch_targets[i], batch_masks[i])
        total_loss += loss
        if accum_grads is None:
            accum_grads = [g.copy() for g in grads]
        else:
            for j, g in enumerate(grads):
                accum_grads[j] += g

    avg_loss = total_loss / batch_size
    avg_grads = [g / batch_size for g in accum_grads]
    return avg_loss, avg_grads


# ── Adam optimizer ────────────────────────────────────────────────────

class Adam:
    def __init__(self, params: list[np.ndarray], lr=3e-3,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, params: list[np.ndarray], grads: list[np.ndarray],
             max_norm: float = 1.0):
        """Update params in-place. Returns global gradient norm."""
        global_norm = np.sqrt(sum(np.sum(g**2) for g in grads))
        if global_norm > max_norm:
            scale = max_norm / (global_norm + 1e-10)
            grads = [g * scale for g in grads]

        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return global_norm


# ── Gradient checking ─────────────────────────────────────────────────

def check_gradients(model: CharLM, token_ids: np.ndarray,
                    targets: np.ndarray, mask: np.ndarray,
                    eps=1e-5, tol=1e-3):
    """Compare analytical gradients with numerical gradients."""
    loss, anal_grads = forward_backward(model, token_ids, targets, mask)
    params = model.get_params()

    # Build parameter names
    names = ['embedding', 'pos_encoding']
    for i in range(model.config.num_blocks):
        for pname in ['W_Q', 'W_K', 'W_V', 'W_O',
                      'W1', 'b1', 'W2', 'b2',
                      'gamma1', 'beta1', 'gamma2', 'beta2']:
            names.append(f'b{i}_{pname}')

    print(f"  Checking {len(params)} parameter groups...")

    all_ok = True
    for idx, (p, ag, name) in enumerate(zip(params, anal_grads, names)):
        n_check = min(10, p.size)
        check_rng = np.random.RandomState(idx)
        flat_indices = check_rng.choice(p.size, n_check, replace=False)

        max_rel_err = 0.0
        for fi in flat_indices:
            multi_idx = np.unravel_index(fi, p.shape)
            old_val = p[multi_idx]

            p[multi_idx] = old_val + eps
            model.set_params(params)
            loss_plus, _ = forward_backward(model, token_ids, targets, mask)

            p[multi_idx] = old_val - eps
            model.set_params(params)
            loss_minus, _ = forward_backward(model, token_ids, targets, mask)

            p[multi_idx] = old_val
            model.set_params(params)

            num_grad = (loss_plus - loss_minus) / (2 * eps)
            ana_grad = ag[multi_idx]

            denom = max(abs(num_grad) + abs(ana_grad), 1e-8)
            rel_err = abs(num_grad - ana_grad) / denom
            max_rel_err = max(max_rel_err, rel_err)

        status = "OK" if max_rel_err < tol else "FAIL"
        if max_rel_err >= tol:
            all_ok = False
        print(f"    {name:15s}  max_rel_err={max_rel_err:.2e}  [{status}]")

    return all_ok


# ── Main training loop ────────────────────────────────────────────────

def train(args):
    config = TransformerConfig()
    model = CharLM(config)

    print("Loading Tiny Shakespeare...")
    text = load_shakespeare()
    print(f"  {len(text):,} characters")

    # Split train/val (90% / 10%)
    split = int(len(text) * 0.9)
    train_text = text[:split]
    val_text = text[split:]

    train_ids, train_tgt, train_mask = make_dataset(train_text, config.seq_len)
    val_ids, val_tgt, val_mask = make_dataset(val_text, config.seq_len)

    print(f"  Train chunks: {len(train_ids)}, Val chunks: {len(val_ids)}")
    print(f"  Config: d_model={config.d_model}, seq_len={config.seq_len}, "
          f"heads={config.num_heads}, d_ff={config.d_ff}, blocks={config.num_blocks}")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print()

    # Gradient check
    if args.check_grad:
        print("=== Gradient Check ===")
        test_ids = train_ids[0]
        test_tgt = train_tgt[0]
        test_mask = train_mask[0]
        ok = check_gradients(model, test_ids, test_tgt, test_mask)
        print(f"\n  {'All checks passed!' if ok else 'Some checks FAILED!'}")
        if not ok:
            return
        print()

    # Training
    rng = np.random.RandomState(0)
    params = model.get_params()
    optimizer = Adam(params, lr=args.lr)
    n_train = len(train_ids)
    best_val_loss = float('inf')

    print(f"=== Training: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"lr={args.lr} ===\n")

    warmup_epochs = max(1, args.epochs // 15)

    for epoch in range(1, args.epochs + 1):
        # Linear warmup + cosine decay
        if epoch <= warmup_epochs:
            lr = args.lr * epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / max(args.epochs - warmup_epochs, 1)
            lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (1 + math.cos(math.pi * progress))
        optimizer.lr = lr

        # Shuffle training data
        perm = rng.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, args.batch_size):
            end = min(start + args.batch_size, n_train)
            idx = perm[start:end]
            batch_loss, grads = train_step(
                model, train_ids[idx], train_tgt[idx], train_mask[idx])
            optimizer.step(params, grads, max_norm=args.max_norm)
            epoch_loss += batch_loss
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validation loss
        val_loss = 0.0
        n_val_batches = 0
        for start in range(0, len(val_ids), args.batch_size):
            end = min(start + args.batch_size, len(val_ids))
            batch_loss = 0.0
            for i in range(start, end):
                l, _ = forward_backward(model, val_ids[i], val_tgt[i], val_mask[i])
                batch_loss += l
            val_loss += batch_loss / (end - start)
            n_val_batches += 1
        avg_val_loss = val_loss / max(n_val_batches, 1)

        # Accuracy on validation
        n_correct = 0
        n_total = 0
        for i in range(len(val_ids)):
            r = model.forward(val_ids[i])
            pred = r['predicted_ids']
            valid = val_mask[i] > 0
            n_correct += np.sum((pred == val_tgt[i]) & valid)
            n_total += np.sum(valid)
        accuracy = n_correct / max(n_total, 1) * 100

        print(f"  Epoch {epoch:3d}/{args.epochs}  "
              f"train_loss={avg_train_loss:.3f}  "
              f"val_loss={avg_val_loss:.3f}  "
              f"acc={accuracy:.1f}%  "
              f"lr={lr:.1e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "weights", "charlm.npz")
            model.save(save_path)

        sys.stdout.flush()

    print(f"\nDone. Best val_loss={best_val_loss:.3f}")
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "weights", "charlm.npz")
    print(f"Weights saved to {save_path}")

    # Show sample predictions
    print("\n=== Sample Predictions ===")
    for i in range(min(5, len(val_ids))):
        ids = val_ids[i]
        r = model.forward(ids)
        pred_ids = r['predicted_ids']
        inp_str = decode_ids(ids)[:20]
        pred_str = decode_ids(pred_ids)[:20]
        print(f"  in: {repr(inp_str):22s} | pred: {repr(pred_str)}")


def main():
    parser = argparse.ArgumentParser(description="Train character-level LM")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--check-grad", action="store_true",
                        help="Run gradient check before training")
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
