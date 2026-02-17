"""
GPT Visualizer — Data Sonification
===========================================
Ultra-short sine blips, digital clicks, needle tones,
white noise bursts. Sounds fire ONLY on discrete events
— element arrivals, multiplications, activations.

Pure numpy synthesis → stereo 16-bit WAV.
"""

import os
import argparse
import numpy as np

from audio.common import SR, write_wav
from transformer.parameters import TransformerConfig
from transformer.computation import TransformerBlock
from animation.timeline import AnimationTimeline


# ─── Micro-Sound Palette ────────────────────────────────────────────

def _sine_blip(freq, dur_samples):
    """Ultra-short pure sine blip with hanning window."""
    te = np.linspace(0, dur_samples / SR, dur_samples, endpoint=False)
    return np.sin(2 * np.pi * freq * te) * np.hanning(dur_samples)


def _click(dur_samples, polarity=1.0):
    """Single-sample digital click with ringing decay."""
    c = np.zeros(dur_samples)
    c[0] = polarity
    for i in range(1, min(dur_samples, 8)):
        c[i] = -c[i - 1] * 0.55
    return c


def _needle(freq, dur_samples):
    """High frequency needle tone with exponential decay."""
    freq = min(freq, 5000)
    te = np.linspace(0, dur_samples / SR, dur_samples, endpoint=False)
    return np.sin(2 * np.pi * freq * te) * np.exp(-te * 300)


def _white_burst(dur_samples, rng):
    """Precise white noise burst with hanning window."""
    return rng.randn(dur_samples) * np.hanning(dur_samples)


def _data_tone(f1, f2, dur_samples):
    """Two-frequency FSK data modem tone."""
    te = np.linspace(0, dur_samples / SR, dur_samples, endpoint=False)
    switch = (te * 80).astype(int) % 2
    freq_arr = np.where(switch, f1, f2)
    phase = 2 * np.pi * np.cumsum(freq_arr) / SR
    return np.sin(phase) * np.hanning(dur_samples)


def _freq_sweep(f_start, f_end, dur_samples):
    """Exponential frequency sweep with hanning window."""
    te = np.linspace(0, dur_samples / SR, dur_samples, endpoint=False)
    ratio = max(f_end / max(f_start, 1), 0.01)
    freq = f_start * (ratio ** (te / (dur_samples / SR)))
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return np.sin(phase) * np.hanning(dur_samples)


# ─── Event → Sound Mapping ──────────────────────────────────────────

SOUND_KINDS   = ['needle', 'click', 'sine_blip', 'white_burst', 'data_tone']
SOUND_WEIGHTS = [0.45,     0.35,    0.05,        0.05,          0.10]

# Global RNG for randomizing sound types (seeded for reproducibility)
_event_rng = np.random.RandomState(777)


def _rand_kind():
    """Pick a weighted-random micro-sound type (favors needle & click)."""
    return _event_rng.choice(SOUND_KINDS, p=SOUND_WEIGHTS)


def _rand_dur():
    """Random micro-duration in ms (0.3–8ms)."""
    return _event_rng.uniform(0.3, 8.0)


def _val_to_freq(val, vmax):
    """Map a matrix value to frequency (200–5000 Hz)."""
    normalized = abs(val) / max(vmax, 1e-6)
    return 200 + 4800 * min(normalized, 1.0)


def _val_to_vol(val, vmax, lo=0.05, hi=0.3):
    """Map a matrix value to volume."""
    normalized = abs(val) / max(vmax, 1e-6)
    return lo + (hi - lo) * min(normalized, 1.0)


def _diag_delay(row, col, rows, cols, stagger=0.3):
    """Diagonal wave delay matching the visual animation."""
    diag_max = max(rows + cols - 2, 1)
    return (row + col) / diag_max * stagger


# ─── Event Generators ───────────────────────────────────────────────

def events_appear(matrix, t_start, duration):
    """Element fly-in: random micro-sound per element, diagonal wave timing."""
    rows, cols = matrix.shape
    vmax = max(np.max(np.abs(matrix)), 0.01)
    events = []

    for r in range(rows):
        for c in range(cols):
            val = matrix[r, c]
            delay = _diag_delay(r, c, rows, cols, 0.3)
            t = t_start + duration * (delay + (1.0 - 0.3) * 0.6)
            freq = _val_to_freq(val, vmax)
            vol = _val_to_vol(val, vmax, 0.06, 0.2)
            events.append((t, _rand_kind(), freq, vol, _rand_dur()))

    return events


def events_matmul(A, B, C, t_start, duration):
    """Matmul column sweep: random sounds on B-row hits and C emergence."""
    m, k = A.shape
    _, n = C.shape
    c_vmax = max(np.max(np.abs(C)), 0.01)
    b_vmax = max(np.max(np.abs(B)), 0.01)
    events = []

    col_window = min(3.0 / max(n, 1), 0.5)
    col_stride = (1.0 - col_window) / max(n - 1, 1) if n > 1 else 0.0

    for j in range(n):
        col_start_t = col_stride * j
        col_end_t = col_start_t + col_window

        # Phase 2: B walks down A's rows
        walk_start = col_start_t + col_window * 0.20
        walk_end = col_start_t + col_window * 0.55
        walk_dur = walk_end - walk_start

        for i in range(m):
            row_t = walk_start + walk_dur * (i / max(m - 1, 1))
            t = t_start + duration * row_t

            for p in range(min(k, 4)):
                val = B[p, j]
                freq = _val_to_freq(val, b_vmax)
                vol = _val_to_vol(val, b_vmax, 0.03, 0.12)
                events.append((t + p * 0.002, _rand_kind(), freq, vol, _rand_dur()))

        # Phase 4: C elements emerge
        emerge_start = col_start_t + col_window * 0.65
        emerge_dur = col_window * 0.35

        for i in range(m):
            delay = _diag_delay(i, 0, m, 1, 0.3)
            emerge_t = emerge_start + emerge_dur * (delay + 0.4)
            t = t_start + duration * min(emerge_t, col_end_t - 0.01)

            val = C[i, j]
            freq = _val_to_freq(val, c_vmax)
            vol = _val_to_vol(val, c_vmax, 0.05, 0.18)
            events.append((t, _rand_kind(), freq, vol, _rand_dur()))

    return events


def events_softmax(pre, post, t_start, duration):
    """Softmax: random sound per element transformation."""
    rows, cols = pre.shape
    post_vmax = max(np.max(np.abs(post)), 0.01)
    events = []

    for r in range(rows):
        for c in range(cols):
            delay = _diag_delay(r, c, rows, cols, 0.4)
            t = t_start + duration * (delay + (1.0 - 0.4) * 0.5)

            val = post[r, c]
            freq = _val_to_freq(val, post_vmax)
            vol = _val_to_vol(val, post_vmax, 0.04, 0.15)
            events.append((t, _rand_kind(), freq, vol, _rand_dur()))

    return events


def events_relu(pre, post, t_start, duration):
    """ReLU: killed elements get random sound. Survivors = silence."""
    rows, cols = pre.shape
    pre_vmax = max(np.max(np.abs(pre)), 0.01)
    events = []

    for r in range(rows):
        for c in range(cols):
            delay = _diag_delay(r, c, rows, cols, 0.4)
            t = t_start + duration * (delay + (1.0 - 0.4) * 0.5)

            if post[r, c] == 0 and pre[r, c] < 0:
                vol = _val_to_vol(pre[r, c], pre_vmax, 0.08, 0.25)
                events.append((t, _rand_kind(), _val_to_freq(pre[r, c], pre_vmax), vol, _rand_dur()))

    return events


def events_row_select(probs, pred_pos, t_start, duration):
    """Row selection: prediction row elements sound as they scale by probability."""
    rows, cols = probs.shape
    events = []

    # Phase 1 (0-40%): non-prediction rows fade — quiet downward sweeps
    fade_dur = duration * 0.4
    for r in range(rows):
        if r == pred_pos:
            continue
        row_max = max(np.max(np.abs(probs[r])), 0.01)
        # One sweep per fading row
        t = t_start + fade_dur * (r / max(rows - 1, 1))
        freq = 200 + 800 * (row_max / max(np.max(np.abs(probs)), 0.01))
        events.append((t, 'sweep', freq, 0.06, 4.0, freq * 0.3))

    # Phase 2 (30-100%): prediction row scales up — sound per column
    scale_start = t_start + duration * 0.3
    scale_dur = duration * 0.7
    row = probs[pred_pos]
    for c in range(cols):
        p = row[c]
        if p < 0.003:
            continue
        t = scale_start + scale_dur * (c / max(cols - 1, 1))
        freq = 250 + 4000 * p
        vol = 0.04 + 0.22 * p
        dur_ms = 1.5 + 8.0 * p
        events.append((t, _rand_kind(), freq, vol, dur_ms))

    return events


def events_argmax(probs, pred_pos, t_start, duration):
    """Argmax scan: sound per token proportional to its probability."""
    row = probs[pred_pos]
    cols = len(row)
    events = []

    for c in range(cols):
        p = row[c]
        if p < 0.005:
            continue
        # Sweep left-to-right across columns
        t = t_start + duration * (c / max(cols - 1, 1))
        # Higher probability → higher pitch and louder
        freq = 300 + 4000 * p
        vol = 0.04 + 0.26 * p
        dur_ms = 1.0 + 7.0 * p
        events.append((t, _rand_kind(), freq, vol, dur_ms))

    return events


def events_add(A, B, C, t_start, duration):
    """Residual add: random sounds at fly, merge, and emerge moments."""
    rows, cols = A.shape
    a_vmax = max(np.max(np.abs(A)), 0.01)
    b_vmax = max(np.max(np.abs(B)), 0.01)
    c_vmax = max(np.max(np.abs(C)), 0.01)
    events = []

    for r in range(rows):
        for c in range(cols):
            delay = _diag_delay(r, c, rows, cols, 0.15)

            t_fly = t_start + duration * (delay + 0.1)
            freq_b = _val_to_freq(B[r, c], b_vmax)
            vol_b = _val_to_vol(B[r, c], b_vmax, 0.03, 0.1)
            events.append((t_fly, _rand_kind(), freq_b, vol_b, _rand_dur()))

            t_merge = t_start + duration * (delay + 0.45)
            freq_a = _val_to_freq(A[r, c], a_vmax)
            vol_a = _val_to_vol(A[r, c], a_vmax, 0.02, 0.08)
            events.append((t_merge, _rand_kind(), freq_a, vol_a, _rand_dur()))

            t_emerge = t_start + duration * (delay + 0.65)
            freq_c = _val_to_freq(C[r, c], c_vmax)
            vol_c = _val_to_vol(C[r, c], c_vmax, 0.04, 0.12)
            events.append((t_emerge, _rand_kind(), freq_c, vol_c, _rand_dur()))

    return events


def events_layernorm(pre, post, t_start, duration):
    """LayerNorm: random sounds per element, freq from pre/post values."""
    rows, cols = pre.shape
    pre_vmax = max(np.max(np.abs(pre)), 0.01)
    post_vmax = max(np.max(np.abs(post)), 0.01)
    events = []

    for r in range(rows):
        for c in range(cols):
            delay = _diag_delay(r, c, rows, cols, 0.4)
            t = t_start + duration * (delay + (1.0 - 0.4) * 0.5)

            freq_pre = _val_to_freq(pre[r, c], pre_vmax)
            freq_post = _val_to_freq(post[r, c], post_vmax)
            vol = _val_to_vol(post[r, c], post_vmax, 0.04, 0.14)
            # Randomly pick sweep or another kind
            if _event_rng.random() < 0.4:
                events.append((t, 'sweep', freq_pre, vol, _rand_dur(), freq_post))
            else:
                events.append((t, _rand_kind(), freq_pre, vol, _rand_dur()))

    return events


# ─── Render Events to Audio ─────────────────────────────────────────

def render_events(events, total_samples):
    """Render all events into a mono audio buffer."""
    out = np.zeros(total_samples)
    rng = np.random.RandomState(42)

    for ev in events:
        # Unpack: (time, kind, freq, vol, dur_ms, ...)
        time_s = ev[0]
        kind = ev[1]
        freq = ev[2]
        vol = ev[3]
        dur_ms = ev[4]

        pos = int(time_s * SR)
        dur = max(int(dur_ms * 0.001 * SR), 2)
        dur = min(dur, total_samples - pos)
        if pos < 0 or pos >= total_samples or dur <= 0:
            continue

        if kind == 'sine_blip':
            grain = _sine_blip(freq, dur)
        elif kind == 'click':
            grain = _click(dur, polarity=rng.choice([-1.0, 1.0]))
        elif kind == 'needle':
            grain = _needle(freq, dur)
        elif kind == 'white_burst':
            grain = _white_burst(dur, rng)
        elif kind == 'data_tone':
            f2 = freq * rng.choice([1.5, 2.0, 3.0])
            grain = _data_tone(freq, f2, dur)
        elif kind == 'sweep':
            freq_end = ev[5] if len(ev) > 5 else freq * 0.5
            grain = _freq_sweep(freq, freq_end, dur)
        else:
            continue

        end = min(pos + len(grain), total_samples)
        out[pos:end] += grain[:end - pos] * vol

    return out


# ─── Minimal DSP ──────────────────────────────────────────────────────

def _highpass_simple(x, cutoff=30):
    """Single-pole highpass to remove DC."""
    rc = 1.0 / (2 * np.pi * cutoff)
    dt = 1.0 / SR
    alpha = rc / (rc + dt)
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * (y[i - 1] + x[i] - x[i - 1])
    return y


def _compress(x, threshold=0.12, ratio=4.0, attack_s=0.005, release_s=0.08):
    """Dynamic range compressor to even out volume across stages.

    Loud sections (matmul with many overlapping events) are reduced,
    quiet sections (sparse clicks) are preserved, then peak-normalize
    at the end brings everything to a consistent level.
    """
    n = len(x)
    att = np.exp(-1.0 / max(attack_s * SR, 1))
    rel = np.exp(-1.0 / max(release_s * SR, 1))

    # Envelope follower (peak detector with attack/release smoothing)
    env = np.zeros(n)
    env[0] = abs(x[0])
    for i in range(1, n):
        sample = abs(x[i])
        if sample > env[i - 1]:
            env[i] = att * env[i - 1] + (1.0 - att) * sample
        else:
            env[i] = rel * env[i - 1] + (1.0 - rel) * sample

    # Gain: compress above threshold
    # gain = (env / threshold) ^ (1/ratio - 1)  for env > threshold
    gain = np.ones(n)
    above = env > threshold
    gain[above] = (env[above] / threshold) ** (1.0 / ratio - 1.0)

    return x * gain


# ─── Main Composition ───────────────────────────────────────────────

def build_soundtrack(results, config, timeline=None, logits_only=False):
    """Build the complete data sonification soundtrack from transformer events."""
    if timeline is None:
        timeline = AnimationTimeline()
    total_duration = timeline.total_duration + timeline.return_duration
    total_samples = int(total_duration * SR)

    print(f"  Timeline: {timeline.total_duration:.1f}s + "
          f"{timeline.return_duration:.1f}s return = {total_duration:.1f}s")
    print(f"  Samples: {total_samples} @ {SR}Hz")
    print()

    all_events = []

    for stage in timeline.stages:
        name = stage.stage_name
        t0 = stage.start_time
        appear_dur = stage.appear_duration
        compute_dur = stage.compute_duration
        t_appear = t0
        t_compute = t0 + appear_dur

        print(f"    {name}: t={t0:.1f}s  "
              f"[appear {appear_dur:.1f}s | compute {compute_dur:.1f}s]")

        # ── Input: silent (no sound on initial appear) ──
        if name == 'input':
            pass

        # ── QKV Projection ──
        elif name == 'qkv_projection':
            # Appear: input copies fly in
            all_events += events_appear(results['input'], t_appear, appear_dur)

            # Compute: three parallel matmuls
            if compute_dur > 0:
                for w_key, out_key in [('W_Q', 'Q'), ('W_K', 'K'), ('W_V', 'V')]:
                    all_events += events_matmul(
                        results['input'], results[w_key], results[out_key],
                        t_compute, compute_dur)

        # ── Multi-Head Attention ──
        elif name == 'multi_head_attn':
            nh = config.num_heads
            dk = config.d_k

            # Appear: Q_h and K_h slices fly in from QKV stage
            for h in range(nh):
                q_h = results['Q'].reshape(config.seq_len, nh, dk)[:, h, :]
                k_h = results['K'].reshape(config.seq_len, nh, dk)[:, h, :]
                all_events += events_appear(q_h, t_appear, appear_dur)
                all_events += events_appear(k_h, t_appear, appear_dur)

            for gi, (seg_s, seg_e) in enumerate(stage.group_segments):
                seg_dur = (seg_e - seg_s) * compute_dur
                seg_t = t_compute + seg_s * compute_dur
                if seg_dur <= 0:
                    continue

                for h in range(nh):
                    q_h = results['Q'].reshape(config.seq_len, nh, dk)[:, h, :]
                    k_h = results['K'].reshape(config.seq_len, nh, dk)[:, h, :]
                    v_h = results['V'].reshape(config.seq_len, nh, dk)[:, h, :]

                    if gi == 0:
                        # Q × K^T
                        all_events += events_matmul(
                            q_h, k_h.T, results['attention_scores'][h],
                            seg_t, seg_dur)
                    elif gi == 1:
                        # Softmax
                        all_events += events_softmax(
                            results['attention_scores'][h],
                            results['attention_weights'][h],
                            seg_t, seg_dur)
                    elif gi == 2:
                        # Weights × V
                        all_events += events_matmul(
                            results['attention_weights'][h], v_h,
                            results['head_outputs'][h],
                            seg_t, seg_dur)

        # ── Concat + Output Projection ──
        elif name == 'concat_output_proj':
            all_events += events_appear(results['concat'], t_appear, appear_dur)
            if compute_dur > 0:
                all_events += events_matmul(
                    results['concat'], results['W_O'], results['attn_output'],
                    t_compute, compute_dur)

        # ── Residual + LayerNorm 1 ──
        elif name == 'residual_ln1':
            # Appear: input (skip connection) and attn_output fly in
            all_events += events_appear(results['input'], t_appear, appear_dur)
            all_events += events_appear(results['attn_output'], t_appear, appear_dur)

            for gi, (seg_s, seg_e) in enumerate(stage.group_segments):
                seg_dur = (seg_e - seg_s) * compute_dur
                seg_t = t_compute + seg_s * compute_dur
                if seg_dur <= 0:
                    continue
                if gi == 0:
                    all_events += events_add(
                        results['input'], results['attn_output'],
                        results['residual1'], seg_t, seg_dur)
                elif gi == 1:
                    all_events += events_layernorm(
                        results['residual1'], results['layernorm1'],
                        seg_t, seg_dur)

        # ── FFN ──
        elif name == 'ffn':
            # Appear: layernorm1 output flies in
            all_events += events_appear(results['layernorm1'], t_appear, appear_dur)

            for gi, (seg_s, seg_e) in enumerate(stage.group_segments):
                seg_dur = (seg_e - seg_s) * compute_dur
                seg_t = t_compute + seg_s * compute_dur
                if seg_dur <= 0:
                    continue
                if gi == 0:
                    all_events += events_matmul(
                        results['layernorm1'], results['W1'],
                        results['ffn_pre_relu'], seg_t, seg_dur)
                elif gi == 1:
                    all_events += events_relu(
                        results['ffn_pre_relu'], results['ffn_hidden'],
                        seg_t, seg_dur)
                elif gi == 2:
                    # W2 flies in silently (first 20%), sound only during compute
                    compute_start = seg_t + seg_dur * 0.20
                    compute_len = seg_dur * 0.80
                    all_events += events_matmul(
                        results['ffn_hidden'], results['W2'],
                        results['ffn_output'], compute_start, compute_len)

        # ── Residual + LayerNorm 2 ──
        elif name == 'residual_ln2':
            # Appear: layernorm1 (skip connection) and ffn_output fly in
            all_events += events_appear(results['layernorm1'], t_appear, appear_dur)
            all_events += events_appear(results['ffn_output'], t_appear, appear_dur)

            for gi, (seg_s, seg_e) in enumerate(stage.group_segments):
                seg_dur = (seg_e - seg_s) * compute_dur
                seg_t = t_compute + seg_s * compute_dur
                if seg_dur <= 0:
                    continue
                if gi == 0:
                    all_events += events_add(
                        results['layernorm1'], results['ffn_output'],
                        results['residual2'], seg_t, seg_dur)
                elif gi == 1:
                    all_events += events_layernorm(
                        results['residual2'], results['layernorm2'],
                        seg_t, seg_dur)

        # ── Output: elements appear ──
        elif name == 'output':
            all_events += events_appear(results['output'], t_appear, appear_dur)

        # ── Blocks 2-4: simplified transition ──
        elif name in ('block_2', 'block_3', 'block_4'):
            # stage block_N shows transition: block_{N-2} output → block_{N-1} output
            n = int(name.split('_')[1])
            pre_key = f'block_{n - 2}'
            post_key = f'block_{n - 1}'
            if pre_key in results and 'output' in results[pre_key]:
                all_events += events_appear(
                    results[pre_key]['output'], t_appear, appear_dur)
                if compute_dur > 0 and post_key in results and 'output' in results[post_key]:
                    all_events += events_softmax(
                        results[pre_key]['output'],
                        results[post_key]['output'],
                        t_compute, compute_dur)

        # ── Output Projection: matmul ──
        elif name == 'output_projection':
            if 'W_out' in results and 'logits' in results:
                all_events += events_appear(
                    results['output'], t_appear, appear_dur)
                if compute_dur > 0:
                    all_events += events_matmul(
                        results['output'], results['W_out'],
                        results['logits'], t_compute, compute_dur)

        # ── Token Probabilities: softmax + selection ──
        elif name == 'token_probs':
            if 'logits' in results and 'probs' in results:
                if not logits_only:
                    all_events += events_appear(
                        results['logits'], t_appear, appear_dur)
                for gi, (seg_s, seg_e) in enumerate(stage.group_segments):
                    seg_dur = (seg_e - seg_s) * compute_dur
                    seg_t = t_compute + seg_s * compute_dur
                    if seg_dur <= 0:
                        continue
                    if gi == 0:
                        all_events += events_softmax(
                            results['logits'], results['probs'],
                            seg_t, seg_dur)
                    elif gi == 1:
                        # Row selection: prediction row scales, others fade
                        pred_pos = int(results.get('pred_pos', 0))
                        all_events += events_row_select(
                            results['probs'], pred_pos, seg_t, seg_dur)
                    elif gi == 2:
                        # Argmax: sound proportional to probability
                        pred_pos = int(results.get('pred_pos', 0))
                        probs = results['probs']
                        all_events += events_argmax(
                            probs, pred_pos, seg_t, seg_dur)

        # ── Char Display: minimal appear sound ──
        elif name == 'char_display':
            if appear_dur > 0:
                all_events += events_appear(
                    results['input'][:1, :], t_appear, appear_dur)

    print(f"\n  Total events: {len(all_events)}")
    print("  Rendering audio...")

    mono = render_events(all_events, total_samples)

    # Remove DC
    mono = _highpass_simple(mono, 30)

    # Tame harsh highs with simple single-pole lowpass
    _lp_rc = 1.0 / (2 * np.pi * 6000)
    _lp_dt = 1.0 / SR
    _lp_alpha = _lp_dt / (_lp_rc + _lp_dt)
    lp_out = np.zeros_like(mono)
    lp_out[0] = mono[0]
    for i in range(1, len(mono)):
        lp_out[i] = lp_out[i - 1] + _lp_alpha * (mono[i] - lp_out[i - 1])
    mono = lp_out

    # Compress dynamics so loud/quiet sections are more even
    mono = _compress(mono)

    # Normalize
    peak = np.max(np.abs(mono))
    if peak > 0:
        mono = mono / peak * 0.85

    # Stereo: slight delay + pan variation based on time
    left = mono.copy()
    right = np.zeros_like(mono)
    delay = int(0.0002 * SR)  # ~0.2ms
    right[delay:] = mono[:-delay]

    stereo = np.column_stack([left, right])
    return (stereo * 32767).astype(np.int16)


def main():
    parser = argparse.ArgumentParser(description="Generate data sonification soundtrack")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = args.output or os.path.join(base_dir, "assets", "transformer_sound.wav")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("=" * 55)
    print("  Data Sonification")
    print("=" * 55)
    print()

    config = TransformerConfig()
    transformer = TransformerBlock(config)
    x = np.random.RandomState(123).randn(config.seq_len, config.d_model).astype(np.float32) * 0.5
    results = transformer.forward(x)

    audio = build_soundtrack(results, config)
    write_wav(output_path, audio)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Saved: {output_path} ({file_size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
