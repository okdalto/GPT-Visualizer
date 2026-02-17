"""
Industrial Generative Electronic Music v2
=================================================
INDUSTRIAL / MACHINE variant.
Blazing fast tempo, machine-gun beats, metallic percussion,
industrial textures, cellular automata glitches, FM aggression.

Pure numpy synthesis -> stereo WAV.
"""

import numpy as np
import os

from audio.common import SR, write_wav

# Seeded RNG for reproducible audio synthesis
_rng = np.random.RandomState(12345)

# ─── Global Parameters ───────────────────────────────────────────────
BPM = 155            # FAST industrial tempo
DURATION = 120
BEAT = 60.0 / BPM
BAR = BEAT * 4

total_samples = int(SR * DURATION)


# ─── DSP Utilities ───────────────────────────────────────────────────

def soft_clip(x, threshold=0.8):
    return np.tanh(x / threshold) * threshold


def hard_clip(x, level=1.0):
    return np.clip(x, -level, level)


def env_adsr(length, attack=0.01, decay=0.05, sustain=0.7, release=0.1, sr=SR):
    a = int(attack * sr)
    d = int(decay * sr)
    r = int(release * sr)
    s = max(0, length - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, max(a, 1)),
        np.linspace(1, sustain, max(d, 1)),
        np.ones(max(s, 0)) * sustain,
        np.linspace(sustain, 0, max(r, 1))
    ])
    return env[:length]


def biquad_coeffs(filter_type, fc, sr=SR, Q=0.707):
    w0 = 2 * np.pi * fc / sr
    alpha = np.sin(w0) / (2 * Q)
    if filter_type == 'lowpass':
        b0 = (1 - np.cos(w0)) / 2
        b1 = 1 - np.cos(w0)
        b2 = (1 - np.cos(w0)) / 2
    elif filter_type == 'highpass':
        b0 = (1 + np.cos(w0)) / 2
        b1 = -(1 + np.cos(w0))
        b2 = (1 + np.cos(w0)) / 2
    elif filter_type == 'bandpass':
        b0 = alpha
        b1 = 0
        b2 = -alpha
    else:
        raise ValueError(filter_type)
    a0 = 1 + alpha
    a1 = -2 * np.cos(w0)
    a2 = 1 - alpha
    return (np.array([b0, b1, b2]) / a0,
            np.array([1.0, a1 / a0, a2 / a0]))


def apply_biquad(x, b, a):
    y = np.zeros_like(x)
    z1, z2 = 0.0, 0.0
    for i in range(len(x)):
        inp = x[i]
        out = b[0] * inp + z1
        z1 = b[1] * inp - a[1] * out + z2
        z2 = b[2] * inp - a[2] * out
        y[i] = out
    return y


def lowpass(x, cutoff, sr=SR, order=2):
    cutoff = np.clip(cutoff, 30, sr / 2 - 100)
    b, a = biquad_coeffs('lowpass', cutoff, sr, Q=0.707)
    y = apply_biquad(x, b, a)
    if order >= 4:
        y = apply_biquad(y, b, a)
    return y


def highpass(x, cutoff, sr=SR, order=2):
    cutoff = np.clip(cutoff, 30, sr / 2 - 100)
    b, a = biquad_coeffs('highpass', cutoff, sr, Q=0.707)
    y = apply_biquad(x, b, a)
    if order >= 4:
        y = apply_biquad(y, b, a)
    return y


def bandpass(x, low, high, sr=SR):
    cutoff = np.clip((low + high) / 2, 30, sr / 2 - 100)
    Q = cutoff / max(high - low, 50)
    Q = np.clip(Q, 0.3, 10)
    b, a = biquad_coeffs('bandpass', cutoff, sr, Q=Q)
    return apply_biquad(x, b, a)


def resonant_lp(x, cutoff, resonance=3.0, sr=SR):
    cutoff = np.clip(cutoff, 30, sr / 2 - 100)
    b, a = biquad_coeffs('lowpass', cutoff, sr, Q=resonance)
    return apply_biquad(x, b, a)


def sawtooth(phase):
    return 2.0 * (phase / (2 * np.pi) % 1.0) - 1.0


def square_wave(phase):
    return np.sign(np.sin(phase))


def pulse_wave(phase, width=0.3):
    return np.where((phase / (2 * np.pi) % 1.0) < width, 1.0, -1.0)


def reverb_simple(x, decay=0.3, delays_ms=(23, 37, 53, 71, 97)):
    out = x.copy()
    for d in delays_ms:
        ds = int(d * SR / 1000)
        if ds < len(x):
            delayed = np.zeros_like(x)
            delayed[ds:] = x[:-ds] * decay
            out += delayed
    return out * 0.6


def delay_effect(x, delay_time=0.375, feedback=0.4, mix=0.3):
    ds = int(delay_time * SR)
    buf = np.zeros_like(x)
    for i in range(4):
        offset = ds * (i + 1)
        if offset < len(x):
            tap = np.zeros_like(x)
            tap[offset:] = x[:-offset] * (feedback ** (i + 1))
            buf += tap
    return x * (1 - mix) + buf * mix


def bitcrush(x, bits=6, downsample=4):
    """Lo-fi digital degradation"""
    levels = 2 ** bits
    crushed = np.round(x * levels) / levels
    if downsample > 1:
        idx = np.arange(len(crushed))
        crushed = crushed[(idx // downsample) * downsample]
    return crushed


# ─── INDUSTRIAL KICK ─────────────────────────────────────────────────

def synth_kick(length_s=0.25):
    """Hard, punchy industrial kick - shorter for fast tempo"""
    n = int(length_s * SR)
    t = np.linspace(0, length_s, n, endpoint=False)

    # Aggressive pitch sweep: 200Hz -> 45Hz
    freq_env = 45 + 155 * np.exp(-t * 50)
    phase = 2 * np.pi * np.cumsum(freq_env) / SR
    osc = np.sin(phase)

    # Distortion harmonics
    osc += 0.3 * np.sin(phase * 2) * np.exp(-t * 60)
    osc += 0.15 * np.sin(phase * 3) * np.exp(-t * 80)
    osc = soft_clip(osc * 1.5, 0.7)

    # Click transient
    click = _rng.randn(n) * np.exp(-t * 800) * 0.4

    amp = np.exp(-t * 12) * (1 - np.exp(-t * 800))
    sub = np.sin(2 * np.pi * 45 * t) * np.exp(-t * 10) * 0.4

    kick = (osc * amp + sub + click)
    return lowpass(kick, 10000)


# ─── INDUSTRIAL HI-HAT ──────────────────────────────────────────────

def synth_hihat(length_s=0.03, open_hat=False):
    """Ultra-short metallic machine hi-hat"""
    n = int(length_s * SR)
    t = np.linspace(0, length_s, n, endpoint=False)

    noise = _rng.randn(n)

    # Harsh metallic resonances
    metallic = np.zeros(n)
    for freq in [317.0, 523.3, 739.9, 1003.0, 1511.0, 1897.0, 2543.0, 3917.0]:
        metallic += square_wave(2 * np.pi * freq * t) * 0.12

    raw = noise * 0.5 + metallic * 0.5
    filtered = highpass(raw, 5000)

    if open_hat:
        amp = np.exp(-t * 20) * (1 - np.exp(-t * 3000))
    else:
        amp = np.exp(-t * 100) * (1 - np.exp(-t * 3000))

    return filtered * amp


# ─── METALLIC PERCUSSION ─────────────────────────────────────────────

def synth_metal_hit(length_s=0.08, freq_base=800):
    """Industrial metallic percussion hit"""
    n = int(length_s * SR)
    t = np.linspace(0, length_s, n, endpoint=False)

    # Inharmonic metallic partials (non-integer ratios)
    metal = np.zeros(n)
    ratios = [1.0, 1.47, 2.09, 2.56, 3.14, 4.13, 5.78]
    for i, r in enumerate(ratios):
        f = freq_base * r
        if f < SR / 2 - 200:
            metal += np.sin(2 * np.pi * f * t) * (0.3 / (i + 1))

    amp = np.exp(-t * 40) * (1 - np.exp(-t * 2000))

    # Ring modulation for extra harshness
    ring = np.sin(2 * np.pi * 137 * t)
    metal = metal * 0.7 + metal * ring * 0.3

    return highpass(metal * amp, 500)


def synth_machine_noise(length_s=0.05):
    """Short burst of filtered, bitcrushed noise"""
    n = int(length_s * SR)
    t = np.linspace(0, length_s, n, endpoint=False)

    noise = _rng.randn(n)
    # Resonant bandpass
    filtered = bandpass(noise, 2000, 5000)
    # Bitcrush for digital grit
    crushed = bitcrush(filtered, bits=5, downsample=3)

    amp = np.exp(-t * 60) * (1 - np.exp(-t * 2000))
    return crushed * amp


# ─── CLAP ────────────────────────────────────────────────────────────

def synth_clap(length_s=0.1):
    """Tight industrial clap"""
    n = int(length_s * SR)
    t = np.linspace(0, length_s, n, endpoint=False)

    noise = _rng.randn(n)
    filtered = bandpass(noise, 1200, 7000)

    env = np.zeros(n)
    for offset in [0, 0.006, 0.012, 0.018]:
        idx = int(offset * SR)
        burst_len = min(int(0.003 * SR), n - idx)
        if idx < n:
            env[idx:idx + burst_len] += 1.0

    env += np.exp(-t * 40) * 0.5
    env *= (1 - np.exp(-t * 4000))

    return filtered * env


# ─── BASS SYNTH ──────────────────────────────────────────────────────

def synth_bass_note(freq, length_s, filter_env_amount=3000):
    """Aggressive distorted FM bass"""
    n = int(length_s * SR)
    t = np.linspace(0, length_s, n, endpoint=False)

    # Aggressive FM
    mod_depth = freq * 4.0 * np.exp(-t * 6)
    modulator = np.sin(2 * np.pi * freq * t) * mod_depth

    carrier = np.sin(2 * np.pi * freq * t + modulator)
    sub = np.sin(2 * np.pi * freq * t) * 0.5

    raw = carrier * 0.7 + sub
    # Distortion
    raw = soft_clip(raw * 2.0, 0.6)

    cutoff_env = 200 + filter_env_amount * np.exp(-t * 7)
    filtered = np.zeros(n)
    chunk = SR // 20
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        co = cutoff_env[i]
        filtered[i:end] = lowpass(raw[i:end], co)

    amp = env_adsr(n, attack=0.002, decay=0.05, sustain=0.7, release=0.03)
    return filtered * amp


# ─── PAD ─────────────────────────────────────────────────────────────

def synth_pad(freq, length_s, detune_cents=20):
    """Ultra-dark drone pad - no melody, just texture and weight"""
    n = int(length_s * SR)
    t = np.linspace(0, length_s, n, endpoint=False)

    pad = np.zeros(n)
    detune_ratio = 2 ** (detune_cents / 1200)
    # Heavy detuned drone cluster
    for d in [1.0 / detune_ratio, 1.0, detune_ratio,
              0.5 / detune_ratio, 0.5, 0.5 * detune_ratio,
              # Add dissonant partials: tritone + minor 2nd
              2 ** (6 / 12), 2 ** (1 / 12) * 0.5]:
        f = freq * d
        pad += sawtooth(2 * np.pi * f * t) * 0.08
        pad += pulse_wave(2 * np.pi * f * 0.998 * t, width=0.15) * 0.04

    # Very slow LFO, much darker filter
    lfo = np.sin(2 * np.pi * 0.04 * t) * 0.5 + 0.5
    cutoff_base = 200 + 600 * lfo  # Much darker

    filtered = np.zeros(n)
    chunk = SR // 10
    for i in range(0, n, chunk):
        end = min(i + chunk, n)
        co = cutoff_base[i]
        filtered[i:end] = lowpass(pad[i:end], co)

    amp = env_adsr(n, attack=2.0, decay=1.0, sustain=0.4, release=2.0)
    return filtered * amp


# ─── MACHINE GUN RHYTHM GENERATOR ────────────────────────────────────

def gen_machine_gun(length_s, seed=99):
    """
    Rapid-fire 32nd note machine gun percussion bursts.
    Alternates between dense fills and breathing space.
    """
    rng = np.random.RandomState(seed)
    n = int(length_s * SR)
    out = np.zeros(n)

    thirtysecond = int(BEAT * SR / 8)  # 32nd notes!
    total_steps = n // thirtysecond

    for step in range(total_steps):
        bar_pos = step % 64  # 64 thirty-second notes per bar at this speed
        pos = step * thirtysecond

        # Create dense bursts: machine gun for 8 steps, then sparse for 8
        burst_phase = (step // 8) % 4
        if burst_phase == 0:
            # Dense machine gun burst
            prob = 0.85
        elif burst_phase == 1:
            prob = 0.6
        elif burst_phase == 2:
            # Accent pattern
            prob = 0.4 if (step * 3) % 5 < 2 else 0.15
        else:
            prob = 0.2

        if rng.random() > prob:
            continue

        # Alternate between different percussive hits
        hit_type = rng.choice(['metal', 'noise', 'click', 'zap'])

        if hit_type == 'metal':
            freq = rng.choice([600, 900, 1200, 1800, 2400])
            hit = synth_metal_hit(length_s=rng.uniform(0.02, 0.06),
                                  freq_base=freq)
            vol = rng.uniform(0.15, 0.35)
        elif hit_type == 'noise':
            hit = synth_machine_noise(length_s=rng.uniform(0.01, 0.04))
            vol = rng.uniform(0.1, 0.25)
        elif hit_type == 'click':
            cl = int(rng.uniform(0.001, 0.008) * SR)
            hit = rng.randn(cl) * np.exp(-np.linspace(0, 1, cl) * 20)
            hit = highpass(hit, 2000)
            vol = rng.uniform(0.15, 0.4)
        else:  # zap - short pitched blip
            cl = int(0.015 * SR)
            te = np.linspace(0, 0.015, cl, endpoint=False)
            freq_sweep = 3000 * np.exp(-te * 200) + 200
            ph = 2 * np.pi * np.cumsum(freq_sweep) / SR
            hit = np.sin(ph) * np.exp(-te * 150)
            vol = rng.uniform(0.1, 0.3)

        end = min(pos + len(hit), n)
        out[pos:end] += hit[:end - pos] * vol

    return out


# ─── GLITCH TEXTURE (Cellular Automata) ──────────────────────────────

def gen_glitch_texture(length_s, density=0.45, seed=42):
    """Denser glitch textures with more industrial character"""
    rng = np.random.RandomState(seed)
    n = int(length_s * SR)
    out = np.zeros(n)

    # Rule 110 cellular automata
    cells = np.zeros(64, dtype=int)
    cells[32] = 1
    pattern = []
    for _ in range(256):
        pattern.append(cells.copy())
        new_cells = np.zeros_like(cells)
        for i in range(1, len(cells) - 1):
            neighborhood = cells[i-1] * 4 + cells[i] * 2 + cells[i+1]
            new_cells[i] = (110 >> neighborhood) & 1
        cells = new_cells
    pattern = np.array(pattern)

    sixteenth = int(BEAT * SR / 4)

    for bar_idx in range(int(length_s / BAR)):
        for step in range(16):
            ca_row = (bar_idx * 16 + step) % 256
            ca_col = step * 4 % 64
            if pattern[ca_row, ca_col] == 1 and rng.random() < density:
                pos = int(bar_idx * BAR * SR + step * sixteenth)
                if pos >= n:
                    break

                event_type = rng.choice([
                    'grain', 'click', 'blip', 'stutter',
                    'ring', 'bitcrush_burst', 'metal_grain'
                ])
                event_len = int(rng.uniform(0.002, 0.05) * SR)
                event_len = min(event_len, n - pos)
                te = np.linspace(0, event_len / SR, event_len, endpoint=False)

                if event_type == 'grain':
                    freq = rng.uniform(200, 5000)
                    grain = np.sin(2 * np.pi * freq * te) * np.hanning(event_len)
                elif event_type == 'click':
                    grain = rng.randn(event_len) * np.exp(-te * 800)
                elif event_type == 'blip':
                    freq = rng.choice([440, 880, 1760, 2500])
                    grain = np.sin(2 * np.pi * freq * te) * np.exp(-te * 150)
                elif event_type == 'stutter':
                    freq = rng.uniform(100, 500)
                    grain = square_wave(2 * np.pi * freq * te) * np.exp(-te * 100)
                elif event_type == 'ring':
                    f1 = rng.uniform(300, 2000)
                    f2 = rng.uniform(50, 300)
                    grain = np.sin(2*np.pi*f1*te) * np.sin(2*np.pi*f2*te)
                    grain *= np.exp(-te * 80)
                elif event_type == 'bitcrush_burst':
                    grain = rng.randn(event_len)
                    grain = bitcrush(grain, bits=rng.randint(2, 6), downsample=rng.randint(2, 8))
                    grain *= np.exp(-te * 60)
                else:  # metal_grain
                    base = rng.uniform(400, 3000)
                    grain = np.zeros(event_len)
                    for ratio in [1.0, 1.47, 2.09, 3.14]:
                        f = base * ratio
                        if f < SR / 2 - 200:
                            grain += np.sin(2*np.pi*f*te) * 0.25
                    grain *= np.exp(-te * 50) * np.hanning(event_len)

                grain *= rng.uniform(0.1, 0.5)
                end = min(pos + event_len, n)
                out[pos:end] += grain[:end - pos]

    return highpass(out, 300)


# ─── RAPID ARPEGGIATOR ──────────────────────────────────────────────

def gen_arpeggio(length_s, root=220):
    """Cold, atonal arpeggiator - no melodic scale, pure data patterns"""
    n = int(length_s * SR)
    out = np.zeros(n)

    # Chromatic / atonal intervals - cold, no tonal center
    # Tritones, minor 2nds, major 7ths - maximally dissonant
    intervals = [0, 1, 6, 7, 11, 13, 18, 23, 25]
    freqs = [root * (2 ** (i / 12)) for i in intervals]

    # 32nd notes
    thirtysecond = int(BEAT * SR / 8)
    total_steps = n // thirtysecond

    rng = np.random.RandomState(777)

    for step in range(total_steps):
        prob = 0.7 if (step * 5) % 8 < 4 else 0.35
        if rng.random() > prob:
            continue

        note_idx = (step * 3 + step // 5) % len(freqs)
        freq = freqs[note_idx]

        pos = step * thirtysecond
        # Very short, clinical notes
        note_len = int(rng.uniform(0.008, 0.04) * SR)
        note_len = min(note_len, n - pos)

        t_note = np.linspace(0, note_len / SR, note_len, endpoint=False)

        # Pure sine only - cold digital tone
        osc = np.sin(2 * np.pi * freq * t_note)
        amp = env_adsr(note_len, attack=0.0005, decay=0.005, sustain=0.2, release=0.005)

        osc = osc * amp * rng.uniform(0.06, 0.2)
        end = min(pos + note_len, n)
        out[pos:end] += osc[:end - pos]

    out = bandpass(out, 800, 8000)
    out = delay_effect(out, delay_time=BEAT * 0.5, feedback=0.25, mix=0.25)
    return out


# ─── DATA SONIFICATION ──────────────────────────────────────────────

def gen_data_stream(length_s, seed=314):
    """
    Rapid-fire pure sine micro-tones,
    ultra-precise digital clicks, and data-stream sonification.
    Extremely fast sequences of short pitched events.
    """
    rng = np.random.RandomState(seed)
    n = int(length_s * SR)
    out = np.zeros(n)

    # Event grid: 64th notes (twice as fast as 32nd)
    sixtyfourth = int(BEAT * SR / 16)
    total_steps = n // sixtyfourth

    for step in range(total_steps):
        pos = step * sixtyfourth

        # Data-stream pattern: dense bursts then silence
        cycle = step % 32
        if cycle < 12:
            prob = 0.9   # Dense burst
        elif cycle < 16:
            prob = 0.0   # Gap
        elif cycle < 24:
            prob = 0.7   # Medium density
        else:
            prob = 0.1   # Sparse tail

        if rng.random() > prob:
            continue

        event = rng.choice([
            'sine_blip', 'click', 'needle', 'white_burst', 'data_tone'
        ], p=[0.3, 0.25, 0.15, 0.15, 0.15])

        if event == 'sine_blip':
            # Pure sine micro-tone
            dur = int(rng.uniform(0.0005, 0.008) * SR)
            dur = min(dur, n - pos)
            freq = rng.choice([1000, 2000, 3000, 4000, 5000])
            te = np.linspace(0, dur / SR, dur, endpoint=False)
            grain = np.sin(2 * np.pi * freq * te)
            grain *= np.hanning(dur)
            vol = rng.uniform(0.15, 0.4)

        elif event == 'click':
            # Single-sample digital click
            dur = int(rng.uniform(0.0002, 0.002) * SR)
            dur = max(dur, 2)
            dur = min(dur, n - pos)
            grain = np.zeros(dur)
            grain[0] = rng.choice([-1.0, 1.0])
            if dur > 1:
                grain[1] = -grain[0] * 0.5
            vol = rng.uniform(0.3, 0.6)

        elif event == 'needle':
            # High frequency needle tone
            dur = int(rng.uniform(0.003, 0.02) * SR)
            dur = min(dur, n - pos)
            freq = rng.uniform(3000, 6000)
            te = np.linspace(0, dur / SR, dur, endpoint=False)
            grain = np.sin(2 * np.pi * freq * te)
            grain *= np.exp(-te * 300)
            vol = rng.uniform(0.08, 0.2)

        elif event == 'white_burst':
            # Precise white noise burst
            dur = int(rng.uniform(0.0005, 0.005) * SR)
            dur = min(dur, n - pos)
            grain = rng.randn(dur)
            grain *= np.hanning(dur)
            vol = rng.uniform(0.1, 0.3)

        else:  # data_tone
            # Two-frequency data modem tone
            dur = int(rng.uniform(0.005, 0.02) * SR)
            dur = min(dur, n - pos)
            te = np.linspace(0, dur / SR, dur, endpoint=False)
            f1 = rng.choice([1200, 2400, 4800])
            f2 = f1 * rng.choice([1.5, 2.0, 3.0])
            # FSK-like modulation
            switch = (te * 80).astype(int) % 2
            freq_arr = np.where(switch, f1, f2)
            phase = 2 * np.pi * np.cumsum(freq_arr) / SR
            grain = np.sin(phase) * np.hanning(dur)
            vol = rng.uniform(0.1, 0.25)

        end = min(pos + len(grain), n)
        out[pos:end] += grain[:end - pos] * vol

    return out


def gen_micro_clicks(length_s, seed=271):
    """
    Ultra-precise micro-click patterns.
    Grid-quantized digital impulses at extreme speeds.
    """
    rng = np.random.RandomState(seed)
    n = int(length_s * SR)
    out = np.zeros(n)

    # 128th note grid (extreme precision)
    grid = int(BEAT * SR / 32)
    total_steps = n // grid

    for step in range(total_steps):
        pos = step * grid

        # Binary pattern from mathematical sequence
        # Thue-Morse sequence for non-repeating pattern
        bit_count = bin(step).count('1')
        is_active = bit_count % 2 == 0

        if not is_active and rng.random() > 0.1:
            continue
        if is_active and rng.random() > 0.7:
            continue

        # Micro click: 1-10 samples
        click_len = rng.randint(1, min(12, n - pos))
        click = np.zeros(click_len)
        click[0] = rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 0.8)
        for i in range(1, click_len):
            click[i] = -click[i-1] * rng.uniform(0.3, 0.7)

        end = min(pos + click_len, n)
        out[pos:end] += click[:end - pos]

    return out


def gen_electrical_pulse(length_s, seed=161):
    """
    Electrical impulse patterns,
    binary-coded rhythms, EMG-like signal bursts.
    """
    rng = np.random.RandomState(seed)
    n = int(length_s * SR)
    out = np.zeros(n)

    # Binary encoding of a message as rhythm
    # "DATA" in ASCII: 68 65 84 65
    binary_msg = ''.join(format(b, '08b') for b in [68, 65, 84, 65])
    binary_msg = binary_msg * 100  # Repeat

    bit_duration = int(BEAT * SR / 8)
    total_bits = min(len(binary_msg), n // bit_duration)

    for i in range(total_bits):
        if binary_msg[i] == '0':
            continue

        pos = i * bit_duration

        # EMG-style electrical impulse
        pulse_type = rng.choice(['emg', 'square_burst', 'fm_zap'])

        if pulse_type == 'emg':
            dur = int(rng.uniform(0.003, 0.015) * SR)
            dur = min(dur, n - pos)
            te = np.linspace(0, dur / SR, dur, endpoint=False)
            # Biphasic pulse (like real EMG)
            freq = rng.uniform(200, 800)
            grain = np.sin(2 * np.pi * freq * te) * np.exp(-te * 300)
            grain *= rng.randn(dur) * 0.3 + 0.7  # Noisy modulation
            vol = rng.uniform(0.15, 0.35)

        elif pulse_type == 'square_burst':
            dur = int(rng.uniform(0.002, 0.01) * SR)
            dur = min(dur, n - pos)
            te = np.linspace(0, dur / SR, dur, endpoint=False)
            freq = rng.choice([100, 200, 400, 800, 1600])
            grain = square_wave(2 * np.pi * freq * te)
            grain *= np.exp(-te * 200)
            vol = rng.uniform(0.1, 0.3)

        else:  # fm_zap
            dur = int(rng.uniform(0.005, 0.02) * SR)
            dur = min(dur, n - pos)
            te = np.linspace(0, dur / SR, dur, endpoint=False)
            carrier_f = rng.uniform(500, 3000)
            mod_f = rng.uniform(50, 500)
            mod_depth = carrier_f * 3 * np.exp(-te * 100)
            phase = 2 * np.pi * (carrier_f * te + mod_depth / (2*np.pi*mod_f) * np.sin(2*np.pi*mod_f*te))
            grain = np.sin(phase) * np.exp(-te * 120)
            vol = rng.uniform(0.1, 0.25)

        end = min(pos + len(grain), n)
        out[pos:end] += grain[:end - pos] * vol

    return highpass(out, 200)


# ─── SHEPARD TONE (auditory illusion - endlessly rising) ────────────

def gen_shepard_tone(length_s, direction='up', cycle_s=8.0):
    """
    Shepard tone: auditory illusion of endlessly rising (or falling) pitch.
    Multiple octave-spaced sine waves with Gaussian spectral envelope
    create the illusion of infinite ascent.
    """
    n = int(length_s * SR)
    t = np.linspace(0, length_s, n, endpoint=False)
    out = np.zeros(n)

    base_freq = 55.0  # A1
    num_octaves = 7
    # Gaussian envelope centered around ~1000Hz in log space
    center_log = np.log2(1000)
    sigma = 1.8  # Width in octaves

    for i in range(num_octaves):
        # Each partial sweeps through one octave over cycle_s seconds
        if direction == 'up':
            sweep = (t / cycle_s + i / num_octaves) % 1.0
        else:
            sweep = (1.0 - t / cycle_s + i / num_octaves) % 1.0

        # Frequency: sweeps from base_freq to base_freq * 2^num_octaves
        freq = base_freq * (2 ** (sweep * num_octaves))

        # Gaussian amplitude envelope in log-frequency space
        log_freq = np.log2(freq)
        amplitude = np.exp(-0.5 * ((log_freq - center_log) / sigma) ** 2)

        # Generate with phase accumulation for smooth frequency sweep
        phase = 2 * np.pi * np.cumsum(freq) / SR
        out += np.sin(phase) * amplitude

    # Normalize
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak

    return out


# ─── STUTTER / BUFFER-REPEAT GLITCH ────────────────────────────────

def gen_stutter_glitch(length_s, seed=503):
    """
    Buffer-repeat glitch: randomly captures tiny audio fragments
    and repeats them rapidly, creating machine-like stuttering.
    Inspired by glitch / live processing aesthetics.
    """
    rng = np.random.RandomState(seed)
    n = int(length_s * SR)
    out = np.zeros(n)

    pos = 0
    while pos < n:
        # Decide: stutter event or silence gap
        if rng.random() < 0.3:
            # STUTTER EVENT
            # Grab a tiny buffer (0.5ms - 30ms)
            buf_len = int(rng.uniform(0.0005, 0.03) * SR)
            buf_len = min(buf_len, n - pos)

            # Source for the buffer: sine, noise, or click
            src_type = rng.choice(['sine', 'noise', 'click', 'fm'])
            if src_type == 'sine':
                freq = rng.choice([220, 440, 880, 1760, 3520, 7040])
                tb = np.linspace(0, buf_len / SR, buf_len, endpoint=False)
                buf = np.sin(2 * np.pi * freq * tb)
            elif src_type == 'noise':
                buf = rng.randn(buf_len)
            elif src_type == 'click':
                buf = np.zeros(buf_len)
                buf[0] = rng.choice([-1.0, 1.0])
                for ci in range(1, min(5, buf_len)):
                    buf[ci] = -buf[ci - 1] * 0.6
            else:  # fm
                freq = rng.uniform(200, 2000)
                mod = rng.uniform(50, 500)
                tb = np.linspace(0, buf_len / SR, buf_len, endpoint=False)
                buf = np.sin(2 * np.pi * freq * tb +
                             3 * np.sin(2 * np.pi * mod * tb))

            # Repeat the buffer N times rapidly
            repeats = rng.randint(2, 32)
            vol = rng.uniform(0.1, 0.4)

            for r in range(repeats):
                rpos = pos + r * buf_len
                if rpos + buf_len > n:
                    break
                end = min(rpos + buf_len, n)
                out[rpos:end] += buf[:end - rpos] * vol

            pos += repeats * buf_len
        else:
            # SILENCE GAP
            gap = int(rng.uniform(0.01, 0.15) * SR)
            pos += gap

    return highpass(out, 150)


# ─── SINE SWEEP TEST PATTERNS ──────────────────────────────────────

def gen_test_pattern_sweeps(length_s, seed=619):
    """
    Test pattern style: rapid sine frequency sweeps,
    calibration tones, and systematic frequency scanning.
    Like a machine testing every frequency in rapid succession.
    """
    rng = np.random.RandomState(seed)
    n = int(length_s * SR)
    out = np.zeros(n)

    # Schedule sweep events
    pos = 0
    while pos < n:
        if rng.random() < 0.5:
            # RAPID SWEEP (ascending or descending)
            sweep_dur = int(rng.uniform(0.01, 0.15) * SR)
            sweep_dur = min(sweep_dur, n - pos)
            ts = np.linspace(0, sweep_dur / SR, sweep_dur, endpoint=False)

            f_start = rng.choice([100, 200, 500, 1000, 2000])
            if rng.random() < 0.5:
                # Ascending sweep
                f_end = f_start * rng.uniform(4, 20)
                f_end = min(f_end, 6000)
            else:
                # Descending sweep
                f_end = f_start / rng.uniform(2, 10)
                f_end = max(f_end, 20)

            # Exponential frequency sweep
            freq_sweep = f_start * (f_end / f_start) ** (ts / (sweep_dur / SR))
            phase = 2 * np.pi * np.cumsum(freq_sweep) / SR
            grain = np.sin(phase)
            # Sharp window
            grain *= np.hanning(sweep_dur)

            vol = rng.uniform(0.08, 0.25)
            end = min(pos + sweep_dur, n)
            out[pos:end] += grain[:end - pos] * vol

            pos += sweep_dur + int(rng.uniform(0.005, 0.05) * SR)

        elif rng.random() < 0.4:
            # CALIBRATION TONE BURST (exact frequency, precise duration)
            tone_dur = int(rng.choice([0.005, 0.01, 0.02, 0.05, 0.1]) * SR)
            tone_dur = min(tone_dur, n - pos)
            freq = rng.choice([100, 400, 1000, 2500, 4000, 5000, 6000])
            tt = np.linspace(0, tone_dur / SR, tone_dur, endpoint=False)
            grain = np.sin(2 * np.pi * freq * tt)
            # Rectangle window with tiny fade to avoid clicks
            fade = min(int(0.0005 * SR), tone_dur // 4)
            if fade > 0:
                grain[:fade] *= np.linspace(0, 1, fade)
                grain[-fade:] *= np.linspace(1, 0, fade)

            vol = rng.uniform(0.1, 0.3)
            end = min(pos + tone_dur, n)
            out[pos:end] += grain[:end - pos] * vol

            pos += tone_dur + int(rng.uniform(0.002, 0.03) * SR)

        else:
            # FREQUENCY STAIRCASE (stepping through discrete frequencies)
            num_steps = rng.randint(4, 20)
            step_dur = int(rng.uniform(0.003, 0.015) * SR)
            base_f = rng.choice([200, 500, 1000, 2000])

            for si in range(num_steps):
                spos = pos + si * step_dur
                if spos + step_dur > n:
                    break
                freq = base_f * (2 ** (si / 12))  # Chromatic steps
                tt = np.linspace(0, step_dur / SR, step_dur, endpoint=False)
                grain = np.sin(2 * np.pi * freq * tt) * np.hanning(step_dur)
                end = min(spos + step_dur, n)
                out[spos:end] += grain[:end - spos] * 0.15

            pos += num_steps * step_dur + int(rng.uniform(0.02, 0.1) * SR)

    return out


# ─── INDUSTRIAL NOISE RISER ─────────────────────────────────────────

def gen_noise_riser(length_s, start_time, rise_duration=8.0):
    """Filtered noise sweep that builds tension"""
    n = int(length_s * SR)
    out = np.zeros(n)

    start_sample = int(start_time * SR)
    rise_samples = int(rise_duration * SR)
    end_sample = min(start_sample + rise_samples, n)

    noise = _rng.randn(end_sample - start_sample) * 0.3
    t_rise = np.linspace(0, 1, end_sample - start_sample)

    # Sweep filter from 200Hz to 8000Hz
    chunk = SR // 20
    filtered = np.zeros_like(noise)
    for i in range(0, len(noise), chunk):
        end = min(i + chunk, len(noise))
        progress = i / len(noise)
        cutoff = 200 + 7800 * progress ** 2
        filtered[i:end] = resonant_lp(noise[i:end], cutoff, resonance=4.0)

    # Volume ramp
    filtered *= t_rise ** 2 * 0.5

    out[start_sample:end_sample] += filtered[:end_sample - start_sample]
    return out


# ─── ARRANGEMENT ─────────────────────────────────────────────────────

def build_arrangement():
    """
    Industrial arrangement @ 155 BPM:
    - 0:00       FULL ASSAULT from the first sample
    - 0:56-1:12  Breakdown (sparse, noise riser)
    - 1:12-1:48  Climax (maximum density)
    - 1:48-2:00  Outro
    """
    print("=" * 55)
    print("  Industrial Electronic Music v3")
    print("=" * 55)
    print(f"  BPM: {BPM} | Duration: {DURATION}s | SR: {SR}Hz")
    print()

    mix = np.zeros(total_samples)

    main_end = 56
    break_end = 72
    climax_end = 108

    # ── PAD ────────────────────────────────────────────────────────
    print("  [1/14] Dark industrial pad...")

    # Single drone root - no chord changes, pure dark texture
    drone_roots = [55.0, 55.0, 58.27, 55.0]  # A1 drone with subtle tritone shift
    chord_bar_len = BAR * 8  # Longer sustain, less movement

    pad_track = np.zeros(total_samples)
    for i, root in enumerate(drone_roots):
        for rep in range(int(DURATION / (chord_bar_len * len(drone_roots))) + 1):
            start = int((rep * len(drone_roots) + i) * chord_bar_len * SR)
            length = int(chord_bar_len * SR)
            if start >= total_samples:
                break
            end = min(start + length, total_samples)
            actual_len = end - start
            note = synth_pad(root, actual_len / SR)[:actual_len]
            pad_track[start:end] += note

    # Pad subdued - just dark texture bed, not melodic
    # Full volume from 0:00 - no fade in
    pad_vol = np.ones(total_samples) * 0.25
    pad_vol[int(main_end * SR):int(break_end * SR)] = 0.35
    s = int(climax_end * SR)
    pad_vol[s:] = np.linspace(0.3, 0, total_samples - s)

    pad_track = reverb_simple(pad_track, decay=0.35)
    mix += pad_track * pad_vol

    # ── KICK ───────────────────────────────────────────────────────
    print("  [2/14] Industrial kick pattern...")

    kick_single = synth_kick()
    kick_track = np.zeros(total_samples)
    beat_samples = int(BEAT * SR)
    eighth_samples = beat_samples // 2  # 8th note grid for faster kicks

    # Fast irregular kick on 8th note grid (16 steps per bar)
    # Dense but broken - no steady pulse
    kick_pattern_16 = [
        True,  False, True,  True,   # 1 . . 2
        False, True,  False, False,  # . . . .
        True,  True,  False, True,   # 3 . . .
        False, False, True,  False,  # . . . .
    ]

    for step in range(int(DURATION * 2 / BEAT)):  # 8th note steps
        pos = step * eighth_samples
        t_sec = pos / SR

        if main_end <= t_sec < break_end:
            # Sparse in breakdown
            if step % 16 != 0:
                continue
        else:
            if not kick_pattern_16[step % 16]:
                continue

        end = min(pos + len(kick_single), total_samples)
        kick_track[pos:end] += kick_single[:end - pos]

    mix += kick_track * 0.85

    # ── HI-HATS (machine-fast) ────────────────────────────────────
    print("  [3/14] Machine-speed hi-hats...")

    hat_track = np.zeros(total_samples)
    # 32nd note hats for insane speed!
    thirtysecond_samples = beat_samples // 8
    rng_hat = np.random.RandomState(123)

    for step in range(int(DURATION * 8 / BEAT)):
        pos = step * thirtysecond_samples
        t_sec = pos / SR

        if main_end <= t_sec < break_end:
            if step % 8 != 4:
                continue

        # Dense hi-hat pattern
        is_hit = (step * 11) % 16 < 9
        if not is_hit and rng_hat.random() > 0.2:
            continue

        is_open = step % 32 in [8, 24] and rng_hat.random() < 0.2
        hat = synth_hihat(
            length_s=0.08 if is_open else 0.02,
            open_hat=is_open
        )

        velocity = rng_hat.uniform(0.1, 0.35)
        # Ghost note dynamics
        if step % 8 == 0:
            velocity *= 1.4
        elif step % 4 == 0:
            velocity *= 1.2
        elif step % 2 == 0:
            velocity *= 1.0
        else:
            velocity *= 0.6  # Ghost notes

        end = min(pos + len(hat), total_samples)
        hat_track[pos:end] += hat[:end - pos] * velocity

    hat_vol = np.ones(total_samples) * 0.45
    s = int(climax_end * SR)
    hat_vol[s:] = np.linspace(0.45, 0, total_samples - s)
    mix += hat_track * hat_vol

    # ── INDUSTRIAL NOISE HITS (replaces clap - no 2&4 backbeat) ────
    print("  [4/14] Irregular industrial noise hits...")

    clap_single = synth_clap()
    clap_track = np.zeros(total_samples)
    rng_clap = np.random.RandomState(456)

    # Irregular hits driven by prime numbers - never feels like backbeat
    for beat in range(int(DURATION / BEAT)):
        pos = beat * beat_samples
        t_sec = pos / SR

        if main_end <= t_sec < break_end:
            continue

        # Hit only on prime-numbered beats within each 16-beat cycle
        beat_in_cycle = beat % 16
        if beat_in_cycle not in [2, 3, 5, 7, 11, 13]:
            continue
        # Further randomize
        if rng_clap.random() > 0.5:
            continue

        end = min(pos + len(clap_single), total_samples)
        clap_track[pos:end] += clap_single[:end - pos]

    clap_track = reverb_simple(clap_track, decay=0.1, delays_ms=(13, 29, 43))
    mix += clap_track * 0.25

    # ── MACHINE GUN PERCUSSION ────────────────────────────────────
    print("  [5/14] Machine gun percussion (32nd notes)...")

    machine_track = gen_machine_gun(DURATION, seed=99)
    # Full from 0:00 but tamed
    machine_vol = np.ones(total_samples) * 0.35
    machine_vol[int(main_end * SR):int(break_end * SR)] = np.linspace(0.35, 0.08, int((break_end - main_end) * SR))
    machine_vol[int(break_end * SR):int(climax_end * SR)] = 0.45  # Full force in climax
    s = int(climax_end * SR)
    machine_vol[s:] = np.linspace(0.45, 0, total_samples - s)

    mix += machine_track * machine_vol

    # ── SUB DRONE (replaces rhythmic bass) ─────────────────────────
    print("  [6/14] Continuous sub-bass drone...")

    bass_track = np.zeros(total_samples)
    bass_freq = 55.0  # A1 continuous drone

    # Generate continuous sub-bass drone - no rhythm, just weight
    t_full = np.linspace(0, DURATION, total_samples, endpoint=False)
    # Pure sub sine + slight FM wobble
    wobble = np.sin(2 * np.pi * 0.15 * t_full) * 5  # Very slow FM
    bass_drone = np.sin(2 * np.pi * bass_freq * t_full + wobble) * 0.5
    # Add second harmonic with slow beating
    bass_drone += np.sin(2 * np.pi * bass_freq * 2.003 * t_full) * 0.15
    bass_drone = lowpass(bass_drone, 120)

    # Full volume from 0:00 - no fade in
    bass_vol = np.ones(total_samples) * 0.5
    bass_vol[int(main_end * SR):int(break_end * SR)] = np.linspace(0.5, 0.15, int((break_end - main_end) * SR))
    bass_vol[int(break_end * SR):int(climax_end * SR)] = 0.55
    s = int(climax_end * SR)
    bass_vol[s:] = np.linspace(0.55, 0, total_samples - s)

    bass_track = soft_clip(bass_drone * 1.5, 0.6)
    mix += bass_track * bass_vol

    # ── [ARPEGGIO REMOVED - too trot-like] ─────────────────────────
    print("  [7/14] (Arpeggio removed - replaced by data layers)")

    # ── GLITCH TEXTURES ───────────────────────────────────────────
    print("  [8/14] Cellular automata glitch (high density)...")

    glitch_track = gen_glitch_texture(DURATION, density=0.45, seed=42)
    # Full volume from 0:00
    glitch_vol = np.ones(total_samples) * 0.35
    glitch_vol[int(main_end * SR):int(break_end * SR)] = np.linspace(0.35, 0.5, int((break_end - main_end) * SR))
    glitch_vol[int(break_end * SR):int(climax_end * SR)] = 0.55
    s = int(climax_end * SR)
    glitch_vol[s:] = np.linspace(0.55, 0, total_samples - s)

    mix += glitch_track * glitch_vol

    # ── DATA STREAM ───────────────────────────────────────────────
    print("  [9/14] Data sonification stream...")

    data_stream_track = gen_data_stream(DURATION, seed=314)
    # Data stream dominant but not overwhelming
    data_stream_vol = np.ones(total_samples) * 0.6
    data_stream_vol[int(main_end * SR):int(break_end * SR)] = np.linspace(0.6, 0.7, int((break_end - main_end) * SR))
    data_stream_vol[int(break_end * SR):int(climax_end * SR)] = 0.7
    s = int(climax_end * SR)
    data_stream_vol[s:] = np.linspace(0.7, 0.2, total_samples - s)

    mix += data_stream_track * data_stream_vol

    # ── MICRO CLICKS ──────────────────────────────────────────────
    print("  [10/14] Micro-click patterns...")

    clicks_track = gen_micro_clicks(DURATION, seed=271)
    # Micro clicks present but tamed
    clicks_vol = np.ones(total_samples) * 0.35
    clicks_vol[int(main_end * SR):int(break_end * SR)] = 0.5
    clicks_vol[int(break_end * SR):int(climax_end * SR)] = 0.4
    s = int(climax_end * SR)
    clicks_vol[s:] = np.linspace(0.6, 0.1, total_samples - s)

    mix += clicks_track * clicks_vol

    # ── ELECTRICAL PULSE ──────────────────────────────────────────
    print("  [11/14] Electrical impulse patterns...")

    impulse_track = gen_electrical_pulse(DURATION, seed=161)
    # Impulse pulses present but balanced
    impulse_vol = np.ones(total_samples) * 0.35
    impulse_vol[int(main_end * SR):int(break_end * SR)] = np.linspace(0.35, 0.15, int((break_end - main_end) * SR))
    impulse_vol[int(break_end * SR):int(climax_end * SR)] = 0.4
    s = int(climax_end * SR)
    impulse_vol[s:] = np.linspace(0.4, 0, total_samples - s)

    mix += impulse_track * impulse_vol

    # ── STUTTER GLITCH ─────────────────────────────────────────────
    print("  [12/14] Stutter/buffer-repeat glitch...")

    stutter_track = gen_stutter_glitch(DURATION, seed=503)
    stutter_vol = np.ones(total_samples) * 0.18
    stutter_vol[int(main_end * SR):int(break_end * SR)] = np.linspace(0.18, 0.3, int((break_end - main_end) * SR))
    stutter_vol[int(break_end * SR):int(climax_end * SR)] = 0.25
    s = int(climax_end * SR)
    stutter_vol[s:] = np.linspace(0.25, 0, total_samples - s)

    mix += stutter_track * stutter_vol

    # ── TEST PATTERN SWEEPS ────────────────────────────────────────
    print("  [13/14] Test pattern sine sweeps...")

    sweep_track = gen_test_pattern_sweeps(DURATION, seed=619)
    sweep_vol = np.ones(total_samples) * 0.15
    sweep_vol[int(main_end * SR):int(break_end * SR)] = 0.22
    sweep_vol[int(break_end * SR):int(climax_end * SR)] = 0.2
    s = int(climax_end * SR)
    sweep_vol[s:] = np.linspace(0.2, 0.03, total_samples - s)

    mix += sweep_track * sweep_vol

    # ── NOISE RISERS ──────────────────────────────────────────────
    print("  [14/14] Industrial noise risers...")

    # Riser before climax (breakdown → climax transition)
    mix += gen_noise_riser(DURATION, start_time=64, rise_duration=8)

    # ── FINAL MIXDOWN ─────────────────────────────────────────────
    print("\n  Mixdown & mastering...")

    mix = highpass(mix, 28)

    # Harder compression/saturation for industrial sound
    mix = soft_clip(mix * 1.5, threshold=0.75)

    # Gentle lowpass to tame harsh highs
    mix = lowpass(mix, 8000, order=2)

    # Stereo
    left = mix.copy()
    right = np.zeros_like(mix)
    stereo_delay = int(0.0003 * SR)
    right[stereo_delay:] = mix[:-stereo_delay]

    left = left + reverb_simple(left, decay=0.12, delays_ms=(23, 37, 53)) * 0.06
    right = right + reverb_simple(right, decay=0.12, delays_ms=(29, 41, 59)) * 0.06

    left += _rng.randn(total_samples) * 0.0008
    right += _rng.randn(total_samples) * 0.0008

    stereo = np.column_stack([left, right])
    peak = np.max(np.abs(stereo))
    if peak > 0:
        stereo = stereo / peak * 0.95  # Louder master

    stereo_16 = (stereo * 32767).astype(np.int16)
    return stereo_16


# ─── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, 'industrial_v2.wav')

    audio = build_arrangement()
    write_wav(output_path, audio)

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n  Done! Saved to: {output_path}")
    print(f"  File size: {file_size_mb:.1f} MB")
    print(f"  Duration: {DURATION}s | BPM: {BPM} | Stereo 16-bit {SR}Hz")
    print()
    print("  Track structure:")
    print("  ─────────────────────────────────────────")
    print("  0:00 - 0:12  Dark ambient + machine noise")
    print("  0:12 - 0:28  Kick + machine gun build")
    print("  0:28 - 0:56  Full industrial assault")
    print("  0:56 - 1:12  Breakdown + noise riser")
    print("  1:12 - 1:48  Maximum density climax")
    print("  1:48 - 2:00  Outro")
    print("  ─────────────────────────────────────────")
