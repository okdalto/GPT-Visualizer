"""Generate individual WAV samples for each micro-sound type.

Usage:
    python -m audio.preview_sounds
    # Creates assets/sample_*.wav files
"""
import os
import numpy as np
from audio.common import SR, write_wav
from audio.sonification import (
    _sine_blip, _click, _needle, _white_burst, _data_tone, _freq_sweep,
)


def make_sample(name, gen_fn, freq=1000, dur_ms=50, repeats=5, gap_ms=150):
    """Generate a sample WAV with repeated instances of a sound."""
    dur_samples = int(dur_ms * 0.001 * SR)
    gap_samples = int(gap_ms * 0.001 * SR)
    total = repeats * (dur_samples + gap_samples) + SR  # +1s padding
    out = np.zeros(total)

    for i in range(repeats):
        pos = i * (dur_samples + gap_samples) + gap_samples
        # Vary frequency per repeat for demonstration
        f = freq * (0.5 + i * 0.3)
        grain = gen_fn(f, dur_samples)
        end = min(pos + len(grain), total)
        out[pos:end] += grain[:end - pos] * 0.7

    # Normalize
    peak = np.max(np.abs(out))
    if peak > 0:
        out = out / peak * 0.85

    # Stereo
    stereo = np.column_stack([out, out])
    return (stereo * 32767).astype(np.int16), name


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(base_dir, "assets", "samples")
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(42)

    samples = [
        ("sine_blip", lambda f, d: _sine_blip(f, d)),
        ("click", lambda f, d: _click(d, polarity=1.0)),
        ("needle", lambda f, d: _needle(f, d)),
        ("white_burst", lambda f, d: _white_burst(d, rng)),
        ("data_tone", lambda f, d: _data_tone(f, f * 1.5, d)),
        ("freq_sweep_up", lambda f, d: _freq_sweep(f, f * 3, d)),
        ("freq_sweep_down", lambda f, d: _freq_sweep(f * 3, f, d)),
    ]

    print("=== Generating sound samples ===\n")
    for name, gen_fn in samples:
        audio, _ = make_sample(name, gen_fn, freq=800, dur_ms=80, repeats=6)
        path = os.path.join(out_dir, f"sample_{name}.wav")
        write_wav(path, audio)
        print(f"  {name:20s} -> {path}")

    print(f"\nDone. {len(samples)} samples saved to {out_dir}/")


if __name__ == "__main__":
    main()
