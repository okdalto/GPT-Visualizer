"""Shared audio constants and WAV I/O."""

import wave
import numpy as np

SR = 44100


def write_wav(filename, data, sr=SR):
    """Write stereo 16-bit WAV file."""
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
