import os
import sys
import numpy as np
from scipy.io import wavfile

# Make sure the repo root (which contains the 'ligotools' folder) is on sys.path
this_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
sys.path.insert(0, repo_root)

from ligotools.utils import whiten, write_wavfile, reqshift


def test_whiten_preserves_length_and_finiteness():
    """
    whiten() should return an array of the same shape as input
    and with finite (non-NaN, non-inf) values.
    """
    dt = 1.0 / 1024
    t = np.arange(0, 1, dt)
    strain = np.sin(2 * np.pi * 50 * t)  # simple sinusoid

    # flat PSD = 1 at all frequencies
    interp_psd = lambda f: np.ones_like(f)

    w = whiten(strain, interp_psd, dt)

    # same shape
    assert w.shape == strain.shape
    # all finite
    assert np.all(np.isfinite(w))


def test_reqshift_zero_shift_returns_same_signal():
    """
    If fshift=0, reqshift should not change the signal (apart from tiny FP error).
    """
    fs = 1024
    t = np.linspace(0, 1, fs, endpoint=False)
    data = np.sin(2 * np.pi * 100 * t)

    shifted = reqshift(data, fshift=0, sample_rate=fs)

    # Should be almost exactly the same
    assert np.allclose(shifted, data)


def test_write_wavfile_creates_valid_wav(tmp_path):
    """
    write_wavfile() should create a readable WAV file with the expected
    sample rate and length.
    """
    fs = 1024
    t = np.linspace(0, 1, fs, endpoint=False)
    data = 0.5 * np.sin(2 * np.pi * 50 * t)

    out_path = tmp_path / "test.wav"
    write_wavfile(str(out_path), fs, data)

    # file exists
    assert out_path.exists()

    # can be read and has expected properties
    rate, d = wavfile.read(str(out_path))
    assert rate == fs
    assert len(d) == len(data)
