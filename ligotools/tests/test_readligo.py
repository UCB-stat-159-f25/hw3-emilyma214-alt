import os
import sys
import numpy as np

this_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
sys.path.insert(0, repo_root)

from ligotools import readligo as rl



def test_loaddata_strain_and_time_lengths_match():
    """
    Integration-style test using the real H1 LOSC file.

    Checks that:
    - loaddata returns non-empty strain and time arrays
    - strain and time have the same length
    - channel_dict is a non-empty dict and has a DEFAULT key
    """
    # Make sure the filename matches exactly what is in your data/ folder.
    fn_H1 = os.path.join("data", "H-H1_LOSC_4_V2-1126259446-32.hdf5")

    strain, time, channel_dict = rl.loaddata(fn_H1, ifo="H1")

    # Basic sanity checks
    assert strain is not None
    assert time is not None

    # Lengths of strain and time should match
    assert len(strain) == len(time)

    # channel_dict should be a non-empty dict with DEFAULT key
    assert isinstance(channel_dict, dict)
    assert len(channel_dict) > 0
    assert "DEFAULT" in channel_dict


def test_read_hdf5_returns_positive_ts_and_nonzero_strain():
    """
    Unit-ish test on read_hdf5: we don't care about exact numbers,
    just that:
    - strain is non-empty and not all zeros
    - ts (time spacing) is positive
    - gpsStart is positive
    """
    fn_H1 = os.path.join("data", "H-H1_LOSC_4_V2-1126259446-32.hdf5")

    strain, gpsStart, ts, qmask, shortnames, injmask, injnames = rl.read_hdf5(fn_H1)

    # Non-empty strain
    assert isinstance(strain, np.ndarray)
    assert strain.size > 0
    assert not np.all(strain == 0)

    # Positive time spacing and reasonable GPS start
    assert ts > 0
    assert gpsStart > 0


def test_dq_channel_to_seglist_simple_pattern():
    """
    Pure unit test of dq_channel_to_seglist using a synthetic channel.

    Channel:
    index:  0 1 2 3 4 5 6 7 8
    value:  0 0 1 1 1 0 1 1 0

    With fs=1, we expect two segments:
    [2,5) and [6,8) -> slices slice(2,5) and slice(6,8)
    """
    channel = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0])

    segs = rl.dq_channel_to_seglist(channel, fs=1)

    # We should get exactly two segments
    assert len(segs) == 2

    first, second = segs

    # First segment from index 2 to 5
    assert isinstance(first, slice)
    assert first.start == 2
    assert first.stop == 5

    # Second segment from index 6 to 8
    assert isinstance(second, slice)
    assert second.start == 6
    assert second.stop == 8
