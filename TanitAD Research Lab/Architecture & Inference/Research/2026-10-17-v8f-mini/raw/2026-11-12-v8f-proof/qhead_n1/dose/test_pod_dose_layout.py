"""Tests of pod_dose_layout.py: the layout arrays, the refusals, and that the source reads nothing but the kinematics."""
import json
import os

import numpy as np
import pytest

import dose_testlib as TL  # noqa: E402
import pod_dose_layout as PL  # noqa: E402


def fake_loader(frames, seed=0):
    rng = np.random.RandomState(seed)
    data = {"k%d" % i: tuple(rng.randn(f).astype(np.float32) + j for j in range(3)) for i, f in enumerate(frames)}
    return data, (lambda c: data[c])


def test_layout_arrays_equal_the_loader_values_clip_by_clip():
    frames = [5, 7, 6]
    data, fn = fake_loader(frames)
    L = PL.build_layout(["k0", "k1", "k2"], fn)
    assert list(L["clips"]) == ["k0", "k1", "k2"] and list(L["frames"]) == frames
    assert L["clip_idx"].dtype == np.int32 and L["t"].dtype == np.int16 and L["kin"].dtype == np.float32 and L["kin"].shape == (18, 3)
    r = 0
    for i, f in enumerate(frames):
        v, a, y = data["k%d" % i]
        assert np.array_equal(L["kin"][r:r + f, 0], v) and np.array_equal(L["kin"][r:r + f, 1], a) and np.array_equal(L["kin"][r:r + f, 2], y)
        assert np.array_equal(L["t"][r:r + f], np.arange(f)) and (L["clip_idx"][r:r + f] == i).all()
        r += f


def test_clip_index_follows_the_order_given():
    _, fn = fake_loader([4, 4])
    L = PL.build_layout(["k1", "k0"], fn)
    assert list(L["clips"]) == ["k1", "k0"] and list(L["clip_idx"][:4]) == [0] * 4 and list(L["clip_idx"][4:]) == [1] * 4


def test_unequal_kinematic_lengths_are_refused():
    def fn(c):
        return np.zeros(5), np.zeros(4), np.zeros(5)
    with pytest.raises(SystemExit) as e:
        PL.build_layout(["k0"], fn)
    assert "REFUSED" in str(e.value)


def test_empty_clip_is_refused():
    with pytest.raises(SystemExit):
        PL.build_layout(["k0"], lambda c: (np.zeros(0), np.zeros(0), np.zeros(0)))


def test_main_refuses_an_existing_output_before_reading_anything(tmp_path):
    out = tmp_path / "lay.npz"
    out.write_bytes(b"x")
    with pytest.raises(SystemExit) as e:
        PL.main(["--tree", str(tmp_path / "no_tree"), "--cache", str(tmp_path / "no_cache"), "--split", "train", "--lo", "0", "--hi", "10", "--out", TL.sh(out)])
    assert "exists" in str(e.value)
    assert out.read_bytes() == b"x"


def test_main_refuses_split_other_than_train():
    with pytest.raises(SystemExit):
        PL.main(["--tree", "t", "--cache", "c", "--split", "eval", "--lo", "0", "--hi", "1", "--out", "o"])


def test_source_reads_only_the_kinematic_keys_and_never_the_tokens_values():
    src = open(os.path.join(TL.SRC, "pod_dose_layout.py"), encoding="utf-8").read()
    for k in ('d["v"]', 'd["a"]', 'd["yaw_rate"]'):
        assert k in src
    assert 'd["pool4x10"].shape' in src and 'd["pool4x10"].float' not in src and "cuda()" not in src and "to(\"cuda" not in src


def test_saved_layout_loads_with_every_key(tmp_path):
    _, fn = fake_loader([3, 4])
    L = PL.build_layout(["k0", "k1"], fn)
    p = tmp_path / "l.npz"
    np.savez(p, **L)
    z = np.load(p, allow_pickle=False)
    assert set(z.files) == {"clips", "clip_idx", "t", "kin", "frames"}
