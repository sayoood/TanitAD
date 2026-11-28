"""test_rbvis.py -- run the WHOLE file (never -k only).  CPU only.  Mutation: a reader of frame t+1 must turn test_reader_causal RED (test_mutant_is_caught asserts that)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rbvis as R  # noqa: E402


def _x(n=20):
    x = np.arange(n, dtype=np.float64)[:, None, None] * np.ones((n, 40, 1024))
    return x


def causal_ok(reader):
    """the window for t must be frames t-3..t and unchanged when every frame > t is replaced by garbage"""
    x = _x()
    for t in (3, 9, 19):
        w = reader(x, t)
        if not np.array_equal(w[:, 0, 0], np.arange(t - 3, t + 1)):
            return False
        y = x.copy()
        y[t + 1:] = -999.0
        if not np.array_equal(reader(y, t), w) if t + 1 < len(x) else False:
            return False
    return True


def test_reader_causal():
    assert causal_ok(R.read_window)


def test_mutant_is_caught():
    mutant = lambda x, t: R.read_window(x, t, _shift=1) if t + 1 < len(x) else R.read_window(x, t, _shift=0)  # noqa: E731
    assert not causal_ok(mutant)


def test_shift_regression_differs():
    x = _x()
    assert not np.array_equal(R.read_window(x, 9), R.read_window(x, 9, _shift=1))


def test_reader_refuses_t_lt_3():
    with pytest.raises(AssertionError):
        R.read_window(_x(), 2)


def test_subsample_deterministic_per_clip():
    rows = np.arange(10)
    clip = np.array([0, 0, 0, 0, 0, 1, 1, 1, 2, 2])
    k = R.subsample(rows, clip)
    assert k.tolist() == [True, False, True, False, True, True, False, True, True, False]
    assert np.array_equal(k, R.subsample(rows, clip))


def test_timing_and_verdict_rules():
    assert R.timing_cls(np.array([0, 1, 2, 3, 4])).tolist() == [0, 1, 1, 2, 2]
    sep = {"delta": 0.05, "separated": True}
    uns = {"delta": 0.05, "separated": False}
    low = {"delta": 0.02, "separated": True}
    assert R.verdict([sep, sep], [sep, sep])["reading"] == "VIS-CARRIES+HEAD-GAP"
    assert R.verdict([sep, sep], [sep, low])["reading"].startswith("VIS-CARRIES (head")
    assert R.verdict([sep, uns], [sep, sep])["reading"] == "VIS-SILENT"
    assert R.verdict([uns, uns], [sep, sep])["suggestive_unseparated"] is True


def test_probe_has_only_feature_input_and_runs():
    import torch
    A, P = R.make_nets()
    ad, pr = A(None, rand_seed=0), P(0)
    f = torch.randn(3, 4, 40, 1024).to(torch.float8_e4m3fn)
    out = pr(ad.tokens(f).flatten(1, 2))
    assert out.shape == (3, 5)
    assert not any(p.requires_grad for p in ad.parameters())


def test_label_never_in_reader_signature():
    import inspect
    assert list(inspect.signature(R.read_window).parameters) == ["x", "t", "_shift"]


def test_adapter_loads_checkpoint_tensors():
    import torch
    A, _ = R.make_nets()
    sd = {"adapter.0.weight": torch.ones(1024), "adapter.0.bias": torch.zeros(1024), "adapter.1.weight": torch.zeros(256, 1024) + 0.5,
          "adapter.1.bias": torch.zeros(256), "pos": torch.zeros(40, 256) + 2, "time": torch.zeros(4, 256) + 3}
    a = A(sd)
    assert float(a.lin.weight.mean()) == 0.5 and float(a.pos.mean()) == 2 and float(a.time.mean()) == 3
