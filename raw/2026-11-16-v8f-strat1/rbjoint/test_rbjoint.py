"""test_rbjoint.py -- run the WHOLE file (never -k only).  CPU only.
Mutations: a reader of frame t+1 must turn test_reader_causal RED (test_mutant_is_caught asserts that); a label rule that ignores the validity must be caught."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rbjoint as R  # noqa: E402


def _x(n=20):
    return np.arange(n, dtype=np.float64)[:, None, None] * np.ones((n, 5, 3))


def causal_ok(reader, grid):
    """window for t uses frames <= t only and is unchanged when every frame > t is replaced by garbage"""
    x = _x()
    for t in (3, 9, 18):
        w = reader(x, t, grid)
        if w[:, 0, 0].max() > t:
            return False
        y = x.copy()
        y[t + 1:] = -999.0
        if not np.array_equal(reader(y, t, grid), w):
            return False
    return True


@pytest.mark.parametrize("grid", ["X4", "X8"])
def test_reader_causal(grid):
    assert causal_ok(R.read_window, grid)
    assert list(R.read_window(_x(), 9, "X4")[:, 0, 0]) == [6, 7, 8, 9]
    assert list(R.read_window(_x(), 9, "X8")[:, 0, 0]) == [6, 9]


@pytest.mark.parametrize("grid", ["X4", "X8"])
def test_mutant_is_caught(grid):
    mutant = lambda x, t, g: R.read_window(x, t, g, _shift=1)  # noqa: E731
    assert not causal_ok(mutant, grid)


def test_fold_clip_disjoint_and_deterministic():
    ids = [f"clip-{i}" for i in range(500)]
    f = np.array([R.fold_of(c) for c in ids])
    assert set(f.tolist()) == set(range(5))
    assert (f == np.array([R.fold_of(c) for c in ids])).all()
    # a window-level split (the mutant) would not be clip-disjoint: clip-level folds are by construction
    clip = np.repeat(np.arange(500), 7)
    fw = f[clip]
    for k in range(5):
        assert len(set(clip[fw == k]) & set(clip[fw != k])) == 0


def test_yaw_class():
    assert R.yaw_class([0.3, -0.3, 0.0, 0.05, -0.05, 0.0501]).tolist() == [1, 2, 0, 0, 0, 1]


def test_shuffle_moves_labels_between_clips():
    y = np.repeat(np.arange(5), 100)
    ys = R.shuffle_labels(y, 0)
    assert sorted(ys.tolist()) == sorted(y.tolist()) and (ys != y).mean() > 0.6
    assert (R.shuffle_labels(y, 0) == ys).all()


def test_ed_matrix_shape_and_inputs():
    X = np.array([[1, 3, 2, 0.5, 1.0], [1, 0, 0, 0.0, 0.0], [1, 99, 4, 3.0, 1.0]])
    E = R.ed_matrix(X)
    assert E.shape == (3, R.ED_DIM)
    assert E[0, 1 + 3] == 1 and E[0, 1 + R.EXIT_BINS + 2] == 1 and E[2, 1 + R.EXIT_BINS - 1] == 1


def _P(**kw):
    base = {k: {"delta": 0.0, "separated": False} for k in ("VED8-S", "VED8-ED", "VED8-RAW8", "RAW8-S", "RAW8-RAW4", "RAW4-S", "ED-S")}
    for k, v in kw.items():
        base[k.replace("_", "-")] = {"delta": v, "separated": True}
    return base


def test_labels_rule_cases():
    m = R.MARGIN
    assert R.labels_rule(_P(), True) == (["SILENT"], "SILENT")
    assert R.labels_rule(_P(VED8_S=m, VED8_ED=m, VED8_RAW8=m), True)[1] == "JOINT"
    assert R.labels_rule(_P(VED8_S=m, VED8_ED=m), True)[0] == ["SILENT"]          # missing the RAW8 leg: not JOINT
    assert R.labels_rule(_P(RAW8_S=m, RAW8_RAW4=m), True) == (["GRID", "VIS"], "GRID")
    assert R.labels_rule(_P(RAW4_S=m), True) == (["VIS"], "VIS")
    assert R.labels_rule(_P(ED_S=m), True) == (["TEXT"], "TEXT")
    assert R.labels_rule(_P(ED_S=m - 1e-6), True)[0] == ["SILENT"]
    p = _P(RAW4_S=m)
    p["RAW4-S"]["separated"] = False
    assert R.labels_rule(p, True)[0] == ["SILENT"]                                  # not separated
    assert R.labels_rule(_P(RAW4_S=m), False) == (["INVALID"], "INVALID")           # validity failure voids every label


def test_pos_yaw_rule():
    assert R.pos_yaw_pass({"delta": 0.10, "separated": True})
    assert not R.pos_yaw_pass({"delta": 0.0999, "separated": True})
    assert not R.pos_yaw_pass({"delta": 0.3, "separated": False})


def test_gather_reads_only_frames_le_t(tmp_path):
    """extract's reader over a synthetic clip: each frame filled with its index; the window values must be frames <= t"""
    torch = pytest.importorskip("torch")
    n = 12
    f4 = torch.arange(n, dtype=torch.float32)[:, None, None].expand(n, 40, 8).clone().to(torch.float8_e4m3fn)
    f8 = torch.arange(n, dtype=torch.float32)[:, None, None].expand(n, 160, 8).clone().to(torch.float8_e4m3fn)
    for t in (3, 7, 11):
        assert R.read_window(f4, t, "X4").float()[:, 0, 0].tolist() == [t - 3, t - 2, t - 1, t]
        assert R.read_window(f8, t, "X8").float()[:, 0, 0].tolist() == [t - 3, t]
    with pytest.raises(AssertionError):
        R.read_window(f4, 11, "X4", _shift=1)       # t+1 is out of the clip: the mutant cannot pass silently at the end


def test_probe_learns_planted_signal_and_ed_head_path():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    R.SHAPES["XT"] = (2, 4)
    n = 120
    y = (np.arange(n) % 2).astype(np.int64)
    x = np.random.RandomState(0).randint(0, 120, size=(n, 2, 4, 1024)).astype(np.uint8)
    x[y == 1, :, :, :256] = np.clip(x[y == 1, :, :, :256].astype(int) + 60, 0, 120).astype(np.uint8)
    F = torch.from_numpy(x)
    old = (R.EPOCHS, R.CKPTS)
    R.EPOCHS, R.CKPTS = 4, (2, 4)
    try:
        idx = np.arange(n)
        r = R.fit_arm("XT", F, y, idx[:90], idx[90:], {"te": (F, None, idx[90:])}, 2, None, 0, lambda *a: None)
        assert (r["logits"]["te"].argmax(1) == y[90:]).mean() > 0.8
        ED = torch.zeros(n, R.ED_DIM)
        r2 = R.fit_arm("XT", F, y, idx[:90], idx[90:], {"te": (F, ED, idx[90:])}, 2, ED, 0, lambda *a: None)
        assert r2["logits"]["te"].shape == (30, 2)
    finally:
        R.EPOCHS, R.CKPTS = old
        R.SHAPES.pop("XT", None)


def test_no_kinematic_input_in_vision_arms():
    """the arms' inputs are the cached pooled-grid frames, the ED route-text matrix, and nothing else; yaw is only a target"""
    import inspect
    src = inspect.getsource(R.extract) + inspect.getsource(R.fit_arm)
    assert "kin3" not in src and "pose_grid" not in src and "steer" not in src and "p10" not in src
    assert R.ARMS_VIS["YAW4"][1] == "yaw" and all(v[1] == "tac" for k, v in R.ARMS_VIS.items() if "YAW" not in k)
