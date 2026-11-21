"""p8_bev_reel — the compositor must be honest: belief and truth stay separable,
and the frame builds without any GPU/corpus dependency."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import p8_bev_reel as R                                              # noqa: E402


@pytest.fixture()
def occ():
    a = np.zeros((120, 64), np.float32)
    a[10:14, 30:34] = 1.0          # an agent ahead
    b = np.zeros((120, 64), np.float32)
    b[10:14, 30:34] = 1.0          # same agent (agreement)
    b[40:44, 20:24] = 1.0          # one the belief missed
    return a, b


def test_lazy_imports_only():
    # module must import with no torch-CUDA / corpus present
    assert "torch" not in dir(R)
    assert hasattr(R, "compose_frame") and hasattr(R, "overlay_rgb")


def test_raster_orientation_forward_is_up(occ):
    pred, _ = occ
    img = R.raster_to_rgb(pred, R.BELIEF)
    assert img.shape == (120, 64, 3)
    # x=0 (ego) is the LAST row after the flip → ego marker at the bottom
    assert img[-2, 32].tolist() == [248, 250, 252]


def test_overlay_separates_belief_truth_agreement(occ):
    pred, gt = occ
    img = R.overlay_rgb(pred, gt)
    cols = {tuple(c) for c in img.reshape(-1, 3)}
    assert tuple(R.AGREE) in cols          # cells both marked
    assert tuple(R.TRUTH) in cols          # GT-only cell (the miss)
    # a belief-only cell appears when the prediction over-reaches
    over = pred.copy()
    over[80:84, 10:14] = 1.0
    cols2 = {tuple(c) for c in R.overlay_rgb(over, gt).reshape(-1, 3)}
    assert tuple(R.BELIEF) in cols2


def test_compose_frame_shape_and_dtype(occ):
    pred, gt = occ
    cam = (np.random.default_rng(0).random((176, 624, 3)) * 255).astype(np.uint8)
    f = R.compose_frame(cam, pred, gt, "window 1/12  k=10", "sub")
    assert f.dtype == np.uint8 and f.ndim == 3 and f.shape[2] == 3
    assert f.shape[1] == 1280 and f.shape[0] > R.PANE_H


def test_compose_frame_accepts_float_and_stacked_camera(occ):
    pred, gt = occ
    cam9 = np.random.default_rng(1).random((176, 624, 9)).astype(np.float32)
    f = R.compose_frame(cam9, pred, gt, "stacked-history camera")
    assert f.shape[2] == 3


def test_tau_star_reads_gate_json(tmp_path):
    import json
    (tmp_path / "p8_gate.json").write_text(json.dumps({"mini_eval": {"tau_star": 0.2}}))
    assert R._tau_star(str(tmp_path)) == 0.2
    assert R._tau_star(str(tmp_path / "missing")) == 0.5   # honest default
