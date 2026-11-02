"""Small pytest file of trigger probe 5.  CUDA_VISIBLE_DEVICES=-1  C:/Users/Admin/venvs/tanitad/Scripts/python.exe -I -m pytest -q test_trigger_probe5.py"""
import hashlib
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import numpy as np
import pytest
import torch

import align_check as AC
import crops_lib as C
import encode_crops as EC
import run_p5 as R
import tp_lib as L


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_tp_lib_is_probe1s():
    assert sha(f"{HERE}/tp_lib.py") == sha("E:/Projects/TanitAD-artifacts/dagland/trigger_probe/tp_lib.py")


def test_boxes_and_token_grids_agree_with_the_encoder():
    grids = C.check_boxes()
    assert grids == EC.CROP_GRID == {"TL": (21, 72), "LEAD": (21, 60)}
    assert C.FULL_H == 3 * C.CACHE_H and C.FULL_W == 3 * C.CACHE_W
    f3, f1 = C.frame_full().f_ref, C.frame_cache().f_ref
    assert abs(f3 / f1 - 3.0) < 1e-9 and abs(f1 - 305.5774907364391) < 1e-9


def test_crop_is_a_slice_of_the_full_rectification():
    clip = "bf5ec76e-e8b5-4f0d-a9cc-4fe1687d3d3a"
    intr = C.intr_for(clip)
    vid = torch.from_numpy(np.random.default_rng(0).integers(0, 255, (1, 3, 1080, 1920), dtype=np.uint8))
    full = C.rectify(vid, intr, C.frame_full())
    for k, (r0, r1, c0, c1) in C.BOXES.items():
        assert torch.equal(full[:, :, r0:r1, c0:c1], C.rectify(vid, intr, C.frame_full(), C.BOXES[k]))


def test_pool4x10_is_the_caches_recipe():
    h = torch.randn(3, 5 + 640, 8)
    ref = torch.stack([torch.stack([h[t, 5:].reshape(16, 40, 8)[4 * r:4 * r + 4, 4 * c:4 * c + 4].mean((0, 1)) for r in range(4) for c in range(10)]) for t in range(3)])
    assert torch.allclose(EC.pool4x10(h), ref, atol=1e-6)


def test_crop_feats_layout():
    h = torch.randn(2, 5 + 21 * 72, 16)
    f = EC.crop_feats(h, 21, 72)
    assert f.shape == (2, 10, 16)
    assert torch.allclose(f[:, 0], h[:, 0].float()) and torch.allclose(f[:, 1], h[:, 5:].float().mean(1), atol=1e-6)


def test_align_gate_function():
    assert AC.verdict(np.full(32, 0.03))
    assert not AC.verdict(np.r_[np.full(31, 0.03), 0.6])        # one wrong frame (~1.0 reads 0.6+) refuses
    assert not AC.verdict(np.full(32, 0.06))


def test_controls_run_for_every_source_name():
    arms = R.control_arms()
    assert {"E", "V+E", "TL+E", "LEAD+E", "C+E", "C+V+E"} <= set(arms) and len(arms) == 6


def test_decision_rule_cells():
    ok = {"pstop": True, "pmove": True}
    assert R.rule({"pstop": (0.04, 0.01), "pmove": (0.0, -0.01)}, ok) == "CROPS HELP"
    assert R.rule({"pstop": (0.04, -0.01), "pmove": (0.0, -0.01)}, ok) == "INCONCLUSIVE"          # point estimate without CI > 0
    assert R.rule({"pstop": (0.005, -0.01), "pmove": (0.009, -0.01)}, ok) == "CROPS DO NOT HELP"
    assert R.rule({"pstop": (0.005, -0.01), "pmove": (0.009, -0.01)}, {"pstop": True, "pmove": False}) == "INCONCLUSIVE"   # an invalid cell cannot support NOT-HELP
    assert R.rule({"pstop": (0.04, 0.01)}, ok) == "INCOMPLETE"


def test_planted_signal_and_null_through_the_real_pipeline():
    rng = np.random.default_rng(0)
    n = 1500
    clip = np.repeat(np.arange(150), 10)
    sha12 = np.array([f"{int(i * 2654435761) % (16 ** 11):011x}0" for i in clip])
    outer, inner = L.outer_fold(sha12), L.inner_fold(sha12)
    y = rng.integers(0, 2, n)
    noise = rng.standard_normal((n, 64))
    planted = noise.copy()
    planted[:, 3] += 2.0 * (2 * y - 1)
    E = rng.standard_normal((n, 8))
    cnt = L.boot_counts(clip, n_boot=200)
    res = {}
    for nm, S in (("planted", planted), ("null", noise)):
        o, _ = L.cv_oof(lambda f, S=S: np.concatenate([S, E], 1), y, outer, inner, "linear")
        res[nm] = L.auc_ci_fold(o, y, clip, outer, n_boot=200, counts=cnt)
    assert res["planted"]["auc"] > 0.9
    assert res["null"]["lo"] <= 0.5 <= res["null"]["hi"]


def test_controls_ok_flags_a_missing_leak():
    def cell(auc, lo=0.45, hi=0.55):
        return {"foldmean": {"auc": auc, "lo": lo, "hi": hi}}
    res = {"controls": {}, "paired": {"C1:C+V+E_minus_V+E": {"foldmean": {"lo": -0.01, "hi": 0.01}}}}
    for nm in R.control_arms():
        res["controls"][f"C1:{nm}"] = cell(0.5)
        res["controls"][f"C2:{nm}"] = cell(0.5)
    for nm in R.LEAK_ARMS:
        res["controls"][f"C3:{nm}"] = cell(1.0)
    assert R.controls_ok(res)[0]
    res["controls"]["C3:C+V+E"] = cell(0.8)
    ok, why = R.controls_ok(res)
    assert not ok and any("C3 leak C+V+E" in w for w in why)
