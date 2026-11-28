"""Known-answer tests for e0h_dose.py (tier 2: analytic controls, no mutation suite).  Synthetic GT planes with a KNOWN sidewalk area, so the calibration target is a literal."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e0h_dose as H  # noqa: E402


def _gts(n=6):
    """n windows; each: drivable (code 1) everywhere, plus a sidewalk block of exactly 20 x 40 = 800 cells inside the corridor of band 10-20 m (rows 120..139, cols 280..319)."""
    out = []
    for _ in range(n):
        g = np.ones((1000, 600), np.uint8)
        g[120:140, 280:320] = H.SW
        out.append(g)
    return out


def test_all_p1_is_the_gt_sidewalk_plane_exactly():
    g = _gts(1)[0]
    pl = H.synth_planes(g, None, 10, "dose")
    assert np.array_equal(pl["psw"], 255 * (g == H.SW).astype(np.uint8))
    assert int((pl["psw"] > 0).sum()) == 800


def test_all_flag_map_is_constant():
    pl = H.synth_planes(_gts(1)[0], None, 10, "all")
    assert pl["psw"].min() == 255 and pl["pdrv"].min() == 255


def test_calibration_hits_p_literally():
    gts = _gts(6)                                         # TP in band 10-20 = 6 x 800 = 4800 (literal); other bands TP = 0 -> no stamps there
    for p, fp_expect in ((0.5, 4800), (0.8, 1200)):       # FP = TP (1-p)/p : 4800 and 1200 (literals)
        st, rz = H.draw_stamps(gts, p, 10, seed=1)
        r = rz["10-20"]
        assert r["TP"] == 4800
        assert fp_expect <= r["FP"] <= fp_expect + 100     # overshoot < one stamp (100 cells)
        assert abs(r["precision"] - p) < 0.01
        assert all(rz[b]["FP"] == 0 for b in rz if b != "10-20")


def test_false_flags_never_on_gt_sidewalk_and_never_outside_corridor():
    gts = _gts(6)
    st, _ = H.draw_stamps(gts, 0.5, 10, seed=2)
    for k, lst in st.items():
        pl = H.synth_planes(gts[k], lst, 10, "dose")
        fp = (pl["psw"] > 0) & (gts[k] != H.SW)
        assert not (fp & (gts[k] == H.SW)).any()
        cols = np.nonzero(fp.any(0))[0]
        assert cols.min() >= H.CORR[0] and cols.max() < H.CORR[1]
        rows = np.nonzero(fp.any(1))[0]
        assert rows.min() >= 100 and rows.max() < 200       # only band 10-20 m has TP


def test_p1_draws_nothing():
    st, rz = H.draw_stamps(_gts(3), 1.0, 10, seed=3)
    assert st == {} and rz["10-20"]["FP"] == 0 and rz["10-20"]["precision"] == 1.0
