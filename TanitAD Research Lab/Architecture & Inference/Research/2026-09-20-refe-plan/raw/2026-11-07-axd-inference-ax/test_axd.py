"""YAXD CPU unit tests (CUDA_VISIBLE_DEVICES=-1): transforms, every gate with its deliberate-regression (RED) arm, the gain arithmetic and the committed rule."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import axd_common as A  # noqa: E402
import axd_analyze as AN  # noqa: E402
import ax_stats as S  # noqa: E402

RNG = np.random.default_rng(0)


def test_transforms_exact():
    for r in RNG.normal(0, 1.2, 200).astype(np.float32):
        assert A.transform("IDT", float(r), None) == r                      # identity is bit-exact on float32
        assert A.transform("AX0", float(r), None) == 0.0
        assert A.transform("CLAMP0", float(r), None) == max(r, np.float32(0))
        assert A.transform("CLAMPm05", float(r), None) == max(r, np.float32(-0.5))
        assert A.transform("SCALE05", float(r), None) == np.float32(r) * np.float32(0.5)
    h = np.array([-1.0, -0.5, 0.0])
    assert A.transform("SMOOTH4", 1.5, h) == np.float32(0.0)
    assert A.transform("SMOOTH4", 1.5, h, -0.5, "inst") == np.float32(-0.125)
    assert A.transform("SMOOTH4", 1.5, h, -0.5, "sust") == np.float32(-0.5)
    assert A.transform("IDT", 0.2, None, 0.5, "abs") == np.float32(0.5)


def _rec(n=30, nc=12, regress=None):
    raw = RNG.normal(0, 1, (n, 7)).astype(np.float32)
    hist = RNG.normal(0, 1, (n, 3))
    a0p = RNG.normal(0, 5, (n, 8, 3)).astype(np.float32)
    rec = {"raw": raw, "hist": hist, "a0_poses": a0p, "a0_seam_poses": a0p.copy(), "ax_export_t0": raw[:, 2].astype(np.float64) + 1e-7,
           "cruise_raw": raw[:nc], "cruise_hist": hist[:nc], "cruise_token": np.arange(nc)}
    for arm in ("IDT", "IDTm") + A.CANDIDATES:
        t = "IDT" if arm == "IDTm" else arm
        fed = raw.copy()
        if not (regress == "transform_not_applied"):
            fed[:, 2] = AN.expected_fed(t, raw, hist)
        rec[f"fed__{arm}"] = fed
        rec[f"poses__{arm}"] = a0p.copy() if t == "IDT" else a0p + 0.1
    for name, t, d, mode in A.gain_arms():
        fed = raw[:nc].copy()
        fed[:, 2] = AN.expected_fed(t, raw[:nc], hist[:nc], d, mode) if regress != "transform_not_applied" else raw[:nc, 2]
        rec[f"gfed__{name}"] = fed
        rec[f"gpose__{name}"] = (a0p[:nc] + 0.1) if t == "AX0" else a0p[:nc] + 0.01 * d
    if regress == "ax0_leak":
        rec["gpose__AX0@+"] = rec["gpose__AX0@+"] + 0.05
    if regress == "idt_perturbed":
        rec["fed__IDT"][:, 2] += np.float32(1e-3)
        rec["poses__IDT"] = rec["poses__IDT"] + 1e-4
    if regress == "memo_stale":
        rec["poses__IDTm"] = rec["poses__IDTm"] + 1e-3
    return rec


def test_gates_green():
    r = _rec()
    assert AN.gate_idt(r)["PASS"] and AN.gate_memo(r)["PASS"] and AN.gate_hist(r)["PASS"] and AN.gate_reach_idt_gain(r)["PASS"]
    for t in A.CANDIDATES:
        assert AN.gate_reach(r, t)["PASS"], t
    assert AN.gate_zero(r, r["poses__AX0"][:12])["PASS"]


def test_red_transform_not_applied():
    r = _rec(regress="transform_not_applied")
    assert not AN.gate_reach(r, "CLAMP0")["PASS"] and not AN.gate_reach(r, "AX0")["PASS"] and not AN.gate_reach_idt_gain(r)["PASS"]


def test_red_ax0_leak():
    r = _rec(regress="ax0_leak")
    assert not AN.gate_zero(r, r["poses__AX0"][:12])["PASS"]


def test_red_other_slot_and_no_change():
    r = _rec()
    r["fed__SCALE05"][:, 0] += 1.0
    assert not AN.gate_reach(r, "SCALE05")["PASS"]
    r = _rec()
    r["raw"][:, 2] = 0.7                                                    # nothing for CLAMP0 to change -> reach unproven
    r["fed__CLAMP0"] = r["raw"].copy()
    assert AN.gate_reach(r, "CLAMP0")["n_tokens_changed"] == 0 and not AN.gate_reach(r, "CLAMP0")["PASS"]


def test_red_idt_and_memo():
    assert not AN.gate_idt(_rec(regress="idt_perturbed"))["PASS"]
    assert not AN.gate_memo(_rec(regress="memo_stale"))["PASS"]
    r = _rec()
    r["a0_seam_poses"] = r["a0_seam_poses"] + 1e-5
    assert not AN.gate_idt(r)["PASS"]


def test_red_hist():
    r = _rec()
    r["ax_export_t0"] = r["ax_export_t0"] + 0.01
    assert not AN.gate_hist(r)["PASS"]


def test_gain_arithmetic():
    """a planner whose v@4s is exactly v + g * ax_fed: the instrument must return g (IDT), g/2 (SCALE05), 0 (AX0)."""
    n, g = 40, 1.1
    logs = [f"L{i % 8}" for i in range(n)]
    bt = S.Boot(logs, n=500)
    raw = np.zeros((n, 7), np.float32)
    raw[:, 2] = RNG.uniform(-0.25, 0.25, n)
    hist = np.zeros((n, 3))

    def poses_for(axf):                                                    # straight line along x with p8 - p7 = 0.5 * v4
        v = 8.0 + g * axf
        p = np.zeros((n, 8, 3), np.float32)
        p[:, :, 0] = np.arange(1, 9)[None] * 0.5 * v[:, None]
        return p
    rec = {}
    for t in ("IDT", "SCALE05", "AX0", "CLAMP0"):
        rec[f"poses__{t}"] = poses_for(AN.expected_fed(t, raw, hist).astype(np.float64))
        for name, tt, d, mode in A.gain_arms():
            if tt == t:
                rec[f"gpose__{name}"] = poses_for(AN.expected_fed(t, raw, hist, d, mode).astype(np.float64))
    sel = np.arange(n)
    gi = AN.gains(rec, bt, "IDT", sel)
    assert abs(gi["down"]["point"] - g) < 1e-4 and abs(gi["sym"]["point"] - g) < 1e-4 and abs(gi["probe_literal_sym"]["point"] - g) < 1e-4
    assert abs(AN.gains(rec, bt, "SCALE05", sel)["down"]["point"] - g / 2) < 1e-4
    assert AN.gains(rec, bt, "AX0", sel)["down"]["point"] == 0.0
    gc = AN.gains(rec, bt, "CLAMP0", sel)["down"]["point"]
    assert 0.0 <= gc < 0.3 * g                                             # ax0 in (-0.25, 0.25): the clamp removes most of the down-side


def test_rule():
    gi = {"down": {"point": 1.1, "ci95": [0.95, 1.3]}}
    good = {"down": {"point": 0.3, "ci95": [0.1, 0.5]}}
    p_ok, p_bad, p_mid = {"ci95": [-0.3, 0.4]}, {"ci95": [-2.0, -0.8]}, {"ci95": [-0.9, 0.1]}
    assert AN.decide("X", [], p_ok, good, gi, 0.1, True)["verdict"] == "CANDIDATE"
    assert AN.decide("X", ["G-IDT"], p_ok, good, gi, 0.1, True)["verdict"] == "VOID"
    assert AN.decide("X", [], p_bad, good, gi, 0.1, True)["verdict"] == "REFUTED"
    assert AN.decide("X", [], p_ok, {"down": {"point": 0.9, "ci95": [0.7, 1.1]}}, gi, 0.1, True)["verdict"] == "REFUTED"
    r = AN.decide("X", [], p_mid, good, gi, 0.1, True)
    assert r["verdict"] == "NOT PROVEN" and r["failing"] == ["pdms_ni"]
    assert AN.decide("X", [], p_ok, good, gi, 0.4, True)["failing"] == ["d_lon"]
    assert AN.decide("X", [], p_ok, good, gi, 0.1, False)["failing"] == ["cruise_floor"]
    assert AN.decide("X", [], p_ok, {"down": {"point": 0.5, "ci95": [0.2, 0.96]}}, gi, 0.1, True)["failing"] == ["gain_ci_abs", "gain_ci_rel"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    bad = 0
    for f in fns:
        try:
            f()
            print("PASS", f.__name__)
        except Exception as e:
            bad += 1
            print("FAIL", f.__name__, repr(e))
    print(f"{len(fns) - bad}/{len(fns)} PASS")
    sys.exit(1 if bad else 0)
