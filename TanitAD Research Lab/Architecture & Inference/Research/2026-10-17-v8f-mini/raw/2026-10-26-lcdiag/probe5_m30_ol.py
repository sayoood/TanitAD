"""PROBE 5 (DESCRIPTIVE): is the curvature already in the OPEN-LOOP plan? L2 (arm - P10) on the m = 30 dumps (one shot: the executed track IS the tick-0
plan, logged state), the m = 5 dumps, and the m = 10 tick-0 plan `ol` scored as a track; split into probe 3's 'slow windows' (HACC m10 causal set) and the rest.
python probe5_m30_ol.py > probe5.log"""
import os
import sys
import numpy as np
sys.path.insert(0, "C:/Users/Admin/lcdiag")
import common as C  # noqa: E402
for s in (0, 1):
    ref = C.load("causal", s)
    pa, pf = C.pair_err(ref["cl"], ref["g"]), C.pair_err(ref["fl"], ref["g"])
    okr = (pa["c"] > 0) & (pf["c"] > 0)
    W = (pa["fast"] & (pa["vX"] < C.VMIN) & okr[:, None]).any(1)
    print(f"== s{s}: slow-window set from HACC m10 causal: {W.sum()} windows")
    for tag, path in (("m10 causal", None), ("m30 logged", f"{C.SCR}/consolidated/dumps_HACC_s{s}_m30.npz"),
                      ("m5 logged", f"{C.SCR}/consolidated/dumps_HACC_s{s}_m5.npz"), ("m10 frozen-scene", f"{C.SCR}/consolidated/dumps_HACC_s{s}_m10_frozen.npz"),
                      ("m10 limit-off", f"{C.SCR}/consolidated/dumps_HACC_s{s}_m10_limoff.npz")):
        d = ref if path is None else {k: v for k, v in np.load(path).items()}
        assert np.array_equal(d["eid"], ref["eid"]) and np.allclose(d["g"], ref["g"])
        for arm in ("cl", "ol"):
            if tag != "m10 causal" and arm == "ol":
                continue
            X = d[arm]
            a, f = C.pair_err(X, d["g"]), C.pair_err(d["fl"], d["g"])
            ok = (a["c"] > 0) & (f["c"] > 0)
            tot = C.boot(a["sh"].sum(1)[ok], f["sh"].sum(1)[ok], d["eid"][ok])
            exc = (a["sh"] - f["sh"]).sum(1)
            slowfrac = float((a["fast"] & (a["vX"] < C.VMIN)).sum() / a["fast"].sum())
            print(f"  {tag:17s} {arm}: L2 {tot['delta']:+.4f} [{tot['lo']:+.4f}, {tot['hi']:+.4f}] | from slow-window set {exc[W & ok].sum() / ok.sum():+.4f}, rest {exc[~W & ok].sum() / ok.sum():+.4f} | arm-slow frac of GT-fast pairs {slowfrac:.4f}")
