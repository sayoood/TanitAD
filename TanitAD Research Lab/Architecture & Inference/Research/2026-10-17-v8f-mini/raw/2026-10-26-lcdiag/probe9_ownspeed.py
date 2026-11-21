"""PROBE 9 (DESCRIPTIVE): at the SAME own executed speed, is HACC's curvature error on GT-fast pairs worse than P10's? Mean |dk| (1/m) per own-speed bin,
GT-fast pairs only, per arm, and the counterfactual L2 excess if HACC's per-bin error were P10's (frequency vs severity decomposition).
python probe9_ownspeed.py > probe9.log"""
import sys
import numpy as np
sys.path.insert(0, "C:/Users/Admin/lcdiag")
import common as C  # noqa: E402
bins = [0.5, 1, 2, 3.5, 5.2, 8.6, 11.8, 99]
for cfg in ("causal", "plainB_logged"):
    for s in (0, 1):
        d = C.load(cfg, s)
        pa, pf = C.pair_err(d["cl"], d["g"]), C.pair_err(d["fl"], d["g"])
        ok = (pa["c"] > 0) & (pf["c"] > 0)
        line = []
        cf = np.zeros_like(pa["e"])                       # counterfactual: HACC pair error replaced by P10's mean error in the same own-speed bin
        for lo, hi in zip(bins[:-1], bins[1:]):
            ma = pa["fast"] & ok[:, None] & (pa["vX"] >= lo) & (pa["vX"] < hi)
            mf = pf["fast"] & ok[:, None] & (pf["vX"] >= lo) & (pf["vX"] < hi)
            ea = float(pa["e"][ma].mean()) if ma.any() else np.nan
            ef = float(pf["e"][mf].mean()) if mf.any() else np.nan
            line.append(f"{lo}-{hi}: H {ea:.4f} (n {int(ma.sum())}) F {ef:.4f} (n {int(mf.sum())})")
            cf[ma] = ef if np.isfinite(ef) else ea
        sh_cf = np.where(pa["fast"], cf, 0) / np.maximum(pa["c"], 1)[:, None]
        tot = (pa["sh"] - pf["sh"])[ok].sum(1).mean()
        tot_cf = (sh_cf - pf["sh"])[ok].sum(1).mean()
        print(f"== {cfg} s{s}: L2 excess {tot:+.4f}; if HACC had P10's per-own-speed-bin error: {tot_cf:+.4f} (frequency part), severity part {tot - tot_cf:+.4f}")
        for l in line:
            print("   ", l)
