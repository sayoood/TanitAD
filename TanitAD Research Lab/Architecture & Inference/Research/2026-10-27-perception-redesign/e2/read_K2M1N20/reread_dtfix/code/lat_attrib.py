"""lat_attrib.py -- MM 4a3d05, 2026-11-12: where does K2's LATERAL regression vs K0 live?  CPU, the corrected (dt 0.5) reader's own loaders, DEV only,
paired episode-cluster bootstrap (taniteval.ci, B 2000 seed 0) per stratum.  Seeds 0 and 1.  Diagnostic, not a verdict.
  1. signed bias: mean signed cross error (P_y - G_y) at the last waypoint and over the horizon -- a side offset vs symmetric noise
  2. strata of LAT_cross_mae_m delta: GT path curvature |dyaw| tercile, v0 tercile, straight vs curving, K2-K0 plan disagreement size
Writes lat_attrib.json beside the re-read."""
import json
import sys
from pathlib import Path

import numpy as np

B = Path("C:/Users/Admin/r9e2_read_K2M1N20/reread_dtfix")
SRC = B / "src"
sys.path[:0] = ["F:/refcv9_e3/famfix_mm/tree/stack/scripts", "F:/refcv9_e3/famfix_mm/tree/stack", "F:/refcv9_e3/famfix_mm/tree/taniteval"]
import r9_e2_families as F  # noqa: E402
from taniteval import ci  # noqa: E402

dev = F.load_dev(SRC / "DEV_HOLD_SPLIT.json")
out = {"tier": "T1 (self-action open loop on logged eval139 frames; os arm)", "estimator": "ci.paired_episode_cluster_bootstrap B 2000 seed 0, delta K2 - K0",
       "windows": "DEV", "seeds": {}}


def boot(a, b, eid, m):
    m = np.asarray(m, bool) & np.isfinite(a) & np.isfinite(b)
    if m.sum() < 30:
        return {"n": int(m.sum()), "note": "too few windows"}
    r = ci.paired_episode_cluster_bootstrap(a[m], b[m], [e for e, k in zip(eid, m) if k], n_boot=2000, seed=0)
    return {"n": int(r["n_windows"]), "n_ep": int(r["n_episodes"]), "K2": round(float(a[m].mean()), 5), "K0": round(float(b[m].mean()), 5),
            "delta": round(float(r["delta"]), 5), "lo": round(float(r["lo"]), 5), "hi": round(float(r["hi"]), 5), "sep": bool(r["separated"])}


for S in ("0", "1"):
    A, _ = F.restrict_dev(F.load_dump(SRC / "E2_K2_pf" / f"dump_s{S}"), dev)
    Bd, _ = F.restrict_dev(F.load_dump(SRC / "E2_K0_pf" / f"dump_s{S}"), dev)
    pa, pb, eid, rep = F.align(A, Bd)
    G = pa["G"]
    ea, eb = pa["P"] - G, pb["P"] - G
    res = {"n_windows": len(eid)}
    # 1. signed bias (ego frame, +y = left)
    res["signed_cross_final_m"] = boot(ea[:, -1, 1], eb[:, -1, 1], eid, np.ones(len(eid)))
    res["signed_cross_mean_m"] = boot(ea[..., 1].mean(1), eb[..., 1].mean(1), eid, np.ones(len(eid)))
    res["abs_cross_mean_m"] = boot(np.abs(ea[..., 1]).mean(1), np.abs(eb[..., 1]).mean(1), eid, np.ones(len(eid)))
    # strata
    head_end = np.arctan2(G[:, -1, 1] - G[:, -2, 1], G[:, -1, 0] - G[:, -2, 0])
    curv = np.abs(head_end)
    v0 = np.linalg.norm(G[:, 0], axis=-1) / A["dt_s"]
    dis = np.linalg.norm(pa["P"][:, -1] - pb["P"][:, -1], axis=-1)
    ca, cb = np.abs(ea[..., 1]).mean(1), np.abs(eb[..., 1]).mean(1)
    strata = {}
    for name, x in (("gt_heading_change", curv), ("v0", v0), ("k2_k0_final_disagreement", dis)):
        q = np.quantile(x, [1 / 3, 2 / 3])
        for lab, m in (("low", x <= q[0]), ("mid", (x > q[0]) & (x <= q[1])), ("high", x > q[1])):
            strata[f"{name}:{lab}"] = {"range": [round(float(x[m].min()), 4), round(float(x[m].max()), 4)], **boot(ca, cb, eid, m)}
    straight = curv < np.deg2rad(2.0)
    strata["straight(<2deg)"] = boot(ca, cb, eid, straight)
    strata["curving(>=2deg)"] = boot(ca, cb, eid, ~straight)
    stopped = v0 < 1.0
    strata["moving(v0>=1)"] = boot(ca, cb, eid, ~stopped)
    res["abs_cross_mean_by_stratum"] = strata
    out["seeds"][S] = res
    print("seed", S, "signed final", res["signed_cross_final_m"], "\n  signed mean", res["signed_cross_mean_m"])
    for k, v in strata.items():
        print("   ", k, v)
(B / "read" / "lat_attrib.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print("wrote", B / "read" / "lat_attrib.json")
