"""score_e2abl.py -- MM 4a3d05, 2026-11-12: read the zero-training planner-wire knockouts of K2 (E2_K2abl_{mapzero,boxzero}_pf) against K2 and K0, DEV,
inference seeds 0 and 1, with the CORRECTED four-family reader (r9_e2_families v2, dt from grid.dt_s) + the lateral strata of lat_attrib (straight < 2 deg,
v0 13-33 m/s, signed bias).  Paired episode-cluster bootstrap B 2000 seed 0 (taniteval.ci), delta = A - B.  T1 self-action open loop.  DIAGNOSTIC (one training
run; no pre-registered bar) -- the question it answers: does switching a planner wire off remove K2's lateral excess over K0?
Writes C:/Users/Admin/r9e2_read_K2M1N20/abl/score_e2abl.json."""
import json
import sys
from pathlib import Path

import numpy as np

B = Path("C:/Users/Admin/r9e2_read_K2M1N20")
SRC, ABL = B / "reread_dtfix" / "src", B / "abl"
sys.path[:0] = ["F:/refcv9_e3/famfix_mm/tree/stack/scripts", "F:/refcv9_e3/famfix_mm/tree/stack", "F:/refcv9_e3/famfix_mm/tree/taniteval"]
import r9_e2_families as F  # noqa: E402
from taniteval import ci  # noqa: E402

dev = F.load_dev(SRC / "DEV_HOLD_SPLIT.json")
ARMS = {"K2": SRC / "E2_K2_pf", "K0": SRC / "E2_K0_pf", "K2_mapzero": ABL / "E2_K2abl_mapzero_pf", "K2_boxzero": ABL / "E2_K2abl_boxzero_pf", "K2_drvzero": ABL / "E2_K2abl_drvzero_pf"}
PAIRS = [("K2_drvzero", "K2"), ("K2_drvzero", "K0"), ("K2_mapzero", "K2"), ("K2_mapzero", "K0"), ("K2_boxzero", "K2"), ("K2_boxzero", "K0"), ("K2", "K0")]
LAT = ("LAT_cross_mae_m", "LAT_cross_abs_final_m", "LAT_curvature_mae_1pm_valid", "LAT_yaw_rate_mae_radps_valid", "LAT_heading_mae_deg")
LON = ("LON_accel_mae_mps2", "LON_speed_mae_mps")


def boot(a, b, eid, m):
    m = np.asarray(m, bool) & np.isfinite(a) & np.isfinite(b)
    r = ci.paired_episode_cluster_bootstrap(a[m], b[m], [e for e, k in zip(eid, m) if k], n_boot=2000, seed=0)
    return {"n": int(r["n_windows"]), "A": round(float(a[m].mean()), 5), "B": round(float(b[m].mean()), 5), "delta": round(float(r["delta"]), 5),
            "lo": round(float(r["lo"]), 5), "hi": round(float(r["hi"]), 5), "sep": bool(r["separated"])}


out = {"tier": "T1 (self-action open loop on logged eval139 frames; os arm)", "windows": "DEV", "estimator": "taniteval.ci.paired_episode_cluster_bootstrap B 2000 seed 0",
       "reader": "r9_e2_families v2 (dt from grid.dt_s)", "class": "DIAGNOSTIC", "seeds": {}}
SEEDS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["0", "1"]
OUTN = sys.argv[2] if len(sys.argv) > 2 else "score_e2abl.json"
for S in SEEDS:
    res = {}
    for a, b in PAIRS:
        if not ((ARMS[a] / f"dump_s{S}" / "manifest.json").is_file() and (ARMS[b] / f"dump_s{S}" / "manifest.json").is_file()):
            print(f"s{S} {a} - {b}: SKIP (dump missing)"); continue
        A, _ = F.restrict_dev(F.load_dump(ARMS[a] / f"dump_s{S}"), dev)
        Bd, _ = F.restrict_dev(F.load_dump(ARMS[b] / f"dump_s{S}"), dev)
        pa, pb, eid, rep = F.align(A, Bd)
        ra, rb = F.window_rows(pa["P"], pa["G"], A["dt_s"]), F.window_rows(pb["P"], pb["G"], A["dt_s"])
        one = np.ones(len(eid), bool)
        G = pa["G"]
        straight = np.abs(np.arctan2(G[:, -1, 1] - G[:, -2, 1], G[:, -1, 0] - G[:, -2, 0])) < np.deg2rad(2.0)
        v0 = np.linalg.norm(G[:, 0], axis=-1) / A["dt_s"]
        fast = v0 >= 13.2
        r = {k: boot(ra[k], rb[k], eid, one) for k in LAT + LON + ("ade_m",)}
        r["cross_mae_straight"] = boot(ra["LAT_cross_mae_m"], rb["LAT_cross_mae_m"], eid, straight)
        r["cross_mae_fast"] = boot(ra["LAT_cross_mae_m"], rb["LAT_cross_mae_m"], eid, fast)
        r["signed_cross_final"] = boot(pa["P"][:, -1, 1] - G[:, -1, 1], pb["P"][:, -1, 1] - G[:, -1, 1], eid, one)
        r["_pairing"] = {"n_common": rep["n_common"], "gt_max_abs_diff_m": rep["gt_max_abs_diff_m"], "window_set_sha256": rep["window_set_sha256"]}
        res[f"{a} - {b}"] = r
        print(f"s{S} {a} - {b}: " + " | ".join(f"{k.replace('LAT_', '').replace('_mae', '')} {v['delta']:+.4f}[{v['lo']:+.4f},{v['hi']:+.4f}]{'*' if v['sep'] else ''}"
                                             for k, v in r.items() if k in ("LAT_cross_mae_m", "LAT_curvature_mae_1pm_valid", "LAT_yaw_rate_mae_radps_valid",
                                                                            "cross_mae_straight", "cross_mae_fast", "LON_accel_mae_mps2", "ade_m")))
    out["seeds"][S] = res
ABL.mkdir(parents=True, exist_ok=True)
(ABL / OUTN).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
print("wrote", ABL / OUTN)
