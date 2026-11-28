#!/usr/bin/env python3
"""P4 follow-on F2 (eval/PREREG_P4_TEACHER_NAVTEST.md §9, EXPLORATORY): tracker-aware plan compensation.

D1 found that 87 % of the teacher's NAVSIM drivable-area failures are NOT in its own rollout: the 8-pose plan is
compliant, NAVSIM's LQR tracker + kinematic bicycle (refe/navsim_pdm, vendored verbatim; the simulator the harness
runs) deviates from it, and the TRACKED footprint leaves the drivable area. This tool corrects the 8 REFERENCE poses so
that the tracker reproduces the intended plan p:
    r_0 = p;  r_{k+1} = r_k + (p - track(r_k))     at the 8 plan times, ego frame, heading wrapped; 3 iterations
and keeps the iterate with the smallest max |track - p| (xy). Nothing about the intended path changes; only what is
handed to the tracker.

  python eval/run_p4.py --script p4_tracker_comp.py --log <log> -- --seam <in.npz> --out-seam <out.npz>
         [--tokens <W3 json>] [--shard i/N] [--check-dac]
Writes <out-seam> (same format, arm suffixed `_trackcomp`) + <out-seam>.report.json with per-token tracking deviation
before/after (and, with --check-dac, navsim_dac's simulated DAC before/after -- the instrument, 100 % agreement with the
harness in D1; reported, never a substitute for the harness).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PKG, "refe"))
sys.path.insert(0, HERE)

EXPORT = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
TEST_DB = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
ITERS = 3


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def tracked8(ND, plans8: np.ndarray, ego) -> np.ndarray:
    """NAVSIM's simulation of each plan, sampled at the 8 plan times, as rear-axle poses in the ego frame [P, 8, 3]."""
    sim = ND.simulated_states(plans8, ego)                              # [P, 41, S], world frame, rear axle
    s = sim[:, 5::5][:, :8, :3]
    ra = ego.rear_axle
    c, sn = math.cos(-ra.heading), math.sin(-ra.heading)
    dx, dy = s[..., 0] - ra.x, s[..., 1] - ra.y
    return np.stack([dx * c - dy * sn, dx * sn + dy * c, wrap(s[..., 2] - ra.heading)], -1)


def compensate(ND, p: np.ndarray, ego):
    r = p.copy()
    best_r, best_dev, devs = p.copy(), None, []
    for k in range(ITERS + 1):
        s = tracked8(ND, r[None], ego)[0]
        dev = float(np.linalg.norm(s[:, :2] - p[:, :2], axis=1).max())
        devs.append(dev)
        if best_dev is None or dev < best_dev:
            best_r, best_dev = r.copy(), dev
        if k == ITERS:
            break
        e = np.concatenate([p[:, :2] - s[:, :2], wrap(p[:, 2] - s[:, 2])[:, None]], -1)
        r = r + e
        r[:, 2] = wrap(r[:, 2])
    return best_r, devs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", required=True)
    ap.add_argument("--out-seam", required=True)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--check-dac", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    import navsim_dac as ND
    import navtrain_scenarios as NS
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    z = np.load(a.seam)
    toks = [str(t) for t in z["token"]]
    pos = {t: i for i, t in enumerate(toks)}
    if a.tokens:
        want = set(json.load(open(a.tokens, encoding="utf-8"))["tokens"])
        toks = [t for t in toks if t in want]
    by: dict = {}
    for t in toks:
        by.setdefault(exp[t]["log_name"], []).append(t)
    i, n = (int(x) for x in a.shard.split("/"))
    logs = sorted(by)[i::n]
    out_p, out_t, out_fp, rep_rows = [], [], [], []
    t0 = time.time()
    for li, lg in enumerate(logs):
        for sc in NS.build_scenarios_for_log(os.path.join(TEST_DB, f"{lg}.db"), by[lg], history_rows=1, future_rows=80):
            tok = sc._initial_lidar_token
            p = np.asarray(z["poses"][pos[tok]], np.float64)
            rec = {"token": tok}
            try:
                ego = sc.get_ego_state_at_iteration(0)
                r, devs = compensate(ND, p, ego)
                rec.update({"dev_before_m": devs[0], "dev_after_m": min(devs), "devs": devs,
                            "ref_shift_max_m": float(np.linalg.norm(r[:, :2] - p[:, :2], axis=1).max())})
                if a.check_dac:
                    polys = ND.drivable_polygons(sc.map_api, ego)
                    v = ND.violations(polys, ND.simulated_corners(np.stack([p, r]), ego))
                    rec.update({"dac_viol_before": bool(v[0]), "dac_viol_after": bool(v[1])})
            except Exception as e:                                               # noqa: BLE001
                r = p
                rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            out_t.append(tok)
            out_fp.append(str(z["fingerprint"][pos[tok]]))
            out_p.append(r.astype(np.float32))
            rep_rows.append(rec)
        if (li + 1) % 5 == 0 or li + 1 == len(logs):
            print(f"  [{li + 1}/{len(logs)}] {len(out_t)} tokens {time.time() - t0:.0f}s", flush=True)
    suffix = "" if n == 1 else f".s{i}of{n}"
    seam_out = a.out_seam if n == 1 else a.out_seam.replace(".npz", f"{suffix}.npz")
    np.savez(seam_out, token=np.array(out_t), fingerprint=np.array(out_fp), poses=np.stack(out_p),
             sampling=z["sampling"], arm=np.array(str(z["arm"]) + "_trackcomp"))
    ok = [r for r in rep_rows if "error" not in r]
    db = np.array([r["dev_before_m"] for r in ok]) if ok else np.zeros(1)
    da = np.array([r["dev_after_m"] for r in ok]) if ok else np.zeros(1)
    rep = {"seam_in": a.seam, "seam_out": seam_out, "iters": ITERS, "n": len(rep_rows), "errors": len(rep_rows) - len(ok),
           "dev_before_m": {"median": float(np.median(db)), "p95": float(np.percentile(db, 95)), "max": float(db.max())},
           "dev_after_m": {"median": float(np.median(da)), "p95": float(np.percentile(da, 95)), "max": float(da.max())},
           "seconds": round(time.time() - t0, 1), "rows": rep_rows}
    if a.check_dac and ok:
        rep["dac_viol_before"] = sum(r["dac_viol_before"] for r in ok)
        rep["dac_viol_after"] = sum(r["dac_viol_after"] for r in ok)
        rep["repaired"] = sum(r["dac_viol_before"] and not r["dac_viol_after"] for r in ok)
        rep["broken"] = sum((not r["dac_viol_before"]) and r["dac_viol_after"] for r in ok)
    json.dump(rep, open(os.path.splitext(seam_out)[0] + ".report.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "rows"}, indent=1))
    print(f"ZZTRACKCOMP_DONE {len(out_t)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
