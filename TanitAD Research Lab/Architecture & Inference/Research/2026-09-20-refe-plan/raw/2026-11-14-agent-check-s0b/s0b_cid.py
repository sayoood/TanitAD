"""C-ID (label identity, dev box): the cheap L-OCC raster of s0b_raster vs the FAITHFUL NAVSIM observation of the same token.

Faithful path = refe.navsim_pdm_targets.metric_cache_parts(scenario) (NAVSIM MetricCacheProcessor._interpolate_gt_observation, navsim 3e8291b)
on a nuPlan scenario built from the local DB + maps; its 51 occupancy maps' polygons are rasterised with an INDEPENDENT but equal-convention
rule (a cell is occupied iff its centre intersects the polygon, shapely.intersects_xy, boundary inclusive).  Bar: cell agreement >= 0.99.
ALSO re-scores the token's banked 64 proposals with NAVSIM (score_plans on executed_plans8) and compares nc / ttc with the fit pack.

  python s0b_cid.py --n 64 --out <dir>        (MM; ~2-4 s per token ESTIMATED -> minutes)     smoke: --n 2
Status: PASS / FAIL / NOT RUN (blocker named).  Writes cid_result.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
HERE = Path(__file__).resolve().parent
V1 = "F:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896"
NUP = "F:/Archive/devbox-C/navsim/nuplan-devkit"
MAP_ROOT = "E:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
sys.path[:0] = [str(HERE), V1, str(HERE / "ref" / "refe")] + ([NUP] if os.environ.get("S0B_USE_F_NUPLAN") else [])   # default: the venv nuplan (DriveRL fork); F: devkit needs pytest
os.environ.setdefault("NUPLAN_MAPS_ROOT", MAP_ROOT)
import numpy as np  # noqa: E402

import s0b_raster as RS  # noqa: E402

BAR = 0.99


def poly_raster(obs, ego_xy, ego_yaw):
    """Faithful observation -> bool (51, NX*NY): cell centre intersects any non-red-light polygon (all objects, no radius cap)."""
    import shapely
    from shapely import affinity
    out = np.zeros((51, RS.NX * RS.NY), dtype=bool)
    c, s = np.cos(-ego_yaw), np.sin(-ego_yaw)
    for t in range(51):
        occ = obs[t]
        for tok, g in zip(occ.tokens, occ._geometries):
            if "red_light" in tok:
                continue
            xy = np.asarray(g.exterior.coords)[:, :2] - np.asarray(ego_xy)
            lx = c * xy[:, 0] - s * xy[:, 1]
            ly = s * xy[:, 0] + c * xy[:, 1]
            if lx.max() < RS.X0 or lx.min() >= RS.X0 + RS.NX * RS.RES or ly.max() < RS.Y0 or ly.min() >= RS.Y0 + RS.NY * RS.RES:
                continue
            poly = shapely.Polygon(np.stack([lx, ly], 1))
            r0 = max(int(np.floor((lx.min() - RS.X0) / RS.RES)), 0); r1 = min(int(np.floor((lx.max() - RS.X0) / RS.RES)), RS.NX - 1)
            c0 = max(int(np.floor((ly.min() - RS.Y0) / RS.RES)), 0); c1 = min(int(np.floor((ly.max() - RS.Y0) / RS.RES)), RS.NY - 1)
            rr, cc = np.meshgrid(np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
            hit = shapely.intersects_xy(poly, RS.X0 + (rr + 0.5) * RS.RES, RS.Y0 + (cc + 0.5) * RS.RES)
            out[t, (rr * RS.NY + cc)[hit]] = True
    return out


def dilate1(a):
    """One-cell (8-neighbour) dilation of (51, NX, NY) bool."""
    p = np.pad(a, ((0, 0), (1, 1), (1, 1)))
    out = np.zeros_like(a)
    for dx in (0, 1, 2):
        for dy in (0, 1, 2):
            out |= p[:, dx:dx + a.shape[1], dy:dy + a.shape[2]]
    return out


def compare(cheap, faith):
    a = cheap[:, :RS.NXY].reshape(51, RS.NX, RS.NY)
    b = faith.reshape(51, RS.NX, RS.NY)
    agree = float((a == b).mean())
    uni = (a | b).sum()
    iou = float((a & b).sum() / uni) if uni else 1.0
    tol_b_in_a = float((b & dilate1(a)).sum() / max(b.sum(), 1))        # faithful occupied cells that the cheap raster has within 1 cell
    tol_a_in_b = float((a & dilate1(b)).sum() / max(a.sum(), 1))
    return {"cell_agreement": agree, "iou": iou, "faithful_occ_within_1cell_of_cheap": tol_b_in_a,
            "cheap_occ_within_1cell_of_faithful": tol_a_in_b, "n_occ_cheap": int(a.sum()), "n_occ_faithful": int(b.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = {"status": "NOT RUN", "n_requested": a.n}
    t_all = time.time()
    try:
        import s0b_run as RUN
        import navsim_pdm_targets as PT
        import navtrain_scenarios as NS
        pack = np.load(RUN.PACK, allow_pickle=False)
        pk = {k: pack[k] for k in ("log", "token", "traj_xy", "yaw", "pdm", "pdm_names", "has_pdm", "is_heldout")}
        names = [str(n) for n in pk["pdm_names"]]
        found = [l.strip() for l in open(HERE / "found_logs.txt", encoding="utf-8") if l.strip()]
        rows = RUN.covered_sets(pk, found)
        rng = np.random.default_rng(20261114)
        pick = sorted(rng.choice(len(rows), size=min(a.n, len(rows)), replace=False).tolist())
        per, ldbs = [], {}
        for j in pick:
            r = rows[j]
            log, tok = str(pk["log"][r]), str(pk["token"][r])
            t0 = time.time()
            rec = {"log": log, "token": tok}
            try:
                db = RUN.db_path(log)
                sc = next(iter(NS.build_scenarios_for_log(db, [tok], map_root=MAP_ROOT, history_rows=40, future_rows=100)), None)
                if sc is None:
                    rec["error"] = "scenario_guard_rejected_token"
                    per.append(rec); continue
                parts = PT.metric_cache_parts(sc)
                ego = parts[1].rear_axle
                faith = poly_raster(parts[2], (ego.x, ego.y), ego.heading)
                ldb = ldbs.setdefault(log, RS.LogDB(db))
                st = RS.gather_set_tracks(ldb, ldb.pc_index[tok])
                cheap = RS.build_arms(st, which=("R-true",))["R-true"]
                rec["ego_pose_equal"] = bool(np.allclose([ego.x, ego.y, ego.heading], [st.ego[0], st.ego[1], st.ego[2]], atol=1e-6))
                rec.update(compare(cheap, faith))
                props = np.concatenate([pk["traj_xy"][r], pk["yaw"][r][..., None]], axis=-1).astype(np.float64)
                sco = PT.score_plans(parts, PT.executed_plans8(props))
                pdm = pk["pdm"][r].astype(np.float64)
                rec["truth"] = {k: {"n_mismatch": int((np.asarray(sco[k]) != pdm[:, names.index(k)]).sum()),
                                    "max_abs_diff": float(np.abs(np.asarray(sco[k]) - pdm[:, names.index(k)]).max())}
                                for k in ("nc", "ttc", "dac", "pdms")}
            except Exception as e:                                           # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                rec["tb"] = traceback.format_exc()[-600:]
            rec["s"] = round(time.time() - t0, 2)
            per.append(rec)
            print(f"{tok} {rec.get('cell_agreement')} {rec.get('error', '')} {rec['s']}s", flush=True)
        good = [p for p in per if "cell_agreement" in p]
        res.update({"per_token": per, "n_ok": len(good), "n_error": len(per) - len(good), "wall_s": round(time.time() - t_all, 1)})
        if not good:
            res["status"] = "NOT RUN"
            res["blocker"] = per[0].get("error") if per else "no tokens"
        else:
            ag = np.array([p["cell_agreement"] for p in good])
            res["cell_agreement_mean"] = float(ag.mean()); res["cell_agreement_min"] = float(ag.min())
            for k in ("iou", "faithful_occ_within_1cell_of_cheap", "cheap_occ_within_1cell_of_faithful"):
                res[k + "_mean"] = float(np.mean([p[k] for p in good]))
            res["bar"] = BAR
            res["status"] = "PASS" if (ag.min() >= BAR and len(good) == len(per)) else "FAIL"
            tr = [p["truth"] for p in good if "truth" in p]
            res["truth_reproduction"] = {"n_tokens": len(tr), "nc_mismatches": int(sum(t["nc"]["n_mismatch"] for t in tr)),
                                         "ttc_mismatches": int(sum(t["ttc"]["n_mismatch"] for t in tr)),
                                         "exact": bool(all(t["nc"]["n_mismatch"] == 0 and t["ttc"]["n_mismatch"] == 0 for t in tr))}
    except Exception as e:                                                   # noqa: BLE001
        res["status"] = "NOT RUN"
        res["blocker"] = f"{type(e).__name__}: {str(e)[:400]}"
        res["tb"] = traceback.format_exc()[-900:]
    with open(out / "cid_result.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_token"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
