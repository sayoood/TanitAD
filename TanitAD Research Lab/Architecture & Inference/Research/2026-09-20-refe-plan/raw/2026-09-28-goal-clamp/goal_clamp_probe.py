#!/usr/bin/env python3
"""EXPLORATORY (coordinator go, 2026-09-28 ~01:45): does clamping an OUT-OF-RANGE goal input collapse REFe's fan on the
navtest tokens whose goal lies > 300 m ahead? CPU only; zero training; the untouched snapshot 015 (= the live model's
decoder side) on the fp32 eval cache; the planner's own selection rule (v1) and Amendment 7's repair ON, as live.

⛔ NOT ADMISSIBLE FOR ADOPTION: the tokens were SELECTED BY THE DEFECT (goal p2 > 300 m == the 20 whole-fan tokens of
raw/2026-09-28-m6b-tangent). A confirmation needs FRESH tokens (the drafted amendment says how).

Conditions, the goal's TWO points scaled by ONE common factor s = min(1, C / |p2|), i.e. clamped along p2's own bearing
with p1 kept at its fraction of p2 (route_goal_positions places p1 at 1/2 of the route progress to p2):
  none  -- the control: must reproduce the GPU dump's picks (raw/2026-09-28-m6b-tangent/dumps/base.npz)
  c219  -- C = 219.16 m, the proxy TRAIN cache's goal-p2 p99.9
  c150  -- C = 150 m
Reads: fan distance at 4 s vs GT, the executed plan's heading[18]/[19] vs GT, whole-fan beyond pi, implied initial speed,
and the NAVSIM harness PDMS of the executed plan (repair ON) on these tokens (score_navtest_refe.py, unchanged).

    python raw/2026-09-28-goal-clamp/goal_clamp_probe.py            -> goal_clamp_probe.json (+ seams, score CSVs)
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG / "eval"))
sys.path.insert(0, str(PKG / "refe"))
import eval_checkpoint as EC  # noqa: E402

ECD = Path("D:/Projects/TanitAD/data/refe_proxy/cache_eval_ep015")
WORK = Path("D:/Projects/TanitAD/data/refe_proxy/goal_clamp")            # NO spaces (the harness's Hydra grammar)
GPU_DUMP = PKG / "raw" / "2026-09-28-m6b-tangent" / "dumps" / "base.npz"
THRESH = 300.0
CONDS = {"none": None, "c219": 219.16, "c150": 150.0, "fallback": "straight"}
# "fallback" (added after the trace, goal_trace.json): the route does not cover the ego on these logs (nearest route
# roadblock 373 / 421 m away), so route_goal_positions projects the ego onto the route START. The in-distribution
# replacement is the goal route_goal_positions WOULD give on a straight route from the ego: p2 = max(v0, 5) * 12 s
# ahead along the ego heading, p1 at half of it (its own horizon_s / min_speed_mps defaults, planner.py:184).


def main() -> int:
    if os.environ.get("REFE_GC_CHILD") != "1":                   # the planner needs the driverl env (as proxy_eval run)
        env = EC.env_driverl()
        env["REFE_GC_CHILD"] = "1"
        env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + str(PKG / "refe") + os.pathsep + str(PKG / "eval")
        env["OMP_NUM_THREADS"] = "6"
        return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__), *sys.argv[1:]], cwd=str(PKG / "eval"), env=env)
    import torch
    import planner as PL
    import proxy_eval as PE
    import proxy_train as PT
    SEAM = PE.SEAM
    model, _cfg, _st = PT.build(PE.SNAP, "cpu")
    model.eval()
    stub = type("RuleStub", (), {"rule": "navsim_v1", "V1_W": PL.REFePlanner.V1_W, "PDM_W": PL.REFePlanner.PDM_W})()
    exp = json.load(gzip.open(SEAM.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
    tokj, _w3, _a5 = PE.tokens_1123()
    want = set(tokj["tokens"])
    frames = [json.loads(l) for l in open(ECD / "frames.jsonl", encoding="utf-8")]
    rows = {tuple(json.loads(l)["key"]): json.loads(l) for l in open(ECD / "rows.jsonl", encoding="utf-8")}
    sel = [fr for fr in frames if fr["key"][1] in want
           and np.linalg.norm(np.array(rows[tuple(fr["key"])]["goal"][2:4], float)) > THRESH]
    n_per = {}
    for fr in frames:
        n_per[fr["shard"]] = max(n_per.get(fr["shard"], 0), fr["index"] + 1)
    mm = {s: (np.memmap(ECD / f"shard_{s:04d}.scene.bin", dtype=np.float32, mode="r", shape=(n, 64, 256)),
              np.memmap(ECD / f"shard_{s:04d}.visual.bin", dtype=np.float32, mode="r", shape=(n, 7680, 256)))
          for s, n in n_per.items()}
    gd = np.load(GPU_DUMP)
    gi = {str(t): i for i, t in enumerate(gd["token"])}
    WORK.mkdir(parents=True, exist_ok=True)
    toks = [fr["key"][1] for fr in sel]
    tl = {t: tokj["token_log"][t] for t in toks}
    tokfile = WORK / "tokens_goal_gt300.json"
    json.dump({"tokens": toks, "token_log": tl}, open(tokfile, "w", encoding="utf-8"))
    res = {"what": "EXPLORATORY goal clamp on the goal-p2 > 300 m navtest tokens (selected BY the defect -- not "
                   "admissible for adoption)", "n_tokens": len(toks), "tokens": toks, "conditions": {}}
    for cname, C in CONDS.items():
        T_, P_, L_ = [], [], []
        for fr in sel:
            r = rows[tuple(fr["key"])]
            sc = torch.from_numpy(np.array(mm[fr["shard"]][0][fr["index"]]))[None]
            vi = torch.from_numpy(np.array(mm[fr["shard"]][1][fr["index"]]))[None]
            ego = torch.tensor([r["ego"]], dtype=torch.float32)
            g = np.array(r["goal"], np.float64)
            if C == "straight":
                s_ = max(float(r["ego"][6]), 5.0) * 12.0
                g = np.array([s_ / 2.0, 0.0, s_, 0.0])
            elif C is not None:
                g = g * min(1.0, C / float(np.linalg.norm(g[2:4])))
            goal = torch.tensor([g], dtype=torch.float32)
            with torch.no_grad():
                traj = PT.decode(model, sc, ego, goal)
                logits = PT.score(model, traj, vi)
                k = int(PL.REFePlanner.aggregate(stub, logits)[0].argmax())
            T_.append(traj[0].float().numpy())
            P_.append(k)
            L_.append(logits[0].float().numpy())
        T = np.stack(T_).astype(np.float64)
        P = np.array(P_)
        H = np.stack([np.asarray(rows[tuple(fr["key"])]["human_future_poses"], np.float64) for fr in sel])
        ar = np.arange(len(P))
        d4 = np.linalg.norm(T[:, :, 19, :2], axis=-1)
        gt4 = np.linalg.norm(H[:, 7, :2], axis=-1)
        vinit = np.linalg.norm(T[:, :, 0, :2], axis=-1) / 0.2
        blk = {"clamp_m": C,
               "goal_p2_m_mean_after": float(np.mean([
                   max(float(rows[tuple(fr["key"])]["ego"][6]), 5.0) * 12.0 if C == "straight" else
                   np.linalg.norm(np.array(rows[tuple(fr["key"])]["goal"][2:4], float))
                   * (1.0 if C is None else min(1.0, C / np.linalg.norm(np.array(rows[tuple(fr["key"])]["goal"][2:4], float))))
                   for fr in sel])),
               "gt_dist_4s_m_mean": float(gt4.mean()),
               "fan_dist_4s_m_[min,median,max]_mean": [float(d4.min(1).mean()), float(np.median(d4, 1).mean()),
                                                      float(d4.max(1).mean())],
               "pick_dist_4s_m_mean": float(d4[ar, P].mean()),
               "pick_abs_dist_err_vs_gt_m_mean": float(np.abs(d4[ar, P] - gt4).mean()),
               "pick_implied_initial_speed_mps_mean": float(vinit[ar, P].mean()),
               "ego_v0_mps_mean": float(np.mean([rows[tuple(fr["key"])]["ego"][6] for fr in sel])),
               "pick_h18_vs_gt4s_median_rad": float(np.median(np.abs(PE.wrap(T[ar, P, 18, 2] - H[:, 7, 2])))),
               "pick_h19_vs_gt4s_median_rad": float(np.median(np.abs(PE.wrap(T[ar, P, 19, 2] - H[:, 7, 2])))),
               "whole_fan_beyond_pi_tokens": int((np.abs(T[..., 19, 2]) > np.pi).all(1).sum()),
               "pct_raw_h19_beyond_pi": 100.0 * float(np.mean(np.abs(T[..., 19, 2]) > np.pi)),
               "tokens_with_ANY_slot_shorter_than_gt": int((d4 < gt4[:, None]).any(1).sum())}
        if cname == "none":
            gidx = [gi[t] for t in toks]
            blk["control_vs_gpu_dump"] = {"picks_equal": f"{int((P == gd['pick'][gidx]).sum())}/{len(P)}",
                                          "max_abs_traj_diff": float(np.abs(T - gd["traj"][gidx]).max())}
        lb = f"refe_gc_{cname}_on"
        poses = [SEAM.to_navsim(PL.repair_last_heading(T[i, P[i]])).astype(np.float32) for i in range(len(P))]
        sp = WORK / f"{lb}.npz"
        np.savez(sp, token=np.array(toks), fingerprint=np.array([exp[t]["fingerprint"] for t in toks]),
                 poses=np.stack(poses), sampling=np.array([SEAM.NAVSIM_N, SEAM.NAVSIM_DT]), arm=np.array(lb))
        np.savez_compressed(HERE / f"dump_{cname}.npz", token=np.array(toks), traj=T.astype(np.float32),
                            logits=np.stack(L_), pick=P, human=H.astype(np.float32))
        lf = HERE / f"score_{lb}.log"
        EC.run([EC.DRIVERL_PY, "score_navtest_refe.py", "--label", lb, "--seam", str(sp), "--tokens", str(tokfile),
                "--out", str(WORK / "score")], str(PKG / "eval"), dict(os.environ, PYTHONIOENCODING="utf-8"), str(lf))
        csvp = WORK / "score" / lb / f"{lb}.csv"
        st = PE.status_of(lf.read_text(encoding="utf-8", errors="replace")) if lf.exists() else None
        blk["harness"] = {"status": (st or {}).get("status"), "rows": (st or {}).get("csv_valid_rows"),
                          "summary_x100": (st or {}).get("summary_x100_4dp")}
        if csvp.exists():
            shutil.copy2(csvp, HERE / f"{lb}.csv")
            rr = PE.read_csv(csvp)
            blk["pdms_x100_mean"] = 100.0 * float(np.mean([float(x["score"]) for x in rr.values()]))
        res["conditions"][cname] = blk
        print(cname, json.dumps(blk, default=float)[:900], flush=True)
    json.dump(res, open(HERE / "goal_clamp_probe.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    print("ZZGOALCLAMP_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
