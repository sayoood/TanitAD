#!/usr/bin/env python3
"""Amendment 8, REPORTED NOT GATING: the sanitised goal on the SELECTION tokens the trigger fires on (53 of the 1,123;
EXPLORATORY -- the defect was found on them). CPU only: the fp32 eval cache (snapshot 015's planner path, proven
bit-exact against the GPU forward by M6's G2) decoded with the ON goal the patched planner computed for each token
(`a8_goal_checks_sibling.json` rows), the v1 rule, Amendment 7's repair; scored by the unchanged harness; compared
per token with M6's untouched-snapshot seam `refe_m6_base_on` (the same model, the same repair, the OFF goal).
A control decode with the OFF goal must reproduce the banked GPU dump's picks.
    python a8_selection53.py            -> a8_selection53.json (+ seam, CSV, log)
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parents[1]
sys.path.insert(0, str(PKG / "eval"))
sys.path.insert(0, str(PKG / "refe"))
import eval_checkpoint as EC  # noqa: E402

ECD = Path("D:/Projects/TanitAD/data/refe_proxy/cache_eval_ep015")
WORK = Path("D:/Projects/TanitAD/data/refe_proxy/goal_clamp/sel53")
BASE_CSV = PKG / "raw" / "2026-09-27-m6-proxy" / "score" / "refe_m6_base_on" / "refe_m6_base_on.csv"
GPU_DUMP = PKG / "raw" / "2026-09-28-m6b-tangent" / "dumps" / "base.npz"


def main() -> int:
    if os.environ.get("REFE_S53_CHILD") != "1":
        env = EC.env_driverl()
        env["REFE_S53_CHILD"] = "1"
        env["PYTHONPATH"] = env.get("PYTHONPATH", "") + os.pathsep + str(PKG / "refe") + os.pathsep + str(PKG / "eval")
        env["OMP_NUM_THREADS"] = "4"
        return subprocess.call([EC.DRIVERL_PY, os.path.abspath(__file__)], cwd=str(PKG / "eval"), env=env)
    import torch
    import planner as PL
    import proxy_eval as PE
    import proxy_train as PT
    SEAM = PE.SEAM
    chk = json.load(open(HERE / "a8_goal_checks_sibling.json", encoding="utf-8"))["rows"]
    sel = sorted(t for t, r in chk.items() if not r["fresh"] and (r["diag"] or {}).get("triggered"))
    model, _cfg, _st = PT.build(PE.SNAP, "cpu")
    model.eval()
    stub = type("RuleStub", (), {"rule": "navsim_v1", "V1_W": PL.REFePlanner.V1_W, "PDM_W": PL.REFePlanner.PDM_W})()
    exp = json.load(gzip.open(SEAM.W3_EXPORT, "rt", encoding="utf-8"))["tokens"]
    tokj, _w3, _a5 = PE.tokens_1123()
    frames = [json.loads(l) for l in open(ECD / "frames.jsonl", encoding="utf-8")]
    rows = {tuple(json.loads(l)["key"]): json.loads(l) for l in open(ECD / "rows.jsonl", encoding="utf-8")}
    n_per = {}
    for fr in frames:
        n_per[fr["shard"]] = max(n_per.get(fr["shard"], 0), fr["index"] + 1)
    mm = {s: (np.memmap(ECD / f"shard_{s:04d}.scene.bin", dtype=np.float32, mode="r", shape=(n, 64, 256)),
              np.memmap(ECD / f"shard_{s:04d}.visual.bin", dtype=np.float32, mode="r", shape=(n, 7680, 256)))
          for s, n in n_per.items()}
    frs = {fr["key"][1]: fr for fr in frames if fr["key"][1] in set(sel)}
    gd = np.load(GPU_DUMP)
    gi = {str(t): i for i, t in enumerate(gd["token"])}
    WORK.mkdir(parents=True, exist_ok=True)
    tokfile = WORK / "tokens_sel53.json"
    json.dump({"tokens": sel, "token_log": {t: tokj["token_log"][t] for t in sel}}, open(tokfile, "w", encoding="utf-8"))
    res = {"what": "EXPLORATORY (selection tokens): the sanitised goal on the 1,123's triggered tokens, CPU decode of "
                   "the fp32 eval cache; OFF = M6's refe_m6_base_on (same snapshot, same repair)", "n": len(sel)}
    out = {}
    for tag in ("off_control", "on"):
        P_, poses = [], []
        for t in sel:
            fr = frs[t]
            r = rows[tuple(fr["key"])]
            sc = torch.from_numpy(np.array(mm[fr["shard"]][0][fr["index"]]))[None]
            vi = torch.from_numpy(np.array(mm[fr["shard"]][1][fr["index"]]))[None]
            ego = torch.tensor([r["ego"]], dtype=torch.float32)
            g = np.asarray(chk[t]["goal_off" if tag == "off_control" else "goal_on"], dtype=np.float32)
            with torch.no_grad():
                traj = PT.decode(model, sc, ego, torch.from_numpy(g)[None])
                logits = PT.score(model, traj, vi)
                k = int(PL.REFePlanner.aggregate(stub, logits)[0].argmax())
            P_.append(k)
            poses.append(SEAM.to_navsim(PL.repair_last_heading(traj[0, k].float().numpy().astype(np.float64))).astype(np.float32))
        out[tag] = {"pick": P_}
        if tag == "off_control":
            gidx = [gi[t] for t in sel]
            res["control_picks_equal_gpu_dump"] = f"{int((np.array(P_) == gd['pick'][gidx]).sum())}/{len(sel)}"
            res["control_goal_equals_eval_cache_goal"] = all(
                np.allclose(np.asarray(chk[t]["goal_off"]), np.asarray(rows[tuple(frs[t]["key"])]["goal"]), atol=1e-4)
                for t in sel)
            continue
        lb = "refe_a8_sel53_on"
        sp = WORK / f"{lb}.npz"
        np.savez(sp, token=np.array(sel), fingerprint=np.array([exp[t]["fingerprint"] for t in sel]),
                 poses=np.stack(poses), sampling=np.array([SEAM.NAVSIM_N, SEAM.NAVSIM_DT]), arm=np.array(lb))
        while PE.free_ram_gb() < 6.0:
            time.sleep(30)
        lf = HERE / "a8" / "score" / f"{lb}.log"
        lf.parent.mkdir(parents=True, exist_ok=True)
        EC.run([EC.DRIVERL_PY, "score_navtest_refe.py", "--label", lb, "--seam", str(sp), "--tokens", str(tokfile),
                "--out", str(WORK / "score")], str(PKG / "eval"), dict(os.environ, PYTHONIOENCODING="utf-8"), str(lf))
        st = PE.status_of(lf.read_text(encoding="utf-8", errors="replace")) if lf.exists() else None
        res["harness"] = {"status": (st or {}).get("status"), "rows": (st or {}).get("csv_valid_rows")}
        csv = WORK / "score" / lb / f"{lb}.csv"
        if csv.exists():
            shutil.copy2(csv, HERE / "a8" / "score" / f"{lb}.csv")
            on = PE.read_csv(csv)
            base = PE.read_csv(BASE_CSV)
            D = {t: 100.0 * (float(on[t]["score"]) - float(base[t]["score"])) for t in sel if t in on and t in base}
            mu, lo, hi = PE.boot(D, tokj["token_log"])
            res.update({"pdms_off_base_on": 100.0 * float(np.mean([float(base[t]["score"]) for t in D])),
                        "pdms_on": 100.0 * float(np.mean([float(on[t]["score"]) for t in D])),
                        "D_on_minus_off": mu, "ci95": [lo, hi], "n_scored": len(D),
                        "n_logs": len({tokj["token_log"][t] for t in D}),
                        "estimator": "paired log-cluster bootstrap, 10,000 resamples, 95 %, seed 20260927"})
    json.dump(res, open(HERE / "a8_selection53.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
    print(json.dumps(res, default=float))
    print("ZZA8_SEL53_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
