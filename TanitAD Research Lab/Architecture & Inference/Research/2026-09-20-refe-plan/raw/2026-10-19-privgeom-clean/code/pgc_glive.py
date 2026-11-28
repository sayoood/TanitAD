"""PRIVGEOM-CLEAN extraction G-LIVE, identity and cost (LABEL-FREE; runs the FROZEN `pg_extract` unchanged).

  live      G-LIVE on 2 of the 76 unseen logs.  The frozen `process_log` runs in memory; ONLY the 9 sign-clean |clearance| columns (+ the store keys) are looked at; the recomputed verdicts `rl6` and every sign-bearing
            or NC column are DROPPED IMMEDIATELY, never printed, never written, never compared with anything.  What is reported: shapes, finiteness, the cap, within-set variation per column, the sha256 of the
            block bytes (the read's full extraction must reproduce it), timings, peak RSS.  NO label is opened, NO DAC verdict is persisted or inspected.
  identity  The frozen extraction on ONE SEEN log reproduces PRIVGEOM-SAFE's banked `scalars/<log>.npz` (g, rl6): bit-for-bit.  (SEEN: its labels were opened by that arm; nothing here is a read of the 76.)

    python pgc_glive.py live --out <dir> [--n 2]
    python pgc_glive.py identity --out <dir> [--log <seen log>]
Interpreter: C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -B -W ignore ; CUDA_VISIBLE_DEVICES=-1 ; OMP_NUM_THREADS=1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np                                                                       # noqa: E402

import pg_extract as X                                                                   # noqa: E402

SAFE = "C:/Users/Admin/par18/privgeom"
SEED = 20261019 + 76
BLOCK_COLS = list(range(2, 11))
SETS_LO, SETS_HI = 40, 160                              # the 2 G-LIVE logs: the first two of a seeded permutation with a median-like size (the unseen median is 84 sets/log)


def sha_bytes(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def pick_logs(cov: dict, n: int) -> list:
    unseen = list(cov["logs_in_order"])
    order = [unseen[i] for i in np.random.default_rng(SEED).permutation(len(unseen))]
    ok = [x for x in order if SETS_LO <= cov["per_log"][x]["n_sets"] <= SETS_HI]
    return ok[:n]


def stage_live(a) -> int:
    cov = json.load(open(a.coverage, encoding="utf-8"))
    names = a.logs.split(",") if a.logs else pick_logs(cov, a.n)
    X.wait_ram(X.MIN_START_GB, "(start gate)")
    elig = X.eligible_logs()
    os.makedirs(a.out, exist_ok=True)
    t_imp = time.time()
    eng = X.Engine()
    t_setup = time.time() - t_imp
    recs = []
    for name in names:
        assert name in cov["per_log"], f"{name} is not one of the 76 unseen logs"
        t0, c0 = time.time(), time.process_time()
        o = X.process_log(eng, name, elig[name], None, None)
        wall, cpu = time.time() - t0, time.process_time() - c0
        res, tm = o["res"], o["timing"]
        ok = res["ok"].copy()
        # ---- from here on ONLY the sign-clean block and the keys are kept; the verdicts and every sign / NC column are dropped without being read
        blk = np.abs(np.asarray(res["g"], np.float64)[..., BLOCK_COLS])
        del res["rl6"], res["g"]
        okb = blk[ok]
        within_sd = okb.std(axis=1)                                                       # [n_ok, 9] std over the 64 proposals of each set
        rec = {"log": name, "n_sets": int(len(ok)), "n_ok": int(ok.sum()), "n_fail": int((~ok).sum()), "n_tokens": int(tm["n_tokens"]), "shape_block": list(blk.shape),
               "all_finite_where_ok": bool(np.isfinite(okb).all()), "min": float(okb.min()), "max": float(okb.max()), "cap_respected_le_10": bool(okb.max() <= 10.0 + 1e-9), "nonneg": bool(okb.min() >= 0.0),
               "frac_sets_with_within_set_variation_per_col": [float((within_sd[:, k] > 1e-6).mean()) for k in range(9)],
               "col_t00_constant_within_every_set": bool((within_sd[:, 0] <= 1e-6).all()),                    # the start clearance is SHARED by the 64 proposals of a set (by construction)
               "cols_t05_to_t40_vary_across_proposals": bool((within_sd[:, 1:].max(0) > 1e-6).all()),
               "block_sha256_ok_sets": sha_bytes(okb), "block_sha256_all_sets": sha_bytes(np.where(ok[:, None, None], blk, 0.0)),
               "cpu_s": cpu, "keys_sha256": sha_bytes(np.stack([res["step"], res["rank"], res["ckpt_step"], res["store_row"]], 1)), "wall_s": wall, "t_build_s": tm["t_build_s"],
               "t_parts_s_per_token": [t["t_parts_s"] for t in tm["tokens"]], "t_prep_s_per_token": [t["t_prep_s"] for t in tm["tokens"]], "t_score_s_per_set": [s["t_score_s"] for s in tm["sets"]],
               "t_geom_s_per_set": [s["t_geom_s"] for s in tm["sets"]], "peak_gb": tm["peak_gb"], "fail": tm["fail"][:5]}
        recs.append(rec)
        print(f"G-LIVE {name}: {rec['n_ok']}/{rec['n_sets']} sets ok, {rec['n_tokens']} tokens, finite {rec['all_finite_where_ok']}, wall {wall:.0f} s, peak {tm['peak_gb']:.2f} GB", flush=True)
        del blk, okb, o, res
    out = {"schema": "privgeom-clean-glive-v1", "disclosure": "the frozen NAVSIM scorer computed the verdicts internally (the geometry is read from its state); they were dropped in memory, never read, printed, stored or compared. No stored label was opened.",
           "n_logs": len(recs), "logs": names, "engine_setup_s": t_setup, "records": recs, "pass": bool(all(r["all_finite_where_ok"] and r["n_ok"] > 0 and r["cap_respected_le_10"] and r["nonneg"] and r["col_t00_constant_within_every_set"] and r["cols_t05_to_t40_vary_across_proposals"] and r["shape_block"][1:] == [64, 9] for r in recs))}
    with open(f"{a.out}/glive.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
    print("G-LIVE pass:", out["pass"])
    return 0 if out["pass"] else 6


def stage_identity(a) -> int:
    name = a.log
    if not name:
        sub = json.load(open(f"{SAFE}/run/subset.json", encoding="utf-8"))["logs_in_order"]
        sizes = {n: len(X.store_keys(n)["token"]) for n in sub}
        name = min(sizes, key=lambda n: (sizes[n] if sizes[n] >= 6 else 10 ** 6, n))     # a small SEEN log (>= 6 sets so that more than one token can be exercised)
    X.wait_ram(X.MIN_START_GB, "(start gate)")
    elig = X.eligible_logs()
    eng = X.Engine()
    t0 = time.time()
    o = X.process_log(eng, name, elig[name], None, None)
    wall = time.time() - t0
    new = o["res"]
    old = dict(np.load(f"{SAFE}/run/scalars/{name}.npz", allow_pickle=False))
    rep = {"schema": "privgeom-clean-identity-v1", "log": name, "seen_by_safe": True, "n_sets": int(len(new["ok"])), "n_ok_new": int(new["ok"].sum()), "n_ok_banked": int(old["ok"].sum()), "wall_s": wall, "keys_equal": {}}
    for k in ("token", "step", "rank", "ckpt_step", "store_row", "ok"):
        rep["keys_equal"][k] = bool(np.array_equal(new[k], old[k]))
    m = new["ok"] & old["ok"]
    rep["g_bit_identical"] = bool(np.array_equal(new["g"][m], old["g"][m]))
    rep["g_max_abs_diff"] = float(np.abs(new["g"][m].astype(np.float64) - old["g"][m].astype(np.float64)).max()) if m.any() else None
    rep["rl6_bit_identical"] = bool(np.array_equal(new["rl6"][m], old["rl6"][m]))
    rep["block_bit_identical"] = bool(np.array_equal(np.abs(new["g"][m][..., BLOCK_COLS]), np.abs(old["g"][m][..., BLOCK_COLS])))
    rep["pass"] = bool(all(rep["keys_equal"].values()) and rep["g_bit_identical"] and rep["rl6_bit_identical"] and rep["n_ok_new"] == rep["n_ok_banked"] and m.any())
    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/identity_extract.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items()}, indent=1))
    return 0 if rep["pass"] else 7


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("live", "identity"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--logs", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--coverage", default="C:/Users/Admin/par18/pgclean/run/coverage.json")
    a = ap.parse_args(argv)
    return {"live": stage_live, "identity": stage_identity}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
