#!/usr/bin/env python3
"""zt5_nav.py -- PREREG_ZT5 section 3 on NAVTEST, DAC level only (diagnostic; no PDMS of any fix is computed here).

Units: ZT3's 144 tracker-made u400 picks (T_A, per_token_A.jsonl tracker_made) and CONTROLS = u400 picks of the C3
random stratum with DAC = 1 at the pick.  Per unit: AG1, NG1 (base sim DAC == the C3 table's DAC at the pick), the
single-factor counterfactuals (CF-H, CF-S, CF-V, CF-A, CF-P; CF-D needs the 20-pose plan and is dev-only) and the
descriptors.

    PYTHONPATH=<navsim v1.1>;<this dir> navsim-crun python zt5_nav.py --c3 <c3 dir> --zt3 <ZT3 pkg> --out ../data/nav \
        [--shard 0/1] [--limit N] [--mutate noop_cf]
"""
from __future__ import annotations

import argparse
import json
import lzma
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import zt5_lib as L  # noqa: E402
import zt5_sim as Z  # noqa: E402

MC_ROOT = Path("D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c3", required=True)
    ap.add_argument("--zt3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    si, sn = map(int, a.shard.split("/"))
    T = np.load(Path(a.c3) / "table.npz")
    meta = json.load(open(Path(a.c3) / "tokens.json", encoding="utf-8"))
    J = np.load(Path(a.zt3) / "data" / "jobs.npz")
    jrow = {str(t): i for i, t in enumerate(J["token"])}
    tm = [json.loads(ln) for ln in open(Path(a.zt3) / "data" / "per_token_A.jsonl", encoding="utf-8")]
    tm_tok = [r["token"] for r in tm if r["tracker_made"]]
    ti = {str(t): i for i, t in enumerate(T["token"])}
    units = []
    for t in tm_tok:
        units.append(("TM", t, J["traj"][jrow[t], 0].astype(np.float64), float(T["sub"][ti[t], T["pick"][ti[t]], 1])))
    for t in T["token"]:
        t = str(t)
        i = ti[t]
        if meta["stratum"][t] == "random" and float(T["sub"][i, T["pick"][i], 1]) == 1.0:
            units.append(("CTRL", t, T["proposals"][i, T["pick"][i]].astype(np.float64), 1.0))
    units = [u for k, u in enumerate(units) if k % sn == si]
    if a.limit:
        units = units[: a.limit]
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    fo = open(od / f"nav_{si}of{sn}.jsonl", "w", encoding="utf-8")
    t_start = time.time()
    for q, (kind, t, p8, exp_dac) in enumerate(units):
        lg = meta["token_log"][t]
        mc = pickle.loads(lzma.decompress(open(MC_ROOT / lg / "unknown" / t / "metric_cache.pkl", "rb").read()))
        ego, dm = mc.ego_state, mc.drivable_area_map
        idcs = Z.drivable_idx(dm)
        if kind == "TM":                                   # ZT3's jobs carry A_P exactly as the table (ZT3 G2)
            assert np.array_equal(p8.astype(np.float32), T["proposals"][ti[t], T["pick"][ti[t]]])
        cf, pc = Z.counterfactuals(p8, ego, dm, idcs)
        R = Z.raw_states(p8[None], ego)
        S = Z.simulate(R, ego)
        rec = {"kind": kind, "token": t, "log": lg, "stratum": meta["stratum"][t], "AG1": Z.admissible_ego_ok(ego),
               "NG1": cf["base"] == exp_dac, "exp_dac": exp_dac, "cf": cf}
        rec.update(Z.descriptors(p8, ego, R, S))
        fo.write(json.dumps(rec) + "\n")
        if q % 50 == 0 or q == len(units) - 1:
            print(f"[{q + 1}/{len(units)}] {kind} {t} elapsed {time.time() - t_start:.0f}s", flush=True)
            fo.flush()
    fo.close()
    print(f"ZZNAV_DONE {si}of{sn} n={len(units)} wall={time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
