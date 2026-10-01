#!/usr/bin/env python3
"""Does refe/navsim_dac.py agree with NAVSIM's OWN drivable-area verdict? (gate for the labeller)

For every (token, proposal) of an E-6 table (proptable/<name>/table.npz: the proposals on NAVSIM's
8-pose grid EXACTLY as NAVSIM scored them, and NAVSIM's DAC column), rebuild the token's scenario from
the navtest DB, compute `navsim_dac_violation(grid='navsim8')` and compare. Also asserts that the
module's `to_navsim` equals the seam's on random REFe trajectories.

  python eval/validate_navsim_dac.py --name sub200_ep011 [--name sub200_ep005]
Prints the confusion matrix per table and ZZNAVSIMDAC_AGREE <pct> ... ; exit 0 when >= --min-agree.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "refe"))
DATA = "D:/Projects/TanitAD/data/refe_navtest"
TOKENS = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw/"
          "A1_sub200_tokens.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--min-agree", type=float, default=0.95)
    a = ap.parse_args()
    import navsim_dac as ND
    import refe_navtest_seam as SEAM
    rng = np.random.default_rng(0)
    for _ in range(50):                                    # the copied conversion == the seam's
        p = np.cumsum(rng.normal(0, 1, (20, 3)), 0)
        p[:, 2] = ND.wrap(p[:, 2])
        assert np.allclose(ND.to_navsim(p), SEAM.to_navsim(p), atol=0, rtol=0), "to_navsim drifted"
    print("  to_navsim: identical to the seam's on 50 random trajectories", flush=True)
    import navtrain_scenarios as NS
    tl = json.load(open(TOKENS, encoding="utf-8"))["token_log"]
    worst = 1.0
    for name in a.name:
        T = np.load(os.path.join(DATA, "proptable", name, "table.npz"))
        toks = [str(t) for t in T["token"]]
        P, navsim_pass = T["proposals"], T["sub"][:, :, 1] == 1.0                  # [N,64,8,3], [N,64]
        mine = np.full(navsim_pass.shape, np.nan)
        by_log: dict = {}
        for i, t in enumerate(toks):
            by_log.setdefault(tl[t], []).append(i)
        t0 = time.time()
        for lg, idx in sorted(by_log.items()):
            db = os.path.join(SEAM.TEST_DB_DIR, f"{lg}.db")
            scs = {sc._initial_lidar_token: sc for sc in
                   NS.build_scenarios_for_log(db, [toks[i] for i in idx], history_rows=1, future_rows=80)}
            for i in idx:
                sc = scs.get(toks[i])
                if sc is None:
                    continue
                ego = sc.get_ego_state_at_iteration(0)
                mine[i] = ND.navsim_dac_violation(P[i], ego, sc.map_api, grid="navsim8")
        ok = np.isfinite(mine)
        mine_pass = mine == 0.0
        a_ = navsim_pass[ok]
        b_ = mine_pass[ok]
        agree = float((a_ == b_).mean())
        tp, tn = int((a_ & b_).sum()), int((~a_ & ~b_).sum())
        fp, fn = int((~a_ & b_).sum()), int((a_ & ~b_).sum())
        tok_agree = float(np.mean([(navsim_pass[i] == mine_pass[i]).all() for i in range(len(toks)) if ok[i].all()]))
        print(f"  {name}: {ok.sum():,} proposals over {int(ok.all(1).sum())} tokens in {time.time() - t0:.0f} s")
        print(f"    NAVSIM pass & mine pass {tp:,} | NAVSIM fail & mine fail {tn:,} | NAVSIM fail, mine PASS {fp:,} "
              f"| NAVSIM pass, mine FAIL {fn:,}")
        print(f"    agreement {100 * agree:.2f} %  (NAVSIM fail rate {100 * (~a_).mean():.1f} %, mine "
              f"{100 * (~b_).mean():.1f} %); tokens with all 64 identical {100 * tok_agree:.1f} %", flush=True)
        worst = min(worst, agree)
    print(f"ZZNAVSIMDAC_AGREE {100 * worst:.2f}")
    return 0 if worst >= a.min_agree else 1


if __name__ == "__main__":
    sys.exit(main())
