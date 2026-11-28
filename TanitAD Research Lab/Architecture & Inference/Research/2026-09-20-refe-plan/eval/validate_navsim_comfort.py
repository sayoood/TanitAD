#!/usr/bin/env python3
"""Does refe/navsim_dac.py's comfort verdict agree with NAVSIM's OWN comfort column? (gate for the labeller, option 2)

For every (token, proposal) of an E-6 table (proptable/<name>/table.npz: the proposals on NAVSIM's 8-pose grid EXACTLY
as NAVSIM scored them, and NAVSIM's sub-scores), rebuild the token's scenario from the navtest DB, run ONE simulation
per token through `navsim_dac_and_comfort(grid='navsim8')` and compare BOTH labels with NAVSIM's columns: comfort (the
new label) and drivable area (the label validated at 100.00 % before this refactor -- it must still agree).

  python eval/validate_navsim_comfort.py --name sub200_ep012 --name sub200_ep013
Prints a confusion matrix per table and ZZNAVSIMCOMFORT_AGREE <pct> ZZNAVSIMDAC_AGREE <pct>; exit 0 when both >= --min-agree.
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


def confusion(navsim_ok: np.ndarray, mine_ok: np.ndarray) -> dict:
    a, b = navsim_ok, mine_ok
    return {"agree": float((a == b).mean()), "both_ok": int((a & b).sum()), "both_fail": int((~a & ~b).sum()),
            "navsim_fail_mine_ok": int((~a & b).sum()), "navsim_ok_mine_fail": int((a & ~b).sum()),
            "navsim_fail_rate": float((~a).mean()), "mine_fail_rate": float((~b).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", action="append", required=True)
    ap.add_argument("--min-agree", type=float, default=0.999)
    ap.add_argument("--out", default=os.path.join(DATA, "proptable", "navsim_comfort_validation.json"))
    a = ap.parse_args()
    import navsim_dac as ND
    import refe_navtest_seam as SEAM
    import navtrain_scenarios as NS
    tl = json.load(open(TOKENS, encoding="utf-8"))["token_log"]
    worst_c = worst_d = 1.0
    rep = {}
    for name in a.name:
        T = np.load(os.path.join(DATA, "proptable", name, "table.npz"))
        names = [str(x) for x in T["sub_names"]]
        ic, idac = names.index("comfort"), names.index("drivable_area_compliance")
        toks = [str(t) for t in T["token"]]
        P = T["proposals"]                                                            # [N, 64, 8, 3]
        nav_c, nav_d = T["sub"][:, :, ic] == 1.0, T["sub"][:, :, idac] == 1.0
        mine_c, mine_d = np.full(nav_c.shape, np.nan), np.full(nav_d.shape, np.nan)
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
                viol, comf = ND.navsim_dac_and_comfort(P[i], sc.get_ego_state_at_iteration(0), sc.map_api, grid="navsim8")
                mine_d[i], mine_c[i] = viol, comf
        ok = np.isfinite(mine_c) & np.isfinite(mine_d)
        cc = confusion(nav_c[ok], mine_c[ok] == 1.0)
        cd = confusion(nav_d[ok], mine_d[ok] == 0.0)
        rep[name] = {"proposals": int(ok.sum()), "tokens": int(ok.all(1).sum()), "seconds": round(time.time() - t0, 1),
                     "comfort": cc, "drivable_area": cd}
        print(f"  {name}: {ok.sum():,} proposals over {int(ok.all(1).sum())} tokens in {time.time() - t0:.0f} s")
        print(f"    COMFORT  agreement {100 * cc['agree']:.2f} %  both ok {cc['both_ok']:,} | both fail {cc['both_fail']:,} | "
              f"NAVSIM fail, mine ok {cc['navsim_fail_mine_ok']:,} | NAVSIM ok, mine fail {cc['navsim_ok_mine_fail']:,}  "
              f"(NAVSIM uncomfortable {100 * cc['navsim_fail_rate']:.1f} %)")
        print(f"    DRIVABLE agreement {100 * cd['agree']:.2f} %  (NAVSIM fail {100 * cd['navsim_fail_rate']:.1f} %, "
              f"mine {100 * cd['mine_fail_rate']:.1f} %)", flush=True)
        worst_c, worst_d = min(worst_c, cc["agree"]), min(worst_d, cd["agree"])
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)
    print(f"ZZNAVSIMCOMFORT_AGREE {100 * worst_c:.2f} ZZNAVSIMDAC_AGREE {100 * worst_d:.2f}")
    return 0 if worst_c >= a.min_agree and worst_d >= a.min_agree else 1


if __name__ == "__main__":
    sys.exit(main())
