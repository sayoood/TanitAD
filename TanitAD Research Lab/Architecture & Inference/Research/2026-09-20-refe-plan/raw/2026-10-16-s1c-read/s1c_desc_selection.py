#!/usr/bin/env python3
"""s1c_desc_selection.py -- PREREG_S1C section 8: arm B's pick, read DESCRIPTIVELY against the three 400-update N arms, stamped BUDGET-CONFOUNDED.

Levels only (pick PDMS x100 and its six sub-scores, fan random / best, zero picks): NO paired difference, NO CI, NO classification.  Why: the S1c arm's scorer pair (A, B) trained 25x the updates of an N arm
at batch 8 (no head-only mode exists), and there is no budget-matched N arm (an `N-u10k` control, ~4.2 A40-h, is a PI decision), so a level difference is not attributable to the head.
The sets are the held-out navtrain sets (24 logs, 3,137 sets) the harness's `selection` read uses, through the programme's own functions: rl_val_eval.eval_set / summarise, aux_s1_readout.eligible_sets,
the S1 read driver's memory-safe loader shim (rl_val_eval.load_scores retains the whole score array per row: 33.5 GB per arm on the real dumps), rule navsim_v1 (S1 read AMB-6).

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I s1c_desc_selection.py --scores H1u10k-s0=<scores_update10000_B.npz> N-s0=... N-s1=... N-s2=... --pdm <heldout_pdm> --out desc.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PK = HERE.parent.parent
for _p in (PK / "eval", PK / "refe"):
    sys.path.insert(0, str(_p))
import aux_s1_readout as RO  # noqa: E402
import rl_reward  # noqa: E402
import rl_val_eval as RV  # noqa: E402
import s1_read as S1R  # noqa: E402  -- load_scores_copy (the memory shim) and nothing else

TOOL = "s1c_desc_selection/1"
RULE = "navsim_v1"
STAMP = ("BUDGET-CONFOUNDED: the S1c arm's scorer pair trained 25x the updates of an N arm (10,000 x 8 rows vs 400 x 64 rows), so a level difference is NOT attributable to the head; "
         "levels only, no paired difference, no interval, no classification (PREREG_S1C section 8)")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def levels(scores_paths: dict, pdm_dir: str, rule: str = RULE) -> dict:
    scores = {a: S1R.load_scores_copy(p, RV) for a, p in scores_paths.items()}
    pdm, pst = RV.load_pdm(pdm_dir)
    ks = RO.eligible_sets(scores, pdm)
    if not ks:
        raise ValueError("no eligible set (no common key with a PDM record of the same checkpoint)")
    cfg = rl_reward.RewardCfg(lam_d=0.0, w_prog=0.5, guard=True, use_k=False, use_l=False)
    per = {a: {k: RV.eval_set(sc[k], {"tg": pdm[k]["tg"], "terms": None, "L": None}, rule, cfg) for k in ks} for a, sc in scores.items()}
    return {"tool": TOOL, "rule": rule, "n_sets": len(ks), "n_logs": len({k[0] for k in ks}), "stamp": STAMP, "arms": {a: RV.summarise(per[a]) for a in sorted(per)},
            "inputs": {"scores_sha256": {a: sha256_file(p) for a, p in scores_paths.items()}, "pdm_dir": Path(pdm_dir).name, "pdm_stats": pst},
            "tier": "T0-like: banked held-out sets, the scorer's pick looked up in frozen PDM labels; a selection read, never driving performance"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scores", nargs="+", required=True)
    ap.add_argument("--pdm", required=True)
    ap.add_argument("--rule", default=RULE, choices=RV.RULES)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    paths = {}
    for spec in a.scores:
        name, _, p = spec.partition("=")
        RO.parse_arm(name)
        paths[name] = p
    try:
        res = levels(paths, a.pdm, a.rule)
    except (ValueError, KeyError) as e:
        print(f"ZZS1CDESC_REFUSED {e}", flush=True)
        return 3
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(res, f, indent=1, sort_keys=True, default=lambda o: o.item() if isinstance(o, (np.floating, np.integer)) else str(o))
    print(f"ZZS1CDESC_OK arms {sorted(res['arms'])} sets {res['n_sets']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
