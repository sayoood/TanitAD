"""INVENTORY every banked artifact produced through ``pc6_linear_readout.ridge_fit``
and tabulate what the unpenalised-intercept repair does to its verdicts.

⛔ ENUMERATED BY OPENING THE ARTIFACTS, NEVER BY READING WHAT WAS WRITTEN ABOUT
THEM. That is C91 (2026-08-17): a verdict inventory taken from a headline instead
of the artifact reported 2 verdicts where there were 5. Every row below is read
out of a JSON file on disk; nothing is copied from a report.

TWO PRODUCERS SHARE THE DEFECTIVE SOLVE:
  * ``pc6_linear_readout.main``            -> ``pc6_ridge_*.json``   (1 target)
  * ``ll1_ladder.py`` (imports the same    -> ``ll_*.json``          (11 targets)
    ``ridge_fit``, no second implementation)

``ll1_ladder.py`` carries its OWN repair route, recorded per file as
``fit_mode``: ``"pc6"`` is the INCUMBENT (penalised intercept, defective) and
``"centred"`` is the repair (centre y, drop the bias column — algebraically the
same unpenalised-intercept fit as ``intercept_col=-1``). This script reports how
much of the banked corpus sits on each.

ZERO GPU, zero refit — this step is pure JSON arithmetic. The refits themselves
are ``pc6_refit_unbiased.py``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

# ll_rep_<tag>.json is the repaired counterpart of ll_<tag>.json
REPAIR_PAIRS = [("ll_rep_s11250.json", "ll_s11250.json"),
                ("ll_rep_nullmatched.json", "ll_nullmatched.json"),
                ("ll_rep_orcdir.json", "ll_orcdir.json"),
                ("ll_rep_proxyv0.json", "ll_proxyv0.json")]


def verdict(rec) -> str:
    if rec["K1_PASSES"]:
        return "PASS"
    return "FAIL-separated" if rec["K1_separated"] else "not-separated"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc6-raw", required=True)
    ap.add_argument("--ladder-raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    inv, counts = [], {"PASS": 0, "FAIL-separated": 0, "not-separated": 0}

    # ---- producer 1: pc6_ridge_*.json (one target: lead_gap) ----
    for p in sorted(glob.glob(os.path.join(a.pc6_raw, "pc6_ridge_*.json"))):
        d = json.loads(Path(p).read_text("utf-8"))
        v = verdict(d)
        counts[v] += 1
        inv.append({"file": os.path.basename(p), "producer": "pc6_linear_readout",
                    "fit_mode": "pc6 (INCUMBENT — penalised intercept)",
                    "arm": d["arm"], "target": "lead_gap", "verdict": v,
                    "K1_delta": d["K1_delta"], "err_m": d["ridge_err_m"],
                    "alpha": d["alpha_chosen"]})

    # ---- producer 2: ll_*.json (11 targets each) ----
    lad = {}
    for p in sorted(glob.glob(os.path.join(a.ladder_raw, "ll_*.json"))):
        d = json.loads(Path(p).read_text("utf-8"))
        if "targets" not in d:
            continue
        lad[os.path.basename(p)] = d
        mode = d.get("fit_mode")
        for tgt, t in d["targets"].items():
            rec = t["per_seed"]["0"]
            v = verdict(rec)
            counts[v] += 1
            inv.append({"file": os.path.basename(p), "producer": "ll1_ladder",
                        "fit_mode": ("pc6 (INCUMBENT — penalised intercept)"
                                     if mode == "pc6" else f"{mode} (REPAIRED)"),
                        "arm": d["arm"], "target": tgt, "rung": t.get("rung"),
                        "verdict": v, "K1_delta": rec["K1_delta"],
                        "err_m": rec["err"], "alpha": rec["alpha_chosen"]})

    on_incumbent = [r for r in inv if r["fit_mode"].startswith("pc6")]
    on_repaired = [r for r in inv if not r["fit_mode"].startswith("pc6")]
    inc_fail = [r for r in on_incumbent if r["verdict"] == "FAIL-separated"]

    print(f"ARTIFACT INVENTORY (opened, not quoted)")
    print(f"  files read:               "
          f"{len(set(r['file'] for r in inv))}")
    print(f"  verdict rows total:       {len(inv)}")
    print(f"  ON THE INCUMBENT SOLVE:   {len(on_incumbent)}")
    print(f"  on the repaired solve:    {len(on_repaired)}")
    print(f"  verdicts: {counts}")
    print(f"  ⛔ separated-FAIL verdicts still on the biased floor: {len(inc_fail)}")

    # ---- the 4 paired re-reads: what the repair actually did ----
    print(f"\nPAIRED RE-READS ({len(REPAIR_PAIRS)} arms x 11 targets):")
    print(f"{'arm':26} {'target':16} {'old K1':>9} {'new K1':>9} "
          f"{'old verdict':16} {'new verdict':16} change")
    pairs, changed = [], 0
    for rep_f, inc_f in REPAIR_PAIRS:
        if rep_f not in lad or inc_f not in lad:
            print(f"  MISSING PAIR {rep_f} / {inc_f}")
            continue
        R, I = lad[rep_f], lad[inc_f]
        for tgt in I["targets"]:
            if tgt not in R["targets"]:
                continue
            ri = I["targets"][tgt]["per_seed"]["0"]
            rr = R["targets"][tgt]["per_seed"]["0"]
            vo, vn = verdict(ri), verdict(rr)
            ch = "—" if vo == vn else f"{vo} -> {vn}"
            changed += vo != vn
            rec = {"arm": I["arm"], "target": tgt,
                   "rung": I["targets"][tgt].get("rung"),
                   "old_K1": ri["K1_delta"], "new_K1": rr["K1_delta"],
                   "old_K1_ci": [ri["K1_lo"], ri["K1_hi"]],
                   "new_K1_ci": [rr["K1_lo"], rr["K1_hi"]],
                   "old_err_m": ri["err"], "new_err_m": rr["err"],
                   "c_const_err": ri["c_const_err"],
                   "old_verdict": vo, "new_verdict": vn,
                   "verdict_changed": vo != vn,
                   "old_alpha": ri["alpha_chosen"], "new_alpha": rr["alpha_chosen"]}
            pairs.append(rec)
            print(f"{I['arm'][:26]:26} {tgt:16} {ri['K1_delta']:+9.3f} "
                  f"{rr['K1_delta']:+9.3f} {vo:16} {vn:16} {ch}")
    print(f"\nverdict changes in the paired re-reads: {changed} / {len(pairs)}")

    payload = {
        "_evidence_class": "MEASURED (ours; every row read out of a banked JSON "
                           "artifact on disk — never from a report headline)",
        "eval_tier": "T0-DIAGNOSTIC",
        "defect": "C92 penalised intercept (pc6_linear_readout.ridge_fit)",
        "n_files": len(set(r["file"] for r in inv)),
        "n_verdict_rows": len(inv),
        "n_on_incumbent_solve": len(on_incumbent),
        "n_on_repaired_solve": len(on_repaired),
        "verdict_counts": counts,
        "n_separated_FAIL_still_on_biased_floor": len(inc_fail),
        "inventory": inv,
        "paired_rereads": pairs,
        "n_verdict_changes_in_paired_rereads": changed,
    }
    Path(a.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
