"""⭐ MEASURE the claim that ``centred`` and ``unpen`` are the same repair.

``ll1_ladder._solve`` offers two routes to an unpenalised intercept:

* ``centred`` — re-derived locally: centre ``y``, drop the bias column, add
  ``mean(y)`` back at prediction time. This is what the 4 banked ``ll_rep_*``
  files used.
* ``unpen``   — the repair taken from the MODULE: ``ridge_fit(Z, y, alpha,
  intercept_col=-1)``. This is what this package's re-read used.

They are algebraically identical **on the full fit**: the feature block is
z-scored with the PROBE-TRAIN mean, so ``Xc' 1 = 0``, the normal equations go
block-diagonal, and the unpenalised-intercept solution is exactly
``b = mean(y)`` with the centred slope solve.

⛔ BUT THAT ARGUMENT DOES NOT COVER THE INNER SPLIT, where alpha is chosen on a
SUBSET of the train rows whose feature mean is NOT exactly zero. There the two
routes differ, and if that difference ever flips an alpha choice the two are not
interchangeable. **"Algebraically identical" is a HYPOTHESIS until the alphas and
the deltas are compared on real data** — this script is the measurement, and it
exists because "should be equivalent" is how a scope error enters a report.

⛔ T0-DIAGNOSTIC.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PAIRS = [("ll_rep_s11250.json", "llR_s11250.json"),
         ("ll_rep_nullmatched.json", "llR_nullmatched.json"),
         ("ll_rep_orcdir.json", "llR_orcdir.json"),
         ("ll_rep_proxyv0.json", "llR_proxyv0.json")]

FIELDS = ["err", "K1_delta", "K1_lo", "K1_hi", "c_const_err", "corr", "pred_sd"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--banked", required=True)
    ap.add_argument("--reread", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    rows, worst = [], {f: 0.0 for f in FIELDS}
    alpha_diff = verdict_diff = 0
    for bf, rf in PAIRS:
        bp, rp = os.path.join(a.banked, bf), os.path.join(a.reread, rf)
        if not (os.path.exists(bp) and os.path.exists(rp)):
            print(f"  MISSING PAIR {bf} / {rf}")
            continue
        B = json.loads(Path(bp).read_text("utf-8"))
        R = json.loads(Path(rp).read_text("utf-8"))
        assert B["fit_mode"] == "centred" and R["fit_mode"] == "unpen"
        for tgt in B["targets"]:
            if tgt not in R["targets"]:
                continue
            b, r = (B["targets"][tgt]["per_seed"]["0"],
                    R["targets"][tgt]["per_seed"]["0"])
            d = {f: abs(float(b[f]) - float(r[f])) for f in FIELDS}
            for f in FIELDS:
                worst[f] = max(worst[f], d[f])
            same_a = b["alpha_chosen"] == r["alpha_chosen"]
            same_v = (b["K1_PASSES"] == r["K1_PASSES"]
                      and b["K1_separated"] == r["K1_separated"])
            alpha_diff += not same_a
            verdict_diff += not same_v
            rows.append({"file": bf, "target": tgt,
                         "centred_alpha": b["alpha_chosen"],
                         "unpen_alpha": r["alpha_chosen"],
                         "same_alpha": same_a, "same_verdict": same_v,
                         "abs_diff": {f: round(d[f], 10) for f in FIELDS},
                         "centred_K1": b["K1_delta"], "unpen_K1": r["K1_delta"]})

    print(f"centred vs unpen — {len(rows)} paired rows "
          f"({len(PAIRS)} arms x 11 targets)")
    print(f"  alpha choices that DIFFER : {alpha_diff}")
    print(f"  verdicts that DIFFER      : {verdict_diff}")
    print("  worst absolute difference per field:")
    for f in FIELDS:
        print(f"    {f:14} {worst[f]:.10g}")
    for r in rows:
        if not (r["same_alpha"] and r["same_verdict"]):
            print(f"  ⚠️ {r['file']:24} {r['target']:15} "
                  f"alpha {r['centred_alpha']:g} -> {r['unpen_alpha']:g}  "
                  f"K1 {r['centred_K1']:+.4f} -> {r['unpen_K1']:+.4f}")

    payload = {
        "_evidence_class": "MEASURED (ours; both sides opened from JSON on disk)",
        "eval_tier": "T0-DIAGNOSTIC",
        "question": "are ll1's two repair routes interchangeable in practice?",
        "n_paired_rows": len(rows),
        "n_alpha_differences": alpha_diff,
        "n_verdict_differences": verdict_diff,
        "worst_abs_diff": {f: round(worst[f], 12) for f in FIELDS},
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
