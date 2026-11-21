"""THE C97 GUARD HOLE — MEASURE THE BLAST RADIUS, DO NOT MOVE THE CONSTANT.

⛔ WHY THIS DOES NOT EDIT `SD_RATIO_FLAT_FLOOR`. The reported hole is real:
`SD_RATIO_FLAT_FLOOR = 0.05` (`taniteval/taniteval/degeneracy.py:144`) does not
flag C97's own headline case at `sd_ratio = 0.091`. But `degeneracy.py`'s own
docstring calls this "the one tunable in the module", and C95/C97 is the pair
that establishes the rule this programme keeps re-learning:

  ⭐ EVERY CORRECTION TO A CRITERION IS A CANDIDATE BIAS IN THE OPPOSITE
    DIRECTION — this programme built a rejects-everything guard and a
    passes-everything guard WITHIN ONE DAY.

Raising the floor to 0.10 to catch one case is exactly that move made blind. So
this measures, over the **580 banked screen rows**, what each candidate floor
would do IN BOTH DIRECTIONS, and hands the PI the sweep instead of a new number.

⛔ NOTHING HERE CHANGES A VERDICT. The floor is `screen_banked_k1`'s LAYER-3
screen; the module states that layer 2 is what attributes a verdict to the
latent. A floor change alters what gets ESCALATED, never what is concluded.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

FLOORS = (0.02, 0.03, 0.05, 0.07, 0.091, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d = json.loads(Path(a.screen).read_text("utf-8"))
    rows = d["rows"]
    incumbent = float(d.get("SD_RATIO_FLAT_FLOOR_module", 0.05))

    recs = []
    for r in rows:
        g = r["module_guard"]
        recs.append({
            "file": r["file"], "arm": r["arm"], "target": r["target"],
            "seed": r["seed"],
            "sd_ratio": float(g["sd_ratio"]),
            "K1_PASSES": bool(r.get("K1_PASSES")),
            "K1_separated": bool(r.get("K1_separated")),
            "exceeds": bool(g["k1_exceeds_own_spread"]),
        })

    base_flat = {i for i, r in enumerate(recs) if r["sd_ratio"] < incumbent}
    sweep = []
    for f in FLOORS:
        flat = {i for i, r in enumerate(recs) if r["sd_ratio"] < f}
        newly = flat - base_flat            # ⇒ the guard REJECTS MORE
        unflag = base_flat - flat           # ⇒ the guard PASSES MORE
        # the direction that actually matters: rows that CLAIM a K1 PASS and
        # would newly be screened as a flat line
        newly_on_passes = [recs[i] for i in newly if recs[i]["K1_PASSES"]]
        unflag_on_passes = [recs[i] for i in unflag if recs[i]["K1_PASSES"]]
        sweep.append({
            "floor": f,
            "n_flat": len(flat),
            "frac_flat": round(len(flat) / len(recs), 4),
            "REJECTS_MORE_newly_flagged": len(newly),
            "REJECTS_MORE_newly_flagged_that_claim_K1_PASS": len(newly_on_passes),
            "PASSES_MORE_unflagged": len(unflag),
            "PASSES_MORE_unflagged_that_claim_K1_PASS": len(unflag_on_passes),
            "newly_flagged_K1_PASS_rows": [
                {k: v for k, v in r.items() if k != "exceeds"}
                for r in sorted(newly_on_passes,
                                key=lambda r: r["sd_ratio"])[:25]],
        })

    band = sorted(r["sd_ratio"] for r in recs if 0.0 <= r["sd_ratio"] <= 0.35)
    hist = Counter(round(v, 2) for v in band)

    out = {
        "_evidence_class": "MEASURED (ours; sweep over the 580 BANKED screen "
                           "rows in er10_k1_guard_screen.json — no refit)",
        "eval_tier": "T0-DIAGNOSTIC",
        "module": "taniteval/taniteval/degeneracy.py",
        "constant": "SD_RATIO_FLAT_FLOOR",
        "incumbent_value": incumbent,
        "⛔ NOT CHANGED": (
            "this run MEASURES the sweep; it does not edit the constant. "
            "C95/C97: every correction to a criterion is a candidate bias in "
            "the opposite direction, and this programme built a "
            "rejects-everything guard and a passes-everything guard within one "
            "day. The PI decides the floor; this is the evidence."),
        "n_rows": len(recs),
        "c97_headline_case": {
            "reported_sd_ratio": 0.091,
            "flagged_at_incumbent_0.05": False,
            "note": "0.091 > 0.05, so `flat_line` is False and the row is NOT "
                    "screened — the hole, reproduced from the constant itself.",
            "rows_in_this_bank_within_0.085_0.097": [
                r for r in recs if 0.085 <= r["sd_ratio"] <= 0.097],
        },
        "sd_ratio_histogram_0_to_0.35_bin0.01": {
            str(k): v for k, v in sorted(hist.items())},
        "sweep": sweep,
    }
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), "utf-8")
    for s in sweep:
        print(f"floor {s['floor']:.3f}  flat {s['n_flat']:3d}/{len(recs)} "
              f"({s['frac_flat']:.3f})  +flagged {s['REJECTS_MORE_newly_flagged']:3d} "
              f"(of which claim K1 PASS: "
              f"{s['REJECTS_MORE_newly_flagged_that_claim_K1_PASS']:3d})  "
              f"-flagged {s['PASSES_MORE_unflagged']:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
