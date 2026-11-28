"""⭐ INTEGRATION — apply the SIBLING AGENT'S formal C97 guard to every K1 row.

⛔ WHY THIS FILE EXISTS RATHER THAN A SECOND GUARD. My brief said a formal
degeneracy guard was being built concurrently and that I should USE IT IF IT
LANDS. It landed: `taniteval/taniteval/degeneracy.py` (`k1_guard`,
`screen_banked_k1`, `SD_RATIO_FLAT_FLOOR`, and an EXACT `K1 = K1B + K1C`
decomposition). ⇒ **this file IMPORTS it and re-screens every banked row rather
than re-asserting my own inline `pred_sd/gt_sd` rule.**

⚠️ It also RECONCILES the two floors, which differ and must not be conflated:
my inline stamp used **0.10**, the landed module's `SD_RATIO_FLAT_FLOOR` is
**0.05**. The module's is the programme's; both are reported per row so no
reader has to guess which produced a verdict.

⭐ `screen_banked_k1` is LAYER 1 + 3 from banked numbers alone — no refit, no
predictions — so this costs zero compute and can screen every row of every arm.
Its own docstring is explicit that PASSING the screen is **not** evidence a
verdict is latent-attributable; that needs layer 2, which needs the predictions.
Stated here so the output is not over-read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(_REPO / "taniteval"))

from taniteval.degeneracy import (GUARD_VERDICTS,  # noqa: E402
                                  SD_RATIO_FLAT_FLOOR, screen_banked_k1)

RAW = Path(__file__).resolve().parents[1] / "raw"


def main() -> int:
    out = {"_evidence_class": "MEASURED (ours; the SIBLING module "
                              "`taniteval.degeneracy.screen_banked_k1` applied "
                              "to our banked rows — not a second guard)",
           "eval_tier": "T0-DIAGNOSTIC",
           "module": "taniteval.degeneracy",
           "SD_RATIO_FLAT_FLOOR_module": SD_RATIO_FLAT_FLOOR,
           "sd_ratio_floor_used_inline_by_er10": 0.10,
           "GUARD_VERDICTS": list(GUARD_VERDICTS),
           "note": "screen_banked_k1 is LAYER 1+3 only. Passing it is NOT "
                   "evidence a verdict is latent-attributable (its own "
                   "docstring); that needs layer 2 and the predictions.",
           "rows": [], "summary": {}}
    counts: dict[str, int] = {}
    for p in sorted(RAW.glob("er10_*.json")):
        d = json.loads(p.read_text("utf-8"))
        if "arms" not in d:
            continue
        for arm, a in d["arms"].items():
            for t, rec in a.get("targets", {}).items():
                # ⚠️ TWO PRODUCER SHAPES, and the consumer must not assume one:
                # `er10_pool_ladder.py` nests per projection seed; the
                # no-projection supplement `er10_full_ridge.py` has a single
                # fit and writes the fields at the target level. C94's class in
                # miniature — handled, not assumed.
                per = rec.get("per_seed") or {"-": rec}
                for seed, s in per.items():
                    g = screen_banked_k1(s["K1_delta"], s["pred_sd"],
                                         s["gt_sd"], s.get("K1_separated"))
                    v = g.get("screen_verdict", "?")
                    counts[v] = counts.get(v, 0) + 1
                    if s.get("K1_PASSES") or v.startswith("SUSPECT"):
                        out["rows"].append(
                            {"file": p.name, "arm": arm, "target": t,
                             "seed": seed,
                             "K1_delta": s["K1_delta"],
                             "K1_separated": s.get("K1_separated"),
                             "K1_PASSES": s.get("K1_PASSES"),
                             "pred_sd": s["pred_sd"], "gt_sd": s["gt_sd"],
                             "er10_inline_K1_DEGENERATE":
                                 s.get("K1_DEGENERATE"),
                             "module_guard": g})
    out["summary"] = {"verdict_counts": counts,
                      "n_rows_listed": len(out["rows"])}
    dst = RAW / "er10_k1_guard_screen.json"
    dst.write_text(json.dumps(out, indent=1), "utf-8")
    print(json.dumps(counts, indent=1))
    print(f"[guard] {len(out['rows'])} rows listed -> {dst}")
    # the rows that MATTER: a PASS the module does not call OK
    bad = [r for r in out["rows"]
           if r["K1_PASSES"] and r["module_guard"].get("screen_verdict","").startswith("SUSPECT")]
    print(f"[guard] ⛔ K1 PASSES PROVEN to contain a constant-offset component (layer 1+3): {len(bad)}")
    for r in bad[:40]:
        print(f"    {r['file']:28s} {r['arm']:6s} {r['target']:14s} s{r['seed']} "
              f"K1={r['K1_delta']:+.4f} psd/gsd="
              f"{r['pred_sd']/max(r['gt_sd'],1e-12):.4f} -> "
              f"{r['module_guard'].get('screen_verdict')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
