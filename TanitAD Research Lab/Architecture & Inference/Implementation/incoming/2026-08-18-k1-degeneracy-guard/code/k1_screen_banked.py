"""LAYER-1 SCREEN over EVERY banked ridge verdict row — zero compute, zero GPU.

⛔ ENUMERATED BY OPENING THE ARTIFACTS, NEVER FROM A HEADLINE (C91). Every row is
read out of a JSON on disk.

WHY THIS RUNS FIRST. ``taniteval.degeneracy.screen_banked_k1`` needs only
``K1_delta``, ``pred_sd`` and ``gt_sd`` — all three are already recorded in every
``pc6_ridge_*.json`` and every ``ll_*.json``. Because

    |K1B| <= mean|pred - mean(pred)| <= pred_sd

(reverse triangle inequality, then Jensen), a row whose ``|K1_delta|`` EXCEEDS its
own ``pred_sd`` has a PROVEN constant-offset component of at least
``|K1_delta| - pred_sd``. That is a theorem, not a threshold, and it screens all
214 banked rows before a single refit is paid for.

⚠️ A row that PASSES the screen is not thereby attributable to the latent — that
needs layer 2 (``k1_guard``'s K1B, which requires the predictions). The screen
can only ever say SUSPECT, never CLEAN.

⛔ T0-DIAGNOSTIC.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

def _repo_root() -> Path:
    """Walk UP to the checkout rather than counting `parents[n]`.

    ⚠️ A hard-coded index is exactly the kind of thing that breaks silently when
    a package is moved one directory; the marker file is the fact that settles it.
    """
    for p in Path(__file__).resolve().parents:
        if (p / "taniteval" / "taniteval" / "ci.py").exists():
            return p
    raise SystemExit("[screen] ⛔ could not locate the repo root from "
                     f"{__file__} — no ancestor holds taniteval/taniteval/ci.py")


sys.path.insert(0, str(_repo_root() / "taniteval"))
from taniteval.degeneracy import screen_banked_k1                # noqa: E402


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

    rows = []
    for p in sorted(glob.glob(os.path.join(a.pc6_raw, "pc6_ridge_*.json"))):
        d = json.loads(Path(p).read_text("utf-8"))
        rows.append({"file": os.path.basename(p), "producer": "pc6",
                     "fit_mode": "pc6", "arm": d["arm"], "target": "lead_gap",
                     "rung": "OBJECT", "verdict": verdict(d),
                     "K1_delta": d["K1_delta"], "pred_sd": d["pred_sd_m"],
                     "gt_sd": d["gt_sd_m"], "K1_separated": d["K1_separated"]})
    for p in sorted(glob.glob(os.path.join(a.ladder_raw, "ll_*.json"))):
        d = json.loads(Path(p).read_text("utf-8"))
        if "targets" not in d:
            continue
        for tgt, t in d["targets"].items():
            r = t["per_seed"]["0"]
            rows.append({"file": os.path.basename(p), "producer": "ll1",
                         "fit_mode": d.get("fit_mode"), "arm": d["arm"],
                         "target": tgt, "rung": t.get("rung"),
                         "verdict": verdict(r), "K1_delta": r["K1_delta"],
                         "pred_sd": r["pred_sd"], "gt_sd": r["gt_sd"],
                         "K1_separated": r["K1_separated"]})

    for r in rows:
        s = screen_banked_k1(r["K1_delta"], r["pred_sd"], r["gt_sd"],
                             k1_separated=r["K1_separated"])
        r.update({k: s[k] for k in ("sd_ratio", "flat_line",
                                    "k1_exceeds_own_spread",
                                    "min_constant_component")})
        r["screened_SUSPECT"] = bool(r["k1_exceeds_own_spread"]
                                     or r["flat_line"])

    sep = [r for r in rows if r["verdict"] != "not-separated"]
    susp = [r for r in sep if r["screened_SUSPECT"]]
    inc = [r for r in sep if r["fit_mode"] == "pc6"]
    rep = [r for r in sep if r["fit_mode"] != "pc6"]

    print("LAYER-1 SCREEN over banked ridge verdict rows (zero compute)")
    print(f"  rows read                              : {len(rows)}")
    print(f"  rows carrying a VERDICT (separated)    : {len(sep)}")
    print(f"  ...SCREENED SUSPECT                    : {len(susp)}")
    print(f"     of which on the INCUMBENT solve     : "
          f"{sum(1 for r in susp if r['fit_mode'] == 'pc6')} / {len(inc)}")
    print(f"     of which on the REPAIRED solve      : "
          f"{sum(1 for r in susp if r['fit_mode'] != 'pc6')} / {len(rep)}")
    print(f"  PASS rows screened suspect             : "
          f"{sum(1 for r in susp if r['verdict'] == 'PASS')}"
          f" / {sum(1 for r in sep if r['verdict'] == 'PASS')}")
    print(f"  FAIL-sep rows screened suspect         : "
          f"{sum(1 for r in susp if r['verdict'] == 'FAIL-separated')}"
          f" / {sum(1 for r in sep if r['verdict'] == 'FAIL-separated')}")
    print()
    print(f"{'file':26} {'target':15} {'fit':8} {'verdict':15} {'K1':>9} "
          f"{'pred_sd':>9} {'sd_rat':>7} {'minConst':>9} why")
    for r in sorted(susp, key=lambda x: (x["file"], x["target"])):
        why = ",".join(([" |K1|>pred_sd"] if r["k1_exceeds_own_spread"] else [])
                       + ([" flat"] if r["flat_line"] else []))
        print(f"{r['file'][:26]:26} {r['target'][:15]:15} {r['fit_mode'][:8]:8} "
              f"{r['verdict']:15} {r['K1_delta']:+9.3f} {r['pred_sd']:9.3f} "
              f"{r['sd_ratio']:7.4f} {r['min_constant_component']:9.3f}{why}")

    payload = {
        "_evidence_class": "DERIVED (algebra on banked K1_delta/pred_sd/gt_sd; "
                           "every row opened from a JSON on disk, C91)",
        "eval_tier": "T0-DIAGNOSTIC",
        "layer": "1 (exact bound) + 3 (sd_ratio screen)",
        "bound": "|K1B| <= pred_mad <= pred_sd",
        "caveat": "PASSING the screen is NOT evidence of latent attribution; "
                  "that requires layer 2 (k1_guard, needs predictions).",
        "n_rows": len(rows),
        "n_with_verdict": len(sep),
        "n_screened_suspect": len(susp),
        "n_suspect_incumbent": sum(1 for r in susp if r["fit_mode"] == "pc6"),
        "n_suspect_repaired": sum(1 for r in susp if r["fit_mode"] != "pc6"),
        "rows": rows,
    }
    Path(a.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
