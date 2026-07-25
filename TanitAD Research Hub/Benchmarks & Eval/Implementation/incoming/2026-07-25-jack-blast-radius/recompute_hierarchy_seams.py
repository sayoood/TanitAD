#!/usr/bin/env python3
"""`_jack` blast-radius sweep — HIERARCHY-SEAM RECOMPUTATION.

The hierarchy panel publishes each seam twice without saying so:

  * ``<metric>.real`` / ``.mean_ctx`` / ``.zero_ctx`` / ``.none`` — these are
    ``hierarchy._mean``, i.e. plain **full-set** means over all 881 windows
    (verified at ``taniteval/taniteval/hierarchy.py:393-397``).
  * ``<metric>.delta_*`` — the ``_jack`` block, a **mean-of-split-means** over 8
    overlapping random 20 % episode holdouts (``overlapping_holdout_se``).

So the TRUE full-set paired delta point estimate is just the difference of the
two published full-set means, and every ``_jack`` delta in every hierarchy
artifact is therefore correctable **in its point estimate** with no re-run, no
GPU and no raw window dump.

What is NOT recoverable from the artifact is the *interval*: the paired
episode-cluster bootstrap needs the per-window arrays, which the hierarchy panel
does not persist. Those are reported as `interval NOT recomputable`.

READ-ONLY. Run:  <venv-python> recompute_hierarchy_seams.py --repo <repo-root>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True

# metric block -> (delta key, minuend field, subtrahend field, lower_is_better)
# orientation follows hierarchy.py: every delta is "helps-positive".
SEAM_SPECS = {
    "seam_nav_to_strategic": [
        ("delta_nav_vs_follow", "route_acc_nav", "route_acc_follow", False),
        ("delta_nav_vs_zeronav", "route_acc_nav", "route_acc_zeronav", False),
        ("delta_follow_vs_zeronav", "route_acc_follow", "route_acc_zeronav", False),
    ],
    "seam_ctx_to_tactical.maneuver_acc": [
        ("delta_real_vs_mean", "real", "mean_ctx", False),
        ("delta_real_vs_zero", "real", "zero_ctx", False),
    ],
    "seam_ctx_to_tactical.wp_ade_2s": [
        ("delta_real_vs_mean", "mean_ctx", "real", True),
        ("delta_real_vs_zero", "zero_ctx", "real", True),
    ],
    "seam_ctx_to_tactical.goal_latent_cos": [
        ("delta_real_vs_mean", "real", "mean_ctx", False),
        ("delta_real_vs_zero", "real", "zero_ctx", False),
    ],
    "seam_intent_to_operative.latent_rel_err": [
        ("delta_rel_real_vs_mean", "mean_intent", "real", True),
        ("delta_rel_real_vs_none", "none", "real", True),
    ],
    "seam_intent_to_operative.latent_cos": [
        ("delta_cos_real_vs_mean", "real", "mean_intent", False),
        ("delta_cos_real_vs_none", "real", "none", False),
    ],
    "h18_grounded_vs_ungrounded": [
        ("delta_ungrounded_minus_grounded", "ungrounded_tactical_head_ade_2s",
         "grounded_op_rollout_ade_2s", False),
    ],
}
# the delta node lives beside the values for the seam-level specs, but one level
# up for the intent seam (deltas sit on the seam, values in the sub-block).
DELTA_AT_SEAM_LEVEL = {"seam_intent_to_operative.latent_rel_err",
                       "seam_intent_to_operative.latent_cos"}
# practical floors the hierarchy panel adjudicates against (hierarchy.py MIN_*)
FLOORS = {"maneuver_acc": 0.02, "goal_latent_cos": 0.01, "wp_ade_2s": 0.05}


def get(doc, dotted):
    cur = doc
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return None
        cur = cur[p]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    out_dir = Path(a.out) if a.out else Path(__file__).resolve().parent

    files = []
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__"}]
        for fn in filenames:
            if fn.startswith("hierarchy") and fn.endswith(".json"):
                files.append(Path(dirpath) / fn)
    files = sorted(set(files))
    print(f"[hier] {len(files)} hierarchy artifacts")

    all_rows = []
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8", errors="replace"))
        rel = str(f.relative_to(repo)).replace("\\", "/")
        for block, specs in SEAM_SPECS.items():
            node = get(doc, block)
            if not isinstance(node, dict):
                continue
            delta_parent = (get(doc, block.split(".")[0])
                            if block in DELTA_AT_SEAM_LEVEL else node)
            for dkey, plus, minus, lower_better in specs:
                pub = (delta_parent or {}).get(dkey)
                if not isinstance(pub, dict) or pub.get("mean") is None:
                    continue
                pv, mv = node.get(plus), node.get(minus)
                if not isinstance(pv, (int, float)) or not isinstance(mv, (int, float)):
                    continue
                true_delta = round(float(pv) - float(mv), 6)
                published = float(pub["mean"])
                ratio = (round(published / true_delta, 4)
                         if abs(true_delta) > 1e-12 else None)
                metric = block.split(".")[-1]
                floor = FLOORS.get(metric)
                row = {
                    "file": rel, "block": block, "delta": dkey,
                    "n": pub.get("n"),
                    "published_jack_split_mean": published,
                    "published_jack_ci95": pub.get("ci95"),
                    "published_separated": pub.get("separated"),
                    "true_full_set_delta": true_delta,
                    "bias_ratio_published_over_true": ratio,
                    "sign_flip": bool(published * true_delta < 0),
                    "components": {plus: pv, minus: mv},
                    "practical_floor": floor,
                    "clears_floor_published": (None if floor is None
                                               else bool(abs(published) >= floor)),
                    "clears_floor_true": (None if floor is None
                                          else bool(abs(true_delta) >= floor)),
                    "floor_verdict_flips": (
                        None if floor is None
                        else bool((abs(published) >= floor)
                                  != (abs(true_delta) >= floor))),
                    "interval_recomputable": False,
                    "interval_note": "paired episode-cluster bootstrap needs the "
                                     "per-window arrays; the hierarchy panel does "
                                     "not persist them — a re-run of "
                                     "taniteval.hierarchy is required",
                }
                all_rows.append(row)

    all_rows.sort(key=lambda r: -abs(r["bias_ratio_published_over_true"] or 0))
    p = out_dir / "jack_hierarchy_recompute.json"
    p.write_text(json.dumps({"rows": all_rows}, indent=1), encoding="utf-8")

    print(f"{'file':<52}{'delta':<32}{'pub':>10}{'true':>10}{'ratio':>9}  flags")
    for r in all_rows:
        flags = []
        if r["sign_flip"]:
            flags.append("SIGN-FLIP")
        if r["floor_verdict_flips"]:
            flags.append("FLOOR-VERDICT-FLIP")
        print(f"{os.path.basename(r['file']):<52}"
              f"{r['block'].split('.')[-1] + '/' + r['delta']:<32}"
              f"{r['published_jack_split_mean']:>10.4f}"
              f"{r['true_full_set_delta']:>10.4f}"
              f"{(r['bias_ratio_published_over_true'] or 0):>9.3f}  "
              f"{' '.join(flags)}")
    print(f"[hier] wrote {p.name}  ({len(all_rows)} deltas)")

    # ---- feed the refinement back into the inventory ---------------------- #
    # The mechanical sweep marks hierarchy artifacts SUSPECT (a `_jack` delta
    # with no `full_set` BLOCK). That is too pessimistic: the panel publishes
    # the full-set component means beside each delta, so the POINT ESTIMATE is
    # correctable — only the interval is not.
    inv = out_dir / "jack_artifact_inventory.jsonl"
    if inv.exists() and all_rows:
        fixed = {r["file"] for r in all_rows}
        lines = []
        for ln in inv.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            rec = json.loads(ln)
            rec["verdict_refined"] = rec.get("verdict")
            if rec.get("file") in fixed:
                rec["verdict_refined"] = "CORRECTED_POINT_ONLY"
                rec["refined_why"] = (
                    "hierarchy panel: `_jack` deltas are correctable from the "
                    "sibling full-set component means published in the same "
                    "artifact; the paired interval still needs a re-run")
            lines.append(json.dumps(rec, ensure_ascii=False))
        inv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        # keep the CSV in sync — it is the copy a human opens
        import csv
        recs = [json.loads(x) for x in lines]
        csvp = out_dir / "jack_artifact_inventory.csv"
        if csvp.exists():
            cols = list(csv.DictReader(csvp.open(encoding="utf-8")).fieldnames)
            if "verdict_refined" not in cols:
                cols.insert(cols.index("verdict") + 1, "verdict_refined")
            with csvp.open("w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
                w.writeheader()
                for r in recs:
                    w.writerow(r)
        print(f"[hier] refined {len(fixed)} inventory verdicts -> "
              f"CORRECTED_POINT_ONLY (jsonl + csv)")


if __name__ == "__main__":
    main()
