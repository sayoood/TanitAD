#!/usr/bin/env python3
"""W1 -- run E1's STAGE-1 cross-check on a scored NavSim CSV, standalone.

    python w1_stage1_check.py <scored.csv> [--split navhard_two_stage] [--arm CV] [--json <out>]

WHY THIS EXISTS AS A SCRIPT and not only inside the benchmark: the check was wired into
`navsim/benchmark.py` at ~10:50 on 2026-09-20, AFTER the live navhard run's parent process had
already imported that module. A running process does not re-read its source, so THAT run cannot
emit `stage1_reference_check` however correct the code is -- the same "a snapshot cannot see a
later change" shape that the per-arm blob capture exists for. This script closes the gap for any
CSV already on disk, with no GPU and no re-scoring.

Compares, per E1's relay:
  (a) our stage-1 sub-metrics against E1's MEASURED run -- an IDENTITY check (same split, same
      scorer, same 450 scenes), NOT a tolerance;
  (b) the PUBLISHED navhard leaderboard, under TRUNCATION to 1 dp -- 8/8 truncating vs 4/8
      rounding, read by W1 from the banked PDF (library key 2506.04218, p.8 Table 2, col 'CV [8]').
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "taniteval"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--split", default="navhard_two_stage")
    ap.add_argument("--arm", default="CV")
    ap.add_argument("--json", dest="out", default=None)
    a = ap.parse_args(argv)

    from taniteval.bench.navsim import profiles as P, summarize as S

    refs = json.loads((REPO / "taniteval/taniteval/bench/navsim/references.json").read_text(encoding="utf-8"))
    ref = ((refs.get(a.split) or {}).get(a.arm) or {}).get("stage_one")
    if not ref:
        print(f"REFUSED: no banked stage-1 reference for {a.split}/{a.arm}")
        return 2

    y = P.read_split_yaml(a.split)
    stage_of = {t: 1 for t in y["stage_one"]}
    stage_of.update({t: 2 for t in y["stage_two"]})
    tok = S.load_scores(a.csv, stage_of)                       # refuses a pdm_score column (E2's rule)
    s1 = S.stage1_submetrics(tok)
    out = S.stage1_reference_check(s1["values_x100"], ref, n=s1["n"])

    print(f"stage-1 rows: {s1['n']} (expected {ref.get('n')})")
    if s1["n"] != ref.get("n"):
        print("  \u26a0 n DIFFERS from the banked reference -- the comparison below is NOT like-for-like")
    print(f"{'metric':6s} {'ours x100':>12s} {'E1':>9s} {'=':>3s}   {'trunc':>7s} {'pub':>7s} {'=':>3s}")
    for k in S.STAGE1_SUB:
        e = out["vs_e1_measured"].get(k, {})
        p = out["vs_published"].get(k, {})
        ours = e.get("ours_x100")
        print(f"{k:6s} {('%.4f' % ours) if ours is not None else 'n/a':>12s} {str(e.get('e1_x100')):>9s} "
              f"{'OK' if e.get('equal_2dp') else 'XX':>3s}   {str(p.get('ours_truncated_1dp')):>7s} "
              f"{str(p.get('published')):>7s} {'OK' if p.get('match_truncating') else 'XX':>3s}")
    print(f"\nidentical_to_e1 = {out['identical_to_e1']}   "
          f"published: {out['n_match_truncating']}/{out['n_compared']} truncating, "
          f"{out['n_match_rounding']}/{out['n_compared']} rounding")
    if not out["identical_to_e1"]:
        print("\u26d4 NOT identical to E1 -- chase this FIRST, ahead of everything else.")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"-> {a.out}")
    return 0 if out["identical_to_e1"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
