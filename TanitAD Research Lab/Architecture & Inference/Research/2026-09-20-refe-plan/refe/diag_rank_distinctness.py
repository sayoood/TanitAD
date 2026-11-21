"""Do the target bank's RANKS carry different data, or the same data wearing different labels?

⛔ THE DEFECT THIS PINS, MEASURED 2026-09-21, AND I COMMITTED IT MYSELF THE SAME NIGHT.
`--rank` in `build_targets.py` and `build_scorer_targets.py` is a LABEL stamped onto each row. The
actual rank-1 data lives in a DIFFERENT rollout directory -- the teacher is re-rolled against a
different goal/route variant, and those rollouts are banked separately (`C:/dzo/m-nr-n` is rank 0,
`C:/dzo/m-nr-r1` is rank 1). Building with `--run <rank-0 dir> --rank 1` therefore produces a
**relabelled copy of rank 0**, and everything downstream reports it as new data:

  * the target bank read "2,182 tuples" when it held **1,091 distinct tuples, duplicated**;
  * scorer coverage "doubled" from 9.7 % to 19.4 % while adding **zero** supervision;
  * ~30 minutes of compute produced a file whose only new bytes were the digit in `"rank"`.

MEASURED before the fix: **0 of 1,091** shared keys differed in trajectory, goal OR image, and the
two scorer files were **byte-identical apart from the rank field** (2,332 of 2,332 trajectories and
2,332 of 2,332 PDM targets identical). MEASURED after rebuilding rank 1 from `m-nr-r1`:
**1,091 of 1,091** trajectories differ, goals differ on 1,064 with a median separation of 3.57 m.

⭐ WHY A SIZE OR md5 CHECK WOULD NOT HAVE CAUGHT IT. The two files had **exactly** the same byte
count (6,302,277) yet **different** md5s -- because the one field that changed was the same width
in both. A size comparison says "identical", an md5 comparison says "different", and both are
useless. The only admissible check is on the CONTENT THAT MATTERS: the trajectories.

## The criterion, and why it is not "everything must differ"

* **Trajectories MUST differ on every shared key.** A different route means a different teacher
  rollout. This is the real test.
* **Images MUST be identical.** Same frames, different route -- if the images differ, the two banks
  are not aligned and the comparison is meaningless.
* **Goals MAY coincide on some frames.** ⚠️ My first version of this check demanded that ALL goals
  differ, and it failed on correct data: 27 of 1,091 frames genuinely share a goal, in the two logs
  where the route variants pass through the same look-ahead point. A threshold tighter than the
  phenomenon is a failing test, not a failing fix. The failure condition is a frame where the goal
  **and** the trajectory are both identical -- that is a true duplicate, and there were **0**.

Usage:
  python diag_rank_distinctness.py [--targets <dir>] [--self-test]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys


def load(path: str) -> dict:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[(d["log_name"], d["token"], d["step"])] = d
    return out


def compare(a: dict, b: dict) -> dict:
    """Pure over two banks, so the self-test can feed it a deliberate duplicate."""
    shared = set(a) & set(b)
    n = len(shared)
    diff_traj = sum(1 for k in shared if a[k]["traj"] != b[k]["traj"])
    diff_goal = sum(1 for k in shared if a[k].get("goal") != b[k].get("goal"))
    diff_img = sum(1 for k in shared if a[k].get("image") != b[k].get("image"))
    true_dupes = sum(1 for k in shared
                     if a[k]["traj"] == b[k]["traj"] and a[k].get("goal") == b[k].get("goal"))
    seps = sorted(math.dist(a[k]["goal"][:2], b[k]["goal"][:2]) for k in shared
                  if a[k].get("goal") != b[k].get("goal") and len(a[k].get("goal", [])) >= 2)
    return {
        "_n": n, "_diff_traj": diff_traj, "_diff_goal": diff_goal, "_diff_img": diff_img,
        "_dupes": true_dupes, "_median_sep": (seps[len(seps) // 2] if seps else float("nan")),
        "1 the two ranks share keys at all": n > 0,
        "2 EVERY shared key has a DIFFERENT trajectory": n > 0 and diff_traj == n,
        "3 images are IDENTICAL (same frames, different route)": n > 0 and diff_img == 0,
        "4 no frame is a TRUE duplicate (same traj AND same goal)": true_dupes == 0,
    }


# ⭐ A ZERO-VELOCITY PLAN IS ROUTE-INDEPENDENT BY CONSTRUCTION, SO IT IS A POSITIVE CONTROL.
# MEASURED 2026-09-21 on the corrected banks: `stopped` is identical on **199 of 199** shared
# frames while every route-DEPENDENT candidate is identical on **0 of 199**. That asymmetry is a
# far stronger statement than "most rows differ": it says the route moved AND that the thing which
# must not move did not. ⚠️ If `stopped` ever DIFFERS across ranks, something is wrong upstream --
# the plan is supposed to be the same standing-still trajectory whatever route was requested.
# (`over-curb` is aimed at the NEAREST curb, which can coincide from both routes; MEASURED 5/199.)
ROUTE_INDEPENDENT = {"stopped"}
MAY_COINCIDE = {"over-curb"}


def compare_scorer(a: dict, b: dict) -> dict:
    """Rank-distinctness for the per-candidate PDM bank, keyed by (log, token, step, candidate)."""
    shared = set(a) & set(b)
    dep = [k for k in shared if k[3] not in ROUTE_INDEPENDENT | MAY_COINCIDE]
    ind = [k for k in shared if k[3] in ROUTE_INDEPENDENT]
    same_dep = sum(1 for k in dep if a[k]["traj"] == b[k]["traj"])
    same_ind = sum(1 for k in ind if a[k]["traj"] == b[k]["traj"])
    diff_tgt = sum(1 for k in shared if a[k].get("targets") != b[k].get("targets"))
    return {
        "_shared": len(shared), "_dep": len(dep), "_ind": len(ind),
        "_same_dep": same_dep, "_same_ind": same_ind, "_diff_tgt": diff_tgt,
        "S1 the banks share candidate keys": len(shared) > 0,
        "S2 EVERY route-dependent candidate differs": len(dep) > 0 and same_dep == 0,
        "S3 CONTROL the route-INDEPENDENT candidate is IDENTICAL":
            len(ind) > 0 and same_ind == len(ind),
        "S4 the PDM targets moved too": diff_tgt > 0.9 * len(shared),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="D:/Projects/TanitAD/data/refe_targets_4cam")
    ap.add_argument("--scorer", nargs=2, metavar=("RANK0_JSONL", "RANK1_JSONL"),
                    help="also check the per-candidate PDM bank")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)

    files = sorted(glob.glob(os.path.join(a.targets, "targets_rank*.jsonl")))
    files = [f for f in files if "stats" not in os.path.basename(f)]
    if len(files) < 2:
        print(f"  need at least two rank files under {a.targets!r}, found {len(files)}")
        return 2
    banks = {os.path.basename(f): load(f) for f in files}
    for nm, bk in banks.items():
        print(f"  {nm:28s} {len(bk):,} rows")

    names = list(banks)
    r = compare(banks[names[0]], banks[names[1]])
    print(f"\n  {names[0]} vs {names[1]}: {r['_n']:,} shared keys")
    print(f"    trajectories differing : {r['_diff_traj']:,} / {r['_n']:,}")
    print(f"    goals differing        : {r['_diff_goal']:,} / {r['_n']:,}   "
          f"(median separation {r['_median_sep']:.2f} m)")
    print(f"    images differing       : {r['_diff_img']:,} / {r['_n']:,}   (must be 0)")
    print(f"    TRUE duplicates        : {r['_dupes']:,}   (same traj AND same goal)")
    print()
    for k, v in r.items():
        if not k.startswith("_"):
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    if a.self_test:
        print("\n== SELF-TEST: the arms must go RED on a relabelled copy ==")
        # the exact historical defect: rank 1 built from rank 0's rollouts, only the label changed
        dupe = {k: dict(v, rank=1) for k, v in banks[names[0]].items()}
        m = compare(banks[names[0]], dupe)
        red2 = not m["2 EVERY shared key has a DIFFERENT trajectory"]
        red4 = not m["4 no frame is a TRUE duplicate (same traj AND same goal)"]
        keep3 = m["3 images are IDENTICAL (same frames, different route)"]
        print(f"  [{'RED ' if red2 else 'FAIL'}] arm 2 on a relabelled copy "
              f"({m['_diff_traj']} of {m['_n']} trajectories differ)")
        print(f"  [{'RED ' if red4 else 'FAIL'}] arm 4 on a relabelled copy "
              f"({m['_dupes']} true duplicates)")
        # ⚠️ AND A CONTROL THAT MUST STAY GREEN: the image arm is about ALIGNMENT, not novelty, so
        # a relabelled copy must NOT trip it. An instrument where every arm fires on every fault
        # cannot localise anything.
        print(f"  [{'GREEN' if keep3 else 'FAIL'}] CONTROL arm 3 stays green on a relabelled copy "
              f"(it tests alignment, not novelty)")
        if not (red2 and red4 and keep3):
            print("\nSELF_TEST_FAILED")
            return 2
        print("\nSELF_TEST_OK")

    good = all(v for k, v in r.items() if not k.startswith("_"))

    if a.scorer:
        def load_c(p):
            out = {}
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        d = json.loads(line)
                        out[(d["log_name"], d["token"], d["step"], d["candidate"])] = d
            return out
        s = compare_scorer(load_c(a.scorer[0]), load_c(a.scorer[1]))
        print(f"\n  SCORER bank: {s['_shared']:,} shared candidate keys")
        print(f"    route-DEPENDENT identical  : {s['_same_dep']:,} / {s['_dep']:,}   (must be 0)")
        print(f"    route-INDEPENDENT identical: {s['_same_ind']:,} / {s['_ind']:,}   "
              f"(must be ALL -- a standing-still plan cannot depend on the route)")
        print(f"    PDM targets differing      : {s['_diff_tgt']:,} / {s['_shared']:,}")
        for k, v in s.items():
            if not k.startswith("_"):
                print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        good = good and all(v for k, v in s.items() if not k.startswith("_"))

    print("\n" + ("RANKS_ARE_DISTINCT" if good else "RANKS_ARE_A_RELABELLED_COPY"))
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
