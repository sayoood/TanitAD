#!/usr/bin/env python3
"""The `[S-J]` intermittent-failure diagnosis, as a re-runnable measurement.

⛔ A test that passes on retry has not been fixed (C84). So this script does not
re-run and hope: it (A) computes the exact exposure rate the mechanism PREDICTS,
(B) measures the current per-seed outcome, and (C) banks the pre-fix numbers
with their provenance so the before/after is auditable after the fix landed.

THE MECHANISM (see EVIDENCE_AND_FLAKE.md §2.2):
  `_grad_census` used `setdefault`, so a zero-parameter group was ABSENT from
  the census. `STAGE_GROUPS["S-J"] is MODULE_GROUPS`, which includes `interp`,
  empty at the default build. The final assertion is
  `any(census[g]["grad"] for g in want)` over a `set` of `str` — iteration
  order is randomised per process by PYTHONHASHSEED and `any` SHORT-CIRCUITS,
  so `census['interp']` is reached only when `interp` sorts first.

⇒ predicted failure rate == P(interp is first in the set) — computed in (A).

Usage:  python3 g2_flake_diagnosis.py [--seeds 400] [--out raw/flake_diagnosis.json]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[6]
STACK = REPO / "stack"
PY = sys.executable
NODE = "tests/test_v6_ladder_edges.py::" \
       "test_after_init_from_exactly_the_intended_groups_train"

#: The 8 group names of `tanitad.models.v6.MODULE_GROUPS` at the commit that
#: introduced the defect. Read from source below; this is only the fallback.
GROUPS = ("encoder", "readout", "predictor_op", "layer_tac", "layer_str",
          "planner", "aux", "interp")

#: ⛔ MEASURED PRE-FIX and banked here, because the fix is now in the tree and
#: these cannot be re-derived without reverting it. Harness:
#: `flakerun.sh` — n separate processes, whole file, PYTHONUTF8=1
#: OMP_NUM_THREADS=6, venv torch 2.11.0+cu128 / pytest 9.1.1, win32.
PRE_FIX = {
    "evidence_class": "MEASURED (banked; pre-fix tree)",
    "harness": "25 separate pytest processes, whole file, identical env",
    "n_runs": 25, "n_failed": 3, "rate": 3 / 25,
    "failed_runs": [2, 8, 21],
    "signature": "KeyError: 'interp' at test_after_init_from_exactly_the_"
                 "intended_groups_train[S-J], tests/test_v6_ladder_edges.py:459",
    "distinct_signatures": 1,
    "note": "no other failure occurred in 25 x 26 = 650 test executions",
    "deterministic_pre_fix": {"3": "FAIL", "12": "FAIL", "0": "PASS",
                              "5": "PASS"},
    "pre_date_ab": {
        "question": "does the failure pre-date today's changes?",
        "answer": "NO",
        "test_file_identical_ee02ff7_vs_HEAD": True,
        "ee02ff7_result": {"3": "PASS", "12": "PASS", "0": "PASS",
                           "5": "PASS"},
        "interp_entered_MODULE_GROUPS": "06b8782 2026-08-16 20:57:13 +0200",
        "prev_v6_commit_without_interp": "ee02ff7 2026-08-16 16:55",
        "test_file_created": "5725d95 2026-08-16 02:26:12 +0200",
        "gap_hours": 18.5,
    },
}


def module_groups() -> tuple[str, ...]:
    sys.path.insert(0, str(STACK))
    try:
        from tanitad.models.v6 import MODULE_GROUPS
        return tuple(MODULE_GROUPS)
    except Exception:                                    # pragma: no cover
        return GROUPS


def hash_order_sweep(groups, n_seeds: int) -> dict:
    """(A) P(`interp` is iterated first) over `n_seeds` PYTHONHASHSEED values.

    This IS the predicted failure rate: every OTHER group is built and carries
    gradient at S-J, so `any()` short-circuits truthy on any of them, and the
    KeyError is reachable only from position 0.
    """
    # ⚠️ `set(...)`, not the tuple. The test builds
    # `want = set(stage_trainable_groups(stage))`, and it is the SET whose
    # iteration order is hash-randomised — a tuple iterates deterministically
    # and would report a predicted rate of 0.0, which is how this measurement
    # can silently answer the wrong question.
    code = f"s = set({list(groups)!r})\nprint(next(iter(s)))"
    first = collections.Counter()
    for seed in range(n_seeds):
        env = dict(os.environ, PYTHONHASHSEED=str(seed))
        out = subprocess.run([PY, "-c", code], capture_output=True, text=True,
                             env=env).stdout.strip()
        first[out] += 1
    return {"n_seeds": n_seeds,
            "first_element_counts": dict(first.most_common()),
            "predicted_fail_rate": first["interp"] / n_seeds,
            "predicted_fail_seeds": first["interp"]}


def per_seed_now(seeds) -> dict:
    """(B) The CURRENT outcome at each named seed — post-fix, these must all
    pass, including the two that fail deterministically without the fix."""
    got = {}
    for s in seeds:
        env = dict(os.environ, PYTHONHASHSEED=str(s), PYTHONUTF8="1",
                   OMP_NUM_THREADS="6")
        r = subprocess.run([PY, "-m", "pytest", "-q", NODE], cwd=str(STACK),
                           capture_output=True, text=True, env=env)
        # ⛔ parse the RETURN CODE, never grep the text: pytest prints
        # "xfailed", which contains "failed" — the self-matching-filter trap
        got[str(s)] = {"rc": r.returncode,
                       "verdict": "PASS" if r.returncode == 0 else "FAIL",
                       "last_line": (r.stdout.strip().splitlines() or [""])[-1]}
    return got


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=400)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "raw" / "flake_diagnosis.json"))
    a = ap.parse_args(argv)

    groups = module_groups()
    out = {
        "target": NODE + "[S-J]",
        "mechanism": ("zero-parameter group `interp` absent from a `setdefault` "
                      "census + PYTHONHASHSEED-randomised `set` iteration order "
                      "+ short-circuiting `any()`"),
        "refuted_hypothesis": (
            "'seeds the model but draws its batch from global RNG' — REFUTED: "
            "mk() calls torch.manual_seed and nothing between it and the gt_wp "
            "draw touches the global stream, and the failure is a KeyError on "
            "a structural dict lookup that no data value can reach"),
        "module_groups": list(groups),
        "A_predicted": hash_order_sweep(groups, a.seeds),
        "B_current_per_seed": per_seed_now([3, 12, 0, 5, 7]),
        "C_pre_fix_measured": PRE_FIX,
    }
    out["A_predicted"]["vs_observed_pre_fix"] = {
        "predicted": out["A_predicted"]["predicted_fail_rate"],
        "observed": PRE_FIX["rate"], "observed_n": PRE_FIX["n_runs"]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("A_predicted", "B_current_per_seed")}, indent=1))
    print(f"WROTE {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
