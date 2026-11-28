#!/usr/bin/env python3
"""TF-N2 navtest roll: the REGISTERED teacher navtest protocol (eval/teacher_navtest_seam.rollouts, unchanged: 10 Hz, CPU, rank 0,
step 0, history_rows=1 / future_rows=80, logs partitioned over shards) with ONE protocol knob exposed, the positive-jerk limit.
Runs inside eval/ through `eval/run_p4.py --script tfn2_nt_roll.py -- <args>` (driverl-eval venv, env_driverl()). CPU only.

  --arm R0        no override: the build is bit-identical to the registered protocol (TEACHER_OVERRIDES stays EMPTY)
  --arm RJ5       --pos-jerk-limit 5.0: positive_jerk_limit_when_nonnegative_acc 1.0 -> 5.0 (the teacher's own `max_jerk_long`)
Flags: --tokens <W3 json {"tokens": [...]}> (default: all 12,146)  --shard i/N  --out-dir <dir>  [--export <json.gz>, --db-dir <dir>: SMOKE only]

Reachability (the class "built, tested, unreachable from its caller"):
  * the override is applied BEFORE S._init builds the planner, and re-applied after every planner.initialize by the build hook
    (planner.initialize overwrites the jerk limit from the release yaml; planner.py:136-147);
  * after EVERY roll the knobs of the BUILT planner are read back (BTR.effective_knobs) and compared with the request; a mismatch makes the row an
    ERROR row (KNOB-NOT-REACHED), never a result, and every ok row stores `knobs` and `arm`;
  * TFN2_MUTATE_DROP_OVERRIDE=1 (TEST ONLY) drops the override but keeps the request: every row must then be an error row.
"""
import argparse
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import teacher_navtest_seam as S  # noqa: E402  (puts ../refe on sys.path)
import build_teacher_rollouts as BTR  # noqa: E402

REGISTERED = {"route_goal_min_speed_mps": 5.0, "route_goal_horizon_s": 12.0, "policy_interval_s": 0.0,
              "positive_jerk_limit_when_nonnegative_acc": 1.0}
ARM_JERK = {"R0": None, "RJ5": 5.0}          # the ONLY two arms of this card; nothing else is reachable from the CLI


def overrides_for(arm: str) -> dict:
    j = ARM_JERK[arm]
    return {} if j is None else {"positive_jerk_limit_when_nonnegative_acc": float(j)}


def expected_knobs(ov: dict) -> dict:
    e = dict(REGISTERED)
    e.update(ov)
    return e


def knob_mismatch(built: dict, expected: dict, tol: float = 1e-9) -> list:
    return [f"{k}: built {built.get(k)} != requested {v}" for k, v in expected.items()
            if built.get(k) is None or abs(built[k] - v) > tol]


class _KnobFile:
    """wraps the shard file: every ok row gets `knobs` (read back after its roll) and `arm`; every other call is delegated"""

    def __init__(self, f, state, arm):
        self._f, self._st, self._arm = f, state, arm

    def write(self, s):
        try:
            r = json.loads(s)
        except json.JSONDecodeError:
            return self._f.write(s)
        if "traj20" in r:
            r["knobs"], r["arm"] = self._st["knobs"], self._arm
            s = json.dumps(r) + "\n"
        return self._f.write(s)

    def __getattr__(self, k):
        return getattr(self._f, k)


def install(arm: str, mutate: bool = False) -> dict:
    """patch the registered roller in place; returns the request. Called BEFORE S._init (which builds the planner)."""
    ov = overrides_for(arm)
    BTR.TEACHER_OVERRIDES.clear()
    if not mutate:
        BTR.TEACHER_OVERRIDES.update(ov)
    exp = expected_knobs(ov)
    state = {"knobs": None}
    orig_roll, orig_open = S.roll, BTR.open_append

    def roll(sc, pos):
        g = orig_roll(sc, pos)
        k = BTR.effective_knobs(S._S["planner"])
        bad = knob_mismatch(k, exp)
        if bad:
            raise RuntimeError("KNOB-NOT-REACHED " + "; ".join(bad))
        state["knobs"] = k
        return g

    S.roll = roll
    BTR.open_append = lambda p: _KnobFile(orig_open(p), state, arm)
    return {"overrides": ov, "expected": exp}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARM_JERK))
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--export", default=None, help="SMOKE only: another token export (json.gz)")
    ap.add_argument("--db-dir", default=None, help="SMOKE only: another nuPlan DB dir")
    a = ap.parse_args()
    mut = os.environ.get("TFN2_MUTATE_DROP_OVERRIDE") == "1"
    req = install(a.arm, mutate=mut)
    if mut:
        print("  MUTATION (test only): override dropped before the build, request kept", flush=True)
    if a.export:
        S.EXPORT = a.export
    print(f"  TF-N2 navtest roll: arm {a.arm}, request {req}, export {S.EXPORT}", flush=True)
    ns = types.SimpleNamespace(tokens=a.tokens, shard=a.shard, out_dir=a.out_dir, device="cpu", threads=a.threads, tts=0,
                               db_dir=a.db_dir or S.TEST_DB)
    return S.rollouts(ns)


if __name__ == "__main__":
    sys.exit(main())
