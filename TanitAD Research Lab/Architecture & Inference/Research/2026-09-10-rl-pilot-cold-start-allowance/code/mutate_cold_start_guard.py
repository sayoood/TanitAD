"""MUTATION HARNESS for the cold-start declared allowance (PI queue item 8).

⛔ WHY THIS EXISTS. ``CLAUDE.md``: *"a check that shares the defect it checks for
is green forever"*, and the memory note *"guards need mutation, not
inspection"* — an AST census once read 0 suspects on BOTH the fixed and the
broken trainer. A test suite that is green proves nothing about the guard until
you have watched it go RED against a build with the guard REMOVED.

This script reintroduces the two real historical defects, one at a time, into a
DISPOSABLE copy of the tree, and runs
``tests/test_refc_cold_start_allowance.py`` against each.

    M1  remove the (R) run-condition gate — the `if run_v0: raise` block in
        `plan_cold_start_load`. This is "option (c) implemented without its
        refusal": the allowance would then default `anchor_controls` for a
        v0-CONDITIONED decoder, which is the all-zero action space
        `refc.py:2277` calls "a plausible-looking WRONG experiment".

    M2  replace the whole allowance with `strict=False` — the forbidden option
        (a) from the queue item, i.e. what the next person hitting the
        traceback would reach for.

⭐ EVERY EXPECTATION IS A LITERAL. The expected RED test ids are written out
below, not derived from a run. A mutant that fails a DIFFERENT set of tests than
the ones named is reported as UNEXPECTED, because "some test failed" is not the
same claim as "the arm that guards this defect failed".

Usage (from an off-Drive clone; the mutant tree is created beside it):

    python mutate_cold_start_guard.py --stack C:/path/to/stack --json out.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

TEST = "tests/test_refc_cold_start_allowance.py"
MODULE = "tanitad/refs/cold_start.py"

# ---- LITERAL expectations ------------------------------------------------- #
#: M1 removes the run-condition gate. These arms MUST go RED.
M1_EXPECT_RED = (
    "test_REGRESSION_missing_controls_with_v0_conditioned_is_refused",
    "test_the_flag_is_read_from_the_module_not_from_a_config_argument",
    "test_ONE_conditioned_owner_refuses_the_WHOLE_load",
)
#: M2 replaces the allowance with `strict=False`. These arms MUST go RED.
#: ⭐ Note it is a SUPERSET of M1's: `strict=False` also silently swallows a
#: second missing key, which is the property that distinguishes the declared
#: allowance from the blanket relaxation.
M2_EXPECT_RED = (
    "test_REGRESSION_missing_controls_with_v0_conditioned_is_refused",
    "test_REGRESSION_checkpoint_config_claiming_v0_true_is_refused",
    "test_a_SECOND_missing_key_is_refused_even_though_the_first_is_allowed",
    "test_an_unexpected_key_is_refused",
    "test_strict_false_appears_nowhere_in_the_module",
    "test_the_flag_is_read_from_the_module_not_from_a_config_argument",
    "test_owner_without_the_flag_is_refused_not_assumed",
    "test_two_conflicting_config_copies_are_refused_not_tie_broken",
    "test_require_ckpt_confirmation_refuses_when_the_leaf_is_absent",
    # ⭐ and these three fail because the mutant's stamp is a FABRICATION: it
    # reports `defaulted_keys: []` and `source: checkpoint` for a load that
    # silently defaulted a tensor. That the stamp tests catch a lying stamp is
    # the property that makes the run record evidence rather than decoration.
    "test_july_checkpoint_loads_and_the_model_holds_all_136_keys",
    "test_absent_ckpt_config_is_reported_UNVERIFIED_never_as_agreement",
    "test_the_stamp_lands_in_the_runs_config_json",
    "test_ONE_conditioned_owner_refuses_the_WHOLE_load",
    "test_two_unconditioned_owners_both_default_cleanly",
)

_M1_ANCHOR_START = "    if run_v0:\n"
_M1_ANCHOR_END = "    # ---- (C) THE CHECKPOINT CONDITION"

_M2_BODY = '''

# ---- MUTANT M2: the forbidden option (a) --------------------------------- #
def load_cold_start(model, state_dict, *, ckpt_cfg_leaves=None,
                    require_ckpt_confirmation=False):
    """MUTANT — `strict=False`. This is the fix the queue item forbids."""
    model.load_state_dict(state_dict, strict=False)
    return {"anchor_controls_source": "checkpoint", "defaulted_keys": [],
            "strict": True, "keys_built": len(model.state_dict()),
            "keys_after_load": len(model.state_dict()),
            "ckpt_confirmation": "CONFIRMED", "ckpt_v0_conditioned": False,
            "ckpt_v0_source": "ABSENT", "keys_in_checkpoint": len(state_dict),
            "run_v0_source": "mutant"}
'''


def _mutate_m1(src: str) -> str:
    """Delete the `if run_v0: raise ColdStartRefused(...)` block."""
    i = src.index(_M1_ANCHOR_START)
    j = src.index(_M1_ANCHOR_END)
    out = src[:i] + "    # MUTANT M1: the run-condition gate was REMOVED here.\n\n" + src[j:]
    assert "if run_v0:" not in out, "M1 did not remove the gate"
    return out


def _mutate_m2(src: str) -> str:
    """Append a `strict=False` override of `load_cold_start`."""
    return src + _M2_BODY


MUTANTS = {"M1_remove_run_gate": (_mutate_m1, M1_EXPECT_RED),
           "M2_strict_false": (_mutate_m2, M2_EXPECT_RED)}


def _run(stack: str, py: str) -> tuple[set[str], str]:
    env = dict(os.environ, PYTHONPATH=stack, PYTHONIOENCODING="utf-8",
               PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([py, "-m", "pytest", TEST, "-q", "--no-header", "-p",
                        "no:cacheprovider"],
                       cwd=stack, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    # ⛔ Read the ARTIFACT (the FAILED lines), not the exit code: a collection
    # error also exits non-zero and would otherwise read as "the guard fired".
    red = set(re.findall(r"^FAILED .*::(\w+)", p.stdout, re.M))
    return red, (p.stdout + p.stderr)[-4000:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True, help="an OFF-DRIVE stack/ tree")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    stack = os.path.abspath(a.stack)
    report: dict = {"_what": "mutation proof for tanitad.refs.cold_start",
                    "_evidence_class": "MEASURED (ours; this run)",
                    "baseline": {}, "mutants": {}}

    base_red, base_out = _run(stack, a.python)
    report["baseline"] = {"failed": sorted(base_red),
                          "verdict": "GREEN" if not base_red else "RED"}
    print(f"[mut] BASELINE: {report['baseline']['verdict']} "
          f"({len(base_red)} failed)")
    if base_red:
        print(base_out[-1500:])
        return 2

    src_path = os.path.join(stack, MODULE)
    src = io.open(src_path, encoding="utf-8").read()

    ok = True
    for name, (fn, expect) in MUTANTS.items():
        tmp = tempfile.mkdtemp(prefix=f"mut-{name}-")
        mstack = os.path.join(tmp, "stack")
        shutil.copytree(stack, mstack,
                        ignore=shutil.ignore_patterns("__pycache__",
                                                      ".pytest_cache"))
        io.open(os.path.join(mstack, MODULE), "w", encoding="utf-8",
                newline="").write(fn(src))
        red, out = _run(mstack, a.python)
        missing = sorted(set(expect) - red)
        extra = sorted(red - set(expect))
        entry = {"expected_red": sorted(expect), "actually_red": sorted(red),
                 "expected_but_GREEN": missing, "unexpected_red": extra,
                 "verdict": "PROVEN" if not missing else "NOT PROVEN"}
        if missing:
            entry["tail"] = out[-1200:]
            ok = False
        report["mutants"][name] = entry
        print(f"[mut] {name}: {entry['verdict']} — {len(red)} RED "
              f"(expected {len(expect)}); missing={missing} extra={extra}")
        shutil.rmtree(tmp, ignore_errors=True)

    report["verdict"] = "ALL MUTANTS CAUGHT" if ok else "A MUTANT SURVIVED"
    print(f"[mut] {report['verdict']}")
    if a.json:
        io.open(a.json, "w", encoding="utf-8", newline="").write(
            json.dumps(report, indent=1, sort_keys=True))
        print(f"[mut] wrote {a.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
