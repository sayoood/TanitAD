"""DELIBERATE-REGRESSION ARMS: reintroduce each defect, prove the test goes RED.

⛔⛔ WHY THIS IS THE ONLY ADMISSIBLE PROOF. A green test suite says nothing
about whether a guard CAN fire. The programme measured this four times in one
night (CLAUDE.md 2026-09-07): a check whose expected value is an expression
over the code under test is green forever, and an AST census read "0 suspects"
on BOTH the fixed and the broken trainer. ⇒ The evidence that a guard works is
that REMOVING IT TURNS A TEST RED.

Each arm below edits the REAL source on disk, runs the REAL pytest, records the
outcome, and restores the file — with an **md5 assertion on the restore**, so a
crash mid-run cannot leave a mutated trainer behind pretending to be fine.

Usage:  TAC_WT=<worktree> python mutation_proof.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

WT = os.environ["TAC_WT"]
PY = os.environ["TAC_PY"]   # no baked default: an absolute home path
#                            may not live in a repo artifact
STACK = os.path.join(WT, "stack")
TRAINER = os.path.join(STACK, "scripts", "refc_v3_train.py")
READER = os.path.join(STACK, "tanitad", "refs", "refcv6_max_speed.py")

ENV = dict(os.environ)
ENV["PYTHONPATH"] = f"{STACK};{WT};{os.path.join(WT, 'taniteval')}"
ENV["PYTHONIOENCODING"] = "utf-8"
ENV["CUDA_VISIBLE_DEVICES"] = ""
ENV["OMP_NUM_THREADS"] = "6"


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def run_test(node):
    """-> 'PASS' | 'FAIL'. ⛔ The status is read from the RETURN CODE of a
    DIRECT subprocess, never through a pipe: `$?` after a pipeline is the last
    element's status and would report `tail`'s success."""
    # ⛔⛔ `encoding="utf-8", errors="replace"` IS LOAD-BEARING ON THIS BOX.
    # MEASURED here: without it `text=True` decodes the child's output as
    # cp1252, and pytest's ⛔/⭐ characters raise `UnicodeDecodeError` INSIDE
    # subprocess's reader THREAD. The exception is printed but does not
    # propagate, so the run continues and the traceback lands in this script's
    # stdout — i.e. **ahead of the JSON**, producing an artifact that parses as
    # nothing while every number in it is correct. That is the cp1252 family
    # CLAUDE.md records as a TRUNCATED artifact that reads like a complete one.
    r = subprocess.run(
        [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", node],
        cwd=STACK, env=ENV, capture_output=True, text=True, timeout=900,
        encoding="utf-8", errors="replace")
    return "PASS" if r.returncode == 0 else "FAIL"


#: (name, file, old, new, test node, what the defect IS)
ARMS = [
    ("the zero-weight refusal is deleted", TRAINER,
     "        if _w6 <= 0.0:\n            raise SystemExit(",
     "        if False:\n            raise SystemExit(",
     "tests/test_refcv6_tactical_training.py::"
     "test_RED_decoder_with_zero_weight_refuses",
     "a 2,262,020-parameter head builds, trains at zero gradient, and stamps "
     "`tac_decoder_v6: true` -- the tac_goal_tok_head defect at 200x scale"),

    ("the BEV-token refusal is deleted", TRAINER,
     "        if _dbev > 0:\n            raise SystemExit(",
     "        if False:\n            raise SystemExit(",
     "tests/test_refcv6_tactical_training.py::"
     "test_RED_each_dead_configuration_refuses_for_its_own_reason",
     "the decoder declares a `bev` key/value source that never arrives, and "
     "the arm reads as 'the map adds nothing to behaviours'"),

    ("the weight row is removed from the exhaustive audit", TRAINER,
     '    "w_tac_v6": {\n        "flag": "--w-tac-v6",',
     '    "w_tac_v6_DISABLED": {\n        "flag": "--w-tac-v6",',
     "tests/test_refcv6_tactical_training.py::"
     "test_the_new_weight_has_a_gate_row_and_the_gate_discriminates",
     "the head becomes INVISIBLE to the effective-weight audit -- the exact "
     "blindness that hid tac_goal_tok_head for 40,284 steps"),

    ("the sid-before-valid ordering is reverted", READER,
     "    if n_no_sid:\n        raise SpeedMaxStampError(\n"
     '            f"[refcv6-vmax] ⛔ {n_no_sid} of {n_rows} sidecar rows '
     'carry no "',
     "    if False:\n        raise SpeedMaxStampError(\n"
     '            f"[refcv6-vmax] ⛔ {n_no_sid} of {n_rows} sidecar rows '
     'carry no "',
     "tests/test_refcv6_tactical_training.py::"
     "test_a_sid_less_sidecar_names_the_CAUSE_not_the_symptom",
     "a clip_id-keyed sidecar is diagnosed as 'no row carries a valid "
     "ceiling', which is FALSE -- a symptom named as the cause"),

    ("the MANEUVER_WEIGHT budget assertion is neutered",
     os.path.join(STACK, "tanitad", "refs", "refcv6_tactical.py"),
     "        if self.total() > self.budget + 1e-9:",
     "        if False:",
     "tests/test_refcv6_tactical_training.py::"
     "test_the_loss_weights_stay_inside_the_MANEUVER_WEIGHT_budget",
     "the tactical layer silently inflates the total objective and every "
     "loss curve becomes incomparable with refcv5-v2's"),
]

rows = []
for name, path, old, new, node, defect in ARMS:
    src = open(path, encoding="utf-8").read()
    before = md5(path)
    if old not in src:
        rows.append({"arm": name, "verdict": "⛔ ANCHOR NOT FOUND",
                     "why": "the mutation could not be applied, so this arm "
                            "proves NOTHING -- it is not a pass",
                     "node": node})
        continue
    if src.count(old) != 1:
        rows.append({"arm": name, "verdict": "⛔ ANCHOR NOT UNIQUE",
                     "n": src.count(old), "node": node})
        continue
    # GREEN first: the unmutated test must pass, or RED proves nothing.
    green = run_test(node)
    open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
    try:
        red = run_test(node)
    finally:
        open(path, "w", encoding="utf-8").write(src)
        after = md5(path)
    rows.append({
        "arm": name,
        "defect_reintroduced": defect,
        "node": node,
        "unmutated": green,
        "mutated": red,
        "restored_md5_matches": before == after,
        "verdict": ("OK — the guard is REACHABLE"
                    if green == "PASS" and red == "FAIL" and before == after
                    else "⛔ NOT PROVEN"),
    })

out = {
    "n_arms": len(rows),
    "n_ok": sum(1 for r in rows if r["verdict"].startswith("OK")),
    "⛔": "an arm is proven ONLY when the unmutated test PASSES and the "
         "mutated test FAILS. A green suite alone says nothing about whether "
         "a guard can fire.",
    "arms": rows,
}
print(json.dumps(out, indent=1, ensure_ascii=False))
sys.exit(0 if out["n_ok"] == out["n_arms"] else 1)
