"""hold_r9_lockwait (MM 2026-11-04): ONE GPU-lock wait for the refcv9 gate3 family -- `R9_LOCK_WAIT_S` in `stack/ops/r9e1_gate3/r9gate3_study.sh`
(default 43200 s = 12 h), read by `r9gate3_thor_gate.sh` (explore + final gate: 1x) and `r9gate3_chain.sh` (training: 2x; the explore deadlines of stages
A and C: + 1x, because the explore itself may be waiting for the lock).

Why: the e1 chain (the E1 arms), the e2 chain (K0) and the record queues share ONE non-FIFO Thor lock, and an arm trains ~5 h. The gate waited 3 h, the
chain's training 6 h and its explore deadlines 90 min / 2 h, so the loser of a boundary race was REFUSED -- a lost arm and another restart -- instead of
waiting its turn.

Proved by EXECUTION where the script runs off-Thor: `r9gate3_thor_gate.sh explore C0` in the test_r9gate3_study sandbox with flock shimmed to RECORD its
arguments -- the default reaches flock as `-w 43200`, an override reaches it unchanged, a non-integer REFUSES before any lock is taken. The chain needs
Thor, so its three waits are asserted on the SOURCE, and each assertion has a mutation arm that restores the old literal and must go RED.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_r9gate3_study as S  # noqa: E402

GATE3 = S.GATE3
CHAIN = GATE3 / "r9gate3_chain.sh"
needs_bash = S.needs_bash


def _explore(tmp: Path, env: dict | None):
    """thor_gate explore C0 (study e1) in the sandbox; flock records `$*` -> (rc, stdout, [flock arg lines] or None, launch_gate reached?)."""
    sb = S._sandbox(tmp)
    (sb / "bin" / "flock").write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FLOCK_OUT"\nexit 0\n', encoding="utf-8", newline="\n")
    S._arm_files(sb, "r9e1", "r9e1", "C0", None)
    fl, rec = sb / "flock_args.txt", sb / "recorded_args.txt"
    e = dict(env or {}, PATH=(sb / "bin").as_posix() + os.pathsep + os.environ.get("PATH", ""), REC_OUT=rec.as_posix(),
             REAL_PY=Path(sys.executable).as_posix(), FLOCK_OUT=fl.as_posix())
    rc, out, _ = S._run([(sb / "kit" / "r9gate3_thor_gate.sh").as_posix(), "explore", "C0"], env=e, cwd=sb.as_posix(), timeout=120)
    return rc, out.replace(sb.as_posix(), ""), (fl.read_text(encoding="utf-8").splitlines() if fl.exists() else None), rec.exists()


# ================================================================================================ the study file ==
@needs_bash
@pytest.mark.parametrize("env,want", [(None, "43200"), ({"R9_LOCK_WAIT_S": "60"}, "60"), ({"R9_STUDY": "e2"}, "43200")])
def test_the_study_file_defines_ONE_lock_wait_default_12_h_and_honours_an_override(env, want):
    rc, out, _ = S._run(["-c", "set -u; . '%s' || exit $?; echo \"W=[$R9_LOCK_WAIT_S]\"" % S.STUDY.as_posix()], env=env)
    assert rc == 0 and f"W=[{want}]" in out, (env, rc, out)


@needs_bash
@pytest.mark.parametrize("val", ["", "3h", "12 h", "-5"])
def test_a_non_integer_lock_wait_REFUSES(val):
    rc, out, _ = S._run(["-c", "set -u; . '%s' || exit $?; echo SOURCED" % S.STUDY.as_posix()], env={"R9_LOCK_WAIT_S": val})
    assert rc == 10 and "ZZR9LOCKWAIT-REFUSEDZZ" in out and "SOURCED" not in out, (val, rc, out)


# ================================================================================================ the thor gate (executed) ==
@needs_bash
def test_thor_gate_waits_the_study_lock_wait_by_default_12_h(tmp_path):
    rc, out, fl, reached = _explore(tmp_path, None)
    assert fl is not None and len(fl) == 1, (fl, out)
    assert fl[0].split()[:2] == ["-w", "43200"], (fl, out)
    assert reached and "ZZR9E1G3-C0-explore-DONEZZ" in out and "(<= 43200 s)" in out, out


@needs_bash
def test_thor_gate_override_reaches_flock_unchanged(tmp_path):
    rc, out, fl, _ = _explore(tmp_path, {"R9_LOCK_WAIT_S": "77"})
    assert fl is not None and fl[0].split()[:2] == ["-w", "77"], (fl, out)


@needs_bash
def test_thor_gate_with_a_bad_lock_wait_refuses_before_any_lock(tmp_path):
    rc, out, fl, reached = _explore(tmp_path, {"R9_LOCK_WAIT_S": "3h"})
    assert rc == 10 and fl is None and not reached, (rc, fl, out)
    assert "ZZR9E1G3-C0-explore-REFUSEDZZ" in out, out


# ================================================================================================ the chain (source) ==
CHAIN_WAITS = {   # name -> (the line's fixed part that must be present, the OLD literal that must be absent)
    "stage A smoke deadline": ("-gt $(( 5400 + R9_LOCK_WAIT_S )) ] && stop EXPLORE-SMOKE-TIMEOUT", "-gt 5400 ] && stop EXPLORE-SMOKE-TIMEOUT"),
    "stage C explore deadline": ("-gt $(( 7200 + R9_LOCK_WAIT_S )) ] && { kill", "-gt 7200 ] && { kill"),
    "training lock wait": ("flock -w $(( 2 * R9_LOCK_WAIT_S )) 9 || stop LOCK-TIMEOUT", "flock -w 21600 9"),
}


def _chain_problems(text: str) -> list:
    bad = []
    for name, (want, old) in CHAIN_WAITS.items():
        if text.count(want) != 1:
            bad.append(f"{name}: '{want}' x{text.count(want)}")
        if old in text:
            bad.append(f"{name}: the old literal '{old}' is back")
    if ". \"$HERE/r9gate3_study.sh\"" not in text:
        bad.append("the chain no longer sources the study file (R9_LOCK_WAIT_S would be unset)")
    return bad


def test_the_chain_takes_all_three_waits_from_the_study_lock_wait():
    assert _chain_problems(CHAIN.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("name", sorted(CHAIN_WAITS))
def test_MUTATION_restoring_an_old_chain_literal_goes_RED(name):
    want, old = CHAIN_WAITS[name]
    text = CHAIN.read_text(encoding="utf-8")
    mutated = text.replace(want, old)
    assert mutated != text
    probs = _chain_problems(mutated)
    assert any(p.startswith(name) for p in probs), (name, probs)


def test_no_gate3_script_keeps_a_literal_lock_wait():
    for f in ("r9gate3_thor_gate.sh", "r9gate3_chain.sh"):
        t = (GATE3 / f).read_text(encoding="utf-8")
        for lit in ("flock -w 10800", "flock -w 21600", "flock -w 3600", "flock -w 7200"):
            assert lit not in t, (f, lit)
        assert t.count("flock -w") == 1 and "R9_LOCK_WAIT_S" in t.split("flock -w", 1)[1].split("\n", 1)[0], f
