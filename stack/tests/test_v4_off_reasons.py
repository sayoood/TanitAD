"""⛔ A PREFLIGHT MAY NOT NAME A REMEDY THAT DOES NOT EXIST.

THE DEFECT (found 2026-08-03)
-----------------------------
``train_flagship_v4.preflight_asserts`` refused a ``--v2-train-cache`` run without
``--require-parity`` and told the operator to *"record why this arm is deliberately
non-parity"* — **through a flag the parser did not have**. The sentence read like an
escape hatch and was a dead end: the only way past the guard was to satisfy it, so a
legitimately non-parity v2 arm (toy episodes, the 9 000-clip corpus ``4b7eeeac222d``,
an OOD probe) was UNRUNNABLE through this trainer.

⚠️ SAME CLASS, DIFFERENT COSTUME as ``PREFLIGHT: OK`` covering an input it never
looked at (``tests/test_preflight_paths.py``): in both, the guard's OUTPUT and the
guard's BEHAVIOUR disagree, and the message is the thing that misleads.

WHAT IS PINNED HERE
-------------------
1. RED — the guard really refuses, and really names the flag (a guard that cannot
   fail is cover).
2. The flag is a **REASON, NOT A BOOLEAN**. Empty string does not unlock it. This is
   the design constraint carried over verbatim from ``--heldout-off-reason``: a
   boolean records that someone wanted past the guard, a reason records WHY, and only
   the second survives into ``config.json`` as evidence.
3. The reason **survives ``_staged_command`` reconstruction** — for BOTH flags. The
   staged string is what a human copies onto the pod; a dropped reason means the
   copied command trips its own preflight and the run never starts.
   ⚠️ ``--heldout-off-reason`` was missing from that reconstruction from the day it
   was added; this file is what stops it recurring for either flag.
4. The reason is **ECHOED**, not merely stored. Both flags advertised "echoed at
   launch"; neither was — ``heldout_off_reason`` reached exactly three code sites
   (parser, ``NOT_A_PATH``, the preflight that requires it), so its only surface was
   the ``args`` blob in ``config.json``, read after the fact by whoever goes looking.
5. The two parity flags are **mutually exclusive** — a command asserting both
   "enforce parity" and "this arm is deliberately non-parity" would write a false
   provenance record into ``config.json``.

The exhaustiveness contract in ``test_preflight_paths.py`` covers the classification
of the new argument automatically (``NOT_A_PATH``); it is asserted again here so a
failure points at this feature rather than at a generic list.
"""
from __future__ import annotations

import shlex
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_flagship_v4 as T  # noqa: E402

REASON = "OOD probe on comma2k19 — never cross-compared with the parity arms"


def _v2_argv(tmp: Path, *extra: str) -> list[str]:
    """A ``--print-launch`` command on the V2 path whose every OTHER gate passes, so
    anything reported here comes from the parity/held-out reason layer alone."""
    for d in ("v2train", "v2val"):
        (tmp / d).mkdir(exist_ok=True)
    (tmp / "anchors.pt").write_bytes(b"\0")
    argv = ["--print-launch", "--from-scratch",
            "--v2-train-cache", str(tmp / "v2train"),
            "--v2-val-cache", str(tmp / "v2val"),
            "--v2-subframe", "none",
            "--anchors-dense", str(tmp / "anchors.pt"),
            "--out", str(tmp / "run"),
            "--steps", "30000", "--gate-step", "10000",
            "--batch", "8", "--accum", "8",
            "--phase-a-steps", "2000", "--phase-b-steps", "8000",
            "--heldout-every", "2000", "--heldout-episodes", "8",
            "--heldout-patience", "2"]
    return argv + list(extra)


def _problems(argv) -> list[str]:
    return T.preflight_asserts(T.build_parser().parse_args(argv))


def _parity_problems(argv) -> list[str]:
    return [p for p in _problems(argv) if p.startswith("[PARITY]")]


# --------------------------------------------------------------------------- #
# RED — the guard really refuses, and names a flag that really exists.         #
# --------------------------------------------------------------------------- #
def test_v2_cache_without_require_parity_is_BLOCKED(tmp_path):
    p = _parity_problems(_v2_argv(tmp_path))
    assert p, "the [PARITY] guard did not fire at all"
    assert "--parity-off-reason" in p[0], p


def test_the_flag_the_refusal_names_EXISTS_on_the_parser():
    """⭐ THE BUG, as one assertion. The refusal text advertised a remedy; the
    parser had no such option, so following the instruction produced
    ``error: unrecognized arguments``."""
    dests = {a.dest for a in T.build_parser()._actions}
    opts = {o for a in T.build_parser()._actions for o in a.option_strings}
    assert "parity_off_reason" in dests
    assert "--parity-off-reason" in opts


def test_recording_a_reason_clears_the_parity_refusal(tmp_path):
    """…and it is not blanket-passing either: one flag changes, one verdict."""
    assert _parity_problems(_v2_argv(tmp_path, "--parity-off-reason", REASON)) == []


def test_require_parity_still_clears_it_the_other_way(tmp_path):
    """The pre-existing remedy must be untouched by the new one."""
    p = [x for x in _parity_problems(_v2_argv(tmp_path, "--require-parity"))
         if "without --require-parity" in x]
    assert p == [], p


# --------------------------------------------------------------------------- #
# It is a REASON, not a --force boolean.                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("empty", ["", "   "])
def test_an_EMPTY_reason_does_not_unlock_the_guard(tmp_path, empty):
    """⛔ The mechanism is the sentence a human had to type. ``--parity-off-reason ''``
    is a boolean wearing a string's clothes and must not pass."""
    p = _parity_problems(_v2_argv(tmp_path, "--parity-off-reason", empty))
    assert p, f"empty reason {empty!r} unlocked the parity guard"


def test_it_is_not_spelled_as_a_bare_force_flag():
    """A ``--force`` boolean is the shape this deliberately is NOT — stated as a test
    so a future 'simplification' has to delete an assertion to make it."""
    opts = {o for a in T.build_parser()._actions for o in a.option_strings}
    assert "--force" not in opts and "--force-parity" not in opts
    act = next(a for a in T.build_parser()._actions
               if a.dest == "parity_off_reason")
    assert act.nargs is None and act.const is None, \
        "--parity-off-reason must take a VALUE, not act as a store_true"


def test_the_two_parity_flags_are_mutually_exclusive(tmp_path):
    p = _parity_problems(_v2_argv(tmp_path, "--require-parity",
                                  "--parity-off-reason", REASON))
    assert any("BOTH" in x for x in p), p


# --------------------------------------------------------------------------- #
# The reason must SURVIVE the staged command — for BOTH flags.                 #
# --------------------------------------------------------------------------- #
def test_parity_reason_survives_staged_command(tmp_path):
    a = T.build_parser().parse_args(
        _v2_argv(tmp_path, "--parity-off-reason", REASON))
    staged = T._staged_command(a)
    assert "--parity-off-reason" in staged
    # quoted, so a multi-word reason survives a copy-paste as ONE argv element
    assert REASON in " ".join(shlex.split(staged))
    assert shlex.split(staged)[shlex.split(staged).index("--parity-off-reason") + 1] \
        == REASON


def test_heldout_reason_survives_staged_command_too(tmp_path):
    """⚠️ REGRESSION. ``--heldout-off-reason`` was absent from ``_staged_command``
    from the day it was added, so the copied command tripped the very preflight the
    reason exists to satisfy — the run never started, and the operator had to
    rediscover a justification they had already written."""
    a = T.build_parser().parse_args(
        _v2_argv(tmp_path, "--no-heldout-gate", "--heldout-off-reason", REASON,
                 "--parity-off-reason", REASON))
    toks = shlex.split(T._staged_command(a))
    assert "--heldout-off-reason" in toks
    assert toks[toks.index("--heldout-off-reason") + 1] == REASON


def test_a_staged_command_with_no_reasons_carries_neither_flag(tmp_path):
    """No spurious empty flags when the guards are simply satisfied."""
    staged = T._staged_command(
        T.build_parser().parse_args(_v2_argv(tmp_path, "--require-parity")))
    assert "--parity-off-reason" not in staged
    assert "--heldout-off-reason" not in staged


def test_the_staged_command_reparses_and_still_passes_preflight(tmp_path):
    """⭐ THE END-TO-END PROPERTY: what the human copies must RUN. Reconstruct,
    re-parse, re-check — the round trip is the only thing that proves a dropped
    argument is really impossible rather than merely unlikely."""
    argv = _v2_argv(tmp_path, "--parity-off-reason", REASON)
    a = T.build_parser().parse_args(argv)
    toks = shlex.split(T._staged_command(a))
    # drop the launcher prefix (`PYTHONPATH=... python3 scripts/train_flagship_v4.py`)
    toks = toks[toks.index("scripts/train_flagship_v4.py") + 1:]
    assert _parity_problems(toks) == []


# --------------------------------------------------------------------------- #
# The reason must be ECHOED, not merely stored.                                #
# --------------------------------------------------------------------------- #
def test_parity_reason_is_printed_at_launch(tmp_path, capsys):
    rc = T.main(_v2_argv(tmp_path, "--parity-off-reason", REASON))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "PREFLIGHT: OK" in out
    assert "NON-PARITY" in out and REASON in out


def test_heldout_reason_is_printed_at_launch(tmp_path, capsys):
    """⚠️ REGRESSION: advertised as "echoed at launch"; it reached only
    ``config.json``'s ``args`` blob."""
    T.main(_v2_argv(tmp_path, "--parity-off-reason", "x-ood-probe",
                    "--no-heldout-gate", "--heldout-off-reason", REASON))
    out = capsys.readouterr().out
    assert "HELD-OUT GATE OFF" in out and REASON in out


def test_no_banner_when_no_guard_is_off(tmp_path, capsys):
    T.main(_v2_argv(tmp_path, "--require-parity"))
    out = capsys.readouterr().out
    assert "NON-PARITY — recorded reason" not in out
    assert "HELD-OUT GATE OFF" not in out


def test_a_heldout_reason_without_the_flag_is_not_announced(tmp_path):
    """A reason recorded while the gate is still ON must not claim it is off."""
    a = T.build_parser().parse_args(
        _v2_argv(tmp_path, "--require-parity", "--heldout-off-reason", REASON))
    assert T._off_reason_banner(a) == []


# --------------------------------------------------------------------------- #
# Classification contract (the generic test's assertion, localised).           #
# --------------------------------------------------------------------------- #
def test_the_new_argument_is_classified_as_a_non_path():
    assert "parity_off_reason" in T.NOT_A_PATH
    assert "parity_off_reason" not in T.PATH_ARGS
