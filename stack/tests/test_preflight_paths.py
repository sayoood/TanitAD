"""⛔ `PREFLIGHT: OK` MUST NOT COVER AN INPUT IT NEVER LOOKED AT.

THE DEFECT (MEASURED 2026-07-27)
--------------------------------
``--anchors-dense /workspace/experiments/anchors/anchors_dense_1to20.pt`` is in
BOTH published v5 launch commands and **does not exist on pod2** — that
directory is empty; the real file is
``/workspace/experiments/flagship_v4_anchors_dense.pt``. ``--print-launch``
verified that the *argument was present*, never that the *path existed*, and
printed ``PREFLIGHT: OK`` for a command that dies after the model build.

⚠️ ``--require-parity`` had already been caught in the same shape (a cache whose
``corpus_key_of`` resolved to ``None``). Two occurrences of one class, so the
fix is not another hand-maintained check but an **exhaustiveness contract**:
:data:`train_flagship_v4.PATH_ARGS` ∪ :data:`train_flagship_v4.NOT_A_PATH` must
cover every free-form string argument the parser accepts. Adding
``--foo-cache`` without classifying it fails this file.

⭐ The RED half — :func:`test_preflight_BLOCKS_a_missing_anchors_file` — is what
makes the rest meaningful. A preflight that cannot fail is cover, not a guard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import train_flagship_v4 as T  # noqa: E402


def _base_argv(tmp: Path, **over) -> list[str]:
    """A --print-launch command whose NON-path gates all pass, so anything this
    file reports comes from the path layer alone."""
    (tmp / "traincache").mkdir(exist_ok=True)
    (tmp / "valcache").mkdir(exist_ok=True)
    (tmp / "anchors.pt").write_bytes(b"\0")
    args = {
        "--train-cache": str(tmp / "traincache"),
        "--val-cache": str(tmp / "valcache"),
        "--anchors-dense": str(tmp / "anchors.pt"),
        "--out": str(tmp / "run"),
        "--steps": "30000", "--gate-step": "10000",
        "--batch": "8", "--accum": "8",
        "--phase-a-steps": "2000", "--phase-b-steps": "8000",
        "--heldout-every": "2000", "--heldout-episodes": "8",
        "--heldout-patience": "2",
    }
    args.update(over)
    out = ["--print-launch", "--from-scratch"]
    for k, v in args.items():
        if v is not None:
            out += [k, v]
    return out


def _problems(argv) -> list[str]:
    return T.preflight_asserts(T.build_parser().parse_args(argv))


# --------------------------------------------------------------------------- #
# RED — demonstrate the preflight FAILING.                                     #
# --------------------------------------------------------------------------- #
def test_preflight_BLOCKS_a_missing_anchors_file(tmp_path, capsys):
    """⭐ THE DEMONSTRATED FAILURE, on the exact flag that shipped broken."""
    argv = _base_argv(tmp_path,
                      **{"--anchors-dense": str(tmp_path / "anchors"
                                                / "anchors_dense_1to20.pt")})
    rc = T.main(argv)
    out = capsys.readouterr().out
    assert rc == 2
    assert "PREFLIGHT: BLOCKED" in out
    assert "[PATH-PREFLIGHT] --anchors-dense" in out
    assert "DOES NOT EXIST" in out


def test_the_same_command_with_the_real_file_is_OK(tmp_path, capsys):
    """…and it is not blanket-refusing either: one path changes, one verdict."""
    rc = T.main(_base_argv(tmp_path))
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "PREFLIGHT: OK" in out
    assert "[PATH-PREFLIGHT]" not in out


@pytest.mark.parametrize("flag,dest", [("--poses-train", "poses_train"),
                                       ("--labels-val", "labels_val"),
                                       ("--probes", "probes"),
                                       ("--anchors-coarse", "anchors_coarse")])
def test_every_other_input_path_is_checked_too(tmp_path, flag, dest):
    """⚠️ ``--poses-*`` and ``--labels-*`` are free-form string paths that the
    original hand-written preflight never mentioned. This is the 'checks only
    some of its inputs' failure, enumerated."""
    p = _problems(_base_argv(tmp_path, **{flag: str(tmp_path / "nope.json")}))
    assert any(f"[PATH-PREFLIGHT] {flag}" in x for x in p), p


def test_a_missing_output_parent_is_blocked_not_the_output_itself(tmp_path):
    """``--out`` need not exist; its PARENT must, or the run builds the model
    and then dies writing ``config.json``."""
    ok = _problems(_base_argv(tmp_path, **{"--out": str(tmp_path / "new-run")}))
    assert not [x for x in ok if "[PATH-PREFLIGHT] --out" in x], ok
    bad = _problems(_base_argv(tmp_path,
                               **{"--out": str(tmp_path / "no" / "such" / "run")}))
    assert any("[PATH-PREFLIGHT] --out" in x for x in bad), bad


def test_trunk_none_is_a_choice_not_a_missing_file(tmp_path):
    """'none' is the documented spelling for 'no warm-start'. Treating it as a
    path would refuse the deliberate case."""
    argv = [x for x in _base_argv(tmp_path) if x != "--from-scratch"]
    assert not [x for x in _problems(argv + ["--trunk", "none"])
                if "[PATH-PREFLIGHT] --trunk" in x]
    assert any("[PATH-PREFLIGHT] --trunk" in x
               for x in _problems(argv + ["--trunk", str(tmp_path / "no.pt")]))


def test_a_file_given_where_a_directory_is_required_is_reported(tmp_path):
    (tmp_path / "notadir").write_bytes(b"")
    p = _problems(_base_argv(tmp_path,
                             **{"--train-cache": str(tmp_path / "notadir")}))
    assert any("is not a DIRECTORY" in x for x in p), p


# --------------------------------------------------------------------------- #
# The contract that stops the NEXT flag slipping through.                      #
# --------------------------------------------------------------------------- #
def test_path_classification_is_exhaustive_over_the_parser():
    """⭐ THE REAL FIX. Every free-form string argument is either a classified
    path or an explicitly listed non-path. This is what a hand-maintained check
    list cannot give you — and ``--poses-*``/``--labels-*`` prove the list would
    have been incomplete from day one."""
    parser = T.build_parser()
    freeform = {act.dest for act in parser._actions
                if act.dest != "help" and not act.choices and act.nargs != 0
                and act.type in (None, str)}
    classified = set(T.PATH_ARGS) | set(T.NOT_A_PATH)
    unclassified = sorted(freeform - classified)
    assert unclassified == [], (
        f"unclassified free-form string args {unclassified} — add them to "
        f"train_flagship_v4.PATH_ARGS (checked at preflight) or NOT_A_PATH "
        f"(with the reason). This test exists because --anchors-dense shipped "
        f"in two published launch commands pointing at a path that does not "
        f"exist, under PREFLIGHT: OK.")
    stale = sorted(classified - freeform)
    assert stale == [], f"classified args no longer on the parser: {stale}"


def test_every_path_arg_the_staged_command_prints_is_classified(tmp_path):
    """The staged command is what a human copies. Any ``--flag <path>`` pair it
    emits must be one the preflight looked at."""
    a = T.build_parser().parse_args(_base_argv(tmp_path))
    staged = T._staged_command(a)
    flags_in_cmd = {tok for tok in staged.split() if tok.startswith("--")}
    printed_paths = {flag for _, (_, flag) in T.PATH_ARGS.items()}
    for flag in sorted(flags_in_cmd & printed_paths):
        dest = flag.lstrip("-").replace("-", "_")
        assert dest in T.PATH_ARGS, flag
