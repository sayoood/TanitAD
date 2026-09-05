"""The STRATEGIC nav-compliance family died on every refcv3 arm because a manifest
key held a dict where the reader wanted a path. These pin the shape contract.

⛔ **The defect was invisible in exactly the way that matters**: `from_refcv3_dump`'s
caller wraps the call in `except Exception` and converts it to a REFUSAL, so the
family read UNAVAILABLE with a reason nobody parsed, while LONGITUDINAL / LATERAL /
TACTICAL reported normally. A binding metric family (`CLAUDE.md`: *every eval reports
four metric families*) was absent from every published refcv3 eval, and no run failed.

MEASURED 2026-09-05 on the RL panel: 3/3 strategic nav-compliance rows lost across
4,823 windows / 141 episodes of compute that had already been paid for.
"""
from __future__ import annotations

import ast
import io
import os

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAV = os.path.join(REPO, "taniteval", "taniteval", "nav_compliance.py")
ARM = os.path.join(REPO, "taniteval", "tools", "refcv3_arm.py")


def _read(p: str) -> str:
    with io.open(p, encoding="utf-8") as fh:
        return fh.read()


def _resolver():
    """`resolve_labels_path` loaded from source, without importing the package.

    `taniteval` is a namespace package that shadows on this tree (see CLAUDE.md),
    so the test compiles the one function rather than importing the module.
    """
    tree = ast.parse(_read(NAV))
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "resolve_labels_path"),
              None)
    assert fn is not None, "resolve_labels_path is gone from nav_compliance.py"
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), NAV, "exec"), ns)
    return ns["resolve_labels_path"]


# --------------------------------------------------------------- the three shapes


def test_the_provenance_dict_yields_its_path_not_a_typeerror():
    """The shape EVERY banked dump actually carries."""
    corpus = {"labels": {"path": "/data/v7_2.jsonl.gz", "md5": "abc", "n_records": 9}}
    assert _resolver()(None, corpus) == "/data/v7_2.jsonl.gz"


def test_the_explicit_labels_path_key_is_preferred():
    """The shape written from 2026-09-05 onward, alongside the provenance block."""
    corpus = {"labels_path": "/data/new.jsonl.gz",
              "labels": {"path": "/data/new.jsonl.gz", "md5": "abc"}}
    assert _resolver()(None, corpus) == "/data/new.jsonl.gz"


def test_a_bare_string_still_reads_the_pre_join_era():
    assert _resolver()(None, {"labels": "/data/old.jsonl.gz"}) == "/data/old.jsonl.gz"


def test_an_explicit_argument_beats_the_manifest():
    corpus = {"labels": {"path": "/data/from_manifest.gz"}}
    assert _resolver()("/data/from_cli.gz", corpus) == "/data/from_cli.gz"


def test_a_manifest_with_no_labels_at_all_returns_falsy_not_an_exception():
    """The caller must reach its own REFUSAL branch, which names the missing blob."""
    assert not _resolver()(None, {})


# ------------------------------------------------- the deliberate-regression control


def test_the_old_reading_really_did_raise_so_this_test_can_fail():
    """⛔ Without this the suite proves nothing: a gate never shown to FAIL the
    defect certifies nothing. This reproduces the exact pre-fix expression."""
    with pytest.raises(TypeError):
        os.path.exists({"path": "/data/v7_2.jsonl.gz"})   # noqa: PTH110


# ------------------------------------------------------------- the writer's shape


def test_the_writer_publishes_the_path_under_an_unambiguous_key():
    """`**join` overwrites any literal `"labels"`, so the path needs its own key."""
    src = _read(ARM)
    assert '"labels_path": a.labels' in src, "refcv3_arm.py stopped publishing the path"
    # the collision itself must not come back: a literal "labels": a.labels inside
    # the corpus dict is dead code that LOOKS like it publishes the path.
    live = [ln for ln in src.splitlines()
            if '"labels": a.labels' in ln and not ln.lstrip().startswith("#")]
    assert not live, f"the dead, misleading literal is back: {live}"


def test_join_still_carries_the_provenance_block_under_labels():
    """The dict is not a bug — it is the richer record, and the reader now reads it."""
    src = _read(ARM)
    assert 'join = {"labels": {"path": a.labels' in src


# ------------------------------------------ a defect must not look like a refusal


def _arm_defect_tuple():
    """`DEFECT_EXCEPTIONS` evaluated from source.

    Importing `refcv3_arm` runs a preflight that can `sys.exit`, so the constant is
    read the same way `resolve_labels_path` is: compiled out of the module, not
    imported.
    """
    tree = ast.parse(_read(ARM))
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "DEFECT_EXCEPTIONS"
                for t in n.targets):
            return eval(compile(ast.Expression(n.value), ARM, "eval"))  # noqa: S307
    raise AssertionError("DEFECT_EXCEPTIONS is gone from refcv3_arm.py")


def test_a_typeerror_from_our_own_code_is_classified_as_a_defect():
    """The exact exception that removed the STRATEGIC family from every arm."""
    assert isinstance(TypeError("x"), _arm_defect_tuple())


def test_a_missing_input_is_NOT_a_defect_so_honest_refusals_still_refuse():
    """⛔ The four-families rule permits dropping a family that genuinely cannot
    be computed, WITH its reason. Classifying those as defects would make the
    distinction useless in the other direction."""
    d = _arm_defect_tuple()
    assert not isinstance(FileNotFoundError("no labels"), d)
    assert not isinstance(ValueError("this file is not what you said it was"), d)
    assert not isinstance(KeyError("absent"), d)


def test_the_record_marks_the_block_and_collects_it_at_the_top_level():
    """A driver must be able to exit non-zero rather than publish a
    family-shaped hole; that needs the flag ON the block and a list beside it."""
    src = _read(ARM)
    assert '_blk["defect"] = True' in src
    assert 'ref.setdefault("_defects", [])' in src
    assert '"defect_type"' in src


def test_the_defect_reason_says_defect_so_a_reader_cannot_miss_it():
    src = _read(ARM)
    assert "DEFECT (not a refusal)" in src, \
        "the reason text must name it — the old one read like an ordinary refusal"
