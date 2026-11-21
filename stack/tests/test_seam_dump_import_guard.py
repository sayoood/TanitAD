"""A missing optional module must NEVER kill the trainer at a checkpoint.

WHY THIS EXISTS (2026-08-18, found by the Thor closure audit). In
`train_v6_staged.py` the seam-dump block imported `SeamDumpError` INSIDE the
`try` whose `except` clause names it:

    try:
        from taniteval.seam_dump import SeamDumpError, ...
        ...
    except SeamDumpError as e:      # <- unbound if the import failed
        ...
    except Exception as e:          # <- NEVER REACHED
        ...

If the import raises, Python evaluates `except SeamDumpError`, hits an unbound
name, and that error PROPAGATES OUT OF THE WHOLE `try` STATEMENT -- the broad
handler below is not consulted. MEASURED: it escapes as
`UnboundLocalError: cannot access local variable 'SeamDumpError'`.

⛔ WHY IT MATTERED: the block sits IMMEDIATELY BEFORE `_save_ckpt`, so the
failure kills the trainer at a checkpoint boundary. It fires exactly when
`taniteval` is off `PYTHONPATH` -- the live run's own configuration. S-W escaped
only because the chain emits `--dump-seam-plan` on S-T/S-S/S-J and not on S-W.
**S-T would have hit it.**

The first test below reproduces the language-level trap directly, so it fails
against the old shape and passes against the new one. The rest pin the repaired
source so the shape cannot come back.
"""
import ast
import inspect
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "scripts" / "train_v6_staged.py"


def _seam_block():
    """The `if getattr(a, "dump_seam_plan", ...)` block, as an AST node.

    ⚠️ ANCHOR ON THE CONDITION, NOT ON TOKEN CONTAINMENT OR SIZE. Two wrong
    finders were written before this one, and each produced confidently wrong
    assertions:
      * "first `If` containing both tokens" returned the ENCLOSING
        `if step % a.save_every == 0` checkpoint block, silently widening every
        assertion to a block the repair does not live in;
      * "smallest such `If`" returned the INNER `if seam_dump_from_plan is not
        None:` sentinel check, silently narrowing them;
      * "matches `getattr(a, "dump_seam_plan", ...)`" returned an unrelated
        `if not getattr(a, "dump_seam_plan", None): return ""` guard clause in
        another function entirely.
    The scope error THREE TIMES, inside a test helper written to catch a scope
    error — and each wrong finder returned something plausible rather than
    failing. ⇒ A locator must be pinned by a CONJUNCTION that only the intended
    node satisfies, and it must ASSERT UNIQUENESS rather than take a first match.
    """
    src = _SRC.read_text(encoding="utf-8")
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.get_source_segment(src, node.test) or ""
        seg = ast.get_source_segment(src, node) or ""
        if ("getattr(a" in test and "dump_seam_plan" in test
                and "seam_dump_from_plan(" in seg and "SeamDumpError" in seg):
            hits.append((node, seg))
    if not hits:
        pytest.fail("seam-dump block not found in train_v6_staged.py")
    assert len(hits) == 1, (
        f"locator matched {len(hits)} blocks; it must identify exactly one, "
        "or every assertion below silently applies to the wrong scope"
    )
    return hits[0]


# ------------------------------------------------------- the trap, reproduced --
def test_an_except_clause_naming_a_symbol_imported_inside_its_own_try_escapes():
    """The language-level fact the defect rested on. Not a mock -- the real rule."""
    def old_shape():
        try:
            from definitely_not_a_real_module_xyz import Boom
            return "ran"
        except Boom:                                          # noqa: F821
            return "specific"
        except Exception:                                     # noqa: BLE001
            return "broad"

    with pytest.raises((UnboundLocalError, NameError)):
        old_shape()


def test_the_repaired_shape_degrades_to_a_note():
    """Same failure, guarded import: returns instead of raising."""
    def new_shape():
        try:
            from definitely_not_a_real_module_xyz import Boom
        except Exception:                                     # noqa: BLE001
            Boom = worker = None
        else:
            worker = Boom
        if worker is None:
            return "note"
        try:
            return "ran"
        except Boom:                                          # pragma: no cover
            return "specific"

    assert new_shape() == "note"


# ------------------------------------------------------------- source pins --
def test_the_import_is_not_inside_the_try_that_handles_its_own_symbol():
    """The structural invariant: no `try` may both import and except a name."""
    node, _ = _seam_block()
    offenders = []
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Try):
            continue
        imported = {
            alias.asname or alias.name
            for st in ast.walk(sub) if isinstance(st, ast.ImportFrom)
            for alias in st.names
        }
        for handler in sub.handlers:
            if handler.type is None:
                continue
            named = {
                n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name)
            }
            clash = named & imported
            if clash:
                offenders.append((sub.lineno, sorted(clash)))
    assert not offenders, (
        "a `try` imports a name AND excepts it -- if the import fails the "
        "except clause is unbound and the error escapes the whole statement, "
        f"skipping every later handler: {offenders}"
    )


def test_the_guarded_import_binds_a_sentinel_on_failure():
    """Failure must leave a checkable value, not a dangling name."""
    _, seg = _seam_block()
    assert "seam_dump_from_plan = None" in seg or "= seam_dump_from_plan = None" in seg
    assert "if seam_dump_from_plan is not None:" in seg


def test_the_broad_handler_still_exists_so_a_dump_failure_never_kills_a_run():
    """The repair must not have removed the never-kill-a-run guarantee."""
    node, _ = _seam_block()
    broad = [
        h for sub in ast.walk(node) if isinstance(sub, ast.Try)
        for h in sub.handlers
        if isinstance(h.type, ast.Name) and h.type.id == "Exception"
    ]
    assert len(broad) >= 2, (
        "expected a broad handler on BOTH the guarded import and the dump body"
    )


def test_the_defect_is_documented_where_the_next_reader_will_look():
    _, seg = _seam_block()
    low = seg.lower()
    for token in ("closure audit", "checkpoint boundary", "pythonpath"):
        assert token in low, f"seam-dump block must name {token!r}"


def test_the_module_still_parses_and_the_block_is_reachable():
    src = _SRC.read_text(encoding="utf-8")
    ast.parse(src)
    assert src.count("dump_seam_plan") >= 3
