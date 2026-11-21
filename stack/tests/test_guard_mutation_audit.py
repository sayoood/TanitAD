"""The guard-for-the-guards: `guard_mutation_audit`'s registry cannot rot.

⛔ WHY. `guard_mutation_audit.py` is the only evidence in the repo that the M18
provenance guards are load-bearing rather than decorative — it reintroduces
each MEASURED defect and checks the named test goes red. Its entire value rests
on its anchors still matching the shipped source. A refactor that re-words
`_build_rig_camera` would leave the audit applying nothing, catching nothing,
and reporting a clean run: **a green audit certifying an absent guard**, which
is strictly worse than no audit at all — it converts an unknown into a false
reassurance.

These tests are FAST (no pytest subprocess, no mutation, no model) so they run
in the ordinary suite. The audit itself is the on-demand instrument:

    python stack/scripts/guard_mutation_audit.py

⚠️ These tests do NOT re-run the mutations. They check the audit is still
*capable* of running them. The separation is deliberate: a test that mutates
the source it is running from cannot be run in parallel, and a suite that
mutates the working tree is a trap of its own.
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import guard_mutation_audit as gma                            # noqa: E402

STACK = Path(__file__).resolve().parents[1]


def test_the_registry_is_not_empty_and_covers_both_sides_of_the_defect():
    """The defect had a trainer half and a loss half, and both must be armed.

    `model._rig_camera = None` (trainer) and the `cam is not None` guard
    (`refc_agents`) are the two ends of the SAME silent no-op. An audit that
    only mutated the trainer would certify a guard chain with an unexercised
    link.
    """
    assert len(gma.MUTATIONS) >= 6
    paths = {m.path for m in gma.MUTATIONS}
    assert "scripts/refc_v3_train.py" in paths
    assert "tanitad/refs/refc_agents.py" in paths


def test_every_anchor_is_present_in_the_shipped_source_exactly_once():
    """The rot check. A failure here is NOT a licence to delete the entry —
    re-point it at the guard's current text, or the guard loses its evidence.

    ⛔ EXCLUDES THE MUTATION CURRENTLY APPLIED, and only that one. During an
    audit the applied mutation's anchor is legitimately absent, so a blanket
    check goes red on every mutation — which MEASURABLY destroyed the audit's
    resolution (see `RUNNING_ENV`): with this test red every time, `named` was
    never the only failure and no mutation could ever read ESCAPED. Every
    OTHER anchor is still checked, which is exactly what catches
    `registry_anchor_rotted`.
    """
    applied = os.environ.get(gma.RUNNING_ENV, "")
    muts = [m for m in gma.MUTATIONS if m.key != applied]
    assert len(muts) >= len(gma.MUTATIONS) - 1
    bad = gma.check_anchors(muts)
    assert not bad, "the mutation registry has rotted:\n  " + "\n  ".join(bad)


def test_the_rot_check_ITSELF_can_fail():
    """A control that has never read the wrong value certifies nothing —
    including this one. `check_anchors` is what stands between the audit and
    silent vacuity, so it is shown here to reject both failure modes: an
    anchor that is absent, and an anchor that is present more than once (where
    `replace(..., 1)` would mutate whichever came first — a different
    experiment under the same name).
    """
    absent = gma.Mutation(
        key="synthetic_absent", defect="x", path="scripts/refc_v3_train.py",
        old="THIS STRING IS NOT IN THE TRAINER", new="y", caught_by=())
    assert gma.check_anchors([absent]), "check_anchors passed an ABSENT anchor"

    # `raise SystemExit(` appears 38x in the trainer (MEASURED 2026-09-05) —
    # an anchor that a `replace(..., 1)` would apply to whichever came first.
    dup = gma.Mutation(
        key="synthetic_duplicate", defect="x",
        path="scripts/refc_v3_train.py", old="raise SystemExit(", new="y",
        caught_by=())
    problems = gma.check_anchors([dup])
    assert problems and "occurs" in problems[0], (
        "check_anchors accepted a NON-UNIQUE anchor, so a rotted registry "
        "could mutate a different site than the one it names")

    missing_file = gma.Mutation(
        key="synthetic_nofile", defect="x", path="scripts/does_not_exist.py",
        old="a", new="b", caught_by=())
    assert gma.check_anchors([missing_file])


def _test_functions(rel: str) -> set[str]:
    tree = ast.parse((STACK / rel).read_text(encoding="utf-8"))
    return {n.name for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")}


def test_every_named_guard_actually_exists():
    """A `caught_by` node id that names a renamed or deleted test can never
    match, so the audit would report MISCREDITED forever and the real message
    — *this guard is gone* — would be buried in a verdict about crediting.
    """
    cache: dict[str, set[str]] = {}
    for m in gma.MUTATIONS:
        for nodeid in m.caught_by:
            rel, _, name = nodeid.partition("::")
            assert (STACK / rel).is_file(), f"{m.key}: no such test file {rel}"
            names = cache.setdefault(rel, _test_functions(rel))
            assert name in names, (
                f"{m.key} names {nodeid}, which does not exist. Either the "
                "guard was renamed (re-point the registry) or it was deleted "
                "(the defect is unguarded again).")


def test_every_mutation_names_at_least_one_guard():
    """A mutation with no `caught_by` can only ever read ESCAPED or
    MISCREDITED — it asks a question whose answer is not a verdict.
    """
    for m in gma.MUTATIONS:
        assert m.caught_by, f"{m.key} names no guard"
        assert m.defect.strip(), f"{m.key} does not say what defect it is"
        assert m.new != m.old, f"{m.key} is a no-op mutation"


def test_the_test_files_the_audit_runs_all_exist():
    """The audit's verdicts are only as wide as the files it runs: a defect
    caught by a suite the audit never invokes reads as ESCAPED.
    """
    for rel in gma.TEST_FILES:
        assert (STACK / rel).is_file(), f"audit runs a missing file: {rel}"


def test_every_mutated_file_is_reachable_from_the_stack_root():
    """`STACK` is resolved from the script's own location, so a move into
    another directory would silently retarget every path.
    """
    assert (gma.STACK / "scripts" / "refc_v3_train.py").is_file()
    for m in gma.MUTATIONS:
        assert (gma.STACK / m.path).is_file(), f"{m.key}: {m.path} missing"


def test_every_line_the_audit_PRINTS_is_cp1252_safe():
    """MEASURED 2026-09-05: the audit crashed on its own ESCAPED branch.

    This console is cp1252 (see the repo's text-encoding sweep), and
    ``print("⛔ the suite stayed GREEN ...")`` raised ``UnicodeEncodeError``.
    The verdict that says *your guard is decorative* — the single most
    important thing this tool can ever say, and the branch least likely to be
    exercised — died instead of being read. Docstrings keep their symbols;
    only what reaches stdout is checked, because only that is encoded.

    ``_harden_streams`` is the backstop for text that is not ours (a
    mutation's prose, a pytest tail). It is not a licence to put symbols back
    into these literals: with ``backslashreplace`` the loud line still prints
    as an escape, which is unreadable exactly when it matters.
    """
    src = (gma.STACK / "scripts" / "guard_mutation_audit.py").read_text(
        encoding="utf-8")
    offenders = []
    for node in ast.walk(ast.parse(src)):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id in ("print", "SystemExit")):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str) \
                    and any(ord(c) > 127 for c in sub.value):
                offenders.append(f"L{sub.lineno}: {ascii(sub.value[:60])}")
    assert not offenders, (
        "these literals reach stdout and are not cp1252-encodable:\n  "
        + "\n  ".join(offenders))


def test_the_stream_backstop_is_installed_before_anything_prints():
    """`_harden_streams` must be the first statement of `main`, or the
    registry-rot message — which prints before it — could still crash.
    """
    src = (gma.STACK / "scripts" / "guard_mutation_audit.py").read_text(
        encoding="utf-8")
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    first = fn.body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call) \
        and getattr(first.value.func, "id", None) == "_harden_streams", (
            "main() does not start with _harden_streams()")


def test_the_audit_runs_THIS_file_too():
    """Otherwise the two `registry_*` mutations — the ones that rot the audit
    on purpose — would read ESCAPED: the guard that catches them lives here,
    and a mutation is only ever as caught as the files the audit runs.
    """
    assert "tests/test_guard_mutation_audit.py" in gma.TEST_FILES


def test_the_audit_never_leaves_a_backup_directory_behind():
    """The snapshot dir is removed on the way out, success or failure. If one
    is present in a clean checkout, a previous run died mid-mutation and the
    working tree may still carry an injected defect.

    ⚠️ Skipped while an audit is in flight (`RUNNING_ENV`), where the dir
    exists because THIS process is the run. Without that, the audit's own
    baseline would be red and every verdict after it unreadable.
    """
    if os.environ.get(gma.RUNNING_ENV):
        import pytest
        pytest.skip("an audit is in flight; the snapshot dir is expected")
    assert not gma.BACKUP.exists(), (
        f"{gma.BACKUP} exists — a mutation run died. Run "
        "`python stack/scripts/guard_mutation_audit.py` to restore, and "
        "verify the two mutated files against git before trusting any "
        "test result from this tree.")
