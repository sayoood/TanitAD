#!/usr/bin/env python3
"""Audit a remote box's ``stack/`` against the repo over the LAUNCH IMPORT CLOSURE.

⛔ **THE CLASS THIS EXISTS TO KILL (RETRACTION_LOG C99, 2026-08-18).** An agent
shipped the three files it had *changed* to Thor, verified **all three md5s
byte-identical on both sides**, and the code still could not run::

    ImportError: cannot import name 'K_MAX_GRID' from 'refc_dump_latents'

Thor's ``refc_dump_latents.py`` was **11,629 B with zero occurrences of
``K_MAX_GRID``** against the repo's **30,089 B with it at line 96** — a 2.6×
smaller, badly stale dependency that was never listed *because it had not been
edited*. Two rules fall out, and this script mechanises both:

1. **The ship set is the IMPORT CLOSURE, computed — never the diff.** Staleness
   is a property of the **target**, not of your **changes**. A hand-listed
   dependency set is a guess about what a launch touches. (Precedent: a launch
   closure derived by AST came to **76** dests where **13** had been
   hand-listed.)
2. ⚠️ **md5 agreement proves TRANSFER, not FUNCTION.** Three green checksums on
   the wrong file set is a confident wrong answer — the same shape as reading
   ``df`` on a pod. Only a **real import** is evidence a box can run the code,
   so ``--verify-import`` is part of the tool, not an afterthought.

⚠️ **THE CRLF TRAP — it produced 7 false "drift" rows out of 10 on 2026-08-16.**
This repo's working tree is **CRLF**; Thor is **LF**. A naive md5 comparison
reports drift that does not exist. Every comparison here is made on **two**
digests:

===============  ==================================================
``md5_raw``      md5 of the bytes exactly as they sit on disk
``md5_lf``       md5 after ``b"\\r\\n" -> b"\\n"`` **only** (no other
                 rewriting: no trailing-whitespace strip, no final-
                 newline fixup, no encoding round-trip)
===============  ==================================================

and the verdict names which one decided it, so a CRLF artifact can never again
be reported as drift:

===================  ==============================================
``SAME``             ``md5_raw`` equal — identical bytes
``CRLF_ONLY``        ``md5_raw`` differs, ``md5_lf`` equal — **NOT
                     drift**, a line-ending artifact
``DRIFT``            ``md5_lf`` differs — genuinely different code
``MISSING_REMOTE``   in the closure, absent on the box → an import
                     that cannot resolve
``MISSING_LOCAL``    on the box, absent in the repo (only reachable
                     via ``--remote-closure``)
===================  ==============================================

⛔ **AND THE SAME CLASS ONE LEVEL UP AGAIN (C105, 2026-08-18).** This tool's own
entry-point list was a hand-list of **7**. Widening it to **14** grew the closure
from 120 files to **134 and found three more stale files on Thor**, one of them
``taniteval/v0_antiecho.py`` at 46,905 B, absent from the box entirely. **A
closure is only as complete as the set it is closed OVER.**

⇒ The entry points are now **DERIVED** (:func:`derive_entries`) from
:data:`DEFAULT_LAUNCH_SOURCES` — the ladder launcher, the operator runbook and
the gate protocol — by reading the ``*.py`` tokens they actually name, then
following each named script as a launch source in turn (a fixed point, because
``train_v6_staged.py`` *subprocesses* ``eval_four_families.py``, ``seam_probe.py``
and ``t1_eval.py``, which no import walk can see). MEASURED on this repo:
**52 entry points → 161 files**, a strict superset of both hand-lists.

⚠️ **The recursion bottoms out at the launch-source list, and that list is
PRINTED ON EVERY RUN** (:func:`print_root_set`) together with the coverage gap —
how many launchable scripts the closure does *not* reach. A root set nobody sees
is a root set nobody checks; that is precisely how the 7 survived.

⚠️ **A derivation must WIDEN, never NARROW.** Four instruments C105 carried are
named by no current launch source, so :data:`DEFAULT_ENTRIES` is kept as a
**floor** and its members are flagged ``FLOOR ONLY`` rather than dropped.

Usage
-----
Audit (read-only; zero GPU, one ``ssh -n``)::

    python3 stack/scripts/launch_closure_audit.py \\
        --host tanitad-thor-wifi --remote-root /home/nvidia/TanitAD \\
        --json /tmp/closure.json

Ship exactly the DRIFT/MISSING_REMOTE rows, backing the box's copies up first,
then prove the box can actually run it::

    python3 stack/scripts/launch_closure_audit.py --host tanitad-thor-wifi \\
        --remote-root /home/nvidia/TanitAD --ship \\
        --backup-dir /home/nvidia/_thor_backup_$(date +%F)/ --verify-import

Probe hygiene baked in
----------------------
* ``ssh -n`` on **every** remote call — a nested ssh inside a pipe/heredoc eats
  the rest of the script's stdin and the tail silently never runs.
* The remote computes its own answer and emits **one opaque ``ZZ…ZZ`` frame**;
  nothing here greps the raw stream for a token the command itself contains.
  That is the PTY-echo trap, which has invented a false failure three times.
* Files are shipped **LF-normalised**, because the target is Linux. The md5 the
  ship verifies against is therefore ``md5_lf``, and the script says so.
* ⛔ Nothing here runs ``git`` on the remote. ``git fetch`` on a training box
  hangs, and a failed fetch followed by a checkout **resets the tree**,
  destroying shipped files.
"""

from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# repo geometry
# --------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent           # stack/scripts
_STACK = _HERE.parent                             # stack
_REPO = _STACK.parent                             # repo root

#: sys.path roots, in the order a launched entry point establishes them.
#: ``train_v6_staged.py`` does ``sys.path.insert(0, scripts/)`` then
#: ``sys.path.insert(1, stack/)``; ``e_wc2_sigma_star.py`` additionally inserts
#: ``<repo>/taniteval`` before ``from taniteval import ci``.
DEFAULT_ROOTS = ("stack/scripts", "stack", "taniteval")

#: Directories an **executable** lives in. A ``.py`` outside these is a library
#: module reached by ``import``, never something a launch line *runs*. This is
#: the filter that keeps the derived entry set to launchable things.
DEFAULT_ENTRY_ROOTS = ("stack/scripts", "taniteval/tools", "stack/ops")

#: ⛔ **C105's DEFAULT, WHICH IS WHY IT IS NOT THE DEFAULT ANY MORE.** These
#: seven are the v6 ladder as it was hand-listed on 2026-08-18. Their closure is
#: **120 files**; widening the hand-list to 14 grew it to **134 and found three
#: more stale files on Thor** — `four_families.py`, `hierarchy.py`, and
#: `taniteval/v0_antiecho.py` (46,905 B) which was absent from the box entirely.
#: Kept only so a prior audit can be reproduced exactly (``--entry-mode ladder``).
LEGACY_LADDER_ENTRIES = (
    "stack/scripts/train_v6_staged.py",     # the trainer
    "stack/scripts/v6_chain.py",            # the ladder launcher / adjudicator
    "stack/scripts/v6_dump_sw_latents.py",  # S-T step 1 (the S-W latent dump)
    "stack/scripts/e_wc2_sigma_star.py",    # S-T step 2 (sigma*)
    "stack/scripts/run_gate.py",            # the gate battery
    "stack/scripts/gate_emitters.py",
    "stack/scripts/watch_gates.py",
)

#: The measured 14 of C105 — the widened hand-list. This is the **fallback**
#: when derivation is switched off (``--entry-mode fixed``), and the floor the
#: derivation is asserted against: a derivation that reaches fewer files than
#: this is a regression, not a simplification.
DEFAULT_ENTRIES = LEGACY_LADDER_ENTRIES + (
    "taniteval/tools/t1_eval.py",
    "taniteval/tools/eval_four_families.py",
    "taniteval/tools/seam_probe.py",
    "taniteval/tools/t1_summary.py",
    "stack/scripts/run_spectral.py",
    "stack/scripts/refc_dump_latents.py",
    "stack/scripts/v5_guard.py",
)

#: ⭐ **THE HAND-LIST THAT REPLACES THE HAND-LIST — and it is a different KIND of
#: list, which is the whole point.** These are **LAUNCH SOURCES**: files that
#: *emit or document* launch command lines. The entry points are then read out of
#: them (see :func:`derive_entries`) instead of being remembered.
#:
#: C99 fixed a hand-listed **ship set** and kept a hand-listed **entry-point
#: set**; C105 measured the cost. A list of launch *sources* changes when the
#: launch **mechanism** changes — a new runbook, a new chain — which is rare and
#: loud. The entry-point list changed every time an instrument joined the ladder,
#: which is frequent and silent. ⚠️ **The recursion has to bottom out somewhere
#: and this is where; the tool therefore PRINTS this set on every run**, so the
#: assumption is visible rather than buried at line 109 of a script.
DEFAULT_LAUNCH_SOURCES = (
    # the ladder launcher — every S-* step's command line is emitted from here
    "stack/scripts/v6_chain.py",
    # the operator's 3 a.m. document; §2's launch lines are pinned against
    # `v6_chain.py commands` by stack/tests/test_runbook_commands.py
    "TanitAD Research Hub/Architecture & Inference/Implementation/incoming/"
    "2026-08-07-hierarchical-wm-redesign/V6_GO_PACKAGE.md",
    # the gate battery runbook — where run_gate.py / gate_emitters.py come from
    "Project Steering/GATE_PROTOCOL.md",
)

#: A launch source names its scripts in prose, in f-strings, and in shell blocks,
#: so the scan is over **text**, not over an AST — an AST cannot see
#: ``f"{cfg.python} scripts/v6_dump_sw_latents.py "`` split across a line, nor a
#: markdown code fence at all.
_PY_TOKEN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./+-]*\.py\b")

#: Tests are executed by pytest, not by a launch line, and the suite already
#: guards them. Including them would drag every fixture into the ship set.
_ENTRY_EXCLUDE = ("__pycache__/", "/tests/", "tests/", "/.git/", ".claude/")

# --------------------------------------------------------------------------
# entry-point DERIVATION — the C105 fix
# --------------------------------------------------------------------------


@dataclass
class EntrySet:
    """A derived set of entry points, with the provenance of every member."""

    entries: list[str] = field(default_factory=list)          # repo-relative
    sources: list[str] = field(default_factory=list)          # launch sources
    #: entry -> the launch sources that named it. This is the audit trail: an
    #: entry point nobody can explain is an entry point nobody should trust.
    named_by: dict[str, list[str]] = field(default_factory=dict)
    #: tokens that looked like a script and resolved to nothing. Usually prose
    #: (``foo.py`` in a sentence) or a pod-only path; recorded, never guessed at.
    unresolved: list[str] = field(default_factory=list)
    #: a bare basename that matched more than one file. ⚠️ **Never silently
    #: pick one** — C105's locator returned "something plausible" three times.
    #: All matches are taken (over-coverage is cheap in a drift audit) and the
    #: ambiguity is reported.
    ambiguous: dict[str, list[str]] = field(default_factory=dict)
    #: ⚠️ Entry points present ONLY because the floor (:data:`DEFAULT_ENTRIES`)
    #: carries them — **no current launch source names these.** They are audited
    #: on inherited authority, not on a derived launch path, and saying so is the
    #: point: this is the residue of the hand-list, made visible instead of
    #: pretending the derivation is complete.
    from_floor: list[str] = field(default_factory=list)
    mode: str = "derived"


def _rel_to_repo(p: Path, repo: Path) -> str:
    return str(p.relative_to(repo)).replace("\\", "/")


def _excluded(rel: str) -> bool:
    return any(part in rel for part in _ENTRY_EXCLUDE) or \
        Path(rel).name.startswith("test_")


def _resolve_script(token: str, repo: Path, entry_roots: list[Path]) -> list[Path]:
    """Every repo file a launch-line token could name.

    Tokens arrive in four shapes and all four are real, measured in the runbook
    and the chain on 2026-08-18::

        stack/scripts/v6_chain.py              repo-relative
        scripts/train_v6_staged.py             relative to the launch `cd`
        /root/TanitAD/stack/scripts/v6_chain.py    an absolute POD path
        t1_eval.py                             a bare basename in prose

    So every **path suffix** of the token is tried against the repo root and
    against each entry root. A bare basename is deliberately allowed to match
    inside an entry root — that is how ``t1_eval.py`` reaches
    ``taniteval/tools/t1_eval.py`` — and it is the only shape that can be
    ambiguous, which the caller records.
    """
    parts = [p for p in token.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    hits: list[Path] = []
    seen: set[Path] = set()
    for i in range(len(parts)):
        suffix = "/".join(parts[i:])
        for base in [repo, *entry_roots]:
            cand = base / suffix
            try:
                resolved = cand.resolve()
            except OSError:
                continue
            if not cand.is_file():
                continue
            try:
                rel = _rel_to_repo(resolved, repo)
            except ValueError:
                continue                      # outside the repo — not ours
            if _excluded(rel) or resolved in seen:
                continue
            seen.add(resolved)
            hits.append(resolved)
    return hits


def _under_entry_root(path: Path, entry_roots: list[Path]) -> bool:
    return any(str(path).startswith(str(r)) for r in entry_roots)


def derive_entries(sources: list[Path], repo: Path, entry_roots: list[Path],
                   floor: tuple[str, ...] | list[str] = ()) -> EntrySet:
    """Read the entry points out of what the runbook and the chain INVOKE.

    ⛔ **This function is C105's fix.** A closure is only as complete as the set
    it is closed over, and the entry set had been a hand-list — so the audit
    inherited exactly the defect it was built to kill, one level up.

    The walk is a **fixed point, not a single pass**: a script named by a launch
    source is itself a launch source, because it may shell out further. That
    matters concretely — ``train_v6_staged.py`` subprocesses
    ``taniteval/tools/eval_four_families.py``, ``seam_probe.py`` and
    ``t1_eval.py``, none of which any *import* walk can see (they are argv, not
    imports), and all three were on C105's stale list.

    ⚠️ **Scope, stated because it is the assumption that remains**: this reads
    text for ``*.py`` tokens. A script invoked through a name assembled at
    runtime, or through a shell script this never reaches, is invisible — the
    same blind spot the AST walk has for ``import_module(f"{x}")``. The
    ``uncovered_executables`` report exists so that gap is measured rather than
    assumed away.

    ⚠️ **And it is a superset of what the sources INVOKE, not an equal.** The
    scan reads text, so a script merely *mentioned* in a usage docstring joins
    the set. That is deliberate: in a drift audit over-coverage costs a few more
    md5s, and under-coverage is what cost C99 and C105.

    ⛔ **``floor`` exists because a derivation must WIDEN, never NARROW.**
    MEASURED 2026-08-18: the derivation reaches 44 entry points but **not**
    ``watch_gates.py``, ``t1_summary.py``, ``run_spectral.py`` or ``v5_guard.py``
    — four instruments C105's widened hand-list carried that **no current launch
    source names** (``t1_summary.py`` is invoked from a *shell* chain,
    ``run_spectral.py`` only from ``run_orthogonality.py``). Silently dropping
    them would be this tool regressing coverage while looking more principled.
    They are kept and **flagged** in :attr:`EntrySet.from_floor`.
    """
    queue = list(dict.fromkeys(p.resolve() for p in sources if p.is_file()))
    scanned: set[Path] = set()
    entries: dict[Path, set[str]] = {}
    unresolved: set[str] = set()
    ambiguous: dict[str, list[str]] = {}
    source_rels = [_rel_to_repo(p, repo) for p in queue]

    while queue:
        src = queue.pop()
        if src in scanned:
            continue
        scanned.add(src)
        try:
            text = src.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        src_rel = _rel_to_repo(src, repo)
        for token in dict.fromkeys(_PY_TOKEN.findall(text)):
            hits = _resolve_script(token, repo, entry_roots)
            if not hits:
                unresolved.add(token)
                continue
            if len(hits) > 1 and "/" not in token:
                ambiguous[token] = sorted(_rel_to_repo(h, repo) for h in hits)
            for hit in hits:
                if not _under_entry_root(hit, entry_roots):
                    continue                  # a library module, not a launch
                entries.setdefault(hit, set()).add(src_rel)
                if hit not in scanned:
                    queue.append(hit)         # a script can launch a script

    derived = {_rel_to_repo(p, repo) for p in entries}
    from_floor = sorted(f for f in floor
                        if f not in derived and (repo / f).is_file())
    return EntrySet(
        entries=sorted(derived | set(from_floor)),
        sources=sorted(source_rels),
        named_by={_rel_to_repo(k, repo): sorted(v) for k, v in entries.items()},
        unresolved=sorted(unresolved),
        ambiguous=ambiguous,
        from_floor=from_floor,
        mode="derived",
    )


def executables_under(entry_roots: list[Path], repo: Path) -> list[str]:
    """Every ``__main__``-guarded script in the executable directories.

    This is the **denominator**. The closure covers what the derivation reached;
    this says what could be launched at all, so the difference is a measured
    coverage gap instead of an unexamined assumption. *(Absence found at one
    location is not absence — and neither is coverage.)*
    """
    out: list[str] = []
    for root in entry_roots:
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*.py")):
            rel = _rel_to_repo(p.resolve(), repo)
            if _excluded(rel):
                continue
            try:
                if "__main__" in p.read_text(encoding="utf-8", errors="replace"):
                    out.append(rel)
            except OSError:
                continue
    return sorted(set(out))


# --------------------------------------------------------------------------
# import-closure walk (AST)
# --------------------------------------------------------------------------


@dataclass
class Closure:
    """Result of an import-closure walk."""

    files: list[str] = field(default_factory=list)      # repo-relative, sorted
    #: reached by module-level imports only — these run at process START, so a
    #: problem here fails fast and loud.
    eager: list[str] = field(default_factory=list)
    #: reachable ONLY through a function-level import. **This is the C99 risk
    #: set**: invisible to any startup check, fails when the path fires.
    deferred_only: list[str] = field(default_factory=list)
    #: EVERY import site of these is inside a ``try/except ImportError``. A
    #: failed import here is TOLERATED by design, not a launch blocker.
    guarded_only: list[str] = field(default_factory=list)
    entries: list[str] = field(default_factory=list)
    external: list[str] = field(default_factory=list)   # stdlib / site-packages
    unparsed: list[str] = field(default_factory=list)   # syntax errors
    #: file -> the module names it pulled into the closure (for provenance)
    why: dict[str, list[str]] = field(default_factory=dict)


def _module_candidates(mod: str, roots: list[Path]) -> list[Path]:
    """Every on-disk file a dotted module name could resolve to."""
    rel = mod.replace(".", "/")
    out: list[Path] = []
    for root in roots:
        out.append(root / f"{rel}.py")
        out.append(root / rel / "__init__.py")
    return out


def _resolve(mod: str, roots: list[Path]) -> Path | None:
    for cand in _module_candidates(mod, roots):
        if cand.is_file():
            return cand
    return None


def _package_inits(mod: str, roots: list[Path]) -> list[Path]:
    """``tanitad.models.v6`` also imports ``tanitad`` and ``tanitad.models``."""
    parts = mod.split(".")
    out: list[Path] = []
    for i in range(1, len(parts)):
        got = _resolve(".".join(parts[:i]), roots)
        if got is not None:
            out.append(got)
    return out


def _module_name_of(path: Path, roots: list[Path]) -> str | None:
    """Inverse of ``_resolve`` — the dotted name a file is imported as."""
    for root in roots:
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        parts = list(rel.parts)
        if parts[-1] == "__init__.py":
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1][: -len(".py")]
        return ".".join(parts)
    return None


_IMPORT_EXC = {"ImportError", "ModuleNotFoundError", "Exception", "BaseException"}


def _handler_catches_import(h: ast.ExceptHandler) -> bool:
    """Does this ``except`` clause swallow a failed import?"""
    if h.type is None:                       # bare `except:`
        return True
    nodes = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
    for n in nodes:
        name = getattr(n, "id", None) or getattr(n, "attr", None)
        if name in _IMPORT_EXC:
            return True
    return False


def _imports_in(tree: ast.AST) -> list[tuple[str, bool, bool]]:
    """Every module name this file imports, **including function-level ones**.

    Function-level imports are not optional to collect: ``train_v6_staged.py``
    defers ``eval_flagship_v4``, ``train_flagship4b``, ``s2_labels`` and
    ``taniteval.seam_dump`` into function bodies, and ``v6_dump_sw_latents.py``
    defers ``driving_diagnostic`` and ``e_wc2_sigma_star``. A module-level-only
    walk misses the majority of this program's real closure.

    ``importlib.import_module("literal")`` is picked up too — the ladder uses it
    to reach the trainer.

    Returns ``(module_name, deferred, guarded)``.

    ``guarded`` says the import sits in a ``try:`` body whose ``except`` catches
    ImportError — i.e. the code is DESIGNED to run without it. ⚠️ This flag is
    what stops the audit manufacturing a false blocker: a direct
    ``import_module()`` probe bypasses the guard and reports a failure the launch
    would never see. MEASURED 2026-08-18: ``lead_state_gate`` fails to import on
    Thor (no pandas), yet ``probe_latent_state.py:117`` wraps it in exactly such
    a guard with documented fallback constants — reporting that as a blocker
    would have been a fabricated finding.

    **``deferred`` is the risk axis**: an
    import nested inside a function body does not run at process start, so a
    stale or absent dependency behind one is invisible to every startup check and
    surfaces only when that code path fires — after the compute is paid for. That
    is C99's shape (``anchor_goal`` behind ``cfg.anchor_goal != "none"``) and the
    ``t1_eval.py`` shape (both arms, 40 episodes rolled, then ``ImportError`` in
    ``analyze()``).
    """
    names: list[tuple[str, bool, bool]] = []
    deferred_nodes: set[int] = set()
    guarded_nodes: set[int] = set()
    for parent in ast.walk(tree):
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(parent):
                deferred_nodes.add(id(child))
        elif isinstance(parent, ast.Try):
            catches_import = any(
                _handler_catches_import(h) for h in parent.handlers)
            if catches_import:
                # Only `try:` BODY is protected — `else:`/`finally:` are not.
                for stmt in parent.body:
                    for child in ast.walk(stmt):
                        guarded_nodes.add(id(child))

    def _add(name: str, node: ast.AST) -> None:
        names.append((name, id(node) in deferred_nodes,
                      id(node) in guarded_nodes))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                _add(a.name, node)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                       # relative — handled by caller
                base = "." * node.level + (node.module or "")
                _add(base, node)
                for a in node.names:
                    _add(f"{base}.{a.name}" if node.module else base + a.name, node)
            elif node.module:
                _add(node.module, node)
                # ``from tanitad.models import v6`` — v6 may itself be a module.
                for a in node.names:
                    _add(f"{node.module}.{a.name}", node)
        elif isinstance(node, ast.Call):
            fn = node.func
            dotted = getattr(fn, "attr", None) or getattr(fn, "id", None)
            if dotted == "import_module" and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    _add(arg.value, node)
    return names


def _abs_from_relative(rel_name: str, owner: Path, roots: list[Path]) -> str | None:
    """Resolve ``from . import x`` / ``from ..y import z`` against the owner."""
    level = len(rel_name) - len(rel_name.lstrip("."))
    tail = rel_name[level:]
    owner_mod = _module_name_of(owner, roots)
    if owner_mod is None:
        return None
    parts = owner_mod.split(".")
    if owner.name != "__init__.py":
        parts = parts[:-1]                       # a module's package is its parent
    if level > 1:
        parts = parts[: -(level - 1)] if level - 1 <= len(parts) else []
    base = ".".join(parts)
    if not base:
        return tail or None
    return f"{base}.{tail}" if tail else base


def compute_closure(entries: list[Path], roots: list[Path]) -> Closure:
    """Transitively walk imports from ``entries``, resolving inside ``roots``.

    Anything that does not resolve to a file under a root is **external**
    (stdlib, torch, numpy, …) and is recorded but not walked.
    """
    external: set[str] = set()
    unparsed: list[str] = []
    why: dict[str, set[str]] = {}
    eager: set[Path] = set()
    seen: set[Path] = set()
    unguarded: set[Path] = set()
    parsed: dict[Path, ast.AST] = {}

    # (path, is_eager). Entry points are eager by definition.
    queue: list[tuple[Path, bool]] = [(p.resolve(), True)
                                      for p in entries if p.is_file()]

    while queue:
        path, is_eager = queue.pop()
        # A file already seen may still need re-walking if it has just been
        # UPGRADED to eager — its module-level imports are eager too.
        if path in seen and not (is_eager and path not in eager):
            continue
        seen.add(path)
        if is_eager:
            eager.add(path)
        if path not in parsed:
            try:
                parsed[path] = ast.parse(path.read_bytes(), filename=str(path))
            except SyntaxError:
                unparsed.append(str(path))
                parsed[path] = ast.parse("")
        for name, deferred, guarded in _imports_in(parsed[path]):
            if name.startswith("."):
                resolved_name = _abs_from_relative(name, path, roots)
                if resolved_name is None:
                    continue
                name = resolved_name
            target = _resolve(name, roots)
            extra = _package_inits(name, roots)
            if target is None and not extra:
                external.add(name.split(".")[0])
                continue
            if not guarded:
                for hit in ([target] if target else []) + extra:
                    unguarded.add(hit)
            # Only a module-level import inside an eagerly-imported file runs at
            # process start; everything else is conditional.
            child_eager = is_eager and not deferred
            for hit in ([target] if target else []) + extra:
                why.setdefault(str(hit), set()).add(name)
                if hit not in seen or (child_eager and hit not in eager):
                    queue.append((hit, child_eager))

    def _rel(p: Path) -> str:
        return str(p.relative_to(_REPO)).replace("\\", "/")

    return Closure(
        files=sorted(_rel(p) for p in seen),
        eager=sorted(_rel(p) for p in eager),
        deferred_only=sorted(_rel(p) for p in seen - eager),
        guarded_only=sorted(_rel(p) for p in seen - unguarded),
        entries=sorted(_rel(p.resolve()) for p in entries if p.is_file()),
        external=sorted(external),
        unparsed=unparsed,
        why={_rel(Path(k)): sorted(v) for k, v in why.items()},
    )


# --------------------------------------------------------------------------
# hashing — raw AND LF-normalised, always both
# --------------------------------------------------------------------------


def _digests(data: bytes) -> tuple[str, str, int]:
    raw = hashlib.md5(data).hexdigest()
    lf = hashlib.md5(data.replace(b"\r\n", b"\n")).hexdigest()
    return raw, lf, len(data)


def local_hashes(rels: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rel in rels:
        p = _REPO / rel
        if not p.is_file():
            out[rel] = {"present": False}
            continue
        raw, lf, n = _digests(p.read_bytes())
        out[rel] = {"present": True, "md5_raw": raw, "md5_lf": lf, "bytes": n}
    return out


# --------------------------------------------------------------------------
# remote side — one `ssh -n`, one opaque ZZ…ZZ frame
# --------------------------------------------------------------------------

_REMOTE_HASH_PY = r'''
import base64, hashlib, json, os, sys
spec = json.load(open(sys.argv[1]))
root, out = spec["root"], {}
for rel in spec["paths"]:
    p = os.path.join(root, rel)
    if not os.path.isfile(p):
        out[rel] = {"present": False}
        continue
    b = open(p, "rb").read()
    out[rel] = {"present": True,
                "md5_raw": hashlib.md5(b).hexdigest(),
                "md5_lf": hashlib.md5(b.replace(b"\r\n", b"\n")).hexdigest(),
                "bytes": len(b)}
print("ZZ" + base64.b64encode(json.dumps(out).encode()).decode() + "ZZ")
'''


#: Remote-side AST symbol census — answers *what* differs, not merely *that* it
#: does. C99's stale file was 2.6× small and missing ``K_MAX_GRID``; a size and a
#: symbol list say that in one line, where two md5s only say "different".
_REMOTE_SYMS_PY = r'''
import ast, base64, json, os, sys
spec = json.load(open(sys.argv[1]))
root, out = spec["root"], {}
for rel in spec["paths"]:
    p = os.path.join(root, rel)
    if not os.path.isfile(p):
        out[rel] = None
        continue
    try:
        tree = ast.parse(open(p, "rb").read(), filename=p)
    except SyntaxError as e:
        out[rel] = {"syntax_error": str(e)[:200]}
        continue
    syms = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            syms.append(n.name)
        elif isinstance(n, ast.Assign):
            syms.extend(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            syms.append(n.target.id)
    out[rel] = {"symbols": sorted(set(syms)), "lines": len(open(p, "rb").read().splitlines())}
print("ZZ" + base64.b64encode(json.dumps(out).encode()).decode() + "ZZ")
'''


def local_symbols(rel: str) -> dict | None:
    """Top-level symbol census of a repo file (same shape as the remote one)."""
    p = _REPO / rel
    if not p.is_file():
        return None
    data = p.read_bytes()
    try:
        tree = ast.parse(data, filename=str(p))
    except SyntaxError as e:
        return {"syntax_error": str(e)[:200]}
    syms: list[str] = []
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            syms.append(n.name)
        elif isinstance(n, ast.Assign):
            syms.extend(t.id for t in n.targets if isinstance(t, ast.Name))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            syms.append(n.target.id)
    return {"symbols": sorted(set(syms)), "lines": len(data.splitlines())}


def _demangle_posix(value: str | None) -> str | None:
    """Undo Git-Bash/MSYS argument path conversion.

    ⛔ **MEASURED 2026-08-18, and it produced a 100 %-wrong audit.** Invoked from
    Git Bash on Windows, ``--remote-root /home/nvidia/TanitAD`` reaches Python as
    ``C:/Program Files/Git/home/nvidia/TanitAD`` — MSYS rewrites POSIX-looking
    argument paths. Every remote ``isfile`` then returns False and the tool
    reports **120/120 MISSING_REMOTE**: a clean, plausible, catastrophic-looking
    finding that is pure artifact. Same family as ``df`` on a pod — a probe
    answering a different question than the one asked.

    ``MSYS_NO_PATHCONV=1`` prevents it; this strips it if it already happened, so
    the tool cannot silently produce that answer again.
    """
    if not value or value.startswith("/"):
        return value
    for marker in ("/home/", "/root/", "/workspace/", "/mnt/", "/opt/", "/tmp/"):
        idx = value.find(marker)
        if idx > 0 and (value[1:3] in (":/", ":\\")):
            fixed = value[idx:]
            print(f"  !! MSYS path conversion detected: {value!r} -> {fixed!r} "
                  f"(set MSYS_NO_PATHCONV=1 to prevent it)", file=sys.stderr)
            return fixed
    return value


def _ssh(host: str, remote_cmd: str, ssh_bin: str, timeout: int = 180) -> str:
    """Run one remote command. ``-n`` is MANDATORY (stdin-eating trap)."""
    cmd = [ssh_bin, "-n", "-o", "ConnectTimeout=20", "-o", "BatchMode=yes",
           host, remote_cmd]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"ssh failed rc={res.returncode}: {res.stderr[-800:]}")
    return res.stdout


def _unframe(stream: str) -> str:
    """Pull the ``ZZ…ZZ`` payload out. Never grep the raw stream for our words."""
    for line in stream.splitlines():
        line = line.strip()
        if len(line) > 4 and line.startswith("ZZ") and line.endswith("ZZ"):
            return line[2:-2]
    raise RuntimeError(f"no ZZ frame in remote output: {stream[-800:]!r}")


#: Windows ``CreateProcess`` caps a whole command line at 32,767 chars, and it
#: fails with a bare ``FileNotFoundError [WinError 206]`` that names no cause.
#: MEASURED 2026-08-18: a 4-file ship (~290 KB → ~390 KB base64) died there. So
#: **nothing large ever travels on the command line** — payloads go through this
#: chunked pusher, which also keeps ``ssh -n`` on every single call.
_CHUNK = 12000


def _push_b64(host: str, remote_path: str, data: bytes, ssh_bin: str) -> int:
    """Write ``data`` to ``remote_path`` as base64, in command-line-safe chunks.

    base64's alphabet is ``A-Za-z0-9+/=`` — no shell metacharacters, no leading
    ``-`` — so the chunks need no quoting. ``printf %s <arg>`` puts the payload
    in the ARGUMENT position, never the format string: a literal ``(77%)`` once
    truncated a commit message because it landed in the format position.
    """
    b64 = base64.b64encode(data).decode()
    for i, off in enumerate(range(0, len(b64), _CHUNK)):
        part = b64[off:off + _CHUNK]
        op = ">" if i == 0 else ">>"
        _ssh(host, f"printf %s {part} {op} {remote_path}", ssh_bin)
    return len(b64)


def _stage_remote(host: str, prog: str, spec_obj: dict, ssh_bin: str,
                  tag: str) -> tuple[str, str]:
    """Put a remote program and its (possibly large) JSON spec on the box."""
    prog_path, spec_path = f"/tmp/_closure_{tag}.py", f"/tmp/_closure_{tag}.json"
    _ssh(host, "mkdir -p /tmp", ssh_bin)
    _push_b64(host, prog_path + ".b64", prog.encode(), ssh_bin)
    _push_b64(host, spec_path + ".b64", json.dumps(spec_obj).encode(), ssh_bin)
    _ssh(host, f"base64 -d < {prog_path}.b64 > {prog_path} && "
               f"base64 -d < {spec_path}.b64 > {spec_path}", ssh_bin)
    return prog_path, spec_path


def remote_hashes(host: str, root: str, rels: list[str], ssh_bin: str,
                  py: str) -> dict[str, dict]:
    prog, spec = _stage_remote(host, _REMOTE_HASH_PY,
                               {"root": root, "paths": rels}, ssh_bin, "hash")
    out = _ssh(host, f"{py} {prog} {spec}", ssh_bin)
    return json.loads(base64.b64decode(_unframe(out)))


def remote_symbols(host: str, root: str, rels: list[str], ssh_bin: str,
                   py: str) -> dict:
    prog, spec = _stage_remote(host, _REMOTE_SYMS_PY,
                               {"root": root, "paths": rels}, ssh_bin, "syms")
    out = _ssh(host, f"{py} {prog} {spec}", ssh_bin)
    return json.loads(base64.b64decode(_unframe(out)))


def explain_drift(host: str, root: str, rows: list[dict], ssh_bin: str,
                  py: str) -> dict:
    """For every DRIFT row, name the symbols the box is MISSING."""
    drifted = [r["path"] for r in rows if r["verdict"] == "DRIFT"]
    if not drifted:
        return {}
    rsyms = remote_symbols(host, root, drifted, ssh_bin, py)
    out = {}
    for rel in drifted:
        loc, rem = local_symbols(rel), rsyms.get(rel)
        if not loc or not rem or "symbols" not in loc or "symbols" not in rem:
            out[rel] = {"local": loc, "remote": rem}
            continue
        out[rel] = {
            "missing_on_remote": sorted(set(loc["symbols"]) - set(rem["symbols"])),
            "extra_on_remote": sorted(set(rem["symbols"]) - set(loc["symbols"])),
            "local_lines": loc["lines"], "remote_lines": rem["lines"],
            "n_local_symbols": len(loc["symbols"]),
            "n_remote_symbols": len(rem["symbols"]),
        }
    return out


# --------------------------------------------------------------------------
# verdicts
# --------------------------------------------------------------------------

VERDICTS = ("SAME", "CRLF_ONLY", "DRIFT", "MISSING_REMOTE", "MISSING_LOCAL")


def verdict(loc: dict, rem: dict) -> str:
    if not loc.get("present"):
        return "MISSING_LOCAL"
    if not rem.get("present"):
        return "MISSING_REMOTE"
    if loc["md5_raw"] == rem["md5_raw"]:
        return "SAME"
    if loc["md5_lf"] == rem["md5_lf"]:
        return "CRLF_ONLY"
    return "DRIFT"


def head_lf_digests(rels: list[str]) -> dict[str, str]:
    """LF-normalised md5 of each path **as committed at HEAD**.

    ⛔ **MEASURED 2026-08-18, and it manufactured a DRIFT row on the model
    itself.** This tool hashes the **WORKING TREE**, which is correct for
    shipping (you ship what you have) and wrong for *attribution*: with several
    agents live — this programme's normal state — a sibling's **uncommitted**
    edit makes the local file differ from the box and reads as *the box is
    stale*. ``stack/tanitad/models/v6.py`` came back DRIFT while Thor's copy was
    **byte-identical to HEAD**; the dev box was simply ahead by an unstaged
    ``FROZEN_EXTERNAL_*`` change. Reporting that as remote drift would have been
    a fabricated finding, and shipping "the fix" would have pushed a sibling's
    half-finished work onto a box running a 5-day job.

    ⇒ Every DRIFT row is now annotated ``local_dirty_vs_head``, so *"the box is
    behind"* and *"my tree is ahead"* can never again be the same row.
    """
    out: dict[str, str] = {}
    for rel in rels:
        try:
            res = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=str(_REPO),
                                 capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            continue
        if res.returncode == 0:
            out[rel] = hashlib.md5(res.stdout.replace(b"\r\n", b"\n")).hexdigest()
    return out


def build_table(closure: Closure, loc: dict, rem: dict,
                head: dict[str, str] | None = None) -> list[dict]:
    head = head or {}
    rows = []
    for rel in closure.files:
        l, r = loc.get(rel, {}), rem.get(rel, {})
        v = verdict(l, r)
        row = {"path": rel, "verdict": v,
               "reach": "eager" if rel in set(closure.eager) else "deferred_only",
               "local_bytes": l.get("bytes"), "remote_bytes": r.get("bytes"),
               "local_md5_lf": l.get("md5_lf"), "remote_md5_lf": r.get("md5_lf")}
        if rel in head:
            row["head_md5_lf"] = head[rel]
            row["local_dirty_vs_head"] = l.get("md5_lf") != head[rel]
            row["remote_matches_head"] = r.get("md5_lf") == head[rel]
        if v == "DRIFT" and l.get("bytes") and r.get("bytes"):
            row["size_ratio_local_over_remote"] = round(l["bytes"] / r["bytes"], 3)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# ship — backup first, LF-normalised bytes, md5-verified on both sides
# --------------------------------------------------------------------------

_REMOTE_PUSH_PY = r'''
import base64, hashlib, json, os, shutil, sys
spec = json.load(open(sys.argv[1]))
root, backup, stage, out = spec["root"], spec["backup"], spec["stage"], {}
for idx, rel in spec["files"].items():
    dest = os.path.join(root, rel)
    rec = {}
    # BACK UP FIRST -- always, before a single byte of dest is touched.
    if os.path.isfile(dest):
        os.makedirs(os.path.join(backup, os.path.dirname(rel)), exist_ok=True)
        shutil.copy2(dest, os.path.join(backup, rel))
        rec["backed_up_md5"] = hashlib.md5(open(dest, "rb").read()).hexdigest()
        rec["backed_up_bytes"] = os.path.getsize(dest)
        rec["backed_up_to"] = os.path.join(backup, rel)
    else:
        rec["backed_up_md5"] = None
        rec["backed_up_to"] = None
    data = base64.b64decode(open(os.path.join(stage, idx + ".b64"), "rb").read())
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".closure_tmp"
    open(tmp, "wb").write(data)
    os.replace(tmp, dest)                       # atomic; never a torn file
    got = open(dest, "rb").read()
    rec["md5_after"] = hashlib.md5(got).hexdigest()
    rec["bytes_after"] = len(got)
    out[rel] = rec
print("ZZ" + base64.b64encode(json.dumps(out).encode()).decode() + "ZZ")
'''


def ship(host: str, root: str, rels: list[str], backup: str, ssh_bin: str,
         py: str) -> dict:
    """Push LF-normalised copies of ``rels``, backing up the box's originals.

    ⚠️ The digest the caller must verify against is ``md5_lf``, **not**
    ``md5_raw``: we deliberately send LF bytes to a Linux target, so the file
    that lands equals the repo file with ``\\r\\n`` collapsed. Verifying against
    ``md5_raw`` would report a false failure on every single file.
    """
    stage = "/tmp/_closure_stage"
    _ssh(host, f"rm -rf {stage} && mkdir -p {stage}", ssh_bin)
    manifest = {}
    for i, rel in enumerate(rels):
        data = (_REPO / rel).read_bytes().replace(b"\r\n", b"\n")
        _push_b64(host, f"{stage}/{i}.b64", data, ssh_bin)
        manifest[str(i)] = rel
    prog, spec = _stage_remote(
        host, _REMOTE_PUSH_PY,
        {"root": root, "backup": backup, "stage": stage, "files": manifest},
        ssh_bin, "push")
    out = _ssh(host, f"{py} {prog} {spec}", ssh_bin, 600)
    return json.loads(base64.b64decode(_unframe(out)))


# --------------------------------------------------------------------------
# the only sufficient evidence: a real import
# --------------------------------------------------------------------------


def verify_import(host: str, root: str, mods: list[str], ssh_bin: str,
                  py: str) -> dict:
    """Import every closure module for real, on the box, with PYTHONPATH set.

    ⚠️ This is the step md5 cannot replace. C99's three files were present and
    byte-perfect and the launch still could not run — the failure only surfaced
    when something actually *imported*.
    """
    prog = ("import importlib, json, sys\n"
            "mods = json.load(open(sys.argv[1]))['mods']\n"
            "ok, bad = [], {}\n"
            "for m in mods:\n"
            "    try:\n"
            "        importlib.import_module(m); ok.append(m)\n"
            "    except BaseException as e:\n"
            "        bad[m] = f'{type(e).__name__}: {e}'[:300]\n"
            "print('ZZ' + json.dumps({'n_ok': len(ok), 'n_bad': len(bad),"
            " 'bad': bad}) + 'ZZ')\n")
    prog_path, spec_path = _stage_remote(host, prog, {"mods": mods}, ssh_bin, "imp")
    # ⚠️ OMP_NUM_THREADS is not cosmetic: torch spawns ~113 threads PER PROCESS,
    # and this probe runs beside a live trainer. Keep the footprint small.
    # ⚠️ `py` must be the venv interpreter that owns torch — system python3 will
    # report a fake ImportError wall for every module that touches torch.
    remote = (f"cd {root}/stack && OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "
              f"PYTHONPATH={root}/stack:{root}/stack/scripts:{root}/taniteval "
              f"{py} {prog_path} {spec_path}")
    return json.loads(_unframe(_ssh(host, remote, ssh_bin, 900)))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def resolve_entry_set(args, entry_roots: list[Path]) -> EntrySet:
    """Turn ``--entry`` / ``--entry-mode`` into the concrete root set."""
    if args.entry:
        return EntrySet(entries=sorted(args.entry), mode="explicit")
    if args.entry_mode == "derived":
        eset = derive_entries([_REPO / s for s in args.launch_source],
                              _REPO, entry_roots, floor=DEFAULT_ENTRIES)
        missing_src = [s for s in args.launch_source if not (_REPO / s).is_file()]
        if missing_src:
            # ⛔ A launch source that has MOVED silently narrows the whole audit
            # — the exact failure shape this tool exists to kill. Never quiet.
            print(f"  !! launch sources not found (the audit is NARROWER than it "
                  f"looks): {missing_src}", file=sys.stderr)
        return eset
    if args.entry_mode == "ladder":
        return EntrySet(entries=sorted(LEGACY_LADDER_ENTRIES), mode="ladder")
    if args.entry_mode == "executable":
        return EntrySet(entries=executables_under(entry_roots, _REPO),
                        mode="executable")
    return EntrySet(entries=sorted(DEFAULT_ENTRIES), mode="fixed")


def print_root_set(eset: EntrySet, args, closure: Closure,
                   launchable: list[str], uncovered: list[str]) -> None:
    """⭐ Print the assumption the whole audit rests on, every single run.

    C105's finding was not the seven stale files — it was that the briefed
    entry-point set was itself a guess, and that the guess was **invisible**
    because it lived as a tuple at the top of a script. A root set nobody sees
    is a root set nobody checks.
    """
    bar = "=" * 74
    print(bar)
    print(f"ROOT SET  (mode={eset.mode})  — the assumption this audit rests on")
    print(bar)
    if eset.mode == "derived":
        print(f"launch sources ({len(eset.sources)}) — entry points are READ "
              f"from these, not remembered:")
        for s in eset.sources:
            print(f"    {s}")
    elif eset.mode == "explicit":
        print("  --entry given explicitly: derivation SKIPPED. Anything the "
              "launch touches that you did not list is unaudited.")
    elif eset.mode == "ladder":
        print("  C105's original 7-entry hand-list, for reproducing that audit "
              "ONLY. It missed 3 stale files; do not decide on it.")
    print(f"\nentry points ({len(eset.entries)}):")
    for e in eset.entries:
        why = eset.named_by.get(e)
        if why:
            tail = f"   <- {', '.join(Path(w).name for w in why)}"
        elif e in eset.from_floor:
            tail = "   <- FLOOR ONLY (no launch source names it)"
        else:
            tail = ""
        print(f"    {e}{tail}")
    if eset.from_floor:
        print(f"\n  ⚠️ {len(eset.from_floor)} entry point(s) survive only because "
              f"the C105 floor carries them — no current launch source names "
              f"them, so they are audited on INHERITED authority: "
              f"{eset.from_floor}")
    if eset.ambiguous:
        print(f"\n  ~ ambiguous bare basenames (ALL matches taken, none "
              f"silently chosen): {eset.ambiguous}")
    print(f"\nsys.path roots: {args.roots}")
    print(f"executable dirs: {args.entry_roots}")
    print(f"\nCOVERAGE: closure {len(closure.files)} files · "
          f"{len(launchable) - len(uncovered)}/{len(launchable)} launchable "
          f"scripts covered · {len(uncovered)} NOT covered")
    if uncovered:
        print("  ⚠️ uncovered scripts are launchable and unaudited by this run. "
              "They are not on the derived launch path; if one becomes so, add "
              "its launch source." + ("" if args.show_uncovered
                                      else "  (--show-uncovered to list)"))
        if args.show_uncovered:
            for u in uncovered:
                print(f"    UNCOVERED  {u}")
    if eset.unresolved:
        print(f"  ~ {len(eset.unresolved)} script-shaped tokens resolved to "
              f"nothing (prose / pod-only paths); not guessed at.")
    print(bar)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--entry", nargs="*", default=None,
                    help="explicit repo-relative entry points; overrides "
                         "--entry-mode entirely (and prints that it did)")
    ap.add_argument("--entry-mode", default="derived",
                    choices=("derived", "fixed", "ladder", "executable"),
                    help="derived (DEFAULT): read the entry points out of the "
                         "launch sources; fixed: the measured 14 of C105; "
                         "ladder: C105's original 7, to reproduce that audit; "
                         "executable: EVERY __main__-guarded script under the "
                         "entry roots — the paranoid superset")
    ap.add_argument("--launch-source", nargs="*",
                    default=list(DEFAULT_LAUNCH_SOURCES),
                    help="files that emit or document launch command lines; "
                         "the derivation reads entry points out of these")
    ap.add_argument("--entry-roots", nargs="*", default=list(DEFAULT_ENTRY_ROOTS),
                    help="directories an executable may live in")
    ap.add_argument("--show-uncovered", action="store_true",
                    help="list the launchable scripts this closure does NOT "
                         "cover (the measured gap; the count always prints)")
    ap.add_argument("--roots", nargs="*", default=list(DEFAULT_ROOTS))
    ap.add_argument("--host", default=None,
                    help="ssh alias of the target box; omit for a local closure")
    ap.add_argument("--remote-root", default="/home/nvidia/TanitAD")
    ap.add_argument("--remote-python", default="python3",
                    help="interpreter for the cheap remote hash/symbol probes")
    ap.add_argument("--import-python", default=None,
                    help="interpreter for --verify-import; MUST be the venv that "
                         "owns torch (e.g. /home/nvidia/venvs/tanitad-train/bin/"
                         "python), else every torch-touching module reports a "
                         "fake ImportError. Defaults to --remote-python.")
    ap.add_argument("--ssh-bin", default=None,
                    help="default: native OpenSSH on Windows, else 'ssh'")
    ap.add_argument("--ship", action="store_true",
                    help="push DRIFT + MISSING_REMOTE rows (LF-normalised)")
    ap.add_argument("--backup-dir", default=None,
                    help="remote dir for the box's originals; required by --ship")
    ap.add_argument("--ship-dirty", action="store_true",
                    help="also ship files that are UNCOMMITTED locally. Off by "
                         "default: a DRIFT row on a dirty file usually means "
                         "your tree moved, not that the box is stale.")
    ap.add_argument("--verify-import", action="store_true",
                    help="import every closure module on the box for real")
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args(argv)

    ssh_bin = args.ssh_bin
    if ssh_bin is None:
        win = Path("C:/Windows/System32/OpenSSH/ssh.exe")
        ssh_bin = str(win) if win.is_file() else (shutil.which("ssh") or "ssh")

    args.remote_root = _demangle_posix(args.remote_root)
    args.backup_dir = _demangle_posix(args.backup_dir)

    roots = [_REPO / r for r in args.roots]
    entry_roots = [(_REPO / r).resolve() for r in args.entry_roots]

    # ---- the root set: derived, not remembered -------------------------
    eset = resolve_entry_set(args, entry_roots)
    entries = [_REPO / e for e in eset.entries]
    missing_entries = [str(e) for e in entries if not e.is_file()]
    closure = compute_closure(entries, roots)

    launchable = executables_under(entry_roots, _REPO)
    uncovered = sorted(set(launchable) - set(closure.files))

    print_root_set(eset, args, closure, launchable, uncovered)

    print(f"\nclosure: {len(closure.files)} files from {len(closure.entries)} entry "
          f"points over roots {args.roots}")
    print(f"  eager (imported at process start): {len(closure.eager)}   "
          f"deferred-only (function-level, the C99 risk set): "
          f"{len(closure.deferred_only)}")
    if missing_entries:
        print(f"  !! entry points not found: {missing_entries}")
    if closure.unparsed:
        print(f"  !! unparsed: {closure.unparsed}")

    report: dict = {"closure": asdict(closure), "roots": args.roots,
                    "missing_entries": missing_entries,
                    "entry_set": asdict(eset),
                    "entry_roots": args.entry_roots,
                    "launchable_scripts": launchable,
                    "uncovered_executables": uncovered}

    if not args.host:
        for f in closure.files:
            print(f"  {f}")
        if args.json:
            Path(args.json).write_text(json.dumps(report, indent=2))
        return 0

    loc = local_hashes(closure.files)
    rem = remote_hashes(args.host, args.remote_root, closure.files,
                        ssh_bin, args.remote_python)
    head = head_lf_digests(closure.files)
    rows = build_table(closure, loc, rem, head)
    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICTS}
    # ⛔ A DRIFT row whose LOCAL file is dirty vs HEAD is not evidence the box is
    # stale — it is evidence YOUR tree moved. Separate them before anyone reads
    # a count, and never let --ship push a sibling's uncommitted work.
    dirty = [r["path"] for r in rows
             if r["verdict"] == "DRIFT" and r.get("local_dirty_vs_head")]
    report.update({"host": args.host, "remote_root": args.remote_root,
                   "rows": rows, "counts": counts,
                   "drift_explained_by_local_uncommitted_edits": dirty,
                   "compared_side": "WORKING TREE (annotated against HEAD)"})

    print(f"\n{args.host}:{args.remote_root}  " +
          "  ".join(f"{k}={counts[k]}" for k in VERDICTS))
    print("  compared side: WORKING TREE (each row also checked against HEAD)")
    for r in rows:
        if r["verdict"] in ("DRIFT", "MISSING_REMOTE", "MISSING_LOCAL"):
            tag = ""
            if r.get("local_dirty_vs_head"):
                tag = ("  <- ⚠️ LOCAL IS DIRTY vs HEAD"
                       + ("; REMOTE == HEAD, so this is YOUR uncommitted edit, "
                          "NOT box staleness" if r.get("remote_matches_head")
                          else ""))
            print(f"  {r['verdict']:<15} [{r['reach']:<13}] {r['path']}  "
                  f"local={r['local_bytes']} remote={r['remote_bytes']}"
                  + (f" ratio={r['size_ratio_local_over_remote']}"
                     if "size_ratio_local_over_remote" in r else "") + tag)
    if dirty:
        print(f"\n  ⚠️ {len(dirty)} of {counts['DRIFT']} DRIFT rows are explained "
              f"by UNCOMMITTED local edits, not by the box: {dirty}")
        print("     Shipping these would push work-in-progress onto the box.")

    why = explain_drift(args.host, args.remote_root, rows, ssh_bin,
                        args.remote_python)
    if why:
        report["drift_detail"] = why
        print("\nwhat actually differs (top-level symbol census):")
        for rel, d in sorted(why.items()):
            miss = d.get("missing_on_remote", [])
            print(f"  {rel}: lines {d.get('remote_lines')} -> "
                  f"{d.get('local_lines')}, missing on remote "
                  f"({len(miss)}): {miss[:12]}")

    stale = [r["path"] for r in rows
             if r["verdict"] in ("DRIFT", "MISSING_REMOTE")]
    if args.ship and dirty and not args.ship_dirty:
        # ⛔ Refuse by default. The box runs a 5-day job; a sibling agent's
        # half-written module is the last thing that should land on it.
        held = [p for p in stale if p in set(dirty)]
        stale = [p for p in stale if p not in set(dirty)]
        print(f"\n  ⛔ HOLDING BACK {len(held)} file(s) that are uncommitted "
              f"locally: {held}\n     (--ship-dirty to override, deliberately)")
        report["ship_held_back_dirty"] = held

    if args.ship and stale:
        if not args.backup_dir:
            print("--ship requires --backup-dir", file=sys.stderr)
            return 2
        print(f"\nshipping {len(stale)} files (LF-normalised), backup -> "
              f"{args.backup_dir}")
        res = ship(args.host, args.remote_root, stale, args.backup_dir,
                   ssh_bin, args.remote_python)
        bad = [rel for rel, rec in res.items()
               if rec["md5_after"] != loc[rel]["md5_lf"]]
        report["ship"] = {"result": res, "mismatched": bad,
                          "verified_against": "md5_lf (LF bytes to a Linux box)"}
        print(f"  shipped={len(res)}  md5-mismatch={len(bad)} {bad}")

    if args.verify_import:
        mods = sorted({m for m in
                       (_module_name_of(_REPO / f, roots) for f in closure.files)
                       if m})
        imp_py = args.import_python or args.remote_python
        print(f"\nreal-import probe: {len(mods)} modules on {args.host} "
              f"via {imp_py}")
        res = verify_import(args.host, args.remote_root, mods, ssh_bin, imp_py)
        # ⚠️ A direct import_module() BYPASSES a `try/except ImportError` the
        # real launch path relies on. Classify, or the probe fabricates blockers.
        guarded_mods = {m for m in
                        (_module_name_of(_REPO / f, roots)
                         for f in closure.guarded_only) if m}
        res["blocking"] = sorted(m for m in res["bad"] if m not in guarded_mods)
        res["tolerated_guarded"] = sorted(m for m in res["bad"]
                                          if m in guarded_mods)
        report["import_probe"] = res
        print(f"  n_ok={res['n_ok']}  n_bad={res['n_bad']}  "
              f"(BLOCKING={len(res['blocking'])}, "
              f"guarded/tolerated={len(res['tolerated_guarded'])})")
        eager_mods = {m for m in (_module_name_of(_REPO / f, roots)
                                  for f in closure.eager) if m}
        res["reach"] = {m: ("eager" if m in eager_mods else "deferred_only")
                        for m in res["bad"]}
        for m, e in sorted(res["bad"].items()):
            tag = "GUARDED " if m in guarded_mods else "BLOCKING"
            print(f"    {tag} [{res['reach'][m]:<13}] {m}: {e}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")

    return 1 if (counts["DRIFT"] or counts["MISSING_REMOTE"]) and not args.ship else 0


if __name__ == "__main__":
    raise SystemExit(main())
