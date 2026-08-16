"""AST guard: a GATE may never be decided by the banned estimator family.

WHY AN AST WALK AND NOT A REGEX
-------------------------------
Two failure modes killed the obvious regex version, and both are already in the
programme's trap list:

1. **A regex matches its own documentation.** A guard grepping for ``_jack_``
   fires on the comment that says *"``_jack_`` is retired"* — the same
   self-match that makes ``pgrep -f <trainer>`` kill your own ssh session and
   makes a log monitor report a failure that never happened. An AST walk never
   sees a comment or a docstring, so the retirement notice is invisible to it
   **by construction**, not by a cleverer pattern.
2. **A name-keyed regex misses the inlined form.** ``G1_pass`` is not always
   spelled ``x = _jack_paired(...)`` then ``bool(x["mean"] > 0)``; it can be
   ``bool(_jack_paired(a, b, e, s)["mean"] >= 0.2)`` on one line, or reached
   through two intermediate variables. Taint propagation over the syntax tree
   follows the DATA, so the spelling does not matter.

WHAT IT CHECKS
--------------
For every module in scope:

* **Taint sources** — any call to a banned estimator: the ``_jack_*`` family,
  ``overlapping_holdout_se``, or anything whose name is in :data:`BANNED_CALLS`.
* **Taint propagation** — to fixpoint: a name assigned from an expression that
  contains a taint source, or that reads an already-tainted name, becomes
  tainted. Subscripts, comprehensions and tuple targets all propagate.
* **Deciding expressions** — the value of any dict key or assignment target
  whose name carries a verdict (:func:`is_deciding_name`).
* **The violation** — a deciding expression that reads a tainted name or calls
  a banned estimator directly.

THE ONE EXEMPTION is an explicit ``_LEGACY`` suffix. A key named
``G4_pass_LEGACY`` is *supposed* to carry the deprecated verdict, and the suffix
is the greppable, deliberate declaration that it is not the decision. Nothing
else is exempt — not a comment, not a docstring, not a ``# noqa``.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

__all__ = ["BANNED_CALLS", "BANNED_CALL_RE", "Violation", "is_deciding_name",
           "is_banned_callable", "banned_import_aliases",
           "scan_source", "scan_file", "scan_paths"]

#: Exact names that are the banned estimator, wherever they are bound.
BANNED_CALLS = frozenset({
    "overlapping_holdout_se",
    "_jack", "_jack_scalar", "_jack_paired", "jack_scalar", "jack_paired",
    "_agg_jack", "jackknife_ci",
})
#: The whole ``_jack_*`` family, so a new sibling is caught the day it is written.
BANNED_CALL_RE = re.compile(r"^_{0,2}jack(_|$)")

#: A key/target name that carries a verdict. ``_pass`` as a token, or an
#: explicit verdict/gate word. Deliberately broad — a false positive costs one
#: rename, a false negative costs a wrong gate.
_DECIDING_RE = re.compile(
    r"(^|_)(pass|passed|verdict|gated?_ok|admissible)($|_)|^G\d+_pass", re.I)
#: The single documented exemption: an explicitly-labelled legacy reproduction.
_LEGACY_RE = re.compile(r"(_LEGACY$|^legacy_|_legacy$)")


class Violation(tuple):
    """``(path, lineno, key, reason)`` — printable in a pytest assert message."""

    __slots__ = ()

    def __new__(cls, path, lineno, key, reason):
        return super().__new__(cls, (str(path), int(lineno), str(key),
                                     str(reason)))

    def __str__(self):
        p, ln, key, why = self
        return f"{p}:{ln}: {key} <- {why}"


def is_banned_callable(node: ast.AST, aliases=()) -> str | None:
    """Name of the banned estimator this ``Call``/``Attribute``/``Name`` names.

    ``aliases`` are extra local names bound to a banned estimator by an import
    — ``from taniteval.planner_p2 import _jack_paired as agg`` makes the call
    site read ``agg(...)``, which no name-based rule would catch on its own."""
    if isinstance(node, ast.Call):
        return is_banned_callable(node.func, aliases)
    if isinstance(node, ast.Attribute):
        name = node.attr
    elif isinstance(node, ast.Name):
        name = node.id
    else:
        return None
    if name in BANNED_CALLS or BANNED_CALL_RE.match(name) or name in aliases:
        return name
    return None


def banned_import_aliases(tree: ast.AST) -> dict[str, str]:
    """``local_name -> original_name`` for imports of a banned estimator.

    Closes the rename loophole: a banned estimator imported under an innocent
    alias is still the banned estimator."""
    out: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                orig = a.name.rsplit(".", 1)[-1]
                if orig in BANNED_CALLS or BANNED_CALL_RE.match(orig):
                    out[a.asname or orig] = orig
    return out


def is_deciding_name(name: str) -> bool:
    """True if a key/variable of this name carries a gate verdict."""
    if not name:
        return False
    if _LEGACY_RE.search(name):
        return False
    return bool(_DECIDING_RE.search(name))


def _names_read(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _banned_calls_in(node: ast.AST, aliases=()) -> set[str]:
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            b = is_banned_callable(n, aliases)
            if b:
                out.add(b)
    return out


def _target_names(target: ast.AST) -> list[str]:
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out = []
        for e in target.elts:
            out += _target_names(e)
        return out
    return []


def _tainted_names(tree: ast.AST, aliases=()) -> dict[str, str]:
    """Fixpoint taint: ``name -> why``.

    One pass is not enough — ``heldout = {k: _jack_scalar(...)}`` then
    ``cb = heldout["closed_bike"]`` then ``bool(cb["mean"] < thr)`` needs two.
    Iterating to a fixpoint means an arbitrarily long laundering chain still
    reaches the verdict.
    """
    tainted: dict[str, str] = {}
    assigns = [n for n in ast.walk(tree)
               if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))]
    for _ in range(len(assigns) + 2):
        changed = False
        for a in assigns:
            value = a.value
            if value is None:
                continue
            targets = (a.targets if isinstance(a, ast.Assign)
                       else [a.target])
            names = [n for t in targets for n in _target_names(t)]
            if not names:
                continue
            direct = _banned_calls_in(value, aliases)
            via = _names_read(value) & set(tainted)
            if not direct and not via:
                continue
            why = (f"calls {sorted(direct)[0]}" if direct
                   else f"reads tainted {sorted(via)[0]}")
            for nm in names:
                if nm not in tainted:
                    tainted[nm] = why
                    changed = True
        if not changed:
            break
    return tainted


def _deciding_exprs(tree: ast.AST):
    """Yield ``(name, lineno, value_node)`` for every verdict-carrying binding."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str) \
                        and is_deciding_name(k.value):
                    yield k.value, getattr(k, "lineno", node.lineno), v
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = (node.targets if isinstance(node, ast.Assign)
                       else [node.target])
            for t in targets:
                for nm in _target_names(t):
                    if is_deciding_name(nm) and node.value is not None:
                        yield nm, node.lineno, node.value
                # d["G4_pass"] = ... — a subscript store with a literal key
                if isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant) \
                        and isinstance(t.slice.value, str) \
                        and is_deciding_name(t.slice.value) \
                        and node.value is not None:
                    yield t.slice.value, node.lineno, node.value


def scan_source(src: str, path="<src>") -> list[Violation]:
    """Every deciding expression in ``src`` that is decided by a banned estimator."""
    tree = ast.parse(src, filename=str(path))
    aliases = banned_import_aliases(tree)
    tainted = _tainted_names(tree, aliases)
    out: list[Violation] = []
    for name, lineno, value in _deciding_exprs(tree):
        direct = _banned_calls_in(value, aliases)
        via = sorted(_names_read(value) & set(tainted))
        if direct:
            out.append(Violation(path, lineno, name,
                                 f"calls the BANNED estimator "
                                 f"{sorted(direct)[0]}() directly"))
        elif via:
            out.append(Violation(path, lineno, name,
                                 f"reads {via[0]!r}, which {tainted[via[0]]}"))
    return out


def scan_file(path) -> list[Violation]:
    p = Path(path)
    try:
        src = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        src = p.read_text(encoding="utf-8", errors="replace")
    try:
        return scan_source(src, p)
    except SyntaxError:
        return []          # not our file to police; the owning suite will fail


def scan_paths(roots, pattern="**/*.py", skip=()) -> list[Violation]:
    """Scan every ``.py`` under each root. Missing roots are skipped, not fatal."""
    out: list[Violation] = []
    for root in roots:
        r = Path(root)
        if not r.exists():
            continue
        files = [r] if r.is_file() else sorted(r.glob(pattern))
        for f in files:
            if any(s in f.as_posix() for s in skip):
                continue
            out += scan_file(f)
    return out
