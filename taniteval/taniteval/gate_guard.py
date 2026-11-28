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
           "scan_source", "scan_file", "scan_paths",
           "NORMAL_QUANTILES", "is_declared_estimator_name",
           "scan_source_shapes", "scan_file_shapes", "scan_paths_shapes"]

#: Exact names that are the banned estimator, wherever they are bound.
BANNED_CALLS = frozenset({
    "overlapping_holdout_se",
    "_jack", "_jack_scalar", "_jack_paired", "jack_scalar", "jack_paired",
    "_agg_jack", "jackknife_ci",
    # the ddof=1 VARIANT found 2026-08-23 in stack/scripts/driving_diagnostic.py.
    # It was the unnamed clone; `mean_ci` is its back-compat alias.
    "overlapping_holdout_mean_ci", "mean_ci",
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
    site read ``agg(...)``, which no name-based rule would catch on its own.

    ⭐ A **DECLARED** name (:data:`_DECLARED_RE` — anything containing
    ``overlapping_holdout`` or ``jackknife``) is banned here too, and the
    asymmetry with the shape detector is the point: declaring the estimator
    buys a function the right to CONTAIN the arithmetic, never the right to
    DECIDE with it. MEASURED 2026-08-28: renaming
    ``recompute_ci.naive_published`` to
    ``reproduce_overlapping_holdout_published`` correctly exempted it from the
    shape detector — and simultaneously left a verdict computed from its output
    invisible to this guard, because the new name was in neither
    :data:`BANNED_CALLS` nor :data:`BANNED_CALL_RE`. The declared exemption had
    a hole exactly where it was widest."""
    if isinstance(node, ast.Call):
        return is_banned_callable(node.func, aliases)
    if isinstance(node, ast.Attribute):
        name = node.attr
    elif isinstance(node, ast.Name):
        name = node.id
    else:
        return None
    if name in BANNED_CALLS or BANNED_CALL_RE.match(name) or name in aliases \
            or _DECLARED_RE.search(name):
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
                if orig in BANNED_CALLS or BANNED_CALL_RE.match(orig) \
                        or _DECLARED_RE.search(orig):
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


def _files_under(root, pattern, skip):
    """Every file under ``root`` matching ``pattern``, minus the ``skip`` tokens.

    ⛔ **THE SKIP IS MATCHED RELATIVE TO ``root``, AND THAT IS LOAD-BEARING**
    (MEASURED 2026-08-23). It used to match ``f.as_posix()`` — the ABSOLUTE
    path — against tokens like ``"/.claude/"``. Every agent in this programme
    works in a git worktree at ``<repo>/.claude/worktrees/<name>/``, so that
    token matched the CHECKOUT ROOT and the guard skipped **373 of 373 files
    and scanned nothing**. The suite went green because it had no input, not
    because the tree was clean — an instrument structurally unable to report
    the answer it is cited for, which is the same class as ``df`` hiding the
    per-pod quota.

    Matching relative to ``root`` makes the tokens mean what they read as: a
    directory *inside the scanned tree*, never an accident of where the tree
    happens to live.
    """
    r = Path(root)
    if not r.exists():
        return []
    if r.is_file():
        return [r]
    out = []
    for f in sorted(r.glob(pattern)):
        rel = f.relative_to(r).as_posix()
        # leading+trailing slash so a token like "/experiments/" matches a path
        # SEGMENT, not a substring of a filename
        if any(s in f"/{rel}" for s in skip):
            continue
        out.append(f)
    return out


def scan_paths(roots, pattern="**/*.py", skip=()) -> list[Violation]:
    """Scan every ``.py`` under each root. Missing roots are skipped, not fatal."""
    out: list[Violation] = []
    for root in roots:
        for f in _files_under(root, pattern, skip):
            out += scan_file(f)
    return out


# =========================================================================== #
# ⭐ THE ARITHMETIC-SHAPE DETECTOR — closes the CLASS, not the case            #
# =========================================================================== #
"""WHY A SECOND, NAME-INDEPENDENT DETECTOR EXISTS
=================================================

Everything above matches a NAME. :data:`BANNED_CALLS` and
:data:`BANNED_CALL_RE` are a list of spellings, and
:func:`banned_import_aliases` closes the *import*-rename loophole. None of that
sees an estimator that was simply **re-typed under an innocent name**.

MEASURED 2026-08-23: ``stack/scripts/driving_diagnostic.py`` defined::

    def mean_ci(vals):
        n = len(vals); m = sum(vals) / n
        std = (sum((v - m) ** 2 for v in vals) / max(1, n - 1)) ** 0.5
        return {"mean": ..., "ci95": round(1.96 * std / n ** 0.5, 4), ...}

That is the forbidden estimator — ``z · dispersion / sqrt(n)`` over a vector of
**per-split means** — under a name no grep for ``jack`` or
``overlapping_holdout`` will ever return, carrying no ``estimator`` field, and
reachable from three live callers. It survived two independent audits *because
both audits searched for names*. **A name-keyed guard against a mathematical
defect is always one rename behind.**

⚠️ **THE SHAPE ALONE IS NOT A VERDICT, AND THIS DETECTOR DOES NOT PRETEND IT IS.**
``1.96 * sd / sqrt(n)`` over **independent** samples is a perfectly correct
normal-approximation standard error — ``tanitad/eval/bakeoff.mean_ci95`` uses it
over independent training SEEDS and is fine. What makes it the forbidden
estimator is the **provenance of the input** (a set of *overlapping* random
holdouts of one 40-episode pool), and provenance is a semantic property that no
syntax tree can settle.

⇒ The detector therefore reports the SHAPE, and admissibility is settled by an
explicit, reasoned ALLOWLIST that lives in the test
(``tests/test_no_jack_in_gates.py::SHAPE_ALLOWLIST``), not in this module. The
consequences are the point:

* a NEW site is a **failing test on the day it is written**, whatever it is
  called — the author cannot land it without writing down why the inputs are
  independent;
* the allowlist is a short, reviewable list of *reasoned exceptions* rather than
  a growing list of *spellings to ban*;
* the reason is checked in beside the code, so the next auditor reads the
  argument instead of re-deriving it.

THE ONE IN-MODULE EXEMPTION is a **declared** estimator: a function whose own
name already says it is the banned family (:func:`is_declared_estimator_name` —
``ci.overlapping_holdout_se``, the ``_jack_*`` family, anything ``_LEGACY``).
Those are the quarantined reproductions; they are supposed to contain this
arithmetic, and the name guard above already governs where their output may go.
"""

#: Two-sided normal quantiles. A float literal from this set multiplying a
#: dispersion is the "z ·" of a ``z · SE`` construction. 1.96 is the one that
#: matters here; the others are included so a 90 %/99 % restatement of the same
#: defect is caught by the same rule.
NORMAL_QUANTILES = frozenset({
    1.96, 1.959964, 1.95996, 1.9599639845400545,   # 95 %
    1.64, 1.645, 1.6449, 1.6448536269514722,       # 90 %
    2.58, 2.576, 2.5758, 2.5758293035489004,       # 99 %
})
#: Callables that return a dispersion.
_DISPERSION_FUNCS = frozenset({"std", "nanstd", "stdev", "pstdev", "tstd"})
#: Names that hold one. Deliberately broad — a false positive costs one
#: allowlist line with a reason; a false negative costs a wrong interval.
_DISPERSION_NAME_RE = re.compile(
    r"(^|_)(std|sd|sigma|stdev|stddev|dispersion|scale)($|_)", re.I)
#: Callables that return a square root.
_SQRT_FUNCS = frozenset({"sqrt", "isqrt"})
#: Names that hold a variance, so ``sqrt(var)`` reads as a dispersion.
_VAR_NAME_RE = re.compile(r"(^|_)(var|variance|ss|sumsq)($|_)", re.I)


#: A name that ANNOUNCES the estimator it implements. Only these self-exempt.
_DECLARED_RE = re.compile(r"overlapping_holdout|jackknife", re.I)


def is_declared_estimator_name(name: str) -> bool:
    """True if this function name ALREADY declares itself the banned family.

    The quarantined reproductions (``overlapping_holdout_se``,
    ``overlapping_holdout_mean_ci``, the ``_jack_*`` family, anything
    ``_LEGACY``) are *supposed* to contain the arithmetic — that is what they
    are for, and the name guard above already governs where their output may go.

    ⚠️ **Membership of :data:`BANNED_CALLS` is deliberately NOT sufficient.**
    That set exists to stop a banned RESULT reaching a verdict, so it also holds
    innocent-sounding aliases like ``mean_ci``. Letting those self-exempt would
    hand every future clone a one-line escape: add your name to the ban list and
    the shape detector stops looking at you. Only a name that *says what it is*
    counts as a declaration."""
    if not name:
        return False
    return bool(_DECLARED_RE.search(name) or BANNED_CALL_RE.match(name)
                or _LEGACY_RE.search(name))


def _const(node):
    """The float value of a numeric constant node, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _const(node.operand)
        return None if v is None else -v
    return None


def _is_quantile(node) -> bool:
    v = _const(node)
    if v is None:
        return False
    # match on the printed value, so 1.96 and 1.9599639845400545 are one rule
    return any(abs(v - q) < 5e-4 for q in NORMAL_QUANTILES)


def _callee_name(node):
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    return f.attr if isinstance(f, ast.Attribute) else \
        f.id if isinstance(f, ast.Name) else None


def _is_sqrt(node) -> bool:
    """``sqrt(x)`` or ``x ** 0.5`` — the divisor of a standard error."""
    if _callee_name(node) in _SQRT_FUNCS:
        return True
    return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow)
            and _const(node.right) == 0.5)


def _sqrt_operand(node):
    if isinstance(node, ast.Call) and node.args:
        return node.args[0]
    if isinstance(node, ast.BinOp):
        return node.left
    return None


def _is_dispersion(node, disp_names=()) -> bool:
    """A standard deviation, however it was spelled.

    ``disp_names`` are local names PROVEN to hold a dispersion by
    :func:`_dispersion_names`. Without them the detector was still name-keyed
    in its numerator: ``spread = np.std(vals)`` followed by
    ``1.96 * spread / sqrt(n)`` slipped through, because ``spread`` is not in
    :data:`_DISPERSION_NAME_RE`. MEASURED 2026-08-28 by a deliberate-regression
    fixture in ``test_no_jack_in_gates.py``. That is the same "one rename
    behind" failure the shape detector exists to end, one level down — so the
    numerator now follows the DATA, exactly as :func:`_se_names` already did
    for the whole ``dispersion / sqrt(n)`` expression."""
    if _callee_name(node) in _DISPERSION_FUNCS:
        return True
    if isinstance(node, ast.Name) and (_DISPERSION_NAME_RE.search(node.id)
                                       or node.id in disp_names):
        return True
    if isinstance(node, ast.Attribute) and _DISPERSION_NAME_RE.search(node.attr):
        return True
    if _is_sqrt(node):
        # sqrt(var) / (sum((v-m)**2)/(n-1)) ** 0.5 — a stdev computed by hand,
        # which is exactly how the unnamed clone spelled it.
        inner = _sqrt_operand(node)
        if inner is None:
            return False
        if isinstance(inner, ast.Name) and _VAR_NAME_RE.search(inner.id):
            return True
        for n in ast.walk(inner):
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow) \
                    and _const(n.right) == 2:
                return True          # a hand-rolled sum of squared deviations
    return False


def _muldiv(node, num=None, den=None):
    """Flatten a ``*``/``/`` chain into ``(numerator_nodes, denominator_nodes)``.

    ``1.96 * std / n ** 0.5`` and ``(1.96 * std) / sqrt(n)`` and
    ``1.96 * (std / sqrt(n))`` are the same expression written three ways; the
    guard must not depend on which one an author picked."""
    num = [] if num is None else num
    den = [] if den is None else den
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Div)):
        _muldiv(node.left, num, den)
        if isinstance(node.op, ast.Mult):
            _muldiv(node.right, num, den)
        else:
            _muldiv(node.right, den, num)
        return num, den
    num.append(node)
    return num, den


def _dispersion_names(tree, seed=()) -> set:
    """Local names bound to a dispersion — ``spread = np.std(v)`` and friends.

    Fixpoint, so ``a = np.std(v)`` then ``b = a`` then ``1.96 * b / sqrt(n)``
    is still caught. Deliberately narrow: only a name whose VALUE is itself a
    dispersion qualifies, so ``mean = np.mean(v)`` does not become one."""
    out = set(seed)
    for _ in range(4):
        changed = False
        for n in ast.walk(tree):
            if not isinstance(n, (ast.Assign, ast.AnnAssign)) or n.value is None:
                continue
            if not _is_dispersion(n.value, out):
                continue
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for nm in (m for t in targets for m in _target_names(t)):
                if nm not in out:
                    out.add(nm)
                    changed = True
        if not changed:
            break
    return out


def _is_se_shape(node, disp_names=()) -> bool:
    """``dispersion / sqrt(...)`` — a standard error, without the z."""
    if not isinstance(node, ast.BinOp):
        return False
    num, den = _muldiv(node)
    return (any(_is_dispersion(x, disp_names) for x in num)
            and any(_is_sqrt(x) for x in den))


def _se_names(tree, disp_names=()) -> dict:
    """Names bound to a bare ``dispersion / sqrt(n)`` — the laundered form.

    ``se = std / sqrt(n)`` on one line and ``ci = 1.96 * se`` on the next is the
    same estimator split over two statements, and is the first thing an author
    reaches for when a one-line guard starts complaining."""
    out = {}
    for _ in range(3):                     # short chains, fixpoint in practice
        changed = False
        for n in ast.walk(tree):
            if not isinstance(n, (ast.Assign, ast.AnnAssign)) or n.value is None:
                continue
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            names = [nm for t in targets for nm in _target_names(t)]
            hit = _is_se_shape(n.value, disp_names) or bool(
                _names_read(n.value) & set(out))
            if not hit:
                continue
            for nm in names:
                if nm not in out:
                    out[nm] = n.lineno
                    changed = True
        if not changed:
            break
    return out


def _enclosing_functions(tree) -> dict:
    """``node -> nearest enclosing function name`` for every node in ``tree``."""
    owner = {}
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for n in ast.walk(fn):
                owner.setdefault(n, fn.name)
    return owner


def scan_source_shapes(src, path="<src>") -> list[Violation]:
    """Every ``z · dispersion / sqrt(n)`` construction in ``src``.

    Name-independent by construction: it reads the ARITHMETIC. See the module
    note above for why the result is a shape report and not a verdict, and why
    the admissibility allowlist lives in the test rather than here."""
    tree = ast.parse(src, filename=str(path))
    owner = _enclosing_functions(tree)
    disp = _dispersion_names(tree)
    se = _se_names(tree, disp)
    out: list[Violation] = []
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp) or \
                not isinstance(node.op, (ast.Mult, ast.Div)):
            continue
        num, den = _muldiv(node)
        if not any(_is_quantile(x) for x in num):
            continue
        direct = (any(_is_dispersion(x, disp) for x in num)
                  and any(_is_sqrt(x) for x in den))
        via = sorted({x.id for x in num
                      if isinstance(x, ast.Name) and x.id in se})
        if not direct and not via:
            continue
        fname = owner.get(node, "<module>")
        if is_declared_estimator_name(fname):
            continue                       # a quarantined reproduction, by name
        key = (fname, node.lineno)
        if key in seen:
            continue
        seen.add(key)
        why = ("multiplies a normal quantile by `dispersion / sqrt(n)`"
               if direct else
               f"multiplies a normal quantile by {via[0]!r}, bound to "
               f"`dispersion / sqrt(n)` at line {se[via[0]]}")
        out.append(Violation(
            path, node.lineno, fname,
            f"{why} — the ARITHMETIC of `overlapping_holdout_se` under a name "
            f"no name-keyed rule can see. If the inputs really are independent "
            f"this is a valid SE: add it to SHAPE_ALLOWLIST with the reason."))
    return out


def scan_file_shapes(path) -> list[Violation]:
    p = Path(path)
    try:
        src = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        src = p.read_text(encoding="utf-8", errors="replace")
    try:
        return scan_source_shapes(src, p)
    except SyntaxError:
        return []


def scan_paths_shapes(roots, pattern="**/*.py", skip=()) -> list[Violation]:
    """:func:`scan_paths` for the arithmetic detector."""
    out: list[Violation] = []
    for root in roots:
        for f in _files_under(root, pattern, skip):
            out += scan_file_shapes(f)
    return out
