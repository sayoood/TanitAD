"""taniteval.stack_guard — ⛔ REFUSE to evaluate on a ``tanitad`` the caller did
not ask for.

THE DEFECT THIS MODULE REMOVES (MEASURED 2026-07-27, ``SMALL_VALIDATION.md`` §5)
--------------------------------------------------------------------------------
**28 modules of this package CARRIED** a literal
``sys.path.insert(0, "/root/TanitAD/stack")`` — 52 lines in all (``bench``,
``closedloop``, ``corridor``, ``data``, ``driving``, ``blind_baseline``,
``rollout``, …). ``insert(0, …)`` puts that tree in FRONT of whatever the caller
put on ``PYTHONPATH``, so **merely importing ``taniteval.<anything>`` re-pointed
``tanitad`` at that tree**. They are gone as of 2026-07-27 and
``tests/test_stack_guard.py::test_no_live_hardcoded_stack_insert_survives_in_the_package``
fails if one returns.

MEASURED on pod2 2026-07-27: ``/root/TanitAD/stack`` there is a **12 MB tree with
no ``.git``, no ``tanitad/train/heldout_gate.py`` and no ``resolve_v2_frames``**
— i.e. pre-v5. The RED transcript
(``…/2026-07-27-small-validation/raw/eval_plumbing_RED.txt``) shows the process
resolving ``tanitad`` from ``/root/TanitAD/stack`` *while* ``PYTHONPATH`` pointed
only at ``/workspace/TanitAD/stack``.

⚠️ **The crash is the LUCKY case.** A module that exists in *both* trees resolves
silently to the stale one and the eval publishes a number computed by pre-v5
code — **a plausible wrong number rather than an error**, which is this
program's worst failure mode.

⚠️ **AND THE OBVIOUS SYNC CHECK DOES NOT CATCH IT.** A bare ``import tanitad``
resolves correctly; the shadowing only happens once ``taniteval`` is imported.
"Verify with a real ``import tanitad``" is an INSUFFICIENT instruction and was
given as sufficient.

THE THREE LAYERS, IN THE ORDER THEY FIRE
----------------------------------------
1. :func:`ensure_stack_on_path` — the *cause*. It replaces the hardcoded
   ``sys.path.insert(0, "/root/TanitAD/stack")``. The legacy tree is still used
   when **nothing else was named** (that is the deployed pod workflow, and it is
   pinned by ``tests/test_stack_guard.py::test_legacy_only_layout_is_unchanged``),
   but it is **never placed in front of a stack the caller named**.
2. :class:`StackSentinel` — the *tripwire*. A ``sys.meta_path`` finder that
   refuses any ``tanitad`` / ``tanitad.*`` import whose origin is outside the
   pinned root. It fires **no matter who** put the stale tree on ``sys.path`` —
   a module this package never fixed, a top-level script, a copied command.
   ⭐ This is the layer that survives "the next person copies a command from
   somewhere else".
3. :func:`assert_stack` — the *capability probe*. Identity is necessary but not
   sufficient: a tree can be the right PATH and still be pre-v5 (a pod that was
   never `git pull`ed). ``require=`` names symbols that must resolve, e.g.
   ``tanitad.train.heldout_gate:PRIMARY_NAME``.

WHAT COUNTS AS "THE ONE THE CALLER INTENDED"
--------------------------------------------
:func:`resolve_intended_stack`, in order — every source is caller-controlled
except the last:

======  ==========================================================  ============
order   source                                                      class
======  ==========================================================  ============
1       an explicit argument                                        explicit
2       ``TANITEVAL_STACK_OVERRIDE``                                explicit
3       an already-imported ``tanitad`` (someone got there first)   explicit
4       the first ``sys.path`` entry providing ``tanitad``,          explicit
        EXCLUDING the legacy tree (i.e. ``PYTHONPATH`` / ``cwd``)   (PYTHONPATH)
5       ``/root/TanitAD/stack`` — the deployed pod tree             fallback
6       an installed/editable ``tanitad`` (``find_spec``)            fallback
======  ==========================================================  ============

⭐ **The guard can therefore only fire when the caller named a stack and got a
different one.** When nothing is named, intent *is* the legacy tree and there is
nothing to violate — which is why turning this on cannot break the deployed eval
path that produced every published closed-loop number.

MODES
-----
``TANITEVAL_STACK_GUARD`` ∈ ``error`` (default) · ``warn`` · ``off``.
⚠️ ``off`` is recorded in :func:`report` so an artifact can never hide that the
guard was disabled.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import os
import sys
from pathlib import Path

__all__ = [
    "LEGACY_STACK", "ENV_OVERRIDE", "ENV_MODE", "V5_CAPABILITIES",
    "StackShadowError", "StackSentinel",
    "resolve_intended_stack", "ensure_stack_on_path", "install_sentinel",
    "uninstall_sentinel", "assert_stack", "report", "main",
]

#: The tree 28 modules of this package hardcoded until 2026-07-27. On pod2
#: (MEASURED) it is 12 MB, has no ``.git`` and predates v5; on ``tanitad-eval``
#: it is the ONLY tree and therefore still the right answer there.
LEGACY_STACK = "/root/TanitAD/stack"

ENV_OVERRIDE = "TANITEVAL_STACK_OVERRIDE"
ENV_MODE = "TANITEVAL_STACK_GUARD"

#: Symbols that exist only in a post-v5 ``stack``. Used by ``--require v5`` and
#: by :func:`assert_stack(require="v5")`. ``module:symbol``; a bare ``module``
#: means "importable".
V5_CAPABILITIES = (
    "tanitad.train.heldout_gate:PRIMARY_NAME",
    "tanitad.train.heldout_goal:make_goal_kwargs_fn",
    "tanitad.data.parity:register_v2_geometry_sibling",
    "tanitad.geometry:frame_from_args",
    "train_flagship_v4:resolve_v2_frames",
)

_GUARDED_TOP = "tanitad"


class StackShadowError(ImportError):
    """A ``tanitad`` module resolved outside the pinned stack root.

    ⭐ Raised, not warned, by default — the whole point is that the silent path
    publishes a number."""


def _eprint(msg: str) -> None:
    """stderr, but never the reason the guard itself dies.

    ⚠️ The banner carries ⛔/⭐ like the rest of the program, and a Windows
    console at cp1252 raises ``UnicodeEncodeError`` on those. A guard that
    crashes while reporting is a guard that stopped a run for the wrong reason,
    so the ASCII fallback is not cosmetic."""
    try:
        print(msg, file=sys.stderr, flush=True)
    except UnicodeEncodeError:                                # pragma: no cover
        print(msg.encode("ascii", "replace").decode("ascii"),
              file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# path helpers                                                                 #
# --------------------------------------------------------------------------- #
def _norm(p) -> str:
    try:
        return str(Path(p).resolve())
    except (OSError, ValueError):                             # pragma: no cover
        return str(p)


def _is_stack_root(p) -> bool:
    """A stack root is a directory that provides the ``tanitad`` package."""
    try:
        return Path(p, "tanitad", "__init__.py").is_file()
    except (OSError, ValueError):                             # pragma: no cover
        return False


def _under(child, parent) -> bool:
    c, p = _norm(child), _norm(parent)
    if os.name == "nt":
        c, p = c.lower(), p.lower()
    return c == p or c.startswith(p.rstrip(os.sep) + os.sep)


def _root_of_module(mod) -> str | None:
    """``<root>`` such that ``<root>/tanitad/__init__.py`` is ``mod.__file__``."""
    f = getattr(mod, "__file__", None)
    if not f:
        paths = list(getattr(mod, "__path__", []) or [])
        if not paths:
            return None
        return _norm(Path(paths[0]).parent)
    return _norm(Path(f).resolve().parent.parent)


def mode() -> str:
    m = (os.environ.get(ENV_MODE) or "error").strip().lower()
    return m if m in ("error", "warn", "off") else "error"


def resolve_intended_stack(explicit=None) -> tuple[str | None, str]:
    """Return ``(root, provenance)`` — see the module docstring's table.

    ⚠️ ``provenance`` is returned, not just the path, because the *evidence
    class* of the answer differs: ``legacy-fallback`` means **nothing was
    named**, and the guard must then have no teeth."""
    if explicit:
        return _norm(explicit), "explicit-argument"
    ov = os.environ.get(ENV_OVERRIDE)
    if ov:
        return _norm(ov), f"env:{ENV_OVERRIDE}"
    mod = sys.modules.get(_GUARDED_TOP)
    if mod is not None:
        r = _root_of_module(mod)
        if r:
            return r, "already-imported:tanitad"
    legacy = _norm(LEGACY_STACK)
    for entry in list(sys.path):
        cand = _norm(entry or os.getcwd())
        if cand == legacy:
            continue
        if _is_stack_root(cand):
            return cand, "sys.path"
    # ⛔ The legacy tree OUTRANKS an installed one on purpose. On a deployed pod
    # it is the intent, and every published closed-loop number came from it; a
    # stray `pip install -e` must not silently re-point that host.
    if _is_stack_root(legacy):
        return legacy, "legacy-fallback"
    # An editable/installed `tanitad` is reachable through a custom meta-path
    # finder and never appears on sys.path at all — that is the dev box
    # (`__editable__.tanitad-0.0.1.pth`). Without this branch the guard would
    # find no root there and silently install NOTHING.
    try:
        import importlib.util as _u
        spec = _u.find_spec(_GUARDED_TOP)
    except Exception:                                         # noqa: BLE001
        spec = None
    if spec is not None:
        origin = spec.origin or next(iter(spec.submodule_search_locations or []),
                                     None)
        if origin:
            p = Path(origin)
            root = p.parent.parent if p.name == "__init__.py" else p.parent
            return _norm(root), "installed:find_spec"
    return None, "none"


def discover_alternative_stacks(pinned: str) -> list[str]:
    """Other stack roots visible on this host that are NOT the pinned one.

    ⭐ Why this exists — MEASURED 2026-07-27 across the fleet:

    ==============  ============================  ===============================
    host            ``/root/TanitAD/stack``       ``/workspace/TanitAD/stack``
    ==============  ============================  ===============================
    ``tanitad-eval`` ✅ the ONLY tree (git)       ❌ absent
    ``pod3``         ❌ absent                    ✅ the only tree (git)
    ``pod2``         ⚠️ **present, no .git, pre-v5**  ⚠️ **present, git, post-v5**
    ==============  ============================  ===============================

    On the first two hosts there is nothing to be ambiguous about. **On pod2
    there are two, and a caller who names neither gets the pre-v5 one in
    silence** — which is the whole defect. The resolution order is NOT changed
    here (that would move the deployed eval host), but the ambiguity is made
    LOUD, so "I didn't set anything" stops looking like "there was nothing to
    set".
    """
    cands: list[Path] = []
    try:  # the stack that sits next to the HARNESS actually running
        cands.append(Path(__file__).resolve().parents[2] / "stack")
    except Exception:                                         # pragma: no cover
        pass
    try:                                                      # an installed one
        import importlib.util as _u
        sp = _u.find_spec(_GUARDED_TOP)
        if sp is not None:
            o = sp.origin or next(iter(sp.submodule_search_locations or []), None)
            if o:
                p = Path(o)
                cands.append(p.parent.parent if p.name == "__init__.py"
                             else p.parent)
    except Exception:                                         # noqa: BLE001
        pass
    out = {_norm(c) for c in cands
           if _is_stack_root(c) and not _under(c, pinned) and not _under(pinned, c)}
    return sorted(out)


def _insert_front(p: str) -> None:
    if p and os.path.isdir(p):
        n = _norm(p)
        for e in list(sys.path):
            if _norm(e or os.getcwd()) == n:
                sys.path.remove(e)
        sys.path.insert(0, p)


# --------------------------------------------------------------------------- #
# layer 2 — the tripwire                                                       #
# --------------------------------------------------------------------------- #
class StackSentinel:
    """A ``sys.meta_path`` finder that refuses ``tanitad`` outside ``root``.

    ⭐ **Why a meta-path finder and not a one-shot assert.** A one-shot check at
    import time is defeated by the next ``sys.path.insert`` — and there are 28 of
    them in this package alone, plus every top-level driver script and every
    command a person copies from a document. The finder is consulted on **every**
    ``tanitad`` import for the life of the process, so the check cannot be
    outrun.

    It delegates to :class:`importlib.machinery.PathFinder` — the finder that
    would have served the import anyway — and only inspects the answer, so it
    changes *no* resolution semantics when nothing is shadowed.
    """

    def __init__(self, root: str, *, provenance: str = "", on_violation=None):
        self.root = _norm(root)
        self.provenance = provenance
        self.violations: list[dict] = []
        self._busy = False
        self._on_violation = on_violation

    # importlib.abc.MetaPathFinder protocol
    def find_spec(self, fullname, path=None, target=None):
        if not (fullname == _GUARDED_TOP
                or fullname.startswith(_GUARDED_TOP + ".")):
            return None
        if self._busy:                                        # re-entrancy guard
            return None
        self._busy = True
        try:
            # ⚠️ Delegate to EVERY other finder, not just PathFinder. An
            # editable install (`__editable__.tanitad-*.pth` on the dev box)
            # serves `tanitad` from a custom meta-path finder and never puts it
            # on sys.path — a PathFinder-only sentinel would return None there
            # and inspect nothing, i.e. be inert exactly where it looks armed.
            spec = None
            for finder in list(sys.meta_path):
                if finder is self:
                    continue
                fs = getattr(finder, "find_spec", None)
                if fs is None:
                    continue
                try:
                    spec = fs(fullname, path, target)
                except ImportError:
                    raise
                except Exception:                             # noqa: BLE001
                    spec = None
                if spec is not None:
                    break
        finally:
            self._busy = False
        if spec is None:
            return None
        origins = []
        if spec.origin and spec.origin not in ("built-in", "frozen", "namespace"):
            origins.append(spec.origin)
        origins += list(spec.submodule_search_locations or [])
        bad = [o for o in origins if not _under(o, self.root)]
        if bad:
            self._violate(fullname, bad)
        return spec

    def _violate(self, fullname, bad):
        rec = {"module": fullname, "origin": [_norm(b) for b in bad],
               "pinned_root": self.root, "provenance": self.provenance}
        self.violations.append(rec)
        msg = (
            f"⛔ STACK SHADOWING — `{fullname}` would be imported from\n"
            f"    {rec['origin']}\n"
            f"  but the pinned stack root is\n"
            f"    {self.root}   (intent: {self.provenance or 'unknown'})\n"
            f"  This is the defect that publishes a PLAUSIBLE WRONG NUMBER rather\n"
            f"  than an error: 28 taniteval modules used to sys.path.insert(0,\n"
            f"  {LEGACY_STACK!r}), and on pod2 that tree is pre-v5.\n"
            f"  ⇒ export {ENV_OVERRIDE}=<the stack you mean> BEFORE python3 starts,\n"
            f"    and make sure no other stack tree precedes it on sys.path.\n"
            f"    (sys.path[0:4] = {sys.path[0:4]})\n"
            f"  To downgrade deliberately: {ENV_MODE}=warn — it is recorded in the\n"
            f"  guard's report() so no artifact can hide it."
        )
        if self._on_violation is not None:
            self._on_violation(rec, msg)
            return
        m = mode()
        if m == "error":
            raise StackShadowError(msg)
        if m == "warn":
            _eprint("[taniteval.stack_guard] " + msg)


def install_sentinel(root: str, *, provenance: str = "") -> StackSentinel | None:
    """Install (or re-point) the single sentinel at ``sys.meta_path[0]``."""
    if mode() == "off" or not root:
        return None
    uninstall_sentinel()
    s = StackSentinel(root, provenance=provenance)
    sys.meta_path.insert(0, s)
    # An already-imported tanitad predates the sentinel, so check it by hand —
    # otherwise the one import that matters most is the one import never seen.
    mod = sys.modules.get(_GUARDED_TOP)
    if mod is not None:
        r = _root_of_module(mod)
        if r and not _under(r, s.root):
            s._violate(_GUARDED_TOP, [getattr(mod, "__file__", r)])
    return s


def uninstall_sentinel() -> None:
    for f in [f for f in sys.meta_path if isinstance(f, StackSentinel)]:
        sys.meta_path.remove(f)


def active_sentinel() -> StackSentinel | None:
    for f in sys.meta_path:
        if isinstance(f, StackSentinel):
            return f
    return None


# --------------------------------------------------------------------------- #
# layer 1 — the cause                                                          #
# --------------------------------------------------------------------------- #
def ensure_stack_on_path(*, explicit=None, legacy: str = LEGACY_STACK,
                         install: bool = True) -> dict:
    """⭐ THE REPLACEMENT FOR ``sys.path.insert(0, "/root/TanitAD/stack")``.

    Semantics, stated so the deployed path is provably unchanged:

    * **nothing named** (``provenance == "legacy-fallback"`` or ``"none"``) →
      the legacy tree and its ``scripts/`` go to the FRONT, exactly as the
      hardcoded lines did. This is the deployed pod workflow and every published
      closed-loop number came through it.
    * **something named** → *that* root and its ``scripts/`` go to the front and
      **the legacy tree is not added at all**. It is not appended either: a
      silent fallback to a pre-v5 tree is the same wrong-number failure one step
      later, and a loud ``ImportError`` is the better outcome.
    """
    root, prov = resolve_intended_stack(explicit)
    if root is None:
        # No stack anywhere we can see — behave exactly like the old lines so a
        # host we have never met is no worse off than before.
        _insert_front(os.path.join(legacy, "scripts"))
        _insert_front(legacy)
        return {"root": None, "provenance": prov, "legacy_used": True,
                "sentinel": False, "mode": mode()}
    _insert_front(os.path.join(root, "scripts"))
    _insert_front(root)
    sentinel = install_sentinel(root, provenance=prov) if install else None
    alts: list[str] = []
    if prov == "legacy-fallback":
        # ⚠️ Nothing was named, so the guard has no teeth BY DESIGN — but on a
        # host that carries two stack trees that silence is exactly the failure.
        # Shout; do not refuse (refusing here would move the deployed eval host).
        alts = discover_alternative_stacks(root)
        if alts and mode() != "off":
            _eprint(
                "[taniteval.stack_guard] ⚠️ AMBIGUOUS STACK — you named none, so "
                f"the deployed tree {root!r} was used, but this host ALSO has "
                f"{alts}. MEASURED on pod2: those two differ by a whole release "
                f"(no heldout_gate in the deployed one). Set "
                f"{ENV_OVERRIDE}=<the stack you mean> to make this a decision.")
    return {"root": root, "provenance": prov,
            "legacy_used": _norm(root) == _norm(legacy),
            "ambiguous_alternatives": alts,
            "sentinel": sentinel is not None, "mode": mode()}


# --------------------------------------------------------------------------- #
# layer 3 — the capability probe                                               #
# --------------------------------------------------------------------------- #
def _probe(spec_str: str) -> str | None:
    """Return an error string, or ``None`` when the capability is present."""
    modname, _, sym = spec_str.partition(":")
    try:
        m = importlib.import_module(modname)
    except Exception as ex:                                   # noqa: BLE001
        return f"{spec_str} -> IMPORT FAILED {type(ex).__name__}: {ex}"
    if sym and not hasattr(m, sym):
        return (f"{spec_str} -> module resolved from "
                f"{getattr(m, '__file__', '?')} has NO attribute {sym!r} "
                f"(a PRE-v5 tree looks exactly like this)")
    return None


def assert_stack(expected=None, *, require=(), label: str = "") -> dict:
    """⛔ The explicit, callable guard. Returns the report; raises on violation.

    ``require`` accepts ``"v5"`` (⇒ :data:`V5_CAPABILITIES`) or an iterable of
    ``"module:symbol"`` strings."""
    root, prov = resolve_intended_stack(expected)
    if root is None:
        raise StackShadowError(
            "no `tanitad` stack is resolvable at all — set "
            f"{ENV_OVERRIDE}=<stack dir> or put one on PYTHONPATH.")
    # ⚠️ Put the named root AND its scripts/ on the path before probing, or a
    # capability like `train_flagship_v4:resolve_v2_frames` reports IMPORT
    # FAILED merely because scripts/ was absent — a guard failing for the wrong
    # reason, which is its own way of costing a run.
    ensure_stack_on_path(explicit=root)
    bad: list[str] = []
    try:
        tanitad = importlib.import_module(_GUARDED_TOP)
    except StackShadowError:
        raise
    except Exception as ex:                                   # noqa: BLE001
        raise StackShadowError(f"`import tanitad` failed under root {root}: "
                               f"{type(ex).__name__}: {ex}") from ex
    got = _root_of_module(tanitad)
    if got and not _under(got, root):
        bad.append(f"tanitad -> {tanitad.__file__} (root {got})")
    caps = V5_CAPABILITIES if require == "v5" else tuple(require or ())
    for c in caps:
        err = _probe(c)
        if err:
            bad.append(err)
    rep = report()
    rep["required"] = list(caps)
    rep["label"] = label
    rep["ok"] = not bad
    rep["problems"] = bad
    if bad and mode() == "error":
        raise StackShadowError(
            f"⛔ STACK GUARD REFUSED{' [' + label + ']' if label else ''} — "
            f"pinned root {root} (intent: {prov}); offenders:\n  "
            + "\n  ".join(bad)
            + f"\n⇒ export {ENV_OVERRIDE}={root} BEFORE python3 starts, and make "
              f"sure no stale tree (e.g. {LEGACY_STACK}) precedes it.")
    if bad:
        _eprint("[taniteval.stack_guard] ⚠️ " + "; ".join(bad))
    return rep


def report() -> dict:
    """A bankable record. ⚠️ Always carries ``mode`` so ``off`` is visible."""
    root, prov = resolve_intended_stack()
    t = sys.modules.get(_GUARDED_TOP)
    s = active_sentinel()
    if s is not None:
        # ⚠️ Report what was PINNED, not what re-resolves now: ensure_stack_on_path
        # has already put the root on sys.path, so a fresh resolve would answer
        # "sys.path" for a root that actually came from an editable install or
        # the env var — a provenance that quietly launders itself.
        root, prov = s.root, (s.provenance or prov)
    return {
        "pinned_root": root,
        "provenance": prov,
        "mode": mode(),
        "sentinel_installed": s is not None,
        "sentinel_violations": list(s.violations) if s else [],
        "tanitad_file": getattr(t, "__file__", None) if t else None,
        "legacy_stack": LEGACY_STACK,
        "legacy_present": os.path.isdir(LEGACY_STACK),
        "legacy_is_pinned": bool(root) and _norm(root) == _norm(LEGACY_STACK),
        f"env:{ENV_OVERRIDE}": os.environ.get(ENV_OVERRIDE),
        f"env:{ENV_MODE}": os.environ.get(ENV_MODE),
        "sys_path_head": list(sys.path[:5]),
    }


def main(argv=None) -> int:
    """``python3 -m taniteval.stack_guard --stack <dir> --require v5``.

    Exit 0 = the stack the caller named is the stack that will be imported.
    Exit 2 = it is not. ⭐ Put this in front of an eval command and a stale tree
    costs one second instead of one wrong number."""
    import argparse
    import json
    ap = argparse.ArgumentParser(prog="taniteval.stack_guard")
    ap.add_argument("--stack", help=f"the stack you mean (else {ENV_OVERRIDE} "
                                    f"/ PYTHONPATH / the deployed tree)")
    ap.add_argument("--require", default="", help="'v5' or a comma-separated "
                                                  "list of module:symbol")
    ap.add_argument("--json", help="write the report here")
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    req = a.require if a.require == "v5" else \
        tuple(x for x in a.require.split(",") if x)
    try:
        rep = assert_stack(a.stack, require=req, label=a.label)
        rc = 0 if rep.get("ok") else 2
    except StackShadowError as ex:
        rep = report()
        rep["ok"] = False
        rep["problems"] = [str(ex)]
        _eprint(str(ex))
        rc = 2
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items()
                      if k in ("ok", "pinned_root", "provenance", "mode",
                               "tanitad_file", "problems")}, indent=2))
    return rc


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
