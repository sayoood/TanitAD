"""MM-C12 stack resolution + preflight, shared by the MM-E19 probe set.

⛔ THE DEFECT THIS EXISTS FOR. `latentmotion.py` hardcoded
``sys.path.insert(0, r"C:\\Users\\Admin\\tanitad-mirror\\stack")`` — a tree that
EXISTS on the dev box and whose ``models/v6.py`` matches neither the repo nor G:
(MEASURED 2026-09-01: normalized md5 ``7d6a8183`` vs repo/G:/wt ``38d74671``/
``4156cc7b`` and Thor ``d4bcd90d``). ``load_trunk_auto`` REBUILDS the model from
the checkpoint's config, so the code version is load-bearing: a silent import
from the wrong tree produces numbers that LOOK like results. Same family as the
`taniteval` namespace-shadow trap: once the wrong module is bound, no sys.path
edit undoes it — so the import is verified HERE, eagerly, before anything heavy.

The rules:
  * an EXPLICITLY requested stack that does not hold ``tanitad/__init__.py``
    is a REFUSAL, never a fall-through to candidates;
  * with no request, known-good candidates are tried in order — and
    ``tanitad-mirror`` is deliberately NOT one of them;
  * after import, ``tanitad.__file__`` must resolve INSIDE the chosen stack,
    or the probe refuses (exit 2) — this catches a pre-poisoned interpreter
    and the wrong-tree silent import alike;
  * the resolved ``tanitad.__file__`` is returned so the caller can STAMP it
    into its output (the ``actdiv_thor.py`` idiom; prereg MM-E19 §"TWO PROBE
    DEFECTS", defect 2).

Verified trees for Thor-trained v7tiny/v6-family checkpoints (2026-09-01,
LF-normalized md5 vs Thor ``/home/nvidia/TanitAD/stack``):
  * ``tanitad/eval/v6_probe_trunk.py``  IDENTICAL to Thor
  * ``scripts/eval_flagship_v4.py``     IDENTICAL to Thor
  * ``tanitad/models/v6.py``            dev is a NAV-CONDITIONING SUPERSET of
    Thor's (all additions gated on ``cfg.nav_cond``, default False → identical
    construction + forward for arms without ``--nav-cond``)
  * ``scripts/train_v6_staged.py``      differs (same nav additions); the
    STRICT ``load_state_dict`` is the runtime compatibility check.
"""
from __future__ import annotations

import pathlib
import sys

#: Known-good candidates, in preference order. ⛔ ``tanitad-mirror`` is
#: DELIBERATELY absent — it is the stale third tree MM-C12 is about.
STACK_CANDIDATES = (
    pathlib.Path(r"C:\Users\Admin\tanitad-wt\stack"),   # dev-box RUN mirror
    pathlib.Path("/home/nvidia/TanitAD/stack"),          # Thor (authoritative
                                                         # for Thor-trained ckpts)
)


def refuse(msg: str) -> "NoReturn":  # noqa: F821 - py<3.11 friendly
    print(f"[REFUSED] {msg}", file=sys.stderr)
    print("[REFUSED] pass --stack <dir containing tanitad/__init__.py> "
          "(dev box: C:/Users/Admin/tanitad-wt/stack; Thor: "
          "/home/nvidia/TanitAD/stack). ⛔ NOT tanitad-mirror — that tree is "
          "stale (MM-C12).", file=sys.stderr)
    raise SystemExit(2)


def _holds_tanitad(p: pathlib.Path) -> bool:
    try:
        return (p / "tanitad" / "__init__.py").is_file()
    except OSError:        # a flapping mount (G: Errno 22) reads as absence
        return False


def _repo_stack_near(anchor: pathlib.Path):
    """<repo>/stack derived by walking up from ``anchor`` — the last-resort
    candidate (on G: the import may flake, but a flake is a crash, not a wrong
    number)."""
    for parent in anchor.resolve().parents:
        cand = parent / "stack"
        if _holds_tanitad(cand):
            return cand
    return None


def resolve_stack(requested=None,
                  anchor: pathlib.Path | None = None) -> pathlib.Path:
    """Choose the stack dir. Explicit-but-invalid REFUSES; no fall-through."""
    if requested:
        p = pathlib.Path(requested)
        if not _holds_tanitad(p):
            refuse(f"requested --stack {p} does not hold tanitad/__init__.py")
        return p.resolve()
    for c in STACK_CANDIDATES:
        if _holds_tanitad(c):
            return c.resolve()
    near = _repo_stack_near(anchor or pathlib.Path(__file__))
    if near is not None:
        return near.resolve()
    refuse("no usable tanitad stack found (no --stack given, no known-good "
           "candidate present)")


def preflight_stack(stack: pathlib.Path) -> str:
    """Insert ``stack``, import ``tanitad``, VERIFY the tree, return __file__.

    Exit 2 (never a silent wrong import) on: pre-poisoned interpreter, import
    failure, or resolution outside ``stack``. Also pre-imports the load-bearing
    ``tanitad.eval.v6_probe_trunk`` so a missing module fails in seconds, not
    after the expensive rollout (the `t1_eval.py` trap).
    """
    stack = pathlib.Path(stack).resolve()
    sp = str(stack)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    if "tanitad" in sys.modules:   # bound before we got here — verify, or refuse
        got = pathlib.Path(getattr(sys.modules["tanitad"], "__file__", "") or
                           "<namespace-package: no __file__>")
        try:
            inside = stack in got.resolve().parents
        except OSError:
            inside = False
        if not inside:
            refuse(f"tanitad is ALREADY IMPORTED from {got}, not from the "
                   f"requested stack {stack} — a pre-poisoned interpreter "
                   f"(namespace-shadow class); start a fresh process")
    try:
        import tanitad
    except Exception as e:  # noqa: BLE001 - the refusal must name the cause
        refuse(f"cannot import tanitad from {stack}: {type(e).__name__}: {e}")
    got = pathlib.Path(tanitad.__file__).resolve()
    if stack not in got.parents:
        refuse(f"tanitad imported from the WRONG TREE: {got} "
               f"(requested {stack}) — MM-C12")
    print(f"  [stack] tanitad imported from: {got}", flush=True)
    try:
        import tanitad.eval.v6_probe_trunk  # noqa: F401  (preflight only)
    except Exception as e:  # noqa: BLE001
        refuse(f"stack at {stack} lacks a working "
               f"tanitad.eval.v6_probe_trunk: {type(e).__name__}: {e}")
    return str(got)
