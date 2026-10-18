#!/usr/bin/env python
"""zerograd_gate.py -- POST-TRAINING ZERO-GRADIENT GATE: which trainable parameters NEVER learned?

WHY THIS EXISTS
---------------
CLAUDE.md, "BUILT, TESTED, AND UNREACHABLE FROM ITS CALLER" (2026-09-10/11): a mechanism that is wired
into the model but not into the loss passes every unit test, trains for days, and is discovered only
after the arm is trained (and twice after it was published).  refcv5-v2 carried THREE such heads --
``offset_head`` (6,160 params), ``tac_goal_tok_head`` (11,286) and ``scorer.goal_point`` (1,026, exactly
zero: zero-init + zero gradient = a head that emits the constant origin) -- 18,472 parameters in total.
The durable instrument that found them,
``TanitAD Research Lab/Architecture & Inference/Implementation/incoming/2026-09-10-refcv5v2-zerograd-heads/
probe_zerograd3.py``, was declared "a POST-TRAINING GATE" in CLAUDE.md but lived only in an incoming
folder.  This file is that gate.

THE IDEA (from probe_zerograd3.py)
----------------------------------
The optimizer state lives in the checkpoint.  Adam allocates a parameter's state LAZILY, on its first
non-None ``.grad``.  So a registered parameter with NO state entry never received a gradient in the whole
run (case a).  The checkpoint therefore proves, with no GPU and no data, which parameters were dead.

TWO CASES -- the second is the one the probe's counting argument misses
-----------------------------------------------------------------------
(a) ``NO_STATE``   the parameter has no optimizer-state entry: ``.grad`` was None at every step (the head's
                   output never reached the loss, or the module was detached / never called).
(b) ``ZERO_STATE`` the entry exists but EVERY floating-point state tensor is EXACTLY 0 (``exp_avg`` and
                   ``exp_avg_sq`` for Adam/AdamW): only exact-zero gradients ever arrived.  This is the
                   ``--w-x`` flag with ``default=0.0`` case: ``0.0 * term`` yields a zeros GRAD TENSOR, not
                   None, so Adam allocates state and "357 registered vs 351 with state" (case a alone)
                   reads clean while the head is dead.  Both moments are tested, not just ``exp_avg_sq``,
                   because ``g*g`` underflows to 0 for |g| < ~3e-23 while ``exp_avg`` does not.

NAMING THE PARAMETERS (the only heuristic in this file, and it never decides the verdict)
-----------------------------------------------------------------------------------------
A checkpoint stores optimizer state under integer ids, not names.  The id order is the optimizer's
parameter order; the names live in ``model.state_dict()`` which ALSO holds buffers (BatchNorm statistics,
cached anchors ...) that are not parameters.  probe_zerograd3.py aligned the two with a hard-coded list of
buffer-name fragments (two of them repo-specific) and a greedy shape match.  This gate keeps only torch's
own norm-buffer prior (below) and replaces the greedy match with an exact count:

* If the checkpoint carries ``param_names`` (a list in optimizer-id order -- the trainer can add it for
  free: ``[n for n, p in model.named_parameters()]``), or ``--param-names FILE`` is given, naming is EXACT.
* Otherwise an order-preserving shape alignment of ids against floating-point ``state_dict`` entries is
  solved by dynamic programming that COUNTS ALL consistent alignments.  A parameter is named only if it
  has a single candidate entry across every valid alignment; otherwise ALL candidate names are listed and
  the row is marked ``ambiguous`` (a no-state parameter has no shape of its own, so a same-shaped buffer
  next to it is a genuine ambiguity -- reported, not guessed).  The only built-in prior is that torch's
  norm-layer buffers (``running_mean`` / ``running_var`` / ``num_batches_tracked``) are never parameters
  (``--exclude-entry`` adds repo-specific buffers, ``--no-default-excludes`` removes the prior); it narrows
  names and cannot change a verdict.
* The verdict is computed from the IDS.  Naming can only change what is printed and what ``--allow`` may
  suppress -- and ``--allow`` suppresses a row only if EVERY candidate name matches (fail closed).

EXIT CODES      0 PASS   1 FAIL (untrained parameter / failed control)   2 ERROR (could not evaluate:
                unreadable checkpoint, no optimizer state, optimizer without per-parameter float state).
                An un-evaluable checkpoint is NEVER a PASS.

KNOWN BLIND SPOTS (read before trusting a PASS)
-----------------------------------------------
* A parameter FROZEN BY DESIGN looks exactly like a broken one when the optimizer was built over
  ``model.parameters()`` (it never gets a grad).  The checkpoint cannot tell them apart -- pass
  ``--allow`` for each, so the exemption is explicit and visible in the report.
* Plain Adam with coupled L2 ``weight_decay`` adds ``wd * p`` to the gradient, so a head with a zero LOSS
  gradient still gets a non-zero grad and is NOT flagged by case (b).  AdamW (decoupled; the repo's
  ``--opt dd`` path) and ``weight_decay=0`` (the repo's default Adam) are covered.
* A parameter that received gradients in only SOME steps (conditional heads) is reported under
  ``info.partial_grad`` and does not fail the gate.
* SGD without momentum keeps no per-parameter state at all; the gate then exits 2 rather than flagging
  every parameter.

USAGE
-----
    python stack/scripts/zerograd_gate.py --ckpt run/ckpt.pt [--allow REGEX ...] [--control REGEX ...]
                                          [--param-names FILE.json] [--json-out report.json] [--sha256]

``--allow`` / ``--control`` use ``re.fullmatch`` against the parameter name (``--allow 'core\\.enc\\..*'``)
so an exemption cannot silently widen; an ``--allow`` that matches no offender is reported as unused (a
typo is visible).  ``--control REGEX`` is the same-breath positive control: at least one parameter whose
name fully matches must be TRAINED, otherwise the gate FAILS (a probe that cannot see a head that must
have trained is broken, and a gate that finds nothing must prove it can look).

Standalone: needs only ``torch`` (CPU).  The checkpoint is loaded ``weights_only=True`` + ``mmap`` first
and falls back to ``weights_only=False`` (the probe's mode; our own checkpoints carry arbitrary dicts) --
the mode used is recorded in the report.  ASCII-only output (cp1252-safe on the Windows dev box).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

GATE_VERSION = "1"
EXIT_PASS, EXIT_FAIL, EXIT_ERROR = 0, 1, 2

TRAINED, NO_STATE, ZERO_STATE = "TRAINED", "NO_STATE", "ZERO_STATE"
CASE_LABEL = {NO_STATE: "a_no_state", ZERO_STATE: "b_zero_state"}

_MODEL_KEYS = ("model", "state_dict", "model_state_dict")
_OPT_KEYS = ("opt", "optimizer", "optim", "optimizer_state_dict")
# torch.nn BatchNorm/InstanceNorm own buffers: by construction never parameters, so they are not name candidates.
# This only narrows NAMING (never the verdict); extend with --exclude-entry, drop with --no-default-excludes.
DEFAULT_EXCLUDE_ENTRIES = (r"(.*\.)?(running_mean|running_var|num_batches_tracked)",)
_DP_CELL_LIMIT = 30_000_000          # ids x entries; above this the name alignment is skipped (verdict unaffected)
_CAP = 1e100                          # saturating alignment counts: only ==0, ==1, >1 matter


class GateError(RuntimeError):
    """The checkpoint could not be evaluated (exit 2) -- never silently converted into a PASS.

    ``kind`` classifies WHY, so a caller (``stack/ops/sup_refcv7.sh``) can tell the one benign case -- a checkpoint
    that carries no optimizer state at all (``no_optimizer_state``: the gate does not apply) -- from every case in
    which the gate could not look (``unreadable``, ``no_param_state``, ``bad_param_names`` ...).  The kind is written
    to the ``--json-out`` report as ``error_kind``; the verdict is the JSON, never the exit code.
    """

    def __init__(self, msg: str, kind: str = "other"):
        super().__init__(msg)
        self.kind = kind


# ------------------------------------------------------------------------------------------------ #
# loading                                                                                          #
# ------------------------------------------------------------------------------------------------ #
def load_ckpt(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """CPU load.  weights_only=True+mmap first, then the probe's weights_only=False.  Returns (ckpt, how)."""
    import torch
    if not os.path.isfile(path):
        raise GateError("checkpoint not found: %s" % path, kind="unreadable")
    errors = []
    for weights_only, mmap in ((True, True), (False, True), (False, False)):
        try:
            ck = torch.load(path, map_location="cpu", weights_only=weights_only, mmap=mmap)
        except Exception as e:  # noqa: BLE001 -- every load failure is recorded, then the next mode is tried
            errors.append("weights_only=%s mmap=%s: %s: %s" % (weights_only, mmap, type(e).__name__, str(e)[:160]))
            continue
        if not isinstance(ck, dict):
            raise GateError("checkpoint is a %s, not a dict" % type(ck).__name__, kind="unreadable")
        return ck, {"weights_only": weights_only, "mmap": mmap, "earlier_load_failures": errors}
    raise GateError("could not load %s: %s" % (path, " | ".join(errors)), kind="unreadable")


def _pick(ck: Dict[str, Any], keys: Sequence[str]) -> Tuple[Optional[str], Optional[dict]]:
    for k in keys:
        if isinstance(ck.get(k), dict):
            return k, ck[k]
    return None, None


# ------------------------------------------------------------------------------------------------ #
# the two checks                                                                                   #
# ------------------------------------------------------------------------------------------------ #
def _float_state_tensors(entry: Any) -> List[Any]:
    import torch
    if not isinstance(entry, dict):
        return []
    return [v for k, v in entry.items() if k != "step" and torch.is_tensor(v) and v.is_floating_point()]


def classify_state(entry: Any) -> str:
    """TRAINED / NO_STATE (case a) / ZERO_STATE (case b) for ONE optimizer-state entry (None if absent).

    ``count_nonzero`` counts NaN as non-zero, so a diverged (NaN) state is never mistaken for 'zero grads'.
    """
    import torch
    ts = _float_state_tensors(entry)
    if not ts:                                           # absent, empty, or only a step counter
        return NO_STATE
    if all(int(torch.count_nonzero(t)) == 0 for t in ts):
        return ZERO_STATE
    return TRAINED


def _state_shape(entry: Any) -> Optional[Tuple[int, ...]]:
    ts = _float_state_tensors(entry)
    if not ts:
        return None
    if "exp_avg" in entry and hasattr(entry["exp_avg"], "shape"):
        return tuple(entry["exp_avg"].shape)
    shapes = {tuple(t.shape) for t in ts}
    return shapes.pop() if len(shapes) == 1 else None


def _step_of(entry: Any) -> Optional[int]:
    try:
        s = entry.get("step") if isinstance(entry, dict) else None
        return None if s is None else int(float(s))
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------------------------------------ #
# naming: order-preserving shape alignment with ALL-ALIGNMENTS counting                             #
# ------------------------------------------------------------------------------------------------ #
def align_candidates(id_shapes: Sequence[Optional[Tuple[int, ...]]],
                     entry_shapes: Sequence[Tuple[int, ...]]) -> Tuple[float, List[List[int]]]:
    """Count order-preserving injective maps ids -> entries that respect every KNOWN shape, and return for
    each id the set of entry indices it takes in at least one such map.

    ``id_shapes[i]`` is None for an id with no state (shape unknown: matches anything).
    Returns (n_alignments, candidates) -- n_alignments is 0 when none exists, 1 when the naming is unique
    (saturating float: only 0 / 1 / >1 are meaningful).  Forward/backward DP, O(n*m) vectorised per row.
    """
    import numpy as np
    n, m = len(id_shapes), len(entry_shapes)
    if n == 0:
        return 1.0, []
    if n > m:
        return 0.0, [[] for _ in range(n)]
    sid: Dict[Tuple[int, ...], int] = {}
    ent = np.array([sid.setdefault(tuple(s), len(sid)) for s in entry_shapes], dtype=np.int64)
    compat = np.ones((n, m), dtype=bool)
    for i, s in enumerate(id_shapes):
        if s is not None:
            compat[i] = (ent == sid[tuple(s)]) if tuple(s) in sid else False
    F = np.zeros((n, m))
    F[0] = compat[0]
    for i in range(1, n):
        pre = np.concatenate(([0.0], np.cumsum(F[i - 1])[:-1]))      # sum over j' < j
        F[i] = np.minimum(compat[i] * pre, _CAP)
    S = np.zeros((n + 1, m + 1))
    S[n, :] = 1.0
    for i in range(n - 1, -1, -1):
        t = compat[i] * S[i + 1, 1:]                                 # id i at entry j, rest after j
        S[i, :m] = np.minimum(np.cumsum(t[::-1])[::-1], _CAP)
    total = float(S[0, 0])
    if total <= 0.0:
        return 0.0, [[] for _ in range(n)]
    through = F * S[1:, 1:]
    return total, [np.nonzero(through[i] > 0)[0].tolist() for i in range(n)]


def _prod(shape: Sequence[int]) -> int:
    return int(math.prod(shape))


def resolve_names(ids_by_group: List[List[Any]], id_shape: Dict[Any, Optional[tuple]],
                  entries: List[Tuple[str, tuple]], param_names: Optional[Any]) -> Dict[str, Any]:
    """Map optimizer ids -> candidate parameter names.  Returns {'mode', 'unique', 'n_alignments',
    'cands': {id: [entry idx ...]} | None, 'names': {id: [name ...]} | None, 'note': str}."""
    ids = [i for g in ids_by_group for i in g]
    out: Dict[str, Any] = {"mode": "failed", "unique": False, "n_alignments": 0.0, "names": None, "note": ""}
    if param_names is not None:
        if isinstance(param_names, dict):
            names = {i: [str(param_names[i])] for i in ids if i in param_names}
            names.update({i: [str(param_names[str(i)])] for i in ids if i not in names and str(i) in param_names})
        else:
            if len(param_names) != len(ids):
                raise GateError("param_names has %d entries but the optimizer registers %d parameters"
                                % (len(param_names), len(ids)), kind="bad_param_names")
            names = {i: [str(n)] for i, n in zip(ids, param_names)}
        if len(names) != len(ids):
            raise GateError("param_names does not cover every optimizer id", kind="bad_param_names")
        shape_of = dict(entries)
        for i in ids:
            sh, nm = id_shape.get(i), names[i][0]
            if sh is not None and nm in shape_of and tuple(shape_of[nm]) != tuple(sh):
                raise GateError("param_names INCONSISTENT with the checkpoint: id %s named %r has state shape %s "
                                "but the model entry is %s" % (i, nm, tuple(sh), tuple(shape_of[nm])),
                                kind="bad_param_names")
        out.update(mode="param_names", unique=True, n_alignments=1.0, names=names)
        return out
    if len(ids) * len(entries) > _DP_CELL_LIMIT:
        out["note"] = "name alignment skipped: %d ids x %d entries is too large (verdict unaffected)" % (
            len(ids), len(entries))
        return out
    ent_shapes = [s for _, s in entries]
    ent_names = [n for n, _ in entries]
    total, cands = align_candidates([id_shape.get(i) for i in ids], ent_shapes)
    if total > 0:
        out.update(mode="global-shape-dp", unique=(total == 1.0), n_alignments=total,
                   names={i: [ent_names[j] for j in c] for i, c in zip(ids, cands)})
        return out
    # Fall back to one alignment per optimizer group (e.g. the --opt dd groups list the encoder FIRST, which
    # is not state_dict order).  Each group is an order-preserving subsequence of the entries.
    names: Dict[Any, List[str]] = {}
    unique, tot = True, 1.0
    for g in ids_by_group:
        t, c = align_candidates([id_shape.get(i) for i in g], ent_shapes)
        if t <= 0:
            out["note"] = "no order-preserving shape alignment exists (global or per group); names unavailable"
            return out
        unique, tot = unique and t == 1.0, min(tot * t, _CAP)
        names.update({i: [ent_names[j] for j in cj] for i, cj in zip(g, c)})
    taken: Dict[str, Any] = {}
    for i, nm in names.items():                                  # two ids claiming the same entry => inconsistent
        if len(nm) == 1:
            if nm[0] in taken and taken[nm[0]] != i:
                out["note"] = "per-group alignments collide on %r; names unavailable" % nm[0]
                return out
            taken[nm[0]] = i
    out.update(mode="per-group-shape-dp", unique=unique, n_alignments=tot, names=names)
    return out


# ------------------------------------------------------------------------------------------------ #
# the analysis                                                                                     #
# ------------------------------------------------------------------------------------------------ #
def _allowed_by(patterns: Sequence[str], names: Sequence[str]) -> Optional[List[str]]:
    """The patterns that exempt a row, or None.  EVERY candidate name must be fully matched by some pattern
    (fail closed: an ambiguous row is exempt only if it is exempt whichever candidate it really is)."""
    if not names:
        return None
    used: List[str] = []
    for n in names:
        hit = next((p for p in patterns if re.fullmatch(p, n)), None)
        if hit is None:
            return None
        if hit not in used:
            used.append(hit)
    return used


def analyze(ck: Dict[str, Any], allow: Sequence[str] = (), controls: Sequence[str] = (),
            param_names: Optional[Any] = None,
            exclude_entries: Sequence[str] = DEFAULT_EXCLUDE_ENTRIES) -> Dict[str, Any]:
    """Pure analysis of an already-loaded checkpoint dict.  Raises GateError if it cannot be evaluated."""
    import torch
    mk, sd = _pick(ck, _MODEL_KEYS)
    ok, opt = _pick(ck, _OPT_KEYS)
    if opt is None or "param_groups" not in opt or not isinstance(opt.get("state"), dict):
        raise GateError("the checkpoint carries no optimizer state (keys %s) -- a weights-only file cannot be "
                        "gated; run the gate on the TRAINER's resume checkpoint" % sorted(map(str, ck.keys())),
                        kind="no_optimizer_state")
    # Exact names, in order of preference: --param-names FILE (explicit), then the checkpoint's own ``param_names``
    # (written by refc_v3_train.py since 2026-10-18: optimizer-id order), then the shape alignment below.
    pn_source = "file" if param_names is not None else None
    if param_names is None and ck.get("param_names") is not None:
        param_names, pn_source = ck["param_names"], "ckpt"
    state = opt["state"]
    ids_by_group = [list(g["params"]) for g in opt["param_groups"]]
    ids = [i for g in ids_by_group for i in g]
    if not ids:
        raise GateError("the optimizer registers no parameters", kind="no_params")
    if not any(_float_state_tensors(state.get(i)) for i in ids):
        raise GateError("no registered parameter has a floating-point optimizer state tensor (plain SGD without "
                        "momentum?) -- the gate cannot tell trained from untrained and refuses to guess",
                        kind="no_param_state")

    entries: List[Tuple[str, tuple]] = []
    n_excluded = 0
    if sd is not None:
        for n, t in sd.items():
            if not (torch.is_tensor(t) and (t.is_floating_point() or t.is_complex())):
                continue
            if any(re.fullmatch(x, str(n)) for x in exclude_entries):
                n_excluded += 1
                continue
            entries.append((str(n), tuple(t.shape)))
    elif param_names is None:
        raise GateError("no model state_dict in the checkpoint and no param_names: nothing to name parameters from",
                        kind="no_names")

    kind = {i: classify_state(state.get(i)) for i in ids}
    id_shape = {i: _state_shape(state.get(i)) for i in ids}
    pn_rejected: Optional[str] = None
    try:
        res = resolve_names(ids_by_group, id_shape, entries, param_names)
    except GateError as e:
        # An EMBEDDED list that does not fit this optimizer (e.g. a checkpoint rewritten after it was saved) must
        # not block the gate: the verdict never depends on naming.  Fall back to the shape alignment and SAY SO.
        # An EXPLICIT --param-names file that does not fit stays an error -- the caller asked for exactness.
        if pn_source != "ckpt" or e.kind != "bad_param_names":
            raise
        pn_rejected, pn_source = str(e), None
        res = resolve_names(ids_by_group, id_shape, entries, None)
    names: Optional[Dict[Any, List[str]]] = res["names"]
    shape_of = dict(entries)

    steps = [s for s in (_step_of(state.get(i)) for i in ids) if s is not None]
    step_max = max(steps) if steps else None

    rows: List[Dict[str, Any]] = []
    for i in ids:
        if kind[i] == TRAINED:
            continue
        cand = names.get(i, []) if names else []
        sh = id_shape[i]
        if sh is not None:
            lo = hi = _prod(sh)
        else:
            sizes = [_prod(shape_of[n]) for n in cand if n in shape_of]
            lo, hi = (min(sizes), max(sizes)) if sizes else (None, None)
        allowed_by = _allowed_by(allow, cand) if allow else None
        rows.append({"id": int(i) if isinstance(i, int) else i, "case": CASE_LABEL[kind[i]], "names": cand,
                     "ambiguous": len(cand) > 1, "state_shape": None if sh is None else list(sh),
                     "numel_lo": lo, "numel_hi": hi, "allowed_by": allowed_by})
    offenders = [r for r in rows if r["allowed_by"] is None]
    allowed = [r for r in rows if r["allowed_by"] is not None]
    used = {p for r in allowed for p in r["allowed_by"]}
    unused_allow = [p for p in allow if p not in used]

    def _tot(rs, key):
        vals = [r[key] for r in rs]
        return None if any(v is None for v in vals) else int(sum(vals))

    # same-breath positive controls: a parameter that MUST have trained, and did
    named = names or {}
    trained_names = [named[i][0] for i in ids if kind[i] == TRAINED and len(named.get(i, [])) == 1]
    control_rows = []
    for p in controls:
        hits = [n for n in trained_names if re.fullmatch(p, n)]
        control_rows.append({"pattern": p, "trained_matches": len(hits), "example": hits[:3],
                             "ok": len(hits) > 0})

    partial = []
    if step_max:
        partial = [{"id": int(i) if isinstance(i, int) else i, "step": _step_of(state.get(i)),
                    "fraction": round(_step_of(state.get(i)) / step_max, 4),
                    "names": named.get(i, [])}
                   for i in ids if kind[i] == TRAINED and _step_of(state.get(i)) is not None
                   and _step_of(state.get(i)) < step_max]

    failed_controls = [c for c in control_rows if not c["ok"]]
    verdict = "FAIL" if (offenders or failed_controls) else "PASS"
    return {
        "gate": "zerograd_gate", "gate_version": GATE_VERSION, "verdict": verdict,
        "model_key": mk, "optimizer_key": ok,
        "optimizer": {"n_groups": len(ids_by_group), "n_registered": len(ids),
                      "n_with_state_entry": sum(1 for i in ids if i in state),
                      "n_trained": sum(1 for i in ids if kind[i] == TRAINED),
                      "n_case_a_no_state": sum(1 for i in ids if kind[i] == NO_STATE),
                      "n_case_b_zero_state": sum(1 for i in ids if kind[i] == ZERO_STATE),
                      "step_max": step_max, "ckpt_step": ck.get("step") if isinstance(ck.get("step"), int) else None},
        "alignment": {"mode": res["mode"], "unique": res["unique"], "n_alignments": res["n_alignments"],
                      "param_names_source": pn_source, "param_names_rejected": pn_rejected,
                      "n_model_float_entries": len(entries), "n_entries_excluded": n_excluded,
                      "exclude_entries": list(exclude_entries), "note": res["note"],
                      "n_ids_name_ambiguous": sum(1 for i in ids if len(named.get(i, [])) > 1),
                      "n_ids_unnamed": sum(1 for i in ids if not named.get(i))},
        "offenders": offenders, "allowed": allowed, "unused_allow": unused_allow,
        "untrained_numel": {"offending_lo": _tot(offenders, "numel_lo"), "offending_hi": _tot(offenders, "numel_hi"),
                            "allowed_lo": _tot(allowed, "numel_lo"), "allowed_hi": _tot(allowed, "numel_hi")},
        "controls": control_rows,
        "info": {"partial_grad": partial},
    }


# ------------------------------------------------------------------------------------------------ #
# CLI                                                                                              #
# ------------------------------------------------------------------------------------------------ #
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def _fmt_names(r: Dict[str, Any]) -> str:
    if not r["names"]:
        return "<unnamed: no usable alignment>"
    if len(r["names"]) == 1:
        return r["names"][0]
    return "AMBIGUOUS{%s}" % " | ".join(r["names"])


def _fmt_numel(r: Dict[str, Any]) -> str:
    lo, hi = r["numel_lo"], r["numel_hi"]
    if lo is None:
        return "numel=?"
    return "numel=%d" % lo if lo == hi else "numel=%d..%d" % (lo, hi)


def render(rep: Dict[str, Any]) -> str:
    o, a, u = rep["optimizer"], rep["alignment"], rep["untrained_numel"]
    L = ["zerograd_gate v%s  ckpt=%s" % (rep["gate_version"], rep.get("ckpt", "<in-memory>")),
         "  registered=%d  with-state-entry=%d  trained=%d  case(a) no-state=%d  case(b) zero-state=%d  "
         "opt-step-max=%s" % (o["n_registered"], o["n_with_state_entry"], o["n_trained"], o["n_case_a_no_state"],
                              o["n_case_b_zero_state"], o["step_max"]),
         "  naming: %s  unique=%s  n_alignments=%s%s" % (
             a["mode"], a["unique"], ("%.0f" % a["n_alignments"]) if a["n_alignments"] < 1e15 else ">1e15",
             ("  [" + a["note"] + "]") if a["note"] else "")]
    if a.get("param_names_rejected"):
        L.append("  WARN the checkpoint's embedded param_names were REJECTED (%s); names come from the shape alignment"
                 % a["param_names_rejected"][:200])
    for r in rep["offenders"]:
        L.append("  UNTRAINED  id %-4s %-12s %-40s %s" % (r["id"], r["case"], _fmt_names(r), _fmt_numel(r)))
    for r in rep["allowed"]:
        L.append("  allowed    id %-4s %-12s %-40s %s  (by %s)" % (r["id"], r["case"], _fmt_names(r),
                                                                   _fmt_numel(r), ", ".join(r["allowed_by"])))
    for p in rep["unused_allow"]:
        L.append("  WARN unused --allow pattern (matched no offender): %r" % p)
    for c in rep["controls"]:
        L.append("  control %-40r %s (%d trained matches)" % (c["pattern"], "ok" if c["ok"] else "FAILED",
                                                              c["trained_matches"]))
    L.append("  untrained numel (not allowed): %s..%s   allowed: %s..%s" % (
        u["offending_lo"], u["offending_hi"], u["allowed_lo"], u["allowed_hi"]))
    if rep["info"]["partial_grad"]:
        L.append("  info: %d trained parameters got grads in only part of the run (see --json-out info.partial_grad)"
                 % len(rep["info"]["partial_grad"]))
    L.append("VERDICT: %s" % rep["verdict"])
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description="Post-training zero-gradient gate (see module docstring).")
    ap.add_argument("--ckpt", required=True, help="trainer checkpoint holding model + optimizer state")
    ap.add_argument("--allow", action="append", default=[], metavar="REGEX",
                    help="exempt parameters whose full name re.fullmatch-es REGEX (repeatable; e.g. frozen-by-design)")
    ap.add_argument("--control", action="append", default=[], metavar="REGEX",
                    help="positive control: >=1 TRAINED parameter must fully match REGEX (repeatable)")
    ap.add_argument("--param-names", default=None, metavar="FILE.json",
                    help="JSON list (optimizer-id order) or {id: name} -- makes naming exact")
    ap.add_argument("--exclude-entry", action="append", default=[], metavar="REGEX",
                    help="state_dict entry names (re.fullmatch) that are buffers, not parameters: never name candidates "
                         "(repeatable; BatchNorm running stats are excluded by default)")
    ap.add_argument("--no-default-excludes", action="store_true", help="do not exclude BatchNorm buffers by default")
    ap.add_argument("--json-out", default=None, help="write the full report here")
    ap.add_argument("--sha256", action="store_true", help="record the checkpoint's sha256 in the report")
    a = ap.parse_args(argv)
    try:
        ck, how = load_ckpt(a.ckpt)
        pn = None
        if a.param_names:
            with open(a.param_names, encoding="utf-8") as f:
                pn = json.load(f)
        excl = ([] if a.no_default_excludes else list(DEFAULT_EXCLUDE_ENTRIES)) + list(a.exclude_entry)
        rep = analyze(ck, allow=a.allow, controls=a.control, param_names=pn, exclude_entries=excl)
    except Exception as e:  # noqa: BLE001 -- GateError AND any crash: an uncaught exception exits 1 == FAIL
        kind = e.kind if isinstance(e, GateError) else "internal_error"
        if not isinstance(e, GateError):
            import traceback
            traceback.print_exc()
        print("zerograd_gate ERROR (%s): %s" % (kind, e), file=sys.stderr)
        if a.json_out:
            with open(a.json_out, "w", encoding="utf-8") as f:
                json.dump({"gate": "zerograd_gate", "gate_version": GATE_VERSION, "verdict": "ERROR",
                           "ckpt": a.ckpt, "error": str(e), "error_kind": kind}, f, indent=2)
        return EXIT_ERROR
    rep["ckpt"] = a.ckpt.replace("\\", "/")
    rep["ckpt_size_bytes"] = os.path.getsize(a.ckpt)
    rep["load"] = how
    if a.sha256:
        rep["ckpt_sha256"] = _sha256(a.ckpt)
    print(render(rep))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            json.dump(rep, f, indent=2, ensure_ascii=True)
    return EXIT_PASS if rep["verdict"] == "PASS" else EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
