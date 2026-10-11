"""Weight EMA and the B1 guard (refcv8 R8-WARMUP, Master Mind ruling; ``SPEC_REFCV8_DRAFT.md`` WP-D sec. 2.7).

THE RULING (``Project Steering/SPEC_REFCV8_DRAFT.md``, R8-WARMUP + the B1 bullet of the entry rules): re-warm both LR
groups over 1,000 steps to refcv7's peaks, cosine over the run, **EMA 0.9998 evaluated at every in-run eval and
saved**, and the **B1 guard (literal)**: *if the EMA box3d AP@2 m on the in-run eval set at refcv8 step 1,000 is below
refcv7-50,400's value on the same set, the head re-warm is cut to the trunk's 5e-5.*

This module is the part of that ruling that is NOT a hyper-parameter: the shadow weights, the swap that evaluates
them, and the guard's decision. ``scripts/refc_v3_train.py`` wires it (``--ema-decay``, ``--b1-guard-ref-ap2m``,
``--b1-guard-step``).

WHAT THE SHADOW HOLDS
    One tensor per TRAINABLE parameter and per floating-point BUFFER of ``model.state_dict()`` (de-duplicated by object
    identity, so a tied weight is shadowed once), kept in fp32 (fp64 stays fp64) whatever dtype the live tensor has.
    Frozen parameters and integer buffers are not shadowed: they never change, so their EMA is themselves. Memory is
    ``numel * 4`` bytes (``WeightEMA.nbytes``), on the same device as the live tensor.
    ``ema <- ema + (1 - d) * (p - ema)`` (``lerp``), which equals ``d * ema + (1 - d) * p`` and, unlike the written-out
    form, leaves a CONSTANT tensor exactly constant (``lerp(x, x, w) == x`` bit for bit; ``d*x + (1-d)*x`` can move
    by an ulp, and 50,000 updates of it would drift every frozen-in-practice buffer).

THE SWAP
    ``swapped_in(model)`` EXCHANGES live and shadow tensor by tensor (one tensor-sized temporary, no second model on
    the device): inside the block the model holds the EMA weights and the shadow holds the live ones; on exit the
    exchange is repeated and every tensor is back **bit for bit** (pinned by ``tests/test_refcv8_ema_guard.py``). A
    live tensor whose dtype differs from its shadow (a bf16 parameter) cannot be exchanged losslessly, so those are
    stashed instead of exchanged: the stash keeps the live value and the shadow is never overwritten by a rounded copy.
    ``update`` and ``state_dict`` REFUSE while swapped in - the shadow then holds live weights.

THE GUARD
    ``b1_guard_apply`` runs at the FIRST in-run eval row at or after ``guard_step`` that carries the EMA metric
    (``ema_eval_box3d_ap2m``). ``ema < ref`` (strict) -> the head group's peak LR (``initial_lr``, which the schedule
    multiplies - the cosine continues from there) is set to the trunk group's peak. It is decided ONCE: the state is
    saved in ``ckpt.pt`` so a resumed run neither re-decides nor forgets. A missing or non-finite metric REFUSES - a
    guard that cannot read its input does not guess.
"""
from __future__ import annotations

import contextlib
import math
from typing import Optional

import torch
from torch import Tensor

__all__ = ["SCHEMA", "B1_METRIC_KEY", "B1_DEFAULT_STEP", "WeightEMA", "EMAStateError", "B1GuardError",
           "b1_guard_state", "b1_guard_decide", "b1_head_cut", "b1_guard_apply"]

SCHEMA = "weight_ema/1"
#: the key the in-run eval row carries for the EMA weights' box3d AP@2 m (class-agnostic, raw, no NMS):
#: ``eval_box3d_ap2m`` (``tanitad/eval/detection_metrics.py`` ``summarise``, == ``eval_box3d_det_ap2_all_all``)
#: under the ``ema_`` prefix.
B1_METRIC_KEY = "ema_eval_box3d_ap2m"
B1_DEFAULT_STEP = 1000


class EMAStateError(ValueError):
    """A shadow that cannot be used as asked: wrong keys / shapes / decay on load, or used while swapped in."""


class B1GuardError(ValueError):
    """The guard cannot be decided (metric absent or not finite) or is not wired to a head/encoder pair."""


def _shadow_dtype(t: Tensor) -> torch.dtype:
    return t.dtype if t.dtype in (torch.float32, torch.float64) else torch.float32


def _live_tensors(model) -> dict:
    """name -> live tensor for everything the EMA covers. De-duplicated by OBJECT identity (a tied parameter appears
    under several state_dict names and must be exchanged once: exchanging it twice would hand the second shadow the
    first's EMA value and the swap-out would restore the wrong thing)."""
    out, seen = {}, set()
    for name, t in model.state_dict(keep_vars=True).items():
        if id(t) in seen:
            continue
        covered = t.requires_grad if isinstance(t, torch.nn.Parameter) else t.is_floating_point()
        if not covered:
            continue
        seen.add(id(t))
        out[name] = t
    return out


class WeightEMA:
    """Exponential moving average of a model's trainable weights. See the module docstring."""

    def __init__(self, model, decay: float) -> None:
        decay = float(decay)
        if not (0.0 < decay < 1.0) or not math.isfinite(decay):
            raise EMAStateError(f"EMA decay {decay!r} must be in (0, 1)")
        live = _live_tensors(model)
        if not live:
            raise EMAStateError("the model has no trainable parameter and no floating buffer to average")
        self.decay = decay
        self.num_updates = 0
        self.shadow = {n: t.detach().to(_shadow_dtype(t)).clone() for n, t in live.items()}
        self._swapped = False
        self._stash: dict = {}

    @property
    def swapped(self) -> bool:
        """True while the model holds the EMA weights (between ``swap_in`` and ``swap_out``)."""
        return self._swapped

    # ------------------------------------------------------------------ size
    @property
    def numel(self) -> int:
        return sum(s.numel() for s in self.shadow.values())

    @property
    def nbytes(self) -> int:
        return sum(s.numel() * s.element_size() for s in self.shadow.values())

    # ---------------------------------------------------------------- guards
    def _live_checked(self, model) -> dict:
        live = _live_tensors(model)
        if set(live) != set(self.shadow):
            missing = sorted(set(self.shadow) - set(live))[:3]
            extra = sorted(set(live) - set(self.shadow))[:3]
            raise EMAStateError(
                "the model's trainable/float tensors are not the ones the EMA was built from "
                f"(shadow-only e.g. {missing}, model-only e.g. {extra}) - a param was (un)frozen or renamed")
        return live

    # ---------------------------------------------------------------- update
    @torch.no_grad()
    def update(self, model) -> None:
        """One EMA step. Call right after ``optimizer.step()``."""
        if self._swapped:
            raise EMAStateError("update() while swapped in: the shadow holds the LIVE weights right now")
        live = self._live_checked(model)
        w = 1.0 - self.decay
        fast_s, fast_l = [], []
        for n, t in live.items():
            s = self.shadow[n]
            if t.dtype == s.dtype and t.device == s.device:
                fast_s.append(s)
                fast_l.append(t.detach())
            else:
                s.lerp_(t.detach().to(device=s.device, dtype=s.dtype), w)
        if fast_s:
            torch._foreach_lerp_(fast_s, fast_l, w)
        self.num_updates += 1

    # ------------------------------------------------------------------ swap
    @torch.no_grad()
    def _exchange(self, name: str, t: Tensor) -> None:
        s = self.shadow[name]
        tmp = t.detach().clone()
        t.copy_(s)
        s.copy_(tmp)

    @torch.no_grad()
    def swap_in(self, model) -> None:
        """Put the EMA weights into the model (live <-> shadow, tensor by tensor)."""
        if self._swapped:
            raise EMAStateError("swap_in() while already swapped in")
        live = self._live_checked(model)
        done, stash = [], {}
        try:
            for n, t in live.items():
                s = self.shadow[n]
                if t.dtype == s.dtype and t.device == s.device:
                    self._exchange(n, t)
                else:
                    stash[n] = t.detach().clone()          # a lossy cast must not overwrite the shadow
                    t.copy_(s.to(device=t.device, dtype=t.dtype))
                done.append(n)
        except BaseException:
            for n in reversed(done):                        # roll back what was exchanged, then re-raise
                t = live[n]
                if n in stash:
                    t.copy_(stash[n])
                else:
                    self._exchange(n, t)
            raise
        self._stash, self._swapped = stash, True

    @torch.no_grad()
    def swap_out(self, model) -> None:
        """Undo :meth:`swap_in`: the live weights come back bit for bit and the shadow is the EMA again."""
        if not self._swapped:
            raise EMAStateError("swap_out() without a swap_in()")
        live = self._live_checked(model)
        for n, t in live.items():
            if n in self._stash:
                t.copy_(self._stash[n])
            else:
                self._exchange(n, t)
        self._stash, self._swapped = {}, False

    @contextlib.contextmanager
    def swapped_in(self, model):
        """``with ema.swapped_in(model): ...`` - EMA weights inside, live weights restored on ANY exit."""
        self.swap_in(model)
        try:
            yield model
        finally:
            self.swap_out(model)

    # ----------------------------------------------------------------- state
    def state_dict(self) -> dict:
        """What ``ckpt.pt['ema']`` carries. The shadow tensors are returned by reference: ``torch.save`` them at once."""
        if self._swapped:
            raise EMAStateError("state_dict() while swapped in: the shadow holds the LIVE weights right now")
        return {"schema": SCHEMA, "decay": self.decay, "num_updates": int(self.num_updates), "shadow": self.shadow}

    @torch.no_grad()
    def load_state_dict(self, sd: dict, model=None) -> None:
        if self._swapped:
            raise EMAStateError("load_state_dict() while swapped in")
        if not isinstance(sd, dict) or sd.get("schema") != SCHEMA:
            raise EMAStateError(f"not a {SCHEMA} EMA state: schema {None if not isinstance(sd, dict) else sd.get('schema')!r}")
        if float(sd["decay"]) != self.decay:
            raise EMAStateError(f"the checkpoint's EMA decay {sd['decay']!r} != this run's {self.decay!r}")
        src = sd["shadow"]
        if set(src) != set(self.shadow):
            raise EMAStateError(
                f"the checkpoint's EMA covers {len(src)} tensors, this model {len(self.shadow)}; "
                f"checkpoint-only e.g. {sorted(set(src) - set(self.shadow))[:3]}, "
                f"model-only e.g. {sorted(set(self.shadow) - set(src))[:3]}")
        for n, v in src.items():
            if tuple(v.shape) != tuple(self.shadow[n].shape):
                raise EMAStateError(f"EMA tensor {n}: checkpoint shape {tuple(v.shape)} != {tuple(self.shadow[n].shape)}")
        if model is not None:
            self._live_checked(model)
        for n, v in src.items():
            self.shadow[n].copy_(v)
        self.num_updates = int(sd["num_updates"])

    def model_state_dict(self, model) -> dict:
        """``model.state_dict()`` with every shadowed entry replaced by its EMA value (cast to the live dtype). The
        result ALIASES the shadow for fp32 tensors - save it at once. This is the 'model-only' EMA file's payload."""
        if self._swapped:
            raise EMAStateError("model_state_dict() while swapped in")
        live = self._live_checked(model)
        id2name = {id(t): n for n, t in live.items()}
        out = {}
        for k, t in model.state_dict(keep_vars=True).items():
            n = id2name.get(id(t))
            out[k] = (t.detach() if n is None
                      else self.shadow[n].detach().to(device=t.device, dtype=t.dtype))
        return out


# =================================================================== the B1 guard
def b1_guard_state(ref_ap2m: float, guard_step: int = B1_DEFAULT_STEP) -> dict:
    """The guard's persistent state (JSON-able; saved in ``ckpt.pt['b1_guard']``)."""
    ref = float(ref_ap2m)
    if not math.isfinite(ref) or not (0.0 < ref <= 1.0):
        raise B1GuardError(f"--b1-guard-ref-ap2m {ref_ap2m!r}: an AP is in (0, 1]")
    if int(guard_step) < 1:
        raise B1GuardError(f"--b1-guard-step {guard_step!r} must be >= 1")
    return {"armed": True, "ref_ap2m": ref, "guard_step": int(guard_step), "metric": B1_METRIC_KEY,
            "decided": False, "decision": None}


def b1_guard_decide(ema_ap2m, ref_ap2m: float) -> str:
    """``'cut'`` iff the EMA's box3d AP@2 m is STRICTLY below the reference, else ``'keep'``. A missing or non-finite
    EMA value is not a number the rule can be applied to: it raises."""
    if ema_ap2m is None or not math.isfinite(float(ema_ap2m)):
        raise B1GuardError(f"{B1_METRIC_KEY} is {ema_ap2m!r}: the guard cannot be decided and will not guess")
    return "cut" if float(ema_ap2m) < float(ref_ap2m) else "keep"


def b1_head_cut(opt) -> tuple:
    """Cut the head group's PEAK lr to the trunk's: ``head['initial_lr'] = min(head, encoder)``. ``initial_lr`` is what
    ``apply_lr_schedule`` multiplies by the cosine, so the schedule continues from the new peak. Returns
    ``(before, after)``. Needs the ``--opt dd`` pair of groups (``encoder`` = the trunk, ``head``)."""
    by = {g.get("name"): g for g in opt.param_groups}
    if "encoder" not in by or "head" not in by:
        raise B1GuardError(f"the B1 guard needs the 'encoder' and 'head' groups of --opt dd; found {sorted(map(str, by))}")
    for nm in ("encoder", "head"):
        if "initial_lr" not in by[nm]:
            raise B1GuardError(f"group {nm!r} has no latched initial_lr: the schedule has not run, so there is no peak to cut")
    before, trunk = float(by["head"]["initial_lr"]), float(by["encoder"]["initial_lr"])
    after = min(before, trunk)
    by["head"]["initial_lr"] = after
    return before, after


def b1_guard_apply(state: Optional[dict], step: int, erow: dict, opt) -> Optional[dict]:
    """The guard, evaluated on one in-run eval row. Returns the row to log (and mutates ``state`` and ``opt``) the one
    time it decides, ``None`` otherwise (not armed / decided / before ``guard_step`` / the EMA eval of this row failed
    and the next eval will try again). Raises ``B1GuardError`` when the row has no (finite) EMA metric."""
    if state is None or not state.get("armed") or state.get("decided"):
        return None
    if int(step) < int(state["guard_step"]):
        return None
    if erow.get("ema_eval_error"):
        return None
    if B1_METRIC_KEY not in erow:
        raise B1GuardError(f"the step-{step} eval row has no {B1_METRIC_KEY!r}: the guard is armed but the EMA box3d AP@2 m "
                           f"is not being measured (--slot-vis1 / --w-box3d / --ema-decay)")
    v = erow[B1_METRIC_KEY]
    decision = b1_guard_decide(v, state["ref_ap2m"])
    before = after = None
    if decision == "cut":
        before, after = b1_head_cut(opt)
    state.update(decided=True, decision=decision, decided_at_step=int(step), ema_ap2m=float(v),
                 head_peak_lr_before=before, head_peak_lr_after=after)
    return {"step": int(step), "b1_guard": decision, "b1_cut": decision == "cut",
            "b1_ema_box3d_ap2m": float(v), "b1_ref_box3d_ap2m": float(state["ref_ap2m"]),
            "b1_guard_step": int(state["guard_step"]),
            "b1_head_peak_lr_before": before, "b1_head_peak_lr_after": after}
