"""refcv9 ``--r9-alloc-force-schedule``: remove the train/inference candidate-POOL mismatch (H-R9-ALLOC-ORACLE-1).

refcv8 TRAINING teacher-forces the GT joint hypothesis into the allocated candidates
(``refcv8_train.r8_before_forward`` -> ``RefCV3Model.set_r8_teacher`` -> ``refc._r8_extend`` ->
``refcv8_conditioning.allocate(force=GT, force_min=4)``), so the GT hypothesis is ALWAYS in the M allocated candidates
while training and is NOT at inference, where the pool is the predicted posterior's alone. This module schedules the
probability that a window's GT hypothesis is forced into its pool, as a function of the training step:

* ``const:<p>``                      -- p at every step (``const:1.0`` = refcv8 as trained; ``const:0.0`` = the
  predicted-only pool of inference on every training window);
* ``linear:<p0>:<p1>:<start>:<end>`` -- p0 up to step ``start``, linear to p1 at step ``end``, p1 after.

Absent (the default, every refcv8 run): nothing is attached, nothing is drawn, ``force_pool`` never reaches the
decoder -- the forward and the optimiser step are bit-identical to the tip.

The per-window coin comes from a PRIVATE stream: a CPU ``torch.Generator`` seeded from (``--r9-alloc-force-seed``,
step) -- never the global RNG and never refcv8's dedicated ``R8Generator`` -- so turning the flag on moves no other
random draw, and a resumed run (or the launch gate's repeated forward at one step) reads the same coins.

⛔ The GT CONSTRAINT teacher-forcing is unchanged and stays keyed on the GT id: ``refc._r8_extend`` overrides the
constraint only on an allocated candidate that IS the GT hypothesis (``hyp == force``) and only where the
scheduled-sampling coin ``force_tf`` is set. A window whose GT is NOT forced therefore has that override exactly when
its predicted posterior allocated the GT hypothesis on its own, and on no candidate otherwise -- nothing downstream
(the matched L1 reads ``r8_hyp_alloc == gt_hyp`` per row, the constraint override reads ``hyp == force``) assumes the
GT is in the pool.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

__all__ = ["ForceSchedule", "PoolForce", "parse_schedule", "DEFAULT_SEED", "MODEL_ATTR", "is_on", "pin_args",
           "attach", "apply_step", "stamp", "register_dvb_levers"]

#: the private coin stream's default seed (``--r9-alloc-force-seed``)
DEFAULT_SEED = 20261027
#: the attribute the attached :class:`PoolForce` lives on (a plain attribute: never in the state dict)
MODEL_ATTR = "_r9_alloc_force"
_KINDS = ("const", "linear")
_M64 = (1 << 64) - 1


@dataclass(frozen=True)
class ForceSchedule:
    """P(GT hypothesis forced into the window's allocated pool) as a function of the training step."""
    spec: str
    kind: str
    p0: float
    p1: float
    start: int
    end: int

    def p_at(self, step: int) -> float:
        s = int(step)
        if self.kind == "const" or s <= self.start:
            return float(self.p0)
        if s >= self.end:
            return float(self.p1)
        return float(self.p0 + (self.p1 - self.p0) * (s - self.start) / float(self.end - self.start))

    def to_dict(self) -> dict:
        return {"spec": self.spec, "kind": self.kind, "p0": self.p0, "p1": self.p1, "start": self.start,
                "end": self.end}


def _prob(tok: str, spec: str) -> float:
    try:
        p = float(tok)
    except ValueError:
        raise ValueError(f"[refcv9] --r9-alloc-force-schedule {spec!r}: {tok!r} is not a probability") from None
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        raise ValueError(f"[refcv9] --r9-alloc-force-schedule {spec!r}: probability {tok!r} is outside [0, 1]")
    return p


def _step(tok: str, spec: str) -> int:
    if not tok.isdigit():
        raise ValueError(f"[refcv9] --r9-alloc-force-schedule {spec!r}: step {tok!r} is not a non-negative integer")
    return int(tok)


def parse_schedule(spec: Any) -> ForceSchedule:
    """``const:<p>`` | ``linear:<p0>:<p1>:<start>:<end>`` (start < end, integer steps; every p in [0, 1]).
    Anything else raises ``ValueError`` -- a malformed schedule is REFUSED, never read as a default."""
    if not isinstance(spec, str) or not spec or spec != spec.strip() or any(c.isspace() for c in spec):
        raise ValueError(f"[refcv9] --r9-alloc-force-schedule {spec!r}: expected 'const:<p>' or "
                         "'linear:<p0>:<p1>:<start>:<end>' (no whitespace)")
    parts = spec.split(":")
    kind = parts[0]
    if kind not in _KINDS:
        raise ValueError(f"[refcv9] --r9-alloc-force-schedule {spec!r}: kind {kind!r} not in {_KINDS}")
    if kind == "const":
        if len(parts) != 2:
            raise ValueError(f"[refcv9] --r9-alloc-force-schedule {spec!r}: 'const' takes exactly one value")
        p = _prob(parts[1], spec)
        return ForceSchedule(spec, "const", p, p, 0, 0)
    if len(parts) != 5:
        raise ValueError(f"[refcv9] --r9-alloc-force-schedule {spec!r}: 'linear' takes p0:p1:start:end")
    p0, p1 = _prob(parts[1], spec), _prob(parts[2], spec)
    s0, s1 = _step(parts[3], spec), _step(parts[4], spec)
    if s1 <= s0:
        raise ValueError(f"[refcv9] --r9-alloc-force-schedule {spec!r}: end step {s1} must exceed start step {s0}")
    return ForceSchedule(spec, "linear", p0, p1, s0, s1)


def _mix(seed: int, step: int) -> int:
    """splitmix64 over (seed, step) -> a 63-bit torch seed. A pure function: the coin of (seed, step) never depends
    on how many draws came before it."""
    x = (int(seed) * 0x9E3779B97F4A7C15 + (int(step) + 1) * 0xD1B54A32D192ED03) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return (x ^ (x >> 31)) & ((1 << 63) - 1)


class PoolForce:
    """The attached schedule + its private coin stream. ``step`` is set by the trainer before every step
    (:func:`apply_step`); :meth:`mask` is called once per training forward by ``r8_before_forward``."""

    def __init__(self, sched: ForceSchedule, seed: int = DEFAULT_SEED):
        self.sched = sched
        self.seed = int(seed)
        self.step = 0
        self.last_p: float | None = None

    @property
    def spec(self) -> str:
        return self.sched.spec

    def p(self) -> float:
        return self.sched.p_at(self.step)

    def coins(self, b: int) -> Tensor:
        """[b] float64 uniforms in [0, 1) from the PRIVATE generator of (seed, step)."""
        g = torch.Generator(device="cpu")
        g.manual_seed(_mix(self.seed, self.step))
        return torch.rand((int(b),), generator=g, dtype=torch.float64)

    def mask(self, hyp: Tensor) -> Tensor:
        """[B] bool on ``hyp``'s device: True = force this window's GT hypothesis into its pool. p = 1 -> all True
        and p = 0 -> all False exactly (the uniforms lie in [0, 1))."""
        p = self.p()
        self.last_p = p
        return (self.coins(hyp.shape[0]) < p).to(hyp.device)


def is_on(args) -> bool:
    return getattr(args, "r9_alloc_force_schedule", None) is not None


def pin_args(args) -> ForceSchedule | None:
    """The refusals, at argv time (before anything is built). Absent -> None and the seed must hold its default."""
    seed = getattr(args, "r9_alloc_force_seed", DEFAULT_SEED)
    if not is_on(args):
        if seed is not None and int(seed) != DEFAULT_SEED:
            raise SystemExit("[refcv9] --r9-alloc-force-seed without --r9-alloc-force-schedule: it seeds nothing")
        return None
    try:
        sched = parse_schedule(args.r9_alloc_force_schedule)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if not bool(getattr(args, "refcv8", False)) or int(getattr(args, "r8_n_alloc", 0) or 0) <= 0:
        raise SystemExit("[refcv9] --r9-alloc-force-schedule needs the refcv8 allocated pool (--refcv8 and "
                         "--r8-n-alloc > 0): without it no GT hypothesis is ever forced and the schedule is dead")
    if bool(getattr(args, "r9_calib_stage", False)):
        raise SystemExit("[refcv9] --r9-alloc-force-schedule with --r9-calib-stage: the calibration stage runs the "
                         "INFERENCE path, where no GT hypothesis is forced -- the schedule would be dead")
    return sched


def attach(model, args) -> dict | None:
    """Attach the schedule to the BUILT model (training only reads it). Refuses a build without an allocated pool.
    -> the config.json stamp, or None when the flag is absent (nothing attached)."""
    sched = pin_args(args)
    if sched is None:
        return None
    dec = getattr(getattr(model, "core", model), "decoder", None)
    r8cfg = getattr(dec, "r8_cfg", None)
    if not bool(getattr(model, "r8_enabled", False)) or r8cfg is None or int(getattr(r8cfg, "n_alloc", 0) or 0) <= 0:
        raise SystemExit("[refcv9] --r9-alloc-force-schedule: the BUILT model has no refcv8 allocated pool")
    pf = PoolForce(sched, int(getattr(args, "r9_alloc_force_seed", DEFAULT_SEED)))
    setattr(model, MODEL_ATTR, pf)
    return stamp(pf, int(getattr(args, "steps", 0) or 0))


def apply_step(model, step: int) -> None:
    """Called by the trainer before every step's forward (next to refcv8's scheduled-sampling ratio)."""
    pf = getattr(model, MODEL_ATTR, None)
    if pf is not None:
        pf.step = int(step)


def stamp(pf: PoolForce, steps: int) -> dict:
    last = max(int(steps) - 1, 0)
    return dict(pf.sched.to_dict(), seed=pf.seed, p_at_step0=pf.sched.p_at(0), p_at_last_step=pf.sched.p_at(last),
                last_step=last, stream="private torch.Generator(cpu) seeded splitmix64(seed, step)",
                gt_constraint_tf="unchanged: keyed on the GT id, applies only to an allocated GT-hypothesis candidate")


def register_dvb_levers(register) -> None:
    """G-DVB: argv against the BUILT model -- the attached schedule's spec / seed, read off the model attribute.
    Expectations are LITERALS from argv (the spec string itself; the default seed 20261027)."""
    from tanitad.train import declared_vs_built as dvb

    def _sched(m, a):
        want = dvb._a(a, "r9_alloc_force_schedule", None)
        pf = getattr(m, MODEL_ATTR, None)
        return dvb._eq("r9_alloc_force_schedule", want, None if pf is None else pf.spec,
                       f"model.{MODEL_ATTR}.spec")

    def _seed(m, a):
        want = dvb._a(a, "r9_alloc_force_seed", DEFAULT_SEED)
        pf = getattr(m, MODEL_ATTR, None)
        if pf is None:
            return dvb._eq("r9_alloc_force_seed", DEFAULT_SEED, None if want is None else int(want),
                           "argv (schedule off: must hold the default)")
        return dvb._eq("r9_alloc_force_seed", None if want is None else int(want), pf.seed,
                       f"model.{MODEL_ATTR}.seed")

    register("r9_alloc_force_schedule", "built", _sched)
    register("r9_alloc_force_seed", "built", _seed)
