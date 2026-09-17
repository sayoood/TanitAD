"""refcv6 -- the GRADIENT-CONFLICT DETECTOR between the planning loss and the
perception losses, on the SHARED TRUNK.

``SPEC_REFCV6_V2.md`` §6: *"Both losses reach the trunk. The gradient-conflict
detector (+1 / 0 / -1 analytic controls, 30x mutation) runs in every arm."*
The design is pre-registered in ``TanitAD Research Lab/Architecture &
Inference/Research/2026-09-11-lidar-bev-gt/PREREG_BEV_CAPACITY_COMPETITION.md``
§4 (lines 95-120 of that file), and this module implements THAT, not a variant.

    g_traj = dL_traj/dtheta_trunk        g_aux = dL_perception/dtheta_trunk
    conflict(t) = cos(g_traj, g_aux)

⛔ **LOGGED AND REPORTED. NEVER A STOP RULE.** Nothing in this module halts,
clips, re-weights or projects a gradient. It measures. The one thing it *does*
refuse is to start when its own controls miss (:meth:`GradientConflictDetector.
self_check`), because a probe that cannot read a known value is not evidence.

## Why it exists, MEASURED

Auxiliary perception has cost this programme twice already:

* ``D-P1-AGENTCOND-1`` -- DiffusionDrive's agent head at 17 M parameters.
* WP-D's BEV aux at 108 M, which regressed lateral accuracy **3.3-6.0x the
  replicate floor** (``.../2026-09-07-wpd-bev-aux/PANEL_RESULT.md``).
* ``E-DEC-18b`` (PSG, the v6/v7 stack) -- the aux term sat at ~0.3-0.9 against
  an objective at ~0.03, i.e. **10-30x**, and destroyed the encoder at every
  weight tested. That is the MEASURED mechanism the 30x mutation re-introduces.

refcv6 hangs **two** heads (SAM3 map + 3-D boxes) on one trunk, so the question
is not whether to look but whether we can see it while it happens.

## ⛔⛔ THE FINDING THAT CHANGES HOW THIS IS READ

**The cosine is SCALE-INVARIANT, so it CANNOT see the 30x mutation.**
``cos(g, 30 g_aux) == cos(g, g_aux)`` identically -- not approximately: positive
scaling is exactly the transformation a cosine quotients out. ``E-DEC-18``'s
measured mechanism was **magnitude** ("10-30x"), not angle, so a detector that
reports only ``cos`` is blind to the precise defect it is cited for.

Two scale-SENSITIVE channels are therefore reported beside it, from the SAME two
gradients, at no extra backward:

| field | definition | 30x aux reads |
|---|---|---|
| ``cos`` | ``<g_traj, g_aux> / (|g_traj| |g_aux|)`` | **unchanged** (blind) |
| ``ratio`` | ``|g_aux| / |g_traj|`` | **30x** |
| ``proj`` | ``<g_traj, g_aux> / |g_traj|^2`` | **30x** |

``proj`` is the signed multiple of ``g_traj`` that the aux term adds along the
planner's own descent direction: ``proj == -1`` means the aux gradient exactly
cancels the planning gradient. It is ``cos * ratio``, and it is the field that
answers *"is the aux about to take the trunk over"*. See ``RESULT.md`` of
``2026-09-17-refcv6-conflict-detector`` for the measurement, and the ESCALATION
it raises against pre-registered criterion ``B2``.

## The three analytic controls

⛔ Pre-registered, each with a target derived INDEPENDENTLY of this code:

| control | must read | why it is not a tautology |
|---|---|---|
| ``cos(g_traj, g_traj)`` | **exactly +1.0** | an identity, independent of training |
| ``cos(g_traj, g_aux)``, aux **detached** | ``cos`` is **NaN** (0/0), reported conflict **exactly 0.0**, ``degenerate`` set | isolates "shares gradient" from "is present" |
| ``cos(g_traj, -g_traj)`` | **exactly -1.0** | fixes the sign convention |

⚠️ **The detached row reconciles two texts that do not quite agree.** The prereg
says the control "must read exactly 0.0" *and* "report ``NaN`` -> refuse, never
0 by accident". Both are honoured by reporting **both numbers**: :attr:`Group
Reading.cos` is the raw quotient and is ``NaN`` when a norm is zero (no epsilon
is ever added, so "orthogonal" and "absent" can never collide), while
:attr:`GroupReading.conflict` -- the field a reader reads as *the* conflict --
is ``0.0`` **only** when ``degenerate`` is set and ``norm_aux`` is exactly
``0.0``, both of which are logged beside it. A 0.0 in the log always carries its
receipt.

⭐ **Exactness is real, not a tolerance.** The denominator is
``sqrt(norm_traj_sq * norm_aux_sq)``, NOT ``sqrt(a) * sqrt(b)``: for the self
control the two sums are the same float ``S``, and IEEE-754 ``sqrt`` is
correctly rounded, so ``sqrt(S*S) == S`` exactly and the quotient is exactly
``1.0``. Written the other way it is ``S / (sqrt(S)*sqrt(S))``, which is 1 ulp
off for most ``S`` -- a tolerance where an identity was available.

## ⚠️ Cost -- and the prereg's estimate was optimistic

The prereg budgets this at *"one extra backward over the trunk -- no extra arm,
no extra GPU-day"*. **MEASURED 2026-09-17** on the dev box (CPU, batch 2, the
real ``RefCV3Model`` + BEV aux, three trunk sizes x 3 interleaved replicates x
median of 15 steps; ``.../2026-09-17-refcv6-conflict-detector/raw/overhead.json``):

| mode | extra backwards | 77.5 k trunk | 1.57 M | 6.95 M | training step |
|---|---|---|---|---|---|
| :data:`MODE_PROBE` (default) | 2 | +71.7 % | +83.8 % | **+104.2 %** | **bit-identical** |
| :data:`MODE_REUSE` | 1 | +26.4 % | +44.2 % | +67.8 % | NOT bit-identical |
| :data:`MODE_SUBTRACT` | 1 | +26.3 % | +53.1 % | **+62.2 %** | **bit-identical** |

⇒ the pre-registered statistic **DOUBLES** the step at the largest trunk
measured, and the overhead RISES with trunk size, so refcv6's 21.8-42 M trunk
will be at least that. It is not free, and ``--conflict-every N`` is the only
lever that makes it cheap -- the overhead divides by ``N``, but a median over
sparse steps is a different statistic and the run record must say so.
⚠️ CPU figures; the GPU figure is UNVERIFIED, and the dev box is shared, so the
banked file keeps every replicate rather than an average.

⭐ :data:`MODE_SUBTRACT` dominates :data:`MODE_REUSE` -- same cost, and the step
stays bit-identical -- so ``reuse`` survives only as the literal reading of the
prereg's accounting. But ``subtract`` answers against ``L_total - L_aux``, not
``L_traj``: a **different statistic**, stamped in every row as ``cd_plan_side``.

``MODE_PROBE`` never touches ``.grad`` (``torch.autograd.grad`` does not
accumulate), so the caller's ``loss.backward()`` writes exactly the bytes it
wrote before -- the detector ON and the detector off are the same arm. That is
stronger than the flag requires and is the reason this instrument can never be
blamed for a result.

## Use

    det = GradientConflictDetector.for_model(model, ConflictConfig(enabled=True))
    det.self_check(loss_traj, loss_aux)        # once, at the first step; RAISES
    reading = det.measure(loss_traj, loss_aux) # every step
    row.update(reading.row())                  # -> metrics.jsonl
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field

import torch

__all__ = [
    "MODE_PROBE", "MODE_REUSE", "ROW_PREFIX", "POOLED_NAME", "HEADS_NAME",
    "ConflictConfig", "GroupReading", "ConflictReading", "ControlResult",
    "GradientConflictDetector", "cosine_stats", "default_group_of",
    "enabled_for_arm", "ControlFailure", "TRUNK_PREFIX_CANDIDATES",
    "resolve_trunk_prefixes", "MODE_SUBTRACT", "PLAN_SIDE",
]

#: probe: two extra partial backwards, ``.grad`` untouched, training bit-identical.
MODE_PROBE = "probe"
#: reuse: the two gradients ARE the training gradient; one extra backward, but the
#: accumulation order differs from a fused ``backward()`` and so is not bit-identical.
MODE_REUSE = "reuse"
#: ⭐ subtract: ONE extra backward -- the aux path only -- and the training
#: ``backward()`` is untouched, so the step stays bit-identical. ``g_plan`` is
#: recovered by linearity as ``p.grad - g_aux``, read BEFORE ``clip_grad_norm_``.
#: ⚠️ Its planning side is therefore ``L_total - L_aux`` (the WHOLE planning
#: objective) and NOT the pre-registered ``L_traj``. That is a different
#: statistic and every row it writes says so in ``cd_plan_side``.
MODE_SUBTRACT = "subtract"

#: which loss the planning side of the cosine was taken from, per mode.
PLAN_SIDE = {MODE_PROBE: "traj", MODE_REUSE: "traj",
             MODE_SUBTRACT: "total_minus_aux"}

#: ``metrics.jsonl`` key prefix. Follows the ``gp_`` convention of
#: ``refc_v3_train._grad_probe_row``: with the detector off, NO key is emitted
#: and the log schema is identical to the pre-detector trainer's.
ROW_PREFIX = "cd_"

#: the pooled row's group name (every selected trunk parameter).
POOLED_NAME = "trunk"
#: the non-trunk group. Its cosine is a STRUCTURAL control: the planning loss and
#: the perception losses touch disjoint head parameters, so the dot product is
#: exactly 0.0 with both norms non-zero -- a genuine 0.0, not a degenerate one.
HEADS_NAME = "heads"

#: ⚠️ The shared trunk lives at DIFFERENT dotted paths on the two models this
#: trainer builds: ``RefCModel.encoder`` and ``RefCV3Model.core.encoder``. They
#: are disjoint prefixes (``core.encoder.x`` does not start with ``encoder.``),
#: so listing both selects exactly one. MEASURED 2026-09-17: with only
#: ``encoder.`` the v3 model selects ZERO parameters -- caught by the refusal
#: below, which is the whole reason that refusal is not a warning.
TRUNK_PREFIX_CANDIDATES = ("encoder.", "core.encoder.")

_STAGE_RE = re.compile(r"^(?:layer|stage|stages|block|blocks)(\d*)$")
_NUM_RE = re.compile(r"^\d+$")


def resolve_trunk_prefixes(model, candidates=TRUNK_PREFIX_CANDIDATES) -> tuple:
    """-> the ONE candidate prefix that selects trainable parameters on ``model``.

    ⛔ Refuses zero matches, and names what the model actually has, because the
    failure it prevents is silent: a prefix that matches nothing makes the
    detector log ``NaN`` for an entire run while ``config.json`` says it was on.
    """
    named = list(model.named_parameters())
    hit = tuple(p for p in candidates
                if any(n.startswith(p) and q.requires_grad for n, q in named))
    if not hit:
        tops = sorted({n.split(".")[0] for n, _ in named})
        raise ValueError(
            f"none of the trunk prefixes {list(candidates)} selects a trainable "
            f"parameter of {type(model).__name__}. Its top-level parameter "
            f"groups are {tops}. Name the shared trunk explicitly rather than "
            f"letting the detector measure the empty set.")
    return hit


class ControlFailure(RuntimeError):
    """⛔ An analytic control did not read its known value. The probe refuses."""


# --------------------------------------------------------------------------- #
#  the cosine, and the two scale-sensitive channels beside it
# --------------------------------------------------------------------------- #
def cosine_stats(ga, gb, *, dtype=torch.float64) -> dict:
    """``cos`` / ``ratio`` / ``proj`` of two gradient lists, plus their receipts.

    ``ga`` / ``gb`` are equal-length sequences of tensors-or-``None`` over the
    SAME parameters, in the same order (``None`` == the loss never reached that
    parameter, which is a zero contribution and is counted, not skipped).

    ⛔ **No epsilon is added anywhere.** A zero-norm side yields ``cos = NaN``
    and ``degenerate = True``; it must never be indistinguishable from a real
    orthogonality. Accumulation is in ``float64`` by default: the ``+1``/``-1``
    controls are identities and are reported as identities, not as tolerances.
    """
    if len(ga) != len(gb):
        raise ValueError(f"gradient lists differ in length: {len(ga)} vs {len(gb)}")
    dot = torch.zeros((), dtype=dtype)
    saq = torch.zeros((), dtype=dtype)
    sbq = torch.zeros((), dtype=dtype)
    n_params = n_tensors = 0
    n_none_a = n_none_b = 0
    for a, b in zip(ga, gb):
        ref = a if a is not None else b
        if ref is None:
            continue
        n_tensors += 1
        n_params += int(ref.numel())
        av = bv = None
        if a is None:
            n_none_a += 1
        else:
            av = a.detach().reshape(-1).to(dtype)
            saq += av.dot(av)
        if b is None:
            n_none_b += 1
        else:
            bv = b.detach().reshape(-1).to(dtype)
            sbq += bv.dot(bv)
        if av is not None and bv is not None:
            dot += av.dot(bv)
    d = float(dot)
    aq = float(saq)
    bq = float(sbq)
    # ⭐ sqrt(aq*bq), never sqrt(aq)*sqrt(bq): see the module docstring. The
    # product can only overflow above ~1e154 per side, which a gradient norm
    # does not reach; the fallback is flagged rather than silent.
    prod = aq * bq
    scaled = False
    if math.isinf(prod) and not (math.isinf(aq) or math.isinf(bq)):
        prod = (aq * 1e-150) * (bq * 1e-150)
        scaled = True
    den = math.sqrt(prod) * (1e150 if scaled else 1.0)
    degenerate = (aq == 0.0) or (bq == 0.0)
    cos = (d / den) if den != 0.0 else float("nan")
    na = math.sqrt(aq)
    nb = math.sqrt(bq)
    return {
        "cos": cos,
        "ratio": (nb / na) if na != 0.0 else float("nan"),
        "proj": (d / aq) if aq != 0.0 else float("nan"),
        "dot": d,
        "norm_traj": na,
        "norm_aux": nb,
        "degenerate": bool(degenerate),
        "n_params": int(n_params),
        "n_tensors": int(n_tensors),
        "n_none_traj": int(n_none_a),
        "n_none_aux": int(n_none_b),
        "den_rescaled": bool(scaled),
    }


# --------------------------------------------------------------------------- #
#  parameter grouping -- "the trunk is in conflict" and "the last stage is in
#  conflict" imply different fixes, so they are never pooled into one number
# --------------------------------------------------------------------------- #
def default_group_of(name: str) -> str:
    """Group a TRUNK parameter by the backbone's OWN naming -- never a literal.

    ``encoder.net.conv1.weight``       -> ``stem``
    ``encoder.net.layer3.1.bn2.bias``  -> ``stage_layer3``
    ``encoder.net.stages.2.blocks...`` -> ``stage_stages2``
    ``encoder.fuse16.proj.weight``     -> ``fuse``

    ⚠️ The stage token is taken POSITIONALLY (the first segment that names a
    stage), not by substring: ``layer1.0.conv1.weight`` contains ``conv1`` and a
    substring rule would file half of every ResNet stage under ``stem``.
    """
    segs = str(name).split(".")
    for i, s in enumerate(segs):
        m = _STAGE_RE.match(s)
        if m:
            idx = m.group(1)
            if not idx and i + 1 < len(segs) and _NUM_RE.match(segs[i + 1]):
                idx = segs[i + 1]
            return f"stage_{s}{idx}" if not m.group(1) else f"stage_{s}"
        if s.startswith("fuse"):
            return "fuse"
    return "stem"


@dataclass(frozen=True)
class ConflictConfig:
    """Declared decisions of :class:`GradientConflictDetector`.

    ``enabled``: ⛔ the flag. ON by default for refcv6 arms (:func:`enabled_for_arm`),
    off otherwise. With it off nothing is constructed, nothing is computed and no
    ``metrics.jsonl`` key is emitted.
    """

    enabled: bool = False
    #: dotted prefixes that define theta_trunk. ⚠️ ``RefCModel`` holds the trunk at
    #: ``encoder.`` and ``RefCV3Model`` WRAPS it at ``core.encoder.``; a default of
    #: ``encoder.`` alone silently selects nothing on the v3 model, which is how
    #: this was caught. Both are listed and they are disjoint by construction, so
    #: exactly one matches. :func:`resolve_trunk_prefixes` is the explicit route.
    trunk_prefixes: tuple = TRUNK_PREFIX_CANDIDATES
    #: also report the non-trunk parameters as one group (the structural control).
    include_heads: bool = True
    #: log every N steps. 1 = every step, which is what the prereg asks for.
    every: int = 1
    mode: str = MODE_PROBE
    #: accumulate the dot products in float64. ⚠️ This is about the PRECISION of
    #: the per-step reading -- a 21.8-90 M-term sum in float32 loses digits --
    #: and NOT about the controls: MEASURED 2026-09-17, the +-1 identities read
    #: exactly on 12/12 seeds in BOTH dtypes, because the ``sqrt(a*b)``
    #: denominator makes them structural rather than precision-dependent.
    float64: bool = True
    #: ⛔ refuse to run when an analytic control misses (prereg §4).
    require_controls: bool = True

    def __post_init__(self) -> None:
        if self.mode not in PLAN_SIDE:
            raise ValueError(
                f"mode must be one of {sorted(PLAN_SIDE)}, got {self.mode!r}")
        if int(self.every) < 1:
            raise ValueError(f"every must be >= 1, got {self.every}")
        if not self.trunk_prefixes:
            raise ValueError(
                "no trunk prefix: the detector would measure the empty set and "
                "report NaN for every step. Name the shared trunk explicitly.")

    def as_dict(self) -> dict:
        return {"enabled": bool(self.enabled),
                "trunk_prefixes": list(self.trunk_prefixes),
                "include_heads": bool(self.include_heads),
                "every": int(self.every), "mode": str(self.mode),
                "float64": bool(self.float64),
                "require_controls": bool(self.require_controls)}


@dataclass(frozen=True)
class GroupReading:
    """One parameter group's reading. ``cos`` is raw; ``conflict`` is reported."""

    name: str
    cos: float
    ratio: float
    proj: float
    dot: float
    norm_traj: float
    norm_aux: float
    degenerate: bool
    n_params: int
    n_tensors: int
    n_none_traj: int
    n_none_aux: int

    @property
    def conflict(self) -> float:
        """⭐ The REPORTED conflict. ``0.0`` only when the aux gradient is
        exactly the zero vector on this group -- the detached control -- and
        then ``degenerate`` and ``norm_aux == 0.0`` are logged beside it, so the
        0.0 is never a silently-smoothed quotient. Otherwise it is ``cos``."""
        if self.degenerate and self.norm_aux == 0.0 and self.norm_traj != 0.0:
            return 0.0
        return self.cos

    @classmethod
    def from_stats(cls, name: str, st: dict) -> "GroupReading":
        return cls(name=str(name), cos=st["cos"], ratio=st["ratio"],
                   proj=st["proj"], dot=st["dot"], norm_traj=st["norm_traj"],
                   norm_aux=st["norm_aux"], degenerate=st["degenerate"],
                   n_params=st["n_params"], n_tensors=st["n_tensors"],
                   n_none_traj=st["n_none_traj"], n_none_aux=st["n_none_aux"])

    def as_dict(self) -> dict:
        d = {k: getattr(self, k) for k in
             ("name", "cos", "ratio", "proj", "dot", "norm_traj", "norm_aux",
              "degenerate", "n_params", "n_tensors", "n_none_traj", "n_none_aux")}
        d["conflict"] = self.conflict
        return d


@dataclass(frozen=True)
class ConflictReading:
    """A whole step: the pooled trunk reading plus every group's."""

    pooled: GroupReading
    groups: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    step: int = -1
    #: ⛔ WHICH planning loss the cosine was taken against -- ``traj`` (the
    #: pre-registered one) or ``total_minus_aux`` (:data:`MODE_SUBTRACT`). In the
    #: log row on every step, because the two are different statistics and a
    #: reader who cannot tell them apart will compare them.
    plan_side: str = PLAN_SIDE[MODE_PROBE]

    def row(self, prefix: str = ROW_PREFIX) -> dict:
        """-> the ``metrics.jsonl`` fields. ⛔ NOT rounded: a real 1e-8 reading
        rounded to 5 dp is 0.0, which is the exact signature of the defect this
        measures -- the caller merges this AFTER its own rounding pass, the way
        ``_grad_probe_row`` is merged."""
        r = {f"{prefix}cos": self.pooled.cos,
             f"{prefix}conflict": self.pooled.conflict,
             f"{prefix}ratio": self.pooled.ratio,
             f"{prefix}proj": self.pooled.proj,
             f"{prefix}gn_traj": self.pooled.norm_traj,
             f"{prefix}gn_aux": self.pooled.norm_aux,
             f"{prefix}degenerate": float(self.pooled.degenerate),
             f"{prefix}n_params": float(self.pooled.n_params),
             f"{prefix}plan_side": self.plan_side,
             f"{prefix}ms": self.elapsed_s * 1e3}
        for g in self.groups.values():
            if g.name == POOLED_NAME:
                continue
            r[f"{prefix}{g.name}_cos"] = g.cos
            r[f"{prefix}{g.name}_ratio"] = g.ratio
            r[f"{prefix}{g.name}_proj"] = g.proj
            r[f"{prefix}{g.name}_gn_traj"] = g.norm_traj
            r[f"{prefix}{g.name}_gn_aux"] = g.norm_aux
        return r

    def as_dict(self) -> dict:
        return {"pooled": self.pooled.as_dict(),
                "groups": {k: v.as_dict() for k, v in self.groups.items()},
                "elapsed_s": float(self.elapsed_s), "step": int(self.step),
                "plan_side": str(self.plan_side)}


@dataclass(frozen=True)
class ControlResult:
    """The three analytic controls, with what each READ and what it MUST read."""

    self_cos: float
    negated_cos: float
    detached_cos: float
    detached_conflict: float
    detached_norm_aux: float
    detached_degenerate: bool
    ok: bool
    failures: tuple = ()

    def as_dict(self) -> dict:
        return {"self_cos": self.self_cos, "negated_cos": self.negated_cos,
                "detached_cos": self.detached_cos,
                "detached_conflict": self.detached_conflict,
                "detached_norm_aux": self.detached_norm_aux,
                "detached_degenerate": bool(self.detached_degenerate),
                "ok": bool(self.ok), "failures": list(self.failures)}


# --------------------------------------------------------------------------- #
#  the detector
# --------------------------------------------------------------------------- #
class GradientConflictDetector:
    """Per-step ``cos(g_traj, g_aux)`` on the shared trunk. Logged, never a gate."""

    def __init__(self, named_params, cfg: ConflictConfig):
        self.cfg = cfg
        named = [(n, p) for n, p in named_params]
        trainable = [(n, p) for n, p in named if p.requires_grad]
        is_trunk = [bool(self._in_trunk(n)) for n, _ in trainable]
        self.trunk_names = [n for (n, _), t in zip(trainable, is_trunk) if t]
        self.trunk_params = [p for (_, p), t in zip(trainable, is_trunk) if t]
        self.head_names = [n for (n, _), t in zip(trainable, is_trunk) if not t]
        self.head_params = [p for (_, p), t in zip(trainable, is_trunk) if not t]
        if not self.trunk_params:
            frozen = sum(1 for n, p in named
                         if self._in_trunk(n) and not p.requires_grad)
            tops = sorted({n.split(".")[0] for n, _ in named})
            why = (f"{frozen} parameter tensors match the prefix but are FROZEN"
                   if frozen else
                   f"NO parameter matches the prefix at all; the model's "
                   f"top-level parameter groups are {tops}")
            raise ValueError(
                f"no TRAINABLE parameter matches trunk prefixes "
                f"{list(cfg.trunk_prefixes)}: {why} ({len(trainable)} trainable "
                f"parameters exist). The detector would measure the empty set "
                f"and log NaN for the whole run while config.json said it was "
                f"on. A frozen trunk has no conflict to detect and the arm "
                f"should say so; a mis-named prefix is a wiring bug.")
        # group index lists, over `self.trunk_params` order
        groups: dict = {}
        for i, n in enumerate(self.trunk_names):
            groups.setdefault(default_group_of(self._strip(n)), []).append(i)
        self.group_index = dict(sorted(groups.items()))
        self._dtype = torch.float64 if cfg.float64 else torch.float32
        self._controls: ControlResult | None = None
        #: MODE_REUSE only -- the last step's (params, g_traj, g_aux).
        self.last_grads = None

    # -- construction ------------------------------------------------------- #
    @classmethod
    def for_model(cls, model, cfg: ConflictConfig) -> "GradientConflictDetector | None":
        """-> a detector, or ``None`` when the flag is off. ⛔ With the flag off
        NOTHING is constructed: the off path cannot differ from the pre-detector
        trainer even in an allocation."""
        if not cfg.enabled:
            return None
        return cls(model.named_parameters(), cfg)

    def _in_trunk(self, name: str) -> bool:
        return any(str(name).startswith(p) for p in self.cfg.trunk_prefixes)

    def _strip(self, name: str) -> str:
        for p in self.cfg.trunk_prefixes:
            if name.startswith(p):
                return name[len(p):]
        return name

    # -- the measurement ---------------------------------------------------- #
    def grads(self, loss, params, *, retain_graph: bool = True,
              create_graph: bool = False):
        """``dloss/dparams``, ``None`` where the loss never reached a parameter.

        ⛔ Uses ``torch.autograd.grad``, which does NOT accumulate into ``.grad``.
        In :data:`MODE_PROBE` that is the whole safety argument: the caller's own
        ``backward()`` still produces exactly the bytes it produced before.
        """
        if not loss.requires_grad:
            raise ValueError(
                "the loss does not require grad, so no gradient reaches the "
                "trunk and the reading would be a vacuous NaN. If this is the "
                "detached CONTROL, call `controls()`, which builds it explicitly.")
        return torch.autograd.grad(loss, list(params), retain_graph=retain_graph,
                                   create_graph=create_graph, allow_unused=True)

    def measure(self, loss_traj, loss_aux, *, step: int = -1,
                retain_graph: bool = True, also_heads: bool | None = None
                ) -> "ConflictReading":
        """One step's reading. ⛔ Does not touch ``.grad`` and does not step."""
        t0 = time.perf_counter()
        if also_heads is None:
            also_heads = self.cfg.include_heads
        params = list(self.trunk_params)
        n_trunk = len(params)
        if also_heads and self.head_params:
            params = params + list(self.head_params)
        g_t = self.grads(loss_traj, params, retain_graph=True)
        g_a = self.grads(loss_aux, params, retain_graph=retain_graph)
        if self.cfg.mode == MODE_REUSE:
            # the two partial gradients ARE the training gradient; hold them for
            # `accumulate_` so the caller's own backward can be skipped.
            self.last_grads = (params, g_t, g_a)
        return self._read(g_t[:n_trunk], g_a[:n_trunk],
                          g_t[n_trunk:], g_a[n_trunk:], step=step, t0=t0)

    def measure_after_backward(self, loss_aux, *, step: int = -1,
                               retain_graph: bool = False,
                               also_heads: bool | None = None
                               ) -> "ConflictReading":
        """:data:`MODE_SUBTRACT` -- ONE extra backward, training bit-identical.

        ⛔ **Call it AFTER ``total.backward(retain_graph=True)`` and BEFORE
        ``clip_grad_norm_``**, which rescales ``.grad`` in place: a reading taken
        after the clip is a reading of a rescaled gradient and its ``ratio`` is
        wrong by the clip factor.

        ``g_aux`` costs one backward over the aux path; ``g_plan`` is then
        ``p.grad - g_aux`` by linearity, for free. ⚠️ ``p.grad`` carries EVERY
        loss term, so the planning side here is the whole planning objective and
        not the pre-registered ``L_traj`` -- :data:`PLAN_SIDE` records that in
        the row, and the two are not interchangeable in a B2 verdict.
        """
        if self.cfg.mode != MODE_SUBTRACT:
            raise RuntimeError(
                f"measure_after_backward is {MODE_SUBTRACT!r} only; this "
                f"detector is {self.cfg.mode!r}")
        t0 = time.perf_counter()
        if also_heads is None:
            also_heads = self.cfg.include_heads
        params = list(self.trunk_params)
        n_trunk = len(params)
        if also_heads and self.head_params:
            params = params + list(self.head_params)
        if all(p.grad is None for p in params):
            raise RuntimeError(
                "every `.grad` is None: `measure_after_backward` was called "
                "before the training backward, so the planning side would be "
                "the zero vector and every reading a degenerate NaN")
        g_a = self.grads(loss_aux, params, retain_graph=retain_graph)
        g_t = [None if p.grad is None else
               (p.grad if a is None else p.grad - a)
               for p, a in zip(params, g_a)]
        return self._read(g_t[:n_trunk], g_a[:n_trunk],
                          g_t[n_trunk:], g_a[n_trunk:], step=step, t0=t0)

    def accumulate_(self, params=None, g_traj=None, g_aux=None) -> int:
        """:data:`MODE_REUSE` only -- add ``g_traj + g_aux`` into ``.grad`` so the
        caller can SKIP its own ``backward()``. That is the prereg's "one extra
        backward over the trunk" accounting.

        ⚠️ **NOT bit-identical to a fused ``backward()``**: the two partial sums
        are rounded before they are added, so the result differs in the last
        ulp. Opt-in for that reason. -> the number of parameters written.
        """
        if self.cfg.mode != MODE_REUSE:
            raise RuntimeError(
                f"accumulate_ is {MODE_REUSE!r} only; this detector is "
                f"{self.cfg.mode!r}, whose contract is that `.grad` is untouched")
        if params is None:
            if self.last_grads is None:
                raise RuntimeError(
                    "accumulate_ was called before measure(): there are no "
                    "gradients to write, and writing none would silently train "
                    "a step on a zero gradient")
            params, g_traj, g_aux = self.last_grads
        n = 0
        for p, a, b in zip(params, g_traj, g_aux):
            g = a if b is None else (b if a is None else a + b)
            if g is None:
                continue
            p.grad = g if p.grad is None else p.grad + g
            n += 1
        return n

    def _read(self, gt_trunk, ga_trunk, gt_head, ga_head, *, step: int,
              t0: float) -> "ConflictReading":
        dt = self._dtype
        pooled = GroupReading.from_stats(
            POOLED_NAME, cosine_stats(gt_trunk, ga_trunk, dtype=dt))
        groups = {POOLED_NAME: pooled}
        for gname, idx in self.group_index.items():
            st = cosine_stats([gt_trunk[i] for i in idx],
                              [ga_trunk[i] for i in idx], dtype=dt)
            groups[gname] = GroupReading.from_stats(gname, st)
        if gt_head:
            groups[HEADS_NAME] = GroupReading.from_stats(
                HEADS_NAME, cosine_stats(gt_head, ga_head, dtype=dt))
        return ConflictReading(pooled=pooled, groups=groups,
                               elapsed_s=time.perf_counter() - t0, step=int(step),
                               plan_side=PLAN_SIDE[self.cfg.mode])

    # -- the three analytic controls ---------------------------------------- #
    def controls(self, loss_traj, loss_aux=None, *, retain_graph: bool = True
                 ) -> ControlResult:
        """Read the three pre-registered controls on THIS model and THIS batch.

        ``cos(g,g) == +1`` exactly, ``cos(g,-g) == -1`` exactly, and a detached
        aux reads ``cos = NaN`` / ``conflict = 0.0`` / ``norm_aux == 0.0``.

        ⛔ The detached control is built here rather than taken from the caller:
        ``loss_aux.detach()`` alone has no ``grad_fn`` and ``autograd.grad``
        would raise instead of reading zero. It is multiplied by a probe scalar
        that is deliberately NOT a trunk parameter, which is exactly "a head that
        does not share the trunk" -- ``allow_unused`` then returns ``None`` for
        every theta_trunk, i.e. the zero vector, which is the analytic target.
        (``tests/test_refcv6_grad_conflict.py`` repeats the control against a
        REAL detached head, so the surrogate is checked, not trusted.)
        """
        p = list(self.trunk_params)
        g = self.grads(loss_traj, p, retain_graph=True)
        st_self = cosine_stats(g, g, dtype=self._dtype)
        gneg = [None if x is None else -x for x in g]
        st_neg = cosine_stats(g, gneg, dtype=self._dtype)
        src = loss_aux if loss_aux is not None else loss_traj
        probe = torch.ones((), dtype=src.dtype, device=src.device,
                           requires_grad=True)
        det_loss = src.detach() * probe
        g_det = torch.autograd.grad(det_loss, p, retain_graph=retain_graph,
                                    allow_unused=True)
        st_det = cosine_stats(g, g_det, dtype=self._dtype)
        det_reading = GroupReading.from_stats("detached", st_det)
        fails = []
        if st_self["cos"] != 1.0:
            fails.append(f"cos(g,g) read {st_self['cos']!r}, must be exactly +1.0")
        if st_neg["cos"] != -1.0:
            fails.append(f"cos(g,-g) read {st_neg['cos']!r}, must be exactly -1.0")
        if not st_det["degenerate"]:
            fails.append("the detached control is not degenerate: the aux "
                         "gradient on the trunk is non-zero, so the head is NOT "
                         "detached and the control measures nothing")
        if st_det["norm_aux"] != 0.0:
            fails.append(f"detached |g_aux| read {st_det['norm_aux']!r}, must be "
                         f"exactly 0.0")
        if not math.isnan(st_det["cos"]):
            fails.append(f"detached cos read {st_det['cos']!r}; a 0/0 quotient "
                         f"must surface as NaN, never as a smoothed 0.0")
        if det_reading.conflict != 0.0:
            fails.append(f"detached conflict read {det_reading.conflict!r}, "
                         f"must be exactly 0.0")
        res = ControlResult(
            self_cos=st_self["cos"], negated_cos=st_neg["cos"],
            detached_cos=st_det["cos"], detached_conflict=det_reading.conflict,
            detached_norm_aux=st_det["norm_aux"],
            detached_degenerate=st_det["degenerate"],
            ok=not fails, failures=tuple(fails))
        self._controls = res
        return res

    def self_check(self, loss_traj, loss_aux=None, *, retain_graph: bool = True
                   ) -> ControlResult:
        """:meth:`controls`, and ⛔ **RAISE** if any control missed.

        ``PREREG_BEV_CAPACITY_COMPETITION.md`` §4: *"THE PROBE REFUSES TO RUN IF
        ANY MISSES."* A cross-check re-derived from the thing it checks measures
        determinism; these three are derived independently of this code, so a
        miss means the reading is not evidence and the run should not spend the
        card pretending otherwise.
        """
        res = self.controls(loss_traj, loss_aux, retain_graph=retain_graph)
        if not res.ok and self.cfg.require_controls:
            raise ControlFailure(
                "the gradient-conflict detector's analytic controls did not read "
                "their known values, so its per-step number is not evidence:\n  - "
                + "\n  - ".join(res.failures))
        return res

    # -- provenance --------------------------------------------------------- #
    def provenance(self) -> dict:
        return {"config": self.cfg.as_dict(),
                "n_trunk_tensors": len(self.trunk_params),
                "n_trunk_params": int(sum(p.numel() for p in self.trunk_params)),
                "n_head_tensors": len(self.head_params),
                "n_head_params": int(sum(p.numel() for p in self.head_params)),
                "groups": {g: {"n_tensors": len(i),
                               "n_params": int(sum(self.trunk_params[k].numel()
                                                   for k in i))}
                           for g, i in self.group_index.items()},
                "controls": (self._controls.as_dict() if self._controls else None)}


# --------------------------------------------------------------------------- #
#  the flag
# --------------------------------------------------------------------------- #
def enabled_for_arm(refcv6_flags=None, *, override=None, aux_present: bool = True
                    ) -> bool:
    """⛔ **ON by default for refcv6 arms, off otherwise.**

    ``refcv6_flags``: the run's refcv6 block (``DecoderConfig.refcv6``).
    ``None`` == not a refcv6 arm (``refc_v3_train`` stamps ``refcv6: null`` for
    the baseline precisely so absence and baseline are distinguishable).

    ``override``: ``--conflict-detector on|off``. ``None`` == take the default.

    ``aux_present``: with no perception loss there is no second gradient; the
    detector stays off rather than logging NaN for a whole run.
    """
    if override is not None:
        return bool(override)
    return bool(refcv6_flags is not None and aux_present)
