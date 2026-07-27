"""THE MID-RUN HELD-OUT GATE — the fix for the largest measured waste in the program.

WHY THIS EXISTS
---------------
MEASURED on the v4 30k run: **~29.5 GPU-h — half the run — went into training past
the best checkpoint** while *every training term improved*. Held-out selection was
separated-WORSE the whole time and nothing was watching. There was no held-out
early-stop signal at all; the four diagnostics that would have shown it were
computed 601x and discarded by the row-writer.

This module supplies the missing signal: probe the **deployable surface** on
**held-out episodes** at a fixed step cadence, and **stop the run when the
held-out primary is separated-worse for two consecutive probes**.

⚠️ **THE PRIMARY IS THE MAP-FREE COMPOSITE, NOT ``ade_0_2s``.** This is not a
style preference, it is the pre-registered axis:

* the **ADE-optimal pick collides 4.7x more often** than the rule-optimal pick
  (3.36 % vs 0.71 %, separated);
* ``PUBLISHED`` L2/ADE vs closed-loop Driving Score is **rho = -0.36, p = 0.43**,
  while Ego Progress is **rho = 0.83**.

An early-stop gated on ADE would therefore stop the run at the wrong place with
full confidence. :class:`HeldoutGate` refuses to accept ADE as its primary —
:meth:`HeldoutGate.observe` is fed the per-window **composite** and nothing else,
and ``ade_0_2s`` travels only as a reported diagnostic.

WHAT "THE DEPLOYABLE SURFACE" MEANS HERE
-----------------------------------------
:class:`DeployableSurfacePlanner` is the trained stack behaving exactly as the car
would: encode the observed window, run the head, and return **the SELECTED
trajectory** (``wp_seq``). Not the oracle-in-fan, not the best candidate, not a
teacher-forced rollout. Selection is the thing that regressed on v4, so a probe
that bypasses the selector cannot see the defect it exists to catch.

The surface is measured under **pseudo-simulation** (:mod:`taniteval.pseudosim`),
which is **0.00 % out-of-envelope** by construction — sequential closed-loop is
90.2 % out of envelope at K=185 and cannot be used mid-run.

⭐ **WHAT GOAL THE PROBE HANDS THE HEAD** (2026-07-27). The real ``FlagshipV4Head``
has ``cond_vtarget=True`` and **refuses** to plan without a ``vt_band``, so until
this date ``--heldout-gate`` crashed at its first probe — ~2 000 optimizer steps
and several GPU-hours into the run. The fix is a **signature change**:
``goal_kwargs_fn(states, v0)`` (not ``(batch_size, device)``, which could carry
neither), plus a named option from :mod:`tanitad.train.heldout_goal` selected by
:attr:`HeldoutGateConfig.goal_option` / ``--heldout-goal``. The default is
:data:`GOAL_OPTION_DEFAULT` — read its note, **it is an agent's choice pending the
PI's override**, and it is one flag away from either alternative.

THE DECISION RULE — pre-registered here, before any checkpoint exists
---------------------------------------------------------------------
1. The first probe becomes the **incumbent**.
2. Every later probe is compared to the incumbent with the **paired
   episode-cluster bootstrap** (``taniteval.ci``, B=2000, unit = held-out
   episode), on the windows finite in BOTH arms. The probes are byte-aligned by
   construction (same episodes, same anchors, same grid, deterministic order), and
   :class:`WindowAlignmentError` fails loud if they ever are not — an unpaired
   "paired" test is not a weaker test, it is a wrong one.
3. ``separated_worse`` = the paired CI excludes zero **and** the delta is negative
   (the composite is higher-is-better).
4. The incumbent advances **only on a SEPARATED improvement**. A point estimate
   that merely drifted up must not become the bar, or the stop rule fires on the
   incumbent's luck rather than the model's decline.
5. **Two consecutive ``separated_worse`` probes stop the run.** One does not:
   ⚠️ a single separated probe is a sample, and this program has a standing rule
   that an unpowered read is not a refutation.

⚠️ **The admitted component set is PINNED at the first probe.**
:func:`taniteval.pseudosim.discriminative_range` decides admissibility from the
observed data, so re-deriving it every probe would let the composite silently
change definition mid-run and compare two different metrics. That is the
forking-paths failure ``GATE_PROTOCOL`` §0.3 forbids. The pinned ranges travel in
every emitted node and in the checkpoint.

RESUME SAFETY. The streak, the incumbent and the pinned ranges round-trip through
:meth:`HeldoutGate.state_dict` / :meth:`HeldoutGate.load_state_dict`, so a pod
restart cannot silently reset the gate to "never seen a bad probe" — which would
reproduce the exact failure this module removes.

NOTHING HERE LAUNCHES A RUN.
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

__all__ = [
    "HeldoutGateConfig", "HeldoutGate", "DeployableSurfacePlanner",
    "WindowAlignmentError", "GateNotUsableError", "NonDensePlanError",
    "probe_grid", "PRIMARY_NAME", "PRIMARY_RATIONALE", "REFUSED_PRIMARY",
    "GOAL_OPTION_DEFAULT", "GOAL_OPTION_PROVENANCE",
]

#: ⛔ The progress term the gate scores under. VERSIONED (2026-07-28): the
#: published ``clamp_v1`` term is ONE-SIDED — it charges nothing for over-travel,
#: and MEASURED on the 2026-07-27 panel's own rows it returned **n.s.** on a
#: **5.65x** along-track RMS improvement while SEPARATING a **3.36x** degradation
#: of the same axis. A v5 run gated on it cannot see a longitudinal lever.
#: Set to ``"clamp_v1"`` to reproduce a pre-2026-07-28 gate exactly.
PROGRESS_TERM = "twosided_v2"
#: The primary the gate stops on. Named — INCLUDING the progress term — so it can
#: never be quietly swapped OR silently redefined under a stable name.
PRIMARY_NAME = f"pseudosim_composite_PSS_recovery_progress@{PROGRESS_TERM}"
#: What every PSS number published before 2026-07-28 was computed under.
PRIMARY_NAME_PUBLISHED_THROUGH_20260727 = (
    "pseudosim_composite_PSS_recovery_progress@clamp_v1")
PRIMARY_RATIONALE = (
    "MEASURED: the ADE-optimal pick collides 4.7x more often than the rule-"
    "optimal pick (3.36 % vs 0.71 %, separated). PUBLISHED: L2/ADE vs closed-"
    "loop Driving Score rho = -0.36 (p = 0.43); Ego Progress rho = 0.83. An "
    "early-stop gated on ADE stops the run in the wrong place with confidence.")
#: What this gate will NOT stop on, at any threshold.
REFUSED_PRIMARY = "ade_0_2s (diagnostic only — see PRIMARY_RATIONALE)"

#: ⭐ THE GOAL STATE THE PROBE CONDITIONS ON (2026-07-27).
#:
#: ⚠️ **This default is an AGENT'S CHOICE PENDING THE PI's OVERRIDE, not the PI's
#: decision.** ``VTBAND_DECISION.md`` prices all of them and explicitly declines to
#: choose; the wiring stream picked ``"dropped"`` so the gate could run at all, and
#: made the alternatives **one flag away** (``--heldout-goal band0|produced``) so
#: overriding costs nothing. Change this constant OR pass the flag — both work, and
#: neither requires touching the probe.
#:
#: WHY ``dropped`` (MEASURED, `VTBAND_DECISION.md` §2–§5, arm
#: ``flagship-v4-fromscratch`` @ step 15000, 528 windows / 8 held-out episodes):
#:
#: * it is **in-distribution by construction** — ``V15Config.goal_dropout = 0.5``
#:   ships, ``V4Config`` inherits it and the trainer never overrides it, so
#:   ``VT_DROPPED``/``ROUTE_DROPPED`` are **learned embedding rows** seen on ~50 %
#:   of every training batch. They are the most frequently trained value of that
#:   channel, not a zero-fill;
#: * it makes :attr:`DeployableSurfacePlanner.provenance`'s "no-route state" claim
#:   TRUE, which ``band0`` does not;
#: * it gives the early-stop the **largest detectable drop** (−0.4157 paired vs a
#:   randomised-selection degradation) because it has the highest baseline (0.6383).
#:
#: ⛔ ``band0`` is NOT a neutral zero: ``VTARGET_TOKENS[0] == "v_stop"`` and
#: ``route 0 == ROUTE_LEFT``. It costs −0.0621 [−0.0878, −0.0371] (separated), 93 %
#: of it from the VTARGET channel alone, by making the planner travel 9.1 % less.
#: ⛔ ``zeros_naive`` (also zeroing ``vt_speed``) is WORSE THAN USELESS — it brakes
#: the probe and NaNs ``recovery``, so the gate reads HEALTHIER while probing a
#: braking planner. It is reachable only by naming it explicitly.
GOAL_OPTION_DEFAULT = "dropped"
#: Who chose :data:`GOAL_OPTION_DEFAULT`, carried into every probe record so a
#: reader of the JSON never has to guess whether this was adjudicated.
GOAL_OPTION_PROVENANCE = (
    "default 'dropped' chosen by the vtband-WIRING stream 2026-07-27, PENDING THE "
    "PI's OVERRIDE — VTBAND_DECISION.md priced the options and deliberately did "
    "not choose. Override with --heldout-goal {band0,produced}; nothing else "
    "changes.")


class WindowAlignmentError(AssertionError):
    """Two probes did not evaluate the same windows — a paired test is invalid."""


class GateNotUsableError(RuntimeError):
    """The gate could not establish a usable primary. Never downgraded to a warning.

    A silently-disabled early-stop is precisely the defect this module removes, so
    an unusable gate raises rather than letting a run proceed unwatched.
    """


class NonDensePlanError(ValueError):
    """The head does not emit a dense 0.1 s plan, so pseudo-sim scoring is invalid."""


# --------------------------------------------------------------------------- #
# lazy taniteval import (mirrors scripts/gate_emitters.py) — keeps `tanitad`     #
# importable on a box that has not put `taniteval` on the path.                 #
# --------------------------------------------------------------------------- #
def _taniteval():
    """Import and return ``(pseudosim, ci)``, inserting the repo paths if needed."""
    repo = Path(__file__).resolve().parents[3]          # <repo>
    for p in (repo / "taniteval", repo / "stack"):
        s = str(p)
        if p.is_dir() and s not in sys.path:
            sys.path.insert(0, s)
    from taniteval import ci as _ci                      # noqa: PLC0415
    from taniteval import pseudosim as _ps               # noqa: PLC0415
    return _ps, _ci


def probe_grid(**kw):
    """The MID-RUN probe grid: a strict subset of the shipped 0 %-out-of-envelope grid.

    The full shipped grid is 7 headings x 3 longitudinal offsets = 21 points per
    anchor. Mid-run this runs every ``every`` steps for the whole run, so the
    default here is the cheapest grid that still measures the thing the gate is
    for: ``recovery`` is **undefined at the unperturbed point by construction**, so
    a probe grid without a perturbed heading measures progress only and would be
    blind to exactly the error-recovery collapse we are watching for.

    ``(-8, 0, +8) deg x (0,)`` keeps two perturbed headings and the reference
    point at 3 planner calls per anchor — 7x cheaper than the full grid — and is
    still inside the MEASURED envelope (|dpsi| <= 12 deg), which
    :func:`taniteval.pseudosim.assert_grid_in_envelope` re-proves on every probe.
    """
    ps, _ = _taniteval()
    kw.setdefault("dyaw_deg", (-8.0, 0.0, 8.0))
    kw.setdefault("dlon_steps", (0,))
    return ps.GridSpec(**kw)


# --------------------------------------------------------------------------- #
# the deployable surface                                                       #
# --------------------------------------------------------------------------- #
class DeployableSurfacePlanner:
    """(world, head) -> the ``.traj(frames, v0, goal)`` protocol pseudo-sim calls.

    THE DEPLOYABLE SURFACE, precisely: ``encode_window`` -> ``head`` -> the
    **selected** trajectory ``out["wp_seq"]``. No oracle, no fan statistic, no
    teacher forcing. v4's regression was in SELECTION, so a probe that read the
    oracle would have reported the run as healthy while it decayed.

    ⭐ ``goal_kwargs_fn(states, v0) -> dict`` supplies the head's optional
    conditioning. **The protocol takes ``(states, v0)``, NOT ``(batch_size,
    device)``** — that is a 2026-07-27 SIGNATURE change, and it is the fix, not a
    cosmetic one. The old signature carried neither ``v0`` nor ``states``, so it
    could not express *any* goal option faithfully:

    * ``_goal_inputs`` sets ``vt_speed = v0``, and :meth:`FlagshipV15Head.select`
      clamps ``vt_speed`` into the reachable band around ``v0`` — so a fn without
      ``v0`` either fabricates a speed target or zeroes it, and **zeroing it makes
      the selector rank up the maximally decelerating candidate**
      (``v_goal = (v0 - 5)⁺``; see :mod:`tanitad.train.heldout_goal`);
    * ``produced`` (the model's own ``goal_head``) is a function of the encoded
      window, which the old ``traj`` computed *after* it had already built ``kw``.

    ⛔ ``goal_kwargs_fn=None`` means **no goal channels at all**, and a head with
    ``cond_vtarget=True`` REFUSES it (``ValueError``). That refusal is the head's
    own ``ego_guard`` and is correct; it is not a fallback. The gate never takes
    that path — :meth:`HeldoutGate.probe` builds a fn from
    :attr:`HeldoutGateConfig.goal_option` (default ``"dropped"``).

    The choice is RECORDED verbatim in :attr:`provenance`, because a readout taken
    with a route input is a different measurement from one taken without, and the
    program has already been burned by a decoder evaluated at ``nav_cmd=None`` it
    never trained against.
    """

    def __init__(self, world, head, *, device="cpu", amp: bool = False,
                 goal_kwargs_fn=None, expect_dense: bool = True,
                 goal_option: str | None = None):
        self.world, self.head = world, head
        self.device = device
        self.amp = bool(amp)
        self.goal_kwargs_fn = goal_kwargs_fn
        horizons = tuple(getattr(head.cfg, "horizons", ()))
        if expect_dense and horizons != tuple(range(1, len(horizons) + 1)):
            raise NonDensePlanError(
                f"pseudo-simulation scores accelerations/jerk by finite "
                f"differences at dt=0.1 s, so it needs a DENSE consecutive plan "
                f"(horizons == 1..K). This head emits {horizons!r}. Scoring a "
                f"coarse plan as if it were dense silently divides every "
                f"derivative by the wrong dt — pass expect_dense=False only with "
                f"a written reason.")
        self.horizons = horizons
        self.goal_option = goal_option
        if goal_kwargs_fn is None:
            cond = ("⛔ NONE — no goal channels are supplied at all. A head with "
                    "cond_vtarget/cond_route REFUSES this (ValueError); it is "
                    "the RED baseline, not a withheld-goal default.")
        else:
            cond = (f"goal option {goal_option!r} via goal_kwargs_fn(states, v0)"
                    if goal_option else "caller-supplied goal_kwargs_fn(states, v0)")
        self.provenance = {
            "surface": "deployable (encode_window -> head -> SELECTED wp_seq)",
            "selected_not_oracle": True,
            "goal_conditioning": cond,
            "goal_option": goal_option,
            "goal_protocol": "goal_kwargs_fn(states, v0)",
            "horizons": list(horizons),
            "amp": self.amp,
        }
        if goal_option:
            from tanitad.train.heldout_goal import OPTION_MEANING
            self.provenance["goal_option_meaning"] = OPTION_MEANING.get(
                goal_option, goal_option)

    @torch.no_grad()
    def traj(self, frames, v0, goal=None):
        """frames [B, W, C, H, W] · v0 [B] -> the SELECTED plan [B, S, 2].

        ⭐ ``states`` is encoded BEFORE ``kw`` is built — that ordering is the
        wiring fix, not an incidental refactor: ``produced`` derives the goal
        from ``states``, so building ``kw`` first made that option unreachable.
        """
        was_training = self.head.training
        self.head.eval(); self.world.eval()
        try:
            amp_on = self.amp and str(self.device) == "cuda"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
                states = self.world.encode_window(frames.to(self.device))
                v0d = v0.to(self.device)
                kw = (self.goal_kwargs_fn(states, v0d)
                      if self.goal_kwargs_fn is not None else {})
                if isinstance(goal, dict):
                    kw = {**kw, **goal}
                out = self.head(states, v0d, **kw)
            return out["wp_seq"].float().cpu()
        finally:
            self.world.train(was_training); self.head.train(was_training)


# --------------------------------------------------------------------------- #
# config                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class HeldoutGateConfig:
    """Everything the gate decides on. Registered BEFORE any checkpoint exists."""

    every: int = 2000              # step cadence between probes
    episodes: int = 8              # held-out episodes probed (unit of the bootstrap)
    stride: int = 8                # anchor stride inside an episode
    horizon: int = 20              # plan steps scored (2.0 s at dt=0.1)
    batch: int = 16
    n_boot: int = 2000             # taniteval.ci.DEFAULT_N_BOOT
    seed: int = 0
    patience: int = 2              # consecutive separated-worse probes that STOP
    amp: bool = False
    enabled: bool = True
    #: minimum step before the first probe (a probe at step 0 is the warm trunk)
    first_probe_step: int = 0
    grid: object | None = None     # a GridSpec; None -> probe_grid()
    weights: dict | None = None    # composite weights; None -> pseudosim default
    #: ⭐ THE GEOMETRY THE PROBE RE-RENDERS THROUGH (2026-07-27). A
    #: :class:`~tanitad.data.calib.CanonicalFrame` (or its ``to_dict()``), read
    #: from the cache the held-out episodes came from — never re-derived.
    #: ``None`` == the deployed 256x256 pinhole frame, and
    #: ``taniteval.clhorizon.assert_warp_frame`` REFUSES that on any other
    #: raster. ``clhorizon.LEGACY_WARP`` states "the shipped 266/128 constants
    #: on whatever raster" for tests that pin pre-2026-07-27 behaviour.
    frame: object | None = None
    #: ⛔ the VERSIONED ego-progress term (see :data:`PROGRESS_TERM`). Pin it to
    #: ``"clamp_v1"`` only to reproduce a pre-2026-07-28 gate; that term cannot
    #: see over-travel and therefore cannot see a longitudinal lever.
    progress_term: str = PROGRESS_TERM
    #: ⭐ WHAT GOAL STATE THE PROBE CONDITIONS ON — see :data:`GOAL_OPTION_DEFAULT`.
    #: One of :data:`tanitad.train.heldout_goal.CANDIDATES` (plus the priced trap
    #: and the two channel-isolation diagnostics). ``--heldout-goal`` on the
    #: trainer sets it; changing it is one flag, by design.
    goal_option: str = GOAL_OPTION_DEFAULT

    def resolved_grid(self):
        return probe_grid() if self.grid is None else self.grid


# --------------------------------------------------------------------------- #
# the gate                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class _Probe:
    step: int
    value: float
    per_window: list
    eid: list
    digest: str
    n_windows: int
    n_episodes: int


class HeldoutGate:
    """Probe the deployable surface on held-out episodes and stop a decayed run.

    Two entry points, deliberately split so the DECISION is testable without a
    GPU, a dataset or a model:

    * :meth:`observe` — pure arithmetic on a per-window primary. This is where the
      stop rule lives and where the adversarial tests drive it.
    * :meth:`probe` — runs pseudo-simulation on the deployable surface and hands
      the resulting per-window composite to :meth:`observe`.
    """

    def __init__(self, cfg: HeldoutGateConfig | None = None):
        self.cfg = cfg or HeldoutGateConfig()
        self.history: list[dict] = []
        self.worse_streak = 0
        self.stop = False
        self.stop_reason: str | None = None
        self._incumbent: _Probe | None = None
        self._pinned_ranges: dict | None = None
        self._pinned_admitted: dict | None = None

    # ------------------------------------------------------------- cadence --
    def due(self, step: int) -> bool:
        """Is a probe due at ``step``? (cadence is fixed, never data-dependent)."""
        if not self.cfg.enabled or self.cfg.every <= 0:
            return False
        return step >= self.cfg.first_probe_step and step % self.cfg.every == 0

    # ------------------------------------------------- the decision (pure) --
    def observe(self, step: int, per_window, eid, *, diagnostics=None) -> dict:
        """Fold one probe's per-window PRIMARY into the stop decision.

        ``per_window`` are the per-(window, grid point) composite values (NaN
        allowed — a NaN is an undefined window, e.g. ``recovery`` at the
        unperturbed point, not a zero). ``eid`` are their held-out episode ids, the
        bootstrap's resampling unit.

        ``diagnostics`` (e.g. ``ade_0_2s``) is carried into the record and NEVER
        consulted by the rule — see :data:`REFUSED_PRIMARY`.
        """
        import numpy as np
        _, cci = _taniteval()

        v = np.asarray(per_window, dtype=float)
        eid = [str(e) for e in eid]
        if v.ndim != 1:
            raise ValueError(f"per_window must be 1-D, got {v.shape}")
        if len(eid) != v.size:
            raise ValueError(f"eid/per_window length mismatch: "
                             f"{len(eid)} vs {v.size}")
        finite = np.isfinite(v)
        if finite.sum() < 2 or len(set(np.asarray(eid)[finite])) < 2:
            raise GateNotUsableError(
                f"probe at step {step} has {int(finite.sum())} finite windows over "
                f"{len(set(np.asarray(eid)[finite]))} episodes — an episode-cluster "
                f"bootstrap needs >= 2 of each. The gate FAILS LOUD rather than "
                f"silently disabling itself: an unwatched run is the defect this "
                f"module exists to remove.")
        digest = _digest(eid)
        point = float(np.nanmean(v))
        probe = _Probe(step=step, value=point, per_window=v.tolist(), eid=eid,
                       digest=digest, n_windows=int(finite.sum()),
                       n_episodes=int(len(set(eid))))

        rec = {
            "step": step,
            "primary": PRIMARY_NAME,
            "primary_value": round(point, 6),
            "n_windows": probe.n_windows,
            "n_episodes": probe.n_episodes,
            "diagnostics": dict(diagnostics or {}),
            "_diagnostics_are_not_the_rule": REFUSED_PRIMARY,
        }

        if self._incumbent is None:
            self._incumbent = probe
            rec.update({"role": "incumbent (first probe)", "paired": None,
                        "separated_worse": False, "worse_streak": 0,
                        "stop": False, "incumbent_step": step})
            self.history.append(rec)
            return rec

        inc = self._incumbent
        if probe.digest != inc.digest:
            raise WindowAlignmentError(
                f"probe at step {step} evaluated a DIFFERENT window set than the "
                f"incumbent at step {inc.step} (episode-id digest "
                f"{probe.digest[:12]} vs {inc.digest[:12]}). The paired "
                f"episode-cluster bootstrap requires the same windows in the same "
                f"order; running it on mismatched arms is not a weaker test, it is "
                f"a wrong one. Fix the probe's episode/stride/grid determinism.")

        a = np.asarray(probe.per_window, dtype=float)
        b = np.asarray(inc.per_window, dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 2 or len(set(np.asarray(eid)[m])) < 2:
            raise GateNotUsableError(
                f"probe at step {step} shares only {int(m.sum())} jointly-finite "
                f"windows with the incumbent — cannot form a paired interval.")
        paired = cci.paired_episode_cluster_bootstrap(
            a[m], b[m], list(np.asarray(eid)[m]),
            n_boot=self.cfg.n_boot, seed=self.cfg.seed)
        # composite is HIGHER-IS-BETTER: worse = the paired delta is negative.
        separated_worse = bool(paired["separated"] and paired["delta"] < 0)
        separated_better = bool(paired["separated"] and paired["delta"] > 0)

        self.worse_streak = self.worse_streak + 1 if separated_worse else 0
        if separated_better:
            # ⚠️ the incumbent advances ONLY on a separated improvement — a lucky
            # point estimate must never become the bar the stop rule fires against.
            self._incumbent = probe
        if self.worse_streak >= self.cfg.patience and not self.stop:
            self.stop = True
            self.stop_reason = (
                f"held-out primary ({PRIMARY_NAME}) separated-WORSE than the "
                f"incumbent (step {inc.step}) for {self.worse_streak} consecutive "
                f"probes (patience={self.cfg.patience}); last delta "
                f"{paired['delta']} [{paired['lo']}, {paired['hi']}], "
                f"{paired['estimator']}, n_episodes={paired['n_episodes']}. "
                f"Training past this point is the ~29.5 GPU-h this gate exists to "
                f"stop spending.")

        rec.update({
            "role": ("new incumbent (separated better)" if separated_better
                     else "challenger"),
            "incumbent_step": self._incumbent.step,
            "incumbent_value": round(self._incumbent.value, 6),
            "paired": paired,
            "separated_worse": separated_worse,
            "separated_better": separated_better,
            "worse_streak": self.worse_streak,
            "patience": self.cfg.patience,
            "stop": self.stop,
            "stop_reason": self.stop_reason,
        })
        self.history.append(rec)
        return rec

    # ---------------------------------------------------- the probe (GPU) --
    def probe(self, step: int, world, head, episodes, *, device="cpu",
              goal_kwargs_fn=None, diagnostics=None, verbose=False,
              frame=None, goal_head=None) -> dict:
        """Run pseudo-simulation on the deployable surface, then :meth:`observe`.

        ``episodes`` are HELD-OUT episode objects (``.poses`` / ``.frames``).
        Determinism matters: the same episodes, stride and grid must be passed
        every probe or :meth:`observe` raises :class:`WindowAlignmentError`.

        ⭐ **GOAL CONDITIONING.** When ``goal_kwargs_fn`` is not given, one is
        built from :attr:`HeldoutGateConfig.goal_option` (default
        :data:`GOAL_OPTION_DEFAULT` == ``"dropped"``). Before 2026-07-27 no fn was
        built at all and the probe handed the real head ``kw = {}``, which it
        refuses — ``--heldout-gate`` therefore died at its FIRST probe, ~2 000
        optimizer steps and several GPU-hours into a run. ``goal_head`` is
        required only by ``goal_option="produced"``.

        ⭐ ``frame`` is the :class:`~tanitad.data.calib.CanonicalFrame` the
        held-out pixels were BUILT at — the TRAIN frame, i.e. the sub-frame if
        ``--v2-subframe`` is in force. It reaches
        ``clhorizon.warp_frames`` and decides the re-render's projection.
        ``None`` keeps the deployed 256x256 pinhole warp and is REFUSED on any
        other raster: **the probe grid is ``(-8, 0, +8) deg``, and on v5's
        176x624 cylindrical frame the deployed warp misplaces source pixels by
        a mean of 46.3 px against a true shift of 42.7 px** — so an unstated
        frame here would stop (or fail to stop) a live run on an observation
        the camera could never have produced.
        """
        ps, _ = _taniteval()
        grid = self.cfg.resolved_grid()
        option = getattr(self.cfg, "goal_option", GOAL_OPTION_DEFAULT)
        if goal_kwargs_fn is None:
            from tanitad.train import heldout_goal as _HGoal
            # ⛔ raises rather than falling back: 'dropped' on a config with
            # goal_dropout == 0 would probe untrained N(0, 0.02) rows, and
            # 'produced' without a goal_head has no model-side producer. A gate
            # that silently substituted something cheaper is the failure this
            # module exists to remove.
            goal_kwargs_fn = _HGoal.make_goal_kwargs_fn(
                option, head.cfg, goal_head=goal_head)
        planner = DeployableSurfacePlanner(
            world, head, device=device, amp=self.cfg.amp,
            goal_kwargs_fn=goal_kwargs_fn, goal_option=option)
        frame = self.cfg.frame if frame is None else frame
        pw = ps.pseudo_evaluate(planner, episodes, grid, device=device,
                                stride=self.cfg.stride,
                                horizon=self.cfg.horizon,
                                batch=self.cfg.batch, verbose=verbose,
                                frame=frame)
        if pw.get("_empty"):
            raise GateNotUsableError(
                f"pseudo_evaluate produced no windows at step {step}: the "
                f"held-out episodes are shorter than window+horizon+|dlon|. Give "
                f"the gate longer episodes or a smaller horizon.")
        value, eid, node = self._composite_of(pw)
        rec = self.observe(step, value, eid, diagnostics=diagnostics)
        rec["pseudosim"] = {
            "grid": node.get("grid"),
            "envelope_proof": pw.get("envelope_proof"),
            "traffic_mode": node.get("traffic_mode"),
            "rollout_steps_executed": pw.get("rollout_steps_executed", 0),
            "planner_calls": pw.get("planner_calls"),
            "horizon_s": pw.get("horizon_s"),
            "estimator": node.get("_estimator"),
            "surface": planner.provenance,
            "warp": pw.get("warp"),
            "goal_option": option,
            "goal_option_provenance": GOAL_OPTION_PROVENANCE,
            "components_admitted": self._pinned_admitted,
            "components_pinned_at_step": self.history[0]["step"],
        }
        return rec

    def _composite_of(self, pw):
        """Per-window composite under the PINNED admitted-component set."""
        import numpy as np
        ps, _ = _taniteval()
        term = getattr(self.cfg, "progress_term", PROGRESS_TERM)
        sc = ps.score_windows(pw, progress_term=term)
        comps = {k: sc[k] for k in ("ego_progress", "recovery", "comfort")}
        comps["no_collision"] = None
        comps["ttc"] = None
        if self._pinned_ranges is None:
            ranges = ps.discriminative_range(comps)
            try:
                comp = ps.composite(comps, ranges, weights=self.cfg.weights,
                                    progress_term=term)
            except ps.VacuousMetric as exc:
                raise GateNotUsableError(
                    f"the FIRST probe cannot form a composite: {exc} The gate "
                    f"raises instead of falling back to ADE — see "
                    f"PRIMARY_RATIONALE.") from exc
            self._pinned_ranges = ranges
            self._pinned_admitted = dict(comp["weights_admitted"])
        else:
            # ⚠️ PINNED: re-deriving admissibility per probe would let the metric
            # change definition mid-run and compare two different composites.
            comp = ps.composite(comps, self._pinned_ranges,
                                weights=self.cfg.weights, progress_term=term)
        node = {"metric_id": comp.get("name"), "progress_term": term,
                "grid": pw.get("grid"), "traffic_mode": pw.get("traffic_mode"),
                "_estimator": f"paired episode-cluster bootstrap "
                              f"(B={self.cfg.n_boot}, unit = held-out episode)"}
        return np.asarray(comp["value"], dtype=float), list(pw["eid"]), node

    # --------------------------------------------------------- resume state --
    def state_dict(self) -> dict:
        inc = self._incumbent
        return {
            "worse_streak": int(self.worse_streak),
            "stop": bool(self.stop),
            "stop_reason": self.stop_reason,
            "history": list(self.history),
            "pinned_ranges": self._pinned_ranges,
            "pinned_admitted": self._pinned_admitted,
            "incumbent": (None if inc is None else
                          {"step": inc.step, "value": inc.value,
                           "per_window": list(inc.per_window), "eid": list(inc.eid),
                           "digest": inc.digest, "n_windows": inc.n_windows,
                           "n_episodes": inc.n_episodes}),
        }

    def load_state_dict(self, sd: dict) -> None:
        """Restore EXACTLY, streak included.

        A restart that reset ``worse_streak`` to 0 would let a decayed run walk
        past two bad probes forever by dying every other probe — the same
        unwatched-run failure with extra steps.
        """
        if not sd:
            return
        self.worse_streak = int(sd.get("worse_streak", 0))
        self.stop = bool(sd.get("stop", False))
        self.stop_reason = sd.get("stop_reason")
        self.history = list(sd.get("history", []))
        self._pinned_ranges = sd.get("pinned_ranges")
        self._pinned_admitted = sd.get("pinned_admitted")
        inc = sd.get("incumbent")
        self._incumbent = None if inc is None else _Probe(**inc)


def _digest(eid) -> str:
    """Identity of the evaluated window set — the paired test's alignment key."""
    h = hashlib.sha256()
    for e in eid:
        h.update(str(e).encode()); h.update(b"\x00")
    return h.hexdigest()
