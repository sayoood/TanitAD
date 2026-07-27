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
]

#: The primary the gate stops on. Named, so it can never be quietly swapped.
PRIMARY_NAME = "pseudosim_composite_PSS_recovery_progress"
PRIMARY_RATIONALE = (
    "MEASURED: the ADE-optimal pick collides 4.7x more often than the rule-"
    "optimal pick (3.36 % vs 0.71 %, separated). PUBLISHED: L2/ADE vs closed-"
    "loop Driving Score rho = -0.36 (p = 0.43); Ego Progress rho = 0.83. An "
    "early-stop gated on ADE stops the run in the wrong place with confidence.")
#: What this gate will NOT stop on, at any threshold.
REFUSED_PRIMARY = "ade_0_2s (diagnostic only — see PRIMARY_RATIONALE)"


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

    ``goal_kwargs_fn(batch_size, device) -> dict`` supplies the head's optional
    conditioning. The default is the **withheld/unknown** state — zeros, i.e. the
    ``vt_band``/``route`` defaults ``_goal_inputs`` falls back to — which is what a
    deployed car has when no route is supplied. That choice is RECORDED in
    :attr:`provenance` because a readout taken with a route input is a different
    measurement from one taken without, and the program has already been burned by
    a decoder evaluated at ``nav_cmd=None`` it never trained against.
    """

    def __init__(self, world, head, *, device="cpu", amp: bool = False,
                 goal_kwargs_fn=None, expect_dense: bool = True):
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
        self.provenance = {
            "surface": "deployable (encode_window -> head -> SELECTED wp_seq)",
            "selected_not_oracle": True,
            "goal_conditioning": ("withheld/unknown defaults (zeros) — the "
                                  "deployed no-route state"
                                  if goal_kwargs_fn is None else
                                  "caller-supplied goal_kwargs_fn"),
            "horizons": list(horizons),
            "amp": self.amp,
        }

    @torch.no_grad()
    def traj(self, frames, v0, goal=None):
        """frames [B, W, C, H, W] · v0 [B] -> the SELECTED plan [B, S, 2]."""
        was_training = self.head.training
        self.head.eval(); self.world.eval()
        try:
            b = int(frames.shape[0])
            kw = (self.goal_kwargs_fn(b, self.device)
                  if self.goal_kwargs_fn is not None else {})
            if goal is not None:
                kw = {**kw, **goal} if isinstance(goal, dict) else kw
            amp_on = self.amp and str(self.device) == "cuda"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
                states = self.world.encode_window(frames.to(self.device))
                out = self.head(states, v0.to(self.device), **kw)
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
              goal_kwargs_fn=None, diagnostics=None, verbose=False) -> dict:
        """Run pseudo-simulation on the deployable surface, then :meth:`observe`.

        ``episodes`` are HELD-OUT episode objects (``.poses`` / ``.frames``).
        Determinism matters: the same episodes, stride and grid must be passed
        every probe or :meth:`observe` raises :class:`WindowAlignmentError`.
        """
        ps, _ = _taniteval()
        grid = self.cfg.resolved_grid()
        planner = DeployableSurfacePlanner(
            world, head, device=device, amp=self.cfg.amp,
            goal_kwargs_fn=goal_kwargs_fn)
        pw = ps.pseudo_evaluate(planner, episodes, grid, device=device,
                                stride=self.cfg.stride,
                                horizon=self.cfg.horizon,
                                batch=self.cfg.batch, verbose=verbose)
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
            "components_admitted": self._pinned_admitted,
            "components_pinned_at_step": self.history[0]["step"],
        }
        return rec

    def _composite_of(self, pw):
        """Per-window composite under the PINNED admitted-component set."""
        import numpy as np
        ps, _ = _taniteval()
        sc = ps.score_windows(pw)
        comps = {k: sc[k] for k in ("ego_progress", "recovery", "comfort")}
        comps["no_collision"] = None
        comps["ttc"] = None
        if self._pinned_ranges is None:
            ranges = ps.discriminative_range(comps)
            try:
                comp = ps.composite(comps, ranges, weights=self.cfg.weights)
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
                                weights=self.cfg.weights)
        node = {"grid": pw.get("grid"), "traffic_mode": pw.get("traffic_mode"),
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
