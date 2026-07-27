"""THE ``vt_band`` DECISION — the candidate goal states for the mid-run held-out gate.

⛔ **NOTHING HERE IS WIRED IN AND NO DEFAULT CHANGES.** ``heldout_gate.py`` and
``train_flagship_v4.py`` are untouched by this module. It exists so the choice
the PI has to make can be *measured* instead of argued, and so that whichever
option is chosen is one named import rather than a fresh patch.

THE DEFECT THIS PRICES (MEASURED 2026-07-27, pod2, real caches)
---------------------------------------------------------------
``train_flagship_v4 --heldout-gate`` dies at its FIRST probe with
``ValueError: cond_vtarget is on but no vt_band supplied``:
:class:`~tanitad.train.heldout_gate.DeployableSurfacePlanner.traj` builds
``kw = {}``, and the real ``v4_config()`` head requires ``vt_band``. With
``--heldout-every 2000`` that is ~2 000 optimizer steps — several GPU-hours —
before the run dies. See ``stack/tests/test_heldout_gate_real_head.py``.

⭐ **THE CRASH IS NOT A BUG IN THE HEAD.** ``FlagshipV15Head.condition`` refusing
a missing ``vt_band`` is exactly what :mod:`taniteval.ego_guard` had to be
BOLTED ON to do for ``ego=`` — refuse to score a conditioned checkpoint blind
rather than silently produce a number. The head already had the guard. The bug is
that ``DeployableSurfacePlanner`` walks into it with an empty dict while its own
``provenance`` advertises "withheld/unknown defaults (zeros)".

WHAT ``zeros`` ACTUALLY MEANS — the reason this is a decision, not a patch
--------------------------------------------------------------------------
``train_flagship_v4._goal_inputs`` falls back to ``torch.zeros`` for BOTH
categorical channels. Read against the vocabularies those indices are not nulls
(all MEASURED, source in parentheses):

===============  =======  ==================================================
channel          index 0  what index 0 IS
===============  =======  ==================================================
``vt_band``      0        ``VTARGET_TOKENS[0]`` == ``"v_stop"`` — the STOPPED
                          target-speed band (``tanitad/lake/vocab.py``)
``route``        0        ``ROUTE_LEFT`` — ``refb_labels.py:76`` says
                          ``ROUTE_LEFT, ROUTE_STRAIGHT, ROUTE_RIGHT = range(3)``
``route_graded`` 0.0      graded curvature 0.0 == STRAIGHT, which CONTRADICTS
                          the ``route=LEFT`` it is passed beside
===============  =======  ==================================================

⇒ the "zeros" option does not probe a neutral planner. It probes one told, on
**every held-out window**: *"you are coming to a stop, and you are turning left"*
— with a graded channel simultaneously saying *straight*. That combination is not
merely a real band instead of a null; it is **self-contradictory**, and (checked
against the label minter, :func:`v4_labels`) it is a combination the training
distribution can barely contain: ``route_graded`` is
``tanh(mean_curv / CURV_TURN_PER_M)`` and a genuine ``ROUTE_LEFT`` window carries
``route_graded >= tanh(1) ~= 0.7616``, never 0.0.

AND IT MOVES THE SELECTOR, NOT ONLY THE EMBEDDING
--------------------------------------------------
``FlagshipV15Head.condition`` returns ``vt_keep = (band != VT_DROPPED)``, and
:meth:`FlagshipV15Head.select` multiplies the longitudinal selection penalty by
it. So the two candidates differ in the RANKING as well as in the conditioning:

* ``vt_band = 0``          -> ``vt_keep = True``  -> the longitudinal term is **ON**
* ``vt_band = VT_DROPPED`` -> ``vt_keep = False`` -> the longitudinal term is **OFF**

``sel_gate`` is zero-init, so at step 0 this is a no-op — and it is a LEARNED
scalar, so by the first real probe (step 2 000) it generally is not. An option
priced on an untrained model would report these as identical; that is the C13
shape and is why the numbers in ``VTBAND_DECISION.md`` are taken on a TRAINED
head (``flagship-v4-fromscratch`` @ step 29999), not on a from-scratch smoke.

IS ``VT_DROPPED`` IN-DISTRIBUTION? — established IN CODE, not by analogy
-------------------------------------------------------------------------
The brief's ``v2_ego_dropout`` precedent is *"a zeroed input is a real ablation
the model has seen"*. The equivalent for ``vt_band`` exists and is STRONGER,
because it is a learned embedding row rather than a zero-fill:

* ``V15Config.goal_dropout = 0.5`` (``flagship_v15.py``), inherited unchanged by
  ``V4Config``; ``train_flagship_v4.py`` **never overrides it** (grep: no
  occurrence outside ``_goal_inputs``), so the real v5 run trains at 0.5.
* ``condition()`` masks ``band -> VT_DROPPED`` and ``r -> ROUTE_DROPPED`` on a
  per-example Bernoulli(0.5) draw, i.e. **~50 % of every training batch**.
* ``VT_DROPPED = N_VTARGET_BANDS = 23`` and ``ROUTE_DROPPED = 4`` are their own
  ``nn.Embedding`` rows, deliberately distinct from the labeler's ``ROUTE_UNKNOWN
  = 3`` — the source comment says the two states are different on purpose.

⇒ ``VT_DROPPED`` is not an out-of-distribution probe. It is the single most
frequently trained value of that channel. ⚠️ It is still a *withheld-capability*
measurement, which is the ``ego=`` shape the brief flags — the honest statement
is that the model has **seen and been trained to handle** the withheld state,
which is precisely what ``ego=None`` (skipping the embedding entirely) was NOT.

⚠️ Note the asymmetry the dropout creates: ``goal_dropout`` only ever masks
TOWARD ``VT_DROPPED``. Band 0 (``v_stop``) is seen only on the ~50 % undropped
half AND only on windows whose true target speed really is a stop.

⛔ TODAY'S SIGNATURE CANNOT EXPRESS *ANY* OPTION FAITHFULLY — INCLUDING band0
-----------------------------------------------------------------------------
``goal_kwargs_fn(batch_size, device)`` receives neither ``v0`` nor ``states``.
But ``_goal_inputs`` sets ``vt_speed = v0``, and ``vt_speed`` is what
:meth:`FlagshipV15Head.select` clamps into the longitudinal ranking term::

    v_goal = max(min(vt_speed, v0 + reach), (v0 - reach).clamp_min(0))
    reach  = sel_accel_max * horizons[-1] * 0.1 = 2.5 * 20 * 0.1 = 5.0 m/s

so the three reachable choices are three DIFFERENT planners (MEASURED, at
``v0 = [12, 8, 3] m/s``):

=========================  ==========================  =========================
``vt_speed`` passed        ``v_goal``                   what the selector does
=========================  ==========================  =========================
``v0``  (``_goal_inputs``) ``[12.0, 8.0, 3.0]``         hold-v0 (a no-op prior)
``0``   (the naive patch)  ``[7.0, 3.0, 0.0]``          ⛔ rank up the MAXIMALLY
                                                        DECELERATING candidate
omitted (``None``)         —                            term skipped entirely
=========================  ==========================  =========================

⚠️ **This is the trap the tripwire's docstring points at.** "Build the zeros the
provenance promises" reads as the conservative patch and is the worst of the
three: it brakes on every window. And a braking plan has ``s_along -> 0``, so
``xt_hold = |dlat + s_along*tan(dpsi)|`` -> 0 with the gate's ``dlat = 0`` grid,
which makes ``recovery`` **NaN by construction** — the composite goes UP. A gate
patched that way reports a healthier run while probing a planner that brakes.
(Same mechanism a sibling stream hit at +0.1698; see ``VTBAND_DECISION.md`` §5.)

⇒ **whichever option is chosen, the fix is a SIGNATURE CHANGE, not a default
change.** This module therefore standardises on ``goal_kwargs_fn(states, v0)``,
which every option can express, and keeps the legacy ``(b, device)`` adapters
only so the naive patch can be measured rather than argued about.

THE OPTIONS
-----------
``band0``     ``_goal_inputs``' own fallback, verbatim (``vt_speed = v0``).
``dropped``   the learned withheld rows, i.e. ``goal_modes.neutral_goal``'s
              ``goal_dropout > 0`` branch.
``produced``  the model's OWN goal, from ``goal_head`` on the encoded
              observation window (``scripts/goal_modes.py``, already shipped and
              already the ``--goal-mode produced`` eval path). Needs ``states``.
``oracle``    the TRUE band minted from the held-out window's own future poses.
              ⛔ Priced and REFUSED — see :func:`oracle_availability`.

Every option returns kwargs filtered to the head's actual ``cond_*`` switches, so
none of them can inject a channel the head is not conditioned on.
"""
from __future__ import annotations

import torch

__all__ = ["OPTIONS", "OPTION_MEANING", "make_goal_kwargs_fn",
           "StatesAwareSurfacePlanner", "oracle_availability",
           "band0_kwargs", "dropped_kwargs", "describe_index_zero"]

#: The candidate goal states, in the order ``VTBAND_DECISION.md`` prices them.
#: ``crash_today`` is the shipped behaviour and is the RED baseline;
#: ``zeros_naive`` is the patch the provenance string invites, priced as a trap.
OPTIONS = ("crash_today", "zeros_naive", "band0", "dropped", "produced", "oracle",
           # ---- channel-isolation DIAGNOSTICS (not candidates) -------------- #
           # band0 gets BOTH categoricals wrong. These two split the penalty so
           # the PI can see whether a partial fix would be worth anything.
           "band0_vt_only", "band0_route_only")

OPTION_MEANING = {
    "crash_today": (
        "kw = {} — no goal channels at all. The head REFUSES (ValueError). This "
        "is the shipped default and the RED baseline: the gate cannot run."),
    "zeros_naive": (
        "band0's categoricals AND vt_speed = 0 — literally 'the zeros the "
        "provenance promises'. ⛔ vt_speed = 0 makes the selector rank up the "
        "maximally decelerating candidate (v_goal = (v0-5)+), which brakes the "
        "probe and NaNs out `recovery`. Priced as a TRAP, not a candidate."),
    "band0": (
        "vt_band=0 (v_stop), route=0 (ROUTE_LEFT), route_graded=0.0 (STRAIGHT). "
        "The probe measures a planner told it is stopping and turning left, on "
        "every held-out window, with a self-contradictory graded channel. The "
        "longitudinal selection term stays ON (vt_keep=True)."),
    "dropped": (
        "vt_band=VT_DROPPED(23), route=ROUTE_DROPPED(4), route_graded=0.0. The "
        "probe measures the deployed NO-GOAL surface, on the embedding rows "
        "goal_dropout=0.5 trains on ~50 % of every batch. The longitudinal "
        "selection term is masked OFF (vt_keep=False)."),
    "produced": (
        "vt_band/route/route_graded derived by the model's own goal_head from "
        "the encoded observation window only. No future, no label. The probe "
        "measures the DEPLOYABLE surface — what the car would actually have."),
    "oracle": (
        "vt_band/route minted from the held-out window's own FUTURE poses. "
        "REFUSED: it leaks, and it is not even well-defined at a perturbed "
        "grid point. See oracle_availability()."),
    "band0_vt_only": (
        "DIAGNOSTIC, not a candidate: vt_band=0 (v_stop) with route=ROUTE_DROPPED. "
        "Isolates the VTARGET half of band0's penalty."),
    "band0_route_only": (
        "DIAGNOSTIC, not a candidate: route=0 (ROUTE_LEFT) + route_graded=0.0 "
        "with vt_band=VT_DROPPED. Isolates the ROUTE half of band0's penalty."),
}

#: The options a real gate could actually be configured with. The rest are the
#: RED baseline, a priced trap, or channel-isolation diagnostics.
CANDIDATES = ("band0", "dropped", "produced")

# Vocabulary constants, re-exported so a caller never hard-codes them.
from tanitad.models.flagship_v15 import (N_ROUTE_CLASSES,  # noqa: E402
                                         N_VTARGET_BANDS, ROUTE_DROPPED,
                                         VT_DROPPED)


def describe_index_zero() -> dict:
    """What index 0 of each categorical goal channel MEANS. Read from the
    vocabularies at call time so this can never drift from them."""
    from tanitad.lake.vocab import VTARGET_TOKENS
    return {
        "vt_band_0": VTARGET_TOKENS[0],
        "vt_band_0_is_a_real_band": True,
        "vt_band_n_real_bands": N_VTARGET_BANDS,
        "vt_band_dropped_index": VT_DROPPED,
        "route_0": "ROUTE_LEFT",
        "route_n_classes_incl_unknown": N_ROUTE_CLASSES,
        "route_dropped_index": ROUTE_DROPPED,
        "route_graded_0": "0.0 == straight (tanh(mean_curv/CURV_TURN_PER_M))",
        "_read": ("index 0 is a REAL class on both categorical channels; "
                  "neither is a null. route=0 (LEFT) beside route_graded=0.0 "
                  "(STRAIGHT) is additionally self-inconsistent."),
    }


# --------------------------------------------------------------------------- #
# the two options that fit today's goal_kwargs_fn(b, device) signature          #
# --------------------------------------------------------------------------- #
def _filter(cfg, kw: dict) -> dict:
    """Keep only channels the head is conditioned on (mirrors ``goal_modes._filter``)."""
    out = {}
    if getattr(cfg, "cond_vtarget", False):
        out["vt_band"] = kw["vt_band"]
        out["vt_speed"] = kw["vt_speed"]
    if getattr(cfg, "cond_route", False):
        out["route"] = kw["route"]
        out["route_graded"] = kw["route_graded"]
    return out


def band0_kwargs(cfg, v0: torch.Tensor, *, naive_vt_speed: bool = False) -> dict:
    """``_goal_inputs``' zeros fallback. ⚠️ index 0 is ``v_stop`` / ``ROUTE_LEFT``.

    ``naive_vt_speed=True`` additionally zeroes ``vt_speed`` — the ``zeros_naive``
    TRAP option, which makes the selector chase ``(v0 - 5 m/s)`` instead of ``v0``.
    """
    b, device = int(v0.shape[0]), v0.device
    return _filter(cfg, {
        "vt_band": torch.zeros(b, dtype=torch.long, device=device),
        "vt_speed": torch.zeros_like(v0) if naive_vt_speed else v0,
        "route": torch.zeros(b, dtype=torch.long, device=device),
        "route_graded": torch.zeros(b, dtype=v0.dtype, device=device)})


def dropped_kwargs(cfg, v0: torch.Tensor) -> dict:
    """The learned withheld rows — ``VT_DROPPED`` / ``ROUTE_DROPPED``.

    ⚠️ Only admissible because ``goal_dropout > 0`` trains those rows. With
    ``goal_dropout == 0`` they sit at their ``N(0, 0.02)`` init and feeding them
    injects untrained noise; :func:`make_goal_kwargs_fn` REFUSES that case rather
    than silently producing a number, which is the same reasoning
    ``goal_modes.neutral_goal`` documents.

    ⚠️ ``vt_speed`` is still ``v0``, matching ``_goal_inputs`` — but with
    ``vt_band = VT_DROPPED`` the head sets ``vt_keep = False`` and
    :meth:`select` multiplies the whole longitudinal term by it, so the term is
    masked off regardless. That masking is the POINT: a goal withheld from the
    decoder must not sneak back in through the ranking (``select``'s docstring).
    """
    b, device = int(v0.shape[0]), v0.device
    return _filter(cfg, {
        "vt_band": torch.full((b,), VT_DROPPED, dtype=torch.long, device=device),
        "vt_speed": v0,
        "route": torch.full((b,), ROUTE_DROPPED, dtype=torch.long, device=device),
        "route_graded": torch.zeros(b, dtype=v0.dtype, device=device)})


def make_goal_kwargs_fn(option: str, cfg, *, goal_head=None):
    """-> ``goal_kwargs_fn(states, v0) -> dict``, the protocol every option fits.

    ⛔ Deliberately NOT ``(b, device)``: see the module docstring — that signature
    cannot carry ``v0``, so it cannot express even ``band0`` faithfully.
    ⛔ Raises for ``oracle`` rather than silently substituting something cheaper.
    """
    if option not in OPTIONS:
        raise ValueError(f"option must be one of {OPTIONS}, got {option!r}")
    if option == "crash_today":
        return None                      # the shipped default: no fn at all
    if option == "zeros_naive":
        return lambda states, v0: band0_kwargs(cfg, v0, naive_vt_speed=True)
    if option == "band0":
        return lambda states, v0: band0_kwargs(cfg, v0)
    if option == "dropped":
        gd = float(getattr(cfg, "goal_dropout", 0.0) or 0.0)
        if gd <= 0:
            raise ValueError(
                "option 'dropped' needs goal_dropout > 0 on the TRAINED config: "
                "with goal_dropout == 0 the VT_DROPPED / ROUTE_DROPPED embedding "
                "rows were never trained and sit at their N(0, 0.02) init, so "
                "probing them measures initialisation noise, not the withheld "
                "surface. (v4_config() ships goal_dropout = 0.5, so this refusal "
                "should never fire on a real v5 run — it fires on a config that "
                "silently turned the dropout off.)")
        return lambda states, v0: dropped_kwargs(cfg, v0)
    if option in ("band0_vt_only", "band0_route_only"):
        # channel-isolation diagnostics: one categorical at index 0, the other at
        # its learned DROPPED row. vt_speed stays v0 (faithful to _goal_inputs).
        vt_zero = option == "band0_vt_only"

        def _iso(states, v0):
            b, dev = int(v0.shape[0]), v0.device
            kw = {
                "vt_band": (torch.zeros(b, dtype=torch.long, device=dev) if vt_zero
                            else torch.full((b,), VT_DROPPED, dtype=torch.long,
                                            device=dev)),
                "vt_speed": v0,
                "route": (torch.full((b,), ROUTE_DROPPED, dtype=torch.long,
                                     device=dev) if vt_zero
                          else torch.zeros(b, dtype=torch.long, device=dev)),
                "route_graded": torch.zeros(b, dtype=v0.dtype, device=dev)}
            return _filter(cfg, kw)
        return _iso
    if option == "produced":
        if goal_head is None:
            raise ValueError(
                "option 'produced' needs the checkpoint's goal_head — there is "
                "no model-side producer for route / route_graded / vt_band "
                "without it (the same refusal goal_modes.resolve_goal raises).")

        def _produced(states, v0):
            import sys
            from pathlib import Path
            s = str(Path(__file__).resolve().parents[2] / "scripts")
            if s not in sys.path:
                sys.path.insert(0, s)
            import goal_modes as GM
            sc = GM.produce_goal_scalars(goal_head, states)
            return GM._filter(cfg, GM.scalars_to_goal(sc, v0))
        return _produced
    raise NotImplementedError(
        "option 'oracle' needs per-window FUTURE labels — see "
        "oracle_availability(); it is refused, not unimplemented.")


# --------------------------------------------------------------------------- #
# the option that does NOT fit today's signature                               #
# --------------------------------------------------------------------------- #
class StatesAwareSurfacePlanner:
    """``DeployableSurfacePlanner`` under the ``(states, v0)`` protocol.

    ⭐ **The point of this class is to make the signature gap visible and priced.**
    ``goal_kwargs_fn(batch_size, device)`` receives neither ``v0`` nor ``states``,
    so it cannot express ``band0`` faithfully (``vt_speed = v0``) and cannot
    express ``produced`` at all (the goal is a function of
    ``world.encode_window(frames)``, which ``traj`` computes *after* it has
    already built ``kw``). Wiring ANY option into the shipped gate is therefore
    this reordering, not a default change:

    .. code-block:: python

        states = self.world.encode_window(frames.to(self.device))
        kw = (self.goal_kwargs_fn(states, v0d)         # <- states/v0, not (b, device)
              if self.goal_kwargs_fn is not None else {})
        out = self.head(states, v0d, **kw)

    Written as a separate class rather than a patch so ``heldout_gate.py`` stays
    byte-identical while every option is measured through it. ⚠️ It is otherwise
    a faithful copy of ``DeployableSurfacePlanner.traj`` — same ``eval()``
    discipline, same autocast, same ``wp_seq`` read, same train-mode restore — so
    an option difference measured here is an option difference, not a harness one.
    ``test_vtband_options.py`` pins that equivalence against a stub head.
    """

    def __init__(self, world, head, *, device="cpu", amp=False,
                 goal_kwargs_fn=None, option="?", expect_dense=True):
        from tanitad.train.heldout_gate import DeployableSurfacePlanner
        self._base = DeployableSurfacePlanner(
            world, head, device=device, amp=amp, expect_dense=expect_dense)
        self.world, self.head = world, head
        self.device, self.amp = device, bool(amp)
        self.goal_kwargs_fn = goal_kwargs_fn
        self.horizons = self._base.horizons
        self.provenance = {
            **self._base.provenance,
            "goal_conditioning": f"vtband-option={option}: "
                                 f"{OPTION_MEANING.get(option, option)}",
            "vtband_option": option,
            "protocol": "goal_kwargs_fn(states, v0)",
        }

    @torch.no_grad()
    def traj(self, frames, v0, goal=None):
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
# the third option the brief asks about, priced                                #
# --------------------------------------------------------------------------- #
def oracle_availability() -> dict:
    """Can the gate supply the TRUE band from the held-out window's own labels?

    Priced rather than dismissed. Three findings, the third of which is fatal on
    its own even if the first two were solved.
    """
    return {
        "mechanically_reachable": True,
        "how": ("pseudosim.pseudo_evaluate already takes goals=<obj> and calls "
                "goals.get(ep_i, last, device); DeployableSurfacePlanner.traj "
                "merges that dict into kw. So a per-window oracle needs NO "
                "signature change — unlike 'produced'. HeldoutGate.probe simply "
                "never passes goals=."),
        "blocker_1_labels_not_in_scope": (
            "the gate is handed val EPISODE objects (train_flagship_v4: "
            "hg_eps = val_eps[:n]), not FlagshipV4Dataset rows. vt_band/route "
            "are minted per WINDOW by v4_labels from full-episode poses, so they "
            "are re-derivable — but that is a new minting pass inside the "
            "training loop, on every probe, at the gate's own stride."),
        "blocker_2_leaks": (
            "route/route_graded/vt_band are minted from the ego's own FUTURE "
            "poses (route <= 25 s forward; vt_band = the 85th percentile of "
            "future speed over 10-20 s). An early-stop that stops the run on a "
            "signal computed from the future is not measuring the deployable "
            "surface, and GATE_PROTOCOL 0.8 already demotes oracle-fed MODE-B "
            "numbers for exactly this reason."),
        "blocker_3_FATAL_undefined_at_the_probe_points": (
            "⛔ The gate does not probe the logged state. It probes the ego "
            "ROTATED by dyaw in (-8, 0, +8) deg (heldout_gate.probe_grid). The "
            "oracle label is minted from the UNPERTURBED future poses, so at "
            "the two perturbed points — 2 of the 3 grid points, and the only "
            "ones on which `recovery` is even defined — the 'true' goal is the "
            "goal of a state the planner is not in. Feeding it makes the probe "
            "measure recovery-toward-a-goal-for-a-different-pose. There is no "
            "true label for a synthesised state, so this is not a data gap that "
            "could be closed by plumbing."),
        "verdict": ("REFUSED. Available in principle, leaking in practice, and "
                    "ill-defined at 2 of the gate's 3 grid points."),
    }
