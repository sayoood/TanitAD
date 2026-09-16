"""refcv6 SELECTION seams: nav compliance, the speed ceiling, the behaviour gate.

Three things reach the ranked score in refcv6, and each is separately ablatable:

  1. :func:`nav_compliance_prior` — the **parameter-free** nav-compliance term.
     It already existed (``refc_selector_targets.compliance_target``) and was
     **never on the forward path** — MEASURED, ``REFCV6_CLARIFICATION.md`` §3.1:
     *"the rule-based nav-compliance scorer exists but is not on the forward
     path (refc_selector_targets.py:406-450)"*. This module is the adapter that
     puts it there, behind ONE zero-init gate.
  2. :class:`SpeedCeilingFilter` — the max-speed ceiling, as an **ARGMAX FILTER**
     rather than a score term.
  3. :class:`BehaviourSelectionGate` — the valid-behaviour set, as a zero-init
     ``17 -> n_anchors`` graft with the five situation tokens **structurally
     dead**.

⛔⛔ WHY THE SPEED CEILING IS A FILTER AND NOT A SCORE TERM. The acceptance bar
is *"the planned maximum must stay <= the limit on >= 99 %"*. A soft penalty
behind a LEARNED gate cannot make that promise — the gate is free to stay at
zero, which is the whole point of a zero-init gate. So the ceiling takes the
shape ``refc.py`` already uses for a hard constraint, S2's reachability band:
**it filters the argmax only**; ``score`` is returned unmasked so no ``-inf``
can reach a cross-entropy, and a row whose survivor set is empty keeps its whole
fan (an everywhere-unreachable window is a measurement failure, not a licence to
emit nothing). ⚠️ The empty-row fallback is exactly why obedience is 99 %, not
100 %, and why the test reports the empty-row count beside the rate.

⛔⛔ ADMISSIBILITY, AND IT BINDS THIS FILE HARDEST. PI 2026-08-03, still
binding: ``tac_SIT``, the traffic-light tokens and ``YIELD`` may be auxiliary
TARGETS only — **never inputs to the goal or to selection**. The behaviour set
this module feeds into selection contains five such tokens, so
:class:`BehaviourSelectionGate` carries a FROZEN admissibility mask that zeroes
their columns, and :func:`assert_situation_columns_dead` proves they are dead by
**MUTATION** — it writes large values into the refused columns, re-runs the
forward, and refuses if anything moved. An inspection of the mask would pass on
a build whose forward stopped applying it.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from tanitad.models.vocab_v7 import TACTICAL_GOAL_TOKENS_V7
from tanitad.refs.refc_selector_targets import compliance_target
from tanitad.refs.refcv6_tactical import (N_GOAL_TOKENS,
                                          SITUATION_OUTPUT_TOKENS,
                                          SituationInputRefused)

__all__ = [
    "SELECTION_ADMISSIBLE_TOKENS", "SELECTION_REFUSED_TOKENS",
    "admissibility_mask", "nav_compliance_prior", "SpeedCeilingFilter",
    "BehaviourSelectionGate", "assert_situation_columns_dead",
    "planned_max_speed", "step_durations_s",
]

#: The goal tokens that MAY reach selection: the 22 minus the situation
#: outputs. MEASURED here: 22 - 5 = 17 admissible
#: (``YIELD``, ``TRAFFIC_LIGHT_REACT``, ``..._RED``, ``..._YELLOW``,
#: ``..._GREEN`` are refused).
SELECTION_REFUSED_TOKENS: tuple[str, ...] = tuple(
    t for t in TACTICAL_GOAL_TOKENS_V7 if t in SITUATION_OUTPUT_TOKENS)
SELECTION_ADMISSIBLE_TOKENS: tuple[str, ...] = tuple(
    t for t in TACTICAL_GOAL_TOKENS_V7 if t not in SITUATION_OUTPUT_TOKENS)


def admissibility_mask(dtype: torch.dtype = torch.float32) -> Tensor:
    """``[22]`` in {0, 1}: 0 on every token the PI's ruling refuses in selection."""
    return torch.tensor(
        [0.0 if t in SITUATION_OUTPUT_TOKENS else 1.0
         for t in TACTICAL_GOAL_TOKENS_V7], dtype=dtype)


# ---------------------------------------------------------------------------
# 1. nav compliance — parameter-free, behind ONE zero-init gate
# ---------------------------------------------------------------------------

def nav_compliance_prior(cand: Tensor, nav_cmd: Tensor, *, tau_rad: float,
                         stall_m: float = 0.05) -> tuple[Tensor, Tensor]:
    """``(prior [B, N] in {0,1}, informative [B, N] bool)``.

    A thin adapter over ``refc_selector_targets.compliance_target``, which is
    the torch mirror of ``taniteval.nav_compliance``: 1.0 where the candidate's
    terminal heading has the COMMANDED sign and a magnitude of at least
    ``tau_rad``.

    ⛔ ``follow`` and ``straight`` carry no commanded side, so the predicate is
    **undefined**, not "failed" — ``compliance_target`` returns its own mask and
    this function returns the masked product, i.e. the term is IDENTICALLY ZERO
    on uninformative windows. That matters: nav is constant (``follow``) on
    ~75-79 % of windows (the C6 confound), and a term that read "0 = does not
    comply" there would push the ranked score around on three quarters of the
    corpus for no reason.

    ⚠️ ``tau_rad`` HAS NO DEFAULT, here or upstream. It is derived per corpus by
    ``nav_compliance.derive_tolerance(pos, neg)`` and the arm must state the
    value it used; an invented tolerance is a number with no evidence class
    deciding what "compliant" means.

    ⭐ PARAMETER-FREE ON PURPOSE. A learned ``nav -> n_anchors`` matrix could
    become a route-shaped shortcut; a geometric predicate cannot. This is the
    same argument ``refc.py`` makes for ``lan_gate`` and for the S6 goal terms,
    and it is why this term takes one scalar gate and nothing else.
    """
    ok, mask = compliance_target(cand, nav_cmd, tau_rad, stall_m=stall_m)
    return ok, mask.to(torch.bool)


# ---------------------------------------------------------------------------
# 2. the speed ceiling — an ARGMAX FILTER
# ---------------------------------------------------------------------------

def step_durations_s(horizons: Sequence[int], tick_s: float = 0.1
                     ) -> tuple[float, ...]:
    """Seconds spanned by each waypoint interval, DERIVED from the horizons.

    ⛔⛔ THE DT IS DERIVED, NEVER ASSUMED, AND THIS IS THE WHOLE POINT OF THIS
    FUNCTION. ``RefCConfig.trajectory.horizons`` is ``(5, 10, 15, 20)`` TICKS at
    ``anchor_dt = 0.1`` s — so the waypoint spacing is **0.5 s, not 0.1 s**.
    Dividing by 0.1 would report every planned speed 5x too high and the
    obedience test would "fail" a perfectly obedient model. ``refc.py`` makes
    exactly this derivation for its own prefix projection and says why: *"The
    prefix dt is DERIVED from anchor_slots, never assumed."* One rule, derived
    twice, is one rule; a typed 0.1 here would be a second, wrong one.

    The origin is tick 0 (the ego at t0), so interval ``i`` spans
    ``(h_i - h_{i-1}) * tick_s`` with ``h_{-1} = 0``.
    """
    h = [int(x) for x in horizons]
    if not h:
        raise ValueError("horizons must be non-empty (in ticks)")
    if any(b <= a for a, b in zip(h, h[1:])) or h[0] <= 0:
        raise ValueError(
            f"[refcv6-sel] horizons must be strictly increasing positive "
            f"ticks, got {h}. A non-monotone grid gives a negative duration "
            f"and a speed with the wrong sign.")
    prev = 0
    out = []
    for k in h:
        out.append((k - prev) * float(tick_s))
        prev = k
    return tuple(out)


def planned_max_speed(cand: Tensor, *, horizons: Sequence[int],
                      tick_s: float = 0.1,
                      v0: Tensor | None = None) -> Tensor:
    """``[B, N]`` — the maximum speed each candidate trajectory reaches, m/s.

    ``cand`` is ``[B, N, S, 2]`` in METRES, ego frame; ``horizons`` are its
    waypoint TICKS (``RefCConfig.trajectory.horizons`` /
    ``AnchoredDiffusionDecoder.anchor_horizons``). Speed on interval ``i`` is
    ``|p_i - p_{i-1}| / dt_i`` with ``dt_i`` from :func:`step_durations_s`; the
    first interval is measured from the origin, so no waypoint is dropped.

    ⚠️ ``v0`` is accepted and IGNORED for the maximum itself — it is the speed
    the ego already has, not one the plan chose, and including it would make
    every plan "violate" a ceiling the ego was already above. It is in the
    signature because the obedience test reports it beside the verdict.
    """
    if cand.dim() != 4 or cand.shape[-1] != 2:
        raise ValueError(f"cand must be [B, N, S, 2], got {tuple(cand.shape)}")
    dts = step_durations_s(horizons, tick_s)
    if len(dts) != cand.shape[2]:
        raise ValueError(
            f"[refcv6-sel] {len(dts)} horizons but the fan has "
            f"{cand.shape[2]} waypoints. A mismatched grid silently pairs "
            f"waypoint i with interval j and reports a speed that belongs to "
            f"neither.")
    zero = torch.zeros_like(cand[:, :, :1, :])
    seq = torch.cat([zero, cand], dim=2)                     # [B, N, S+1, 2]
    step = (seq[:, :, 1:, :] - seq[:, :, :-1, :]).norm(dim=-1)   # [B, N, S]
    dt = torch.tensor(dts, device=cand.device, dtype=step.dtype)
    return (step / dt.view(1, 1, -1)).max(dim=-1).values


class SpeedCeilingFilter(nn.Module):
    """The fed set-speed, as a survivor mask over the fan. **No parameters.**

    ``forward(cand, v_limit_ms) -> (keep [B, N] bool, telemetry)``.

    ⛔ IT FILTERS THE ARGMAX ONLY (the S2 pattern). The caller must apply it as
    ``rank = score.masked_fill(~keep, -inf)`` and leave ``score`` untouched, so
    no ``-inf`` reaches a cross-entropy.

    ⛔ A ROW WITH NO SURVIVOR KEEPS ITS WHOLE FAN, and the count is reported. A
    window where every candidate exceeds the ceiling is a fact about the
    vocabulary (the 117 anchors are rolled from ``v0``; at a high ``v0`` and a
    30 km/h ceiling there may be nothing compliant), not a licence to emit
    nothing. ⚠️ This is the ONLY reason the obedience bar is 99 % and not 100 %,
    so ``n_rows_empty`` is part of the verdict, never a footnote.
    """

    def __init__(self, *, horizons: Sequence[int], tick_s: float = 0.1,
                 tol_ms: float = 0.0):
        super().__init__()
        self.horizons = tuple(int(h) for h in horizons)
        self.tick_s = float(tick_s)
        self.tol_ms = float(tol_ms)
        # Derived once, at construction, so a malformed grid fails at BUILD
        # time rather than on the first forward of a GPU run.
        self.step_dt_s = step_durations_s(self.horizons, self.tick_s)

    @property
    def n_params(self) -> int:
        return 0

    def forward(self, cand: Tensor, v_limit_ms: Tensor
                ) -> tuple[Tensor, dict]:
        vmax = planned_max_speed(cand, horizons=self.horizons,
                                 tick_s=self.tick_s)             # [B, N]
        lim = v_limit_ms.reshape(-1, 1).to(vmax.dtype) + self.tol_ms
        keep = vmax <= lim
        empty = ~keep.any(dim=1)
        keep = keep | empty[:, None]
        tele = {
            "speed_frac_candidates_clipped": round(
                float(1.0 - keep.to(torch.float32).mean().detach()), 4),
            "speed_rows_empty": int(empty.sum()),
            "speed_frac_rows_empty": round(
                float(empty.to(torch.float32).mean().detach()), 4),
            "_reads": ("a row with NO compliant candidate keeps its whole fan; "
                       "those rows are where the obedience rate can fall below "
                       "1.0 and they are counted, not hidden"),
        }
        return keep, tele


# ---------------------------------------------------------------------------
# 3. the valid-behaviour set -> selection
# ---------------------------------------------------------------------------

class BehaviourSelectionGate(nn.Module):
    """``[B, 22] behaviour probabilities -> [B, N] additive log-space term``.

    A bias-free ``Linear(22, n_anchors)``, **zero-init**, so the ranked score is
    bit-identical to today's at step 0 and every later change is attributable —
    the ``ctx_to_cond`` / ``lon_to_anchor`` discipline.

    ⛔⛔ FIVE COLUMNS ARE STRUCTURALLY DEAD. ``YIELD`` and the four
    traffic-light tokens are SITUATION OUTPUTS, and the PI's 2026-08-03 ruling
    refuses them as inputs to selection. The mask is a registered BUFFER (so it
    travels with the checkpoint and a load cannot drop it) and it is applied on
    EVERY forward, not at init — an init-time zeroing is undone by the first
    optimizer step.

    ⚠️ The input is the DETACHED probability vector from
    ``refcv6_tactical.planner_feeds``. If it were not detached, the planning
    loss would reach the tactical layer and the cheapest descent direction is to
    reshape the behaviour head into whatever correlates with the trajectory —
    after which its per-class numbers measure the planner's convenience rather
    than what the labels taught. (⛔ NOT a nav argument: the PI ruled on
    2026-09-16 that deriving the turn command from nav is by design.)
    """

    def __init__(self, n_anchors: int, n_tokens: int = N_GOAL_TOKENS):
        super().__init__()
        if n_tokens != N_GOAL_TOKENS:
            raise ValueError(
                f"[refcv6-sel] n_tokens {n_tokens} != the tactical goal "
                f"vocabulary {N_GOAL_TOKENS}. Size it from the vocabulary, "
                f"never from a literal (the z_tac refusal, same reason).")
        self.proj = nn.Linear(n_tokens, int(n_anchors), bias=False)
        nn.init.zeros_(self.proj.weight)
        self.register_buffer("admissible",
                             admissibility_mask(torch.float32).view(1, -1))
        self.refused_tokens = tuple(SELECTION_REFUSED_TOKENS)

    @property
    def n_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))

    def forward(self, valid_behaviour: Tensor) -> Tensor:
        if valid_behaviour.shape[-1] != N_GOAL_TOKENS:
            raise ValueError(
                f"[refcv6-sel] expected [B, {N_GOAL_TOKENS}] behaviour "
                f"probabilities, got {tuple(valid_behaviour.shape)}")
        v = valid_behaviour.to(self.proj.weight.dtype)
        # ⛔ THE MASK IS APPLIED TO THE INPUT, EVERY FORWARD. Masking the
        # WEIGHT instead would leave the refused columns' gradient live and one
        # `optimizer.step()` away from mattering again.
        return self.proj(v * self.admissible.to(v.dtype))


@torch.no_grad()
def assert_situation_columns_dead(gate: BehaviourSelectionGate,
                                  *, n_probe: int = 8,
                                  magnitude: float = 1e3,
                                  atol: float = 1e-6) -> dict:
    """⛔ PROVE — BY MUTATION — that no refused token can move the ranked score.

    Not an inspection of ``gate.admissible``. This ACTIVELY writes
    ``magnitude`` into every refused column of a probe input, re-runs the
    forward, and REFUSES if the output moved. It then writes the same magnitude
    into an ADMISSIBLE column and refuses if the output did **not** move — a
    guard that reads "nothing moves" because the whole layer is dead proves
    nothing, and that is the failure mode this second half exists to catch.

    Returns the two measured deltas so a run record carries the numbers rather
    than the word "checked".

    ⛔⛔ IT PROBES **THIS** OBJECT, NOT A COPY OF IT. The first version of this
    function built a fresh ``BehaviourSelectionGate``, loaded ``gate``'s
    state_dict into it and probed the copy — and the mutation test
    (``test_deleting_the_mask_MAKES_THE_GUARD_FAIL``) caught it immediately: a
    gate whose ``forward`` had been replaced by an UNMASKED one still passed,
    because the copy's forward was the correct one. A guard that probes a
    reconstruction cannot see a defect in the object. The weights are therefore
    perturbed IN PLACE (the layer is zero-init, so a live layer has to be
    manufactured for the positive control to mean anything) and restored
    bitwise in a ``finally``.
    """
    w = gate.proj.weight
    dev, dt = w.device, w.dtype
    ref_idx = [i for i, t in enumerate(TACTICAL_GOAL_TOKENS_V7)
               if t in SITUATION_OUTPUT_TOKENS]
    adm_idx = [i for i, t in enumerate(TACTICAL_GOAL_TOKENS_V7)
               if t not in SITUATION_OUTPUT_TOKENS]
    if not ref_idx or not adm_idx:
        raise SituationInputRefused(
            "[refcv6-sel] ⛔ the admissibility partition is degenerate "
            f"(refused {len(ref_idx)}, admissible {len(adm_idx)}). With no "
            f"refused column there is nothing to prove dead, and the guard "
            f"would pass vacuously.")

    saved = w.detach().clone()
    try:
        w.normal_(std=1.0)
        x = torch.zeros(n_probe, N_GOAL_TOKENS, device=dev, dtype=dt)
        base = gate(x)
        xr = x.clone()
        xr[:, ref_idx] = magnitude
        d_refused = float((gate(xr) - base).abs().max())
        xa = x.clone()
        xa[:, adm_idx] = magnitude
        d_admissible = float((gate(xa) - base).abs().max())
    finally:
        # ⛔ BITWISE RESTORE. This runs inside `RefCV3Model.__init__`; a probe
        # that left the gate non-zero would silently un-do the zero-init and
        # make the arm's step-0 parity claim false.
        w.copy_(saved)

    if d_refused > atol:
        raise SituationInputRefused(
            f"[refcv6-sel] ⛔ PI 2026-08-03 VIOLATED: driving the REFUSED "
            f"tokens {[TACTICAL_GOAL_TOKENS_V7[i] for i in ref_idx]} moved the "
            f"selection term by {d_refused:.6g} (> {atol:g}). A situation "
            f"classifier's output is reaching selection. The admissibility "
            f"mask is not being applied on the forward path.")
    if d_admissible <= atol:
        raise SituationInputRefused(
            f"[refcv6-sel] ⛔ the positive control FAILED: driving the "
            f"ADMISSIBLE tokens moved the selection term by only "
            f"{d_admissible:.6g}. The refused-column check above therefore "
            f"proves nothing — a dead layer passes it. Fix the gate before "
            f"reading its verdict.")
    return {"delta_refused": d_refused, "delta_admissible": d_admissible,
            "refused_tokens": [TACTICAL_GOAL_TOKENS_V7[i] for i in ref_idx],
            "n_admissible": len(adm_idx), "atol": atol,
            "_is": "a MUTATION proof, not an inspection of the mask"}
