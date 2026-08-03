"""REF-C's SELECTION SURFACE — the primitives D-SEL adds to ``refc.py``.

WHY A SELECTION MODULE, AND WHY NOW
============================================================================
REF-C beats flagship v1 open-loop and closed-loop with the separation ENTIRELY
LATERAL, and every one of its MEASURED defects sits on the SELECTION surface —
not on the proposal surface. Read together they are one defect, not five:

  1. **The refined fan is ranked by the UNREFINED score.**
     ``AnchoredDiffusionDecoder.forward`` runs the denoise loop as
     ``_, off = self._decode(...)`` — the refined confidence is DISCARDED — and
     then ``argmax`` es the t=0 classifier score over the REFINED trajectories.
     Scoring and refinement are decoupled. (Source: ``refc.py``, the ``for i in
     range(steps)`` loop. Restated by ``flagship_v15.V15Decoder``, which repairs
     exactly this and calls it "THE SCORING FIX".)
  2. **The consequence is a RANKING failure, not a coverage failure.** MEASURED
     on REF-C's own fan, corpus-wide (881 canonical val windows / 40 episodes):
     oracle-in-fan **0.1640 m** against a selected score an order above it, with
     the pick more than 2x worse than the fan's best in **45.4 %** of windows.
     The 0.16 m plan was already IN the fan. (Earlier single-clip figures
     0.295 / 65 % were restated 2026-07-20 and must not be quoted.)
  3. **72.08 % of the emitted fan is not physically flyable.** The anchor
     vocabulary is blameless — ``furthest_point_sample`` returns ``pool[chosen]``
     — but the UNBOUNDED offset head refines anchors into candidates implying up
     to 171.5 km/h against a val GT max of 132.4. MEASURED on
     ``fan_refc-xl-30k.pt``: the bounded-acceleration band removes 72.08 % of
     candidates, the ADE-oracle survives 100 %, no window is left empty, and the
     paired episode-cluster delta is exactly **0.0000**.
  4. **The grafts that reach the score are UNCLAMPED.** ``maneuver_to_anchor`` /
     ``lat_to_anchor`` / ``lon_to_anchor`` / ``lan_gate`` are all added straight
     to ``conf``. ``scripts/refc_train.py`` already LOGS ``graft_lat_norm`` /
     ``graft_lon_norm`` / ``conf_norm`` — the instrument exists and there is no
     actuator. A graft that swamps the base score is not a prior, it is a second
     selector (flagship v4 §6.2 discipline 4; that failure mode fired at 2.80x).
  5. **The consequences of candidates never reach the ranking.** This is
     ``cond_imagination``, which ``flagship_v15`` calls "THE NOVEL PART" and
     which was hard-wired OFF in the flagship for months.

⇒ D-SEL rebuilds the ranking, and ONLY the ranking. Every lever here is
zero-init or param-free, so a disabled flag gives a byte-identical state_dict.

WHAT DOES **NOT** TRANSFER FROM THE FLAGSHIP, AND WHY (argued from source)
============================================================================
* ``imagine_probes`` — the flagship's shared probe-vocabulary rollout — does NOT
  transfer, for TWO independent reasons, either of which is sufficient:
    (a) **It has no candidate axis.** ``flagship_v15.
        IMAGINATION_HAS_CANDIDATE_AXIS = False``, MEASURED 2026-07-27: 32 tokens
        invariant to ``n_anchors``, IDENTICAL for all 256 candidates. It can
        CONDITION a decode; it can never RANK one. REF-C's measured defect is
        ranking.
    (b) **REF-C has no rollable predictor.** The flagship rolls a frozen 20-step
        latent predictor over ACTIONS. REF-C's world model is ``law_head``:
        ``[pooled, traj] -> pooled_{t+5}`` (``refc_train.LAW_AHEAD = 5``). It
        consumes a TRAJECTORY and emits a POOLED vector, so it cannot be
        iterated — there is no ``fmap`` to re-decode from. A probe roll is
        structurally unavailable.
  The form that DOES transfer is ``imagine_candidates``: ONE consequence per
  candidate. REF-C can afford exactly that, because ``law_head`` is a single
  MLP evaluation per candidate — see :func:`consequence_scores`.
* ``vision_rank`` (the rank-16 projection) does NOT transfer. The swamping
  dose-response was MEASURED on a FLAT 2048-d readout state entering a flat
  reader (``flagship_v4``'s factorised heads read ``states[:, -1]`` directly).
  REF-C's decoder cross-attends 64 SPATIAL tokens and its tactical head reads a
  mean-pooled vector — a different object. Porting the projection would be an
  unmeasured capacity CUT, which is the same evidence-transfer error
  ``V4Config.sel_reach_clamp`` refuses to make in the other direction.
* ``lambda_plan`` (the grad-scale trunk seam) does NOT transfer. It exists
  because the flagship warm-starts a trunk and must scale the planner's gradient
  into it. REF-C trains end-to-end from scratch under ONE optimizer; there is no
  trunk/planner boundary to scale.
* Prior-corrected decoding at ``tau > 0`` is NOT re-proposed. On the 1232
  windows the 5-way label can represent, the patch is NOT separated from doing
  nothing (macro-F1 +0.0107 [-0.0418, +0.0665]) and it costs precision
  (0.2340 -> 0.1711). It stays default-off where D-TAC1B left it.

ONE-IMPLEMENTATION RULE
============================================================================
:func:`reachability_mask` and :func:`assert_candidate_axis` are RE-EXPORTS of
``tanitad.models.flagship_v15``, not copies. ``refc.py`` already retired a
self-contained copy of ``advect`` for exactly this reason ("two implementations
of the same warp is how geometries drift apart"), and the 72.08 % measurement is
tied to that specific function. The import is deferred to CALL TIME because
``flagship_v15`` imports ``refc``, and ``refc`` imports this module — a
module-level import would close the cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["assert_candidate_axis", "reachability_mask", "SeamState",
           "apply_seam_clamp", "consequence_scores", "NoCandidateAxis"]


# ============================================================================
# Re-exports (deferred: flagship_v15 imports refc, refc imports this module)
# ============================================================================

def _v15():
    from tanitad.models import flagship_v15
    return flagship_v15


def __getattr__(name: str):
    """PEP 562 module ``__getattr__`` — ``refc_select.NoCandidateAxis`` IS
    ``flagship_v15.NoCandidateAxis``, resolved at first access.

    It must be the SAME class object, not a look-alike: a caller writing
    ``except refc_select.NoCandidateAxis`` has to catch what the guard actually
    raises. A module-level alias would need an import at import time and would
    close the ``flagship_v15 -> refc -> refc_select`` cycle.
    """
    if name == "NoCandidateAxis":
        return _v15().NoCandidateAxis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def assert_candidate_axis(x: Tensor, n_candidates: int, *, name: str,
                          axis: int = 1) -> Tensor:
    """Refuse a would-be per-candidate tensor that cannot discriminate candidates.

    THE re-export of ``flagship_v15.assert_candidate_axis`` — one implementation,
    deliberately. It fails in BOTH degenerate ways: no candidate axis at all, and
    an axis that is CONSTANT along its length (which carries no candidate
    information, so any ranking built on it ranks nothing).

    In REF-C this is the runtime guard on :func:`consequence_scores`. The
    flagship built this guard AFTER measuring that ``imagine_probes`` had been
    returning a candidate-invariant tensor all along and had made E-V5-1's
    imagination negative over-determined; REF-C gets it BEFORE the run.
    """
    return _v15().assert_candidate_axis(x, n_candidates, name=name, axis=axis)


def reachability_mask(fan: Tensor, v0: Tensor, *, accel_max: float = 2.5,
                      horizon_s: float = 2.0) -> Tensor:
    """[B, N] bool — which candidates a bounded-acceleration ego could fly.

    THE re-export of ``flagship_v15.reachability_mask``. The 72.08 % / oracle-
    survives-100 % / paired-delta-0.0000 measurement was made with this exact
    function on REF-C-XL's own emitted fan, so a re-implementation here would
    silently detach the number from the code it describes.
    """
    return _v15().reachability_mask(fan, v0, accel_max=accel_max,
                                    horizon_s=horizon_s)


# ============================================================================
# Seam norm clamp — the actuator for a telemetry REF-C already emits
# ============================================================================

@dataclass
class SeamState:
    """Sustained-saturation counter for one graft surface.

    A PLAIN dataclass held as a python attribute, never a buffer: it must not
    enter ``state_dict`` and change checkpoint compatibility (the same choice
    ``flagship_v4`` makes for ``_seam_sat_steps``).
    """
    sat_steps: int = 0


def apply_seam_clamp(base: Tensor, graft: Tensor, *, clamp: float,
                     fail: float, fail_frac: float, patience: int,
                     state: SeamState, surface: str) -> tuple[Tensor, dict]:
    """``base + clamped(graft)`` with per-sample norm control and telemetry.

    ``base`` [B, N] is the score the selector would rank without any prior;
    ``graft`` [B, N] is the SUM of every prior added to that surface. The total
    graft is rescaled IN-GRAPH so its per-sample norm never exceeds ``clamp`` x
    the base-score norm.

    ``clamp <= 0`` disables the clamp entirely and returns ``base + graft``, so
    the default path is byte-identical to REF-C today. **Below the clamp the
    rescale factor is exactly 1.0** (``clamp / ratio.clamp_min(clamp)`` is
    ``clamp / clamp``), and ``x * 1.0`` is exact in IEEE-754 — so turning the
    clamp ON does not perturb a well-behaved run either. That bit-identity is
    the whole reason the clamp is admissible as a default-on lever later.

    ⭐ THE FAIL-LOUD IS A POPULATION-OVER-TIME CONDITION, NOT A BATCH MAX. The
    flagship's first version fired on ``ratio.max()`` and one sample of 64 could
    kill a run: MEASURED, it lost BOTH wide arms of a geometry validation at
    ~step 350 on arms that were training at or below their control on every loss
    term (C51). So all three of ``mean ratio > fail``, ``fail_frac`` of the batch
    at the clamp, and ``patience`` CONSECUTIVE steps must hold together, and the
    counter resets the moment any of them breaks.

    ⚠️ When it fires, the clamp is HOLDING — it cannot fail by construction. The
    condition being reported is that above the clamp the graft's strength is a
    no-op, so the prior-strength axis is unreadable in that regime.
    """
    if clamp <= 0.0:
        state.sat_steps = 0
        return base + graft, {}
    b_norm = base.norm(dim=-1).clamp_min(1e-9)                    # [B]
    ratio = graft.norm(dim=-1) / b_norm                           # [B]
    pre_mean = float(ratio.mean().detach())
    pre_max = float(ratio.max().detach())
    bound_frac = float((ratio > clamp).to(ratio.dtype).mean().detach())
    saturated = pre_mean > fail and bound_frac > fail_frac
    state.sat_steps = (state.sat_steps + 1) if saturated else 0
    if patience > 0 and state.sat_steps >= patience:
        state.sat_steps = 0                     # a caught error stays recoverable
        raise RuntimeError(
            f"REF-C {surface}-surface graft seam SATURATED: mean pre-clamp "
            f"ratio {pre_mean:.3f} > {fail} AND {bound_frac:.1%} of the batch "
            f"at/above the clamp (> {fail_frac:.0%}), sustained "
            f"{patience} consecutive steps. ⚠️ THE IN-GRAPH CLAMP IS HOLDING — "
            f"it cannot fail by construction — so this is NOT a code fault. It "
            f"is a TRAINING-DYNAMICS condition: above the clamp the graft's "
            f"strength is a NO-OP, so the prior-strength axis is unreadable "
            f"here and any sweep read in this regime shows SATURATION, not a "
            f"finding. Inspect seam_* in train_log.jsonl before changing "
            f"seam_clamp / seam_fail.")
    scale = clamp / ratio.clamp_min(clamp)                        # <= 1, ==1 below
    return base + graft * scale[:, None], {
        f"seam_{surface}_ratio_preclamp_mean": round(pre_mean, 4),
        f"seam_{surface}_ratio_preclamp_max": round(pre_max, 4),
        f"seam_{surface}_ratio_max": round(min(pre_max, clamp), 4),
        f"seam_{surface}_bound_frac": round(bound_frac, 4),
        f"seam_{surface}_sat_steps": state.sat_steps,
    }


# ============================================================================
# Candidate-conditioned consequence scoring — the transferable cond_imagination
# ============================================================================

def consequence_scores(fan: Tensor, ctx: Tensor, cons_head, feat_proj,
                       conf_head, *, detach: bool = True,
                       guard: bool = True) -> Tensor:
    """Score every candidate by the CONSEQUENCE of flying it. [B, N].

    ``fan`` [B, N, S, 2] is the refined candidate set, ``ctx`` [B, F] the pooled
    image latent, ``cons_head`` REF-C's ``law_head`` — a trajectory-conditioned
    latent world model ``[pooled, traj] -> pooled_{t+0.5s}``, already trained by
    the LAW MSE against a ``no_grad``-encoded future frame. Evaluating it once
    per candidate answers, per candidate, "what would the world look like if I
    flew this?" — which is what ``cond_imagination`` is for, in the ONE form
    REF-C's world model can express (see the module docstring).

    ⭐ CAPACITY CONTROL: this function introduces **ZERO new parameters**. The
    consequence latent lives in the same ``feat_dim`` space as the conv-map
    tokens, so it is projected by the decoder's OWN ``feat_proj`` and scored by
    the decoder's OWN ``conf_head`` — i.e. by the same notion of "is this a good
    plan" the decoder already learned. The scale mismatch between a raw
    projection and a post-attention query is removed by a PARAMETER-FREE
    ``layer_norm``. The only parameter D-SEL adds for this lever is the single
    zero-init ``cons_gate`` scalar on the decoder. A dedicated projection would
    have cost ~270 k on REF-C-base — the class of mistake that once cost
    +272,001 params where +897 sufficed.

    ⚠️ ``detach`` (default True) runs ``cons_head`` under ``no_grad``, so the
    ranking objective cannot corrupt the world model: LAW stays trained by its
    own MSE alone. This mirrors the flagship's FROZEN-predictor discipline —
    ``_imagination_inputs`` rolls under ``no_grad`` precisely so the head's
    gradient never becomes a 20-step backprop into the predictor. ``feat_proj``
    and ``conf_head`` still receive gradient through this path, which is a real
    coupling and is why the gate starts at zero and is ablatable to zero.

    ``guard`` runs :func:`assert_candidate_axis` on the result. A CONSTANT score
    along the candidate axis means the fan has collapsed or the consequence is
    candidate-blind — either way the ranking would be ranking nothing, which is
    the exact silent failure ``imagine_probes`` shipped with for months.
    """
    b, n = fan.shape[:2]
    inp = torch.cat([ctx[:, None].expand(b, n, ctx.shape[-1]),
                     fan.reshape(b, n, -1)], dim=-1)
    if detach:
        with torch.no_grad():
            cons = cons_head(inp)                                 # [B, N, F]
    else:
        cons = cons_head(inp)
    q = feat_proj(cons)                                           # [B, N, d]
    q = F.layer_norm(q, (q.shape[-1],))                           # param-free
    s = conf_head(q).squeeze(-1)                                  # [B, N]
    if guard:
        assert_candidate_axis(s, n, name="refc consequence score")
    return s
