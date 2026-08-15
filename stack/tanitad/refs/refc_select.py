"""REF-C's SELECTION SURFACE — the primitives D-SEL adds to ``refc.py``.

WHY A SELECTION MODULE, AND WHY NOW
============================================================================
REF-C is the programme's reference model, and every one of its MEASURED defects
sits on the SELECTION surface — on WHICH candidate is emitted, not on which are
proposed. Read together they are one defect, not five. Each number below carries
its ARM and its artifact, because two of them are commonly quoted against the
wrong arm:

  1. **The refined fan is ranked by the UNREFINED score.** MEASURED, source.
     ``AnchoredDiffusionDecoder.forward`` ran the denoise loop as
     ``_, off = self._decode(...)`` — the refined confidence was DISCARDED — and
     then ``argmax`` ed the t=0 classifier score over the REFINED trajectories.
     Scoring and refinement were decoupled. (Restated by
     ``flagship_v15.V15Decoder``, which repairs exactly this and calls it "THE
     SCORING FIX". S1 is that repair, brought home.)
  2. **The consequence is a RANKING failure, not a coverage failure.** MEASURED
     on 881 canonical val windows / 40 episodes, ``taniteval/results/
     scaleab_refc-base-30k_vs_refc-xl-30k.json`` (independently duplicated in
     ``taniteval/results/planfan_clips_summary.json``):

       ==============  =========  ==========  ============  ===================
       arm             anchors    selected    oracle-in-fan  pick >2x worse
       ==============  =========  ==========  ============  ===================
       refc-XL-30k     256        0.4714      **0.1640**     **45.4 %**
       refc-base-30k   128        0.4728      0.1914         41.09 %
       refc-XL @128    128        —           0.2624         31.9 %
       ==============  =========  ==========  ============  ===================

     ⚠️ **0.1640 / 45.4 % are REF-C-XL's, not base's** — a frequent mis-citation.
     ⚠️ Earlier single-clip figures 0.295 / 65 % were restated 2026-07-20 and
     must not be quoted.
     ⛔ **AND THE GAP IS NOT AVAILABLE HEADROOM.** ``MODEL_REGISTRY.md`` §4.1
     carries a standing caveat: *the oracle gap is ~92 % irreducible*; REF-C
     v1.2's learned re-scorer, across 47 trained arms, recovered at most 8.4 %
     of it and its headline (0.46251 vs 0.47144) is **NOT separated**
     (+0.00893 [-0.0062, +0.0250]). D-SEL is therefore NOT pitched as
     "recover 0.31 m". S1 differs from v1.2 IN KIND — v1.2 re-ranked a FROZEN
     decoder post hoc, whereas S1 puts the ranking objective INSIDE training so
     the refined readout is shaped by it — but "different in kind" is an
     argument, and this programme settles those with a pre-registered
     experiment, not with an argument.
  3. **72.08 % of the emitted fan is not physically flyable — and removing it is
     EXACTLY INERT.** The anchor vocabulary is blameless (``furthest_point_
     sample`` returns ``pool[chosen]``); the UNBOUNDED offset head refines
     anchors into candidates implying up to 171.5 km/h against a val GT max of
     132.4. MEASURED on ``fan_refc-xl-30k.pt`` (REF-C-XL, 881 windows),
     ``…/incoming/2026-07-27-percandidate-labels/raw/t1_clip_fansize.json``:
     72.08 % of candidates removed · empty survivor sets **0.00 %** · oracle
     survives **100 %** · ``as_trained_ade`` 0.4714 == ``clipped_ade`` 0.4714,
     paired delta **exactly 0.0**.
     ⇒ **S2 is a PRECONDITION, not a win.** Nobody should expect it to move ADE;
     it is here because it deletes 72.08 % of the search space at zero cost and
     therefore makes any PER-CANDIDATE computation **3.58x cheaper** — which is
     what lets S3 exist at all. Pitching it as an improvement would be the
     mistake, not running it.
  4. **The grafts that reach the score are UNCLAMPED.** MEASURED, source.
     ``maneuver_to_anchor`` / ``lat_to_anchor`` / ``lon_to_anchor`` / ``lan_gate``
     were all added straight to ``conf``. ``scripts/refc_train.py`` already LOGS
     ``graft_lat_norm`` / ``graft_lon_norm`` / ``conf_norm`` — **the instrument
     exists and there is no actuator.** A graft that swamps the base score is not
     a prior, it is a second selector (flagship v4 §6.2 discipline 4; that
     failure mode fired at 2.80x).
  5. **The consequences of candidates never reach the ranking.** This is
     ``cond_imagination``, which ``flagship_v15`` calls "THE NOVEL PART" and
     which was hard-wired OFF in the flagship for months.

⇒ D-SEL rebuilds the ranking, and ONLY the ranking. Every lever here is
zero-init or param-free, so a disabled flag gives a byte-identical state_dict.

⚠️ **WHAT THIS MODULE DOES NOT REST ON.** An earlier framing — *"REF-C's
separation from flagship v1 is ENTIRELY LATERAL"* — is **RETRACTED**
(``RETRACTION_LOG.md`` R-2026-08-03-C): ``dist_to_gt_traj_m`` and
``cross_track_abs_m`` are byte-identical in ``metrics_empty.json`` (four lateral
metrics, not five), and on the shipped HQ render ADE separates at +7.1642
[+5.2654, +8.9661]. The four lateral separations hold and widen; "entirely
lateral" does not, and no argument here depends on it.

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

__all__ = ["assert_candidate_axis", "reachability_mask",
           "anchor_reachability_mask", "anchor_prefilter_report", "SeamState",
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


def anchor_reachability_mask(anchors: Tensor, v0: Tensor, *,
                             accel_max: float = 2.5,
                             horizon_s: float = 2.0) -> Tensor:
    """[B, N] bool — THE SAME BAND, applied to the RAW ANCHORS *before* decoding.

    WHY THIS IS NOT A SECOND IMPLEMENTATION. It calls :func:`reachability_mask`
    on the anchors themselves, so the geometry is the one function the 72.08 %
    was measured with. Only the *stage* differs, and that is the whole point:
    ``v0`` is known PRE-decode, so the band can be evaluated on ``anchors``
    while ``anchors + offset`` does not exist yet. S2 (``refc.py``) filters the
    DECODED fan and therefore saves nothing — it can only narrow an argmax that
    has already cost N decodes.

    MEASURED (2026-08-04, `be2da04`): filtering anchors and decoding only the
    survivors leaves the **selection index identical on 881/881 windows** —
    the same integer, not a tolerance — so all four metric families and
    distance-keeping move by exactly 0.0, at 2.78x fewer decodes
    (small 64->23, base 128->46, XL 256->92).

    ⚠️ **Reachability is a property of the anchor set and ``v0``, not of the
    model.** A rebuilt or re-fit anchor set changes the survivor counts, so the
    2.78x is a property of *those* anchors and must be re-measured, never
    inherited, when the anchors change.

    ⚠️ **``ego_keep`` still binds.** ``v0`` is the speed BEFORE ego-dropout;
    filtering a sample whose speed was withheld from the conditioning would leak
    the channel back in through the candidate set — an even more direct leak
    than the ranking one S2 guards, because here the withheld channel decides
    which candidates *exist*. Callers must OR in ``~ego_keep`` exactly as
    ``refc.py`` does for S2.
    """
    if anchors.dim() == 3:                       # [N, S, 2] -> [B, N, S, 2]
        anchors = anchors[None].expand(v0.shape[0], *anchors.shape)
    return reachability_mask(anchors, v0, accel_max=accel_max,
                             horizon_s=horizon_s)


def anchor_prefilter_report(anchor_keep: Tensor, sel_idx: Tensor) -> dict:
    """THE RUNTIME GUARD `be2da04` says to ship with the fixed-budget policy.

    ``anchor_keep`` [B, N] from :func:`anchor_reachability_mask`; ``sel_idx``
    [B] the argmax the FULL fan produced. Returns per-batch telemetry and,
    crucially, ``winner_survives_frac`` — the fraction of rows whose full-fan
    winner is inside the pre-decode band.

    **Why a guard and not a proof.** `be2da04` separates two claims that are
    easy to merge and must not be: the VARIABLE-width policy (decode every
    survivor) is structurally exact, while a FIXED ``N_suff`` budget is an
    EMPIRICAL CALIBRATION — XL's worst window had **102** survivors against a
    budget of 92, and it only held because the winner's rank never exceeded 92
    *on that corpus*. A budget calibrated on one corpus is not a guarantee on
    the next, so the equivalence is asserted per-run, not assumed.

    ``winner_survives_frac < 1.0`` means the prefilter would have CHANGED the
    emitted trajectory. That is a correctness failure, not a speed regression.
    """
    if anchor_keep.dim() != 2:
        raise ValueError(f"anchor_keep must be [B, N], got {tuple(anchor_keep.shape)}")
    b, n = anchor_keep.shape
    if sel_idx.shape[0] != b:
        raise ValueError(f"sel_idx batch {sel_idx.shape[0]} != anchor_keep batch {b}")
    survivors = anchor_keep.sum(dim=1)
    won = anchor_keep.gather(1, sel_idx.reshape(b, 1).long()).squeeze(1)
    return {
        "n_candidates": int(n),
        "survivors_mean": float(survivors.to(torch.float64).mean()),
        "survivors_max": int(survivors.max()),
        "decode_speedup": (float(n) / float(survivors.to(torch.float64).mean())
                           if float(survivors.to(torch.float64).mean()) > 0
                           else float("inf")),
        "winner_survives_frac": float(won.to(torch.float64).mean()),
        "rows_empty": int((survivors == 0).sum()),
    }


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
