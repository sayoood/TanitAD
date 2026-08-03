"""Factorised tactical vocabulary for REF-C — LATERAL (3) x LONGITUDINAL (3).

WHY THIS MODULE EXISTS — the defect, re-derived from source
============================================================================
``refc.py`` emits ONE 5-way softmax (``N_MANEUVERS = 5``) over
``(lane_keep, turn_left, turn_right, accelerate, brake_stop)``: THREE lateral
classes and TWO longitudinal ones in a single mutually-exclusive simplex.
``scripts/refb_labels.py`` mints that label by a PRIORITY collapse of two
orthogonal axes (``classify_maneuver`` L100-109 / ``classify_maneuver_v2``
L339-347): ``turn > brake > accel > lane_keep``.

The 5-way label is therefore a deterministic function of a (lat, lon) PAIR:

    man5 = lat                                if lat != lane_keep
         = {brake_stop, lane_keep, accelerate}[lon]   otherwise

which is exactly :data:`COLLAPSE_TABLE`. Under that map a factorised posterior
``p_lat x p_lon`` induces the 5-way posterior

    P5(turn_left)  = P_lat(turn_left)
    P5(turn_right) = P_lat(turn_right)
    P5(lane_keep)  = P_lat(lane_keep) * P_lon(steady)
    P5(accelerate) = P_lat(lane_keep) * P_lon(accelerate)      <-- a PRODUCT
    P5(brake_stop) = P_lat(lane_keep) * P_lon(brake_stop)      <-- a PRODUCT

and the 5-way ``argmax`` emits ``accelerate`` only when

    P_lat(lane_keep) * P_lon(accelerate)  >  P_lat(lane_keep) * P_lon(steady)
                                          >  P_lat(turn_left)
                                          >  P_lat(turn_right)

The first line is the longitudinal question we actually want answered. Lines
two and three are a LATERAL comparison that has nothing to do with it, and they
are the mixing: any lateral uncertainty raises the bar a longitudinal class must
clear. That is the mechanism, in algebra, from source.

MEASURED (REF-C-base step 29999, canonical val, **n = 1364 windows / 39
episodes**; run directory ``TanitAD Research Hub/Architecture & Inference/
Implementation/incoming/2026-08-03-dtac1-tactical-head/``, substrate
``dtac1_substrate_refc-base-30k.pt``, banked in that directory):

    class          n true   n predicted   recall
    lane_keep         818       1078      0.9743
    turn_left         174        165      0.8218
    turn_right        109        114      0.8349
    accelerate        146          0      0.0000
    brake_stop        117          7      0.0256

(An earlier n = 859 table from ``…/2026-08-03-lan-refc-e0/LAN_E0_RESULTS.md``
section 5 is the SAME 39 episodes at a sparser stride and is NOT window-for-
window comparable; the 1364-window grid is the decision-grade one.)

The turns are emitted at very nearly their true rate (165 vs 174, 114 vs 109) —
so ``P_lat`` is well learned and the naive reading "the lateral classes win
every argmax" is WRONG. 1078 - 818 = 260 excess ``lane_keep`` against
146 + 117 - 7 = 256 missing longitudinal, i.e. the failure is the
WITHIN-lane_keep comparison ``P_lon(steady)`` vs ``P_lon(brake)`` vs
``P_lon(accel)``, which is degenerate.

⚠️ CORRECTED 2026-08-03 (adversarial R1): this used to read "the longitudinal
mass lands ENTIRELY in lane_keep". It does not. Of 263 true-longitudinal
windows, 241 (**91.6 %**) go to ``lane_keep`` and **19 (7.2 %) go to a LATERAL
class**. "Almost entirely" is the claim the measurement supports.

Three separable causes follow, and this module + ``refc.py``'s gated
``factored_maneuver`` seam make each independently ablatable. (Cited by SYMBOL,
not line number: the L925/L916-922/L966 citations this docstring used to carry
were stale the day they were written — adversarial R14 — and moved again when
the F1 seam was decoupled.)

  F1 INPUT.     ``RefCModel.forward`` computed ``man_logits =
                self.maneuver_head(pooled)`` where ``pooled`` is the IMAGE
                embedding alone, while the ego speed ``v0`` reached only
                ``self.measurement`` -> the decoder condition. The longitudinal
                label IS ``dv = v(t+2s) - v(t)``, so the head was asked a
                question about speed while blind to speed. (``refc1``'s
                ``speed_cls`` already concatenates the measurement, so the fix
                is a pattern the same file uses.) FIXED, gated:
                ``RefCConfig.tactical_speed_input`` — INDEPENDENT of
                ``factored_maneuver`` since 2026-08-03, so
                ``refc.refc_f1only_config()`` is an INPUT-only arm on the
                unchanged 5-way head (+384 params, MEASURED).
  F2 STRUCTURE. The single softmax multiplies the longitudinal decision by the
                lateral one (algebra above) and cannot represent
                "lane_keep AND braking" at all. It is also the LABEL's defect:
                MEASURED 132 / 1364 = **9.68 %** of windows carry a live
                longitudinal manoeuvre AND are labelled a turn, so no decode
                rule can recover them. That is what needs the retrain.
  F3 DECISION.  Even factorised, a 3-way softmax under an unweighted CE learns
                P(lon|x); its argmax emits a minority class only where that
                class's posterior exceeds the majority's. On these windows the
                longitudinal marginal is brake 0.1122 / steady 0.7104 /
                accelerate 0.1774, so factorisation ALONE does not guarantee
                emission (MEASURED: at tau = 0 brake recall 0.0719, accelerate
                0.0455). Prior-corrected decoding (:func:`logit_adjust`) is the
                matching decision-rule fix.
                ⚠️ AND IT IS SMALLER THAN IT FIRST LOOKED. With tau chosen
                LEAVE-ONE-EPISODE-OUT rather than off val (``scripts/
                refc_tactical_tau_select.py``, results ``DTAC1B_RESULTS.md``):
                brake recall 0.0719 -> 0.4248 at precision 0.2340 -> 0.1711, and
                on the 1232 windows the 5-way LABEL can represent the whole patch
                is **NOT separated** from doing nothing (paired episode-cluster
                bootstrap: macro-F1 +0.0107 [-0.0418, +0.0665]). Never quote its
                recall without its precision.

CONTRACT / PROVENANCE
============================================================================
Thresholds and class orders below are MIRRORED from ``scripts/refb_labels.py``
rather than imported: ``tanitad.refs.*`` must not import from ``scripts/``
(stated at ``refc.py`` L97-99 for ``N_MANEUVERS``, same rule as
``LAN_FEATS_PER_ANCHOR``). ``tests/test_refc_tactical.py`` pins every mirrored
constant equal to its ``refb_labels`` original AND fuzzes
``collapse(factor(...)) == classify_maneuver*(...)`` elementwise, so a silent
divergence fails loudly instead of mis-labelling a training run.

Nothing here changes any existing label: the factorisation is a REFINEMENT of
the same thresholds, and its projection through :data:`COLLAPSE_TABLE` is
byte-identical to the 5-way labeler the trainer uses today.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

# ---- legacy 5-way vocabulary (order pinned against refb_labels) -------------
LANE_KEEP, TURN_LEFT, TURN_RIGHT, ACCELERATE, BRAKE_STOP = range(5)
MAN5_NAMES = ("lane_keep", "turn_left", "turn_right", "accelerate",
              "brake_stop")
N_MAN5 = 5

# ---- the factorised axes ----------------------------------------------------
# LATERAL keeps the 5-way lateral order so index 0/1/2 map straight across.
LAT_CLASSES = ("lane_keep", "turn_left", "turn_right")
LAT_LANE_KEEP, LAT_TURN_LEFT, LAT_TURN_RIGHT = range(3)
N_LAT = len(LAT_CLASSES)

# LONGITUDINAL is ordered by SIGN (brake < steady < accelerate) so the class
# index is monotone in dv — an ordinal structure a later ordinal/regression
# variant can exploit, and which makes the confusion matrix readable.
LON_CLASSES = ("brake_stop", "steady", "accelerate")
LON_BRAKE_STOP, LON_STEADY, LON_ACCELERATE = range(3)
N_LON = len(LON_CLASSES)

# ---- thresholds MIRRORED from scripts/refb_labels.py (pinned by tests) ------
YAW_TURN_RAD = 0.15                 # refb_labels.YAW_TURN_RAD   (v1 turn gate)
DV_ACCEL_MS = 1.0                   # refb_labels.DV_ACCEL_MS
DV_BRAKE_MS = -1.0                  # refb_labels.DV_BRAKE_MS
STOP_V_MS = 0.3                     # refb_labels.STOP_V_MS
MOVING_V_MS = 1.0                   # refb_labels.MOVING_V_MS
MIN_ARC_M = 0.10                    # refb_labels.MIN_ARC_M
CURV_TURN_MAN_PER_M = 1.0 / 60.0    # refb_labels.CURV_TURN_MAN_PER_M (v2 gate)
LABEL_HORIZON = 20                  # refb_labels.LABEL_HORIZON (2 s @ 10 Hz)

# ---- the collapse map: COLLAPSE_TABLE[lat, lon] -> 5-way class --------------
# This IS the priority rule `turn > brake > accel > lane_keep`, written as a
# table. Read row-wise: a turn absorbs the longitudinal decision entirely (the
# information-destroying step), and only the lane_keep row lets a longitudinal
# class survive into the 5-way label.
COLLAPSE_TABLE: tuple[tuple[int, ...], ...] = (
    (BRAKE_STOP, LANE_KEEP, ACCELERATE),      # lat = lane_keep
    (TURN_LEFT, TURN_LEFT, TURN_LEFT),        # lat = turn_left   (lon LOST)
    (TURN_RIGHT, TURN_RIGHT, TURN_RIGHT),     # lat = turn_right  (lon LOST)
)


def collapse_table(device=None, dtype=torch.long) -> Tensor:
    """:data:`COLLAPSE_TABLE` as a [N_LAT, N_LON] tensor."""
    return torch.tensor(COLLAPSE_TABLE, dtype=dtype, device=device)


def wrap_to_pi(a: Tensor) -> Tensor:
    """Wrap angles to (-pi, pi] — mirrors ``refb_labels.wrap_to_pi``."""
    return a - (2 * math.pi) * torch.floor((a + math.pi) / (2 * math.pi))


# ============================================================================
# Label side — factorise the SAME kinematics the 5-way labeler reads
# ============================================================================

def factor_from_kinematics(dyaw: Tensor, dv: Tensor, v0: Tensor, v1: Tensor,
                           kappa: Tensor | None = None
                           ) -> tuple[Tensor, Tensor]:
    """(dyaw, dv, v0, v1) -> (lat [.], lon [.]) int64 class ids.

    ``kappa is None`` reproduces the **v1** gate (``refb_labels.
    classify_maneuver``: a turn is ``|dyaw| > YAW_TURN_RAD``); passing the
    window-mean curvature reproduces the **v2** gate
    (``classify_maneuver_v2``: ``|kappa| >= CURV_TURN_MAN_PER_M``). Both
    branches use the IDENTICAL longitudinal rule, which is the whole point —
    the lateral gate is the only thing v1 and v2 ever disagreed about.

    LONGITUDINAL rule, lifted verbatim from the 5-way labeler's assignment
    order (``accel`` first, then ``brake`` OVERWRITES it):

        brake_stop  <=  dv < DV_BRAKE_MS  OR  (v1 < STOP_V_MS and
                                               v0 >= MOVING_V_MS)
        accelerate  <=  dv > DV_ACCEL_MS  and not brake
        steady      <=  otherwise

    ``brake`` and ``accelerate`` are provably disjoint at these thresholds
    (``dv > +1`` with ``v1 < 0.3`` needs ``v0 < -0.7``), so ``lon`` is a genuine
    3-way partition and not another priority collapse.
    """
    lat = torch.full(dyaw.shape, LAT_LANE_KEEP, dtype=torch.long,
                     device=dyaw.device)
    if kappa is None:
        turn_l = dyaw > YAW_TURN_RAD
        turn_r = dyaw < -YAW_TURN_RAD
    else:
        turn = kappa.abs() >= CURV_TURN_MAN_PER_M
        turn_l = turn & (dyaw > 0)
        turn_r = turn & (dyaw < 0)
    lat[turn_l] = LAT_TURN_LEFT
    lat[turn_r] = LAT_TURN_RIGHT

    lon = torch.full(dv.shape, LON_STEADY, dtype=torch.long, device=dv.device)
    lon[dv > DV_ACCEL_MS] = LON_ACCELERATE
    brake = (dv < DV_BRAKE_MS) | ((v1 < STOP_V_MS) & (v0 >= MOVING_V_MS))
    lon[brake] = LON_BRAKE_STOP
    return lat, lon


def collapse(lat: Tensor, lon: Tensor) -> Tensor:
    """(lat, lon) -> the shipped 5-way class (the priority collapse, as a
    gather over :data:`COLLAPSE_TABLE`). Exactly inverts nothing — it is the
    lossy direction, and that loss is the defect this module exists to bound."""
    tbl = collapse_table(device=lat.device)
    return tbl[lat.reshape(-1), lon.reshape(-1)].reshape(lat.shape)


def window_factored_labels(pose_last: Tensor, future_poses: Tensor,
                           horizon: int = LABEL_HORIZON
                           ) -> tuple[Tensor, Tensor]:
    """Batch **v1**-faithful factored labels — the endpoint rule the REF-C
    trainer uses today (``refb_labels.window_maneuver_labels``, which
    ``scripts/refc_train.py`` L345-346 calls).

    ``pose_last`` [B, 4] = (x, y, yaw, v); ``future_poses`` [B, H, 4] with
    H >= horizon. Returns ``(lat [B], lon [B])`` int64.
    """
    if future_poses.shape[1] < horizon:
        raise ValueError(f"future_poses has only {future_poses.shape[1]} "
                         f"steps; label horizon needs {horizon}")
    p1 = future_poses[:, horizon - 1]
    dyaw = wrap_to_pi(p1[:, 2] - pose_last[:, 2])
    return factor_from_kinematics(dyaw, p1[:, 3] - pose_last[:, 3],
                                  pose_last[:, 3], p1[:, 3])


def window_factored_labels_v2(pose_last: Tensor, future_poses: Tensor,
                              horizon: int = LABEL_HORIZON
                              ) -> tuple[Tensor, Tensor]:
    """Batch **v2**-faithful factored labels — the curvature-gated rule
    (``refb_labels.window_maneuver_labels_v2``), so a gentle highway curve stays
    ``lane_keep`` instead of being called a turn."""
    if future_poses.shape[1] < horizon:
        raise ValueError(f"future_poses has only {future_poses.shape[1]} "
                         f"steps; label horizon needs {horizon}")
    sub = torch.cat([pose_last[:, None, :], future_poses[:, :horizon]], dim=1)
    dyaw = wrap_to_pi(sub[:, -1, 2] - sub[:, 0, 2])
    seg = (sub[:, 1:, :2] - sub[:, :-1, :2]).norm(dim=-1)
    kappa = dyaw / seg.sum(dim=1).clamp_min(MIN_ARC_M)
    return factor_from_kinematics(dyaw, sub[:, -1, 3] - sub[:, 0, 3],
                                  sub[:, 0, 3], sub[:, -1, 3], kappa=kappa)


# ============================================================================
# Model side — forward (factored -> 5-way) and INVERSE (5-way -> factored)
# ============================================================================

def derive_man5_logprobs(lat_logits: Tensor, lon_logits: Tensor) -> Tensor:
    """(lat_logits [B, 3], lon_logits [B, 3]) -> 5-way LOG-probabilities [B, 5].

    The exact push-forward of ``p_lat x p_lon`` through
    :data:`COLLAPSE_TABLE`, in log space::

        log P5(lane_keep)  = log P_lat(lane_keep) + log P_lon(steady)
        log P5(turn_left)  = log P_lat(turn_left)
        log P5(turn_right) = log P_lat(turn_right)
        log P5(accelerate) = log P_lat(lane_keep) + log P_lon(accelerate)
        log P5(brake_stop) = log P_lat(lane_keep) + log P_lon(brake_stop)

    The result is a proper distribution (it sums to 1) BY CONSTRUCTION, so a
    factored model can keep publishing ``maneuver_logits`` [B, 5] and every
    downstream reader (plan_fan HUD, eval harness, closed-loop logger) keeps
    working unchanged. Emitted as log-probs, which is what ``graft_maneuver``
    consumes anyway (``refc.py`` applies ``log_softmax`` to the 5-way logits).
    """
    lat = torch.log_softmax(lat_logits, dim=-1)
    lon = torch.log_softmax(lon_logits, dim=-1)
    lk = lat[:, LAT_LANE_KEEP]
    return torch.stack([lk + lon[:, LON_STEADY],
                        lat[:, LAT_TURN_LEFT],
                        lat[:, LAT_TURN_RIGHT],
                        lk + lon[:, LON_ACCELERATE],
                        lk + lon[:, LON_BRAKE_STOP]], dim=-1)


def invert_man5(man5_logits: Tensor, eps: float = 1e-12
                ) -> tuple[Tensor, Tensor]:
    """5-way logits [B, 5] -> ``(log P_lat [B, 3], log P_lon [B, 3])``.

    The EXACT inverse of :func:`derive_man5_logprobs` under the factorisation
    assumption::

        P_lat(turn_left)  = P5(turn_left)
        P_lat(turn_right) = P5(turn_right)
        P_lat(lane_keep)  = P5(lane_keep) + P5(accelerate) + P5(brake_stop)
        P_lon(.)          = P5(.) / P_lat(lane_keep)     (conditional)

    This is the instrument that makes the cheapest discriminating experiment
    possible: it recovers, from an ALREADY-TRAINED 5-way head, the conditional
    longitudinal posterior the mixed argmax never lets surface. If that
    conditional argmax emits brake/accelerate at the label base rate then the
    information is present and the defect is READOUT-only; if it stays pinned to
    ``steady`` then the head carries no longitudinal information at all and the
    fix must reach the head's INPUT. Both readings are pre-registered.

    Note the conditional is undefined where ``P_lat(lane_keep) -> 0``; the
    ``eps`` floor makes it a uniform-ish conditional there rather than a NaN,
    and callers that care should mask on ``P_lat(lane_keep)``.
    """
    lp5 = torch.log_softmax(man5_logits, dim=-1)
    lk = torch.logsumexp(lp5[:, [LANE_KEEP, ACCELERATE, BRAKE_STOP]], dim=-1)
    log_lat = torch.stack([lk, lp5[:, TURN_LEFT], lp5[:, TURN_RIGHT]], dim=-1)
    lk_c = lk.clamp_min(math.log(eps))
    log_lon = torch.stack([lp5[:, BRAKE_STOP] - lk_c,
                           lp5[:, LANE_KEEP] - lk_c,
                           lp5[:, ACCELERATE] - lk_c], dim=-1)
    return log_lat, torch.log_softmax(log_lon, dim=-1)


def logit_adjust(logits: Tensor, log_prior: Tensor, tau: float) -> Tensor:
    """Prior-corrected logits ``logits - tau * log_prior`` (Menon et al. 2021,
    *Long-tail learning via logit adjustment*, ICLR).

    ``tau = 0`` is the identity, so every default path is unchanged; ``tau = 1``
    decodes the BALANCED posterior ``p(y|x) / pi_y``, i.e. it asks "which class
    is most surprising given its base rate" instead of "which class is most
    likely", and is the decision-rule half of the fix (F3 in the module
    docstring). A UNIFORM ``log_prior`` leaves the argmax unchanged at any
    ``tau`` — subtracting a constant from every logit is a no-op — which is why
    the model's prior buffers start uniform and an un-updated prior can never
    silently alter a published decode.
    """
    if tau == 0.0:
        return logits
    return logits - tau * log_prior.to(logits.dtype).reshape(
        *([1] * (logits.dim() - 1)), -1)


def prior_centered_logprobs(logits: Tensor, log_prior: Tensor) -> Tensor:
    """``log_softmax(logits) - log_prior`` — the log-likelihood RATIO.

    What a *prior* should contribute to a ranking is evidence, not base rate.
    Feeding ``log_softmax`` straight into ``maneuver_to_anchor`` hands the graft
    a vector dominated by a near-constant offset (a class at 1 % base rate sits
    at ``log p ~ -4.6`` on almost every window), so the learned
    ``Linear(k, n_anchors)`` spends its capacity on a constant per-anchor bias.
    Centering by the class log-prior removes that offset.

    HONEST SCOPE: the graft Linear is ``bias=False``, so subtracting a constant
    vector ``c`` only shifts its output by the constant ``W c`` — a per-anchor
    bias the layer could in principle have learned anyway. The claim here is
    therefore about CONDITIONING (signal-to-constant ratio at the graft input),
    **not** expressivity, and it is gated (``graft_prior_center``) so it can be
    ablated to zero and measured rather than believed.
    """
    return torch.log_softmax(logits, dim=-1) - log_prior.to(
        logits.dtype).reshape(*([1] * (logits.dim() - 1)), -1)


def class_log_prior(idx: Tensor, n_classes: int, eps: float = 1e-6) -> Tensor:
    """Empirical log class prior of an index tensor -> [n_classes]."""
    cnt = torch.bincount(idx.reshape(-1), minlength=n_classes).to(torch.float32)
    return (cnt / cnt.sum().clamp_min(1.0)).clamp_min(eps).log()
