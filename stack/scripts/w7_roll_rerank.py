"""W7 = X0 full — the WM-roll MPC re-rank on the FROZEN W4 unicycle fan,
now the PRIMARY selection mechanism for v5.8f by MEASURED ELIMINATION.

WHY W7 IS PRIMARY (the elimination chain, every step banked):
  * the frozen v5f selector on the W4 fan is near-uninformed — selected ADE
    **0.7933** vs oracle **0.1077** (``w4_gate.json``; p7_regrade
    ``v5f-frozen-argmax``: sel_rank_pct_mean 0.2593);
  * W4b recalibration (feat **0.5600** held out, kin **0.5637**) FAILED G1
    (<= 0.45) with train monitors at 0.21-0.33 — MEMORISATION through the
    pooled query surface, not a generalising rule (``w4b_gate_feat.json``:
    G2 engaged, "W7 ... becomes the primary selection mechanism");
  * W4c (spatial cross-attention port) is the last fast scoring attempt —
    its G-null branch verbatim: "selection moves ENTIRELY to W7 WM-roll
    re-rank (already primary per W4b's G2)";
  * P7-regrade shows NEITHER current arm carries calibrated scores:
    Spearman(score-spread, realised error) rho 0.0542 (CI [-0.1395, 0.2387])
    for the rescorer-top8-kincost arm and 0.2622 (CI [0.0909, 0.4098]) for
    frozen-argmax — both FAIL the P7 gate rho >= 0.3 with CI excluding 0
    (``p7_regrade.json``). Scores don't know when they're wrong; an explicit
    per-candidate CONSEQUENCE CHECK is the remaining lever.

THE MECHANISM (DIFFUSION_MPC_SYNTHESIS.md L1 + experiment X0): keep
EVERYTHING frozen. Selector prunes the 256-fan to top-K; for each survivor,
feed its OWN (a, kappa) control sequence to the predictor as actions, roll k
steps, decode the imagined latents to waypoints, and score each candidate on
ITS OWN imagined future with an explicit, auditable cost. Emit the argmin.
This is the MPC-style path: the roll is IMAGINATION-CLOSED — the actions are
the candidate's, NOT the recorded ones — so unlike every T0 fan metric it
never teacher-forces the ground-truth future through the roll.

⭐ THE ROLL AND THE DECODE (cited, byte-reused — the stage_a_probes seams):
  * roll: ``tanitad.models.metric_dynamics.rollout_transitions``
    (metric_dynamics.py:247-266) — the same latents-out roll the canary/P8/W3
    family uses; ``trans[j][1]`` is ẑ_{t+j+1}.
  * decode: ``decode_transitions(grounding.step["op"], trans, k)``
    (metric_dynamics.py:269-280) — pinned in its docstring to reproduce
    ``rollout_decode`` (the canary decode, train_flagship_v4.py:584-586)
    exactly: same latents, same step readout, same ``accumulate_se2``
    (metric_dynamics.py:114-139) SE(2) accumulation. Ego +x fwd / +y left.
  * action lift: ``train_p8_occupancy.lift_actions3`` (train_p8_occupancy.py:
    417-424) — the exact ``canary_rollout`` 3-channel speed-append
    (train_flagship_v4.py:578-580). ⚠️ Stated limitation inherited from W3:
    the speed channel stays at the OBSERVED v0 for EVERY candidate (that is
    the trunk's own rollout contract — the canary holds it constant too), so
    candidates differ through the steer/accel channels only.
  * kappa -> steer encoding: ``stage_a_probes.steer_of_kappa`` (stage_a_
    probes.py:174-177) — ``steer_road_rad = atan(WHEELBASE * kappa)`` with
    the LEGACY constant wheelbase 2.9 m (physicalai.py:621 mint;
    ``DEFAULT_WHEELBASE_MODE = "const2p9"`` physicalai.py:69/82 — every
    published cache is that regime). Candidate curvature enters the
    predictor through the SAME encoding the corpus actions were minted with.
  * action placement: the stage_a convention (stage_a_probes.py:44-47,
    219-232) — the roll consumes ``aw2[:, -1]`` as step 0 then
    ``fa2[:, :k-1]``; only the LAST window action and the future actions
    carry the candidate's controls. Earlier window actions stay the OBSERVED
    history — replacing them would ask the model to re-explain the past, not
    imagine a candidate future.
  * states: captured at ``world.readout``'s output (:class:`StateTap`) — the
    pooled per-frame states ``encode_window`` returns (fourbrain.py:470-478)
    — a forward HOOK, not a second encoder pass (the W4c SpatialTokenTap
    pattern, train_w4c_spatial.py:155-197), with the same strict
    exactly-one-capture contract.

PRUNING (``--prune-rule``, default ``frozen``; ``--topk``, default 8 — the
L1 "selector prunes the fan 256 -> top-8"):
  * ``frozen``: top-K by the FROZEN selector's own per-candidate sel_score on
    the new fan — ``out["refined_logits"]`` (refc.py:1514; the surface
    p7_regrade ranked ``v5f-frozen-argmax`` by). Reference pruning: no new
    trainable part, attribution stays clean.
  * ``oracle-debug``: top-K by TRUE per-candidate error. ⛔ DIAGNOSTIC ONLY —
    it leaks GT into the shortlist, so a run under it is the re-rank's UPPER
    BOUND given perfect pruning, never a claimable number. The gate JSON
    refuses to emit a pass/fail verdict for it.
  ⭐ ARGMAX-INCLUSION GUARANTEE (:func:`force_include`): the frozen
  selector's own pick ``out["sel_idx"]`` (its FULL select(), incl. vt-keep —
  not a recomputed argmax) is ALWAYS in the shortlist, replacing the
  weakest-ranked slot when absent. Stated precisely: this guarantees the
  shortlist's ORACLE is never worse than the frozen pick — W7 cannot lose by
  exclusion. Whether the realised PICK beats the frozen pick still depends
  on the cost ranking it correctly; that is exactly what the gate measures
  (with an oracle cost the guarantee is exact: W7 <= frozen per window).

COST (each term flag-weighted; the DEFAULTS ARE KNOBS, NOT MEASURED OPTIMA —
say so wherever quoted; a weight sweep is a follow-up, not smuggled in here):
  (a) ROLL-CONSISTENCY (``--w-roll``, default 1.0) — ADE between the
      candidate's own unicycle waypoints and the WM-rolled decoded waypoints
      under the SAME actions (:func:`roll_consistency`). The WM's veto: a
      candidate whose consequences the WM predicts differently from what the
      candidate itself claims is suspect. (Requires W3's controllability
      finding to hold on this trunk — if the predictor ignores actions, this
      term is fiction; the calibration block below is the in-run check.)
  (b) KINEMATIC (``--w-kin``, default 0.2) — ``mean|a| + 0.5*mean|jerk|``
      FROM THE CONTROLS: ``tanitad.models.v58f.kinematic_cost`` (v58f.py:
      99-116, IMPORTED not re-derived). Meaningful on the CLEAN fan only —
      W1 refuted this family on the 97.6 %-infeasible old fan (ranking
      jitter); on the unicycle fan the controls ARE the kinematics
      (violations 0.0), which is what re-admits it (v58f.py:26-36).
  (c) PROGRESS (``--w-prog``, default 0.0 = OFF) — negative arc length of
      the candidate's waypoints (:func:`progress_arc_length`).
  Selection = argmin cost inside the shortlist (:func:`select_from_cost`).

⛔ PRE-REGISTERED GATE (V58F_FUSION.md §3 W7 row, verbatim: "≥50 % sel_gap
closed at T1"; this instrument is the T0-FIRST read per the brief's stated
order — the T1 confirmation runs pod-side via taniteval/tools/t1_eval.py
before any capability claim):
    (0.7933 - W7_selected) / (0.7933 - 0.1077) >= 0.5  <=>  W7 <= 0.4505 m
  on the same 881-window grid. ALSO REPORTED (not the gate): the tougher
  read vs the deployed interim arm — (0.4815 - W7) / (0.4815 - 0.1077) —
  because closing half the FROZEN gap while losing to rescorer-top8-kincost
  would be a pass that changes nothing; both fractions are stated.

P7-STYLE CALIBRATION BLOCK: Spearman(cost of the selected candidate,
realised error of the selected candidate) across windows, with the
EPISODE-CLUSTER bootstrap CI (:func:`cluster_bootstrap_spearman`; rank
convention = tools_p7_calibration.py:15-19), plus the mean WITHIN-window
Spearman(cost, error) over the shortlist. W7's costs should be calibrated
where the learned scores were not (P7 refs above); read against the P7 gate
rho >= 0.3 with CI excluding 0.

TIER: **T0** — the OUTER eval conditions on logged frames and scores against
the logged future, so the number is a WM diagnostic, never driving
performance (EVAL_DOCTRINE.md). The roll INSIDE is imagination-closed under
the candidate's own actions (the MPC-style path, the T1-flavoured
mechanism), but that does not upgrade the tier of the outer metric.

ESTIMATOR: point estimates over the eval grid; per-window arrays banked to
``<out>/w7_eval_windows.pt`` for the pod-side episode-cluster bootstrap
(taniteval.selgap / taniteval/ci.py — computed in-process when importable,
best-effort). Never overlapping_holdout_se.

⚠️ POD-SIDE ONLY for the full path: this box has no GPU, no v5f checkpoint,
no W4 checkpoint, no v2 corpus. Runnable (and run) here: ``python -m
py_compile`` + the pure-part CPU tests ``stack/tests/test_w7.py``. Nothing
trains — W7 is a training-free re-rank instrument; never run it on a
training pod (CLAUDE.md invariant).

Usage (pod5; PYTHONPATH=/workspace/TanitAD/stack required or it dies with
ModuleNotFound: tanitad):

  python3 w7_roll_rerank.py \
      --ckpt /workspace/experiments/flagship-v5f-w120-30k/ckpt_30k_final.pt \
      --w4-ckpt /workspace/experiments/w4-unicycle-head-c/unicycle_emission.pt \
      --anchors-dense /workspace/experiments/anchors/anchors_dense_1to20.pt \
      --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
      --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
      --v2-subframe 176x624 \
      --out /workspace/experiments/w7-roll-rerank
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent))
# stack root too, so `import tanitad` resolves even without the pod-side
# PYTHONPATH (harmless when PYTHONPATH is set — same directory).
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from stage_a_probes import WHEELBASE, steer_of_kappa  # noqa: E402
from train_w4b_selector import (EXPECTED_GRID_WINDOWS,  # noqa: E402
                                fan_ade, load_w4_emission,
                                selected_family_sums)
from train_v58f_unicycle_head import (DT, OffsetFeatureTap,  # noqa: E402
                                      module_md5)
from tanitad.models.v58f import kinematic_cost  # noqa: E402  (torch-only)

# ---- the banked references (registry §1.13 / w4_gate / p7_regrade, verbatim)
REF_FROZEN_ARGMAX = 0.7933        # frozen v5f selector's pick on the W4 fan
REF_RESCORER_TOP8_KINCOST = 0.4815  # p7_regrade v58f-rescorer-top8-kincost
REF_ORACLE = 0.1077               # W4 fan oracle ADE (881 grid)
REF_W4B_FEAT = 0.5600             # W4b feat held-out selected ADE (failed G1)
REF_W4B_KIN = 0.5637              # W4b kin held-out selected ADE (failed G1)
# ---- the pre-registered gate (V58F_FUSION.md §3, W7 row) --------------------
GATE_FRAC = 0.5                   # ">=50 % of sel_gap closed"
#: (0.7933 - W7)/(0.7933 - 0.1077) >= 0.5  <=>  W7 <= 0.4505 — this literal
#: threshold is the authoritative form (the fraction is shown at float
#: precision alongside).
W7_GATE_THRESHOLD = 0.4505
# ---- P7 calibration references (p7_regrade.json, verbatim) ------------------
P7_GATE_RHO = 0.3                 # PRE-REGISTERED: rho >= 0.3, CI excluding 0
REF_P7_RHO_RESCORER = 0.0542      # rescorer-top8-kincost, CI [-0.1395, 0.2387]
REF_P7_RHO_FROZEN = 0.2622        # frozen-argmax,        CI [0.0909, 0.4098]
# ---- knobs (defaults are KNOBS, NOT MEASURED OPTIMA — module docstring) -----
TOPK_DEFAULT = 8                  # the L1 "256 -> top-8" prune
ROLL_K_DEFAULT = 10               # 1 s @ 10 Hz (the W3/P3 probe horizon)
W_ROLL_DEFAULT = 1.0
W_KIN_DEFAULT = 0.2
W_PROG_DEFAULT = 0.0
PRUNE_RULES = ("frozen", "oracle-debug")


# ============================================================================
# the states tap — encode_window's pooled output, captured not recomputed
# ============================================================================
class StateTap:
    """Capture the pooled per-frame states ``[B*W, S]`` at ``world.readout``'s
    output (``encode = readout(encoder(x))``, ``encode_window`` reshapes —
    fourbrain.py:470-478) via a forward hook — the W4c ``SpatialTokenTap``
    pattern (train_w4c_spatial.py:155-197): identical numerics to what
    ``frozen_forward`` computed, zero extra compute.

    STRICT contract: :meth:`states` requires EXACTLY ONE captured call since
    :meth:`clear`. ``frozen_forward`` runs ``encode_window`` once per batch;
    the goal/imagination inputs are predictor-only (train_flagship_v4.py:
    220-265) and ``rollout_transitions`` touches only ``world.predictor`` —
    so a second readout pass appearing later must fail loudly, never silently
    roll from the wrong states."""

    def __init__(self, readout: torch.nn.Module):
        self._buf: list[Tensor] = []
        self._h = readout.register_forward_hook(
            lambda _m, _args, output: self._buf.append(output))

    def clear(self) -> None:
        self._buf.clear()

    def n_calls(self) -> int:
        return len(self._buf)

    def states(self, b: int, w: int) -> Tensor:
        """``(B, W) -> [B, W, S]`` — encode_window's return, detached (frozen
        trunk), dtype preserved (the roll runs under the same autocast the
        canary uses; casting here would change the roll's numerics)."""
        if len(self._buf) != 1:
            raise RuntimeError(
                f"StateTap: {len(self._buf)} readout passes captured, "
                "expected exactly 1 — was clear() called before "
                "frozen_forward, and does something now run encode/"
                "encode_window twice?")
        t = self._buf[0]
        if t.ndim != 2 or t.shape[0] != b * w:
            raise ValueError(
                f"StateTap: captured {tuple(t.shape)}, expected "
                f"[{b}*{w}={b * w}, S] — not the encode_window path?")
        return t.reshape(b, w, -1).detach()

    def remove(self) -> None:
        self._h.remove()


# ============================================================================
# pure helpers (CPU-testable — tests/test_w7.py)
# ============================================================================
def shortlist_indices(prune_rule: str, k: int, *, scores: Tensor | None = None,
                      err: Tensor | None = None) -> Tensor:
    """Prune the fan to a top-``k`` shortlist ``[B, k]`` of candidate indices.

    * ``"frozen"``: top-k by the FROZEN selector's per-candidate sel_score
      (``out["refined_logits"]``, higher = preferred) — the reference prune.
    * ``"oracle-debug"``: top-k by TRUE error (lower = better). ⛔ DIAGNOSTIC
      ONLY — leaks GT into the shortlist; the gate JSON refuses a verdict
      under it (:func:`build_w7_gate`).

    ``k`` clips to the fan size. Slot 0 is the strongest-ranked candidate,
    the LAST slot the weakest — :func:`force_include` relies on that order."""
    if prune_rule not in PRUNE_RULES:
        raise ValueError(f"prune_rule must be one of {PRUNE_RULES}, got "
                         f"{prune_rule!r}")
    key = scores if prune_rule == "frozen" else \
        (None if err is None else -err)
    if key is None:
        need = "scores" if prune_rule == "frozen" else "err"
        raise ValueError(f"prune_rule={prune_rule!r} requires {need}")
    if key.ndim != 2:
        raise ValueError(f"ranking key must be [B, N], got {tuple(key.shape)}")
    kk = min(int(k), key.shape[1])
    if kk < 1:
        raise ValueError(f"topk must be >= 1, got {k}")
    return key.topk(kk, dim=1).indices                       # [B, kk]


def force_include(shortlist: Tensor, frozen_idx: Tensor) -> Tensor:
    """⭐ THE ARGMAX-INCLUSION GUARANTEE: return a shortlist that ALWAYS
    contains the frozen selector's own pick, replacing the WEAKEST-ranked
    (last) slot when absent — rows that already contain it are untouched.

    Precisely stated (module docstring): this guarantees the shortlist's
    ORACLE is never worse than the frozen pick — W7 cannot lose by exclusion.
    The realised PICK equals-or-beats the frozen pick only when the cost
    ranks the two correctly; with an oracle cost the guarantee is exact."""
    if shortlist.ndim != 2 or frozen_idx.ndim != 1 or \
            shortlist.shape[0] != frozen_idx.shape[0]:
        raise ValueError(f"need shortlist [B, k] + frozen_idx [B], got "
                         f"{tuple(shortlist.shape)} / "
                         f"{tuple(frozen_idx.shape)}")
    present = (shortlist == frozen_idx[:, None]).any(dim=1)  # [B]
    out = shortlist.clone()
    out[~present, -1] = frozen_idx[~present]
    return out


def candidate_roll_actions(aw2: Tensor, a_cand: Tensor, kappa_cand: Tensor,
                           roll_k: int, wheelbase: float = WHEELBASE
                           ) -> tuple[Tensor, Tensor]:
    """The candidate's controls as the actions the WM roll consumes.

    ``aw2 [B, W, 2]`` recorded window actions (steer_road_rad, accel_mps2);
    ``a_cand``/``kappa_cand [B, P, Kh]`` the shortlisted candidates' own
    (a, kappa) sequences (``Kh >= roll_k``). Returns
    ``(aw2_cf [B*P, W, 2], fa2_cf [B*P, roll_k-1, 2])`` in B-major/P-minor
    order — the SAME flattening as ``states.repeat_interleave(P, dim=0)``.

    Placement is the stage_a convention (stage_a_probes.py:219-232, the
    INVERSE direction of its ``rolled_action_sequence``): the roll consumes
    ``aw2[:, -1]`` as step 0 and ``fa2[:, :k-1]`` after — so the candidate's
    step-0 control lands in the LAST window slot and steps 1..k-1 in the
    futures. Earlier window actions stay the OBSERVED history (stage_a_
    probes.py:44-47). kappa maps through the corpus steer encoding
    ``steer = atan(wheelbase * kappa)`` (stage_a_probes.steer_of_kappa,
    physicalai.py:621; legacy const2p9)."""
    if aw2.ndim != 3 or aw2.shape[-1] != 2:
        raise ValueError(f"aw2 must be [B, W, 2], got {tuple(aw2.shape)}")
    if a_cand.ndim != 3 or a_cand.shape != kappa_cand.shape:
        raise ValueError(f"a_cand/kappa_cand must be matching [B, P, Kh], "
                         f"got {tuple(a_cand.shape)} vs "
                         f"{tuple(kappa_cand.shape)}")
    if a_cand.shape[0] != aw2.shape[0]:
        raise ValueError(f"batch mismatch: aw2 {aw2.shape[0]} vs candidates "
                         f"{a_cand.shape[0]}")
    if a_cand.shape[-1] < roll_k:
        raise ValueError(f"candidate horizon {a_cand.shape[-1]} < roll_k="
                         f"{roll_k} — the roll would run out of controls")
    if roll_k < 1:
        raise ValueError(f"roll_k must be >= 1, got {roll_k}")
    b, p, _kh = a_cand.shape
    w = aw2.shape[1]
    steer = steer_of_kappa(kappa_cand, wheelbase)            # [B, P, Kh]
    aw = aw2[:, None].expand(b, p, w, 2).clone()             # [B, P, W, 2]
    aw[:, :, -1, 0] = steer[..., 0]
    aw[:, :, -1, 1] = a_cand[..., 0]
    fa = torch.stack([steer[..., 1:roll_k], a_cand[..., 1:roll_k]],
                     dim=-1)                                 # [B,P,roll_k-1,2]
    return aw.reshape(b * p, w, 2), fa.reshape(b * p, roll_k - 1, 2)


def roll_consistency(fan_wp: Tensor, wm_wp: Tensor) -> Tensor:
    """The WM's veto term: ADE between the candidate's OWN unicycle waypoints
    and the WM-rolled decoded waypoints under the SAME actions.
    ``(fan_wp [B, P, k, 2], wm_wp [B, P, k, 2]) -> [B, P]``. Identical
    trajectories -> exactly 0 (pinned in tests)."""
    if fan_wp.shape != wm_wp.shape or fan_wp.ndim != 4 or \
            fan_wp.shape[-1] != 2:
        raise ValueError(f"need matching [B, P, k, 2], got "
                         f"{tuple(fan_wp.shape)} vs {tuple(wm_wp.shape)}")
    return (fan_wp - wm_wp).norm(dim=-1).mean(dim=-1)


def progress_arc_length(fan: Tensor) -> Tensor:
    """Arc length of each candidate's waypoints (origin prepended — the ego
    starts at 0, the W4 geometry): ``[..., K, 2] -> [...]`` in metres. The
    optional progress term is MINUS this (longer path = more progress)."""
    if fan.ndim < 3 or fan.shape[-1] != 2:
        raise ValueError(f"fan must be [..., K, 2], got {tuple(fan.shape)}")
    z = fan.new_zeros(fan.shape[:-2] + (1, 2))
    return torch.diff(torch.cat([z, fan], dim=-2),
                      dim=-2).norm(dim=-1).sum(dim=-1)


def w7_cost(roll_ade: Tensor, kin: Tensor, arc: Tensor, *,
            w_roll: float = W_ROLL_DEFAULT, w_kin: float = W_KIN_DEFAULT,
            w_prog: float = W_PROG_DEFAULT) -> Tensor:
    """The W7 candidate cost (module docstring; weights are KNOBS, not
    measured optima): ``w_roll * roll_consistency + w_kin * kinematic_cost
    - w_prog * arc_length``. All three inputs ``[B, P]`` -> ``[B, P]``."""
    if not (roll_ade.shape == kin.shape == arc.shape):
        raise ValueError(f"term shape mismatch: {tuple(roll_ade.shape)} / "
                         f"{tuple(kin.shape)} / {tuple(arc.shape)}")
    return (w_roll * roll_ade.float() + w_kin * kin.float()
            - w_prog * arc.float())


def select_from_cost(shortlist: Tensor, cost: Tensor) -> Tensor:
    """argmin cost INSIDE the shortlist, mapped back to fan indices:
    ``(shortlist [B, P], cost [B, P]) -> sel [B]``."""
    if shortlist.shape != cost.shape or shortlist.ndim != 2:
        raise ValueError(f"need matching [B, P], got "
                         f"{tuple(shortlist.shape)} vs {tuple(cost.shape)}")
    pick = cost.argmin(dim=1)                                # [B]
    return shortlist.gather(1, pick[:, None]).squeeze(1)


# ============================================================================
# P7-style calibration (pure; rank convention = tools_p7_calibration.py:15-19)
# ============================================================================
def spearman(a, b) -> float:
    """Spearman rho via double-argsort ranks — the tools_p7_calibration.py:
    15-19 convention (no tie averaging; float costs/errors make ties
    measure-zero, stated). ONE guard ADDED over that source: a constant
    vector -> nan explicitly, because stable argsort would otherwise hand a
    constant input distinct ranks in memory order and mint a spurious rho —
    a probe reporting the wrong scope is worse than no probe (CLAUDE.md)."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape or a.size < 2:
        raise ValueError(f"need two equal-length vectors of n >= 2, got "
                         f"{a.shape} vs {b.shape}")
    if np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")                       # no defined rank order
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def cluster_bootstrap_spearman(x, y, eids, n_boot: int = 2000,
                               seed: int = 0) -> dict:
    """Episode-cluster bootstrap CI on Spearman rho — the estimator the
    programme's decision rules demand (CLAUDE.md: never
    overlapping_holdout_se; p7_regrade's ``rho_ci_cluster`` shape).

    Resamples EPISODES with replacement (windows travel with their episode),
    recomputes rho per replicate, percentile 95 % interval. Degenerate draws
    (constant x or y) yield nan and are dropped from the percentiles, counted
    in ``n_boot_degenerate``."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    eids = np.asarray(eids).ravel()
    if not (x.shape == y.shape == eids.shape):
        raise ValueError(f"x/y/eids must match, got {x.shape}/{y.shape}/"
                         f"{eids.shape}")
    rho = spearman(x, y)
    uniq = np.unique(eids)
    out = {"spearman_rho": round(rho, 4), "n": int(x.size),
           "n_episodes": int(uniq.size), "n_boot": int(n_boot), "seed": seed,
           "estimator": "episode_cluster_bootstrap",
           "rank_convention": "tools_p7_calibration.py:15-19 (double "
                              "argsort, no tie averaging)"}
    if uniq.size < 2:
        out["rho_ci_cluster"] = None
        out["ci_note"] = (f"only {uniq.size} episode(s) — a cluster "
                          "bootstrap needs >= 2 resampling units; CI not "
                          "computable, stated rather than faked")
        return out
    idx_by_ep = [np.where(eids == e)[0] for e in uniq]
    rng = np.random.default_rng(seed)
    rhos = []
    n_degen = 0
    for _ in range(int(n_boot)):
        pick = rng.integers(0, uniq.size, uniq.size)
        rows = np.concatenate([idx_by_ep[j] for j in pick])
        r = spearman(x[rows], y[rows])
        if np.isnan(r):
            n_degen += 1
        else:
            rhos.append(r)
    if not rhos:
        out["rho_ci_cluster"] = None
        out["ci_note"] = "every bootstrap replicate degenerate"
        return out
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    out["rho_ci_cluster"] = [round(float(lo), 4), round(float(hi), 4)]
    out["n_boot_degenerate"] = n_degen
    return out


# ============================================================================
# gate JSON (pure: dict-in, dict-out; every branch pinned by tests/test_w7.py)
# ============================================================================
def build_w7_gate(mini: dict, *, prune: dict, cost_cfg: dict, calib: dict,
                  roll: dict) -> dict:
    """The pre-registered W7 gate record. Gate verbatim from V58F_FUSION.md §3
    (W7 row: ">=50 % sel_gap closed"); the brief's definition made explicit:
    ``(0.7933 - W7)/(0.7933 - 0.1077) >= 0.5  <=>  W7 <= 0.4505``. The
    tougher read vs the 0.4815 arm is REPORTED alongside, never substituted.
    Under ``oracle-debug`` pruning the verdict is REFUSED (pass = None): the
    shortlist leaked GT, the number is an upper bound, not a result."""
    sel = mini["selected_ade"]
    denom_frozen = REF_FROZEN_ARGMAX - REF_ORACLE
    denom_resc = REF_RESCORER_TOP8_KINCOST - REF_ORACLE
    frac_frozen = (REF_FROZEN_ARGMAX - sel) / denom_frozen
    frac_resc = (REF_RESCORER_TOP8_KINCOST - sel) / denom_resc
    diagnostic = prune["rule"] == "oracle-debug"
    g_pass = None if diagnostic else bool(sel <= W7_GATE_THRESHOLD)
    return {
        "item": ("W7 = X0 full — WM-roll MPC re-rank on the frozen W4 "
                 "unicycle fan (DIFFUSION_MPC_SYNTHESIS.md L1/X0; "
                 "V58F_FUSION.md §3 W7) — PRIMARY selection mechanism by "
                 "measured elimination (W4b G2 engaged feat 0.5600 / kin "
                 "0.5637; p7_regrade: neither arm's scores calibrated)"),
        "gate_W7_selgap_closed": {
            "rule": ("PRE-REGISTERED (V58F_FUSION.md §3 W7 row, verbatim): "
                     "'≥50 % sel_gap closed at T1'. This run is the T0-FIRST "
                     "read per the brief; the T1 confirmation "
                     "(taniteval/tools/t1_eval.py) runs pod-side before any "
                     "capability claim."),
            "definition": (f"({REF_FROZEN_ARGMAX} - W7)/({REF_FROZEN_ARGMAX} "
                           f"- {REF_ORACLE}) >= {GATE_FRAC}  <=>  W7 <= "
                           f"{W7_GATE_THRESHOLD} m — the threshold is the "
                           "authoritative form; the fraction is shown at "
                           "float precision"),
            "w7_selected_ade": sel,
            "threshold_m": W7_GATE_THRESHOLD,
            "frac_selgap_closed_vs_frozen_argmax": round(frac_frozen, 6),
            "pass": g_pass,
            **({"verdict_refused_reason":
                "oracle-debug pruning leaked GT into the shortlist — this "
                "run is the perfect-pruning UPPER BOUND, diagnostic only, "
                "never a claimable gate result"} if diagnostic else {}),
        },
        "tougher_read_vs_rescorer_top8_kincost": {
            "note": ("NOT the pre-registered gate — reported per the brief: "
                     "closing half the frozen gap while losing to the "
                     "deployed interim arm (0.4815) would be a pass that "
                     "changes nothing, so both fractions are stated"),
            "reference_arm_selected_ade": REF_RESCORER_TOP8_KINCOST,
            "frac_selgap_closed_vs_0.4815": round(frac_resc, 6),
            "w7_beats_rescorer_arm": bool(sel < REF_RESCORER_TOP8_KINCOST),
        },
        "argmax_inclusion_guarantee": {
            "statement": ("the frozen selector's own pick (out['sel_idx'] — "
                          "its FULL select(), incl. vt-keep) is ALWAYS in "
                          "the shortlist (force_include), so the shortlist's "
                          "ORACLE is never worse than the frozen pick — W7 "
                          "cannot lose by exclusion; the realised pick "
                          "beats the frozen pick only where the cost ranks "
                          "correctly, which is what this gate measures"),
            "frozen_pick_in_shortlist_frac": mini.get(
                "frozen_in_shortlist_frac"),
            "w7_pick_equals_frozen_frac": mini.get("sel_matches_frozen_frac"),
        },
        "prune": prune,
        "cost": cost_cfg,
        "roll": roll,
        "calibration_p7": calib,
        "reference": {
            "frozen_argmax_selected_ade": REF_FROZEN_ARGMAX,
            "rescorer_top8_kincost_selected_ade": REF_RESCORER_TOP8_KINCOST,
            "w4_oracle_ade": REF_ORACLE,
            "w4b_feat_selected_ade": REF_W4B_FEAT,
            "w4b_kin_selected_ade": REF_W4B_KIN,
            "_source": ("w4_gate.json / w4b_gate_feat.json / p7_regrade.json "
                        "— banked 881-grid numbers; the in-run recomputed "
                        "frozen/oracle land in mini_eval as the grid-parity "
                        "cross-check"),
        },
        "mini_eval": mini,
        "tier": "T0",
        "_tier_note": ("T0 — the OUTER eval conditions on logged frames and "
                       "scores against the logged future: a WM diagnostic, "
                       "NEVER driving performance (EVAL_DOCTRINE.md). The "
                       "roll inside is imagination-closed under the "
                       "candidate's OWN actions (the MPC-style path, not GT "
                       "teacher-forcing) — that does not upgrade the outer "
                       "tier; T1 confirmation is pod-side t1_eval."),
        "_estimator_note": ("POINT ESTIMATES over the eval grid (episodes<40, "
                            "stride 8). The decision-grade interval for any "
                            "registry claim is the EPISODE-CLUSTER BOOTSTRAP "
                            "(taniteval.selgap / taniteval/ci.py) on the "
                            "banked per-window arrays (w7_eval_windows.pt) — "
                            "run it before publishing; never "
                            "overlapping_holdout_se."),
        "_evidence_class": "MEASURED (ours; artifact = this JSON)",
    }


# ============================================================================
# the mini-eval — the whole instrument (W7 trains nothing)
# ============================================================================
@torch.no_grad()
def mini_eval(world, grounding, head, tap, st_tap, emission, ds_val, device, *,
              probes, amp_on, w4_cond: str, prune_rule: str, topk: int,
              roll_k: int, w_roll: float, w_kin: float, w_prog: float,
              episodes: int = 40, stride: int = 8, batch: int = 16,
              out_dir: str | None = None) -> dict:
    """W7 selected / frozen / oracle / shortlist-oracle ADE + calibration over
    the eval-default grid (episodes<40, stride 8 -> the 881 grid) — grid-
    comparable with the banked W4/W4b/p7_regrade numbers. Per-window arrays
    banked to ``w7_eval_windows.pt`` for the pod-side cluster bootstrap."""
    from torch.utils.data import default_collate

    from train_flagship_v4 import _to_device
    from train_p8_occupancy import lift_actions3
    from train_v58f_unicycle_head import frozen_forward
    from tanitad.models.metric_dynamics import (decode_transitions,
                                                rollout_transitions)
    grid = [i for i, (e, t) in enumerate(ds_val.index)
            if e < episodes and t % stride == 0]
    if not grid:
        raise SystemExit("[w7] mini-eval selected 0 windows — check "
                         "--episodes/--stride against the val cache")
    step_readout = grounding.step["op"]
    K = len(head.cfg.horizons)
    errs, costs_all, short_all, roll_all, kin_all = [], [], [], [], []
    sel_all_l, frozen_all_l, eids = [], [], []
    fam = {k: 0.0 for k in ("speed_mae", "accel_mae", "heading_mae_rad",
                            "yaw_rate_mae_rads", "curvature_mae_1pm")}
    sums = {"selected": 0.0, "frozen_selected": 0.0, "oracle": 0.0,
            "shortlist_oracle": 0.0, "winner_hit": 0.0, "rank_pct": 0.0,
            "frozen_in_short": 0.0, "sel_eq_frozen": 0.0,
            "winner_in_short": 0.0, "w7_le_frozen": 0.0}
    within_rhos: list[float] = []
    n = 0
    t0 = time.time()
    for b0 in range(0, len(grid), batch):
        idx = grid[b0:b0 + batch]
        b = _to_device(default_collate([ds_val[i] for i in idx]), device)
        b_, w_ = b["frames"].shape[:2]
        st_tap.clear()
        out, emis_feat, v0, tgt = frozen_forward(
            world, head, tap, b, device, probes=probes, amp_on=amp_on,
            cond=w4_cond)
        states = st_tap.states(b_, w_)                       # [B, W, S]
        a_ctl, kappa, fan = emission(emis_feat, v0)
        a_ctl, kappa, fan = a_ctl.float(), kappa.float(), fan.float()
        err = fan_ade(fan, tgt)                              # [B, N]
        # ---- prune + argmax-inclusion --------------------------------------
        sc = out.get("refined_logits")
        if prune_rule == "frozen":
            if sc is None:
                raise SystemExit(
                    "[w7] head emitted no 'refined_logits' — the frozen "
                    "selector's sel_score surface is missing; reference "
                    "pruning impossible (oracle-debug remains a diagnostic, "
                    "not a substitute)")
            sc = sc.detach().float()
        short = shortlist_indices(prune_rule, topk, scores=sc, err=err)
        short = force_include(short, out["sel_idx"])
        p = short.shape[1]
        # ---- the roll: candidate controls as actions -----------------------
        a_s = a_ctl.gather(1, short[:, :, None].expand(-1, -1, K))
        k_s = kappa.gather(1, short[:, :, None].expand(-1, -1, K))
        fan_s = fan.gather(1, short[:, :, None, None].expand(-1, -1, K, 2))
        aw2 = b["actions"].float()
        if aw2.shape[-1] != 2:
            raise SystemExit(f"[w7] batch actions have {aw2.shape[-1]} "
                             "channels, expected the 2-channel (steer, accel)"
                             " corpus format — lift_actions3 appends speed")
        aw2_cf, fa2_cf = candidate_roll_actions(aw2, a_s, k_s, roll_k)
        v0_rep = v0.repeat_interleave(p, dim=0)              # B-major/P-minor
        aw3, fa3 = lift_actions3(aw2_cf, fa2_cf, v0_rep)
        st_rep = states.repeat_interleave(p, dim=0)          # [B*P, W, S]
        dev_type = st_rep.device.type
        with torch.autocast(dev_type, dtype=torch.bfloat16,
                            enabled=amp_on and dev_type == "cuda"):
            trans = rollout_transitions(world.predictor, st_rep,
                                        aw3.to(st_rep.dtype),
                                        fa3.to(st_rep.dtype), roll_k)
            wm_wp, _dpose = decode_transitions(step_readout, trans, roll_k)
        wm_wp = wm_wp.float().reshape(b_, p, roll_k, 2)
        # ---- cost + selection (float32 end to end) -------------------------
        r_ade = roll_consistency(fan_s[:, :, :roll_k], wm_wp)      # [B, P]
        kin = kinematic_cost(a_s)                                  # [B, P]
        arc = progress_arc_length(fan_s)                           # [B, P]
        cost = w7_cost(r_ade, kin, arc, w_roll=w_roll, w_kin=w_kin,
                       w_prog=w_prog)
        sel = select_from_cost(short, cost)                        # [B]
        # ---- accumulate ----------------------------------------------------
        bs = err.shape[0]
        ar = torch.arange(bs, device=err.device)
        e_sel = err[ar, sel]
        e_frozen = err[ar, out["sel_idx"]]
        win = err.argmin(dim=1)
        short_err = err.gather(1, short)                           # [B, P]
        sums["selected"] += float(e_sel.sum())
        sums["frozen_selected"] += float(e_frozen.sum())
        sums["oracle"] += float(err.min(dim=1).values.sum())
        sums["shortlist_oracle"] += float(short_err.min(dim=1).values.sum())
        sums["winner_hit"] += float((sel == win).float().sum())
        sums["rank_pct"] += float(
            ((err < e_sel[:, None]).sum(dim=1).float()
             / max(err.shape[1] - 1, 1)).sum())
        sums["frozen_in_short"] += float(
            (short == out["sel_idx"][:, None]).any(dim=1).float().sum())
        sums["sel_eq_frozen"] += float((sel == out["sel_idx"]).float().sum())
        sums["winner_in_short"] += float(
            (short == win[:, None]).any(dim=1).float().sum())
        sums["w7_le_frozen"] += float((e_sel <= e_frozen + 1e-9).float()
                                      .sum())
        for key, v in selected_family_sums(fan[ar, sel], tgt).items():
            fam[key] += v
        for r in range(bs):                     # within-window calibration
            within_rhos.append(spearman(cost[r].cpu().numpy(),
                                        short_err[r].cpu().numpy()))
        errs.append(err.cpu())
        costs_all.append(cost.cpu())
        short_all.append(short.cpu())
        roll_all.append(r_ade.cpu())
        kin_all.append(kin.cpu())
        sel_all_l.append(sel.cpu())
        frozen_all_l.append(out["sel_idx"].cpu())
        eids.extend(int(ds_val.index[i][0]) for i in idx)
        n += bs
        if (b0 // batch) % 8 == 0:
            print(f"[w7] {min(b0 + batch, len(grid))}/{len(grid)} windows "
                  f"({time.time() - t0:.0f} s)", flush=True)
    err_all = torch.cat(errs)                                # [Nw, N]
    cost_all = torch.cat(costs_all)                          # [Nw, P]
    short_cat = torch.cat(short_all)                         # [Nw, P]
    sel_cat = torch.cat(sel_all_l)                           # [Nw]
    frozen_cat = torch.cat(frozen_all_l)
    eid_t = torch.tensor(eids)
    if out_dir is not None:
        torch.save({"fan_err_ade": err_all, "shortlist": short_cat,
                    "cost": cost_all, "roll_ade": torch.cat(roll_all),
                    "kincost": torch.cat(kin_all), "sel_idx": sel_cat,
                    "frozen_sel_idx": frozen_cat, "eid": eid_t,
                    "_read": ("per-window per-candidate dense ADE of the W4 "
                              "unicycle fan + the W7 shortlist, per-shortlist "
                              "roll-consistency / kinematic / total cost, W7 "
                              "and frozen picks, episode ids — the input to "
                              "the pod-side episode-cluster bootstrap "
                              "(taniteval.selgap / ci.py) and any weight "
                              "re-sweep without a re-roll")},
                   os.path.join(out_dir, "w7_eval_windows.pt"))
    # ---- P7-style calibration: Spearman(cost, realised error) --------------
    ar_all = torch.arange(err_all.shape[0])
    err_sel = err_all[ar_all, sel_cat].numpy()
    cost_sel = cost_all.gather(
        1, (short_cat == sel_cat[:, None]).float().argmax(dim=1)[:, None]
    ).squeeze(1).numpy()
    across = cluster_bootstrap_spearman(cost_sel, err_sel, eids)
    ci = across.get("rho_ci_cluster")
    across["gate_rho_ge_0.3_ci_excl_0"] = (
        None if ci is None else
        bool(across["spearman_rho"] >= P7_GATE_RHO and ci[0] > 0.0))
    wr = np.asarray(within_rhos, dtype=np.float64)
    wr_ok = wr[~np.isnan(wr)]
    calib = {
        "across_windows_cost_vs_realised_error": across,
        "within_window_cost_vs_error_over_shortlist": {
            "rho_mean": (round(float(wr_ok.mean()), 4) if wr_ok.size
                         else None),
            "rho_median": (round(float(np.median(wr_ok)), 4) if wr_ok.size
                           else None),
            "n_windows": int(wr_ok.size),
            "n_dropped_degenerate": int(wr.size - wr_ok.size),
            "note": (f"rank correlation over only {int(short_cat.shape[1])} "
                     "shortlist candidates per window — coarse by "
                     "construction; the across-window block is the P7 "
                     "analogue"),
        },
        "p7_reference": {
            "gate": (f"PRE-REGISTERED (WM_PHYSICS_PROOF P7): rho >= "
                     f"{P7_GATE_RHO} with CI excluding 0"),
            "rescorer_top8_kincost_rho": REF_P7_RHO_RESCORER,
            "rescorer_top8_kincost_ci": [-0.1395, 0.2387],
            "frozen_argmax_rho": REF_P7_RHO_FROZEN,
            "frozen_argmax_ci": [0.0909, 0.4098],
            "read": ("both learned-score arms FAILED P7 (p7_regrade.json) — "
                     "W7's explicit costs should be calibrated where scores "
                     "were not; this block is that check on the same grid"),
        },
    }
    # best-effort selgap CI — stack does not DEPEND on taniteval (its rule).
    selgap_ci: dict | str
    try:
        root = Path(__file__).resolve().parents[2] / "taniteval"
        if str(root) not in sys.path:
            sys.path.append(str(root))
        from taniteval.selgap import selgap as _selgap
        selgap_ci = _selgap(err_all.numpy(), sel_cat.numpy(), eids,
                            scores=None,
                            level="operative_w7_roll_rerank")
    except Exception as ex:                                  # noqa: BLE001
        selgap_ci = (f"taniteval unavailable here ({type(ex).__name__}: {ex})"
                     " — rescore pod-side from w7_eval_windows.pt")
    res = {
        "n_windows": n,
        "n_candidates": int(err_all.shape[1]),
        "shortlist_k": int(short_cat.shape[1]),
        "grid": {"episodes": episodes, "stride": stride, "batch": batch,
                 "expected_n": EXPECTED_GRID_WINDOWS,
                 "matches_banked_grid": n == EXPECTED_GRID_WINDOWS},
        "selected_ade": round(sums["selected"] / n, 6),
        "frozen_selected_ade_in_run": round(sums["frozen_selected"] / n, 6),
        "oracle_ade_in_run": round(sums["oracle"] / n, 6),
        "shortlist_oracle_ade": round(sums["shortlist_oracle"] / n, 6),
        "sel_gap": round((sums["selected"] - sums["oracle"]) / n, 6),
        "winner_hit_frac": round(sums["winner_hit"] / n, 6),
        "sel_rank_pct_mean": round(sums["rank_pct"] / n, 6),
        "frozen_in_shortlist_frac": round(sums["frozen_in_short"] / n, 6),
        "sel_matches_frozen_frac": round(sums["sel_eq_frozen"] / n, 6),
        "winner_in_shortlist_frac": round(sums["winner_in_short"] / n, 6),
        "w7_pick_le_frozen_pick_frac": round(sums["w7_le_frozen"] / n, 6),
        "families": {
            "LONGITUDINAL": {
                "speed_mae_ms": round(fam["speed_mae"] / n, 6),
                "accel_mae_ms2": round(fam["accel_mae"] / n, 6),
                "headway_ttc": ("not computable here: no lead-agent channel "
                                "in this instrument — pod-side taniteval "
                                "harness item, stated per the 2026-08-02 "
                                "rule"),
            },
            "LATERAL": {
                "heading_mae_rad": round(fam["heading_mae_rad"] / n, 6),
                "yaw_rate_mae_rads": round(fam["yaw_rate_mae_rads"] / n, 6),
                "curvature_mae_1pm": round(fam["curvature_mae_1pm"] / n, 6),
                "note": "waypoint-derived adjuncts (atan2 of finite "
                        "differences; noisy near standstill)",
            },
            "TACTICAL": {
                "note": ("re-rank decision quality IS this instrument's "
                         "family: shortlist coverage of the true winner + "
                         "the pick's rank"),
                "winner_hit_frac": round(sums["winner_hit"] / n, 6),
                "winner_in_shortlist_frac":
                    round(sums["winner_in_short"] / n, 6),
                "sel_rank_pct_mean": round(sums["rank_pct"] / n, 6),
            },
            "STRATEGIC": ("n/a: no route/goal label exists on PhysicalAI-AV "
                          "(settled, five probes — CLAUDE.md rule 2); stated "
                          "per the 2026-08-02 rule"),
        },
        "calibration": calib,
        "selgap_ci": selgap_ci,
        "wallclock_s": round(time.time() - t0, 1),
    }
    return res


# ============================================================================
# main (POD-SIDE: needs GPU + the v5f checkpoint + the W4 emission + corpus)
# ============================================================================
def build_args(argv=None):
    ap = argparse.ArgumentParser("w7_roll_rerank", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="v5f checkpoint "
                    "(keys: model, grounding, head[, goal_head])")
    ap.add_argument("--w4-ckpt", required=True,
                    help="trained W4 unicycle_emission.pt (the W4 trainer's "
                         "save format) — loaded FROZEN")
    ap.add_argument("--head-config", default=None,
                    help="run config.json (default: sibling of --ckpt)")
    ap.add_argument("--anchors-dense", default=None,
                    help="trained dense-anchor buffer (pass explicitly)")
    ap.add_argument("--probe-vocab", default=None,
                    help="probe_vocab.pt for cond_imagination heads "
                         "(default: sibling of --ckpt)")
    # corpus (VAL only — W7 trains nothing); the W4-family arg surface
    ap.add_argument("--v2-val-cache", required=True, nargs="+",
                    help="v2 compressed VAL split dir(s)")
    ap.add_argument("--v2-lru", type=int, default=64)
    ap.add_argument("--v2-subframe", default=None, metavar="HxW",
                    help="centred sub-frame the model reads (e.g. 176x624) — "
                         "MUST match the run; cross-checked vs config.json")
    ap.add_argument("--require-parity", action="store_true")
    from tanitad.geometry import add_geometry_args
    add_geometry_args(ap)      # --frame-h/--frame-w/--frame-hfov/--projection
    # W7 knobs (defaults are KNOBS, not measured optima — module docstring)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=TOPK_DEFAULT,
                    help="shortlist size (the L1 '256 -> top-8' prune)")
    ap.add_argument("--prune-rule", choices=PRUNE_RULES, default="frozen",
                    help="'frozen' = the frozen selector's refined_logits "
                         "(reference); 'oracle-debug' = true-error top-K — "
                         "⛔ DIAGNOSTIC ONLY, leaks GT, gate verdict refused")
    ap.add_argument("--roll-k", type=int, default=ROLL_K_DEFAULT,
                    help="WM roll horizon in 0.1 s ticks (10 = 1 s @ 10 Hz)")
    ap.add_argument("--w-roll", type=float, default=W_ROLL_DEFAULT,
                    help="roll-consistency weight (KNOB, not an optimum)")
    ap.add_argument("--w-kin", type=float, default=W_KIN_DEFAULT,
                    help="kinematic-cost weight (KNOB, not an optimum)")
    ap.add_argument("--w-prog", type=float, default=W_PROG_DEFAULT,
                    help="progress (negative arc length) weight — default 0")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    # mini-eval grid (eval defaults — the 881 grid)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=16)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = build_args(argv)
    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[w7] WARNING: cuda unavailable, falling back to cpu",
              flush=True)
        device = "cpu"
    amp_on = (device == "cuda") and not a.no_amp
    os.makedirs(a.out, exist_ok=True)

    from eval_flagship_v4 import (_eval_cfg, _plan, build_v2_val_episodes,
                                  load_v4_from_ck, resolve_eval_frames)
    from flagship_v4_data import FlagshipV4Dataset
    from tanitad.data import parity

    # ---- geometry FIRST, cross-checked against the run's own config.json ----
    cfg = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(
        a, cfg, label="w7_roll_rerank")
    plan = _plan(cfg)
    if plan.max_horizon < a.roll_k - 1:
        raise SystemExit(f"[w7] --roll-k {a.roll_k} needs {a.roll_k - 1} "
                         f"future actions but plan.max_horizon is only "
                         f"{plan.max_horizon}")
    head_cfg_path = a.head_config or str(Path(a.ckpt).parent / "config.json")
    run_cfg = None
    if Path(head_cfg_path).exists():
        try:
            run_cfg = json.loads(Path(head_cfg_path).read_text())
        except Exception as ex:
            print(f"[w7] WARNING: could not parse {head_cfg_path}: {ex}",
                  flush=True)
    frame_check = parity.assert_eval_frame_matches_run(
        run_cfg, model_frame, label="--ckpt vs W7 eval frame",
        cache_frame=cache_frame)
    if not frame_check["checked"]:
        print(f"[w7] ⚠ FRAME UNVERIFIED: {frame_check['note']}", flush=True)

    # ---- frozen v5f: EXACTLY the W4/W4b/W4c loader --------------------------
    print(f"[w7] loading checkpoint {a.ckpt} ...", flush=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and "head" in ck):
        raise SystemExit("[w7] --ckpt has no 'head' key — W7 re-ranks the v4 "
                         "planner head's fan; a plain trunk has no fan.")
    world, grounding, head, base_step, hcfg, goal_head = load_v4_from_ck(
        ck, device,
        head_config_path=(head_cfg_path if Path(head_cfg_path).exists()
                          else None),
        anchors_dense_path=a.anchors_dense, frame=model_frame)
    del ck
    horizons = head.cfg.horizons
    if tuple(horizons) != tuple(range(1, len(horizons) + 1)):
        raise SystemExit(f"[w7] head horizons {horizons} are not contiguous "
                         f"1..K @10 Hz — the W4 fan is defined on the dense "
                         f"tick only.")
    K = len(horizons)
    if K < a.roll_k:
        raise SystemExit(f"[w7] --roll-k {a.roll_k} > fan horizon K={K} — "
                         "the candidate has no controls to roll that far")

    probes = None
    if getattr(head.cfg, "cond_imagination", False):
        pv = Path(a.probe_vocab or (Path(a.ckpt).parent / "probe_vocab.pt"))
        if not pv.exists():
            raise SystemExit(f"[w7] cond_imagination head but no {pv} — a "
                             "silent skip would score a head minus 32 inputs")
        probes = torch.load(pv, map_location=device)
        print(f"[w7] imagination probes: {tuple(probes.shape)}", flush=True)

    # ---- the frozen W4 emission (the fan under re-rank) ---------------------
    feat_dim_q = int(head.decoder.offset_head.in_features)
    emission, w4_cond, w4_meta = load_w4_emission(
        a.w4_ckpt, device, k_expected=K, offset_in_features=feat_dim_q)
    print(f"[w7] W4 emission loaded: cond={w4_cond} "
          f"feat_dim={w4_meta['feat_dim']} K={K} (step {w4_meta['w4_step']}, "
          f"base {w4_meta['w4_base_step']}) — FROZEN", flush=True)

    # ---- frozen means PROVED frozen (nothing trains; still checksummed) -----
    assert not any(p.requires_grad
                   for m in (world, grounding, head, emission)
                   for p in m.parameters())
    md5_before = module_md5(world, head, emission, grounding)
    print(f"[w7] trunk+head+emission+grounding frozen · base step "
          f"{base_step} · md5 {md5_before[:12]}", flush=True)

    # ---- val data (same grid as W4/W4b/eval) --------------------------------
    val_eps, val_prov = build_v2_val_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    ds_val = FlagshipV4Dataset(val_eps, window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    print(f"[w7] val {len(val_eps)} eps / {len(ds_val)} windows", flush=True)

    # ---- taps ---------------------------------------------------------------
    tap = OffsetFeatureTap(head.decoder.offset_head)   # feeds the W4 emission
    st_tap = StateTap(world.readout)                   # encode_window's states
    print(f"[w7] state tap: world.readout output (fourbrain.py:470-478; "
          f"hook, not recompute) · prune={a.prune_rule} topk={a.topk} "
          f"roll_k={a.roll_k} weights=(roll {a.w_roll}, kin {a.w_kin}, "
          f"prog {a.w_prog}) — KNOBS, not optima", flush=True)
    if a.prune_rule == "oracle-debug":
        print("[w7] ⛔ oracle-debug pruning: GT leaks into the shortlist — "
              "this run is the perfect-pruning UPPER BOUND, diagnostic only; "
              "the gate verdict will be REFUSED", flush=True)

    t0 = time.time()
    ev = mini_eval(world, grounding, head, tap, st_tap, emission, ds_val,
                   device, probes=probes, amp_on=amp_on, w4_cond=w4_cond,
                   prune_rule=a.prune_rule, topk=a.topk, roll_k=a.roll_k,
                   w_roll=a.w_roll, w_kin=a.w_kin, w_prog=a.w_prog,
                   episodes=a.episodes, stride=a.stride, batch=a.eval_batch,
                   out_dir=a.out)
    md5_after = module_md5(world, head, emission, grounding)
    if not ev["grid"]["matches_banked_grid"]:
        print(f"[w7] ⚠ grid has {ev['n_windows']} windows, banked references "
              f"are on {EXPECTED_GRID_WINDOWS} — comparisons to "
              f"{REF_FROZEN_ARGMAX}/{REF_RESCORER_TOP8_KINCOST}/{REF_ORACLE} "
              f"are cross-grid; say so wherever quoted", flush=True)

    calib = ev.pop("calibration")
    prune = {
        "rule": a.prune_rule, "topk": a.topk,
        "diagnostic_only": a.prune_rule == "oracle-debug",
        "surface": ("out['refined_logits'] — the frozen selector's "
                    "per-candidate sel_score on the new fan (refc.py:1514; "
                    "the p7_regrade frozen-argmax ranking surface)"
                    if a.prune_rule == "frozen" else
                    "TRUE per-candidate error — ⛔ GT leak, upper bound only"),
        "argmax_inclusion": "force_include(shortlist, out['sel_idx'])",
    }
    cost_cfg = {
        "terms": {
            "roll_consistency": {"weight": a.w_roll,
                                 "def": "ADE(candidate unicycle waypoints, "
                                        "WM-rolled decoded waypoints) over "
                                        "the roll horizon — the WM's veto"},
            "kinematic": {"weight": a.w_kin,
                          "def": "mean|a| + 0.5*mean|jerk| FROM THE CONTROLS "
                                 "(tanitad.models.v58f.kinematic_cost, "
                                 "v58f.py:99-116, imported)"},
            "progress": {"weight": a.w_prog,
                         "def": "MINUS candidate arc length (origin "
                                "prepended); default OFF"},
        },
        "selection": "argmin cost inside the shortlist",
        "_weights_note": ("DEFAULTS ARE KNOBS, NOT MEASURED OPTIMA — a "
                          "weight sweep re-scores from the banked "
                          "w7_eval_windows.pt without a re-roll"),
    }
    roll = {
        "k": a.roll_k, "seconds": a.roll_k * DT,
        "fn": "tanitad.models.metric_dynamics.rollout_transitions "
              "(metric_dynamics.py:247-266)",
        "decode": "decode_transitions(grounding.step['op'], ...) "
                  "(metric_dynamics.py:269-280) — pinned twin of the "
                  "canary's rollout_decode (train_flagship_v4.py:584-586)",
        "action_lift": "train_p8_occupancy.lift_actions3 (train_p8_"
                       "occupancy.py:417-424) — the canary 3-channel "
                       "speed-append; ⚠️ speed channel held at OBSERVED v0 "
                       "for every candidate (the trunk's rollout contract — "
                       "the stage_a stated limitation)",
        "kappa_to_steer": "stage_a_probes.steer_of_kappa — steer_road_rad = "
                          f"atan({WHEELBASE} * kappa) (physicalai.py:621, "
                          "legacy const2p9 regime)",
        "action_placement": "candidate step 0 in the LAST window slot, steps "
                            "1..k-1 as futures; history OBSERVED (stage_a_"
                            "probes.py:44-47, 219-232 convention)",
        "states": "StateTap on world.readout (hook, not recompute — the W4c "
                  "SpatialTokenTap pattern)",
        "imagination_closed": True,
    }
    gate = build_w7_gate(ev, prune=prune, cost_cfg=cost_cfg, calib=calib,
                         roll=roll)
    gate.update({
        "base_ckpt": a.ckpt, "base_step": base_step,
        "w4_ckpt": a.w4_ckpt, "w4_meta": w4_meta,
        "n_trainable": 0,
        "_training_note": "W7 trains NOTHING — a training-free re-rank "
                          "instrument over frozen artifacts",
        "frozen_proof": {"md5_before": md5_before, "md5_after": md5_after,
                         "identical": md5_before == md5_after,
                         "modules": "world + head + W4 emission + grounding"},
        "wall_s": round(time.time() - t0, 1),
    })
    with open(os.path.join(a.out, "w7_gate.json"), "w") as gf:
        json.dump(gate, gf, indent=1)
    tap.remove()
    st_tap.remove()
    print(f"\n[W7 SUMMARY] {json.dumps(gate, indent=1)}", flush=True)
    if not gate["frozen_proof"]["identical"]:
        raise SystemExit("⛔ TRUNK/HEAD/EMISSION/GROUNDING CHANGED DURING THE "
                         "RUN — run invalid")
    g = gate["gate_W7_selgap_closed"]
    t = gate["tougher_read_vs_rescorer_top8_kincost"]
    verdict = ("VERDICT REFUSED (oracle-debug diagnostic)" if g["pass"] is None
               else "PASS" if g["pass"] else "FAIL")
    print(f"[W7 GATE] {verdict} (selected {ev['selected_ade']:.4f} vs "
          f"threshold {W7_GATE_THRESHOLD}; frac closed vs frozen "
          f"{g['frac_selgap_closed_vs_frozen_argmax']:.4f}; tougher read vs "
          f"{REF_RESCORER_TOP8_KINCOST}: "
          f"{t['frac_selgap_closed_vs_0.4815']:.4f}, beats it: "
          f"{t['w7_beats_rescorer_arm']})", flush=True)
    print(f"[W7 CROSS-CHECK] in-run frozen "
          f"{ev['frozen_selected_ade_in_run']:.4f} (banked "
          f"{REF_FROZEN_ARGMAX}) · in-run oracle "
          f"{ev['oracle_ade_in_run']:.4f} (banked {REF_ORACLE}) · shortlist "
          f"oracle {ev['shortlist_oracle_ade']:.4f}", flush=True)
    ac = calib["across_windows_cost_vs_realised_error"]
    print(f"[W7 CALIBRATION] rho {ac['spearman_rho']} CI "
          f"{ac.get('rho_ci_cluster')} (P7 gate >= {P7_GATE_RHO} excl 0; "
          f"score refs: rescorer {REF_P7_RHO_RESCORER}, frozen "
          f"{REF_P7_RHO_FROZEN})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
