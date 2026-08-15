"""W4b — selector recalibration on the FROZEN W4 unicycle fan (v5.8f).

WHY (PREREG_W4B_SELECTOR.md, registered 2026-08-10 BEFORE launch — both
outcomes bound in advance): W4 passed both its gates (oracle 0.1077, accel MAE
0.774, violations 0.0) but the FROZEN v5f selector's pick on the
re-parameterised fan is near-uninformed — selected ADE **0.7933** vs 0.4056
(its own pick on the old fan). Hypothesis: **calibration, not information
loss** — the selection signal is still in the trunk features; only the score
head's mapping to the new fan geometry is stale.

ONE LEVER (attribution): freeze trunk, grounding, goal_head, the W4
:class:`~train_v58f_unicycle_head.UnicycleEmission`, the anchors — EVERYTHING
except a NEW :class:`W4bRescorer`. Selection = argmax of the new logits; no
re-rank, no WM roll (that stays W7's lever — attribution must remain
separable).

TWO PRE-REGISTERED VARIANTS (``--variant``, run separately pod-side):
  * ``feat`` (default): input = the per-candidate offset-head query ``q``
    [B, N, d] — the SAME tensor the W4 emission conditions on, captured by the
    SAME :class:`~train_v58f_unicycle_head.OffsetFeatureTap` (imported, not
    duplicated).
  * ``kin``: additionally the candidate's OWN (a, kappa) control sequence,
    flattened ([B, N, 2K], bound-normalised) — kinematics-aware scoring.

LOSS: the SAME margin ranking loss the tactical/E4.3 code uses —
``tanitad.models.tactical.ranking_loss`` (imported, not re-derived): winner =
the GT-nearest candidate of the NEW fan (per-candidate dense ADE against
``refb_labels.waypoint_targets``), loss = mean over the N-1 losers of
relu(margin + logit_loser − logit_winner). Why not CE: redesign §3.3 — CE on a
one-hot winner punishes every non-winner symmetrically and saturates; ranking
asks only for the ORDER, which is the selection contract.

⛔ PRE-REGISTERED GATES (PREREG_W4B_SELECTOR.md, verbatim — written to
``<out>/w4b_gate.json``):
  * **G1 (recalibration suffices):** selected ADE on the unicycle fan
    **<= 0.45 m** on the same 881-window grid (within ~10 % of the old
    selector-on-old-fan 0.4056, against a better oracle of 0.1077). PASS ⇒
    v5.8f assembly proceeds as W4-fan + recalibrated selector; W7 then attacks
    the remaining sel_gap (0.45 → 0.11 headroom).
  * **G2 (recalibration insufficient):** selected ADE > 0.45 ⇒ the
    per-candidate conditioning does not carry enough selection signal for this
    fan; selector demotes to top-K pruner and W7 (WM-roll re-rank on the clean
    fan) becomes the primary selection mechanism. Supporting measurement,
    reported either way: top-8 oracle on the new fan; if **top-8 oracle
    <= 0.15** the pruner role is viable.

MEASUREMENT CONTRACT: same 881-window grid (episodes<40, stride 8); selected /
oracle / top-{4,8,16} oracle / sel_gap. The numbers in the gate JSON are POINT
ESTIMATES; the decision-grade interval for any registry claim is the
episode-cluster bootstrap (``taniteval.selgap`` / ``taniteval/ci.py``) — the
per-window arrays are banked to ``<out>/w4b_eval_windows.pt`` so the rescore
needs no re-run, and the CI is computed in-process when ``taniteval`` is
importable (best-effort: ``stack`` does not DEPEND on ``taniteval``). Tier
stamp **T0** (diagnostic — this conditions on logged frames; it is not a
driving-performance claim). Four-families note: LONGITUDINAL (speed/accel MAE)
and LATERAL (heading / yaw-rate / curvature error) are reported on the
SELECTED trajectory; TACTICAL = selector rank quality (this instrument's
family); STRATEGIC n/a — no route/goal label on PhysicalAI (stated per the
2026-08-02 rule). Headway/TTC need a lead-agent instrument — pod-side
taniteval harness, stated, not silently dropped.

⛔ FROZEN MEANS PROVED FROZEN (train_unicycle_readout.py contract): trunk +
head + grounding are ``requires_grad_(False)`` (inside
``eval_flagship_v4.load_v4_from_ck``), the W4 emission is loaded and
``requires_grad_(False)``'d here, the optimiser is built over the rescorer's
parameters ONLY, and world+head+emission are md5-checksummed before/after.

⚠️ POD-SIDE ONLY for the full path: this box has no GPU, no v5f checkpoint, no
W4 checkpoint, no v2 corpus. Runnable (and run) here: ``python -m py_compile``
+ the pure-part CPU tests ``stack/tests/test_w4b.py`` (rescorer shapes/params,
ranking-loss sanity, gate-JSON logic both branches, kin input-dim handling).

Usage (pod5; PYTHONPATH=/workspace/TanitAD/stack required or trainers die with
ModuleNotFound: tanitad):

  python3 train_w4b_selector.py \
      --ckpt /workspace/experiments/flagship-v5f-.../ckpt_step30000.pt \
      --w4-ckpt /workspace/experiments/w4-unicycle-head/unicycle_emission.pt \
      --anchors-dense /workspace/experiments/anchors/anchors_dense_1to20.pt \
      --v2-cache  /workspace/data/physicalai-train-e438721ae894-w120-256x640cyl \
      --v2-val-cache /workspace/data/physicalai-val-0c5f7dac3b11-w120-256x640cyl \
      --frame-h 256 --frame-w 640 --frame-hfov 120 --projection cylindrical \
      --v2-subframe 176x624 --variant feat \
      --out /workspace/experiments/w4b-selector-feat

(second arm: ``--variant kin --out /workspace/experiments/w4b-selector-kin``)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
# stack root too, so `import tanitad` resolves even without the pod-side
# PYTHONPATH (harmless when PYTHONPATH is set — same directory).
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))
from train_v58f_unicycle_head import (A_MAX, DT, KAPPA_MAX,  # noqa: E402
                                      OffsetFeatureTap, UnicycleEmission,
                                      module_md5, speeds_and_accels)
from tanitad.models.tactical import ranking_loss  # noqa: E402  (torch-only)

# ---- the pre-registered constants (PREREG_W4B_SELECTOR.md, verbatim) --------
GATE_SELECTED_ADE = 0.45          # G1: selected ADE on the unicycle fan (m)
GATE_TOP8_PRUNER = 0.15           # G2 branch: pruner viable iff top-8 <= this
REF_FROZEN_SELECTOR_NEW_FAN = 0.7933   # frozen v5f selector on the W4 fan
REF_OLD_SELECTOR_OLD_FAN = 0.4056      # frozen selector on its own old fan
REF_W4_ORACLE = 0.1077                 # W4 fan oracle ADE (registry §1.13)
TOPK = (4, 8, 16)                 # prereg top-k oracle sizes
EXPECTED_GRID_WINDOWS = 881       # the banked v5f/W4 eval grid size


# ============================================================================
# the ONLY trainable module of W4b
# ============================================================================
class W4bRescorer(nn.Module):
    """Per-candidate score head over the frozen W4 unicycle fan.

    ``forward(q [B, N, F](, a_ctl [B, N, K], kappa [B, N, K])) -> logits
    [B, N]`` (N = 256 candidates on the flagship fan). Inputs per variant:

    * ``feat``:  the offset-head query ``q`` alone — the SAME per-candidate
      tensor the W4 emission conditions on (OffsetFeatureTap; refc.py
      :1193-1198). Extra (a, kappa) args are accepted and IGNORED so the call
      site is variant-uniform.
    * ``kin``:   ``q`` concatenated with the candidate's own (a, kappa)
      sequence flattened — bound-normalised (a/A_MAX, kappa/KAPPA_MAX) so both
      control channels enter at O(1) scale. ``in_dim = F + 2K``.

    MLP: 2 layers, hidden 256 (the W4/prereg module scale). The FINAL layer is
    ZERO-INIT: at step 0 every candidate scores 0 — a defined, uninformed
    uniform warm start (argmax degenerates to index 0), not noise.
    """

    def __init__(self, feat_dim: int, k: int = 20, hidden: int = 256,
                 variant: str = "feat"):
        super().__init__()
        if variant not in ("feat", "kin"):
            raise ValueError(f"variant must be 'feat' or 'kin', got {variant!r}")
        self.feat_dim, self.k, self.variant = int(feat_dim), int(k), variant
        self.in_dim = self.feat_dim + (2 * self.k if variant == "kin" else 0)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)     # uniform warm start
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, q: Tensor, a_ctl: Tensor | None = None,
                kappa: Tensor | None = None) -> Tensor:
        if q.ndim != 3 or q.shape[-1] != self.feat_dim:
            raise ValueError(f"q must be [B, N, {self.feat_dim}], got "
                             f"{tuple(q.shape)}")
        if self.variant == "kin":
            if a_ctl is None or kappa is None:
                raise ValueError("variant='kin' requires a_ctl and kappa")
            if a_ctl.shape != q.shape[:2] + (self.k,) or \
                    kappa.shape != q.shape[:2] + (self.k,):
                raise ValueError(
                    f"a_ctl/kappa must be [B, N, {self.k}], got "
                    f"{tuple(a_ctl.shape)} / {tuple(kappa.shape)}")
            x = torch.cat([q, a_ctl / A_MAX, kappa / KAPPA_MAX], dim=-1)
        else:
            x = q                               # (a, kappa) ignored by design
        return self.net(x).squeeze(-1)          # [B, N]


# ============================================================================
# pure helpers (CPU-testable — tests/test_w4b.py)
# ============================================================================
def fan_ade(fan: Tensor, tgt: Tensor) -> Tensor:
    """Per-candidate dense ADE: ``(fan [B, N, K, 2], tgt [B, K, 2]) ->
    [B, N]`` — mean over the K dense horizons of the Euclidean waypoint error.
    The SAME per-candidate error the W4 trainer's winner selection and the
    banked oracle/selected numbers are computed from."""
    if fan.ndim != 4 or tgt.ndim != 3 or fan.shape[-1] != 2:
        raise ValueError(f"bad shapes fan {tuple(fan.shape)} tgt "
                         f"{tuple(tgt.shape)}")
    return (fan - tgt[:, None]).norm(dim=-1).mean(dim=-1)


def topk_oracle_per_window(err: Tensor, scores: Tensor, k: int) -> Tensor:
    """``(err [B, N], scores [B, N], k) -> [B]``: per window, the min error
    among the k candidates the SCORER ranks highest ("what would an oracle
    re-ranker recover if the selector shortlisted k instead of committing to
    1" — the taniteval.selgap ``selector_scores`` ranking, computed here
    per-batch so it accumulates without holding the corpus). k >= N clips to N
    (== full oracle); monotone non-increasing in k by construction."""
    if err.shape != scores.shape or err.ndim != 2:
        raise ValueError(f"err/scores must be matching [B, N], got "
                         f"{tuple(err.shape)} vs {tuple(scores.shape)}")
    kk = min(int(k), err.shape[1])
    top = scores.topk(kk, dim=1).indices                     # [B, kk]
    return err.gather(1, top).min(dim=1).values              # [B]


def build_w4b_gate(mini: dict, *, variant: str) -> dict:
    """The pre-registered W4b gate record from a mini-eval dict (pure:
    JSON-in, JSON-out; both branches pinned by tests/test_w4b.py).

    G1 and G2 are the PREREG_W4B_SELECTOR.md gates verbatim, with both
    consequences bound in advance. Every number here is a POINT ESTIMATE over
    the eval grid; the registry row carries the episode-cluster bootstrap CI
    (pod-side rescore from the banked per-window arrays)."""
    sel = mini["selected_ade"]
    top8 = mini["oracle_topk"]["8"]
    g1_pass = bool(sel <= GATE_SELECTED_ADE)
    return {
        "item": ("W4b — selector recalibration on the frozen unicycle fan "
                 "(PREREG_W4B_SELECTOR.md, registered 2026-08-10 pre-launch)"),
        "variant": variant,
        "gate_G1_recalibration_suffices": {
            "rule": f"selected ADE on the unicycle fan <= {GATE_SELECTED_ADE}"
                    " m on the same 881-window grid",
            "selected_ade": sel,
            "threshold_m": GATE_SELECTED_ADE,
            "pass": g1_pass,
            "consequence_if_pass": (
                "v5.8f assembly proceeds as W4-fan + recalibrated selector; "
                "W7 then attacks the remaining sel_gap (0.45 -> 0.11 "
                "headroom)"),
        },
        "gate_G2_recalibration_insufficient": {
            "rule": (f"selected ADE > {GATE_SELECTED_ADE} => selector demotes "
                     "to top-K pruner and W7 (WM-roll re-rank on the clean "
                     "fan) becomes the primary selection mechanism"),
            "engaged": not g1_pass,
            "top8_oracle": top8,
            "pruner_viable": bool(top8 <= GATE_TOP8_PRUNER),
            "pruner_threshold_m": GATE_TOP8_PRUNER,
            "note": "top-8 oracle reported either way, per the prereg",
        },
        "reference": {
            "frozen_selector_selected_ade_new_fan": REF_FROZEN_SELECTOR_NEW_FAN,
            "old_selector_selected_ade_old_fan": REF_OLD_SELECTOR_OLD_FAN,
            "w4_oracle_ade_new_fan": REF_W4_ORACLE,
        },
        "mini_eval": mini,
        "tier": "T0",
        "_tier_note": ("T0 teacher-forced diagnostic — conditioned on logged "
                       "frames; NEVER quotable as driving performance "
                       "(EVAL_DOCTRINE.md)"),
        "_estimator_note": ("POINT ESTIMATES over the eval grid (episodes<40, "
                            "stride 8). The decision-grade interval for any "
                            "registry claim is the EPISODE-CLUSTER BOOTSTRAP "
                            "(taniteval.selgap / taniteval/ci.py) on the "
                            "banked per-window arrays (w4b_eval_windows.pt) — "
                            "run it before publishing; never "
                            "overlapping_holdout_se."),
        "_evidence_class": "MEASURED (ours; artifact = this JSON)",
    }


def selected_family_sums(wp_sel: Tensor, tgt: Tensor,
                         dt: float = DT) -> dict[str, float]:
    """Batch SUMS of the LONGITUDINAL / LATERAL adjunct errors on the selected
    trajectory (caller divides by n): speed/accel MAE (longitudinal), heading /
    yaw-rate / curvature MAE (lateral), all waypoint-derived in float32 with
    the origin prepended (the speeds_and_accels geometry). Heading via atan2 of
    finite differences — noisy near standstill; an ADJUNCT, stated as such."""
    wp_sel, tgt = wp_sel.float(), tgt.float()
    bs = wp_sel.shape[0]
    sp_p, ac_p = speeds_and_accels(wp_sel, dt)
    sp_g, ac_g = speeds_and_accels(tgt, dt)
    z = wp_sel.new_zeros(bs, 1, 2)
    dp = torch.diff(torch.cat([z, wp_sel], dim=1), dim=1)    # [B, K, 2]
    dg = torch.diff(torch.cat([z, tgt], dim=1), dim=1)
    psi_p = torch.atan2(dp[..., 1], dp[..., 0])
    psi_g = torch.atan2(dg[..., 1], dg[..., 0])
    dpsi = torch.remainder(psi_p - psi_g + math.pi, 2 * math.pi) - math.pi
    yr_p = torch.diff(psi_p, dim=1) / dt
    yr_g = torch.diff(psi_g, dim=1) / dt
    ds_p = dp.norm(dim=-1).clamp_min(1e-3)
    ds_g = dg.norm(dim=-1).clamp_min(1e-3)
    kap_p = torch.diff(psi_p, dim=1) / ds_p[:, 1:]
    kap_g = torch.diff(psi_g, dim=1) / ds_g[:, 1:]
    return {
        "speed_mae": float((sp_p - sp_g).abs().mean()) * bs,
        "accel_mae": float((ac_p - ac_g).abs().mean()) * bs,
        "heading_mae_rad": float(dpsi.abs().mean()) * bs,
        "yaw_rate_mae_rads": float((yr_p - yr_g).abs().mean()) * bs,
        "curvature_mae_1pm": float((kap_p - kap_g).abs().mean()) * bs,
    }


# ============================================================================
# W4 emission loading (the save format of train_v58f_unicycle_head.py)
# ============================================================================
def load_w4_emission(path: str, device, k_expected: int,
                     offset_in_features: int) -> tuple[UnicycleEmission, str,
                                                       dict]:
    """Load + FREEZE the trained W4 ``unicycle_emission.pt``.

    The W4 trainer saves ``{"emission": state_dict, "cond_mode", "feat_dim",
    "k", "step", "args", "base_ckpt", "base_step"}`` — read back verbatim,
    cross-checked against the loaded head (K and, for cond='feature', the
    offset-head width) so a mismatched checkpoint fails loudly, not as a
    silent shape error mid-train."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("emission", "cond_mode", "feat_dim", "k"):
        if key not in ck:
            raise SystemExit(f"[w4b] --w4-ckpt {path} has no '{key}' key — "
                             "not a W4 unicycle_emission.pt")
    if int(ck["k"]) != int(k_expected):
        raise SystemExit(f"[w4b] W4 emission K={ck['k']} != head horizons "
                         f"K={k_expected}")
    cond = str(ck["cond_mode"])
    if cond == "feature" and int(ck["feat_dim"]) != int(offset_in_features):
        raise SystemExit(f"[w4b] W4 emission feat_dim={ck['feat_dim']} != "
                         f"offset_head.in_features={offset_in_features} — "
                         "wrong trunk for this emission")
    emission = UnicycleEmission(feat_dim=int(ck["feat_dim"]),
                                k=int(ck["k"])).to(device)
    emission.load_state_dict(ck["emission"])
    emission.eval()
    emission.requires_grad_(False)
    meta = {"w4_step": ck.get("step"), "w4_base_ckpt": ck.get("base_ckpt"),
            "w4_base_step": ck.get("base_step"), "cond_mode": cond,
            "feat_dim": int(ck["feat_dim"]), "k": int(ck["k"])}
    return emission, cond, meta


# ============================================================================
# end-of-run mini-eval — SAME grid rule as W4/eval (episodes<40, stride 8)
# ============================================================================
@torch.no_grad()
def mini_eval(world, head, tap, emission, rescorer, ds_val, device, *,
              probes, amp_on, w4_cond: str, episodes: int = 40,
              stride: int = 8, batch: int = 16,
              out_dir: str | None = None) -> dict:
    """selected / oracle / top-{4,8,16} oracle / sel_gap of the NEW logits on
    the NEW (W4 unicycle) fan over the eval-default grid ``e < episodes and
    t % stride == 0`` — grid-comparable with the banked W4/v5f numbers.
    Per-window arrays are banked (``w4b_eval_windows.pt``) for the pod-side
    episode-cluster bootstrap; the taniteval.selgap CI is computed in-process
    when the package is importable (best-effort, recorded either way)."""
    from torch.utils.data import default_collate
    from train_flagship_v4 import _to_device
    from train_v58f_unicycle_head import frozen_forward
    rescorer.eval()
    grid = [i for i, (e, t) in enumerate(ds_val.index)
            if e < episodes and t % stride == 0]
    if not grid:
        raise SystemExit("[w4b] mini-eval selected 0 windows — check "
                         "--episodes/--stride against the val cache")
    errs, logits_all, eids = [], [], []
    fam = {k: 0.0 for k in ("speed_mae", "accel_mae", "heading_mae_rad",
                            "yaw_rate_mae_rads", "curvature_mae_1pm")}
    sums = {"selected": 0.0, "oracle": 0.0, "frozen_selected": 0.0,
            "winner_hit": 0.0, "rank_pct": 0.0}
    sums.update({f"top{k}": 0.0 for k in TOPK})
    n = 0
    t0 = time.time()
    for b0 in range(0, len(grid), batch):
        idx = grid[b0:b0 + batch]
        b = _to_device(default_collate([ds_val[i] for i in idx]), device)
        out, emis_feat, v0, tgt = frozen_forward(
            world, head, tap, b, device, probes=probes, amp_on=amp_on,
            cond=w4_cond)
        q = tap.last().detach().float()
        a_ctl, kappa, fan = emission(emis_feat, v0)
        logits = rescorer(q, a_ctl, kappa)                   # [B, N]
        err = fan_ade(fan.float(), tgt)                      # [B, N]
        bs = err.shape[0]
        ar = torch.arange(bs, device=err.device)
        sel = logits.argmax(dim=1)
        win = err.argmin(dim=1)
        e_sel = err[ar, sel]
        sums["selected"] += float(e_sel.sum())
        sums["oracle"] += float(err.min(dim=1).values.sum())
        sums["frozen_selected"] += float(err[ar, out["sel_idx"]].sum())
        sums["winner_hit"] += float((sel == win).float().sum())
        sums["rank_pct"] += float(
            ((err < e_sel[:, None]).sum(dim=1).float()
             / max(err.shape[1] - 1, 1)).sum())
        for k in TOPK:
            sums[f"top{k}"] += float(
                topk_oracle_per_window(err, logits, k).sum())
        for key, v in selected_family_sums(fan[ar, sel], tgt).items():
            fam[key] += v
        errs.append(err.cpu())
        logits_all.append(logits.cpu())
        eids.extend(int(ds_val.index[i][0]) for i in idx)
        n += bs
    rescorer.train()
    err_all = torch.cat(errs)                                # [Nw, N]
    log_all = torch.cat(logits_all)
    sel_all = log_all.argmax(dim=1)
    if out_dir is not None:
        torch.save({"fan_err_ade": err_all, "logits": log_all,
                    "sel_idx": sel_all, "eid": torch.tensor(eids),
                    "_read": ("per-window per-candidate dense ADE of the W4 "
                              "unicycle fan + W4b rescorer logits over the "
                              "eval grid — the input to the pod-side "
                              "episode-cluster bootstrap (taniteval.selgap)")},
                   os.path.join(out_dir, "w4b_eval_windows.pt"))
    # best-effort selgap CI — stack does not DEPEND on taniteval (its rule).
    selgap_ci: dict | str
    try:
        root = Path(__file__).resolve().parents[2] / "taniteval"
        if str(root) not in sys.path:
            sys.path.append(str(root))
        from taniteval.selgap import selgap as _selgap
        selgap_ci = _selgap(err_all.numpy(), sel_all.numpy(), eids,
                            scores=log_all.numpy(),
                            level="operative_w4b_rescorer")
    except Exception as ex:                                  # noqa: BLE001
        selgap_ci = (f"taniteval unavailable here ({type(ex).__name__}: {ex})"
                     " — rescore pod-side from w4b_eval_windows.pt")
    res = {
        "n_windows": n,
        "n_candidates": int(err_all.shape[1]),
        "grid": {"episodes": episodes, "stride": stride, "batch": batch,
                 "expected_n": EXPECTED_GRID_WINDOWS,
                 "matches_banked_grid": n == EXPECTED_GRID_WINDOWS},
        "selected_ade": round(sums["selected"] / n, 6),
        "oracle_ade": round(sums["oracle"] / n, 6),
        "sel_gap": round((sums["selected"] - sums["oracle"]) / n, 6),
        "oracle_topk": {str(k): round(sums[f"top{k}"] / n, 6) for k in TOPK},
        "frozen_selected_ade": round(sums["frozen_selected"] / n, 6),
        "winner_hit_frac": round(sums["winner_hit"] / n, 6),
        "sel_rank_pct_mean": round(sums["rank_pct"] / n, 6),
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
                "note": "selector rank quality IS this instrument's family",
                "winner_hit_frac": round(sums["winner_hit"] / n, 6),
                "sel_rank_pct_mean": round(sums["rank_pct"] / n, 6),
            },
            "STRATEGIC": ("n/a: no route/goal label exists on PhysicalAI-AV "
                          "(settled, five probes — CLAUDE.md rule 2); stated "
                          "per the 2026-08-02 rule"),
        },
        "selgap_ci": selgap_ci,
        "wallclock_s": round(time.time() - t0, 1),
    }
    return res


# ============================================================================
# main (POD-SIDE: needs GPU + the v5f checkpoint + the W4 emission + corpora)
# ============================================================================
def build_args(argv=None):
    ap = argparse.ArgumentParser("train_w4b_selector", description=__doc__,
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
    # corpus (v2 compressed only — the ONLY format v5 has); same seams as W4
    ap.add_argument("--v2-cache", required=True, nargs="+",
                    help="v2 compressed TRAIN split dir(s) — the canonical "
                         "physicalai-train-e438721ae894 build")
    ap.add_argument("--v2-val-cache", required=True, nargs="+",
                    help="v2 compressed VAL split dir(s)")
    ap.add_argument("--v2-lru", type=int, default=64)
    ap.add_argument("--v2-subframe", default=None, metavar="HxW",
                    help="centred sub-frame the model reads (e.g. 176x624) — "
                         "MUST match the run; cross-checked vs config.json")
    ap.add_argument("--require-parity", action="store_true")
    from tanitad.geometry import add_geometry_args
    add_geometry_args(ap)      # --frame-h/--frame-w/--frame-hfov/--projection
    # training
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2000,
                    help="prereg budget: ~2000 steps, <= 2 h pod5")
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--eps-per-batch", type=int, default=4,
                    help="episode-grouped sampling (the MooseFS I/O shape)")
    ap.add_argument("--save-every", type=int, default=500,
                    help="checkpoint + metrics.json cadence")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--variant", choices=("feat", "kin"), default="feat",
                    help="'feat' = offset-head query q only; 'kin' = q + the "
                         "candidate's own (a, kappa) flattened (the SECOND "
                         "pre-registered variant) — run separately pod-side")
    ap.add_argument("--margin", type=float, default=0.1,
                    help="ranking margin (tanitad.models.tactical default)")
    # mini-eval grid (eval defaults — the 881 grid)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--eval-batch", type=int, default=16)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = build_args(argv)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    rng = random.Random(a.seed)
    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[w4b] WARNING: cuda unavailable, falling back to cpu",
              flush=True)
        device = "cpu"
    amp_on = (device == "cuda") and not a.no_amp
    os.makedirs(a.out, exist_ok=True)

    from torch.utils.data import default_collate

    from eval_flagship_v4 import (_eval_cfg, _plan, build_v2_val_episodes,
                                  load_v4_from_ck, resolve_eval_frames)
    from flagship_v4_data import FlagshipV4Dataset
    from tanitad.data import parity
    from train_flagship_v4 import _to_device
    from train_v58f_unicycle_head import (build_train_episodes,
                                          frozen_forward, make_sampler)

    # ---- geometry FIRST, cross-checked against the run's own config.json ----
    cfg = _eval_cfg()
    cache_frame, model_frame = resolve_eval_frames(
        a, cfg, label="train_w4b_selector")
    plan = _plan(cfg)
    head_cfg_path = a.head_config or str(Path(a.ckpt).parent / "config.json")
    run_cfg = None
    if Path(head_cfg_path).exists():
        try:
            run_cfg = json.loads(Path(head_cfg_path).read_text())
        except Exception as ex:
            print(f"[w4b] WARNING: could not parse {head_cfg_path}: {ex}",
                  flush=True)
    frame_check = parity.assert_eval_frame_matches_run(
        run_cfg, model_frame, label="--ckpt vs W4b train frame",
        cache_frame=cache_frame)
    if not frame_check["checked"]:
        print(f"[w4b] ⚠ FRAME UNVERIFIED: {frame_check['note']}", flush=True)

    # ---- frozen v5f: EXACTLY the W4 trainer's loader ------------------------
    print(f"[w4b] loading checkpoint {a.ckpt} ...", flush=True)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and "head" in ck):
        raise SystemExit("[w4b] --ckpt has no 'head' key — W4b rescores the "
                         "v4 planner head's fan; a plain trunk has no fan.")
    world, grounding, head, base_step, hcfg, goal_head = load_v4_from_ck(
        ck, device,
        head_config_path=(head_cfg_path if Path(head_cfg_path).exists()
                          else None),
        anchors_dense_path=a.anchors_dense, frame=model_frame)
    del ck
    horizons = head.cfg.horizons
    if tuple(horizons) != tuple(range(1, len(horizons) + 1)):
        raise SystemExit(f"[w4b] head horizons {horizons} are not contiguous "
                         f"1..K @10 Hz — the W4 fan is defined on the dense "
                         f"tick only.")
    K = len(horizons)

    probes = None
    if getattr(head.cfg, "cond_imagination", False):
        pv = Path(a.probe_vocab or (Path(a.ckpt).parent / "probe_vocab.pt"))
        if not pv.exists():
            raise SystemExit(f"[w4b] cond_imagination head but no {pv} — a "
                             "silent skip would score a head minus 32 inputs")
        probes = torch.load(pv, map_location=device)
        print(f"[w4b] imagination probes: {tuple(probes.shape)}", flush=True)

    # ---- the frozen W4 emission (the fan under rescoring) -------------------
    feat_dim_q = int(head.decoder.offset_head.in_features)
    emission, w4_cond, w4_meta = load_w4_emission(
        a.w4_ckpt, device, k_expected=K, offset_in_features=feat_dim_q)
    print(f"[w4b] W4 emission loaded: cond={w4_cond} "
          f"feat_dim={w4_meta['feat_dim']} K={K} (step {w4_meta['w4_step']}, "
          f"base {w4_meta['w4_base_step']}) — FROZEN", flush=True)

    # ---- frozen means PROVED frozen (three locks, W4 contract) --------------
    assert not any(p.requires_grad
                   for m in (world, grounding, head, emission)
                   for p in m.parameters())
    md5_before = module_md5(world, head, emission)
    print(f"[w4b] trunk+head+emission frozen · base step {base_step} · "
          f"md5 {md5_before[:12]}", flush=True)

    # ---- data (same grid as W4/eval) ----------------------------------------
    train_eps, train_prov = build_train_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    val_eps, val_prov = build_v2_val_episodes(
        a, cache_frame=cache_frame, train_frame=model_frame)
    ds_train = FlagshipV4Dataset(train_eps, window=cfg.predictor.window,
                                 max_horizon=plan.max_horizon,
                                 maneuver_h=plan.maneuver_h,
                                 channels=cfg.encoder.in_channels)
    ds_val = FlagshipV4Dataset(val_eps, window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    print(f"[w4b] train {len(train_eps)} eps / {len(ds_train)} windows; "
          f"val {len(val_eps)} eps / {len(ds_val)} windows", flush=True)
    sample = make_sampler(ds_train, a.eps_per_batch, rng)

    # ---- the ONLY trainable module ------------------------------------------
    tap = OffsetFeatureTap(head.decoder.offset_head)
    rescorer = W4bRescorer(feat_dim=feat_dim_q, k=K,
                           variant=a.variant).to(device)
    n_par = sum(p.numel() for p in rescorer.parameters())
    print(f"[w4b] W4bRescorer variant={a.variant} feat_dim={feat_dim_q} "
          f"in_dim={rescorer.in_dim} K={K} ({n_par / 1e6:.3f} M trainable; "
          f"frozen everything else)", flush=True)
    opt = torch.optim.AdamW(rescorer.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)

    log_path = os.path.join(a.out, "train_log.jsonl")
    fh = open(log_path, "a")
    fh.write(json.dumps({
        "run": "w4b-selector-rescorer", "args": vars(a),
        "base_ckpt": a.ckpt, "base_step": base_step,
        "w4_ckpt": a.w4_ckpt, "w4_meta": w4_meta,
        "trunk_head_emission_md5": md5_before, "n_trainable": n_par,
        "variant": a.variant, "in_dim": rescorer.in_dim, "horizons_K": K,
        "train_parity": {"n_dirs": len(a.v2_cache)},
        "_evidence_class": "MEASURED (ours; artifact = this log)"}) + "\n")
    fh.flush()

    history: list[dict] = []
    acc = {"loss": 0.0, "selected_ade": 0.0, "oracle_ade": 0.0,
           "winner_hit": 0.0, "n": 0}
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = sample(a.bs)
        b = _to_device(default_collate([ds_train[i] for i in idx]), device)
        out, emis_feat, v0, tgt = frozen_forward(
            world, head, tap, b, device, probes=probes, amp_on=amp_on,
            cond=w4_cond)
        with torch.no_grad():
            a_ctl, kappa, fan = emission(emis_feat, v0)
            err = fan_ade(fan.float(), tgt)                  # [B, N] targets
        q = tap.last().detach().float()
        # rescorer in float32 OUTSIDE autocast — same numerics the gate is
        # evaluated at (the W4 discipline).
        logits = rescorer(q, a_ctl, kappa)
        loss = ranking_loss(logits, err, a.margin)           # THE tactical loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(rescorer.parameters(), 5.0)
        opt.step()
        sched.step()

        bs = err.shape[0]
        ar = torch.arange(bs, device=err.device)
        with torch.no_grad():
            sel = logits.argmax(dim=1)
            acc["loss"] += float(loss.detach()) * bs
            acc["selected_ade"] += float(err[ar, sel].sum())
            acc["oracle_ade"] += float(err.min(dim=1).values.sum())
            acc["winner_hit"] += float((sel == err.argmin(dim=1)).float()
                                       .sum())
            acc["n"] += bs

        if step % a.log_every == 0:
            rec = {"step": step, "loss": round(float(loss.detach()), 5),
                   "selected_ade": round(float(err[ar, sel].mean()), 4),
                   "oracle_ade": round(float(err.min(dim=1).values.mean()),
                                       4),
                   "gnorm": round(float(gnorm), 3),
                   "lr": sched.get_last_lr()[0],
                   "elapsed_s": round(time.time() - t0, 1)}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"[{step}] {rec}", flush=True)

        if step % a.save_every == 0:
            n_ = max(acc["n"], 1)
            row = {"step": step,
                   **{k: round(v / n_, 5) for k, v in acc.items()
                      if k != "n"},
                   "elapsed_s": round(time.time() - t0, 1)}
            history.append(row)
            acc = {k: 0.0 for k in acc} | {"n": 0}
            with open(os.path.join(a.out, "metrics.json"), "w") as mf:
                json.dump({"history": history, "args": vars(a),
                           "base_step": base_step, "w4_meta": w4_meta,
                           "_read": "rows are TRAIN-batch running means over "
                                    "the last save window; the gate numbers "
                                    "are the held-out mini-eval in "
                                    "w4b_gate.json",
                           "_evidence_class": "MEASURED (ours)"}, mf,
                          indent=1)
            torch.save({"rescorer": rescorer.state_dict(),
                        "variant": a.variant, "feat_dim": feat_dim_q,
                        "in_dim": rescorer.in_dim, "k": K, "step": step,
                        "args": vars(a), "base_ckpt": a.ckpt,
                        "base_step": base_step, "w4_ckpt": a.w4_ckpt,
                        "w4_meta": w4_meta},
                       os.path.join(a.out, "w4b_rescorer.pt"))
            fh.write(json.dumps({"per500": row}) + "\n")
            fh.flush()
            print(f"[w4b @{step}] {row}", flush=True)

    # ---- frozen proof + the pre-registered gates ----------------------------
    md5_after = module_md5(world, head, emission)
    ev = mini_eval(world, head, tap, emission, rescorer, ds_val, device,
                   probes=probes, amp_on=amp_on, w4_cond=w4_cond,
                   episodes=a.episodes, stride=a.stride, batch=a.eval_batch,
                   out_dir=a.out)
    if not ev["grid"]["matches_banked_grid"]:
        print(f"[w4b] ⚠ grid has {ev['n_windows']} windows, banked references "
              f"are on {EXPECTED_GRID_WINDOWS} — comparisons to 0.7933/"
              f"0.4056/0.1077 are cross-grid; say so wherever quoted",
              flush=True)
    gate = build_w4b_gate(ev, variant=a.variant)
    gate.update({
        "steps": a.steps, "base_ckpt": a.ckpt, "base_step": base_step,
        "w4_ckpt": a.w4_ckpt, "w4_meta": w4_meta,
        "n_trainable": n_par, "in_dim": rescorer.in_dim, "margin": a.margin,
        "frozen_proof": {"md5_before": md5_before, "md5_after": md5_after,
                         "identical": md5_before == md5_after,
                         "modules": "world + head + W4 emission"},
        "wall_s": round(time.time() - t0, 1),
    })
    with open(os.path.join(a.out, "w4b_gate.json"), "w") as gf:
        json.dump(gate, gf, indent=1)
    fh.write(json.dumps({"summary": gate}) + "\n")
    fh.close()
    tap.remove()
    print(f"\n[W4B SUMMARY] {json.dumps(gate, indent=1)}", flush=True)
    if not gate["frozen_proof"]["identical"]:
        raise SystemExit("⛔ TRUNK/HEAD/EMISSION CHANGED DURING TRAINING — "
                         "run invalid")
    g1 = gate["gate_G1_recalibration_suffices"]
    g2 = gate["gate_G2_recalibration_insufficient"]
    print(f"[W4B GATE G1] {'PASS' if g1['pass'] else 'FAIL'} "
          f"(selected {ev['selected_ade']:.4f} vs {GATE_SELECTED_ADE}; "
          f"frozen-selector ref {REF_FROZEN_SELECTOR_NEW_FAN}, "
          f"old-fan ref {REF_OLD_SELECTOR_OLD_FAN}, "
          f"oracle {ev['oracle_ade']:.4f} vs ref {REF_W4_ORACLE})",
          flush=True)
    print(f"[W4B GATE G2] engaged={g2['engaged']} top8_oracle "
          f"{g2['top8_oracle']:.4f} pruner_viable={g2['pruner_viable']} "
          f"(<= {GATE_TOP8_PRUNER})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
