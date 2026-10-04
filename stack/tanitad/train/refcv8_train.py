"""refcv8 WP-B -- the TRAINER side of the tactical conditioning (DESIGN.md sec. 3; the model side is
`tanitad/refs/refcv8_conditioning.py` + the seams in `refc.py` / `refc_v3.py` / `refcv6_tactical.py`).

Four things live here so the 11k-line trainer only CALLS them:

  1. :func:`r8_before_forward` -- the per-batch inputs the model needs BEFORE its forward: the GT hypothesis and
     the constraint targets (labels may use ego, PI 2026-08-03), the scheduled-sampling coins (teacher forcing of
     the GT constraint into the allocated candidates' generation), and the route-checkpoint / nav-argument inputs
     with their TRAINING treatment (MM binding 2026-10-04 + WP-A INTEGRATION.md sec. 3.2): route-checkpoint dropout
     >= 0.3 AND the registered along / lateral noise on the kept rows (the clean point leaks, E2'); nav-argument
     dropout with the token KEPT (a leaderboard-legal NavSim agent receives the bare command only).
  2. :func:`r8_losses` -- constraint-head loss, matched L1 on the allocated candidates, L_sat, the listwise
     selection loss (X1, replacing the single-winner E9 CE when on), the Hydra-style sub-score BCE (X1h).
  3. The v9 label release (WP-A, ``tanitad/data/v9_labels.py``): :class:`R8LabelJoin` (the per-window join on the
     trainer's OWN clock), :func:`v9_integrity_census` (the release contract the trainer relies on, checked at load),
     :func:`v9_goal_census` (the goal pos_weight / class mask FROM THE v9 TRAIN WINDOWS), :func:`partial_label_loss`
     (the ``-log sum_{c in allowed} p_c`` loss the v9 partial labels need), :func:`navsim_legal_nav_cmd`.
  4. LABEL-STATE ISOLATION (D1 F2 / X4). With WP-A's ``v7_labels`` fix the goal-negative policy TRAVELS on the label
     objects (``V7Label.goal_geometry_tokens``) and :func:`v7_policy_travels` is True, so the trainer installs no
     scope at all. :class:`V7PolicyScope` is the fallback for an UNFIXED module only (snapshot at load, applied
     around each target computation).

⛔ Every random draw here uses the decoder's DEDICATED generator (`decoder.r8_gen`), never the global RNG.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from tanitad.refs import refcv8_conditioning as r8c

__all__ = ["r8_before_forward", "r8_losses", "V7PolicyScope", "R8LabelJoin", "tf_ratio", "RC_DROPOUT_MIN",
           "RC_NOISE_ALONG_M", "RC_NOISE_LAT_M", "V9_RELEASE_MD5", "GEOMETRY_GOALS", "scale_nav", "scale_rc",
           "rc_training_noise", "v9_integrity_census", "v9_goal_census", "partial_label_loss",
           "navsim_legal_nav_cmd", "v7_policy_travels", "load_v9_join"]

#: MM binding 2026-10-04 (i): route-checkpoint dropout >= 0.3
RC_DROPOUT_MIN = 0.3
#: WP-A INTEGRATION.md sec. 3.2 / 4 (E2' CONFIRMED at exactly these): the training noise on a KEPT route checkpoint,
#: in the route-tangent frame. ⛔ The clean point carries +0.021 more future-speed information than the road-level
#: route (FAILS E2'); the noised one -0.035 (passes). Smaller values are not certified and are refused.
RC_NOISE_ALONG_M = 2.0
RC_NOISE_LAT_M = 0.75
#: the released files (WP-A LANDING_READY_WPA.txt ## WPA-S3); another release must be named by md5 explicitly
V9_RELEASE_MD5 = {"train": "f63ece410b725febb8a5242cf2b01d3c", "eval139": "6b5c7f207cffc3b7eb3cd527fd433599"}
#: v9 SPEC sec. 4.1: the GEOMETRY goals (dense, from the ego log). A reversing window carries none of them; the VLM
#: goals (sec. 4.3) keep their own timing rule.
GEOMETRY_GOALS = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "STOP_POINT",
                  "SPEED_BAND")
#: v9 / INTEGRATION sec. 8: absence claims need the full 8-s band (V7). Frozen v7 ids.
ABSENCE_LAT_V7 = (0,)                 # LANE_KEEP
ABSENCE_LON_V7 = (1, 4, 5)            # CRUISE (= v9 KEEP), CREEP, HOLD
LC_LAT_V7 = (1, 2)                    # LANE_CHANGE_L / _R -- never an exact class in this release
H_ABS_MIN_S = 8.0
#: model input widths (`refc_v3.R8_NAV_DIMS` / `R8_RC_DIMS` must agree; pinned in the tests)
NAV_IN_DIMS = 6                       # log1p(d_next/50), log1p(d_end/50), dyaw/90, side, log1p(lookahead/50), known
RC_IN_DIMS = 4                        # x/50, y/50, psi/90, valid  (NOW frame, RC-A50 by default)
NAV_D_CLIP_M = 1000.0
#: INTEGRATION.md sec. 5 (Master Mind 2026-10-04): NavSim `driving_command` (left, forward, right, unknown) -> our
#: nav_cmd, with nav_args_known = 0 and route_cp_valid = 0 on the LEGAL row.
NAVSIM_LEGAL_NAV_CMD = {"left": 1, "forward": 0, "right": 2, "unknown": 0}


def tf_ratio(step: int, steps: int, start: float, end: float) -> float:
    """Scheduled sampling (1506.03099 p4, the LINEAR form): the teacher-forcing ratio at ``step``."""
    if steps <= 1:
        return float(end)
    a = min(max(step / float(steps - 1), 0.0), 1.0)
    return float(start + (end - start) * a)


def _gen(model):
    g = getattr(model.core.decoder, "r8_gen", None)
    if g is None:
        raise ValueError("[refcv8] the decoder carries no dedicated generator -- the seams are not built")
    return g


# ================================================================================================================= #
# input scaling and the training treatment of the two supplied inputs                                               #
# ================================================================================================================= #
def scale_nav(raw: Tensor, known: Tensor) -> Tensor:
    """v9 ``nav_args`` [.., 5] = (d_next m, d_end m, dyaw_next deg, side +-1, lookahead m) + known -> the model's
    [.., 6]. Distances: ``log1p(min(d, 1000) / 50)`` (~d/50 near, compressed far: the plan horizon is <= ~180 m while
    the release carries up to 3,162 m). Unknown rows are exactly zeros next to a 0 (the model re-applies the bit)."""
    raw = raw.to(torch.float32)
    k = known.to(torch.float32).reshape(*raw.shape[:-1], 1)

    def lg(d):
        return torch.log1p(d.clamp(0.0, NAV_D_CLIP_M) / 50.0)
    out = torch.stack([lg(raw[..., 0]), lg(raw[..., 1]), raw[..., 2] / 90.0, raw[..., 3].clamp(-1.0, 1.0),
                       lg(raw[..., 4])], -1)
    return torch.cat([out * k, k], -1)


def scale_rc(raw: Tensor, valid: Tensor) -> Tensor:
    """v9 ``route_cp`` [.., 3] = (x m, y m, psi deg) in the NOW frame + valid -> the model's [.., 4]."""
    raw = raw.to(torch.float32)
    v = valid.to(torch.float32).reshape(*raw.shape[:-1], 1)
    out = torch.stack([raw[..., 0] / 50.0, raw[..., 1] / 50.0, raw[..., 2] / 90.0], -1)
    return torch.cat([out * v, v], -1)


def rc_training_noise(raw: Tensor, valid: Tensor, gen, *, sigma_along_m: float = RC_NOISE_ALONG_M,
                      sigma_lat_m: float = RC_NOISE_LAT_M) -> Tensor:
    """The registered training noise on the KEPT checkpoints: along-track N(0, sigma_along), lateral N(0, sigma_lat)
    in the ROUTE-TANGENT frame (the tangent is the checkpoint's own heading psi). psi itself is not noised. Invalid
    rows are returned unchanged (they are zeros and stay zeros). ``gen`` is the decoder's ``R8Generator``."""
    raw = raw.to(torch.float32).clone()
    b = raw.shape[0]
    n = gen.randn((b, 2), raw.device) * torch.tensor([sigma_along_m, sigma_lat_m], device=raw.device)
    psi = torch.deg2rad(raw[:, 2])
    c, s = torch.cos(psi), torch.sin(psi)
    dx = n[:, 0] * c - n[:, 1] * s
    dy = n[:, 0] * s + n[:, 1] * c
    v = valid.to(torch.bool).reshape(b)
    raw[:, 0] = torch.where(v, raw[:, 0] + dx, raw[:, 0])
    raw[:, 1] = torch.where(v, raw[:, 1] + dy, raw[:, 1])
    return raw


def navsim_legal_nav_cmd(driving_command) -> int:
    """NavSim's 4-dim one-hot ``driving_command`` (left, forward, right, unknown) -> our ``nav_cmd`` (the LEGAL row:
    nav_args_known = 0, route_cp_valid = 0). ⛔ Refuses anything that is not exactly one-hot."""
    v = [float(x) for x in driving_command]
    if len(v) != 4 or sorted(v) != [0.0, 0.0, 0.0, 1.0]:
        raise ValueError(f"[refcv8] NavSim driving_command must be a 4-dim one-hot, got {v}")
    return NAVSIM_LEGAL_NAV_CMD[("left", "forward", "right", "unknown")[v.index(1.0)]]


def r8_before_forward(model, batch: Mapping[str, Tensor], device, traj_tgt: Tensor, slot_valid: Tensor,
                      pose_last: Tensor, fut_ext: Tensor, fut_valid: Tensor, v0: Tensor) -> dict:
    """Set the model's one-shot TEACHER and return the extra forward inputs + the label tensors the losses need.

    Returns {"fwd": {"r8_nav": ..., "r8_rc": ...}, "gt_lat3", "gt_lon6", "gt_hyp", "cons_t": dict, "lat_t", ...}.
    The GT hypothesis is the TAG of the GT plan (the same tagger the candidates are tagged with), so "the candidate
    belongs to the GT hypothesis" is decided by one rule, not two.

    Inputs come from the v9 join as RAW values (``r8_nav_raw`` / ``r8_nav_known``, ``r8_rc_raw`` / ``r8_rc_valid``)
    and are scaled HERE, after the training treatment, so the noise is applied in metres:
    * TRAINING: nav args dropped with p ``model._r8_nav_args_dropout`` (known -> 0, args -> 0, the token in
      ``nav_cmd`` untouched); RC dropped with p ``model._r8_rc_dropout`` (>= 0.3, refused below), then the registered
      noise on the kept rows (sigmas on ``model._r8_rc_noise``, refused below RC_NOISE_*).
    * EVAL: no dropout, no noise; ``model._r8_legal_row`` True -> the NavSim-LEGAL row (known 0, RC invalid).
    * ``model._r8_no_rc`` True (``--r8-no-rc``): the checkpoint is never fed (the switch-off the MM asked for)."""
    cfg = model.cfg.refcv8
    gen = _gen(model)
    b = traj_tgt.shape[0]
    dev = traj_tgt.device
    slot_t = [float(h) * 0.1 for h in model.cfg.core.trajectory.horizons]
    gl, go = r8c.tag_paths(traj_tgt[:, None], v0, slot_t)
    gl, go = gl[:, 0], go[:, 0]
    ok = slot_valid[:, -1].to(torch.bool)
    gl = torch.where(ok, gl, torch.full_like(gl, -1))
    go = torch.where(ok, go, torch.full_like(go, -1))
    hyp = torch.where(ok, r8c.joint_id(gl.clamp_min(0), go.clamp_min(0)), torch.full_like(gl, -1))
    ct = r8c.constraint_targets(traj_tgt, slot_valid, pose_last, fut_ext, fut_valid)
    lat_t, lat_v, lon_t, lon_v = r8c.normalise_constraints(ct)
    cons = torch.cat([torch.nan_to_num(lat_t), torch.nan_to_num(lon_t)], -1)
    tf = gen.rand((b,), dev) < float(getattr(model, "_r8_tf_ratio", cfg.tf_start))
    if model.training:
        model.set_r8_teacher(hyp, cons, tf)
    fwd: dict[str, Any] = {}
    legal = (not model.training) and bool(getattr(model, "_r8_legal_row", False))
    if "r8_nav_raw" in batch:
        nav = batch["r8_nav_raw"].to(device).to(torch.float32)
        known = batch["r8_nav_known"].to(device).to(torch.bool).reshape(b).clone()
        if model.training and float(getattr(model, "_r8_nav_args_dropout", 0.0)) > 0.0:
            known &= ~(gen.rand((b,), dev) < float(model._r8_nav_args_dropout))   # the TOKEN is untouched
        if legal:
            known = torch.zeros_like(known)
        fwd["r8_nav"] = scale_nav(nav, known)
    if "r8_rc_raw" in batch and not bool(getattr(model, "_r8_no_rc", False)):
        rc = batch["r8_rc_raw"].to(device).to(torch.float32)
        valid = batch["r8_rc_valid"].to(device).to(torch.bool).reshape(b).clone()
        if model.training:
            p = float(getattr(model, "_r8_rc_dropout", RC_DROPOUT_MIN))
            if p < RC_DROPOUT_MIN:
                raise SystemExit(f"[refcv8] route-checkpoint dropout {p} < {RC_DROPOUT_MIN} (MM binding 2026-10-04)")
            sa, sl = getattr(model, "_r8_rc_noise", (RC_NOISE_ALONG_M, RC_NOISE_LAT_M))
            if float(sa) < RC_NOISE_ALONG_M or float(sl) < RC_NOISE_LAT_M:
                raise SystemExit(f"[refcv8] route-checkpoint training noise ({sa}, {sl}) m is below the CERTIFIED "
                                 f"({RC_NOISE_ALONG_M}, {RC_NOISE_LAT_M}) m -- the clean point leaks (WP-A E2')")
            valid &= ~(gen.rand((b,), dev) < p)
            rc = rc_training_noise(rc, valid, gen, sigma_along_m=float(sa), sigma_lat_m=float(sl))
        if legal:
            valid = torch.zeros_like(valid)
        fwd["r8_rc"] = scale_rc(rc, valid)
    gl_v7 = torch.where(gl >= 0, torch.as_tensor(r8c.LAT3_V7_IDS, device=dev)[gl.clamp_min(0)],
                        torch.full_like(gl, -100))
    go_v7 = torch.where(go >= 0, torch.as_tensor(r8c.LON6_V7_IDS, device=dev)[go.clamp_min(0)],
                        torch.full_like(go, -100))
    return {"fwd": fwd, "gt_lat3": gl, "gt_lon6": go, "gt_hyp": hyp, "gt_lat_v7": gl_v7, "gt_lon_v7": go_v7,
            "cons_t": ct, "lat_t": lat_t, "lat_v": lat_v, "lon_t": lon_t, "lon_v": lon_v, "tf": tf}


def r8_losses(model, out: Mapping[str, Tensor], prep: Mapping[str, Any], traj_tgt: Tensor,
              slot_valid: Tensor) -> tuple[Tensor, dict]:
    """The refcv8 loss terms (weights from `cfg.refcv8`). Returns (weighted sum, telemetry).

    ⛔ No term is guarded behind `if w > 0` for a BUILT head: a zero weight multiplies (the 42-of-138 lesson), except
    the listwise / sub-score terms, which are computed only when their head / weight exists (no head = no tensor)."""
    cfg = model.cfg.refcv8
    dev = traj_tgt.device
    total = traj_tgt.new_zeros(())
    tele: dict[str, Any] = {}
    # (1) constraint heads -- the GT-active query only
    if "r8_cons_lat" in out:
        lc, ltel = r8c.constraint_head_loss(out["r8_cons_lat"], out["r8_cons_lon"], prep["gt_lat_v7"],
                                            prep["gt_lon_v7"], prep["lat_t"], prep["lat_v"], prep["lon_t"],
                                            prep["lon_v"])
        total = total + float(cfg.w_cons) * lc
        tele.update(r8_cons=lc.detach(), **ltel)
    if "r8_alloc" not in out:
        return total, tele
    fan = out["anchor_traj"]
    alloc = out["r8_alloc"].to(torch.bool)
    sv = slot_valid.to(fan.dtype)
    err = ((fan - traj_tgt[:, None]).norm(dim=-1) * sv[:, None]).sum(-1) / sv.sum(-1, keepdim=True).clamp_min(1.0)
    # (2) matched L1 on the best allocated candidate OF THE GT HYPOTHESIS (WTA, MTR/TNT-style hard assignment)
    if bool(alloc.any()) and "r8_hyp_alloc" in out:
        nb = int(out["r8_n_base"])
        m = out["r8_hyp_alloc"].shape[1]
        gt_h = prep["gt_hyp"]
        is_gt = (out["r8_hyp_alloc"] == gt_h[:, None]) & (gt_h[:, None] >= 0)        # [B, M]
        e_a = err[:, nb:nb + m].masked_fill(~is_gt, float("inf"))
        has = torch.isfinite(e_a).any(-1)
        if bool(has.any()):
            j = e_a.argmin(-1)
            ar = torch.arange(fan.shape[0], device=dev)
            rec = fan[ar, nb + j]
            l1 = (((rec - traj_tgt).abs().sum(-1)) * sv).sum(-1) / (sv.sum(-1) * 2).clamp_min(1.0)
            la = l1[has].mean()
        else:
            la = fan.sum() * 0.0
        total = total + float(cfg.w_alloc_l1) * la
        tele.update(r8_alloc_l1=la.detach(), r8_alloc_gt_rows=int(has.sum().item()))
        # (3) L_sat on the allocated candidates, against the constraint each was GENERATED under (phi's fields)
        phi = out["r8_phi"]
        c_gen = r8c.denorm_cons(phi[..., 11:15])
        cv = phi[..., 15]
        ls = r8c.constraint_satisfaction_loss(fan, c_gen, cv, alloc)
        total = total + float(cfg.w_sat) * ls
        tele["r8_sat"] = ls.detach()
    # (4) X1 listwise selection over the reach survivors (incl. the allocated candidates)
    if float(cfg.w_listwise) > 0.0:
        keep = out.get("reach_keep")
        keep = torch.ones_like(err, dtype=torch.bool) if keep is None else keep.to(torch.bool)
        ll = r8c.listwise_selection_loss(out["sel_score_v3"], fan.detach(), traj_tgt, slot_valid, keep,
                                         t=float(cfg.list_t), speed_scale_m=float(cfg.list_speed_scale_m),
                                         dir_scale_deg=float(cfg.list_dir_scale_deg))
        total = total + float(cfg.w_listwise) * ll
        tele["r8_listwise"] = ll.detach()
    # (5) X1h sub-score critics
    if "r8_sub_logits" in out:
        y, w = r8c.subscore_targets(fan.detach(), traj_tgt, slot_valid, out["r8_lat3"], prep["gt_lat3"])
        bce = F.binary_cross_entropy_with_logits(out["r8_sub_logits"].float(), y, reduction="none")
        lsb = (bce * w).sum() / w.sum().clamp_min(1.0)
        total = total + float(cfg.w_subscore) * lsb
        tele["r8_subscore"] = lsb.detach()
    tele["r8_emit_frac"] = out["r8_emit_keep"].float().mean().detach()
    return total, tele


# ================================================================================================================= #
# v9: the partial-label loss                                                                                        #
# ================================================================================================================= #
def partial_label_loss(logits: Tensor, allowed: Tensor) -> tuple[Tensor, int]:
    """``mean_rows -log sum_{c in allowed} softmax(z)_c`` over the rows with at least one allowed class (v9
    INTEGRATION sec. 3.1). An exact label is a ONE-bit mask, so on those rows this IS cross-entropy; an all-false
    row contributes nothing. Returns (loss, n rows). ⛔ Never guarded: no row -> an attached zero."""
    z = logits.float()
    al = allowed.to(torch.bool)
    has = al.any(-1)
    lse_all = torch.logsumexp(z, -1)
    lse_al = torch.logsumexp(z.masked_fill(~al, float("-inf")), -1)
    per = torch.where(has, lse_all - lse_al, torch.zeros_like(lse_all))
    n = int(has.sum().item())
    return per.sum() / float(max(n, 1)), n


def _exact_ce(logits: Tensor, target: Tensor, ignore_index: int) -> Tensor:
    """EXACTLY `refcv6_tactical.tactical_behaviour_losses._ce` with no class weight (the trainer passes none)."""
    t = target.reshape(-1).long()
    per = F.cross_entropy(logits, t.clamp_min(0), reduction="none")
    keep = (t != ignore_index).to(per.dtype)
    return (per * keep).sum() / keep.sum().clamp_min(1.0)


def v9_partial_correction(lat_logits: Tensor, lon_logits: Tensor, lat_t: Tensor, lon_t: Tensor, lat_allowed: Tensor,
                          lon_allowed: Tensor, weights, *, ignore_index: int = -100) -> tuple[Tensor, dict]:
    """The term that turns the tactical loss's exact-row CE into the v9 COMBINED partial-label mean, inside the SAME
    weights: ``w_lat * (PL_all - CE_exact) + w_lon * (...)``. Added to ``tactical_behaviour_losses``' total (which
    holds ``w_lat * CE_exact``), the lat term becomes ``w_lat * PL_all`` -- the mean of ``-log sum_allowed p`` over
    every row with an allowed class (an exact row's mask is its one bit, so those rows are CE). Nothing is inflated:
    the budget split of ``TacticalLossWeights`` is unchanged."""
    pl_lat, n_lat = partial_label_loss(lat_logits, lat_allowed)
    pl_lon, n_lon = partial_label_loss(lon_logits, lon_allowed)
    ce_lat = _exact_ce(lat_logits, lat_t, ignore_index)
    ce_lon = _exact_ce(lon_logits, lon_t, ignore_index)
    corr = float(weights.lat_ce) * (pl_lat - ce_lat) + float(weights.lon_ce) * (pl_lon - ce_lon)
    n_ex_lat = int((lat_t.reshape(-1) != ignore_index).sum().item())
    n_ex_lon = int((lon_t.reshape(-1) != ignore_index).sum().item())
    return corr, {"r8v9_lat_pl": pl_lat.detach(), "r8v9_lon_pl": pl_lon.detach(),
                  "r8v9_n_lat_partial": float(n_lat - n_ex_lat), "r8v9_n_lon_partial": float(n_lon - n_ex_lon),
                  "r8v9_n_lat_supervised": float(n_lat), "r8v9_n_lon_supervised": float(n_lon)}


# ================================================================================================================= #
# v9: the release contract, the join, the goal census                                                              #
# ================================================================================================================= #
def _v9():
    try:
        from tanitad.data import v9_labels as V9
    except ImportError as e:            # pragma: no cover - exercised only on a tree without WP-A's reader
        raise SystemExit("[refcv8] --r8-v9-labels needs WP-A's reader `tanitad/data/v9_labels.py`, which is not in "
                         f"this tree ({e})") from None
    return V9


def _bits(a, i: int):
    return ((np.asarray(a).astype(np.int64) >> int(i)) & 1).astype(bool)


def v9_integrity_census(rel, lat_variant: str = "a") -> dict:
    """The release properties this trainer RELIES ON (MM item 4 / INTEGRATION sec. 8), counted over every row.
    Every ``violations`` entry must be 0 -- :func:`load_v9_join` refuses otherwise. Derived from the raw fields, not
    from the reader's decoding, so it cross-checks the builder rather than restating it."""
    V9 = _v9()
    R = rel.rows
    lat = np.asarray(R[f"lat_v7id_{lat_variant}"]).astype(np.int64)
    lat_al = np.asarray(R[f"lat_allowed_v7_{lat_variant}"]).astype(np.int64)
    lon = np.asarray(R["lon_v7id"]).astype(np.int64)
    lon_al = np.asarray(R["lon_allowed_v7"]).astype(np.int64)
    gy = np.asarray(R["goal_y"]).astype(np.int64)
    gw = np.asarray(R["goal_w"]).astype(np.int64)
    rev = np.asarray(R["reversing"]).astype(np.int64) == 1
    short = np.asarray(R["h_obs_s"]).astype(np.float64) < H_ABS_MIN_S - 1e-6
    gi = {t: V9.GOAL22.index(t) for t in V9.GOAL22}
    geo_w = np.zeros(len(gw), bool)
    for t in GEOMETRY_GOALS:
        geo_w |= _bits(gw, gi[t])

    def neg(t):
        return _bits(gw, gi[t]) & ~_bits(gy, gi[t])

    def pos(t):
        return _bits(gw, gi[t]) & _bits(gy, gi[t])
    ex_lat, ex_lon = lat >= 0, lon >= 0
    one = np.left_shift(1, np.clip(lat, 0, 30))
    one_lon = np.left_shift(1, np.clip(lon, 0, 30))
    viol = {
        "lc_exact_action": int(np.isin(lat, LC_LAT_V7).sum()),
        "reversing_with_action_label": int((rev & ((lat_al > 0) | (lon_al > 0))).sum()),
        "reversing_with_geometry_goal": int((rev & geo_w).sum()),
        "absence_class_on_short_band": int((short & (np.isin(lat, ABSENCE_LAT_V7) | np.isin(lon, ABSENCE_LON_V7))
                                            ).sum()),
        # a TURN_x negative ENTAILED by the other side's positive is a presence claim, not an absence claim
        # (MEASURED on the release: every short-band TURN_L negative sits on a TURN_R positive, 11 / 11 eval139)
        "absence_goal_on_short_band": int((short & (pos("FOLLOW_LANE") | (neg("TURN_L") & ~pos("TURN_R"))
                                                    | (neg("TURN_R") & ~pos("TURN_L")) | neg("STOP_POINT"))).sum()),
        "exact_label_mask_mismatch": int((ex_lat & (lat_al != one)).sum() + (ex_lon & (lon_al != one_lon)).sum()),
        "speed_band_supervised": int(_bits(gw, gi["SPEED_BAND"]).sum()),
    }
    return {"n_rows": int(len(lat)), "violations": viol,
            "n_reversing": int(rev.sum()), "n_short_band": int(short.sum()),
            "lat_partial": int(((lat < 0) & (lat_al > 0)).sum()), "lon_partial": int(((lon < 0) & (lon_al > 0)).sum()),
            "lc_goal_positive_vlm": int((pos("LANE_CHANGE_L") | pos("LANE_CHANGE_R")).sum())}


class R8LabelJoin:
    """The v9 release joined to the trainer's windows. Key ``(sid, k)`` with ``k = t + w - 1 + raw_offset`` (= t + 9
    at w 8, n_stack 3) and the trainer's ``now_s`` CHECKED against the release clock (``row_for_now`` refuses a
    mismatch > 1e-6 s). The release is an immutable, picklable ``V9Release`` held on the INSTANCE; no module state.

    ``item`` returns RAW inputs (scaled in :func:`r8_before_forward`, after the training treatment) and the targets in
    the frozen v7 ids (D-WPA-2). ⛔ ``nav_t_next_s`` / ``nav_token_ttime`` / ``rcH_*`` are never read (the reader's
    ``window_inputs`` does not expose them)."""

    def __init__(self, release, *, rc_variant: str = "A50", lat_variant: str = "a", census: dict | None = None):
        V9 = _v9()
        if rc_variant not in V9.RC_VARIANTS:
            raise SystemExit(f"[refcv8] --r8-rc-variant {rc_variant!r} not in {V9.RC_VARIANTS}")
        if lat_variant not in ("a", "b"):
            raise SystemExit(f"[refcv8] --r8-v9-lat-variant {lat_variant!r} must be 'a' or 'b'")
        self.rel = release
        self.rc_variant = str(rc_variant)
        self.lat_variant = str(lat_variant)
        man = release.manifest()
        self.manifest = {"md5": release.md5, "split": release.split, "path": release.path,
                         "schema": man.get("schema"), "base_commit": man.get("base_commit"),
                         "builder_md5": man.get("builder_md5"), "rc_variant": self.rc_variant,
                         "lat_variant": self.lat_variant, "census": census}

    def row(self, sid: int, k: int, now_s: float) -> int:
        return _v9().row_for_now(self.rel, int(sid), int(k), float(now_s))

    def item(self, sid: int, k: int, now_s: float) -> dict[str, Tensor]:
        V9 = _v9()
        r = V9.row_for_now(self.rel, int(sid), int(k), float(now_s))
        inp = V9.window_inputs(self.rel, r, rc_variant=self.rc_variant)
        tg = V9.window_targets(self.rel, r, lat_variant=self.lat_variant, ids="v7")
        return {"r8_nav_token": torch.tensor(int(inp["nav_token"]), dtype=torch.long),
                "r8_nav_raw": torch.as_tensor(np.asarray(inp["nav_args"], np.float32)),
                "r8_nav_known": torch.tensor(bool(inp["nav_args_valid"])),
                "r8_rc_raw": torch.as_tensor(np.asarray(inp["rc"], np.float32)),
                "r8_rc_valid": torch.tensor(bool(inp["rc_valid"])),
                "v9_lat": torch.tensor(int(tg["lat"]), dtype=torch.long),
                "v9_lat_allowed": torch.as_tensor(np.asarray(tg["lat_allowed"], bool)),
                "v9_lon": torch.tensor(int(tg["lon"]), dtype=torch.long),
                "v9_lon_allowed": torch.as_tensor(np.asarray(tg["lon_allowed"], bool)),
                "v9_goal_y": torch.as_tensor(np.asarray(tg["goal_y"], np.float32)),
                "v9_goal_w": torch.as_tensor(np.asarray(tg["goal_w"], np.float32))}


def load_v9_join(npz_path: str, *, expect_md5: str, rc_variant: str = "A50", lat_variant: str = "a") -> R8LabelJoin:
    """Load + md5-check + contract census; REFUSES a release that violates any property the trainer relies on."""
    V9 = _v9()
    try:
        rel = V9.load_v9_release(str(npz_path), expect_md5=str(expect_md5))
    except V9.V9LabelError as e:
        raise SystemExit(str(e)) from None
    census = v9_integrity_census(rel, lat_variant)
    bad = {k: v for k, v in census["violations"].items() if v}
    if bad:
        raise SystemExit(f"[refcv8] ⛔ the v9 release {rel.path} violates the contract the trainer relies on: "
                         f"{bad} (v9 INTEGRATION sec. 8 / MM item 4)")
    return R8LabelJoin(rel, rc_variant=rc_variant, lat_variant=lat_variant, census=census)


def v9_goal_census(join: R8LabelJoin, rows) -> dict:
    """The 22-token goal census over the release ROWS the dataset's windows join to -- the v9 analogue of
    ``v7_labels.goal_supervision_census`` (which counts per CLIP record). Same keys, so ``tac_goal_head.mask_report``
    and the pos_weight rule read it unchanged."""
    V9 = _v9()
    rows = np.asarray(rows, dtype=np.int64)
    gy = np.asarray(join.rel.rows["goal_y"]).astype(np.int64)[rows]
    gw = np.asarray(join.rel.rows["goal_w"]).astype(np.int64)[rows]
    out = {}
    for i, tok in enumerate(V9.GOAL22):
        w = _bits(gw, i)
        y = _bits(gy, i)
        p, n = int((w & y).sum()), int((w & ~y).sum())
        out[tok] = {"pos": p, "neg": n, "ignored": int((~w).sum()), "prevalence": p / max(len(rows), 1),
                    "provenance": ["v9"], "supervised_negative": n > 0, "negatives_policy": "v9-release",
                    "entailed_false_by": []}
    return out


def v9_goal_pos_weight(census: dict, tokens, cap: float) -> list:
    """``n_neg / n_pos`` capped at ``cap``; 0.0 for a class with no positive (it must be masked too)."""
    out = []
    for t in tokens:
        p, n = int(census[t]["pos"]), int(census[t]["neg"])
        out.append(0.0 if p == 0 else float(min(n / p, cap)))
    return out


# ================================================================================================================= #
# label-state isolation                                                                                             #
# ================================================================================================================= #
def v7_policy_travels(v7l_module) -> bool:
    """True when ``v7_labels`` carries the goal-negative policy ON the label objects (WP-A's fix, base 86f0c46e ->
    abb1f64c): then the trainer needs no scope, and must not install one."""
    fields = getattr(getattr(v7l_module, "V7Label", None), "__dataclass_fields__", {}) or {}
    mfields = getattr(getattr(v7l_module, "LabelManifest", None), "__dataclass_fields__", {}) or {}
    return "goal_geometry_tokens" in fields and "goal_geometry_tokens" in mfields


def v7_scope_for(v7l_module):
    """What the trainer installs on a dataset right after a label load: ``None`` when the policy travels with the
    labels (the fixed module), a :class:`V7PolicyScope` snapshot otherwise (the unfixed tip module)."""
    return None if v7_policy_travels(v7l_module) else V7PolicyScope(v7l_module)


class V7PolicyScope:
    """The v7 negative policy of ONE loaded split, snapshotted at construction and applied around each target
    computation -- the FALLBACK for an unfixed ``v7_labels`` whose policy is module state (whatever was loaded LAST:
    the eval blob, in the trainer's order). The snapshot travels with the pickled dataset into a worker."""

    _FIELDS = ("_MEASURED_GEOMETRY_TOKENS", "_MEASURED_COT_TOKENS")

    def __init__(self, v7l_module):
        self.module_name = v7l_module.__name__
        self.snapshot = {f: frozenset(getattr(v7l_module, f)) for f in self._FIELDS}

    @contextmanager
    def applied(self):
        import importlib
        mod = importlib.import_module(self.module_name)
        prev = {f: getattr(mod, f) for f in self._FIELDS}
        try:
            for f, v in self.snapshot.items():
                setattr(mod, f, v)
            yield
        finally:
            for f, v in prev.items():
                setattr(mod, f, v)

    def to_dict(self) -> dict:
        return {f: sorted(v) for f, v in self.snapshot.items()}


# ================================================================================================================= #
# G-DVB: every refcv8 trainer flag, declared against what the BUILT model holds (the SPEC_REFCV7 sec. 2 rule)       #
# ================================================================================================================= #
def _register_gdvb() -> None:
    from tanitad.train import declared_vs_built as dvb

    def _r8(m):
        return getattr(dvb._dec(m), "r8_cfg", None)

    def _c_enable(m, a):
        want = bool(dvb._a(a, "refcv8", False))
        out = dvb._eq("refcv8", want, bool(getattr(m, "r8_enabled", False)), "model.r8_enabled")
        out += dvb._eq("refcv8", want, _r8(m) is not None, "core.decoder.r8_cfg is not None")
        if want:
            td = getattr(m, "tac_decoder_v6", None)
            out += dvb._eq("refcv8", True, getattr(td, "r8_cons_lat", None) is not None,
                           "model.tac_decoder_v6.r8_cons_lat is not None")
        return out

    def _field(dest, attr, conv, default):
        def chk(m, a):
            on = bool(dvb._a(a, "refcv8", False))
            want = conv(dvb._a(a, dest, default))
            if not on:
                return dvb._eq(dest, conv(default), want, "argv (refcv8 off: must hold the default)")
            c = _r8(m)
            got = None if c is None else getattr(c, attr, None)
            if isinstance(want, float):
                return dvb._near(dest, want, got, f"core.decoder.r8_cfg.{attr}")
            return dvb._eq(dest, want, got, f"core.decoder.r8_cfg.{attr}")
        return chk

    for dest, attr, conv, dflt, kind in (
            ("r8_n_alloc", "n_alloc", int, 0, "built"),
            ("r8_alloc_top_k", "alloc_top_k", int, 4, "built"),
            ("r8_alloc_emit", "alloc_emit", bool, False, "built"),
            ("r8_prior_free_group", "prior_free_group", bool, False, "built"),
            ("r8_prior_free_emit", "prior_free_emit", bool, False, "built"),
            ("r8_lat_prior_dropout", "lat_prior_dropout", float, 0.0, "built"),
            ("r8_cond_dropout", "cond_dropout", float, 0.15, "built"),
            ("r8_tf_start", "tf_start", float, 1.0, "built"),
            ("r8_tf_end", "tf_end", float, 0.25, "built"),
            ("r8_base_constraints", "base_constraints", bool, False, "built"),
            ("r8_seed", "seed", int, 20261004, "built"),
            ("w_r8_cons", "w_cons", float, 0.0, "loss"),
            ("w_r8_sat", "w_sat", float, 0.0, "loss"),
            ("w_r8_alloc_l1", "w_alloc_l1", float, 0.0, "loss"),
            ("w_r8_listwise", "w_listwise", float, 0.0, "loss")):
        dvb.register(dest, kind, _field(dest, attr, conv, dflt))

    def _c_no_mod(m, a):
        on = bool(dvb._a(a, "refcv8", False))
        want = bool(dvb._a(a, "r8_no_modulate_base", False))
        if not on:
            return dvb._eq("r8_no_modulate_base", False, want, "argv (refcv8 off)")
        c = _r8(m)
        return dvb._eq("r8_no_modulate_base", want, None if c is None else (not bool(c.modulate_base)),
                       "not core.decoder.r8_cfg.modulate_base")

    _sub_field = _field("w_r8_subscore", "w_subscore", float, 0.0)

    def _c_sub(m, a):
        want = bool(dvb._a(a, "refcv8", False)) and float(dvb._a(a, "w_r8_subscore", 0.0) or 0.0) > 0.0
        return _sub_field(m, a) + dvb._eq("w_r8_subscore", want,
                                          getattr(dvb._dec(m), "r8_sub", None) is not None,
                                          "core.decoder.r8_sub is not None")

    dvb.register("refcv8", "built", _c_enable)
    dvb.register("r8_no_modulate_base", "built", _c_no_mod)
    dvb.register("w_r8_subscore", "loss", _c_sub)
    dvb.register("r8_rc_dropout", "runtime", reason=(
        "set on the model every step by the train loop (`model._r8_rc_dropout`) and read by "
        "refcv8_train.r8_before_forward, which REFUSES a value below 0.3 (MM binding 2026-10-04)"))
    dvb.register("r8_nav_args_dropout", "runtime", reason=(
        "set on the model every step (`model._r8_nav_args_dropout`); read by r8_before_forward: a dropped "
        "row's args are zeroed with the known bit cleared, the nav TOKEN is kept"))
    dvb.register("r8_rc_noise_along_m", "runtime", reason=(
        "set on the model (`model._r8_rc_noise`) and read by r8_before_forward on TRAINING rows only; refused "
        "below the E2'-certified 2.0 m (WP-A INTEGRATION sec. 4: the clean checkpoint leaks)"))
    dvb.register("r8_rc_noise_lat_m", "runtime", reason=(
        "as --r8-rc-noise-along-m, the lateral sigma in the route-tangent frame; refused below 0.75 m"))
    dvb.register("r8_no_rc", "runtime", reason=(
        "`model._r8_no_rc`: r8_before_forward never feeds the route checkpoint (the MM's switch-off while the RC "
        "ruling awaits PI confirmation); the model then sees an explicitly invalid row"))
    for dest, why in (("r8_v9_labels", "the WP-A v9 release (train) behind refcv8_train.R8LabelJoin; md5-checked "
                                       "(--r8-v9-md5), contract-censused, stamped in config.json[refcv8][v9]"),
                      ("r8_v9_labels_eval", "the WP-A v9 release (eval139) behind R8LabelJoin, md5-checked and "
                                            "stamped"),
                      ("r8_v9_md5", "the md5 the TRAIN release must have (load_v9_release refuses others)"),
                      ("r8_v9_eval_md5", "the md5 the EVAL release must have"),
                      ("r8_v9_lat_variant", "which v9 lateral variant supervises lat (a junction = default, b heading "
                                            "= the pre-registered arm), read by R8LabelJoin.item"),
                      ("r8_rc_variant", "selects which v9 route-checkpoint variant R8LabelJoin reads "
                                        "(RC-A30/50/80 or RC-B, v9 SPEC sec. 6.2)"),
                      ("r8_nav_from_v9", "makes V3Dataset feed the v9 per-frame announced nav token as nav_cmd "
                                         "(R8-2) instead of the per-clip v7 token")):
        dvb.register(dest, "data", reason=why)


_register_gdvb()
