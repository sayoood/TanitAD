"""REF-C trainer: behavior-clone Anchored-Diffusion-C (tanitad/refs/refc.py)
from cached episodes.

Mirrors scripts/refb_train.py: the SAME fail-loud dataset (imported, not copied
— FailLoudWindowDataset with the derived nav_cmd/nav_valid/route_target fields),
the same cached-mode ep_*.pt episode dirs, the same atomic-ckpt/resume/jsonl-
step-log/--workers machinery. Divergences are the POINT of the arm:

    optimizer   Adam, lr 1e-4 (the DiffusionDrive/TCP operating point — NOT the
                main run's AdamW/3e-4; batch/warmup/save/log cadence still read
                programmatically from base250cam_config().train)
    decoder     --mode {classifier, diffusion}: classifier is the 0-step anchor-
                selection floor; diffusion refines the winning modes with the
                truncated-denoise steps (cfg.decoder.diffusion_steps). Both train
                the SAME weight set (classifier == diffusion at 0 steps).
    anchors     --anchors <file.pt>: install the FPS anchor vocabulary built by
                scripts/build_refc_anchors.py (else the model's built-in default
                synthetic-FPS anchors are used).
    labels      --labels {v1, v21}: the ROUTE AUX target derivation. ``v1`` is
                what REF-C-XL trained with (refb_labels.route_target(nav_cmd) —
                circular with the fed command AND straight-by-default). ``v21``
                re-derives the route target from refb_labels.route_from_future_v21
                (adaptive horizon, never-straight-by-default, ROUTE_UNKNOWN=3 as
                an out-of-CE-range sentinel that the route CE MASKS OUT — never
                clamped to `straight`). nav_cmd (a model INPUT) keeps the v1
                derivation under both settings, so v21 changes ONE thing.
    losses      traj-recon L1   (1.0)  the GT-assigned anchor's reconstructed
                                       ego-frame trajectory vs the target
                                       [refc1: fixed-distance path checkpoints
                                       via refb_labels.path_targets]
                anchor-cls CE   (1.0)  classify the GT-nearest anchor
                LAW MSE         (0.5)  predicted next pooled latent vs the
                                       no_grad-encoded frames at t+5
                route CE        (0.1)  strategic route-heading aux
                maneuver CE     (0.1)  kinematic maneuver pseudo-label aux
                [refc1] speed CE (0.2) target-speed class (4 bins, [0,30] m/s)

v0 = pose_last[:, 3] is ALWAYS fed (the model applies /10 scaling and the per-
sample ego-dropout p=0.5 internally, training-gated).

Usage (only AFTER Sayed's GO — implementation ships untrained):
  python scripts/refc_train.py --data-root /workspace/data \
      --out /workspace/experiments/refc-30k --steps 30000 --mode diffusion \
      --anchors /workspace/experiments/refc_anchors.pt
Smoke (CPU):
  python scripts/refc_train.py --data-root <cache> --out <dir> --steps 10 \
      --batch 8 --smoke --log-every 1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import refb_labels
from refb_train import FailLoudWindowDataset, load_cached_episodes
from tanitad.data.lan import LanConfig, lan_window_features
from tanitad.config import base250cam_config
from tanitad.refs import refc_tactical
from tanitad.refs.refc import (N_ROUTE, RefCModel, param_breakdown, refc_config,
                               LanConfig as RefCLanConfig,
                               refc_small_config, refc_smoke_config,
                               refc_xl_config)
from tanitad.train.train_worldmodel import cosine_lr

# Loss weights (module docstring). traj/anchor-cls are the co-equal primaries
# (this stack IS the trajectory decoder), LAW rides at 0.5, route/maneuver are
# 0.1 strategic/tactical aux shaping, speed-class (refc1) at 0.2.
TRAJ_WEIGHT = 1.0
ANCHOR_CLS_WEIGHT = 1.0
LAW_WEIGHT = 0.5
ROUTE_WEIGHT = 0.1
MANEUVER_WEIGHT = 0.1
SPEED_CLS_WEIGHT = 0.2        # refc1 only

# D-TAC1: with --factored-maneuver the ONE 5-way CE becomes TWO. The total
# tactical aux pressure is held at EXACTLY MANEUVER_WEIGHT so the arm differs
# from the base run in STRUCTURE, not in loss budget — "the aux got louder" is
# the first confound anyone would raise, and it is removed by construction here
# rather than argued about afterwards. The 5-way CE is NOT kept alongside: the
# 5-way label is a deterministic function of the (lat, lon) pair, so supervising
# the pair supervises it, and `maneuver_logits` is still emitted (derived).
LAT_WEIGHT = MANEUVER_WEIGHT / 2.0     # 0.05
LON_WEIGHT = MANEUVER_WEIGHT / 2.0     # 0.05

# D-SEL S1: the RANKING term. ``anchor_logits`` is supervised against the
# GT-nearest ORIGINAL anchor (the DiffusionDrive vocabulary-assignment CE, which
# also picks the reconstruction target); ``sel_score`` is supervised against the
# GT-nearest REFINED trajectory — the quantity ``argmax`` actually ranks.
#
# ⚠️ S1 IS THESE TWO CHANGES TOGETHER AND CANNOT BE SPLIT. Ranking on the refined
# confidence while leaving it unsupervised would rank on an untrained readout;
# supervising it while ranking on the classifier score would train a head nothing
# reads. So the arm carries a NEW loss term, and the honest statement is that the
# lever is "rank the refined fan AND supervise that ranking", not "one flag".
# Weight 1.0 matches ``flagship_v15.REFINED_CLS_WEIGHT`` (the same recipe, in the
# module that first repaired this decoder), so the arm inherits an operating
# point rather than inventing one.
REFINED_CLS_WEIGHT = 1.0

# S6 goal-head supervision, held at ROUTE_WEIGHT so the goal arm differs from
# the control in STRUCTURE, not in aux loss budget — the same discipline
# LAT_WEIGHT/LON_WEIGHT apply to the factored tactical seam.
GOAL_WEIGHT = ROUTE_WEIGHT             # 0.1

TCP_LR = 1e-4                 # Adam lr — the DiffusionDrive/TCP operating point
LAW_AHEAD = 5                 # LAW target: pooled latent 0.5 s (5 steps) ahead
SPEED_AHEAD = 5              # refc1 speed target: v at t+5 (same 0.5 s horizon)

MILESTONES = (5000, 15000, 20000, 30000)   # scaling-study gate protocol (D-030)


# ---- v2.1 route labels (opt-in; --labels v21) --------------------------------

class RouteV21Dataset(FailLoudWindowDataset):
    """FailLoudWindowDataset whose ROUTE AUX TARGET is re-derived by the v2.1
    labeler (``refb_labels.route_from_future_v21``).

    ONLY the route target and its validity mask change. ``nav_cmd`` is a model
    INPUT and keeps the exact v1 derivation REF-C-XL trained with, so a v21 run
    differs from an XL-style run in the route LABEL SET alone — not in what the
    measurement encoder is fed (this mirrors flagship v1.5's `route_v21` mint,
    which likewise left the fed command on the v1 path).

    ``route_target`` carries ``refb_labels.ROUTE_UNKNOWN`` (= 3), deliberately
    OUTSIDE the 3-wide CE class range, whenever ``route_valid`` is False. The
    trainer masks those windows out of the route CE. It is NEVER clamped to
    `straight` — an unmasked cross-entropy raising an index error is the
    intended fail-loud behaviour (the silent straight-clamp is the exact bug the
    v2.1 labeler exists to remove).

    ``use_net_dyaw`` defaults to False per Sayed's 2026-07-20 ruling: a wide
    sweep is ROAD FOLLOWING, not a route event.
    """

    def __init__(self, *a, use_net_dyaw: bool = False, **kw):
        super().__init__(*a, **kw)
        self.use_net_dyaw = bool(use_net_dyaw)

    def __getitem__(self, i: int):
        item = super().__getitem__(i)          # v1 nav_cmd/nav_valid + clones
        e_i, t = self.index[i]
        r = refb_labels.route_from_future_v21(
            self.episodes[e_i].poses, t + self.window - 1,
            use_net_dyaw=self.use_net_dyaw)
        item["route_target"] = torch.tensor(int(r["route"]), dtype=torch.long)
        item["route_valid"] = torch.tensor(bool(r["valid"]))
        return item

    def label_stats(self, n: int = 4000, seed: int = 0) -> dict:
        """Route-label provenance row over ``n`` sampled windows (config.json)."""
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(self.index), generator=g)[:min(n, len(self))]
        counts = [0, 0, 0, 0]
        reasons: dict[str, int] = {}
        for i in idx.tolist():
            e_i, t = self.index[i]
            r = refb_labels.route_from_future_v21(
                self.episodes[e_i].poses, t + self.window - 1,
                use_net_dyaw=self.use_net_dyaw)
            counts[int(r["route"])] += 1
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        tot = max(int(idx.numel()), 1)
        return {"n_sampled": tot,
                "route_counts": {"left": counts[0], "straight": counts[1],
                                 "right": counts[2], "UNKNOWN": counts[3]},
                "route_frac": [round(c / tot, 4) for c in counts],
                "valid_frac": round((tot - counts[3]) / tot, 4),
                "reasons": {k: round(v / tot, 4) for k, v in sorted(reasons.items())}}


# ---- LAN route input (opt-in; --graft-lan) -----------------------------------

def lan_dataset_class(base):
    """Wrap ANY FailLoudWindowDataset subclass so each item carries ``lan``.

    Composes with the v1 / v21 / v3 label sets instead of forking them, because
    LAN changes a model INPUT and must not perturb the route CE TARGET — the
    same separation ``RouteV21Dataset`` documents for ``nav_cmd``.
    """

    class _WithLan(base):
        def __init__(self, *a, lan_cfg=None, **kw):
            super().__init__(*a, **kw)
            self.lan_cfg = lan_cfg or LanConfig()

        def __getitem__(self, i: int):
            item = super().__getitem__(i)
            e_i, t = self.index[i]
            f = lan_window_features(self.episodes[e_i].poses,
                                    t + self.window - 1, self.lan_cfg)
            item["lan"] = torch.from_numpy(f)      # already float32 [K*4]
            return item

        def lan_stats(self, n: int = 4000, seed: int = 0) -> dict:
            """Route-INPUT coverage over sampled windows (config.json row).

            The number that matters: ``any_valid_frac``. The 4-way ``nav_cmd``
            it replaces is valid on 0.21-0.25 of windows (MEASURED, all four
            arms) — if LAN is not materially higher, the input is not fixed and
            the experiment should not run.
            """
            g = torch.Generator().manual_seed(seed)
            idx = torch.randperm(len(self.index), generator=g)[:min(n, len(self))]
            k = self.lan_cfg.k
            per_anchor = [0.0] * k
            any_valid = 0
            for i in idx.tolist():
                e_i, t = self.index[i]
                f = lan_window_features(self.episodes[e_i].poses,
                                        t + self.window - 1, self.lan_cfg)
                v = f.reshape(k, -1)[:, 3]
                per_anchor = [a + float(b) for a, b in zip(per_anchor, v)]
                any_valid += int(v.any())
            tot = max(int(idx.numel()), 1)
            return {"n_sampled": tot,
                    "arclengths_m": list(self.lan_cfg.arclengths_m),
                    "min_lead_m": self.lan_cfg.min_lead_m,
                    "per_anchor_valid_frac": [round(x / tot, 4)
                                              for x in per_anchor],
                    "any_valid_frac": round(any_valid / tot, 4)}

    _WithLan.__name__ = f"Lan{base.__name__}"
    _WithLan.__qualname__ = _WithLan.__name__
    return _WithLan


# ---- v3 labels (opt-in; --labels v3) -----------------------------------------

class RouteV3Dataset(RouteV21Dataset):
    """RouteV21Dataset plus the v3 VOCABULARY TOKEN + DISTANCE-TO-MANEUVER and
    the FACTORIZED tactical slots.

    THE 3-CLASS CE TARGET IS UNCHANGED. ``route_from_future_v3`` calls
    ``route_from_future_v21`` for its decision and only ever upgrades the TOKEN,
    so ``route_target`` / ``route_valid`` / ``nav_cmd`` here are byte-identical
    to a ``--labels v21`` run (pinned by tests/test_refc_labels_v3_wiring.py).
    A v3 run therefore differs from a v21 run ONLY by the ADDITIONAL fields,
    which no head consumes yet:

        route_token_idx  [] long   index into refb_labels.ROUTE_V3_TOKENS, or
                                   len(ROUTE_V3_TOKENS) for `unknown`
        route_dist_idx   [] long   index into refb_labels.DIST_BAND_TOKENS
        lat_idx / lon_idx[] long   index into the KINEMATICALLY MINTABLE subsets
                                   of the frozen vocab LATMANEUVER / LONMODE
                                   slots, sentinel = len(table) for `unknown`
        lon_active       [] bool   a longitudinal decision is live

    Wiring the heads that consume them is a RETRAIN and Sayed's call — see
    "TanitAD Research Hub/Architecture & Inference/
     V3_FACTORIZED_TACTICAL_HEAD_SPEC.md". This dataset exists so the label set
    is selectable and measurable BEFORE that decision, exactly as v21 was.
    """

    ROUTE_TOKENS = refb_labels.ROUTE_V3_TOKENS
    DIST_TOKENS = refb_labels.DIST_BAND_TOKENS
    LAT_TOKENS = refb_labels.LAT_KINEMATIC_TOKENS
    LON_TOKENS = refb_labels.LON_KINEMATIC_TOKENS

    @staticmethod
    def _idx(tok: str, table) -> int:
        return table.index(tok) if tok in table else len(table)   # sentinel

    def __getitem__(self, i: int):
        item = super().__getitem__(i)             # v21 route target, UNCHANGED
        e_i, t = self.index[i]
        poses = self.episodes[e_i].poses
        last = t + self.window - 1
        r = refb_labels.route_from_future_v3(poses, last,
                                             use_net_dyaw=self.use_net_dyaw)
        tac = refb_labels.tactical_from_future_v3(poses, last)
        item["route_token_idx"] = torch.tensor(
            self._idx(r["token"], self.ROUTE_TOKENS), dtype=torch.long)
        item["route_dist_idx"] = torch.tensor(
            self._idx(r["dist_band"], self.DIST_TOKENS), dtype=torch.long)
        item["lat_idx"] = torch.tensor(
            self._idx(tac["lat"]["token"], self.LAT_TOKENS), dtype=torch.long)
        item["lon_idx"] = torch.tensor(
            self._idx(tac["lon"]["token"], self.LON_TOKENS), dtype=torch.long)
        item["lon_active"] = torch.tensor(bool(tac["lon"]["active"]))
        return item

    def label_stats(self, n: int = 4000, seed: int = 0) -> dict:
        """v21 provenance row + the v3 token / distance / factorization mix."""
        out = super().label_stats(n, seed)
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(self.index), generator=g)[:min(n, len(self))]
        tok, dist, lon, lat = {}, {}, {}, {}
        upgraded = collapsed = rb_cand = 0
        for i in idx.tolist():
            e_i, t = self.index[i]
            poses = self.episodes[e_i].poses
            last = t + self.window - 1
            r = refb_labels.route_from_future_v3(poses, last,
                                                 use_net_dyaw=self.use_net_dyaw)
            tac = refb_labels.tactical_from_future_v3(poses, last)
            tok[r["token"]] = tok.get(r["token"], 0) + 1
            dist[r["dist_band"]] = dist.get(r["dist_band"], 0) + 1
            lo, la = tac["lon"]["token"], tac["lat"]["token"]
            lon[lo] = lon.get(lo, 0) + 1
            lat[la] = lat.get(la, 0) + 1
            upgraded += int(r["upgraded"])
            rb_cand += int(r["roundabout_candidate"])
            collapsed += int(tac["collapsed"])
        tot = max(int(idx.numel()), 1)
        out.update(v3_token=tok, v3_dist_band=dist, v3_lon=lon, v3_lat=lat,
                   v3_upgraded=upgraded,
                   v3_upgraded_frac=round(upgraded / tot, 4),
                   v3_roundabout_candidates=rb_cand,
                   v3_collapsed_by_5way=collapsed,
                   v3_collapsed_frac=round(collapsed / tot, 4))
        return out


# ---- losses ------------------------------------------------------------------

def compute_losses(model: RefCModel, batch: dict, device: str = "cpu",
                   mode: str = "diffusion") -> dict:
    """One forward pass -> all loss components (tensors, differentiable).

    Anchor assignment: the GT trajectory target is assigned to its NEAREST anchor
    (flattened L2); anchor-cls CE classifies that index and traj-recon L1
    regresses the reconstructed trajectory FROM the assigned anchor. The LAW
    target is the pooled latent LAW_AHEAD steps past the window, encoded under
    no_grad through the SAME encoder; the prediction path keeps gradients THROUGH
    the decoded trajectory — the point of the aux. ``mode`` picks the decoder's
    inference mode (classifier == 0 steps, diffusion == cfg.diffusion_steps)."""
    frames = batch["frames"].to(device)            # [B, W, C, H, W']
    fut_frames = batch["future_frames"].to(device)  # [B, Hmax, C, H, W']
    fut_poses = batch["future_poses"].to(device)   # [B, Hmax, 4]
    pose_last = batch["pose_last"].to(device)      # [B, 4]
    nav_cmd = batch["nav_cmd"].to(device)          # [B] long (derived)
    nav_valid = batch["nav_valid"].to(device)      # [B] bool
    route_tgt = batch["route_target"].to(device)   # [B] long
    v0 = pose_last[:, 3]                            # [B] current ego speed (t0)
    # LAN route corridor (only when the graft is on AND the dataset emits it —
    # a missing key must not be silently defaulted to "no route").
    lan = batch["lan"].to(device) if "lan" in batch else None

    steps = model.cfg.decoder.diffusion_steps if mode == "diffusion" else 0
    out = model(frames, nav_cmd=nav_cmd, v0=v0, steps=steps, lan=lan)
    cfg = model.cfg
    b = frames.shape[0]

    # Trajectory target: time-indexed ego-frame waypoints, or (refc1) fixed-
    # distance path checkpoints via the arc-length resample.
    if cfg.refc1:
        traj_tgt = refb_labels.path_targets(pose_last, fut_poses, cfg.path_dists)
    else:
        traj_tgt = refb_labels.waypoint_targets(pose_last, fut_poses,
                                                cfg.trajectory.horizons)

    # Anchor assignment: nearest anchor to the GT trajectory (flattened L2).
    anchors = model.decoder.anchors.to(traj_tgt.dtype)     # [N, S, 2]
    dist = ((traj_tgt[:, None] - anchors[None]) ** 2).sum(dim=(-1, -2))  # [B, N]
    a_star = dist.argmin(dim=1)                            # [B]

    # anchor-cls CE + traj-recon L1 (reconstruction FROM the assigned anchor).
    ar = torch.arange(b, device=device)
    loss_cls = F.cross_entropy(out["anchor_logits"], a_star)
    recon = out["anchor_traj"][ar, a_star]
    loss_traj = (recon - traj_tgt).abs().mean()

    # D-SEL: supervise the RANKED score against the oracle REFINED candidate,
    # and report the selection diagnostic the fleet asked for as a standing
    # metric — ``oracle_ade`` is the best plan available in the fan, ``sel_ade``
    # is the one taken, and ``sel_gap`` separates "cannot propose it" (both high)
    # from "cannot rank it" (gap high). REF-C's published gap is the second.
    #
    # ⭐ THE CE FIRES FOR EVERY LEVER THAT TOUCHES THE RANKED SCORE, not only
    # for S1 — and that is a STRUCTURAL DEPENDENCY, not a convenience. ``argmax``
    # has NO gradient, and the only other consumers of the selected trajectory
    # (``traj``, and ``law_pred`` through it) differentiate w.r.t. the fan, never
    # w.r.t. the score. So without this term ``cons_gate`` and
    # ``route_to_anchor`` receive EXACTLY ZERO gradient and are dead parameters —
    # found by ``tests/test_refc_select.py`` on the first run of the gradient
    # check, which is what that check is for. (``flagship_v15.v15_losses`` states
    # the same mechanism for its ``sel_gate``: "supervising the score rather than
    # the bare logits is also what gives the longitudinal gate a gradient —
    # argmax has none".)
    sel_extra: dict = {}
    loss_rcls = torch.zeros((), device=out["pooled"].device)
    if cfg.sel_refined or cfg.graft_cons or cfg.graft_route:
        fan_err = (out["anchor_traj"] - traj_tgt[:, None]).norm(dim=-1).mean(-1)
        r_star = fan_err.argmin(dim=1)                      # [B] oracle index
        loss_rcls = F.cross_entropy(out["sel_score"], r_star.detach())
        with torch.no_grad():
            oracle = fan_err.min(dim=1).values
            sel_err = fan_err[ar, out["sel_idx"]]
            sel_extra = {
                "cls_refined": loss_rcls.detach(),
                "oracle_ade": oracle.mean(),
                "sel_ade": sel_err.mean(),
                "sel_gap": (sel_err - oracle).mean(),
                "rank_acc": (out["sel_idx"] == r_star).float().mean(),
                # the headline REF-C selection pathology, in-loop: MEASURED
                # 45.4 % on refc-xl-30k / 41.09 % on refc-base-30k at 30 k.
                "frac_sel_2x_worse": (sel_err > 2.0 * oracle).float().mean(),
            }

    # LAW latent MSE: no_grad target through the same encoder.
    with torch.no_grad():
        law_tgt = model.encode_pooled(fut_frames[:, LAW_AHEAD - 1])
    loss_law = (out["law_pred"] - law_tgt).pow(2).mean()

    # Maneuver CE (kinematic pseudo-labels) + route-heading CE.
    man_tgt = refb_labels.window_maneuver_labels(
        pose_last, fut_poses, horizon=max(cfg.trajectory.horizons))
    lat_extra: dict = {}
    if cfg.factored_maneuver:
        # D-TAC1. The SAME endpoint kinematics the 5-way labeler reads, split
        # onto its two axes — so the arm's labels are a refinement of the base
        # arm's, never a different label set. The projection back through the
        # priority collapse is asserted equal to `man_tgt` below (a
        # component-vs-family self-consistency control: a silent divergence
        # between the two labelers would train the arm on a different target
        # and make the A/B non-attributable).
        lat_tgt, lon_tgt = refc_tactical.window_factored_labels(
            pose_last, fut_poses, horizon=max(cfg.trajectory.horizons))
        if not bool((refc_tactical.collapse(lat_tgt, lon_tgt) == man_tgt).all()):
            raise ValueError(
                "factored tactical labels do not collapse to the 5-way label — "
                "refc_tactical and refb_labels have DRIFTED; refusing to train "
                "on a target that is not the documented one")
        loss_lat = F.cross_entropy(out["lat_logits"], lat_tgt)
        loss_lon = F.cross_entropy(out["lon_logits"], lon_tgt)
        # `loss_man` is multiplied by MANEUVER_WEIGHT in the total below, so the
        # fractions here make the EFFECTIVE weights exactly LAT_WEIGHT/LON_WEIGHT.
        loss_man = ((LAT_WEIGHT / MANEUVER_WEIGHT) * loss_lat
                    + (LON_WEIGHT / MANEUVER_WEIGHT) * loss_lon)
        if model.training:
            model.update_tactical_prior(lat_tgt, lon_tgt)
        lat_extra = {
            "lat": loss_lat.detach(), "lon": loss_lon.detach(),
            "lat_acc": (out["lat_logits"].argmax(-1) == lat_tgt).float().mean(),
            "lon_acc": (out["lon_logits"].argmax(-1) == lon_tgt).float().mean(),
            # THE metric this whole change exists to move: what fraction of
            # windows the model DECIDES are longitudinally active. The base arm
            # scores 7/859 = 0.008 on the 5-way surface (LAN_E0_RESULTS.md).
            "lon_active_pred": (out["lon_decision"]
                                != refc_tactical.LON_STEADY).float().mean(),
            "lon_active_tgt": (lon_tgt
                               != refc_tactical.LON_STEADY).float().mean(),
        }
    else:
        loss_man = F.cross_entropy(out["maneuver_logits"], man_tgt)
    if "route_valid" in batch:
        # v2.1 labels: mask on the ROUTE validity. route_target is
        # ROUTE_UNKNOWN (=3, out of CE range) wherever invalid — masked out, and
        # NEVER clamped to `straight`. An UNKNOWN surviving the mask is a
        # labeler contract violation: raise, do not train a wrong class.
        mask = batch["route_valid"].to(device)
        if bool(mask.any()):
            tgt_v = route_tgt[mask]
            if int(tgt_v.max()) >= N_ROUTE:
                raise ValueError(
                    f"ROUTE_UNKNOWN survived the valid mask (max target "
                    f"{int(tgt_v.max())} >= n_route {N_ROUTE}) — the v2.1 "
                    f"contract is route<3 <=> valid=True")
            loss_route = F.cross_entropy(out["route_logits"][mask], tgt_v)
        else:                       # no judgeable window in this batch
            loss_route = torch.zeros((), device=out["pooled"].device)
    else:
        # v1 labels (what REF-C-XL trained with) — byte-identical path,
        # including the fall-back-to-all-windows behaviour.
        mask = nav_valid if bool(nav_valid.any()) else torch.ones_like(nav_valid)
        loss_route = F.cross_entropy(out["route_logits"][mask], route_tgt[mask])
    route_acc = ((out["route_logits"][mask].argmax(-1) == route_tgt[mask])
                 .float().mean() if bool(mask.any())
                 else torch.zeros((), device=out["pooled"].device))

    # S6: the PREDICTED GEOMETRIC goal. The head reads the IMAGE EMBEDDING only
    # (`RefCModel.goal_provenance()`); the LAN corridor appears here as the
    # TRAINING LABEL and nowhere else — the sanctioned direction of "LABELS MAY
    # USE EGO; INFERENCE IS VISION-ONLY".
    loss_goal = torch.zeros((), device=out["pooled"].device)
    goal_extra: dict = {}
    if cfg.graft_goal:
        if lan is None:
            raise ValueError(
                "graft_goal is on but the batch carries no `lan` field — the "
                "goal head would be a DEAD parameter with no label. Pass "
                "--graft-lan (which mints the corridor) alongside --graft-goal; "
                "the corridor is the LABEL only, never a model input here.")
        g_dir, g_dist, g_valid = RefCModel.goal_targets(lan, cfg.lan.k)
        if bool(g_valid.any()):
            mv = g_valid
            # bearing: cosine distance on unit vectors (angle-only, so a
            # magnitude the head cannot know is never regressed)
            l_dir = (1.0 - (out["goal_bearing"][mv] * g_dir[mv]).sum(-1)).mean()
            l_dist = (out["goal_dist_pref"][mv] - g_dist[mv]).abs().mean()
            loss_goal = l_dir + l_dist
            goal_extra = {"goal_dir": l_dir.detach(), "goal_dist": l_dist.detach(),
                          "goal_valid_frac": g_valid.float().mean(),
                          # the K7 instrument, readable per log line
                          "goal_gate": model.decoder.goal_gate.detach().abs()[0],
                          "goal_dist_gate":
                              model.decoder.goal_dist_gate.detach().abs()[0]}

    # refc1: target-speed classification (bins over [0, speed_max]).
    if cfg.refc1:
        v_tgt = fut_poses[:, SPEED_AHEAD - 1, 3].clamp(0.0, cfg.speed_max)
        edges = torch.linspace(0.0, cfg.speed_max, cfg.speed_bins + 1,
                               device=v_tgt.device)[1:-1]
        cls_tgt = torch.bucketize(v_tgt, edges)
        loss_speed_cls = F.cross_entropy(out["speed_logits"], cls_tgt)
        speed_mae = (out["target_speed"].detach() - v_tgt).abs().mean()
    else:
        loss_speed_cls = torch.zeros((), device=out["pooled"].device)
        speed_mae = torch.zeros((), device=out["pooled"].device)

    loss = (TRAJ_WEIGHT * loss_traj + ANCHOR_CLS_WEIGHT * loss_cls
            + REFINED_CLS_WEIGHT * loss_rcls
            + LAW_WEIGHT * loss_law + ROUTE_WEIGHT * loss_route
            + MANEUVER_WEIGHT * loss_man + SPEED_CLS_WEIGHT * loss_speed_cls
            + GOAL_WEIGHT * loss_goal)
    anchor_acc = (out["anchor_logits"].argmax(dim=1) == a_star).float().mean()
    man_acc = (out["maneuver_logits"].argmax(dim=1) == man_tgt).float().mean()
    return {"loss": loss, "traj": loss_traj, "cls": loss_cls, "law": loss_law,
            "route": loss_route, "man": loss_man,
            "speed_cls": loss_speed_cls, "speed_mae": speed_mae,
            "anchor_acc": anchor_acc, "man_acc": man_acc,
            "route_acc": route_acc, "route_valid_frac": mask.float().mean(),
            "nav_follow_frac": (nav_cmd == 0).float().mean(),
            **sel_extra, **goal_extra, "sel_tele": out["sel_tele"],
            **lat_extra,
            **({"graft_lat_norm": model.decoder.lat_to_anchor(
                out["lat_logits"].detach().log_softmax(-1)).norm(dim=-1).mean(),
                "graft_lon_norm": model.decoder.lon_to_anchor(
                    out["lon_logits"].detach().log_softmax(-1)
                ).norm(dim=-1).mean(),
                "conf_norm": out["anchor_logits"].detach().norm(dim=-1).mean()}
               if cfg.factored_maneuver and cfg.graft_maneuver else {}),
            **({"lan_valid_frac": out["lan_dir"][:, 2].mean()}
               if "lan_dir" in out else {}),
            "pooled": out["pooled"]}


def assert_selection_params_are_alive(model) -> dict:
    """⛔ FAIL-LOUD: every D-SEL parameter must have a real gradient.

    Called once, after the FIRST backward. A zero-init graft is gated, and a
    gated graft is indistinguishable from a DEAD one until you look at its
    gradient — ``lon_to_anchor``, ``lan_gate`` and ``ctx_to_cond`` all start at
    exactly 0 on purpose, so "the weight is zero" proves nothing either way.
    The programme has already paid for this distinction twice: the flagship
    trained ``cond_imagination`` at ZERO tokens for a whole run, and D-TAC1's
    ``tactical_speed_input`` was coupled to ``factored_maneuver`` in a way that
    silently deleted the F1-only ablation ("a conservative guard that makes an
    effect unattributable is not conservative").

    Raising here costs seconds; discovering it in a post-mortem costs GPU-days.
    """
    checks, dead = {}, []
    for name, p in model.named_parameters():
        if not any(t in name for t in ("route_to_anchor", "cons_gate",
                                       "goal_gate", "goal_dist_gate",
                                       "goal_head")):
            continue
        g = 0.0 if p.grad is None else float(p.grad.abs().sum())
        checks[name] = g
        if p.grad is None or g == 0.0:
            dead.append(name)
    if dead:
        raise RuntimeError(
            f"D-SEL parameters received NO gradient: {dead}. They are dead, not "
            f"gated. The ranked score must be SUPERVISED for a graft on it to "
            f"train — `argmax` has no gradient and nothing else in the loss "
            f"differentiates w.r.t. `sel_score`. Check that compute_losses "
            f"builds `loss_rcls` for this flag combination.")
    return checks


def _save_ckpt(path: Path, model, opt, step: int) -> None:
    # atomic write: a kill mid-save must not corrupt the resume point
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": step}, tmp)
    tmp.replace(path)
    print(f"[ckpt] saved at step {step}", flush=True)
    # Milestone archive (D-030 scaling study): ckpt.pt is overwritten every
    # save_every, so preserve 5k/15k/20k/30k for the gate protocol. Atomic —
    # a bare copy can leave a truncated file that exists() calls done forever.
    for m in MILESTONES:
        if step >= m:
            arch = path.with_name(f"ckpt_step{m}.pt")
            if not arch.exists():
                from tanitad.train.ckpt_io import atomic_archive
                atomic_archive(path, arch)
                print(f"[ckpt] milestone archived: {arch.name}", flush=True)


def train(args) -> dict:
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    torch.manual_seed(args.seed)

    # Cadence/batch from the main run's config object (mirrors refb_train);
    # the OPTIMIZER is deliberately TCP's (Adam, lr 1e-4) — the arm's point.
    main_tr = base250cam_config().train
    lr = args.lr if args.lr is not None else TCP_LR
    batch = args.batch if args.batch is not None else main_tr.batch_size
    warmup = args.warmup if args.warmup is not None else main_tr.warmup_steps
    save_every = args.save_every if args.save_every is not None \
        else main_tr.save_every
    log_every = args.log_every if args.log_every is not None \
        else main_tr.log_every

    # Scale preset: --smoke (tiny CPU) overrides; else --config small/base/xl
    # selects the size-vs-data-scaling study arm (all three share refc.py's
    # anchored-diffusion algorithm; only widths/depths/anchor-vocab differ).
    _presets = {"small": refc_small_config, "base": refc_config,
                "xl": refc_xl_config}
    cfg = refc_smoke_config() if args.smoke else _presets[args.config]()
    # ⭐ THE CORRIDOR IS NEEDED IN TWO DIFFERENT ROLES, AND THEY ARE SEPARATED.
    #   --graft-lan  : the corridor as a MODEL INPUT (the SUPPLIED route).
    #   --graft-goal : the corridor as the goal head's TRAINING LABEL ONLY.
    # Both need the dataset to emit the `lan` field; only the first may build the
    # input pathway. Conflating them would give the S6 arm a supplied route AND a
    # predicted one — precisely what the PI ruling of 2026-08-03 forbids ("a
    # supplied route is optimistic by construction on PhysicalAI, whose only
    # route supplier is the ego's own future path").
    want_lan_field = bool(args.graft_lan or args.graft_goal)
    if want_lan_field:
        cfg.lan = RefCLanConfig(k=len(args.lan_arclengths))
    if args.graft_lan:
        cfg.graft_lan = True
        cfg.route_dropout = args.route_dropout
    cfg.refc1 = bool(args.refc1)       # gated BEFORE build (module presence)
    # D-TAC1 (gated BEFORE build — module presence, REF-B convention).
    cfg.factored_maneuver = bool(args.factored_maneuver)
    cfg.tactical_speed_input = bool(args.tactical_speed_input)
    cfg.man_prior_tau = float(args.man_prior_tau)
    cfg.graft_prior_center = not args.no_graft_prior_center
    # --man-prior-tau is the F3 DECISION lever and it acts on the per-axis
    # priors, which only exist with the factored seam — so it still requires it.
    # --tactical-speed-input (F1 INPUT) does NOT: it applies to the shipped 5-way
    # head too, and that is exactly the arm (`refc_f1only_config`) that isolates
    # INPUT from STRUCTURE. Coupling them left F1 estimable only as
    # `full - f2only`, where the two arms also differ in the head itself.
    if args.man_prior_tau and not args.factored_maneuver:
        raise SystemExit("--man-prior-tau requires --factored-maneuver (it "
                         "adjusts the per-axis lat/lon class priors, which only "
                         "the factored seam registers)")
    # --- D-SEL: the selection surface (gated BEFORE build, module presence) ---
    cfg.sel_refined = bool(args.sel_refined)
    cfg.sel_reach_clamp = bool(args.sel_reach_clamp)
    cfg.sel_accel_max = float(args.sel_accel_max)
    cfg.graft_cons = bool(args.graft_cons)
    cfg.graft_route = bool(args.graft_route)
    cfg.graft_goal = bool(args.graft_goal)
    cfg.seam_clamp = float(args.seam_clamp)
    cfg.ego_valid_channel = bool(args.ego_valid_channel)
    # ⛔ C6 GUARD. Under --labels v1 the route CE target is
    # ``refb_labels.route_target(nav_cmd)`` — a deterministic function of a model
    # INPUT. Grafting that readout onto SELECTION would pipe the nav echo into
    # the ranking and any route effect measured afterwards would be circular.
    # This is the same failure class as RETRACTION_LOG C6 and R-2026-08-03-l
    # (flagship's route_class_accuracy = 1.0000 is an oracle-conditioning echo,
    # not a decision). Refuse at ARGUMENT-PARSE time, not after a GPU-day.
    # S6 mints its OWN label (`want_lan_field` above) and must NOT be forced to
    # turn on the supplied-route model input to get it. But `--graft-goal
    # --graft-lan` together IS admissible as an explicit contrast arm, so it is
    # allowed and merely announced — the config.json records both, and the
    # provenance row says which one the goal seam actually reads.
    if cfg.graft_goal and cfg.graft_lan:
        print("[d-sel] ⚠️ --graft-goal WITH --graft-lan: this arm carries BOTH "
              "the SUPPLIED corridor (a model input) and the PREDICTED goal. "
              "That is a contrast arm, not the S6 arm — refc_goal_config() "
              "keeps graft_lan OFF so the goal is predicted and nothing is "
              "supplied (PI ruling 2026-08-03).", flush=True)
    if cfg.graft_route and args.labels == "v1":
        raise SystemExit(
            "--graft-route requires --labels v21 or v3. Under v1 the route CE "
            "target is route_target(nav_cmd) — circular with a model INPUT — so "
            "grafting route_logits onto the ranked score would train selection "
            "on a nav echo (RETRACTION_LOG C6 / R-2026-08-03-l).")
    model = RefCModel(cfg).to(device)
    # Install the FPS anchor vocabulary (else the built-in default anchors).
    if args.anchors:
        anc = torch.load(args.anchors, map_location=device, weights_only=True)
        anc = anc["anchors"] if isinstance(anc, dict) else anc
        model.decoder.load_anchors(anc.to(device))
        print(f"[refc] loaded {tuple(anc.shape)} anchors from {args.anchors}",
              flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    max_h = max(max(cfg.trajectory.horizons), LAW_AHEAD, SPEED_AHEAD)
    train_eps, train_dir = load_cached_episodes(args.data_root, "*train*",
                                                args.episodes)
    ds_kw = dict(window=cfg.window, max_horizon=max_h,
                 channels=cfg.encoder.in_channels)
    label_stats: dict | None = None
    lan_stats: dict | None = None
    lan_cfg = LanConfig(arclengths_m=tuple(args.lan_arclengths),
                        min_lead_m=args.lan_min_lead_m) if want_lan_field else None
    lan_kw = {"lan_cfg": lan_cfg} if want_lan_field else {}
    if args.labels in ("v21", "v3"):
        dcls = RouteV21Dataset if args.labels == "v21" else RouteV3Dataset
        if want_lan_field:
            dcls = lan_dataset_class(dcls)
        ds = dcls(train_eps, use_net_dyaw=args.use_net_dyaw, **lan_kw, **ds_kw)
        label_stats = ds.label_stats()
        print(f"[labels] {args.labels} route (use_net_dyaw="
              f"{args.use_net_dyaw}): {json.dumps(label_stats)}", flush=True)
    else:
        dcls = (lan_dataset_class(FailLoudWindowDataset) if want_lan_field
                else FailLoudWindowDataset)
        ds = dcls(train_eps, **lan_kw, **ds_kw)
    if want_lan_field:
        lan_stats = ds.lan_stats()
        print(f"[lan] route-input coverage: {json.dumps(lan_stats)}", flush=True)
    assert len(ds) >= batch, \
        f"only {len(ds)} windows for batch {batch} — add episodes"
    dl_kw = dict(batch_size=batch, shuffle=True, drop_last=True)
    if getattr(args, "workers", 0) > 0:
        dl_kw.update(num_workers=args.workers, persistent_workers=True,
                     prefetch_factor=2, pin_memory=True)
    dl = DataLoader(ds, **dl_kw)
    print(f"[refc] train: {len(train_eps)} episodes / {len(ds)} windows "
          f"from {train_dir} (mode={args.mode})", flush=True)
    # D-SEL preflight banner: the arm PRINTS ITS OWN AXES. The DATA-vs-ARCH
    # conflation (a `--v2-cache` read as `--v2`) is only structurally visible if
    # every lever is on one line of the log before step 0. `sel_on` False with a
    # D-SEL flag set would be a wiring bug, so the banner asserts rather than
    # merely reports.
    _sel = cfg.selection()
    dsel_row = {
        "sel_on": _sel.any_on, "sel_refined": cfg.sel_refined,
        "sel_reach_clamp": cfg.sel_reach_clamp,
        "sel_accel_max": cfg.sel_accel_max, "horizon_s": _sel.horizon_s,
        "graft_cons": cfg.graft_cons, "cons_detach": cfg.cons_detach,
        "graft_route": cfg.graft_route, "seam_clamp": cfg.seam_clamp,
        "graft_goal": cfg.graft_goal,
        "goal_provenance": RefCModel.goal_provenance() if cfg.graft_goal
        else None,
        "ego_valid_channel": cfg.ego_valid_channel,
        "labels": args.labels, "mode": args.mode,
        "n_selection_params": param_breakdown(model)["selection"],
        "n_goal_params": param_breakdown(model)["goal"],
        # S1 is inert at 0 denoise steps BY CONSTRUCTION — say so rather than
        # let a `--mode classifier --sel-refined` run look like a live arm.
        "s1_inert_because_classifier_mode": (cfg.sel_refined
                                             and args.mode == "classifier"),
    }
    print(json.dumps({"d_sel": dsel_row}), flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"arch": "REF-C (Anchored-Diffusion-C)",
         "cfg": dataclasses.asdict(cfg), "args": vars(args),
         "optimizer": {"kind": "Adam (DiffusionDrive/TCP)", "lr": lr,
                       "warmup": warmup, "schedule": "cosine (main run's)"},
         "loss_weights": {"traj": TRAJ_WEIGHT, "cls": ANCHOR_CLS_WEIGHT,
                          "law": LAW_WEIGHT, "route": ROUTE_WEIGHT,
                          "man": MANEUVER_WEIGHT, "speed_cls": SPEED_CLS_WEIGHT,
                          **({"lat": LAT_WEIGHT, "lon": LON_WEIGHT,
                              "man_total_held_at": MANEUVER_WEIGHT}
                             if cfg.factored_maneuver else {})},
         # Label provenance — the artifact must describe its own labels.
         "labels": {
             "label_set": args.labels,
             "route_derivation": (
                 "refb_labels.route_from_future_v21" if args.labels == "v21"
                 else "refb_labels.route_from_future_v3 (v21 CE target "
                      "byte-identical; TOKEN + distance-to-maneuver ADDED, no "
                      "head consumes them yet)" if args.labels == "v3"
                 else "refb_labels.route_target(nav_command)"),
             "use_net_dyaw": (bool(args.use_net_dyaw)
                              if args.labels in ("v21", "v3") else None),
             "nav_cmd_derivation": "refb_labels.nav_command (v1, unchanged)",
             "lan": ({"derivation": "tanitad.data.lan.lan_window_features "
                                    "(S1 ego_future, arc-length resample)",
                      "route_dropout": args.route_dropout,
                      "stats": lan_stats} if args.graft_lan else None),
             "maneuver_derivation": ("refb_labels.window_maneuver_labels "
                                     "(v1 5-way) — UNCHANGED in every label "
                                     "set; the LAT x LON factorization is a "
                                     "MODEL change (retrain), specified in "
                                     "V3_FACTORIZED_TACTICAL_HEAD_SPEC.md"),
             "route_unknown_handling": ("masked out of the route CE "
                                        "(ROUTE_UNKNOWN=3, never clamped)"
                                        if args.labels in ("v21", "v3")
                                        else "n/a"),
             "train_label_stats": label_stats},
         # D-SEL axes + the goal-provenance declaration. The artifact must
         # describe its own goal, exactly as it already describes its own labels:
         # the PI's admissibility ruling is checkable from the run directory
         # alone, without re-reading source.
         "d_sel": dsel_row,
         "data": {"cache_dir": str(train_dir), "n_episodes": len(train_eps),
                  "n_windows": len(ds)},
         "param_breakdown": param_breakdown(model)},
        indent=2, default=str), encoding="utf-8")

    # Interruptible-pod resume (refb_train convention).
    step = 0
    ckpt_path = out_dir / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"]) + 1
        print(f"[resume] checkpoint found — resuming at step {step}",
              flush=True)

    data_iter = iter(dl)
    t_data = t_step = 0.0
    last_log: dict = {}
    _sel_checked = False        # D-SEL dead-parameter guard runs once
    while step < args.steps:
        cur_lr = cosine_lr(step, args.steps, warmup, lr)
        for pg in opt.param_groups:
            pg["lr"] = cur_lr
        t_s0 = time.perf_counter()
        t_d0 = time.perf_counter()
        try:
            batch_d = next(data_iter)
        except StopIteration:
            data_iter = iter(dl)
            batch_d = next(data_iter)
        t_data += time.perf_counter() - t_d0

        opt.zero_grad(set_to_none=True)
        out = compute_losses(model, batch_d, device, mode=args.mode)
        out["loss"].backward()
        if not _sel_checked and (cfg.graft_cons or cfg.graft_route
                                 or cfg.graft_goal):
            print(json.dumps({"d_sel_gradients": {
                k: round(v, 8) for k, v in
                assert_selection_params_are_alive(model).items()}}), flush=True)
            _sel_checked = True
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        t_step += time.perf_counter() - t_s0

        if step > 0 and step % save_every == 0:
            _save_ckpt(ckpt_path, model, opt, step)

        if step % log_every == 0 or step == args.steps - 1:
            sc = lambda t: round(float(t.detach()), 5)  # noqa: E731
            last_log = {
                "step": step, "loss": sc(out["loss"]),
                "traj": sc(out["traj"]), "cls": sc(out["cls"]),
                "law": sc(out["law"]), "route": sc(out["route"]),
                "man": sc(out["man"]), "speed_cls": sc(out["speed_cls"]),
                "speed_mae": sc(out["speed_mae"]),
                "anchor_acc": sc(out["anchor_acc"]), "man_acc": sc(out["man_acc"]),
                "route_acc": sc(out["route_acc"]),
                "route_valid_frac": sc(out["route_valid_frac"]),
                "nav_follow_frac": sc(out["nav_follow_frac"]),
                # LAN arm only: a dead route input must be visible in the FIRST
                # log line, not inferred after a 30 k run.
                **({"lan_valid_frac": sc(out["lan_valid_frac"])}
                   if "lan_valid_frac" in out else {}),
                # D-SEL: the ranking diagnostic and the seam telemetry. Same
                # rule as lan_valid_frac — a graft that swamps the score, or a
                # ranking that never improves, must be visible in the FIRST log
                # line rather than reconstructed from a 30 k post-mortem. The
                # seam ratios were already computed and thrown away before D-SEL
                # (graft_lat_norm/conf_norm were logged with no actuator).
                **{k: sc(out[k]) for k in
                   ("cls_refined", "oracle_ade", "sel_ade", "sel_gap",
                    "rank_acc", "frac_sel_2x_worse", "goal_dir", "goal_dist",
                    "goal_valid_frac", "goal_gate", "goal_dist_gate")
                   if k in out},
                **out.get("sel_tele", {}),
                "gnorm": round(gnorm, 4), "lr": cur_lr,
                "data_s": round(t_data, 1), "step_s": round(t_step, 1),
            }
            t_data = t_step = 0.0
            print(json.dumps(last_log), flush=True)
        step += 1

    _save_ckpt(ckpt_path, model, opt, step - 1)     # final resume point
    metrics = {"final": last_log, "steps": step, "device": device,
               "param_breakdown": param_breakdown(model),
               "n_params_trainable": sum(p.numel() for p in model.parameters()
                                         if p.requires_grad)}
    # Light val row (REAL-only val dir), if present.
    try:
        val_eps, _ = load_cached_episodes(args.data_root, "*val*",
                                          min(args.episodes or 8, 8))
        vkw = dict(window=cfg.window, max_horizon=max_h,
                   channels=cfg.encoder.in_channels)
        vcls = {"v21": RouteV21Dataset, "v3": RouteV3Dataset}.get(args.labels)
        # ⚠️ The val set MUST carry the same route input as train, or the arm is
        # evaluated on a route it never receives — which is exactly the
        # nav_cmd=None confound (RETRACTION_LOG C6) reintroduced by omission.
        if want_lan_field:
            vcls = lan_dataset_class(vcls if vcls is not None
                                     else FailLoudWindowDataset)
            vkw = dict(vkw, lan_cfg=lan_cfg)
        vds = (vcls(val_eps, use_net_dyaw=args.use_net_dyaw, **vkw)
               if vcls is not None and issubclass(vcls, RouteV21Dataset)
               else vcls(val_eps, **vkw) if vcls is not None
               else FailLoudWindowDataset(val_eps, **vkw))
        model.eval()
        with torch.no_grad():
            vb = torch.utils.data.default_collate(
                [vds[i] for i in range(min(16, len(vds)))])
            vout = compute_losses(model, vb, device, mode=args.mode)
        metrics["val"] = {k: round(float(vout[k]), 5)
                          for k in ("traj", "cls", "law", "route", "man",
                                    "speed_cls", "speed_mae", "anchor_acc",
                                    "man_acc", "route_acc", "route_valid_frac",
                                    "nav_follow_frac")}
        # D-SEL selection diagnostic on val, when the arm carries it.
        metrics["val"].update({k: round(float(vout[k]), 5) for k in
                               ("cls_refined", "oracle_ade", "sel_ade",
                                "sel_gap", "rank_acc", "frac_sel_2x_worse")
                               if k in vout})
        if "lan_valid_frac" in vout:
            metrics["val"]["lan_valid_frac"] = round(
                float(vout["lan_valid_frac"]), 5)
    except AssertionError:
        pass                                        # no val cache dir
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2),
                                          encoding="utf-8")
    print(json.dumps({"done": True, "steps": step, "out": str(out_dir)}),
          flush=True)
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="epcache root containing *train*/*val* dirs of "
                         "ep_*.pt (the train_worldmodel --data cached layout)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--mode", choices=("classifier", "diffusion"),
                    default="diffusion",
                    help="decoder mode: classifier = 0-step anchor selection "
                         "floor; diffusion = truncated-denoise refinement")
    ap.add_argument("--config", choices=("small", "base", "xl"), default="base",
                    help="scale preset (ignored under --smoke): small ~55M "
                         "(64 anchors) / base ~110M (128) / xl ~260M (256)")
    ap.add_argument("--anchors", default=None,
                    help="FPS anchor vocabulary .pt (build_refc_anchors.py); "
                         "default = the model's built-in synthetic-FPS anchors")
    ap.add_argument("--labels", choices=("v1", "v21", "v3"), default="v1",
                    help="route AUX target derivation: v1 = what REF-C-XL "
                         "trained with (route_target(nav_cmd), straight-by-"
                         "default); v21 = refb_labels.route_from_future_v21 "
                         "(adaptive horizon, ROUTE_UNKNOWN masked out of the "
                         "CE); v3 = v21 PLUS the frozen-vocabulary route TOKEN, "
                         "the distance-to-maneuver band and the factorized "
                         "LAT/LON tactical slots as EXTRA batch fields (CE "
                         "targets byte-identical to v21; the new fields have no "
                         "head yet — see V3_FACTORIZED_TACTICAL_HEAD_SPEC.md)")
    ap.add_argument("--use-net-dyaw", action="store_true",
                    help="v21 only: count a >=45 deg net heading change as a "
                         "route turn. OFF per Sayed 2026-07-20 (a wide sweep is "
                         "ROAD FOLLOWING)")
    ap.add_argument("--batch", type=int, default=None,
                    help="default: the main run's batch (base250cam)")
    ap.add_argument("--lr", type=float, default=None,
                    help=f"default: the TCP/DiffusionDrive Adam lr ({TCP_LR})")
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--warmup", type=int, default=None,
                    help="default: the main run's warmup (base250cam, 2000)")
    ap.add_argument("--graft-lan", action="store_true",
                    help="LAN: feed a lane-anchored route corridor alongside "
                         "nav_cmd (see PREREG_lan_refc.md). Additive and "
                         "zero-init: OFF is byte-identical to a model without "
                         "it, so this flag alone defines the LAN arm")
    ap.add_argument("--lan-arclengths", type=float, nargs="+",
                    default=[20.0, 40.0, 80.0, 160.0],
                    help="LAN route-anchor arc-lengths in metres (ascending)")
    ap.add_argument("--lan-min-lead-m", type=float, default=5.0,
                    help="extra leak-guard margin past the 2 s path length")
    ap.add_argument("--route-dropout", type=float, default=0.5,
                    help="per-sample Bernoulli mask of the LAN route (train "
                         "only) so the planner never becomes route-DEPENDENT")
    # --- D-TAC1: the factorised tactical head (three separable levers) -------
    ap.add_argument("--factored-maneuver", action="store_true",
                    help="F2 STRUCTURE: replace the mixed 5-way tactical "
                         "softmax with independent lateral(3) x longitudinal(3) "
                         "heads + split lat/lon anchor grafts (lon zero-init). "
                         "maneuver_logits is still emitted, derived exactly.")
    ap.add_argument("--tactical-speed-input", action="store_true",
                    help="F1 INPUT: the tactical head reads the ego speed. The "
                         "shipped head reads the image embedding alone while "
                         "its own label is dv = v(t+2s) - v(t). Usable WITH or "
                         "WITHOUT --factored-maneuver; alone it is the "
                         "INPUT-only arm (refc_f1only_config).")
    ap.add_argument("--man-prior-tau", type=float, default=0.0,
                    help="F3 DECISION: logit-adjustment strength for the "
                         "REPORTED tactical class (1.0 = balanced posterior). "
                         "0 = today's argmax. Requires --factored-maneuver.")
    ap.add_argument("--no-graft-prior-center", action="store_true",
                    help="feed the anchor grafts the raw log-posterior instead "
                         "of the log-likelihood ratio (ablation).")
    # --- D-SEL: the SELECTION surface (see tanitad/refs/refc_select.py) ------
    ap.add_argument("--sel-refined", action="store_true",
                    help="S1: rank the refined fan with the REFINED confidence "
                         "and supervise it (adds the cls_refined CE). Today the "
                         "post-denoise trajectories are ranked by the t=0 "
                         "classifier score. Inert at --mode classifier.")
    ap.add_argument("--sel-reach-clamp", action="store_true",
                    help="S2: restrict the argmax to candidates a bounded-"
                         "acceleration ego could fly. MEASURED INERT on ADE "
                         "(paired delta exactly 0.0) — it is a precondition "
                         "that makes per-candidate compute 3.58x cheaper, not "
                         "an improvement. Never applied where ego-dropout "
                         "withheld v0.")
    ap.add_argument("--sel-accel-max", type=float, default=2.5,
                    help="m/s^2 for that band (flagship v1.5's own constant)")
    ap.add_argument("--graft-cons", action="store_true",
                    help="S3: score each candidate by its CONSEQUENCE through "
                         "law_head (REF-C's trajectory-conditioned world model) "
                         "and let that reach the ranking. +1 parameter; the "
                         "world model runs under no_grad.")
    ap.add_argument("--graft-route", action="store_true",
                    help="S5: the strategic route READOUT reaches the ranked "
                         "score (zero-init). REQUIRES --labels v21/v3 — under "
                         "v1 the route target is circular with nav_cmd.")
    ap.add_argument("--graft-goal", action="store_true",
                    help="S6: a PREDICTED GEOMETRIC goal (bearing + signed "
                         "along-track preference) reaches the ranked score "
                         "through the SAME param-free geometric compatibility "
                         "LAN uses, on TWO independent zero-init gates. The "
                         "head reads the IMAGE EMBEDDING ONLY; the LAN corridor "
                         "is its TRAINING LABEL and nothing else, so this "
                         "REQUIRES --graft-lan. Geometric-and-predicted, per "
                         "the PI ruling of 2026-08-03 — never categorical and "
                         "never supplied. Carries no situation-classifier "
                         "output in any form (RefCModel.goal_provenance()).")
    ap.add_argument("--seam-clamp", type=float, default=0.0,
                    help="S4: cap the TOTAL anchor graft at this multiple of "
                         "the base score's norm (<=0 disables; 1.0 is v4's). "
                         "Below the cap the rescale is exactly 1.0.")
    ap.add_argument("--ego-valid-channel", action="store_true",
                    help="X15: feed an explicit 'v0 is present' flag beside the "
                         "ego-dropped speed, to the measurement encoder and (if "
                         "--tactical-speed-input) the tactical head. 0.0 m/s is "
                         "in-distribution 'stationary', so the zero-fill is a "
                         "confident lie.")
    ap.add_argument("--refc1", action="store_true",
                    help="REF-C.1: fixed-distance path checkpoints at "
                         "(2,5,10,20) m + target-speed classification head")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers (0 = in-loop decode, old behavior)")
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny config (CI/CPU smoke; 1-channel 64 px episodes)")
    args = ap.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
