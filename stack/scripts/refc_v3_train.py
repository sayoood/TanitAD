"""REF-C v3 trainer — the goal-mediated hierarchy on the supervised arm.

Design: ``TanitAD Research Lab/Architecture & Inference/Research/
2026-08-18-refc-v3-design/REFC_V3_DESIGN.md`` (edge list E1..E12).
Experiment: ``PREREG_REFC_V3.md`` (E-V3DOM-1 — arms v3-H / v3-F, 3 seeds,
both outcomes committed). ⛔ Registered BEFORE any training step; do not launch
while Thor trains.

WHAT THIS SCRIPT IS AND IS NOT
============================================================================
A THIN COMPOSITION over ``refc_train.py``'s measured machinery — the same
fail-loud dataset stack (``RouteV21Dataset``: v2.1 labels, the measured label
choice), the same optimizer convention (Adam lr 1e-4 / warmup / cosine — the
arm's point), the same loss weights for every SHARED surface (imported, not
copied, so the two trainers cannot drift apart silently). What is NEW here and
only here:

  * ``V3Dataset`` — the PARITY-PRESERVING 6 s horizon. ``max_horizon`` stays 20
    for window ENUMERATION (``_contract.py:120`` re-selects windows otherwise —
    the canonical 406,099 must stay bit-identical), and steps 21..60 arrive by
    CLAMP + per-step validity mask. Tactical goal labels via
    ``refb_labels.goal_tac_targets`` (E4.1 layout, clamp+valid by contract).
  * ``compute_losses_v3`` — masked trajectory/assignment losses over the 8-slot
    horizon (exact-zero gradient at invalid slots, pinned by
    ``tests/test_refc_v3.py``), the factored tactical CE on BOTH decision
    surfaces (core aux — identical in both arms — and the z_tac heads on the H
    arm), the masked goal losses, and the survivor-set selection CE.
  * ``--arm hier|flat`` — the dominance pair. The config delta between the two
    is DERIVED and REFUSED if it is not exactly the registered lever set
    (C122's rule, enforced at build, not documented).
  * ``--preflight`` — builds everything, pins the delta, runs the C115
    freeze-history gate and the E11 mini intervention audit, runs one synthetic
    forward+loss, and EXITS. The launch line runs this first; a flag that
    parses and then dies mid-run is the class that cost ~3.1 GPU-days.
  * ``--synth-episodes N`` — CI-only synthetic corpus so the smoke test runs
    end-to-end with zero data. REFUSED together with ``--data-root``.

Done-marker discipline: on completion this trainer writes ``summary.json`` with
``{"done": true}`` IN THE SAME RUN — the v5f supervisor resurrection (a
finished run relaunched for 2 days) is the reason.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The preflight's own banners carry ⛔/✅. On a cp1252 console (the Windows dev
# box, where a preflight is MEANT to be run before spending a GPU day) printing
# them raises UnicodeEncodeError — MEASURED: the run below died on the "✅ PASS"
# line AFTER every check had passed, so a PASSING preflight exited non-zero with
# a traceback. A gate whose success path crashes is a false negative.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):        # already utf-8, or not a TTY
        pass

import refb_labels  # noqa: E402
from refc_train import (  # noqa: E402  — SHARED surfaces, imported not copied
    ANCHOR_CLS_WEIGHT, LAT_WEIGHT, LAW_AHEAD, LAW_WEIGHT, LON_WEIGHT,
    ROUTE_WEIGHT, TRAJ_WEIGHT, RouteV21Dataset, lan_dataset_class,
)
from refb_train import load_cached_episodes  # noqa: E402

from tanitad.data.lan import LanConfig as DataLanConfig  # noqa: E402

from tanitad.refs import refc  # noqa: E402
from tanitad.refs import refc_tactical as tac  # noqa: E402
from tanitad.data import v7_labels as v7l
from tanitad.refs import refc_v3 as v3  # noqa: E402

# --- v3-only loss weights (everything shared is imported above) --------------
#: tactical goal regression (E8) — sized with the route/maneuver aux family;
#: the goal is a 12-number readout, not a second trajectory head.
GOAL_TAC_WEIGHT = 0.5
#: strategic goal (E3) — the ROUTE aux family's weight, deliberately: it is the
#: same kind of signal (leak-guarded LAN label) on a different head.
GOAL_STR_WEIGHT = ROUTE_WEIGHT
#: survivor-set selection CE (E9) — parallel to ANCHOR_CLS_WEIGHT: it is the
#: same "which candidate" question asked of the blended score.
SEL_V3_WEIGHT = 1.0
#: the registered dominance lever set — build REFUSES any other delta (C122).
REGISTERED_DELTA_KEYS = {"hier", "core.graft_target_latent"}

MILESTONES = (5000, 15000, 20000, 30000)
MAX_H_EXT = max(v3.V3_HORIZONS)            # 60 — fetched by clamp, never enum


def _pin_trainer_cfg(cfg: v3.RefCV3Config, args) -> v3.RefCV3Config:
    """The trainer's OWN pins on a freshly built config (both arms, all paths).

    * ``tac_vocab_version = "kin3"`` — THE documented contract this trainer
      never wired (found 2026-09-01): ``RefCV3Config``'s field comment says
      *"the kinematic trainer passes 'kin3' explicitly because its labels are
      the 3x3 kinematic classes"*, and the E6 comment says *"The trainer PINS
      its version"* — but nothing pinned it, so the v7.0 default (PI mandate
      2026-08-27) built 8-wide z_tac heads that ``compute_losses_v3``
      supervised with 3-class kinematic labels. That is EXACTLY the *"train
      silently WRONG classes"* case the width refusal names, invisible to it
      because the refusal checked only the CORE's head — and
      ``derive_man5_logprobs`` (a [B,3]x[B,3] contract) then read v7.0's
      LANE_CHANGE_L/R slots as turn_left/right into the H19 prior, an improper
      AND mis-labelled 5-way. MEASURED before the pin: preflight PASSED with
      ``tac_heads 14,364`` and ``lat_tac``/``lon_tac`` init-CE at ln(8).
      The v7.0 head is the go-forward space and NEEDS v7 labels
      (``SPEC_V7_LABEL_TRAINER_WIRING.md`` — label artifacts not yet
      deliverable), so a flag would be a dead switch today; pin, don't offer.
    * ``--image-hw`` — build the encoder at the CORPUS's geometry (e.g.
      256 640 for the B1 ``*.v2ep.pt`` cache). Param count is UNCHANGED (the
      trunk is fully convolutional; feat_dim = base_width*8 regardless), so
      the registered capacity ledger still holds; only compute changes.
      The delta gate is unaffected: both arms are pinned identically, and
      ``config_delta`` is derived from configs built through this same helper.
    """
    # ⭐ The vocabulary follows THE LABELS, not a hardcode. With --v7-labels
    # the released v7.2 tactical vocabulary (8x8) supervises the heads (PI
    # 2026-09-02, MANDATORY); otherwise the kinematic 3x3 derivation does, and
    # pinning it here is what stopped an 8-wide build training on 3 classes.
    cfg.tac_vocab_version = ("v7.0" if getattr(args, "v7_labels", None)
                             else "kin3")
    if args.image_hw:
        h, w = (int(args.image_hw[0]), int(args.image_hw[1]))
        enc = cfg.core.encoder
        cfg.core.encoder = refc.CNNEncoderConfig(
            in_channels=enc.in_channels, image_size=h,
            image_width=None if w == h else w,
            base_width=enc.base_width, blocks=enc.blocks)
    return cfg


# ============================================================================
# Dataset — parity-preserving 6 s
# ============================================================================

class V3Dataset(RouteV21Dataset):
    #: clip-stable-id -> V7Label, or None for the kin3 path. Set by the trainer
    #: rather than passed through the ctor, because the base class owns the
    #: signature and widening it would touch every RouteV21 consumer.
    v7_by_sid: dict | None = None
    v7_dt: float = 0.1

    """RouteV21Dataset + clamped/masked 6 s future + E4.1 tactical goals.

    ``max_horizon`` MUST stay at the caller's 20: enumeration parity. The
    extended future is fetched here per item from the episode's own poses."""

    def __getitem__(self, i: int):
        item = super().__getitem__(i)
        e_i, t = self.index[i]
        ep = self.episodes[e_i]
        w = self.window
        T = ep.poses.shape[0]
        idx = torch.arange(t + w, t + w + MAX_H_EXT)
        item["future_poses_ext"] = ep.poses[idx.clamp(max=T - 1)]   # [60, 4]
        item["future_valid_ext"] = idx <= (T - 1)                   # [60] bool
        g, gv = refb_labels.goal_tac_targets(ep.poses, t + w - 1,
                                             v3.GOAL_TAU_STEPS)
        item["goal_tac"] = g                                        # [K, 4]
        item["goal_tac_valid"] = gv                                 # [K] bool
        # ---- v7.2 tactical labels (PI 2026-09-02: MANDATORY) --------------
        # ⭐ Joined on `stable_episode_id(clip_id)` — the v7.2 clip index names
        # it "the ONLY admissible join key", and `LazyV2Episode` carries the
        # stable id (not the clip_id string), so this is the join the artifacts
        # were built for rather than a string match we invented.
        # The window's NOW is the last OBSERVED frame: t + w - 1 at dt.
        if self.v7_by_sid is not None:
            lab = self.v7_by_sid.get(int(ep.episode_id))
            if lab is None:
                lat_v7 = lon_v7 = v7l.IGNORE_ID       # clip has no record
            else:
                lat_v7, lon_v7 = v7l.tactical_class_ids(
                    lab, (t + w - 1) * self.v7_dt)
            item["lat_v7"] = torch.tensor(lat_v7, dtype=torch.long)
            item["lon_v7"] = torch.tensor(lon_v7, dtype=torch.long)
        return item


def _synth_episodes(n: int, cfg: refc.RefCConfig, seed: int = 0,
                    min_frames: int = 40):
    """CI-only synthetic corpus (unicycle drives, tiny frames). NEVER a
    substitute for the parity cache — refused alongside --data-root.

    ``min_frames`` exists because the DEFAULT corpus is too SHORT to carry a
    LAN route and that is not obvious from reading it: T=40 at 2-8 m/s is
    ~8-32 m of total path, while the shortest LAN anchor sits at 20 m ARC-LENGTH
    *beyond* a leak guard of ~2 s x v + 5 m. MEASURED (D-LAN-COV, this package):
    every anchor of every window is masked on the default corpus, so a LAN
    preflight built on it would compute ``goal_str`` over an all-invalid label
    and report a green 0.0 — a loss that exists and cannot learn. The LAN arm
    therefore asks for a corpus long enough for the question to be answerable.
    """
    import types
    g = torch.Generator().manual_seed(seed)
    h, wpx = cfg.encoder.image_hw()
    eps = []
    for e in range(n):
        T = max(min_frames, 40) + 8 * e
        v = 2.0 + 6.0 * torch.rand((), generator=g)
        yr = (torch.rand((), generator=g) - 0.5) * 0.4
        yaw = torch.cumsum(torch.full((T,), float(yr)) * 0.1, dim=0)
        xy = torch.cumsum(
            torch.stack([v * torch.cos(yaw), v * torch.sin(yaw)], -1) * 0.1, 0)
        poses = torch.cat([xy, yaw[:, None],
                           torch.full((T, 1), float(v))], dim=-1)
        eps.append(types.SimpleNamespace(
            frames=(torch.rand(T, cfg.encoder.in_channels, h, wpx,
                               generator=g) * 255).to(torch.uint8),
            actions=torch.zeros(T, 2),
            poses=poses,
            episode_id=f"synth-{e:03d}"))
    return eps


# ============================================================================
# Losses — masked 6 s + hierarchy terms
# ============================================================================

def compute_losses_v3(model: v3.RefCV3Model, batch: dict, device: str,
                      mode: str = "diffusion") -> dict:
    cfg = model.cfg
    core = cfg.core
    frames = batch["frames"].to(device)
    fut_frames = batch["future_frames"].to(device)
    fut_ext = batch["future_poses_ext"].to(device)          # [B, 60, 4]
    fut_valid = batch["future_valid_ext"].to(device)        # [B, 60] bool
    pose_last = batch["pose_last"].to(device)
    nav_cmd = batch["nav_cmd"].to(device)
    nav_valid = batch["nav_valid"].to(device)
    route_tgt = batch["route_target"].to(device)
    goal_tac = batch["goal_tac"].to(device)                 # [B, K, 4]
    goal_valid = batch["goal_tac_valid"].to(device)         # [B, K] bool
    lan = batch["lan"].to(device) if "lan" in batch else None
    v0 = pose_last[:, 3]
    b = frames.shape[0]
    steps = core.decoder.diffusion_steps if mode == "diffusion" else 0

    out = model(frames, nav_cmd=nav_cmd, v0=v0, steps=steps, lan=lan)

    # ---- trajectory target over the 8-slot 6 s horizon, masked -------------
    traj_tgt = refb_labels.waypoint_targets(pose_last, fut_ext,
                                            core.trajectory.horizons)
    slot_valid = torch.stack([fut_valid[:, h - 1]
                              for h in core.trajectory.horizons], dim=1)
    sv = slot_valid.to(traj_tgt.dtype)                      # [B, S]
    anchors = model.core.decoder.anchors.to(traj_tgt.dtype)  # [N, S, 2]
    dist = (((traj_tgt[:, None] - anchors[None]) ** 2).sum(-1)
            * sv[:, None]).sum(-1)                          # [B, N] valid-only
    a_star = dist.argmin(dim=1)
    ar = torch.arange(b, device=device)
    loss_cls = F.cross_entropy(out["anchor_logits"], a_star)
    recon = out["anchor_traj"][ar, a_star]                  # [B, S, 2]
    denom = (sv.sum() * 2).clamp_min(1.0)
    loss_traj = (((recon - traj_tgt).abs().sum(-1)) * sv).sum() / denom

    # ---- factored tactical CE (2 s labels; the shared aux surface) ---------
    # ⭐ TWO SUPERVISION SOURCES, ONE REFUSAL EACH (PI 2026-09-02 made v7.2
    # MANDATORY for this launch). `lat_v7` present => the released v7.2
    # tactical vocabulary (8x8), joined per window by stable episode id;
    # absent => the runtime kin3 derivation (3x3) from poses. The head width
    # must match ITS OWN source, and a mismatch is REFUSED, never truncated --
    # a wider head fed narrower labels trains silently wrong classes, which is
    # exactly the defect found in this trainer on 2026-09-01.
    # ⛔⛔ TWO TACTICAL SURFACES, TWO VOCABULARIES — AND THE CORE'S IS FIXED.
    # MEASURED 2026-09-02: the core's heads are `nn.Linear(aux_hidden,
    # N_LAT_MAN)` with `N_LAT_MAN = tac.N_LAT` a MODULE-LEVEL CONSTANT
    # (refc.py:146,1742) — the REF-C core is STRUCTURALLY kin3, and
    # `tac_vocab_version` only sizes v3's OWN z_tac heads. Widening the core
    # would also have to redefine `derive_man5_logprobs` (a push-forward
    # DEFINED on the 3x3 kinematic vocabulary) and the H19 `lat_to_anchor`
    # prior — a change to the shared core, not a wiring job.
    # ⇒ v7.2 supervises the surface it can actually reach: the HIERARCHY's
    # tactical decision heads (z_tac). The core's legacy aux head keeps its
    # kinematic derivation. Both are trained; they are NOT the same label set,
    # and the launch record says so rather than implying one vocabulary.
    lat_k, lon_k = tac.window_factored_labels(pose_last, fut_ext[:, :20])
    if (out["lat_logits"].shape[-1] != tac.N_LAT
            or out["lon_logits"].shape[-1] != tac.N_LON):
        raise RuntimeError(
            f"CORE head/label mismatch: heads are "
            f"{out['lat_logits'].shape[-1]}x{out['lon_logits'].shape[-1]} vs "
            f"kinematic {tac.N_LAT}x{tac.N_LON}. The core surface is kin3 by "
            f"construction (refc.py N_LAT_MAN); only the v3 z_tac heads follow "
            f"tac_vocab_version.")
    loss_lat = F.cross_entropy(out["lat_logits"], lat_k)
    loss_lon = F.cross_entropy(out["lon_logits"], lon_k)
    model.core.update_tactical_prior(lat_k, lon_k)
    # the z_tac surface's labels: v7.2 when supplied, else the same kin3
    use_v7 = "lat_v7" in batch
    if use_v7:
        lat_t, lon_t = batch["lat_v7"].to(device), batch["lon_v7"].to(device)
        n_lat_expect, n_lon_expect = (len(v7l.HEADS["tac_lat"]),
                                      len(v7l.HEADS["tac_lon"]))
        src = f"v7.2 ({n_lat_expect}x{n_lon_expect})"
    else:
        lat_t, lon_t = lat_k, lon_k
        n_lat_expect = n_lon_expect = tac.N_LAT
        src = f"kin3 ({tac.N_LAT}x{tac.N_LON})"
    # ⛔ -100 marks a window this clip's single record does not describe. CE
    # ignores those rows; an ALL-ignored batch is NaN, not zero, so it is
    # SKIPPED (the same guard refav1 needed -- measured, not assumed).
    lat_ok = int((lat_t != v7l.IGNORE_ID).sum())
    lon_ok = int((lon_t != v7l.IGNORE_ID).sum())
    loss_lat_tac = torch.zeros((), device=device)
    loss_lon_tac = torch.zeros((), device=device)
    if cfg.hier:                       # the H arm's z_tac decision surface
        # the SAME refusal for the v3 heads — the core-only check above let an
        # 8-wide v7.0 build train silently against kin3 labels (2026-09-01).
        if (out["lat_logits_tac"].shape[-1] != n_lat_expect
                or out["lon_logits_tac"].shape[-1] != n_lon_expect):
            raise RuntimeError(
                f"z_tac head/label vocabulary mismatch: lat "
                f"{out['lat_logits_tac'].shape[-1]} / lon "
                f"{out['lon_logits_tac'].shape[-1]} vs {src} -- the model was "
                f"built with tac_vocab_version={model.tac_vocab_version!r}. "
                f"The z_tac heads and the core heads MUST share one source; "
                f"this check mirrors the core one because a v3 build once "
                f"passed the core check and trained 8-wide z_tac heads "
                f"against 3-class labels (2026-09-01).")
        loss_lat_tac = (F.cross_entropy(out["lat_logits_tac"], lat_t) if lat_ok
                        else torch.zeros((), device=device))
        loss_lon_tac = (F.cross_entropy(out["lon_logits_tac"], lon_t) if lon_ok
                        else torch.zeros((), device=device))

    # ---- route aux (v2.1 masked CE — refc_train's convention) --------------
    # ⛔⛔ FIXED 2026-09-02, MEASURED ON THE A40: this masked by `nav_valid`,
    # which is a DIFFERENT VALIDITY FIELD than the target it guards. v2.1's
    # `route_target` is ROUTE_UNKNOWN (= 3, deliberately OUTSIDE the 3-class CE
    # range) exactly where `route_valid` is False — and a window can carry a
    # VALID nav command with an UNKNOWN route. Those rows survived the
    # nav_valid mask and hit the CE, producing
    #   `nll_loss_forward_reduce_cuda_kernel_2d: t >= 0 && t < n_classes`
    # — an async device-side assert that surfaces at the NEXT CUDA call
    # (conv2d), so the traceback blames the wrong line entirely.
    # ⚠️ WHY IT SURVIVED EVERY SMOKE: it is window-frequency dependent. The
    # 2-step batch-2 smoke and the batch-4 run never drew an offending window;
    # batch 20 hit it inside 14 steps. A 30 k launch would have died hours in.
    # ⇒ mask on `route_valid` when the v2.1 labeler supplies it (mirroring
    # `refc_train.py`'s shared path verbatim), and keep the FAIL-LOUD check —
    # an UNKNOWN surviving the mask is a labeler contract violation, never
    # something to clamp to `straight`.
    if "route_valid" in batch:
        mask = batch["route_valid"].to(device)
        if bool(mask.any()):
            tgt_v = route_tgt[mask]
            n_route = out["route_logits"].shape[-1]
            if int(tgt_v.max()) >= n_route:
                raise ValueError(
                    f"ROUTE_UNKNOWN survived the valid mask (max target "
                    f"{int(tgt_v.max())} >= n_route {n_route}) — the v2.1 "
                    f"contract is route<3 <=> valid=True")
            loss_route = F.cross_entropy(out["route_logits"][mask], tgt_v)
        else:                        # no judgeable window in this batch
            loss_route = torch.zeros((), device=device)
    else:
        mask = nav_valid
        loss_route = (F.cross_entropy(out["route_logits"][mask],
                                      route_tgt[mask])
                      if bool(mask.any())
                      else torch.zeros((), device=device))

    # ---- LAW aux (0.5 s pooled-latent target, no_grad encode) --------------
    with torch.no_grad():
        law_tgt = model.core.encode_pooled(fut_frames[:, LAW_AHEAD - 1]
                                           .to(device))
    loss_law = F.mse_loss(out["law_pred"], law_tgt)

    loss = (TRAJ_WEIGHT * loss_traj + ANCHOR_CLS_WEIGHT * loss_cls
            + LAW_WEIGHT * loss_law + ROUTE_WEIGHT * loss_route
            + LAT_WEIGHT * (loss_lat + loss_lat_tac)
            + LON_WEIGHT * (loss_lon + loss_lon_tac))

    extra: dict = {}
    # ⭐ WHICH label set trained the tactical decision surface, and how many
    # rows survived the in-band mask, IN EVERY LOG ROW. A run that silently
    # fell back to kin3 (missing --v7-labels, or a corpus whose clips have no
    # record) would otherwise look identical to a v7.2 run in the log.
    extra["tac_label_v7"] = 1.0 if use_v7 else 0.0
    extra["tac_label_rows"] = float(lat_ok)
    if cfg.hier:
        # E8 — masked goal regression (each level trains by its OWN label).
        loss_goal = v3.masked_goal_loss(out["g_tac"], goal_tac, goal_valid)
        loss = loss + GOAL_TAC_WEIGHT * loss_goal
        extra["goal_tac"] = loss_goal
        with torch.no_grad():
            e2 = (out["g_tac"][:, model._tau_slot_2s(), :2]
                  - goal_tac[:, model._tau_slot_2s(), :2]).norm(dim=-1)
            m2 = goal_valid[:, model._tau_slot_2s()]
            extra["goal2s_err_m"] = (e2[m2].mean() if bool(m2.any())
                                     else torch.zeros((), device=device))
        # E3 — strategic goal off the leak-guarded LAN label (train-only, E12).
        if lan is not None:
            bearing_t, dist_t, valid_t = refc.RefCModel.goal_targets(
                lan, core.lan.k)
            loss_gstr = v3.strategic_goal_loss(out["g_str"], bearing_t,
                                               dist_t, valid_t)
            loss = loss + GOAL_STR_WEIGHT * loss_gstr
            extra["goal_str"] = loss_gstr
        # E9 — survivor-set CE on the blended score (the gates' ONLY gradient:
        # argmax has none; the goal head is firewalled by detach).
        fan_err = (((out["anchor_traj"] - traj_tgt[:, None]).norm(dim=-1)
                    * sv[:, None]).sum(-1)
                   / sv.sum(-1, keepdim=True).clamp_min(1.0))     # [B, N]
        loss_sel = v3.selection_ce(out["sel_score_v3"], fan_err.detach(),
                                   out.get("reach_keep"))
        loss = loss + SEL_V3_WEIGHT * loss_sel
        extra["sel_v3"] = loss_sel
        extra["goal_gate"] = model.goal_gate.detach()
        # ⭐ CAVEAT-B (PI 2026-09-02): the gate value alone cannot distinguish
        # "has not opened YET" from "will never open". Emit the score scale it
        # multiplies and its own gradient, so the 30 k read answers the
        # question with data instead of a 14-step glance.
        if "goal_score_absmean" in out:
            extra["goal_score_absmean"] = out["goal_score_absmean"]
        if model.goal_gate.grad is not None:
            extra["goal_gate_grad"] = model.goal_gate.grad.detach().abs()
        # E13 telemetry: nav actually reached the tactical/strategic states on
        # this batch. A conditioning edge that silently no-ops (nav_cmd=None
        # everywhere) is the advertised-but-inert defect; this makes it visible.
        if "nav_injected" in out:
            extra["nav_injected"] = float(bool(out["nav_injected"]))

    return {"loss": loss, "traj": loss_traj, "cls": loss_cls, "law": loss_law,
            "route": loss_route, "lat": loss_lat, "lon": loss_lon,
            "lat_tac": loss_lat_tac, "lon_tac": loss_lon_tac,
            "anchor_acc": (out["anchor_logits"].argmax(1) == a_star)
            .float().mean(), "slot_valid_frac": sv.mean(), **extra}


# ============================================================================
# Preflight — everything that must be true BEFORE a GPU day
# ============================================================================

def _lan_arm_preflight(cfg, args) -> int:
    """Exercise the LAN → ``goal_str`` pathway BEFORE a GPU day.

    ⚠️ THIS ARM EXISTS BECAUSE THE PREFLIGHT COULD NOT SEE THE THING IT WAS
    BEING CITED FOR. MEASURED (D-LAN-PF, this package): ``preflight()`` builds
    a bare ``V3Dataset``, which emits no ``lan`` key, so ``compute_losses_v3``
    skips ``loss_gstr`` BY DESIGN and ``goal_str`` never appears in the loss
    dict at all — with or without ``--goal-str``. The quantity people read
    instead, ``route``, is the v2.1 NAV-derived CE masked by ``nav_valid``; it
    is bit-identical with and without every LAN flag and can never testify
    about LAN. Same class as C9/C13/C14: an instrument structurally unable to
    report the answer it is quoted for.

    Three checks, in the order that makes a failure diagnosable:
      1. the dataset actually emits ``lan`` at the pinned width;
      2. the LABEL is live — ``any_valid_frac`` > 0. A ``goal_str`` computed
         over an all-invalid label is 0.0 and looks like a pass;
      3. ``goal_str`` is PRESENT, FINITE and NON-ZERO.
    Then a DELIBERATE-REGRESSION control (programme 6.2): with the leak guard
    pushed to +inf the label must go dead and ``goal_str`` must collapse to
    0.0. A check that cannot fail is not evidence, so the control runs every
    time and its own failure fails the preflight.
    """
    k = len(args.lan_arclengths)
    lan_cfg = DataLanConfig(arclengths_m=tuple(args.lan_arclengths),
                            min_lead_m=args.lan_min_lead_m)
    cfg.core.lan = refc.LanConfig(k=k)
    torch.manual_seed(0)
    model = v3.RefCV3Model(cfg)

    # Long enough that the leak guard is a GUARD and not a wall — see
    # _synth_episodes.__doc__ for the arithmetic that sets this floor.
    # ⚠️ Derived from the ARC-LENGTHS ONLY, deliberately NOT from min_lead_m:
    # the regression control below cranks min_lead_m to +inf, and a corpus that
    # grew with it would change TWO things at once — the control would compare
    # a different corpus rather than isolate the guard. The cap keeps a
    # pathological --lan-arclengths from asking for an unallocatable corpus
    # (at 4000 frames the reachable path is ~800 m, so anchors beyond that
    # still read dead, which is the correct verdict and not an evasion).
    need_m = max(args.lan_arclengths) + 50.0
    min_frames = min(int(need_m / (2.0 * 0.1)) + 4 * cfg.core.window, 4000)
    eps = _synth_episodes(2, cfg.core, seed=0, min_frames=min_frames)

    def _batch(lc):
        ds = lan_dataset_class(V3Dataset)(
            eps, window=cfg.core.window, max_horizon=20,
            channels=cfg.core.encoder.in_channels, lan_cfg=lc)
        return (torch.utils.data.default_collate([ds[0], ds[1]]),
                ds.lan_stats(n=512, seed=0))

    batch, stats = _batch(lan_cfg)
    if "lan" not in batch:
        print("[v3-preflight] ⛔ FAIL: lan dataset emitted no `lan` key")
        return 6
    if tuple(batch["lan"].shape[1:]) != (k * 4,):
        print(f"[v3-preflight] ⛔ FAIL: lan width {tuple(batch['lan'].shape)} "
              f"!= pinned {k * 4}")
        return 6
    print(f"[v3-preflight] lan label coverage: {json.dumps(stats)}")
    if stats["any_valid_frac"] <= 0.0:
        print("[v3-preflight] ⛔ FAIL: every LAN anchor is masked — `goal_str` "
              "would be 0.0 over an ALL-INVALID label and read as a pass. The "
              "strategic goal head would train on nothing (C9 class).")
        return 6

    losses = compute_losses_v3(model, batch, "cpu", mode="diffusion")
    gs = losses.get("goal_str")
    if gs is None:
        print("[v3-preflight] ⛔ FAIL: `goal_str` absent from the loss dict — "
              "the LAN pathway did not fire even though `lan` was present.")
        return 6
    gs = float(gs.detach())
    if not (gs == gs and abs(gs) != float("inf")):
        print(f"[v3-preflight] ⛔ FAIL: goal_str non-finite ({gs})")
        return 6
    if gs == 0.0:
        print("[v3-preflight] ⛔ FAIL: goal_str is exactly 0.0 on a LIVE label "
              "— the strategic head carries no gradient.")
        return 6

    # ---- deliberate-regression control: the check must be ABLE to fail ------
    dead_cfg = DataLanConfig(arclengths_m=tuple(args.lan_arclengths),
                             min_lead_m=1e9)
    dead_batch, dead_stats = _batch(dead_cfg)
    dead_gs = float(compute_losses_v3(model, dead_batch, "cpu",
                                      mode="diffusion")["goal_str"].detach())
    if dead_stats["any_valid_frac"] != 0.0 or dead_gs != 0.0:
        print(f"[v3-preflight] ⛔ FAIL: the regression control did NOT go dead "
              f"(valid_frac={dead_stats['any_valid_frac']}, "
              f"goal_str={dead_gs}) — this preflight cannot detect a dead LAN "
              f"label, so its PASS means nothing.")
        return 6
    print(f"[v3-preflight] lan arm OK: goal_str={gs:.4f} live / "
          f"{dead_gs:.4f} under the +inf-guard control "
          f"(control able to fail: True)")
    return 0


def preflight(args) -> int:
    print("[v3-preflight] building both arms + pinning the delta …")
    cfg_h = v3.refc_v3_sized_config(args.size, hier=True)
    cfg_f = v3.refc_v3_sized_config(args.size, hier=False)
    if args.smoke:
        cfg_h, cfg_f = (v3.refc_v3_smoke_config(True),
                        v3.refc_v3_smoke_config(False))
    cfg_h, cfg_f = _pin_trainer_cfg(cfg_h, args), _pin_trainer_cfg(cfg_f, args)
    delta = v3.config_delta(cfg_h, cfg_f)
    if set(delta) != REGISTERED_DELTA_KEYS:
        print(f"[v3-preflight] ⛔ FAIL: config delta {sorted(delta)} != "
              f"registered {sorted(REGISTERED_DELTA_KEYS)} — amend "
              f"PREREG_REFC_V3.md BEFORE launch.")
        return 2
    cfg = cfg_h if args.arm == "hier" else cfg_f
    model = v3.RefCV3Model(cfg)
    bd = v3.param_breakdown_v3(model)
    print(f"[v3-preflight] arm={args.arm} params={bd}")
    torch.manual_seed(0)
    h, wpx = cfg.core.encoder.image_hw()
    frames = torch.rand(2, cfg.core.window, cfg.core.encoder.in_channels,
                        h, wpx)
    if cfg.hier:
        rep = v3.freeze_history_report(model, frames,
                                       v0=torch.tensor([3.0, 7.0]))
        print(f"[v3-preflight] freeze-history gate: {rep}")
        if not rep["pass"]:
            print("[v3-preflight] ⛔ FAIL: the C115 gate — the H arm is "
                  "flat-in-disguise; the experiment would be VOID (OUTCOME V).")
            return 3
        model.eval()
        with torch.no_grad():
            a = model(frames, v0=torch.tensor([0.0, 0.0]))
            bq = model(frames, v0=torch.tensor([9.0, 4.0]))
            c = model(torch.rand_like(frames), v0=torch.tensor([0.0, 0.0]))
        for k in ("g_str", "g_tac", "z_tac"):
            if not torch.equal(a[k], bq[k]):
                print(f"[v3-preflight] ⛔ FAIL: v0 leaked into {k} (E11)")
                return 4
            if torch.equal(a[k], c[k]):
                print(f"[v3-preflight] ⛔ FAIL: frames do not move {k} — "
                      f"probe UNPOWERED, not clean (C109)")
                return 4
        model.train()
    # one synthetic end-to-end loss step (fail here, not on the pod). The lan
    # LABEL pathway is exercised on the train path, not here — the hier loss
    # skips loss_gstr when `lan` is absent BY DESIGN, and the preflight's job
    # is the shared+goal surfaces.
    eps = _synth_episodes(2, cfg.core, seed=0)
    ds = V3Dataset(eps, window=cfg.core.window, max_horizon=20,
                   channels=cfg.core.encoder.in_channels)
    batch = torch.utils.data.default_collate([ds[0], ds[1]])
    # ⛔ THE PREFLIGHT MUST EXERCISE THE LAUNCH CONFIG, NOT A NEIGHBOUR OF IT.
    # MEASURED 2026-09-02: with --v7-labels the model is built with 8-wide
    # z_tac heads, but the synthetic preflight corpus carries no v7 records, so
    # the loss fell back to kin3 and the vocabulary refusal fired — the gate
    # working correctly on a config nobody would launch. A preflight that can
    # only pass in a shape the real run never takes is not a preflight, so the
    # synthetic batch is given v7-SHAPED labels (one in-band row and one
    # ignored row, which also exercises the -100 path the real corpus produces
    # on ~76 % of windows).
    if getattr(args, "v7_labels", None):
        n_lat, n_lon = len(v7l.HEADS["tac_lat"]), len(v7l.HEADS["tac_lon"])
        batch["lat_v7"] = torch.tensor([0, v7l.IGNORE_ID], dtype=torch.long)
        batch["lon_v7"] = torch.tensor([n_lon - 1, v7l.IGNORE_ID],
                                       dtype=torch.long)
        assert n_lat >= 1 and n_lon >= 1
    losses = compute_losses_v3(model, batch, "cpu", mode="diffusion")
    bad = [k for k, t in losses.items()
           if torch.is_tensor(t) and not bool(t.isfinite().all())]
    if bad:
        print(f"[v3-preflight] ⛔ FAIL: non-finite losses {bad}")
        return 5
    print(f"[v3-preflight] loss step OK: "
          f"{ {k: round(float(t.detach()), 4) for k, t in losses.items() if torch.is_tensor(t) and t.ndim == 0} }")
    # ⛔ `route` above is the v2.1 NAV-derived CE, NOT the LAN route. It is
    # masked by nav_valid and is bit-identical with and without every LAN flag
    # (MEASURED, D-LAN-PF) — reading it as evidence about LAN is a category
    # error the LAN arm below exists to prevent.
    if (args.goal_str or args.graft_lan) and cfg.hier:
        rc = _lan_arm_preflight(cfg, args)
        if rc:
            return rc
    print("[v3-preflight] ✅ PASS")
    return 0


# ============================================================================
# Train loop
# ============================================================================

def train(args) -> dict:
    # MEASURED on the A40 pod 2026-09-02: the FIRST real launch died within
    # seconds -- `DataLoader worker killed by signal: Bus error ... out of
    # shared memory`. B1 payloads are ~34 MB/clip and torch's DEFAULT tensor
    # sharing passes them through /dev/shm, which is 24 GB in this container;
    # 4 workers x an LRU of decoded clips exhausts it almost immediately.
    # `file_system` sharing moves those handles off /dev/shm at no cost here
    # (the payloads are already files on disk) -- the same fix the throughput
    # sweep needed. MUST be set before any worker forks, hence the very top.
    if getattr(args, "workers", 0) > 0:
        import torch.multiprocessing as _tmp
        _tmp.set_sharing_strategy("file_system")
        print(f"[v3] tensor sharing = file_system (workers={args.workers}; "
              f"/dev/shm 24 GB vs ~34 MB/clip payloads)", flush=True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    torch.manual_seed(args.seed)
    cfg = _pin_trainer_cfg(
        v3.refc_v3_smoke_config(args.arm == "hier") if args.smoke else
        v3.refc_v3_sized_config(args.size, hier=args.arm == "hier"), args)
    # ⛔ The delta is derived AT THE SAME SIZE both arms run at. Deriving it at a
    # different rung would compare a config pair neither arm uses. The pair is
    # pinned through the same helper as the built arm, so the recorded delta is
    # the delta of the configs that actually train.
    delta = v3.config_delta(
        _pin_trainer_cfg(v3.refc_v3_sized_config(args.size, hier=True), args),
        _pin_trainer_cfg(v3.refc_v3_sized_config(args.size, hier=False), args))
    if set(delta) != REGISTERED_DELTA_KEYS:
        raise SystemExit(f"[v3] ⛔ config delta {sorted(delta)} != registered "
                         f"{sorted(REGISTERED_DELTA_KEYS)} — amend the prereg "
                         f"first (C122).")
    if args.graft_lan or args.goal_str:
        cfg.core.lan = refc.LanConfig(k=len(args.lan_arclengths))

    model = v3.RefCV3Model(cfg).to(device)
    if args.anchors:
        anc = torch.load(args.anchors, map_location=device,
                         weights_only=True)
        model.core.decoder.load_anchors(anc.to(device))

    # data — raw epcache, v2 compressed cache, or CI-synthetic; exactly one
    n_src = sum(1 for s in (args.data_root, args.v2_cache,
                            args.synth_episodes) if s)
    if n_src != 1:
        raise SystemExit("[v3] pass exactly one of --data-root / --v2-cache / "
                         "--synth-episodes (the synthetic corpus is CI-only "
                         "and must never masquerade as a training cache)")
    v2_parity = None
    if args.synth_episodes:
        eps = _synth_episodes(args.synth_episodes, cfg.core, seed=args.seed)
    elif args.v2_cache:
        # the flagship's --v2-cache recipe (train_flagship4b.py), verbatim in
        # structure: membership guard BEFORE any GPU work, lazy LRU-bounded
        # providers (contract-identical episode surface), and the independent
        # provider-vs-guard count cross-check. `require=False` keeps
        # deliberately-non-parity corpora usable behind ONE loud line;
        # --require-parity turns the same check into a refusal.
        from tanitad.data import parity
        from tanitad.data.v2_dataset import build_v2_providers
        v2_parity = parity.assert_v2_parity_cache(
            args.v2_cache, label="v3 v2-cache", require=args.require_parity)
        eps = build_v2_providers(args.v2_cache, lru_size=args.v2_lru)
        if v2_parity.get("parity") and len(eps) != v2_parity["episodes_loaded"]:
            raise parity.ParityViolation(
                f"PARITY VIOLATION [v3 v2-cache]: the guard verified "
                f"{v2_parity['episodes_loaded']} clip files but the loader "
                f"built {len(eps)} providers — the _v2manifest.pt sidecar "
                f"disagrees with the directory; rebuild it "
                f"(build_v2_providers(..., rebuild=True)) and re-run.")
        if args.episodes:
            eps = eps[:args.episodes]
        print(f"[v3] {len(eps)} lazy v2 providers from {args.v2_cache} "
              f"(lru {args.v2_lru})")
    else:
        eps, train_dir = load_cached_episodes(args.data_root, "*train*",
                                              args.episodes)
        print(f"[v3] {len(eps)} episodes from {train_dir}")
    # ⛔ geometry is asserted against the EPISODES, not against the flag: a
    # 256x640 corpus fed to a 256x256 build would run (conv is size-agnostic)
    # and silently train a model whose config lies about its input.
    eh, ew = cfg.core.encoder.image_hw()
    fh, fw = int(eps[0].frames.shape[-2]), int(eps[0].frames.shape[-1])
    if (fh, fw) != (eh, ew):
        raise SystemExit(
            f"[v3] ⛔ geometry mismatch: encoder built for {eh}x{ew} but the "
            f"corpus emits {fh}x{fw} — pass --image-hw {fh} {fw} (params are "
            f"unchanged; only compute scales), or point at a matching cache.")
    want_lan = bool(args.graft_lan or args.goal_str)
    dcls = lan_dataset_class(V3Dataset) if want_lan else V3Dataset
    kw = dict(window=cfg.core.window, max_horizon=20,   # ⛔ parity: NEVER 60
              channels=cfg.core.encoder.in_channels)
    if want_lan:
        kw["lan_cfg"] = DataLanConfig(
            arclengths_m=tuple(args.lan_arclengths),
            min_lead_m=args.lan_min_lead_m)
    ds = dcls(eps, **kw)
    # ---- v7.2 label join (PI 2026-09-02: MANDATORY for this launch) --------
    if args.v7_labels:
        from tanitad.data.v2_dataset import stable_episode_id
        labels, manifest = v7l.load_v7_labels(args.v7_labels,
                                              allow_oracle_nav=True)
        by_sid = {stable_episode_id(l.clip_id): l for l in labels}
        ds.v7_by_sid = by_sid
        ds.v7_dt = 0.1
        # ⛔ COVERAGE IS REPORTED, NOT ASSUMED. MEASURED 2026-09-02: the v7.2
        # train set joins 4,572/4,713 = 97.0 % of B1 but only 190/2,400 =
        # 7.9 % of the PARITY corpus — so the same flag on the wrong cache
        # silently supervises the tactical heads on ~8 % of clips. The launch
        # record must carry the number, and a low one must be loud.
        hit = sum(1 for e in eps if int(e.episode_id) in by_sid)
        frac = hit / max(len(eps), 1)
        print(f"[v3] v7.2 labels: {len(labels)} records, joined "
              f"{hit}/{len(eps)} episodes = {100 * frac:.1f} % "
              f"(release {manifest.release if hasattr(manifest, 'release') else '?'})")
        if frac < 0.5:
            raise SystemExit(
                f"[v3] ⛔ v7.2 label coverage {100 * frac:.1f} % — refusing to "
                f"launch. The PI made v7.2 supervision MANDATORY; below half "
                f"the corpus the tactical/strategic heads would train on a "
                f"minority of clips while the run LOOKED labelled. Check the "
                f"cache is B1 (97.0 %) and not the parity corpus (7.9 %).")
    # launch-line P4: the run PRINTS its episode/window counts at start — the
    # only way a parity claim about the enumeration is checkable from the log.
    print(f"[v3] {len(eps)} episodes -> {len(ds)} windows "
          f"(window {cfg.core.window}, max_horizon 20, "
          f"image_hw {cfg.core.encoder.image_hw()}, "
          f"tac_vocab {cfg.tac_vocab_version})")
    # ---- held-out eval split (PI 2026-09-02: "an eval step at 100 step") ---
    # ⛔ THE LEAK THIS CLOSES, MEASURED: the v7.2 release splits 4,572 train /
    # 147 eval with ZERO intersection, but the raw B1 corpus is 4,713 = 4,572 +
    # 141 of those eval clips. Training on "all of B1" therefore trains on 141
    # eval clips, and every eval number off them would be optimistic and
    # inadmissible. The launch trains the 4,572 and evaluates the held-out set.
    eval_dl = None
    if args.eval_cache and args.eval_every:
        # the SAME provider call the train side uses (refc_v3_train:680) —
        # a second construction path would be a second contract.
        from tanitad.data.v2_dataset import build_v2_providers
        e_eps = build_v2_providers([args.eval_cache], lru_size=args.v2_lru)
        overlap = ({int(e.episode_id) for e in eps}
                   & {int(e.episode_id) for e in e_eps})
        if overlap:
            raise SystemExit(
                f"[v3] ⛔ REFUSING: {len(overlap)} episodes appear in BOTH the "
                f"train cache and the eval cache. A held-out split that is not "
                f"held out measures memorisation.")
        e_ds = dcls(e_eps, **kw)
        if args.eval_labels:
            from tanitad.data.v2_dataset import stable_episode_id
            e_lab, _ = v7l.load_v7_labels(args.eval_labels,
                                          allow_oracle_nav=True)
            e_ds.v7_by_sid = {stable_episode_id(l.clip_id): l for l in e_lab}
            e_ds.v7_dt = 0.1
        # FIXED **and REPRESENTATIVE** windows.
        # ⛔ shuffle=False ALONE IS A TRAP, and it bit this eval on its first
        # run: taking the first N windows takes them from the START of the
        # first episodes (window NOW ~0.7-2 s), and every v7.2 record is
        # anchored at t0 = 8.0 s with a +-2 s band — so `eval_lat_tac` read
        # exactly 0.0000, an eval that silently never measured the tactical
        # heads while reporting a clean number. ⇒ take a DETERMINISTIC RANDOM
        # subset (seeded, computed once) and iterate it in a fixed order: still
        # the same windows at every step, but drawn from the whole corpus.
        g_ev = torch.Generator().manual_seed(12345)
        n_need = args.eval_batches * args.batch
        perm = torch.randperm(len(e_ds), generator=g_ev)[:n_need].tolist()
        eval_dl = torch.utils.data.DataLoader(
            torch.utils.data.Subset(e_ds, perm),
            batch_size=args.batch, shuffle=False,
            num_workers=0, drop_last=True)
        print(f"[v3] held-out eval: {len(e_eps)} episodes -> {len(e_ds)} "
              f"windows, {args.eval_batches} fixed batches every "
              f"{args.eval_every} steps", flush=True)

    # ⛔ IN-FLIGHT BATCHES ARE THE SHM COST, NOT THE CLIP CACHE. MEASURED
    # 2026-09-02: two launches died on `Bus error ... out of shared memory`
    # even with file_system sharing. One collated batch is
    #   20 x window 8 x 9ch x 256 x 640 x 4B ~= 940 MB
    # so workers x prefetch_factor batches are in flight at once: 4 x 2 = 8
    # ~= 8 GB against a 24 GB /dev/shm, plus the eval loader's own. The knob
    # that actually bounds this is prefetch_factor, and it had been left at
    # torch's default of 2.
    pf = {"prefetch_factor": args.prefetch_factor} if args.workers > 0 else {}
    if args.workers > 0:
        print(f"[v3] loader: workers={args.workers} prefetch={args.prefetch_factor}"
              f" -> <={args.workers * args.prefetch_factor} batches in flight",
              flush=True)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        **pf,
        drop_last=True, persistent_workers=args.workers > 0)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = lambda s: (s + 1) / max(1, args.warmup) if s < args.warmup else \
        0.5 * (1.0 + math.cos(math.pi * (s - args.warmup)
                              / max(1, args.steps - args.warmup)))   # noqa: E731

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    ck = out_dir / "ckpt.pt"
    if ck.exists():
        state = torch.load(ck, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])          # strict — resume exact
        opt.load_state_dict(state["opt"])
        step = int(state["step"])
        print(f"[v3] resumed {args.arm} at step {step}")
    (out_dir / "config.json").write_text(json.dumps({
        "arm": args.arm, "seed": args.seed, "argv": sys.argv[1:],
        "registered_delta": {k: [repr(a), repr(b)]
                             for k, (a, b) in delta.items()},
        "param_breakdown": v3.param_breakdown_v3(model),
        "goal_provenance": refc.RefCModel.goal_provenance(),
        "provenance_roles": v3.RefCV3Model.provenance_roles(),
        "horizons": list(cfg.core.trajectory.horizons),
        "goal_tau_steps": list(cfg.goal_tau_steps),
        "admission_sigma_m": cfg.admission_sigma_m,
        # the 2026-09-01 B1-readiness fields — the config.json is the ONLY
        # durable record of what corpus/geometry/vocab a run actually used.
        "image_hw": list(cfg.core.encoder.image_hw()),
        "tac_vocab_version": cfg.tac_vocab_version,
        "v2_cache": args.v2_cache, "require_parity": bool(args.require_parity),
        "v2_parity": v2_parity,
    }, indent=1), encoding="utf-8")

    log = (out_dir / "metrics.jsonl").open("a", encoding="utf-8")
    t0, model = time.time(), model.train()
    it = iter(dl)
    while step < args.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl)
            batch = next(it)
        for g in opt.param_groups:
            g["lr"] = args.lr * sched(step)
        losses = compute_losses_v3(model, batch, device, mode=args.mode)
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        step += 1
        if step % args.log_every == 0 or step == args.steps:
            # ⚠️ The tensor filter silently DROPPED every plain-float
            # diagnostic — `tac_label_v7`, `tac_label_rows`, `nav_injected`
            # were emitted by the loss and never reached the log, so "did the
            # v7.2 labels actually train?" was answerable only by INFERRING it
            # from the CE magnitude (lat 1.22 ~ ln 3 vs lat_tac 2.13 ~ ln 8).
            # A diagnostic that does not survive to the log is not a
            # diagnostic. Scalars now pass through as themselves.
            row = {k: (round(float(v.detach()), 5) if torch.is_tensor(v)
                       else round(float(v), 5))
                   for k, v in losses.items()
                   if (torch.is_tensor(v) and v.ndim == 0)
                   or isinstance(v, (int, float, bool))}
            row.update(step=step, elapsed_s=round(time.time() - t0, 1),
                       lr=opt.param_groups[0]["lr"])
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[v3:{args.arm}] step {step} "
                  f"loss {row['loss']:.4f} traj {row['traj']:.4f}")
        # ---- held-out eval (PI 2026-09-02) --------------------------------
        # ⚠️ WHAT THIS IS AND IS NOT: an in-training MONITOR at T0 on the same
        # loss surface, over a FIXED set of held-out windows. It is NOT the
        # four-metric-family result and must never be quoted as one — the
        # binding families (longitudinal / lateral / tactical / strategic with
        # paired episode-cluster CIs) are a separate T1 job. What it buys is
        # the ability to see generalisation move DURING a 25 h run instead of
        # after it, and to catch train-only improvement early.
        if eval_dl is not None and (step % args.eval_every == 0
                                    or step == args.steps):
            model.eval()
            acc, nb_e = {}, 0
            with torch.no_grad():
                for eb in eval_dl:
                    if nb_e >= args.eval_batches:
                        break
                    el = compute_losses_v3(model, eb, device, mode=args.mode)
                    for k, v in el.items():
                        if torch.is_tensor(v) and v.ndim == 0:
                            acc[k] = acc.get(k, 0.0) + float(v.detach())
                        elif isinstance(v, (int, float, bool)):
                            acc[k] = acc.get(k, 0.0) + float(v)
                    nb_e += 1
            model.train()
            if nb_e:
                erow = {f"eval_{k}": round(v / nb_e, 5) for k, v in acc.items()}
                erow.update(step=step, eval_batches=nb_e,
                            eval_windows=nb_e * args.batch)
                log.write(json.dumps(erow) + chr(10))
                log.flush()
                print(f"[v3:eval] step {step} "
                      f"loss {erow.get('eval_loss', float('nan')):.4f} "
                      f"traj {erow.get('eval_traj', float('nan')):.4f} "
                      f"lat_tac {erow.get('eval_lat_tac', float('nan')):.4f}",
                      flush=True)

        if step % args.save_every == 0 or step == args.steps:
            torch.save({"model": model.state_dict(),
                        "opt": opt.state_dict(), "step": step}, ck)
        if step in MILESTONES:
            torch.save({"model": model.state_dict(), "step": step},
                       out_dir / f"ckpt_{step}.pt")
    # ⛔ the done-marker, SAME turn as completion (the v5f supervisor lesson).
    (out_dir / "summary.json").write_text(
        json.dumps({"done": True, "step": step, "arm": args.arm,
                    "seed": args.seed,
                    "wallclock_s": round(time.time() - t0, 1)}),
        encoding="utf-8")
    log.close()
    print(f"[v3:{args.arm}] DONE at {step} — summary.json written")
    return {"step": step}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--size", choices=tuple(v3.V3_SIZES), default="small",
                    help="encoder rung. 'small' is the AS-REGISTERED arm "
                         "(62,930,419); 'xl' is the D-008 >=250M rung "
                         "(217,760,775). ⛔ The size axis moves the ENCODER "
                         "ONLY — the H-vs-F delta is identical at every rung "
                         "(pinned by tests). ⚠️ Anything but 'small' VOIDS the "
                         "registered cost line in PREREG_REFC_V3.md; amend "
                         "BEFORE any read, never after.")
    ap.add_argument("--arm", choices=("hier", "flat"), required=True,
                    help="v3-H (goal cascade) or v3-F (flat) — the dominance "
                         "pair; delta pinned to the registered lever set")
    ap.add_argument("--data-root", default=None,
                    help="raw epcache root of ep_*.pt (parity-guarded; the "
                         "physicalai-train-e438721ae894 convention)")
    ap.add_argument("--v2-cache", nargs="+", default=None,
                    help="v2 compressed cache dir(s) of *.v2ep.pt (e.g. the B1 "
                         "corpus physicalai-b1-w120-256x640cyl). Swaps the raw "
                         "loader for the lazy LRU-bounded v2 providers; the "
                         "window contract is identical. Pair with "
                         "--image-hw 256 640 for the B1 geometry — the run "
                         "REFUSES a geometry mismatch either way.")
    ap.add_argument("--prefetch-factor", type=int, default=1,
                    help="batches prefetched PER WORKER. One collated batch is "
                         "~940 MB at batch 20, so workers x prefetch bounds "
                         "shared-memory use; torch's default of 2 put 8 "
                         "batches in flight and exhausted a 24 GB /dev/shm.")
    ap.add_argument("--eval-cache", default=None,
                    help="HELD-OUT v2 cache (the v7.2 eval split). ⛔ MUST be "
                         "disjoint from --v2-cache: 141 of the 147 v7.2 eval "
                         "clips sit inside the raw B1 corpus, so training on "
                         "'all of B1' trains on the eval set (MEASURED "
                         "2026-09-02).")
    ap.add_argument("--eval-labels", default=None,
                    help="s2_labels_v7.2_eval.jsonl.gz for the eval split")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="run the held-out eval every N steps (0 = off)")
    ap.add_argument("--eval-batches", type=int, default=8,
                    help="FIXED number of eval batches — the same windows "
                         "every time, so step-to-step deltas are the model "
                         "moving and not the sample moving")
    ap.add_argument("--v7-labels", default=None,
                    help="s2_labels_v7.2_*.jsonl.gz — the RELEASED tactical "
                         "vocabulary (8x8). Joined per window on "
                         "stable_episode_id(clip_id), the v7.2 index's own "
                         "'ONLY admissible join key'. Sets tac_vocab_version "
                         "to v7.0; without it the trainer derives kin3 (3x3) "
                         "from poses. Coverage is PRINTED and a run below "
                         "50 %% is REFUSED.")
    ap.add_argument("--v2-lru", type=int, default=6,
                    help="per-process LRU of decoded-payload clips for "
                         "--v2-cache. ⚠️ B1 payloads are ~34 MB/clip "
                         "(256x640 PNG) — the old '2-4 MB' sizing is 8-17x "
                         "low (MEASURED, V5F_SIGKILL.md), and under "
                         "shuffle the hit rate is ~lru/n_clips, so big "
                         "values buy RAM pressure, not throughput. 6 "
                         "matches the v6 chain's setting.")
    ap.add_argument("--require-parity", action="store_true",
                    help="REFUSE unless --v2-cache references a REGISTERED "
                         "corpus key (B1 is unregistered as of 2026-09-01: "
                         "expect the loud NON-PARITY line without this flag; "
                         "registering B1 is the cross-arm-comparability "
                         "instrument)")
    ap.add_argument("--image-hw", type=int, nargs=2, default=None,
                    metavar=("H", "W"),
                    help="build the encoder at this input geometry (B1: "
                         "256 640). Params unchanged (fully-conv trunk); "
                         "compute scales with area. Applies to --preflight "
                         "too, so the gate runs at the launch geometry.")
    ap.add_argument("--synth-episodes", type=int, default=0,
                    help="CI-ONLY synthetic corpus (mutually exclusive with "
                         "--data-root / --v2-cache)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--mode", choices=("classifier", "diffusion"),
                    default="diffusion")
    ap.add_argument("--anchors", default=None,
                    help="6 s anchor vocabulary (build_refc_anchors.py over "
                         "V3_HORIZONS; the model's synthetic default otherwise)")
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--episodes", type=int, default=0, help="0 = all")
    ap.add_argument("--goal-str", action="store_true",
                    help="train the strategic goal head (needs the lan LABEL "
                         "field — minted WITHOUT building the input pathway, "
                         "the refc_goal_config discipline)")
    ap.add_argument("--graft-lan", action="store_true",
                    help="supplied-corridor MODEL INPUT — ⛔ NOT part of any "
                         "registered v3 arm (E12); exists for diagnostics only")
    ap.add_argument("--lan-arclengths", type=float, nargs="+",
                    default=[10.0, 20.0, 40.0, 80.0])
    ap.add_argument("--lan-min-lead-m", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU config (CI)")
    ap.add_argument("--preflight", action="store_true",
                    help="build + pin delta + C115 gate + E11 audit + one "
                         "synthetic loss step, then exit")
    args = ap.parse_args(argv)
    if args.preflight:
        raise SystemExit(preflight(args))
    train(args)


if __name__ == "__main__":
    main()
