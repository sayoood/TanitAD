"""Phase-0 THREE-ARM gate comparison — flagship vs REF-A vs REF-B on ONE val set.

WHAT THIS ANSWERS (Phase 0 Plan §4 + DRIVING_DIAGNOSTIC_FRAMEWORK §3)
---------------------------------------------------------------------
Three arms train on the BYTE-IDENTICAL PhysicalAI set; only the architecture
differs (the controlled comparison):

  flagship  261M 4-brain: from-scratch ViT encoder + operative/tactical/strategic
            predictors + H15 imagination + metric-dynamics grounding + SIGReg.
  REF-A     frozen-DINO features -> trainable adapter -> shared predictor
            (`--adapter grid`); trains on DINO features, not raw frames.
  REF-B     from-scratch ViT, behaviour-cloning; direct tactical waypoint heads,
            NO world model.

THE Phase-0 question this harness answers off-pod: **does the 4-brain flagship
beat REF-A, REF-B, and the trivial baselines on the decode + open-loop metrics
(the necessary conditions for the hierarchy edge)?** The closed-loop gates
D4-D6 remain the arbiters (arXiv 2512.24497; gates.py doctrine) and are computed
elsewhere — this harness NEVER reads a passing decode number as a driving claim.

RIGOROUS METRIC IDENTITY (the whole point of a parity comparison)
-----------------------------------------------------------------
Everything that can be identical IS identical, structurally in code:

  1. ONE evaluation grid. The GT ego-waypoints, the trivial baselines, the
     episode ids, the curvature/speed strata are computed ONCE from the val
     poses (:func:`build_reference_grid`) and shared by every arm. No arm sees
     a different target or a different window.
  2. ONE decode function. Every arm's compact state (`encode_window(...)[:,-1]`)
     goes through the SAME :func:`decode_parity` — the same frozen ``RidgeProbe``
     ladder, the same ``gates.run_d1`` instrument-doctrine gate, the same
     route-resampled episode splits. The ONLY thing that differs per arm is the
     state tensor (that is the architecture axis under test).
  3. ONE metric definition. ``ade_0_2s`` is the 4-waypoint mean (0.5/1/1.5/2 s)
     from ``driving_diagnostic.scalar_metrics`` — reused verbatim, so the D1
     number, the grounded-rollout number and the baselines all mean the same
     thing.

The DECODE PATH of the trajectory metric is necessarily per-architecture (that
is what we are comparing): flagship & REF-A roll their grounded operative
predictor under true actions (``metric_dynamics.rollout_decode``); REF-B has no
world model, so its native trajectory is its direct tactical waypoint head. The
metric and the episodes are identical; only the mechanism differs — labelled
honestly in the table.

SAME-EPISODE GUARANTEE (fairness)
---------------------------------
Flagship & REF-B read raw frame val episodes; REF-A reads DINO-feature val
episodes. :func:`load_common_val` intersects the two by ``episode_id`` and keeps
only episodes present in BOTH, then asserts their ``poses`` match to
``--pose-tol`` (the same clip in two representations must have the same
odometry). The reference grid is built from those shared poses, so every arm is
scored on the SAME windows of the SAME clips. The exact episode-id list used is
emitted in the report.

VAL ARTIFACTS TO PROVISION (I do not build these — I consume them)
------------------------------------------------------------------
  frame val cache  (flagship + REF-B):  <root>/*val*/ep_*.pt  (mixing.save_episode
        contract: frames_u8, actions, poses, episode_id)  — the deterministic
        val_frac=0.2 split of the 2376-set (see phase0_go_criteria.md).
  DINO feature val (REF-A):  <feat-dir>/ep_*.pt  (dino_precompute contract:
        feats_fp16 [T,256,768], actions, poses, episode_id) for the SAME val
        episode ids.  REF-A/REF-B builds currently SKIP val to save disk — this
        harness states exactly what to provision; provisioning is handled
        upstream.

Usage (dev-box 4060 or pod), 3-arm:
  python scripts/compare_arms.py \
      --flagship-ckpt  <dir>/ckpt.pt  --flagship-config flagship4b \
      --refa-ckpt      <dir>/ckpt.pt  --refa-adapter grid \
      --refb-ckpt      <dir>/ckpt.pt \
      --frame-cache-dirs /workspace/data/physicalai/_epcache \
      --refa-feat-dir    /opt/dino_feats/physicalai-val-dinov2-b14 \
      --episodes 150 --out <dir>/arm_compare

Any arm may be omitted (evaluate whichever checkpoints have landed).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reused VERBATIM (no reinvention) — the proven diagnostic + gate primitives.
import driving_diagnostic as dd  # noqa: E402
from tanitad.eval.gates import (I2Input, run_d1, run_d2,  # noqa: E402
                                run_d3, split_by_episode)
from tanitad.instruments.numerics import strict_numerics  # noqa: E402
from tanitad.models.metric_dynamics import rollout_decode  # noqa: E402

WP_STEPS = dd.WP_STEPS                 # (5,10,15,20) = 0.5/1/1.5/2 s @10Hz
K_MAX = dd.K_MAX                       # 20
CAMERA_ADE_MAX = 1.0                   # Plan §4 D1 camera threshold (metres, ade@1s)
WP_IDX = torch.tensor([k - 1 for k in WP_STEPS])


# --------------------------------------------------------------------------- #
# Arm abstraction — the ONLY per-architecture surface                          #
# --------------------------------------------------------------------------- #
@dataclass
class ArmSpec:
    name: str                                  # "flagship" | "refa" | "refb"
    kind: str                                  # "frame" | "feature"
    model: torch.nn.Module
    window: int
    encode_window: Callable[[torch.Tensor], torch.Tensor]  # win -> [B,W,S]
    encode_one: Callable[[torch.Tensor], torch.Tensor]     # [B,*] -> [B,S] (I2)
    state_dim: int
    step: int
    # Imagination gates (D2/D3) — only arms with an action-conditioned predictor.
    predictor: Optional[torch.nn.Module] = None
    imagine1: Optional[Callable] = None        # (states,actions) -> z_{t+1}
    # Grounded/native trajectory decode.
    grounded_step_readout: Optional[torch.nn.Module] = None  # flagship/refa
    native_waypoints: Optional[Callable] = None              # refb direct head
    native_label: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def has_imagination(self) -> bool:
        return self.imagine1 is not None


# --------------------------------------------------------------------------- #
# Val loading + same-episode guarantee                                         #
# --------------------------------------------------------------------------- #
def _corpus_of(cache_dir: str) -> str:
    low = str(cache_dir).lower()
    if "comma" in low:
        return "comma2k19"
    if "physical" in low:
        return "physicalai"
    return Path(cache_dir).name


def load_frame_val(cache_dirs, episodes):
    """Load val frame episodes from *val* dirs. Returns [(ToyEpisode, corpus)]."""
    from tanitad.data.mixing import load_episode
    out = []
    for cd in cache_dirs:
        val_dirs = sorted(Path(cd).glob("*val*"))
        if not val_dirs:
            print(f"[compare] WARNING no *val* dir under {cd}", flush=True)
            continue
        for p in sorted(val_dirs[-1].glob("ep_*.pt"))[:episodes]:
            out.append((load_episode(str(p), mmap=True), _corpus_of(cd)))
    return out


def load_feature_val(feat_dir, episodes):
    """Load val DINO-feature episodes. Returns [dict(feats_fp16,actions,poses,id)]."""
    if feat_dir is None:
        return []
    fd = Path(feat_dir)
    # Accept either a direct dir of ep_*.pt or a parent holding *val* subdirs.
    files = sorted(fd.glob("ep_*.pt"))
    if not files:
        val_dirs = sorted(fd.glob("*val*"))
        if val_dirs:
            files = sorted(val_dirs[-1].glob("ep_*.pt"))
    out = []
    for p in files[:episodes]:
        d = torch.load(str(p), map_location="cpu", weights_only=True)
        out.append(d)
    return out


def _poses4(poses: torch.Tensor) -> torch.Tensor:
    """Normalise a poses tensor to [T,4] (x,y,yaw,v). Feature caches may store
    [T,3] (x,y,yaw) — pad v=0 so the shared geometry helpers accept them."""
    if poses.shape[-1] >= 4:
        return poses[:, :4].float()
    pad = torch.zeros(poses.shape[0], 4 - poses.shape[-1])
    return torch.cat([poses.float(), pad], dim=-1)


def load_common_val(frame_val, feat_eps, need_feature: bool, pose_tol: float):
    """Intersect frame + feature val by episode_id; assert poses match.

    Returns (common_frame_val, feat_by_id, common_ids). When ``need_feature`` is
    False (no REF-A arm) the feature set is ignored and every frame episode is
    kept. When it is True, only episodes present in BOTH with matching poses
    survive — the mechanical fairness guarantee.
    """
    if not need_feature:
        ids = [int(ep.episode_id) for ep, _ in frame_val]
        return frame_val, {}, ids
    feat_by_id = {int(d["episode_id"]): d for d in feat_eps}
    common, kept_ids = [], []
    for ep, corp in frame_val:
        eid = int(ep.episode_id)
        if eid not in feat_by_id:
            continue
        fp = _poses4(feat_by_id[eid]["poses"])
        gp = _poses4(ep.poses)
        n = min(fp.shape[0], gp.shape[0])
        dev = float((fp[:n, :3] - gp[:n, :3]).abs().max()) if n else float("inf")
        if dev > pose_tol:
            print(f"[compare] episode {eid}: frame/feature pose mismatch "
                  f"{dev:.4g} > {pose_tol} — dropped (not the same clip)",
                  flush=True)
            continue
        common.append((ep, corp))
        kept_ids.append(eid)
    return common, feat_by_id, kept_ids


# --------------------------------------------------------------------------- #
# ONE evaluation grid — GT + baselines + strata, shared by every arm           #
# --------------------------------------------------------------------------- #
@dataclass
class RefGrid:
    windows: list          # [(ep_index, t_start, last)]
    gt: torch.Tensor       # [N,4,2]  shared GT ego waypoints
    base: dict             # {name: [N,4,2]} trivial baselines
    eid: list              # [N] episode ids
    corpus: list           # [N] corpus tags
    speed: torch.Tensor    # [N]
    head_deg: torch.Tensor # [N]
    episodes: list         # the frame episodes (for state collection + I2)
    window: int


def build_reference_grid(frame_val, window: int, stride: int) -> RefGrid:
    """Compute the shared evaluation grid ONCE from the val poses."""
    windows, EID, COR = [], [], []
    GT, SPD, HDG = [], [], []
    BP = {n: [] for n in dd.BASELINES}
    episodes = [ep for ep, _ in frame_val]
    for ei, (ep, corp) in enumerate(frame_val):
        T = ep.frames.shape[0]
        poses = _poses4(ep.poses)
        for t in range(0, T - window - K_MAX, stride):
            last = t + window - 1
            windows.append((ei, t, last))
            lt = torch.tensor([last])
            GT.append(dd.gt_ego_waypoints(poses, lt))
            bp = dd.baseline_waypoints(poses, lt)
            for n in dd.BASELINES:
                BP[n].append(bp[n])
            SPD.append(poses[last, 3:4])
            HDG.append(dd.net_heading_change_deg(poses, lt))
            EID.append(int(ep.episode_id))
            COR.append(corp)
    assert windows, "no eval windows — val episodes too short for window+K_MAX"
    return RefGrid(
        windows=windows,
        gt=torch.cat(GT).float(),
        base={n: torch.cat(BP[n]).float() for n in dd.BASELINES},
        eid=EID, corpus=COR,
        speed=torch.cat(SPD).float().reshape(-1),
        head_deg=torch.cat(HDG).float().reshape(-1),
        episodes=episodes, window=window)


# --------------------------------------------------------------------------- #
# Per-arm STATE collection — indexed to the SAME grid                          #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_states_frame(arm: ArmSpec, grid: RefGrid, device, batch: int):
    """Encode the last-frame compact state for every grid window (frame arm)."""
    S = []
    W = grid.window
    for i in range(0, len(grid.windows), batch):
        chunk = grid.windows[i:i + batch]
        fw = []
        for ei, t, _ in chunk:
            fr = grid.episodes[ei].frames
            fr = fr.float().div(255.0) if fr.dtype == torch.uint8 else fr.float()
            fw.append(fr[t:t + W])
        st = arm.encode_window(torch.stack(fw).to(device))[:, -1].cpu()
        S.append(st.float())
    return torch.cat(S)


@torch.no_grad()
def collect_states_feature(arm: ArmSpec, grid: RefGrid, feat_by_id, device,
                           batch: int):
    """Encode the last-frame adapter state for every grid window (feature arm),
    indexed by episode_id -> the SAME windows the frame arms use."""
    S = []
    W = grid.window
    feats_cache = {}
    for i in range(0, len(grid.windows), batch):
        chunk = grid.windows[i:i + batch]
        fw = []
        for ei, t, _ in chunk:
            eid = int(grid.episodes[ei].episode_id)
            if eid not in feats_cache:
                feats_cache[eid] = feat_by_id[eid]["feats_fp16"]
            fw.append(feats_cache[eid][t:t + W].float())
        st = arm.encode_window(torch.stack(fw).to(device))[:, -1].cpu()
        S.append(st.float())
    return torch.cat(S)


# --------------------------------------------------------------------------- #
# THE identical decode — every arm goes through this, unchanged                #
# --------------------------------------------------------------------------- #
def decode_parity(states: torch.Tensor, grid: RefGrid, arm: ArmSpec, device,
                  n_splits: int, val_frac: float, seed: int,
                  mlp_epochs: int, i2: Optional[I2Input]) -> dict:
    """Frozen-probe decodability — the rigorous parity metric (identical code).

    Returns the instrument-doctrine D1 gate (gates.run_d1, camera unit) PLUS the
    ridge/MLP held-out-vs-oracle ladder (driving_diagnostic), so the table has
    both the gated D1 number and the representation ceiling. ``i2`` is the arm's
    batch-consistency instrument input (built by the caller from an encoder
    sample) — required for the D1 gate to be admissible (D-004).
    """
    # --- D1 (instrument-doctrine gate, camera <1.0m) — the headline parity row.
    d1 = run_d1(states, grid.gt, grid.eid, unit="camera", i2=i2,
                pooled_states=None, n_splits=n_splits, val_frac=val_frac,
                seed=seed)

    # --- Ridge/MLP ladder held-out vs oracle in-distribution ceiling (dd §2).
    splits = [split_by_episode(grid.eid, val_frac, s)
              for s in range(seed, seed + n_splits)]
    flat = lambda idx: grid.gt[idx].reshape(len(idx), 2 * len(WP_STEPS))
    probes = [("ridge", 1.0, "ridge_a1"), ("ridge", 10.0, "ridge_a10"),
              ("ridge", 100.0, "ridge_a100"), ("mlp", 1.0, "mlp")]
    held, oracle = {}, {}
    for kind, alpha, key in probes:
        ho, orc = [], []
        for tr, va in splits:
            pred, _ = dd.fit_predict(kind, alpha, states[tr], flat(tr),
                                     states[va], mlp_epochs)
            ho.append(dd.scalar_metrics(dd.de_of(pred, grid.gt[va])))
            po, _ = dd.fit_predict(kind, alpha, states[va], flat(va),
                                   states[va], mlp_epochs)
            orc.append(dd.scalar_metrics(dd.de_of(po, grid.gt[va])))
        held[key] = dd.agg_metric_dicts(ho)
        oracle[key] = dd.agg_metric_dicts(orc)
    best_key = min(held, key=lambda k: held[k]["ade_0_2s"]["mean"])
    best_ho = held[best_key]["ade_0_2s"]["mean"]
    best_orc = oracle[best_key]["ade_0_2s"]["mean"]
    return {
        "d1": d1.to_dict(),
        "d1_ade_0_2s": d1.metrics["ade@1s"],       # run_d1 "ade@1s" == ade_0_2s
        "d1_status": d1.status,
        "decode_ladder": {"held_out": held, "oracle_ceiling": oracle,
                          "best_probe": best_key},
        "best_heldout_ade_0_2s": best_ho,
        "oracle_ceiling_ade_0_2s": best_orc,
        "heldout_over_oracle": (best_ho / best_orc) if best_orc else None,
    }


# --------------------------------------------------------------------------- #
# Grounded / native trajectory ADE — per-arch mechanism, identical metric      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def grounded_rollout_ade(arm: ArmSpec, grid: RefGrid, feat_by_id, device,
                         batch: int) -> Optional[dict]:
    """flagship/REF-A: roll the operative predictor under TRUE actions, decode
    each transition with the grounded step-readout, SE(2)-accumulate to ego
    waypoints. REF-B: its direct tactical waypoint head. Returns None if the arm
    exposes no native trajectory decode. Metric + episodes are grid-identical."""
    W = grid.window
    if arm.native_waypoints is not None:              # REF-B direct head
        PRED = []
        for i in range(0, len(grid.windows), batch):
            chunk = grid.windows[i:i + batch]
            fw = []
            for ei, t, _ in chunk:
                fr = grid.episodes[ei].frames
                fr = fr.float().div(255.0) if fr.dtype == torch.uint8 else fr.float()
                fw.append(fr[t:t + W])
            PRED.append(arm.native_waypoints(torch.stack(fw).to(device)).cpu())
        pred = torch.cat(PRED).float()
    elif arm.grounded_step_readout is not None and arm.predictor is not None:
        PRED = []
        for i in range(0, len(grid.windows), batch):
            chunk = grid.windows[i:i + batch]
            sw, aw, fa = [], [], []
            for ei, t, _ in chunk:
                ep = grid.episodes[ei]
                if arm.kind == "feature":
                    eid = int(ep.episode_id)
                    src = feat_by_id[eid]
                    feats = src["feats_fp16"][t:t + W].float()
                    acts = src["actions"]
                else:
                    fr = ep.frames
                    fr = fr.float().div(255.0) if fr.dtype == torch.uint8 else fr.float()
                    feats = fr[t:t + W]
                    acts = ep.actions
                sw.append(feats)
                aw.append(acts[t:t + W].float())
                fa.append(acts[t + W:t + W + K_MAX].float())
            win = torch.stack(sw).to(device)
            states = arm.encode_window(win)                       # [b,W,S]
            actions = torch.stack(aw).to(device)
            fut_a = torch.stack(fa).to(device)
            wp_full, _ = rollout_decode(arm.predictor, states, actions, fut_a,
                                        arm.grounded_step_readout, K_MAX)
            PRED.append(wp_full.index_select(1, WP_IDX.to(device)).cpu())
        pred = torch.cat(PRED).float()
    else:
        return None

    de = dd.de_of(pred, grid.gt)
    cv_de = dd.de_of(grid.base["constant_velocity"], grid.gt)
    full = dd.scalar_metrics(de)
    cv = dd.scalar_metrics(cv_de)
    # straight stratum (the DRIVING_DIAGNOSTIC §3 curve-vs-straight bar).
    curv = [dd.curvature_bucket(float(h)) for h in grid.head_deg]
    strat = dd._strat(curv, de, cv_de)
    straight = strat.get("straight")
    return {
        "mechanism": arm.native_label,
        "ade_0_2s": full["ade_0_2s"],
        "de@1s": full["de@1s"], "de@2s": full["de@2s"],
        "cv_ade_0_2s": cv["ade_0_2s"], "cv_de@1s": cv["de@1s"],
        "beats_cv_overall": bool(full["ade_0_2s"] < cv["ade_0_2s"]),
        "straight": (None if straight is None else {
            "model_ade@1s": straight["model_ade@1s"],
            "cv_ade@1s": straight["cv_ade@1s"],
            "beats_cv": bool(straight["model_ade@1s"] < straight["cv_ade@1s"]),
            "n": straight["n"]}),
        "by_curvature": strat,
    }


# --------------------------------------------------------------------------- #
# D2 / D3 imagination gates — arms with an action-conditioned predictor only    #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def imagination_gates(arm: ArmSpec, grid: RefGrid, feat_by_id, device, batch,
                      val_frac, seed, i2: Optional[I2Input]) -> Optional[dict]:
    """D2 (imagination usable for selection) + D3 (imagined vs oracle decode) on
    arms that own an action-conditioned predictor (flagship, REF-A). REF-B has
    no world model -> None (the pre-registered structural gap)."""
    if not arm.has_imagination:
        return None
    W = grid.window
    k_max = max(arm.predictor.cfg.horizons)
    zc, z1, zi1, zik, ztk, d1, dk, act, prev = ([] for _ in range(9))
    for i in range(0, len(grid.windows), batch):
        chunk = grid.windows[i:i + batch]
        sw, aw = [], []
        nxt1, nxtk, disp1, dispk, acts_last, prevst = [], [], [], [], [], []
        for ei, t, last in chunk:
            ep = grid.episodes[ei]
            poses = _poses4(ep.poses)
            if arm.kind == "feature":
                src = feat_by_id[int(ep.episode_id)]
                feats = src["feats_fp16"].float()
                acts = src["actions"].float()
            else:
                fr = ep.frames
                fr = fr.float().div(255.0) if fr.dtype == torch.uint8 else fr.float()
                feats = fr
                acts = ep.actions.float()
            sw.append(feats[t:t + W])
            aw.append(acts[t:t + W])
            nxt1.append(feats[t + W])
            nxtk.append(feats[t + W + k_max - 1])
            p0, yaw0 = poses[last, :2], poses[last, 2]
            disp1.append(dd._ego(poses[last + 1, :2] - p0, yaw0))
            dispk.append(dd._ego(poses[last + k_max, :2] - p0, yaw0))
            acts_last.append(acts[last])
            prevst.append(torch.stack([poses[last, 3],
                                       dd._wrap(poses[last, 2] - poses[last - 1, 2])]))
        win = torch.stack(sw).to(device)
        states = arm.encode_window(win)
        awt = torch.stack(aw).to(device)
        preds = arm.predictor(states, awt)
        z_next1 = arm.encode_window(torch.stack(nxt1).unsqueeze(1).to(device))[:, 0] \
            if arm.kind == "feature" else \
            arm.encode_one(torch.stack(nxt1).to(device))
        z_nextk = arm.encode_window(torch.stack(nxtk).unsqueeze(1).to(device))[:, 0] \
            if arm.kind == "feature" else \
            arm.encode_one(torch.stack(nxtk).to(device))
        zc.append(states[:, -1].cpu()); z1.append(z_next1.cpu())
        zi1.append(preds[1].cpu()); zik.append(preds[k_max].cpu())
        ztk.append(z_nextk.cpu())
        d1.append(torch.stack(disp1)); dk.append(torch.stack(dispk))
        act.append(torch.stack(acts_last)); prev.append(torch.stack(prevst))
    zc, z1, zi1 = torch.cat(zc), torch.cat(z1), torch.cat(zi1)
    zik, ztk = torch.cat(zik), torch.cat(ztk)
    d1, dk = torch.cat(d1).float(), torch.cat(dk).float()
    act, prev = torch.cat(act).float(), torch.cat(prev).float()
    d2 = run_d2(zc, z1, zi1, d1, grid.eid, i2=i2, actions=act, prev_state=prev,
                val_frac=val_frac, seed=seed)
    d3 = run_d3(zc, ztk, zik, dk, grid.eid, i2=i2, val_frac=val_frac, seed=seed)
    return {"d2": d2.to_dict(), "d2_status": d2.status,
            "d2_dir_acc": d2.metrics["direction_acc"],
            "d2_p4_dir_acc": d2.metrics.get("p4_forward_dynamics_dir_acc"),
            "d3": d3.to_dict(), "d3_status": d3.status,
            "d3_ratio": d3.metrics["ratio"],
            "d3_horizon_s": k_max / 10.0}


# --------------------------------------------------------------------------- #
# Arm builders — construct + load each checkpoint in its real save format       #
# --------------------------------------------------------------------------- #
def _load_ck(path, device):
    return torch.load(path, map_location=device, weights_only=True)


def build_flagship(ckpt, config_name, device) -> ArmSpec:
    from tanitad.config import (flagship4b_config, flagship4b_reduced_config,
                                flagship4b_smoke_config)
    from tanitad.models.fourbrain import WorldModel
    from tanitad.models.metric_dynamics import HierarchicalGrounding
    cfg = {"flagship4b": flagship4b_config,
           "flagship4b_reduced": flagship4b_reduced_config,
           "smoke": flagship4b_smoke_config}[config_name]()
    world = WorldModel(cfg)
    ck = _load_ck(ckpt, device)
    world.load_state_dict(ck["model"] if "model" in ck else ck)
    world = world.to(device).eval()
    step_readout = None
    notes = []
    if "grounding" in ck:
        gr = HierarchicalGrounding(world.state_dim).to(device).eval()
        gr.load_state_dict(ck["grounding"])
        step_readout = gr.step["op"]
    else:
        notes.append("no 'grounding' key — grounded-rollout ADE unavailable")
    return ArmSpec(
        name="flagship", kind="frame", model=world,
        window=world.predictor.cfg.window, encode_window=world.encode_window,
        encode_one=world.encode, state_dim=world.state_dim,
        step=int(ck.get("step", -1)) if isinstance(ck, dict) else -1,
        predictor=world.predictor,
        imagine1=lambda s, a: world.imagine(s, a)[1],
        grounded_step_readout=step_readout,
        native_label="grounded operative rollout under true actions "
                     "(rollout_decode -> SE(2))",
        notes=notes)


def build_refa(ckpt, adapter, smoke, device, n_tokens=256, d_dino=768) -> ArmSpec:
    from tanitad.config import PredictorConfig
    from tanitad.models.metric_dynamics import StepDisplacementReadout
    from tanitad.refs.refa import RefAModel, refa_predictor_config
    pred_cfg = (PredictorConfig(d_model=64, depth=2, n_heads=2, window=4,
                                horizons=(1, 2, 4), action_dim=2)
                if smoke else refa_predictor_config())
    model = RefAModel(pred_cfg=pred_cfg, adapter_kind=adapter,
                      n_tokens=n_tokens, d_dino=d_dino)
    ck = _load_ck(ckpt, device)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    step_readout = None
    notes = []
    if "step_readout" in ck:
        step_readout = StepDisplacementReadout(model.state_dim).to(device).eval()
        step_readout.load_state_dict(ck["step_readout"])
    else:
        notes.append("no 'step_readout' key — grounded-rollout ADE unavailable")
    return ArmSpec(
        name="refa", kind="feature", model=model,
        window=model.pred_cfg.window, encode_window=model.encode_window,
        encode_one=model.encode, state_dim=model.state_dim,
        step=int(ck.get("step", -1)), predictor=model.predictor,
        imagine1=lambda s, a: model.predict(s, a)[1],
        grounded_step_readout=step_readout,
        native_label="grounded operative rollout under true actions "
                     "(adapter state; rollout_decode -> SE(2))",
        notes=notes)


def build_refb(ckpt, smoke, device) -> ArmSpec:
    from tanitad.refs.refb import RefBModel, refb_config, refb_smoke_config
    cfg = refb_smoke_config() if smoke else refb_config()
    model = RefBModel(cfg)
    ck = _load_ck(ckpt, device)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model = model.to(device).eval()

    def native_wp(frames_win):
        wp = model(frames_win)["waypoints"]
        return torch.stack([wp[k] for k in WP_STEPS], dim=1)   # [B,4,2]

    return ArmSpec(
        name="refb", kind="frame", model=model, window=cfg.window,
        encode_window=model.encode_window, encode_one=model.encode,
        state_dim=model.state_dim, step=int(ck.get("step", -1)),
        native_waypoints=native_wp,
        native_label="direct tactical waypoint head (behaviour cloning, "
                     "NO world model)",
        notes=["REF-B is the pre-registered no-world-model reference: no "
               "imagination (D2/D3 N/A), no grounded rollout — its native "
               "trajectory is the BC waypoint head"])


# --------------------------------------------------------------------------- #
# Comparison + verdict                                                         #
# --------------------------------------------------------------------------- #
def _min_winner(vals: dict):
    v = {k: x for k, x in vals.items() if x is not None}
    return (min(v, key=v.get) if v else None)


def _max_winner(vals: dict):
    v = {k: x for k, x in vals.items() if x is not None}
    return (max(v, key=v.get) if v else None)


def build_verdict(per_arm: dict, baselines: dict) -> dict:
    """Phase-0 hierarchy-edge verdict on the NECESSARY (decode + open-loop)
    conditions. Explicitly NOT a driving-competence claim — D4-D6 closed-loop
    arbitrate (gates.py doctrine)."""
    d1 = {a: r["decode"]["d1_ade_0_2s"] for a, r in per_arm.items()}
    grounded = {a: (r["grounded"]["ade_0_2s"] if r.get("grounded") else None)
                for a, r in per_arm.items()}
    d2 = {a: (r["imagination"]["d2_dir_acc"] if r.get("imagination") else None)
          for a, r in per_arm.items()}
    cv = baselines["constant_velocity"]["ade_0_2s"]
    per_metric = {
        "d1_decode_ade_0_2s": {"winner_lowest": _min_winner(d1), "values": d1,
                               "note": "frozen-probe parity metric (identical "
                                       "code path); lower is better"},
        "grounded_traj_ade_0_2s": {"winner_lowest": _min_winner(grounded),
                                   "values": grounded,
                                   "note": "per-arch mechanism, identical metric"},
        "d2_direction_acc": {"winner_highest": _max_winner(d2), "values": d2,
                             "note": "imagination usable for selection; higher "
                                     "is better (REF-B N/A)"},
    }
    fl = per_arm.get("flagship")
    edge = None
    if fl is not None:
        others_d1 = [v for a, v in d1.items() if a != "flagship" and v is not None]
        others_gr = [grounded[a] for a in grounded
                     if a != "flagship" and grounded[a] is not None]
        fl_d1 = d1["flagship"]
        fl_gr = grounded["flagship"]
        beats_refs_d1 = (bool(all(fl_d1 < o for o in others_d1))
                         if others_d1 else None)
        beats_refs_gr = (bool(fl_gr is not None
                              and all(fl_gr < o for o in others_gr))
                         if others_gr else None)
        beats_cv = (bool(fl_gr < cv) if fl_gr is not None else None)
        edge = {
            "flagship_d1_ade_0_2s": fl_d1,
            "flagship_grounded_ade_0_2s": fl_gr,
            "flagship_beats_refs_on_d1_decode": beats_refs_d1,
            "flagship_beats_refs_on_grounded_traj": beats_refs_gr,
            "flagship_grounded_beats_cv_floor": beats_cv,
            "flagship_d1_gate": fl["decode"]["d1_status"],
            "flagship_d2_gate": (fl["imagination"]["d2_status"]
                                 if fl.get("imagination") else None),
            "flagship_d3_gate": (fl["imagination"]["d3_status"]
                                 if fl.get("imagination") else None),
        }
    return {
        "per_metric": per_metric,
        "hierarchy_edge_necessary_conditions": edge,
        "DOCTRINE": ("D1-D3 + open-loop grounded ADE are NECESSARY, not "
                     "sufficient (arXiv 2512.24497). This verdict decides the "
                     "decode/open-loop conditions ONLY; the closed-loop gates "
                     "D4-D6 (interactive success, blocked-route, simple->complex "
                     "slope) remain the arbiters of the hierarchy edge and are "
                     "computed in sim, not here."),
    }


def render_markdown(report: dict) -> str:
    arms = list(report["arms"].keys())
    L = ["# Phase-0 three-arm comparison",
         "",
         f"- git: `{report['git_hash']}`  |  device: {report['eval']['device']}",
         f"- common val episodes: **{report['val']['n_common_episodes']}** "
         f"({report['val']['n_windows']} windows), "
         f"episode ids: `{report['val']['common_episode_ids'][:12]}"
         f"{'...' if len(report['val']['common_episode_ids']) > 12 else ''}`",
         f"- arms: " + ", ".join(f"{a} (step {report['arms'][a]['step']})"
                                 for a in arms),
         ""]
    b = report["baselines"]
    L += ["## Trivial baselines (shared, model-free floor)",
          "", "| baseline | ade@1s | ade@2s | ade_0_2s |",
          "|---|---|---|---|"]
    for n in dd.BASELINES:
        m = b[n]
        L.append(f"| {n} | {m['ade@1s']:.3f} | {m['ade@2s']:.3f} | "
                 f"{m['ade_0_2s']:.3f} |")
    # main comparison table
    L += ["", "## Comparison table (arm x metric)", ""]
    header = "| metric | " + " | ".join(arms) + " |"
    L += [header, "|" + "---|" * (len(arms) + 1)]

    def row(label, fn):
        cells = []
        for a in arms:
            try:
                cells.append(fn(report["arms"][a]))
            except Exception:
                cells.append("—")
        return f"| {label} | " + " | ".join(cells) + " |"

    def f3(x):
        return "—" if x is None else f"{x:.3f}"
    L += [
        row("D1 decode ade_0_2s (parity, frozen probe)",
            lambda r: f3(r["decode"]["d1_ade_0_2s"])),
        row("D1 gate (camera <1.0m)", lambda r: r["decode"]["d1_status"]),
        row("best held-out ade_0_2s (ladder)",
            lambda r: f3(r["decode"]["best_heldout_ade_0_2s"])),
        row("oracle-ceiling ade_0_2s (in-dist)",
            lambda r: f3(r["decode"]["oracle_ceiling_ade_0_2s"])),
        row("held-out / oracle ratio",
            lambda r: f3(r["decode"]["heldout_over_oracle"])),
        row("grounded/native traj ade_0_2s",
            lambda r: f3(r["grounded"]["ade_0_2s"]) if r.get("grounded") else "N/A"),
        row("native beats CV (overall)",
            lambda r: str(r["grounded"]["beats_cv_overall"]) if r.get("grounded") else "N/A"),
        row("native beats CV (straight)",
            lambda r: (str(r["grounded"]["straight"]["beats_cv"])
                       if r.get("grounded") and r["grounded"]["straight"] else "N/A")),
        row("D2 direction-acc (imag usable)",
            lambda r: f3(r["imagination"]["d2_dir_acc"]) if r.get("imagination") else "N/A"),
        row("D2 gate (>0.7)",
            lambda r: r["imagination"]["d2_status"] if r.get("imagination") else "N/A"),
        row("D3 imagined/oracle ratio",
            lambda r: f3(r["imagination"]["d3_ratio"]) if r.get("imagination") else "N/A"),
        row("D3 gate (<=1.5x)",
            lambda r: r["imagination"]["d3_status"] if r.get("imagination") else "N/A"),
    ]
    # verdict
    v = report["verdict"]
    L += ["", "## Per-metric winner", "",
          "| metric | winner | note |", "|---|---|---|"]
    for m, d in v["per_metric"].items():
        w = d.get("winner_lowest") or d.get("winner_highest") or "—"
        L.append(f"| {m} | **{w}** | {d['note']} |")
    edge = v["hierarchy_edge_necessary_conditions"]
    if edge:
        L += ["", "## Hierarchy-edge necessary conditions (flagship)", ""]
        for k, val in edge.items():
            L.append(f"- `{k}`: **{val}**")
    L += ["", "## Doctrine", "", v["DOCTRINE"], ""]
    for a in arms:
        for nt in report["arms"][a]["notes"]:
            L.append(f"- [{a}] {nt}")
    return "\n".join(L) + "\n"


def compare(arms: list, frame_val, feat_by_id, device, *, n_splits, val_frac,
            seed, mlp_epochs, batch, stride, git_hash, oracle_target) -> dict:
    grid = build_reference_grid(frame_val, arms[0].window, stride)
    # every arm must share the window (identical anchors) — fail loud otherwise.
    for a in arms:
        assert a.window == grid.window, (
            f"arm {a.name} window {a.window} != grid window {grid.window}; "
            "identical-window is required for same-anchor parity")
    base_de = {n: dd.scalar_metrics(dd.de_of(grid.base[n], grid.gt))
               for n in dd.BASELINES}
    per_arm = {}
    with strict_numerics():
        for arm in arms:
            print(f"[compare] arm={arm.name} kind={arm.kind} "
                  f"state_dim={arm.state_dim} step={arm.step}", flush=True)
            if arm.kind == "frame":
                states = collect_states_frame(arm, grid, device, batch)
                fr = grid.episodes[0].frames[:16]
                fr = fr.float().div(255.0) if fr.dtype == torch.uint8 else fr.float()
                i2 = I2Input(encode_fn=arm.encode_one, frames=fr.to(device),
                             batch_size=8)
            else:
                states = collect_states_feature(arm, grid, feat_by_id, device, batch)
                eid0 = int(grid.episodes[0].episode_id)
                fs = feat_by_id[eid0]["feats_fp16"][:16].float().to(device)
                i2 = I2Input(encode_fn=arm.encode_one, frames=fs, batch_size=8)
            decode = decode_parity(states, grid, arm, device, n_splits,
                                   val_frac, seed, mlp_epochs, i2)
            grounded = grounded_rollout_ade(arm, grid, feat_by_id, device, batch)
            imag = imagination_gates(arm, grid, feat_by_id, device, batch,
                                     val_frac, seed, i2)
            per_arm[arm.name] = {"step": arm.step, "kind": arm.kind,
                                 "state_dim": arm.state_dim, "decode": decode,
                                 "grounded": grounded, "imagination": imag,
                                 "notes": arm.notes}
    verdict = build_verdict(per_arm, base_de)
    return {
        "exp": "phase0-three-arm-compare", "git_hash": git_hash,
        "eval": {"device": str(device), "n_splits": n_splits,
                 "val_frac": val_frac, "seed": seed, "stride": stride,
                 "mlp_epochs": mlp_epochs, "waypoint_steps": list(WP_STEPS),
                 "camera_ade_max_m": CAMERA_ADE_MAX,
                 "oracle_ceiling_target_m": oracle_target},
        "val": {"n_common_episodes": len(frame_val),
                "n_windows": len(grid.windows),
                "common_episode_ids": [int(e.episode_id) for e, _ in frame_val]},
        "baselines": base_de,
        "arms": per_arm,
        "verdict": verdict,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flagship-ckpt", default=None)
    ap.add_argument("--flagship-config", default="flagship4b",
                    choices=["flagship4b", "flagship4b_reduced", "smoke"])
    ap.add_argument("--refa-ckpt", default=None)
    ap.add_argument("--refa-adapter", default="grid", choices=["grid", "pool"])
    ap.add_argument("--refa-smoke", action="store_true",
                    help="REF-A trained with --smoke (small predictor trunk)")
    ap.add_argument("--refb-ckpt", default=None)
    ap.add_argument("--refb-smoke", action="store_true")
    ap.add_argument("--frame-cache-dirs", nargs="+", required=True,
                    help="val frame caches (flagship + REF-B) — <root>/*val*/ep_*.pt")
    ap.add_argument("--refa-feat-dir", default=None,
                    help="val DINO features for REF-A (same episode ids)")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--episodes", type=int, default=150,
                    help="common val subset size (per frame cache dir)")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--n-splits", type=int, default=8)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mlp-epochs", type=int, default=60)
    ap.add_argument("--pose-tol", type=float, default=1e-2,
                    help="max |frame-feature| pose deviation to accept as same clip")
    ap.add_argument("--oracle-target", type=float, default=1.65,
                    help="grounded-ADE maturity reference (m). Repo-documented "
                         "oracle ceiling is 1.52-1.65m; a 0.68m target has been "
                         "cited by steering but is NOT in-tree — override here "
                         "once confirmed. The harness ALSO measures the ceiling.")
    ap.add_argument("--git-hash", default="unknown")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0
              else "cpu")
    need_feature = args.refa_ckpt is not None

    frame_val = load_frame_val(args.frame_cache_dirs, args.episodes)
    assert frame_val, "no frame val episodes loaded"
    feat_eps = load_feature_val(args.refa_feat_dir, args.episodes) if need_feature else []
    if need_feature and not feat_eps:
        raise SystemExit("--refa-ckpt given but no DINO feature val episodes "
                         "found under --refa-feat-dir")
    common_frame_val, feat_by_id, common_ids = load_common_val(
        frame_val, feat_eps, need_feature, args.pose_tol)
    assert common_frame_val, "no common val episodes across arms"
    print(f"[compare] {len(common_frame_val)} common val episodes "
          f"(need_feature={need_feature})", flush=True)

    arms = []
    if args.flagship_ckpt:
        arms.append(build_flagship(args.flagship_ckpt, args.flagship_config, device))
    if args.refa_ckpt:
        arms.append(build_refa(args.refa_ckpt, args.refa_adapter, args.refa_smoke,
                               device))
    if args.refb_ckpt:
        arms.append(build_refb(args.refb_ckpt, args.refb_smoke, device))
    assert arms, "no arms — supply at least one of --flagship/--refa/--refb-ckpt"

    report = compare(arms, common_frame_val, feat_by_id, device,
                     n_splits=args.n_splits, val_frac=args.val_frac,
                     seed=args.seed, mlp_epochs=args.mlp_epochs, batch=args.batch,
                     stride=args.stride, git_hash=args.git_hash,
                     oracle_target=args.oracle_target)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "arm_compare.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "arm_compare.md").write_text(render_markdown(report))
    print("\n" + render_markdown(report), flush=True)
    print(f"[compare] report -> {out/'arm_compare.json'}", flush=True)
    print("ARM_COMPARE_DONE", flush=True)


if __name__ == "__main__":
    main()
