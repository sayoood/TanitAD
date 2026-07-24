"""eval_flagship_v4.py -- the v4-aware held-out eval harness.

WHY THIS EXISTS (2026-07-23): no v4-aware held-out eval driver existed before
this file. ``eval_flagship_v16.py`` STRICT-loads the v1.5/v1.6 head, which is
architecturally incompatible with v4's ``FlagshipV4Head`` (dense 20-step
horizons, factorised LAT x LON x DIST selection, the lambda_plan seam) -- and
nothing emitted a ``windows_<key>.pt`` for a v4 checkpoint, so
``taniteval.driving.from_windows()``'s episode-cluster-bootstrap primary
(ade@2s + miss@2m) was unreachable and CONTINUE/RESTART decisions on the whole
flagship line were BLOCKED (GATE_PROTOCOL.md).

TWO MODES. Per GATE_PROTOCOL.md O-03, MODE A must be run and must PASS before
MODE B's output is ever trusted to judge a checkpoint.

MODE A -- ``--canary-only`` (auto-selected when the checkpoint has no 'head'
    key, i.e. a plain flagship WorldModel like flagship4b-speedjerk-30k).
    Runs ONLY the WM-integrity canary: the deterministic operative-predictor
    rollout under TRUE actions -> grounding -> SE(2) -> ADE@2s. This is the
    SAME quantity ``train_flagship_v4.canary_rollout`` computes, and it is the
    SAME quantity behind flagship v1's registry headline (0.4522 heldout /
    0.4271 full-set) -- v1 has no separate "planner", so its canonical
    TanitEval number already IS this rollout (MODEL_REGISTRY.md 1.2: "the
    intent-free operative path that produces the trajectory ADE@2s scores").
    Use this against flagship4b-speedjerk-30k FIRST to prove the harness's
    encode / rollout / grounding / SE(2) plumbing is correct before it ever
    touches a v4 checkpoint.

MODE B -- a real v4/v4.1 checkpoint (keys: model, grounding, head[, goal_head]).
    Runs the PLANNER PATH (FlagshipV4Head-selected trajectory, lambda_plan=1,
    NOT fed true future actions) over the val cache, at BOTH:
      (i)  the head's own DENSE horizons (1..20 steps, train-loop-comparable
           -- this is what the trainer's in-loop ``evaluate_planner`` reports
           every ``--eval-every``, and it is NOT the same statistic as the
           historical "ade_0_2s": a mean over 20 dense steps 0.1-2.0s is
           diluted by the small early-horizon errors and reads LOWER than a
           mean over just the 4 endpoint waypoints).
      (ii) the historical 4-WAYPOINT convention (steps 5/10/15/20 = 0.5-2s,
           the ONLY convention any other arm in MODEL_REGISTRY.md is quoted
           in) -- persisted to windows_<key>.pt for
           ``taniteval.driving.from_windows()``'s episode-cluster-bootstrap
           ``ade_0_2s`` / ``miss_2m`` (the gate's actual primary metric).
    ALSO runs the WM canary on the (now jointly fine-tuned) trunk
    (-> wm_canary_ade_2s secondary) and reads seam_norm_ratio_max off the
    head's own forward telemetry (-> seam_norm_ratio_max secondary).

Usage (eval pod; PYTHONPATH must include this dir's parent AND this dir):

  # MODE A -- validate the harness against the KNOWN v1 number
  python3 eval_flagship_v4.py \\
      --ckpt /root/models/flagship-30k/ckpt.pt --canary-only \\
      --val-cache /root/valdata/physicalai-val-0c5f7dac3b11 \\
      --key v1-validation --out /root/taniteval/results/v1-validation.json

  # MODE B -- the real v4.1 gate eval (only AFTER mode A passes)
  python3 eval_flagship_v4.py \\
      --ckpt /root/models/flagship-v4.1-10k/ckpt_step10000.pt \\
      --anchors-dense /root/models/flagship-v4.1-10k/flagship_v4_anchors_dense.pt \\
      --val-cache /root/valdata/physicalai-val-0c5f7dac3b11 \\
      --key flagship-v4.1-10k --out /root/taniteval/results/flagship-v4.1-10k.json
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

WP_STEPS = (5, 10, 15, 20)           # 0.5/1/1.5/2 s @10 Hz -- the ONLY convention
                                      # any other MODEL_REGISTRY.md row is quoted in
K_MAX = max(WP_STEPS)
REGISTRY_V1_HELDOUT = 0.4522          # MODEL_REGISTRY.md 1.2, 8-split episode
                                      # jackknife heldout mean (flagship-30k)
REGISTRY_V1_FULLSET = 0.4271          # same row, plain corpus-wide mean -- the
                                      # methodology-matched target for a plain-
                                      # mean canary rollout (no split/bootstrap)
VALIDATION_TOL = 0.05                 # metres -- "small tolerance" per the brief


# ============================================================================
# shared setup -- the v1 trunk architecture EVERY flagship arm shares
# ============================================================================
def _eval_cfg():
    """CLAUDE.md source of truth: speed_input, action_dim=3, grad-ckpt OFF."""
    from tanitad.config import flagship4b_config
    cfg = flagship4b_config()
    cfg.speed_input = True
    cfg.predictor = dc.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dc.replace(cfg.tactical_pred, action_dim=3)
    object.__setattr__(cfg.encoder, "grad_checkpoint", False)
    return cfg


def _plan(cfg):
    """Byte-identical to train_flagship_v4.train()'s call, so the val cache
    windows exactly the way the real run's own in-loop eval windowed it."""
    from tanitad.train.flagship_losses import horizon_plan
    return horizon_plan(cfg, op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20)


def build_val_dataset_base(val_cache, cfg, plan):
    """Plain FlagshipWindowDataset (v1/v2.1 keys only) -- used for MODE A so the
    validation exercises the MINIMUM moving parts (no v4 label minting)."""
    from tanitad.data.mixing import load_episode
    from train_flagship4b import FlagshipWindowDataset
    files = sorted(Path(val_cache).glob("ep_*.pt"))
    if not files:
        raise SystemExit(f"[v4-eval] no ep_*.pt under {val_cache}")
    eps = [load_episode(str(p), mmap=True) for p in files]
    ds = FlagshipWindowDataset(eps, window=cfg.predictor.window,
                               max_horizon=plan.max_horizon,
                               maneuver_h=plan.maneuver_h,
                               channels=cfg.encoder.in_channels)
    print(f"[v4-eval] MODE A val dataset (base): {len(eps)} episodes, "
          f"{len(ds)} windows (window={cfg.predictor.window} "
          f"max_horizon={plan.max_horizon})", flush=True)
    return ds


def build_val_dataset_v4(val_cache, cfg, plan):
    """FlagshipV4Dataset (mints v3 factorised + strategic labels on the fly) --
    needed for MODE B because the head's _goal_inputs reads vt_band/route/
    route_graded off the batch."""
    from tanitad.data.mixing import load_episode
    from flagship_v4_data import FlagshipV4Dataset
    files = sorted(Path(val_cache).glob("ep_*.pt"))
    if not files:
        raise SystemExit(f"[v4-eval] no ep_*.pt under {val_cache}")
    eps = [load_episode(str(p), mmap=True) for p in files]
    ds = FlagshipV4Dataset(eps, window=cfg.predictor.window,
                           max_horizon=plan.max_horizon, maneuver_h=plan.maneuver_h,
                           channels=cfg.encoder.in_channels)
    print(f"[v4-eval] MODE B val dataset (v4): {len(eps)} episodes, "
          f"{len(ds)} windows (window={cfg.predictor.window} "
          f"max_horizon={plan.max_horizon})", flush=True)
    return ds


def run_canary(world, grounding, ds_val, device, episodes, stride, batch):
    """Thin wrapper around train_flagship_v4.canary_rollout -- reused, not
    reimplemented, so this harness inherits the SAME rollout/grounding/SE(2)
    mechanics the design already anchors flagship v1's 0.452 against."""
    from train_flagship_v4 import canary_rollout
    t0 = time.time()
    out = canary_rollout(world, grounding, ds_val, device, horizons=WP_STEPS,
                         k_max=K_MAX, episodes=episodes, stride=stride,
                         batch=batch, amp=(str(device) == "cuda"))
    out["wallclock_s"] = round(time.time() - t0, 1)
    return out


# ============================================================================
# MODE A -- load a plain v1-shaped checkpoint (model + grounding, no head)
# ============================================================================
def load_v1_from_ck(ck: dict, device):
    """Inline equivalent of v15_prep.load_frozen_v1, taking an ALREADY-loaded
    ckpt dict (avoids reading a 3+ GB file twice). Refuses a non-speed trunk
    the same way (near-identical-name inversion risk, CLAUDE.md source of
    truth)."""
    from tanitad.models.fourbrain import WorldModel
    from tanitad.train.flagship_losses import build_grounding

    sd = ck["model"]
    a_dim = sd["predictor.act_emb.0.weight"].shape[1]
    if a_dim != 3:
        raise SystemExit(
            f"REFUSING: predictor action_dim={a_dim}, not 3. This must be the "
            "speed arm (flagship4b-speedjerk-30k), NOT the no-speed ablation "
            "control flagship4b-phase0-30k (CLAUDE.md source of truth).")
    cfg = _eval_cfg()
    world = WorldModel(cfg)
    world.load_state_dict(sd)                             # STRICT
    world = world.to(device).eval()
    for p in world.parameters():
        p.requires_grad_(False)
    grounding = build_grounding(world.state_dim, device=device)
    grounding.load_state_dict(ck["grounding"])            # STRICT
    grounding.eval()
    for p in grounding.parameters():
        p.requires_grad_(False)
    step = int(ck.get("step", -1))
    print(f"[v4-eval] MODE A: loaded v1-shaped ckpt, step={step}, "
          f"state_dim={world.state_dim} (FROZEN)", flush=True)
    return world, grounding, step


# ============================================================================
# MODE B -- load a v4 checkpoint (model + grounding + head [+ goal_head])
# ============================================================================
def load_v4_from_ck(ck: dict, device, head_config_path=None,
                    anchors_dense_path=None, cond_imagination_override=None):
    from tanitad.models.fourbrain import WorldModel
    from tanitad.models.flagship_v4 import FlagshipV4Head, V4Config, v4_config
    from tanitad.refs.refc import DecoderConfig
    from tanitad.train.flagship_losses import build_grounding

    a_dim = ck["model"]["predictor.act_emb.0.weight"].shape[1]
    if a_dim != 3:
        raise SystemExit(f"REFUSING: predictor action_dim={a_dim}, not 3 -- "
                         "not a speed-input v4 trunk.")

    cfg = _eval_cfg()
    world = WorldModel(cfg)
    world.load_state_dict(ck["model"])                    # STRICT
    world = world.to(device).eval()
    for p in world.parameters():
        p.requires_grad_(False)

    grounding = build_grounding(world.state_dim, device=device)
    grounding.load_state_dict(ck["grounding"])            # STRICT
    grounding.eval()
    for p in grounding.parameters():
        p.requires_grad_(False)

    hcfg = v4_config()
    src = "v4_config() defaults (NO sibling config.json found -- risk of "\
          "architecture mismatch if the real run overrode any field)"
    if head_config_path and Path(head_config_path).exists():
        hj = json.loads(Path(head_config_path).read_text())
        hc = dict(hj.get("head_cfg", hj))
        dec = hc.get("decoder")
        if isinstance(dec, dict):
            hc["decoder"] = DecoderConfig(**dec)
        for tk in ("horizons", "imag_read"):
            if tk in hc and isinstance(hc[tk], list):
                hc[tk] = tuple(hc[tk])
        hcfg = V4Config(**hc)
        src = f"sibling config.json ({head_config_path})"
    if cond_imagination_override is not None:
        hcfg.cond_imagination = cond_imagination_override
        src += f" [cond_imagination OVERRIDDEN to {cond_imagination_override}]"
    hcfg.state_dim = world.state_dim
    hcfg.window = cfg.predictor.window

    head = FlagshipV4Head(hcfg).to(device)
    if anchors_dense_path and Path(anchors_dense_path).exists():
        anc = torch.load(anchors_dense_path, map_location=device,
                         weights_only=False)
        head.load_anchors(
            (anc["anchors"] if isinstance(anc, dict) else anc).to(device))
        print(f"[v4-eval] loaded TRAINED dense anchors from {anchors_dense_path}",
              flush=True)
    else:
        print("[v4-eval] WARNING: no --anchors-dense found -- scoring against "
              "the head's DEFAULT (seed-0 FPS) anchor buffer. If the real run "
              "loaded a trained anchors file (check config.json "
              "args.anchors_dense) this will NOT reproduce its numbers.",
              flush=True)
    head.load_state_dict(ck["head"])                      # STRICT
    head = head.to(device).eval()
    for p in head.parameters():
        p.requires_grad_(False)

    step = int(ck.get("step", -1))
    print(f"[v4-eval] MODE B: loaded v4 head cfg from {src}\n"
          f"  n_anchors={hcfg.n_anchors} horizons={hcfg.horizons[0]}.."
          f"{hcfg.horizons[-1]} (n={len(hcfg.horizons)}) "
          f"cond(states/imag/vtarget/route)="
          f"{hcfg.cond_states}/{hcfg.cond_imagination}/{hcfg.cond_vtarget}/"
          f"{hcfg.cond_route} factorised={hcfg.factorised} step={step}",
          flush=True)
    return world, grounding, head, step, hcfg


@torch.no_grad()
def collect_planner(world, grounding, head, ds_val, device, dd, episodes,
                    stride, batch, wp_steps=WP_STEPS):
    """v4 PLANNER PATH: head-selected trajectory (lambda_plan=1, NOT fed true
    future actions), re-encoding the CURRENT (jointly fine-tuned) trunk.

    Returns ``(data, diag)``. ``data`` is windows_<key>.pt-ready
    (pred/gt/cv/eid/speed/head_deg/wp_steps/method) at the historical
    4-waypoint resolution, via driving_diagnostic's exact GT/CV/head_deg
    convention -- the SAME convention every other MODEL_REGISTRY.md row uses,
    so this arm is directly comparable. ``diag`` carries both the head's own
    DENSE-horizon quantities (train-loop-comparable) and a self-computed
    4-waypoint oracle/ADE (a cross-check against taniteval.driving's number
    computed from the SAME persisted windows via a completely different code
    path)."""
    from torch.utils.data import default_collate
    from train_flagship_v4 import _goal_inputs, _to_device
    from tanitad.models.flagship_v15 import v15_losses
    import refb_labels

    head.eval()
    horizons = head.cfg.horizons
    if not set(wp_steps) <= set(horizons):
        raise SystemExit(f"[v4-eval] wp_steps {wp_steps} not a subset of the "
                         f"head's own horizons {horizons}")
    wp_pos = [horizons.index(k) for k in wp_steps]

    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < episodes and t % stride == 0]
    if not sel:
        raise SystemExit("[v4-eval] no windows selected -- check "
                         "--episodes/--stride against the val cache size")

    P, G, C, EID, SPD, HDG = [], [], [], [], [], []
    dense_ade_sum = dense_oracle_sum = dense_selgap_sum = dense_missfde_sum = 0.0
    wp_oracle_sum = wp_ade_sum = 0.0
    seam_ratios: list[float] = []
    pose_cache: dict[int, torch.Tensor] = {}
    n = 0
    t0 = time.time()

    for b0 in range(0, len(sel), batch):
        idx = sel[b0:b0 + batch]
        items = [ds_val[i] for i in idx]
        b = _to_device(default_collate(items), device)
        v0 = b["pose_last"][:, 3].float()
        traj_tgt = refb_labels.waypoint_targets(
            b["pose_last"].float(), b["future_poses"][:, :max(horizons)].float(),
            horizons)
        st = world.encode_window(b["frames"])
        out = head(st, v0, lambda_plan=1.0, **_goal_inputs(head.cfg, b, v0))
        lg = v15_losses(out, head.decoder.anchors, traj_tgt)

        bs = len(idx)
        dense_ade_sum += float(lg["ade"]) * bs
        dense_oracle_sum += float(lg["oracle_ade"]) * bs
        dense_selgap_sum += float(lg["sel_gap"]) * bs
        fde_dense = (out["traj"][:, -1] - traj_tgt[:, -1]).norm(dim=-1)
        dense_missfde_sum += float((fde_dense > 2.0).float().sum())

        seam = out.get("telemetry", {}).get("seam_norm_ratio_max")
        if seam is not None:
            seam_ratios.append(float(seam))

        # ---- 4-waypoint sub-selection: the historical convention -----------
        pred4 = out["traj"][:, wp_pos]                             # [b,4,2]
        tgt4 = traj_tgt[:, wp_pos]                                  # [b,4,2]
        fan4 = out["anchor_traj"][:, :, wp_pos, :]                  # [b,N,4,2]
        fan_err4 = (fan4 - tgt4[:, None]).norm(dim=-1).mean(dim=-1)  # [b,N]
        wp_oracle_sum += float(fan_err4.min(dim=1).values.sum())
        wp_ade_sum += float((pred4 - tgt4).norm(dim=-1).mean(dim=-1).sum())
        P.append(pred4.float().cpu())
        n += bs

        for i in idx:
            e_i, t = ds_val.index[i]
            po = pose_cache.get(e_i)
            if po is None:
                po = torch.as_tensor(ds_val.episodes[e_i].poses,
                                     dtype=torch.float32)
                pose_cache[e_i] = po
            last = torch.tensor([t + ds_val.window - 1])
            G.append(dd.gt_ego_waypoints(po, last, wp_steps=wp_steps))
            C.append(dd.baseline_waypoints(po, last,
                                           wp_steps=wp_steps)["constant_velocity"])
            HDG.append(dd.net_heading_change_deg(po, last))
            EID.append(int(ds_val.episodes[e_i].episode_id))
        SPD.append(v0.float().cpu())
        if b0 % (batch * 10) == 0:
            print(f"  [v4-eval] planner-path {n}/{len(sel)} windows "
                  f"({time.time() - t0:.0f}s)", flush=True)

    data = {
        "pred": torch.cat(P), "gt": torch.cat(G).float(),
        "cv": torch.cat(C).float(), "eid": EID,
        "speed": torch.cat(SPD).float(), "head_deg": torch.cat(HDG).float(),
        "wp_steps": list(wp_steps),
        "method": (f"flagship-v4: joint-trained trunk (re-encoded via "
                  f"world.encode_window) + FlagshipV4Head dense-"
                  f"{len(horizons)}-step anchored planner (argmax-conf, "
                  f"lambda_plan=1.0, {head.decoder.anchors.shape[0]} anchors), "
                  f"4wp sub-selected at steps {wp_steps}"),
    }
    diag = {
        "n_windows": n,
        "wallclock_s": round(time.time() - t0, 1),
        "dense_headhorizons_ade_2s": dense_ade_sum / n,
        "dense_headhorizons_oracle_ade": dense_oracle_sum / n,
        "dense_headhorizons_sel_gap": dense_selgap_sum / n,
        "dense_headhorizons_miss_at_2m": dense_missfde_sum / n,
        "wp4_oracle_ade_0_2s": wp_oracle_sum / n,
        "wp4_ade_0_2s_selfcomputed": wp_ade_sum / n,
        "seam_norm_ratio_max": max(seam_ratios) if seam_ratios else None,
        "n_seam_samples": len(seam_ratios),
        "horizons_dense": list(horizons),
        "wp_steps": list(wp_steps),
    }
    return data, diag


def _dig_metric(ev: dict, metric: str):
    """Best-effort read of a metric's point estimate from a merged results
    JSON, trying the same node paths run_gate.py._cluster_node/_read_eval_metric
    tries (cluster_bootstrap.model / driving.cluster_bootstrap.model /
    headline / driving.headline / full_set.model / driving.full_set.model)."""
    for path in (("cluster_bootstrap", "model"),
                ("driving", "cluster_bootstrap", "model"),
                ("headline",), ("driving", "headline"),
                ("full_set", "model"), ("driving", "full_set", "model")):
        node = ev
        ok = True
        for k in path:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                ok = False
                break
        if ok and isinstance(node, dict) and metric in node:
            v = node[metric]
            return v.get("mean") if isinstance(v, dict) else v
    return None


# ============================================================================
# main
# ============================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        "eval_flagship_v4", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head-config", default=None,
                    help="sibling config.json (default: auto-detect "
                         "<ckpt-dir>/config.json)")
    ap.add_argument("--anchors-dense", default=None,
                    help="trained dense-anchor buffer. STRONGLY recommended to "
                         "pass explicitly -- the local path will differ from "
                         "whatever config.json's args.anchors_dense recorded "
                         "(that path lives on the TRAINING pod, not this one)")
    ap.add_argument("--cond-imagination", choices=("auto", "true", "false"),
                    default="auto")
    ap.add_argument("--canary-only", action="store_true",
                    help="force MODE A even if the ckpt has a 'head' key")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-bench", action="store_true",
                    help="skip the legacy taniteval.bench.run call")
    ap.add_argument("--skip-driving", action="store_true",
                    help="skip the taniteval.driving tier-0 panel")
    ap.add_argument("--results-dir", default=None,
                    help="dir for windows_<key>.pt / driving_<key>.json "
                         "(default: dirname(--out))")
    a = ap.parse_args(argv)

    device = a.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[v4-eval] WARNING: cuda unavailable, falling back to cpu",
              flush=True)
        device = "cpu"

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_dir = Path(a.results_dir) if a.results_dir else out_path.parent
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v4-eval] loading checkpoint {a.ckpt} ...", flush=True)
    t_load0 = time.time()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    print(f"[v4-eval] ckpt loaded in {time.time() - t_load0:.1f}s, "
          f"keys={sorted(ck.keys()) if isinstance(ck, dict) else type(ck)}",
          flush=True)
    is_v4 = isinstance(ck, dict) and ("head" in ck) and not a.canary_only

    cfg = _eval_cfg()
    plan = _plan(cfg)

    if not is_v4:
        # ------------------------------- MODE A -------------------------------
        ds_val = build_val_dataset_base(a.val_cache, cfg, plan)
        world, grounding, step = load_v1_from_ck(ck, device)
        can = run_canary(world, grounding, ds_val, device, a.episodes,
                         a.stride, a.batch)
        delta_heldout = can["canary_ade@2s"] - REGISTRY_V1_HELDOUT
        delta_fullset = can["canary_ade@2s"] - REGISTRY_V1_FULLSET
        reproduces = abs(delta_fullset) <= VALIDATION_TOL
        result = {
            "mode": "MODE_A_canary_only_validation",
            "evidence_class": "MEASURED (ours; artifact = this JSON)",
            "ckpt": a.ckpt, "ckpt_step": step, "key": a.key,
            "val_cache": a.val_cache, "episodes": a.episodes,
            "stride": a.stride, "batch": a.batch,
            "n_windows": can["n"], "wallclock_s": can["wallclock_s"],
            "canary_ade_2s_MEASURED": can["canary_ade@2s"],
            "registry_reference": {
                "ade_0_2s_heldout_8split_jackknife": REGISTRY_V1_HELDOUT,
                "ade_0_2s_full_set_plain_mean": REGISTRY_V1_FULLSET,
                "source": "Project Steering/MODEL_REGISTRY.md section 1.2 "
                          "(flagship4b-speedjerk-30k, TanitEval key "
                          "flagship-30k, step 29999)",
                "note": ("canary_rollout computes a PLAIN corpus-wide mean "
                        "over all selected windows -- the methodology-"
                        "matched comparison is the FULL-SET figure (0.4271), "
                        "not the 8-split episode-jackknife heldout mean "
                        "(0.4522, a different statistical construction).")},
            "delta_vs_full_set": round(delta_fullset, 4),
            "delta_vs_heldout": round(delta_heldout, 4),
            "tolerance_m": VALIDATION_TOL,
            "HARNESS_VALIDATED": bool(reproduces),
            "verdict": (
                "HARNESS VALIDATED -- reproduces the registry v1 number "
                "within tolerance; safe to proceed to a v4 checkpoint "
                "(GATE_PROTOCOL O-03 satisfied)."
                if reproduces else
                "HARNESS NOT VALIDATED -- does NOT reproduce v1's known "
                "number within tolerance. DO NOT proceed to score any v4 "
                "checkpoint with this harness until the discrepancy is "
                "found and fixed."),
        }
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        print(f"\n[v4-eval] -> {out_path}", flush=True)
        return 0 if reproduces else 1

    # ---------------------------------- MODE B --------------------------------
    ds_val = build_val_dataset_v4(a.val_cache, cfg, plan)
    ckpt_dir = Path(a.ckpt).parent
    head_cfg_path = a.head_config or (ckpt_dir / "config.json")
    anchors_path = a.anchors_dense
    if anchors_path is None and Path(head_cfg_path).exists():
        try:
            hj = json.loads(Path(head_cfg_path).read_text())
            cand = hj.get("args", {}).get("anchors_dense")
            if cand and Path(cand).exists():
                anchors_path = cand
        except Exception:
            pass
    cond_imag_override = {"auto": None, "true": True,
                          "false": False}[a.cond_imagination]

    world, grounding, head, step, hcfg = load_v4_from_ck(
        ck, device, head_config_path=head_cfg_path,
        anchors_dense_path=anchors_path,
        cond_imagination_override=cond_imag_override)
    del ck  # free the raw state-dict copy before the eval loop

    import driving_diagnostic as dd

    print("[v4-eval] running WM canary (wm_canary_ade_2s secondary)...",
          flush=True)
    can = run_canary(world, grounding, ds_val, device, a.episodes, a.stride,
                     a.batch)
    print(f"[v4-eval] canary_ade@2s={can['canary_ade@2s']:.4f} n={can['n']} "
          f"({can['wallclock_s']:.0f}s)", flush=True)

    print("[v4-eval] running the planner path (the gate primary)...",
          flush=True)
    data, diag = collect_planner(world, grounding, head, ds_val, device, dd,
                                 a.episodes, a.stride, a.batch)

    wp = results_dir / f"windows_{a.key}.pt"
    torch.save({k: data[k] for k in
               ("pred", "gt", "cv", "eid", "speed", "head_deg", "wp_steps")}, wp)
    print(f"[v4-eval] windows -> {wp} (enables the episode-cluster-bootstrap "
          f"primary via taniteval.driving.from_windows())", flush=True)

    res_json_path = results_dir / f"{a.key}.json"
    res: dict = {}
    if not a.skip_bench:
        try:
            from taniteval import bench
            res = bench.run(data)
        except Exception as ex:
            print(f"[v4-eval] bench.run FAILED (non-fatal -- driving.py still "
                 f"runs below): {type(ex).__name__}: {str(ex)[:300]}",
                 flush=True)
    res.setdefault("key", a.key)
    res["method"] = data["method"]
    res["ckpt"] = a.ckpt
    res["ckpt_step"] = step
    res["v4_diagnostics"] = diag
    res["wm_canary_ade_2s"] = can["canary_ade@2s"]
    res["wm_canary_n"] = can["n"]
    res_json_path.write_text(json.dumps(res, indent=2, default=str),
                             encoding="utf-8")
    print(f"[v4-eval] -> {res_json_path}", flush=True)

    if not a.skip_driving:
        try:
            from taniteval import driving as tdriving
            tdriving.run_and_save(a.key, res_dir=results_dir)
        except Exception as ex:
            print(f"[v4-eval] taniteval.driving FAILED: {type(ex).__name__}: "
                 f"{str(ex)[:300]}", flush=True)

    merged = json.loads(res_json_path.read_text())
    ade_02s = _dig_metric(merged, "ade_0_2s")
    miss_2m = _dig_metric(merged, "miss_2m")
    if miss_2m is None:
        miss_2m = _dig_metric(merged, "miss_rate@2m")

    def _sec(value, thr, direction):
        if value is None:
            return {"value": None, "threshold": thr, "pass": None,
                   "note": "NOT COMPUTED"}
        ok = (value <= thr) if direction == "<=" else (value >= thr)
        return {"value": value, "threshold": thr, "pass": bool(ok)}

    summary = {
        "key": a.key, "ckpt": a.ckpt, "ckpt_step": step,
        "evidence_class": "MEASURED (ours; artifacts = "
                          f"{res_json_path.name}, {wp.name})",
        "gate_primary_ade_0_2s": _sec(ade_02s, 0.60, "<="),
        "kill_secondaries": {
            "wm_canary_ade_2s": _sec(can["canary_ade@2s"], 0.55, "<="),
            "oracle_in_fan": {
                **_sec(diag["wp4_oracle_ade_0_2s"], 0.30, "<="),
                "note": "4-waypoint resolution (steps 5/10/15/20), comparable "
                        "to v1.5-ab's 0.3073 -- NOT the dense-20 "
                        "'oracle_ade@2s' the in-loop train log prints"},
            "miss_at_2m": _sec(miss_2m, 0.10, "<="),
            "seam_norm_ratio_max": _sec(diag["seam_norm_ratio_max"], 1.0, "<="),
            "encoder_touching_levers": {
                "value": 2, "threshold": 2, "pass": True,
                "evidence_class": "PUBLISHED (V4_FLAGSHIP_DESIGN.md / "
                                  "--print-launch design audit)",
                "note": "static design fact (lambda_plan + strategic = 2 of "
                        "2 encoder-touching levers, door CLOSED per "
                        "MODEL_REGISTRY.md retraction 07-21); not a GPU "
                        "measurement, not re-derived here"},
            "speed_benefit_recovered_frac": {
                "value": None, "pass": None,
                "note": "NOT BUILT this session -- new metric (P8), needs "
                        "its own definition off the two in-repo train logs "
                        "per V4_FLAGSHIP_DESIGN.md 17.3; no emitter exists "
                        "yet anywhere in the codebase"},
            "deploy_tick_p99_ms": {
                "value": None, "pass": None,
                "note": "NOT MEASURED this session -- needs the "
                        "efficiency.py latency-panel harness (CUDA-graph "
                        "capture, batch-1 profiling under gpu_lock.sh "
                        "exclusivity); out of scope for a correctness-first "
                        "harness build, flagged as the first thing to cut "
                        "per V4_FLAGSHIP_DESIGN.md 8 if it misses"},
            "nonav_route_beats_majority": {
                "value": None, "pass": None,
                "note": "NOT REACHABLE on this checkpoint -- v4.1's "
                        "goal_head (GoalScalarHead) only regresses "
                        "CONTINUOUS scalars (ttm/curv_3s/curv_5s/tspeed_5s); "
                        "no ROUTE classifier exists yet (P6 strategic "
                        "planner not landed). taniteval.hierarchy.py's "
                        "vision_route_beats_majority needs a nav-"
                        "conditioned route head this checkpoint does not "
                        "have -- this is the produced-goal fallback "
                        "(V4_FLAGSHIP_DESIGN.md 2.6) territory"},
        },
        "diagnostics_dense_headhorizons_train_loop_comparable": {
            k: diag[k] for k in (
                "dense_headhorizons_ade_2s", "dense_headhorizons_oracle_ade",
                "dense_headhorizons_sel_gap", "dense_headhorizons_miss_at_2m")},
        "cross_check_ade_0_2s_selfcomputed_vs_driving_py": {
            "selfcomputed_from_forward_pass": diag["wp4_ade_0_2s_selfcomputed"],
            "driving_py_from_persisted_windows": ade_02s,
            "agree_within_1pct": bool(
                ade_02s is not None and
                abs(diag["wp4_ade_0_2s_selfcomputed"] - ade_02s)
                < 0.01 * max(ade_02s, 1e-6)),
            "note": "two independent code paths over the SAME forward-pass "
                    "output (direct tensor math here vs. persisted-tensor "
                    "reload + taniteval.driving.tier0 there); disagreement "
                    "would indicate an ego-frame convention mismatch"},
    }
    diag_out = results_dir / f"{a.key}_v4_diagnostics.json"
    diag_out.write_text(json.dumps(summary, indent=2, default=str),
                        encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)
    print(f"\n[v4-eval] -> {diag_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
