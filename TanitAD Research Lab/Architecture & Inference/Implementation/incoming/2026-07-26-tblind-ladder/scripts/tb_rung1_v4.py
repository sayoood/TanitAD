#!/usr/bin/env python3
"""RUNG 1 — does the readout swap move **v4's** ``wm_canary_ade_2s``?  eval-only.

`…/2026-07-26-blind-imagination/` MEASURED on **v1** that decoding the grounded
rollout with the 20-step-calibrated ``grounding.step["str"]`` (or the 16-step
``["tac"]``) instead of the 4-step ``["op"]`` halves ``ade_0_2s``. It explicitly
did NOT claim the swap clears Bar B, because Bar B is a **v4** number.

This settles it on v4's own checkpoint (`flagship-v4-fromscratch-30k` @ 29999,
whose own ``metrics.json`` carries ``canary_ade@2s = 1.1409059762954712``).

⛔ REPRODUCTION GATE FIRST (PRE_REGISTRATION.md §4): the ``op`` readout must
reproduce that committed number within ±0.01 m. Nothing else is read until it
does.

⛔ AND A DELIBERATELY FAILING INPUT (M3, both directions): a RANDOMLY
RE-INITIALISED step readout must produce a much worse canary. Without it, a
harness that silently ignored the ``level`` argument would report three
identical numbers and look like a clean negative.

SELF-CONTAINED on purpose: it loads the world + grounding itself (the canary
needs nothing else) so it does not depend on ``eval_flagship_v4.py`` being
present on the host. The rollout mechanics are the program's own
``metric_dynamics.rollout_decode`` — reused, never reimplemented.

Host: **pod2 only** (A40, verified idle: 0 MiB, 0 %, no trainer).
pod1 (training), pod3 and the eval pod are never touched. The val cache is read
only.

Usage (pod2):
    PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/stack/scripts \
    python3 tb_rung1_v4.py \
      --ckpt /workspace/experiments/flagship-v4-fromscratch/ckpt.pt \
      --val-cache /workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11 \
      --out rung1_v4_readout_swap.json
"""
from __future__ import annotations

import argparse
import copy
import dataclasses as dc
import json
import sys
import time
from pathlib import Path

import torch

for _p in ("/workspace/TanitAD/stack", "/workspace/TanitAD/stack/scripts",
           "/root/TanitAD/stack", "/root/TanitAD/stack/scripts",
           "/root/taniteval", "/workspace/taniteval"):
    if Path(_p).is_dir() and _p not in sys.path:
        sys.path.insert(0, _p)

WP_STEPS = (5, 10, 15, 20)          # 0.5 / 1 / 1.5 / 2 s @ 10 Hz — the ONLY convention
K_MAX = max(WP_STEPS)
COMMITTED_V4_CANARY = 1.1409059762954712   # v4fs metrics.json + the raw gate JSON
GATE_TOL = 0.01                     # m — the v5 stream reproduced this same
                                    # quantity at 1.1381 through an independent
                                    # path (Δ 0.0028), so ±0.01 is a real gate
BAR_B = 0.55
LEVELS = ("op", "tac", "str")


# --------------------------------------------------------------------------- #
def eval_cfg():
    """The v1 trunk architecture every flagship arm shares (CLAUDE.md: speed
    input, action_dim 3, grad-checkpoint off for eval)."""
    from tanitad.config import flagship4b_config
    cfg = flagship4b_config()
    cfg.speed_input = True
    cfg.predictor = dc.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dc.replace(cfg.tactical_pred, action_dim=3)
    object.__setattr__(cfg.encoder, "grad_checkpoint", False)
    return cfg


def load_world_and_grounding(ckpt_path, device):
    """World model + HierarchicalGrounding, STRICT. Nothing else is needed for
    the canary, so nothing else is built (no head, no goal head)."""
    from tanitad.models.fourbrain import WorldModel
    from tanitad.train.flagship_losses import build_grounding
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a_dim = ck["model"]["predictor.act_emb.0.weight"].shape[1]
    if a_dim != 3:
        raise SystemExit(f"REFUSING: predictor action_dim={a_dim}, not 3 — "
                         "not a speed-input trunk (near-identical-name "
                         "inversion risk, CLAUDE.md source of truth).")
    cfg = eval_cfg()
    world = WorldModel(cfg)
    world.load_state_dict(ck["model"])                    # STRICT
    world = world.to(device).eval()
    grounding = build_grounding(world.state_dim, device=device)
    grounding.load_state_dict(ck["grounding"])            # STRICT
    grounding.eval()
    for p in list(world.parameters()) + list(grounding.parameters()):
        p.requires_grad_(False)
    step = int(ck.get("step", -1))
    has_head = "head" in ck
    del ck
    return world, grounding, cfg, step, has_head


def build_val_dataset(val_cache, cfg, mode="v4"):
    """The SAME windowing the real run's own in-loop eval used."""
    from tanitad.data import parity
    from tanitad.data.mixing import load_episode
    from tanitad.train.flagship_losses import horizon_plan
    plan = horizon_plan(cfg, op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20)
    parity.assert_val_cache(val_cache, label="--val-cache")
    files = sorted(Path(val_cache).glob("ep_*.pt"))
    if not files:
        raise SystemExit(f"no ep_*.pt under {val_cache}")
    eps = [load_episode(str(p), mmap=True) for p in files]
    if mode == "v4":
        from flagship_v4_data import FlagshipV4Dataset as DS
    else:
        from train_flagship4b import FlagshipWindowDataset as DS
    ds = DS(eps, window=cfg.predictor.window, max_horizon=plan.max_horizon,
            maneuver_h=plan.maneuver_h, channels=cfg.encoder.in_channels)
    print(f"[rung1] val dataset ({mode}): {len(eps)} episodes, {len(ds)} windows",
          flush=True)
    return ds


# --------------------------------------------------------------------------- #
@torch.no_grad()
def canary_per_window(world, grounding, ds_val, device, *, level="op",
                      episodes=40, stride=8, batch=16, amp=True,
                      randomise_readout=False):
    """``train_flagship_v4.canary_rollout`` with TWO changes and no others:

    1. the step readout is ``grounding.step[level]`` instead of hard-wired
       ``["op"]`` — the whole point of the experiment;
    2. it returns the **per-window** errors and their episode ids, so the
       program's episode-cluster bootstrap can be applied. The committed canary
       reports a plain mean, which cannot carry an interval.

    ``randomise_readout`` replaces the readout with a freshly-initialised one of
    the same class — the deliberately-failing input.
    """
    from tanitad.models.flagship_v15 import SPEED_SCALE
    from tanitad.models.metric_dynamics import gt_ego_waypoints, rollout_decode

    step_readout = grounding.step[level]
    if randomise_readout:
        step_readout = copy.deepcopy(step_readout)
        n_reset = 0
        for m in step_readout.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
                n_reset += 1
        if n_reset == 0:                       # the control must actually perturb
            for p in step_readout.parameters():
                torch.nn.init.normal_(p, std=0.02)
        step_readout = step_readout.to(device).eval()

    sel = [i for i, (e, t) in enumerate(ds_val.index)
           if e < episodes and t % stride == 0]
    if not sel:
        raise SystemExit("no val windows selected — wrong cache or stride")
    amp_on = amp and str(device) == "cuda"
    wp_idx = torch.tensor([k - 1 for k in WP_STEPS], device=device)
    errs, eids = [], []
    for b0 in range(0, len(sel), batch):
        idx = sel[b0:b0 + batch]
        items = [ds_val[i] for i in idx]
        eids += [int(ds_val.index[i][0]) for i in idx]
        fr = torch.stack([x["frames"] for x in items]).to(device)
        aw2 = torch.stack([x["actions"] for x in items]).to(device).float()
        fa2 = torch.stack([x["future_actions"] for x in items]).to(device).float()
        fp = torch.stack([x["future_poses"] for x in items]).to(device).float()
        pl = torch.stack([x["pose_last"] for x in items]).to(device).float()
        vch = (pl[:, 3] / SPEED_SCALE)[:, None, None]
        aw = torch.cat([aw2, vch.expand(-1, aw2.shape[1], -1)], dim=-1)
        fa = torch.cat([fa2, vch.expand(-1, fa2.shape[1], -1)], dim=-1)
        gt = gt_ego_waypoints(pl, fp, WP_STEPS)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
            states = world.encode_window(fr)
            wp_full, _ = rollout_decode(world.predictor, states, aw, fa,
                                        step_readout, K_MAX)
        pred = wp_full.index_select(1, wp_idx).float()
        errs.append((pred - gt).norm(dim=-1).mean(dim=1).cpu())
    return torch.cat(errs).double().numpy(), [str(e) for e in eids]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--val-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mode", choices=["v4", "base"], default="v4")
    a = ap.parse_args()

    from taniteval import ci as _ci

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"block": "tblind_ladder/rung1_v4_readout_swap", "version": "1.0.0",
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "host": "pod2", "torch": torch.__version__,
           "python": sys.version.split()[0], "ckpt": a.ckpt,
           "val_cache": a.val_cache, "episodes": a.episodes, "stride": a.stride,
           "dataset_mode": a.mode, "wp_steps": list(WP_STEPS),
           "committed_v4_canary": COMMITTED_V4_CANARY, "bar_B": BAR_B,
           "gate_tol_m": GATE_TOL,
           "trained_rollout_lengths_steps": {"op": 4, "tac": 16, "str": 20},
           "estimator": ("episode_cluster_bootstrap / paired "
                         "(taniteval/ci.py) B=2000 seed 0, resampling unit = "
                         "val episode")}

    t0 = time.time()
    world, grounding, cfg, step, has_head = load_world_and_grounding(a.ckpt, dev)
    res["ckpt_step"] = step
    res["ckpt_has_head"] = has_head
    res["ckpt_load_s"] = round(time.time() - t0, 1)
    res["readout_levels_present"] = sorted(grounding.step.keys())
    print(f"[rung1] ckpt step={step} head={has_head} "
          f"levels={res['readout_levels_present']} "
          f"({res['ckpt_load_s']}s)", flush=True)

    ds_val = build_val_dataset(a.val_cache, cfg, mode=a.mode)
    res["n_val_windows_total"] = len(ds_val)

    # ---- ⛔ GATE: the op readout must reproduce the committed number -------- #
    t0 = time.time()
    per_op, eid = canary_per_window(world, grounding, ds_val, dev, level="op",
                                    episodes=a.episodes, stride=a.stride,
                                    batch=a.batch)
    mean_op = float(per_op.mean())
    res["gate"] = {
        "level": "op", "n_windows": int(per_op.size),
        "n_episode_clusters": len(set(eid)),
        "wm_canary_ade_2s_recomputed": round(mean_op, 6),
        "committed": COMMITTED_V4_CANARY,
        "abs_diff": round(abs(mean_op - COMMITTED_V4_CANARY), 6),
        "tol": GATE_TOL,
        "GATE_PASS": bool(abs(mean_op - COMMITTED_V4_CANARY) < GATE_TOL),
        "seconds": round(time.time() - t0, 1)}
    print(f"[rung1] GATE op={mean_op:.6f} vs {COMMITTED_V4_CANARY:.6f} -> "
          f"{'PASS' if res['gate']['GATE_PASS'] else 'FAIL'}", flush=True)
    if not res["gate"]["GATE_PASS"]:
        res["BLOCKED"] = ("the op readout did not reproduce the committed v4 "
                          "canary; PRE_REGISTRATION.md §4 voids the run")
        Path(a.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        return 2

    # ---- ⛔ DELIBERATELY FAILING INPUT ------------------------------------- #
    per_rnd, _ = canary_per_window(world, grounding, ds_val, dev, level="op",
                                   episodes=a.episodes, stride=a.stride,
                                   batch=a.batch, randomise_readout=True)
    res["failing_input_control"] = {
        "what": "grounding.step['op'] re-initialised at random",
        "wm_canary_ade_2s": round(float(per_rnd.mean()), 6),
        "ratio_vs_trained": round(float(per_rnd.mean() / mean_op), 3),
        "PASS_much_worse": bool(per_rnd.mean() > 2.0 * mean_op),
        "why": ("if the harness silently ignored the readout argument all "
                "levels would agree and the negative would be vacuous")}
    print(f"[rung1] failing-input control: {res['failing_input_control']}",
          flush=True)

    # ---- ⭐ THE SWAP ------------------------------------------------------- #
    per = {"op": per_op}
    for lv in LEVELS:
        if lv == "op":
            continue
        if lv not in grounding.step:
            res.setdefault("missing_levels", []).append(lv)
            continue
        t0 = time.time()
        per[lv], _ = canary_per_window(world, grounding, ds_val, dev, level=lv,
                                       episodes=a.episodes, stride=a.stride,
                                       batch=a.batch)
        print(f"[rung1] level={lv} canary={per[lv].mean():.6f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    res["levels"] = {}
    for lv, v in per.items():
        res["levels"][lv] = {
            "wm_canary_ade_2s": round(float(v.mean()), 6),
            "ci95_episode_cluster_bootstrap": _ci.episode_cluster_bootstrap(
                v, eid, n_boot=2000, seed=0),
            "clears_bar_B": bool(v.mean() <= BAR_B),
            "fall_vs_op_x": round(float(mean_op / v.mean()), 4),
            "rel_change_vs_op_pct": round(float(v.mean() / mean_op - 1.0) * 100, 2)}
    res["paired_vs_op"] = {
        lv: _ci.paired_episode_cluster_bootstrap(per_op, v, eid, n_boot=2000,
                                                 seed=0)
        for lv, v in per.items() if lv != "op"}

    # ---- adjudication, mechanical, against §4 of the pre-registration ------ #
    best = min(per, key=lambda k: per[k].mean())
    res["VERDICT"] = {
        "best_level": best,
        "wm_canary_ade_2s_op": round(mean_op, 6),
        "wm_canary_ade_2s_best": round(float(per[best].mean()), 6),
        "fall_x": round(float(mean_op / per[best].mean()), 4),
        "required_fall_x_for_bar_B": round(COMMITTED_V4_CANARY / BAR_B, 4),
        "MOVES_the_canary": bool(
            best != "op" and (1.0 - per[best].mean() / mean_op) >= 0.25
            and res["paired_vs_op"].get(best, {}).get("separated", False)),
        "CLEARS_BAR_B": bool(per[best].mean() <= BAR_B),
        "statement": ("a canary improvement is NOT a Bar-B clearance unless "
                      "the swapped value is <= 0.55"),
        "rule_applied": "PRE_REGISTRATION.md §4, fixed before computing"}
    Path(a.out).write_text(json.dumps(res, indent=2, default=float),
                           encoding="utf-8")
    print(json.dumps(res["VERDICT"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
