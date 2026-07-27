#!/usr/bin/env python3
"""GAP 1 — emit the REAL-vs-IMAGINED decode bands for a checkpoint, eval-side.

WHY THIS EXISTS
---------------
v1's trainer logs a PAIRED pair of numbers every step:

    g_{lvl}_mid_de_m   MetricInverseDynamics on REAL latent pairs (z_t, z_{t+k})
    g_{lvl}_fwd_ade_m  StepDisplacementReadout on the predictor's IMAGINED rollout

Their ratio is the program's deepest architectural finding (v1: 2.36x at 0-1k ->
33.3x at 28-30k).  X4 found NO `g_*` key in any committed v4 log and concluded
v4 "does not carry the instrument".

⚠️ THE CORRECTION THIS SCRIPT IS BUILT ON.  The string-level claim is true; the
inference is not.  `train_flagship_v4.py:95` calls `flagship_loss(world,
grounding, ...)`, `flagship_losses.py:359` calls `grounding_losses(...)` and
merges its log verbatim (`**g_log`, :425), and `train_flagship_v4.py:861` puts
`grounding.parameters()` in the optimizer.  So v4 COMPUTES all six numbers every
step and TRAINS the heads -- it just forwards ONE of them
(`train_flagship_v4.py:159`, `g_op_fwd_ade_m`) into the JSONL row.  The gap is
LOGGING, not instrumentation.  Therefore a v4 checkpoint's `grounding` is a
genuine v4-trained instrument, and on the FROM-SCRATCH arm it is purely v4-trained
(random init, no v1 warm-start: `train_flagship_v4.py:974`).

WHAT IT MEASURES
----------------
Reproduces `metric_dynamics.grounding_losses`' two logged quantities EXACTLY, on
the SAME batch and the SAME forward pass, per window so a paired episode-cluster
bootstrap is possible:

    mid_de[lvl]  = mean over that level's horizons of ||dxy_pred - dxy_true||
    fwd_ade[lvl] = ||accumulated waypoints - GT ego waypoints||, mean over k

plus the pre-registered controls (see PRE_REGISTRATION.md 1.3):
  * SHUFFLE control on invdyn['op'] -- must FAIL (mismatched real pairs).
  * dt self-check -- realised displacement / logged speed ~ 0.1 s.
  * strict load of BOTH `model` and `grounding`.

and the X2 `v0`-shortcut ablations (PRE_REGISTRATION.md 2), each recomputing BOTH
sides on the identical windows:
  * v0_zero     -- the v0 action channel set to 0
  * v0_shuffled -- v0 taken from another window in the batch
  * v0_half / v0_double -- v0 x 0.5 / x 2.0

⚠️ The REAL side MUST be bit-exactly unchanged under every ablation: `invdyn`
reads only encoder latents and the encoder never sees `v0`.  A non-zero move
means the ablation is wired wrong and every imagined number here is inadmissible
(falsifier F-A).

Corpus: the PARITY val cache `physicalai-val-0c5f7dac3b11`.  Nothing here
re-selects episodes: the first N cache files in sorted order are consumed whole.
🔒 PhysicalAI-AV is gated-confidential: only latents, poses and scalars are
written; no clip UUID, no frame, no raw content.

Usage (pod3):
  PYTHONPATH=/workspace/TanitAD/stack python v4_grounding_probe.py \
      --ckpt /workspace/v4instr/v4fs_ckpt.pt --arm v4fs \
      --cache-dir /workspace/s3parity/views/physicalai-val-0c5f7dac3b11 \
      --episodes 60 --out /workspace/v4instr/out
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

SPEED_SCALE = 10.0          # hard contract (D-A3); taniteval.rollout.append_ego
WINDOW = 8
LEVEL_CFG = {"op": ((1, 2, 4), 4), "tac": ((8, 16), 16), "str": ((20,), 20)}

# v1's committed train-band anchors, 28-30k (MANIFOLD_MISMATCH_RESEARCH.md 2.2,
# reproduced at 1e-4 by x4_log_sweep.py).  Used ONLY as the fidelity anchor.
V1_TRAIN_OP_MID = 1.0129
V1_TRAIN_OP_FWD = 0.0304
FIDELITY_FWD_TOL = 3.0
FIDELITY_MID_TOL = 2.0

ABLATIONS = ("baseline", "v0_zero", "v0_shuffled", "v0_gshuffled",
             "v0_half", "v0_double", "act_zero")
# ⚠️ `v0_shuffled` permutes WITHIN a 32-window chunk of ONE episode, where ego
# speed is strongly autocorrelated -- so it is a WEAK perturbation and its
# magnitude is reported (`mean_abs_dv0_mps`) rather than assumed.
# `v0_gshuffled` draws from a GLOBAL pool of every window's v0 across all
# episodes, which is the honest "a plausible but wrong speed" input.
# `act_zero` zeroes the TWO BASE (CAN) action channels and KEEPS v0 -- the
# discriminator that separates "v0 specifically" from "any action perturbation".


def build_model(ckpt_path, device):
    """WorldModel + HierarchicalGrounding, BOTH loaded strict.  Mirrors
    `eval_flagship_v4.load_v1_from_ck` / `load_v4_from_ck`'s trunk half."""
    import dataclasses as dc

    from tanitad.config import flagship4b_config
    from tanitad.models.fourbrain import WorldModel
    from tanitad.models.metric_dynamics import HierarchicalGrounding

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a_dim = int(ck["model"]["predictor.act_emb.0.weight"].shape[1])
    cfg = flagship4b_config()
    cfg.predictor = dc.replace(cfg.predictor, action_dim=a_dim)
    if cfg.tactical_pred is not None:
        cfg.tactical_pred = dc.replace(cfg.tactical_pred, action_dim=a_dim)
    cfg.encoder = dc.replace(cfg.encoder, grad_checkpoint=False)
    world = WorldModel(cfg)
    world.load_state_dict(ck["model"])                       # STRICT
    grounding = HierarchicalGrounding(world.state_dim)
    grounding.load_state_dict(ck["grounding"])               # STRICT
    return (world.to(device).eval(), grounding.to(device).eval(),
            int(ck.get("step", -1)), world.state_dim, a_dim)


@torch.no_grad()
def encode_episode(world, frames, device, batch):
    out = []
    for i in range(0, int(frames.shape[0]), batch):
        x = torch.as_tensor(frames[i:i + batch]).to(device)
        x = x.float().div_(255.0) if x.dtype == torch.uint8 else x.float()
        out.append(world.encode(x).float().cpu())
    return torch.cat(out)


def _apply_ablation(v0: torch.Tensor, kind: str, gen, pool=None):
    """Returns ``(v0_used, base_action_scale)``.

    ``v0`` is [B,1] ALREADY divided by SPEED_SCALE (the action-channel value).
    ``base_action_scale`` multiplies the TWO CAN action channels (1.0 = untouched).
    """
    if kind == "baseline":
        return v0, 1.0
    if kind == "act_zero":
        return v0, 0.0                       # v0 kept, CAN channels zeroed
    if kind == "v0_zero":
        return torch.zeros_like(v0), 1.0
    if kind == "v0_half":
        return v0 * 0.5, 1.0
    if kind == "v0_double":
        return v0 * 2.0, 1.0
    if kind == "v0_shuffled":
        perm = torch.randperm(v0.shape[0], generator=gen).to(v0.device)
        return v0[perm], 1.0
    if kind == "v0_gshuffled":
        idx = torch.randint(pool.shape[0], (v0.shape[0],), generator=gen)
        return pool.to(v0.device)[idx.to(v0.device)].view(-1, 1), 1.0
    raise ValueError(kind)


@torch.no_grad()
def probe(world, grounding, eps, states, device, stride, ablations,
          shuffle_seed=0):
    """Per-window mid_de / fwd_ade at every level, for every v0 ablation.

    Every ablation reuses the SAME windows, the SAME encoder latents and the SAME
    invdyn call order, so the arrays are paired element-by-element.
    """
    from tanitad.models.metric_dynamics import (decode_transitions,
                                                gt_ego_waypoints,
                                                relative_ego_pose,
                                                rollout_transitions)

    per = {ab: {f"{l}_{s}": [] for l in LEVEL_CFG for s in ("mid", "fwd")}
           for ab in ablations}
    per["baseline"]["op_mid_shuffled"] = []
    # ⚠️ `op_mid_shuffled` permutes partners WITHIN a <=32-window chunk of ONE
    # episode, where the latent is strongly autocorrelated -- a WEAK mismatch.
    # `op_mid_gshuffled` takes the partner from a DIFFERENT EPISODE, which is
    # the honest "this head is fed the wrong latent" control.
    per["baseline"]["op_mid_gshuffled"] = []
    eid, t0s, dt_num, dt_den = [], [], [], []
    dv0 = {ab: [] for ab in ablations}
    kmax = max(k for _, k in LEVEL_CFG.values())

    # GLOBAL v0 pool (m/s ÷ SPEED_SCALE) over every window of every episode --
    # built from poses only, so it costs nothing.
    pool = torch.cat([
        ep.poses[:, 3].float() / SPEED_SCALE for ep in eps]).cpu()

    for ep_i, ep in enumerate(eps):
        st = states[ep_i].to(device)
        T = min(int(st.shape[0]), int(ep.actions.shape[0]),
                int(ep.poses.shape[0]))
        starts = list(range(0, max(0, T - WINDOW - kmax), stride))
        if not starts:
            continue
        poses = ep.poses.float().to(device)
        acts = ep.actions.float().to(device)
        for i in range(0, len(starts), 32):
            ch = starts[i:i + 32]
            last = torch.tensor(ch, device=device) + WINDOW - 1
            s = torch.stack([st[t:t + WINDOW] for t in ch])            # [B,W,S]
            a0 = torch.stack([acts[t:t + WINDOW] for t in ch])
            fa0 = torch.stack([acts[t + WINDOW: t + WINDOW + kmax] for t in ch])
            v0 = poses[last, 3:4] / SPEED_SCALE                        # [B,1]
            pose_last = poses[last]
            fut_p = torch.stack([poses[t + WINDOW: t + WINDOW + kmax]
                                 for t in ch])                         # [B,K,4]
            fut_s = torch.stack([st[t + WINDOW: t + WINDOW + kmax]
                                 for t in ch]).to(device)              # [B,K,S]
            z_t = s[:, -1]

            # dt self-check: realised 1-step displacement / logged speed
            d1 = (fut_p[:, 0, :2] - pose_last[:, :2]).norm(dim=-1)
            dt_num.append(float(d1.sum()))
            dt_den.append(float(pose_last[:, 3].clamp_min(1e-6).sum()))

            g = torch.Generator(device="cpu").manual_seed(shuffle_seed + ep_i)
            for ab in ablations:
                v0a, base_scale = _apply_ablation(v0, ab, g, pool)
                dv0[ab].append(float(
                    (v0a - v0).abs().mul(SPEED_SCALE).sum()))
                a = torch.cat(
                    [a0 * base_scale,
                     v0a[:, None, :].expand(-1, a0.shape[1], -1)], -1)
                fa = torch.cat(
                    [fa0 * base_scale,
                     v0a[:, None, :].expand(-1, fa0.shape[1], -1)], -1)
                trans = rollout_transitions(world.predictor, s, a, fa, kmax)
                for lvl, (hor, fwd_k) in LEVEL_CFG.items():
                    de = 0.0
                    for kh in hor:
                        dp = grounding.invdyn[lvl](z_t, fut_s[:, kh - 1])
                        tgt = relative_ego_pose(pose_last, fut_p[:, kh - 1])
                        de = de + (dp[..., :2] - tgt[..., :2]).norm(dim=-1)
                    per[ab][f"{lvl}_mid"].append((de / len(hor)).cpu())
                    wp, _ = decode_transitions(grounding.step[lvl], trans, fwd_k)
                    gt = gt_ego_waypoints(pose_last, fut_p, range(1, fwd_k + 1))
                    per[ab][f"{lvl}_fwd"].append(
                        (wp - gt).norm(dim=-1).mean(1).cpu())

            # --- the deliberately failing input: mismatched real pairs -------
            gs = torch.Generator(device="cpu").manual_seed(
                1000 + shuffle_seed + ep_i)
            perm = torch.randperm(z_t.shape[0], generator=gs).to(device)
            de = 0.0
            for kh in LEVEL_CFG["op"][0]:
                dp = grounding.invdyn["op"](z_t, fut_s[perm, kh - 1])
                tgt = relative_ego_pose(pose_last, fut_p[:, kh - 1])
                de = de + (dp[..., :2] - tgt[..., :2]).norm(dim=-1)
            per["baseline"]["op_mid_shuffled"].append(
                (de / len(LEVEL_CFG["op"][0])).cpu())

            # --- the STRONG version: partner from a DIFFERENT EPISODE --------
            oj = (ep_i + 17) % len(eps)
            if oj == ep_i:
                oj = (ep_i + 1) % len(eps)
            ost = states[oj]
            de = 0.0
            for kh in LEVEL_CFG["op"][0]:
                ridx = torch.randint(int(ost.shape[0]), (z_t.shape[0],),
                                     generator=gs)
                dp = grounding.invdyn["op"](z_t, ost[ridx].to(device))
                tgt = relative_ego_pose(pose_last, fut_p[:, kh - 1])
                de = de + (dp[..., :2] - tgt[..., :2]).norm(dim=-1)
            per["baseline"]["op_mid_gshuffled"].append(
                (de / len(LEVEL_CFG["op"][0])).cpu())

            eid += [str(ep.episode_id)] * len(ch)
            t0s += ch
        del st
    out = {ab: {k: torch.cat(v) for k, v in d.items()}
           for ab, d in per.items()}
    n = max(1, len(eid))
    dv0m = {ab: sum(v) / n for ab, v in dv0.items()}
    return (out, eid, t0s, (sum(dt_num) / max(1e-9, sum(dt_den))), dv0m,
            float(pool.mean() * SPEED_SCALE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--arm", required=True, help="label, e.g. v4fs / v1")
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--ablations", default=",".join(ABLATIONS))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    abl = tuple(x for x in a.ablations.split(",") if x)

    from tanitad.data.mixing import load_episode

    t0 = time.time()
    world, grounding, step, S, a_dim = build_model(a.ckpt, a.device)
    print(f"[load] arm={a.arm} step={step} state_dim={S} action_dim={a_dim} "
          f"({time.time()-t0:.0f}s)", flush=True)

    files = sorted(f for f in os.listdir(a.cache_dir)
                   if f.startswith("ep_") and f.endswith(".pt"))[: a.episodes]
    eps, states = [], []
    for i, f in enumerate(files):
        ep = load_episode(os.path.join(a.cache_dir, f), mmap=True)
        states.append(encode_episode(world, ep.frames, a.device, a.batch))
        eps.append(ep)
        if i % 20 == 0:
            print(f"[encode] {i}/{len(files)} ({time.time()-t0:.0f}s)",
                  flush=True)
    print(f"[encode] done {len(eps)} eps ({time.time()-t0:.0f}s)", flush=True)

    per, eid, t0s, dt, dv0m, v0_mean = probe(world, grounding, eps, states,
                                             a.device, a.stride, abl)
    means = {ab: {k: float(v.mean()) for k, v in d.items()}
             for ab, d in per.items()}
    b = means["baseline"]

    ok_fwd = b["op_fwd"] <= FIDELITY_FWD_TOL * V1_TRAIN_OP_FWD
    ok_mid = b["op_mid"] <= FIDELITY_MID_TOL * V1_TRAIN_OP_MID
    ok_shuf = b["op_mid_shuffled"] > 1.5 * b["op_mid"]
    ok_dt = 0.09 <= dt <= 0.11

    # F-A: the REAL side must be BIT-EXACT under every v0 ablation
    real_exact = {}
    for ab in abl:
        if ab == "baseline":
            continue
        real_exact[ab] = {
            lvl: bool(torch.equal(per[ab][f"{lvl}_mid"],
                                  per["baseline"][f"{lvl}_mid"]))
            for lvl in LEVEL_CFG}

    torch.save({"per_window": per, "eid": eid, "t0": t0s, "arm": a.arm,
                "ckpt": a.ckpt, "step": step},
               os.path.join(a.out, f"grounding_perwindow_{a.arm}.pt"))

    out = {
        "experiment": "GAP 1 - v4 real-vs-imagined decode bands, eval-side",
        "arm": a.arm, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": "pod3 (A40)", "device": a.device, "torch": torch.__version__,
        "ckpt": a.ckpt, "ckpt_step": step, "state_dim": S, "action_dim": a_dim,
        "corpus": {"cache_dir": a.cache_dir, "n_episodes": len(eps),
                   "PARITY": "parity val split physicalai-val-0c5f7dac3b11",
                   "episode_reselection": "none - first N files, consumed whole"},
        "n_windows": len(eid), "n_episode_clusters": len(set(eid)),
        "window": WINDOW, "stride": a.stride, "level_cfg": {
            k: [list(v[0]), v[1]] for k, v in LEVEL_CFG.items()},
        "means": {ab: {k: round(v, 6) for k, v in d.items()}
                  for ab, d in means.items()},
        "ratios_real_over_imagined": {
            ab: {lvl: round(d[f"{lvl}_mid"] / max(1e-12, d[f"{lvl}_fwd"]), 4)
                 for lvl in LEVEL_CFG} for ab, d in means.items()},
        "imagined_degradation_x_vs_baseline": {
            ab: {lvl: round(d[f"{lvl}_fwd"]
                            / max(1e-12, b[f"{lvl}_fwd"]), 4)
                 for lvl in LEVEL_CFG}
            for ab, d in means.items() if ab != "baseline"},
        "ablation_magnitude": {
            "mean_abs_delta_v0_mps": {k: round(v, 4)
                                      for k, v in dv0m.items()},
            "corpus_mean_speed_mps": round(v0_mean, 4)},
        "prereg_validation": {
            "V1_fidelity_imagined": {
                "rule": f"op_fwd <= {FIDELITY_FWD_TOL}x {V1_TRAIN_OP_FWD}",
                "got": round(b["op_fwd"], 4), "PASS": bool(ok_fwd),
                "applies_to": "v1 only (the anchor); reported for every arm"},
            "V2_fidelity_real_pair": {
                "rule": f"op_mid <= {FIDELITY_MID_TOL}x {V1_TRAIN_OP_MID}",
                "got": round(b["op_mid"], 4), "PASS": bool(ok_mid),
                "applies_to": "v1 only (the anchor); reported for every arm"},
            "V3_shuffle_control_must_fail": {
                "rule": "op_mid_shuffled > 1.5x op_mid",
                "got": round(b["op_mid_shuffled"], 4), "PASS": bool(ok_shuf),
                "x_vs_matched": round(
                    b["op_mid_shuffled"] / max(1e-12, b["op_mid"]), 4),
                "caveat": "WITHIN-EPISODE permutation -> a weak mismatch; see "
                          "V3b"},
            "V3b_cross_episode_shuffle_must_fail": {
                "rule": "op_mid_gshuffled > 1.5x op_mid",
                "got": round(b["op_mid_gshuffled"], 4),
                "PASS": bool(b["op_mid_gshuffled"] > 1.5 * b["op_mid"]),
                "x_vs_matched": round(
                    b["op_mid_gshuffled"] / max(1e-12, b["op_mid"]), 4)},
            "V5_dt_self_check": {"rule": "0.09 <= dt <= 0.11",
                                 "got": round(dt, 5), "PASS": bool(ok_dt)},
            "FA_real_side_bit_exact_under_v0_ablation": real_exact},
        "elapsed_s": round(time.time() - t0, 1),
    }
    fp = os.path.join(a.out, f"grounding_bands_{a.arm}.json")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["means"], indent=1))
    print(json.dumps(out["ratios_real_over_imagined"], indent=1))
    print(json.dumps(out["prereg_validation"], indent=1))
    print("wrote", fp, flush=True)


if __name__ == "__main__":
    main()
