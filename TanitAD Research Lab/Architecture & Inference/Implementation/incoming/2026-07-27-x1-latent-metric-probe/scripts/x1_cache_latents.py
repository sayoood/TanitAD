#!/usr/bin/env python3
"""X1 stage 1 — cache FROZEN-ENCODER latents + run the fidelity check.

Pre-registered as X1 in
``TanitAD Research Hub/Architecture & Inference/Research/2026-07-27-imagination-perception-manifold/``
(MANIFOLD_MISMATCH_RESEARCH.md §6):

    "Cache encoded latents for N val windows with v1's frozen encoder. Fit, on
     CPU, three probes from (z_t, z_{t+k}) -> true Δpose: ridge, the same 2-layer
     MLP as MetricInverseDynamics, and that MLP at 4x width. Report metre error
     at k in {1,2,4,20} against the mean-speed baseline.  This asks the only
     question that matters: is the metric ego-motion IN the latent at all, or did
     the trunk never encode it?"

THIS script does the GPU half: encode every frame of N episodes into the compact
state the grounding heads read, and — on the SAME latents — evaluate the TRAINED
heads so the probes have a PAIRED anchor rather than a cross-set one.

⚠️ CORPUS PARITY, STATED NOT BURIED.  The dev-box episode cache is keyed
``physicalai-val-bb543bdf7836`` (100 eps), **not** the parity key
``e438721ae894`` and not the eval pod's ``physicalai-val-0c5f7dac3b11``.  Nothing
produced here is cross-arm comparable to a committed leaderboard number, and no
number from here may enter MODEL_REGISTRY as an arm result.  It is admissible for
X1 because X1's question — "is metric ego-motion recoverable from this encoder's
latent at all" — is answered by a WITHIN-RUN contrast (fresh probe vs the trained
head vs baselines) measured on one and the same windows.

BOTH-DIRECTIONS VALIDATION, pre-registered before the run:
  * FIDELITY (must SUCCEED).  The trained ``grounding.step['op']`` decoding the
    predictor's IMAGINED rollout on these cached latents must land within 3x of
    v1's committed train-band 0.0304 m, and the trained ``grounding.invdyn['op']``
    on REAL pairs within 2x of 1.0129 m.  If the imagined decode is not small,
    the encoding/preprocessing path is wrong and NOTHING else here is admissible.
  * SHUFFLE CONTROL (must FAIL).  The same trained ``invdyn['op']`` fed
    MISMATCHED real pairs (z_t from one window, z_{t+k} from a different window)
    must degrade badly.  If it does not, the head is ignoring its input and every
    "recovery" number would be an artefact.

🔒 PhysicalAI-AV is gated-confidential: no clip UUID, no frame, no raw content is
written by this script.  Only latents, poses, and scalar statistics.

Usage (dev box):
  python x1_cache_latents.py --ckpt C:/Users/Admin/tanitad-data/eval/v1_speedjerk_ckpt.pt \
      --cache-dir C:/Users/Admin/tanitad-data/physicalai/_epcache/physicalai-val-bb543bdf7836 \
      --episodes 60 --out-dir C:/Users/Admin/tanitad-data/eval/x1_latents \
      --artifacts <repo>/.../2026-07-27-x1-latent-metric-probe/artifacts
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..",
                                    "..", "..", ".."))
for _p in (os.path.join(REPO, "stack"), os.path.join(REPO, "taniteval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tanitad.config import flagship4b_config                      # noqa: E402
from tanitad.data.mixing import load_episode                      # noqa: E402
from tanitad.models.fourbrain import WorldModel                   # noqa: E402
from tanitad.models.metric_dynamics import (                      # noqa: E402
    HierarchicalGrounding, decode_transitions, gt_ego_waypoints,
    relative_ego_pose, rollout_transitions)

SPEED_SCALE = 10.0          # hard contract (D-A3); taniteval.rollout.append_ego
WINDOW = 8
LEVEL_CFG = {"op": ((1, 2, 4), 4), "tac": ((8, 16), 16), "str": ((20,), 20)}

# v1's committed train-band anchors (MANIFOLD_MISMATCH_RESEARCH.md §2.2, 28-30k),
# reproduced by x4_log_sweep.py at 1e-4 before this script was written.
V1_TRAIN_OP_MID = 1.0129
V1_TRAIN_OP_FWD = 0.0304
FIDELITY_FWD_TOL = 3.0      # imagined decode must be within 3x  (pre-registered)
FIDELITY_MID_TOL = 2.0      # real-pair decode must be within 2x (pre-registered)


def build_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = flagship4b_config()
    object.__setattr__(cfg.predictor, "action_dim", 3)          # v1 = speed_input
    if cfg.tactical_pred is not None:
        object.__setattr__(cfg.tactical_pred, "action_dim", 3)
    object.__setattr__(cfg.encoder, "grad_checkpoint", False)
    model = WorldModel(cfg)
    model.load_state_dict(ck["model"])                          # STRICT
    grounding = HierarchicalGrounding(model.state_dim)
    grounding.load_state_dict(ck["grounding"])                  # STRICT
    return (model.to(device).eval(), grounding.to(device).eval(),
            int(ck.get("step", -1)), model.state_dim)


@torch.no_grad()
def encode_episode(model, frames, device, batch):
    """Every frame -> [T, S] float32 CPU. Byte-identical to
    ``blindimag.encode_episode_states`` (which is pinned to ``encode_window``)."""
    out = []
    for i in range(0, int(frames.shape[0]), batch):
        x = torch.as_tensor(frames[i:i + batch]).to(device)
        x = x.float().div_(255.0) if x.dtype == torch.uint8 else x.float()
        out.append(model.encode(x).float().cpu())
    return torch.cat(out)


def row_profile(frames, n_sample=8):
    """Per-episode mean row-intensity profile [H] — the RIG PROXY for X1b.

    The two PhysicalAI front-wide rigs differ by ~215 px in principal point, so
    the horizon/hood boundary sits at a different image ROW.  A row-mean profile
    is the cheapest statistic that carries that, it is content-free (no clip id,
    no recognisable image), and it lets the rig split be MEASURED rather than
    assumed.  Averaged over a few frames to damp scene content.
    """
    T = int(frames.shape[0])
    idx = torch.linspace(0, T - 1, min(n_sample, T)).long()
    x = torch.as_tensor(frames[idx]).float()
    if x.max() > 1.5:
        x = x / 255.0
    return x.mean(dim=(0, 1, 3))                                  # [H]


@torch.no_grad()
def fidelity(model, grounding, eps, states, device, stride, shuffle_seed=0):
    """Trained-head evaluation on the cached latents, per window.

    Reproduces `grounding_losses`' two logged quantities exactly:
      mid_de[lvl]  = mean over that level's horizons of ||Δxy_pred − Δxy_true||
      fwd_ade[lvl] = ||accumulated waypoints − GT ego waypoints||, mean over k
    plus the SHUFFLE CONTROL on invdyn['op'].
    """
    per = {f"{l}_mid": [] for l in LEVEL_CFG}
    per.update({f"{l}_fwd": [] for l in LEVEL_CFG})
    per["op_mid_shuffled"] = []
    eid, t0s = [], []
    kmax = max(k for _, k in LEVEL_CFG.values())

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
            a = torch.stack([acts[t:t + WINDOW] for t in ch])
            fa = torch.stack([acts[t + WINDOW: t + WINDOW + kmax] for t in ch])
            v0 = (poses[last, 3:4] / SPEED_SCALE)                      # [B,1]
            a = torch.cat([a, v0[:, None, :].expand(-1, a.shape[1], -1)], -1)
            fa = torch.cat([fa, v0[:, None, :].expand(-1, fa.shape[1], -1)], -1)
            pose_last = poses[last]                                    # [B,4]
            fut_p = torch.stack([poses[t + WINDOW: t + WINDOW + kmax]
                                 for t in ch])                         # [B,K,4]
            fut_s = torch.stack([st[t + WINDOW: t + WINDOW + kmax]
                                 for t in ch]).to(device)              # [B,K,S]
            z_t = s[:, -1]
            trans = rollout_transitions(model.predictor, s, a, fa, kmax)

            for lvl, (hor, fwd_k) in LEVEL_CFG.items():
                de = 0.0
                for kh in hor:
                    dp = grounding.invdyn[lvl](z_t, fut_s[:, kh - 1])
                    tgt = relative_ego_pose(pose_last, fut_p[:, kh - 1])
                    de = de + (dp[..., :2] - tgt[..., :2]).norm(dim=-1)
                per[f"{lvl}_mid"].append((de / len(hor)).cpu())
                wp, _ = decode_transitions(grounding.step[lvl], trans, fwd_k)
                gt = gt_ego_waypoints(pose_last, fut_p, range(1, fwd_k + 1))
                per[f"{lvl}_fwd"].append((wp - gt).norm(dim=-1).mean(1).cpu())

            # --- the deliberately failing input: mismatched real pairs ------- #
            g = torch.Generator(device="cpu").manual_seed(shuffle_seed + ep_i)
            perm = torch.randperm(z_t.shape[0], generator=g).to(device)
            de = 0.0
            for kh in LEVEL_CFG["op"][0]:
                dp = grounding.invdyn["op"](z_t, fut_s[perm, kh - 1])
                tgt = relative_ego_pose(pose_last, fut_p[:, kh - 1])
                de = de + (dp[..., :2] - tgt[..., :2]).norm(dim=-1)
            per["op_mid_shuffled"].append((de / len(LEVEL_CFG["op"][0])).cpu())

            eid += [str(ep.episode_id)] * len(ch)
            t0s += ch
        del st
    return {k: torch.cat(v) for k, v in per.items()}, eid, t0s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--artifacts", default=os.path.join(
        os.path.dirname(__file__), "..", "artifacts"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--stride", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.artifacts, exist_ok=True)

    dev = a.device
    if dev == "cuda":
        free, tot = torch.cuda.mem_get_info()
        print(f"[gpu] free {free/2**30:.2f} / {tot/2**30:.2f} GiB", flush=True)
        if free < 2.5 * 2 ** 30:            # a sibling agent shares this GPU
            print("[gpu] < 2.5 GiB free -> falling back to CPU", flush=True)
            dev = "cpu"

    t0 = time.time()
    model, grounding, step, S = build_model(a.ckpt, dev)
    print(f"[load] step={step} state_dim={S} ({time.time()-t0:.0f}s)", flush=True)

    files = sorted(f for f in os.listdir(a.cache_dir) if f.startswith("ep_")
                   and f.endswith(".pt"))[: a.episodes]
    eps, states, profiles = [], [], []
    for i, f in enumerate(files):
        ep = load_episode(os.path.join(a.cache_dir, f), mmap=True)
        st = encode_episode(model, ep.frames, dev, a.batch)
        eps.append(ep)
        states.append(st)
        profiles.append(row_profile(ep.frames))
        if i % 10 == 0:
            print(f"[encode] {i}/{len(files)}  T={st.shape[0]} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[encode] done {len(eps)} eps ({time.time()-t0:.0f}s)", flush=True)

    per, eid, t0s = fidelity(model, grounding, eps, states, dev, a.stride)
    fid = {k: float(v.mean()) for k, v in per.items()}
    ok_fwd = fid["op_fwd"] <= FIDELITY_FWD_TOL * V1_TRAIN_OP_FWD
    ok_mid = fid["op_mid"] <= FIDELITY_MID_TOL * V1_TRAIN_OP_MID
    ok_shuf = fid["op_mid_shuffled"] > 1.5 * fid["op_mid"]

    torch.save({"states": [s.half() for s in states],
                "poses": [e.poses.float() for e in eps],
                "actions": [e.actions.float() for e in eps],
                "episode_id": [int(e.episode_id) for e in eps],
                "row_profile": torch.stack(profiles),
                "state_dim": S, "ckpt_step": step,
                "cache_dir": a.cache_dir},
               os.path.join(a.out_dir, "x1_latents.pt"))
    torch.save({k: v.state_dict() for k, v in
                [("invdyn", grounding.invdyn), ("step", grounding.step)]},
               os.path.join(a.out_dir, "x1_grounding_heads.pt"))
    torch.save({"per_window": per, "eid": eid, "t0": t0s},
               os.path.join(a.out_dir, "x1_fidelity_perwindow.pt"))

    out = {
        "experiment": "X1 stage 1 — frozen-encoder latent cache + fidelity check",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": "dev box", "device": dev, "torch": torch.__version__,
        "ckpt": a.ckpt, "ckpt_step": step, "state_dim": S,
        "corpus": {"cache_dir": a.cache_dir, "n_episodes": len(eps),
                   "PARITY": "NON-PARITY (key bb543bdf7836, not e438721ae894)",
                   "admissible_for": "within-run contrasts only"},
        "n_windows": len(eid), "window": WINDOW, "stride": a.stride,
        "trained_head_on_our_windows": {k: round(v, 4) for k, v in fid.items()},
        "v1_committed_train_band_28_30k": {"op_mid_de_m": V1_TRAIN_OP_MID,
                                           "op_fwd_ade_m": V1_TRAIN_OP_FWD},
        "prereg_validation": {
            "fidelity_imagined_decode": {
                "rule": f"op_fwd <= {FIDELITY_FWD_TOL}x {V1_TRAIN_OP_FWD}",
                "got": round(fid["op_fwd"], 4), "PASS": bool(ok_fwd)},
            "fidelity_real_pair_decode": {
                "rule": f"op_mid <= {FIDELITY_MID_TOL}x {V1_TRAIN_OP_MID}",
                "got": round(fid["op_mid"], 4), "PASS": bool(ok_mid)},
            "shuffle_control_must_fail": {
                "rule": "op_mid_shuffled > 1.5x op_mid",
                "got": round(fid["op_mid_shuffled"], 4), "PASS": bool(ok_shuf)},
            "ALL_PASS": bool(ok_fwd and ok_mid and ok_shuf)},
        "elapsed_s": round(time.time() - t0, 1),
        "latent_cache": os.path.join(a.out_dir, "x1_latents.pt"),
    }
    fp = os.path.join(a.artifacts, "x1_cache_fidelity.json")
    with open(fp, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["prereg_validation"], indent=1))
    print(json.dumps(out["trained_head_on_our_windows"], indent=1))
    print("wrote", fp)


if __name__ == "__main__":
    main()
