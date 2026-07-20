"""flagship v1.5 — one-time preparation off the FROZEN v1 trunk.

Three artifacts, all deterministic:

  ``states``   per-frame frozen encoder states for every cached episode
               (``WorldModel.encode`` = ViT encoder + spatial readout), fp16.
               This is the REF-A feature-cache pattern: pay the encoder once,
               then head-only training is hours instead of days. 2,376 episodes
               x 199 frames x 2048 dims x fp16 = ~1.9 GB.

  ``anchors``  the 256-trajectory FPS vocabulary, built over the REAL ego-frame
               waypoint targets of every training window (the identical
               procedure and code path as ``build_refc_anchors.py``, but reading
               the extracted pose cache instead of re-reading 278 GB of frames).

  ``probes``   the imagination probe vocabulary: FPS over REAL future
               ``(steer, accel)`` sequences. These are the action sequences the
               frozen predictor is rolled forward under to produce conditioning
               (b).

Usage (pod2):
  PYTHONPATH=/workspace/TanitAD/stack python3 v15_prep.py states \
      --ckpt /workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt \
      --cache <epcache-split-dir> --out /workspace/v15/states_train.pt
  PYTHONPATH=... python3 v15_prep.py anchors --poses /workspace/v15/poses_train.pt \
      --out /workspace/v15/anchors256.pt --n 256
  PYTHONPATH=... python3 v15_prep.py probes  --poses /workspace/v15/poses_train.pt \
      --out /workspace/v15/probes8.pt --n 8
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from tanitad.config import flagship4b_config
from tanitad.refs.refc import furthest_point_sample

WINDOW = 8
HORIZONS = (5, 10, 15, 20)
K_MAX = max(HORIZONS)


# --------------------------------------------------------------------------- #
# the frozen trunk                                                            #
# --------------------------------------------------------------------------- #
def load_frozen_v1(ckpt: str, device: str = "cuda"):
    """Rebuild the DEPLOYED v1 and load it STRICT. Fails loud if the checkpoint
    is not the speed arm — ``flagship4b-phase0-30k`` (the no-speed ablation
    CONTROL, 2.918 m) has an almost identical name and would silently train the
    head on the wrong trunk."""
    from tanitad.models.fourbrain import WorldModel
    from tanitad.models.metric_dynamics import HierarchicalGrounding

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = ck["model"]
    a_dim = sd["predictor.act_emb.0.weight"].shape[1]
    if a_dim != 3:
        raise SystemExit(
            f"REFUSING: {ckpt} has predictor action_dim={a_dim}, not 3. "
            "flagship v1.5 must sit on the speed arm "
            "(flagship4b-speedjerk-30k, speed_input=true), NOT on "
            "flagship4b-phase0-30k (the no-speed ablation control).")
    cfg = flagship4b_config()
    object.__setattr__(cfg.predictor, "action_dim", 3)
    if cfg.tactical_pred is not None:
        object.__setattr__(cfg.tactical_pred, "action_dim", 3)
    object.__setattr__(cfg.encoder, "grad_checkpoint", False)   # eval-only
    model = WorldModel(cfg)
    model.load_state_dict(sd)                                   # STRICT
    grounding = HierarchicalGrounding(model.state_dim)
    grounding.load_state_dict(ck["grounding"])                  # STRICT
    model = model.to(device).eval()
    grounding = grounding.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in grounding.parameters():
        p.requires_grad_(False)
    step = int(ck.get("step", -1))
    print(f"[trunk] loaded {ckpt} step={step} action_dim=3 state_dim="
          f"{model.state_dim} (FROZEN)", flush=True)
    return model, grounding, step


# --------------------------------------------------------------------------- #
# states                                                                      #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def build_states(args):
    model, _, step = load_frozen_v1(args.ckpt, args.device)
    files = sorted(f for f in os.listdir(args.cache) if f.endswith(".pt"))
    if args.episodes:
        files = files[:args.episodes]
    out_states, eids, t0 = [], [], time.time()
    for i, f in enumerate(files):
        d = torch.load(os.path.join(args.cache, f), map_location="cpu",
                       weights_only=True, mmap=True)
        fr = d["frames_u8"]
        t_len = fr.shape[0]
        zs = []
        for b0 in range(0, t_len, args.batch):
            x = fr[b0:b0 + args.batch].to(args.device).float().div_(255.0)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=(args.device == "cuda")):
                z = model.encode(x)
            zs.append(z.float().half().cpu())
        out_states.append(torch.cat(zs))                 # [T, S] fp16
        eids.append(f)
        del d, fr
        if i % 100 == 0:
            el = time.time() - t0
            print(f"  {i}/{len(files)}  {el:.0f}s  "
                  f"eta {el / max(i, 1) * (len(files) - i):.0f}s", flush=True)
    torch.save({"eids": eids, "states": out_states, "trunk_ckpt": args.ckpt,
                "trunk_step": step, "state_dim": out_states[0].shape[-1]},
               args.out)
    print(f"[states] wrote {args.out}: {len(eids)} eps, "
          f"{sum(s.shape[0] for s in out_states)} frames, "
          f"{time.time() - t0:.0f}s", flush=True)


# --------------------------------------------------------------------------- #
# anchors / probes (both FPS, both off the pose cache)                        #
# --------------------------------------------------------------------------- #
def _ego(dxy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """World delta -> ego frame of ``yaw`` (refb_labels.ego_frame convention)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * dxy[:, 0] + s * dxy[:, 1],
                        -s * dxy[:, 0] + c * dxy[:, 1]], dim=-1)


def waypoint_pool(poses_list, horizons=HORIZONS) -> torch.Tensor:
    """Ego-frame waypoint targets for EVERY window -> [M, len(horizons), 2].

    Identical to ``build_refc_anchors.episode_traj_pool`` /
    ``refb_labels.waypoint_targets``, vectorised per episode off the pose cache.
    """
    pool = []
    for p in poses_list:
        po = torch.as_tensor(p, dtype=torch.float32)
        n = po.shape[0] - K_MAX
        if n <= 0:
            continue
        idx = torch.arange(n)
        yaw = po[idx, 2]
        wps = [_ego(po[idx + k, :2] - po[idx, :2], yaw) for k in horizons]
        pool.append(torch.stack(wps, dim=1))
    return torch.cat(pool, dim=0)


def action_pool(actions_list, k: int = K_MAX) -> torch.Tensor:
    """Every real future action sequence -> [M, k, 2]."""
    pool = []
    for a in actions_list:
        ac = torch.as_tensor(a, dtype=torch.float32)
        n = ac.shape[0] - k
        if n <= 0:
            continue
        idx = torch.arange(n)
        pool.append(torch.stack([ac[idx + j] for j in range(k)], dim=1))
    return torch.cat(pool, dim=0)


def _fps_indices(pool: torch.Tensor, n: int, seed: int = 0) -> torch.Tensor:
    """``refc.furthest_point_sample`` returning INDICES, so the caller can FPS in
    a whitened space and keep the original-space rows. Same greedy algorithm,
    same seeding — pinned against the parent by tests."""
    m = pool.shape[0]
    if n > m:
        raise ValueError(f"cannot FPS {n} from a pool of {m}")
    flat = pool.reshape(m, -1)
    g = torch.Generator(device=flat.device).manual_seed(seed)
    first = int(torch.randint(m, (1,), generator=g, device=flat.device))
    chosen = [first]
    dist = ((flat - flat[first]) ** 2).sum(dim=-1)
    for _ in range(n - 1):
        nxt = int(torch.argmax(dist))
        chosen.append(nxt)
        dist = torch.minimum(dist, ((flat - flat[nxt]) ** 2).sum(dim=-1))
    return torch.tensor(chosen)


def build_fps(args, kind: str):
    """FPS a vocabulary out of the real data.

    ``anchors`` live in (x, y) METRES — one physical unit, so plain flattened-L2
    FPS is correct and is exactly what ``build_refc_anchors.py`` does.

    ``probes`` live in (steer, accel), which are NOT commensurable: measured on
    this corpus ``std(accel)/std(steer) = 9.7``. Plain L2 FPS therefore spends
    every pick on the accel axis and returns eight essentially STRAIGHT probes
    (measured: |steer| <= 0.038 across all 8, against a channel std of 0.080) —
    an imagination vocabulary that can never ask "what if I turn". Probes are
    thus percentile-clipped (the corpus has |accel| outliers out to 11.7 m/s^2,
    ~15 sigma, which FPS otherwise grabs first) and per-channel WHITENED before
    the FPS, with the ORIGINAL un-whitened sequences returned.
    """
    d = torch.load(args.poses, weights_only=False)
    if kind == "anchors":
        pool = waypoint_pool(d["poses"])
    else:
        pool = action_pool(d["actions"])
    g = torch.Generator().manual_seed(args.seed)
    if pool.shape[0] > args.max_pool:
        pool = pool[torch.randperm(pool.shape[0], generator=g)[:args.max_pool]]
    meta_extra = {}
    if kind == "probes" and not args.no_whiten:
        flat = pool.reshape(-1, pool.shape[-1])
        lo = torch.quantile(flat, args.clip_q, dim=0)
        hi = torch.quantile(flat, 1.0 - args.clip_q, dim=0)
        keep = ((pool >= lo) & (pool <= hi)).all(dim=-1).all(dim=-1)
        if int(keep.sum()) >= args.n:
            pool = pool[keep]
        std = pool.reshape(-1, pool.shape[-1]).std(dim=0).clamp_min(1e-6)
        sel_i = _fps_indices(pool / std, args.n, args.seed)
        sel = pool[sel_i].contiguous()
        meta_extra = {"whitened": True, "channel_std": [round(float(x), 5)
                                                        for x in std],
                      "clip_q": args.clip_q,
                      "pool_after_clip": int(pool.shape[0])}
    else:
        sel = furthest_point_sample(pool, args.n, seed=args.seed).contiguous()
    meta = {"method": "fps", "kind": kind, "n": args.n,
            "pool_size": int(pool.shape[0]), "source": args.poses,
            "seed": args.seed, "shape": list(sel.shape), **meta_extra}
    torch.save({("anchors" if kind == "anchors" else "probes"): sel, **meta},
               args.out)
    print(json.dumps({"saved": args.out, **meta,
                      "extent_x": [round(float(sel[..., 0].min()), 2),
                                   round(float(sel[..., 0].max()), 2)],
                      "extent_y": [round(float(sel[..., 1].min()), 2),
                                   round(float(sel[..., 1].max()), 2)]}),
          flush=True)


# --------------------------------------------------------------------------- #
# labels — VTARGET (raw + fixed) and ROUTE (legacy + v2.1), as ONE artifact    #
# --------------------------------------------------------------------------- #
def build_labels(args):
    """Mint every per-window goal label once, both label generations, so the
    "do the repaired labels help?" ablation is a switch and not a rebuild.

    Per episode, for every window last-index L = t + WINDOW - 1:
      vt_band_v2   fixed mint  (smoothed track, enforced lookahead floor,
                                DROPPED where untrustworthy)
      vt_band_raw  the pre-fix mint (planner_p2 verbatim, silent hold-speed
                                fallback) — the label ablation's control
      route_v21    v2.1 class {0 left, 1 straight, 2 right, 3 UNKNOWN}
      route_graded tanh(mean_curv / junction units), threshold-free
      route_legacy v1 class from nav_command -> route_target, i.e. the broken
                                labeler that silently emits STRAIGHT
    """
    import refb_labels
    from tanitad.lake.vocab import VTARGET_TOKENS, vtarget_band
    from tanitad.lake.vtarget import vtarget_raw, vtarget_v2

    toks = list(VTARGET_TOKENS)
    memo: dict[float, int] = {}

    def band_ix(x: float) -> int:
        k = round(float(x), 3)
        if k not in memo:
            memo[k] = toks.index(vtarget_band(k))
        return memo[k]

    d = torch.load(args.poses, weights_only=False)
    out: dict[str, list] = {k: [] for k in
                            ("vt_band_v2", "vt_band_raw", "vt_valid",
                             "vt_v2", "vt_raw", "lookahead", "route_v21",
                             "route_valid", "route_graded", "route_legacy")}
    stats = {"n_windows": 0, "vt_valid": 0, "route_valid": 0,
             "route_v21_counts": [0, 0, 0, 0], "route_legacy_counts": [0, 0, 0],
             "route_reason": {}}
    t0 = time.time()
    for e, p in enumerate(d["poses"]):
        po = torch.as_tensor(p, dtype=torch.float32)
        t_len = po.shape[0]
        n = t_len - WINDOW - K_MAX
        if n <= 0:
            for k in out:
                out[k].append(torch.zeros(0))
            continue
        last = np.arange(WINDOW - 1, WINDOW - 1 + n, dtype=np.int64)
        v = po[:, 3].numpy().astype(np.float64)
        vt2, ok2, look, _ = vtarget_v2(v, last, min_lookahead=args.min_lookahead)
        vtr, _okr = vtarget_raw(v, last)
        b2 = np.where(ok2, [band_ix(x) for x in vt2], len(toks))   # DROPPED=23
        br = np.array([band_ix(x) for x in vtr])                   # always a band
        r21, rok, rgr, rleg, reasons = [], [], [], [], {}
        for l in last.tolist():
            rr = refb_labels.route_from_future_v21(po, int(l))
            r21.append(int(rr["route"])); rok.append(bool(rr["valid"]))
            rgr.append(float(rr["graded_route"]))
            reasons[rr["reason"]] = reasons.get(rr["reason"], 0) + 1
            cmd, _ = refb_labels.nav_command(po, int(l))
            rleg.append(int(refb_labels.route_target(cmd)))
        out["vt_band_v2"].append(torch.as_tensor(b2, dtype=torch.long))
        out["vt_band_raw"].append(torch.as_tensor(br, dtype=torch.long))
        out["vt_valid"].append(torch.as_tensor(ok2))
        out["vt_v2"].append(torch.as_tensor(vt2, dtype=torch.float32))
        out["vt_raw"].append(torch.as_tensor(vtr, dtype=torch.float32))
        out["lookahead"].append(torch.as_tensor(look, dtype=torch.int16))
        out["route_v21"].append(torch.as_tensor(r21, dtype=torch.long))
        out["route_valid"].append(torch.as_tensor(rok))
        out["route_graded"].append(torch.as_tensor(rgr, dtype=torch.float32))
        out["route_legacy"].append(torch.as_tensor(rleg, dtype=torch.long))
        stats["n_windows"] += n
        stats["vt_valid"] += int(ok2.sum())
        stats["route_valid"] += int(sum(rok))
        for c in r21:
            stats["route_v21_counts"][c] += 1
        for c in rleg:
            stats["route_legacy_counts"][c] += 1
        for k, c in reasons.items():
            stats["route_reason"][k] = stats["route_reason"].get(k, 0) + c
        if e % 200 == 0:
            print(f"  {e}/{len(d['poses'])}  {time.time() - t0:.0f}s", flush=True)
    nw = max(stats["n_windows"], 1)
    stats["vt_valid_frac"] = round(stats["vt_valid"] / nw, 4)
    stats["route_valid_frac"] = round(stats["route_valid"] / nw, 4)
    stats["route_v21_frac"] = [round(c / nw, 4) for c in stats["route_v21_counts"]]
    stats["route_legacy_frac"] = [round(c / nw, 4)
                                  for c in stats["route_legacy_counts"]]
    stats["route_turn_frac_v21"] = round(
        (stats["route_v21_counts"][0] + stats["route_v21_counts"][2]) / nw, 4)
    stats["route_turn_frac_legacy"] = round(
        (stats["route_legacy_counts"][0] + stats["route_legacy_counts"][2]) / nw, 4)
    torch.save({"eids": d["eids"], **out, "stats": stats,
                "min_lookahead": args.min_lookahead}, args.out)
    print(json.dumps({"saved": args.out, **stats}, indent=2), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("states")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--episodes", type=int, default=0)
    s.add_argument("--device", default="cuda")

    for kind in ("anchors", "probes"):
        p = sub.add_parser(kind)
        p.add_argument("--poses", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--n", type=int, default=256 if kind == "anchors" else 8)
        p.add_argument("--max-pool", type=int, default=200_000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--no-whiten", action="store_true",
                       help="probes: skip the per-channel whitening (steer and "
                            "accel differ ~10x in scale, so plain L2 FPS "
                            "returns only straight probes)")
        p.add_argument("--clip-q", type=float, default=0.005,
                       help="probes: per-channel quantile clip before FPS")

    lb = sub.add_parser("labels")
    lb.add_argument("--poses", required=True)
    lb.add_argument("--out", required=True)
    lb.add_argument("--min-lookahead", type=int, default=50)

    a = ap.parse_args(argv)
    if a.cmd == "states":
        build_states(a)
    elif a.cmd == "labels":
        build_labels(a)
    else:
        build_fps(a, a.cmd)


if __name__ == "__main__":
    main()
