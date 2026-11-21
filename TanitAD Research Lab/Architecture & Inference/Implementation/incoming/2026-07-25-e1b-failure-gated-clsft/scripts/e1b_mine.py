"""E1b step 1 — MINE recoverable pre-failure states from PARITY-TRAIN episodes.

Rolls REF-C base closed-loop (E1a's real-footage low-OOD loop body, VERBATIM via
`import e1a_horizon`) to horizon K on parity-TRAIN episodes and collects the states
in the window JUST BEFORE corridor-departure first crosses — the *recoverable*
pre-failure states (still in-corridor, |XTE| below the corridor half-width, but
heading out toward the first crossing). Already-departed states (|XTE| past the
half-width) are EXCLUDED as unrecoverable, exactly as the brief specifies.

For each mined state we store the EXACT model input the rollout saw (the real
footage window slice + the (dlat, dpsi) warp), the current ego speed v0, and the
R2LPL RECOVERY TARGET: the logged corridor path ahead of the nearest reference,
expressed in the CURRENT (offset) ego frame — i.e. the demonstration that returns
the ego to and then follows the corridor. The nearest anchor to that target is the
"return anchor" the CL-SFT score head is taught to rank up.

NO LEAK: mining source is physicalai-train-e438721ae894 ONLY (proved episode-id
disjoint from the held-out eval set, intersection 0 — probe_substrate.json).
Renderer-free, kinematic-bicycle closed loop (the imagination instrument, not
AlpaSim). Read-only on every checkpoint and cache.

Usage:
  PYTHONPATH=/workspace/TanitAD/stack /workspace/venv/bin/python e1b_mine.py \
    --train-dir /workspace/pai_epcache/physicalai-train-e438721ae894 \
    --refc-ckpt /workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt \
    --K 185 --episodes 600 --out /workspace/e1b/mined_buffer.pt
"""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch

for _p in ("/workspace/TanitAD/stack", "/workspace/TanitAD/stack/scripts",
           "/workspace/e1a_e2a"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse E1a's trusted loop primitives VERBATIM (do not reinvent).
import e1a_horizon as e1a
from e1a_horizon import (W, DT, WHEELBASE, WP_STEPS, sampling_homography,
                         warp_batch, wp_to_control, _wrap, load_refc)
from tanitad.data.mixing import load_episode


@torch.no_grad()
def mine_rollout(model, ep_paths, device, K, primary, warn_min, h_pre,
                 stride, batch, target_states):
    """Closed-loop rollout with per-step tracing + failure-gating -> mined records.

    Loop body identical to e1a_horizon.rollout; adds per-step traces of
    (mstar, dlat, dpsi, ego pose, v0) so a mined state can be reconstructed, and
    the pre-failure gate. Returns (records, stats)."""
    records = []
    n_dep = n_win = 0
    for ep_path in ep_paths:
        if target_states and len(records) >= target_states:
            break
        ep = load_episode(str(ep_path), mmap=True)
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 \
            else ep.frames.float()
        poses = ep.poses.float()
        T = fr.shape[0]
        starts = list(range(0, T - W - K, stride))
        if not starts:
            continue
        for bi in range(0, len(starts), batch):
            ch = starts[bi:bi + batch]
            b = len(ch)
            n_win += b
            t0 = torch.tensor(ch)
            last = t0 + W - 1
            idx = last[:, None] + torch.arange(0, K + 1)[None]
            Pxy = poses[idx][..., :2]
            Pyaw = poses[idx][..., 2]
            ex = poses[last, 0].clone(); ey = poses[last, 1].clone()
            eyaw = poses[last, 2].clone(); ev = poses[last, 3].clone()
            ar = torch.arange(b)
            lat_t = torch.zeros(b, K)
            tr_ms = torch.zeros(b, K, dtype=torch.long)
            tr_dlat = torch.zeros(b, K); tr_dpsi = torch.zeros(b, K)
            tr_ex = torch.zeros(b, K); tr_ey = torch.zeros(b, K)
            tr_eyaw = torch.zeros(b, K); tr_ev = torch.zeros(b, K)
            for k in range(K):
                d = (Pxy - torch.stack([ex, ey], -1)[:, None]).norm(dim=-1)
                mstar = d.argmin(dim=1)
                pref = Pxy[ar, mstar]; yref = Pyaw[ar, mstar]
                dx = ex - pref[:, 0]; dy = ey - pref[:, 1]
                dlat = -torch.sin(yref) * dx + torch.cos(yref) * dy
                dpsi = _wrap(eyaw - yref)
                lat_t[:, k] = dlat.abs()
                tr_ms[:, k] = mstar; tr_dlat[:, k] = dlat; tr_dpsi[:, k] = dpsi
                tr_ex[:, k] = ex; tr_ey[:, k] = ey
                tr_eyaw[:, k] = eyaw; tr_ev[:, k] = ev
                wins = [fr[int(t0[i] + mstar[i]):int(t0[i] + mstar[i]) + W]
                        for i in range(b)]
                fw = torch.stack(wins).to(device)
                Hs = torch.stack([
                    sampling_homography(float(dlat[i]),
                                        float(math.degrees(dpsi[i])), 1.5, 0.0)
                    for i in range(b)])
                fw = warp_batch(fw, Hs)
                w_look = model(fw, nav_cmd=None, v0=ev.to(device),
                               steps=2)["traj"][:, 0].cpu()
                steer, accel = wp_to_control(w_look, ev)
                ex = ex + ev * torch.cos(eyaw) * DT
                ey = ey + ev * torch.sin(eyaw) * DT
                eyaw = eyaw + ev / WHEELBASE * torch.tan(steer) * DT
                ev = (ev + accel * DT).clamp_min(0.0)
            # ---- failure-gate each window ----
            for i in range(b):
                lat = lat_t[i]
                crossed = torch.nonzero(lat > primary, as_tuple=False)
                if len(crossed) == 0:
                    continue                       # never departs (success)
                kc = int(crossed[0])
                if kc == 0:
                    continue                       # departed immediately
                n_dep += 1
                lo = max(0, kc - h_pre)
                for k in range(lo, kc):
                    lk = float(lat[k])
                    if lk > primary or lk < warn_min:
                        continue                   # in-corridor AND heading out
                    ref_idx = int(t0[i]) + W - 1 + int(tr_ms[i, k])
                    if ref_idx + max(WP_STEPS) >= T:
                        continue                   # need full reference future
                    exi = float(tr_ex[i, k]); eyi = float(tr_ey[i, k])
                    eyawi = float(tr_eyaw[i, k])
                    c, s = math.cos(eyawi), math.sin(eyawi)
                    tgt = []
                    for h in WP_STEPS:
                        px = float(poses[ref_idx + h, 0])
                        py = float(poses[ref_idx + h, 1])
                        ddx = px - exi; ddy = py - eyi
                        tgt.append([c * ddx + s * ddy, -s * ddx + c * ddy])
                    records.append({
                        "ep_path": str(ep_path),
                        "episode_id": str(ep.episode_id),
                        "slice_start": int(t0[i]) + int(tr_ms[i, k]),
                        "dlat": float(tr_dlat[i, k]),
                        "dpsi": float(tr_dpsi[i, k]),
                        "v0": float(tr_ev[i, k]),
                        "recovery_target": tgt,
                        "k": int(k), "k_cross": kc, "lat_k": lk,
                    })
    return records, {"n_windows": n_win, "n_departing_windows": n_dep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir",
                    default="/workspace/pai_epcache/physicalai-train-e438721ae894")
    ap.add_argument("--refc-ckpt",
                    default="/workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt")
    ap.add_argument("--refc-preset", default="base")
    ap.add_argument("--K", type=int, default=185, help="mining rollout horizon")
    ap.add_argument("--episodes", type=int, default=600,
                    help="parity-train episodes to roll (cap; 1 window each at K=185)")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--corridor-halfwidth", type=float, default=1.75,
                    help="corridor primary threshold (E1a standing definition)")
    ap.add_argument("--warn-min", type=float, default=0.30,
                    help="min |XTE| for a state to count as 'heading out'")
    ap.add_argument("--h-pre", type=int, default=10,
                    help="pre-failure lead-up window (steps before first crossing)")
    ap.add_argument("--target-states", type=int, default=0,
                    help="stop once this many states are mined (0 = use --episodes)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/workspace/e1b/mined_buffer.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ep_files = sorted(Path(args.train_dir).glob("ep_*.pt"))
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(ep_files))[:args.episodes]
    ep_paths = [ep_files[i] for i in sorted(order.tolist())]
    print(f"[mine] {len(ep_paths)}/{len(ep_files)} parity-train episodes "
          f"({args.train_dir}) | K={args.K} | dev {device}", flush=True)

    model, step, cfg = load_refc(args.refc_ckpt, args.refc_preset, device)
    print(f"[mine] REF-C {args.refc_preset} step {step} anchors "
          f"{tuple(model.decoder.anchors.shape)}", flush=True)

    t0 = time.time()
    records, stats = mine_rollout(
        model, ep_paths, device, args.K, args.corridor_halfwidth,
        args.warn_min, args.h_pre, args.stride, args.batch, args.target_states)
    dt = time.time() - t0

    # provenance + strata
    eids = sorted({r["episode_id"] for r in records})
    lat_k = np.array([r["lat_k"] for r in records]) if records else np.array([])
    v0s = np.array([r["v0"] for r in records]) if records else np.array([])
    meta = {
        "_experiment": "E1b failure-mining (recoverable pre-failure states)",
        "mining_source": args.train_dir,
        "refc_ckpt": args.refc_ckpt, "refc_step": step,
        "K": args.K, "corridor_halfwidth_m": args.corridor_halfwidth,
        "warn_min_m": args.warn_min, "h_pre_steps": args.h_pre,
        "n_episodes_rolled": len(ep_paths),
        "n_windows": stats["n_windows"],
        "n_departing_windows": stats["n_departing_windows"],
        "n_mined_states": len(records),
        "n_distinct_episodes_mined": len(eids),
        "lat_k_m": {"min": float(lat_k.min()) if lat_k.size else None,
                    "mean": float(lat_k.mean()) if lat_k.size else None,
                    "max": float(lat_k.max()) if lat_k.size else None},
        "v0_mps": {"min": float(v0s.min()) if v0s.size else None,
                   "mean": float(v0s.mean()) if v0s.size else None,
                   "max": float(v0s.max()) if v0s.size else None},
        "mine_wall_s": round(dt, 1),
        "leak_guard": ("mined ONLY from physicalai-train-e438721ae894; "
                       "episode-id disjoint from held-out eval (intersection 0, "
                       "probe_substrate.json)"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": records, "meta": meta}, args.out)
    Path(args.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[mine] {len(records)} states from {len(eids)} episodes | "
          f"{stats['n_departing_windows']}/{stats['n_windows']} windows departed | "
          f"|XTE| mean {meta['lat_k_m']['mean']} | {dt:.0f}s -> {args.out}",
          flush=True)
    print("E1B_MINE_DONE", flush=True)


if __name__ == "__main__":
    main()
