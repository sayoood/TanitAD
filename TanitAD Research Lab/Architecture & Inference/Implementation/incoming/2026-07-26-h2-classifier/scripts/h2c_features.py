"""H2 classifier — STEP 2 (pod2, GPU): align the label grid to the episode grid, then extract
FROZEN flagship-v1 encoder features.

Two clocks, one alignment problem
---------------------------------
The L2 label grid is ``arange(t0, t1, 100 ms)`` on the **egomotion/obstacle** clock. The episode
grid is ``linspace(t_first, t_last, n)`` on the **camera** clock, minus the 2 frames the D-015
3-frame stack consumes. They are not the same index and there is no stored offset. Alignment is
recovered per clip by maximising the Pearson correlation of the two ego-speed series (both carry
``v`` at 10 Hz) over an integer lag; the achieved correlation is published per clip and a clip is
ADMITTED only above the pre-registered floor (``--min-corr``, 0.99).

The feature
-----------
``WorldModel.encode(frames)`` — the 2048-d ``SpatialGridReadout`` state the world model itself
consumes. Encoder + readout are FROZEN (``eval()``, ``no_grad``); this stream trains a head, not a
backbone. Frames are converted exactly as the training datasets do: ``uint8 -> float / 255``
(``tanitad.data._contract.to_float_frames``).

usage (pod2):
  PYTHONPATH=/workspace/TanitAD/stack python3 h2c_features.py \
      --bundle /workspace/h2clf/bundle --out /workspace/h2clf/feats \
      --ckpt /workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

CACHES = {
    "train": "/workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894",
    "val": "/workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11",
}


def LABEL_COLS(Z, pre: str):
    """Every per-clip label column stored under this clip's prefix, in file order."""
    return [n for n in Z.files if n.startswith(pre)]


def best_lag(v_lab: np.ndarray, v_ep: np.ndarray, max_lag: int = 80):
    """Integer lag L maximising corr(v_lab[gi], v_ep[gi - L]); returns (L, corr, n_overlap).

    `v_lab` is indexed by the label grid index `gi`; `v_ep` by the episode frame index `j`.
    """
    best = (0, -2.0, 0)
    nl, ne = len(v_lab), len(v_ep)
    for L in range(-max_lag, max_lag + 1):
        g0, g1 = max(0, L), min(nl, ne + L)          # gi range with 0 <= gi-L < ne
        if g1 - g0 < 60:
            continue
        a = v_lab[g0:g1]
        b = v_ep[g0 - L:g1 - L]
        sa, sb = a.std(), b.std()
        if sa < 1e-6 or sb < 1e-6:
            continue
        c = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
        if c > best[1]:
            best = (L, c, int(g1 - g0))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--min-corr", type=float, default=0.99)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from tanitad.config import flagship4b_config
    from tanitad.eval.ckpt_compat import build_world_from_ckpt

    meta = json.load(open(os.path.join(args.bundle, "h2c_meta.json")))
    Z = np.load(os.path.join(args.bundle, "h2c_labels.npz"))
    if args.limit:
        meta = meta[:args.limit]

    t0 = time.time()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    # `build_world_from_ckpt` is the tool that OWNS this fact: v1 is a speed-input checkpoint
    # (action_dim=3), so a vanilla flagship4b_config() build fails on act_emb/inv_dyn shapes.
    # It STRICT-loads, which is the guarantee that the encoder weights are v1's, not random.
    world, speed_input, src = build_world_from_ckpt(flagship4b_config(), ck, args.ckpt)
    world = world.to(args.device).eval()
    for p in world.parameters():
        p.requires_grad_(False)
    n_enc = sum(p.numel() for p in world.encoder.parameters())
    n_ro = sum(p.numel() for p in world.readout.parameters())
    print(f"[feat] STRICT-loaded v1 trunk in {time.time()-t0:.1f}s; encoder {n_enc:,} + readout "
          f"{n_ro:,} params; state_dim={world.state_dim}; speed_input={speed_input} ({src})",
          flush=True)

    aligns, kept, dropped = [], 0, 0
    tstart = time.time()
    for m in meta:
        k = m["k"]
        gi = Z[f"c{k}_gi"].astype(np.int64)
        v_lab_rows = Z[f"c{k}_ego_v"].astype(np.float64)
        nl = int(gi.max()) + 1
        v_lab = np.full(nl, np.nan)
        v_lab[gi] = v_lab_rows
        v_lab = np.nan_to_num(v_lab, nan=float(np.nanmean(v_lab)))

        ep = torch.load(os.path.join(CACHES[m["cache"]], m["file"]),
                        map_location="cpu", weights_only=True, mmap=True)
        v_ep = ep["poses"][:, 3].to(torch.float64).numpy()
        L, c, nov = best_lag(v_lab, v_ep)
        rec = {"k": k, "chunk": m["chunk"], "side": m["side"], "lag": L, "corr": round(c, 6),
               "n_overlap": nov, "T_episode": int(ep["poses"].shape[0]), "n_rows": m["n_rows"]}
        if c < args.min_corr:
            rec["admitted"] = False
            aligns.append(rec)
            dropped += 1
            continue

        j = gi - L                                   # episode frame index for each label row
        ok = (j >= 0) & (j < v_ep.shape[0])
        rec["admitted"] = True
        rec["n_rows_aligned"] = int(ok.sum())
        aligns.append(rec)

        frames = ep["frames_u8"]                     # [T, 9, 256, 256] uint8
        feats = np.empty((frames.shape[0], world.state_dim), dtype=np.float16)
        with torch.no_grad():
            for b in range(0, frames.shape[0], args.batch):
                x = frames[b:b + args.batch].to(args.device, non_blocking=True)
                x = x.float().div_(255.0)
                s = world.encode(x)                  # [B, 2048]
                feats[b:b + args.batch] = s.to(torch.float16).cpu().numpy()
        pre = f"c{k}_"
        cols = {name[len(pre):]: Z[name][ok] for name in LABEL_COLS(Z, pre)}
        np.savez_compressed(os.path.join(args.out, f"clip_{k:04d}.npz"),
                            feats=feats, j=j[ok].astype(np.int32),
                            lag=np.int32(L), corr=np.float32(c), **cols)
        kept += 1
        if kept % 25 == 0:
            el = time.time() - tstart
            print(f"[feat] {kept} kept / {dropped} dropped / {len(meta)} total "
                  f"({el:.0f}s, {el/max(kept,1):.2f}s per clip)", flush=True)

    corrs = np.array([a["corr"] for a in aligns])
    summary = {
        "n_clips": len(meta), "kept": kept, "dropped": dropped,
        "drop_frac": round(dropped / max(len(meta), 1), 4),
        "min_corr_floor": args.min_corr,
        "corr_quantiles": {q: round(float(np.quantile(corrs, float(q))), 6)
                           for q in ("0.0", "0.01", "0.05", "0.5", "0.95", "1.0")},
        "lag_hist": {str(int(L)): int(n) for L, n in
                     zip(*np.unique([a["lag"] for a in aligns if a["admitted"]],
                                    return_counts=True))},
        "encoder": {"ckpt": args.ckpt, "encoder_params": int(n_enc),
                    "readout_params": int(n_ro), "state_dim": int(world.state_dim),
                    "frozen": True, "preprocess": "uint8 -> float/255 (to_float_frames)"},
        "wallclock_s": round(time.time() - tstart, 1),
        "per_clip": aligns,
    }
    json.dump(summary, open(os.path.join(args.out, "align_summary.json"), "w"), indent=2)
    print(json.dumps({kk: vv for kk, vv in summary.items() if kk != "per_clip"}, indent=2))
    if summary["drop_frac"] > 0.10:
        print("[feat] ⛔ MORE THAN 10 % OF CLIPS FAILED ALIGNMENT — the pre-registration declares "
              "the alignment method UNFIT and the run BLOCKED.", flush=True)


if __name__ == "__main__":
    main()
