"""Our flagship's prediction at the SAME clips and the SAME t0 Alpamayo saw.

⛔ THE ALIGNMENT IS THE WHOLE EXPERIMENT. Alpamayo was run at clip time
`t0_us = 5,100,000` on our OOD-val clips. To compare, our arm must predict from
the window whose origin lands at that same clip instant — not at "roughly there".
The episode grid is an AFFINE reparametrisation of the clip clock with spacing
~0.1007 s, NOT 0.1, so `index = t0 / 0.1` drifts ~0.13 s over 200 steps, about
1.8 m of displacement at 13.6 m/s. `lead_source.register_poses_to_time` fits the
real mapping; the residual is recorded per clip so a bad registration is visible
rather than silently absorbed.

⛔ HORIZONS DIFFER AND MUST BE CUT, NOT COMPARED RAW. Alpamayo emits 64 waypoints
to 6.4 s; we emit 20 to 2.0 s at the same 0.1 s tick. Its first 20 waypoints
align exactly with ours, so the comparison is made on THAT window and the fact is
stated with every number. A 6.4 s ADE against a 2.0 s ADE is not a comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-config", required=True)
    ap.add_argument("--ego-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from taniteval import lead_source as ls, loaders
    from taniteval.cam_overlay import ego_future_path
    from taniteval.data import load_frames
    from taniteval.flagship_overlay import K, WINDOW
    from taniteval.corpus_overlay import episode_rollouts
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from build_lead_block_helpers import ego_series, index_zips, read_member

    manifest = json.load(open(a.manifest))[:a.n]
    h = loaders.load({"arch": "flagship-worldmodel-v2", "ckpt": a.ckpt,
                      "run_config": a.run_config, "speed_input": True},
                     device=a.device)
    model, sr = h["model"].eval(), h["step_readout"]
    print(f"[flagship] step {h.get('step')} loaded", flush=True)

    ego_idx = index_zips(a.ego_dir)
    out = []
    for i, entry in enumerate(manifest):
        clip, t0_s = entry["clip_id"], entry["t0_us"] / 1e6
        ep = load_frames([f"{a.corpus}/ep_{i:05d}.pt"])[0]
        poses = ep.poses.float()
        df = read_member(ego_idx.get(clip), clip) if clip in ego_idx else None
        if df is None:
            out.append({"i": i, "clip_id": clip, "error": "no egomotion"})
            continue
        ego = ego_series(df)
        try:
            reg = ls.register_poses_to_time(poses[:, :2].numpy(), ego["t"],
                                            ego["x"], ego["y"])
        except Exception as e:
            out.append({"i": i, "clip_id": clip,
                        "error": f"registration: {type(e).__name__}"})
            continue
        t_s = np.asarray(reg["t_s"])
        # window origins exactly as rollout.collect emits them
        last_idx = ls.window_last_indices(poses.shape[0])
        j = int(np.argmin(np.abs(t_s[last_idx] - t0_s)))
        last = int(last_idx[j])
        dt_err = float(t_s[last] - t0_s)

        preds = episode_rollouts(model, sr, ep.feats, poses, ep.actions.float(),
                                 "frames", True, False, a.device)
        if last not in preds:
            out.append({"i": i, "clip_id": clip,
                        "error": f"window {last} not in rollout"})
            continue
        wp = preds[last]["wp"].numpy()                 # [20, 2] dense, 0.1..2.0 s
        gt = ego_future_path(poses, last, K).numpy()   # [20, 2]
        ade = float(np.linalg.norm(wp - gt, axis=-1).mean())
        out.append({"i": i, "clip_id": clip, "t0_s": t0_s,
                    "window_last": last, "t_at_window_s": float(t_s[last]),
                    "align_err_s": round(dt_err, 4),
                    "v0_mps": float(poses[last, 3]),
                    "ade_2s_m": round(ade, 4),
                    "pred": wp.tolist(), "gt": gt.tolist(),
                    "man": preds[last].get("man"), "route": preds[last].get("route")})
        print(f"[{i}] {clip[:8]} win={last} dt={dt_err:+.3f}s "
              f"ade2s={ade:.4f} v0={poses[last,3]:.1f}", flush=True)
    json.dump(out, open(a.out, "w"), indent=1)
    ok = [r for r in out if "ade_2s_m" in r]
    print(f"[out] {a.out}  {len(ok)}/{len(out)} scored", flush=True)
    if ok:
        print(f"[flagship] mean ADE@2s {np.mean([r['ade_2s_m'] for r in ok]):.4f} m "
              f"| max |align_err| {max(abs(r['align_err_s']) for r in ok):.3f} s",
              flush=True)


if __name__ == "__main__":
    main()
