#!/usr/bin/env python3
"""refc_perwindow.py — the REF-C comparison arm for the v4 co-primary, on the
SAME windows.

The gate card's reference numbers (REF-C base, K=185: overall 0.5877, junction
0.8414, peak XTE 38.94 m) were MEASURED on ``physicalai-val-heldout-79d4e3d2d4c6``
(44 episodes, 43 windows) — NOT on the registered 40-episode val cache the v4
gate is scored on. A cross-val-set comparison is a scale reference, not a paired
test. This file re-runs REF-C through E1a's OWN loop on the registered val cache
so the two arms share windows and the delta can be PAIRED.

``e1a_horizon.rollout`` / ``load_refc`` are IMPORTED, not copied, so the REF-C
arm is E1a's loop body verbatim; only the per-window tensors are additionally
persisted (so ``pair_arms.py`` can compute the paired episode-cluster bootstrap)
and the emitter is ``taniteval.corridor.stratified`` (the registered one), the
same emitter the v4 arm uses.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

for _p in ("/workspace/_v4gate", "/root/TanitAD/stack",
           "/root/TanitAD/stack/scripts", "/root/taniteval"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import e1a_horizon as E                                        # noqa: E402
from tanitad.data.mixing import load_episode                   # noqa: E402
from tanitad.instruments.numerics import strict_numerics       # noqa: E402
from taniteval import ci as _ci                                # noqa: E402
from taniteval import corridor as _corr                        # noqa: E402
from v4_corridor_cl import DT, W, emit                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refc-ckpt", default="/root/models/refc-base-30k/ckpt.pt")
    ap.add_argument("--refc-preset", default="base")
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--p1-json", default="/root/lanekeep/lowood_flagship_ci.json")
    ap.add_argument("--horizons", default="185,20")
    ap.add_argument("--episodes", type=int, default=999)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--corridor-grid", default="1.0,1.75,2.5")
    ap.add_argument("--corridor-halfwidth", type=float, default=1.75)
    ap.add_argument("--junction-deg", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Ks = [int(x) for x in a.horizons.split(",")]
    thresholds = [float(x) for x in a.corridor_grid.split(",")]
    ood = E.OODMap(a.p1_json)

    files = sorted(Path(a.val_dir).glob("ep_*.pt"))[:a.episodes]
    episodes = [load_episode(str(p), mmap=True) for p in files]
    Ts = [int(e.poses.shape[0]) for e in episodes]
    model, step, cfg = E.load_refc(a.refc_ckpt, a.refc_preset, device)
    print(f"[refc] {len(episodes)} eps | T [{min(Ts)},{max(Ts)}] | step {step} | "
          f"anchors {tuple(model.decoder.anchors.shape)} | horizons "
          f"{cfg.trajectory.horizons}", flush=True)

    res = {
        "_experiment": "REF-C comparison arm for the v4 co-primary, SAME windows",
        "_evidence_class": "MEASURED (ours; artifact = this JSON)",
        "_loop": "e1a_horizon.rollout / load_refc IMPORTED verbatim",
        "_emitter": "taniteval.corridor.stratified",
        "_estimator": "episode_cluster_bootstrap (taniteval/ci.py) B=2000",
        "refc_ckpt": a.refc_ckpt, "refc_step": step, "refc_preset": a.refc_preset,
        "n_anchors": int(model.decoder.anchors.shape[0]), "denoise_steps": 2,
        "val_dir": a.val_dir, "n_episodes": len(episodes),
        "episode_T_min": min(Ts), "episode_T_max": max(Ts),
        "stride": a.stride, "horizons_K": Ks,
        "corridor_thresholds_m": thresholds,
        "corridor_primary_m": a.corridor_halfwidth,
        "junction_deg": a.junction_deg,
        "_provenance_caveat": (
            "This is the REF-C base checkpoint AVAILABLE ON THE EVAL POD "
            f"({a.refc_ckpt}, step {step}). The gate card's reference numbers "
            "came from /workspace/experiments/refc-diffusion-base-v21-30k/ckpt.pt "
            "on a DIFFERENT pod and a DIFFERENT val set; whether the two are the "
            "same run is UNVERIFIED here. Treat this as an independently "
            "measured REF-C base point on the registered val cache, not as a "
            "reproduction of the card's number."),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))

    with strict_numerics():
        for K in Ks:
            t = time.time()
            pw = E.rollout(model, episodes, device, K, a.stride, a.batch)
            pw["_rollout_steps_executed"] = -1     # e1a's rollout does not count
            blk = emit(pw, ood, K, thresholds, a.corridor_halfwidth,
                       a.junction_deg)
            res.setdefault("all_windows", {})[str(K)] = blk
            o = blk["overall"]
            print(f"[refc] K={K:4d} n_win={o['n_windows']} n_ep={o['n_episodes']} "
                  f"CDR={o['corridor_departure_rate']['mean']:.4f} "
                  f"[{o['corridor_departure_rate']['lo']:.4f},"
                  f"{o['corridor_departure_rate']['hi']:.4f}] "
                  f"peakXTE={o['peak_xte_m']['mean']:.3f} "
                  f"({time.time() - t:.0f}s)", flush=True)
            Path(a.out).write_text(json.dumps(res, indent=2, default=str))
            torch.save({k: pw[k] for k in ("lat", "yaw", "ade2s", "hd2s", "hdK",
                                           "speed", "eid", "t0", "epi",
                                           "de_fixed", "fixed_steps")},
                       str(Path(a.out).with_suffix("")) + f"_perwindow_K{K}.pt")
    print("REFC_DONE", flush=True)


if __name__ == "__main__":
    main()
