"""Re-run the hierarchy panel so its manoeuvre-coherence κ carries a GATE SWEEP.

⛔ WHY A RE-RUN IS UNAVOIDABLE (RETRACTION_LOG R-2026-08-06-yawgate). Every banked panel
stored `traj_dir` / `gt_dir` — the classes AFTER `_dir_of` applied `DIR_YAW_RAD = 0.15`.
The continuous net yaw was discarded, so no banked panel can be re-read at another gate
without putting the model back on a GPU. The fix (banking the raw net yaw) makes every
FUTURE panel free to sweep; this run is the one-off cost of the panels already published.

MEASURED, the reason it matters: on the Alpamayo comparison the 0.15 gate is ~6.5× the
typical 2 s turn — the human's own median |net yaw| is 0.023 rad and only 17.9 % of
windows exceed it — and the ranking of two arms REVERSED between 0.15 and 0.10.

⛔ NOT THE REGISTRY PATH. `runner.hier` resolves a model key through `registry.MODELS` and
reads `VAL` from `/root/valdata/...`, neither of which is what this pod holds. Driving
`hierarchy.run` directly with an explicit loader dict is the same pattern `flagship_at_t0.py`
used successfully, and it makes the checkpoint and corpus arguments VISIBLE in the command
rather than resolved by a table that may have drifted.

⚠️ Run on the IDLE pod. Never eval on a pod that is training.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run-config", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arch", default="flagship-worldmodel-v2")
    ap.add_argument("--label", required=True)
    a = ap.parse_args()

    import torch

    from taniteval import hierarchy, loaders
    from taniteval.data import load_frames

    files = sorted(glob.glob(os.path.join(a.corpus, "ep_*.pt")))[:a.episodes]
    if not files:
        raise SystemExit(f"no ep_*.pt under {a.corpus}")
    print(f"[hier] {len(files)} episodes from {a.corpus}", flush=True)

    L = loaders.load({"arch": a.arch, "ckpt": a.ckpt, "run_config": a.run_config,
                      "speed_input": True}, device=a.device)
    print(f"[hier] {a.label} step={L.get('step')} loaded", flush=True)

    eps = load_frames(files)
    t0 = time.time()
    res = hierarchy.run(L["model"], L["step_readout"], eps, a.device,
                        speed_input=True, max_eps=a.episodes, stride=a.stride)
    res["_rerun"] = {
        "label": a.label, "ckpt": a.ckpt, "corpus": a.corpus,
        "ckpt_step": L.get("step"), "n_episodes": len(files), "stride": a.stride,
        "wall_s": round(time.time() - t0, 1),
        "_why": ("re-run to obtain consistency.gate_sensitivity, which the banked "
                 "panels cannot produce — they stored only the thresholded direction "
                 "classes. See RETRACTION_LOG R-2026-08-06-yawgate."),
        "_evidence_class": "MEASURED (ours)",
        "torch": torch.__version__,
    }
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=str)

    gs = (res.get("consistency") or {}).get("gate_sensitivity") or {}
    if gs.get("status") == "UNAVAILABLE":
        # ⛔ FAIL LOUD. If this appears, the pod is running a STALE hierarchy.py and the
        # whole point of the re-run was missed — do not let it look like a success.
        raise SystemExit("gate_sensitivity UNAVAILABLE — the pod's taniteval is STALE. "
                         f"reason: {gs.get('reason')}")
    tm = gs.get("gt_turn_magnitude", {})
    print(f"\n[{a.label}] n_windows={res.get('n_windows')} "
          f"step={L.get('step')} in {res['_rerun']['wall_s']}s")
    print(f"  human net yaw over 2 s: median {tm.get('median_abs_net_yaw_rad')} · "
          f"p90 {tm.get('p90_abs_net_yaw_rad')} rad · "
          f"{tm.get('frac_above_published_gate')} above the {gs.get('published_gate_rad')} gate")
    print(f"  {'gate':>6} {'man~traj κ':>12} {'traj~gt κ':>12} {'frac turning':>13}")
    for g, v in gs.get("per_gate", {}).items():
        print(f"  {g:>6} {str(v['maneuver_vs_trajectory_kappa']):>12} "
              f"{str(v['trajectory_vs_gt_kappa']):>12} {v['frac_gt_turning']:>13}")
    print(f"  kappa_range {gs.get('kappa_range')} · VERDICT_STABLE "
          f"{gs.get('verdict_stable')}")
    print(f"\n[out] {a.out}", flush=True)


if __name__ == "__main__":
    main()
