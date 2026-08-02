"""E-CTRV — re-adjudicate the driving block's floor verdicts against CTRV.

PRE-REGISTERED (written before the driver was run; see PREREGISTRATION.md).

Runs on the eval pod:

    cd /workspace && PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/taniteval \\
      python3 run_ctrv_readjudication.py \\
        --val-dir /workspace/val40cache \\
        --results-dir /workspace/TanitAD/taniteval/results \\
        --out /workspace/ctrv_readjudication.json

CPU-only. It touches no GPU, loads no checkpoint and re-runs no model: every
arm's per-window predictions are read from the banked ``windows_<arm>.pt``
dumps, and the floors are pure kinematics from the val poses.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctrv_floor import build_floors, verify_alignment  # noqa: E402

from taniteval import ci as _ci  # noqa: E402
from taniteval import driving as D  # noqa: E402

# metrics carried through the re-adjudication (subset of driving.PAIRED that
# the turn-stratum verdicts actually rest on, plus the headline)
METRICS = ("ade_0_2s", "fde_2s", "miss_2m", "lat_abs_2s_m",
           "heading_mae_2s_deg", "heading_med_2s_deg",
           "pathgeom_crosstrack_m", "long_abs_2s_m", "speed_mae_mps")
FLOOR_KEYS = ("cv", "holdv0", "ctrv_gated", "ctrv")
N_BOOT = 2000
SEED = 0


def _reduce_name(k):
    return D.REDUCE.get(k, "mean")


def _point(v, k):
    return float(_ci.resolve_reducer(_reduce_name(k))(np.asarray(v, dtype=float)))


def load_episodes(val_dir):
    """(eid, poses, T_usable) in rollout.collect order (sorted ep_*.pt)."""
    files = sorted(glob.glob(os.path.join(val_dir, "ep_*.pt")))
    eps = []
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        T = min(d["frames_u8"].shape[0], d["actions"].shape[0],
                d["poses"].shape[0])
        eps.append((i, d["poses"].float(), int(T)))
    return files, eps


def paired(floor_pw, model_pw, eid, key, idx=None):
    """floor - model  (positive = the MODEL wins), paired episode-cluster."""
    a = np.asarray(floor_pw[key], dtype=float)
    b = np.asarray(model_pw[key], dtype=float)
    e = list(eid)
    if idx is not None:
        a, b, e = a[idx], b[idx], [e[i] for i in idx]
    return _ci.paired_episode_cluster_bootstrap(
        a, b, e, n_boot=N_BOOT, seed=SEED, reduce=_reduce_name(key))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", default="/workspace/val40cache")
    ap.add_argument("--results-dir",
                    default="/workspace/TanitAD/taniteval/results")
    ap.add_argument("--out", default="/workspace/ctrv_readjudication.json")
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--v-gate", type=float, default=2.0)
    a = ap.parse_args()

    t0 = time.time()
    files, eps = load_episodes(a.val_dir)
    built = build_floors(eps, v_gate=a.v_gate)
    print(f"[floors] {len(files)} episodes -> {built['gt'].shape[0]} windows "
          f"({time.time() - t0:.1f}s)", flush=True)

    dumps = sorted(glob.glob(os.path.join(a.results_dir, "windows_*.pt")))
    if a.arms:
        dumps = [d for d in dumps
                 if Path(d).stem.replace("windows_", "") in a.arms]

    out = {
        "block": "bench-eval/E-CTRV",
        "question": ("does the driving block's straight-line-only floor family "
                     "(cv, holdv0) manufacture the model's turn/lateral wins?"),
        "prereg": "PREREGISTRATION.md (this package)",
        "val_dir": a.val_dir, "n_episodes": len(files),
        "v_gate_mps": a.v_gate,
        "estimator": {
            "interval": "episode_cluster_bootstrap",
            "delta": "paired_episode_cluster_bootstrap",
            "n_boot": N_BOOT, "seed": SEED,
            "resampling_unit": "val episode",
            "orientation": "every paired delta is floor - model; "
                           "positive = the model wins",
        },
        "floors_compared": list(FLOOR_KEYS),
        "arms": {}, "alignment": {}, "floor_vs_floor": {},
    }

    for path in dumps:
        arm = Path(path).stem.replace("windows_", "")
        win = torch.load(path, map_location="cpu", weights_only=False)
        align = verify_alignment(built, win)
        out["alignment"][arm] = align
        if not align["aligned"]:
            print(f"[SKIP] {arm}: alignment FAILED {align}", flush=True)
            out["arms"][arm] = {"refused": "window alignment failed", **align}
            continue

        gt = win["gt"].float()
        eid = list(win["eid"])
        pw = {"model": D.per_window(win["pred"].float(), gt)}
        for f in FLOOR_KEYS:
            src = built["holdv0"] if f == "holdv0" else built[f]
            pw[f] = D.per_window(src.float(), gt)

        row = {"n_windows": int(gt.shape[0]),
               "n_episodes": len(set(eid)),
               "method": win.get("method"),
               "point": {}, "vs_floor_paired": {}, "strata": {}}
        for k in METRICS:
            row["point"][k] = {src: round(_point(pw[src][k], k), 4)
                               for src in ("model",) + FLOOR_KEYS}
        for f in FLOOR_KEYS:
            row["vs_floor_paired"][f] = {
                k: paired(pw[f], pw["model"], eid, k) for k in METRICS}

        # ---- strata: exactly the ones the published verdicts live on ------ #
        head_deg = win["head_deg"]
        v0 = win["speed"].float()
        masks = {}
        cb = D.curv_buckets(head_deg)
        for lab in ("straight", "gentle", "sharp"):
            masks[f"curv_{lab}"] = np.asarray(cb == lab)
        for lab, m in D.kinematic_strata(gt, v0, head_deg).items():
            masks[lab] = np.asarray(m)
        sp, _thr = D.speed_strata(v0)
        for lab, m in sp.items():
            masks[f"speed_{lab}"] = np.asarray(m.numpy())

        for lab, m in masks.items():
            idx = np.nonzero(m)[0]
            if idx.size == 0:
                continue
            srow = {"n": int(idx.size),
                    "low_confidence": bool(idx.size < D.MIN_N_STRATUM),
                    "point": {}, "vs_ctrv_gated_paired": {},
                    "vs_cv_paired": {}}
            for k in ("ade_0_2s", "lat_abs_2s_m", "heading_mae_2s_deg",
                      "pathgeom_crosstrack_m"):
                srow["point"][k] = {
                    src: round(_point(np.asarray(pw[src][k])[idx], k), 4)
                    for src in ("model",) + FLOOR_KEYS}
                srow["vs_ctrv_gated_paired"][k] = paired(
                    pw["ctrv_gated"], pw["model"], eid, k, idx)
                srow["vs_cv_paired"][k] = paired(pw["cv"], pw["model"], eid,
                                                 k, idx)
            row["strata"][lab] = srow

        # ---- the verdict-flip ledger -------------------------------------- #
        flips = []
        for k in METRICS:
            cvp, ctp = row["vs_floor_paired"]["cv"][k], \
                row["vs_floor_paired"]["ctrv_gated"][k]
            if _fav(cvp) != _fav(ctp):
                flips.append({"scope": "overall", "metric": k,
                              "vs_cv": _fav(cvp), "vs_ctrv_gated": _fav(ctp),
                              "delta_cv": cvp["delta"],
                              "delta_ctrv_gated": ctp["delta"]})
        for lab, srow in row["strata"].items():
            for k, cvp in srow["vs_cv_paired"].items():
                ctp = srow["vs_ctrv_gated_paired"][k]
                if _fav(cvp) != _fav(ctp):
                    flips.append({"scope": lab, "metric": k, "n": srow["n"],
                                  "vs_cv": _fav(cvp),
                                  "vs_ctrv_gated": _fav(ctp),
                                  "delta_cv": cvp["delta"],
                                  "delta_ctrv_gated": ctp["delta"]})
        row["verdict_flips"] = flips
        out["arms"][arm] = row
        print(f"[{arm}] ade model {row['point']['ade_0_2s']['model']} · "
              f"cv {row['point']['ade_0_2s']['cv']} · "
              f"ctrv_gated {row['point']['ade_0_2s']['ctrv_gated']} · "
              f"{len(flips)} verdict flips  ({time.time() - t0:.0f}s)",
              flush=True)

    # ---- floor vs floor (model-independent) ------------------------------- #
    any_arm = next((p for p in dumps), None)
    if any_arm:
        win = torch.load(any_arm, map_location="cpu", weights_only=False)
        gt, eid = win["gt"].float(), list(win["eid"])
        fpw = {f: D.per_window(built[f].float(), gt) for f in FLOOR_KEYS}
        for f in ("cv", "holdv0", "ctrv"):
            out["floor_vs_floor"][f"ctrv_gated_minus_{f}"] = {
                k: paired(fpw[f], fpw["ctrv_gated"], eid, k)
                for k in METRICS}
        # per-window win counts (the 2026-07-15 "best-of-3" question)
        de = {f: np.asarray(fpw[f]["ade_0_2s"], dtype=float)
              for f in ("cv", "holdv0", "ctrv_gated")}
        stack = np.stack([de["cv"], de["holdv0"], de["ctrv_gated"]])
        wins = stack.argmin(0)
        out["floor_vs_floor"]["win_counts_ade_0_2s"] = {
            "cv": int((wins == 0).sum()), "holdv0": int((wins == 1).sum()),
            "ctrv_gated": int((wins == 2).sum()), "n": int(wins.size)}
        out["floor_vs_floor"]["best_of_3_ade_0_2s"] = round(
            float(stack.min(0).mean()), 4)

    out["wallclock_s"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"[done] {a.out}  {out['wallclock_s']}s", flush=True)


def _fav(p):
    if not p.get("separated"):
        return "tie"
    return "model" if p["delta"] > 0 else "floor"


if __name__ == "__main__":
    main()
