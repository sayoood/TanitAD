"""Cross-corpus trivial-baseline floor: Cosmos urban vs comma2k19 highway.

Answers the pre-registered question from
`Benchmarks & Eval/DRIVING_DIAGNOSTIC_FRAMEWORK.md` section A/C: is the
constant-velocity floor that underpins the "10-15x worse than CV" verdict
corpus- and curvature-dependent? If yes, the diagnostic's skill DENOMINATOR must
be stratified (and NAVSIM v2's CV triviality filter, arXiv 2506.04218, is
likewise stratum-sensitive).

DATA-ONLY, $0, CPU. No model output is computed here -- this establishes the
honest baseline floor the model's future ADE is divided by.

  Cosmos:  local vehicle_pose tars (30 fps, stride 3 -> 10 Hz), poses_to_signals.
  comma:   local eval episodes ep_*.pt  (poses [T,4] = [x,y,yaw,v] at 10 Hz).

Usage:
  python run_baseline_floor.py \
      --cosmos-root C:/Users/Admin/tanitad-data/cosmos_bench3 \
      --comma-root  C:/Users/Admin/tanitad-data/eval/comma2k19-val-61c46fca8f7f \
      --out results_baseline_floor.json
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import baseline_predictors as bp

SRC_FPS, TARGET_HZ = 30.0, 10.0
DT = 1.0 / TARGET_HZ
BASELINES = ("cv", "go_straight", "ctrv")
HORIZONS = (1.0, 2.0)


def _ensure_stack_on_path() -> None:
    for c in [Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD/stack"),
              Path(r"C:/Users/Admin/wt-be/stack")]:
        if (c / "tanitad").is_dir():
            sys.path.insert(0, str(c))
            return


def load_cosmos(root: Path) -> list[np.ndarray]:
    """Each clip -> poses [T,4]=[x,y,yaw,v] at 10 Hz via the shared derivation."""
    _ensure_stack_on_path()
    import io, tarfile
    from tanitad.data.cosmos_drive import poses_to_signals
    stride = int(round(SRC_FPS / TARGET_HZ))
    out = []
    for tar in sorted(glob.glob(str(root / "vehicle_pose" / "*.tar"))):
        frames = {}
        with tarfile.open(tar) as tf:
            for m in tf.getmembers():
                if m.name.endswith(".vehicle_pose.npy"):
                    idx = int(m.name.split(".")[-3])
                    frames[idx] = np.load(io.BytesIO(tf.extractfile(m).read())).reshape(4, 4)
        ego4 = np.stack([frames[i] for i in sorted(frames)]).astype(np.float64)[::stride]
        _, poses = poses_to_signals(ego4, DT)
        out.append(poses.astype(np.float64))
    return out


def load_comma(root: Path) -> list[np.ndarray]:
    import torch
    out = []
    for f in sorted(glob.glob(str(root / "ep_*.pt"))):
        d = torch.load(f, map_location="cpu", weights_only=False)
        out.append(d["poses"].numpy().astype(np.float64))
    return out


def _agg(vals: list[float]) -> dict:
    a = np.asarray(vals, dtype=float)
    return {"n": int(a.size),
            "median": round(float(np.median(a)), 3),
            "mean": round(float(np.mean(a)), 3),
            "p90": round(float(np.percentile(a, 90)), 3)}


def summarize(recs: list[dict]) -> dict:
    """Overall + per-stratum baseline ADE and the best-baseline floor."""
    def block(subset):
        if not subset:
            return None
        d = {"n_anchors": len(subset)}
        for h in HORIZONS:
            hk = f"{h:g}s"
            per_base = {b: _agg([r[f"ade_{b}_{hk}"] for r in subset]) for b in BASELINES}
            d[hk] = {"ade": per_base}
            # best-baseline floor per anchor -> the honest denominator
            floor = [min(r[f"ade_{b}_{hk}"] for b in BASELINES) for r in subset]
            d[hk]["best_baseline_floor"] = _agg(floor)
            # which baseline wins most often
            wins = defaultdict(int)
            for r in subset:
                wins[min(BASELINES, key=lambda b: r[f"ade_{b}_{hk}"])] += 1
            d[hk]["win_counts"] = dict(wins)
        return d

    out = {"overall": block(recs)}
    for key, field in (("by_curvature", "curv_stratum"), ("by_speed", "speed_stratum")):
        out[key] = {}
        strata = sorted({r[field] for r in recs})
        for s in strata:
            out[key][s] = block([r for r in recs if r[field] == s])
    # population (starvation check, framework section D2)
    pop = defaultdict(int)
    for r in recs:
        pop[r["curv_stratum"]] += 1
    tot = len(recs)
    out["curvature_population"] = {s: {"n": n, "frac": round(n / tot, 3)}
                                   for s, n in sorted(pop.items())}
    return out


def run_corpus(name: str, sequences: list[np.ndarray]) -> dict:
    recs = []
    for poses in sequences:
        recs.extend(bp.evaluate_sequence(poses, DT, HORIZONS))
    summary = summarize(recs)
    summary["corpus"] = name
    summary["n_sequences"] = len(sequences)
    summary["ego_speed_mps"] = _agg([r["speed"] for r in recs])
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosmos-root", required=True)
    ap.add_argument("--comma-root", required=True)
    ap.add_argument("--out", default="results_baseline_floor.json")
    args = ap.parse_args()

    result = {}
    result["cosmos_urban"] = run_corpus("cosmos_urban", load_cosmos(Path(args.cosmos_root)))
    result["comma_highway"] = run_corpus("comma_highway", load_comma(Path(args.comma_root)))

    Path(args.out).write_text(json.dumps(result, indent=1))

    # console summary
    for corpus in ("cosmos_urban", "comma_highway"):
        r = result[corpus]
        print(f"\n=== {corpus}  ({r['n_sequences']} seq, "
              f"{r['overall']['n_anchors']} anchors, "
              f"v={r['ego_speed_mps']['median']} m/s median) ===")
        for h in HORIZONS:
            hk = f"{h:g}s"
            ov = r["overall"][hk]
            floor = ov["best_baseline_floor"]
            print(f"  @{hk}: floor(best-baseline) ADE median={floor['median']} m "
                  f"(mean {floor['mean']}) | "
                  + " ".join(f"{b}={ov['ade'][b]['median']}" for b in BASELINES)
                  + f" | wins={ov['win_counts']}")
        print("  curvature population:", {s: v["frac"] for s, v in r["curvature_population"].items()})
        print("  floor by curvature @1s:",
              {s: (r["by_curvature"][s]["1s"]["best_baseline_floor"]["median"]
                   if r["by_curvature"].get(s) else None)
               for s in ("straight", "gentle", "sharp")})
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
