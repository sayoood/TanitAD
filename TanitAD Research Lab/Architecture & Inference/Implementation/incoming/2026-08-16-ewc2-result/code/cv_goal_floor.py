"""The 0-PARAMETER goal floor: what σ do the trivial kinematic baselines reach?

WHY THIS IS NOT A SIDE QUEST. E-WC2 returns a σ and a verdict, but a bare σ cannot
distinguish two very different worlds:

  * *"a 6 s goal is not predictable on this corpus"* — then SEL-1 is dead as posed;
  * *"a 6 s goal is predictable, but NOT FROM THESE LATENTS"* — then SEL-1's estimand
    is intact and the defect is the feature surface.

§3.1's own requirement table already contains the discriminator: its **CV goal
(deployable, 0 params)** row scores **0.786** at N=256 against the supervised
selector's 0.471, i.e. a constant-velocity extrapolation is a *usable* goal. If a
ridge on frozen REF-C latents cannot match a baseline that uses ONE pose difference
and no learning at all, the verdict is about the surface, not about goals.

⛔ These baselines are NOT admissible as a deployed goal head — they read the ego
pose history, which is exactly the privileged channel the vision-only rule forbids at
inference. They are a FLOOR for interpretation, and they are labelled as such in the
output. (The ridge, by contrast, runs on `pooled`+`ctx`, both VISION_ONLY.)

0 GPU, poses only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import driving_diagnostic as dd
import e_wc2_sigma_star as E
import refc_dump_latents as R
from taniteval import data


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", required=True)
    ap.add_argument("--dump", required=True, help="for the GT endpoints + mask")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--steps", default="20,60")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    steps = [int(x) for x in a.steps.split(",")]

    d = torch.load(a.dump, map_location="cpu", weights_only=False)
    files = data.list_val_episodes(a.val, a.episodes)
    eps = [data.RawEp(data.load_episode(str(f), mmap=True), i)
           for i, f in enumerate(files)]

    BASE, EID = {}, []
    for ep in eps:
        starts = R.window_starts(ep.poses.shape[0])
        last = torch.tensor([t + R.WINDOW - 1 for t in starts])
        b = dd.baseline_waypoints(ep.poses.float(), last, wp_steps=steps)
        for k, v in b.items():
            BASE.setdefault(k, []).append(v.float())
        EID.extend([ep.episode_id] * len(starts))
    BASE = {k: torch.cat(v) for k, v in BASE.items()}

    gt = d["gt_endpoint"].float()
    valid = d["endpoint_valid"].bool()
    assert [int(x) for x in d["eid"]] == EID, (
        "rebuilt eid does not match the dump's — the baselines would be compared "
        "against another window's endpoint")

    rep = {"_what": "0-parameter kinematic goal floor vs the E-WC2 ridge",
           "_admissibility": ("⛔ INADMISSIBLE as a deployed goal head — these read "
                              "the EGO POSE HISTORY, the privileged channel the "
                              "vision-only rule forbids at inference. A floor for "
                              "interpretation only."),
           "_estimator": "full-set point estimate; sigma_perax = sqrt(mean(|e|^2)/2)",
           "n_windows_total": int(gt.shape[0]), "steps": steps, "horizons": {}}
    for i, k in enumerate(steps):
        m = valid[:, i] & torch.isfinite(gt[:, i]).all(dim=-1)
        row = {"horizon_s": k / 10.0, "n": int(m.sum()),
               "n_episodes": len(set(np.asarray(EID)[m.numpy()].tolist()))}
        for name, pred in BASE.items():
            res = (pred[:, i][m] - gt[:, i][m]).numpy()
            s = E.sigma_from_residuals(res)
            row[name] = {kk: round(vv, 6) for kk, vv in s.items()
                         if isinstance(vv, float)}
        rep["horizons"][f"{k/10.0:g}s"] = row
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
