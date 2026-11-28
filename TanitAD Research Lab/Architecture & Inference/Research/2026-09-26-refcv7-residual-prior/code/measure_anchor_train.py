#!/usr/bin/env python3
"""refcv7 NEW-1, the anchor question on the TRAIN side: does the refcv6 117-control vocabulary,
reinterpreted as RESIDUALS on the prior, cover the human's plan at least as well as the same
vocabulary rolled ABSOLUTE (refcv6)? CPU, metadata only (poses + recorded actions), no model.

Surface: the train-a6 v2 cache manifest (139 TRAIN clips, never an eval clip), every window
t0 = W-1, W-1+stride, ... with a complete 60-step future. GT = refb_labels.waypoint_targets on
the clip's own poses (the trainer's target function). Oracle-in-vocabulary = min over the 117
anchors of the window's ADE. Paired episode-cluster bootstrap (taniteval.ci), cluster = clip.

usage: python measure_anchor_train.py --repo <tree> [--stride 5]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--cache", default="D:/refcv6_eval_kit/data/refcv6-b1-416x1024-train-a6/_v2manifest.pt")
    ap.add_argument("--anchors", default="D:/refcv6_eval_kit/data/anchors/refc_anchors_6s_v0cond_alat_117.pt")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(PKG, "raw", "measure_anchor_train.json"))
    a = ap.parse_args()
    repo = os.path.abspath(a.repo)
    for p in (os.path.join(repo, "stack", "scripts"), os.path.join(repo, "stack"),
              os.path.join(repo, "taniteval")):
        sys.path.insert(0, p)
    import torch
    import tanitad
    assert os.path.abspath(tanitad.__file__).startswith(os.path.join(repo, "stack")), tanitad.__file__
    from tanitad.models import kinematic_prior as KP
    from tanitad.refs.refc_sampler import roll_controls
    from taniteval import ci
    import refb_labels
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    cm = torch.load(a.cache, map_location="cpu", weights_only=False)
    art = torch.load(a.anchors, map_location="cpu", weights_only=False)
    ctrl = art["controls"].float()
    N = ctrl.shape[0]
    HZ = (5, 10, 15, 20, 30, 40, 50, 60)
    W, S2 = 8, [0, 1, 2, 3]
    modes = ["ha0_ext", "ha0_ext_pose", "cv_yawrate"]
    orc = {k: [] for k in ["abs2", "abs6"] + [f"{m}_{h}" for m in modes for h in (2, 6)]}
    # the trainer's a_star (GT-nearest anchor over all 8 slots, squared L2 -- refc_v3_train.py
    # :3668-3671): how concentrated is the classification target in each space?
    astar = {k: [] for k in ["abs"] + modes}
    i_zero = int(((ctrl == 0).all(dim=-1)).nonzero().flatten()[0])
    prior_ade = {f"{m}_{h}": [] for m in modes for h in (2, 6)}
    eid = []
    t_start = time.time()
    for ci_, (P, A) in enumerate(zip(cm["poses"], cm["actions"])):
        P, A = P.float(), A.float()
        T_ = int(P.shape[0])
        t0s = torch.arange(W - 1, T_ - 60, a.stride)
        if t0s.numel() == 0:
            continue
        idx = t0s[:, None] - (W - 1) + torch.arange(W)[None]
        ph, ah = P[idx], A[idx]
        v0 = P[t0s, 3]
        fut = torch.stack([P[t + 1: t + 61] for t in t0s.tolist()])        # [n, 60, 4]
        g = refb_labels.waypoint_targets(P[t0s], fut, HZ)                   # [n, 8, 2]
        n = int(t0s.numel())
        dl = ctrl[None, :, None, :].expand(n, N, len(HZ), 2)
        bank = roll_controls(dl, v0, HZ, control_units="alat")
        e = torch.linalg.norm(bank - g[:, None], dim=-1)                    # [n, N, 8]
        orc["abs2"].append(e[..., S2].mean(-1).min(1).values)
        orc["abs6"].append(e.mean(-1).min(1).values)
        astar["abs"].append(((bank - g[:, None]) ** 2).sum(-1).sum(-1).argmin(1))
        for m in modes:
            a0, k0 = KP.prior_controls(m, ph, W, ah if KP.needs_actions(m) else None)
            br = KP.roll_plan(dl, a0, k0, v0, HZ, control_units="alat")
            er = torch.linalg.norm(br - g[:, None], dim=-1)
            orc[f"{m}_2"].append(er[..., S2].mean(-1).min(1).values)
            orc[f"{m}_6"].append(er.mean(-1).min(1).values)
            astar[m].append(((br - g[:, None]) ** 2).sum(-1).sum(-1).argmin(1))
            pp = KP.prior_path(a0, k0, v0, HZ)
            ep = torch.linalg.norm(pp - g, dim=-1)
            prior_ade[f"{m}_2"].append(ep[:, S2].mean(-1))
            prior_ade[f"{m}_6"].append(ep.mean(-1))
        eid += [ci_] * n
    cat = lambda d: {k: torch.cat(v).double().numpy() for k, v in d.items()}  # noqa: E731
    O, PA = cat(orc), cat(prior_ade)
    out = {"tool": "measure_anchor_train.py", "cache": a.cache, "anchors": a.anchors,
           "surface": "TRAIN (train-a6 manifest), windows with a complete 60-step future",
           "stride": a.stride, "n_windows": len(eid), "n_clips": len(set(eid)),
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap n_boot 2000 seed 0, cluster = clip",
           "oracle_in_vocab_mean_m": {k: float(v.mean()) for k, v in O.items()},
           "prior_ade_mean_m": {k: float(v.mean()) for k, v in PA.items()},
           "wall_s": round(time.time() - t_start, 1)}
    out["a_star_target"] = {}
    for k, v in astar.items():
        idx_all = torch.cat(v)
        p = torch.bincount(idx_all, minlength=N).double() / idx_all.numel()
        nz = p[p > 0]
        out["a_star_target"][k] = {
            "entropy_nats": float(-(nz * nz.log()).sum()), "max_entropy_nats": float(np.log(N)),
            "n_anchors_used": int((p > 0).sum()), "frac_at_zero_anchor": float(p[i_zero]),
            "top1_frac": float(p.max())}
    for m in modes:
        for h in (2, 6):
            out[f"residual_{m}_minus_absolute_0_{h}s"] = ci.paired_episode_cluster_bootstrap(
                O[f"{m}_{h}"], O[f"abs{h}"], eid, n_boot=2000, seed=0)
        for h in (2, 6):
            if m != "ha0_ext":
                out[f"prior_{m}_minus_prior_ha0_ext_0_{h}s"] = ci.paired_episode_cluster_bootstrap(
                    PA[f"{m}_{h}"], PA[f"ha0_ext_{h}"], eid, n_boot=2000, seed=0)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: out[k] for k in ("n_windows", "n_clips", "oracle_in_vocab_mean_m",
                                          "prior_ade_mean_m", "wall_s")}, indent=1))
    for k, v in out.items():
        if isinstance(v, dict) and "delta" in v:
            print(k, v["delta"], [v["lo"], v["hi"]], "separated" if v["separated"] else "tie")
    return 0


if __name__ == "__main__":
    sys.exit(main())
