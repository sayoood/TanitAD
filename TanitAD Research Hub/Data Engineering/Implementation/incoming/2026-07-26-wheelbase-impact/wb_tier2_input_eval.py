#!/usr/bin/env python3
"""TIER 2 — the INPUT-side impact of the wrong `WHEELBASE` on flagship-v1.

Runs on `tanitad-eval` (A40, free). Re-runs the deployed v1 (`flagship-30k`,
full-set ADE@2s 0.4271 on 881 windows / 40 clean-val episodes) with the steer
ACTION CHANNEL corrected per clip, PAIRED on exactly the same windows, and
compares against the shipped 2.9-derived channel.

    steer_shipped = atan(2.9   * kappa)
    steer_true    = atan(L_clip * kappa) = atan((L_clip/2.9) * tan(steer_shipped))
                    ^ exact inversion; no re-derivation from raw egomotion needed

⚠️ WHAT THIS CAN AND CANNOT ANSWER — state it with every number.
CAN: how much the trained-as-is model's open-loop trajectory moves when the
     steer channel it is fed at inference is corrected. That bounds the
     INPUT-side effect.
CANNOT: what a model TRAINED on corrected labels would do. A perturbed input at
     inference is a distribution shift for this checkpoint; a retrained model
     would have learned the corrected channel's statistics. Tier 2 is a
     sensitivity measurement, not a counterfactual training result.

Seven arms, all paired on the same windows so the shared per-window difficulty
cancels inside every bootstrap draw:
  A  shipped      steer as trained/evaluated (reproduces 0.4271)
  B  corrected    per-clip true wheelbase (gain 0.941 .. 1.109, the real defect)
  C  zeroed       steer := 0              (destroy the channel entirely)
  D  x1.5         steer *= 1.5            (~5x the real gain error, uniform)
  E  global_mean  ONE better constant: L* = 2.9568 (the corpus clip-mean
                  wheelbase, MEASURED) -> uniform gain 1.0196. Isolates the
                  BIAS half of the defect.
  F  dispersion   gain = L_clip / L*      Isolates the DISPERSION half — the
                  part no single constant can ever remove. B = E o F.
  G  harness_2p7  uniform gain 2.7/2.9 = 0.9310 — the convention the CLOSED-LOOP
                  harness actually feeds (`closedloop.py:99 WHEELBASE = 2.7`
                  vs the trainer's 2.9). Measures that train/serve skew.
C and D exist to CALIBRATE B: a |dADE| for B that is small only because the
model ignores steer altogether is a different finding from one that is small
because the correction is small, and only C/D can tell those apart. E and F
exist because the PI's option set contains "one better constant" implicitly —
if E carries the effect and F does not, a global constant is enough.

Estimator: paired episode-cluster bootstrap (`taniteval.ci`), B=2000, over the
40 val episodes. NEVER `overlapping_holdout_se`.

Usage (on the pod):
  python wb_tier2_input_eval.py --wheelbase-json val40_wheelbase.json \
      --out /root/wb_tier2_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import bench, ci, data, lateral, loaders, rollout  # noqa: E402
from taniteval.registry import MODELS  # noqa: E402

VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
SHIPPED_WB = 2.9
# MEASURED (wheelbase_population.json / tier1_per_clip.csv): clip-mean true
# wheelbase over the 2400 canonical train order lines. The best SINGLE constant.
CORPUS_MEAN_WB = 2.9568
HARNESS_WB = 2.7          # taniteval/closedloop.py:99


def entry(key):
    e = [m for m in MODELS if m["key"] == key]
    assert e, f"unknown model {key}"
    return e[0]


def patch_steer(eps, mode, wb_by_idx):
    """Return a deep-enough copy of the episode views with steer transformed."""
    outs = []
    for i, ep in enumerate(eps):
        a = ep.actions.clone()
        s = a[:, 0].double()
        if mode == "shipped":
            pass
        elif mode == "corrected":
            L = wb_by_idx[i]
            a[:, 0] = torch.atan(torch.tan(s) * (L / SHIPPED_WB)).float()
        elif mode == "zeroed":
            a[:, 0] = 0.0
        elif mode == "x1.5":
            a[:, 0] = torch.atan(torch.tan(s) * 1.5).float()
        elif mode == "global_mean":
            a[:, 0] = torch.atan(torch.tan(s) * (CORPUS_MEAN_WB / SHIPPED_WB)).float()
        elif mode == "dispersion":
            L = wb_by_idx[i]
            a[:, 0] = torch.atan(torch.tan(s) * (L / CORPUS_MEAN_WB)).float()
        elif mode == "harness_2p7":
            a[:, 0] = torch.atan(torch.tan(s) * (HARNESS_WB / SHIPPED_WB)).float()
        else:
            raise ValueError(mode)

        class V:                                    # same duck type as RawEp
            pass
        v = V()
        v.feats, v.poses, v.episode_id = ep.feats, ep.poses, ep.episode_id
        v.actions = a
        outs.append(v)
    return outs


def per_window_ade(win):
    de = torch.linalg.norm(win["pred"] - win["gt"], dim=-1).numpy().astype(float)
    return de.mean(axis=1)                          # ADE@2s per window [N]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flagship-30k")
    ap.add_argument("--wheelbase-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    wbmap = json.loads(Path(a.wheelbase_json).read_text())
    e = entry(a.model)
    dev = "cuda"
    L = loaders.load(e, dev)
    files = data.list_val_episodes(VAL, a.episodes) if hasattr(
        data, "list_val_episodes") else sorted(Path(VAL).glob("ep_*.pt"))[:a.episodes]
    files = list(files)
    print(f"[t2] {len(files)} val episodes", flush=True)
    eps = data.load_frames(files)
    wb_by_idx = [float(wbmap[Path(f).name]) for f in files]
    print(f"[t2] wheelbase mix: "
          f"{dict(zip(*np.unique(np.round(wb_by_idx, 3), return_counts=True)))}",
          flush=True)

    res = {"_meta": {
        "evidence_class": "MEASURED",
        "model": a.model, "ckpt": e["ckpt"], "val": VAL,
        "n_episodes": len(files), "shipped_WHEELBASE": SHIPPED_WB,
        "estimator": "paired_episode_cluster_bootstrap",
        "n_boot": a.n_boot,
        "limitation": ("inference-time input perturbation on a checkpoint "
                       "TRAINED with the 2.9 labels; bounds the input-side "
                       "effect only, NOT the training-side effect"),
    }, "arms": {}}

    wins, ades = {}, {}
    ARMS = ("shipped", "corrected", "zeroed", "x1.5", "global_mean",
            "dispersion", "harness_2p7")
    for mode in ARMS:
        t0 = time.time()
        veps = patch_steer(eps, mode, wb_by_idx)
        win = rollout.collect(L["model"], L["step_readout"], veps, dev,
                              speed_input=bool(e.get("speed_input")),
                              yaw_input=bool(e.get("yaw_input")),
                              dyn_input=bool(e.get("dyn_input")))
        wins[mode] = win
        ades[mode] = per_window_ade(win)
        full = bench._suite(win["pred"], win["gt"])
        res["arms"][mode] = {
            "full_set": {k: round(float(v), 6) for k, v in full.items()},
            "n_windows": int(win["pred"].shape[0]),
            "elapsed_s": round(time.time() - t0, 1),
        }
        print(f"[t2] {mode:10s} ade_0_2s={full['ade_0_2s']:.6f} "
              f"({res['arms'][mode]['elapsed_s']}s)", flush=True)

    eid = wins["shipped"]["eid"]
    res["_meta"]["n_windows"] = len(eid)

    # ---- paired deltas vs the shipped arm --------------------------------
    res["paired_vs_shipped"] = {}
    for mode in ARMS[1:]:
        blk = ci.paired_episode_cluster_bootstrap(
            ades[mode], ades["shipped"], eid, n_boot=a.n_boot, reduce="mean")
        blk["per_window_corr"] = float(np.corrcoef(ades[mode],
                                                   ades["shipped"])[0, 1])
        blk["max_abs_per_window_delta"] = float(
            np.abs(ades[mode] - ades["shipped"]).max())
        blk["mean_abs_per_window_delta"] = float(
            np.abs(ades[mode] - ades["shipped"]).mean())
        # trajectory-level: how far do the two predicted paths differ at all?
        dp = torch.linalg.norm(wins[mode]["pred"] - wins["shipped"]["pred"],
                               dim=-1).numpy()
        blk["pred_path_displacement_m"] = {
            "mean@2s": float(dp[:, -1].mean()),
            "p95@2s": float(np.quantile(dp[:, -1], 0.95)),
            "max@2s": float(dp[:, -1].max())}
        res["paired_vs_shipped"][mode] = blk
        print(f"[t2] paired {mode:10s} dADE={blk['delta']:+.5f} "
              f"[{blk['lo']:+.5f}, {blk['hi']:+.5f}] sep={blk['separated']}",
              flush=True)

    # ---- lateral / longitudinal decomposition ----------------------------
    res["lateral"] = {}
    for mode in ARMS:
        try:
            res["lateral"][mode] = lateral.block(wins[mode], mode="ego",
                                                 n_boot=a.n_boot)
        except Exception as ex:
            res["lateral"][mode] = {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}
    for mode in ARMS[1:]:
        try:
            res["lateral"][f"paired_cross_track_{mode}_vs_shipped"] = \
                lateral.paired_cross_track(wins[mode]["pred"],
                                           wins["shipped"]["pred"],
                                           wins["shipped"]["gt"], eid,
                                           n_boot=a.n_boot)
        except Exception as ex:
            res["lateral"][f"paired_cross_track_{mode}_vs_shipped"] = {
                "error": f"{type(ex).__name__}: {str(ex)[:200]}"}

    # ---- stratified by the wheelbase population --------------------------
    ep_wb = np.array([wb_by_idx[int(i)] for i in eid], float)
    res["by_wheelbase_population"] = {}
    for Lw in sorted(set(np.round(ep_wb, 3).tolist())):
        m = np.isclose(ep_wb, Lw, atol=1e-3)
        sub_eid = [int(i) for i, k in zip(eid, m) if k]
        n_ep = len(set(sub_eid))
        blk = {"n_windows": int(m.sum()), "n_episodes": n_ep,
               "wheelbase": float(Lw),
               "gain_true_over_shipped": float(Lw / SHIPPED_WB),
               "ade_shipped": float(ades["shipped"][m].mean()),
               "ade_corrected": float(ades["corrected"][m].mean()),
               "ade_zeroed": float(ades["zeroed"][m].mean())}
        if n_ep >= 3:
            blk["paired_corrected_vs_shipped"] = ci.paired_episode_cluster_bootstrap(
                ades["corrected"][m], ades["shipped"][m], sub_eid,
                n_boot=a.n_boot, reduce="mean")
        else:
            blk["paired_corrected_vs_shipped"] = {
                "skipped": f"only {n_ep} episodes — refuse an episode-cluster CI"}
        res["by_wheelbase_population"][f"{Lw:.3f}"] = blk

    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    print("[t2] wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
