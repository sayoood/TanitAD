#!/usr/bin/env python3
"""Does a held-out read PREDICT navtest? The same two scorers, the same 3,137 held-out sets, two truths.

SFT-3 arm A read +0.56 [+0.08, +1.14] on held-out against the TEACHER'S labels (v5) and -0.20 [-1.30, +0.92] on navtest
(eval/RESULT_SFT3.md). SFT-4 reads its stage 1 against NAVSIM PDM targets of the executed plans
(refe/onpolicy_relabel_pdm.py, validated against NAVSIM's own cache). This script re-reads arm A vs the deployed scorer on
the held-out sets that carry BOTH labels, picking with navsim_v1 exactly as scorer_finetune.evaluate does (bf16 autocast,
batch 16):
  truth = teacher labels -> must reproduce the stage-1 read (control: the same sets, the same picks);
  truth = PDM targets    -> if it reads about -0.2 like navtest, SFT-4's stage 1 is a predictive yardstick;
                            if it reads like the teacher truth, held-out reads are optimistic whatever the labels.
Paired log-cluster bootstrap over the held-out logs (10,000, seed 20260927).
    python heldout_truth_compare.py --base <model_final.pt> --arm <model_sft_A.pt> --pdm-labels-heldout <dir> --out <json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.environ.get("REFE_CODE", os.path.dirname(os.path.abspath(__file__))))
import ckpt_io  # noqa: E402
import train as T  # noqa: E402
from model import REFe, REFeConfig  # noqa: E402

V1_W = (5.0, 5.0, 2.0)


def agg_v1(p):
    return p[..., 0] * p[..., 1] * (V1_W[0] * p[..., 2] + V1_W[1] * p[..., 3] + V1_W[2] * p[..., 4]) / sum(V1_W)


def key(r):
    return (r.get("log_name", ""), r.get("token", ""), int(r.get("step", 0)), int(r.get("rank", 0)))


def boot_logs(diff: dict, logs: dict, n=10000, seed=20260927):
    toks = sorted(diff)
    by: dict = {}
    for i, t in enumerate(toks):
        by.setdefault(logs[t], []).append(i)
    groups = [np.asarray(v) for _k, v in sorted(by.items())]
    vals = np.asarray([diff[t] for t in toks], np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        means[b] = vals[np.concatenate([groups[g] for g in pick])].mean()
    return float(vals.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load(path, cfg, dev, backbone):
    m = REFe(ckpt_io.config_for_checkpoint(cfg, path)).to(dev)
    ckpt_io.load_for_inference(m, path, map_location=dev, backbone=backbone)
    return m.eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--targets", default="/workspace/data/refe_navtrain/train_grow")
    ap.add_argument("--images", default="/workspace/data/navtrain_pixels")
    ap.add_argument("--calib", default="/workspace/data/refe_navtrain/train_grow/calib_table.json")
    ap.add_argument("--heldout", default="/workspace/data/refe_heldout/sets")
    ap.add_argument("--pdm-labels-heldout", required=True)
    ap.add_argument("--backbone", default="vitl16")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    cfg = REFeConfig.for_backbone(a.backbone)
    ho_t = T.OnPolicyBank(a.heldout, cfg.n_proposals, cfg.horizon_steps)                       # the teacher's labels (v5)
    ho_p = T.OnPolicyBank(a.heldout, cfg.n_proposals, cfg.horizon_steps, pdm_labels=a.pdm_labels_heldout)
    ds = T.TargetBank(a.targets, a.images, cfg, calib=a.calib, onpolicy=ho_t)
    idx = [i for i, r in enumerate(ds.rows) if key(r) in ho_p.pdm_keys and key(r) in ho_t.by]
    same_props = all(np.array_equal(ho_t.by[key(ds.rows[i])][1], ho_p.by[key(ds.rows[i])][1]) for i in idx)
    print(f"held-out sets with both truths: {len(idx)} (PDM {ho_p.n_pdm}, stale {ho_p.n_pdm_stale}); proposals identical: {same_props}",
          flush=True)
    base, arm = load(a.base, cfg, dev, a.backbone), load(a.arm, cfg, dev, a.backbone)
    dl = DataLoader(Subset(ds, idx), batch_size=16, shuffle=False, num_workers=2)
    rec, j = {}, 0
    with torch.no_grad():
        for batch in dl:
            img, ego, goal, _tgt, cxy, _ctg, _cm, _ci, cal = batch
            img, ego, goal, cxy = (x.to(dev, non_blocking=True) for x in (img, ego, goal, cxy))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, _, sb = base(img, ego, goal, calib=cal if cal.numel() else None, score_extra=cxy)
                _, _, sa = arm(img, ego, goal, calib=cal if cal.numel() else None, score_extra=cxy)
            ib = agg_v1(torch.sigmoid(sb.float())).argmax(1).tolist()
            ia = agg_v1(torch.sigmoid(sa.float())).argmax(1).tolist()
            for bi in range(len(ib)):
                r = ds.rows[idx[j]]; j += 1
                k = key(r)
                tt = agg_v1(torch.as_tensor(ho_t.by[k][2]))
                tp = agg_v1(torch.as_tensor(ho_p.by[k][2]))
                rec["|".join(map(str, k))] = {"log": k[0], "ib": ib[bi], "ia": ia[bi],
                                              "t_b": float(tt[ib[bi]]), "t_a": float(tt[ia[bi]]),
                                              "p_b": float(tp[ib[bi]]), "p_a": float(tp[ia[bi]]),
                                              "t_best": float(tt.max()), "p_best": float(tp.max())}
    logs = {t: v["log"] for t, v in rec.items()}
    out = {"n_sets": len(rec), "n_logs": len(set(logs.values())), "proposals_identical_across_truths": same_props,
           "picks_identical": sum(v["ib"] == v["ia"] for v in rec.values()), "base": a.base, "arm": a.arm,
           "estimator": "paired log-cluster bootstrap over the held-out logs, 10,000 resamples, seed 20260927"}
    for nm, (fb, fa, fbest) in {"teacher_truth": ("t_b", "t_a", "t_best"), "pdm_truth": ("p_b", "p_a", "p_best")}.items():
        d = {t: 100 * (v[fa] - v[fb]) for t, v in rec.items()}
        mu, lo, hi = boot_logs(d, logs)
        out[nm] = {"pick_base": 100 * float(np.mean([v[fb] for v in rec.values()])),
                   "pick_arm": 100 * float(np.mean([v[fa] for v in rec.values()])),
                   "best_of_64": 100 * float(np.mean([v[fbest] for v in rec.values()])),
                   "arm_minus_base": {"mean": mu, "ci95": [lo, hi]},
                   "zero_picks": {"base": sum(v[fb] == 0 for v in rec.values()), "arm": sum(v[fa] == 0 for v in rec.values())}}
    json.dump(out, open(a.out, "w"), indent=1)
    json.dump(rec, open(os.path.splitext(a.out)[0] + "_rows.json", "w"))
    print("ZZTRUTHCMP", json.dumps({k: out[k] for k in ("n_sets", "picks_identical", "teacher_truth", "pdm_truth")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
