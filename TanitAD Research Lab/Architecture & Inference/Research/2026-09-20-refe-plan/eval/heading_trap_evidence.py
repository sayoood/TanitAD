#!/usr/bin/env python3
"""EVIDENCE for the step-19 heading defect's MODEL-SIDE cause (measures.py M6). CPU, local artifacts only.

  A  the symptom per native step, ep015 dump (all 64 slots x 200 sub200 tokens): raw heading range, share beyond
     +-pi, wrapped error vs the path tangent, correlation with the step's x
  B  per snapshot 005..015 (stored E-6 tables; their 4.0 s pose IS native step 19, copied verbatim): all-slot raw
     heading, the WTA winner's wrapped error vs the HUMAN future at 4.0 s and at 3.5 s, corr(raw heading, x)
  C  traj_head's last Linear per local snapshot 001..015: the step-19 heading row vs the other heading rows and x19
  D  the TARGETS: every local live-bank row (DEV10 rank 0 + rank 1), steps 17-19 -- wrapped error vs the tangent,
     |heading| range, share beyond +-pi, max |target heading| (what the plain-loss seam guard reads)
  E  the loss: model.wta_loss's gradient on the winner's heading is identical at every step, and a heading on the
     -2*pi branch scores ~0 (the branch ambiguity)
Writes raw/2026-09-27-training-measures/m6_heading_trap_evidence.json.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG / "refe"))
DATA = Path("D:/Projects/TanitAD/data")
OUT = PKG / "raw" / "2026-09-27-training-measures" / "m6_heading_trap_evidence.json"
EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def tangent(T):
    P = np.concatenate([np.zeros(T.shape[:-2] + (1, 3)), T], -2)
    d = np.diff(P[..., :2], axis=-2)
    return np.arctan2(d[..., 1], d[..., 0]), np.linalg.norm(d, axis=-1) > 0.05


def main() -> int:
    import torch
    res = {}
    T = np.load(DATA / "refe_navtest/proptable/sub200_ep015/stop_candidate_dump.npz")["traj"].astype(np.float64)
    tan, mv = tangent(T)
    err = np.abs(wrap(T[..., 2] - tan))
    res["A_ep015_dump_per_step"] = {str(k): {
        "raw_min": round(float(T[..., k, 2].min()), 3), "raw_median": round(float(np.median(T[..., k, 2])), 3),
        "raw_max": round(float(T[..., k, 2].max()), 3),
        "pct_beyond_pi": round(100 * float(np.mean(np.abs(T[..., k, 2]) > np.pi)), 1),
        "median_wrap_err_vs_tangent": round(float(np.median(err[..., k][mv[..., k]])), 4),
        "corr_with_x": round(float(np.corrcoef(T[..., k, 2].ravel(), T[..., k, 0].ravel())[0, 1]), 3)}
        for k in range(20)}
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    B = {}
    for e in ("005", "008", "011", "012", "013", "014", "015"):
        z = np.load(DATA / f"refe_navtest/proptable/sub200_ep{e}/table.npz")
        P = z["proposals"].astype(np.float64)
        G = np.array([exp[t]["human_future_poses"] for t in z["token"]], dtype=np.float64)
        w = np.abs(P[..., :2] - G[:, None, :, :2]).sum(-1).mean(-1).argmin(1)
        ar = np.arange(len(w))
        B[e] = {"all_slots_raw_h4_median": round(float(np.median(P[..., 7, 2])), 3),
                "all_slots_pct_beyond_pi": round(100 * float(np.mean(np.abs(P[..., 7, 2]) > np.pi)), 1),
                "winner_median_abs_err_4.0s": round(float(np.median(np.abs(wrap(P[ar, w, 7, 2] - G[:, 7, 2])))), 3),
                "winner_median_abs_err_3.5s": round(float(np.median(np.abs(wrap(P[ar, w, 6, 2] - G[:, 6, 2])))), 3),
                "corr_raw_h4_x4_all_slots": round(float(np.corrcoef(P[..., 7, 2].ravel(), P[..., 7, 0].ravel())[0, 1]), 3)}
    res["B_per_snapshot_sub200_vs_human"] = B
    C = {}
    for e in ("001", "002", "003", "005", "008", "011", "012", "013", "014", "015"):
        sd = torch.load(DATA / f"refe_runs_eval/snap_epoch{e}.pt", map_location="cpu", weights_only=False)
        W = sd["model_partial"]["traj_head.2.weight"].double()
        n = W.norm(dim=1)
        C[e] = {"step": sd["meta"].get("step"), "heading_rows_0_18_median": round(float(n[2:57:3].median()), 3),
                "heading_rows_0_18_max": round(float(n[2:57:3].max()), 3), "heading_row_19": round(float(n[59]), 3),
                "x19_row": round(float(n[57]), 3)}
    res["C_traj_head_row_norms"] = C
    D = {}
    for name, p in (("rank0", DATA / "refe_navtrain10/r0/targets_rank0.jsonl"),
                    ("rank1", DATA / "refe_navtrain10/aug/targets_aug.jsonl")):
        Tt = np.array([json.loads(l)["traj"] for l in open(p, encoding="utf-8")], dtype=np.float64)
        ta, tm = tangent(Tt)
        et = np.abs(wrap(Tt[..., 2] - ta))
        D[name] = {"rows": len(Tt), "max_abs_target_heading": round(float(np.abs(Tt[..., 2]).max()), 3)}
        for k in (17, 18, 19):
            D[name][f"step{k}"] = {"median_wrap_err_vs_tangent": round(float(np.median(et[:, k][tm[:, k]])), 4),
                                   "max_wrap_err_vs_tangent": round(float(et[:, k][tm[:, k]].max()), 3),
                                   "pct_beyond_pi": round(100 * float(np.mean(np.abs(Tt[:, k, 2]) > np.pi)), 2)}
    res["D_live_bank_targets"] = D
    from model import wta_loss
    g = torch.Generator().manual_seed(0)
    B_, M_ = 6, 64
    tgt = torch.zeros(B_, 20, 3)
    tgt[..., 0] = torch.linspace(1, 40, 20)
    tgt[..., 2] = 0.3
    traj = (torch.randn(B_, M_, 20, 3, generator=g) * 2).requires_grad_(True)
    loss, idx = wta_loss(traj, tgt)
    loss.backward()
    gw = traj.grad[torch.arange(B_), idx][..., 2].abs().mean(0)
    t2 = traj.detach().clone()
    t2[torch.arange(B_), idx, 19, 2] = 0.3 - 2 * math.pi
    t3 = traj.detach().clone()
    t3[torch.arange(B_), idx, 19, 2] = 0.3
    l2, _ = wta_loss(t2, tgt)
    l3, _ = wta_loss(t3, tgt)
    res["E_wta_loss"] = {"winner_heading_grad_by_step": [round(float(v), 7) for v in gw],
                         "same_at_every_step": bool(torch.allclose(gw, gw[0].expand_as(gw))),
                         "loss_with_winner_h19_on_target": float(l3), "loss_with_winner_h19_at_target_minus_2pi": float(l2),
                         "branch_shift_costs": float(l2 - l3)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"A_step19": res["A_ep015_dump_per_step"]["19"], "A_step18": res["A_ep015_dump_per_step"]["18"],
                      "B_ep005": B["005"], "C_ep015": C["015"], "C_ep001": C["001"],
                      "D_rank0_step19": D["rank0"]["step19"], "D_max": [D[k]["max_abs_target_heading"] for k in D],
                      "E": {k: v for k, v in res["E_wta_loss"].items() if k != "winner_heading_grad_by_step"}}, indent=1))
    print(f"ZZHEADING_EVIDENCE_OK -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
