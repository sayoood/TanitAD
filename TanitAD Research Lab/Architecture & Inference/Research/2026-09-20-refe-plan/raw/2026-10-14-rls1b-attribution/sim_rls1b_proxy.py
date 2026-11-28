#!/usr/bin/env python3
"""RL-S1b lever ranking by a CPU PROXY of the scorer fine-tune (zero A40 time).  Evidence class of every output: ESTIMATED (a proxy), never MEASURED.

WHY A PROXY.  attrib_rls1b.py (Q1, Q5) found that R0's TTC loss is NOT the reward's preference (58/59 TTC-worsening pick changes are DISPREFERRED
by R0's own reward) and is carried mostly by the EPO-trained PROGRESS outputs (EP / C / lane: swapping the safety outputs back to the BCE-only
arm leaves TTC at -0.80).  That is a GENERALISATION effect, and label-only reward statistics cannot rank levers on it.  The proxy reproduces the
generalisation channel cheaply: a small MLP adjusts the frozen BASE scorer's 7 logits per proposal from per-proposal features (the base logits +
5 trajectory features), trained on one log-half of RL-val with the trainer's own loss (BCE on the six PDM sub-scores + the teacher-lane BCE, and,
for an R arm, + lam_pg * (-J + beta KL(pi_SFT || pi)) with the policy, eps, T and ramp of scorer_finetune.epo_loss), and read on the OTHER
log-half by the deployed rule navsim_v1_lane.  2 log-disjoint folds x 3 proxy seeds; the R arm is paired with the C arm of the same fold+seed.

PRE-REGISTERED BEFORE ANY PROXY RESULT EXISTED (2026-10-14):
  FACE VALIDITY: the proxy may rank levers ONLY if its R0 replica ('a_R0|all', lam 1, beta 0.1) reproduces R0's measured signature on the held-out
  halves, pooled over folds and seeds: EP delta > 0 AND TTC delta < 0 versus the proxy C.  If it fails, the proxy ranks nothing and the lever is
  chosen on the MEASURED attribution alone.
  RANKING RULE: among the levers, the one whose pooled TTC delta is >= -0.5 (g5's bound, applied to the proxy as a screen) with the largest
  pooled PDMS delta; ties -> the smaller change to R0.  RL-val is a selection tier: this is lever selection, never an endpoint.

    python sim_rls1b_proxy.py --dl <download root> --out proxy.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import attrib_rls1b as AT  # noqa: E402  (loaders, candidate rewards, the selection rule)

RV, RR = AT.RV, AT.RR
T_POL, EPS, BETA = 0.1, 1e-6, 0.1
STEPS, RAMP_FRAC, LR, HID = 400, 0.2, 1e-3, 64
CONFIGS = [  # name, reward, routing, lam, beta
    ("a_R0|all", "a_R0", "all", 1.0, BETA),
    ("b_guard_off|all", "b_guard_off", "all", 1.0, BETA),
    ("c1_ttc_nc_hard_mult|all", "c1_ttc_nc_hard_mult", "all", 1.0, BETA),
    ("c2_pen_2of3|all", "c2_pen_2of3", "all", 1.0, BETA),
    ("d_lex|all", "d_lex", "all", 1.0, BETA),
    ("a_R0|progress", "a_R0", "progress", 1.0, BETA),
    ("c1_ttc_nc_hard_mult|progress", "c1_ttc_nc_hard_mult", "progress", 1.0, BETA),
    ("a_R0|all|lam0.5", "a_R0", "all", 0.5, BETA),
    ("a_R0|all|beta1.0", "a_R0", "all", 1.0, 1.0),
]


def traj_feats(sets_dir, keys, ck):
    want = {k: n for n, k in enumerate(keys)}
    out = np.full((len(keys), 64, 5), np.nan)
    import glob
    for fp in sorted(glob.glob(os.path.join(sets_dir, "onpolicy_*.jsonl"))):
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                d = json.loads(ln)
                k = (d["log_name"], d["token"], int(d["step"]), int(d["rank"]))
                if k not in want or int(d["ckpt_step"]) != ck[k]:
                    continue
                t = np.asarray(d["traj"], np.float64)                       # [64, 20, 2]
                seg = np.linalg.norm(np.diff(np.concatenate([np.zeros((64, 1, 2)), t], 1), axis=-1), axis=-1)
                out[want[k]] = np.stack([seg.sum(-1), t[:, -1, 0], t[:, -1, 1], seg[:, -1], np.abs(t[:, :, 1]).max(-1)], -1)
    if np.isnan(out).any():
        raise SystemExit("trajectory features missing for some sets")
    return out


def lane_targets(dir_, keys, ck):
    want = {k: n for n, k in enumerate(keys)}
    out = np.full((len(keys), 64), np.nan)
    import glob
    for fp in sorted(glob.glob(os.path.join(dir_, "teacher_lane_r*.jsonl"))):
        with open(fp, encoding="utf-8") as fh:
            for ln in fh:
                d = json.loads(ln)
                k = tuple(d.get("key", ()))
                k = (k[0], k[1], int(k[2]), int(k[3]))
                if k not in want or int(d.get("ckpt_step", -1)) != ck[k] or len(d.get("props") or []) != 64:
                    continue
                dist = np.asarray([np.nan if p.get("center_line.CenterLine.info") is None else float(p["center_line.CenterLine.info"])
                                   for p in d["props"]])
                out[want[k]] = np.clip(1.0 - (dist - 0.5) / 1.5, 0.0, 1.0)  # train.OnPolicyBank LANE_D0 / LANE_D1
    return out


class Adj(nn.Module):
    def __init__(self, f):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(f, HID), nn.ReLU(), nn.Linear(HID, HID), nn.ReLU(), nn.Linear(HID, 7))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)              # step 0 == the base scorer exactly

    def forward(self, x, s0):
        return s0 + self.net(x)


def sel(p):
    return p[..., 0] * p[..., 1] * (5.0 * p[..., 2] + 5.0 * p[..., 3] + 2.0 * p[..., 4]) / 12.0 * p[..., 6]


def logpi_of(s, route):
    p = torch.sigmoid(s)
    if route == "progress":
        p = torch.cat([p[..., :2].detach(), p[..., 2:3], p[..., 3:4].detach(), p[..., 4:]], -1)
    return torch.log_softmax(torch.log(sel(p) + EPS) / T_POL, dim=1)


def train_one(x, s0, tg6, lane, lmask, R, route, lam, beta, seed):
    torch.manual_seed(seed)
    m = Adj(x.shape[-1]).double()
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    with torch.no_grad():
        lp0 = logpi_of(s0, "all")
    for t in range(STEPS):
        s = m(x, s0)
        loss = F.binary_cross_entropy_with_logits(s[..., :6], tg6)
        loss = loss + F.binary_cross_entropy_with_logits(s[..., 6][lmask], lane[lmask])          # --lane-weight 1.0
        if R is not None:
            lpg = lam * min(1.0, t / (RAMP_FRAC * STEPS))
            if lpg > 0:
                lp = logpi_of(s, route)
                J = (lp.exp() * R).sum(1)
                kl = (lp0.exp() * (lp0 - lp)).sum(1)
                loss = loss + lpg * (-J + beta * kl).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    torch.set_num_threads(4)
    X = os.path.join(a.dl, "data", "x")
    kv, tgv, dv, ckv, _ = AT.load_side(os.path.join(X, "pdm_val", "pdm_val"), os.path.join(X, "terms_val", "terms_val"))
    s0 = AT.load_scores_aligned(os.path.join(a.dl, "r0", "final", "scores_step0_base.npz"), kv, ckv)
    tf = traj_feats(os.path.join(X, "sets_val", "sets_val"), kv, ckv)
    ln = lane_targets(os.path.join(X, "tlane_val", "tlane_heldout"), kv, ckv)
    sv = AT.comps(tgv)
    feats = np.concatenate([s0, tf], -1)
    fold = np.asarray([zlib.crc32(k[0].encode()) % 2 for k in kv])            # log-disjoint halves
    Rall = {c: AT.candidate(c, tgv, dv, sv) for c in {cfg[1] for cfg in CONFIGS}}
    res = {"tool": "sim_rls1b_proxy/1", "class": "ESTIMATED (proxy)", "steps": STEPS, "lr": LR, "hidden": HID, "seeds": a.seeds,
           "folds": [int((fold == f).sum()) for f in (0, 1)], "lane_labelled_sets": int(np.isfinite(ln).all(-1).sum()), "runs": {}}
    per = {}                                           # cfg -> list of per-set delta arrays (pooled over folds, seeds)
    t0 = time.time()
    for f in (0, 1):
        tr, ev = fold != f, fold == f
        mu, sd = feats[tr].reshape(-1, feats.shape[-1]).mean(0), feats[tr].reshape(-1, feats.shape[-1]).std(0) + 1e-9
        xt = torch.tensor((feats[tr] - mu) / sd)
        xe = torch.tensor((feats[ev] - mu) / sd)
        s0t, s0e = torch.tensor(s0[tr]), torch.tensor(s0[ev])
        tg6 = torch.tensor(tgv[tr][..., :6])
        lnt = torch.tensor(np.nan_to_num(ln[tr]))
        lmask = torch.tensor(np.isfinite(ln[tr]))
        ne = int(ev.sum())
        are = np.arange(ne)
        sev = {k: v[ev] for k, v in sv.items()}
        for seed in range(a.seeds):
            mc = train_one(xt, s0t, tg6, lnt, lmask, None, "all", 0.0, BETA, seed)
            with torch.no_grad():
                ic = AT.picks(mc(xe, s0e).numpy())
            for name, rw, route, lam, beta in CONFIGS:
                mr = train_one(xt, s0t, tg6, lnt, lmask, torch.tensor(Rall[rw][tr]), route, lam, beta, seed)
                with torch.no_grad():
                    sr = mr(xe, s0e).numpy()
                ir = AT.picks(sr)
                d = {q: 100.0 * (sev[q.lower()][are, ir] - sev[q.lower()][are, ic]) for q in ("TTC", "NC", "EP")}
                d["PDMS"] = 100.0 * (sev["pdms"][are, ir] - sev["pdms"][are, ic])
                per.setdefault(name, []).append((np.asarray(kv, dtype=object)[ev], d))
                res["runs"][f"{name}|fold{f}|seed{seed}"] = {q: float(v.mean()) for q, v in d.items()}
    res["seconds"] = round(time.time() - t0, 1)
    summ = {}
    for name, lst in per.items():
        keys = [tuple(k) for ks, _ in lst for k in ks]
        out = {}
        for q in ("PDMS", "EP", "TTC", "NC"):
            vals = np.concatenate([d[q] for _, d in lst])
            dd = {f"{i}": float(v) for i, v in enumerate(vals)}
            tl = {f"{i}": keys[i][0] for i in range(len(keys))}
            m_, lo, hi = AT.PE.boot(dd, tl, n=2000)
            runs = [float(d[q].mean()) for _, d in lst]
            out[q] = {"mean": m_, "ci95_logs": [lo, hi], "run_min": min(runs), "run_max": max(runs)}
        summ[name] = out
    res["pooled_vs_proxyC_x100"] = summ
    r0 = summ["a_R0|all"]
    res["face_validity"] = {"EP_up": r0["EP"]["mean"] > 0, "TTC_down": r0["TTC"]["mean"] < 0,
                            "admissible": bool(r0["EP"]["mean"] > 0 and r0["TTC"]["mean"] < 0),
                            "measured_R0_ratio_TTC_over_EP": -1.00 / 3.47, "proxy_ratio": r0["TTC"]["mean"] / r0["EP"]["mean"] if r0["EP"]["mean"] else None}
    ok = [n for n, v in summ.items() if v["TTC"]["mean"] >= -0.5]
    res["ranking"] = {"passes_TTC_screen": sorted(ok, key=lambda n: -summ[n]["PDMS"]["mean"]),
                      "note": "ranking is admissible only if face_validity.admissible; pooled over 2 folds x seeds, CI = log bootstrap (2,000) over the "
                              "pooled per-set deltas, which ignores proxy-seed variance -- run_min/run_max show it"}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("ZZPROXY_OK", a.out, res["seconds"], "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
