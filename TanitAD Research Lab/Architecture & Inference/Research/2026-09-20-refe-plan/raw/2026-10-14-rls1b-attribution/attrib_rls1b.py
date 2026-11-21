#!/usr/bin/env python3
"""RL-S1b attribution (zero training, CPU): is R0's TTC loss the REWARD's preference, or the SCORER's generalisation?

R0 (BCE + EPO, reward PDMS - 0.15 D + g2) was KILLED by K3/g5: TTC -1.00 vs C0 on RL-val while PDMS +1.16 (raw/2026-10-14-rls1-r0-kill).
This script reads ONLY banked artifacts (the HF private repos Sayood/tanitad-refe-rls1{,-c0,-r0}; sha256 checked against the kill package and the
data manifest) and answers four questions:

  Q1 (RL-val, the changed picks)   For every set whose pick changed C0 -> R0: is R0's pick TTC/NC-failing, did it gain EP, and does R0's OWN
                                    reward rank R0's pick above C0's pick?  (reward-preferred change vs reward-dispreferred change)
  Q2 (RL-val, the heads)            Did EPO corrupt the SAFETY heads?  Within-scene AUC of the NC / DAC / TTC logits for base, C0-A, R0-A
                                    (the same-pass BCE-only arm of the R0 run) and R0-B, at u250 / u500 / u1000; and, at R0's TTC-failing picks,
                                    the TTC probability R0-B assigns vs the one C0-A assigns to the SAME proposal.
  Q3 (RL-TRAIN labels, the reward)  For each candidate reward: reward-optimal (argmax) and Boltzmann(R / tau) mass on TTC- / NC-failing proposals,
                                    and the share of the policy-gradient push pi_i (R_i - J) that lands on them under the oracle policy
                                    softmax(log(PDMS + eps) / T) (label-only; the scorer's RL-train scores were never dumped).
  Q4 (RL-val, the gradient routing) The step-0 EPO gradient on each head's logits under the REAL base policy (the base scorer's RL-val scores,
                                    rule navsim_v1_lane, T 0.1, eps 1e-6 = scorer_finetune.epo_loss), per candidate reward and per routing
                                    ('all' heads = R0; 'progress' = NC / DAC / TTC detached in the policy): the share of each head's push that is
                                    ANTI-ALIGNED with its own BCE label (raises a failing proposal's safety logit or lowers a passing one's).

Tiers: RL-val is a SELECTION tier (diagnosis / hyper-parameter selection only, never an endpoint); RL-train numbers are label-only reward
properties.  Estimator for every RL-val delta: paired log-cluster bootstrap (proxy_eval.boot, 10,000, seed 20260927): 'another draw of LOGS'
only -- one training seed per arm, training and inference variance NOT measured (H-ESTIM-SEED-1).

    python attrib_rls1b.py --dl <download root> --out attribution.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PK = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(PK, "refe"))
sys.path.insert(0, os.path.join(PK, "eval"))
import proxy_eval as PE  # noqa: E402
import rl_reward as RR  # noqa: E402
import rl_val_eval as RV  # noqa: E402

TOL = RR.SAFE_TOL
T_POL, EPS = 0.1, 1e-6                         # scorer_finetune.py --epo-temp default, EPO_EPS
EXPECT = {"c0/final/scores_update1000_A.npz": "5a524381fa8075dd5d74ca390525dada829935770403d2f7fb8ea431cebdb9f5",   # rlval_vs_C0.json inputs
          "r0/final/scores_update1000_B.npz": "65dcd60e12744717ec535cbbbc07d4aa6ef4f52a8be1438baf98c1ac232aea46",
          "data/data/rlfan_pdm_train.tar.gz": "4b63f8880a76c60058d7956d00d3755f51bf5d339f377d640df96241955644c6",       # MANIFEST_rlfan_sha256.txt
          "data/data/rlfan_pdm_val.tar.gz": "3dd6cba33cb1395f6c18b0204299a235384a160ea1456395f132de82fefbaae7",
          "data/data/rlfan_terms_train.tar.gz": "efde6f43cd5afc8cfdd887e84a15d6c8057f86ee0574a95c9f78e4e13831651f",
          "data/data/rlfan_terms_val.tar.gz": "6f12b90fa14abc86effa78643d78a72aff8b1145ec6bd66d84290b9d2fa3fce7"}
TAGS = ("step0", "update250", "update500", "update1000")
HEADS = {"NC": 0, "DAC": 1, "EP": 2, "TTC": 3, "C": 4}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------ data
def load_side(pdm_dir, terms_dir):
    pdm, st_p = RV.load_pdm(pdm_dir)
    terms, st_t = RV.load_terms(terms_dir)
    keys = sorted(k for k in pdm if k in terms and terms[k]["ckpt"] == pdm[k]["ckpt"])
    tg = np.stack([pdm[k]["tg"] for k in keys])
    d = np.stack([terms[k]["d"] for k in keys])
    ck = {k: pdm[k]["ckpt"] for k in keys}
    return keys, tg, d, ck, {"pdm": st_p, "terms": st_t, "joined": len(keys)}


def load_scores_aligned(path, keys, ck):
    """RV.load_scores' semantics (key_of, duplicate = error, logit space), but each array is read ONCE (RV.load_scores re-reads the whole
    [N, M, C] array per row through NpzFile.__getitem__, 3,000 x 10.75 MB of churn)."""
    z = np.load(path, allow_pickle=False)
    if ("space" in z.files and str(z["space"]) != "logit"):
        raise SystemExit(f"{path}: not logit space")
    lg, tk, st, rk, cs, sc = z["log"], z["token"], z["step"], z["rank"], z["ckpt_step"], z["scores"]
    idx = {}
    for i in range(len(tk)):
        k = RV.key_of((lg[i], tk[i], st[i], rk[i]))
        if k in idx:
            raise SystemExit(f"{path}: duplicate key {k}")
        idx[k] = i
    miss = [k for k in keys if k not in idx or int(cs[idx[k]]) != ck[k]]
    if miss:
        raise SystemExit(f"{path}: {len(miss)} sets missing or stale")
    return np.asarray(sc[[idx[k] for k in keys]], np.float64)


# ------------------------------------------------------------------------------------------------------------------ rewards
def comps(tg):
    s = RR.split_components(tg)
    s["pdms"] = RR.pdms_v1(s["nc"], s["dac"], s["ep"], s["ttc"], s["c"])
    s["safe"] = RR.safe_mask(s["nc"], s["dac"], s["ttc"])
    s["ttc_fail"] = s["ttc"] < 1.0 - TOL
    s["nc_fail"] = s["nc"] < 1.0 - TOL
    s["tn_fail"] = s["ttc_fail"] | s["nc_fail"]
    return s


def guarded(base, s, w_prog=0.5):
    a = RR.progress_shortfall(s["ep"], s["safe"])
    return base + RR.guard_penalty(base, w_prog * a, s["safe"])


def candidate(name, tg, d, s):
    """The candidate rewards of the brief (a)-(d).  All share PDMS and -0.15 D; each differs from R0 in ONE stated way."""
    lam_d = 0.15
    base = s["pdms"] - lam_d * d
    if name == "a_R0":
        return RR.reward(tg, RR.RewardCfg(), d=d)["R"]
    if name == "b_guard_off":
        return RR.reward(tg, RR.RewardCfg(guard=False), d=d)["R"]
    if name == "c1_ttc_nc_hard_mult":                 # PDMS x 1[TTC = 1] x 1[NC = 1]: a TTC/NC failure zeroes the PDMS part, then g2 as in R0
        return guarded(s["pdms"] * (~s["ttc_fail"]) * (~s["nc_fail"]) - lam_d * d, s)
    if name.startswith("c2_pen_"):                    # R0 - lam_s ((1 - TTC) + (1 - NC))
        lam_s = {"c2_pen_5of12": 5.0 / 12.0, "c2_pen_2of3": 2.0 / 3.0}[name]
        return RR.reward(tg, RR.RewardCfg(), d=d)["R"] - lam_s * ((1.0 - s["ttc"]) + (1.0 - s["nc"]))
    if name in ("d_lex", "d_lex_guard_off"):          # every unsafe (NC/DAC/TTC < 1) proposal strictly below every safe one, per scene, margin 0.05
        r = RR.reward(tg, RR.RewardCfg(guard=(name == "d_lex")), d=d)["R"]
        return lexify(r, s["safe"])
    raise ValueError(name)


def lexify(r, safe, margin=0.05):
    mu = np.where(~safe, r, -np.inf).max(-1, keepdims=True)
    ms = np.where(safe, r, np.inf).min(-1, keepdims=True)
    shift = np.where(np.isfinite(mu) & np.isfinite(ms), np.maximum(0.0, mu - ms + margin), 0.0)
    return np.where(safe, r, r - shift)


CANDS = ("a_R0", "b_guard_off", "c1_ttc_nc_hard_mult", "c2_pen_5of12", "c2_pen_2of3", "d_lex", "d_lex_guard_off")


def softmax(x, axis=-1):
    z = x - x.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


def reward_profile(R, s, logpi):
    """label-only properties of one reward on a set of scenes.  logpi [N, M] = the policy whose gradient we read (oracle or base)."""
    n = R.shape[0]
    i = R.argmax(-1)
    ar = np.arange(n)
    out = {"argmax": {"PDMS": float(s["pdms"][ar, i].mean()), "EP": float(s["ep"][ar, i].mean()), "TTC_fail_share": float(s["ttc_fail"][ar, i].mean()),
                      "NC_fail_share": float(s["nc_fail"][ar, i].mean()), "unsafe_share": float((~s["safe"])[ar, i].mean())}}
    for tau in (0.02, 0.1):
        p = softmax(R / tau)
        out[f"boltzmann_tau{tau}"] = {"mass_TTC_fail": float((p * s["ttc_fail"]).sum(-1).mean()), "mass_NC_fail": float((p * s["nc_fail"]).sum(-1).mean()),
                                      "E_PDMS": float((p * s["pdms"]).sum(-1).mean()), "E_EP": float((p * s["ep"]).sum(-1).mean())}
    pi = np.exp(logpi)
    J = (pi * R).sum(-1, keepdims=True)
    adv = pi * (R - J)                                # = dJ / dz_i, z = the policy's logit (log(A + eps) / T)
    pos = np.maximum(adv, 0.0)
    tot = pos.sum()
    out["pg_push"] = {"pos_share_on_TTC_fail": float((pos * s["ttc_fail"]).sum() / tot), "pos_share_on_NC_fail": float((pos * s["nc_fail"]).sum() / tot),
                      "pos_share_on_unsafe": float((pos * ~s["safe"]).sum() / tot),
                      "net_on_TTC_fail_per_scene": float((adv * s["ttc_fail"]).sum() / n), "net_on_unsafe_per_scene": float((adv * ~s["safe"]).sum() / n),
                      "pos_mass_per_scene": float(tot / n),
                      "frac_TTC_fail_proposals_pushed_UP": float(((adv > 0) & s["ttc_fail"]).sum() / max(1, s["ttc_fail"].sum())),
                      "frac_NC_fail_proposals_pushed_UP": float(((adv > 0) & s["nc_fail"]).sum() / max(1, s["nc_fail"].sum()))}
    return out


# ------------------------------------------------------------------------------------------------------------------ selection / heads
def sel_v1_lane(p):
    return p[..., 0] * p[..., 1] * (5.0 * p[..., 2] + 5.0 * p[..., 3] + 2.0 * p[..., 4]) / 12.0 * p[..., 6]


def picks(logits):
    p = RV.sigmoid(logits)
    return sel_v1_lane(p).argmax(-1)                  # ties -> lowest index (RV.select's rule)


def within_auc(score, pos):
    """pooled within-scene AUC: P(score of a label-1 proposal > score of a label-0 proposal of the SAME scene), ties 0.5; scenes with both classes."""
    num = den = 0.0
    nsc = 0
    for x, y in zip(score, pos):
        a, b = x[y], x[~y]
        if a.size == 0 or b.size == 0:
            continue
        nsc += 1
        diff = a[:, None] - b[None, :]
        num += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        den += diff.size
    return {"auc": float(num / den) if den else None, "scenes": nsc, "pairs": int(den)}


def epo_grad(logits, R, route):
    """d(-J)/d(logits) at pi = pi_SFT (KL gradient is exactly 0 there), J = sum_i pi_i R_i, pi = softmax(log(sel(sigmoid(s)) + eps) / T).
    route 'all' = scorer_finetune.epo_loss; 'progress' = the NC / DAC / TTC probabilities detached inside the policy."""
    import torch
    s = torch.tensor(logits, dtype=torch.float64, requires_grad=True)
    p = torch.sigmoid(s)
    if route == "progress":
        p = torch.cat([p[..., :2].detach(), p[..., 2:3], p[..., 3:4].detach(), p[..., 4:]], -1)
    a = p[..., 0] * p[..., 1] * (5.0 * p[..., 2] + 5.0 * p[..., 3] + 2.0 * p[..., 4]) / 12.0 * p[..., 6]
    logpi = torch.log_softmax(torch.log(a + EPS) / T_POL, dim=1)
    J = (logpi.exp() * torch.as_tensor(R)).sum(1)
    (-J.mean()).backward()
    return s.grad.numpy()


def head_alignment(grad, logits, s):
    """per head: share of the EPO push (-grad) that is ANTI-aligned with the head's own BCE target (y - p)."""
    p = RV.sigmoid(logits)
    push = -grad
    lab = {"NC": s["nc"], "DAC": s["dac"], "EP": s["ep"], "TTC": s["ttc"], "C": s["c"]}
    out = {}
    for h, j in HEADS.items():
        v, e = push[..., j], lab[h] - p[..., j]
        m = np.abs(v).sum()
        out[h] = {"abs_push": float(m), "anti_aligned_share": float(np.abs(v)[v * e < 0].sum() / m) if m > 0 else 0.0}
    v = push[..., 3]
    passing = ~s["ttc_fail"]
    out["TTC_push_on_passing_DOWN_share"] = float(np.abs(v)[passing & (v < 0)].sum() / max(np.abs(v).sum(), 1e-300))
    out["lane_abs_push"] = float(np.abs(push[..., 6]).sum())
    return out


def boot_delta(a, b, keys, scale=100.0):
    d = {RV.kstr(k): scale * (float(a[n]) - float(b[n])) for n, k in enumerate(keys)}
    tl = {RV.kstr(k): k[0] for k in keys}
    mu, lo, hi = PE.boot(d, tl)
    return {"mean": mu, "ci95": [lo, hi]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", required=True, help="download root holding data/ c0/ r0/ (hf_hub_download local_dir layout)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"tool": "attrib_rls1b/1", "inputs": {}}
    for rel, want in EXPECT.items():
        got = sha(os.path.join(a.dl, rel))
        res["inputs"][rel] = got
        if got != want:
            raise SystemExit(f"sha256 mismatch {rel}: {got} != {want}")
    X = os.path.join(a.dl, "data", "x")

    # ---------------------------------------------------------------- RL-val
    kv, tgv, dv, ckv, stv = load_side(os.path.join(X, "pdm_val", "pdm_val"), os.path.join(X, "terms_val", "terms_val"))
    res["rl_val_join"] = stv
    sv = comps(tgv)
    L = {}
    L["base"] = load_scores_aligned(os.path.join(a.dl, "r0", "final", "scores_step0_base.npz"), kv, ckv)
    for t in TAGS:
        L[f"C0A_{t}"] = load_scores_aligned(os.path.join(a.dl, "c0", "final", f"scores_{t}_A.npz"), kv, ckv)
        L[f"R0A_{t}"] = load_scores_aligned(os.path.join(a.dl, "r0", "final", f"scores_{t}_A.npz"), kv, ckv)
        L[f"R0B_{t}"] = load_scores_aligned(os.path.join(a.dl, "r0", "final", f"scores_{t}_B.npz"), kv, ckv)
    n = len(kv)
    ar = np.arange(n)
    R0v = RR.reward(tgv, RR.RewardCfg(), d=dv)["R"]
    PK_ = {nm: picks(x) for nm, x in L.items()}

    # selfcheck: our pick lookup reproduces rlval_vs_C0.json's arm means (navsim_v1_lane)
    res["selfcheck_vs_kill_package"] = {"C0A_u1000_PDMS_x100": float(100 * sv["pdms"][ar, PK_["C0A_update1000"]].mean()),
                                        "R0B_u1000_PDMS_x100": float(100 * sv["pdms"][ar, PK_["R0B_update1000"]].mean()),
                                        "expect": [90.05977159722221, 91.22299347222223]}

    # trajectory of the sub-score deltas (R0-B and R0-A vs C0-A) -- the same-pass arm A is a same-seed BCE-only replicate-like floor
    traj = {}
    for t in TAGS[1:]:
        row = {}
        for arm in ("R0B", "R0A"):
            ia, ic = PK_[f"{arm}_{t}"], PK_[f"C0A_{t}"]
            row[arm] = {q: boot_delta(sv[q.lower()][ar, ia], sv[q.lower()][ar, ic], kv) for q in ("TTC", "NC", "EP")}
            row[arm]["PDMS"] = boot_delta(sv["pdms"][ar, ia], sv["pdms"][ar, ic], kv)
        traj[t] = row
    res["rlval_delta_trajectory_x100"] = traj

    # Q1: the changed picks at u1000
    ic, ir = PK_["C0A_update1000"], PK_["R0B_update1000"]
    ch = ic != ir
    tw = ch & sv["ttc_fail"][ar, ir] & ~sv["ttc_fail"][ar, ic]
    tb = ch & ~sv["ttc_fail"][ar, ir] & sv["ttc_fail"][ar, ic]
    nw = ch & sv["nc_fail"][ar, ir] & ~sv["nc_fail"][ar, ic]
    rank_R = (-R0v).argsort(-1).argsort(-1)            # 0 = R0-reward-best proposal of the fan
    pR, pC = RV.sigmoid(L["R0B_update1000"]), RV.sigmoid(L["C0A_update1000"])
    pB = RV.sigmoid(L["base"])

    def block(m):
        k = int(m.sum())
        if k == 0:
            return {"n": 0}
        rr, rc = R0v[ar, ir][m], R0v[ar, ic][m]
        am = R0v.argmax(-1)
        return {"n": k,
                "reward_prefers_R0_pick_share": float((rr > rc).mean()), "reward_tie_share": float((rr == rc).mean()),
                "mean_R0reward_R0pick_minus_C0pick": float((rr - rc).mean()),
                "R0pick_R0reward_rank_median": float(np.median(rank_R[ar, ir][m])), "C0pick_R0reward_rank_median": float(np.median(rank_R[ar, ic][m])),
                "R0pick_is_R0reward_argmax_share": float((ir == am)[m].mean()),
                "R0reward_argmax_TTC_fail_share": float(sv["ttc_fail"][ar, am][m].mean()),
                "EP_R0pick_minus_C0pick": float((sv["ep"][ar, ir] - sv["ep"][ar, ic])[m].mean()),
                "PDMS_R0pick_minus_C0pick": float((sv["pdms"][ar, ir] - sv["pdms"][ar, ic])[m].mean()),
                "pTTC_by_R0B_at_R0pick": float(pR[ar, ir, 3][m].mean()), "pTTC_by_C0A_at_R0pick": float(pC[ar, ir, 3][m].mean()),
                "pTTC_by_base_at_R0pick": float(pB[ar, ir, 3][m].mean()),
                "pTTC_by_R0B_at_C0pick": float(pR[ar, ic, 3][m].mean()), "pTTC_by_C0A_at_C0pick": float(pC[ar, ic, 3][m].mean()),
                "pEP_by_R0B_at_R0pick": float(pR[ar, ir, 2][m].mean()), "pEP_by_C0A_at_R0pick": float(pC[ar, ir, 2][m].mean()),
                "selscore_C0A_prefers_C0pick_by": float((sel_v1_lane(pC)[ar, ic] - sel_v1_lane(pC)[ar, ir])[m].mean()),
                "selscore_R0B_prefers_R0pick_by": float((sel_v1_lane(pR)[ar, ir] - sel_v1_lane(pR)[ar, ic])[m].mean())}

    res["Q1_changed_picks_u1000"] = {"n_sets": n, "changed": int(ch.sum()), "all_changed": block(ch), "TTC_worse": block(tw), "TTC_better": block(tb),
                                     "NC_worse": block(nw), "safe_to_safe": block(ch & sv["safe"][ar, ir] & sv["safe"][ar, ic])}

    # Q2: within-scene head AUCs, safety heads (label 1 = pass)
    q2 = {}
    for nm in ["base"] + [f"{arm}_{t}" for t in TAGS[1:] for arm in ("C0A", "R0A", "R0B")]:
        q2[nm] = {h: within_auc(L[nm][..., HEADS[h]], ~sv[f"{h.lower()}_fail"]) for h in ("TTC", "NC")}
        q2[nm]["DAC"] = within_auc(L[nm][..., 1], sv["dac"] >= 1.0 - TOL)
        q2[nm]["EP_ge_0.99"] = within_auc(L[nm][..., 2], sv["ep"] >= 0.99)
        # the progress leak: within-scene Pearson of the TTC logit with the TRUE EP, among TTC-passing proposals
        cs = []
        for x, e, ok in zip(L[nm][..., 3], sv["ep"], ~sv["ttc_fail"]):
            if ok.sum() > 3 and np.std(x[ok]) > 0 and np.std(e[ok]) > 0:
                cs.append(np.corrcoef(x[ok], e[ok])[0, 1])
        q2[nm]["TTC_logit_vs_trueEP_corr_on_passing"] = float(np.mean(cs))
    res["Q2_within_scene_auc"] = q2

    # Q2b: paired log-cluster bootstrap of the within-scene AUC DELTA (ratio estimator: resample logs, recompute sum(num)/sum(den) per arm)
    def auc_parts(score, pos):
        num, den = np.zeros(n), np.zeros(n)
        for j, (x, y) in enumerate(zip(score, pos)):
            a_, b_ = x[y], x[~y]
            if a_.size and b_.size:
                df = a_[:, None] - b_[None, :]
                num[j], den[j] = (df > 0).sum() + 0.5 * (df == 0).sum(), df.size
        return num, den
    logs = sorted({k[0] for k in kv})
    gid = np.asarray([logs.index(k[0]) for k in kv])
    groups = [np.flatnonzero(gid == g) for g in range(len(logs))]

    def boot_auc_delta(nA, dA, nB, dB, nb=10000, seed=20260927):
        rng = np.random.default_rng(seed)
        mu = nA.sum() / dA.sum() - nB.sum() / dB.sum()
        out = np.empty(nb)
        for b in range(nb):
            ix = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
            out[b] = nA[ix].sum() / dA[ix].sum() - nB[ix].sum() / dB[ix].sum()
        return {"delta": float(mu), "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]}
    q2b = {}
    for h, col, pos in (("TTC", 3, ~sv["ttc_fail"]), ("NC", 0, ~sv["nc_fail"]), ("DAC", 1, sv["dac"] >= 1.0 - TOL)):
        nC, dC = auc_parts(L["C0A_update1000"][..., col], pos)
        for arm in ("R0B", "R0A"):
            nX, dX = auc_parts(L[f"{arm}_update1000"][..., col], pos)
            q2b[f"{h}|{arm}_minus_C0A_u1000"] = boot_auc_delta(nX, dX, nC, dC)
    res["Q2b_auc_delta_boot_u1000"] = q2b

    # Q5: zero-training HYBRID selections on RL-val (which OUTPUTS carry the TTC loss?).  Safety outputs = NC, DAC, TTC (cols 0, 1, 3);
    # progress outputs = EP, C, lane (cols 2, 4, 6); DDC (col 5) is not in the rule.  A hybrid is a DIAGNOSTIC, not a deployable arm.
    def hybrid(safety_src, prog_src):
        x = L[prog_src].copy()
        x[..., [0, 1, 3]] = L[safety_src][..., [0, 1, 3]]
        return x
    hyb = {"H1_safety_R0A_progress_R0B": hybrid("R0A_update1000", "R0B_update1000"),
           "H2_safety_C0A_progress_R0B": hybrid("C0A_update1000", "R0B_update1000"),
           "H3_TTC_only_from_R0A": None,
           "H4_safety_R0B_progress_R0A": hybrid("R0B_update1000", "R0A_update1000"),
           "R0B_itself": L["R0B_update1000"], "R0A_itself": L["R0A_update1000"]}
    x3 = L["R0B_update1000"].copy()
    x3[..., 3] = L["R0A_update1000"][..., 3]
    hyb["H3_TTC_only_from_R0A"] = x3
    q5 = {}
    for nm, x in hyb.items():
        ih = picks(x)
        q5[nm] = {q: boot_delta(sv[q.lower()][ar, ih], sv[q.lower()][ar, ic], kv) for q in ("TTC", "NC", "EP", "DAC")}
        q5[nm]["PDMS"] = boot_delta(sv["pdms"][ar, ih], sv["pdms"][ar, ic], kv)
        q5[nm]["unsafe_picks"] = int((~sv["safe"][ar, ih]).sum())
    q5["C0A_unsafe_picks"] = int((~sv["safe"][ar, ic]).sum())
    res["Q5_hybrid_vs_C0A_u1000_x100"] = q5

    # Q4: step-0 gradient routing on RL-val with the REAL base policy
    q4 = {}
    for c in CANDS:
        Rc = candidate(c, tgv, dv, sv)
        for route in ("all", "progress"):
            g = epo_grad(L["base"], Rc, route)
            q4[f"{c}|{route}"] = head_alignment(g, L["base"], sv)
    res["Q4_step0_head_alignment_rlval_base"] = q4

    # Q3 on RL-val too, with the real base policy
    pb = RV.sigmoid(L["base"])
    logpi_b = np.log(sel_v1_lane(pb) + EPS) / T_POL
    logpi_b = logpi_b - np.log(np.exp(logpi_b - logpi_b.max(-1, keepdims=True)).sum(-1, keepdims=True)) - logpi_b.max(-1, keepdims=True)
    res["Q3_reward_profile_rlval_basepolicy"] = {c: reward_profile(candidate(c, tgv, dv, sv), sv, logpi_b) for c in CANDS}
    del L

    # ---------------------------------------------------------------- RL-train (labels only)
    kt, tgt, dt, ckt, stt = load_side(os.path.join(X, "pdm_train", "pdm_train"), os.path.join(X, "terms_train", "terms_train"))
    res["rl_train_join"] = stt
    q3 = {}
    agg: dict = {c: [] for c in CANDS}
    base_rates = {"TTC_fail_share_of_proposals": [], "NC_fail_share_of_proposals": [], "scenes_with_safe": [], "scenes_with_TTC_fail": []}
    CH = 1000
    for lo in range(0, len(kt), CH):
        tg, d = tgt[lo:lo + CH], dt[lo:lo + CH]
        s = comps(tg)
        lp = np.log(s["pdms"] + EPS) / T_POL
        lp = lp - lp.max(-1, keepdims=True)
        lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
        base_rates["TTC_fail_share_of_proposals"].append(float(s["ttc_fail"].mean()))
        base_rates["NC_fail_share_of_proposals"].append(float(s["nc_fail"].mean()))
        base_rates["scenes_with_safe"].append(float(s["safe"].any(-1).mean()))
        base_rates["scenes_with_TTC_fail"].append(float(s["ttc_fail"].any(-1).mean()))
        for c in CANDS:
            agg[c].append((len(tg), reward_profile(candidate(c, tg, d, s), s, lp)))

    def wavg(lst):
        tot = sum(w for w, _ in lst)

        def rec(path):
            vals = []
            for w, r in lst:
                x = r
                for p_ in path:
                    x = x[p_]
                vals.append((w, x))
            return sum(w * x for w, x in vals) / tot
        first = lst[0][1]
        return {k: {kk: rec((k, kk)) for kk in v} for k, v in first.items()}
    for c in CANDS:
        q3[c] = wavg(agg[c])
    res["Q3_reward_profile_rltrain_oraclepolicy"] = q3
    res["rl_train_base_rates"] = {k: float(np.mean(v)) for k, v in base_rates.items()}
    res["notes"] = ["pg_push.* is a SHARE of the positive part of pi_i (R_i - J): scale-free across rewards; net_* is per scene in reward units",
                    "RL-train policy = oracle softmax(log(PDMS + eps)/0.1) (label-only); RL-val policy = the REAL base scorer (navsim_v1_lane)",
                    "R0A = arm A (BCE only) of the R0 pass: same flags and seed as C0-A, differs only by RNG consumption -- a replicate-LIKE floor, NOT a seed floor"]
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("ZZATTRIB_OK", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
