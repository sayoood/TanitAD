#!/usr/bin/env python3
"""RL-1 step 1: where does RL's TTC loss live?  Banked RL-val live dumps, CPU numpy.  Pre-registered in out/raw/PREREG_RL1_A_MECHANISM.md (sha256 beside it).
    python mechanism.py --arm r0 [--dl C:/Users/Admin/rls1b_dl] --out out/raw/mechanism_r0.json
"""
import argparse, json, os
import numpy as np
import rlh
from rlh import HNAME, sigm, pick

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="r0")
ap.add_argument("--dl", default=rlh.DL)
ap.add_argument("--states", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--controls", action="store_true")
a = ap.parse_args()
DEFAULT = {"r0": (250, 500, 750, 1000), "r0b": (250, 500), "r0d": (250, 500, 750, 1000)}
STATES = tuple(int(x) for x in a.states.split(",")) if a.states else DEFAULT[a.arm]
HERE = os.path.dirname(os.path.abspath(__file__))

kv, ck, s, logs, st = rlh.load_labels()
N = len(kv)
ar = np.arange(N)
BT = rlh.Boot(logs)
base0 = rlh.load_dump(f"{a.dl}/{a.arm}/final/scores_step0_base.npz", kv, ck)
ttc_fail, nc_fail = s["ttc_fail"], s["nc_fail"]
GRP = {"safety": (0, 1, 3), "progress": (2, 4), "other": (5, 6)}


def sub_ci(vec, mask):
    """mean of vec over the masked sets, paired log-cluster bootstrap (ratio estimator)"""
    if mask.sum() == 0:
        return None
    return list(BT.ci_ratio(np.where(mask, vec, 0.0), mask.astype(np.float64)))


def drift(LB, LR):
    out = {}
    for h in range(7):
        d = LB[..., h] - LR[..., h]
        dc = d - d.mean(-1, keepdims=True)
        rc = LR[..., h] - LR[..., h].mean(-1, keepdims=True)
        rms_d, rms_r = float(np.sqrt((dc ** 2).mean())), float(np.sqrt((rc ** 2).mean()))
        out[HNAME[h]] = {"mean_logit_drift": float(d.mean()), "rms_within_scene_drift": rms_d,
                         "rel_within_scene_drift": rms_d / rms_r if rms_r > 0 else None,
                         "mean_abs_prob_drift": float(np.abs(sigm(LB[..., h]) - sigm(LR[..., h])).mean())}
    return out


def auc_block(LB, LR):
    out = {}
    pos = {"NC": ~nc_fail, "DAC": s["dac"] >= 1.0 - 1e-6, "TTC": ~ttc_fail, "EP>=0.99": s["ep"] >= 0.99}
    col = {"NC": 0, "DAC": 1, "TTC": 3, "EP>=0.99": 2}
    for h, p in pos.items():
        nB, dB = rlh.auc_parts(LB[..., col[h]], p)
        nR, dR = rlh.auc_parts(LR[..., col[h]], p)
        out[h] = {"auc_B": float(nB.sum() / dB.sum()), "auc_ref": float(nR.sum() / dR.sum()),
                  "delta_B_minus_ref": list(BT.ci_ratio_diff(nB, dB, nR, dR))}
    return out


def flip_block(LB, LR, fail):
    iB, iR = pick(LB), pick(LR)
    fB, fR = fail[ar, iB], fail[ar, iR]
    W, Z = fB & ~fR, ~fB & fR
    res = {"n_changed": int((iB != iR).sum()), "n_B_fails_ref_passes": int(W.sum()), "n_ref_fails_B_passes": int(Z.sum()),
           "net_loss_sets": int(W.sum() - Z.sum())}
    return res, iB, iR, W, Z


def m3_block(LB, LR, iB, iR, W, head):
    pB, pR, p0 = sigm(LB[ar, iB, head]), sigm(LR[ar, iB, head]), sigm(base0[ar, iB, head])
    out = {"n": int(W.sum()), "head": HNAME[head],
           "p_B_at_Bpick": sub_ci(pB, W), "p_ref_at_Bpick": sub_ci(pR, W), "p_base0_at_Bpick": sub_ci(p0, W),
           "B_minus_ref": sub_ci(pB - pR, W), "B_minus_base0": sub_ci(pB - p0, W)}
    for k in ("B_minus_ref", "B_minus_base0"):
        c = out[k]
        out[k + "_label"] = None if c is None else ("OVERCONFIDENT" if c[1] > 0 else ("NOT-OVERCONFIDENT" if c[2] < 0 else "UNRESOLVED"))

    def pe(L, i):
        return sigm(L[ar, i, 2])
    out["EP_B_minus_ref_at_Bpick"] = sub_ci(pe(LB, iB) - pe(LR, iB), W)
    out["EP_B_minus_ref_at_refpick"] = sub_ci(pe(LB, iR) - pe(LR, iR), W)

    def rk(L, i):
        return (L[ar, :, head] < L[ar, i, head][:, None]).mean(1)
    out["head_rank_pct_at_Bpick_B_minus_ref"] = sub_ci(rk(LB, iB) - rk(LR, iB), W)
    return out


def m4_block(LB, LR, iB, iR, mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return {"n": 0}
    phi, mB, mR = rlh.shapley_pairs(LB, LR, iB, iR, idx)
    tot = mB - mR
    eff = float(np.abs(phi.sum(1) - tot).max())
    mB_d, mR_d = rlh.margin_direct(LB, iB, iR, idx), rlh.margin_direct(LR, iB, iR, idx)
    endpoints = float(max(np.abs(mB - mB_d).max(), np.abs(mR - mR_d).max()))
    full = np.zeros((N, 7))
    full[idx] = phi
    den = np.zeros(N)
    den[idx] = tot
    out = {"n": int(len(idx)), "efficiency_max_abs": eff, "endpoint_max_abs": endpoints,
           "mean_m_ref": float(mR.mean()), "mean_m_B": float(mB.mean()), "mean_margin_change": float(tot.mean()), "min_margin_change": float(tot.min())}
    for g, cols in GRP.items():
        out[f"share_{g}"] = list(BT.ci_ratio(full[:, list(cols)].sum(1), den))
    out["per_head_share"] = {HNAME[h]: list(BT.ci_ratio(full[:, h], den)) for h in range(7)}
    sh = out["share_safety"]
    out["label"] = "MECH-SAFETY" if sh[1] > 0.5 else ("MECH-PROGRESS" if sh[2] < 0.5 else "MECH-MIXED")
    return out


def state_block(LB, LA):
    blk = {}
    for rname, LR in (("own_A", LA), ("base0", base0)):
        r = {"drift": drift(LB, LR), "auc": auc_block(LB, LR)}
        for fname, fail, head in (("ttc", ttc_fail, 3), ("nc", nc_fail, 0)):
            f, iB, iR, W, Z = flip_block(LB, LR, fail)
            r[f"flip_{fname}"] = f
            r[f"M3_{fname}_head_on_W"] = m3_block(LB, LR, iB, iR, W, head)
            r[f"M4_{fname}_W"] = m4_block(LB, LR, iB, iR, W)
            r[f"M4_{fname}_Z"] = m4_block(LB, LR, iB, iR, Z)
        r["M4_all_changed"] = m4_block(LB, LR, pick(LB), pick(LR), pick(LB) != pick(LR))
        blk[rname] = r
    return blk


res = {"tool": "mechanism/1", "arm": a.arm, "states": {},
       "prereg_sha256": open(os.path.join(HERE, "out/raw/PREREG_RL1_A_MECHANISM.sha256")).read().split()[0]}
for u in STATES:
    LA = rlh.load_dump(f"{a.dl}/{a.arm}/final/scores_update{u}_A.npz", kv, ck)
    LB = rlh.load_dump(f"{a.dl}/{a.arm}/final/scores_update{u}_B.npz", kv, ck)
    res["states"][f"u{u}"] = state_block(LB, LA)
    print("done", a.arm, u, flush=True)

if a.controls:
    C = {}
    u = STATES[-1]
    LA = rlh.load_dump(f"{a.dl}/{a.arm}/final/scores_update{u}_A.npz", kv, ck)
    LB = rlh.load_dump(f"{a.dl}/{a.arm}/final/scores_update{u}_B.npz", kv, ck)
    z = state_block(LA, LA)["own_A"]
    C["zero_control"] = bool(all(v["mean_logit_drift"] == 0.0 and v["rms_within_scene_drift"] == 0.0 for v in z["drift"].values())
                             and z["flip_ttc"]["n_B_fails_ref_passes"] == 0 and z["flip_ttc"]["n_changed"] == 0 and z["M4_all_changed"]["n"] == 0
                             and all(v["delta_B_minus_ref"][0] == 0.0 for v in z["auc"].values()))
    effmax, endmax = 0.0, 0.0
    for stn, blk in res["states"].items():
        for rn, r in blk.items():
            for k, v in r.items():
                if k.startswith("M4_") and v.get("n", 0) > 0:
                    effmax = max(effmax, v["efficiency_max_abs"])
                    endmax = max(endmax, v["endpoint_max_abs"])
    C["shapley_efficiency_max_abs"], C["shapley_endpoint_max_abs"] = effmax, endmax
    rng = np.random.default_rng(7)
    iR = pick(LA)
    kpl = np.array([int(rng.choice(np.delete(np.arange(64), iR[j]))) for j in range(N)])
    LBp = LA.copy()
    LBp[ar, kpl, 2] += 40.0                                                      # only EP raised on one proposal per set
    iB = pick(LBp)
    o = m4_block(LBp, LA, iB, iR, iB != iR)
    C["planted_EP"] = {"n": o["n"], "share_progress": o["share_progress"][0], "share_safety": o["share_safety"][0],
                       "ok": bool(o["n"] > 100 and abs(o["share_progress"][0] - 1) < 1e-9 and abs(o["share_safety"][0]) < 1e-9)}
    LBq = LA.copy()
    LBq[ar, iR, 3] -= 40.0                                                       # only TTC lowered on ref's pick
    iB2 = pick(LBq)
    o2 = m4_block(LBq, LA, iB2, iR, iB2 != iR)
    C["planted_TTC"] = {"n": o2["n"], "share_progress": o2["share_progress"][0], "share_safety": o2["share_safety"][0],
                        "ok": bool(o2["n"] > 100 and abs(o2["share_safety"][0] - 1) < 1e-9 and abs(o2["share_progress"][0]) < 1e-9)}
    if a.arm == "r0" and 1000 in STATES:
        c0 = rlh.load_dump(f"{a.dl}/c0/final/scores_update1000_A.npz", kv, ck)
        ic, ir = pick(c0), pick(LB)
        ch = ic != ir
        C["banked_counts"] = {"changed": int(ch.sum()), "ttc_worse": int((ch & ttc_fail[ar, ir] & ~ttc_fail[ar, ic]).sum()),
                              "ttc_better": int((ch & ~ttc_fail[ar, ir] & ttc_fail[ar, ic]).sum()), "nc_worse": int((ch & nc_fail[ar, ir] & ~nc_fail[ar, ic]).sum()),
                              "expect": [2329, 59, 29, 14]}
        C["banked_pdms_x100"] = {"C0A": float(100 * s["pdms"][ar, ic].mean()), "R0B": float(100 * s["pdms"][ar, ir].mean()),
                                 "expect": [90.05977159722221, 91.22299347222223]}
        nB, dB = rlh.auc_parts(LB[..., 0], ~nc_fail)
        nC, dC = rlh.auc_parts(c0[..., 0], ~nc_fail)
        C["banked_auc_NC"] = {"R0B_u1000": float(nB.sum() / dB.sum()), "C0A_u1000": float(nC.sum() / dC.sum()), "expect": [0.9186, 0.9271]}
    C["boot_identity_max_abs"] = rlh.ctl_boot(kv, logs, 100 * (s["ttc"][ar, pick(LB)] - s["ttc"][ar, pick(LA)]))[2]
    res["controls"] = C

json.dump(res, open(a.out, "w"), indent=1)
print("wrote", a.out)
