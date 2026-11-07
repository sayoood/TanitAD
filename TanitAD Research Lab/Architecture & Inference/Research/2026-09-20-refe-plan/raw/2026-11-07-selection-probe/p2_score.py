#!/usr/bin/env python3
"""P2 -- the SINGLE held-out read of PREREG_SEL.md's primary rule gBG(0.20, 0.05). CPU only.

Refuses to run unless PREREG_SEL.md hashes to PREREG_SEL.sha256. Gates G0 / G0b / G0c (with its RED arm) run first,
on picks + logits only. Then the held-out verdict (mechanical, section 2 of the prereg), then the descriptive families.
Writes p2_result.json.
"""
import hashlib
import json
import sys

import numpy as np

import sel_common as sc
import sel_rules as sr

DELTA, EPS = 0.20, 0.05
W = {"zero": 529 / 12146, "random": 11617 / 12146}

# ---- prereg binding
pre = open(sc.OUT + "/PREREG_SEL.md", "rb").read()
want = open(sc.OUT + "/PREREG_SEL.sha256").read().strip()
got = hashlib.sha256(pre).hexdigest()
if got != want:
    sys.exit(f"REFUSED: PREREG_SEL.md sha256 {got} != frozen {want}")

D = sc.load()
N = len(D["token"])
r = np.arange(N)
L = D["logits"]
v0, ax0, v4 = D["v0"], D["ax0"], D["v4_all"]
s = sr.speed_ratio(v4, v0)
GATE = (v0 > 5.0) & (np.abs(ax0) < 0.3)
P = sr.probs(L)
agg = sc.agg_np(L, np.float64)


def rule_vec(delta, eps, flip_guard=False, gated=True):
    base = sc.pick_from_logits(L, "torch")
    if delta == 0:
        return base
    band = agg >= (1.0 - delta) * agg[r, base][:, None]
    pb = P[r, base][:, None, :]
    if flip_guard:
        guard = np.all(P[..., [0, 1, 3]] >= pb[..., [0, 1, 3]] + eps, axis=-1)
    else:
        guard = np.all(P[..., [0, 1, 3]] >= pb[..., [0, 1, 3]] - eps, axis=-1)
    adm = band & guard
    pk = np.where(adm.any(-1), np.argmax(np.where(adm, s + 1e-6 * agg, -1.0), -1), base)
    return np.where(GATE, pk, base) if gated else pk


def rule_loop(i, delta, eps):
    """Section 1 of PREREG_SEL.md, implemented literally for ONE token."""
    b = int(np.argmax(agg[i]))
    if not (v0[i] > 5.0 and abs(ax0[i]) < 0.3):
        return b
    best, bestkey = b, -np.inf
    for k in range(agg.shape[1]):
        if not agg[i, k] >= (1 - delta) * agg[i, b]:
            continue
        if any(P[i, k, j] < P[i, b, j] - eps for j in (0, 1, 3)):
            continue
        key = s[i, k] + 1e-6 * agg[i, k]
        if key > bestkey:
            best, bestkey = k, key
    return best


gates = {}
gates["G0_pick_repro"] = int((sc.pick_from_logits(L, "torch") == D["pick"]).sum())
gates["G0b_delta0_repro"] = int((rule_vec(0.0, EPS) == D["pick"]).sum())
loop = np.array([rule_loop(i, DELTA, EPS) for i in range(N)])
vec = rule_vec(DELTA, EPS)
red = rule_vec(DELTA, EPS, flip_guard=True)
gates["G0c_loop_eq_vec"] = int((loop == vec).sum())
gates["G0c_RED_flipguard_disagree"] = int((loop != red).sum())
gates["G0c_vec_eq_sel_rules_pick_BG_gated"] = int((np.where(GATE, sr.pick_BG(L, s, DELTA, EPS), D["pick"]) == vec).sum())
gates["n"] = N
gates["PASS"] = bool(gates["G0_pick_repro"] == N and gates["G0b_delta0_repro"] == N and gates["G0c_loop_eq_vec"] == N
                     and gates["G0c_RED_flipguard_disagree"] > 0 and gates["G0c_vec_eq_sel_rules_pick_BG_gated"] == N)
print("gates", gates)
if not gates["PASS"]:
    json.dump({"gates": gates}, open(sc.OUT + "/p2_result.json", "w"), indent=1)
    sys.exit("INSTRUMENT FAILURE: gates not green -- no verdict")

# ---- held-out population
split = json.load(open(sc.OUT + "/split.json"))
HO = np.isin(D["log"], split["heldout_logs"])
assert not np.isin(D["log"][HO], split["train_logs"]).any()
hidx = np.where(HO)[0]
hl = D["log"][hidx]
ul, li = np.unique(hl, return_inverse=True)
nl = len(ul)
draws = np.random.default_rng(20260927).choice(nl, size=(10000, nl), replace=True)
strat = D["stratum"][hidx]
cru = D["cruise"][hidx]
gv = D["gt_valid"][hidx]
pdms, sub = D["pdms"][hidx], D["sub"][hidx]
hr = np.arange(len(hidx))
base_h = D["pick"][hidx]


def per_draw_mean(vals, mask):
    vals = np.asarray(vals, np.float64)
    m = np.asarray(mask, bool) & np.isfinite(vals)
    sm = np.bincount(li[m], weights=vals[m], minlength=nl)
    ct = np.bincount(li[m], minlength=nl).astype(np.float64)
    S, C = sm[draws].sum(1), ct[draws].sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return S / C, sm.sum() / max(ct.sum(), 1e-12)


def ci(arr, pt):
    a = arr[np.isfinite(arr)]
    return {"point": float(pt), "lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5)), "n_draws_finite": int(len(a))}


def mix_stat(vals):
    dz, pz = per_draw_mean(vals, strat == "zero")
    dr, pr = per_draw_mean(vals, strat == "random")
    return ci(100 * (W["zero"] * dz + W["random"] * dr), 100 * (W["zero"] * pz + W["random"] * pr))


def mean_stat(vals, mask, scale=1.0):
    d, p = per_draw_mean(vals, mask)
    return ci(scale * d, scale * p)


def arm_summary(pk_full):
    pk = pk_full[hidx]
    pd = pdms[hr, pk]
    v4p = v4[hidx][hr, pk]
    ratio = v4p / np.maximum(v0[hidx], 1e-6)
    proposals = D["proposals"][hidx]
    o = {"pdms_mix_x100": mix_stat(pd), "pdms_unweighted_x100": mean_stat(pd, np.ones(len(hidx), bool), 100),
         "pdms_random_x100": mean_stat(pd, strat == "random", 100), "pdms_zero_x100": mean_stat(pd, strat == "zero", 100),
         "slow_share_cruise": mean_stat((ratio < 0.85).astype(float), cru),
         "overspeed_share_cruise": mean_stat((ratio > 1.15).astype(float), cru),
         "v4_over_v0_cruise": mean_stat(ratio, cru),
         "bias_v4_minus_gt_cruise": mean_stat(v4p - D["v4_gt"][hidx], cru & gv),
         "bias_v4_minus_gt_gate": mean_stat(v4p - D["v4_gt"][hidx], GATE[hidx] & gv),
         "bias_v4_minus_gt_all": mean_stat(v4p - D["v4_gt"][hidx], gv),
         "nc_fail_rate": mean_stat((sub[hr, pk, 0] < 1).astype(float), np.ones(len(hidx), bool)),
         "dac_fail_rate": mean_stat((sub[hr, pk, 1] < 1).astype(float), np.ones(len(hidx), bool)),
         "ttc_fail_rate": mean_stat((sub[hr, pk, 3] < 1).astype(float), np.ones(len(hidx), bool)),
         "ddc_mean": mean_stat(sub[hr, pk, 5], np.ones(len(hidx), bool)),
         "pick_changed_share": float((pk != base_h).mean()),
         "abs_y4_m_cruise": mean_stat(np.abs(proposals[hr, pk, -1, 1]), cru),
         "abs_heading4_rad_cruise": mean_stat(np.abs(proposals[hr, pk, -1, 2]), cru)}
    return o, pk, pd, ratio


def paired(pk_new_full):
    o_new, pk_new, pd_new, ratio_new = arm_summary(pk_new_full)
    pd_b = pdms[hr, base_h]
    ratio_b = v4[hidx][hr, base_h] / np.maximum(v0[hidx], 1e-6)
    v4n = v4[hidx][hr, pk_new]
    v4b = v4[hidx][hr, base_h]
    prop = D["proposals"][hidx]
    Z = np.load(sc.EGO + "/full/results.npz")
    zt = {str(t): i for i, t in enumerate(Z["token"])}
    h8 = Z["human8"][[zt[t] for t in D["token"][hidx]]].astype(np.float64)
    ade = lambda pk: np.linalg.norm(prop[hr, pk, :, :2] - h8[..., :2], axis=-1).mean(-1)
    d = {"d_pdms_mix_x100": mix_stat(pd_new - pd_b),
         "d_pdms_unweighted_x100": mean_stat(pd_new - pd_b, np.ones(len(hidx), bool), 100),
         "d_pdms_random_x100": mean_stat(pd_new - pd_b, strat == "random", 100),
         "d_pdms_zero_x100": mean_stat(pd_new - pd_b, strat == "zero", 100),
         "d_slow_share_cruise": mean_stat((ratio_new < 0.85).astype(float) - (ratio_b < 0.85).astype(float), cru),
         "d_overspeed_share_cruise": mean_stat((ratio_new > 1.15).astype(float) - (ratio_b > 1.15).astype(float), cru),
         "d_bias_cruise": mean_stat(v4n - v4b, cru & gv), "d_bias_gate": mean_stat(v4n - v4b, GATE[hidx] & gv),
         "d_ttc_fail": mean_stat((sub[hr, pk_new, 3] < 1).astype(float) - (sub[hr, base_h, 3] < 1).astype(float), np.ones(len(hidx), bool)),
         "d_nc_fail": mean_stat((sub[hr, pk_new, 0] < 1).astype(float) - (sub[hr, base_h, 0] < 1).astype(float), np.ones(len(hidx), bool)),
         "d_dac_fail": mean_stat((sub[hr, pk_new, 1] < 1).astype(float) - (sub[hr, base_h, 1] < 1).astype(float), np.ones(len(hidx), bool)),
         "d_abs_y4_m_changed": mean_stat(np.abs(prop[hr, pk_new, -1, 1]) - np.abs(prop[hr, base_h, -1, 1]), pk_new != base_h),
         "d_ade_human_m_cruise": mean_stat(ade(pk_new) - ade(base_h), cru),
         "ade_human_m_cruise_base": mean_stat(ade(base_h), cru),
         "n_changed": int((pk_new != base_h).sum()), "n_changed_worse": int(((pd_new < pd_b - 1e-9) & (pk_new != base_h)).sum()),
         "n_changed_better": int(((pd_new > pd_b + 1e-9) & (pk_new != base_h)).sum())}
    return o_new, d


res = {"prereg_sha256": got, "gates": gates, "n_heldout_tokens": int(len(hidx)), "n_heldout_logs": int(nl),
       "n_heldout_cruise": int(cru.sum()), "n_heldout_gate": int(GATE[hidx].sum()),
       "n_heldout_by_stratum": {k: int((strat == k).sum()) for k in ("zero", "random")}}
res["banked"], _, _, _ = arm_summary(D["pick"])
res["primary"], res["primary_paired"] = paired(vec)
dp = res["primary_paired"]["d_pdms_mix_x100"]
dsl = res["primary_paired"]["d_slow_share_cruise"]
cond_i = dp["lo"] > -0.5
cond_ii = (dsl["point"] <= -0.10) and (dsl["hi"] < 0)
harm = (dp["point"] < -0.5) and (dp["hi"] < 0)
no_fall = dsl["point"] >= 0
verdict = "ADOPT-for-sim" if (cond_i and cond_ii) else ("REFUTED" if (harm or no_fall) else "NOT PROVEN")
res["verdict"] = {"cond_i_pdms_noninferior": bool(cond_i), "cond_ii_slow_drop": bool(cond_ii), "harm_shown": bool(harm),
                  "slow_share_did_not_fall": bool(no_fall), "VERDICT": verdict}
print("VERDICT", res["verdict"])
# ---- secondary rules, DESCRIPTIVE only
sec = {"Sg(1)": np.where(GATE, sr.pick_S(L, s, 1), D["pick"]), "Bg(0.2)": np.where(GATE, sr.pick_B(L, s, 0.2), D["pick"]),
       "L(0.25)": sr.pick_L(L, 0.25), "E(0.5)": sr.pick_E(L, 0.5), "uBG(0.2,0.05)": sr.pick_BG(L, s, 0.2, 0.05)}
res["secondary_DESCRIPTIVE"] = {}
for k, pk in sec.items():
    _, d = paired(pk)
    res["secondary_DESCRIPTIVE"][k] = {q: d[q] for q in ("d_pdms_mix_x100", "d_slow_share_cruise", "d_bias_cruise", "d_ttc_fail", "n_changed")}
# ---- sim token (TRAIN log: spec check, not evidence)
i = int(np.where(D["token"] == sc.SIM_TOKEN)[0][0])
res["sim_token"] = {"token": sc.SIM_TOKEN, "in_train": bool(D["log"][i] in split["train_logs"]), "v0": float(v0[i]), "ax0": float(ax0[i]),
                    "gate_on": bool(GATE[i]), "banked_pick": int(D["pick"][i]), "rule_pick": int(vec[i]),
                    "changed": bool(vec[i] != D["pick"][i]),
                    "v4_over_v0_banked": float(v4[i, D["pick"][i]] / v0[i]), "v4_over_v0_rule": float(v4[i, vec[i]] / v0[i]),
                    "agg_banked": float(agg[i, D["pick"][i]]), "agg_rule": float(agg[i, vec[i]]),
                    "pdms_banked": float(D["pdms"][i, D["pick"][i]]), "pdms_rule": float(D["pdms"][i, vec[i]]),
                    "n_admitted": int(((agg[i] >= 0.8 * agg[i, D["pick"][i]]) & np.all(P[i][:, [0, 1, 3]] >= P[i, D["pick"][i], [0, 1, 3]] - EPS, -1)).sum())}
print("sim", res["sim_token"])
json.dump(res, open(sc.OUT + "/p2_result.json", "w"), indent=1)
print("wrote p2_result.json")
