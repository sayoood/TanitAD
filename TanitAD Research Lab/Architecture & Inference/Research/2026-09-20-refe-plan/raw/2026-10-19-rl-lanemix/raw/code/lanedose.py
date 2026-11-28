#!/usr/bin/env python3
"""RL-2 step 3: LANE-DOSE composites of the RL scorer.  Pre-registered in PREREG_RL2_LANEDOSE.md (sha256 beside it).  CPU numpy only.
    python lanedose.py --out out/raw/lanedose_rlval.json

Family:  x = RL{EP, C} + BASE{NC, DAC, TTC, DDC} + lane logit  (1-a)*base + a*RL    (L(a), a in 0.25, 0.5, 0.75, 1.0)
                                                         or   base + clip(RL - base, -tau, +tau)   (C(tau), tau in 0.6, 1.4, 2.6)
Per arm: choose c* on the FIT half of RL-val logs (mechanical rule of the PREREG), read c* ONCE on the HELD half.
Estimator: paired log-cluster bootstrap, 10,000 resamples, seed 20260927 (rlh.Boot) built on the logs of the half being read.
Tier: RL-val SELECTION tier (true NAVSIM v1.1 sub-scores of the argmax pick), not T1.
"""
import argparse, hashlib, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rlh  # noqa: E402
from rlh import pick, compose  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--dry", action="store_true", help="outcome-free dry run: split sanity + pick-identity controls only, no label scoring")
ap.add_argument("--prereg-sha", required=True, help="path of PREREG_RL2_LANEDOSE.sha256 (the script refuses to run without it)")
a = ap.parse_args()

R0D = "C:/Users/Admin/par18/rl/r0d_dl"
ARMS = [("r0", 1000, rlh.DL), ("r0b", 500, rlh.DL), ("r0d", 1000, R0D)]     # first = PRIMARY (formal verdict)
ALPHAS = (0.25, 0.5, 0.75, 1.0)
TAUS = (0.6, 1.4, 2.6)
CONFIGS = [("L", x) for x in ALPHAS] + [("C", x) for x in TAUS]            # 7 selectable configs, in this order (tie rule: first)
SPLIT_SEED, N_FIT, PERM_SEED = 20261019, 38, 20261020
Q = ("pdms", "nc", "dac", "ep", "ttc", "c", "ddc")
BAR_S = -0.5


def cname(c):
    return f"{c[0]}{c[1]}"


kv, ck, s, logs, st = rlh.load_labels()
N = len(kv)
ar = np.arange(N)
ul = sorted(set(logs.tolist()))
assert len(ul) == 75 and N == 3000, (len(ul), N)
perm_logs = np.random.default_rng(SPLIT_SEED).permutation(len(ul))
fit_set = set(ul[i] for i in perm_logs[:N_FIT])
m_fit = np.asarray([l in fit_set for l in logs.tolist()])
m_held = ~m_fit
BT = {"FIT": rlh.Boot(logs[m_fit]), "HELD": rlh.Boot(logs[m_held]), "ALL": rlh.Boot(logs)}
MASK = {"FIT": m_fit, "HELD": m_held, "ALL": np.ones(N, bool)}
base0 = rlh.load_dump(f"{rlh.DL}/r0/final/scores_step0_base.npz", kv, ck)
ib0 = pick(base0)


def delta(ia, ib, half, qs=Q):
    m = MASK[half]
    return {q: list(BT[half].ci((100.0 * (s[q][ar, ia] - s[q][ar, ib]))[m])) for q in qs}


def label(d):
    ok_gain = d["pdms"][1] > 0
    adv = [c for c in ("nc", "dac", "ttc") if d[c][2] < 0]
    return ("PASS-VAL" if (ok_gain and not adv) else "GAIN-LOST-VAL" if not adv else "SAFETY-LOST-VAL" if ok_gain else "BOTH-LOST-VAL"), adv


def deficit(d):
    return max(0.0, -d["pdms"][1]) + sum(max(0.0, -d[c][2]) for c in ("nc", "dac", "ttc"))


def barS(dA):
    """programme bar S vs own A on point estimates: TTC and NC may not fall by more than 0.5; returns (pass, shortfall)"""
    sf = max(0.0, BAR_S - dA["ttc"][0]) + max(0.0, BAR_S - dA["nc"][0])
    return bool(dA["ttc"][0] >= BAR_S and dA["nc"][0] >= BAR_S), sf


def build(B, cfg):
    x = compose(B, base0, (2, 4))
    if cfg is None:                                  # V1 = alpha 0
        return x
    kind, v = cfg
    if kind == "L":
        x[..., 6] = (1.0 - v) * base0[..., 6] + v * B[..., 6]
    else:
        x[..., 6] = base0[..., 6] + np.clip(B[..., 6] - base0[..., 6], -v, v)
    return x


def retention(ip, iB, half):
    """ratio of means: delta(config) / delta(B itself), both vs base0, same resamples (a ratio estimator)"""
    m = MASK[half]
    out = {}
    for q in ("pdms", "ttc", "nc"):
        num = np.where(m, 100.0 * (s[q][ar, ip] - s[q][ar, ib0]), 0.0)[m]
        den = np.where(m, 100.0 * (s[q][ar, iB] - s[q][ar, ib0]), 0.0)[m]
        out[q] = list(BT[half].ci_ratio(num, den))
    return out


def full_block(ip, iV1, iAo, iB, half):
    d0 = delta(ip, ib0, half)
    lab, adv = label(d0)
    dA = delta(ip, iAo, half, ("pdms", "ttc", "nc", "ep"))
    sp, sf = barS(dA)
    return {"vs_base0": d0, "vs_ownA": dA, "vs_V1": delta(ip, iV1, half), "vs_B": delta(ip, iB, half, ("pdms", "ttc", "nc", "ep")),
            "label": lab, "adverse": adv, "deficit": deficit(d0), "barS_pass": sp, "barS_shortfall": sf,
            "retention_of_B": retention(ip, iB, half), "picks_changed_vs_base0": int((ip != ib0).sum()), "picks_changed_vs_V1": int((ip != iV1).sum())}


res = {"tool": "lanedose/1", "prereg_sha256": open(a.prereg_sha).read().split()[0], "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
       "split": {"seed": SPLIT_SEED, "n_fit_logs": N_FIT, "n_fit_sets": int(m_fit.sum()), "n_held_logs": len(ul) - N_FIT, "n_held_sets": int(m_held.sum()),
                 "fit_logs": sorted(fit_set)}, "arms": {}, "controls": {}}
CT = res["controls"]
CT["split_sanity"] = {"disjoint_complete": bool((m_fit ^ m_held).all()), "logs_disjoint": bool(not (set(logs[m_fit].tolist()) & set(logs[m_held].tolist()))),
                      "n_logs_fit": len(set(logs[m_fit].tolist())), "n_logs_held": len(set(logs[m_held].tolist()))}

for arm, u, dl in ARMS:
    B = rlh.load_dump(f"{dl}/{arm}/final/scores_update{u}_B.npz", kv, ck)
    Aown = rlh.load_dump(f"{dl}/{arm}/final/scores_update{u}_A.npz", kv, ck)
    iB, iAo = pick(B), pick(Aown)
    X = {None: build(B, None)}
    for c in CONFIGS:
        X[c] = build(B, c)
    P = {k: pick(v) for k, v in X.items()}
    iV1 = P[None]
    D1 = pick(compose(B, base0, (2, 4, 6)))
    # --- controls for this arm (picks only)
    CT[f"{arm}_u{u}"] = {
        "alpha0_equals_V1": bool(np.array_equal(pick(compose(B, base0, (2, 4))), iV1)),
        "tau0_equals_V1": bool(np.array_equal(pick(build(B, ("C", 0.0))), iV1)),
        "alpha1_equals_D1": bool(np.array_equal(P[("L", 1.0)], D1)),
        "tauInf_equals_D1": bool(np.array_equal(pick(build(B, ("C", 1e9))), D1)),
        "algebra_max_abs": float(max(np.abs((1 - v) * base0[..., 6] + v * B[..., 6] - (base0[..., 6] + v * (B[..., 6] - base0[..., 6]))).max() for v in ALPHAS)),
    }
    if a.dry:
        print("DRY", arm, u, json.dumps(CT[f"{arm}_u{u}"]), flush=True)
        continue
    blk = {"state": f"{arm}_u{u}", "fit": {}, "held": {}, "refs_held": {}}
    # --- FIT: all 7 configs + reference rows
    fit = {cname(c): full_block(P[c], iV1, iAo, iB, "FIT") for c in CONFIGS}
    blk["fit"]["configs"] = fit
    blk["fit"]["V1"] = {"vs_base0": delta(iV1, ib0, "FIT"), "label": label(delta(iV1, ib0, "FIT"))[0]}
    blk["fit"]["B_itself"] = {"vs_base0": delta(iB, ib0, "FIT"), "label": label(delta(iB, ib0, "FIT"))[0]}
    order = sorted(range(len(CONFIGS)), key=lambda i: (round(fit[cname(CONFIGS[i])]["deficit"], 9), round(fit[cname(CONFIGS[i])]["barS_shortfall"], 9),
                                                      -fit[cname(CONFIGS[i])]["vs_base0"]["pdms"][0], i))
    cstar = CONFIGS[order[0]]
    feasible = [cname(c) for c in CONFIGS if fit[cname(c)]["label"] == "PASS-VAL"]
    # dose-response diagnostics on FIT (alpha grid only)
    G = {cname(("L", x)): fit[cname(("L", x))]["vs_V1"]["pdms"][0] for x in ALPHAS}
    K = {cname(("L", x)): min(fit[cname(("L", x))]["vs_V1"]["ttc"][0], fit[cname(("L", x))]["vs_V1"]["nc"][0]) for x in ALPHAS}
    comove = bool(len(feasible) == 0 and all(G[k] > 0 and K[k] < 0 for k in G))
    free = bool(any(G[k] > 0 and K[k] >= 0 for k in G))
    nogain = bool(not any(G[k] > 0 for k in G))
    blk["fit"].update({"feasible_PASS_configs": feasible, "F_empty": len(feasible) == 0, "G_dPDMS_vs_V1": G, "K_min_TTC_NC_vs_V1": K,
                       "COMOVE_FIT": comove, "FREE_POINT_FIT": free, "NOGAIN_FIT": nogain, "selected": cname(cstar),
                       "selection_order": [cname(CONFIGS[i]) for i in order]})
    # --- HELD: c* ONCE, plus the references (V1, B itself, own A) that are not candidates
    held = full_block(P[cstar], iV1, iAo, iB, "HELD")
    blk["held"] = {"config": cname(cstar), **held}
    blk["refs_held"] = {"B_itself_vs_base0": delta(iB, ib0, "HELD"), "V1_vs_base0": delta(iV1, ib0, "HELD"), "ownA_vs_base0": delta(iAo, ib0, "HELD")}
    blk["refs_held"]["B_itself_label"] = label(blk["refs_held"]["B_itself_vs_base0"])[0]
    blk["held"]["UNDERPOWERED"] = bool(blk["refs_held"]["B_itself_vs_base0"]["pdms"][1] <= 0)
    H, S = blk["held"]["label"], blk["held"]["barS_pass"]
    if H == "PASS-VAL":
        row = "R1-L-PASS" if S else "R2-L-PASS-BASE"
    elif H in ("SAFETY-LOST-VAL", "BOTH-LOST-VAL"):
        row = "R3-L-SAFETY-LOST-HELD"
    else:
        row = "R4a-L-GAIN-LOST-HELD-COMOVE" if comove else ("R4b-L-GAIN-LOST-HELD-FREEPOINT" if free else "R4c-L-GAIN-LOST-HELD-NOGAIN")
    blk["row"] = row
    # --- POST-VERDICT DESCRIPTIVE table (FIT union HELD; no selection, no verdict; HELD already consumed by c*)
    desc = {cname(c): full_block(P[c], iV1, iAo, iB, "ALL") for c in CONFIGS}
    blk["desc_all"] = {"note": "DESCRIPTIVE, post-verdict, FIT+HELD union; does not select and does not change the row", "configs": desc,
                       "V1_vs_base0": delta(iV1, ib0, "ALL"), "B_itself_vs_base0": delta(iB, ib0, "ALL"), "ownA_vs_base0": delta(iAo, ib0, "ALL")}
    Ga = {cname(("L", x)): desc[cname(("L", x))]["vs_V1"]["pdms"][0] for x in ALPHAS}
    Ka = {cname(("L", x)): min(desc[cname(("L", x))]["vs_V1"]["ttc"][0], desc[cname(("L", x))]["vs_V1"]["nc"][0]) for x in ALPHAS}
    blk["desc_all"]["G_dPDMS_vs_V1"], blk["desc_all"]["K_min_TTC_NC_vs_V1"] = Ga, Ka
    blk["desc_all"]["COMOVE_ALL_pointwise"] = bool(all(Ga[k] > 0 and Ka[k] < 0 for k in Ga))
    res["arms"][f"{arm}_u{u}"] = blk
    print(f"[{arm} u{u}] FIT feasible={feasible} selected={cname(cstar)} COMOVE_FIT={comove} FREE={free} NOGAIN={nogain}", flush=True)
    print(f"   HELD {cname(cstar)}: {H} S={S} dPDMS={['%.3f' % x for x in held['vs_base0']['pdms']]} dTTC={['%.3f' % x for x in held['vs_base0']['ttc']]} "
          f"dNC={['%.3f' % x for x in held['vs_base0']['nc']]} UNDERPOWERED={blk['held']['UNDERPOWERED']} -> {row}", flush=True)

# --- global controls (primary arm)
if a.dry:
    print("DRY split", json.dumps(CT["split_sanity"]), "n_fit_sets", int(m_fit.sum()), "n_held_sets", int(m_held.sum()))
    sys.exit(0)
B = rlh.load_dump(f"{rlh.DL}/r0/final/scores_update1000_B.npz", kv, ck)
perm = np.random.default_rng(PERM_SEED).permutation(N)
xr = compose(B, base0[perm], (2, 4, 6))                                          # L(1.0) with NC, DAC, TTC, DDC from the base0 of OTHER sets
dr = delta(pick(xr), ib0, "ALL")
CT["regression_arm_L1_shuffled_safety_SRC"] = {"vs_base0": dr, "pdms_lb_below_zero": bool(dr["pdms"][1] < 0)}
iB = pick(B)
d = 100 * (s["ttc"][ar, iB] - s["ttc"][ar, ib0])
CT["boot_identity_max_abs"] = {"ALL": rlh.ctl_boot(kv, logs, d)[2]}
for h in ("FIT", "HELD"):
    idx = np.flatnonzero(MASK[h])
    CT["boot_identity_max_abs"][h] = rlh.ctl_boot([kv[i] for i in idx], logs[idx], d[idx])[2]
# reproduction of RL-1's banked R0 u1000 V1 and D1 rows (full-set Boot, same estimator)
bk = json.load(open("C:/Users/Admin/par18/rl/out/raw/composites_rlval.json"))["states"]["r0_u1000"]
a1 = res["arms"]["r0_u1000"]["desc_all"]
mx1 = max(abs(x - y) for q in Q for x, y in zip(a1["V1_vs_base0"][q], bk["V1"]["vs_base0"][q]))
mx2 = max(abs(x - y) for q in Q for x, y in zip(a1["configs"]["L1.0"]["vs_base0"][q], bk["descriptive"]["D1_H1shape_base0"]["vs_base0"][q]))
CT["reproduce_RL1_V1_and_D1_max_abs"] = {"V1_alpha0": mx1, "D1_alpha1": mx2}
json.dump(res, open(a.out, "w"), indent=1)
print("controls:", json.dumps(CT)[:2500])
print("WROTE", a.out)
