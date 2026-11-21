#!/usr/bin/env python3
"""RL-S1 O2 lever (i): the TTC / NC head-calibration read, EXACTLY as registered in eval/PREREG_RL_S1.md section 11 (appended 2026-10-14T23:33:56Z,
before this script computed any calibrated pick).  CPU, zero training, banked dumps only (sha256-pinned in calib_common.PIN).

Primary P: per-head Platt map sigmoid(a z + b), a in [0.01, 100], b in [-50, 50], on TTC (col 3) and NC (col 0); fit = mean BCE against the proposal's own
sub-score label over all 64 proposals of the fitting sets (scipy L-BFGS-B from (1, 0), float64).  Cross-fitting: 5 folds over the 75 RL-val logs,
fold = rank(sha256("<seed>|" + log)) mod 5, seed 20261015; each set is scored with the map fit on the OTHER four folds.  The same procedure for every dump.
Bar (per read, calibrated R vs calibrated control, same tag): B1 TTC and NC point deltas (x100) >= -0.5; B2 PDMS 95 % lower bound > 0.
Deciding reads: R0-B u500 + u1000 vs C0-A; R0b-B u250 + u500 vs C0-A AND vs R0b-A.  An arm passes iff B1 and B2 hold at both of its reads (and, for R0b, both controls).
Controls: CTRL-ID, CTRL-REACH, CTRL-LEAK (11.4).  Diagnostics D1-D6 and secondaries S1-S3 (11.5) decide nothing.

    python calib_read.py --out calib_read.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calib_common as CC  # noqa: E402

RV = CC.RV
PREREG_SECTION11_APPENDED_UTC = "2026-10-14T23:33:56Z"
PREREG_SHA256_AFTER_APPEND = "97b9e6e8652c4972491853cbaba5445c429dd293423fd21f3299afaca7ecaeeb"
K_FOLDS, FOLD_SEED, ALT_FOLD_SEEDS = 5, 20261015, (20261016, 20261017, 20261018)
A_BOUNDS, B_BOUNDS = (0.01, 100.0), (-50.0, 50.0)
ISO_CLIP = (1e-6, 1.0 - 1e-6)
HEAD_LABEL = {0: "nc", 1: "dac", 2: "ep", 3: "ttc", 4: "c"}
VARIANTS = {"P": ("platt", (0, 3)),              # primary: NC, TTC
            "S1": ("platt", (0, 1, 3)),          # + DAC
            "S2": ("iso", (0, 3)),               # isotonic on the probability
            "S3": ("platt", (0, 1, 2, 3, 4))}    # every output the rule uses (no lane label on RL-val)
DUMPS = {("C0A", t): f"c0/final/scores_{t}_A.npz" for t in ("update250", "update500", "update750", "update1000")}
DUMPS.update({("R0B", t): f"r0/final/scores_{t}_B.npz" for t in ("update250", "update500", "update750", "update1000")})
DUMPS.update({("R0A", t): f"r0/final/scores_{t}_A.npz" for t in ("update500", "update1000")})
DUMPS.update({("R0bB", t): f"r0b/final/scores_{t}_B.npz" for t in ("update250", "update500")})
DUMPS.update({("R0bA", t): f"r0b/final/scores_{t}_A.npz" for t in ("update250", "update500")})
DECIDING = (("R0", "R0B", "update500", "C0A"), ("R0", "R0B", "update1000", "C0A"),
            ("R0b", "R0bB", "update250", "C0A"), ("R0b", "R0bB", "update500", "C0A"),
            ("R0b", "R0bB", "update250", "R0bA"), ("R0b", "R0bB", "update500", "R0bA"))
REPORTED = (("R0", "R0B", "update250", "C0A"), ("R0", "R0B", "update750", "C0A"),
            ("R0", "R0B", "update500", "R0A"), ("R0", "R0B", "update1000", "R0A"))


# ------------------------------------------------------------------------------------------------------------------ folds
def folds_of(keys, seed):
    logs = sorted({k[0] for k in keys})
    order = sorted(logs, key=lambda lg: hashlib.sha256(f"{seed}|{lg}".encode()).hexdigest())
    fold_of_log = {lg: r % K_FOLDS for r, lg in enumerate(order)}
    return np.asarray([fold_of_log[k[0]] for k in keys]), fold_of_log


# ------------------------------------------------------------------------------------------------------------------ maps
def fit_platt(z, y):
    z, y = z.ravel(), y.ravel()

    def f(x):
        u = x[0] * z + x[1]
        loss = np.mean(np.logaddexp(0.0, u) - y * u)
        g = RV.sigmoid(u) - y
        return loss, np.array([np.mean(g * z), np.mean(g)])
    r = minimize(f, np.array([1.0, 0.0]), jac=True, method="L-BFGS-B", bounds=[A_BOUNDS, B_BOUNDS])
    return {"a": float(r.x[0]), "b": float(r.x[1]), "converged": bool(r.success), "bce": float(r.fun)}


def apply_platt(m, z):
    return RV.sigmoid(m["a"] * z + m["b"])


def fit_iso(z, y):
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(RV.sigmoid(z.ravel()), y.ravel())
    return ir


def apply_iso(ir, z):
    return np.clip(ir.predict(RV.sigmoid(z.ravel())).reshape(z.shape), *ISO_CLIP)


def calibrate(logits, s, variant, fold, keys, insample=False):
    """-> probabilities [N, 64, 7] with the variant's heads replaced by their cross-fitted (or in-sample) calibrated values, and the map record."""
    method, heads = VARIANTS[variant]
    p = RV.sigmoid(logits)
    rec = {}
    parts = [(None, np.ones(len(keys), bool), np.ones(len(keys), bool))] if insample else \
        [(f, fold != f, fold == f) for f in range(K_FOLDS)]
    for f, fit_m, sc_m in parts:
        if not insample:                                             # CTRL-LEAK
            fl, sl = {keys[i][0] for i in np.flatnonzero(fit_m)}, {keys[i][0] for i in np.flatnonzero(sc_m)}
            assert not (fl & sl), f"fold {f}: fitting and scored logs overlap"
        for h in heads:
            y = s[HEAD_LABEL[h]]
            if method == "platt":
                m = fit_platt(logits[fit_m][..., h], y[fit_m])
                p[sc_m, :, h] = apply_platt(m, logits[sc_m][..., h])
                rec[f"fold{f}|col{h}"] = m
            else:
                ir = fit_iso(logits[fit_m][..., h], y[fit_m])
                p[sc_m, :, h] = apply_iso(ir, logits[sc_m][..., h])
                rec[f"fold{f}|col{h}"] = {"n_thresholds": int(len(ir.X_thresholds_))}
    return p, rec


# ------------------------------------------------------------------------------------------------------------------ diagnostics
def bce(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def ece(p, y, nb=15):
    p, y = p.ravel(), y.ravel()
    o = np.argsort(p, kind="stable")
    tot = 0.0
    for ch in np.array_split(o, nb):
        tot += len(ch) * abs(p[ch].mean() - y[ch].mean())
    return float(tot / len(p))


def quality(p_raw, p_cal, s, heads=(0, 3)):
    out = {}
    for h in heads:
        y = s[HEAD_LABEL[h]]
        out[HEAD_LABEL[h].upper()] = {"bce_raw": bce(p_raw[..., h], y), "bce_cal": bce(p_cal[..., h], y), "ece_raw": ece(p_raw[..., h], y),
                                      "ece_cal": ece(p_cal[..., h], y), "mean_p_raw": float(p_raw[..., h].mean()), "mean_p_cal": float(p_cal[..., h].mean()),
                                      "mean_label": float(y.mean())}
    return out


def bar(tab):
    b1 = bool(tab["TTC"]["mean"] >= -0.5 and tab["NC"]["mean"] >= -0.5)
    b2 = bool(tab["PDMS"]["ci95"][0] > 0)
    return {"B1_g5": b1, "B2_pdms_lo_gt_0": b2, "pass": b1 and b2,
            "TTC": tab["TTC"]["mean"], "NC": tab["NC"]["mean"], "PDMS": tab["PDMS"]["mean"], "PDMS_lo": tab["PDMS"]["ci95"][0]}


def read_set(reads, picks_cal, s, keys):
    out = {}
    for arm, rd, t, ctl in reads:
        tab = CC.delta_table(s, picks_cal[(rd, t)], picks_cal[(ctl, t)], keys)
        out[f"{rd}_{t}_vs_{ctl}"] = {"arm": arm, "table": tab, "bar": bar(tab)}
    return out


def verdict(dec):
    def arm_pass(arm):
        rows = [v for v in dec.values() if v["arm"] == arm]
        return bool(rows) and all(r["bar"]["pass"] for r in rows)
    r0, r0b = arm_pass("R0"), arm_pass("R0b")
    return {"R0_pass": r0, "R0b_pass": r0b, "outcome": "PASS" if r0 else ("PASS-R0b" if r0b else "FAIL")}


# ------------------------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dl", default=CC.DL_DEFAULT)
    ap.add_argument("--x", default=CC.X_DEFAULT)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    pre = os.path.join(CC.PK, "eval", "PREREG_RL_S1.md")
    res = {"tool": "calib_read/1", "prereg": {"file": "eval/PREREG_RL_S1.md", "section": 11, "appended_utc": PREREG_SECTION11_APPENDED_UTC,
                                              "sha256_expected": PREREG_SHA256_AFTER_APPEND, "sha256_now": CC.sha(pre)},
           "tier": "RL-val (selection tier; never an endpoint)", "rule": "navsim_v1_lane",
           "estimator": "paired log-cluster bootstrap, proxy_eval.boot, 10,000, seed 20260927 -- another draw of LOGS only; maps held fixed",
           "units": "points = x100 of the per-set sub-score at the pick", "repos": CC.REPOS}
    if res["prereg"]["sha256_now"] != PREREG_SHA256_AFTER_APPEND:
        raise SystemExit("PREREG_RL_S1.md changed since section 11 was registered")
    res["inputs_sha256"] = CC.verify(a.dl, sorted(set(DUMPS.values())) + ["data/data/rlfan_pdm_val.tar.gz", "data/data/rlfan_terms_val.tar.gz"])
    keys, tg, ck, s, st = CC.load_val(a.x)
    res["rl_val_join"], res["n_sets"], res["n_logs"] = st, len(keys), len({k[0] for k in keys})
    L = {d: CC.load_scores_aligned(os.path.join(a.dl, rel), keys, ck) for d, rel in DUMPS.items()}
    raw_picks = {d: CC.picks_from_logits(z) for d, z in L.items()}
    fold, fold_of_log = folds_of(keys, FOLD_SEED)
    res["folds"] = {"seed": FOLD_SEED, "k": K_FOLDS, "logs_per_fold": [int(sum(1 for v in fold_of_log.values() if v == f)) for f in range(K_FOLDS)],
                    "sets_per_fold": [int((fold == f).sum()) for f in range(K_FOLDS)], "fold_of_log": fold_of_log}

    # ---- CTRL-ID: identity maps reproduce the raw picks; raw R0 u1000 - C0 matches the kill package
    ctrl = {"identity_changed_picks": {}, "reach_changed_picks": {}}
    for d, z in L.items():
        p = RV.sigmoid(z)
        q = p.copy()
        for h in (0, 3):
            q[..., h] = RV.sigmoid(1.0 * z[..., h] + 0.0)
        ctrl["identity_changed_picks"][f"{d[0]}_{d[1]}"] = int((CC.picks_from_probs(q) != raw_picks[d]).sum())
        for h, nm in ((3, "TTC"), (0, "NC")):                       # CTRL-REACH: a constant map must change the pick somewhere
            q = p.copy()
            q[..., h] = 0.5
            ctrl["reach_changed_picks"][f"{d[0]}_{d[1]}|{nm}"] = int((CC.picks_from_probs(q) != raw_picks[d]).sum())
    ar = np.arange(len(keys))
    ctrl["raw_R0B_u1000_PDMS_x100"] = float(100 * s["pdms"][ar, raw_picks[("R0B", "update1000")]].mean())
    ctrl["raw_C0A_u1000_PDMS_x100"] = float(100 * s["pdms"][ar, raw_picks[("C0A", "update1000")]].mean())
    ctrl["expect_rlval_vs_C0"] = [91.22299347222223, 90.05977159722221]
    ctrl["CTRL_ID_ok"] = bool(all(v == 0 for v in ctrl["identity_changed_picks"].values())
                              and abs(ctrl["raw_R0B_u1000_PDMS_x100"] - 91.22299347222223) < 1e-9
                              and abs(ctrl["raw_C0A_u1000_PDMS_x100"] - 90.05977159722221) < 1e-9)
    ctrl["CTRL_REACH_ok"] = bool(all(v > 0 for v in ctrl["reach_changed_picks"].values()))
    res["controls"] = ctrl
    if not (ctrl["CTRL_ID_ok"] and ctrl["CTRL_REACH_ok"]):
        res["VOID"] = "a control did not read its known value (11.4)"
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
        raise SystemExit("VOID: control failed")
    print("controls OK", time.time() - t0, flush=True)

    # ---- the raw (uncalibrated) reads, for reference
    res["raw_reads"] = read_set(DECIDING + REPORTED, raw_picks, s, keys)

    # ---- PRIMARY (P), cross-fitted, fold seed 20261015
    Pcal, maps, qual = {}, {}, {}
    for d, z in L.items():
        p, rec = calibrate(z, s, "P", fold, keys)
        Pcal[d] = CC.picks_from_probs(p)
        maps[f"{d[0]}_{d[1]}"] = rec
        qual[f"{d[0]}_{d[1]}"] = quality(RV.sigmoid(z), p, s)
    res["ctrl_leak_ok"] = True                                       # calibrate() asserts per fold; reaching here means no overlap
    dec = read_set(DECIDING, Pcal, s, keys)
    res["PRIMARY"] = {"deciding": dec, "reported": read_set(REPORTED, Pcal, s, keys), "verdict": verdict(dec),
                      "picks_changed_by_calibration": {f"{d[0]}_{d[1]}": int((Pcal[d] != raw_picks[d]).sum()) for d in L}}
    res["D1_quality_heldout"] = qual
    res["D4_maps"] = maps
    res["D2_calC0_minus_rawC0"] = {t: CC.delta_table(s, Pcal[("C0A", t)], raw_picks[("C0A", t)], keys)
                                   for t in ("update250", "update500", "update750", "update1000")}
    res["D3_calR_minus_rawC0"] = {f"{rd}_{t}": CC.delta_table(s, Pcal[(rd, t)], raw_picks[("C0A", t)], keys) for _a, rd, t, _c in DECIDING[:4]}
    print("primary", res["PRIMARY"]["verdict"], time.time() - t0, flush=True)

    # ---- D5 in-sample (never read against the bar)
    ins = {d: CC.picks_from_probs(calibrate(z, s, "P", fold, keys, insample=True)[0]) for d, z in L.items()}
    d5 = read_set(DECIDING, ins, s, keys)
    res["D5_insample_NOT_A_RESULT"] = {k: v["bar"] for k, v in d5.items()}

    # ---- D6 fold-seed stability of the primary
    d6 = {}
    for sd in ALT_FOLD_SEEDS:
        fo, _ = folds_of(keys, sd)
        pk = {d: CC.picks_from_probs(calibrate(z, s, "P", fo, keys)[0]) for d, z in L.items()}
        dd = read_set(DECIDING, pk, s, keys)
        d6[str(sd)] = {"bars": {k: v["bar"] for k, v in dd.items()}, "verdict": verdict(dd)}
    res["D6_fold_seed_stability"] = d6
    res["D6_verdict_flips"] = [sd for sd, v in d6.items() if v["verdict"]["outcome"] != res["PRIMARY"]["verdict"]["outcome"]]
    print("D6 done", time.time() - t0, flush=True)

    # ---- secondaries S1-S3 (deciding nothing)
    sec = {}
    for v in ("S1", "S2", "S3"):
        pk = {d: CC.picks_from_probs(calibrate(z, s, v, fold, keys)[0]) for d, z in L.items()}
        dd = read_set(DECIDING, pk, s, keys)
        sec[v] = {"spec": VARIANTS[v], "deciding_reads": dd, "verdict_under_same_bar": verdict(dd)}
        print(v, sec[v]["verdict_under_same_bar"], time.time() - t0, flush=True)
    res["SECONDARIES_NOT_DECIDING"] = sec

    o = res["PRIMARY"]["verdict"]["outcome"]
    res["OUTCOME"] = {"outcome": o + (", fold-unstable" if (o != "FAIL" and res["D6_verdict_flips"]) else ""),
                      "next_lever": {"PASS": "section 12 pre-registration, then full-navtest R0 u1000 + cal vs C0 u1000 + cal (dev box)",
                                     "PASS-R0b": "section 12 pre-registration, then full-navtest R0b u500 + cal vs C0 u500 + cal (dev box)",
                                     "FAIL": "O2 (ii): separate safety head = PI decision; if S3 cleared the bar, its own pre-registration first"}[o],
                      "secondary_cleared_where_P_failed": [v for v in sec if o == "FAIL" and sec[v]["verdict_under_same_bar"]["outcome"] != "FAIL"]}
    res["runtime_s"] = time.time() - t0
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print("ZZCALIB_READ_OK", res["OUTCOME"], a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
