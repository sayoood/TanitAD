#!/usr/bin/env python3
"""The H-SEL-1 fresh-token confirmation read (PREREG_SEL_CONFIRM.md). CPU only. Tier T0 (open loop, one step, logged ego history).

Statistics are p2_score.py's, line for line (ratio estimator over resampled logs, rng = default_rng(20260927).choice(n_logs, (10000, n_logs)), 95 %
percentile, the SAME draws for every statistic and both arms). The ONE structural difference is written in the prereg: a fresh token is never in the
zeroed stratum (all 529 zeroed u400 tokens are C3 tokens), so the full-navtest mix is read with the zeroed-stratum delta at its STRUCTURAL FLOOR 0
(banked PDMS there is exactly 0 and PDMS >= 0, so the true zeroed delta is >= 0): mix = W_zero * 0 + W_random * delta_random. This can only make
the statistic more conservative.

    python selc_analyze.py --replay-p2           # G-ANA: the same code on P2's held-out C3 tokens must reproduce p2_result.json EXACTLY; RED arm must not
    python selc_analyze.py --verdict             # reads dump/dump_confirm.npz + score/verdict_columns.npz -> confirm_result.json + CONFIRM_RESULT.md
    python selc_analyze.py --table               # after the 64-column table: secondary H-SEL-LANE-1 + descriptive rules -> confirm_table_result.json + CONFIRM_TABLE_APPENDIX.md
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
YSEL = "E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107"
sys.path.insert(0, YSEL)
sys.path.insert(0, HERE)
import selc_rule as R  # noqa: E402

W = {"zero": 529 / 12146, "random": 11617 / 12146}
B, SEED = 10000, 20260927
PREREG = os.path.join(HERE, "PREREG_SEL_CONFIRM.md")
PREREG_SHA = os.path.join(HERE, "PREREG_SEL_CONFIRM.sha256")
DT = 0.2


def gt_speed4(pose20, t20):
    i = 19
    dq = np.linalg.norm(pose20[:, i, :2] - pose20[:, i - 1, :2], axis=-1)
    return dq / (t20[:, i] - t20[:, i - 1])


def cruise_mask(gt_valid, v0, ax0, v4_gt):
    """refe_ego_probe_20261106/make_cruise_tokens.py: gv & v0 > 5 & |ax0| < 0.3 & |GT v4 - v0| < 1 (GT for the LABEL only, never at inference)."""
    with np.errstate(invalid="ignore"):
        return np.asarray(gt_valid, bool) & (v0 > 5.0) & (np.abs(ax0) < 0.3) & (np.abs(np.where(gt_valid, v4_gt, np.nan) - v0) < 1.0)


class Reader:
    """P2's estimator bound to one population (its logs, its draws)."""

    def __init__(self, log, stratum):
        self.ul, self.li = np.unique(log, return_inverse=True)
        self.nl = len(self.ul)
        self.draws = np.random.default_rng(SEED).choice(self.nl, size=(B, self.nl), replace=True)
        self.strat = np.asarray(stratum)

    def per_draw_mean(self, vals, mask):
        vals = np.asarray(vals, np.float64)
        m = np.asarray(mask, bool) & np.isfinite(vals)
        sm = np.bincount(self.li[m], weights=vals[m], minlength=self.nl)
        ct = np.bincount(self.li[m], minlength=self.nl).astype(np.float64)
        S, C = sm[self.draws].sum(1), ct[self.draws].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return S / C, sm.sum() / max(ct.sum(), 1e-12)

    @staticmethod
    def ci(arr, pt):
        a = arr[np.isfinite(arr)]
        return {"point": float(pt), "lo": float(np.percentile(a, 2.5)), "hi": float(np.percentile(a, 97.5)), "n_draws_finite": int(len(a))}

    def mix_stat(self, vals):
        """P2's mix; a stratum with NO tokens in this population enters at its structural floor 0 (stated, never silent)."""
        has_z = bool((self.strat == "zero").any())
        dr, pr = self.per_draw_mean(vals, self.strat == "random")
        if has_z:
            dz, pz = self.per_draw_mean(vals, self.strat == "zero")
        else:
            dz, pz = np.zeros_like(dr), 0.0
        return self.ci(100 * (W["zero"] * dz + W["random"] * dr), 100 * (W["zero"] * pz + W["random"] * pr))

    def mean_stat(self, vals, mask, scale=1.0):
        d, p = self.per_draw_mean(vals, mask)
        if not np.isfinite(d).any():
            return {"point": None, "lo": None, "hi": None, "n_draws_finite": 0, "n": int(np.asarray(mask, bool).sum())}
        return self.ci(scale * d, scale * p)


def arm_summary(Rd, A, pk):
    """P2 arm_summary on the population A (arrays already restricted), pick pk [n]."""
    n = len(pk)
    hr = np.arange(n)
    pd = A["pdms"][hr, pk] if A["pdms"].ndim == 2 else A["pdms_of"](pk)
    sub = A["sub_of"](pk)
    v4p = A["v4_all"][hr, pk]
    ratio = v4p / np.maximum(A["v0"], 1e-6)
    prop = A["prop_of"](pk)
    cru, gv, one = A["cruise"], A["gt_valid"], np.ones(n, bool)
    o = {"pdms_mix_x100": Rd.mix_stat(pd), "pdms_unweighted_x100": Rd.mean_stat(pd, one, 100),
         "pdms_random_x100": Rd.mean_stat(pd, Rd.strat == "random", 100), "pdms_zero_x100": Rd.mean_stat(pd, Rd.strat == "zero", 100),
         "slow_share_cruise": Rd.mean_stat((ratio < 0.85).astype(float), cru),
         "overspeed_share_cruise": Rd.mean_stat((ratio > 1.15).astype(float), cru),
         "v4_over_v0_cruise": Rd.mean_stat(ratio, cru),
         "bias_v4_minus_gt_cruise": Rd.mean_stat(v4p - A["v4_gt"], cru & gv),
         "bias_v4_minus_gt_gate": Rd.mean_stat(v4p - A["v4_gt"], A["gate"] & gv),
         "bias_v4_minus_gt_all": Rd.mean_stat(v4p - A["v4_gt"], gv),
         "nc_fail_rate": Rd.mean_stat((sub[:, 0] < 1).astype(float), one),
         "dac_fail_rate": Rd.mean_stat((sub[:, 1] < 1).astype(float), one),
         "ttc_fail_rate": Rd.mean_stat((sub[:, 3] < 1).astype(float), one),
         "ddc_mean": Rd.mean_stat(sub[:, 5], one),
         "pick_changed_share": float((pk != A["base"]).mean()),
         "abs_y4_m_cruise": Rd.mean_stat(np.abs(prop[:, -1, 1]), cru),
         "abs_heading4_rad_cruise": Rd.mean_stat(np.abs(prop[:, -1, 2]), cru)}
    return o, pd, ratio, sub, prop


def paired(Rd, A, pk_new):
    o_new, pd_new, ratio_new, sub_n, prop_n = arm_summary(Rd, A, pk_new)
    base = A["base"]
    n = len(base)
    hr = np.arange(n)
    _, pd_b, ratio_b, sub_b, prop_b = arm_summary(Rd, A, base)
    v4n, v4b = A["v4_all"][hr, pk_new], A["v4_all"][hr, base]
    one, cru, gv = np.ones(n, bool), A["cruise"], A["gt_valid"]
    h8 = A["human8"]
    ade = lambda prop: np.linalg.norm(prop[..., :2] - h8[..., :2], axis=-1).mean(-1)
    d = {"d_pdms_mix_x100": Rd.mix_stat(pd_new - pd_b),
         "d_pdms_unweighted_x100": Rd.mean_stat(pd_new - pd_b, one, 100),
         "d_pdms_random_x100": Rd.mean_stat(pd_new - pd_b, Rd.strat == "random", 100),
         "d_pdms_zero_x100": Rd.mean_stat(pd_new - pd_b, Rd.strat == "zero", 100),
         "d_slow_share_cruise": Rd.mean_stat((ratio_new < 0.85).astype(float) - (ratio_b < 0.85).astype(float), cru),
         "d_overspeed_share_cruise": Rd.mean_stat((ratio_new > 1.15).astype(float) - (ratio_b > 1.15).astype(float), cru),
         "d_bias_cruise": Rd.mean_stat(v4n - v4b, cru & gv), "d_bias_gate": Rd.mean_stat(v4n - v4b, A["gate"] & gv),
         "d_ttc_fail": Rd.mean_stat((sub_n[:, 3] < 1).astype(float) - (sub_b[:, 3] < 1).astype(float), one),
         "d_nc_fail": Rd.mean_stat((sub_n[:, 0] < 1).astype(float) - (sub_b[:, 0] < 1).astype(float), one),
         "d_dac_fail": Rd.mean_stat((sub_n[:, 1] < 1).astype(float) - (sub_b[:, 1] < 1).astype(float), one),
         "d_abs_y4_m_changed": Rd.mean_stat(np.abs(prop_n[:, -1, 1]) - np.abs(prop_b[:, -1, 1]), pk_new != base),
         "d_ade_human_m_cruise": Rd.mean_stat(ade(prop_n) - ade(prop_b), cru),
         "ade_human_m_cruise_base": Rd.mean_stat(ade(prop_b), cru),
         "n_changed": int((pk_new != base).sum()), "n_changed_worse": int(((pd_new < pd_b - 1e-9) & (pk_new != base)).sum()),
         "n_changed_better": int(((pd_new > pd_b + 1e-9) & (pk_new != base)).sum())}
    return o_new, d


def verdict_of(d):
    """PREREG_SEL section 2 committed outcomes, verbatim logic of p2_score.py."""
    dp, dsl = d["d_pdms_mix_x100"], d["d_slow_share_cruise"]
    slow_ok = dsl.get("point") is not None             # n_cruise = 0 => the clause is unmeasurable: never a pass, never a refutation
    cond_i = dp["lo"] > -0.5
    cond_ii = slow_ok and (dsl["point"] <= -0.10) and (dsl["hi"] < 0)
    harm = (dp["point"] < -0.5) and (dp["hi"] < 0)
    no_fall = slow_ok and dsl["point"] >= 0
    v = "ADOPT-for-sim" if (cond_i and cond_ii) else ("REFUTED" if (harm or no_fall) else "NOT PROVEN")
    return {"cond_i_pdms_noninferior": bool(cond_i), "cond_ii_slow_drop": bool(cond_ii), "harm_shown": bool(harm), "slow_share_did_not_fall": bool(no_fall),
            "slow_clause_measurable": bool(slow_ok), "VERDICT": v}


# ---------------------------------------------------------------------------------------------------------------- G-ANA replay
def replay_p2() -> dict:
    """The analysis code on P2's 505 held-out C3 tokens must reproduce p2_result.json's primary block EXACTLY (both arms from C3's table)."""
    import sel_common as sc
    D = sc.load()
    split = json.load(open(YSEL + "/split.json"))
    HO = np.isin(D["log"], split["heldout_logs"])
    h = np.where(HO)[0]
    N = len(D["token"])
    rule = R.rule_vec(D["logits"], D["v4_all"], D["v0"], D["ax0"])
    Z = np.load(sc.EGO + "/full/results.npz")
    zt = {str(t): i for i, t in enumerate(Z["token"])}
    A = {"pdms": D["pdms"][h], "sub_of": None, "v4_all": D["v4_all"][h], "v0": D["v0"][h], "cruise": D["cruise"][h], "gt_valid": D["gt_valid"][h],
         "v4_gt": D["v4_gt"][h], "gate": ((D["v0"] > 5.0) & (np.abs(D["ax0"]) < 0.3))[h], "base": D["pick"][h],
         "human8": Z["human8"][[zt[t] for t in D["token"][h]]].astype(np.float64)}
    hr = np.arange(len(h))
    subH, propH = D["sub"][h], D["proposals"][h]
    A["sub_of"] = lambda pk: subH[hr, pk]
    A["prop_of"] = lambda pk: propH[hr, pk]
    Rd = Reader(D["log"][h], D["stratum"][h])
    o, d = paired(Rd, A, rule[h])
    ref = json.load(open(YSEL + "/p2_result.json"))
    keys = ["d_pdms_mix_x100", "d_pdms_unweighted_x100", "d_pdms_random_x100", "d_pdms_zero_x100", "d_slow_share_cruise", "d_overspeed_share_cruise",
            "d_bias_cruise", "d_bias_gate", "d_ttc_fail", "d_nc_fail", "d_dac_fail", "d_abs_y4_m_changed", "d_ade_human_m_cruise", "ade_human_m_cruise_base"]
    diffs = {}
    for k in keys:
        a, b = d[k], ref["primary_paired"][k]
        diffs[k] = max(abs(a[q] - b[q]) for q in ("point", "lo", "hi"))
    for k in ("n_changed", "n_changed_worse", "n_changed_better"):
        diffs[k] = abs(d[k] - ref["primary_paired"][k])
    for k in ("pdms_mix_x100", "slow_share_cruise", "bias_v4_minus_gt_cruise", "dac_fail_rate"):
        diffs["arm_" + k] = max(abs(o[k][q] - ref["primary"][k][q]) for q in ("point", "lo", "hi"))
    v = verdict_of(d)
    # RED arm: the same reader with the arms SWAPPED must not reproduce (sign flips), and banked-vs-banked must read exactly 0 with CI [0, 0]
    _, d_sw = paired(Rd, dict(A, base=rule[h]), D["pick"][h])
    _, d_id = paired(Rd, A, D["pick"][h])
    # the CRUISE formula here must reproduce the ego probe's tokens_cruise.json on all 1,058
    cm = cruise_mask(D["gt_valid"], D["v0"], D["ax0"], D["v4_gt"])
    g = {"max_abs_diff_vs_p2_result": max(diffs.values()), "per_key": diffs, "verdict_replayed": v["VERDICT"], "verdict_p2": ref["verdict"]["VERDICT"],
         "RED_swapped_arms_d_pdms_mix": d_sw["d_pdms_mix_x100"], "RED_identity_d_pdms_mix": d_id["d_pdms_mix_x100"],
         "cruise_formula_equals_tokens_cruise": int((cm == D["cruise"]).sum()), "n": int(N)}
    g["PASS"] = bool(g["max_abs_diff_vs_p2_result"] <= 1e-12 and v["VERDICT"] == ref["verdict"]["VERDICT"]
                     and abs(d_sw["d_pdms_mix_x100"]["point"] + d["d_pdms_mix_x100"]["point"]) <= 1e-12 and d_sw["d_pdms_mix_x100"]["point"] != d["d_pdms_mix_x100"]["point"]
                     and d_id["d_pdms_mix_x100"]["point"] == 0.0 and d_id["d_pdms_mix_x100"]["lo"] == 0.0 and d_id["d_pdms_mix_x100"]["hi"] == 0.0
                     and g["cruise_formula_equals_tokens_cruise"] == N)
    # the zero-floor mix on the random stratum only (what a fresh read computes), for the record
    rnd = D["stratum"][h] == "random"
    hh = h[rnd]
    hr2 = np.arange(len(hh))
    A2 = {k: (v_[rnd] if isinstance(v_, np.ndarray) else v_) for k, v_ in A.items()}
    A2["sub_of"] = lambda pk: D["sub"][hh][hr2, pk]
    A2["prop_of"] = lambda pk: D["proposals"][hh][hr2, pk]
    _, d2 = paired(Reader(D["log"][hh], D["stratum"][hh]), A2, rule[hh])
    g["DESCRIPTIVE_heldout_random_only_zero_floor_mix"] = d2["d_pdms_mix_x100"]
    g["DESCRIPTIVE_heldout_random_only_slow"] = d2["d_slow_share_cruise"]
    return g


# ---------------------------------------------------------------------------------------------------------------- the fresh read
def load_fresh(dump_npz, cols_npz):
    Dd = np.load(dump_npz, allow_pickle=False)
    C = np.load(cols_npz, allow_pickle=False)
    tok = np.array([str(t) for t in Dd["token"]])
    assert (np.array([str(t) for t in C["token"]]) == tok).all()
    ego = Dd["ego"].astype(np.float64)
    v0, ax0 = ego[:, 6], ego[:, 2]
    v4_all = R.step_speed4(Dd["traj"])
    gv = Dd["gt_valid"].astype(bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        v4_gt = gt_speed4(Dd["gt_pose20"].astype(np.float64), Dd["gt_t20"].astype(np.float64))
    base = Dd["pick"].astype(np.int64)
    rule = C["pick_rule"].astype(np.int64)
    n = len(tok)
    hr = np.arange(n)
    props8 = Dd["props8"].astype(np.float64)
    pd_b, pd_r = C["pdms_banked"].astype(np.float64), C["pdms_rule"].astype(np.float64)
    sub_b, sub_r = C["sub_banked"].astype(np.float64), C["sub_rule"].astype(np.float64)

    def pdms_of(pk):
        assert ((pk == base) | (pk == rule)).all()
        return np.where(pk == base, pd_b, pd_r) if not (pk == rule).all() else pd_r

    def sub_of(pk):
        assert ((pk == base) | (pk == rule)).all()
        return np.where((pk == base)[:, None], sub_b, sub_r) if not (pk == rule).all() else sub_r
    A = {"pdms": np.zeros((0,)), "pdms_of": pdms_of, "sub_of": sub_of, "prop_of": lambda pk: props8[hr, pk], "v4_all": v4_all, "v0": v0, "ax0": ax0,
         "cruise": cruise_mask(gv, v0, ax0, v4_gt), "gt_valid": gv, "v4_gt": v4_gt, "gate": (v0 > 5.0) & (np.abs(ax0) < 0.3), "base": base,
         "human8": Dd["human8"].astype(np.float64), "rule": rule, "token": tok, "log": np.array([str(x) for x in Dd["log"]]), "logits": Dd["logits"]}
    # pdms_of(pk) where the rule equals the banked pick on unchanged tokens: both columns are the same proposal there (asserted in the gates)
    return A


def check_prereg():
    got = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    want = open(PREREG_SHA).read().split()[0].strip()
    if got != want:
        sys.exit(f"REFUSED: PREREG_SEL_CONFIRM.md sha256 {got} != frozen {want}")
    return got


def guard(out_dir, plumbing):
    """--plumbing skips the prereg pin ONLY outside the package (a CPU plumbing run can never write the real CONFIRM_RESULT.md)."""
    if not plumbing:
        return check_prereg()
    if os.path.abspath(out_dir) == os.path.abspath(HERE):
        sys.exit("REFUSED: --plumbing needs --out-dir outside the package")
    return "PLUMBING-NO-PREREG"


def fmt(c, p=3, scale=1.0):
    if c is None or c.get("point") is None:
        return "n/a"
    return f"{scale * c['point']:+.{p}f} [{scale * c['lo']:+.{p}f}, {scale * c['hi']:+.{p}f}]"


def verdict_read(out_dir, tokens_json=None, plumbing=False):
    sha = guard(out_dir, plumbing)
    A = load_fresh(os.path.join(out_dir, "dump", "dump_confirm.npz"), os.path.join(out_dir, "score", "verdict_columns.npz"))
    doc = json.load(open(tokens_json or os.path.join(HERE, "tokens_confirm.json")))
    stratum = np.array(["random"] * len(A["token"]))
    Rd = Reader(A["log"], stratum)
    o_b, _, _, _, _ = arm_summary(Rd, A, A["base"])
    o_r, d = paired(Rd, A, A["rule"])
    v = verdict_of(d)
    res = {"prereg_sha256": sha, "tier": "T0 (open loop, one step, logged ego history)", "n_tokens": int(len(A["token"])), "n_logs": int(Rd.nl),
           "n_cruise": int(A["cruise"].sum()), "n_gate": int(A["gate"].sum()), "n_gt_valid": int(A["gt_valid"].sum()),
           "estimator": "paired log-cluster bootstrap over the fresh logs, 10,000 resamples, seed 20260927, 95 % percentile; answers 'another draw of logs' only "
                        "(one checkpoint; deterministic planner, ego probe C1) -- says nothing about training or inference seed",
           "mix_note": "no fresh token is zeroed (all 529 are C3 tokens): mix = W_zero*0 (structural floor) + W_random*delta_random",
           "banked": o_b, "rule": o_r, "paired": d, "verdict": v}
    # DESCRIPTIVE: by log class (held-out logs vs the 13 never-in-C3 logs) -- the leakage / winner's-curse check, never the verdict
    lc = np.array([doc["log_class"][l] for l in A["log"]])
    res["by_log_class_DESCRIPTIVE"] = {}
    for cls in ("heldout", "new"):
        m = lc == cls
        if m.sum() < 2:
            continue
        idx = np.where(m)[0]
        Am = {k: (v_[idx] if isinstance(v_, np.ndarray) and v_.ndim >= 1 and len(v_) == len(m) else v_) for k, v_ in A.items()}
        hr = np.arange(len(idx))
        p8 = np.load(os.path.join(out_dir, "dump", "dump_confirm.npz"))["props8"][idx].astype(np.float64)
        C = np.load(os.path.join(out_dir, "score", "verdict_columns.npz"))
        pb, pr_, sb, sr_ = C["pdms_banked"][idx], C["pdms_rule"][idx], C["sub_banked"][idx], C["sub_rule"][idx]
        bm, rm = Am["base"], Am["rule"]
        Am["pdms_of"] = lambda pk, bm=bm, pb=pb, pr_=pr_: np.where(pk == bm, pb, pr_)
        Am["sub_of"] = lambda pk, bm=bm, sb=sb, sr_=sr_: np.where((pk == bm)[:, None], sb, sr_)
        Am["prop_of"] = lambda pk, p8=p8, hr=hr: p8[hr, pk]
        _, dm = paired(Reader(Am["log"], stratum[idx]), Am, rm)
        res["by_log_class_DESCRIPTIVE"][cls] = {"n_tokens": int(m.sum()), "n_logs": int(len(set(Am["log"]))), "n_cruise": int(Am["cruise"].sum()),
                                                 "d_pdms_mix_x100": dm["d_pdms_mix_x100"], "d_slow_share_cruise": dm["d_slow_share_cruise"]}
    json.dump(res, open(os.path.join(out_dir, "confirm_result.json"), "w"), indent=1)
    write_md(res, os.path.join(out_dir, "CONFIRM_RESULT.md"))
    return res


def write_md(r, p):
    d, b, o = r["paired"], r["banked"], r["rule"]
    v = r["verdict"]
    L = [f"# CONFIRM_RESULT -- H-SEL-1 fresh-token confirmation of the FROZEN gBG(0.20, 0.05) (stream YSEL2)", "",
         f"**Committed verdict: {v['VERDICT']}** (mechanical, `PREREG_SEL_CONFIRM.md` sha256 `{r['prereg_sha256']}`). Written by `selc_analyze.py --verdict`; "
         f"every number MEASURED, artifact `confirm_result.json`. **Tier T0.** Population: {r['n_tokens']} fresh navtest tokens / {r['n_logs']} logs "
         f"(disjoint from the 1,058 C3 tokens, outside YSEL's 61 TRAIN logs, all non-zeroed), {r['n_cruise']} CRUISE.", "",
         f"Estimator: {r['estimator']}.", f"Mix: {r['mix_note']}.", "",
         "## The decision", "| condition | bar | measured | holds |", "|---|---|---|---|",
         f"| (i) PDMS non-inferior (zero-floor mix, x100) | lower CI > -0.5 | {fmt(d['d_pdms_mix_x100'])} | {'YES' if v['cond_i_pdms_noninferior'] else 'NO'} |",
         f"| (ii) CRUISE slow-down share drop | <= -10 pts and CI-hi < 0 | {fmt(d['d_slow_share_cruise'], 1, 100)} pts | {'YES' if v['cond_ii_slow_drop'] else 'NO'} |",
         f"| harm (REFUTED branch) | point < -0.5 and hi < 0 | {'yes' if v['harm_shown'] else 'no'} | -- |",
         f"| slow-down did not fall (REFUTED branch) | point >= 0 | {'yes' if v['slow_share_did_not_fall'] else 'no'} | -- |", "",
         f"Changed picks {d['n_changed']} (better {d['n_changed_better']}, worse {d['n_changed_worse']} by true PDMS).", "",
         "## Four families (banked -> rule; paired delta with CI)", "| family | metric | banked | rule | paired delta |", "|---|---|---|---|---|",
         f"| LONGITUDINAL | plan v4 - GT v4, CRUISE (m/s) | {fmt(b['bias_v4_minus_gt_cruise'])} | {fmt(o['bias_v4_minus_gt_cruise'])} | {fmt(d['d_bias_cruise'])} |",
         f"| | plan v4 - GT v4, ego gate set | {fmt(b['bias_v4_minus_gt_gate'])} | {fmt(o['bias_v4_minus_gt_gate'])} | {fmt(d['d_bias_gate'])} |",
         f"| | slow-down share (v4/v0 < 0.85), CRUISE | {fmt(b['slow_share_cruise'])} | {fmt(o['slow_share_cruise'])} | {fmt(d['d_slow_share_cruise'])} |",
         f"| | overspeed share (v4/v0 > 1.15), CRUISE | {fmt(b['overspeed_share_cruise'])} | {fmt(o['overspeed_share_cruise'])} | {fmt(d['d_overspeed_share_cruise'])} |",
         f"| | mean plan v4/v0, CRUISE | {fmt(b['v4_over_v0_cruise'])} | {fmt(o['v4_over_v0_cruise'])} | -- |",
         f"| LATERAL (what the proposals allow) | ADE to human future, CRUISE (m) | {fmt(d['ade_human_m_cruise_base'])} | -- | {fmt(d['d_ade_human_m_cruise'])} |",
         f"| | abs(y) at 4 s on changed picks (m) | -- | -- | {fmt(d['d_abs_y4_m_changed'])} |",
         f"| | DAC failure rate | {fmt(b['dac_fail_rate'])} | {fmt(o['dac_fail_rate'])} | {fmt(d['d_dac_fail'])} |",
         f"| | DDC mean | {fmt(b['ddc_mean'])} | {fmt(o['ddc_mean'])} | -- |",
         f"| safety | NC fail rate | {fmt(b['nc_fail_rate'])} | {fmt(o['nc_fail_rate'])} | {fmt(d['d_nc_fail'])} |",
         f"| | TTC fail rate | {fmt(b['ttc_fail_rate'])} | {fmt(o['ttc_fail_rate'])} | {fmt(d['d_ttc_fail'])} |",
         f"| PDMS | zero-floor mix x100 | {fmt(b['pdms_mix_x100'], 2)} | {fmt(o['pdms_mix_x100'], 2)} | {fmt(d['d_pdms_mix_x100'], 2)} |",
         f"| | unweighted x100 (random stratum) | {fmt(b['pdms_unweighted_x100'], 2)} | {fmt(o['pdms_unweighted_x100'], 2)} | {fmt(d['d_pdms_unweighted_x100'], 2)} |",
         "| TACTICAL | -- | n/a: no manoeuvre-decision label on these tokens and REFe emits no tactical output here (n = 0) | | |",
         "| STRATEGIC | -- | **PAUSED** (PI ruling 2026-09-19, `PI_DECISION_QUEUE.md` item 25) | | |", "",
         "## DESCRIPTIVE by log class (never the verdict; the leakage / shared-log check of PREREG section 3)"]
    for cls, x in r.get("by_log_class_DESCRIPTIVE", {}).items():
        L.append(f"* {cls}: {x['n_tokens']} tokens / {x['n_logs']} logs / {x['n_cruise']} CRUISE: dPDMS mix {fmt(x['d_pdms_mix_x100'])}, "
                 f"d slow {fmt(x['d_slow_share_cruise'], 1, 100)} pts")
    L += ["", "The 64-column per-proposal table (secondary H-SEL-LANE-1 + descriptive rules) follows in `CONFIRM_TABLE_APPENDIX.md` when `markers/selc_full.DONE` exists."]
    with open(p, "wb") as f:
        f.write(("\n".join(L) + "\n").encode("utf-8"))


# ---------------------------------------------------------------------------------------------------------------- the table appendix
def table_read(out_dir, plumbing=False):
    sha = guard(out_dir, plumbing)
    import sel_rules as sr
    Dd = np.load(os.path.join(out_dir, "dump", "dump_confirm.npz"), allow_pickle=False)
    T = np.load(os.path.join(out_dir, "score", "table_confirm.npz"), allow_pickle=False)
    C = np.load(os.path.join(out_dir, "score", "verdict_columns.npz"), allow_pickle=False)
    tok = np.array([str(t) for t in Dd["token"]])
    assert (np.array([str(t) for t in T["token"]]) == tok).all()
    n = len(tok)
    hr = np.arange(n)
    ego = Dd["ego"].astype(np.float64)
    v0, ax0 = ego[:, 6], ego[:, 2]
    L = Dd["logits"]
    v4_all = R.step_speed4(Dd["traj"])
    s = R.speed_ratio(v4_all, v0)
    GATE = (v0 > 5.0) & (np.abs(ax0) < 0.3)
    gv = Dd["gt_valid"].astype(bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        v4_gt = gt_speed4(Dd["gt_pose20"].astype(np.float64), Dd["gt_t20"].astype(np.float64))
    base = Dd["pick"].astype(np.int64)
    props8 = Dd["props8"].astype(np.float64)
    A = {"pdms": T["pdms"].astype(np.float64), "sub_of": lambda pk: T["sub"][hr, pk].astype(np.float64), "prop_of": lambda pk: props8[hr, pk],
         "v4_all": v4_all, "v0": v0, "cruise": cruise_mask(gv, v0, ax0, v4_gt), "gt_valid": gv, "v4_gt": v4_gt, "gate": GATE, "base": base,
         "human8": Dd["human8"].astype(np.float64)}
    log = np.array([str(x) for x in Dd["log"]])
    Rd = Reader(log, np.array(["random"] * n))
    out = {"prereg_sha256": sha, "tier": "T0", "n_tokens": n, "n_logs": int(Rd.nl)}
    # gates: the table at the banked and the rule pick equals the verdict columns bit for bit
    rule = C["pick_rule"].astype(np.int64)
    out["G_table_at_banked_eq_column"] = int((T["pdms"][hr, base] == C["pdms_banked"]).sum())
    out["G_table_at_rule_eq_column"] = int((T["pdms"][hr, rule] == C["pdms_rule"]).sum())
    out["G_PASS"] = bool(out["G_table_at_banked_eq_column"] == n and out["G_table_at_rule_eq_column"] == n)
    # SECONDARY pre-registered (PREREG section 7): H-SEL-LANE-1 = L(0.25), ungated, its own bars
    pk_L = sr.pick_L(L, 0.25)
    _, dL = paired(Rd, A, pk_L)
    lo, pt, hi = dL["d_pdms_mix_x100"]["lo"], dL["d_pdms_mix_x100"]["point"], dL["d_pdms_mix_x100"]["hi"]
    out["H_SEL_LANE_1"] = {"rule": "L(0.25): NC*DAC*(5EP+5TTC+2C)/12 * p_lane**0.25, argmax, ungated", "paired": dL,
                           "VERDICT": "CONFIRMED (PDMS superiority)" if lo > 0 else ("REFUTED" if (pt < 0 and hi < 0) else "NOT PROVEN")}
    # DESCRIPTIVE: the other P2 secondary rules, and the selection headroom on fresh tokens
    sec = {"Sg(1)": np.where(GATE, sr.pick_S(L, s, 1), base), "Bg(0.2)": np.where(GATE, sr.pick_B(L, s, 0.2), base), "E(0.5)": sr.pick_E(L, 0.5),
           "uBG(0.2,0.05)": sr.pick_BG(L, s, 0.2, 0.05), "gBG(0.2,0.05)=H-SEL-1": rule}
    out["secondary_DESCRIPTIVE"] = {}
    for k, pk in sec.items():
        _, d = paired(Rd, A, pk)
        out["secondary_DESCRIPTIVE"][k] = {q: d[q] for q in ("d_pdms_mix_x100", "d_slow_share_cruise", "d_bias_cruise", "d_ttc_fail", "n_changed")}
    P = A["pdms"] * 100
    out["headroom_DESCRIPTIVE_x100"] = {"actual": Rd.mean_stat(P[hr, base], np.ones(n, bool)), "random_mean_of_64": Rd.mean_stat(P.mean(1), np.ones(n, bool)),
                                        "best_of_64": Rd.mean_stat(P.max(1), np.ones(n, bool))}
    json.dump(out, open(os.path.join(out_dir, "confirm_table_result.json"), "w"), indent=1)
    md = ["# CONFIRM_TABLE_APPENDIX -- the 64-column per-proposal table on the fresh tokens (stream YSEL2)", "",
          f"Tier T0. {n} tokens / {Rd.nl} logs. PREREG sha256 `{sha}`. Gates: table at banked pick == banked column {out['G_table_at_banked_eq_column']}/{n}; "
          f"table at rule pick == rule column {out['G_table_at_rule_eq_column']}/{n} -> {'PASS' if out['G_PASS'] else 'FAIL (appendix VOID)'}.", "",
          f"## SECONDARY pre-registered: H-SEL-LANE-1 L(0.25) -> **{out['H_SEL_LANE_1']['VERDICT']}**",
          f"dPDMS zero-floor mix x100 {fmt(dL['d_pdms_mix_x100'], 2)}; d slow {fmt(dL['d_slow_share_cruise'], 1, 100)} pts; d TTC fail {fmt(dL['d_ttc_fail'])}; "
          f"d DAC fail {fmt(dL['d_dac_fail'])}; changed {dL['n_changed']} (better {dL['n_changed_better']}, worse {dL['n_changed_worse']}).", "",
          "## DESCRIPTIVE (can change no verdict)"]
    for k, x in out["secondary_DESCRIPTIVE"].items():
        md.append(f"* {k}: dPDMS {fmt(x['d_pdms_mix_x100'], 2)}, d slow {fmt(x['d_slow_share_cruise'], 1, 100)} pts, changed {x['n_changed']}")
    hd = out["headroom_DESCRIPTIVE_x100"]
    md.append(f"* headroom (unweighted, x100): actual {fmt(hd['actual'], 2)}, random-of-64 {fmt(hd['random_mean_of_64'], 2)}, best-of-64 {fmt(hd['best_of_64'], 2)}")
    with open(os.path.join(out_dir, "CONFIRM_TABLE_APPENDIX.md"), "wb") as f:
        f.write(("\n".join(md) + "\n").encode("utf-8"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-p2", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--tokens", default=None)
    ap.add_argument("--plumbing", action="store_true")
    a = ap.parse_args(argv)
    if a.replay_p2:
        g = replay_p2()
        json.dump(g, open(os.path.join(HERE, "gate_ana_replay.json"), "w"), indent=1)
        print(json.dumps({k: g[k] for k in ("max_abs_diff_vs_p2_result", "verdict_replayed", "verdict_p2", "cruise_formula_equals_tokens_cruise", "PASS")}, indent=1))
        print("DESCRIPTIVE heldout random-only zero-floor mix", g["DESCRIPTIVE_heldout_random_only_zero_floor_mix"], g["DESCRIPTIVE_heldout_random_only_slow"])
        return 0 if g["PASS"] else 2
    if a.verdict:
        r = verdict_read(a.out_dir, a.tokens, a.plumbing)
        print("VERDICT", r["verdict"])
        return 0
    if a.table:
        r = table_read(a.out_dir, a.plumbing)
        print("table G_PASS", r["G_PASS"], "H-SEL-LANE-1", r["H_SEL_LANE_1"]["VERDICT"])
        return 0 if r["G_PASS"] else 2
    ap.print_help()
    return 4


if __name__ == "__main__":
    sys.exit(main())
