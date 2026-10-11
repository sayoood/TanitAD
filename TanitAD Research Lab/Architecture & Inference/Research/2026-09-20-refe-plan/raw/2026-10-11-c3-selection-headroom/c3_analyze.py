#!/usr/bin/env python3
"""C3 selection-headroom -- the READOUT (eval/P6_GO_SIGNAL.md section 3 "C3"; operationalisation in OPERATIONALISATION.md,
written before any u400 proposal was scored).  numpy only; fits NOTHING.

    python c3_analyze.py --dir <workdir containing table.npz> [--tokens tokens.json]
    python c3_analyze.py --selftest        # synthetic worlds with known answers + a deliberate-regression arm

Reads (all PDMS values are x100; "mix" = the full-navtest re-weighting 529/12146 zero + 11617/12146 non-zeroed):
  (a) rescuable share  = share of ZERO-stratum tokens with >= 1 proposal scoring NC == 1 AND DAC == 1
  (b) actual / random / oracle PDMS per stratum and in the mix, with PAIRED differences
        random = per token the mean PDMS over 100 uniform draws of the 64 (default_rng(20261011).integers, tokens sorted)
        oracle = per token max PDMS over the 64
  (c) EP headroom on the NON-zeroed stratum: EP(best SAFE proposal) - EP(pick), over tokens with >= 1 safe proposal;
        PRIMARY best = highest-EP proposal among the safe ones; variants are reported as DESCRIPTIVE
Estimator: paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927 -- the programme's draws
(`rng.choice(unique_logs, len(unique_logs), replace=True)`), both strata resampled on the SAME log draw.
Verdict (committed rule, precedence as written in OPERATIONALISATION.md): fidelity gate fails or random >= actual in the
mix -> INSTRUMENT FAILURE; else rescuable >= 50 % -> SCORER-BOUND; else GENERATION-BOUND.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time

import numpy as np

NC, DAC, EP, TTC, CMF, DDC = range(6)
SEED_BOOT = 20260927
SEED_DRAW = 20261011
N_BOOT = 10000
N_RANDOM_DRAWS = 100
N_FULL, N_ZERO_FULL = 12146, 529
W_ZERO, W_NZ = N_ZERO_FULL / N_FULL, (N_FULL - N_ZERO_FULL) / N_FULL
FULL_NAVTEST_ACTUAL = 88.24694109622521          # banked u400 full-navtest PDMS x100 (result.json primary.pdms_arm)
EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
RESCUE_BAR = 0.50


# ------------------------------------------------------------------------------------------------ bootstrap
class Boot:
    """Log-cluster bootstrap with the programme's draws; counts matrix C[n_boot, n_logs]."""

    def __init__(self, logs, n_boot=N_BOOT, seed=SEED_BOOT):
        self.ul, self.li = np.unique(np.asarray(logs), return_inverse=True)
        L = len(self.ul)
        rng = np.random.default_rng(seed)
        self.C = np.zeros((n_boot, L), dtype=np.float64)
        for b in range(n_boot):
            self.C[b] = np.bincount(rng.choice(L, size=L, replace=True), minlength=L)
        self.L = L

    def ratio(self, num, den):
        """per-draw  sum_l c_l num_l / sum_l c_l den_l  (tokens-level arrays; den is the count weight)."""
        n = np.bincount(self.li, weights=num, minlength=self.L)
        d = np.bincount(self.li, weights=den, minlength=self.L)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.C @ n) / (self.C @ d)


def ci(draws):
    d = draws[np.isfinite(draws)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))] if len(d) else [None, None]


def est(point, draws):
    lo, hi = ci(draws)
    return {"mean": None if point is None else float(point), "ci95": [lo, hi]}


# ------------------------------------------------------------------------------------------------ readout
def verdict(fidelity_pass, rescuable, random_mix, actual_mix):
    if (not fidelity_pass) or (random_mix >= actual_mix):
        return "INSTRUMENT FAILURE"
    return "SCORER-BOUND" if rescuable >= RESCUE_BAR else "GENERATION-BOUND"


def readout(token, pdms, sub, pick, actual, zero, logs, fidelity_pass=True, n_boot=N_BOOT, extra=None):
    """token[N], pdms[N,M] (0..1), sub[N,M,6], pick[N], actual[N] (0..1, the banked per-token score), zero[N] bool, logs[N]."""
    order = np.argsort(token)                                  # draws must not depend on dump order
    token, pdms, sub, pick, actual, zero, logs = (np.asarray(x)[order] for x in (token, pdms, sub, pick, actual, zero, logs))
    N, M = pdms.shape
    ar = np.arange(N)
    nz = ~zero
    B = Boot(logs, n_boot)
    safe = (sub[..., NC] == 1.0) & (sub[..., DAC] == 1.0)       # [N, M]
    resc = safe.any(1)
    rng = np.random.default_rng(SEED_DRAW)
    rnd = pdms[ar[:, None], rng.integers(0, M, size=(N, N_RANDOM_DRAWS))].mean(1)
    rnd_exact = pdms.mean(1)
    orc = pdms.max(1)
    at_pick = pdms[ar, pick]
    out = {"N": int(N), "M": int(M), "n_zero": int(zero.sum()), "n_nonzero": int(nz.sum()), "n_logs": int(B.L),
           "n_logs_zero": int(len(np.unique(logs[zero]))), "estimator": f"paired log-cluster bootstrap over {B.L} logs, {n_boot} resamples, "
           f"percentile 95 %, seed {SEED_BOOT}; both strata on the SAME log draw",
           "weights": {"zero": W_ZERO, "nonzero": W_NZ}}

    def stratum(v, mask):                                      # mean x100 and per-draw means
        d = B.ratio(v * mask * 100.0, mask.astype(float))
        return float(v[mask].mean() * 100.0), d

    # ---- (a)
    pa = float(resc[zero].mean())
    da = B.ratio(resc * zero.astype(float), zero.astype(float))
    out["a_rescuable"] = {"share": est(pa, da), "n_zero": int(zero.sum()), "n_rescuable": int(resc[zero].sum()),
                          "bar": RESCUE_BAR, "point_estimate_ge_bar": bool(pa >= RESCUE_BAR),
                          "ci_straddles_bar": bool(ci(da)[0] is not None and ci(da)[0] < RESCUE_BAR <= ci(da)[1]),
                          "mean_n_safe_proposals_zero_stratum": float(safe[zero].sum(1).mean()),
                          "share_with_any_PDMS_gt_0_zero_stratum": float((pdms[zero].max(1) > 0).mean()),
                          "share_rescuable_nonzero_stratum": float(resc[nz].mean())}
    # ---- (b)
    reads = {"actual": actual, "random": rnd, "oracle": orc, "random_exact_mean_of_64": rnd_exact}
    S = {}
    for name, v in reads.items():
        m0, d0 = stratum(v, zero)
        m1, d1 = stratum(v, nz)
        S[name] = (m0, d0, m1, d1, W_ZERO * m0 + W_NZ * m1, W_ZERO * d0 + W_NZ * d1)
    out["b_pdms_x100"] = {}
    for name in reads:
        m0, d0, m1, d1, mm, dm = S[name]
        out["b_pdms_x100"][name] = {"zero_stratum": est(m0, d0), "nonzero_stratum": est(m1, d1), "full_mix": est(mm, dm)}
    out["b_pdms_x100"]["table_at_pick_check"] = {"zero_stratum": float(at_pick[zero].mean() * 100), "nonzero_stratum": float(at_pick[nz].mean() * 100),
                                                 "max_abs_vs_banked": float(np.abs(at_pick - actual).max())}
    out["b_pdms_x100"]["sanity_mix_actual_vs_banked_full_navtest"] = {
        "mix_actual": S["actual"][4], "banked_full_navtest_pdms": FULL_NAVTEST_ACTUAL,
        "banked_inside_mix_ci": bool(ci(S["actual"][5])[0] <= FULL_NAVTEST_ACTUAL <= ci(S["actual"][5])[1])}
    out["b_paired_differences_x100"] = {}
    for a_, b_ in (("oracle", "actual"), ("actual", "random"), ("oracle", "random")):
        r = {}
        for key, i0, i1 in (("zero_stratum", 0, 1), ("nonzero_stratum", 2, 3)):
            diff = S[a_][i0] - S[b_][i0] if i0 == 0 else S[a_][2] - S[b_][2]
            dd = S[a_][1] - S[b_][1] if i0 == 0 else S[a_][3] - S[b_][3]
            r[key] = est(diff, dd)
        r["full_mix"] = est(S[a_][4] - S[b_][4], S[a_][5] - S[b_][5])
        out["b_paired_differences_x100"][f"{a_}_minus_{b_}"] = r
    og = {k: float(w * (S["oracle"][i] - S["actual"][i])) for k, w, i in (("zero_stratum", W_ZERO, 0), ("nonzero_stratum", W_NZ, 2))}
    out["b_oracle_gap_decomposition_mix_x100"] = {**og, "total": og["zero_stratum"] + og["nonzero_stratum"]}
    # ---- (c) EP headroom on the non-zeroed stratum
    ep = sub[..., EP]
    has = resc & nz
    best_ep = np.where(safe, ep, -np.inf).max(1)
    pick_ep = ep[ar, pick]
    head = np.where(has, best_ep - pick_ep, 0.0)
    den = has.astype(float)
    # variant 2: EP of the max-PDMS safe proposal; variant 3: tokens whose pick is itself safe; variant 4: PDMS-scale
    kb = np.where(safe, pdms, -np.inf).argmax(1)
    head2 = np.where(has, ep[ar, kb] - pick_ep, 0.0)
    pick_safe = safe[ar, pick] & nz
    den3 = pick_safe.astype(float)
    orc_safe = np.where(safe, pdms, -np.inf).max(1)
    head4 = np.where(has, orc_safe - at_pick, 0.0) * 100.0
    out["c_ep_headroom"] = {
        "label": "PRIMARY: EP(highest-EP safe proposal) - EP(pick); non-zeroed stratum, tokens with >= 1 safe proposal",
        "n_nonzero": int(nz.sum()), "n_with_safe": int(has.sum()), "n_without_safe": int((nz & ~resc).sum()),
        "primary": est(float(head[has].mean()) if has.any() else None, B.ratio(head, den)),
        "primary_x100": est(float(head[has].mean() * 100) if has.any() else None, B.ratio(head * 100, den)),
        "DESCRIPTIVE_variant_EP_of_max_PDMS_safe_minus_pick": est(float(head2[has].mean()) if has.any() else None, B.ratio(head2, den)),
        "DESCRIPTIVE_variant_pick_itself_safe": {"n": int(pick_safe.sum()), **est(float(head[pick_safe].mean()) if pick_safe.any() else None,
                                                                              B.ratio(head * pick_safe, den3))},
        "DESCRIPTIVE_PDMS_x100_of_best_safe_minus_pick_over_tokens_with_safe": est(float(head4[has].mean()) if has.any() else None,
                                                                                    B.ratio(head4, den)),
        "share_of_nonzero_tokens_where_pick_is_the_highest_EP_safe_proposal": float((head[has] <= 1e-12).mean()) if has.any() else None}
    # ---- verdict
    r_mix, a_mix = S["random"][4], S["actual"][4]
    out["verdict_inputs"] = {"fidelity_gate_pass": bool(fidelity_pass), "rescuable_share": pa, "random_mix": r_mix, "actual_mix": a_mix,
                             "random_lt_actual": bool(r_mix < a_mix)}
    out["VERDICT"] = verdict(fidelity_pass, pa, r_mix, a_mix)
    if extra:
        out.update(extra(dict(token=token, pdms=pdms, sub=sub, pick=pick, safe=safe, resc=resc, zero=zero, nz=nz, B=B, ar=ar)))
    return out


# ------------------------------------------------------------------------------------------ descriptive blocks
def descriptive(ctx, export_path=EXPORT, logits=None):
    """DESCRIPTIVE ONLY -- no part of this feeds the verdict. Breakdowns of the zero stratum by cause, v0 and command."""
    token, sub, pick, safe, resc, zero, ar = (ctx[k] for k in ("token", "sub", "pick", "safe", "resc", "zero", "ar"))
    d = {"label": "DESCRIPTIVE -- breakdowns carry no decision; nothing is fitted on these tokens"}
    pk = sub[ar, pick]
    cause = np.where((pk[:, NC] == 0) & (pk[:, DAC] == 0), "NC=0 and DAC=0", np.where(pk[:, NC] == 0, "NC=0 only",
                     np.where(pk[:, DAC] == 0, "DAC=0 only", "other")))
    d["zero_by_cause_at_pick"] = {c: {"n": int(((cause == c) & zero).sum()), "rescuable": int(((cause == c) & zero & resc).sum())}
                                  for c in ("NC=0 only", "DAC=0 only", "NC=0 and DAC=0", "other")}
    d["zero_stratum_TTC_at_pick_eq_0_n"] = int(((pk[:, TTC] == 0) & zero).sum())
    if os.path.exists(export_path):
        exp = json.load(gzip.open(export_path, "rt", encoding="utf-8"))["tokens"]
    else:
        exp = {}
    if exp and all(t in exp for t in token):
        v0 = np.array([float(np.hypot(*exp[t]["ego_statuses"][-1]["ego_velocity"])) for t in token])
        cmd = np.array([int(np.argmax(exp[t]["ego_statuses"][-1]["driving_command"])) for t in token])
        names = {0: "left", 1: "straight", 2: "right", 3: "unknown"}
        bins = [(0, 0.5), (0.5, 3), (3, 8), (8, 14), (14, 99)]
        d["zero_by_v0_mps"] = {f"[{lo},{hi})": {"n": int((zero & (v0 >= lo) & (v0 < hi)).sum()),
                                                "rescuable": int((zero & resc & (v0 >= lo) & (v0 < hi)).sum())} for lo, hi in bins}
        d["zero_by_command"] = {names[c]: {"n": int((zero & (cmd == c)).sum()), "rescuable": int((zero & resc & (cmd == c)).sum())} for c in names}
        d["nonzero_by_command_n"] = {names[c]: int((~zero & (cmd == c)).sum()) for c in names}
    if logits is not None and logits.shape[-1] >= 7:
        p = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        agg = p[..., 0] * p[..., 1] * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / 12.0 * p[..., 6]
        d["scorer_aggregate_reconstruction_argmax_equals_pick_share"] = float((agg.argmax(1) == pick).mean())
        rk = (-agg).argsort(1).argsort(1)                       # rank 0 = the scorer's top
        best_safe_rank = np.where(safe, rk, 10**6).min(1)
        r = best_safe_rank[zero & resc]
        d["zero_rescuable_scorer_rank_of_its_best_ranked_safe_proposal"] = {
            "note": "0 = the pick itself (impossible when the pick is unsafe); k = the scorer would have to look k+1 deep",
            "n": int(len(r)), "median": float(np.median(r)) if len(r) else None,
            "share_rank_le_2": float((r <= 2).mean()) if len(r) else None, "share_rank_le_7": float((r <= 7).mean()) if len(r) else None}
    return d


# ------------------------------------------------------------------------------------------------ report
def fmt(e, digits=2, pct=False):
    if e is None or e.get("mean") is None:
        return "n/a"
    f = (lambda x: f"{100 * x:.{digits}f} %") if pct else (lambda x: f"{x:.{digits}f}")
    return f"{f(e['mean'])} [{f(e['ci95'][0])}, {f(e['ci95'][1])}]"


def render(o, wd, tokens_path):
    a, b, c, vi = o["a_rescuable"], o["b_pdms_x100"], o["c_ep_headroom"], o["verdict_inputs"]
    L = []
    P = L.append
    P(f"**Verdict by the committed rule: {o['VERDICT']}.**\n")
    P(f"Population (MEASURED, `{tokens_path}`): {o['n_zero']} zeroed + {o['n_nonzero']} non-zeroed tokens, {o['n_logs']} logs "
      f"({o['n_logs_zero']} carry a zeroed token); {o['M']} proposals each.  Estimator: {o['estimator']}.\n")
    P("| read | value (95 % CI) | artifact |\n|---|---|---|")
    P(f"| (a) rescuable share of zeroed tokens | **{fmt(a['share'], 1, True)}** ({a['n_rescuable']}/{a['n_zero']}); bar 50 % | `table.npz` -> `readout.json` a_rescuable |")
    for nm in ("actual", "random", "oracle"):
        e = b[nm]
        P(f"| (b) {nm} PDMS x100, full-navtest mix | **{fmt(e['full_mix'])}** (zeroed {fmt(e['zero_stratum'])}; non-zeroed {fmt(e['nonzero_stratum'])}) | `readout.json` b_pdms_x100 |")
    d = o["b_paired_differences_x100"]
    for k in ("actual_minus_random", "oracle_minus_actual", "oracle_minus_random"):
        P(f"| (b) paired {k.replace('_', ' ')}, mix | {fmt(d[k]['full_mix'])} (zeroed {fmt(d[k]['zero_stratum'])}; non-zeroed {fmt(d[k]['nonzero_stratum'])}) | `readout.json` b_paired_differences_x100 |")
    P(f"| (c) EP headroom, best safe minus pick, non-zeroed tokens with a safe proposal ({c['n_with_safe']}/{c['n_nonzero']}) | **{fmt(c['primary'], 4)}** EP ({fmt(c['primary_x100'], 2)} x100) | `readout.json` c_ep_headroom |")
    P("")
    P(f"Fidelity gate: {'PASS' if vi['fidelity_gate_pass'] else 'FAIL'} (`gate_fidelity.json`).  Random < actual in the mix: {vi['random_lt_actual']} "
      f"(random {vi['random_mix']:.3f}, actual {vi['actual_mix']:.3f}).  Sanity: the mix-weighted actual {b['sanity_mix_actual_vs_banked_full_navtest']['mix_actual']:.3f} "
      f"vs the banked full-navtest {FULL_NAVTEST_ACTUAL:.3f}: banked value inside the mix CI = {b['sanity_mix_actual_vs_banked_full_navtest']['banked_inside_mix_ci']}.")
    return "\n".join(L)


# ------------------------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir")
    ap.add_argument("--tokens")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    wd = a.dir
    tpath = a.tokens or os.path.join(wd, "tokens.json")
    T = np.load(os.path.join(wd, "table.npz"))
    tj = json.load(open(tpath, encoding="utf-8"))
    token = np.array([str(t) for t in T["token"]])
    zero = np.array([tj["stratum"][t] == "zero" for t in token])
    logs = np.array([tj["token_log"][t] for t in token])
    g = json.load(open(os.path.join(wd, "gate_fidelity.json")))
    pdms = T["pdms"]
    assert np.isfinite(pdms).all() and T["valid"].all(), "table has invalid / non-finite cells -- INSTRUMENT FAILURE"
    t0 = time.time()
    holder = {}

    def extra(ctx):
        holder["d"] = descriptive(ctx, logits=T["logits"][np.argsort(token)] if "logits" in T.files else None)
        return {"DESCRIPTIVE": holder["d"]}

    o = readout(token, pdms, T["sub"], T["pick"], T["pdms_at_pick_banked"], zero, logs, fidelity_pass=bool(g["PASS"]), n_boot=a.n_boot, extra=extra)
    o["provenance"] = {"table": os.path.join(wd, "table.npz"), "tokens": tpath, "seconds": round(time.time() - t0, 1),
                       "ckpt_md5_prefix": "8e382eeb", "rule": "navsim_v1_lane"}
    json.dump(o, open(os.path.join(wd, "readout.json"), "w"), indent=1)
    open(os.path.join(wd, "readout_tables.md"), "w", encoding="utf-8").write(render(o, wd, tpath))
    print(render(o, wd, tpath))
    print("ZZC3_VERDICT", o["VERDICT"])
    return 0


# ------------------------------------------------------------------------------------------------ selftest
def _world(kind, n_per=529, M=64, n_logs=100, seed=0):
    """Synthetic table with a KNOWN answer. kind: scorer | generation | random_pick | perfect | uniform."""
    rng = np.random.default_rng(seed)
    N = 2 * n_per
    token = np.array([f"{i:016x}" for i in rng.permutation(N)])
    zero = np.arange(N) < n_per
    logs = np.array([f"log{rng.integers(0, n_logs)}" for _ in range(N)])
    sub = np.zeros((N, M, 6))
    sub[..., NC] = 1.0
    sub[..., DAC] = 1.0
    sub[..., EP] = rng.uniform(0.3, 1.0, (N, M))
    sub[..., TTC] = 1.0
    sub[..., CMF] = 1.0
    sub[..., DDC] = 1.0
    unsafe_p = {"scorer": 0.55, "generation": 0.97, "random_pick": 0.55, "perfect": 0.55, "uniform": 0.0}[kind]
    bad = rng.random((N, M)) < unsafe_p
    if kind == "generation":                                   # in the zero stratum, 90 % of tokens have NO safe proposal at all
        no_safe = zero & (rng.random(N) < 0.90)
        bad[no_safe] = True
        bad[zero & ~no_safe, 0] = False
    sub[..., DAC] = np.where(bad, 0.0, 1.0)
    pd = sub[..., NC] * sub[..., DAC] * (5 * sub[..., EP] + 5 * sub[..., TTC] + 2 * sub[..., CMF]) / 12.0
    if kind == "uniform":
        sub[:] = sub[:, :1]
        pd = pd[:, :1].repeat(M, 1)
        pick = np.zeros(N, dtype=int)
    elif kind == "perfect":
        pick = pd.argmax(1)
    elif kind == "random_pick":
        pick = rng.integers(0, M, N)
    else:                                                      # a decent-but-imperfect scorer: avoids unsafe on non-zeroed tokens, fails on zeroed
        pick = np.where(zero, np.where(bad[np.arange(N), 0], 0, 0), 0)
        for i in range(N):
            if zero[i]:
                pick[i] = int(np.flatnonzero(bad[i])[0]) if bad[i].any() else 0     # a zeroing (unsafe) pick, as in the banked data
            else:
                ok = np.flatnonzero(~bad[i])
                pick[i] = int(ok[0]) if len(ok) and rng.random() < 0.9 else int(np.flatnonzero(bad[i])[0]) if bad[i].any() else 0
    actual = pd[np.arange(N), pick]
    return token, pd, sub, pick, actual, zero, logs


def selftest():
    ok = True

    def chk(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")

    nb = 400
    # (1) scorer-bound world: rescuable high, actual (picks safe on non-zeroed) > random
    o = readout(*_world("scorer"), n_boot=nb)
    chk("scorer world -> SCORER-BOUND", o["VERDICT"] == "SCORER-BOUND",
        f"(rescuable {o['a_rescuable']['share']['mean']:.3f}, random {o['verdict_inputs']['random_mix']:.2f}, actual {o['verdict_inputs']['actual_mix']:.2f})")
    # (2) generation-bound world: rescuable ~ 0.10 of zeroed
    o2 = readout(*_world("generation", seed=1), n_boot=nb)
    chk("generation world -> GENERATION-BOUND", o2["VERDICT"] == "GENERATION-BOUND", f"(rescuable {o2['a_rescuable']['share']['mean']:.3f})")
    # (3) random-pick world: the 'actual' pick is itself random -> random ~ actual -> INSTRUMENT FAILURE by the rule, or at least never SCORER-BOUND by luck
    o3 = readout(*_world("random_pick", seed=2), n_boot=nb)
    chk("random-pick world is NOT called SCORER-BOUND", o3["VERDICT"] != "SCORER-BOUND" or o3["verdict_inputs"]["random_lt_actual"], f"-> {o3['VERDICT']}")
    # (4) fidelity gate failure -> INSTRUMENT FAILURE regardless of the numbers
    o4 = readout(*_world("scorer", seed=3), fidelity_pass=False, n_boot=nb)
    chk("fidelity failure -> INSTRUMENT FAILURE", o4["VERDICT"] == "INSTRUMENT FAILURE")
    # (5) analytic identity: perfect scorer => actual == oracle EXACTLY (diff 0, CI [0, 0]); (6) identical columns => random == actual == oracle EXACTLY
    o5 = readout(*_world("perfect", seed=4), n_boot=nb)
    d5 = o5["b_paired_differences_x100"]["oracle_minus_actual"]["full_mix"]
    chk("perfect scorer: oracle - actual == 0.0 exactly with CI [0, 0]", d5["mean"] == 0.0 and d5["ci95"] == [0.0, 0.0], f"({d5})")
    o6 = readout(*_world("uniform", seed=5), n_boot=nb)
    d6 = o6["b_paired_differences_x100"]["oracle_minus_random"]["full_mix"]
    chk("identical columns: oracle == random (to float rounding, 1e-9)", abs(d6["mean"]) < 1e-9 and max(abs(x) for x in d6["ci95"]) < 1e-9, f"({d6})")
    chk("identical columns: random >= actual -> INSTRUMENT FAILURE", o6["VERDICT"] == "INSTRUMENT FAILURE")
    # (7) rescuable is exactly the NC==1&DAC==1 definition, on a hand-built case
    tok = np.array(["a", "b", "c", "d"])
    sub = np.ones((4, 3, 6))
    sub[0, :, DAC] = 0           # zeroed token with no safe proposal
    sub[1, 1, DAC] = 0
    sub[1, 2, NC] = 0.5          # one safe proposal (k=0) among three
    pdm = np.array([[0, 0, 0], [0.9, 0, 0], [0.8, 0.8, 0.8], [0.7, 0.7, 0.7]], dtype=float)
    oo = readout(tok, pdm, sub, np.array([0, 1, 0, 0]), pdm[np.arange(4), [0, 1, 0, 0]], np.array([True, True, False, False]),
                 np.array(["L1", "L1", "L2", "L2"]), n_boot=50)
    chk("hand case: rescuable = 1 of 2 zeroed (NC=0.5 is NOT safe)", oo["a_rescuable"]["n_rescuable"] == 1 and oo["a_rescuable"]["share"]["mean"] == 0.5)
    # (8) DELIBERATE-REGRESSION ARM: swap the roles of random and actual in the verdict -> the scorer world must flip
    mut = verdict(True, o["a_rescuable"]["share"]["mean"], o["verdict_inputs"]["actual_mix"], o["verdict_inputs"]["random_mix"])
    chk("mutation (random<->actual swapped) is caught", mut == "INSTRUMENT FAILURE" and o["VERDICT"] == "SCORER-BOUND", f"(mutated -> {mut})")
    mut2 = verdict(True, 0.49, 50.0, 60.0)
    chk("verdict boundary: 0.49 -> GENERATION-BOUND, 0.50 -> SCORER-BOUND", mut2 == "GENERATION-BOUND" and verdict(True, 0.50, 50.0, 60.0) == "SCORER-BOUND")
    # (9) draws are independent of row order
    w = _world("scorer", seed=7)
    perm = np.random.default_rng(9).permutation(len(w[0]))
    oa = readout(*w, n_boot=nb)
    ob = readout(*(x[perm] for x in w), n_boot=nb)
    chk("readout is invariant to row order", oa["b_pdms_x100"]["random"]["full_mix"] == ob["b_pdms_x100"]["random"]["full_mix"])
    print("ZZC3_SELFTEST_" + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
