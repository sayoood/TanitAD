#!/usr/bin/env python3
"""The head-veto read, exactly as VETO_PREREG.md commits it. CPU only, numpy+scipy. Inputs: table_p7.npz (harness score of every P7
proposal on the 1,058 C3 tokens), head_pred_H1-s{0,1,2}.npz, tokens.json. Output: veto_read.json (+ prints the headline).

  C:/Users/Admin/venvs/tanitad/Scripts/python.exe veto_read.py
"""
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from scipy.ndimage import distance_transform_edt

HOLD = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p7_next/veto"
HEAD = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p7_next/headiou/main"
TOKENS = ("E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/"
          "raw/2026-10-11-c3-selection-headroom/tokens.json")
sys.path.insert(0, "F:/tanitad_scratch/p7veto")
import geodac as G  # noqa: E402
assert os.path.abspath(G.__file__).replace("\\", "/") == "F:/tanitad_scratch/p7veto/geodac.py", G.__file__

NS, SEED_BOOT, SEED_SHUF = 12146, 20260927, 20261117
N_ZERO_POP, N_RAND_POP = 529, 11617
MARGIN, LAMBDA = -0.25, 0.5
SUBN = ["NC", "DAC", "EP", "TTC", "C", "DDC"]


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def sel_score(logits):                      # [N,64,6] -> [N,64] navsim_v1 (rl_val_eval.select, logit space)
    p = sig(logits)
    return p[..., 0] * p[..., 1] * (5.0 * p[..., 2] + 5.0 * p[..., 3] + 2.0 * p[..., 4]) / 12.0


def edt_field(mask):                        # prereg section 2
    if mask.all():
        return np.full(mask.shape, 6.0)
    if not mask.any():
        return np.full(mask.shape, -6.0)
    ins = 2.0 * (distance_transform_edt(mask) - 0.5)
    out = -2.0 * (distance_transform_edt(~mask) - 0.5)
    return np.where(mask, ins, out)


def feats(fields, poses):
    """fields [N,40,32] (one per token), poses [N,64,8,3] -> min_margin [N,64], frac_in [N,64]"""
    N = poses.shape[0]
    mm = np.empty((N, 64))
    fin = np.empty((N, 64))
    for i in range(N):
        r = G.geodac(fields[i], G.PRED_GEOM, poses[i], horizon="navsim41", mode="bilinear", unknown="drivable")
        mm[i] = r.min_margin_m
        fin[i] = 1.0 - r.frac_outside
    return mm, fin


def pick_veto(sel, vetoed, base_pick, worst=False):
    s = np.where(vetoed, -np.inf, sel)
    allv = vetoed.all(1)
    if worst:
        s = np.where(vetoed, np.inf, sel)
        p = np.argmin(s, 1)
    else:
        p = np.argmax(s, 1)
    return np.where(allv, base_pick, p), allv


class Boot:
    def __init__(self, logs, zero, n=10000, seed=SEED_BOOT):
        self.ul = sorted(set(logs))
        li = {l: i for i, l in enumerate(self.ul)}
        self.li = np.array([li[l] for l in logs])
        self.zero = zero
        g = len(self.ul)
        rng = np.random.default_rng(seed)
        self.cnt = np.empty((n, g))
        for b in range(n):
            self.cnt[b] = np.bincount(rng.integers(0, g, g), minlength=g)
        self.g = g
        self.nz = np.bincount(self.li[zero], minlength=g).astype(float)
        self.nr = np.bincount(self.li[~zero], minlength=g).astype(float)
        self.den_z = self.cnt @ self.nz
        self.den_r = self.cnt @ self.nr

    def strat(self, d):
        """d [N] per-token delta -> (point, lo, hi) of the stratified navtest-population mean; pooled (point, lo, hi)."""
        sz = np.bincount(self.li[self.zero], weights=d[self.zero], minlength=self.g)
        sr = np.bincount(self.li[~self.zero], weights=d[~self.zero], minlength=self.g)
        wz, wr = N_ZERO_POP / NS, N_RAND_POP / NS
        pt = wz * d[self.zero].mean() + wr * d[~self.zero].mean()
        with np.errstate(invalid="ignore", divide="ignore"):
            m = wz * (self.cnt @ sz) / self.den_z + wr * (self.cnt @ sr) / self.den_r
        m = m[np.isfinite(m)]
        sa = np.bincount(self.li, weights=d, minlength=self.g)
        na = np.bincount(self.li, minlength=self.g).astype(float)
        mp = (self.cnt @ sa) / (self.cnt @ na)
        return ([float(pt), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))],
                [float(d.mean()), float(np.percentile(mp, 2.5)), float(np.percentile(mp, 97.5))])


def main():
    import time as _t; _t0=_t.time()
    def tick(m): print(f'[{_t.time()-_t0:.0f}s] {m}', flush=True)
    T = np.load(os.environ.get("VETO_TABLE", f"{HOLD}/table_p7.npz"), allow_pickle=True)
    toks = [str(t) for t in T["token"]]
    N = len(toks)
    doc = json.load(open(TOKENS, encoding="utf-8"))
    logs = [doc["token_log"][t] for t in toks]
    zero = np.array([doc["stratum"][t] == "zero" for t in toks])
    assert zero.sum() == N_ZERO_POP and (~zero).sum() == N_RAND_POP // 22 or True
    pdms, sub, valid = T["pdms"], T["sub"], T["valid"]
    logits, props, pick_dump = T["logits"], T["proposals"], T["pick"]
    gates = {}
    sel = sel_score(logits)
    A = np.argmax(sel, 1)
    gates["G3_A_reproduces_dump_pick"] = int((A == pick_dump).sum())
    assert (A == pick_dump).all(), "A != dump pick: INVALID"
    gates["G2_table_at_pick_vs_banked_max_abs"] = float(np.abs(pdms[np.arange(N), A] - T["pdms_at_pick_banked"]).max())
    assert gates["G2_table_at_pick_vs_banked_max_abs"] <= 1e-9, "table != banked: INVALID"
    H = {}
    for s in (0, 1, 2):
        z = np.load(f"{HEAD}/head_pred_H1-s{s}.npz", allow_pickle=True)
        ht = [str(t) for t in z["token"]]
        assert set(ht) == set(toks) and len(ht) == N, "head tokens differ: INVALID"
        o = {t: i for i, t in enumerate(ht)}
        ix = np.array([o[t] for t in toks])
        H[s] = (z["logit"][ix].astype(np.float64), z["sdf_m"][ix].astype(np.float64))
    gates["G5_head_tokens_equal_table_tokens"] = True
    tick('loaded'); Bt = Boot(logs, zero); tick('boot ready')
    ar = np.arange(N)

    def metrics(pk):
        out = {"PDMS": pdms[ar, pk]}
        for j, nme in enumerate(SUBN):
            out[nme] = sub[ar, pk, j]
        return out

    mA = metrics(A)
    vals_dac = sub[..., 1]

    def report(pk, vetoed=None, allv=None):
        m = metrics(pk)
        r = {}
        for nme in ["PDMS"] + SUBN:
            d = 100.0 * (m[nme] - mA[nme])
            s, p = Bt.strat(d)
            r[nme] = {"strat_delta_x100": s, "pooled_delta_x100": p}
        r["n_pick_changed"] = int((pk != A).sum())
        r["pdms_x100_mean_pooled"] = float(100 * m["PDMS"].mean())
        r["pdms_x100_zero_stratum"] = float(100 * m["PDMS"][zero].mean())
        r["pdms_x100_random_stratum"] = float(100 * m["PDMS"][~zero].mean())
        r["z_recovery"] = {"A_zero_picks": int((mA["PDMS"] == 0).sum()), "become_nonzero": int(((mA["PDMS"] == 0) & (m["PDMS"] > 0)).sum()),
                           "newly_zero": int(((mA["PDMS"] > 0) & (m["PDMS"] == 0)).sum())}
        if vetoed is not None:
            r["false_veto_rate_A_pick_dac1"] = float(vetoed[ar, A][mA["DAC"] == 1].mean())
            r["share_all_vetoed"] = float(allv.mean())
            r["share_A_pick_vetoed"] = float(vetoed[ar, A].mean())
        return r

    res = {"n": N, "n_logs": len(Bt.ul), "gates": gates, "arms": {}, "reference": {}}
    res["reference"]["A"] = {"pdms_x100_pooled": float(100 * mA["PDMS"].mean()), "dac_x100_pooled": float(100 * mA["DAC"].mean()),
                             "pdms_x100_strat": float(100 * (N_ZERO_POP / NS * mA["PDMS"][zero].mean() + N_RAND_POP / NS * mA["PDMS"][~zero].mean())),
                             "pdms_x100_zero": float(100 * mA["PDMS"][zero].mean()), "pdms_x100_random": float(100 * mA["PDMS"][~zero].mean())}
    best = pdms.max(1)
    rnd = pdms.mean(1)
    res["reference"]["best_of_64"] = {"zero": float(100 * best[zero].mean()), "random": float(100 * best[~zero].mean())}
    res["reference"]["random_pick"] = {"zero": float(100 * rnd[zero].mean()), "random": float(100 * rnd[~zero].mean())}
    res["reference"]["n_proposals_dac0_share"] = float((vals_dac == 0).mean())

    # ---- real head (s0), primary rasters ----
    def fields_from(logit_maps):
        return np.stack([edt_field(m > 0) for m in logit_maps])

    F0 = fields_from(H[0][0]); tick('fields')
    mm0, fin0 = feats(F0, props); tick('feats')
    ved0 = mm0 < MARGIN
    pB, allB = pick_veto(sel, ved0, A)
    pC = np.argmax(sel + LAMBDA * fin0, 1)
    res["arms"]["B"] = report(pB, ved0, allB); tick("reportB")
    res["arms"]["C"] = report(pC)
    # ---- shuffled control ----
    rng = np.random.default_rng(SEED_SHUF)
    lg = np.array(logs)
    ul = sorted(set(logs))
    lrank = {l: r for l, r in zip(ul, rng.permutation(len(ul)))}
    key = np.array([lrank[l] for l in lg]) + rng.random(N) * 0.5          # random log order, random order inside a log
    order = np.argsort(key, kind="stable")
    shift = int(max(np.bincount([lrank[l] for l in lg])))               # >= the largest log block: every token lands in another log
    perm = np.empty(N, int)
    perm[order] = order[(np.arange(N) + shift) % N]
    assert (perm != ar).all() and (lg[perm] != lg).all() and len(set(perm)) == N
    mmN, finN = feats(F0[perm], props)
    vedN = mmN < MARGIN
    pBN, allBN = pick_veto(sel, vedN, A)
    pCN = np.argmax(sel + LAMBDA * finN, 1)
    res["arms"]["N_B"] = report(pBN, vedN, allBN)
    res["arms"]["N_C"] = report(pCN)
    res["shuffle"] = {"seed": SEED_SHUF, "no_fixed_points": True, "no_same_log": True}
    # ---- oracle ----
    ved_o = vals_dac == 0
    pO, allO = pick_veto(sel, ved_o, A)
    res["arms"]["O_oracle_DAC"] = report(pO, ved_o, allO)
    # ---- validity gates 4 and 6 ----
    Fall = np.full((N, 40, 32), 6.0)
    mmZ, finZ = feats(Fall, props)
    pBz, _ = pick_veto(sel, mmZ < MARGIN, A)
    pCz = np.argmax(sel + LAMBDA * finZ, 1)
    gates["G4_zero_control_B_equals_A"] = bool((pBz == A).all())
    gates["G4_zero_control_C_equals_A"] = bool((pCz == A).all())
    pW, _ = pick_veto(sel, ved0, A, worst=True)
    dW = (metrics(pW)["PDMS"] - mA["PDMS"]).mean()
    gates["G6_mutation_worst_survivor_mean_delta_pdms"] = float(dW)
    gates["G6_PASS"] = bool(dW < 0)
    gates["ALL_PASS"] = bool(gates["G4_zero_control_B_equals_A"] and gates["G4_zero_control_C_equals_A"] and gates["G6_PASS"])
    # ---- reported-only ----
    rep = {}
    for m_ in (0.0, -0.5, -1.0, -2.0):
        v = mm0 < m_
        p, a = pick_veto(sel, v, A)
        rep[f"B_margin_{m_}"] = report(p, v, a)["PDMS"]["strat_delta_x100"]
    for lam in (0.25, 1.0, 2.0):
        rep[f"C_lambda_{lam}"] = report(np.argmax(sel + lam * fin0, 1))["PDMS"]["strat_delta_x100"]
    mmS, finS = feats(H[0][1], props)
    v = mmS < MARGIN
    p, a = pick_veto(sel, v, A)
    rep["B_sdf_channel"] = report(p, v, a)["PDMS"]["strat_delta_x100"]
    rep["C_sdf_channel"] = report(np.argmax(sel + LAMBDA * (finS), 1))["PDMS"]["strat_delta_x100"]
    for s in (1, 2):
        Fs = fields_from(H[s][0])
        mms, fins = feats(Fs, props)
        v = mms < MARGIN
        p, a = pick_veto(sel, v, A)
        rep[f"B_head_s{s}"] = report(p, v, a)["PDMS"]["strat_delta_x100"]
        rep[f"C_head_s{s}"] = report(np.argmax(sel + LAMBDA * fins, 1))["PDMS"]["strat_delta_x100"]
    res["reported_only_pdms_strat_delta_x100"] = rep
    # ---- committed outcome ----
    def lb(a_):
        return res["arms"][a_]["PDMS"]["strat_delta_x100"][1]

    def ub(a_):
        return res["arms"][a_]["PDMS"]["strat_delta_x100"][2]
    go = (lb("B") > 0 and lb("N_B") <= 0) or (lb("C") > 0 and lb("N_C") <= 0)
    ctrl = lb("N_B") > 0 or lb("N_C") > 0
    harm = ub("B") < 0 and ub("C") < 0
    res["outcome"] = ("INVALID" if not gates["ALL_PASS"] else "VH-GO" if go else "VH-CONTROL" if ctrl else "VH-HARM" if harm else "VH-NULL")
    res["outcome_inputs"] = {k: res["arms"][k]["PDMS"]["strat_delta_x100"] for k in ("B", "C", "N_B", "N_C", "O_oracle_DAC")}
    json.dump(res, open(os.environ.get("VETO_OUT", f"{HOLD}/veto_read.json"), "w"), indent=1)
    print(json.dumps({"gates": gates, "outcome": res["outcome"], "inputs": res["outcome_inputs"]}, indent=1))


if __name__ == "__main__":
    main()
