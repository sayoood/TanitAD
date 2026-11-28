"""VETODIAG: CPU-only descriptive diagnostic on the C1 navtest harness-exact table. DIAGNOSTIC, not a result.
Boot class copied from p5c_read.py (refs/heads/agent/arch-inf-20260803:
 TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/eval/p5c/p5c_read.py), seed 20260927, 10000 draws.
agg32 from refe/safehead_data.py:70-75; column order ROW_NAMES = (nc, dac, ep, ttc, c, ddc)."""
import json, sys
import numpy as np

A = "E:/Projects/TanitAD-artifacts/"
OUT = A + "vetodiag/"
M = ("nc", "dac", "ttc", "ep", "c", "ddc", "pdms")


class Boot:
    def __init__(self, logs, n_boot=10000, seed=20260927):
        ul = sorted(set(logs)); pos = {l: i for i, l in enumerate(ul)}
        self.li = np.array([pos[l] for l in logs]); self.L = len(ul)
        self.n = np.bincount(self.li, minlength=self.L).astype(float)
        self.D = np.random.default_rng(seed).integers(0, self.L, (n_boot, self.L))
        self.nD = self.n[self.D].sum(1)

    def ci(self, v):
        v = np.asarray(v, float)
        s = np.bincount(self.li, weights=v, minlength=self.L)
        m = s[self.D].sum(1) / self.nD
        return {"point": float(v.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}


def agg32(L):
    L = np.asarray(L, np.float32)
    p = 1.0 / (1.0 + np.exp(-L))
    return (p[..., 0] * p[..., 1]) ** 1.0 * (5 * p[..., 2] + 5.0 * p[..., 3] + 2 * p[..., 4]) / (7 + 5.0)


def sig(x):
    x = np.asarray(x, np.float32)
    return 1.0 / (1.0 + np.exp(-x))


# ---- load
rows = {}
for f in ("full_C1_s0of2.jsonl", "full_C1_s1of2.jsonl"):
    for ln in open(A + "scorer_gym_20261107/p5c_next/propscore/" + f, encoding="utf-8"):
        if ln.strip():
            r = json.loads(ln); rows[r["token"]] = r
z = np.load(A + "trkscore/work/trk_c1/trk_c1.npz", allow_pickle=True)
tok = [str(t) for t in z["token"]]
kship = z["k_ship"].astype(int)
S = z["s_plan"].astype(np.float32)
N = len(tok)
assert N == 12146 and all(t in rows for t in tok), (N, len(rows))
logs = [rows[t]["log"] for t in tok]
T = {m: np.array([rows[t]["tr"][m] for t in tok], np.float64) for m in M}
ar = np.arange(N)
a = agg32(S)
ok1 = int((a.argmax(1) == kship).sum())
pd_ship = T["pdms"][ar, kship]
mean_ship = float(pd_ship.mean() * 100)
print("argmax match", ok1, "mean ship pdms", mean_ship)
chk = {"argmax_match": ok1, "n": N, "mean_ship_pdms": mean_ship}
if ok1 != N or abs(mean_ship - 92.5548) > 5e-5:
    chk["STOP"] = True
    json.dump(chk, open(OUT + "VETODIAG.json", "w"), indent=1)
    print("CHECK FAILED"); sys.exit(1)

bt = Boot(logs)
nc_s, dac_s = T["nc"][ar, kship], T["dac"][ar, kship]
passing = (T["nc"] > 0) & (T["dac"] > 0)
zeroed = ((nc_s == 0) | (dac_s == 0)) & passing.any(1)
res = {"checks": chk}
# Q1
best_pd = np.where(passing, T["pdms"], -1).max(1)  # best passing proposal pdms
best_any = T["pdms"].max(1)
regret_pass = float(((best_pd - pd_ship)[zeroed]).sum() / N * 100)
res["Q1"] = {"n_zeroed": int(zeroed.sum()), "n_nc0": int((zeroed & (nc_s == 0)).sum()),
             "n_dac0": int((zeroed & (dac_s == 0)).sum()), "n_both": int((zeroed & (nc_s == 0) & (dac_s == 0)).sum()),
             "regret_best_passing_pdms_x100_over_N": regret_pass,
             "regret_best_any_pdms_x100_over_N": float(((best_any - pd_ship)[zeroed]).sum() / N * 100),
             "n_ship_zero_but_no_passing_alternative": int((((nc_s == 0) | (dac_s == 0)) & ~passing.any(1)).sum())}
print(res["Q1"])
# Q2
pn, pdc = sig(S[ar, kship, 0]), sig(S[ar, kship, 1])
pf = 1 - pn * pdc
qs = [0.05, 0.25, 0.5, 0.75, 0.95]
nz = ~zeroed
def qd(v, m): return {str(q): float(np.quantile(v[m], q)) for q in qs}
from scipy.stats import rankdata
r = rankdata(pf); n1, n0 = zeroed.sum(), nz.sum()
auc = float((r[zeroed].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
res["Q2"] = {"sigma_nc_zeroed": qd(pn, zeroed), "sigma_nc_nonzeroed": qd(pn, nz),
             "sigma_dac_zeroed": qd(pdc, zeroed), "sigma_dac_nonzeroed": qd(pdc, nz),
             "p_fail_zeroed": qd(pf, zeroed), "p_fail_nonzeroed": qd(pf, nz), "auc_pfail": auc}
print(res["Q2"]["auc_pfail"])
# Q3
PF = 1 - sig(S[..., 0]) * sig(S[..., 1])  # [N,64]


def apply(tau, idx=None):
    sel = kship.copy()
    sw = PF[ar, kship] > tau
    ok = PF <= tau
    cand = np.where(ok, a, -np.inf)
    alt = cand.argmax(1)
    has = ok.any(1)
    do = sw & has
    sel[do] = alt[do]
    return sel, do


grid = [0.001, 0.005, 0.01, 0.02, 0.03] + [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)] + [1.0]
ulogs = sorted(set(logs)); fold_of = {l: i % 5 for i, l in enumerate(ulogs)}
fold = np.array([fold_of[l] for l in logs])
selpd = {}
for t in grid:
    s, _ = apply(t); selpd[t] = T["pdms"][ar, s]
sel_oof = kship.copy(); taus = []
for f in range(5):
    tr = fold != f
    best = max(grid, key=lambda t: (selpd[t][tr].mean(), t))  # tie -> larger tau (fewer switches)
    taus.append(best)
    s, _ = apply(best)
    te = fold == f
    sel_oof[te] = s[te]
res["Q3"] = {"fold_taus": taus}
def summarize(sel):
    d = {}
    for m in M:
        d[m] = bt.ci((T[m][ar, sel] - T[m][ar, kship]) * 100)
    sw = sel != kship
    d["n_switched"] = int(sw.sum()); d["n_switched_zeroed"] = int((sw & zeroed).sum())
    d["precision"] = float((sw & zeroed).sum() / max(sw.sum(), 1)); d["recall"] = float((sw & zeroed).sum() / zeroed.sum())
    d["mean_pdms_x100"] = float(T["pdms"][ar, sel].mean() * 100)
    d["n_switched_pdms_worse"] = int((sw & (T["pdms"][ar, sel] < pd_ship)).sum())
    d["n_switched_pdms_better"] = int((sw & (T["pdms"][ar, sel] > pd_ship)).sum())
    return d
res["Q3"]["oof"] = summarize(sel_oof)
curve = []
for t in grid:
    s, _ = apply(t); sw = s != kship
    curve.append({"tau": t, "n_switched": int(sw.sum()), "dPDMS": float((T["pdms"][ar, s] - pd_ship).mean() * 100),
                  "dEP": float((T["ep"][ar, s] - T["ep"][ar, kship]).mean() * 100),
                  "precision": float((sw & zeroed).sum() / max(sw.sum(), 1))})
res["Q3"]["insample_curve_NOT_A_RESULT"] = curve
# Q4
okz = zeroed[:, None] & passing
selA = kship.copy(); selB = kship.copy()
altA = np.where(passing, a, -np.inf).argmax(1)
altB = np.where(passing, T["pdms"], -np.inf).argmax(1)
selA[zeroed] = altA[zeroed]; selB[zeroed] = altB[zeroed]
res["Q4"] = {"a_agg32_best_passing": summarize(selA), "b_true_best_passing": summarize(selB)}
json.dump(res, open(OUT + "VETODIAG.json", "w"), indent=1)
o = res["Q3"]["oof"]
print(json.dumps({k: o[k] for k in ("pdms", "ep", "nc", "dac", "ttc", "c", "n_switched", "precision", "recall")}, indent=0))
print(res["Q4"]["a_agg32_best_passing"]["pdms"], res["Q4"]["a_agg32_best_passing"]["ep"], res["Q4"]["b_true_best_passing"]["pdms"])
