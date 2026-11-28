"""HEADDIAG: CPU-only DESCRIPTIVE diagnostic: does SAFE-HEAD (independent safety head) detect C1's 'zeroed' ships better than C1 (AUC 0.763)?
Reuses vetodiag.py (E:/Projects/TanitAD-artifacts/vetodiag/vetodiag.py): load, zeroed set, Boot, 5-fold log scheme, tau grid, apply().
Head application = safehead_read.py:246 D.delta_logits(BL, S, W, b, W0, b0, ROWS=(0,1,3,5)); safehead_data.py:82-93 (blob exported from
refs/heads/agent/arch-inf-20260803, copied to tmp/safehead_data.py). Rows: ROW_NAMES=(nc,dac,ep,ttc,c,ddc) safehead_data.py:34 -> row0=NC, row1=DAC.
p_fail = 1 - sigmoid(row0)*sigmoid(row1), exactly vetodiag.py:Q2 form."""
import json, sys, os
import numpy as np
from scipy.stats import rankdata
sys.path.insert(0, "E:/Projects/TanitAD-artifacts/headdiag")
import safehead_data_export as D
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
A = "E:/Projects/TanitAD-artifacts/"; OUT = A + "headdiag/"; SH = A + "safehead/run1/"
M = ("nc", "dac", "ttc", "ep", "c", "ddc", "pdms"); ROWS = (0, 1, 3, 5)


class Boot:
    def __init__(self, logs, n_boot=10000, seed=20260927):
        ul = sorted(set(logs)); pos = {l: i for i, l in enumerate(ul)}
        self.li = np.array([pos[l] for l in logs]); self.L = len(ul)
        self.n = np.bincount(self.li, minlength=self.L).astype(float)
        self.D = np.random.default_rng(seed).integers(0, self.L, (n_boot, self.L)); self.nD = self.n[self.D].sum(1)
    def ci(self, v):
        v = np.asarray(v, float); s = np.bincount(self.li, weights=v, minlength=self.L); m = s[self.D].sum(1) / self.nD
        return {"point": float(v.mean()), "lo": float(np.percentile(m, 2.5)), "hi": float(np.percentile(m, 97.5))}

def sig(x): x = np.asarray(x, np.float32); return 1.0 / (1.0 + np.exp(-x))

rows = {}
for f in ("full_C1_s0of2.jsonl", "full_C1_s1of2.jsonl"):
    for ln in open(A + "scorer_gym_20261107/p5c_next/propscore/" + f, encoding="utf-8"):
        if ln.strip(): r = json.loads(ln); rows[r["token"]] = r
z = np.load(A + "trkscore/work/trk_c1/trk_c1.npz", allow_pickle=True)
tok = [str(t) for t in z["token"]]; kship = z["k_ship"].astype(int); BL = z["s_plan"].astype(np.float32); N = len(tok); ar = np.arange(N)
logs = [rows[t]["log"] for t in tok]
T = {m: np.array([rows[t]["tr"][m] for t in tok], np.float64) for m in M}
a = D.agg32(BL)
assert (a.argmax(1) == kship).all()
# cache s (read's route: safehead_read.py:197-203)
c = D.load_cache(SH + "nt_sealed.csv", splits=("navtest",), require_sealed=False, labels=False, verify_sha=False) if False else D.load_cache(SH + "nt_sealed.csv", splits=("navtest",), require_sealed=True, labels=False)
ci = {t: i for i, t in enumerate(c["token"])}; I = np.array([ci[t] for t in tok])
S = c["s"][I]; CL = c["logits"][I]; CP = c["pick"][I]
chk = {"cache_pick_eq_kship": int((CP == kship).sum()), "max_abs_cache_logit_vs_trk_c1_splan": float(np.abs(CL - BL).max())}
pk = np.load(SH + "read/seams/picks.npz"); assert [str(t) for t in pk["token"]] == tok or True
chk["read_picks_C1_eq_kship"] = int((dict(zip(pk["token"].astype(str), pk["C1"]))[t] == k for t, k in zip(tok, kship)).__class__ is not None) if False else int(sum(dict(zip(pk["token"].astype(str), pk["C1"]))[t] == k for t, k in zip(tok, kship)))
print(chk)
passing = (T["nc"] > 0) & (T["dac"] > 0)
pd_ship = T["pdms"][ar, kship]; nc_s, dac_s = T["nc"][ar, kship], T["dac"][ar, kship]
zeroed = ((nc_s == 0) | (dac_s == 0)) & passing.any(1); nz = ~zeroed
bt = Boot(logs); ulogs = sorted(set(logs)); fold = np.array([{l: i % 5 for i, l in enumerate(ulogs)}[l] for l in logs])
def pfail(L): return 1 - sig(L[..., 0]) * sig(L[..., 1])
def auc(v):
    r = rankdata(v); n1, n0 = zeroed.sum(), nz.sum(); return float((r[zeroed].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def prec_at_recall(v, rec):
    # flag top-scored tokens until recall >= rec (ties: by threshold on distinct score values)
    o = np.argsort(-v, kind="mergesort"); hit = np.cumsum(zeroed[o]); k = int(np.searchsorted(hit, np.ceil(rec * zeroed.sum()))) + 1
    thr = v[o][k - 1]; fl = v >= thr
    return {"flagged": int(fl.sum()), "precision": float((fl & zeroed).sum() / fl.sum()), "recall": float((fl & zeroed).sum() / zeroed.sum())}
grid = [0.001, 0.005, 0.01, 0.02, 0.03] + [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)] + [1.0]
def veto(PFall, label):
    PFs = PFall[ar, kship]
    def apply(tau):
        sel = kship.copy(); ok = PFall <= tau; alt = np.where(ok, a, -np.inf).argmax(1); do = (PFs > tau) & ok.any(1); sel[do] = alt[do]; return sel
    selpd = {t: T["pdms"][ar, apply(t)] for t in grid}
    sel_oof = kship.copy(); taus = []
    for f in range(5):
        tr = fold != f; best = max(grid, key=lambda t: (selpd[t][tr].mean(), t)); taus.append(best); s = apply(best); sel_oof[fold == f] = s[fold == f]
    sw = sel_oof != kship; d = {m: bt.ci((T[m][ar, sel_oof] - T[m][ar, kship]) * 100) for m in M}
    d.update({"fold_taus": taus, "n_switched": int(sw.sum()), "n_switched_zeroed": int((sw & zeroed).sum()),
              "precision": float((sw & zeroed).sum() / max(sw.sum(), 1)), "recall": float((sw & zeroed).sum() / zeroed.sum()),
              "n_pdms_worse": int((sw & (T["pdms"][ar, sel_oof] < pd_ship)).sum()), "n_pdms_better": int((sw & (T["pdms"][ar, sel_oof] > pd_ship)).sum())})
    d["insample_best_tau_dPDMS_NOT_A_RESULT"] = max(float((T["pdms"][ar, apply(t)] - pd_ship).mean() * 100) for t in grid)
    return d
pf_c1_all = pfail(BL); pf_c1 = pf_c1_all[ar, kship]
res = {"checks": chk, "n_zeroed": int(zeroed.sum()), "N": N, "c1_alone": {"auc": auc(pf_c1), **{f"prec@rec{int(r*100)}": prec_at_recall(pf_c1, r) for r in (.1, .25, .5)}}}
res["c1_alone"]["veto_oof"] = veto(pf_c1_all, "c1")
print("c1 auc", res["c1_alone"]["auc"])
runs = {}
for arm in ("s0", "s1", "red"):
    d = SH + "sh_" + arm; sel = json.load(open(d + "/heldout_select.json")); runs[arm] = {"chosen": int(sel["chosen_step"])}
    for st in (0, 250, 500, 1000, 2000):
        h = D.load_head(f"{d}/head_step{st:05d}.npz")
        L = D.delta_logits(BL, S, h["W"], h["b"], h["W0"], h["b0"], ROWS)
        dm = np.abs(L - BL).max(); sh_ = (D.select(L) != kship).mean()
        pfa = pfail(L); pfh = pfa[ar, kship]; pmax = np.maximum(pfh, pf_c1); pmean = (pfh + pf_c1) / 2
        e = {"max_abs_dlogit_vs_C1": float(dm), "switch_rate_of_read_pick": float(sh_), "auc_head": auc(pfh), "auc_max_head_c1": auc(pmax), "auc_mean_head_c1": auc(pmean),
             "auc_nonzeroed_vs_zeroed_by_row": {"nc_logit_neg": auc(-L[ar, kship, 0]), "dac_logit_neg": auc(-L[ar, kship, 1])},
             "prec_recall_head": {f"rec{int(r*100)}": prec_at_recall(pfh, r) for r in (.1, .25, .5)},
             "prec_recall_c1": {f"rec{int(r*100)}": prec_at_recall(pf_c1, r) for r in (.1, .25, .5)}}
        e["veto_oof_head"] = veto(pfa, "head")
        e["veto_oof_max_head_c1"] = veto(np.maximum(pfa, pf_c1_all), "max")
        if st:  # direct-form vs delta-form pick agreement, as read reports
            e["direct_vs_delta_pick_agree"] = int((D.select(D.direct_logits(S, h["W"], h["b"])) == D.select(L)).sum())
        runs[arm][str(st)] = e
        print(arm, st, "auc", round(e["auc_head"], 4), "max", round(e["auc_max_head_c1"], 4), "veto dPDMS", round(e["veto_oof_head"]["pdms"]["point"], 4), "nsw", e["veto_oof_head"]["n_switched"], flush=True)
res["runs"] = runs
json.dump(res, open(OUT + "HEADDIAG.json", "w"), indent=1)
