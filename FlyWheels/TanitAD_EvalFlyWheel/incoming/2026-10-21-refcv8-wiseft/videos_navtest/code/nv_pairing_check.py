"""nv_pairing_check.py -- positive control that each plan array is the one the scorer scored: Spearman(plan path length over 4.0 s, the scorer row's
`progress_raw_m[1]` = the AGENT proposal's raw progress in metres) over all 12,146 tokens, against a MISPAIRED control (plan order permuted).
Matched must be strongly positive, mispaired ~0.  (Plan progress vs the normalised EP column was tried first and is NOT a clean control: EP is a ratio to the
PDM reference progress, which co-varies with speed -- matched rho 0.08-0.10, prior -0.03 -- so it cannot discriminate; kept in the JSON as `ep_rho`.)"""
import json, sys
import numpy as np
sys.path.insert(0, ".")
import nv_select as S
B = S.load_bars()
rng = np.random.default_rng(0)
out = {}
def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])
ROWS = {a: c[:-4] + "_rows.jsonl" for a, c in (("W", S.CSV["W"]), ("E_LEGAL", S.CSV_EMA["LEGAL"]), ("E_RT", S.CSV_EMA["RT"]), ("P", S.CSV["P"]))}
for arm, seam, csv in (("W", S.SEAMS["W"], S.CSV["W"]), ("E_LEGAL", S.SEAMS["E_LEGAL"], S.CSV_EMA["LEGAL"]), ("E_RT", S.SEAMS["E_RT"], S.CSV_EMA["RT"]), ("P", S.SEAMS["P"], S.CSV["P"])):
    rows = {}
    for ln in open(ROWS[arm], encoding="utf-8"):
        r = json.loads(ln); rows[r["token"]] = float(r["progress_raw_m"][1])
    z = np.load(seam, allow_pickle=False)
    toks = [str(t) for t in z["token"]]
    sc = B.read_scores_csv(csv)
    keep = [i for i, t in enumerate(toks) if t in sc.index and str(z["source"][i]) != "cv_standin"]
    def plen(P):
        xy = np.vstack([[0.0, 0.0], np.asarray(P, np.float64)[:, :2]])
        return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
    prog = np.array([plen(z["poses"][i]) for i in keep])
    ep = np.array([float(sc.loc[toks[i], "EP"]) for i in keep])
    ag = np.array([rows[toks[i]] for i in keep])
    r = spearman(prog, ag)
    perm = rng.permutation(len(prog))
    r0 = spearman(prog[perm], ag)
    out[arm] = {"n": len(keep), "n_tokens_seam": len(toks), "sources": sorted(set(str(s) for s in z["source"])), "spearman_planlen_vs_scorer_progress_raw_m_matched": round(r, 4),
                "spearman_mispaired_control": round(r0, 4), "ep_rho": round(spearman(prog, ep), 4)}
    print(arm, out[arm])
json.dump(out, open("C:/Users/Admin/refcv8_pf/navsim_video/work/pairing_check.json", "w"), indent=1)
