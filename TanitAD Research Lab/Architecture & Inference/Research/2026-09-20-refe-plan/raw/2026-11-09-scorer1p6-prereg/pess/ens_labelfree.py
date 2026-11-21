"""LABEL-FREE (reads no navtest truth, no K key): how often would an ensemble / pessimistic selector over P6's fine-tunes pick a
different proposal than P6 on E3? Logits only. Decides nothing; sizes a candidate lever before a card is written."""
import json, sys
import numpy as np
R = "F:/tanitad_scratch/p6arms/"
ARMS = {"P6": "pass/P6_control", "C0": "pass/C0", "C0r": "pass/C0r", "S1E": "pass/S1E", "C0r2": "pass_rep_c0r2_s1e1_s1z1/C0r2",
        "S1E1": "pass_rep_c0r2_s1e1_s1z1/S1E1", "S1Z1": "pass_rep_c0r2_s1e1_s1z1/S1Z1", "S1Z": "pass_s1z/S1Z", "S1ZSM": "pass_s1zsm/S1ZSM"}
NC, DAC, EP, TTC, C, DDC = range(6)
def rule(x): return x[..., NC] * x[..., DAC] * (5 * x[..., EP] + 5 * x[..., TTC] + 2 * x[..., C]) / 12
tok0 = prop0 = None; P = {}
for a, d in ARMS.items():
    z = np.load(R + d + "/e3_logits.npz")
    if tok0 is None: tok0 = z["token"]; prop0 = z["proposals"]
    assert (z["token"] == tok0).all(), a
    assert np.array_equal(z["proposals"], prop0), a  # scorer-only arms keep P6's proposals bit for bit
    P[a] = 1 / (1 + np.exp(-z["logits"].astype(np.float64)))
del prop0
names = list(P); FT = [a for a in names if a != "P6"]
pick = {a: rule(P[a]).argmax(1) for a in names}
N = len(tok0); out = {"n_tokens": int(N), "members": names, "note": "label-free, logits only; no truth read; no K key"}
out["changed_vs_P6_single"] = {a: round(float((pick[a] != pick["P6"]).mean()), 4) for a in FT}
S = np.stack([P[a] for a in FT])            # [m, N, 64, 6]
SA = np.stack([P[a] for a in names])        # with P6
cand = {}
cand["mean_FT"] = S.mean(0)
cand["mean_all"] = SA.mean(0)
def pess(X):
    Y = X.mean(0).copy()
    for h in (NC, DAC, TTC): Y[..., h] = X[..., h].min(0)
    return Y
cand["pess_min_FT"] = pess(S)
cand["pess_min_all"] = pess(SA)
L = np.log(np.clip(S, 1e-7, 1 - 1e-7)) - np.log(np.clip(1 - S, 1e-7, 1))
mu, sd = L.mean(0), L.std(0)
for k in (1.0, 2.0):
    Y = cand["mean_FT"].copy()
    for h in (NC, DAC, TTC): Y[..., h] = 1 / (1 + np.exp(-(mu[..., h] - k * sd[..., h])))
    cand[f"lcb_k{k:g}_FT"] = Y
pk = {}
for c, Y in cand.items():
    pk[c] = rule(Y).argmax(1)
    out[f"changed_vs_P6[{c}]"] = round(float((pk[c] != pick["P6"]).mean()), 4)
    out[f"changed_vs_meanFT[{c}]"] = round(float((pk[c] != pk.get("mean_FT", pk[c])).mean()), 4)
ix = np.arange(N); p6p = pick["P6"]
for h, n in ((NC, "NC"), (DAC, "DAC"), (TTC, "TTC")):
    s = sd[ix, p6p, h]
    out[f"logit_sd_at_P6_pick[{n}]"] = {q: round(float(np.quantile(s, q)), 4) for q in (0.5, 0.9, 0.99)}
    out[f"logit_sd_all[{n}]"] = {q: round(float(np.quantile(sd[..., h], q)), 4) for q in (0.5, 0.9, 0.99)}
json.dump(out, open(sys.argv[1], "w"), indent=1); print(json.dumps(out, indent=1))
