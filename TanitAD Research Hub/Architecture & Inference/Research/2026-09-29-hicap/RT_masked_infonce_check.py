"""Red-team check (2026-09-29): does Proposition 'Per-node InfoNCE optimum' (f* = PMI + c(s)) hold for the
loss HiCAP actually specifies -- in-batch InfoNCE WITH CLM's false-negative (same-action) mask?

Tabular critic f(s,a) (no representational limit), two states, three actions, marginal (0.70,0.20,0.10) --
the paper's own worked example -- trained by Adam on sampled batches. Compares unmasked vs masked, row-only
(state->action, the direction Prop. 1 analyses) vs bidirectional (CLM's symmetric loss).
Evidence class: MEASURED (this script; numpy only; seed 0).
"""
import json, numpy as np
P = np.array([[0.6, 0.3, 0.1], [0.8, 0.1, 0.1]]); ps = np.array([0.5, 0.5]); marg = ps @ P

def softmax_rows(L):
    g = np.exp(L - L.max(1, keepdims=True)); return g / g.sum(1, keepdims=True)

def run(mask, bidir, B=64, steps=15000, lr=0.05, seed=0):
    rng = np.random.default_rng(seed); f = np.zeros((2, 3)); m = np.zeros_like(f); v = np.zeros_like(f)
    for t in range(1, steps + 1):
        s = rng.choice(2, B, p=ps); a = np.array([rng.choice(3, p=P[i]) for i in s])
        L = f[s][:, a]
        if mask:
            same = (a[:, None] == a[None, :]) & ~np.eye(B, dtype=bool); L = np.where(same, -1e9, L)
        gr = softmax_rows(L); gr[np.arange(B), np.arange(B)] -= 1
        gL = gr / B
        if bidir:
            gc = softmax_rows(L.T); gc[np.arange(B), np.arange(B)] -= 1
            gL = (gr + gc.T) / (2 * B)
        G = np.zeros_like(f)
        for i in range(B):
            np.add.at(G[s[i]], a, gL[i])
        m = 0.9 * m + 0.1 * G; v = 0.999 * v + 0.001 * G ** 2
        f -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return f - f.mean(1, keepdims=True)

pmi = np.log(P / marg); pmi -= pmi.mean(1, keepdims=True)
out = {"P": P.tolist(), "marginal": marg.tolist(), "pmi_centred": np.round(pmi, 3).tolist(), "runs": {}}
for mask in (False, True):
    for bidir in (False, True):
        f = run(mask, bidir)
        implied = softmax_rows(f + np.log(marg))
        out["runs"][f"mask={mask},bidir={bidir}"] = {
            "critic_centred": np.round(f, 3).tolist(), "argmax": f.argmax(1).tolist(),
            "softmax(f+log q) i.e. beta=1 corrected": np.round(implied, 3).tolist()}
print(json.dumps(out, indent=1))
