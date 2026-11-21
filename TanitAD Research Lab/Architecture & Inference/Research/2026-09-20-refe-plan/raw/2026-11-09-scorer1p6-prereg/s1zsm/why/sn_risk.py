"""Descriptive, navtrain FIT: per set, does each head keep >= 1 speed-neutral pair (band + both directions)?  Then the chance that G18's windows read 0 by luck, by Monte Carlo over random micro-batches of 8 sets."""
import numpy as np
Z = np.load("F:/tanitad_scratch/p6fit/fit_pack8.npz", allow_pickle=False)
ok = Z["has_pdm"].astype(bool)
L = Z["logits"][ok].astype(np.float64); PD = Z["pdm"][ok].astype(np.float64); XY = Z["traj_xy"][ok].astype(np.float64)
NC, DAC, EP, TTC, C = 0, 1, 2, 3, 4
P = 1 / (1 + np.exp(-L)); R = P[..., NC] * P[..., DAC] * (5 * P[..., EP] + 5 * P[..., TTC] + 2 * P[..., C]) / 12
V = ~np.isnan(PD[..., 6]) & np.isfinite(XY).all((-1, -2))
v4 = np.linalg.norm(XY[:, :, 7] - XY[:, :, 6], axis=-1) / 0.5; vfirst = np.linalg.norm(XY[:, :, 0], axis=-1) / 0.5
S1Z = np.zeros((len(L), 3), bool); SN = np.zeros((len(L), 3), bool)
for s in range(len(L)):
    v = V[s]
    if not v.any(): continue
    v0 = float(np.median(vfirst[s][v]))
    clean = v & (PD[s, :, NC] == 1) & (PD[s, :, DAC] == 1) & (PD[s, :, TTC] == 1)
    top = np.argsort(-np.where(v, R[s], -np.inf), kind="stable")[:8]
    ade = np.linalg.norm(XY[s][:, None] - XY[s][None, :], axis=-1).mean(-1)
    for k, h in enumerate((NC, DAC, TTC)):
        hard = np.zeros(64, bool); hard[top] = True; hard &= v & (PD[s, :, h] < 1)
        ii, jj = np.nonzero(clean[:, None] & hard[None, :] & (ade <= 1.0))
        if not len(ii): continue
        S1Z[s, k] = True
        dv = v4[s, ii] - v4[s, jj]; band = np.abs(dv) / max(v0, 1.0) < 0.15
        SN[s, k] = (dv[band] > 0).any() and (dv[band] <= 0).any()
print("per-set share  S1Z", S1Z.mean(0).round(4), " SN", SN.mean(0).round(4))
rng = np.random.default_rng(0); N = len(L)
for name, M in (("S1Z", S1Z), ("SN", SN)):
    trials = 4000
    mb = M[rng.integers(0, N, size=(trials, 8))]             # trials x 8 sets x 3 heads
    all3 = mb.any(1).all(-1).mean()                           # one micro-batch has every head
    p_probe_miss = (1 - all3) ** 400
    live = {}
    for nmb in (16, 64, 128):
        w = M[rng.integers(0, N, size=(2000, nmb * 8))].any(1)  # does each head appear in the window
        live[nmb] = float((~w.all(-1)).mean())
    print(f"{name}: P(one micro-batch has all 3 heads) {all3:.4f}; P(probe search of 400 finds none) {p_probe_miss:.4f}; P(live window has a head at 0) " + ", ".join(f"{k} mb: {v:.4f}" for k, v in live.items()))
    for lim in (400, 1000, 2000):
        print(f"   probe miss at search limit {lim}: {(1 - all3) ** lim:.4f}; expected search length {1 / max(all3, 1e-9):.0f} micro-batches")
