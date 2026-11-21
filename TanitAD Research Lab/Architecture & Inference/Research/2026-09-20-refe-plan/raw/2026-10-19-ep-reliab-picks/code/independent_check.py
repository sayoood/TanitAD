"""EP-RELIAB-PICKS: an INDEPENDENT re-derivation of the headline point estimates, written after result.json existed and sharing no code with epreliab_lib / pickrule_lib (own picks, own loops).
It is a cross-check of the instrument, not part of the registered decision; it decides nothing.  Reads arrays.npz READ ONLY."""
import json
import sys

import numpy as np

z = np.load("C:/Users/Admin/par18/pickrule/run/prepare/arrays.npz", allow_pickle=False)
z6, lab = z["z6"].astype(np.float64), z["lab"].astype(np.float64)
N = len(z6)
sg = lambda a: 1.0 / (1.0 + np.exp(-a))                                                          # noqa: E731


def picks(zz):
    p = sg(zz)
    agg = p[:, :, 0] * p[:, :, 1] * (p[:, :, 2] * 5.0 + p[:, :, 3] * 5.0 + p[:, :, 4] * 2.0) / 12.0
    return np.array([int(np.argmax(agg[i])) for i in range(N)])


out = {}
rec = json.load(open("C:/Users/Admin/par18/epreliab/run/results/result.json"))
pa = picks(z6)
for s in (0, 1):
    zE = z6.copy()
    zE[:, :, 2] = z[f"ep_OOF_s{s}"].astype(np.float64)
    pb = picks(zE)
    sig = sg(zE[:, :, 2])
    y = lab[:, :, 2]
    ghat = np.array([sig[i, pb[i]] - sig[i, pa[i]] for i in range(N)])
    g = np.array([y[i, pb[i]] - y[i, pa[i]] for i in range(N)])
    ye, yp = np.array([y[i, pb[i]] for i in range(N)]), np.array([y[i, pa[i]] for i in range(N)])
    se, sp = np.array([sig[i, pb[i]] for i in range(N)]), np.array([sig[i, pa[i]] for i in range(N)])
    # slope over the safe proposals, set-demeaned, by an explicit loop
    sxy = sxx = 0.0
    for i in range(N):
        m = (lab[i, :, 0] >= 1.0) & (lab[i, :, 1] >= 1.0)
        if m.sum() >= 2:
            a, b = sig[i, m] - sig[i, m].mean(), y[i, m] - y[i, m].mean()
            sxy += float((a * b).sum())
            sxx += float((a * a).sum())
    # sequential attribution by an explicit per-set loop (NC, DAC, EP, TTC, C)
    contrib = np.zeros(5)
    tot = 0.0
    for i in range(N):
        A, B = lab[i, pa[i], :5].copy(), lab[i, pb[i], :5]
        f = lambda t: t[0] * t[1] * (5 * t[2] + 5 * t[3] + 2 * t[4]) / 12.0                       # noqa: E731
        prev = f(A)
        cur = A.copy()
        for k in range(5):
            cur[k] = B[k]
            v = f(cur)
            contrib[k] += v - prev
            prev = v
        tot += f(B) - f(A)
    mine = {"n_changed": int((pa != pb).sum()), "pred_gain": ghat.mean(), "real_gain": g.mean(), "delta": (ghat - g).mean(), "M_e": ye.sum() / se.sum(), "M_p": yp.sum() / sp.sum(),
            "D_M": yp.sum() / sp.sum() - ye.sum() / se.sum(), "slope": sxy / sxx, "dpdms_x100": 100 * tot / N, "seq_x100": [float(100 * c / N) for c in contrib]}
    lib = rec["per_seed"][str(s)]
    ref = {"n_changed": lib["n_changed"], "pred_gain": lib["G"]["pred_gain"]["point"], "real_gain": lib["G"]["real_gain"]["point"], "delta": lib["G"]["delta"]["point"], "M_e": lib["G"]["M_e"]["point"],
           "M_p": lib["G"]["M_p"]["point"], "D_M": lib["G"]["D_M"]["point"], "slope": lib["scale"]["slope"]["point"], "dpdms_x100": lib["attribution"]["total_x100"]["point"],
           "seq_x100": [lib["attribution"]["sequential_x100"][t]["point"] for t in lib["attribution"]["order"]]}
    worst = max(abs(float(mine[k]) - float(ref[k])) for k in ref if k != "seq_x100")
    worst = max(worst, max(abs(a - b) for a, b in zip(mine["seq_x100"], ref["seq_x100"])))
    out[f"seed{s}"] = {"independent": mine, "library": ref, "max_abs_difference": worst, "agree_1e-9": bool(worst < 1e-9)}
    print(f"seed {s}: max |independent - library| = {worst:.3e}  ({'AGREE' if worst < 1e-9 else 'DISAGREE'})", flush=True)
json.dump(out, open("C:/Users/Admin/par18/epreliab/run/results/independent_check.json", "w", encoding="utf-8", newline="\n"), indent=1, default=float)
sys.exit(0 if all(v["agree_1e-9"] for v in out.values()) else 1)
