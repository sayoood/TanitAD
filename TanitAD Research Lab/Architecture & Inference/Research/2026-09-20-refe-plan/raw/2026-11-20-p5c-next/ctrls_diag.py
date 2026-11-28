import numpy as np, sys
for arm, f in [("C1", "stage2_p5c_c1"), ("C0", "stage2_p5c_c0"), ("base", "stage2_p7_full")]:
    z = np.load(f"E:/Projects/TanitAD/data/refe_navtest/seams/{f}.proposals.npz", allow_pickle=False)
    L = z["logits"]; pk = z["pick"]
    print(arm, "logits dtype", L.dtype, L.shape)
    for dt in (np.float64, np.float32):
        x = L.astype(dt); p = 1 / (1 + np.exp(-x))
        g = (p[..., 0] * p[..., 1]) * (5 * p[..., 2] + 5 * p[..., 3] + 2 * p[..., 4]) / dt(12)
        sel = g.argmax(1); bad = np.where(sel != pk)[0]
        print(" ", dt.__name__, "disagree", len(bad))
        for i in bad[:5]:
            s = np.sort(g[i])[::-1]
            print("   tok", z["token"][i], "pick", pk[i], "sel", sel[i], "g[pick]", repr(float(g[i, pk[i]])), "g[sel]", repr(float(g[i, sel[i]])),
                  "gap", float(g[i, sel[i]] - g[i, pk[i]]))
