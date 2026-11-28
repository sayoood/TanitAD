import numpy as np, glob
L = np.load(r"F:/strat_out/band_labels/strategic_geom_eval.npz", allow_pickle=True)
in141 = L["clip__in_cache141"].astype(bool); off = L["clip__off"]; n = L["clip__n"]; ck = L["ck_xyh"]; v = L["ck_valid_span"]
cidx = np.where(in141)[0]; print("cached", len(cidx), "horizons", L["horizons_s"][:2])
F = sorted(glob.glob(r"E:/Projects/TanitAD-artifacts/moved_from_c/hlat/run/dump_B_s0_m10/ep*.npz"))
def err(c, shift, D):
    e = []
    for w, g in zip(D["ws"], D["g"]):
        j = int(w) + shift
        if j < 0 or j >= n[c]: continue
        r = off[c] + j
        if v[r, 0]: e.append(np.hypot(*(ck[r, 0, :2] - g[29])))
    return e
# which NOW offset, clip order assumed = file order among the cached clips
best = {}
for shift in range(-2, 20):
    E = []
    for i, f in enumerate(F[:30]):
        E += err(cidx[i], shift, np.load(f, allow_pickle=True))
    best[shift] = (float(np.median(E)) if E else 1e9, len(E))
s = min(best, key=lambda k: best[k][0]); print({k: round(v_[0], 3) for k, v_ in best.items()}); print("best shift", s, best[s])
E, spd = [], []
for i, f in enumerate(F):
    D = np.load(f, allow_pickle=True); E += err(cidx[i], s, D)
E = np.array(E); print("all 141 clips, shift", s, "n", len(E), "median m", round(float(np.median(E)), 3), "p90", round(float(np.percentile(E, 90)), 3), "p99", round(float(np.percentile(E, 99)), 3), "frac>5m", round(float((E > 5).mean()), 4))
# control: wrong clip must not match
W = []
for i, f in enumerate(F):
    W += err(cidx[(i + 7) % len(cidx)], s, np.load(f, allow_pickle=True))
print("control wrong clip median m", round(float(np.median(W)), 2), "n", len(W))
G = L["ck_gap_s"]; S = L["ck_valid_span"]
T = np.load(r"F:/strat_out/band_labels/strategic_geom_train.npz", allow_pickle=True); G = T["ck_gap_s"]; S = T["ck_valid_span"]; St = T["ck_valid"]
for h, name in enumerate(T["horizons_s"]):
    a = S[:, h]; g = G[:, h]
    print(float(name), "span", round(float(a.mean()), 4), "strict", round(float(St[:, h].mean()), 4), *[f"gap<={c}: {float((a & (g <= c)).mean()):.4f}" for c in (1.0, 2.0, 2.5, 3.0)])
