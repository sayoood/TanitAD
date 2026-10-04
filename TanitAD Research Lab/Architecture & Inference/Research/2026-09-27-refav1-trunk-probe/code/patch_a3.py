"""SPEC A3: add P10 (the SAME MLP on kinematics only) and P11 (P9 with shuffled trunk rows) to probe_te.py, plus
their pairs and strata. Resumable: P0-P9 load from out_te/arm_*.npz; the permutation for P11 is drawn in arm order
(after P5 and P8), so a fresh run and a resumed run draw identical permutations."""
import sys

p = "C:/Users/Admin/refav1_probe/probe_te.py"
s = open(p, encoding="utf-8").read()


def rep(old, new, n=1):
    global s
    c = s.count(old)
    if c != n:
        sys.exit(f"anchor matched {c} (want {n}) -- refusing:\n{old[:160]}")
    s = s.replace(old, new)


# 1. the kinematics-only MLP (identical net, optimiser, grids, inner CV; no image block)
rep("def main():",
    '''def fit_mlp_kin(kf, y, base, g, groups, wds=(1e-4, 1e-2), epochs=60):
    """SPEC A3 P10: the IDENTICAL MLP / optimiser / inner CV as fit_mlp, on the kinematic features ONLY."""
    ug = np.unique(groups); inner = {e: i % P.N_INNER for i, e in enumerate(ug)}
    fold = np.array([inner[e] for e in groups])
    score = {}
    for f in range(P.N_INNER):
        tr, te = fold != f, fold == f
        for wd in wds:
            _, _, cur = mlp_curve(kf[tr], y[tr], kf[te], base[te], g[te], wd, epochs)
            score.setdefault(wd, []).append(np.array(cur) * te.sum())
    tot = len(groups)
    best, best_s = None, np.inf
    for wd, v in score.items():
        c = np.sum(v, 0) / tot
        e = int(np.argmin(c))
        if c[e] < best_s:
            best, best_s = (0, wd, e + 1), float(c[e])
    net, st, _ = mlp_curve(kf, y, None, None, None, best[1], best[2])
    return net, st, best


def main():''')

# 2. the two arms, appended after P9
rep('''"P8_shuf_sp": "trunk_sp", "P9_trunk_sp_mlp": "trunk_sp"}''',
    '''"P8_shuf_sp": "trunk_sp", "P9_trunk_sp_mlp": "trunk_sp",
            "P10_kin_mlp": None, "P11_shuf_sp_mlp": "trunk_sp"}''')

# 3. P11 draws its permutation in arm order
rep('''perm = rng.permutation(len(gT2)) if a in ("P5_shuf", "P8_shuf_sp") else None''',
    '''perm = rng.permutation(len(gT2)) if a in ("P5_shuf", "P8_shuf_sp", "P11_shuf_sp_mlp") else None''')

# 4. the fits: P11 = P9 with permuted image rows; P10 = the kinematics-only MLP
rep('''            if a == "P9_trunk_sp_mlp":
                ro, net, st, best = fit_mlp(imgT, okr, FT["kin"][ok], y[ok], b[ok], g[ok], T["ep"][ok])''',
    '''            if a == "P10_kin_mlp":
                net, st, best = fit_mlp_kin(FT["kin"][ok], y[ok], b[ok], g[ok], T["ep"][ok])
                xm, xs, ym = st
                with torch.no_grad():
                    yh = net(torch.tensor((FE["kin"] - xm) / xs, dtype=torch.float32)).numpy() + ym
                out[hz] = (bE + yh.reshape(-1, *bE.shape[1:]), [0, float(best[1]), int(best[2])])
                continue
            if a in ("P9_trunk_sp_mlp", "P11_shuf_sp_mlp"):
                mrows = okr if perm is None else perm[okr]
                ro, net, st, best = fit_mlp(imgT, mrows, FT["kin"][ok], y[ok], b[ok], g[ok], T["ep"][ok])''')

# 5. pairs + strata rows for A3
rep('''("P5_shuf", "P2_kin"), ("P8_shuf_sp", "P2_kin"), ("P3_raw", "P2_kin"), ("P0_kdx", "P0_kdx")]''',
    '''("P5_shuf", "P2_kin"), ("P8_shuf_sp", "P2_kin"), ("P3_raw", "P2_kin"), ("P0_kdx", "P0_kdx"),
             ("P9_trunk_sp_mlp", "P10_kin_mlp"), ("P11_shuf_sp_mlp", "P10_kin_mlp"), ("P10_kin_mlp", "P2_kin")]''')
rep('''for x, y in (("P6_trunk_sp", "P2_kin"), ("P9_trunk_sp_mlp", "P2_kin"), ("P2_kin", "P0_kdx")):''',
    '''for x, y in (("P6_trunk_sp", "P2_kin"), ("P9_trunk_sp_mlp", "P2_kin"), ("P2_kin", "P0_kdx"),
                             ("P9_trunk_sp_mlp", "P10_kin_mlp")):''')
open(p, "w", encoding="utf-8").write(s)
print("A3 arms P10/P11 added")
