import sys, json, numpy as np
exec(open("costattr_read.py").read().split("def main")[0])
import navrb2_lib as NL, navrb2_read as N2, strat1_lib as L, strat1_read as S, obd_read as OR, obd_geom as G
CI = L.load_ci(); geom = G.build()
bk = lambda x: f"{SR}/t1/dump_STRAT1-{x}_m10_causal_heads"
dg = S.load_dump_n(bk("C0_s0"), 141); J = geom.join(dg); glat, glon = geom.labels(dg["g"])
out = {}
for s in SEEDS:
    P = {}
    for n, d in (("C", f"{E}/dumps141_hb02/dump_N1DC-N1DC_s{s}_m10_causal_heads"), ("B", f"{E}/dumps141_hb02/dump_N1DC-N1D_s{s}_m10_causal_heads"), ("U", f"{H}/dumps141_hb02/dump_N1DC-N1DC_s{s}U_m10_causal_heads")):
        x = OR.load_dump(d); P[n] = OR.to_cells(geom.arm_arrays(x["cl"], dg["g"], J, dg["eid"]), glat, glon)
    bad = {}; allfin_max = 0.0
    for c in OR.CELLS:
        a, b, u = (np.asarray(P[k][c], float) for k in "CBU")
        f = lambda *z: np.logical_and.reduce([np.isfinite(q) for q in z])
        m3 = f(a, b, u)
        if not m3.any(): continue
        # card bookkeeping as the reader does it (pairwise masks)
        mCB, mUB, mCU = f(a, b), f(u, b), f(a, u)
        d = (a[mCU].mean() - u[mCU].mean()) + (u[mUB].mean() - b[mUB].mean()) - (a[mCB].mean() - b[mCB].mean())
        d3 = (a[m3].mean() - u[m3].mean()) + (u[m3].mean() - b[m3].mean()) - (a[m3].mean() - b[m3].mean())
        allfin_max = max(allfin_max, abs(d3))
        if abs(d) > 1e-12: bad[c] = {"gap": float(d), "n": [int(mCB.sum()), int(mUB.sum()), int(mCU.sum()), int(m3.sum())]}
    out[s] = {"n_cells_failing": len(bad), "common_finite_max_abs": allfin_max, "failing": bad}
    print(s, len(bad), allfin_max, list(bad)[:30])
json.dump(out, open("kadd_diag.json", "w"), indent=1)
