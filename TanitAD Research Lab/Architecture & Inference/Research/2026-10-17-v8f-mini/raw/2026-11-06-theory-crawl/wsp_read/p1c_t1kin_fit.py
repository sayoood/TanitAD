"""T1-kin fixed-point CIs (descriptive, post-spec). Clip-cluster bootstrap B 2000 seed 0 of the linear map fit over v0 in [0,12). Writes P1C_RESULT.json."""
import json, os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import p1_return_map as P
import p1b_controls as C
out = {}
for arm, path in C.DUMPS.items():
    z = np.load(path, allow_pickle=True)
    plan, gt, v0, ci = z["plan"], z["gt"], z["v0"], z["ci"]
    ok = z["gt_ok"].astype(bool) & (v0 >= 0) & (v0 < 12)
    sec = lambda p: np.hypot(p[:, 11, 0] - p[:, 7, 0], p[:, 11, 1] - p[:, 7, 1]) / 0.8
    gp, gh = sec(plan), sec(gt)
    u = np.unique(ci[ok]); si = np.searchsorted(u, ci)
    def mom(y):
        M = np.zeros((len(u), 5))
        for j, w in enumerate((np.ones_like(v0), v0, v0 * v0, y, v0 * y)):
            M[:, j] = np.bincount(si[ok], weights=w[ok], minlength=len(u))
        return M
    rng = np.random.default_rng(0); idx = rng.integers(0, len(u), (2000, len(u)))
    res = {}
    for nm, y in (("plan", gp), ("logged", gh)):
        M = mom(y); a, b, vs = P.fit_from_M(M.sum(0)); _, bb, bvs = P.fit_from_M(M[idx].sum(1))
        res[nm] = {"slope": float(b), "slope_ci": P.pct(bb), "v_star": None if not np.isfinite(vs) else float(vs), "v_star_ci": P.pct(bvs), "v_star_defined_frac": float(np.isfinite(bvs).mean())}
    Mp, Mh = mom(gp), mom(gh)
    _, _, vp = P.fit_from_M(Mp[idx].sum(1)); _, _, vh = P.fit_from_M(Mh[idx].sum(1))
    d = np.where(np.isfinite(vp) & np.isfinite(vh), vp - vh, np.nan)
    res["vstar_plan_minus_logged_ci"] = P.pct(d); res["n_rows"] = int(ok.sum()); res["n_clips"] = int(len(u))
    out[arm] = res
    print(arm, json.dumps(res))
json.dump(out, open(P.HERE + "/P1C_RESULT.json", "w"), indent=1)
