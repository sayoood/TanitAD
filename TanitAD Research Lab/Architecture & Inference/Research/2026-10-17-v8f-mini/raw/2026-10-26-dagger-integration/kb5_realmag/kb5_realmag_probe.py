"""K-B5 on REAL clip-local magnitudes (2026-10-26, Master Mind). The bank's K-B5 (v8f_dagger.kb5_measure) gates the float32 relative-transform error at 1e-4 m;
float32 rounding grows with |coord|, so a clip far from its origin could fail on storage precision alone. This probe runs the library's own kb5_measure on own-state
records SIMULATED from the 141 real eval-141 clips' pose_grid (the logged history moved by a rigid drift <= 3 m, <= 0.5 rad about NOW, as the matcher admits),
stored as float32 as the bank stores them. Eval-141 is never banked or trained on; it is used here only for its coordinate magnitudes.
python -I kb5_realmag_probe.py <di/tree> <eval clips dir>"""
import glob, importlib.util, sys
import numpy as np, torch
TREE, CL = sys.argv[1], sys.argv[2]
sys.path.insert(0, TREE + "/stack")                                  # the tree's own tanitad, never the venv's install
spec = importlib.util.spec_from_file_location("dg", TREE + "/stack/tanitad/data/v8f_dagger.py"); DG = importlib.util.module_from_spec(spec); spec.loader.exec_module(DG)
rng, F, rows, cmax = np.random.default_rng(0), 10, [], []
for f in sorted(glob.glob(CL + "/*.pt")):
    xyh = torch.load(f, map_location="cpu", weights_only=False)["pose_grid"].numpy().astype(np.float64)[:, :3]
    cmax.append(np.abs(xyh[:, :2]).max())
    for t in range(F, len(xyh)):
        hist = xyh[t - F + 1:t + 1]
        th, d = rng.uniform(-0.5, 0.5), rng.uniform(-3, 3, 2)
        c, s, n = np.cos(th), np.sin(th), hist[-1, :2]
        q = hist[:, :2] - n
        own = np.c_[n + d + np.c_[q[:, 0] * c - q[:, 1] * s, q[:, 0] * s + q[:, 1] * c], hist[:, 2] + th]
        rows.append((own.astype(np.float32).astype(np.float64), xyh[t], np.abs(xyh[t, :2]).max()))
cmax = np.array(cmax)
print(f"clips {len(cmax)}; clip max |coord| m: median {np.median(cmax):.1f}, p95 {np.percentile(cmax, 95):.1f}, max {cmax.max():.1f}; clips >= 512 m: {int((cmax >= 512).sum())}")
own, log, mag = np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows]), np.array([r[2] for r in rows])
m = DG.kb5_measure(own, log)
print("ALL", {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in m.items()})
for lo, hi in ((0, 256), (256, 512), (512, 1024), (1024, 1e9)):
    sel = (mag >= lo) & (mag < hi)
    if sel.any():
        mm = DG.kb5_measure(own[sel], log[sel])
        print(f"|coord| [{lo}, {hi}) m: n {int(sel.sum())}  rel {mm['max_rel_transform_err_m']:.3e}  now {mm['max_pose_now_err_m']:.3e}  rel_float64 {mm['max_rel_transform_err_float64_m']:.1e}")
print(f"tol {DG.KB5_TOL_M}; PASS on all: {DG.kb5_pass({**m, 'n_records': m['n']})}")
