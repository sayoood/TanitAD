"""Known-value check of the read-only scipy overlay in tanitad-train: the bridge's knots_to_navsim on ANALYTIC paths.
straight line x = v t  -> poses (v t, 0, 0) exactly; reverse-direction import identity recorded."""
import json, os, sys
import numpy as np, scipy, timm, torch
sys.path.insert(0, sys.argv[1])
import refcv8_bridge as B
v = 10.0
kt = np.asarray(B.KNOT_T_S, np.float64)
straight = np.stack([v * kt, np.zeros_like(kt)], 1)
p = B.knots_to_navsim(straight)
t = np.arange(1, 9) * 0.5
ok_s = bool(np.allclose(p[:, 0], v * t, atol=1e-9) and np.allclose(p[:, 1], 0, atol=1e-12) and np.allclose(p[:, 2], 0, atol=1e-12))
rec = {"scipy": scipy.__version__, "scipy_file": scipy.__file__, "timm": timm.__version__, "torch": torch.__version__,
       "numpy": np.__version__, "python": sys.version.split()[0], "cuda": torch.cuda.is_available(),
       "straight_line_exact": ok_s, "max_abs_x_err": float(np.abs(p[:, 0] - v * t).max())}
print("ZZOVL " + json.dumps(rec))
