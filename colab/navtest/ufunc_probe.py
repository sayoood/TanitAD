# Platform float probe (runs in the NAVSIM harness venv on BOTH the dev box and the VM): sha256 of NumPy transcendental
# ufuncs and a small OpenBLAS solve on FIXED inputs. Equal hashes on both platforms rule a function out as a source of
# the G-H residual; different hashes show a platform-level (libm / compiler / kernel) difference exists. ASCII only.
import hashlib
import json
import sys

import numpy as np

rng = np.random.default_rng(20260927)
x = rng.uniform(-50.0, 50.0, 200_000)
y = rng.uniform(-50.0, 50.0, 200_000)
p = rng.uniform(1e-3, 1e3, 200_000)
fs = {"sin": np.sin(x), "cos": np.cos(x), "tan": np.tan(x / 40), "arctan2": np.arctan2(y, x), "exp": np.exp(x / 5),
      "log": np.log(p), "sqrt": np.sqrt(p), "hypot": np.hypot(x, y), "power": np.power(p, 0.37),
      "arcsin": np.arcsin(np.clip(x / 50, -1, 1)), "sum": np.cumsum(x), "dot": np.array([x @ y])}
A = rng.normal(size=(64, 64))
b = rng.normal(size=64)
fs["solve"] = np.linalg.solve(A + 64 * np.eye(64), b)
fs["matmul"] = (A @ A.T).ravel()
out = {"numpy": np.__version__, "platform": sys.platform,
       "sha256": {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16] for k, v in fs.items()}}
try:
    import shapely
    from shapely.geometry import LineString, Point
    ls = LineString(np.stack([np.linspace(0, 300, 400), 5 * np.sin(np.linspace(0, 9, 400))], 1))
    proj = np.array([ls.project(Point(a, b_)) for a, b_ in zip(x[:5000] * 3 + 150, y[:5000] / 5)])
    out["shapely"] = shapely.__version__ + " / GEOS " + shapely.geos_version_string
    out["sha256"]["shapely_project"] = hashlib.sha256(proj.tobytes()).hexdigest()[:16]
except Exception as e:  # noqa: BLE001
    out["shapely_error"] = repr(e)[:200]
print("ZZUFUNC " + json.dumps(out))
