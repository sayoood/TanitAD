"""Analytic checks of band_lib.band_rows: constant speed, constant curvature, full-band == comps (t1k_read), additive ADE identity, edge-shift regression."""
import ast
import json
import math
import sys

sys.path.insert(0, "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/horizon_review/band_eval")
import numpy as np
import torch
import band_lib as BL
from band_lib import FF, DT

RES = {}
T = np.arange(1, 31) * DT


def line(v, n=3):
    return np.tile(np.stack([v * T, np.zeros(30)], -1)[None], (n, 1, 1))


def circle(R, v, n=3):
    th = v * DT / R
    i = np.arange(1, 31)
    return np.tile(np.stack([R * np.sin(i * th), R * (1 - np.cos(i * th))], -1)[None], (n, 1, 1))


def close(a, b, tol=1e-9):
    return bool(np.allclose(a, b, rtol=0, atol=tol, equal_nan=True))


checks = []


def rec(name, ok, detail=""):
    checks.append({"check": name, "pass": bool(ok), "detail": detail})
    print(("PASS " if ok else "FAIL ") + name, detail)


# ---- 1. constant speed, straight: GT 10 m/s, pred 10.5 m/s
G, X = line(10.0), line(10.5)
for lab, (a, b) in BL.BANDS.items():
    r = BL.band_rows(X, G, a, b)
    t = T[a:b]
    rec(f"const-speed {lab} speed MAE = 0.5", close(r["speed_mae_mps"], 0.5), str(r["speed_mae_mps"][0]))
    rec(f"const-speed {lab} accel MAE = 0", close(r["accel_mae_mps2"], 0.0, 1e-9))
    rec(f"const-speed {lab} ADE = 0.5*mean(t)", close(r["ade_m"], 0.5 * t.mean()), f"{r['ade_m'][0]:.6f} vs {0.5 * t.mean():.6f}")
    rec(f"const-speed {lab} along MAE = 0.5*mean(t)", close(r["along_mae_m"], 0.5 * t.mean()))
    rec(f"const-speed {lab} FDE = 0.5*t_end", close(r["fde_m"], 0.5 * t[-1]), f"{r['fde_m'][0]:.4f} (t_end {t[-1]:.1f})")
    rec(f"const-speed {lab} cross / heading / yaw / curvature = 0",
        all(close(r[k], 0.0) for k in ("cross_mae_m", "heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm")))
# accel error with a speed ramp: GT constant 10 m/s, pred accelerating a=1 m/s^2 -> accel MAE exactly 1 in every band
Xa = np.tile(np.stack([10 * T + 0.5 * 1.0 * T ** 2, np.zeros(30)], -1)[None], (3, 1, 1))
# the chord speed of step i is the mean speed over the step, so the step-to-step difference is exactly a*dt/dt = 1
for lab, (a, b) in BL.BANDS.items():
    r = BL.band_rows(Xa, G, a, b)
    rec(f"ramp a=1 {lab} accel MAE = 1", close(r["accel_mae_mps2"], 1.0, 1e-9), str(r["accel_mae_mps2"][0]))

# ---- 2. constant curvature: GT circle R=50 v=10, pred circle R=40 v=10 (chord geometry, exact)
R1, R2, v = 50.0, 40.0, 10.0
G, X = circle(R1, v), circle(R2, v)
th1, th2 = v * DT / R1, v * DT / R2
ds1, ds2 = 2 * R1 * math.sin(th1 / 2), 2 * R2 * math.sin(th2 / 2)
k1, k2 = th1 / ds1, th2 / ds2
for lab, (a, b) in BL.BANDS.items():
    r = BL.band_rows(X, G, a, b)
    idx = np.arange(a, b)
    rec(f"circle {lab} yaw-rate MAE = |th2-th1|/dt", close(r["yaw_rate_mae_degps"], math.degrees(abs(th2 - th1) / DT), 1e-9),
        f"{r['yaw_rate_mae_degps'][0]:.6f}")
    rec(f"circle {lab} curvature MAE = |k2-k1|", close(r["curvature_mae_1pm"], abs(k2 - k1), 1e-7), f"{r['curvature_mae_1pm'][0]:.8f} vs {abs(k2 - k1):.8f} (tol 1e-7: the instrument adds _EPS to ds)")
    want_h = math.degrees(np.mean((idx + 0.5) * abs(th2 - th1)))
    rec(f"circle {lab} heading MAE = mean((i+1/2)|th2-th1|)", close(r["heading_mae_deg"], want_h, 1e-9), f"{r['heading_mae_deg'][0]:.6f} vs {want_h:.6f}")
    rec(f"circle {lab} speed MAE = |ds2-ds1|/dt", close(r["speed_mae_mps"], abs(ds2 - ds1) / DT, 1e-9))
    rec(f"circle {lab} accel MAE = 0", close(r["accel_mae_mps2"], 0.0, 1e-9))

# ---- 3. a slice over the FULL band equals the existing comps (t1k_read.py, AST-extracted) -- analytic AND a real dump
src = "E:/Projects/TanitAD-artifacts/dagreread/reader_src/t1k_read.py"
ns = {"np": np, "torch": torch, "FF": FF, "math": math, "DT": DT, "PB": None}
for node in ast.parse(open(src, encoding="utf-8").read()).body:
    if isinstance(node, ast.FunctionDef) and node.name == "comps":
        exec(compile(ast.Module([node], []), src, "exec"), ns)
comps = ns["comps"]
MAP = {"ade_m": "ade6_m", "speed_mae_mps": "speed_mae_mps", "along_mae_m": "along_mae_m", "accel_mae_mps2": "accel_mae_mps2", "cross_mae_m": "cross_mae_m",
       "heading_mae_deg": "heading_mae_deg", "yaw_rate_mae_degps": "yaw_rate_mae_degps", "curvature_mae_1pm": "curvature_mae_1pm"}


def full_vs_comps(X, G, tag):
    c = comps(X, G)
    r = BL.band_rows(X, G, 0, 30)
    worst = max(float(np.nanmax(np.abs(r[k] - c[m]))) if np.isfinite(c[m]).any() else 0.0 for k, m in MAP.items())
    nanmatch = all(np.array_equal(np.isnan(r[k]), np.isnan(c[m])) for k, m in MAP.items())
    rec(f"full-band slice == comps ({tag})", worst == 0.0 and nanmatch, f"max |diff| {worst:.3g}, NaN pattern equal {nanmatch}")


rng = np.random.default_rng(0)
Gs = np.stack([8 * T + 0.3 * np.sin(T), 0.4 * T ** 2 / (1 + T)], -1)[None] * rng.uniform(0.3, 1.5, (50, 1, 1))
Xs = Gs + rng.normal(0, 0.3, Gs.shape).cumsum(1) * 0.1
full_vs_comps(Xs, Gs, "synthetic 50 windows")
d = BL.load_dir("E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1/dump_DAGOACC_s0_m10_causal_heads")
full_vs_comps(d["cl"], d["g"], "DAG-OACC s0 real dump, 2399 windows")
full_vs_comps(d["fl"], d["g"], "P10 real dump")

# ---- 4. additive identity on ADE: 10*ade_op + 20*ade_tac == 30*ade_full (real dump)
op, tc, fu = (BL.band_rows(d["cl"], d["g"], *BL.BANDS[k]) for k in ("operative_0_2s", "tactical_2_6s", "full_0_6s"))
rec("ADE additive identity 10*op + 20*tac = 30*full (real)", close(10 * op["ade_m"] + 20 * tc["ade_m"], 30 * fu["ade_m"], 1e-9))
rec("FDE: op FDE = e[knot 10], tac FDE = full FDE = e[knot 30]", close(tc["fde_m"], fu["fde_m"], 0.0) and
    close(op["fde_m"], np.linalg.norm(d["cl"][:, 9] - d["g"][:, 9], axis=-1), 1e-12))

# ---- 5. deliberate regression: shift the band edge by one knot must change the result
op1 = BL.band_rows(d["cl"], d["g"], 0, 11)
tc1 = BL.band_rows(d["cl"], d["g"], 11, 30)
chg = {k: (float(np.nanmean(op[k])), float(np.nanmean(op1[k]))) for k in BL.METRICS}
chg_t = {k: (float(np.nanmean(tc[k])), float(np.nanmean(tc1[k]))) for k in BL.METRICS}
moved = {k: abs(v[0] - v[1]) > 1e-9 for k, v in chg.items()}
moved_t = {k: abs(v[0] - v[1]) > 1e-9 for k, v in chg_t.items() if k != "fde_m"}  # fde_m is the value at knot 30 for both edges: unchanged by construction
rec("regression: operative edge 10 -> 11 changes every metric mean", all(moved.values()), json.dumps({k: round(v[1] - v[0], 6) for k, v in chg.items()}))
rec("regression: tactical edge 10 -> 11 changes every metric mean except fde_m (knot 30 in both)", all(moved_t.values()), json.dumps({k: round(v[1] - v[0], 6) for k, v in chg_t.items()}))
# and the analytic check itself goes red on the shifted band
G, X = line(10.0), line(10.5)
r = BL.band_rows(X, G, 0, 11)
rec("regression: analytic const-speed ADE check goes RED with the shifted edge", not close(r["ade_m"], 0.5 * T[0:10].mean()), f"{r['ade_m'][0]:.6f} vs expected {0.5 * T[0:10].mean():.6f}")

n_pass = sum(c["pass"] for c in checks)
RES = {"n_checks": len(checks), "n_pass": n_pass, "checks": checks,
       "regression_shifted_edge_mean_deltas": {"operative_0_11": {k: v for k, v in chg.items()}, "tactical_11_30": {k: v for k, v in chg_t.items()}}}
json.dump(RES, open("E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/horizon_review/band_eval/ANALYTIC_CHECKS.json", "w"), indent=1)
print(f"{n_pass}/{len(checks)} PASS")
sys.exit(0 if n_pass == len(checks) else 1)
