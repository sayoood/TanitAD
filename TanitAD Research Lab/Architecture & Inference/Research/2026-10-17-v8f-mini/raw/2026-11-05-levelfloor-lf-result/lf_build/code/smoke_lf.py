#!/usr/bin/env python3
"""smoke_lf.py -- the LEVEL-FLOOR smoke on real data (CPU), and the wall-time comparison.

  S1  floor ON, kappa 0.5, floor heads LLR_s0, planner C0_s0, --scene reindexed, composite (--smooth sg72 --feasible kbox), the FIRST 6 windows, 12 threads.
  S2  the BARE arm of A3_RULING_1 (--smooth sg72, no --feasible: the banked bases' flags; raw -> S -> F), the same windows.
  S3  --smooth off --feasible off (raw -> F), the same windows.
For each: bind / cap / band rates over the replans (the ticks' lf_bind / lf_cap / lf_vhat), the lift where F binds, the dump's dtypes, and an INDEPENDENT recomputation of the issued plan from the dumped fields:
  composite: plan == kbox(F(plan_s)) with F re-run on (plan_s, v0, lf_vhat, lf_pi, lf_gap / 40, kappa);   bare (sg72, no kbox): plan == F(S(plan_raw, P));   smooth off: plan == F(plan_raw).   (the dump stores float32 head values, so the tolerance is 1e-4 m)
  T   wall time: the first 64 windows, reindexed composite, 12 threads, floor OFF vs ON (kappa 0.5), twice each; the tool's own loop time (``[64/64] N s``, no model load) and the process wall time.
Writes raw/SMOKE_LF.json and logs/SMOKE_LF_*.out.   usage: python -I smoke_lf.py [--skip-timing]"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
ART = "E:/Projects/TanitAD-artifacts"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
TREE = "F:/TanitAD-scratch/motion_c/lf_read/tree"
TOOL = f"{TREE}/stack/scripts/v8f_t1_eval_ri.py"
SUBSET = "E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_read/code/run_tool_subset.py"
SM = "F:/TanitAD-scratch/motion_c/lf_read/smoke"
C0 = f"{ART}/v8f-recover/M2b_s0_recft_C0/ckpt_last.pt"
LLR = f"{ART}/v8f-recover/M2b_s0_recft_LLR/ckpt_last.pt"


def tool_module():
    for p in (f"{TREE}/stack", f"{TREE}/taniteval"):
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location("v8f_t1_eval_lf_smoke", TOOL)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def run(tag, n_win, threads, smooth, feasible, floor_kappa):
    out = f"{SM}/{tag}"
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), PYTHONDONTWRITEBYTECODE="1")
    cmd = [PY, "-I", SUBSET, "--tool", TOOL, "--first", str(n_win), "--", "run", "--ckpt", C0, "--tag", tag, "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache",
           "--eval-grid", f"{DATA}/eval_grid_windows.json", "--eval-join", f"{DATA}/join_eval141.npz", "--out-dir", out, "--replan-steps", "10", "--nav", "deadreckon", "--schedule", "periodic", "--limit", "causal",
           "--smooth", smooth, "--scene", "reindexed", "--dump-ticks", f"{out}/ticks", "--dump-heads", "--device", "cpu", "--n-boot", "20"] + (["--feasible", feasible] if feasible != "off" else [])
    if floor_kappa is not None:
        cmd += ["--level-floor", "on", "--floor-kappa", str(floor_kappa), "--floor-ckpt", LLR]
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall = time.time() - t0
    open(f"{PK}/logs/SMOKE_LF_{tag}.out", "w", encoding="utf-8", newline="\n").write(" ".join(cmd) + "\n" + r.stdout + r.stderr)
    if r.returncode:
        raise SystemExit(f"RUN FAILED {tag} rc {r.returncode}: {r.stderr[-1200:]}")
    loop = [float(x) for x in re.findall(r"\[\d+/\d+\] (\d+)s", r.stdout)]
    return {"out": out, "wall_s": wall, "loop_s": loop[-1] if loop else None, "n_windows": n_win, "threads": threads}


def check(T1, tag, kappa, smooth, feasible):
    from tanitad.refs import feasible_decode as _fd  # noqa: F401  (the tool imports it lazily; load it here so a failure is loud)
    z = {k: v for k, v in np.load(f"{SM}/{tag}/ticks/ticks_{tag}.npz", allow_pickle=True).items()}
    op = T1._lf()
    n = len(z["plan"])
    bind, cap = z["lf_bind"], z["lf_cap"]
    band = z["lf_vhat"].astype(np.float64) >= 3.0
    r = {"replans": int(n), "windows": int(len(set(zip(z["ci"].tolist(), z["t0"].tolist())))), "dtypes": {k: str(z[k].dtype) for k in ("lf_bind", "lf_cap", "lf_vf", "lf_vhat", "lf_pi", "lf_gap", "lf_lift", "lf_bind_k10")},
         "band_rate": float(band.mean()), "bind_rate": float(bind.mean()), "bind_k10_rate": float(z["lf_bind_k10"].mean()), "cap_rate": float(cap.mean()), "n_bind": int(bind.sum()), "n_cap": int(cap.sum()),
         "lift_mean_ms_where_bound": float(z["lf_lift"][bind].mean()) if bind.any() else None, "lift_max_ms": float(z["lf_lift"].max()),
         "lift_zero_where_not_bound": bool((z["lf_lift"][~bind] == 0.0).all()), "vf_nan_iff_no_band": bool(np.array_equal(np.isnan(z["lf_vf"]), ~band)),
         "vhat_range_ms": [float(z["lf_vhat"].min()), float(z["lf_vhat"].max())], "v0_range_ms": [float(z["v0"].min()), float(z["v0"].max())], "fields_present": sorted(k for k in z if k.startswith("lf_")),
         "scene": str(z["scene"]), "level_floor": str(z["level_floor"]), "floor_kappa": float(z["floor_kappa"])}
    composite = feasible == "kbox"
    base = z["plan_s"] if composite else T1.smooth_plan(z["plan_raw"].astype(np.float32), z["P"].astype(np.float32), smooth)      # the plan F received: after S (raw when S is off)
    f_plan, info = op.apply_floor_batch(base.astype(np.float32), z["v0"].astype(np.float64), z["lf_vhat"].astype(np.float64), z["lf_pi"].astype(np.float64), z["lf_gap"].astype(np.float64) / op.GAP_SCALE_M, kappa)
    issued = T1.feasible_plan(f_plan, "kbox", v0=z["v0"].astype(np.float64)) if composite else f_plan
    d = np.abs(issued.astype(np.float64) - z["plan"])
    r["recompute_max_abs_m"] = float(d.max())
    r["recompute_bind_agreement"] = float((info["bind"] == bind).mean())
    r["plan_differs_from_raw_rows"] = int((np.abs(z["plan"] - z["plan_raw"]).reshape(n, -1).max(1) > 0).sum())
    r["F_alone_changed_rows"] = int((np.abs(f_plan.astype(np.float64) - base.astype(np.float64)).reshape(n, -1).max(1) > 0).sum())
    r["PASS"] = bool(r["recompute_max_abs_m"] <= 1e-4 and r["recompute_bind_agreement"] == 1.0 and r["lift_zero_where_not_bound"] and r["vf_nan_iff_no_band"])
    hdr = json.load(open(f"{SM}/{tag}/t1v8f_{tag}.json", encoding="utf-8"))["t1v8f"]
    r["header"] = {k: hdr.get(k) for k in ("level_floor", "floor_kappa", "floor_ckpt", "floor_ckpt_sha256", "scene", "smooth", "feasible")}
    r["header_detail_counts"] = hdr["level_floor_detail"]["counts"]
    r["header_parity"] = hdr["level_floor_detail"]["heads_parity_vs_forward_infer"]
    r["header_keys_read"] = hdr["level_floor_detail"]["floor_model_batch_keys_read"]
    return r


def main():
    T1 = tool_module()
    res = {"card": "LEVEL-FLOOR smoke: kappa 0.5, s0, 6 windows, reindexed", "clock": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    s1 = run("SMK_LF_composite", 6, 12, "sg72", "kbox", 0.5)
    res["S1_composite"] = dict(s1, **check(T1, "SMK_LF_composite", 0.5, "sg72", "kbox"))
    s2 = run("SMK_LF_bare", 6, 12, "sg72", "off", 0.5)
    res["S2_bare"] = dict(s2, **check(T1, "SMK_LF_bare", 0.5, "sg72", "off"))
    s3 = run("SMK_LF_raw", 6, 12, "off", "off", 0.5)
    res["S3_smooth_off"] = dict(s3, **check(T1, "SMK_LF_raw", 0.5, "off", "off"))
    if "--skip-timing" not in sys.argv:
        t = {}
        for rep in (1, 2):
            for name, kap in (("off", None), ("on_k0.5", 0.5)):
                t.setdefault(name, []).append(run(f"TIME_{name}_{rep}", 64, 12, "sg72", "kbox", kap))
        res["timing_64_windows_composite_12t"] = {k: {"loop_s": [v["loop_s"] for v in vs], "wall_s": [round(v["wall_s"], 1) for v in vs], "loop_s_per_window_min": min(v["loop_s"] for v in vs) / 64.0,
                                                      "wall_s_per_window_min": min(v["wall_s"] for v in vs) / 64.0} for k, vs in t.items()}
    res["PASS"] = bool(res["S1_composite"]["PASS"] and res["S2_bare"]["PASS"] and res["S3_smooth_off"]["PASS"] and res["S2_bare"]["n_bind"] > 0)
    json.dump(res, open(f"{PK}/raw/SMOKE_LF.json", "w", encoding="utf-8", newline="\n"), indent=1, default=str)
    for k in ("S1_composite", "S2_bare", "S3_smooth_off"):
        v = res[k]
        print(k, {x: v[x] for x in ("replans", "windows", "band_rate", "bind_rate", "bind_k10_rate", "cap_rate", "n_bind", "n_cap", "lift_mean_ms_where_bound", "recompute_max_abs_m", "recompute_bind_agreement", "PASS")})
    if "timing_64_windows_composite_12t" in res:
        print("timing", json.dumps(res["timing_64_windows_composite_12t"]))
    print("SMOKE_LF", "PASS" if res["PASS"] else "FAIL")
    sys.exit(0 if res["PASS"] else 2)


if __name__ == "__main__":
    main()
