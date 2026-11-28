#!/usr/bin/env python3
"""ka_lf.py -- the known answers of the LEVEL-FLOOR tool (KA-LF1, KA-LF2), on real data, CPU only.

The LF read tree (F:/TanitAD-scratch/motion_c/lf_read/tree: the LLR read tree with the RI tool 0417c838 replaced by its LF patch via apply_lf_patch.py, plus stack/scripts/level_floor.py) re-rolls the FIRST 64 windows
of the eval grid (run_tool_subset.py, the banked batches) and every plan array is compared with the BANKED rollout on the same windows (cl_read/code/ka_new.py's method and comparison, loaded BY FILE PATH with its sha256
checked, its globals re-pointed, its comparison code untouched; only its ``run_tool`` is replaced by the same command PLUS the case's extra flags):
  KA-LF1  floor OFF (no floor flag at all):
    LF1_A  C0 s0, --scene logged,    16 threads, --smooth sg72              vs the banked C0_s0     (C:/Users/Admin/rr_scratch/out/C0_s0)
    LF1_B  C0 s0, --scene reindexed, 12 threads, --smooth sg72              vs the banked RIC0_s0   (F:/TanitAD-scratch/motion_c/ri/out/RIC0_s0)
    LF1_C  C0 s0, --scene reindexed, 12 threads, --smooth sg72 --feasible kbox   vs the banked RIC0C_s0 (beside; the composite path)
  KA-LF2  the floor ON with kappa = 0 (``--level-floor on --floor-kappa 0 --floor-ckpt LLR_s0``), the same three cases; each vs the banked rollout AND vs the LF1 run's own arrays (bit for bit).
PASS = every tick row matched and the max |plan / plan_raw / tau / P| diff and the max |cl / ol / fl| dump diff EXACTLY 0.0 (the CL package's tolerance was 1e-4; the LF known answer is 0).
Writes raw/KA_LF.json and raw/KA_LF_<case>.json / logs/KA_LF_<case>_run.out.   usage: python -I ka_lf.py [--only LF1_A,LF1_B,...] [--skip-run]"""
import glob
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
PK = os.path.dirname(HERE)
KA = "E:/Projects/TanitAD-artifacts/dagland/motion_c/cl_read/code/ka_new.py"
KA_SHA = "c886c0c3f3fb0f2f48f51dc84ff43fdfd142214febeb66c3d6ddd2ff05b250e4"
RDB = "E:/Projects/TanitAD-artifacts/v8f-recover"
TREE = "F:/TanitAD-scratch/motion_c/lf_read/tree"
SM = "F:/TanitAD-scratch/motion_c/lf_read/ka"
LLR = f"{RDB}/M2b_s0_recft_LLR/ckpt_last.pt"
RR = "C:/Users/Admin/rr_scratch/out/C0_s0"
RI = "F:/TanitAD-scratch/motion_c/ri/out"
BANK = {"A": (f"{RR}/ticks/ticks_C0_s0.npz", f"{RR}/dump_C0_s0"), "B": (f"{RI}/RIC0_s0/ticks/ticks_RIC0_s0.npz", f"{RI}/RIC0_s0/dump_RIC0_s0"),
        "C": (f"{RI}/RIC0C_s0/ticks/ticks_RIC0C_s0.npz", f"{RI}/RIC0C_s0/dump_RIC0C_s0")}
SPEC = {"A": dict(scene="logged", th=16, feas="off", decides=True), "B": dict(scene="reindexed", th=12, feas="off", decides=True), "C": dict(scene="reindexed", th=12, feas="kbox", decides=False)}
FLOOR_ARGS = ["--level-floor", "on", "--floor-kappa", "0", "--floor-ckpt", LLR]
CASES = {}
for _g, _extra in (("LF1", []), ("LF2", FLOOR_ARGS)):
    for _k, _s in SPEC.items():
        CASES[f"{_g}_{_k}"] = dict(_s, key=_k, extra=_extra, ref_t=BANK[_k][0], ref_d=BANK[_k][1], name=f"{_g}_{_k}_C0s0_{_s['scene']}_{_s['feas']}_first64_{_s['th']}t_vs_banked_{os.path.basename(os.path.dirname(os.path.dirname(BANK[_k][0])))}")
EXTRA = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_ka():
    got = sha(KA)
    if got != KA_SHA:
        raise SystemExit(f"REFUSED: {KA} is {got}, not the pinned {KA_SHA}")
    spec = importlib.util.spec_from_file_location("ka_new_lf", KA)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_run_tool(m):
    """ka_new.run_tool's command, verbatim, plus ``EXTRA['args']`` (the case's floor flags); the wall time is kept in EXTRA['wall_s']."""
    def run_tool(threads, n_win, first=False):
        os.makedirs(m.OUT, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads), PYTHONDONTWRITEBYTECODE="1")
        cmd = ([m.PY, "-I", f"{m.HERE}/run_tool_subset.py", "--tool", m.TOOL, "--first", str(n_win), "--"] if first else [m.PY, "-I", m.TOOL]) + ["run", "--ckpt", f"{m.CKD}/ckpt_last.pt", "--tag", m.TAG,
               "--tracks", f"{m.ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{m.ART}/sfresh_inputs/cache", "--eval-grid", f"{m.DATA}/eval_grid_windows.json", "--eval-join", f"{m.DATA}/join_eval141.npz",
               "--out-dir", m.OUT, "--replan-steps", "10", "--nav", "deadreckon", "--schedule", "periodic", "--limit", "causal", "--smooth", "sg72", "--scene", m.SCENE,
               "--dump-ticks", f"{m.OUT}/ticks", "--dump-heads", "--device", "cpu"] + ([] if first else ["--max-windows", str(n_win)]) + ["--n-boot", "20"] + (["--feasible", m.FEAS] if m.FEAS != "off" else []) + EXTRA.get("args", [])
        t0 = time.time()
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        EXTRA["wall_s"] = time.time() - t0
        open(f"{PK}/logs/{m.RESULT}_run.out", "w", encoding="utf-8", newline="\n").write(" ".join(cmd) + "\n" + r.stdout + r.stderr)
        if r.returncode:
            print("KALF RUN FAILED rc", r.returncode, r.stderr[-1500:])
            sys.exit(1)
        print(f"{m.RESULT} run: {EXTRA['wall_s']:.1f} s, {threads} threads, {n_win} windows")
    return run_tool


def dumps(m, d):
    return m.dumps(d)


def compare_runs(m, a_dir, a_tag, b_dir, b_tag):
    """Bit comparison of two runs of the SAME windows: every common ticks field and the dump arrays; returns max diffs and whether every common field is bit-identical."""
    za = {k: v for k, v in np.load(f"{a_dir}/ticks/ticks_{a_tag}.npz", allow_pickle=True).items()}
    zb = {k: v for k, v in np.load(f"{b_dir}/ticks/ticks_{b_tag}.npz", allow_pickle=True).items()}
    out = {"ticks": {}, "dumps": {}, "only_in_a": sorted(set(za) - set(zb)), "only_in_b": sorted(set(zb) - set(za))}
    bit = True
    for k in sorted(set(za) & set(zb)):
        x, y = za[k], zb[k]
        if x.dtype.kind in "fiub" and x.shape == y.shape:
            same = bool(np.array_equal(x, y, equal_nan=x.dtype.kind == "f"))
            d = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size and x.dtype.kind in "fiub" and not same else 0.0
            out["ticks"][k] = {"bit_identical": same, "max_abs": d}
            bit &= same
        elif x.dtype.kind in "fiub":
            out["ticks"][k] = {"bit_identical": False, "shape": [list(x.shape), list(y.shape)]}
            bit = False
    da, db = dumps(m, f"{a_dir}/dump_{a_tag}"), dumps(m, f"{b_dir}/dump_{b_tag}")
    for k in ("cl", "ol", "fl", "g", "v0"):
        same = bool(np.array_equal(da[k], db[k]))
        out["dumps"][k] = {"bit_identical": same}
        bit &= same
    out["all_common_fields_bit_identical"] = bit
    return out


def run_case(m, key, skip_run):
    c = CASES[key]
    m.PK, m.TREE, m.CKD = PK, TREE, f"{RDB}/M2b_s0_recft_C0"
    EXTRA["args"] = c["extra"]
    out_dir, tag = f"{SM}/KA_{key}", f"KALF_{key}"
    argv = ["ka_lf", "--first", "--threads", str(c["th"]), "--max-windows", "64", "--scene", c["scene"], "--feasible", c["feas"], "--tree", TREE, "--tag", tag, "--out-dir", out_dir,
            "--ref-ticks", c["ref_t"], "--ref-dump", c["ref_d"], "--result-name", f"KA_LF_{key}"] + (["--skip-run"] if skip_run else [])
    old, rc = sys.argv, None
    sys.argv = argv
    try:
        m.main()
    except SystemExit as e:
        rc = e.code
    finally:
        sys.argv = old
    d = json.load(open(f"{PK}/raw/KA_LF_{key}.json", encoding="utf-8"))
    hdr = json.load(open(f"{out_dir}/t1v8f_{tag}.json", encoding="utf-8"))["t1v8f"]
    res = {"key": key, "name": c["name"], "decides": c["decides"], "rc": rc, "max_plan_diff_m": d["max_plan_diff_m"], "max_dump_diff_m": d["max_dump_diff_m"], "rows": d["n_tick_rows_matched"], "windows": d["n_windows_new"],
           "threads": c["th"], "scene": c["scene"], "feasible": c["feas"], "floor": bool(c["extra"]), "ckpt_sha256": d["ckpt_sha256"], "tool_sha256": d["tool_sha256"], "reference_ticks": c["ref_t"],
           "wall_s": EXTRA.get("wall_s"), "header_level_floor": hdr.get("level_floor"), "header_floor_kappa": hdr.get("floor_kappa"), "header_floor_ckpt_sha256": hdr.get("floor_ckpt_sha256"),
           "level_floor_detail": hdr.get("level_floor_detail")}
    res["PASS"] = bool(d["n_tick_rows_matched"] > 0 and not d["rows_missing_in_banked"] and d["max_plan_diff_m"] == 0.0 and d["max_dump_diff_m"] == 0.0)
    return res


def main():
    only = list(CASES)
    for i, t in enumerate(sys.argv):
        if t == "--only":
            only = sys.argv[i + 1].split(",")
    skip = "--skip-run" in sys.argv
    m = load_ka()
    m.run_tool = make_run_tool(m)
    res, out = {"card": "KA-LF1 / KA-LF2 of the LEVEL-FLOOR build: the LF tool with the floor off / on at kappa 0 re-rolls C0 s0 and matches the banked rollouts", "tol_m": 0.0, "tree": TREE, "cases": {}}, {}
    for k in only:
        out[k] = run_case(m, k, skip)
        res["cases"][k] = out[k]
    for k in only:                                                                                   # KA-LF2 vs the LF1 run's own arrays, bit for bit
        if k.startswith("LF2_") and f"LF1_{k[4:]}" in out:
            k1 = f"LF1_{k[4:]}"
            out[k]["vs_floor_off_run"] = compare_runs(m, f"{SM}/KA_{k}", f"KALF_{k}", f"{SM}/KA_{k1}", f"KALF_{k1}")
            out[k]["PASS"] = bool(out[k]["PASS"] and out[k]["vs_floor_off_run"]["all_common_fields_bit_identical"])
    res["model_sha256"], res["data_sha256"], res["trainer_sha256"] = (sha(f"{TREE}/{p}") for p in ("stack/tanitad/models/v8f_mini.py", "stack/tanitad/data/v8f_mini_data.py", "stack/scripts/v8f_mini_train.py"))
    res["tool_sha256"], res["operator_sha256"] = sha(f"{TREE}/stack/scripts/v8f_t1_eval_ri.py"), sha(f"{TREE}/stack/scripts/level_floor.py")
    res["decides"] = "KA-LF1 A and B, KA-LF2 A and B (C is beside)"
    res["PASS"] = bool(all(out[k]["PASS"] for k in out if out[k]["decides"]))
    res["max_plan_diff_m"] = max(v["max_plan_diff_m"] for v in out.values())
    res["clock"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(res, open(f"{PK}/raw/KA_LF.json", "w", encoding="utf-8", newline="\n"), indent=1, default=str)
    print("KA_LF", {k: (v["PASS"], v["max_plan_diff_m"], v["max_dump_diff_m"], v["rows"], round(v["wall_s"] or 0, 1)) for k, v in out.items()}, "->", "PASS" if res["PASS"] else "FAIL")
    sys.exit(0 if res["PASS"] else 2)


if __name__ == "__main__":
    main()
