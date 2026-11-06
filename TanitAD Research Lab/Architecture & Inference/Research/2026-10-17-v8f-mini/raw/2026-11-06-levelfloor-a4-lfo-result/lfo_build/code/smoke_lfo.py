#!/usr/bin/env python3
"""smoke_lfo.py -- the LFO tool on REAL data with ``--floor-vhat-source logged`` (64 windows, reindexed, C0_s0 + the LLR_s0 heads, kappa 0.75, CPU 12 threads): the wiring check of KA-O3 on the real rollout.
Cases: LFO_bare (--smooth sg72) and LFO_comp (--smooth sg72 --feasible kbox). Checks per case, on the dumped ticks:
  * ``lf_vhat`` equals the READER's ``logged_speed`` of the same tick's dumped ``gt`` / ``gt_ok`` (float32 rounding, 1e-5 m/s) and is NaN exactly where that is NaN;
  * a tick with no complete logged route has no floor (``lf_vf`` NaN, ``lf_bind`` False); a tick with v_log < 3 m/s has no floor either;
  * v_f = kappa * v_log wherever the lead cap did not bind (1e-4 m/s, float32); v_f <= kappa * v_log everywhere;
  * ``lf_vhat_heads`` is present and differs from ``lf_vhat`` on some tick (the heads' estimate is a different quantity);
  * the result JSON header carries ``floor_vhat_source`` "logged" and ``label`` "CEILING"; the ticks file carries ``floor_vhat_source`` and ``label``.
Writes raw/SMOKE_LFO.json.   usage: python -I smoke_lfo.py"""
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
ART = "E:/Projects/TanitAD-artifacts"
MC = f"{ART}/dagland/motion_c"
RDB = f"{ART}/v8f-recover"
PY = "C:/Users/Admin/venvs/tanitad/Scripts/python.exe"
SW = "F:/TanitAD-scratch/motion_c/lfo_build/smoke"
TOOL = "F:/TanitAD-scratch/motion_c/lfo_read/tree/stack/scripts/v8f_t1_eval_ri.py"
DATA = f"{ART}/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
KAPPA = 0.75
CASES = {"LFO_bare": [], "LFO_comp": ["--feasible", "kbox"]}


def load_reader():
    spec = importlib.util.spec_from_file_location("lf_read_for_smoke_lfo", f"{MC}/lf_read/code/lf_read.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_case(tag, extra, n_win=64):
    od = f"{SW}/{tag}"
    if os.path.isdir(od):
        os.replace(od, f"{od}_old_{time.strftime('%H%M%S')}")
    os.makedirs(od)
    cmd = [PY, "-I", TOOL, "run", "--ckpt", f"{RDB}/M2b_s0_recft_C0/ckpt_last.pt", "--tag", tag, "--tracks", f"{ART}/sfresh_inputs/inputs/native_tracks.pt", "--eval-cache", f"{ART}/sfresh_inputs/cache",
           "--eval-grid", f"{DATA}/eval_grid_windows.json", "--eval-join", f"{DATA}/join_eval141.npz", "--out-dir", od, "--replan-steps", "10", "--nav", "deadreckon", "--schedule", "periodic", "--limit", "causal",
           "--dump-ticks", f"{od}/ticks", "--dump-heads", "--device", "cpu", "--smooth", "sg72", "--scene", "reindexed", "--level-floor", "on", "--floor-kappa", str(KAPPA),
           "--floor-ckpt", f"{RDB}/M2b_s0_recft_LLR/ckpt_last.pt", "--floor-vhat-source", "logged", "--max-windows", str(n_win), "--n-boot", "20"] + extra
    env = dict(os.environ, OMP_NUM_THREADS="12", MKL_NUM_THREADS="12", OPENBLAS_NUM_THREADS="12", CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1")
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    open(f"{PK}/logs/SMOKE_LFO_{tag}.out", "w", encoding="utf-8", newline="\n").write(" ".join(cmd) + "\n" + r.stdout[-3000:] + r.stderr[-3000:])
    if r.returncode:
        raise SystemExit(f"smoke {tag} failed rc={r.returncode}: {r.stderr[-800:]}")
    return od, time.time() - t0


def check(LFR, tag, od):
    t = dict(np.load(f"{od}/ticks/ticks_{tag}.npz", allow_pickle=True))
    h = json.load(open(f"{od}/t1v8f_{tag}.json", encoding="utf-8"))["t1v8f"]
    vh = np.asarray(t["lf_vhat"], np.float64)
    vf = np.asarray(t["lf_vf"], np.float64)
    bind, cap = np.asarray(t["lf_bind"], bool), np.asarray(t["lf_cap"], bool)
    v_log, ok = LFR.logged_speed(t["gt"], t["gt_ok"])
    nan_same = bool(np.array_equal(np.isnan(vh), np.isnan(v_log)))
    d = float(np.nanmax(np.abs(vh - v_log))) if np.isfinite(v_log).any() else None
    nofloor_a = ~np.isfinite(v_log)
    nofloor_b = np.isfinite(v_log) & (v_log < 3.0)
    band = np.isfinite(v_log) & (v_log >= 3.0)
    uncapped = band & ~cap
    res = {"tag": tag, "n_ticks": int(len(vh)), "n_no_route_ticks": int(nofloor_a.sum()), "n_below_band_ticks": int(nofloor_b.sum()), "n_band_ticks": int(band.sum()), "n_bind": int(bind.sum()), "n_cap": int(cap.sum()),
           "lf_vhat_equals_reader_logged_speed_max_abs": d, "nan_pattern_equal": nan_same,
           "no_floor_where_no_route": bool(np.isnan(vf[nofloor_a]).all() and not bind[nofloor_a].any()), "no_floor_below_3mps": bool(np.isnan(vf[nofloor_b]).all() and not bind[nofloor_b].any()),
           "vf_equals_kappa_vlog_where_uncapped_max_abs": float(np.abs(vf[uncapped] - KAPPA * v_log[uncapped]).max()) if uncapped.any() else None,
           "vf_le_kappa_vlog_everywhere": bool((vf[band] <= KAPPA * v_log[band] + 1e-4).all()),
           "heads_column_present": "lf_vhat_heads" in t, "heads_differs_from_oracle": bool("lf_vhat_heads" in t and np.nanmax(np.abs(np.asarray(t["lf_vhat_heads"], np.float64) - vh)) > 1e-3),
           "header_floor_vhat_source": h.get("floor_vhat_source"), "header_label": h.get("label"), "ticks_floor_vhat_source": str(t.get("floor_vhat_source")), "ticks_label": str(t.get("label")),
           "header_vision_only_text": (h.get("level_floor_detail") or {}).get("vision_only", "")[:60], "open_channels_names_the_oracle": any("LEVEL INPUT (CEILING" in c for c in h.get("open_channels", []))}
    res["PASS"] = bool(nan_same and d is not None and d <= 1e-5 and res["no_floor_where_no_route"] and res["no_floor_below_3mps"] and res["vf_le_kappa_vlog_everywhere"]
                       and (res["vf_equals_kappa_vlog_where_uncapped_max_abs"] or 0.0) <= 1e-4 and res["heads_column_present"] and res["heads_differs_from_oracle"] and res["header_floor_vhat_source"] == "logged"
                       and res["header_label"] == "CEILING" and res["ticks_floor_vhat_source"] == "logged" and res["ticks_label"] == "CEILING" and res["header_vision_only_text"].startswith("NO.") and res["open_channels_names_the_oracle"]
                       and res["n_bind"] > 0 and res["n_no_route_ticks"] > 0)
    return res


def main():
    LFR = load_reader()
    os.makedirs(f"{PK}/logs", exist_ok=True)
    out = {"card": "LFO smoke: --floor-vhat-source logged on 64 real windows", "tool": TOOL, "cases": {}}
    for tag, extra in CASES.items():
        od, wall = run_case(tag, extra)
        out["cases"][tag] = check(LFR, tag, od)
        out["cases"][tag]["wall_s"] = wall
        print(tag, "PASS" if out["cases"][tag]["PASS"] else "FAIL", {k: v for k, v in out["cases"][tag].items() if k != "PASS"}, flush=True)
    out["PASS"] = all(v["PASS"] for v in out["cases"].values())
    out["clock"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(out, open(f"{PK}/raw/SMOKE_LFO.json", "w", encoding="utf-8", newline="\n"), indent=1, default=str)
    sys.exit(0 if out["PASS"] else 2)


if __name__ == "__main__":
    main()
