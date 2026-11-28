#!/usr/bin/env python3
"""Score ONE arm with the OFFICIAL, UNMODIFIED NavSim v2 two-stage scorer (NAVSIM VENV).

    python code/score_arm6.py --arm CV_official --official-agent constant_velocity_agent
    python code/score_arm6.py --arm STOP_zero --seam <e2>/raw/seam_STOP_zero.npz
    python code/score_arm6.py --arm R6_A1 --seam raw/seam_R6_A1.npz --split navhard_two_stage

⭐ ADAPTED from E2's ``2026-09-19-navsim-refcv4b-bridge/code/score_arm.py`` (the refcv4b
bridge). What is IDENTICAL, on purpose, so that a refcv6 row and an E2 row are scored under ONE
harness: the interpreter (``C:/Users/Admin/navsim-crun/venv``), the devkit copy and data mirror
(``C:/Users/Admin/navsim-crun``), the metric caches, the Hydra overrides, ``PYTHONHASHSEED=1``,
the thread caps, ``CUDA_VISIBLE_DEVICES=-1``, E1's wrapper with ``--patch-loader`` (the ONE
behaviour patch: the Windows loader separator), E2's seam agent (``tanitad_seam_agent.py`` —
IMPORTED from E2's package by ``PYTHONPATH``, never copied), and E2's count guards.

What CHANGED, and why:
  * ``--exp-tag`` (default ``e6``): the devkit output dir and experiment name no longer collide
    with E2's scratch (``exp/e2/...``); nothing of E2's is overwritten.
  * ``--out``: every artifact goes to THIS package's ``raw/`` (or the directory given).
  * the wrapper is E1's CURRENT file on D: (sha256 recorded per run). ⚠️ It differs from the
    copy E2's STOP run used (``c40d18e1…``) in two OBSERVATION/robustness changes only — the RAM
    guard's sustain (3 -> 60 samples) and a pre-aggregation CSV dump. Whether that holds is not
    argued: it is exactly what the harness-reproduction control (``raw/HARNESS_REPRO.json``)
    measures by re-scoring E2's CV and STOP and comparing every per-token cell.

Every run is REFUSED unless it did the work (E2's K8 guard): the log reports
``successful == expected`` and ``failed == 0``; the CSV holds exactly that many valid token rows
plus the 3 ``extended_pdm_score_*`` summary rows; for a seam arm the agent call log shows one
call per token with the seam / stand-in split the seam declares. "Success over an empty set"
is a FAILURE.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
NAVSIM = "C:/Users/Admin/navsim"            # junction -> D:/Archive/devbox-C/navsim (E1/E2's)
CR = "C:/Users/Admin/navsim-crun"           # E1's verified C: mirror (devkit + data + venv)
WRAP = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
        "2026-09-19-navsim-warmup-reference-epdms/code/navsim_win.py")
#: E2's seam agent, IMPORTED (PYTHONPATH), never copied. The ev6 archive is preferred (it is the
#: tip the refcv6 run was launched from); the D: copy is the fallback. Both are recorded.
E2_CODE_CANDIDATES = (
    os.path.join(os.path.abspath(os.path.join(PKG, "..", "..")),
                 "2026-09-19-navsim-refcv4b-bridge", "code"),
    "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
    "2026-09-19-navsim-refcv4b-bridge/code",
)
ENV = {
    "NUPLAN_MAP_VERSION": "nuplan-maps-v1.0",
    "NUPLAN_MAPS_ROOT": f"{CR}/data/maps",
    "NAVSIM_EXP_ROOT": f"{NAVSIM}/exp",
    "NAVSIM_DEVKIT_ROOT": f"{CR}/devkit",
    "OPENSCENE_DATA_ROOT": f"{CR}/data/openscene",
    "PYTHONHASHSEED": "1",
    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
    "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8",
    "CUDA_VISIBLE_DEVICES": "-1",   # Windows drops EMPTY env vars; -1 hides every GPU
}
SPLITS = {
    "warmup_two_stage": {"n_stage1": 16, "n_stage2": 204,
                         "cache": f"{NAVSIM}/exp/metric_cache_warmup_two_stage",
                         "syn_sensors": f"{NAVSIM}/data/openscene/warmup_two_stage/sensor_blobs",
                         "syn_scenes": f"{CR}/data/openscene/warmup_two_stage/synthetic_scene_pickles"},
    "navhard_two_stage": {"n_stage1": 450, "n_stage2": 5462,
                          # ⚠️ NOT E2's path ({NAVSIM}/exp/...): MEASURED 2026-09-24, that directory
                          # holds per-log folders but NO metadata CSV (the loader dies IndexError);
                          # the COMPLETE cache (5,912 rows) is the C: mirror's, the one W7 scored with.
                          "cache": f"{CR}/exp/metric_cache_navhard_two_stage",
                          "syn_sensors": f"{CR}/data/openscene/navhard_two_stage/sensor_blobs",
                          "syn_scenes": f"{CR}/data/openscene/navhard_two_stage/synthetic_scene_pickles"},
}


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def e2_code_dir() -> str:
    for c in E2_CODE_CANDIDATES:
        if os.path.isfile(os.path.join(c, "tanitad_seam_agent.py")):
            return os.path.abspath(c)
    raise SystemExit(f"⛔ E2's tanitad_seam_agent.py not found in {E2_CODE_CANDIDATES}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seam", default="")
    ap.add_argument("--official-agent", default="")
    ap.add_argument("--split", default="warmup_two_stage", choices=sorted(SPLITS))
    ap.add_argument("--cache", default="")
    ap.add_argument("--out", default=os.path.join(PKG, "raw"))
    ap.add_argument("--exp-tag", default="e6")
    ap.add_argument("--worker", default="sequential")
    ap.add_argument("--ram-floor-mb", type=float, default=3000.0)
    ap.add_argument("--python", default=f"{CR}/venv/Scripts/python.exe")
    a = ap.parse_args(argv)
    sp = SPLITS[a.split]
    cache = a.cache or sp["cache"]
    if not os.path.isdir(os.path.join(cache, "metadata")):
        print(f"⛔ no metric cache at {cache}")
        return 2
    os.makedirs(a.out, exist_ok=True)
    tag = a.arm if a.split == "warmup_two_stage" else f"{a.arm}__{a.split}"
    exp_name = f"{a.exp_tag}_{a.split}_{a.arm}"
    call_log = os.path.join(a.out, f"score_{tag}.calls.jsonl")
    if os.path.exists(call_log):
        os.remove(call_log)
    ov = [f"train_test_split={a.split}", f"experiment_name={exp_name}",
          f"metric_cache_path={cache}", f"synthetic_sensor_path={sp['syn_sensors']}",
          f"synthetic_scenes_path={sp['syn_scenes']}", f"worker={a.worker}"]
    seam_sha = None
    if a.official_agent:
        ov.append(f"agent={a.official_agent}")
    else:
        if not a.seam or not os.path.exists(a.seam):
            print(f"⛔ seam file missing: {a.seam!r}")
            return 2
        seam_sha = sha256_file(a.seam)
        ov += ["agent=constant_velocity_agent",
               "agent._target_=tanitad_seam_agent.TanitADSeamAgent",
               f"+agent.seam_file={os.path.abspath(a.seam).replace(os.sep, '/')}",
               f"+agent.call_log={os.path.abspath(call_log).replace(os.sep, '/')}"]
    e2c = e2_code_dir()
    env = dict(os.environ)
    env.update(ENV)
    env["PYTHONPATH"] = e2c + os.pathsep + env.get("PYTHONPATH", "")
    hook_dir = os.path.join(a.out, f"score_{tag}_wrapper")
    os.makedirs(hook_dir, exist_ok=True)
    out_dir = f"{NAVSIM}/exp/{a.exp_tag}/{a.split}_{a.arm}"
    ov.append(f"output_dir={out_dir}")
    cmd = [a.python, WRAP, "--script", "pdm_score", "--label", tag, "--out-dir", hook_dir,
           "--patch-loader", "--ram-floor-mb", str(a.ram_floor_mb), "--dump-final-scores",
           os.path.join(hook_dir, f"{tag}_final_scores_frame.csv"), "--"] + ov
    wrap_sha = sha256_file(WRAP)
    agent_sha = sha256_file(os.path.join(e2c, "tanitad_seam_agent.py"))
    log_path = os.path.join(a.out, f"score_{tag}.log")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("CMD: " + " ".join(cmd) + "\n")
        fh.flush()
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                            cwd=NAVSIM).returncode
    wall = time.time() - t0
    text = open(log_path, encoding="utf-8", errors="replace").read()
    m_ok = re.findall(r"Number of successful scenarios:\s*(\d+)", text)
    m_bad = re.findall(r"Number of failed scenarios:\s*(\d+)", text)
    m_res = re.findall(r"Results are stored in:\s*(\S+\.csv)", text)
    expected = sp["n_stage1"] + sp["n_stage2"]
    fails = []
    n_ok = int(m_ok[-1]) if m_ok else None
    n_bad = int(m_bad[-1]) if m_bad else None
    if n_ok is None:
        fails.append("summary line 'Number of successful scenarios' ABSENT — cannot confirm work")
    elif n_ok != expected:
        fails.append(f"successful {n_ok} != expected {expected}")
    if n_bad not in (0,):
        fails.append(f"failed scenarios = {n_bad}")
    csv_src = m_res[-1].rstrip(".") if m_res else None
    if csv_src is None:
        cands = sorted(glob.glob(f"{out_dir}/*.csv"), key=os.path.getmtime)
        csv_src = cands[-1] if cands else None
    n_rows = n_valid = None
    csv_dst = os.path.join(a.out, f"score_{tag}.csv")
    if csv_src and os.path.exists(csv_src):
        shutil.copyfile(csv_src, csv_dst)
        with open(csv_dst, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        tok_rows = [r for r in rows if not r["token"].startswith("extended_pdm_score")]
        n_rows = len(tok_rows)
        n_valid = sum(1 for r in tok_rows if r["valid"] in ("True", "true", "1"))
        if n_rows != expected or n_valid != expected:
            fails.append(f"CSV token rows {n_rows} / valid {n_valid} != expected {expected}")
        summ = [r["token"] for r in rows if r["token"].startswith("extended_pdm_score")]
        if len(summ) != 3:
            fails.append(f"CSV summary rows {summ}")
    else:
        fails.append("no result CSV found")
    calls = None
    if not a.official_agent:
        calls = {"seam": 0, "cv_standin": 0, "UNKNOWN_TOKEN": 0}
        if os.path.exists(call_log):
            for ln in open(call_log, encoding="utf-8"):
                r = json.loads(ln)
                if r.get("event") == "call":
                    calls[r["source"]] = calls.get(r["source"], 0) + 1
        import numpy as np
        z = np.load(a.seam, allow_pickle=False)
        src = [str(x) for x in z["source"]]
        want = {"seam": sum(1 for s in src if s != "cv_standin"),
                "cv_standin": sum(1 for s in src if s == "cv_standin")}
        bad = {k: v for k, v in calls.items() if k not in ("seam", "cv_standin") and v}
        if bad or calls["seam"] != want["seam"] or calls["cv_standin"] != want["cv_standin"]:
            fails.append(f"agent calls {calls} != seam declaration {want}")
    rep = {"arm": a.arm, "split": a.split, "tag": tag, "cmd": cmd, "rc": rc,
           "wall_s": round(wall, 1), "expected_tokens": expected, "log_successful": n_ok,
           "log_failed": n_bad, "csv_src": csv_src,
           "csv": csv_dst if os.path.exists(csv_dst) else None,
           "csv_token_rows": n_rows, "csv_valid_rows": n_valid, "agent_calls": calls,
           "cache": cache, "worker": a.worker, "wrapper": WRAP, "wrapper_sha256": wrap_sha,
           "seam": os.path.abspath(a.seam) if a.seam else None, "seam_sha256": seam_sha,
           "seam_agent_dir": e2c, "seam_agent_sha256": agent_sha,
           "mirror": CR, "exp_output_dir": out_dir,
           "status": "FAIL" if fails else "PASS", "failures": fails}
    with open(os.path.join(a.out, f"score_{tag}.counts.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: rep[k] for k in ("arm", "split", "status", "log_successful",
                                          "log_failed", "csv_valid_rows", "agent_calls",
                                          "wall_s", "failures")}), flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
