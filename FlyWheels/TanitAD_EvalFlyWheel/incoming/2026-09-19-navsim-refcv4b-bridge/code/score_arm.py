#!/usr/bin/env python3
"""Score ONE arm with the OFFICIAL, UNMODIFIED NavSim two-stage scorer (NAVSIM VENV).

    python code/score_arm.py --arm A1_ego_cmd --seam raw/seam_A1_ego_cmd.npz
    python code/score_arm.py --arm CV_official --official-agent constant_velocity_agent

It runs the official two-stage runner ``navsim.planning.script.run_pdm_score`` as its own
process with Hydra OVERRIDES only (no devkit file is edited), THROUGH stream E1's
wrapper ``…/2026-09-19-navsim-warmup-reference-epdms/code/navsim_win.py --patch-loader``
— the SAME interpreter, devkit copy, data mirror (``C:/Users/Admin/navsim-crun``, E1's
md5-verified mirror: 215 files, 0 mismatches), loader patch, PYTHONHASHSEED and thread
caps E1 uses for its CV / human numbers, so the arms are scored under ONE harness.
⚠️ WHY THE WRAPPER IS REQUIRED (E1, MEASURED): ``MetricCacheLoader`` splits cache paths
on "/" but nuPlan writes backslash paths on Windows -> IndexError; the wrapper's only
behaviour patch replaces the separator (pinned by E1's test_navsim_win_patch.py). Its
sha256 is recorded per run. It copies the CSV and the full log into ``raw/``, and then
REFUSES a run that did no work:

* the log's own summary must report ``successful == expected`` and ``failed == 0``
  (expected = 16 stage-1 + 204 stage-2 = 220 on warmup_two_stage);
* the CSV must hold exactly that many valid token rows plus the 3 summary rows;
* for a seam arm, the agent's call log must show one call per token with the
  seam/stand-in split the seam declares.
"Success over an empty set" is a FAILURE (smoke_assert.py's class).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
NAVSIM = "C:/Users/Admin/navsim"
CR = "C:/Users/Admin/navsim-crun"          # E1's verified C: mirror (D: is I/O-saturated)
WRAP = ("D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/"
        "2026-09-19-navsim-warmup-reference-epdms/code/navsim_win.py")
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
                          "cache": f"{NAVSIM}/exp/metric_cache_navhard_two_stage",
                          # the verified C: working copy (EXTRACT_DONE.json ok: 5,462 synthetic
                          # pickles, 13,715 F0/L0/R0 images each, 76 log dirs). ⚠️ its stage-1
                          # LOGS must also be in {CR}/data/openscene/navsim_logs/test — check first.
                          "syn_sensors": f"{CR}/data/openscene/navhard_two_stage/sensor_blobs",
                          "syn_scenes": f"{CR}/data/openscene/navhard_two_stage/synthetic_scene_pickles"},
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seam", default="")
    ap.add_argument("--official-agent", default="")
    ap.add_argument("--split", default="warmup_two_stage")
    ap.add_argument("--cache", default="")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "raw"))
    ap.add_argument("--worker", default="sequential")
    ap.add_argument("--python", default=f"{CR}/venv/Scripts/python.exe")
    a = ap.parse_args(argv)
    sp = SPLITS[a.split]
    cache = a.cache or sp["cache"]
    if not os.path.isdir(os.path.join(cache, "metadata")):
        print(f"⛔ no metric cache at {cache}")
        return 2
    exp_name = f"e2_{a.split}_{a.arm}"
    call_log = os.path.join(a.out, f"score_{a.arm}.calls.jsonl")
    if os.path.exists(call_log):
        os.remove(call_log)
    ov = [f"train_test_split={a.split}", f"experiment_name={exp_name}",
          f"metric_cache_path={cache}", f"synthetic_sensor_path={sp['syn_sensors']}",
          f"synthetic_scenes_path={sp['syn_scenes']}", f"worker={a.worker}"]
    if a.official_agent:
        ov.append(f"agent={a.official_agent}")
    else:
        if not a.seam or not os.path.exists(a.seam):
            print(f"⛔ seam file missing: {a.seam!r}")
            return 2
        ov += ["agent=constant_velocity_agent",
               "agent._target_=tanitad_seam_agent.TanitADSeamAgent",
               f"+agent.seam_file={os.path.abspath(a.seam).replace(os.sep, '/')}",
               f"+agent.call_log={os.path.abspath(call_log).replace(os.sep, '/')}"]
    env = dict(os.environ)
    env.update(ENV)
    env["PYTHONPATH"] = HERE + os.pathsep + env.get("PYTHONPATH", "")
    hook_dir = os.path.join(a.out, f"score_{a.arm}_wrapper")
    os.makedirs(hook_dir, exist_ok=True)
    ov.append(f"output_dir={NAVSIM}/exp/e2/{a.split}_{a.arm}")
    cmd = [a.python, WRAP, "--script", "pdm_score", "--label", a.arm, "--out-dir", hook_dir,
           "--patch-loader", "--dump-final-scores",
           os.path.join(hook_dir, f"{a.arm}_final_scores_frame.csv"), "--"] + ov
    import hashlib
    wrap_sha = hashlib.sha256(open(WRAP, "rb").read()).hexdigest()
    log_path = os.path.join(a.out, f"score_{a.arm}.log")
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
        cands = sorted(glob.glob(f"{NAVSIM}/exp/e2/{a.split}_{a.arm}/*.csv"), key=os.path.getmtime)
        csv_src = cands[-1] if cands else None
    n_rows = n_valid = None
    csv_dst = os.path.join(a.out, f"score_{a.arm}.csv")
    if csv_src and os.path.exists(csv_src):
        shutil.copyfile(csv_src, csv_dst)
        import csv
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
    rep = {"arm": a.arm, "split": a.split, "cmd": cmd, "rc": rc, "wall_s": round(wall, 1),
           "expected_tokens": expected, "log_successful": n_ok, "log_failed": n_bad,
           "csv_src": csv_src, "csv": csv_dst if os.path.exists(csv_dst) else None,
           "csv_token_rows": n_rows, "csv_valid_rows": n_valid, "agent_calls": calls,
           "cache": cache, "worker": a.worker, "wrapper": WRAP, "wrapper_sha256": wrap_sha,
           "mirror": CR, "status": "FAIL" if fails else "PASS",
           "failures": fails}
    with open(os.path.join(a.out, f"score_{a.arm}.counts.json"), "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=1)
    print(json.dumps({k: rep[k] for k in ("arm", "status", "log_successful", "log_failed",
                                          "csv_valid_rows", "agent_calls", "wall_s",
                                          "failures")}))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
