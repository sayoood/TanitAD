#!/usr/bin/env python3
"""Score EVERY one of P7's 64 proposals on the 1,058 C3 tokens with the harness (score_e.py UNCHANGED), serially, resumable.

Same method as raw/2026-10-11-c3-selection-headroom/c3_run.py stage_score (one single-proposal seam per index k), but
on P7's proposals from stage2_p7_full.proposals.npz. Writes seams + score dirs under F:/tanitad_scratch/p7veto/ and the
table to veto/table_p7.npz (pdms, sub, valid per token x proposal) + veto/gate_table.json.

  python score_p7_cols.py score [--only 0,1]     # scoring (long; run detached)
  python score_p7_cols.py table                  # build table + gates G1/G2
"""
import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
import time

import numpy as np

HOLD = "E:/Projects/TanitAD-artifacts/scorer_gym_20261107/p7_next/veto"
WORK = "F:/tanitad_scratch/p7veto"
PK = "F:/tanitad_scratch/p7veto/tree/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
EVAL = PK + "/eval"
HARNESS = EVAL + "/harness"
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"
NAVSIM_ROOT = "F:/Archive/devbox-C/navsim"
DUMP = "E:/Projects/TanitAD/data/refe_navtest/seams/stage2_p7_full.proposals.npz"
BANKED_CSV = "E:/Projects/TanitAD/data/refe_navtest/score/refe_stage2_p7_full/refe_stage2_p7_full.csv"
TOKENS = ("E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/"
          "raw/2026-10-11-c3-selection-headroom/tokens.json")
SUB = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress",
       "time_to_collision_within_bound", "comfort", "driving_direction_compliance")
M = 64


def label(k):
    return f"refe_p7veto_p{k:02d}"


def seam_k(k):
    return f"{WORK}/seams/{label(k)}.npz"


def csv_k(k):
    return f"{WORK}/score/{label(k)}/{label(k)}.csv"


def free_mb():
    """min(free physical, free commit) in MB (the dev box is memory-tight)."""
    r = subprocess.run(["powershell", "-NoProfile", "-Command",
                        "$o=Get-CimInstance Win32_OperatingSystem; [int]([math]::Min($o.FreePhysicalMemory,$o.FreeVirtualMemory)/1024)"],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip())
    except ValueError:
        return 0


def status_of(txt):
    for line in reversed(txt.splitlines()):
        if line.startswith("{") and '"status"' in line:
            return json.loads(line)
    return None


def load_subset():
    doc = json.load(open(TOKENS, encoding="utf-8"))
    toks = list(doc["tokens"])
    D = np.load(DUMP)
    idx = {str(t): i for i, t in enumerate(D["token"])}
    ii = np.array([idx[t] for t in toks])
    return doc, toks, D, ii


def read_csv(p):
    rows = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["token"] == "average":
            continue
        rows[r["token"]] = (r["valid"] == "True", float(r["score"]), tuple(float(r[c]) for c in SUB))
    return rows


def stage_score(only):
    sys.path.insert(0, EVAL)
    import harness_pins as HP
    hp = HP.require(pathlib_path(HARNESS), caller=pathlib_path(EVAL + "/score_navtest_refe.py"))
    print("harness pins ok", hp["n_files"], hp["pins_file_sha256"], flush=True)
    doc, toks, D, ii = load_subset()
    N = len(toks)
    os.makedirs(f"{WORK}/seams", exist_ok=True)
    os.makedirs(f"{WORK}/logs", exist_ok=True)
    P = D["proposals"][ii]
    ks = list(range(M)) if only is None else only
    for k in range(M):
        if not os.path.exists(seam_k(k)):
            np.savez(seam_k(k), token=D["token"][ii], fingerprint=D["fingerprint"][ii], poses=P[:, k].astype(np.float32),
                     sampling=D["sampling"], arm=np.array(f"REFe_p7veto_p{k:02d}"))
    t1 = time.time()
    for n, k in enumerate(ks):
        lp = f"{WORK}/logs/p{k:02d}.log"
        ok = False
        for att in range(1, 5):
            if os.path.exists(csv_k(k)) and os.path.exists(lp):
                st = status_of(open(lp, encoding="utf-8", errors="replace").read())
                if st and st.get("status") == "PASS" and st.get("csv_valid_rows") == N:
                    ok = True
                    break
            while free_mb() < 4000:
                time.sleep(30)
            env = dict(os.environ, PYTHONIOENCODING="utf-8", RUN_V1_FIX1_ALLOW_DRIFT="1",
                       TANITAD_NAVSIM_ROOT=NAVSIM_ROOT, TANITAD_REFE_PKG=PK, CUDA_VISIBLE_DEVICES="-1",
                       TMP=f"{WORK}/tmp", TEMP=f"{WORK}/tmp")
            os.makedirs(f"{WORK}/tmp", exist_ok=True)
            with open(lp, "w", encoding="utf-8") as f:
                subprocess.call([PY_D, "score_e.py", "--label", label(k), "--seam", seam_k(k), "--tokens", TOKENS,
                                 "--out", f"{WORK}/score"], cwd=HARNESS, env=env, stdout=f, stderr=subprocess.STDOUT)
            st = status_of(open(lp, encoding="utf-8", errors="replace").read())
            if st and st.get("status") == "PASS" and st.get("csv_valid_rows") == N:
                ok = True
                break
            time.sleep(60)
        print(f"[{time.strftime('%H:%M:%S')}] col {k} {'PASS' if ok else 'FAIL'} ({n + 1}/{len(ks)}, {time.time() - t1:.0f}s)", flush=True)
        json.dump({"done": n + 1, "of": len(ks), "last_k": k, "ok": ok, "elapsed_s": round(time.time() - t1)},
                  open(f"{HOLD}/SCORE_STATUS.json", "w"))
        if not ok:
            print("ZZP7VETO_SCORE_FAIL", k, flush=True)
            return 1
    print("ZZP7VETO_SCORE_DONE", flush=True)
    return 0


def pathlib_path(p):
    import pathlib
    return pathlib.Path(p)


def stage_table():
    doc, toks, D, ii = load_subset()
    N = len(toks)
    pdms = np.full((N, M), np.nan)
    sub = np.full((N, M, 6), np.nan)
    valid = np.zeros((N, M), bool)
    for k in range(M):
        rows = read_csv(csv_k(k))
        for i, t in enumerate(toks):
            valid[i, k], pdms[i, k], sub[i, k] = rows[t][0], rows[t][1], rows[t][2]
    pick = D["pick"][ii]
    ban = read_csv(BANKED_CSV)
    at = pdms[np.arange(N), pick]
    dd = np.array([abs(at[i] - ban[t][1]) for i, t in enumerate(toks)])
    sd = np.array([np.abs(sub[i, pick[i]] - np.array(ban[t][2])).max() for i, t in enumerate(toks)])
    g = {"G1_all_valid": bool(valid.all()), "G1_finite": bool(np.isfinite(pdms).all()),
         "G2_max_abs_pdms_diff_at_pick_vs_banked": float(dd.max()), "G2_max_abs_sub_diff": float(sd.max()),
         "G2_n_diff_gt_1e-9": int((dd > 1e-9).sum()), "G2_PASS": bool(dd.max() <= 1e-9 and sd.max() <= 1e-9),
         "banked_csv": BANKED_CSV}
    np.savez_compressed(f"{HOLD}/table_p7.npz", token=np.array(toks), pdms=pdms, sub=sub, valid=valid,
                        logits=D["logits"][ii], proposals=D["proposals"][ii], pick=pick, sub_names=np.array(SUB),
                        pdms_at_pick_banked=np.array([ban[t][1] for t in toks]))
    json.dump(g, open(f"{HOLD}/gate_table.json", "w"), indent=1)
    print(json.dumps(g, indent=1))
    return 0 if (g["G1_all_valid"] and g["G1_finite"] and g["G2_PASS"]) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("score", "table"))
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    only = None if a.only is None else [int(x) for x in a.only.split(",")]
    sys.exit(stage_score(only) if a.stage == "score" else stage_table())
