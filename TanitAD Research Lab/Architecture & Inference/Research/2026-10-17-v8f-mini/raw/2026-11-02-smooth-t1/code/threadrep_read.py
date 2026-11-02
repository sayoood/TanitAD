#!/usr/bin/env python3
"""threadrep_read.py -- SMOOTH-T1 thread-noise REPLICATE (supplementary; decides nothing).  The four sg72 rollouts re-run at 2 threads (smooth_t1_threadrep_chain.sh, out_t2/) are extracted and
read with the SAME machinery as smooth_t1_read.py (same functions, same rule), and set beside the card's rollouts (1 thread).  The difference between the two is float32 thread-order noise
and nothing else (same tool, checkpoints, arguments).  Writes <raw>/THREADREP.json.

    python threadrep_read.py --out-dir-t2 F:/TanitAD-scratch/smooth_t1/out_t2 --raw <package>/raw
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
spec = importlib.util.spec_from_file_location("smooth_t1_read", f"{HERE}/smooth_t1_read.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir-t2", required=True)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    GD = R.load_gd()
    arr2 = f"{a.raw}/arrays_t2"
    os.makedirs(arr2, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    for k in ("CTRL_s0", "CTRL_s1", "SOUP_s0", "SOUP_s1"):
        tag, arm, seed = R.RUNS[k]
        dst = f"{arr2}/{tag}.npz"
        if os.path.exists(dst):
            continue
        cmd = [R.PY, "-I", f"{HERE}/extract_smooth.py", "--arm", arm, "--seed", str(seed), "--out", dst, "--dump-dir", f"{a.out_dir_t2}/{tag}/dump_{tag}", "--ticks", f"{a.out_dir_t2}/{tag}/ticks/ticks_{tag}.npz"]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        open(f"{R.PK}/logs/threadrep_extract_{tag}.log", "w", encoding="utf-8", newline="\n").write(r.stdout + r.stderr)
        if r.returncode:
            raise SystemExit(f"extract {tag} failed rc={r.returncode}: {r.stderr[-400:]}")
    Y = {(arm, s): np.load(f"{R.GD_DIR}/raw/arrays/{arm}_s{s}.npz", allow_pickle=True) for arm in ("CTRL", "SOUP") for s in (0, 1)}
    labels = GD.LABELS
    main_json = json.load(open(f"{a.raw}/SMOOTH_T1.json", encoding="utf-8"))
    res = {"what": "thread-noise replicate: the same four sg72 rollouts at 2 threads (card rollouts: 1 thread; banked Y: 16); decides nothing", "tier": "T1-kin", "evidence_class": "MEASURED (ours)",
           "estimator": f"paired episode-cluster bootstrap, B={a.n_boot}, seed 0; answers another draw of EPISODES only", "arms": {}}
    for arm in ("CTRL", "SOUP"):
        per_seed, rows_out = {}, {}
        for s in (0, 1):
            zx, zy = np.load(f"{arr2}/{R.RUNS[f'{arm}_s{s}'][0]}.npz", allow_pickle=True), Y[(arm, s)]
            cls, masks = GD.classify(zy["v0"], zy["g"])
            tabs = {"XP": {}, "XY": {}}
            for i, lab in enumerate(labels):
                for nm, (u, v) in (("XP", (zx[f"a_{i}"], zx[f"b_{i}"])), ("XY", (zx[f"a_{i}"], zy[f"a_{i}"]))):
                    r = GD.decompose(u, v, zy["eid"], cls, GD.POOL[lab], a.n_boot)
                    tabs[nm][lab] = {"delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "separated": r["separated"]}
            per_seed[s] = {"A": R.read_a(GD, tabs["XP"]), "XY": R.read_xy(GD, tabs["XY"])}
            m = main_json["arms"][arm]["rows"][f"s{s}"]
            diff = {}
            for nm in ("XP", "XY"):
                flips = [l for l in labels if bool(tabs[nm][l]["separated"]) != bool(m[nm][l]["separated"]) or (np.sign(tabs[nm][l]["delta"]) != np.sign(m[nm][l]["delta"]) and tabs[nm][l]["separated"])]
                diff[nm] = {"max_abs_delta_diff": max(abs(tabs[nm][l]["delta"] - m[nm][l]["delta"]) for l in labels), "rows_whose_separated_flag_or_sign_differs": flips,
                            "max_abs_delta_diff_by_row": {l: abs(tabs[nm][l]["delta"] - m[nm][l]["delta"]) for l in labels}}
            rows_out[f"s{s}"] = {"XP": tabs["XP"], "XY": tabs["XY"], "vs_card_rollouts": diff}
        res["arms"][arm] = {"per_seed": {f"s{s}": per_seed[s] for s in (0, 1)}, "rows": rows_out, **R.arm_outcome(per_seed), "outcome_of_card_rollouts": main_json["arms"][arm]["outcome"]}
        print(arm, "replicate outcome", res["arms"][arm]["outcome"], "| card rollouts", main_json["arms"][arm]["outcome"])
    out = f"{a.raw}/THREADREP.json"
    json.dump(res, open(out, "w", encoding="utf-8", newline="\n"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("THREADREP OK ->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
