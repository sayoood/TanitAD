#!/usr/bin/env python3
"""Dev-box smoke of STAGE C on a handful of C3 tokens (any device): runs `fp_forward.py run` exactly as the pod would, then checks it against what is already banked.

    python fp_smoke.py --work W --log LOG --k 3 [--device cpu|cuda] [--swap-cams]

Checks (small-scale IC2 / IC3 / the C3 logits): IC2 ic2_maxabs == 0; BASE logits vs the C3 table's logits (u400 on the dev-box RTX 4060; a CPU run differs by float noise, reported);
BASE pick -> executed pose vs the BANKED u400 seam pose (<= 5e-3 m); the C0A / R0B / M25 picks and how many differ from BASE.  With --swap-cams the first 8 tokens get camera slots 1 and 2
exchanged (D3, a DELIBERATE REGRESSION): the pose agreement with the banked seam must then FAIL.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fp_picks as P  # noqa: E402

FRAMES = "E:/Projects/TanitAD/data/refe_navtest/frames"
CKPT = "E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt"
STATES = ["C0A=C:/Users/Admin/rls1_adam_dl/rls1-c0/scorer_states/scorer_states_u1000.pt:A", "R0B=C:/Users/Admin/rls1_adam_dl/rls1-r0/scorer_states/scorer_states_u1000.pt:B"]
C3 = f"{P.PK}/raw/2026-10-11-c3-selection-headroom/table.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--swap-cams", action="store_true")
    a = ap.parse_args()
    sm = f"{a.work}/smoke{'_swap' if a.swap_cams else ''}"
    os.makedirs(sm, exist_ok=True)
    exp = json.load(gzip.open(P.EXPORT, "rt", encoding="utf-8"))["tokens"]
    c3 = np.load(C3, allow_pickle=False)
    c3t = [str(t) for t in c3["token"]]
    ci = {t: i for i, t in enumerate(c3t)}
    cap = json.load(open(f"{a.work}/cap/inputs_{a.log}.json", encoding="utf-8"))
    pick = [t for t in c3t if t in cap][: a.k]
    inputs = {}
    for t in pick:
        r = cap[t]
        inputs[t] = {"ego": r["ego"], "goal": r["goal"], "frames": [x.replace("\\", "/").rsplit("/", 1)[-1] for x in r["frames"]], "log": a.log}
    json.dump(inputs, open(f"{sm}/inputs.json", "w"))
    np.savez(f"{sm}/calib.npz", **{a.log: np.load(f"{a.work}/cap/calib_{a.log}.npy")})
    env = dict(os.environ, OMP_NUM_THREADS="4", REFE_BACKBONE_ROOT="E:/Projects/TanitAD/data/backbones", PYTHONPATH=f"{P.PK}/refe")
    cmd = ["C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe", "-I", f"{HERE}/fp_forward.py", "run", "--inputs", f"{sm}/inputs.json", "--calib", f"{sm}/calib.npz",
           "--ckpt", CKPT, "--frames-dir", FRAMES, "--device", a.device, "--out", f"{sm}/out.npz", "--repeat-first", "2"]
    for s in STATES:
        cmd += ["--state", s]
    if a.swap_cams:
        cmd += ["--mutate-swap-cams"]
    # `-I` drops PYTHONPATH: the model code is put on sys.path by a tiny shim
    shim = f"{sm}/shim.py"
    open(shim, "w").write(f"import sys, runpy\nsys.path.insert(0, r'{P.PK}/refe')\nsys.argv = [r'{HERE}/fp_forward.py'] + {cmd[3:]!r}\nrunpy.run_path(r'{HERE}/fp_forward.py', run_name='__main__')\n")
    rc = subprocess.call([cmd[0], "-I", shim], env=env)
    print("fp_forward rc", rc)
    z = np.load(f"{sm}/out.npz", allow_pickle=False)
    toks = [str(t) for t in z["token"]]
    meta = json.loads(str(z["meta_json"]))
    AT = P.load_attrib()
    pl = P.points_logits({k[7:]: z[k] for k in z.files if k.startswith("logits_")})
    pk = P.pick_all(AT, pl)
    zb = np.load(P.BANKED_U400_SEAM, allow_pickle=False)
    bi = {str(t): i for i, t in enumerate(zb["token"])}
    res = {"tokens": toks, "ic2_maxabs": float(z["ic2_maxabs"].max()), "ic5_forward": meta.get("ic5_forward"), "device": meta["device"], "swap_cams": a.swap_cams}
    res["BASE_logits_vs_C3_max_abs"] = float(max(np.abs(z["logits_BASE"][i] - c3["logits"][ci[t]]).max() for i, t in enumerate(toks)))
    poses = {p: P.executed_poses(z["traj"], pk[p]) for p in P.POINTS}
    d_base = [float(np.abs(poses["BASE"][i] - zb["poses"][bi[t]]).max()) for i, t in enumerate(toks)]
    res["BASE_pose_vs_banked_u400_max_dpose_m"] = d_base
    res["frac_equal_with_swap" if a.swap_cams else "frac_equal"] = float(np.mean([d <= P.IC3_POSE_TOL_M for d in d_base]))
    res["BASE_pick_vs_C3_pick"] = [[int(pk["BASE"][i]), int(c3["pick"][ci[t]])] for i, t in enumerate(toks)]
    res["pick_differs_from_BASE"] = {p: int((pk[p] != pk["BASE"]).sum()) for p in P.POINTS if p != "BASE"}
    json.dump(res, open(f"{sm}/smoke_result.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
