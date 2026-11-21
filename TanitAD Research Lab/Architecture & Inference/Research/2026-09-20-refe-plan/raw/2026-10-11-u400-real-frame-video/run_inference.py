"""Re-run the u400 planner on ONLY the video's tokens (per log, resumable) and gate it against the BANKED full-navtest plans.

For each log in video_runs.json: refe_navtest_seam.py (UNCHANGED, same flags as eval/stage2_sft.py: --rule navsim_v1_lane, default goal fix
pdm_route, repair_last_heading on) with --dump-proposals -> all 64 proposals + the scorer's raw logits from the SAME forward pass.
FIDELITY GATE (per log, then overall): proposals[pick] must reproduce the BANKED executed plan
`raw/2026-10-05-sft-stage2/sft4_B_u400_navtest_full/sft4_B_u400_navtest_full_seam.npz` (`poses`) to <= TOL_M metres, and
argmax(aggregate(logits)) must equal the recorded pick.   Stops at the first failing log (GATE_FAIL.json).

    C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe run_inference.py            (long: background it)
Writes inference/<log>.dump.npz, inference/<log>.gate.json, inference/FIDELITY_GATE.json and u400_video_proposals.npz.
"""
import json
import os
import socket
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PKG, "eval"))
import eval_checkpoint as EC  # noqa: E402

DRV = "E:"
CKPT = f"{DRV}/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt"
EXPECT_MD5 = "8e382eeb"
BANKED = os.path.join(PKG, "raw", "2026-10-05-sft-stage2", "sft4_B_u400_navtest_full", "sft4_B_u400_navtest_full_seam.npz")
DATA = f"{DRV}/Projects/TanitAD/data/refe_navtest"
EXPORT = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
DBDIR = f"{DRV}/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
TOL_M = 1e-3
V1_W = (5.0, 5.0, 2.0)
LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
MARK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"
PY_D = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"


def agg(logits):
    """planner.aggregate under rule navsim_v1_lane (numpy mirror): NC*DAC*(5EP+5TTC+2C)/12 * p_lane"""
    p = 1 / (1 + np.exp(-logits.astype(np.float64)))
    return p[..., 0] * p[..., 1] * (p[..., 2] * V1_W[0] + p[..., 3] * V1_W[1] + p[..., 4] * V1_W[2]) / sum(V1_W) * p[..., 6]


def take_lock(me):
    while True:
        if not os.path.exists(MARK):
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, me.encode())
                os.close(fd)
                return
            except FileExistsError:
                pass
        print(f"waiting for the GPU lock {time.strftime('%T')}", flush=True)
        time.sleep(60)


def release(me):
    try:
        if open(LOCK, encoding="utf-8").read() == me:
            os.remove(LOCK)
    except OSError:
        pass


def gate(dump_path, banked):
    D = np.load(dump_path)
    tok = [str(t) for t in D["token"]]
    P, L, pick = D["proposals"].astype(np.float64), D["logits"], D["pick"]
    bi = {str(t): i for i, t in enumerate(banked["token"])}
    B = banked["poses"].astype(np.float64)
    ref = np.stack([B[bi[t]] for t in tok])
    got = P[np.arange(len(tok)), pick]
    d = np.abs(got - ref)
    dxy = np.linalg.norm(got[..., :2] - ref[..., :2], axis=-1).max(1)          # per token, max over the 8 poses (metres)
    dh = np.abs((got[..., 2] - ref[..., 2] + np.pi) % (2 * np.pi) - np.pi).max(1)
    pick_re = np.argmax(agg(L), axis=1)
    return {"n": len(tok), "max_abs_delta_any_component": float(d.max()), "max_xy_delta_m": float(dxy.max()),
            "max_heading_delta_rad": float(dh.max()),
            "tokens_identical": int((d.reshape(len(tok), -1).max(1) == 0).sum()), "tokens_gt_tol": int((dxy > TOL_M).sum()),
            "worst_tokens": [tok[i] for i in np.argsort(-dxy)[:3]], "pick_matches_argmax_of_logits": int((pick_re == pick).sum()),
            "tol_m": TOL_M, "pass": bool((dxy <= TOL_M).all() and (pick_re == pick).all())}


def main():
    runs = json.load(open(os.path.join(HERE, "video_runs.json")))
    logs = []
    for r in runs:
        if r["log"] not in logs:
            logs.append(r["log"])
    outd = os.path.join(HERE, "inference")
    os.makedirs(outd, exist_ok=True)
    banked = np.load(BANKED)
    env = {k: (v.replace("D:/", "E:/") if isinstance(v, str) else v) for k, v in EC.env_driverl().items()}
    results = {}
    for log in logs:
        dump = os.path.join(outd, f"{log}.dump.npz")
        gp = os.path.join(outd, f"{log}.gate.json")
        if not os.path.exists(dump):
            tf = os.path.join(HERE, "tokens_by_log", f"{log}.json")
            me = json.dumps({"job": f"u400-video-{log}", "pid": os.getpid(), "host": socket.gethostname(),
                             "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            take_lock(me)
            try:
                t0 = time.time()
                cmd = [PY_D, "refe_navtest_seam.py", "--ckpt", CKPT, "--frames", f"{DATA}/frames", "--db-dir", DBDIR, "--export", EXPORT,
                       "--tokens", tf, "--out", os.path.join(outd, f"{log}.seam.npz"), "--arm", "REFe_sft4_B_u400_video",
                       "--rule", "navsim_v1_lane", "--dump-proposals", dump]
                rc = subprocess.call(cmd, cwd=os.path.join(PKG, "eval"), env=env,
                                     stdout=open(os.path.join(outd, f"{log}.seam.log"), "w"), stderr=subprocess.STDOUT)
                print(f"{log}: seam rc={rc} {time.time() - t0:.0f} s", flush=True)
            finally:
                release(me)
            if not os.path.exists(dump):
                json.dump({"log": log, "reason": "no dump written (see .seam.log)"}, open(os.path.join(HERE, "GATE_FAIL.json"), "w"), indent=1)
                return 1
        g = gate(dump, banked)
        g["log"] = log
        json.dump(g, open(gp, "w"), indent=1)
        results[log] = g
        print(f"GATE {log}: n={g['n']} max_xy={g['max_xy_delta_m']:.3e} m max_head={g['max_heading_delta_rad']:.3e} rad  >tol={g['tokens_gt_tol']} "
              f"pick==argmax {g['pick_matches_argmax_of_logits']}/{g['n']}  {'PASS' if g['pass'] else 'FAIL'}", flush=True)
        if not g["pass"]:
            json.dump(g, open(os.path.join(HERE, "GATE_FAIL.json"), "w"), indent=1)
            print("STOP: fidelity gate failed", flush=True)
            return 2
    toks, props, logits, picks = [], [], [], []
    for log in logs:
        D = np.load(os.path.join(outd, f"{log}.dump.npz"))
        toks += [str(t) for t in D["token"]]
        props.append(D["proposals"])
        logits.append(D["logits"])
        picks.append(D["pick"])
    np.savez_compressed(os.path.join(HERE, "u400_video_proposals.npz"), token=np.array(toks), proposals=np.concatenate(props),
                        logits=np.concatenate(logits), pick=np.concatenate(picks), rule=np.array("navsim_v1_lane"), ckpt=np.array(CKPT))
    allg = {"overall": {"n": sum(g["n"] for g in results.values()),
                        "max_xy_delta_m": max(g["max_xy_delta_m"] for g in results.values()),
                        "max_heading_delta_rad": max(g["max_heading_delta_rad"] for g in results.values()),
                        "max_abs_delta_any_component": max(g["max_abs_delta_any_component"] for g in results.values()),
                        "tokens_identical": sum(g["tokens_identical"] for g in results.values()),
                        "tokens_gt_tol": sum(g["tokens_gt_tol"] for g in results.values()), "tol_m": TOL_M,
                        "pass": all(g["pass"] for g in results.values())},
            "banked": BANKED, "ckpt": CKPT, "per_log": results}
    json.dump(allg, open(os.path.join(HERE, "inference", "FIDELITY_GATE.json"), "w"), indent=1)
    print("ZZINFER_DONE", json.dumps(allg["overall"]), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
