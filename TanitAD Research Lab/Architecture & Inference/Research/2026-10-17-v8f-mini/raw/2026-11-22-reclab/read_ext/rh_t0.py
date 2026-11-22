"""rh_t0.py -- guard G-T0 of the ROLLHIST card (section 3.2): the policy-rolled T0 ADE6 of RH minus B0, per seed pair; the upper bound of the paired episode-cluster interval must be <= +F_T0 = +0.0705 m.
Instrument = the DSCALE one (datascale/read3/ds_run.py `mech_argvs`, suffix "": `v8f_mini_eval.py run ... --project --families`; per-window `ade6`, `eid` in eval_<tag>.npz; the programme's ADE6 of
v8f_dscale_read.py / ds_read_b3.py `ade_means`).  Tier: T0 (world-model / tactical diagnostic; never driving performance).  Estimator: taniteval.ci.paired_episode_cluster_bootstrap, B 2,000, seed 0.
  python rh_t0.py run [ARM ...]      the E0 T0 eval of each checkpoint (CPU, 2 threads) -> out/t0/eval_<ARM>.npz      (ARM in RH_s0 RH_s1 B0_s0 B0_s1 RHp3_s0 RHp3_s1 RL_s0 RL_s1; default all that have a checkpoint)
  python rh_t0.py read [--seeds 0,1] [--t0 DIR] [--out JSON]   the guard (HOLDS / FAILS per seed pair, pooled beside); a missing eval is NOT COMPUTABLE, never a pass.
  python rh_t0.py argv ARM           print the rendered argv (dry check)"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
sys.path.insert(0, HERE)
import rh_common as C  # noqa: E402

T0DIR = C.OUT + "/t0"
RAWT = C.ART + "/dagt1_tree_b43c7a9a/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw"
SF = C.SF


def argv_t0(arm, out=None, ckpt=None, nwin=0):
    out = out or T0DIR
    ck = ckpt or f"{C.CK}/{arm}/ckpt_last.pt"
    a = [C.PY, C.TREE + "/stack/scripts/v8f_mini_eval.py", "run", "--ckpt", ck, "--tag", arm,
         "--eval-grid", C.INPUTS["--eval-grid"], "--eval-kin", RAWT + "/eval141_kin.npz", "--eval-join", C.INPUTS["--eval-join"],
         "--v9-eval", SF + "/inputs/v9/v9_labels_eval139.npz", "--lead-eval", SF + "/inputs/v9/lead_eval139.npz", "--speed-eval", SF + "/inputs/v9/refcv6_speed_max_v8_eval.jsonl",
         "--kin-mode", "yaw", "--p10", SF + "/inputs/p10/p10_prior.pt", "--eval-cache", SF + "/cache", "--tracks", SF + "/inputs/native_tracks.pt", "--out-dir", out, "--device", "cpu",
         "--project", "--families"]
    return a + (["--max-windows", str(nwin)] if nwin else [])


def run(arms):
    os.makedirs(T0DIR, exist_ok=True)
    rc = 0
    for arm in arms:
        if os.path.exists(f"{T0DIR}/eval_{arm}.npz"):
            print("skip (exists)", arm, flush=True)
            continue
        with open(f"{T0DIR}/log_{arm}.txt", "w", encoding="utf-8") as lf:
            r = subprocess.run(argv_t0(arm), env=C.env(), stdout=lf, stderr=subprocess.STDOUT, encoding="utf-8")
        ok = r.returncode == 0 and os.path.exists(f"{T0DIR}/eval_{arm}.npz")
        print(arm, "rc", r.returncode, "ok" if ok else "FAILED", flush=True)
        rc |= (not ok)
    return rc


def guard(seeds, t0dir, boot=None):
    sys.path.insert(0, C.TREE + "/taniteval")
    sys.path.insert(0, C.TREE + "/stack")
    from taniteval.ci import paired_episode_cluster_bootstrap as pb
    out = {"instrument": "T0 E0 eval (--project --families) `ade6` per window, the DSCALE instrument", "tier": "T0", "F_T0_m": C.F_T0, "evidence_F": "abs(UE_s1.A - UE_s0.A) = abs(2.0376 - 1.9671), FACTS E3, INHERITED; one seed pair, a weak floor",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap B 2,000 seed 0", "per_seed": {}, "holds_at": []}
    for s in seeds:
        fa, fb = f"{t0dir}/eval_{C.A_KIND}_s{s}.npz", f"{t0dir}/eval_B0_s{s}.npz"            # P03: the treated arm is RHp3 under RH_AKIND=RHp3 (default RH: unchanged)
        if not (os.path.exists(fa) and os.path.exists(fb)):
            out["per_seed"][str(s)] = {"status": "NOT COMPUTABLE", "why": "T0 eval missing"}
            continue
        A, B = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True)
        if not (len(A["ade6"]) == len(B["ade6"]) == 2399 and np.array_equal(A["eid"], B["eid"])):
            out["per_seed"][str(s)] = {"status": "NOT COMPUTABLE", "why": "windows differ"}
            continue
        r = pb(A["ade6"], B["ade6"], A["eid"], n_boot=2000, seed=0)
        hi = float(r["hi"])
        holds = bool(hi <= C.F_T0)
        out["per_seed"][str(s)] = {"status": "HOLDS" if holds else "FAILS", "point_m": float(r["delta"]), "lo": float(r["lo"]), "hi": hi, "mean_ade6_RH": float(np.mean(A["ade6"])),
                                   "mean_ade6_B0": float(np.mean(B["ade6"])), "n_windows": 2399, "n_episodes": int(len(np.unique(A["eid"])))}
    out["holds"] = None if any(v["status"] == "NOT COMPUTABLE" for v in out["per_seed"].values()) else all(v["status"] == "HOLDS" for v in out["per_seed"].values())
    out["fails_at_all_seeds"] = None if out["holds"] is None else all(v["status"] == "FAILS" for v in out["per_seed"].values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "read", "argv"))
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--t0", default=T0DIR)
    ap.add_argument("--out", default=C.OUT + "/read/GT0_ROLLHIST.json")
    a = ap.parse_args()
    if a.cmd == "argv":
        print(" ".join(argv_t0(a.arms[0])))
        return 0
    if a.cmd == "run":
        return run(a.arms or [f"{k}_s{s}" for k in C.KINDS for s in C.SEEDS])
    r = guard([int(x) for x in a.seeds.split(",")], a.t0)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(r, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: r[k] for k in ("holds", "fails_at_all_seeds")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
