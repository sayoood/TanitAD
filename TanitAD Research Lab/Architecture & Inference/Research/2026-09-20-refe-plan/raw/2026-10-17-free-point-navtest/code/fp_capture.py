#!/usr/bin/env python3
"""STAGE A of the free-point navtest read (PLAN.md): capture the planner's MODEL INPUTS for every navtest token, on the dev box CPU, with NO model weights.

It runs `refe_navtest_seam.py`'s OWN loop (scenario build -> `REFePlanner.initialize` -> `planner.infer`), unchanged, with exactly two things stubbed:
  * `planner.REFe` -> a stub module that returns zeros (so no weights are loaded and no GPU is used), and
  * `REFePlanner._image_for` -> resolves the SAME frame members the real one would (`self.frames.resolve`), checks the camera count, returns a dummy tensor.
Everything that builds the inputs is the real code: `_ego_vec`, `_goal_for` (goal fix `pdm_route`, the adopted config), `_calib_for` (the per-log rig). The seam's
`--record-inputs` writes, per token, `ego` (7), `goal`, `frames` (the 4 zip members) -- the format Amendment 8's gate (e) already defined; this tool adds the per-log rig.

    python fp_capture.py plan   --out-dir D                         -> D/plan.json (logs, tokens per log, shards)
    python fp_capture.py shard  --out-dir D --shard I/N             -> loops its logs, one CHILD process per log (a native crash costs one log)
    python fp_capture.py one    --out-dir D --log LOG               -> D/cap/inputs_<log>.json + D/cap/calib_<log>.npy
    python fp_capture.py merge  --out-dir D                         -> D/inputs_full.json, D/calib.npz, D/capture_report.json (refuses unless all 12,146 tokens are present)
    python fp_capture.py check-a9 --out-dir D                       -> IC1: bit-equality with the A9 recorded inputs (3,410 tokens)
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import time

PK = "C:/fp/PK"                                   # the landed package, extracted from 19785c8 (short path: MAX_PATH)
SEAM_DIR = f"{PK}/eval"
REFE_DIR = f"{PK}/refe"
DATA = "E:/Projects/TanitAD/data/refe_navtest"
FRAMES = f"{DATA}/frames"
DB_DIR = "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
EXPORT = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
TOKENS_ALL = f"{DATA}/navtest_full_tokens.json"
CKPT = "E:/Projects/TanitAD/data/refe_runs_eval/sft4_peek/model_sft4_B_u400.pt"
CKPT_MD5_PREFIX = "8e382eeb"
A9_DIR = ("E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-01-goal-trigger/a9")
PY = "C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe"


def env_for_children():
    sys.path.insert(0, SEAM_DIR)
    import eval_checkpoint as EC
    e = {k: (v.replace("D:/", "E:/") if isinstance(v, str) else v) for k, v in EC.env_driverl().items()}
    e["OMP_NUM_THREADS"] = "2"
    return e


def load_tokens():
    doc = json.load(open(TOKENS_ALL, encoding="utf-8"))
    toks = doc["tokens"] if isinstance(doc, dict) else doc
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    by_log: dict = {}
    for t in toks:
        by_log.setdefault(exp[t]["log_name"], []).append(t)
    return toks, by_log


def cmd_plan(a):
    toks, by_log = load_tokens()
    logs = sorted(by_log)
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump({"n_tokens": len(toks), "n_logs": len(logs), "tokens_per_log": {l: len(by_log[l]) for l in logs}},
              open(f"{a.out_dir}/plan.json", "w"), indent=0)
    print(f"plan: {len(toks)} tokens over {len(logs)} logs -> {a.out_dir}/plan.json")
    return 0


# ----------------------------------------------------------------------------------------------------------- one log (child)
def cmd_one(a):
    import numpy as np
    sys.path.insert(0, REFE_DIR)
    sys.path.insert(0, SEAM_DIR)
    import torch
    import ckpt_io
    import planner as PLM

    toks, by_log = load_tokens()
    lt = by_log[a.log]
    cap = f"{a.out_dir}/cap"
    os.makedirs(cap, exist_ok=True)
    tj = f"{cap}/tokens_{a.log}.json"
    json.dump({"tokens": lt, "token_log": {t: a.log for t in lt}}, open(tj, "w"))

    # read the checkpoint's own meta (mmap: cheap) so the stub reproduces what the real load would have told the planner
    sd = torch.load(CKPT, map_location="cpu", weights_only=False, mmap=True)
    meta = dict(sd.get("meta") or {})
    del sd
    n_comp = int(meta.get("n_score_components", 6))
    psc = bool(meta.get("per_sample_calib", False))
    nav_mode = meta.get("nav_mode", "goal")
    if not (n_comp == 7 and psc and nav_mode == "goal"):
        print(f"REFUSED: the checkpoint meta is n_score_components={n_comp} per_sample_calib={psc} nav_mode={nav_mode}; this capture is for a 7-output goal-mode per-sample-rig model")
        return 4

    class StubREFe(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.M = int(getattr(cfg, "n_proposals", 64))

        def forward(self, img, ego, goal, calib=None, **kw):
            return torch.zeros(1, self.M, 20, 3), torch.zeros(1, self.M, self.cfg.n_score_components)

    def fake_load(model, path, map_location="cpu", backbone=None, meta_out=None):
        if meta_out is not None:
            meta_out.update(meta)
        return "FULL"
    PLM.REFe = StubREFe
    ckpt_io.load_for_inference = fake_load

    def image_stub(self, ego):
        paths = self.frames.resolve(self._log_hint, int(ego.time_point.time_us)) if self._log_hint else None
        if not paths:
            return None
        if len(paths) != self.cfg.n_cameras:
            raise RuntimeError(f"resolved {len(paths)} frames, the model wants {self.cfg.n_cameras}")
        return torch.zeros(1, len(paths), 3, 8, 8)
    PLM.REFePlanner._image_for = image_stub

    cal: dict = {}
    orig_cal = PLM.REFePlanner._calib_for

    def calib_rec(self, log_name):
        out = orig_cal(self, log_name)
        cal[log_name] = out.detach().cpu().numpy().astype(np.float64)
        return out
    PLM.REFePlanner._calib_for = calib_rec

    import refe_navtest_seam as SEAM
    out_in = f"{cap}/inputs_{a.log}.json"
    sys.argv = ["refe_navtest_seam.py", "--ckpt", CKPT, "--frames", FRAMES, "--db-dir", DB_DIR, "--export", EXPORT, "--tokens", tj,
                "--out", f"{cap}/stubseam_{a.log}.npz", "--arm", "FP_CAPTURE_STUB", "--rule", "navsim_v1_lane", "--device", "cpu",
                "--record-inputs", out_in + ".tmp"]
    rc = SEAM.main()        # rc 1 (frame-control gate on the 36 declared C1 tokens) is expected and irrelevant here; the inputs are what we keep
    if not os.path.exists(out_in + ".tmp"):
        print(f"FAILED {a.log}: no inputs written (seam rc {rc})")
        return 2
    rec = json.load(open(out_in + ".tmp", encoding="utf-8"))
    missing = [t for t in lt if t not in rec]
    if missing:
        print(f"FAILED {a.log}: {len(missing)} of {len(lt)} tokens missing from the recorded inputs")
        return 3
    if set(cal) != {a.log}:
        print(f"FAILED {a.log}: rig recorded for {sorted(cal)}")
        return 3
    np.save(f"{cap}/calib_{a.log}.npy", cal[a.log])
    os.replace(out_in + ".tmp", out_in)                       # the inputs file is the completion marker
    for f in (f"{cap}/stubseam_{a.log}.npz", f"{cap}/stubseam_{a.log}.report.json", tj):
        try:
            os.remove(f)
        except OSError:
            pass
    print(f"ZZCAPTURE_LOG_OK {a.log} {len(lt)}")
    return 0


# ----------------------------------------------------------------------------------------------------------- shard driver
def cmd_shard(a):
    toks, by_log = load_tokens()
    logs = sorted(by_log)
    i, n = (int(x) for x in a.shard.split("/"))
    mine = [l for k, l in enumerate(logs) if k % n == i]
    env = env_for_children()
    cap = f"{a.out_dir}/cap"
    os.makedirs(cap, exist_ok=True)
    done = failed = 0
    t0 = time.time()
    for l in mine:
        if os.path.exists(f"{cap}/inputs_{l}.json") and os.path.exists(f"{cap}/calib_{l}.npy"):
            done += 1
            continue
        ok = False
        for attempt in (1, 2):
            with open(f"{cap}/child_{l}.log", "w", encoding="utf-8") as f:
                rc = subprocess.call([PY, "-I", os.path.abspath(__file__), "one", "--out-dir", a.out_dir, "--log", l], env=env,
                                     stdout=f, stderr=subprocess.STDOUT)
            if rc == 0 and os.path.exists(f"{cap}/inputs_{l}.json"):
                ok = True
                break
        done += int(ok)
        failed += int(not ok)
        print(f"shard {a.shard}: {l} {'OK' if ok else 'FAILED'}  done {done} failed {failed} of {len(mine)}  {time.time() - t0:.0f}s", flush=True)
    print(f"ZZCAPTURE_SHARD_{'OK' if failed == 0 else 'INCOMPLETE'} {a.shard} done {done} failed {failed}")
    return 0 if failed == 0 else 1


# ----------------------------------------------------------------------------------------------------------- merge / checks
def cmd_merge(a):
    import numpy as np
    toks, by_log = load_tokens()
    cap = f"{a.out_dir}/cap"
    inputs, calib, miss = {}, {}, []
    for l in sorted(by_log):
        p = f"{cap}/inputs_{l}.json"
        if not (os.path.exists(p) and os.path.exists(f"{cap}/calib_{l}.npy")):
            miss.append(l)
            continue
        inputs.update(json.load(open(p, encoding="utf-8")))
        calib[l] = np.load(f"{cap}/calib_{l}.npy")
    if miss or set(inputs) != set(toks):
        print(f"REFUSED: {len(miss)} logs missing {miss[:5]}; tokens {len(inputs)} of {len(toks)}")
        return 2
    # the frame members are recorded with the dev-box drive prefix; the pod needs only '<zip basename>::<member>'
    for t, r in inputs.items():
        r["frames"] = [fr.replace("\\", "/").rsplit("/", 1)[-1] for fr in r["frames"]]
        if len(r["frames"]) != 4 or any("::" not in fr for fr in r["frames"]):
            print(f"REFUSED: token {t} frames {r['frames']}")
            return 2
    tl ={t: l for l, ts in by_log.items() for t in ts}
    for t, r in inputs.items():
        r["log"] = tl[t]
    json.dump({t: {k: r[k] for k in ("ego", "goal", "frames", "log")} for t, r in inputs.items()},
              open(f"{a.out_dir}/inputs_full.json", "w", encoding="utf-8"))
    np.savez(f"{a.out_dir}/calib.npz", **{l: calib[l] for l in sorted(calib)})
    gd = {t: r.get("goal_diag") for t, r in inputs.items()}
    rep = {"tokens": len(inputs), "logs": len(calib), "goal_fix_counts": {k: sum(1 for g in gd.values() if (g or {}).get("route_fix") == k) for k in {(g or {}).get("route_fix") for g in gd.values()}},
           "goal_changed": sum(1 for g in gd.values() if (g or {}).get("changed")), "goal_len_set": sorted({len(r["goal"]) for r in inputs.values()}),
           "ego_len_set": sorted({len(r["ego"]) for r in inputs.values()}), "calib_shapes": sorted({str(v.shape) for v in calib.values()}),
           "nonfinite": sum(1 for r in inputs.values() if not all(math.isfinite(x) for x in r["ego"] + r["goal"]))}
    json.dump(rep, open(f"{a.out_dir}/capture_report.json", "w"), indent=1)
    print(json.dumps(rep))
    print("ZZCAPTURE_MERGE_OK")
    return 0 if rep["nonfinite"] == 0 else 2


def cmd_check_a9(a):
    """IC1: every token that has an A9 recorded-inputs file must match the capture BIT FOR BIT (ego, goal) and in its frame members."""
    cur = json.load(open(f"{a.out_dir}/inputs_full.json", encoding="utf-8"))
    n = bad = 0
    ex = []
    seen = {}
    for fn in sorted(os.listdir(A9_DIR)):
        if not (fn.startswith("inputs_") and fn.endswith(".json")) or fn in ("inputs_pdm_route.json",):
            pass
        if not (fn.startswith("inputs_") and fn.endswith(".json")):
            continue
        if "pdm_route" not in fn and "adopted_rest" not in fn:
            continue                                              # only the arms that ran the ADOPTED goal fix (pdm_route / the unchanged rest)
        rec = json.load(open(f"{A9_DIR}/{fn}", encoding="utf-8"))
        for t, r in rec.items():
            if t not in cur or t in seen:
                continue
            seen[t] = fn
            n += 1
            fr = [x.replace("\\", "/").rsplit("/", 1)[-1] for x in r["frames"]]
            if r["ego"] != cur[t]["ego"] or r["goal"] != cur[t]["goal"] or fr != cur[t]["frames"]:
                bad += 1
                if len(ex) < 5:
                    ex.append({"token": t, "file": fn, "ego_equal": r["ego"] == cur[t]["ego"], "goal_equal": r["goal"] == cur[t]["goal"], "frames_equal": fr == cur[t]["frames"]})
    res = {"IC1_tokens_compared": n, "IC1_mismatches": bad, "examples": ex, "pass": bool(n >= 3000 and bad == 0)}
    json.dump(res, open(f"{a.out_dir}/IC1.json", "w"), indent=1)
    print(json.dumps(res))
    print("ZZIC1_" + ("PASS" if res["pass"] else "FAIL"))
    return 0 if res["pass"] else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("plan", "shard", "one", "merge", "check-a9"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    return {"plan": cmd_plan, "shard": cmd_shard, "one": cmd_one, "merge": cmd_merge, "check-a9": cmd_check_a9}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
