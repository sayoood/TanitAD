"""SFT-4 stage-2 PLUMBING IDENTITY (G-EVAL for the 7-output scorer, before any SFT-4 checkpoint exists).

The deployed model_final.pt is re-saved as a 7-output checkpoint whose 7th row is constant (weight 0, bias 1.5) and whose
rows 0-5 are copied exactly, with meta n_score_components = 7. Through the UNCHANGED navtest seam:
  arm base : model_final.pt,              --rule navsim_v1        (the deployed system, route fix ON, A7 ON)
  arm lane7: model_final_lane7_identity,  --rule navsim_v1_lane   (the path SFT-4's winner will take)
p(lane) is the same constant for every proposal, so navsim_v1_lane must pick exactly what navsim_v1 picks. The planner runs
in fp32, where the SFT-4 smoke measured the 7-row vs 6-row head residual at exactly 0.0, so the bars are EXACT:
  - logits 0-5 bit-identical, logit 6 == 1.5 on every proposal;
  - the same pick on every token; seam poses bit-identical.
Anything else means stage 2 would score a plumbing artefact. Runs under the shared dev-box GPU lock.
    python eval/stage2_lane_identity.py [--n 200]
Prints ZZLANE_IDENTITY PASS|FAIL and writes raw/2026-10-04-sft4-stage2-identity/result.json.
"""
import argparse
import json
import os
import random
import socket
import subprocess
import sys
import time

import numpy as np

PKG = "E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan"
sys.path.insert(0, f"{PKG}/eval")
sys.path.insert(0, f"{PKG}/refe")
import eval_checkpoint as EC  # noqa: E402

LOCK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.lock"
MARK = "C:/Users/Admin/qland/work/refcv7/devbox_gpu.chain_active"
D = "E:/Projects/TanitAD/data/refe_navtest"
BASE = "E:/Projects/TanitAD/data/refe_runs_eval/model_final.pt"
LANE7 = "E:/Projects/TanitAD/data/refe_runs_eval/model_final_lane7_identity.pt"
OUT = f"{PKG}/raw/2026-10-04-sft4-stage2-identity"
SEAMS = f"{D}/seams/sft4_identity"


def build_lane7():
    import torch
    import ckpt_io
    sd = torch.load(BASE, map_location="cpu", weights_only=False)
    st = sd["model"]
    w, b = st["score_head.weight"], st["score_head.bias"]
    assert w.shape[0] == 6 and b.shape[0] == 6, (w.shape, b.shape)
    w7 = torch.zeros((7, w.shape[1]), dtype=w.dtype); w7[:6] = w
    b7 = torch.zeros((7,), dtype=b.dtype); b7[:6] = b; b7[6] = 1.5
    st["score_head.weight"], st["score_head.bias"] = w7, b7
    meta = dict(sd.get("meta", {}) or {})
    meta["n_score_components"] = 7
    meta["stage2_lane_identity"] = {"base": BASE, "row6": "weight 0, bias 1.5", "rows0to5": "copied exactly"}
    sd["meta"] = meta
    ckpt_io.atomic_save(sd, LANE7)
    from model import REFeConfig
    n = ckpt_io.config_for_checkpoint(REFeConfig.for_backbone("vitl16"), LANE7).n_score_components
    assert n == 7, n
    return {"lane7_ckpt": LANE7, "config_for_checkpoint_n": n}


def fix(v):
    return v.replace("D:/", "E:/") if isinstance(v, str) else v


def take_lock(me):
    while True:
        if not os.path.exists(MARK):
            try:
                fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, me.encode()); os.close(fd)
                return
            except FileExistsError:
                pass
        try:
            held = open(LOCK, encoding="utf-8").read()[:100]
        except OSError:
            held = "(marker)"
        print(f"waiting for the GPU lock {time.strftime('%T')}: {held}", flush=True)
        time.sleep(60)


def release(me):
    try:
        if open(LOCK, encoding="utf-8").read() == me:
            os.remove(LOCK); print(f"GPU lock released {time.strftime('%T')}", flush=True)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True); os.makedirs(SEAMS, exist_ok=True)
    res = {"build": build_lane7() if not os.path.exists(LANE7) else {"lane7_ckpt": LANE7, "reused": True}}
    conf = json.load(open(f"{D}/a5_confirm_tokens.json", encoding="utf-8"))
    toks = sorted(conf["tokens"])
    random.Random(20261004).shuffle(toks)
    sub = sorted(toks[: a.n])
    tj = f"{OUT}/tokens_identity{a.n}.json"
    json.dump({"tokens": sub, "source": "a5_confirm_tokens.json", "seed": 20261004}, open(tj, "w"))
    me = json.dumps({"job": "refe-sft4-stage2-lane-identity", "pid": os.getpid(), "host": socket.gethostname(),
                     "acquired": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    take_lock(me)
    print(f"lock taken {time.strftime('%F %T')}", flush=True)
    env = {k: fix(v) for k, v in EC.env_driverl().items()}
    arms = {"base": (BASE, "navsim_v1"), "lane7": (LANE7, "navsim_v1_lane")}
    try:
        for arm, (ck, rule) in arms.items():
            cmd = [EC.DRIVERL_PY, "refe_navtest_seam.py", "--ckpt", ck, "--frames", f"{D}/frames",
                   "--db-dir", "E:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test",
                   "--export", "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz",
                   "--tokens", tj, "--out", f"{SEAMS}/seam_{arm}.npz", "--arm", f"REFe_identity_{arm}",
                   "--rule", rule, "--dump-proposals", f"{SEAMS}/props_{arm}.npz"]
            t0 = time.time()
            with open(f"{OUT}/seam_{arm}.log", "w", encoding="utf-8") as f:
                rc = subprocess.call(cmd, cwd=f"{PKG}/eval", env=env, stdout=f, stderr=subprocess.STDOUT)
            res[f"seam_{arm}"] = {"rc": rc, "seconds": round(time.time() - t0, 1)}
            print(f"arm {arm}: rc {rc} {time.time() - t0:.0f} s", flush=True)
            if rc:
                break
    finally:
        release(me)
    if any(res.get(f"seam_{k}", {}).get("rc", 1) for k in arms):
        res["verdict"] = "FAIL (a seam did not complete)"
    else:
        sb, sl = np.load(f"{SEAMS}/seam_base.npz"), np.load(f"{SEAMS}/seam_lane7.npz")
        pb, pl = np.load(f"{SEAMS}/props_base.npz"), np.load(f"{SEAMS}/props_lane7.npz")
        same_tok = bool((sb["token"] == sl["token"]).all()) and bool((pb["token"] == pl["token"]).all())
        lb, ll = pb["logits"], pl["logits"]
        res.update({
            "rows": int(len(sb["token"])), "tokens_aligned": same_tok,
            "logit_shapes": [list(lb.shape), list(ll.shape)],
            "max_abs_logits0to5": float(np.abs(ll[..., :6] - lb).max()),
            "logit6_all_1p5": bool((ll[..., 6] == 1.5).all()),
            "picks_identical": int((pb["pick"] == pl["pick"]).sum()),
            "poses_identical_rows": int((sb["poses"] == sl["poses"]).all(axis=(1, 2)).sum()),
            "rules": [str(pb["rule"]), str(pl["rule"])],
            "reports": {k: json.load(open(f"{SEAMS}/seam_{k}.report.json"))["frame_control"] for k in arms}})
        n = res["rows"]
        ok = (same_tok and res["max_abs_logits0to5"] == 0.0 and res["logit6_all_1p5"] and res["picks_identical"] == n
              and res["poses_identical_rows"] == n and res["rules"] == ["navsim_v1", "navsim_v1_lane"] and n > 0)
        res["verdict"] = "PASS" if ok else "FAIL"
    json.dump(res, open(f"{OUT}/result.json", "w"), indent=1)
    print("ZZLANE_IDENTITY", res["verdict"], json.dumps({k: v for k, v in res.items() if k not in ("build", "reports")}))
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
