#!/usr/bin/env python3
"""Overfitting-curve driver (NON-DESTRUCTIVE).

Injects the refa-dynin intermediate ckpts (5k/15k/20k) as TEMP registry arms
IN-MEMORY only (mutates registry.MODELS in this process — no file on the eval
pod is edited), then runs the standard core `run` + `pathspeed` panels on the
full physicalai val (40 eps / 881 windows). Writes results/<key>.json,
results/windows_<key>.pt, results/pathspeed_<key>.json exactly like the runner.
30k is already registered/evaluated; this fills 5k/15k/20k.

Usage:  python3 eval_overfit_curve.py [key1 key2 ...]
"""
import sys, time, traceback

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")

import taniteval.registry as reg
from taniteval import runner, pathspeed

BASE = dict(
    family="TanitAD", arch="refa-plus", config="flagship4b", d_dino=768,
    adapter="temporal", feat_kind="dinov2", encoder="frozen DINOv2-B/14",
    encoder_frozen=True, speed_input=True, dyn_input=True, action_dim=4,
    four_brain=True, hf=None, anti_collapse="frozen encoder",
)
TEMP = [
    dict(key="refa-dynin-5k",  name="REF-A dyn-in 4B (step 5000)",
         ckpt="/root/models/refa-dynin-5k/ckpt.pt", **BASE),
    dict(key="refa-dynin-15k", name="REF-A dyn-in 4B (step 15000)",
         ckpt="/root/models/refa-dynin-15k/ckpt.pt", **BASE),
    dict(key="refa-dynin-20k", name="REF-A dyn-in 4B (step 20000)",
         ckpt="/root/models/refa-dynin-20k/ckpt.pt", **BASE),
]
by_key = {e["key"]: e for e in TEMP}

want = sys.argv[1:] or list(by_key)
# inject only the requested temp arms (idempotent)
for k in want:
    if k in by_key and not any(m["key"] == k for m in reg.MODELS):
        reg.MODELS.append(by_key[k])

for key in want:
    print(f"\n########## {key} ##########", flush=True)
    t0 = time.time()
    try:
        runner.run_one(key, episodes=40)
    except Exception:
        print(f"[core] {key} FAILED"); traceback.print_exc()
    try:
        pathspeed.run_and_save(key, episodes=40)
    except Exception:
        print(f"[pathspeed] {key} FAILED"); traceback.print_exc()
    print(f"[done] {key} in {round(time.time()-t0,1)}s", flush=True)

print("\nALL OVERFIT-CURVE EVALS COMPLETE", flush=True)
