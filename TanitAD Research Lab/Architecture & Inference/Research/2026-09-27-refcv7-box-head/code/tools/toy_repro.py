"""toy_repro.py <tree> <out.json> [--seeds 0,1] [--threads N] [--log-every 100] [--steps 1200] [--arms main,memory_zeros]
[--const-lr] -- an INSTRUMENT (no assertion): runs the TOY of ``test_TOY_the_loop_memorises_and_the_memory_zeros_arm_cannot``
exactly as the tree's own test does (the tree's ``_Toy`` and ``G.run_arm``, lr 1e-3, the given seed), and records every
logged row plus the platform (torch, BLAS, threads). ``--const-lr`` forces ``lr_factor = 1`` (the tip's loop) on a tree
that carries A17. The decoder has no dropout and the eval rule draws no random numbers, so ``--log-every`` changes only
how often the rows are read, never the training trajectory."""
import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import torch

ap = argparse.ArgumentParser()
ap.add_argument("tree")
ap.add_argument("out")
ap.add_argument("--seeds", default="0")
ap.add_argument("--threads", type=int, default=0)
ap.add_argument("--log-every", type=int, default=100)
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--arms", default="main,memory_zeros")
ap.add_argument("--const-lr", action="store_true")
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
T = Path(a.tree).resolve()
spec = importlib.util.spec_from_file_location("tgbo_repro", str(T / "stack" / "tests" / "test_g_box_overfit.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)
import tanitad  # noqa: E402

if not str(Path(tanitad.__file__).resolve()).startswith(str(T)):
    raise SystemExit(f"tanitad imported from {tanitad.__file__}, not from {T}")
G = M.G
if a.const_lr:
    G.lr_factor = lambda s, steps=2000: 1.0
KEYS = ("step", "ap2m", "prec", "rec", "n_conf", "centre_p50_m", "presence", "loss_mean_since_last", "lr_factor")
out = {"tree": str(T), "has_A17": hasattr(G, "lr_factor"), "const_lr": a.const_lr, "steps": a.steps,
       "platform": {"machine": platform.machine(), "torch": torch.__version__, "threads": torch.get_num_threads(),
                    "blas": [ln.strip() for ln in torch.__config__.show().splitlines() if "BLAS" in ln][:3]},
       "runs": []}
for seed in [int(s) for s in a.seeds.split(",")]:
    for arm in a.arms.split(","):
        t0 = time.time()
        r = G.run_arm(M._Toy(), arm, steps=a.steps, log_every=a.log_every, lr=1e-3, seed=seed)
        rows = [{k: row[k] for k in KEYS if k in row} for row in r["rows"]]
        out["runs"].append({"seed": seed, "arm": arm, "verdict": r["verdict"], "rows": rows,
                            "wall_s": round(time.time() - t0, 1)})
        f = rows[-1]
        print(f"seed {seed} {arm:13s} step {f['step']}: ap2m {f['ap2m']:.4f} prec {f['prec']:.4f} rec {f['rec']:.4f} "
              f"loss {f.get('loss_mean_since_last', float('nan')):.4f} -> {r['verdict']} ({time.time() - t0:.0f} s)",
              flush=True)
Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
