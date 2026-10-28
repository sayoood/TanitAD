#!/usr/bin/env python3
"""Probe: can the DAG-ORC reader's Rig be built from another script, and does it reproduce a banked Read-A row? (read-only; CPU; 2 threads)"""
import json
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "2"
sys.dont_write_bytecode = True
ORC = sys.argv[1] if len(sys.argv) > 1 else "E:/Projects/TanitAD-artifacts/dagorc_rebase/read"
sys.path.insert(0, ORC)
t0 = time.time()
import dag_orc_read as O  # noqa: E402

a = O.default_args(work_dir="F:/TanitAD-scratch/speedsub_t1/orc_work", n_boot=2000)
rig = O.Rig(a)
print("rig built", round(time.time() - t0, 1), "s", flush=True)
d = rig.A.dump("E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1/dump_DAGOACC_s0_m10_causal_heads")
print({k: getattr(v, "shape", None) for k, v in d.items()}, flush=True)
cache = {"cx": rig.A.tacdk_context(d), "P": {}}
dumps = {"identity": d, "P10": d}
cj, tj = rig.pair_table(dumps, cache, "identity", "P10", d)
print("ADE6 identity-P10:", {k: cj["ADE6"][k] for k in ("delta", "lo", "hi", "separated")}, flush=True)
print("TAC rows keys:", list(tj)[:8] if isinstance(tj, dict) else type(tj), round(time.time() - t0, 1), "s", flush=True)
