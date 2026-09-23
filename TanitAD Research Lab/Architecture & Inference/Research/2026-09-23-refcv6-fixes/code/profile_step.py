"""Where does a refcv6 training step spend its time? The real train(), profiled.

Runs the real `train()` with the argv a Thor smoke recorded (overriding --out/--steps and turning
the in-run eval off), and wraps `next_train_batch` -- called exactly once per step -- to drive a
`torch.profiler` schedule: 3 steps skipped, 1 warm-up, 2 recorded. Writes the operator tables
sorted by self CUDA time and by self CPU time, plus wall-clock per phase (data wait, step body).

    python profile_step.py <smoke config.json> <out dir>
"""
import json
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, schedule

sys.argv, cfg_path, out_dir = sys.argv[:1], sys.argv[1], sys.argv[2]
import refc_v3_train as T  # noqa: E402

argv = json.load(open(cfg_path, encoding="utf-8"))["argv"]


def _drop(av, flag, n=1):
    out, i = [], 0
    while i < len(av):
        if av[i] == flag:
            i += 1 + n
            continue
        out.append(av[i])
        i += 1
    return out


for f in ("--out", "--steps", "--eval-cache", "--eval-labels", "--eval-every",
          "--eval-batches", "--speed-max-sidecar-v6-eval"):
    argv = _drop(argv, f, 1)
argv += ["--out", out_dir, "--steps", "7"]

state = {"n": 0, "t_prev": None, "data_s": [], "step_s": []}


def _ready(prof):
    for key, fname in (("self_cuda_time_total", "by_cuda"), ("self_cpu_time_total", "by_cpu")):
        with open(f"{out_dir}_{fname}.txt", "w", encoding="utf-8") as fh:
            fh.write(prof.key_averages().table(sort_by=key, row_limit=45))
    print("PROF tables written", flush=True)


prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
               schedule=schedule(wait=3, warmup=1, active=2, repeat=1),
               on_trace_ready=_ready, record_shapes=False)
real_next = T.next_train_batch


def timed_next(it, dl, sampler, dpos):
    now = time.perf_counter()
    if state["t_prev"] is not None:
        torch.cuda.synchronize()
        state["step_s"].append(time.perf_counter() - state["t_prev"])
    t0 = time.perf_counter()
    out = real_next(it, dl, sampler, dpos)
    torch.cuda.synchronize()
    state["data_s"].append(time.perf_counter() - t0)
    state["t_prev"] = time.perf_counter()
    prof.step()
    state["n"] += 1
    return out


T.next_train_batch = timed_next
with prof:
    T.train(T.build_parser().parse_args(argv))
print("PROF data_wait_s", [round(x, 3) for x in state["data_s"]], flush=True)
print("PROF step_total_s", [round(x, 3) for x in state["step_s"]], flush=True)
print("PROF_DONE", flush=True)
