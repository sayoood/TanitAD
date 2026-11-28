#!/usr/bin/env python3
"""v2-corpus throughput bench for `flagship-v2corpus-30k` (TanitAD).

WHY: pod1 (RTX A6000) is running `train_flagship4b.py --v2 ... --batch-size 16
--accum 4 --grad-checkpoint` at 10.65 s/step with data_s/step = 1.25 s (~12 %),
i.e. the job is COMPUTE-bound, not input-bound.  This bench measures, on a FREE
device, the two compute-side levers at CONSTANT effective batch 64:

    (1) batch x accum   : 16x4 (production baseline) | 32x2 | 64x1
    (2) --grad-checkpoint on vs off

METHOD / FIDELITY
-----------------
* Real `WorldModel(flagship4b_config())` with the exact `--v2` lever block
  copied from train_flagship4b.py, `--sigreg-free-dims 64`, rollout_k 12,
  speed_input (action_dim 3), v2 labels.
* Real `flagship_loss` + real `h15_loss` + real AdamW + real clip_grad_norm,
  in the SAME order and with the SAME per-micro-batch `.item()` sync the
  trainer performs (that sync is part of production step time).
* Batches are produced by the trainer's OWN `FlagshipWindowDataset` over
  contract episodes, at the production geometry (9ch x 256px, window 8,
  max_horizon 20).  Frame VALUES are synthetic; every shape, dtype and code
  path is production.  Compute here is value-independent (no data-dependent
  control flow beyond h15's random gate and dropout masks, both
  value-independent), so s/step transfers.
* Batches are pre-materialised on CPU and reused, so the input pipeline is
  removed from the measurement: this isolates COMPUTE, which is the lever.
  Loader effects (--workers, --v2-lru) are bounded analytically instead --
  see V2_THROUGHPUT_BENCH.md.
* Effective batch is 64 in EVERY config, so all arms see the same number of
  samples per optimizer step.

Utilisation is sampled continuously in a background thread (>=10 samples per
config, per the project's "never trust one sample" rule).

Run (eval pod, A40):
    PYTHONPATH=/root/v2bench/stack python3 bench_v2_throughput.py \
        --out /root/v2bench/results.json --timed-steps 12 --warmup-steps 3
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

STACK = os.environ.get("TANITAD_STACK", "/root/v2bench/stack")
if STACK not in sys.path:
    sys.path.insert(0, STACK)
_SCRIPTS = str(Path(STACK) / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import numpy as np  # noqa: E402

from tanitad.config import flagship4b_config  # noqa: E402
from tanitad.data._contract import (assert_contract,  # noqa: E402
                                    finite_diff_accel)
from tanitad.data.toy_driving import ToyEpisode  # noqa: E402
from tanitad.models.fourbrain import WorldModel  # noqa: E402
from tanitad.train.flagship_losses import (LossWeights, build_grounding,  # noqa: E402
                                           flagship_loss, horizon_plan)
from torch.utils.data import default_collate  # noqa: E402

import train_flagship4b as TF  # noqa: E402

SIGREG_VARIANT = "full_relaxed"
EFFECTIVE_BATCH = 64


# --------------------------------------------------------------------------- #
# production config: mirrors train_flagship4b.train() for the live invocation  #
# --------------------------------------------------------------------------- #
def production_cfg(grad_checkpoint: bool):
    """`--config flagship4b --v2 --sigreg-free-dims 64 [--grad-checkpoint]`."""
    cfg = flagship4b_config()
    if grad_checkpoint:
        cfg.encoder.grad_checkpoint = True
    # --- verbatim --v2 lever block (train_flagship4b.py) ---
    cfg.v2_ego_to_planners = True
    cfg.v2_ego_dropout = 0.25
    cfg.v2_fa_dropout = 0.3
    cfg.v2_goal_decode = True
    cfg.v2_nav_dropout = 0.5
    cfg.v2_traj_jerk = 0.02
    cfg.v2_gated_intent = True
    cfg.v2_anchor_tactical = True
    cfg.v2_route_from_vision = True
    cfg.v2_encoder_ego_decorr = True
    cfg.v2_labels = True
    cfg.v2_invdyn_gradscale = 0.25
    # speed-input (implied by --v2): action_dim 2 -> 3
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    cfg.train.rollout_k = 12                    # --v2 default
    cfg.loss.sigreg.free_dims = 64              # --sigreg-free-dims 64
    return cfg


def production_weights():
    """LossWeights at the trainer's argparse defaults (the live invocation
    overrides none of them)."""
    return LossWeights(
        pred=1.0, tacpred=0.5, roll=0.5, goal=0.5, wp=1.0, man=0.5,
        route=0.5, route_vis=0.3, invdyn=2.0, fwd=1.0,
        sigreg=production_cfg(False).loss.sigreg.weight,
        inv=production_cfg(False).loss.inv_dyn_weight, decorr=0.05)


# --------------------------------------------------------------------------- #
# synthetic corpus at PRODUCTION geometry (9ch x 256px)                        #
# --------------------------------------------------------------------------- #
def _poses(T: int, dt=0.1, v0=12.0, yaw_rate=0.0, accel=0.0):
    rows, x, y, yaw, v = [], 0.0, 0.0, 0.0, v0
    for _ in range(T):
        rows.append([x, y, yaw, v])
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += yaw_rate * dt
        v = max(0.0, v + accel * dt)
    return torch.tensor(rows, dtype=torch.float32)


def _episode(T, eid, channels, size, yaw_rate=0.0, accel=0.0):
    """A contract episode at the D-015 9-channel RGB-stack geometry.

    Built directly rather than via `assemble_episode`, whose `assert_contract`
    call is hard-wired to the 1-channel BEV toy contract; the real 9-ch corpora
    (comma2k19 / PhysicalAI) validate with `channels=9`, which is what we do.
    """
    g = torch.Generator().manual_seed(1000 + eid)
    frames = torch.rand(T, channels, size, size, generator=g)
    poses = _poses(T, yaw_rate=yaw_rate, accel=accel)
    poses_arr = poses.numpy().astype(np.float32)
    acc = finite_diff_accel(poses_arr[:, 3], 0.1)
    actions = np.column_stack([np.full(T, yaw_rate, np.float32), acc])
    ep = ToyEpisode(frames=frames,
                    actions=torch.from_numpy(actions).to(torch.float32),
                    poses=torch.from_numpy(poses_arr).to(torch.float32),
                    episode_id=eid)
    assert_contract(ep, channels=channels)
    return ep


def build_pool(cfg, plan, n_batches: int, ep_len: int, seed: int = 0):
    """`n_batches` collated CPU batches of EFFECTIVE_BATCH samples, built through
    the trainer's own FlagshipWindowDataset (so every key/shape is production)."""
    ch, size = cfg.encoder.in_channels, cfg.encoder.image_size
    kin = [(0.06, 0.0), (-0.06, 0.0), (0.0, 0.0), (0.0, -1.2),
           (0.02, 0.6), (-0.03, -0.4)]
    eps = [_episode(ep_len, i, ch, size, yr, ac)
           for i, (yr, ac) in enumerate(kin)]
    ds = TF.FlagshipWindowDataset(eps, window=cfg.predictor.window,
                                  max_horizon=plan.max_horizon,
                                  maneuver_h=plan.maneuver_h,
                                  channels=ch, labels_v2=cfg.v2_labels)
    n_need = n_batches * EFFECTIVE_BATCH
    assert len(ds) >= EFFECTIVE_BATCH, f"only {len(ds)} windows"
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(ds), generator=g).tolist()
    idxs = [order[i % len(order)] for i in range(n_need)]
    pool = []
    for b in range(n_batches):
        chunk = idxs[b * EFFECTIVE_BATCH:(b + 1) * EFFECTIVE_BATCH]
        pool.append(default_collate([ds[i] for i in chunk]))
    return pool, len(ds)


def slice_batch(batch, lo, hi):
    return {k: (v[lo:hi] if torch.is_tensor(v) else v) for k, v in batch.items()}


# --------------------------------------------------------------------------- #
# utilisation sampler                                                          #
# --------------------------------------------------------------------------- #
class UtilSampler(threading.Thread):
    """Background nvidia-smi poller (>=10 samples per timed region).

    Also records SM clock and power draw: this A40 is a 300 W part that
    SW_POWER_CAP-throttles under sustained tensor load, so a config that raises
    utilisation can LOSE clock. Recording clocks lets the report normalise for
    that instead of mistaking a power-cap artifact for a config effect.
    """

    def __init__(self, period=0.4, gpu_index=0):
        super().__init__(daemon=True)
        self.period, self.gpu_index = period, gpu_index
        self.samples: list[int] = []
        self.clocks: list[int] = []
        self.power: list[float] = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,clocks.sm,power.draw",
                     "--format=csv,noheader,nounits",
                     f"--id={self.gpu_index}"],
                    capture_output=True, text=True, timeout=5)
                v = out.stdout.strip().splitlines()
                if v:
                    parts = [p.strip() for p in v[0].split(",")]
                    self.samples.append(int(float(parts[0])))
                    self.clocks.append(int(float(parts[1])))
                    self.power.append(float(parts[2]))
            except Exception:
                pass
            self._halt.wait(self.period)

    def stop(self):
        self._halt.set()
        self.join(timeout=10)


# --------------------------------------------------------------------------- #
# one config                                                                   #
# --------------------------------------------------------------------------- #
def run_config(name, batch_size, accum, grad_ckpt, pool, device,
               warmup_steps, timed_steps, seed=0):
    assert batch_size * accum == EFFECTIVE_BATCH, (batch_size, accum)
    cfg = production_cfg(grad_ckpt)
    plan = horizon_plan(cfg, op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20)
    weights = production_weights()

    torch.manual_seed(seed)                    # identical init across configs
    model = WorldModel(cfg).to(device)
    grounding = build_grounding(model.state_dim, device=device)
    params = list(model.parameters()) + list(grounding.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, betas=cfg.train.betas,
                            weight_decay=0.05)
    model.train()
    grounding.train()

    n_params = sum(p.numel() for p in params)
    rec = {"config": name, "batch_size": batch_size, "accum": accum,
           "effective_batch": batch_size * accum,
           "grad_checkpoint": grad_ckpt, "trainable_params": n_params,
           "status": "ok"}

    def one_step(pb):
        opt.zero_grad(set_to_none=True)
        h15_fired = 0
        t0 = time.perf_counter()
        for m in range(accum):
            b = slice_batch(pb, m * batch_size, (m + 1) * batch_size)
            frames = b["frames"].to(device, non_blocking=True)
            fut = b["future_frames"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                states = model.encode_window(frames)
                fut_states = model.encode_window(fut[:, plan.needed_fut])
                total, log, parts = flagship_loss(
                    model, grounding, b, states, fut_states, plan, cfg,
                    weights=weights, sigreg_variant=SIGREG_VARIANT,
                    sigreg_free_dims=cfg.loss.sigreg.free_dims,
                    pose_scale=10.0, fwd_step_weight=0.5, device=device)
                loss_h15 = TF.h15_loss(model, frames, fut, cfg, device)
                total = total + cfg.h15.weight * loss_h15
            (total / accum).backward()
            # the trainer performs this .item() sync every micro-batch
            if float(loss_h15.item()) != 0.0:
                h15_fired += 1
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        torch.cuda.synchronize()
        return time.perf_counter() - t0, h15_fired

    try:
        torch.manual_seed(seed + 7)
        for i in range(warmup_steps):
            one_step(pool[i % len(pool)])

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        sampler = UtilSampler()
        sampler.start()
        torch.manual_seed(seed + 99)           # same RNG stream for every config
        times, fired = [], 0
        t_wall0 = time.perf_counter()
        for i in range(timed_steps):
            dt, hf = one_step(pool[i % len(pool)])
            times.append(dt)
            fired += hf
        wall = time.perf_counter() - t_wall0
        sampler.stop()

        rec.update({
            "s_per_step_median": round(statistics.median(times), 4),
            "s_per_step_mean": round(statistics.fmean(times), 4),
            "s_per_step_min": round(min(times), 4),
            "s_per_step_max": round(max(times), 4),
            "s_per_step_stdev": round(statistics.stdev(times), 4)
            if len(times) > 1 else 0.0,
            "timed_steps": timed_steps,
            "wall_s": round(wall, 2),
            "h15_fired_micro": fired,
            "h15_micro_total": timed_steps * accum,
            "peak_mem_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
            "peak_mem_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
            "gpu_util_samples": sampler.samples,
            "gpu_util_n": len(sampler.samples),
            "gpu_util_mean": round(statistics.fmean(sampler.samples), 1)
            if sampler.samples else None,
            "gpu_util_median": statistics.median(sampler.samples)
            if sampler.samples else None,
            "gpu_util_min": min(sampler.samples) if sampler.samples else None,
            "gpu_util_max": max(sampler.samples) if sampler.samples else None,
            "sm_clock_mean_mhz": round(statistics.fmean(sampler.clocks), 1)
            if sampler.clocks else None,
            "sm_clock_min_mhz": min(sampler.clocks) if sampler.clocks else None,
            "power_mean_w": round(statistics.fmean(sampler.power), 1)
            if sampler.power else None,
            "power_max_w": max(sampler.power) if sampler.power else None,
            "step_times": [round(t, 4) for t in times],
        })
    except torch.cuda.OutOfMemoryError as e:
        rec.update({"status": "OOM", "error": str(e)[:400],
                    "peak_mem_alloc_gib": round(
                        torch.cuda.max_memory_allocated() / 2**30, 3)})
    except Exception as e:  # noqa: BLE001
        rec.update({"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:600]})

    del model, grounding, opt, params
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return rec


# --------------------------------------------------------------------------- #
# profiler                                                                     #
# --------------------------------------------------------------------------- #
def profile_config(name, batch_size, accum, grad_ckpt, pool, device, seed=0):
    """One profiled optimizer step -> top CUDA-time operators."""
    from torch.profiler import ProfilerActivity, profile

    cfg = production_cfg(grad_ckpt)
    plan = horizon_plan(cfg, op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20)
    weights = production_weights()
    torch.manual_seed(seed)
    model = WorldModel(cfg).to(device)
    grounding = build_grounding(model.state_dim, device=device)
    params = list(model.parameters()) + list(grounding.parameters())
    opt = torch.optim.AdamW(params, lr=3e-4, betas=cfg.train.betas,
                            weight_decay=0.05)
    model.train(); grounding.train()

    def step(pb):
        opt.zero_grad(set_to_none=True)
        for m in range(accum):
            b = slice_batch(pb, m * batch_size, (m + 1) * batch_size)
            frames = b["frames"].to(device)
            fut = b["future_frames"].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                states = model.encode_window(frames)
                fut_states = model.encode_window(fut[:, plan.needed_fut])
                total, log, parts = flagship_loss(
                    model, grounding, b, states, fut_states, plan, cfg,
                    weights=weights, sigreg_variant=SIGREG_VARIANT,
                    sigreg_free_dims=cfg.loss.sigreg.free_dims,
                    pose_scale=10.0, fwd_step_weight=0.5, device=device)
                loss_h15 = TF.h15_loss(model, frames, fut, cfg, device)
                total = total + cfg.h15.weight * loss_h15
            (total / accum).backward()
            float(loss_h15.item())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        torch.cuda.synchronize()

    step(pool[0])                                        # warm
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        step(pool[0])
    tbl = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25)
    rows = []
    for ev in sorted(prof.key_averages(), key=lambda e: -e.self_device_time_total)[:25]:
        rows.append({"op": ev.key,
                     "self_cuda_ms": round(ev.self_device_time_total / 1000, 2),
                     "cuda_total_ms": round(ev.device_time_total / 1000, 2),
                     "count": ev.count})
    del model, grounding, opt, params
    torch.cuda.empty_cache()
    return {"config": name, "table": tbl, "top_ops": rows}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/root/v2bench/results.json")
    ap.add_argument("--timed-steps", type=int, default=12)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--pool-batches", type=int, default=2)
    ap.add_argument("--ep-len", type=int, default=110)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated config names")
    ap.add_argument("--interleave", default="",
                    help="comma-separated configs measured ROUND-ROBIN with the "
                         "order REVERSED on odd repeats (ABBA). Use when the "
                         "device is thermally drifting or shares the GPU with "
                         "another tenant: a sequential sweep then confounds "
                         "config with time, an interleaved one does not.")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    device = "cuda"
    props = torch.cuda.get_device_properties(0)
    cfg0 = production_cfg(True)
    plan0 = horizon_plan(cfg0, op_fwd_k=4, tac_fwd_k=16, str_fwd_k=20)

    print(f"[env] {props.name} {props.total_memory/2**30:.1f} GiB | "
          f"torch {torch.__version__} | {platform.node()}", flush=True)
    print(f"[cfg] {cfg0.encoder.in_channels}ch x {cfg0.encoder.image_size}px, "
          f"window {cfg0.predictor.window}, max_horizon {plan0.max_horizon}, "
          f"needed_fut {len(plan0.needed_fut)}, rollout_k {cfg0.train.rollout_k}",
          flush=True)

    t0 = time.time()
    pool, n_windows = build_pool(cfg0, plan0, args.pool_batches, args.ep_len)
    per_sample_mb = sum(v.numel() * v.element_size() for v in pool[0].values()
                        if torch.is_tensor(v)) / 1e6 / EFFECTIVE_BATCH
    print(f"[pool] {len(pool)} x {EFFECTIVE_BATCH} from {n_windows} windows in "
          f"{time.time()-t0:.1f}s ({per_sample_mb:.1f} MB/sample)", flush=True)

    grid = [
        ("gc_on_16x4",   16, 4, True),    # production baseline
        ("gc_on_32x2",   32, 2, True),
        ("gc_on_64x1",   64, 1, True),
        ("gc_off_16x4",  16, 4, False),
        ("gc_off_32x2",  32, 2, False),
        ("gc_off_64x1",  64, 1, False),
        # micro-batch 8 pair: the ONLY micro-batch where grad-checkpoint OFF is
        # expected to fit on a 44 GiB card, so it yields the clean gc on/off
        # compute ratio at MATCHED micro-batch, plus the activation-memory slope
        # needed to extrapolate what gc-off would demand at micro-batch 16.
        ("gc_on_8x8",     8, 8, True),
        ("gc_off_8x8",    8, 8, False),
    ]
    if args.only:
        keep = {s.strip() for s in args.only.split(",")}
        grid = [g for g in grid if g[0] in keep]

    results = {
        "meta": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": platform.node(),
            "gpu": props.name,
            "gpu_total_gib": round(props.total_memory / 2**30, 2),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "effective_batch": EFFECTIVE_BATCH,
            "timed_steps": args.timed_steps,
            "warmup_steps": args.warmup_steps,
            "geometry": {
                "in_channels": cfg0.encoder.in_channels,
                "image_size": cfg0.encoder.image_size,
                "window": cfg0.predictor.window,
                "max_horizon": plan0.max_horizon,
                "needed_fut": len(plan0.needed_fut),
                "rollout_k": cfg0.train.rollout_k,
                "sigreg_free_dims": cfg0.loss.sigreg.free_dims,
            },
            "per_sample_mb": round(per_sample_mb, 2),
            "note": ("synthetic frame VALUES at production geometry; input "
                     "pipeline deliberately excluded to isolate compute"),
        },
        "configs": [],
    }

    if args.interleave:
        want = [s.strip() for s in args.interleave.split(",") if s.strip()]
        by_name = {g[0]: g for g in grid}
        missing = [w for w in want if w not in by_name]
        assert not missing, f"unknown configs: {missing}"
        plan_runs = []
        for r in range(args.repeats):
            seq = want if r % 2 == 0 else list(reversed(want))
            plan_runs.extend((r, by_name[w]) for w in seq)
        print(f"[interleave] {len(plan_runs)} instances "
              f"({args.repeats} repeats x {len(want)} configs, ABBA order)",
              flush=True)
        results["meta"]["design"] = {
            "mode": "interleaved", "configs": want, "repeats": args.repeats,
            "order": "reversed on odd repeats (ABBA)",
            "why": ("the A40 power-throttles under sustained load AND shares "
                    "this pod with another tenant's jobs; a sequential sweep "
                    "would confound config with elapsed time"),
        }
        per_cfg: dict[str, list] = {w: [] for w in want}
        for (r, (name, bs, ac, gc)) in plan_runs:
            print(f"\n[run] repeat {r} {name}: batch {bs} x accum {ac}, "
                  f"grad_ckpt={gc}", flush=True)
            rec = run_config(name, bs, ac, gc, pool, device,
                             args.warmup_steps, args.timed_steps)
            rec["repeat"] = r
            if rec["status"] == "ok":
                print(f"      s/step median {rec['s_per_step_median']:.3f} | "
                      f"util {rec['gpu_util_mean']}% | "
                      f"clk {rec['sm_clock_mean_mhz']} MHz | "
                      f"peak {rec['peak_mem_reserved_gib']:.2f} GiB", flush=True)
                per_cfg[name].append(rec)
            else:
                print(f"      {rec['status']}: {rec.get('error','')[:150]}",
                      flush=True)
            results["configs"].append(rec)
            Path(args.out).write_text(json.dumps(results, indent=2))
        # pooled per-config summary across repeats
        summary = {}
        for w, recs in per_cfg.items():
            if not recs:
                continue
            allt = [t for rr in recs for t in rr["step_times"]]
            summary[w] = {
                "n_instances": len(recs), "n_steps": len(allt),
                "pooled_median_s": round(statistics.median(allt), 4),
                "per_repeat_median_s": [rr["s_per_step_median"] for rr in recs],
                "peak_mem_reserved_gib": recs[0]["peak_mem_reserved_gib"],
                "peak_mem_alloc_gib": recs[0]["peak_mem_alloc_gib"],
                "gpu_util_mean": round(statistics.fmean(
                    [rr["gpu_util_mean"] for rr in recs]), 1),
                "sm_clock_mean_mhz": round(statistics.fmean(
                    [rr["sm_clock_mean_mhz"] for rr in recs]), 1),
            }
        base = summary.get("gc_on_16x4")
        if base:
            for w, s in summary.items():
                s["speedup_vs_gc_on_16x4"] = round(
                    base["pooled_median_s"] / s["pooled_median_s"], 4)
        results["summary"] = summary
        print("\n=== POOLED SUMMARY (median over all repeats) ===", flush=True)
        for w, s in summary.items():
            print(f"  {w:14s} {s['pooled_median_s']:7.3f} s/step "
                  f"(n={s['n_steps']}) | speedup "
                  f"{s.get('speedup_vs_gc_on_16x4','-')}x | "
                  f"util {s['gpu_util_mean']}% | clk {s['sm_clock_mean_mhz']} | "
                  f"peak {s['peak_mem_reserved_gib']} GiB", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\n[done] -> {args.out}", flush=True)
        return

    for (name, bs, ac, gc) in grid:
        print(f"\n[run] {name}: batch {bs} x accum {ac}, grad_ckpt={gc}",
              flush=True)
        rec = run_config(name, bs, ac, gc, pool, device,
                         args.warmup_steps, args.timed_steps)
        if rec["status"] == "ok":
            print(f"      s/step median {rec['s_per_step_median']:.3f} "
                  f"(mean {rec['s_per_step_mean']:.3f}, "
                  f"sd {rec['s_per_step_stdev']:.3f}) | "
                  f"util {rec['gpu_util_mean']}% n={rec['gpu_util_n']} | "
                  f"clk {rec['sm_clock_mean_mhz']} MHz | "
                  f"{rec['power_mean_w']} W | "
                  f"peak {rec['peak_mem_reserved_gib']:.2f} GiB", flush=True)
        else:
            print(f"      {rec['status']}: {rec.get('error','')[:200]}",
                  flush=True)
        results["configs"].append(rec)
        Path(args.out).write_text(json.dumps(results, indent=2))

    if args.profile:
        ok = [c for c in results["configs"] if c["status"] == "ok"]
        if ok:
            base = next((c for c in ok if c["config"] == "gc_on_16x4"), ok[0])
            best = min(ok, key=lambda c: c["s_per_step_median"])
            profs = []
            for c in {base["config"]: base, best["config"]: best}.values():
                print(f"\n[profile] {c['config']}", flush=True)
                p = profile_config(c["config"], c["batch_size"], c["accum"],
                                   c["grad_checkpoint"], pool, device)
                print(p["table"][:4000], flush=True)
                profs.append(p)
            results["profiles"] = [{"config": p["config"],
                                    "top_ops": p["top_ops"]} for p in profs]
            Path(args.out).write_text(json.dumps(results, indent=2))
            for p in profs:
                Path(args.out).with_name(
                    f"profile_{p['config']}.txt").write_text(p["table"])

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n[done] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
