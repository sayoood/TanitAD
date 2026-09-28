#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""refcv7 step-cost harness -- runs the REAL trainer (`refc_v3_train.main`) with instrumentation.

Nothing in the tree is edited: every timer is a wrapper installed on a module attribute the
trainer looks up at call time, or a module forward hook installed on the built model (at
`build_optimizer(model, args)`, which receives it). Modes:

  plain     one CUDA sync per step boundary only -> the unperturbed per-step wall clock.
  timers    CUDA-synchronised timers around every named phase (data wait, forward+loss,
            backward, conflict probe, clip, optimiser, logging, save, eval) and around the
            named sub-parts (trunk main pass, stride-8 tap pass, each top-level module's
            forward, the 10 cm map branch parts, the Hungarian matcher on the CPU, the match
            cost build, the slot losses, the map loss and its per-class signal, the box
            detection census). Forward hooks that fire DURING a backward (gradient-checkpoint
            recompute) are booked under `<name>@recompute`. Inclusive times; the summary
            states the nesting.
  torchprof torch.profiler over a window of steps, NO syncs added; reports the GPU busy
            fraction (union of kernel intervals / wall) and the CUDA time per named scope,
            forward AND backward (backward ops linked to their forward scope by the autograd
            sequence number).
  loader    captures the TRAIN dataset object at DataLoader construction and times
            `ds[i]` (and each item helper) in-process for N samples, then collation of one
            batch; no model step is taken.

Every run writes <out>/profile_<mode>.json (+ steps.jsonl) and never claims more than it
measured: warm-up steps are excluded by index, conflict / log steps are separated.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import statistics
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent


class _Abort(Exception):
    """Raised by the loader mode once the dataset is captured."""


class Clock:
    def __init__(self, cuda: bool, sync: bool):
        self.cuda = cuda
        self.sync_on = sync and cuda
        self.steps: list[dict] = []
        self.cur: dict = defaultdict(float)
        self.cnt: dict = defaultdict(int)
        self.t_step0 = None
        self.in_backward = 0
        self.lock = threading.RLock()
        self.meta: dict = {}
        self.depth = 0

    def sync(self):
        if self.sync_on:
            import torch
            torch.cuda.synchronize()

    def add(self, name: str, dt: float):
        with self.lock:
            self.cur[name] += dt
            self.cnt[name] += 1

    def step_boundary(self, extra: dict | None = None):
        import torch
        if self.cuda:
            torch.cuda.synchronize()
        now = time.perf_counter()
        if self.t_step0 is not None:
            rec = {"wall": now - self.t_step0}
            rec.update({k: round(v, 6) for k, v in self.cur.items()})
            rec["_n"] = dict(self.cnt)
            if self.cuda:
                rec["max_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 3)
            if extra:
                rec.update(extra)
            self.steps.append(rec)
        self.cur = defaultdict(float)
        self.cnt = defaultdict(int)
        self.t_step0 = now


CLOCK: Clock | None = None


def timed_fn(fn, name, *, sync=True, name_fn=None):
    @functools.wraps(fn)
    def w(*a, **k):
        c = CLOCK
        if c is None:
            return fn(*a, **k)
        nm = name_fn(a, k) if name_fn else name
        if sync:
            c.sync()
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            if sync:
                c.sync()
            c.add(nm, time.perf_counter() - t)
    w.__wrapped_by_rc7__ = True
    return w


def wrap_attr(obj, attr, name, **kw):
    f = getattr(obj, attr, None)
    if f is None:
        return False
    setattr(obj, attr, timed_fn(f, name, **kw))
    return True


# --------------------------------------------------------------------------------------- #
# module hooks                                                                              #
# --------------------------------------------------------------------------------------- #
def install_module_hooks(model, clock: Clock, max_depth: int, record_fn=None):
    """Forward pre/post hooks with syncs on every module at dotted depth <= max_depth."""
    import torch
    stack_by_thread: dict = defaultdict(list)
    names = []
    for name, mod in model.named_modules():
        if not name:
            continue
        d = name.count(".")
        if d > max_depth:
            continue
        if isinstance(mod, (torch.nn.ModuleList, torch.nn.ModuleDict)):
            continue
        names.append(name)

        def pre(m, inp, _n=name):
            tid = threading.get_ident()
            clock.sync()
            rf = None
            if record_fn is not None:
                rf = record_fn(_n)
                rf.__enter__()
            stack_by_thread[tid].append((_n, time.perf_counter(), rf))

        def post(m, inp, out, _n=name):
            tid = threading.get_ident()
            st = stack_by_thread[tid]
            if not st:
                return
            n0, t0, rf = st.pop()
            clock.sync()
            dt = time.perf_counter() - t0
            tag = "mod:" + n0 + ("@recompute" if clock.in_backward else "")
            clock.add(tag, dt)
            if rf is not None:
                rf.__exit__(None, None, None)
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
    return names


def install_record_hooks(model, max_depth: int):
    """torchprof mode: record_function scopes around module forwards, no syncs."""
    import torch
    stack_by_thread: dict = defaultdict(list)
    for name, mod in model.named_modules():
        if not name or name.count(".") > max_depth:
            continue
        if isinstance(mod, (torch.nn.ModuleList, torch.nn.ModuleDict)):
            continue

        def pre(m, inp, _n=name):
            rf = torch.autograd.profiler.record_function("mod:" + _n)
            rf.__enter__()
            stack_by_thread[threading.get_ident()].append(rf)

        def post(m, inp, out, _n=name):
            st = stack_by_thread[threading.get_ident()]
            if st:
                st.pop().__exit__(None, None, None)
        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)


def rf_fn(fn, name):
    import torch

    @functools.wraps(fn)
    def w(*a, **k):
        with torch.autograd.profiler.record_function(name):
            return fn(*a, **k)
    return w


# --------------------------------------------------------------------------------------- #
# the instrumentation plan (one list, used by timers and by torchprof)                      #
# --------------------------------------------------------------------------------------- #
def phase_targets(T):
    """(object, attribute, name, sync) -- the named phases and sub-parts."""
    import torch
    from tanitad.models import agent_slots as AS
    from tanitad.models import slot_presence as SP
    from tanitad.models import box3d_head as B3
    from tanitad.models import map_head_hires as MH
    from tanitad.models import timm_trunk as TT
    from tanitad.models import refcv6_perception_branch as PB
    from tanitad.eval import detection_metrics as DM
    from tanitad.train import grad_conflict as GC
    tg = [
        (T, "frames_to_device", "h2d_frames", True),
        (torch.nn.utils, "clip_grad_norm_", "clip_grad", True),
        (T, "_train_row_scalars", "log_row", True),
        (T, "_grad_probe_row", "grad_probe_row", True),
        (PB, "grad_reach_report", "grad_reach", True),
        (MH, "grad_reach_report_hires", "grad_reach_hires", True),
        (T, "apply_lr_schedule", "lr_sched", False),
        # conflict detector
        (GC.GradientConflictDetector, "measure", "conflict_measure", True),
        (GC.GradientConflictDetector, "self_check", "conflict_selfcheck", True),
        (T, "_conflict_terms", "conflict_terms", True),
        # trunk
        (TT.TimmResNetTrunk, "forward_features", "trunk_main", True),
        (TT.TimmResNetTrunk, "s8_from_normalised", "trunk_s8_tap", True),
        # matching: the cost build (GPU ops + one .cpu() sync per element) and the CPU solver
        (AS, "_match_cost", "match_cost_build", True),
        (AS, "hungarian", "hungarian_cpu", False),
        (AS, "match_slots", "match_slots", True),
        (AS, "slot_set_loss", "slot_set_loss", True),
        (B3, "box3d_set_loss", "box3d_set_loss", True),
        (SP, "refined_slot_losses", "refined_slot_losses", True),
        (SP, "presence_term", "presence_term", True),
        (SP, "ignore_presence_weight", "ignore_presence_weight", True),
        (DM, "window_packs", "det_window_packs", True),
        (DM, "train_row_keys", "det_train_row_keys", True),
        # map loss and its per-class signal
        (MH, "hires_map_ce", "map_hires_ce", True),
        (MH, "per_class_signal", "map_per_class_signal", True),
        (MH, "per_class_log_values", "map_per_class_log_values", True),
        (MH, "map_hires_loss_row", "map_hires_loss_row", True),
        (MH, "derive_near_geometry", "map_near_geometry", True),
    ]
    return tg


def install_timers(T, clock: Clock, hook_depth: int):
    import torch
    installed = []
    for obj, attr, name, sync in phase_targets(T):
        # refc_v3_train imports some of these by module alias; patch the defining object
        if wrap_attr(obj, attr, name, sync=sync):
            installed.append(name)
    # the whole-step phases
    orig_next = T.next_train_batch

    def next_train_batch(it, dl, sampler, dpos):
        clock.step_boundary()
        t = time.perf_counter()
        r = orig_next(it, dl, sampler, dpos)
        clock.add("data_wait", time.perf_counter() - t)
        return r
    T.next_train_batch = next_train_batch

    orig_cl = T.compute_losses_v3

    def compute_losses_v3(model, batch, device, **kw):
        name = "fwd_loss" if model.training else "eval_fwd_loss"
        clock.sync()
        t = time.perf_counter()
        r = orig_cl(model, batch, device, **kw)
        clock.sync()
        clock.add(name, time.perf_counter() - t)
        return r
    T.compute_losses_v3 = compute_losses_v3

    orig_bw = torch.Tensor.backward

    def backward(self, *a, **k):
        clock.sync()
        t = time.perf_counter()
        clock.in_backward += 1
        try:
            return orig_bw(self, *a, **k)
        finally:
            clock.in_backward -= 1
            clock.sync()
            clock.add("backward", time.perf_counter() - t)
    torch.Tensor.backward = backward

    orig_grad = torch.autograd.grad

    def agrad(*a, **k):
        clock.sync()
        t = time.perf_counter()
        clock.in_backward += 1
        try:
            return orig_grad(*a, **k)
        finally:
            clock.in_backward -= 1
            clock.sync()
            clock.add("autograd_grad", time.perf_counter() - t)
    torch.autograd.grad = agrad
    # grad_conflict imports torch and calls torch.autograd.grad -> covered by the patch above

    from torch.optim.optimizer import (register_optimizer_step_post_hook,
                                       register_optimizer_step_pre_hook)
    st = {}

    def opre(opt, a, k):
        clock.sync()
        st["t"] = time.perf_counter()

    def opost(opt, a, k):
        clock.sync()
        clock.add("opt_step", time.perf_counter() - st.pop("t", time.perf_counter()))
    register_optimizer_step_pre_hook(opre)
    register_optimizer_step_post_hook(opost)

    orig_save = torch.save

    def save(*a, **k):
        t = time.perf_counter()
        r = orig_save(*a, **k)
        clock.add("save", time.perf_counter() - t)
        return r
    torch.save = save
    T.torch.save = save

    orig_bo = T.build_optimizer

    def build_optimizer(model, args):
        names = install_module_hooks(model, clock, hook_depth)
        clock.meta["hooked_modules"] = len(names)
        clock.meta["module_tree"] = module_tree(model, 3)
        return orig_bo(model, args)
    T.build_optimizer = build_optimizer
    return installed


def module_tree(model, depth):
    rows = []
    for name, mod in model.named_modules():
        if not name or name.count(".") > depth:
            continue
        n = sum(p.numel() for p in mod.parameters())
        nt = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        rows.append([name, type(mod).__name__, n, nt])
    return rows


def install_plain(T, clock: Clock):
    orig_next = T.next_train_batch

    def next_train_batch(it, dl, sampler, dpos):
        clock.step_boundary({"ab_on": int(AB["cur_on"])} if AB["lever"] else None)
        t = time.perf_counter()
        r = orig_next(it, dl, sampler, dpos)
        clock.add("data_wait", time.perf_counter() - t)
        return r
    T.next_train_batch = next_train_batch
    import torch
    orig_save = torch.save

    def save(*a, **k):
        t = time.perf_counter()
        r = orig_save(*a, **k)
        clock.add("save", time.perf_counter() - t)
        return r
    torch.save = save
    orig_cl = T.compute_losses_v3

    def compute_losses_v3(model, batch, device, **kw):
        if model.training:
            return orig_cl(model, batch, device, **kw)
        t = time.perf_counter()
        r = orig_cl(model, batch, device, **kw)
        clock.add("eval_fwd_loss", time.perf_counter() - t)
        return r
    T.compute_losses_v3 = compute_losses_v3


# --------------------------------------------------------------------------------------- #
# torch.profiler mode                                                                       #
# --------------------------------------------------------------------------------------- #
def install_torchprof(T, clock: Clock, hook_depth: int, wait: int, active: int, out: Path):
    import torch
    from torch.profiler import ProfilerActivity, profile, schedule
    for obj, attr, name, sync in phase_targets(T):
        f = getattr(obj, attr, None)
        if f is not None:
            setattr(obj, attr, rf_fn(f, "fn:" + name))
    orig_cl = T.compute_losses_v3
    T.compute_losses_v3 = rf_fn(orig_cl, "phase:fwd_loss")
    orig_bw = torch.Tensor.backward

    def backward(self, *a, **k):
        with torch.autograd.profiler.record_function("phase:backward"):
            return orig_bw(self, *a, **k)
    torch.Tensor.backward = backward
    orig_bo = T.build_optimizer

    def build_optimizer(model, args):
        install_record_hooks(model, hook_depth)
        return orig_bo(model, args)
    T.build_optimizer = build_optimizer

    state = {"prof": None, "n": 0, "t_start": None, "t_end": None}

    def on_ready(p):
        state["done"] = True
        analyse_profile(p, out, clock, state)

    prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                   schedule=schedule(wait=wait, warmup=1, active=active, repeat=1),
                   on_trace_ready=on_ready, record_shapes=False, with_stack=False)
    orig_next = T.next_train_batch

    def next_train_batch(it, dl, sampler, dpos):
        clock.step_boundary()
        if state["prof"] is None:
            prof.__enter__()
            state["prof"] = prof
        else:
            if not state.get("done"):
                prof.step()
        with torch.autograd.profiler.record_function("phase:data_wait"):
            t = time.perf_counter()
            r = orig_next(it, dl, sampler, dpos)
            clock.add("data_wait", time.perf_counter() - t)
        return r
    T.next_train_batch = next_train_batch
    return state


def _union_len(iv):
    iv = sorted(iv)
    tot, cs, ce = 0.0, None, None
    for s, e in iv:
        if cs is None:
            cs, ce = s, e
        elif s <= ce:
            ce = max(ce, e)
        else:
            tot += ce - cs
            cs, ce = s, e
    if cs is not None:
        tot += ce - cs
    return tot


def analyse_profile(p, out: Path, clock: Clock, state: dict):
    """GPU busy fraction and CUDA time per scope, fwd + bwd (sequence-number link)."""
    import torch
    evs = p.events()
    kern = []            # (start_us, end_us)
    for e in evs:
        dt = str(getattr(e, "device_type", ""))
        if "CUDA" in dt and e.time_range.elapsed_us() > 0:
            kern.append((e.time_range.start, e.time_range.end))
    # step spans from the ProfilerStep#N scopes
    steps = [e for e in evs if e.name.startswith("ProfilerStep#")]
    wall = sum(e.time_range.elapsed_us() for e in steps)
    t0 = min((e.time_range.start for e in steps), default=0)
    t1 = max((e.time_range.end for e in steps), default=0)
    kin = [(max(s, t0), min(e, t1)) for s, e in kern if e > t0 and s < t1]
    busy = _union_len(kin)
    # scope attribution: CPU-side scopes (record_function) with their cuda self time
    scope_evs = [e for e in evs if e.name.startswith(("mod:", "fn:", "phase:"))]
    by_scope = defaultdict(lambda: {"cpu_us": 0.0, "cuda_us": 0.0, "n": 0})
    for e in scope_evs:
        d = by_scope[e.name]
        d["cpu_us"] += e.time_range.elapsed_us()
        d["cuda_us"] += float(getattr(e, "device_time_total", 0.0) or getattr(e, "cuda_time_total", 0.0) or 0.0)
        d["n"] += 1
    # backward attribution: forward ops (with sequence_nr) inside a scope -> scope; the backward
    # evaluate_function events with the same sequence_nr carry the backward kernels.
    fwd_scope = {}
    # innermost enclosing scope per op via cpu_parent chain
    for e in evs:
        sn = getattr(e, "sequence_nr", -1)
        if sn is None or sn < 0 or e.name.startswith("autograd::engine"):
            continue
        par = e.cpu_parent
        sc = None
        while par is not None:
            if par.name.startswith("mod:") or par.name.startswith("fn:"):
                sc = par.name
                break
            par = par.cpu_parent
        if sc is not None and sn not in fwd_scope:
            fwd_scope[sn] = sc
    bwd = defaultdict(float)
    bwd_unattr = 0.0
    for e in evs:
        if not e.name.startswith("autograd::engine::evaluate_function"):
            continue
        sn = getattr(e, "sequence_nr", -1)
        cu = float(getattr(e, "device_time_total", 0.0) or getattr(e, "cuda_time_total", 0.0) or 0.0)
        sc = fwd_scope.get(sn)
        if sc is None:
            bwd_unattr += cu
        else:
            bwd[sc] += cu
    ka = p.key_averages()
    top = []
    for r in sorted(ka, key=lambda r: -float(getattr(r, "device_time_total", 0.0) or getattr(r, "cuda_time_total", 0.0) or 0.0))[:60]:
        top.append({"name": r.key, "n": r.count,
                    "cuda_ms": round(float(getattr(r, "device_time_total", 0.0) or getattr(r, "cuda_time_total", 0.0) or 0.0) / 1e3, 3),
                    "cpu_ms": round(r.cpu_time_total / 1e3, 3)})
    n_steps = len(steps)
    res = {"n_profiled_steps": n_steps, "wall_ms_total": round(wall / 1e3, 3),
           "wall_ms_per_step": round(wall / 1e3 / max(n_steps, 1), 3),
           "gpu_busy_ms_per_step": round(busy / 1e3 / max(n_steps, 1), 3),
           "gpu_busy_frac": round(busy / wall, 4) if wall else None,
           "scopes": {k: {"cpu_ms_per_step": round(v["cpu_us"] / 1e3 / max(n_steps, 1), 3),
                          "cuda_ms_per_step_fwd_incl": round(v["cuda_us"] / 1e3 / max(n_steps, 1), 3),
                          "calls_per_step": round(v["n"] / max(n_steps, 1), 2)}
                      for k, v in sorted(by_scope.items())},
           "bwd_cuda_ms_per_step_by_fwd_scope": {k: round(v / 1e3 / max(n_steps, 1), 3)
                                                 for k, v in sorted(bwd.items(), key=lambda x: -x[1])},
           "bwd_cuda_ms_per_step_unattributed": round(bwd_unattr / 1e3 / max(n_steps, 1), 3),
           "top_ops_by_cuda": top}
    (out / "torchprof_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    try:
        p.export_chrome_trace(str(out / "trace.json"))
    except Exception as ex:                 # noqa: BLE001
        res["trace_error"] = repr(ex)
    print("[rc7prof] torchprof analysed: busy %.3f of wall, %.1f ms/step"
          % (res["gpu_busy_frac"] or -1, res["wall_ms_per_step"]), flush=True)


# --------------------------------------------------------------------------------------- #
# loader mode                                                                               #
# --------------------------------------------------------------------------------------- #
def install_loader(T, n_samples: int, out: Path, collate_batch: int, warm_clips: int = 0):
    import torch
    real_dl = torch.utils.data.DataLoader
    cap = {}

    class DLCapture(real_dl):
        def __init__(self, ds, *a, **k):
            smp = k.get("sampler")
            if smp is not None and type(smp).__name__ == "ResumableEpochSampler" and "ds" not in cap:
                cap["ds"] = ds
                cap["kw"] = {kk: (vv if isinstance(vv, (int, float, bool, str)) else type(vv).__name__)
                             for kk, vv in k.items()}
                run_loader_bench(T, ds, n_samples, out, collate_batch, cap["kw"], warm_clips)
                raise _Abort()
            super().__init__(ds, *a, **k)
    T.torch.utils.data.DataLoader = DLCapture


def _clear_lrus(ds):
    """Empty every in-process cache a training sample would miss on Thor."""
    n = 0
    for e in getattr(ds, "episodes", []) or []:
        fr = getattr(e, "frames", None)
        c = getattr(fr, "_cache", None)
        if c is not None and getattr(c, "_lru", None):
            c._lru = None
            n += 1
    for nm in ("map_fine_store", "map_store"):
        st = getattr(ds, nm, None)
        if st is not None and hasattr(st, "_open"):
            st._open.clear()
            st._order.clear()
            n += 1
    return n


def run_loader_bench(T, ds, n, out: Path, collate_batch: int, kw, warm_clips: int = 0):
    import random
    import torch
    import torchvision.io as tvio
    clock = Clock(cuda=False, sync=False)
    global CLOCK
    CLOCK = clock
    V3 = type(ds)
    names = []
    for cls in type(ds).__mro__:
        for attr in ("_agent_item", "_agent_future_item", "_map_item", "_map_fine_item",
                     "_vis1_item", "_frames", "_load_frames", "_decode"):
            if attr in cls.__dict__:
                setattr(cls, attr, timed_fn(cls.__dict__[attr], f"ds.{cls.__name__}.{attr}", sync=False))
                names.append(f"{cls.__name__}.{attr}")
        if "__getitem__" in cls.__dict__ and cls is not V3:
            setattr(cls, "__getitem__", timed_fn(cls.__dict__["__getitem__"],
                                                 f"ds.{cls.__name__}.__getitem__", sync=False))
            names.append(f"{cls.__name__}.__getitem__")
    tvio.decode_png = timed_fn(tvio.decode_png, "png_decode", sync=False)
    from tanitad.data import v2_dataset as V2D
    V2D.V2CompressedCache._payload = timed_fn(V2D.V2CompressedCache._payload, "v2ep_payload_load",
                                              sync=False)
    names += ["torchvision.io.decode_png", "V2CompressedCache._payload"]
    rng = random.Random(0)
    if warm_clips > 0:
        eps = sorted({e for e, _ in ds.index})
        pick = set(rng.sample(eps, min(warm_clips, len(eps))))
        pool = [i for i, (e, _) in enumerate(ds.index) if e in pick]
        idx = [rng.choice(pool) for _ in range(n)]
        for i in sorted(set(idx)):             # warm the OS file cache (untimed)
            ds[i]
            _clear_lrus(ds)
        for i in sorted(set(idx)):             # twice: the D: reads are slow and variable
            ds[i]
            _clear_lrus(ds)
    else:
        idx = [rng.randrange(len(ds)) for _ in range(n)]
    per = []
    items = []
    for j, i in enumerate(idx):
        if warm_clips > 0:
            _clear_lrus(ds)
        clock.cur = defaultdict(float)
        clock.cnt = defaultdict(int)
        t = time.perf_counter()
        it = ds[i]
        dt = time.perf_counter() - t
        rec = {"i": i, "total": dt}
        rec.update(clock.cur)
        rec["n_png_decode"] = clock.cnt.get("png_decode", 0)
        per.append(rec)
        if len(items) < collate_batch:
            items.append(it)
    from torch.utils.data import default_collate
    t = time.perf_counter()
    b = default_collate(items)
    t_coll = time.perf_counter() - t
    nbytes = 0
    for v in b.values() if isinstance(b, dict) else []:
        if torch.is_tensor(v):
            nbytes += v.numel() * v.element_size()
    keys = sorted({k for r in per for k in r if k not in ("i",)})
    summ = {}
    for k in keys:
        vals = [r.get(k, 0.0) for r in per]
        summ[k] = {"median_s": round(statistics.median(vals), 5),
                   "mean_s": round(statistics.fmean(vals), 5),
                   "p90_s": round(sorted(vals)[int(0.9 * (len(vals) - 1))], 5)}
    res = {"n_samples": n, "len_ds": len(ds), "wrapped": names, "per_sample": summ,
           "warm_clips": warm_clips,
           "regime": ("WARM OS file cache, in-process LRUs cleared before every sample"
                      if warm_clips else "cold: this box's D: reads included"),
           "collate_batch": collate_batch, "collate_s": round(t_coll, 4),
           "collated_bytes": nbytes, "dataloader_kw": kw,
           "first_sample_s": round(per[0]["total"], 4) if per else None}
    (out / "loader_bench.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (out / "loader_samples.jsonl").write_text("\n".join(json.dumps(r) for r in per), encoding="utf-8")
    print("[rc7prof] loader: median %.3f s/sample over %d (collate %d: %.3f s)"
          % (summ["total"]["median_s"], n, collate_batch, t_coll), flush=True)


# --------------------------------------------------------------------------------------- #
# summary                                                                                   #
# --------------------------------------------------------------------------------------- #
# --------------------------------------------------------------------------------------- #
# levers applied by monkeypatch (A/B on the SAME tree; each one is a candidate code change) #
# --------------------------------------------------------------------------------------- #
STEP = {"k": -1, "log_every": 50, "steps": 0}


def _is_log_step() -> bool:
    k = STEP["k"]
    return ((k + 1) % max(STEP["log_every"], 1) == 0) or (k + 1 == STEP["steps"])


AB = {"lever": None, "block": 3, "warm": 6, "cur_on": False, "model": None}


def _ab_on(k: int) -> bool:
    if AB["lever"] is None or k < AB["warm"]:
        return False
    return ((k - AB["warm"]) // max(AB["block"], 1)) % 2 == 1


def _ab_apply(on: bool):
    """Flip the flag-type levers at a step boundary (the function-type ones read AB['cur_on'])."""
    import torch
    lv = AB["lever"]
    if lv == "cudnn_benchmark":
        torch.backends.cudnn.benchmark = bool(on)
    elif lv == "ckpt_off":
        br = getattr(AB["model"], "_map_hires", None)
        if br is not None:
            # frozen dataclass: the branch reads cfg.grad_ckpt at every forward
            object.__setattr__(br.cfg, "grad_ckpt", not on)
    elif lv == "compile_map":
        br = getattr(AB["model"], "_map_hires", None)
        if br is not None and AB.get("compiled"):
            for name, (eager, comp) in AB["compiled"].items():
                getattr(br, name).forward = comp if on else eager


def install_step_counter(T):
    orig = T.next_train_batch

    def next_train_batch(it, dl, sampler, dpos):
        r = orig(it, dl, sampler, dpos)
        STEP["k"] += 1
        if AB["lever"]:
            AB["cur_on"] = _ab_on(STEP["k"])
            _ab_apply(AB["cur_on"])
        return r
    T.next_train_batch = next_train_batch


def install_ab(T, lever: str, block: int, warm: int):
    """IN-PROCESS A/B: the lever is ON in alternate blocks of `block` steps after `warm` -- same
    process, same data stream, same GPU state, so the run-to-run noise (MEASURED 0.12 s/step at b2
    between two identical R7 runs) cancels. Function-type levers wrap their target and consult
    AB['cur_on']; flag-type levers are flipped at the step boundary by `_ab_apply`."""
    import numpy as np
    import torch
    from tanitad.models import agent_slots as AS
    from tanitad.models import map_head_hires as MH
    from tanitad.eval import detection_metrics as DM
    AB.update(lever=lever, block=int(block), warm=int(warm))
    on = lambda: AB["cur_on"] and torch.is_grad_enabled() and not _is_log_step()  # noqa: E731
    if lever in ("percls", "percls_census"):
        orig_row = MH.map_hires_loss_row

        def map_hires_loss_row(*a, **k):
            if on():
                k["with_metrics"] = False
            return orig_row(*a, **k)
        MH.map_hires_loss_row = map_hires_loss_row
    if lever in ("census", "percls_census"):
        orig_wp, orig_rk = DM.window_packs, DM.train_row_keys

        def window_packs(*a, **k):
            if on():
                return []
            return orig_wp(*a, **k)

        def train_row_keys(pk, *a, **k):
            return {} if not pk else orig_rk(pk, *a, **k)
        DM.window_packs, DM.train_row_keys = window_packs, train_row_keys
    if lever == "scipy":
        from scipy.optimize import linear_sum_assignment as lsa
        orig_h = AS.hungarian

        def hungarian(cost):
            if not AB["cur_on"]:
                return orig_h(cost)
            c = np.asarray(cost, dtype=np.float64)
            if c.size == 0:
                return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
            if c.shape[0] > c.shape[1]:
                r, k = lsa(c.T)
                return k.astype(np.int64), r.astype(np.int64)
            r, k = lsa(c)
            return r.astype(np.int64), k.astype(np.int64)
        AS.hungarian = hungarian
    if lever in ("ckpt_off", "compile_map"):
        orig_bo = T.build_optimizer

        def build_optimizer(model, args):
            AB["model"] = model
            if lever == "compile_map":
                # torch.compile of the 10 cm branch's encoder / decoder / near modules (the
                # GN + GELU + conv stacks at 400x240 and 1000x600). Numerically EQUIVALENT, not
                # bit-identical (fused kernels). backend from --compile-backend: 'inductor' on
                # Thor; 'eager' on the dev box only proves the plumbing (no Triton there).
                br = model._map_hires
                AB["compiled"] = {}
                for name in ("encoder", "refine", "near"):
                    m = getattr(br, name, None)
                    if m is None:
                        continue
                    eager = m.forward
                    comp = torch.compile(eager, backend=AB.get("backend", "inductor"),
                                         dynamic=False)
                    AB["compiled"][name] = (eager, comp)
            return orig_bo(model, args)
        T.build_optimizer = build_optimizer


def install_levers(T, levers: list, out: Path):
    """Each lever = the code change it names, emulated without editing the tree.

    percls_logonly  map_hires_loss_row(with_metrics = this step is LOGGED) in training: the
                    per-class 10 cm signal is pure (no_grad) and only a LOG row reads it.
    census_logonly  detection_metrics.window_packs / train_row_keys skipped in training on
                    a step whose row is not logged (the census keys only reach LOG rows).
    scipy_lsa       agent_slots.hungarian -> scipy.optimize.linear_sum_assignment, with the
                    pair ORDER of `hungarian` (target order when transposed), so a matching
                    assignment is the same tensors in the same order.
    capture_costs   save the first 400 cost matrices (float64) after step 5 -> raw bank.
    """
    import numpy as np
    import torch
    from tanitad.models import agent_slots as AS
    from tanitad.models import map_head_hires as MH
    from tanitad.eval import detection_metrics as DM
    done = []
    if "percls_logonly" in levers:
        orig_row = MH.map_hires_loss_row

        def map_hires_loss_row(*a, **k):
            if torch.is_grad_enabled() and not _is_log_step():
                k["with_metrics"] = False
            return orig_row(*a, **k)
        MH.map_hires_loss_row = map_hires_loss_row
        done.append("percls_logonly")
    if "census_logonly" in levers:
        orig_wp, orig_rk = DM.window_packs, DM.train_row_keys

        def window_packs(*a, **k):
            if torch.is_grad_enabled() and not _is_log_step():
                return []
            return orig_wp(*a, **k)

        def train_row_keys(pk, *a, **k):
            if not pk:
                return {}
            return orig_rk(pk, *a, **k)
        DM.window_packs, DM.train_row_keys = window_packs, train_row_keys
        done.append("census_logonly")
    if "scipy_lsa" in levers:
        from scipy.optimize import linear_sum_assignment as lsa

        def hungarian(cost):
            c = np.asarray(cost, dtype=np.float64)
            if c.size == 0:
                return (np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64))
            if not np.isfinite(c).all():
                raise ValueError("cost contains non-finite entries")
            if c.shape[0] > c.shape[1]:
                r, k = lsa(c.T)
                return k.astype(np.int64), r.astype(np.int64)
            r, k = lsa(c)
            return r.astype(np.int64), k.astype(np.int64)
        AS.hungarian = hungarian
        done.append("scipy_lsa")
    if "capture_costs" in levers:
        bank = []
        orig_mc = AS._match_cost

        def _match_cost(*a, **k):
            c = orig_mc(*a, **k)
            if STEP["k"] >= 5 and len(bank) < 400:
                bank.append(np.array(c, copy=True))
                if len(bank) in (50, 400):
                    np.savez_compressed(out / "cost_bank.npz",
                                        **{f"c{i:04d}": x for i, x in enumerate(bank)})
            return c
        AS._match_cost = _match_cost
        done.append("capture_costs")
    return done


def summarise(clock: Clock, warm: int, conflict_every: int, log_every: int, steps_total: int,
              conflict_on: bool = True):
    rows = clock.steps
    # rows[k] is the step that STARTED with the k-th next_train_batch call (0-based step k)
    use = [dict(r, step=k) for k, r in enumerate(rows) if k >= warm]
    timed = any("conflict_measure" in r for r in rows)
    for r in use:
        if timed:
            r["conflict"] = int("conflict_measure" in r)
        else:       # plain / torchprof: the loop's own modulo (controls read long before warm)
            r["conflict"] = int(conflict_on and r["step"] % max(conflict_every, 1) == 0)
        r["logged"] = int(((r["step"] + 1) % log_every == 0) or (r["step"] + 1 == steps_total))
    plain = [r for r in use if not r["conflict"] and not r["logged"] and "save" not in r
             and "eval_fwd_loss" not in r]
    conf = [r for r in use if r["conflict"] and not r["logged"] and "save" not in r]
    keys = sorted({k for r in use for k in r if not k.startswith("_") and k not in
                   ("step", "conflict", "logged", "is_conflict_step", "is_log_step")})

    def med(rs, k):
        v = [r.get(k, 0.0) for r in rs]
        return round(statistics.median(v), 5) if v else None

    def mean(rs, k):
        v = [r.get(k, 0.0) for r in rs]
        return round(statistics.fmean(v), 5) if v else None
    out = {"n_rows": len(rows), "warm_excluded": warm, "n_plain": len(plain),
           "n_conflict": len(conf),
           "plain_median": {k: med(plain, k) for k in keys},
           "plain_mean": {k: mean(plain, k) for k in keys},
           "conflict_median": {k: med(conf, k) for k in keys} if conf else {},
           "wall_all_median": med(use, "wall"), "wall_all_mean": mean(use, "wall")}
    if conf and plain:
        dc = out["conflict_median"]["wall"] - out["plain_median"]["wall"]
        out["conflict_extra_s"] = round(dc, 4)
        out["conflict_amortised_s_per_step"] = round(dc / max(conflict_every, 1), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--rung", required=True)
    ap.add_argument("--pathmap", required=True)
    ap.add_argument("--mode", choices=("plain", "timers", "torchprof", "loader", "opcount"),
                    default="timers")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warm", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--prefetch", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=1000000)
    ap.add_argument("--save-every", type=int, default=1000000)
    ap.add_argument("--hook-depth", type=int, default=2)
    ap.add_argument("--prof-wait", type=int, default=8)
    ap.add_argument("--prof-active", type=int, default=6)
    ap.add_argument("--loader-n", type=int, default=24)
    ap.add_argument("--loader-clips", type=int, default=0,
                    help="WARM-CACHE mode: sample windows from K clips only, read them once "
                         "(OS file cache warm), then time each sample with every in-process LRU "
                         "CLEARED first (the training regime: ~0 hit rate over 4,369 clips) -- "
                         "i.e. CPU cost without this box's slow D: reads")
    ap.add_argument("--edit", action="append", default=[],
                    help="extra edit: set:--flag:v1,v2 | drop:--flag")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-eval-at-end", action="store_true",
                    help="drop --eval-cache so the final-step eval does not run")
    ap.add_argument("--require-cpu", action="store_true",
                    help="hide every GPU (CUDA_VISIBLE_DEVICES=-1), pass --device cpu, and REFUSE "
                         "if torch still sees a CUDA device")
    ap.add_argument("--keep-compile", action="store_true",
                    help="keep --trunk-compile (Thor; Inductor needs Triton, absent on the dev box)")
    ap.add_argument("--ab", default=None,
                    choices=("percls", "census", "percls_census", "scipy", "ckpt_off",
                             "cudnn_benchmark", "compile_map"),
                    help="in-process A/B of ONE lever in alternating blocks (plain mode)")
    ap.add_argument("--ab-block", type=int, default=3)
    ap.add_argument("--compile-backend", default="inductor")
    ap.add_argument("--lever", action="append", default=[],
                    choices=("percls_logonly", "census_logonly", "scipy_lsa", "capture_costs"))
    a = ap.parse_args()

    if a.require_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        a.edit = list(a.edit) + ["set:--device:cpu"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(Path(a.tree) / "stack"), str(Path(a.tree) / "stack" / "scripts"),
                    str(Path(a.tree) / "taniteval"), str(HERE)]
    import rc7_argv
    extra = []
    for e in a.edit:
        parts = e.split(":", 2)
        if parts[0] == "set":
            vals = [v for v in parts[2].split(",") if v != ""] if len(parts) > 2 else []
            extra.append(("set", parts[1], vals))
        else:
            extra.append(("drop", parts[1]))
    if a.no_eval_at_end:
        extra += [("drop", "--eval-cache"), ("drop", "--eval-labels"),
                  ("drop", "--speed-max-sidecar-v6-eval")]
    run_dir = out / "run"
    spec = rc7_argv.build(a.tree, a.rung, a.pathmap, batch=a.batch, steps=a.steps,
                          out=str(run_dir).replace("\\", "/"), workers=a.workers,
                          log_every=a.log_every, eval_every=a.eval_every,
                          save_every=a.save_every, extra_edits=extra, prefetch=a.prefetch,
                          keep_compile=a.keep_compile)
    import torch
    if a.require_cpu and (torch.cuda.device_count() != 0 or torch.cuda.is_available()):
        raise SystemExit("--require-cpu but torch sees a CUDA device: REFUSED")
    import tanitad
    tf = Path(tanitad.__file__).resolve()
    if not str(tf).lower().startswith(str(Path(a.tree).resolve()).lower()):
        raise SystemExit(f"tanitad imported from {tf}, not from the tree {a.tree}")
    import refc_v3_train as T
    cuda = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0 and not a.require_cpu)
    global CLOCK
    CLOCK = Clock(cuda=cuda, sync=(a.mode == "timers"))
    rec = {"mode": a.mode, "rung": a.rung, "args": vars(a), "tanitad_file": str(tf),
           "torch": torch.__version__, "cuda": cuda,
           "device": torch.cuda.get_device_name(0) if cuda else None,
           "host": platform.node(), "python": sys.version.split()[0],
           "omp_threads": os.environ.get("OMP_NUM_THREADS"),
           "torch_threads": torch.get_num_threads(), "spec": spec,
           "t_start": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "run_spec.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    state = None
    if a.mode == "timers":
        rec["instrumented"] = install_timers(T, CLOCK, a.hook_depth)
    elif a.mode == "plain":
        install_plain(T, CLOCK)
    elif a.mode == "torchprof":
        state = install_torchprof(T, CLOCK, a.hook_depth, a.prof_wait, a.prof_active, out)
    elif a.mode == "loader":
        install_loader(T, a.loader_n, out, min(a.batch, a.loader_n), a.loader_clips)
    elif a.mode == "opcount":
        import opcount
        state = None
        _oc_finish = opcount.install(T, phase_targets, out, a.hook_depth,
                                     set(range(a.warm, a.steps)))
    STEP["log_every"], STEP["steps"] = int(a.log_every), int(a.steps)
    if a.ab:
        AB["backend"] = a.compile_backend
        install_ab(T, a.ab, a.ab_block, a.warm)
        rec["ab"] = {"lever": a.ab, "block": a.ab_block, "warm": a.warm}
    install_step_counter(T)
    rec["levers"] = install_levers(T, a.lever, out)
    (out / "run_spec.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    err = None
    try:
        T.main(spec["argv"])
    except _Abort:
        pass
    except SystemExit as e:                          # noqa: BLE001
        if e.code not in (0, None):
            err = f"SystemExit: {e.code}"
    except BaseException as e:                       # noqa: BLE001
        import traceback
        traceback.print_exc()
        err = f"{type(e).__name__}: {e}"
    if a.mode == "opcount":
        _oc_finish()
    if state is not None and state.get("prof") is not None and not state.get("done"):
        try:
            state["prof"].__exit__(None, None, None)
        except Exception:                            # noqa: BLE001
            pass
    CLOCK.step_boundary()
    rec["t_total_s"] = round(time.time() - t0, 1)
    rec["error"] = err
    rec["meta"] = {k: v for k, v in CLOCK.meta.items() if not k.startswith("_")}
    rec["peak_mem_gb"] = (round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if cuda else None)
    ce = 10
    try:
        from launch_gate import flag_values
        ce = int((flag_values(spec["argv"], "--conflict-every") or ["1"])[0])
    except Exception:                                # noqa: BLE001
        pass
    if a.mode in ("timers", "plain", "torchprof"):
        cd_off = False
        try:
            from launch_gate import flag_values
            cd_off = (flag_values(spec["argv"], "--conflict-detector") or ["auto"])[0] == "off"
        except Exception:                            # noqa: BLE001
            pass
        rec["summary"] = summarise(CLOCK, a.warm, ce, a.log_every, a.steps,
                                   conflict_on=not cd_off)
        with open(out / "steps.jsonl", "w", encoding="utf-8") as fh:
            for k, r in enumerate(CLOCK.steps):
                fh.write(json.dumps(dict(r, step_index=k)) + "\n")
    (out / f"profile_{a.mode}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print("[rc7prof] done rung=%s mode=%s err=%s wall_median=%s"
          % (a.rung, a.mode, err, (rec.get("summary") or {}).get("plain_median", {}).get("wall")),
          flush=True)
    return 0 if err is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
