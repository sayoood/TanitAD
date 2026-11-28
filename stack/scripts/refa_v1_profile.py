#!/usr/bin/env python3
"""refav1 STEP PROFILE — where does one training step's wall-clock go?

Deploy & Optimization FlyWheel instrument (2026-09-02). READ-ONLY on the model,
trainer and loader: `refa_v1.py`, `refa_v1_train.py` and `refav1_loader.py`
are imported exactly as they are; every marker lives in THIS file (instance-
level wrappers + `torch.profiler.record_function` + CUDA events).

WHY: the live refav1 run on Thor reports 20.66 s/step (bs 8; 21,109 steps =
5.05 days/epoch) and nobody has measured where a step's time goes. This script
answers it on the dev box (RTX 4060, 8 GB) in FRACTIONS. Absolute seconds here
are not Thor's, and dev-box -> Thor scaling is NOT linear (Thor's 20 SMs
saturate at batch 8 — CLAUDE.md), so every Thor number derived from this is
ESTIMATED and says so.

ARMS (each banks into <out>/profile.json under its own key, incrementally, so
a killed run still leaves the arms that finished):
  fit         max batch that fits a FULL step (fwd+loss+bwd+clip+AdamW), per precision
  step        warm-up + measured full steps in three loader states (live / pre-staged
              / LRU-cold) — wall-clock, CPU-side and GPU-side (CUDA event) time per
              phase — then the same steps under torch.profiler (CPU+CUDA): per-marker
              self/total time, top-20 kernels, chrome trace, GPU-busy fraction,
              backward kernel time attributed to its forward phase
  sweep       no-grad forward vs batch (1,2,4,8) for the whole forward and for the
              operative rollout alone, at fp32 / tf32 / bf16
  components  isolated fwd+bwd per component at the profile batch, with a K-ladder
              for the 30-step operative rollout (K=30 fp32 does not fit in 8 GB)
  eig         the participation instrument (`covariance_eigs` on 4096x1024, fp64)
              and the per-channel target-std instruments, standalone
  graph       manual torch.cuda.CUDAGraph replay of one operative step (forward)
              vs eager — the launch-overhead ceiling lever (c) can remove
  loader      LRU-hot / LRU-cold `batch()`, raw per-episode miss cost, cold-page-cache
              read cost, slice+dequant cost

IMPORT PINNING: `--stack-root` inserts the tree at sys.path[0] and ASSERTS the
imported `tanitad` came from it. The venv's editable install points at the G:
mount, and an MSYS-style `/c/...` PYTHONPATH is silently not a Windows path
(`C:/c/...`) — both route the import to G:, which cannot run the stack.

Run (dev box, off-Drive mirror):
  PYTHONPATH="C:/Users/Admin/tanitad-wt/stack" PYTHONIOENCODING=utf-8 python \
    stack/scripts/refa_v1_profile.py --stack-root C:/Users/Admin/tanitad-wt/stack \
    --cache C:/Users/Admin/refav1_probe/profcache_fp8 --episodes C:/Users/Admin/refav1_probe/eps \
    --labels <v7.2 train jsonl.gz> --nav <same> --out <pkg>/raw --arms fit
"""
from __future__ import annotations

import argparse
import bisect
import gc
import hashlib
import inspect
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule

OOM = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)
VRAM_BUDGET_GB = float("inf")          # set from --mem-fraction in main()
OFFLOAD = False                        # set from --offload-activations in main()
OFFLOAD_PIN = False                    # --offload-pin; MEASURED 2026-09-02: pinning ~7 GB of
                                       # saved activations dies with a CUDA-runtime
                                       # `cudaErrorMemoryAllocation` on Windows/WDDM


def fwd_ctx():
    """Activation offload for a step that does not fit VRAM (kernel fractions
    stay valid; wall-clock does not)."""
    return (torch.autograd.graph.save_on_cpu(pin_memory=OFFLOAD_PIN) if OFFLOAD
            else nullcontext())

#: The live Thor launch line, 2026-09-02 (read from the running process by the
#: Master Mind). Defaults below reproduce it; the profile batch is whatever fits.
LIVE = dict(bs=8, lru=64, target_space="frozen", detach_aux=True, bptt_truncate=15,
            min_participation=0.0, lr=3e-4, adapter_lr_mult=0.1, clip=1.0, seed=0)
FWD_EXCL = ("feats", "actions", "future_feats")
SNAPSHOT_FILES = ("tanitad/refs/refa_v1.py", "tanitad/data/refav1_loader.py",
                  "scripts/refa_v1_train.py")


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #
def pin_stack(root: str | None) -> Path:
    if root:
        sys.path.insert(0, str(Path(root).resolve()))
    import tanitad  # noqa: F401
    got = Path(tanitad.__file__).resolve()
    if root and Path(root).resolve() not in got.parents:
        raise SystemExit(
            f"[refav1-profile] tanitad imported from {got}, not under --stack-root "
            f"{root}: the editable install won (the G: trap). Refusing to profile "
            "the wrong tree.")
    return got


def git_blob_sha(p: Path) -> dict:
    raw = p.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    h = hashlib.sha1(); h.update(f"blob {len(lf)}\0".encode()); h.update(lf)
    return {"path": str(p), "bytes": len(raw), "sha256_raw": hashlib.sha256(raw).hexdigest(),
            "git_blob_sha1_lf": h.hexdigest(), "had_crlf": b"\r\n" in raw}


def bank(out_dir: Path, key: str, payload) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "profile.json"
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d[key] = payload
    d["_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, default=_jsonable), encoding="utf-8")
    os.replace(tmp, p)
    print(f"[bank] {key} -> {p}", flush=True)


def _jsonable(o):
    if torch.is_tensor(o):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def med(xs):
    return float(statistics.median(xs)) if xs else float("nan")


@contextmanager
def precision(name: str, cache: bool = True):
    """fp32 = the live trainer (no TF32 for matmul: PyTorch default);
    tf32 = the one-line lever; bf16 = autocast on the forward (weight-cast
    cache on, as a trainer would have it; off inside CUDA-graph capture)."""
    old = torch.backends.cuda.matmul.allow_tf32
    try:
        if name == "fp32":
            torch.backends.cuda.matmul.allow_tf32 = False
            yield
        elif name == "tf32":
            torch.backends.cuda.matmul.allow_tf32 = True
            yield
        elif name == "bf16":
            torch.backends.cuda.matmul.allow_tf32 = False
            with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=cache):
                yield
        else:
            raise ValueError(name)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


class PhaseTimer:
    """CUDA-event spans per named phase — GPU-timeline time between the point
    the stream reaches the start marker and the end marker (includes idle gaps
    inside the phase, which is exactly what separates launch-bound from
    compute-bound when compared with the profiler's kernel sum)."""

    def __init__(self):
        self.ev: dict[str, list] = {}
        self.cpu: dict[str, list] = {}
        self._t: dict[str, float] = {}
        self.enabled = True

    def begin(self, name: str) -> None:
        if not self.enabled:
            return
        s = torch.cuda.Event(enable_timing=True); s.record()
        self.ev.setdefault(name, []).append([s, None])
        self._t[name] = time.perf_counter()

    def end(self, name: str) -> None:
        if not self.enabled:
            return
        e = torch.cuda.Event(enable_timing=True); e.record()
        self.ev[name][-1][1] = e
        self.cpu.setdefault(name, []).append(time.perf_counter() - self._t.pop(name))

    def collect(self) -> dict:
        """Call AFTER torch.cuda.synchronize(). Sums over calls within the
        collected window (ms); also the call count and CPU-side seconds."""
        out = {}
        for k, lst in self.ev.items():
            out[k] = {"gpu_span_ms": sum(s.elapsed_time(e) for s, e in lst if e is not None),
                      "n_calls": len(lst),
                      "cpu_s": sum(self.cpu.get(k, []))}
        self.ev, self.cpu, self._t = {}, {}, {}
        return out


def instrument(model, timer: PhaseTimer) -> None:
    """Instance-level wrappers only — the source files are untouched."""
    import torch.nn.functional as F
    from tanitad.eval import spectral

    def wrap(fn, name):
        def w(*a, **k):
            with record_function(name):
                timer.begin(name)
                try:
                    return fn(*a, **k)
                finally:
                    timer.end(name)
        return w

    def by_T(fn, base):
        def w(x):
            t = int(x.shape[1]) if x.dim() >= 2 else -1
            nm = {4: f"{base}_window", 30: f"{base}_future", 2: f"{base}_str_ext"}.get(t, f"{base}_T{t}")
            return wrap(fn, nm)(x)
        return w

    model.adapter.forward = by_T(model.adapter.forward, "P/adapter")
    model.std.forward = by_T(model.std.forward, "P/std")
    model.encode = wrap(model.encode, "P/encode")
    model._run_brains = wrap(model._run_brains, "P/brains")
    model.operative.rollout = wrap(model.operative.rollout, "P/op_rollout")
    model.tactical.rollout = wrap(model.tactical.rollout, "P/tac_rollout")
    model.strategic.rollout = wrap(model.strategic.rollout, "P/str_rollout")
    model.strategic.subspace = wrap(model.strategic.subspace, "P/str_subspace")
    model.to_enc.forward = wrap(model.to_enc.forward, "P/to_enc")
    model.proposal.forward = wrap(model.proposal.forward, "P/proposal")
    model.lat_head.forward = wrap(model.lat_head.forward, "P/heads")
    model.lon_head.forward = wrap(model.lon_head.forward, "P/heads")
    state = {"n": 0}
    orig_tf = model._tac_field

    def tac_field(field):
        state["n"] += 1
        return wrap(orig_tf, "P/tac_field_init" if state["n"] == 1
                    else "P/tac_field_targets")(field)
    model._tac_field = tac_field
    orig_fwd = model.forward

    def fwd(*a, **k):
        state["n"] = 0
        return orig_fwd(*a, **k)
    model.forward = fwd
    # forward() imports these at call time from the module, so a module-level
    # patch is what it sees. F.* patches are global to this process (a profiler).
    spectral.covariance_eigs = wrap(spectral.covariance_eigs, "P/participation_eig")
    spectral.participation_ratio = wrap(spectral.participation_ratio, "P/participation_ratio")
    F.mse_loss = wrap(F.mse_loss, "P/loss_mse")
    F.cross_entropy = wrap(F.cross_entropy, "P/loss_ce")


# --------------------------------------------------------------------------- #
# build — the trainer's own construction path, reused not re-typed
# --------------------------------------------------------------------------- #
def build(args, dev):
    scripts_dir = (Path(args.stack_root) / "scripts" if args.stack_root
                   else Path(__file__).resolve().parent)
    sys.path.insert(0, str(scripts_dir.resolve()))
    import refa_v1_train as tr                     # the trainer, as it is
    if args.stack_root and Path(tr.__file__).resolve().parent != scripts_dir.resolve():
        raise SystemExit(f"[refav1-profile] refa_v1_train imported from {tr.__file__}, "
                         f"not from {scripts_dir}")
    ns = SimpleNamespace(no_hierarchy=False, w_cf=0.0, cf_negs=3, cf_at_step=4,
                         motion_inject=False, target_space=args.target_space,
                         w_aux_head=0.0, proposal_k=1, w_sigreg=0.0, var_floor=0.0,
                         min_participation=args.min_participation,
                         bptt_truncate=args.bptt_truncate, detach_aux=args.detach_aux,
                         smoke=False)
    tr.verify_cache(Path(args.cache))
    torch.manual_seed(args.seed)
    model = tr.build_model(ns).to(dev)
    cfg = model.cfg
    cfg.sanity()
    from tanitad.data.refav1_loader import RefAV1Windows
    data = RefAV1Windows(args.cache, args.episodes, op_window=cfg.op_window,
                         op_steps=cfg.op_steps, str_dt=cfg.str_dt,
                         str_ext_steps=cfg.str_ext_steps, lru=args.lru, seed=args.seed,
                         labels_path=Path(args.labels) if args.labels else None,
                         nav_path=Path(args.nav) if args.nav else None)
    print(f"loader: {len(data)} windows over {len(data.names)} episodes", flush=True)
    with torch.no_grad():
        fit = data.batch(max(args.bs, 8))["feats"]
        model.std.fit(fit.reshape(-1, cfg.d_enc).to(dev))
    fwd_params = {n for n in inspect.signature(model.forward).parameters if n != "self"}
    return tr, model, data, fwd_params


def make_opt(model, lr, mult):
    adapter_p = list(model.adapter.parameters())
    ids = {id(p) for p in adapter_p}
    rest_p = [p for p in model.parameters() if id(p) not in ids]
    return torch.optim.AdamW([{"params": adapter_p, "lr": lr * mult},
                              {"params": rest_p, "lr": lr}], weight_decay=0.01)


def to_dev(b: dict, dev, fwd_params):
    feats = b["feats"].to(dev)
    actions = b["actions"].to(dev)
    future = b["future_feats"].to(dev)
    kw = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()
          if k in fwd_params and k not in FWD_EXCL and v is not None}
    return feats, actions, future, kw


def fetch(data, bs, mode, staged):
    t = time.perf_counter()
    with record_function("P/loader_fetch"):
        if mode == "prestaged":
            b = staged
        else:
            if mode == "coldlru":
                data._lru.clear()               # forces the miss path (live: ~98.6 % misses)
            b = data.batch(bs)
    return b, time.perf_counter() - t


def train_step(model, opt, batch, dev, fwd_params, clip, prec, timer):
    timer.begin("P/h2d")
    with record_function("P/h2d"):
        feats, actions, future, kw = to_dev(batch, dev, fwd_params)
    timer.end("P/h2d")
    timer.begin("P/forward")
    with record_function("P/forward"), precision(prec), fwd_ctx():
        out = model(feats, actions, future_feats=future, **kw)
    timer.end("P/forward")
    loss = out["loss"]
    with record_function("P/zero_grad"):
        opt.zero_grad(set_to_none=True)
    timer.begin("P/backward")
    with record_function("P/backward"):
        loss.backward()
    timer.end("P/backward")
    timer.begin("P/clip_grad")
    with record_function("P/clip_grad"):
        gnorm = nn.utils.clip_grad_norm_(model.parameters(), clip)
    timer.end("P/clip_grad")
    timer.begin("P/opt_step")
    with record_function("P/opt_step"):
        opt.step()
    timer.end("P/opt_step")
    return out, loss, gnorm


def free():
    gc.collect(); torch.cuda.empty_cache()


def out_dir_of(args) -> Path:
    args.out.mkdir(parents=True, exist_ok=True)
    return args.out


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def arm_fit(args, model, data, fwd_params, dev, timer):
    """One full step per (precision, bs): fits or OOMs, with peak memory."""
    res = []
    b8 = data.batch(8)
    for prec in args.fit_precisions:
        for bs in args.fit_bs:
            sub = {k: (v[:bs] if torch.is_tensor(v) else v) for k, v in b8.items()}
            opt = make_opt(model, args.lr, args.adapter_lr_mult)
            free(); torch.cuda.reset_peak_memory_stats()
            row = {"precision": prec, "bs": bs}
            try:
                t0 = time.perf_counter()
                timer.enabled = False
                train_step(model, opt, sub, dev, fwd_params, args.clip, prec, timer)
                torch.cuda.synchronize()
                mm = torch.cuda.max_memory_allocated() / 2**30
                row.update(fits=mm <= VRAM_BUDGET_GB, first_step_s=time.perf_counter() - t0,
                           max_mem_gb=mm, spilled=mm > VRAM_BUDGET_GB)
            except OOM as e:
                row.update(fits=False, max_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                           error=str(e).splitlines()[0][:200])
            finally:
                timer.enabled = True
                opt.zero_grad(set_to_none=True); del opt; free()
            print(f"[fit] {row}", flush=True)
            res.append(row)
    return {"rows": res, "total_gpu_gb": torch.cuda.get_device_properties(0).total_memory / 2**30}


def run_steps(args, model, opt, data, fwd_params, dev, timer, mode, staged, n_warm, n_meas,
              prof=None):
    """n_warm + n_meas steps with NO per-step sync (the trainer has none except
    the forward's own float() syncs), so loader/launch overlap is as live."""
    per_step = []
    torch.cuda.synchronize()
    t_all = time.perf_counter()
    for i in range(n_warm + n_meas):
        if i == n_warm:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            timer.collect()                         # discard the warm-up spans
            t_all = time.perf_counter()
        t0 = time.perf_counter()
        b, t_load = fetch(data, args.bs, mode, staged)
        out, loss, gnorm = train_step(model, opt, b, dev, fwd_params, args.clip,
                                      args.precision, timer)
        per_step.append({"cpu_step_s": time.perf_counter() - t0, "loader_cpu_s": t_load})
        if prof is not None:
            prof.step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_all
    ph = timer.collect()
    rows = per_step[n_warm:]
    for k in ph:
        ph[k]["gpu_span_ms"] /= n_meas
        ph[k]["cpu_s"] /= n_meas
        ph[k]["n_calls"] /= n_meas
    mm = torch.cuda.max_memory_allocated() / 2**30
    return {"mode": mode, "n_warm": n_warm, "n_meas": n_meas, "bs": args.bs,
            "precision": args.precision, "offload_activations": OFFLOAD,
            "spilled_over_vram_budget": mm > VRAM_BUDGET_GB,
            "wall_s_per_step": wall / n_meas,
            "cpu_side_s_per_step_median": med([r["cpu_step_s"] for r in rows]),
            "loader_cpu_s_median": med([r["loader_cpu_s"] for r in rows]),
            "loader_cpu_s_all": [r["loader_cpu_s"] for r in rows],
            "phases_per_step": ph,
            "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30,
            "loss_last": float(loss), "gnorm_last": float(gnorm)}


def _ka_dump(prof, n_meas):
    ka = prof.key_averages()
    markers, kernels = {}, []
    tot_dev = 0.0
    for e in ka:
        dev_t = float(getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0.0)))
        self_dev = float(getattr(e, "self_device_time_total",
                                 getattr(e, "self_cuda_time_total", 0.0)))
        # ⚠️ Kineto mirrors every record_function range onto the GPU timeline as a
        # `gpu_user_annotation` with device_type CUDA; counting those as kernels
        # double-counts their children (MEASURED: 1031 vs 793 ms/step). Exclude them.
        is_kernel = (getattr(e, "device_type", None) == torch.autograd.DeviceType.CUDA
                     and not e.key.startswith("P/") and not e.key.startswith("ProfilerStep"))
        if e.key.startswith("P/"):
            markers[e.key] = {"count_per_step": e.count / n_meas,
                              "cpu_total_ms": e.cpu_time_total / 1e3 / n_meas,
                              "cpu_self_ms": e.self_cpu_time_total / 1e3 / n_meas,
                              "dev_total_ms": dev_t / 1e3 / n_meas,
                              "dev_self_ms": self_dev / 1e3 / n_meas}
        if is_kernel:
            tot_dev += self_dev
            kernels.append({"name": e.key, "count_per_step": e.count / n_meas,
                            "dev_self_ms_per_step": self_dev / 1e3 / n_meas,
                            "avg_us": self_dev / max(e.count, 1)})
    kernels.sort(key=lambda r: -r["dev_self_ms_per_step"])
    for r in kernels:
        r["pct_of_kernel_time"] = 100.0 * r["dev_self_ms_per_step"] * n_meas * 1e3 / max(tot_dev, 1e-9)
    return markers, kernels, tot_dev / 1e3 / n_meas


def trace_stats(path: Path) -> dict:
    """GPU-busy fraction (union of kernel intervals / window), kernel-size
    histogram, and backward kernel time attributed to its forward phase via the
    autograd sequence numbers (best effort; reports 'unavailable' if absent)."""
    d = json.loads(path.read_text(encoding="utf-8"))
    ev = d.get("traceEvents", [])
    gpu = [(e["ts"], e["ts"] + e["dur"]) for e in ev
           if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    kern = [e for e in ev if e.get("cat") == "kernel"]
    steps = [e for e in ev if str(e.get("name", "")).startswith("ProfilerStep#") and "dur" in e]
    if steps:
        w0 = min(e["ts"] for e in steps); w1 = max(e["ts"] + e["dur"] for e in steps)
    else:
        allts = [e["ts"] for e in ev if "ts" in e and "dur" in e]
        w0, w1 = min(allts), max(e["ts"] + e["dur"] for e in ev if "ts" in e and "dur" in e)
    gpu.sort()
    busy, cur_s, cur_e = 0.0, None, None
    for s, e in gpu:
        s, e = max(s, w0), min(e, w1)
        if e <= s:
            continue
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    window = max(w1 - w0, 1e-9)
    durs = sorted(e["dur"] for e in kern)
    n = len(durs)
    hist = {"n_kernels": n,
            "median_us": durs[n // 2] if n else None,
            "frac_kernels_lt_10us": (sum(1 for x in durs if x < 10) / n) if n else None,
            "frac_kernels_lt_50us": (sum(1 for x in durs if x < 50) / n) if n else None,
            "frac_time_in_kernels_lt_50us": (sum(x for x in durs if x < 50) / max(sum(durs), 1e-9)) if n else None}
    out = {"window_s": window / 1e6, "gpu_busy_s": busy / 1e6, "gpu_busy_frac": busy / window,
           "n_profiler_steps": len(steps), "kernel_hist": hist}
    out["backward_attribution"] = _attribute_backward(ev, w0, w1)
    return out


def _attribute_backward(ev, w0, w1) -> dict:
    ann = {}
    for e in ev:
        if e.get("cat") == "user_annotation" and str(e.get("name", "")).startswith("P/") and "dur" in e:
            ann.setdefault(e.get("tid"), []).append((e["ts"], e["ts"] + e["dur"], e["name"]))
    if not ann:
        return {"status": "unavailable: no P/ annotations in trace"}
    for tid in ann:
        ann[tid].sort()
    starts = {tid: [x[0] for x in lst] for tid, lst in ann.items()}

    def innermost(tid, t):
        """The innermost P/ range enclosing time t on thread tid. `P/forward`
        itself wins only when no sub-marker encloses t — those kernels are the
        forward's own glue (the `_chan_std` reductions, `field.mean`, stacks,
        slices) and are reported as `P/forward` rather than dropped."""
        lst = ann.get(tid)
        if not lst:
            return None
        i = bisect.bisect_right(starts[tid], t) - 1
        best = None
        for j in range(i, max(-1, i - 64), -1):
            s, e, nm = lst[j]
            if s <= t <= e:
                if best is None or (e - s) < (best[1] - best[0]):
                    best = (s, e, nm)
        return best[2] if best else None

    seq2phase = {}
    bw_nodes = {}
    n_seq = 0
    for e in ev:
        if e.get("cat") != "cpu_op" or "dur" not in e:
            continue
        a = e.get("args", {}) or {}
        seq = a.get("Sequence number")
        if seq is None:
            continue
        n_seq += 1
        nm = str(e.get("name", ""))
        if nm.startswith("autograd::engine::evaluate_function"):
            bw_nodes.setdefault(e.get("tid"), []).append((e["ts"], e["ts"] + e["dur"], seq))
        else:
            ph = innermost(e.get("tid"), e["ts"])
            if ph and seq not in seq2phase:
                seq2phase[seq] = ph
    if n_seq == 0:
        return {"status": "unavailable: trace carries no 'Sequence number' args"}
    for tid in bw_nodes:
        bw_nodes[tid].sort()
    bw_starts = {tid: [x[0] for x in lst] for tid, lst in bw_nodes.items()}
    launches = {}
    for e in ev:
        if e.get("cat") == "cuda_runtime" and "dur" in e:
            c = (e.get("args", {}) or {}).get("correlation")
            if c is not None:
                launches[c] = (e.get("tid"), e["ts"])
    fwd_ms, bwd_ms = {}, {}
    unattributed = 0.0
    for e in ev:
        if e.get("cat") != "kernel" or "dur" not in e:
            continue
        if e["ts"] < w0 or e["ts"] > w1:
            continue
        c = (e.get("args", {}) or {}).get("correlation")
        L = launches.get(c)
        if L is None:
            unattributed += e["dur"]; continue
        tid, t = L
        lst = bw_nodes.get(tid)
        phase = None
        if lst:
            i = bisect.bisect_right(bw_starts[tid], t) - 1
            for j in range(i, max(-1, i - 8), -1):
                s, en, seq = lst[j]
                if s <= t <= en:
                    phase = seq2phase.get(seq, "bwd:unmapped-seq"); break
        if phase is not None:
            bwd_ms[phase] = bwd_ms.get(phase, 0.0) + e["dur"] / 1e3
        else:
            ph = innermost(tid, t)
            if ph is None:
                unattributed += e["dur"]
            else:
                fwd_ms[ph] = fwd_ms.get(ph, 0.0) + e["dur"] / 1e3
    return {"status": "ok", "n_forward_seq": len(seq2phase),
            "forward_kernel_ms_by_phase": fwd_ms, "backward_kernel_ms_by_phase": bwd_ms,
            "unattributed_kernel_ms": unattributed / 1e3,
            "note": "totals over all profiler steps in the trace; divide by n_profiler_steps"}


def arm_step(args, model, data, fwd_params, dev, timer, out_dir, key):
    opt = make_opt(model, args.lr, args.adapter_lr_mult)
    staged = data.batch(args.bs)
    res = {"bs": args.bs, "precision": args.precision, "wall": {}}
    for mode in args.loader_modes:
        free()
        res["wall"][mode] = run_steps(args, model, opt, data, fwd_params, dev, timer, mode,
                                      staged, args.warmup, args.measured)
        print(f"[step/{mode}] {res['wall'][mode]['wall_s_per_step']:.3f} s/step  "
              f"loader_cpu {res['wall'][mode]['loader_cpu_s_median']:.3f} s  "
              f"maxmem {res['wall'][mode]['max_mem_gb']:.2f} GB", flush=True)
        bank(out_dir, key, res)
    # --- the profiler pass (live loader) ------------------------------------ #
    # ⚠️ MEASURED 2026-09-02: ten profiled OFFLOADED steps drove the host to
    # 23.8 GB working set / 1.3 GB free (commit 62.9 of 63.4 GB) and never
    # finished — the offload doubles the event volume. `--prof-measured` bounds
    # the profiler pass independently of the wall pass.
    free()
    trace_path = out_dir / "trace.json"
    n_meas = args.prof_measured or args.measured
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=0, warmup=1, active=n_meas, repeat=1),
                 record_shapes=False, profile_memory=False, with_stack=False) as prof:
        wall = run_steps(args, model, opt, data, fwd_params, dev, timer, "live", staged,
                         1, n_meas, prof=prof)
    markers, kernels, tot_kernel_ms = _ka_dump(prof, n_meas)
    prof.export_chrome_trace(str(trace_path))
    size = trace_path.stat().st_size
    ts = trace_stats(trace_path)
    if size > 50 * 2**20:
        big = trace_path.with_name("trace.FULL-not-shipped.json")
        os.replace(trace_path, big)
        # a shippable small trace: 2 steps
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     schedule=schedule(wait=0, warmup=1, active=2, repeat=1)) as prof2:
            run_steps(args, model, opt, data, fwd_params, dev, timer, "live", staged, 1, 2,
                      prof=prof2)
        prof2.export_chrome_trace(str(trace_path))
        ts["shipped_trace_note"] = (f"full {n_meas}-step trace was {size/2**20:.1f} MB (>50 MB) "
                                    f"and is kept locally as {big.name}; trace.json is a 2-step trace")
        big.unlink(missing_ok=True)
    res["profiler"] = {"wall_under_profiler": wall, "markers_per_step_ms": markers,
                       "top_kernels": kernels[:20], "n_kernel_kinds": len(kernels),
                       "kernel_time_ms_per_step": tot_kernel_ms,
                       "trace_bytes": trace_path.stat().st_size, "trace": ts}
    bank(out_dir, key, res)
    lines = ["| # | kernel | calls/step | ms/step | avg us | % kernel time |", "|---|---|---|---|---|---|"]
    for i, k in enumerate(kernels[:20], 1):
        lines.append(f"| {i} | `{k['name'][:90]}` | {k['count_per_step']:.0f} | "
                     f"{k['dev_self_ms_per_step']:.1f} | {k['avg_us']:.1f} | {k['pct_of_kernel_time']:.1f} |")
    (out_dir / "top_kernels.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    del opt; free()
    return res


def _timed(fn, n_warm=2, n_meas=5, sync=True):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    xs = []
    for _ in range(n_meas):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        xs.append(s.elapsed_time(e))
    return {"median_ms": med(xs), "all_ms": xs}


@torch.no_grad()
def arm_sweep(args, model, data, fwd_params, dev, timer):
    timer.enabled = False
    b8 = data.batch(8)
    res = {"forward_full": [], "op_rollout": []}
    for prec in args.sweep_precisions:
        for bs in args.sweep_bs:
            sub = {k: (v[:bs] if torch.is_tensor(v) else v) for k, v in b8.items()}
            feats, actions, future, kw = to_dev(sub, dev, fwd_params)
            free(); torch.cuda.reset_peak_memory_stats()
            try:
                with precision(prec):
                    r = _timed(lambda: model(feats, actions, future_feats=future, **kw))
                row = {"precision": prec, "bs": bs, **r,
                       "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
            except OOM as e:
                row = {"precision": prec, "bs": bs, "oom": str(e).splitlines()[0][:120]}
            print(f"[sweep/forward_full] {row.get('precision')} bs{bs} {row.get('median_ms', 'OOM')}", flush=True)
            res["forward_full"].append(row)
            with precision(prec):
                field = model.encode(feats); last = model._last_state(field)
                brains = model._run_brains(field.mean(-2), kw.get("nav_cmd"))
                intent = brains["intent"]
            free(); torch.cuda.reset_peak_memory_stats()
            try:
                with precision(prec):
                    r = _timed(lambda: model.operative.rollout(last, actions, intent=intent))
                row = {"precision": prec, "bs": bs, **r,
                       "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
            except OOM as e:
                row = {"precision": prec, "bs": bs, "oom": str(e).splitlines()[0][:120]}
            print(f"[sweep/op_rollout] {row.get('precision')} bs{bs} {row.get('median_ms', 'OOM')}", flush=True)
            res["op_rollout"].append(row)
            del field, last, brains, intent; free()
    timer.enabled = True
    return res


def arm_components(args, model, data, fwd_params, dev, timer):
    import torch.nn.functional as F
    timer.enabled = False
    cfg = model.cfg
    b = data.batch(args.bs)
    feats, actions, future, kw = to_dev(b, dev, fwd_params)
    prec = args.precision
    B = feats.shape[0]
    with torch.no_grad(), precision(prec):
        field = model.encode(feats); last = model._last_state(field)
        pooled_win = field.mean(-2)
        brains = model._run_brains(pooled_win, kw.get("nav_cmd")); intent = brains["intent"]
        tgt_std = model.std(future); tgt = model.adapter(tgt_std)
        step = max(1, int(round(cfg.tac_dt / cfg.op_dt)))
        tq = torch.stack([model._tac_field(tgt[:, i]) for i in range(tgt.shape[1])], 1)[:, step - 1::step]
        sstep = max(1, int(round(cfg.str_dt / cfg.op_dt)))
        st = model.strategic.subspace(tgt.flatten(0, 1)).reshape(B, tgt.shape[1], -1)[:, sstep - 1::sstep]
        ext_t, ext_a = kw.get("str_ext_targets"), kw.get("str_ext_actions")
        st_e = None
        if ext_t is not None:
            text = model.adapter(model.std(ext_t))
            st_e = model.strategic.subspace(text.flatten(0, 1)).reshape(B, ext_t.shape[1], -1)
        tac_a = actions[:, ::step][:, :cfg.tac_steps]
        str_a = actions[:, ::sstep][:, :cfg.str_steps]
        full_a = torch.cat([str_a, ext_a], 1) if ext_a is not None else str_a
    res = {"bs": B, "precision": prec, "items": {}}

    def fwd_bwd(name, fwd, loss_fn, n_warm=1, n_meas=5):
        free(); torch.cuda.reset_peak_memory_stats()
        fw, bw = [], []
        try:
            for i in range(n_warm + n_meas):
                model.zero_grad(set_to_none=True)
                s0 = torch.cuda.Event(enable_timing=True); s1 = torch.cuda.Event(enable_timing=True)
                s2 = torch.cuda.Event(enable_timing=True)
                s0.record()
                with precision(prec):
                    out = fwd()
                s1.record()
                loss = loss_fn(out)
                loss.backward()
                s2.record(); torch.cuda.synchronize()
                if i >= n_warm:
                    fw.append(s0.elapsed_time(s1)); bw.append(s1.elapsed_time(s2))
                del out, loss
            row = {"fwd_ms": med(fw), "bwd_ms": med(bw), "fwd_all": fw, "bwd_all": bw,
                   "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        except OOM as e:
            row = {"oom": str(e).splitlines()[0][:120],
                   "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        model.zero_grad(set_to_none=True); free()
        print(f"[components] {name}: {row}", flush=True)
        res["items"][name] = row

    def fwd_only(name, fn, grad_recorded=True, n_warm=1, n_meas=5):
        free(); torch.cuda.reset_peak_memory_stats()
        try:
            ctx = nullcontext() if grad_recorded else torch.no_grad()
            with ctx, precision(prec):
                r = _timed(fn, n_warm, n_meas)
            row = {"fwd_ms": r["median_ms"], "fwd_all": r["all_ms"], "grad_recorded": grad_recorded,
                   "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        except OOM as e:
            row = {"oom": str(e).splitlines()[0][:120]}
        free()
        print(f"[components] {name}: {row}", flush=True)
        res["items"][name] = row

    # 1. encode (std + adapter on the observed window) fwd+bwd
    fwd_bwd("encode_window", lambda: model.encode(feats), lambda o: o.float().pow(2).mean())
    # 2. brains fwd+bwd
    pw = pooled_win.detach().requires_grad_(True)
    fwd_bwd("brains", lambda: model._run_brains(pw, kw.get("nav_cmd")),
            lambda o: o["intent"].float().pow(2).mean() + o["ctx"].float().pow(2).mean()
            + (o["route_logits"].float().pow(2).mean() if o.get("route_logits") is not None else 0))
    # 3. target path as live: std + adapter on the 30-step future, GRAD-RECORDED then detached
    fwd_only("target_adapter_future[grad-recorded,as-live]", lambda: model.adapter(model.std(future)), True)
    fwd_only("target_adapter_future[no_grad]", lambda: model.adapter(model.std(future)), False)
    fwd_only("target_tac_field_x30[grad-recorded,as-live]",
             lambda: torch.stack([model._tac_field(tgt[:, i]) for i in range(tgt.shape[1])], 1), True)
    fwd_only("target_str_subspace[grad-recorded,as-live]",
             lambda: model.strategic.subspace(tgt.flatten(0, 1)), True)
    if ext_t is not None:
        fwd_only("target_adapter_str_ext[grad-recorded,as-live]",
                 lambda: model.adapter(model.std(ext_t)), True)
    # 4. operative rollout fwd+bwd on a K-ladder (to_enc + mse included, as live)
    last_g = last.detach().requires_grad_(True); intent_g = intent.detach().requires_grad_(True)
    for K in args.k_ladder:
        a_k = actions[:, :K]; tgt_k = tgt_std[:, :K]
        fwd_bwd(f"op_rollout_K{K}[+to_enc+mse]",
                lambda: model.operative.rollout(last_g, a_k, intent=intent_g),
                lambda o: F.mse_loss(model.to_enc(o), tgt_k))
    # 4b. the SAME rollout fwd+bwd under torch.profiler at a K that fits without
    #     offload: GPU-busy fraction and kernel histogram of a REAL fwd+bwd, which
    #     the offloaded full step cannot give (its PCIe traffic starves the GPU).
    if args.k_profile:
        K = args.k_profile
        a_k = actions[:, :K]; tgt_k = tgt_std[:, :K]
        free()
        try:
            n_act = args.prof_measured or 3
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                         schedule=schedule(wait=0, warmup=2, active=n_act, repeat=1)) as prof:
                for _ in range(2 + n_act):
                    model.zero_grad(set_to_none=True)
                    with record_function("P/op_rollout"), precision(prec):
                        o = model.operative.rollout(last_g, a_k, intent=intent_g)
                    with record_function("P/loss_mse"):
                        l = F.mse_loss(model.to_enc(o), tgt_k)
                    with record_function("P/backward"):
                        l.backward()
                    torch.cuda.synchronize()
                    prof.step()
                    del o, l
            tp = out_dir_of(args) / f"trace_components_op_rollout_K{K}_bs{B}_{prec}.json"
            prof.export_chrome_trace(str(tp))
            markers, kernels, tot = _ka_dump(prof, n_act)
            res["profiled_op_rollout"] = {"K": K, "n_meas": n_act, "markers_per_step_ms": markers,
                                          "kernel_time_ms_per_step": tot, "top_kernels": kernels[:20],
                                          "trace": trace_stats(tp), "trace_file": tp.name,
                                          "trace_bytes": tp.stat().st_size}
            print(f"[components] profiled op_rollout K{K}: kernel {tot:.0f} ms/step, GPU busy "
                  f"{res['profiled_op_rollout']['trace']['gpu_busy_frac']*100:.1f} %", flush=True)
        except OOM as e:
            res["profiled_op_rollout"] = {"K": K, "oom": str(e).splitlines()[0][:120]}
        model.zero_grad(set_to_none=True); free()
    # 5. tactical: init field + rollout fwd+bwd
    fwd_bwd("tac_rollout_K10[+tac_field_init+mse]",
            lambda: model.tactical.rollout(model._tac_field(last_g), tac_a, intent=intent_g),
            lambda o: F.mse_loss(o, tq[:, :o.shape[1]].detach()))
    # 6. strategic subspace + rollout (in-window + ext) fwd+bwd
    def _str_fwd():
        return model.strategic.rollout(model.strategic.subspace(last_g), full_a)
    def _str_loss(o):
        l = F.mse_loss(o[:, :cfg.str_steps], st[:, :cfg.str_steps].detach())
        if st_e is not None:
            l = l + F.mse_loss(o[:, cfg.str_steps:], st_e.detach())
        return l
    fwd_bwd("str_subspace+rollout_K4[+mse]", _str_fwd, _str_loss)
    # 7. proposal + heads (tiny)
    fwd_bwd("proposal+heads", lambda: (model.proposal(pooled_win[:, -1].detach()),
                                       model.lat_head(intent_g), model.lon_head(intent_g)),
            lambda o: sum(x.float().pow(2).mean() for x in o))
    # 8. optimizer + clip alone (grads = ones)
    opt = make_opt(model, args.lr, args.adapter_lr_mult)
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    opt.step()                                     # materialise AdamW state
    r_clip = _timed(lambda: nn.utils.clip_grad_norm_(model.parameters(), args.clip))
    r_opt = _timed(lambda: opt.step())
    r_zero = _timed(lambda: [p.grad.zero_() for p in model.parameters()])
    res["items"]["clip_grad_norm"] = {"ms": r_clip["median_ms"]}
    res["items"]["adamw_step"] = {"ms": r_opt["median_ms"]}
    res["items"]["grad_zero_(inplace,ref)"] = {"ms": r_zero["median_ms"]}
    opt.zero_grad(set_to_none=True); del opt
    print(f"[components] clip {r_clip['median_ms']:.1f} ms, adamw {r_opt['median_ms']:.1f} ms", flush=True)
    res["n_params"] = sum(p.numel() for p in model.parameters())
    timer.enabled = True
    free()
    return res


@torch.no_grad()
def arm_fwdprof(args, model, data, fwd_params, dev, timer, out_dir):
    """The whole live forward (loss assembly + instruments included) under
    torch.profiler, no grad: per-marker CPU/CUDA time for EVERY forward phase.
    Exists because the full fwd+bwd step does not fit an 8 GB card and its
    activation-offloaded form exhausts a 32 GB host (MEASURED 2026-09-02)."""
    timer.enabled = False
    b = data.batch(args.bs)
    feats, actions, future, kw = to_dev(b, dev, fwd_params)
    n_act = args.prof_measured or 3
    free(); torch.cuda.reset_peak_memory_stats()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=schedule(wait=0, warmup=2, active=n_act, repeat=1)) as prof:
        for _ in range(2 + n_act):
            with record_function("P/forward"), precision(args.precision):
                out = model(feats, actions, future_feats=future, **kw)
            torch.cuda.synchronize()
            prof.step()
            del out
    tp = out_dir / f"trace_forward_nograd_bs{args.bs}_{args.precision}.json"
    prof.export_chrome_trace(str(tp))
    markers, kernels, tot = _ka_dump(prof, n_act)
    res = {"bs": args.bs, "precision": args.precision, "n_meas": n_act,
           "markers_per_step_ms": markers, "kernel_time_ms_per_step": tot,
           "top_kernels": kernels[:20], "n_kernel_kinds": len(kernels),
           "trace": trace_stats(tp), "trace_file": tp.name, "trace_bytes": tp.stat().st_size,
           "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    print(f"[fwdprof] bs{args.bs} {args.precision}: kernel {tot:.0f} ms/step, GPU busy "
          f"{res['trace']['gpu_busy_frac']*100:.1f} %, trace {tp.stat().st_size/2**20:.1f} MB", flush=True)
    timer.enabled = True
    free()
    return res


@torch.no_grad()
def arm_eig(args, model, data, fwd_params, dev, timer):
    from tanitad.eval import spectral
    timer.enabled = False
    b = data.batch(args.bs)
    feats, actions, future, kw = to_dev(b, dev, fwd_params)
    tgt_std = model.std(future); tgt = model.adapter(tgt_std)
    z = tgt.reshape(-1, tgt.shape[-1]).float()
    zc = z[:4096]
    res = {"z_rows_total": int(z.shape[0]), "eig_rows": int(zc.shape[0]), "d": int(zc.shape[1])}
    # exactly the live call: covariance_eigs (fp64) + participation_ratio, on the GPU tensor
    res["live_gpu_fp64"] = _timed(lambda: spectral.participation_ratio(spectral.covariance_eigs(zc)),
                                  n_warm=2, n_meas=10)
    res["live_call_cpu"] = _timed(lambda: spectral.participation_ratio(spectral.covariance_eigs(zc.cpu())),
                                  n_warm=1, n_meas=5)

    def fp32_variant(x):
        xc = x - x.mean(0, keepdim=True)
        cov = (xc.T @ xc) / max(x.shape[0] - 1, 1)
        e = torch.linalg.eigvalsh(cov).clamp_min(0)
        return float(e.sum() ** 2 / (e ** 2).sum().clamp_min(1e-30))
    res["fp32_variant_gpu"] = _timed(lambda: fp32_variant(zc), n_warm=2, n_meas=10)
    res["participation_value"] = {"live_fp64": spectral.participation_ratio(spectral.covariance_eigs(zc)),
                                  "fp32_variant": fp32_variant(zc)}
    # the three per-channel std instruments (refa_v1.py `_chan_std`), on the live shapes
    cfg = model.cfg
    step = max(1, int(round(cfg.tac_dt / cfg.op_dt)))
    tq = torch.stack([model._tac_field(tgt[:, i]) for i in range(tgt.shape[1])], 1)[:, step - 1::step]
    sstep = max(1, int(round(cfg.str_dt / cfg.op_dt)))
    st = model.strategic.subspace(tgt.flatten(0, 1)).reshape(tgt.shape[0], tgt.shape[1], -1)[:, sstep - 1::sstep]

    def _chan_std(x):
        return float(x.detach().float().reshape(-1, x.shape[-1]).std(dim=0).mean())
    res["chan_std_op"] = {"shape": list(tgt_std.shape), **_timed(lambda: _chan_std(tgt_std), 2, 10)}
    res["chan_std_tac"] = {"shape": list(tq.shape), **_timed(lambda: _chan_std(tq), 2, 10)}
    res["chan_std_str"] = {"shape": list(st.shape), **_timed(lambda: _chan_std(st), 2, 10)}
    res["adapter_std_logstep"] = {"note": "trainer log-step extra (every --log-every=50): model.encode(feats).std(dim=(0,1,2)).mean()",
                                  **_timed(lambda: float(model.encode(feats).std(dim=(0, 1, 2)).mean()), 1, 5)}
    res["sync_points_per_step"] = "4 host syncs inside forward: 3x _chan_std float() + participation_ratio float()"
    print(f"[eig] live fp64 GPU {res['live_gpu_fp64']['median_ms']:.1f} ms, fp32 variant "
          f"{res['fp32_variant_gpu']['median_ms']:.1f} ms, chan_std_op {res['chan_std_op']['median_ms']:.1f} ms", flush=True)
    timer.enabled = True
    free()
    return res


@torch.no_grad()
def arm_graph(args, model, dev, timer):
    timer.enabled = False
    res = {}
    cfg = model.cfg
    d_int = cfg.tactical_cfg.d_intent
    for bs in args.graph_bs:
        for prec in args.graph_precisions:
            z0 = torch.randn(bs, cfg.n_tokens, cfg.d_state, device=dev)
            a = torch.randn(bs, cfg.op_steps, cfg.a_dim, device=dev) * 0.1
            it = torch.randn(bs, d_int, device=dev)
            row = {"bs": bs, "precision": prec, "steps": cfg.op_steps}
            try:
                def eager():
                    zz = z0.clone()
                    with precision(prec):
                        for k in range(cfg.op_steps):
                            zz = model.operative.step(zz, a[:, k], intent=it)
                    return zz
                row["eager"] = _timed(eager, 2, 5)
                # capture ONE step with static buffers; replay K times, feeding the
                # output back through a copy (the rollout carries state).
                sz = z0.clone(); sa = a[:, 0].clone()
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s), precision(prec, cache=False):
                    for _ in range(3):
                        model.operative.step(sz, sa, intent=it)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with precision(prec, cache=False):
                    with torch.cuda.graph(g):
                        s_out = model.operative.step(sz, sa, intent=it)

                def replay():
                    sz.copy_(z0)
                    for k in range(cfg.op_steps):
                        sa.copy_(a[:, k]); g.replay(); sz.copy_(s_out)
                    return sz
                row["graph_replay"] = _timed(replay, 2, 5)
                with precision(prec):
                    ref = eager()
                sz.copy_(z0)
                for k in range(cfg.op_steps):
                    sa.copy_(a[:, k]); g.replay(); sz.copy_(s_out)
                row["max_abs_diff_vs_eager"] = float((sz - ref).abs().max())
                row["speedup"] = row["eager"]["median_ms"] / max(row["graph_replay"]["median_ms"], 1e-9)
                del g
            except Exception as e:                  # capture failures are a finding, not a crash
                row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            print(f"[graph] {row}", flush=True)
            res[f"bs{bs}_{prec}"] = row
            free()
    timer.enabled = True
    return res


def arm_loader(args, data, dev):
    res = {"lru": args.lru, "n_episodes": len(data.names), "n_windows": len(data),
           "join_report": getattr(data, "join_report", {})}
    bs = args.bs
    # hot: LRU warm (all episodes resident) — the dev-box state
    for _ in range(2):
        data.batch(bs)
    xs = []
    for _ in range(10):
        t = time.perf_counter(); data.batch(bs); xs.append(time.perf_counter() - t)
    res["batch_lru_hot_s"] = {"median": med(xs), "all": xs}
    # cold: LRU cleared before every batch (the live 1.4 % hit-rate state), miss count measured
    misses = {"n": 0}
    orig = data._episode

    def counting(nm):
        if nm not in data._lru:
            misses["n"] += 1
        return orig(nm)
    data._episode = counting
    xs, ms = [], []
    for _ in range(10):
        data._lru.clear(); misses["n"] = 0
        t = time.perf_counter(); data.batch(bs); xs.append(time.perf_counter() - t); ms.append(misses["n"])
    data._episode = orig
    res["batch_lru_cold_s"] = {"median": med(xs), "all": xs, "misses_per_batch": ms,
                               "note": f"only {len(data.names)} episodes here, so a batch of {bs} "
                                       f"repeats episodes; live (4713 eps, lru {args.lru}) misses ~every window"}
    # raw miss cost per episode: torch.load of the fp8 cache file + the v2ep (warm page cache)
    nm = data.names[0]
    f8 = Path(args.cache) / f"{nm}.pt"; ep = Path(args.episodes) / f"{nm}.v2ep.pt"
    xs8, xse = [], []
    for _ in range(5):
        t = time.perf_counter(); torch.load(f8, map_location="cpu", weights_only=True); xs8.append(time.perf_counter() - t)
        t = time.perf_counter(); torch.load(ep, map_location="cpu", weights_only=False); xse.append(time.perf_counter() - t)
    res["miss_cost_warm_pagecache_s"] = {"fp8_cache_file": {"bytes": f8.stat().st_size, "median": med(xs8), "all": xs8},
                                         "v2ep_file": {"bytes": ep.stat().st_size, "median": med(xse), "all": xse},
                                         "per_window_total_median": med(xs8) + med(xse),
                                         "per_batch_of_8_est_s": 8 * (med(xs8) + med(xse))}
    # cold page-cache read: untouched fp16 probe files (first read vs second read)
    cold_dir = Path(args.cold_read_dir) if args.cold_read_dir else None
    if cold_dir and cold_dir.exists():
        used = set(data.names)
        cands = [p for p in sorted(cold_dir.glob("*.pt")) if p.stem not in used and p.stem != "index"][-3:]
        rows = []
        for p in cands:
            t = time.perf_counter(); torch.load(p, map_location="cpu", weights_only=True); c = time.perf_counter() - t
            t = time.perf_counter(); torch.load(p, map_location="cpu", weights_only=True); w = time.perf_counter() - t
            rows.append({"file": p.name, "bytes": p.stat().st_size, "first_read_s": c, "second_read_s": w,
                         "first_read_MBps": p.stat().st_size / 2**20 / max(c, 1e-9)})
        res["cold_pagecache_reads"] = rows
    # slice + dequant cost for one window (fp8 -> fp32 .float() on the window and the 30-step future)
    F_, v, kap = data._episode(nm)
    t_idx = data.W - 1 + 5
    xs = []
    for _ in range(20):
        t = time.perf_counter()
        F_[t_idx - data.W + 1:t_idx + 1].float(); F_[t_idx + 1:t_idx + 1 + data.K].float()
        torch.stack([F_[t_idx + c] for c in data.ext_close]).float()
        data._kin_actions(v, kap, t_idx, data.K)
        xs.append(time.perf_counter() - t)
    res["slice_dequant_per_window_s"] = {"median": med(xs), "all": xs, "dtype_on_disk": str(F_.dtype),
                                         "per_batch_of_8_est_s": 8 * med(xs)}
    print(f"[loader] hot {res['batch_lru_hot_s']['median']:.3f}s  cold {res['batch_lru_cold_s']['median']:.3f}s  "
          f"miss/window {res['miss_cost_warm_pagecache_s']['per_window_total_median']:.3f}s  "
          f"slice+dequant/window {res['slice_dequant_per_window_s']['median']:.4f}s", flush=True)
    return res


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack-root", default=None, help="pin + assert the tanitad tree (the G: trap)")
    ap.add_argument("--cache", required=True); ap.add_argument("--episodes", required=True)
    ap.add_argument("--labels", default=None); ap.add_argument("--nav", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arms", default="fit",
                    help="comma list: fit,step,sweep,components,eig,graph,loader,fwdprof")
    ap.add_argument("--bank-suffix", default="", help="appended to the bank key of components/fwdprof")
    ap.add_argument("--bs", type=int, default=LIVE["bs"], help="profile batch (step/components/eig)")
    ap.add_argument("--lru", type=int, default=LIVE["lru"])
    ap.add_argument("--target-space", default=LIVE["target_space"])
    ap.add_argument("--detach-aux-targets", dest="detach_aux", action="store_true", default=LIVE["detach_aux"])
    ap.add_argument("--no-detach-aux-targets", dest="detach_aux", action="store_false")
    ap.add_argument("--bptt-truncate", type=int, default=LIVE["bptt_truncate"])
    ap.add_argument("--min-participation", type=float, default=LIVE["min_participation"])
    ap.add_argument("--lr", type=float, default=LIVE["lr"]); ap.add_argument("--adapter-lr-mult", type=float, default=LIVE["adapter_lr_mult"])
    ap.add_argument("--clip", type=float, default=LIVE["clip"]); ap.add_argument("--seed", type=int, default=LIVE["seed"])
    ap.add_argument("--precision", default="fp32", choices=("fp32", "tf32", "bf16"), help="for step/components")
    ap.add_argument("--warmup", type=int, default=5); ap.add_argument("--measured", type=int, default=10)
    ap.add_argument("--prof-measured", type=int, default=None,
                    help="steps in the profiler pass (default = --measured). Keep small under "
                         "--offload-activations: 10 offloaded steps exhausted a 32 GB host")
    ap.add_argument("--loader-modes", default="live,prestaged,coldlru")
    ap.add_argument("--fit-bs", default="8,4,2,1"); ap.add_argument("--fit-precisions", default="fp32,bf16")
    ap.add_argument("--sweep-bs", default="1,2,4,8"); ap.add_argument("--sweep-precisions", default="fp32,tf32,bf16")
    ap.add_argument("--k-ladder", default="5,10,15,20,30")
    ap.add_argument("--k-profile", type=int, default=15,
                    help="components arm: also profile the op-rollout fwd+bwd at this K "
                         "(0 = skip). 15 fits an 8 GB card at bs 1 fp32 without offload")
    ap.add_argument("--graph-bs", default="1,8"); ap.add_argument("--graph-precisions", default="fp32,bf16")
    ap.add_argument("--cold-read-dir", default=None, help="dir of untouched .pt files for the cold-page-cache read")
    ap.add_argument("--mem-fraction", type=float, default=0.80,
                    help="cap the caching allocator at this fraction of VRAM so an over-commit "
                         "becomes a real OOM. MEASURED 2026-09-02 on Windows/WDDM: without the cap "
                         "PyTorch 'allocated' 22.49 GiB on an 8 GiB card before failing — the "
                         "driver pages to host RAM and every timing in that state is garbage")
    ap.add_argument("--offload-activations", action="store_true",
                    help="wrap the forward in torch.autograd.graph.save_on_cpu so a full step that "
                         "does not fit VRAM still runs; ONLY the per-phase KERNEL-time fractions "
                         "are admissible from such a run (wall-clock carries the PCIe traffic)")
    ap.add_argument("--offload-pin", action="store_true",
                    help="pin the offloaded activations (fails on Windows/WDDM at ~7 GB)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    for k in ("loader_modes", "fit_precisions", "sweep_precisions", "graph_precisions"):
        setattr(args, k, [s for s in getattr(args, k).split(",") if s])
    for k in ("fit_bs", "sweep_bs", "k_ladder", "graph_bs"):
        setattr(args, k, [int(s) for s in getattr(args, k).split(",") if s])
    arms = [s for s in args.arms.split(",") if s]
    global OFFLOAD, OFFLOAD_PIN
    OFFLOAD = bool(args.offload_activations)
    OFFLOAD_PIN = bool(args.offload_pin)

    got = pin_stack(args.stack_root)
    dev = torch.device(args.device)
    if dev.type != "cuda":
        raise SystemExit("this profiler measures CUDA kernels; --device cuda")
    torch.backends.cuda.matmul.allow_tf32 = False          # the live trainer's default
    p = torch.cuda.get_device_properties(0)
    torch.cuda.set_per_process_memory_fraction(args.mem_fraction)
    global VRAM_BUDGET_GB
    VRAM_BUDGET_GB = args.mem_fraction * p.total_memory / 2**30
    stack_root = got.parent.parent
    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0],
           "gpu": p.name, "gpu_total_gb": p.total_memory / 2**30, "sm_count": p.multi_processor_count,
           "allocator_cap_fraction": args.mem_fraction, "allocator_cap_gb": VRAM_BUDGET_GB,
           "offload_activations": args.offload_activations,
           "tanitad_imported_from": str(got), "argv": sys.argv[1:], "live_launch": LIVE,
           "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
           "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
           "alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
           "snapshot": {f: git_blob_sha((stack_root / f).resolve()) for f in SNAPSHOT_FILES},
           "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    bank(args.out, "env", env)

    timer = PhaseTimer()
    tr, model, data, fwd_params = build(args, dev)
    bank(args.out, "loader_join_report", getattr(data, "join_report", {}))
    instrument(model, timer)
    print(f"model {sum(q.numel() for q in model.parameters()):,} params; arms {arms}; bs {args.bs}; "
          f"precision {args.precision}", flush=True)

    # order: the weight-mutating arms (step, components' optimizer timing) last
    if "fit" in arms:
        bank(args.out, "fit", arm_fit(args, model, data, fwd_params, dev, timer))
    if "loader" in arms:
        bank(args.out, "loader", arm_loader(args, data, dev))
    if "eig" in arms:
        bank(args.out, "eig", arm_eig(args, model, data, fwd_params, dev, timer))
    if "sweep" in arms:
        bank(args.out, "sweep", arm_sweep(args, model, data, fwd_params, dev, timer))
    if "graph" in arms:
        bank(args.out, "graph", arm_graph(args, model, dev, timer))
    if "fwdprof" in arms:
        bank(args.out, f"fwdprof_{args.precision}_bs{args.bs}{args.bank_suffix}",
             arm_fwdprof(args, model, data, fwd_params, dev, timer, args.out))
    if "step" in arms:
        key = f"step_{args.precision}_bs{args.bs}"
        bank(args.out, key, arm_step(args, model, data, fwd_params, dev, timer, args.out, key))
    if "components" in arms:
        bank(args.out, f"components_{args.precision}_bs{args.bs}{args.bank_suffix}",
             arm_components(args, model, data, fwd_params, dev, timer))
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
