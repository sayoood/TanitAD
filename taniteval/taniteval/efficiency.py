"""TanitEval — inference-efficiency panel (a DEFAULT axis of every eval).

Answers, for any registered arm, the deployment question the accuracy panels
cannot: *what does one planning step cost?* — wall-clock latency (mean / p50 /
p95 / p99 over warmed iterations), a STAGE BREAKDOWN of where the budget goes,
analytic-op FLOPs, peak GPU memory, params measured at instantiation, batched
throughput, and the 10 Hz real-time headroom.

Why a stage breakdown and not just a number: the arms compute *differently*.
  * flagship / REF-A (world-model): encode the 8-frame window -> roll the
    operative predictor forward **20 sequential steps** @10 Hz, decoding a
    metric Δpose per step. The rollout is inherently serial.
  * REF-C (anchored diffusion): encode -> ONE classifier pass over **N anchors
    in parallel** -> `diffusion_steps` truncated-denoise passes -> argmax
    select. Wide, shallow, ~3 decoder passes total.
Only a per-stage read says whether a latency gap is the encoder (shared burden)
or the decode strategy (the architectural claim).

FAIRNESS CONTRACT (the classic way to publish a fake 2x speedup is to let the
precision drift between arms):
  * `precision` is applied IDENTICALLY to every arm and RECORDED in the JSON:
      fp32   TF32 OFF for matmul AND cudnn  (strictest apples-to-apples)
      tf32   TF32 ON  for matmul AND cudnn  (A40 default-ish deployment)
      amp16  torch.autocast(float16) on top of tf32
  * `torch.cuda.synchronize()` brackets every timed region; warmup discarded.
  * batch / window / horizon are fixed across arms (batch 1 = the deployment
    case; the throughput sweep is reported separately).
  * host->device transfer and uint8->float conversion are EXCLUDED from the
    model-compute number and reported separately as `input_prep`.
  * the GPU's other-process state is sampled and stored (`gpu_state`) so a
    contaminated run is visible in the artifact rather than silently published.

REF-A caveat: its frozen DINOv2/I-JEPA encoder is EXTERNAL to the checkpoint,
so a features-in arm's `plan_step` EXCLUDES the frozen encoder forward. Flagged
as `excludes_frozen_encoder` — never compare it to a pixels-in arm unadjusted.
"""
from __future__ import annotations

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

RES = Path("/root/taniteval/results")
VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"

WINDOW = 8            # shared state window (every arm)
K_MAX = 20            # 20 steps @10 Hz = the 2 s planning horizon
DT_HZ = 10.0          # control rate the arms are trained/evaluated at
RT_BUDGET_MS = 1000.0 / DT_HZ          # 100 ms
PRECISIONS = ("fp32", "tf32", "amp16")


# ---------------------------------------------------------------------------
# environment / precision
# ---------------------------------------------------------------------------
def _set_precision(mode: str):
    """Apply `mode` globally and return (autocast_ctx_factory, recorded_flags).

    Both TF32 switches move TOGETHER — matmul-only TF32 would silently favour
    the conv-heavy arm (REF-C) over the matmul-heavy arm (flagship ViT)."""
    assert mode in PRECISIONS, f"precision {mode} not in {PRECISIONS}"
    tf32 = mode in ("tf32", "amp16")
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cudnn.benchmark = True          # identical for every arm
    if mode == "amp16":
        def ctx():
            return torch.autocast("cuda", dtype=torch.float16)
    else:
        def ctx():
            return nullcontext()
    return ctx, {
        "precision": mode,
        "autocast": "float16" if mode == "amp16" else None,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "weights_dtype": "float32",
    }


def _gpu_state(idx: int = 0) -> dict:
    """Sample the GPU so a contaminated benchmark is visible in the artifact."""
    out = {"name": None, "other_compute_procs": None, "util_pct": None,
           "mem_used_mb": None, "sm_clock_mhz": None, "exclusive": None}
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,"
             "clocks.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits", f"--id={idx}"],
            capture_output=True, text=True, timeout=20)
        f = [s.strip() for s in q.stdout.strip().split(",")]
        if len(f) >= 6:
            out.update(name=f[0], util_pct=float(f[1]), mem_used_mb=float(f[2]),
                       sm_clock_mhz=float(f[3]), temp_c=float(f[4]),
                       power_w=float(f[5]))
        p = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        procs = [ln for ln in p.stdout.strip().splitlines() if ln.strip()]
        mypid = str(subprocess.os.getpid())
        others = [ln for ln in procs if not ln.split(",")[0].strip() == mypid]
        out["other_compute_procs"] = len(others)
        out["other_compute_detail"] = others[:6]
        out["exclusive"] = (len(others) == 0)
    except Exception as e:                                   # fail loud, not fatal
        out["error"] = f"{type(e).__name__}: {str(e)[:80]}"
    return out


def _env() -> dict:
    return {"torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available()
            else None,
            "capability": list(torch.cuda.get_device_capability(0))
            if torch.cuda.is_available() else None}


# ---------------------------------------------------------------------------
# timing primitives
# ---------------------------------------------------------------------------
def _pct(xs, q):
    s = sorted(xs)
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


@torch.no_grad()
def _timeit(fn, ctx, iters=200, warmup=30, device="cuda") -> dict:
    """Per-iteration CUDA-event latency (ms). Warmup discarded; every region is
    bracketed by an explicit synchronize."""
    with ctx():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize(device)
    evs = [(torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    with ctx():
        for a, b in evs:
            a.record()
            fn()
            b.record()
    torch.cuda.synchronize(device)
    ms = [a.elapsed_time(b) for a, b in evs]
    n = len(ms)
    return {"mean_ms": round(sum(ms) / n, 4), "p50_ms": round(_pct(ms, 0.50), 4),
            "p95_ms": round(_pct(ms, 0.95), 4), "p99_ms": round(_pct(ms, 0.99), 4),
            "min_ms": round(min(ms), 4), "max_ms": round(max(ms), 4),
            "std_ms": round((sum((x - sum(ms) / n) ** 2 for x in ms) / n) ** 0.5, 4),
            "iters": n, "warmup": warmup}


@torch.no_grad()
def _peak_mem(fn, ctx, device="cuda") -> dict:
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    with ctx():
        fn()
    torch.cuda.synchronize(device)
    return {"peak_alloc_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1e6, 1),
            "weights_resident_mb": round(base / 1e6, 1),
            "activation_mb": round(
                (torch.cuda.max_memory_allocated(device) - base) / 1e6, 1)}


@torch.no_grad()
def _flops(fn, ctx) -> dict:
    """FLOPs of ONE plan step, PROFILER-DERIVED (not analytic): torch's
    FlopCounterMode instruments the dispatched aten ops. It counts the ops that
    dominate — conv, mm/addmm/bmm/baddbmm, scaled_dot_product_attention — and
    NOT elementwise/norm/activation work, so treat it as a matmul-and-conv
    lower bound (the standard convention).

    THE TRAP THIS FUNCTION FIXES: `nn.MultiheadAttention` in eval takes torch's
    fused `_native_multi_head_attention` fast path, which FlopCounterMode does
    NOT instrument — so a transformer arm's attention FLOPs silently vanish
    while a conv arm's are fully counted, understating the transformer by
    ~35 %. The COUNT therefore runs with the MHA fast path disabled (complete
    attention accounting); TIMING is untouched and keeps the fast path, because
    it is a legitimate runtime optimisation. Both facts are recorded."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception as e:
        return {"error": f"FlopCounterMode unavailable: {e}"}
    fastpath_off = False
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        fastpath_off = True
    except Exception:
        pass
    try:
        with ctx():
            fn()                                   # warm (cudnn algo choice)
        counter = FlopCounterMode(display=False)
        with counter, ctx():
            fn()
        total = counter.get_total_flops()
        counts = counter.get_flop_counts()
        by_op = {str(k): int(v) for k, v in
                 counts.get("Global", {}).items()}
        return {"total_flops": int(total), "gflops": round(total / 1e9, 3),
                "by_op_gflops": {k.split(".")[-1]: round(v / 1e9, 3)
                                 for k, v in sorted(by_op.items(),
                                                    key=lambda kv: -kv[1])},
                "mha_fastpath_disabled_for_count": fastpath_off,
                "method": "torch.utils.flop_counter.FlopCounterMode "
                          "(conv/matmul/sdpa; excludes elementwise+norm). MHA "
                          "fast path disabled during the COUNT ONLY so "
                          "attention is not silently dropped; timing keeps it."}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    finally:
        if fastpath_off:
            torch.backends.mha.set_fastpath_enabled(True)


# ---------------------------------------------------------------------------
# params — measured at instantiation
# ---------------------------------------------------------------------------
def _params(L) -> dict:
    """Every module that runs at inference, counted from the LIVE objects.

    DEDUPLICATED BY TENSOR IDENTITY: the flagship's `step_readout` IS
    `grounding.step['op']`, so a naive sum over (model, grounding,
    step_readout) double-counts it and inflates the arm by ~2 M. Shared /
    tied tensors are counted once, and the overlap is reported."""
    def cnt(m):
        return sum(p.numel() for p in m.parameters())

    model = L["model"]
    seen, tot = set(), 0

    def add(m):
        nonlocal tot
        n = 0
        for p in m.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            tot += p.numel()
            n += p.numel()
        return n

    unique_model = add(model)
    by = {n: cnt(m) for n, m in model.named_children()}
    extra, extra_unique = {}, {}
    for name in ("grounding", "step_readout"):
        if L.get(name) is not None:
            extra[name] = cnt(L[name])
            extra_unique[name] = add(L[name])
    bufs = sum(b.numel() for b in model.buffers())
    dup = sum(extra.values()) + cnt(model) - tot
    return {"total_params": tot, "total_params_m": round(tot / 1e6, 4),
            "model_params_m": round(cnt(model) / 1e6, 4),
            "buffers_m": round(bufs / 1e6, 4),
            "by_module_m": {k: round(v / 1e6, 4)
                            for k, v in sorted(by.items(), key=lambda kv: -kv[1])
                            if v},
            "aux_m": {k: round(v / 1e6, 4) for k, v in extra.items()},
            "aux_unique_m": {k: round(v / 1e6, 4)
                             for k, v in extra_unique.items()},
            "double_counted_m": round(dup / 1e6, 4),
            "unique_model_m": round(unique_model / 1e6, 4),
            "note": "measured at instantiation from live nn.Modules, "
                    "DEDUPLICATED by tensor identity (step_readout is a "
                    "submodule of grounding — counting both inflates the arm). "
                    "Parameters only; buffers — e.g. REF-C's anchor vocabulary "
                    "— counted separately."}


# ---------------------------------------------------------------------------
# per-architecture plan step + stages
# ---------------------------------------------------------------------------
def _window_inputs(entry, L, ep, device, batch):
    """One real val window, replicated to `batch`. Shapes are exactly what the
    accuracy harness feeds, so the latency describes the SCORED path."""
    from taniteval import rollout as ro
    t, last = 0, WINDOW - 1
    feats = ep.feats
    fw = torch.as_tensor(feats[t:t + WINDOW])[None]                # [1,W,...]
    if fw.dtype == torch.uint8:
        fw = fw.float().div_(255.0)
    else:
        fw = fw.float()
    fw = fw.to(device).repeat(batch, *([1] * (fw.dim() - 1))).contiguous()
    aw = ep.actions[t:t + WINDOW][None].to(device).repeat(batch, 1, 1)
    fa = ep.actions[t + WINDOW:t + WINDOW + K_MAX][None].to(device) \
        .repeat(batch, 1, 1)
    poses = ep.poses
    aw, fa = ro.append_ego(aw, fa, poses, torch.tensor([last] * batch),
                           bool(entry.get("speed_input")),
                           bool(entry.get("yaw_input")),
                           bool(entry.get("dyn_input")), device)
    v0 = poses[torch.tensor([last] * batch), 3].to(device)
    return dict(fw=fw, aw=aw, fa=fa, v0=v0)


def build_case(entry, L, ep, device="cuda", batch=1) -> dict:
    """Return {plan_step, stages{name: callable}, meta} for this arm."""
    arch = entry["arch"]
    model = L["model"]
    x = _window_inputs(entry, L, ep, device, batch)
    fw, aw, fa, v0 = x["fw"], x["aw"], x["fa"], x["v0"]
    stages, meta = {}, {"batch": batch, "window": WINDOW,
                        "input_shape": list(fw.shape)}

    if arch in ("flagship-worldmodel", "flagship-worldmodel-v2", "refa-plus"):
        from tanitad.models.metric_dynamics import rollout_decode
        sr = L["step_readout"]

        def plan_step():
            states = model.encode_window(fw)
            return rollout_decode(model.predictor, states, aw, fa, sr, K_MAX)[0]

        states0 = model.encode_window(fw)
        z0 = model.predictor(states0, aw)[1]
        stages["encode_window_%dframes" % WINDOW] = lambda: model.encode_window(fw)
        stages["encode_1frame"] = lambda: (model.encode(fw[:, -1])
                                           if fw.dim() == 5
                                           else model.encode_window(fw[:, -1:]))
        stages["rollout_k20"] = lambda: rollout_decode(
            model.predictor, states0, aw, fa, sr, K_MAX)
        stages["rollout_k1"] = lambda: rollout_decode(
            model.predictor, states0, aw, fa, sr, 1)
        stages["predictor_1call"] = lambda: model.predictor(states0, aw)
        stages["step_readout_1call"] = lambda: sr(states0[:, -1], z0)
        if getattr(model, "tactical_policy", None) is not None:
            from tanitad.models.fourbrain import run_hierarchy
            stages["hierarchy_strategic+tactical"] = lambda: run_hierarchy(
                model, states0, aw)
        meta["decode"] = (f"grounded step-readout rollout: encode {WINDOW}-frame "
                          f"window -> {K_MAX} SEQUENTIAL predictor steps @10 Hz, "
                          f"per-step metric Δpose -> SE(2) accumulate "
                          f"(intent-free operative path = the scored one)")
        meta["sequential_steps"] = K_MAX
        meta["excludes_frozen_encoder"] = (arch == "refa-plus")

    elif arch == "refc":
        cfg = model.cfg
        steps = int(cfg.decoder.diffusion_steps) \
            if entry.get("mode", "diffusion") == "diffusion" else 0
        b, w = fw.shape[:2]

        def plan_step():
            return model(fw, nav_cmd=None, v0=v0, steps=steps)

        # replicate forward() so each stage is timed on real intermediates
        if cfg.hierarchy:
            fmap_all, pooled_all = model.encoder(fw.reshape(b * w, *fw.shape[2:]))
            pooled_seq = pooled_all.reshape(b, w, -1)
            pooled = pooled_seq[:, -1]
            fmap0 = fmap_all.reshape(b, w, *fmap_all.shape[1:])[:, -1]
            ctx0 = model.strategic(pooled_seq)

            def enc_win():
                fa_, pa_ = model.encoder(fw.reshape(b * w, *fw.shape[2:]))
                return model.strategic(pa_.reshape(b, w, -1))
        else:
            fmap0, pooled = model.encoder(fw[:, -1])
            ctx0 = None

            def enc_win():
                return model.encoder(fw[:, -1])
        if cfg.graft_imagination:
            fmap_ref = model.imagination(fmap0)[0]
            stages["imagination_h15"] = lambda: model.imagination(fmap0)
        else:
            fmap_ref = fmap0
        import torch.nn.functional as _F
        nav = _F.one_hot(torch.zeros(b, dtype=torch.long, device=fw.device), 4) \
            .to(pooled.dtype)
        vv = (v0.to(pooled.dtype) / 10.0).reshape(b, 1)
        m0 = model.measurement(torch.cat([vv, nav], dim=-1))
        man0 = model.maneuver_head(pooled)
        stages["encode_window_%dframes" % WINDOW] = enc_win
        stages["encode_1frame"] = lambda: model.encoder(fw[:, -1])
        stages["aux_heads_maneuver+route"] = lambda: (model.maneuver_head(pooled),
                                                      model.route_head(pooled))
        stages["decoder_classifier_steps0"] = lambda: model.decoder(
            fmap_ref, m0, ctx=ctx0, maneuver_logits=man0, steps=0)
        stages["decoder_full_steps%d" % steps] = lambda: model.decoder(
            fmap_ref, m0, ctx=ctx0, maneuver_logits=man0, steps=steps)
        stages["law_head"] = lambda: model.law_head(torch.cat(
            [pooled, model.decoder(fmap_ref, m0, ctx=ctx0,
                                   maneuver_logits=man0, steps=0)["traj"]
             .reshape(b, -1)], dim=-1))
        meta["decode"] = (
            f"anchored diffusion: encode -> 1 classifier pass over "
            f"{cfg.anchors.n_anchors} anchors IN PARALLEL -> {steps} truncated-"
            f"denoise passes -> argmax-confidence select "
            f"({1 + steps} decoder passes total)")
        meta["n_anchors"] = int(cfg.anchors.n_anchors)
        meta["denoise_steps"] = steps
        meta["decoder_passes"] = 1 + steps
        meta["sequential_steps"] = 1 + steps
        meta["excludes_frozen_encoder"] = False

    elif arch == "refb":
        kw = {}
        if entry.get("yaw_input"):
            import refb_labels as rl
            last = torch.tensor([WINDOW - 1] * fw.shape[0])
            kw["yr0"] = (rl.wrap_to_pi(ep.poses[last, 2] - ep.poses[last - 1, 2])
                         / 0.1).to(device)

        def plan_step():
            return model(fw, nav_cmd=None, v0=v0, **kw)

        if getattr(model, "encoder", None) is not None:
            stages["encode_window_%dframes" % WINDOW] = lambda: model.encoder(
                fw.reshape(fw.shape[0] * fw.shape[1], *fw.shape[2:]))
            stages["encode_1frame"] = lambda: model.encoder(fw[:, -1])
        meta["decode"] = ("planner heads: encode window -> DIRECT per-horizon "
                          "waypoint regression (no rollout, no anchor fan)")
        meta["sequential_steps"] = 1
        meta["excludes_frozen_encoder"] = False
    else:
        raise ValueError(f"efficiency: unsupported arch {arch}")

    stages["plan_step"] = plan_step
    return {"plan_step": plan_step, "stages": stages, "meta": meta,
            "inputs": x}


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure(entry, L, ep, device="cuda", precision="fp32", batch=1,
            iters=200, warmup=30, stages=True, flops=True,
            throughput_batches=(1, 2, 4, 8, 16, 32)) -> dict:
    # Save the process-wide backend flags and restore them on the way out.
    # `measure` runs INSIDE the accuracy harness (runner.run_one); leaving TF32
    # or cudnn.benchmark flipped would silently change the numerics of the NEXT
    # arm's accuracy eval in a run-all. Efficiency measurement must not be able
    # to move an ADE.
    _saved = (torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        return _measure(entry, L, ep, device, precision, batch, iters, warmup,
                        stages, flops, throughput_batches)
    finally:
        (torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark) = _saved


@torch.no_grad()
def _measure(entry, L, ep, device, precision, batch, iters, warmup, stages,
             flops, throughput_batches) -> dict:
    ctx, prec = _set_precision(precision)
    case = build_case(entry, L, ep, device, batch)
    out = {"precision": prec, "env": _env(),
           "gpu_state_before": _gpu_state(), "meta": case["meta"],
           "params": _params(L)}

    out["plan_step"] = _timeit(case["plan_step"], ctx, iters, warmup, device)
    out["memory"] = _peak_mem(case["plan_step"], ctx, device)
    if flops:
        out["flops"] = _flops(case["plan_step"], ctx)
    if stages:
        st = {}
        for name, fn in case["stages"].items():
            if name == "plan_step":
                continue
            try:
                st[name] = _timeit(fn, ctx, max(30, iters // 2), warmup, device)
            except Exception as e:
                st[name] = {"error": f"{type(e).__name__}: {str(e)[:110]}"}
        out["stages"] = st
        out["stage_shares"] = _shares(st, out["plan_step"]["mean_ms"],
                                      case["meta"])

    # input prep (host->device + uint8->float), reported but NOT in plan_step
    fr = ep.feats[0:WINDOW]
    if hasattr(fr, "dtype") and fr.dtype == torch.uint8:
        out["input_prep"] = _timeit(
            lambda: torch.as_tensor(ep.feats[0:WINDOW])[None].to(device)
            .float().div_(255.0), ctx, 30, 5, device)

    if throughput_batches:
        out["throughput"] = _throughput(entry, L, ep, device, ctx,
                                        throughput_batches)

    # Achieved arithmetic throughput — the read that explains a latency gap that
    # runs the WRONG way vs FLOPs. A serial, launch-bound decode leaves the GPU
    # idle between tiny kernels; a wide parallel decode saturates it.
    g = (out.get("flops") or {}).get("gflops")
    if g:
        ms = out["plan_step"]["mean_ms"]
        out["compute_efficiency"] = {
            "achieved_tflops": round(g / 1e3 / (ms / 1e3), 3),
            "gflops_per_plan_step": g,
            "note": "GFLOPs / wall-clock. Low value => latency is bound by "
                    "kernel launches / serialisation, not by arithmetic; more "
                    "FLOPs at a HIGHER achieved rate can still be faster."}

    p = out["plan_step"]
    out["realtime"] = {
        "budget_ms_at_%dhz" % int(DT_HZ): RT_BUDGET_MS,
        "headroom_ms_p50": round(RT_BUDGET_MS - p["p50_ms"], 3),
        "headroom_ms_p99": round(RT_BUDGET_MS - p["p99_ms"], 3),
        "budget_used_pct_p50": round(100 * p["p50_ms"] / RT_BUDGET_MS, 2),
        "budget_used_pct_p99": round(100 * p["p99_ms"] / RT_BUDGET_MS, 2),
        "max_control_rate_hz_p50": round(1000.0 / max(p["p50_ms"], 1e-9), 1),
        "max_control_rate_hz_p99": round(1000.0 / max(p["p99_ms"], 1e-9), 1),
        "meets_10hz_p99": bool(p["p99_ms"] < RT_BUDGET_MS),
    }
    out["gpu_state_after"] = _gpu_state()
    ex = out["gpu_state_before"].get("exclusive")
    out["contamination_check"] = {
        "gpu_exclusive_before": ex,
        "gpu_exclusive_after": out["gpu_state_after"].get("exclusive"),
        "valid": bool(ex and out["gpu_state_after"].get("exclusive")),
        "note": "False => another process shared the GPU; re-run, do not "
                "publish this row.",
    }
    return out


def _shares(st, total_ms, meta):
    """Derived per-stage shares of the plan-step budget (the interesting read)."""
    g = lambda k: (st.get(k) or {}).get("mean_ms")            # noqa: E731
    enc = g(f"encode_window_{WINDOW}frames")
    sh = {}
    if enc:
        sh["encoder_pct"] = round(100 * enc / total_ms, 1)
        sh["encoder_ms"] = enc
        sh["post_encoder_ms"] = round(total_ms - enc, 4)
        sh["post_encoder_pct"] = round(100 * (total_ms - enc) / total_ms, 1)
    e1 = g("encode_1frame")
    if enc and e1:
        sh["encode_1frame_ms"] = e1
        sh["cached_window_saving_ms"] = round(enc - e1, 4)
        sh["plan_step_ms_if_encoder_cached"] = round(total_ms - enc + e1, 4)
    r20, r1 = g("rollout_k20"), g("rollout_k1")
    if r20:
        sh["rollout_ms"] = r20
        sh["rollout_pct"] = round(100 * r20 / total_ms, 1)
    if r20 and r1:
        sh["per_rollout_step_ms"] = round((r20 - r1) / (K_MAX - 1), 4)
        sh["serial_rollout_note"] = (
            f"{K_MAX} sequential steps; marginal cost "
            f"{round((r20 - r1) / (K_MAX - 1), 4)} ms/step")
    d0 = g("decoder_classifier_steps0")
    dn = next((g(k) for k in st if k.startswith("decoder_full_steps")), None)
    if d0 is not None:
        sh["decoder_classifier_ms"] = d0
        sh["decoder_classifier_pct"] = round(100 * d0 / total_ms, 1)
    if d0 is not None and dn is not None:
        sh["denoise_ms"] = round(dn - d0, 4)
        sh["denoise_pct"] = round(100 * (dn - d0) / total_ms, 1)
        sh["decoder_total_ms"] = dn
        sh["decoder_total_pct"] = round(100 * dn / total_ms, 1)
        n = meta.get("denoise_steps") or 0
        if n:
            sh["per_denoise_pass_ms"] = round((dn - d0) / n, 4)
    im = g("imagination_h15")
    if im:
        sh["imagination_ms"] = im
        sh["imagination_pct"] = round(100 * im / total_ms, 1)
    hi = g("hierarchy_strategic+tactical")
    if hi:
        sh["hierarchy_if_run_ms"] = hi
        sh["hierarchy_note"] = ("NOT in the scored plan step — the operative "
                                "eval path is intent-free; this is what turning "
                                "the tactical+strategic brains on would add "
                                "(before cadence amortisation)")
    # Honesty check: isolated stage timings need NOT sum to the end-to-end. A
    # launch-bound stage measured alone starves the GPU, while inside the full
    # step the CPU runs ahead during the encoder's big kernels and hides part of
    # the launch cost. A sum ABOVE 100 % is therefore evidence of launch-bound
    # serialisation, not a bug — surface it instead of hiding it.
    parts = [v for k, v in ((k, g(k)) for k in
                            (f"encode_window_{WINDOW}frames", "rollout_k20",
                             "decoder_full_steps2", "imagination_h15",
                             "law_head", "aux_heads_maneuver+route"))
             if v is not None]
    if parts and total_ms:
        s = sum(parts)
        sh["isolated_stage_sum_ms"] = round(s, 4)
        sh["isolated_stage_sum_pct_of_plan_step"] = round(100 * s / total_ms, 1)
        sh["stage_sum_note"] = (
            ">100% => the stages overlap CPU-launch with GPU-execute inside the "
            "full step (launch-bound); <100% => untimed glue. Treat shares as "
            "attribution, not an exact partition.")
    return sh


@torch.no_grad()
def _throughput(entry, L, ep, device, ctx, batches) -> dict:
    rows = {}
    best = None
    for b in batches:
        try:
            case = build_case(entry, L, ep, device, b)
            t = _timeit(case["plan_step"], ctx, 30, 8, device)
            wps = round(b * 1000.0 / t["mean_ms"], 1)
            rows[str(b)] = {"mean_ms": t["mean_ms"], "p50_ms": t["p50_ms"],
                            "windows_per_s": wps}
            if best is None or wps > best[1]:
                best = (b, wps)
            del case
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            rows[str(b)] = {"error": "OOM"}
            torch.cuda.empty_cache()
            break
        except Exception as e:
            rows[str(b)] = {"error": f"{type(e).__name__}: {str(e)[:90]}"}
            torch.cuda.empty_cache()
            break
    return {"by_batch": rows,
            "best_windows_per_s": best[1] if best else None,
            "best_batch": best[0] if best else None}


# ===========================================================================
# INFERENCE LEVERS — turning PROJECTED optimisations into MEASURED ones
# ===========================================================================
"""The baseline panel above answers *what does one tick cost?*. This section
answers the follow-up the deployment question actually needs: *what does one
tick cost AFTER the obvious optimisations, and does the answer still decode the
same trajectory?*

The flagship diagnosis from the baseline run is that the tick is
LAUNCH-BOUND, not arithmetic-bound (3.7-4.3 achieved TFLOPs on an A40 whose
fp32 peak is ~37; `amp16` is SLOWER than `tf32`). That diagnosis picks the
levers:

  L1  CUDA-GRAPH THE ROLLOUT. The 20-step rollout is 20 SEQUENTIAL batch-1
      predictor forwards — the maximally launch-bound shape. Three captures are
      measured separately because they remove different costs:
        `graph_step`      one graph per step, replayed 20x  -> removes per-kernel
                          launch overhead, KEEPS 19 CPU round-trips;
        `graph_rollout`   ONE graph spanning all 20 steps   -> removes both;
        `graph_fulltick`  ONE graph spanning encode+rollout -> removes both, plus
                          the encoder's launches.
      The gap `graph_rollout` - `graph_step` is exactly the CPU round-trip cost.
      `torch.compile` is measured as the second path to the same goal
      (`reduce-overhead` = inductor+cudagraphs, and the Triton-free
      `backend="cudagraphs"`), because our own 2026-07-18 data says manual
      capture wins on Windows and the pod is the only place to test Linux.

  L2  ENCODER WINDOW CACHE. Every tick re-encodes 8 frames; 7 were encoded on
      previous ticks. The baseline JSON only PROJECTS the saving from stage
      timings (`plan_step_ms_if_encoder_cached`). Here the rolling cache is
      implemented and the real end-to-end tick is timed INCLUDING the cache
      bookkeeping — and the trajectory is compared, because per-frame encoding
      (batch 1) is not bit-guaranteed to equal windowed encoding (batch 8).

  L3  TRUE fp16 WEIGHTS (`model.half()`), not autocast. Autocast pays a cast on
      every op and measured SLOWER than tf32 here; fp16 weights pay the cast
      once, offline.

  L4  COMPOSITION — the levers measured TOGETHER, because "additive" is a claim,
      not a fact.

  L5  ROLLOUT-LENGTH SENSITIVITY — latency at k=20/10/5. LATENCY ONLY: a shorter
      rollout changes what the model predicts, and that accuracy question needs
      the canonical harness, not this file.

EVERY lever ships an ACCURACY row beside its speed row (`equivalence`): the
decoded trajectory is compared against the eager reference on the same real val
windows, and ADE@2s is recomputed for both. A fast wrong answer is worthless,
so a lever with no equivalence block must not be published."""


@torch.no_grad()
def _peak_mem_live(fn, ctx, device="cuda") -> dict:
    """Peak-memory probe that is SAFE while CUDA graphs are alive.

    `_peak_mem` calls `torch.cuda.empty_cache()`, which must not be done while
    captured graphs hold private memory pools. This variant only resets the
    peak counters. NOTE the read is different in kind for a graphed variant: a
    graph replay allocates NOTHING (it reuses its captured pool), so its
    `activation_mb` is ~0 while `resident_mb` carries the permanently-held
    graph pool. Both are reported; neither alone is the deployment number."""
    torch.cuda.synchronize(device)
    base = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    with ctx():
        fn()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return {"resident_mb": round(base / 1e6, 1),
            "peak_alloc_mb": round(peak / 1e6, 1),
            "activation_mb": round((peak - base) / 1e6, 1),
            "reserved_mb": round(torch.cuda.memory_reserved(device) / 1e6, 1),
            "note": "resident_mb includes every model/graph pool ALIVE in this "
                    "process (the sweep holds several at once) — it is not the "
                    "footprint of a single-variant deployment."}


def _tmap(o, f):
    """Map `f` over every tensor in a (nested) tuple/list/dict."""
    if torch.is_tensor(o):
        return f(o)
    if isinstance(o, (list, tuple)):
        return type(o)(_tmap(x, f) for x in o)
    if isinstance(o, dict):
        return {k: _tmap(v, f) for k, v in o.items()}
    return o


class GraphedFn:
    """Manual `torch.cuda.CUDAGraph` capture of a STATIC-SHAPE callable.

    Records the kernel-launch sequence once and replays it with a single launch
    — the matching lever for a launch-bound pass. `run()` INCLUDES the input
    `copy_`s, i.e. it is the deployed cost, not a replay-only number.

    Two capture traps this class handles, both of which produce a *plausible
    wrong number* rather than a crash:
      * capture must happen on a side stream after a real warmup, or cuDNN/cuBLAS
        autotuning and lazy allocations get baked into the graph;
      * the outputs live in the graph's private pool and are OVERWRITTEN by the
        next replay — anything kept must be cloned, which is why `run()` returns
        the static tensors and the callers clone explicitly.
    """

    def __init__(self, fn, inputs: dict, warmup: int = 5):
        self.fn = fn
        self.static_in = {k: v.clone() for k, v in inputs.items()}
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(warmup):
                fn(**self.static_in)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self.static_out = fn(**self.static_in)

    def run(self, **inputs):
        for k, v in inputs.items():
            self.static_in[k].copy_(v)
        self.graph.replay()
        return self.static_out


class RollingStateCache:
    """L2 — rolling window of ENCODER STATES so a tick encodes ONE frame.

    The deployment fact the baseline ignores: at 10 Hz, 7 of the 8 frames in
    this tick's window were already encoded on earlier ticks. `push` performs
    the shift, so timing a tick that calls `push` includes the bookkeeping —
    the number is an end-to-end measurement, not an arithmetic projection.
    """

    def __init__(self, states):                      # [B, W, S]
        self.buf = states.clone()

    def push(self, s_new):                           # [B, S] -> [B, W, S]
        self.buf = torch.cat([self.buf[:, 1:], s_new.unsqueeze(1)], dim=1)
        return self.buf

    def fill(self, states):
        self.buf = states.clone()
        return self.buf


class Lever:
    """One measurable inference variant.

    `tick()`  — the timed deployment tick (same protocol as `plan_step`).
    `traj(fw, aw, fa)` — decode ONE real window to waypoints [B, k, 2], float32
                          and CLONED (graph outputs are otherwise overwritten),
                          so speed always ships with accuracy.
    """

    def __init__(self, name, tick, traj, meta):
        self.name, self.tick, self.traj, self.meta = name, tick, traj, meta


def _wm_parts(entry, L):
    arch = entry.get("arch")
    if arch not in ("flagship-worldmodel", "flagship-worldmodel-v2", "refa-plus"):
        raise ValueError(
            f"inference levers target the world-model arms whose tick is a "
            f"SERIAL rollout; arch {arch!r} has no such rollout")
    return L["model"], L["step_readout"]


LEVER_ORDER = ("eager", "graph_step", "graph_rollout", "graph_fulltick",
               "compile_rollout", "compile_cudagraphs",
               "enc_cache", "enc_cache_graph",
               "drop_horizons", "drop_horizons_graph", "autocast16_eager",
               "fp16_eager", "fp16_fp32acc", "fp16_graph_rollout",
               "fp16_enc_cache_graph", "all_levers", "fan_shared_encoder")


def prune_horizons(predictor, keep=(1,)):
    """L7 — return a copy of `predictor` that computes ONLY the horizon heads
    the rollout consumes.

    `PredictorConfig.horizons = (1, 2, 4)` and `OperativePredictor.forward`
    evaluates ALL THREE heads every call, but `rollout_decode` reads only
    ``[1]`` — so 2 of 3 state-dim readout matmuls are computed and discarded,
    20x per tick. Pure inference-path change: the kept head's weights are the
    same tensors, so the output must be bit-identical (equivalence-checked).
    Returns a DEEP COPY so the measured baseline predictor is never mutated."""
    import copy as _copy
    p = _copy.deepcopy(predictor)
    from torch import nn as _nn
    p.heads = _nn.ModuleDict({str(kk): p.heads[str(kk)] for kk in keep})
    p.cfg = _copy.copy(p.cfg)
    p.cfg.horizons = tuple(keep)
    return p


@torch.no_grad()
def build_levers(entry, L, ep, device="cuda", batch=1, k=K_MAX, want=None
                 ) -> tuple[dict, dict]:
    """Build every inference variant on ONE real val window.

    Returns `(levers, build_errors)`. A variant that fails to build (capture
    error, missing Triton, OOM) is recorded in `build_errors` and simply absent
    from `levers` — FAIL LOUD, never silently substitute the eager path."""
    import copy as _copy

    from tanitad.models.metric_dynamics import (accumulate_se2,  # noqa: E402
                                                rollout_decode,
                                                rollout_transitions)
    model, sr = _wm_parts(entry, L)
    want = set(want) if want else set(LEVER_ORDER)
    x = _window_inputs(entry, L, ep, device, batch)
    fw, aw, fa = x["fw"], x["aw"], x["fa"]
    levers: dict = {}
    errs: dict = {}
    # Per-stage diagnostics. The end-to-end ticks say WHETHER a lever wins; these
    # say WHERE — specifically whether the 2026-07-18 orthogonality claim holds
    # (encoder compute-bound => precision; rollout launch-bound => graph).
    stages: dict = {}

    def _enc1(m, f):
        """Encode the LAST frame of a window only (the cached tick's encode)."""
        return (m.encode(f[:, -1]) if f.dim() == 5
                else m.encode_window(f[:, -1:])[:, 0])

    def _enc_each(m, f):
        """Encode every frame of the window SEPARATELY (batch 1 each) — the
        state a rolling cache actually holds. Compare against `encode_window`
        (batch W) to expose any batch-dependent numeric drift."""
        return torch.stack([_enc1(m, f[:, i:i + 1]) for i in range(f.shape[1])],
                           dim=1)

    def _add(name, tick, traj, meta):
        if name in want:
            levers[name] = Lever(name, tick, traj, meta)

    def _try(name, fn, gate=None):
        """Build a variant; a build failure is RECORDED, never silently swapped
        for the eager path (that would publish the baseline as a speedup)."""
        if (gate if gate is not None else name) not in want:
            return
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            errs[name] = f"{type(e).__name__}: {str(e)[:200]}"

    # ---------------- reference: the published plan_step -------------------
    def _eager_tick():
        st = model.encode_window(fw)
        return rollout_decode(model.predictor, st, aw, fa, sr, k)[0]

    def _eager_traj(f, a, fu):
        st = model.encode_window(f)
        return rollout_decode(model.predictor, st, a, fu, sr, k)[0].float()

    _add("eager", _eager_tick, _eager_traj,
         {"lever": "—", "desc": "baseline: encode 8-frame window -> 20 eager "
                                "sequential predictor steps -> SE(2)",
          "weights_dtype": "float32"})

    states0 = model.encode_window(fw)
    stages[f"fp32_encode_window_{WINDOW}f"] = lambda: model.encode_window(fw)
    stages["fp32_encode_1frame"] = lambda: _enc1(model, fw)
    stages[f"fp32_rollout_k{k}"] = lambda: rollout_decode(
        model.predictor, states0, aw, fa, sr, k)
    stages["fp32_predictor_1call"] = lambda: model.predictor(states0, aw)

    # ---------------- L1a: one graph per step, replayed k times ------------
    def _mk_graph_step():
        def _step(states, actions):
            z = model.predictor(states, actions)[1]
            return z, sr(states[:, -1], z)

        g = GraphedFn(_step, {"states": states0, "actions": aw})

        def _roll(st, a, fu):
            win_s, win_a = st, a
            # preallocated Δpose buffer: the per-step graph's output is
            # overwritten by the next replay, so it MUST be copied out — that
            # copy is a real cost of this variant and is timed.
            dp = torch.empty(st.shape[0], k, 3, device=st.device, dtype=st.dtype)
            for j in range(k):
                z, d = g.run(states=win_s, actions=win_a)
                dp[:, j].copy_(d)
                if j < k - 1:
                    win_s = torch.cat([win_s[:, 1:], z.unsqueeze(1)], dim=1)
                    win_a = torch.cat([win_a[:, 1:], fu[:, j].unsqueeze(1)],
                                      dim=1)
            return accumulate_se2(dp)

        _add("graph_step",
             lambda: _roll(model.encode_window(fw), aw, fa),
             lambda f, a, fu: _roll(model.encode_window(f), a, fu).float(),
             {"lever": "L1a", "desc": f"eager encode + {k} replays of a ONE-STEP "
                                      f"CUDA graph (removes kernel-launch cost, "
                                      f"keeps {k - 1} CPU round-trips)",
              "weights_dtype": "float32"})

    _try("graph_step", _mk_graph_step)

    # ---------------- L1b: ONE graph over the whole k-step rollout ---------
    def _roll_fn(states, actions, future_actions):
        return rollout_decode(model.predictor, states, actions,
                              future_actions, sr, k)

    graph_roll = {}

    def _mk_graph_rollout():
        g = GraphedFn(_roll_fn, {"states": states0, "actions": aw,
                                 "future_actions": fa})
        graph_roll["g"] = g

        def _tick():
            st = model.encode_window(fw)
            return g.run(states=st, actions=aw, future_actions=fa)[0]

        def _traj(f, a, fu):
            st = model.encode_window(f)
            return g.run(states=st, actions=a,
                         future_actions=fu)[0].clone().float()

        _add("graph_rollout", _tick, _traj,
             {"lever": "L1b", "desc": f"eager encode + ONE CUDA graph spanning "
                                      f"all {k} rollout steps (removes launch "
                                      f"cost AND the CPU round-trips)",
              "weights_dtype": "float32"})

    _try("graph_rollout", _mk_graph_rollout)

    # ---------------- L1c: ONE graph over encode + rollout -----------------
    def _mk_graph_fulltick():
        def _full(frames, actions, future_actions):
            st = model.encode_window(frames)
            return rollout_decode(model.predictor, st, actions,
                                  future_actions, sr, k)

        g = GraphedFn(_full, {"frames": fw, "actions": aw,
                              "future_actions": fa})
        _add("graph_fulltick",
             lambda: g.run(frames=fw, actions=aw, future_actions=fa)[0],
             lambda f, a, fu: g.run(frames=f, actions=a,
                                    future_actions=fu)[0].clone().float(),
             {"lever": "L1c", "desc": "the WHOLE tick (encode 8 frames + "
                                      f"{k}-step rollout) as ONE CUDA graph",
              "weights_dtype": "float32"})

    _try("graph_fulltick", _mk_graph_fulltick)

    # ---------------- L1d/L1e: torch.compile as the second path ------------
    def _mk_compile(name, kwargs, lever, desc):
        def _mk():
            c = torch.compile(_roll_fn, **kwargs)
            for _ in range(6):                       # trace + autotune + warm
                c(states0, aw, fa)
            torch.cuda.synchronize()

            def _tick():
                st = model.encode_window(fw)
                return c(st, aw, fa)[0]

            def _traj(f, a, fu):
                st = model.encode_window(f)
                return c(st, a, fu)[0].clone().float()

            _add(name, _tick, _traj,
                 {"lever": lever, "desc": desc, "weights_dtype": "float32"})
        return _mk

    _try("compile_rollout", _mk_compile(
        "compile_rollout", {"mode": "reduce-overhead"}, "L1d",
        "eager encode + torch.compile(mode='reduce-overhead') rollout "
        "(inductor + cudagraph trees; needs Triton)"))
    _try("compile_cudagraphs", _mk_compile(
        "compile_cudagraphs", {"backend": "cudagraphs"}, "L1e",
        "eager encode + torch.compile(backend='cudagraphs') rollout "
        "(Triton-free; the route that was 20x SLOWER on the Windows dev box)"))

    # ---------------- L2: rolling encoder-state cache ----------------------
    def _mk_enc_cache():
        cache = RollingStateCache(states0)

        def _tick():
            st = cache.push(_enc1(model, fw))
            return rollout_decode(model.predictor, st, aw, fa, sr, k)[0]

        def _traj(f, a, fu):
            st = cache.fill(_enc_each(model, f))
            return rollout_decode(model.predictor, st, a, fu, sr, k)[0].float()

        _add("enc_cache", _tick, _traj,
             {"lever": "L2", "desc": "encode ONE frame + rolling state cache "
                                     "(bookkeeping included) + eager rollout",
              "weights_dtype": "float32"})

    _try("enc_cache", _mk_enc_cache)

    # ---------------- L4: cache + graph composed ---------------------------
    def _mk_enc_cache_graph():
        g = graph_roll.get("g")
        if g is None:
            raise RuntimeError("graph_rollout did not build; cannot compose")
        cache = RollingStateCache(states0)

        def _tick():
            st = cache.push(_enc1(model, fw))
            return g.run(states=st, actions=aw, future_actions=fa)[0]

        def _traj(f, a, fu):
            st = cache.fill(_enc_each(model, f))
            return g.run(states=st, actions=a,
                         future_actions=fu)[0].clone().float()

        _add("enc_cache_graph", _tick, _traj,
             {"lever": "L4 = L1b+L2", "desc": "cached 1-frame encode + "
                                              "whole-rollout CUDA graph",
              "weights_dtype": "float32"})

    _try("enc_cache_graph", _mk_enc_cache_graph)

    # ---------------- L7: drop the unused multi-horizon heads --------------
    def _mk_drop_horizons():
        pruned = prune_horizons(model.predictor, keep=(1,))
        stages[f"fp32_rollout_k{k}_pruned"] = lambda: rollout_decode(
            pruned, states0, aw, fa, sr, k)

        def _tick():
            st = model.encode_window(fw)
            return rollout_decode(pruned, st, aw, fa, sr, k)[0]

        _add("drop_horizons", _tick,
             lambda f, a, fu: rollout_decode(
                 pruned, model.encode_window(f), a, fu, sr, k)[0].float(),
             {"lever": "L7", "desc": "eager, predictor evaluates ONLY the k=1 "
                                     "head the rollout consumes (2 of 3 heads "
                                     f"were computed and discarded {k}x/tick)",
              "weights_dtype": "float32"})

        def _mk_dh_graph():
            g = GraphedFn(lambda states, actions, future_actions: rollout_decode(
                pruned, states, actions, future_actions, sr, k),
                {"states": states0, "actions": aw, "future_actions": fa})
            _add("drop_horizons_graph",
                 lambda: g.run(states=model.encode_window(fw), actions=aw,
                               future_actions=fa)[0],
                 lambda f, a, fu: g.run(states=model.encode_window(f),
                                        actions=a, future_actions=fu
                                        )[0].clone().float(),
                 {"lever": "L7+L1b", "desc": "pruned heads + whole-rollout "
                                             "CUDA graph",
                  "weights_dtype": "float32"})
        _try("drop_horizons_graph", _mk_dh_graph)

    _try("drop_horizons", _mk_drop_horizons)

    # -------- L3 control: AUTOCAST fp16, in the same block as fp16 weights ---
    # The baseline panel measured `amp16` (autocast) SLOWER than `tf32` — but in a
    # separate block, so "autocast vs fp16 weights" was never a within-block
    # comparison. This lever carries its own autocast context, so the two fp16
    # strategies are timed back to back on the same warm GPU.
    def _mk_autocast():
        def _tick():
            with torch.autocast("cuda", dtype=torch.float16):
                st = model.encode_window(fw)
                return rollout_decode(model.predictor, st, aw, fa, sr, k)[0]

        def _traj(f, a, fu):
            with torch.autocast("cuda", dtype=torch.float16):
                st = model.encode_window(f)
                return rollout_decode(model.predictor, st, a, fu,
                                      sr, k)[0].float()

        _add("autocast16_eager", _tick, _traj,
             {"lever": "L3-control", "desc": "torch.autocast(float16) over the "
                                             "fp32-weight eager tick (per-op "
                                             "casts, weights stay fp32)",
              "weights_dtype": "float32", "autocast": "float16"})

    _try("autocast16_eager", _mk_autocast)

    # ---------------- L3/L4: TRUE fp16 weights (not autocast) --------------
    fp16_wanted = {"fp16_eager", "fp16_fp32acc", "fp16_graph_rollout",
                   "fp16_enc_cache_graph", "all_levers"} & want
    if fp16_wanted:
        def _mk_fp16():
            m16 = _copy.deepcopy(model).half().eval()
            sr16 = _copy.deepcopy(sr).half().eval()
            f16, a16, fu16 = fw.half(), aw.half(), fa.half()
            st16_0 = m16.encode_window(f16)
            stages[f"fp16_encode_window_{WINDOW}f"] = \
                lambda: m16.encode_window(f16)
            stages["fp16_encode_1frame"] = lambda: _enc1(m16, f16)
            stages[f"fp16_rollout_k{k}"] = lambda: rollout_decode(
                m16.predictor, st16_0, a16, fu16, sr16, k)
            stages["fp16_predictor_1call"] = lambda: m16.predictor(st16_0, a16)

            def _h(t):
                return t.half()

            _add("fp16_eager",
                 lambda: rollout_decode(m16.predictor,
                                        m16.encode_window(f16), a16, fu16,
                                        sr16, k)[0],
                 lambda f, a, fu: rollout_decode(
                     m16.predictor, m16.encode_window(_h(f)), _h(a), _h(fu),
                     sr16, k)[0].float(),
                 {"lever": "L3", "desc": "model.half() fp16 WEIGHTS end-to-end "
                                         "(incl. the SE(2) accumulate)",
                  "weights_dtype": "float16"})

            # fp16 compute, fp32 SE(2) accumulate. The dead-reckoning integrates
            # 20 Δposes up to ~30 m, where fp16's spacing is ~0.03 m — so if the
            # fp16 waypoint shift is large this variant says whether it came
            # from the NETWORK or merely from the (free) accumulator.
            def _roll16_acc32(m, s, st, a, fu):
                trans = rollout_transitions(m.predictor, st, a, fu, k)
                dp = torch.stack([s(t0, t1) for t0, t1 in trans], dim=1).float()
                return accumulate_se2(dp)

            _add("fp16_fp32acc",
                 lambda: _roll16_acc32(m16, sr16, m16.encode_window(f16),
                                       a16, fu16),
                 lambda f, a, fu: _roll16_acc32(
                     m16, sr16, m16.encode_window(_h(f)), _h(a), _h(fu)),
                 {"lever": "L3'", "desc": "fp16 weights, fp32 SE(2) accumulate "
                                          "(isolates accumulator rounding from "
                                          "network error)",
                  "weights_dtype": "float16"})

            g16 = GraphedFn(lambda states, actions, future_actions:
                            rollout_decode(m16.predictor, states, actions,
                                           future_actions, sr16, k),
                            {"states": st16_0, "actions": a16,
                             "future_actions": fu16})

            def _t16g():
                st = m16.encode_window(f16)
                return g16.run(states=st, actions=a16,
                               future_actions=fu16)[0]

            _add("fp16_graph_rollout", _t16g,
                 lambda f, a, fu: g16.run(
                     states=m16.encode_window(_h(f)), actions=_h(a),
                     future_actions=_h(fu))[0].clone().float(),
                 {"lever": "L4 = L1b+L3", "desc": "fp16 weights + whole-rollout "
                                                  "CUDA graph",
                  "weights_dtype": "float16"})

            c16 = RollingStateCache(st16_0)

            def _t16cg():
                st = c16.push(_enc1(m16, f16))
                return g16.run(states=st, actions=a16,
                               future_actions=fu16)[0]

            def _tr16cg(f, a, fu):
                st = c16.fill(_enc_each(m16, _h(f)))
                return g16.run(states=st, actions=_h(a),
                               future_actions=_h(fu))[0].clone().float()

            _add("fp16_enc_cache_graph", _t16cg, _tr16cg,
                 {"lever": "L4 = L1b+L2+L3",
                  "desc": "ALL THREE composed: fp16 weights + cached 1-frame "
                          "encode + whole-rollout CUDA graph",
                  "weights_dtype": "float16"})

            # everything at once — the number the 10 Hz verdict is read off
            pruned16 = prune_horizons(m16.predictor, keep=(1,))
            gall = GraphedFn(lambda states, actions, future_actions:
                             rollout_decode(pruned16, states, actions,
                                            future_actions, sr16, k),
                             {"states": st16_0, "actions": a16,
                              "future_actions": fu16})
            call = RollingStateCache(st16_0)

            _add("all_levers",
                 lambda: gall.run(states=call.push(_enc1(m16, f16)),
                                  actions=a16, future_actions=fu16)[0],
                 lambda f, a, fu: gall.run(
                     states=call.fill(_enc_each(m16, _h(f))), actions=_h(a),
                     future_actions=_h(fu))[0].clone().float(),
                 {"lever": "L1b+L2+L3+L7",
                  "desc": "EVERY lever composed: fp16 weights + pruned horizon "
                          "heads + cached 1-frame encode + whole-rollout graph",
                  "weights_dtype": "float16"})

            # THE CEM / imagine-and-select tick. A K-candidate plan fan shares
            # ONE observation history — only the ACTION sequence differs — so the
            # encoder runs ONCE at batch 1 and only the ROLLOUT fans out to K.
            # `all_levers` at batch K re-encodes K windows and therefore
            # OVER-COUNTS the encoder K-fold; this is the honest planner tick.
            c1 = RollingStateCache(st16_0[:1])

            def _fan_tick():
                s = c1.push(_enc1(m16, f16[:1]))          # [1, W, S] — encode ONCE
                return gall.run(states=s.expand(batch, -1, -1),
                                actions=a16, future_actions=fu16)[0]

            def _fan_traj(f, a, fu):
                s = c1.fill(_enc_each(m16, _h(f[:1])))
                return gall.run(states=s.expand(f.shape[0], -1, -1),
                                actions=_h(a), future_actions=_h(fu)
                                )[0].clone().float()

            _add("fan_shared_encoder", _fan_tick, _fan_traj,
                 {"lever": "L1b+L2+L3+L7 @ fan",
                  "desc": "planner tick: encode the shared observation ONCE at "
                          "batch 1, broadcast the state, fan ONLY the rollout to "
                          "K candidates (identical to all_levers at K=1)",
                  "weights_dtype": "float16"})

        _try("fp16_weights_group", _mk_fp16, gate=sorted(fp16_wanted)[0])

    # the graphed rollout as a STAGE, so the encoder/rollout attribution can be
    # read under the graph too (not just end-to-end)
    if "g" in graph_roll:
        stages[f"graph_rollout_k{k}"] = lambda: graph_roll["g"].run(
            states=states0, actions=aw, future_actions=fa)

    return levers, errs, stages


# ---------------------------------------------------------------------------
# equivalence — every speed row ships an accuracy row
# ---------------------------------------------------------------------------
@torch.no_grad()
def equiv_windows(entry, L, eps, device="cuda", n=32, stride=8, k=K_MAX):
    """Real val windows + GT waypoints for the numerical-equivalence check.

    Same construction as `taniteval.rollout.collect` (canonical ego channels,
    canonical WP_STEPS) so the ADE recomputed here is the SAME quantity the
    accuracy harness reports — just over a small window subset."""
    from driving_diagnostic import WP_STEPS, gt_ego_waypoints
    from taniteval import rollout as ro
    out = []
    for e in eps:
        feats = e.feats
        T = min(feats.shape[0], e.actions.shape[0], e.poses.shape[0])
        for t in range(0, max(0, T - WINDOW - k), stride):
            if len(out) >= n:
                break
            last = torch.tensor([t + WINDOW - 1])
            fw = torch.as_tensor(feats[t:t + WINDOW])[None]
            fw = (fw.float().div_(255.0) if fw.dtype == torch.uint8
                  else fw.float()).to(device)
            aw = e.actions[t:t + WINDOW][None].to(device)
            fa = e.actions[t + WINDOW:t + WINDOW + k][None].to(device)
            aw, fa = ro.append_ego(aw, fa, e.poses, last,
                                   bool(entry.get("speed_input")),
                                   bool(entry.get("yaw_input")),
                                   bool(entry.get("dyn_input")), device)
            out.append({"fw": fw, "aw": aw, "fa": fa,
                        "gt": gt_ego_waypoints(e.poses, last).float()})
        if len(out) >= n:
            break
    return out, list(WP_STEPS)


@torch.no_grad()
def _trajs(lever, wins):
    return torch.cat([lever.traj(w["fw"], w["aw"], w["fa"]).cpu()
                      for w in wins], dim=0)                    # [N, k, 2]


def _ade(pred_wp, gt):
    """`ade_0_2s` exactly as `bench._suite` defines it: mean displacement error
    over the 4 canonical horizons (0.5/1/1.5/2 s)."""
    return float(torch.linalg.norm(pred_wp - gt, dim=-1).mean())


def equivalence(base_traj, var_traj, gt, wp_steps) -> dict:
    """Numerical agreement of a variant against the eager reference.

    Reports the deviation on the FULL decoded trajectory and the change in the
    scored quantity (`ade_0_2s`). For a CUDA graph — which replays the identical
    kernels — anything materially above float noise is a CAPTURE BUG, not a
    precision trade, and must be treated that way."""
    idx = torch.tensor([s - 1 for s in wp_steps])
    d = (var_traj - base_traj)
    denom = base_traj.abs().clamp_min(1e-6)
    b_wp, v_wp = base_traj.index_select(1, idx), var_traj.index_select(1, idx)
    shift = torch.linalg.norm(v_wp - b_wp, dim=-1)              # [N, 4] metres
    ade_b, ade_v = _ade(b_wp, gt), _ade(v_wp, gt)
    return {
        "n_windows": int(base_traj.shape[0]),
        "max_abs_dev_m": round(float(d.abs().max()), 9),
        # element-wise relative error (comparable to the 2.8e-7 CUDA-graph
        # precedent) AND the same deviation against the trajectory's own scale.
        # The element-wise form divides by a LATERAL coordinate that is legitimately
        # near zero on straight driving, so it can read large while the physical
        # error is sub-millimetre — always quote the metres too.
        "rel_err_max": round(float((d.abs() / denom).max()), 9),
        "rel_err_max_vs_traj_scale": round(
            float(d.abs().max() / base_traj.abs().max().clamp_min(1e-9)), 9),
        "wp_shift_m_mean": round(float(shift.mean()), 9),
        "wp_shift_m_max": round(float(shift.max()), 9),
        "wp_shift_m_at_2s_mean": round(float(shift[:, -1].mean()), 9),
        "cosine": round(float(torch.nn.functional.cosine_similarity(
            base_traj.reshape(1, -1), var_traj.reshape(1, -1)).item()), 9),
        "ade_0_2s_reference": round(ade_b, 8),
        "ade_0_2s_variant": round(ade_v, 8),
        "ade_0_2s_delta_m": round(ade_v - ade_b, 9),
        "finite": bool(torch.isfinite(var_traj).all()),
        "note": "reference = the eager lever of THIS sweep, same precision "
                "flags, same windows. ADE here is over these windows only — it "
                "is an equivalence check, NOT the canonical 40-episode heldout "
                "number (that is the accuracy harness's job).",
    }


# ---------------------------------------------------------------------------
# the lever measurement
# ---------------------------------------------------------------------------
@torch.no_grad()
def measure_levers(entry, L, ep, eps=None, device="cuda", precision="tf32",
                   batch=1, iters=200, warmup=30, k=K_MAX, equiv_n=32,
                   want=None, mem=True) -> dict:
    """Time + verify every inference variant under ONE precision context.

    Backend flags are saved/restored exactly like `measure` — an efficiency
    probe must never be able to move a later accuracy number."""
    _saved = (torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        return _measure_levers(entry, L, ep, eps, device, precision, batch,
                               iters, warmup, k, equiv_n, want, mem)
    finally:
        (torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark) = _saved


@torch.no_grad()
def _measure_levers(entry, L, ep, eps, device, precision, batch, iters, warmup,
                    k, equiv_n, want, mem) -> dict:
    ctx, prec = _set_precision(precision)
    levers, errs, stagefns = build_levers(entry, L, ep, device, batch, k, want)
    out = {"precision": prec, "env": _env(), "gpu_state_before": _gpu_state(),
           "protocol": {"batch": batch, "window": WINDOW, "rollout_k": k,
                        "iters": iters, "warmup": warmup,
                        "timing": "per-iteration torch.cuda.Event, "
                                  "torch.cuda.synchronize() bracketed, warmup "
                                  "discarded (identical to the baseline panel)"},
           "build_errors": errs, "levers": {}, "gpu_state_samples": []}

    wins, wp_steps = [], []
    if eps and equiv_n:
        try:
            wins, wp_steps = equiv_windows(entry, L, eps, device, equiv_n, 8, k)
            if k < max(wp_steps):                # canonical horizons need k>=20
                wins = []
                out["equivalence_error"] = (
                    f"rollout k={k} < max WP_STEP {max(wp_steps)}: the canonical "
                    f"ADE horizons are not decodable, equivalence skipped")
        except Exception as e:                                   # noqa: BLE001
            out["equivalence_error"] = f"{type(e).__name__}: {str(e)[:160]}"
    base_traj, gt = None, None
    if wins and "eager" not in levers:
        wins = []
        out["equivalence_error"] = (
            "the eager reference lever is absent, so no variant can be checked "
            "against it — every speedup would be unaccompanied by an accuracy "
            "number. Equivalence skipped rather than referenced to a variant.")
    if wins:
        gt = torch.cat([w["gt"] for w in wins], dim=0)
        with ctx():
            base_traj = _trajs(levers["eager"], wins)

    order = [n for n in LEVER_ORDER if n in levers]
    for name in order:
        lv = levers[name]
        row = {"meta": lv.meta}
        try:
            row["tick"] = _timeit(lv.tick, ctx, iters, warmup, device)
            if mem:
                row["memory"] = _peak_mem_live(lv.tick, ctx, device)
        except Exception as e:                                   # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        if wins and base_traj is not None:
            try:
                with ctx():
                    vt = _trajs(lv, wins)
                row["equivalence"] = equivalence(base_traj, vt, gt, wp_steps)
            except Exception as e:                               # noqa: BLE001
                row["equivalence"] = {"error": f"{type(e).__name__}: "
                                               f"{str(e)[:160]}"}
        out["levers"][name] = row
        out["gpu_state_samples"].append(
            {"after": name, **{kk: _gpu_state().get(kk)
                               for kk in ("other_compute_procs", "util_pct",
                                          "sm_clock_mhz", "temp_c")}})

    # derived: speedup + realtime verdict, referenced to the eager lever
    base = (out["levers"].get("eager", {}).get("tick") or {})
    b50, b99 = base.get("p50_ms"), base.get("p99_ms")
    for name, row in out["levers"].items():
        t = row.get("tick")
        if not t:
            continue
        row["speedup_vs_eager"] = {
            "p50": round(b50 / t["p50_ms"], 4) if b50 else None,
            "p99": round(b99 / t["p99_ms"], 4) if b99 else None,
            "saved_ms_p50": round(b50 - t["p50_ms"], 4) if b50 else None}
        row["realtime"] = {
            "budget_ms_at_%dhz" % int(DT_HZ): RT_BUDGET_MS,
            "budget_used_pct_p50": round(100 * t["p50_ms"] / RT_BUDGET_MS, 2),
            "budget_used_pct_p99": round(100 * t["p99_ms"] / RT_BUDGET_MS, 2),
            "max_control_rate_hz_p50": round(1000.0 / max(t["p50_ms"], 1e-9), 1),
            "max_control_rate_hz_p99": round(1000.0 / max(t["p99_ms"], 1e-9), 1),
            "meets_10hz_p99": bool(t["p99_ms"] < RT_BUDGET_MS)}

    # per-stage attribution: WHERE each lever acts (the orthogonality question)
    st: dict = {}
    for name, fn in stagefns.items():
        try:
            st[name] = _timeit(fn, ctx, max(30, iters // 2), warmup, device)
        except Exception as e:                                   # noqa: BLE001
            st[name] = {"error": f"{type(e).__name__}: {str(e)[:110]}"}
    out["stages"] = st
    g = lambda n: (st.get(n) or {}).get("mean_ms")               # noqa: E731
    e32, e16 = g(f"fp32_encode_window_{WINDOW}f"), g(f"fp16_encode_window_{WINDOW}f")
    r32, r16 = g(f"fp32_rollout_k{k}"), g(f"fp16_rollout_k{k}")
    rg = g(f"graph_rollout_k{k}")
    out["orthogonality"] = {
        "encoder_fp32_ms": e32, "encoder_fp16_ms": e16,
        "encoder_fp16_speedup": round(e32 / e16, 4) if e32 and e16 else None,
        "rollout_fp32_ms": r32, "rollout_fp16_ms": r16,
        "rollout_fp16_speedup": round(r32 / r16, 4) if r32 and r16 else None,
        "rollout_graph_ms": rg,
        "rollout_graph_speedup": round(r32 / rg, 4) if r32 and rg else None,
        "claim": "2026-07-18 (RTX 4060): the encoder is COMPUTE-bound so "
                 "precision helps it, and the predictor is LAUNCH-bound so only "
                 "a CUDA graph helps it. This block is the A40 test of that "
                 "claim: it holds iff encoder_fp16_speedup >> "
                 "rollout_fp16_speedup and rollout_graph_speedup is large."}

    # the decisive read: which lever removes the CPU round-trips
    gs = (out["levers"].get("graph_step", {}).get("tick") or {}).get("p50_ms")
    gr = (out["levers"].get("graph_rollout", {}).get("tick") or {}).get("p50_ms")
    if gs and gr:
        out["cpu_roundtrip_cost"] = {
            "graph_step_p50_ms": gs, "graph_rollout_p50_ms": gr,
            "delta_ms": round(gs - gr, 4),
            "per_step_ms": round((gs - gr) / max(k - 1, 1), 4),
            "note": f"both replay the SAME kernels; the only difference is that "
                    f"graph_step returns to the CPU {k - 1} times. The delta is "
                    f"therefore the per-step CPU round-trip cost."}
    out["gpu_state_after"] = _gpu_state()
    ex = out["gpu_state_before"].get("exclusive")
    mid = [s for s in out["gpu_state_samples"]
           if (s.get("other_compute_procs") or 0) > 0]
    out["contamination_check"] = {
        "gpu_exclusive_before": ex,
        "gpu_exclusive_after": out["gpu_state_after"].get("exclusive"),
        "intrusions_mid_run": len(mid),
        "valid": bool(ex and out["gpu_state_after"].get("exclusive")
                      and not mid),
        "note": "False => another process shared the GPU at some point during "
                "the sweep; re-run, do not publish this row."}
    return out


K_SWEEP_ACCURACY_CAVEAT = (
    "UNMEASURED — shortening the rollout changes WHAT THE MODEL PREDICTS; only "
    "the latency side is measured here. Evaluating the accuracy cost needs the "
    "canonical harness. Do not read an accuracy verdict from this block.")


def _k_sweep_summary(rows: dict, ks) -> dict:
    """Derive the saved-ms curve and attach the accuracy caveat.

    Pure (no GPU) so the caveat itself is test-pinned: the single way this
    latency-only block does damage is by being quoted as if it decided the
    accuracy question."""
    ref = max(ks)
    for by_k in rows.values():
        base = (by_k.get(str(ref)) or {}).get("p50_ms")
        if base:
            by_k[f"_saved_ms_p50_vs_k{ref}"] = {
                str(kk): round(base - by_k[str(kk)]["p50_ms"], 4)
                for kk in ks if (by_k.get(str(kk)) or {}).get("p50_ms")}
    return {"ks": list(ks), "by_variant": rows,
            "ACCURACY": K_SWEEP_ACCURACY_CAVEAT}


@torch.no_grad()
def rollout_k_sweep(entry, L, ep, device="cuda", precision="tf32", batch=1,
                    iters=200, warmup=30, ks=(20, 10, 5),
                    variants=("eager", "graph_rollout")) -> dict:
    """L5 — tick latency vs rollout length. LATENCY ONLY.

    A shorter rollout changes WHAT THE MODEL PREDICTS; that accuracy question
    is not answerable here and is deliberately left unanswered rather than
    guessed at."""
    _saved = (torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        ctx, prec = _set_precision(precision)
        rows: dict = {}
        for kk in ks:
            lv, errs, _ = build_levers(entry, L, ep, device, batch, kk,
                                       want=variants)
            for name, lever in lv.items():
                rows.setdefault(name, {})[str(kk)] = _timeit(
                    lever.tick, ctx, iters, warmup, device)
            for name, e in errs.items():
                rows.setdefault(name, {})[str(kk)] = {"error": e}
            del lv
        return {"precision": prec, **_k_sweep_summary(rows, ks)}
    finally:
        (torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark) = _saved


STRIDED_ACCURACY_CAVEAT = (
    "UNMEASURED and NOT CLAIMABLE. The k=2/k=4 heads are already trained and "
    "present in the deployed checkpoint, so a strided roll reaches the SAME 2 s "
    "horizon in fewer predictor calls WITHOUT retraining the predictor — but the "
    "step readout was calibrated on 0.1 s transitions and here decodes 0.2 s / "
    "0.4 s ones, so the decoded trajectory is not valid until the readout is "
    "recalibrated against a FROZEN predictor. This block supplies the latency "
    "side only and states that dependency; no equivalence is reported because "
    "the output is not expected to match.")


@torch.no_grad()
def strided_head_latency(entry, L, ep, device="cuda", precision="tf32", batch=1,
                         iters=200, warmup=30, k_total=K_MAX) -> dict:
    """L5b — LATENCY ONLY: reach the 2 s horizon in fewer SEQUENTIAL predictor
    calls using the already-trained multi-step heads.

    The serial chain — not the arithmetic — is what costs; a k=2 head halves the
    number of links, a k=4 head quarters it. Distinct from L5a (truncating the
    k=1 rollout), which shortens the horizon instead of striding it."""
    from tanitad.models.metric_dynamics import accumulate_se2
    _saved = (torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        ctx, prec = _set_precision(precision)
        model, sr = _wm_parts(entry, L)
        x = _window_inputs(entry, L, ep, device, batch)
        fw, aw, fa = x["fw"], x["aw"], x["fa"]
        states0 = model.encode_window(fw)
        hs = tuple(model.predictor.cfg.horizons)

        def _roll(h):
            n = k_total // h

            def _fn(states, actions, future_actions):
                ws, wa, dp = states, actions, []
                for j in range(n):
                    z = model.predictor(ws, wa)[h]
                    dp.append(sr(ws[:, -1], z))
                    if j < n - 1:
                        ws = torch.cat([ws[:, 1:], z.unsqueeze(1)], dim=1)
                        wa = torch.cat(
                            [wa[:, 1:],
                             future_actions[:, (j + 1) * h - 1].unsqueeze(1)],
                            dim=1)
                return accumulate_se2(torch.stack(dp, dim=1))
            return _fn, n

        rows = {}
        for h in hs:
            if k_total % h:
                rows[f"head_k{h}"] = {"error": f"{k_total} not divisible by {h}"}
                continue
            fn, n = _roll(h)
            row = {"predictor_calls": n, "head": h}
            try:
                row["eager"] = _timeit(
                    lambda: fn(model.encode_window(fw), aw, fa), ctx, iters,
                    warmup, device)
                g = GraphedFn(fn, {"states": states0, "actions": aw,
                                   "future_actions": fa})
                row["graphed"] = _timeit(
                    lambda: g.run(states=model.encode_window(fw), actions=aw,
                                  future_actions=fa), ctx, iters, warmup,
                    device)
            except Exception as e:                               # noqa: BLE001
                row["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            rows[f"head_k{h}"] = row
        base = (rows.get("head_k1", {}).get("eager") or {}).get("p50_ms")
        if base:
            for r in rows.values():
                if isinstance(r, dict) and (r.get("eager") or {}).get("p50_ms"):
                    r["speedup_vs_head_k1_eager"] = round(
                        base / r["eager"]["p50_ms"], 4)
        return {"precision": prec, "horizons_available": list(hs),
                "k_total": k_total, "by_head": rows,
                "ACCURACY": STRIDED_ACCURACY_CAVEAT}
    finally:
        (torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark) = _saved


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def _load_eps(key, device="cuda", n_eps=1):
    from taniteval import data, loaders
    from taniteval.registry import MODELS
    e = [m for m in MODELS if m["key"] == key]
    assert e, f"unknown model {key}"
    e = e[0]
    L = loaders.load(e, device)
    files = data.list_val_episodes(VAL, n_eps)
    eps = (data.load_frames(files) if L["feed"] == "frames"
           else data.load_features(files, L["feed"], device))
    return e, L, eps


def _load(key, device="cuda"):
    e, L, eps = _load_eps(key, device, 1)
    return e, L, eps[0]


def quick(entry, L, ep, device="cuda", iters=100, warmup=20) -> dict:
    """The DEFAULT axis: cheap enough to run inside every `runner run`.
    batch=1, fp32, stage breakdown + FLOPs, no throughput sweep (~5-15 s)."""
    return measure(entry, L, ep, device=device, precision="fp32", batch=1,
                   iters=iters, warmup=warmup, stages=True, flops=True,
                   throughput_batches=None)


def run_and_save(key, device="cuda", precisions=("fp32",), batch=1, iters=200,
                 warmup=30, throughput=True, res_dir=RES) -> dict:
    """Full efficiency run for one arm -> results/eff_<key>.json."""
    e, L, ep = _load(key, device)
    out = {"key": key,
           "model": {k: e.get(k) for k in ("key", "name", "arch", "encoder",
                                           "config_preset", "mode")},
           "ckpt_step": L["step"], "protocol": {
               "batch": batch, "window": WINDOW, "horizon_steps": K_MAX,
               "control_hz": DT_HZ, "iters": iters, "warmup": warmup,
               "timing": "per-iteration torch.cuda.Event, "
                         "torch.cuda.synchronize() bracketed, warmup discarded",
               "excluded": "host->device copy + uint8->float (reported "
                           "separately as input_prep)"}}
    tb = (1, 2, 4, 8, 16, 32) if throughput else None
    for prec in precisions:
        out[prec] = measure(e, L, ep, device=device, precision=prec,
                            batch=batch, iters=iters, warmup=warmup,
                            stages=True, flops=True,
                            throughput_batches=tb if prec == precisions[0]
                            else None)
    # cost-per-accuracy: pair the latency with THIS arm's own scored ADE
    acc = res_dir / f"{key}.json"
    if acc.exists():
        try:
            d = json.loads(acc.read_text())
            hm = d["heldout"]["model"]["ade_0_2s"]
            fs = d.get("full_set", {}).get("model", {}).get("ade_0_2s")
            ms = out[precisions[0]]["plan_step"]["p50_ms"]
            out["cost_per_accuracy"] = {
                "ade_0_2s_heldout": hm["mean"], "ade_ci95": hm.get("ci95"),
                "ade_0_2s_full_set": fs if not isinstance(fs, dict)
                else fs.get("mean"),
                "plan_step_p50_ms": ms,
                "ms_per_metre_of_ade_beaten_vs_cv": None,
                "note": "latency and ADE side by side — the deployment "
                        "trade-off read; ADE from this arm's own results JSON"}
        except Exception as ex:
            out["cost_per_accuracy"] = {"error": str(ex)[:120]}
    res_dir.mkdir(parents=True, exist_ok=True)
    # A contaminated run must NEVER overwrite a clean one. `tanitad-eval` is
    # shared: on 2026-07-20 a neighbouring VLM job appeared mid-session twice
    # and a 20-iteration smoke silently clobbered a clean 200-iteration result
    # with a 2x-slower number. Quarantine instead.
    dirty = [p for p in precisions
             if not out.get(p, {}).get("contamination_check", {}).get("valid")]
    dest = res_dir / f"eff_{key}.json"
    if dirty:
        import time as _t
        dest = res_dir / f"eff_{key}.CONTAMINATED-{_t.strftime('%Y%m%d-%H%M%S')}.json"
        out["QUARANTINED"] = (
            f"GPU was NOT exclusive during {dirty}; this file is quarantined and "
            f"eff_{key}.json was left untouched. Re-run on an idle GPU.")
        print(f"[eff] !! {key}: GPU NOT EXCLUSIVE during {dirty} — "
              f"QUARANTINED to {dest.name}; eff_{key}.json left untouched. "
              f"DO NOT PUBLISH these numbers.", flush=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    p = out[precisions[0]]["plan_step"]
    rt = out[precisions[0]]["realtime"]
    sh = out[precisions[0]].get("stage_shares", {})
    print(f"[eff] {key} step={out['ckpt_step']} {precisions[0]} b{batch}: "
          f"p50={p['p50_ms']:.2f}ms p95={p['p95_ms']:.2f} p99={p['p99_ms']:.2f} "
          f"| enc {sh.get('encoder_pct', '—')}% "
          f"| {out[precisions[0]].get('flops', {}).get('gflops', '—')} GFLOPs "
          f"@ {out[precisions[0]].get('compute_efficiency', {}).get('achieved_tflops', '—')} TFLOP/s "
          f"| {out[precisions[0]]['memory']['peak_alloc_mb']} MB "
          f"| {rt['budget_used_pct_p99']}% of the 100 ms budget (p99) "
          f"| {'MEETS' if rt['meets_10hz_p99'] else 'MISSES'} 10 Hz", flush=True)
    return out


def run_levers_and_save(key, device="cuda", precisions=("fp32", "tf32"),
                        batch=1, iters=200, warmup=30, k=K_MAX, equiv_n=32,
                        equiv_eps=2, want=None, k_sweep=(20, 10, 5),
                        res_dir=RES) -> dict:
    """Full inference-lever sweep for one arm -> results/eff_levers_<key>.json.

    Same contamination contract as `run_and_save`: a sweep measured on a shared
    GPU is QUARANTINED to its own file and never overwrites a clean one."""
    e, L, eps = _load_eps(key, device, equiv_eps)
    out = {"key": key,
           "model": {kk: e.get(kk) for kk in ("key", "name", "arch", "encoder",
                                              "config_preset", "mode")},
           "ckpt_step": L["step"],
           "what": "inference LEVERS: measured optimisations of the plan tick "
                   "(L1 CUDA graph / L2 encoder cache / L3 fp16 weights / "
                   "L4 composition / L5 rollout-length latency)",
           "baseline_artifact": f"eff_{key}.json (same protocol, same pod)",
           "protocol": {"batch": batch, "window": WINDOW, "rollout_k": k,
                        "control_hz": DT_HZ, "iters": iters, "warmup": warmup,
                        "equiv_windows": equiv_n, "equiv_episodes": equiv_eps,
                        "timing": "per-iteration torch.cuda.Event, "
                                  "torch.cuda.synchronize() bracketed, warmup "
                                  "discarded — IDENTICAL to the baseline panel"}}
    for prec in precisions:
        out[prec] = measure_levers(e, L, eps[0], eps, device=device,
                                   precision=prec, batch=batch, iters=iters,
                                   warmup=warmup, k=k, equiv_n=equiv_n,
                                   want=want)
        for name, row in out[prec]["levers"].items():
            t = row.get("tick") or {}
            eq = row.get("equivalence") or {}
            print(f"[lev] {prec:5s} {name:22s} p50={t.get('p50_ms', '-')} "
                  f"p99={t.get('p99_ms', '-')} "
                  f"x{row.get('speedup_vs_eager', {}).get('p50', '-')} "
                  f"| maxdev {eq.get('max_abs_dev_m', '-')} m "
                  f"| dADE {eq.get('ade_0_2s_delta_m', '-')} m", flush=True)
    if k_sweep:
        out["rollout_k_latency"] = {
            p: rollout_k_sweep(e, L, eps[0], device=device, precision=p,
                               batch=batch, iters=iters, warmup=warmup,
                               ks=k_sweep)
            for p in precisions}
        out["strided_head_latency"] = {
            p: strided_head_latency(e, L, eps[0], device=device, precision=p,
                                    batch=batch, iters=iters, warmup=warmup)
            for p in precisions}
    res_dir = Path(res_dir)
    res_dir.mkdir(parents=True, exist_ok=True)
    dirty = [p for p in precisions
             if not out.get(p, {}).get("contamination_check", {}).get("valid")]
    dest = res_dir / f"eff_levers_{key}.json"
    if dirty:
        import time as _t
        dest = (res_dir /
                f"eff_levers_{key}.CONTAMINATED-{_t.strftime('%Y%m%d-%H%M%S')}.json")
        out["QUARANTINED"] = (
            f"GPU was NOT exclusive during {dirty}; quarantined, "
            f"eff_levers_{key}.json left untouched. Re-run on an idle GPU.")
        print(f"[lev] !! {key}: GPU NOT EXCLUSIVE during {dirty} — QUARANTINED "
              f"to {dest.name}. DO NOT PUBLISH these numbers.", flush=True)
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"[lev] wrote {dest}", flush=True)
    return out


@torch.no_grad()
def fan_latency(entry, L, ep, device="cuda", precision="tf32",
                batches=(1, 2, 4, 8, 16), iters=50, warmup=10,
                variants=("eager", "graph_rollout", "all_levers")) -> dict:
    """Is imagine-and-select / CEM planning affordable? — LATENCY ONLY.

    The arithmetic that makes CEM look impossible ("8 candidates x 20 steps")
    assumes the fan costs 8 SEQUENTIAL ticks. It does not: a K-candidate fan is
    ONE rollout at batch K — the same 20 sequential predictor calls, each doing
    K times the work per kernel. On a LAUNCH-bound pass that is nearly free;
    that is the same mechanism (batch dilutes launch-boundedness) that made the
    2026-07-18 CUDA graph worth 2.57x at batch 1 but only 1.33x at batch 9,
    read in the other direction.

    Reports measured cost per candidate against the naive Kx projection, so the
    planning-budget decision is made on a measurement rather than on
    multiplication. No accuracy claim: which candidate a planner would pick is
    a planner question, not a latency one."""
    _saved = (torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        ctx, prec = _set_precision(precision)
        rows: dict = {}
        for b in batches:
            try:
                lv, errs, _ = build_levers(entry, L, ep, device, b, K_MAX,
                                           want=variants)
            except torch.cuda.OutOfMemoryError:
                rows[str(b)] = {"error": "OOM at build"}
                torch.cuda.empty_cache()
                break
            row = {}
            for name, lever in lv.items():
                try:
                    t = _timeit(lever.tick, ctx, iters, warmup, device)
                    row[name] = {"p50_ms": t["p50_ms"], "p99_ms": t["p99_ms"],
                                 "ms_per_candidate": round(t["p50_ms"] / b, 4)}
                except Exception as e:                           # noqa: BLE001
                    row[name] = {"error": f"{type(e).__name__}: {str(e)[:90]}"}
            row["build_errors"] = errs
            rows[str(b)] = row
            del lv
            torch.cuda.empty_cache()
        # the decisive contrast: measured fan cost vs the naive Kx projection
        for name in variants:
            one = (rows.get("1", {}).get(name) or {}).get("p50_ms")
            if not one:
                continue
            for b in batches:
                cell = rows.get(str(b), {}).get(name)
                if isinstance(cell, dict) and cell.get("p50_ms"):
                    cell["naive_Kx_projection_ms"] = round(one * b, 4)
                    cell["cheaper_than_naive_by"] = round(
                        one * b / cell["p50_ms"], 3)
        return {"precision": prec, "batches": list(batches), "by_batch": rows,
                "ACCURACY": "not applicable — this is a planning-BUDGET "
                            "measurement. Which candidate a planner selects, "
                            "and whether selection improves driving, are "
                            "separate questions the closed-loop harness owns."}
    finally:
        (torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark) = _saved


@torch.no_grad()
def tail_replicates(key="flagship-30k", device="cuda", precision="fp32",
                    reps=5, iters=200, warmup=30,
                    variants=("eager", "graph_rollout", "graph_fulltick"),
                    res_dir=RES) -> dict:
    """Does a CUDA graph COLLAPSE the run-to-run TAIL spread, not just the mean?

    Measured on this arm (2026-07-20 repeatability probe): the eager tick's p50
    is stable to ~1 % across identical back-to-back runs, but its **p99 swings
    ~26 %** — while REF-C's swings 1 %. A control loop is specified on its tail,
    so a lever that only moves the mean has not fixed the deployment problem.
    A graph replay is ONE launch of a fixed kernel sequence, so the prediction
    is that its tail spread collapses; this measures it instead of assuming it."""
    _saved = (torch.backends.cuda.matmul.allow_tf32,
              torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark)
    try:
        e, L, eps = _load_eps(key, device, 1)
        ctx, prec = _set_precision(precision)
        levers, errs, _ = build_levers(e, L, eps[0], device, 1, K_MAX,
                                       want=variants)
        out = {"key": key, "ckpt_step": L["step"], "precision": prec,
               "env": _env(), "protocol": {"reps": reps, "iters": iters,
                                           "warmup": warmup},
               "question": "does CUDA-graph capture collapse the run-to-run p99 "
                           "spread (the deployment-relevant number), not just "
                           "the mean?",
               "gpu_state_before": _gpu_state(), "build_errors": errs,
               "by_variant": {}}
        for name in (n for n in LEVER_ORDER if n in levers):
            rows = []
            for r in range(reps):
                t = _timeit(levers[name].tick, ctx, iters, warmup, device)
                g = _gpu_state()
                rows.append({**{kk: t[kk] for kk in
                                ("mean_ms", "p50_ms", "p95_ms", "p99_ms",
                                 "min_ms", "max_ms", "std_ms")},
                             "tail_ratio_p99_over_p50": round(
                                 t["p99_ms"] / max(t["p50_ms"], 1e-9), 4),
                             "exclusive": g.get("exclusive"),
                             "temp_c": g.get("temp_c"),
                             "sm_clock_mhz": g.get("sm_clock_mhz")})
                print(f"[tail] {name:16s} rep{r} p50={t['p50_ms']:8.3f} "
                      f"p99={t['p99_ms']:8.3f} max={t['max_ms']:8.3f} "
                      f"excl={g.get('exclusive')}", flush=True)

            def spread(vals):
                return {"min": round(min(vals), 4), "max": round(max(vals), 4),
                        "spread_pct": round(100 * (max(vals) - min(vals))
                                            / max(min(vals), 1e-9), 2)}
            out["by_variant"][name] = {
                "reps": rows,
                "p50_across_reps": spread([x["p50_ms"] for x in rows]),
                "p99_across_reps": spread([x["p99_ms"] for x in rows]),
                "tail_ratio_mean": round(
                    sum(x["tail_ratio_p99_over_p50"] for x in rows) / len(rows),
                    4)}
        out["gpu_state_after"] = _gpu_state()
        ex = out["gpu_state_before"].get("exclusive")
        out["contamination_check"] = {
            "gpu_exclusive_before": ex,
            "gpu_exclusive_after": out["gpu_state_after"].get("exclusive"),
            "valid": bool(ex and out["gpu_state_after"].get("exclusive")
                          and all(r["exclusive"] for v in
                                  out["by_variant"].values() for r in v["reps"])),
            "note": "every replicate carries its own exclusivity flag; one dirty "
                    "replicate invalidates the block."}
        res_dir = Path(res_dir)
        res_dir.mkdir(parents=True, exist_ok=True)
        dest = res_dir / (f"eff_levers_tail_{key}.json"
                          if out["contamination_check"]["valid"] else
                          f"eff_levers_tail_{key}.CONTAMINATED.json")
        dest.write_text(json.dumps(out, indent=2, default=str))
        print(f"[tail] wrote {dest}", flush=True)
        return out
    finally:
        (torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark) = _saved


def lever_table(res_dir=RES, key="flagship-30k", precision="tf32") -> str:
    """Plain-text lever table (lever / tick before-after / speedup / accuracy /
    10 Hz verdict) — the shape the research note publishes."""
    f = Path(res_dir) / f"eff_levers_{key}.json"
    if not f.exists():
        return f"(no {f.name})"
    d = json.loads(f.read_text())
    blk = d.get(precision) or {}
    base = ((blk.get("levers", {}).get("eager") or {}).get("tick") or {})
    lines = [f"{key} · {precision} · eager reference "
             f"p50 {base.get('p50_ms')} / p99 {base.get('p99_ms')} ms "
             f"· clean={blk.get('contamination_check', {}).get('valid')}",
             f"{'lever':24s} {'p50':>8s} {'p99':>8s} {'x p50':>7s} "
             f"{'maxdev_m':>10s} {'dADE_m':>9s}  10Hz@p99"]
    for name, row in blk.get("levers", {}).items():
        t, eq = row.get("tick") or {}, row.get("equivalence") or {}
        rt = row.get("realtime") or {}
        lines.append(
            f"{name:24s} {t.get('p50_ms', 0):8.2f} {t.get('p99_ms', 0):8.2f} "
            f"{(row.get('speedup_vs_eager') or {}).get('p50') or 0:7.2f} "
            f"{eq.get('max_abs_dev_m', float('nan')):10.2e} "
            f"{eq.get('ade_0_2s_delta_m', float('nan')):9.2e}  "
            f"{'YES' if rt.get('meets_10hz_p99') else 'no'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# report panel
# ---------------------------------------------------------------------------
def panel_rows(res_dir=RES) -> str:
    """HTML rows for the dashboard's inference-efficiency panel."""
    rows = []
    for f in sorted(Path(res_dir).glob("eff_*.json")):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        # Render CANONICAL artifacts only. Quarantined runs
        # (eff_<key>.CONTAMINATED-*.json) and hand-kept replicates live beside
        # the canonical file and must not appear as extra leaderboard rows.
        if f.name != f"eff_{d.get('key')}.json" or "QUARANTINED" in d:
            continue
        prec = next((p for p in PRECISIONS if p in d), None)
        if not prec:
            continue
        e = d[prec]
        p, rt = e["plan_step"], e["realtime"]
        sh = e.get("stage_shares", {})
        fl = e.get("flops", {})
        acc = d.get("cost_per_accuracy", {})
        ok = rt["meets_10hz_p99"]
        cls = "good" if rt["budget_used_pct_p99"] < 50 else \
            "warn" if ok else "crit"
        mix = []
        if sh.get("encoder_pct") is not None:
            mix.append(f"enc {sh['encoder_pct']}%")
        if sh.get("rollout_pct") is not None:
            mix.append(f"rollout {sh['rollout_pct']}%")
        if sh.get("decoder_total_pct") is not None:
            mix.append(f"decoder {sh['decoder_total_pct']}%")
        if sh.get("denoise_pct") is not None:
            mix.append(f"(denoise {sh['denoise_pct']}%)")
        if sh.get("imagination_pct") is not None:
            mix.append(f"imag {sh['imagination_pct']}%")
        tf = (e.get("compute_efficiency") or {}).get("achieved_tflops")
        if tf:
            mix.append(f"— {tf} TFLOP/s achieved")
        ade = acc.get("ade_0_2s_heldout")
        # A latency row measured on a SHARED GPU is not a result. Say so in the
        # table rather than letting it pass as clean.
        dirty = e.get("contamination_check", {}).get("valid") is False
        warn = ("<div class='meta' style='color:var(--crit)'>⚠ GPU NOT "
                "EXCLUSIVE during timing — re-run before quoting</div>"
                if dirty else "")
        rows.append(
            f"<tr><td><div class='mname'>{d.get('model', {}).get('name', d['key'])}"
            f"</div><div class='meta'>{e['meta'].get('decode', '')[:96]}</div>"
            f"{warn}</td>"
            f"<td class='r'><span class='big'>{p['p50_ms']:.1f}</span>"
            f"<span class='meta'> ms</span></td>"
            f"<td class='r mono'>{p['p95_ms']:.1f} / {p['p99_ms']:.1f}</td>"
            f"<td class='mono' style='font-size:11px'>{' · '.join(mix)}</td>"
            f"<td class='r mono'>{fl.get('gflops', '—')}</td>"
            f"<td class='r mono'>{e['memory']['peak_alloc_mb']:.0f} MB</td>"
            f"<td class='r mono'>{e['params']['total_params_m']:.1f} M</td>"
            f"<td class='r mono'>{ade if ade is None else f'{ade:.4f}'}</td>"
            f"<td class='r'><span class='pill {cls}'>"
            f"{rt['budget_used_pct_p99']:.0f}% of 100 ms</span></td>"
            f"<td class='r mono'>{prec}</td></tr>")
    return "\n".join(rows)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser("taniteval.efficiency")
    ap.add_argument("--model", required=True,
                    help="registry key, or 'all'")
    ap.add_argument("--precision", default="fp32",
                    help="comma list of " + ",".join(PRECISIONS))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--no-throughput", action="store_true")
    ap.add_argument("--levers", action="store_true",
                    help="run the INFERENCE-LEVER sweep (CUDA graph / encoder "
                         "cache / fp16 weights / composition / k-sweep) instead "
                         "of the baseline panel")
    ap.add_argument("--want", default=None,
                    help="levers: comma list restricting which variants to "
                         "build (default: all of " + ",".join(LEVER_ORDER) + ")")
    ap.add_argument("--equiv-n", type=int, default=32)
    ap.add_argument("--equiv-eps", type=int, default=2)
    ap.add_argument("--no-k-sweep", action="store_true")
    a = ap.parse_args(argv)
    precs = tuple(p.strip() for p in a.precision.split(","))
    keys = [a.model]
    if a.model == "all":
        from taniteval.registry import MODELS
        keys = [m["key"] for m in MODELS]
    for k in keys:
        try:
            if a.levers:
                run_levers_and_save(
                    k, precisions=precs, batch=a.batch, iters=a.iters,
                    warmup=a.warmup, equiv_n=a.equiv_n,
                    equiv_eps=a.equiv_eps,
                    want=(tuple(w.strip() for w in a.want.split(","))
                          if a.want else None),
                    k_sweep=None if a.no_k_sweep else (20, 10, 5))
            else:
                run_and_save(k, precisions=precs, batch=a.batch, iters=a.iters,
                             warmup=a.warmup, throughput=not a.no_throughput)
        except Exception as e:
            print(f"[eff] {k} FAILED: {type(e).__name__}: {str(e)[:160]}",
                  flush=True)


if __name__ == "__main__":
    main()
