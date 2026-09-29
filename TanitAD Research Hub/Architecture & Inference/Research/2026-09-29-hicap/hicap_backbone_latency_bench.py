#!/usr/bin/env python3
"""HiCAP H-HC11 on-device backbone latency benchmark.

Decides pre-registered hypothesis H-HC11 ("latency and precision", Paper/HiCAP/sec_protocol.tex):
    >= 1,000 ticks on Thor: fast-path p95, VLM refresh p95 at the chosen camera count,
    NVFP4-vs-BF16 feature drift;  pass = fast path p95 <= 25 ms, refresh p95 <= 50 ms (one camera),
    drift within the margin.

===========================================================================================
STATUS (fail loud): WRITTEN WITHOUT A GPU.  The `hf` backend has NEVER been executed.
    Built in a cloud container with no torch, no CUDA and no route to huggingface.co (HTTP 403 on the
    proxy).  What IS verified: the pure logic (percentiles, token arithmetic, scheduler, token age,
    verdict, feature comparison, JSON, CLI) via the `mock` backend and pytest.  What is UNVERIFIED and
    must be shaken out by a human on a Jetson Thor the first time: everything under `class HFBackend`
    (model loading, processor size override, hook placement, hidden-state tuple length, CUDA-event
    timing, the high-priority-stream fast-path thread).  The API surface it targets was read from the
    transformers 5.17.0 wheel source (modeling_qwen3_vl.py, modeling_qwen3_5.py, output_capturing.py)
    on 2026-09-29, NOT executed.  The model ids in CANDIDATES come from repo docs; existence on the Hub
    is UNVERIFIED (see --list-candidates).
===========================================================================================

EVIDENCE CLASSES.  A `--backend mock` run writes `"evidence_class": "MOCK"`; its verdict block prints
MOCK_ONLY for every measured row and can never print PASS.  Mock latencies are the output of a
documented arithmetic model (see MockBackend) - they are NOT measurements and NOT the paper's ESTIMATED
Table rows; they exist only to exercise the scheduler/stats/verdict/JSON/CLI code paths.
Only a `--backend hf` run on Thor is `MEASURED`.

WHAT IS TIMED (per refresh tick, batch 1, CUDA events - never wall-clock around async launches)
    vision_ms       ViT forward   (forward pre/post hooks on core.visual)
    llm_prefill_ms  decoder stack incl. output_hidden_states=True capture (hooks on core.language_model),
                    NO generate, use_cache disabled via the configs (not a call kwarg: kwargs flow into the vision blocks), lm_head not executed (we call the multimodal core,
                    `model.model`, not the ForConditionalGeneration wrapper, and drop lm_head)
    pool_ms         adaptive-average-pool over the token axis of each tap to <= --pool-tokens
    residual_ms     total - vision - llm - pool  (embedding merge, M-RoPE index, dtype cast, [H2D])
    total_ms        first event before the model call -> last event after pooling
The tick loop is PACED at --refresh-hz (default 2 Hz) with a "latest-wins" trigger: frame j is captured at
slot time j*period; a refresh that overruns makes the next start LATE; if whole slots elapsed while busy
they are DROPPED (only the newest pending frame is processed).  Token age (fast-path tick time minus
capture time of the newest READY token set) is computed from that timestamped schedule.

CONCURRENT FAST PATH (--concurrent-fastpath).  Thor has no preemption, so the 10 Hz fast path is measured
WHILE a refresh runs.  The fast path is a SYNTHETIC PROXY (NOT the HiCAP fusion transformer): 6 pre-norm
blocks, d=512, 4 heads, FFN 2048 (~18.9 M params, random weights, seeded), 128 backbone + 64 query/motion
= 192 tokens, batch 1, model dtype, launched at --fast-hz on a torch.cuda.Stream(priority=-1) in its own
thread while the refresh runs on the default (lowest-priority) stream.  Phases, in order (order chosen so
the "alone" baseline is measured on a hot device, like the concurrent phase):
    B  refresh alone          --ticks paced refreshes                       (headline refresh stats)
    A  fast path alone        --fast-baseline-ticks (>= 500)                 (baseline)
    C  refresh + fast path    --concurrent-ticks refreshes, fast thread live (contention)
Fast-path per-tick latency = CUDA events on the high-priority stream (primary, as briefed) AND wall
submit->complete (secondary; exposes CPU launch/GIL contention that events cannot see).  The concurrent p95
is over ALL fast ticks of phase C (the deployed truth); the p95 over only the ticks that overlapped a
refresh is reported too, as a SUPPLEMENTARY, non-gating row (it is not in the pre-registration).

MEMORY.  Headline field = in-process torch.cuda.max_memory_allocated() (the only admissible probe on Thor
unified memory: mem_get_info(), free, tegrastats and VmRSS all lie there - CLAUDE.md).  --log-inadmissible-memory
also logs mem_get_info() and VmRSS, labelled INADMISSIBLE_ON_THOR, purely so a reader can see the disagreement.

TOKENS.  Qwen3-VL-lineage processors: patch 16, 2x2 merge => 32 px/token; 256x640 => (256/32)*(640/32) =
8*20 = 160 tokens per frame-camera (Qwen3-VL README: "compression ratio is 32"; cosmos_reason2_utils/vision.py
IMAGE_PATCH_SIZE = 16, qwen_vl_utils SPATIAL_MERGE_SIZE = 2).  The HF backend READS patch_size / merge_size from
the loaded processor and FAILS LOUD if the produced image-token count differs from this arithmetic.  Each of the
cameras x frames-per-camera frames is treated as its own image (160 tokens each); this matches the cost model
(RA_vlm_backbones.md 5.2: 3 cam x 3 frames => 1,504 LLM tokens = 9*160 + 64).  Qwen3-VL VIDEO input would halve
the ViT/LLM tokens (temporal patch 2) - NOT modelled here.  A ViT-feature cache (only the newest frame per camera
through the ViT, RA 5.3) is NOT implemented: this bench times the un-cached refresh, an upper bound.
PITFALL caught while reading the repos: the Cosmos-Reason2 inference sample sets shortest_edge = 256 tokens *
1024 px, which would UPSCALE a 256x640 frame to 286 tokens; this script overrides the processor size to
[4, 1280] tokens/image so 256x640 stays at 160, and asserts it.

PRECISION.  bf16/fp16 run on the HF backend.  fp8/nvfp4 FAIL LOUD: build the engine with TensorRT-Edge-LLM
(`tensorrt-edgellm-quantize llm --quantization nvfp4 --lm_head_quantization nvfp4 --visual_quantization fp8`,
docs/source/user_guide/features/quantization.md), dump pooled features on the SAME frames in the schema below, and
compare with `--compare-features BF16.pt NVFP4.pt`.  UNVERIFIED: the TRT-Edge-LLM docs we read document no API to
return intermediate prefill hidden states for a plain VLM; the exporter may need extra ONNX outputs at the 3 taps.
No numeric drift margin is pre-registered ("within the margin", sec_protocol.tex); --drift-margin-* have NO default
and without them the drift verdict is NO_MARGIN_PREREGISTERED (numbers only).

FEATURE FILE SCHEMA (`--save-features x.pt|x.npz`; .pt needs torch, .npz needs only numpy):
    meta = {"schema": "hicap.h_hc11.pooled_features/1", "model", "dtype", "evidence_class", "taps": [layer idx],
            "tap_convention", "pool_tokens", "frames_source": "synthetic"|"real", "frames_fingerprint",
            "n_windows", "frame_ids"...}
    features: tap_0, tap_1, tap_2 -> float32 array [n_windows, pooled_tokens, hidden]
  .pt: torch.save({"meta": meta, "features": {"tap_0": Tensor, ...}});  .npz: keys meta_json + tap_k.
Drift from SYNTHETIC frames is labelled INADMISSIBLE_FOR_DRIFT (noise features are unrepresentative); use
--frames-dir with real PNG/JPG frames for the drift run, the same directory and order for both precisions.

THREADS.  `torch` spawns ~113 threads per process; main() sets OMP_NUM_THREADS (default 6, --omp-threads)
BEFORE torch is imported (an already-exported value is respected and recorded) and calls torch.set_num_threads.

PYTHONPATH.  This script imports nothing from `tanitad`, so PYTHONPATH=/workspace/TanitAD/stack is NOT needed to
run it.  (That trap applies to the trainers/eval; export it anyway in any shell that will also launch TanitAD tools.)
The script never sets the Jetson power mode / clocks (needs sudo); it RECORDS `nvpmodel -q` when readable.

Exit codes: 0 ok | 1 runtime error | 2 usage / fail-loud configuration (UsageError) | 3 backend unavailable (no torch / no CUDA).
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

SCHEMA = "hicap.h_hc11.backbone_latency_bench"
SCHEMA_VERSION = "1.0.0"
FEATURE_SCHEMA = "hicap.h_hc11.pooled_features/1"

# --------------------------------------------------------------------------------------
# Pre-registered thresholds.  Provenance is part of the number.
#   refresh_sync_p95_ms / fastpath_p95_ms / min_ticks : Paper/HiCAP/sec_protocol.tex, row H-HC11
#   refresh_async_p95_ms / token_age_slack_ms / duty_cycle_max : the implementation brief (the
#       asynchronous reading); NOT present in the sec_protocol.tex row as read 2026-09-29 (INHERITED).
#   min_fast_ticks : implementation brief (>= 500 fast-path ticks per phase).
# --------------------------------------------------------------------------------------
PREREG = {
    "refresh_sync_p95_ms": 50.0,
    "refresh_async_p95_ms": 250.0,
    "token_age_slack_ms": 500.0,
    "duty_cycle_max": 0.5,
    "fastpath_p95_ms": 25.0,
    "min_ticks": 1000,
    "min_fast_ticks": 500,
}

STATUSES = ("PASS", "FAIL", "UNMEASURED", "MOCK_ONLY")


class UsageError(Exception):
    """Fail-loud configuration / usage error: main() prints it to stderr and exits with code 2."""

MIN_TOKENS_PER_IMAGE = 4        # qwen_vl_utils IMAGE_MIN_TOKEN_NUM
MAX_TOKENS_PER_IMAGE = 1280     # Qwen3-VL README: recommended 256-1280 tokens per image
DEFAULT_PROMPT_TOKENS = 64      # sec_eff.tex Table tab:thor: n = 160/camera + 64 prompt/query

PROXY_SPEC = {"d_model": 512, "layers": 6, "heads": 4, "ffn": 2048, "tokens": 192,
              "backbone_tokens": 128, "query_motion_tokens": 64, "batch": 1}

# HF ids: from repo docs read 2026-09-29 (shallow clones).  Hub existence is UNVERIFIED (huggingface.co 403 from the
# build container).  `params_b` is nominal and used only by the mock cost model.
_TRT_SUP = "TensorRT-Edge-LLM docs/source/user_guide/getting_started/supported-models.md"
CANDIDATES = {
    "qwen3-vl-2b": dict(hf_id="Qwen/Qwen3-VL-2B-Instruct", params_b=2.0, id_source=f"{_TRT_SUP}:143", loader="AutoModelForImageTextToText -> Qwen3VLForConditionalGeneration (transformers 5.17 modeling_auto.py:1166)"),
    "qwen3-vl-4b": dict(hf_id="Qwen/Qwen3-VL-4B-Instruct", params_b=4.0, id_source=f"{_TRT_SUP}:144", loader="AutoModelForImageTextToText -> Qwen3VLForConditionalGeneration"),
    "qwen3-vl-8b": dict(hf_id="Qwen/Qwen3-VL-8B-Instruct", params_b=8.0, id_source=f"{_TRT_SUP}:145", loader="AutoModelForImageTextToText -> Qwen3VLForConditionalGeneration"),
    "cosmos-reason2-2b": dict(hf_id="nvidia/Cosmos-Reason2-2B", params_b=2.0, id_source="cosmos-reason2/README.md:62; scripts/inference_sample.py:55", loader="Qwen3VLForConditionalGeneration (inference_sample.py:56); Qwen3-VL architecture"),
    "cosmos-reason2-8b": dict(hf_id="nvidia/Cosmos-Reason2-8B", params_b=8.0, id_source="cosmos-reason2/README.md:63", loader="Qwen3-VL architecture (README.md:278)"),
    "cosmos3-edge": dict(hf_id="nvidia/Cosmos3-Edge", params_b=4.0, id_source=f"{_TRT_SUP}:292; cosmos/README.md model table", loader="UNVERIFIED: the reasoner's transformers class is not established from the repo docs (cosmos inference_benchmarks.md says 'raw Hugging Face Transformers in eager mode'); may need --trust-remote-code or the cosmos-framework loader; reasoner = 2.4 B (RA 1.4)"),
    "qwen3.5-4b": dict(hf_id="Qwen/Qwen3.5-4B", params_b=4.0, id_source=f"{_TRT_SUP}:158", loader="AutoModelForImageTextToText -> Qwen3_5ForConditionalGeneration (transformers 5.17 modeling_auto.py:1163); Gated-DeltaNet layers use fla/causal_conv1d kernels or a SLOW torch fallback (stamp.optional_kernels)"),
}


# ======================================================================================
# 1. Pure statistics
# ======================================================================================
def percentile(values, q):
    """Linear-interpolation percentile, identical to numpy.percentile(..., method='linear')."""
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        raise ValueError("percentile of an empty sequence")
    if n == 1:
        return xs[0]
    k = (n - 1) * (float(q) / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(values):
    """n / mean / std / min / p50 / p95 / p99 / max.  Empty input -> n=0 and None fields (JSON-safe)."""
    xs = sorted(float(v) for v in values)
    n = len(xs)
    keys = ("mean", "std", "min", "p50", "p95", "p99", "max")
    if n == 0:
        return {"n": 0, **{k: None for k in keys}}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    return {
        "n": n, "mean": mean, "std": math.sqrt(var), "min": xs[0],
        "p50": percentile(xs, 50), "p95": percentile(xs, 95), "p99": percentile(xs, 99), "max": xs[-1],
    }


# ======================================================================================
# 2. Token arithmetic (Qwen3-VL-lineage processors)
# ======================================================================================
def _round_by(x, f):
    return round(x / f) * f


def _ceil_by(x, f):
    return math.ceil(x / f) * f


def _floor_by(x, f):
    return math.floor(x / f) * f


def smart_resize(height, width, factor, min_pixels, max_pixels):
    """Mirror of qwen_vl_utils.vision_process.smart_resize / transformers qwen2_vl smart_resize."""
    h_bar = max(factor, _round_by(height, factor))
    w_bar = max(factor, _round_by(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by(height / beta, factor)
        w_bar = _floor_by(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by(height * beta, factor)
        w_bar = _ceil_by(width * beta, factor)
    return int(h_bar), int(w_bar)


def tokens_per_image(height, width, patch=16, merge=2, min_tokens=MIN_TOKENS_PER_IMAGE,
                     max_tokens=MAX_TOKENS_PER_IMAGE):
    """Merged visual tokens for one frame: (H'/(patch*merge)) * (W'/(patch*merge)).  256x640 -> 160."""
    factor = patch * merge
    hb, wb = smart_resize(height, width, factor, min_tokens * factor ** 2, max_tokens * factor ** 2)
    return (hb // factor) * (wb // factor)


def token_budget(cameras, frames_per_camera, height, width, prompt_tokens=DEFAULT_PROMPT_TOKENS,
                 patch=16, merge=2):
    per = tokens_per_image(height, width, patch, merge)
    images = int(cameras) * int(frames_per_camera)
    return {"images_per_tick": images, "visual_tokens_per_image": per, "visual_tokens": images * per,
            "prompt_tokens": int(prompt_tokens), "llm_input_tokens": images * per + int(prompt_tokens),
            "patch_size": patch, "merge_size": merge}


def pooled_token_counts(llm_input_tokens, pool_tokens, n_taps):
    per_tap = min(int(llm_input_tokens), int(pool_tokens))
    return {"pooled_tokens_per_tap": per_tap, "n_taps": int(n_taps), "pooled_tokens_total": per_tap * int(n_taps)}


def parse_taps(spec):
    """'auto' or a comma list of non-negative ints -> 'auto' | [ints].  Range is checked against the OBSERVED layer count later."""
    if isinstance(spec, str) and spec.strip().lower() == "auto":
        return "auto"
    taps = [int(t) for t in (spec.split(",") if isinstance(spec, str) else spec) if str(t).strip() != ""]
    if not taps:
        raise ValueError("--taps: empty tap list")
    if any(t < 0 for t in taps):
        raise ValueError(f"--taps: negative index in {taps}")
    return taps


def resolve_taps(spec, n_layers):
    """Tap indices into HF `hidden_states` (0 = embedding output, i = output of decoder layer i, L = post-final-norm).
    'auto' -> round(L/3), round(2L/3), L."""
    parsed = parse_taps(spec)
    if parsed == "auto":
        return [max(1, int(round(n_layers / 3.0))), max(1, int(round(2.0 * n_layers / 3.0))), int(n_layers)]
    for t in parsed:
        if t > n_layers:
            raise ValueError(f"--taps: index {t} outside [0, {n_layers}] (hidden_states has {n_layers + 1} entries)")
    return parsed


TAP_CONVENTION = ("HF hidden_states tuple, len = num_hidden_layers + 1: index 0 = inputs_embeds, index i (1..L-1) = output "
                  "of decoder layer i, index L = last_hidden_state AFTER the final norm (transformers 5.17 "
                  "output_capturing.py tie_last_hidden_states; taps below L are un-normalised residual stream)")


# ======================================================================================
# 3. Scheduler, token age, fast-path bookkeeping (backend-independent)
# ======================================================================================
class RealClock:
    virtual = False

    def __init__(self):
        self._t0 = time.perf_counter()

    def now_ms(self):
        return (time.perf_counter() - self._t0) * 1e3

    def sleep_until_ms(self, t_ms):
        while True:
            rem = t_ms - self.now_ms()
            if rem <= 0:
                return
            time.sleep((rem - 0.8) / 1e3 if rem > 1.0 else 1e-4)   # sleep releases the GIL; <=1 ms polling

    def advance(self, d_ms):  # real time advances by itself
        return None


class VirtualClock:
    virtual = True

    def __init__(self):
        self.t = 0.0

    def now_ms(self):
        return self.t

    def sleep_until_ms(self, t_ms):
        self.t = max(self.t, float(t_ms))

    def advance(self, d_ms):
        self.t += float(d_ms)


def run_refresh_schedule(clock, refresh_fn, n_ticks, period_ms, pace=True, late_tol_ms=1.0, lead_ms=20.0):
    """Run `n_ticks` refreshes on a timestamped fixed-rate trigger, latest-wins.

    Frame j is captured at slot time t0 + j*period.  The executor is serial.  When it becomes free at time `now`,
    it takes the NEWEST slot whose capture time has passed; older unserved slots are dropped.  A refresh whose start
    is later than its capture time by more than `late_tol_ms` is LATE.  `refresh_fn(slot) -> dict` of stage ms with
    'total_ms'; on a virtual clock the scheduler advances time by total_ms, on the real clock refresh_fn blocks.
    Returns (records, summary).  pace=False runs back-to-back: dropped/late are None (no deployed schedule).
    """
    if period_ms <= 0:
        raise ValueError("period_ms must be > 0")
    t0 = clock.now_ms() + (lead_ms if pace else 0.0)
    records, dropped, late, next_slot = [], 0, 0, 0
    while len(records) < n_ticks:
        if pace:
            now = clock.now_ms()
            newest = int(math.floor((now - t0) / period_ms + 1e-9))
            if newest > next_slot:
                dropped += newest - next_slot
                slot = newest
            else:
                slot = next_slot
            sched = t0 + slot * period_ms
            if now < sched:
                clock.sleep_until_ms(sched)
        else:
            slot = next_slot
            sched = clock.now_ms()
        start = clock.now_ms()
        stages = refresh_fn(slot)
        if clock.virtual:
            clock.advance(stages["total_ms"])
        end = clock.now_ms()
        is_late = bool(pace and (start - sched) > late_tol_ms)
        late += int(is_late)
        records.append({"slot": slot, "scheduled_ms": sched, "start_ms": start, "end_ms": end,
                        "late": is_late, "stages": stages})
        next_slot = slot + 1
    lateness = [r["start_ms"] - r["scheduled_ms"] for r in records] if pace else []
    summary = {
        "policy": "latest-wins (frame j captured at slot time j*period; newest pending frame processed; older dropped)",
        "paced": bool(pace), "period_ms": period_ms, "n_executed": len(records),
        "n_dropped": dropped if pace else None, "n_late": late if pace else None,
        "late_tolerance_ms": late_tol_ms if pace else None,
        "max_lateness_ms": max(lateness) if lateness else None,
        "n_overrun": sum(1 for r in records if r["stages"]["total_ms"] > period_ms),
        "span_slots": (records[-1]["slot"] + 1) if records else 0,
    }
    return records, summary


def token_ages(records, fast_period_ms):
    """Age (ms) of the newest READY token set as seen by each fast-path tick.

    Fast ticks sit on the absolute grid m*fast_period_ms.  A refresh's tokens are ready at its end_ms and carry
    the capture time of its slot.  age(f) = f - capture(newest refresh with end <= f), for every grid point from the
    first ready time to the last ready time.  Steady state without queueing: age in [d, period + d)."""
    if not records or fast_period_ms <= 0:
        return []
    recs = sorted(records, key=lambda r: r["end_ms"])
    ends = [r["end_ms"] for r in recs]
    first, last = ends[0], ends[-1]
    m = int(math.ceil(first / fast_period_ms - 1e-12))
    ages, k = [], 0
    while m * fast_period_ms <= last + 1e-9:
        f = m * fast_period_ms
        while k + 1 < len(recs) and ends[k + 1] <= f + 1e-12:
            k += 1
        ages.append(f - recs[k]["scheduled_ms"])
        m += 1
    return ages


def classify_overlap(fast_ticks, busy):
    """Split fast-path ticks by whether [start,end] intersects any refresh busy interval (start,end)."""
    busy = sorted(busy)
    starts = [b[0] for b in busy]
    ends_prefix_max, cur = [], -math.inf
    for b in busy:
        cur = max(cur, b[1])
        ends_prefix_max.append(cur)
    over, non = [], []
    for t in fast_ticks:
        i = bisect.bisect_left(starts, t["end_ms"]) - 1     # last busy interval starting before the tick ends
        hit = i >= 0 and ends_prefix_max[i] > t["start_ms"]
        (over if hit else non).append(t)
    return over, non


def overlap_ms(a0, a1, intervals):
    tot = 0.0
    for b0, b1 in intervals:
        lo, hi = max(a0, b0), min(a1, b1)
        if hi > lo:
            tot += hi - lo
    return tot


def stage_stats(records):
    keys = ["vision_ms", "llm_prefill_ms", "pool_ms", "residual_ms", "total_ms"]
    return {k: summarize([r["stages"][k] for r in records if k in r["stages"]]) for k in keys}


def stability(records):
    """First-half vs second-half p50/p95 of total_ms (thermal / clock-throttle indicator)."""
    tot = [r["stages"]["total_ms"] for r in records]
    if len(tot) < 4:
        return None
    h = len(tot) // 2
    a, b = tot[:h], tot[h:]
    p50a, p50b = percentile(a, 50), percentile(b, 50)
    return {"p50_first_half_ms": p50a, "p50_second_half_ms": p50b,
            "p95_first_half_ms": percentile(a, 95), "p95_second_half_ms": percentile(b, 95),
            "p50_ratio_second_over_first": (p50b / p50a) if p50a else None}


def summarize_fast(ticks, busy=None):
    out = {"n": len(ticks), "gpu_event_ms": summarize([t["gpu_ms"] for t in ticks]),
           "wall_ms": summarize([t["wall_ms"] for t in ticks])}
    if busy is not None:
        over, non = classify_overlap(ticks, busy)
        out["overlapped_with_refresh"] = {"n": len(over), "gpu_event_ms": summarize([t["gpu_ms"] for t in over]),
                                          "wall_ms": summarize([t["wall_ms"] for t in over])}
        out["not_overlapped"] = {"n": len(non), "gpu_event_ms": summarize([t["gpu_ms"] for t in non])}
    return out


# ======================================================================================
# 4. Verdict (pre-registered thresholds)
# ======================================================================================
def _row(name, value, thr, unit, readings, gating, note, status_override=None, reason=None):
    return {"name": name, "status": status_override or "UNMEASURED", "value": value, "op": "<=",
            "threshold": thr, "unit": unit, "readings": readings, "gating": gating, "note": note, "reason": reason}


def combine_status(statuses):
    """FAIL dominates; else UNMEASURED if any; else PASS.  (MOCK handled by the caller.)"""
    if any(s == "FAIL" for s in statuses):
        return "FAIL"
    if any(s in ("UNMEASURED", "MOCK_ONLY") for s in statuses):
        return "UNMEASURED" if any(s == "UNMEASURED" for s in statuses) else "MOCK_ONLY"
    return "PASS"


def evaluate_verdict(m, evidence_class, prereg=None):
    """Evaluate the pre-registered thresholds.

    m keys (any may be None -> UNMEASURED): n_ticks, refresh_p95_ms, refresh_mean_ms, refresh_hz, token_age_p95_ms,
    fast_concurrent_n, fast_concurrent_p95_ms, fast_overlapped_p95_ms, fast_overlapped_n, refresh_concurrent_p95_ms,
    refresh_concurrent_n,
    preprocess_allowance_ms, tick_budget_ms, on_thor (True/False/None).
    Status rules: value <= threshold -> PASS else FAIL; None -> UNMEASURED; a PASS with fewer than the required ticks
    is downgraded to UNMEASURED (a FAIL stays FAIL); evidence_class == 'MOCK' -> MOCK_ONLY for every row that has a
    value (a mock run can never print PASS or FAIL)."""
    pr = dict(PREREG)
    pr.update(prereg or {})
    mock = (evidence_class == "MOCK")
    n_ticks = m.get("n_ticks")
    ticks_ok = n_ticks is not None and n_ticks >= pr["min_ticks"]
    fast_n = m.get("fast_concurrent_n")
    fast_ok = fast_n is not None and fast_n >= pr["min_fast_ticks"]
    rows = {}

    def add(name, value, thr, unit, readings, note, gating=True, need_ticks=True, need_fast=False, n_for_ticks=None, min_n=None):
        r = _row(name, value, thr, unit, readings, gating, note)
        n_have = n_ticks if n_for_ticks is None else n_for_ticks
        n_need = pr["min_ticks"] if min_n is None else min_n
        have_ticks = n_have is not None and n_have >= n_need
        if value is None or thr is None:
            r["status"], r["reason"] = "UNMEASURED", "value or threshold not measured in this run"
        else:
            r["status"] = "PASS" if value <= thr else "FAIL"
            if r["status"] == "PASS" and need_ticks and not have_ticks:
                r["status"], r["reason"] = "UNMEASURED", f"{n_have} measured ticks < {n_need} required"
            elif r["status"] == "PASS" and need_fast and not fast_ok:
                r["status"], r["reason"] = "UNMEASURED", f"{fast_n} concurrent fast-path ticks < {pr['min_fast_ticks']} required"
            if mock:
                r["status"], r["reason"] = "MOCK_ONLY", "mock backend: numbers are arithmetic, not measurements"
        rows[name] = r

    rp95, rmean, hz = m.get("refresh_p95_ms"), m.get("refresh_mean_ms"), m.get("refresh_hz")
    add("refresh_sync_p95_le_50ms", rp95, pr["refresh_sync_p95_ms"], "ms", ["synchronous"],
        "refresh p95 at the chosen camera count (sec_protocol.tex H-HC11 names ONE camera)")
    add("refresh_async_p95_le_250ms", rp95, pr["refresh_async_p95_ms"], "ms", ["asynchronous"],
        "refresh p95 (asynchronous reading; threshold INHERITED from the brief)")
    age_thr = None if rp95 is None else pr["token_age_slack_ms"] + rp95
    add("token_age_p95_le_500ms_plus_refresh_p95", m.get("token_age_p95_ms"), age_thr, "ms", ["asynchronous"],
        "p95 token age from the timestamped tick schedule <= 500 ms + refresh p95")
    duty = None if (rmean is None or hz is None) else (rmean / 1e3) * hz
    add("duty_cycle_le_0.5", duty, pr["duty_cycle_max"], "fraction", ["asynchronous"],
        "duty cycle = mean refresh time x refresh_hz")
    add("fastpath_concurrent_p95_le_25ms", m.get("fast_concurrent_p95_ms"), pr["fastpath_p95_ms"], "ms",
        ["synchronous", "asynchronous"],
        "fast-path PROXY p95 (CUDA events) measured WHILE a refresh runs on the same GPU; all fast ticks of the "
        "concurrent phase", need_fast=True)
    fp95, allowance = m.get("fast_concurrent_p95_ms"), m.get("preprocess_allowance_ms")
    e2e = None if (fp95 is None or allowance is None) else fp95 + allowance
    add("e2e_command_latency_proxy_le_tick_budget", e2e, m.get("tick_budget_ms"), "ms", ["synchronous", "asynchronous"],
        "camera-frame-to-command PROXY = fast-path concurrent p95 + fixed preprocessing allowance (ASSUMPTION, "
        "--preprocess-allowance-ms); excludes exposure/ISP/transport/actuation", need_fast=True)
    add("supplementary_fastpath_p95_overlapped_only_le_25ms", m.get("fast_overlapped_p95_ms"), pr["fastpath_p95_ms"], "ms",
        [], "SUPPLEMENTARY (not pre-registered): p95 over only the fast ticks that overlapped a refresh; the all-ticks p95 "
        "is diluted by the non-overlapped share; needs >= 100 overlapped ticks to be more than an anecdote", gating=False,
        need_fast=True, n_for_ticks=m.get("fast_overlapped_n"), min_n=100)
    add("supplementary_refresh_p95_under_concurrency_le_250ms", m.get("refresh_concurrent_p95_ms"),
        pr["refresh_async_p95_ms"], "ms", [], "SUPPLEMENTARY: refresh p95 with the fast path live", gating=False,
        n_for_ticks=m.get("refresh_concurrent_n"))

    on_thor = m.get("on_thor")
    if mock:
        why = "mock backend"
    elif on_thor is not True:
        why = "hardware not confirmed as Thor (H-HC11 is defined on Thor)"
    elif not ticks_ok:
        why = f"fewer than {pr['min_ticks']} measured ticks"
    else:
        why = "MEASURED on Thor with >= min ticks"
    readings = {}
    for reading in ("synchronous", "asynchronous"):
        names = [n for n, r in rows.items() if reading in r["readings"] and r["gating"]]
        overall = "MOCK_ONLY" if mock else combine_status([rows[n]["status"] for n in names])
        # a reading DECIDES only if the run is real, on Thor, long enough, and no gating row of that reading is UNMEASURED
        decides = (not mock) and (on_thor is True) and ticks_ok and overall in ("PASS", "FAIL")
        readings[reading] = {"rows": names, "overall": overall, "decides_h_hc11": bool(decides)}
    return {
        "thresholds": pr,
        "threshold_provenance": {
            "refresh_sync_p95_ms, fastpath_p95_ms, min_ticks": "Paper/HiCAP/sec_protocol.tex row H-HC11",
            "refresh_async_p95_ms, token_age_slack_ms, duty_cycle_max, min_fast_ticks": "implementation brief (INHERITED; not in the sec_protocol.tex row)",
            "preprocess_allowance_ms": "ASSUMPTION (CLI argument)",
        },
        "evidence_class": evidence_class,
        "rows": rows,
        "readings": readings,
        "drift": {"status": "UNMEASURED", "reason": "a single run cannot measure drift: run --save-features for BF16 and for "
                  "the NVFP4 (TensorRT-Edge-LLM) features, then --compare-features; no numeric margin is pre-registered"},
        "decides_h_hc11_gate": why,
        "note": ("the synchronous and asynchronous readings are alternative deployments, so there is deliberately no combined "
                 "overall status: the reading that matches the deployed schedule decides; both exclude drift "
                 "(H-HC11 needs fast path AND refresh AND drift)"),
    }


# ======================================================================================
# 5. Feature files and drift comparison
# ======================================================================================
def frames_fingerprint(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(a.tobytes())
    return h.hexdigest()


def save_features(path, meta, feats):
    """feats: {'tap_0': ndarray[W,P,D], ...} float32.  .npz -> numpy only; .pt -> torch.save."""
    p = str(path)
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    if p.endswith(".npz"):
        import numpy as np
        np.savez(p, meta_json=np.array(json.dumps(meta)), **feats)
    elif p.endswith(".pt"):
        try:
            import torch
        except Exception as e:  # noqa: BLE001
            raise UsageError(f"error: --save-features {p}: writing .pt needs torch ({e}); use a .npz path or run where torch exists")
        torch.save({"meta": meta, "features": {k: torch.from_numpy(v) for k, v in feats.items()}}, p)
    else:
        raise UsageError(f"error: --save-features must end in .pt or .npz, got {p}")


def load_features(path):
    p = str(path)
    if p.endswith(".npz"):
        import numpy as np
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta_json"]))
        return meta, {k: np.asarray(z[k]) for k in z.files if k != "meta_json"}
    if p.endswith(".pt"):
        try:
            import torch
        except Exception as e:  # noqa: BLE001
            raise UsageError(f"error: reading {p} needs torch ({e}); re-save the features as .npz or run where torch exists")
        d = torch.load(p, map_location="cpu", weights_only=False)
        return d["meta"], {k: v.float().numpy() for k, v in d["features"].items()}
    raise UsageError(f"error: feature file must end in .pt or .npz, got {p}")


def compare_pooled(ref, cand):
    """Per-tap cosine similarity and relative L2 drift of pooled features (ref = the BF16 arm, cand = the other)."""
    import numpy as np
    if sorted(ref) != sorted(cand):
        raise ValueError(f"tap keys differ: {sorted(ref)} vs {sorted(cand)}")
    out = {}
    for key in sorted(ref):
        a = np.asarray(ref[key], dtype=np.float64)
        b = np.asarray(cand[key], dtype=np.float64)
        if a.shape != b.shape or a.ndim != 3:
            raise ValueError(f"{key}: shape mismatch or not [windows, tokens, width]: {a.shape} vs {b.shape}")
        eps = 1e-12
        cos_tok = (a * b).sum(-1) / np.maximum(np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), eps)
        af, bf = a.reshape(a.shape[0], -1), b.reshape(b.shape[0], -1)
        cos_flat = (af * bf).sum(1) / np.maximum(np.linalg.norm(af, axis=1) * np.linalg.norm(bf, axis=1), eps)
        rel = np.linalg.norm(af - bf, axis=1) / np.maximum(np.linalg.norm(af, axis=1), eps)
        out[key] = {
            "n_windows": int(a.shape[0]), "n_tokens": int(a.shape[1]), "width": int(a.shape[2]),
            "cos_token_mean": float(cos_tok.mean()), "cos_token_p05": float(percentile(cos_tok.ravel().tolist(), 5)),
            "cos_token_min": float(cos_tok.min()), "cos_flat_mean": float(cos_flat.mean()),
            "cos_flat_min": float(cos_flat.min()), "rel_l2_mean": float(rel.mean()), "rel_l2_max": float(rel.max()),
        }
    return out


def drift_verdict(per_tap, meta_a, meta_b, margin_cos=None, margin_rel_l2=None):
    """No numeric margin is pre-registered; without --drift-margin-* the verdict is NO_MARGIN_PREREGISTERED."""
    reasons = []
    frames_a, frames_b = meta_a.get("frames_source"), meta_b.get("frames_source")
    fp_a, fp_b = meta_a.get("frames_fingerprint"), meta_b.get("frames_fingerprint")
    comparable = (fp_a is not None) and (fp_a == fp_b)
    if not comparable:
        reasons.append("frames_fingerprint differs or is missing: the two runs did not see identical frames; drift is meaningless")
    synthetic = "synthetic" in (frames_a, frames_b)
    if synthetic:
        reasons.append("synthetic frames were used in at least one run: INADMISSIBLE_FOR_DRIFT (noise features are unrepresentative)")
    mock = "MOCK" in (meta_a.get("evidence_class"), meta_b.get("evidence_class"))
    if mock:
        reasons.append("at least one file is from a MOCK run")
    if margin_cos is None and margin_rel_l2 is None:
        status = "NO_MARGIN_PREREGISTERED"
        reasons.append("no numeric margin is pre-registered (sec_protocol.tex says 'within the margin'); pass --drift-margin-cos/--drift-margin-rel-l2 explicitly (ASSUMPTION) to get PASS/FAIL")
    else:
        ok = True
        if margin_cos is not None:
            ok &= all(v["cos_token_mean"] >= margin_cos for v in per_tap.values())
        if margin_rel_l2 is not None:
            ok &= all(v["rel_l2_mean"] <= margin_rel_l2 for v in per_tap.values())
        status = "PASS" if ok else "FAIL"
    if mock:
        status = "MOCK_ONLY"
    elif synthetic:
        status = "INADMISSIBLE_FOR_DRIFT"
    elif not comparable:
        status = "INADMISSIBLE_FRAMES_DIFFER"
    return {"status": status, "margin_cos_token_mean_ge": margin_cos, "margin_rel_l2_mean_le": margin_rel_l2,
            "margin_provenance": "ASSUMPTION (CLI)" if (margin_cos is not None or margin_rel_l2 is not None) else "none pre-registered",
            "reasons": reasons}


# ======================================================================================
# 6. Environment stamp
# ======================================================================================
def _read_text(path):
    try:
        return Path(path).read_text(errors="replace").replace("\x00", "").strip()
    except Exception:  # noqa: BLE001
        return None


def collect_stamp(torch=None, transformers=None):
    st = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
          "omp_num_threads": os.environ.get("OMP_NUM_THREADS"), "pythonpath": os.environ.get("PYTHONPATH"),
          "jetson_device_tree_model": _read_text("/proc/device-tree/model") or "UNKNOWN"}
    try:
        r = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=5)
        txt = (r.stdout or r.stderr or "").strip()
        st["nvpmodel"] = txt if (r.returncode == 0 and txt) else f"UNKNOWN (nvpmodel exit {r.returncode}: {txt[:120]})"
    except Exception as e:  # noqa: BLE001
        st["nvpmodel"] = f"UNKNOWN ({type(e).__name__})"
    st["power_mode_note"] = "recorded only; this script never sets the power mode or clocks"
    st["optional_kernels"] = {n: (importlib.util.find_spec(n) is not None) for n in ("fla", "causal_conv1d", "flash_attn")}
    if torch is not None:
        st["torch"] = torch.__version__
        st["cuda"] = torch.version.cuda
        try:
            st["cudnn"] = torch.backends.cudnn.version()
        except Exception:  # noqa: BLE001
            st["cudnn"] = None
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(dev)
            st["device_name"] = torch.cuda.get_device_name(dev)
            st["device_capability"] = list(torch.cuda.get_device_capability(dev))
            st["sm_count"] = props.multi_processor_count
    else:
        st["torch"] = "NOT_IMPORTED"
    st["transformers"] = getattr(transformers, "__version__", "NOT_IMPORTED") if transformers is not None else "NOT_IMPORTED"
    hay = f"{st['jetson_device_tree_model']} {st.get('device_name', '')}".lower()
    st["on_thor"] = True if "thor" in hay else (None if (st["jetson_device_tree_model"] == "UNKNOWN" and "device_name" not in st) else False)
    return st


# ======================================================================================
# 7. Mock backend (no torch) - deterministic pseudo-latencies, labelled MOCK
# ======================================================================================
class BackendUnavailable(RuntimeError):
    pass


class MockBackend:
    """Deterministic arithmetic stand-in.  NOT a measurement.

    refresh:  vision = n_images * vit_ms_per_image * m1
              llm    = (llm_base_ms + llm_ms_per_token * (params_b/4) * llm_tokens) * m1
              pool   = (0.05 + 2e-5 * llm_tokens * n_taps) * m1 ;  residual = 0.3 * m1
              m1 = max(0.5, 1 + N(0, jitter)) ; with prob spike_prob m1 *= 2 ; the first 5 calls (cold start) m1 *= 3
    fast path: gpu = fast_base_ms * m2 * (1 + alpha * ov),  ov = overlap([f, f+base], refresh busy)/base in [0,1],
               m2 = max(0.5, 1 + N(0, jitter)) ; wall = gpu + launch_ms
    Defaults are illustrative orders of magnitude only."""
    virtual = True
    evidence_class = "MOCK"
    name = "mock"

    def __init__(self, cfg, clock):
        self.cfg, self.clock = cfg, clock
        self.rng = random.Random(cfg.mock_seed)
        self.fast_rng = random.Random(cfg.mock_seed + 1)
        self.jit_rng = random.Random(cfg.mock_seed + 2)
        self.calls = 0
        self.tb = token_budget(cfg.cameras, cfg.frames_per_camera, cfg.height, cfg.width, cfg.prompt_tokens)
        self.params_b = cfg.params_b
        self.n_layers = 36
        self.taps = resolve_taps(cfg.taps, self.n_layers)

    def setup(self):
        pooled = pooled_token_counts(self.tb["llm_input_tokens"], self.cfg.pool_tokens, len(self.taps))
        return {
            "stamp": {"mock": True, "python": sys.version.split()[0], "platform": platform.platform(), "torch": "NOT_IMPORTED",
                      "transformers": "NOT_IMPORTED", "device_name": "MOCK", "nvpmodel": "UNKNOWN (mock)", "on_thor": None,
                      "omp_num_threads": os.environ.get("OMP_NUM_THREADS")},
            "model": {"key": self.cfg.model_key, "hf_id": self.cfg.hf_id, "weight_bytes": None, "weight_gb": None,
                      "weight_note": "MOCK: no weights loaded", "num_hidden_layers": self.n_layers, "hidden_size": 64,
                      "taps": self.taps, "tap_convention": TAP_CONVENTION + " [mock: nominal 36 layers]"},
            "tokens": {**self.tb, **pooled, "visual_tokens_actual": self.tb["visual_tokens"],
                       "llm_input_tokens_actual": self.tb["llm_input_tokens"], "basis": "arithmetic (mock; no processor run)"},
            "processor": {"patch_size": 16, "merge_size": 2, "note": "MOCK: constants, not read from a processor"},
            "notes": [],
        }

    def reset_peak(self):
        return None

    def warmup(self, n):
        for i in range(n):
            self.refresh(-1 - i)

    def refresh(self, slot):
        c = self.cfg
        self.calls += 1
        m1 = max(0.5, 1.0 + self.rng.gauss(0.0, c.mock_jitter))
        if self.rng.random() < c.mock_spike_prob:
            m1 *= 2.0
        if self.calls <= 5:
            m1 *= 3.0
        scale = self.params_b / 4.0
        vision = self.tb["images_per_tick"] * c.mock_vit_ms_per_image * m1
        llm = (c.mock_llm_base_ms + c.mock_llm_ms_per_token * scale * self.tb["llm_input_tokens"]) * m1
        pool = (0.05 + 2e-5 * self.tb["llm_input_tokens"] * len(self.taps)) * m1
        resid = 0.3 * m1
        return {"vision_ms": vision, "llm_prefill_ms": llm, "pool_ms": pool, "residual_ms": resid,
                "total_ms": vision + llm + pool + resid}

    def peak_memory_bytes(self):
        return None

    def inadmissible_memory(self):
        return None

    def _fast_ms(self, f, busy):
        c = self.cfg
        m2 = max(0.5, 1.0 + self.fast_rng.gauss(0.0, c.mock_jitter))
        ov = min(1.0, overlap_ms(f, f + c.mock_fast_base_ms, busy) / c.mock_fast_base_ms) if busy else 0.0
        return c.mock_fast_base_ms * m2 * (1.0 + c.mock_contention_alpha * ov)

    def fast_proxy_info(self):
        return {**PROXY_SPEC, "kind": "MOCK arithmetic model (no network executed)", "params": None,
                "model": "gpu = base*m2*(1+alpha*ov)", "base_ms": self.cfg.mock_fast_base_ms,
                "alpha": self.cfg.mock_contention_alpha}

    def _jit(self, period_ms):
        return self.jit_rng.uniform(-self.cfg.fast_jitter_frac, self.cfg.fast_jitter_frac) * period_ms

    def fast_alone(self, n_ticks, period_ms):
        t0 = self.clock.now_ms()
        out = []
        for i in range(n_ticks):
            st = t0 + (i + 1) * period_ms + self._jit(period_ms)
            g = self._fast_ms(st, [])
            out.append({"start_ms": st, "end_ms": st + g, "gpu_ms": g, "wall_ms": g + self.cfg.mock_launch_ms})
        self.clock.sleep_until_ms(t0 + (n_ticks + 1) * period_ms)
        return out, {"missed": 0}

    def fast_concurrent(self, run, period_ms):
        t_begin = self.clock.now_ms()
        result = run()
        t_end = self.clock.now_ms()
        busy = [(r["start_ms"], r["end_ms"]) for r in result[0]]
        out, i = [], 1
        while t_begin + i * period_ms <= t_end:
            st = t_begin + i * period_ms + self._jit(period_ms)
            g = self._fast_ms(st, busy)
            out.append({"start_ms": st, "end_ms": st + g, "gpu_ms": g, "wall_ms": g + self.cfg.mock_launch_ms})
            i += 1
        return result, out, {"missed": 0}

    def collect_features(self):
        try:
            import numpy as np
        except Exception as e:  # noqa: BLE001
            raise UsageError(f"error: --save-features with the mock backend needs numpy ({e})")
        n_win, per_tap, width = 2, min(self.tb["llm_input_tokens"], self.cfg.pool_tokens), 64
        feats = {}
        for t_i, _ in enumerate(self.taps):
            rng = np.random.default_rng([self.cfg.seed, t_i])
            feats[f"tap_{t_i}"] = rng.standard_normal((n_win, per_tap, width)).astype(np.float32)
        fp = hashlib.sha256(f"mock-frames-seed-{self.cfg.seed}".encode()).hexdigest()
        return {"frames_fingerprint": fp, "frames_source": "synthetic", "n_windows": n_win}, feats

    def frames_info(self):
        fp = hashlib.sha256(f"mock-frames-seed-{self.cfg.seed}".encode()).hexdigest()
        return {"source": "synthetic", "seed": self.cfg.seed, "fingerprint_sha256": fp, "n_windows": 2,
                "note": "MOCK: no pixels generated", "drift_label": "INADMISSIBLE_FOR_DRIFT"}


# ======================================================================================
# 8. HF backend (UNVERIFIED - never executed; requires torch + CUDA + transformers)
# ======================================================================================
def _param_bytes(module):
    n = sum(p.numel() * p.element_size() for p in module.parameters())
    n += sum(b.numel() * b.element_size() for b in module.buffers())
    return int(n)


class HFBackend:
    """Real backend.  Imports torch/transformers lazily; raises BackendUnavailable (never mocks) if they are absent."""
    virtual = False
    evidence_class = "MEASURED"
    name = "hf"

    def __init__(self, cfg, clock):
        self.cfg, self.clock = cfg, clock
        try:
            import torch
        except Exception as e:  # noqa: BLE001
            raise BackendUnavailable(
                f"--backend hf needs torch, which failed to import ({type(e).__name__}: {e}). Refusing to fall back to the mock "
                "backend: a mock number must never be mistaken for a measurement. Run this on the Jetson Thor "
                "(or use --backend mock explicitly to exercise the harness).") from e
        if not torch.cuda.is_available():
            raise BackendUnavailable("--backend hf needs a CUDA device (CUDA events are the only admissible timer); "
                                     "torch.cuda.is_available() is False. Refusing to time on CPU.")
        self.torch = torch
        self.device = torch.device("cuda")
        self._cur = None
        self._taps = None
        self._proxy = None
        self.windows = []

    # ---- setup -----------------------------------------------------------------------------------
    def setup(self):
        import inspect
        torch, cfg = self.torch, self.cfg
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor
        try:
            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", cfg.omp_threads)))
        except Exception:  # noqa: BLE001
            pass
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[cfg.dtype]
        notes = []
        load_kw = dict(attn_implementation=cfg.attn, trust_remote_code=cfg.trust_remote_code)
        try:
            model = AutoModelForImageTextToText.from_pretrained(cfg.hf_id, dtype=dtype, **load_kw)
        except TypeError:   # transformers < 4.56 spells it torch_dtype; the dtype guard below still catches a silent upcast
            notes.append("from_pretrained(dtype=) raised TypeError; retried with torch_dtype=")
            model = AutoModelForImageTextToText.from_pretrained(cfg.hf_id, torch_dtype=dtype, **load_kw)
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        by_dtype = {}
        for p in model.parameters():
            by_dtype[str(p.dtype)] = by_dtype.get(str(p.dtype), 0) + p.numel() * p.element_size()
        tot = sum(by_dtype.values())
        if by_dtype.get(str(dtype), 0) < 0.9 * tot:
            raise RuntimeError(f"loaded parameter bytes by dtype {by_dtype}; fewer than 90% are {dtype}. The `dtype=` kwarg may have "
                               "been ignored by this transformers version (silent fp32 upcast); refusing to time it.")
        weight_full = _param_bytes(model)
        core = getattr(model, "model", None)
        if core is None:
            core = model
        vis, llm = getattr(core, "visual", None), getattr(core, "language_model", None)
        if vis is None or llm is None:
            raise RuntimeError("cannot locate core.visual / core.language_model (expected as in Qwen3VLModel / Qwen3_5Model, "
                               "transformers 5.17 source). Refusing to fold vision+LLM into one stage silently.")
        lm_head_dropped = False
        if not cfg.keep_lm_head and getattr(model, "lm_head", None) is not None:
            model.lm_head = None
            lm_head_dropped = True
        self.model, self.core, self.vis, self.llm = model, core, vis, llm
        for obj in (model.config, getattr(model.config, "text_config", None), getattr(llm, "config", None)):
            if obj is not None:
                try:
                    obj.use_cache = False   # HF resolves use_cache from self.config; a call-time kwarg would leak into the vision blocks
                except Exception:  # noqa: BLE001
                    pass
        weight_core = _param_bytes(core)
        self._install_hooks()
        text_cfg = getattr(model.config, "text_config", model.config)
        self.n_layers_cfg = int(getattr(text_cfg, "num_hidden_layers", 0) or 0)
        hidden = int(getattr(text_cfg, "hidden_size", 0) or 0)
        # processor + token guard
        self.processor = AutoProcessor.from_pretrained(cfg.hf_id, trust_remote_code=cfg.trust_remote_code)
        ip = self.processor.image_processor
        patch, merge = getattr(ip, "patch_size", None), getattr(ip, "merge_size", None)
        if not patch or not merge:
            raise RuntimeError(f"cannot read patch_size/merge_size from the processor's image_processor ({type(ip).__name__})")
        factor = patch * merge
        try:
            ip.size = {"shortest_edge": MIN_TOKENS_PER_IMAGE * factor ** 2, "longest_edge": MAX_TOKENS_PER_IMAGE * factor ** 2}
        except Exception as e:  # noqa: BLE001
            notes.append(f"could not assign image_processor.size ({e}); relying on the processor defaults + the token-count guard")
        self.image_token_id = getattr(self.processor, "image_token_id", None) or getattr(model.config, "image_token_id", None)
        if self.image_token_id is None:
            raise RuntimeError("cannot determine image_token_id from the processor or model config")
        self.fwd_params = set(inspect.signature(core.forward).parameters)
        self._build_windows(patch, merge)
        exp = token_budget(cfg.cameras, cfg.frames_per_camera, cfg.height, cfg.width, cfg.prompt_tokens, patch, merge)
        w0 = self.windows[0]["cpu"]
        vis_actual = int((w0["input_ids"] == self.image_token_id).sum().item())
        all_actual = int(w0["input_ids"].shape[1])
        if vis_actual != exp["visual_tokens"]:
            raise RuntimeError(f"image-token count {vis_actual} != arithmetic {exp['visual_tokens']} "
                               f"({exp['images_per_tick']} images x {exp['visual_tokens_per_image']}). The processor resized the frames; "
                               "fix --resolution / processor size before timing (see the Cosmos-Reason2 min-pixels pitfall in the header).")
        self.tb = exp
        torch.cuda.synchronize()
        notes += [
            "output_hidden_states=True is forwarded by HF core.forward(**kwargs) to the vision tower as well (references only, no copies): "
            "expected negligible extra cost inside vision_ms; UNVERIFIED",
            "pixel_values are pre-cast to the model dtype and (unless --include-h2d) resident on the device before the timed region",
            "CPU preprocessing (PIL resize + processor) is NOT in total_ms; see refresh.cpu_preprocess_ms",
            "frame_mode=images: every frame is an independent image; no ViT-feature cache, no video temporal merge",
        ]
        if lm_head_dropped:
            notes.append("lm_head dropped (prefill-only design); if tied to embed_tokens this frees no memory")
        self._info = {
            "stamp": collect_stamp(torch, transformers),
            "model": {"key": cfg.model_key, "hf_id": cfg.hf_id, "weight_bytes": weight_core, "weight_gb": weight_core / 1e9,
                      "weight_bytes_full_checkpoint": weight_full, "param_bytes_by_dtype": by_dtype,
                      "weight_note": "bytes of parameters+buffers of the multimodal core (vision tower + language model incl. embed_tokens), "
                                     "lm_head excluded when dropped", "lm_head_dropped": lm_head_dropped,
                      "num_hidden_layers_config": self.n_layers_cfg, "hidden_size": hidden, "class": type(model).__name__,
                      "attn_implementation": cfg.attn, "dtype": cfg.dtype},
            "tokens": {**exp, "visual_tokens_actual": vis_actual, "llm_input_tokens_actual": all_actual, "basis": "measured from the processor output",
                       "prompt_tokens_actual_incl_markers_and_template": all_actual - vis_actual},
            "processor": {"class": type(self.processor).__name__, "image_processor": type(ip).__name__, "patch_size": patch,
                          "merge_size": merge, "temporal_patch_size": getattr(ip, "temporal_patch_size", None),
                          "size_override_tokens_per_image": [MIN_TOKENS_PER_IMAGE, MAX_TOKENS_PER_IMAGE]},
            "notes": notes,
        }
        return self._info

    def _install_hooks(self):
        torch = self.torch

        def mk(name):
            def pre(module, args):
                if self._cur is None:
                    return
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                self._cur[name].append([e, None])

            def post(module, args, output):
                if self._cur is None:
                    return
                e = torch.cuda.Event(enable_timing=True)
                e.record()
                self._cur[name][-1][1] = e
            return pre, post
        for name, mod in (("vision", self.vis), ("llm", self.llm)):
            pre, post = mk(name)
            mod.register_forward_pre_hook(pre)
            mod.register_forward_hook(post)

    # ---- frames and inputs -----------------------------------------------------------------------
    def _load_frames(self):
        import numpy as np
        from PIL import Image
        cfg = self.cfg
        per_window = cfg.cameras * cfg.frames_per_camera
        H, W = cfg.height, cfg.width
        if cfg.frames_dir:
            files = sorted(p for p in Path(cfg.frames_dir).iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
            if len(files) < per_window:
                raise UsageError(f"error: --frames-dir has {len(files)} PNG/JPG frames; need >= cameras*frames_per_camera = {per_window}")
            n_win = max(1, min(cfg.max_windows, len(files) // per_window))
            wins, frame_ids = [], []
            for w in range(n_win):
                imgs = []
                for f in files[w * per_window:(w + 1) * per_window]:
                    im = Image.open(f).convert("RGB")
                    if im.size != (W, H):
                        im = im.resize((W, H), Image.BICUBIC)
                    imgs.append(im)
                    frame_ids.append(f.name)
                wins.append(imgs)
            source = "real"
        else:
            rng = np.random.default_rng(cfg.seed)
            wins = [[Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8)) for _ in range(per_window)]]
            frame_ids, source = [f"synthetic_noise_seed{cfg.seed}_{i}" for i in range(per_window)], "synthetic"
        fp = frames_fingerprint([np.asarray(im, dtype=np.uint8) for w in wins for im in w])
        self._frames_meta = {"source": source, "seed": cfg.seed if source == "synthetic" else None, "fingerprint_sha256": fp,
                             "n_windows": len(wins), "frames_dir": str(cfg.frames_dir) if cfg.frames_dir else None,
                             "frame_ids_first_window": frame_ids[:per_window],
                             "drift_label": "INADMISSIBLE_FOR_DRIFT" if source == "synthetic" else "ADMISSIBLE_FOR_DRIFT_IF_SAME_FRAMES_IN_BOTH_RUNS"}
        return wins

    def _process(self, images, n_filler):
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": " ".join(["road"] * max(1, n_filler))}]
        text = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        return self.processor(text=[text], images=images, return_tensors="pt")

    def _build_windows(self, patch, merge):
        torch, cfg = self.torch, self.cfg
        wins = self._load_frames()
        # calibrate the filler so that (all tokens - image tokens) ~ prompt_tokens, on window 0
        n_fill = max(1, cfg.prompt_tokens // 2)
        for _ in range(3):
            enc = self._process(wins[0], n_fill)
            nonvis = int(enc["input_ids"].shape[1]) - int((enc["input_ids"] == self.image_token_id).sum().item())
            if nonvis == cfg.prompt_tokens or (nonvis > cfg.prompt_tokens and n_fill == 1):
                break
            n_fill = max(1, n_fill + (cfg.prompt_tokens - nonvis))
        self.windows = []
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[cfg.dtype]
        for imgs in wins:
            enc = self._process(imgs, n_fill)
            d = {k: v for k, v in enc.items() if hasattr(v, "to")}
            if "pixel_values" in d:
                d["pixel_values"] = d["pixel_values"].to(dtype)
            if "mm_token_type_ids" in self.fwd_params and "mm_token_type_ids" not in d:
                d["mm_token_type_ids"] = (d["input_ids"] == self.image_token_id).long()
            cpu = {k: (v.pin_memory() if cfg.include_h2d else v) for k, v in d.items()}
            dev = {k: v.to(self.device) for k, v in d.items()}
            self.windows.append({"cpu": cpu, "dev": dev})
        # CPU preprocessing wall time (CPU-only work, so perf_counter is legitimate here)
        t = []
        for _ in range(min(10, 3 + len(wins))):
            t0 = time.perf_counter()
            self._process(wins[0], n_fill)
            t.append((time.perf_counter() - t0) * 1e3)
        self._cpu_pre = summarize(t)

    # ---- one refresh -----------------------------------------------------------------------------
    def _pool(self, hs):
        F = self.torch.nn.functional
        outs = []
        for t in self._taps:
            h = hs[t]
            k = min(int(h.shape[1]), self.cfg.pool_tokens)
            outs.append(F.adaptive_avg_pool1d(h.transpose(1, 2), k).transpose(1, 2))
        return outs

    def _ensure_taps(self, hs):
        if self._taps is None:
            n_obs = len(hs) - 1
            if self.n_layers_cfg and n_obs != self.n_layers_cfg:
                self._info["notes"].append(f"hidden_states length {len(hs)} != num_hidden_layers+1 ({self.n_layers_cfg + 1}); taps derived from the OBSERVED length")
            self._taps = resolve_taps(self.cfg.taps, n_obs)
            self._n_obs_layers = n_obs
            self._info["model"]["taps"] = self._taps
            self._info["model"]["hidden_states_len"] = len(hs)
            self._info["model"]["tap_convention"] = TAP_CONVENTION
            self._info["tokens"].update(pooled_token_counts(self._info["tokens"]["llm_input_tokens_actual"], self.cfg.pool_tokens, len(self._taps)))

    def _forward(self, w):
        with self.torch.inference_mode():
            inputs = w["dev"]
            out = self.core(**inputs, output_hidden_states=True)
            hs = getattr(out, "hidden_states", None)
            if hs is None:
                raise RuntimeError("model returned no hidden_states with output_hidden_states=True")
            self._ensure_taps(hs)
            return hs

    def refresh(self, slot):
        torch = self.torch
        w = self.windows[slot % len(self.windows)]
        ev = lambda: torch.cuda.Event(enable_timing=True)  # noqa: E731
        e_t0, e_p0, e_p1 = ev(), ev(), ev()
        self._cur = {"vision": [], "llm": []}
        try:
            with torch.inference_mode():
                e_t0.record()
                if self.cfg.include_h2d:
                    inputs = {k: v.to(self.device, non_blocking=True) for k, v in w["cpu"].items()}
                else:
                    inputs = w["dev"]
                out = self.core(**inputs, output_hidden_states=True)
                hs = getattr(out, "hidden_states", None)
                if hs is None:
                    raise RuntimeError("model returned no hidden_states with output_hidden_states=True")
                self._ensure_taps(hs)
                e_p0.record()
                self._pool(hs)
                e_p1.record()
            e_p1.synchronize()
            pairs = self._cur
        finally:
            self._cur = None
        vision = sum(s.elapsed_time(e) for s, e in pairs["vision"] if e is not None)
        llm = sum(s.elapsed_time(e) for s, e in pairs["llm"] if e is not None)
        pool = e_p0.elapsed_time(e_p1)
        total = e_t0.elapsed_time(e_p1)
        return {"vision_ms": vision, "llm_prefill_ms": llm, "pool_ms": pool, "residual_ms": total - vision - llm - pool,
                "total_ms": total}

    def warmup(self, n):
        for i in range(n):
            self.refresh(i)
        self.torch.cuda.synchronize()

    def reset_peak(self):
        self.torch.cuda.reset_peak_memory_stats()

    def peak_memory_bytes(self):
        return int(self.torch.cuda.max_memory_allocated())

    def inadmissible_memory(self):
        torch = self.torch
        out = {"label": "INADMISSIBLE_ON_THOR"}
        try:
            free, total = torch.cuda.mem_get_info()
            out["mem_get_info_free_bytes"], out["mem_get_info_total_bytes"] = int(free), int(total)
        except Exception as e:  # noqa: BLE001
            out["mem_get_info_error"] = str(e)
        txt = _read_text("/proc/self/status") or ""
        mt = re.search(r"VmRSS:\s+(\d+)\s+kB", txt)
        out["VmRSS_bytes"] = int(mt.group(1)) * 1024 if mt else None
        return out

    # ---- fast-path proxy -------------------------------------------------------------------------
    def _build_proxy(self):
        torch = self.torch
        nn, F = torch.nn, torch.nn.functional
        d, L, H, ffn, S = (PROXY_SPEC[k] for k in ("d_model", "layers", "heads", "ffn", "tokens"))
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[self.cfg.dtype]

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.n1, self.qkv, self.proj = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
                self.n2, self.fc1, self.fc2 = nn.LayerNorm(d), nn.Linear(d, ffn), nn.Linear(ffn, d)

            def forward(self, x):
                b, n, _ = x.shape
                q, k, v = self.qkv(self.n1(x)).view(b, n, 3, H, d // H).permute(2, 0, 3, 1, 4)
                a = F.scaled_dot_product_attention(q, k, v)
                x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
                return x + self.fc2(F.gelu(self.fc1(self.n2(x))))

        class Proxy(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([Block() for _ in range(L)])
                self.out = nn.LayerNorm(d)

            def forward(self, x):
                for blk in self.blocks:
                    x = blk(x)
                return self.out(x)

        torch.manual_seed(self.cfg.seed)
        self._proxy = Proxy().to(self.device, dtype).eval()
        g = torch.Generator(device=self.device).manual_seed(self.cfg.seed)
        self._proxy_x = torch.randn(PROXY_SPEC["batch"], S, d, generator=g, device=self.device, dtype=dtype)
        torch.cuda.synchronize()
        n_params = sum(p.numel() for p in self._proxy.parameters())
        self._proxy_info = {**PROXY_SPEC, "kind": "SYNTHETIC PROXY (NOT the HiCAP fusion transformer): pre-norm blocks, random weights",
                            "params": int(n_params), "dtype": self.cfg.dtype, "seed": self.cfg.seed,
                            "approx_gflop": 2.0 * n_params * S / 1e9, "stream_priority": -1,
                            "refresh_stream": "default (priority 0 = lowest)"}

    def fast_proxy_info(self):
        return self._proxy_info if self._proxy is not None else {**PROXY_SPEC, "kind": "not built"}

    def _fast_loop(self, period_ms, n_ticks, stop, out, meta, ready=None):
        torch = self.torch
        try:
            if self._proxy is None:
                self._build_proxy()
            stream = torch.cuda.Stream(device=self.device, priority=-1)
            with torch.cuda.stream(stream), torch.inference_mode():
                for _ in range(20):
                    self._proxy(self._proxy_x)
            stream.synchronize()
            if ready is not None:
                ready.set()
            rng = random.Random(self.cfg.seed + 17)
            jf = self.cfg.fast_jitter_frac
            t_first = self.clock.now_ms() + period_ms
            i, done, missed = 0, 0, 0
            while not stop.is_set() and (n_ticks is None or done < n_ticks):
                target = t_first + i * period_ms + rng.uniform(-jf, jf) * period_ms   # nominal grid + per-tick jitter (mean rate kept)
                i += 1
                if target < self.clock.now_ms() - 1.0:      # that slot already passed while the previous tick ran
                    missed += 1
                    continue
                self.clock.sleep_until_ms(target)
                if stop.is_set():
                    break
                t_start = self.clock.now_ms()
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream), torch.inference_mode():
                    e0.record(stream)
                    self._proxy(self._proxy_x)
                    e1.record(stream)
                e1.synchronize()
                t_end = self.clock.now_ms()
                out.append({"start_ms": t_start, "end_ms": t_end, "gpu_ms": e0.elapsed_time(e1), "wall_ms": t_end - t_start})
                done += 1
            meta["missed"] = missed
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller
            meta["error"] = f"{type(e).__name__}: {e}"
        finally:
            if ready is not None:
                ready.set()

    def fast_alone(self, n_ticks, period_ms):
        out, meta = [], {}
        self._fast_loop(period_ms, n_ticks, threading.Event(), out, meta)
        if meta.get("error"):
            raise RuntimeError(f"fast-path (alone) failed: {meta['error']}")
        return out, meta

    def fast_concurrent(self, run, period_ms):
        stop, ready, out, meta = threading.Event(), threading.Event(), [], {}
        th = threading.Thread(target=self._fast_loop, args=(period_ms, None, stop, out, meta, ready), daemon=True)
        th.start()
        ready.wait(timeout=120)
        if meta.get("error"):
            th.join(timeout=5)
            raise RuntimeError(f"fast-path thread failed before the concurrent phase: {meta['error']}")
        try:
            result = run()
        finally:
            stop.set()
            th.join(timeout=30)
        if meta.get("error"):
            raise RuntimeError(f"fast-path thread failed: {meta['error']}")
        return result, out, meta

    # ---- features / frames -----------------------------------------------------------------------
    def collect_features(self):
        import numpy as np
        per_tap = [[] for _ in self._taps]
        for w in self.windows:
            hs = self._forward(w)
            with self.torch.inference_mode():
                for i, p in enumerate(self._pool(hs)):
                    per_tap[i].append(p[0].float().cpu().numpy())
        feats = {f"tap_{i}": np.stack(v).astype(np.float32) for i, v in enumerate(per_tap)}
        return {"frames_fingerprint": self._frames_meta["fingerprint_sha256"], "frames_source": self._frames_meta["source"],
                "n_windows": len(self.windows)}, feats

    def frames_info(self):
        return self._frames_meta


# ======================================================================================
# 9. Orchestration
# ======================================================================================
def _now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def run_benchmark(cfg, backend, clock):
    """Run phases B (refresh alone), A (fast alone), C (concurrent) and assemble the result dict."""
    info = backend.setup()
    backend.reset_peak()
    period = 1000.0 / cfg.refresh_hz
    fast_period = 1000.0 / cfg.fast_hz
    pace = not cfg.no_pace
    _log(f"[{backend.name}] warmup {cfg.warmup} ticks ...")
    backend.warmup(cfg.warmup)

    _log(f"[{backend.name}] phase B: {cfg.ticks} paced refresh ticks at {cfg.refresh_hz} Hz (alone) ...")
    recs_b, sched_b = run_refresh_schedule(clock, backend.refresh, cfg.ticks, period, pace, cfg.late_tol_ms)
    peak = backend.peak_memory_bytes()
    inad = backend.inadmissible_memory() if cfg.log_inadmissible_memory else None

    fast = None
    recs_c = sched_c = None
    fast_alone_ticks = conc_ticks = None
    meta_a = meta_c = {}
    if cfg.concurrent_fastpath:
        _log(f"[{backend.name}] phase A: fast path alone, {cfg.fast_baseline_ticks} ticks at {cfg.fast_hz} Hz ...")
        fast_alone_ticks, meta_a = backend.fast_alone(cfg.fast_baseline_ticks, fast_period)
        _log(f"[{backend.name}] phase C: {cfg.concurrent_ticks} refresh ticks with the fast path live ...")

        def run_c():
            return run_refresh_schedule(clock, backend.refresh, cfg.concurrent_ticks, period, pace, cfg.late_tol_ms)
        (recs_c, sched_c), conc_ticks, meta_c = backend.fast_concurrent(run_c, fast_period)
        peak_c = backend.peak_memory_bytes()
        busy_c = [(r["start_ms"], r["end_ms"]) for r in recs_c]
        alone = summarize_fast(fast_alone_ticks)
        conc = summarize_fast(conc_ticks, busy_c)
        a95 = alone["gpu_event_ms"]["p95"]
        c95 = conc["gpu_event_ms"]["p95"]
        o95 = conc["overlapped_with_refresh"]["gpu_event_ms"]["p95"]
        fast = {
            "proxy": backend.fast_proxy_info(), "fast_hz": cfg.fast_hz, "fast_tick_jitter_frac": cfg.fast_jitter_frac,
            "overlap_share_of_concurrent_ticks": (conc["overlapped_with_refresh"]["n"] / conc["n"]) if conc["n"] else None,
            "primary_timer": "CUDA events on the high-priority stream",
            "secondary_timer": "wall submit->complete (exposes CPU launch/GIL contention; harness artifact in Python)",
            "baseline_position": "measured after phase B (hot device), before phase C",
            "alone": {**alone, "missed_slots": meta_a.get("missed")},
            "concurrent": {**conc, "missed_slots": meta_c.get("missed"), "refresh_ticks": cfg.concurrent_ticks},
            "p95_inflation_ratio_all_ticks": (c95 / a95) if (a95 and c95 is not None) else None,
            "p95_inflation_ratio_overlapped_only": (o95 / a95) if (a95 and o95 is not None) else None,
            "wall_minus_event_p95_ms": ((conc["wall_ms"]["p95"] - c95) if c95 is not None else None),
            "harness_launch_overhead_suspected": bool(c95 is not None and conc["wall_ms"]["p95"] is not None
                                                      and conc["wall_ms"]["p95"] - c95 > 5.0),
            "peak_allocated_bytes_with_proxy": peak_c,
        }
    else:
        peak_c = None

    # tokens / age / duty
    ages_b = token_ages(recs_b, fast_period) if pace else []
    ages_c = token_ages(recs_c, fast_period) if (recs_c and pace) else []
    stages_b = stage_stats(recs_b)
    tot = stages_b["total_ms"]
    refresh = {
        "warmup_ticks": cfg.warmup, "n_ticks": len(recs_b), "stages": stages_b, "stability": stability(recs_b),
        "schedule": sched_b, "refresh_hz": cfg.refresh_hz, "period_ms": period,
        "cpu_preprocess_ms": getattr(backend, "_cpu_pre", None),
    }
    result = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "hypothesis": "H-HC11",
        "evidence_class": backend.evidence_class,
        "evidence_note": ("MOCK: arithmetic pseudo-latencies from MockBackend; NOT measurements; the verdict block refuses PASS/FAIL"
                          if backend.evidence_class == "MOCK" else
                          "MEASURED (ours): CUDA-event timings from this run; artifact = this JSON"),
        "created_utc": _now_utc(), "argv": sys.argv[1:],
        "script": {"path": str(Path(__file__).resolve()), "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "config": {
            "backend": backend.name, "model_key": cfg.model_key, "hf_id": cfg.hf_id, "cameras": cfg.cameras,
            "frames_per_camera": cfg.frames_per_camera, "resolution_hw": [cfg.height, cfg.width], "dtype": cfg.dtype,
            "ticks": cfg.ticks, "warmup": cfg.warmup, "taps": cfg.taps, "pool_tokens": cfg.pool_tokens,
            "refresh_hz": cfg.refresh_hz, "fast_hz": cfg.fast_hz, "paced": pace, "batch": 1,
            "frame_mode": "images", "vit_cache": False, "include_h2d": bool(getattr(cfg, "include_h2d", False)),
            "concurrent_fastpath": bool(cfg.concurrent_fastpath), "tick_budget_ms": cfg.tick_budget_ms,
            "preprocess_allowance_ms": cfg.preprocess_allowance_ms, "preprocess_allowance_provenance": "ASSUMPTION (CLI)",
            "percentile_method": "linear interpolation (numpy default)",
        },
        "stamp": info["stamp"], "model": info["model"], "tokens": info["tokens"], "processor": info.get("processor"),
        "frames": backend.frames_info(),
        "refresh": refresh,
        "token_age_ms": {"alone_phase": summarize(ages_b) if ages_b else None,
                         "concurrent_phase": summarize(ages_c) if ages_c else None,
                         "definition": "fast-tick time minus capture time of the newest READY token set; fast ticks on the "
                                       f"{cfg.fast_hz} Hz grid; needs a paced schedule"},
        "duty_cycle": {"mean_refresh_ms": tot["mean"], "refresh_hz": cfg.refresh_hz,
                       "value": (tot["mean"] / 1e3 * cfg.refresh_hz) if tot["mean"] is not None else None},
        "memory": {
            "headline_probe": "torch.cuda.max_memory_allocated() (in-process; the only admissible probe on Thor)",
            "peak_allocated_bytes": peak, "peak_allocated_gb": (peak / 1e9) if peak is not None else None,
            "gb_definition": "1e9 bytes", "weight_bytes": info["model"].get("weight_bytes"),
            "peak_window": "after model load (peak reset), through warmup and phase B; phase C peak (incl. fast-path proxy) under fast_path",
            "inadmissible_on_thor": inad,
        },
        "refresh_concurrent": ({"n_ticks": len(recs_c), "stages": stage_stats(recs_c), "schedule": sched_c} if recs_c else None),
        "fast_path": fast,
        "notes": list(info.get("notes", [])) + [
            "ticks are serially correlated (thermal/clock drift): p95 from one contiguous run is a POINT estimate with no interval; "
            "repeat the run for a decision-grade interval (block bootstrap over runs)",
            "fast-path concurrent p95 uses all fast ticks of phase C; refresh busy share (~duty cycle) dilutes it - see the "
            "supplementary overlapped-only row",
            "baseline (phase A) is measured on a warm device after phase B, not cold",
        ] + ([
            "concurrency relies on PyTorch creating side streams non-blocking (so they do not serialise against the legacy default "
            "stream): recalled from c10/cuda/CUDAStream.cpp, NOT re-read; UNVERIFIED",
        ] if (cfg.concurrent_fastpath and backend.evidence_class != "MOCK") else []),
    }

    on_thor = info["stamp"].get("on_thor")
    fast_c95 = fast["concurrent"]["gpu_event_ms"]["p95"] if fast else None
    fast_o95 = fast["concurrent"]["overlapped_with_refresh"]["gpu_event_ms"]["p95"] if fast else None
    e2e_allow = cfg.preprocess_allowance_ms
    result["verdict"] = evaluate_verdict({
        "n_ticks": len(recs_b), "refresh_p95_ms": tot["p95"], "refresh_mean_ms": tot["mean"], "refresh_hz": cfg.refresh_hz,
        "token_age_p95_ms": summarize(ages_b)["p95"] if ages_b else None,
        "fast_concurrent_n": (fast["concurrent"]["n"] if fast else None), "fast_concurrent_p95_ms": fast_c95,
        "fast_overlapped_p95_ms": fast_o95,
        "fast_overlapped_n": (fast["concurrent"]["overlapped_with_refresh"]["n"] if fast else None),
        "refresh_concurrent_p95_ms": (stage_stats(recs_c)["total_ms"]["p95"] if recs_c else None),
        "refresh_concurrent_n": (len(recs_c) if recs_c else None),
        "preprocess_allowance_ms": e2e_allow, "tick_budget_ms": cfg.tick_budget_ms, "on_thor": on_thor,
    }, backend.evidence_class)
    result["e2e_proxy"] = ({"camera_frame_to_command_ms": (fast_c95 + e2e_allow), "fast_path_concurrent_p95_ms": fast_c95,
                            "preprocess_allowance_ms": e2e_allow, "allowance_label": "ASSUMPTION",
                            "tick_budget_ms": cfg.tick_budget_ms,
                            "excludes": "camera exposure/ISP/transport, actuation, token age (the VLM tokens are cached)"}
                           if fast_c95 is not None else None)

    if cfg.save_features:
        _log(f"[{backend.name}] collecting pooled features -> {cfg.save_features}")
        fmeta, feats = backend.collect_features()
        meta = {"schema": FEATURE_SCHEMA, "model": cfg.hf_id, "dtype": cfg.dtype, "evidence_class": backend.evidence_class,
                "taps": info["model"].get("taps"), "tap_convention": info["model"].get("tap_convention"),
                "pool_tokens": cfg.pool_tokens, "cameras": cfg.cameras, "frames_per_camera": cfg.frames_per_camera,
                "resolution_hw": [cfg.height, cfg.width], **fmeta}
        save_features(cfg.save_features, meta, feats)
        result["saved_features"] = {"path": str(cfg.save_features), "meta": meta,
                                    "shapes": {k: list(v.shape) for k, v in feats.items()}}
    return result


# ======================================================================================
# 10. CLI
# ======================================================================================
def _thor_commands():
    lines = []
    for key, c in CANDIDATES.items():
        extra = " --trust-remote-code" if key == "cosmos3-edge" else ""
        lines.append(f"  # {key}  ({c['hf_id']})\n"
                     f"  OMP_NUM_THREADS=6 python3 hicap_backbone_latency_bench.py --backend hf --model {key} --cameras 1 "
                     f"--frames-per-camera 3 --dtype bf16 --ticks 1000 --warmup 50 --concurrent-fastpath{extra} "
                     f"--frames-dir /data/hicap_frames --save-features out/{key}_bf16.pt --out out/{key}_1cam_bf16.json")
    return "\n".join(lines)


EPILOG = f"""\
Thor command lines (one per candidate; run each on the Jetson Thor, idle box, state the power mode: this script only
RECORDS `nvpmodel -q`, it never sets it; run `sudo nvpmodel -m 0 && sudo jetson_clocks` yourself if that is the mode you
intend to report).  OMP_NUM_THREADS=6 is REQUIRED (torch otherwise spawns ~113 threads).  PYTHONPATH is NOT needed by
this script (it imports nothing from tanitad); if your shell also launches TanitAD tools use
PYTHONPATH=/workspace/TanitAD/stack.

{_thor_commands()}

Camera panel (1 / 3 / 7; the protocol row names ONE camera for the 50 ms refresh threshold):
  for C in 1 3 7; do OMP_NUM_THREADS=6 python3 hicap_backbone_latency_bench.py --backend hf --model qwen3.5-4b --cameras $C \\
      --ticks 1000 --warmup 50 --concurrent-fastpath --out out/qwen3.5-4b_${{C}}cam_bf16.json; done
--frames-per-camera 1 reproduces the token counts of Table tab:thor (n = 224 / 544 / 1184); the default 3 frames per camera
without a ViT-feature cache is the un-cached upper bound (n = 544 / 1504 / 3424).  Wall time at the defaults is ~12 min per run
(1,000 paced refreshes at 2 Hz = 500 s, + 50 s fast-path baseline + 150 s concurrent phase); the loop is paced on purpose
(thermals, the deployed schedule).

Precision drift (H-HC11 'NVFP4 vs BF16'): the BF16 features come from the command above (same --frames-dir!).  Build the NVFP4
engine with TensorRT-Edge-LLM (tensorrt-edgellm-quantize llm --quantization nvfp4 --lm_head_quantization nvfp4
--visual_quantization fp8), dump pooled features for the SAME frames in the feature-file schema (see the module
docstring), then:
  python3 hicap_backbone_latency_bench.py --compare-features out/qwen3-vl-4b_bf16.pt out/qwen3-vl-4b_nvfp4.pt --out out/drift.json
Synthetic frames (no --frames-dir) are fine for LATENCY but their drift is labelled INADMISSIBLE_FOR_DRIFT.

Harness smoke test with no GPU (numbers are MOCK, verdict prints MOCK_ONLY):
  python3 hicap_backbone_latency_bench.py --backend mock --model qwen3-vl-4b --cameras 3 --concurrent-fastpath --out /tmp/mock.json
"""


def build_parser():
    p = argparse.ArgumentParser(
        prog="hicap_backbone_latency_bench.py",
        description="HiCAP H-HC11 backbone latency benchmark (frozen VLM prefill-only: ViT -> LLM prefill -> hidden states at 3 depths "
                    "-> pooled <=128 tokens). hf backend is UNVERIFIED (written without a GPU); mock backend is MOCK only.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=("hf", "mock"), default="hf")
    p.add_argument("--model", default=None, help="candidate key (see --list-candidates), an HF id, or a local path (default for mock: qwen3-vl-4b)")
    p.add_argument("--list-candidates", action="store_true", help="print the candidate table with id provenance and exit")
    p.add_argument("--cameras", type=int, choices=(1, 3, 7), default=1, help="default 1: the protocol row states the 50 ms refresh threshold for ONE camera")
    p.add_argument("--frames-per-camera", type=int, default=3)
    p.add_argument("--resolution", default="256x640", help="HxW per frame (default 256x640, cylindrical crop)")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp8", "nvfp4"), default="bf16",
                   help="bf16/fp16 run; fp8/nvfp4 fail loud -> use TensorRT-Edge-LLM and --compare-features")
    p.add_argument("--ticks", type=int, default=1000, help="measured refresh ticks (protocol requires >= 1000)")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--taps", default="auto", help="'auto' (L/3, 2L/3, L) or comma-separated hidden_states indices")
    p.add_argument("--pool-tokens", type=int, default=128)
    p.add_argument("--refresh-hz", type=float, default=2.0)
    p.add_argument("--fast-hz", type=float, default=10.0)
    p.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS, help="non-image tokens target (markers+template+text)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames-dir", default=None, help="real PNG/JPG frames (sorted; windows of cameras*frames each). Omit -> synthetic noise")
    p.add_argument("--max-windows", type=int, default=8)
    p.add_argument("--save-features", default=None, help="write pooled features (.pt via torch.save, or .npz)")
    p.add_argument("--compare-features", nargs=2, metavar=("REF", "CAND"), default=None,
                   help="per-tap cosine / relative-L2 drift between two feature files (REF = BF16 arm) and exit")
    p.add_argument("--drift-margin-cos", type=float, default=None, help="ASSUMPTION; no margin is pre-registered")
    p.add_argument("--drift-margin-rel-l2", type=float, default=None, help="ASSUMPTION; no margin is pre-registered")
    p.add_argument("--out", default="-", help="JSON output path ('-' = stdout; progress goes to stderr)")
    p.add_argument("--concurrent-fastpath", action="store_true", help="also measure the 10 Hz fast-path proxy alone and concurrent with refresh")
    p.add_argument("--fast-jitter-frac", type=float, default=0.25,
                   help="per-tick uniform jitter of the fast-path trigger, as a fraction of its period (mean rate kept). Avoids "
                        "phase-locking the 10 Hz fast path to the 2 Hz refresh grid, which could give ~0%% overlap by luck; 0 = strictly periodic")
    p.add_argument("--fast-baseline-ticks", type=int, default=500)
    p.add_argument("--concurrent-ticks", type=int, default=300, help="refresh ticks in the concurrent phase (300 @ 2 Hz = 1,500 fast ticks)")
    p.add_argument("--preprocess-allowance-ms", type=float, default=8.0, help="ASSUMPTION: fixed preprocessing allowance for the e2e proxy")
    p.add_argument("--tick-budget-ms", type=float, default=100.0)
    p.add_argument("--no-pace", action="store_true", help="run refreshes back-to-back (no deployed schedule; age/late/dropped become UNMEASURED)")
    p.add_argument("--late-tol-ms", type=float, default=1.0)
    p.add_argument("--attn", choices=("sdpa", "eager", "flash_attention_2"), default="sdpa")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--keep-lm-head", action="store_true", help="do not drop lm_head (default: dropped; prefill-only)")
    p.add_argument("--include-h2d", action="store_true", help="include the host-to-device copy of the inputs in the timed region")
    p.add_argument("--omp-threads", type=int, default=6, help="OMP_NUM_THREADS default, set before torch is imported")
    p.add_argument("--log-inadmissible-memory", action="store_true", help="also log mem_get_info()/VmRSS, labelled INADMISSIBLE_ON_THOR")
    g = p.add_argument_group("mock backend model (illustrative; MOCK)")
    g.add_argument("--mock-seed", type=int, default=0)
    g.add_argument("--mock-vit-ms-per-image", type=float, default=11.4)
    g.add_argument("--mock-llm-base-ms", type=float, default=8.0)
    g.add_argument("--mock-llm-ms-per-token", type=float, default=0.045)
    g.add_argument("--mock-jitter", type=float, default=0.04)
    g.add_argument("--mock-spike-prob", type=float, default=0.01)
    g.add_argument("--mock-fast-base-ms", type=float, default=3.0)
    g.add_argument("--mock-contention-alpha", type=float, default=2.0)
    g.add_argument("--mock-launch-ms", type=float, default=0.3)
    return p


def normalize_args(args):
    """Validate and fill derived fields; fail loud (SystemExit 2) on unsupported configuration."""
    if args.dtype in ("fp8", "nvfp4"):
        raise UsageError(
            f"error: --dtype {args.dtype} is not supported by this harness (the HF backend runs bf16/fp16 only; no silent "
            "downgrade to bf16).\n"
            "  -> Build the engine with TensorRT-Edge-LLM (tensorrt-edgellm-quantize llm --quantization nvfp4 "
            "--lm_head_quantization nvfp4 --visual_quantization fp8), dump pooled features on the SAME frames in the schema in "
            "this file's docstring, and run:\n"
            "     hicap_backbone_latency_bench.py --compare-features BF16_features.pt NVFP4_features.pt\n"
            "  UNVERIFIED: the TRT-Edge-LLM docs we read document no API to return intermediate prefill hidden states.")
    m = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", args.resolution)
    if not m:
        raise UsageError(f"error: --resolution must be HxW like 256x640, got {args.resolution!r}")
    height, width = int(m.group(1)), int(m.group(2))
    if args.ticks < 1 or args.warmup < 0 or args.pool_tokens < 1 or args.refresh_hz <= 0 or args.fast_hz <= 0:
        raise UsageError("error: --ticks>=1, --warmup>=0, --pool-tokens>=1, --refresh-hz>0, --fast-hz>0 required")
    if args.frames_per_camera < 1:
        raise UsageError("error: --frames-per-camera must be >= 1")
    if not (0.0 <= args.fast_jitter_frac < 0.5):
        raise UsageError("error: --fast-jitter-frac must be in [0, 0.5)")
    model = args.model
    if model is None:
        if args.backend == "hf":
            raise UsageError("error: --model is required for --backend hf (a candidate key, an HF id or a local path); see --list-candidates")
        model = "qwen3-vl-4b"
    if model in CANDIDATES:
        key, hf_id, params_b = model, CANDIDATES[model]["hf_id"], CANDIDATES[model]["params_b"]
    else:
        key, hf_id = "custom", model
        pm = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model)
        params_b = float(pm.group(1)) if pm else 4.0
    try:
        parse_taps(args.taps)
    except ValueError as e:
        raise UsageError(f"error: {e}")
    if args.save_features and not (args.save_features.endswith(".pt") or args.save_features.endswith(".npz")):
        raise UsageError("error: --save-features must end in .pt or .npz")
    cfg = SimpleNamespace(**vars(args))
    cfg.model_key, cfg.hf_id, cfg.params_b, cfg.height, cfg.width = key, hf_id, params_b, height, width
    return cfg


def _write_json(obj, out):
    txt = json.dumps(obj, indent=2, sort_keys=False, default=str) + "\n"
    if out == "-":
        sys.stdout.write(txt)
    else:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(txt)


def cmd_compare(args):
    ref_path, cand_path = args.compare_features
    meta_a, fa = load_features(ref_path)
    meta_b, fb = load_features(cand_path)
    for label, meta in (("REF", meta_a), ("CAND", meta_b)):
        if meta.get("schema") != FEATURE_SCHEMA:
            raise UsageError(f"error: {label} feature file has schema {meta.get('schema')!r}, expected {FEATURE_SCHEMA!r}")
    per_tap = compare_pooled(fa, fb)
    notes = []
    if meta_a.get("taps") != meta_b.get("taps"):
        notes.append(f"tap layer indices differ: REF {meta_a.get('taps')} vs CAND {meta_b.get('taps')} (compared by ordinal)")
    out = {"schema": SCHEMA + ".drift", "schema_version": SCHEMA_VERSION, "hypothesis": "H-HC11", "created_utc": _now_utc(),
           "ref": {"path": str(ref_path), "meta": meta_a}, "cand": {"path": str(cand_path), "meta": meta_b},
           "per_tap": per_tap, "verdict": drift_verdict(per_tap, meta_a, meta_b, args.drift_margin_cos, args.drift_margin_rel_l2),
           "notes": notes + ["cos_token_*: cosine along the hidden axis per (window, pooled token); rel_l2 = ||ref-cand||/||ref|| per window "
                             "over the flattened [tokens, width]; head-accuracy retention is NOT measured here"]}
    _write_json(out, args.out)
    _log(f"drift verdict: {out['verdict']['status']}")
    return 0


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return _main(args)
    except UsageError as e:
        print(str(e), file=sys.stderr)
        return 2


def _main(args):
    if args.list_candidates:
        for k, c in CANDIDATES.items():
            print(f"{k:18s} {c['hf_id']:28s} id source: {c['id_source']}\n{'':18s} loader: {c['loader']}")
        print("\nHub existence of every id is UNVERIFIED (huggingface.co was HTTP 403 from the build container).")
        return 0
    if args.compare_features:
        return cmd_compare(args)
    cfg = normalize_args(args)
    os.environ.setdefault("OMP_NUM_THREADS", str(cfg.omp_threads))   # BEFORE any torch import (hf backend imports lazily)
    try:
        if cfg.backend == "mock":
            clock = VirtualClock()
            backend = MockBackend(cfg, clock)
        else:
            clock = RealClock()
            backend = HFBackend(cfg, clock)
        result = run_benchmark(cfg, backend, clock)
    except BackendUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except UsageError:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    _write_json(result, cfg.out)
    v = result["verdict"]
    _log(f"evidence_class={result['evidence_class']}  refresh p95={result['refresh']['stages']['total_ms']['p95']:.2f} ms  "
         f"verdict sync={v['readings']['synchronous']['overall']} async={v['readings']['asynchronous']['overall']}  "
         f"gate: {v['decides_h_hc11_gate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
