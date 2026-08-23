#!/usr/bin/env python3
"""Phase 2 (real TensorRT hardware): ONNX export with REAL weights, TRT-FP16
and TRT-INT8 (entropy-calibrated, real data) engine builds for encoder +
predictor, whole-graph latency (Gate A), and a best-effort per-layer profile.

This is the real-hardware half of the mandated FP16-vs-INT8 benchmark; Phase 1
(bench_p1_accuracy.py) is the PyTorch per-block accuracy-sensitivity half.
Both write into the shared /workspace/int8_bench/orin_int8_benchmark.json.

MEASURED on pod1 (NVIDIA RTX A6000, SM 8.6), 2026-07-23.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

STACK = "/workspace/int8_bench/stack_clean"
sys.path.insert(0, STACK)

from tanitad.config import flagship4b_config                      # noqa: E402
from tanitad.eval.ckpt_compat import (build_world_from_ckpt,      # noqa: E402
                                      append_speed_channel, SPEED_SCALE)
from tanitad.data.mixing import load_episode                       # noqa: E402

import tensorrt as trt                                              # noqa: E402

DEVICE = "cuda"
CKPT = "/workspace/int8_bench/ckpt/ckpt.pt"
EPCACHE = Path("/workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894")
ONNX_DIR = Path("/workspace/int8_bench/onnx")
OUT = Path("/workspace/int8_bench/orin_int8_benchmark.json")
WINDOW = 8
N_CALIB = 256
OPSET = 17
TOL = 1e-4

LOG = trt.Logger(trt.Logger.ERROR)
TRT_VER = trt.__version__


def log(msg):
    print(f"[bench-p2] {msg}", flush=True)


def merge_report(key, val):
    rep = json.loads(OUT.read_text()) if OUT.exists() else {}
    rep[key] = val
    rep["_last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    OUT.write_text(json.dumps(rep, indent=2, default=str))
    log(f"banked report key={key!r} -> {OUT}")


# --------------------------------------------------------------------------- #
# ONNX-export-safe attention (this pod's torch 2.4.1 legacy exporter has no   #
# symbolic for aten::_native_multi_head_attention at opset 17 -- the          #
# 2026-07-22 intake exported fine because THAT pod ran torch 2.8.0. Rather    #
# than upgrade torch/CUDA on a shared pod (risk: cu128 wheels need a newer    #
# driver than this pod's 550.127.08), decompose nn.MultiheadAttention into    #
# primitive ops (Linear/matmul/softmax) that have been ONNX-exportable for    #
# many opsets. Reads the SAME parameter TENSORS (not copies) -> byte-         #
# identical weights, mathematically identical forward.                       #
# --------------------------------------------------------------------------- #
class DecomposedMHA(nn.Module):
    def __init__(self, mha: nn.MultiheadAttention):
        super().__init__()
        assert mha.batch_first, "decomposition assumes batch_first=True"
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj

    def forward(self, query, key, value, attn_mask=None, need_weights=False):
        b, n, d = query.shape
        wq, wk, wv = self.in_proj_weight.chunk(3, dim=0)
        if self.in_proj_bias is not None:
            bq, bk, bv = self.in_proj_bias.chunk(3, dim=0)
        else:
            bq = bk = bv = None
        q = torch.nn.functional.linear(query, wq, bq)
        k = torch.nn.functional.linear(key, wk, bk)
        v = torch.nn.functional.linear(value, wv, bv)
        q = q.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            else:
                scores = scores + attn_mask
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        out = self.out_proj(out)
        return out, None


def make_export_safe_(module: nn.Module) -> int:
    """In-place: replace every nn.MultiheadAttention in `module` with a
    DecomposedMHA sharing the same parameters. Returns count replaced."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            setattr(module, name, DecomposedMHA(child))
            n += 1
        else:
            n += make_export_safe_(child)
    return n


@torch.no_grad()
def verify_decomposition_parity(world, n=4):
    """Sanity-check: DecomposedMHA must reproduce the ORIGINAL nn.Multihead
    Attention's output bit-for-bit (up to float rounding) before we trust the
    exported graph. Runs BEFORE the in-place patch."""
    import copy
    ref = copy.deepcopy(world).eval()
    probe = copy.deepcopy(world).eval()
    make_export_safe_(probe.encoder)
    make_export_safe_(probe.predictor)
    frames = torch.randn(n, 9, 256, 256, device=DEVICE)
    states = torch.randn(n, WINDOW, world.state_dim, device=DEVICE)
    actions = torch.randn(n, WINDOW, world.cfg.predictor.action_dim, device=DEVICE)
    e_ref, e_probe = ref.encode(frames), probe.encode(frames)
    p_ref, p_probe = ref.imagine(states, actions)[1], probe.imagine(states, actions)[1]
    return {
        "encoder_max_abs_delta": float((e_ref - e_probe).abs().max()),
        "predictor_max_abs_delta": float((p_ref - p_probe).abs().max()),
    }


# --------------------------------------------------------------------------- #
# Model + ONNX export (REAL weights, static shapes)                          #
# --------------------------------------------------------------------------- #
class EncoderReadout(nn.Module):
    def __init__(self, world):
        super().__init__()
        self.world = world

    def forward(self, frames: Tensor) -> Tensor:
        return self.world.encode(frames)


class Predictor1Step(nn.Module):
    """Exports ONLY the k=1 head (the one `rollout_decode` actually consumes at
    inference) as a single-tensor-output graph -- simpler TRT I/O than the
    3-tuple export in the 2026-07-22 intake, same underlying computation."""

    def __init__(self, world):
        super().__init__()
        self.world = world

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        return self.world.imagine(states, actions)[1]


def load_world():
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    world, speed_input, src = build_world_from_ckpt(flagship4b_config(), ck, ckpt_path=CKPT)
    return world.to(DEVICE).eval()


def export_onnx(world):
    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    enc_mod = EncoderReadout(world).eval()
    pred_mod = Predictor1Step(world).eval()
    S = world.state_dim
    A = world.cfg.predictor.action_dim
    enc_path = str(ONNX_DIR / "encoder_readout_real.onnx")
    pred_path = str(ONNX_DIR / "predictor_1step_real.onnx")

    frames_ex = torch.randn(1, 9, 256, 256, device=DEVICE)
    states_ex = torch.randn(1, WINDOW, S, device=DEVICE)
    actions_ex = torch.randn(1, WINDOW, A, device=DEVICE)

    report = {"opset": OPSET, "window": WINDOW, "state_dim": S, "action_dim": A}
    with torch.no_grad():
        try:
            torch.onnx.export(enc_mod, (frames_ex,), enc_path, input_names=["frames"],
                              output_names=["state"], opset_version=OPSET, dynamo=False,
                              do_constant_folding=True)
            report["encoder"] = {"ok": True, "onnx_mb": round(Path(enc_path).stat().st_size / 1e6, 1)}
        except Exception as e:
            report["encoder"] = {"ok": False, "error": str(e)[:500]}
        try:
            torch.onnx.export(pred_mod, (states_ex, actions_ex), pred_path,
                              input_names=["states", "actions"], output_names=["z_h1"],
                              opset_version=OPSET, dynamo=False, do_constant_folding=True)
            report["predictor"] = {"ok": True, "onnx_mb": round(Path(pred_path).stat().st_size / 1e6, 1)}
        except Exception as e:
            report["predictor"] = {"ok": False, "error": str(e)[:500]}

    # parity vs eager (real weights this time, not random init)
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    if report["encoder"]["ok"]:
        sess = ort.InferenceSession(enc_path, so, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            t_out = enc_mod(frames_ex.cpu()).numpy() if False else enc_mod(frames_ex).cpu().numpy()
        o_out = sess.run(None, {"frames": frames_ex.cpu().numpy()})[0]
        report["encoder"]["parity_max_abs_delta"] = float(np.abs(t_out - o_out).max())
    if report["predictor"]["ok"]:
        sess = ort.InferenceSession(pred_path, so, providers=["CPUExecutionProvider"])
        with torch.no_grad():
            t_out = pred_mod(states_ex, actions_ex).cpu().numpy()
        o_out = sess.run(None, {"states": states_ex.cpu().numpy(), "actions": actions_ex.cpu().numpy()})[0]
        report["predictor"]["parity_max_abs_delta"] = float(np.abs(t_out - o_out).max())

    # node-name inspection (informs whether block-level grouping is feasible)
    import onnx as onnx_lib
    for key, path in (("encoder", enc_path), ("predictor", pred_path)):
        if report[key]["ok"]:
            m = onnx_lib.load(path)
            names = [n.name for n in m.graph.node][:40]
            optypes = {}
            for n in m.graph.node:
                optypes[n.op_type] = optypes.get(n.op_type, 0) + 1
            report[key]["n_nodes"] = len(m.graph.node)
            report[key]["sample_node_names"] = names
            report[key]["op_type_counts"] = optypes

    return report, enc_path, pred_path


# --------------------------------------------------------------------------- #
# Real calibration data (torch cuda tensors, batch=1, matching the static     #
# deploy shape -- TRT calibration must feed the SAME shape the engine builds  #
# at)                                                                          #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def harvest_calib(world, n=N_CALIB):
    files = sorted(EPCACHE.glob("ep_*.pt"))[:n]
    enc_batches, states_batches, actions_batches = [], [], []
    for f in files:
        ep = load_episode(str(f), mmap=True)
        fr = ep.frames.float().div(255.0) if ep.frames.dtype == torch.uint8 else ep.frames
        T = fr.shape[0]
        if T < WINDOW + 1:
            continue
        t0 = max(0, T // 2 - WINDOW)
        frame0 = fr[t0].unsqueeze(0).to(DEVICE).contiguous()
        enc_batches.append(frame0)
        fw = fr[t0:t0 + WINDOW].unsqueeze(0).to(DEVICE)
        aw = ep.actions[t0:t0 + WINDOW].unsqueeze(0).to(DEVICE)
        last = t0 + WINDOW - 1
        v0 = (ep.poses[last, 3:4] / SPEED_SCALE).to(DEVICE).unsqueeze(0)
        aw3 = append_speed_channel(aw, v0).contiguous()
        st = world.encode_window(fw).contiguous()
        states_batches.append(st)
        actions_batches.append(aw3)
    return enc_batches, states_batches, actions_batches


class TorchEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, input_names, batches, cache_path):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.input_names = input_names
        self.batches = batches               # list of tuples of cuda tensors, matching input_names order
        self.cache_path = cache_path
        self.idx = 0

    def get_batch_size(self):
        return int(self.batches[0][0].shape[0])

    def get_batch(self, names):
        if self.idx >= len(self.batches):
            return None
        batch = self.batches[self.idx]
        self.idx += 1
        ptrs = []
        for nm in names:
            j = self.input_names.index(nm)
            ptrs.append(int(batch[j].data_ptr()))
        return ptrs

    def read_calibration_cache(self):
        p = Path(self.cache_path)
        return p.read_bytes() if p.exists() else None

    def write_calibration_cache(self, cache):
        Path(self.cache_path).write_bytes(cache)


# --------------------------------------------------------------------------- #
# TRT engine build + latency + profiling                                     #
# --------------------------------------------------------------------------- #
def _np_dtype(dt):
    return {trt.float32: np.float32, trt.float16: np.float16, trt.int32: np.int32,
            trt.int8: np.int8, trt.bool: np.bool_}.get(dt, np.float32)


def _torch_dtype(dt):
    return {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
            trt.int8: torch.int8, trt.bool: torch.bool}.get(dt, torch.float32)


def build_engine(onnx_path, precision, calibrator=None, workspace_gb=4):
    builder = trt.Builder(LOG)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    net = builder.create_network(flag)
    parser = trt.OnnxParser(net, LOG)
    ok = parser.parse(Path(onnx_path).read_bytes())
    if not ok:
        return None, {"parse_ok": False, "errors": [str(parser.get_error(i))
                      for i in range(parser.num_errors)][:5]}
    cfg = builder.create_builder_config()
    try:
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    except Exception:
        pass
    try:
        cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    except Exception:
        pass
    if precision == "fp16":
        cfg.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        cfg.set_flag(trt.BuilderFlag.FP16)          # fallback path per BENCHMARK_PLAN runbook (--int8 --fp16)
        cfg.set_flag(trt.BuilderFlag.INT8)
        cfg.int8_calibrator = calibrator
    t0 = time.time()
    ser = builder.build_serialized_network(net, cfg)
    build_s = round(time.time() - t0, 1)
    if ser is None:
        return None, {"parse_ok": True, "build_ok": False, "build_s": build_s}
    # NOTE (fixed 2026-07-23): this pod's tensorrt_bindings.IHostMemory does not
    # support len() directly (TypeError) even though it is the identical TRT
    # 10.16.1.11 wheel used successfully on pod3 -- a binding-level quirk, not
    # a version mismatch. bytes(ser) goes through the buffer protocol instead,
    # which is robust regardless of whether __len__ is exposed.
    ser_bytes = bytes(ser)
    runtime = trt.Runtime(LOG)
    eng = runtime.deserialize_cuda_engine(ser_bytes)
    return eng, {"parse_ok": True, "build_ok": True, "build_s": build_s,
                 "engine_mb": round(len(ser_bytes) / 1e6, 1)}


def fusion_check(eng):
    try:
        insp = eng.create_engine_inspector()
        info = insp.get_engine_information(trt.LayerInformationFormat.JSON)
        j = json.loads(info) if isinstance(info, str) else info
        layers = j.get("Layers", []) if isinstance(j, dict) else []
        names = [(l.get("Name", "") if isinstance(l, dict) else str(l)) for l in layers]
        low = " ".join(names).lower()
        has_softmax = "softmax" in low
        has_myelin = ("myelin" in low) or ("foreign" in low)
        return {"n_layers": len(layers), "standalone_softmax": has_softmax,
                "myelin_or_foreign": has_myelin,
                "verdict": ("fused" if (has_myelin and not has_softmax)
                           else "LIKELY NOT fused" if has_softmax else "inconclusive"),
                "layer_names_sample": names[:30]}
    except Exception as ex:
        return {"error": f"{type(ex).__name__}: {str(ex)[:200]}"}


def latency(eng, warmup=20, reps=100):
    ctx = eng.create_execution_context()
    bufs = {}
    for i in range(eng.num_io_tensors):
        name = eng.get_tensor_name(i)
        shape = tuple(eng.get_tensor_shape(name))
        dt = _torch_dtype(eng.get_tensor_dtype(name))
        t = torch.zeros(shape, dtype=dt, device="cuda")
        bufs[name] = t
        ctx.set_tensor_address(name, t.data_ptr())
    stream = torch.cuda.Stream()

    def run():
        ctx.execute_async_v3(stream.cuda_stream)
    for _ in range(warmup):
        run()
    stream.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(stream); run(); e.record(stream); stream.synchronize()
        ts.append(s.elapsed_time(e))
    a = np.array(ts)
    return {"p50_ms": round(float(np.percentile(a, 50)), 4),
            "p99_ms": round(float(np.percentile(a, 99)), 4),
            "mean_ms": round(float(a.mean()), 4)}


class _Profiler(trt.IProfiler):
    def __init__(self):
        trt.IProfiler.__init__(self)
        self.times = {}
        self.calls = {}

    def report_layer_time(self, layer_name, ms):
        self.times[layer_name] = self.times.get(layer_name, 0.0) + ms
        self.calls[layer_name] = self.calls.get(layer_name, 0) + 1


def per_layer_profile(eng, reps=30):
    try:
        ctx = eng.create_execution_context()
        prof = _Profiler()
        ctx.profiler = prof
        bufs = {}
        names = []
        for i in range(eng.num_io_tensors):
            name = eng.get_tensor_name(i)
            names.append(name)
            shape = tuple(eng.get_tensor_shape(name))
            dt = _torch_dtype(eng.get_tensor_dtype(name))
            t = torch.zeros(shape, dtype=dt, device="cuda")
            bufs[name] = t
            ctx.set_tensor_address(name, t.data_ptr())
        bindings = [bufs[n].data_ptr() for n in names]
        for _ in range(5):
            ctx.execute_v2(bindings)
        torch.cuda.synchronize()
        prof.times, prof.calls = {}, {}
        for _ in range(reps):
            ctx.execute_v2(bindings)
        torch.cuda.synchronize()
        out = {k: round(v / reps, 5) for k, v in sorted(prof.times.items(), key=lambda kv: -kv[1])}
        return {"ok": True, "per_layer_ms": out, "n_layers_reported": len(out),
                "total_ms": round(sum(out.values()), 4)}
    except Exception as ex:
        return {"ok": False, "error": f"{type(ex).__name__}: {str(ex)[:300]}"}


def main():
    log("=== load real-weight world ===")
    world = load_world()
    log(f"world params = {sum(p.numel() for p in world.parameters())}")

    log("=== verify MHA decomposition parity (before touching the export copy) ===")
    parity = verify_decomposition_parity(world)
    merge_report("mha_decomposition_parity", parity)
    log(f"decomposition parity: {parity}")
    assert parity["encoder_max_abs_delta"] < 1e-4, "decomposed encoder attention diverges -- do not trust export"
    assert parity["predictor_max_abs_delta"] < 1e-4, "decomposed predictor attention diverges -- do not trust export"

    log("=== patch world for ONNX-safe export (torch 2.4.1 has no MHA opset-17 symbol) ===")
    n_patched_enc = make_export_safe_(world.encoder)
    n_patched_pred = make_export_safe_(world.predictor)
    log(f"patched {n_patched_enc} encoder MHA + {n_patched_pred} predictor MHA -> DecomposedMHA")

    log("=== export ONNX (real weights) ===")
    export_report, enc_path, pred_path = export_onnx(world)
    merge_report("onnx_export_real_weights", export_report)
    log(json.dumps({k: v for k, v in export_report.items() if k in ("encoder", "predictor")}, indent=2)[:2000])

    log("=== harvest real calibration data ===")
    enc_batches, states_batches, actions_batches = harvest_calib(world, n=N_CALIB)
    log(f"calib batches: encoder={len(enc_batches)} predictor={len(states_batches)}")

    trt_report = {"trt_version": TRT_VER, "gpu": torch.cuda.get_device_name(0),
                 "n_calib_batches": len(enc_batches)}

    # ---------------- ENCODER ----------------
    for prec in ("fp16", "int8"):
        log(f"--- building ENCODER engine: {prec} ---")
        block = {}
        try:
            calib = None
            if prec == "int8":
                calib = TorchEntropyCalibrator(["frames"], [(b,) for b in enc_batches],
                                               "/workspace/int8_bench/enc_calib.cache")
            eng, meta = build_engine(enc_path, prec, calibrator=calib)
            block.update(meta)
            if eng is not None:
                block["latency"] = latency(eng)
                block["fusion"] = fusion_check(eng)
                block["per_layer_profile"] = per_layer_profile(eng)
        except Exception as ex:
            block["error"] = f"{type(ex).__name__}: {str(ex)[:400]}"
        trt_report.setdefault("encoder", {})[prec] = block
        merge_report("trt_engines", trt_report)
        log(f"encoder {prec}: {block.get('latency', block.get('error'))}")

    # ---------------- PREDICTOR ----------------
    for prec in ("fp16", "int8"):
        log(f"--- building PREDICTOR engine: {prec} ---")
        block = {}
        try:
            calib = None
            if prec == "int8":
                calib = TorchEntropyCalibrator(
                    ["states", "actions"],
                    [(s, a) for s, a in zip(states_batches, actions_batches)],
                    "/workspace/int8_bench/pred_calib.cache")
            eng, meta = build_engine(pred_path, prec, calibrator=calib)
            block.update(meta)
            if eng is not None:
                block["latency"] = latency(eng)
                block["fusion"] = fusion_check(eng)
                block["per_layer_profile"] = per_layer_profile(eng)
        except Exception as ex:
            block["error"] = f"{type(ex).__name__}: {str(ex)[:400]}"
        trt_report.setdefault("predictor", {})[prec] = block
        merge_report("trt_engines", trt_report)
        log(f"predictor {prec}: {block.get('latency')}")

    # ---------------- reproduction check vs the 2026-07-22 staged baseline ----------------
    baseline = {"encoder_fp16_p50_ms": 1.205, "predictor_fp16_p50_ms": 0.666}
    repro = {
        "baseline_source": ("TanitAD Research Hub/.../incoming/2026-07-22-orin-thor-deployment/"
                            "artifacts/trt_fp16_report.json (random-init weights, same A40-class arch)"),
        "this_run_gpu": torch.cuda.get_device_name(0),
        "encoder_fp16_p50_ms_this_run": trt_report.get("encoder", {}).get("fp16", {}).get("latency", {}).get("p50_ms"),
        "predictor_fp16_p50_ms_this_run": trt_report.get("predictor", {}).get("fp16", {}).get("latency", {}).get("p50_ms"),
        "note": ("pod1 is an A6000 (SM 8.6), the 2026-07-22 baseline was measured on an A40 (SM 8.6) -- "
                 "same architecture generation, different SKU, so a close-but-not-identical reproduction "
                 "is expected; a large deviation would indicate an env difference worth flagging."),
    }
    merge_report("fp16_baseline_reproduction", repro)
    log(f"reproduction check: {repro}")

    log("PHASE 2 COMPLETE")


if __name__ == "__main__":
    main()
