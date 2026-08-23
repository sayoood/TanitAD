"""Build a static-shape TensorRT FP16 engine from the exported ONNX (A40 PROXY) and report:
  - does the ONNX->TRT-FP16 build succeed for our ViT encoder + predictor?
  - FP16 engine latency (CUDA events)
  - is the encoder's MHA FUSED? (the #4537 risk) via the engine inspector JSON
An A40 (SM86) engine is NOT portable to Orin (SM87)/Thor; this is a proxy that proves the PATH + measures
FP16 + checks fusion. Writes /root/trt_fp16_report.json."""
from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
import torch
import tensorrt as trt

TRT = trt.__version__
LOG = trt.Logger(trt.Logger.ERROR)


def _np_dtype(dt):
    return {trt.float32: np.float32, trt.float16: np.float16, trt.int32: np.int32,
            trt.int8: np.int8, trt.bool: np.bool_}.get(dt, np.float32)


def build_engine(onnx_path):
    builder = trt.Builder(LOG)
    try:
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        net = builder.create_network(flag)
    except Exception:
        net = builder.create_network()
    parser = trt.OnnxParser(net, LOG)
    ok = parser.parse(Path(onnx_path).read_bytes())
    perr = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not ok:
        return None, {"parse_ok": False, "errors": perr[:5]}
    cfg = builder.create_builder_config()
    cfg.set_flag(trt.BuilderFlag.FP16)
    try:
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    except Exception:
        pass
    t0 = time.time()
    ser = builder.build_serialized_network(net, cfg)
    build_s = round(time.time() - t0, 1)
    if ser is None:
        return None, {"parse_ok": True, "build_ok": False, "build_s": build_s}
    rt = trt.Runtime(LOG)
    eng = rt.deserialize_cuda_engine(ser)
    return eng, {"parse_ok": True, "build_ok": True, "build_s": build_s,
                 "engine_mb": round(len(ser) / 1e6, 1)}


def fusion_check(eng):
    """Search the engine layer info for standalone softmax (=> MHA NOT fused) vs myelin/foreign fused block."""
    try:
        insp = eng.create_engine_inspector()
        info = insp.get_engine_information(trt.LayerInformationFormat.JSON)
        j = json.loads(info) if isinstance(info, str) else info
        layers = j.get("Layers", []) if isinstance(j, dict) else []
        names = [ (l.get("Name","") if isinstance(l, dict) else str(l)) for l in layers ]
        low = " ".join(names).lower()
        n_layers = len(layers)
        has_softmax = "softmax" in low
        has_myelin = ("myelin" in low) or ("foreign" in low)
        has_attn = ("attention" in low) or ("mha" in low) or ("fmha" in low)
        verdict = ("fused (myelin/foreign block, no standalone softmax)" if (has_myelin and not has_softmax)
                   else "LIKELY NOT fused (standalone softmax present)" if has_softmax
                   else "inconclusive")
        return {"n_layers": n_layers, "standalone_softmax": has_softmax, "myelin_or_foreign": has_myelin,
                "attention_named": has_attn, "verdict": verdict}
    except Exception as ex:
        return {"error": "%s: %s" % (type(ex).__name__, str(ex)[:200])}


def latency(eng, warmup=15, reps=80):
    ctx = eng.create_execution_context()
    bufs = {}
    for i in range(eng.num_io_tensors):
        name = eng.get_tensor_name(i)
        shape = tuple(eng.get_tensor_shape(name))
        dt = _np_dtype(eng.get_tensor_dtype(name))
        t = torch.zeros(shape, dtype=torch.from_numpy(np.zeros(1, dt)).dtype, device="cuda")
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
    return {"p50_ms": round(float(np.percentile(a, 50)), 3), "p99_ms": round(float(np.percentile(a, 99)), 3)}


def main():
    rep = {"trt_version": TRT, "gpu": torch.cuda.get_device_name(0), "precision": "FP16 (fp32 fallback allowed)",
           "note": "A40 SM86 PROXY engine — NOT portable to Orin SM87/Thor; proves path + FP16 latency + MHA fusion"}
    for key, path in [("encoder", "/workspace/deploy_onnx/encoder_readout_f4b.onnx"),
                      ("predictor", "/workspace/deploy_onnx/predictor_f4b.onnx")]:
        block = {}
        try:
            eng, meta = build_engine(path)
            block.update(meta)
            if eng is not None:
                block["fp16_latency"] = latency(eng)
                block["fusion"] = fusion_check(eng)
        except Exception as ex:
            block["error"] = "%s: %s" % (type(ex).__name__, str(ex)[:300])
        rep[key] = block
    Path("/root/trt_fp16_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
