"""Static-shape ONNX export of deployed flagship-v1 (flagship4b) operative path + A40 torch latency reference.

Deliverable for 2026-07-22 Orin/Thor deployment prep. Extends the 2026-07-08 export
(which used base250cam_config @ action_dim=2, step 6500, CPU) to the EXACT deployed
architecture (flagship4b, action_dim=3 speed channel) with STATIC shapes for a
static-shape ONNX->TRT-FP16 engine, run on the A40 (SM86) proxy.

Latency is architecture- not weight-determined (established: registry 1.2, 07-20 levers note),
so random init is faithful for graph fidelity + latency. Real-weight re-export is a trivial
--ckpt swap once flagship4b-speedjerk-30k is reachable (currently only on pod2/eval, off-limits).

Writes JSON to --report.
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn

sys.path.insert(0, "/workspace/TanitAD/stack")
from tanitad.config import flagship4b_config
from tanitad.models.fourbrain import WorldModel

OPSET = 17
TOL = 1e-4
N_PARITY = 5


class EncoderReadout(nn.Module):
    def __init__(self, world): super().__init__(); self.world = world
    def forward(self, frames: Tensor) -> Tensor: return self.world.encode(frames)


class PredictorTuple(nn.Module):
    def __init__(self, world):
        super().__init__(); self.world = world
        self.horizons = tuple(world.predictor.cfg.horizons)
    def forward(self, states: Tensor, actions: Tensor):
        out = self.world.imagine(states, actions)
        return tuple(out[k] for k in self.horizons)


def try_export(module, args, path, in_names, out_names) -> dict:
    try:
        torch.onnx.export(module, args, path, input_names=in_names, output_names=out_names,
                          opset_version=OPSET, dynamo=False, do_constant_folding=True)
        return {"ok": True, "onnx_mb": round(Path(path).stat().st_size / 1e6, 1)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:400]}


def parity(module, make_inputs, path, in_names) -> dict:
    import onnxruntime as ort
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    maxd = []
    with torch.no_grad():
        for i in range(N_PARITY):
            g = torch.Generator().manual_seed(100 + i)
            inp = make_inputs(g)
            t_out = module(*inp)
            t_list = list(t_out) if isinstance(t_out, tuple) else [t_out]
            feed = {in_names[j]: inp[j].numpy() for j in range(len(in_names))}
            o_list = sess.run(out_names, feed)
            for t, o in zip(t_list, o_list):
                maxd.append(float(np.abs(t.numpy() - o).max()))
    return {"max_abs_delta": max(maxd), "tol": TOL, "pass": max(maxd) <= TOL, "n": N_PARITY}


def cuda_time(fn, warmup=20, reps=100) -> dict:
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    a = np.array(ts)
    return {"p50_ms": round(float(np.percentile(a, 50)), 3), "p99_ms": round(float(np.percentile(a, 99)), 3)}


def gpu_latency(world, W, S, A, K=20) -> dict:
    """Torch A40 reference: predictor 1-call fp32 vs fp16, and K-step predictor-loop eager vs CUDA-graph.
    This is a PREDICTOR-ONLY rollout proxy (no step-readout / window-slide) — it reproduces the MECHANISM
    (graph is the rollout lever, fp16 is not) independently; it is NOT the full planning tick."""
    dev = "cuda"
    res = {"note": "predictor-only rollout proxy on A40; not the full tick (cite 07-20 levers note for the tick)"}
    pred = PredictorTuple(world).to(dev).eval()
    st = torch.randn(1, W, S, device=dev); ac = torch.randn(1, W, A, device=dev)
    with torch.no_grad():
        res["predictor_1call_fp32"] = cuda_time(lambda: pred(st, ac))
        pred16 = PredictorTuple(world).to(dev).half().eval()
        st16 = st.half(); ac16 = ac.half()
        res["predictor_1call_fp16"] = cuda_time(lambda: pred16(st16, ac16))

        def eager_roll():
            s = st
            for _ in range(K):
                z = pred(s, ac)[0]
                s = torch.cat([s[:, 1:], z[:, -1:].detach()], dim=1) if z.dim() == 3 else s
        res["rollout_k%d_eager_fp32" % K] = cuda_time(eager_roll, warmup=10, reps=50)

        # CUDA-graph capture of a single predictor step, replayed K times
        s_static = st.clone(); a_static = ac.clone()
        stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): _ = pred(s_static, a_static)
        torch.cuda.current_stream().wait_stream(stream)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out_static = pred(s_static, a_static)
        def graph_roll():
            for _ in range(K): g.replay()
        res["rollout_k%d_graph_step_fp32" % K] = cuda_time(graph_roll, warmup=10, reps=50)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--skip-gpu", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    cfg = flagship4b_config()
    adim_note = "action_dim=2 (base flagship4b)"
    try:
        cfg.predictor.action_dim = 3  # deployed v1 speed channel (--speed-input)
        adim_note = "action_dim=3 (deployed v1 speed+jerk)"
    except Exception:
        pass
    try:
        world = WorldModel(cfg)
        built = True
    except Exception as e:
        # fall back to action_dim=2 if the 3-dim assembly fails elsewhere
        cfg = flagship4b_config(); world = WorldModel(cfg)
        adim_note = "action_dim=2 (fallback; 3-dim assembly raised %s)" % type(e).__name__
        built = True
    step = -1
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        world.load_state_dict(sd, strict=False); step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    world = world.eval()
    params_m = sum(p.numel() for p in world.parameters()) / 1e6
    W = cfg.predictor.window; S = world.state_dim; A = cfg.predictor.action_dim

    def enc_in(g): return (torch.randn(1, 9, 256, 256, generator=g),)
    def pred_in(g): return (torch.randn(1, W, S, generator=g), torch.randn(1, W, A, generator=g))

    report = {"exp": "flagship4b-deployed-static-onnx-export+A40-latency", "date": "2026-07-22",
              "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "ckpt": args.ckpt or "random-init (latency+graph-fidelity are weight-independent)",
              "ckpt_step": step, "params_m": round(params_m, 2), "action_dim_note": adim_note,
              "opset": OPSET, "static_shapes": True,
              "deploy_shapes": {"encoder": "[1,9,256,256]->[1,2048] (per-frame, cache-friendly)",
                                "predictor": "states[1,%d,%d],actions[1,%d,%d]->(z1,z2,z4)" % (W, S, W, A)},
              "graphs": {}}

    enc_mod = EncoderReadout(world); pred_mod = PredictorTuple(world)
    enc_path = str(out / "encoder_readout_f4b.onnx"); pred_path = str(out / "predictor_f4b.onnx")

    g0 = torch.Generator().manual_seed(0)
    eb = try_export(enc_mod, enc_in(g0), enc_path, ["frames"], ["state"])
    if eb["ok"]: eb["parity"] = parity(enc_mod, enc_in, enc_path, ["frames"])
    report["graphs"]["encoder_readout"] = eb

    g0 = torch.Generator().manual_seed(0)
    on = ["z_h%d" % k for k in pred_mod.horizons]
    pb = try_export(pred_mod, pred_in(g0), pred_path, ["states", "actions"], on)
    if pb["ok"]: pb["parity"] = parity(pred_mod, pred_in, pred_path, ["states", "actions"])
    report["graphs"]["predictor"] = pb

    if not args.skip_gpu and torch.cuda.is_available():
        try:
            report["a40_torch_latency"] = gpu_latency(world, W, S, A)
        except Exception as e:
            report["a40_torch_latency"] = {"error": "%s: %s" % (type(e).__name__, str(e)[:300])}

    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
