"""ONNX export of the TanitAD-4B operative path (encoder+readout, predictor).

Production & Optimization backlog P0.2. Measured deliverable (G-P2): export
fidelity (accuracy delta = max|Δ| PyTorch vs ONNXRuntime under pinned fp32
numerics) reported next to the speed delta (same-device Torch-CPU vs ORT-CPU
latency). No CUDA EP is installed here, so the speed number is CPU-only and is
NOT the deployment story (that is TensorRT-on-Orin, later) — it is an honest
same-device ORT baseline.

Two graphs, matching how the stack deploys the operative tick:
  1. encoder+readout : frames [1,9,256,256] -> state [1,2048]  (runs every frame)
  2. predictor       : (states [1,W,2048], actions [1,W,2]) -> (z1,z2,z4) tuple
     (runs on the causal state window; multi-horizon heads become tuple outputs)

Export uses eval() (grad-checkpoint auto-off, F-5 lever disabled) and the legacy
TorchScript exporter at a pinned opset. Falsifier: any op (MHA / FiLM / causal
mask) that will not export is DOCUMENTED as a plugin/rewrite need — the model is
never hacked to force an export (backlog P0.2 rule).

Usage:
  python export_encoder_predictor.py --ckpt <ckpt_full.pt> --out-dir <off-Drive dir> \
      --report <parity.json> [--opset 17]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from tanitad.config import base250cam_config
from tanitad.instruments.numerics import strict_numerics
from tanitad.models.fourbrain import WorldModel

OPSET_DEFAULT = 17
PARITY_TOL = 1e-4          # backlog P0.2 target: max|Δz| <= 1e-4 fp32
N_PARITY_INPUTS = 5        # distinct random inputs for the parity sweep
LAT_WARMUP, LAT_REPS = 5, 30


class EncoderReadout(nn.Module):
    """frames [B,9,256,256] -> compact state [B,2048] (world.encode)."""

    def __init__(self, world: WorldModel):
        super().__init__()
        self.world = world

    def forward(self, frames: Tensor) -> Tensor:
        return self.world.encode(frames)


class PredictorTuple(nn.Module):
    """(states, actions) -> tuple of horizon states (ONNX has no dict output)."""

    def __init__(self, world: WorldModel):
        super().__init__()
        self.world = world
        self.horizons = tuple(world.predictor.cfg.horizons)

    def forward(self, states: Tensor, actions: Tensor):
        out = self.world.imagine(states, actions)
        return tuple(out[k] for k in self.horizons)


def _try_export(module: nn.Module, args: tuple, path: str, input_names, output_names,
                opset: int) -> dict:
    """Export one module; return {ok, error, unexportable_hint}."""
    try:
        torch.onnx.export(
            module, args, path,
            input_names=input_names, output_names=output_names,
            opset_version=opset, dynamo=False,
            do_constant_folding=True,
        )
        return {"ok": True}
    except Exception as e:  # noqa: BLE001 - we want the raw reason
        msg = str(e)
        hint = None
        low = msg.lower()
        if "multiheadattention" in low or "scaled_dot_product" in low or "aten::_native_multi" in low:
            hint = "nn.MultiheadAttention op path unsupported at this opset"
        elif "triu" in low:
            hint = "causal-mask torch.triu unsupported -> bake mask as constant buffer"
        return {"ok": False, "error": msg[:500], "unexportable_hint": hint}


def _ort_session(path: str):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1  # pinned: deterministic, comparable to torch 1-thread
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def _parity(module: nn.Module, make_inputs, path: str, input_names, n: int) -> dict:
    """max/mean |Δ| between torch fp32 and ORT-CPU over n random inputs."""
    sess = _ort_session(path)
    out_names = [o.name for o in sess.get_outputs()]
    max_abs, mean_abs = [], []
    with torch.no_grad():
        for i in range(n):
            g = torch.Generator().manual_seed(1000 + i)
            inputs = make_inputs(g)
            t_out = module(*inputs)
            t_list = list(t_out) if isinstance(t_out, tuple) else [t_out]
            feed = {input_names[j]: inputs[j].numpy() for j in range(len(input_names))}
            o_list = sess.run(out_names, feed)
            for t, o in zip(t_list, o_list):
                d = np.abs(t.numpy() - o)
                max_abs.append(float(d.max()))
                mean_abs.append(float(d.mean()))
    return {"max_abs_delta": max(max_abs), "mean_abs_delta": float(np.mean(mean_abs)),
            "n_inputs": n, "n_output_tensors": len(out_names),
            "tol": PARITY_TOL, "pass": max(max_abs) <= PARITY_TOL}


def _lat_torch(module, make_inputs) -> dict:
    g = torch.Generator().manual_seed(0)
    inp = make_inputs(g)
    with torch.no_grad():
        for _ in range(LAT_WARMUP):
            module(*inp)
        ts = []
        for _ in range(LAT_REPS):
            s = time.perf_counter()
            module(*inp)
            ts.append((time.perf_counter() - s) * 1e3)
    return {"p50_ms": round(float(np.percentile(ts, 50)), 3),
            "p95_ms": round(float(np.percentile(ts, 95)), 3)}


def _lat_ort(path, make_inputs, input_names) -> dict:
    sess = _ort_session(path)
    out_names = [o.name for o in sess.get_outputs()]
    g = torch.Generator().manual_seed(0)
    inp = make_inputs(g)
    feed = {input_names[j]: inp[j].numpy() for j in range(len(input_names))}
    for _ in range(LAT_WARMUP):
        sess.run(out_names, feed)
    ts = []
    for _ in range(LAT_REPS):
        s = time.perf_counter()
        sess.run(out_names, feed)
        ts.append((time.perf_counter() - s) * 1e3)
    return {"p50_ms": round(float(np.percentile(ts, 50)), 3),
            "p95_ms": round(float(np.percentile(ts, 95)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None,
                    help="ckpt_full.pt; if omitted, exports random-init (parity is "
                         "weight-independent graph fidelity, but real weights are honest)")
    ap.add_argument("--out-dir", required=True, help="off-Drive dir for the .onnx files")
    ap.add_argument("--report", required=True)
    ap.add_argument("--opset", type=int, default=OPSET_DEFAULT)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = base250cam_config()
    world = WorldModel(cfg)
    step = -1
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
        world.load_state_dict(sd)
        step = int(ck.get("step", -1)) if isinstance(ck, dict) else -1
    world = world.eval()
    params_b = sum(p.numel() for p in world.parameters()) / 1e9

    W = cfg.predictor.window          # 8
    S = world.state_dim               # 2048
    A = cfg.predictor.action_dim      # 2

    def enc_inputs(g):
        return (torch.randn(1, 9, 256, 256, generator=g),)

    def pred_inputs(g):
        return (torch.randn(1, W, S, generator=g), torch.randn(1, W, A, generator=g))

    enc_path = str(out_dir / "encoder_readout.onnx")
    pred_path = str(out_dir / "predictor.onnx")
    enc_mod = EncoderReadout(world)
    pred_mod = PredictorTuple(world)
    horizons = pred_mod.horizons

    report: dict = {
        "exp": "onnx-export-parity-operative-path",
        "backlog": "Production&Optimization P0.2",
        "ckpt_step": step, "ckpt": args.ckpt or "random-init",
        "params_billions": round(params_b, 4),
        "opset": args.opset,
        "device": "cpu (fp32, strict numerics); ORT CPUExecutionProvider, 1 thread",
        "note_speed": "CPU-only same-device speed delta; NOT the deployment target "
                      "(TensorRT-on-Orin later). ORT-GPU EP not installed.",
        "graphs": {},
    }

    with strict_numerics():
        # --- encoder+readout ---
        g0 = torch.Generator().manual_seed(0)
        exp = _try_export(enc_mod, enc_inputs(g0), enc_path,
                          ["frames"], ["state"], args.opset)
        enc_block = {"export": exp, "input": "[1,9,256,256]", "output": "[1,2048]"}
        if exp["ok"]:
            enc_block["parity"] = _parity(enc_mod, enc_inputs, enc_path, ["frames"],
                                          N_PARITY_INPUTS)
            enc_block["latency_torch_cpu"] = _lat_torch(enc_mod, enc_inputs)
            enc_block["latency_ort_cpu"] = _lat_ort(enc_path, enc_inputs, ["frames"])
            enc_block["onnx_mb"] = round(Path(enc_path).stat().st_size / 1e6, 1)
        report["graphs"]["encoder_readout"] = enc_block

        # --- predictor ---
        g0 = torch.Generator().manual_seed(0)
        out_names = [f"z_h{k}" for k in horizons]
        exp = _try_export(pred_mod, pred_inputs(g0), pred_path,
                          ["states", "actions"], out_names, args.opset)
        pred_block = {"export": exp, "input": f"states[1,{W},{S}], actions[1,{W},{A}]",
                      "outputs": out_names}
        if exp["ok"]:
            pred_block["parity"] = _parity(pred_mod, pred_inputs, pred_path,
                                           ["states", "actions"], N_PARITY_INPUTS)
            pred_block["latency_torch_cpu"] = _lat_torch(pred_mod, pred_inputs)
            pred_block["latency_ort_cpu"] = _lat_ort(pred_path, pred_inputs,
                                                     ["states", "actions"])
            pred_block["onnx_mb"] = round(Path(pred_path).stat().st_size / 1e6, 1)
        report["graphs"]["predictor"] = pred_block

    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
