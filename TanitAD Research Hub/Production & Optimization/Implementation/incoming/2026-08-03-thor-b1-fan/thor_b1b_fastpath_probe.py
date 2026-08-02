"""B1b — the MHA-fastpath mechanism, probed at a SECOND location and a SECOND opset.

Why this exists: B1 found that on the PREDICTOR at opset 17, the fastpath flag changed
NOTHING -- same node count (1223 both), same ORT parity (3.6e-7 ON vs 3.7e-7 OFF), same
engine latency (1.154 vs 1.187 ms). That does not reproduce the runbook's §3 claim
("opset 17, fastpath ON -> rel-err 0.726, SILENTLY WRONG").

⛔ Absence found at ONE location is not absence (operating standard rule 2). Before saying
anything about the mechanism, probe:
  (a) opset 18 with fastpath ON -- the runbook says it fails LOUDLY with
      `aten::_native_multi_head_attention` unsupported. If it exports clean, the fused op is
      simply not being produced here, and the fastpath story cannot be the cause of 0.726.
  (b) the ENCODER, not just the predictor -- the fused op may live on the other tower.
  (c) opset 18 fastpath OFF, as the control.

Competing hypothesis being tested: `thor_trt_accuracy.json` (18:13, superseded) reports
fp32 0.72824 and fp16 0.72818 -- near-IDENTICAL across precisions, which the runbook's own
learning #8 names as "the signature of a WIRING/TEST bug, not a precision problem". So the
0.726 may be that wiring bug, misattributed to the fastpath.
"""
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("~/TanitAD/stack"))
sys.path.insert(0, os.path.expanduser("~/TanitAD/stack/scripts"))
sys.path.insert(0, "/usr/lib/python3.12/dist-packages")

import dataclasses
from types import SimpleNamespace

import numpy as np
import torch

from tanitad.config import flagship4b_config
from tanitad.models.fourbrain import WorldModel
from train_flagship_v4 import resolve_v2_frames

DEV, H, W = "cuda", 176, 624
OUT = os.path.expanduser("~/trt_b1b")
os.makedirs(OUT, exist_ok=True)
out = {"purpose": "probe the MHA-fastpath mechanism at a 2nd location + 2nd opset"}

cfg = flagship4b_config()
ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0,
                     projection="cylindrical", v2_subframe="176x624", f_ref=None)
resolve_v2_frames(ns, cfg, label="b1b")
cfg.speed_input = True
cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
if getattr(cfg, "tactical_pred", None) is not None:
    cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
model = WorldModel(cfg).to(DEV).eval()
Wn, C, A = cfg.predictor.window, cfg.encoder.in_channels, cfg.predictor.action_dim
S = model.state_dim

# how many nn.MultiheadAttention modules are actually IN each tower?
out["mha_module_census"] = {
    "predictor": sum(1 for m in model.predictor.modules()
                     if isinstance(m, torch.nn.MultiheadAttention)),
    "encoder": sum(1 for m in model.encoder.modules()
                   if isinstance(m, torch.nn.MultiheadAttention)),
    "whole_model": sum(1 for m in model.modules()
                       if isinstance(m, torch.nn.MultiheadAttention)),
}


class PredWrap(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, states, actions):
        return self.p(states, actions)[1]


class EncWrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, frames):
        return self.m.encode_window(frames)


def probe(tag, mod, args, opset, fastpath):
    torch.backends.mha.set_fastpath_enabled(fastpath)
    path = f"{OUT}/{tag}.onnx"
    rec = {"opset": opset, "fastpath": fastpath}
    with torch.no_grad():
        ref = mod(*args).float().cpu().numpy()
    try:
        torch.onnx.export(mod, args, path,
                          input_names=[f"in{i}" for i in range(len(args))],
                          output_names=["out"], opset_version=opset, dynamo=False)
        rec["export_ok"] = True
    except Exception as e:
        rec["export_ok"] = False
        rec["err"] = f"{type(e).__name__}: {str(e)[:300]}"
        return rec
    try:
        import onnx
        m = onnx.load(path, load_external_data=False)
        rec["n_nodes"] = len(m.graph.node)
        rec["has_fused_mha_op"] = any(
            "MultiHeadAttention" in n.op_type or "native_multi_head" in n.op_type
            for n in m.graph.node)
    except Exception as e:
        rec["onnx_scan"] = f"{type(e).__name__}"
    try:
        import onnxruntime as ort
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        feed = {f"in{i}": a.cpu().numpy() for i, a in enumerate(args)}
        y = s.run(["out"], feed)[0]
        rec["ort_rel_err_vs_eager"] = round(
            float(np.linalg.norm(y - ref) / max(np.linalg.norm(ref), 1e-12)), 9)
    except Exception as e:
        rec["ort_rel_err_vs_eager"] = None
        rec["ort_err"] = f"{type(e).__name__}: {str(e)[:200]}"
    return rec


pw = PredWrap(model.predictor).eval()
st = torch.randn(1, Wn, S, device=DEV)
ac = torch.randn(1, Wn, A, device=DEV)
ew = EncWrap(model).eval()
fr = torch.randn(1, Wn, C, H, W, device=DEV)

res = {}
for opset in (17, 18):
    for fp in (True, False):
        res[f"predictor_op{opset}_fastpath{'ON' if fp else 'OFF'}"] = probe(
            f"pred_op{opset}_fp{int(fp)}", pw, (st, ac), opset, fp)
for fp in (True, False):
    res[f"encoder_op17_fastpath{'ON' if fp else 'OFF'}"] = probe(
        f"enc_op17_fp{int(fp)}", ew, (fr,), 17, fp)
torch.backends.mha.set_fastpath_enabled(False)
out["probes"] = res

# ------------------------------------------------------------------ reading
def clean(r):
    e = r.get("ort_rel_err_vs_eager")
    return bool(r.get("export_ok") and e is not None and e < 1e-4)


all_clean = all(clean(r) for r in res.values())
any_fused = any(r.get("has_fused_mha_op") for r in res.values())
out["READING"] = {
    "every_cell_parity_clean": all_clean,
    "any_graph_carries_a_fused_mha_op": any_fused,
    "verdict": (
        "⛔ THE RUNBOOK'S §3 MECHANISM DOES NOT REPRODUCE at any probed cell: no export carries a "
        "fused MHA op, opset 18 does not fail, and parity is clean with the fastpath ON. The 0.726 "
        "in the superseded thor_trt_accuracy.json is then most likely the WIRING bug its own "
        "near-identical-across-precisions signature indicates (learning #8), not the fastpath. "
        "⚠️ SCOPE: this probes torch 2.13 / this model / opsets 17-18 only."
        if all_clean and not any_fused else
        "MECHANISM PARTIALLY REPRODUCES — see the per-cell table; do NOT generalise either way"),
}

print(json.dumps(out, indent=1), flush=True)
with open(os.path.expanduser("~/thor_b1b_fastpath_probe.json"), "w") as f:
    json.dump(out, f, indent=1)
