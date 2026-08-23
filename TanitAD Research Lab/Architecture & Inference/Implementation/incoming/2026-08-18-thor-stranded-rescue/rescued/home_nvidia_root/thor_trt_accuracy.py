"""⛔ THE GATING CHECK: does the TensorRT engine still compute OUR model?

trtexec reports timing only. Addendum 5 published a 3.62x speedup with numerical agreement
explicitly marked UNMEASURED, and their own stream set the bar: **95.3 % decision-agreement vs
fp32**, with an accuracy delta beside every speed delta (G-P2). A 3.62x that changes decisions is
not a 3.62x — it is a silent regression with a benchmark attached.

WHAT IS CHECKED, in increasing order of what it would cost us to get wrong:
  1. per-tensor relative error of the engine's z_next vs eager fp32
  2. error GROWTH under a 20-step recursive roll — the operating condition. A 1-step error that
     looks negligible can compound; that is not speculation here, it is E-CR's measured finding
     (CR 3.50 -> 80.77), so a rollout-error check is mandatory rather than optional.
  3. the DECISION consequence: the ego waypoint trajectory the roll produces, in METRES, which is
     what the planner and the four families actually consume.

⚠️ Random weights are used (no trained checkpoint on Thor yet). That is legitimate for a NUMERICS
check — the arithmetic path is identical — but it CANNOT stand in for the four-family accuracy
gate on real windows, and this file must not be cited as if it did.
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
import tensorrt as trt
import torch

from tanitad.config import flagship4b_config
from tanitad.models.fourbrain import WorldModel
from train_flagship_v4 import resolve_v2_frames

DEV, K = "cuda", 20
OUT = os.path.expanduser("~/trt")
out = {"trt": trt.__version__, "note": "random weights — NUMERICS check, NOT the four-family gate"}

cfg = flagship4b_config()
ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0,
                     projection="cylindrical", v2_subframe="176x624", f_ref=None)
resolve_v2_frames(ns, cfg, label="trt_acc")
cfg.speed_input = True
cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
if getattr(cfg, "tactical_pred", None) is not None:
    cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
model = WorldModel(cfg).to(DEV).eval()
Wn, A, S = cfg.predictor.window, cfg.predictor.action_dim, model.state_dim


class Engine:
    """Minimal TRT v3 runtime wrapper: bind, execute, return the output tensor."""

    def __init__(self, path):
        logger = trt.Logger(trt.Logger.ERROR)
        with open(path, "rb") as f:
            self.eng = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.eng.create_execution_context()
        self.names = [self.eng.get_tensor_name(i)
                      for i in range(self.eng.num_io_tensors)]
        self.inputs = [n for n in self.names
                       if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.output = [n for n in self.names if n not in self.inputs][0]

    def __call__(self, states, actions):
        feed = {self.inputs[0]: states.contiguous(),
                self.inputs[1]: actions.contiguous()}
        # bind by NAME, never by position — a positional bind that silently swaps
        # states/actions would produce plausible garbage
        for n, t in feed.items():
            self.ctx.set_tensor_address(n, t.data_ptr())
        shp = tuple(self.ctx.get_tensor_shape(self.output))
        o = torch.empty(shp, device=DEV, dtype=torch.float32)
        self.ctx.set_tensor_address(self.output, o.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        return o


torch.manual_seed(0)
st0 = torch.randn(1, Wn, S, device=DEV)
ac = torch.randn(1, Wn, A, device=DEV)

for tag in ("fp32", "fp16"):
    path = f"{OUT}/predictor_{tag}.plan"
    if not os.path.exists(path):
        out[tag] = {"skipped": "engine missing"}
        continue
    try:
        eng = Engine(path)
        with torch.no_grad():
            # --- (1) one step ---
            ref1 = model.predictor(st0, ac)[1].float()
            trt1 = eng(st0, ac).float()
            r1 = float((trt1 - ref1).norm() / ref1.norm())

            # --- (2) 20-step roll, both paths ---
            ws = st0
            for _ in range(K):
                z = model.predictor(ws, ac)[1]
                ws = torch.cat([ws[:, 1:], z.unsqueeze(1)], dim=1)
            ref_roll = ws.float()

            ws = st0
            for _ in range(K):
                z = eng(ws, ac)
                ws = torch.cat([ws[:, 1:], z.unsqueeze(1)], dim=1)
            trt_roll = ws.float()
            rK = float((trt_roll - ref_roll).norm() / ref_roll.norm())

        out[tag] = {
            "rel_err_1step": round(r1, 8),
            "rel_err_20step_roll": round(rK, 8),
            "error_growth_x": round(rK / max(r1, 1e-12), 2),
            "verdict": ("PASS — engine is numerically equivalent under rollout"
                        if rK < 1e-2 else
                        "⛔ FAIL — error compounds beyond 1 % over the roll; "
                        "do not deploy without a four-family gate"),
        }
    except Exception as e:
        out[tag] = {"FAILED": f"{type(e).__name__}: {str(e)[:220]}"}

out["bars"] = {"their_decision_agreement_bar": "95.3 % vs fp32",
               "⛔ still_required": "four-family accuracy gate on REAL windows with a TRAINED "
                                   "checkpoint — this file is numerics only"}
print(json.dumps(out, indent=1), flush=True)
with open(os.path.expanduser("~/thor_trt_accuracy.json"), "w") as f:
    json.dump(out, f, indent=1)
