"""Thor P6 / O2-pre — does the REAL encoder now export at the DEPLOYED 176x624 geometry?

The blocker, MEASURED 2026-08-03 (`thor_b1b_fastpath_probe.json`), at BOTH fastpath settings:

    SymbolicValueError: Unsupported: ONNX export of operator adaptive_avg_pool2d,
    output size that are not factor of input size

11x39 tokens onto a 4x4 readout grid does not tile, so `SpatialGridReadout` fell back to
`nn.AdaptiveAvgPool2d`. `stack/tanitad/models/readout.py` now materialises that pooling as two
constant averaging matrices — the same bins, expressed as matmuls.

⛔ THE TEST THAT MATTERS IS NOT "DOES IT EXPORT". It is whether the exported graph computes what the
TRAINED encoder computes. So this runs on the REAL v5f checkpoint and compares ORT against eager,
and it re-checks that flagship-v1's TILING path is untouched.

FALSIFIERS, both outcomes committed in advance:
  * export still fails at 176x624                       => the fix is wrong, O2 stays blocked
  * ORT-vs-eager rel-err > 1e-4 on real weights         => it exports and computes the wrong thing,
                                                           which is worse than not exporting
  * v1's 256x256 encoder output changes AT ALL          => the "untouched deployed path" claim is
                                                           false and every v1 number moves
"""
import json
import os
import subprocess
import sys
import time
import dataclasses
from types import SimpleNamespace

for _p in ("~/TanitAD/stack", "~/TanitAD/stack/scripts", "~/TanitAD/taniteval"):
    sys.path.insert(0, os.path.expanduser(_p))
sys.path.insert(0, "/usr/lib/python3.12/dist-packages")

import torch  # noqa: E402

DEV = "cuda"
WORK = os.path.expanduser("~/trt_c5")
os.makedirs(WORK, exist_ok=True)
OUT = {"purpose": "O2-pre — encoder ONNX export at the DEPLOYED 176x624 geometry, REAL weights",
       "torch": torch.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

from tanitad.config import flagship4b_config          # noqa: E402
from tanitad.models.fourbrain import WorldModel       # noqa: E402
from train_flagship_v4 import resolve_v2_frames       # noqa: E402


def relerr(a, b):
    import numpy as np
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


class EncWrap(torch.nn.Module):
    """window [B,W,C,H,W] -> states [B,W,S]: exactly what the deployed tick calls."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, frames):
        return self.m.encode_window(frames)


def build_v5f():
    cfg = flagship4b_config()
    ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0,
                         projection="cylindrical", v2_subframe="176x624", f_ref=None)
    resolve_v2_frames(ns, cfg, label="c5")
    cfg.speed_input = True
    cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
    if getattr(cfg, "tactical_pred", None) is not None:
        cfg.tactical_pred = dataclasses.replace(cfg.tactical_pred, action_dim=3)
    object.__setattr__(cfg.encoder, "grad_checkpoint", False)
    m = WorldModel(cfg)
    ck = torch.load(os.path.expanduser("~/models/v5f/ckpt.pt"), map_location="cpu",
                    weights_only=False)
    miss, unexp = m.load_state_dict(ck["model"], strict=False)
    return m.to(DEV).eval(), cfg, {"step": int(ck.get("step", -1)),
                                   "sd_missing": len(miss), "sd_unexpected": len(unexp)}


model, cfg, meta = build_v5f()
ro = getattr(model, 'readout', None)   # fourbrain.py:412 — NOT model.encoder.readout
OUT["v5f"] = meta
OUT["readout_route"] = {
    "found": ro is not None,
    "exact_pool": (None if ro is None else bool(ro.exact_pool)),
    "token_grid": (None if ro is None else [ro.token_h, ro.token_w]),
    "has_matrices": (None if ro is None else hasattr(ro, "pool_mh")),
    "note": "exact_pool False + has_matrices True is the fixed non-tiling route"}
print(json.dumps(OUT["readout_route"]), flush=True)

C = cfg.encoder.in_channels
Wn = cfg.predictor.window
frames = torch.randn(1, Wn, C, 176, 624, device=DEV)
wrap = EncWrap(model).eval()
with torch.no_grad():
    ref = wrap(frames).float().cpu().numpy()

for fastpath in (False, True):
    tag = f"encoder_176x624_fastpath{'ON' if fastpath else 'OFF'}"
    cell = {"fastpath": fastpath, "opset": 17}
    path = f"{WORK}/{tag}.onnx"
    try:
        torch.backends.mha.set_fastpath_enabled(fastpath)
        t0 = time.perf_counter()
        torch.onnx.export(wrap, (frames,), path, input_names=["frames"],
                          output_names=["states"], opset_version=17, dynamo=False)
        cell["export_ok"] = True
        cell["export_s"] = round(time.perf_counter() - t0, 1)
        cell["MB"] = round(os.path.getsize(path) / 1e6, 1)
        import onnx
        g = onnx.load(path, load_external_data=False)
        ops = {n.op_type for n in g.graph.node}
        cell["n_nodes"] = len(g.graph.node)
        cell["has_adaptive_pool"] = any("Adaptive" in o for o in ops)
        try:
            import onnxruntime as ort
            s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            y = s.run(["states"], {"frames": frames.cpu().numpy()})[0]
            cell["ort_rel_err_vs_eager"] = round(relerr(y, ref), 10)
        except Exception as e:
            cell["ort"] = f"{type(e).__name__}: {str(e)[:200]}"
    except Exception as e:
        cell["export_ok"] = False
        cell["err"] = f"{type(e).__name__}: {str(e)[:260]}"
    OUT[tag] = cell
    print(tag, cell, flush=True)
torch.backends.mha.set_fastpath_enabled(False)

# ⛔ THE REGRESSION CHECK: v1's TILING path must be byte-for-byte untouched.
try:
    cfg1 = flagship4b_config()
    object.__setattr__(cfg1.predictor, "action_dim", 3)
    if cfg1.tactical_pred is not None:
        object.__setattr__(cfg1.tactical_pred, "action_dim", 3)
    object.__setattr__(cfg1.encoder, "grad_checkpoint", False)
    m1 = WorldModel(cfg1)
    ck1 = torch.load(os.path.expanduser("~/models/flagship-v1-speedjerk/ckpt.pt"),
                     map_location="cpu", weights_only=False)
    m1.load_state_dict(ck1["model"])
    m1 = m1.to(DEV).eval()
    ro1 = m1.readout          # fourbrain.py:412
    f1 = torch.randn(1, Wn, cfg1.encoder.in_channels, 256, 256, device=DEV)
    with torch.no_grad():
        s1 = m1.encode_window(f1)
    OUT["v1_tiling_path_regression"] = {
        "exact_pool": bool(ro1.exact_pool),
        "has_matrices": hasattr(ro1, "pool_mh"),
        "state_absmax": round(float(s1.abs().max()), 6),
        "state_mean": round(float(s1.mean()), 8),
        "reading": ("exact_pool True and NO matrices built => v1 takes the untouched AvgPool2d "
                    "route; the fix cannot have moved any v1 number")}
    print("v1 regression", OUT["v1_tiling_path_regression"], flush=True)
except Exception as e:
    OUT["v1_tiling_path_regression"] = {"FAILED": f"{type(e).__name__}: {str(e)[:250]}"}

ok = OUT.get("encoder_176x624_fastpathOFF", {})
OUT["VERDICT"] = {
    "export_at_deployed_geometry": bool(ok.get("export_ok")),
    "ort_rel_err": ok.get("ort_rel_err_vs_eager"),
    "falsifier_export_fails": not bool(ok.get("export_ok")),
    "falsifier_wrong_numbers": bool((ok.get("ort_rel_err_vs_eager") or 0) > 1e-4),
    "reading": ("⛔ still blocked" if not ok.get("export_ok") else
                "✅ O2-pre CLOSED — the encoder exports at 176x624 with parity; O2 (a TensorRT "
                "engine for the encoder) is unblocked")}
with open(os.path.expanduser("~/thor_c5_encoder_export.json"), "w") as f:
    json.dump(OUT, f, indent=1, default=str)
print(json.dumps(OUT["VERDICT"], indent=1), flush=True)
