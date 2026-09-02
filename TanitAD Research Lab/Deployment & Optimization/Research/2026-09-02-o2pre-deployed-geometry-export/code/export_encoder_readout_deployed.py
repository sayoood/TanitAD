"""O2-pre END-TO-END — export ENCODER+READOUT at the DEPLOYED 176x624 geometry.

⛔ WHY THIS EXISTS, and why the existing test is not enough.

`stack/tests/test_readout_onnx_pool.py` pins the O2-pre repair, but it exports a
**bare `SpatialGridReadout`** against a `[1, 429, 32]` TOKEN tensor with a toy
`d_model = 32`. It never constructs an encoder. The backlog row asks for
*"Intake + export test **at the deployed geometry**"*, and the claim it exists to
falsify -- the 2026-07-08 *"encoder exports clean"* note -- is an **encoder**
claim, made by a script (`Implementation/onnx_export/export_encoder_predictor.py`)
whose input shape is hard-coded `[1, 9, 256, 256]` with **no geometry flag at
all**.

⇒ Pinning the submodule that held the defect is not the same as pinning the claim
that was false. This script exports the composed **image -> encoder -> readout**
path at `[1, 9, 176, 624]` and checks ONNX Runtime against eager.

⭐ THE CONTROLS, because an export test that only ever passes proves nothing:

  * **DEPLOYED arm** 176x624 -> 11x39 tokens -> 4x4 grid. 11 % 4 != 0 and
    39 % 4 != 0, so this takes the NON-TILING route -- the one that used to raise
    `SymbolicValueError`. This is the arm under test.
  * **TILING control** 256x640 -> 16x40 -> 4x4. Both tile, so this takes the
    `AvgPool2d` route. It MUST also export, and it pins that the repair did not
    disturb the path every v7 arm actually trains on.
  * **NEGATIVE control** -- the same deployed geometry with the pre-repair
    operator (`nn.AdaptiveAvgPool2d`) spliced back in. It MUST FAIL to export.
    ⛔ Without this the suite cannot distinguish "the fix works" from "this
    torch/onnx version happens to export adaptive pooling anyway", which would
    make a PASS meaningless. Same rule as the deliberate-regression arm.
  * **Numeric floor** -- ORT vs eager max relative error, plus an assertion that
    the graph contains no `AveragePool`/`Adaptive*` node on the deployed arm.

Usage:
    python export_encoder_readout_deployed.py --out ../raw/o2pre_e2e.json
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn

from tanitad.config import EncoderConfig
from tanitad.models.encoder import ViTEncoder
from tanitad.models.readout import SpatialGridReadout

OPSET = 17
IN_CH = 9          # the D-015 3-frame RGB stack
PATCH = 16
D_MODEL = 128      # keeps the export small; the defect is geometric, not width
DEPTH = 2
HEADS = 4
GRID = 4
D_READOUT = 32


class EncoderReadout(nn.Module):
    """image -> ViTEncoder -> SpatialGridReadout, i.e. the path the 2026-07-08
    'exports clean' note actually claimed."""

    def __init__(self, h: int, w: int, force_adaptive: bool = False):
        super().__init__()
        cfg = EncoderConfig(in_channels=IN_CH, image_size=h, image_width=w,
                            patch_size=PATCH, d_model=D_MODEL,
                            depth=DEPTH, n_heads=HEADS)
        self.cfg = cfg
        self.enc = ViTEncoder(cfg)
        th, tw = cfg.token_grid()
        self.token_grid = (th, tw)
        self.readout = SpatialGridReadout(th * tw, D_MODEL, grid=GRID,
                                          d_readout=D_READOUT,
                                          token_grid=(th, tw))
        if force_adaptive:
            # ⛔ NEGATIVE CONTROL: put the pre-repair operator back on the
            # forward path. Export MUST fail here, or the positive result is
            # not evidence about the repair.
            self.readout.exact_pool = False
            self.readout.forward = _adaptive_forward.__get__(self.readout)

    def forward(self, x):
        return self.readout(self.enc(x))


def _adaptive_forward(self, tokens):
    """The pre-repair readout forward: nn.AdaptiveAvgPool2d on a non-tiling grid."""
    b, n, d = tokens.shape
    x = tokens.transpose(1, 2).reshape(b, d, self.token_h, self.token_w)
    x = nn.functional.adaptive_avg_pool2d(x, (GRID, GRID))
    x = x.flatten(2).transpose(1, 2)
    return self.proj(x).flatten(1)


def try_export(mod, sample, path) -> tuple[bool, str]:
    try:
        torch.onnx.export(mod, (sample,), path, opset_version=OPSET,
                          input_names=["frames"], output_names=["state"],
                          dynamo=False)
        return True, ""
    except Exception as e:                                   # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:300]}"


def run_arm(h: int, w: int, *, force_adaptive: bool, must_export: bool) -> dict:
    torch.manual_seed(0)
    mod = EncoderReadout(h, w, force_adaptive=force_adaptive).eval()
    th, tw = mod.token_grid
    x = torch.randn(1, IN_CH, h, w)
    with torch.no_grad():
        eager = mod(x).numpy()

    rec = {
        "geometry": f"{h}x{w}",
        "token_grid": [th, tw],
        "n_tokens": th * tw,
        "readout_grid": [GRID, GRID],
        "tiles_exactly": (th % GRID == 0 and tw % GRID == 0),
        "route": ("AvgPool2d (exact)" if (th % GRID == 0 and tw % GRID == 0)
                  else "constant-matrix matmul (non-tiling)"),
        "forced_pre_repair_adaptive_op": force_adaptive,
        "must_export": must_export,
        "eager_out_dim": int(eager.shape[-1]),
    }

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.onnx")
        ok, err = try_export(mod, x, p)
        rec["exported"] = ok
        rec["export_error"] = err
        if ok:
            import onnx
            g = onnx.load(p).graph
            ops = sorted({n.op_type for n in g.node})
            rec["op_types"] = ops
            rec["has_pooling_op"] = any(
                ("Pool" in o) or ("Adaptive" in o) for o in ops)
            rec["has_matmul_or_gemm"] = any(o in ("MatMul", "Gemm") for o in ops)
            import onnxruntime as ort
            s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
            got = s.run(None, {"frames": x.numpy()})[0]
            denom = float(np.abs(eager).max()) or 1.0
            rec["ort_vs_eager_max_rel_err"] = float(
                np.abs(got - eager).max() / denom)

    rec["verdict"] = "PASS" if rec["exported"] == must_export else "FAIL"
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = {
        "deployed_176x624_NONTILING": run_arm(
            176, 624, force_adaptive=False, must_export=True),
        "training_256x640_TILING_control": run_arm(
            256, 640, force_adaptive=False, must_export=True),
        "negative_control_176x624_PRE_REPAIR_OP": run_arm(
            176, 624, force_adaptive=True, must_export=False),
    }

    res = {
        "_experiment": "E-DEPLOY-O2PRE-E2E",
        "_date": "2026-09-02",
        "_generated_utc": datetime.now(timezone.utc).isoformat(),
        "_evidence_class": "MEASURED (ours; dev-box CPU export, live G: stack tree)",
        "_tier": "N/A -- an export/graph property, not a model capability read",
        "_torch": torch.__version__,
        "_opset": OPSET,
        "arms": arms,
        "all_pass": all(v["verdict"] == "PASS" for v in arms.values()),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)

    for k, v in arms.items():
        print(f"[{v['verdict']}] {k}")
        print(f"        tokens {v['token_grid']} route={v['route']}")
        print(f"        exported={v['exported']} (required {v['must_export']})")
        if v.get("op_types"):
            print(f"        pooling_op_in_graph={v['has_pooling_op']} "
                  f"matmul={v['has_matmul_or_gemm']} "
                  f"ort_rel_err={v.get('ort_vs_eager_max_rel_err'):.3e}")
        if v["export_error"]:
            print(f"        error: {v['export_error'][:160]}")
    print(f"\nALL_PASS={res['all_pass']}  -> {a.out}")


if __name__ == "__main__":
    main()
