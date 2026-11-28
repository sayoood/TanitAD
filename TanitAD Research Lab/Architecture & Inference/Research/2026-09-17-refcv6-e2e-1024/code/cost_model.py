"""What does one 256 x 1024 step actually COST, and what would it cost on a GPU?

The CPU seconds/clip in ``raw/forward.json`` are MEASURED but say nothing about
a pod. To project, the arithmetic has to be MEASURED too, so the multiply-adds
are counted from the SHAPES A REAL FORWARD PRODUCES -- forward hooks on every
``Conv2d`` and ``Linear``, no analytic model of the architecture, no published
FLOP table scaled by a pixel ratio.

⛔ The projection is then ESTIMATED and says so, and it states the achieved-
throughput assumption it divides by rather than hiding it in a single number.

⚠️ It also separates the two halves the SPEC's compute plan confuses:
``K x W`` trunk passes (the dominant term) and everything downstream of them.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

WT = Path(os.environ.get("TANITAD_WT", Path(__file__).resolve().parents[5]))
sys.path.insert(0, str(WT / "stack"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


def count_macs(module: nn.Module, run) -> tuple[int, dict]:
    """MACs of one call of ``run()``, counted from REAL tensor shapes."""
    total = {"conv": 0, "linear": 0}
    hs = []

    def conv_hook(m, inp, out):
        total["conv"] += int(out.numel() * m.in_channels //
                             m.groups * m.kernel_size[0] * m.kernel_size[1])

    def lin_hook(m, inp, out):
        total["linear"] += int(out.numel() * m.in_features)

    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            hs.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hs.append(m.register_forward_hook(lin_hook))
    try:
        run()
    finally:
        for h in hs:
            h.remove()
    return total["conv"] + total["linear"], total


def main():
    from e2e_1024 import frame_of_cache
    from tanitad.models import timm_trunk as tt
    cache = os.environ.get(
        "E2E_CACHE", "D:/Projects/TanitAD-artifacts/v2ep-eval139-256x1024cyl")
    fr = frame_of_cache(cache)
    W = int(os.environ.get("E2E_WINDOW", "8"))
    res = {"evidence_class": "MEASURED (MAC count) + ESTIMATED (projection)",
           "frame": {"height": fr.height, "width": fr.width},
           "window_positions": W, "arms": {}}
    for name in ("resnet101.a1_in1k", "resnet34.a1_in1k"):
        trunk = tt.build_timm_trunk(in_channels=9, model_name=name,
                                    mode="shared", fuse="concat1x1",
                                    fuse_identity_init=True,
                                    image_hw=(fr.height, fr.width))
        trunk.eval()
        x = torch.zeros(1, 9, fr.height, fr.width)
        with torch.no_grad():
            t0 = time.time()
            macs, per = count_macs(trunk, lambda: trunk.forward_features(x))
            t_one = time.time() - t0
        row = {
            "macs_one_window_position": macs,
            "macs_by_kind": per,
            "gflops_one_window_position_fwd": 2.0 * macs / 1e9,
            "window_positions": W,
            "gflops_one_sample_fwd": 2.0 * macs * W / 1e9,
            # backward is ~2x the forward MACs (dgrad + wgrad)
            "gflops_one_sample_fwd_plus_bwd": 3.0 * (2.0 * macs * W) / 1e9,
            "cpu_s_one_window_position_fwd_measured": t_one,
            "n_trunk_passes_per_sample": W * (9 // 3),
        }
        res["arms"][name] = row
        print("[cost] %-20s %.1f GFLOP/sample fwd+bwd (trunk only)"
              % (name, row["gflops_one_sample_fwd_plus_bwd"]), flush=True)
        del trunk, x

    # ---- tie it to the MEASURED wall clock ------------------------------- #
    fwd = Path(__file__).resolve().parents[1] / "raw" / "forward.json"
    logs = Path(__file__).resolve().parents[1] / "logs" / "forward_full139.jsonl"
    ts = []
    if logs.is_file():
        for line in logs.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    ts.append(json.loads(line)["t_s"])
                except Exception:
                    pass
    elif fwd.is_file():
        ts = [r["t_s"] for r in json.loads(fwd.read_text(encoding="utf-8"))["rows"]]
    if ts:
        import statistics as st
        mean_s = st.mean(ts)
        g = res["arms"]["resnet101.a1_in1k"]["gflops_one_sample_fwd_plus_bwd"]
        res["measured_cpu"] = {
            "n_clips": len(ts), "mean_s_per_clip": mean_s,
            "trunk_gflops_per_clip": g,
            "effective_cpu_gflops": g / mean_s,
            "note": "trunk only; the true achieved rate is higher because the "
                    "clip also decodes PNGs and runs the whole planner",
        }
        res["projection"] = {
            "evidence_class": "ESTIMATED",
            "assumption": "an A40 sustains 40-60 TFLOP/s bf16 on convs of this "
                          "shape (30-40 pct of the 150 TFLOP/s peak); the "
                          "dev-box RTX 4060 sustains ~5-8 TFLOP/s",
            "a40_s_per_clip_trunk_only": [g / 60e3, g / 40e3],
            "rtx4060_s_per_clip_trunk_only": [g / 8e3, g / 5e3],
            "a40_s_for_139_clips": [139 * g / 60e3, 139 * g / 40e3],
            "rtx4060_s_for_139_clips": [139 * g / 8e3, 139 * g / 5e3],
            "cpu_s_for_139_clips_measured_rate": 139 * mean_s,
            "caveat": "COMPUTE ONLY. A real pod step is bounded by the PNG "
                      "decode and the v2 LRU as often as by the GPU -- the "
                      "epcache rule (CLAUDE.md: price what the CONSUMER reads).",
        }
    out = Path(__file__).resolve().parents[1] / "raw" / "cost_model.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
