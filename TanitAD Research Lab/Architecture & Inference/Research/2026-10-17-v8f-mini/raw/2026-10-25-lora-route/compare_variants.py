"""Probe 4: combine the dumps of probe 2 into the DECOMPOSITION of the 0.079 / 0.090 'route gap' (pooled fp8 [T,40,1024], clips 8f387977 and bf5ec76e).  CPU only; reads the .pt dumps + json of repro_route.py.

Terms (all rel-L2 of pooled fp8, same frames steps within each row):
  DEVICE-ONLY   thor416 frames, CPU fp32 (test_cpu_fp32) vs dev-box GPU bf16 (test_runscript)                  -> what a CPU-fp32 reproduction can NEVER get below vs a GPU-bf16 cache
  DTYPE-ONLY    production frames (native256 rebuild), CPU bf16 vs CPU fp32, the same 12 steps                 -> the bf16-vs-fp32 part of that, MEASURED on the production route's own frames
  ROUTE-ONLY    native256 rebuild vs the thor416 cache, BOTH CPU fp32 (device matched)
  ROUTE+DEVICE  native256 rebuild vs the thor416 cache, rebuild in CPU bf16 / fp32, cache GPU bf16   (the identity control's ROUTE number)
  LOCAL416      local 416 rebuild -> resize vs the thor416 cache, CPU fp32 both: does the LOCAL builder reproduce THOR's 416 product?
  NATIVE-vs-LOCAL416  the two LOCAL routes against each other (CPU fp32 both)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

D = Path(__file__).resolve().parent
REF = {"cpu_fp32": Path("C:/Users/Admin/refav1_hj_cache/test_cpu_fp32/train/clips"), "gpu_bf16": Path("C:/Users/Admin/refav1_hj_cache/test_runscript/train/clips")}
CLIPS = ["8f387977-4086-466d-bba4-b04f8cdc0743", "bf5ec76e-e8b5-4f0d-a9cc-4fe1687d3d3a"]


def st(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return {"rel_l2": float((a - b).norm() / b.norm()), "max_abs": float((a - b).abs().max()), "frac_bitwise_equal": float((a == b).float().mean()), "n_values": int(a.numel())}


def load(stem):
    p = D / f"{stem}.pt"
    return torch.load(p, map_location="cpu", weights_only=False) if p.is_file() else None


def ref(name, c, steps):
    return torch.load(REF[name] / f"{c}.pt", map_location="cpu", weights_only=True)["pool4x10"][steps]


def main() -> int:
    V = {k: load(k) for k in ("repro_native256_fp32", "repro_local416_fp32", "repro_native256_bf16_control", "repro_native256_fp32_12", "repro_native256_fp32_full")}
    out = {"available": [k for k, v in V.items() if v is not None], "per_clip": {}}
    for c in CLIPS:
        r = {}
        # DEVICE-ONLY on the 21-step grid (and on all frames in probe_provenance.json)
        if V["repro_native256_fp32"]:
            s21 = V["repro_native256_fp32"][c]["steps"]
            r["DEVICE_ONLY_thor416_cpu_fp32_vs_gpu_bf16__21steps"] = st(ref("cpu_fp32", c, s21), ref("gpu_bf16", c, s21))
            n = V["repro_native256_fp32"][c]["pooled4x10_fp8"]
            r["ROUTE_ONLY_native256_cpu_fp32_vs_thor416_cpu_fp32__21steps"] = st(n, ref("cpu_fp32", c, s21))
            r["ROUTE_PLUS_DEVICE_native256_cpu_fp32_vs_thor416_gpu_bf16__21steps"] = st(n, ref("gpu_bf16", c, s21))
            if V["repro_local416_fp32"]:
                l = V["repro_local416_fp32"][c]["pooled4x10_fp8"]
                r["LOCAL416_cpu_fp32_vs_thor416_cpu_fp32__21steps"] = st(l, ref("cpu_fp32", c, s21))
                r["LOCAL416_cpu_fp32_vs_thor416_gpu_bf16__21steps"] = st(l, ref("gpu_bf16", c, s21))
                r["NATIVE256_vs_LOCAL416_both_cpu_fp32__21steps"] = st(n, l)
        # DTYPE-ONLY on the 12-step grid (production frames)
        if V["repro_native256_bf16_control"] and V["repro_native256_fp32_12"]:
            assert V["repro_native256_bf16_control"][c]["steps"] == V["repro_native256_fp32_12"][c]["steps"]
            s12 = V["repro_native256_bf16_control"][c]["steps"]
            b, f = V["repro_native256_bf16_control"][c]["pooled4x10_fp8"], V["repro_native256_fp32_12"][c]["pooled4x10_fp8"]
            r["DTYPE_ONLY_production_frames_cpu_bf16_vs_cpu_fp32__12steps"] = st(b, f)
            r["CONTROL_native256_cpu_bf16_vs_thor416_gpu_bf16__12steps(identity_control_ROUTE_case)"] = st(b, ref("gpu_bf16", c, s12))
            r["native256_cpu_fp32_vs_thor416_gpu_bf16__12steps"] = st(f, ref("gpu_bf16", c, s12))
        out["per_clip"][c[:8]] = r
    # what the amendment can use: a CPU fp32 reference cache of the production route (all frames) vs the production md5 (cannot match: different device) -- recorded, not claimed
    if V["repro_native256_fp32_full"]:
        out["cpu_fp32_reference_cache_production_route"] = {c[:8]: {"steps": len(V["repro_native256_fp32_full"][c]["steps"]), "shape": list(V["repro_native256_fp32_full"][c]["pooled4x10_fp8"].shape)} for c in CLIPS}
        out["device_only_all_frames_see"] = "probe_provenance.json:device_only_floor_same_frames"
    json.dump(out, open(D / "compare_variants.json", "w", encoding="utf-8"), indent=1)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
