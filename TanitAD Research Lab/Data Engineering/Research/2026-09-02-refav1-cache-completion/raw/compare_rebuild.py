"""Equivalence of the CPU rebuild against (i) the Sep-1 kept-local GPU-bf16
artifact of the SAME episode from the original build and (ii) its own fp16
pre-quantisation field. Read-only; banks compare_rebuild.json."""
import json
import sys
from pathlib import Path

import torch

REB = Path("C:/Users/Admin/refav1_probe/rebuild")
SHIP = Path("C:/Users/Admin/refav1_probe/ship")
CLIP = "16d325e9-dbfe-439c-a0ed-4ff8850ad2e2"

new8 = torch.load(REB / f"{CLIP}.pt", map_location="cpu", weights_only=True)
new16 = torch.load(REB / f"{CLIP}.fp16.pt", map_location="cpu", weights_only=True)
old8 = torch.load(SHIP / f"{CLIP}.pt", map_location="cpu", weights_only=True)
rep = {"new_fp8": {"dtype": str(new8.dtype), "shape": list(new8.shape)},
       "new_fp16": {"dtype": str(new16.dtype), "shape": list(new16.shape)},
       "sep01_gpu_fp8": {"dtype": str(old8.dtype), "shape": list(old8.shape)}}
assert new8.shape == old8.shape == new16.shape, rep
n8, o8, n16 = new8.float(), old8.float(), new16.float()

def stats(a: torch.Tensor, b: torch.Tensor, name: str) -> dict:
    d = a - b
    return {"pair": name,
            "exact_equal_frac": float((a == b).float().mean()),
            "max_abs_diff": float(d.abs().max()),
            "mean_abs_diff": float(d.abs().mean()),
            "rel_mse": float((d ** 2).sum() / (b ** 2).sum()),
            "rel_l2": float(d.norm() / b.norm()),
            "cos_per_token_min": float(torch.nn.functional.cosine_similarity(
                a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), dim=-1).min()),
            "cos_per_token_mean": float(torch.nn.functional.cosine_similarity(
                a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), dim=-1).mean())}

rep["cpu_fp8_vs_sep01_gpu_fp8"] = stats(n8, o8, "cpu-bf16 rebuild (fp8) vs Sep-1 GPU-bf16 (fp8)")
rep["cpu_fp8_vs_cpu_fp16"] = stats(n8, n16, "cpu rebuild fp8 vs its own fp16 (quantisation only)")
rep["sep01_gpu_fp8_vs_cpu_fp16"] = stats(o8, n16, "Sep-1 GPU fp8 vs cpu fp16")
# how far apart are two fp8 codes that differ? (in e4m3 ulps of the larger)
diff = (n8 != o8)
rep["frac_elements_differ"] = float(diff.float().mean())
if diff.any():
    a, b = n8[diff], o8[diff]
    ulp = torch.maximum(a.abs(), b.abs()).clamp_min(1e-6) / 8.0   # 3 mantissa bits
    rep["differing_elems_median_ulps"] = float(((a - b).abs() / ulp).median())
    rep["differing_elems_max_ulps"] = float(((a - b).abs() / ulp).max())
rep["mean_abs"] = {"new_fp8": float(n8.abs().mean()), "sep01_gpu_fp8": float(o8.abs().mean()),
                   "new_fp16": float(n16.abs().mean())}
rep["per_frame_rel_l2_max"] = float(max((n8[t] - o8[t]).norm() / o8[t].norm()
                                        for t in range(n8.shape[0])))
print(json.dumps(rep, indent=1))
(REB / "compare_rebuild.json").write_text(json.dumps(rep, indent=1))
print("banked ->", REB / "compare_rebuild.json")
