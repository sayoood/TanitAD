#!/usr/bin/env python3
"""Is `uint8 -> float32 / 255` on the GPU bit-identical to train.py:832's `astype(np.float32) / 255.0` on the CPU?
Checks all 256 values, for a CPU-scalar divisor (PyTorch may multiply by the reciprocal) and a CUDA 0-dim divisor."""
import numpy as np
import torch

ref = np.arange(256, dtype=np.uint8).astype(np.float32) / 255.0          # train.py's expression (cv2 output is uint8)
u8 = torch.arange(256, dtype=torch.uint8, device="cuda")
cands = {
    "gpu x.float() / 255.0 (CPU scalar)": (u8.float() / 255.0),
    "gpu x.float() / cuda_t255 (0-dim CUDA tensor)": (u8.float() / torch.full((), 255.0, device="cuda")),
    "gpu x.float().div(full_like 255)": (u8.float().div(torch.full((256,), 255.0, device="cuda"))),
    "cpu torch x.float() / 255.0": (u8.cpu().float() / 255.0),
}
print("numpy", np.__version__, "torch", torch.__version__)
for k, v in cands.items():
    got = v.cpu().numpy()
    nbad = int((got.view(np.uint32) != ref.view(np.uint32)).sum())
    print(f"{k:48s} bit-identical: {nbad == 0}  (mismatching values: {nbad}/256)")
