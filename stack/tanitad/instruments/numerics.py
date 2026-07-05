"""Pinned numerics for measurement paths (I2 hardening, found 2026-07-06).

Finding: at 261 M scale on CUDA, batch-1 vs batched encodings deviated by
~8.8e-4 relative — no batch-statistics layer exists in the model; the cause is
TF32/cuDNN selecting different kernels (different reduction orders, ~1e-3
precision) per batch size. Harmless for TRAINING throughput; NOT harmless for
measurement: frozen probes are calibrated on batched encodings and consumed
batch-1 at deployment, so the measurement path must be bit-stable across
batch sizes.

Doctrine: every probe fit, every gate evaluation, and the I2 check itself run
inside strict_numerics(). Training keeps fast kernels.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch


@contextmanager
def strict_numerics():
    """Disable TF32 and cuDNN autotuning for bit-stable eval across batch sizes."""
    if not torch.cuda.is_available():
        yield
        return
    prev = (torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.benchmark)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        (torch.backends.cuda.matmul.allow_tf32,
         torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark) = prev
