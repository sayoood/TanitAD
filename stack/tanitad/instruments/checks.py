"""Instrument doctrine I1-I4 (D-004) — the measurement harness is validated
BEFORE the model is judged. No model claim ships without these rows.

I1 oracle-decode: decode REAL futures per candidate; ranking must be ~perfect,
   else the harness (not the model) is broken.
I2 batch-consistency: encode(frame, batch=1) == encode(frame, batch=B) to 1e-4.
   Deployment is batch-1 streaming; batch-statistic layers violate this silently.
I3 route/episode-level splits: probes fitted on random-frame splits leak scene
   identity (measured 4x optimistic in ALPS-4B).
I4 persistence baseline: imag_relative < 1 or the predictor is worse than
   doing nothing; no predictive claim below this bar.
"""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def i1_oracle_decode_row(probe, real_future_states: Tensor, targets: Tensor) -> float:
    """R^2 of the probe on REAL future latents. Must be high (~>0.9 on toy)
    before any imagination-decode claim is made with the same probe family."""
    return probe.r2(real_future_states, targets)


@torch.no_grad()
def i2_batch_consistency(encode_fn, frames: Tensor, batch_size: int = 32,
                         tol: float = 1e-4) -> tuple[bool, float]:
    """encode_fn: [B, ...] -> [B, S]. Compares batch-1 vs batched encodings.

    Returns (pass, max relative deviation). Any deviation means a
    batch-statistic layer sits in the inference path — banned (D-004).
    """
    frames = frames[:batch_size]
    z_batched = encode_fn(frames)
    z_single = torch.cat([encode_fn(frames[i:i + 1]) for i in range(frames.shape[0])])
    denom = z_batched.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rel = ((z_batched - z_single).norm(dim=-1, keepdim=True) / denom).max()
    return bool(rel < tol), float(rel)


def i3_episode_split(episode_ids: list[int], val_frac: float = 0.2,
                     seed: int = 0) -> tuple[list[int], list[int]]:
    """Disjoint EPISODE-level split. In real data: split by drive/route/day."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(episode_ids), generator=g).tolist()
    n_val = max(1, int(len(episode_ids) * val_frac))
    val = [episode_ids[i] for i in perm[:n_val]]
    train = [episode_ids[i] for i in perm[n_val:]]
    assert not set(train) & set(val)
    return train, val


@torch.no_grad()
def i4_imag_relative(z_pred: Tensor, z_true: Tensor, z_prev: Tensor) -> float:
    """||z_hat - z_true|| / ||z_true - z_prev|| (mean over batch).
    < 1: predictor beats persistence. >= 1: no predictive claim allowed."""
    scale = (z_true - z_prev).norm(dim=-1).mean().clamp_min(1e-8)
    return float((z_pred - z_true).norm(dim=-1).mean() / scale)


def instrument_rows(**rows) -> dict:
    """Standard instrument block for every metrics.json (protocol §6)."""
    return {"instruments": rows}
