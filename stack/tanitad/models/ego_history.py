"""refcv6 §2b — the EGO-HISTORY encoder (PI 2026-09-16).

⭐ The PI's words: *"I think it is important to process the image frame history
and also ego data history as inputs for our refcv6"*. Today only **v0 at t0**
reaches the model (`refc_v3_train.py:2311`, `v0 = pose_last[:, 3]`); the rest of
the observed window's ego state — how the speed got here, whether the car is
braking, whether it is mid-turn — is discarded before the condition is built.

⛔⛔ **ADMISSIBILITY, AND IT IS THE WHOLE POINT OF THIS FILE.** The binding PI
ruling of 2026-09-02 is that *measured v0 at t0 is a legal initial state*. PAST
ego data is the same class: it is measured, it is available to a real vehicle at
tick time, and it says nothing about the future. **Ego FUTURE is not**, and this
programme has been burned by exactly that seam before (`--ego-dropout`,
`withheld_speed`, `flagship_v15`'s `vt_keep`: "a goal that was withheld from the
decoder must not sneak back in through the ranking").

So the encoder does not take "the past"; it takes **the whole sequence and the
number of leading steps that are past**, and slices. That turns admissibility
into something a test can MUTATE rather than read:

    seq2 = seq.clone(); seq2[:, n_past:] = <anything>
    assert torch.equal(enc(seq, n_past), enc(seq2, n_past))

If a future index is ever read, that assertion goes red. An encoder that simply
accepted a pre-sliced ``[B, T_past, C]`` could not be tested this way at all —
the slicing would happen in a caller nobody audits.

============================================================================
THE CHANNELS
============================================================================
Three per step, all derived from the window's own pose track
``(x, y, yaw, v)``:

* **speed** ``v`` (m/s) — measured;
* **longitudinal acceleration** ``dv/dt`` (m/s²) — a FIRST DIFFERENCE of the
  past speeds, so step ``i`` uses ``i`` and ``i-1`` and never ``i+1``;
* **yaw rate** ``dyaw/dt`` (rad/s) — the same, with wrapping.

⛔ **The differences are BACKWARD, and that is load-bearing.** A centred
difference at the last past step would read ``v[n_past]`` — the first FUTURE
sample — and leak a measurement no vehicle has yet. The mutation test above is
what catches it; :func:`ego_channels_from_poses` is where it would happen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

__all__ = [
    "EgoHistoryConfig", "EgoHistoryEncoder", "ego_channels_from_poses",
    "EGO_CHANNELS",
]

#: (speed, longitudinal acceleration, yaw rate)
EGO_CHANNELS: int = 3


def _wrap_pi(a: Tensor) -> Tensor:
    """Wrap an angle difference into ``(-pi, pi]``.

    ⛔ Without this a heading crossing +-pi reads as a ~6.28 rad/s yaw rate —
    a physically impossible number that a network will happily fit as "this is
    a sharp turn", once per lap of the compass.
    """
    return (a + math.pi) % (2 * math.pi) - math.pi


def ego_channels_from_poses(poses: Tensor, n_past: int,
                            dt: float = 0.1) -> Tensor:
    """``[B, T, 4]`` poses ``(x, y, yaw, v)`` -> ``[B, n_past, 3]`` ego channels.

    ⛔ Reads ONLY ``poses[:, :n_past]``. Every derivative is a BACKWARD
    difference, and step 0 — which has no predecessor inside the window — gets
    a zero rate rather than borrowing step 1's. Borrowing forward is the
    subtle version of a future read: it is still inside the past window here,
    but the same habit applied at the last step reads the first future sample.
    """
    if poses.ndim != 3 or poses.shape[-1] < 4:
        raise ValueError(
            f"ego_channels_from_poses expects [B, T, >=4] (x, y, yaw, v), got "
            f"{tuple(poses.shape)}")
    n = int(n_past)
    if n < 1 or n > poses.shape[1]:
        raise ValueError(
            f"n_past {n_past} outside [1, {poses.shape[1]}] — a window with no "
            f"past has no ego history, and one longer than the tensor would "
            f"silently read the future.")
    past = poses[:, :n]                                   # ⛔ THE ONLY SLICE
    v = past[..., 3]
    yaw = past[..., 2]
    dv = torch.zeros_like(v)
    dv[:, 1:] = (v[:, 1:] - v[:, :-1]) / float(dt)
    dyaw = torch.zeros_like(yaw)
    dyaw[:, 1:] = _wrap_pi(yaw[:, 1:] - yaw[:, :-1]) / float(dt)
    return torch.stack([v, dv, dyaw], dim=-1)             # [B, n, 3]


@dataclass
class EgoHistoryConfig:
    """⛔ ``enable=False`` (the DEFAULT) constructs NOTHING."""

    enable: bool = False
    steps: int = 8                  # the observed window, `core.window`
    channels: int = EGO_CHANNELS
    hidden: int = 64
    out_dim: int = 32
    kind: str = "gru"               # "gru" | "conv1d"
    dt: float = 0.1
    #: ⭐ Zero-init the output projection so the condition is UNCHANGED at
    #: step 0. That makes ego history a removable graft: any later difference
    #: is learned, not an artefact of having added a vector to the condition.
    zero_init_out: bool = True

    def __post_init__(self) -> None:
        if str(self.kind) not in ("gru", "conv1d"):
            raise ValueError(f"kind {self.kind!r} not in ('gru', 'conv1d')")
        if int(self.steps) < 2:
            raise ValueError(
                f"steps {self.steps} < 2 — a one-step 'history' carries no "
                f"rate at all and the arm would be `v0` under a new name.")

    def as_dict(self) -> dict:
        return {"enable": bool(self.enable), "steps": int(self.steps),
                "channels": int(self.channels), "hidden": int(self.hidden),
                "out_dim": int(self.out_dim), "kind": str(self.kind),
                "dt": float(self.dt),
                "zero_init_out": bool(self.zero_init_out)}


class EgoHistoryEncoder(nn.Module):
    """``[B, T, C]`` + ``n_past`` -> ``[B, out_dim]``.

    ``kind="gru"`` is a single-layer GRU read at its last PAST step;
    ``kind="conv1d"`` is a causal dilated 1-D stack. Both are strictly
    causal by construction, and :meth:`forward` proves it by slicing before it
    computes anything.
    """

    def __init__(self, cfg: EgoHistoryConfig | None = None):
        super().__init__()
        self.cfg = cfg or EgoHistoryConfig(enable=True)
        c, h = int(self.cfg.channels), int(self.cfg.hidden)
        if str(self.cfg.kind) == "gru":
            self.body = nn.GRU(c, h, batch_first=True)
        else:
            # ⛔ LEFT-PADDED, so output step i depends only on inputs <= i.
            # A symmetric `padding=k//2` would make the last output read the
            # future if this were ever run on an unsliced sequence.
            self.body = nn.Sequential(
                nn.ConstantPad1d((2, 0), 0.0), nn.Conv1d(c, h, 3), nn.ReLU(),
                nn.ConstantPad1d((4, 0), 0.0), nn.Conv1d(h, h, 3, dilation=2),
                nn.ReLU())
        self.out = nn.Linear(h, int(self.cfg.out_dim))
        if self.cfg.zero_init_out:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)

    def forward(self, seq: Tensor, n_past: int | None = None) -> Tensor:
        """⛔ ``seq`` MAY contain future steps; they are sliced away HERE.

        Passing ``n_past=None`` means "the whole tensor is past" and is the
        convenience path for a caller that already sliced. The mutation test
        uses the explicit form, because that is the one that can go red.
        """
        if seq.ndim != 3:
            raise ValueError(f"ego history expects [B, T, C], got "
                             f"{tuple(seq.shape)}")
        n = int(seq.shape[1] if n_past is None else n_past)
        if n < 1 or n > seq.shape[1]:
            raise ValueError(
                f"n_past {n_past} outside [1, {seq.shape[1]}]")
        past = seq[:, :n]                                 # ⛔ THE ONLY SLICE
        if isinstance(self.body, nn.GRU):
            _, hN = self.body(past)
            feat = hN[-1]
        else:
            feat = self.body(past.transpose(1, 2))[..., -1]
        return self.out(feat)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
