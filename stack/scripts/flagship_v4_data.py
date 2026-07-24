"""FlagshipV4Dataset — FlagshipWindowDataset + the v4 label set, on the fly.

``train_flagship_v4.v4_loss_step`` reads ``lat_target`` / ``lon_target`` /
``dist_target`` (the factorised tactical CE), the ``route`` / ``route_graded`` /
``vt_band`` / ``vt_speed`` conditioning goal, and — for the strategic-scalar loss
— ``strat_scalars`` / ``strat_scalar_mask``. ``FlagshipWindowDataset`` emits none
of them. This subclass ADDS exactly those keys and nothing else: every v1 / v2.1
key (``frames`` / ``future_frames`` / ``future_poses`` / ``pose_last`` /
``actions`` / ``nav_cmd`` / ``route_target`` / ``maneuver_label`` / ``pose_prev``)
passes through byte-identical, so a v1/v2.1 batch off this dataset is unchanged.

The labels are minted per window from the FULL episode poses (the factorised
labelers look 4–25 s ahead, past the window's own ``future_poses``), exactly as
``FlagshipWindowDataset`` already mints ``maneuver_label`` on the fly. This is the
CORRECTNESS/PROOF path; for a full 30 k-step run the same labels are precomputed
once into a v15-format cache by ``v4_labels.build`` (the recipe staged in the
repo, the multi-GB tensors pod-side) and indexed — the on-the-fly route-v3
curvature segmentation per window per epoch is too heavy at corpus scale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import v4_labels  # noqa: E402
from tanitad.lake.vtarget import savgol  # noqa: E402
from train_flagship4b import FlagshipWindowDataset  # noqa: E402

_LONG = ("lat_target", "lon_target", "dist_target", "stop_dist_target",
         "route", "route_token", "vt_band")
_FLOAT = ("route_graded", "vt_speed")


class FlagshipV4Dataset(FlagshipWindowDataset):
    """Additive v4 labels. Same ctor as ``FlagshipWindowDataset`` plus the mint
    knobs (``min_lookahead`` / ``use_net_dyaw``) that ``v4_labels`` /
    ``v15_prep`` share, so a v4 batch is parity-consistent with the v1.6 cache."""

    def __init__(self, episodes, window: int, max_horizon: int, maneuver_h: int,
                 channels: int | None = None, labels_v2: bool = False,
                 min_lookahead: int = 50, use_net_dyaw: bool = False):
        super().__init__(episodes, window, max_horizon, maneuver_h,
                         channels=channels, labels_v2=labels_v2)
        self.min_lookahead = min_lookahead
        self.use_net_dyaw = use_net_dyaw
        self._vs: dict[int, np.ndarray] = {}          # per-episode savgol memo

    def _smoothed_v(self, e_i: int, poses: torch.Tensor) -> np.ndarray:
        vs = self._vs.get(e_i)
        if vs is None:
            if len(self._vs) >= 16:
                self._vs.pop(next(iter(self._vs)))
            vs = self._vs[e_i] = savgol(poses[:, 3].numpy().astype(np.float64))
        return vs

    def __getitem__(self, i: int) -> dict:
        item = super().__getitem__(i)                 # v1/v2.1 keys, byte-identical
        e_i, t = self.index[i]
        poses = torch.as_tensor(self.episodes[e_i].poses, dtype=torch.float32)
        t_last = t + self.window - 1
        w = v4_labels.mint_window(poses, t_last,
                                  v_smoothed=self._smoothed_v(e_i, poses),
                                  min_lookahead=self.min_lookahead,
                                  use_net_dyaw=self.use_net_dyaw)
        for k in _LONG:
            item[k] = torch.tensor(w[k], dtype=torch.long)
        for k in _FLOAT:
            item[k] = torch.tensor(w[k], dtype=torch.float32)
        item["strat_scalars"] = w["strat_scalars"]            # [4] float
        item["strat_scalar_mask"] = w["strat_scalar_mask"]    # [4] bool
        return item
