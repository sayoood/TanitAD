"""Stage-2 loader for REF-A v1 — the piece that was deliberately missing.

WHAT "THE DATALOADER GAP" MEANT (PI question 2026-09-01): `refa_v1_train.py`
could only train on `SmokeData` (random tensors); the module binding the
stage-1 DINOv3 cache to real windows did not exist, BY DESIGN — "the trainer
must not invent a loader for a cache that does not exist yet". The cache
design is now settled (parity corpus, 0.2 s grid, 295.9 GiB), so this is it.

INPUTS
  cache_dir    <episode_id>.pt   fp16 [T_c, n_tokens, d_enc] on the 0.2 s grid
  episode_dir  <episode_id>.v2ep.pt — actions/poses at 10 Hz (frames NOT read)

GRID: cache index j <-> v2ep frame 2j (0.2 s = every 2nd frame at 10 Hz).
⛔ An episode whose cache length disagrees with ceil(T_ep/2) is REFUSED — a
mis-gridded cache is a silent time-warp, not a smaller dataset.

ACTIONS — (a, kappa), and the channel order is MEASURED, not assumed:
  v2ep `actions[:, 0]` correlates r = 0.995 with pose-derived curvature and
  `actions[:, 1]` only r = 0.47 with pose-derived accel (6 episodes,
  2026-09-01) ⇒ the STORED order is (kappa, accel-like) — the REVERSE of
  RefAV1Config's `a_dim: (a, kappa)`. This loader emits
      a      = (v[2(j+1)] - v[2j]) / 0.2      # poses ch3, exact on the grid
      kappa  = actions[2j, 0]                  # the measured true-kappa channel
  so a silent channel swap cannot reach the model.

LABELS/NAV: exposed as None for now — the v7.2 s2 join keys on clip_id and is
the next increment; the model-side hooks (lat/lon/route CE, nav_cmd) exist.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
from torch import Tensor

__all__ = ["RefAV1Windows", "split_episodes"]


def split_episodes(names: list[str], val_frac: float = 0.15,
                   seed: int = 0) -> tuple[list[str], list[str]]:
    """Episode-disjoint split, deterministic in (names, seed)."""
    order = sorted(names)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(order), generator=g).tolist()
    n_val = max(1, int(round(len(order) * val_frac)))
    val = sorted(order[i] for i in perm[:n_val])
    train = sorted(set(order) - set(val))
    return train, val


class RefAV1Windows:
    """Deterministic window sampler over the stage-1 cache + v2ep kinematics."""

    def __init__(self, cache_dir: str | Path, episode_dir: str | Path, *,
                 op_window: int, op_steps: int, op_dt: float = 0.2,
                 str_dt: float = 3.0, str_ext_steps: int = 2,
                 episodes: list[str] | None = None,
                 lru: int = 32, seed: int = 0):
        self.cache_dir = Path(cache_dir)
        self.episode_dir = Path(episode_dir)
        self.W, self.K, self.dt = int(op_window), int(op_steps), float(op_dt)
        self.str_dt, self.k_ext = float(str_dt), int(str_ext_steps)
        if abs(self.dt - 0.2) > 1e-9:
            raise ValueError(f"op_dt {op_dt}: this loader's grid mapping "
                             "(cache j <-> frame 2j) is derived for 0.2 s")
        # ext tick k (1-based) closes at 6.0 + k*str_dt after the window start
        self.ext_close = [int(round((6.0 + k * self.str_dt) / self.dt))
                          for k in range(1, self.k_ext + 1)]
        self.ext_open = [int(round((6.0 + (k - 1) * self.str_dt) / self.dt))
                         for k in range(1, self.k_ext + 1)]
        self.reach = self.ext_close[-1] if self.ext_close else self.K

        names = episodes if episodes is not None else sorted(
            p.stem for p in self.cache_dir.glob("*.pt") if p.stem != "index")
        if not names:
            raise FileNotFoundError(f"no cached episodes under {self.cache_dir}")
        self.names: list[str] = []
        self.windows: list[tuple[int, int]] = []      # (episode idx, t)
        self._T: dict[str, int] = {}
        for nm in names:
            ep = self.episode_dir / f"{nm}.v2ep.pt"
            if not ep.exists():
                raise FileNotFoundError(f"cache episode {nm} has no v2ep at {ep}")
            t_c = self._cache_len(nm)
            t_ep = self._ep_len(nm)
            want = math.ceil(t_ep / 2)
            if t_c != want:
                raise ValueError(
                    f"{nm}: cache length {t_c} != ceil(T_ep/2) = {want} — a "
                    "mis-gridded cache is a silent time-warp, refused")
            ei = len(self.names)
            self.names.append(nm)
            self._T[nm] = t_c
            # t is the LAST observed cache index; +1 below it because the
            # kinematic action at the final future step reads v at j+1.
            for t in range(self.W - 1, t_c - self.reach - 1):
                self.windows.append((ei, t))
        if not self.windows:
            raise ValueError(
                f"0 windows: episodes too short for W={self.W} + reach "
                f"{self.reach} (need T_c > {self.W - 1 + self.reach + 1})")
        g = torch.Generator().manual_seed(seed)
        self._order = torch.randperm(len(self.windows), generator=g).tolist()
        self._cursor = 0
        self._lru_n = int(lru)
        self._lru: dict[str, tuple[Tensor, Tensor, Tensor]] = {}

    # ------------------------------------------------------------- internals
    def _cache_len(self, nm: str) -> int:
        return torch.load(self.cache_dir / f"{nm}.pt", map_location="cpu",
                          weights_only=True, mmap=True).shape[0]

    def _ep_len(self, nm: str) -> int:
        o = torch.load(self.episode_dir / f"{nm}.v2ep.pt",
                       map_location="cpu", weights_only=False)
        return int(o["poses"].shape[0])

    def _episode(self, nm: str):
        """(features fp16 [T_c,N,d], v [T_ep], kappa [T_ep]) — LRU-cached."""
        hit = self._lru.pop(nm, None)
        if hit is None:
            feats = torch.load(self.cache_dir / f"{nm}.pt",
                               map_location="cpu", weights_only=True)
            o = torch.load(self.episode_dir / f"{nm}.v2ep.pt",
                           map_location="cpu", weights_only=False)
            hit = (feats, o["poses"][:, 3].float(), o["actions"][:, 0].float())
        self._lru[nm] = hit                          # re-insert as most recent
        while len(self._lru) > self._lru_n:
            self._lru.pop(next(iter(self._lru)))
        return hit

    def _kin_actions(self, v: Tensor, kap: Tensor, j0: int, n: int) -> Tensor:
        """(a, kappa) for cache steps j0 .. j0+n-1 (each spans 0.2 s)."""
        idx = torch.arange(j0, j0 + n)
        f = idx * 2                                   # 10 Hz frame of step open
        f_next = torch.clamp((idx + 1) * 2, max=v.shape[0] - 1)
        a = (v[f_next] - v[f]) / self.dt
        return torch.stack([a, kap[f]], dim=-1)       # ⭐ (a, kappa) — swapped

    # ------------------------------------------------------------------ api
    def __len__(self) -> int:
        return len(self.windows)

    def batch(self, bs: int) -> dict:
        feats, fut, act, ext_t, ext_a = [], [], [], [], []
        for _ in range(bs):
            ei, t = self.windows[self._order[self._cursor]]
            self._cursor = (self._cursor + 1) % len(self._order)
            nm = self.names[ei]
            F, v, kap = self._episode(nm)
            feats.append(F[t - self.W + 1:t + 1].float())
            fut.append(F[t + 1:t + 1 + self.K].float())
            act.append(self._kin_actions(v, kap, t, self.K))
            if self.k_ext:
                ext_t.append(torch.stack([F[t + c] for c in self.ext_close]
                                         ).float())
                ext_a.append(torch.stack(
                    [self._kin_actions(v, kap, t + o, 1)[0]
                     for o in self.ext_open]))
        out = {"feats": torch.stack(feats), "future_feats": torch.stack(fut),
               "actions": torch.stack(act),
               # the v7.2 label join and the nav join key on clip_id and are
               # the NEXT increment; the model-side hooks already exist.
               "lat_label": None, "lon_label": None, "route_label": None,
               "nav_cmd": None}
        if self.k_ext:
            out["str_ext_targets"] = torch.stack(ext_t)
            out["str_ext_actions"] = torch.stack(ext_a)
        return out
