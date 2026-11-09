"""PREREG_V8F_LADDER: the world-model-only window sampler and the frozen-adapter pooled path's data half.

A WM-only run needs no label, no T / S tick and no 6 s future: it reads ``F`` consecutive cached frames and the actions between them. ``wm_windows`` therefore lists EVERY anchor ``t`` in ``[F-1, T-1]`` of a
clip (``train_windows`` needs ``t <= T - 31`` for the 30 future knots and a 91-frame past-only window then fits no clip of 101 frames: ``t >= 90`` and ``t <= 70``). Every window is FULL (``F`` real frames, no
clamped or repeated frame); a clip with fewer than ``F`` frames has no window. Which steps a LEVEL can score inside the window is the level's own mask (``HJEPAStack.rollout_loss``).

``V8FMiniWMWindows`` returns ``act`` [F, 2] and either ``pool`` [F, d_feat] (the token mean of the parameter-free layer-normalised features, :func:`norm_mean_tokens`: the frozen adapter's pooled path, 1/n_tok of
the memory) or ``feats`` [F, n_tok, d_feat] (the reference path). ``collate_wm`` stacks them. Nothing here reads labels, the join, or a tick.
"""
from __future__ import annotations

import numpy as np

__all__ = ["norm_mean_tokens", "wm_windows", "V8FMiniWMWindows", "collate_wm", "WM_BATCH_KEYS_POOLED", "WM_BATCH_KEYS_FEATS"]

WM_BATCH_KEYS_POOLED = ("pool", "act", "t", "ci")
WM_BATCH_KEYS_ZCACHE = ("zmean", "act", "t", "ci")
WM_BATCH_KEYS_FEATS = ("feats", "act", "t", "ci")


def norm_mean_tokens(feats, eps: float = 1e-5):
    """``feats`` [..., n_tok, d] -> [..., d] float32: the mean over tokens of ``(f - mean_d f) / sqrt(var_d f + eps)`` -- ``torch.nn.LayerNorm`` WITHOUT its affine (biased variance, the module's default eps).
    The adapter's LayerNorm affine and its Linear are applied after the token mean by ``V8FMini.pooled_adapt_mean``."""
    import torch
    x = torch.as_tensor(feats).float()
    return torch.nn.functional.layer_norm(x, (x.shape[-1],), eps=eps).mean(-2)


def wm_windows(store, J: dict, clips, *, F: int, stride: int = 1) -> list:
    """All full ``F``-frame windows of ``clips``: ``(clip_index, t, 0, 0)`` for ``t = F-1, F-1+stride, .. <= T-1`` (``T`` = the join's frame count of the clip). No label requirement. ``store`` is unused (kept so the call
    reads like ``train_windows``)."""
    if F < 2 or stride < 1:
        raise ValueError("F >= 2 and stride >= 1")
    ci_of = {c: i for i, c in enumerate(J["clip"].tolist())}
    out = []
    for c in clips:
        ci = ci_of[c]
        T = int(J["n"][ci])
        for t in range(F - 1, T, stride):
            out.append((ci, t, 0, 0))
    return out


class V8FMiniWMWindows:
    """Dataset over ``wm_windows``. ``item(ci, t, ..)`` -> numpy dict. ``pooled`` True = ``pool`` (the token-mean path); False = ``feats``. The per-clip pooled table (all frames) is computed once per clip and kept in a
    small LRU, so the ~11 windows of a clip share it."""

    def __init__(self, store, J: dict, windows, *, F: int, pooled: bool = True, lru: int = 64, zcache: str = "", ego: bool = False):
        self.store, self.J, self.windows = store, J, list(windows)
        self.ego = bool(ego)                               # PREREG_V8F_EGOWM: also return ``v`` [F], the MEASURED speed (kin3 column 0, the ego state S / T / O read) of every window frame; False = the earlier batch, key for key
        self.F, self.pooled, self.lru = int(F), bool(pooled), int(lru)
        self.zcache = str(zcache or "")                    # a directory of per-clip ``<clip>.npz`` with ``zmean`` [T, D] (the frozen adapter's mean token), ``ok`` [T], ``t_s`` [T]; "" = not used
        self._nm: dict = {}
        self._z: dict = {}
        self._small: dict = {}                             # zcache runs: clip -> the action arrays and the frame count, held in RAM (see ``prewarm``)
        self.window_start = self.F - 1                     # the first t a window can have (read back by the gate)

    def __len__(self):
        return len(self.windows)

    def _table(self, clip: str, d: dict):
        if clip in self._nm:
            self._nm[clip] = self._nm.pop(clip)
            return self._nm[clip]
        tab = norm_mean_tokens(d["feats"]).numpy()        # [T, d_feat] float32
        self._nm[clip] = tab
        while len(self._nm) > self.lru:
            self._nm.pop(next(iter(self._nm)))
        return tab

    def zcache_missing(self, clips) -> list:
        """Clips of ``clips`` with no ``<clip>.npz`` in the cache directory (empty = complete)."""
        import os
        return [c for c in clips if not os.path.exists(os.path.join(self.zcache, f"{c}.npz"))]

    def _zmean(self, clip: str):
        if clip in self._z:
            self._z[clip] = self._z.pop(clip)
            return self._z[clip]
        import os
        with np.load(os.path.join(self.zcache, f"{clip}.npz")) as f:
            zm, ok = np.asarray(f["zmean"], np.float32), np.asarray(f["ok"], bool)
        self._z[clip] = (zm, ok)
        while len(self._z) > self.lru:
            self._z.pop(next(iter(self._z)))
        return self._z[clip]

    def ok_windows(self) -> list:
        """A zcache run keeps only the windows whose frames are all ``ok`` (the ``frame_ok`` of the M2b windows, which ``item`` refuses otherwise). Rewrites ``self.windows`` and returns it; the list is the same for every arm because ``ok`` depends on the kinematics, not on the model."""
        keep = []
        for w in self.windows:
            _, ok = self._zmean(str(self.J["clip"][w[0]]))
            lo, hi = int(w[1]) - self.F + 1, int(w[1])
            if lo >= 0 and hi < len(ok) and ok[lo:hi + 1].all():
                keep.append(w)
        self.windows = keep
        return keep

    def _kin_small(self, clip: str) -> dict:
        s = self._small.get(clip)
        if s is None:
            d = self.store.get(clip)
            s = self._small[clip] = {"a": d["a"], "ch1": d["ch1"], "n": len(d["pose"]), "v0": np.asarray(d["kin3"][:, 0], np.float32)}
        return s

    def prewarm(self) -> int:
        """A zcache run needs only each clip's action arrays from the clip file, so hold them in RAM: every 20 MB clip file (mostly image features, on a USB HDD on box 2) is read once, here at start-up, and not again. The windows are unchanged."""
        for ci in sorted({int(w[0]) for w in self.windows}):
            self._kin_small(str(self.J["clip"][ci]))
        return len(self._small)

    def item(self, ci: int, t: int, a_t: int = 0, a_s: int = 0) -> dict:
        clip = str(self.J["clip"][ci])
        d = self._kin_small(clip) if self.zcache else self.store.get(clip)
        n_frames = d["n"] if self.zcache else len(d["pose"])
        idx = np.arange(t - self.F + 1, t + 1)
        if idx[0] < 0 or idx[-1] >= n_frames:
            raise IndexError(f"{clip}: window t={t} F={self.F} leaves the clip ({n_frames} frames)")
        it = {"act": np.stack([np.nan_to_num(d["a"][idx]), d["ch1"][idx]], -1).astype(np.float32), "t": t, "ci": ci}
        if self.ego:                                       # PREREG_V8F_EGOWM: the measured speed of every frame of the window (row 0 of a clip is NaN: a window that holds it is refused, never zero-filled)
            v = np.asarray(d["v0"] if self.zcache else d["kin3"][:, 0], np.float32)[idx]
            if not np.isfinite(v).all():
                raise IndexError(f"{clip}: window t={t} F={self.F} holds a frame with no measured speed")
            it["v"] = v
        if self.zcache:                                    # the box-2 per-frame cache of the frozen adapter's mean token: no feature tensor is read for the adapter at all
            zm, ok = self._zmean(clip)
            if len(zm) <= idx[-1] or not ok[idx].all():
                raise IndexError(f"{clip}: zcache holds {len(zm)} frames / ok={bool(ok[idx].all()) if len(ok) > idx[-1] else None} for window t={t} F={self.F}")
            it["zmean"] = zm[idx]
        elif self.pooled:
            it["pool"] = self._table(clip, d)[idx]
        else:
            it["feats"] = d["feats"][idx]
        return it

    def __getitem__(self, i):
        ci, t, a_t, a_s = self.windows[i]
        return self.item(ci, t, a_t, a_s)


def collate_wm(items: list) -> dict:
    """numpy items -> a dict of torch tensors (``pool`` float32, ``feats`` float16, ``act`` float32, ``t`` / ``ci`` long)."""
    import torch
    out = {"act": torch.as_tensor(np.stack([it["act"] for it in items]), dtype=torch.float32)}
    if "zmean" in items[0]:
        out["zmean"] = torch.as_tensor(np.stack([it["zmean"] for it in items]), dtype=torch.float32)
    elif "pool" in items[0]:
        out["pool"] = torch.as_tensor(np.stack([it["pool"] for it in items]), dtype=torch.float32)
    else:
        out["feats"] = torch.as_tensor(np.stack([it["feats"] for it in items]), dtype=torch.float16)
    if "v" in items[0]:                                    # PREREG_V8F_EGOWM
        out["v"] = torch.as_tensor(np.stack([it["v"] for it in items]), dtype=torch.float32)
    for k in ("t", "ci"):
        out[k] = torch.as_tensor([it[k] for it in items], dtype=torch.long)
    return out
