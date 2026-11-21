"""l7cap_hooks.py -- the pure pieces of the L7 train-split fan capture (no roll, no checkpoint, no GPU).

Everything here OBSERVES. Nothing here feeds a tensor back into the forward, draws a random number, or changes a returned
value: the identity control (a) in `controls_l7cap.py` and `tests/test_l7cap_hooks.py` pin that.

  install_scene_capture(RV3, torch)   patches `RefCV3Model._scene_hook` so the closure it returns ALSO stores DETACHED copies of the four
                                      scene sources (`agent_tokens, agent_pad, bev_tokens, bev_pad`) in the forward's `cache` dict under
                                      `r7_agent_tokens / r7_agent_pad / r7_bev_tokens / r7_bev_pad`. `refc_v3.py:1958` stores them only
                                      when `self.refcv7_wta is not None`, which is False for refcv8, so without this the sources are
                                      invisible. `cache` is merged into the returned `out` at the end of `forward` (refc_v3.py `out.update(cache)`).
                                      The `cond` tensor (nav + fed ceiling + v0 + a0) is DELIBERATELY NOT stored: the scorer may not read it
                                      (DESIGN section 4), so the bank cannot carry it.
  crop_drivable(codes, extent_outer)  SAM3 10 cm codes -> the oracle's 0.5 m drivable fraction + seen mask in the /2 geometry (120 x 64).
  FORBIDDEN_BANK_KEYS                 keys that must never be written into the INPUT bank (`fan_extras.npz`, `l7cap_scene.npz`).
  psapi peak helper                   `arm_peak("tag")` prints the process's PEAK working set at exit.
"""
from __future__ import annotations

import os

import numpy as np

SCENE_KEYS = ("r7_agent_tokens", "r7_agent_pad", "r7_bev_tokens", "r7_bev_pad")
#: never in the INPUT bank: they carry nav / the fed ceiling / labels (DESIGN section 4)
FORBIDDEN_BANK_KEYS = frozenset({
    "cond", "r7_cond", "z_tac", "v_max_ms", "v_max_valid", "gt_max_speed_6s", "g", "gt", "gt_valid", "gt_future",
    "r7_agent_fut", "r7_agent_fut_valid", "r7_agent_fut_label", "nav_cmd", "r8_nav", "r8_rc", "tac_goal_logits",
    "tacv6_goal_logits", "g_tac", "g_str", "map_fine", "map_drivable_u8", "map_notseen_u8", "map_seen"})

#: test-only mutation switch for the deliberate-regression arms; the production path never sets it
MUTATION = os.environ.get("L7CAP_MUTATION", "")


def install_scene_capture(RV3, torch, mutation: str | None = None):
    """Patch `RV3.RefCV3Model._scene_hook`. Returns the original (so a test can restore it). Idempotent."""
    mut = MUTATION if mutation is None else mutation
    cls = RV3.RefCV3Model
    if getattr(cls._scene_hook, "_l7cap_patched", False):
        return cls._scene_hook._l7cap_orig
    orig = cls._scene_hook

    def _scene_hook(self, cache, *a, **k):
        hook = orig(self, cache, *a, **k)
        if hook is None:
            return None

        def hook2(*args, **kw):
            res = hook(*args, **kw)
            names = ("agent_tokens", "agent_pad", "bev_tokens", "bev_pad")
            vals = dict(zip(names, args))
            vals.update({n: kw[n] for n in names if n in kw})
            with torch.no_grad():
                for n, key in zip(names, SCENE_KEYS):
                    v = vals.get(n)
                    if mut == "swap_sources":                       # deliberate regression: agent <-> bev
                        v = vals.get({"agent_tokens": "bev_tokens", "bev_tokens": "agent_tokens",
                                      "agent_pad": "bev_pad", "bev_pad": "agent_pad"}[n])
                    if torch.is_tensor(v) and key not in cache:
                        cache[key] = v.detach().clone()
            return res
        return hook2

    _scene_hook._l7cap_patched = True
    _scene_hook._l7cap_orig = orig
    cls._scene_hook = _scene_hook
    return orig


def uninstall_scene_capture(RV3):
    cls = RV3.RefCV3Model
    if getattr(cls._scene_hook, "_l7cap_patched", False):
        cls._scene_hook = cls._scene_hook._l7cap_orig


def to_fp16_checked(x: np.ndarray) -> tuple[np.ndarray, dict]:
    """float -> float16 for the bank, REFUSING silent overflow: if any finite value exceeds fp16's range the array stays float32."""
    x = np.asarray(x)
    if x.dtype == np.bool_ or not np.issubdtype(x.dtype, np.floating):
        return x, {"stored": str(x.dtype)}
    fin = np.isfinite(x)
    mx = float(np.abs(x[fin]).max()) if fin.any() else 0.0
    if mx < 6.0e4:
        return x.astype(np.float16), {"stored": "float16", "max_abs": mx, "n_nonfinite": int((~fin).sum())}
    return x.astype(np.float32), {"stored": "float32", "max_abs": mx, "reason": "max_abs >= 6e4 would overflow fp16"}


# ------------------------------------------------------------------------------------------------ SAM3 drivable crop
def crop_drivable(codes: np.ndarray, extent_outer, extent_inner=None) -> tuple[np.ndarray, np.ndarray]:
    """10 cm SAM3 codes of the model's map extent -> (drivable [N,120,64] uint8 fraction 0..255, seen [N,120,64] bool) in the ORACLE's
    geometry (`refcv7_oracle.py:30` : 0.5 m cells, 120 x 64, x in [0, 60], y in [-16, +16], column 0 = RIGHT).

    The crop and the block pooling are the stack's own: `EXTENT_V2.window_in(outer)` and `block_fraction_u8` (semantic_map_gt_fine.py,
    whose own control `compare_v2_window` proves the /3 window is byte-identical to the /2 arrays). `seen` follows
    `semantic_map_gt.py:331`: `2 * (255 - not_seen_fraction) >= 255`. Channel 1 = "drivable" (CHANNELS[1])."""
    from tanitad.data import semantic_map_gt_fine as F
    inner = extent_inner or F.EXTENT_V2
    c = np.asarray(codes)
    if c.ndim == 2:
        c = c[None]
    r0, r1, c0, c1 = inner.window_in(extent_outer)
    cart = F.block_fraction_u8(c[:, r0:r1, c0:c1])                  # [N, 9, 120, 64] uint8
    drv = cart[:, 1]
    ns = cart[:, F.NOT_SEEN_CHANNEL].astype(np.int32)
    seen = (2 * (255 - ns)) >= 255
    return drv, seen


# ------------------------------------------------------------------------------------------------ peak working set
def peak_ws_gb() -> tuple[float, float] | None:
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC()
        c.cb = ctypes.sizeof(PMC)
        k32, ps = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        if ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize / 2 ** 30, c.PeakPagefileUsage / 2 ** 30
    except Exception:       # noqa: BLE001
        pass
    return None
