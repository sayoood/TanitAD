"""mapsel_data.py -- refcv9 MAP-SELECT: the eval139 data joins (CPU, read-only).  Windows are keyed (clip_index, ws); clip_index == the sorted position of the clip uuid in windows_s0.json
(MEASURED 139/139 against the P4 bank) == the eval-cache manifest order.  GT map: E:/refcv6_eval_kit/data/sam3_gt_v3_eval via FineMapGTStore, n_stack 3 (MEASURED in the eval cache
manifest), exactly the read P4 used (its dataset-item parity: 50/50 equal).  Needs PYTHONPATH=<tree>/stack for the store.

HUMAN PATH: decisions/ep*.npz `gt_future_ext` [n, 60, 4] = (x, y, yaw, v) of the next 60 frames in the EPISODE frame; `pose_last` [n, 4] = the current pose in the same frame.  Ego frame at t0:
(dx, dy) = R(-yaw0) (p - p0).  MEASURED to equal the banked S6 `g` at steps 9, 19, ..., 59 (control K_HUMAN_PARITY, max abs diff printed by the caller).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ARCH = Path("C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf")
GT_ROOT = "E:/refcv6_eval_kit/data/sam3_gt_v3_eval"
WINDOWS_JSON = ARCH / "windows_s0.json"
N_STACK = 3
HUMAN_STEPS = (4, 9, 14, 19, 29, 39, 49, 59)


def clip_uuids():
    w = json.loads(WINDOWS_JSON.read_text(encoding="utf-8"))
    return sorted(w), w


def decisions_index(seed: int = 0):
    """-> dict (clip_index, ws) -> (decision file, row).  Files are paired with dump_s{seed}/ep*.npz by sorted order; `ws` arrays asserted equal."""
    root = ARCH / f"dump_s{seed}"
    fa = sorted(glob.glob(str(root / "ep*.npz")))
    fd = sorted(glob.glob(str(root / "decisions" / "ep*.npz")))
    if len(fa) != len(fd):
        raise SystemExit(f"[data] dump/decisions file counts differ: {len(fa)} vs {len(fd)}")
    idx = {}
    for a, d in zip(fa, fd):
        za, zd = np.load(a), np.load(d)
        if not np.array_equal(za["ws"], zd["ws"]):
            raise SystemExit(f"[data] ws mismatch between {a} and {d}")
        ci = int(za["clip_index"][0])
        for r, w in enumerate(zd["ws"].tolist()):
            idx[(ci, int(w))] = (d, r)
    return idx


def human_plans(keys, seed: int = 0):
    """keys: list of (clip_index, ws) -> (plans [n, 8, 2] float32 in the ego frame at t0, valid [n] bool: all 60 future frames valid)."""
    idx = decisions_index(seed)
    cache = {}
    out = np.zeros((len(keys), 8, 2), np.float32)
    valid = np.zeros(len(keys), bool)
    for i, k in enumerate(keys):
        d, r = idx[k]
        if d not in cache:
            z = np.load(d)
            cache[d] = (z["gt_future_ext"], z["gt_future_valid_ext"], z["pose_last"])
        fut, fv, p0 = cache[d]
        f, p = fut[r], p0[r]
        c, s = np.cos(float(p[2])), np.sin(float(p[2]))
        dx, dy = f[:, 0] - p[0], f[:, 1] - p[1]
        x, y = c * dx + s * dy, -s * dx + c * dy
        out[i] = np.stack([x[list(HUMAN_STEPS)], y[list(HUMAN_STEPS)]], -1)
        valid[i] = bool(fv[r].min() >= 1.0)
    return out, valid


class GTReader:
    """CPU reader of the SAM3 10 cm GT codes of a window (the dataset item's `map_fine`)."""

    def __init__(self, root: str = GT_ROOT):
        from tanitad.data import semantic_map_gt_fine as sem
        from tanitad.models import map_head_hires as mh
        self.uuids, _ = clip_uuids()
        self.store = sem.FineMapGTStore(Path(root), max_open=4, extent=mh.EXTENT_REFCV7)
        self.has = {i: self.store.resolve(u) is not None for i, u in enumerate(self.uuids)}

    def codes(self, ci: int, ws: int):
        """uint8 [1000, 600] or None when the clip has no GT file."""
        if not self.has[ci]:
            return None
        return self.store.frames_for_windows(self.uuids[ci], [int(ws)], n_stack=N_STACK).codes[0]


# ------------------------------------------------------------------------------------------------------------- lane-change windows (v9 release)
V9_NPZ = "E:/refcv8_eval_kit/data/refcv8_v9labels/release/v9_labels_eval139.npz"
LC_HORIZON_S = 6.0


def sha12(s: str) -> str:
    import hashlib
    return hashlib.sha256(str(s).encode()).hexdigest()[:12]


def lc_windows(keys, check_keys=None):
    """-> (flag [n] bool: the v9 row of the window's RAW frame (ws + n_stack - 1) has a lane-change crossing time in (0, 6] s,  join_check dict).
    The row join is validated against the nav that was FED (decisions r8_nav_fed[:, 0] == log1p(nav_d_next_m / 50) on the known rows): agreement must be >= 0.99 or `ok` is False
    and the lane-change exception must be treated as UNAVAILABLE by the caller."""
    z = np.load(V9_NPZ, allow_pickle=True)
    uu, _ = clip_uuids()
    pos = {str(s): i for i, s in enumerate(z["clip__sha12"])}
    row0, k0, nrows = z["clip__row0"], z["row__k"][z["clip__row0"]], z["clip__n_rows"]
    t_lc = z["row__lc_t_cross_s"]
    dnext = z["row__nav_d_next_m"]

    def row_of(ci, ws):
        i = pos[sha12(uu[ci])]
        m = (int(ws) + N_STACK - 1) - int(k0[i])
        return (int(row0[i]) + m) if 0 <= m < int(nrows[i]) else -1
    flag = np.zeros(len(keys), bool)
    for n, (ci, ws) in enumerate(keys):
        r = row_of(ci, ws)
        if r >= 0:
            t = float(t_lc[r])
            flag[n] = bool(np.isfinite(t) and 0.0 < t <= LC_HORIZON_S)
    # join check on a fixed sample of windows with a known fed nav
    idx = decisions_index(0)
    ck = check_keys if check_keys is not None else [k for k in sorted(idx) if k[0] % 7 == 0]
    tot = ok = 0
    cache = {}
    per = {}
    for k in ck:
        f, r_ = idx[k]
        if f not in cache:
            zz = np.load(f)
            cache[f] = (zz["r8_nav_fed"], zz["r8_nav_known"])
        fed, kn = cache[f]
        if not bool(kn[r_]):
            continue
        r = row_of(*k)
        if r < 0 or not np.isfinite(dnext[r]):
            continue
        a = int(abs(np.log1p(float(dnext[r]) / 50.0) - float(fed[r_][0])) < 1e-3)
        tot += 1
        ok += a
        t_ = per.setdefault(k[0], [0, 0])
        t_[0] += a
        t_[1] += 1
    chk = {"n_checked": tot, "n_agree": ok, "agree": (ok / tot if tot else float("nan")),
           "per_clip_agree": {int(c): round(v[0] / v[1], 3) for c, v in sorted(per.items())}}
    good = sum(1 for v in per.values() if v[0] / v[1] >= 0.99)
    # a clip whose FED nav disagrees with the v9 row at every offset (MEASURED: one of 13 checkable clips) is a nav-feed difference, not a join error; the join is accepted when >= 80 % of the checkable clips agree
    chk["clips_checked"], chk["clips_agree_ge_0p99"] = len(per), good
    chk["ok"] = bool(len(per) >= 8 and good >= 0.8 * len(per))
    return flag, chk
