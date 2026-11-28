"""v5_lib.py -- data layer of chapter 8 (DAgger: trained on its own states) of the v8F-mini evidence video v5.  CPU only, read-only on every input except its own OUT.

* the four float64 T1-kin dump sets of the decisive runs (DAG / DAG-0 x seed 0 / 1, `--limit causal --dump-ticks --dump-heads`, plain) -> per-window GT g and the executed track cl
* the per-tick files of the same runs (S's checkpoints in the window frame, T's two heads' logits)
* data/dag_cells_<ARM>_s<seed>.npz: the label cells (route-matched label class of T's lateral / longitudinal head, S's GT route checkpoints) produced by `extract_dag_cells.py`
  with the programme's OWN reader `t1hd_read.py` (control F1: reproduces the DAG_READ.json F1 numbers to 1e-9)
* data/DAG_READ.json (sha256 3393fdd5...): every card number is a pointer into it, taken by `NJ` and registered for CONTROLS_V5
* data/selection_v5.json: the seven scenes, written by select_scenes5.py BEFORE any chapter-8 frame was rendered
* the deliberate-regression hook: env V5_SWAP_ARMS=<scene tag> makes the data-access layer return DAG-0's arrays when DAG is asked for (and vice versa) for that ONE scene
  (the control must then FAIL).
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

import v3_lib as L
from v3_lib import OUT, CUR, LAT5, LON7  # noqa: F401

ART = "E:/Projects/TanitAD-artifacts/v8f-dagger-t1"
ARMS = ("DAG", "DAG0")
ARM_LAB = {"DAG": "DAG", "DAG0": "DAG-0"}
SEEDS5 = (0, 1)
DT = 0.2
TICK_IDX = (9, 19)                         # the 2 s and 4 s replan ticks: knot i is at t = 0.2 (i + 1) s
DUMPS = {(a, s): f"{ART}/dump_{a}_s{s}_m10_causal_heads" for a in ARMS for s in SEEDS5}
TICKS = {(a, s): f"{ART}/ticks/ticks_{a}_s{s}_m10_causal_heads.npz" for a in ARMS for s in SEEDS5}
CELLS = {(a, s): f"{OUT}/data/dag_cells_{a}_s{s}.npz" for a in ARMS for s in SEEDS5}
SWAP_TAG = os.environ.get("V5_SWAP_ARMS", "")
OTHER = {"DAG": "DAG0", "DAG0": "DAG"}
MINUS = "\u2212"

_SEL = json.load(open(f"{OUT}/data/selection_v5.json", encoding="utf-8"))
SCN = _SEL["scenes"]                       # 7 dicts: tag, source, stratum, grid_row, clip, clip_index, ws, gt_mean_speed, delta_s0_m
DR = json.load(open(f"{OUT}/data/DAG_READ.json", encoding="utf-8"))
ARGV = {"DAG": json.load(open(f"{OUT}/data/argv_M2b_s0_dag.json", encoding="utf-8"))["argv"], "DAG0": json.load(open(f"{OUT}/data/argv_M2b_s0_dag0.json", encoding="utf-8"))["argv"]}
_D: dict = {}
_T: dict = {}
_C: dict = {}


def mm(x, spec="{:.2f}"):
    return spec.format(x).replace("-", MINUS)


# ------------------------------------------------------------------------------------------------ raw arrays
def load():
    """{(arm, seed): dict(g, cl, ws, clip_index)} in the dump order; asserts that the four sets share windows and GT."""
    if _D:
        return _D
    for key, d in DUMPS.items():
        fs = sorted(glob.glob(f"{d}/ep*.npz"))
        assert len(fs) == 141, (key, len(fs))
        acc = {k: [] for k in ("g", "cl", "ws", "clip_index")}
        for f in fs:
            with np.load(f) as z:
                for k in acc:
                    acc[k].append(z[k])
        _D[key] = {k: np.concatenate(v) for k, v in acc.items()}
    ref = _D[("DAG", 0)]
    for key, d in _D.items():
        assert len(d["ws"]) == 2399 and (d["ws"] == ref["ws"]).all() and (d["clip_index"] == ref["clip_index"]).all() and np.array_equal(d["g"], ref["g"]), key
    return _D


def ticks(arm, seed):
    """per-window tick arrays [N, 3, ...] of one run (row index = (window, tick))."""
    if (arm, seed) not in _T:
        z = np.load(TICKS[(arm, seed)], allow_pickle=False)
        win, tick = z["win"], z["tick"]
        row = np.full((2399, 3), -1, np.int64)
        row[win, tick] = np.arange(len(win))
        assert (row >= 0).all()
        _T[(arm, seed)] = {"s_ckpt_w0": z["s_ckpt_w0"][row].astype(np.float64), "lat_logits": z["t_lat_logits"][row].astype(np.float64), "lon_logits": z["t_lon_logits"][row].astype(np.float64),
                           "pose": z["pose"][row], "j": z["j"][row], "t0": z["t0"][row]}
    return _T[(arm, seed)]


def cells(arm, seed):
    if (arm, seed) not in _C:
        with np.load(CELLS[(arm, seed)]) as z:
            _C[(arm, seed)] = {k: z[k] for k in z.files}
    return _C[(arm, seed)]


def scene(i):
    return SCN[i]


def _a(arm, i):
    return OTHER[arm] if (SWAP_TAG and SWAP_TAG == SCN[i]["tag"]) else arm


# ------------------------------------------------------------------------------------------------ per-scene accessors (the regression hook lives HERE and nowhere else)
def gt_track(i):
    return load()[("DAG", 0)]["g"][SCN[i]["grid_row"]]


def o_track(arm, seed, i):
    return load()[(_a(arm, i), seed)]["cl"][SCN[i]["grid_row"]]


def s_set(arm, seed, i, k):
    """S's three checkpoints (x forward, y left, psi) of tick k in the window frame, and the human's route checkpoints (ok-masked)."""
    r = SCN[i]["grid_row"]
    a = _a(arm, i)
    c = cells(a, seed)
    return ticks(a, seed)["s_ckpt_w0"][r, k], c["gt_w0"][r, k], c["gt_ok"][r, k]


def tcell(arm, seed, i, k, head):
    """(pred class, true class or -1) of T's `head` ('lat' / 'lon') at tick k."""
    r = SCN[i]["grid_row"]
    c = cells(_a(arm, i), seed)
    return int(c[f"{head}_pred"][r, k]), int(c[f"{head}_true"][r, k])


def s_err(arm, seed, i, k):
    r = SCN[i]["grid_row"]
    return cells(_a(arm, i), seed)["s_pos_err"][r, k]


def ade(pred, g):
    return np.linalg.norm(np.asarray(pred, np.float64) - np.asarray(g, np.float64), axis=-1).mean(-1)


def ade6(arm, seed, i):
    return float(ade(o_track(arm, seed, i), gt_track(i)))


def per_window_delta(seed):
    d = load()
    return ade(d[("DAG", seed)]["cl"], d[("DAG", seed)]["g"]) - ade(d[("DAG0", seed)]["cl"], d[("DAG0", seed)]["g"])


def frame_for(i, f):
    """the LOGGED camera frame (both arms read the same) at +0.2 f s of the window."""
    s = SCN[i]
    return L.v2ep_frame(s["clip"], int(s["ws"] + f))


# ------------------------------------------------------------------------------------------------ registered numbers

def plen(track):
    """length (m) of a 6 s path: the sum of the 30 segments, starting at the origin."""
    p = np.vstack([np.zeros((1, 2)), np.asarray(track, np.float64)[:, :2]])
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def plen_reg(who, seed, i):
    """path length of the human ('GT') or of an arm's executed track, 1 decimal; registered (kind 'plen')."""
    s = "{:.1f}".format(plen(gt_track(i) if who == "GT" else o_track(who, seed, i)))
    L.REG.append({"kind": "plen", "scene": CUR["scene"], "who": who, "seed": seed, "tag": SCN[i]["tag"], "grid_row": SCN[i]["grid_row"], "shown": s})
    return s

def ade_reg(arm, seed, i, spec="{:.2f}"):
    s = spec.format(ade6(arm, seed, i))
    L.REG.append({"kind": "ade6", "scene": CUR["scene"], "arm": arm, "seed": seed, "tag": SCN[i]["tag"], "grid_row": SCN[i]["grid_row"], "spec": spec, "shown": s})
    return s


def dlt_reg(seed, i, spec="{:+.2f}"):
    s = mm(ade6("DAG", seed, i) - ade6("DAG0", seed, i), spec)
    L.REG.append({"kind": "ade6d", "scene": CUR["scene"], "seed": seed, "tag": SCN[i]["tag"], "grid_row": SCN[i]["grid_row"], "spec": spec, "shown": s})
    return s


def tinfo(arm, seed, i, k, head):
    """(pred name, true name or None, status 'match' / 'MISS' / 'no label'); registered (kind 'tcell')."""
    names = LAT5 if head == "lat" else LON7
    p, t = tcell(arm, seed, i, k, head)
    st = "no label" if t < 0 else ("match" if p == t else "MISS")
    L.REG.append({"kind": "tcell", "scene": CUR["scene"], "arm": arm, "seed": seed, "tag": SCN[i]["tag"], "grid_row": SCN[i]["grid_row"], "tick": k, "head": head,
                  "pred": names[p], "true": names[t] if t >= 0 else None, "status": st, "shown": f"{names[p]}|{names[t] if t >= 0 else '-'}|{st}"})
    return names[p], (names[t] if t >= 0 else None), st


def serr_reg(arm, seed, i, k):
    """S's position error at 30 / 50 / 80 m of tick k: three strings ('-' where the label is unscored); registered (kind 'serr')."""
    e = s_err(arm, seed, i, k)
    out = ["-" if not np.isfinite(x) else "{:.1f}".format(x) for x in e]
    L.REG.append({"kind": "serr", "scene": CUR["scene"], "arm": arm, "seed": seed, "tag": SCN[i]["tag"], "grid_row": SCN[i]["grid_row"], "tick": k, "shown": "/".join(out)})
    return out


def tcount_reg(arm, seed, i, head):
    """(correct, scored) over ticks 1 and 2 (the model's OWN states) of T's `head`; registered (kind 'tcount')."""
    ok = n = 0
    for k in (1, 2):
        p, t = tcell(arm, seed, i, k, head)
        if t >= 0:
            n += 1
            ok += int(p == t)
    s = f"{ok}/{n}"
    L.REG.append({"kind": "tcount", "scene": CUR["scene"], "arm": arm, "seed": seed, "tag": SCN[i]["tag"], "grid_row": SCN[i]["grid_row"], "head": head, "shown": s})
    return s


def NJ(ptr, spec="{:+.4f}", scale=1.0):
    """formatted value of DAG_READ.json at `ptr` ('rows/s0/R-TLAT/delta'), typographic minus; registered (kind 'dagj'): the control re-reads the file in another process."""
    v = L.getp(DR, ptr)
    v = v * scale if scale != 1.0 else v
    s = spec.format(v).replace("-", MINUS)
    L.REG.append({"kind": "dagj", "scene": CUR["scene"], "ptr": ptr, "spec": spec, "scale": scale, "shown": s})
    return s


def BJ(ptr):
    """boolean / string flag of DAG_READ.json, registered (kind 'dagb') so the control can re-read it."""
    v = L.getp(DR, ptr)
    L.REG.append({"kind": "dagb", "scene": CUR["scene"], "ptr": ptr, "shown": str(v)})
    return v


def TJ(ptr, upto=None):
    """a verbatim string of DAG_READ.json shown on screen (kind 'dagt'); `upto` keeps the part before its first occurrence."""
    v = L.getp(DR, ptr)
    assert isinstance(v, str), ptr
    if upto:
        v = v.split(upto)[0]
    L.REG.append({"kind": "dagt", "scene": CUR["scene"], "ptr": ptr, "upto": upto, "shown": v})
    return v


def AV(arm, flag, spec=None):
    """value of a training flag in argv.json of the seed-0 train dir of `arm` (kind 'argv'); with `spec` it is formatted as a float ('{:,.0f}' for the step count)."""
    a = ARGV[arm]
    v = a[a.index(flag) + 1]
    shown = spec.format(float(v)) if spec else v
    L.REG.append({"kind": "argv", "scene": CUR["scene"], "arm": arm, "flag": flag, "spec": spec, "shown": shown})
    return shown


def sep_word(ptr_prefix):
    """'better' / 'WORSE' / 'n.s.' of a DAG_READ S-POS cell: the JSON's own `separated` flag and the sign of delta (lower position error = better); registered."""
    sepr, d = L.getp(DR, ptr_prefix + "/separated"), L.getp(DR, ptr_prefix + "/delta")
    w = "n.s." if not sepr else ("better" if d < 0 else "WORSE")
    L.REG.append({"kind": "dagw", "scene": CUR["scene"], "ptr": ptr_prefix, "shown": w})
    return w


# ------------------------------------------------------------------------------------------------ F1 (own implementation; the control compares it with DAG_READ.json)
def f1_stats(true, pred, n):
    """macro-F1 over the classes PRESENT among the labelled cells (taniteval.v8f_consistency.macro_f1), per-class F1 and support."""
    true, pred = np.asarray(true).ravel(), np.asarray(pred).ravel()
    m = (true >= 0) & (true < n)
    yt, yp = true[m], pred[m]
    cm = np.zeros((n, n), np.int64)
    np.add.at(cm, (yt, yp), 1)
    f = np.full(n, np.nan)
    for c in range(n):
        tp = cm[c, c]
        fp, fn = cm[:, c].sum() - tp, cm[c, :].sum() - tp
        if cm[c].sum() > 0:
            f[c] = 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    return float(np.nanmean(f)), f, cm.sum(1), int(m.sum())


def class_f1(arm, seed, head):
    c = cells(arm, seed)
    n = 5 if head == "lat" else 7
    return f1_stats(c[f"{head}_true"][:, 1:], c[f"{head}_pred"][:, 1:], n)
