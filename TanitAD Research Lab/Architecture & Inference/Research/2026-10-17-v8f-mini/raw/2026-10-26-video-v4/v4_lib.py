"""v4_lib.py -- data layer of chapter 7 (T1-kin) of the v8F-mini evidence video v4.  CPU only, read-only on every input except its own OUT.

* the four float64 T1-kin dump sets (HACC causal / strict x seed 0 / 1) -> per-window arrays in the dump order (clip_index, ws), the SAME order for all four
* the seven pre-declared scenes (data/selection_v4.json, written by select_scenes.py BEFORE any executed track was read)
* the four RESULT files (data/results/*.md, byte-identical to commit abc12d3f; git blob shas in VIDEO_NOTES_V4.md): every number a card shows is a verbatim
  cell of one of them, taken by `RC` / `RT` and registered for CONTROLS_V4 (control: the shown string is in the file, in the right row).
* the deliberate-regression hook: env V4_SWAP_CLFL=<scene tag> swaps the model's and the floor's executed tracks of that ONE scene at the data-access layer
  (the control must then FAIL).
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

import v3_lib as L
from v3_lib import SC, OUT, CUR  # noqa: F401

DUMPS = {("causal", 0): "C:/Users/Admin/t1kc_run/dump_HACC_s0_m10_causal", ("causal", 1): "C:/Users/Admin/t1kc_run/dump_HACC_s1_m10_causal",
         ("strict", 0): "C:/Users/Admin/t1ks_run/dump_HACC_s0_m10_strict", ("strict", 1): "C:/Users/Admin/t1ks_run/dump_HACC_s1_m10_strict"}
JOIN = "C:/Users/Admin/t1k/TanitAD Research Lab/Architecture & Inference/Research/2026-10-17-v8f-mini/raw/join_eval141.npz"
RDIR = f"{OUT}/data/results"
RFILES = ("RESULT_V8F_T1KIN_CAUSALBUCKET", "RESULT_V8F_T1KIN_STRICT", "RESULT_V8F_T1_LATCOND", "RESULT_V8F_HACC_LAT")
DT = 0.2
SWAP_TAG = os.environ.get("V4_SWAP_CLFL", "")
TICK_IDX = (9, 19)                        # the 2 s and 4 s replan ticks: knot i is at t = 0.2 (i + 1) s
SEEDS4 = (0, 1)

_SEL = json.load(open(f"{OUT}/data/selection_v4.json"))
SCN = _SEL["scenes"]                      # 7 dicts: tag, source, grid_row, clip, clip_index, ws, gt_mean_speed, (v3_window)
_D: dict = {}
_JOIN: dict = {}


def join():
    if not _JOIN:
        with np.load(JOIN, allow_pickle=False) as z:
            _JOIN.update({k: z[k] for k in ("clip", "off", "n")})
    return _JOIN


def load():
    """{(run, seed): dict(g, cl, fl, ol, ws, clip_index)} concatenated in the dump order; asserts the four sets share windows, GT and the floor."""
    if _D:
        return _D
    for key, d in DUMPS.items():
        fs = sorted(glob.glob(f"{d}/ep*.npz"))
        assert len(fs) == 141, (key, len(fs))
        acc = {k: [] for k in ("g", "cl", "fl", "ol", "ws", "clip_index")}
        for f in fs:
            with np.load(f) as z:
                for k in acc:
                    acc[k].append(z[k])
        _D[key] = {k: np.concatenate(v) for k, v in acc.items()}
    ref = _D[("causal", 0)]
    for key, d in _D.items():
        assert len(d["ws"]) == 2399
        assert (d["ws"] == ref["ws"]).all() and (d["clip_index"] == ref["clip_index"]).all(), key
        assert np.array_equal(d["g"], ref["g"]), key
    return _D


def scene(i):
    return SCN[i]


def trk(run, seed, i, who):
    """the executed track [30, 2] (forward, left) of scene i: who in {'gt', 'model', 'floor'}.  The regression hook lives HERE and nowhere else."""
    d = load()[(run, seed)]
    r = SCN[i]["grid_row"]
    if who == "gt":
        return d["g"][r]
    swap = SWAP_TAG and SWAP_TAG == SCN[i]["tag"]
    if who == "model":
        return d["fl"][r] if swap else d["cl"][r]
    if who == "floor":
        return d["cl"][r] if swap else d["fl"][r]
    raise KeyError(who)


def ade(pred, g):
    return np.linalg.norm(np.asarray(pred, np.float64) - np.asarray(g, np.float64), axis=-1).mean(-1)


def ade6(run, seed, i, who):
    return float(ade(trk(run, seed, i, who), trk(run, seed, i, "gt")))


def per_window_delta(run, seed):
    """[2399] ADE6(model) - ADE6(floor), straight from the dumps (no swap: eval-wide)."""
    d = load()[(run, seed)]
    return ade(d["cl"], d["g"]) - ade(d["fl"], d["g"])


def mm(x, spec="{:.2f}"):
    """format with the typographic minus of the RESULT files."""
    return spec.format(x).replace("-", "\u2212")


def frame_for(i, run, f):
    """camera frame of scene i at +0.2 f s: causal = the LOGGED frame at cache step ws + f; strict = the FIRST frame (ws) held."""
    s = SCN[i]
    j = s["ws"] + (f if run == "causal" else 0)
    J = join()
    assert s["ws"] + 30 < int(J["n"][s["clip_index"]]), (s["tag"], "window end beyond the clip")
    return L.v2ep_frame(s["clip"], int(j))


# ------------------------------------------------------------------------------------------------ RESULT files
_RT: dict = {}


def rtext(name):
    if name not in _RT:
        _RT[name] = open(f"{RDIR}/{name}.md", encoding="utf-8", newline="").read().replace("\r\n", "\n")
    return _RT[name]


_INT = re.compile(r"^([\u2212+\-]?\d+(?:\.\d+)?) \[([\u2212+\-]?\d+(?:\.\d+)?), ([\u2212+\-]?\d+(?:\.\d+)?)\]\s*(.*)$")


def row_cells(name, row_key, nth=0):
    hits = []
    for line in rtext(name).splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if row_key in cells[0]:
                hits.append(cells)
    if len(hits) <= nth or (nth == 0 and len(hits) != 1):
        raise KeyError(f"{name}: row key {row_key!r} matched {len(hits)} table rows")
    return hits[nth]


def RC(name, row_key, col, nth=0, interval=True):
    """verbatim cell `col` (1 = s0, 2 = s1) of the table row whose first cell contains `row_key`.  Returns dict(val, lo, hi, word, raw); registers it."""
    raw = row_cells(name, row_key, nth)[col]
    plain = raw.replace("**", "").replace("\u2705", "").strip()
    out = {"raw": raw, "plain": plain, "file": name, "row": row_key, "col": col, "nth": nth}
    m = _INT.match(plain)
    pt = re.match(r"^([−+\-]?\d+(?:\.\d+)?)(.*)$", plain)
    if m:
        out.update(val=m.group(1), lo=m.group(2), hi=m.group(3), word=m.group(4).strip().lstrip(",").strip())
    elif interval:
        raise ValueError(f"{name}/{row_key}/{col}: no 'v [lo, hi]' cell: {plain!r}")
    elif pt:
        out.update(val=pt.group(1), word=pt.group(2).strip().lstrip(",").strip())
    L.REG.append({"kind": "res", "scene": CUR["scene"], "file": name, "row": row_key, "col": col, "nth": nth, "shown": plain})
    return out


def status(lo, hi, better_if_neg=True):
    """sign of the interval -> 'better' / 'WORSE' / 'n.s.' (lower is better for every row used here)."""
    lo, hi = float(lo.replace("\u2212", "-")), float(hi.replace("\u2212", "-"))
    if hi < 0:
        return "better"
    if lo > 0:
        return "WORSE"
    return "n.s."


def RT(name, snippet):
    """a verbatim snippet of a RESULT file shown on screen; registered."""
    assert snippet in rtext(name), (name, snippet)
    L.REG.append({"kind": "txt", "scene": CUR["scene"], "file": name, "shown": snippet})
    return snippet


def ade_reg(run, seed, i, who, spec="{:.2f}"):
    v = ade6(run, seed, i, who)
    s = spec.format(v)
    L.REG.append({"kind": "ade6", "scene": CUR["scene"], "run": run, "seed": seed, "who": who, "tag": SCN[i]["tag"], "grid_row": SCN[i]["grid_row"], "spec": spec, "shown": s})
    return s


def dlt_reg(run, seed, i, spec="{:+.2f}"):
    v = ade6(run, seed, i, "model") - ade6(run, seed, i, "floor")
    s = spec.format(v).replace("-", "\u2212")
    L.REG.append({"kind": "ade6d", "scene": CUR["scene"], "run": run, "seed": seed, "tag": SCN[i]["tag"], "grid_row": SCN[i]["grid_row"], "spec": spec, "shown": s})
    return s
