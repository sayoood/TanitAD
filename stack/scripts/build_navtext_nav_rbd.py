#!/usr/bin/env python3
"""build_navtext_nav_rbd.py -- PREREG_V8F_NAVRBD section 2: the 15-column ROUNDABOUT exit / countdown channel ``nav_rb`` [N, 15], ROW-ALIGNED to the join.

Columns (all float32; every column is 0 wherever c0 is 0):
  c0       state     = 1 iff the navtok row at the step has flag 0 and a roundabout class (RB_* / ROTARY*): exactly the NAVRB2 channel
  c1..c6   exit_cur  = one-hot of the exit number in the router text at the step, bins {1, 2, 3, 4, 5, 6+}; all zero when the text has no exit number
  c7..c12  exit_last = one-hot of the exit number of the LATEST row at or before the step with flag 0 and an exit number (BACKWARD read only), same bins; all zero if none
  c13      dist      = max(0, dist_text - arc travelled since the row's time) / 100, clipped to [0, 3] (the ``clip_nav`` rule of build_navtext_nav.py, sign -1); 0 when the text has no distance
  c14      dist_valid= 1 iff the router text carries a distance

Reads ``{split}_cls_id, _flag, _dist_m, _text_i, _age_s`` and ``vocab`` of a navtok table, the texts (``--texts``, text_i -> text; exit number by ``navtok.parse_full``), the join's ``now_s`` and the
native tracks (arc, for c13 only).  No join label, no v9 nav, no situation module.  Splits ``train`` / ``eval`` (the 10k held-out goes in as ``eval``).

G-CLOCK-RBD, both splits, before anything is written: (a) every column equals an independent per-step recompute (c13: the pinned ``clip_nav`` itself); (b) the age of every row read (the NOW row and the
exit-last row) is >= 0 at the step that uses it; (c) the structural checks (zero off c0, at most one 1 per one-hot block, c13 in [0, 3], c14 in {0, 1}); (d) the deliberate-regression arm (every row read
``--regression-shift`` = 5 steps = 1 s AHEAD) must FAIL (n differ > 0 and age negative somewhere); (e) with ``--c0-ref`` c0 equals the NAVRB2 1-column table bit for bit; (f) optional expected counts.
usage: python build_navtext_nav_rbd.py --navtok T.npz --texts texts.json --join-train J --join-eval J --tracks native_tracks.pt [--c0-ref navtext_nav_rb.npz] [--grid g.json] --out DIR [--out-name N.npz]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RB_PREFIXES = ("RB_", "ROTARY")
STEP_S = 0.2                      # the cache step (10 Hz track, every second row)
REGRESSION_SHIFT = 5              # 1 s ahead: the deliberate-regression arm
LIVE_SHIFT = 0                    # the live read: the step itself (a mutant sets 5)
SPLITS = ("train", "eval")
NBINS = 6                         # exit bins {1, 2, 3, 4, 5, 6+}
DIM = 15
C_EC, C_EL, C_D, C_DV = 1, 1 + NBINS, 1 + 2 * NBINS, 2 + 2 * NBINS      # 1, 7, 13, 14
GROUPS = {"c0": (0, 1), "exit_cur": (1, 7), "exit_last": (7, 13), "dist": (13, 14), "dist_valid": (14, 15)}
OUT_NAME = "navtext_nav_rbd.npz"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rb_class_mask(vocab):
    return np.array([str(c).startswith(RB_PREFIXES) for c in vocab], bool)


def exit_array(texts, parse_full):
    """text_i -> exit number (-1 if the text carries none), by ``navtok.parse_full``."""
    out = np.full(len(texts), -1, np.int32)
    for i, t in enumerate(texts):
        e = parse_full(t).exit_n
        out[i] = -1 if e is None else int(e)
    return out


def load_navtok(path=None):
    """Import ``navtok`` (the frozen grammar) from a directory or file path; falls back to an installed module."""
    if path:
        p = path if path.endswith(".py") else os.path.join(path, "navtok.py")
        spec = importlib.util.spec_from_file_location("navtok_rbd", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules["navtok_rbd"] = m
        spec.loader.exec_module(m)
        return m
    import navtok
    return navtok


def load_tracks(path):
    import torch
    o = torch.load(path, map_location="cpu", weights_only=True)
    return {c: t.double().numpy() for c, t in o["tracks"].items()}


def exit_bin(ex):
    """exit number -> bin 0..5 ({1..5, 6+}); -1 for 'no exit number' (ex < 1)."""
    ex = np.asarray(ex)
    return np.where(ex >= 1, np.minimum(ex, NBINS) - 1, -1)


def source_rows(off, n, shift: int = 0):
    """The table row each step reads: the step itself (shift 0) or ``shift`` steps later in the SAME clip (clamped to the clip's last step; the regression arm)."""
    src = np.arange(int(off[-1] + n[-1]), dtype=np.int64)
    if shift:
        for o, k in zip(np.asarray(off, np.int64), np.asarray(n, np.int64)):
            src[o:o + k] = o + np.minimum(np.arange(k) + shift, k - 1)
    return src


def _fill(t):
    """NaN of a per-step time array filled by linear interpolation over the finite entries (clamped at the ends); the rule of build_navtext_nav._fill."""
    t = np.asarray(t, np.float64)
    ok = np.isfinite(t)
    if ok.all():
        return t
    if not ok.any():
        return np.arange(len(t)) * 0.2
    return np.interp(np.arange(len(t)), np.flatnonzero(ok), t[ok])


def clip_arc(track, now_s, n):
    """(arc at every 10 Hz row, time at every row, time of each cache step): the clip_nav arc machinery."""
    tr = np.asarray(track, np.float64)
    step = np.linalg.norm(np.diff(tr[:, :2], axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(np.where(np.isfinite(step), step, 0.0))])
    tn = _fill(now_s)
    t_rows = np.interp(np.arange(len(tr), dtype=np.float64) / 2.0, np.arange(n), tn)
    return arc, t_rows, tn


def last_exit_rows(valid):
    """For each step k of one clip: the latest step <= k with ``valid`` (-1 if none). BACKWARD only."""
    idx = np.where(valid, np.arange(len(valid)), -1)
    return np.maximum.accumulate(idx)


def clip_columns(cls, flag, dist, text_i, age, isrb, ex_t, now_s, track, shift: int = 0):
    """One clip: (cols [n, 15] f32, src rows (clip-local) of the NOW read, src rows of the exit-last read, n_unusable_track)."""
    n = len(cls)
    s = np.minimum(np.arange(n) + shift, n - 1)                                                  # the row each step reads (shift 0: itself)
    c, f = cls[s], flag[s]
    on = (f == 0) & (c >= 0) & isrb[np.clip(c, 0, None)]
    cols = np.zeros((n, DIM), np.float32)
    cols[:, 0] = on
    ti = text_i[s]
    ex = np.where((f == 0) & (ti >= 0), ex_t[np.clip(ti, 0, None)], -1)
    b = exit_bin(ex)
    ar = np.arange(n)
    m = b >= 0
    cols[ar[m], C_EC + b[m]] = 1.0                                                                # exit_cur
    exv = np.where((flag == 0) & (text_i >= 0), ex_t[np.clip(text_i, 0, None)], -1)             # the per-row exit number (flag 0 rows only)
    last = last_exit_rows(exv >= 1)[s]                                                            # latest row <= the read row with flag 0 and an exit number
    bl = np.where(last >= 0, exit_bin(exv[np.clip(last, 0, None)]), -1)
    m = bl >= 0
    cols[ar[m], C_EL + bl[m]] = 1.0                                                               # exit_last (backward only)  # MUT:exit_last
    d = dist[s].astype(np.float64)
    hasd = np.isfinite(d)
    cols[:, C_DV] = hasd
    arc, t_rows, tn = clip_arc(track, now_s, n)
    r = np.minimum(2 * s, len(track) - 1)
    ag = age[s].astype(np.float64)
    t0 = np.where(np.isfinite(ag), tn[s] - ag, tn[s])
    travelled = arc[r] - np.interp(t0, t_rows, arc)                                               # MUT:arc
    usable = hasd & np.isfinite(travelled)
    dd = np.where(usable, np.maximum(0.0, np.where(hasd, d, 0.0) - np.where(usable, travelled, 0.0)), 0.0)
    cols[:, C_D] = np.where(usable, np.clip(dd / 100.0, 0.0, 3.0), 0.0)
    cols[:, 1:] *= on[:, None]                                                                    # MUT:gate (silent off roundabouts)
    return cols, s, last, int((on & hasd & ~np.isfinite(travelled)).sum())


def build_columns(tab, split, texts_ex, J, tracks, shift: int = 0, clips=None):
    """The [N, 15] table of one split plus the rows read."""
    off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
    cls, flag = np.asarray(tab[f"{split}_cls_id"]), np.asarray(tab[f"{split}_flag"])
    dist, ti, age = (np.asarray(tab[f"{split}_{k}"]) for k in ("dist_m", "text_i", "age_s"))
    isrb = rb_class_mask(tab["vocab"])
    N = int(off[-1] + n[-1])
    out = np.zeros((N, DIM), np.float32)
    src = np.zeros(N, np.int64)
    srcl = np.zeros(N, np.int64)
    unusable = 0
    for ci, (name, o, k) in enumerate(zip(np.asarray(J["clip"]).astype(str), off, n)):
        o, k = int(o), int(k)
        sl = slice(o, o + k)
        cols, s, last, bad = clip_columns(cls[sl], flag[sl], dist[sl], ti[sl], age[sl], isrb, texts_ex, np.asarray(J["now_s"][sl]), tracks[name], shift)
        out[sl], src[sl], srcl[sl], unusable = cols, o + s, np.where(last >= 0, o + last, -1), unusable + bad
    return out, src, srcl, unusable


# ------------------------------------------------------------------ the independent recompute (scalar, per step) and G-CLOCK-RBD
def _clip_nav_module():
    spec = importlib.util.spec_from_file_location("build_navtext_nav_ref", os.path.join(HERE, "build_navtext_nav.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def reference_columns(tab, split, texts_ex, J, tracks):
    """Per-step scalar recompute. c0 / c1..c14 by explicit loops over the ON steps; c13 by the pinned ``clip_nav`` (every roundabout class mapped to a side, so it returns clip(d / 100, 0, 3))."""
    CN = _clip_nav_module()
    vocab = [str(v) for v in tab["vocab"]]
    off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
    cls, flag = np.asarray(tab[f"{split}_cls_id"]).tolist(), np.asarray(tab[f"{split}_flag"]).tolist()
    dist, ti, age = np.asarray(tab[f"{split}_dist_m"]), np.asarray(tab[f"{split}_text_i"]).tolist(), np.asarray(tab[f"{split}_age_s"])
    ex_t = [int(x) for x in texts_ex]
    side = np.array([1 if v.startswith(RB_PREFIXES) else 0 for v in vocab], np.int8)
    out = np.zeros((int(off[-1] + n[-1]), DIM), np.float32)
    for name, o, k in zip(np.asarray(J["clip"]).astype(str), off.tolist(), n.tolist()):
        nav, _, _, _ = CN.clip_nav(np.asarray(cls[o:o + k]), dist[o:o + k], np.asarray(flag[o:o + k]), age[o:o + k], np.asarray(J["now_s"][o:o + k]), tracks[name], side, -1)
        for j in range(k):
            g = o + j
            if not (flag[g] == 0 and cls[g] >= 0 and vocab[cls[g]].startswith(("RB_", "ROTARY"))):
                continue
            out[g, 0] = 1.0
            e = ex_t[ti[g]] if ti[g] >= 0 else -1
            if e >= 1:
                out[g, 1 + (min(e, 6) - 1)] = 1.0
            for q in range(j, -1, -1):                                                               # backward scan: the latest row at or before the step with flag 0 and an exit number
                eq = ex_t[ti[o + q]] if (flag[o + q] == 0 and ti[o + q] >= 0) else -1
                if eq >= 1:
                    out[g, 7 + (min(eq, 6) - 1)] = 1.0
                    break
            out[g, 13] = nav[j, 3] if np.isfinite(dist[g]) else 0.0
            out[g, 14] = 1.0 if np.isfinite(dist[g]) else 0.0
    return out


def structure_problems(cols):
    """The structural checks of section 2: zero off c0, <= 1 per one-hot block, c13 in [0, 3], c14 in {0, 1}."""
    off0 = cols[:, 0] == 0
    p = {"n_nonzero_where_c0_zero": int((cols[off0] != 0).any(1).sum()),
         "n_exit_cur_gt1": int((cols[:, 1:7].sum(1) > 1).sum()), "n_exit_last_gt1": int((cols[:, 7:13].sum(1) > 1).sum()),
         "n_c13_out_of_range": int(((cols[:, 13] < 0) | (cols[:, 13] > 3)).sum()), "n_c14_not_binary": int((~np.isin(cols[:, 14], (0.0, 1.0))).sum()),
         "n_onehot_not_binary": int((~np.isin(cols[:, :13], (0.0, 1.0))).sum())}
    return p, all(v == 0 for v in p.values())


def ages_used(tab, split, src, srcl):
    """Age (s) of every row read, at the step that uses it: NOW row ``age[src] - 0.2 (src - j)`` and exit-last row ``age[last] - 0.2 (last - j)`` (the row's own age plus the time since it)."""
    age = np.asarray(tab[f"{split}_age_s"]).astype(np.float64)
    j = np.arange(len(src))
    a_now = age[src] - STEP_S * (src - j)
    ok_l = srcl >= 0
    a_last = np.where(ok_l, age[np.clip(srcl, 0, None)] - STEP_S * (srcl - j), np.inf)
    return a_now, a_last


def clock_rbd(tab, split, cols, src, srcl, ref, texts_ex=None, expect=None, grid=None, off=None, n=None):
    on = cols[:, 0] > 0
    a_now, a_last = ages_used(tab, split, src, srcl)
    d = (cols != ref).any(1)
    st, st_ok = structure_problems(cols)
    r = {"n_steps": int(len(on)), "n_on": int(on.sum()), "n_differ_from_recompute": int(d.sum()), "per_column_n_differ": {k: int((cols[:, a:b] != ref[:, a:b]).any(1).sum()) for k, (a, b) in GROUPS.items()},
         "min_age_now_row_s": float(a_now[on].min()) if on.any() else None, "min_age_last_row_s": float(a_last[on & np.isfinite(a_last)].min()) if (on & np.isfinite(a_last)).any() else None,
         "n_age_negative": int((a_now[on] < 0).sum() + (a_last[on] < 0).sum()), "structure": st}
    ok = r["n_differ_from_recompute"] == 0 and r["n_age_negative"] == 0 and st_ok
    r["n_nonzero_per_column"] = [int((cols[:, c] != 0).sum()) for c in range(DIM)]
    if expect:
        if expect.get("steps") is not None:
            ok &= r["n_on"] == int(expect["steps"])
        if grid is not None:
            idx = {str(c): i for i, c in enumerate(np.asarray(tab[f"{split}_clip"]).astype(str))}
            order = sorted(idx)
            nw, eps = 0, set()
            for clip, ts in grid["windows"].items():
                for t in ts:
                    if cols[int(off[idx[clip]]) + int(t), 0] > 0:
                        nw += 1
                        eps.add(order.index(clip))
            r["grid_windows_on_at_now"], r["grid_episodes_on_at_now"] = nw, len(eps)
            if expect.get("windows") is not None:
                ok &= nw == int(expect["windows"]) and len(eps) == int(expect["episodes"])
    r["pass"] = bool(ok)
    return r


def check_aligned(tab, J, split, what):
    for k in ("clip", "off", "n"):
        a, b = np.asarray(tab[f"{split}_{k}"]), np.asarray(J[k])
        if a.shape != b.shape or not np.array_equal(a.astype(str) if k == "clip" else a, b.astype(str) if k == "clip" else b):
            raise SystemExit(f"REFUSED: navtok {split}_{k} does not equal {what}'s (misaligned rows)")


def check_texts(tab, split, texts, parse_full):
    """The texts file is THIS table's: parse(texts[text_i]) reproduces the table's class and distance at every distinct text_i."""
    import numpy as _np
    ti = _np.unique(_np.asarray(tab[f"{split}_text_i"]))
    ti = ti[ti >= 0]
    cls, dist = _np.asarray(tab[f"{split}_cls_id"]), _np.asarray(tab[f"{split}_dist_m"])
    first = {}
    for k, t in enumerate(_np.asarray(tab[f"{split}_text_i"]).tolist()):
        if t >= 0 and t not in first:
            first[t] = k
    vocab = [str(v) for v in tab["vocab"]]
    bad = 0
    for t, k in first.items():
        p = parse_full(texts[t])
        if vocab[cls[k]] != p.cls or not (np.isnan(dist[k]) if p.dist_m is None else np.float32(p.dist_m) == dist[k]):
            bad += 1
    return {"n_distinct_text_i": len(first), "n_mismatch": bad}


def build(tab, texts_ex, joins, tracks, expect_eval=None, grid=None, c0_ref=None, shift_reg=REGRESSION_SHIFT):
    """Returns (arrays for the npz, report, ok)."""
    out, rep, ok = {}, {}, True
    for split in SPLITS:
        if f"{split}_cls_id" not in tab.files or joins.get(split) is None:
            continue
        J = joins[split]
        check_aligned(tab, J, split, "the join")
        off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
        cols, src, srcl, bad = build_columns(tab, split, texts_ex, J, tracks, LIVE_SHIFT)
        regc, rsrc, rsrcl, _ = build_columns(tab, split, texts_ex, J, tracks, shift_reg)
        ref = reference_columns(tab, split, texts_ex, J, tracks)
        ex = expect_eval if split == "eval" else None
        g = grid if split == "eval" else None
        qa = clock_rbd(tab, split, cols, src, srcl, ref, expect=ex, grid=g, off=off, n=n)
        qa_reg = clock_rbd(tab, split, regc, rsrc, rsrcl, ref, expect=ex, grid=g, off=off, n=n)
        qa["n_on_steps_with_unusable_track"] = bad
        rep[split] = {"G-CLOCK-RBD": qa, f"G-CLOCK-RBD_regression_arm_shift_{shift_reg}": qa_reg}
        if c0_ref is not None and f"{split}_nav_rb" in c0_ref:
            r0 = np.asarray(c0_ref[f"{split}_nav_rb"]).reshape(-1)
            rep[split]["c0_equals_ref_bit_for_bit"] = bool(r0.shape == cols[:, 0].shape and np.array_equal(r0, cols[:, 0]))
            rep[split]["c0_ref_n_on"] = int((r0 > 0).sum())
            ok &= rep[split]["c0_equals_ref_bit_for_bit"]
        rep[split]["G-CLOCK-RBD_pass"] = bool(qa["pass"] and not qa_reg["pass"])
        ok &= rep[split]["G-CLOCK-RBD_pass"]
        out.update({f"{split}_clip": np.asarray(tab[f"{split}_clip"]).astype(str), f"{split}_off": off, f"{split}_n": n, f"{split}_nav_rb": cols})
    return out, rep, bool(ok)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--navtok", required=True)
    ap.add_argument("--texts", required=True, help="texts.json: text_i -> text of THIS navtok table")
    ap.add_argument("--navtok-src", default="", help="directory (or file) holding navtok.py, the frozen grammar")
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--join-train", default="")
    ap.add_argument("--join-eval", default="")
    ap.add_argument("--c0-ref", default="", help="the NAVRB2 1-column table: c0 must equal it bit for bit")
    ap.add_argument("--grid", default="", help="eval grid windows json")
    ap.add_argument("--expect-eval-steps", type=int, default=None)
    ap.add_argument("--expect-rb-windows", type=int, default=None)
    ap.add_argument("--expect-rb-episodes", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-name", default=OUT_NAME)
    a = ap.parse_args(argv)
    NT = load_navtok(a.navtok_src or None)
    tab = np.load(a.navtok, allow_pickle=False)
    texts = json.load(open(a.texts, encoding="utf-8"))
    ex_t = exit_array(texts, NT.parse_full)
    joins = {}
    for sp, p in (("train", a.join_train), ("eval", a.join_eval)):
        if p:
            with np.load(p, allow_pickle=False) as z:
                joins[sp] = {k: z[k] for k in z.files}
    tracks = load_tracks(a.tracks)
    c0 = np.load(a.c0_ref, allow_pickle=False) if a.c0_ref else None
    grid = json.load(open(a.grid, encoding="utf-8")) if a.grid else None
    exp = {"steps": a.expect_eval_steps, "windows": a.expect_rb_windows, "episodes": a.expect_rb_episodes}
    out, rep, ok = build(tab, ex_t, joins, tracks, exp, grid, c0)
    for sp in joins:
        rep.setdefault(sp, {})["texts_check"] = check_texts(tab, sp, texts, NT.parse_full)
        ok &= rep[sp]["texts_check"]["n_mismatch"] == 0
    rep["inputs"] = {"navtok_sha256": sha_file(a.navtok), "texts_sha256": sha_file(a.texts), "navtok_py_sha256": sha_file(NT.__file__), "tracks_sha256": sha_file(a.tracks),
                     "join_train_sha256": sha_file(a.join_train) if a.join_train else None, "join_eval_sha256": sha_file(a.join_eval) if a.join_eval else None,
                     "c0_ref_sha256": sha_file(a.c0_ref) if a.c0_ref else None, "grid_sha256": sha_file(a.grid) if a.grid else None,
                     "builder_sha256": sha_file(os.path.abspath(__file__))}
    rep["rule"] = "PREREG_V8F_NAVRBD section 2: 15 columns from the router text rows at or before the step; c13 dead-reckoned by the clip_nav arc rule; all zero where c0 is 0"
    rep["columns"] = {k: list(v) for k, v in GROUPS.items()}
    if not ok:
        print(json.dumps(rep, indent=1, default=str)[:4000], file=sys.stderr)
        print("REFUSED: G-CLOCK-RBD failed (or its regression arm passed); no table written", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    out["meta_json"] = np.array(json.dumps({"rule": rep["rule"], "inputs": rep["inputs"], "columns": rep["columns"]}))
    dst = os.path.join(a.out, a.out_name)
    np.savez_compressed(dst, **out)
    rep["table_sha256"] = sha_file(dst)
    json.dump(rep, open(os.path.join(a.out, a.out_name.replace(".npz", "_report.json")), "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({k: rep[k] for k in rep if k in ("train", "eval", "table_sha256")}, indent=1, default=str)[:3500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
