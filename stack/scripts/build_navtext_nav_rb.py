#!/usr/bin/env python3
"""build_navtext_nav_rb.py -- PREREG_V8F_NAVRB2: the ROUNDABOUT state channel ``nav_rb`` [N, 1] of the route-text nav, ROW-ALIGNED to the join.

Rule (card section 2, nothing fitted): ``nav_rb[j] = 1`` iff the navtok row at cache step ``j`` (the latest router row with ts <= now - 195 ms, built by navtok/build_table.py) has ``flag == 0``
and a roundabout class (a vocab name starting ``RB_`` or ``ROTARY``); else 0.  It reads ``{split}_cls_id``, ``{split}_flag``, ``{split}_age_s`` (the clock check only) and ``vocab`` of the
navtok table and NOTHING else: no track, no ego speed, no join label, no v9 nav, no situation module.  The row alignment comes from the pinned STRAT1 text-nav table (``--align``,
navtext_nav.npz, whose clip / off / n the STRAT1 builder checked equal to the join's) and, when given, from the joins themselves.

G-CLOCK-RB (written before the table, both splits): (a) the state equals the NAVRB rule recomputed independently per step; (b) every row the state is read from has age >= 0 at the step it
is used (the row lies in the past of NOW); (c) on eval: ``--expect-eval-steps`` steps on, and (with ``--grid``) ``--expect-rb-windows`` / ``--expect-rb-episodes`` grid windows / episodes on at
NOW (the NAVRB measurement: 1,334 / 235 / 16).  The deliberate-regression arm (the state read ``--regression-shift`` = 5 steps = 1 s AHEAD) must FAIL; the build refuses to write otherwise.
usage:  python build_navtext_nav_rb.py --navtok navtok_table.npz --align navtext_nav.npz [--join-train J --join-eval J] [--grid eval_grid_windows.json] --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
for p in (os.path.join(REPO, "stack"),):
    if p not in sys.path:
        sys.path.insert(0, p)

RB_PREFIXES = ("RB_", "ROTARY")
STEP_S = 0.2                      # the cache step (10 Hz track, every second row)
REGRESSION_SHIFT = 5              # 1 s ahead: the deliberate-regression arm
SPLITS = ("train", "eval")
OUT_NAME = "navtext_nav_rb.npz"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rb_class_mask(vocab):
    """class id -> True for a roundabout class (``RB_*`` / ``ROTARY*``)."""
    return np.array([str(c).startswith(RB_PREFIXES) for c in vocab], bool)


def source_rows(off, n, shift: int = 0):
    """The table row each step reads: the step itself (shift 0) or ``shift`` steps later in the SAME clip (clamped to the clip's last step; the regression arm)."""
    src = np.arange(int(off[-1] + n[-1]), dtype=np.int64)
    if shift:
        for o, k in zip(np.asarray(off, np.int64), np.asarray(n, np.int64)):
            src[o:o + k] = o + np.minimum(np.arange(k) + shift, k - 1)
    return src


def state(cls_id, flag, vocab, off, n, shift: int = 0):
    """``nav_rb`` [N, 1] float32 and the source row of each step."""
    isrb = rb_class_mask(vocab)
    cls_id, flag = np.asarray(cls_id), np.asarray(flag)
    src = source_rows(off, n, shift)
    c, f = cls_id[src], flag[src]
    on = (f == 0) & (c >= 0) & isrb[np.clip(c, 0, None)]
    return on.astype(np.float32)[:, None], src


def reference_state(cls_id, flag, vocab):
    """The NAVRB rule (navrb.py ``step_arrays`` + ``is_state``), per step, written independently of :func:`state`."""
    out = np.zeros(len(cls_id), bool)
    names = [str(v) for v in vocab]
    for j, (c, f) in enumerate(zip(np.asarray(cls_id).tolist(), np.asarray(flag).tolist())):
        out[j] = bool(f == 0 and c >= 0 and (names[c].startswith("RB_") or names[c].startswith("ROTARY")))
    return out


def grid_counts(nav_rb, off, clips, grid):
    """Eval grid windows (and episodes = the sorted clip index) whose NOW step has the state on."""
    idx = {str(c): i for i, c in enumerate(np.asarray(clips).astype(str))}
    order = sorted(idx)
    nw, eps = 0, set()
    for clip, ts in grid["windows"].items():
        ci = idx[clip]
        for t in ts:
            if nav_rb[int(off[ci]) + int(t), 0] > 0:
                nw += 1
                eps.add(order.index(clip))
    return nw, len(eps)


def clock_rb(tab, split, nav_rb, src, grid=None, expect=None):
    """G-CLOCK-RB on one split. ``expect`` = {"steps", "windows", "episodes"} (eval) or None."""
    cls, flag, age = (np.asarray(tab[f"{split}_{k}"]) for k in ("cls_id", "flag", "age_s"))
    ref = reference_state(cls, flag, tab["vocab"])
    on = nav_rb[:, 0] > 0
    j = np.arange(len(on))
    age_eff = age[src].astype(np.float64) - STEP_S * (src - j)              # the source row's age at the step that USES it
    r = {"n_steps": int(len(on)), "n_on": int(on.sum()), "n_clips_on": int(sum(1 for o, k in zip(tab[f"{split}_off"], tab[f"{split}_n"]) if on[int(o):int(o + k)].any())),
         "n_differ_from_rule": int((on != ref).sum()), "min_age_used_s": float(age_eff[on].min()) if on.any() else None,
         "n_age_negative": int((age_eff[on] < 0).sum())}
    ok = r["n_differ_from_rule"] == 0 and r["n_age_negative"] == 0
    if expect:
        if expect.get("steps") is not None:
            ok &= r["n_on"] == int(expect["steps"])
        if grid is not None:
            r["grid_windows_on_at_now"], r["grid_episodes_on_at_now"] = grid_counts(nav_rb, tab[f"{split}_off"], tab[f"{split}_clip"], grid)
            ok &= r["grid_windows_on_at_now"] == int(expect["windows"]) and r["grid_episodes_on_at_now"] == int(expect["episodes"])
    r["pass"] = bool(ok)
    return r


def check_aligned(tab, other, split, what):
    for k in ("clip", "off", "n"):
        a, b = np.asarray(tab[f"{split}_{k}"]), np.asarray(other[k] if k in other else other[f"{split}_{k}"])
        if a.shape != b.shape or not np.array_equal(a.astype(str) if k == "clip" else a, b.astype(str) if k == "clip" else b):
            raise SystemExit(f"REFUSED: navtok {split}_{k} does not equal {what}'s (misaligned rows)")


def build(tab, align, joins=None, grid=None, expect_eval=None, shift_reg=REGRESSION_SHIFT):
    """Returns (arrays for the npz, report, ok)."""
    out, rep, ok = {}, {}, True
    for split in SPLITS:
        if f"{split}_cls_id" not in tab.files:
            continue
        check_aligned(tab, align, split, "the align table")
        if joins and joins.get(split) is not None:
            check_aligned(tab, joins[split], split, "the join")
        off, n = np.asarray(tab[f"{split}_off"]), np.asarray(tab[f"{split}_n"])
        nav_rb, src = state(tab[f"{split}_cls_id"], tab[f"{split}_flag"], tab["vocab"], off, n, 0)
        reg, src_r = state(tab[f"{split}_cls_id"], tab[f"{split}_flag"], tab["vocab"], off, n, shift_reg)
        ex = expect_eval if split == "eval" else None
        g = grid if split == "eval" else None
        qa, qa_reg = clock_rb(tab, split, nav_rb, src, g, ex), clock_rb(tab, split, reg, src_r, g, ex)
        rep[split] = {"G-CLOCK-RB": qa, f"G-CLOCK-RB_regression_arm_shift_{shift_reg}": qa_reg, "G-CLOCK-RB_pass": bool(qa["pass"] and not qa_reg["pass"])}
        ok &= rep[split]["G-CLOCK-RB_pass"]
        out.update({f"{split}_clip": np.asarray(tab[f"{split}_clip"]).astype(str), f"{split}_off": off, f"{split}_n": n, f"{split}_nav_rb": nav_rb})
    return out, rep, bool(ok)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--navtok", required=True)
    ap.add_argument("--align", required=True, help="navtext_nav.npz (the pinned STRAT1 text nav: its clip / off / n are the join's)")
    ap.add_argument("--join-train", default="")
    ap.add_argument("--join-eval", default="")
    ap.add_argument("--grid", default="", help="eval_grid_windows.json")
    ap.add_argument("--expect-eval-steps", type=int, default=1334)
    ap.add_argument("--expect-rb-windows", type=int, default=235)
    ap.add_argument("--expect-rb-episodes", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    tab = np.load(a.navtok, allow_pickle=False)
    align = np.load(a.align, allow_pickle=False)
    joins = {}
    if a.join_train or a.join_eval:
        from tanitad.data import v8f_mini_data as D
        joins = {"train": D.load_join(a.join_train) if a.join_train else None, "eval": D.load_join(a.join_eval) if a.join_eval else None}
    grid = json.load(open(a.grid, encoding="utf-8")) if a.grid else None
    exp = {"steps": a.expect_eval_steps, "windows": a.expect_rb_windows, "episodes": a.expect_rb_episodes}
    out, rep, ok = build(tab, align, joins, grid, exp)
    rep["inputs"] = {"navtok_sha256": sha_file(a.navtok), "align_sha256": sha_file(a.align), "grid_sha256": sha_file(a.grid) if a.grid else None,
                     "join_train_sha256": sha_file(a.join_train) if a.join_train else None, "join_eval_sha256": sha_file(a.join_eval) if a.join_eval else None}
    rep["rule"] = "PREREG_V8F_NAVRB2 section 2: nav_rb = 1 iff navtok flag == 0 and class RB_* / ROTARY* at the step; no track, no ego, no label"
    rep["rb_classes"] = [str(v) for v, m in zip(tab["vocab"], rb_class_mask(tab["vocab"])) if m]
    if not ok:
        print(json.dumps(rep, indent=1, default=str)[:3000], file=sys.stderr)
        print("REFUSED: G-CLOCK-RB failed (or its regression arm passed); no table written", file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    out["meta_json"] = np.array(json.dumps({"rule": rep["rule"], "inputs": rep["inputs"], "rb_classes": rep["rb_classes"]}))
    dst = os.path.join(a.out, OUT_NAME)
    np.savez_compressed(dst, **out)
    rep["navtext_nav_rb_sha256"] = sha_file(dst)
    json.dump(rep, open(os.path.join(a.out, "navtext_nav_rb_report.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: rep[k] for k in rep if k in ("train", "eval", "navtext_nav_rb_sha256")}, indent=1)[:2500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
