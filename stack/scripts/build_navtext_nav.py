"""PREREG_V8F_STRAT1 (BUILD_SPEC section 5): the offline text-nav builder. CPU only (numpy, torch for ``native_tracks.pt``). One file per run, ``navtext_nav.npz`` (+ ``navtext_nav_report.json``).

Inputs: ``navtok_table.npz`` (the router-text table: ``<split>_cls_id`` into ``vocab``, ``_dist_m``, ``_flag``, ``_age_s``), the join (``clip`` / ``off`` / ``n`` / ``now_s``), the 10 Hz ``native_tracks.pt`` (row ``2 j`` is cache step ``j``).

RULE (fixed before any number), per cache step with table flag 0 (any other flag: all five channels 0, ``nav_d`` NaN, as the join's no-row steps):
  1. side: TURN_L / TURN_SHARP_L / TURN_SLIGHT_L -> L; TURN_R / TURN_SHARP_R / TURN_SLIGHT_R -> R; every other class (UTURN, CONTINUE*, KEEP*, MERGE*, RAMP*, RB_*, ROTARY_EXIT, UNPARSED) -> none.
  2. no side: ``[1, 0, 0, 0, 0]`` (FOLLOW), ``nav_d`` NaN.
  3. side with a distance: COUNTDOWN ``d = max(0, dist_m - (arc(now_s) - arc(now_s - age_s)))`` on the LOGGED track's cumulative arc, interpolated in time (row times from ``J["now_s"]`` at even rows, linear between).
  4. side without a distance (bare text): ``d = 0``.
  5. ``d_ann = max(30, 6 v_now)`` (the v9 rule's literals), ``v_now`` = the track speed at row ``2 j``.
  6. token = side if ``d <= d_ann`` else FOLLOW; ``ch3 = clip(d / 100, 0, 3)``; ``ch4 = 1``; ``nav_d = d``.
Channel order (the v9 nav's): [FOLLOW, L, R, d / 100, row present].

G-CLOCK-NAV (launch blocker): on steps where the v9 token is L or R, the text side equals it and both distances are valid, ``median |d_countdown - v9 nav_d| <= median |dist_raw - v9 nav_d|``; the deliberate-regression
arm is the same QA with the sign flipped (``dist + travelled``): it must FAIL.
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

LEFT = ("TURN_L", "TURN_SHARP_L", "TURN_SLIGHT_L")
RIGHT = ("TURN_R", "TURN_SHARP_R", "TURN_SLIGHT_R")
D_ANN_MIN, D_ANN_PER_MS = 30.0, 6.0          # the v9 rule's literals (build_v9_labels.py:44-45, INHERITED)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def side_of(vocab):
    """class id -> 0 none / 1 L / 2 R"""
    s = np.zeros(len(vocab), np.int8)
    for i, c in enumerate(vocab):
        s[i] = 1 if str(c) in LEFT else (2 if str(c) in RIGHT else 0)
    return s


def _fill(t):
    """Fill NaN of a per-step time array by linear interpolation over the finite entries (clamped at the ends)."""
    t = np.asarray(t, np.float64)
    ok = np.isfinite(t)
    if ok.all():
        return t, 0
    if not ok.any():
        return np.arange(len(t)) * 0.2, len(t)
    return np.interp(np.arange(len(t)), np.flatnonzero(ok), t[ok]), int((~ok).sum())


def clip_nav(cls_id, dist, flag, age, now_s, track, side, sign: int = -1, ann_cut: bool = True):
    """One clip. Returns (nav [n, 5] f32, nav_d [n] f32, clamp [n] bool, d_raw [n] f32). ``sign`` -1 = the countdown, +1 = the deliberate regression (distance PLUS travelled).
    ``ann_cut`` False (PREREG_V8F_CKPT120 ``nav_s``): step 5 is d_ann = infinity, so the token is the text's TURN-class side at ANY distance (ch3 = clip(d / 100, 0, 3), ch4 = 1); True (the default) = STRAT1's rule, unchanged."""
    n = len(cls_id)
    nav = np.zeros((n, 5), np.float32)
    nav_d = np.full(n, np.nan, np.float32)
    clamp = np.zeros(n, bool)
    d_raw = np.full(n, np.nan, np.float32)
    tr = np.asarray(track, np.float64)
    xy = tr[:, :2]
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(np.where(np.isfinite(step), step, 0.0))])                # cumulative arc at every 10 Hz row (a NaN row, e.g. the first two of the native track, adds 0: MEASURED 2026-11-16, a NaN poisons the whole cumsum and every countdown reads 0)
    tn, _ = _fill(now_s)                                                                           # time of cache step j = track row 2 j
    rows = np.arange(len(tr), dtype=np.float64)
    t_rows = np.interp(rows / 2.0, np.arange(n), tn)                                                # linear between the even rows
    for j in range(n):
        if flag[j] != 0:
            continue
        sd = int(side[cls_id[j]])
        r = min(2 * j, len(tr) - 1)
        if sd == 0:
            nav[j] = [1, 0, 0, 0, 0]
            continue
        has_d = bool(np.isfinite(dist[j]))
        if has_d:
            t0 = float(tn[j]) - float(age[j]) if np.isfinite(age[j]) else float(tn[j])
            if t0 < t_rows[0]:
                clamp[j] = True
            travelled = float(arc[r] - np.interp(t0, t_rows, arc))
            if not np.isfinite(travelled):
                continue                                                                           # an unusable track step: left as the no-row state (counted in the report as n_unusable_track)
            d = max(0.0, float(dist[j]) - travelled) if sign < 0 else float(dist[j]) + travelled      # sign +1 = the deliberate-regression arm
            d_raw[j] = dist[j]
        else:
            d = 0.0
        v_now = float(tr[r, 3]) if tr.shape[1] > 3 and np.isfinite(tr[r, 3]) else 0.0
        d_ann = max(D_ANN_MIN, D_ANN_PER_MS * v_now)
        tok = sd if (d <= d_ann or not ann_cut) else 0
        nav[j, tok] = 1.0
        nav[j, 3] = np.clip(d / 100.0, 0.0, 3.0)
        nav[j, 4] = 1.0
        nav_d[j] = d
    return nav, nav_d, clamp, d_raw


def build_split(J, tab, split, tracks, sign: int = -1, ann_cut: bool = True):
    cls, dist, flag, age = (np.asarray(tab[f"{split}_{k}"]) for k in ("cls_id", "dist_m", "flag", "age_s"))
    side = side_of(tab["vocab"])
    N = int(J["off"][-1] + J["n"][-1])
    nav = np.zeros((N, 5), np.float32)
    nav_d = np.full(N, np.nan, np.float32)
    clamp = np.zeros(N, bool)
    d_raw = np.full(N, np.nan, np.float32)
    for c, (name, o, n) in enumerate(zip(np.asarray(J["clip"]).astype(str), J["off"], J["n"])):
        o, n = int(o), int(n)
        a, b, cl, dr = clip_nav(cls[o:o + n], dist[o:o + n], flag[o:o + n], age[o:o + n], np.asarray(J["now_s"][o:o + n]), tracks[name], side, sign, ann_cut)
        nav[o:o + n], nav_d[o:o + n], clamp[o:o + n], d_raw[o:o + n] = a, b, cl, dr
    return nav, nav_d, clamp, d_raw


def clock_nav_qa(J, nav, nav_d, d_raw, sign_label="countdown"):
    """G-CLOCK-NAV on the steps where the v9 token is L or R, the text side equals it and both distances are valid. Returns the two medians and the verdict."""
    v9, v9d = np.asarray(J["nav"]), np.asarray(J["nav_d"], np.float64)
    v9_side = np.where(v9[:, 1] > 0, 1, np.where(v9[:, 2] > 0, 2, 0))
    tx_side = np.where(nav[:, 1] > 0, 1, np.where(nav[:, 2] > 0, 2, 0))
    m = (v9_side > 0) & (v9_side == tx_side) & np.isfinite(v9d) & np.isfinite(nav_d) & np.isfinite(d_raw)
    e_new = np.abs(nav_d[m].astype(np.float64) - v9d[m])
    e_raw = np.abs(d_raw[m].astype(np.float64) - v9d[m])
    med_new, med_raw = (float(np.median(e_new)) if m.any() else float("nan")), (float(np.median(e_raw)) if m.any() else float("nan"))
    return {"n_steps": int(m.sum()), "median_abs_err_" + sign_label: med_new, "median_abs_err_raw": med_raw, "pass": bool(m.any() and med_new <= med_raw)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--navtok", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--join-train", required=True)
    ap.add_argument("--join-eval", default="")
    ap.add_argument("--out", required=True, help="directory for navtext_nav.npz and navtext_nav_report.json")
    ap.add_argument("--nav-s", action="store_true", help="PREREG_V8F_CKPT120: build the S-side channel nav_s (step 5 with d_ann = infinity: the next TURN-class side at any distance) as navtext_nav_s.npz / navtext_nav_s_report.json; default = STRAT1's navtext_nav.npz")
    a = ap.parse_args(argv)
    from tanitad.data import v8f_mini_data as D
    tab = np.load(a.navtok, allow_pickle=False)
    tracks = D.load_native_tracks(a.tracks)
    out, rep = {}, {"inputs": {"navtok_sha256": sha_file(a.navtok), "tracks_sha256": sha_file(a.tracks), "join_train_sha256": sha_file(a.join_train)},
                    "rule": "BUILD_SPEC section 5 (countdown, d_ann = max(30, 6 v), UTURN none, bare text d = 0)" if not a.nav_s else "BUILD_SPEC section 5 with step 5 d_ann = infinity (PREREG_V8F_CKPT120 nav_s: countdown, no announcement cut, UTURN none, bare text d = 0)"}
    for split, jp in (("train", a.join_train), ("eval", a.join_eval)):
        if not jp:
            continue
        J = D.load_join(jp)
        for k in ("clip", "off", "n"):
            if not np.array_equal(np.asarray(tab[f"{split}_{k}"]).astype(str) if k == "clip" else tab[f"{split}_{k}"], np.asarray(J[k]).astype(str) if k == "clip" else J[k]):
                raise SystemExit(f"REFUSED: navtok {split}_{k} does not equal the join's (misaligned rows)")
        nav, nav_d, clamp, d_raw = build_split(J, tab, split, tracks, -1, not a.nav_s)
        plus = build_split(J, tab, split, tracks, +1, not a.nav_s)
        qa, qa_plus = clock_nav_qa(J, nav, nav_d, d_raw), clock_nav_qa(J, plus[0], plus[1], plus[3], "plus")
        ok = bool(qa["pass"] and not qa_plus["pass"])
        v9tok = np.where(J["nav"][:, 1] > 0, 1, np.where(J["nav"][:, 2] > 0, 2, 0))
        rep[split] = {"G-CLOCK-NAV": qa, "G-CLOCK-NAV_regression_arm_plus_travelled": qa_plus, "G-CLOCK-NAV_pass": ok, "n_rowclamp": int(clamp.sum()),
                      "n_steps_text_side_L_or_R": int(((nav[:, 1] + nav[:, 2]) > 0).sum()), "n_steps_text_ne_v9": int((nav != np.asarray(J["nav"])).any(1).sum())}
        out.update({f"{split}_clip": np.asarray(J["clip"]).astype(str), f"{split}_off": np.asarray(J["off"]), f"{split}_n": np.asarray(J["n"]), f"{split}_nav": nav, f"{split}_nav_d": nav_d, f"{split}_rowclamp": clamp})
        if not ok:
            print(f"G-CLOCK-NAV FAILED on {split}: {rep[split]}", file=sys.stderr)
    os.makedirs(a.out, exist_ok=True)
    out["meta_json"] = np.array(json.dumps({"rule": rep["rule"], "inputs": rep["inputs"], "left": LEFT, "right": RIGHT, "d_ann": [D_ANN_MIN, D_ANN_PER_MS] if not a.nav_s else "inf"}))
    stem = "navtext_nav_s" if a.nav_s else "navtext_nav"
    dst = os.path.join(a.out, stem + ".npz")
    np.savez_compressed(dst, **out)
    rep["navtext_nav_sha256"] = sha_file(dst)
    json.dump(rep, open(os.path.join(a.out, stem + "_report.json"), "w"), indent=1)
    print(json.dumps({k: rep[k] for k in rep if k in ("train", "eval", "navtext_nav_sha256")}, indent=1)[:2500])
    return 0 if all(rep[s]["G-CLOCK-NAV_pass"] for s in ("train", "eval") if s in rep) else 2


if __name__ == "__main__":
    sys.exit(main())
