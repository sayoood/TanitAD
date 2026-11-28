"""make_part_joins.py -- the eight part joins of card PREREG_V8F_PROOF_ADD98 (ERRPART10K). numpy and the standard library only (runs on the pod with /workspace/venv/bin/python).
Usage: python make_part_joins.py --eval-join NPZ --dump-dir DIR --qmap-dir DIR --out-dir NEWDIR   writes M_s0 ... F_s1 as <ARM>_s<seed>.npz and PART_JOINS_RECORD.json
       python make_part_joins.py --selftest                                                        synthetic join and dump; the clean build passes q0 and q1, each planted fault turns a check red

Derived from smoothgap/make_smooth_joins.py (card ADD97): the helpers valid_mask, qmap_apply, frame_table, reader_read_mask, e0_stats and error_stats are copies of it.
A part join is the evaluation join with only `lead_gap` changed, and only on the MIX frames: frames with a valid label gap L (lead_valid == 1, finite, 0 <= L <= 60: the test of
LabelSource._ok in distrule/v8f_t1_eval_rule.py:88) AND a reader read (dump `valid`, finite raw gap, inside the join's rows of its clip) of the reader seed of the model seed. On such a frame
    g = min(r, Q(r))   with Q the q50 map QMAP_LR16_s<s>_q50.json of that reader seed (qmap_apply, closeread/closeread_fit.py:347),   e = g - L
Parts (card section 2):
    m  the mean of e over the mix frames of the clip                       r = e - m
    s  the mean of r over the mix frames at most 5 frames before and 5 frames after the frame (the frame itself included) inside the same RUN, the window cut at the run ends
    f  r - s
    a RUN is a set of neighbouring mix frames of one clip (same clip, join row + 1) in which the label gap changes by at most 3.0 m from one frame to the next.
Arms (P is the gap that the join carries on a mix frame):   M: P = L + m     W: P = L + r     S: P = L + s     F: P = L + f
Model seed s uses reader seed s; no join is shared between the seeds. No limit is applied; the values below 0 and above 60 are counted. A frame with a valid label gap and no reader read keeps L;
every other frame is unchanged. The dtype of every key is kept. Refuses an existing out dir.
Dump rows: `cid` indexes the dump's own array `clips`, `clip_index` is the row of that clip in the evaluation join (closeread/closeread_dump.py:165,180-184); the join row of a frame is
off[index of the clip NAME in the join's `clip`] + frame."""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD98"
GAP_MAX_M = 60.0
TOL_M = 1e-4
MAX_NO_READ_FRACTION = 0.05
MIN_CLIP_FRAMES = 5
HALF_WINDOW = 5                      # frames on each side of a frame in the window of the slow part (1 s on each side)
RUN_JUMP_M = 3.0                     # the largest change of the label gap between neighbouring frames inside one run
SEEDS = (0, 1)
# (arm, part that the join carries)
ARMS = (("M", "m"), ("W", "r"), ("S", "s"), ("F", "f"))
ARM_NAMES = tuple(x[0] for x in ARMS)
SPEC = {x[0]: x for x in ARMS}
FAULTS = ("mean_other_clip", "slow_across_runs", "swap_parts", "window_other", "other_seed", "other_field")


def join_names():
    return ["%s_s%d" % (a, s) for a, _x in ARMS for s in SEEDS]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def equal_key(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    return bool(np.array_equal(a, b, equal_nan=True)) if a.dtype.kind == "f" else bool(np.array_equal(a, b))


def valid_mask(J):
    g = np.asarray(J["lead_gap"], np.float64)
    return (np.asarray(J["lead_valid"]) == 1) & np.isfinite(g) & (g >= 0.0) & (g <= GAP_MAX_M)


def qmap_apply(g, xs, ys):
    """g_used = min(g, Q(g)) of qcal_map.apply_map (distrule/qcal_map.py:33); NaN stays NaN. Own copy of closeread_fit.qmap_apply; a test requires them equal."""
    g = np.asarray(g, np.float64)
    out = np.array(g, copy=True)
    fin = np.isfinite(g)
    gg = g[fin]
    q = np.interp(gg, xs, ys)
    lo = gg < xs[0]
    q[lo] = ys[0] * gg[lo] / xs[0]
    hi = gg > xs[-1]
    q[hi] = gg[hi] - (xs[-1] - ys[-1])
    out[fin] = np.minimum(gg, q)
    return out


def frame_table(J, D):
    """join row of every dump frame. -> dict(ji [rows] join clip index or -1, row [rows] join row or -1, inrange bool, names_not_in_join int)"""
    names = [str(c) for c in J["clip"]]
    idx = {c: i for i, c in enumerate(names)}
    dn = [str(c) for c in D["clips"]]
    of = np.array([idx.get(c, -1) for c in dn], np.int64)
    cid = np.asarray(D["cid"], np.int64)
    frame = np.asarray(D["frame"], np.int64)
    ji = of[cid]
    off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
    jc = np.where(ji >= 0, ji, 0)
    inrange = (ji >= 0) & (frame >= 0) & (frame < n[jc])
    row = np.where(inrange, off[jc] + frame, -1)
    return {"ji": ji, "row": row, "inrange": inrange, "names_not_in_join": int(sum(1 for c in dn if c not in idx))}


def reader_read_mask(D, tab):
    """a frame has a reader read when its window is valid, its raw gap is finite and it lies inside the join's rows of its clip"""
    return np.asarray(D["valid"], bool) & np.isfinite(np.asarray(D["gap"], np.float64)) & tab["inrange"]


def e0_stats(J, D, tab, reader_seed):
    """facts for control q0 (the dump is on the frames of the join); the same facts as control e0 of card ADD96"""
    gap = np.asarray(J["lead_gap"], np.float64)
    vm = valid_mask(J)
    lab = np.asarray(D["lab_gap"], np.float64)
    has = np.isfinite(lab) & np.asarray(D["lab_present"], bool)
    inr = tab["inrange"]
    sel = has & inr
    diff = np.abs(lab[sel] - gap[tab["row"][sel]])
    key = np.stack([tab["ji"][inr], np.asarray(D["frame"], np.int64)[inr]], 1)
    n_dup = int(len(key) - len(np.unique(key, axis=0))) if len(key) else 0
    return {"reader_seed": int(reader_seed), "dump_seed": int(np.asarray(D["seed"])) if "seed" in D else None, "n_dump_frames": int(len(tab["row"])), "n_label_frames": int(has.sum()),
            "n_label_frames_without_join_row": int((has & ~inr).sum()), "n_label_frames_compared": int(sel.sum()),
            "max_abs_diff": float(diff.max()) if diff.size else None, "n_label_but_join_invalid": int((~vm[tab["row"][sel]]).sum()),
            "n_clip_index_mismatch": int((np.asarray(D["clip_index"], np.int64)[inr] != tab["ji"][inr]).sum()), "n_names_not_in_join": tab["names_not_in_join"], "n_duplicate_keys": n_dup}


def error_stats(err, ji, row):
    """statistics of the error err = P - L on the mix frames, as row 154 defines them (kinclose/kinclose6.py:49-85): mean and sd over all frames;
    over the clips with at least 5 such frames: root mean square about the clip mean (fast), sd of the clip means (const), root mean square and sd of the change between neighbouring frames (step)."""
    out = {"n_frames": int(len(err)), "mean": None, "sd": None, "n_frames_in_clips_ge5": 0, "rms_about_clip_mean": None, "sd_of_clip_means": None, "step_rms": None, "step_sd": None, "n_step_pairs": 0}
    if len(err) == 0:
        return out
    err = np.asarray(err, np.float64)
    out["mean"], out["sd"] = float(err.mean()), float(err.std())
    o = np.argsort(row, kind="stable")
    e, c, r = err[o], np.asarray(ji)[o], np.asarray(row)[o]
    u, inv, cnt = np.unique(c, return_inverse=True, return_counts=True)
    keep = cnt[inv] >= MIN_CLIP_FRAMES
    if not keep.any():
        return out
    e, c, r, inv = e[keep], c[keep], r[keep], inv[keep]
    inv2 = np.unique(c, return_inverse=True)[1]
    cm = np.bincount(inv2, weights=e) / np.bincount(inv2)
    out["n_frames_in_clips_ge5"] = int(len(e))
    out["rms_about_clip_mean"] = float(np.sqrt(np.mean((e - cm[inv2]) ** 2)))
    w = cm[inv2]
    out["sd_of_clip_means"] = float(np.sqrt(max(np.mean(w ** 2) - np.mean(w) ** 2, 0.0)))
    adj = np.flatnonzero((np.diff(r) == 1) & (c[1:] == c[:-1]))
    de = e[adj + 1] - e[adj]
    out["n_step_pairs"] = int(len(adj))
    if len(adj):
        out["step_rms"], out["step_sd"] = float(np.sqrt(np.mean(de ** 2))), float(np.std(de))
    return out


def J_arrays(J):
    return [kk for kk in J if isinstance(J[kk], np.ndarray)]


def clip_of_row(J):
    """join clip index of every join row (-1 where no clip covers the row)"""
    off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
    c = np.full(len(J["lead_gap"]), -1, np.int64)
    for i in range(len(n)):
        c[off[i]:off[i] + n[i]] = i
    return c


# ------------------------------------------------------------------------------------------------ the build routine (vectorised across clips and runs)
def reader_rows(J, D, tab, qm):
    """-> (g [join rows] float64 mapped reader gap, NaN where no read, read [join rows] bool)"""
    T = len(J["lead_gap"])
    m = reader_read_mask(D, tab)
    g = np.full(T, np.nan)
    rd = np.zeros(T, bool)
    rows = tab["row"][m]
    g[rows] = qmap_apply(np.asarray(D["gap"], np.float64)[m], qm["xs"], qm["ys"])
    rd[rows] = True
    return g, rd


def run_structure(mix, clip, L, across=False):
    """positions of the mix rows, the first position of every run in them, and the length of every run. A run: same clip, row + 1, and a change of the label gap of at most RUN_JUMP_M.
    across=True is a planted fault: the label gap rule is not applied (a run is every stretch of neighbouring mix frames)."""
    idx = np.flatnonzero(mix)
    new = np.ones(idx.size, bool)
    if idx.size > 1:
        cont = (idx[1:] == idx[:-1] + 1) & (clip[idx[1:]] == clip[idx[:-1]])
        if not across:
            cont &= np.abs(L[idx[1:]] - L[idx[:-1]]) <= RUN_JUMP_M
        new[1:] = ~cont
    start = np.flatnonzero(new)
    length = np.diff(np.append(start, idx.size))
    return idx, start, length


def part_arrays(g, mix, clip, L, faults=()):
    """the four parts on the mix rows. g: mapped reader gap [rows], mix: bool [rows], clip: join clip index [rows], L: label gap of the join as float64 [rows].
    -> dict(idx [mix rows], start, length [runs], m, r, s, f: arrays over all rows, NaN outside the mix frames)"""
    T = len(g)
    idx, start, length = run_structure(mix, clip, L, across=("slow_across_runs" in faults))
    out = {k: np.full(T, np.nan) for k in ("m", "r", "s", "f")}
    out.update({"idx": idx, "start": start, "length": length})
    if idx.size == 0:
        return out
    e = g[idx] - L[idx]
    uc, inv = np.unique(clip[idx], return_inverse=True)
    cm = np.bincount(inv, weights=e) / np.bincount(inv)
    mm = cm[inv]
    if "mean_other_clip" in faults:
        mm = cm[(inv + 1) % len(uc)]
    r = e - mm
    half = 3 if "window_other" in faults else HALF_WINDOW
    cs = np.concatenate([[0.0], np.cumsum(r)])
    pos = np.arange(idx.size)
    rs = np.repeat(start, length)
    re_ = rs + np.repeat(length, length) - 1
    lo = np.maximum(pos - half, rs)
    hi = np.minimum(pos + half, re_)
    s = (cs[hi + 1] - cs[lo]) / (hi - lo + 1)
    out["m"][idx], out["r"][idx], out["s"][idx], out["f"][idx] = mm, r, s, r - s
    return out


def processed(spec, pa, Lf, faults=()):
    """-> P [join rows] of one join (NaN outside the mix frames). spec: (arm, part); pa: part_arrays"""
    arm, part = spec
    if "swap_parts" in faults:
        part = {"s": "f", "f": "s"}.get(part, part)
    P = np.full(len(Lf), np.nan)
    ix = pa["idx"]
    P[ix] = Lf[ix] + pa[part][ix]
    return P


# ------------------------------------------------------------------------------------------------ the check routine (plain loops, written separately)
def qmap_scalar(r, xs, ys):
    """one value of g = min(r, Q(r)), as plain Python floats"""
    if r < xs[0]:
        q = ys[0] * r / xs[0]
    elif r > xs[-1]:
        q = r - (xs[-1] - ys[-1])
    else:
        q = float(np.interp(r, xs, ys))
    return min(r, q)


def second_parts(J, D, qm, s):
    """the parts of every mix frame of model seed s computed a second time from the dump and the map, clip by clip and frame by frame with plain loops.
    -> (dict {join row: dict(g, L, e, m, r, s, f)}, list of run lengths)"""
    names = [str(c) for c in J["clip"]]
    off = {c: int(o) for c, o in zip(names, J["off"])}
    cnt = {c: int(k) for c, k in zip(names, J["n"])}
    lead, lv = J["lead_gap"], J["lead_valid"]
    xs = [float(x) for x in qm["xs"]]
    ys = [float(y) for y in qm["ys"]]
    reads = {}
    for i in range(len(D["frame"])):
        if not bool(D["valid"][i]):
            continue
        r = float(D["gap"][i])
        if not math.isfinite(r):
            continue
        c = str(D["clips"][int(D["cid"][i])])
        f = int(D["frame"][i])
        if c not in off or f < 0 or f >= cnt[c]:
            continue
        reads[(c, f)] = qmap_scalar(r, xs, ys)
    out, runs = {}, []
    for c in names:
        fr = []
        for f in range(cnt[c]):
            row = off[c] + f
            L = float(lead[row])
            if int(lv[row]) == 1 and math.isfinite(L) and 0.0 <= L <= GAP_MAX_M and (c, f) in reads:
                fr.append((f, row, reads[(c, f)], L))
        if not fr:
            continue
        es = [g - L for _f, _r, g, L in fr]
        m = sum(es) / len(es)
        rs = [e - m for e in es]
        run, cur = [], []
        for k, (f, row, g, L) in enumerate(fr):
            if cur and (f == fr[k - 1][0] + 1 and abs(L - fr[k - 1][3]) <= RUN_JUMP_M):
                cur.append(k)
            else:
                if cur:
                    run.append(cur)
                cur = [k]
        run.append(cur)
        for ks in run:
            runs.append(len(ks))
            for j, k in enumerate(ks):
                lo, hi = max(j - HALF_WINDOW, 0), min(j + HALF_WINDOW, len(ks) - 1)
                sv = sum(rs[ks[i]] for i in range(lo, hi + 1)) / (hi - lo + 1)
                f, row, g, L = fr[k]
                out[row] = {"g": g, "L": L, "e": es[k], "m": m, "r": rs[k], "s": sv, "f": rs[k] - sv}
    return out, runs


def measure_file(path, J, second, runs2, spec, pa):
    """facts of one saved join, computed back from the file and from the second routine (independent of how the file was built)"""
    with np.load(path, allow_pickle=False) as z:
        back = {kk: z[kk] for kk in z.files}
    part = spec[1]
    rr = np.array(sorted(second), np.int64)
    gapJ = np.asarray(J["lead_gap"], np.float64)
    saved = np.asarray(back["lead_gap"], np.float64)
    vm = valid_mask(J)
    ev = np.array([second[r]["L"] + second[r][part] for r in rr], np.float64)
    dev = np.abs(saved[rr] - ev) if rr.size else np.zeros(0)
    other = np.ones(len(gapJ), bool)
    other[rr] = False
    ch = (saved != gapJ) & ~(np.isnan(saved) & np.isnan(gapJ))
    noread = vm & other
    keys_eq = {kk: equal_key(back[kk], J[kk]) for kk in J if isinstance(J[kk], np.ndarray) and kk != "lead_gap" and kk in back}
    n_label = int(vm.sum())
    mixed = saved[rr]
    ji_row = clip_of_row(J)
    cm = {}
    for r, v in zip(rr, saved[rr] - gapJ[rr]):
        cm.setdefault(int(ji_row[r]), []).append(float(v))
    clip_mean_max = max((abs(sum(v) / len(v)) for v in cm.values()), default=0.0)
    return {"n_valid_label_gap": n_label, "n_with_reader_read": int(len(rr)), "n_without_reader_read": int(n_label - len(rr)), "fraction_without_reader_read": float((n_label - len(rr)) / max(n_label, 1)),
            "n_mixed_above_60": int((mixed > GAP_MAX_M).sum()), "n_mixed_below_0": int((mixed < 0.0).sum()), "min_mixed": float(mixed.min()) if mixed.size else None,
            "max_mixed": float(mixed.max()) if mixed.size else None,
            "max_abs_dev_saved_minus_processed_second_routine": float(dev.max()) if dev.size else None,
            "clip_mean_P_minus_L_max_abs": float(clip_mean_max), "n_clips_with_mix_frames": int(len(cm)),
            "n_runs": int(len(pa["length"])), "median_run_length": float(np.median(pa["length"])) if len(pa["length"]) else None,
            "n_runs_second_routine": int(len(runs2)), "median_run_length_second_routine": float(np.median(runs2)) if runs2 else None,
            "n_label_frames_without_read_not_label_gap": int(ch[noread].sum()), "n_changed_outside_mixed_frames": int(ch[other].sum()), "n_frames_changed_in_file": int(ch.sum()),
            "keys_in_file_equal_to_input": bool(set(J_arrays(J)) <= set(back)), "other_keys_equal_to_input": keys_eq,
            "other_keys_all_equal": bool(all(keys_eq.values()) and len(keys_eq) == len(J_arrays(J)) - 1 and set(J_arrays(J)) <= set(back)),
            "lead_gap_dtype_kept": bool(back["lead_gap"].dtype == np.asarray(J["lead_gap"]).dtype),
            "error_statistics_processed_minus_label": error_stats(saved[rr] - gapJ[rr], ji_row[rr], rr)}


def identity_facts(paths, J, second):
    """the two sum identities of control q1 on the SAVED float32 values of the four joins of one model seed, read back from the files:
    (P_M - L) + (P_W - L) = g - L and (P_S - L) + (P_F - L) = P_W - L, on every mix frame (g from the second routine)"""
    got = {}
    for arm, p in paths.items():
        with np.load(p, allow_pickle=False) as z:
            got[arm] = np.asarray(z["lead_gap"], np.float64)
    rr = np.array(sorted(second), np.int64)
    L = np.asarray(J["lead_gap"], np.float64)[rr]
    g = np.array([second[r]["g"] for r in rr], np.float64)
    d1 = np.abs((got["M"][rr] - L) + (got["W"][rr] - L) - (g - L))
    d2 = np.abs((got["S"][rr] - L) + (got["F"][rr] - L) - (got["W"][rr] - L))
    return {"n_frames": int(len(rr)), "max_abs_dev_M_plus_W_minus_g_minus_L": float(d1.max()) if d1.size else None,
            "max_abs_dev_S_plus_F_minus_W": float(d2.max()) if d2.size else None}


def build(J, dumps, qmaps, out_dir, faults=(), meta=None, eval_join_sha=None):
    """writes the eight joins and the record. J: the evaluation join as a dict of arrays; dumps, qmaps: {seed: dict}. -> record"""
    if os.path.exists(out_dir):
        raise SystemExit("REFUSED: %s exists" % out_dir)
    os.makedirs(out_dir)
    meta = meta or {}
    rec = {"card": CARD, "eval_join_sha256": eval_join_sha, "n_clips": int(len(J["n"])), "n_rows": int(len(J["lead_gap"])), "n_valid_label_gap": int(valid_mask(J).sum()), "gap_max_m": GAP_MAX_M,
           "window_half_frames": HALF_WINDOW if "window_other" not in faults else 3, "run_jump_max_m": RUN_JUMP_M,
           "formula": "P on the mix frames (valid label gap and a reader read of the reader seed of the model seed): e = g - L with g = min(r, Q(r)) of the q50 map; m clip mean of e; r = e - m; "
                      "s mean of r over the mix frames within 5 frames before and after (frame included) inside the same run (neighbouring mix frames of one clip, label gap change <= 3.0 m, window cut at the run ends); "
                      "f = r - s; M: L + m, W: L + r, S: L + s, F: L + f",
           "seeds": {}, "arms": {}, "identities": {}, "faults_for_selftest": sorted(faults)}
    tabs, G, R, sec = {}, {}, {}, {}
    for s in SEEDS:
        tabs[s] = frame_table(J, dumps[s])
        e0 = e0_stats(J, dumps[s], tabs[s], s)
        mm = meta.get(s, {})
        rec["seeds"][str(s)] = {"dump_file": mm.get("dump_file"), "dump_sha256": mm.get("dump_sha256"), "qmap_file": mm.get("qmap_file"), "qmap_sha256": mm.get("qmap_sha256"),
                                "qmap_dir": mm.get("qmap_dir"), "e0": e0}
        G[s], R[s] = reader_rows(J, dumps[s], tabs[s], qmaps[s])
        sec[s] = second_parts(J, dumps[s], qmaps[s], s)
    dt = np.asarray(J["lead_gap"]).dtype
    gapJ = np.asarray(J["lead_gap"], np.float64)
    vm = valid_mask(J)
    clip = clip_of_row(J)
    for s in SEEDS:
        sb = (1 - s) if "other_seed" in faults else s
        mix = R[sb] & vm
        pa = part_arrays(G[sb], mix, clip, gapJ, faults)
        paths = {}
        for spec in ARMS:
            arm = spec[0]
            name = "%s_s%d" % (arm, s)
            path = os.path.join(out_dir, name + ".npz")
            P = processed(spec, pa, gapJ, faults)
            new = gapJ.copy()
            new[mix] = P[mix]
            K = dict(J)
            K["lead_gap"] = new.astype(dt)
            if "other_field" in faults and arm == "S" and s == 0:
                K["lead_valid"] = np.asarray(J["lead_valid"]).copy()
                K["lead_valid"][0] = 1 - K["lead_valid"][0]
            np.savez(path, **{kk: x for kk, x in K.items() if isinstance(x, np.ndarray)})
            paths[arm] = path
            a = measure_file(path, J, sec[s][0], sec[s][1], spec, pa)
            a.update({"arm": arm, "part": spec[1], "reader_seeds": [s], "model_seed": s, "file": name + ".npz", "sha256": sha_file(path)})
            rec["arms"][name] = a
        rec["identities"][str(s)] = identity_facts(paths, J, sec[s][0])
    finish(rec)
    return rec


def finish(rec, out_dir=None):
    rec["q0"] = q0_check(rec)
    rec["q1"] = q1_check(rec)
    if out_dir:
        with open(os.path.join(out_dir, "PART_JOINS_RECORD.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rec, fh, indent=1)


def q0_check(rec):
    """control q0 from a record: control e0 of card ADD96 computed again for this build: on every dump frame that carries a label gap, that gap equals the lead gap of the evaluation join
    (largest absolute difference at most 1e-4 m), both reader seeds, with the further facts of e0"""
    bad = []
    try:
        for s in SEEDS:
            e = rec["seeds"][str(s)]["e0"]
            if e["reader_seed"] != s or (e.get("dump_seed") is not None and e["dump_seed"] != s):
                bad.append("seed %d: the dump records seed %s" % (s, e.get("dump_seed")))
            if not e["n_label_frames"] > 0 or e["n_label_frames_compared"] != e["n_label_frames"]:
                bad.append("seed %d: %s label frames, %s compared" % (s, e["n_label_frames"], e["n_label_frames_compared"]))
            d = e["max_abs_diff"]
            if d is None or not (d <= TOL_M):
                bad.append("seed %d: largest label gap difference %s (bar %g)" % (s, d, TOL_M))
            for key in ("n_label_but_join_invalid", "n_clip_index_mismatch", "n_names_not_in_join", "n_duplicate_keys", "n_label_frames_without_join_row"):
                if e[key] != 0:
                    bad.append("seed %d: %s = %s" % (s, key, e[key]))
    except (KeyError, TypeError) as ex:
        bad.append("record incomplete: %s: %s" % (type(ex).__name__, ex))
    return {"PASS": not bad, "failures": bad}


def q1_check(rec, expect_qmap_dir=None):
    """control q1 (part joins) from a record: for every join the saved gap on every mix frame equals the part computed a second time by a separate routine within 1e-4 m; on every mix frame the saved values
    obey (P_M - L) + (P_W - L) = g - L and (P_S - L) + (P_F - L) = P_W - L within 1e-4 m; the mean of P_W - L over the mix frames of every clip is 0 within 1e-4 m; frames with a valid label gap and no reader
    read carry the label gap and are at most 5 percent; every other field equals the evaluation join; the two maps are the q50 maps of the two reader seeds. The counts of P below 0 and above 60 are recorded,
    no pass criterion."""
    bad = []
    try:
        if rec.get("card") != CARD:
            bad.append("record is for card %s" % rec.get("card"))
        if rec.get("window_half_frames") != HALF_WINDOW or rec.get("run_jump_max_m") != RUN_JUMP_M:
            bad.append("window of %s frames on each side and run rule %s m in the record, expected %d and %g" % (rec.get("window_half_frames"), rec.get("run_jump_max_m"), HALF_WINDOW, RUN_JUMP_M))
        want = {"%s_s%d" % (a, s): (a, x, s) for a, x in ARMS for s in SEEDS}
        if sorted(rec["arms"]) != sorted(want):
            bad.append("arms in the record: %s" % sorted(rec["arms"]))
        for nm, (arm, part, s) in want.items():
            a = rec["arms"].get(nm)
            if a is None:
                continue
            if a["arm"] != arm or a["part"] != part or a["model_seed"] != s or a["reader_seeds"] != [s]:
                bad.append("%s: arm, part or seed in the record is not (%s, %s, %d)" % (nm, arm, part, s))
            dev = a["max_abs_dev_saved_minus_processed_second_routine"]
            if dev is None or not (dev <= TOL_M):
                bad.append("%s: largest deviation of the saved gap from the part computed a second time is %s (bar %g)" % (nm, dev, TOL_M))
            if a["n_runs"] != a["n_runs_second_routine"] or a["median_run_length"] != a["median_run_length_second_routine"]:
                bad.append("%s: %s runs (median length %s) in the build, %s (median length %s) in the second routine" % (nm, a["n_runs"], a["median_run_length"], a["n_runs_second_routine"],
                                                                                                                    a["median_run_length_second_routine"]))
            cmx = a["clip_mean_P_minus_L_max_abs"]
            if arm == "W" and (cmx is None or not (cmx <= TOL_M)):
                bad.append("%s: the largest clip mean of P minus L is %s (has to be 0 within %g)" % (nm, cmx, TOL_M))
            if not a["fraction_without_reader_read"] <= MAX_NO_READ_FRACTION:
                bad.append("%s: %.4f of the frames with a valid label gap have no reader read (bar %.2f)" % (nm, a["fraction_without_reader_read"], MAX_NO_READ_FRACTION))
            if a["n_label_frames_without_read_not_label_gap"] != 0:
                bad.append("%s: %d frames with a valid label gap and no reader read do not carry the label gap" % (nm, a["n_label_frames_without_read_not_label_gap"]))
            if a["other_keys_all_equal"] is not True or a["n_changed_outside_mixed_frames"] != 0:
                bad.append("%s: another field or a frame outside the mix frames differs from the evaluation join" % nm)
            for key in ("n_mixed_below_0", "n_mixed_above_60"):
                if not (isinstance(a.get(key), int) and a[key] >= 0):
                    bad.append("%s: no count %s in the record" % (nm, key))
            if a.get("lead_gap_dtype_kept") is not True:
                bad.append("%s: lead_gap changed its dtype" % nm)
            if not (isinstance(a.get("sha256"), str) and len(a["sha256"]) == 64):
                bad.append("%s: no sha256 of the file" % nm)
            if a["n_with_reader_read"] <= 0:
                bad.append("%s: no mix frame" % nm)
        shas = [(rec["arms"].get(nm) or {}).get("sha256") for nm in want]
        if len(set(shas)) != len(shas):
            bad.append("two of the eight join files are one file (equal sha256)")
        for s in SEEDS:
            idn = rec["identities"][str(s)]
            for key, word in (("max_abs_dev_M_plus_W_minus_g_minus_L", "(P_M - L) + (P_W - L) = g - L"), ("max_abs_dev_S_plus_F_minus_W", "(P_S - L) + (P_F - L) = P_W - L")):
                v = idn.get(key)
                if v is None or not (v <= TOL_M):
                    bad.append("seed %d: the sum identity %s deviates by %s (bar %g)" % (s, word, v, TOL_M))
            if not (isinstance(idn.get("n_frames"), int) and idn["n_frames"] > 0):
                bad.append("seed %d: no frames in the sum identities" % s)
            sd = rec["seeds"][str(s)]
            if sd.get("qmap_file") != "QMAP_LR16_s%d_q50.json" % s:
                bad.append("seed %d: map file is %s, not QMAP_LR16_s%d_q50.json" % (s, sd.get("qmap_file"), s))
            for kk in ("qmap_sha256", "dump_sha256"):
                if not (isinstance(sd.get(kk), str) and len(sd[kk]) == 64):
                    bad.append("seed %d: no %s in the record" % (s, kk))
            if expect_qmap_dir is not None and str(sd.get("qmap_dir", "")).replace("\\", "/").rstrip("/") != str(expect_qmap_dir).replace("\\", "/").rstrip("/"):
                bad.append("seed %d: the map folder is %s, expected %s" % (s, sd.get("qmap_dir"), expect_qmap_dir))
    except (KeyError, TypeError) as ex:
        bad.append("record incomplete: %s: %s" % (type(ex).__name__, ex))
    return {"PASS": not bad, "failures": bad}


# ------------------------------------------------------------------------------------------------ synthetic data for the self-test and the tests
def synthetic_join(seed=3, n_clips=20):
    """clips with a drifting label gap, small steps and now and then a jump of 9 m, so that runs of different lengths and run boundaries exist"""
    rng = np.random.default_rng(seed)
    n = rng.integers(100, 200, n_clips).astype(np.int64)
    off = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64)
    T = int(n.sum())
    valid = (rng.random(T) < 0.95).astype(np.int8)
    gap = np.zeros(T)
    for i in range(n_clips):
        k = int(n[i])
        st = rng.normal(0.0, 0.8, k)
        st[rng.random(k) < 0.04] += rng.choice([-9.0, 9.0])
        gap[off[i]:off[i] + k] = np.clip(20.0 + np.cumsum(st), 4.0, 40.0)
    gap = gap.astype(np.float32)
    gap[rng.random(T) < 0.02] = np.nan
    return {"clip": np.array(["c%02d_%s" % (i, "ab" * (i % 3 + 1)) for i in range(n_clips)]), "off": off, "n": n, "lead_valid": valid, "lead_gap": gap,
            "speed": rng.normal(10, 2, T).astype(np.float32), "has": np.ones(T, bool), "n_clips_missing_labels": np.asarray(0, np.int64)}


def synthetic_dump(J, reader_seed, seed=11):
    rng = np.random.default_rng(seed * 100 + reader_seed)
    names = [str(c) for c in J["clip"]]
    order = list(rng.permutation(len(names)))               # the dump's own clip order differs from the join's
    vm = valid_mask(J)
    gap = np.asarray(J["lead_gap"], np.float64)
    cols = {k: [] for k in ("cid", "clip_index", "frame", "p", "gap", "clos", "valid", "lab_present", "lab_gap", "lab_closing", "lab_run")}
    for pos, ji in enumerate(order):
        o, n = int(J["off"][ji]), int(J["n"][ji])
        fr = np.arange(n)
        L = np.where(vm[o:o + n], gap[o:o + n], np.nan)
        off_clip = rng.normal(0.0, 2.5)                     # an offset that stays with the clip
        drift = 2.0 * np.sin(fr / 12.0 + rng.uniform(0, 6.0))          # a slow part
        r = np.maximum(np.where(np.isfinite(L), L, rng.uniform(4.0, 40.0, n)) + 2.0 + reader_seed + off_clip + drift + rng.normal(0.0, 2.0, n), 1.0)
        r[rng.random(n) < 0.03] += 25.0
        ok = (fr >= 4) & (rng.random(n) > 0.004)
        for kk, v in (("cid", np.full(n, pos)), ("clip_index", np.full(n, ji)), ("frame", fr), ("p", rng.random(n)), ("gap", np.where(ok, r, np.nan)), ("clos", np.zeros(n)), ("valid", ok),
                      ("lab_present", np.isfinite(L)), ("lab_gap", L), ("lab_closing", np.zeros(n)), ("lab_run", np.zeros(n, np.int64))):
            cols[kk].append(v)
    D = {k: np.concatenate(v) for k, v in cols.items()}
    D["clips"] = np.asarray([names[i] for i in order])
    D["dt_s"] = np.float64(0.2)
    D["seed"] = np.int64(reader_seed)
    return D


def synthetic_qmap(reader_seed):
    return {"xs": np.array([2.0, 10.0, 30.0, 60.0]), "ys": np.array([1.5, 8.0 - reader_seed, 25.0 - reader_seed, 50.0 - reader_seed])}


def synthetic_meta():
    return {s: {"dump_file": "reader_s%d_score.npz" % s, "dump_sha256": "%d" % s * 64, "qmap_file": "QMAP_LR16_s%d_q50.json" % s, "qmap_sha256": "a%d" % s * 32, "qmap_dir": "/workspace/v8proof/integ10k/qmap_sweep"}
            for s in SEEDS}


def synthetic_all():
    J = synthetic_join()
    return J, {s: synthetic_dump(J, s) for s in SEEDS}, {s: synthetic_qmap(s) for s in SEEDS}


def selftest(here):
    base = os.path.join(here, "tmp_selftest_%d" % os.getpid())
    os.makedirs(base)
    J, dumps, qmaps = synthetic_all()
    ok = True
    r = build(J, dumps, qmaps, os.path.join(base, "clean"), meta=synthetic_meta())
    ok = ok and r["q0"]["PASS"] and r["q1"]["PASS"]
    print("SELFTEST clean build: q0 %s q1 %s %s" % (r["q0"]["PASS"], r["q1"]["PASS"], (r["q0"]["failures"] + r["q1"]["failures"])[:2]))
    for f in FAULTS:
        r = build(J, dumps, qmaps, os.path.join(base, "f_" + f), faults=(f,), meta=synthetic_meta())
        caught = not r["q1"]["PASS"]
        ok = ok and caught
        print("SELFTEST fault %-16s caught by q1: %s" % (f, caught))
    bad = dict(dumps[1])
    lg = np.array(bad["lab_gap"], copy=True)
    lg[np.flatnonzero(np.isfinite(lg))[5]] += 0.01
    bad["lab_gap"] = lg
    r = build(J, {0: dumps[0], 1: bad}, qmaps, os.path.join(base, "f_label"), meta=synthetic_meta())
    caught = not r["q0"]["PASS"]
    ok = ok and caught
    print("SELFTEST fault %-16s caught by q0: %s" % ("label_gap", caught))
    again = None
    try:
        build(J, dumps, qmaps, os.path.join(base, "clean"), meta=synthetic_meta())
    except SystemExit as ex:
        again = str(ex)
    ok = ok and again is not None and again.startswith("REFUSED")
    print("SELFTEST existing out dir refused: %s" % (again is not None))
    dd, qd = os.path.join(base, "dump"), os.path.join(base, "qmap")
    os.makedirs(dd)
    os.makedirs(qd)
    ej = os.path.join(base, "join_eval.npz")
    np.savez(ej, **J)
    for s in SEEDS:
        np.savez(os.path.join(dd, "reader_s%d_score.npz" % s), **dumps[s])
        with open(os.path.join(qd, "QMAP_LR16_s%d_q50.json" % s), "w", encoding="utf-8") as fh:
            json.dump({"xs": qmaps[s]["xs"].tolist(), "ys": qmaps[s]["ys"].tolist()}, fh)
    rc = main(["--eval-join", ej, "--dump-dir", dd, "--qmap-dir", qd, "--out-dir", os.path.join(base, "viafiles")])
    ok = ok and rc == 0
    print("SELFTEST via files rc %d" % rc)
    print("SELFTEST %s (temp folder %s)" % ("PASS" if ok else "FAIL", base))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--eval-join")
    ap.add_argument("--dump-dir")
    ap.add_argument("--qmap-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(os.path.dirname(os.path.abspath(__file__)))
    if not (a.eval_join and a.dump_dir and a.qmap_dir and a.out_dir):
        sys.exit("REFUSED: --eval-join, --dump-dir, --qmap-dir and --out-dir are required")
    if os.path.exists(a.out_dir):
        sys.exit("REFUSED: %s exists" % a.out_dir)
    with np.load(a.eval_join, allow_pickle=False) as z:
        J = {k: z[k] for k in z.files}
    dumps, qmaps, meta = {}, {}, {}
    for s in SEEDS:
        dp = os.path.join(a.dump_dir, "reader_s%d_score.npz" % s)
        qp = os.path.join(a.qmap_dir, "QMAP_LR16_s%d_q50.json" % s)
        with np.load(dp, allow_pickle=False) as z:
            dumps[s] = {k: z[k] for k in z.files}
        with open(qp, encoding="utf-8") as fh:
            q = json.load(fh)
        qmaps[s] = {"xs": np.asarray(q["xs"], np.float64), "ys": np.asarray(q["ys"], np.float64)}
        meta[s] = {"dump_file": dp, "dump_sha256": sha_file(dp), "qmap_file": os.path.basename(qp), "qmap_sha256": sha_file(qp), "qmap_dir": os.path.abspath(a.qmap_dir).replace("\\", "/")}
    rec = build(J, dumps, qmaps, a.out_dir, meta=meta, eval_join_sha=sha_file(a.eval_join))
    finish(rec, a.out_dir)
    print("ZZ PART-JOINS q0 %s q1 %s" % ("PASS" if rec["q0"]["PASS"] else "FAIL", "PASS" if rec["q1"]["PASS"] else "FAIL"))
    for f in rec["q0"]["failures"] + rec["q1"]["failures"]:
        print("   " + f)
    return 0 if rec["q0"]["PASS"] and rec["q1"]["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
