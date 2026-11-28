"""make_smooth_joins.py -- the eight processed joins of card PREREG_V8F_PROOF_ADD97 (SMOOTHGAP10K). numpy and the standard library only (runs on the pod with /workspace/venv/bin/python).
Usage: python make_smooth_joins.py --eval-join NPZ --dump-dir DIR --qmap-dir DIR --out-dir NEWDIR   writes E_s0 ... ES25_s1 as <ARM>_s<seed>.npz and SMOOTH_JOINS_RECORD.json
       python make_smooth_joins.py --selftest                                                        synthetic join and dump; the clean build passes p0 and p1, each planted fault turns a check red

Derived from errcut/make_mix_joins.py (card ADD96): the helpers valid_mask, qmap_apply, frame_table, reader_read_mask, e0_stats and error_stats are copies of it.
A processed join is the evaluation join with only `lead_gap` changed, and only on the MIX frames: frames with a valid label gap L (lead_valid == 1, finite, 0 <= L <= 60: the test of
LabelSource._ok in distrule/v8f_t1_eval_rule.py:88) AND a reader read (dump `valid`, finite raw gap, inside the join's rows of its clip) of every reader seed the arm uses. On such a frame
    g_s = min(r_s, Q_s(r_s))   with Q_s the q50 map QMAP_LR16_s<s>_q50.json of reader seed s (qmap_apply, closeread/closeread_fit.py:347)
Arms (P is the processed gap that the join carries on a mix frame):
    E      P = (g_0 + g_1) / 2                                     one join for both model seeds (E_s0 equal to E_s1 in every array); a mix frame needs a read of both reader seeds
    S50_s  causal smoothing of g_s, a = 0.5                       model seed s uses reader seed s
    S25_s  causal smoothing of g_s, a = 0.25
    ES25   the same smoothing, a = 0.25, of (g_0 + g_1) / 2       one join for both model seeds; a mix frame needs a read of both reader seeds
Smoothing: over each RUN of neighbouring mix frames of a clip (same clip, frame number + 1), P_t = a * g_t + (1 - a) * P_(t-1), P = g at the first frame of the run. A run ends where the next
frame of the clip is no mix frame. No limit is applied; the values below 0 and above 60 are counted. A frame with a valid label gap and no reader read keeps L; every other frame is unchanged.
The dtype of every key is kept. Refuses an existing out dir.
Dump rows: `cid` indexes the dump's own array `clips`, `clip_index` is the row of that clip in the evaluation join (closeread/closeread_dump.py:165,180-184); the join row of a frame is
off[index of the clip NAME in the join's `clip`] + frame."""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD97"
GAP_MAX_M = 60.0
TOL_M = 1e-4
MAX_NO_READ_FRACTION = 0.05
MIN_CLIP_FRAMES = 5
CAUSAL_PROBE_M = 7.0
SEEDS = (0, 1)
# (arm, a, uses the mean of both reader seeds); a None: no smoothing
ARMS = (("E", None, True), ("S50", 0.5, False), ("S25", 0.25, False), ("ES25", 0.25, True))
ARM_NAMES = tuple(x[0] for x in ARMS)
SPEC = {x[0]: x for x in ARMS}
BOTH = ("E", "ES25")                       # arms with one join for both model seeds


def join_names():
    return ["%s_s%d" % (a, s) for a, _x, _y in ARMS for s in SEEDS]


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
    """facts for control p0 (the dump is on the frames of the join); the same facts as control e0 of card ADD96"""
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


# ------------------------------------------------------------------------------------------------ the build routine (vectorised across runs)
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


def run_structure(mix, clip):
    """positions of the mix rows, the first position of every run in them, and the length of every run. A run: same clip, row + 1."""
    idx = np.flatnonzero(mix)
    new = np.ones(idx.size, bool)
    if idx.size > 1:
        new[1:] = ~((idx[1:] == idx[:-1] + 1) & (clip[idx[1:]] == clip[idx[:-1]]))
    start = np.flatnonzero(new)
    length = np.diff(np.append(start, idx.size))
    return idx, start, length


def smooth_runs(g, mix, clip, a, across=False):
    """causal exponential smoothing of g over the runs of mix rows; -> P [rows], NaN outside mix. across=True is a planted fault: the recursion runs on across a frame with no read."""
    P = np.full(len(g), np.nan)
    idx = np.flatnonzero(mix)
    if idx.size == 0:
        return P
    if across:
        newr = np.ones(idx.size, bool)
        if idx.size > 1:
            newr[1:] = clip[idx[1:]] != clip[idx[:-1]]
        start = np.flatnonzero(newr)
        length = np.diff(np.append(start, idx.size))
    else:
        idx, start, length = run_structure(mix, clip)
    cur = g[idx[start]].copy()
    P[idx[start]] = cur
    for j in range(1, int(length.max())):
        live = np.flatnonzero(length > j)
        pos = start[live] + j
        cur[live] = a * g[idx[pos]] + (1.0 - a) * cur[live]
        P[idx[pos]] = cur[live]
    return P


def smooth_centred(g, mix, clip):
    """planted fault: a centred three-frame average inside the runs (uses the later frame)"""
    P = np.full(len(g), np.nan)
    idx, start, length = run_structure(mix, clip)
    for s0, ln in zip(start, length):
        rows = idx[s0:s0 + ln]
        v = g[rows]
        for i in range(ln):
            P[rows[i]] = float(np.mean(v[max(i - 1, 0):i + 2]))
    return P


def processed(spec, s, G, R, vm, clip, faults=()):
    """-> (P [join rows], mix [join rows]) of one join. G, R: {reader seed: g rows, read rows}"""
    arm, a, use_mean = spec
    if use_mean:
        g = 0.5 * (G[0] + G[1])
        read = R[0] & R[1]
        if "mean_twice" in faults:
            g = 0.5 * (G[0] + G[0])
            read = R[0] & R[1]
    else:
        sb = (1 - s) if "other_seed" in faults else s
        g = G[sb]
        read = R[sb]
    mix = read & vm
    if a is None:
        P = np.where(mix, g, np.nan)
    elif "centred" in faults:
        P = smooth_centred(g, mix, clip)
    else:
        a_used = a * 1.5 if "wrong_a" in faults else a
        P = smooth_runs(g, mix, clip, a_used, across=("run_across" in faults))
    return P, mix


def causal_probe(spec, s, G, R, vm, clip, faults=()):
    """control p1, causality: change the reader gap of the LAST mix frame of every run by +7.0 m, recompute the smoothed series, and record the change at the other mix frames of the runs (has to be 0)."""
    P, mix = processed(spec, s, G, R, vm, clip, faults)
    idx, start, length = run_structure(mix, clip)
    last = idx[start + length - 1] if idx.size else idx
    G2 = {k: v.copy() for k, v in G.items()}
    for k in G2:
        G2[k][last] = G2[k][last] + CAUSAL_PROBE_M
    P2, _ = processed(spec, s, G2, R, vm, clip, faults)
    isl = np.zeros(len(P), bool)
    isl[last] = True
    earlier = mix & ~isl
    d = np.abs(P2 - P)
    return {"causal_probe_m": CAUSAL_PROBE_M, "causal_n_runs": int(len(last)), "causal_n_earlier_frames_checked": int(earlier.sum()),
            "causal_max_abs_change_earlier_frames": float(d[earlier].max()) if earlier.any() else 0.0,
            "causal_last_frame_change_min": float(d[last].min()) if len(last) else None, "causal_last_frame_change_max": float(d[last].max()) if len(last) else None,
            "causal_last_frame_change_expected_min": (CAUSAL_PROBE_M if spec[1] is None else spec[1] * CAUSAL_PROBE_M)}


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


def second_routine(J, Ds, qms, spec, s):
    """the processed gap of one join computed a second time from the dumps and the maps, clip by clip and frame by frame. -> dict {join row: P}"""
    arm, a, use_mean = spec
    seeds = (0, 1) if use_mean else (s,)
    names = [str(c) for c in J["clip"]]
    off = {c: int(o) for c, o in zip(names, J["off"])}
    cnt = {c: int(k) for c, k in zip(names, J["n"])}
    lead, lv = J["lead_gap"], J["lead_valid"]
    reads = {}
    for sd in seeds:
        D = Ds[sd]
        xs = [float(x) for x in qms[sd]["xs"]]
        ys = [float(y) for y in qms[sd]["ys"]]
        d = {}
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
            d[(c, f)] = qmap_scalar(r, xs, ys)
        reads[sd] = d
    out = {}
    for c in names:
        prev = None
        for f in range(cnt[c]):
            row = off[c] + f
            L = float(lead[row])
            label_ok = int(lv[row]) == 1 and math.isfinite(L) and 0.0 <= L <= GAP_MAX_M
            if not (label_ok and all((c, f) in reads[sd] for sd in seeds)):
                prev = None
                continue
            g = (reads[0][(c, f)] + reads[1][(c, f)]) / 2.0 if use_mean else reads[s][(c, f)]
            p = g if (a is None or prev is None) else a * g + (1.0 - a) * prev
            prev = p
            out[row] = p
    return out


def measure_file(path, J, Ds, qms, spec, s):
    """facts of one saved join, computed back from the file and from the dumps and maps (independent of how the file was built)"""
    with np.load(path, allow_pickle=False) as z:
        back = {kk: z[kk] for kk in z.files}
    exp = second_routine(J, Ds, qms, spec, s)
    rr = np.array(sorted(exp), np.int64)
    gapJ = np.asarray(J["lead_gap"], np.float64)
    saved = np.asarray(back["lead_gap"], np.float64)
    vm = valid_mask(J)
    ev = np.array([exp[r] for r in rr], np.float64)
    dev = np.abs(saved[rr] - ev) if rr.size else np.zeros(0)
    other = np.ones(len(gapJ), bool)
    other[rr] = False
    ch = (saved != gapJ) & ~(np.isnan(saved) & np.isnan(gapJ))
    noread = vm & other
    keys_eq = {kk: equal_key(back[kk], J[kk]) for kk in J if isinstance(J[kk], np.ndarray) and kk != "lead_gap" and kk in back}
    n_label = int(vm.sum())
    mixed = saved[rr]
    ji_row = clip_of_row(J)
    return {"n_valid_label_gap": n_label, "n_with_reader_read": int(len(rr)), "n_without_reader_read": int(n_label - len(rr)), "fraction_without_reader_read": float((n_label - len(rr)) / max(n_label, 1)),
            "n_mixed_above_60": int((mixed > GAP_MAX_M).sum()), "n_mixed_below_0": int((mixed < 0.0).sum()), "min_mixed": float(mixed.min()) if mixed.size else None,
            "max_mixed": float(mixed.max()) if mixed.size else None,
            "max_abs_dev_saved_minus_processed_second_routine": float(dev.max()) if dev.size else None,
            "n_label_frames_without_read_not_label_gap": int(ch[noread].sum()), "n_changed_outside_mixed_frames": int(ch[other].sum()), "n_frames_changed_in_file": int(ch.sum()),
            "keys_in_file_equal_to_input": bool(set(J_arrays(J)) <= set(back)), "other_keys_equal_to_input": keys_eq,
            "other_keys_all_equal": bool(all(keys_eq.values()) and len(keys_eq) == len(J_arrays(J)) - 1 and set(J_arrays(J)) <= set(back)),
            "lead_gap_dtype_kept": bool(back["lead_gap"].dtype == np.asarray(J["lead_gap"]).dtype),
            "error_statistics_processed_minus_label": error_stats(saved[rr] - gapJ[rr], ji_row[rr], rr)}


def build(J, dumps, qmaps, out_dir, faults=(), meta=None, eval_join_sha=None):
    """writes the eight joins and the record. J: the evaluation join as a dict of arrays; dumps, qmaps: {seed: dict}. -> record"""
    if os.path.exists(out_dir):
        raise SystemExit("REFUSED: %s exists" % out_dir)
    os.makedirs(out_dir)
    meta = meta or {}
    rec = {"card": CARD, "eval_join_sha256": eval_join_sha, "n_clips": int(len(J["n"])), "n_rows": int(len(J["lead_gap"])), "n_valid_label_gap": int(valid_mask(J).sum()), "gap_max_m": GAP_MAX_M,
           "formula": "P on the mix frames (valid label gap and a reader read of every reader seed used): E (g0+g1)/2; S50_s, S25_s causal smoothing a=0.5, 0.25 of g_s; ES25 causal smoothing a=0.25 of (g0+g1)/2; "
                      "g_s = min(r_s, Q_s(r_s)) with the q50 map of reader seed s; restart at the first frame of every run of neighbouring mix frames",
           "seeds": {}, "arms": {}, "pairs_equal": {}, "faults_for_selftest": sorted(faults)}
    tabs, G, R = {}, {}, {}
    for s in SEEDS:
        tabs[s] = frame_table(J, dumps[s])
        e0 = e0_stats(J, dumps[s], tabs[s], s)
        mm = meta.get(s, {})
        rec["seeds"][str(s)] = {"dump_file": mm.get("dump_file"), "dump_sha256": mm.get("dump_sha256"), "qmap_file": mm.get("qmap_file"), "qmap_sha256": mm.get("qmap_sha256"),
                                "qmap_dir": mm.get("qmap_dir"), "e0": e0}
        G[s], R[s] = reader_rows(J, dumps[s], tabs[s], qmaps[s])
    dt = np.asarray(J["lead_gap"]).dtype
    gapJ = np.asarray(J["lead_gap"], np.float64)
    vm = valid_mask(J)
    clip = clip_of_row(J)
    for spec in ARMS:
        arm = spec[0]
        for s in SEEDS:
            name = "%s_s%d" % (arm, s)
            path = os.path.join(out_dir, name + ".npz")
            if arm in BOTH and s == 1:
                shutil.copyfile(os.path.join(out_dir, "%s_s0.npz" % arm), path)          # one join for both model seeds: the same bytes
            else:
                P, mix = processed(spec, s, G, R, vm, clip, faults)
                new = gapJ.copy()
                new[mix] = P[mix]
                K = dict(J)
                K["lead_gap"] = new.astype(dt)
                if "other_field" in faults and arm == "S50" and s == 0:
                    K["lead_valid"] = np.asarray(J["lead_valid"]).copy()
                    K["lead_valid"][0] = 1 - K["lead_valid"][0]
                np.savez(path, **{kk: x for kk, x in K.items() if isinstance(x, np.ndarray)})
            a = measure_file(path, J, dumps, qmaps, spec, s)
            a.update(causal_probe(spec, s, G, R, vm, clip, faults))
            a.update({"arm": arm, "a": spec[1], "mean_of_both_reader_seeds": bool(spec[2]), "reader_seeds": [0, 1] if spec[2] else [s], "model_seed": s, "file": name + ".npz", "sha256": sha_file(path)})
            rec["arms"][name] = a
    for arm in BOTH:
        with np.load(os.path.join(out_dir, "%s_s0.npz" % arm), allow_pickle=False) as z0, np.load(os.path.join(out_dir, "%s_s1.npz" % arm), allow_pickle=False) as z1:
            rec["pairs_equal"][arm] = bool(sorted(z0.files) == sorted(z1.files) and all(equal_key(z0[k], z1[k]) for k in z0.files))
    finish(rec)
    return rec


def finish(rec, out_dir=None):
    rec["p0"] = p0_check(rec)
    rec["p1"] = p1_check(rec)
    if out_dir:
        with open(os.path.join(out_dir, "SMOOTH_JOINS_RECORD.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rec, fh, indent=1)


def p0_check(rec):
    """control p0 from a record: control e0 of card ADD96 computed again for this build: on every dump frame that carries a label gap, that gap equals the lead gap of the evaluation join
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


def p1_check(rec, expect_qmap_dir=None):
    """control p1 (processed joins) from a record: for every join the saved gap on every mix frame equals the processed gap computed a second time by a separate routine within 1e-4 m;
    the smoothed value does not change when the reader gap of a later frame is changed (change at the earlier frames 0); frames with a valid label gap and no reader read carry the label gap and
    are at most 5 percent; every other field equals the evaluation join; the two maps are the q50 maps of the two reader seeds."""
    bad = []
    try:
        if rec.get("card") != CARD:
            bad.append("record is for card %s" % rec.get("card"))
        want = {"%s_s%d" % (a, s): (a, x, m, s) for a, x, m in ARMS for s in SEEDS}
        if sorted(rec["arms"]) != sorted(want):
            bad.append("arms in the record: %s" % sorted(rec["arms"]))
        for nm, (arm, aa, mean, s) in want.items():
            a = rec["arms"].get(nm)
            if a is None:
                continue
            if a["arm"] != arm or a["a"] != aa or a["mean_of_both_reader_seeds"] != mean or a["model_seed"] != s or a["reader_seeds"] != ([0, 1] if mean else [s]):
                bad.append("%s: arm, a, mean or seed in the record is not (%s, %s, %s, %d)" % (nm, arm, aa, mean, s))
            dev = a["max_abs_dev_saved_minus_processed_second_routine"]
            if dev is None or not (dev <= TOL_M):
                bad.append("%s: largest deviation of the saved gap from the processed gap computed a second time is %s (bar %g)" % (nm, dev, TOL_M))
            ce = a["causal_max_abs_change_earlier_frames"]
            if ce is None or not (ce == 0.0):
                bad.append("%s: the smoothed value at an earlier frame changes by %s when the gap of the last frame of a run is changed by %g m (has to be 0)" % (nm, ce, a["causal_probe_m"]))
            lo = a["causal_last_frame_change_min"]
            if a["causal_n_runs"] <= 0 or lo is None or not (lo >= a["causal_last_frame_change_expected_min"] - 1e-9):
                bad.append("%s: the causality probe did not reach the last frames (change %s, expected at least %s)" % (nm, lo, a["causal_last_frame_change_expected_min"]))
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
        for arm in BOTH:
            if rec["pairs_equal"].get(arm) is not True:
                bad.append("%s_s0 and %s_s1 are not equal in every array" % (arm, arm))
            x0, x1 = rec["arms"].get(arm + "_s0"), rec["arms"].get(arm + "_s1")
            if x0 and x1 and x0["sha256"] != x1["sha256"]:
                bad.append("%s_s0 and %s_s1 are not the same file" % (arm, arm))
        for s in SEEDS:
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
    rng = np.random.default_rng(seed)
    n = rng.integers(100, 200, n_clips).astype(np.int64)
    off = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64)
    T = int(n.sum())
    valid = (rng.random(T) < 0.9).astype(np.int8)
    gap = rng.uniform(4.0, 40.0, T).astype(np.float32)
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
        r = np.maximum(np.where(np.isfinite(L), L, rng.uniform(4.0, 40.0, n)) + rng.normal(2.0 + reader_seed, 3.0, n), 1.0)
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
    ok = ok and r["p0"]["PASS"] and r["p1"]["PASS"]
    print("SELFTEST clean build: p0 %s p1 %s %s" % (r["p0"]["PASS"], r["p1"]["PASS"], (r["p0"]["failures"] + r["p1"]["failures"])[:2]))
    for f in ("centred", "wrong_a", "mean_twice", "run_across", "other_field", "other_seed"):
        r = build(J, dumps, qmaps, os.path.join(base, "f_" + f), faults=(f,), meta=synthetic_meta())
        caught = not r["p1"]["PASS"]
        ok = ok and caught
        print("SELFTEST fault %-12s caught by p1: %s" % (f, caught))
    bad = dict(dumps[1])
    lg = np.array(bad["lab_gap"], copy=True)
    lg[np.flatnonzero(np.isfinite(lg))[5]] += 0.01
    bad["lab_gap"] = lg
    r = build(J, {0: dumps[0], 1: bad}, qmaps, os.path.join(base, "f_label"), meta=synthetic_meta())
    caught = not r["p0"]["PASS"]
    ok = ok and caught
    print("SELFTEST fault %-12s caught by p0: %s" % ("label_gap", caught))
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
    print("ZZ SMOOTH-JOINS p0 %s p1 %s" % ("PASS" if rec["p0"]["PASS"] else "FAIL", "PASS" if rec["p1"]["PASS"] else "FAIL"))
    for f in rec["p0"]["failures"] + rec["p1"]["failures"]:
        print("   " + f)
    return 0 if rec["p0"]["PASS"] and rec["p1"]["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
