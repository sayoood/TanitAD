"""make_crop_joins.py -- the eight joins of card PREREG_V8F_PROOF_ADD101 (CROPRULE10K). numpy and the standard library only (runs on the pod with /workspace/venv/bin/python).
Usage: python make_crop_joins.py --eval-join NPZ --dump-root DIR --out-dir NEWDIR   writes <ARM>_s<k>.npz for ARM in KB5 KB1 KD5 KD1, k in 0 1, and CROP_JOINS_RECORD.json
       python make_crop_joins.py --selftest                                         synthetic join and dumps; the clean build passes q0 and q1, each planted fault turns a check red

Derived from smoothgap/make_smooth_joins.py (card ADD97): the helpers valid_mask, frame_table, e0_stats, error_stats, clip_of_row, equal_key are copies of it (e0_stats has the further facts of q0).
A join of this card is the evaluation join with only `lead_gap` changed, and only on the MIX frames: frames with a valid label gap L (lead_valid == 1, finite, 0 <= L <= 60: the test of
LabelSource._ok in distrule/v8f_t1_eval_rule.py:88) AND a pilot output (dump `valid`, finite gap, inside the join's rows of its clip). On such a frame the join carries the pilot's gap AS DUMPED:
no q50 map, no minimum, no smoothing, no mean over seeds. A frame with a valid label gap and no pilot output keeps L (counted); every other frame is unchanged. The dtype of every key is kept.
Arms (card section 2) and the dump folder of each: KB5 B_H5, KB1 B_H1, KD5 D_H5, KD1 D_H1; the join of model seed k reads the dump reader_s<k>_score.npz of that folder (the pilot's fit seed k).
Dump format: cropread/fit/FIT_DEFS.md section 2 and cropread/fit/crop_dump.py:54-70 (dump_from_present): `cid` indexes the dump's own array `clips`, `clip_index` is the row of that clip in the
evaluation join (closeread/closeread_dump.py:165,180-184); the join row of a frame is off[index of the clip NAME in the join's `clip`] + frame. Refuses an existing out dir."""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD101"
GAP_MAX_M = 60.0
TOL_M = 1e-4
SEEDS = (0, 1)
ARMS = ("KB5", "KB1", "KD5", "KD1")
ARM_DIR = {"KB5": "B_H5", "KB1": "B_H1", "KD5": "D_H5", "KD1": "D_H1"}


def join_names():
    return ["%s_s%d" % (a, s) for a in ARMS for s in SEEDS]


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


def frame_table(J, D):
    """join row of every dump frame. -> dict(ji [rows] join clip index or -1, row [rows] join row or -1, inrange bool, names_not_in_join int, of [dump clips] join clip index or -1)"""
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
    return {"ji": ji, "row": row, "inrange": inrange, "names_not_in_join": int(sum(1 for c in dn if c not in idx)), "of": of}


def pilot_read_mask(D, tab):
    """a frame has a pilot output when its window is valid, its gap is finite and it lies inside the join's rows of its clip"""
    return np.asarray(D["valid"], bool) & np.isfinite(np.asarray(D["gap"], np.float64)) & tab["inrange"]


def e0_stats(J, D, tab, seed):
    """facts for control q0 (the dump is on the frames of the join): control e0 of card ADD96 as in make_smooth_joins.py, with the card's further words: every dump row has its join row,
    clip order (clip_index) agrees on every row, and the number of rows of every dump clip equals the number of frames of its join clip"""
    gap = np.asarray(J["lead_gap"], np.float64)
    vm = valid_mask(J)
    lab = np.asarray(D["lab_gap"], np.float64)
    has = np.isfinite(lab) & np.asarray(D["lab_present"], bool)
    inr = tab["inrange"]
    sel = has & inr
    diff = np.abs(lab[sel] - gap[tab["row"][sel]])
    key = np.stack([tab["ji"][inr], np.asarray(D["frame"], np.int64)[inr]], 1)
    n_dup = int(len(key) - len(np.unique(key, axis=0))) if len(key) else 0
    n = np.asarray(J["n"], np.int64)
    of = tab["of"]
    cnt = np.bincount(np.asarray(D["cid"], np.int64), minlength=len(of))
    known = of >= 0
    n_fc = int((cnt[known] != n[of[known]]).sum())
    has_ji = tab["ji"] >= 0
    n_not_in_dump = int(len(n) - len(np.unique(of[known])))
    return {"dump_seed": int(np.asarray(D["seed"])) if "seed" in D else None, "n_dump_rows": int(len(tab["row"])), "n_dump_clips": int(len(of)), "n_label_frames": int(has.sum()),
            "n_rows_without_join_row": int((~inr).sum()), "n_label_frames_without_join_row": int((has & ~inr).sum()), "n_label_frames_compared": int(sel.sum()),
            "max_abs_diff": float(diff.max()) if diff.size else None, "n_label_but_join_invalid": int((~vm[tab["row"][sel]]).sum()),
            "n_clip_index_mismatch": int((np.asarray(D["clip_index"], np.int64)[has_ji] != tab["ji"][has_ji]).sum()), "n_names_not_in_join": tab["names_not_in_join"],
            "n_duplicate_keys": n_dup, "n_frame_count_mismatch_clips": n_fc, "n_join_clips_not_in_dump": n_not_in_dump}


MIN_CLIP_FRAMES = 5


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
    off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
    c = np.full(len(J["lead_gap"]), -1, np.int64)
    for i in range(len(n)):
        c[off[i]:off[i] + n[i]] = i
    return c


# ------------------------------------------------------------------------------------------------ the build routine (whole arrays)
def pilot_rows(J, D, tab):
    """-> (g [join rows] float64 pilot gap as dumped, NaN where no output, out [join rows] bool)"""
    T = len(J["lead_gap"])
    m = pilot_read_mask(D, tab)
    g = np.full(T, np.nan)
    rd = np.zeros(T, bool)
    rows = tab["row"][m]
    g[rows] = np.asarray(D["gap"], np.float64)[m]
    rd[rows] = True
    return g, rd


def smooth_causal(g, mix, clip, a):
    """planted fault only: causal smoothing inside the runs of mix rows"""
    P = np.array(g, copy=True)
    prev = None
    last = -2
    for r in np.flatnonzero(mix):
        if prev is not None and r == last + 1 and clip[r] == clip[last]:
            P[r] = a * g[r] + (1.0 - a) * prev
        prev, last = P[r], r
    return P


FAULT_SWAP_HEAD = {"KB5": "KB1", "KB1": "KB5", "KD5": "KD1", "KD1": "KD5"}
FAULT_SWAP_KIND = {"KB5": "KD5", "KB1": "KD1", "KD5": "KB5", "KD1": "KB1"}


def processed(arm, s, G, R, J, vm, clip, faults=()):
    """-> (P [join rows], mix [join rows]) of one join. G, R: {(arm, seed): g rows, output rows}. The faults are planted for the self-test and the tests."""
    src = (arm, s)
    if "other_seed" in faults:
        src = (arm, 1 - s)
    if "other_head" in faults:
        src = (FAULT_SWAP_HEAD[arm], s)
    if "other_kind" in faults:
        src = (FAULT_SWAP_KIND[arm], s)
    g, read = G[src], R[src]
    mix = read & vm
    if "outside_mix" in faults:
        mix = read
    P = np.where(mix, g, np.nan)
    if "minimum" in faults:
        P = np.where(mix, np.minimum(g, np.asarray(J["lead_gap"], np.float64)), np.nan)
    if "smoothed" in faults:
        P = np.where(mix, smooth_causal(np.where(mix, g, 0.0), mix, clip, 0.5), np.nan)
    if "shifted" in faults:
        P = np.where(mix, np.roll(np.where(mix, g, 0.0), 1), np.nan)
    if "scaled" in faults:
        P = np.where(mix, g * 0.9, np.nan)
    return P, mix


# ------------------------------------------------------------------------------------------------ the check routine (plain loops, written separately)
def second_routine(J, D):
    """the processed gap of one join computed a second time from the dump alone, clip by clip and frame by frame. -> dict {join row: P}"""
    names = [str(c) for c in J["clip"]]
    off = {c: int(o) for c, o in zip(names, J["off"])}
    cnt = {c: int(k) for c, k in zip(names, J["n"])}
    lead, lv = J["lead_gap"], J["lead_valid"]
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
        reads[(c, f)] = r
    out = {}
    for c in names:
        for f in range(cnt[c]):
            row = off[c] + f
            L = float(lead[row])
            label_ok = int(lv[row]) == 1 and math.isfinite(L) and 0.0 <= L <= GAP_MAX_M
            if label_ok and (c, f) in reads:
                out[row] = reads[(c, f)]
    return out


def measure_file(path, J, D):
    """facts of one saved join, computed back from the file and from the dump (independent of how the file was built)"""
    with np.load(path, allow_pickle=False) as z:
        back = {kk: z[kk] for kk in z.files}
    exp = second_routine(J, D)
    rr = np.array(sorted(exp), np.int64)
    gapJ = np.asarray(J["lead_gap"], np.float64)
    saved = np.asarray(back["lead_gap"], np.float64)
    vm = valid_mask(J)
    ev = np.array([exp[r] for r in rr], np.float64)
    dev = np.abs(saved[rr] - ev) if rr.size else np.zeros(0)
    other = np.ones(len(gapJ), bool)
    other[rr] = False
    ch = (saved != gapJ) & ~(np.isnan(saved) & np.isnan(gapJ))
    noout = vm & other
    keys_eq = {kk: equal_key(back[kk], J[kk]) for kk in J if isinstance(J[kk], np.ndarray) and kk != "lead_gap" and kk in back}
    n_label = int(vm.sum())
    mixed = saved[rr]
    ji_row = clip_of_row(J)
    return {"n_valid_label_gap": n_label, "n_with_pilot_output": int(len(rr)), "n_without_pilot_output": int(n_label - len(rr)), "fraction_without_pilot_output": float((n_label - len(rr)) / max(n_label, 1)),
            "n_mixed_above_60": int((mixed > GAP_MAX_M).sum()), "n_mixed_below_0": int((mixed < 0.0).sum()), "min_mixed": float(mixed.min()) if mixed.size else None,
            "max_mixed": float(mixed.max()) if mixed.size else None,
            "max_abs_dev_saved_minus_second_routine": float(dev.max()) if dev.size else None,
            "n_label_frames_without_output_not_label_gap": int(ch[noout].sum()), "n_changed_outside_mixed_frames": int(ch[other].sum()), "n_frames_changed_in_file": int(ch.sum()),
            "keys_in_file_equal_to_input": bool(set(J_arrays(J)) <= set(back)), "other_keys_equal_to_input": keys_eq,
            "other_keys_all_equal": bool(all(keys_eq.values()) and len(keys_eq) == len(J_arrays(J)) - 1 and set(J_arrays(J)) <= set(back)),
            "lead_valid_unchanged": bool("lead_valid" in back and equal_key(back["lead_valid"], J["lead_valid"])),
            "lead_gap_dtype_kept": bool(back["lead_gap"].dtype == np.asarray(J["lead_gap"]).dtype),
            "error_statistics_pilot_minus_label": error_stats(saved[rr] - gapJ[rr], ji_row[rr], rr)}


def build(J, dumps, out_dir, faults=(), meta=None, eval_join_sha=None):
    """writes the eight joins and returns the record. J: the evaluation join as a dict of arrays; dumps: {(arm, seed): dict}; meta: {(arm, seed): {...}}"""
    if os.path.exists(out_dir):
        raise SystemExit("REFUSED: %s exists" % out_dir)
    os.makedirs(out_dir)
    meta = meta or {}
    rec = {"card": CARD, "eval_join_sha256": eval_join_sha, "n_clips": int(len(J["n"])), "n_rows": int(len(J["lead_gap"])), "n_valid_label_gap": int(valid_mask(J).sum()), "gap_max_m": GAP_MAX_M,
           "formula": "gap on the mix frames (valid label gap and a pilot output) = the pilot's gap as dumped (no map, no minimum, no smoothing, no mean); every other frame unchanged; "
                      "frames with a valid label gap and no pilot output keep the label gap and are counted",
           "arms_dump_dir": dict(ARM_DIR), "dumps": {}, "arms": {}, "faults_for_selftest": sorted(faults)}
    tabs, G, R = {}, {}, {}
    for arm in ARMS:
        for s in SEEDS:
            k = (arm, s)
            tabs[k] = frame_table(J, dumps[k])
            mm = meta.get(k, {})
            rec["dumps"]["%s_s%d" % k] = {"dump_file": mm.get("dump_file"), "dump_dir_name": mm.get("dump_dir_name"), "dump_sha256": mm.get("dump_sha256"), "e0": e0_stats(J, dumps[k], tabs[k], s)}
            G[k], R[k] = pilot_rows(J, dumps[k], tabs[k])
    dt = np.asarray(J["lead_gap"]).dtype
    gapJ = np.asarray(J["lead_gap"], np.float64)
    vm = valid_mask(J)
    clip = clip_of_row(J)
    for arm in ARMS:
        for s in SEEDS:
            name = "%s_s%d" % (arm, s)
            path = os.path.join(out_dir, name + ".npz")
            P, mix = processed(arm, s, G, R, J, vm, clip, faults)
            new = gapJ.copy()
            new[mix] = P[mix]
            if "no_output_zero" in faults:
                new[vm & ~R[(arm, s)]] = 0.0
            K = dict(J)
            K["lead_gap"] = new.astype(dt)
            if "other_field" in faults and name == "KB5_s0":
                K["lead_valid"] = np.asarray(J["lead_valid"]).copy()
                K["lead_valid"][0] = 1 - K["lead_valid"][0]
            np.savez(path, **{kk: x for kk, x in K.items() if isinstance(x, np.ndarray)})
            a = measure_file(path, J, dumps[(arm, s)])
            a.update({"arm": arm, "dump_dir": ARM_DIR[arm], "model_seed": s, "file": name + ".npz", "sha256": sha_file(path)})
            rec["arms"][name] = a
    finish(rec)
    return rec


def finish(rec, out_dir=None):
    rec["q0"] = q0_check(rec)
    rec["q1"] = q1_check(rec)
    if out_dir:
        with open(os.path.join(out_dir, "CROP_JOINS_RECORD.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rec, fh, indent=1)


def q0_check(rec):
    """control q0 from a record (card section 4): the pilot's dumps are on the frames of the evaluation join: every dump row has its join row, clip order (clip_index) and the number of
    frames of every clip agree, no key is twice in a dump, and the label gap of the dump equals the lead gap of the join on every label frame (largest difference at most 1e-4 m; the
    check of control e0 of card ADD96). All eight dumps."""
    bad = []
    try:
        if sorted(rec["dumps"]) != sorted(join_names()):
            bad.append("dumps in the record: %s" % sorted(rec["dumps"]))
        for nm in join_names():
            d = rec["dumps"].get(nm)
            if d is None:
                continue
            e, s = d["e0"], int(nm.rsplit("_s", 1)[1])
            if e.get("dump_seed") is not None and e["dump_seed"] != s:
                bad.append("%s: the dump records seed %s" % (nm, e.get("dump_seed")))
            if not e["n_label_frames"] > 0 or e["n_label_frames_compared"] != e["n_label_frames"]:
                bad.append("%s: %s label frames, %s compared" % (nm, e["n_label_frames"], e["n_label_frames_compared"]))
            x = e["max_abs_diff"]
            if x is None or not (x <= TOL_M):
                bad.append("%s: largest label gap difference %s (bar %g)" % (nm, x, TOL_M))
            for key in ("n_label_but_join_invalid", "n_clip_index_mismatch", "n_names_not_in_join", "n_duplicate_keys", "n_label_frames_without_join_row", "n_rows_without_join_row",
                        "n_frame_count_mismatch_clips"):
                if e[key] != 0:
                    bad.append("%s: %s = %s" % (nm, key, e[key]))
    except (KeyError, TypeError) as ex:
        bad.append("record incomplete: %s: %s" % (type(ex).__name__, ex))
    return {"PASS": not bad, "failures": bad}


def q1_check(rec):
    """control q1 (the joins) from a record: the saved gap of every join on every mix frame equals the gap computed a second time by a separate routine from the dump within 1e-4 m;
    frames with a valid label gap and no pilot output carry the label gap and are counted (no bar); every other field equals the evaluation join; the lead marking is unchanged;
    no frame outside the mix frames changed; the dump of every join is the folder of its arm and the file of its model seed."""
    bad = []
    try:
        if rec.get("card") != CARD:
            bad.append("record is for card %s" % rec.get("card"))
        want = {"%s_s%d" % (a, s): (a, s) for a in ARMS for s in SEEDS}
        if sorted(rec["arms"]) != sorted(want):
            bad.append("arms in the record: %s" % sorted(rec["arms"]))
        for nm, (arm, s) in want.items():
            a = rec["arms"].get(nm)
            if a is None:
                continue
            if a["arm"] != arm or a["model_seed"] != s or a.get("dump_dir") != ARM_DIR[arm]:
                bad.append("%s: arm, seed or dump folder in the record is not (%s, %d, %s)" % (nm, arm, s, ARM_DIR[arm]))
            d = (rec.get("dumps") or {}).get(nm) or {}
            if d.get("dump_file") is None or os.path.basename(str(d["dump_file"]).replace("\\", "/")) != "reader_s%d_score.npz" % s:
                bad.append("%s: dump file is %s, not reader_s%d_score.npz" % (nm, d.get("dump_file"), s))
            if d.get("dump_dir_name") != ARM_DIR[arm]:
                bad.append("%s: dump folder is %s, not %s" % (nm, d.get("dump_dir_name"), ARM_DIR[arm]))
            if not (isinstance(d.get("dump_sha256"), str) and len(d["dump_sha256"]) == 64):
                bad.append("%s: no sha256 of the dump" % nm)
            dev = a["max_abs_dev_saved_minus_second_routine"]
            if dev is None or not (dev <= TOL_M):
                bad.append("%s: largest deviation of the saved gap from the gap computed a second time is %s (bar %g)" % (nm, dev, TOL_M))
            if a["n_label_frames_without_output_not_label_gap"] != 0:
                bad.append("%s: %d frames with a valid label gap and no pilot output do not carry the label gap" % (nm, a["n_label_frames_without_output_not_label_gap"]))
            if not (isinstance(a.get("n_without_pilot_output"), int) and isinstance(a.get("fraction_without_pilot_output"), float)):
                bad.append("%s: the frames with a valid label gap and no pilot output are not counted" % nm)
            if a["other_keys_all_equal"] is not True or a["n_changed_outside_mixed_frames"] != 0:
                bad.append("%s: another field or a frame outside the mix frames differs from the evaluation join" % nm)
            if a.get("lead_valid_unchanged") is not True:
                bad.append("%s: the lead marking (lead_valid) is not the one of the evaluation join" % nm)
            for key in ("n_mixed_below_0", "n_mixed_above_60"):
                if not (isinstance(a.get(key), int) and a[key] >= 0):
                    bad.append("%s: no count %s in the record" % (nm, key))
            if a.get("lead_gap_dtype_kept") is not True:
                bad.append("%s: lead_gap changed its dtype" % nm)
            if not (isinstance(a.get("sha256"), str) and len(a["sha256"]) == 64):
                bad.append("%s: no sha256 of the file" % nm)
            if a["n_with_pilot_output"] <= 0:
                bad.append("%s: no mix frame" % nm)
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


def synthetic_dump(J, arm, seed, kind_seed=11):
    """a pilot dump of (arm, model seed): the present dump's rows (own clip order, every frame of every clip), `gap` the head's output in (0, 60), NaN where the head has no output"""
    ai = ARMS.index(arm)
    rng = np.random.default_rng(kind_seed * 1000 + ai * 10 + seed)
    names = [str(c) for c in J["clip"]]
    order = list(rng.permutation(len(names)))
    vm = valid_mask(J)
    gap = np.asarray(J["lead_gap"], np.float64)
    hist = 5 if arm.endswith("5") else 1
    cols = {k: [] for k in ("cid", "clip_index", "frame", "p", "gap", "clos", "valid", "lab_present", "lab_gap", "lab_closing", "lab_run")}
    for pos, ji in enumerate(order):
        o, n = int(J["off"][ji]), int(J["n"][ji])
        fr = np.arange(n)
        L = np.where(vm[o:o + n], gap[o:o + n], np.nan)
        r = np.clip(np.where(np.isfinite(L), L, rng.uniform(4.0, 40.0, n)) + rng.normal(0.5, 3.0, n), 0.5, 59.5)
        ok = (fr >= hist - 1) & (rng.random(n) > 0.004)
        for kk, v in (("cid", np.full(n, pos)), ("clip_index", np.full(n, ji)), ("frame", fr), ("p", np.ones(n)), ("gap", np.where(ok, r, np.nan)), ("clos", np.zeros(n)), ("valid", ok),
                      ("lab_present", np.isfinite(L)), ("lab_gap", L), ("lab_closing", np.zeros(n)), ("lab_run", np.zeros(n, np.int64))):
            cols[kk].append(v)
    D = {k: np.concatenate(v) for k, v in cols.items()}
    D["clips"] = np.asarray([names[i] for i in order])
    D["dt_s"] = np.float64(0.2)
    D["seed"] = np.int64(seed)
    return D


def synthetic_meta():
    return {(a, s): {"dump_file": "/workspace/v8proof/cropread/run1/dump/%s/reader_s%d_score.npz" % (ARM_DIR[a], s), "dump_dir_name": ARM_DIR[a], "dump_sha256": ("%d" % (ARMS.index(a) * 2 + s)) * 64}
            for a in ARMS for s in SEEDS}


def synthetic_all():
    J = synthetic_join()
    return J, {(a, s): synthetic_dump(J, a, s) for a in ARMS for s in SEEDS}


def selftest(here):
    base = os.path.join(here, "tmp_selftest_%d" % os.getpid())
    os.makedirs(base)
    J, dumps = synthetic_all()
    ok = True
    r = build(J, dumps, os.path.join(base, "clean"), meta=synthetic_meta())
    ok = ok and r["q0"]["PASS"] and r["q1"]["PASS"]
    print("SELFTEST clean build: q0 %s q1 %s %s" % (r["q0"]["PASS"], r["q1"]["PASS"], (r["q0"]["failures"] + r["q1"]["failures"])[:2]))
    for f in ("other_seed", "other_head", "other_kind", "minimum", "smoothed", "shifted", "scaled", "outside_mix", "no_output_zero", "other_field"):
        r = build(J, dumps, os.path.join(base, "f_" + f), faults=(f,), meta=synthetic_meta())
        caught = not r["q1"]["PASS"]
        ok = ok and caught and r["q0"]["PASS"]
        print("SELFTEST fault %-14s caught by q1: %s" % (f, caught))
    k0 = ("KD5", 1)

    def q0_fault(label, fn):
        nonlocal ok
        d = dict(dumps)
        d[k0] = fn(dict(dumps[k0]))
        r = build(J, d, os.path.join(base, "f_" + label), meta=synthetic_meta())
        caught = not r["q0"]["PASS"]
        ok = ok and caught
        print("SELFTEST fault %-14s caught by q0: %s" % (label, caught))

    def lab(D):
        lg = np.array(D["lab_gap"], copy=True)
        lg[np.flatnonzero(np.isfinite(lg))[5]] += 0.01
        D["lab_gap"] = lg
        return D

    def cidx(D):
        D["clip_index"] = (np.asarray(D["clip_index"]) + 1) % len(J["n"])
        return D

    def extra_row(D):
        for k in ("cid", "clip_index", "frame", "p", "gap", "clos", "valid", "lab_present", "lab_gap", "lab_closing", "lab_run"):
            a = np.asarray(D[k])
            D[k] = np.concatenate([a, a[-1:]])
        D["frame"][-1] = int(J["n"][int(D["clip_index"][-1])]) + 3                 # a row beyond the frames of its join clip
        return D

    def drop_frames(D):
        keep = np.ones(len(D["frame"]), bool)
        keep[7] = False
        keep[8] = False
        for k in ("cid", "clip_index", "frame", "p", "gap", "clos", "valid", "lab_present", "lab_gap", "lab_closing", "lab_run"):
            D[k] = np.asarray(D[k])[keep]
        return D

    def seed_(D):
        D["seed"] = np.int64(0)
        return D
    q0_fault("label_gap", lab)
    q0_fault("clip_index", cidx)
    q0_fault("row_beyond_join", extra_row)
    q0_fault("frame_count", drop_frames)
    q0_fault("dump_seed", seed_)
    again = None
    try:
        build(J, dumps, os.path.join(base, "clean"), meta=synthetic_meta())
    except SystemExit as ex:
        again = str(ex)
    ok = ok and again is not None and again.startswith("REFUSED")
    print("SELFTEST existing out dir refused: %s" % (again is not None))
    root = os.path.join(base, "dump")
    ej = os.path.join(base, "join_eval.npz")
    np.savez(ej, **J)
    for a in ARMS:
        os.makedirs(os.path.join(root, ARM_DIR[a]))
        for s in SEEDS:
            np.savez(os.path.join(root, ARM_DIR[a], "reader_s%d_score.npz" % s), **dumps[(a, s)])
    rc = main(["--eval-join", ej, "--dump-root", root, "--out-dir", os.path.join(base, "viafiles")])
    ok = ok and rc == 0
    print("SELFTEST via files rc %d" % rc)
    print("SELFTEST %s (temp folder %s)" % ("PASS" if ok else "FAIL", base))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--eval-join")
    ap.add_argument("--dump-root")
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest(os.path.dirname(os.path.abspath(__file__)))
    if not (a.eval_join and a.dump_root and a.out_dir):
        sys.exit("REFUSED: --eval-join, --dump-root and --out-dir are required")
    if os.path.exists(a.out_dir):
        sys.exit("REFUSED: %s exists" % a.out_dir)
    with np.load(a.eval_join, allow_pickle=False) as z:
        J = {k: z[k] for k in z.files}
    dumps, meta = {}, {}
    for arm in ARMS:
        for s in SEEDS:
            dp = os.path.join(a.dump_root, ARM_DIR[arm], "reader_s%d_score.npz" % s)
            with np.load(dp, allow_pickle=False) as z:
                dumps[(arm, s)] = {k: z[k] for k in z.files}
            meta[(arm, s)] = {"dump_file": os.path.abspath(dp).replace("\\", "/"), "dump_dir_name": ARM_DIR[arm], "dump_sha256": sha_file(dp)}
    rec = build(J, dumps, a.out_dir, meta=meta, eval_join_sha=sha_file(a.eval_join))
    finish(rec, a.out_dir)
    print("ZZ CROP-JOINS q0 %s q1 %s" % ("PASS" if rec["q0"]["PASS"] else "FAIL", "PASS" if rec["q1"]["PASS"] else "FAIL"))
    for f in rec["q0"]["failures"] + rec["q1"]["failures"]:
        print("   " + f)
    return 0 if rec["q0"]["PASS"] and rec["q1"]["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
