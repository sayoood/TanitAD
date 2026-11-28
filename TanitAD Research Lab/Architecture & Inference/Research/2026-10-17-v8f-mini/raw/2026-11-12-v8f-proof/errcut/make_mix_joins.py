"""make_mix_joins.py -- the eight mixed joins of card PREREG_V8F_PROOF_ADD96 (ERRCUT10K). numpy and the standard library only (runs on the pod with /workspace/venv/bin/python).
Usage: python make_mix_joins.py --eval-join NPZ --dump-dir DIR --qmap-dir DIR --out-dir NEWDIR   writes K100_s0 ... K12_s1 as <ARM>_s<seed>.npz and MIX_JOINS_RECORD.json
       python make_mix_joins.py --selftest                                                        synthetic join and dump; the clean build passes e0 and e1, each planted fault turns a check red

A mixed join is the evaluation join with only `lead_gap` changed, and only on the frames that have a valid label gap L (lead_valid == 1, finite, 0 <= L <= 60: the test of LabelSource._ok in
distrule/v8f_t1_eval_rule.py:88) AND a reader read in the per-frame dump of card ADD78 (reader seed s; window valid and finite raw gap r). On those frames
    g = min(r, Q(r))   with Q the q50 map QMAP_LR16_s<s>_q50.json of that reader seed, applied as qmap_apply (closeread/closeread_fit.py:347)
    M_k = L + k * (g - L)
Arms: K100 (k = 1), K50 (1/2), K25 (1/4), K12 (1/8). Model seed s uses the dump and the map of reader seed s. No limit is applied: a mixed value above 60 m stays and is counted.
Frames with a valid label gap and no reader read keep L; frames without a valid label gap are unchanged. The dtype of every key is kept (lead_gap stays as stored). Refuses an existing out dir.
Dump rows: `cid` indexes the dump's own array `clips`, `clip_index` is the row of that clip in the evaluation join (closeread/closeread_dump.py:165,180-184); the join row of a frame is
off[index of the clip NAME in the join's `clip`] + frame."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD96"
GAP_MAX_M = 60.0
TOL_M = 1e-4
MAX_NO_READ_FRACTION = 0.05
MIN_CLIP_FRAMES = 5
SEEDS = (0, 1)
ARMS = (("K100", 1.0), ("K50", 0.5), ("K25", 0.25), ("K12", 0.125))


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
    """facts for control e0 (the dump is on the frames of the join)"""
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
    """statistics of the planted error err = M_k - L on the frames with a reader read, as row 154 defines them (kinclose/kinclose6.py:49-85): mean and sd over all frames;
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


def mixed_values(J, D, tab, qm, k, raw_gap=False):
    """-> (mask of the frames that are mixed [dump rows], row [dump rows], L, g, M); all float64"""
    vm = valid_mask(J)
    gap = np.asarray(J["lead_gap"], np.float64)
    m = reader_read_mask(D, tab)
    m = m & vm[np.where(m, tab["row"], 0)]
    row = tab["row"]
    L = gap[np.where(m, row, 0)]
    r = np.asarray(D["gap"], np.float64)
    g = r if raw_gap else qmap_apply(r, qm["xs"], qm["ys"])
    M = L + k * (g - L)
    return m, row, L, g, M


def measure_file(path, J, D, qm, k):
    """facts of one saved join, computed back from the file and from the dump and map of the right reader seed (independent of how the file was built)"""
    with np.load(path, allow_pickle=False) as z:
        back = {kk: z[kk] for kk in z.files}
    tab = frame_table(J, D)
    m, row, L, g, _ = mixed_values(J, D, tab, qm, k)
    gapJ = np.asarray(J["lead_gap"], np.float64)
    saved = np.asarray(back["lead_gap"], np.float64)
    vm = valid_mask(J)
    rr = row[m]
    dev = np.abs((saved[rr] - gapJ[rr]) - k * (g[m] - L[m]))
    mixed_saved = saved[rr]
    other = np.ones(len(gapJ), bool)
    other[rr] = False
    ch = (saved != gapJ) & ~(np.isnan(saved) & np.isnan(gapJ))
    keys_eq = {kk: equal_key(back[kk], J[kk]) for kk in J if isinstance(J[kk], np.ndarray) and kk != "lead_gap" and kk in back}
    n_label = int(vm.sum())
    n_read_rows = len(np.unique(rr))
    return {"n_valid_label_gap": n_label, "n_with_reader_read": int(n_read_rows), "n_without_reader_read": int(n_label - n_read_rows),
            "fraction_without_reader_read": float((n_label - n_read_rows) / max(n_label, 1)), "n_mixed_above_60": int((mixed_saved > GAP_MAX_M).sum()),
            "min_mixed": float(mixed_saved.min()) if mixed_saved.size else None, "n_mixed_below_0": int((mixed_saved < 0.0).sum()), "max_abs_dev_mixed_minus_label_from_k_times_gap_minus_label": float(dev.max()) if dev.size else None,
            "n_changed_outside_mixed_frames": int(ch[other].sum()), "n_frames_changed_in_file": int(ch.sum()), "keys_in_file_equal_to_input": bool(set(J_arrays(J)) <= set(back)),
            "other_keys_equal_to_input": keys_eq, "other_keys_all_equal": bool(all(keys_eq.values()) and len(keys_eq) == len(J_arrays(J)) - 1 and set(J_arrays(J)) <= set(back)),
            "lead_gap_dtype_kept": bool(back["lead_gap"].dtype == np.asarray(J["lead_gap"]).dtype),
            "error_statistics_mixed_minus_label": error_stats(saved[rr] - gapJ[rr], tab["ji"][m], rr)}


def J_arrays(J):
    return [kk for kk in J if isinstance(J[kk], np.ndarray)]


def build(J, dumps, qmaps, out_dir, faults=(), meta=None, eval_join_sha=None):
    """writes the eight joins and the record. J: the evaluation join as a dict of arrays; dumps, qmaps: {seed: dict}. -> record"""
    if os.path.exists(out_dir):
        raise SystemExit("REFUSED: %s exists" % out_dir)
    os.makedirs(out_dir)
    meta = meta or {}
    rec = {"card": CARD, "eval_join_sha256": eval_join_sha, "n_clips": int(len(J["n"])), "n_rows": int(len(J["lead_gap"])), "n_valid_label_gap": int(valid_mask(J).sum()), "gap_max_m": GAP_MAX_M,
           "formula": "M_k = L + k * (g - L), g = min(r, Q(r)) with the q50 map of the reader seed; frames with a valid label gap and a reader read", "seeds": {}, "arms": {},
           "faults_for_selftest": sorted(faults)}
    tabs = {}
    for s in SEEDS:
        tabs[s] = frame_table(J, dumps[s])
        e0 = e0_stats(J, dumps[s], tabs[s], s)
        mm = meta.get(s, {})
        rec["seeds"][str(s)] = {"dump_file": mm.get("dump_file"), "dump_sha256": mm.get("dump_sha256"), "qmap_file": mm.get("qmap_file"), "qmap_sha256": mm.get("qmap_sha256"),
                                "qmap_dir": mm.get("qmap_dir"), "e0": e0}
    dt = np.asarray(J["lead_gap"]).dtype
    gapJ = np.asarray(J["lead_gap"], np.float64)
    for arm, k in ARMS:
        for s in SEEDS:
            sb = (1 - s) if "other_seed" in faults else s
            kb = k * 1.5 if "wrong_k" in faults else k
            m, row, L, g, M = mixed_values(J, dumps[sb], tabs[sb], qmaps[s], kb, raw_gap=("raw_gap" in faults))
            new = gapJ.copy()
            new[row[m]] = M[m]
            K = dict(J)
            K["lead_gap"] = new.astype(dt)
            if "other_field" in faults and arm == "K50" and s == 0:
                K["lead_valid"] = np.asarray(J["lead_valid"]).copy()
                K["lead_valid"][0] = 1 - K["lead_valid"][0]
            name = "%s_s%d" % (arm, s)
            path = os.path.join(out_dir, name + ".npz")
            np.savez(path, **{kk: x for kk, x in K.items() if isinstance(x, np.ndarray)})
            a = measure_file(path, J, dumps[s], qmaps[s], k)
            a.update({"arm": arm, "k": k, "reader_seed": s, "model_seed": s, "file": name + ".npz", "sha256": sha_file(path)})
            rec["arms"][name] = a
    finish(rec)
    return rec


def finish(rec, out_dir=None):
    rec["e0"] = e0_check(rec)
    rec["e1"] = e1_check(rec)
    if out_dir:
        with open(os.path.join(out_dir, "MIX_JOINS_RECORD.json"), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(rec, fh, indent=1)


def e0_check(rec):
    """control e0 from a record: on every dump frame that carries a label gap, that gap equals the lead gap of the evaluation join (largest absolute difference at most 1e-4 m), both reader seeds"""
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


def e1_check(rec, expect_qmap_dir=None):
    """control e1 (mixed joins) from a record"""
    bad = []
    try:
        if rec.get("card") != CARD:
            bad.append("record is for card %s" % rec.get("card"))
        want = {"%s_s%d" % (a, s): (a, k, s) for a, k in ARMS for s in SEEDS}
        if sorted(rec["arms"]) != sorted(want):
            bad.append("arms in the record: %s" % sorted(rec["arms"]))
        for nm, (arm, k, s) in want.items():
            a = rec["arms"].get(nm)
            if a is None:
                continue
            if a["k"] != k or a["arm"] != arm or a["reader_seed"] != s or a["model_seed"] != s:
                bad.append("%s: arm, k or seed in the record is not (%s, %g, %d)" % (nm, arm, k, s))
            dev = a["max_abs_dev_mixed_minus_label_from_k_times_gap_minus_label"]
            if dev is None or not (dev <= TOL_M):
                bad.append("%s: largest deviation of (M - L) from k * (g - L) is %s (bar %g)" % (nm, dev, TOL_M))
            if not a["fraction_without_reader_read"] <= MAX_NO_READ_FRACTION:
                bad.append("%s: %.4f of the frames with a valid label gap have no reader read (bar %.2f)" % (nm, a["fraction_without_reader_read"], MAX_NO_READ_FRACTION))
            if a["other_keys_all_equal"] is not True or a["n_changed_outside_mixed_frames"] != 0:
                bad.append("%s: another field or a frame outside the mixed frames differs from the evaluation join" % nm)
            if not (isinstance(a.get("n_mixed_below_0"), int) and a["n_mixed_below_0"] >= 0):          # note N1: counted and left as it is, no bar
                bad.append("%s: no count of the mixed values below 0 in the record" % nm)
            if a.get("lead_gap_dtype_kept") is not True:
                bad.append("%s: lead_gap changed its dtype" % nm)
            if not (isinstance(a.get("sha256"), str) and len(a["sha256"]) == 64):
                bad.append("%s: no sha256 of the file" % nm)
            if a["n_with_reader_read"] <= 0:
                bad.append("%s: no frame mixed" % nm)
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
        ok = (fr >= 4) & (rng.random(n) > 0.01)
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
    ok = ok and r["e0"]["PASS"] and r["e1"]["PASS"]
    print("SELFTEST clean build: e0 %s e1 %s %s" % (r["e0"]["PASS"], r["e1"]["PASS"], (r["e0"]["failures"] + r["e1"]["failures"])[:2]))
    for f, which in (("wrong_k", "e1"), ("raw_gap", "e1"), ("other_seed", "e1"), ("other_field", "e1")):
        r = build(J, dumps, qmaps, os.path.join(base, "f_" + f), faults=(f,), meta=synthetic_meta())
        caught = not r[which]["PASS"]
        ok = ok and caught
        print("SELFTEST fault %-12s caught by %s: %s" % (f, which, caught))
    bad = dict(dumps[1])
    lg = np.array(bad["lab_gap"], copy=True)
    lg[np.flatnonzero(np.isfinite(lg))[5]] += 0.01
    bad["lab_gap"] = lg
    r = build(J, {0: dumps[0], 1: bad}, qmaps, os.path.join(base, "f_label"), meta=synthetic_meta())
    caught = not r["e0"]["PASS"]
    ok = ok and caught
    print("SELFTEST fault %-12s caught by e0: %s" % ("label_gap", caught))
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
    print("ZZ MIX-JOINS e0 %s e1 %s" % ("PASS" if rec["e0"]["PASS"] else "FAIL", "PASS" if rec["e1"]["PASS"] else "FAIL"))
    for f in rec["e0"]["failures"] + rec["e1"]["failures"]:
        print("   " + f)
    return 0 if rec["e0"]["PASS"] and rec["e1"]["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
