"""make_scale_gapfiles.py -- the five gap files of card PREREG_V8F_PROOF_ADD106 (CROPSCALE10K) with controls g2 (the fits) and g3 (the gap files).
  VGAPL_s2, VGAPL_s3, VGAPL_s4  the dumped gap of the head on five frames (D-H5) of the fits on the large fit set, fit seeds 2, 3, 4 (build of card ADD104, make_seed_gapfile.py, kept)
  VGAPLE                        the arithmetic mean of those three dumped gaps (build of card ADD105, make_ens_gapfile.py, with three dumps instead of five), output only where all three have one
  VGAPSE3                       the same mean of the three 500-clip fits of seeds 2, 3, 4 (their dumps of card ADD104), output only where all three have one
Same array layout as the files of cards ADD104 and ADD105: clip_index (int64), frame (int64), gap (float64, metres, NaN where there is no output), has_out (bool), clips (the join's clip names), seed (int64: the fit seed
for VGAPL_s<S>, -1 for the two mean files). Nothing is mapped, clipped or combined with a minimum; the dump's `valid` field is not used. numpy and the standard library only (runs on the pod with /workspace/venv/bin/python).
Usage: python make_scale_gapfiles.py --eval-join NPZ --out-dir NEWDIR [--fit-root DIR] [--large-list FILE] [--pilot-list FILE] [--eval-list FILE] [--old-fits DIR] [--mid-root DIR] [--rec104 JSON] [--expect-clips 141]
         reads <fit-root>/fits/D_H5_s<S>/{FIT_RECORD.json,head.pt}, <fit-root>/dump/D_H5/reader_s<S>_{score.npz,meta.json} (S = 2, 3, 4; large fits), the lists, the records of the fits of seeds 0 and 1 in <old-fits>,
         the 500-clip fits of seeds 2, 3, 4 in <mid-root>/fits and <mid-root>/dump/D_H5 and the gap-file record of card ADD104; writes NEWDIR/VGAPL_s<S>.npz, VGAPLE.npz, VGAPSE3.npz and
         NEWDIR/CROPSCALE_GAPFILES_RECORD.json with g2 and g3. Refuses an existing out dir.
       python make_scale_gapfiles.py --selftest     synthetic world; the clean build passes g2 and g3, each planted fault of each clause turns its control red
Control g2 (card section 5), clause by clause, for each of the three large fits: (a) the record names seed S, arm D, head H5; (b) the settings of card ADD99 (record flag, equal to the constants, equal to seed 0's);
(c) no permuted labels (record flag and the arguments of the call; the arguments name arm D, head H5, seed S); (d) the fit clips of the record equal the clips with role fit of the large list, in order, the list hash
in the record is the hash of the list on disk, and the pilot's 450 fit clips are among them; (e) the selection clips of the record are the pilot's 50 selection clips (and the clips with role sel of the large list);
(f) no evaluation clip among the clips read, evaluation list hash as on disk and as in the record of seed 0; (g) the record's head hash is the hash of the head file on disk; (h) the three head files differ from one
another and from the five on record (seeds 0, 1 of card ADD99; the 500-clip fits of seeds 2, 3, 4 of card ADD104); (i) the large list starts with the pilot's 500 lines (an addition: it makes (d) and (e) checkable).
Control g3 (card section 5): v0 (the three dumps are on the frames of the evaluation join, as in card ADD104), meta (each dump's meta names the head file of its fit), build (each saved gap file equals a second routine on
every row within 1e-9 m), mean (the two mean files equal a second routine's mean within 1e-6 m and have an output exactly where all three dumps have one; the three dumps are on the same rows; the 500-clip dumps are
the ones the record of card ADD104 names, and that record reads x0 and x1 PASS)."""
import argparse
import copy
import hashlib
import json
import os
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD106"
FIT_CARD, FIT_ARM, FIT_HEAD, FIT_RECORD = "PREREG_V8F_PROOF_ADD99", "D", "H5", "FIT_RECORD.json"
CARD_HPARAMS = {"lr": 1e-3, "weight_decay": 0.05, "batch": 256, "epochs": 40, "warmup_epochs": 1, "beta_m": 1.0, "token_drop": 0.1, "d_model": 256, "n_heads": 4}      # cropread/fit/crop_fit.py:32
GAP_MAX_M = 60.0
TOL_M = 1e-4                                 # label gap of the dump against the join (v0, as in card ADD104)
TOL_SAVED = 1e-9                             # saved single gap file against the second routine (as in card ADD104)
TOL_MEAN_M = 1e-6                            # saved mean file against the second routine (card section 5, g3)
LARGE_SEEDS, MID_SEEDS, OLD_SEEDS = (2, 3, 4), (2, 3, 4), (0, 1)
N_PILOT, N_PILOT_SEL = 500, 50               # the pilot's 500 train clips, of them 50 selection clips (card section 2)
DUMP_DIR = "D_H5"
FIRST_OUTPUT_FRAME = 4
L_NAMES = ["VGAPL_s%d" % s for s in LARGE_SEEDS]
MEAN_L, MEAN_M = "VGAPLE", "VGAPSE3"
RECORD_NAME = "CROPSCALE_GAPFILES_RECORD.json"
WS = "/workspace/v8proof"
D_FIT_ROOT, D_LARGE_LIST = WS + "/cropscale/fitroot", WS + "/cropscale/lists/train_final.txt"
D_PILOT_LIST, D_EVAL_LIST = WS + "/cropread/run1/lists/train_final.txt", WS + "/cropread/run1/lists/eval141.txt"
D_OLD_FITS, D_MID_ROOT = WS + "/cropread/run1/fits", WS + "/visseed10k/fitroot"
D_REC104 = WS + "/visseed10k/gapfiles/VISSEED_GAPFILES_RECORD.json"
T0_TIER = "T0 (a probe on logged frames, a diagnostic of a reader; not driving performance)"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_text(lst):
    return hashlib.sha256("\n".join(lst).encode("utf-8")).hexdigest()


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


# ============================================================================================ the frame table and the single gap file (kept from make_seed_gapfile.py)
def frame_table(J, D):
    """join row of every dump row. -> dict(ji, row, inrange, names_not_in_join, of)"""
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


def _valid_mask(J):
    g = np.asarray(J["lead_gap"], np.float64)
    return (np.asarray(J["lead_valid"]) == 1) & np.isfinite(g) & (g >= 0.0) & (g <= GAP_MAX_M)


def e0_stats(J, D, tab, seed, expect_clips):
    """facts for control v0: the dump is on the frames of the evaluation join."""
    gap = np.asarray(J["lead_gap"], np.float64)
    lab = np.asarray(D["lab_gap"], np.float64)
    has = np.isfinite(lab) & np.asarray(D["lab_present"], bool)
    inr = tab["inrange"]
    sel = has & inr
    diff = np.abs(lab[sel] - gap[tab["row"][sel]])
    frame = np.asarray(D["frame"], np.int64)
    key = np.stack([tab["ji"][inr], frame[inr]], 1)
    n_dup = int(len(key) - len(np.unique(key, axis=0))) if len(key) else 0
    n = np.asarray(J["n"], np.int64)
    of = tab["of"]
    cnt = np.bincount(np.asarray(D["cid"], np.int64), minlength=len(of))
    known = of >= 0
    n_fc = int((cnt[known] != n[of[known]]).sum())
    has_ji = tab["ji"] >= 0
    jn = [str(c) for c in J["clip"]]
    dn = [str(c) for c in D["clips"]]
    return {"dump_seed": int(np.asarray(D["seed"])) if "seed" in D else None, "n_dump_rows": int(len(tab["row"])), "n_dump_clips": int(len(of)), "n_join_clips": int(len(jn)),
            "clips_same_order_as_join": bool(dn == jn), "expected_clips": int(expect_clips), "n_label_frames": int(has.sum()), "n_label_frames_compared": int(sel.sum()),
            "n_rows_without_join_row": int((~inr).sum()), "max_abs_diff_label_gap": float(diff.max()) if diff.size else None, "n_label_but_join_invalid": int((~_valid_mask(J)[tab["row"][sel]]).sum()),
            "n_clip_index_mismatch": int((np.asarray(D["clip_index"], np.int64)[has_ji] != tab["ji"][has_ji]).sum()), "n_names_not_in_join": tab["names_not_in_join"], "n_duplicate_keys": n_dup,
            "n_frame_count_mismatch_clips": n_fc, "n_join_rows": int(n.sum())}


def build_arrays(J, D, tab, seed, fault=None):
    """-> the arrays of the gap file. One row per join frame; gap from the dump row of that frame."""
    off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
    names = [str(c) for c in J["clip"]]
    nrow = int(n.sum())
    ci = np.repeat(np.arange(len(n), dtype=np.int64), n)
    fr = np.concatenate([np.arange(k, dtype=np.int64) for k in n]) if len(n) else np.zeros(0, np.int64)
    gap = np.full(nrow, np.nan)
    dg = np.asarray(D["gap"], np.float64)
    ok = tab["inrange"]
    gap[tab["row"][ok]] = dg[ok]
    if fault == "scale":
        gap = gap * 0.9
    elif fault == "shift":
        gap = np.concatenate([[np.nan], gap[:-1]])
    elif fault == "min_label":
        lg = np.asarray(J["lead_gap"], np.float64)
        gap = np.where(np.isfinite(gap) & np.isfinite(lg), np.minimum(gap, lg), gap)
    elif fault == "clip60":
        gap = np.where(np.isfinite(gap), np.clip(gap, 0.0, GAP_MAX_M), gap)
    elif fault == "zero_nooutput":
        gap = np.where(np.isfinite(gap), gap, 0.0)
    has = np.isfinite(gap)
    return {"clip_index": ci, "frame": fr, "gap": gap, "has_out": has, "clips": np.asarray(names), "seed": np.int64(seed)}


def second_routine(J, D):
    """a plain loop clip by clip and frame by frame from the dump alone -> {(join clip index, frame): gap or None}. Written apart from build_arrays."""
    where = {}
    for i, c in enumerate([str(c) for c in J["clip"]]):
        where[c] = i
    out = {}
    cl = [str(c) for c in D["clips"]]
    for r in range(len(D["cid"])):
        c = cl[int(D["cid"][r])]
        if c not in where:
            continue
        g = float(D["gap"][r])
        out[(where[c], int(D["frame"][r]))] = g if np.isfinite(g) else None
    return out


def _dump_valid_rows(J, D, arr):
    tab = frame_table(J, D)
    v = np.zeros(len(arr["clip_index"]), bool)
    ok = tab["inrange"]
    v[tab["row"][ok]] = np.asarray(D["valid"], bool)[ok]
    return v


def measure_file(J, D, arr, dump_path, dump_dir_name, file_path):
    """facts of the build check for one single gap file"""
    ref = second_routine(J, D)
    n = np.asarray(J["n"], np.int64)
    ci, fr, gap, has = arr["clip_index"], arr["frame"], arr["gap"], arr["has_out"]
    dev, n_out_mismatch, n_missing = 0.0, 0, 0
    for i in range(len(ci)):
        k = (int(ci[i]), int(fr[i]))
        if k not in ref:
            n_missing += 1
            continue
        r = ref[k]
        if r is None:
            n_out_mismatch += int(bool(has[i]) or np.isfinite(gap[i]))
        elif not has[i]:
            n_out_mismatch += 1
        else:
            dev = max(dev, abs(float(gap[i]) - r))
    keys = ci * (1 << 20) + fr
    fin = gap[has]
    return {"file": os.path.basename(file_path), "file_sha256": sha_file(file_path) if os.path.isfile(file_path) else None, "dump_file": dump_path, "dump_dir_name": dump_dir_name,
            "dump_sha256": sha_file(dump_path) if os.path.isfile(dump_path) else None, "n_rows": int(len(ci)), "n_join_rows": int(n.sum()), "n_with_output": int(has.sum()), "n_without_output": int((~has).sum()),
            "max_abs_dev_saved_minus_second_routine": dev, "n_rows_without_dump_row": n_missing, "n_output_flag_mismatch": n_out_mismatch, "keys_sorted_unique": bool(np.all(np.diff(keys) > 0)),
            "rows_cover_the_join_in_order": bool(len(ci) == int(n.sum()) and np.array_equal(ci, np.repeat(np.arange(len(n)), n))),
            "n_finite_but_dump_not_valid": int((has & ~_dump_valid_rows(J, D, arr)).sum()), "n_out_below_0": int((fin < 0.0).sum()), "n_out_above_60": int((fin > GAP_MAX_M).sum()),
            "min_out": float(fin.min()) if fin.size else None, "max_out": float(fin.max()) if fin.size else None}


# ============================================================================================ the mean files (build of card ADD105 with three dumps)
def on_join_rows(J, D, key):
    """the dump's field `key` laid on the join rows (NaN or False where the dump has no row)"""
    nrow = int(np.asarray(J["n"], np.int64).sum())
    tab = frame_table(J, D)
    ok = tab["inrange"]
    v = np.asarray(D[key])
    out = np.zeros(nrow, bool) if v.dtype == bool else np.full(nrow, np.nan)
    out[tab["row"][ok]] = v[ok]
    return out


def mean_arrays(J, dl, fault=None):
    """the arrays of a mean file: one row per join frame; gap = arithmetic mean of the dumped gaps of the dumps in dl where ALL are finite, NaN elsewhere."""
    n = np.asarray(J["n"], np.int64)
    names = [str(c) for c in J["clip"]]
    ci = np.repeat(np.arange(len(n), dtype=np.int64), n)
    fr = np.concatenate([np.arange(k, dtype=np.int64) for k in n]) if len(n) else np.zeros(0, np.int64)
    G = np.stack([on_join_rows(J, D, "gap").astype(np.float64) for D in dl], 0)
    allfin = np.isfinite(G).all(axis=0)
    gap = np.full(G.shape[1], np.nan)
    gap[allfin] = G[:, allfin].mean(axis=0)
    has = allfin.copy()
    if fault == "offset":
        gap = gap + 5e-6                                                           # 5e-6 m: above the bar of 1e-6
    elif fault == "mean_of_two":
        gap = np.where(allfin, G[:2].mean(axis=0), np.nan)
    elif fault == "median":
        gap = np.where(allfin, np.median(G, axis=0), np.nan)
    elif fault == "mean_of_available":
        anyfin = np.isfinite(G).any(axis=0)
        gap = np.where(anyfin, np.nanmean(np.where(np.isfinite(G), G, np.nan), axis=0), np.nan)
        has = np.isfinite(gap)
    elif fault == "flag_all_true":
        has = np.ones_like(allfin)
    elif fault == "zero_nooutput":
        gap = np.where(np.isfinite(gap), gap, 0.0)
    elif fault == "flag_false_on_a_row":
        has = allfin.copy()
        has[np.flatnonzero(allfin)[0]] = False
    return {"clip_index": ci, "frame": fr, "gap": gap, "has_out": has, "clips": np.asarray(names), "seed": np.int64(-1)}


def second_mean(J, dl):
    """plain loops over the dumps alone -> {(join clip index, frame): mean gap or None}; None where a dump has no row or no finite gap on that row. Written apart from mean_arrays."""
    where = {str(c): i for i, c in enumerate(J["clip"])}
    per = []
    for D in dl:
        cl = [str(c) for c in D["clips"]]
        d = {}
        for r in range(len(D["cid"])):
            c = cl[int(D["cid"][r])]
            if c not in where:
                continue
            g = float(D["gap"][r])
            d[(where[c], int(D["frame"][r]))] = g if np.isfinite(g) else None
        per.append(d)
    out = {}
    for i in range(len(J["clip"])):
        for f in range(int(J["n"][i])):
            v = [d.get((i, f)) for d in per]
            if all(x is not None for x in v):
                t = 0.0
                for x in v:
                    t += x
                out[(i, f)] = t / float(len(per))
            else:
                out[(i, f)] = None
    return out


def row_keys(D):
    cl = [str(c) for c in D["clips"]]
    return sorted((cl[int(c)], int(f)) for c, f in zip(D["cid"], D["frame"]))


def mean_facts(J, dl, arr):
    """the facts of the mean check for one mean file: the dumps on the same rows, the saved arrays against the second routine."""
    nj = {str(c): int(k) for c, k in zip(J["clip"], np.asarray(J["n"], np.int64))}
    base, sr = None, []
    for D in dl:
        ks = row_keys(D)
        base = ks if base is None else base
        sr.append({"n_rows": len(ks), "n_duplicate_rows": int(len(ks) - len(set(ks))), "n_rows_not_in_join": int(sum(1 for c, fr in ks if c not in nj or not (0 <= fr < nj[c]))),
                   "same_set_as_first": bool(len(ks) == len(base) and ks == base)})
    n = np.asarray(J["n"], np.int64)
    ref = second_mean(J, dl)
    ci, fr, gap, has = arr["clip_index"], arr["frame"], arr["gap"], arr["has_out"]
    dev, n_flag, n_missing = 0.0, 0, 0
    for i in range(len(ci)):
        k = (int(ci[i]), int(fr[i]))
        if k not in ref:
            n_missing += 1
            continue
        r = ref[k]
        if r is None:
            n_flag += int(bool(has[i]) or bool(np.isfinite(gap[i])))
        elif not has[i] or not np.isfinite(gap[i]):
            n_flag += 1
        else:
            dev = max(dev, abs(float(gap[i]) - r))
    saved = {"n_rows": int(len(ci)), "n_join_rows": int(n.sum()), "rows_cover_the_join_in_order": bool(len(ci) == int(n.sum()) and np.array_equal(ci, np.repeat(np.arange(len(n)), n))),
             "max_abs_dev_saved_minus_second_routine_m": float(dev), "n_rows_without_second_routine_row": n_missing, "n_output_flag_mismatch": n_flag, "n_with_output": int(has.sum()),
             "n_with_output_second_routine": int(sum(1 for v in ref.values() if v is not None)), "n_without_output": int((~has).sum()),
             "saved_gap_is_nan_exactly_where_flag_false": bool(np.array_equal(np.isfinite(gap), has))}
    return {"n_dumps": len(dl), "same_rows": sr, "saved": saved}


def mean_check(f):
    """the mean clauses of control g3 from the facts of mean_facts. -> {"PASS", "failures", "clauses"}"""
    bad, cl = [], {}
    try:
        sr = f["same_rows"]
        ok_rows = all(v["same_set_as_first"] and v["n_duplicate_rows"] == 0 and v["n_rows_not_in_join"] == 0 and v["n_rows"] == sr[0]["n_rows"] for v in sr)
        cl["the_dumps_are_on_the_same_rows"] = bool(ok_rows and f["n_dumps"] == 3)
        if not cl["the_dumps_are_on_the_same_rows"]:
            bad.append("the three dumps are not on the same rows (each once, each in the join): %s" % sr)
        sv = f["saved"]
        dev = sv["max_abs_dev_saved_minus_second_routine_m"]
        cl["saved_mean_equals_the_second_routine_within_1e-6_m"] = bool(dev is not None and dev <= TOL_MEAN_M and sv["n_rows_without_second_routine_row"] == 0)
        if not cl["saved_mean_equals_the_second_routine_within_1e-6_m"]:
            bad.append("saved mean: largest deviation from the second routine %s (bar %g), %s rows without a second-routine row" % (dev, TOL_MEAN_M, sv["n_rows_without_second_routine_row"]))
        cl["output_exactly_where_all_three_have_one"] = bool(sv["n_output_flag_mismatch"] == 0 and sv["saved_gap_is_nan_exactly_where_flag_false"] and sv["n_with_output"] == sv["n_with_output_second_routine"])
        if not cl["output_exactly_where_all_three_have_one"]:
            bad.append("output flag: %s rows differ from the dumps (%s with output saved, %s by the second routine)" % (sv["n_output_flag_mismatch"], sv["n_with_output"], sv["n_with_output_second_routine"]))
        if not (sv["rows_cover_the_join_in_order"] and sv["n_rows"] == sv["n_join_rows"]):
            bad.append("the rows of the saved file do not cover the join once in order")
        if not sv["n_with_output"] > 0:
            bad.append("no frame has an output of all three dumps")
    except (KeyError, TypeError, IndexError) as ex:
        bad.append("facts incomplete: %s: %s" % (type(ex).__name__, ex))
    return {"PASS": not bad, "failures": bad, "clauses": cl}


def mid_binding_check(F):
    """the 500-clip dumps behind VGAPSE3 are the ones the record of card ADD104 names (sha256), that record reads x0 and x1 PASS, and each dump's meta names the head file of its 500-clip fit."""
    bad = []
    try:
        r4 = F["rec104"]
        if not (r4["x0_PASS"] is True and r4["x1_PASS"] is True):
            bad.append("the record of card ADD104 does not read x0 PASS and x1 PASS: %s" % {k: r4[k] for k in ("x0_PASS", "x1_PASS")})
        for s in MID_SEEDS:
            d, named = F["mid_dumps"]["s%d" % s], r4["dump_sha256"].get("s%d" % s)
            if not (isinstance(named, str) and len(named) == 64 and named == d.get("dump_sha256")):
                bad.append("500-clip dump of seed %d: sha256 %s is not the one the record of card ADD104 names (%s)" % (s, str(d.get("dump_sha256"))[:16], str(named)[:16]))
            dm = d.get("dump_meta")
            hd = F["mid"]["s%d" % s].get("head_disk_sha256")
            if not (isinstance(dm, dict) and isinstance(hd, str) and dm.get("head_pt_sha256") == hd and dm.get("seed") == s and dm.get("dump_sha256") == d.get("dump_sha256")):
                bad.append("500-clip dump of seed %d: its meta does not name the head file of its fit, its seed and its own sha256" % s)
    except (KeyError, TypeError, AttributeError) as ex:
        bad.append("facts incomplete: %s: %s" % (type(ex).__name__, ex))
    return {"PASS": not bad, "failures": bad}


# ============================================================================================ g3 parts on the large dumps
def v0_check(rec):
    """control v0 (card ADD104 section 4, kept): the three large dumps are on the frames of the evaluation join."""
    bad = []
    try:
        if sorted(rec["dumps"]) != sorted(L_NAMES):
            bad.append("dumps in the record: %s" % sorted(rec["dumps"]))
        for nm in L_NAMES:
            d = rec["dumps"].get(nm)
            if d is None:
                continue
            e, s = d["e0"], int(nm.rsplit("_s", 1)[1])
            if e.get("dump_seed") is not None and e["dump_seed"] != s:
                bad.append("%s: the dump records seed %s" % (nm, e.get("dump_seed")))
            if e["n_join_clips"] != e["expected_clips"]:
                bad.append("%s: the join has %s clips, expected %s" % (nm, e["n_join_clips"], e["expected_clips"]))
            if e["n_dump_clips"] != e["n_join_clips"] or e["clips_same_order_as_join"] is not True:
                bad.append("%s: the dump's clips are not the join's clips in the same order (%s dump clips)" % (nm, e["n_dump_clips"]))
            if not e["n_label_frames"] > 0 or e["n_label_frames_compared"] != e["n_label_frames"]:
                bad.append("%s: %s label frames, %s compared" % (nm, e["n_label_frames"], e["n_label_frames_compared"]))
            x = e["max_abs_diff_label_gap"]
            if x is None or not (x <= TOL_M):
                bad.append("%s: largest label gap difference %s (bar %g)" % (nm, x, TOL_M))
            if e["n_dump_rows"] != e["n_join_rows"]:
                bad.append("%s: %s dump rows, %s join rows" % (nm, e["n_dump_rows"], e["n_join_rows"]))
            for key in ("n_label_but_join_invalid", "n_clip_index_mismatch", "n_names_not_in_join", "n_duplicate_keys", "n_rows_without_join_row", "n_frame_count_mismatch_clips"):
                if e[key] != 0:
                    bad.append("%s: %s = %s" % (nm, key, e[key]))
    except (KeyError, TypeError) as ex:
        bad.append("record incomplete: %s: %s" % (type(ex).__name__, ex))
    return {"PASS": not bad, "failures": bad}


def build_check(rec):
    """each saved single gap file equals the second routine within 1e-9 m on every row, the flag is the finiteness of the dumped gap, the rows cover the join once in order."""
    bad = []
    try:
        if rec.get("card") != CARD:
            bad.append("record is for card %s" % rec.get("card"))
        if sorted(k for k in rec["files"] if k in L_NAMES) != sorted(L_NAMES):
            bad.append("files in the record: %s" % sorted(rec["files"]))
        for nm in L_NAMES:
            a = rec["files"].get(nm)
            if a is None:
                continue
            s = int(nm.rsplit("_s", 1)[1])
            if a["fit_seed"] != s or a["dump_dir_name"] != DUMP_DIR or os.path.basename(str(a["dump_file"]).replace("\\", "/")) != "reader_s%d_score.npz" % s:
                bad.append("%s: dump is %s/%s, not %s/reader_s%d_score.npz" % (nm, a.get("dump_dir_name"), a.get("dump_file"), DUMP_DIR, s))
            for k in ("file_sha256", "dump_sha256"):
                if not (isinstance(a.get(k), str) and len(a[k]) == 64):
                    bad.append("%s: no %s" % (nm, k))
            dev = a["max_abs_dev_saved_minus_second_routine"]
            if dev is None or not (dev <= TOL_SAVED):
                bad.append("%s: largest deviation from the second routine %s (bar %g)" % (nm, dev, TOL_SAVED))
            if a["n_rows_without_dump_row"] != 0 or a["n_output_flag_mismatch"] != 0:
                bad.append("%s: %s rows without a dump row, %s output flags differ from the dumped gap" % (nm, a["n_rows_without_dump_row"], a["n_output_flag_mismatch"]))
            if a["keys_sorted_unique"] is not True or a["rows_cover_the_join_in_order"] is not True or a["n_rows"] != a["n_join_rows"]:
                bad.append("%s: the rows do not cover the join once in order" % nm)
            if not a["n_with_output"] > 0:
                bad.append("%s: no frame has an output" % nm)
    except (KeyError, TypeError) as ex:
        bad.append("record incomplete: %s: %s" % (type(ex).__name__, ex))
    return {"PASS": not bad, "failures": bad}


def meta_check(rec, F):
    """each large dump's meta file (crop_dump.py:103-106) names the sha256 of the head file of its fit (the file on disk), its fit seed, arm D, head H5, no permuted labels, and the sha256 of the dump file read."""
    bad = []
    try:
        for s in LARGE_SEEDS:
            nm = "VGAPL_s%d" % s
            dm = rec["dumps"][nm].get("dump_meta")
            f = F["large"]["s%d" % s]
            if not isinstance(dm, dict):
                bad.append("%s: no dump meta file" % nm)
                continue
            h = dm.get("head_pt_sha256")
            if not (isinstance(h, str) and len(h) == 64) or h != f.get("head_disk_sha256"):
                bad.append("%s: the dump meta names head %s, the file of the fit is %s" % (nm, str(h)[:16], str(f.get("head_disk_sha256"))[:16]))
            if dm.get("seed") != s or dm.get("arm") != FIT_ARM or dm.get("head") != FIT_HEAD or dm.get("permute_labels") is not False or dm.get("card") != FIT_CARD:
                bad.append("%s: the dump meta is for seed %s arm %s head %s" % (nm, dm.get("seed"), dm.get("arm"), dm.get("head")))
            if dm.get("dump_sha256") != rec["files"][nm]["dump_sha256"]:
                bad.append("%s: the dump meta names another dump file than the one read" % nm)
    except (KeyError, TypeError, AttributeError) as ex:
        bad.append("record incomplete: %s: %s" % (type(ex).__name__, ex))
    return {"PASS": not bad, "failures": bad}


def g3_check(rec, F):
    parts = {"v0": v0_check(rec), "meta": meta_check(rec, F), "build": build_check(rec), "mean_large": mean_check(rec["means"][MEAN_L]["facts"]),
             "mean_mid": mean_check(rec["means"][MEAN_M]["facts"]), "mid_binding": mid_binding_check(F)}
    bad = ["%s: %s" % (k, x) for k, v in parts.items() for x in v["failures"]]
    return {"PASS": all(v["PASS"] for v in parts.values()) and not bad, "failures": bad, "parts": {k: v["PASS"] for k, v in parts.items()}}


# ============================================================================================ g2 (the fits)
def _flag(argv, flag):
    return argv[argv.index(flag) + 1] if (flag in argv and argv.index(flag) + 1 < len(argv)) else None


def g2_check(F):
    """control g2 from the facts of collect(). -> {"PASS", "failures", "clauses"} with one entry per clause (a to i) per seed."""
    bad, cl = [], {}

    def no(clause, msg):
        cl[clause] = False
        bad.append("%s: %s" % (clause, msg))
    try:
        L = F["lists"]
        lg_ids = [c for c, _ in L["large"]]
        pil = [c for c, _ in L["pilot"]]
        pil_fit, pil_sel = [c for c, r in L["pilot"] if r == "fit"], [c for c, r in L["pilot"] if r == "sel"]
        lg_fit, lg_sel = [c for c, r in L["large"] if r == "fit"], [c for c, r in L["large"] if r == "sel"]
        eval_ids = set(L["eval_ids"])
        for k in ("a", "b", "c", "d", "e", "f", "g", "h", "i"):
            cl.setdefault(k, True)
        # (i) the large list starts with the pilot's 500 lines; the pilot has 450 fit and 50 selection clips
        if len(pil) != N_PILOT or len(pil_sel) != N_PILOT_SEL or L["large"][:N_PILOT] != L["pilot"]:
            no("i", "the large list does not start with the pilot's %d lines (pilot: %d lines, %d sel)" % (N_PILOT, len(pil), len(pil_sel)))
        if len(set(lg_ids)) != len(lg_ids) or set(lg_ids) & eval_ids or not eval_ids:
            no("f", "the large list repeats a clip or holds an evaluation clip, or there is no evaluation list")
        ref = F["old"]["s0"]
        heads = {}
        for kind, seeds in (("large", LARGE_SEEDS), ("mid", MID_SEEDS), ("old", OLD_SEEDS)):
            for s in seeds:
                f = F[kind]["s%d" % s]
                if "error" in f:
                    no("a" if kind == "large" else "h", "%s fit of seed %d: record unreadable: %s" % (kind, s, f["error"]))
                    continue
                nm = "%s fit seed %d" % (kind, s)
                rec_h, disk_h = f["head_pt_sha256_record"], f["head_disk_sha256"]
                if kind == "large":
                    if f["seed"] != s or isinstance(f["seed"], bool) or f["arm"] != FIT_ARM or f["head"] != FIT_HEAD:
                        no("a", "%s: the record names seed %s arm %s head %s" % (nm, f["seed"], f["arm"], f["head"]))
                    if f["card"] != FIT_CARD or f["hparams_equal_card"] is not True or f["hparams"] != CARD_HPARAMS or f["hparams"] != ref["hparams"]:
                        no("b", "%s: the settings are not those of card ADD99 (card %s, equal flag %s)" % (nm, f["card"], f["hparams_equal_card"]))
                    av = f.get("argv") or []
                    if f["permute_labels"] is not False or "--permute-labels" in av:
                        no("c", "%s: permuted labels (flag %s)" % (nm, f["permute_labels"]))
                    if not (_flag(av, "--arm") == FIT_ARM and _flag(av, "--head") == FIT_HEAD and _flag(av, "--seed") == str(s)):
                        no("c", "%s: the arguments of the call do not name arm D, head H5, seed %d" % (nm, s))
                    if f["fit_clips"] != lg_fit or not f["fit_clips"]:
                        no("d", "%s: the fit clips of the record (%d) are not the clips with role fit of the large list (%d), in order" % (nm, len(f["fit_clips"] or []), len(lg_fit)))
                    if not (isinstance(f["train_list_sha256"], str) and f["train_list_sha256"] == L["large_sha256"]):
                        no("d", "%s: the list hash of the record is not the hash of the large list on disk" % nm)
                    if not set(pil_fit) <= set(f["fit_clips"] or []):
                        no("d", "%s: the pilot's fit clips are not all among the fit clips" % nm)
                    if f["sel_clips"] != pil_sel or f["sel_clips"] != lg_sel or len(f["sel_clips"] or []) != N_PILOT_SEL:
                        no("e", "%s: the selection clips are not the pilot's %d selection clips" % (nm, N_PILOT_SEL))
                    got = set(f["fit_clips"] or []) | set(f["sel_clips"] or [])
                    if got & eval_ids:
                        no("f", "%s: %d evaluation clips among the clips read" % (nm, len(got & eval_ids)))
                    if len(got) != len(f["fit_clips"] or []) + len(f["sel_clips"] or []):
                        no("f", "%s: a clip is read twice (fit and selection together)" % nm)
                    if not (isinstance(f["eval_list_sha256"], str) and f["eval_list_sha256"] == L["eval_list_sha256"] == ref["eval_list_sha256"]):
                        no("f", "%s: the evaluation list hash of the record is not that of the list on disk and of seed 0" % nm)
                    if not (isinstance(rec_h, str) and len(rec_h) == 64) or disk_h != rec_h:
                        no("g", "%s: the head file on disk is %s, the record says %s" % (nm, str(disk_h)[:16] if disk_h else "missing", str(rec_h)[:16]))
                else:
                    if not (isinstance(rec_h, str) and len(rec_h) == 64):
                        no("h", "%s: the record has no head sha256" % nm)
                        continue
                    if disk_h is not None and disk_h != rec_h:
                        no("h", "%s: the head file on disk is not the one of its record" % nm)
                if isinstance(rec_h, str):
                    heads.setdefault(rec_h, []).append("%s%d" % (kind, s))
        dup = {h[:16]: v for h, v in heads.items() if len(v) > 1}
        if dup:
            no("h", "head files that are the same file: %s" % dup)
        if len(heads) != 8 and not dup:
            no("h", "%d distinct head files, expected 8 (three large, three 500-clip, two of card ADD99)" % len(heads))
    except (KeyError, TypeError, AttributeError, IndexError) as ex:
        bad.append("facts incomplete: %s: %s" % (type(ex).__name__, ex))
        cl["incomplete"] = False
    return {"PASS": not bad, "failures": bad, "clauses": cl}


# ============================================================================================ collecting the facts from disk
def fit_dir(root, s, suffix=""):
    return os.path.join(root, "%s_%s_s%d%s" % (FIT_ARM, FIT_HEAD, s, suffix))


def fit_facts(d):
    """facts of one fit from its FIT_RECORD.json (keys of cropread/fit/crop_fit.py:300-304) and the sha256 of its head.pt on disk. Unreadable -> {"error": ...}."""
    rp, hp = os.path.join(d, FIT_RECORD), os.path.join(d, "head.pt")
    try:
        with open(rp, encoding="utf-8") as fh:
            r = json.load(fh)
        cr = r.get("clips_read") or {}
        return {"dir": d, "seed": r.get("seed"), "card": r.get("card"), "arm": r.get("arm"), "head": r.get("head"), "permute_labels": r.get("permute_labels"), "argv": r.get("argv"),
                "hparams": r.get("hparams"), "hparams_equal_card": r.get("hparams_equal_card"), "train_list_sha256": r.get("train_list_sha256"), "eval_list_sha256": r.get("eval_list_sha256"),
                "fit_clips": cr.get("fit"), "sel_clips": cr.get("sel"), "n_samples": r.get("n_samples"), "head_pt_sha256_record": r.get("head_pt_sha256"),
                "head_disk_sha256": sha_file(hp) if os.path.isfile(hp) else None, "tool_sha256": r.get("tool_sha256"), "kept_epoch": r.get("kept_epoch"), "kept_sel_loss": r.get("kept_sel_loss"),
                "record_sha256": sha_file(rp)}
    except (OSError, ValueError, AttributeError) as ex:
        return {"dir": d, "error": "%s: %s" % (type(ex).__name__, ex)}


def read_list(p):
    """-> [[clip, role], ...] in file order (the format of cropread/fit/crop_fit.py:61, `<clip> fit|sel`); a malformed line is kept as [line, None] so that g2 sees it"""
    out = []
    with open(p, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if ln:
                q = ln.split()
                out.append([q[0], q[1]] if len(q) == 2 and q[1] in ("fit", "sel") else [ln, None])
    return out


def read_ids(p):
    with open(p, encoding="utf-8") as fh:
        return [ln.split()[0] for ln in fh if ln.strip()]


def read_json(p):
    try:
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def dump_facts(dump_root, s):
    mp = os.path.join(dump_root, DUMP_DIR, "reader_s%d_meta.json" % s)
    p = os.path.join(dump_root, DUMP_DIR, "reader_s%d_score.npz" % s)
    return {"dump_file": p, "dump_dir_name": DUMP_DIR, "dump_sha256": sha_file(p) if os.path.isfile(p) else None, "dump_meta": read_json(mp)}


def collect(P):
    """P: dict of paths (keys fit_root, large_list, pilot_list, eval_list, old_fits, mid_root, rec104). -> F, the facts every control reads."""
    L = {"large": read_list(P["large_list"]), "pilot": read_list(P["pilot_list"]), "eval_ids": read_ids(P["eval_list"]), "large_sha256": sha_file(P["large_list"]),
         "pilot_sha256": sha_file(P["pilot_list"]), "eval_list_sha256": sha_file(P["eval_list"])}
    r4 = read_json(P["rec104"]) or {}
    F = {"lists": L, "large": {"s%d" % s: fit_facts(fit_dir(os.path.join(P["fit_root"], "fits"), s)) for s in LARGE_SEEDS},
         "mid": {"s%d" % s: fit_facts(fit_dir(os.path.join(P["mid_root"], "fits"), s)) for s in MID_SEEDS},
         "old": {"s%d" % s: fit_facts(fit_dir(P["old_fits"], s)) for s in OLD_SEEDS},
         "large_dumps": {"s%d" % s: dump_facts(os.path.join(P["fit_root"], "dump"), s) for s in LARGE_SEEDS},
         "mid_dumps": {"s%d" % s: dump_facts(os.path.join(P["mid_root"], "dump"), s) for s in MID_SEEDS},
         "rec104": {"x0_PASS": (r4.get("x0") or {}).get("PASS") is True, "x1_PASS": (r4.get("x1") or {}).get("PASS") is True,
                    "dump_sha256": {"s%d" % s: (((r4.get("files") or {}).get("VGAPF_s%d" % s)) or {}).get("dump_sha256") for s in MID_SEEDS}, "found": bool(r4)}}
    return F


def light_fits(F):
    """the facts of the fits for the record: the clip lists replaced by counts and hashes"""
    out = {}
    for kind in ("large", "mid", "old"):
        out[kind] = {}
        for k, f in F[kind].items():
            g = {a: b for a, b in f.items() if a not in ("fit_clips", "sel_clips")}
            g["n_fit_clips"] = len(f["fit_clips"]) if f.get("fit_clips") is not None else None
            g["n_sel_clips"] = len(f["sel_clips"]) if f.get("sel_clips") is not None else None
            g["fit_clips_sha256"] = sha_text(f["fit_clips"]) if f.get("fit_clips") is not None else None
            g["sel_clips_sha256"] = sha_text(f["sel_clips"]) if f.get("sel_clips") is not None else None
            out[kind][k] = g
    L = F["lists"]
    out["lists"] = {"large_sha256": L["large_sha256"], "pilot_sha256": L["pilot_sha256"], "eval_list_sha256": L["eval_list_sha256"], "n_large": len(L["large"]), "n_large_fit": sum(1 for _, r in L["large"] if r == "fit"),
                    "n_large_sel": sum(1 for _, r in L["large"] if r == "sel"), "n_pilot": len(L["pilot"]), "n_eval": len(L["eval_ids"])}
    out["mid_dumps"], out["large_dumps"], out["rec104"] = F["mid_dumps"], F["large_dumps"], F["rec104"]
    return out


def fit_facts_for_reader(F):
    """the per-fit facts the reader prints with no name: kept epoch, selection loss, number of fit clips"""
    out = {}
    for kind, tag in (("large", "L"), ("mid", "M"), ("old", "O")):
        for k, f in F[kind].items():
            out["%s%s" % (tag, k[1:])] = {"fit_seed": f.get("seed"), "kept_epoch": f.get("kept_epoch"), "kept_sel_loss": f.get("kept_sel_loss"), "n_fit_clips": len(f.get("fit_clips") or []),
                                          "n_sel_clips": len(f.get("sel_clips") or []), "n_samples": f.get("n_samples"), "head_pt_sha256": f.get("head_pt_sha256_record")}
    return out


def t0_diagnostic(J, DL, DM):
    """a diagnostic (the card's read is the reader's): mean absolute gap error on the rows with a label lead and an output of all six dumps (three large, three 500-clip)."""
    dl = list(DL) + list(DM)
    G = np.stack([on_join_rows(J, D, "gap").astype(np.float64) for D in dl], 0)
    lab = on_join_rows(J, dl[0], "lab_gap").astype(np.float64)
    labp = on_join_rows(J, dl[0], "lab_present")
    sel = labp & np.isfinite(lab) & np.isfinite(G).all(axis=0)
    out = {"tier": T0_TIER, "n_rows_with_label_lead_and_output_of_all_six": int(sel.sum()), "mean_abs_gap_error_m": None}
    if sel.any():
        e = np.abs(G[:, sel] - lab[sel]).mean(axis=1)
        out["mean_abs_gap_error_m"] = {"L%d" % s: float(e[i]) for i, s in enumerate(LARGE_SEEDS)}
        out["mean_abs_gap_error_m"].update({"M%d" % s: float(e[3 + i]) for i, s in enumerate(MID_SEEDS)})
        out["ratio_of_means_large_over_500"] = float(e[:3].mean() / e[3:].mean())
    return out


# ============================================================================================ build
def build(J, DL, DM, F, out_dir, fault=None, eval_join_sha=None, expect_clips=None, inputs=None):
    """DL, DM: {seed: dump dict} of the large and of the 500-clip fits. Writes the five files and the record. -> record. fault: a planted fault of the build (selftest and tests)."""
    if os.path.exists(out_dir):
        raise SystemExit("REFUSED: %s exists" % out_dir)
    os.makedirs(out_dir)
    rec = {"card": CARD, "eval_join_sha256": eval_join_sha, "inputs": inputs or {}, "files": {}, "dumps": {}, "means": {}, "fits": light_fits(F)}
    sf = fault[2:] if fault and fault.startswith("s_") else None                 # "s_<name>": a planted fault of the single files; "m_<name>": of the mean files
    mf = fault[2:] if fault and fault.startswith("m_") else None
    for s in LARGE_SEEDS:
        nm = "VGAPL_s%d" % s
        D = DL[s]
        tab = frame_table(J, D)
        arr = build_arrays(J, D, tab, s, sf)
        fp = os.path.join(out_dir, nm + ".npz")
        np.savez(fp, **arr)
        m = F["large_dumps"]["s%d" % s]
        rec["dumps"][nm] = {"dump_file": m["dump_file"], "dump_dir_name": m["dump_dir_name"], "dump_meta": m["dump_meta"], "e0": e0_stats(J, D, tab, s, expect_clips if expect_clips is not None else len(J["clip"]))}
        f = measure_file(J, D, arr, m["dump_file"], m["dump_dir_name"], fp)
        f["fit_seed"] = s
        rec["files"][nm] = f
    for nm, dd, key in ((MEAN_L, DL, "large_dumps"), (MEAN_M, DM, "mid_dumps")):
        dl = [dd[s] for s in LARGE_SEEDS]
        arr = mean_arrays(J, dl, mf)
        fp = os.path.join(out_dir, nm + ".npz")
        np.savez(fp, **arr)
        rec["means"][nm] = {"facts": mean_facts(J, dl, arr)}
        rec["files"][nm] = {"file": nm + ".npz", "file_sha256": sha_file(fp), "n_rows": int(len(arr["clip_index"])), "n_with_output": int(arr["has_out"].sum()), "n_without_output": int((~arr["has_out"]).sum()),
                            "dump_set": "large fits (this card)" if nm == MEAN_L else "500-clip fits of card ADD104", "dump_sha256": {"fit%d" % s: F[key]["s%d" % s]["dump_sha256"] for s in LARGE_SEEDS}}
    rec["v0"] = v0_check(rec)
    rec["build"] = build_check(rec)
    rec["meta"] = meta_check(rec, F)
    rec["g2"] = g2_check(F)
    rec["g3"] = g3_check(rec, F)
    rec["fit_facts_for_reader"] = fit_facts_for_reader(F)
    if rec["g3"]["PASS"]:
        rec["t0_diagnostic"] = t0_diagnostic(J, [DL[s] for s in LARGE_SEEDS], [DM[s] for s in MID_SEEDS])
    with open(os.path.join(out_dir, RECORD_NAME), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    return rec


# ============================================================================================ synthetic data (selftest and tests)
def synthetic_join(n_clips=4, seed=1):
    rng = np.random.RandomState(seed)
    ns = [12 + 3 * i for i in range(n_clips)]
    off = np.concatenate([[0], np.cumsum(ns)[:-1]]).astype(np.int64)
    tot = int(sum(ns))
    gap = rng.uniform(3.0, 55.0, tot).astype(np.float32)
    valid = (rng.rand(tot) < 0.7).astype(np.int64)
    gap[rng.rand(tot) < 0.1] = np.nan
    return {"clip": np.asarray(["clip%02d" % i for i in range(n_clips)]), "off": off, "n": np.asarray(ns, np.int64), "lead_gap": gap, "lead_valid": valid}


def synthetic_dump(J, seed, rng_seed=7):
    """a dump in the format of cropread/fit/FIT_DEFS.md section 2: every frame of every clip once; the head H5 has no output at frames 0 to 3; one more NaN at frame 9 for odd seeds"""
    rng = np.random.RandomState(rng_seed + seed)
    names = [str(c) for c in J["clip"]]
    cid, fr, ci, lab, labp, valid, gap = [], [], [], [], [], [], []
    for c, n in enumerate(np.asarray(J["n"], np.int64)):
        for f in range(int(n)):
            r = int(J["off"][c]) + f
            g = float(J["lead_gap"][r])
            ok = int(J["lead_valid"][r]) == 1 and np.isfinite(g) and 0.0 <= g <= GAP_MAX_M
            cid.append(c)
            fr.append(f)
            ci.append(c)
            lab.append(g if ok else np.nan)
            labp.append(ok)
            valid.append(bool(rng.rand() < 0.9))
            hg = np.nan if f < FIRST_OUTPUT_FRAME else (g if np.isfinite(g) else 20.0) + rng.randn() * 2.0
            if f == 9 and seed % 2 == 1:
                hg = np.nan
            if f == 7:
                hg = 64.0 + c + 0.1 * seed                                         # above 60 m, kept as dumped
            gap.append(hg)
    return {"cid": np.asarray(cid, np.int64), "frame": np.asarray(fr, np.int64), "clip_index": np.asarray(ci, np.int64), "lab_gap": np.asarray(lab, np.float64), "lab_present": np.asarray(labp, bool),
            "clips": np.asarray(names), "gap": np.asarray(gap, np.float64), "valid": np.asarray(valid, bool), "seed": np.int64(seed)}


def _w(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)


def _fit_record(d, s, clips_fit, clips_sel, train_sha, eval_sha, head_bytes, argv_extra=(), kept=30):
    os.makedirs(d, exist_ok=True)
    hp = os.path.join(d, "head.pt")
    with open(hp, "wb") as fh:
        fh.write(head_bytes)
    r = {"card": FIT_CARD, "tier": "T0", "arm": FIT_ARM, "head": FIT_HEAD, "seed": s, "permute_labels": False,
         "argv": ["crop_fit.py", "--arm", FIT_ARM, "--head", FIT_HEAD, "--seed", str(s), "--root", "r", "--join-train", "j", "--device", "cuda"] + list(argv_extra), "tool_sha256": "a" * 64,
         "train_list_sha256": train_sha, "eval_list_sha256": eval_sha, "clips_read": {"fit": clips_fit, "sel": clips_sel}, "n_samples": {"fit": 10 * len(clips_fit), "sel": 10 * len(clips_sel)},
         "hparams": dict(CARD_HPARAMS), "hparams_equal_card": True, "kept_epoch": kept + s, "kept_sel_loss": 1.0 + 0.1 * s, "head_pt_sha256": sha_file(hp)}
    _w(os.path.join(d, FIT_RECORD), json.dumps(r))
    return hp


def synthetic_world(base, n_new=20):
    """a tiny disk world in base: lists (pilot 450 fit + 50 sel, large = pilot + n_new fit), the evaluation list, large fits with dumps and metas, the 500-clip fits with dumps and metas, the fits of seeds 0 and 1, the
    record of card ADD104. -> (J, DL, DM, P) with P the paths of collect()."""
    J = synthetic_join()
    pilot = [["pl%03d" % i, "fit" if i < 450 else "sel"] for i in range(N_PILOT)]
    large = pilot + [["nw%03d" % i, "fit"] for i in range(n_new)]
    P = {"fit_root": os.path.join(base, "fitroot"), "large_list": os.path.join(base, "lists", "large.txt"), "pilot_list": os.path.join(base, "run1", "lists", "train_final.txt"),
         "eval_list": os.path.join(base, "run1", "lists", "eval141.txt"), "old_fits": os.path.join(base, "run1", "fits"), "mid_root": os.path.join(base, "mid"), "rec104": os.path.join(base, "mid", "REC104.json")}
    _w(P["large_list"], "".join("%s %s\n" % (c, r) for c, r in large))
    _w(P["pilot_list"], "".join("%s %s\n" % (c, r) for c, r in pilot))
    _w(P["eval_list"], "".join("ev%03d\n" % i for i in range(7)))
    tsha, psha, esha = sha_file(P["large_list"]), sha_file(P["pilot_list"]), sha_file(P["eval_list"])
    pf, ps = [c for c, r in pilot if r == "fit"], [c for c, r in pilot if r == "sel"]
    lf = [c for c, r in large if r == "fit"]
    DL, DM, r104 = {}, {}, {"files": {}, "x0": {"PASS": True}, "x1": {"PASS": True}}
    for s in OLD_SEEDS:
        _fit_record(fit_dir(P["old_fits"], s), s, pf, ps, psha, esha, ("old head %d " % s).encode() * 20)
    for s in MID_SEEDS:
        hp = _fit_record(fit_dir(os.path.join(P["mid_root"], "fits"), s), s, pf, ps, psha, esha, ("mid head %d " % s).encode() * 20)
        DM[s] = synthetic_dump(J, s, rng_seed=100)
        dp = os.path.join(P["mid_root"], "dump", DUMP_DIR, "reader_s%d_score.npz" % s)
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        np.savez(dp, **DM[s])
        _w(os.path.join(P["mid_root"], "dump", DUMP_DIR, "reader_s%d_meta.json" % s), json.dumps({"card": FIT_CARD, "arm": FIT_ARM, "head": FIT_HEAD, "seed": s, "permute_labels": False, "head_pt_sha256": sha_file(hp),
                                                                                              "dump_sha256": sha_file(dp)}))
        r104["files"]["VGAPF_s%d" % s] = {"dump_sha256": sha_file(dp)}
    _w(P["rec104"], json.dumps(r104))
    for s in LARGE_SEEDS:
        hp = _fit_record(fit_dir(os.path.join(P["fit_root"], "fits"), s), s, lf, ps, tsha, esha, ("large head %d " % s).encode() * 20)
        DL[s] = synthetic_dump(J, s, rng_seed=7)
        dp = os.path.join(P["fit_root"], "dump", DUMP_DIR, "reader_s%d_score.npz" % s)
        os.makedirs(os.path.dirname(dp), exist_ok=True)
        np.savez(dp, **DL[s])
        _w(os.path.join(P["fit_root"], "dump", DUMP_DIR, "reader_s%d_meta.json" % s), json.dumps({"card": FIT_CARD, "arm": FIT_ARM, "head": FIT_HEAD, "seed": s, "permute_labels": False,
                                                                                                   "head_pt_sha256": sha_file(hp), "dump_sha256": sha_file(dp)}))
    return J, DL, DM, P


def g2_faults(F):
    """each planted fault of each clause of g2 turns it red -> dict name -> bool (True = turned red as it must); the clean facts pass"""
    out = {"clean_g2_passes": g2_check(F)["PASS"]}

    def red(mut, clause=None):
        f = copy.deepcopy(F)
        mut(f)
        r = g2_check(f)
        return (not r["PASS"]) and (clause is None or r["clauses"].get(clause) is False)
    n = lambda f, s, kind="large": f[kind]["s%d" % s]                                          # noqa: E731
    out["a_seed_in_record"] = red(lambda f: n(f, 3).update(seed=2), "a")
    out["a_seed_is_a_bool"] = red(lambda f: n(f, 3).update(seed=True), "a")
    out["a_arm_b"] = red(lambda f: n(f, 3).update(arm="B"), "a")
    out["a_head_h1"] = red(lambda f: n(f, 3).update(head="H1"), "a")
    out["b_card_name"] = red(lambda f: n(f, 2).update(card="PREREG_V8F_PROOF_ADD100"), "b")
    out["b_flag_false"] = red(lambda f: n(f, 2).update(hparams_equal_card=False), "b")
    out["b_other_lr"] = red(lambda f: n(f, 4).update(hparams=dict(CARD_HPARAMS, lr=0.01)), "b")
    out["c_permuted_flag"] = red(lambda f: n(f, 3).update(permute_labels=True), "c")
    out["c_permuted_in_argv"] = red(lambda f: n(f, 3).update(argv=list(n(f, 3)["argv"]) + ["--permute-labels"]), "c")
    out["c_argv_seed"] = red(lambda f: n(f, 3).update(argv=["crop_fit.py", "--arm", "D", "--head", "H5", "--seed", "4"]), "c")
    out["c_argv_arm"] = red(lambda f: n(f, 3).update(argv=["crop_fit.py", "--arm", "B", "--head", "H5", "--seed", "3"]), "c")
    out["d_fit_clips_are_the_pilots_450"] = red(lambda f: n(f, 2).update(fit_clips=[c for c, r in f["lists"]["pilot"] if r == "fit"]), "d")
    out["d_one_fit_clip_missing"] = red(lambda f: n(f, 2).update(fit_clips=n(f, 2)["fit_clips"][:-1]), "d")
    out["d_other_clip_order"] = red(lambda f: n(f, 3).update(fit_clips=list(reversed(n(f, 3)["fit_clips"]))), "d")
    out["d_list_hash"] = red(lambda f: n(f, 4).update(train_list_sha256="c" * 64), "d")
    out["d_pilot_hash_instead"] = red(lambda f: n(f, 4).update(train_list_sha256=f["lists"]["pilot_sha256"]), "d")
    out["e_one_sel_clip_replaced"] = red(lambda f: n(f, 2).update(sel_clips=n(f, 2)["sel_clips"][:-1] + ["nw000"]), "e")
    out["e_49_sel_clips"] = red(lambda f: n(f, 2).update(sel_clips=n(f, 2)["sel_clips"][:-1]), "e")
    out["f_eval_clip_in_fit"] = red(lambda f: n(f, 2)["fit_clips"].__setitem__(0, "ev001"), "f")
    out["f_eval_clip_in_sel"] = red(lambda f: n(f, 4)["sel_clips"].__setitem__(0, "ev002"), "f")
    out["f_eval_list_hash_in_record"] = red(lambda f: n(f, 4).update(eval_list_sha256="d" * 64), "f")
    out["f_eval_clip_in_large_list"] = red(lambda f: f["lists"]["eval_ids"].append("nw001"), "f")
    out["g_head_disk_not_record"] = red(lambda f: n(f, 3).update(head_disk_sha256="e" * 64), "g")
    out["g_head_file_missing"] = red(lambda f: n(f, 3).update(head_disk_sha256=None), "g")
    out["h_head_equals_old_seed_0"] = red(lambda f: n(f, 2).update(head_pt_sha256_record=n(f, 0, "old")["head_pt_sha256_record"], head_disk_sha256=n(f, 0, "old")["head_pt_sha256_record"]), "h")
    out["h_head_equals_old_seed_1"] = red(lambda f: n(f, 3).update(head_pt_sha256_record=n(f, 1, "old")["head_pt_sha256_record"], head_disk_sha256=n(f, 1, "old")["head_pt_sha256_record"]), "h")
    out["h_head_equals_500_clip_fit"] = red(lambda f: n(f, 4).update(head_pt_sha256_record=n(f, 4, "mid")["head_pt_sha256_record"], head_disk_sha256=n(f, 4, "mid")["head_pt_sha256_record"]), "h")
    out["h_two_large_heads_equal"] = red(lambda f: n(f, 4).update(head_pt_sha256_record=n(f, 3)["head_pt_sha256_record"], head_disk_sha256=n(f, 3)["head_disk_sha256"]), "h")
    out["h_mid_record_unreadable"] = red(lambda f: f["mid"].update(s3={"dir": "x", "error": "OSError: no file"}), "h")
    out["a_large_record_unreadable"] = red(lambda f: f["large"].update(s3={"dir": "x", "error": "OSError: no file"}), "a")
    out["i_large_list_does_not_start_with_pilot"] = red(lambda f: f["lists"]["large"].__setitem__(0, ["pl000", "sel"]), "i")
    out["i_pilot_has_49_sel"] = red(lambda f: f["lists"]["pilot"].__setitem__(499, ["pl499", "fit"]), "i")
    out["no_evaluation_list"] = red(lambda f: f["lists"].update(eval_ids=[]), "f")
    return out


def g3_faults(J, DL, DM, F, base):
    """each planted fault of each part of g3 turns it red -> dict name -> bool"""
    out = {}
    i = [0]

    def run(fault=None, dl=None, dm=None, f=None):
        i[0] += 1
        rec = build(J, dl or DL, dm or DM, f or F, os.path.join(base, "g3f_%03d" % i[0]), fault=fault)
        return rec

    r = run()
    out["clean_g3_passes"] = bool(r["g3"]["PASS"] and r["g2"]["PASS"])
    for fl in ("scale", "shift", "min_label", "clip60", "zero_nooutput"):
        out["build_" + fl] = not run("s_" + fl)["g3"]["PASS"]
    for fl in ("offset", "mean_of_two", "median", "mean_of_available", "flag_all_true", "zero_nooutput", "flag_false_on_a_row"):
        out["mean_" + fl] = not run("m_" + fl)["g3"]["PASS"]

    def dumps_with(mut, which="L", s=2):
        dl, dm = {k: dict(v) for k, v in DL.items()}, {k: dict(v) for k, v in DM.items()}
        mut((dl if which == "L" else dm)[s])
        return dl, dm
    cols = ("cid", "frame", "clip_index", "lab_gap", "lab_present", "gap", "valid")
    keep = np.ones(len(DL[2]["cid"]), bool)
    keep[5] = False
    perm = np.array([1, 0] + list(range(2, len(J["clip"]))))
    inv = np.argsort(perm)
    muts = {"v0_wrong_seed": lambda d: d.update(seed=np.int64(3)),
            "v0_clips_other_order": lambda d: d.update(clips=d["clips"][perm], cid=inv[d["cid"]]),
            "v0_label_gap_off": lambda d: d.update(lab_gap=d["lab_gap"] + 0.01),
            "v0_missing_row": lambda d: d.update({k: np.asarray(v)[keep] for k, v in d.items() if k in cols}),
            "v0_duplicate_row": lambda d: d.update({k: np.concatenate([np.asarray(v), np.asarray(v)[:1]]) for k, v in d.items() if k in cols}),
            "v0_row_beyond_join": lambda d: d.update(frame=np.where(np.arange(len(d["frame"])) == 3, 999, d["frame"])),
            "v0_clip_index_wrong": lambda d: d.update(clip_index=np.where(np.arange(len(d["clip_index"])) == 20, 0, d["clip_index"]))}
    for nm, mut in muts.items():
        dl, dm = dumps_with(mut)
        out[nm] = not run(dl=dl, dm=dm)["g3"]["PASS"]
    for nm in ("v0_missing_row", "v0_duplicate_row", "v0_row_beyond_join"):                         # the same planted fault in a 500-clip dump turns the mean check red (the same rows)
        dl, dm = dumps_with(muts[nm], "M", 3)
        out["mid_" + nm[3:]] = not run(dl=dl, dm=dm)["g3"]["PASS"]

    def facts_with(mut):
        f = copy.deepcopy(F)
        mut(f)
        return f
    out["meta_head_sha_of_the_dump"] = not run(f=facts_with(lambda f: f["large_dumps"]["s3"]["dump_meta"].update(head_pt_sha256="0" * 64)))["g3"]["PASS"]
    out["meta_seed"] = not run(f=facts_with(lambda f: f["large_dumps"]["s2"]["dump_meta"].update(seed=3)))["g3"]["PASS"]
    out["meta_missing"] = not run(f=facts_with(lambda f: f["large_dumps"].update(s4={**f["large_dumps"]["s4"], "dump_meta": None})))["g3"]["PASS"]
    out["meta_dump_sha"] = not run(f=facts_with(lambda f: f["large_dumps"]["s2"]["dump_meta"].update(dump_sha256="1" * 64)))["g3"]["PASS"]
    out["meta_permuted"] = not run(f=facts_with(lambda f: f["large_dumps"]["s2"]["dump_meta"].update(permute_labels=True)))["g3"]["PASS"]
    out["mid_dump_not_the_one_of_rec104"] = not run(f=facts_with(lambda f: f["rec104"]["dump_sha256"].update(s3="2" * 64)))["g3"]["PASS"]
    out["mid_rec104_x1_not_pass"] = not run(f=facts_with(lambda f: f["rec104"].update(x1_PASS=False)))["g3"]["PASS"]
    out["mid_rec104_x0_not_pass"] = not run(f=facts_with(lambda f: f["rec104"].update(x0_PASS=False)))["g3"]["PASS"]
    out["mid_rec104_missing"] = not run(f=facts_with(lambda f: f["rec104"].update(x0_PASS=False, x1_PASS=False, dump_sha256={"s2": None, "s3": None, "s4": None}, found=False)))["g3"]["PASS"]
    out["mid_dump_meta_head"] = not run(f=facts_with(lambda f: f["mid_dumps"]["s4"]["dump_meta"].update(head_pt_sha256="3" * 64)))["g3"]["PASS"]
    return out


def selftest():
    ok = True
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_selftest_scale_%d" % os.getpid())
    os.makedirs(base)
    J, DL, DM, P = synthetic_world(os.path.join(base, "world"))
    F = collect(P)
    rec = build(J, DL, DM, F, os.path.join(base, "clean"), eval_join_sha="9" * 64)
    print("clean build: g2 %s g3 %s parts %s" % (rec["g2"]["PASS"], rec["g3"]["PASS"], rec["g3"]["parts"]))
    ok &= rec["g2"]["PASS"] and rec["g3"]["PASS"] and "t0_diagnostic" in rec and sorted(rec["files"]) == sorted(L_NAMES + [MEAN_L, MEAN_M])
    z = load_npz(os.path.join(base, "clean", "VGAPL_s2.npz"))
    ok &= bool(np.isnan(z["gap"][~z["has_out"]]).all()) and int((~z["has_out"]).sum()) >= 4 * len(J["clip"]) and float(z["gap"][z["has_out"]].max()) > 60.0 and int(z["seed"]) == 2
    zm = load_npz(os.path.join(base, "clean", "VGAPLE.npz"))
    ok &= sorted(zm.keys()) == sorted(["clip_index", "frame", "gap", "has_out", "clips", "seed"]) and int(zm["seed"]) == -1
    f2 = g2_faults(F)
    for k, v in sorted(f2.items()):
        print("g2 planted fault %-42s turned red: %s" % (k, v))
    f3 = g3_faults(J, DL, DM, F, base)
    for k, v in sorted(f3.items()):
        print("g3 planted fault %-42s turned red: %s" % (k, v))
    ok &= all(f2.values()) and all(f3.values())
    print("ZZ MAKE-SCALE-GAPFILES-SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--eval-join")
    ap.add_argument("--out-dir")
    ap.add_argument("--fit-root", default=D_FIT_ROOT)
    ap.add_argument("--large-list", default=D_LARGE_LIST)
    ap.add_argument("--pilot-list", default=D_PILOT_LIST)
    ap.add_argument("--eval-list", default=D_EVAL_LIST)
    ap.add_argument("--old-fits", default=D_OLD_FITS)
    ap.add_argument("--mid-root", default=D_MID_ROOT)
    ap.add_argument("--rec104", default=D_REC104)
    ap.add_argument("--expect-clips", type=int, default=141)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.eval_join and a.out_dir):
        print("REFUSED: --eval-join and --out-dir are required")
        return 2
    P = {"fit_root": a.fit_root, "large_list": a.large_list, "pilot_list": a.pilot_list, "eval_list": a.eval_list, "old_fits": a.old_fits, "mid_root": a.mid_root, "rec104": a.rec104}
    for k in ("large_list", "pilot_list", "eval_list", "rec104"):
        if not os.path.isfile(P[k]):
            print("REFUSED: no file %s" % P[k])
            return 2
    J = load_npz(a.eval_join)
    F = collect(P)
    DL, DM = {}, {}
    for s in LARGE_SEEDS:
        for key, dd, tgt in (("large_dumps", DL, s), ("mid_dumps", DM, s)):
            p = F[key]["s%d" % s]["dump_file"]
            if not os.path.isfile(p):
                print("REFUSED: no dump %s" % p)
                return 2
            dd[tgt] = load_npz(p)
    inputs = {k: {"file": v, "sha256": sha_file(v)} for k, v in P.items() if k in ("large_list", "pilot_list", "eval_list", "rec104")}
    inputs["eval_join"] = {"file": a.eval_join, "sha256": sha_file(a.eval_join)}
    rec = build(J, DL, DM, F, a.out_dir, eval_join_sha=sha_file(a.eval_join), expect_clips=a.expect_clips, inputs=inputs)
    P2 = lambda k: "PASS" if rec[k]["PASS"] else "FAIL"                                                    # noqa: E731
    print("ZZ SCALE-GAPFILES g2 %s g3 %s (parts %s)" % (P2("g2"), P2("g3"), rec["g3"]["parts"]))
    for k in ("g2", "g3"):
        for x in rec[k]["failures"][:8]:
            print(k, x)
    return 0 if rec["g2"]["PASS"] and rec["g3"]["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
