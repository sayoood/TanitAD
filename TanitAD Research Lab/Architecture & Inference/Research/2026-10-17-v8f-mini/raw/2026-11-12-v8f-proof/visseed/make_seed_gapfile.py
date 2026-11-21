"""make_seed_gapfile.py -- the three gap files of the head on five frames (D-H5) of the fit seeds 2, 3 and 4 for card PREREG_V8F_PROOF_ADD104 (VISSEED10K), with controls x0 (the fits) and x1 (the gap files).
Derived from make_vis_gapfile.py (card ADD102) by counted edits (gen_seed_gapfile.py): seeds, file names, record name, the fit-record check x0 and the dump-meta clause of x1 are the changes; the build is the same. numpy and the standard library only (runs on the pod with /workspace/venv/bin/python).
Usage: python make_seed_gapfile.py --eval-join NPZ --fit-root DIR --out-dir NEWDIR [--add99-fits DIR] [--eval-list FILE] [--expect-clips 141]
         reads FITROOT/dump/D_H5/reader_s<S>_score.npz and reader_s<S>_meta.json (the fixed-window head H5 of card ADD99, fit seed S) and FITROOT/fits/D_H5_s<S>/FIT_RECORD.json for S in 2 3 4, and the records of fit seeds 0 and 1
         in --add99-fits (default /workspace/v8proof/cropread/run1/fits); writes NEWDIR/VGAPF_s<S>.npz and NEWDIR/VISSEED_GAPFILES_RECORD.json with x0 (the fits) and x1 = v0 (the dumps are on the join frames) + the build check
         (saved gap equals a second routine) + the meta clause (each dump meta names the sha256 of the head file of its fit)
       python make_seed_gapfile.py --selftest     synthetic join and dumps; the clean build passes v0 and the build check, each planted fault turns one red
A gap file has one row per frame of the evaluation join (every clip, frames 0 to n-1, in the order of the join's clips): clip_index (int64, row of the clip in the evaluation join), frame (int64),
gap (float64, the head's output as dumped, metres; NaN where the head has no output), has_out (bool: the dumped gap is finite), clips (the join's clip names), seed (int64). The head's gap is NOT
mapped, NOT clipped to 0 to 60 m and NOT combined with a minimum; the dump's `valid` field (which carries the present reader's own validity) is not used: a frame has an output when its dumped gap is finite.
Dump format: cropread/fit/FIT_DEFS.md section 2, cropread/fit/crop_dump.py:54-70: `cid` indexes the dump's own array `clips`, `clip_index` is the row of that clip in the evaluation join.
Control v0 of the card (section 4): the dumps are on the frames of the evaluation join: the dump's clips are the join's clips in the same order (141), every dump row has its join row, the rows of every clip
are the frames 0 to n-1 of its join clip once each. Kept from the kit of card ADD101 (an addition): the label gap of the dump equals the lead gap of the join within 1e-4 m on every label frame.
The build check compares every saved row with a second routine (plain loops over the dump alone). Refuses an existing out dir."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD104"
GAP_MAX_M = 60.0
TOL_M = 1e-4
TOL_SAVED = 1e-9
SEEDS = (2, 3, 4)                           # fit seeds (the dump of fit seed S is reader_s<S>_score.npz, made on the rows of the present reader's dump of seed 0)
DUMP_DIR = "D_H5"
FIRST_OUTPUT_FRAME = 4                      # the head H5 reads five frames (cropread/fit/crop_dump.py:29-35): no output at frames 0 to 3


def file_names():
    return ["VGAPF_s%d" % s for s in SEEDS]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def frame_table(J, D):
    """join row of every dump row. -> dict(ji [rows] join clip index of the dump's clip name or -1, row [rows], inrange bool, names_not_in_join int, of [dump clips] join index or -1)"""
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


def _valid_mask(J):
    g = np.asarray(J["lead_gap"], np.float64)
    return (np.asarray(J["lead_valid"]) == 1) & np.isfinite(g) & (g >= 0.0) & (g <= GAP_MAX_M)


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
    names = [str(c) for c in J["clip"]]
    where = {}
    for i, c in enumerate(names):
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


def measure_file(J, D, arr, dump_path, dump_dir_name, file_path):
    """facts of the build check for one gap file"""
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
        else:
            if not has[i]:
                n_out_mismatch += 1
            else:
                dev = max(dev, abs(float(gap[i]) - r))
    keys = ci * (1 << 20) + fr
    first = {}
    for c in range(len(n)):
        m = (ci == c) & has
        first[c] = int(fr[m].min()) if m.any() else None
    fc = {}
    for v in first.values():
        fc[str(v)] = fc.get(str(v), 0) + 1
    fin = gap[has]
    return {"file": os.path.basename(file_path), "file_sha256": sha_file(file_path) if os.path.isfile(file_path) else None, "dump_file": dump_path, "dump_dir_name": dump_dir_name,
            "dump_sha256": sha_file(dump_path) if os.path.isfile(dump_path) else None, "n_rows": int(len(ci)), "n_join_rows": int(n.sum()), "n_with_output": int(has.sum()), "n_without_output": int((~has).sum()),
            "max_abs_dev_saved_minus_second_routine": dev, "n_rows_without_dump_row": n_missing, "n_output_flag_mismatch": n_out_mismatch, "keys_sorted_unique": bool(np.all(np.diff(keys) > 0)),
            "rows_cover_the_join_in_order": bool(len(ci) == int(n.sum()) and np.array_equal(ci, np.repeat(np.arange(len(n)), n))),
            "first_output_frame_per_clip_counts": fc, "n_clips_without_any_output": int(sum(1 for v in first.values() if v is None)), "n_finite_but_dump_not_valid": int((has & ~_dump_valid_rows(J, D, arr)).sum()),
            "n_out_below_0": int((fin < 0.0).sum()), "n_out_above_60": int((fin > GAP_MAX_M).sum()), "min_out": float(fin.min()) if fin.size else None, "max_out": float(fin.max()) if fin.size else None}


def _dump_valid_rows(J, D, arr):
    """the dump's `valid` field laid on the join rows (False where the dump has no row); counted only, never used to decide an output"""
    tab = frame_table(J, D)
    v = np.zeros(len(arr["clip_index"]), bool)
    ok = tab["inrange"]
    v[tab["row"][ok]] = np.asarray(D["valid"], bool)[ok]
    return v


def v0_check(rec):
    """control v0 from a record (card section 4)."""
    bad = []
    try:
        if sorted(rec["dumps"]) != sorted(file_names()):
            bad.append("dumps in the record: %s" % sorted(rec["dumps"]))
        for nm in file_names():
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
    """the gap files against the second routine and their own facts (an addition to the card's control v0): every row of the join once, in order; the saved gap equals the second routine within 1e-9;
    the output flag is the finiteness of the dumped gap; the file of every model seed is built from the file of its dump (folder D_H5, reader_s<k>_score.npz)."""
    bad = []
    try:
        if rec.get("card") != CARD:
            bad.append("record is for card %s" % rec.get("card"))
        if sorted(rec["files"]) != sorted(file_names()):
            bad.append("files in the record: %s" % sorted(rec["files"]))
        for nm in file_names():
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


ADD99_FITS = "/workspace/v8proof/cropread/run1/fits"
FIT_CARD, FIT_ARM, FIT_HEAD, FIT_RECORD = "PREREG_V8F_PROOF_ADD99", "D", "H5", "FIT_RECORD.json"
OLD_SEEDS = (0, 1)                          # the two fits of card ADD99 on record (cropread/run1/fits/D_H5_s0, D_H5_s1)
N_TRAIN = 500                               # card ADD104 section 4 x0: the 500 train clips (fit 450 and sel 50 of card ADD99, section 2 of the card)
CARD_HPARAMS = {"lr": 1e-3, "weight_decay": 0.05, "batch": 256, "epochs": 40, "warmup_epochs": 1, "beta_m": 1.0, "token_drop": 0.1, "d_model": 256, "n_heads": 4}      # cropread/fit/crop_fit.py:32


def fit_dir_name(s):
    """cropread/fit/crop_fit.py:48 fit_name(arm, head, seed): D_H5_s<seed>"""
    return "%s_%s_s%d" % (FIT_ARM, FIT_HEAD, s)


def read_ids(p):
    with open(p, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def fit_facts(fits_dir, s):
    """facts of one fit from its FIT_RECORD.json (keys of cropread/fit/crop_fit.py:300-304) and the sha256 of its head.pt on disk. Unreadable -> {"error": ...}."""
    d = os.path.join(fits_dir, fit_dir_name(s))
    rp, hp = os.path.join(d, FIT_RECORD), os.path.join(d, "head.pt")
    try:
        with open(rp, encoding="utf-8") as fh:
            r = json.load(fh)
        cr = r.get("clips_read") or {}
        return {"dir": d, "seed": r.get("seed"), "card": r.get("card"), "arm": r.get("arm"), "head": r.get("head"), "permute_labels": r.get("permute_labels"), "argv": r.get("argv"),
                "hparams": r.get("hparams"), "hparams_equal_card": r.get("hparams_equal_card"), "train_list_sha256": r.get("train_list_sha256"), "eval_list_sha256": r.get("eval_list_sha256"),
                "fit_clips": cr.get("fit"), "sel_clips": cr.get("sel"), "head_pt_sha256_record": r.get("head_pt_sha256"), "head_disk_sha256": sha_file(hp) if os.path.isfile(hp) else None,
                "tool_sha256": r.get("tool_sha256"), "kept_epoch": r.get("kept_epoch"), "kept_sel_loss": r.get("kept_sel_loss"), "record_sha256": sha_file(rp)}
    except (OSError, ValueError, AttributeError) as ex:
        return {"dir": d, "error": "%s: %s" % (type(ex).__name__, ex)}


def collect(fit_root, add99_fits, eval_list, dump_root):
    """-> (fits, meta): the facts of the three new fits and of the two fits on record, the evaluation list, and per fit seed the dump file, its folder name and its meta file (crop_dump.py:103-106)."""
    have = os.path.isfile(eval_list)
    fits = {"new": {"s%d" % s: fit_facts(os.path.join(fit_root, "fits"), s) for s in SEEDS}, "old": {"s%d" % s: fit_facts(add99_fits, s) for s in OLD_SEEDS},
            "eval_list": eval_list, "eval_list_sha256": sha_file(eval_list) if have else None, "eval_ids": read_ids(eval_list) if have else []}
    meta = {}
    for s in SEEDS:
        mp = os.path.join(dump_root, DUMP_DIR, "reader_s%d_meta.json" % s)
        dm = None
        if os.path.isfile(mp):
            try:
                with open(mp, encoding="utf-8") as fh:
                    dm = json.load(fh)
            except ValueError:
                dm = None
        meta[s] = {"dump_file": os.path.join(dump_root, DUMP_DIR, "reader_s%d_score.npz" % s), "dump_dir_name": DUMP_DIR, "dump_meta": dm}
    return fits, meta


def _flag(argv, flag):
    return argv[argv.index(flag) + 1] if (flag in argv and argv.index(flag) + 1 < len(argv)) else None


def x0_check(fits):
    """control x0 of the card (section 4), from the facts of collect(): each new fit record names its seed (2, 3, 4), head H5 on arm D, the settings of card ADD99, the 500 train clips and no evaluation clip; the train
    list hash equals that of the fits of card ADD99; the head files differ from one another and from those of seeds 0 and 1. Additions that only make the card's words checkable: the record's own head hash equals the
    file on disk, the clips are those of the two fits on record, the evaluation list hash equals theirs, the arguments of the call name the seed."""
    bad = []
    try:
        new = [(s, fits["new"]["s%d" % s]) for s in SEEDS]
        old = [(s, fits["old"]["s%d" % s]) for s in OLD_SEEDS]
        for s, f in new + old:
            if "error" in f:
                bad.append("seed %d: fit record unreadable: %s" % (s, f["error"]))
        if bad:
            return {"PASS": False, "failures": bad}
        ref = fits["old"]["s0"]
        eval_ids = set(fits["eval_ids"])
        if not eval_ids or not (isinstance(fits.get("eval_list_sha256"), str) and len(fits["eval_list_sha256"]) == 64):
            bad.append("no evaluation list")
        heads = {}
        for s, f in new + old:
            nm = "seed %d" % s
            if f["arm"] != FIT_ARM or f["head"] != FIT_HEAD:
                bad.append("%s: arm %s head %s" % (nm, f["arm"], f["head"]))
            if f["seed"] != s or isinstance(f["seed"], bool):
                bad.append("%s: the record names seed %s" % (nm, f["seed"]))
            if (s in SEEDS) and (f["card"] != FIT_CARD or f["permute_labels"] is not False):
                bad.append("%s: card %s permute_labels %s" % (nm, f["card"], f["permute_labels"]))
            if s in SEEDS:
                av = f.get("argv") or []
                if not (_flag(av, "--arm") == FIT_ARM and _flag(av, "--head") == FIT_HEAD and _flag(av, "--seed") == str(s) and "--permute-labels" not in av):
                    bad.append("%s: the arguments of the call do not name arm D, head H5, seed %d" % (nm, s))
                if f["hparams_equal_card"] is not True or f["hparams"] != CARD_HPARAMS:
                    bad.append("%s: the settings are not those of card ADD99" % nm)
            if f["hparams"] != ref["hparams"]:
                bad.append("%s: the settings differ from those of seed 0" % nm)
            clips = list(f["fit_clips"] or []) + list(f["sel_clips"] or [])
            if len(clips) != N_TRAIN or len(set(clips)) != len(clips):
                bad.append("%s: %d train clips read (%d distinct), expected %d" % (nm, len(clips), len(set(clips)), N_TRAIN))
            if set(clips) & eval_ids:
                bad.append("%s: %d evaluation clips among the clips read" % (nm, len(set(clips) & eval_ids)))
            if f["fit_clips"] != ref["fit_clips"] or f["sel_clips"] != ref["sel_clips"]:
                bad.append("%s: the clips read are not those of seed 0" % nm)
            for k in ("train_list_sha256", "eval_list_sha256"):
                if not (isinstance(f[k], str) and len(f[k]) == 64) or f[k] != ref[k]:
                    bad.append("%s: %s is not the one of seed 0" % (nm, k))
            if f["eval_list_sha256"] != fits.get("eval_list_sha256"):
                bad.append("%s: eval_list_sha256 is not the sha256 of the evaluation list on disk" % nm)
            rec_h, disk_h = f["head_pt_sha256_record"], f["head_disk_sha256"]
            if not (isinstance(rec_h, str) and len(rec_h) == 64):
                bad.append("%s: the record has no head sha256" % nm)
                continue
            if (s in SEEDS and disk_h is None) or (disk_h is not None and disk_h != rec_h):
                bad.append("%s: the head file on disk is %s, the record says %s" % (nm, (disk_h or "missing")[:16], rec_h[:16]))
            heads.setdefault(rec_h, []).append(s)
        dup = {h[:16]: v for h, v in heads.items() if len(v) > 1}
        if dup:
            bad.append("head files that are the same file: %s" % dup)
    except (KeyError, TypeError, AttributeError) as ex:
        bad.append("facts incomplete: %s: %s" % (type(ex).__name__, ex))
    return {"PASS": not bad, "failures": bad}


def meta_check(rec):
    """control x1, third clause: each dump's meta file (crop_dump.py:103-106) names the sha256 of the head file of its fit (the file on disk), its fit seed, arm D, head H5, and the sha256 of the dump file."""
    bad = []
    try:
        for s in SEEDS:
            nm = "VGAPF_s%d" % s
            dm = rec["dumps"][nm].get("dump_meta")
            f = rec["fits"]["new"]["s%d" % s]
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


def synthetic_fit_world(root, J, dumps):
    """a tiny fit root (new fits, dumps with meta files, evaluation list) and a folder of the two fits on record -> (fits, meta, fit_root, old_root). Synthetic."""
    fit_root, old_root = os.path.join(root, "fitroot"), os.path.join(root, "old_fits")
    os.makedirs(os.path.join(fit_root, "lists"))
    ev = os.path.join(fit_root, "lists", "eval141.txt")
    with open(ev, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("".join("ev%03d\n" % i for i in range(7)))
    train = ["tr%03d" % i for i in range(N_TRAIN)]
    dump_root = os.path.join(fit_root, "dump")
    synthetic_meta(dumps, dump_root)
    for s in OLD_SEEDS + SEEDS:
        d = os.path.join(old_root if s in OLD_SEEDS else os.path.join(fit_root, "fits"), fit_dir_name(s))
        os.makedirs(d)
        hp = os.path.join(d, "head.pt")
        with open(hp, "wb") as fh:
            fh.write(("head of fit seed %d " % s).encode("ascii") * 20)
        r = {"card": FIT_CARD, "tier": "T0", "arm": FIT_ARM, "head": FIT_HEAD, "seed": s, "permute_labels": False,
             "argv": ["crop_fit.py", "--arm", FIT_ARM, "--head", FIT_HEAD, "--seed", str(s), "--root", "r", "--join-train", "j", "--device", "cuda"], "tool_sha256": "a" * 64,
             "train_list_sha256": "b" * 64, "eval_list_sha256": sha_file(ev), "clips_read": {"fit": train[:450], "sel": train[450:]}, "hparams": dict(CARD_HPARAMS), "hparams_equal_card": True,
             "kept_epoch": 30 + s, "kept_sel_loss": 1.0 + 0.1 * s, "head_pt_sha256": sha_file(hp)}
        with open(os.path.join(d, FIT_RECORD), "w", encoding="utf-8", newline="\n") as fh:
            json.dump(r, fh)
        if s in SEEDS:
            dm = {"card": FIT_CARD, "arm": FIT_ARM, "head": FIT_HEAD, "seed": s, "permute_labels": False, "head_pt_sha256": sha_file(hp),
                  "dump_sha256": sha_file(os.path.join(dump_root, DUMP_DIR, "reader_s%d_score.npz" % s))}
            with open(os.path.join(dump_root, DUMP_DIR, "reader_s%d_meta.json" % s), "w", encoding="utf-8", newline="\n") as fh:
                json.dump(dm, fh)
    fits, meta = collect(fit_root, old_root, ev, dump_root)
    return fits, meta, fit_root, old_root


def x_faults(base, J, dumps):
    """x0 and the meta clause of x1 on planted faults: the clean world passes, each fault turns its check red -> dict name -> bool (True = as it must)"""
    import copy
    fits, meta, fit_root, old_root = synthetic_fit_world(os.path.join(base, "xworld"), J, dumps)
    rec = build(J, dumps, os.path.join(base, "xbuild"), meta=meta, expect_clips=len(J["clip"]), fits=fits)
    out = {"clean_x0_and_x1_pass": bool(rec["x0"]["PASS"] and rec["x1"]["PASS"] and rec["meta"]["PASS"])}

    def red0(mut):
        f = copy.deepcopy(fits)
        mut(f)
        return not x0_check(f)["PASS"]
    n = lambda f, s: f["new"]["s%d" % s]                                                      # noqa: E731
    out["seed_in_record"] = red0(lambda f: n(f, 3).update(seed=2))
    out["arm_b"] = red0(lambda f: n(f, 3).update(arm="B"))
    out["head_h1"] = red0(lambda f: n(f, 3).update(head="H1"))
    out["permuted"] = red0(lambda f: n(f, 3).update(permute_labels=True))
    out["argv_seed"] = red0(lambda f: n(f, 3).update(argv=["crop_fit.py", "--arm", "D", "--head", "H5", "--seed", "4"]))
    out["card_name"] = red0(lambda f: n(f, 2).update(card="PREREG_V8F_PROOF_ADD100"))
    out["hparams_not_card"] = red0(lambda f: n(f, 2).update(hparams_equal_card=False))
    out["hparams_other_lr"] = red0(lambda f: n(f, 4).update(hparams=dict(CARD_HPARAMS, lr=0.01)))
    out["train_list_hash"] = red0(lambda f: n(f, 4).update(train_list_sha256="c" * 64))
    out["eval_list_hash_in_record"] = red0(lambda f: n(f, 4).update(eval_list_sha256="d" * 64))
    out["eval_clip_in_train"] = red0(lambda f: (n(f, 2)["fit_clips"].__setitem__(0, "ev001")))
    out["499_clips"] = red0(lambda f: n(f, 2).update(sel_clips=n(f, 2)["sel_clips"][:-1]))
    out["other_clip_order"] = red0(lambda f: n(f, 3).update(fit_clips=list(reversed(n(f, 3)["fit_clips"]))))
    out["head_is_the_one_of_seed_0"] = red0(lambda f: n(f, 2).update(head_pt_sha256_record=f["old"]["s0"]["head_pt_sha256_record"], head_disk_sha256=f["old"]["s0"]["head_pt_sha256_record"]))
    out["two_new_heads_equal"] = red0(lambda f: n(f, 4).update(head_pt_sha256_record=n(f, 3)["head_pt_sha256_record"], head_disk_sha256=n(f, 3)["head_disk_sha256"]))
    out["head_disk_not_record"] = red0(lambda f: n(f, 3).update(head_disk_sha256="e" * 64))
    out["head_file_missing"] = red0(lambda f: n(f, 3).update(head_disk_sha256=None))
    out["old_record_unreadable"] = red0(lambda f: f["old"].update(s1={"dir": "x", "error": "OSError: no file"}))
    out["no_evaluation_list"] = red0(lambda f: f.update(eval_ids=[], eval_list_sha256=None))

    def redm(mut):
        r = copy.deepcopy(rec)
        mut(r)
        return (not meta_check(r)["PASS"])
    out["meta_head_sha"] = redm(lambda r: r["dumps"]["VGAPF_s3"]["dump_meta"].update(head_pt_sha256="0" * 64))
    out["meta_seed"] = redm(lambda r: r["dumps"]["VGAPF_s2"]["dump_meta"].update(seed=3))
    out["meta_missing"] = redm(lambda r: r["dumps"]["VGAPF_s4"].update(dump_meta=None))
    out["meta_dump_sha"] = redm(lambda r: r["dumps"]["VGAPF_s2"]["dump_meta"].update(dump_sha256="1" * 64))
    return out


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def build(J, dumps, out_dir, faults=(), meta=None, expect_clips=None, eval_join_sha=None, fits=None):
    """dumps {seed: dict}; writes the files and the record. -> record. faults: names of planted faults (selftest and tests)."""
    if os.path.exists(out_dir):
        raise SystemExit("REFUSED: %s exists" % out_dir)
    os.makedirs(out_dir)
    rec = {"card": CARD, "eval_join_sha256": eval_join_sha, "files": {}, "dumps": {}, "fits": fits}
    for s in SEEDS:
        nm = "VGAPF_s%d" % s
        D = dumps[s]
        tab = frame_table(J, D)
        fault = faults[0] if faults and faults[0] in ("scale", "shift", "min_label", "clip60", "zero_nooutput") else None
        arr = build_arrays(J, D, tab, s, fault)
        fp = os.path.join(out_dir, nm + ".npz")
        np.savez(fp, **arr)
        m = (meta or {}).get(s) or {}
        dp, dn = m.get("dump_file", "<synthetic>"), m.get("dump_dir_name", DUMP_DIR)
        rec["dumps"][nm] = {"dump_file": dp, "dump_dir_name": dn, "dump_meta": m.get("dump_meta"), "e0": e0_stats(J, D, tab, s, expect_clips if expect_clips is not None else len(J["clip"]))}
        f = measure_file(J, D, arr, dp, dn, fp)
        f["fit_seed"] = s
        rec["files"][nm] = f
    rec["v0"] = v0_check(rec)
    rec["build"] = build_check(rec)
    rec["meta"] = meta_check(rec)
    rec["x0"] = x0_check(rec.get("fits"))
    rec["x1"] = {"PASS": bool(rec["v0"]["PASS"] and rec["build"]["PASS"] and rec["meta"]["PASS"]), "parts": ["v0", "build", "meta"]}
    with open(os.path.join(out_dir, "VISSEED_GAPFILES_RECORD.json"), "w", encoding="utf-8", newline="\n") as fh:
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


def synthetic_dump(J, seed, rng_seed=7, out_range=True):
    """a dump in the format of FIT_DEFS section 2: every frame of every clip once; the head H5 has no output at frames 0 to 3."""
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
            if out_range and f == 7:
                hg = 64.0 + c                            # a head output above 60 m, kept as dumped
            gap.append(hg)
    return {"cid": np.asarray(cid, np.int64), "frame": np.asarray(fr, np.int64), "clip_index": np.asarray(ci, np.int64), "lab_gap": np.asarray(lab, np.float64), "lab_present": np.asarray(labp, bool),
            "clips": np.asarray(names), "gap": np.asarray(gap, np.float64), "valid": np.asarray(valid, bool), "seed": np.int64(seed)}


def synthetic_all():
    J = synthetic_join()
    return J, {s: synthetic_dump(J, s) for s in SEEDS}


def synthetic_meta(dumps, root):
    """writes the synthetic dumps as root/D_H5/reader_s<k>_score.npz and returns the meta of build()"""
    os.makedirs(os.path.join(root, DUMP_DIR), exist_ok=True)
    meta = {}
    for s in SEEDS:
        p = os.path.join(root, DUMP_DIR, "reader_s%d_score.npz" % s)
        np.savez(p, **dumps[s])
        meta[s] = {"dump_file": p, "dump_dir_name": DUMP_DIR}
    return meta


def v0_faults(J, dumps):
    """each planted fault of the dumps turns v0 red -> dict name -> bool (True = turned red as it must)"""
    out = {}

    def red(mut):
        d2 = {s: dict(dumps[s]) for s in SEEDS}
        mut(d2)
        rec = {"card": CARD, "files": {}, "dumps": {}}
        for s in SEEDS:
            tab = frame_table(J, d2[s])
            rec["dumps"]["VGAPF_s%d" % s] = {"e0": e0_stats(J, d2[s], tab, s, len(J["clip"]))}
        return not v0_check(rec)["PASS"]
    out["wrong_seed"] = red(lambda d: d[2].update(seed=np.int64(3)))
    perm = np.array([1, 0] + list(range(2, len(J["clip"]))))
    inv = np.argsort(perm)
    out["clips_other_order"] = red(lambda d: d[2].update(clips=d[2]["clips"][perm], cid=inv[d[2]["cid"]], clip_index=d[2]["clip_index"]))
    out["label_gap_off_by_0.01"] = red(lambda d: d[2].update(lab_gap=d[2]["lab_gap"] + 0.01))
    keep = np.ones(len(dumps[2]["cid"]), bool)
    keep[5] = False
    out["missing_row"] = red(lambda d: d[2].update({k: np.asarray(v)[keep] for k, v in d[2].items() if k in ("cid", "frame", "clip_index", "lab_gap", "lab_present", "gap", "valid")}))
    out["duplicate_row"] = red(lambda d: d[2].update({k: np.concatenate([np.asarray(v), np.asarray(v)[:1]]) for k, v in d[2].items() if k in ("cid", "frame", "clip_index", "lab_gap", "lab_present", "gap", "valid")}))
    out["row_beyond_join"] = red(lambda d: d[2].update(frame=np.where(np.arange(len(d[2]["frame"])) == 3, 999, d[2]["frame"])))
    out["clip_index_wrong"] = red(lambda d: d[2].update(clip_index=np.where(np.arange(len(d[2]["clip_index"])) == 20, 0, d[2]["clip_index"])))
    return out


def build_faults(J, dumps, base, meta=None):
    """each planted fault of the build turns the build check red -> dict name -> bool"""
    out = {}
    for i, f in enumerate(("scale", "shift", "min_label", "clip60", "zero_nooutput")):
        rec = build(J, dumps, os.path.join(base, "f_" + f), faults=(f,), meta=meta)
        out[f] = not rec["build"]["PASS"]
    return out


def selftest():
    import tempfile
    ok = True
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_selftest_seedgap_%d" % os.getpid())
    os.makedirs(base)
    J, dumps = synthetic_all()
    fits, meta, _, _ = synthetic_fit_world(os.path.join(base, "world"), J, dumps)
    rec = build(J, dumps, os.path.join(base, "clean"), meta=meta, fits=fits)
    print("clean build: v0 %s build %s meta %s x0 %s x1 %s" % (rec["v0"]["PASS"], rec["build"]["PASS"], rec["meta"]["PASS"], rec["x0"]["PASS"], rec["x1"]["PASS"]))
    ok &= rec["v0"]["PASS"] and rec["build"]["PASS"] and rec["meta"]["PASS"] and rec["x0"]["PASS"] and rec["x1"]["PASS"]
    ok &= sorted(rec["files"]) == ["VGAPF_s2", "VGAPF_s3", "VGAPF_s4"]
    z = load_npz(os.path.join(base, "clean", "VGAPF_s2.npz"))
    ok &= bool(np.isnan(z["gap"][~z["has_out"]]).all()) and int((~z["has_out"]).sum()) >= 4 * len(J["clip"]) and z["gap"][z["has_out"]].max() > 60.0
    vf = v0_faults(J, dumps)
    bf = build_faults(J, dumps, base, meta)
    print("v0 planted faults turned red:", vf)
    print("build planted faults turned red:", bf)
    ok &= all(vf.values()) and all(bf.values())
    xf = x_faults(base, J, dumps)
    print("x0 and meta planted faults:", xf)
    ok &= all(xf.values())
    print("ZZ MAKE-SEED-GAPFILE-SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--eval-join")
    ap.add_argument("--fit-root")
    ap.add_argument("--add99-fits", default=ADD99_FITS)
    ap.add_argument("--eval-list")
    ap.add_argument("--out-dir")
    ap.add_argument("--expect-clips", type=int, default=141)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.eval_join and a.fit_root and a.out_dir):
        print("REFUSED: --eval-join, --fit-root and --out-dir are required")
        return 2
    J = load_npz(a.eval_join)
    dump_root = os.path.join(a.fit_root, "dump")
    fits, meta = collect(a.fit_root, a.add99_fits, a.eval_list or os.path.join(a.fit_root, "lists", "eval141.txt"), dump_root)
    dumps = {}
    for s in SEEDS:
        p = meta[s]["dump_file"]
        if not os.path.isfile(p):
            print("REFUSED: no dump %s" % p)
            return 2
        dumps[s] = load_npz(p)
    rec = build(J, dumps, a.out_dir, meta=meta, expect_clips=a.expect_clips, eval_join_sha=sha_file(a.eval_join), fits=fits)
    P = lambda k: "PASS" if rec[k]["PASS"] else "FAIL"                                                    # noqa: E731
    print("ZZ SEED-GAPFILES x0 %s x1 %s (v0 %s build %s meta %s)" % (P("x0"), P("x1"), P("v0"), P("build"), P("meta")))
    for k in ("x0", "v0", "build", "meta"):
        for x in rec[k]["failures"][:6]:
            print(k, x)
    return 0 if rec["x0"]["PASS"] and rec["x1"]["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
