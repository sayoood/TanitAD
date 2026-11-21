"""make_ens_gapfile.py -- the gap file of card PREREG_V8F_PROOF_ADD105 (VISENS10K): VGAPE5.npz, the arithmetic mean of the five dumped gaps of the head on five frames (D-H5) of the fit seeds 0, 1, 2, 3, 4,
on every row of the evaluation join where all five dumps have a finite gap, no output elsewhere. Control e0 of the card and the T0 diagnostic of card section 3 go into VISENS_GAPFILES_RECORD.json.
numpy and the standard library only (runs on the pod with /workspace/venv/bin/python).
Usage: python make_ens_gapfile.py --eval-join NPZ --out-dir NEWDIR [--old-dump-root DIR] [--new-dump-root DIR] [--rec102 JSON] [--rec104 JSON] [--expect-clips 141]
         reads <old-dump-root>/D_H5/reader_s<k>_score.npz for k = 0, 1 (default /workspace/v8proof/cropread/run1/dump) and <new-dump-root>/D_H5/reader_s<S>_score.npz for S = 2, 3, 4 (default
         /workspace/v8proof/visseed10k/fitroot/dump); the gap-file record of card ADD102 (default /workspace/v8proof/visrule10k/gapfiles/VISRULE_GAPFILES_RECORD.json) and that of card ADD104
         (default /workspace/v8proof/visseed10k/gapfiles/VISSEED_GAPFILES_RECORD.json); writes NEWDIR/VGAPE5.npz and NEWDIR/VISENS_GAPFILES_RECORD.json. Refuses an existing out dir.
       python make_ens_gapfile.py --selftest     synthetic join and five dumps; the clean build passes e0, each planted fault of each clause turns it red
Gap-file layout = the layout of the gap files of card ADD104 (make_seed_gapfile.py, from make_vis_gapfile.py of card ADD102): one row per frame of the evaluation join (every clip, frames 0 to n-1, in the order of
the join's clips): clip_index (int64), frame (int64), gap (float64, metres, NaN where there is no output), has_out (bool), clips (the join's clip names), seed (int64; -1 here: the file is not one fit seed's).
The gap is NOT mapped, NOT clipped and NOT combined with a minimum; the dump's `valid` field is not used. A row has an output exactly when all five dumps have a finite gap on it.
Control e0 (card section 4), clause by clause: (a) the dumps of fit seeds 0 and 1 have the sha256 that the record of card ADD102 names for them (files.VGAP_s<k>.dump_sha256); (b) the dumps of fit seeds 2, 3, 4
have the sha256 that the record of card ADD104 names (files.VGAPF_s<S>.dump_sha256) and that record reads x0 PASS and x1 PASS; (c) the five dumps are on the same rows (clip name and frame; every row once;
every row has its join row); (d) the saved gap equals a second routine's mean (plain loops over the five dumps alone) on every row within 1e-6 m; (e) a row has an output exactly when all five dumps have one."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

CARD = "PREREG_V8F_PROOF_ADD105"
FITS = (0, 1, 2, 3, 4)
OLD, NEW = (0, 1), (2, 3, 4)
DUMP_DIR = "D_H5"
FILE_NAME = "VGAPE5"
RECORD_NAME = "VISENS_GAPFILES_RECORD.json"
TOL_SAVED_M = 1e-6
OLD_ROOT = "/workspace/v8proof/cropread/run1/dump"
NEW_ROOT = "/workspace/v8proof/visseed10k/fitroot/dump"
REC102 = "/workspace/v8proof/visrule10k/gapfiles/VISRULE_GAPFILES_RECORD.json"
REC104 = "/workspace/v8proof/visseed10k/gapfiles/VISSEED_GAPFILES_RECORD.json"
T0_TIER = "T0 (a probe on logged frames, a diagnostic of a reader; not driving performance)"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def frame_table(J, D):
    """join row of every dump row. -> dict(ji [rows] join clip index or -1, row [rows] join row or -1, inrange bool)"""
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
    return {"ji": ji, "row": row, "inrange": inrange}


def on_join_rows(J, D, key):
    """the dump's field `key` laid on the join rows (NaN or False where the dump has no row)"""
    nrow = int(np.asarray(J["n"], np.int64).sum())
    tab = frame_table(J, D)
    ok = tab["inrange"]
    v = np.asarray(D[key])
    if v.dtype == bool:
        out = np.zeros(nrow, bool)
    else:
        out = np.full(nrow, np.nan)
    out[tab["row"][ok]] = v[ok]
    return out


def build_arrays(J, dumps, fault=None):
    """-> the arrays of the gap file. One row per join frame; gap = arithmetic mean of the five dumped gaps where all five are finite, NaN elsewhere."""
    n = np.asarray(J["n"], np.int64)
    names = [str(c) for c in J["clip"]]
    ci = np.repeat(np.arange(len(n), dtype=np.int64), n)
    fr = np.concatenate([np.arange(k, dtype=np.int64) for k in n]) if len(n) else np.zeros(0, np.int64)
    G = np.stack([on_join_rows(J, dumps[s], "gap").astype(np.float64) for s in FITS], 0)
    allfin = np.isfinite(G).all(axis=0)
    gap = np.full(G.shape[1], np.nan)
    gap[allfin] = G[:, allfin].mean(axis=0)
    has = allfin.copy()
    if fault == "offset":
        gap = gap + 5e-6                                                           # 5e-6 m: above the bar of 1e-6
    elif fault == "mean_of_four":
        gap = np.where(allfin, G[:4].mean(axis=0), np.nan)
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


def second_routine(J, dumps):
    """plain loops over the five dumps alone -> {(join clip index, frame): mean gap or None}; None where a dump has no row or no finite gap on that row. Written apart from build_arrays."""
    where = {str(c): i for i, c in enumerate(J["clip"])}
    per = []
    for s in FITS:
        D = dumps[s]
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
                out[(i, f)] = t / 5.0
            else:
                out[(i, f)] = None
    return out


def row_keys(D):
    """sorted list of (clip name, frame) of a dump"""
    cl = [str(c) for c in D["clips"]]
    return sorted((cl[int(c)], int(f)) for c, f in zip(D["cid"], D["frame"]))


def facts_of(J, dumps, shas, rec102, rec104, arr):
    """the facts control e0 reads. dumps {fit seed: dict}; shas {fit seed: sha256 of the dump file}; rec102, rec104: the two records (dicts or None); arr: the arrays of the saved gap file."""
    f = {"hash_named": {}, "same_rows": {}, "saved": {}}
    for s in FITS:
        rec, key = (rec102, "VGAP_s%d" % s) if s in OLD else (rec104, "VGAPF_s%d" % s)
        named = ((((rec or {}).get("files") or {}).get(key)) or {}).get("dump_sha256")
        f["hash_named"]["fit%d" % s] = {"dump_sha256": shas.get(s), "named_by_record": named, "record_of_card": "ADD102" if s in OLD else "ADD104",
                                        "equal": bool(isinstance(named, str) and len(named) == 64 and named == shas.get(s))}
    r4 = rec104 or {}
    f["add104_record"] = {"x0_PASS": (r4.get("x0") or {}).get("PASS") is True, "x1_PASS": (r4.get("x1") or {}).get("PASS") is True}
    nj = {str(c): int(k) for c, k in zip(J["clip"], np.asarray(J["n"], np.int64))}
    base = None
    for s in FITS:
        D = dumps[s]
        ks = row_keys(D)
        if base is None:
            base = ks
        f["same_rows"]["fit%d" % s] = {"n_rows": len(ks), "n_duplicate_rows": int(len(ks) - len(set(ks))), "n_rows_not_in_join": int(sum(1 for c, fr in ks if c not in nj or not (0 <= fr < nj[c]))),
                                       "same_set_as_fit0": bool(len(ks) == len(base) and ks == base)}
    n = np.asarray(J["n"], np.int64)
    ref = second_routine(J, dumps)
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
    n_has_ref = int(sum(1 for v in ref.values() if v is not None))
    f["saved"] = {"n_rows": int(len(ci)), "n_join_rows": int(n.sum()), "rows_cover_the_join_in_order": bool(len(ci) == int(n.sum()) and np.array_equal(ci, np.repeat(np.arange(len(n)), n))),
                  "max_abs_dev_saved_minus_second_routine_m": float(dev), "n_rows_without_second_routine_row": n_missing, "n_output_flag_mismatch": n_flag,
                  "n_with_output": int(has.sum()), "n_with_output_second_routine": n_has_ref, "n_without_output": int((~has).sum()),
                  "saved_gap_is_nan_exactly_where_flag_false": bool(np.array_equal(np.isfinite(gap), has))}
    return f


def e0_check(f):
    """control e0 (card section 4) from the facts. -> {"PASS", "failures", "clauses"}"""
    bad, cl = [], {}
    try:
        a = [k for k in ("fit0", "fit1") if not f["hash_named"][k]["equal"]]
        b = [k for k in ("fit2", "fit3", "fit4") if not f["hash_named"][k]["equal"]]
        cl["a_dumps_of_fit_seeds_0_1_have_the_hashes_of_the_record_of_card_ADD102"] = not a
        cl["b_dumps_of_fit_seeds_2_3_4_have_the_hashes_of_the_record_of_card_ADD104"] = not b
        cl["b_record_of_card_ADD104_reads_x0_and_x1_PASS"] = bool(f["add104_record"]["x0_PASS"] and f["add104_record"]["x1_PASS"])
        if a:
            bad.append("dump hash is not the one the record of card ADD102 names: %s" % a)
        if b:
            bad.append("dump hash is not the one the record of card ADD104 names: %s" % b)
        if not cl["b_record_of_card_ADD104_reads_x0_and_x1_PASS"]:
            bad.append("the record of card ADD104 does not read x0 PASS and x1 PASS: %s" % f["add104_record"])
        sr = f["same_rows"]
        ok_rows = all(v["same_set_as_fit0"] and v["n_duplicate_rows"] == 0 and v["n_rows_not_in_join"] == 0 and v["n_rows"] == sr["fit0"]["n_rows"] for v in sr.values())
        cl["c_the_five_dumps_are_on_the_same_rows"] = ok_rows
        if not ok_rows:
            bad.append("the five dumps are not on the same rows (each once, each in the join): %s" % {k: v for k, v in sr.items() if not (v["same_set_as_fit0"] and v["n_duplicate_rows"] == 0 and v["n_rows_not_in_join"] == 0)})
        sv = f["saved"]
        dev = sv["max_abs_dev_saved_minus_second_routine_m"]
        cl["d_saved_gap_equals_the_second_routine_within_1e-6_m"] = bool(dev is not None and dev <= TOL_SAVED_M and sv["n_rows_without_second_routine_row"] == 0)
        if not cl["d_saved_gap_equals_the_second_routine_within_1e-6_m"]:
            bad.append("saved gap: largest deviation from the second routine %s (bar %g), %s rows without a second-routine row" % (dev, TOL_SAVED_M, sv["n_rows_without_second_routine_row"]))
        cl["e_output_flag_exactly_when_all_five_have_one"] = bool(sv["n_output_flag_mismatch"] == 0 and sv["saved_gap_is_nan_exactly_where_flag_false"] and sv["n_with_output"] == sv["n_with_output_second_routine"])
        if not cl["e_output_flag_exactly_when_all_five_have_one"]:
            bad.append("output flag: %s rows differ from the five dumps (%s with output saved, %s by the second routine)" % (sv["n_output_flag_mismatch"], sv["n_with_output"], sv["n_with_output_second_routine"]))
        if not (sv["rows_cover_the_join_in_order"] and sv["n_rows"] == sv["n_join_rows"]):
            bad.append("the rows of the saved file do not cover the join once in order")
        if not sv["n_with_output"] > 0:
            bad.append("no frame has an output of all five dumps")
    except (KeyError, TypeError) as ex:
        bad.append("facts incomplete: %s: %s" % (type(ex).__name__, ex))
    return {"PASS": not bad, "failures": bad, "clauses": cl}


def t0_gap_error(J, dumps):
    """T0 diagnostic (card section 3): on the rows with a label lead (dump fields lab_gap, lab_present) and an output of all five dumps, the mean absolute gap error of the mean gap and of each single fit.
    The label fields are read from the dump of fit seed 0 and compared with the other four (n of rows where they differ is reported)."""
    G = np.stack([on_join_rows(J, dumps[s], "gap").astype(np.float64) for s in FITS], 0)
    lab = on_join_rows(J, dumps[0], "lab_gap").astype(np.float64)
    labp = on_join_rows(J, dumps[0], "lab_present")
    n_lab_differs = 0
    for s in FITS[1:]:
        l2, p2 = on_join_rows(J, dumps[s], "lab_gap").astype(np.float64), on_join_rows(J, dumps[s], "lab_present")
        n_lab_differs += int((~np.isclose(l2, lab, equal_nan=True, rtol=0, atol=0)).sum() + (p2 != labp).sum())
    sel = labp & np.isfinite(lab) & np.isfinite(G).all(axis=0)
    out = {"tier": T0_TIER, "n_rows_with_label_lead_and_output_of_all_five": int(sel.sum()), "n_rows_label_fields_differ_between_dumps": n_lab_differs,
           "mean_abs_gap_error_m": {"mean_of_five": None, "fit0": None, "fit1": None, "fit2": None, "fit3": None, "fit4": None}}
    if sel.any():
        out["mean_abs_gap_error_m"]["mean_of_five"] = float(np.abs(G[:, sel].mean(axis=0) - lab[sel]).mean())
        for i, s in enumerate(FITS):
            out["mean_abs_gap_error_m"]["fit%d" % s] = float(np.abs(G[i, sel] - lab[sel]).mean())
    return out


def build(J, dumps, shas, rec102, rec104, out_dir, fault=None, eval_join_sha=None, meta=None):
    """writes VGAPE5.npz and the record; -> record. fault: name of a planted fault (selftest and tests). meta: paths and record hashes for the record."""
    if os.path.exists(out_dir):
        raise SystemExit("REFUSED: %s exists" % out_dir)
    os.makedirs(out_dir)
    arr = build_arrays(J, dumps, fault)
    fp = os.path.join(out_dir, FILE_NAME + ".npz")
    np.savez(fp, **arr)
    f = facts_of(J, dumps, shas, rec102, rec104, arr)
    e0 = e0_check(f)
    rec = {"card": CARD, "eval_join_sha256": eval_join_sha, "files": {FILE_NAME: {"file": FILE_NAME + ".npz", "file_sha256": sha_file(fp), "n_rows": int(len(arr["clip_index"])),
                                                                                  "n_with_output": int(arr["has_out"].sum()), "n_without_output": int((~arr["has_out"]).sum()),
                                                                                  "dump_sha256": {"fit%d" % s: shas.get(s) for s in FITS}}},
           "inputs": (meta or {}), "facts": f, "e0": e0}
    if e0["PASS"]:
        rec["t0_gap_error_diagnostic"] = t0_gap_error(J, dumps)
    with open(os.path.join(out_dir, RECORD_NAME), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def dump_path(root, s):
    return os.path.join(root, DUMP_DIR, "reader_s%d_score.npz" % s)


def load_inputs(old_root, new_root, rec102_path, rec104_path):
    """-> (dumps, shas, rec102, rec104, meta) read from disk; raises SystemExit(REFUSED) on a missing file"""
    dumps, shas, paths = {}, {}, {}
    for s in FITS:
        p = dump_path(old_root if s in OLD else new_root, s)
        if not os.path.isfile(p):
            raise SystemExit("REFUSED: no dump %s" % p)
        dumps[s], shas[s], paths["fit%d" % s] = load_npz(p), sha_file(p), p
    recs = []
    for p in (rec102_path, rec104_path):
        if not os.path.isfile(p):
            raise SystemExit("REFUSED: no record %s" % p)
        recs.append(json.load(open(p, encoding="utf-8")))
    meta = {"dump_files": paths, "record_add102": {"file": rec102_path, "sha256": sha_file(rec102_path)}, "record_add104": {"file": rec104_path, "sha256": sha_file(rec104_path)}}
    return dumps, shas, recs[0], recs[1], meta


def check_now(eval_join, gap_file, old_root, new_root, rec102_path, rec104_path):
    """the reader's second look: facts and e0 read again from disk now, on the gap file as it lies on disk. -> (e0 dict, facts, shas, file sha256)"""
    J = load_npz(eval_join)
    dumps, shas, r102, r104, meta = load_inputs(old_root, new_root, rec102_path, rec104_path)
    arr = load_npz(gap_file)
    f = facts_of(J, dumps, shas, r102, r104, arr)
    return e0_check(f), f, shas, sha_file(gap_file), meta


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
    """a dump in the format of cropread/fit/FIT_DEFS.md section 2: every frame of every clip once; the head H5 has no output at frames 0 to 3; one more NaN per clip at frame 9 for odd fit seeds"""
    rng = np.random.RandomState(rng_seed + seed)
    names = [str(c) for c in J["clip"]]
    cid, fr, ci, lab, labp, valid, gap = [], [], [], [], [], [], []
    for c, n in enumerate(np.asarray(J["n"], np.int64)):
        for f in range(int(n)):
            r = int(J["off"][c]) + f
            g = float(J["lead_gap"][r])
            ok = int(J["lead_valid"][r]) == 1 and np.isfinite(g) and 0.0 <= g <= 60.0
            cid.append(c)
            fr.append(f)
            ci.append(c)
            lab.append(g if ok else np.nan)
            labp.append(ok)
            valid.append(bool(rng.rand() < 0.9))
            hg = np.nan if f < 4 else (g if np.isfinite(g) else 20.0) + rng.randn() * 2.0
            if f == 9 and seed % 2 == 1:
                hg = np.nan
            if f == 7:
                hg = 64.0 + c + 0.1 * seed                                         # above 60 m, kept as dumped
            gap.append(hg)
    return {"cid": np.asarray(cid, np.int64), "frame": np.asarray(fr, np.int64), "clip_index": np.asarray(ci, np.int64), "lab_gap": np.asarray(lab, np.float64), "lab_present": np.asarray(labp, bool),
            "clips": np.asarray(names), "gap": np.asarray(gap, np.float64), "valid": np.asarray(valid, bool), "seed": np.int64(seed)}


def synthetic_world(base=None):
    """-> (J, dumps, shas, rec102, rec104); the shas are of synthetic files written under base when given (else of the arrays' bytes)"""
    J = synthetic_join()
    dumps = {s: synthetic_dump(J, s) for s in FITS}
    shas = {}
    for s in FITS:
        if base:
            os.makedirs(os.path.join(base, "old" if s in OLD else "new", DUMP_DIR), exist_ok=True)
            p = dump_path(os.path.join(base, "old" if s in OLD else "new"), s)
            np.savez(p, **dumps[s])
            shas[s] = sha_file(p)
        else:
            shas[s] = hashlib.sha256(np.asarray(dumps[s]["gap"]).tobytes() + bytes([s])).hexdigest()
    rec102 = {"files": {"VGAP_s%d" % s: {"dump_sha256": shas[s]} for s in OLD}}
    rec104 = {"files": {"VGAPF_s%d" % s: {"dump_sha256": shas[s]} for s in NEW}, "x0": {"PASS": True}, "x1": {"PASS": True}}
    return J, dumps, shas, rec102, rec104


def e0_of(J, dumps, shas, r102, r104, fault=None):
    arr = build_arrays(J, dumps, fault)
    return e0_check(facts_of(J, dumps, shas, r102, r104, arr))


def planted_faults(J, dumps, shas, r102, r104):
    """each planted fault of each clause of e0 turns it red -> dict name -> bool (True = turned red as it must)"""
    import copy
    out = {}

    def with_dumps(mut):
        d2 = {s: dict(dumps[s]) for s in FITS}
        mut(d2)
        return not e0_of(J, d2, shas, r102, r104)["PASS"]
    cols = ("cid", "frame", "clip_index", "lab_gap", "lab_present", "gap", "valid")
    # clause a, b: the hash named by the records
    for s in FITS:
        r1, r4 = copy.deepcopy(r102), copy.deepcopy(r104)
        if s in OLD:
            r1["files"]["VGAP_s%d" % s]["dump_sha256"] = "0" * 64
        else:
            r4["files"]["VGAPF_s%d" % s]["dump_sha256"] = "0" * 64
        out["record_names_another_hash_fit%d" % s] = not e0_of(J, dumps, shas, r1, r4)["PASS"]
    out["dump_file_changed_fit0"] = not e0_of(J, dumps, {**shas, 0: "1" * 64}, r102, r104)["PASS"]
    out["dump_file_changed_fit4"] = not e0_of(J, dumps, {**shas, 4: "1" * 64}, r102, r104)["PASS"]
    r4 = copy.deepcopy(r104)
    r4["x0"]["PASS"] = False
    out["add104_x0_not_pass"] = not e0_of(J, dumps, shas, r102, r4)["PASS"]
    r4 = copy.deepcopy(r104)
    r4["x1"]["PASS"] = False
    out["add104_x1_not_pass"] = not e0_of(J, dumps, shas, r102, r4)["PASS"]
    r4 = copy.deepcopy(r104)
    del r4["x1"]
    out["add104_x1_missing"] = not e0_of(J, dumps, shas, r102, r4)["PASS"]
    out["no_record_of_add102"] = not e0_of(J, dumps, shas, None, r104)["PASS"]
    out["no_record_of_add104"] = not e0_of(J, dumps, shas, r102, None)["PASS"]
    # clause c: the same rows
    keep = np.ones(len(dumps[3]["cid"]), bool)
    keep[5] = False
    out["a_row_missing_in_fit3"] = with_dumps(lambda d: d[3].update({k: np.asarray(v)[keep] for k, v in d[3].items() if k in cols}))
    out["a_row_duplicated_in_fit2"] = with_dumps(lambda d: d[2].update({k: np.concatenate([np.asarray(v), np.asarray(v)[:1]]) for k, v in d[2].items() if k in cols}))
    out["a_frame_moved_in_fit4"] = with_dumps(lambda d: d[4].update(frame=np.where(np.arange(len(d[4]["frame"])) == 6, 5, d[4]["frame"])))
    out["a_clip_renamed_in_fit1"] = with_dumps(lambda d: d[1].update(clips=np.asarray(["x" + str(c) if i == 0 else str(c) for i, c in enumerate(d[1]["clips"])])))
    out["a_row_beyond_the_join_in_fit0"] = with_dumps(lambda d: d[0].update(frame=np.where(np.arange(len(d[0]["frame"])) == 3, 999, d[0]["frame"])))
    # clauses d, e: the saved gap and the flag
    for fault in ("offset", "mean_of_four", "median", "mean_of_available", "flag_all_true", "zero_nooutput", "flag_false_on_a_row"):
        out["build_" + fault] = not e0_of(J, dumps, shas, r102, r104, fault)["PASS"]
    return out


def selftest():
    ok = True
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_selftest_ens_%d" % os.getpid())
    os.makedirs(base)
    J, dumps, shas, r102, r104 = synthetic_world(os.path.join(base, "dumps"))
    rec = build(J, dumps, shas, r102, r104, os.path.join(base, "clean"), eval_join_sha="9" * 64)
    print("clean build: e0 %s %s" % (rec["e0"]["PASS"], rec["e0"]["clauses"]))
    ok &= rec["e0"]["PASS"] and "t0_gap_error_diagnostic" in rec
    z = load_npz(os.path.join(base, "clean", FILE_NAME + ".npz"))
    G = np.stack([dumps[s]["gap"] for s in FITS], 0)
    ok &= bool(np.isnan(z["gap"][~z["has_out"]]).all()) and int((~z["has_out"]).sum()) >= 4 * len(J["clip"]) and bool(z["has_out"].any()) and float(z["gap"][z["has_out"]].max()) > 60.0
    ok &= int(z["seed"]) == -1 and sorted(z.keys()) == sorted(["clip_index", "frame", "gap", "has_out", "clips", "seed"])
    pf = planted_faults(J, dumps, shas, r102, r104)
    for k, v in sorted(pf.items()):
        print("planted fault %-40s turned red: %s" % (k, v))
    ok &= all(pf.values())
    print("ZZ MAKE-ENS-GAPFILE-SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--eval-join")
    ap.add_argument("--out-dir")
    ap.add_argument("--old-dump-root", default=OLD_ROOT)
    ap.add_argument("--new-dump-root", default=NEW_ROOT)
    ap.add_argument("--rec102", default=REC102)
    ap.add_argument("--rec104", default=REC104)
    ap.add_argument("--expect-clips", type=int, default=141)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.eval_join and a.out_dir):
        print("REFUSED: --eval-join and --out-dir are required")
        return 2
    J = load_npz(a.eval_join)
    if len(J["clip"]) != a.expect_clips:
        print("REFUSED: the join has %d clips, expected %d" % (len(J["clip"]), a.expect_clips))
        return 2
    dumps, shas, r102, r104, meta = load_inputs(a.old_dump_root, a.new_dump_root, a.rec102, a.rec104)
    rec = build(J, dumps, shas, r102, r104, a.out_dir, eval_join_sha=sha_file(a.eval_join), meta=meta)
    print("ZZ ENS-GAPFILE e0 %s" % ("PASS" if rec["e0"]["PASS"] else "FAIL"))
    for x in rec["e0"]["failures"][:8]:
        print("e0", x)
    return 0 if rec["e0"]["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
