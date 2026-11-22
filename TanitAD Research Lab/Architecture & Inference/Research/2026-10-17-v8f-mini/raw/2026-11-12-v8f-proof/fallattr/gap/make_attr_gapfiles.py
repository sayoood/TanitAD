"""make_attr_gapfiles.py -- the four gap files of card PREREG_V8F_PROOF_ADD110 (FALLATTR10K) with control a0.

  VGAPA_LA.npz  the label gap on every label row; the base file elsewhere
  VGAPA_LN.npz  the label gap on the label rows with a label gap below 20 m; the base file elsewhere
  VGAPA_LM.npz  the label gap on the label rows with a label gap from 20 m to below 40 m; the base file elsewhere
  VGAPA_LF.npz  the label gap on the label rows with a label gap of 40 m and above; the base file elsewhere
Label rows of the evaluation join: lead_valid == 1 and lead_gap finite and 0 <= lead_gap <= 60. On a row of its range a file carries the label gap and has_out True (whether or not the base has an
output there); on every other row it carries gap and has_out of the base file laid on the join rows. Nothing is smoothed, clipped or blended. The label gap enters as a tool of attribution only.
Array layout of the base file (card ADD106): clip_index int64, frame int64, gap float64 (NaN where no output), has_out bool, clips (the join's clip names), seed int64 (-1 for a composed file). No other field.
Usage: python make_attr_gapfiles.py --eval-join NPZ --base NPZ --base-record JSON --out-dir NEWDIR    (refuses an existing out dir; exit 0 only when pass_a0 is true)
       python make_attr_gapfiles.py --selftest
Control a0 clauses: second_routine, ranges_partition, outside_is_base, base_is_recorded, layout. Record: NEWDIR/FALLATTR_GAPFILES_RECORD.json.
numpy and the standard library only."""
import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np

GAP_MAX_M = 60.0
NEAR_M, MID_M = 20.0, 40.0
ARMS = ("LA", "LN", "LM", "LF")
FIELDS = ("clip_index", "frame", "gap", "has_out", "clips", "seed")
DTYPES = {"clip_index": "int64", "frame": "int64", "gap": "float64", "has_out": "bool", "seed": "int64"}
RECORD_NAME = "FALLATTR_GAPFILES_RECORD.json"
TOL_M = 1e-9
CLAUSES = ("second_routine", "ranges_partition", "outside_is_base", "base_is_recorded", "layout")


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def fname(arm):
    return "VGAPA_%s.npz" % arm


# ============================================================================================ the join frames and the masks
def join_frames(J):
    """-> names, n, cum (file row of the first frame of each clip), idx (join row of every file row), ci, fr"""
    names = [str(c) for c in J["clip"]]
    n = np.asarray(J["n"], np.int64)
    off = np.asarray(J["off"], np.int64)
    ci = np.repeat(np.arange(len(n), dtype=np.int64), n)
    fr = np.concatenate([np.arange(k, dtype=np.int64) for k in n]) if len(n) else np.zeros(0, np.int64)
    idx = off[ci] + fr
    cum = np.concatenate([[0], np.cumsum(n)[:-1]]).astype(np.int64) if len(n) else np.zeros(0, np.int64)
    return names, n, cum, idx, ci, fr


def label_arrays(J, idx):
    g = np.asarray(J["lead_gap"], np.float64)[idx]
    v = np.asarray(J["lead_valid"])[idx]
    lab = (v == 1) & np.isfinite(g) & (g >= 0.0) & (g <= GAP_MAX_M)
    return g, lab


def range_masks(g, lab, fault=None):
    """the row mask of each arm. fault: planted fault of the ranges (selftest and tests)."""
    with np.errstate(invalid="ignore"):
        m = {"LA": lab.copy(), "LN": lab & (g < NEAR_M), "LM": lab & (g >= NEAR_M) & (g < MID_M), "LF": lab & (g >= MID_M)}
        if fault == "lm_incl_40":
            m["LM"] = lab & (g >= NEAR_M) & (g <= MID_M)
        elif fault == "lf_drop_last":
            w = np.flatnonzero(m["LF"])
            if len(w):
                m["LF"][w[-1]] = False
    return m


def lay_base(names, n, cum, B):
    """gap and has_out of the base file laid on the join rows by (clip name, frame); NaN and False where the base has no row"""
    nrow = int(n.sum())
    gap = np.full(nrow, np.nan)
    has = np.zeros(nrow, bool)
    where = {c: i for i, c in enumerate(names)}
    bn = [str(c) for c in B["clips"]]
    bj = np.array([where.get(bn[int(c)], -1) for c in B["clip_index"]], np.int64) if len(B["clip_index"]) else np.zeros(0, np.int64)
    bf = np.asarray(B["frame"], np.int64)
    jc = np.where(bj >= 0, bj, 0)
    ok = (bj >= 0) & (bf >= 0) & (bf < n[jc]) if len(bj) else np.zeros(0, bool)
    rows = cum[jc] + bf
    gap[rows[ok]] = np.asarray(B["gap"], np.float64)[ok]
    has[rows[ok]] = np.asarray(B["has_out"], bool)[ok]
    return gap, has


def count_source_changes(clip_index, mask):
    """number of consecutive frame pairs inside one clip where exactly one of the two frames carries the label gap"""
    ci, m = np.asarray(clip_index), np.asarray(mask, bool)
    if len(m) < 2:
        return 0
    return int(((m[1:] != m[:-1]) & (ci[1:] == ci[:-1])).sum())


def compose(J, B, fault=None):
    """-> dict(names, n, ci, fr, g, lab, masks, base_gap, base_has, arrays{arm: dict})"""
    names, n, cum, idx, ci, fr = join_frames(J)
    g, lab = label_arrays(J, idx)
    masks = range_masks(g, lab, fault)
    bg, bh = lay_base(names, n, cum, B)
    arrays = {}
    for arm in ARMS:
        m = masks[arm]
        lg = g.copy()
        if fault == "offset_label" and arm == "LN":
            lg = lg + 1e-6
        gap = np.where(m, lg, bg)
        has = np.where(m, True, bh)
        if fault == "outside_zero":
            gap = np.where(m, gap, np.where(np.isfinite(gap), gap, 0.0))
        arr = {"clip_index": ci.copy(), "frame": fr.copy(), "gap": gap.astype(np.float32) if fault == "f32gap" else gap, "has_out": has, "clips": np.asarray(names), "seed": np.int64(0 if fault == "seed0" else -1)}
        if fault == "extra_field":
            arr["valid"] = has.copy()
        arrays[arm] = arr
    return {"names": names, "n": n, "ci": ci, "fr": fr, "g": g, "lab": lab, "masks": masks, "base_gap": bg, "base_has": bh, "arrays": arrays}


# ============================================================================================ the clauses
def second_routine_clause(J, B, files):
    """plain loops over the raw join and the raw base, written apart from compose(); compares each file reloaded from disk with the expected value of every row"""
    off, n = J["off"], J["n"]
    names = [str(c) for c in J["clip"]]
    bkey = {}
    bn = [str(c) for c in B["clips"]]
    for r in range(len(B["frame"])):
        bkey[(bn[int(B["clip_index"][r])], int(B["frame"][r]))] = (float(B["gap"][r]), bool(B["has_out"][r]))
    exp = {a: {} for a in ARMS}
    nlab = 0
    for c, nm in enumerate(names):
        for f in range(int(n[c])):
            gg = float(J["lead_gap"][int(off[c]) + f])
            vv = int(J["lead_valid"][int(off[c]) + f])
            label = (vv == 1) and (gg == gg) and (gg >= 0.0) and (gg <= GAP_MAX_M)
            nlab += 1 if label else 0
            inr = {"LA": label, "LN": label and gg < 20.0, "LM": label and gg >= 20.0 and gg < 40.0, "LF": label and gg >= 40.0}
            for a in ARMS:
                exp[a][(nm, f)] = (gg, True) if inr[a] else bkey.get((nm, f), (float("nan"), False))
    facts, ok = {}, True
    for a in ARMS:
        z = load_npz(files[a])
        fn = [str(c) for c in z["clips"]]
        worst, nan_eq, has_eq, seen = 0.0, True, True, set()
        for r in range(len(z["frame"])):
            key = (fn[int(z["clip_index"][r])], int(z["frame"][r]))
            seen.add(key)
            e = exp[a].get(key)
            if e is None:
                nan_eq = False
                continue
            fg, fh = float(z["gap"][r]), bool(z["has_out"][r])
            if (fg != fg) != (e[0] != e[0]):
                nan_eq = False
            elif fg == fg:
                worst = max(worst, abs(fg - e[0]))
            if fh != e[1]:
                has_eq = False
        rows_ok = len(z["frame"]) == len(exp[a]) and seen == set(exp[a].keys())
        f_ok = rows_ok and nan_eq and has_eq and worst <= TOL_M
        ok = ok and f_ok
        facts[a] = {"pass": bool(f_ok), "largest_abs_gap_difference_m": worst, "has_out_equal": has_eq, "nan_pattern_equal": nan_eq, "same_rows": rows_ok}
    return {"pass": bool(ok), "tolerance_m": TOL_M, "independent_label_rows": nlab, "arms": facts}


def ranges_partition_clause(C):
    m, g = C["masks"], C["g"]
    pairs = (("LN", "LM"), ("LN", "LF"), ("LM", "LF"))
    overlap = {a + "_" + b: int((m[a] & m[b]).sum()) for a, b in pairs}
    union = m["LN"] | m["LM"] | m["LF"]
    union_eq = bool(np.array_equal(union, m["LA"]))
    la_eq_label = bool(np.array_equal(m["LA"], C["lab"]))
    in_range = bool((g[m["LN"]] < NEAR_M).all() and ((g[m["LM"]] >= NEAR_M) & (g[m["LM"]] < MID_M)).all() and (g[m["LF"]] >= MID_M).all())
    ok = all(v == 0 for v in overlap.values()) and union_eq and la_eq_label and in_range
    return {"pass": bool(ok), "pairwise_overlap_rows": overlap, "union_equals_la": union_eq, "la_equals_label_rows": la_eq_label, "rows_inside_their_ranges": in_range,
            "n_label_rows": {a: int(m[a].sum()) for a in ARMS}}


def outside_is_base_clause(C, files):
    facts, ok = {}, True
    for a in ARMS:
        z = load_npz(files[a])
        off_rows = ~C["masks"][a]
        if len(z["gap"]) != len(off_rows):
            facts[a] = {"pass": False, "reason": "row count"}
            ok = False
            continue
        gd = np.array_equal(np.asarray(z["gap"], np.float64)[off_rows], C["base_gap"][off_rows], equal_nan=True)
        hd = np.array_equal(np.asarray(z["has_out"], bool)[off_rows], C["base_has"][off_rows])
        facts[a] = {"pass": bool(gd and hd), "gap_equal": bool(gd), "has_out_equal": bool(hd), "n_rows_off_range": int(off_rows.sum())}
        ok = ok and facts[a]["pass"]
    return {"pass": bool(ok), "arms": facts}


def find_hash(rec, h):
    out = []
    for k, v in (rec.get("files") or {}).items():
        if isinstance(v, dict) and v.get("file_sha256") == h:
            out.append(k)
    return out


def base_is_recorded_clause(base_path, base_rec_path):
    h = sha_file(base_path)
    with open(base_rec_path, encoding="utf-8") as fh:
        rec = json.load(fh)
    named = find_hash(rec, h)
    return {"pass": bool(named), "base_sha256": h, "named_in_record_as": named, "record": os.path.basename(base_rec_path)}


def layout_clause(C, B, files):
    facts, ok = {}, True
    base_fields = sorted(B.keys())
    base_rows = bool(len(B.get("frame", [])) == len(C["ci"]) and np.array_equal(B.get("clip_index"), C["ci"]) and np.array_equal(B.get("frame"), C["fr"]) and [str(c) for c in B.get("clips", [])] == C["names"])
    for a in ARMS:
        z = load_npz(files[a])
        why = []
        if not base_rows:
            why.append("the base file has not the rows and clips of the join")
        if sorted(z.keys()) != sorted(FIELDS):
            why.append("fields %s" % sorted(z.keys()))
        if sorted(z.keys()) != base_fields:
            why.append("fields differ from the base file")
        for k, d in DTYPES.items():
            if k in z and str(z[k].dtype) != d:
                why.append("dtype of %s is %s" % (k, z[k].dtype))
            if k in z and k in B and str(z[k].dtype) != str(B[k].dtype):
                why.append("dtype of %s differs from the base" % k)
        if "clips" in z and z["clips"].dtype.kind != "U":
            why.append("clips dtype")
        if "seed" in z and (z["seed"].ndim != 0 or int(z["seed"]) != -1):
            why.append("seed is not -1")
        if not why:
            if not (np.array_equal(z["clip_index"], C["ci"]) and np.array_equal(z["frame"], C["fr"]) and [str(c) for c in z["clips"]] == C["names"]):
                why.append("rows or clips differ from the join")
        facts[a] = {"pass": not why, "why": why}
        ok = ok and not why
    return {"pass": bool(ok), "base_has_the_rows_and_clips_of_the_join": base_rows, "arms": facts}


# ============================================================================================ build
def build(join_path, base_path, base_rec_path, out_dir, fault=None, base_for_build=None):
    """-> record. base_for_build: a base file path used only for composing (selftest: a fault where the file named as base is not the one recorded)."""
    if os.path.exists(out_dir):
        raise SystemExit("REFUSED: %s exists" % out_dir)
    J = load_npz(join_path)
    B = load_npz(base_for_build or base_path)
    C = compose(J, B, fault)
    os.makedirs(out_dir)
    files = {}
    for a in ARMS:
        p = os.path.join(out_dir, fname(a))
        np.savez(p, **C["arrays"][a])
        files[a] = p
    cl = {"second_routine": second_routine_clause(J, B, files), "ranges_partition": ranges_partition_clause(C), "outside_is_base": outside_is_base_clause(C, files),
          "base_is_recorded": base_is_recorded_clause(base_path if base_for_build is None else base_for_build, base_rec_path), "layout": layout_clause(C, B, files)}
    rec_files = {}
    for a in ARMS:
        rec_files[a] = {"path": files[a].replace("\\", "/"), "sha256": sha_file(files[a]), "n_rows": int(len(C["ci"])), "n_label_rows": int(C["masks"][a].sum()),
                        "n_source_changes": count_source_changes(C["ci"], C["masks"][a])}
    rec = {"pass_a0": bool(all(cl[k]["pass"] for k in CLAUSES)), "clauses": cl, "files": rec_files,
           "base": {"path": base_path.replace("\\", "/"), "sha256": sha_file(base_path), "named_in_record_add106": bool(cl["base_is_recorded"]["pass"])},
           "eval_join_sha256": sha_file(join_path), "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(os.path.join(out_dir, RECORD_NAME), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    return rec


# ============================================================================================ synthetic data (selftest and tests)
def synthetic_join(n_clips=4, seed=1):
    rng = np.random.RandomState(seed)
    ns = [14 + 3 * i for i in range(n_clips)]
    off = np.concatenate([[0], np.cumsum(ns)[:-1]]).astype(np.int64)
    tot = int(sum(ns))
    gap = rng.uniform(2.0, 70.0, tot).astype(np.float32)
    valid = (rng.rand(tot) < 0.75).astype(np.int64)
    gap[rng.rand(tot) < 0.08] = np.nan
    edge = [20.0, 40.0, 60.0, 60.01, 19.99, 0.0, 40.0, -1.0]
    for i, e in enumerate(edge):                                              # edge values on valid rows of clip 0
        gap[i + 2] = e
        valid[i + 2] = 1
    return {"clip": np.asarray(["clip%02d" % i for i in range(n_clips)]), "off": off, "n": np.asarray(ns, np.int64), "lead_gap": gap, "lead_valid": valid}


def synthetic_base(J, seed=3):
    rng = np.random.RandomState(seed)
    n = np.asarray(J["n"], np.int64)
    ci = np.repeat(np.arange(len(n), dtype=np.int64), n)
    fr = np.concatenate([np.arange(k, dtype=np.int64) for k in n])
    gap = rng.uniform(5.0, 60.0, len(ci))
    gap[fr < 4] = np.nan
    gap[rng.rand(len(ci)) < 0.1] = np.nan
    return {"clip_index": ci, "frame": fr, "gap": gap, "has_out": np.isfinite(gap), "clips": np.asarray([str(c) for c in J["clip"]]), "seed": np.int64(-1)}


def write_world(d):
    """join, base, base record on disk under d -> (join path, base path, record path, other base path)"""
    os.makedirs(d, exist_ok=True)
    J = synthetic_join()
    B = synthetic_base(J)
    jp, bp, rp, op = (os.path.join(d, x) for x in ("join.npz", "VGAPLE.npz", "CROPSCALE_GAPFILES_RECORD.json", "other_base.npz"))
    np.savez(jp, **J)
    np.savez(bp, **B)
    B2 = dict(B)
    B2["gap"] = B["gap"] + np.where(np.isfinite(B["gap"]), 0.5, 0.0)
    np.savez(op, **B2)
    with open(rp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"files": {"VGAPLE": {"file_sha256": sha_file(bp)}, "VGAPL_s2": {"file_sha256": "0" * 64}}}, fh)
    return jp, bp, rp, op


FAULTS = {"second_routine": ("offset_label", "outside_zero"), "ranges_partition": ("lm_incl_40", "lf_drop_last"), "outside_is_base": ("outside_zero",), "layout": ("f32gap", "extra_field", "seed0")}


def selftest(tmp_root=None):
    root = tmp_root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_selftest_attr_%d" % os.getpid())
    jp, bp, rp, op = write_world(os.path.join(root, "world"))
    bad = []
    rec = build(jp, bp, rp, os.path.join(root, "clean"))
    for k in CLAUSES:
        if not rec["clauses"][k]["pass"]:
            bad.append("clean build: clause %s is red" % k)
    if not rec["pass_a0"]:
        bad.append("clean build: pass_a0 is false")
    n = 0
    for clause, fl in FAULTS.items():
        for f in fl:
            n += 1
            r = build(jp, bp, rp, os.path.join(root, "f%d" % n), fault=f)
            if r["clauses"][clause]["pass"] or r["pass_a0"]:
                bad.append("fault %s did not turn clause %s red" % (f, clause))
    r = build(jp, bp, rp, os.path.join(root, "fbase"), base_for_build=op)
    if r["clauses"]["base_is_recorded"]["pass"] or r["pass_a0"]:
        bad.append("an unrecorded base did not turn base_is_recorded red")
    for b in bad:
        print("SELFTEST FAIL: " + b)
    print("ZZ ATTR-GAPFILES SELFTEST " + ("OK" if not bad else "FAIL"))
    return 0 if not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-join")
    ap.add_argument("--base")
    ap.add_argument("--base-record")
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.eval_join and a.base and a.base_record and a.out_dir):
        ap.error("--eval-join, --base, --base-record and --out-dir are required")
    rec = build(a.eval_join, a.base, a.base_record, a.out_dir)
    for k in CLAUSES:
        print("clause %s %s" % (k, rec["clauses"][k]["pass"]))
    print("ZZ ATTR-GAPFILES pass_a0 %s" % rec["pass_a0"])
    return 0 if rec["pass_a0"] else 1


if __name__ == "__main__":
    sys.exit(main())
