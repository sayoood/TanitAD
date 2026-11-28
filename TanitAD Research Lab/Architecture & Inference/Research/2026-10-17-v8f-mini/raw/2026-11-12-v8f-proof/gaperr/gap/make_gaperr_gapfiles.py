"""make_gaperr_gapfiles.py -- the four gap files of card PREREG_V8F_PROOF_ADD111 (GAPERR10K, section 2) with control b0.

  VGAPA_NR.npz  fast part removed: label gap plus the slow part s on every error row
  VGAPA_BR.npz  slow part removed: label gap plus the fast part f on every error row
  VGAPA_SM.npz  base file only: median of the base gap over the trailing run of a row (at most 5 frames)
  VGAPA_SL.npz  base file only: end value of the least-squares line over the trailing run (3 or more frames), else the base gap
Terms (card): label rows = lead_valid 1 and lead_gap finite and 0 <= lead_gap <= 60. Error rows = label rows where the base file has an output (and a finite gap there). e = base gap minus label gap on an
error row. Error run = a largest set of error rows of one clip on consecutive frames. s = mean of e over the rows of the run within 5 frames before and 5 frames after the row, the row included
(fewer near the ends of the run). f = e - s. NR and BR: a label row with no base output has the label gap and an output (as in the LA file of card ADD110); every other row has gap and output flag of the base file.
Trailing run of a row with a base output: the row and the frames before it in the same clip, going back while each frame has a base output, is the frame before the next, and differs from the next by at most
3.0 m in the base gap; at most 5 frames. SM and SL keep the output flag of the base file on every row; a row with no base output keeps its base gap field.
The functions that build SM and SL (smooth_gap) take base arrays only; no label array is in their signature.
Array layout of the base file (card ADD106): clip_index int64, frame int64, gap float64 (NaN where no output), has_out bool, clips (the join's clip names), seed int64 (-1 for a composed file). No other field.
Usage: python make_gaperr_gapfiles.py --join NPZ --base NPZ --base-record JSON --la-file NPZ --out-dir NEWDIR    (refuses an existing out dir; exit 0 only when pass_b0 is true)
       python make_gaperr_gapfiles.py --selftest
Control b0 clauses: second_routine, error_identity, la_equivalence, smoothing_properties, no_label_in_smoothing, base_is_recorded, layout. Record: NEWDIR/GAPERR_GAPFILES_RECORD.json.
The record holds the T0 table (recorded frames, not driving performance); the printout holds no gap value and no rms value.
numpy and the standard library only."""
import argparse
import datetime
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

GAP_MAX_M = 60.0
NEAR_M, MID_M = 20.0, 40.0
WIN = 5
RUN_MAX = 5
JUMP_M = 3.0
ARMS = ("NR", "BR", "SM", "SL")
FIELDS = ("clip_index", "frame", "gap", "has_out", "clips", "seed")
DTYPES = {"clip_index": "int64", "frame": "int64", "gap": "float64", "has_out": "bool", "seed": "int64"}
RECORD_NAME = "GAPERR_GAPFILES_RECORD.json"
TOL_M = 1e-9
CLAUSES = ("second_routine", "error_identity", "la_equivalence", "smoothing_properties", "no_label_in_smoothing", "base_is_recorded", "layout")
REC_KEYS = ("pass_b0", "clauses", "files", "base", "la_file", "eval_join_sha256", "t0", "written_utc")


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


# ============================================================================================ the join frames
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


# ============================================================================================ error parts (slow and fast)
def error_parts(ci, fr, lab, bh, g, bg, win=WIN):
    """-> err (error rows), e, s, f (NaN off the error rows)"""
    err = lab & bh & np.isfinite(bg)
    nrow = len(err)
    e = np.full(nrow, np.nan)
    e[err] = bg[err] - g[err]
    s = np.full(nrow, np.nan)
    i = 0
    while i < nrow:
        if not err[i]:
            i += 1
            continue
        j = i
        while j + 1 < nrow and err[j + 1] and ci[j + 1] == ci[i] and fr[j + 1] == fr[j] + 1:
            j += 1
        cs = np.concatenate([[0.0], np.cumsum(e[i:j + 1])])
        for r in range(i, j + 1):
            lo, hi = max(i, r - win), min(j, r + win)
            s[r] = (cs[hi - i + 1] - cs[lo - i]) / (hi - lo + 1)
        i = j + 1
    return err, e, s, e - s


# ============================================================================================ the two smoothing forms (base arrays only)
def smooth_gap(clip_index, frame, gap, has_out, kind, fault=None):
    """SM or SL of the base gap. Inputs are base arrays only (clip index, frame, gap, output flag): no label array is a parameter."""
    gap = np.asarray(gap, np.float64)
    ok = np.asarray(has_out, bool) & np.isfinite(gap)
    out = gap.copy()
    run_max = 3 if (fault == "sm_median3" and kind == "SM") else RUN_MAX
    for i in range(len(gap)):
        if not ok[i]:
            continue
        j, k = i, 1
        while k < run_max and j > 0 and ok[j - 1] and clip_index[j - 1] == clip_index[j] and frame[j - 1] == frame[j] - 1 and abs(gap[j - 1] - gap[j]) <= JUMP_M:
            j -= 1
            k += 1
        y = gap[j:i + 1]
        if kind == "SM":
            out[i] = float(np.median(y))
        elif fault == "sl_mean":
            out[i] = float(np.mean(y))
        elif k >= 3:
            p = np.polyfit(np.arange(k, dtype=np.float64), y, 1)
            out[i] = float(p[0] * (k - 1) + p[1])
    return out


# ============================================================================================ compose
def make_arr(ci, fr, names, gap, has, fault):
    arr = {"clip_index": ci.copy(), "frame": fr.copy(), "gap": gap.astype(np.float32) if fault == "f32gap" else gap, "has_out": has, "clips": np.asarray(names), "seed": np.int64(0 if fault == "seed0" else -1)}
    if fault == "extra_field":
        arr["valid"] = has.copy()
    return arr


def compose(J, B, fault=None):
    """-> dict(names, n, ci, fr, g, lab, base_gap, base_has, err, e, s, f, arrays{arm: dict})"""
    names, n, cum, idx, ci, fr = join_frames(J)
    g, lab = label_arrays(J, idx)
    bg, bh = lay_base(names, n, cum, B)
    err, e, s, f = error_parts(ci, fr, lab, bh, g, bg, WIN - 1 if fault == "win4" else WIN)
    gaps, hass = {}, {}
    with np.errstate(invalid="ignore"):
        for arm, part in (("NR", s), ("BR", -f if fault == "br_sign" else f)):
            gp = np.where(err, g + part, np.where(lab, g, bg))
            if fault == "nr_nonerr_offset" and arm == "NR":
                gp = np.where(lab & ~err, gp + 1e-6, gp)
            gaps[arm], hass[arm] = gp, bh | lab
        for arm in ("SM", "SL"):
            gp = smooth_gap(ci, fr, bg, bh, arm, fault)
            if fault == "sm_label_leak" and arm == "SM":
                gp = np.where(lab, 0.5 * gp + 0.5 * g, gp)
            gaps[arm] = gp
            hass[arm] = np.ones(len(bh), bool) if (fault == "sm_flag" and arm == "SM") else bh.copy()
    arrays = {a: make_arr(ci, fr, names, gaps[a], hass[a], fault) for a in ARMS}
    return {"names": names, "n": n, "ci": ci, "fr": fr, "g": g, "lab": lab, "base_gap": bg, "base_has": bh, "err": err, "e": e, "s": s, "f": f, "arrays": arrays}


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
    n_err = 0
    for c, nm in enumerate(names):
        nf = int(n[c])
        lab, bgl, bhl = [], [], []
        for f in range(nf):
            gg = float(J["lead_gap"][int(off[c]) + f])
            vv = int(J["lead_valid"][int(off[c]) + f])
            lab.append(gg if ((vv == 1) and (gg == gg) and (gg >= 0.0) and (gg <= GAP_MAX_M)) else None)
            bgv, bhv = bkey.get((nm, f), (float("nan"), False))
            bgl.append(bgv)
            bhl.append(bhv)
        iserr = [lab[f] is not None and bhl[f] and bgl[f] == bgl[f] and abs(bgl[f]) != float("inf") for f in range(nf)]
        e = [(bgl[f] - lab[f]) if iserr[f] else None for f in range(nf)]
        s = [None] * nf
        f0 = 0
        while f0 < nf:
            if not iserr[f0]:
                f0 += 1
                continue
            f1 = f0
            while f1 + 1 < nf and iserr[f1 + 1]:
                f1 += 1
            for r in range(f0, f1 + 1):
                vals = [e[q] for q in range(max(f0, r - 5), min(f1, r + 5) + 1)]
                s[r] = sum(vals) / len(vals)
            f0 = f1 + 1
        for f in range(nf):
            if iserr[f]:
                n_err += 1
                exp["NR"][(nm, f)] = (lab[f] + s[f], True)
                exp["BR"][(nm, f)] = (lab[f] + (e[f] - s[f]), True)
            elif lab[f] is not None:
                exp["NR"][(nm, f)] = (lab[f], True)
                exp["BR"][(nm, f)] = (lab[f], True)
            else:
                exp["NR"][(nm, f)] = (bgl[f], bhl[f])
                exp["BR"][(nm, f)] = (bgl[f], bhl[f])
            # the two smoothing forms, from the base only
            usable = bhl[f] and bgl[f] == bgl[f] and abs(bgl[f]) != float("inf")
            if not usable:
                exp["SM"][(nm, f)] = (bgl[f], bhl[f])
                exp["SL"][(nm, f)] = (bgl[f], bhl[f])
                continue
            vals = [bgl[f]]
            q = f - 1
            while len(vals) < 5 and q >= 0 and bhl[q] and bgl[q] == bgl[q] and abs(bgl[q] - vals[0]) <= 3.0:
                vals.insert(0, bgl[q])
                q -= 1
            srt = sorted(vals)
            k = len(srt)
            med = srt[k // 2] if k % 2 == 1 else 0.5 * (srt[k // 2 - 1] + srt[k // 2])
            exp["SM"][(nm, f)] = (med, bhl[f])
            if k >= 3:
                xm = (k - 1) / 2.0
                ym = sum(vals) / k
                sxx = sum((x - xm) ** 2 for x in range(k))
                sxy = sum((x - xm) * (vals[x] - ym) for x in range(k))
                exp["SL"][(nm, f)] = (ym + (sxy / sxx) * ((k - 1) - xm), bhl[f])
            else:
                exp["SL"][(nm, f)] = (bgl[f], bhl[f])
    facts, ok = {}, True
    for a in ARMS:
        z = load_npz(files[a])
        fn = [str(c) for c in z["clips"]]
        worst, nan_eq, has_eq, seen = 0.0, True, True, set()
        for r in range(len(z["frame"])):
            key = (fn[int(z["clip_index"][r])], int(z["frame"][r]))
            seen.add(key)
            ex = exp[a].get(key)
            if ex is None:
                nan_eq = False
                continue
            fg, fh = float(z["gap"][r]), bool(z["has_out"][r])
            if (fg != fg) != (ex[0] != ex[0]):
                nan_eq = False
            elif fg == fg:
                worst = max(worst, abs(fg - ex[0]))
            if fh != ex[1]:
                has_eq = False
        rows_ok = len(z["frame"]) == len(exp[a]) and seen == set(exp[a].keys())
        f_ok = rows_ok and nan_eq and has_eq and worst <= TOL_M
        ok = ok and f_ok
        facts[a] = {"pass": bool(f_ok), "largest_abs_gap_difference_m": worst, "has_out_equal": has_eq, "nan_pattern_equal": nan_eq, "same_rows": rows_ok}
    return {"pass": bool(ok), "tolerance_m": TOL_M, "independent_error_rows": n_err, "arms": facts}


def error_identity_clause(C, files):
    """on every error row label plus s plus f equals the base gap (read from the saved NR and BR files: (g + s) + (g + f) - g)"""
    zn, zb = load_npz(files["NR"]), load_npz(files["BR"])
    err, g, bg = C["err"], C["g"], C["base_gap"]
    if len(zn["gap"]) != len(err) or len(zb["gap"]) != len(err):
        return {"pass": False, "reason": "row count"}
    gn, gb = np.asarray(zn["gap"], np.float64), np.asarray(zb["gap"], np.float64)
    d = np.abs((gn[err] + gb[err] - g[err]) - bg[err])
    worst = float(d.max()) if len(d) else 0.0
    ok = bool(len(d) > 0 and np.isfinite(d).all() and worst <= TOL_M)
    return {"pass": ok, "n_error_rows": int(err.sum()), "largest_abs_difference_m": worst, "tolerance_m": TOL_M}


def la_equivalence_clause(C, files, la_path):
    """NR and BR equal the file LA of card ADD110 on every row that is not an error row (gap and output flag)"""
    try:
        la = load_npz(la_path)
    except Exception as ex:
        return {"pass": False, "reason": "LA file unreadable: %s" % type(ex).__name__}
    n = len(C["ci"])
    if sorted(la.keys()) != sorted(FIELDS) or len(la["gap"]) != n or not (np.array_equal(la["clip_index"], C["ci"]) and np.array_equal(la["frame"], C["fr"])) \
            or [str(c) for c in la["clips"]] != C["names"]:
        return {"pass": False, "reason": "the LA file has not the layout or the rows of the join"}
    rest = ~C["err"]
    facts, ok = {}, True
    for a in ("NR", "BR"):
        z = load_npz(files[a])
        gd = np.array_equal(np.asarray(z["gap"], np.float64)[rest], np.asarray(la["gap"], np.float64)[rest], equal_nan=True)
        hd = np.array_equal(np.asarray(z["has_out"], bool)[rest], np.asarray(la["has_out"], bool)[rest])
        facts[a] = {"pass": bool(gd and hd), "gap_equal": bool(gd), "has_out_equal": bool(hd)}
        ok = ok and facts[a]["pass"]
    return {"pass": bool(ok), "n_rows_not_error": int(rest.sum()), "arms": facts}


def property_series():
    """small hand-made series of one clip: (name, gap, has_out)"""
    f = np.arange(12, dtype=np.float64)
    nan = float("nan")
    return {"constant": (np.full(12, 12.5), np.ones(12, bool)),
            "ramp_down": (40.0 - 1.0 * f, np.ones(12, bool)),
            "ramp_up_2": (10.0 + 2.0 * f, np.ones(12, bool)),
            "ramp_step_3": (10.0 + 3.0 * f, np.ones(12, bool)),
            "jump": (np.asarray([10.0] * 5 + [20.0] * 5), np.ones(10, bool)),
            "hole": (np.asarray([10.0, 11.0, 12.0, nan, 14.0, 15.0]), np.asarray([1, 1, 1, 0, 1, 1], bool))}


def smoothing_properties_clause(C, files, fault=None):
    facts = {}
    for a in ("SM", "SL"):
        z = load_npz(files[a])
        facts[a + "_flags_are_the_base_flags"] = bool(len(z["has_out"]) == len(C["base_has"]) and np.array_equal(np.asarray(z["has_out"], bool), C["base_has"]))
        nog = ~C["base_has"]
        facts[a + "_no_output_rows_keep_the_base_gap"] = bool(len(z["gap"]) == len(nog) and np.array_equal(np.asarray(z["gap"], np.float64)[nog], C["base_gap"][nog], equal_nan=True))
    S = property_series()

    def run(name, kind):
        y, h = S[name]
        k = np.zeros(len(y), np.int64)
        fr = np.arange(len(y), dtype=np.int64)
        return smooth_gap(k, fr, y, h, kind, fault)
    y = S["constant"][0]
    facts["constant_course_unchanged_SM"] = bool(np.array_equal(run("constant", "SM"), y))
    facts["constant_course_unchanged_SL"] = bool(np.abs(run("constant", "SL") - y).max() <= TOL_M)
    late_ok, back_ok = True, True
    for nm in ("ramp_down", "ramp_up_2", "ramp_step_3"):
        y = S[nm][0]
        back_ok = back_ok and bool(np.abs(run(nm, "SL") - y).max() <= TOL_M)
        sm = run(nm, "SM")
        late_ok = late_ok and bool(all(abs(sm[i] - y[i - 2]) <= TOL_M for i in range(4, len(y))))
    facts["straight_course_back_from_SL"] = back_ok
    facts["SM_two_frames_late_once_five_frames_are_in_the_run"] = late_ok
    yj = S["jump"][0]
    facts["a_jump_above_3_m_ends_the_run"] = bool(np.abs(run("jump", "SM") - yj).max() <= TOL_M and np.abs(run("jump", "SL") - yj).max() <= TOL_M)
    sm, sl = run("hole", "SM"), run("hole", "SL")
    facts["a_frame_without_output_ends_the_run"] = bool(math.isnan(sm[3]) and math.isnan(sl[3]) and sm[4] == 14.0 and sl[4] == 14.0 and abs(sm[5] - 14.5) <= TOL_M and sl[5] == 15.0)
    return {"pass": bool(all(facts.values())), "facts": facts}


def no_label_in_smoothing_clause(J, B, files, fault=None):
    """SM and SL rebuilt with the label fields of the join replaced by other values: the saved arrays are byte for byte the same"""
    n = len(J["lead_gap"])
    variants = {"reversed_shifted_flipped": (np.asarray(J["lead_gap"], np.float32)[::-1] + np.float32(7.0), 1 - np.asarray(J["lead_valid"])),
                "constant_all_valid": (np.full(n, 1.0, np.float32), np.ones(n, np.int64)),
                "nan_none_valid": (np.full(n, np.nan, np.float32), np.zeros(n, np.int64))}
    facts, ok = {}, True
    for vn, (lg, lv) in variants.items():
        J2 = dict(J)
        J2["lead_gap"], J2["lead_valid"] = lg, lv
        C2 = compose(J2, B, fault)
        for a in ("SM", "SL"):
            z = load_npz(files[a])
            same = sorted(z.keys()) == sorted(C2["arrays"][a].keys())
            if same:
                for k in z:
                    r = np.asarray(C2["arrays"][a][k])
                    same = same and z[k].dtype == r.dtype and z[k].shape == r.shape and z[k].tobytes() == r.tobytes()
            facts["%s_%s" % (a, vn)] = bool(same)
            ok = ok and same
    return {"pass": bool(ok), "facts": facts}


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


# ============================================================================================ record parts
def n_diff_rows(arr, C):
    gd = np.asarray(arr["gap"], np.float64)
    same_gap = (gd == C["base_gap"]) | (np.isnan(gd) & np.isnan(C["base_gap"]))
    return int(((~same_gap) | (np.asarray(arr["has_out"], bool) != C["base_has"])).sum())


def rms(x):
    return float(math.sqrt(float(np.mean(np.asarray(x, np.float64) ** 2)))) if len(x) else None


def t0_table(C):
    """T0 (recorded frames, not driving performance): per range of the label gap, over the error rows"""
    g, err = C["g"], C["err"]
    sm, sl = C["arrays"]["SM"]["gap"], C["arrays"]["SL"]["gap"]
    with np.errstate(invalid="ignore"):
        ranges = {"near": err & (g < NEAR_M), "middle": err & (g >= NEAR_M) & (g < MID_M), "far": err & (g >= MID_M), "all": err}
    out = {}
    for k, m in ranges.items():
        out[k] = {"n_error_rows": int(m.sum()), "rms_e": rms(C["e"][m]), "rms_s": rms(C["s"][m]), "rms_f": rms(C["f"][m]),
                  "rms_sm": rms(np.asarray(sm, np.float64)[m] - g[m]), "rms_sl": rms(np.asarray(sl, np.float64)[m] - g[m])}
    return out


# ============================================================================================ build
def build(join_path, base_path, base_rec_path, la_path, out_dir, fault=None, base_for_build=None):
    """-> record. base_for_build: a base file path used only for composing (selftest: the file named as base is not the one recorded)."""
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
    cl = {"second_routine": second_routine_clause(J, B, files), "error_identity": error_identity_clause(C, files), "la_equivalence": la_equivalence_clause(C, files, la_path),
          "smoothing_properties": smoothing_properties_clause(C, files, fault), "no_label_in_smoothing": no_label_in_smoothing_clause(J, B, files, fault),
          "base_is_recorded": base_is_recorded_clause(base_path if base_for_build is None else base_for_build, base_rec_path), "layout": layout_clause(C, B, files)}
    rec_files = {}
    for a in ARMS:
        rec_files[a] = {"n_rows": int(len(C["ci"])), "n_diff_rows": n_diff_rows(C["arrays"][a], C),
                        "n_label_value_rows": int(C["lab"].sum()) if a in ("NR", "BR") else 0,
                        "path": files[a].replace("\\", "/"), "sha256": sha_file(files[a])}
    rec = {"pass_b0": bool(all(cl[k]["pass"] for k in CLAUSES)), "clauses": cl, "files": rec_files,
           "base": {"path": base_path.replace("\\", "/"), "sha256": sha_file(base_path), "named_in_record_add106": bool(cl["base_is_recorded"]["pass"])},
           "la_file": {"path": la_path.replace("\\", "/"), "sha256": sha_file(la_path)},
           "eval_join_sha256": sha_file(join_path), "t0": t0_table(C),
           "written_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    with open(os.path.join(out_dir, RECORD_NAME), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(rec, fh, indent=1)
    return rec


# ============================================================================================ synthetic data (selftest and tests)
def synthetic_join(n_clips=4, seed=1):
    rng = np.random.RandomState(seed)
    ns = [44 + 7 * i for i in range(n_clips)]
    off = np.concatenate([[0], np.cumsum(ns)[:-1]]).astype(np.int64)
    gap, valid = [], []
    for c, nn in enumerate(ns):
        f = np.arange(nn, dtype=np.float64)
        gap.append(30.0 + 28.0 * np.sin(f / 7.0 + c))
        v = np.ones(nn, np.int64)
        for st in rng.randint(8, nn - 8, 2):
            v[st:st + 4] = 0
        v[rng.rand(nn) < 0.03] = 0
        valid.append(v)
    gap = np.concatenate(gap).astype(np.float32)
    valid = np.concatenate(valid)
    gap[rng.rand(len(gap)) < 0.02] = np.nan
    for i, e in enumerate([20.0, 40.0, 60.0, 60.01, 19.99, 0.0, 40.0, -1.0]):          # edge values on valid rows of clip 0
        gap[i + 10] = e
        valid[i + 10] = 1
    return {"clip": np.asarray(["clip%02d" % i for i in range(n_clips)]), "off": off, "n": np.asarray(ns, np.int64), "lead_gap": gap, "lead_valid": valid}


def synthetic_base(J, seed=3):
    rng = np.random.RandomState(seed)
    n = np.asarray(J["n"], np.int64)
    ci = np.repeat(np.arange(len(n), dtype=np.int64), n)
    fr = np.concatenate([np.arange(k, dtype=np.int64) for k in n])
    lg = np.asarray(J["lead_gap"], np.float64)
    gap = np.where(np.isfinite(lg), lg, 25.0) + rng.normal(0.0, 0.8, len(ci))
    gap[rng.rand(len(ci)) < 0.05] += 6.0                                               # jumps above 3 m
    gap = np.clip(gap, 0.5, None)
    gap[fr < 4] = np.nan
    gap[rng.rand(len(ci)) < 0.06] = np.nan
    return {"clip_index": ci, "frame": fr, "gap": gap, "has_out": np.isfinite(gap), "clips": np.asarray([str(c) for c in J["clip"]]), "seed": np.int64(-1)}


def synthetic_la(J, B):
    """the file LA of card ADD110 as that card defines it: the label gap on every label row, the base file elsewhere"""
    names, n, cum, idx, ci, fr = join_frames(J)
    g, lab = label_arrays(J, idx)
    bg, bh = lay_base(names, n, cum, B)
    return make_arr(ci, fr, names, np.where(lab, g, bg), bh | lab, None)


def write_world(d):
    """join, base, base record, other base, LA file, other LA file on disk under d -> (join, base, record, other base, la, other la)"""
    os.makedirs(d, exist_ok=True)
    J = synthetic_join()
    B = synthetic_base(J)
    jp, bp, rp, op, lp, lop = (os.path.join(d, x) for x in ("join.npz", "VGAPLE.npz", "CROPSCALE_GAPFILES_RECORD.json", "other_base.npz", "VGAPA_LA.npz", "other_la.npz"))
    np.savez(jp, **J)
    np.savez(bp, **B)
    B2 = dict(B)
    B2["gap"] = B["gap"] + np.where(np.isfinite(B["gap"]), 0.5, 0.0)
    np.savez(op, **B2)
    with open(rp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"files": {"VGAPLE": {"file_sha256": sha_file(bp)}, "VGAPL_s2": {"file_sha256": "0" * 64}}}, fh)
    LA = synthetic_la(J, B)
    np.savez(lp, **LA)
    LA2 = dict(LA)
    LA2["gap"] = np.asarray(LA["gap"], np.float64) + 0.5
    np.savez(lop, **LA2)
    return jp, bp, rp, op, lp, lop


FAULTS = {"second_routine": ("win4", "nr_nonerr_offset"), "error_identity": ("br_sign",), "la_equivalence": ("nr_nonerr_offset",),
          "smoothing_properties": ("sl_mean", "sm_median3", "sm_flag"), "no_label_in_smoothing": ("sm_label_leak",), "layout": ("f32gap", "extra_field", "seed0")}


def selftest(tmp_root=None):
    root = tmp_root or tempfile.mkdtemp(prefix="gaperr_selftest_")
    jp, bp, rp, op, lp, lop = write_world(os.path.join(root, "world"))
    bad = []
    rec = build(jp, bp, rp, lp, os.path.join(root, "clean"))
    for k in CLAUSES:
        if not rec["clauses"][k]["pass"]:
            bad.append("clean build: clause %s is red" % k)
    if not rec["pass_b0"]:
        bad.append("clean build: pass_b0 is false")
    if sorted(rec.keys()) != sorted(REC_KEYS):
        bad.append("record keys")
    n = 0
    for clause, fl in FAULTS.items():
        for f in fl:
            n += 1
            r = build(jp, bp, rp, lp, os.path.join(root, "f%d" % n), fault=f)
            if r["clauses"][clause]["pass"] or r["pass_b0"]:
                bad.append("fault %s did not turn clause %s red" % (f, clause))
    r = build(jp, bp, rp, lp, os.path.join(root, "fbase"), base_for_build=op)
    if r["clauses"]["base_is_recorded"]["pass"] or r["pass_b0"]:
        bad.append("an unrecorded base did not turn base_is_recorded red")
    r = build(jp, bp, rp, lop, os.path.join(root, "fla"))
    if r["clauses"]["la_equivalence"]["pass"] or r["pass_b0"]:
        bad.append("another LA file did not turn la_equivalence red")
    for b in bad:
        print("SELFTEST FAIL: " + b)
    if tmp_root is None:
        shutil.rmtree(root, ignore_errors=True)
    print("ZZ GAPERR-GAPFILES SELFTEST " + ("OK" if not bad else "FAIL"))
    return 0 if not bad else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--join")
    ap.add_argument("--base")
    ap.add_argument("--base-record")
    ap.add_argument("--la-file")
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.join and a.base and a.base_record and a.la_file and a.out_dir):
        ap.error("--join, --base, --base-record, --la-file and --out-dir are required")
    rec = build(a.join, a.base, a.base_record, a.la_file, a.out_dir)
    for k in CLAUSES:
        print("clause %s %s" % (k, rec["clauses"][k]["pass"]))
    print("ZZ GAPERR-GAPFILES pass_b0 %s" % rec["pass_b0"])
    return 0 if rec["pass_b0"] else 1


if __name__ == "__main__":
    sys.exit(main())
