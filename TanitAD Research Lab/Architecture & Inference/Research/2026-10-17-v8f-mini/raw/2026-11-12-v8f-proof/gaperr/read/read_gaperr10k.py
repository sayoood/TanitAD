#!/usr/bin/env python
"""read_gaperr10k.py -- read of card PREREG_V8F_PROOF_ADD111 (GAPERR10K): which part of the error of the camera gap carries the price in falling behind, and do two smoothing forms with no label recover it.
Arms NR and BR (attribution arms: the label gap plus the slow part / plus the fast part of the gap error; label values in the input, never candidates) and SM and SL (the base gap file smoothed over the
trailing run, no label enters), two model seeds each, eight rollouts at T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s); references GEc, RLc, RR and LA (banked
rollouts; LA of card ADD110); control run V0 of card ADD102.
Generated from read_fallattr10k.py (card ADD110) by make_read_gaperr10k.py with asserted text substitutions: every engine call and every measure, pair and bootstrap routine is the code of that reader and
of the model reader read_cropscale10k.py of card ADD106 (imported as a sibling module, path by --model-reader); only what this card adds is written in the substituted blocks.
Stage controls: runs b0 to b5, the grid read and the STRATEGIC family; writes GAPERR10K_CONTROLS.json (payload when all pass, `withheld` otherwise) and prints only `b<i> PASS|FAIL` lines and a last line
  `ZZ READ-GAPERR10K controls all_pass <True|False>`. The engine's own output goes to <out>.engine.log.
Stage names: refuses unless the controls file has b0 to b5 all with pass true; writes GAPERR10K_RESULT.json and a names log.
--selftest: synthetic world; last line `ZZ READ-GAPERR10K SELFTEST OK` or FAIL."""

import argparse
import contextlib
import datetime
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
POD_MODEL = "/workspace/v8proof/cropscale/kit_read/read_cropscale10k.py"
DEV_MODEL = os.path.join(_HERE, "..", "..", "cropscale", "read", "read_cropscale10k.py")
MD = MSG = R = EC = None
MODEL_PATH = None


def locate_model(path=""):
    for p in (path, os.environ.get("GAPERR_MODEL_READER", ""), POD_MODEL, DEV_MODEL):
        if p and os.path.isfile(p):
            return os.path.abspath(p)
    raise ImportError("model reader read_cropscale10k.py not found (tried %r, %s, %s)" % (path, POD_MODEL, DEV_MODEL))


def bind(path=""):
    """imports the model reader of card ADD106 as a module from its file (its own siblings resolve from its folder)"""
    global MD, MSG, R, EC, MODEL_PATH
    p = locate_model(path)
    if MODEL_PATH == p and MD is not None:
        return MD
    spec = importlib.util.spec_from_file_location("read_cropscale10k_model", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["read_cropscale10k_model"] = mod
    spec.loader.exec_module(mod)
    MD, MSG, R, EC, MODEL_PATH = mod, mod.MSG, mod.R, mod.EC, p
    return mod


CARD, PRE = "PREREG_V8F_PROOF_ADD111", "GAPERR10K"
SEEDS = (0, 1)
ARMS = ("NR", "BR", "SM", "SL")
ATTR_ARMS = ("NR", "BR")
SMOOTH_ARMS = ("SM", "SL")
RUN = {x: "RQ50KF5P30Hb20b" + x for x in ARMS}
REFS = {"GEc": "RQ50KF5P30Hb20gGEc", "RR": "RQ50KF5P30Hb20", "RLc": "RLcKF5P30Hb20", "LA": "RQ50KF5P30Hb20aLA"}
STRAT_REFS = ("GEc", "RLc", "RR")
V0_RUN = "RQ50KF5P30Hb20vV0"
MEASURES = (("F", "FALLING-BEHIND"), ("C", "CONTACT"), ("U", "UNSAFE"))
EXPECTED_LEVELS = {"GEc": {"C": 0.0062, "F": 0.5154, "U": 0.0839}, "RLc": {"C": 0.0, "F": 0.4908, "U": 0.0714}, "RR": {"C": 0.0954, "F": 0.4954, "U": 0.1339},
                   "LA": {"C": 0.0, "F": 0.4877, "U": 0.0839}}
LEVEL_TOL = 1e-4
WRAPPER_SHA = "28bc31f21ecb4ccb"
CHECK_LAST = "ZZ CHECK-WRAPPER-V PASS"
JOIN_NAME = "join_eval141.npz"
TOL_CARRY = 1e-9
TOL_LABEL_SHARE = 1e-3
GAP_MAX_M = 60.0
RUN_MAX, STEP_MAX, SLOW_HALF = 5, 3.0, 5
RECORD_KEYS = ("pass_b0", "clauses", "files", "base", "la_file", "eval_join_sha256", "t0", "written_utc")
FILE_KEYS = ("n_rows", "n_diff_rows", "n_label_value_rows", "path", "sha256")
T0_RANGES = ("near", "middle", "far", "all")
T0_KEYS = ("n_error_rows", "rms_e", "rms_s", "rms_f", "rms_sm", "rms_sl")
CONTROLS = tuple("b%d" % i for i in range(6))
TIER = ("T1-kin/hb0.2 (kinematic rollout on the model's own actions, executor of record v8f_t1_eval_hb.py with the heading bound 0.2 1/m), pre-roll of 30 steps (6 s); "
        "the label gap enters the gap files of NR and BR only, as a tool of attribution (not admissible at inference, never candidates); SM and SL are made of the base file alone")
FAM_NOTE = ("the rule changes the planned speed along the path through the distance to the lead; it does not move the path, so the lateral family is expected to be unmoved by it, "
            "and the tactical family is not read by the rule; the strategic family is read from the strategic grid of the engine at T1-kin")
WS = "/workspace/v8proof"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_npz(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def read_text(p):
    return io.open(p, encoding="utf-8", errors="replace").read() if p and os.path.isfile(p) else ""


def read_json(p):
    try:
        return json.load(io.open(p, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def to_epoch(x):
    """a number of seconds, or an ISO time (with or without a trailing Z) -> float seconds; None for anything else"""
    if x is None or x == "":
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    try:
        return float(s)
    except ValueError:
        pass
    try:
        d = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=datetime.timezone.utc)
    return d.timestamp()


def gapfile_path(gap_dir, arm):
    return os.path.join(gap_dir, "VGAPA_%s.npz" % arm)


# ============================================================================================ the join, the gap files and b0
def label_rows(J):
    """-> (gap, label): the join's lead gap and the rows that carry a label lead (valid flag 1, finite gap from 0 to 60 m)"""
    g = np.asarray(J["lead_gap"], np.float64)
    return g, (np.asarray(J["lead_valid"]) == 1) & np.isfinite(g) & (g >= 0.0) & (g <= GAP_MAX_M)


def file_on_join(J, z):
    """a gap file laid on the join rows by this routine: (has, gap, n_bad_keys, n_dup_keys); rows the file does not name have no output"""
    n, off = np.asarray(J["n"], np.int64), np.asarray(J["off"], np.int64)
    nrow = int(n.sum())
    ci, fr = np.asarray(z["clip_index"], np.int64), np.asarray(z["frame"], np.int64)
    ok = (ci >= 0) & (ci < len(n))
    ok[ok] &= (fr[ok] >= 0) & (fr[ok] < n[ci[ok]])
    rows = off[ci[ok]] + fr[ok]
    has = np.zeros(nrow, bool)
    gap = np.full(nrow, np.nan)
    h = np.asarray(z["has_out"], bool)[ok]
    g = np.asarray(z["gap"], np.float64)[ok]
    has[rows] = h & np.isfinite(g)
    gap[rows] = np.where(h & np.isfinite(g), g, np.nan)
    return has, gap, int((~ok).sum()), int(len(rows) - len(np.unique(rows)))


def raw_on_join(J, z):
    """a gap file laid on the join rows with its raw gap field kept: (named, raw); rows the file does not name are not named"""
    n, off = np.asarray(J["n"], np.int64), np.asarray(J["off"], np.int64)
    nrow = int(n.sum())
    ci, fr = np.asarray(z["clip_index"], np.int64), np.asarray(z["frame"], np.int64)
    ok = (ci >= 0) & (ci < len(n))
    ok[ok] &= (fr[ok] >= 0) & (fr[ok] < n[ci[ok]])
    rows = off[ci[ok]] + fr[ok]
    named = np.zeros(nrow, bool)
    raw = np.full(nrow, np.nan)
    named[rows] = True
    raw[rows] = np.asarray(z["gap"], np.float64)[ok]
    return named, raw


def error_rows(J, bhas):
    """error rows (card section 2): label rows (valid flag 1, finite label gap from 0 to 60 m) where the base file has an output"""
    g, lab = label_rows(J)
    return lab & np.asarray(bhas, bool)


def slow_fast(J, err, e):
    """second routine of b0: the error runs (largest sets of error rows of one clip on consecutive frames), the slow part s of a row (the mean of e over the rows of its run whose frame lies within 5 frames before
    and 5 frames after the row, the row included) and the fast part f = e - s; both nan off the error rows"""
    n, off = np.asarray(J["n"], np.int64), np.asarray(J["off"], np.int64)
    s = np.full(len(e), np.nan)
    for c in range(len(n)):
        k = 0
        while k < n[c]:
            if not err[off[c] + k]:
                k += 1
                continue
            m = k
            while m + 1 < n[c] and err[off[c] + m + 1]:
                m += 1
            for i in range(k, m + 1):
                lo, hi = max(k, i - SLOW_HALF), min(m, i + SLOW_HALF)
                s[off[c] + i] = float(np.mean(e[off[c] + lo:off[c] + hi + 1]))
            k = m + 1
    return s, e - s


def compose_attr(J, arm, bhas, bgap):
    """the gap file of NR or BR on the join rows by the definition of card section 2: label gap plus s (NR) or plus f (BR) on every error row; a label row with no output of the base file has the label gap and an
    output; every other row has gap and output flag of the base file. -> (has, gap) with nan where there is no output"""
    g, lab = label_rows(J)
    err = error_rows(J, bhas)
    e = np.where(err, bgap - g, np.nan)
    s, f = slow_fast(J, err, e)
    has, gap = np.array(bhas, bool), np.array(bgap, np.float64)
    nob = lab & ~has
    has[nob], gap[nob] = True, g[nob]
    part = s if arm == "NR" else f
    gap[err] = g[err] + part[err]
    return has, np.where(has, gap, np.nan)


def trailing_run_start(bhas, bgap, r, first):
    """first row of the trailing run of row r (which has a base output): the frames before it in the same clip (first row `first`), going back while the frame has a base output and differs from the next frame
    of the run by at most 3.0 m in the base gap; at most 5 frames in all"""
    lo = r
    while r - lo + 1 < RUN_MAX and lo - 1 >= first and bhas[lo - 1] and abs(bgap[lo - 1] - bgap[lo]) <= STEP_MAX:
        lo -= 1
    return lo


def smooth_arms(J, bhas, bgap):
    """-> (sm, sl): SM is the median of the base gap over the trailing run, SL the end value of the least-squares line through it when the run has 3 or more frames (the base gap itself when it has fewer); nan
    where the base has no output. Only the clip lengths and offsets of the join are read: no label field enters"""
    n, off = np.asarray(J["n"], np.int64), np.asarray(J["off"], np.int64)
    sm, sl = np.full(len(bgap), np.nan), np.full(len(bgap), np.nan)
    for c in range(len(n)):
        for k in range(int(n[c])):
            r = int(off[c]) + k
            if not bhas[r]:
                continue
            lo = trailing_run_start(bhas, bgap, r, int(off[c]))
            v = np.asarray(bgap[lo:r + 1], np.float64)
            sm[r] = float(np.median(v))
            if len(v) >= 3:
                x = np.arange(len(v), dtype=np.float64)
                xm, vm = x.mean(), v.mean()
                b = float(((x - xm) * (v - vm)).sum() / ((x - xm) ** 2).sum())
                sl[r] = float(vm + b * (x[-1] - xm))
            else:
                sl[r] = float(bgap[r])
    return sm, sl


def label_free(J, bhas, bgap):
    """SM and SL come out byte for byte the same when the label fields of the join are replaced by other values"""
    Jx = dict(J)
    Jx["lead_gap"] = np.asarray(J["lead_gap"], np.float64)[::-1] * 0.5 + 3.0
    Jx["lead_valid"] = 1 - np.asarray(J["lead_valid"]).astype(np.int64)
    a, b = smooth_arms(J, bhas, bgap), smooth_arms(Jx, bhas, bgap)
    return a[0].tobytes() == b[0].tobytes() and a[1].tobytes() == b[1].tobytes()


def course_checks():
    """the properties of the smoothing forms stated in b0, on synthetic courses with the routine of this reader: a constant course comes back unchanged (SM and SL, within 1e-9); a straight course comes back from SL
    within 1e-9 and from SM two frames late once five frames are in the run. -> list of failures"""
    bad = []
    Jc = {"n": np.array([12]), "off": np.array([0])}
    has = np.ones(12, bool)
    const = np.full(12, 17.3)
    sm, sl = smooth_arms(Jc, has, const)
    if not (np.max(np.abs(sm - const)) <= TOL_CARRY and np.max(np.abs(sl - const)) <= TOL_CARRY):
        bad.append("a constant course does not come back unchanged")
    line = 30.0 - 0.5 * np.arange(12.0)
    sm, sl = smooth_arms(Jc, has, line)
    if not np.max(np.abs(sl - line)) <= TOL_CARRY:
        bad.append("SL does not give a straight course back")
    if not np.max(np.abs(sm[4:] - line[2:10])) <= TOL_CARRY:
        bad.append("SM does not give a straight course back two frames late once five frames are in the run")
    return bad


def file_diff(ehas, egap, has, gap):
    """largest absolute difference of two gap files on the rows with an output; inf when the output flags differ"""
    if not np.array_equal(ehas, has):
        return float("inf")
    if not ehas.any():
        return 0.0
    return float(np.max(np.abs(egap[ehas] - gap[ehas])))


def _filled(d, keys):
    return isinstance(d, dict) and all(k in d and d[k] is not None for k in keys)


def ctl_b0(rec, files, J, join_sha, join_name, base_arrays=None, base_disk_sha=None, la_arrays=None, la_disk_sha=None):
    """b0 (the gap files are what the card says). rec: the gap-file record (exactly the keys RECORD_KEYS); files {arm: path}; J: the evaluation join; join_sha, join_name: of the join file; base_arrays, la_arrays:
    the base file and the file LA of card ADD110 as dicts of arrays, base_disk_sha and la_disk_sha their hashes on disk.
    Clauses: the record has exactly its keys, pass_b0 true and every clause pass true, its files entries and its t0 block are filled; each gap file on disk has the hash and the row count the record names; by the
    second routines of this reader (compose_attr, smooth_arms, written apart from the gap-file tool) each file equals the definition of card section 2 on every join row (at most 1e-9 m, equal output flags);
    on every error row NR plus BR minus the label gap equals the base gap (label plus s plus f); NR and BR equal LA on every row that is not an error row; SM and SL have the output flags of the base file, keep the
    raw gap field of the base on rows with no output, come out byte for byte the same when the label fields are replaced (no label enters), and give a constant course back unchanged, SL a straight course back
    and SM the same course two frames late; the base file and the file LA are the ones whose hashes the record names; the join is the pinned one (name and the hash the record names)."""
    bad = []
    if not isinstance(rec, dict):
        return {"pass": False, "failures": ["no gap-file record"]}
    if sorted(rec) != sorted(RECORD_KEYS):
        bad.append("the record has the top keys %s, not exactly %s" % (sorted(rec), sorted(RECORD_KEYS)))
    if rec.get("pass_b0") is not True:
        bad.append("the record does not read pass_b0 true")
    cl = rec.get("clauses")
    if not isinstance(cl, dict) or not cl:
        bad.append("the record has no clauses")
    else:
        for k, v in sorted(cl.items()):
            if not (isinstance(v, dict) and v.get("pass") is True):
                bad.append("record clause %s is not pass" % k)
    rf = rec.get("files")
    if not isinstance(rf, dict) or sorted(rf) != sorted(ARMS):
        bad.append("the record's files entries are %s, not exactly %s" % (sorted(rf) if isinstance(rf, dict) else rf, sorted(ARMS)))
        rf = rf if isinstance(rf, dict) else {}
    for arm in ARMS:
        if arm in rf and not _filled(rf[arm], FILE_KEYS):
            bad.append("%s: the record's file entry lacks one of %s" % (arm, list(FILE_KEYS)))
    t0 = rec.get("t0")
    if not isinstance(t0, dict) or sorted(t0) != sorted(T0_RANGES) or not all(_filled(t0[r_], T0_KEYS) for r_ in t0):
        bad.append("the record's t0 block lacks a range of %s or a key of %s" % (list(T0_RANGES), list(T0_KEYS)))
    if join_name != JOIN_NAME:
        bad.append("the evaluation join file is named %s, not %s" % (join_name, JOIN_NAME))
    if not join_sha or rec.get("eval_join_sha256") != join_sha:
        bad.append("the record's eval_join_sha256 is not the hash of the evaluation join")
    base = rec.get("base") if isinstance(rec.get("base"), dict) else {}
    if base.get("named_in_record_add106") is not True:
        bad.append("the record does not say the base file is named in the record of card ADD106")
    if base_arrays is None:
        bad.append("the base file could not be read")
    elif not base_disk_sha or not str(base.get("sha256") or "") or not base_disk_sha.startswith(str(base.get("sha256")).lower()) or len(str(base.get("sha256"))) < 16:
        bad.append("the base file on disk is not the one whose hash the record names")
    lf = rec.get("la_file") if isinstance(rec.get("la_file"), dict) else {}
    if la_arrays is None:
        bad.append("the file LA of card ADD110 could not be read")
    elif not la_disk_sha or len(str(lf.get("sha256") or "")) < 16 or not la_disk_sha.startswith(str(lf.get("sha256")).lower()):
        bad.append("the file LA on disk is not the one whose hash the record names")
    arrs, raws, vals = {}, {}, {}
    for arm in ARMS:
        fr = rf.get(arm)
        p = files.get(arm)
        if not isinstance(fr, dict) or not p or not os.path.isfile(p):
            bad.append("%s: no record entry or no file on disk" % arm)
            continue
        s = sha_file(p)
        want = str(fr.get("sha256") or "").lower()
        if len(want) < 16 or not s.startswith(want):
            bad.append("%s: the file on disk has another sha256 than the record names" % arm)
        z = load_npz(p)
        if fr.get("n_rows") is not None and int(fr.get("n_rows")) != len(z["gap"]):
            bad.append("%s: the file has %d rows, the record names %s" % (arm, len(z["gap"]), fr.get("n_rows")))
        has, gap, nbad, ndup = file_on_join(J, z)
        if nbad or ndup:
            bad.append("%s: %d rows name a clip or frame outside the join, %d rows repeat a key" % (arm, nbad, ndup))
        arrs[arm] = (has, gap)
        raws[arm] = raw_on_join(J, z)
    if base_arrays is not None:
        bhas, bgap, _, _ = file_on_join(J, base_arrays)
        bnamed, braw = raw_on_join(J, base_arrays)
        g, _lab = label_rows(J)
        err = error_rows(J, bhas)
        vals["n_error_rows"] = int(err.sum())
        for arm in ATTR_ARMS:
            if arm in arrs:
                eh, eg = compose_attr(J, arm, bhas, bgap)
                d = file_diff(eh, eg, *arrs[arm])
                vals["max_abs_diff_" + arm] = d
                if not (d <= TOL_CARRY):
                    bad.append("%s: the file differs from the definition of card section 2 (largest difference %s)" % (arm, d))
        if all(x in arrs for x in ATTR_ARMS):
            d = float(np.max(np.abs(arrs["NR"][1][err] + arrs["BR"][1][err] - g[err] - bgap[err]))) if err.any() else 0.0
            vals["max_abs_label_plus_s_plus_f_minus_base"] = d
            if not (d <= TOL_CARRY):
                bad.append("label plus s plus f is not the base gap on every error row (largest difference %s)" % d)
        if la_arrays is not None:
            lh, lg = file_on_join(J, la_arrays)[:2]
            ne = ~err
            for arm in ATTR_ARMS:
                if arm in arrs:
                    h, gp = arrs[arm]
                    if not np.array_equal(h[ne], lh[ne]):
                        bad.append("%s: the output flags differ from the file LA on rows that are not error rows" % arm)
                    else:
                        m_ = h & ne
                        d = float(np.max(np.abs(gp[m_] - lg[m_]))) if m_.any() else 0.0
                        vals["max_abs_diff_vs_LA_" + arm] = d
                        if not (d <= TOL_CARRY):
                            bad.append("%s: differs from the file LA on a row that is not an error row (largest difference %s)" % (arm, d))
        sm, sl = smooth_arms(J, bhas, bgap)
        for arm, ref in (("SM", sm), ("SL", sl)):
            if arm in arrs:
                d = file_diff(bhas, np.where(bhas, ref, np.nan), *arrs[arm])
                vals["max_abs_diff_" + arm] = d
                if not (d <= TOL_CARRY):
                    bad.append("%s: the file differs from the definition of card section 2 (largest difference %s; inf means other output flags than the base file)" % (arm, d))
                nm_, rw_ = raws[arm]
                mk = nm_ & bnamed & ~bhas
                if mk.any() and not np.array_equal(rw_[mk], braw[mk], equal_nan=True):
                    bad.append("%s: a row with no base output does not keep its base gap field" % arm)
        if not label_free(J, bhas, bgap):
            bad.append("SM and SL change when the label fields of the join are replaced")
    bad += course_checks()
    return {"pass": not bad, "failures": bad, "values": vals}


# ============================================================================================ b1 to b5
def ctl_b1(wrapper_sha16, check_text, hashes_text, run_shas):
    """b1: as the g4 of the model reader for the eight rollouts: the wrapper file is 28bc31f21ecb4ccb, the pod check record of card ADD102 names it with a pass, every run text names it"""
    lines = [x for x in (check_text or "").splitlines() if x.strip()]
    bad = []
    if wrapper_sha16 != WRAPPER_SHA:
        bad.append("the wrapper file is %s, not %s" % (wrapper_sha16, WRAPPER_SHA))
    if not lines or lines[-1].strip() != CHECK_LAST:
        bad.append("the last line of the pod check is %r" % (lines[-1].strip() if lines else None))
    if WRAPPER_SHA not in (hashes_text or ""):
        bad.append("the pod check hashes do not name %s" % WRAPPER_SHA)
    other = sorted(k for k, v in run_shas.items() if v is not None and not v.startswith(WRAPPER_SHA))
    if other:
        bad.append("run files with another wrapper hash: %s" % other)
    n_found = sum(1 for v in run_shas.values() if v is not None)
    if n_found != len(run_shas) or len(run_shas) != 8:
        bad.append("run files with a wrapper hash: %d of %d (eight expected)" % (n_found, len(run_shas)))
    return {"pass": not bad, "failures": bad, "wrapper_sha16": wrapper_sha16, "run_files_found": n_found}


def rollout_keys():
    return sorted("%s_s%d" % (x, m) for x in ARMS for m in SEEDS)


def b2_from_files(a, rec, J):
    """-> per-rollout dict for ctl_b2: the model reader's w2_one on each of the eight rollouts with its gap file (VGAPA_<ARM>.npz), the control run V0 for steps with no output"""
    per = {}
    for m in SEEDS:
        v0 = MD.lead_multi(MD.rows_of(MD.load_logs(a.pull, V0_RUN, m)))
        for x in ARMS:
            key = "%s_s%d" % (x, m)
            try:
                ref = MD.gapfile_ref(gapfile_path(a.gap_dir, x))
                sha = (((rec or {}).get("files") or {}).get(x) or {}).get("sha256")
                logs = MD.load_logs(a.pull, RUN[x], m)
                rows = MD.rows_of(logs)
                r = MD.w2_one(rows, ref, v0, meta_expect={"vis_gap_file_sha256": sha, "vis_closing": 1}, meta=MD.meta_of(logs[0]))
                per[key] = r
            except (SystemExit, OSError, ValueError, KeyError) as ex:
                per[key] = {"PASS": False, "failures": ["cannot read the rollout: %s: %s" % (type(ex).__name__, ex)]}
    return per


def ctl_b2(per_run):
    """b2: as the g5 of the model reader for the eight rollouts (gap received equals the gap file within 1e-3 m on steps with a lead and an output; steps with a lead and no output equal V0; meta names the gap file and
    the closing switch; the closing input is non-zero on a positive share of the steps with a lead)"""
    bad = []
    if sorted(per_run) != rollout_keys():
        bad.append("rollouts read: %s, expected %s" % (sorted(per_run), rollout_keys()))
    for k, v in sorted(per_run.items()):
        bad += ["%s: %s" % (k, x) for x in v.get("failures", [])]
        if v.get("PASS") is not True and not v.get("failures"):
            bad.append("%s: not pass" % k)
    shares = {k: {"share_of_rule_steps_with_a_lead": v.get("share_of_rule_steps_with_a_lead"), "share_of_lead_steps_with_nonzero_closing": v.get("share_of_lead_steps_with_nonzero_closing"),
                  "n_rows": v.get("n_rows"), "n_lead_rows": v.get("n_lead_rows")} for k, v in per_run.items()}
    return {"pass": not bad and bool(per_run), "failures": bad, "values": {"rollout_shares": shares}}


def ctl_b3(levels):
    """b3: the pooled levels of GEc, RLc, RR and LA recomputed by the reader equal 0.0062 / 0.5154 / 0.0839, 0.0000 / 0.4908 / 0.0714, 0.0954 / 0.4954 / 0.1339 and 0.0000 / 0.4877 / 0.0839 (contact, falling behind, unsafe) within 1e-4"""
    bad = []
    for k, want in EXPECTED_LEVELS.items():
        for c, v in want.items():
            got = ((levels or {}).get(k) or {}).get(c)
            if got is None or not np.isfinite(got) or abs(float(got) - v) > LEVEL_TOL + 1e-12:
                bad.append("%s %s: %s, expected %.4f within %g" % (k, c, got, v, LEVEL_TOL))
    return {"pass": not bad, "failures": bad}


def last_line(text):
    ls = [x for x in (text or "").splitlines() if x.strip()]
    return ls[-1] if ls else ""


def ctl_b4(text_gap, text_read, mtime_gap, mtime_read, queue_start):
    """b4: the last non-empty line of each lead_pytest.out holds `passed` (a count above 0) and neither `failed` nor `error`; and the newer of the two files is older than the first start line of the queue
    (seconds since the epoch of the file modification times and of the first token of GAPERR10K_START.txt)"""
    bad = []
    for nm, t in (("kit_gap", text_gap), ("kit_read", text_read)):
        ln = last_line(t)
        m = re.findall(r"(\d+) passed", ln)
        if not (m and int(m[-1]) > 0) or re.search(r"failed|error", ln, re.I):
            bad.append("%s: the last line of the test output is %r" % (nm, ln[:120]))
    mg, mr, qs = to_epoch(mtime_gap), to_epoch(mtime_read), to_epoch(queue_start)
    if mg is None or mr is None or qs is None:
        bad.append("a time is missing (gap tests %s, read tests %s, queue start %s)" % (mtime_gap, mtime_read, queue_start))
    elif not (max(mg, mr) < qs):
        bad.append("the newer test output (%s) is not older than the first start line of the queue (%s)" % (max(mg, mr), qs))
    return {"pass": not bad, "failures": bad}


def ctl_b5(grid_all_pass, strat_a, strat_b):
    """b5: as the g8 of the model reader: the engine's own controls of the grid read and of the strategic grid pass"""
    r = MD.ctl_g8(grid_all_pass, strat_a, strat_b)
    return {"pass": bool(r["PASS"]), "failures": [] if r["PASS"] else ["grid_all_pass %s, strategic stage A %s, stage B %s" % (grid_all_pass, strat_a, strat_b)], "grid_all_pass": grid_all_pass,
            "strategic_stage_A_pass": strat_a, "strategic_stage_B_pass": strat_b}


def run_shas_of(pull):
    return {"%s_s%d" % (x, m): MD.run_file_sha(pull, RUN[x], m) for x in ARMS for m in SEEDS}


def _rec_path(rec, key):
    d = rec.get(key) if isinstance(rec, dict) else None
    return d.get("path") if isinstance(d, dict) else ""


def compute_controls(a, rec, J, m, wsha, join_sha):
    """the six controls from the files on disk and the engine measures `m` (levels, grid_all_pass, strategic_a, strategic_b). The base file is read at --base-file, else at base.path of the record; the file LA
    at --la-file, else at la_file.path of the record; the gap file of an arm by the name VGAPA_<ARM>.npz in --gap-dir"""
    base = bsha = la = lsha = None
    bp = getattr(a, "base_file", "") or _rec_path(rec, "base")
    if bp and os.path.isfile(bp):
        base, bsha = load_npz(bp), sha_file(bp)
    lp = getattr(a, "la_file", "") or _rec_path(rec, "la_file")
    if lp and os.path.isfile(lp):
        la, lsha = load_npz(lp), sha_file(lp)
    files = {x: gapfile_path(a.gap_dir, x) for x in ARMS}
    c = {}
    c["b0"] = ctl_b0(rec, files, J, join_sha, os.path.basename(a.eval_join), base, bsha, la, lsha)
    c["b1"] = ctl_b1(wsha, read_text(a.wrapper_check), read_text(a.wrapper_hashes), run_shas_of(a.pull))
    c["b2"] = ctl_b2(b2_from_files(a, rec, J))
    c["b3"] = ctl_b3(m["levels"])
    c["b4"] = ctl_b4(read_text(a.tests_gap), read_text(a.tests_read), a.tests_mtime_gap, a.tests_mtime_read, a.queue_start)
    c["b5"] = ctl_b5(m["grid_all_pass"], m["strategic_a"], m["strategic_b"])
    return c


def assemble_controls(controls, payload, inputs):
    ok = sorted(controls) == sorted(CONTROLS) and all(isinstance(v, dict) and v.get("pass") is True for v in controls.values())
    vals = {k: v["values"] for k, v in controls.items() if isinstance(v, dict) and "values" in v}
    flat = {k: ({kk: vv for kk, vv in v.items() if kk != "values"} if isinstance(v, dict) else v) for k, v in controls.items()}
    pl = dict(payload)
    pl["control_values"] = vals
    res = {"card": CARD, "tier": TIER, "controls": flat, "all_pass": ok, "inputs": inputs}
    res["payload" if ok else "withheld"] = pl
    return res


# ============================================================================================ names (two separate derivations)
def _lbl(m):
    return dict(MEASURES)[m]


def names_of(payload):
    """card section 3: per arm X and measure M against GEc LOWERS / RAISES / NOT-SEPARATED; the parts from falling behind of NR and BR; per smoothing arm PAYS-SAFETY / RECOVERS / NO-GAIN"""
    pr = payload["pairs"]
    out = {}
    for x in ARMS:
        for m, label in MEASURES:
            iv = pr[x + "-VS-GEc"][m]["interval"]
            if iv[1] < 0.0:
                out["%s_%s" % (x, label)] = "%s-%s-LOWERS-%s" % (PRE, x, label)
            elif iv[0] > 0.0:
                out["%s_%s" % (x, label)] = "%s-%s-RAISES-%s" % (PRE, x, label)
            else:
                out["%s_%s" % (x, label)] = "%s-%s-%s-NOT-SEPARATED" % (PRE, x, label)
    nr_low = pr["NR-VS-GEc"]["F"]["interval"][1] < 0.0
    br_low = pr["BR-VS-GEc"]["F"]["interval"][1] < 0.0
    if nr_low and not br_low:
        out["PARTS"] = PRE + "-FAST-PART-CARRIES"
    elif br_low and not nr_low:
        out["PARTS"] = PRE + "-SLOW-PART-CARRIES"
    elif nr_low and br_low:
        out["PARTS"] = PRE + "-BOTH-PARTS-CARRY"
    else:
        out["PARTS"] = PRE + "-PARTS-NOT-SEPARATED"
    for x in SMOOTH_ARMS:
        if pr[x + "-VS-GEc"]["C"]["interval"][0] > 0.0 or pr[x + "-VS-GEc"]["U"]["interval"][0] > 0.0:
            out["SMOOTH_" + x] = "%s-%s-PAYS-SAFETY" % (PRE, x)
        elif pr[x + "-VS-GEc"]["F"]["interval"][1] < 0.0:
            out["SMOOTH_" + x] = "%s-%s-RECOVERS" % (PRE, x)
        else:
            out["SMOOTH_" + x] = "%s-%s-NO-GAIN" % (PRE, x)
    return out


def _sg(iv):
    return -1 if iv[1] < 0.0 else (1 if iv[0] > 0.0 else 0)


def derive_all(payload):
    """independent second derivation of every name from the signs of the intervals (written apart from names_of)"""
    pr = payload["pairs"]
    out = {}
    word = {-1: "%s-%s-LOWERS-%s", 1: "%s-%s-RAISES-%s", 0: "%s-%s-%s-NOT-SEPARATED"}
    sg = {}
    for x in ARMS:
        for m, label in MEASURES:
            s = _sg(pr[x + "-VS-GEc"][m]["interval"])
            sg[(x, m)] = s
            out["%s_%s" % (x, label)] = word[s] % (PRE, x, label)
    table = {(True, False): "-FAST-PART-CARRIES", (False, True): "-SLOW-PART-CARRIES", (True, True): "-BOTH-PARTS-CARRY", (False, False): "-PARTS-NOT-SEPARATED"}
    out["PARTS"] = PRE + table[(sg[("NR", "F")] == -1, sg[("BR", "F")] == -1)]
    for x in SMOOTH_ARMS:
        if 1 in (sg[(x, "C")], sg[(x, "U")]):
            tail = "-PAYS-SAFETY"
        elif sg[(x, "F")] == -1:
            tail = "-RECOVERS"
        else:
            tail = "-NO-GAIN"
        out["SMOOTH_" + x] = "%s-%s%s" % (PRE, x, tail)
    return out


def all_names(payload):
    got = names_of(payload)
    exp = derive_all(payload)
    diff = ["%s: got %s, expected %s" % (k, got.get(k), v) for k, v in sorted(exp.items()) if got.get(k) != v]
    diff += ["%s: name present that the card does not have" % k for k in sorted(got) if k not in exp]
    if diff:
        raise SystemExit("REFUSED: the names are not the ones the intervals give: %s" % diff)
    return got


def flat_names(names):
    out = []
    for k in sorted(names):
        v = names[k]
        out += list(v) if isinstance(v, list) else [v]
    return out


# ============================================================================================ the pod part (grid read and strategic read)
def measure(a, E, log):
    """grid read (levels, pairs with intervals, families) of the four arms and GEc, RR, RLc against each other, then the strategic grid at T1-kin. Engine calls as in the model reader."""
    H, Q3, RH = E
    g30 = R.jl(os.path.join(a.grid_dir, "grid_p30.json"))
    J = H.load_join()
    keys = R.grid_keys(g30, [str(c) for c in J["clip"]])
    runs = {x: RUN[x] for x in ARMS}
    runs.update(REFS)
    for k, r in runs.items():
        R.ALL_RUNS.setdefault(r, ("RLc" if k == "RLc" else "RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    R.ALL_RUNS.setdefault(EC.SETTINGS["AGE"]["runs"]["RR"], ("RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    pairs_def = {}
    for x in ARMS:
        for ref in ("GEc", "RLc", "RR", "LA"):
            pairs_def["%s-VS-%s" % (x, ref)] = (RUN[x], REFS[ref])
    spread = "R0Hb20"
    run_list = [spread] + sorted(set(runs.values()))
    G = R.read_grid(a, E, "30", keys, run_list, spread, pairs_def, {}, log)
    lv = {k: {c: float(np.mean(G["levels"][r][c])) for c in ("C", "U", "F")} for k, r in runs.items()}
    lv_seed = {k: {c: [float(v) for v in np.asarray(G["levels"][r][c]).reshape(-1)[:2]] for c in ("C", "U", "F")} for k, r in runs.items()}
    pairs = {nm: R.ucf(p["headway"]) for nm, p in G["pairs"].items()}
    fams = {nm: {f_: R.reading(R.cells_of(p["family"]), f_) for f_ in R.NAMED} for nm, p in G["pairs"].items()}
    strat_runs = set(RUN.values()) | {REFS[k] for k in STRAT_REFS}
    p_of = {"R0": 0}
    p_of.update({r: 30 for r in strat_runs})
    SR, cells_s, eid_s, S = R.strategic_grid(a, keys, sorted(p_of), p_of, log, {})
    sa, sb = bool(SR.get("stage_A_pass")), bool(SR.get("stage_B_pass"))
    strat = {"controls": dict({k: v for k, v in SR["controls"].items() if k != "s3"}, s3=SR["controls"]["s3"].get("pass")), "pairs": {}}
    if sa and sb:
        for x in ARMS:
            for ref in ("GEc", "RLc"):
                pr = S.pair_read(cells_s, RUN[x], REFS[ref], eid_s, ticks=(2,))
                pos = pr["tick2"]["pos"]
                strat["pairs"]["%s-VS-%s" % (x, ref)] = {"A": RUN[x], "B": REFS[ref], "pos_30_50_80m_tick_t0_plus_20": [{"arc_m": arc, "mean": b["mean"], "lo": b["lo"], "hi": b["hi"],
                                                         "verdict": b["verdict"], "n_per_seed": b.get("n_per_seed")} for arc, b in zip(S.ARCS, pos)],
                                                         "reading": R.family_of([c["verdict"] for c in pos])}
    return {"levels": lv, "levels_per_seed": lv_seed, "pairs": pairs, "families": fams, "grid_all_pass": G["controls"].get("ALL_PASS") is True, "strategic": strat, "strategic_a": sa, "strategic_b": sb}


def run_controls(a):
    elog = a.out + ".engine.log"
    if os.path.exists(elog):
        raise SystemExit("REFUSED: %s exists" % elog)
    with io.open(elog, "w", encoding="utf-8", newline="\n") as fh, contextlib.redirect_stdout(fh):
        E = R.engine(a)
        m = measure(a, E, lambda s: print(s, flush=True))
    rec = read_json(a.gap_record)
    J = load_npz(a.eval_join)
    jsha = sha_file(a.eval_join)
    wsha = sha_file(a.wrapper_file)[:16] if a.wrapper_file and os.path.isfile(a.wrapper_file) else None
    controls = compute_controls(a, rec, J, m, wsha, jsha)
    rf = (rec or {}).get("files") or {}
    counts = {x: {k: (rf.get(x) or {}).get(k) for k in ("n_rows", "n_diff_rows", "n_label_value_rows")} for x in ARMS}
    payload = {"levels": m["levels"], "levels_per_seed": m["levels_per_seed"], "pairs": m["pairs"], "families": m["families"], "strategic": m["strategic"], "family_note": FAM_NOTE,
               "gapfile_counts_from_record": counts, "t0_from_record": (rec or {}).get("t0")}
    inputs = {"model_reader": MODEL_PATH and sha_file(MODEL_PATH), "this_reader": sha_file(os.path.abspath(__file__)), "eval_join_sha256": jsha, "wrapper_file_sha16": wsha,
              "gap_record_sha256": sha_file(a.gap_record) if os.path.isfile(a.gap_record) else None,
              "gap_files": {x: sha_file(gapfile_path(a.gap_dir, x)) for x in ARMS if os.path.isfile(gapfile_path(a.gap_dir, x))}}
    return assemble_controls(controls, payload, inputs)


def fmt_pair(nm, p):
    return "; ".join("%s %s" % (label, json.dumps(p[m], sort_keys=True, default=str)) for m, label in MEASURES)


def print_sections(payload, names):
    L = ["TIER " + TIER]
    L += flat_names(names)
    lv, ls = payload["levels"], payload["levels_per_seed"]
    for k in sorted(lv):
        L.append("level %s pooled C %.4f U %.4f F %.4f; per model seed %s" % (k, lv[k]["C"], lv[k]["U"], lv[k]["F"], json.dumps(ls.get(k), sort_keys=True)))
    pr, st, fm = payload["pairs"], (payload.get("strategic") or {}).get("pairs") or {}, payload.get("families") or {}
    for x in ARMS:
        for ref in ("GEc", "RLc", "RR", "LA"):
            nm = "%s-VS-%s" % (x, ref)
            L.append("%s: %s" % (nm, fmt_pair(nm, pr[nm])))
    for x in ARMS:
        for ref in ("GEc", "RLc"):
            nm = "%s-VS-%s" % (x, ref)
            L.append("families %s: %s strategic %s" % (nm, json.dumps(fm.get(nm) or {}, sort_keys=True), (st.get(nm) or {}).get("reading", "not read")))
    for x in ARMS:
        L.append("gap file %s: " % x + json.dumps((payload.get("gapfile_counts_from_record") or {}).get(x), sort_keys=True))
    t0 = payload.get("t0_from_record") if isinstance(payload.get("t0_from_record"), dict) else {}
    for rg in T0_RANGES:
        L.append("t0 %s (T0 on the evaluation frames, recorded frames, not driving performance): %s" % (rg, json.dumps(t0.get(rg), sort_keys=True)))
    sh = (payload.get("control_values") or {}).get("b2", {}).get("rollout_shares") or {}
    for k in sorted(sh):
        L.append("rollout %s: %s" % (k, json.dumps(sh[k], sort_keys=True)))
    return L


# ============================================================================================ synthetic world for the selftest and the tests
def make_world(base, fault=None):
    """a tiny world (3 clips of 20 frames) in which every control passes (fault None), or exactly the one control `fault` (b0 to b5) fails. The gap files are written here by a generation apart from the composition
    routines of the reader (loops, statistics.median, np.polyfit). -> (a, rec, J, m, wsha, join_sha)"""
    import statistics
    base = str(base)
    os.makedirs(base, exist_ok=True)
    gd, pull, lad = os.path.join(base, "gapfiles"), os.path.join(base, "pull"), os.path.join(base, "la")
    for d_ in (gd, pull, lad):
        os.makedirs(d_, exist_ok=True)
    nk = 20
    n, off = np.array([nk] * 3), np.array([0, nk, 2 * nk])
    nrow = 3 * nk
    fr, ci = np.tile(np.arange(nk), 3), np.repeat(np.arange(3), nk)
    lab = np.where(fr <= 15, 12.0 + 2.0 * fr, 65.0)
    lab[fr == 17] = np.nan
    lab[fr == 19] = 20.0
    valid = np.ones(nrow, np.int64)
    valid[fr == 16] = 0
    J = {"clip": np.array(["c0", "c1", "c2"]), "off": off, "n": n, "lead_gap": lab, "lead_valid": valid}
    jp = os.path.join(base, JOIN_NAME)
    np.savez(jp, **J)
    jsha = sha_file(jp)
    bgap = 15.0 + 1.2 * fr + 0.7 * (-1.0) ** fr + 0.6 * np.sin(0.9 * fr)
    bgap[(ci == 1) & (fr >= 10)] += 8.0
    bhas = np.ones(nrow, bool)
    bhas[[3, nk, 2 * nk + 10]] = False
    bz = {"clip_index": ci, "frame": fr, "gap": np.where(bhas, bgap, np.nan), "has_out": bhas, "clips": J["clip"], "seed": np.int64(-1)}
    bp = os.path.join(base, "VGAPLE.npz")
    np.savez(bp, **bz)
    labrow = [bool(valid[r] == 1 and np.isfinite(lab[r]) and 0.0 <= lab[r] <= 60.0) for r in range(nrow)]
    e = {r: float(bgap[r] - lab[r]) for r in range(nrow) if labrow[r] and bhas[r]}
    runs, cur = [], []
    for r in sorted(e):
        if cur and r == cur[-1] + 1 and r // nk == cur[-1] // nk:
            cur.append(r)
        else:
            if cur:
                runs.append(cur)
            cur = [r]
    if cur:
        runs.append(cur)
    slow = {}
    for run in runs:
        for r in run:
            near = [q for q in run if abs(q - r) <= 5]
            slow[r] = sum(e[q] for q in near) / len(near)
    gaps, hass = {}, {}
    for arm in ("NR", "BR"):
        gp, hs = [float(v) if bhas[r] else float("nan") for r, v in enumerate(bgap)], [bool(v) for v in bhas]
        for r in range(nrow):
            if labrow[r] and not bhas[r]:
                gp[r], hs[r] = float(lab[r]), True
        for r in e:
            gp[r] = float(lab[r]) + (slow[r] if arm == "NR" else e[r] - slow[r])
        gaps[arm], hass[arm] = np.array(gp), np.array(hs)
    for arm in ("SM", "SL"):
        gp = [float("nan")] * nrow
        for r in range(nrow):
            if not bhas[r]:
                continue
            run = [r]
            while len(run) < 5 and run[0] - 1 >= (r // nk) * nk and bhas[run[0] - 1] and abs(bgap[run[0] - 1] - bgap[run[0]]) <= 3.0:
                run.insert(0, run[0] - 1)
            vs = [float(bgap[q]) for q in run]
            if arm == "SM":
                gp[r] = statistics.median(vs)
            elif len(vs) >= 3:
                gp[r] = float(np.polyval(np.polyfit(np.arange(len(vs)), vs, 1), len(vs) - 1))
            else:
                gp[r] = vs[-1]
        gaps[arm], hass[arm] = np.array(gp), bhas.copy()
    lrow = np.array(labrow)
    lz = {"clip_index": ci, "frame": fr, "gap": np.where(lrow, lab, np.where(bhas, bgap, np.nan)), "has_out": lrow | bhas, "clips": J["clip"], "seed": np.int64(-1)}
    lap = os.path.join(lad, "VGAPA_LA.npz")
    np.savez(lap, **lz)
    t0 = {rg: {"n_error_rows": 3, "rms_e": 1.0, "rms_s": 0.8, "rms_f": 0.6, "rms_sm": 0.9, "rms_sl": 0.7} for rg in T0_RANGES}
    rec = {"pass_b0": fault != "b0", "clauses": {"c1": {"pass": True}, "c2": {"pass": True}}, "files": {}, "base": {"path": bp, "sha256": sha_file(bp), "named_in_record_add106": True},
           "la_file": {"path": lap, "sha256": sha_file(lap)}, "eval_join_sha256": jsha, "t0": t0, "written_utc": "2026-11-23T00:00:00Z"}
    arm_gap = {}
    for arm in ARMS:
        has, gap = hass[arm], np.where(hass[arm], gaps[arm], np.nan)
        arm_gap[arm] = (has, gap)
        order = np.random.RandomState(4).permutation(nrow) if arm == "SL" else np.arange(nrow)
        z = {"clip_index": ci[order], "frame": fr[order], "gap": gap[order], "has_out": has[order], "clips": J["clip"], "seed": np.int64(-1)}
        p = gapfile_path(gd, arm)
        np.savez(p, **z)
        diff = int(np.sum((has != bhas) | (np.nan_to_num(gap, nan=-1.0) != np.nan_to_num(np.where(bhas, bgap, np.nan), nan=-1.0))))
        rec["files"][arm] = {"n_rows": nrow, "n_diff_rows": diff, "n_label_value_rows": int(sum(1 for r in range(nrow) if has[r] and labrow[r] and abs(gap[r] - lab[r]) <= 1e-9)),
                             "path": p, "sha256": sha_file(p)}
    json.dump(rec, io.open(os.path.join(gd, "GAPERR_GAPFILES_RECORD.json"), "w", encoding="utf-8"))
    v0 = 20.0 + 0.1 * np.arange(nrow)
    clos = np.where(np.arange(nrow) % 2 == 0, 0.5, 0.0)
    half = nrow // 2

    def write_run(run, m, gap_from, sha, wrap_hash):
        d = os.path.join(pull, "I10%s_s%d" % (run, m), "ticks")
        os.makedirs(d, exist_ok=True)
        for nm, sl in (("rule_log_I10%s_s%d.npz", slice(0, half)), ("rule_log_mid_I10%s_s%d.npz", slice(half, nrow))):
            np.savez(os.path.join(d, nm % (run, m)), ci=ci[sl], j=fr[sl], present=np.ones(len(ci[sl]), bool), g_src=gap_from[sl].copy(), c_src=clos[sl], vis_gap_file_sha256=np.array(sha or "none"),
                     vis_closing=np.array(1))
        io.open(os.path.join(d, "run_I10%s_s%d.txt" % (run, m)), "w", encoding="utf-8").write("run text\nwrapper sha256 %s\n" % wrap_hash)

    wrap = WRAPPER_SHA + "0" * 48
    for m in SEEDS:
        write_run(V0_RUN, m, v0, None, wrap)
        for x in ARMS:
            gs = np.where(arm_gap[x][0], arm_gap[x][1], v0) + 1e-5
            if fault == "b2" and x == "NR" and m == 0:
                gs[2] += 0.5
            write_run(RUN[x], m, gs, rec["files"][x]["sha256"], wrap if not (fault == "b1" and x == "SL" and m == 1) else "f" * 64)
    chk = os.path.join(base, "CHECK_WRAPPER_V_DONE.txt")
    io.open(chk, "w", encoding="utf-8").write("a\n" + (CHECK_LAST if fault != "b1" else "ZZ CHECK-WRAPPER-V FAIL") + "\n\n")
    hs = os.path.join(base, "CHECK_WRAPPER_V.hashes")
    io.open(hs, "w", encoding="utf-8").write("sha %s\n" % WRAPPER_SHA)
    tg, tr = os.path.join(base, "gap_pytest.out"), os.path.join(base, "read_pytest.out")
    io.open(tg, "w", encoding="utf-8").write("....\n== 12 passed in 3.1s ==\n\n")
    io.open(tr, "w", encoding="utf-8").write("....\n== 30 passed in 9.0s ==\n")

    class A:
        pass
    a = A()
    a.pull, a.gap_dir, a.eval_join, a.base_file, a.la_file = pull, gd, jp, "", ""
    a.wrapper_check, a.wrapper_hashes, a.tests_gap, a.tests_read = chk, hs, tg, tr
    a.tests_mtime_gap, a.tests_mtime_read, a.queue_start = 1000.0, 1010.0, 1100.0
    if fault == "b4":
        a.tests_mtime_read = 1200.0
    m = {"levels": {k: dict(v) for k, v in EXPECTED_LEVELS.items()}, "grid_all_pass": True, "strategic_a": True, "strategic_b": fault != "b5"}
    if fault == "b3":
        m["levels"]["LA"]["F"] += 0.001
    return a, rec, J, m, WRAPPER_SHA, jsha


def mk_payload(f_gec=None, f_rl=None, c=None, u=None):
    """a synthetic payload of pairs: f_gec[arm] and f_rl[arm] are the intervals of falling behind against GEc and RLc, c[arm] and u[arm] those of contact and unsafe against GEc; default 'holds 0'"""
    z = [-0.01, 0.01]
    pr = {}
    for x in ARMS:
        for ref in ("GEc", "RLc", "RR", "LA"):
            pr["%s-VS-%s" % (x, ref)] = {"F": {"interval": list(z)}, "C": {"interval": list(z)}, "U": {"interval": list(z)}}
        for ref, d in (("GEc", f_gec), ("RLc", f_rl)):
            if d and x in d:
                pr["%s-VS-%s" % (x, ref)]["F"]["interval"] = list(d[x])
        for m, d in (("C", c), ("U", u)):
            if d and x in d:
                pr["%s-VS-GEc" % x][m]["interval"] = list(d[x])
    return {"pairs": pr}


def selftest():
    ok = True
    bind(os.environ.get("GAPERR_MODEL_READER", ""))
    with tempfile.TemporaryDirectory() as td:
        for fault in (None,) + CONTROLS:
            a, rec, J, m, wsha, jsha = make_world(os.path.join(td, "w_%s" % fault), fault)
            c = compute_controls(a, rec, J, m, wsha, jsha)
            flags = {k: v["pass"] for k, v in c.items()}
            want = {k: (k != fault) for k in CONTROLS}
            if flags != want:
                print("SELFTEST fault %s: flags %s" % (fault, flags))
                ok = False
    ok &= MD.WRAPPER_SHA == WRAPPER_SHA and MD.CHECK_LAST == CHECK_LAST and MD.V0_RUN == V0_RUN and MD.BANKED["RR"] == REFS["RR"] and MD.BANKED["RLc"] == REFS["RLc"] and MD.RUN["GEc"] == REFS["GEc"]
    low, high = [-0.05, -0.01], [0.01, 0.05]
    nm = all_names(mk_payload({"NR": low}))
    ok &= nm["PARTS"] == PRE + "-FAST-PART-CARRIES" and nm["NR_FALLING-BEHIND"] == PRE + "-NR-LOWERS-FALLING-BEHIND"
    nm = all_names(mk_payload({"BR": low}))
    ok &= nm["PARTS"] == PRE + "-SLOW-PART-CARRIES"
    nm = all_names(mk_payload({"NR": low, "BR": low}))
    ok &= nm["PARTS"] == PRE + "-BOTH-PARTS-CARRY"
    nm = all_names(mk_payload())
    ok &= nm["PARTS"] == PRE + "-PARTS-NOT-SEPARATED" and nm["SMOOTH_SM"] == PRE + "-SM-NO-GAIN" and nm["SMOOTH_SL"] == PRE + "-SL-NO-GAIN"
    nm = all_names(mk_payload({"SM": low, "SL": low}, c={"SM": high}))
    ok &= nm["SMOOTH_SM"] == PRE + "-SM-PAYS-SAFETY" and nm["SMOOTH_SL"] == PRE + "-SL-RECOVERS" and nm["SM_CONTACT"] == PRE + "-SM-RAISES-CONTACT"
    print("ZZ READ-GAPERR10K SELFTEST %s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage", choices=("controls", "names"))
    ap.add_argument("--selftest", action="store_true")
    for nme in ("pull", "code", "grid-dir", "w", "tree", "wrapper-file", "wrapper-check", "wrapper-hashes", "controls", "eval-join", "gap-record", "gap-dir", "tests-gap", "tests-read",
                "tests-mtime-gap", "tests-mtime-read", "queue-start", "model-reader", "base-file", "la-file", "fa"):
        ap.add_argument("--" + nme, default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.stage or not a.out:
        print("REFUSED: --stage and --out are required")
        return 2
    bind(a.model_reader)
    a.n_boot_control = 300
    if os.path.exists(a.out):
        print("REFUSED: %s exists" % a.out)
        return 2
    if a.stage == "controls":
        fa = a.fa or WS + "/gaperr"
        a.gap_dir = a.gap_dir or fa + "/gapfiles"
        a.gap_record = a.gap_record or a.gap_dir + "/GAPERR_GAPFILES_RECORD.json"
        a.eval_join = a.eval_join or WS + "/hierrep/pins/" + JOIN_NAME
        a.tests_gap = a.tests_gap or fa + "/kit_gap/lead_pytest.out"
        a.tests_read = a.tests_read or fa + "/kit_read/lead_pytest.out"
        res = run_controls(a)
        io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o)))
        for k in CONTROLS:
            print("%s %s" % (k, "PASS" if (res["controls"].get(k) or {}).get("pass") else "FAIL"))
        print("ZZ READ-GAPERR10K controls all_pass %s" % res["all_pass"])
        return 0
    cj = read_json(a.controls) or {}
    if cj.get("card") != CARD or cj.get("all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != sorted(CONTROLS) \
            or any((v or {}).get("pass") is not True for v in cj.get("controls", {}).values()):
        print("REFUSED: the controls file is not all-pass with b0 to b5 present: no name is written")
        print("ZZ READ-GAPERR10K names refused")
        return 3
    logp = a.log or (os.path.splitext(a.out)[0] + "_names.log")
    if os.path.exists(logp):
        print("REFUSED: %s exists" % logp)
        return 2
    names = all_names(cj["payload"])
    res = {"card": CARD, "tier": TIER, "controls_file": os.path.abspath(a.controls), "controls_sha256": sha_file(a.controls), "names": names, "values": cj["payload"]}
    io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1))
    lines = print_sections(cj["payload"], names) + ["ZZ READ-GAPERR10K names done"]
    io.open(logp, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    for ln in lines:
        print(ln)
    return 0


try:
    bind(os.environ.get("GAPERR_MODEL_READER", ""))
except ImportError as _ex:                      # the model reader is bound again in main() from --model-reader; a missing one is reported there
    print("NOTE: %s" % _ex, file=sys.stderr)

if __name__ == "__main__":
    sys.exit(main())
