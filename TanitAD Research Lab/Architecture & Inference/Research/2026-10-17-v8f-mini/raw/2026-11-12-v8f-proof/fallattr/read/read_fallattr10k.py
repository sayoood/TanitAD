#!/usr/bin/env python
"""read_fallattr10k.py -- read of card PREREG_V8F_PROOF_ADD110 (FALLATTR10K): where in the gap range does the price in falling behind of the camera-fed lead rule come from.
Arms LA, LN, LM, LF (the label gap composed into the gap file of the base GEc on all / near / middle / far label rows), two model seeds each, eight rollouts at T1-kin/hb0.2 (kinematic rollout on the
model's own actions), pre-roll of 30 steps (6 s); references GEc, RLc, RR (banked rollouts); control run V0 of card ADD102.
Derived from read_cropscale10k.py (card ADD106), which is imported as a sibling module (path by --model-reader) and called for everything it already does; only what this card adds is written here.
Stage controls: runs a0 to a5, the grid read and the STRATEGIC family; writes FALLATTR10K_CONTROLS.json (payload when all pass, `withheld` otherwise) and prints only `a<i> PASS|FAIL` lines and a last line
  `ZZ READ-FALLATTR10K controls all_pass <True|False>`. The engine's own output goes to <out>.engine.log.
Stage names: refuses unless the controls file has a0 to a5 all with pass true; writes FALLATTR10K_RESULT.json and a names log.
--selftest: synthetic world; last line `ZZ READ-FALLATTR10K SELFTEST OK` or FAIL."""
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
    for p in (path, os.environ.get("FALLATTR_MODEL_READER", ""), POD_MODEL, DEV_MODEL):
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


CARD, PRE = "PREREG_V8F_PROOF_ADD110", "FALLATTR10K"
SEEDS = (0, 1)
ARMS = ("LA", "LN", "LM", "LF")
RANGE_ARMS = ("LN", "LM", "LF")
RANGE_WORD = {"LN": "NEAR", "LM": "MIDDLE", "LF": "FAR"}
RUN = {x: "RQ50KF5P30Hb20a" + x for x in ARMS}
REFS = {"GEc": "RQ50KF5P30Hb20gGEc", "RR": "RQ50KF5P30Hb20", "RLc": "RLcKF5P30Hb20"}
V0_RUN = "RQ50KF5P30Hb20vV0"
MEASURES = (("F", "FALLING-BEHIND"), ("C", "CONTACT"), ("U", "UNSAFE"))
EXPECTED_LEVELS = {"GEc": {"C": 0.0062, "F": 0.5154, "U": 0.0839}, "RLc": {"C": 0.0, "F": 0.4908, "U": 0.0714}, "RR": {"C": 0.0954, "F": 0.4954, "U": 0.1339}}
LEVEL_TOL = 1e-4
WRAPPER_SHA = "28bc31f21ecb4ccb"
CHECK_LAST = "ZZ CHECK-WRAPPER-V PASS"
JOIN_NAME = "join_eval141.npz"
TOL_CARRY = 1e-9
TOL_LABEL_SHARE = 1e-3
GAP_MAX_M = 60.0
NEAR_M, MID_M = 20.0, 40.0
CONTROLS = tuple("a%d" % i for i in range(6))
TIER = ("T1-kin/hb0.2 (kinematic rollout on the model's own actions, executor of record v8f_t1_eval_hb.py with the heading bound 0.2 1/m), pre-roll of 30 steps (6 s); "
        "the label gap enters the gap files as a tool of attribution only: no arm of this card is a candidate")
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


# ============================================================================================ the join, ranges and a0
def label_rows(J):
    """-> (gap, label): the join's lead gap and the rows that carry a label lead (valid flag 1, finite gap from 0 to 60 m)"""
    g = np.asarray(J["lead_gap"], np.float64)
    return g, (np.asarray(J["lead_valid"]) == 1) & np.isfinite(g) & (g >= 0.0) & (g <= GAP_MAX_M)


def in_range(arm, g):
    with np.errstate(invalid="ignore"):
        if arm == "LA":
            return np.ones(len(g), bool)
        if arm == "LN":
            return g < NEAR_M
        if arm == "LM":
            return (g >= NEAR_M) & (g < MID_M)
        if arm == "LF":
            return g >= MID_M
    raise ValueError(arm)


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


def carried_rows(J, arm, has, gap):
    """the rows that carry the label gap in the file of `arm`: label rows of the arm's range with an output equal to the label gap within 1e-9"""
    g, lab = label_rows(J)
    with np.errstate(invalid="ignore"):
        return lab & in_range(arm, g) & has & (np.abs(gap - g) <= TOL_CARRY)


def composition_diff(J, arm, has, gap, bhas, bgap):
    """largest absolute difference of a composed file from what the card says (label gap where the range holds, gap and output flag of the base elsewhere); inf when the output flags differ"""
    g, lab = label_rows(J)
    sel = lab & in_range(arm, g)
    ehas = np.where(sel, True, bhas)
    egap = np.where(sel, g, bgap)
    if not np.array_equal(ehas, has):
        return float("inf")
    both = ehas & has
    if not both.any():
        return 0.0
    return float(np.max(np.abs(egap[both] - gap[both])))


def ctl_a0(rec, files, J, join_sha, join_name, base_arrays=None, base_disk_sha=None):
    """a0 (the gap files are what the card says). rec: the gap-file record; files {arm: path}; J: the evaluation join; join_sha, join_name: of the join file; base_arrays: the base file as a dict of arrays, base_disk_sha its hash.
    Clauses: the record has pass_a0 true and every clause pass true; each file on disk has the hash the record names and the row count of the record; every composed file equals, on every join row, the label gap where its
    range holds and the gap and output flag of the base elsewhere (by this routine, at most 1e-9 m); the rows that carry the label gap in LN, LM, LF are disjoint and together those of LA; the base file is the one the record
    names (named_in_record_add106 true, its hash on disk equal); the join is the pinned one (name and the hash the record names)."""
    bad = []
    if not isinstance(rec, dict):
        return {"pass": False, "failures": ["no gap-file record"]}
    if rec.get("pass_a0") is not True:
        bad.append("the record does not read pass_a0 true")
    cl = rec.get("clauses")
    if not isinstance(cl, dict) or not cl:
        bad.append("the record has no clauses")
    else:
        for k, v in sorted(cl.items()):
            if not (isinstance(v, dict) and v.get("pass") is True):
                bad.append("record clause %s is not pass" % k)
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
    arrs, vals = {}, {}
    for arm in ARMS:
        fr = (rec.get("files") or {}).get(arm)
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
    carried = {}
    if base_arrays is not None:
        bhas, bgap, _, _ = file_on_join(J, base_arrays)
        for arm, (has, gap) in arrs.items():
            d = composition_diff(J, arm, has, gap, bhas, bgap)
            vals["max_abs_diff_" + arm] = d
            if not (d <= TOL_CARRY):
                bad.append("%s: the file differs from the composition the card names (largest difference %s)" % (arm, d))
    for arm, (has, gap) in arrs.items():
        carried[arm] = carried_rows(J, arm, has, gap)
        vals["n_rows_carrying_label_gap_" + arm] = int(carried[arm].sum())
    if all(a in carried for a in ARMS):
        cn, cm, cf, ca = carried["LN"], carried["LM"], carried["LF"], carried["LA"]
        if int((cn & cm).sum() + (cn & cf).sum() + (cm & cf).sum()):
            bad.append("the label-gap rows of LN, LM and LF are not disjoint")
        if not np.array_equal(cn | cm | cf, ca):
            bad.append("the label-gap rows of LN, LM and LF together are not the rows of LA")
    return {"pass": not bad, "failures": bad, "values": vals}


# ============================================================================================ a1 to a5
def ctl_a1(wrapper_sha16, check_text, hashes_text, run_shas):
    """a1: as the g4 of the model reader for the eight rollouts: the wrapper file is 28bc31f21ecb4ccb, the pod check record of card ADD102 names it with a pass, every run text names it"""
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


def label_share(J, rows):
    """share of the steps with a lead whose received gap equals the join's label gap within 1e-3 m (the join's label rows only)"""
    ci, j, pres, g, _ = rows
    n, off = np.asarray(J["n"], np.int64), np.asarray(J["off"], np.int64)
    lg, lab = label_rows(J)
    idx = np.flatnonzero(pres)
    if not len(idx):
        return None
    c, f = ci[idx], j[idx]
    ok = (c >= 0) & (c < len(n))
    ok[ok] &= (f[ok] >= 0) & (f[ok] < n[c[ok]])
    hit = 0
    for k in np.flatnonzero(ok):
        r = int(off[c[k]] + f[k])
        if lab[r] and abs(g[idx[k]] - lg[r]) <= TOL_LABEL_SHARE:
            hit += 1
    return hit / float(len(idx))


def a2_from_files(a, rec, J):
    """-> per-rollout dict for ctl_a2: the model reader's w2_one on each of the eight rollouts with its gap file (VGAPA_<ARM>.npz), the control run V0 for steps with no output"""
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
                r["share_of_lead_steps_with_label_gap"] = label_share(J, rows)
                per[key] = r
            except (SystemExit, OSError, ValueError, KeyError) as ex:
                per[key] = {"PASS": False, "failures": ["cannot read the rollout: %s: %s" % (type(ex).__name__, ex)]}
    return per


def ctl_a2(per_run):
    """a2: as the g5 of the model reader for the eight rollouts (gap received equals the gap file within 1e-3 m on steps with a lead and an output; steps with a lead and no output equal V0; meta names the gap file and
    the closing switch; the closing input is non-zero on a positive share of the steps with a lead)"""
    bad = []
    if sorted(per_run) != rollout_keys():
        bad.append("rollouts read: %s, expected %s" % (sorted(per_run), rollout_keys()))
    for k, v in sorted(per_run.items()):
        bad += ["%s: %s" % (k, x) for x in v.get("failures", [])]
        if v.get("PASS") is not True and not v.get("failures"):
            bad.append("%s: not pass" % k)
    shares = {k: {"share_of_rule_steps_with_a_lead": v.get("share_of_rule_steps_with_a_lead"), "share_of_lead_steps_with_nonzero_closing": v.get("share_of_lead_steps_with_nonzero_closing"),
                  "share_of_lead_steps_with_label_gap": v.get("share_of_lead_steps_with_label_gap"), "n_rows": v.get("n_rows"), "n_lead_rows": v.get("n_lead_rows")} for k, v in per_run.items()}
    return {"pass": not bad and bool(per_run), "failures": bad, "values": {"rollout_shares": shares}}


def ctl_a3(levels):
    """a3: the pooled levels of GEc, RLc and RR recomputed by the reader equal 0.0062 / 0.5154 / 0.0839, 0.0000 / 0.4908 / 0.0714 and 0.0954 / 0.4954 / 0.1339 (contact, falling behind, unsafe) within 1e-4"""
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


def ctl_a4(text_gap, text_read, mtime_gap, mtime_read, queue_start):
    """a4: the last non-empty line of each lead_pytest.out holds `passed` (a count above 0) and neither `failed` nor `error`; and the newer of the two files is older than the first start line of the queue
    (seconds since the epoch of the file modification times and of the first token of FALLATTR10K_START.txt)"""
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


def ctl_a5(grid_all_pass, strat_a, strat_b):
    """a5: as the g8 of the model reader: the engine's own controls of the grid read and of the strategic grid pass"""
    r = MD.ctl_g8(grid_all_pass, strat_a, strat_b)
    return {"pass": bool(r["PASS"]), "failures": [] if r["PASS"] else ["grid_all_pass %s, strategic stage A %s, stage B %s" % (grid_all_pass, strat_a, strat_b)], "grid_all_pass": grid_all_pass,
            "strategic_stage_A_pass": strat_a, "strategic_stage_B_pass": strat_b}


def run_shas_of(pull):
    return {"%s_s%d" % (x, m): MD.run_file_sha(pull, RUN[x], m) for x in ARMS for m in SEEDS}


def compute_controls(a, rec, J, m, wsha, join_sha):
    """the six controls from the files on disk and the engine measures `m` (levels, grid_all_pass, strategic_a, strategic_b)"""
    base = None
    bsha = None
    bp = getattr(a, "base_file", "") or (((rec or {}).get("base") or {}).get("path") if isinstance(rec, dict) else "")
    if bp and os.path.isfile(bp):
        base, bsha = load_npz(bp), sha_file(bp)
    files = {x: gapfile_path(a.gap_dir, x) for x in ARMS}
    c = {}
    c["a0"] = ctl_a0(rec, files, J, join_sha, os.path.basename(a.eval_join), base, bsha)
    c["a1"] = ctl_a1(wsha, read_text(a.wrapper_check), read_text(a.wrapper_hashes), run_shas_of(a.pull))
    c["a2"] = ctl_a2(a2_from_files(a, rec, J))
    c["a3"] = ctl_a3(m["levels"])
    c["a4"] = ctl_a4(read_text(a.tests_gap), read_text(a.tests_read), a.tests_mtime_gap, a.tests_mtime_read, a.queue_start)
    c["a5"] = ctl_a5(m["grid_all_pass"], m["strategic_a"], m["strategic_b"])
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
    """card section 3: per arm X and measure M against GEc LOWERS / RAISES / NOT-SEPARATED; the whole from LA; the range names of LN, LM, LF"""
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
    la_g = pr["LA-VS-GEc"]["F"]["interval"]
    la_r = pr["LA-VS-RLc"]["F"]["interval"]
    lowers = la_g[1] < 0.0
    if lowers and la_r[0] <= 0.0:
        out["WHOLE"] = PRE + "-GAP-CARRIES-THE-PRICE"
    elif lowers and la_r[0] > 0.0:
        out["WHOLE"] = PRE + "-GAP-CARRIES-PART"
    elif (not lowers) and la_r[0] > 0.0:
        out["WHOLE"] = PRE + "-PRICE-NOT-IN-THE-GAP"
    else:
        out["WHOLE"] = PRE + "-NOT-DECIDED"
    rng = [PRE + "-PRICE-IN-" + RANGE_WORD[x] for x in RANGE_ARMS if pr[x + "-VS-GEc"]["F"]["interval"][1] < 0.0]
    out["RANGE"] = rng if rng else [PRE + "-NO-RANGE-SEPARATED"]
    return out


def _sg(iv):
    return -1 if iv[1] < 0.0 else (1 if iv[0] > 0.0 else 0)


def derive_all(payload):
    """independent second derivation of every name from the signs of the intervals (written apart from names_of)"""
    pr = payload["pairs"]
    out = {}
    word = {-1: "%s-%s-LOWERS-%s", 1: "%s-%s-RAISES-%s", 0: "%s-%s-%s-NOT-SEPARATED"}
    sf = {}
    for x in ARMS:
        for m, label in MEASURES:
            s = _sg(pr[x + "-VS-GEc"][m]["interval"])
            if m == "F":
                sf[x] = s
            out["%s_%s" % (x, label)] = word[s] % (PRE, x, label)
    lo_rl = pr["LA-VS-RLc"]["F"]["interval"][0]
    above_rl = _sg(pr["LA-VS-RLc"]["F"]["interval"]) == 1
    assert above_rl == (lo_rl > 0.0)
    table = {(True, False): "-GAP-CARRIES-THE-PRICE", (True, True): "-GAP-CARRIES-PART", (False, True): "-PRICE-NOT-IN-THE-GAP", (False, False): "-NOT-DECIDED"}
    out["WHOLE"] = PRE + table[(sf["LA"] == -1, above_rl)]
    hit = [RANGE_WORD[x] for x in ("LN", "LM", "LF") if sf[x] == -1]
    out["RANGE"] = [PRE + "-PRICE-IN-" + w for w in hit] if hit else [PRE + "-NO-RANGE-SEPARATED"]
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
        for ref in ("GEc", "RLc", "RR"):
            pairs_def["%s-VS-%s" % (x, ref)] = (RUN[x], REFS[ref])
    spread = "R0Hb20"
    run_list = [spread] + sorted(set(runs.values()))
    G = R.read_grid(a, E, "30", keys, run_list, spread, pairs_def, {}, log)
    lv = {k: {c: float(np.mean(G["levels"][r][c])) for c in ("C", "U", "F")} for k, r in runs.items()}
    lv_seed = {k: {c: [float(v) for v in np.asarray(G["levels"][r][c]).reshape(-1)[:2]] for c in ("C", "U", "F")} for k, r in runs.items()}
    pairs = {nm: R.ucf(p["headway"]) for nm, p in G["pairs"].items()}
    fams = {nm: {f_: R.reading(R.cells_of(p["family"]), f_) for f_ in R.NAMED} for nm, p in G["pairs"].items()}
    strat_runs = set(RUN.values()) | set(REFS.values())
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
    counts = {x: {k: (rf.get(x) or {}).get(k) for k in ("n_label_rows", "n_source_changes")} for x in ARMS}
    payload = {"levels": m["levels"], "levels_per_seed": m["levels_per_seed"], "pairs": m["pairs"], "families": m["families"], "strategic": m["strategic"], "family_note": FAM_NOTE,
               "gapfile_counts_from_record": counts}
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
        for ref in ("GEc", "RLc", "RR"):
            nm = "%s-VS-%s" % (x, ref)
            L.append("%s: %s" % (nm, fmt_pair(nm, pr[nm])))
    for x in ARMS:
        for ref in ("GEc", "RLc"):
            nm = "%s-VS-%s" % (x, ref)
            L.append("families %s: %s strategic %s" % (nm, json.dumps(fm.get(nm) or {}, sort_keys=True), (st.get(nm) or {}).get("reading", "not read")))
    for x in ARMS:
        L.append("gap file %s: " % x + json.dumps((payload.get("gapfile_counts_from_record") or {}).get(x), sort_keys=True))
    sh = (payload.get("control_values") or {}).get("a2", {}).get("rollout_shares") or {}
    for k in sorted(sh):
        L.append("rollout %s: %s" % (k, json.dumps(sh[k], sort_keys=True)))
    return L


# ============================================================================================ synthetic world for the selftest and the tests
def make_world(base, fault=None):
    """a tiny world in which every control passes (fault None), or exactly the one control `fault` (a0 to a5) fails. -> (a, rec, J, m, wsha, join_sha)"""
    base = str(base)
    os.makedirs(base, exist_ok=True)
    gd, pull = os.path.join(base, "gapfiles"), os.path.join(base, "pull")
    os.makedirs(gd, exist_ok=True)
    os.makedirs(pull, exist_ok=True)
    n, off = np.array([10, 10, 10]), np.array([0, 10, 20])
    lab = np.array([5.0, 15.0, 25.0, 35.0, 45.0, 55.0, np.nan, 70.0, 30.0, 10.0] * 3)
    valid = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 1] * 3)
    J = {"clip": np.array(["c0", "c1", "c2"]), "off": off, "n": n, "lead_gap": lab, "lead_valid": valid}
    jp = os.path.join(base, JOIN_NAME)
    np.savez(jp, **J)
    jsha = sha_file(jp)
    ci = np.repeat(np.arange(3), 10)
    fr = np.tile(np.arange(10), 3)
    bgap = 40.0 + 0.37 * np.arange(30)
    bhas = fr != 3
    bz = {"clip_index": ci, "frame": fr, "gap": np.where(bhas, bgap, np.nan), "has_out": bhas, "clips": J["clip"], "seed": np.int64(-1)}
    bp = os.path.join(base, "VGAPLE.npz")
    np.savez(bp, **bz)
    g, lm = label_rows(J)
    rec = {"pass_a0": fault != "a0", "clauses": {"c1": {"pass": True}, "c2": {"pass": True}}, "files": {}, "base": {"path": bp, "sha256": sha_file(bp), "named_in_record_add106": True}, "eval_join_sha256": jsha}
    arm_gap, arm_has = {}, {}
    for x in ARMS:
        sel = lm & in_range(x, g)
        gap, has = np.where(sel, g, bgap), np.where(sel, True, bhas)
        arm_gap[x], arm_has[x] = np.where(has, gap, np.nan), has
        order = np.arange(30)
        if x == "LM":
            order = np.random.RandomState(4).permutation(30)
        z = {"clip_index": ci[order], "frame": fr[order], "gap": arm_gap[x][order], "has_out": has[order], "clips": J["clip"], "seed": np.int64(-1)}
        p = gapfile_path(gd, x)
        np.savez(p, **z)
        ch = sum(int(sel[r] != sel[r - 1]) for r in range(1, 30) if ci[r] == ci[r - 1])
        rec["files"][x] = {"path": p, "sha256": sha_file(p), "n_rows": 30, "n_label_rows": int(sel.sum()), "n_source_changes": ch}
    recp = os.path.join(gd, "FALLATTR_GAPFILES_RECORD.json")
    json.dump(rec, io.open(recp, "w", encoding="utf-8"))
    v0 = 20.0 + 0.1 * np.arange(30)
    clos = np.where(np.arange(30) % 2 == 0, 0.5, 0.0)

    def write_run(run, m, gap_from, sha, wrap_hash):
        d = os.path.join(pull, "I10%s_s%d" % (run, m), "ticks")
        os.makedirs(d, exist_ok=True)
        for nm, sl in (("rule_log_I10%s_s%d.npz", slice(0, 15)), ("rule_log_mid_I10%s_s%d.npz", slice(15, 30))):
            np.savez(os.path.join(d, nm % (run, m)), ci=ci[sl], j=fr[sl], present=np.ones(15, bool), g_src=gap_from[sl].copy(), c_src=clos[sl], vis_gap_file_sha256=np.array(sha or "none"),
                     vis_closing=np.array(1))
        io.open(os.path.join(d, "run_I10%s_s%d.txt" % (run, m)), "w", encoding="utf-8").write("run text\nwrapper sha256 %s\n" % wrap_hash)

    wrap = WRAPPER_SHA + "0" * 48
    for m in SEEDS:
        write_run(V0_RUN, m, v0, None, wrap)
        for x in ARMS:
            gs = np.where(arm_has[x], arm_gap[x], v0) + 1e-5
            if fault == "a2" and x == "LA" and m == 0:
                gs[2] += 0.5
            write_run(RUN[x], m, gs, rec["files"][x]["sha256"], wrap if not (fault == "a1" and x == "LF" and m == 1) else "f" * 64)
    chk = os.path.join(base, "CHECK_WRAPPER_V_DONE.txt")
    io.open(chk, "w", encoding="utf-8").write("a\n" + (CHECK_LAST if fault != "a1" else "ZZ CHECK-WRAPPER-V FAIL") + "\n\n")
    hs = os.path.join(base, "CHECK_WRAPPER_V.hashes")
    io.open(hs, "w", encoding="utf-8").write("sha %s\n" % WRAPPER_SHA)
    tg, tr = os.path.join(base, "gap_pytest.out"), os.path.join(base, "read_pytest.out")
    io.open(tg, "w", encoding="utf-8").write("....\n== 12 passed in 3.1s ==\n\n")
    io.open(tr, "w", encoding="utf-8").write("....\n== 30 passed in 9.0s ==\n")

    class A:
        pass
    a = A()
    a.pull, a.gap_dir, a.eval_join, a.base_file = pull, gd, jp, ""
    a.wrapper_check, a.wrapper_hashes, a.tests_gap, a.tests_read = chk, hs, tg, tr
    a.tests_mtime_gap, a.tests_mtime_read, a.queue_start = 1000.0, 1010.0, 1100.0
    if fault == "a4":
        a.tests_mtime_read = 1200.0
    m = {"levels": {k: dict(v) for k, v in EXPECTED_LEVELS.items()}, "grid_all_pass": True, "strategic_a": True, "strategic_b": fault != "a5"}
    if fault == "a3":
        m["levels"]["GEc"]["F"] += 0.001
    return a, rec, J, m, WRAPPER_SHA, jsha


def mk_payload(f_gec=None, f_rl=None, c=None, u=None):
    """a synthetic payload of pairs: f_gec[arm] and f_rl[arm] are the intervals of falling behind against GEc and RLc, c[arm] and u[arm] those of contact and unsafe against GEc; default 'holds 0'"""
    z = [-0.01, 0.01]
    pr = {}
    for x in ARMS:
        for ref in ("GEc", "RLc", "RR"):
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
    bind(os.environ.get("FALLATTR_MODEL_READER", ""))
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
    cases = [
        (mk_payload({"LA": [-0.05, -0.01]}, {"LA": [-0.02, 0.01]}), "-GAP-CARRIES-THE-PRICE", "-NO-RANGE-SEPARATED"),
        (mk_payload({"LA": [-0.05, -0.01]}, {"LA": [-0.05, -0.01]}), "-GAP-CARRIES-THE-PRICE", "-NO-RANGE-SEPARATED"),
        (mk_payload({"LA": [-0.05, -0.01], "LF": [-0.04, -0.001]}, {"LA": [0.001, 0.03]}), "-GAP-CARRIES-PART", "-PRICE-IN-FAR"),
        (mk_payload({"LA": [-0.02, 0.03], "LN": [-0.04, -0.001], "LM": [-0.04, -0.001]}, {"LA": [0.01, 0.05]}), "-PRICE-NOT-IN-THE-GAP", "-PRICE-IN-NEAR"),
        (mk_payload({"LA": [-0.02, 0.03]}, {"LA": [-0.01, 0.02]}), "-NOT-DECIDED", "-NO-RANGE-SEPARATED"),
    ]
    for pl, whole, rng in cases:
        nm = all_names(pl)
        ok &= nm["WHOLE"] == PRE + whole and PRE + rng in nm["RANGE"]
    nm = all_names(mk_payload({"LA": [-0.05, -0.01], "LN": [-0.04, -0.001], "LM": [-0.04, -0.001], "LF": [-0.04, -0.001]}, {"LA": [0.0, 0.01]}))
    ok &= nm["RANGE"] == [PRE + "-PRICE-IN-NEAR", PRE + "-PRICE-IN-MIDDLE", PRE + "-PRICE-IN-FAR"]
    nm = all_names(mk_payload(c={"LN": [-0.03, -0.001], "LM": [0.001, 0.03]}))
    ok &= nm["LN_CONTACT"] == PRE + "-LN-LOWERS-CONTACT" and nm["LM_CONTACT"] == PRE + "-LM-RAISES-CONTACT" and nm["LF_CONTACT"] == PRE + "-LF-CONTACT-NOT-SEPARATED"
    print("ZZ READ-FALLATTR10K SELFTEST %s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage", choices=("controls", "names"))
    ap.add_argument("--selftest", action="store_true")
    for nme in ("pull", "code", "grid-dir", "w", "tree", "wrapper-file", "wrapper-check", "wrapper-hashes", "controls", "eval-join", "gap-record", "gap-dir", "tests-gap", "tests-read",
                "tests-mtime-gap", "tests-mtime-read", "queue-start", "model-reader", "base-file", "fa"):
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
        fa = a.fa or WS + "/fallattr"
        a.gap_dir = a.gap_dir or fa + "/gapfiles"
        a.gap_record = a.gap_record or a.gap_dir + "/FALLATTR_GAPFILES_RECORD.json"
        a.eval_join = a.eval_join or WS + "/hierrep/pins/" + JOIN_NAME
        a.tests_gap = a.tests_gap or fa + "/kit_gap/lead_pytest.out"
        a.tests_read = a.tests_read or fa + "/kit_read/lead_pytest.out"
        res = run_controls(a)
        io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o)))
        for k in CONTROLS:
            print("%s %s" % (k, "PASS" if (res["controls"].get(k) or {}).get("pass") else "FAIL"))
        print("ZZ READ-FALLATTR10K controls all_pass %s" % res["all_pass"])
        return 0
    cj = read_json(a.controls) or {}
    if cj.get("card") != CARD or cj.get("all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != sorted(CONTROLS) \
            or any((v or {}).get("pass") is not True for v in cj.get("controls", {}).values()):
        print("REFUSED: the controls file is not all-pass with a0 to a5 present: no name is written")
        print("ZZ READ-FALLATTR10K names refused")
        return 3
    logp = a.log or (os.path.splitext(a.out)[0] + "_names.log")
    if os.path.exists(logp):
        print("REFUSED: %s exists" % logp)
        return 2
    names = all_names(cj["payload"])
    res = {"card": CARD, "tier": TIER, "controls_file": os.path.abspath(a.controls), "controls_sha256": sha_file(a.controls), "names": names, "values": cj["payload"]}
    io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1))
    lines = print_sections(cj["payload"], names) + ["ZZ READ-FALLATTR10K names done"]
    io.open(logp, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    for ln in lines:
        print(ln)
    return 0


try:
    bind(os.environ.get("FALLATTR_MODEL_READER", ""))
except ImportError as _ex:                      # the model reader is bound again in main() from --model-reader; a missing one is reported there
    print("NOTE: %s" % _ex, file=sys.stderr)

if __name__ == "__main__":
    sys.exit(main())
