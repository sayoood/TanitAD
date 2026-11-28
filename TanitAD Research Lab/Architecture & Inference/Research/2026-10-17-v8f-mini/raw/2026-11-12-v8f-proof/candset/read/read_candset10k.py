#!/usr/bin/env python
"""read_candset10k.py -- read of card PREREG_V8F_PROOF_ADD113 (CANDSET10K) and its note N1: the camera-fed candidate form GEc (reader-fed rule RR with the base gap file VGAPLE and the closing input)
at the settings CONV (no pre-roll, pinned readout), K0 (no pre-roll, readout K), K60 (pre-roll of 60 steps (12 s), readout K) and, as identity, K30 (pre-roll of 30 steps (6 s), banked).
Tier T1-kin/hb0.2 (kinematic rollout on the model's own actions).
It is an importing tool: the engine calls (read_grid, strategic_grid, ucf, reading, cells_of, family_of) and the rule-log routines (rows_of, load_logs, lead_multi, w2_one, gapfile_ref, meta_of,
run_file_sha, ctl_g8) are those of the model reader read_cropscale10k.py of card ADD106 (imported from --model-reader), as the reader of card ADD111 calls them. The existing readers are not edited.
Stage controls: runs d0 to d6 of card section 5 (with the presence clause of note N1 in d2), the grid read and the strategic read of every setting; writes CANDSET10K_CONTROLS.json (values under `payload`
  when all pass, under `withheld` otherwise); prints only `d<i> PASS|FAIL` lines and `ZZ READ-CANDSET10K controls all_pass <True|False>`. The engine's own output goes to <out>.engine.log.
Stage names: refuses unless d0 to d6 are present and pass; writes CANDSET10K_RESULT.json and a names log.
--selftest: synthetic world; last line `ZZ READ-CANDSET10K SELFTEST OK` or FAIL."""

import argparse
import contextlib
import datetime
import glob
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
WS = "/workspace/v8proof"


def locate_model(path=""):
    for p in (path, os.environ.get("CANDSET_MODEL_READER", ""), POD_MODEL, DEV_MODEL):
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


CARD, PRE = "PREREG_V8F_PROOF_ADD113", "CANDSET10K"
SEEDS = (0, 1)
ARMS = ("GEc", "RR", "RLc")
SETTINGS = ("CONV", "K0", "K60", "K30")
NEW_SETTINGS = ("CONV", "K0", "K60")
SPREAD = "R0Hb20"
# run names without the prefix I10; P = pre-roll steps; grid = the key of the engine's grid read
SPEC = {
    "CONV": {"P": 0, "readout": "pinned", "every": None, "grid": "full", "grid_file": "pinned",
             "runs": {"GEc": "RQ50Hb20gGEc", "RR": "RQ50Hb20", "RLc": "RLcHb20"}},
    "K0": {"P": 0, "readout": "K", "every": 5, "grid": "full", "grid_file": "pinned",
           "runs": {"GEc": "RQ50KF5Hb20gGEc", "RR": "RQ50KF5Hb20", "RLc": "RLcKF5Hb20"}},
    "K60": {"P": 60, "readout": "K", "every": 5, "grid": "60", "grid_file": "grid_p60.json",
            "runs": {"GEc": "RQ50KF5P60Hb20gGEc", "RR": "RQ50KF5P60Hb20", "RLc": "RLcKF5P60Hb20"}},
    "K30": {"P": 30, "readout": "K", "every": 5, "grid": "30", "grid_file": "grid_p30.json",
            "runs": {"GEc": "RQ50KF5P30Hb20gGEc", "RR": "RQ50KF5P30Hb20", "RLc": "RLcKF5P30Hb20"}},
}
PAIRS = {"GEc-VS-RR": ("GEc", "RR"), "GEc-VS-RLc": ("GEc", "RLc"), "RR-VS-RLc": ("RR", "RLc")}
STRAT_PAIRS = PAIRS
EXPECTED_K30 = {"GEc": {"C": 0.0062, "F": 0.5154, "U": 0.0839}, "RR": {"C": 0.0954, "F": 0.4954, "U": 0.1339}, "RLc": {"C": 0.0, "F": 0.4908, "U": 0.0714}}
LEVEL_TOL = 1e-4
CONTACT_FLOOR = 0.01
WRAPPER_SHA = "28bc31f21ecb4ccb"
CHECK_LAST = "ZZ CHECK-WRAPPER-V PASS"
CONTROLS = tuple("d%d" % i for i in range(7))
PINNED_GRID_NAME = "eval_grid_windows.json"
DONE_RE = re.compile(r"^ZZ CANDSET10K DONE \S+ all steps done$")
MEASURES = (("C", "CONTACT"), ("U", "UNSAFE"), ("F", "FALLING-BEHIND"))
TIER = ("T1-kin/hb0.2 (kinematic rollout on the model's own actions, executor of record v8f_t1_eval_hb.py with the heading bound 0.2 1/m); CONV: no pre-roll; K0: no pre-roll; "
        "K60: pre-roll of 60 steps (12 s); K30 (banked, identity only): pre-roll of 30 steps (6 s)")
FAM_NOTE = ("the rule changes the planned speed along the path through the distance to the lead; it does not move the path, so the lateral family is expected to be unmoved by it, "
            "and the tactical family is not read by the rule; the strategic family is read from the strategic grid of the engine at T1-kin")
# the first line of a run log: `key value` pairs, tolerant (a field a line does not carry is absent)
FIELD_RE = {"form": r"form (\S+)", "replan": r"replan steps (\S+)", "every": r"rule every (\S+)", "readout": r"readout (\S+)", "kappa": r"kappa (\S+)", "preroll": r"preroll (\S+)",
            "exec": r"exec (\S+)", "nav": r"nav (\S+)", "grid": r"grid (\S+)", "model_seed": r"model seed (\S+)", "reader_seed": r"reader seed (\S+)", "ckpt": r"ckpt sha256 ([0-9a-fA-F]{16,})"}
D2_COMPARED = ("form", "replan", "every", "readout", "kappa", "preroll", "exec", "nav", "grid", "model_seed", "reader_seed", "ckpt")
D2_REQUIRED = ("form", "replan", "every", "readout", "kappa", "model_seed", "reader_seed")
D2_DEFAULTS = (("preroll", "0"), ("grid", PINNED_GRID_NAME), ("nav", "deadreckon"))     # note N1: what the queue form that made a reference run without the field wrote


# ============================================================================================ small helpers
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_text(p):
    return io.open(p, encoding="utf-8", errors="replace").read() if p and os.path.isfile(p) else ""


def read_json(p):
    try:
        return json.load(io.open(p, encoding="utf-8"))
    except (OSError, ValueError, TypeError):
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


def last_line(text):
    ls = [x for x in (text or "").splitlines() if x.strip()]
    return ls[-1] if ls else ""


def first_line(text):
    for x in (text or "").splitlines():
        if x.strip():
            return x.strip()
    return ""


def run_name(S, arm):
    return SPEC[S]["runs"][arm]


def run_folder(pull, name, m):
    return os.path.join(pull, "I10%s_s%d" % (name, m))


def run_text_path(pull, name, m):
    return os.path.join(run_folder(pull, name, m), "ticks", "run_I10%s_s%d.txt" % (name, m))


def preroll_text(P):
    return "no pre-roll" if P == 0 else "pre-roll of %d steps (%d s)" % (P, int(round(P * 0.2)))


def new_keys():
    return ["%s_s%d" % (S, m) for S in NEW_SETTINGS for m in SEEDS]


# ============================================================================================ d2: the first line of a run log (card section 5 under note N1)
def parse_run_line(text):
    """-> {field: value string} for the fields of FIELD_RE that the first line carries (tolerant: a missing field is simply absent)"""
    ln = first_line(text)
    out = {}
    for k, rx in FIELD_RE.items():
        m = re.search(r"(?:^|\s)" + rx + r"(?=\s|$)", ln)
        if m:
            out[k] = m.group(1)
    return out


def _same(x, y):
    if str(x).lower() == str(y).lower():
        return True
    try:
        return abs(float(x) - float(y)) <= 1e-12
    except ValueError:
        return False


def d2_line_failures(new, ref):
    """the presence clause of note N1 on two parsed lines: every field both carry is equal; form, replan steps, rule every, readout, kappa, model seed and reader seed are present in both; where the reference
    carries no pre-roll the new line reads `preroll 0`, no grid the new line reads `grid eval_grid_windows.json`, no nav the new line reads `nav deadreckon`"""
    bad = []
    for k in D2_REQUIRED:
        if k not in new:
            bad.append("the new line carries no %s" % k)
        if k not in ref:
            bad.append("the reference line carries no %s" % k)
    for k in D2_COMPARED:
        if k in new and k in ref and not _same(new[k], ref[k]):
            bad.append("%s: new %s, reference %s" % (k, new[k], ref[k]))
    for k, dflt in D2_DEFAULTS:
        if k not in ref and not (k in new and _same(new[k], dflt)):
            bad.append("the reference line carries no %s, the new line has to read %s %s (it reads %s)" % (k, k, dflt, new.get(k)))
    return bad


# ============================================================================================ the controls
def gap_ref_once(a):
    return MD.gapfile_ref(a.gap_file)


def load_logs_setting(pull, run, m, S):
    """the rule logs of a run at setting S (lead's change before any test run or read). A run with the rule at every 5th step writes two logs (rule_log and rule_log_mid) and the
    model reader's load_logs reads both; a run with the rule at every replan (CONV, pinned readout) writes rule_log only, and load_logs would refuse it. Here: with --rule-every 5 both
    logs are needed (the model reader's routine, unchanged); with no --rule-every the first log is needed and a mid log is read only if it lies on disk."""
    if SPEC[S]["every"] is not None:
        return MD.load_logs(pull, run, m)
    d = os.path.join(pull, "I10%s_s%d" % (run, m), "ticks")
    first = os.path.join(d, "rule_log_I10%s_s%d.npz" % (run, m))
    mid = os.path.join(d, "rule_log_mid_I10%s_s%d.npz" % (run, m))
    if not os.path.isfile(first):
        raise SystemExit("REFUSED: no rule log %s" % first)
    zs = []
    for p in [first] + ([mid] if os.path.isfile(mid) else []):
        with np.load(p, allow_pickle=False) as z:
            zs.append({k: z[k] for k in z.files})
    return zs


def w2_new_runs(a, gref, gsha):
    """-> per-rollout dict: the model reader's w2_one on each of the six new rollouts against the gap file VGAPLE (as control b2 of card ADD111). The gap the reader itself logs, for the lead steps where the file
    has no output, is taken from the rule log of the reference run RR of the same setting and model seed (card ADD102's control run V0 exists at the pre-roll of 30 steps only)."""
    per = {}
    for S in NEW_SETTINGS:
        for m in SEEDS:
            key = "%s_s%d" % (S, m)
            try:
                ref_logs = load_logs_setting(a.pull, run_name(S, "RR"), m, S)
                v0 = MD.lead_multi(MD.rows_of(ref_logs))
                logs = load_logs_setting(a.pull, run_name(S, "GEc"), m, S)
                meta = MD.meta_of(logs[0])
                r = MD.w2_one(MD.rows_of(logs), gref, v0, meta_expect={"vis_closing": 1}, meta=meta)
                ms = str(meta.get("vis_gap_file_sha256") or "").lower()
                if not (len(ms) >= 16 and gsha and (gsha.startswith(ms) or ms.startswith(gsha))):
                    r["failures"] = list(r.get("failures", [])) + ["rule log meta names the gap file hash %s, the file on disk has %s" % (ms[:16] or None, (gsha or "")[:16])]
                    r["PASS"] = False
                per[key] = r
            except (SystemExit, OSError, ValueError, KeyError) as ex:
                per[key] = {"PASS": False, "failures": ["cannot read the rollout or its reference: %s: %s" % (type(ex).__name__, ex)]}
    return per


def ctl_d0(rec, gap_disk_sha, w2):
    """d0: the gap file on disk has the sha256 of the record of card ADD111 (base.sha256, 16 or more hex digits), and the rule log of each of the six new runs agrees with that file as control b2 of card ADD111
    checks it (gap received equals the gap of the file within 1e-3 m on steps with a lead and an output; steps with a lead and no output equal the reader's own logged gap; meta names the file hash and the closing
    switch 1; the closing input is non-zero on a positive share of the steps with a lead)"""
    bad = []
    base = (rec or {}).get("base") if isinstance((rec or {}).get("base"), dict) else {}
    rs = str(base.get("sha256") or "").lower()
    if len(rs) < 16 or not gap_disk_sha or not gap_disk_sha.startswith(rs):
        bad.append("the gap file on disk has sha256 %s, the record of card ADD111 names %s" % ((gap_disk_sha or "none")[:16], rs[:16] or None))
    if sorted(w2) != sorted(new_keys()):
        bad.append("rollouts read: %s, expected %s" % (sorted(w2), sorted(new_keys())))
    for k, v in sorted(w2.items()):
        bad += ["%s: %s" % (k, x) for x in v.get("failures", [])]
        if v.get("PASS") is not True and not v.get("failures"):
            bad.append("%s: not pass" % k)
    shares = {k: {"share_of_rule_steps_with_a_lead": v.get("share_of_rule_steps_with_a_lead"), "share_of_lead_steps_with_nonzero_closing": v.get("share_of_lead_steps_with_nonzero_closing"),
                  "n_rows": v.get("n_rows"), "n_lead_rows": v.get("n_lead_rows")} for k, v in w2.items()}
    return {"pass": not bad and bool(w2), "failures": bad, "values": {"rollout_shares": shares}}


def ctl_d1(wrapper_sha16, check_text, hashes_text, check_files_present, run_shas):
    """d1: the wrapper file has the hash 28bc31f21ecb4ccb; the banked check files are in place, the pod check record ends with `ZZ CHECK-WRAPPER-V PASS` and the hashes name the wrapper; the run text of each of the
    six new runs names a wrapper hash that begins with it"""
    bad = []
    if wrapper_sha16 != WRAPPER_SHA:
        bad.append("the wrapper file is %s, not %s" % (wrapper_sha16, WRAPPER_SHA))
    if not check_files_present:
        bad.append("a banked check file of the wrapper is missing")
    if last_line(check_text).strip() != CHECK_LAST:
        bad.append("the last line of the pod check is %r" % last_line(check_text).strip())
    if WRAPPER_SHA not in (hashes_text or ""):
        bad.append("the pod check hashes do not name %s" % WRAPPER_SHA)
    for k in new_keys():
        v = run_shas.get(k)
        if v is None:
            bad.append("%s: the run text names no wrapper hash" % k)
        elif not v.startswith(WRAPPER_SHA):
            bad.append("%s: the run text names the wrapper hash %s" % (k, v[:16]))
    return {"pass": not bad, "failures": bad, "wrapper_sha16": wrapper_sha16}


def ctl_d2(lines, w2):
    """d2: lines {key: (new text, reference text)} for the six new runs (S_s<k>); the presence clause of note N1 per run (d2_line_failures) and the rule log shows the rule acting on steps with a lead (a share above 0).
    The q-map file of each new run being the one of its reference run is read by the lead from the printed commands of the dry run and logged before the queue start: it is not coded here."""
    bad = []
    if sorted(lines) != sorted(new_keys()):
        bad.append("run lines read: %s, expected %s" % (sorted(lines), sorted(new_keys())))
    for k, (tn, tr) in sorted(lines.items()):
        if not first_line(tn):
            bad.append("%s: no run log of the new run" % k)
            continue
        if not first_line(tr):
            bad.append("%s: no run log of the reference run RR" % k)
            continue
        bad += ["%s: %s" % (k, x) for x in d2_line_failures(parse_run_line(tn), parse_run_line(tr))]
        sh = (w2.get(k) or {}).get("share_of_rule_steps_with_a_lead")
        if not (sh is not None and sh > 0):
            bad.append("%s: the rule log shows no rule step with a lead (share %s)" % (k, sh))
    return {"pass": not bad, "failures": bad}


def ctl_d3(levels_k30):
    """d3: the K30 levels of GEc, RR and RLc (contact, falling behind, unsafe) equal 0.0062/0.5154/0.0839, 0.0954/0.4954/0.1339 and 0.0000/0.4908/0.0714 within 1e-4; read by the same code path as the new settings"""
    bad = []
    for arm, want in EXPECTED_K30.items():
        for c, v in want.items():
            got = ((levels_k30 or {}).get(arm) or {}).get(c)
            if got is None or not np.isfinite(got) or abs(float(got) - v) > LEVEL_TOL + 1e-12:
                bad.append("K30 %s %s: %s, expected %.4f within %g" % (arm, c, got, v, LEVEL_TOL))
    return {"pass": not bad, "failures": bad}


def _pytest_ok(text, nm):
    bad = []
    hits = re.findall(r"(\d+) passed", text or "")
    if not (hits and int(hits[-1]) > 0):
        bad.append("%s: no line with a count of passed tests above 0" % nm)
    if re.search(r"\b\d+ (failed|errors?)\b", text or ""):
        bad.append("%s: the test output has a failed or error count" % nm)
    return bad


def ctl_d4(text_queue, text_read, mtime_queue, mtime_read, queue_start, controls_start):
    """d4: the output of the queue kit's test (written by the lead outside the kit folder) and that of this kit's test each carry `N passed` and no failed or error count; the first is not later than the first start
    epoch of the queue, the second not later than the start of the controls stage"""
    bad = _pytest_ok(text_queue, "queue kit") + _pytest_ok(text_read, "reader kit")
    mq, mr, qs, cs = to_epoch(mtime_queue), to_epoch(mtime_read), to_epoch(queue_start), to_epoch(controls_start)
    if mq is None or mr is None or qs is None or cs is None:
        bad.append("a time is missing (queue test %s, reader test %s, queue start %s, controls start %s)" % (mtime_queue, mtime_read, queue_start, controls_start))
    else:
        if not mq <= qs:
            bad.append("the queue kit's test output (%s) is later than the first start epoch of the queue (%s)" % (mq, qs))
        if not mr <= cs:
            bad.append("the reader kit's test output (%s) is later than the start of the controls stage (%s)" % (mr, cs))
    return {"pass": not bad, "failures": bad}


def ctl_d5(per_setting):
    """d5: for each of the four settings the engine's own controls of the grid read and of the strategic grid pass (as control g8 of the model reader)"""
    bad, det = [], {}
    if sorted(per_setting) != sorted(SETTINGS):
        bad.append("settings read: %s, expected %s" % (sorted(per_setting), sorted(SETTINGS)))
    for S, v in sorted(per_setting.items()):
        r = MD.ctl_g8(v.get("grid_all_pass"), v.get("strategic_a"), v.get("strategic_b"))
        det[S] = {"grid_all_pass": v.get("grid_all_pass"), "strategic_stage_A_pass": v.get("strategic_a"), "strategic_stage_B_pass": v.get("strategic_b")}
        if not r["PASS"]:
            bad.append("%s: grid_all_pass %s, strategic stage A %s, stage B %s%s" % (S, v.get("grid_all_pass"), v.get("strategic_a"), v.get("strategic_b"),
                                                                                    (" (" + v["strategic_reason"] + ")") if v.get("strategic_reason") else ""))
    return {"pass": not bad, "failures": bad, "per_setting": det}


def ctl_d6(done, n_dumps, n_clips, done_marker_text=None):
    """d6: every new run has its done marker and as many dump files as its grid has clips. done {key: bool}, n_dumps {key: int}, n_clips {setting: int or None}"""
    bad = []
    for k in new_keys():
        S = k.split("_")[0]
        if done.get(k) is not True:
            bad.append("%s: no T1_DONE" % k)
        want = n_clips.get(S)
        if want is None:
            bad.append("%s: the grid file of the setting cannot be read" % k)
        elif n_dumps.get(k) != want:
            bad.append("%s: %s dump files, the grid has %s clips" % (k, n_dumps.get(k), want))
    if done_marker_text is not None and not DONE_RE.match(last_line(done_marker_text).strip()):
        bad.append("the queue done marker reads %r" % last_line(done_marker_text)[:100])
    return {"pass": not bad, "failures": bad, "values": {"dump_files": n_dumps, "clips": n_clips}}


def grid_clip_count(p):
    g = read_json(p)
    try:
        return len(g["order"])
    except (TypeError, KeyError):
        return None


def grid_path(a, S):
    gf = SPEC[S]["grid_file"]
    return a.grid_pinned if gf == "pinned" else os.path.join(a.grid_dir, gf)


def compute_controls(a, rec, m, wsha):
    """the seven controls from the files on disk and the engine measures `m` ({"settings": {S: {levels, grid_all_pass, strategic_a, strategic_b}}})"""
    gsha = sha_file(a.gap_file) if os.path.isfile(a.gap_file) else None
    c = {}
    try:
        gref = gap_ref_once(a)
        w2 = w2_new_runs(a, gref, gsha)
    except (SystemExit, OSError, ValueError, KeyError) as ex:
        w2 = {k: {"PASS": False, "failures": ["cannot read the gap file: %s: %s" % (type(ex).__name__, ex)]} for k in new_keys()}
    c["d0"] = ctl_d0(rec, gsha, w2)
    shas = {"%s_s%d" % (S, mm): MD.run_file_sha(a.pull, run_name(S, "GEc"), mm) for S in NEW_SETTINGS for mm in SEEDS}
    present = bool(a.wrapper_check and os.path.isfile(a.wrapper_check) and a.wrapper_hashes and os.path.isfile(a.wrapper_hashes))
    c["d1"] = ctl_d1(wsha, read_text(a.wrapper_check), read_text(a.wrapper_hashes), present, shas)
    lines = {"%s_s%d" % (S, mm): (read_text(run_text_path(a.pull, run_name(S, "GEc"), mm)), read_text(run_text_path(a.pull, run_name(S, "RR"), mm))) for S in NEW_SETTINGS for mm in SEEDS}
    c["d2"] = ctl_d2(lines, w2)
    c["d3"] = ctl_d3(((m.get("settings") or {}).get("K30") or {}).get("levels"))
    c["d4"] = ctl_d4(read_text(a.tests_queue), read_text(a.tests_read), a.tests_mtime_queue, a.tests_mtime_read, a.queue_start, a.controls_start)
    c["d5"] = ctl_d5(m.get("settings") or {})
    done, n_dumps = {}, {}
    for S in NEW_SETTINGS:
        for mm in SEEDS:
            k = "%s_s%d" % (S, mm)
            nm = run_name(S, "GEc")
            done[k] = os.path.isfile(os.path.join(run_folder(a.pull, nm, mm), "T1_DONE"))
            n_dumps[k] = len(glob.glob(os.path.join(run_folder(a.pull, nm, mm), "dump_I10%s_s%d_m10_causal_heads" % (nm, mm), "ep*.npz")))
    n_clips = {S: grid_clip_count(grid_path(a, S)) for S in NEW_SETTINGS}
    c["d6"] = ctl_d6(done, n_dumps, n_clips, read_text(a.done_marker) if getattr(a, "done_marker", "") else None)
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
def _sg(iv):
    return -1 if iv[1] < 0.0 else (1 if iv[0] > 0.0 else 0)


def names_of(payload):
    """card section 4, written from the interval ends"""
    out = {}
    for S in NEW_SETTINGS:
        d = payload["settings"][S]
        pr = d["pairs"]
        p = "%s-%s-" % (PRE, S)
        if d["levels"]["RR"]["C"] < CONTACT_FLOOR:
            out[S + "_CONTACT"] = p + "NO-CONTACT-TO-CUT"
        else:
            iv = pr["GEc-VS-RR"]["C"]["interval"]
            out[S + "_CONTACT"] = p + ("CONTACT-CUT-HOLDS" if iv[1] < 0.0 else ("CONTACT-RAISED" if iv[0] > 0.0 else "CONTACT-NOT-SEPARATED"))
        iv = pr["GEc-VS-RR"]["U"]["interval"]
        out[S + "_UNSAFE"] = p + ("UNSAFE-CUT-HOLDS" if iv[1] < 0.0 else ("UNSAFE-RAISED" if iv[0] > 0.0 else "UNSAFE-NOT-SEPARATED"))
        out[S + "_PRICE"] = p + ("FALLS-BEHIND-MORE" if pr["GEc-VS-RLc"]["F"]["interval"][0] > 0.0 else "NO-SEPARATED-PRICE")
        out[S + "_VS_LABEL"] = p + ("CONTACT-ABOVE-LABEL-FORM" if pr["GEc-VS-RLc"]["C"]["interval"][0] > 0.0 else "CONTACT-AT-LABEL-FORM")
    reads = [out[S + "_CONTACT"].split("-", 2)[2] for S in NEW_SETTINGS]
    graded = [r for r in reads if r != "NO-CONTACT-TO-CUT"]
    if graded and all(r == "CONTACT-CUT-HOLDS" for r in graded):
        out["OVERALL"] = PRE + "-CUT-HOLDS-AT-EVERY-SETTING-WITH-CONTACT"
    elif any(r == "CONTACT-RAISED" for r in graded):
        out["OVERALL"] = PRE + "-CUT-RAISED-AT-A-SETTING"
    else:
        out["OVERALL"] = PRE + "-CUT-NOT-AT-EVERY-SETTING"
    return out


def derive_all(payload):
    """independent second derivation of every name from the signs of the intervals and the pooled contact of RR (written apart from names_of)"""
    out = {}
    verdicts = []
    word_c = {-1: "CONTACT-CUT-HOLDS", 1: "CONTACT-RAISED", 0: "CONTACT-NOT-SEPARATED"}
    word_u = {-1: "UNSAFE-CUT-HOLDS", 1: "UNSAFE-RAISED", 0: "UNSAFE-NOT-SEPARATED"}
    for S in NEW_SETTINGS:
        d = payload["settings"][S]
        pre = "CANDSET10K-" + S + "-"
        rr_c = d["levels"]["RR"]["C"]
        sc = _sg(d["pairs"]["GEc-VS-RR"]["C"]["interval"])
        verdicts.append("NONE" if rr_c < CONTACT_FLOOR else sc)
        out[S + "_CONTACT"] = pre + ("NO-CONTACT-TO-CUT" if rr_c < CONTACT_FLOOR else word_c[sc])
        out[S + "_UNSAFE"] = pre + word_u[_sg(d["pairs"]["GEc-VS-RR"]["U"]["interval"])]
        out[S + "_PRICE"] = pre + ("NO-SEPARATED-PRICE" if _sg(d["pairs"]["GEc-VS-RLc"]["F"]["interval"]) != 1 else "FALLS-BEHIND-MORE")
        out[S + "_VS_LABEL"] = pre + ("CONTACT-AT-LABEL-FORM" if _sg(d["pairs"]["GEc-VS-RLc"]["C"]["interval"]) != 1 else "CONTACT-ABOVE-LABEL-FORM")
    graded = [v for v in verdicts if v != "NONE"]
    if len(graded) > 0 and graded.count(-1) == len(graded):
        tail = "CUT-HOLDS-AT-EVERY-SETTING-WITH-CONTACT"
    elif 1 in graded:
        tail = "CUT-RAISED-AT-A-SETTING"
    else:
        tail = "CUT-NOT-AT-EVERY-SETTING"
    out["OVERALL"] = "CANDSET10K-" + tail
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
    return [names[k] for k in sorted(names)]


# ============================================================================================ the pod part (grid read and strategic read per setting)
def register_runs(S):
    spec = SPEC[S]
    for arm, r in spec["runs"].items():
        R.ALL_RUNS.setdefault(r, ("RLc" if arm == "RLc" else "RR", spec["readout"], spec["P"], "deadreckon", spec["grid"], "plan", spec["every"], 0.2))
    R.ALL_RUNS.setdefault(SPREAD, ("R0", "pinned", 0, "deadreckon", "full", "plan", None, 0.2))


def measure_setting(a, E, S, keys, ctx_cache, log):
    """grid read of the three runs of one setting (levels, pairs with intervals, families) and the strategic grid at T1-kin. Engine calls as in the model reader."""
    H = E[0]
    spec = SPEC[S]
    runs = spec["runs"]
    register_runs(S)
    pairs_def = {nm: (runs[x], runs[y]) for nm, (x, y) in PAIRS.items()}
    G = R.read_grid(a, E, spec["grid"], keys, [SPREAD] + sorted(set(runs.values())), SPREAD, pairs_def, {}, log)
    ok_levels = all(r in (G.get("levels") or {}) for r in runs.values())
    out = {"grid_all_pass": G["controls"].get("ALL_PASS") is True, "counts": G.get("counts") or {}, "levels": {}, "levels_per_seed": {}, "pairs": {}, "families": {}, "strategic": {"pairs": {}},
           "strategic_a": False, "strategic_b": False}
    if out["counts"]:
        out["instrument_bar_met"] = bool(out["counts"].get("n_lead_windows", 0) >= H.MIN_WINDOWS and out["counts"].get("n_episodes_with_lead", 0) >= H.MIN_EPISODES)
    if ok_levels:
        out["levels"] = {arm: {c: float(np.mean(G["levels"][r][c])) for c in ("C", "U", "F")} for arm, r in runs.items()}
        out["levels_per_seed"] = {arm: {c: [float(v) for v in np.asarray(G["levels"][r][c]).reshape(-1)[:2]] for c in ("C", "U", "F")} for arm, r in runs.items()}
    if G.get("pairs"):
        out["pairs"] = {nm: R.ucf(p["headway"]) for nm, p in G["pairs"].items()}
        out["families"] = {nm: {f_: R.reading(R.cells_of(p["family"]), f_) for f_ in R.NAMED} for nm, p in G["pairs"].items()}
    p_of = {"R0": 0}
    p_of.update({r: spec["P"] for r in runs.values()})
    try:
        SR, cells_s, eid_s, SS = R.strategic_grid(a, keys, sorted(p_of), p_of, log, ctx_cache)
    except SystemExit as ex:
        out["strategic_reason"] = "SystemExit: %s" % str(ex)[:200]
        return out
    except Exception as ex:                                                                   # noqa: BLE001
        out["strategic_reason"] = "%s: %s" % (type(ex).__name__, str(ex)[:200])
        return out
    sa, sb = bool(SR.get("stage_A_pass")), bool(SR.get("stage_B_pass"))
    out["strategic_a"], out["strategic_b"] = sa, sb
    out["strategic"]["controls"] = dict({k: v for k, v in SR["controls"].items() if k != "s3"}, s3=SR["controls"]["s3"].get("pass"))
    if sa and sb:
        for nm, (x, y) in STRAT_PAIRS.items():
            pr = SS.pair_read(cells_s, runs[x], runs[y], eid_s, ticks=(2,))
            pos = pr["tick2"]["pos"]
            out["strategic"]["pairs"][nm] = {"A": runs[x], "B": runs[y], "pos_30_50_80m_tick_t0_plus_20": [
                {"arc_m": arc, "mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "verdict": b["verdict"], "n_per_seed": b.get("n_per_seed")} for arc, b in zip(SS.ARCS, pos)],
                "reading": R.family_of([c["verdict"] for c in pos])}
    else:
        out["strategic_reason"] = "the engine's strategic controls did not pass"
    return out


def measure(a, E, log):
    H = E[0]
    J = H.load_join()
    names_ = [str(c) for c in J["clip"]]
    ctx_cache = {}
    res = {}
    for S in SETTINGS:                                                                         # the same code path for the new settings and for K30
        g = R.jl(grid_path(a, S))
        if g is None:
            res[S] = {"grid_all_pass": False, "levels": {}, "pairs": {}, "strategic_a": False, "strategic_b": False, "strategic_reason": "grid file %s cannot be read" % grid_path(a, S)}
            continue
        keys = R.grid_keys(g, names_)
        res[S] = measure_setting(a, E, S, keys, ctx_cache, log)
    return {"settings": res}


def run_controls(a):
    elog = a.out + ".engine.log"
    if os.path.exists(elog):
        raise SystemExit("REFUSED: %s exists" % elog)
    with io.open(elog, "w", encoding="utf-8", newline="\n") as fh, contextlib.redirect_stdout(fh):
        E = R.engine(a)
        m = measure(a, E, lambda s: print(s, flush=True))
    rec = read_json(a.gap_record)
    wsha = sha_file(a.wrapper_file)[:16] if a.wrapper_file and os.path.isfile(a.wrapper_file) else None
    controls = compute_controls(a, rec, m, wsha)
    payload = {"settings": m["settings"], "family_note": FAM_NOTE}
    inputs = {"model_reader": MODEL_PATH and sha_file(MODEL_PATH), "this_reader": sha_file(os.path.abspath(__file__)), "wrapper_file_sha16": wsha,
              "gap_file_sha256": sha_file(a.gap_file) if os.path.isfile(a.gap_file) else None, "gap_record_sha256": sha_file(a.gap_record) if os.path.isfile(a.gap_record) else None}
    return assemble_controls(controls, payload, inputs)


def fmt_pair(p):
    return "; ".join("%s %s" % (label, json.dumps(p[m], sort_keys=True, default=str)) for m, label in MEASURES)


def print_sections(payload, names):
    L = ["TIER " + TIER]
    L += flat_names(names)
    for S in SETTINGS:
        d = payload["settings"][S]
        cn = d.get("counts") or {}
        L.append("setting %s (%s; grid %s): windows %s, windows with a lead %s, episodes with a lead %s, instrument bar of 200 windows and 20 episodes met %s%s" % (
            S, preroll_text(SPEC[S]["P"]), SPEC[S]["grid"], cn.get("n_windows"), cn.get("n_lead_windows"), cn.get("n_episodes_with_lead"), d.get("instrument_bar_met"),
            "; read again as identity" if S not in NEW_SETTINGS else ""))
        for arm in ARMS:
            lv = d["levels"][arm]
            L.append("level %s %s pooled C %.4f U %.4f F %.4f (windows with a lead %s); per model seed %s" % (S, arm, lv["C"], lv["U"], lv["F"], cn.get("n_lead_windows"),
                                                                                                          json.dumps(d["levels_per_seed"].get(arm), sort_keys=True)))
        for nm in PAIRS:
            L.append("%s %s: %s" % (S, nm, fmt_pair(d["pairs"][nm])))
        for nm in PAIRS:
            st = (d.get("strategic") or {}).get("pairs", {}).get(nm)
            L.append("families %s %s: %s strategic %s" % (S, nm, json.dumps((d.get("families") or {}).get(nm) or {}, sort_keys=True),
                                                           st["reading"] if st else "not read (%s)" % d.get("strategic_reason", "no reason recorded")))
    sh = (payload.get("control_values") or {}).get("d0", {}).get("rollout_shares") or {}
    for k in sorted(sh):
        L.append("rollout %s: %s" % (k, json.dumps(sh[k], sort_keys=True)))
    return L


# ============================================================================================ synthetic world for the selftest and the tests
def _write_run(pull, name, m, first, gaps, sha, clos, ci, fr, done=True, ndump=3, meta=True):
    d = os.path.join(pull, "I10%s_s%d" % (name, m))
    os.makedirs(os.path.join(d, "ticks"), exist_ok=True)
    h = len(ci) // 2
    for nm, sl in (("rule_log_I10%s_s%d.npz", slice(0, h)), ("rule_log_mid_I10%s_s%d.npz", slice(h, len(ci)))):
        z = dict(ci=ci[sl], j=fr[sl], present=np.ones(len(ci[sl]), bool), g_src=gaps[sl].copy(), c_src=clos[sl])
        if meta:
            z.update(vis_gap_file_sha256=np.array(sha or "none"), vis_closing=np.array(1))
        np.savez(os.path.join(d, "ticks", nm % (name, m)), **z)
    io.open(os.path.join(d, "ticks", "run_I10%s_s%d.txt" % (name, m)), "w", encoding="utf-8", newline="\n").write(first + "\nrc 0 finished x\n")
    if done:
        io.open(os.path.join(d, "T1_DONE"), "w").write("")
    dd = os.path.join(d, "dump_I10%s_s%d_m10_causal_heads" % (name, m))
    os.makedirs(dd, exist_ok=True)
    for i in range(ndump):
        np.savez(os.path.join(dd, "ep%03d.npz" % i), x=np.zeros(1))


def run_line(arm_name, S, m, wrap, ref):
    sp = SPEC[S]
    every = sp["every"] if sp["every"] else 10
    readout = "pinned" if sp["readout"] == "pinned" else "causal_g"
    ck = ("%02x" % (m + 1)) * 32
    ln = "arm I10%s_s%d form RR replan steps 10 rule every %s readout %s kappa 0.2" % (arm_name, m, every, readout)
    if not (ref and S in ("CONV", "K0")):                                                      # the banked reference lines of CONV and K0 carry no pre-roll, exec, nav or grid field
        ln += " preroll %d exec plan nav deadreckon grid %s" % (sp["P"], {"CONV": PINNED_GRID_NAME, "K0": PINNED_GRID_NAME, "K60": "grid_p60.json", "K30": "grid_p30.json"}[S])
    ln += " model seed %d reader seed %d ckpt sha256 %s wrapper sha256 %s started 2026-11-22T10:57:07Z" % (m, m, ck, wrap)
    return ln


def make_world(base, fault=None):
    """a tiny world (3 clips of 12 frames) in which every control passes (fault None), or exactly the one control `fault` (d0 to d6) fails. -> (a, rec, m, wsha)"""
    base = str(base)
    os.makedirs(base, exist_ok=True)
    pull, gd = os.path.join(base, "pull"), os.path.join(base, "grids")
    for d_ in (pull, gd):
        os.makedirs(d_, exist_ok=True)
    nk = 12
    ci, fr = np.repeat(np.arange(3), nk), np.tile(np.arange(nk), 3)
    n = len(ci)
    has = np.ones(n, bool)
    has[[2, 14, 30]] = False
    gap = 15.0 + 0.9 * fr + 0.3 * ci
    gp = os.path.join(base, "VGAPLE.npz")
    np.savez(gp, clip_index=ci, frame=fr, gap=np.where(has, gap, np.nan), has_out=has, clips=np.array(["c0", "c1", "c2"]), seed=np.int64(-1))
    gsha = sha_file(gp)
    rec = {"base": {"sha256": gsha if fault != "d0_sha" else "0" * 64, "path": gp}}
    reader_gap = 20.0 + 0.2 * np.arange(n)
    wrap = WRAPPER_SHA + "0" * 48
    clos = np.where(np.arange(n) % 2 == 0, 0.5, 0.0)
    if fault == "d0_closing":
        clos = np.zeros(n)
    for S in SETTINGS:
        for m in SEEDS:
            _write_run(pull, run_name(S, "RR"), m, run_line(run_name(S, "RR"), S, m, "f" * 64, True), reader_gap + 1e-5, None, clos, ci, fr, meta=False)
            if S == "K30":
                continue
            g = np.where(has, gap, reader_gap) + 1e-5
            if fault == "d0" and S == "K60" and m == 1:
                g[3] += 0.5
            first = run_line(run_name(S, "GEc"), S, m, wrap if not (fault == "d1_run" and S == "CONV" and m == 0) else "e" * 64, False)
            if fault == "d2" and S == "K0" and m == 0:
                first = first.replace("kappa 0.2", "kappa 0.3")
            if fault == "d2_default" and S == "CONV" and m == 1:
                first = first.replace("preroll 0", "preroll 30")
            _write_run(pull, run_name(S, "GEc"), m, first, g, gsha, clos, ci, fr, done=not (fault == "d6_done" and S == "K60" and m == 0),
                       ndump=3 if not (fault == "d6" and S == "CONV" and m == 0) else 2)
    grid = {"order": ["c0", "c1", "c2"], "windows": {"c0": [0], "c1": [0], "c2": [0]}}
    for nm in (PINNED_GRID_NAME, "grid_p60.json"):
        json.dump(grid, io.open(os.path.join(gd, nm), "w", encoding="utf-8"))
    chk = os.path.join(base, "CHECK_WRAPPER_V_DONE.txt")
    io.open(chk, "w", encoding="utf-8").write("a\n" + (CHECK_LAST if fault != "d1" else "ZZ CHECK-WRAPPER-V FAIL") + "\n\n")
    hs = os.path.join(base, "CHECK_WRAPPER_V.hashes")
    io.open(hs, "w", encoding="utf-8").write("sha %s\n" % WRAPPER_SHA)
    tq, tr = os.path.join(base, "q_pytest.out"), os.path.join(base, "r_pytest.out")
    io.open(tq, "w", encoding="utf-8").write("....\n12 passed in 3.1s\n")
    io.open(tr, "w", encoding="utf-8").write("....\n== 30 passed, 2 warnings in 9.0s ==\n")

    class A:
        pass
    a = A()
    a.pull, a.grid_dir, a.grid_pinned, a.gap_file, a.gap_record = pull, gd, os.path.join(gd, PINNED_GRID_NAME), gp, os.path.join(base, "rec.json")
    a.wrapper_check, a.wrapper_hashes, a.tests_queue, a.tests_read = chk, hs, tq, tr
    a.tests_mtime_queue, a.tests_mtime_read, a.queue_start, a.controls_start, a.done_marker = 1000.0, 1010.0, 1100.0, 1200.0, ""
    if fault == "d4":
        a.tests_mtime_read = 1300.0
    lv = {k: dict(v) for k, v in EXPECTED_K30.items()}
    if fault == "d3":
        lv["GEc"]["U"] += 0.001
    st = {S: {"levels": lv if S == "K30" else {k: dict(v) for k, v in EXPECTED_K30.items()}, "grid_all_pass": True, "strategic_a": True, "strategic_b": True} for S in SETTINGS}
    if fault == "d5":
        st["K60"]["strategic_b"] = False
    return a, rec, {"settings": st}, WRAPPER_SHA


def mk_payload(rr_c=0.1, ivs=None):
    """a synthetic payload for the names: rr_c = {setting: pooled contact of RR} (a number for all), ivs = {(setting, pair, measure): interval}; default 'holds 0'"""
    z = [-0.01, 0.01]
    st = {}
    for S in SETTINGS:
        pr = {nm: {m: {"interval": list(z)} for m in "CUF"} for nm in PAIRS}
        for (s2, nm, m), iv in (ivs or {}).items():
            if s2 == S:
                pr[nm][m]["interval"] = list(iv)
        c = rr_c.get(S, 0.1) if isinstance(rr_c, dict) else rr_c
        st[S] = {"levels": {"RR": {"C": c, "U": 0.1, "F": 0.5}, "GEc": {"C": 0.0, "U": 0.1, "F": 0.5}, "RLc": {"C": 0.0, "U": 0.1, "F": 0.5}}, "pairs": pr}
    return {"settings": st}


def selftest():
    ok = True
    bind(os.environ.get("CANDSET_MODEL_READER", ""))
    with tempfile.TemporaryDirectory() as td:
        for fault in (None,) + CONTROLS:
            a, rec, m, wsha = make_world(os.path.join(td, "w_%s" % fault), fault)
            c = compute_controls(a, rec, m, wsha)
            flags = {k: v["pass"] for k, v in c.items()}
            want = {k: (k != fault) for k in CONTROLS}
            if flags != want:
                print("SELFTEST fault %s: flags %s" % (fault, flags))
                ok = False
    ok &= MD.WRAPPER_SHA == WRAPPER_SHA and MD.CHECK_LAST == CHECK_LAST
    low = [-0.05, -0.01]
    nm = all_names(mk_payload(0.1, {(S, "GEc-VS-RR", "C"): low for S in NEW_SETTINGS}))
    ok &= nm["OVERALL"] == PRE + "-CUT-HOLDS-AT-EVERY-SETTING-WITH-CONTACT"
    nm = all_names(mk_payload(0.005))
    ok &= nm["OVERALL"] == PRE + "-CUT-NOT-AT-EVERY-SETTING" and nm["CONV_CONTACT"] == PRE + "-CONV-NO-CONTACT-TO-CUT"
    print("ZZ READ-CANDSET10K SELFTEST %s" % ("OK" if ok else "FAIL"))
    return 0 if ok else 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage", choices=("controls", "names"))
    ap.add_argument("--selftest", action="store_true")
    for nme in ("pull", "code", "grid-dir", "grid-pinned", "w", "tree", "wrapper-file", "wrapper-check", "wrapper-hashes", "controls", "gap-file", "gap-record", "tests-queue", "tests-read",
                "tests-mtime-queue", "tests-mtime-read", "queue-start", "controls-start", "done-marker", "model-reader"):
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
        a.gap_file = a.gap_file or WS + "/cropscale/gapfiles/VGAPLE.npz"
        a.gap_record = a.gap_record or WS + "/gaperr/gapfiles/GAPERR_GAPFILES_RECORD.json"
        a.grid_pinned = a.grid_pinned or WS + "/hierrep/pins/" + PINNED_GRID_NAME
        a.tests_queue = a.tests_queue or WS + "/candset/lead_tests/lead_pytest_queue.out"
        a.tests_read = a.tests_read or WS + "/candset/lead_tests/lead_pytest_read.out"
        res = run_controls(a)
        io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o)))
        for k in CONTROLS:
            print("%s %s" % (k, "PASS" if (res["controls"].get(k) or {}).get("pass") else "FAIL"))
        print("ZZ READ-CANDSET10K controls all_pass %s" % res["all_pass"])
        return 0
    cj = read_json(a.controls) or {}
    if cj.get("card") != CARD or cj.get("all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != sorted(CONTROLS) \
            or any((v or {}).get("pass") is not True for v in cj.get("controls", {}).values()):
        print("REFUSED: the controls file is not all-pass with d0 to d6 present: no name is written")
        print("ZZ READ-CANDSET10K names refused")
        return 3
    logp = a.log or (os.path.splitext(a.out)[0] + "_names.log")
    if os.path.exists(logp):
        print("REFUSED: %s exists" % logp)
        return 2
    names = all_names(cj["payload"])
    res = {"card": CARD, "tier": TIER, "controls_file": os.path.abspath(a.controls), "controls_sha256": sha_file(a.controls), "names": names, "values": cj["payload"]}
    io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1))
    lines = print_sections(cj["payload"], names) + ["ZZ READ-CANDSET10K names done"]
    io.open(logp, "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    for ln in lines:
        print(ln)
    return 0


try:
    bind(os.environ.get("CANDSET_MODEL_READER", ""))
except ImportError as _ex:                      # bound again in main() from --model-reader; a missing one is reported there
    print("NOTE: %s" % _ex, file=sys.stderr)

if __name__ == "__main__":
    sys.exit(main())
