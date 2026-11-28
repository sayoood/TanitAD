#!/usr/bin/env python
"""read_visens10k.py -- read of card PREREG_V8F_PROOF_ADD105 (VISENS10K): the reader-fed lead rule (form RR) of the wrapper variant of card ADD102 (28bc31f21ecb4ccb, unchanged) with the slope closing input, fed the
MEAN of the five dumped gaps of the head on five frames (fit seeds 0 to 4; arm E5c), on the model's own rollout (T1-kin/hb0.2, pre-roll of 30 steps (6 s)), two model seeds.
Derived from visseed/read_visseed10k.py (card ADD104): functions kept or adapted; the engine of the model reader EC, R, HS, H is imported, not edited.
Stage controls: computes the grid read and the STRATEGIC family of the arm run and the banked RR, RLc, VD5c, VD5cx (and F2, F3, F4 only with --add104-controls on a file that reads x0 to x6 all PASS) inside this stage,
runs e0 to e3 and e5 here and reads e4 (--tests-out); writes CONTROLS_FILE. With every control PASS the values are written under `payload`, else under `withheld` and no name exists.
Stage names: reads CONTROLS_FILE, refuses it unless all six controls are present and pass, writes the names of card section 3 (four and the card name) and binds the result to the controls file by sha256.
The T0 diagnostic of the gap-file record (mean absolute gap error of the mean gap and of each single fit) is carried in the payload and printed only at this stage.
Usage: python read_visens10k.py --stage controls --pull PULL --code CODE --grid-dir DIR --w W --tree TREE --out CONTROLS.json --wrapper-file F --wrapper-check FILE --wrapper-hashes FILE --tests-out FILE
          --gap-record REC.json --gap-dir DIR --eval-join J.npz [--old-dump-root D] [--new-dump-root D] [--rec102 J] [--rec104 J] [--add104-controls VISSEED10K_CONTROLS.json] [--n-boot 2000]
       python read_visens10k.py --stage names --controls CONTROLS.json --out RESULT.json     |   --selftest
Last line: ZZ READ-VISENS10K <stage> rc <rc> controls_all_pass <bool>"""
import argparse
import io
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.environ.get("SR_ESTREAD_DIR"), os.path.join(_HERE, "..", "estclose_read"), _HERE):
    if _p and os.path.isfile(os.path.join(_p, "read_estclose10k_n2.py")):
        sys.path.insert(0, os.path.abspath(_p))
        break
sys.path.insert(0, _HERE)
try:
    import read_estclose10k_n2 as EC        # noqa: E402
    HS, R, np = EC.HS, EC.R, EC.np
except ImportError:                          # the unit tests of the name rules and controls need numpy only
    EC = HS = R = None
    import numpy as np                       # noqa: E402,F401
import make_ens_gapfile as MEG              # noqa: E402

CARD, PRE = "PREREG_V8F_PROOF_ADD105", "VISENS10K"
CARD104 = "PREREG_V8F_PROOF_ADD104"
SEEDS = (0, 1)                                                                 # model seeds
ARM = "E5c"
RUN = {ARM: "RQ50KF5P30Hb20eE5c"}
FITS = (2, 3, 4)                                                               # single-fit arms of card ADD104, read only when that card's controls all pass
RUN_F = {"F%d" % s: "RQ50KF5P30Hb20fF%d" % s for s in FITS}
B102 = {"VD5c": "RQ50KF5P30Hb20vVD5c"}                                          # banked arm of card ADD102 (fit seed k at model seed k)
B103 = {"VD5cx": "RQ50KF5P30Hb20wVD5cx"}                                        # banked arm of card ADD103 (fit seed 1 - k at model seed k)
V0_RUN = "RQ50KF5P30Hb20vV0"                                                   # banked control run of card ADD102: the reader's own gap as RR feeds it, logged at every lead step
BANKED = {"RR": "RQ50KF5P30Hb20", "RLc": "RLcKF5P30Hb20"}
C_EXPECTED = {"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0169}                         # control e3 (card section 4)
H_CARD = 0.0477
TOL_W2 = 1e-3
WRAPPER_SHA = "28bc31f21ecb4ccb"
CHECK_LAST = "ZZ CHECK-WRAPPER-V PASS"
TIER = "T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s)"
FAM_REASON = ("the rule changes the planned speed along the path through the distance to the lead; it does not move the path, so the lateral family is expected to be unmoved by it, "
              "and the tactical family is not read by the rule; the strategic family is read from the strategic grid of the engine")
CONTROLS = ("e0", "e1", "e2", "e3", "e4", "e5")
REF_RUNS = {"RR": BANKED["RR"], "RLc": BANKED["RLc"], "VD5c": B102["VD5c"]}    # the references of the strategic family
ADD104_CONTROLS = ("x0", "x1", "x2", "x3", "x4", "x5", "x6")


# ============================================================================================ names
def names_of(arm, d_rr_c, d_rlc_c, d_vd5c_c, d_rr_f, c_rr, c_rlc):
    """the four names of the arm (card section 3). d_*: interval [lo, hi] of C(X)-C(RR), C(X)-C(RLc), C(X)-C(VD5c), F(X)-F(RR); c_rr, c_rlc: pooled point values of C of the references."""
    p = PRE + "-" + arm + "-"
    H_ = 0.5 * (c_rr - c_rlc)
    n1 = p + ("CUTS-CONTACT" if d_rr_c[1] < 0.0 else ("MORE-CONTACT" if d_rr_c[0] > 0.0 else "CONTACT-NOT-SEPARATED-FROM-RR"))
    n2 = p + ("WITHIN-HALF-THE-GAP" if d_rlc_c[1] <= H_ else ("HALF-THE-GAP-NOT-REACHED" if d_rlc_c[0] > H_ else "HALF-THE-GAP-NOT-DECIDED"))
    n3 = p + ("LESS-CONTACT-THAN-VD5c" if d_vd5c_c[1] < 0.0 else ("MORE-CONTACT-THAN-VD5c" if d_vd5c_c[0] > 0.0 else "CONTACT-NOT-SEPARATED-FROM-VD5c"))
    n4 = p + ("FALLS-BEHIND-MORE" if d_rr_f[0] > 0.0 else ("FALLS-BEHIND-LESS" if d_rr_f[1] < 0.0 else "FALLING-BEHIND-NOT-SEPARATED"))
    return {arm + "_CONTACT": n1, arm + "_HALF": n2, arm + "_VD5C": n3, arm + "_PRICE": n4}


def card_name(names):
    """card name, the first that applies (card section 3): DOES-NOT-CUT when the arm is not CUTS-CONTACT; WITHIN-HALF-THE-GAP when it is WITHIN-HALF-THE-GAP; HALF-THE-GAP-NOT-REACHED when it is
    HALF-THE-GAP-NOT-REACHED; else HALF-THE-GAP-NOT-DECIDED."""
    if not names[ARM + "_CONTACT"].endswith("-" + ARM + "-CUTS-CONTACT"):
        return PRE + "-DOES-NOT-CUT"
    if names[ARM + "_HALF"].endswith("-" + ARM + "-WITHIN-HALF-THE-GAP"):
        return PRE + "-WITHIN-HALF-THE-GAP"
    if names[ARM + "_HALF"].endswith("-" + ARM + "-HALF-THE-GAP-NOT-REACHED"):
        return PRE + "-HALF-THE-GAP-NOT-REACHED"
    return PRE + "-HALF-THE-GAP-NOT-DECIDED"


def card_check(payload, got):
    """independent re-derivation of the card name from the intervals of the payload (not from the other names); {"PASS", "expected"}"""
    pr = payload["pairs"]
    h = 0.5 * (payload["levels"]["RR"]["C"] - payload["levels"]["RLc"]["C"])
    cuts = pr[ARM + "-VS-RR"]["C"]["interval"][1] < 0.0
    lo, hi = pr[ARM + "-VS-RLc"]["C"]["interval"]
    if not cuts:
        exp = "-DOES-NOT-CUT"
    elif hi <= h:
        exp = "-WITHIN-HALF-THE-GAP"
    elif lo > h:
        exp = "-HALF-THE-GAP-NOT-REACHED"
    else:
        exp = "-HALF-THE-GAP-NOT-DECIDED"
    return {"PASS": got.get("CARD") == PRE + exp, "expected": {"CARD": PRE + exp}}


def h_sensitivity(payload):
    """notes, no name: the half-the-gap name again with H = 0.0477 as the card prints it; `differs` is true when that name is not the one made with the point values"""
    pr = payload["pairs"]
    h = 0.5 * (payload["levels"]["RR"]["C"] - payload["levels"]["RLc"]["C"])
    lo, hi = pr[ARM + "-VS-RLc"]["C"]["interval"]
    f = lambda H: "WITHIN" if hi <= H else ("NOT-REACHED" if lo > H else "NOT-DECIDED")      # noqa: E731
    return {"H_point_values": h, "H_card": H_CARD, ARM: {"with_point_values": f(h), "with_H_card": f(H_CARD)}, "differs": bool(f(h) != f(H_CARD))}


def all_names(payload):
    pr = payload["pairs"]
    out = names_of(ARM, pr[ARM + "-VS-RR"]["C"]["interval"], pr[ARM + "-VS-RLc"]["C"]["interval"], pr[ARM + "-VS-VD5c"]["C"]["interval"], pr[ARM + "-VS-RR"]["F"]["interval"],
                   payload["levels"]["RR"]["C"], payload["levels"]["RLc"]["C"])
    out["CARD"] = card_name(out)
    chk = card_check(payload, out)
    if not chk["PASS"]:
        raise SystemExit("REFUSED: the card name %s is not the one the intervals give (%s)" % (out["CARD"], chk["expected"]))
    return out


def single_fit_table(levels_seed, include_f):
    """printed with no name (card section 3): the contact rate of the single-fit arms on record, per model seed and pooled (mean of the two model seeds), beside the arm E5c. levels_seed {key: {"C": [s0, s1]}}.
    VD5c and VD5cx always; F2, F3, F4 only when include_f (card ADD104 read with every control passed), else a line that says they are left out."""
    keys = [ARM, "VD5c", "VD5cx"] + (["F%d" % s for s in FITS] if include_f else [])
    out = {"arms": {}}
    for k in keys:
        c = [float(x) for x in levels_seed[k]["C"]]
        out["arms"][k] = {"model_seed_0": c[0], "model_seed_1": c[1], "pooled": 0.5 * (c[0] + c[1])}
    out["note"] = ("F2, F3, F4 of card ADD104 are included (that card's controls x0 to x6 all read PASS)" if include_f else
                   "F2, F3, F4 of card ADD104 are left out: no controls file of that card with x0 to x6 all PASS was given")
    return out


def assemble_controls(controls, payload):
    ok = bool(controls) and all(isinstance(v, dict) and v.get("PASS") is True for v in controls.values())
    res = {"card": CARD, "tier": TIER, "controls": controls, "controls_all_pass": ok}
    res["payload" if ok else "withheld"] = payload
    return res


# ============================================================================================ controls
def sha_file(p):
    return MEG.sha_file(p)


def add104_ok(path):
    """-> (bool, why). True only for the controls file of card ADD104 whose controls x0 to x6 are all present and PASS (the flag --add104-controls). The values of that file are not read."""
    if not path:
        return False, "no --add104-controls argument"
    if not os.path.isfile(path):
        return False, "the file %s does not exist" % path
    try:
        j = json.load(io.open(path, encoding="utf-8"))
    except (ValueError, OSError):
        return False, "the file cannot be read as json"
    ctl = j.get("controls") if isinstance(j, dict) else None
    if not isinstance(ctl, dict) or j.get("card") != CARD104 or sorted(ctl) != sorted(ADD104_CONTROLS):
        return False, "not a controls file of card ADD104 with x0 to x6"
    if any((not isinstance(v, dict)) or v.get("PASS") is not True for v in ctl.values()) or j.get("controls_all_pass") is not True:
        return False, "a control of card ADD104 does not read PASS"
    return True, "x0 to x6 all PASS"


def ctl_e0(rec, gap_dir, eval_join_sha, now):
    """e0 (the gap file): the record of make_ens_gapfile.py has e0 PASS; the same checks read again from disk now (the five dumps, the two records of the cards ADD102 and ADD104, the gap file as it lies on disk)
    pass and see the files the record saw (dump hashes, record hashes, gap-file hash); the record is for this evaluation join. now = MEG.check_now(...) = (e0, facts, shas, gap file sha256, meta) or None."""
    if not isinstance(rec, dict):
        return {"PASS": False, "failures": ["no gap-file record"]}
    bad = []
    if (rec.get("e0") or {}).get("PASS") is not True:
        bad.append("the record's own e0 is not PASS")
    if rec.get("card") != CARD:
        bad.append("the record is for card %s" % rec.get("card"))
    if now is None:
        bad.append("e0 cannot be read from disk now")
        return {"PASS": False, "failures": bad}
    e0_now, _, shas_now, gsha, meta_now = now
    bad += ["now: " + x for x in e0_now["failures"]]
    fr = (rec.get("files") or {}).get(MEG.FILE_NAME) or {}
    if gsha != fr.get("file_sha256"):
        bad.append("the gap file on disk is not the one of the record")
    if eval_join_sha and rec.get("eval_join_sha256") != eval_join_sha:
        bad.append("the record is for another evaluation join")
    named = fr.get("dump_sha256") or {}
    for s in MEG.FITS:
        if named.get("fit%d" % s) != shas_now.get(s):
            bad.append("fit %d: the dump on disk is not the one the record read" % s)
    for k in ("record_add102", "record_add104"):
        if ((rec.get("inputs") or {}).get(k) or {}).get("sha256") != (meta_now.get(k) or {}).get("sha256"):
            bad.append("%s on disk is not the one the record read" % k)
    return {"PASS": not bad, "failures": bad, "now": e0_now["clauses"], "record_clauses": (rec.get("e0") or {}).get("clauses")}


WRAP_RE = re.compile(r"wrapper sha256 ([0-9a-f]{64})")


def ctl_e1(wrapper_sha16, check_text, hashes_text, run_shas):
    """e1 (the wrapper): the wrapper file is 28bc31f21ecb4ccb, the pod check record of card ADD102 names it with a pass (last line of CHECK_WRAPPER_V_DONE.txt and the hash in CHECK_WRAPPER_V.hashes), and both
    run texts of the two rollouts name that wrapper hash."""
    lines = [x for x in (check_text or "").splitlines() if x.strip()]
    bad = []
    if wrapper_sha16 != WRAPPER_SHA:
        bad.append("the wrapper file is %s, not %s" % (wrapper_sha16, WRAPPER_SHA))
    if not lines or lines[-1].strip() != CHECK_LAST:
        bad.append("the last line of the pod check is %r" % (lines[-1].strip() if lines else None))
    if WRAPPER_SHA not in (hashes_text or ""):
        bad.append("the pod check hashes do not name %s" % WRAPPER_SHA)
    other = [k for k, v in run_shas.items() if v is not None and not v.startswith(WRAPPER_SHA)]
    if other:
        bad.append("run files with another wrapper hash: %s" % sorted(other))
    n_found = sum(1 for v in run_shas.values() if v is not None)
    if n_found != len(run_shas) or not run_shas:
        bad.append("run files with a wrapper hash: %d of %d" % (n_found, len(run_shas)))
    return {"PASS": not bad, "failures": bad, "wrapper_sha16": wrapper_sha16, "pod_check_last_line": lines[-1].strip() if lines else None, "run_files_found": n_found, "run_files_expected": len(run_shas)}


def rows_of(logs):
    """-> (ci, j, present, g_src, c_src) over the logs (tick log and later-look log) of one rollout."""
    cols = {k: [] for k in ("ci", "j", "present", "g_src", "c_src")}
    for z in logs:
        for k in cols:
            cols[k].append(np.asarray(z[k]))
    return (np.concatenate(cols["ci"]).astype(np.int64), np.concatenate(cols["j"]).astype(np.int64), np.concatenate(cols["present"]).astype(bool),
            np.concatenate(cols["g_src"]).astype(np.float64), np.concatenate(cols["c_src"]).astype(np.float64))


def lead_multi(rows):
    """{(ci, j): [g_src, ...]} of the rows with a lead: a frame logged more than once keeps every value (they differ in the float noise of the reader)"""
    ci, j, pres, g, _ = rows
    m = {}
    for i in np.flatnonzero(pres):
        m.setdefault((int(ci[i]), int(j[i])), []).append(float(g[i]))
    return m


def gapfile_ref(path):
    """{(clip index, frame): gap or None} from the gap file as it lies on disk (a plain loop over its rows; None where it has no output)"""
    z = MEG.load_npz(path)
    ci, fr, g, h = np.asarray(z["clip_index"]), np.asarray(z["frame"]), np.asarray(z["gap"], np.float64), np.asarray(z["has_out"], bool)
    return {(int(ci[i]), int(fr[i])): (float(g[i]) if (h[i] and np.isfinite(g[i])) else None) for i in range(len(ci))}


def w2_one(rows, ref_gap, v0_multi, meta_expect=None, meta=None):
    """control e2 on one rollout. rows from rows_of; ref_gap {(ci, frame): gap or None} from the gap file; v0_multi {(ci, j): [gap, ...]} of the control run V0 of card ADD102. On every row with a lead: where the file
    has an output the gap the rule received equals the gap of the file within 1e-3 m; where it has none it equals a gap of V0 at the same clip and frame within 1e-3 m, any of them. The closing input is non-zero on
    a positive share of the lead rows. The rule log meta names the gap file hash and the closing switch of the arm."""
    ci, j, pres, g, c = rows
    n_head = n_bad_head = n_noout = n_ref_missing = n_bad_ref = 0
    for i in np.flatnonzero(pres):
        k = (int(ci[i]), int(j[i]))
        hg = ref_gap.get(k) if ref_gap is not None else None
        if hg is not None:
            n_head += 1
            n_bad_head += int(abs(g[i] - hg) > TOL_W2)
        else:
            n_noout += 1
            refs = v0_multi.get(k)
            if not refs:
                n_ref_missing += 1
            else:
                n_bad_ref += int(min(abs(g[i] - r) for r in refs) > TOL_W2)
    n_lead = int(pres.sum())
    nz_lead = int((c[pres] != 0).sum())
    bad = []
    if n_lead == 0:
        bad.append("no rows with a lead")
    if ref_gap is not None and n_head == 0:
        bad.append("no lead row with a gap-file output")
    if n_bad_head:
        bad.append("%d lead rows with a gap-file output differ from the gap of the file by more than %g m" % (n_bad_head, TOL_W2))
    if n_ref_missing or n_bad_ref:
        bad.append("lead rows checked against V0: %d have no V0 row, %d differ from every V0 value by more than %g m" % (n_ref_missing, n_bad_ref, TOL_W2))
    if nz_lead == 0:
        bad.append("the closing input is 0 on every lead row")
    if meta_expect is not None:
        for k, v in meta_expect.items():
            if str((meta or {}).get(k)) != str(v):
                bad.append("rule log meta %s is %s, expected %s" % (k, (meta or {}).get(k), v))
    return {"PASS": not bad, "failures": bad, "n_rows": int(len(ci)), "n_lead_rows": n_lead, "n_lead_rows_with_head_output": n_head, "n_lead_rows_without_head_output": n_noout,
            "n_bad_head": n_bad_head, "n_ref_missing": n_ref_missing, "n_bad_ref": n_bad_ref, "n_lead_rows_closing_nonzero": nz_lead,
            "share_of_rule_steps_with_a_lead": (n_lead / len(ci)) if len(ci) else None, "share_of_lead_steps_with_nonzero_closing": (nz_lead / n_lead) if n_lead else None}


def ctl_e2(per_run):
    ok = bool(per_run) and sorted(per_run) == sorted("%s_s%d" % (ARM, m) for m in SEEDS) and all(v["PASS"] for v in per_run.values())
    return {"PASS": ok, "per_run": per_run}


def ctl_e3(levels):
    """the reader recomputes C of RR, RLc and VD5c from the banked rollouts: 0.0954, 0.0000 and 0.0169 to four decimals"""
    got = {k: levels.get(k) for k in C_EXPECTED}
    ok = all(v is not None and round(float(v), 4) == C_EXPECTED[k] for k, v in got.items())
    return {"PASS": bool(ok), "C": got, "expected": dict(C_EXPECTED)}


def ctl_e4(tests_text):
    t = tests_text or ""
    m = re.findall(r"(\d+) passed", t)
    ok = bool(m) and int(m[-1]) > 0 and not re.search(r"\d+ (failed|error)", t)
    return {"PASS": ok, "passed": int(m[-1]) if m else None}


def ctl_e5(grid_all_pass, strat_a, strat_b):
    ok = grid_all_pass is True and strat_a is True and strat_b is True
    return {"PASS": bool(ok), "grid_all_pass": grid_all_pass, "strategic_stage_A_pass": strat_a, "strategic_stage_B_pass": strat_b}


# ============================================================================================ the pod part
def load_logs(pull, run, m):
    d = os.path.join(pull, "I10%s_s%d" % (run, m), "ticks")
    names = ["rule_log_I10%s_s%d.npz" % (run, m), "rule_log_mid_I10%s_s%d.npz" % (run, m)]
    zs = []
    for nm in names:
        p = os.path.join(d, nm)
        if not os.path.isfile(p):
            raise SystemExit("REFUSED: no rule log %s" % p)
        with np.load(p, allow_pickle=False) as z:
            zs.append({k: z[k] for k in z.files})
    return zs


def meta_of(z):
    return {k: (z[k].item() if z[k].shape == () else None) for k in ("vis_gap_file_sha256", "vis_closing") if k in z}


def run_file_sha(pull, run, m):
    p = os.path.join(pull, "I10%s_s%d" % (run, m), "ticks", "run_I10%s_s%d.txt" % (run, m))
    if not os.path.isfile(p):
        return None
    mm = WRAP_RE.search(io.open(p, encoding="utf-8", errors="replace").read())
    return mm.group(1) if mm else None


def e2_from_files(a, rec):
    """-> control e2. Arm E5c at model seed m: the gap file VGAPE5 as it lies on disk (its sha256 = the record's, named by the meta of the rule log), closing switch 1; V0 of card ADD102 for the rows with no output."""
    ref = gapfile_ref(os.path.join(a.gap_dir, MEG.FILE_NAME + ".npz"))
    sha = (((rec or {}).get("files") or {}).get(MEG.FILE_NAME) or {}).get("file_sha256")
    per = {}
    for m in SEEDS:
        v0_multi = lead_multi(rows_of(load_logs(a.pull, V0_RUN, m)))
        logs = load_logs(a.pull, RUN[ARM], m)
        per["%s_s%d" % (ARM, m)] = w2_one(rows_of(logs), ref, v0_multi, meta_expect={"vis_gap_file_sha256": sha, "vis_closing": 1}, meta=meta_of(logs[0]))
    return ctl_e2(per)


def measure(a, E, log, include_f):
    """the pod part: grid read (levels, pairs with intervals, families) of the arm run and the banked RR, RLc, VD5c, VD5cx (and F2, F3, F4 when include_f); then the strategic grid. -> dict"""
    H, Q3, RH = E
    g30 = R.jl(os.path.join(a.grid_dir, "grid_p30.json"))
    J = H.load_join()
    keys = R.grid_keys(g30, [str(c) for c in J["clip"]])
    runs = dict(RUN)
    runs.update(BANKED)
    runs.update(B102)
    runs.update(B103)
    if include_f:
        runs.update(RUN_F)
    for k, r in runs.items():
        R.ALL_RUNS.setdefault(r, ("RLc" if k == "RLc" else "RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    R.ALL_RUNS.setdefault(EC.SETTINGS["AGE"]["runs"]["RR"], ("RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    pairs_def = {ARM + "-VS-RR": (RUN[ARM], BANKED["RR"]), ARM + "-VS-RLc": (RUN[ARM], BANKED["RLc"]), ARM + "-VS-VD5c": (RUN[ARM], B102["VD5c"])}
    spread = "R0Hb20"
    run_list = [spread] + sorted(set(runs.values()))
    G = R.read_grid(a, E, "30", keys, run_list, spread, pairs_def, {}, log)
    lv = {k: {c: float(np.mean(G["levels"][r][c])) for c in ("C", "U", "F")} for k, r in runs.items()}
    lv_seed = {k: {c: [float(v) for v in np.asarray(G["levels"][r][c]).reshape(-1)[:2]] for c in ("C", "U", "F")} for k, r in runs.items()}
    pairs = {nm: R.ucf(p["headway"]) for nm, p in G["pairs"].items()}
    fams = {nm: {f_: R.reading(R.cells_of(p["family"]), f_) for f_ in R.NAMED} for nm, p in G["pairs"].items()}
    # the STRATEGIC family as the reader of card ADD102 reads it (engine strategic_grid with its own controls, then the pair read at tick t0 + 20 on the arcs of 30, 50 and 80 m): the arm against RR, RLc and VD5c
    p_of = {"R0": 0}
    p_of.update({r: 30 for k, r in runs.items() if k in (ARM, "RR", "RLc", "VD5c")})
    SR, cells_s, eid_s, S = R.strategic_grid(a, keys, sorted(p_of), p_of, log, {})
    sa, sb = bool(SR.get("stage_A_pass")), bool(SR.get("stage_B_pass"))
    strat = {"controls": dict({k: v for k, v in SR["controls"].items() if k != "s3"}, s3=SR["controls"]["s3"].get("pass")), "pairs": {}}
    if sa and sb:
        for ref, rn in REF_RUNS.items():
            pr = S.pair_read(cells_s, RUN[ARM], rn, eid_s, ticks=(2,))
            pos = pr["tick2"]["pos"]
            strat["pairs"][ARM + "-VS-" + ref] = {"A": RUN[ARM], "B": rn, "pos_30_50_80m_tick_t0_plus_20": [{"arc_m": arc, "mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "verdict": b["verdict"],
                                                                                                          "n_per_seed": b.get("n_per_seed")} for arc, b in zip(S.ARCS, pos)],
                                                  "reading": R.family_of([c["verdict"] for c in pos])}
    return {"levels": lv, "levels_per_seed": lv_seed, "pairs": pairs, "families": fams, "grid_all_pass": G["controls"].get("ALL_PASS") is True, "strategic": strat, "strategic_a": sa, "strategic_b": sb}


def run_controls(a):
    log = lambda s: print(s, flush=True)                                                       # noqa: E731
    include_f, why = add104_ok(a.add104_controls)
    log("single-fit arms F2, F3, F4 of card ADD104: %s (%s)" % ("included" if include_f else "left out", why))
    E = R.engine(a)
    m = measure(a, E, log, include_f)
    rec = R.jl(a.gap_record)
    ejs = sha_file(a.eval_join) if a.eval_join and os.path.isfile(a.eval_join) else None
    rd = lambda p: io.open(p, encoding="utf-8").read() if p and os.path.isfile(p) else ""      # noqa: E731
    now = MEG.check_now(a.eval_join, os.path.join(a.gap_dir, MEG.FILE_NAME + ".npz"), a.old_dump_root, a.new_dump_root, a.rec102, a.rec104)
    c2 = e2_from_files(a, rec)
    run_shas = {"%s_s%d" % (ARM, s): run_file_sha(a.pull, RUN[ARM], s) for s in SEEDS}
    wsha = sha_file(a.wrapper_file)[:16] if a.wrapper_file and os.path.isfile(a.wrapper_file) else None
    controls = {"e0": ctl_e0(rec, a.gap_dir, ejs, now), "e1": ctl_e1(wsha, rd(a.wrapper_check), rd(a.wrapper_hashes), run_shas), "e2": c2,
                "e3": ctl_e3({k: m["levels"][k]["C"] for k in C_EXPECTED}), "e4": ctl_e4(rd(a.tests_out)), "e5": ctl_e5(m["grid_all_pass"], m["strategic_a"], m["strategic_b"])}
    payload = {"levels": m["levels"], "levels_per_seed": m["levels_per_seed"], "pairs": m["pairs"], "families": m["families"], "strategic": m["strategic"], "family_note": FAM_REASON,
               "single_fit_contact": single_fit_table(m["levels_per_seed"], include_f), "t0_gap_error_diagnostic": (rec or {}).get("t0_gap_error_diagnostic"),
               "unsafe_rate": {nm: p["U"] for nm, p in sorted(m["pairs"].items())},
               "falling_behind_against_RLc": {ARM: m["pairs"][ARM + "-VS-RLc"]["F"]},
               "rollout_shares": {k: {"share_of_rule_steps_with_a_lead": v["share_of_rule_steps_with_a_lead"], "share_of_lead_steps_with_nonzero_closing": v["share_of_lead_steps_with_nonzero_closing"],
                                      "n_rows": v["n_rows"], "n_lead_rows": v["n_lead_rows"], "n_lead_rows_closing_nonzero": v["n_lead_rows_closing_nonzero"]} for k, v in c2["per_run"].items()}}
    return assemble_controls(controls, payload)


def print_sections(payload):
    L = []
    row = [ARM]
    for ref in ("RR", "RLc", "VD5c"):
        p = payload["pairs"].get(ARM + "-VS-" + ref)
        if p:
            row.append("vs %s: C %s U %s F %s" % (ref, p["C"]["interval"], p["U"]["interval"], p["F"]["interval"]))
        st = (payload.get("strategic", {}).get("pairs") or {}).get(ARM + "-VS-" + ref)
        fm = (payload.get("families") or {}).get(ARM + "-VS-" + ref) or {}
        row.append("families(%s) %s strategic %s" % (ref, json.dumps(fm, sort_keys=True), st["reading"] if st else None))
    L.append("   ".join(row))
    L.append("levels (pooled over the two model seeds): " + json.dumps(payload.get("levels"), sort_keys=True))
    L.append("levels per model seed: " + json.dumps(payload.get("levels_per_seed"), sort_keys=True))
    L.append("falling behind against RLc: " + json.dumps(payload.get("falling_behind_against_RLc"), sort_keys=True))
    L.append("single-fit arms, contact rate: " + json.dumps(payload.get("single_fit_contact"), sort_keys=True))
    L.append("T0 diagnostic, mean absolute gap error on the rows with a label lead and an output of all five fits: " + json.dumps(payload.get("t0_gap_error_diagnostic"), sort_keys=True))
    L.append("rollout shares: " + json.dumps(payload.get("rollout_shares"), sort_keys=True))
    return L


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage", choices=("controls", "names"), required=True)
    for n in ("pull", "code", "grid-dir", "w", "tree", "wrapper-file", "gap-record", "gap-dir", "eval-join", "wrapper-check", "wrapper-hashes", "tests-out", "controls", "add104-controls"):
        ap.add_argument("--" + n, default="")
    ap.add_argument("--old-dump-root", default=MEG.OLD_ROOT)
    ap.add_argument("--new-dump-root", default=MEG.NEW_ROOT)
    ap.add_argument("--rec102", default=MEG.REC102)
    ap.add_argument("--rec104", default=MEG.REC104)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    a.n_boot_control = 300
    if os.path.exists(a.out):
        print("REFUSED: %s exists" % a.out)
        return 2
    if a.stage == "controls":
        res = run_controls(a)
        io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1, default=lambda o: float(o) if hasattr(o, "item") else str(o)))
        for k, v in res["controls"].items():
            print("%s %s" % (k, "PASS" if v.get("PASS") else "FAIL"))
        print("ZZ READ-VISENS10K controls rc 0 controls_all_pass %s" % res["controls_all_pass"])
        return 0
    cj = json.load(io.open(a.controls, encoding="utf-8"))
    if cj.get("card") != CARD or cj.get("controls_all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != sorted(CONTROLS) \
            or any(v.get("PASS") is not True for v in cj.get("controls", {}).values()):
        print("REFUSED: the controls file is not all-pass: no name is written")
        print("ZZ READ-VISENS10K names rc 3 controls_all_pass False")
        return 3
    names = all_names(cj["payload"])
    res = {"card": CARD, "tier": TIER, "controls_file": os.path.abspath(a.controls), "controls_sha256": sha_file(a.controls), "names": names, "h_note": h_sensitivity(cj["payload"]),
           "values": cj["payload"]}
    io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1))
    for k in sorted(names):
        print(names[k])
    for line in print_sections(cj["payload"]):
        print(line)
    print("ZZ READ-VISENS10K names rc 0 controls_all_pass True")
    return 0


def selftest():
    ok = True
    n = names_of(ARM, [-0.05, -0.01], [0.0, 0.004], [-0.03, 0.001], [-0.02, 0.03], 0.0954, 0.0)               # H = 0.0477
    ok &= n[ARM + "_CONTACT"].endswith("E5c-CUTS-CONTACT") and n[ARM + "_HALF"].endswith("WITHIN-HALF-THE-GAP") and n[ARM + "_VD5C"].endswith("CONTACT-NOT-SEPARATED-FROM-VD5c")
    ok &= n[ARM + "_PRICE"].endswith("FALLING-BEHIND-NOT-SEPARATED") and card_name(n) == PRE + "-WITHIN-HALF-THE-GAP"
    n = names_of(ARM, [0.001, 0.01], [0.05, 0.06], [0.001, 0.03], [0.01, 0.04], 0.0954, 0.0)
    ok &= n[ARM + "_CONTACT"].endswith("E5c-MORE-CONTACT") and n[ARM + "_HALF"].endswith("E5c-HALF-THE-GAP-NOT-REACHED") and n[ARM + "_VD5C"].endswith("MORE-CONTACT-THAN-VD5c")
    ok &= n[ARM + "_PRICE"].endswith("FALLS-BEHIND-MORE") and card_name(n) == PRE + "-DOES-NOT-CUT"
    ok &= ctl_e3({"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0169})["PASS"] and not ctl_e3({"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0175})["PASS"]
    print("ZZ SELFTEST-READ-VISENS10K ok: %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
