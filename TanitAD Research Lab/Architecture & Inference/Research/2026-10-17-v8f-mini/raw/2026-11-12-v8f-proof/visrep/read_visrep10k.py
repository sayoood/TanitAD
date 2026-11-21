#!/usr/bin/env python
"""read_visrep10k.py -- read of card PREREG_V8F_PROOF_ADD103 (VISREP10K): the reader-fed lead rule (form RR) of the wrapper variant of card ADD102 (28bc31f21ecb4ccb, unchanged) with the slope closing input, in three arms:
VD5cx (the gap of the head on five frames of the other fit seed), VRc (the present reader's own gap) and VD1c (the gap of the head on one frame), on the model's own rollout (T1-kin/hb0.2, pre-roll of 30 steps (6 s)).
Derived from visrule/read_visrule10k.py (card ADD102): the engine of the model reader EC, R, HS, H is imported, not edited.
Stage controls: computes the grid read and the STRATEGIC family of the six arm runs and the banked RR, RLc, KD5, VD5c, VD5 inside this stage, runs w0, w1, w2, w3, w5 here and reads w4 (--tests-out); writes CONTROLS_FILE.
With every control PASS the values are written under `payload`, else under `withheld` and no name exists.
Stage names: reads CONTROLS_FILE, refuses it unless all controls pass, writes the names of card section 3 (four per arm and the three card names) and binds the result to the controls file by sha256.
Usage: python read_visrep10k.py --stage controls --pull PULL --code CODE --grid-dir DIR --w W --tree TREE --out CONTROLS.json --wrapper-file F --wrapper-check FILE --wrapper-hashes FILE --tests-out FILE
          --gap-record REC.json --gap-dir DIR --add102-gap-dir DIR --eval-join J.npz --dump-root DIR [--n-boot 2000]
       python read_visrep10k.py --stage names --controls CONTROLS.json --out RESULT.json     |   --selftest
Last line: ZZ READ-VISREP10K <stage> rc <rc> controls_all_pass <bool>"""
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
import make_rep_gapfile as MRG              # noqa: E402

CARD, PRE = "PREREG_V8F_PROOF_ADD103", "VISREP10K"
SEEDS = (0, 1)
ARMS = ("VD5cx", "VRc", "VD1c")
RUN = {"VD5cx": "RQ50KF5P30Hb20wVD5cx", "VRc": "RQ50KF5P30Hb20wVRc", "VD1c": "RQ50KF5P30Hb20wVD1c"}
B102 = {"VD5c": "RQ50KF5P30Hb20vVD5c", "VD5": "RQ50KF5P30Hb20vVD5"}          # banked arms of card ADD102
V0_RUN = "RQ50KF5P30Hb20vV0"                                                   # banked control run of card ADD102: the reader's own gap as RR feeds it, logged at every lead step
BANKED = {"RR": "RQ50KF5P30Hb20", "RLc": "RLcKF5P30Hb20"}
REFS = {"KD5": "RLcKF5P30Hb20pKD5"}                                            # card ADD101, banked
C_EXPECTED = {"RR": 0.0954, "RLc": 0.0, "KD5": 0.0169, "VD5c": 0.0169, "VD5": 0.0769}
H_CARD = 0.0477
TOL_W2 = 1e-3
WRAPPER_SHA = "28bc31f21ecb4ccb"
CHECK_LAST = "ZZ CHECK-WRAPPER-V PASS"
DUMP5 = "D_H5"                                                                 # the head on five frames (VD5cx reads fit seed 1 - k)
TIER = "T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s)"
FAM_REASON = ("the rule changes the planned speed along the path through the distance to the lead; it does not move the path, so the lateral family is expected to be unmoved by it, "
              "and the tactical family is not read by the rule; the strategic family is read from the strategic grid of the engine")
CONTROLS = ("w0", "w1", "w2", "w3", "w4", "w5")
REF_RUNS = {"RR": BANKED["RR"], "RLc": BANKED["RLc"], "VD5c": B102["VD5c"]}    # the references of the strategic family


def _bracket(iv, up, down, mid):
    return up if iv[1] < 0.0 else (down if iv[0] > 0.0 else mid)


def names_of(arm, d_rr_c, d_rlc_c, d_vd5c_c, d_rr_f, c_rr, c_rlc):
    """the four names of one arm (card section 3). d_*: interval [lo, hi] of C(X)-C(RR), C(X)-C(RLc), C(X)-C(VD5c), F(X)-F(RR); c_rr, c_rlc: pooled point values of C of the references."""
    p = PRE + "-" + arm + "-"
    H_ = 0.5 * (c_rr - c_rlc)
    n1 = p + ("CUTS-CONTACT" if d_rr_c[1] < 0.0 else ("MORE-CONTACT" if d_rr_c[0] > 0.0 else "CONTACT-NOT-SEPARATED-FROM-RR"))
    n2 = p + ("WITHIN-HALF-THE-GAP" if d_rlc_c[1] <= H_ else ("HALF-THE-GAP-NOT-REACHED" if d_rlc_c[0] > H_ else "HALF-THE-GAP-NOT-DECIDED"))
    n3 = p + ("LESS-CONTACT-THAN-VD5c" if d_vd5c_c[1] < 0.0 else ("MORE-CONTACT-THAN-VD5c" if d_vd5c_c[0] > 0.0 else "CONTACT-NOT-SEPARATED-FROM-VD5c"))
    n4 = p + ("FALLS-BEHIND-MORE" if d_rr_f[0] > 0.0 else ("FALLS-BEHIND-LESS" if d_rr_f[1] < 0.0 else "FALLING-BEHIND-NOT-SEPARATED"))
    return {arm + "_CONTACT": n1, arm + "_HALF": n2, arm + "_VD5C": n3, arm + "_PRICE": n4}


def replicate_name(contact_name, half_name):
    """card name of the replicate, from the two names of VD5cx: REPLICATED when CUTS-CONTACT and WITHIN-HALF-THE-GAP; NOT-REPLICATED when HALF-THE-GAP-NOT-REACHED or not CUTS-CONTACT; else NOT-DECIDED"""
    cuts = contact_name.endswith("-CUTS-CONTACT")
    if cuts and half_name.endswith("-WITHIN-HALF-THE-GAP"):
        return PRE + "-REPLICATED"
    if half_name.endswith("-HALF-THE-GAP-NOT-REACHED") or not cuts:
        return PRE + "-NOT-REPLICATED"
    return PRE + "-REPLICATION-NOT-DECIDED"


def gap_name(iv):
    """from the pair VD5c minus VRc"""
    return PRE + "-CROP-GAP-" + _bracket(iv, "CUTS-CONTACT", "MORE-CONTACT", "NOT-SEPARATED-FROM-READER-GAP")


def slope_name(iv):
    """from the pair VD5c minus VD5 (both banked)"""
    return PRE + "-SLOPE-" + _bracket(iv, "CUTS-CONTACT", "MORE-CONTACT", "NOT-SEPARATED")


def card_names(payload, names):
    pr = payload["pairs"]
    return {"CARD_REPLICATE": replicate_name(names["VD5cx_CONTACT"], names["VD5cx_HALF"]), "CARD_GAP": gap_name(pr["VD5c-VS-VRc"]["C"]["interval"]),
            "CARD_SLOPE": slope_name(pr["VD5c-VS-VD5"]["C"]["interval"])}


def card_check(payload, got):
    """independent re-derivation of the three card names from the intervals of the payload (not from the other names); {"PASS", "expected"}"""
    pr = payload["pairs"]
    h = 0.5 * (payload["levels"]["RR"]["C"] - payload["levels"]["RLc"]["C"])
    rlo, rhi = pr["VD5cx-VS-RR"]["C"]["interval"]
    hlo, hhi = pr["VD5cx-VS-RLc"]["C"]["interval"]
    cut, within, notreached = rhi < 0.0, hhi <= h, hlo > h
    exp = {"CARD_REPLICATE": PRE + ("-REPLICATED" if (cut and within) else ("-NOT-REPLICATED" if (notreached or not cut) else "-REPLICATION-NOT-DECIDED"))}
    for key, pair, tail in (("CARD_GAP", "VD5c-VS-VRc", ("-CROP-GAP-CUTS-CONTACT", "-CROP-GAP-MORE-CONTACT", "-CROP-GAP-NOT-SEPARATED-FROM-READER-GAP")),
                            ("CARD_SLOPE", "VD5c-VS-VD5", ("-SLOPE-CUTS-CONTACT", "-SLOPE-MORE-CONTACT", "-SLOPE-NOT-SEPARATED"))):
        lo, hi = pr[pair]["C"]["interval"]
        exp[key] = PRE + (tail[0] if hi < 0.0 else (tail[1] if lo > 0.0 else tail[2]))
    return {"PASS": all(got.get(k) == v for k, v in exp.items()), "expected": exp}


def h_sensitivity(payload):
    """notes, no name: the half-the-gap name of each arm again with H = 0.0477 as the card prints it; `differs` is true when that name is not the one made with the point values"""
    pr = payload["pairs"]
    h = 0.5 * (payload["levels"]["RR"]["C"] - payload["levels"]["RLc"]["C"])
    out = {"H_point_values": h, "H_card": H_CARD, "differs": False}
    for arm in ARMS:
        lo, hi = pr[arm + "-VS-RLc"]["C"]["interval"]
        f = lambda H: "WITHIN" if hi <= H else ("NOT-REACHED" if lo > H else "NOT-DECIDED")      # noqa: E731
        out[arm] = {"with_point_values": f(h), "with_H_card": f(H_CARD)}
        out["differs"] = bool(out["differs"] or f(h) != f(H_CARD))
    return out


def all_names(payload):
    out = {}
    pr = payload["pairs"]
    for a in ARMS:
        out.update(names_of(a, pr[a + "-VS-RR"]["C"]["interval"], pr[a + "-VS-RLc"]["C"]["interval"], pr[a + "-VS-VD5c"]["C"]["interval"], pr[a + "-VS-RR"]["F"]["interval"],
                            payload["levels"]["RR"]["C"], payload["levels"]["RLc"]["C"]))
    out.update(card_names(payload, out))
    chk = card_check(payload, out)
    if not chk["PASS"]:
        raise SystemExit("REFUSED: the card names %s are not the ones the intervals give (%s)" % ({k: out[k] for k in chk["expected"]}, chk["expected"]))
    return out


def assemble_controls(controls, payload):
    ok = bool(controls) and all(isinstance(v, dict) and v.get("PASS") is True for v in controls.values())
    res = {"card": CARD, "tier": TIER, "controls": controls, "controls_all_pass": ok}
    res["payload" if ok else "withheld"] = payload
    return res


# ============================================================================================ controls
def sha_file(p):
    return MRG.sha_file(p)


def ctl_w0(rec, gap_dir, add102_dir, eval_join_sha=None):
    """w0: the dumps of D-H1 are on the frames of the evaluation join and the saved gap files equal a second routine's value on every row (the record of make_rep_gapfile.py: v0 and build), the files on disk have the
    hashes of the record and the record is for this evaluation join; and the two gap files of card ADD102 on disk have the hashes of that card's record (checked again here, and equal to what the record saw)."""
    if not isinstance(rec, dict):
        return {"PASS": False, "failures": ["no gap-file record"]}
    c0, cb = MRG.v0_check(rec), MRG.build_check(rec)
    bad = list(c0["failures"]) + list(cb["failures"])
    for nm in MRG.file_names():
        p = os.path.join(gap_dir or "", nm + ".npz")
        if not os.path.isfile(p) or sha_file(p) != (rec.get("files", {}).get(nm) or {}).get("file_sha256"):
            bad.append("%s: file missing or its sha256 is not the record's" % nm)
    if eval_join_sha and rec.get("eval_join_sha256") != eval_join_sha:
        bad.append("the record is for another evaluation join")
    a1 = MRG.add102_check(add102_dir)
    bad += ["add102 now: " + x for x in a1["failures"]]
    seen = ((rec.get("add102") or {}).get("hashes")) or {}
    for nm, v in a1.get("hashes", {}).items():
        if (seen.get(nm) or {}).get("disk") != v.get("disk"):
            bad.append("%s of card ADD102 is not the file the record saw" % nm)
    if (rec.get("w0") or {}).get("PASS") is not True:
        bad.append("the record's own w0 is not PASS")
    return {"PASS": not bad, "failures": bad, "record_v0": c0, "record_build": cb, "add102_now": a1}


WRAP_RE = re.compile(r"wrapper sha256 ([0-9a-f]{64})")


def ctl_w1(wrapper_sha16, check_text, hashes_text, run_shas):
    """w1: the wrapper file is 28bc31f21ecb4ccb, the pod check record of card ADD102 names it with a pass (last line of CHECK_WRAPPER_V_DONE.txt and the hash in CHECK_WRAPPER_V.hashes); in addition every run file
    of the six rollouts that is there names that wrapper hash (six expected)."""
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


def w2_one(rows, ref_gap, v0_multi, meta_expect=None, meta=None):
    """control w2 on one rollout. rows from rows_of; ref_gap {(ci, frame): head gap or None} from the head's dump (a plain loop, make_rep_gapfile.second_routine), or None for the arm with no head (VRc);
    v0_multi {(ci, j): [gap, ...]} of the control run V0 of card ADD102. On every row with a lead: where the head has an output the gap the rule received equals the dumped gap within 1e-3 m; where it has none
    (and for every lead row of VRc) it equals a gap of V0 at the same clip and frame within 1e-3 m, any of them. The closing input is non-zero on a positive share of the lead rows. The rule log meta names the
    gap file hash and the closing switch of the arm."""
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
        bad.append("no lead row with a head output")
    if n_bad_head:
        bad.append("%d lead rows with a head output differ from the dumped gap by more than %g m" % (n_bad_head, TOL_W2))
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


def ctl_w2(per_run):
    ok = bool(per_run) and all(v["PASS"] for v in per_run.values())
    return {"PASS": ok, "per_run": per_run}


def ctl_w3(levels):
    """the reader recomputes C of RR, RLc, KD5, VD5c and VD5 from the banked rollouts: 0.0954, 0.0000, 0.0169, 0.0169 and 0.0769 to four decimals"""
    got = {k: levels.get(k) for k in C_EXPECTED}
    ok = all(v is not None and round(float(v), 4) == C_EXPECTED[k] for k, v in got.items())
    return {"PASS": bool(ok), "C": got, "expected": dict(C_EXPECTED)}


def ctl_w4(tests_text):
    t = tests_text or ""
    m = re.findall(r"(\d+) passed", t)
    ok = bool(m) and int(m[-1]) > 0 and not re.search(r"\d+ (failed|error)", t)
    return {"PASS": ok, "passed": int(m[-1]) if m else None}


def ctl_w5(grid_all_pass, strat_a, strat_b):
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


def w2_from_files(a, rec, rec102):
    """-> (control w2, J). expected meta: VD5cx the sha256 of VGAP_s<1-k> of card ADD102, VD1c that of VGAP1_s<k> (this record), VRc no gap file (the empty string); the closing switch 1 in all."""
    J = MRG.load_npz(a.eval_join)
    refs5, refs1 = {}, {}
    for m in SEEDS:
        refs5[m] = MRG.second_routine(J, MRG.load_npz(os.path.join(a.dump_root, DUMP5, "reader_s%d_score.npz" % m)))
        refs1[m] = MRG.second_routine(J, MRG.load_npz(os.path.join(a.dump_root, MRG.DUMP_DIR, "reader_s%d_score.npz" % m)))
    per = {}
    for m in SEEDS:
        v0_multi = lead_multi(rows_of(load_logs(a.pull, V0_RUN, m)))
        for arm in ARMS:
            logs = load_logs(a.pull, RUN[arm], m)
            if arm == "VD5cx":
                ref, sha = refs5[1 - m], ((rec102 or {}).get("files", {}).get("VGAP_s%d" % (1 - m)) or {}).get("file_sha256")
            elif arm == "VD1c":
                ref, sha = refs1[m], ((rec or {}).get("files", {}).get("VGAP1_s%d" % m) or {}).get("file_sha256")
            else:
                ref, sha = None, ""
            r = w2_one(rows_of(logs), ref, v0_multi, meta_expect={"vis_gap_file_sha256": sha, "vis_closing": 1}, meta=meta_of(logs[0]))
            per["%s_s%d" % (arm, m)] = r
    return ctl_w2(per), J


def measure(a, E, log):
    """the pod part: grid read (levels, pairs with intervals, families) of the six arm runs and the banked RR, RLc, KD5, VD5c, VD5; then the strategic grid. -> dict"""
    H, Q3, RH = E
    g30 = R.jl(os.path.join(a.grid_dir, "grid_p30.json"))
    J = H.load_join()
    keys = R.grid_keys(g30, [str(c) for c in J["clip"]])
    runs = dict(RUN)
    runs.update(BANKED)
    runs.update(REFS)
    runs.update(B102)
    for k, r in runs.items():
        R.ALL_RUNS.setdefault(r, ("RLc" if k in ("RLc", "KD5") else "RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    R.ALL_RUNS.setdefault(EC.SETTINGS["AGE"]["runs"]["RR"], ("RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    pairs_def = {}
    for x in ARMS:
        pairs_def[x + "-VS-RR"] = (RUN[x], BANKED["RR"])
        pairs_def[x + "-VS-RLc"] = (RUN[x], BANKED["RLc"])
        pairs_def[x + "-VS-VD5c"] = (RUN[x], B102["VD5c"])
        pairs_def[x + "-VS-KD5"] = (RUN[x], REFS["KD5"])
    pairs_def["VD5c-VS-VRc"] = (B102["VD5c"], RUN["VRc"])                  # the gap: VD5c minus VRc
    pairs_def["VD5c-VS-VD5"] = (B102["VD5c"], B102["VD5"])                 # the slope: VD5c minus VD5
    spread = "R0Hb20"
    run_list = [spread] + sorted(set(runs.values()))
    G = R.read_grid(a, E, "30", keys, run_list, spread, pairs_def, {}, log)
    lv = {k: {c: float(np.mean(G["levels"][r][c])) for c in ("C", "U", "F")} for k, r in runs.items()}
    lv_seed = {k: {c: [float(v) for v in np.asarray(G["levels"][r][c]).reshape(-1)[:2]] for c in ("C", "U", "F")} for k, r in runs.items()}
    pairs = {nm: R.ucf(p["headway"]) for nm, p in G["pairs"].items()}
    fams = {nm: {f_: R.reading(R.cells_of(p["family"]), f_) for f_ in R.NAMED} for nm, p in G["pairs"].items()}
    # the STRATEGIC family as the reader of card ADD102 reads it (engine strategic_grid with its own controls, then the pair read at tick t0 + 20 on the arcs of 30, 50 and 80 m), every arm against RR, RLc and VD5c
    p_of = {"R0": 0}
    p_of.update({r: 30 for r in runs.values()})
    SR, cells_s, eid_s, S = R.strategic_grid(a, keys, sorted(p_of), p_of, log, {})
    sa, sb = bool(SR.get("stage_A_pass")), bool(SR.get("stage_B_pass"))
    strat = {"controls": dict({k: v for k, v in SR["controls"].items() if k != "s3"}, s3=SR["controls"]["s3"].get("pass")), "pairs": {}}
    if sa and sb:
        for x in ARMS:
            for ref, rn in REF_RUNS.items():
                pr = S.pair_read(cells_s, RUN[x], rn, eid_s, ticks=(2,))
                pos = pr["tick2"]["pos"]
                strat["pairs"][x + "-VS-" + ref] = {"A": RUN[x], "B": rn, "pos_30_50_80m_tick_t0_plus_20": [{"arc_m": arc, "mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "verdict": b["verdict"],
                                                                                                           "n_per_seed": b.get("n_per_seed")} for arc, b in zip(S.ARCS, pos)],
                                                    "reading": R.family_of([c["verdict"] for c in pos])}
    return {"levels": lv, "levels_per_seed": lv_seed, "pairs": pairs, "families": fams, "grid_all_pass": G["controls"].get("ALL_PASS") is True, "strategic": strat, "strategic_a": sa, "strategic_b": sb}


def run_controls(a):
    log = lambda s: print(s, flush=True)                                                       # noqa: E731
    E = R.engine(a)
    m = measure(a, E, log)
    rec = R.jl(a.gap_record)
    rec102 = R.jl(os.path.join(a.add102_gap_dir, MRG.ADD102_RECORD))
    ejs = sha_file(a.eval_join) if a.eval_join and os.path.isfile(a.eval_join) else None
    rd = lambda p: io.open(p, encoding="utf-8").read() if p and os.path.isfile(p) else ""      # noqa: E731
    c2, J = w2_from_files(a, rec, rec102)
    run_shas = {"%s_s%d" % (arm, s): run_file_sha(a.pull, RUN[arm], s) for arm in ARMS for s in SEEDS}
    wsha = sha_file(a.wrapper_file)[:16] if a.wrapper_file and os.path.isfile(a.wrapper_file) else None
    controls = {"w0": ctl_w0(rec, a.gap_dir, a.add102_gap_dir, ejs), "w1": ctl_w1(wsha, rd(a.wrapper_check), rd(a.wrapper_hashes), run_shas), "w2": c2,
                "w3": ctl_w3({k: m["levels"][k]["C"] for k in C_EXPECTED}), "w4": ctl_w4(rd(a.tests_out)), "w5": ctl_w5(m["grid_all_pass"], m["strategic_a"], m["strategic_b"])}
    payload = {"levels": m["levels"], "levels_per_seed": m["levels_per_seed"], "pairs": m["pairs"], "families": m["families"], "strategic": m["strategic"], "family_note": FAM_REASON,
               "unsafe_rate": {nm: p["U"] for nm, p in sorted(m["pairs"].items())},
               "falling_behind_against_RLc": {x: m["pairs"][x + "-VS-RLc"]["F"] for x in ARMS},
               "differences_against_KD5": {nm: {"C": p["C"], "U": p["U"], "F": p["F"]} for nm, p in sorted(m["pairs"].items()) if nm.endswith("-VS-KD5")},
               "rollout_shares": {k: {"share_of_rule_steps_with_a_lead": v["share_of_rule_steps_with_a_lead"], "share_of_lead_steps_with_nonzero_closing": v["share_of_lead_steps_with_nonzero_closing"],
                                      "n_rows": v["n_rows"], "n_lead_rows": v["n_lead_rows"], "n_lead_rows_closing_nonzero": v["n_lead_rows_closing_nonzero"]} for k, v in c2["per_run"].items()}}
    return assemble_controls(controls, payload)


def print_arm_sections(payload):
    L = []
    for x in ARMS:
        row = [x]
        for ref in ("RR", "RLc", "VD5c", "KD5"):
            p = payload["pairs"].get(x + "-VS-" + ref)
            if p:
                row.append("vs %s: C %s U %s F %s" % (ref, p["C"]["interval"], p["U"]["interval"], p["F"]["interval"]))
            if ref == "KD5":
                continue
            st = (payload.get("strategic", {}).get("pairs") or {}).get(x + "-VS-" + ref)
            fm = (payload.get("families") or {}).get(x + "-VS-" + ref) or {}
            row.append("families(%s) %s strategic %s" % (ref, json.dumps(fm, sort_keys=True), st["reading"] if st else None))
        L.append("   ".join(row))
    for nm in ("VD5c-VS-VRc", "VD5c-VS-VD5"):
        p = payload["pairs"].get(nm)
        if p:
            L.append("%s: C %s U %s F %s" % (nm, p["C"]["interval"], p["U"]["interval"], p["F"]["interval"]))
    L.append("rollout shares: " + json.dumps(payload.get("rollout_shares"), sort_keys=True))
    L.append("levels per model seed: " + json.dumps(payload.get("levels_per_seed"), sort_keys=True))
    return L


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage", choices=("controls", "names"), required=True)
    for n in ("pull", "code", "grid-dir", "w", "tree", "wrapper-file", "gap-record", "gap-dir", "add102-gap-dir", "eval-join", "dump-root", "wrapper-check", "wrapper-hashes", "tests-out", "controls"):
        ap.add_argument("--" + n, default="")
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
        print("ZZ READ-VISREP10K controls rc 0 controls_all_pass %s" % res["controls_all_pass"])
        return 0
    cj = json.load(io.open(a.controls, encoding="utf-8"))
    if cj.get("card") != CARD or cj.get("controls_all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != list(CONTROLS) \
            or any(v.get("PASS") is not True for v in cj.get("controls", {}).values()):
        print("REFUSED: the controls file is not all-pass: no name is written")
        print("ZZ READ-VISREP10K names rc 3 controls_all_pass False")
        return 3
    names = all_names(cj["payload"])
    res = {"card": CARD, "tier": TIER, "controls_file": os.path.abspath(a.controls), "controls_sha256": sha_file(a.controls), "names": names, "h_note": h_sensitivity(cj["payload"]),
           "values": cj["payload"]}
    io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1))
    for k in sorted(names):
        print(names[k])
    for line in print_arm_sections(cj["payload"]):
        print(line)
    print("ZZ READ-VISREP10K names rc 0 controls_all_pass True")
    return 0


def selftest():
    ok = True
    n = names_of("VD5cx", [-0.05, -0.01], [0.0, 0.004], [-0.03, 0.001], [-0.02, 0.03], 0.0954, 0.0)               # H = 0.0477
    ok &= n["VD5cx_CONTACT"].endswith("VD5cx-CUTS-CONTACT") and n["VD5cx_HALF"].endswith("WITHIN-HALF-THE-GAP") and n["VD5cx_VD5C"].endswith("CONTACT-NOT-SEPARATED-FROM-VD5c") and n["VD5cx_PRICE"].endswith("FALLING-BEHIND-NOT-SEPARATED")
    n = names_of("VRc", [0.001, 0.01], [0.05, 0.06], [0.001, 0.03], [0.01, 0.04], 0.0954, 0.0)
    ok &= n["VRc_CONTACT"].endswith("VRc-MORE-CONTACT") and n["VRc_HALF"].endswith("VRc-HALF-THE-GAP-NOT-REACHED") and n["VRc_VD5C"].endswith("VRc-MORE-CONTACT-THAN-VD5c") and n["VRc_PRICE"].endswith("FALLS-BEHIND-MORE")
    n = names_of("VD1c", [-0.04, 0.01], [0.01, 0.06], [-0.04, -0.01], [-0.04, -0.01], 0.0954, 0.0)
    ok &= n["VD1c_CONTACT"].endswith("CONTACT-NOT-SEPARATED-FROM-RR") and n["VD1c_HALF"].endswith("HALF-THE-GAP-NOT-DECIDED") and n["VD1c_VD5C"].endswith("LESS-CONTACT-THAN-VD5c") and n["VD1c_PRICE"].endswith("FALLS-BEHIND-LESS")
    ok &= replicate_name(PRE + "-VD5cx-CUTS-CONTACT", PRE + "-VD5cx-WITHIN-HALF-THE-GAP") == PRE + "-REPLICATED"
    ok &= replicate_name(PRE + "-VD5cx-CUTS-CONTACT", PRE + "-VD5cx-HALF-THE-GAP-NOT-REACHED") == PRE + "-NOT-REPLICATED"
    ok &= replicate_name(PRE + "-VD5cx-CONTACT-NOT-SEPARATED-FROM-RR", PRE + "-VD5cx-WITHIN-HALF-THE-GAP") == PRE + "-NOT-REPLICATED"
    ok &= replicate_name(PRE + "-VD5cx-CUTS-CONTACT", PRE + "-VD5cx-HALF-THE-GAP-NOT-DECIDED") == PRE + "-REPLICATION-NOT-DECIDED"
    ok &= gap_name([-0.05, -0.01]).endswith("CROP-GAP-CUTS-CONTACT") and slope_name([0.01, 0.05]).endswith("SLOPE-MORE-CONTACT") and slope_name([-0.01, 0.05]).endswith("SLOPE-NOT-SEPARATED")
    ok &= ctl_w3({"RR": 0.0954, "RLc": 0.0, "KD5": 0.0169, "VD5c": 0.0169, "VD5": 0.0769})["PASS"] and not ctl_w3({"RR": 0.0954, "RLc": 0.0, "KD5": 0.0169, "VD5c": 0.0169, "VD5": 0.0777})["PASS"]
    print("ZZ SELFTEST-READ-VISREP10K ok: %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
