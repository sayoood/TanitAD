#!/usr/bin/env python
"""read_visseed10k.py -- read of card PREREG_V8F_PROOF_ADD104 (VISSEED10K): the reader-fed lead rule (form RR) of the wrapper variant of card ADD102 (28bc31f21ecb4ccb, unchanged) with the slope closing input, fed the gap of
the head on five frames of the NEW fit seeds 2, 3 and 4 (arms F2, F3, F4), on the model's own rollout (T1-kin/hb0.2, pre-roll of 30 steps (6 s)), two model seeds.
Derived from visrep/read_visrep10k.py (card ADD103): written from it (functions kept or adapted; the engine of the model reader EC, R, HS, H is imported, not edited).
Stage controls: computes the grid read and the STRATEGIC family of the six arm runs and the banked RR, RLc, VD5c, VD5cx inside this stage, runs x0 to x3, x4 and x6 here and reads x5 (--tests-out); writes CONTROLS_FILE.
With every control PASS the values are written under `payload`, else under `withheld` and no name exists.
Stage names: reads CONTROLS_FILE, refuses it unless all controls pass, writes the names of card section 3 (four per arm and the card name) and binds the result to the controls file by sha256.
Usage: python read_visseed10k.py --stage controls --pull PULL --code CODE --grid-dir DIR --w W --tree TREE --out CONTROLS.json --wrapper-file F --wrapper-check FILE --wrapper-hashes FILE --tests-out FILE
          --gap-record REC.json --gap-dir DIR --fit-root FITROOT --add99-fits DIR --eval-list FILE --eval-join J.npz [--n-boot 2000]
       python read_visseed10k.py --stage names --controls CONTROLS.json --out RESULT.json     |   --selftest
Last line: ZZ READ-VISSEED10K <stage> rc <rc> controls_all_pass <bool>"""
import argparse
import copy
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
import make_seed_gapfile as MSG             # noqa: E402

CARD, PRE = "PREREG_V8F_PROOF_ADD104", "VISSEED10K"
SEEDS = (0, 1)                                                                 # model seeds
FITS = (2, 3, 4)                                                               # the new fit seeds of the head; arm F<S> reads the gap file of fit seed S
ARMS = tuple("F%d" % s for s in FITS)
RUN = {"F%d" % s: "RQ50KF5P30Hb20fF%d" % s for s in FITS}
B102 = {"VD5c": "RQ50KF5P30Hb20vVD5c"}                                          # banked arm of card ADD102 (fit seed k at model seed k)
B103 = {"VD5cx": "RQ50KF5P30Hb20wVD5cx"}                                        # banked arm of card ADD103 (fit seed 1 - k at model seed k)
V0_RUN = "RQ50KF5P30Hb20vV0"                                                   # banked control run of card ADD102: the reader's own gap as RR feeds it, logged at every lead step
BANKED = {"RR": "RQ50KF5P30Hb20", "RLc": "RLcKF5P30Hb20"}
C_EXPECTED = {"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0169, "VD5cx": 0.0200}
H_CARD = 0.0477
TOL_W2 = 1e-3
WRAPPER_SHA = "28bc31f21ecb4ccb"
CHECK_LAST = "ZZ CHECK-WRAPPER-V PASS"
DUMP5 = "D_H5"
TIER = "T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s)"
FAM_REASON = ("the rule changes the planned speed along the path through the distance to the lead; it does not move the path, so the lateral family is expected to be unmoved by it, "
              "and the tactical family is not read by the rule; the strategic family is read from the strategic grid of the engine")
CONTROLS = ("x0", "x1", "x2", "x3", "x4", "x5", "x6")
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


def holds_name(names):
    """card name by the count rule (card section 3, last name paragraph), from the contact and half names of the three arms: HOLDS when all three are CUTS-CONTACT, at least two are WITHIN-HALF-THE-GAP and none is
    HALF-THE-GAP-NOT-REACHED; DOES-NOT-HOLD when at least one arm is not CUTS-CONTACT or at least one is HALF-THE-GAP-NOT-REACHED; else NOT-DECIDED."""
    cuts = [names[a + "_CONTACT"].endswith("-" + a + "-CUTS-CONTACT") for a in ARMS]
    within = [names[a + "_HALF"].endswith("-" + a + "-WITHIN-HALF-THE-GAP") for a in ARMS]
    notreached = [names[a + "_HALF"].endswith("-" + a + "-HALF-THE-GAP-NOT-REACHED") for a in ARMS]
    if all(cuts) and sum(within) >= 2 and not any(notreached):
        return PRE + "-HOLDS"
    if (not all(cuts)) or any(notreached):
        return PRE + "-DOES-NOT-HOLD"
    return PRE + "-NOT-DECIDED"


def card_check(payload, got):
    """independent re-derivation of the card name from the intervals of the payload (not from the other names); {"PASS", "expected"}"""
    pr = payload["pairs"]
    h = 0.5 * (payload["levels"]["RR"]["C"] - payload["levels"]["RLc"]["C"])
    n_cut = n_within = n_notreached = 0
    for a in ARMS:
        n_cut += int(pr[a + "-VS-RR"]["C"]["interval"][1] < 0.0)
        lo, hi = pr[a + "-VS-RLc"]["C"]["interval"]
        n_within += int(hi <= h)
        n_notreached += int(lo > h)
    if n_cut == len(ARMS) and n_within >= 2 and n_notreached == 0:
        exp = "-HOLDS"
    elif n_cut < len(ARMS) or n_notreached > 0:
        exp = "-DOES-NOT-HOLD"
    else:
        exp = "-NOT-DECIDED"
    return {"PASS": got.get("CARD") == PRE + exp, "expected": {"CARD": PRE + exp}}


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
    out["CARD"] = holds_name(out)
    chk = card_check(payload, out)
    if not chk["PASS"]:
        raise SystemExit("REFUSED: the card name %s is not the one the intervals give (%s)" % (out["CARD"], chk["expected"]))
    return out


def contact_table(levels_seed):
    """printed with no name (card section 3): the contact rate of every fit of the head on record per model seed. levels_seed {run key: {"C": [model seed 0, model seed 1], ...}} with the keys VD5c, VD5cx, F2, F3, F4.
    Fit seed k is read by VD5c at model seed k (card ADD102); fit seed 1-k by VD5cx at model seed k (card ADD103); fit seeds 2, 3, 4 by F2, F3, F4 at both model seeds. -> per fit and per model seed, the pooled value
    (mean of the two model seeds), and the mean and range over the five fits of the pooled value and of each model seed."""
    c = {k: [float(x) for x in v["C"]] for k, v in levels_seed.items()}
    per = {"fit0": [c["VD5c"][0], c["VD5cx"][1]], "fit1": [c["VD5cx"][0], c["VD5c"][1]]}
    for s in FITS:
        per["fit%d" % s] = [c["F%d" % s][0], c["F%d" % s][1]]
    out = {"per_fit": {k: {"model_seed_0": v[0], "model_seed_1": v[1], "pooled": 0.5 * (v[0] + v[1])} for k, v in sorted(per.items())}}
    for nm, sel in (("pooled", lambda v: 0.5 * (v[0] + v[1])), ("model_seed_0", lambda v: v[0]), ("model_seed_1", lambda v: v[1])):
        xs = [sel(v) for _, v in sorted(per.items())]
        out["over_five_fits_" + nm] = {"mean": sum(xs) / len(xs), "min": min(xs), "max": max(xs)}
    return out


def assemble_controls(controls, payload):
    ok = bool(controls) and all(isinstance(v, dict) and v.get("PASS") is True for v in controls.values())
    res = {"card": CARD, "tier": TIER, "controls": controls, "controls_all_pass": ok}
    res["payload" if ok else "withheld"] = payload
    return res


# ============================================================================================ controls
def sha_file(p):
    return MSG.sha_file(p)


def ctl_x0(rec, fits_now):
    """x0 (the fits): the record of make_seed_gapfile.py has x0 PASS, the facts of the fit records and head files read again from disk now pass x0 (MSG.x0_check), and are the files the record saw."""
    if not isinstance(rec, dict):
        return {"PASS": False, "failures": ["no gap-file record"]}
    now = MSG.x0_check(fits_now)
    bad = ["now: " + x for x in now["failures"]]
    if (rec.get("x0") or {}).get("PASS") is not True:
        bad.append("the record's own x0 is not PASS")
    for grp in ("new", "old"):
        for k, f in (fits_now.get(grp) or {}).items():
            seen = (((rec.get("fits") or {}).get(grp)) or {}).get(k) or {}
            if seen.get("record_sha256") != f.get("record_sha256") or seen.get("head_disk_sha256") != f.get("head_disk_sha256"):
                bad.append("%s %s: the fit record or head file is not the one the gap-file record saw" % (grp, k))
    return {"PASS": not bad, "failures": bad, "now": now}


def ctl_x1(rec, gap_dir, eval_join_sha, fits_now, meta_now):
    """x1 (the gap files): v0 (the dumps are on the frames of the evaluation join), the second-routine build check and the dump-meta clause (each meta names the sha256 of the head file of its fit) read from the record and again
    from disk now; the gap files on disk have the hashes of the record, the record is for this evaluation join and its own x1 is PASS."""
    if not isinstance(rec, dict):
        return {"PASS": False, "failures": ["no gap-file record"]}
    c0, cb, cm = MSG.v0_check(rec), MSG.build_check(rec), MSG.meta_check(rec)
    bad = list(c0["failures"]) + list(cb["failures"]) + list(cm["failures"])
    for nm in MSG.file_names():
        p = os.path.join(gap_dir or "", nm + ".npz")
        if not os.path.isfile(p) or sha_file(p) != (rec.get("files", {}).get(nm) or {}).get("file_sha256"):
            bad.append("%s: file missing or its sha256 is not the record's" % nm)
    if eval_join_sha and rec.get("eval_join_sha256") != eval_join_sha:
        bad.append("the record is for another evaluation join")
    r2 = copy.deepcopy(rec)
    r2["fits"] = fits_now
    for s in MSG.SEEDS:
        nm = "VGAPF_s%d" % s
        try:
            r2["dumps"][nm]["dump_meta"] = meta_now[s]["dump_meta"]
            dp = meta_now[s]["dump_file"]
            if not os.path.isfile(dp) or sha_file(dp) != rec["files"][nm]["dump_sha256"]:
                bad.append("%s: the dump file on disk is not the one the record read" % nm)
        except (KeyError, TypeError):
            bad.append("%s: the dump meta or dump file cannot be read now" % nm)
    cn = MSG.meta_check(r2)
    bad += ["now: " + x for x in cn["failures"]]
    if (rec.get("x1") or {}).get("PASS") is not True:
        bad.append("the record's own x1 is not PASS")
    return {"PASS": not bad, "failures": bad, "record_v0": c0, "record_build": cb, "record_meta": cm, "meta_now": cn}


WRAP_RE = re.compile(r"wrapper sha256 ([0-9a-f]{64})")


def ctl_x2(wrapper_sha16, check_text, hashes_text, run_shas):
    """x2 (the wrapper): the wrapper file is 28bc31f21ecb4ccb, the pod check record of card ADD102 names it with a pass (last line of CHECK_WRAPPER_V_DONE.txt and the hash in CHECK_WRAPPER_V.hashes), and every run
    text of the six rollouts names that wrapper hash (six expected)."""
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
    """control x3 on one rollout. rows from rows_of; ref_gap {(ci, frame): head gap or None} from the head's dump of the arm's fit seed (a plain loop, make_seed_gapfile.second_routine); v0_multi {(ci, j): [gap, ...]} of the
    control run V0 of card ADD102. On every row with a lead: where the head has an output the gap the rule received equals the dumped gap within 1e-3 m; where it has none it equals a gap of V0 at the same clip and frame
    within 1e-3 m, any of them. The closing input is non-zero on a positive share of the lead rows. The rule log meta names the gap file hash and the closing switch of the arm."""
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


def ctl_x3(per_run):
    ok = bool(per_run) and all(v["PASS"] for v in per_run.values())
    return {"PASS": ok, "per_run": per_run}


def ctl_x4(levels):
    """the reader recomputes C of RR, RLc, VD5c and VD5cx from the banked rollouts: 0.0954, 0.0000, 0.0169 and 0.0200 to four decimals"""
    got = {k: levels.get(k) for k in C_EXPECTED}
    ok = all(v is not None and round(float(v), 4) == C_EXPECTED[k] for k, v in got.items())
    return {"PASS": bool(ok), "C": got, "expected": dict(C_EXPECTED)}


def ctl_x5(tests_text):
    t = tests_text or ""
    m = re.findall(r"(\d+) passed", t)
    ok = bool(m) and int(m[-1]) > 0 and not re.search(r"\d+ (failed|error)", t)
    return {"PASS": ok, "passed": int(m[-1]) if m else None}


def ctl_x6(grid_all_pass, strat_a, strat_b):
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


def x3_from_files(a, rec):
    """-> control x3. Arm F<S> at model seed m: the gap file of fit seed S (VGAPF_s<S>), the dump of fit seed S in <fit-root>/dump/D_H5, meta sha256 = the record's VGAPF_s<S>, closing switch 1; V0 of card ADD102 for the rows with no output."""
    J = MSG.load_npz(a.eval_join)
    refs = {s: MSG.second_routine(J, MSG.load_npz(os.path.join(a.fit_root, "dump", DUMP5, "reader_s%d_score.npz" % s))) for s in FITS}
    per = {}
    for m in SEEDS:
        v0_multi = lead_multi(rows_of(load_logs(a.pull, V0_RUN, m)))
        for s in FITS:
            logs = load_logs(a.pull, RUN["F%d" % s], m)
            sha = ((rec or {}).get("files", {}).get("VGAPF_s%d" % s) or {}).get("file_sha256")
            per["F%d_s%d" % (s, m)] = w2_one(rows_of(logs), refs[s], v0_multi, meta_expect={"vis_gap_file_sha256": sha, "vis_closing": 1}, meta=meta_of(logs[0]))
    return ctl_x3(per)


def measure(a, E, log):
    """the pod part: grid read (levels, pairs with intervals, families) of the six arm runs and the banked RR, RLc, VD5c, VD5cx; then the strategic grid. -> dict"""
    H, Q3, RH = E
    g30 = R.jl(os.path.join(a.grid_dir, "grid_p30.json"))
    J = H.load_join()
    keys = R.grid_keys(g30, [str(c) for c in J["clip"]])
    runs = dict(RUN)
    runs.update(BANKED)
    runs.update(B102)
    runs.update(B103)
    for k, r in runs.items():
        R.ALL_RUNS.setdefault(r, ("RLc" if k == "RLc" else "RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    R.ALL_RUNS.setdefault(EC.SETTINGS["AGE"]["runs"]["RR"], ("RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    pairs_def = {}
    for x in ARMS:
        pairs_def[x + "-VS-RR"] = (RUN[x], BANKED["RR"])
        pairs_def[x + "-VS-RLc"] = (RUN[x], BANKED["RLc"])
        pairs_def[x + "-VS-VD5c"] = (RUN[x], B102["VD5c"])
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


def fit_selection(rec):
    """the selection loss and the kept epoch of each new fit, from the fit records the gap-file record carries"""
    new = ((rec or {}).get("fits") or {}).get("new") or {}
    return {k: {"kept_epoch": v.get("kept_epoch"), "kept_sel_loss": v.get("kept_sel_loss")} for k, v in sorted(new.items())}


def run_controls(a):
    log = lambda s: print(s, flush=True)                                                       # noqa: E731
    E = R.engine(a)
    m = measure(a, E, log)
    rec = R.jl(a.gap_record)
    ejs = sha_file(a.eval_join) if a.eval_join and os.path.isfile(a.eval_join) else None
    rd = lambda p: io.open(p, encoding="utf-8").read() if p and os.path.isfile(p) else ""      # noqa: E731
    evl = a.eval_list or os.path.join(a.fit_root, "lists", "eval141.txt")
    fits_now, meta_now = MSG.collect(a.fit_root, a.add99_fits, evl, os.path.join(a.fit_root, "dump"))
    c3 = x3_from_files(a, rec)
    run_shas = {"%s_s%d" % (arm, s): run_file_sha(a.pull, RUN[arm], s) for arm in ARMS for s in SEEDS}
    wsha = sha_file(a.wrapper_file)[:16] if a.wrapper_file and os.path.isfile(a.wrapper_file) else None
    controls = {"x0": ctl_x0(rec, fits_now), "x1": ctl_x1(rec, a.gap_dir, ejs, fits_now, meta_now), "x2": ctl_x2(wsha, rd(a.wrapper_check), rd(a.wrapper_hashes), run_shas), "x3": c3,
                "x4": ctl_x4({k: m["levels"][k]["C"] for k in C_EXPECTED}), "x5": ctl_x5(rd(a.tests_out)), "x6": ctl_x6(m["grid_all_pass"], m["strategic_a"], m["strategic_b"])}
    payload = {"levels": m["levels"], "levels_per_seed": m["levels_per_seed"], "pairs": m["pairs"], "families": m["families"], "strategic": m["strategic"], "family_note": FAM_REASON,
               "contact_per_fit": contact_table({k: m["levels_per_seed"][k] for k in ("VD5c", "VD5cx") + ARMS}), "fit_selection": fit_selection(rec),
               "unsafe_rate": {nm: p["U"] for nm, p in sorted(m["pairs"].items())},
               "falling_behind_against_RLc": {x: m["pairs"][x + "-VS-RLc"]["F"] for x in ARMS},
               "rollout_shares": {k: {"share_of_rule_steps_with_a_lead": v["share_of_rule_steps_with_a_lead"], "share_of_lead_steps_with_nonzero_closing": v["share_of_lead_steps_with_nonzero_closing"],
                                      "n_rows": v["n_rows"], "n_lead_rows": v["n_lead_rows"], "n_lead_rows_closing_nonzero": v["n_lead_rows_closing_nonzero"]} for k, v in c3["per_run"].items()}}
    return assemble_controls(controls, payload)


def print_arm_sections(payload):
    L = []
    for x in ARMS:
        row = [x]
        for ref in ("RR", "RLc", "VD5c"):
            p = payload["pairs"].get(x + "-VS-" + ref)
            if p:
                row.append("vs %s: C %s U %s F %s" % (ref, p["C"]["interval"], p["U"]["interval"], p["F"]["interval"]))
            st = (payload.get("strategic", {}).get("pairs") or {}).get(x + "-VS-" + ref)
            fm = (payload.get("families") or {}).get(x + "-VS-" + ref) or {}
            row.append("families(%s) %s strategic %s" % (ref, json.dumps(fm, sort_keys=True), st["reading"] if st else None))
        L.append("   ".join(row))
    L.append("contact rate of every fit of the head on record: " + json.dumps(payload.get("contact_per_fit"), sort_keys=True))
    L.append("selection loss and kept epoch of the new fits: " + json.dumps(payload.get("fit_selection"), sort_keys=True))
    L.append("rollout shares: " + json.dumps(payload.get("rollout_shares"), sort_keys=True))
    L.append("levels per model seed: " + json.dumps(payload.get("levels_per_seed"), sort_keys=True))
    return L


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage", choices=("controls", "names"), required=True)
    for n in ("pull", "code", "grid-dir", "w", "tree", "wrapper-file", "gap-record", "gap-dir", "eval-join", "fit-root", "add99-fits", "eval-list", "wrapper-check", "wrapper-hashes", "tests-out", "controls"):
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
        print("ZZ READ-VISSEED10K controls rc 0 controls_all_pass %s" % res["controls_all_pass"])
        return 0
    cj = json.load(io.open(a.controls, encoding="utf-8"))
    if cj.get("card") != CARD or cj.get("controls_all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != sorted(CONTROLS) \
            or any(v.get("PASS") is not True for v in cj.get("controls", {}).values()):
        print("REFUSED: the controls file is not all-pass: no name is written")
        print("ZZ READ-VISSEED10K names rc 3 controls_all_pass False")
        return 3
    names = all_names(cj["payload"])
    res = {"card": CARD, "tier": TIER, "controls_file": os.path.abspath(a.controls), "controls_sha256": sha_file(a.controls), "names": names, "h_note": h_sensitivity(cj["payload"]),
           "values": cj["payload"]}
    io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1))
    for k in sorted(names):
        print(names[k])
    for line in print_arm_sections(cj["payload"]):
        print(line)
    print("ZZ READ-VISSEED10K names rc 0 controls_all_pass True")
    return 0


def selftest():
    ok = True
    n = names_of("F2", [-0.05, -0.01], [0.0, 0.004], [-0.03, 0.001], [-0.02, 0.03], 0.0954, 0.0)               # H = 0.0477
    ok &= n["F2_CONTACT"].endswith("F2-CUTS-CONTACT") and n["F2_HALF"].endswith("WITHIN-HALF-THE-GAP") and n["F2_VD5C"].endswith("CONTACT-NOT-SEPARATED-FROM-VD5c") and n["F2_PRICE"].endswith("FALLING-BEHIND-NOT-SEPARATED")
    n = names_of("F3", [0.001, 0.01], [0.05, 0.06], [0.001, 0.03], [0.01, 0.04], 0.0954, 0.0)
    ok &= n["F3_CONTACT"].endswith("F3-MORE-CONTACT") and n["F3_HALF"].endswith("F3-HALF-THE-GAP-NOT-REACHED") and n["F3_VD5C"].endswith("F3-MORE-CONTACT-THAN-VD5c") and n["F3_PRICE"].endswith("FALLS-BEHIND-MORE")
    cn = lambda c, h: dict({a + "_CONTACT": PRE + "-" + a + "-" + c[i] for i, a in enumerate(ARMS)}, **{a + "_HALF": PRE + "-" + a + "-" + h[i] for i, a in enumerate(ARMS)})      # noqa: E731
    W_, N_, D_ = "WITHIN-HALF-THE-GAP", "HALF-THE-GAP-NOT-REACHED", "HALF-THE-GAP-NOT-DECIDED"
    ok &= holds_name(cn(["CUTS-CONTACT"] * 3, [W_, W_, D_])) == PRE + "-HOLDS"
    ok &= holds_name(cn(["CUTS-CONTACT"] * 3, [W_, D_, D_])) == PRE + "-NOT-DECIDED"
    ok &= holds_name(cn(["CUTS-CONTACT"] * 3, [W_, W_, N_])) == PRE + "-DOES-NOT-HOLD"
    ok &= holds_name(cn(["CUTS-CONTACT", "CUTS-CONTACT", "CONTACT-NOT-SEPARATED-FROM-RR"], [W_, W_, W_])) == PRE + "-DOES-NOT-HOLD"
    ok &= ctl_x4({"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0169, "VD5cx": 0.0200})["PASS"] and not ctl_x4({"RR": 0.0954, "RLc": 0.0, "VD5c": 0.0169, "VD5cx": 0.0208})["PASS"]
    print("ZZ SELFTEST-READ-VISSEED10K ok: %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
