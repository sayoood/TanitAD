#!/usr/bin/env python
"""read_visrule10k.py -- read of card PREREG_V8F_PROOF_ADD102 (VISRULE10K): the reader-fed lead rule (form RR) with the gap of the fixed-window head H5 (card ADD99, arm D) where the present reader marks a lead and the
head has an output, with and without the closing input computed as the label source computes it, on the model's own rollout (T1-kin/hb0.2, pre-roll of 30 steps (6 s)). Derived from croprule/read_croprule10k.py
(card ADD101): the engine of the model reader EC, R, HS, H is imported, not edited.
Stage controls: computes the grid read and the STRATEGIC family of the arm runs (VD5c, VD5, and the control run V0, two model seeds each) and the banked RR, RLc and KD5 inside this stage, runs v0, v1, v2, v3, v5 here
and reads v4 (--tests-out); writes CONTROLS_FILE. With every control PASS the values are written under `payload`, else under `withheld` and no name exists.
Stage names: reads CONTROLS_FILE, refuses it unless all controls pass, writes the names of card section 3 (four per arm and the card name from VD5c) and binds the result to the controls file by sha256.
Usage: python read_visrule10k.py --stage controls --pull PULL --code CODE --grid-dir DIR --w W --tree TREE --out CONTROLS.json --wrapper-check FILE --tests-out FILE
          --gap-record REC.json --gap-dir DIR --eval-join J.npz --dump-root DIR [--n-boot 2000]
       python read_visrule10k.py --stage names --controls CONTROLS.json --out RESULT.json     |   --selftest
Last line: ZZ READ-VISRULE10K <stage> rc <rc> controls_all_pass <bool>"""
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
import make_vis_gapfile as MVG              # noqa: E402

CARD, PRE = "PREREG_V8F_PROOF_ADD102", "VISRULE10K"
SEEDS = (0, 1)
ARMS = ("VD5c", "VD5")
RUN = {"VD5c": "RQ50KF5P30Hb20vVD5c", "VD5": "RQ50KF5P30Hb20vVD5", "V0": "RQ50KF5P30Hb20vV0"}
BANKED = {"RR": "RQ50KF5P30Hb20", "RLc": "RLcKF5P30Hb20"}
REFS = {"KD5": "RLcKF5P30Hb20pKD5"}                    # the same head's gap with the label marking the lead (card ADD101, banked)
RR_C_EXPECTED, RLC_C_EXPECTED, KD5_C_EXPECTED = 0.0954, 0.0, 0.0169
H_CARD = 0.0477                                        # half the gap as the card prints it (section 2); the names use the point values as the rule of card ADD101 does
N_LEAD = 650                                           # windows with a lead, the two model seeds together (card section 4, control v1; note N1)
TOL_V1 = 2.0 / N_LEAD                                  # two windows of the 650 with a lead, per pooled rate
TOL_V2_HEAD, TOL_V2_REF = 1e-3, 1e-4
TIER = "T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s)"
FAM_REASON = ("the rule changes the planned speed along the path through the distance to the lead; it does not move the path, so the lateral family is expected to be unmoved by it, "
              "and the tactical family is not read by the rule; the strategic family is read from the strategic grid of the engine")
CONTROLS = ("v0", "v1", "v2", "v3", "v4", "v5")


def names_of(arm, d_rr_c, d_rlc_c, d_kd5_c, d_rr_f, c_rr, c_rlc):
    """the four names of one arm (card section 3). d_*: interval [lo, hi] of C(X)-C(RR), C(X)-C(RLc), C(X)-C(KD5), F(X)-F(RR); c_rr, c_rlc: pooled point values of C of the references."""
    p = PRE + "-" + arm + "-"
    H_ = 0.5 * (c_rr - c_rlc)
    n1 = p + ("CUTS-CONTACT" if d_rr_c[1] < 0.0 else ("MORE-CONTACT" if d_rr_c[0] > 0.0 else "CONTACT-NOT-SEPARATED-FROM-RR"))
    n2 = p + ("WITHIN-HALF-THE-GAP" if d_rlc_c[1] <= H_ else ("HALF-THE-GAP-NOT-REACHED" if d_rlc_c[0] > H_ else "HALF-THE-GAP-NOT-DECIDED"))
    n3 = p + ("PRESENCE-COSTS-CONTACT" if d_kd5_c[0] > 0.0 else ("LESS-CONTACT-THAN-KD5" if d_kd5_c[1] < 0.0 else "CONTACT-NOT-SEPARATED-FROM-KD5"))
    n4 = p + ("FALLS-BEHIND-MORE" if d_rr_f[0] > 0.0 else ("FALLS-BEHIND-LESS" if d_rr_f[1] < 0.0 else "FALLING-BEHIND-NOT-SEPARATED"))
    return {arm + "_CONTACT": n1, arm + "_HALF": n2, arm + "_KD5": n3, arm + "_PRICE": n4}


def half_suffix(name):
    for s in ("WITHIN-HALF-THE-GAP", "HALF-THE-GAP-NOT-REACHED", "HALF-THE-GAP-NOT-DECIDED"):
        if name.endswith(s):
            return s
    raise ValueError(name)


def card_names(names):
    """the name of the card (section 3), from VD5c: VISRULE10K-<half>"""
    return {"CARD": PRE + "-" + half_suffix(names["VD5c_HALF"])}


def card_check(payload, got):
    """independent re-derivation of the card name from the interval of the payload (not from the other names); {"PASS", "expected"}"""
    pr = payload["pairs"]
    h = 0.5 * (payload["levels"]["RR"]["C"] - payload["levels"]["RLc"]["C"])
    lo, hi = pr["VD5c-VS-RLc"]["C"]["interval"]
    exp = {"CARD": PRE + ("-WITHIN-HALF-THE-GAP" if hi <= h else ("-HALF-THE-GAP-NOT-REACHED" if lo > h else "-HALF-THE-GAP-NOT-DECIDED"))}
    return {"PASS": all(got.get(k) == v for k, v in exp.items()), "expected": exp}


def h_sensitivity(payload):
    """notes, no name: the half-the-gap name of VD5c and VD5 again with H = 0.0477 as the card prints it; `differs` is true when that name is not the one made with the point values"""
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
        out.update(names_of(a, pr[a + "-VS-RR"]["C"]["interval"], pr[a + "-VS-RLc"]["C"]["interval"], pr[a + "-VS-KD5"]["C"]["interval"], pr[a + "-VS-RR"]["F"]["interval"],
                            payload["levels"]["RR"]["C"], payload["levels"]["RLc"]["C"]))
    out.update(card_names(out))
    chk = card_check(payload, out)
    if not chk["PASS"]:
        raise SystemExit("REFUSED: the name of the card %s is not the one the interval gives (%s)" % ({k: out[k] for k in chk["expected"]}, chk["expected"]))
    return out


def assemble_controls(controls, payload):
    ok = bool(controls) and all(isinstance(v, dict) and v.get("PASS") is True for v in controls.values())
    res = {"card": CARD, "tier": TIER, "controls": controls, "controls_all_pass": ok}
    res["payload" if ok else "withheld"] = payload
    return res


# ============================================================================================ controls
def sha_file(p):
    return MVG.sha_file(p)


def ctl_v0(rec, gap_dir, eval_join_sha=None):
    """the head's dumps are on the frames of the evaluation join (card), plus the build check of the gap files, plus: the files on disk have the hashes of the record and the record is for this evaluation join."""
    if not isinstance(rec, dict):
        return {"PASS": False, "failures": ["no gap-file record"]}
    c0, cb = MVG.v0_check(rec), MVG.build_check(rec)
    bad = list(c0["failures"]) + list(cb["failures"])
    for nm in MVG.file_names():
        p = os.path.join(gap_dir or "", nm + ".npz")
        if not os.path.isfile(p) or sha_file(p) != (rec.get("files", {}).get(nm) or {}).get("file_sha256"):
            bad.append("%s: file missing or its sha256 is not the record's" % nm)
    if eval_join_sha and rec.get("eval_join_sha256") != eval_join_sha:
        bad.append("the record is for another evaluation join")
    return {"PASS": not bad, "failures": bad, "record_v0": c0, "record_build": cb}


def ctl_v1(levels_per_seed):
    """the variant with the new arguments off (run V0) gives the rates of the banked RR rollouts: contact, unsafe rate and falling behind each within 2 windows of the 650 with a lead.
    Reading fixed by the lead in note N1 of the card before any rollout: the 650 windows are the two model seeds together, so each rate is the pooled rate (the mean of the two per-seed
    rates) and the tolerance is 2/650 as a rate; the per-seed differences are printed and carry no part of the control."""
    out = {"tolerance_per_rate": TOL_V1, "n_lead_windows_assumed": N_LEAD, "reading": "pooled over the two model seeds (note N1)", "rows": {}}
    ok = True
    for c in ("C", "U", "F"):
        a, b = levels_per_seed["V0"][c], levels_per_seed["RR"][c]
        d = [abs(float(x) - float(y)) for x, y in zip(a, b)]
        good = len(a) == len(SEEDS) and len(b) == len(SEEDS)
        dp = abs(float(np.mean(a)) - float(np.mean(b))) if good else float("nan")
        ok &= bool(good and dp <= TOL_V1 + 1e-9)
        out["rows"][c] = {"V0": list(a), "RR": list(b), "abs_diff_pooled": dp, "windows_of_650_pooled": dp * N_LEAD, "abs_diff_per_seed": d, "windows_of_650_per_seed": [x * N_LEAD for x in d],
                          "pooled_V0": float(np.mean(a)), "pooled_RR": float(np.mean(b))}
    out["PASS"] = bool(ok)
    return out


def rows_of(logs):
    """-> (ci, j, present, g_src, c_src) over the logs (tick log and later-look log) of one rollout."""
    cols = {k: [] for k in ("ci", "j", "present", "g_src", "c_src")}
    for z in logs:
        for k in cols:
            cols[k].append(np.asarray(z[k]))
    return (np.concatenate(cols["ci"]).astype(np.int64), np.concatenate(cols["j"]).astype(np.int64), np.concatenate(cols["present"]).astype(bool),
            np.concatenate(cols["g_src"]).astype(np.float64), np.concatenate(cols["c_src"]).astype(np.float64))


def lead_map(rows):
    """{(ci, j): g_src} of the rows with a lead; the number of keys seen with two values that differ by more than 1e-6 -> (map, n_inconsistent)"""
    ci, j, pres, g, _ = rows
    m, bad = {}, 0
    for i in np.flatnonzero(pres):
        k = (int(ci[i]), int(j[i]))
        if k in m and abs(m[k] - g[i]) > 1e-6:
            bad += 1
        m[k] = float(g[i])
    return m, bad


def v2_one(rows, ref_gap, v0_map, zero_closing, meta_expect=None, meta=None):
    """control v2 on one rollout of VD5c or VD5. rows from rows_of; ref_gap {(ci, frame): head gap or None} from the head's dump (a plain loop, make_vis_gapfile.second_routine); v0_map {(ci, j): gap}
    of the run V0 (the reader's own gap as RR feeds it). On every row with a lead: where the head has an output the gap the rule received equals the dumped gap within 1e-3 m; where not it equals the gap of
    V0 at the same (clip, frame) within 1e-4 m. With zero_closing (VD5) the closing input is 0 on every row of both logs."""
    ci, j, pres, g, c = rows
    n_head = n_bad_head = n_noout = n_ref_missing = n_bad_ref = 0
    for i in np.flatnonzero(pres):
        k = (int(ci[i]), int(j[i]))
        hg = ref_gap.get(k)
        if hg is not None:
            n_head += 1
            n_bad_head += int(abs(g[i] - hg) > TOL_V2_HEAD)
        else:
            n_noout += 1
            r = v0_map.get(k)
            if r is None:
                n_ref_missing += 1
            else:
                n_bad_ref += int(abs(g[i] - r) > TOL_V2_REF)
    n_c_nz = int((c != 0).sum())
    nz_lead = int((c[pres] != 0).sum())
    bad = []
    if int(pres.sum()) == 0 or n_head == 0:
        bad.append("no lead rows or none with a head output")
    if n_bad_head:
        bad.append("%d lead rows with a head output differ from the dumped gap by more than %g m" % (n_bad_head, TOL_V2_HEAD))
    if n_ref_missing or n_bad_ref:
        bad.append("lead rows without a head output: %d have no V0 row, %d differ from V0 by more than %g m" % (n_ref_missing, n_bad_ref, TOL_V2_REF))
    if zero_closing and n_c_nz:
        bad.append("the closing input is not 0 on %d rows" % n_c_nz)
    if meta_expect is not None:
        for k, v in meta_expect.items():
            if str((meta or {}).get(k)) != str(v):
                bad.append("rule log meta %s is %s, expected %s" % (k, (meta or {}).get(k), v))
    return {"PASS": not bad, "failures": bad, "n_rows": int(len(ci)), "n_lead_rows": int(pres.sum()), "n_lead_rows_with_head_output": n_head, "n_lead_rows_without_head_output": n_noout,
            "n_bad_head": n_bad_head, "n_ref_missing": n_ref_missing, "n_bad_ref": n_bad_ref, "n_rows_closing_nonzero": n_c_nz, "n_lead_rows_closing_nonzero": nz_lead}


def ctl_v2(per_run):
    ok = bool(per_run) and all(v["PASS"] for v in per_run.values())
    return {"PASS": ok, "per_run": per_run}


def ctl_v3(c_rr, c_rlc, c_kd5):
    """the reader recomputes C of RR, RLc and KD5 from the banked rollouts: 0.0954, 0.0000 and 0.0169 to four decimals"""
    ok = all(v is not None for v in (c_rr, c_rlc, c_kd5)) and round(float(c_rr), 4) == RR_C_EXPECTED and round(float(c_rlc), 4) == RLC_C_EXPECTED and round(float(c_kd5), 4) == KD5_C_EXPECTED
    return {"PASS": bool(ok), "C_RR": c_rr, "C_RLc": c_rlc, "C_KD5": c_kd5, "expected_RR": RR_C_EXPECTED, "expected_RLc": RLC_C_EXPECTED, "expected_KD5": KD5_C_EXPECTED}


def ctl_v4(tests_text):
    t = tests_text or ""
    m = re.findall(r"(\d+) passed", t)
    ok = bool(m) and int(m[-1]) > 0 and not re.search(r"\d+ (failed|error)", t)
    return {"PASS": ok, "passed": int(m[-1]) if m else None}


def ctl_v5(grid_all_pass, strat_a, strat_b):
    ok = grid_all_pass is True and strat_a is True and strat_b is True
    return {"PASS": bool(ok), "grid_all_pass": grid_all_pass, "strategic_stage_A_pass": strat_a, "strategic_stage_B_pass": strat_b}


def lead_shares(rows_by_run, J):
    """printed with no name: the share of logged rule steps with a lead in each run, and, from the rows of RR, the share of the frames with a valid label lead that the reader marks and the share of the
    marked frames with no label lead (label lead: lead_valid 1, finite gap 0 to 60 m, the test of LabelSource._ok)."""
    out = {"share_of_rule_steps_with_a_lead": {}}
    for k, rows in rows_by_run.items():
        out["share_of_rule_steps_with_a_lead"][k] = float(rows[2].mean()) if len(rows[2]) else None
    ci, j, pres = rows_by_run["RR"][0], rows_by_run["RR"][1], rows_by_run["RR"][2]
    off, n = np.asarray(J["off"], np.int64), np.asarray(J["n"], np.int64)
    vm = MVG._valid_mask(J)
    inr = (j >= 0) & (j < n[ci])
    lab = np.zeros(len(ci), bool)
    lab[inr] = vm[off[ci[inr]] + j[inr]]
    out["share_of_valid_label_lead_frames_the_reader_marks"] = float((lab & pres).sum() / lab.sum()) if lab.sum() else None
    out["share_of_marked_frames_with_no_label_lead"] = float((pres & ~lab).sum() / pres.sum()) if pres.sum() else None
    out["n_rows_RR"], out["n_label_lead_rows_RR"], out["n_marked_rows_RR"] = int(len(ci)), int(lab.sum()), int(pres.sum())
    return out


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


def v2_from_files(a, rec):
    J = MVG.load_npz(a.eval_join)
    refs = {}
    for m in SEEDS:
        D = MVG.load_npz(os.path.join(a.dump_root, MVG.DUMP_DIR, "reader_s%d_score.npz" % m))
        refs[m] = MVG.second_routine(J, D)
    per, rows_all = {}, {}
    for m in SEEDS:
        v0_logs = load_logs(a.pull, "RQ50KF5P30Hb20vV0", m)
        v0_rows = rows_of(v0_logs)
        v0_map, v0_incons = lead_map(v0_rows)
        rows_all["V0_s%d" % m] = v0_rows
        for arm in ARMS:
            logs = load_logs(a.pull, RUN[arm], m)
            rows = rows_of(logs)
            rows_all["%s_s%d" % (arm, m)] = rows
            exp = {"vis_gap_file_sha256": ((rec or {}).get("files", {}).get("VGAP_s%d" % m) or {}).get("file_sha256"), "vis_closing": 1 if arm == "VD5c" else 0}
            r = v2_one(rows, refs[m], v0_map, zero_closing=(arm == "VD5"), meta_expect=exp, meta=meta_of(logs[0]))
            r["n_v0_keys_with_two_values"] = v0_incons
            per["%s_s%d" % (arm, m)] = r
    rows_all["rr_source"] = "banked RR rule logs"
    for m in SEEDS:                                    # printed shares only: no control reads the banked RR logs; without them the control run V0 (the RR form, the same reader) stands in, and the payload says so
        try:
            rows_all["RR_s%d" % m] = rows_of(load_logs(a.pull, BANKED["RR"], m))
        except SystemExit:
            rows_all["RR_s%d" % m] = rows_all["V0_s%d" % m]
            rows_all["rr_source"] = "run V0 (the banked RR rule logs are not in the pull folder)"
    return ctl_v2(per), rows_all, J


def measure(a, E, log):
    """the pod part: grid read (levels, pairs with intervals, families) of the arm runs, the control run V0 and the banked RR, RLc, KD5; then the strategic grid. -> dict"""
    H, Q3, RH = E
    g30 = R.jl(os.path.join(a.grid_dir, "grid_p30.json"))
    J = H.load_join()
    keys = R.grid_keys(g30, [str(c) for c in J["clip"]])
    runs = dict(RUN)
    runs.update(BANKED)
    runs.update(REFS)
    for k, r in runs.items():
        R.ALL_RUNS.setdefault(r, ("RLc" if k in ("RLc", "KD5") else "RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    R.ALL_RUNS.setdefault(EC.SETTINGS["AGE"]["runs"]["RR"], ("RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    pairs_def = {}
    for x in ARMS:
        pairs_def[x + "-VS-RR"] = (RUN[x], BANKED["RR"])
        pairs_def[x + "-VS-RLc"] = (RUN[x], BANKED["RLc"])
        pairs_def[x + "-VS-KD5"] = (RUN[x], REFS["KD5"])
    pairs_def["V0-VS-RR"] = (RUN["V0"], BANKED["RR"])
    spread = "R0Hb20"
    run_list = [spread] + sorted(set(runs.values()))
    G = R.read_grid(a, E, "30", keys, run_list, spread, pairs_def, {}, log)
    lv = {k: {c: float(np.mean(G["levels"][r][c])) for c in ("C", "U", "F")} for k, r in runs.items()}
    lv_seed = {k: {c: [float(v) for v in np.asarray(G["levels"][r][c]).reshape(-1)[:2]] for c in ("C", "U", "F")} for k, r in runs.items()}
    pairs = {nm: R.ucf(p["headway"]) for nm, p in G["pairs"].items()}
    fams = {nm: {f_: R.reading(R.cells_of(p["family"]), f_) for f_ in R.NAMED} for nm, p in G["pairs"].items()}
    # the STRATEGIC family, as croprule/read_croprule10k.py reads it (engine strategic_grid with its own controls, then the pair read at tick t0 + 20 on the arcs of 30, 50 and 80 m)
    p_of = {"R0": 0}
    p_of.update({r: 30 for r in runs.values()})
    SR, cells_s, eid_s, S = R.strategic_grid(a, keys, sorted(p_of), p_of, log, {})
    sa, sb = bool(SR.get("stage_A_pass")), bool(SR.get("stage_B_pass"))
    strat = {"controls": dict({k: v for k, v in SR["controls"].items() if k != "s3"}, s3=SR["controls"]["s3"].get("pass")), "pairs": {}}
    if sa and sb:
        for x in ARMS:
            for ref in ("RR", "RLc"):
                pr = S.pair_read(cells_s, RUN[x], BANKED[ref], eid_s, ticks=(2,))
                pos = pr["tick2"]["pos"]
                strat["pairs"][x + "-VS-" + ref] = {"A": RUN[x], "B": BANKED[ref], "pos_30_50_80m_tick_t0_plus_20": [{"arc_m": arc, "mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "verdict": b["verdict"],
                                                                                                                  "n_per_seed": b.get("n_per_seed")} for arc, b in zip(S.ARCS, pos)],
                                                    "reading": R.family_of([c["verdict"] for c in pos])}
    return {"levels": lv, "levels_per_seed": lv_seed, "pairs": pairs, "families": fams, "grid_all_pass": G["controls"].get("ALL_PASS") is True, "strategic": strat, "strategic_a": sa, "strategic_b": sb}


def run_controls(a):
    log = lambda s: print(s, flush=True)                                                       # noqa: E731
    E = R.engine(a)
    m = measure(a, E, log)
    rec = R.jl(a.gap_record)
    ejs = sha_file(a.eval_join) if a.eval_join and os.path.isfile(a.eval_join) else None
    rd = lambda p: io.open(p, encoding="utf-8").read() if p and os.path.isfile(p) else ""      # noqa: E731
    c2, rows_all, J = v2_from_files(a, rec)
    controls = {"v0": ctl_v0(rec, a.gap_dir, ejs), "v1": ctl_v1(m["levels_per_seed"]), "v2": c2,
                "v3": ctl_v3(m["levels"]["RR"]["C"], m["levels"]["RLc"]["C"], m["levels"]["KD5"]["C"]),
                "v4": ctl_v4(rd(a.tests_out)), "v5": ctl_v5(m["grid_all_pass"], m["strategic_a"], m["strategic_b"])}
    wc = ctl_wrapper_note(rd(a.wrapper_check), a.wrapper_sha)
    by_run = {}
    for k in ("V0", "VD5c", "VD5", "RR"):
        parts = [rows_all["%s_s%d" % (k, s)] for s in SEEDS]
        by_run[k] = tuple(np.concatenate([p[i] for p in parts]) for i in range(5))
    payload = {"levels": m["levels"], "levels_per_seed": m["levels_per_seed"], "pairs": m["pairs"], "families": m["families"], "strategic": m["strategic"], "family_note": FAM_REASON,
               "unsafe_rate": {nm: p["U"] for nm, p in sorted(m["pairs"].items())}, "differences_against_KD5": {nm: {"C": p["C"], "U": p["U"], "F": p["F"]} for nm, p in sorted(m["pairs"].items()) if nm.endswith("-VS-KD5")},
               "lead_shares": dict(lead_shares(by_run, J), rr_rows_from=rows_all["rr_source"]), "wrapper_check_note": wc,
               "closing_nonzero": {k: {"n_rows": v["n_rows"], "n_lead_rows": v["n_lead_rows"], "n_lead_rows_closing_nonzero": v["n_lead_rows_closing_nonzero"]} for k, v in c2["per_run"].items()}}
    return assemble_controls(controls, payload)


def ctl_wrapper_note(check_text, wrapper_sha16):
    """not a control of the card (v4 is the kit's tests): records the last line of the pod check of the wrapper and whether the wrapper hash the queue used is in it (printed, no name)"""
    lines = [x for x in (check_text or "").splitlines() if x.strip()]
    return {"pod_check_last_line": lines[-1].strip() if lines else None, "wrapper_sha16": wrapper_sha16 or None}


def print_arm_sections(payload):
    L = []
    for x in ARMS:
        row = [x]
        for ref in ("RR", "RLc", "KD5"):
            p = payload["pairs"].get(x + "-VS-" + ref)
            if p:
                row.append("vs %s: C %s U %s F %s" % (ref, p["C"]["interval"], p["U"]["interval"], p["F"]["interval"]))
            if ref == "KD5":
                continue
            st = (payload.get("strategic", {}).get("pairs") or {}).get(x + "-VS-" + ref)
            fm = (payload.get("families") or {}).get(x + "-VS-" + ref) or {}
            row.append("families(%s) %s strategic %s" % (ref, json.dumps(fm, sort_keys=True), st["reading"] if st else None))
        L.append("   ".join(row))
    L.append("lead shares: " + json.dumps(payload.get("lead_shares"), sort_keys=True))
    L.append("levels per model seed: " + json.dumps(payload.get("levels_per_seed"), sort_keys=True))
    return L


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage", choices=("controls", "names"), required=True)
    for n in ("pull", "code", "grid-dir", "w", "tree", "wrapper-sha", "gap-record", "gap-dir", "eval-join", "dump-root", "wrapper-check", "tests-out", "controls"):
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
        print("ZZ READ-VISRULE10K controls rc 0 controls_all_pass %s" % res["controls_all_pass"])
        return 0
    cj = json.load(io.open(a.controls, encoding="utf-8"))
    if cj.get("card") != CARD or cj.get("controls_all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != list(CONTROLS) \
            or any(v.get("PASS") is not True for v in cj.get("controls", {}).values()):
        print("REFUSED: the controls file is not all-pass: no name is written")
        print("ZZ READ-VISRULE10K names rc 3 controls_all_pass False")
        return 3
    names = all_names(cj["payload"])
    res = {"card": CARD, "tier": TIER, "controls_file": os.path.abspath(a.controls), "controls_sha256": sha_file(a.controls), "names": names, "h_note": h_sensitivity(cj["payload"]),
           "values": cj["payload"]}
    io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1))
    for k in sorted(names):
        print(names[k])
    for line in print_arm_sections(cj["payload"]):
        print(line)
    print("ZZ READ-VISRULE10K names rc 0 controls_all_pass True")
    return 0


def selftest():
    ok = True
    n = names_of("VD5c", [-0.05, -0.01], [0.0, 0.004], [-0.03, 0.001], [-0.02, 0.03], 0.0954, 0.0)               # H = 0.0477
    ok &= n["VD5c_CONTACT"].endswith("VD5c-CUTS-CONTACT") and n["VD5c_HALF"].endswith("WITHIN-HALF-THE-GAP") and n["VD5c_KD5"].endswith("CONTACT-NOT-SEPARATED-FROM-KD5") and n["VD5c_PRICE"].endswith("FALLING-BEHIND-NOT-SEPARATED")
    n = names_of("VD5", [0.001, 0.01], [0.05, 0.06], [0.001, 0.03], [0.01, 0.04], 0.0954, 0.0)
    ok &= n["VD5_CONTACT"].endswith("VD5-MORE-CONTACT") and n["VD5_HALF"].endswith("VD5-HALF-THE-GAP-NOT-REACHED") and n["VD5_KD5"].endswith("VD5-PRESENCE-COSTS-CONTACT") and n["VD5_PRICE"].endswith("FALLS-BEHIND-MORE")
    n = names_of("VD5c", [-0.04, 0.01], [0.01, 0.06], [-0.04, -0.01], [-0.04, -0.01], 0.0954, 0.0)
    ok &= n["VD5c_CONTACT"].endswith("CONTACT-NOT-SEPARATED-FROM-RR") and n["VD5c_HALF"].endswith("HALF-THE-GAP-NOT-DECIDED") and n["VD5c_KD5"].endswith("LESS-CONTACT-THAN-KD5") and n["VD5c_PRICE"].endswith("FALLS-BEHIND-LESS")
    c = card_names({"VD5c_HALF": PRE + "-VD5c-WITHIN-HALF-THE-GAP"})
    ok &= c == {"CARD": PRE + "-WITHIN-HALF-THE-GAP"}
    v1 = ctl_v1({"V0": {"C": [0.0092, 0.0123], "U": [0.05, 0.05], "F": [0.1, 0.1]}, "RR": {"C": [0.0092, 0.0123], "U": [0.05, 0.05], "F": [0.1, 0.1]}})
    ok &= v1["PASS"]
    v1b = ctl_v1({"V0": {"C": [0.0092 + 3.0 / 325, 0.0123], "U": [0.05, 0.05], "F": [0.1, 0.1]}, "RR": {"C": [0.0092, 0.0123], "U": [0.05, 0.05], "F": [0.1, 0.1]}})   # three windows of the 650 (note N1: pooled)
    ok &= not v1b["PASS"]
    v1c = ctl_v1({"V0": {"C": [0.0092 + 2.0 / 325, 0.0123], "U": [0.05, 0.05], "F": [0.1, 0.1]}, "RR": {"C": [0.0092, 0.0123], "U": [0.05, 0.05], "F": [0.1, 0.1]}})   # two windows of the 650
    ok &= v1c["PASS"]
    ok &= ctl_v3(0.0954, 0.0, 0.0169)["PASS"] and not ctl_v3(0.0954, 0.0, 0.0177)["PASS"]
    print("ZZ SELFTEST-READ-VISRULE10K ok: %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
