#!/usr/bin/env python
"""read_croprule10k.py -- read of card PREREG_V8F_PROOF_ADD101 (CROPRULE10K): what the pilot's gap of card ADD99 (bound arm and fixed windows, heads H5 and H1) is worth inside the lead rule,
on the model's own rollout (T1-kin/hb0.2, pre-roll of 30 steps (6 s)). Derived from smoothgap/read_smoothgap10k.py (the engine of the model reader EC, R, HS, H is imported, not edited).
Stage controls: computes the grid read and the STRATEGIC family of the eight arm runs (KB5 KB1 KD5 KD1, two model seeds each) and the banked RR, RLc and K100 inside this stage, runs q0, q1, q3, q5 here
and reads q2 (--wrapper-check) and q4 (--tests-out); writes CONTROLS_FILE. With every control PASS the values are written under `payload`, else under `withheld` and no name exists.
Stage names: reads CONTROLS_FILE, refuses it unless all controls pass, writes the names of card section 3 (four per arm and the two names of the card) and binds the result to the controls file by sha256.
Usage: python read_croprule10k.py --stage controls --pull PULL --code CODE --grid-dir DIR --w W --tree TREE --out CONTROLS.json --wrapper-sha HEX16 --joins-record REC.json
          --joins-dir DIR --eval-join J.npz --wrapper-check FILE --tests-out FILE [--n-boot 2000]
       python read_croprule10k.py --stage names --controls CONTROLS.json --out RESULT.json     |   --selftest
Last line: ZZ READ-CROPRULE10K <stage> rc <rc> controls_all_pass <bool>"""
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
import make_crop_joins as MCJ               # noqa: E402

CARD, PRE = "PREREG_V8F_PROOF_ADD101", "CROPRULE10K"
SEEDS = (0, 1)
ARMS = ("KB5", "KB1", "KD5", "KD1")
RUN = {a: "RLcKF5P30Hb20p" + a for a in ARMS}
BANKED = {"RR": "RQ50KF5P30Hb20", "RLc": "RLcKF5P30Hb20"}
GAINS = {"K100": "RLcKF5P30Hb20mK100"}                 # the present reader's gap on the same frames (arm K100 of card ADD96, banked)
RR_C_EXPECTED, RLC_C_EXPECTED, K100_C_EXPECTED = 0.0954, 0.0, 0.0477
H_CARD = 0.0477                                        # half the gap as the card prints it (section 2); the names use the point values as the rule of card ADD97 does
CARD_ARM = {"BOUND": "KB5", "FIXED": "KD5"}
TIER = "T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s)"
FAM_REASON = ("the rule changes the planned speed along the path through the distance to the lead; it does not move the path, so the lateral family is expected to be unmoved by it, "
              "and the tactical family is not read by the rule; the strategic family is read from the strategic grid of the engine")
CONTROLS = ("q0", "q1", "q2", "q3", "q4", "q5")


def names_of(arm, d_rr_c, d_rlc_c, d_rr_f, d_k100_c, c_rr, c_rlc):
    """the four names of one arm (card section 3). d_*: interval [lo, hi] of C(X)-C(RR), C(X)-C(RLc), F(X)-F(RR), C(X)-C(K100); c_rr, c_rlc: pooled point values of C of the references."""
    p = PRE + "-" + arm + "-"
    H_ = 0.5 * (c_rr - c_rlc)
    n1 = p + ("CUTS-CONTACT" if d_rr_c[1] < 0.0 else ("MORE-CONTACT" if d_rr_c[0] > 0.0 else "CONTACT-NOT-SEPARATED-FROM-RR"))
    n2 = p + ("WITHIN-HALF-THE-GAP" if d_rlc_c[1] <= H_ else ("HALF-THE-GAP-NOT-REACHED" if d_rlc_c[0] > H_ else "HALF-THE-GAP-NOT-DECIDED"))
    n3 = p + ("FALLS-BEHIND-MORE" if d_rr_f[0] > 0.0 else ("FALLS-BEHIND-LESS" if d_rr_f[1] < 0.0 else "FALLING-BEHIND-NOT-SEPARATED"))
    n4 = p + ("LESS-CONTACT-THAN-K100" if d_k100_c[1] < 0.0 else ("MORE-CONTACT-THAN-K100" if d_k100_c[0] > 0.0 else "CONTACT-NOT-SEPARATED-FROM-K100"))
    return {arm + "_CONTACT": n1, arm + "_HALF": n2, arm + "_PRICE": n3, arm + "_K100": n4}


def half_suffix(name):
    """the suffix of a half-the-gap name: WITHIN-HALF-THE-GAP, HALF-THE-GAP-NOT-REACHED or HALF-THE-GAP-NOT-DECIDED"""
    for s in ("WITHIN-HALF-THE-GAP", "HALF-THE-GAP-NOT-REACHED", "HALF-THE-GAP-NOT-DECIDED"):
        if name.endswith(s):
            return s
    raise ValueError(name)


def card_names(names):
    """the two names of the card (section 3): from KB5 CROPRULE10K-BOUND-<half>, from KD5 CROPRULE10K-FIXED-<half>"""
    return {"CARD_" + kind: PRE + "-" + kind + "-" + half_suffix(names[arm + "_HALF"]) for kind, arm in CARD_ARM.items()}


def card_check(payload, got):
    """independent re-derivation of the two names of the card from the intervals of the payload (not from the other names); {"PASS", "expected"}. got: {CARD_BOUND, CARD_FIXED}"""
    pr = payload["pairs"]
    h = 0.5 * (payload["levels"]["RR"]["C"] - payload["levels"]["RLc"]["C"])
    exp = {}
    for kind, arm in CARD_ARM.items():
        lo, hi = pr[arm + "-VS-RLc"]["C"]["interval"]
        exp["CARD_" + kind] = PRE + "-" + kind + ("-WITHIN-HALF-THE-GAP" if hi <= h else ("-HALF-THE-GAP-NOT-REACHED" if lo > h else "-HALF-THE-GAP-NOT-DECIDED"))
    return {"PASS": all(got.get(k) == v for k, v in exp.items()), "expected": exp}


def h_sensitivity(payload):
    """notes, no name: the half-the-gap name of KB5 and KD5 again with H = 0.0477 as the card prints it; `differs` is true when that name is not the one made with the point values"""
    pr = payload["pairs"]
    h = 0.5 * (payload["levels"]["RR"]["C"] - payload["levels"]["RLc"]["C"])
    out = {"H_point_values": h, "H_card": H_CARD, "differs": False}
    for arm in CARD_ARM.values():
        lo, hi = pr[arm + "-VS-RLc"]["C"]["interval"]
        f = lambda H: "WITHIN" if hi <= H else ("NOT-REACHED" if lo > H else "NOT-DECIDED")      # noqa: E731
        out[arm] = {"with_point_values": f(h), "with_H_card": f(H_CARD)}
        out["differs"] = bool(out["differs"] or f(h) != f(H_CARD))
    return out


def all_names(payload):
    out = {}
    pr = payload["pairs"]
    for a in ARMS:
        out.update(names_of(a, pr[a + "-VS-RR"]["C"]["interval"], pr[a + "-VS-RLc"]["C"]["interval"], pr[a + "-VS-RR"]["F"]["interval"], pr[a + "-VS-K100"]["C"]["interval"],
                            payload["levels"]["RR"]["C"], payload["levels"]["RLc"]["C"]))
    out.update(card_names(out))
    chk = card_check(payload, out)
    if not chk["PASS"]:
        raise SystemExit("REFUSED: the names of the card %s are not the ones the intervals give (%s)" % ({k: out[k] for k in chk["expected"]}, chk["expected"]))
    return out


def assemble_controls(controls, payload):
    ok = bool(controls) and all(isinstance(v, dict) and v.get("PASS") is True for v in controls.values())
    res = {"card": CARD, "tier": TIER, "controls": controls, "controls_all_pass": ok}
    res["payload" if ok else "withheld"] = payload
    return res


def ctl_q0(rec, eval_join_sha=None):
    c = MCJ.q0_check(rec) if isinstance(rec, dict) else {"PASS": False, "failures": ["no joins record"]}
    bad = list(c["failures"])
    if isinstance(rec, dict) and eval_join_sha and rec.get("eval_join_sha256") != eval_join_sha:
        bad.append("the record is for another evaluation join")
    return {"PASS": not bad, "failures": bad}


def ctl_q1(rec, joins_dir, eval_join_sha=None):
    """control q1 from the record, plus: the join files on disk have the hashes of the record, and the record is for this evaluation join."""
    if not isinstance(rec, dict):
        return {"PASS": False, "failures": ["no joins record"]}
    c = MCJ.q1_check(rec)
    bad = list(c["failures"])
    for nm in MCJ.join_names():
        p = os.path.join(joins_dir or "", nm + ".npz")
        if not os.path.isfile(p) or MCJ.sha_file(p) != (rec.get("arms", {}).get(nm) or {}).get("sha256"):
            bad.append("%s: file missing or its sha256 is not the record's" % nm)
    if eval_join_sha and rec.get("eval_join_sha256") != eval_join_sha:
        bad.append("the record is for another evaluation join")
    return {"PASS": not bad, "failures": bad, "record_q1": c}


def ctl_q2(check_text, hashes_text, wrapper_sha16):
    lines = [x for x in (check_text or "").splitlines() if x.strip()]
    ok = bool(lines and lines[-1].strip() == "ZZ CHECK-WRAPPER-J PASS" and wrapper_sha16 and wrapper_sha16 in (hashes_text or ""))
    return {"PASS": ok, "last_line": lines[-1].strip() if lines else None, "wrapper_sha16_in_check_record": bool(wrapper_sha16 and wrapper_sha16 in (hashes_text or ""))}


def ctl_q3(c_rr, c_rlc, c_k100):
    """the reader recomputes C of RR, RLc and K100 from the banked rollouts: 0.0954, 0.0000 and 0.0477 to four decimals (rows 157 and 158)"""
    ok = all(v is not None for v in (c_rr, c_rlc, c_k100)) and round(float(c_rr), 4) == RR_C_EXPECTED and round(float(c_rlc), 4) == RLC_C_EXPECTED and round(float(c_k100), 4) == K100_C_EXPECTED
    return {"PASS": bool(ok), "C_RR": c_rr, "C_RLc": c_rlc, "C_K100": c_k100, "expected_RR": RR_C_EXPECTED, "expected_RLc": RLC_C_EXPECTED, "expected_K100": K100_C_EXPECTED}


def ctl_q4(tests_text):
    t = tests_text or ""
    m = re.findall(r"(\d+) passed", t)
    ok = bool(m) and int(m[-1]) > 0 and not re.search(r"\d+ (failed|error)", t)
    return {"PASS": ok, "passed": int(m[-1]) if m else None}


def ctl_q5(grid_all_pass, strat_a, strat_b):
    ok = grid_all_pass is True and strat_a is True and strat_b is True
    return {"PASS": bool(ok), "grid_all_pass": grid_all_pass, "strategic_stage_A_pass": strat_a, "strategic_stage_B_pass": strat_b}


def share_rows(levels, pairs):
    """share of the contact gap C(RR)-C(RLc) that each arm removes, with the interval of C(X)-C(RR) over the point gap (reported, names nothing)."""
    gap = levels["RR"]["C"] - levels["RLc"]["C"]
    out = {}
    for a in ARMS:
        d = pairs.get(a + "-VS-RR")
        if d is None or gap == 0.0:
            out[a] = None
        else:
            lo, hi = d["C"]["interval"]
            out[a] = {"share": -d["C"]["pooled"] / gap, "interval": [-hi / gap, -lo / gap], "denominator": "point value of C(RR) minus C(RLc)"}
    return out


def sha_file(p):
    return MCJ.sha_file(p)


def measure(a, E, log):
    """the pod part: grid read (levels, pairs with intervals, families) of the eight arm runs and the banked RR, RLc, K100; then the strategic grid. -> dict"""
    H, Q3, RH = E
    g30 = R.jl(os.path.join(a.grid_dir, "grid_p30.json"))
    J = H.load_join()
    keys = R.grid_keys(g30, [str(c) for c in J["clip"]])
    runs = dict(RUN)
    runs.update(BANKED)
    runs.update(GAINS)
    for r in runs.values():
        R.ALL_RUNS.setdefault(r, ("RLc", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    R.ALL_RUNS.setdefault(EC.SETTINGS["AGE"]["runs"]["RR"], ("RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    pairs_def = {}
    for x in ARMS:
        pairs_def[x + "-VS-RR"] = (RUN[x], BANKED["RR"])
        pairs_def[x + "-VS-RLc"] = (RUN[x], BANKED["RLc"])
        pairs_def[x + "-VS-K100"] = (RUN[x], GAINS["K100"])
    spread = "R0Hb20"
    run_list = [spread] + sorted(set(runs.values()))
    G = R.read_grid(a, E, "30", keys, run_list, spread, pairs_def, {}, log)
    lv = {k: {c: float(np.mean(G["levels"][r][c])) for c in ("C", "U", "F")} for k, r in runs.items()}
    lv_seed = {k: {c: [float(v) for v in np.asarray(G["levels"][r][c]).reshape(-1)[:2]] for c in ("C", "U", "F")} for k, r in runs.items()}
    pairs = {nm: R.ucf(p["headway"]) for nm, p in G["pairs"].items()}
    fams = {nm: {f_: R.reading(R.cells_of(p["family"]), f_) for f_ in R.NAMED} for nm, p in G["pairs"].items()}
    # the STRATEGIC family, as gaprule/read_gaprule_strat.py reads it (engine strategic_grid with its own controls, then the pair read at tick t0 + 20 on the arcs of 30, 50 and 80 m)
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
    rec = R.jl(a.joins_record)
    ejs = MCJ.sha_file(a.eval_join) if a.eval_join and os.path.isfile(a.eval_join) else None
    rd = lambda p: io.open(p, encoding="utf-8").read() if p and os.path.isfile(p) else ""      # noqa: E731
    wc = a.wrapper_check
    controls = {"q0": ctl_q0(rec, ejs), "q1": ctl_q1(rec, a.joins_dir, ejs),
                "q2": ctl_q2(rd(wc), rd(os.path.join(os.path.dirname(wc), "CHECK_WRAPPER_J.hashes")) if wc else "", a.wrapper_sha),
                "q3": ctl_q3(m["levels"]["RR"]["C"], m["levels"]["RLc"]["C"], m["levels"]["K100"]["C"]),
                "q4": ctl_q4(rd(a.tests_out)), "q5": ctl_q5(m["grid_all_pass"], m["strategic_a"], m["strategic_b"])}
    arms_rec = (rec or {}).get("arms", {})
    payload = {"levels": m["levels"], "levels_per_seed": m["levels_per_seed"], "pairs": m["pairs"], "families": m["families"], "strategic": m["strategic"],
               "share_of_contact_gap": share_rows(m["levels"], m["pairs"]), "family_note": FAM_REASON,
               "differences_against_K100": {nm: {"U": p["U"], "F": p["F"]} for nm, p in sorted(m["pairs"].items()) if nm.endswith("-VS-K100")},
               "pilot_gap_on_the_join_frames": {nm: {"error_statistics_pilot_minus_label": x.get("error_statistics_pilot_minus_label"), "n_valid_label_gap": x.get("n_valid_label_gap"),
                                                      "n_with_pilot_output": x.get("n_with_pilot_output"), "n_without_pilot_output": x.get("n_without_pilot_output"),
                                                      "fraction_without_pilot_output": x.get("fraction_without_pilot_output"), "n_mixed_above_60": x.get("n_mixed_above_60"),
                                                      "min_mixed": x.get("min_mixed"), "max_mixed": x.get("max_mixed"), "n_mixed_below_0": x.get("n_mixed_below_0")} for nm, x in sorted(arms_rec.items())}}
    return assemble_controls(controls, payload)


def print_arm_sections(payload):
    L = []
    for x in ARMS:
        row = [x]
        for ref in ("RR", "RLc", "K100"):
            p = payload["pairs"].get(x + "-VS-" + ref)
            if p:
                row.append("vs %s: C %s U %s F %s" % (ref, p["C"]["interval"], p["U"]["interval"], p["F"]["interval"]))
            if ref == "K100":
                continue
            st = (payload.get("strategic", {}).get("pairs") or {}).get(x + "-VS-" + ref)
            fm = payload["families"].get(x + "-VS-" + ref) or {}
            row.append("families(%s) %s strategic %s" % (ref, json.dumps(fm, sort_keys=True), st["reading"] if st else None))
        L.append("   ".join(row))
    return L


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--stage", choices=("controls", "names"), required=True)
    for n in ("pull", "code", "grid-dir", "w", "tree", "wrapper-sha", "joins-record", "joins-dir", "eval-join", "wrapper-check", "tests-out", "controls"):
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
        print("ZZ READ-CROPRULE10K controls rc 0 controls_all_pass %s" % res["controls_all_pass"])
        return 0
    cj = json.load(io.open(a.controls, encoding="utf-8"))
    if cj.get("card") != CARD or cj.get("controls_all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != list(CONTROLS) \
            or any(v.get("PASS") is not True for v in cj.get("controls", {}).values()):
        print("REFUSED: the controls file is not all-pass: no name is written")
        print("ZZ READ-CROPRULE10K names rc 3 controls_all_pass False")
        return 3
    names = all_names(cj["payload"])
    res = {"card": CARD, "tier": TIER, "controls_file": os.path.abspath(a.controls), "controls_sha256": sha_file(a.controls), "names": names, "h_note": h_sensitivity(cj["payload"]),
           "values": cj["payload"]}
    io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1))
    for k in sorted(names):
        print(names[k])
    for line in print_arm_sections(cj["payload"]):
        print(line)
    print("ZZ READ-CROPRULE10K names rc 0 controls_all_pass True")
    return 0


def selftest():
    ok = True
    n = names_of("KB5", [-0.05, -0.01], [0.0, 0.004], [-0.02, 0.03], [-0.03, -0.001], 0.0954, 0.0)               # H = 0.0477
    ok &= n["KB5_CONTACT"].endswith("KB5-CUTS-CONTACT") and n["KB5_HALF"].endswith("WITHIN-HALF-THE-GAP") and n["KB5_PRICE"].endswith("FALLING-BEHIND-NOT-SEPARATED") and n["KB5_K100"].endswith("LESS-CONTACT-THAN-K100")
    n = names_of("KD1", [0.001, 0.01], [0.05, 0.06], [0.01, 0.04], [0.001, 0.03], 0.0954, 0.0)
    ok &= n["KD1_CONTACT"].endswith("KD1-MORE-CONTACT") and n["KD1_HALF"].endswith("KD1-HALF-THE-GAP-NOT-REACHED") and n["KD1_PRICE"].endswith("FALLS-BEHIND-MORE") and n["KD1_K100"].endswith("MORE-CONTACT-THAN-K100")
    n = names_of("KB1", [-0.04, 0.01], [0.01, 0.06], [-0.04, -0.01], [-0.02, 0.02], 0.0954, 0.0)
    ok &= n["KB1_CONTACT"].endswith("CONTACT-NOT-SEPARATED-FROM-RR") and n["KB1_HALF"].endswith("HALF-THE-GAP-NOT-DECIDED") and n["KB1_PRICE"].endswith("FALLS-BEHIND-LESS")
    ok &= n["KB1_K100"].endswith("CONTACT-NOT-SEPARATED-FROM-K100")
    w, nr, nd = "-WITHIN-HALF-THE-GAP", "-HALF-THE-GAP-NOT-REACHED", "-HALF-THE-GAP-NOT-DECIDED"
    c = card_names({"KB5_HALF": PRE + "-KB5" + w, "KD5_HALF": PRE + "-KD5" + nr})
    ok &= c == {"CARD_BOUND": PRE + "-BOUND" + w, "CARD_FIXED": PRE + "-FIXED" + nr}
    c = card_names({"KB5_HALF": PRE + "-KB5" + nd, "KD5_HALF": PRE + "-KD5" + w})
    ok &= c == {"CARD_BOUND": PRE + "-BOUND" + nd, "CARD_FIXED": PRE + "-FIXED" + w}
    print("ZZ SELFTEST-READ-CROPRULE10K ok: %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
