#!/usr/bin/env python
"""read_errcut10k.py -- read of card PREREG_V8F_PROOF_ADD96 (ERRCUT10K): by what factor the present reader's gap error has to be cut for the lead rule, on the model's own rollout
(T1-kin/hb0.2, pre-roll of 30 steps (6 s)). Derived from gaprule/read_gaprule10kb2.py (the engine of the model reader EC, R, HS, H is imported, not edited).
Stage controls: computes the grid read and the STRATEGIC family of the eight arm runs (K100 K50 K25 K12, two model seeds each) and the banked RR and RLc inside this stage, runs e0, e1, e3, e5 here
and reads e2 (--wrapper-check) and e4 (--tests-out); writes CONTROLS_FILE. With every control PASS the values are written under `payload`, else under `withheld` and no name exists.
Stage names: reads CONTROLS_FILE, refuses it unless all controls pass, writes the names of card section 3 (three per arm and the ladder name) and binds the result to the controls file by sha256.
Usage: python read_errcut10k.py --stage controls --pull PULL --code CODE --grid-dir DIR --w W --tree TREE --out CONTROLS.json --wrapper-sha HEX16 --mix-record REC.json
          --joins-dir DIR --eval-join J.npz --wrapper-check FILE --tests-out FILE [--qmap-dir DIR] [--n-boot 2000]
       python read_errcut10k.py --stage names --controls CONTROLS.json --out RESULT.json     |   --selftest
Last line: ZZ READ-ERRCUT10K <stage> rc <rc> controls_all_pass <bool>"""
import argparse
import hashlib
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
for _p in (_HERE, os.path.join(_HERE, "..", "errcut")):
    if os.path.isfile(os.path.join(_p, "make_mix_joins.py")):
        sys.path.insert(0, os.path.abspath(_p))
        break
try:
    import read_estclose10k_n2 as EC        # noqa: E402
    HS, R, np = EC.HS, EC.R, EC.np
except ImportError:                          # the unit tests of the name rules and controls need numpy only
    EC = HS = R = None
    import numpy as np                       # noqa: E402
import make_mix_joins as MMJ                # noqa: E402

CARD, PRE = "PREREG_V8F_PROOF_ADD96", "ERRCUT10K"
SEEDS = (0, 1)
ARMS = ("K100", "K50", "K25", "K12")
CUT = {"K100": 1, "K50": 2, "K25": 4, "K12": 8}
RUN = {a: "RLcKF5P30Hb20m" + a for a in ARMS}
BANKED = {"RR": "RQ50KF5P30Hb20", "RLc": "RLcKF5P30Hb20"}
RR_C_EXPECTED, RLC_C_EXPECTED = 0.0954, 0.0
QMAP_DIR_DEFAULT = "/workspace/v8proof/integ10k/qmap_sweep"
TIER = "T1-kin/hb0.2 (kinematic rollout on the model's own actions), pre-roll of 30 steps (6 s)"
FAM_REASON = ("the rule changes the planned speed along the path through the distance to the lead; it does not move the path, so the lateral family is expected to be unmoved by it, "
              "and the tactical family is not read by the rule; the strategic family is read from the strategic grid of the engine")


def names_of(arm, d_rr_c, d_rlc_c, d_rr_f, c_rr, c_rlc):
    """the three names of one arm (card section 3). d_*: interval [lo, hi] of C(X)-C(RR), C(X)-C(RLc), F(X)-F(RR); c_rr, c_rlc: pooled point values of C of the references."""
    p = PRE + "-" + arm + "-"
    H_ = 0.5 * (c_rr - c_rlc)
    n1 = p + ("CUTS-CONTACT" if d_rr_c[1] < 0.0 else "CONTACT-NOT-SHOWN")
    n2 = p + ("WITHIN-HALF-THE-GAP" if d_rlc_c[1] <= H_ else ("OUTSIDE-HALF-THE-GAP" if d_rlc_c[0] > H_ else "HALF-THE-GAP-NOT-DECIDED"))
    n3 = p + ("FALLS-BEHIND-MORE" if d_rr_f[0] > 0.0 else ("FALLS-BEHIND-LESS" if d_rr_f[1] < 0.0 else "FALLING-BEHIND-NOT-SEPARATED"))
    return {arm + "_CONTACT": n1, arm + "_HALF": n2, arm + "_PRICE": n3}


def within(name):
    return name.endswith("-WITHIN-HALF-THE-GAP")


def ladder_name(half_names):
    """the name of the ladder, judged in the order of the card. half_names: {arm: the HALF name of that arm}"""
    w = {a: within(half_names[a + "_HALF"] if a + "_HALF" in half_names else half_names[a]) for a in ARMS}
    for i, a in enumerate(ARMS):                 # ARMS runs from the smallest cut (1) to the largest (8)
        if w[a] and any(not w[b] for b in ARMS[i + 1:]):
            return PRE + "-LADDER-NOT-MONOTONE"
    if not w["K12"]:
        return PRE + "-CUT-BY-8-NOT-SHOWN-TO-SUFFICE"
    n = min(CUT[a] for a in ARMS if w[a])
    return PRE + "-CUT-BY-%d-SUFFICES" % n


def ladder_check(half_names, name):
    """independent re-derivation of the ladder name from the HALF names; {"PASS", "expected"}"""
    ws = [within(half_names[a + "_HALF"]) for a in ARMS]                         # cuts 1, 2, 4, 8
    mono = all(ws[j] for i in range(4) if ws[i] for j in range(i, 4))
    if not mono:
        exp = PRE + "-LADDER-NOT-MONOTONE"
    elif not ws[3]:
        exp = PRE + "-CUT-BY-8-NOT-SHOWN-TO-SUFFICE"
    else:
        exp = PRE + "-CUT-BY-%d-SUFFICES" % (1, 2, 4, 8)[ws.index(True)]
    return {"PASS": name == exp, "expected": exp}


def all_names(payload):
    out = {}
    pr = payload["pairs"]
    for a in ARMS:
        out.update(names_of(a, pr[a + "-VS-RR"]["C"]["interval"], pr[a + "-VS-RLc"]["C"]["interval"], pr[a + "-VS-RR"]["F"]["interval"], payload["levels"]["RR"]["C"], payload["levels"]["RLc"]["C"]))
    out["LADDER"] = ladder_name(out)
    chk = ladder_check(out, out["LADDER"])
    if not chk["PASS"]:
        raise SystemExit("REFUSED: the ladder name %s is not the one the halves give (%s)" % (out["LADDER"], chk["expected"]))
    return out


def assemble_controls(controls, payload):
    ok = bool(controls) and all(isinstance(v, dict) and v.get("PASS") is True for v in controls.values())
    res = {"card": CARD, "tier": TIER, "controls": controls, "controls_all_pass": ok}
    res["payload" if ok else "withheld"] = payload
    return res


def ctl_e0(rec, eval_join_sha=None):
    c = MMJ.e0_check(rec) if isinstance(rec, dict) else {"PASS": False, "failures": ["no mixed joins record"]}
    bad = list(c["failures"])
    if isinstance(rec, dict) and eval_join_sha and rec.get("eval_join_sha256") != eval_join_sha:
        bad.append("the record is for another evaluation join")
    return {"PASS": not bad, "failures": bad}


def ctl_e1(rec, joins_dir, eval_join_sha=None, expect_qmap_dir=None):
    """control e1 from the record, plus: the join files on disk have the hashes of the record, and the record is for this evaluation join."""
    if not isinstance(rec, dict):
        return {"PASS": False, "failures": ["no mixed joins record"]}
    c = MMJ.e1_check(rec, expect_qmap_dir)
    bad = list(c["failures"])
    for a in ARMS:
        for s in SEEDS:
            nm = "%s_s%d" % (a, s)
            p = os.path.join(joins_dir or "", nm + ".npz")
            if not os.path.isfile(p) or MMJ.sha_file(p) != (rec.get("arms", {}).get(nm) or {}).get("sha256"):
                bad.append("%s: file missing or its sha256 is not the record's" % nm)
    if eval_join_sha and rec.get("eval_join_sha256") != eval_join_sha:
        bad.append("the record is for another evaluation join")
    return {"PASS": not bad, "failures": bad, "record_e1": c}


def ctl_e2(check_text, hashes_text, wrapper_sha16):
    lines = [x for x in (check_text or "").splitlines() if x.strip()]
    ok = bool(lines and lines[-1].strip() == "ZZ CHECK-WRAPPER-J PASS" and wrapper_sha16 and wrapper_sha16 in (hashes_text or ""))
    return {"PASS": ok, "last_line": lines[-1].strip() if lines else None, "wrapper_sha16_in_check_record": bool(wrapper_sha16 and wrapper_sha16 in (hashes_text or ""))}


def ctl_e3(c_rr, c_rlc):
    """the reader recomputes C of RR and RLc from the banked rollouts: 0.0954 and 0.0000 to four decimals (row 157)"""
    ok = c_rr is not None and c_rlc is not None and round(float(c_rr), 4) == RR_C_EXPECTED and round(float(c_rlc), 4) == RLC_C_EXPECTED
    return {"PASS": bool(ok), "C_RR": c_rr, "C_RLc": c_rlc, "expected_RR": RR_C_EXPECTED, "expected_RLc": RLC_C_EXPECTED}


def ctl_e4(tests_text):
    t = tests_text or ""
    m = re.findall(r"(\d+) passed", t)
    ok = bool(m) and int(m[-1]) > 0 and not re.search(r"\d+ (failed|error)", t)
    return {"PASS": ok, "passed": int(m[-1]) if m else None}


def ctl_e5(grid_all_pass, strat_a, strat_b):
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
    return MMJ.sha_file(p)


def measure(a, E, log):
    """the pod part: grid read (levels, pairs with intervals, families) of the eight arm runs and the banked RR, RLc; then the strategic grid. -> dict"""
    H, Q3, RH = E
    g30 = R.jl(os.path.join(a.grid_dir, "grid_p30.json"))
    J = H.load_join()
    keys = R.grid_keys(g30, [str(c) for c in J["clip"]])
    runs = dict(RUN)
    runs.update(BANKED)
    for r in runs.values():
        R.ALL_RUNS.setdefault(r, ("RLc", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    R.ALL_RUNS.setdefault(EC.SETTINGS["AGE"]["runs"]["RR"], ("RR", "K", 30, "deadreckon", "30", "plan", 5, 0.2))
    pairs_def = {}
    for x in ARMS:
        pairs_def[x + "-VS-RR"] = (RUN[x], BANKED["RR"])
        pairs_def[x + "-VS-RLc"] = (RUN[x], BANKED["RLc"])
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
    rec = R.jl(a.mix_record)
    ejs = MMJ.sha_file(a.eval_join) if a.eval_join and os.path.isfile(a.eval_join) else None
    rd = lambda p: io.open(p, encoding="utf-8").read() if p and os.path.isfile(p) else ""      # noqa: E731
    wc = a.wrapper_check
    controls = {"e0": ctl_e0(rec, ejs), "e1": ctl_e1(rec, a.joins_dir, ejs, a.qmap_dir or None),
                "e2": ctl_e2(rd(wc), rd(os.path.join(os.path.dirname(wc), "CHECK_WRAPPER_J.hashes")) if wc else "", a.wrapper_sha), "e3": ctl_e3(m["levels"]["RR"]["C"], m["levels"]["RLc"]["C"]),
                "e4": ctl_e4(rd(a.tests_out)), "e5": ctl_e5(m["grid_all_pass"], m["strategic_a"], m["strategic_b"])}
    arms_rec = (rec or {}).get("arms", {})
    payload = {"levels": m["levels"], "levels_per_seed": m["levels_per_seed"], "pairs": m["pairs"], "families": m["families"], "strategic": m["strategic"],
               "share_of_contact_gap": share_rows(m["levels"], m["pairs"]), "family_note": FAM_REASON,
               "planted_error_statistics": {nm: {"error_statistics_mixed_minus_label": x.get("error_statistics_mixed_minus_label"), "n_valid_label_gap": x.get("n_valid_label_gap"),
                                                 "n_with_reader_read": x.get("n_with_reader_read"), "n_without_reader_read": x.get("n_without_reader_read"), "n_mixed_above_60": x.get("n_mixed_above_60"),
                                                 "min_mixed": x.get("min_mixed"), "n_mixed_below_0": x.get("n_mixed_below_0")} for nm, x in sorted(arms_rec.items())}}
    return assemble_controls(controls, payload)


def print_arm_sections(payload):
    L = []
    for x in ARMS:
        row = [x]
        for ref in ("RR", "RLc"):
            p = payload["pairs"].get(x + "-VS-" + ref)
            if p:
                row.append("vs %s: C %s U %s F %s" % (ref, p["C"]["interval"], p["U"]["interval"], p["F"]["interval"]))
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
    for n in ("pull", "code", "grid-dir", "w", "tree", "wrapper-sha", "mix-record", "joins-dir", "eval-join", "wrapper-check", "tests-out", "controls", "qmap-dir"):
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
        print("ZZ READ-ERRCUT10K controls rc 0 controls_all_pass %s" % res["controls_all_pass"])
        return 0
    cj = json.load(io.open(a.controls, encoding="utf-8"))
    if cj.get("card") != CARD or cj.get("controls_all_pass") is not True or "payload" not in cj or sorted(cj.get("controls", {})) != ["e0", "e1", "e2", "e3", "e4", "e5"] \
            or any(v.get("PASS") is not True for v in cj.get("controls", {}).values()):
        print("REFUSED: the controls file is not all-pass: no name is written")
        print("ZZ READ-ERRCUT10K names rc 3 controls_all_pass False")
        return 3
    names = all_names(cj["payload"])
    res = {"card": CARD, "tier": TIER, "controls_file": os.path.abspath(a.controls), "controls_sha256": sha_file(a.controls), "names": names, "values": cj["payload"]}
    io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, indent=1))
    for k in sorted(names):
        print(names[k])
    for line in print_arm_sections(cj["payload"]):
        print(line)
    print("ZZ READ-ERRCUT10K names rc 0 controls_all_pass True")
    return 0


def selftest():
    ok = True
    n = names_of("K100", [-0.05, -0.01], [0.0, 0.004], [-0.02, 0.03], 0.0954, 0.0)               # H = 0.0477
    ok &= n["K100_CONTACT"].endswith("CUTS-CONTACT") and n["K100_HALF"].endswith("WITHIN-HALF-THE-GAP") and n["K100_PRICE"].endswith("FALLING-BEHIND-NOT-SEPARATED")
    n = names_of("K12", [-0.01, 0.01], [0.05, 0.06], [0.01, 0.04], 0.0954, 0.0)
    ok &= n["K12_CONTACT"].endswith("CONTACT-NOT-SHOWN") and n["K12_HALF"].endswith("OUTSIDE-HALF-THE-GAP") and n["K12_PRICE"].endswith("FALLS-BEHIND-MORE")
    n = names_of("K25", [-0.04, -0.01], [0.01, 0.06], [-0.04, -0.01], 0.0954, 0.0)
    ok &= n["K25_HALF"].endswith("HALF-THE-GAP-NOT-DECIDED") and n["K25_PRICE"].endswith("FALLS-BEHIND-LESS")
    w, o = "-WITHIN-HALF-THE-GAP", "-OUTSIDE-HALF-THE-GAP"
    ladder = lambda *flags: {a + "_HALF": PRE + "-" + a + (w if f else o) for a, f in zip(ARMS, flags)}      # noqa: E731
    ok &= ladder_name(ladder(1, 1, 1, 1)).endswith("CUT-BY-1-SUFFICES") and ladder_name(ladder(0, 0, 1, 1)).endswith("CUT-BY-4-SUFFICES")
    ok &= ladder_name(ladder(0, 0, 0, 0)).endswith("CUT-BY-8-NOT-SHOWN-TO-SUFFICE") and ladder_name(ladder(1, 0, 1, 1)).endswith("LADDER-NOT-MONOTONE")
    ok &= ladder_check(ladder(1, 0, 1, 1), PRE + "-CUT-BY-1-SUFFICES")["PASS"] is False
    print("ZZ SELFTEST-READ-ERRCUT10K ok: %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
