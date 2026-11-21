#!/usr/bin/env python
"""qhead_names_qf.py -- note PREREG_V8F_PROOF_ADD42_N2 section 3 (dose step): the arms LSDQF_s0, LSDQF_s1 (the LSD type only), names with the suffix -QF, and the
controls d1 (the label pack of all clips equals the banked pack on the probe clips) and d2 (the new states equal the banked states; the planted shift fails) gate
every name beside h1 to h8. Made from qhead_names_n2.py by make_qf_tools.py. The text of the earlier forms follows.

qhead_names_n2.py -- note PREREG_V8F_PROOF_ADD42_N2 section 2: control h5 gates PER ARM TYPE (a type with a checkpoint outside the band gets no name and no outcome
number; the other type is read); h1, h4, h6, h7, h8 gate every name; h2 and h3 are read on the arms of the read types. Made from qhead_names_n1.py by make_names_n2.py.
The text of the first form follows.

qhead_names_n1.py -- note PREREG_V8F_PROOF_ADD42_N1, sections 3 to 5 (second version of qhead_names.py): the controls h1 to h8 FIRST; the names only if every control passed. A failed control is printed first, the names are null
and no outcome difference is written (nothing of the sections 7 'beside' either).

  python qhead_names_n1.py --hw RESULT_HEADWAY.json --q3 RESULT_Q3.json --fit-dir DIR --h1-text FILE --tool-file v8f_t1_eval_rule_qhead.py --rule-log-dir DIR --h3 DIR --h6 MUTATIONS_QHEAD_N1.json --out READ_QHEAD_N1.json

Inputs
  --hw, --q3      the outputs of headway_read_r2.py and q3_read_r2.py for the pairs of section 5 (qhead_read.sh)
  --fit-dir       the fit records <BASE>.qhead.json (h4, h5 on the CHECK coverage, h8 on the clip sets)
  --h1-text       the queue log of the FIRST run, holding the line of the cmp of the control rollout against LSRH_s0 (`CMP files=N differing=0`) and the `tool <sha256 prefix>` of its queue start lines.
                  h1 is carried from there only if every recorded tool sha256 is a prefix of the sha256 of --tool-file (the rollout tool file used now); else h1 is reported as missing and fails.
  --tool-file     the rollout tool file used now (v8f_t1_eval_rule_qhead.py)
  --rule-log-dir  where rule_log_<ARM>.npz can be found (<dir>/<ARM>/ticks/, <dir>/<ARM>/ or <dir>/): h2 = lift 0 in every rule log of this card
  --h3            a folder of H3_<ARM>.json files (qhead_h3.py)
  --h6            the result of run_mutations.py (all mutations red, the unmutated control green, the gradient and planted-quantile tests green)
Tier T1-kin (kinematic rollout on the model's own actions, logged frames) for every rollout number; none is driving performance. Family served: LONGITUDINAL.
STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument)
"""
import argparse
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

TYPES = {"LSD": dict(qh="LSDQF", none="LSD", rh="LSDRH", rl="LSDRL", mc="LSDMC", qc="LSDQC", base="LSD")}
SEEDS = (0, 1)
H5_LO, H5_HI = 0.15, 0.25
SUFFIX = "-QF"
FAMILY_LINE = "STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument)"


def QM_sha(path, blk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(blk), b""):
            h.update(b)
    return h.hexdigest()


def jl(p):
    return json.load(io.open(p, encoding="utf-8")) if p and os.path.isfile(p) else None


def both(c, sign):
    """per-base differences all on one side of 0 and the two-level interval excluding 0."""
    return bool(len(c["d_s"]) >= 2 and all((d < 0) if sign < 0 else (d > 0) for d in c["d_s"]) and c["A_excludes_0"])


def pk(x, y):
    return "%s_minus_%s" % (x, y)


def find_rule_log(root, arm):
    for p in (os.path.join(root, arm, "ticks", "rule_log_%s.npz" % arm), os.path.join(root, arm, "rule_log_%s.npz" % arm), os.path.join(root, "rule_log_%s.npz" % arm)):
        if os.path.isfile(p):
            return p
    return None


def h8_check(f):
    """h8 from the fit record: the three clip sets are disjoint half-open ranges in manifest order (fit < select < check), and the clip indices of the rows that entered the gradient steps, the selection
    and the shift lie inside the fit, select and select sets, so that no check row entered any of them."""
    try:
        cs = f["clip_sets"]
        fit, sel, chk = cs["fit"], cs["select"], cs["check"]
        used = f["clip_index_range_of_rows_used"]
        ordered = bool(fit[0] < fit[1] <= sel[0] < sel[1] <= chk[0] < chk[1])
        inside = lambda r, s_: bool(s_[0] <= r[0] and r[1] < s_[1])                                       # noqa: E731
        ok_use = bool(inside(used["gradient_steps"], fit) and inside(used["selection"], sel) and inside(used["shift"], sel))
        no_check = bool(all(r[1] < chk[0] or r[0] >= chk[1] for r in (used["gradient_steps"], used["selection"], used["shift"])))
        return {"PASS": bool(ordered and ok_use and no_check), "ordered_disjoint": ordered, "uses_inside_their_sets": ok_use, "no_check_row_in_any_use": no_check}
    except (KeyError, TypeError, IndexError) as e:
        return {"PASS": False, "why": "record lacks the clip sets or the row ranges: %r" % (e,)}


def controls(a):
    C = {}
    t = io.open(a.h1_text, encoding="utf-8", errors="replace").read() if a.h1_text and os.path.isfile(a.h1_text) else ""
    hits = re.findall(r"CMP files=(\d+) differing=(\d+)", t)
    recorded = sorted(set(re.findall(r"queue start;.*? tool ([0-9a-f]{8,64})(?![0-9a-f])", t)))
    now = QM_sha(a.tool_file) if getattr(a, "tool_file", "") and os.path.isfile(a.tool_file) else ""
    carried = bool(recorded and now and all(now.startswith(x) for x in recorded))
    C["h1_control_rollout_equals_head_fed_dump"] = {"PASS": bool(carried and hits and int(hits[-1][0]) > 0 and int(hits[-1][1]) == 0), "cmp": hits[-1] if hits else None,
                                                     "tool_sha256_recorded_in_first_run": recorded, "tool_sha256_now_prefix": now[:16] or None, "tool_unchanged": carried,
                                                     "why": None if carried else "h1 is missing: the rollout tool of the first run is not byte-identical to the tool file used now (or the sha256 is not recorded)"}
    cov = {}
    for ty in TYPES:
        for s in SEEDS:
            base = "%s_s%d" % (TYPES[ty]["base"], s)
            f0 = jl(os.path.join(a.fit_dir, base + ".qhead.json")) if a.fit_dir else None
            c0 = ((f0 or {}).get("coverage_check") or {}).get("share_q_above_label")
            cov[base] = bool(c0 is not None and H5_LO <= c0 <= H5_HI)
    not_named = {ty: [b for b in ("%s_s%d" % (TYPES[ty]["base"], s) for s in SEEDS) if not cov[b]] for ty in TYPES}
    not_named = {ty: v for ty, v in not_named.items() if v}
    read_types = [ty for ty in TYPES if ty not in not_named]
    h2 = {}
    arms = ["%s_s%d" % (TYPES[ty]["qh"], s) for ty in read_types for s in SEEDS] + ["LSQHI_s0"]
    for arm in arms:
        p = find_rule_log(a.rule_log_dir, arm) if a.rule_log_dir else None
        if p is None:
            h2[arm] = {"PASS": False, "why": "no rule log"}
            continue
        z = np.load(p, allow_pickle=True)
        h2[arm] = {"PASS": bool(int(np.asarray(z["lift"]).sum()) == 0 and len(z["lift"]) > 0), "lift_sum": int(np.asarray(z["lift"]).sum()), "n_rows": int(len(z["lift"]))}
    C["h2_no_plan_moved_ahead"] = {"PASS": bool(all(v["PASS"] for v in h2.values())), "per_arm": h2}
    h3 = {}
    for ty in read_types:
        for s in SEEDS:
            arm = "%s_s%d" % (TYPES[ty]["qh"], s)
            r = jl(os.path.join(a.h3, "H3_%s.json" % arm)) if a.h3 else None
            h3[arm] = {"PASS": bool(r and r.get("PASS")), "a": (r or {}).get("a_in_rollout", {}).get("PASS"), "b": (r or {}).get("b_offline_recompute", {}).get("PASS"), "b_counts": (r or {}).get("b_offline_recompute", {}).get("counts"), "why": None if (r and r.get("PASS")) else ("no H3 file" if not r else r.get("why", "failed"))}
    C["h3_gap_used_equals_min_recomputed"] = {"PASS": bool(all(v["PASS"] for v in h3.values())), "per_arm": h3}
    h4, h5, h8 = {}, {}, {}
    for ty in TYPES:
        for s in SEEDS:
            base = "%s_s%d" % (TYPES[ty]["base"], s)
            f = jl(os.path.join(a.fit_dir, base + ".qhead.json")) if a.fit_dir else None
            if not f:
                h4[base] = {"PASS": False, "why": "no fit record"}
                h5[base] = {"PASS": False, "why": "no fit record"}
                h8[base] = {"PASS": False, "why": "no fit record"}
                continue
            h8[base] = h8_check(f)
            h4[base] = {"PASS": bool(f.get("q4_only_train_files") and not [p for p in f["files_read_sha256"] if "eval" in os.path.basename(p).lower()])}
            cc = f.get("coverage_check") or {}
            c = cc.get("share_q_above_label")
            h5[base] = {"PASS": bool(c is not None and H5_LO <= c <= H5_HI), "coverage_check": c, "n_check_rows": cc.get("n")}
    C["h4_train_files_only"] = {"PASS": bool(all(v["PASS"] for v in h4.values())), "per_base": h4}
    C["h5_check_coverage_in_0.15_0.25"] = {"PASS": bool(read_types), "rule": "per arm type: a type is read only if both of its checkpoints are inside the band (note N1 section 4, note N2 section 2)",
                                           "per_base": h5, "types_read": read_types, "types_not_named": not_named}
    C["h8_check_clips_used_by_no_fit_and_no_selection"] = {"PASS": bool(all(v["PASS"] for v in h8.values())), "per_base": h8}
    m = jl(a.h6)
    C["h6_fit_reaches_parameters_and_learns_a_quantile"] = {"PASS": bool(m and m.get("all_ok")), "n_mutations": (m or {}).get("n_mutations"), "why": None if (m and m.get("all_ok")) else "no mutation result or not all ok"}
    hw, q3 = jl(a.hw), jl(a.q3)
    g5 = bool(hw and hw["controls"].get("h2_planted_regression", {}).get("PASS") and hw["controls"]["h2_planted_regression"].get("named_worse_on_U"))
    C["h7_reader_controls"] = {"PASS": bool(hw and q3 and hw["controls"].get("ALL_PASS") and q3["controls"].get("ALL_PASS") and g5), "headway_ALL_PASS": bool(hw and hw["controls"].get("ALL_PASS")),
                               "family_ALL_PASS": bool(q3 and q3["controls"].get("ALL_PASS")), "g5_planted_regression_read_as_worse": g5}
    d1 = jl(getattr(a, "d1", ""))
    C["d1_label_pack_equals_banked_pack"] = {"PASS": bool(d1 and d1.get("pass") is True), "why": None if d1 else "no d1 record"}
    d2 = {}
    for ty in TYPES:
        for s in SEEDS:
            base = "%s_s%d" % (TYPES[ty]["base"], s)
            v = jl(os.path.join(a.d2_dir, "d2_%s.json" % base)) if getattr(a, "d2_dir", "") else None
            ps = (v or {}).get("planted_shift") or {}
            d2[base] = {"PASS": bool(v and v.get("pass") is True and ps.get("fails") is True), "pass": (v or {}).get("pass"), "planted_shift_fails": ps.get("fails")}
    C["d2_new_states_equal_banked_states"] = {"PASS": bool(all(v["PASS"] for v in d2.values())), "per_base": d2}
    return C, hw, q3


def names(hw, q3, types=None):
    N, B = {}, {}
    for ty, m in TYPES.items():
        if types is not None and ty not in types:
            continue
        n = {}
        seeds_ok = all(s in hw["pairs"].get(pk(m["qh"], m["none"]), {}).get("seeds_read", []) for s in SEEDS)
        if not seeds_ok:
            N[ty] = {"all": "NOT NAMED (both seeds are needed)"}
            continue
        u = lambda x, y: hw["pairs"][pk(x, y)]["cells"]["U"]                                   # noqa: E731
        c_ = lambda x, y: hw["pairs"][pk(x, y)]["cells"]["C"]                                  # noqa: E731
        lev = lambda arm: hw["levels"][arm]["cells"]["U"]["per_seed"]                          # noqa: E731
        un, uq, ul = lev(m["none"]), lev(m["qh"]), lev(m["rl"])
        share = [(un[i] - uq[i]) / (un[i] - ul[i]) if (un[i] - ul[i]) != 0 else None for i in range(2)]
        uqn = u(m["qh"], m["none"])
        n["reaches_oracle"] = "QHEAD-REACHES-ORACLE-QF" if (all(s is not None and s >= 0.5 for s in share) and uqn["A_hi"] < 0) else None
        uqc = u(m["qh"], m["qc"])
        n["U_against_map"] = "QHEAD-CUTS-MORE-QF" if both(uqc, -1) else ("QHEAD-CUTS-LESS-QF" if both(uqc, +1) else "QHEAD-U-EQUAL-QF")
        e = q3["pairs"][pk(m["qh"], m["qc"])]["beside"]["ade6_m|all"]
        n["cost_against_map"] = "QHEAD-CHEAPER-QF" if both(e, -1) else ("QHEAD-DEARER-QF" if both(e, +1) else "QHEAD-COST-EQUAL-QF")
        q = q3["pairs"][pk(m["qh"], m["none"])]["Q3"]
        n["cost_against_no_rule"] = "QHEAD-NO-COST-QF" if q.get("name") == "NO-COST" else "QHEAD-COSTS-QF"
        n["cost_against_no_rule_cells"] = {"name": q.get("name"), "cost_families": q.get("cost_families"), "gain_families": q.get("gain_families")}
        cell = lambda c: {"per_base": [round(x, 4) for x in c["d_s"]], "pooled": round(c["D"], 4), "interval": [round(c["A_lo"], 4), round(c["A_hi"], 4)], "excludes_0": bool(c["A_excludes_0"])}   # noqa: E731
        B[ty] = {"share_of_label_fed_fall_reached_per_base": share, "U_qh_minus_none": cell(uqn), "U_qh_minus_map": cell(uqc), "ade6_all_qh_minus_map_m": cell(e),
                 "U_qh_minus_label_fed": cell(u(m["qh"], m["rl"])), "U_qh_minus_fixed_margin": cell(u(m["qh"], m["mc"])), "U_qh_minus_head_fed": cell(u(m["qh"], m["rh"])),
                 "contact_qh_minus_label_fed": cell(c_(m["qh"], m["rl"])), "contact_qh_minus_fixed_margin": cell(c_(m["qh"], m["mc"]))}
        N[ty] = n
    return N, B


def main(argv=None):
    ap = argparse.ArgumentParser(allow_abbrev=False)
    for k in ("hw", "q3", "fit-dir", "h1-text", "tool-file", "rule-log-dir", "h3", "h6", "d1", "d2-dir"):
        ap.add_argument("--" + k, default="")
    ap.add_argument("--h3-beside-dir", default="", help="folder of H3 files for the beside numbers (default: --h3)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    a.fit_dir, a.h1_text, a.rule_log_dir = a.fit_dir, a.h1_text, a.rule_log_dir
    C, hw, q3 = controls(a)
    allp = all(v["PASS"] for v in C.values())
    res = {"card": "PREREG_V8F_PROOF_ADD42_N2 section 3 (dose step: head fitted on all 4,572 training clips; arms LSDQF)", "reading_rule": "PREREG_V8F_PROOF_ADD42_N2 section 2 (h5 gates per arm type)", "tier": "T1-kin (kinematic rollout on the model's own actions, logged frames; a diagnostic, never driving performance)", "family_line": FAMILY_LINE,
           "controls": C, "controls_all_pass": bool(allp), "names": None, "beside": None}
    print("== controls of note ADD42_N2 section 3 (printed first)")
    for k, v in C.items():
        print("  %-48s %s %s" % (k, "PASS" if v["PASS"] else "FAIL", json.dumps({x: y for x, y in v.items() if x not in ("PASS", "per_arm", "per_base")})[:160]))
    if allp:
        h5c = C["h5_check_coverage_in_0.15_0.25"]
        N, B = names(hw, q3, h5c["types_read"])
        for ty, bases in h5c["types_not_named"].items():
            N[ty] = {"all": "NOT NAMED (h5: the coverage on the check clips is outside the band at %s; note N1 section 4, note N2 section 2); no outcome number of this type is written" % ", ".join(bases)}
        res["names"], res["beside"] = N, B
        for ty in N:
            print("== names", ty, json.dumps(N[ty]))
        for ty in B:
            beside = dict(B[ty])
            h3b = a.h3_beside_dir or a.h3
            if h3b:
                rs = [jl(os.path.join(h3b, "H3_%s_s%d.json" % (TYPES[ty]["qh"], s))) for s in SEEDS]
                beside["first_tick_mean_amount_taken_off_read_m"] = [(r or {}).get("beside", {}).get("mean_amount_taken_off_read_m") for r in rs]
                beside["coverage_on_evaluation_windows"] = [(r or {}).get("beside", {}).get("coverage_eval_windows_share_q_above_label") for r in rs]
            res["beside"][ty] = beside
            print("== beside", ty, json.dumps(beside)[:600])
    else:
        print("== names: NOT GIVEN (a control failed or is not evaluable); the outcome differences are withheld, nothing of them is written")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with io.open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1)
    return 0 if allp else 2


if __name__ == "__main__":
    sys.exit(main())
