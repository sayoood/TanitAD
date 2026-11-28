#!/usr/bin/env python3
"""TF-N6 navtest readout (PREREG_TF_N6_DRAFT.md sections 6 to 9). Reads ONLY harness CSVs (the pinned W3 harness), the build report and the control files.

  read     python n6_nt_analyze.py read --harness-csv N2R=<csv> N4D=<csv> N5A=<csv> N6H=<csv> N6T=<csv> --build-report <n6_build_report.json> --unch <unch.json>
               --pins <N6_PINS.json> --pin-root <run> --proxy-eval <eval dir> --prereg-sha256 <sha> --token-log <navtest_inputs.json.gz> --known36 <json> --k 49
               --a1-analyze <tf_n1_analyze_a1.py> --out <json>
           CONTROLS FIRST (C-COV, C-BUILD, C-UNCH, C-PIN, C-CERT, C-ID-C, C-EST, C-REG). If any fails it prints ZZN6_CONTROLS FAIL, writes a file with NO aggregate and exits 1
           BEFORE any N6-H number is computed. Then the section 6 bars.
  famread  python n6_nt_analyze.py famread --fam-ci <fam_ci json N6H vs N5A> --out <json>     (the four-family guard N6-H minus N5-A)

N6-X (certificate inverted) is not scored: the build proves it changed 0 tokens and its seam is bit-identical to N5-A (C-UNCH, C-X0), so its harness rows ARE the banked N5-A rows.

BARS (section 6): (1) PRIMARY, paper level for N6-H on the primary 12,110: PDMS >= 95.8 AND DAC >= 99.9 (point estimates; paired log-cluster bootstrap CIs vs N5-A reported);
(2) no harm: paired NC and DDC delta (N6-H - N5-A) point >= 0 AND no changed token's harness NC or DDC below N5-A's on that token (TTC and C reported: the declared tolerance);
(3) certificate integrity: every changed token reads harness DAC = 1 (count of agreement with the builder; mismatches named); (4) four-family guard (famread);
(5) K = 49, Bonferroni at K. CONFIRMED iff (1), (2) and (3). Estimator: `Boot` of tf_n1_analyze_a1.py: paired log-cluster bootstrap over the logs, 10,000 draws, seed 20260927, 95 % percentile,
plus the Bonferroni interval at K. It answers "would another draw of LOGS say this?" only; the arms are deterministic (C-DET), so it says nothing about training or inference variance."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "n5"))
import n5_nt_analyze as A5  # noqa: E402  (the PINNED TF-N5 readout: loaders, pins check, the families reader constants)

BAR = {"paper_pdms": 95.8, "paper_dac": 99.9, "near_pdms": 95.3, "near_dac": 99.5, "id_tol": 1e-6, "nc_ddc_floor_x100": -1e-9, "tok_tol": 1e-12}
GUARD, MET, CSV_COL, PAPER, REPORTED = A5.GUARD, A5.MET, A5.CSV_COL, A5.PAPER, A5.REPORTED
NEW_ARMS = ("N6H", "N6T")
NEED = ("N2R", "N4D", "N5A", "N6H", "N6T")
load_a1, sha256, load_csv, load_tlog, pins_check = A5.load_a1, A5.sha256, A5.load_csv, A5.load_tlog, A5.pins_check
NAMING = "RJ5 teacher + P5-R + REPAIR-D + L1 + tolerance (NC/DDC kept, TTC/C free) + human-log fallback on {nh} of 12,110 tokens (N_teacher_tol = {nt}, N_human = {nh})"


def evaluate(a, A=None) -> int:
    A = A or load_a1(a.a1_analyze)
    A.K = a.k
    srcs = dict(s.split("=", 1) for s in a.harness_csv)
    if any(k not in srcs for k in NEED):
        raise SystemExit(f"--harness-csv needs {NEED}")
    arms, n_bad = {}, {}
    for nm, fp in srcs.items():
        arms[nm], n_bad[nm] = load_csv(fp)
    tlog = load_tlog(a.token_log)
    allt = sorted(set.intersection(*[set(v) for v in arms.values()]))
    known = A.load_known36(a.known36)
    prim = [t for t in allt if t not in known]
    tl = {t: tlog[t] for t in allt}
    res: dict = {"prereg": a.prereg_sha256 or "UNSET", "K": a.k, "n_tokens": len(allt), "n_primary": len(prim), "controls": {}, "estimator":
                 f"paired log-cluster bootstrap, 136 logs, 10,000 draws, seed 20260927, 95 percent percentile (eval/proxy_eval.boot draws); Bonferroni at K={a.k}. It answers "
                 "'would another draw of LOGS say this?' only; arms are deterministic (C-DET), so it is silent on training and inference variance"}
    C = res["controls"]
    br = json.load(open(a.build_report, encoding="utf-8"))
    rep = br["repaired"]
    C["C-COV"] = {"pass": len(allt) == a.expect_n and len(prim) == a.expect_primary and all(n_bad[k] == 0 and len(arms[k]) == a.expect_n for k in NEED),
                  "tokens": len(allt), "expect": a.expect_n, "primary": len(prim), "expect_primary": a.expect_primary, "invalid_or_error_rows": {k: v for k, v in n_bad.items() if v}}
    C["C-BUILD"] = {"pass": br.get("verdict") == "PASS", "controls": {k: v.get("pass") for k, v in br.get("controls", {}).items()}}
    un = json.load(open(a.unch, encoding="utf-8")) if a.unch else {}
    C["C-UNCH"] = {"pass": bool(un.get("pass")), "arms": {k: v.get("pass") for k, v in un.get("arms", {}).items()}, "n_residual": un.get("n_residual"),
                   "what": "plan20 and seam files on disk bit-identical to banked N5-A off the applied set; residual count == the card's; N6-X bit-identical to N5-A"}
    C["C-PIN"] = pins_check(a.pins, a.pin_root) if a.pins else {"pass": False, "why": "no --pins"}
    cert_bad = {arm: [t for t in rep.get(arm, []) if t not in arms[arm] or arms[arm][t]["dac"] != 1.0] for arm in NEW_ARMS}
    C["C-CERT"] = {"pass": not any(cert_bad.values()), "exceptions": {k: len(v) for k, v in cert_bad.items()}, "examples": {k: v[:3] for k, v in cert_bad.items()},
                   "certified": {k: len(rep.get(k, [])) for k in NEW_ARMS}, "what": "every token the builder certified (DAC = 1) has harness DAC = 1"}
    worst = {m: 0.0 for m in MET}
    sentinel = sorted(allt, key=lambda t: hashlib.sha1(t.encode()).hexdigest())[:200]
    idc, sent_over = {}, {}
    for arm in NEW_ARMS:
        r = set(rep.get(arm, []))
        over = sover = 0
        for t in allt:
            if t in r:
                continue
            d = max(abs(arms[arm][t][m] - arms["N5A"][t][m]) for m in MET)
            for m in MET:
                worst[m] = max(worst[m], abs(arms[arm][t][m] - arms["N5A"][t][m]))
            over += d > a.id_tol
            if t in sentinel:
                sover += d > a.id_tol
        idc[arm] = {"unchanged_tokens": len(allt) - len(r), "tokens_over_tol": int(over)}
        sent_over[arm] = int(sover)
    C["C-ID-C"] = {"pass": all(v["tokens_over_tol"] == 0 for v in idc.values()) and not any(sent_over.values()), "tol": a.id_tol, "arms": idc, "sentinel_200_over_tol": sent_over,
                   "max_abs_diff_by_metric": worst,
                   "what": "full set: every token outside the applied set has a harness row equal to the banked N5-A row on all 7 metrics (incl. the 200 smallest-sha1 sentinels)"}
    bt = A.Boot(prim, tl)
    if a.proxy_eval:
        spec = importlib.util.spec_from_file_location("proxy_eval_n6", os.path.join(a.proxy_eval, "proxy_eval.py"))
        PE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(PE)
        pt = {t: arms["N2R"][t]["pdms"] for t in prim}
        ref, ours = PE.boot(pt, tl), bt.stat(pt)["_raw"]
        dev = max(abs(x - y) for x, y in zip(ref, ours))
        C["C-EST"] = {"pass": dev <= 1e-9, "max_abs_dev": dev}
    else:
        C["C-EST"] = {"pass": False, "why": "no --proxy-eval"}
    xu = un.get("arms", {}).get("N6X", {})
    C["C-REG"] = {"pass": bool(not rep.get("N6X") and xu.get("pass") and xu.get("bit_identical_to_n5a") and br["controls"].get("C-X0", {}).get("pass")),
                  "n6x_applied": len(rep.get("N6X", [])), "n6x_bit_identical_to_n5a": xu.get("bit_identical_to_n5a"),
                  "what": "the deliberate-regression arm (certificate inverted) changed 0 tokens and its seam and plans are bit-identical to N5-A; its harness rows are the N5-A rows (the harness is deterministic, C-DET)"}
    allpass = all(c["pass"] for c in C.values())
    print("ZZN6_CONTROLS", "PASS" if allpass else "FAIL", json.dumps({k: v["pass"] for k, v in C.items()}), flush=True)
    if not allpass:
        res["verdict"] = "NO VERDICT: a control failed; no N6-H aggregate was computed"
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
        return 1

    def tab(arm, toks):
        out = {m: {k: v for k, v in bt.stat({t: arms[arm][t][m] for t in toks}).items() if k != "_raw"} for m in MET}
        out["zero_rows"] = {m: int(sum(arms[arm][t][m] == 0 for t in toks)) for m in ("pdms", "dac", "nc")}
        return out

    def paired(x, y, metric="pdms"):
        return {k: v for k, v in bt.stat({t: arms[x][t][metric] - arms[y][t][metric] for t in prim}).items() if k != "_raw"}

    def paper_read(arm):
        T = tab(arm, prim)
        p, dc = T["pdms"]["mean"], T["dac"]["mean"]
        lvl = "PAPER LEVEL" if p >= BAR["paper_pdms"] and dc >= BAR["paper_dac"] else "NEAR" if p >= BAR["near_pdms"] and dc >= BAR["near_dac"] else "BELOW"
        mm = {m: T[m]["mean"] / 100.0 for m in ("nc", "dac", "ep", "ttc", "c")}
        f = lambda v: v["nc"] * v["dac"] * (5 * v["ep"] + 5 * v["ttc"] + 2 * v["c"]) / 12.0     # noqa: E731
        return {"pdms": T["pdms"], "dac": T["dac"], "level": lvl, "zero_dac_tokens": T["zero_rows"]["dac"], "gap_to_paper_pdms": round(BAR["paper_pdms"] - p, 4),
                "paper_pdms_point_ok": bool(p >= BAR["paper_pdms"]), "paper_dac_point_ok": bool(dc >= BAR["paper_dac"]),
                "decomposition_mean_field": {"swap_gain_x100": {m: round(100 * (f(dict(mm, **{m: PAPER[m]})) - f(mm)), 4) for m in mm}, "what": "descriptive, never a bar"}, "table": T}

    # ---- bar 1
    pr_h = paper_read("N6H")
    b1 = pr_h["level"] == "PAPER LEVEL"
    # ---- bar 2
    changed = sorted(rep.get("N6H", []))
    harm = {m: paired("N6H", "N5A", m) for m in ("nc", "ddc", "ttc", "c", "ep", "dac")}
    b2_means = harm["nc"]["mean"] >= BAR["nc_ddc_floor_x100"] and harm["ddc"]["mean"] >= BAR["nc_ddc_floor_x100"]
    tok_viol = sorted(t for t in changed if arms["N6H"][t]["nc"] < arms["N5A"][t]["nc"] - BAR["tok_tol"] or arms["N6H"][t]["ddc"] < arms["N5A"][t]["ddc"] - BAR["tok_tol"])
    b2 = bool(b2_means and not tok_viol)
    # ---- bar 3
    agree = [t for t in changed if arms["N6H"][t]["dac"] == 1.0]
    b3 = bool(len(agree) == len(changed) and br["controls"].get("C-CERT-B6", {}).get("pass"))
    branches = [n for n, ok in (("paper_level_not_reached", b1), ("harm", b2), ("cert_integrity", b3)) if not ok]
    src = br.get("source_class", {}).get("N6H", {})
    prim_set = set(prim)
    n_teacher = sum(1 for t, c in src.items() if c == "teacher" and t in prim_set)
    n_human = sum(1 for t, c in src.items() if c == "human" and t in prim_set)
    srcT = br.get("source_class", {}).get("N6T", {})
    still = sorted(t for t in prim if arms["N6H"][t]["dac"] == 0.0)
    rt = br.get("residual_table", {})
    res["naming"] = NAMING.format(nt=n_teacher, nh=n_human)
    res["human_log_notice"] = "the human-log replacement is NOT teacher output; every number quoted for N6-H includes it, and N6-T is the teacher-only attribution"
    res["primary"] = {"contrast": "N6H vs N5A, harness, 136 logs", "naming": res["naming"],
                      "n_teacher_tol_replacements": n_teacher, "n_human_replacements": n_human, "n_changed_total": len(changed), "n_changed_by_N6T": len(srcT),
                      "paper_level_N6H": pr_h, "delta_vs_N5A_paired": {"pdms": paired("N6H", "N5A", "pdms"), "dac": harm["dac"]},
                      "no_harm_paired_x100_N6H_minus_N5A": {k: v for k, v in harm.items() if k != "dac"},
                      "no_harm_changed_token_violations": {"count": len(tok_viol), "tokens": tok_viol[:10]},
                      "certificate_integrity": {"changed": len(changed), "harness_dac1": len(agree), "mismatches": sorted(set(changed) - set(agree))[:10]},
                      "bars": {"1_paper_level_pdms_ge_95.8_and_dac_ge_99.9": bool(b1), "2_no_harm_nc_ddc": bool(b2), "3_certificate_integrity": bool(b3)},
                      "outcome": "CONFIRMED" if (b1 and b2 and b3) else "NOT CONFIRMED", "not_confirmed_branches_in_card_order": branches,
                      "tolerance_declared_ttc_c_reported": {"ttc": harm["ttc"], "c": harm["c"]}}
    res["paper_level"] = {"N6H": pr_h, "N6T_teacher_only": paper_read("N6T"), "N5A_comparator": paper_read("N5A"),
                          "rule": "PAPER LEVEL iff PDMS >= 95.8 AND DAC >= 99.9 (point estimates; <= 12 zero-DAC tokens); NEAR iff >= 95.3 AND >= 99.5; else BELOW"}
    res["secondary"] = {"decides": "nothing", "N6T_minus_N5A": paired("N6T", "N5A"), "N6H_minus_N6T": paired("N6H", "N6T"), "N6H_minus_N2R": paired("N6H", "N2R")}
    res["remaining_zero_dac_N6H"] = {"count": len(still), "tokens": still, "blocking": {t: rt.get(t, {}).get("blocking_class") for t in still}}
    res["families"] = "read in stage families: n6_nt_analyze.py famread (guard flags the student label only)"
    print("ZZN6_VERDICT PRIMARY", res["primary"]["outcome"], "bars", json.dumps(res["primary"]["bars"]), "branches", branches, flush=True)
    print("ZZN6_VERDICT PAPER", pr_h["level"], json.dumps(pr_h["pdms"]), json.dumps(pr_h["dac"]), "zero_dac", pr_h["zero_dac_tokens"], res["naming"], flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0


def cmd_famread(a) -> int:
    fc = json.load(open(a.fam_ci, encoding="utf-8"))
    fam = fc["frames"]["executed_tracked"]["families"]

    def pick(f, k):
        v = fam.get(f, {}).get(k, {})
        d = v.get("diff_B_minus_A") if isinstance(v, dict) else None
        return d if isinstance(d, dict) else {"point": d, "lo": None, "hi": None}
    chk = {"speed_bias_diff_mps": (pick("longitudinal", "speed_bias_mps"), GUARD["speed_bias_diff_mps"]),
           "speed_mae_diff_mps": (pick("longitudinal", "speed_mae_mps"), GUARD["speed_mae_diff_mps"]),
           "cross_mae_diff_m": (pick("lateral", "cross_mae_m"), GUARD["cross_mae_diff_m"])}
    if all(v["point"] is None for v, _b in chk.values()):
        raise SystemExit("REFUSED: fam_ci output carries no paired diff for any guard metric (an old families_run.py, or a changed fam_ci layout): no guard read")
    out = {k: {"diff": v["point"], "ci": [v["lo"], v["hi"]], "bound": b, "exceeded": bool(v["point"] is None or v["point"] > b)} for k, (v, b) in chk.items()}
    res = {"basis": "executed_tracked, W-TRANSP on the harness metric cache, N6-H minus N5-A, paired log-cluster bootstrap (fam_ci.py, B = N6H); the N6-H rows include the human-log replacements",
           "strategic": "PAUSED (PI 2026-09-19)", "K": a.k, "guard": {"checks": out, "flag": any(x["exceeded"] for x in out.values())},
           "reported_not_bars": {f"{fam_}.{k}": pick(fam_, k) for fam_, k in REPORTED}}
    print("ZZN6_GUARD", "FLAG: label-quality regression, goes to the PI" if res["guard"]["flag"] else "no flag", json.dumps(out), flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("read", "famread"))
    ap.add_argument("--harness-csv", nargs="*", default=[])
    ap.add_argument("--build-report")
    ap.add_argument("--unch", default="")
    ap.add_argument("--pins", default="")
    ap.add_argument("--pin-root", default=".")
    ap.add_argument("--proxy-eval", default="")
    ap.add_argument("--prereg-sha256", default="")
    ap.add_argument("--token-log")
    ap.add_argument("--known36", default="")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--a1-analyze", default="")
    ap.add_argument("--expect-n", type=int, default=12146)
    ap.add_argument("--expect-primary", type=int, default=12110)
    ap.add_argument("--id-tol", type=float, default=BAR["id_tol"])
    ap.add_argument("--fam-ci", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return evaluate(a) if a.mode == "read" else cmd_famread(a)


if __name__ == "__main__":
    sys.exit(main())
