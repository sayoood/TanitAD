#!/usr/bin/env python3
"""TF-N5 navtest readout (PREREG_TF_N5_DRAFT.md sections 6, 7, 8, 9). Reads ONLY harness CSVs (the pinned W3 harness), the build report and the control files.

  read     python n5_nt_analyze.py read --harness-csv N2R=<csv> N4D=<csv> N5A=<csv> N5X=<csv> --build-report <n5_build_report.json> --unch <unch.json>
               --pins <N5_PINS.json> --pin-root <run> --proxy-eval <eval dir> --prereg-sha256 <sha> --token-log <navtest_inputs.json.gz> --known36 <json>
               --residual <residual25.json> --k 47 --a1-analyze <tf_n1_analyze_a1.py> --out <json>
           CONTROLS FIRST (C-COV, C-BUILD, C-UNCH, C-PIN, C-CERT, C-ID-C, C-EST, C-REG). If any fails it prints ZZN5_CONTROLS FAIL, writes a file with NO aggregate and
           exits 1 BEFORE any N5-A number is computed. Then the section 6 bars (PRIMARY 1 to 4, paper level, secondary contrasts).
  famread  python n5_nt_analyze.py famread --fam-ci <fam_ci json N5A vs N4D> --out <json>     (the four-family guard; decides only the student-label flag)

PRIMARY: L1 CONFIRMED iff (1) >= 5 of the 20 non-guard residual tokens read DAC = 1 in N5-A (count + exact Clopper-Pearson interval; the log bootstrap is NOT the
estimator of this bar), (2) paired dNC, dDDC, dTTC, dC, dEP each >= -0.10 (x100) AND no token outside the L1-applied set differs from banked N4-D, (3) certificate
integrity (builder C-CERT-B and harness C-CERT), (4) no token that N4-D had at DAC = 1 reads DAC = 0 in N5-A. dPDMS is reported with its paired interval and is NOT a bar.
Estimator for everything else: `Boot` of tf_n1_analyze_a1.py (the draw scheme of eval/proxy_eval.boot; C-EST checks it): paired log-cluster bootstrap over the logs,
10,000 draws, seed 20260927, 95 % percentile, plus the Bonferroni interval at K (47). It answers "would another draw of LOGS say this?" only."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys

import numpy as np

BAR = {"repair_count": 5, "n_non_guard": 20, "no_harm_x100": -0.10, "paper_pdms": 95.8, "paper_dac": 99.9, "near_pdms": 95.3, "near_dac": 99.5,
       "id_tol": 1e-6, "reg_max_repair_share": 0.10, "reg_pdms_slack": 0.05}
GUARD = {"speed_bias_diff_mps": 0.25, "speed_mae_diff_mps": 0.15, "cross_mae_diff_m": 0.05}
MET = ("nc", "dac", "ddc", "ep", "ttc", "c", "pdms")
CSV_COL = {"nc": "no_at_fault_collisions", "dac": "drivable_area_compliance", "ep": "ego_progress", "ttc": "time_to_collision_within_bound",
           "c": "comfort", "ddc": "driving_direction_compliance", "pdms": "score"}
NEW_ARMS = ("N5A", "N5X")
NEED = ("N2R", "N4D", "N5A", "N5X")
TIP_FAMILIES = "w/pk/eval/instruments/families_run.py"
TIP_FAMILIES_SHA256 = "44c62807f1254e2ca4a41e4094097071b96bccaba9e754703ddf4a689e4d73a6"
REPORTED = (("lateral", "heading_mae_deg"), ("lateral", "curvature_mae_1pm"), ("lateral", "yaw_rate_mae_degps"), ("lateral", "cross_bias_m"),
            ("longitudinal", "accel_mae_mps2"), ("longitudinal", "along_mae_m"), ("longitudinal", "headway_min_m"), ("longitudinal", "min_ttc_s"),
            ("tactical", "maneuver_5way_collapsed.accuracy"), ("tactical", "lateral_decision.accuracy"), ("tactical", "longitudinal_decision.accuracy"),
            ("tactical", "goal_setting.goal_point_error_m"))        # reported per family, never pooled (card section 6); decide nothing
PAPER = {"nc": 0.998, "dac": 0.999, "ep": 0.915, "ttc": 0.991, "c": 0.990}


def load_a1(path: str):
    spec = importlib.util.spec_from_file_location("tf_n1_analyze_a1_n5", path)
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)
    return A


def sha256(p: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_csv(p: str) -> tuple[dict, int]:
    out, bad = {}, 0
    with open(p, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["token"] == "average":
                continue
            if r.get("valid", "True") != "True":
                bad += 1
                continue
            out[r["token"]] = {m: float(r[CSV_COL[m]]) for m in MET}
    return out, bad


def load_tlog(p: str) -> dict:
    import gzip
    op = gzip.open if p.endswith(".gz") else open
    d = json.load(op(p, "rt", encoding="utf-8"))
    d = d.get("tokens", d)
    return {t: (v["log_name"] if isinstance(v, dict) else v) for t, v in d.items()}


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list:
    """exact binomial interval for k of n (the estimator of the repair-count bar)"""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def pins_check(pins_path: str, pin_root: str) -> dict:
    """every pinned file matches on disk AND the pinned families_run.py is the TIP version (the TF-N4 A1 families stage failed on an older one)"""
    pins = json.load(open(pins_path, encoding="utf-8"))["files"]
    got = {k: (sha256(os.path.join(pin_root, k)) if os.path.exists(os.path.join(pin_root, k)) else None) for k in pins}
    bad = [k for k in pins if got[k] != pins[k]]
    fam_ok = pins.get(TIP_FAMILIES) == TIP_FAMILIES_SHA256 and got.get(TIP_FAMILIES) == TIP_FAMILIES_SHA256
    return {"pass": not bad and fam_ok, "mismatch": bad[:10], "n": len(pins), "families_run_is_tip": fam_ok}


def evaluate(a, A=None) -> int:
    """the whole read; returns the exit code (0 = verdict written, 1 = a control failed and NO aggregate exists)"""
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
                 f"repair COUNT: exact Clopper-Pearson; everything else paired log-cluster bootstrap, 136 logs, 10,000 draws, seed 20260927, 95 percent percentile "
                 f"(eval/proxy_eval.boot draws); Bonferroni at K={a.k}"}
    C = res["controls"]
    br = json.load(open(a.build_report, encoding="utf-8"))
    rep = br["repaired"]
    resid = json.load(open(a.residual, encoding="utf-8"))
    l1_set, non_guard = set(resid["all_25"]), list(resid["non_guard"])
    C["C-COV"] = {"pass": len(allt) == a.expect_n and len(prim) == a.expect_primary and all(n_bad[k] == 0 and len(arms[k]) == a.expect_n for k in NEED),
                  "tokens": len(allt), "expect": a.expect_n, "primary": len(prim), "expect_primary": a.expect_primary,
                  "invalid_or_error_rows": {k: v for k, v in n_bad.items() if v}}
    C["C-BUILD"] = {"pass": br.get("verdict") == "PASS", "controls": {k: v.get("pass") for k, v in br.get("controls", {}).items()}}
    un = json.load(open(a.unch, encoding="utf-8")) if a.unch else {}
    C["C-UNCH"] = {"pass": bool(un.get("pass")), "arms": {k: v.get("pass") for k, v in un.get("arms", {}).items()}, "n_l1_run": un.get("n_l1_run"),
                   "what": "plan20 and seam files on disk, bit-identical to banked N4-D off the L1-applied set; L1-run count == 25"}
    C["C-PIN"] = pins_check(a.pins, a.pin_root) if a.pins else {"pass": False, "why": "no --pins"}
    cert_bad = {arm: [t for t in rep.get(arm, []) if t not in arms[arm] or arms[arm][t]["dac"] != 1.0] for arm in NEW_ARMS}
    C["C-CERT"] = {"pass": not any(cert_bad.values()), "exceptions": {k: len(v) for k, v in cert_bad.items()}, "examples": {k: v[:3] for k, v in cert_bad.items()},
                   "certified": {k: len(rep.get(k, [])) for k in NEW_ARMS}, "what": "every token the builder L1-certified (DAC = 1) has harness DAC = 1"}
    idc, worst = {}, {m: 0.0 for m in MET}
    sentinel = sorted(allt, key=lambda t: __import__("hashlib").sha1(t.encode()).hexdigest())[:200]
    sent_over = {}
    for arm in NEW_ARMS:
        r = set(rep.get(arm, []))
        over, sover = 0, 0
        for t in allt:
            if t in r:
                continue
            d = max(abs(arms[arm][t][m] - arms["N4D"][t][m]) for m in MET)
            for m in MET:
                worst[m] = max(worst[m], abs(arms[arm][t][m] - arms["N4D"][t][m]))
            over += d > a.id_tol
        for t in sentinel:
            if t not in r:
                sover += max(abs(arms[arm][t][m] - arms["N4D"][t][m]) for m in MET) > a.id_tol
        idc[arm] = {"unchanged_tokens": len(allt) - len(r), "tokens_over_tol": int(over)}
        sent_over[arm] = int(sover)
    C["C-ID-C"] = {"pass": all(v["tokens_over_tol"] == 0 for v in idc.values()) and not any(sent_over.values()), "tol": a.id_tol, "arms": idc,
                   "sentinel_200_over_tol": sent_over, "max_abs_diff_by_metric": worst,
                   "what": "full set: every token outside the L1-applied set has a harness row equal to the banked N4-D row on all 7 metrics (incl. the 200 smallest-sha1 sentinels)"}
    bt = A.Boot(prim, tl)
    if a.proxy_eval:
        spec = importlib.util.spec_from_file_location("proxy_eval_n5", os.path.join(a.proxy_eval, "proxy_eval.py"))
        PE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(PE)
        pt = {t: arms["N2R"][t]["pdms"] for t in prim}
        ref, ours = PE.boot(pt, tl), bt.stat(pt)["_raw"]
        dev = max(abs(x - y) for x, y in zip(ref, ours))
        C["C-EST"] = {"pass": dev <= 1e-9, "max_abs_dev": dev}
    else:
        C["C-EST"] = {"pass": False, "why": "no --proxy-eval"}
    x_fix = sorted(t for t in l1_set if t in arms["N5X"] and arms["N5X"][t]["dac"] == 1.0)
    pX, pD = 100 * np.mean([arms["N5X"][t]["pdms"] for t in prim]), 100 * np.mean([arms["N4D"][t]["pdms"] for t in prim])
    C["C-REG"] = {"pass": bool(len(x_fix) <= BAR["reg_max_repair_share"] * len(l1_set) and pX <= pD + BAR["reg_pdms_slack"]),
                  "N5X_repaired_of_l1_set": len(x_fix), "l1_set": len(l1_set), "N5X_pdms": round(float(pX), 4), "N4D_pdms": round(float(pD), 4),
                  "bar": "N5X repairs <= 10 % of the 25 and PDMS(N5X) <= PDMS(N4D) + 0.05"}
    allpass = all(c["pass"] for c in C.values())
    print("ZZN5_CONTROLS", "PASS" if allpass else "FAIL", json.dumps({k: v["pass"] for k, v in C.items()}), flush=True)
    if not allpass:
        res["verdict"] = "NO VERDICT: a control failed; no N5-A aggregate was computed"
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
        return 1

    def tab(arm, toks):
        out = {m: {k: v for k, v in bt.stat({t: arms[arm][t][m] for t in toks}).items() if k != "_raw"} for m in MET}
        out["zero_rows"] = {m: int(sum(arms[arm][t][m] == 0 for t in toks)) for m in ("pdms", "dac", "nc")}
        return out

    def paired(x, y, toks=None, metric="pdms"):
        return {k: v for k, v in bt.stat({t: arms[x][t][metric] - arms[y][t][metric] for t in (toks or prim)}).items() if k != "_raw"}

    # ---- PRIMARY
    ng_dac1 = sorted(t for t in non_guard if arms["N5A"][t]["dac"] == 1.0)
    cnt = len(ng_dac1)
    b1 = cnt >= BAR["repair_count"]
    harm = {m: paired("N5A", "N4D", metric=m) for m in ("nc", "ddc", "ttc", "c", "ep")}
    b2 = all(v["mean"] >= BAR["no_harm_x100"] for v in harm.values()) and C["C-ID-C"]["pass"]
    b3 = bool(C["C-CERT"]["pass"] and br["controls"].get("C-CERT-B", {}).get("pass"))
    new_fail = sorted(t for t in allt if arms["N4D"][t]["dac"] == 1.0 and arms["N5A"][t]["dac"] != 1.0)
    b4 = not new_fail
    branches = [n for n, ok in (("repair_count_below_5", b1), ("harm", b2), ("cert_integrity", b3), ("new_failure", b4)) if not ok]
    all25 = sorted(t for t in l1_set if arms["N5A"][t]["dac"] == 1.0)
    res["primary"] = {"contrast": "N5A - N4D, harness, 136 logs (RJ5 teacher + P5-R + REPAIR-D + L1 on the REPAIR-D residual with a scorer no-harm guard)",
                      "repair_count": {"non_guard_dac1": cnt, "of": len(non_guard), "exact_ci95": clopper_pearson(cnt, len(non_guard)), "bar": BAR["repair_count"],
                                       "tokens": ng_dac1, "of_all_25": len(all25), "paper_needs_of_25": 13},
                      "delta_pdms_not_a_bar": paired("N5A", "N4D"), "no_harm_paired_x100": harm,
                      "no_new_failure": {"count": len(new_fail), "tokens": new_fail[:10]},
                      "bars": {"1_repair_count_ge_5_of_20": bool(b1), "2_no_harm_each_ge_-0.10_and_identity": bool(b2), "3_certificate_integrity": bool(b3), "4_no_new_failure": bool(b4)},
                      "outcome": "CONFIRMED" if (b1 and b2 and b3 and b4) else "NOT CONFIRMED", "not_confirmed_branches_in_card_order": branches}

    def paper_read(arm):
        T = tab(arm, prim)
        p, dc = T["pdms"]["mean"], T["dac"]["mean"]
        lvl = "PAPER LEVEL" if p >= BAR["paper_pdms"] and dc >= BAR["paper_dac"] else "NEAR" if p >= BAR["near_pdms"] and dc >= BAR["near_dac"] else "BELOW"
        mm = {m: T[m]["mean"] / 100.0 for m in ("nc", "dac", "ep", "ttc", "c")}
        f = lambda v: v["nc"] * v["dac"] * (5 * v["ep"] + 5 * v["ttc"] + 2 * v["c"]) / 12.0     # noqa: E731
        return {"pdms": T["pdms"], "dac": T["dac"], "level": lvl, "zero_dac_tokens": T["zero_rows"]["dac"], "gap_to_paper_pdms": round(BAR["paper_pdms"] - p, 4),
                "decomposition_mean_field": {"swap_gain_x100": {m: round(100 * (f(dict(mm, **{m: PAPER[m]})) - f(mm)), 4) for m in mm}, "what": "descriptive, never a bar"},
                "table": T}
    res["paper_level"] = {"N5A": paper_read("N5A"), "N4D_comparator": paper_read("N4D"),
                          "rule": "PAPER LEVEL iff PDMS >= 95.8 AND DAC >= 99.9 (<= 12 zero-DAC tokens: L1 repairs >= 13 of the 25); NEAR iff >= 95.3 AND >= 99.5; else BELOW"}
    res["secondary"] = {"decides": "nothing", "N5A_minus_N2R": paired("N5A", "N2R"), "N4D_minus_N2R": paired("N4D", "N2R"),
                        "N5X_minus_N4D": paired("N5X", "N4D"), "n5r_builder_only": {"repaired_of_116": len(rep.get("N5R", [])), "n4d_repaired_of_116": 116 - 25,
                                                                                     "what": "builder-side counts only, decides nothing"}}
    res["families"] = "read in stage families: n5_nt_analyze.py famread (guard flags the student label only)"
    print("ZZN5_VERDICT PRIMARY", res["primary"]["outcome"], "repair_count", cnt, "of", len(non_guard), "branches", branches, flush=True)
    print("ZZN5_VERDICT PAPER", res["paper_level"]["N5A"]["level"], json.dumps(res["paper_level"]["N5A"]["pdms"]), flush=True)
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
    res = {"basis": "executed_tracked, W-TRANSP on the harness metric cache, N5-A minus N4-D, paired log-cluster bootstrap (fam_ci.py, B = N5A)", "strategic": "PAUSED (PI 2026-09-19)",
           "guard": {"checks": out, "flag": any(x["exceeded"] for x in out.values())},
           "reported_not_bars": {f"{fam_}.{k}": pick(fam_, k) for fam_, k in REPORTED}}
    print("ZZN5_GUARD", "FLAG: label-quality regression, goes to the PI" if res["guard"]["flag"] else "no flag", json.dumps(out), flush=True)
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
    ap.add_argument("--residual", default="")
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
