#!/usr/bin/env python3
"""TF-N4 navtest readout (PREREG_TF_N4.md sections 6, 7, 8, 10). Reads ONLY harness CSVs (the pinned W3 harness), the build report and the control files.

  read     python n4_nt_analyze.py read --harness-csv N2R=<csv> N4D=<csv> N4X=<csv> N4DM=<csv> N2C=<banked> N1C=<banked> T=<banked> [human=<banked>]
               --build-report <n4_build_report.json> --unch <unch.json> --pins <N4_PINS.json> --pin-root <run> --proxy-eval <eval dir>
               --prereg-sha256 <sha> --token-log <navtest_inputs.json.gz> --known36 <json> --k 45 --a1-analyze <tf_n1_analyze_a1.py> --out <json>
           CONTROLS FIRST (C-COV, C-BUILD, C-UNCH, C-PIN, C-CERT, C-ID-C, C-EST, C-REG). If any fails it prints ZZN4_CONTROLS FAIL, writes a file with NO
           aggregate and exits 1 BEFORE any N4-D number is computed. Then the section 6 bars (PRIMARY 1 to 4, paper level, secondary contrasts).
  famread  python n4_nt_analyze.py famread --fam-ci <fam_ci json N4D vs N2R> --out <json>     (the four-family guard; decides only the student-label flag)

Estimator: `Boot` of tf_n1_analyze_a1.py (the draw scheme of eval/proxy_eval.boot; C-EST checks it): paired log-cluster bootstrap over the logs, 10,000
draws, seed 20260927, 95 % percentile, plus the Bonferroni interval at K (passed in, 45 on the card). It answers "would another draw of LOGS say this?"
only; the plans, labels, repair and harness are deterministic (C-DET, C-ID-C)."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys

import numpy as np

BAR = {"confirm_delta": 0.50, "repair_rate": 0.75, "no_harm_x100": -0.10, "paper_pdms": 95.8, "paper_dac": 99.9, "near_pdms": 95.3, "near_dac": 99.5,
       "id_tol": 1e-6, "reg_max_repair_share": 0.10, "reg_pdms_slack": 0.05}
GUARD = {"speed_bias_diff_mps": 0.25, "speed_mae_diff_mps": 0.15, "cross_mae_diff_m": 0.05}
MET = ("nc", "dac", "ddc", "ep", "ttc", "c", "pdms")
CSV_COL = {"nc": "no_at_fault_collisions", "dac": "drivable_area_compliance", "ep": "ego_progress", "ttc": "time_to_collision_within_bound",
           "c": "comfort", "ddc": "driving_direction_compliance", "pdms": "score"}
NEW_ARMS = ("N4D", "N4X", "N4DM")
CERT_ORDER = ("N2R", "N2C", "N1C", "T")
PAPER = {"nc": 0.998, "dac": 0.999, "ep": 0.915, "ttc": 0.991, "c": 0.990}


def load_a1(path: str):
    spec = importlib.util.spec_from_file_location("tf_n1_analyze_a1_n4", path)
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


def cert2(arms: dict, t: str):
    for c in CERT_ORDER:
        if arms[c][t]["dac"] == 1.0:
            return c, False
    return "N2R", True


def evaluate(a, A=None) -> int:
    """the whole read; returns the exit code (0 = verdict written, 1 = a control failed and NO aggregate exists)"""
    A = A or load_a1(a.a1_analyze)
    A.K = a.k
    srcs = dict(s.split("=", 1) for s in a.harness_csv)
    need = ("N2R",) + NEW_ARMS + ("N2C", "N1C", "T")
    if any(k not in srcs for k in need):
        raise SystemExit(f"--harness-csv needs {need}")
    arms, n_bad = {}, {}
    for nm, fp in srcs.items():
        arms[nm], n_bad[nm] = load_csv(fp)
    tlog = load_tlog(a.token_log)
    allt = sorted(set.intersection(*[set(v) for v in arms.values()]))
    known = A.load_known36(a.known36)
    prim = [t for t in allt if t not in known]
    tl = {t: tlog[t] for t in allt}
    res: dict = {"prereg": a.prereg_sha256 or "UNSET", "K": a.k, "n_tokens": len(allt), "n_primary": len(prim), "controls": {}, "estimator":
                 f"paired log-cluster bootstrap, 136 logs, 10,000 draws, seed 20260927, 95 percent percentile (eval/proxy_eval.boot draws); Bonferroni at K={a.k}"}
    C = res["controls"]
    br = json.load(open(a.build_report, encoding="utf-8"))
    rep = br["repaired"]
    C["C-COV"] = {"pass": len(allt) == a.expect_n and len(prim) == a.expect_primary and all(n_bad[k] == 0 and len(arms[k]) == a.expect_n for k in need),
                  "tokens": len(allt), "expect": a.expect_n, "primary": len(prim), "expect_primary": a.expect_primary,
                  "invalid_or_error_rows": {k: v for k, v in n_bad.items() if v}}
    C["C-BUILD"] = {"pass": br.get("verdict") == "PASS", "controls": {k: v.get("pass") for k, v in br.get("controls", {}).items()}}
    un = json.load(open(a.unch, encoding="utf-8")) if a.unch else {}
    C["C-UNCH"] = {"pass": bool(un.get("pass")), "arms": {k: v.get("pass") for k, v in un.get("arms", {}).items()}, "what": "plan20 and seam files on disk, bit-identical to N2-R off the repaired set"}
    if a.pins:
        pins = json.load(open(a.pins, encoding="utf-8"))["files"]
        got = {k: (sha256(os.path.join(a.pin_root, k)) if os.path.exists(os.path.join(a.pin_root, k)) else None) for k in pins}
        C["C-PIN"] = {"pass": all(got[k] == pins[k] for k in pins), "mismatch": [k for k in pins if got[k] != pins[k]][:10], "n": len(pins)}
    else:
        C["C-PIN"] = {"pass": False, "why": "no --pins"}
    cert_bad = {}
    for arm in NEW_ARMS:
        cert_bad[arm] = [t for t in rep.get(arm, []) if t not in arms[arm] or arms[arm][t]["dac"] != 1.0]
    C["C-CERT"] = {"pass": not any(cert_bad.values()), "exceptions": {k: len(v) for k, v in cert_bad.items()}, "examples": {k: v[:3] for k, v in cert_bad.items()},
                   "certified": {k: len(v) for k, v in rep.items()}, "what": "every token the builder certified (DAC = 1) has harness DAC = 1"}
    idc, worst = {}, {m: 0.0 for m in MET}
    for arm in NEW_ARMS:
        r = set(rep.get(arm, []))
        over = 0
        for t in allt:
            if t in r:
                continue
            d = max(abs(arms[arm][t][m] - arms["N2R"][t][m]) for m in MET)
            for m in MET:
                worst[m] = max(worst[m], abs(arms[arm][t][m] - arms["N2R"][t][m]))
            over += d > a.id_tol
        idc[arm] = {"unchanged_tokens": len(allt) - len(r), "tokens_over_tol": int(over)}
    C["C-ID-C"] = {"pass": all(v["tokens_over_tol"] == 0 for v in idc.values()), "tol": a.id_tol, "arms": idc, "max_abs_diff_by_metric": worst,
                   "what": "full set: every unrepaired token's harness row equals the banked N2-R row on all 7 metrics"}
    bt = A.Boot(prim, tl)
    if a.proxy_eval:
        spec = importlib.util.spec_from_file_location("proxy_eval_n4", os.path.join(a.proxy_eval, "proxy_eval.py"))
        PE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(PE)
        pt = {t: arms["N2R"][t]["pdms"] for t in prim}
        ref, ours = PE.boot(pt, tl), bt.stat(pt)["_raw"]
        dev = max(abs(x - y) for x, y in zip(ref, ours))
        C["C-EST"] = {"pass": dev <= 1e-9, "max_abs_dev": dev}
    else:
        C["C-EST"] = {"pass": False, "why": "no --proxy-eval"}
    zero = [t for t in allt if arms["N2R"][t]["dac"] == 0.0]
    x_fix = [t for t in zero if arms["N4X"][t]["dac"] == 1.0]
    pX, pR = 100 * np.mean([arms["N4X"][t]["pdms"] for t in prim]), 100 * np.mean([arms["N2R"][t]["pdms"] for t in prim])
    C["C-REG"] = {"pass": bool(len(x_fix) <= BAR["reg_max_repair_share"] * len(zero) and pX <= pR + BAR["reg_pdms_slack"]),
                  "N4X_repaired_zero_dac": len(x_fix), "N2R_zero_dac": len(zero), "N4X_pdms": round(float(pX), 4), "N2R_pdms": round(float(pR), 4),
                  "bar": "N4X repairs <= 10 % of the N2-R zero-DAC tokens and PDMS(N4X) <= PDMS(N2R) + 0.05"}
    allpass = all(c["pass"] for c in C.values())
    print("ZZN4_CONTROLS", "PASS" if allpass else "FAIL", json.dumps({k: v["pass"] for k, v in C.items()}), flush=True)
    if not allpass:
        res["verdict"] = "NO VERDICT: a control failed; no N4-D aggregate was computed"
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
        return 1

    def tab(arm, toks):
        out = {m: {k: v for k, v in bt.stat({t: arms[arm][t][m] for t in toks}).items() if k != "_raw"} for m in MET}
        out["zero_rows"] = {m: int(sum(arms[arm][t][m] == 0 for t in toks)) for m in ("pdms", "dac", "nc")}
        return out

    def paired(x, y, toks=None, metric="pdms", src=None):
        S = src or arms
        return {k: v for k, v in bt.stat({t: S[x][t][metric] - S[y][t][metric] for t in (toks or prim)}).items() if k != "_raw"}

    # derived arms: CERT2 (banked N2R / N2C / N1C / T order) and N4-P2 (CERT2's choice; on none-pass tokens REPAIR-D of N2-R = the N4-D row, exactly)
    choice = {t: cert2(arms, t) for t in allt}
    arms["CERT2"] = {t: dict(arms[choice[t][0]][t]) for t in allt}
    arms["N4P2"] = {t: dict(arms["N4D"][t] if choice[t][1] else arms["CERT2"][t]) for t in allt}
    if "human" in srcs:
        pass
    TP = {arm: tab(arm, prim) for arm in ("N2R", "N4D", "N4X", "N4DM", "CERT2", "N4P2") + (("human",) if "human" in arms else ())}
    d = paired("N4D", "N2R")
    zp = [t for t in prim if arms["N2R"][t]["dac"] == 0.0]
    fixed = [t for t in zp if arms["N4D"][t]["dac"] == 1.0]
    rate = len(fixed) / len(zp) if zp else float("nan")
    harm = {m: paired("N4D", "N2R", metric=m) for m in ("nc", "ddc", "ttc", "c", "ep")}
    b1 = d["mean"] >= BAR["confirm_delta"]
    b2 = d["ci95"][0] > 0
    b3 = bool(zp) and rate >= BAR["repair_rate"]
    b4 = all(v["mean"] >= BAR["no_harm_x100"] for v in harm.values()) and C["C-ID-C"]["pass"]
    branches = [n for n, ok in (("repair_rate_below_75", b3), ("harm", b4), ("effect_below_bar_or_ci", b1 and b2)) if not ok]
    res["primary"] = {"contrast": "N4D - N2R, paired PDMS x100, harness, primary 12,110 (RJ5 teacher + P5-R + map-certified DAC repair with a scorer no-harm guard)",
                      "delta": d, "bars": {"1_delta_ge_0.50": bool(b1), "2_ci_lo_gt_0": bool(b2), "3_repair_rate_ge_0.75": bool(b3), "4_no_harm_each_ge_-0.10": bool(b4)},
                      "repair": {"n2r_zero_dac_primary": len(zp), "n4d_dac1_among_them": len(fixed), "rate": round(rate, 4) if zp else None},
                      "no_harm_paired_x100": harm, "outcome": "CONFIRMED" if (b1 and b2 and b3 and b4) else "NOT CONFIRMED",
                      "not_confirmed_branches_in_card_order": branches}

    def paper_read(arm):
        p, dc = TP[arm]["pdms"]["mean"], TP[arm]["dac"]["mean"]
        lvl = ("PAPER LEVEL" if p >= BAR["paper_pdms"] and dc >= BAR["paper_dac"] else "NEAR" if p >= BAR["near_pdms"] and dc >= BAR["near_dac"] else "BELOW")
        mm = {m: TP[arm][m]["mean"] / 100.0 for m in ("nc", "dac", "ep", "ttc", "c")}
        f = lambda v: v["nc"] * v["dac"] * (5 * v["ep"] + 5 * v["ttc"] + 2 * v["c"]) / 12.0     # noqa: E731
        base = f(mm)
        zr = TP[arm]["zero_rows"]["dac"]
        return {"pdms": TP[arm]["pdms"], "dac": TP[arm]["dac"], "level": lvl, "zero_dac_tokens": zr, "gap_to_paper_pdms": round(BAR["paper_pdms"] - p, 4),
                "decomposition_mean_field": {"swap_gain_x100": {m: round(100 * (f(dict(mm, **{m: PAPER[m]})) - base), 4) for m in mm}, "what": "descriptive, never a bar"}}
    res["paper_level"] = {"N4D": paper_read("N4D"), "N4DM_map_only_form_secondary": paper_read("N4DM"),
                          "rule": "PAPER LEVEL iff PDMS >= 95.8 AND DAC >= 99.9; NEAR iff >= 95.3 AND >= 99.5; else BELOW"}
    res["tables"] = {"primary_12110": TP}
    sec = {"decides": "nothing", "N4D_minus_N4DM__value_of_the_future_aware_no_harm_guard": paired("N4D", "N4DM"),
           "N4DM_minus_N2R": paired("N4DM", "N2R"), "N4P2_minus_N2R": paired("N4P2", "N2R"), "N4P2_minus_CERT2": paired("N4P2", "CERT2"),
           "N4D_minus_CERT2": paired("N4D", "CERT2"), "CERT2_minus_N2R": paired("CERT2", "N2R"),
           "N4P2_row_rule": "CERT2's banked row where a candidate passes DAC, else the N4-D row (REPAIR-D of N2-R is N4-D's plan on those tokens): derived, not separately scored",
           "N4DM_repair": {"repaired": len(rep["N4DM"]), "N4D_repaired": len(rep["N4D"]), "N4X_repaired": len(rep["N4X"])}}
    if "human" in arms:
        hum = TP["human"]["pdms"]["mean"]
        sec["human_anchored"] = {"our_human": hum, "paper_human": A.PAPER_HUMAN_PDMS, "human_anchored_paper_level_pdms": round(BAR["paper_pdms"] - (A.PAPER_HUMAN_PDMS - hum), 4),
                                 "N4D_minus_human": paired("N4D", "human")}
    res["secondary"] = sec
    res["families"] = "read in stage families: n4_nt_analyze.py famread (guard flags the student label only)"
    print("ZZN4_VERDICT PRIMARY", res["primary"]["outcome"], json.dumps(d), "repair_rate", res["primary"]["repair"]["rate"], "branches", branches, flush=True)
    print("ZZN4_VERDICT PAPER", res["paper_level"]["N4D"]["level"], json.dumps(res["paper_level"]["N4D"]["pdms"]), flush=True)
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
    out = {k: {"diff": v["point"], "ci": [v["lo"], v["hi"]], "bound": b, "exceeded": bool(v["point"] is None or v["point"] > b)} for k, (v, b) in chk.items()}
    res = {"basis": "executed_tracked, W-TRANSP on the harness metric cache, N4-D minus N2-R, paired log-cluster bootstrap (fam_ci.py)", "strategic": "PAUSED (PI 2026-09-19)",
           "guard": {"checks": out, "flag": any(x["exceeded"] for x in out.values())},
           "reported_not_bars": {"heading_mae_deg": pick("lateral", "heading_mae_deg"), "curvature_mae_1pm": pick("lateral", "curvature_mae_1pm")}}
    print("ZZN4_GUARD", "FLAG: label-quality regression, goes to the PI" if res["guard"]["flag"] else "no flag", json.dumps(out), flush=True)
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
