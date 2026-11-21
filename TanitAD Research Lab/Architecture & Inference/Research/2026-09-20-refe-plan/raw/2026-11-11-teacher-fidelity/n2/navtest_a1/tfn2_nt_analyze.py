#!/usr/bin/env python3
"""TF-N2 navtest readout (PREREG_TF_N2.md). Reads ONLY harness CSVs (the pinned W3 harness, `score_p4_a1.py`) and the control files; no DB, no scorer.

  read     python tfn2_nt_analyze.py read --harness-csv N2R=<csv> N2C=<csv> N2W=<csv> N1C=<banked csv> T=<re-scored T csv> --identity-csv <banked T csv>
               --human-csv <banked human csv> --known36 <json> --build-report <json> --ctl-r0 <json> --cidc <json> --pins <N2_PINS.json> --pin-root <run>
               --proxy-eval <eval dir> --prereg-sha256 <sha> --token-log <navtest_inputs.json.gz> --out <json>
           CONTROLS FIRST (C-COV, C-BUILD, C-PIN, C-EST, C-ID-H, C-ID-C, C-R0, C-REG). If any fails it prints ZZN2_CONTROLS FAIL and exits 1 BEFORE any
           N2R / N2C / CERT2 aggregate is computed. Then the registered outcomes (ZZN2_VERDICT lines).
  famprep  python tfn2_nt_analyze.py famprep --rows "<transp rows glob>" --known36 <json> --out <rows_primary.jsonl>
  famread  python tfn2_nt_analyze.py famread --fam N2R=<fam json> N2C=.. N1C=.. human=.. --out <json>      (the longitudinal guard; descriptive, never a bar)

Estimator: `Boot` of tf_n1_analyze_a1.py (the draw scheme of eval/proxy_eval.boot, reproduced exactly; C-EST checks it): paired log-cluster bootstrap over the
logs, 10,000 draws, seed 20260927, 95 % percentile, plus the Bonferroni interval at K = 43 (this card's ledger entry). It answers "another draw of LOGS?"
only: plans, labels, tracker and harness are deterministic (policy argmax, CPU rollouts), so no training or inference seed applies.
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("tf_n1_analyze_a1", os.path.join(HERE, "tf_n1_analyze_a1.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
K = 44
A.K = K                                              # Boot.stat reads the module global: the Bonferroni interval is at K = 43

MET, CSV_COL, PAPER = A.MET, A.CSV_COL, A.PAPER
CERT_ORDER = ("N2R", "N2C", "N1C", "T")              # card section 4; none-pass -> N2R
ARMS = ("N2R", "N2C", "N2W", "N1C", "T", "human", "CERT2")
BAR = {"confirm_delta": 0.5, "confirm_dac_loss": 0.25, "paper_pdms": 95.8, "paper_dac": 99.9, "near_pdms": 95.3, "near_dac": 99.5,
       "reg_margin": 10.0, "id_tol": 1e-6, "r0_tol_m": 1e-6}
GUARD = {"speed_bias_vs_n1c_mps": 0.50, "speed_mae_vs_n1c_mps": 0.30, "accel_mae_vs_n1c_mps2": 0.15, "speed_bias_abs_mps": 1.00}


def load_csv(p):
    out = {}
    for row in csv.DictReader(open(p, encoding="utf-8")):
        if row["token"] != "average" and row.get("valid", "True") == "True":
            out[row["token"]] = {m: float(row[CSV_COL[m]]) for m in MET}
    return out


def load_tlog(p):
    import gzip
    d = json.load(gzip.open(p, "rt", encoding="utf-8"))
    d = d.get("tokens", d)
    return {t: (v["log_name"] if isinstance(v, dict) else v) for t, v in d.items()}


def cert2(per_arm: dict, t: str):
    for c in CERT_ORDER:
        if per_arm[c][t]["dac"] == 1.0:
            return c, False
    return "N2R", True


def cmd_read(a) -> int:
    srcs = dict(s.split("=", 1) for s in a.harness_csv)
    need = ("N2R", "N2C", "N2W", "N1C", "T")
    if any(k not in srcs for k in need):
        raise SystemExit(f"--harness-csv needs {need}")
    arms = {nm: load_csv(fp) for nm, fp in srcs.items()}
    if a.human_csv:
        arms["human"] = load_csv(a.human_csv)
    tlog = load_tlog(a.token_log)
    allt = sorted(set.intersection(*[set(v) for v in arms.values()]))
    n_err = max(len(v) for v in arms.values()) - len(allt)
    known = A.load_known36(a.known36)
    prim = [t for t in allt if t not in known]
    tl = {t: tlog[t] for t in allt}
    for t in allt:
        ch, none = cert2(arms, t)
        arms.setdefault("CERT2", {})[t] = dict(arms[ch][t])
    choice = {t: cert2(arms, t) for t in allt}
    res: dict = {"prereg": a.prereg_sha256 or "UNSET", "n_tokens": len(allt), "n_error_lines": n_err, "n_primary": len(prim), "controls": {}, "K": K}
    C = res["controls"]
    C["C-COV"] = {"pass": len(allt) == a.expect_n and n_err == 0 and len(prim) == a.expect_primary and all(x in arms for x in ARMS),
                  "tokens": len(allt), "expect": a.expect_n, "primary": len(prim), "expect_primary": a.expect_primary, "error_lines": n_err}
    br = json.load(open(a.build_report, encoding="utf-8")) if a.build_report else {}
    C["C-BUILD"] = {"pass": br.get("verdict") == "PASS", "controls": {k: v.get("pass") for k, v in br.get("controls", {}).items()}}
    if a.pins:
        pins = json.load(open(a.pins, encoding="utf-8"))["files"]
        got = {k: (A.sha256(os.path.join(a.pin_root, k)) if os.path.exists(os.path.join(a.pin_root, k)) else None) for k in pins}
        C["C-PIN"] = {"pass": all(got[k] == pins[k] for k in pins), "mismatch": [k for k in pins if got[k] != pins[k]][:10]}
    else:
        C["C-PIN"] = {"pass": False, "why": "no --pins"}
    bt = A.Boot(prim, tl)
    if a.proxy_eval:
        spec = importlib.util.spec_from_file_location("proxy_eval_n2", os.path.join(a.proxy_eval, "proxy_eval.py"))
        PE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(PE)
        pt = {t: arms["T"][t]["pdms"] for t in prim}
        ref, ours = PE.boot(pt, tl), bt.stat(pt)["_raw"]
        dev = max(abs(x - y) for x, y in zip(ref, ours))
        C["C-EST"] = {"pass": dev <= 1e-9, "max_abs_dev": dev}
    else:
        C["C-EST"] = {"pass": False, "why": "no --proxy-eval"}
    idref = load_csv(a.identity_csv)
    wm = {m: 0.0 for m in MET}
    n_over = miss = 0
    for t in allt:
        if t not in idref:
            miss += 1
            continue
        d = {m: abs(arms["T"][t][m] - idref[t][m]) for m in MET}
        for m in MET:
            wm[m] = max(wm[m], d[m])
        n_over += max(d.values()) > a.id_tol
    C["C-ID-H"] = {"pass": miss == 0 and n_over == 0 and bool(idref), "tol": a.id_tol, "missing_in_reference": miss, "tokens_over_tol": n_over,
                   "max_abs_diff_by_metric": wm, "what": "the harness re-score of the registered T seam against the banked teacher CSV"}
    cc = json.load(open(a.cidc, encoding="utf-8")) if a.cidc else {}
    C["C-ID-C"] = {"pass": bool(cc.get("pass")), **{k: cc.get(k) for k in ("tokens_rescored", "tokens_over_tol", "token_set_mismatch")},
                   "what": "the re-derived N1-C plans (200-token subset), harness re-score, against the banked N1-C CSV"}
    r0 = json.load(open(a.ctl_r0, encoding="utf-8")) if a.ctl_r0 else {}
    C["C-R0"] = {"pass": bool(r0.get("PASS")), **{k: r0.get(k) for k in ("n", "max_abs_m", "bar")}, "what": "R0 re-roll vs the banked teacher rollouts"}
    pC = 100 * np.mean([arms["N1C"][t]["pdms"] for t in prim])
    pW = 100 * np.mean([arms["N2W"][t]["pdms"] for t in prim])
    C["C-REG"] = {"pass": bool(pW <= pC - a.reg_margin), "N2W_pdms": round(float(pW), 4), "N1C_pdms": round(float(pC), 4),
                  "bar": f"PDMS(N2W) <= PDMS(N1C) - {a.reg_margin}"}
    if a.human_csv:
        href = load_csv(a.human_csv)
        dd = [max(abs(arms["human"][t][m] - href[t][m]) for m in MET) for t in allt if t in href]
        res["diag_human_vs_banked"] = {"n": len(dd), "tokens_over_tol": int(sum(x > a.id_tol for x in dd)), "decides": "nothing"}
    allpass = all(c["pass"] for c in C.values())
    print("ZZN2_CONTROLS", "PASS" if allpass else "FAIL", json.dumps({k: v["pass"] for k, v in C.items()}), flush=True)
    if not allpass:
        res["verdict"] = "NO VERDICT: a control failed; no N2R / N2C / CERT2 aggregate was computed"
        json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
        return 1

    def tab(arm, toks, bb):
        out = {m: {k: v for k, v in bb.stat({t: arms[arm][t][m] for t in toks}).items() if k != "_raw"} for m in MET}
        out["zero_rows"] = {m: int(sum(arms[arm][t][m] == 0 for t in toks)) for m in ("pdms", "dac", "nc")}
        return out

    def paired(x, y, toks, bb, metric="pdms"):
        return {k: v for k, v in bb.stat({t: arms[x][t][metric] - arms[y][t][metric] for t in toks}).items() if k != "_raw"}

    bt_all = A.Boot(allt, tl)
    res["scorer_of_record"] = "pinned W3 harness CSVs (score_p4_a1.py -> score_e.py); N1-C and the human are the banked N1-A1 / NAVSIM CSVs"
    res["tables"] = {"primary_12110": {arm: tab(arm, prim, bt) for arm in ARMS},
                     "all_12146_sensitivity": {arm: tab(arm, allt, bt_all) for arm in ("N2R", "N1C", "T", "CERT2")}}
    TP = res["tables"]["primary_12110"]
    d = paired("N2R", "N1C", prim, bt)
    dd = paired("N2R", "N1C", prim, bt, "dac")
    dacR = TP["N2R"]["dac"]["mean"]
    c1, c2, c3 = d["mean"] >= BAR["confirm_delta"], d["ci95"][0] > 0, dd["mean"] >= -BAR["confirm_dac_loss"]
    res["primary"] = {"contrast": "N2R - N1C, paired PDMS x100, harness, primary 12,110", "delta": d, "dac_delta": dd, "N2R_pdms": TP["N2R"]["pdms"],
                      "N1C_pdms": TP["N1C"]["pdms"], "N2R_dac": dacR,
                      "rule": "CONFIRMED iff delta >= +0.5 AND ci95 lo > 0 AND dac_delta >= -0.25",
                      "parts": {"delta_ge_0.5": bool(c1), "ci_lo_gt_0": bool(c2), "dac_delta_ge_-0.25": bool(c3)},
                      "outcome": "CONFIRMED" if (c1 and c2 and c3) else "NOT CONFIRMED",
                      "sensitivity_all_12146": paired("N2R", "N1C", allt, bt_all)}

    def paper_read(arm):
        p, dc = TP[arm]["pdms"]["mean"], TP[arm]["dac"]["mean"]
        lvl = ("PAPER LEVEL" if p >= BAR["paper_pdms"] and dc >= BAR["paper_dac"] else
               "NEAR" if p >= BAR["near_pdms"] and dc >= BAR["near_dac"] else "BELOW")
        mm = {m: TP[arm][m]["mean"] / 100.0 for m in ("nc", "dac", "ep", "ttc", "c")}
        f = lambda v: v["nc"] * v["dac"] * (5 * v["ep"] + 5 * v["ttc"] + 2 * v["c"]) / 12.0     # noqa: E731
        base = f(mm)
        return {"pdms": TP[arm]["pdms"], "dac": TP[arm]["dac"], "level": lvl, "gap_to_paper_pdms": round(BAR["paper_pdms"] - p, 4),
                "decomposition_mean_field": {"formula_on_means_x100": round(100 * base, 4), "residual_actual_minus_formula": round(p - 100 * base, 4),
                                             "swap_gain_x100": {m: round(100 * (f(dict(mm, **{m: PAPER[m]})) - base), 4) for m in mm},
                                             "what": "descriptive, never a bar"}}
    res["paper_level"] = {"N2R": paper_read("N2R"), "rule": "PAPER LEVEL iff PDMS >= 95.8 AND DAC >= 99.9; NEAR iff >= 95.3 AND >= 99.5; else BELOW"}
    hum = TP["human"]["pdms"]["mean"] if "human" in TP else None
    if hum is not None:
        res["paper_level"]["human_anchored_descriptive"] = {"our_human": hum, "paper_human": A.PAPER_HUMAN_PDMS,
                                                            "human_anchored_paper_level": round(BAR["paper_pdms"] - (A.PAPER_HUMAN_PDMS - hum), 4),
                                                            "N2R_minus_human": paired("N2R", "human", prim, bt)}
    cnt: dict = {}
    for t in prim:
        cnt[choice[t][0]] = cnt.get(choice[t][0], 0) + 1
    res["secondary"] = {"decides": "nothing",
                        "N2C_minus_N1C": paired("N2C", "N1C", prim, bt), "N2R_minus_N2C": paired("N2R", "N2C", prim, bt),
                        "CERT2": {"label": "RJ5 teacher + P5-R + map-certified selection over [N2R, N2C, N1C, T] (NEVER the teacher alone)",
                                  "choice_primary": cnt, "none_pass_primary": int(sum(choice[t][1] for t in prim)),
                                  "CERT2_minus_N2R": paired("CERT2", "N2R", prim, bt), "CERT2_minus_N1C": paired("CERT2", "N1C", prim, bt),
                                  "paper_read_descriptive": paper_read("CERT2")}}
    res["families"] = "run by the chain AFTER this read (stage families -> famread); the longitudinal guard is read there, never here"
    res["k_ledger"] = {"entry": "ONE K entry (TF-N2-A1, n = 44); T, N1C (200-token re-score), N2W, N2C, CERT2 and human are non-selection reads", "K": K}
    print("ZZN2_VERDICT PRIMARY", res["primary"]["outcome"], json.dumps(d), "dDAC", json.dumps(dd["mean"]), "N2R DAC", dacR, flush=True)
    print("ZZN2_VERDICT PAPER", res["paper_level"]["N2R"]["level"], json.dumps(res["paper_level"]["N2R"]["pdms"]), flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0


def cmd_famprep(a) -> int:
    known = A.load_known36(a.known36)
    seen, n = set(), 0
    with open(a.out, "w", encoding="utf-8") as fo:
        for fp in sorted(glob.glob(a.rows)):
            for ln in open(fp, encoding="utf-8"):
                if not ln.strip() or '"error"' in ln[:200]:
                    continue
                r = json.loads(ln)
                if "error" in r or r["token"] in known or r["token"] in seen:
                    continue
                seen.add(r["token"])
                fo.write(ln if ln.endswith("\n") else ln + "\n")
                n += 1
    print("ZZN2_FAMPREP", n, "primary rows ->", a.out, flush=True)
    return 0


def lon(fp):
    d = json.load(open(fp, encoding="utf-8"))
    blk = d["frames"]["executed_tracked"]["block"]["longitudinal"]
    return {"n": d.get("n"), "n_logs": d.get("n_logs"), **{k: v for k, v in blk.items() if isinstance(v, (int, float))}}


def guard_eval(n2r: dict, n1c: dict, g=GUARD) -> dict:
    """the PROPOSED longitudinal guard (card section 10). A flag goes to the PI; it never changes the PRIMARY verdict."""
    chk = {"speed_bias_vs_n1c_mps": (n2r["speed_bias_mps"] - n1c["speed_bias_mps"], g["speed_bias_vs_n1c_mps"]),
           "speed_mae_vs_n1c_mps": (n2r["speed_mae_mps"] - n1c["speed_mae_mps"], g["speed_mae_vs_n1c_mps"]),
           "accel_mae_vs_n1c_mps2": (n2r["accel_mae_mps2"] - n1c["accel_mae_mps2"], g["accel_mae_vs_n1c_mps2"]),
           "speed_bias_abs_mps": (n2r["speed_bias_mps"], g["speed_bias_abs_mps"])}
    out = {k: {"value": round(v, 4), "bound": b, "exceeded": bool(v > b)} for k, (v, b) in chk.items()}
    return {"checks": out, "flag": any(x["exceeded"] for x in out.values())}


def cmd_famread(a) -> int:
    F = {nm: lon(fp) for nm, fp in (s.split("=", 1) for s in a.fam)}
    need = ("N2R", "N1C")
    if any(k not in F for k in need):
        raise SystemExit(f"--fam needs {need}")
    res = {"basis": "executed_tracked, W-TRANSP scenery (NOT harness-identical; the pinned transp_score.py, as TF-N1-A1), primary 12,110; point values, no paired "
                    "family CI exists (work item)", "strategic": "PAUSED (PI 2026-09-19)", "longitudinal": F,
           "guard": guard_eval(F["N2R"], F["N1C"]), "navtrain_reference_RJ5R_minus_R0C": {"speed_bias_mps": 0.2837, "speed_mae_mps": 0.1771, "accel_mae_mps2": 0.0640}}
    gd = res["guard"]
    print("ZZN2_GUARD", "FLAG: label-quality regression, goes to the PI" if gd["flag"] else "no flag", json.dumps(gd["checks"]), flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("read", "famprep", "famread"))
    ap.add_argument("--harness-csv", nargs="*", default=[])
    ap.add_argument("--identity-csv")
    ap.add_argument("--human-csv")
    ap.add_argument("--known36")
    ap.add_argument("--build-report")
    ap.add_argument("--ctl-r0")
    ap.add_argument("--cidc")
    ap.add_argument("--pins")
    ap.add_argument("--pin-root", default=".")
    ap.add_argument("--proxy-eval")
    ap.add_argument("--prereg-sha256")
    ap.add_argument("--token-log", default="F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz")
    ap.add_argument("--expect-n", type=int, default=12146)
    ap.add_argument("--expect-primary", type=int, default=12110)
    ap.add_argument("--id-tol", type=float, default=BAR["id_tol"])
    ap.add_argument("--reg-margin", type=float, default=BAR["reg_margin"])
    ap.add_argument("--rows", default="")
    ap.add_argument("--fam", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return {"read": cmd_read, "famprep": cmd_famprep, "famread": cmd_famread}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
