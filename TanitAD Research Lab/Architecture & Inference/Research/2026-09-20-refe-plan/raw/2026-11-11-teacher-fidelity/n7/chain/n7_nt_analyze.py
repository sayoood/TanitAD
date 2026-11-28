#!/usr/bin/env python3
"""TF-N7 navtest readout (PREREG_TF_N7.md sections 4 to 6). Reads ONLY harness CSVs (the pinned W3 harness), the build report and the control files.

  read     python n7_nt_analyze.py read --harness-csv N2R=<csv> N4D=<csv> N5A=<csv> N6T=<csv> N7T3=<csv> N7T=<csv> --build-report <n7_build_report.json> --unch <unch.json>
               --pins <N7_PINS.json> --pin-root <run> --proxy-eval <eval dir> --prereg-sha256 <sha> --token-log <navtest_inputs.json.gz> --known36 <json> --k 50
               --a1-analyze <tf_n1_analyze_a1.py> --out <json>
           CONTROLS FIRST (C-COV, C-BUILD, C-UNCH, C-PIN, C-CERT, C-ID-C, C-EST, C-REG, C-N6SAME). If any fails it prints ZZN7_CONTROLS FAIL, writes a file with NO aggregate and
           exits 1 BEFORE any N7-T3 / N7-T number is computed. Then the section 4 bars for the PRIMARY (N7-T3, tiers T0-T3) and, beside it, for the SECONDARY (N7-T, tiers T0-T4),
           both against N5-A, and the attribution against N6-T.
  famread  python n7_nt_analyze.py famread --fam-ci-t3 <fam_ci N7T3 vs N5A> --fam-ci-t <fam_ci N7T vs N5A> --fam-a <fam_N5A.json> --fam-t3 <fam_N7T3.json> --fam-t <fam_N7T.json>
               --build-report <json> --k 50 --out <json>      (the four-family guard on the primary; the secondary beside it; the changed tokens' own speed MAE per token)

N7-X (certificate inverted) is not scored: the build proves it changed 0 tokens and its seam is bit-identical to N5-A (C-UNCH, C-X0), so its harness rows ARE the banked N5-A rows.

BARS (section 4), on the PRIMARY N7-T3: (1) paper level, PDMS >= 95.8 AND DAC >= 99.9 on the primary 12,110 (point estimates; paired log-cluster bootstrap CIs vs N5-A reported);
(2) no harm: paired NC and DDC delta (N7-T3 - N5-A) point >= 0 AND no changed token's harness NC or DDC below N5-A's AND no regression against N6-T (every token N6-T certified has the
same harness row in N7-T3, and the build's C-N6SAME holds); (3) certificate integrity: every changed token reads harness DAC = 1 and the build's C-CERT-B7 holds; (4) four-family
guard (famread; STRATEGIC = PAUSED); (5) K = 50, Bonferroni at K. CONFIRMED iff (1), (2) and (3) on N7-T3. N7-T gets the same bars as the SECONDARY: they decide nothing.
Estimator: `Boot` of tf_n1_analyze_a1.py: paired log-cluster bootstrap over the logs, 10,000 draws, seed 20260927, 95 % percentile, plus the Bonferroni interval at K. It answers
"would another draw of LOGS say this?" only; the arms are deterministic (C-DET), so it says nothing about training or inference variance."""
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
NEW_ARMS = ("N7T3", "N7T")
NEED = ("N2R", "N4D", "N5A", "N6T", "N7T3", "N7T")
TIERS_OF = {"N7T3": "T0 to T3", "N7T": "T0 to T4"}
load_a1, sha256, load_csv, load_tlog, pins_check = A5.load_a1, A5.sha256, A5.load_csv, A5.load_tlog, A5.pins_check
NAMING = "RJ5 teacher + P5-R + REPAIR-D + L1 + wider generator (tiers {tiers}) + tolerance (NC/DDC kept, TTC/C free), teacher only, N_teacher_tol = {nt} of 20 residual tokens"


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
                   "what": "plan20 and seam files on disk bit-identical to banked N5-A off the applied set; residual count == the card's; N7-X bit-identical to N5-A"}
    C["C-PIN"] = pins_check(a.pins, a.pin_root) if a.pins else {"pass": False, "why": "no --pins"}
    cert_bad = {arm: [t for t in rep.get(arm, []) if t not in arms[arm] or arms[arm][t]["dac"] != 1.0] for arm in NEW_ARMS}
    C["C-CERT"] = {"pass": not any(cert_bad.values()), "exceptions": {k: len(v) for k, v in cert_bad.items()}, "examples": {k: v[:3] for k, v in cert_bad.items()},
                   "certified": {k: len(rep.get(k, [])) for k in NEW_ARMS}, "what": "every token the builder certified (DAC = 1) has harness DAC = 1, per arm"}
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
    # C-N6SAME (read side): the tokens N6-T changed read the SAME harness row in N7-T3 and N7-T (identical plans, deterministic harness)
    n6_changed = sorted(t for t in allt if max(abs(arms["N6T"][t][m] - arms["N5A"][t][m]) for m in MET) > a.id_tol)
    bsame = br.get("controls", {}).get("C-N6SAME", {})
    n6_named = bsame.get("n6t_tokens")
    same_bad = {arm: [t for t in n6_changed if max(abs(arms[arm][t][m] - arms["N6T"][t][m]) for m in MET) > a.id_tol] for arm in NEW_ARMS}
    not_applied = {arm: [t for t in n6_changed if t not in set(rep.get(arm, []))] for arm in NEW_ARMS}
    C["C-N6SAME"] = {"pass": not any(same_bad.values()) and not any(not_applied.values()) and bool(bsame.get("pass")) and (n6_named is None or sorted(n6_named) == n6_changed),
                     "n6t_changed_tokens": len(n6_changed), "harness_rows_differ": {k: v[:3] for k, v in same_bad.items()}, "not_in_applied_set": {k: v[:3] for k, v in not_applied.items()},
                     "build_c_n6same_pass": bool(bsame.get("pass")), "n6t_tokens_agree_with_build_report": n6_named is None or sorted(n6_named) == n6_changed,
                     "what": "the tokens N6-T changed (harness N6-T row != N5-A row) have the identical harness row in N7-T3 and N7-T, and the build's C-N6SAME passed"}
    bt = A.Boot(prim, tl)
    if a.proxy_eval:
        spec = importlib.util.spec_from_file_location("proxy_eval_n7", os.path.join(a.proxy_eval, "proxy_eval.py"))
        PE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(PE)
        pt = {t: arms["N2R"][t]["pdms"] for t in prim}
        ref, ours = PE.boot(pt, tl), bt.stat(pt)["_raw"]
        dev = max(abs(x - y) for x, y in zip(ref, ours))
        C["C-EST"] = {"pass": dev <= 1e-9, "max_abs_dev": dev}
    else:
        C["C-EST"] = {"pass": False, "why": "no --proxy-eval"}
    xu = un.get("arms", {}).get("N7X", {})
    C["C-REG"] = {"pass": bool(not rep.get("N7X") and xu.get("pass") and xu.get("bit_identical_to_n5a") and br["controls"].get("C-X0", {}).get("pass")),
                  "n7x_applied": len(rep.get("N7X", [])), "n7x_bit_identical_to_n5a": xu.get("bit_identical_to_n5a"),
                  "what": "the deliberate-regression arm (certificate inverted) changed 0 tokens and its seam and plans are bit-identical to N5-A; its harness rows are the N5-A rows (the harness is deterministic, C-DET)"}
    allpass = all(c["pass"] for c in C.values())
    print("ZZN7_CONTROLS", "PASS" if allpass else "FAIL", json.dumps({k: v["pass"] for k, v in C.items()}), flush=True)
    if not allpass:
        res["verdict"] = "NO VERDICT: a control failed; no N7-T3 / N7-T aggregate was computed"
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

    prim_set = set(prim)
    rt = br.get("residual_table", {})

    def arm_block(arm):
        pr = paper_read(arm)
        b1 = pr["level"] == "PAPER LEVEL"
        changed = sorted(rep.get(arm, []))
        harm = {m: paired(arm, "N5A", m) for m in ("nc", "ddc", "ttc", "c", "ep", "dac")}
        b2_means = harm["nc"]["mean"] >= BAR["nc_ddc_floor_x100"] and harm["ddc"]["mean"] >= BAR["nc_ddc_floor_x100"]
        tok_viol = sorted(t for t in changed if arms[arm][t]["nc"] < arms["N5A"][t]["nc"] - BAR["tok_tol"] or arms[arm][t]["ddc"] < arms["N5A"][t]["ddc"] - BAR["tok_tol"])
        n6_regress = bool(same_bad[arm] or not_applied[arm] or not bsame.get("pass"))
        b2 = bool(b2_means and not tok_viol and not n6_regress)
        agree = [t for t in changed if arms[arm][t]["dac"] == 1.0]
        b3 = bool(len(agree) == len(changed) and br["controls"].get("C-CERT-B7", {}).get("pass"))
        branches = [n for n, ok in (("paper_level_not_reached", b1), ("harm", b2), ("cert_integrity", b3)) if not ok]
        tiers = br.get("tier_of", {}).get(arm, {})
        n_t = sum(1 for t in tiers if t in prim_set)
        still = sorted(t for t in prim if arms[arm][t]["dac"] == 0.0)
        bkey = "blocking_class_n7t3" if arm == "N7T3" else "blocking_class"
        return {"contrast": f"{arm} vs N5A, harness, 136 logs", "naming": NAMING.format(tiers=TIERS_OF[arm], nt=n_t), "n_teacher_tol_replacements": n_t, "n_changed_total": len(changed),
                "tier_counts": br.get("counts", {}).get(arm, {}).get("by_tier"), "paper_level": pr,
                "delta_vs_N5A_paired": {"pdms": paired(arm, "N5A", "pdms"), "dac": harm["dac"]},
                "no_harm_paired_x100_minus_N5A": {k: v for k, v in harm.items() if k != "dac"},
                "no_harm_changed_token_violations": {"count": len(tok_viol), "tokens": tok_viol[:10]},
                "no_regression_vs_N6T": {"n6t_changed_tokens": len(n6_changed), "harness_rows_differ": same_bad[arm][:5], "not_in_applied_set": not_applied[arm][:5],
                                         "build_c_n6same": bool(bsame.get("pass"))},
                "certificate_integrity": {"changed": len(changed), "harness_dac1": len(agree), "mismatches": sorted(set(changed) - set(agree))[:10]},
                "bars": {"1_paper_level_pdms_ge_95.8_and_dac_ge_99.9": bool(b1), "2_no_harm_nc_ddc_and_no_regression_vs_N6T": bool(b2), "3_certificate_integrity": bool(b3)},
                "outcome": "CONFIRMED" if (b1 and b2 and b3) else "NOT CONFIRMED", "not_confirmed_branches_in_card_order": branches,
                "tolerance_declared_ttc_c_reported": {"ttc": harm["ttc"], "c": harm["c"]},
                "remaining_zero_dac": {"count": len(still), "tokens": still, "blocking": {t: rt.get(t, {}).get(bkey) for t in still}}}

    P, Sx = arm_block("N7T3"), arm_block("N7T")
    res["naming"] = P["naming"]
    res["primary"] = dict(P, arm="N7T3 (PRIMARY, tiers T0 to T3)")
    res["secondary_N7T"] = dict(Sx, arm="N7T (SECONDARY, tiers T0 to T4, decides nothing)")
    pl3, plT = P["paper_level"]["level"], Sx["paper_level"]["level"]
    res["rests_on_T4"] = ("N7-T reaches paper level and N7-T3 does not: the result rests on the T4 slow-down tier (s = 0.5 to 0.7, an edit of the label's speed by 30 to 50 percent) "
                          "and is quoted so; whether a speed-edited label is teacher output is a PI decision" if (plT == "PAPER LEVEL" and pl3 != "PAPER LEVEL") else
                          "does not apply (N7-T3 reaches paper level, or N7-T does not)")
    res["t4_notice"] = "every number quoted for N7-T includes the T4 deep slow-down tier; N7-T3 (T0 to T3, no speed edit beyond s = 0.8) is the primary"
    res["paper_level"] = {"N7T3": P["paper_level"], "N7T": Sx["paper_level"], "N6T_teacher_only": paper_read("N6T"), "N5A_comparator": paper_read("N5A"),
                          "rule": "PAPER LEVEL iff PDMS >= 95.8 AND DAC >= 99.9 (point estimates; <= 12 zero-DAC tokens); NEAR iff >= 95.3 AND >= 99.5; else BELOW"}
    res["attribution"] = {"decides": "nothing", "N7T3_minus_N6T": paired("N7T3", "N6T"), "N7T_minus_N6T": paired("N7T", "N6T"), "N7T_minus_N7T3": paired("N7T", "N7T3"),
                          "N7T3_minus_N2R": paired("N7T3", "N2R"), "N6T_minus_N5A": paired("N6T", "N5A")}
    res["families"] = "read in stage families: n7_nt_analyze.py famread (guard flags the student label only; STRATEGIC = PAUSED)"
    print("ZZN7_VERDICT PRIMARY", P["outcome"], "bars", json.dumps(P["bars"]), "branches", P["not_confirmed_branches_in_card_order"], flush=True)
    print("ZZN7_VERDICT SECONDARY", Sx["outcome"], "bars", json.dumps(Sx["bars"]), res["rests_on_T4"], flush=True)
    print("ZZN7_VERDICT PAPER N7T3", pl3, json.dumps(P["paper_level"]["pdms"]), json.dumps(P["paper_level"]["dac"]), "zero_dac", P["paper_level"]["zero_dac_tokens"], P["naming"], flush=True)
    print("ZZN7_VERDICT PAPER N7T", plT, json.dumps(Sx["paper_level"]["pdms"]), json.dumps(Sx["paper_level"]["dac"]), "zero_dac", Sx["paper_level"]["zero_dac_tokens"], Sx["naming"], flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0


def _speed_mae(j):
    """-> (tokens, per-window speed MAE array) from a families JSON written with --per-window, or (None, reason). Path: frames.executed_tracked.block._per_window.longitudinal.speed_mae_mps.value"""
    try:
        v = j["frames"]["executed_tracked"]["block"]["_per_window"]["longitudinal"]["speed_mae_mps"]["value"]
        toks = j["tokens"]
    except (KeyError, TypeError):
        return None, "no frames.executed_tracked.block._per_window.longitudinal.speed_mae_mps.value / tokens in the families JSON"
    arr = np.array([np.nan if x is None else x for x in v], float)
    if len(arr) != len(toks):
        return None, f"speed_mae_mps array has {len(arr)} entries for {len(toks)} tokens"
    return list(map(str, toks)), arr


def changed_speed_table(fam_a, fam_x, changed: list, tier_of: dict) -> dict:
    ta, va = _speed_mae(fam_a)
    tx, vx = _speed_mae(fam_x)
    if ta is None or tx is None:
        return {"status": "UNAVAILABLE", "reason": va if ta is None else vx, "note": "UNVERIFIED structure: reported as unavailable, never silently dropped"}
    ia, ix = {t: i for i, t in enumerate(ta)}, {t: i for i, t in enumerate(tx)}
    rows = {}
    for t in changed:
        if t in ia and t in ix:
            rows[t] = {"tier": tier_of.get(t), "speed_mae_N5A_mps": float(va[ia[t]]), "speed_mae_arm_mps": float(vx[ix[t]]), "diff_mps": float(vx[ix[t]] - va[ia[t]])}
        else:
            rows[t] = {"tier": tier_of.get(t), "status": "token not in the primary families set"}
    d = [r["diff_mps"] for r in rows.values() if "diff_mps" in r]
    by_tier = {}
    for r in rows.values():
        if "diff_mps" in r:
            by_tier.setdefault(r["tier"], []).append(r["diff_mps"])
    return {"status": "OK", "n_changed": len(changed), "n_read": len(d), "mean_diff_mps": float(np.mean(d)) if d else None,
            "max_abs_diff_mps": float(np.max(np.abs(d))) if d else None, "by_tier_mean_diff_mps": {k: float(np.mean(v)) for k, v in sorted(by_tier.items(), key=lambda x: str(x[0]))},
            "per_token": rows, "what": "the changed tokens' own longitudinal change (executed_tracked speed MAE per window, arm minus N5-A): a T4 slow-down edits speed on purpose"}


def _guard(fc, k, label):
    fam = fc["frames"]["executed_tracked"]["families"]

    def pick(f, key):
        v = fam.get(f, {}).get(key, {})
        d = v.get("diff_B_minus_A") if isinstance(v, dict) else None
        return d if isinstance(d, dict) else {"point": d, "lo": None, "hi": None}
    chk = {"speed_bias_diff_mps": (pick("longitudinal", "speed_bias_mps"), GUARD["speed_bias_diff_mps"]),
           "speed_mae_diff_mps": (pick("longitudinal", "speed_mae_mps"), GUARD["speed_mae_diff_mps"]),
           "cross_mae_diff_m": (pick("lateral", "cross_mae_m"), GUARD["cross_mae_diff_m"])}
    if all(v["point"] is None for v, _b in chk.values()):
        raise SystemExit(f"REFUSED: fam_ci output ({label}) carries no paired diff for any guard metric (an old families_run.py, or a changed fam_ci layout): no guard read")
    out = {k_: {"diff": v["point"], "ci": [v["lo"], v["hi"]], "bound": b, "exceeded": bool(v["point"] is None or v["point"] > b)} for k_, (v, b) in chk.items()}
    return {"guard": {"checks": out, "flag": any(x["exceeded"] for x in out.values())}, "reported_not_bars": {f"{fam_}.{kk}": pick(fam_, kk) for fam_, kk in REPORTED}}


def cmd_famread(a) -> int:
    fc3 = json.load(open(a.fam_ci_t3, encoding="utf-8"))
    prim = _guard(fc3, a.k, "N7T3 vs N5A")
    res = {"basis": "executed_tracked, W-TRANSP on the harness metric cache, N7-T3 minus N5-A, paired log-cluster bootstrap (fam_ci.py, B = N7T3); the primary is teacher only, no human-log token",
           "strategic": "PAUSED (PI 2026-09-19)", "K": a.k, "primary_N7T3": prim,
           "guard": prim["guard"], "reported_not_bars": prim["reported_not_bars"]}
    if a.fam_ci_t:
        res["secondary_N7T"] = _guard(json.load(open(a.fam_ci_t, encoding="utf-8")), a.k, "N7T vs N5A")
    br = json.load(open(a.build_report, encoding="utf-8")) if a.build_report else {}
    if a.fam_a and a.fam_t3:
        fa = json.load(open(a.fam_a, encoding="utf-8"))
        res["changed_tokens_speed_mae_N7T3"] = changed_speed_table(fa, json.load(open(a.fam_t3, encoding="utf-8")), sorted(br.get("repaired", {}).get("N7T3", [])),
                                                                   br.get("tier_of", {}).get("N7T3", {}))
        if a.fam_t:
            res["changed_tokens_speed_mae_N7T"] = changed_speed_table(fa, json.load(open(a.fam_t, encoding="utf-8")), sorted(br.get("repaired", {}).get("N7T", [])),
                                                                      br.get("tier_of", {}).get("N7T", {}))
    for k in ("changed_tokens_speed_mae_N7T3", "changed_tokens_speed_mae_N7T"):
        if k in res:
            print("ZZN7_SPEEDTOK", k, res[k]["status"], json.dumps({x: res[k].get(x) for x in ("n_read", "mean_diff_mps", "max_abs_diff_mps", "by_tier_mean_diff_mps", "reason")}), flush=True)
    print("ZZN7_GUARD", "FLAG: label-quality regression, goes to the PI" if prim["guard"]["flag"] else "no flag", json.dumps(prim["guard"]["checks"]), flush=True)
    if "secondary_N7T" in res:
        print("ZZN7_GUARD N7T", "FLAG: goes to the PI" if res["secondary_N7T"]["guard"]["flag"] else "no flag", json.dumps(res["secondary_N7T"]["guard"]["checks"]), flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("read", "famread"))
    ap.add_argument("--harness-csv", nargs="*", default=[])
    ap.add_argument("--build-report", default="")
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
    ap.add_argument("--fam-ci-t3", default="")
    ap.add_argument("--fam-ci-t", default="")
    ap.add_argument("--fam-a", default="")
    ap.add_argument("--fam-t3", default="")
    ap.add_argument("--fam-t", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    return evaluate(a) if a.mode == "read" else cmd_famread(a)


if __name__ == "__main__":
    sys.exit(main())
