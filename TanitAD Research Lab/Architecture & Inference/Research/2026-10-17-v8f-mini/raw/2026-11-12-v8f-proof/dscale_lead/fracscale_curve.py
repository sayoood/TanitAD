#!/usr/bin/env python
"""Curve read of card PREREG_V8F_PROOF_ADD64 (FRACSCALE-LEAD).

Reads the three reader results (EIGHTH and QUARTER of card ADD64, HALF of card
ADD60), each as its reader wrote it under note N4 of card ADD45, and writes

* the name of each fraction (the reader's LEAD-READ-... with the prefix of its run),
* the curve name by the rule of card ADD64 section 3,
* the table (M2, gap error, M4 per arm at 571 / 1,143 / 2,286 / 4,572 clips),
* the same-seed replicate of the WHOLE arm over the three runs (run-to-run floor).

It computes no interval and tests nothing itself: every interval and every
per-fraction name is the reader's. Standard library only. T0 (a probe on cached
tokens of logged frames); never driving performance.

usage: fracscale_curve.py --eighth R.json --quarter R.json --half R.json --out OUT.json
       fracscale_curve.py --selftest
"""
import argparse
import json
import sys

FRACS = (("EIGHTH", 571), ("QUARTER", 1143), ("HALF", 2286))
WHOLE_N = 4572
SEEDS = (0, 1, 2, 3)
KIT_NAMES = ("LEAD-READ-SCALES-GAP", "LEAD-READ-WORSE-AT-SCALE", "LEAD-READ-FLAT-AT-SCALE")
SCALES, WORSE, FLAT = KIT_NAMES


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def arm_rows(res, prefix):
    """Per-seed M2 r2, gap error and M4 AUROC of the arm `prefix` (P16C or P16U)."""
    out = []
    for s in SEEDS:
        a = res["arms"]["%s_s%d" % (prefix, s)]
        out.append({"seed": s, "M2_r2": float(a["M2"]["r2"]),
                    "gap_mae_m": float(a["gap_mae"]["mae_m"]),
                    "M4_auroc": float(a["M4"]["auroc"])})
    return out


def summary(rows):
    return {k: mean(r[k] for r in rows) for k in ("M2_r2", "gap_mae_m", "M4_auroc")}


def curve_name(kit, mean_m2):
    """kit: {EIGHTH|QUARTER|HALF: reader name or None}; mean_m2: the same keys plus WHOLE."""
    order = [f for f, _ in FRACS]
    if any(kit[f] is None for f in order):
        return None, "a run has a failed control: no curve name"
    if all(kit[f] == SCALES for f in order):
        m = [mean_m2[f] for f in order] + [mean_m2["WHOLE"]]
        if all(m[i] < m[i + 1] for i in range(len(m) - 1)):
            return "FRACSCALE-LEAD-CURVE-RISES-THROUGHOUT", "all three SCALES-GAP and the means are ordered"
        return "FRACSCALE-LEAD-CURVE-MIXED", "all three SCALES-GAP but the means are not ordered"
    for i, f in enumerate(order):
        larger = order[i:]
        smaller = order[:i]
        if all(kit[g] == FLAT for g in larger) and all(kit[g] == SCALES for g in smaller):
            return "FRACSCALE-LEAD-CURVE-FLAT-FROM-%s" % f, "flat from %s upward, SCALES-GAP below" % f
    return "FRACSCALE-LEAD-CURVE-MIXED", "no rule of the card applies"


def build(results):
    """results: {EIGHTH|QUARTER|HALF: reader result dict}."""
    out = {"card": "PREREG_V8F_PROOF_ADD64 (curve read; per-fraction reads under note N4 of card ADD45)",
           "tier": "T0 (a probe on cached tokens of logged frames); never driving performance",
           "fractions": {}, "table": [], "replicate_whole": {}}
    kit, mean_m2, whole_runs = {}, {}, {}
    for f, n in FRACS:
        r = results[f]
        ok = bool(r.get("controls_pass"))
        name = r["names"]["name"] if ok else None
        if ok and name not in KIT_NAMES:
            raise SystemExit("unknown reader name in %s: %r" % (f, name))
        kit[f] = name
        c_rows, u_rows = arm_rows(r, "P16C"), arm_rows(r, "P16U")
        cs, us = summary(c_rows), summary(u_rows)
        mean_m2[f] = cs["M2_r2"]
        whole_runs[f] = u_rows
        tl = r["two_level"]["M2"]
        out["fractions"][f] = {
            "clips": n, "controls_pass": ok,
            "controls_failed": list(r.get("controls_failed", [])),
            "name": ("%sSCALE-%s" % (f, name)) if ok else None,
            "S_M2_whole_minus_fraction": {"mean": tl["mean"], "ci95": tl["ci95"],
                                          "per_seed": tl["per_seed"], "n_pos": tl["n_pos"],
                                          "n_neg": tl["n_neg"]} if ok else "withheld (a control failed)",
            "k2_tol_note_n4": r.get("k2_tol_note_n4"),
        }
        out["table"].append({"arm": f, "clips": n, "mean": cs, "per_seed": c_rows})
    # WHOLE: the mean over the three runs of it, per seed and over the seeds
    per_seed_mean = []
    for i, s in enumerate(SEEDS):
        row = {"seed": s}
        for k in ("M2_r2", "gap_mae_m", "M4_auroc"):
            row[k] = mean(whole_runs[f][i][k] for f, _ in FRACS)
        per_seed_mean.append(row)
    ws = summary(per_seed_mean)
    mean_m2["WHOLE"] = ws["M2_r2"]
    out["table"].append({"arm": "WHOLE (mean over its three runs)", "clips": WHOLE_N,
                         "mean": ws, "per_seed": per_seed_mean})
    # replicate: the WHOLE arm in the three runs, same seeds
    rep, spread_m2, spread_mae = [], [], []
    for i, s in enumerate(SEEDS):
        m2 = {f: whole_runs[f][i]["M2_r2"] for f, _ in FRACS}
        mae = {f: whole_runs[f][i]["gap_mae_m"] for f, _ in FRACS}
        sm2 = max(m2.values()) - min(m2.values())
        smae = max(mae.values()) - min(mae.values())
        spread_m2.append(sm2)
        spread_mae.append(smae)
        rep.append({"seed": s, "M2_r2_by_run": m2, "gap_mae_m_by_run": mae,
                    "spread_M2_r2": sm2, "spread_gap_mae_m": smae})
    floor_m2, floor_mae = max(spread_m2), max(spread_mae)
    out["replicate_whole"] = {
        "per_seed": rep,
        "largest_spread_M2_r2": floor_m2,
        "largest_spread_gap_mae_m": floor_mae,
        "equal_to_the_last_digit": bool(floor_m2 == 0.0 and floor_mae == 0.0),
        "statement": ("the three runs of the WHOLE arm are equal to the last digit: the floor is 0 "
                      "for the same seed and says nothing about another seed (the four seeds do)")
        if (floor_m2 == 0.0 and floor_mae == 0.0) else
        ("largest spread over the three runs of the WHOLE arm at one seed: M2 %.4f, gap error %.4f m"
         % (floor_m2, floor_mae)),
    }
    for f, _ in FRACS:
        fr = out["fractions"][f]
        if fr["controls_pass"]:
            fr["S_inside_run_to_run_floor"] = bool(abs(fr["S_M2_whole_minus_fraction"]["mean"]) < floor_m2)
    name, why = curve_name(kit, mean_m2)
    out["curve"] = {"name": name, "why": why, "mean_M2_r2": mean_m2,
                    "reader_names": kit}
    return out


def _fake(name, c_m2, u_m2, ok=True):
    arms = {}
    for s in SEEDS:
        arms["P16C_s%d" % s] = {"M2": {"r2": c_m2 + 0.01 * s}, "gap_mae": {"mae_m": 4.0}, "M4": {"auroc": 0.9}}
        arms["P16U_s%d" % s] = {"M2": {"r2": u_m2 + 0.01 * s}, "gap_mae": {"mae_m": 3.5}, "M4": {"auroc": 0.9}}
    d = u_m2 - c_m2
    return {"controls_pass": ok, "controls_failed": [] if ok else ["planted"],
            "names": {"name": name}, "arms": arms,
            "two_level": {"M2": {"mean": d, "ci95": [d - 0.01, d + 0.01], "per_seed": [d] * 4,
                                 "n_pos": 4 if d > 0 else 0, "n_neg": 0 if d > 0 else 4}}}


def selftest():
    cases = [
        ("rises", {"EIGHTH": _fake(SCALES, 0.3, 0.7), "QUARTER": _fake(SCALES, 0.4, 0.7),
                   "HALF": _fake(SCALES, 0.5, 0.7)}, "FRACSCALE-LEAD-CURVE-RISES-THROUGHOUT"),
        ("rises but not ordered", {"EIGHTH": _fake(SCALES, 0.45, 0.7), "QUARTER": _fake(SCALES, 0.4, 0.7),
                                   "HALF": _fake(SCALES, 0.5, 0.7)}, "FRACSCALE-LEAD-CURVE-MIXED"),
        ("flat from half", {"EIGHTH": _fake(SCALES, 0.3, 0.7), "QUARTER": _fake(SCALES, 0.4, 0.7),
                            "HALF": _fake(FLAT, 0.7, 0.7)}, "FRACSCALE-LEAD-CURVE-FLAT-FROM-HALF"),
        ("flat from quarter", {"EIGHTH": _fake(SCALES, 0.3, 0.7), "QUARTER": _fake(FLAT, 0.7, 0.7),
                               "HALF": _fake(FLAT, 0.7, 0.7)}, "FRACSCALE-LEAD-CURVE-FLAT-FROM-QUARTER"),
        ("flat from eighth", {"EIGHTH": _fake(FLAT, 0.7, 0.7), "QUARTER": _fake(FLAT, 0.7, 0.7),
                              "HALF": _fake(FLAT, 0.7, 0.7)}, "FRACSCALE-LEAD-CURVE-FLAT-FROM-EIGHTH"),
        ("flat below a scaling fraction", {"EIGHTH": _fake(FLAT, 0.7, 0.7), "QUARTER": _fake(SCALES, 0.4, 0.7),
                                           "HALF": _fake(FLAT, 0.7, 0.7)}, "FRACSCALE-LEAD-CURVE-MIXED"),
        ("a worse fraction", {"EIGHTH": _fake(SCALES, 0.3, 0.7), "QUARTER": _fake(WORSE, 0.8, 0.7),
                              "HALF": _fake(FLAT, 0.7, 0.7)}, "FRACSCALE-LEAD-CURVE-MIXED"),
        ("a failed control", {"EIGHTH": _fake(SCALES, 0.3, 0.7, ok=False), "QUARTER": _fake(SCALES, 0.4, 0.7),
                              "HALF": _fake(FLAT, 0.7, 0.7)}, None),
    ]
    ok = True
    for label, res, want in cases:
        got = build(res)["curve"]["name"]
        good = got == want
        ok = ok and good
        print("selftest %-32s want %-40s got %-40s %s" % (label, want, got, "ok" if good else "FAIL"))
    # known values of the replicate: equal runs read a floor of exactly 0; a planted 0.02 reads 0.02
    r = build(cases[2][1])["replicate_whole"]
    good = r["largest_spread_M2_r2"] == 0.0 and r["equal_to_the_last_digit"]
    ok = ok and good
    print("selftest replicate equal runs read 0: %s" % ("ok" if good else "FAIL"))
    res = {"EIGHTH": _fake(SCALES, 0.3, 0.72), "QUARTER": _fake(SCALES, 0.4, 0.70), "HALF": _fake(FLAT, 0.7, 0.70)}
    r = build(res)["replicate_whole"]
    good = abs(r["largest_spread_M2_r2"] - 0.02) < 1e-12 and not r["equal_to_the_last_digit"]
    ok = ok and good
    print("selftest replicate planted 0.02 reads 0.02: %s" % ("ok" if good else "FAIL"))
    # the withheld run carries no difference
    b = build(cases[7][1])
    good = b["fractions"]["EIGHTH"]["name"] is None and isinstance(b["fractions"]["EIGHTH"]["S_M2_whole_minus_fraction"], str)
    ok = ok and good
    print("selftest failed control withholds the difference: %s" % ("ok" if good else "FAIL"))
    print("FRACSCALE_CURVE_SELFTEST %s" % ok)
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eighth")
    ap.add_argument("--quarter")
    ap.add_argument("--half")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.eighth and a.quarter and a.half and a.out):
        ap.error("--eighth, --quarter, --half and --out are required")
    results = {}
    for f, p in (("EIGHTH", a.eighth), ("QUARTER", a.quarter), ("HALF", a.half)):
        with open(p, encoding="utf-8") as fh:
            results[f] = json.load(fh)
    out = build(results)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    for f, _ in FRACS:
        fr = out["fractions"][f]
        print("%-8s clips %5d name %s" % (f, fr["clips"], fr["name"]))
    for row in out["table"]:
        m = row["mean"]
        print("table %-34s clips %5d M2 %.4f gap error %.3f m M4 %.4f | per seed M2 %s"
              % (row["arm"], row["clips"], m["M2_r2"], m["gap_mae_m"], m["M4_auroc"],
                 ", ".join("%.4f" % r["M2_r2"] for r in row["per_seed"])))
    print("replicate: %s" % out["replicate_whole"]["statement"])
    print("CURVE %s (%s)" % (out["curve"]["name"], out["curve"]["why"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
