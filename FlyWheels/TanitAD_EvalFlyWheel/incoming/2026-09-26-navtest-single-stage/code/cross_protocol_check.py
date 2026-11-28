#!/usr/bin/env python3
"""PREREG.md §3 — the cross-protocol consistency checks, v2 ONE-STAGE vs W3's independent v1.1 run.

    python code/cross_protocol_check.py --run <suite run dir> --e-t0 raw/e_t0_<name>.json \
        --out raw/cross_protocol_<name>.json [--full]

W8, 2026-09-26. Pre-registered (PREREG.md, sha256 in raw/PREREG_HASH.txt) BEFORE any v2 navtest score
existed. Any python (csv only). Every expectation is a LITERAL or a relation between two INDEPENDENT
artifacts (the suite's scores/<arm>.csv and W3's raw/<ARM>_navtest/<ARM>_navtest.csv); nothing is
recomputed from the code under test.

  C-DAC  v2_DAC(t) == v1_DAC(t)                                  (0 mismatches per arm)
  C-NC   v2_NC(t) == v1_NC(t) off E-T0; v2_NC(t) >= v1_NC(t) on E-T0
  C-TTC  v2_TTC(t) >= v1_TTC(t)
  C-DDC  v2_DDC(t) >= v1_DDC(t)
  C-HUMF v2 HUMAN: NC = DAC = TTC = TLC = LK = 1.0 on every token; DDC in {0.5, 1}
  C-FORMULA / C-AVG / C-COUNT   read from the suite's own controls (summary.json), PASS as recorded
  H-HIGH v2 HUMAN EPDMS >= 0.90                                   (FULL split only)
  K-NEG  C-DAC pairing v2 CV with v1 STOP: >= 1000 mismatches on the full split (>= 1 on a subset)
  K-MUT  one DAC cell of v2 CV flipped in memory: exactly 1 mismatch
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
W3 = REPO / "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw"
COLS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "TTC": "time_to_collision_within_bound",
        "DDC": "driving_direction_compliance", "EP": "ego_progress"}
V2_EXTRA = {"TLC": "traffic_light_compliance", "LK": "lane_keeping"}


def rows(path: Path, summary=("average", "average_all_frames")) -> dict:
    with open(path, encoding="utf-8", newline="") as fh:
        rd = csv.DictReader(fh)
        return {r["token"]: r for r in rd if r["token"] not in summary and not r["token"].startswith("extended_")}


def f(x) -> float:
    return float("nan") if x in (None, "", "nan") else float(x)


def compare(v2: dict, v1: dict, col: str, rel: str, exempt: set = frozenset()) -> dict:
    """rel: '==' exact, or '>=' one-sided (v2 >= v1). `exempt` tokens are held to '>=' only."""
    toks = sorted(set(v2) & set(v1))
    bad = []
    for t in toks:
        a, b = f(v2[t][col]), f(v1[t][col])
        ok = (a >= b) if (rel == ">=" or t in exempt) else (a == b)
        if not ok or math.isnan(a) or math.isnan(b):
            bad.append({"token": t, "v2": a, "v1": b})
    return {"relation": rel, "n_compared": len(toks), "n_violations": len(bad), "first_violations": bad[:10],
            "n_exempt_one_sided": len(set(toks) & set(exempt)), "pass": bool(toks) and not bad}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--e-t0", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--full", action="store_true", help="the FULL split (enables H-HIGH and the >=1000 K-NEG bar)")
    ap.add_argument("--e-obs-class", default=None, type=Path,
                    help="AMENDMENT A2: e_obs_classify.py output -> C-NC-CLS / C-TTC-CLS + the discrepancy set")
    a = ap.parse_args(argv)
    summ = json.loads((a.run / "summary.json").read_text(encoding="utf-8"))
    et0 = set(json.loads(a.e_t0.read_text(encoding="utf-8"))["E_T0"])
    v1 = {arm: rows(W3 / f"{arm}_navtest" / f"{arm}_navtest.csv") for arm in ("CV", "STOP", "HUMAN")}
    v2 = {arm: rows(a.run / "scores" / f"{arm}.csv") for arm in ("CV", "STOP", "HUMAN")
          if (a.run / "scores" / f"{arm}.csv").exists()}
    out = {"schema": "w8-cross-protocol/1", "run": str(a.run).replace("\\", "/"), "full_split": a.full,
           "prereg": "PREREG.md (sha256 in raw/PREREG_HASH.txt)", "n_E_T0": len(et0), "arms": {}, "checks": {}}
    verdict = {}
    cls = json.loads(a.e_obs_class.read_text(encoding="utf-8"))["class"] if a.e_obs_class else None
    discrepancy = set()
    for arm, r2 in v2.items():
        blk = {"n_tokens_v2": len(r2), "n_common_with_v1": len(set(r2) & set(v1[arm]))}
        blk["C-DAC"] = compare(r2, v1[arm], COLS["DAC"], "==")
        blk["C-NC"] = compare(r2, v1[arm], COLS["NC"], "==", exempt=et0)
        blk["C-TTC"] = compare(r2, v1[arm], COLS["TTC"], ">=")
        blk["C-DDC"] = compare(r2, v1[arm], COLS["DDC"], ">=")
        ep = [(f(r2[t][COLS["EP"]]), f(v1[arm][t][COLS["EP"]])) for t in sorted(set(r2) & set(v1[arm]))]
        blk["EP_descriptive"] = {"n": len(ep), "n_exactly_equal": sum(1 for x, y in ep if x == y),
                                 "n_within_1e-9": sum(1 for x, y in ep if abs(x - y) <= 1e-9),
                                 "mean_abs_diff": (sum(abs(x - y) for x, y in ep) / len(ep)) if ep else None,
                                 "mean_v2": (sum(x for x, _ in ep) / len(ep)) if ep else None,
                                 "mean_v1": (sum(y for _, y in ep) / len(ep)) if ep else None,
                                 "_note": "EP's normalisation CHANGED (PREREG §2): no bar, descriptive only"}
        for k in ("C-DAC", "C-NC", "C-TTC", "C-DDC"):
            verdict[f"{k}[{arm}]"] = blk[k]["pass"]
        if arm == "HUMAN":
            exc = []
            for t, r in r2.items():
                for k, c in (("NC", COLS["NC"]), ("DAC", COLS["DAC"]), ("TTC", COLS["TTC"]),
                             ("TLC", V2_EXTRA["TLC"]), ("LK", V2_EXTRA["LK"])):
                    if f(r[c]) != 1.0:
                        exc.append({"token": t, "metric": k, "value": f(r[c])})
                if f(r[COLS["DDC"]]) not in (0.5, 1.0):
                    exc.append({"token": t, "metric": "DDC", "value": f(r[COLS["DDC"]])})
            blk["C-HUMF"] = {"n_tokens": len(r2), "n_exceptions": len(exc), "first": exc[:10], "pass": bool(r2) and not exc}
            verdict["C-HUMF"] = blk["C-HUMF"]["pass"]
        s_arm = (summ.get("arms") or {}).get(arm) or {}
        ctl = s_arm.get("controls") or {}
        blk["C-FORMULA"] = {k: (ctl.get("C4_formula") or {}).get(k) for k in ("n_rows", "max_abs_diff", "tol", "pass")}
        blk["C-AVG"] = {k: (ctl.get("C_AVG") or {}).get(k) for k in ("abs_diff", "tol", "n", "pass")}
        cnt = ctl.get("C_COUNT") or {}
        blk["C-COUNT"] = {**cnt, "pass": bool(cnt.get("log_successful") == cnt.get("csv_valid_rows") == len(r2)
                                             and cnt.get("log_failed") == 0 and s_arm.get("status") == "OK")}
        verdict[f"C-FORMULA[{arm}]"] = bool(blk["C-FORMULA"].get("pass"))
        verdict[f"C-AVG[{arm}]"] = bool(blk["C-AVG"].get("pass"))
        verdict[f"C-COUNT[{arm}]"] = blk["C-COUNT"]["pass"]
        blk["headline_EPDMS"] = (s_arm.get("headline") or {}).get("value")
        if cls is not None:                                            # AMENDMENT A2 (PREREG.md)
            ident = {t for t, c in cls.items() if c == "IDENTICAL"}
            sup = {t for t, c in cls.items() if c == "V1_SUPERSET"}
            common = set(r2) & set(v1[arm])
            blk["C-NC-CLS"] = {"IDENTICAL_exact": compare({t: r2[t] for t in common & ident}, v1[arm], COLS["NC"], "=="),
                               "V1_SUPERSET_ge": compare({t: r2[t] for t in common & sup}, v1[arm], COLS["NC"], ">=")}
            blk["C-NC-CLS"]["pass"] = bool(blk["C-NC-CLS"]["IDENTICAL_exact"]["n_violations"] == 0
                                           and blk["C-NC-CLS"]["V1_SUPERSET_ge"]["n_violations"] == 0
                                           and (common & (ident | sup)))
            blk["C-TTC-CLS"] = compare({t: r2[t] for t in common & (ident | sup)}, v1[arm], COLS["TTC"], ">=")
            orig = [v["token"] for v in blk["C-NC"]["first_violations"]] if blk["C-NC"]["n_violations"] <= 10 else None
            nc_diff = sorted(t for t in common if f(r2[t][COLS["NC"]]) != f(v1[arm][t][COLS["NC"]]))
            ttc_lt = sorted(t for t in common if f(r2[t][COLS["TTC"]]) < f(v1[arm][t][COLS["TTC"]]))
            blk["A2_original_violations_by_class"] = {
                "C-NC": {c: sum(1 for t in nc_diff if cls.get(t) == c) for c in ("IDENTICAL", "V1_SUPERSET", "V2_EXTRA")},
                "C-TTC": {c: sum(1 for t in ttc_lt if cls.get(t) == c) for c in ("IDENTICAL", "V1_SUPERSET", "V2_EXTRA")}}
            on_identical = (blk["A2_original_violations_by_class"]["C-NC"]["IDENTICAL"]
                            + blk["A2_original_violations_by_class"]["C-TTC"]["IDENTICAL"])
            verdict[f"C-NC-CLS[{arm}]"] = blk["C-NC-CLS"]["pass"]
            verdict[f"C-TTC-CLS[{arm}]"] = bool(blk["C-TTC-CLS"]["pass"] or not (common & (ident | sup)))
            verdict[f"A2-no-original-violation-on-IDENTICAL[{arm}]"] = on_identical == 0
            dac_diff = [t for t in common if f(r2[t][COLS["DAC"]]) != f(v1[arm][t][COLS["DAC"]])]
            discrepancy.update(nc_diff); discrepancy.update(dac_diff)
        out["arms"][arm] = blk
    if "HUMAN" in v2 and a.full:
        h = out["arms"]["HUMAN"]["headline_EPDMS"]
        out["checks"]["H-HIGH"] = {"value": h, "bar": 0.90, "pass": h is not None and h >= 0.90,
                                   "published_pre_fix_INFERRED_NOT_COMPARABLE": 90.3}
        verdict["H-HIGH"] = out["checks"]["H-HIGH"]["pass"]
    if "CV" in v2:                                                     # the controls on the check itself
        kneg = compare(v2["CV"], v1["STOP"], COLS["DAC"], "==")
        bar = 1000 if a.full else 1
        out["checks"]["K-NEG"] = {"pairing": "v2 CV vs v1 STOP (DAC)", "n_mismatches": kneg["n_violations"],
                                  "bar": f">= {bar}", "pass": kneg["n_violations"] >= bar}
        mut = {t: dict(r) for t, r in v2["CV"].items()}
        t0 = sorted(set(mut) & set(v1["CV"]))[0]
        mut[t0][COLS["DAC"]] = "0.0" if f(mut[t0][COLS["DAC"]]) == 1.0 else "1.0"
        kmut = compare(mut, v1["CV"], COLS["DAC"], "==")
        out["checks"]["K-MUT"] = {"flipped_token": t0, "n_mismatches": kmut["n_violations"], "want": 1,
                                  "pass": kmut["n_violations"] == 1}
        verdict["K-NEG"] = out["checks"]["K-NEG"]["pass"]
        verdict["K-MUT"] = out["checks"]["K-MUT"]["pass"]
    stop, cv = (summ.get("arms") or {}).get("STOP", {}), (summ.get("arms") or {}).get("CV", {})
    hs, hc = (stop.get("headline") or {}).get("value"), (cv.get("headline") or {}).get("value")
    out["directional_prediction_STOP_gt_CV"] = {"STOP": hs, "CV": hc, "holds": (hs is not None and hc is not None and hs > hc),
                                                "_note": "a prediction, not a bar (PREREG §3)"}
    if cls is not None:
        # a2 (PREREG): the counterfactual's token set = the discrepancy set U every 25th token in sorted order
        allt = sorted(set().union(*[set(r) for r in v2.values()]))
        sample = allt[::25]
        cf = sorted(discrepancy | set(sample))
        out["A2_counterfactual_tokens"] = {"n_discrepancy": len(discrepancy), "n_sample_every_25th": len(sample),
                                           "n_total": len(cf), "tokens": cf}
        out["A2_class_counts"] = {c: sum(1 for v in cls.values() if v == c) for c in ("IDENTICAL", "V1_SUPERSET", "V2_EXTRA")}
    out["verdict"] = verdict
    out["all_pass"] = bool(verdict) and all(verdict.values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({"all_pass": out["all_pass"], "failed": [k for k, v in verdict.items() if not v],
                      "n_checks": len(verdict)}, indent=1))
    return 0 if out["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
