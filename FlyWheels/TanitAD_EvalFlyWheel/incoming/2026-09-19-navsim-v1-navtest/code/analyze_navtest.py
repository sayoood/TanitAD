#!/usr/bin/env python3
"""W3 analysis of the v1.1 navtest reference runs — verdicts against the PRE-REGISTERED SPEC.

    python analyze_navtest.py --split-run navtest      # reads raw/{CV,HUMAN,STOP}_navtest/
    python analyze_navtest.py --split-run smoke20      # the 20-token smoke (never a paper verdict)

Reads ONLY the devkit's own per-token CSVs (column ``score`` — v1.1 has no ``pdm_score``) and the
wrapper's observation-only hooks. Writes ``raw/analysis_<split-run>.json``:
  * per arm: n, every term ×100 (4 dp), the SPEC §4 verdict per term (REPRODUCED at the table's
    printed precision / CLOSE |Δ| ≤ 0.5 / NOT REPRODUCED) — navtest only;
  * C5 the navtest construction rule (CV > 0.8 / HUMAN < 0.8 violations, with tokens);
  * C6 STOP's NC / DAC / TTC against the SPEC §5 prediction;
  * the STOP decomposition (EP ≡ 1 regime, per-term means, paired STOP − CV W/T/L, per log,
    per t0-speed band) — SPEC §6.
Interval: none (``UNAVAILABLE`` until W2 registers the log-cluster bootstrap).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
RAW = os.path.join(PKG, "raw")
import pathlib                                                          # noqa: E402
REPO = pathlib.Path(PKG).parents[3]
CLUSTER_MAP = os.path.join(str(REPO), "FlyWheels/TanitAD_EvalFlyWheel/incoming/"
                           "2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json")
TERMS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance",
         "TTC": "time_to_collision_within_bound", "C": "comfort", "EP": "ego_progress",
         "PDMS": "score", "DDC": "driving_direction_compliance"}
def _published():
    """The ONE table: ``taniteval/taniteval/bench/plugins/navsim_v1.py::PUBLISHED`` (SPEC §4 —
    arXiv 2406.15349v2 Tab. 1 p. 7 / Tab. 3 p. 9, + the INHERITED HF leaderboard rows). Imported
    by path so this runs in the NAVSIM venv as well; the plugin is pure stdlib at module level.
    ⛔ No private fallback copy: a second copy of a published number is how two "independent"
    tables drift apart."""
    import importlib.util
    p = REPO / "taniteval" / "taniteval" / "bench" / "plugins" / "navsim_v1.py"
    spec = importlib.util.spec_from_file_location("w3_plugin_published", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    paper = {a: {k: v for k, v in mod.PUBLISHED[a].items() if isinstance(v, (int, float))}
             for a in ("CV", "HUMAN")}
    lb = {"CV": {"PDMS": mod.PUBLISHED["_leaderboard_INHERITED"]["CV"]}}
    return paper, lb, str(p), mod._verdict


PAPER, LB, PUBLISHED_SOURCE, _VERDICT = _published()
SPEEDS = ((0.0, 1.0), (1.0, 4.0), (4.0, 8.0), (8.0, 100.0))


def load(label: str):
    rows = list(csv.DictReader(open(os.path.join(RAW, label, f"{label}.csv"), encoding="utf-8")))
    tok = {r["token"]: r for r in rows if r["token"] != "average" and r["valid"] in ("True", "true", "1")}
    avg = [r for r in rows if r["token"] == "average"]
    hk_p = os.path.join(RAW, label, f"{label}_hooks.json")
    hooks = {}
    if os.path.exists(hk_p):
        for r in json.load(open(hk_p, encoding="utf-8"))["pdm_score_calls"]:
            if "token" in r:
                hooks[r["token"]] = r
    return tok, (avg[0] if avg else None), hooks


def means(tok: dict) -> dict:
    n = len(tok)
    return {k: 100.0 * sum(float(r[c]) for r in tok.values()) / n for k, c in TERMS.items()}


def verdict(got: float, want: float, dp: int) -> dict:
    """The ONE verdict function — the suite plugin's ``_verdict`` (SPEC §4 + AMENDMENT A1: both
    printed-precision readings, because the paper's convention is UNSETTLED). Imported, not
    re-implemented: two implementations of one rule is how two verdicts drift apart."""
    return _VERDICT(got, want, dp)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-run", required=True, help="suffix of the labels, e.g. navtest / smoke20")
    ap.add_argument("--arms", default="CV,HUMAN,STOP",
                    help="arm labels to load, e.g. CV,STOP,A1_ego_cmd (each reads "
                         "raw/<arm>_<split-run>/<arm>_<split-run>.csv)")
    ap.add_argument("--restrict-tokens", default=None,
                    help="JSON list (or {tokens:[…]}) — score EVERY arm on exactly these tokens. "
                         "⛔ Published verdicts are SUPPRESSED under a restriction.")
    ap.add_argument("--tag", default="", help="suffix for the output file, so a subset read never "
                                              "overwrites the full-split analysis")
    a = ap.parse_args(argv)
    sfx = a.split_run
    arms = {}
    for arm in [x for x in a.arms.split(",") if x]:
        lab = f"{arm}_{sfx}"
        if os.path.exists(os.path.join(RAW, lab, f"{lab}.csv")):
            arms[arm] = load(lab)
    if not arms:
        print("⛔ no arm CSVs found")
        return 2
    # ---- the token restriction: one token set, every arm, paired by construction --------- #
    restrict = None
    if a.restrict_tokens:
        doc = json.load(open(a.restrict_tokens, encoding="utf-8"))
        want = set(doc["tokens"] if isinstance(doc, dict) else doc)
        have = [set(t) for t, _, _ in arms.values()]
        keep = set.intersection(*have) & want if have else set()
        if not keep:
            print("⛔ the restriction leaves no token that every arm scored")
            return 2
        for arm, (tok, avg, hooks) in list(arms.items()):
            arms[arm] = ({t: r for t, r in tok.items() if t in keep},
                         None,                      # ⛔ the devkit average row is the FULL split's
                         {t: h for t, h in hooks.items() if t in keep})
        restrict = {"source": os.path.abspath(a.restrict_tokens), "n_requested": len(want),
                    "n_scored_by_every_arm": len(keep),
                    "n_requested_not_in_every_arm": len(want - keep),
                    "arms_restricted": sorted(arms),
                    "_note": ("A PAIRED subset: every arm is read on exactly these tokens. ⛔ It is "
                              "NOT the published split, so paper/leaderboard verdicts are "
                              "suppressed — a 12,146-token published mean and a subset mean are "
                              "different quantities, and comparing them is the scope error this "
                              "programme keeps paying for.")}
    out = {"split_run": sfx, "column": "score", "published_source": PUBLISHED_SOURCE,
           "restricted": restrict, "arms": {}, "estimator": {
        "estimator": "navsim_log_cluster_bootstrap (taniteval/adapters/navsim_ci.py, W2)",
        "cluster_unit": "log_name", "n": len(next(iter(arms.values()))[0]),
        "_note": ("REGISTERED for PDMS_v1_navtest while this package was running; per-arm intervals "
                  "and the paired deltas are below. It answers 'would another draw of LOGS say "
                  "this?' — blind to training and inference variance by construction.")},
        "stamps": {"tier": "T1-family", "loop": {"S1": "OPEN"}, "background": "non-reactive (logged)",
                   "closed_loop": False}}
    for arm, (tok, avg, hooks) in arms.items():
        m = means(tok)
        rec = {"n": len(tok), "x100": {k: round(v, 4) for k, v in m.items()},
               "devkit_average_row_score": float(avg["score"]) if avg else None}
        if restrict is not None:
            rec["published_verdicts"] = ("SUPPRESSED — a restricted token set is not the published "
                                         "split; see out['restricted']")
        elif sfx == "navtest" and arm in PAPER:
            rec["vs_paper_table1"] = {k: verdict(m[k], v, 1) for k, v in PAPER[arm].items()}
        if restrict is None and sfx == "navtest" and arm in LB:
            rec["vs_hf_leaderboard_INHERITED"] = {k: verdict(m[k], v, 4) for k, v in LB[arm].items()}
        out["arms"][arm] = rec
    # ---- C5: the navtest construction rule ------------------------------------------ #
    if "CV" in arms:
        viol = sorted(t for t, r in arms["CV"][0].items() if float(r["score"]) > 0.8)
        out["C5_cv_above_0p8"] = {"n": len(viol), "tokens": viol[:100], "n_total": len(arms["CV"][0])}
    if "HUMAN" in arms:
        viol = sorted(t for t, r in arms["HUMAN"][0].items() if float(r["score"]) < 0.8)
        out["C5_human_below_0p8"] = {"n": len(viol), "tokens": viol[:100],
                                     "n_total": len(arms["HUMAN"][0])}
    # ---- C6 + the STOP decomposition --------------------------------------------------- #
    if "STOP" in arms:
        stok, _, sh = arms["STOP"]
        m = means(stok)
        out["C6_stop_prediction"] = {"NC_ge_98": m["NC"] >= 98.0, "DAC_ge_98": m["DAC"] >= 98.0,
                                     "TTC_ge_95": m["TTC"] >= 95.0,
                                     "values_x100": {k: round(m[k], 4) for k in ("NC", "DAC", "TTC")}}
        ep1 = [t for t, h in sh.items() if h.get("max_compliant_progress_m", 1e9) <= 5.0]
        dec = {"n_tokens_with_hooks": len(sh),
               "n_max_compliant_progress_le_5m": len(ep1),
               "frac_max_compliant_progress_le_5m": (len(ep1) / len(sh)) if sh else None,
               "stop_mean_EP_x100_in_that_regime": (100 * st.mean(float(stok[t]["ego_progress"])
                                                                  for t in ep1 if t in stok) if ep1 else None),
               "stop_mean_EP_x100_elsewhere": (100 * st.mean(float(stok[t]["ego_progress"])
                                                             for t in stok if t not in set(ep1)))
               if len(stok) > len(ep1) else None,
               "stop_progress_m_median": (st.median(h["progress_raw_m"][1] for h in sh.values())
                                          if sh else None),
               "pdm_closed_progress_m_median": (st.median(h["progress_raw_m"][0] for h in sh.values())
                                                if sh else None),
               "floor_5_over_12_x100": round(500 / 12, 4)}
        bands = {}
        for lo, hi in SPEEDS:
            ts = [t for t, h in sh.items() if lo <= h["v0_mps"] < hi and t in stok]
            if ts:
                bands[f"{lo:g}-{hi:g} m/s"] = {
                    "n": len(ts), "STOP_x100": round(100 * st.mean(float(stok[t]["score"]) for t in ts), 4),
                    "STOP_C_x100": round(100 * st.mean(float(stok[t]["comfort"]) for t in ts), 4),
                    "CV_x100": (round(100 * st.mean(float(arms["CV"][0][t]["score"]) for t in ts
                                                    if t in arms["CV"][0]), 4) if "CV" in arms else None)}
        dec["per_t0_speed_band"] = bands
        if "CV" in arms:
            ctok = arms["CV"][0]
            common = sorted(set(stok) & set(ctok))
            d = [float(stok[t]["score"]) - float(ctok[t]["score"]) for t in common]
            dec["paired_STOP_minus_CV"] = {
                "n_common": len(common), "mean_delta_x100": round(100 * st.mean(d), 4),
                "W": sum(x > 1e-12 for x in d), "T": sum(abs(x) <= 1e-12 for x in d),
                "L": sum(x < -1e-12 for x in d)}
        out["STOP_decomposition"] = dec
    # ---- per log (needs the export's token -> log map) ---------------------------------- #
    exp_p = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
    if sfx == "navtest" and os.path.exists(exp_p):
        import gzip
        t2l = {t: r["log_name"] for t, r in json.load(gzip.open(exp_p))["tokens"].items()}
        for arm, (tok, _, _) in arms.items():
            per = {}
            for t, r in tok.items():
                per.setdefault(t2l.get(t, "?"), []).append(float(r["score"]))
            out["arms"][arm]["per_log_x100"] = {k: {"n": len(v), "PDMS": round(100 * st.mean(v), 4)}
                                                for k, v in sorted(per.items())}
    # ---- W2's registered log-cluster interval, per arm and paired against each floor ------ #
    try:
        sys.path.insert(0, os.path.join(REPO, "taniteval"))
        from adapters import navsim_ci as CI
        clusters = json.load(open(CLUSTER_MAP, encoding="utf-8"))["token_to_log_name"]
        for arm, (tok, _, _) in arms.items():
            scores = {t: float(r["score"]) for t, r in tok.items()}
            head = sum(scores.values()) / len(scores)
            out["arms"][arm]["interval"] = CI.interval_from_run(
                protocol="PDMS_v1_navtest", clusters_by_unit=clusters, scores=scores,
                official_value=head)
        for arm in arms:
            for floor in ("STOP", "CV"):
                if floor == arm or floor not in arms:
                    continue
                a_tok, b_tok = arms[arm][0], arms[floor][0]
                common = sorted(set(a_tok) & set(b_tok))
                ca = CI.single_stage_contributions([float(a_tok[t]["score"]) for t in common])
                cb = CI.single_stage_contributions([float(b_tok[t]["score"]) for t in common])
                out["arms"][arm].setdefault("paired_interval", {})[floor] = \
                    CI.paired_log_cluster_bootstrap(ca, cb, [clusters[t] for t in common],
                                                    aggregation=CI.AGG_SINGLE_STAGE)
    except Exception as e:                                             # noqa: BLE001
        out["estimator"]["error"] = f"{type(e).__name__}: {e}"[:300]
    p = os.path.join(RAW, f"analysis_{sfx}{('_' + a.tag) if a.tag else ''}.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "arms"} | {
        "arms": {k: v["x100"] for k, v in out["arms"].items()}}, indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
