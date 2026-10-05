"""D6 P4 analysis -- the registered reads and verdict of SPEC_P4_DRIVABLE_GATE.md s3-s4.   tanitad venv (numpy / pandas only).

    python d6_p4_analyze.py

Inputs (all produced by the P4 code, read as artifacts): raw/p4_agg.json (devkit EPDMS per arm + the G0 identity control), raw/p4_frame_<arm>.csv
(the compute_final_scores frame per arm), raw/p4_picks_<arm>.json, raw/p4_controls_D0.json, raw/p4_controls_KFRAME.json,
raw/p4_controls_EXPORT.json; the banked 50,400 R7_A1 / R7_A1_s1 frames (seed floor); raw/p1x_scored_s*.jsonl (tier-C cross-check).
Estimator: the P2 estimator (``d6_p2_analyze.paired_boot``: paired log-cluster bootstrap, B 2000, seed 0, 2.5/97.5 percentiles) -- EPDMS deltas on
per-mapping-key contributions (cluster = the log of the key's orig token), rate deltas on per-token indicators (cluster = the token's log).
``navsim_ci.paired_log_cluster_bootstrap`` is reported beside it for EPDMS.  Writes raw/p4_result.json and raw/p4_tables.md.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import d6_common as DC  # noqa: E402
import d6_p4_common as C  # noqa: E402
from d6_p2_analyze import paired_boot  # noqa: E402

RAW = C.RAW
ARMS = ("G0", "G1", "G1DER", "GORC")
BAR_DELTA = 0.020
BAR_DAC_PP = 0.030


def frame(arm: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(RAW, f"p4_frame_{arm}.csv"))
    assert len(df) == C.N_TOK and df["token"].is_unique, f"{arm} frame shape"
    return df.set_index("token")


def banked(arm: str) -> pd.DataFrame:
    df = pd.read_csv(C.FRAME.format(arm=arm))
    assert len(df) == C.N_TOK and df["token"].is_unique
    return df.set_index("token")


def official_combined(arm: str) -> float:
    s = pd.read_csv(C.SUMMARY.format(arm=arm))
    return float(s.loc[s["token"] == "extended_pdm_score_combined", "score"].iloc[0])


def key_contrib(df: pd.DataFrame, mapping, ci) -> np.ndarray:
    rows = {t: {"score": float(r["score"]), "weight": float(r["weight"])} for t, r in df[["score", "weight"]].iterrows()}
    return ci.two_stage_key_contributions(rows, mapping, "score")


def rate(df, col):
    return float((df[col] == 0).mean())


def main() -> int:
    ci = DC.load_ci()
    inp, mapping = DC.load_inputs()
    logs_tok = pd.Series({t: r["log_name"] for t, r in inp.items()})
    key_logs = pd.Series([logs_tok[str(e[0])] for e in mapping])
    agg = json.load(open(os.path.join(RAW, "p4_agg.json"), encoding="utf-8"))
    res = {"tag": "MEASURED", "spec": "SPEC_P4_DRIVABLE_GATE.md sha256 f613a15088062726a3340a8dd92fd59b3cd9c08a3c8f99754d078e2f88291a5c",
           "impl_choices": "raw/p4_impl_choices.md", "tier": "NavSim navhard two-stage, LOCAL devkit scoring (T2-class closed-loop metric per SPEC s5)",
           "estimator": "P2 paired log-cluster bootstrap (d6_p2_analyze.paired_boot), B 2000, seed 0; question answered: another draw of LOGS",
           "arms": {}, "controls": {}}
    fr = {a: frame(a) for a in ARMS if os.path.exists(os.path.join(RAW, f"p4_frame_{a}.csv"))}
    order = list(fr["G0"].index)
    contrib = {a: key_contrib(fr[a], mapping, ci) for a in fr}
    # ---- seed floor (banked) ---------------------------------------------------------------------------------------------------
    A1, A1s = banked("R7_A1"), banked("R7_A1_s1")
    e_a1, e_s1 = official_combined("R7_A1"), official_combined("R7_A1_s1")
    floor = abs(e_a1 - e_s1)
    dac_floor = abs(rate(A1, "drivable_area_compliance") - rate(A1s.loc[A1.index], "drivable_area_compliance"))
    res["seed_floor"] = {"EPDMS_R7_A1": e_a1, "EPDMS_R7_A1_s1": e_s1, "abs_delta": floor,
                         "two_x": 2 * floor, "DAC0_abs_delta": dac_floor,
                         "source": "banked step-50,400 navhard summary CSVs (extended_pdm_score_combined), same 5,912 tokens"}
    # ---- per arm -----------------------------------------------------------------------------------------------------------------
    strata = {"S1": set(C.tokens_file("spec_tokens_p1x_S1.txt")), "S1b": set(C.tokens_file("spec_tokens_p1x_S1b.txt")),
              "S2": set(C.tokens_file("spec_tokens_p1x_S2.txt"))}
    strata["rest"] = set(order) - strata["S1"] - strata["S1b"] - strata["S2"]
    g0 = fr["G0"]
    for a in fr:
        df = fr[a]
        picks = (None if a == "G0" else json.load(open(os.path.join(RAW, f"p4_picks_{a}.json"), encoding="utf-8")))
        ch = pd.Series({t: (picks["picks"][t]["changed"] if picks else False) for t in order})
        epd = float(agg[a]["EPDMS"])
        rec = {"EPDMS": epd, "EPDMS_from_key_contributions": float(ci.official_aggregate(contrib[a])),
               "DAC0_rate": rate(df, "drivable_area_compliance"), "NC0_rate": rate(df, "no_at_fault_collisions"),
               "DDC0_rate": rate(df, "driving_direction_compliance"), "TLC0_rate": rate(df, "traffic_light_compliance"),
               "EP_mean_tokens": float(df["ego_progress"].mean()),
               "EP_stage1_devkit": agg[a]["stage1"].get("ego_progress"), "EP_stage2_devkit": agg[a]["stage2"].get("ego_progress"),
               "changed_share": float(ch.mean()), "n_changed": int(ch.sum())}
        if picks:
            rec["gate_summary"] = picks["summary"]
        if a != "G0":
            d_key = pd.Series(contrib[a] - contrib["G0"])
            pt, cil = paired_boot(d_key, key_logs, B=2000, seed=0)
            rec["dEPDMS_vs_G0"] = {"delta": epd - float(agg["G0"]["EPDMS"]), "delta_keymean": pt, "ci95": cil}
            pr = ci.paired_log_cluster_bootstrap(contrib[a], contrib["G0"], list(key_logs), aggregation=ci.AGG_TWO_STAGE,
                                                 n_boot=2000, seed=0)
            rec["dEPDMS_vs_G0_navsim_ci"] = {k: pr.get(k) for k in ("status", "delta", "point", "lo", "hi", "separated",
                                                                    "separated_scope", "n_clusters", "n_units") if k in pr}
            for col, nm in (("drivable_area_compliance", "DAC0"), ("no_at_fault_collisions", "NC0")):
                d = (df.loc[order, col] == 0).astype(float) - (g0.loc[order, col] == 0).astype(float)
                pt, cil = paired_boot(d, logs_tok.loc[order], B=2000, seed=0)
                rec[f"d{nm}_vs_G0"] = {"delta": pt, "ci95": cil}
            d = df.loc[order, "ego_progress"] - g0.loc[order, "ego_progress"]
            pt, cil = paired_boot(d, logs_tok.loc[order], B=2000, seed=0)
            rec["dEP_mean_vs_G0"] = {"delta": pt, "ci95": cil}
        st = {}
        for s, toks in strata.items():
            tl = [t for t in order if t in toks]
            sub = df.loc[tl]
            st[s] = {"n": len(tl), "DAC0": rate(sub, "drivable_area_compliance"), "NC0": rate(sub, "no_at_fault_collisions"),
                     "EP_mean": float(sub["ego_progress"].mean()), "score_mean": float(sub["score"].mean()),
                     "changed_share": float(ch.loc[tl].mean())}
            if a != "G0":
                d = (sub["drivable_area_compliance"] == 0).astype(float) - (g0.loc[tl, "drivable_area_compliance"] == 0).astype(float)
                pt, cil = paired_boot(d, logs_tok.loc[tl], B=2000, seed=0)
                st[s]["dDAC0_vs_G0"] = {"delta": pt, "ci95": cil}
        rec["strata"] = st
        res["arms"][a] = rec

    # ---- controls -------------------------------------------------------------------------------------------------------------------
    ic = agg["G0"]["identity_control"]
    res["controls"]["G0_identity"] = {"PASS": ic["PASS"], "max_abs_diff_any_column": ic["frame_vs_banked"]["max_abs_diff_any_column"],
                                      "EPDMS_rebuilt": ic["EPDMS_rebuilt"], "EPDMS_banked": ic["EPDMS_official_banked"],
                                      "columns": {k: v for k, v in ic["frame_vs_banked"]["per_column"].items()}}
    for nm in ("D0", "KFRAME", "EXPORT"):
        p = os.path.join(RAW, f"p4_controls_{nm}.json")
        res["controls"][nm] = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else "MISSING"
    # tier-C cross-check: an arm's exact row vs P1' tier C where the candidate coincides
    tc = {}
    for p in sorted(glob.glob(os.path.join(RAW, "p1x_scored_s*.jsonl"))):
        for ln in open(p, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "OK" and r.get("tierc"):
                tc[r["token"]] = r["tierc"]
    xc = {"n_compared": 0, "n_equal": 0, "max_abs": 0.0}
    for a in ("G1", "G1DER", "GORC"):
        if a not in fr:
            continue
        picks = json.load(open(os.path.join(RAW, f"p4_picks_{a}.json"), encoding="utf-8"))["picks"]
        for t, r in picks.items():
            if r["changed"] and t in tc and tc[t]["cand"] == r["new"]:
                d = max(abs(float(fr[a].loc[t, k]) - float(tc[t][k])) for k in DC.SUB.values())
                xc["n_compared"] += 1
                xc["n_equal"] += int(d == 0.0)
                xc["max_abs"] = max(xc["max_abs"], d)
    res["controls"]["P1prime_tierC_crosscheck"] = xc

    # ---- verdict (SPEC s4 literals) -------------------------------------------------------------------------------------------------
    def clauses(a):
        r = res["arms"][a]
        d = r["dEPDMS_vs_G0"]["delta"]
        lo, hi = r["dEPDMS_vs_G0"]["ci95"]
        dd = r["dDAC0_vs_G0"]["delta"]
        dlo, dhi = r["dDAC0_vs_G0"]["ci95"]
        return {"dEPDMS_ge_0.020": d >= BAR_DELTA, "dEPDMS_ci_excludes_0": lo > 0 or hi < 0,
                "dEPDMS_gt_2x_seed_floor": d > 2 * floor, "DAC0_falls_ge_3pp": dd <= -BAR_DAC_PP,
                "DAC0_ci_excludes_0": dlo > 0 or dhi < 0}
    g0_ok = bool(ic["PASS"])
    out = {"G0_reproduces": g0_ok}
    if "G1DER" in fr:
        dder = res["arms"]["G1DER"]["dEPDMS_vs_G0"]["delta"]
        out["G1DER_delta"] = dder
        out["G1DER_behaves(delta<=seed_floor)"] = bool(dder <= floor)
    if "G1" in fr:
        cg1 = clauses("G1")
        out["G1_clauses"] = cg1
        g1_pass = all(cg1.values())
    else:
        g1_pass = None
    if "GORC" in fr:
        corc = clauses("GORC")
        out["GORC_clauses_same_literals"] = corc
        out["GORC_passes_same_literals"] = all(corc.values())
    if not g0_ok or (("G1DER_behaves(delta<=seed_floor)" in out) and not out["G1DER_behaves(delta<=seed_floor)"]):
        v = "VOID"
    elif g1_pass is None or "G1DER" not in fr:
        v = "INCOMPLETE"                 # no verdict without BOTH the treatment and the deliberate regression
    else:
        v = "PASS" if g1_pass else "FAIL"
    out["verdict"] = v
    if v == "PASS":
        out["committed_meaning"] = ("PASS => the model's own map knows where it may drive and the selector ignores it: the gate enters refcv8's "
                                    "evaluation as an opt-in inference row on BOTH NavSim rows; a trained DAC-aware selector term becomes a named "
                                    "refcv8 lever.")
    elif v == "FAIL":
        out["committed_meaning"] = (("FAIL with G-ORC passing => the gate idea works but the predicted map is the bottleneck: WP-D map levers move "
                                     "up in refcv8; the trained critic remains the lever.") if out.get("GORC_passes_same_literals") else
                                    ("FAIL with G-ORC also failing => a drivable gate on this fan does not reach the official score; the next "
                                     "lever is the trained listwise selector with sub-score critics (X1), not a gate."))
    res["verdict"] = out
    json.dump(res, open(os.path.join(RAW, "p4_result.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
