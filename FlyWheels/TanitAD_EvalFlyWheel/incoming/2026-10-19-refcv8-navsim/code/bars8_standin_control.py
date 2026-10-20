#!/usr/bin/env python3
"""bars8_standin_control -- the full-split PIPELINE CONTROL of navsim_bars8 (TANITAD VENV, CPU; reads banked CSVs, writes two files).

refcv7-50,400 arms WEAR refcv8 names (R8_LEGAL = the R7_VMAXOFF LEGAL arm, R8_LEGAL_s1 = R7_A1_s1, R8_PRIV = R7_A1) so the layer runs on the
REAL 12,146-token / 136-log / 44-drive split with every interval, stratum and bar rule switched on.  It is a control on the LAYER and
carries the label 'VALIDATION ONLY -- PIPELINE CONTROL ... NOT a refcv8 result'; no number in its output is a refcv8 number.
Known values it must read (INHERITED, MODEL_REGISTRY 15.4.1): LEGAL PDMS 70.2048, paired vs STOP +8.3846 [6.36, 10.24].

    python code/bars8_standin_control.py --out raw/bars_control_refcv7_standin_FULLSPLIT
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import navsim_bars8 as B
t0 = time.time()
ti = B.load_inputs(B.DEFAULT_INPUTS)
stop_rows = B.read_rows_jsonl(os.path.join(B.DEFAULT_W3, "STOP_navtest", "STOP_navtest_rows.jsonl"))
ti["mcp"] = pd.Series({t: float(r["max_compliant_progress_m"]) for t, r in stop_rows.items()})
R7 = os.path.dirname(os.path.dirname(B.DEFAULT_R7))
arms = {"R8_LEGAL": B.read_scores_csv(B.DEFAULT_R7),                                   # STAND-IN: refcv7-50,400 LEGAL plays the bar arm
        "R8_LEGAL_s1": B.read_scores_csv(os.path.join(R7, "r7thor_s50400_R7_A1_s1", "r7thor_s50400_R7_A1_s1.csv")),   # NOT its seed twin
        "R8_PRIV": B.read_scores_csv(os.path.join(R7, "r7thor_s50400_R7_A1", "r7thor_s50400_R7_A1.csv")),            # stand-in for a second arm
        B.R7_ARM: B.read_scores_csv(B.DEFAULT_R7)}
for ref in ("STOP", "CV", "HUMAN"):
    arms[ref] = B.read_scores_csv(os.path.join(B.DEFAULT_W3, f"{ref}_navtest", f"{ref}_navtest.csv"))
meta = {"label": "VALIDATION ONLY -- PIPELINE CONTROL: refcv7-50,400 arms wearing refcv8 names, NOT a refcv8 result", "step": 50400, "weights": "ema"}
res = B.evaluate_navtest(arms, ti, meta, {"n_boot": 2000, "seed": 0})
res["meta"] = meta; res["controls"] = {"note": "stand-in control; arm names are borrowed (R8_LEGAL=R7_VMAXOFF, R8_LEGAL_s1=R7_A1_s1, R8_PRIV=R7_A1)"}
res["validation_only"] = True; res["ineligible_because"] = ["control run"]; res["label"] = meta["label"]
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); out = ap.parse_args().out
os.makedirs(out, exist_ok=True)
json.dump(res, open(os.path.join(out, "NAVSIM_BARS_control_refcv7_standin_FULLSPLIT.json"), "w", encoding="utf-8"), indent=1, default=str)
open(os.path.join(out, "NAVSIM_BARS_control_refcv7_standin_FULLSPLIT.md"), "w", encoding="utf-8").write(B.render_md(res, "control_refcv7_standin_FULLSPLIT"))
print("wall_s", round(time.time() - t0, 1))
