#!/usr/bin/env python3
"""navtest (NAVSIM v1.1) statistics for refcv6 arms against W3's BANKED floors (TANITAD VENV).

PDMS = mean of the official ``score`` column ×100 (W3's ``PDMS_v1_navtest``; v1.1 has no
``pdm_score``). Floors: W3's full-split CSVs (CV 20.6517, STOP 61.8202, HUMAN 94.5514; refcv4b A1
58.9738 for context), RESTRICTED to the refcv6 arm's token set so every arm is read on identical
scenes. Interval: the settled single-stage ``navsim_log_cluster_bootstrap`` and its paired variant
(``taniteval/adapters/navsim_ci.py``, IMPORTED; unit = log, B 2000, seed 0). Answers "another draw
of LOGS?" only; the inference-seed floor is read from an ``_s1`` arm when one exists.

    python code/parse_navtest6.py --arm-csv raw/navtest/r6s1000_R6_A1_sub200/r6s1000_R6_A1_sub200.csv \
        --label PIPELINE-VALIDATION --out raw/summary_navtest_s1000_sub200.json
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

W3RAW = "D:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-v1-navtest/raw"
FLOORS = {"CV": f"{W3RAW}/CV_navtest/CV_navtest.csv", "STOP": f"{W3RAW}/STOP_navtest/STOP_navtest.csv",
          "HUMAN": f"{W3RAW}/HUMAN_navtest/HUMAN_navtest.csv",
          "refcv4b_A1": f"{W3RAW}/A1_navtest/A1_navtest.csv"}
INPUTS = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
TERMS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "EP": "ego_progress",
         "TTC": "time_to_collision_within_bound", "C": "comfort",
         "DDC": "driving_direction_compliance", "PDMS": "score"}


#: every refcv6 number carries this stamp (Master Mind, 2026-09-26)
MODEL_STAMP = ('"F3 detach-only, F4 on the last layer only" — Master Mind audit 2026-09-26 (GOALS_AND_CLAIMS D-REFCV6-F3-WHITELIST, D-REFCV6-LABEL-CLOCK; landed 9d16c441): the F3 per-stage cascade loss never ran (decoder stages 0-2 frozen at init) and tactical labels are read ~0.37 s early; INHERITED')

def read(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "pdm_score" in df.columns:
        raise SystemExit(f"{path}: carries pdm_score — refusing")
    df = df[df.token != "average"].copy()
    df = df[df.valid.astype(str).isin(("True", "true", "1"))]
    return df.set_index("token")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-csv", required=True)
    ap.add_argument("--seed-csv", default="", help="the same arm at inference seed 1 (optional)")
    ap.add_argument("--extra", nargs="*", default=[], help="name=csv of further refcv6 arms")
    ap.add_argument("--tokens", default="", help="restrict EVERY arm to this W3-format subset "
                    "({'tokens': [...]}) — e.g. SPEC §12's 200-token diagnostic arms beside a "
                    "full-split R6_A1")
    ap.add_argument("--census", default="", help="code/vmax_oracle.py output: every R6 pair is "
                    "ALSO split by max-speed census group (SPEC §12 addendum)")
    ap.add_argument("--map", default="", help="speed_limits_navtest.json (needed with --census)")
    ap.add_argument("--bridge", default="", help="the bridge dir: each arm's DEVICE / precision is "
                    "read from its seam manifest and printed beside its score")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    spec = importlib.util.spec_from_file_location(
        "navsim_ci6", "D:/Projects/TanitAD/taniteval/adapters/navsim_ci.py")
    ci = importlib.util.module_from_spec(spec)
    sys.modules["navsim_ci6"] = ci
    spec.loader.exec_module(ci)
    tok2log = {t: r["log_name"] for t, r in
               json.load(gzip.open(INPUTS, "rt", encoding="utf-8"))["tokens"].items()}
    arms = {"R6_A1": read(a.arm_csv)}
    if a.seed_csv:
        arms["R6_A1_s1"] = read(a.seed_csv)
    for e in a.extra:
        k, v = e.split("=", 1)
        arms[k] = read(v)
    toks = sorted(arms["R6_A1"].index)
    if a.tokens:
        sub = json.load(open(a.tokens, encoding="utf-8"))
        sub = sub["tokens"] if isinstance(sub, dict) else sub
        if set(sub) - set(toks):
            raise SystemExit(f"⛔ R6_A1 lacks {len(set(sub) - set(toks))} of the --tokens subset")
        toks = sorted(sub)
    for k, v in FLOORS.items():
        arms[k] = read(v)
    for k, df in arms.items():
        miss = set(toks) - set(df.index)
        if miss:
            raise SystemExit(f"⛔ {k} lacks {len(miss)} of the refcv6 tokens — not paired")
    clusters = [tok2log[t] for t in toks]
    out = {"_label": a.label, "protocol": "PDMS_v1_navtest", "n_tokens": len(toks),
           "model_as_trained": MODEL_STAMP,
           "tokens_subset": os.path.abspath(a.tokens) if a.tokens else None,
           "n_logs": len(set(clusters)), "floors_source": FLOORS,
           "estimator": {"name": ci.ESTIMATOR, "paired": ci.PAIRED_ESTIMATOR,
                         "unit": ci.CLUSTER_UNIT, "aggregation": ci.AGG_SINGLE_STAGE,
                         "module": os.path.abspath(ci.__file__)},
           "arms": {}, "pairs": {}}
    for k, df in arms.items():
        d = df.loc[toks]
        rec = {t: round(100 * float(d[c].mean()), 4) for t, c in TERMS.items()}
        rec["zeroed_by_NC_or_DAC"] = int(((d[TERMS["NC"]] == 0) | (d[TERMS["DAC"]] == 0)).sum())
        if k.startswith("R6_"):
            man = os.path.join(a.bridge, f"seam_{k}.manifest.json") if a.bridge else ""
            m = json.load(open(man, encoding="utf-8")) if man and os.path.exists(man) else {}
            rec["device"] = "/".join(m.get("device", ["UNKNOWN"])) + " " + "/".join(
                m.get("precision", ["UNKNOWN"]))
        else:
            rec["device"] = "banked floor (W3, model-free or log replay)"
        rec["interval"] = ci.log_cluster_bootstrap(
            d["score"].to_numpy(dtype=float), clusters, aggregation=ci.AGG_SINGLE_STAGE,
            official_value=float(d["score"].mean()))
        out["arms"][k] = rec
    for x in [k for k in arms if k.startswith("R6_")]:
        for y in ("STOP", "CV", "HUMAN", "refcv4b_A1") + tuple(k for k in arms
                                                            if k.startswith("R6_") and k != x):
            ax = arms[x].loc[toks, "score"].to_numpy(dtype=float)
            ay = arms[y].loc[toks, "score"].to_numpy(dtype=float)
            dd = ax - ay
            out["pairs"][f"{x}__minus__{y}"] = {
                "delta_x100": round(100 * float(dd.mean()), 4),
                "wins": int((dd > 1e-12).sum()), "ties": int((np.abs(dd) <= 1e-12).sum()),
                "losses": int((dd < -1e-12).sum()),
                "interval": ci.paired_log_cluster_bootstrap(
                    ax, ay, clusters, aggregation=ci.AGG_SINGLE_STAGE,
                    official_a=float(ax.mean()), official_b=float(ay.mean()))}
    if a.census:
        from tanitad.refs.refcv6_max_speed import speed_max_bin, SPEED_MAX_STEPS_KMH_V6
        ora = json.load(open(a.census, encoding="utf-8"))["tokens"]
        mp = json.load(open(a.map, encoding="utf-8"))["tokens"]

        def group(t):
            o, m = ora.get(t, {}), mp.get(t, {})
            if o.get("status") != "limit":
                return "no_oracle"
            if m.get("status") != "limit":
                return "no_map (A1 gets the all-zero 'unknown' row)"
            mb = SPEED_MAX_STEPS_KMH_V6[speed_max_bin(float(m["speed_limit_mps"]))[0]]
            return ("map_above_oracle" if mb > o["bin_kmh"] else
                    "map_equals_oracle" if mb == o["bin_kmh"] else "map_below_oracle")
        g_of = {t: group(t) for t in toks}
        out["census_groups"] = {g: sum(1 for t in toks if g_of[t] == g) for g in sorted(set(g_of.values()))}
        out["pairs_by_census_group"] = {}
        for k in [k for k in out["pairs"] if k.startswith("R6_") and k.split("__minus__")[1].startswith("R6_")]:
            x, y = k.split("__minus__")
            blk = {}
            for g in sorted(set(g_of.values())):
                ts = [t for t in toks if g_of[t] == g]
                dd = (arms[x].loc[ts, "score"].to_numpy(dtype=float)
                      - arms[y].loc[ts, "score"].to_numpy(dtype=float))
                blk[g] = {"n": len(ts), "delta_x100": round(100 * float(dd.mean()), 4) if ts else None,
                          "wins": int((dd > 1e-12).sum()), "ties": int((np.abs(dd) <= 1e-12).sum()),
                          "losses": int((dd < -1e-12).sum())}
            out["pairs_by_census_group"][k] = blk
        out["census_file"] = os.path.abspath(a.census)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    for k, v in out["arms"].items():
        iv = v["interval"]
        print(f"{k:12s} PDMS={v['PDMS']:.4f} NC={v['NC']:.2f} DAC={v['DAC']:.2f} EP={v['EP']:.2f} "
              f"TTC={v['TTC']:.2f} C={v['C']:.2f} zeroed={v['zeroed_by_NC_or_DAC']} "
              f"CI=[{iv.get('lo')}, {iv.get('hi')}] {iv.get('status')}")
    for k, v in out["pairs"].items():
        iv = v["interval"]
        print(f"{k:28s} d={v['delta_x100']:+.4f} W/T/L={v['wins']}/{v['ties']}/{v['losses']} "
              f"CI=[{iv.get('lo')}, {iv.get('hi')}] {iv.get('status')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
