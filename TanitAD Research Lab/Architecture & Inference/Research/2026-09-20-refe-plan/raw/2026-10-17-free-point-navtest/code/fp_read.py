#!/usr/bin/env python3
"""STAGE E of the free-point navtest read (PLAN.md): the pre-registered rows, applied mechanically.  CPU only; reuses `stage3_full` (read_pair, load_table, consistency, compute_families, families_vs) so the
estimator, the sub-score set and the families calls are the ones every banked navtest read used.

    python fp_read.py selftest                       -> the known-value check: banked u400 minus banked deployed through THIS code must equal RESULT_SFT4 to the bit, + the row logic on synthetic tables
    python fp_read.py read --work W --out-dir O      -> O/freepoint_result.json + O/RESULT_TABLES.md   (needs W/seam_*.npz and the harness CSVs `refe_fp_<POINT>`)

Rows (PREREG_FREEPOINT.md, verbatim): DEPLOYABLE-SAFE(point) <=> against BASE: PDMS CI lower bound > 0 AND NC, DAC, TTC CI upper bounds >= 0.
FP-R0B if R0B is safe; else FP-M25 if M25 is; else FP-NONE.   Lever view (no row): R0B - C0A and M25 - C0A.
Estimator: proxy_eval.boot via stage3_full.read_pair: paired log-cluster bootstrap over the 136 logs, 10,000 resamples, seed 20260927 ("another draw of LOGS" only).
"""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import sys

import numpy as np

PK = "C:/fp/PK"
DATA = "E:/Projects/TanitAD/data/refe_navtest"
EXPORT = "E:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
POINTS = ("BASE", "C0A", "R0B", "M25")
KNOWN = {"mean": 3.7908187056340563, "lo": 3.0470289517787528, "hi": 4.542726430876108}          # RESULT_SFT4 / P2_READOUT_RUNBOOK s.0, to the bit
FAM_KEYS = {"LONGITUDINAL": [("longitudinal", k) for k in ("speed_mae_mps", "speed_bias_mps", "along_mae_m", "along_final_bias_m", "accel_mae_mps2", "target_speed_acc.within_1.0_mps")],
            "LATERAL": [("lateral", k) for k in ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "cross_mae_m", "cross_final_mae_m")]}
SUBS = ("NC", "DAC", "EP", "TTC", "C", "DDC")
LONG_MEMBERS, LAT_MEMBERS = ("EP", "TTC", "C"), ("DAC", "DDC")


def load_s3():
    sys.path.insert(0, f"{PK}/eval")
    sys.path.insert(0, f"{PK}/refe")
    spec = importlib.util.spec_from_file_location("stage3_full", f"{PK}/eval/stage3_full.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules["stage3_full"] = m
    spec.loader.exec_module(m)
    return m


def is_safe(pair: dict) -> dict:
    """the prereg's DEPLOYABLE-SAFE, from a read_pair result"""
    p = pair["primary"]["ci95"]
    s = pair["subscores"]
    conds = {"PDMS_lo_gt_0": bool(p[0] > 0), "NC_hi_ge_0": bool(s["NC"]["ci95"][1] >= 0), "DAC_hi_ge_0": bool(s["DAC"]["ci95"][1] >= 0), "TTC_hi_ge_0": bool(s["TTC"]["ci95"][1] >= 0)}
    return {"safe": all(conds.values()), "conditions": conds}


def row_of(safe: dict) -> str:
    if safe["R0B"]["safe"]:
        return "FP-R0B"
    if safe["M25"]["safe"]:
        return "FP-M25"
    return "FP-NONE"


def fmt_ci(r: dict) -> str:
    return f"{r['mean']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]"


def pair_table(title: str, pairs: dict) -> str:
    lines = [f"**{title}**", "", "| point | PDMS | NC | DAC | EP | TTC | C | DDC | better/worse/tied | n logs |", "|---|---|---|---|---|---|---|---|---|---|"]
    for nm, pr in pairs.items():
        p, s = pr["primary"], pr["subscores"]
        lines.append(f"| {nm} | **{fmt_ci(p)}** | " + " | ".join(fmt_ci(s[k]) + (" ADV" if s[k]["adverse"] else "") for k in SUBS) + f" | {p['better']}/{p['worse']}/{p['tied']} | {p['n_logs']} |")
    return "\n".join(lines)


def level_table(tabs: dict, toks: list) -> str:
    lines = ["| point | PDMS | NC | DAC | EP | TTC | C | DDC | zero-PDMS picks |", "|---|---|---|---|---|---|---|---|---|"]
    for nm, t in tabs.items():
        sel = [x for x in toks if x in t]
        m = {k: np.mean([t[x][k] for x in sel]) for k in ("PDMS",) + SUBS}
        lines.append(f"| {nm} | {m['PDMS']:.4f} | " + " | ".join(f"{m[k]:.2f}" for k in SUBS) + f" | {sum(t[x]['PDMS'] == 0 for x in sel)} |")
    return "\n".join(lines)


def fam_summary(fams: dict) -> str:
    out = []
    for fam, keys in FAM_KEYS.items():
        out += [f"**{fam}** (families6, x = point estimate [episode-cluster 95 % interval], unpaired)", "", "| metric | " + " | ".join(fams) + " |", "|---|" + "---|" * len(fams)]
        for sec, k in keys:
            cells = []
            for nm, f in fams.items():
                c = (((f or {}).get("families", {}).get("refcv6", {}).get(sec) or {}).get("ci") or {}).get("components", {}).get(k)
                cells.append(f"{c['mean']:.4g} [{c['lo']:.4g}, {c['hi']:.4g}]" if c else "n/a")
            out.append(f"| {sec}.{k} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["**TACTICAL** (families6; rows WITHOUT an interval, reported not tested)", "", "| decision | " + " | ".join(fams) + " |", "|---|" + "---|" * len(fams)]
    for k in ("lateral_decision", "longitudinal_decision", "maneuver_5way_collapsed"):
        cells = []
        for nm, f in fams.items():
            t = (((f or {}).get("families", {}).get("refcv6", {}).get("tactical") or {}).get(k) or {})
            cells.append(f"acc {t.get('accuracy', float('nan')):.4f}, kappa {t.get('kappa', float('nan')):.4f}" if t else "n/a")
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    out += ["", "**STRATEGIC: PAUSED** (PI 2026-09-19, PI_DECISION_QUEUE item 25) and UNAVAILABLE in NAVSIM (no strategic decision is scored). Reported as paused, never as a pass."]
    return "\n".join(out)


def cmd_selftest(a):
    S3 = load_s3()
    u, d = S3.pick_baseline("u400"), S3.pick_baseline("deployed")
    ok_pins = bool(u["pin_ok"] and d["pin_ok"])
    tu, td = S3.load_table(u["csv"]), S3.load_table(d["csv"])
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    toks = sorted(tu)
    tl = {t: exp[t]["log_name"] for t in toks}
    pr = S3.read_pair(tu, td, toks, tl)
    p = pr["primary"]
    dev = max(abs(p["mean"] - KNOWN["mean"]), abs(p["ci95"][0] - KNOWN["lo"]), abs(p["ci95"][1] - KNOWN["hi"]))
    known_ok = dev <= 1e-12                      # the runbook's own gate (P2_READOUT_RUNBOOK s.0): tolerance 1e-12; np.mean / np.percentile are not bitwise across numpy builds
    print({"pins_ok": ok_pins, "u400_minus_deployed": [p["mean"], p["ci95"]], "max_abs_dev_from_RESULT_SFT4": dev, "equals_RESULT_SFT4_within_1e-12": known_ok})
    # the row logic on hand-built read_pair-shaped dicts
    def fake(plo, nch, dach, ttch):
        sub = {k: {"ci95": [-1.0, 1.0], "adverse": False} for k in SUBS}
        sub["NC"]["ci95"], sub["DAC"]["ci95"], sub["TTC"]["ci95"] = [-1, nch], [-1, dach], [-1, ttch]
        return {"primary": {"ci95": [plo, plo + 1]}, "subscores": sub}
    cases = [("R0B safe", fake(0.1, 0.1, 0.1, 0.1), fake(0.1, 0.1, 0.1, 0.1), "FP-R0B"),
             ("R0B PDMS lo<=0, M25 safe", fake(-0.1, 0.1, 0.1, 0.1), fake(0.1, 0.1, 0.1, 0.1), "FP-M25"),
             ("R0B TTC separated below 0, M25 not", fake(0.5, 0.1, 0.1, -0.1), fake(0.5, 0.1, 0.1, -0.1), "FP-NONE"),
             ("R0B DAC separated below 0, M25 safe", fake(0.5, 0.1, -0.2, 0.1), fake(0.5, 0.1, 0.1, 0.1), "FP-M25"),
             ("hi == 0 counts as NOT separated", fake(0.5, 0.0, 0.0, 0.0), fake(-1, 0, 0, 0), "FP-R0B")]
    rows_ok = True
    for nm, r, m, want in cases:
        got = row_of({"R0B": is_safe(r), "M25": is_safe(m)})
        rows_ok &= (got == want)
        print(f"  [{'PASS' if got == want else 'FAIL'}] {nm}: {got} (want {want})")
    good = ok_pins and known_ok and rows_ok
    print("ZZREAD_SELFTEST_" + ("PASS" if good else "FAIL"))
    return 0 if good else 1


def cmd_read(a):
    S3 = load_s3()
    work, out = a.work, a.out_dir
    os.makedirs(out, exist_ok=True)
    exp = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
    full = json.load(open(f"{DATA}/navtest_full_tokens.json", encoding="utf-8"))
    toks = sorted(full["tokens"] if isinstance(full, dict) else full)
    tl = {t: exp[t]["log_name"] for t in toks}
    banked = S3.pick_baseline("u400")
    tabs, seams, fams = {}, {}, {}
    gates = {}
    for p in POINTS:
        label = f"refe_fp_{p}"
        csvp = f"{DATA}/score/{label}/{label}.csv"
        if not os.path.exists(csvp):
            print(f"REFUSED: {csvp} missing")
            return 2
        st = json.load(open(f"{work}/score_logs/{label}.status.json"))
        gates[f"harness_PASS_{p}"] = st.get("status") == "PASS"
        tabs[p] = S3.load_table(csvp)
        seams[p] = f"{work}/seam_{p}.npz"
    gates["rows_12146_each"] = all(len([t for t in toks if t in tabs[p]]) == len(toks) for p in POINTS)
    banked_tab = S3.load_table(banked["csv"])
    # --- identity controls read from the files the earlier stages wrote
    ident = {}
    for f in ("IC1", "IC3_IC4", "IC5", "IC6", "D1_D3"):
        pth = f"{work}/{f}.json"
        ident[f] = json.load(open(pth)) if os.path.exists(pth) else {"missing": True}
    # --- the pairs
    pairs_base = {p: S3.read_pair(tabs[p], tabs["BASE"], toks, tl) for p in ("C0A", "R0B", "M25")}
    pairs_banked = {p: S3.read_pair(tabs[p], banked_tab, toks, tl) for p in ("BASE", "C0A", "R0B", "M25")}
    lever = {"R0B-C0A": S3.read_pair(tabs["R0B"], tabs["C0A"], toks, tl), "M25-C0A": S3.read_pair(tabs["M25"], tabs["C0A"], toks, tl),
             "R0B-M25": S3.read_pair(tabs["R0B"], tabs["M25"], toks, tl)}
    safe = {p: is_safe(pairs_base[p]) for p in pairs_base}
    safe_banked = {p: is_safe(pairs_banked[p]) for p in ("C0A", "R0B", "M25")}
    row = row_of(safe)
    row_banked = row_of(safe_banked)
    cons = S3.consistency(__import__("pathlib").Path(seams["BASE"]), __import__("pathlib").Path(banked["seam"]), tabs["BASE"], banked_tab, toks)
    # --- families
    famdir = __import__("pathlib").Path(out) / "families"
    fam_toks = set(toks)
    for p in POINTS:
        fams[p] = S3.compute_families(__import__("pathlib").Path(seams[p]), f"fp_{p}", famdir, fam_toks)
    fam_vs = {p: S3.families_vs(fams["BASE"], fams[p]) for p in ("C0A", "R0B", "M25")}
    res = {"tool": "fp_read/1", "tier": S3.TIER, "estimator": S3.ESTIMATOR, "n_tokens": len(toks), "gates": gates, "identity": ident,
           "BASE_vs_banked_consistency": cons, "pairs_vs_BASE_pod": pairs_base, "pairs_vs_banked_u400": pairs_banked, "lever_view": lever,
           "deployable_safe_vs_BASE_pod": safe, "deployable_safe_vs_banked_u400_INFO": safe_banked, "row": row, "row_vs_banked_INFO": row_banked,
           "levels_x100": {p: {k: float(np.mean([tabs[p][t][k] for t in toks])) for k in ("PDMS",) + SUBS} for p in POINTS},
           "banked_u400_level_x100": {k: float(np.mean([banked_tab[t][k] for t in toks])) for k in ("PDMS",) + SUBS},
           "families_adverse_vs_BASE": fam_vs, "strategic": "PAUSED (PI 2026-09-19, item 25); UNAVAILABLE in NAVSIM"}
    json.dump(res, open(f"{out}/freepoint_result.json", "w"), indent=1, default=str)
    md = ["## Levels (mean x100, all 12,146 tokens)", "", level_table({**{p: tabs[p] for p in POINTS}, "banked u400": banked_tab}, toks), "",
          pair_table("Each point minus BASE (BASE as picked in this run; paired, x100)", pairs_base), "",
          pair_table("Sensitivity: each point minus the BANKED u400 table (INFO, not gating)", pairs_banked), "",
          pair_table("Lever view (paired, x100)", lever), "",
          "## DEPLOYABLE-SAFE (vs BASE_pod)", "", *[f"* {p}: **{safe[p]['safe']}** {safe[p]['conditions']}" for p in safe], "", f"## Row: **{row}**  (sensitivity vs banked u400: {row_banked})", "",
          "## Families", "", fam_summary(fams)]
    open(f"{out}/RESULT_TABLES.md", "w", encoding="utf-8").write("\n".join(md))
    print("\n".join(md[:30]))
    print(f"ZZREAD_OK row {row}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("selftest", "read"))
    ap.add_argument("--work", default=".")
    ap.add_argument("--out-dir", default=".")
    a = ap.parse_args()
    return {"selftest": cmd_selftest, "read": cmd_read}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
