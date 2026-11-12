#!/usr/bin/env python3
"""TF-N2 probe read (navtrain held-out only): EP mechanism tables + paired lever contrasts.

Estimator: paired log-cluster bootstrap, 10,000 draws, seed 20260927, 95 % percentile, ratio mean over the tokens of
the drawn logs (TF-2's estimator). It answers "another draw of LOGS"; labels, tracker and scorer are deterministic.
  python tfn2_analyze.py --rows <probe_s*.jsonl ...> --out <json>
"""
import argparse
import collections
import glob
import json

import numpy as np

B, SEED = 10000, 20260927
SUB = ("nc", "dac", "ep", "ttc", "c", "pdms")


def load(pats):
    rows, seen = [], set()
    for p in pats:
        for fp in sorted(glob.glob(p)):
            for ln in open(fp, encoding="utf-8"):
                try:
                    r = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if "error" in r or r["token"] in seen:
                    continue
                seen.add(r["token"])
                rows.append(r)
    return rows


def boot(vals, logs, paired=None):
    """log-cluster bootstrap of the token mean of vals (or vals - paired). -> (point, lo, hi)"""
    v = np.asarray(vals, float) - (0.0 if paired is None else np.asarray(paired, float))
    ul = sorted(set(logs))
    idx = {l: np.array([i for i, x in enumerate(logs) if x == l]) for l in ul}
    s = np.array([v[idx[l]].sum() for l in ul])
    n = np.array([len(idx[l]) for l in ul])
    rng = np.random.default_rng(SEED)
    d = rng.integers(0, len(ul), (B, len(ul)))
    m = s[d].sum(1) / n[d].sum(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def g(r, arm, k, fr="tracked"):
    return r["arms"][arm][fr][k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--extra", nargs="*", default=[], help="RW105 rows (tfn2_probe --extra); the read is then the intersection")
    ap.add_argument("--sel-rows-out", default="", help="write W-TRANSP-format rows whose arm SEL = the N2 candidate's choice")
    a = ap.parse_args()
    R = load(a.rows)
    if a.extra:
        X = {r["token"]: r for r in load(a.extra)}
        R = [r for r in R if r["token"] in X]
        for r in R:
            r["arms"]["RW105"] = X[r["token"]]["arms"]["RW105"]
    logs = [r["log"] for r in R]
    arms = [k for k in R[0]["arms"]]
    out = {"n_tokens": len(R), "n_logs": len(set(logs)), "by_src": dict(collections.Counter(r["src"] for r in R)),
           "estimator": "paired log-cluster bootstrap, B=10000, seed 20260927, 95% percentile, ratio mean",
           "cderiv_max_m": max((r["cderiv_m"] or 0.0) for r in R), "arms": {}, "paired_vs_C": {}}
    for arm in arms:
        d = {k: 100 * np.mean([g(r, arm, k) for r in R]) for k in SUB}
        mult = np.array([g(r, arm, "mult") for r in R])
        ep = np.array([g(r, arm, "ep") for r in R])
        d["zero_mult_pct"] = 100 * float((mult == 0).mean())
        d["ep_alive"] = 100 * float(ep[mult > 0].mean())
        d["pdms_ci"] = [100 * x for x in boot([g(r, arm, "pdms") for r in R], logs)]
        d["transparent_pdms"] = 100 * float(np.mean([g(r, arm, "pdms", "transparent") for r in R]))
        d["transparent_ep"] = 100 * float(np.mean([g(r, arm, "ep", "transparent") for r in R]))
        out["arms"][arm] = d
    for arm in arms:
        if arm == "C":
            continue
        out["paired_vs_C"][arm] = {k: [100 * x for x in boot([g(r, arm, k) for r in R], logs, [g(r, "C", k) for r in R])]
                                   for k in SUB}
    # ---- EP mechanism on the baseline C (Tc20s tracked) --------------------------------------------------------------
    N = len(R)
    v0 = np.array([r["shared"]["v0"] for r in R])
    ep = np.array([g(r, "C", "ep") for r in R]); mult = np.array([g(r, "C", "mult") for r in R])
    raw = np.array([g(r, "C", "raw") for r in R]); raw0 = np.array([g(r, "C", "raw0") for r in R])
    mult0 = np.array([g(r, "C", "mult0") for r in R])
    epH = np.array([g(r, "human", "ep") for r in R])
    lead = np.array([r["shared"]["lead"]["state"] == "LEAD" for r in R])
    gap0 = np.array([r["shared"]["lead"].get("gap0_m") or np.nan for r in R])
    red = np.array([r["mech"]["red_on_route"] > 0 for r in R])
    tp, pp = np.array([r["mech"]["t_path4_m"] for r in R]), np.array([r["mech"]["pdm_path4_m"] for r in R])
    tv4, pv4 = np.array([r["mech"]["t_v4"] for r in R]), np.array([r["mech"]["pdm_v4"] for r in R])
    ta, pa = np.array([r["mech"]["t_amax"] for r in R]), np.array([r["mech"]["pdm_amax"] for r in R])
    loss = 1.0 - ep
    mech = {"total_ep_loss_pts": 100 * loss.mean(),
            "loss_from_zero_mult_pts": 100 * loss[mult == 0].sum() / N,
            "loss_alive_pts": 100 * loss[mult > 0].sum() / N,
            "share_alive_ep_lt1": float(((ep < 0.9999) & (mult > 0)).mean()),
            "share_norm_below_5m(ep=1 by rule)": float((np.maximum(raw0 * mult0, raw * mult) <= 5.0).mean()),
            "share_pdm_failed(mult0=0)": float((mult0 == 0).mean())}
    tabs = {}

    def tab(name, masks):
        t = []
        for lab, m in masks:
            al = m & (mult > 0)
            if m.sum() == 0:
                continue
            t.append({"bin": lab, "n": int(m.sum()), "ep_C": 100 * ep[m].mean(), "ep_C_alive": 100 * ep[al].mean() if al.any() else None,
                      "ep_human": 100 * epH[m].mean(), "loss_pts_of_total": 100 * loss[m].sum() / N,
                      "teacher_path4_m": float(tp[m].mean()), "pdm_path4_m": float(pp[m].mean()),
                      "teacher_v4": float(tv4[m].mean()), "pdm_v4": float(pv4[m].mean()),
                      "teacher_amax": float(np.median(ta[m])), "pdm_amax": float(np.median(pa[m]))})
        tabs[name] = t
    bands = [(0, 0.5), (0.5, 3), (3, 6), (6, 9), (9, 12), (12, 40)]
    tab("v0_band", [(f"[{lo},{hi})", (v0 >= lo) & (v0 < hi)) for lo, hi in bands])
    tab("lead", [("LEAD", lead), ("NO_LEAD", ~lead)])
    tab("lead_gap0", [("gap<10m", lead & (gap0 < 10)), ("10-30m", lead & (gap0 >= 10) & (gap0 < 30)),
                      ("30-80m", lead & (gap0 >= 30)), ("NO_LEAD", ~lead)])
    tab("red_light_on_route", [("RED", red), ("not RED", ~red)])
    start = v0 < 0.5
    tab("stop_start", [("standstill v0<0.5", start), ("moving", ~start),
                       ("standstill & NO_LEAD & not RED", start & ~lead & ~red),
                       ("standstill & (LEAD or RED)", start & (lead | red))])
    tab("speed_lead", [(f"v0<6 & {'LEAD' if L else 'NO_LEAD'}", (v0 < 6) & (lead == L)) for L in (True, False)] +
        [(f"v0>=6 & {'LEAD' if L else 'NO_LEAD'}", (v0 >= 6) & (lead == L)) for L in (True, False)])
    # ---- does the conversion slow the plan?  raw progress, paired, all tokens alive in the three --------------------
    rT_tr = np.array([g(r, "T", "raw") for r in R]); rT_tp = np.array([g(r, "T", "raw", "transparent") for r in R])
    rC_tp = np.array([g(r, "C", "raw", "transparent") for r in R])
    conv = {"C_tracked_minus_T_transparent_m": boot(raw, logs, rT_tp),
            "T_tracked_minus_T_transparent_m": boot(rT_tr, logs, rT_tp),
            "C_tracked_over_T_transparent_median": float(np.median(raw[rT_tp > 1] / rT_tp[rT_tp > 1])),
            "T_transparent_raw_over_teacher_path_median": float(np.median(rT_tp[tp > 1] / tp[tp > 1])),
            "teacher_path_over_pdm_path_median": float(np.median(tp[pp > 1] / pp[pp > 1])),
            "teacher_v4_minus_pdm_v4_median": float(np.median(tv4 - pv4)),
            "teacher_amax_median": float(np.median(ta)), "pdm_amax_median": float(np.median(pa))}
    # ---- admissible post-hoc selection rules (EXPLORATORY; selected on these rows) ----------------------------------
    def pick(rule):
        return [rule(r) for r in R]
    rules = {
        "CERT_P(W110>W105>C by tracked DAC)": lambda r: next((x for x in ("W110", "W105") if g(r, x, "dac") == 1.0), "C"),
        "CERT_P(W105>C by tracked DAC)": lambda r: "W105" if g(r, "W105", "dac") == 1.0 else "C",
        "W110 if v0<6 else C": lambda r: "W110" if r["shared"]["v0"] < 6 else "C",
        "CERT_P v0<6 (W110>W105>C by DAC) else C": lambda r: (next((x for x in ("W110", "W105") if g(r, x, "dac") == 1.0), "C")
                                                          if r["shared"]["v0"] < 6 else "C"),
        "CERT (C>T by DAC, N1 form)": lambda r: "C" if g(r, "C", "dac") == 1.0 else ("T" if g(r, "T", "dac") == 1.0 else "C"),
        "CERT_RP (R>C>T by DAC)": lambda r: next((x for x in ("R", "C", "T") if g(r, x, "dac") == 1.0), "R"),
        "CERT_P+R (W110>W105>R>C>T by DAC)": lambda r: next((x for x in ("W110", "W105", "R", "C", "T") if g(r, x, "dac") == 1.0), "C"),
    }
    if "RW105" in R[0]["arms"]:
        rules["N2 candidate: CERT over RW105>R>C>T by tracked DAC"] = lambda r: next(
            (x for x in ("RW105", "R", "C", "T") if g(r, x, "dac") == 1.0), "RW105")
        rules["RW105 alone"] = lambda r: "RW105"
    sel = {}
    for nm, rule in rules.items():
        ch = pick(rule)
        d = {k: [100 * x for x in boot([g(r, c, k) for r, c in zip(R, ch)], logs, [g(r, "C", k) for r in R])] for k in SUB}
        d["level"] = {k: 100 * float(np.mean([g(r, c, k) for r, c in zip(R, ch)])) for k in SUB}
        d["choice"] = dict(collections.Counter(ch))
        sel[nm] = d
    if a.sel_rows_out and "RW105" in R[0]["arms"]:
        rule = rules["N2 candidate: CERT over RW105>R>C>T by tracked DAC"]
        with open(a.sel_rows_out, "w", encoding="utf-8") as fo:
            for r in R:
                q = {"log": r["log"], "token": r["token"], "shared": r["shared"], "arms": {}}
                for nm in ("C", "R", "W105", "RW105", "human"):
                    q["arms"][nm] = r["arms"][nm]
                q["arms"]["SEL"] = r["arms"][rule(r)]
                fo.write(json.dumps(q) + "\n")
    orc = [max(arms[:-1] if False else [x for x in arms if x not in ("human", "Wreg", "RW105")], key=lambda x: g(r, x, "pdms")) for r in R]
    sel["INADMISSIBLE ceiling: argmax harness PDMS over {T,C,R,W105,W110}"] = {
        "level": {k: 100 * float(np.mean([g(r, c, k) for r, c in zip(R, orc)])) for k in SUB},
        "choice": dict(collections.Counter(orc))}
    out.update({"ep_mechanism": mech, "tables": tabs, "conversion": conv, "selection_rules_EXPLORATORY": sel})
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps({"n": out["n_tokens"], "logs": out["n_logs"],
                      "arms": {k: {x: round(v[x], 2) for x in ("pdms", "dac", "ep", "ttc", "nc", "c", "ep_alive")} for k, v in out["arms"].items()}}, indent=0))
    print(json.dumps({k: {x: [round(y, 2) for y in v[x]] for x in ("pdms", "ep", "dac", "ttc", "nc")} for k, v in out["paired_vs_C"].items()}, indent=0))


if __name__ == "__main__":
    main()
