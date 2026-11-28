"""POST-HOC, READ-ONLY: what SPEC A6 AS WRITTEN (item 7: "Unchanged: ... A2's low-support rule ...") reads
on a banked G0 artifact, versus what the A6 judge AS CODED reads.

⛔ This changes no verdict and no code. The banked `verdict_A6` is the gate of record; this tool only
measures how far the code's A6 departs from the registered A6 text. It does NOT call a modified judge: it
re-reads the banked per-term records (`verdict_A6.terms`: class, seed mean, PI, phi, flip interval) and
applies ONE change -- A2's low-support rule (`g0_refcv7.detection_support`, `A2_MIN_SUPPORT`, the gate's
own functions) to DETECTION terms -- then recomputes the reason list, the class medians and the
mutation detection with the judge's own per-class rules (`g0_refcv7.tol_ok`).

⭐ KNOWN-VALUE CONTROLS (the reconstruction must reproduce what the real judge wrote, else REFUSED):
  * with the low-support rule OFF, the reconstruction of `verdict_A6` must reproduce the banked A6 reason
    list and the banked M1 / M2 / M4 `n_terms_out` EXACTLY;
  * the same reconstruction applied to the banked `verdict_A5` terms (which DO carry `support_n`) must
    reproduce the banked A5 reasons and mutation counts exactly.
Only then is the item-7 reading reported.

    python g0_a6_item7_posthoc.py --g0 raw/step50400/g0.json \
        --metrics D:/refcv7_eval_kit/thor_reads/metrics_final_50400.jsonl --out raw/g0diag_step50400/a6_item7_posthoc.json
    (step 30,000: --g0 raw/g0a6supp_step30000/g0_A6_posthoc_step30000.json --a6-key verdict_A6_POSTHOC
     --g0-rows raw/step30000/g0.json)
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g0_refcv7 as G  # noqa: E402


def reconstruct(terms: dict, inrun: dict, mutations: dict, *, lowsupport: bool, a6: bool,
                carried_support: bool = False):
    """Re-derive reasons / medians / mutation detection from banked per-term records."""
    res = {}
    reasons = []
    for k, r0 in terms.items():
        r = dict(r0)
        c = r.get("cls", r.get("class"))
        r["cls"] = c
        if carried_support and c == "DETECTION_LOWSUPPORT":
            res[k] = r
            continue
        if lowsupport and c == "DETECTION" and not carried_support:
            sup = G.detection_support(k, inrun)
            if sup is not None:
                r["support_n"] = sup
                x, mean = float(r["inrun"]), float(r["mean"])
                if sup < G.A2_MIN_SUPPORT:
                    r.update(cls="DETECTION_LOWSUPPORT", verdict="REPORTED")
                    res[k] = r
                    continue
                t2 = max(0.02, 2.0 / sup)
                r["verdict"] = "OK" if abs(mean - x) <= t2 else "OUT"
                r["abs_dev01"] = abs(mean - x)
        res[k] = r
        if r.get("verdict") in ("OUT", "MISSING"):
            if c == "THRESHOLD_TARGET":
                reasons.append(f"{k} [THRESHOLD_TARGET] in-run {r['inrun']} outside the A6 interval "
                               f"({r.get('tol')})")
            elif r.get("verdict") == "MISSING":
                reasons.append(f"{k}: not produced by the reproduction")
            elif str(c).endswith("/UNDEFINED"):
                reasons.append(f"{k} [UNDEFINED rule] in-run {r['inrun']} vs seeds {r['seed_values'][:3]}")
            else:
                reasons.append(f"{k} [{c}] in-run {float(r['inrun'])} vs seeds mean {r['mean']:.6g} "
                               f"(sd {r['sd']:.3g})")
    med = {}
    dt = [v["abs_dev01"] for v in res.values() if v.get("cls") == "DETECTION" and "abs_dev01" in v]
    med["DETECTION_abs01"] = statistics.median(dt) if dt else None
    med["n_DETECTION_gating"] = len(dt)
    det = {}
    for m, info in mutations.items():
        row = info.get("row") or {}
        outs = []
        for k, r in res.items():
            c = r.get("cls", "")
            if c in ("COUNT",) or r.get("verdict") != "OK" or str(c).endswith("/UNDEFINED"):
                continue
            y, x = row.get(k), r["inrun"]
            if G._isnull(y) or G._isnull(x):
                continue
            if c == "STOCHASTIC":
                bad = not (r["pi_lo"] <= float(y) <= r["pi_hi"])
            elif c == "THRESHOLD_TARGET":
                bad = not (r["a6_lo"] <= float(y) <= r["a6_hi"])
            elif r.get("support_n") is not None:
                bad = abs(float(y) - float(x)) > max(0.02, 2.0 / float(r["support_n"]))
            else:
                bad = not G.tol_ok(c, k, float(x), float(y))[0]
                if bad and a6 and c == "SMOOTH" and r.get("a6_phi") is not None:
                    bad = abs(float(y) - float(x)) > G.A6_K * float(r["a6_phi"])
            if bad:
                outs.append({"term": k, "class": c, "inrun": x, "mutated": y})
        det[m] = {"detected": bool(outs), "n_terms_out": len(outs), "terms_out": outs}
    counts = {}
    for v in res.values():
        counts[v.get("cls")] = counts.get(v.get("cls"), 0) + 1
    return {"reasons_terms": reasons, "medians": med, "mutation_detection": det, "by_class_counts": counts,
            "terms": res}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", required=True, help="artifact holding the A6 verdict (and, by default, mutations)")
    ap.add_argument("--a6-key", default="verdict_A6")
    ap.add_argument("--g0-rows", default=None, help="artifact holding mutations + verdict_A5 (default --g0)")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = json.load(open(a.g0, encoding="utf-8"))
    rows = json.load(open(a.g0_rows, encoding="utf-8")) if a.g0_rows else g
    v6 = g[a.a6_key]
    v5 = rows.get("verdict_A5") or rows.get("verdict")
    step = int(rows["step"])
    ev = [json.loads(ln) for ln in open(a.metrics, encoding="utf-8") if ln.strip()]
    ev = [r for r in ev if r.get("step") == step and "eval_loss" in r]
    assert len(ev) == 1
    inrun = ev[0]
    muts = rows.get("mutations") or {}
    out = {"tool": "g0_a6_item7_posthoc.py", "POST_HOC": True, "step": step,
           "note": "READ-ONLY reconstruction; the banked verdict_A6 is the gate of record and is unchanged"}
    # ---- controls ------------------------------------------------------------------------------
    term_reasons_6 = [x for x in v6["reasons"] if x.split(" ")[0].startswith("eval_")]
    c6 = reconstruct(v6["terms"], inrun, muts, lowsupport=False, a6=True)
    ctl6 = {"reasons_equal": c6["reasons_terms"] == term_reasons_6,
            "mutations_equal": {m: (c6["mutation_detection"][m]["n_terms_out"],
                                    v6["mutation_detection"][m]["n_terms_out"]) for m in v6["mutation_detection"]}}
    ctl6["ok"] = ctl6["reasons_equal"] and all(a_ == b_ for a_, b_ in ctl6["mutations_equal"].values())
    out["control_A6_as_coded"] = ctl6
    if v5 is not None and v5.get("amendment") == "A5":
        term_reasons_5 = [x for x in v5["reasons"] if x.split(" ")[0].startswith("eval_")]
        c5 = reconstruct(v5["terms"], inrun, muts, lowsupport=True, a6=False, carried_support=True)
        ctl5 = {"reasons_equal": c5["reasons_terms"] == term_reasons_5,
                "mutations_equal": {m: (c5["mutation_detection"][m]["n_terms_out"],
                                        v5["mutation_detection"][m]["n_terms_out"]) for m in v5["mutation_detection"]}}
        ctl5["ok"] = ctl5["reasons_equal"] and all(a_ == b_ for a_, b_ in ctl5["mutations_equal"].values())
        out["control_A5"] = ctl5
    else:
        out["control_A5"] = {"ok": None, "why": "no banked A5 verdict in --g0-rows"}
    other = [x for x in v6["reasons"] if not x.split(" ")[0].startswith("eval_")]
    out["banked_A6_nonterm_reasons"] = other
    if not ctl6["ok"] or out["control_A5"].get("ok") is False:
        out["REFUSED"] = "a known-value control failed: the reconstruction is not the judge"
    else:
        r7 = reconstruct(v6["terms"], inrun, muts, lowsupport=True, a6=True)
        reasons = r7["reasons_terms"] + other
        if r7["medians"]["DETECTION_abs01"] is not None and r7["medians"]["DETECTION_abs01"] > 0.005:
            reasons.append("DETECTION median abs dev > 0.005")
        m1 = r7["mutation_detection"].get("m1")
        if m1 is None or not m1["detected"]:
            reasons.append("M1 not detected -> VOID")
        out["A6_as_written_POSTHOC"] = {
            "G0": "PASS" if not reasons else ("VOID" if m1 and not m1["detected"] else "FAIL"),
            "reasons": reasons, "by_class_counts": r7["by_class_counts"], "medians": r7["medians"],
            "mutation_detection": {m: {"detected": d["detected"], "n_terms_out": d["n_terms_out"],
                                       "terms_out": d["terms_out"]}
                                   for m, d in r7["mutation_detection"].items()},
            "lowsupport_reported": sorted(k for k, t in r7["terms"].items() if t.get("cls") == "DETECTION_LOWSUPPORT"
                                          and abs(float(t["inrun"]) - float(t["mean"])) > 0.02)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    pr = {k: out[k] for k in ("control_A6_as_coded", "control_A5") if k in out}
    if "A6_as_written_POSTHOC" in out:
        v = out["A6_as_written_POSTHOC"]
        pr["A6_as_written_POSTHOC"] = {"G0": v["G0"], "reasons": v["reasons"], "by_class_counts": v["by_class_counts"],
                                       "medians": v["medians"],
                                       "mutations": {m: (d["detected"], d["n_terms_out"],
                                                         [(t["term"], t["class"]) for t in d["terms_out"]][:12])
                                                     for m, d in v["mutation_detection"].items()}}
    else:
        pr["REFUSED"] = out.get("REFUSED")
    print(json.dumps(pr, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
