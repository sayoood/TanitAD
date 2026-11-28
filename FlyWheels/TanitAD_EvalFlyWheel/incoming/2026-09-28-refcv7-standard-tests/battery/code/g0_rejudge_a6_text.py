"""G0-A6 AS REGISTERED (TEXT), recomputed by the CORRECTED judge from a banked G0 artifact. Zero GPU.

Master Mind ruling 2026-10-04 (after the G0-50,400 diagnosis): the REGISTERED A6 TEXT is the record. A6 item 7
("Unchanged: ... A2's low-support rule ...") was hashed before any step-50,400 number existed; `g0_refcv7.judge`
did not implement it (`amend in ("A2", "A5")`); the fix (`A2_LOWSUPPORT_AMENDS`) applies the SAME criterion
correctly. Both records are carried:
  * "G0-A6 as registered (text): <verdict>, computed by the corrected judge after the coded verdict was read;
    the criterion is unchanged"
  * "G0-A6 as coded: FAIL (judge defect: the A2 low-support tuple omitted A6)"

Preconditions, asserted on CONTENT (else REFUSED, nothing written as a verdict):
  * the judge in use carries the fix (`"A6" in G.A2_LOWSUPPORT_AMENDS`);
  * the in-run row read from --metrics equals the one the banked G0 judged (every term);
  * with --supp (step 30,000): same checkpoint md5 + window permutation, and the supplement's seed 0 reproduces
    the banked seed 0 within rel 1e-5 on every key both carry;
  * CONTROLS -- the corrected judge must reproduce the banked as-registered, A2 and A5 verdicts EXACTLY
    (G0, reasons, M1/M2/M4 counts): the fix may change A6 only;
  * with --compare: the READ-ONLY reconstruction (`g0_a6_item7_posthoc.py`) must agree (G0, reasons, mutation
    terms). A disagreement is reported and the tool exits 3 -- it never picks one.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g0_refcv7 as G  # noqa: E402

RULING = ("Master Mind ruling 2026-10-04: the REGISTERED A6 text is the record (A6 item 7 keeps A2's low-support "
          "rule; hashed 2026-10-04T08:17:13+02:00, before any step-50,400 number). The coded judge omitted A6 from "
          "the A2 low-support tuple; the corrected judge applies the same criterion. No goalpost moved.")


def _mut(v):
    return {m: (d["detected"], d["n_terms_out"]) for m, d in (v.get("mutation_detection") or {}).items()}


def _same(a, b):
    return a["G0"] == b["G0"] and a["reasons"] == b["reasons"] and _mut(a) == _mut(b)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g0", required=True)
    ap.add_argument("--supp", default=None, help="A6 supplement (step 30,000: its a6 arm + cells)")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--compare", default=None, help="g0_a6_item7_posthoc.py output for the same checkpoint")
    ap.add_argument("--coded-json", default=None, help="where the coded A6 verdict lives if not in --g0")
    ap.add_argument("--coded-key", default="verdict_A6_POSTHOC")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"tool": "g0_rejudge_a6_text.py", "ruling": RULING, "g0": a.g0, "supp": a.supp,
           "written": time.strftime("%FT%T%z"), "judge_a2_lowsupport_amends": list(G.A2_LOWSUPPORT_AMENDS)}
    bad = []
    if "A6" not in G.A2_LOWSUPPORT_AMENDS:
        bad.append("the judge in use does not carry the A6-item-7 fix")
    g0 = json.load(open(a.g0, encoding="utf-8"))
    step = int(g0["step"])
    out.update(step=step, ckpt_md5=g0.get("ckpt_md5"))
    ev = [json.loads(ln) for ln in open(a.metrics, encoding="utf-8") if ln.strip()]
    ev = [r for r in ev if r.get("step") == step and "eval_loss" in r]
    if len(ev) != 1:
        bad.append(f"{len(ev)} in-run eval rows at step {step}")
        inrun = None
    else:
        inrun = ev[0]
        t_in = {k: v.get("inrun") for k, v in g0["verdict"]["terms"].items()}
        dif = [k for k, v in t_in.items() if inrun.get(k) != v]
        if dif:
            bad.append(f"in-run row differs from the judged one on {len(dif)} keys: {dif[:5]}")
    rec = copy.deepcopy(g0)
    if a.supp:
        sp = json.load(open(a.supp, encoding="utf-8"))
        if sp.get("ckpt_md5") != g0.get("ckpt_md5") or sp.get("perm_sha256") != g0.get("perm_sha256"):
            bad.append("supplement checkpoint / permutation differs")
        s0, g_s0 = sp["by_seed"]["0"]["row"], g0["by_seed"]["0"]["row"]
        diff = [k for k in g_s0 if k in s0 and k.startswith("eval_") and not (
            (G._isnull(s0[k]) and G._isnull(g_s0[k])) or s0[k] == g_s0[k]
            or abs(float(s0[k]) - float(g_s0[k])) <= 1e-5 * max(abs(float(g_s0[k])), 1e-9))]
        if diff:
            bad.append(f"supplement seed 0 differs from G0 seed 0 on {len(diff)} keys")
        rec["a6"] = sp.get("a6") or {}
    by_seed = {int(s): {"row": v["row"], "buffers_unchanged": v.get("buffers_unchanged", True)}
               for s, v in g0["by_seed"].items()}
    if bad:
        out["REFUSED"] = bad
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
        print(json.dumps({"REFUSED": bad}))
        return 2
    # ---- controls: the corrected judge changes A6 only ---------------------------------------------
    reg = {s: v for s, v in by_seed.items() if s in G.REGISTERED_SEEDS}
    ctl = {}
    banked = {"as_registered": g0.get("verdict_as_registered"), "A2": g0.get("verdict_A2"),
              "A5": g0.get("verdict_A5") or (g0.get("verdict") if (g0.get("verdict") or {}).get("amendment") == "A5"
                                             else None)}
    recomputed = {"as_registered": G.judge(inrun, reg, rec), "A2": G.judge(inrun, reg, rec, amend="A2"),
                  "A5": G.judge(inrun, by_seed, rec, amend="A5")}
    for k in banked:
        if banked[k] is None:
            ctl[k] = {"ok": None, "why": "not banked"}
            continue
        ctl[k] = {"ok": _same(recomputed[k], banked[k]), "banked": [banked[k]["G0"], _mut(banked[k])],
                  "recomputed": [recomputed[k]["G0"], _mut(recomputed[k])]}
    out["controls_other_verdicts_unchanged"] = ctl
    if any(c["ok"] is False for c in ctl.values()):
        out["REFUSED"] = ["the corrected judge changed a verdict other than A6"]
        json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
        print(json.dumps({"REFUSED": out["REFUSED"], "controls": ctl}, default=str))
        return 2
    v6 = G.judge(inrun, by_seed, rec, amend="A6")
    coded = g0.get("verdict_A6")
    if coded is None and a.coded_json:
        coded = json.load(open(a.coded_json, encoding="utf-8")).get(a.coded_key)
        out["coded_from"] = f"{a.coded_json}:{a.coded_key} (POST HOC for this checkpoint)"
    out["verdict_A6_text"] = v6
    out["record_lines"] = [
        f"G0-A6 as registered (text): {v6['G0']}, computed by the corrected judge after the coded verdict was "
        f"read; the criterion is unchanged",
        "G0-A6 as coded: " + ((coded or {}).get("G0") or "not banked in this artifact (POST-HOC re-judge: "
                                                          "see g0_A6_posthoc)")
        + " (judge defect: the A2 low-support tuple omitted A6)"]
    rc = 0
    if a.compare:
        cp = json.load(open(a.compare, encoding="utf-8"))["A6_as_written_POSTHOC"]
        t_fix = {m: sorted(t["term"] for t in d["first10"]) for m, d in v6["mutation_detection"].items()}
        t_rec = {m: sorted(t["term"] for t in d["terms_out"][:10]) for m, d in cp["mutation_detection"].items()}
        cmpr = {"G0": [v6["G0"], cp["G0"]], "reasons": [v6["reasons"], cp["reasons"]],
                "mutations": [_mut(v6), {m: (d["detected"], d["n_terms_out"]) for m, d in cp["mutation_detection"].items()}],
                "lowsupport_count": [v6["by_class_counts"].get("DETECTION_LOWSUPPORT"),
                                     cp["by_class_counts"].get("DETECTION_LOWSUPPORT")],
                "detection_median": [v6["medians"].get("DETECTION_abs01"), cp["medians"].get("DETECTION_abs01")]}
        cmpr["first10_terms_equal"] = all(set(t_fix[m]) <= set(t["term"] for t in cp["mutation_detection"][m]["terms_out"])
                                          for m in t_fix)
        cmpr["agree"] = (cmpr["G0"][0] == cmpr["G0"][1] and cmpr["reasons"][0] == cmpr["reasons"][1]
                         and cmpr["mutations"][0] == cmpr["mutations"][1]
                         and cmpr["lowsupport_count"][0] == cmpr["lowsupport_count"][1]
                         and abs((cmpr["detection_median"][0] or 0) - (cmpr["detection_median"][1] or 0)) < 1e-12
                         and cmpr["first10_terms_equal"])
        out["compare_with_readonly_reconstruction"] = cmpr
        if not cmpr["agree"]:
            rc = 3
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1, default=str)
    print(json.dumps({"verdict_A6_text": v6["G0"], "reasons": v6["reasons"], "mutations": _mut(v6),
                      "by_class_counts": v6["by_class_counts"], "medians": v6["medians"],
                      "controls": {k: c["ok"] for k, c in ctl.items()},
                      "compare_agree": (out.get("compare_with_readonly_reconstruction") or {}).get("agree"),
                      "record_lines": out["record_lines"]}, indent=1, default=str))
    return rc


if __name__ == "__main__":
    sys.exit(main())
