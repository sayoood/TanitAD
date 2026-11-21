#!/usr/bin/env python3
"""P5-S read-out (eval/PREREG_P5_TRACKER_AWARE_TARGETS.md §9, amendment P5-S). Reads the raw JSON/NPZ only, so no
number in eval/RESULT_P5S.md is hand-transcribed.

  C:/Users/Admin/venvs/tanitad/Scripts/python.exe eval/p5s_compare.py \
      --landed raw/2026-10-11-p5-target-labels --dir raw/2026-10-11-p5s-smoothed --out <p5s_compare.json> > tables.md

Gates: C8 (linear reproduces the landed P5 JSON + NPZ exactly, "seconds" stripped), C2' (each spline arm's knot
deviation <= 1e-9, and the cubic mutate-knot control must FAIL), and every §4 gate of every arm. The committed PASS bar
per smoothed arm: (a) Δ sim8 (Tc20s - T) upper CI bound < 0, (b) no §6 adverse guard, (c) Tc20s median max |acc| AND
median max |heading rate| <= 2x t20's. Everything else printed here is DESCRIPTIVE.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("p5tl", os.path.join(HERE, "p5_target_labels.py"))
P5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P5)                       # module level imports numpy only; same estimator as the arms
ARMS = ("cubic", "pchip")
KNOT_TOL = 1e-9


def diff_paths(a, b, path="", out=None, worst=None):
    """Exact structural comparison. Returns (mismatch paths, max |float diff| over numeric mismatches)."""
    out = [] if out is None else out
    worst = [0.0] if worst is None else worst
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k} (missing in {'landed' if k not in a else 'linear'})")
            else:
                diff_paths(a[k], b[k], f"{path}.{k}", out, worst)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path} (len {len(a)} vs {len(b)})")
        for i, (x, y) in enumerate(zip(a, b)):
            diff_paths(x, y, f"{path}[{i}]", out, worst)
    elif a != b:
        out.append(path)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            worst[0] = max(worst[0], abs(float(a) - float(b)))
    return out, worst[0]


def strip(d):
    return {k: v for k, v in d.items() if k != "seconds"}


def med(v):
    return float(np.median(np.asarray(v, np.float64)))


def dual(recs):
    s = np.array([bool(x["sim8"]) for x in recs])
    i = np.array([bool(x["interp8"]) for x in recs])
    return {"n": int(len(recs)), "sim8_fail": int(s.sum()), "interp8_fail": int(i.sum()),
            "fail_both": int((s & i).sum()), "pass_both": int((~s & ~i).sum()),
            "pass_both_share": float((~s & ~i).mean())}


def arm_read(d):
    ii = d["arm_ii"]
    lt = np.array([r["learn"]["t20"] for r in d["rows"]])
    lc = np.array([r["learn"]["Tc20"] for r in d["rows"]])
    acc_ratio = med(lc[:, 0]) / med(lt[:, 0])
    hr_ratio = med(lc[:, 1]) / med(lt[:, 1])
    gates_ok = all(d["gates"].values())
    letters = {"a_sim8_hi_lt_0": ii["cis"]["sim8"]["hi"] < 0,
               "b_no_adverse_guard": not ii["adverse_guards"],
               "c_learnability_le_2x": acc_ratio <= 2.0 and hr_ratio <= 2.0}
    return {"gates_ok": gates_ok, "failed_gates": [g for g, v in d["gates"].items() if not v],
            "knot_maxdev": d["controls_raw"].get("knot_maxdev"), "verdict_arm_ii": ii["verdict"],
            "adverse_guards": ii["adverse_guards"], "learnability_flag": ii["learnability_flag"],
            "sim8_ci": ii["cis"]["sim8"], "acc_median": {"t20": med(lt[:, 0]), "Tc20s": med(lc[:, 0]),
                                                       "ratio": acc_ratio},
            "heading_rate_median": {"t20": med(lt[:, 1]), "Tc20s": med(lc[:, 1]), "ratio": hr_ratio},
            "letters": letters, "PASS": gates_ok and all(letters.values()),
            "failed_letters": [k[0] for k, v in letters.items() if not v],
            "wrong_sign_Tw_verdict": d["wrong_sign_Tw"]["verdict"]}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--landed", required=True, help="dir holding the landed P5 p5_labels.json + p5_exec.npz")
    ap.add_argument("--dir", required=True, help="dir holding p5s_{linear,cubic,pchip}.{json,npz} + mutate_knot_cubic.json")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    def lj(p):
        return json.load(open(p, encoding="utf-8"))

    landed = lj(os.path.join(a.landed, "p5_labels.json"))
    lin = lj(os.path.join(a.dir, "p5s_linear.json"))
    arms = {m: lj(os.path.join(a.dir, f"p5s_{m}.json")) for m in ARMS}
    mut = lj(os.path.join(a.dir, "mutate_knot_cubic.json"))

    # ---- C8: reproduction ----
    paths, worst = diff_paths(strip(landed), strip(lin))
    zl = np.load(os.path.join(a.landed, "p5_exec.npz"))
    zn = np.load(os.path.join(a.dir, "p5s_linear.npz"))
    npz_bad = sorted(set(zl.files) ^ set(zn.files)) + [k for k in sorted(set(zl.files) & set(zn.files))
                                                       if not np.array_equal(zl[k], zn[k])]
    c8 = {"json_mismatches": len(paths), "json_first": paths[:20], "json_max_abs_float_diff": worst,
          "npz_keys": len(zl.files), "npz_mismatches": npz_bad, "PASS": not paths and not npz_bad}

    # ---- C2' ----
    c2p = {"mutate_knot_cubic_gates_C2": mut["gates"].get("C2_knot_exact"),
           "mutate_knot_cubic_maxdev": mut["controls_raw"].get("knot_maxdev"),
           "mutate_must_fail_and_did": mut["gates"].get("C2_knot_exact") is False,
           "arm_maxdev": {m: arms[m]["controls_raw"].get("knot_maxdev") for m in ARMS}}
    c2p["PASS"] = c2p["mutate_must_fail_and_did"] and all(
        v is not None and v <= KNOT_TOL for v in c2p["arm_maxdev"].values())

    reads = {m: arm_read(arms[m]) for m in ARMS}
    reads["linear (= landed Tc20, reference)"] = arm_read(lin)
    amend_ok = c8["PASS"] and c2p["PASS"]
    if not amend_ok:
        outcome = "GATE_FAIL: no P5-S verdict is read (§9 C8/C2')"
    elif reads["cubic"]["PASS"]:
        outcome = "P5-S PASSED (primary, cubic)"
    elif reads["pchip"]["PASS"]:
        outcome = "P5-S PASSED on the secondary (pchip); the primary (cubic) FAILED " + \
                  ",".join(reads["cubic"]["failed_letters"] or ["gates"])
    else:
        outcome = "P5-S FAILED: cubic " + ",".join(reads["cubic"]["failed_letters"] or ["gates"]) + \
                  "; pchip " + ",".join(reads["pchip"]["failed_letters"] or ["gates"]) + " -> next lever P5-R"

    # ---- descriptive ----
    key = [(r["log"], r["token"]) for r in lin["rows"]]
    desc = {}
    for m in ARMS:
        rows_s = arms[m]["rows"]
        assert [(r["log"], r["token"]) for r in rows_s] == key, f"{m}: row order differs from linear"
        W, li = P5.boot_weights([k[0] for k in key])
        col = lambda rows, ref, k: np.array([float(r[ref][k]) for r in rows])          # noqa: E731
        cis = {f"Tc20s-Tc20 {k}": P5.paired_ci(col(lin["rows"], "Tc20", k), col(rows_s, "Tc20", k), W, li, True)
               for k in ("sim8", "interp8")}
        cis["Tc20s-T interp8"] = P5.paired_ci(col(rows_s, "T", "interp8"), col(rows_s, "Tc20", "interp8"), W, li, True)
        sel = [bool(r["T"]["sim8"]) for r in rows_s]
        tsel = [r["Tc20"] if f else r["T"] for r, f in zip(rows_s, sel)]
        ls = np.array([r["learn"]["t20"] for r, f in zip(rows_s, sel) if f])
        lcs = np.array([r["learn"]["Tc20"] for r, f in zip(rows_s, sel) if f])
        desc[m] = {"paired_cis": cis,
                   "dual": {"T": dual([r["T"] for r in rows_s]), "Tc20": dual([r["Tc20"] for r in lin["rows"]]),
                            "Tc20s": dual([r["Tc20"] for r in rows_s]), "Tsel-s (DERIVED)": dual(tsel)},
                   "tsel_learn_selected": {"n": int(sum(sel)), "acc_t20": med(ls[:, 0]), "acc_Tc20s": med(lcs[:, 0]),
                                           "acc_ratio": med(lcs[:, 0]) / med(ls[:, 0]), "hr_t20": med(ls[:, 1]),
                                           "hr_Tc20s": med(lcs[:, 1]), "hr_ratio": med(lcs[:, 1]) / med(ls[:, 1])},
                   "learnability": arms[m]["learnability"],
                   "per_ref_sim8": {k: v["sim8_rate"] for k, v in arms[m]["per_ref"].items()},
                   "per_ref_interp8": {k: v["interp8_rate"] for k, v in arms[m]["per_ref"].items()},
                   "arm_ii_cis": arms[m]["arm_ii"]["cis"]}

    res = {"prereg": "eval/PREREG_P5_TRACKER_AWARE_TARGETS.md §9 (P5-S)", "C8": c8, "C2prime": c2p,
           "amendment_gates_ok": amend_ok, "arms": reads, "outcome": outcome, "descriptive": desc,
           "n": lin["n"], "n_logs": lin["n_logs"]}
    json.dump(res, open(a.out, "w", encoding="utf-8"), indent=1)

    # ---- tables ----
    pp = lambda c: f"{100 * c['delta']:+.2f} pp [{100 * c['lo']:+.2f}, {100 * c['hi']:+.2f}]"   # noqa: E731
    L = [f"n = {lin['n']} rows, {lin['n_logs']} logs.", "",
         "**Amendment gates**", "", "| gate | result | detail |", "|---|---|---|",
         f"| C8 linear reproduces landed P5 (JSON, `seconds` stripped) | {'PASS' if not paths else '**FAIL**'} | "
         f"{len(paths)} mismatching fields; max float diff {worst:.3g} |",
         f"| C8 linear reproduces landed P5 (NPZ, {c8['npz_keys']} arrays) | {'PASS' if not npz_bad else '**FAIL**'} | "
         f"{'all array_equal' if not npz_bad else ', '.join(npz_bad)} |",
         f"| C2′ cubic knot max dev ≤ 1e-9 | {'PASS' if (c2p['arm_maxdev']['cubic'] or 1) <= KNOT_TOL else '**FAIL**'} | "
         f"{c2p['arm_maxdev']['cubic']:.3g} |",
         f"| C2′ pchip knot max dev ≤ 1e-9 | {'PASS' if (c2p['arm_maxdev']['pchip'] or 1) <= KNOT_TOL else '**FAIL**'} | "
         f"{c2p['arm_maxdev']['pchip']:.3g} |",
         f"| C2′ deliberate regression (`--mutate knot --interp cubic`) must FAIL | "
         f"{'PASS (it failed)' if c2p['mutate_must_fail_and_did'] else '**FAIL** (it passed)'} | "
         f"max dev {c2p['mutate_knot_cubic_maxdev']:.3g} m |", "",
         "**PASS bar per arm (committed)**", "",
         "| arm | §4 gates | arm (ii) verdict | (a) Δ sim8 Tc20s − T | (b) adverse guards | (c) acc median ratio | "
         "(c) heading-rate median ratio | C3 Tw | PASS |", "|---|---|---|---|---|---|---|---|---|"]
    for m, r in reads.items():
        L.append(f"| {m} | {'all PASS' if r['gates_ok'] else 'FAIL: ' + ','.join(r['failed_gates'])} | "
                 f"{r['verdict_arm_ii']} | {pp(r['sim8_ci'])} | {', '.join(r['adverse_guards']) or 'none'} | "
                 f"{r['acc_median']['Tc20s']:.3f} / {r['acc_median']['t20']:.3f} = {r['acc_median']['ratio']:.2f}× | "
                 f"{r['heading_rate_median']['Tc20s']:.3f} / {r['heading_rate_median']['t20']:.3f} = "
                 f"{r['heading_rate_median']['ratio']:.2f}× | {r['wrong_sign_Tw_verdict']} | "
                 f"{'**PASS**' if r['PASS'] else 'FAIL ' + ','.join(r['failed_letters'] or ['gates'])} |")
    L += ["", f"**Outcome: {outcome}**", ""]
    for m in ARMS:
        D = desc[m]
        L += [f"**Descriptive, {m}**", "", "| paired contrast | Δ [95 % CI] |", "|---|---|"]
        L += [f"| {k} | {pp(c)} |" for k, c in D["paired_cis"].items()]
        L += ["", "| ref | n | sim8 fail | interp8 fail | fail both | pass both |", "|---|---|---|---|---|---|"]
        for k, v in D["dual"].items():
            L.append(f"| {k} | {v['n']} | {v['sim8_fail']} | {v['interp8_fail']} | {v['fail_both']} | "
                     f"{v['pass_both']} ({100 * v['pass_both_share']:.2f} %) |")
        t = D["tsel_learn_selected"]
        L += ["", f"Tsel-s learnability on its {t['n']} selected rows (median per-row max): |acc| t20 {t['acc_t20']:.3f} "
              f"vs Tc20s {t['acc_Tc20s']:.3f} m/s² ({t['acc_ratio']:.2f}×); |heading rate| t20 {t['hr_t20']:.3f} vs "
              f"Tc20s {t['hr_Tc20s']:.3f} °/s ({t['hr_ratio']:.2f}×).", "",
              "| label | max acc median | max acc p95 | max heading-rate median °/s | p95 °/s |", "|---|---|---|---|---|"]
        for lab, v in D["learnability"].items():
            L.append(f"| {lab} | {v['max_acc_mps2']['median']:.3f} | {v['max_acc_mps2']['p95']:.3f} | "
                     f"{v['max_heading_rate_dps']['median']:.3f} | {v['max_heading_rate_dps']['p95']:.3f} |")
        L += ["", "| arm (ii) guard, Tc20s − T | Δ [95 % CI] |", "|---|---|"]
        for k, c in D["arm_ii_cis"].items():
            v = pp(c) if c["kind"] == "ratio" else f"{c['delta']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]"
            L.append(f"| {k} | {v} |")
        L.append("")
    print("\n".join(L))
    return 0 if amend_ok else 1


if __name__ == "__main__":
    sys.exit(main())
