"""wua_card.py -- card ADD155: apply the frozen labels across RH_s0..s2 and write the World Understanding Card (json + md). No new statistic is computed here.

  python -I wua_card.py
"""
import json, os, sys
sys.dont_write_bytecode = True
import numpy as np

WORK = "C:/Users/Admin/wua_work"
OUT_MD = "F:/TanitAD-scratch/v8proof/wua/WORLD_UNDERSTANDING_CARD.md"
SEEDS = (0, 1, 2)
BAR = {"lead_gap": 0.10, "ag1_pos": 0.10}
REPS = ("L1", "L2", "L3", "TOK-MEAN", "TOK-GRID")


def J(p):
    return json.load(open(p)) if os.path.exists(p) else None


def lab_enc(cells, bar):
    if any(c is None or "delta" not in c for c in cells):
        return "n/a"
    lo = [c["ci"][0] for c in cells]; mean = float(np.mean([c["delta"] for c in cells]))
    if all(l > 0 for l in lo):
        return "ENCODED-BEYOND-EGO" if mean >= bar else "PARTIAL"
    return "NOT-ENCODED"


def a1_controls(r):
    c = r["controls"]; miss = []
    for k, v in c.items():
        if k.startswith("c1_shuffle") and "delta" in v and abs(v["delta"]) > 0.02:
            miss.append(k)
        if k.startswith("c3_noise") and "delta" in v and v["delta"] > 0.02:
            miss.append(k)
    if c.get("c2_planted|L1", {}).get("delta", 0) < 0.40:
        miss.append("c2_planted")
    if c.get("c4_ego_speed_r2", 0) < 0.95:
        miss.append("c4")
    return miss


def main():
    card = {"card": "ADD155", "A1": {}, "A1_mlp": {}, "A2": {}, "A3": {}, "A4": {}, "A5": None, "controls": {}}
    lin = {s: J(f"{WORK}/a1/RESULT_A1_s{s}.json") for s in SEEDS}
    mlp = {s: J(f"{WORK}/a1/RESULT_A1_s{s}_mlp.json") for s in SEEDS}
    img = {s: J(f"{WORK}/imag/RESULT_IMAG_s{s}.json") for s in SEEDS}
    inval = set()
    for s in SEEDS:
        if lin[s]:
            m = a1_controls(lin[s]); card["controls"][f"A1_s{s}"] = m or "PASS"
            if m:
                inval.add(s)
    if all(lin.values()):
        keys = sorted(set().union(*[r["cells"].keys() for r in lin.values()]))
        for k in keys:
            rep, tg = k.split("|")
            cells = [lin[s]["cells"].get(k) for s in SEEDS]
            bar = BAR.get(tg, 0.05)
            lb = "INVALID" if inval else lab_enc(cells, bar)
            perm = [J(f"{WORK}/a1/PERM_A1_s{s}_{tg}.json") for s in SEEDS]
            pinfo = None
            if inval and all(perm):                      # Amendment 3: two-part rule (card rule AND above all 19 permutation refits at every seed)
                base = lab_enc(cells, bar)
                ex = [pp["exceeds_all"][rep] for pp in perm]
                lb = (base if all(ex) else "NOT-ENCODED") + " (AM3)"
                pinfo = {"exceeds_all": ex, "perm_max": [round(pp["perm_max"][rep], 4) for pp in perm], "align": [pp["alignment_vs_RESULT_A1"][rep] for pp in perm]}
            elif inval:
                lb = "INVALID (uncontrolled)"
            card["A1"][k] = {"label": lb, "bar": bar, "am3": pinfo, "per_seed": [None if c is None else {x: c.get(x) for x in ("delta", "ci", "n", "n_clips", "r2_full", "r2_ego", "auc_full", "auc_ego", "bacc_full", "bacc_ego")} for c in cells]}
    if all(mlp.values()):
        for k in sorted(set().union(*[r["cells"].keys() for r in mlp.values()])):
            tg = k.split("|")[1]
            cells = [mlp[s]["cells"].get(k) for s in SEEDS]
            lb = lab_enc(cells, BAR.get(tg, 0.05))
            lin_lb = card["A1"].get(k, {}).get("label")
            if lb == "ENCODED-BEYOND-EGO" and lin_lb != "ENCODED-BEYOND-EGO":
                lb = "ENCODED-TEMPORAL-ONLY"
            lb = "INSTRUMENT-LIMITED (card rule: %s)" % lb          # Amendment 4: the MLP overfits (dR2 < 0 even on ego speed); no control in the card
            card["A1_mlp"][k] = {"label": lb, "per_seed": [None if c is None else {x: c.get(x) for x in ("delta", "ci", "n")} for c in cells]}
    tl = {s: J(f"{WORK}/a1/TLIN_A1_s{s}.json") for s in SEEDS}
    card["A1_tlin"] = {}
    if all(tl.values()):
        for k in sorted(tl[0]["cells"]):
            rep, tg = k.split("|")
            cells = [tl[s]["cells"][k] for s in SEEDS]
            base = lab_enc([c["observed"] for c in cells], BAR.get(tg, 0.05))
            lb = base if all(c["exceeds_all"] for c in cells) else "NOT-ENCODED"
            single = card["A1"].get(k, {}).get("label", "")
            if lb == "ENCODED-BEYOND-EGO" and not single.startswith("ENCODED-BEYOND-EGO"):
                lb = "ENCODED-TEMPORAL-ONLY"
            card["A1_tlin"][k] = {"label": lb + " (AM4)", "per_seed": [{"delta": c["observed"]["delta"], "ci": c["observed"]["ci"], "n": c["observed"]["n"], "perm_max": c["perm_max"]} for c in cells]}
    if all(img.values()):
        # A2
        diff = [img[s]["A2"].get("DIFF") for s in SEEDS]; same = [img[s]["A2"].get("SAME") for s in SEEDS]
        c5 = all(sm is None or "skipped" in sm or all(v["ci"][0] <= 0.5 <= v["ci"][1] for v in sm.values()) for sm in same)
        c6 = all(d is not None and "EGO" in d and d["EGO"]["auc"] <= 0.60 for d in diff)
        c7 = all(d is not None and abs(d["IMAG-L1"]["c7_swap"] - (1 - d["IMAG-L1"]["auc"])) < 1e-9 for d in diff)
        card["controls"]["A2"] = {"c5_same_contains_half": c5, "c6_ego_auc_le_0.60": c6, "c7_swap": c7, "n_pairs": [img[s]["A2"]["n_pairs"] for s in SEEDS]}
        for sc in ("IMAG-L1", "IMAG-L2", "ENC-L1", "ORACLE-L1", "EGO"):
            per = [d.get(sc) if d else None for d in diff]
            if not c6:
                lb = "MATCH-FAILED"
            elif sc == "EGO":
                lb = "(control)"
            elif all(p and p["ci"][0] > 0.5 and p.get("minus_ego_ci", [0])[0] > 0 for p in per):
                lb = "DISCRIMINATES"
            elif any(p and p["ci"][0] <= 0.5 <= p["ci"][1] for p in per):
                lb = "BLIND"
            else:
                lb = "WEAK"
            card["A2"][sc] = {"label": lb, "per_seed": per, "same_pairs": [sm.get(sc) if sm and "skipped" not in sm else None for sm in same]}
        # A3
        for k in sorted(img[0]["A3"]):
            per = [img[s]["A3"].get(k) for s in SEEDS]
            if any(p is None or "skill" not in p for p in per):
                lb = "n/a"
            elif np.mean([p["oracle_r2"] for p in per]) < 0.10:
                lb = "UNREADABLE"
            elif all(p["ci"][0] > 0 for p in per):
                lb = "IMAGINES"
            elif all(p["ci"][1] < 0 for p in per):
                lb = "WORSE-THAN-COPY"
            elif all(p["ci"][0] <= 0 <= p["ci"][1] for p in per):
                lb = "COPIES"
            else:
                lb = "SEEDS-DISAGREE"
            card["A3"][k] = {"label": lb, "per_seed": per}
        # A4
        for k in sorted(img[0]["A4"]):
            per = [img[s]["A4"].get(k) for s in SEEDS]
            if any(p is None or "slope" not in p for p in per):
                lb = "n/a"
            else:
                mean = float(np.mean([p["slope"] for p in per])); lo = [p["ci"][0] for p in per]; hi = [p["ci"][1] for p in per]
                if all(l > 0 for l in lo):
                    lb = "KINEMATIC" if 0.5 <= mean <= 1.5 else ("WEAK" if mean < 0.5 else "OVERSHOOT")
                elif all(h < 0 for h in hi):
                    lb = "WRONG-SIGN"
                else:
                    lb = "INERT"
            card["A4"][k] = {"label": lb, "per_seed": per}
        card["controls"]["A4_c9"] = [img[s]["controls"]["c9_zero_offset_maxabs"] for s in SEEDS]
    card["A5"] = J(f"{WORK}/a5/RESULT_A5.json")
    card["A5_centre"] = J(f"{WORK}/a5/RESULT_A5_CENTRE.json")
    json.dump(card, open(f"{WORK}/WUA_CARD.json", "w"), indent=1)
    write_md(card)
    print("WROTE", OUT_MD)


def fmt_enc(c):
    if not c or c.get("label") in (None,):
        return "-"
    ps = c["per_seed"]
    if any(p is None or p.get("delta") is None for p in ps):
        return c["label"]
    d = " / ".join("%+.3f" % p["delta"] for p in ps)
    lo = min(p["ci"][0] for p in ps); hi = max(p["ci"][1] for p in ps)
    return "%s<br>%s<br>[%+.3f, %+.3f]" % (c["label"], d, lo, hi)


def write_md(card):
    L = ["# World Understanding Card — v8 flagship RH_s0..s2 (card ADD155, Phase A)", "",
         "MEASURED, tier T0 (world-model diagnostics on logged frames, never driving performance); eval-141 deciding; three training seeds (values s0 / s1 / s2; the interval shown is the widest of the three per-seed 95 % episode-cluster bootstrap intervals, B 2,000). Four families: not applicable (no section scores a rollout). Labels applied by `wua_card.py` exactly as frozen in `PREREG_V8F_PROOF_ADD155.md`.", "",
         "## A1 ENCODED — selectivity over the ego-only control (dR^2, dAUROC or d balanced accuracy; linear probe)", ""]
    tgts = sorted({k.split("|")[1] for k in card["A1"]}, key=lambda t: (t not in BAR, t))
    L.append("| variable | " + " | ".join(REPS) + " |")
    L.append("|---|" + "---|" * len(REPS))
    tgts = [t for t in tgts if t not in ("speed", "accel", "yaw_rate")]
    for t in tgts:
        L.append(f"| `{t}`{' (bar 0.10)' if t in BAR else ''} | " + " | ".join(fmt_enc(card["A1"].get(f"{r}|{t}")) for r in REPS) + " |")
    if card["A1_mlp"]:
        L += ["", "Temporal MLP (state at t-4..t + ego history vs MLP on ego history):", "", "| variable | L1 | L2 | L3 | TOK-MEAN |", "|---|---|---|---|---|"]
        for t in sorted({k.split("|")[1] for k in card["A1_mlp"]}):
            L.append(f"| `{t}` | " + " | ".join(fmt_enc(card["A1_mlp"].get(f"{r}|{t}")) for r in ("L1", "L2", "L3", "TOK-MEAN")) + " |")
    if card.get("A1_tlin"):
        L += ["", "Linear temporal probe P-TLIN (Amendment 4; state at t-4..t + ego history vs ego history; two-part rule with 19 derangements):", "", "| variable | L1 | L2 | L3 | TOK-MEAN |", "|---|---|---|---|---|"]
        for t in sorted({k.split("|")[1] for k in card["A1_tlin"]}):
            L.append(f"| `{t}` | " + " | ".join(fmt_enc(card["A1_tlin"].get(f"{r}|{t}")) for r in ("L1", "L2", "L3", "TOK-MEAN")) + " |")
    if card["A2"]:
        L += ["", "## A2 IMAGINED — ego-matched counterfactual pairs (pair-discrimination AUC of gap change over 2 s)", "", "| scorer | label | AUC s0 / s1 / s2 | minus EGO s0 / s1 / s2 | SAME-pair AUC s0 / s1 / s2 |", "|---|---|---|---|---|"]
        for sc, c in card["A2"].items():
            ps = c["per_seed"]
            au = " / ".join("%.3f [%.3f, %.3f]" % (p["auc"], p["ci"][0], p["ci"][1]) if p else "-" for p in ps)
            me = " / ".join("%+.3f [%+.3f, %+.3f]" % (p["minus_ego"], p["minus_ego_ci"][0], p["minus_ego_ci"][1]) if p and "minus_ego" in p else "-" for p in ps)
            sa = " / ".join("%.3f" % p["auc"] if p else "-" for p in c["same_pairs"])
            L.append(f"| {sc} | {c['label']} | {au} | {me} | {sa} |")
        L.append(f"\nControls: {json.dumps(card['controls'].get('A2'))}")
    if card["A3"]:
        L += ["", "## A3 IMAGINED — skill over copy-last (probe on the rollout vs on the present state)", "", "| level, k (h) | variable | label | skill s0 / s1 / s2 | oracle R^2 | copy - imag MSE [CI] per seed |", "|---|---|---|---|---|---|"]
        for k, c in card["A3"].items():
            ps = c["per_seed"]
            if any(p is None or "skill" not in p for p in ps):
                L.append(f"| {k} | | {c['label']} | | | |"); continue
            lv, kk, var = k.split("|")
            L.append(f"| {lv} {kk} ({ps[0]['h_s']} s) | `{var}` | {c['label']} | " + " / ".join("%+.3f" % p["skill"] for p in ps) + " | " +
                     " / ".join("%.2f" % p["oracle_r2"] for p in ps) + " | " + " / ".join("%+.3g [%+.3g, %+.3g]" % (p["copy_minus_imag"], p["ci"][0], p["ci"][1]) for p in ps) + " |")
    if card["A4"]:
        L += ["", "## A4 IMAGINED under the ego's own action — slope of the imagined response against the kinematic expectation (1 = exact)", "", "| level, k (h) | offset | variable | label | slope [CI] s0 / s1 / s2 |", "|---|---|---|---|---|"]
        for k, c in card["A4"].items():
            ps = c["per_seed"]
            if any(p is None or "slope" not in p for p in ps):
                L.append(f"| {k} | | | {c['label']} | |"); continue
            lv, kk, off, var = k.split("|")
            L.append(f"| {lv} {kk} ({ps[0]['h_s']} s) | {off} | `{var}` | {c['label']} | " + " / ".join("%+.2f [%+.2f, %+.2f]" % (p["slope"], p["ci"][0], p["ci"][1]) for p in ps) + " |")
    L += ["", "## A5 USED", ""]
    if card["A5"]:
        L += ["| seed | abs dv LEAD - abs dv CONTROL (mean-token mask) [CI] | signed dv LEAD | signed dv CONTROL | C-CENTRE abs dv LEAD - centre-matched [CI] | signed dv centre-matched |", "|---|---|---|---|---|---|"]
        for s in SEEDS:
            r = J(f"{WORK}/a5/RESULT_A5_s{s}.json"); c = J(f"{WORK}/a5/RESULT_A5_CENTRE_s{s}.json")
            f = lambda d: "%+.3f [%+.3f, %+.3f]" % (d["point"], d["lo"], d["hi"])
            L.append(f"| s{s} | {f(r['mean']['metric_absdv_lead_minus_control'])} | {f(r['mean']['signed_dv_lead'])} | {f(r['mean']['signed_dv_control'])} | " +
                     (f"{f(c['metric_absdv_lead_minus_centre'])} | {f(c['signed_dv_centre'])} |" if c else "- | - |"))
        L.append("")
        L.append("A5 label: " + json.dumps({k: card["A5"].get(k) for k in card["A5"] if k in ("FINAL", "c11_all_seeds_pass")}) + " ; c10 pooled lead minus control logit drop %+.3f [%+.3f, %+.3f]" % (card["A5"]["c10_pooled_seeds"]["point"], card["A5"]["c10_pooled_seeds"]["lo"], card["A5"]["c10_pooled_seeds"]["hi"]))
        if card.get("A5_centre"):
            L.append("Amendment 2 name: " + card["A5_centre"]["name"])
    L += ["", f"Controls: {json.dumps({k: v for k, v in card['controls'].items() if k != 'A2'})}"]
    open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
