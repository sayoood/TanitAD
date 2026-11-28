"""E0j reader (CPU, numpy only): `pgrad_gram.json` (banded stage; optionally the baseline8 stage) -> `raw/E0J_RESULT.json`
(+ the card's result block). Controls FIRST: ANY failed control makes the reading VOID and NO weight is quoted -- the weight then
falls back to the E1 dose arms (REFCV9_DESIGN 2.2; G-SMOKE-L4 failed linearity at 4.2e-4 once).

    python e0j_read.py --gram <dir>/pgrad_gram.json [--baseline8 <dir>/pgrad_gram.json] --out raw/E0J_RESULT.json [--card E0j_CARD.md]

Decision groups: layer2 = `stage_layer2` (the map's own tap), trunk = whole trunk (sum of the trunk groups), lift = `lift`
(`_map_hires.lift.`); `x_shared_bev_025m` = lift + map_encoder is reported for comparability with the refcv7 P-GRAD. A group's Gram
is the SUM of its disjoint parts' Grams (the dot of concatenated vectors adds).
Estimator for the spread: BATCH BOOTSTRAP over the n=8 batches of 8 windows (resamples WINDOW-DRAW variance only; it is blind to
TRAINING variance (`H-ESTIM-SEED-1`) and to INFERENCE variance, and 8 units is coarse). Per-batch selections are the 8 diagonal blocks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gram_core as gc  # noqa: E402

TARGETS = {"M2a": 0.05, "M2b": 0.15}
FLOORS = {"layer2": 0.20, "trunk": 0.20, "lift": 0.25}
MAP_TARGETS_OTHER = {"trunk": 0.02, "lift": 0.25}            # the design table's other map targets (reported, not the selector's key)
BANKED_REFCV7 = {                                             # MEASURED, repo: .../2026-10-04-refcv8-perception-architecture/raw/pgrad.json (refcv7-50,400)
    "source": "refcv8-perception-architecture/raw/pgrad.json (P-GRAD, refcv7-r101-s0 @ 50,400, 64 TRAIN-DIAG windows); denominators include traj/tac_v6/rest",
    "layer2": {"map_hires": 0.00098, "box3d": 0.32727, "agent": 0.45241},
    "trunk": {"map_hires": 0.00014, "box3d": 0.35717, "agent": 0.50139},
    "x_shared_bev_025m": {"map_hires": -0.00405, "box3d": 0.97294, "agent": 0.0}}
RUN_RANGE_REFCV8 = "refcv8 in-run, whole trunk 0.00018-0.00396 (REFCV9_DESIGN 2.2, INHERITED; in-run statistic, not re-verified here)"


def group_k(rec, name):
    K = {g: np.asarray(k, dtype=np.float64) for g, k in rec["K"].items()}
    if name == "layer2":
        return K["stage_layer2"]
    if name == "trunk":
        return sum(K[g] for g in rec["trunk_groups"])
    if name == "lift":
        return K["lift"]
    if name == "x_shared_bev_025m":
        return K["lift"] + K["map_encoder"]
    return K[name]


def gms_of(K, nb, batches=None, weights=None):
    return gc.pooled(K, nb, batches, weights)[0]


def controls(rec, Kg):
    nb = rec["n_batches"]
    raw = rec["controls_raw"]
    w2 = [rec["w2"][t] for t in gc.TERMS]
    out = {"L": {}, "gram_checks": {}}
    for g in ("layer2", "trunk", "lift"):
        lr = gc.linearity_report(Kg[g], nb, w2)
        out["L"][g] = None if lr is None else {"max_rel_grad_err": lr["max_rel_grad_err"], "max_share_diff": lr["max_share_diff"],
                                               "pooled": lr["pooled"]}
        out["gram_checks"][g] = gc.check_gram(gms_of(Kg[g], nb))
    Ls = [v for v in out["L"].values()]
    l_ran = all(v is not None and v["max_rel_grad_err"] is not None for v in Ls)
    out["L_ran"] = l_ran
    out["L_max_rel_grad_err"] = max(v["max_rel_grad_err"] for v in Ls) if l_ran else None
    out["L_max_share_diff"] = max(v["max_share_diff"] for v in Ls) if l_ran else None
    out["L_pass"] = bool(l_ran and out["L_max_rel_grad_err"] <= 1e-4 and out["L_max_share_diff"] <= 1e-4)
    out["Z_zero_weight_max_abs"] = raw["Z_zero_weight_max_abs"]
    out["Z_pass"] = bool(raw["Z_n_batches"] > 0 and raw["Z_zero_weight_max_abs"] == 0.0)
    out["T_hidden_term_max_rel_err"] = raw["T_hidden_term_max_rel_err"]
    out["T_pass"] = bool(raw["T_n_batches"] > 0 and raw["T_hidden_term_max_rel_err"] <= 1e-4)
    out["value_check_max_rel"] = raw["value_check_max_rel"]
    out["value_pass"] = bool(raw["value_check_max_rel"] <= 1e-6)
    out["S_max_abs"] = gc.sum_gram_check({g: gms_of(Kg[g], nb) for g in Kg})
    out["S_pass"] = bool(out["S_max_abs"] <= 1e-9)
    out["gram_pass"] = bool(all(c["psd"] and c["cs_ok"] and c["symmetric"] for c in out["gram_checks"].values()))
    out["nonzero_terms"] = {g: [float(np.sqrt(max(gms_of(Kg[g], nb)[i, i], 0.0))) for i in range(3)] for g in Kg}
    # liveness: map and box3d must reach every decision group; agent must reach layer2 and the trunk. The agent head does NOT read the
    # BEV lift, so its lift norm is a STRUCTURAL zero (refcv7 P-GRAD: agent exactly 0.0 at x_shared_bev_025m) -- not a dead term.
    need = {"layer2": (0, 1, 2), "trunk": (0, 1, 2), "lift": (0, 1)}
    out["terms_live"] = bool(all(out["nonzero_terms"][g][i] > 0.0 for g, ix in need.items() for i in ix))
    out["required"] = ["L_pass", "Z_pass", "T_pass", "value_pass", "S_pass", "gram_pass", "terms_live"]
    out["all_pass"] = bool(all(out[k] for k in out["required"]))
    return out


def shares_table(Kg, nb, w_map_list, w_box=1.0, w_agent=1.0, groups=("layer2", "trunk", "lift", "x_shared_bev_025m")):
    t = {}
    for g in groups:
        gm = gms_of(Kg[g], nb)
        t[g] = {}
        for w in w_map_list:
            s = gc.share_from_gram(gm, [w, w_box, w_agent])
            t[g][f"{w:g}"] = None if s is None else {k: float(s[i]) for i, k in enumerate(gc.TERMS)}
    return t


def selection(Kg, nb, target, batches=None, weights=None):
    gms = {g: gms_of(Kg[g], nb, batches, weights) for g in ("layer2", "trunk", "lift")}
    return gc.select_w_map(gms, target, group="layer2", floors=FLOORS)


def spread(Kg, nb, target, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    per_batch = []
    for b in range(nb):
        r = selection(Kg, nb, target, batches=[b])
        per_batch.append({"w_star": r.get("w_star"), "floors_all_ok": r.get("floors_all_ok"), "found": r["found"]})
    ws, nfound, nfeas = [], 0, 0
    for _ in range(B):
        m = rng.multinomial(nb, [1.0 / nb] * nb)
        r = selection(Kg, nb, target, weights=m)
        if r["found"]:
            nfound += 1
            ws.append(r["w_star"])
            nfeas += int(bool(r["floors_all_ok"]))
    q = (lambda p: float(np.percentile(ws, p))) if ws else (lambda p: None)
    return {"estimator": "batch bootstrap over n=8 batches of 8 windows (window-draw variance only)", "B": B, "seed": seed,
            "frac_found": nfound / B, "frac_floors_ok": nfeas / B, "w_star_p05": q(5), "w_star_p50": q(50), "w_star_p95": q(95),
            "per_batch_w_star": per_batch}


def read_stage(path):
    rec = json.loads(Path(path).read_text(encoding="utf-8"))
    nb = rec["n_batches"]
    Kg = {g: group_k(rec, g) for g in ("layer2", "trunk", "lift", "x_shared_bev_025m")}
    return rec, nb, Kg


def build(gram_path, baseline_path=None):
    rec, nb, Kg = read_stage(gram_path)
    res = {"probe": "E0j", "schema": "e0j-result-v1", "gram_json": str(gram_path), "ckpt_md5": rec["ckpt_md5"], "n_batches": nb,
           "batch": rec["batch"], "window_draw_sha256": rec["window_draw_sha256"], "w_configured": rec["w_configured"], "w2": rec["w2"],
           "group_sizes": {g: v["n_params"] for g, v in rec["groups"].items()},
           "group_sizes_decision": {"layer2": rec["groups"]["stage_layer2"]["n_params"],
                                    "trunk": sum(rec["groups"][g]["n_params"] for g in rec["trunk_groups"]),
                                    "lift": rec["groups"]["lift"]["n_params"]},
           "design_estimate_lift_params": 131200, "tier": rec["tier"], "measurement_choices": rec["measurement_choices"],
           "evidence_class": "MEASURED (ours, this artifact); one checkpoint, 64 TRAIN-DIAG windows, a FRESH seed-0 banded head"}
    c = controls(rec, Kg)
    res["controls"] = c
    res["VOID"] = not c["all_pass"]
    gm1 = {g: gms_of(Kg[g], nb).tolist() for g in Kg}
    res["gram_pooled"] = gm1
    res["cos_pooled"] = {g: {f"{gc.TERMS[i]}|{gc.TERMS[j]}": float(np.array(gm1[g])[i, j] / np.sqrt(np.array(gm1[g])[i, i] * np.array(gm1[g])[j, j]))
                             for i in range(3) for j in range(i + 1, 3)} for g in Kg
                         if all(np.array(gm1[g])[i, i] > 0 for i in range(3))}
    res["norm_pooled"] = {g: [float(np.sqrt(max(np.array(gm1[g])[i, i], 0.0))) for i in range(3)] for g in Kg}
    res["shares_at_w1"] = shares_table(Kg, nb, [1.0])
    res["shares_vs_w_map"] = shares_table(Kg, nb, [0.25, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0])
    res["banked_refcv7_p_grad"] = BANKED_REFCV7
    res["refcv8_in_run_range"] = RUN_RANGE_REFCV8
    res["per_batch_n"] = [{"n_map_labelled": r["n_map_labelled"], "n_map_windows": r["n_map_windows"], "n_agent_labelled": r["n_agent_labelled"],
                           "term_present": r["term_present"]} for r in rec["per_batch"]]
    if res["VOID"]:
        res["selection"] = None
        res["verdict"] = ("VOID: a control failed (" + ", ".join(k for k in c["required"] if not c[k]) + "). NO w_map is quoted; the weight "
                          "falls back to the E1 dose arms (M2a / M2b run at their declared doses and the arms read the share).")
    else:
        sel = {}
        for arm, tgt in TARGETS.items():
            s = selection(Kg, nb, tgt)
            s["spread"] = spread(Kg, nb, tgt)
            if s["found"]:
                w = s["w_star"]
                s["shares_all_groups_at_w_star"] = shares_table(Kg, nb, [w])
                s["other_map_targets_met"] = {g: bool(s["shares_at_w_star"][g]["map"] >= v) for g, v in MAP_TARGETS_OTHER.items()}
            sel[arm] = s
        res["selection"] = sel
        parts = []
        for arm, s in sel.items():
            parts.append(f"{arm} (layer2 map share >= {TARGETS[arm]}): " + (
                f"w_map* = {s['w_star']:.4g}, box3d floors {'HELD' if s['floors_all_ok'] else 'BROKEN'}" if s["found"] else "NOT REACHABLE"))
        res["verdict"] = "controls PASS. " + "; ".join(parts)
    if baseline_path:
        brec, bnb, bKg = read_stage(baseline_path)
        bc = {"Z_pass": brec["controls_raw"]["Z_n_batches"] == 0 or brec["controls_raw"]["Z_zero_weight_max_abs"] == 0.0,
              "gram_pass": all(gc.check_gram(gms_of(bKg[g], bnb))["psd"] for g in bKg)}
        res["baseline8_w1"] = {"stage": brec["stage"], "ckpt_md5": brec["ckpt_md5"], "window_draw_sha256": brec["window_draw_sha256"],
                               "same_windows_as_banded": brec["window_draw_sha256"] == rec["window_draw_sha256"],
                               "controls_light": bc,
                               "shares_at_w1": shares_table(bKg, bnb, [1.0]), "norm_pooled": {g: [float(np.sqrt(max(gms_of(bKg[g], bnb)[i, i], 0.0))) for i in range(3)] for g in bKg}}
    return res


def card_block(res):
    L = []
    c = res["controls"]
    L.append(f"**Controls (read first).** L linearity: max |share diff| {c['L_max_share_diff']:.2e}, max rel grad err {c['L_max_rel_grad_err']:.2e} "
             f"({'PASS' if c['L_pass'] else 'FAIL'}, <= 1e-4); Z zero-weight max |g| {c['Z_zero_weight_max_abs']} ({'PASS' if c['Z_pass'] else 'FAIL'}); "
             f"T hidden-term {c['T_hidden_term_max_rel_err']:.2e} ({'PASS' if c['T_pass'] else 'FAIL'}); value {c['value_check_max_rel']:.1e}; "
             f"S {c['S_max_abs']:.1e}; Gram sym/PSD {'PASS' if c['gram_pass'] else 'FAIL'}; all terms live {c['terms_live']}." if c["L_ran"] else
             "**Controls.** L did not run.")
    L.append("")
    L.append(f"**Verdict: {'VOID -- ' if res['VOID'] else ''}{res['verdict']}**")
    L.append("")
    L.append("Shares at w = 1 (map 1 / box3d 1 / agent 1; the E1 C0 weights), projected share of the three-term update:")
    L.append("")
    L.append("| group | map | box3d | agent |")
    L.append("|---|---|---|---|")
    for g, d in res["shares_at_w1"].items():
        s = d["1"]
        L.append(f"| {g} | {s['map']:+.4f} | {s['box3d']:+.4f} | {s['agent']:+.4f} |" if s else f"| {g} | n/a | n/a | n/a |")
    if res.get("selection"):
        L.append("")
        L.append("Selected weights (smallest `w_map` with the layer2 map share at target, box3d floors layer2 >= 0.20, trunk >= 0.20, lift >= 0.25):")
        L.append("")
        L.append("| arm | target | w_map* | layer2 map / box3d | trunk map / box3d | lift map / box3d | floors | batch-bootstrap w* p05 / p50 / p95 |")
        L.append("|---|---|---|---|---|---|---|---|")
        for arm, s in res["selection"].items():
            if not s["found"]:
                L.append(f"| {arm} | {s['target']} | not reachable | | | | | |")
                continue
            sh = s["shares_at_w_star"]
            sp = s["spread"]
            f = lambda g: f"{sh[g]['map']:+.4f} / {sh[g]['box3d']:+.4f}"  # noqa: E731
            L.append(f"| {arm} | {s['target']} | {s['w_star']:.4g} | {f('layer2')} | {f('trunk')} | {f('lift')} | "
                     f"{'held' if s['floors_all_ok'] else 'BROKEN'} (w limit {s['w_floor_limit']}) | "
                     f"{sp['w_star_p05']} / {sp['w_star_p50']} / {sp['w_star_p95']} (found {sp['frac_found']:.2f}, floors ok {sp['frac_floors_ok']:.2f}) |")
    if res.get("baseline8_w1"):
        L.append("")
        L.append("Matched-instrument refcv8 baseline (the 8-way 10 cm CE map loss, same windows, w = 1):")
        L.append("")
        L.append("| group | map | box3d | agent |")
        L.append("|---|---|---|---|")
        for g, d in res["baseline8_w1"]["shares_at_w1"].items():
            s = d["1"]
            L.append(f"| {g} | {s['map']:+.4f} | {s['box3d']:+.4f} | {s['agent']:+.4f} |" if s else f"| {g} | n/a | n/a | n/a |")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gram", required=True)
    ap.add_argument("--baseline8")
    ap.add_argument("--out", required=True)
    ap.add_argument("--card")
    a = ap.parse_args(argv)
    res = build(a.gram, a.baseline8)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    blk = card_block(res)
    if a.card:
        p = Path(a.card)
        t = p.read_text(encoding="utf-8")
        b, e = "<!--E0J-RESULT-BEGIN-->", "<!--E0J-RESULT-END-->"
        if b not in t or e not in t:
            raise SystemExit("[e0j-read] the card has no result markers")
        i, j = t.index(b) + len(b), t.index(e)
        p.write_text(t[:i] + "\n" + blk + "\n" + t[j:], encoding="utf-8", newline="\n")
    print(blk)
    print(f"ZZE0J-READ-{'VOID' if res['VOID'] else 'OK'}ZZ")
    return 0 if not res["VOID"] else 3


if __name__ == "__main__":
    sys.exit(main())
