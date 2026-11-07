#!/usr/bin/env python3
"""YF3 ANALYSIS: mechanically evaluates the COMMITTED rule of PREREG_YAWFT.md section 6 for one rung. Writes <root>/analysis/<rung>.json + .md; last line ZZYF3VERDICT <rung> <ADOPT|NOT PROVEN|REFUTED|VOID>.

    python yf3_analyze.py --root <run root> --rung R1|R2|R3 [--tokens <w3 token json>] [--seeds 0,1]

Arms read:  B = <rung>_B_s<seed> ; reference = A0 (R1, valid because gate K3 proves C == A0 bit for bit; if K3 is red the R1 C arm is evaluated and used) or <rung>_C_s<seed> (R2/R3).
Tier stamp (printed in every output): NAVSIM v1.1 PDMS / families6 on the 923 A5 tokens = ego pseudo-simulation of an OPEN-LOOP plan against logged agents; NOT the action-closed-loop tier.
Nothing here may be quoted as "drives better".
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import yf3_common as C  # noqa: E402
import yf3_stats as S  # noqa: E402

sys.path.insert(0, str(C.EVAL))
sys.path.insert(0, str(C.Y2))

TIER = ("NAVSIM v1.1 pseudo-simulation (PDMS) and families6 of an OPEN-LOOP plan against logged agents on the A5 confirmation tokens (n in the header) -- NOT the EVAL_DOCTRINE action-closed-loop tier "
        "(families6 labels itself 'T1-family (stage-1 loop OPEN)'). Cached-frame VAL diagnostics are T0 (teacher-forced). Nothing here may be quoted as 'drives better'.")
TACTICAL = "n/a as a test: REFe has no tactical layer; families6 derives lateral / longitudinal / 5-way decisions from the EXECUTED geometry, so they are reported descriptively beside each arm"
STRATEGIC = "PAUSED (PI ruling 2026-09-19, PI_DECISION_QUEUE.md item 25): not a pass, not omitted. NAVSIM scores no strategic decision and REFe has no strategic layer."
VARIANCE = ("Q1 another draw of LOGS: paired log-cluster bootstrap, 10,000 resamples, percentile 95 %, seed 20260927, the same resample for every arm (answers that and nothing else). "
            "Q2 another TRAINING run: two seeds per arm, floor F = |Delta(s0) - Delta(s1)|; a separated CI from one seed is necessary, not sufficient (H-ESTIM-SEED-1). "
            "Q3 another INFERENCE run: the planner is deterministic (ego probe C1: replicate bit-identical on 265/265 tokens), no inference-seed arm. NOT measured: another BASE training run -- the claim is scoped to this u400 checkpoint.")
SUB = ["no_at_fault_collisions", "drivable_area_compliance", "ego_progress", "time_to_collision_within_bound", "comfort", "driving_direction_compliance", "score"]


def read_csv(p):
    rows = {}
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r.get("token") in (None, "", "average"):
            continue
        rows[r["token"]] = r
    return rows


def load_arm(root, name, toks):
    d = Path(root) / "eval" / name
    pw = np.load(d / "perwindow.npz", allow_pickle=False)
    order = {str(t): i for i, t in enumerate(pw["token"])}
    ix = np.asarray([order[t] for t in toks])
    rows = read_csv(C.jload(d / "arm.json")["score"]["csv"])
    z = np.load(d / "seam.npz")
    zi = {str(t): i for i, t in enumerate(z["token"])}
    return {"name": name, "dir": d, "pw": {k: pw[k][ix] for k in pw.files}, "csv": rows,
            "sub": {k: np.asarray([100.0 * float(rows[t][k]) for t in toks]) for k in SUB},
            "poses": np.asarray([z["poses"][zi[t]] for t in toks]), "fam": C.jload(d / "families.json"), "g7": (C.jload(d / "shim_g7.json") if (d / "shim_g7.json").exists() else None)}


def pooled_mae(pw, num, den):
    return float(pw[num].sum() / pw[den].sum())


def gate_g6(a0, toks):
    """A0 re-run == banked per-token plans AND per-token scores, bit for bit."""
    z = np.load(C.BANKED_SEAM)
    bi = {str(t): i for i, t in enumerate(z["token"])}
    same_plans = all(np.array_equal(z["poses"][bi[t]], a0["poses"][i]) for i, t in enumerate(toks))
    brow = read_csv(C.BANKED_CSV)
    same_scores = all(float(brow[t]["score"]) == float(a0["csv"][t]["score"]) for t in toks)
    mean_pdms = float(np.mean(a0["sub"]["score"]))
    return {"ok": bool(same_plans and same_scores), "plans_bit_identical": bool(same_plans), "scores_identical": bool(same_scores), "pdms_x100": mean_pdms, "banked_pdms_x100": 88.6473,
            "n": len(toks)}


def gate_g7(arm, toks, table, shim_expected: bool):
    g = arm["g7"]
    if g is None:
        return {"ok": False, "why": "no shim_g7.json (the seam did not run under the G7 recorder)"}
    if not shim_expected:
        return {"ok": bool(not g["shim_installed"] and g["n_calls"] == 0), "shim_installed": g["shim_installed"], "n_calls": g["n_calls"], "note": "non-yaw checkpoint: YT's shim must stay out of the planner"}
    fed = g["by_token"]
    want = {t: float(np.float32(C.YAW_SCALE * table[t])) for t in toks}
    exact = sum(1 for t in toks if t in fed and fed[t] == want[t])
    nonzero = sum(1 for t in toks if t in fed and fed[t] != 0.0)
    rep = (g["shim_reports"] or [{}])[0]
    ok = bool(g["shim_installed"] and exact == len(toks) and nonzero >= 0.9 * len(toks) and rep.get("n_incoming_nonzero", 1) == 0 and abs(rep.get("scale", 0) - C.YAW_SCALE) < 1e-12
              and g["value_changed_between_calls"] == 0)
    return {"ok": ok, "n_tokens": len(toks), "tokens_fed": len(fed), "equal_float32_10x_table": exact, "nonzero": nonzero, "nonzero_fraction": nonzero / max(len(toks), 1),
            "n_incoming_nonzero": rep.get("n_incoming_nonzero"), "scale": rep.get("scale"), "navtest_table_sha256": rep.get("table_sha256"), "n_calls": g["n_calls"]}


def collect_gates(root, rung, B_names, ref_names, arms, toks, table):
    root = Path(root)
    g, red = {}, []

    def put(k, ok, **kw):
        g[k] = {"ok": bool(ok), **kw}
        if not ok:
            red.append(k)

    rg = C.jload(root / "gates_real" / "real_cache_gates.json") if (root / "gates_real" / "real_cache_gates.json").exists() else None
    put("G1_table_coverage(YT control is YT's; here: every eval token present)", all(t in table for t in toks), n=len(toks), table_sha256=C.TABLE_TEST_SHA[:12])
    put("G2/G3/G4/G5/G9_real_cache_suite_with_RED_arms", bool(rg and rg.get("_ok")), file="gates_real/real_cache_gates.json")
    tr = []
    for n in B_names + [x for x in ref_names if x != "A0"]:
        p = root / "arms" / n / "gates.json"
        tr.append(bool(p.exists() and C.jload(p).get("pass")))
    put("G4/G5_in_each_trained_arm(gates.json before step 1)", all(tr) and len(tr) > 0, n_arms=len(tr))
    ids = []
    for n in B_names + [x for x in ref_names if x != "A0"]:
        p = root / "identity" / f"{n}.json"
        ids.append(bool(p.exists() and C.jload(p).get("ok")))
    put("G-CKPT/K1-K4/identity(C==A0, B0==A0, trained decode==composed decode)", all(ids) and len(ids) > 0, n=len(ids))
    if "A0" in arms:
        r6 = gate_g6(arms["A0"], toks)
        put("G6_A0_rerun_bit_identical_to_banked", r6.pop("ok"), **r6)
    elif (root / "analysis" / "G6.json").exists():
        r6 = C.jload(root / "analysis" / "G6.json")
        put("G6_A0_rerun_bit_identical_to_banked", r6["ok"], carried_from="analysis/G6.json")
    else:
        put("G6_A0_rerun_bit_identical_to_banked", False, why="A0 not evaluated")
    for n in B_names:
        r7 = gate_g7(arms[n], toks, table, True)
        put(f"G7_{n}", r7.pop("ok"), **r7)
    for n in ref_names:
        r7 = gate_g7(arms[n], toks, table, False)
        put(f"G7_{n}(non-yaw: shim untouched)", r7.pop("ok"), **r7)
    g8 = root / "analysis" / "G8.json"
    put("G8_determinism(CPU col4 torch.equal on a B rerun; eval 10 % of tokens bit-identical)", g8.exists() and C.jload(g8).get("ok", False), file="analysis/G8.json")
    return g, red


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--rung", required=True, choices=("R1", "R2", "R3"))
    ap.add_argument("--tokens", default=str(C.A5_TOKENS))
    ap.add_argument("--seeds", default="0,1")
    a = ap.parse_args(argv)
    root = Path(a.root)
    seeds = [int(s) for s in a.seeds.split(",")]
    toks = sorted(C.jload(a.tokens)["tokens"])
    tl = C.jload(a.tokens).get("token_log") or {}
    import yaw_common as YC
    table, _, tsha = YC.load_yaw_table(C.TABLE_TEST)
    assert tsha == C.TABLE_TEST_SHA, "navtest table sha mismatch"
    B_names = [f"{a.rung}_B_s{s}" for s in seeds]
    c_eq_a0 = None                                   # only meaningful at R1 (R2 / R3: C trains the rest of the network with yaw 0, so C != A0 by design)
    if a.rung == "R1":
        idp = root / "identity" / "R1_C_s0.json"
        c_eq_a0 = bool(idp.exists() and C.jload(idp).get("C_state_dict_equals_A0"))
        ref_names = ["A0"] * len(seeds) if c_eq_a0 else [f"R1_C_s0"] * len(seeds)
    else:
        ref_names = [f"{a.rung}_C_s{s}" for s in seeds]
    need = sorted(set(B_names + ref_names + (["A0"] if (root / "eval" / "A0" / "arm.json").exists() else [])))
    arms = {n: load_arm(root, n, toks) for n in need}
    logs = [arms[B_names[0]]["pw"]["log"][i] for i in range(len(toks))]
    bt = S.Boot(logs)
    # ---- gates
    gates, red = collect_gates(root, a.rung, B_names, sorted(set(ref_names)), arms, toks, table)
    # ---- contrasts (B - Ref), seed-averaged; each statistic is a ratio of per-window sums, so it reproduces the families6 pooled value
    def lat(num, den, scale=1.0):
        arms_b = [(arms[b]["pw"][num], arms[b]["pw"][den]) for b in B_names]
        arms_r = [(arms[r]["pw"][num], arms[r]["pw"][den]) for r in ref_names]
        c = S.contrast(bt, arms_b, arms_r)
        if scale != 1.0:
            c = {**c, "point": c["point"] * scale, "ci95": [c["ci95"][0] * scale, c["ci95"][1] * scale], "per_seed_point": [x * scale for x in c["per_seed_point"]],
                 "per_seed_ci95": [[x * scale for x in ci] for ci in c["per_seed_ci95"]], "F_seed_floor": (c["F_seed_floor"] * scale if c["F_seed_floor"] is not None else None)}
        c["ref_value"] = float(np.mean([pooled_mae(arms[r]["pw"], num, den) for r in ref_names]) * scale)
        c["arm_value"] = float(np.mean([pooled_mae(arms[b]["pw"], num, den) for b in B_names]) * scale)
        return c
    prim = lat("curv_abs_sum", "curv_n")
    secondary = {"heading_mae_deg": lat("head_abs_sum_rad", "head_n", 180.0 / math.pi), "yaw_rate_mae_degps": lat("yaw_abs_sum_radps", "yaw_n", 180.0 / math.pi),
                 "curvature_bias_1pm": lat("curv_signed_sum", "curv_n"), "cross_mae_m": lat("cross_abs_sum", "cross_n"), "cross_bias_m": lat("cross_signed_sum", "cross_n"),
                 "along_mae_m(longitudinal)": lat("along_abs_sum", "along_n"), "along_bias_m(longitudinal)": lat("along_signed_sum", "along_n")}
    pd_b = [(arms[b]["sub"]["score"], None) for b in B_names]
    pd_r = [(arms[r]["sub"]["score"], None) for r in ref_names]
    pdms = S.contrast(bt, pd_b, pd_r)
    pdms["ref_value"] = float(np.mean([arms[r]["sub"]["score"].mean() for r in ref_names]))
    pdms["arm_value"] = float(np.mean([arms[b]["sub"]["score"].mean() for b in B_names]))
    subs = {k: S.contrast(bt, [(arms[b]["sub"][k], None) for b in B_names], [(arms[r]["sub"][k], None) for r in ref_names]) for k in SUB[:-1]}
    # ---- families_adverse (families6 intervals, the stage2_sft.py reading; unpaired interval non-overlap, conservative)
    import proxy_eval as PE
    adverse = [PE.fam_adverse(arms[r]["fam"], arms[b]["fam"]) for b, r in zip(B_names, ref_names)]
    # ---- the committed rule
    dec = S.decide(gates_ok=not red, gates_red=red, prim=prim, ref_mae=prim["ref_value"], pdms=pdms, adverse_by_seed=[x["adverse"] for x in adverse])
    # ---- descriptive: pick change, strata, tactical
    turn = np.asarray([abs(table[t]) > 0.05 for t in toks])
    desc = {"pick_changed_share_vs_ref(any bit differs)": [float(np.mean([not np.array_equal(arms[b]["poses"][i], arms[r]["poses"][i]) for i in range(len(toks))])) for b, r in zip(B_names, ref_names)],
            "pick_changed_share_vs_ref(max pose difference > 1 cm)": [float(np.mean([float(np.abs(arms[b]["poses"][i][:, :2] - arms[r]["poses"][i][:, :2]).max()) > 0.01 for i in range(len(toks))])) for b, r in zip(B_names, ref_names)],
            "n_turn_tokens(|omega_t0|>0.05)": int(turn.sum()), "n_straight_tokens": int((~turn).sum()), "strata_curvature_mae_1pm": {}}
    for nm, mk in (("TURN", turn), ("STRAIGHT", ~turn)):
        row = {}
        for n in sorted(set(B_names + ref_names)):
            pw = arms[n]["pw"]
            row[n] = float(pw["curv_abs_sum"][mk].sum() / max(pw["curv_n"][mk].sum(), 1))
        desc["strata_curvature_mae_1pm"][nm] = row
    desc["tactical"] = {"status": TACTICAL, "per_arm_vs_ref": [x["tactical_no_interval"] for x in adverse]}
    desc["strategic"] = STRATEGIC
    fam_tab = {}
    for n in sorted(set(B_names + ref_names)):
        f = arms[n]["fam"]["families"]["refcv6"]
        fam_tab[n] = {fam: {k: v.get("mean") for k, v in ((f.get(fam) or {}).get("ci") or {}).get("components", {}).items()} for fam in ("longitudinal", "lateral")}
    out = {"rung": a.rung, "verdict": dec["verdict"], "decision": dec, "tier": TIER, "estimator": VARIANCE, "n_tokens": len(toks), "n_logs": len(set(logs)),
           "primary": {"metric": "LATERAL curvature MAE (1/m), executed pick vs human future; B - Ref; negative = B better", "B": B_names, "ref": ref_names, "C_equals_A0_by_K3": c_eq_a0, **prim},
           "secondary_lateral_and_longitudinal(paired B - Ref, same resample)": secondary, "pdms_x100": pdms, "pdms_subscores_x100": subs, "families_adverse": adverse,
           "families_component_means": fam_tab, "descriptive": desc, "gates": gates, "gates_red": red, "tactical": TACTICAL, "strategic": STRATEGIC,
           "yaw_table_navtest_sha256": tsha, "written": C.now()}
    (root / "analysis").mkdir(parents=True, exist_ok=True)
    if "A0" in arms:
        C.jdump(gate_g6(arms["A0"], toks), root / "analysis" / "G6.json")
    C.jdump(out, root / "analysis" / f"{a.rung}.json")
    (root / "analysis" / f"{a.rung}.md").write_text(render_md(out), encoding="utf-8")
    print(f"ZZYF3VERDICT {a.rung} {dec['verdict']}  ({dec['text'][:200]})")
    return 0


def render_md(o):
    p, d = o["primary"], o["decision"]
    L = [f"# YF3 analysis {o['rung']}: **{o['verdict']}**", "", f"Tier: {o['tier']}", "", f"Estimator: {o['estimator']}", "",
         f"n = {o['n_tokens']} tokens over {o['n_logs']} logs. Contrast B - Ref, Ref per seed = {p['ref']}, B = {p['B']}. C == A0 by gate K3: {p['C_equals_A0_by_K3'] if p['C_equals_A0_by_K3'] is not None else 'n/a at this rung (Ref = the same-seed C arm)'}.", "",
         f"## Primary (LATERAL curvature MAE, 1/m)", f"ref {p['ref_value']:.6f}, B {p['arm_value']:.6f}, Delta {p['point']:+.6f}, 95 % CI [{p['ci95'][0]:+.6f}, {p['ci95'][1]:+.6f}], relative improvement {d['relative_improvement'] * 100:+.2f} %",
         f"per seed Delta {[round(x, 6) for x in p['per_seed_point']]}, per-seed CI {[[round(x, 6) for x in c] for c in p['per_seed_ci95']]}, F = {p['F_seed_floor']}", "",
         "## Committed conditions", ""] + [f"* {'PASS' if v else 'FAIL'}  {k}" for k, v in d["conditions"].items()] + ["", f"Rule text: {d['rule']}", f"Verdict reason: {d['text']}", "",
         "## PDMS (x100) guard", f"ref {o['pdms_x100']['ref_value']:.4f}, B {o['pdms_x100']['arm_value']:.4f}, Delta {o['pdms_x100']['point']:+.4f} CI [{o['pdms_x100']['ci95'][0]:+.4f}, {o['pdms_x100']['ci95'][1]:+.4f}], per seed {o['pdms_x100']['per_seed_point']}, F {o['pdms_x100']['F_seed_floor']}",
         "", "| sub-score | Delta | CI |", "|---|---|---|"] + [f"| {k} | {v['point']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |" for k, v in o["pdms_subscores_x100"].items()] + [
         "", "## Secondary lateral / longitudinal (paired, same resample)", "", "| metric | ref | B | Delta | CI |", "|---|---|---|---|---|"] + [
         f"| {k} | {v['ref_value']:.5f} | {v['arm_value']:.5f} | {v['point']:+.5f} | [{v['ci95'][0]:+.5f}, {v['ci95'][1]:+.5f}] |" for k, v in o["secondary_lateral_and_longitudinal(paired B - Ref, same resample)"].items()] + [
         "", f"families_adverse (per seed): {[x['adverse'] for x in o['families_adverse']]}", "", f"TACTICAL: {o['tactical']}", f"STRATEGIC: {o['strategic']}", "", "## Descriptive",
         f"pick changed vs ref (any bit / > 1 cm): {o['descriptive']['pick_changed_share_vs_ref(any bit differs)']} / {o['descriptive']['pick_changed_share_vs_ref(max pose difference > 1 cm)']}; strata curvature MAE: {json.dumps(o['descriptive']['strata_curvature_mae_1pm'])}", "", "## Gates", ""] + [
         f"* {'GREEN' if v['ok'] else 'RED'}  {k}" for k, v in o["gates"].items()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    sys.exit(main())
