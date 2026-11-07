"""result_d: assemble RESULT_WSPD.md / RESULT_WSPD.json from the chain's raw outputs (READ1_D.json, READ1_D_EVAL.json, READ2_D.json, optional DK141 reads), name the verdict by the card's section 4 table.
No number is typed here: every one is read from a raw file. usage: result_d.py  (dev-box CPU; run after raw/CHAIN_D.done or raw/CHAIN_D.mechfail)"""
import os
import sys
import json
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
RAW = HERE + "/raw"
BANK = HERE.rsplit("/", 1)[0] + "/wsp_edit"
FAM = {"LONGITUDINAL": ["speed MAE", "along-track MAE", "accel MAE", "DK-VIOL", "DK-HW", "DK-TG"],
       "LATERAL": ["heading MAE", "yaw-rate MAE", "curvature MAE", "cross-track MAE", "L1 lateral-accel MAE", "L2 curvature MAE (GT>=5.2)"],
       "TACTICAL": ["TAC-LAT", "TAC-LON"], "STRATEGIC (PAUSED, PI item 25)": ["S-POS@30", "S-POS@50", "S-POS@80"], "ADE and M1": ["ADE6", "M1 crawl share"]}
NEXT = {"MECH-FAIL": "the edit is wrong, not the reading: name the defect and rebuild; no T2 stage",
        "HARMS": "the data-matched point joins P3's chain beside WSP-GAIN's; no T2 stage; the perception lever (LoRA) stays the route",
        "MOVES-M1": "the T2 stage on box 2 (after the queued BPC stage); DOMINANT-T2 there is the scale signal, with video, after a replicate as PREREG_T2_ATTRIB.md section 3 requires for a second arm",
        "NO-EFFECT": "the gain shortfall is not the crawl's lever at T1-kin; logged; the LoRA route stays"}


def jl(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def status(fam_tabs, label, row):
    for ft in fam_tabs.values():
        if label in ft["separated_better"]:
            return "better"
        if label in ft["separated_worse"]:
            return "worse"
    if label in ("ADE6", "M1 crawl share"):                       # lower is better for both
        return "ns" if not row["separated"] else ("better" if row["delta"] < 0 else "worse")
    return "ns"


def f(x, nd=4):
    return f"{x:+.{nd}f}"


def main():
    fit, gd = jl(RAW + "/FIT_RESULT_D.json"), jl(RAW + "/GDATA.json")
    r1, r1e, r2 = jl(RAW + "/READ1_D.json"), jl(RAW + "/READ1_D_EVAL.json"), jl(RAW + "/READ2_D.json")
    rb = jl(BANK + "/raw/READ2.json")
    dk = [jl(p) for p in sorted(os.path.join(RAW, n) for n in os.listdir(RAW) if n.startswith("DK141_READ_D") and n.endswith(".json"))]
    out = {"card": "PREREG_WSP_GAIN_D.md", "written_utc_from_date": time.strftime("%Y-%m-%d %H:%M"), "tier": "READ 1: T0-class mechanism check (no rollout). READ 2: T1-kin (logged cameras; reindexed = longitudinal-only approximate re-perception, not T2).",
           "estimator": "READ 2: paired episode-cluster bootstrap B 2,000 seed 0 (another draw of EPISODES only; the edit is a deterministic function of C0, so no training seed is in a D - C0 contrast)",
           "g_data": gd["g_data"], "g_data_ci95": gd["ci95_episode_cluster_bootstrap_B1000_seed0"], "p5_beside": gd["p5_beside"],
           "per_seed_fit": {s: {k: fit["seeds"][s][k] for k in ("g_model_train_median", "delta_g", "delta_m_m_s", "shift_target_m_s", "ckpt_sha256")} for s in ("0", "1")}}
    if r1 is None:
        out["verdict"] = "INCOMPLETE (READ 1 not run)"
    else:
        out["read1_train"] = r1
        out["read1_eval_beside"] = r1e
        if r1["verdict"] == "MECH-FAIL":
            out["verdict"] = "MECH-FAIL"
        elif r2 is None:
            out["verdict"] = "INCOMPLETE (READ 2 not run)"
        else:
            out["read2"] = {k: r2[k] for k in ("card", "tier", "estimator", "reader_sha256", "lf_read_sha256", "disclosure", "controls")}
            D = r2["doses"]["D"]
            out["read2"]["label"] = D["label"]
            out["read2"]["seeds"] = {s: {k: v for k, v in c.items() if k in ("m1_cell", "M1_WSP_minus_RIC0", "M1_RIC0_minus_P10", "M1_WSP_minus_P10", "share_closed", "new_separated_worse_rows", "n_worse_vs_P10",
                                                                              "separated_better_rows_WSP_minus_RIC0", "separated_worse_rows_WSP_minus_RIC0_all", "strategic_rows_beside_PAUSED", "family_tables",
                                                                              "rows_WSP_minus_RIC0", "rows_WSP_minus_P10", "rows_RIC0_minus_P10", "M1_per_class_split", "max_abs_plan_dev_vs_RIC0_m")}
                                     for s, c in D["seeds"].items()}
            out["verdict"] = D["label"]
            if D.get("note"):
                out["read2"]["note"] = D["note"]
    out["next_step_committed_by_card_section_4"] = NEXT.get(out["verdict"], "n/a")
    if rb:
        out["beside_banked_wsp_gain_alpha_1p0"] = {"label": rb["outcome_alpha_1p0"], "new_separated_worse_rows": {s: rb["doses"]["10"]["seeds"][s]["new_separated_worse_rows"] for s in ("s0", "s1")},
                                                   "M1_delta": {s: rb["doses"]["10"]["seeds"][s]["M1_WSP_minus_RIC0"]["delta"] for s in ("s0", "s1")}, "share_closed": {s: rb["doses"]["10"]["seeds"][s]["share_closed"] for s in ("s0", "s1")}}
    if dk:
        out["distance_keeping_beside"] = [d for d in dk]
    json.dump(out, open(HERE + "/RESULT_WSPD.json", "w", encoding="utf-8", newline="\n"), indent=1)
    md = [f"# RESULT_WSPD: E-GAIN-D (the weight edit aimed at the data's own longitudinal gains), card PREREG_WSP_GAIN_D.md", "",
          f"**Verdict by the card's section 4 table: {out['verdict']}.** Next step it commits: {out['next_step_committed_by_card_section_4']}.", "",
          "Tiers: READ 1 [M] T0-class mechanism check; READ 2 [M] T1-kin on eval-141 (scene `reindexed`), STRATEGIC family PAUSED (PI item 25). Estimator: " + out["estimator"] + ".", "",
          "## Fit [M] (20,000 TRAIN fit windows)", "",
          f"g_data: ds/dv0 {gd['g_data']['dsdv0']:.4f} {gd['ci95_episode_cluster_bootstrap_B1000_seed0']['v0']}, ds/da0 {gd['g_data']['dsda0']:.4f} {gd['ci95_episode_cluster_bootstrap_B1000_seed0']['a0']}; P5 beside {gd['p5_beside']['dsdv0']:.4f} / {gd['p5_beside']['dsda0']:.4f}.", "",
          "| seed | g_model (v0, a0) | delta-g (v0, a0) | delta-m m/s | target shift m/s | ckpt sha256 |", "|---|---|---|---|---|---|"]
    for s in ("0", "1"):
        d = fit["seeds"][s]
        md.append(f"| s{s} | {d['g_model_train_median']['dsdv0']:.4f}, {d['g_model_train_median']['dsda0']:.4f} | {d['delta_g']['dg_v']:+.4f}, {d['delta_g']['dg_a']:+.4f} | {d['delta_m_m_s']:+.4f} | {d['shift_target_m_s']:+.4f} | `{d['ckpt_sha256'][:16]}` |")
    if r1:
        md += ["", f"## READ 1 [M] (TRAIN fit windows, +-10 % of g_data at both seeds and both gains): **{r1['verdict']}**", "", "| seed | gain | C0 | edited | g_data | rel err | within |", "|---|---|---|---|---|---|---|"]
        for s in ("s0", "s1"):
            for r in r1["cells"][s]["rows"]:
                md.append(f"| {s} | {r['gain']} | {r['C0']:.4f} | {r['edited']:.4f} | {r['g_data']:.4f} | {r['rel_err']:.3f} | {r['within']} |")
        for s in ("s0", "s1"):
            c = r1["cells"][s]
            md.append(f"\n{s}: mean plan-speed shift vs C0 {c['mean_plan_speed_shift_vs_C0']:+.4f} m/s (target {c['shift_target']:+.4f}); edited mean plan speed {c['mean_edited_plan_speed']:.4f} vs realised mean {c['mean_realised_speed']:.4f}.")
        if r1e:
            md += ["", "READ 1 beside, eval-141 band windows (3-8 m/s, n %d; never deciding):" % r1e["n_windows"], "", "| seed | C0 (v0, a0) | edited (v0, a0) | rel err vs g_data (v0, a0) | mean plan-speed shift |", "|---|---|---|---|---|"]
            for s in ("s0", "s1"):
                c = r1e["cells"][s]
                md.append(f"| {s} | {c['C0_median_gain']['dsdv0']:.4f}, {c['C0_median_gain']['dsda0']:.4f} | {c['edited_median_gain']['dsdv0']:.4f}, {c['edited_median_gain']['dsda0']:.4f} | {c['rel_err_vs_g_data']['dsdv0']:.3f}, {c['rel_err_vs_g_data']['dsda0']:.3f} | {c['mean_plan_speed_shift_eval_m_s']:+.4f} |")
    if "read2" in out:
        D = out["read2"]
        md += ["", f"## READ 2 [M] T1-kin eval-141, E-GAIN-D vs C0 (RIC0) and vs P10: **{D['label']}**" + (f" ({D['note']})" if D.get("note") else ""), "",
               "M1 `share_closed` = -(D - C0)[M1] / (C0 - P10)[M1].", "", "| seed | M1 D-C0 [CI] | M1 C0-P10 | share_closed | n worse vs P10 (D / C0) | new separated-worse rows |", "|---|---|---|---|---|---|"]
        for s, c in D["seeds"].items():
            m = c["M1_WSP_minus_RIC0"]; mp = c["M1_RIC0_minus_P10"]
            md.append(f"| {s} | {f(m['delta'])} [{f(m['lo'])}, {f(m['hi'])}] | {f(mp['delta'])} | {c['share_closed']:.3f} | {c['n_worse_vs_P10']['WSP']} / {c['n_worse_vs_P10']['RIC0']} | {', '.join(c['new_separated_worse_rows']) or 'none'} |")
        md += ["", "Every row, both seeds (delta [CI] = D - C0; status: better / worse = CI excludes 0 in that direction, ns otherwise; vs P10 status in the last columns):", "",
               "| family | row | s0 D-C0 | s0 vs C0 | s0 vs P10 | s1 D-C0 | s1 vs C0 | s1 vs P10 |", "|---|---|---|---|---|---|---|---|"]
        for fam, labs in FAM.items():
            for lab in labs:
                cells = []
                for s in ("s0", "s1"):
                    c = D["seeds"][s]
                    row = c["rows_WSP_minus_RIC0"][lab]
                    sx = status(c["family_tables"]["WSP_minus_RIC0"], lab, row)
                    sp = status(c["family_tables"]["WSP_minus_P10"], lab, c["rows_WSP_minus_P10"][lab])
                    cells += [f"{f(row['delta'])} [{f(row['lo'])}, {f(row['hi'])}]", sx, sp]
                md.append(f"| {fam} | {lab} | " + " | ".join(cells) + " |")
        if "beside_banked_wsp_gain_alpha_1p0" in out:
            b = out["beside_banked_wsp_gain_alpha_1p0"]
            md += ["", f"Beside, banked WSP-GAIN alpha 1.0 (`wsp_edit/raw/READ2.json`): {b['label']}; M1 delta s0 {b['M1_delta']['s0']:+.4f}, s1 {b['M1_delta']['s1']:+.4f}; new separated-worse rows s0 {len(b['new_separated_worse_rows']['s0'])}, s1 {len(b['new_separated_worse_rows']['s1'])}."]
    if dk:
        md += ["", "## Distance keeping beside (dk141, T1-class open loop; see `raw/DK141_READ_D*.json`)", "", "Present: %d file(s). Table: `raw/DK141_TABLE_D.md`." % len(dk)]
    md += ["", "STRATEGIC family: PAUSED (PI item 25); its rows are listed beside and decide nothing. Raw: `raw/READ1_D.json`, `raw/READ1_D_EVAL.json`, `raw/READ2_D.json`. Card: `PREREG_WSP_GAIN_D.md` section 6."]
    open(HERE + "/RESULT_WSPD.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
    print("verdict:", out["verdict"], "| wrote RESULT_WSPD.md/.json")


if __name__ == "__main__":
    main()
