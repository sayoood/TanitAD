"""fam4_read.py -- paired four-family read, arm L_s0 minus BPRIME_s0, T1-kin first tick eval-141 (diagnostic, never driving performance).
Everything numeric is IMPORTED from the frozen read code (stopbal_read, stopperc_s1_read, stopbal_geom); nothing is copy-modified."""
import os, sys, json, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1"); os.environ.setdefault("OMP_NUM_THREADS", "2")
TH = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory"
for p in (TH + "/stopperc/stage1_A/read", TH + "/stopbal/read", TH + "/stopbal/dose2"):
    sys.path.insert(0, p)
sys.dont_write_bytecode = True
import numpy as np
import stopbal_read as SR
import stopperc_s1_read as S1

OUT = "F:/TanitAD-scratch/s0l_read/fam4"
T1 = "F:/TanitAD-scratch/s0l_read/t1"
SD = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/stopperc/s0l/out/stopdec_L"
REF = "E:/Projects/TanitAD-artifacts/v8f-dagoacc-t1/dump_DAGOACC_s0_m10_causal_heads"
NBOOT = 2000
FAM = dict(SR.FAMILY)
FAM.update({"TAC-LAT": "TACTICAL", "TAC-LON": "TACTICAL"})
SPEC = {t: {"dump": f"{T1}/dump_{t}_m10_causal_heads", "stopdec": f"{SD}/stopdec_{t}.npz"} for t in ("L_s0", "BPRIME_s0")}


def cell_json(d):
    d = {k: (v.item() if isinstance(v, (np.floating, np.integer, np.bool_)) else v) for k, v in d.items()}
    for k in ("gain", "cost", "above_floor", "floor"):
        d.pop(k, None)
    return d


def read_pair(CI, a, b, eid, sp, ok, nb):
    sets = {"all": np.ones(len(eid), bool), "sp": sp & ok}
    cells = {}
    for sname in ("all", "sp"):
        for cname in a["cells"]:
            r = cell_json(SR.cell_delta(CI, a["cells"][cname], b["cells"][cname], eid, sets[sname], cname in SR.HIGHER_BETTER, None, nb))
            if r["status"] == "OK":
                better = (r["delta"] > 0) if cname in SR.HIGHER_BETTER else (r["delta"] < 0)
                r["side"] = "L better" if better else ("L worse" if r["delta"] != 0 else "equal")
            band, m = cname.split(":") if ":" in cname else ("tactical_decision", cname)
            r.update(set=sname, band=band, metric=m, family=FAM.get(m, "?"), higher_is_better=cname in SR.HIGHER_BETTER)
            cells[f"{sname}|{cname}"] = r
    return cells


def cue_cells(CI, a, b, eid, sp, ok, nb):
    ca, cb = a["cue"], b["cue"]
    out = {}
    out["cue|auroc p(STOP_POINT) vs stop_hit (has)"] = S1.auc_cell(CI, ca["goal_p"], cb["goal_p"], ca["pos"], eid, ca["has"], None, nb, True)
    out["cue|auroc lon-STOP vs regime stop (has)"] = S1.auc_cell(CI, ca["lon_p"], cb["lon_p"], ca["reg_stop"], eid, ca["has"], None, nb, True)
    ga, gb = np.asarray(a["goal_stop"], np.float64), np.asarray(b["goal_stop"], np.float64)
    out["cue|goal-stop decision hit rate (stop-point windows, higher better)"] = SR.cell_delta(CI, ga, gb, eid, ok & sp, True, None, nb)
    out["cue|goal-stop decision fpr (other has-and-present, lower better)"] = SR.cell_delta(CI, ga, gb, eid, ok & ~sp, False, None, nb)
    for k, v in list(out.items()):
        v = cell_json(v)
        if v.get("status") == "OK" and "side" not in v:
            hb = "higher better" in k or "auroc" in k
            better = (v["delta"] > 0) if hb else (v["delta"] < 0)
            v["side"] = "L better" if better else ("L worse" if v["delta"] != 0 else "equal")
        out[k] = v
    return out


def main():
    t0 = time.time()
    CI = SR.load_ci()
    ref = SR.load_dump(REF)
    geom = SR.build_geom(ref); G = geom["G"]
    geom["glat"], geom["glon"] = G.labels(ref["g"]); geom["J"] = G.join(ref)
    cache, data = {}, {}
    for tag, spec in SPEC.items():
        arm, sd, ctl = SR.arm_from_files(spec, ref, geom, cache)
        arm["cue"] = S1.cue_arrays(sd)
        data[tag] = (arm, sd, ctl)
    eid = ref["eid"]
    ok_a, sp_a = SR.window_sets(data["L_s0"][1]); ok_b, sp_b = SR.window_sets(data["BPRIME_s0"][1])
    assert np.array_equal(ok_a, ok_b) and np.array_equal(sp_a, sp_b), "window sets differ between arms"
    ok, sp = ok_a, sp_a
    L, B = data["L_s0"][0], data["BPRIME_s0"][0]
    has = np.asarray(data["L_s0"][1]["has"], bool)
    res = {"tool": "fam4_read.py", "pair": "a = L_s0, b = BPRIME_s0; delta = a - b (errors lower better; TAC accuracies, AUROC, hit rate higher better)",
           "tier": S1.TIER, "evidence_class": "MEASURED (ours; existing dumps, zero training)",
           "estimator": f"taniteval.ci.paired_episode_cluster_bootstrap via stopbal_read.cell_delta, B={NBOOT}, seed 0, ci.py from {CI.SOURCE}; AUROC via paired_stat_cluster_bootstrap (stopperc_s1_read.auc_cell). Answers another draw of EPISODES only; ONE training seed per arm, NO replicate floor available for this pair; separated is necessary, not sufficient; no cell is a lever effect.",
           "STRATEGIC": "PAUSED (PI item 25); never a pass, never omitted",
           "window_sets": {"all": "every eval window with a finite value in the cell", "sp": "stop-point windows (strat_tlstop==STOP_POINT, has, present speed cell)"},
           "counts": {"n_windows": int(len(eid)), "n_has": int(has.sum()), "n_ok": int(ok.sum()), "n_stop_point": int(sp.sum()),
                      "n_episodes_has": int(len(np.unique(eid[has]))), "n_episodes_all": int(len(np.unique(eid)))},
           "not_computed": "none: distance-keeping cells (dk_hw / dk_tg) are computed from the stopbal_geom METRICS and are NA only on windows without a lead in the corridor (n shown per cell)"}
    cells = read_pair(CI, L, B, eid, sp, ok, NBOOT)
    cue = cue_cells(CI, L, B, eid, sp, ok, NBOOT)
    res["cells"], res["cue_cells"] = cells, cue
    print("main read", time.time() - t0, flush=True)
    ctl = {}
    selfc = read_pair(CI, B, B, eid, sp, ok, NBOOT)
    selfc.update(cue_cells(CI, B, B, eid, sp, ok, NBOOT))
    bad = []
    for k, v in selfc.items():
        if v.get("status") != "OK":
            bad.append((k, v.get("status"))); continue
        if not (v["delta"] == 0.0 and v["lo"] == 0.0 and v["hi"] == 0.0):
            bad.append((k, v["delta"], v["lo"], v["hi"]))
    oks = [abs(v["delta"]) for v in selfc.values() if v.get("status") == "OK"]
    ctl["a_self_BPRIME_vs_BPRIME"] = {"n_cells": len(selfc), "n_exact_zero_ci00": len(selfc) - len(bad), "violations": bad, "pass": not bad, "max_abs_delta": max(oks) if oks else None}
    c = res["counts"]
    ctl["b_window_counts"] = {"expected_S0L": {"n_has": 2365, "n_episodes_has": 139}, "measured": {"n_has": c["n_has"], "n_episodes_has": c["n_episodes_has"]},
                              "auroc_cell_n": {k: v.get("n_windows") for k, v in cue.items() if "auroc" in k},
                              "auroc_cell_episodes": {k: v.get("n_episodes") for k, v in cue.items() if "auroc" in k},
                              "pass": c["n_has"] == 2365 and c["n_episodes_has"] == 139}
    SH = 0.5
    shifted = L["cells"]["tactical_2_6s:ade_m"] + SH
    rr = SR.cell_delta(CI, shifted, L["cells"]["tactical_2_6s:ade_m"], eid, np.ones(len(eid), bool), False, None, NBOOT)
    ctl["c_deliberate_regression"] = {"shift": f"+{SH} m added to L_s0 tactical ADE on every window (in memory, discarded)", "expected": "delta +0.5 exactly, worse side, CI excludes 0",
                                      "measured_delta": rr["delta"], "lo": rr["lo"], "hi": rr["hi"], "separated": rr["separated"], "pass": abs(rr["delta"] - SH) < 1e-9 and rr["lo"] > 0}
    ctl["instrument_consistency"] = {t: data[t][2] for t in data}
    res["controls"] = ctl
    res["controls_pass"] = all(ctl[k]["pass"] for k in ("a_self_BPRIME_vs_BPRIME", "b_window_counts", "c_deliberate_regression"))
    res["wall_s"] = round(time.time() - t0, 1)
    json.dump(res, open(f"{OUT}/RESULT_FAM4_L_vs_BPRIME.json", "w"), indent=1, default=float)
    print("WROTE json | controls_pass", res["controls_pass"], res["wall_s"], "s", flush=True)

    def f(x, n=4):
        return "n/a" if x is None else f"{x:.{n}f}"
    lines = ["# FAM4: L_s0 minus BPRIME_s0 (T1-kin, first tick, eval-141; a diagnostic, never driving performance)", "",
             f"Counts: {c['n_windows']} windows, {c['n_has']} with has, {c['n_ok']} ok, {c['n_stop_point']} stop-point windows, {c['n_episodes_has']} episodes (has).",
             "delta = L_s0 - BPRIME_s0; errors lower-better, TAC accuracies / AUROC / hit higher-better. CI = paired episode-cluster bootstrap, B 2,000, seed 0 (another draw of EPISODES only). "
             "One training seed per arm: no replicate floor available for this pair. separated is necessary, not sufficient; no cell is a lever effect.", "",
             f"Controls: self-pair exact zero {ctl['a_self_BPRIME_vs_BPRIME']['pass']} ({ctl['a_self_BPRIME_vs_BPRIME']['n_cells']} cells); window counts {ctl['b_window_counts']['pass']}; deliberate regression {ctl['c_deliberate_regression']['pass']}.", ""]

    def table(title, rows):
        lines.extend([f"## {title}", "", "| cell | set | delta | CI | separated | side | n win | n ep |", "|---|---|---|---|---|---|---|---|"])
        for k, v in rows:
            if v.get("status") != "OK":
                lines.append(f"| {k} | {v.get('set', '')} | NOT-APPLICABLE | | | | {v.get('n_windows')} | |"); continue
            lines.append(f"| {k} | {v.get('set', '')} | {f(v['delta'])} | [{f(v['lo'])}, {f(v['hi'])}] | {'yes' if v['separated'] else 'no'} | {v.get('side', '')} | {v['n_windows']} | {v['n_episodes']} |")
        lines.append("")
    for fam in ("LONGITUDINAL", "LATERAL", "TACTICAL", "PRIMARY"):
        rows = [(f"{v['band']} {v['metric']}", v) for v in cells.values() if v["family"] == fam]
        rows.sort(key=lambda kv: (kv[1]["set"] != "all", kv[0]))
        if fam == "TACTICAL":
            rows += list(cue.items())
        table(fam + (" (ADE beside)" if fam == "PRIMARY" else ""), rows)
    lines += ["## STRATEGIC", "", "PAUSED (PI item 25); not a pass.", ""]
    sep = [(v["set"], v["band"], v["metric"], v["family"], round(v["delta"], 5), v["side"]) for v in cells.values() if v.get("status") == "OK" and v["separated"]]
    lines.append("SEPARATED_LIST: " + json.dumps(sep))
    open(f"{OUT}/FAM4_L_vs_BPRIME.md", "w", encoding="utf-8").write("\n".join(lines))
    print("WROTE md")


main()
