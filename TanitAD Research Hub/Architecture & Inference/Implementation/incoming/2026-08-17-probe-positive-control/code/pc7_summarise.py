"""PC7 — roll every banked JSON into ONE summary + the rendered tables.

⛔ The tables in the report are GENERATED from this, never transcribed by hand.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

W = Path(sys.argv[1])
OUT = Path(sys.argv[2])
RAW = W / "raw"


def load(p):
    try:
        return json.loads(Path(p).read_text("utf-8"))
    except Exception:
        return None


S: dict = {"_evidence_class": "MEASURED (ours) unless a row says otherwise",
           "eval_tier": "T0-DIAGNOSTIC",
           "estimator": "taniteval.ci.paired_episode_cluster_bootstrap",
           "forbidden": "overlapping_holdout_se",
           "sp2_probe_md5": "aabbee36fce5f164d47a555fad369cbd "
                            "(byte-identical to the parity run's)",
           "slot_probe_arms": {}, "seed_spread": {}, "ridge": {},
           "readout_rules": {}, "strata": {}}

# ---- slot-probe arms (mine) + the parity run's, for one table ---------------
PARITY = Path("C:/Users/Admin/AppData/Local/Temp/claude/"
              "G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD/"
              "8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2/scratchpad/sp2/raw")
for p in sorted(glob.glob(str(RAW / "results_*.json"))) + \
        [str(PARITY / f) for f in ("results_s11250.json", "results_s09000.json",
                                   "results_s02000.json",
                                   "results_NULLMATCHED.json",
                                   "results_s11250_seed1.json",
                                   "results_s11250_seed2.json")]:
    d = load(p)
    if not d or "per_arm" not in d:
        continue
    a = d["per_arm"].get("cells")
    if not a:
        continue
    m = a["lead_gap_abs_err_m"]
    k1 = d["paired"].get("cells vs C-CONST", {})
    k5 = d["paired"].get("cells vs C-EPMEAN", {})
    k2 = d["paired"].get("cells vs cells__C-SHUF", {})
    S["slot_probe_arms"][Path(p).stem] = {
        "run_stamp": d["run_stamp"],
        "source": "THIS RUN" if str(RAW) in p else "parity run (cited)",
        "err_m": m["mean"], "ci": [m["lo"], m["hi"]],
        "median_m": round(a["median_abs_err_m"], 4),
        "pred_mean_m": round(a["mean_pred_gap_m"], 3),
        "gt_mean_m": round(a["mean_gt_gap_m"], 3),
        "n_windows": d["n_scored_windows"],
        "n_clusters": d["n_bootstrap_clusters"],
        "K1_delta": k1.get("delta"), "K1_lo": k1.get("lo"),
        "K1_hi": k1.get("hi"), "K1_separated": k1.get("separated"),
        "K1_PASSES": bool(k1.get("separated") and (k1.get("delta") or 0) < 0),
        "K5_delta": k5.get("delta"), "K5_separated": k5.get("separated"),
        "K5_PASSES": bool(k5.get("separated") and (k5.get("delta") or 0) < 0),
        "K2_delta": k2.get("delta"), "K2_separated": k2.get("separated"),
        "K2_PASSES": bool(k2.get("separated") and (k2.get("delta") or 0) < 0),
        "oracle_slot_median_m": round(
            a["_diag_oracle_slot_abs_err_m"]["median"], 4),
    }

# ---- seed spread, per condition --------------------------------------------
for cond, keys in (("GT-ORACLE-CELLS (this run)",
                    ["results_orc010_seed0", "results_orc010_seed1",
                     "results_orc010_seed2"]),
                   ("GT-ORACLE-DIRECT (this run)",
                    ["results_orcdir_seed0", "results_orcdir_seed1",
                     "results_orcdir_seed2"]),
                   ("v6F@11250 (parity run, cited)",
                    ["results_s11250", "results_s11250_seed1",
                     "results_s11250_seed2"])):
    vals = [S["slot_probe_arms"][k] for k in keys if k in S["slot_probe_arms"]]
    if not vals:
        continue
    e = [v["err_m"] for v in vals]
    k = [v["K1_delta"] for v in vals if v["K1_delta"] is not None]
    # ⚠️ A RANGE NEEDS >=3 POINTS BEFORE IT IS A RANGE. This run drew a
    # spread claim from n=2 and the third seed refuted it (report §2.5), so the
    # renderer now REFUSES to print a range below 3 seeds rather than printing a
    # number a reader would weigh.
    ok3 = len(vals) >= 3
    S["seed_spread"][cond] = {
        "n_seeds": len(vals), "err_m": e,
        "err_range": (round(max(e) - min(e), 4) if ok3 else None),
        "K1": k,
        "K1_range": (round(max(k) - min(k), 4) if (ok3 and k) else None),
        "_range_note": (None if ok3 else
                        f"NO RANGE REPORTED - only {len(vals)} seed(s); a range "
                        f"needs >=3 points"),
        "all_fail_K1": all(not v["K1_PASSES"] for v in vals),
        "all_pass_K1": all(v["K1_PASSES"] for v in vals)}

for p in sorted(glob.glob(str(RAW / "pc6_ridge_*.json"))):
    d = load(p)
    S["ridge"][d["arm"]] = {k: d[k] for k in
                            ("ridge_err_m", "ridge_ci", "c_const_err_m",
                             "K1_delta", "K1_lo", "K1_hi", "K1_PASSES",
                             "K5_PASSES", "corr_pred_gt", "n_eval_windows",
                             "n_eval_clusters", "alpha_chosen")}
for p in sorted(glob.glob(str(RAW / "pc5_rules_*.json"))):
    d = load(p)
    S["readout_rules"][d["arm"]] = {
        k: {kk: v.get(kk) for kk in ("n_windows", "err_m", "K1_delta",
                                     "K1_lo", "K1_hi", "K1_PASSES",
                                     "corr_pred_gt")}
        for k, v in d["rules"].items() if "err_m" in v}
for p in sorted(glob.glob(str(RAW / "pc3_strata_*.json"))):
    d = load(p)
    S["strata"][d["arm"]] = {"lead_geometry": d["lead_geometry"],
                             "range_vs_C-CONST":
                                 d["strata"]["range"]["vs_C-CONST"],
                             "px_vs_C-CONST":
                                 d["strata"]["apparent_px_width"]["vs_C-CONST"]}
g = load(RAW / "pc4_grad.json") or load(RAW / "pc4_grad_smoke.json")
if g:
    S["grad_cosines"] = {
        "_evidence_class": g["_evidence_class"],
        "run_stamp": g["run_stamp"], "o1_k": g["o1_k"], "o5_k": g["o5_k"],
        "live_run_o5_k": g.get("live_run_o5_k"),
        "chance_cos_trunk": g["chance_cos_trunk"],
        "n_params_trunk": g["n_params_trunk"],
        "controls": g.get("controls"),
        "o4_is_not_a_loss_term": g["o4_is_not_a_loss_term"],
        "per_seed": {s: {lv: {"cos_vs_agent": r["cos_vs_AGENT_READOUT_TRUNK"],
                              "cos_vs_rest": r["cos_term_vs_rest_TRUNK"],
                              "rel_pull": r["relative_pull_TRUNK"]}
                         for lv, r in v["levers"].items()}
                     for s, v in g["seeds"].items()},
        "cos_matrix_trunk": {s: v["cos_matrix_trunk"]
                             for s, v in g["seeds"].items()},
        "cos_full_vs_agent": {s: v.get("cos_full_vs_AGENT_TRUNK")
                              for s, v in g["seeds"].items()},
        "o4_sampler": g.get("o4_sampler")}

OUT.write_text(json.dumps(S, indent=1), "utf-8")
print(json.dumps({"slot_arms": len(S["slot_probe_arms"]),
                  "ridge": len(S["ridge"]),
                  "rules": len(S["readout_rules"]),
                  "strata": len(S["strata"]),
                  "grad": "grad_cosines" in S}, indent=1))

# ---- rendered tables, GENERATED (never transcribed) -------------------------
L = ["# RENDERED TABLES — generated by `code/pc7_summarise.py` from the banked JSON",
     "", "⛔ Do not hand-edit. Regenerate.", "",
     "## A. Slot-probe arms — `lead_gap_abs_err_m` (C-CONST 5.133 · C-EPMEAN 3.122)", "",
     "| arm | source | err (m) | CI | K1 Δ | K1 CI | K1 sep | K1 PASS | K2 Δ | K2 sep | pred mean | n win / clust |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for k, v in sorted(S["slot_probe_arms"].items(),
                   key=lambda kv: (kv[1]["source"], kv[0])):
    L.append("| `%s` | %s | **%.4f** | [%.3f, %.3f] | **%+.4f** | [%+.3f, %+.3f] | %s | %s | %+.4f | %s | %.2f | %d / %d |"
             % (v["run_stamp"], v["source"], v["err_m"], v["ci"][0], v["ci"][1],
                v["K1_delta"], v["K1_lo"], v["K1_hi"],
                "yes" if v["K1_separated"] else "no",
                "✅" if v["K1_PASSES"] else "⛔",
                v["K2_delta"], "yes" if v["K2_separated"] else "no",
                v["pred_mean_m"], v["n_windows"], v["n_clusters"]))
L += ["", "## B. Seed spread per condition (one frozen cache, only the optimiser seed differs)", "",
      "| condition | seeds | err (m) | K1 | K1 range | all fail K1 |", "|---|---|---|---|---|---|"]
for k, v in S["seed_spread"].items():
    L.append("| %s | %d | %s | %s | **%s** | %s |"
             % (k, v["n_seeds"], " · ".join(f"{x:.3f}" for x in v["err_m"]),
                " · ".join(f"{x:+.3f}" for x in v["K1"]),
                (v["K1_range"] if v["K1_range"] is not None
                 else "n/a (n<3 seeds)"),
                "yes" if v["all_fail_K1"] else "NO"))
L += ["", "## C. Ridge readout — a DIFFERENT instrument (never comparable to A)", "",
      "| memory | ridge err (m) | CI | C-CONST | K1 Δ | K1 CI | K1 PASS | corr(pred,GT) |",
      "|---|---|---|---|---|---|---|---|"]
for k, v in S["ridge"].items():
    L.append("| %s | **%.3f** | [%.3f, %.3f] | %.3f | %+.3f | [%+.3f, %+.3f] | %s | **%+.3f** |"
             % (k, v["ridge_err_m"], v["ridge_ci"][0], v["ridge_ci"][1],
                v["c_const_err_m"], v["K1_delta"], v["K1_lo"], v["K1_hi"],
                "✅" if v["K1_PASSES"] else "⛔", v["corr_pred_gt"]))
L += ["", "## D. Readout rules on the BANKED heads (R0 = the incumbent, reproduces the headline)", "",
      "| arm | rule | n | err (m) | K1 Δ | K1 PASS | corr |", "|---|---|---|---|---|---|---|"]
for arm, rules in S["readout_rules"].items():
    for rk, rv in rules.items():
        L.append("| %s | `%s` | %d | %.3f | %+.3f | %s | %+.3f |"
                 % (arm, rk, rv["n_windows"], rv["err_m"], rv["K1_delta"],
                    "✅" if rv["K1_PASSES"] else "⛔", rv["corr_pred_gt"]))
L += ["", "## E. Stratification — K1 vs C-CONST per stratum", "",
      "| arm | stratum | n win | n clust | arm (m) | C-CONST (m) | K1 Δ | CI | sep | pred mean ± sd | GT mean |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for arm, d in S["strata"].items():
    for fam in ("range_vs_C-CONST", "px_vs_C-CONST"):
        for r in d[fam]:
            if r.get("n_windows", 0) == 0:
                continue
            L.append("| %s | %s | %d | %d | %.3f | %.3f | %+.3f | [%+.3f, %+.3f] | %s | %s ± %s | %s |"
                     % (arm, r["stratum"], r["n_windows"], r["n_clusters"],
                        r["arm_err_m"], r["ctl_err_m"], r["K1_delta"],
                        r["K1_lo"], r["K1_hi"],
                        "yes" if r["K1_separated"] else "no",
                        r.get("pred_mean_m", "—"), r.get("pred_sd_m", "—"),
                        r.get("gt_mean_m", "—")))
(OUT.parent / "RENDER_TABLES.md").write_text("\n".join(L) + "\n", "utf-8")
print("wrote", OUT.parent / "RENDER_TABLES.md")
