"""mk_readme.py: READ_DAG_HACC.md + MANIFEST_SHA256.txt from the raw JSONs of this package (no hand-typed number). Run after post_chain.sh ended (ZZPOST-DONE)."""
import hashlib
import json
import os

PK = "E:/Projects/TanitAD-artifacts/daghacc"
J = lambda p: json.load(open(f"{PK}/{p}", encoding="utf-8"))  # noqa: E731
R = J("raw/READ_DAG_HACC.json")
C = J("raw/CONTROLS.json")
O = R["outcome"]
fmt = lambda r: f"{r['delta']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}]"  # noqa: E731
L = []
A = L.append
A("# READ_DAG_HACC: PREREG_V8F_DAG_HACC executed as written (2026-10-27)\n")
A(f"**OUTCOME: {O['outcome']}**" + (f" ({O['note']})" if O.get("note") else "") + "\n")
A(f"* Committed next (PREREG section 4, verbatim): *\"{O['committed_next_verbatim']}\"*")
A(f"* Flags: KEEP at both seeds = {O['KEEP_both']}; D1 both = {O['D1_both']}; D2 both = {O['D2_both']}; D3 both = {O['D3_both']}; M1 separated-worse at >= 1 seed = {O['M1_separated_worse_at_ge1_seed']}; "
  f"a LONGITUDINAL row separated-worse at >= 1 seed = {O['LONGITUDINAL_row_separated_worse_at_ge1_seed']}; every registered control passes = {R['controls_all_registered_pass']}.")
A("* Evidence class: MEASURED (ours); every number is copied from `raw/` of this package. Tier: **T1-kin** (self-action OPEN loop; m = 10, causal bucket, dead-reckoned nav, logged scene): never closed loop, never \"drives\". "
  "Estimator: `taniteval.ci.paired_episode_cluster_bootstrap`, B = 2000, seed 0, per seed, over the 141 episodes (2,399 windows); it answers 'would another draw of EPISODES say this?' only. F = 0.1207 m, 2F = 0.2415 m. "
  "The planner is deterministic (no inference-seed question). Zero training; dev-box CPU.\n")
A("## 1. KEEP (DAG-HACC - DAG, ADE6, T1-kin): upper bound must be <= +0.0604 m at both seeds\n")
A("| seed | delta [lo, hi] | separated | DAG-HACC mean | DAG mean | n | bar (hi <=) | holds |\n|---|---|---|---|---|---|---|---|")
for s in ("s0", "s1"):
    k = R["KEEP"][s]
    A(f"| {s} | {fmt(k)} | {'yes' if k['separated'] else 'no'} | {k['mean_arm']:.4f} | {k['mean_DAG']:.4f} | {k['n']} | +{k['bar_hi']:.4f} | **{k['holds']}** |")
A("")
A("## 2. Controls\n")
S = C["SUMMARY"]
A("| control | result |\n|---|---|")
for k, v in S["each"].items():
    A(f"| {k} | {'PASS' if v else '**FAIL**'} |")
A(f"\nAll registered controls pass: **{S['all_registered_controls_pass']}**. Failed: {S['failed']}.")
A("* C-H1 (recipe identity): `v8f_hacc.fit_seed` on M2b's banked fit inputs: max |dw| vs HACC's banked weights = "
  + ", ".join(f"{s}: {C['C_H1_recipe_identity'][s]['max_abs_dw_vs_banked_HACC_weights']:.1e}, lambda* {C['C_H1_recipe_identity'][s]['lam_star']:.6g} == banked {C['C_H1_recipe_identity'][s]['banked_lam_star']:.6g}" for s in ("M2b_s0", "M2b_s1")) + ".")
A("* C-H3 (bake identity, w == 1): max |d| traj30 / traj60 on the first 24 eval windows = " + "; ".join(f"s{s}: {C['C_H3_bake_identity'][f's{s}']['first_24']['traj30']:.1f} / {C['C_H3_bake_identity'][f's{s}']['first_24']['traj60']:.1f} m" for s in (0, 1))
  + " (also 64 windows and through a saved + reloaded checkpoint).")
A("* C-H4 (bake scope): " + "; ".join(f"s{s}: extra keys {C['C_H4_bake_scope'][f's{s}']['extra_state_dict_keys']}, {C['C_H4_bake_scope'][f's{s}']['n_source_model_tensors']} source tensors, non-identical {len(C['C_H4_bake_scope'][f's{s}']['model_tensors_not_bit_identical'])}, "
                                  f"optimizer/step/cfg non-identical {len(C['C_H4_bake_scope'][f's{s}']['non_model_fields_not_identical (opt, step, cfg, ...)'])}" for s in (0, 1)) + ".")
A("* Added C-X1 (not in the PREREG): the plan-production route reproduces HBLEND's banked M2b `fit_windows` (windows, ages, GT identical; traj30 max |d| "
  + ", ".join(f"{k}: {v['traj30_maxabs']:.1e}" for k, v in C["C_X1_added_plan_route_reproduces_banked_M2b_fit_windows"].items()) + " m). My first-draft p30 tolerance (1e-6) was too tight for float32 CPU noise and read False; HBLEND's own bound (1e-4 m) is used and both are in `plans/CX1_route_reproduces_banked_M2b.json`.\n")
A("## 3. The fit (DAG's own T0 plans on HBLEND's W_fit: 141 train clips, 9,582 windows; fit code unchanged)\n")
A("| seed | lambda0 | m* | lambda* | rule branch | train accel (arm / P10) | train curvature | train ADE2 | train ADE6 | mean w along / cross |\n|---|---|---|---|---|---|---|---|---|---|")
import numpy as np  # noqa: E402
for s in (0, 1):
    f = J(f"fit/fit_DAG_s{s}.json")
    g = next(x for x in f["grid"] if x["m"] == f["m_star"])
    w = np.asarray(f["w_star"])
    A(f"| DAG_s{s} | {f['lambda0']:.1f} | {f['m_star']} | {f['lam_star']:.1f} | {f['rule_branch'][:44]} | {g['accel']:.4f} / {g['p_accel']:.4f} | {g['curv']:.5f} / {g['p_curv']:.5f} | {g['ade2']:.4f} / {g['p_ade2']:.4f} | {g['ade6']:.4f} / {g['p_ade6']:.4f} | {w[:, 0].mean():.3f} / {w[:, 1].mean():.3f} |")
A("\nD1 sensitivity (BESIDE, decides nothing): lambda0 from M2b's literal banked w instead of the HBLEND recipe on DAG's plans: " + "; ".join(
    f"s{s}: lambda* {J(f'fit/sensitivity_D1_DAG_s{s}.json')['alt_lam_star']:.0f} vs {J(f'fit/sensitivity_D1_DAG_s{s}.json')['primary_lam_star']:.0f}, max |dw| {J(f'fit/sensitivity_D1_DAG_s{s}.json')['max_abs_w_alt_vs_primary']:.1e}" for s in (0, 1)) + ".\n")
A("## 4. Read A: DAG-HACC - P10 at T1-kin (decisive). Right-hand column: plain DAG - P10 from the banked reread, for the side-by-side\n")
A(open(f"{PK}/raw/READ_A_tables_DAGHACC.md", encoding="utf-8").read())
A("Verdicts (rows D2 tests = every row of the table; M1 / L2 are the crawl rows):\n")
A("| seed | D1 (ADE6 sep-better, <= -2F) | multiple of F | D2 | D3 | separated-worse rows | LONGITUDINAL separated-worse | M1 sep-worse | L2 sep-worse |\n|---|---|---|---|---|---|---|---|---|")
for s in ("s0", "s1"):
    v = R["per_seed_verdict"][s]
    A(f"| {s} | {v['D1']} | {v['D1_multiple_of_F']:.1f} F | {v['D2']} | {v['D3']} | {v['separated_worse_rows']} | {v['LONGITUDINAL_separated_worse']} | {v['M1_separated_worse']} | {v['L2_separated_worse']} |")
A("")
A("## 5. Beside (never decisive)\n")
A("### 5.1 DAG-HACC - DAG on every Read-A row (T1-kin decisive; each row's move attributed)\n")
A("| row | s0 [lo, hi] | sep | s1 [lo, hi] | sep |\n|---|---|---|---|---|")
bd = R["beside"]["DAGHACC_minus_DAG_comps_rows"]
for lab in bd["s0"]:
    a_, b_ = bd["s0"][lab], bd["s1"][lab]
    A(f"| {lab} | {fmt(a_)} | {'yes' if a_['separated'] else 'no'} | {fmt(b_)} | {'yes' if b_['separated'] else 'no'} |")
bt = R["beside"]["DAGHACC_minus_DAG_tacdk_rows"]
for lab in bt["s0"]:
    a_, b_ = bt["s0"][lab], bt["s1"][lab]
    if a_.get("delta") is not None and b_.get("delta") is not None:
        A(f"| {lab} | {fmt(a_)} | {'yes' if a_['separated'] else 'no'} | {fmt(b_)} | {'yes' if b_['separated'] else 'no'} |")
A("\n### 5.2 STRICT (scene frozen, limit causal) and limit-logged, DAG-HACC - P10 verdicts\n")
for nm, key in (("STRICT", "STRICT_verdict"), ("limit logged (S-POS not computed)", "logged_limit_verdict")):
    for s in ("s0", "s1"):
        v = R["beside"][key][s]
        A(f"* {nm} {s}: D1 {v['D1']}, D2 {v['D2']}, D3 {v['D3']}; separated-worse {v['separated_worse_rows']}; not computed {v['not_computed']}")
A("\nTables: `raw/READ_A_beside_tables_DAGHACC.md`.\n")
A("### 5.3 STRATEGIC S-POS@30/50/80 against CTRV (within run, unpaired)\n")
sp = J("raw/spos_daghacc.json")["runs"]
dg = json.load(open("E:/Projects/TanitAD-artifacts/v8f-dagger-t1/read/DAG_READ.json", encoding="utf-8"))
K = "S-POS vs CTRV (ticks>=1, within this run)"
A("| run | S-POS@30 | S-POS@50 | S-POS@80 |\n|---|---|---|---|")
for s in (0, 1):
    for lab in (f"DAGHACC_s{s}_decisive", f"DAGHACCSTRICT_s{s}_beside"):
        r = sp.get(lab, {})
        if "S-POS" in r:
            A(f"| {lab} | " + " | ".join(f"{r['S-POS'][k]['delta']:+.2f} [{r['S-POS'][k]['lo']:+.2f}, {r['S-POS'][k]['hi']:+.2f}]" for k in ("S-POS@30", "S-POS@50", "S-POS@80")) + " |")
        else:
            A(f"| {lab} | NOT COMPUTED: {r.get('reason')} | | |")
    d = dg["beside"]["per_seed"][f"s{s}"]["per_run_t1heads"]["DAG"][K]
    A(f"| DAG_s{s} (plain, DAG_READ.json) | " + " | ".join(f"{d[k]['delta']:+.2f} [{d[k]['lo']:+.2f}, {d[k]['hi']:+.2f}]" for k in ("S-POS@30", "S-POS@50", "S-POS@80")) + " |")
A("\n### 5.4 The four families at T0 (fresh ages, eval-141, T1-class open loop): DAG-HACC - DAG\n")
if os.path.exists(f"{PK}/raw/t0_beside_DAGHACC_minus_DAG.json"):
    t0 = J("raw/t0_beside_DAGHACC_minus_DAG.json")
    A("| row | s0 [lo, hi] | sep | s1 [lo, hi] | sep |\n|---|---|---|---|---|")
    for k in ("ade1", "ade2", "ade6", "fde6", "fw_lon_speed_mae", "fw_lon_accel_mae", "fw_lon_along_mae", "fw_lat_cross_mae", "fw_lat_heading_mae_deg", "fw_lat_yaw_rate_mae_degps", "fw_lat_curvature_mae_1pm",
              "fw_tac_lat_acc", "fw_tac_lon_acc", "consistency_viol_any"):
        a_, b_ = t0["seeds"]["s0"]["rows_DAGHACC_minus_DAG"][k], t0["seeds"]["s1"]["rows_DAGHACC_minus_DAG"][k]
        A(f"| {k} | {a_['mean']:+.4f} [{a_['lo']:+.4f}, {a_['hi']:+.4f}] | {'yes' if a_['separated'] else 'no'} | {b_['mean']:+.4f} [{b_['lo']:+.4f}, {b_['hi']:+.4f}] | {'yes' if b_['separated'] else 'no'} |")
    A("\nTACTICAL decision heads and STRATEGIC checkpoint outputs vs plain DAG, max |diff| (post-map touches the plan only): " + "; ".join(f"{s}: {max(v['tactical_and_strategic_outputs_max_abs_diff_vs_DAG'].values()):.2e}" for s, v in t0["seeds"].items()) + ".")
else:
    A("NOT COMPUTED (t0_beside did not finish; see logs/t0_daghacc.done).")
A("\n## 6. Declarations, deviations, disclosures\n")
A(open(f"{PK}/DECLARATIONS.md", encoding="utf-8").read().split("\n", 3)[3])
A("* **The fit-set cache is not in the repo.** It was read in place (read-only) from the previous HBLEND package in the Master Mind scratchpad "
  "(`scratchpad/hblend/cache/train/clips`, 141 clips, 2.8 GB; `box2/join_train.npz`, 34.7 MB, sha256 `978f41bf...`): it should be banked (e.g. to E: or box 2) before that scratchpad is cleaned.")
A("* HBLEND's weights JSON blobs in the repo are LF; the pinned sha256 in `v8f_hacc.py` is of the CRLF original. The LF blob converted to CRLF reproduces the pinned sha256 for both seeds (checked) and was used.")
A("* All code runs from `code/` copies of tip 2a416ca7 with only scratchpad path constants re-pointed (`code/PATCHES.md`).")
open(f"{PK}/READ_DAG_HACC.md", "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")
# manifest
lines = []
for root, dirs, files in os.walk(PK):
    dirs[:] = [d for d in dirs if d not in ("tip", "pt_tmp", "src", "__pycache__")]
    for f in sorted(files):
        p = os.path.join(root, f)
        if os.path.getsize(p) == 0:
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""):
                h.update(b)
        lines.append(f"{h.hexdigest()}  {os.path.relpath(p, PK).replace(os.sep, '/')}  {os.path.getsize(p)}")
open(f"{PK}/MANIFEST_SHA256.txt", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
print("ZZREADME-DONE", len(lines), "files")
