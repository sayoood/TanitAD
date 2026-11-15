"""make_verdict.py -- writes VERDICT_MOTION_TARGET_FINAL.md and GAC_ROW_DRAFT.md from RESULT_MOTION_TARGET_FINAL.json and the family reads.
Adds the names and the family tables to the JSON. CPU, reads files, writes only into its own folder."""
import json
import numpy as np

OUT = "F:/TanitAD-scratch/v8proof/spatial/q2/final"
R = json.load(open(OUT + "/RESULT_MOTION_TARGET_FINAL.json", encoding="utf-8"))
L = R["lead_read"]; ST = R["step_read"]; P = R["position_read"]
F2 = json.load(open(OUT + "/out_q3_twoseed/RESULT_Q3.json", encoding="utf-8"))
F1 = json.load(open(OUT + "/out_q3_oneseed/RESULT_Q3.json", encoding="utf-8"))
SPREAD = P["spread_six"]["gap R2 (state)"]["range"]  # 0.0450
SPREAD3 = P["spread_three_banked_C2"]["gap R2 (state)"]["range"]


def c(x, k=3):
    if "diff" in x:
        v = x["diff"]
    elif "skill" in x:
        v = x["skill"]
    elif "diff_of_ratios" in x:
        v = x["diff_of_ratios"]
    else:
        v = x["r2"]
    lo, hi = x["ci95"]
    return ("%+." + str(k) + "f [%+." + str(k) + "f, %+." + str(k) + "f]") % (v, lo, hi)


def ratio(t):
    return "%.3f [%.3f, %.3f]" % tuple(t)


def exc_pos(x):
    return x["ci95"][0] > 0


# ---- names
names = {}
for arm, v in L.items():
    n = {}
    rc, rf, vb = v["rollout_minus_copy"], v["rollout_minus_floor"], v["rollout_minus_base_rollout_paired"]
    card = "ADD19" if arm.startswith("PE") or arm.startswith("PF") else "ADD18"
    if card == "ADD18":
        if exc_pos(rc) and exc_pos(rf):
            n["I"] = "MOTION-IMAGINES"
        elif exc_pos(vb):
            n["I"] = "MOTION-BEATS-BASE"
        else:
            n["I"] = "MOTION-FLAT"
        n["V"] = "MOTION-MOVES" if v["change_skill_rollout"]["ci95"][0] > 0 else "MOTION-STILL"
        n["S"] = "STEP-BELOW-COPY" if ST["per_arm"][arm]["ratio_ctx1"][2] < 1 else "STEP-NOT-BELOW"
    else:
        # per-seed level of the TARGET names (a name needs both seeds)
        if exc_pos(rc):
            n["I"] = "TARGET-IMAGINES"
        elif exc_pos(vb):
            n["I"] = "TARGET-BEATS-BASE"
        else:
            n["I"] = "TARGET-FLAT"
        n["V"] = "TARGET-MOVES" if v["change_skill_rollout"]["ci95"][0] > 0 else "TARGET-STILL"
        sp = ST["paired_vs_comparison"]["%s_minus_%s" % (arm, v["comparison_arm"])]
        n["S"] = "STEP-IMPROVES" if sp["ci95"][1] < 0 else "STEP-SAME"
    names[arm] = n
R["names_per_seed"] = names

pairs_R = P["pairs"]


def rname(key):
    d = pairs_R[key]["gap_R2_diff"]
    lose = d["ci95"][1] < 0 and abs(d["diff"]) > SPREAD
    return "LOSES-POSITION" if lose else "KEEPS-POSITION"


R["R_names_per_seed"] = {k: rname(k) for k in pairs_R}
R["R_names_per_seed_spread_three"] = {k: ("LOSES-POSITION" if (v["gap_R2_diff"]["ci95"][1] < 0 and abs(v["gap_R2_diff"]["diff"]) > SPREAD3) else "KEEPS-POSITION") for k, v in pairs_R.items()}
sp2 = P["spread_two"]["gap R2 (state)"]["range"]
R["R_names_per_seed_spread_two"] = {k: ("LOSES-POSITION" if (v["gap_R2_diff"]["ci95"][1] < 0 and abs(v["gap_R2_diff"]["diff"]) > sp2) else "KEEPS-POSITION") for k, v in pairs_R.items()}

# ---- family cells
CELLS = ["LONGITUDINAL:speed_mae_mps", "LONGITUDINAL:along_mae_m", "LATERAL:cross_mae_m", "LATERAL:heading_mae_deg", "LATERAL:curvature_mae_1pm", "LATERAL:yaw_rate_mae_degps",
         "TACTICAL:lon_err_rate", "TACTICAL:lat_err_rate", "TACTICAL:joint_err_rate", "TACTICAL:ade_tac_m"]


def fam_table(res, pair):
    p = res["pairs"][pair]
    rows = ["| cell (all windows, error, below 0 better) | per seed | D | two-level interval | rule |", "|---|---|---|---|---|"]
    for k in CELLS:
        f = p["families"][k]
        rows.append("| %s | %s | %+.4f | [%+.4f, %+.4f] | %s |" % (k, " / ".join("%+.4f" % x for x in f["d_s"]), f["D"], f["A_lo"], f["A_hi"], f["verdict"]))
    return p["Q3"]["name"], "\n".join(rows)


famN = {}
for res, pr in ((F2, ["PE_minus_P", "PE_minus_HIER", "PR_minus_P", "PR_minus_G", "PR_minus_HIER"]), (F1, ["PF_minus_P", "PF_minus_HIER", "PF_minus_PR", "RES_minus_G", "RES_minus_HIER", "RR_minus_G", "RR_minus_HIER"])):
    for pair in pr:
        famN[pair] = fam_table(res, pair)
R["family_names"] = {k: v[0] for k, v in famN.items()}
R["family_files"] = [OUT + "/out_q3_twoseed/RESULT_Q3.json", OUT + "/out_q3_oneseed/RESULT_Q3.json"]
R["family_controls"] = {"twoseed": F2["controls"], "oneseed": F1["controls"]}
R["names_two_seed"] = {
    "PR": {"S": "STEP-BELOW-COPY (both seeds)", "I": "MOTION-FLAT (both seeds)", "V": "MOTION-STILL (both seeds)", "R": "KEEPS-POSITION", "F": F2["pairs"]["PR_minus_G"]["Q3"]["name"]},
    "PE": {"S": "STEP-SAME", "I": "TARGET-FLAT", "V": "TARGET-STILL", "R": "KEEPS-POSITION", "F": F2["pairs"]["PE_minus_P"]["Q3"]["name"]},
}
json.dump(R, open(OUT + "/RESULT_MOTION_TARGET_FINAL.json", "w", encoding="utf-8"), indent=1, default=float)


def lead_rows(arms):
    out = []
    hdr = "| read | " + " | ".join(arms) + " |"
    out += [hdr, "|---|" + "---|" * len(arms)]
    def row(label, fn):
        out.append("| %s | %s |" % (label, " | ".join(fn(a) for a in arms)))
    row("comparison arm of I, V", lambda a: L[a]["comparison_arm"])
    row("S: one step ahead, context 1, error ratio predictor over copy (1,948 windows, 139 episodes; below 1 is better)", lambda a: ratio(ST["per_arm"][a]["ratio_ctx1"]))
    row("S: same ratio minus that of the comparison arm", lambda a: c(ST["paired_vs_comparison"]["%s_minus_%s" % (a, L[a]["comparison_arm"])]))
    row("I: lead gap R2 at +2.0 s, rolled-out state (n 441 windows with a lead at both times)", lambda a: "%.3f" % L[a]["R2_rollout"]["r2"])
    row("I: present state (copy)", lambda a: "%.3f" % L[a]["R2_copy"]["r2"])
    row("I: true future state (ceiling)", lambda a: "%.3f" % L[a]["R2_true_future"]["r2"])
    row("I: rollout minus copy", lambda a: c(L[a]["rollout_minus_copy"]))
    row("I: rollout minus ego-only floor (iv-b)", lambda a: c(L[a]["rollout_minus_floor"]))
    row("I: rollout minus rollout of the comparison arm, paired", lambda a: c(L[a]["rollout_minus_base_rollout_paired"]))
    row("V: change of the gap, skill of the rollout over persistence (441 windows, 48 episodes)", lambda a: c(L[a]["change_skill_rollout"]))
    row("V: ceiling from the true future state", lambda a: c(L[a]["change_skill_ceiling"]))
    row("V: rollout skill minus that of the comparison arm, paired", lambda a: c(list(L[a]["change_pair_vs_base"].values())[0]).replace("", "") if False else "%+.3f [%+.3f, %+.3f]" % (list(L[a]["change_pair_vs_base"].values())[0]["rollout_skill_diff"], list(L[a]["change_pair_vs_base"].values())[0]["ci95"][0], list(L[a]["change_pair_vs_base"].values())[0]["ci95"][1]))
    row("beside: history-seeded rollout R3h minus copy (ADD17, names nothing here)", lambda a: c(L[a]["rollout_R3h_minus_copy"]))
    row("beside: latent error at +2.0 s, skill over copy, R1", lambda a: "%.3f" % L[a]["latent_skill_over_copy_2s"]["R1"])
    row("name at this seed: S / I / V", lambda a: "%s / %s / %s" % (names[a]["S"], names[a]["I"], names[a]["V"]))
    return "\n".join(out)


def rrows(keys):
    out = ["| position, lead gap R2 (state) on the train probe set, 1,200 clips, paired over clips | difference | name by the rule of ADD16 C2 (six-seed spread %.4f) |" % SPREAD, "|---|---|---|"]
    for k in keys:
        out.append("| %s | %s | %s |" % (k.replace("_minus_", " minus "), c(pairs_R[k]["gap_R2_diff"]), R["R_names_per_seed"][k]))
    return "\n".join(out)


ctrl_motion = """* Build (card section 5, CPU, files in `F:/TanitAD-scratch/v8proof/spatial/motion/`): m1 defaults bit-identical to the tree of ADD16 for the argv of G, P and the mean default over 200 steps, max abs difference 0.0 (`checks/m1_identity.json`): **PASS**. m2 RES equals the copy at init, 0.0, flat and cell, control without it 3.8 to 4.4 (`checks/m2345_checks.json`): **PASS**. m3 the rollout term reaches the predictor and the state projection with a non-zero gradient, a target-only frame exactly 0.0: **PASS**. m4 the rollout inside the loss equals the rollout of the stack at read time, 0.0 (`MOTION_DEFS.md` item 9): **PASS**. m5 the total moves by exactly weight times the term, mutation form: **PASS** (m3 to m5 in `m2345_checks.json`). m6 the selfcheck reads the levers back from the BUILT model and refuses a wrong expectation (`checks/m6_selfcheck.json`, four argv shapes): **PASS**. Diagnostics flag on against off, same weights, max abs difference 0.0, shapes PR, RR, RES, ROLL (`checks_lead/lead_diag_check.json`): **PASS**. Tests: 55 passed (`tests.log`).
* Pod gate per arm (selfcheck on real data, PASS token before training): **NOT RE-VERIFIED HERE** (no pod access; the token files are on the pod). Evidence at hand: the dry run of the PR argv on real data read SELFCHECK PASS (row H-V8F-PROOF-35, `TIMES_NOTE.md` 02:13Z), and each of the five arms has a checkpoint with a recorded sha256 (`RESULT_S1_<ARM>.json`, meta `ckpt_sha256_record`).
* Lead read, per arm: t1 to t6 **PASS** for PR_s0, PR_s1, RES_s0, RR_s0 (`out_m/RESULT_RB_<ARM>.json`, `controls_pass` true); change form k1 to k3 **PASS** (`out_m/CHANGE_RB_*`); ADD17 variants x2 to x4 **PASS** (`ctx/RESULT_CTXROLL_<ARM>.json`); x1 (reproduce the banked pixel rollout) is **NOT EVALUATED** for these arms: the file reads `available false` because the pixel job of these arms was not run, and the tool prints PASS by default. It is evaluated for P_s0 only (`ctx/AFTERCHECK_CTXROLL_P_s0.json`, 0.00195 against 0.004). This is a gap in a control, not a failure; the names below are given with that flag.
* Step read: s1 (copy over copy is exactly 1) and s2 (shuffle keeps the point) **PASS** for all arms (`final/STEP_CTX_FINAL.json`).
* Position read (R): t1 to t4 **PASS** for every arm quoted (`spatial/s1/RESULT_S1_<ARM>.json`, `controls_pass` true).
* Family read (F): q1 to q5 **PASS** (window counts 2,399 windows of 141 episodes, 1,305 selection-critical and 1,094 common-case; self is zero 0.0; swap is minus 0.0; planted regression 0.899 m/s read back and named COST-LONGITUDINAL): `out_q3_twoseed/RESULT_Q3.json` (14 dumps) and `out_q3_oneseed/RESULT_Q3.json` (12 dumps), both ALL_PASS."""

ctrl_target = """* Build (card section 5 and note N1, CPU, files in `F:/TanitAD-scratch/v8proof/spatial/target/`): defaults bit-identical to the tree of ADD18 (`checks/t1_identity.json`): **PASS**. Each lever reaches the loss, a control group exactly 0.0 (`checks/t2345_checks.json`, t2 to t5 flat and cell): **PASS**. Selfcheck reads the levers back and refuses a wrong expectation (`checks/t6_selfcheck.json`, default, PE, PF, PEF): **PASS**. Diagnostics flag leaves every weight equal, 0.0 (`checks/t8_diag.json`): **PASS**. Lead independent check: FUTTOK analytic value difference 0.0, frame shifted by one visible at 0.0045, third action channel equals speed over 10 (`checks_lead/lead_target_check.json`): **PASS**. Tests: 87 passed (`tests_lead.log`). Stated limit of N1: the FUTTOK rollout inside the loss equals the read-time rollout within 5.1e-7, not 0.0 (bar 1e-6).
* Pod gate per arm: **NOT RE-VERIFIED HERE** (no pod access). Evidence at hand: dry gates of PE_s0 and PF_s0 (row H-V8F-PROOF-39, `TIMES_NOTE.md` 04:11Z) and the checkpoint sha256 of each arm in `RESULT_S1_<ARM>.json`.
* e1 (read tools on a non-EGOV arm give the same numbers, max abs difference 0.0 on P_s0): **PASS** (`ctx/E1_CTXROLL_P_s0.json`, all files 0.0).
* e2 (EGOV: the speed channel shuffled across windows must raise the one-step error): **PASS** for PE_s0 ratio shuffled over true **1.078** and PE_s1 **1.099** (`ctx/RESULT_CTXROLL_PE_s0.json`, `_PE_s1.json`, control x5_speed_file; recomputed from `DIAG_CTX_PE_s*.npz`: %.3f, %.3f). The speed file equals the logged speed over 10, correlation with logged acceleration 0.99999999 (x5): **PASS**.
* Lead read, per arm: t1 to t6 **PASS** for PE_s0, PE_s1, PF_s0 (`out_t/RESULT_RB_<ARM>.json`); k1 to k3 **PASS**; x2 to x4 **PASS**; x1 **NOT EVALUATED** for these arms (pixel job not run; same flag as for the MOTION card).
* Step, position and family controls: as in the MOTION block (s1, s2; t1 to t4; q1 to q5, all PASS).""" % (ST["per_arm"]["PE_s0"]["e2_shuffled_speed_ratio_over_true"], ST["per_arm"]["PE_s1"]["e2_shuffled_speed_ratio_over_true"])

sixrows = ["| arm | gap R2 (state), train probe set | file |", "|---|---|---|"]
for i in range(6):
    t = "HIER_s%d" % i
    x = P["arm_gap_R2"][t]["gap_R2_state"]
    sixrows.append("| %s | %.4f [%.4f, %.4f] | spatial/s1/RESULT_S1_%s.json |" % (t, x["r2"], x["ci95"][0], x["ci95"][1], t))
spr = ["| quantity | six seeds s0 to s5 (min, max, range) | three seeds s0 to s2 (banked in C2 of ADD16) | two seeds s0, s1 |", "|---|---|---|---|"]
for k in ("gap R2 (state)", "presence AUROC (state)", "change skill H", "change skill N", "change skill E"):
    a, b, d = P["spread_six"][k], P["spread_three_banked_C2"][k], P["spread_two"][k]
    spr.append("| %s | %.4f, %.4f, **%.4f** | %.4f, %.4f, %.4f | %.4f, %.4f, %.4f |" % (k, a["min"], a["max"], a["range"], b["min"], b["max"], b["range"], d["min"], d["max"], d["range"]))

md = []
A = md.append
A("# VERDICT_MOTION_TARGET_FINAL: cards ADD18 (MOTION) and ADD19 (TARGET), the finished reads collated (2026-11-15)")
A("")
A("Written by a teammate of the v8F proof programme, CPU only, no card edited, no banked file edited. Evidence class: MEASURED (ours; every number carries its file in `RESULT_MOTION_TARGET_FINAL.json`). "
  "Tier: **T0** (a world-model diagnostic on logged frames, never driving performance) for the step, lead, change and position reads; **T1-kin** (kinematic rollout on the model's own actions, logged frames) for the family read. "
  "STRATEGIC: **PAUSED** (PI item 25), never a pass. Intervals: episode-cluster bootstrap B 2,000 (lead and step reads: 139 episodes; change form: 48 episodes with a lead pair; position read: clips as clusters, 1,200 clips; family read: two-level bootstrap seeds then episodes, 141 episodes, 2,399 windows). "
  "A separated interval from a one-seed arm is necessary and not sufficient (H-ESTIM-SEED-1); a name that needs both seeds is given only where both seeds exist.")
A("")
A("## 0. What is on disk (and what is not)")
A("")
A("| arm | seeds read | role |\n|---|---|---|\n| PR (rollout loss, K 10, W 1.0, on the cell predictor; note ADD18_N1) | 0 and 1 | comparison arm P of the same seed for I and V; arm minus G for R (card ADD18 section 4) |\n"
  "| RES (residual form, flat predictor) | 0 | comparison arm G_s0 |\n| RR (RES + ROLL, flat predictor) | 0 | comparison arm G_s0 |\n| PE (EGOV, speed as third action channel, cell predictor) | 0 and 1 | comparison arm P of the same seed |\n"
  "| PF (FUTTOK, K 10, W 1.0, cell predictor) | 0 | comparison arms P_s0 and PR_s0 |\n\n"
  "**Not trained, so not read:** ROLL_s0, ROLL_s1 (ROLL alone on the flat predictor), RR_s1, RES_s1, PF_s1, PEF_s0, PEF_s1 (no checkpoint under `C:/Users/Admin/v8proof_c/ckpt/`). "
  "PR is the rollout loss on the CELL predictor (note ADD18_N1); it is not the ROLL arm of card ADD18 section 3, which is the same loss on the flat predictor.")
A("")
A("## 1. Card ADD18 (MOTION)")
A("")
A("### 1.1 Controls first")
A("")
A(ctrl_motion)
A("")
A("### 1.2 PR, both seeds (the only MOTION arm with both seeds)")
A("")
A(lead_rows(["PR_s0", "PR_s1"]))
A("")
A(rrows(["PR_s0_minus_G_s0", "PR_s1_minus_G_s1", "PR_s0_minus_P_s0", "PR_s1_minus_P_s1"]))
A("")
A("**Names (both seeds, the card's rules):** S **`STEP-BELOW-COPY`** (0.844 [0.831, 0.857] and 0.873 [0.864, 0.883]; both intervals below 1; note that PR is worse on this read than P at both seeds, +0.056 [+0.047, +0.064] and +0.077 [+0.067, +0.086]). "
  "I **`MOTION-FLAT`** (rollout minus copy +0.002 [-0.062, +0.065] and +0.025 [-0.038, +0.084]; rollout minus the rollout of P, paired, -0.059 [-0.134, +0.013] and +0.032 [-0.043, +0.105]; no interval of the two deciding lines excludes 0 above). "
  "V **`MOTION-STILL`** (+0.014 [-0.092, +0.130] and +0.091 [-0.015, +0.212]). R **`KEEPS-POSITION`** (arm minus G: -0.0005 [-0.019, +0.018] and -0.024 [-0.043, -0.005]; the seed-1 difference is below 0 and smaller than the six-seed spread %.4f, seed 0 is not below 0). "
  "F **`NO-COST`** for PR against G, against P and against HIER (section 1.5)." % SPREAD)
A("")
A("**Committed next step (card ADD18 section 6, quoted):** \"`MOTION-FLAT` and `MOTION-STILL` with `STEP-BELOW-COPY`: the predictor fits the step and the latent target does not carry the motion of objects; next lever, own card: an explicit target of object motion (the frozen token grid of the future frame, cell by cell).\" "
  "That lever is FUTTOK of card ADD19, arm PF, which has one seed (section 2.3). The second committed line, \"`LOSES-POSITION` or any `COST-<family>`: the arm is not promoted\", does not apply (no such name). The promotion line (`MOTION-IMAGINES` or `MOTION-MOVES` with `KEEPS-POSITION` and `NO-COST`) is not triggered by PR.")
A("")
A("### 1.3 RES and RR, one seed each: reported with numbers, one seed, names nothing")
A("")
A(lead_rows(["RES_s0", "RR_s0"]))
A("")
A(rrows(["RES_s0_minus_G_s0", "RR_s0_minus_G_s0"]))
A("")
A("RES_s0: one seed, names nothing. At this seed the rules read S `STEP-BELOW-COPY` (0.829 [0.814, 0.845]), I `MOTION-FLAT`, V `MOTION-STILL`, R `KEEPS-POSITION` (-0.021 [-0.038, -0.004], smaller than the spread %.4f), F `NO-COST`. "
  "RR_s0: one seed, names nothing. At this seed the rules read S `STEP-BELOW-COPY` (0.870 [0.858, 0.882]), **I `MOTION-IMAGINES`** (rollout minus copy +0.050 [+0.003, +0.098], rollout minus the ego floor +0.306 [+0.087, +0.619], both excluding 0; the lower end of the first interval is 0.003; the history-seeded R3h gives +0.050 [+0.001, +0.101]), "
  "V `MOTION-STILL` (+0.091 [-0.008, +0.212]), R `KEEPS-POSITION` (+0.029 [+0.012, +0.047], above 0), F `NO-COST`. "
  "Between-seed differences of the same read in arm types that have two seeds, for scale: rollout minus copy G 0.018 (-0.054, -0.036), P 0.010 (+0.013, +0.023), PR 0.023 (+0.002, +0.025), PE 0.004 (+0.021, +0.017); the seed floor of the flat predictor with the residual form and the rollout loss is not measured (RR_s1 and RES_s1 do not exist)." % SPREAD)
A("")
A("### 1.4 Attribution sentence of the card")
A("")
A("The card assigns an outcome of RR to RES, to ROLL, or to both, by which single-lever arm carries the same name. **It cannot be given.** RR_s0 reads `MOTION-IMAGINES` at one seed, which names nothing; RES_s0 reads `MOTION-FLAT` at its one seed; the other single-lever arm (ROLL_s0, ROLL_s1) was not trained. "
  "What the files do say: the residual form alone did not carry the I name at seed 0, and the rollout loss on the CELL predictor (PR, both seeds) is `MOTION-FLAT`. To name RR with both seeds the missing arms are RR_s1 (and ROLL_s0 for the attribution).")
A("")
A("### 1.5 Four-family line (T1-kin, ADD12 Q3, 2,399 windows, 141 episodes; STRATEGIC PAUSED)")
A("")
A("LONGITUDINAL, LATERAL, TACTICAL: PR **`NO-COST`** at both seeds against G, against P and against HIER (no cell reads WORSE by the rule; `out_q3_twoseed/VERDICT_Q3.md`); RES_s0 and RR_s0 `NO-COST` against G and against HIER at one seed (`out_q3_oneseed/VERDICT_Q3.md`; one seed, names nothing); STRATEGIC PAUSED (PI item 25). "
  "Said plainly beside the name: against HIER, PR has higher lateral errors at both seeds on three cells, heading +1.273 deg [+0.401, +2.205], curvature +0.0473 1/m [+0.0279, +0.0659], yaw rate +3.68 deg/s [+1.51, +5.98]; each interval excludes 0 on the worse side and each lies inside the six-seed spread of the baseline (2.551 deg, 0.1049 1/m, 10.65 deg/s), so by the rule of Q3 the name stays `NO-COST`. Against P, curvature error is +0.0443 1/m [+0.0090, +0.0863] (INSIDE-SPREAD, worse side); cross-track error is lower than P and HIER (-0.1212 m and -0.1357 m, INSIDE-SPREAD). All other cells OPEN (tables below).")
A("")
for pair in ("PR_minus_P", "PR_minus_G", "PR_minus_HIER"):
    A("**%s**, both seeds, Q3 **%s**" % (pair.replace("_minus_", " minus "), famN[pair][0]))
    A("")
    A(famN[pair][1])
    A("")
for pair in ("RES_minus_G", "RES_minus_HIER", "RR_minus_G", "RR_minus_HIER"):
    A("**%s**, seed 0 only (one seed, names nothing), reads %s" % (pair.replace("_minus_", " minus "), famN[pair][0]))
    A("")
    A(famN[pair][1])
    A("")
A("")
A("## 2. Card ADD19 (TARGET)")
A("")
A("### 2.1 Controls first")
A("")
A(ctrl_target)
A("")
A("### 2.2 PE, both seeds")
A("")
A(lead_rows(["PE_s0", "PE_s1"]))
A("")
A(rrows(["PE_s0_minus_P_s0", "PE_s1_minus_P_s1"]))
A("")
A("**Names (both seeds):** S **`STEP-SAME`** (one-step ratio minus that of P: -0.0031 [-0.0106, +0.0038] and -0.0048 [-0.0117, +0.0013]; both intervals include 0, so `STEP-IMPROVES` is not met; e2 passes, so the predictor does use the speed channel: shuffled over true 1.078 and 1.099). "
  "I **`TARGET-FLAT`** (rollout minus copy +0.021 [-0.028, +0.073] and +0.017 [-0.044, +0.079]; rollout minus the rollout of P, paired, +0.021 [-0.041, +0.080] and -0.011 [-0.100, +0.065]). "
  "V **`TARGET-STILL`** (-0.064 [-0.179, +0.039] and +0.042 [-0.059, +0.142]). R **`KEEPS-POSITION`** (+0.007 [-0.013, +0.026] and +0.009 [-0.012, +0.031]). F **`NO-COST`** against P and against HIER (section 2.4). "
  "The reference beside V that the card asks for, the ego block alone on the same windows, is **not produced by the read tools** (`rb_change.py` reads rollout and ceiling only); the value on the train probe clips, cited from the card, is +0.291 [+0.243, +0.334] (`spatial/s1/VERDICT_S1_G_s0.md`, in-sample) and is not the same windows. "
  "Limit of note ADD19_N1: PE also asks the inverse-dynamics head for the speed, and its auxiliary has another random draw than P of the same seed; a gain would not have been assigned to the predictor input alone (none was found).")
A("")
A("**Committed next step (card ADD19 section 6, quoted):** \"EGOV `STEP-SAME`: the hypothesis of section 1 is not supported; reported as written; the object-level slot is the next lever.\" "
  "The committed control PEi (note ADD19_N1) runs only if PE reads `STEP-IMPROVES` or better at seed 0: it does not. PEF runs only if PE or PF reads at least `TARGET-BEATS-BASE` or `STEP-IMPROVES` at seed 0: neither does (PE_s0 `STEP-SAME`/`TARGET-FLAT`; PF_s0 below), so PEF is not triggered by the card.")
A("")
A("### 2.3 PF, one seed: reported with numbers, one seed, names nothing")
A("")
A(lead_rows(["PF_s0"]))
A("")
A(rrows(["PF_s0_minus_P_s0"]))
A("")
A("PF_s0: one seed, names nothing. At this seed the rules read S `STEP-SAME` (the one-step ratio is 0.914 [0.896, 0.933], above P by +0.126 [+0.119, +0.134]; this is worse than P, not a gain), I `TARGET-FLAT` (rollout minus copy -0.082 [-0.206, +0.018]; minus the rollout of P -0.125 [-0.247, -0.019], below 0), V `TARGET-STILL` (-0.009 [-0.123, +0.090]), R `KEEPS-POSITION`, F `NO-COST` against P and HIER. Said plainly: PF_s0 has higher lateral errors than P (heading +1.09 deg, curvature +0.0626 1/m, yaw rate +7.05 deg/s) and higher than HIER on cross-track, heading, curvature, yaw rate and the tactical lateral error rate (+0.1109), every one INSIDE-SPREAD. "
  "K, the token read (pixel read, learned rollout minus copy at 2.0 s): **missing**. The pixel job was not run for PF_s0; it would come from `spatial/latvis_gw` / `latvis_p` (the pixel read of addendum 10, `gw_read_arm.sh` family), not from `read_t_arm.sh`. "
  "Beside: the +2.0 s latent error of PF_s0 from one state is above the copy (ratio 1.618 [1.527, 1.718] from frame 4, context 1), and the rollout with context grown from its own predictions (R3g) reads -0.871 [-1.384, -0.455] on the lead read; the history-seeded R3h reads +0.015 [-0.026, +0.056]. "
  "The card's line for FUTTOK `TARGET-FLAT` and `TARGET-STILL` at both seeds needs PF_s1, which does not exist; at seed 0 alone it is not a name.")
A("")
A("### 2.4 Four-family line (T1-kin)")
A("")
A("PE **`NO-COST`** at both seeds against P and against HIER (STRATEGIC PAUSED). PF_s0 `NO-COST` against P and HIER, one seed. PF_s0 minus PR_s0 reads `NO-COST GAIN-LONGITUDINAL GAIN-TACTICAL` at one seed (speed error -0.3205 m/s [-0.5074, -0.1625], BETTER outside the spread 0.2997; tactical lon error rate -0.0825 [-0.1189, -0.0500], BETTER); one seed, names nothing, and PR is a different arm from P.")
A("")
for pair in ("PE_minus_P", "PE_minus_HIER"):
    A("**%s**, both seeds, Q3 **%s**" % (pair.replace("_minus_", " minus "), famN[pair][0]))
    A("")
    A(famN[pair][1])
    A("")
for pair in ("PF_minus_P", "PF_minus_HIER", "PF_minus_PR"):
    A("**%s**, seed 0 only (one seed, names nothing), reads %s" % (pair.replace("_minus_", " minus "), famN[pair][0]))
    A("")
    A(famN[pair][1])
    A("")
A("")
A("## 3. Six-seed spread of the trained-state read")
A("")
A("The rule R (ADD16 C2) uses one quantity: the lead gap R2 of the trained state on the train probe set (in-sample for the model), the arm minus the comparison arm, against the **largest difference between baseline seeds on that number** (the range). "
  "Banked R names (rows H-V8F-PROOF-37 for P) used the **three-seed** spread (HIER_s0 to s2, range 0.0450, `spatial/s1/SPREAD_S1_HIER.md`); an earlier two-seed spread (0.0103, `SPREAD_S1_HIER_2seeds.md`) existed before that. The six seeds are now all read:")
A("")
A("\n".join(sixrows))
A("")
A("\n".join(spr))
A("")
A("The range of the gap R2 over six seeds is **%.4f**, equal to the three-seed range (the three added seeds, 0.4364, 0.4240, 0.4253, lie inside [0.4106, 0.4556]). **No R name changes** when the six-seed spread replaces the three-seed spread: the names of PR, PE, PF, RES, RR and of the banked P arms stay `KEEPS-POSITION` (no difference is below 0 with an interval excluding 0 and larger than 0.0450; the largest is PR_s1 minus G_s1 at -0.024 and RR_s0 minus G_s0 at +0.029, a gain). "
  "Had the two-seed spread (0.0103) been used, RES_s0 minus G_s0 (-0.021 [-0.038, -0.004]) and PR_s1 minus G_s1 (-0.024 [-0.043, -0.005]) would each read below 0 and larger than the spread at their seed; neither arm has both seeds below 0, so no both-seed `LOSES-POSITION` would result for PR either. "
  "The six-seed spreads of the other quantities are wider than the three-seed ones (presence AUROC 0.0068 against 0.0023; change skill H 0.0265 against 0.0049); R does not use them. Reproduction check of the paired tool: P_s0 minus G_s0 recomputed from the `OOF_S1_*` files equals the banked +0.0104 (gap R2 0.5622 against 0.5518).")
A("")
A("## 4. Limits")
A("")
A("One training run per arm per seed; two seeds at most; a separated interval from a one-seed arm is necessary and not sufficient, and the six-seed spread of the baseline is measured on the position read only (the lead read has two baseline seeds of G and no spread of its own). The probes show what is linearly readable. "
  "The lead and change reads use 441 windows of 139 episodes (48 with a lead pair); a small effect on the change read cannot be separated on them. The position read is in-sample for the model. T1-kin is a kinematic rollout on logged frames, not driving performance. "
  "x1 of the rollout reads is not evaluated for the new arms (section 1.1). The per-arm pod gate tokens were not re-checked. The one-step difference of ratios (S of ADD19) was computed here with the resampling of `ctx_step.py` (B 2,000, seed 0) because no earlier file holds it. STRATEGIC: PAUSED (PI item 25).")
A("")
A("## 5. Consistency with the banked seed-0 reports (rows H-V8F-PROOF-35 to 40, `GAC_ROWS_29.md` to `GAC_ROWS_33.md`)")
A("")
A("Every seed-0 number in row 40 (PR_s0 +0.002 [-0.062, +0.065], PE_s0 +0.021 [-0.028, +0.073], change skill +0.014 and -0.064, latent skill 0.251 and 0.233, e2 1.078) is reproduced by the files read here. The row reported `IMAGINE-FLAT, no name` at seed 0; with seed 1 read, the card names are `MOTION-FLAT` (PR) and `TARGET-FLAT` (PE), the same direction. "
  "The row 40 sentence \"if both seeds stay flat, the committed step of card ADD19 is the object-level slot\" is the committed step recorded in section 2.2 for PE. For the cell predictor of ADD16 (`CELL-FLAT`, row 37) nothing changes. The only seed-0 report that differs from what a one-seed read of the new arms shows is RR_s0, which was not reported before and reads `MOTION-IMAGINES` at seed 0 alone.")
open(OUT + "/VERDICT_MOTION_TARGET_FINAL.md", "w", encoding="utf-8", newline="\n").write("\n".join(md) + "\n")
print("verdict written", len("\n".join(md)))
