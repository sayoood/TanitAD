"""Build PATCH_PROPOSAL.md + raw/verify_mislabel.json for the refc-base-30k / refcv7 fan-ceiling mislabel.

Zero GPU, read-only. Reads ONLY: git objects at rev 8522e49 of C:/Users/Admin/tanitad-push/.git (never writes there),
the X8 battery raw JSON, D:/refcv7_eval_kit/ckpt/config.json, and the live copies of the two steering files (read-only).
Writes ONLY inside its own package dir.  Run:  python -I code/build_patch.py
(Source is deliberately ASCII and backslash-free: it was authored through a shell heredoc.)
"""
import json, re, subprocess, sys

GIT_DIR = "C:/Users/Admin/tanitad-push/.git"
REV = "8522e49"
GOALS = "Project Steering/GOALS_AND_CLAIMS.md"
REG = "Project Steering/MODEL_REGISTRY.md"
FLY = "E:/Projects/TanitAD/FlyWheels/TanitAD_EvalFlyWheel/incoming"
BAT = FLY + "/2026-09-28-refcv7-standard-tests/battery"
RAW = BAT + "/raw/fanctl_x8"
PKG = FLY + "/2026-10-11-refc-base-30k-mislabel"
LIVE = "D:/Projects/TanitAD/Project Steering"
CFG = "D:/refcv7_eval_kit/ckpt/config.json"
REL_RAW = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/raw/fanctl_x8"
REL_RES = "FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-28-refcv7-standard-tests/battery/RESULT.md"
NL, CR, Q, BT = chr(10), chr(13), chr(34), chr(96)
CHECKMARK = chr(0x2705)


def gshow(path):
    r = subprocess.run(["git", "--git-dir", GIT_DIR, "show", REV + ":" + path], capture_output=True)
    if r.returncode != 0 or not r.stdout:
        sys.exit("INCONCLUSIVE: git show failed/empty: " + path)
    return r.stdout.decode("utf-8")


def lines_of(text):
    return [l.rstrip(CR) for l in text.split(NL)]


def jl(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


goals_txt, reg_txt = gshow(GOALS).replace(CR, ""), gshow(REG).replace(CR, "")
G, R = lines_of(goals_txt), lines_of(reg_txt)
ROWSTART = "| **D-REFCV7-FANCEIL** |"
ROW_I = next(i for i, l in enumerate(G) if l.startswith(ROWSTART))
assert sum(l.startswith(ROWSTART) for l in G) == 1
HDR_I = next(i for i, l in enumerate(R) if l.startswith("### 15.3 Fan ceiling"))
assert R[HDR_I + 2].startswith("Oracle-in-fan ADE **0.1914 m**")
OLD_GOALS, OLD_HDR, OLD_R1, OLD_R2 = G[ROW_I], R[HDR_I], R[HDR_I + 2], R[HDR_I + 3]

banked = json.loads(gshow("taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json"))
driving = json.loads(gshow("taniteval/results/driving_refc-base-30k.json"))
b_gap = banked["gap"]["refc-base-30k"]
b_or = banked["comparisons"]["oracle_in_fan_full"]["a"]
b_sel = driving["headline"]["ade_0_2s"]
b_repro = jl(RAW + "/ctl_repro_refc-base-30k.json")
b_p = b_repro["panel"]
c0, c1 = jl(RAW + "/ctl_refcv7_50400_fan_s0.json"), jl(RAW + "/ctl_refcv7_50400_fan_s1.json")
p0, p1 = c0["panel"], c1["panel"]
cfg = jl(CFG)
sraf = cfg["selection"]["sampler_ranks_the_fan"]
f4 = lambda x: "%.4f" % x
pct = lambda x: "%.2f" % (100 * x)

# ---------------- figures: all re-read from raw, none typed ----------------
doc = {"oracle": "0.1914", "selected": "0.4728", "gap": "0.2813", "gap_lo": "0.2127", "gap_hi": "0.3543", "frac2x": "41.09"}
base = {"oracle": f4(b_or["mean"]), "selected": f4(b_sel["mean"]), "gap": f4(b_gap["sel_gap_mean"]),
        "gap_lo": f4(b_gap["sel_gap_ci"]["lo"]), "gap_hi": f4(b_gap["sel_gap_ci"]["hi"]), "frac2x": pct(b_gap["frac_sel_2x_worse"])}


def fig(p):
    g = p["sel_gap__selected_minus_best"]
    return {"oracle": f4(p["oracle_in_fan"]["mean"]), "selected": f4(p["selected"]["mean"]), "gap": f4(g["delta"]),
            "gap_lo": f4(g["lo"]), "gap_hi": f4(g["hi"]), "frac2x": pct(p["frac_sel_2x_worse"])}


rep, r7_0, r7_1 = fig(b_p), fig(p0), fig(p1)
checks = {}
checks["A1_doc_figures_equal_base30k_banked_raw_git"] = (doc == base)
checks["A2_X8_reproduction_of_base30k_equals_banked"] = (rep == base)
checks["A3_no_doc_figure_equals_refcv7_seed0_or_seed1"] = all(doc[k] != r7_0[k] and doc[k] != r7_1[k] for k in doc)
checks["A4_base30k_881_windows_40_eps_128_anchors"] = ((b_p["n_windows"], b_p["n_episodes"], b_p["n_candidates"]) == (881, 40, 128)
                                                       and (b_gap["sel_gap_ci"]["n_windows"], b_gap["sel_gap_ci"]["n_episodes"]) == (881, 40))
checks["A5_refcv7_4754_windows_139_eps_117_cands_both_seeds"] = ((p0["n_windows"], p0["n_episodes"], p0["n_candidates"]) == (4754, 139, 117)
                                                                 == (p1["n_windows"], p1["n_episodes"], p1["n_candidates"]))
checks["A6_base30k_ckpt_step_29999"] = b_repro["ckpt_step"] == 29999 and b_repro["ckpt"].endswith("refc-base-30k/ckpt.pt")
checks["A7_refcv7_ckpt_md5_declared_b418d0fc"] = c0["stamp"]["ckpt_md5_declared"] == "b418d0fc4a92a6848c246a6a7c50207b"
checks["A8_refcv7_config_sampler_ranks_the_fan_is_False"] = (sraf is False)

live_goals = open(LIVE + "/GOALS_AND_CLAIMS.md", encoding="utf-8", newline="").read().replace(CR, "")
live_reg = open(LIVE + "/MODEL_REGISTRY.md", encoding="utf-8", newline="").read().replace(CR, "")
blk = OLD_R1 + NL + OLD_R2
checks["B1_old_goals_row_verbatim_exactly_once_at_8522e49"] = goals_txt.count(OLD_GOALS) == 1
checks["B2_old_goals_row_verbatim_exactly_once_in_live_worktree"] = live_goals.count(OLD_GOALS) == 1
checks["B3_old_registry_block_verbatim_exactly_once_at_8522e49"] = reg_txt.count(blk) == 1
checks["B4_old_registry_block_verbatim_exactly_once_in_live_worktree"] = live_reg.count(blk) == 1
live_lineno = {"goals_row": live_goals.split(NL).index(OLD_GOALS) + 1, "reg_hdr": live_reg.split(NL).index(OLD_HDR) + 1}

# ---------------- extra control-run values (X8) ----------------
CK = "ckpt_50400.pt md5 b418d0fc4a92a6848c246a6a7c50207b"
rp = p0["random_expectation"]
rm = p0["rand_exp_minus_selected"] if "rand_exp_minus_selected" in p0 else p0["random_exp_minus_selected"]
ex = p0["extra_pickers"]["argmax_refined_logits"]
ex_d = ex["minus_selected"]
top2 = p0["topk_random_by_score"]["refined_logits"]["K2"]
skill = p0["skill_fraction"]["value"]

# ---------------- NEW text (ASCII only; old text is carried verbatim inside the was-note) ----------------
BASE_NOTE = ("those figures are REF-C-base at step 29,999: 128 anchors, 881 windows / 40 val episodes ("
             + BT + "taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json" + BT + " " + BT + "gap[" + Q + "refc-base-30k" + Q + "]" + BT
             + " and " + BT + "comparisons.oracle_in_fan_full.a" + BT + "; ADE@2s " + base["selected"] + " = " + BT
             + "taniteval/results/driving_refc-base-30k.json" + BT + " " + BT + "headline.ade_0_2s.mean" + BT + ", MODEL_REGISTRY REF-C-base results)")
REFCV7_FIG = ("oracle-in-fan **" + r7_0["oracle"] + " m** [" + f4(p0["oracle_in_fan"]["lo"]) + ", " + f4(p0["oracle_in_fan"]["hi"])
              + "] vs selected **" + r7_0["selected"] + " m** [" + f4(p0["selected"]["lo"]) + ", " + f4(p0["selected"]["hi"])
              + "] - delta **+" + r7_0["gap"] + " m [+" + r7_0["gap_lo"] + ", +" + r7_0["gap_hi"] + "]** (paired); a candidate more than 2x better than the "
              "selected one exists on **" + r7_0["frac2x"] + " %** of windows; inference-seed replicate (seed 1, same checkpoint): "
              + r7_1["oracle"] + " / " + r7_1["selected"] + " / +" + r7_1["gap"] + " [+" + r7_1["gap_lo"] + ", +" + r7_1["gap_hi"] + "] / "
              + r7_1["frac2x"] + " %; one training seed")
SCOPE = ("MEASURED for refcv7-r101-s0 at step 50,400 (" + CK + "; PhysicalAI held-out eval-139: " + format(p0["n_windows"], ",") + " windows / "
         + str(p0["n_episodes"]) + " episodes / " + str(p0["n_candidates"]) + " candidates; **T0 DIAGNOSTIC** - the oracle and every difference "
         "containing it use GT, " + BT + "selected" + BT + " alone is the T1 open-loop pick; FULL-SET mean, episode-cluster bootstrap, paired for the delta, "
         "n_boot 2000 seed 0, which answers *another draw of EPISODES?* only)")
CTRL_BODY = ("uniform-random pick " + f4(rp["mean"]) + " m [" + f4(rp["lo"]) + ", " + f4(rp["hi"])
             + "], selected better by +" + f4(rm["delta"]) + " [+" + f4(rm["lo"]) + ", +" + f4(rm["hi"]) + "], skill fraction " + f4(skill)
             + " (point estimate); but the plain argmax of refined_logits is " + f4(ex["ade"]["mean"]) + " (" + f4(ex_d["delta"]) + " ["
             + f4(ex_d["lo"]) + ", " + f4(ex_d["hi"]) + "] vs selected, separated) and a random pick among the model's own top-2 by refined_logits is "
             + f4(top2["ade"]["mean"]) + ", so re-ranking by the model's own logits recovers only " + f4(abs(ex_d["delta"])) + " m of the "
             + r7_0["gap"] + " m gap - the rest is a GT-peeking ceiling")
NOTCOMP = "Not comparable with the base-30k panel (different windows, 117 vs 128 anchors): do not pool or difference them."
ARTI = ("Raw: " + BT + REL_RAW + "/ctl_refcv7_50400_fan_s0.json" + BT + " (+ " + BT + "_s1" + BT + "); narrative " + BT + REL_RES + BT
        + " section D-REFCV7-FANCEIL.")

m = re.match(r"^[|] [*][*]D-REFCV7-FANCEIL[*][*] [|] (.*) [|] (" + CHECKMARK + r" .*) [|]$", OLD_GOALS)
assert m, "row shape changed"
old_c1, old_c2 = m.group(1), m.group(2)
head_c1 = "**refcv7's proposal fan contains much better plans than the one it picks.**"
assert old_c1.startswith(head_c1)
old_c1_tail = old_c1[len(head_c1):].strip()

NEW_GOALS = ("| **D-REFCV7-FANCEIL** | " + head_c1 + " **CORRECTED 2026-10-11 (EvalFlyWheel X8, raw-reproduced): the figures this row carried were "
             + BT + "refc-base-30k" + BT + "'s, not refcv7's.** *(corrected 2026-10-11: was " + Q + old_c1_tail + Q + " - " + BASE_NOTE + ".)* "
             + SCOPE + ": " + REFCV7_FIG + ". " + BT + "sampler_ranks_the_fan" + BT + " = **false** (this clause WAS refcv7's: " + BT + "config.json" + BT
             + " " + BT + "selection.sampler_ranks_the_fan" + BT + " False). " + ARTI + " " + NOTCOMP + " | "
             + old_c2 + " *(update 2026-10-11: random-pick control now RUN on refcv7: " + CTRL_BODY + "; see " + BT + REL_RES + BT + " section D-REFCV7-FANCEIL.)* |")

NEW_R = [("**CORRECTED 2026-10-11 - the figures this subsection carried belong to " + BT + "refc-base-30k" + BT + ", not to refcv7.** "
          "*(corrected 2026-10-11: was " + Q + OLD_R1 + " " + OLD_R2 + Q + " - " + BASE_NOTE + ".)*"),
         ("**refcv7-r101-s0 @ 50,400** (" + CK + "; " + format(p0["n_windows"], ",") + " windows / " + str(p0["n_episodes"]) + " episodes / "
          + str(p0["n_candidates"]) + " candidates; T0 DIAGNOSTIC, FULL-SET mean, episode-cluster bootstrap n_boot 2000 - *another draw of EPISODES?* only): "
          + REFCV7_FIG + "; " + BT + "sampler_ranks_the_fan" + BT + " false (" + BT + "ckpt/config.json" + BT + " " + BT + "selection" + BT + ")."),
         ("Random-pick control now RUN on refcv7 (was: not yet run): " + CTRL_BODY + ". " + ARTI + " " + NOTCOMP + " " + BT + "D-REFCV7-FANCEIL" + BT + ".")]
assert OLD_HDR == "### 15.3 Fan ceiling (MEASURED, T1-family open loop, PhysicalAI held-out)"
NEW_HDR = ("### 15.3 Fan ceiling (MEASURED, T0 diagnostic - oracle uses GT, " + BT + "selected" + BT + " alone is the T1 open-loop pick; PhysicalAI held-out) "
           "*(corrected 2026-10-11: was " + Q + "MEASURED, T1-family open loop, PhysicalAI held-out" + Q + ")*")

B_GOALS = ("| **D-REFCV7-FANCEIL-CORRECTION** | **CORRECTION 2026-10-11 to the row immediately above: its five figures are " + BT + "refc-base-30k" + BT
           + "'s, not refcv7's** - " + BASE_NOTE + ". " + SCOPE + ": " + REFCV7_FIG + ". " + ARTI + " " + NOTCOMP
           + " | The row above keeps its original text for audit; its " + BT + "sampler_ranks_the_fan" + BT + " = false clause is refcv7's and stands. *(random-pick control now RUN on refcv7: "
           + CTRL_BODY + ".)* |")
B_REG = [("**CORRECTION 2026-10-11 (inserted above the two lines that follow; they are retained verbatim for audit): the figures in those two lines belong to "
          + BT + "refc-base-30k" + BT + ", not to refcv7** - " + BASE_NOTE + "."), NEW_R[1], NEW_R[2]]

# ---------------- mechanical guards: simulated apply, in memory only ----------------
tok = lambda s: set(re.findall(r"[0-9]+(?:[.][0-9]+)?", s))
def superset(old, new):
    return tok(old) <= tok(new), sorted(tok(old) - tok(new))
ok1, miss1 = superset(OLD_GOALS, NEW_GOALS)
ok2, miss2 = superset(OLD_R1 + " " + OLD_R2, " ".join(NEW_R))
ok3, miss3 = superset(OLD_HDR, NEW_HDR)
checks["C1_numeric_token_superset_GOALS_row"] = ok1
checks["C2_numeric_token_superset_REGISTRY_15_3_block"] = ok2
checks["C3_numeric_token_superset_REGISTRY_15_3_header"] = ok3
newG = G[:ROW_I] + [NEW_GOALS] + G[ROW_I + 1:]
newR = R[:HDR_I + 2] + NEW_R + R[HDR_I + 4:]
rm_g, ad_g, rm_r, ad_r = set(G) - set(newG), set(newG) - set(G), set(R) - set(newR), set(newR) - set(R)
checks["C4_optionA_removes_exactly_1_GOALS_line_adds_1"] = (rm_g == {OLD_GOALS} and len(ad_g) == 1)
checks["C5_optionA_removes_exactly_2_REGISTRY_lines_adds_3"] = (rm_r == {OLD_R1, OLD_R2} and len(ad_r) == 3)
bG = G[:ROW_I + 1] + [B_GOALS] + G[ROW_I + 1:]
bR = R[:HDR_I + 2] + B_REG[:1] + R[HDR_I + 2:HDR_I + 4] + B_REG[1:] + R[HDR_I + 4:]
checks["C6_optionB_removes_zero_lines_in_both_files"] = (set(G) <= set(bG)) and (set(R) <= set(bR))
checks["C7_goals_row_keeps_3_cell_shape"] = (NEW_GOALS.count(" | ") == OLD_GOALS.count(" | ") and NEW_GOALS.startswith("| ") and NEW_GOALS.endswith(" |"))
checks["C8_file_level_numeric_tokens_lost_GOALS_is_zero"] = (len(tok(goals_txt) - tok(NL.join(newG))) == 0)
checks["C9_file_level_numeric_tokens_lost_REGISTRY_is_zero"] = (len(tok(reg_txt) - tok(NL.join(newR))) == 0)
checks["C10_new_text_is_ascii_outside_carried_old_text"] = all(ord(c) < 128 for c in (BASE_NOTE + REFCV7_FIG + SCOPE + CTRL_BODY + NOTCOMP + ARTI + NEW_HDR.replace("", "")))

# deliberate-regression controls: each guard must go RED on a knowingly broken input
checks["M1_superset_guard_goes_red_when_an_old_figure_is_dropped"] = (superset(OLD_GOALS, NEW_GOALS.replace("0.4728", "X"))[0] is False)
checks["M2_arm_guard_goes_red_when_a_doc_figure_equals_refcv7"] = (not all(dict(doc, oracle=r7_0["oracle"])[k] != r7_0[k] for k in doc))
checks["M3_oldtext_uniqueness_guard_goes_red_on_a_mutated_line"] = (goals_txt.count(OLD_GOALS + "x") == 0 and goals_txt.count(OLD_GOALS) == 1)
checks["M4_line_superset_guard_goes_red_for_option_A_in_GOALS"] = (not (set(G) <= set(newG)))

out = {
    "rev": REV, "script": "code/build_patch.py", "checks": checks, "all_pass": all(checks.values()),
    "line_numbers": {"at_8522e49": {"goals_row": ROW_I + 1, "registry_header_15_3": HDR_I + 1, "registry_old_lines": [HDR_I + 3, HDR_I + 4]},
                     "live_worktree_D": live_lineno},
    "figures": {"doc_row": doc, "base30k_banked_git": base, "base30k_battery_repro": rep, "refcv7_s0": r7_0, "refcv7_s1": r7_1},
    "windows_eps_cands": {"base30k": [b_p["n_windows"], b_p["n_episodes"], b_p["n_candidates"]], "refcv7": [p0["n_windows"], p0["n_episodes"], p0["n_candidates"]]},
    "tier_stamp_raw": c0["stamp"]["tier"],
    "refcv7_config_sampler_ranks_the_fan": sraf,
    "overstatement_ratio_doc_over_true_refcv7": {k: round(float(doc[k]) / float(r7_0[k]), 3) for k in ("oracle", "selected", "gap")},
    "token_misses": {"goals": miss1, "registry_block": miss2, "registry_hdr": miss3},
}
with open(PKG + "/raw/verify_mislabel.json", "w", encoding="utf-8", newline=NL) as f:
    json.dump(out, f, indent=1)
print("all_pass", out["all_pass"])
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)
print("token misses", miss1, miss2, miss3)

# ---------------- render PATCH_PROPOSAL.md from the three template files ----------------
ratio = lambda k: "%.2f" % (float(doc[k]) / float(r7_0[k]))
tbl = NL.join("| " + k + " | " + doc[k] + " | " + base[k] + " | " + rep[k] + " | " + r7_0[k] + " | " + r7_1[k] + " |" for k in doc)
chk = NL.join("| " + BT + k + BT + " | " + ("PASS" if v else "FAIL") + " |" for k, v in checks.items())
T = ""
for part in ("a", "b", "c"):
    with open(PKG + "/code/patch_template_" + part + ".md", encoding="utf-8", newline="") as f:
        T += f.read().replace(CR, "")
rep_map = {"REV": REV, "RELRAW": REL_RAW, "RELRES": REL_RES, "LG": str(live_lineno["goals_row"]), "LR": str(live_lineno["reg_hdr"]),
           "LR2": str(live_lineno["reg_hdr"] + 2), "LR3": str(live_lineno["reg_hdr"] + 3), "G1": str(ROW_I + 1), "R0": str(HDR_I + 1),
           "R2": str(HDR_I + 3), "R3": str(HDR_I + 4), "R4": str(HDR_I + 4), "SRAF": str(sraf), "TBL": tbl, "CHK": chk,
           "BW": "%d / %d / %d" % (b_p["n_windows"], b_p["n_episodes"], b_p["n_candidates"]),
           "CW": "%d / %d / %d" % (p0["n_windows"], p0["n_episodes"], p0["n_candidates"]),
           "RSEL": ratio("selected"), "RORA": ratio("oracle"), "RGAP": ratio("gap"), "F0": r7_0["frac2x"], "F1": r7_1["frac2x"],
           "OLDG": OLD_GOALS, "NEWG": NEW_GOALS, "BG": B_GOALS, "OLDR1": OLD_R1, "OLDR2": OLD_R2, "NR0": NEW_R[0], "NR1": NEW_R[1], "NR2": NEW_R[2],
           "OLDH": OLD_HDR, "NEWH": NEW_HDR, "BR0": B_REG[0], "BR1": B_REG[1], "BR2": B_REG[2], "TIER": c0["stamp"]["tier"],
           "R70": r7_0["oracle"], "R7S": r7_0["selected"], "R7G": r7_0["gap"], "R7L": r7_0["gap_lo"], "R7H": r7_0["gap_hi"], "R7F": r7_0["frac2x"]}
md = T
for k in sorted(rep_map, key=len, reverse=True):
    md = md.replace("@@" + k + "@@", rep_map[k])
assert "@@" not in md, "unreplaced marker"
with open(PKG + "/PATCH_PROPOSAL.md", "w", encoding="utf-8", newline=NL) as f:
    f.write(md)
print("wrote PATCH_PROPOSAL.md", len(md))
