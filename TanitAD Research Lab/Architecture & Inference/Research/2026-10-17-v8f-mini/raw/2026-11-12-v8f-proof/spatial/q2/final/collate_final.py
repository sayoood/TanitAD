"""collate_final.py -- collates the finished reads of cards ADD18 (MOTION) and ADD19 (TARGET) into RESULT_MOTION_TARGET_FINAL.json.
CPU only; reads result files; writes only into its own folder. Runs the cheap parts that were not yet written: the paired one-step difference
(S of ADD19), the paired position difference on the train probe set (R) and the six-seed spread of the baseline."""
import json
import os
import re
import sys

import numpy as np

Q2 = "F:/TanitAD-scratch/v8proof/spatial/q2"
S1 = "F:/TanitAD-scratch/v8proof/spatial/s1"
CR = "C:/Users/Admin/v8proof_c/ctxroll"
OUT = Q2 + "/final"
sys.path.insert(0, S1)
import s1_read as S  # noqa: E402

R = {"sources": {}}


def jl(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_pair_i(path):
    t = open(path, "r", encoding="utf-8").read()
    m = re.search(r"\(i\) rolled-out state, (\S+) minus (\S+): ([+-][\d.]+) \[([+-][\d.]+), ([+-][\d.]+)\]", t)
    return {"a": m.group(1)[:-1] if m.group(1).endswith(",") else m.group(1), "diff": float(m.group(3)), "ci95": [float(m.group(4)), float(m.group(5))]}


ARMS = {  # arm: (folder, comparison arm for I/V/S, kind)
    "PR_s0": ("out_m", "P_s0"), "PR_s1": ("out_m", "P_s1"), "RES_s0": ("out_m", "G_s0"), "RR_s0": ("out_m", "G_s0"),
    "PE_s0": ("out_t", "P_s0"), "PE_s1": ("out_t", "P_s1"), "PF_s0": ("out_t", "P_s0"),
}
lead = {}
for arm, (fol, base) in ARMS.items():
    rb = jl("%s/%s/RESULT_RB_%s.json" % (Q2, fol, arm))
    ch = jl("%s/%s/CHANGE_RB_%s_%s.json" % (Q2, fol, arm, base))
    pr = parse_pair_i("%s/%s/PAIR_RB_%s_minus_%s.md" % (Q2, fol, arm, base))
    cx = jl("%s/ctx/RESULT_CTXROLL_%s.json" % (Q2, arm))
    ca = ch["arms"][arm]
    lead[arm] = {
        "files": ["%s/%s/RESULT_RB_%s.json" % (Q2, fol, arm), "%s/%s/CHANGE_RB_%s_%s.json" % (Q2, fol, arm, base), "%s/%s/PAIR_RB_%s_minus_%s.md" % (Q2, fol, arm, base), "%s/ctx/RESULT_CTXROLL_%s.json" % (Q2, arm)],
        "comparison_arm": base, "n_windows": rb["n_lead_both_windows"], "controls_t1_t6": {k: v["pass"] for k, v in rb["controls"].items()}, "controls_t_pass": rb["controls_pass"],
        "R2_rollout": rb["rb"]["i"], "R2_copy": rb["rb"]["ii"], "R2_true_future": rb["rb"]["iii"], "R2_ego_floor_ivb": rb["rb"]["ivb"], "R2_ego_iv": rb["rb"]["iv"],
        "rollout_minus_copy": rb["i_minus_ii"], "rollout_minus_floor": rb["i_minus_floor"], "future_minus_copy": rb["iii_minus_ii"], "floor_used": rb["floor_used"],
        "tool_outcome_this_seed": rb["outcome_this_seed"], "rollout_minus_base_rollout_paired": pr,
        "change_skill_rollout": ca["rollout"], "change_skill_ceiling": ca["ceiling"], "change_n_windows": ca["n_windows"], "change_n_episodes": ca["n_episodes_with_windows"],
        "change_pair_vs_base": ch["pairs"], "change_controls_k1_k3": ch["controls_pass"],
        "ctxroll_controls_x1_x5_pass": cx["controls_pass"], "x1_available": cx["controls"]["x1_reproduce_banked_rollout"]["available"],
        "x5_speed": cx["controls"].get("x5_speed_file"),
        "latent_skill_over_copy_2s": cx["latent_mse_at_2s"]["skill_over_copy"],
    }
    for v in ("R3g", "R3h"):
        z = jl("%s/ctx/RESULT_RB_%s_%s.json" % (Q2, arm, v))
        lead[arm]["rollout_%s_minus_copy" % v] = z["i_minus_ii"]
        lead[arm]["controls_%s_pass" % v] = z["controls_pass"]
R["lead_read"] = lead

# comparison arms (banked)
comp = {}
for arm in ("G_s0", "G_s1", "P_s0", "P_s1"):
    fol = "out" if arm.startswith("G") else "out_p"
    rb = jl("%s/%s/RESULT_RB_%s.json" % (Q2, fol, arm))
    base = "G_s%s" % arm[-1]
    comp[arm] = {"file": "%s/%s/RESULT_RB_%s.json" % (Q2, fol, arm), "R2_rollout": rb["rb"]["i"], "R2_copy": rb["rb"]["ii"], "rollout_minus_copy": rb["i_minus_ii"], "controls_pass": rb["controls_pass"]}
    if arm.startswith("P"):
        ch = jl("%s/out_p/CHANGE_RB_%s_G.json" % (Q2, arm))
        comp[arm]["change_skill_rollout"] = ch["arms"][arm]["rollout"]
        comp[arm]["change_skill_ceiling"] = ch["arms"][arm]["ceiling"]
    else:
        ch = jl("%s/ctx/CHANGE_CTX_%s.json" % (Q2, arm)) if os.path.exists("%s/ctx/CHANGE_CTX_%s.json" % (Q2, arm)) else None
R["comparison_arms_lead_read"] = comp

# S: one-step ratio, and paired difference of ratios
def boot_ratio_diff(za, zb, B=2000, seed=0):
    eid = za["eid"].astype(str)
    assert (eid == zb["eid"].astype(str)).all()
    uep = sorted(set(eid)); ix = {e: i for i, e in enumerate(uep)}; epi = np.array([ix[e] for e in eid]); n = len(uep)
    f = lambda z, k: np.bincount(epi, weights=z[k], minlength=n)
    na, da, nb, db = f(za, "step_c1"), f(za, "step_copy"), f(zb, "step_c1"), f(zb, "step_copy")
    rng = np.random.default_rng(seed); idx = rng.integers(0, n, size=(B, n))
    d = na[idx].sum(1) / da[idx].sum(1) - nb[idx].sum(1) / db[idx].sum(1)
    pt = na.sum() / da.sum() - nb.sum() / db.sum()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(pt), float(lo), float(hi), n


step = {}
stepj = jl(OUT + "/STEP_CTX_FINAL.json")
for arm, v in stepj["arms"].items():
    step[arm] = {"windows": v["windows"], "episodes": v["episodes"], "ratio_ctx1": v["ratios"]["step_context_1"], "ratio_ctx3": v["ratios"]["step_context_3"], "controls_s1_s2": v["controls"]}
step_pairs = {}
for arm, base in (("PE_s0", "P_s0"), ("PE_s1", "P_s1"), ("PF_s0", "P_s0"), ("PR_s0", "P_s0"), ("PR_s1", "P_s1"), ("RES_s0", "G_s0"), ("RR_s0", "G_s0")):
    za = np.load("%s/DIAG_CTX_%s.npz" % (CR, arm), allow_pickle=True); zb = np.load("%s/DIAG_CTX_%s.npz" % (CR, base), allow_pickle=True)
    pt, lo, hi, n = boot_ratio_diff(za, zb)
    step_pairs["%s_minus_%s" % (arm, base)] = {"diff_of_ratios": pt, "ci95": [lo, hi], "n_episodes": n, "estimator": "episode-cluster bootstrap B 2000 seed 0 (ctx_step.py resampling), difference of the two ratios predictor over copy"}
for arm in ("PE_s0", "PE_s1"):
    z = np.load("%s/DIAG_CTX_%s.npz" % (CR, arm), allow_pickle=True)
    step[arm]["e2_shuffled_speed_ratio_over_true"] = float(z["step_c1_shuf"].sum() / z["step_c1"].sum())
R["step_read"] = {"per_arm": step, "paired_vs_comparison": step_pairs, "files": [OUT + "/STEP_CTX_FINAL.json", CR + "/DIAG_CTX_<ARM>.npz"]}

# R: position on the train probe set
def run(tag):
    return jl("%s/RESULT_S1_%s.json" % (S1, tag))


def pair(a, b):
    rA = run(a); rB = run(b)
    zA = np.load("%s/%s" % (S1, rA["oof_file"])); zB = np.load("%s/%s" % (S1, rB["oof_file"]))
    zA = {k: zA[k] for k in zA.files}; zB = {k: zB[k] for k in zB.files}
    d, n = S.paired_from(zA, zB, 2000)
    g = d["gap R2 (state)"]
    return {"gap_R2_diff": g, "controls_pass": [rA["controls_pass"], rB["controls_pass"]], "n": n}


pos = {"arm_gap_R2": {}, "pairs": {}}
for tag in ("G_s0", "G_s1", "P_s0", "P_s1", "PR_s0", "PR_s1", "PE_s0", "PE_s1", "PF_s0", "RES_s0", "RR_s0", "HIER_s0", "HIER_s1", "HIER_s2", "HIER_s3", "HIER_s4", "HIER_s5"):
    r = run(tag)
    pos["arm_gap_R2"][tag] = {"gap_R2_state": r["main"]["r_a"]["estimates"]["gap|s"], "controls_pass": r["controls_pass"], "file": "%s/RESULT_S1_%s.json" % (S1, tag)}
for a, b in (("PR_s0", "G_s0"), ("PR_s1", "G_s1"), ("PR_s0", "P_s0"), ("PR_s1", "P_s1"), ("PE_s0", "P_s0"), ("PE_s1", "P_s1"), ("PF_s0", "P_s0"), ("RES_s0", "G_s0"), ("RR_s0", "G_s0")):
    pos["pairs"]["%s_minus_%s" % (a, b)] = pair(a, b)
    print("pair", a, b, pos["pairs"]["%s_minus_%s" % (a, b)]["gap_R2_diff"], flush=True)
# spreads of the baseline
hier = [run("HIER_s%d" % i) for i in range(6)]
sp6 = S.spread_of(hier)
sp3 = S.spread_of(hier[:3])
sp2 = S.spread_of(hier[:2])
pos["spread_six"] = sp6
pos["spread_three_banked_C2"] = sp3
pos["spread_two"] = sp2
pos["spread_files"] = {"three": S1 + "/SPREAD_S1_HIER.json", "two": S1 + "/SPREAD_S1_HIER_2seeds.json", "six": "computed here from RESULT_S1_HIER_s0..s5.json by s1_read.spread_of"}
R["position_read"] = pos
json.dump(R, open(OUT + "/RESULT_MOTION_TARGET_FINAL.json", "w", encoding="utf-8"), indent=1, default=float)
print("written")
