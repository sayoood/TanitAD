#!/usr/bin/env python3
"""POST-HOC / DESCRIPTIVE diagnosis: WHY does refcv8's shipped stage (R8_RT, EMA-48,852) score ~10 PDMS below its own model-free prior
(PRIOR_ha0p) on the NavSim v1 navtrain selection sample (2,089 tokens, 30 Las Vegas logs, ONE training seed, inference seeds s0/s1)?

NOT pre-registered, decides nothing.  Pure CPU, numpy only.  Reads only the banked scorer rows / seam npz / inputs under RUN (read-only).

  python prior_gap_diag.py [RUN_DIR] [OUT_DIR]

Estimator, stated once: every interval is an INDEPENDENT LOG-CLUSTER BOOTSTRAP (cluster = log_name, 30 clusters, B = 2000, seed 7,
common draws for all cells), PAIRED on tokens, ratio-of-sums within a cell.  It answers "would another draw of LOGS say this?" -- it does
NOT answer "would another TRAINING run say this?" (one training seed) nor "would another INFERENCE run?" (s0 vs s1 reported beside).
"""
import csv
import gzip
import itertools
import json
import os
import sys
import collections

import numpy as np

RUN = sys.argv[1] if len(sys.argv) > 1 else "C:/Users/Admin/refcv8_pf/navsim_ckpt/run"
OUT = sys.argv[2] if len(sys.argv) > 2 else "C:/Users/Admin/refcv8_pf/navsim_ckpt/prior_gap_diag"
SEL = os.path.join(RUN, "navtrain_sel")
BR = os.path.join(SEL, "ema48852", "bridge")
B_BOOT, SEED = 2000, 7
EPS = 1e-9
LABEL = ("POST-HOC / DESCRIPTIVE; NavSim v1 open-loop PDM scorer on the navtrain selection sample (2,089 tokens, 30 logs, all us-nv-las-vegas-strip); "
         "refcv8-r101-s0 EMA-48,852 = ONE training seed; inference seeds s0/s1; NOT pre-registered; decides nothing")

# ----------------------------------------------------------------------------------------------------------------- load
inp = json.load(gzip.open(os.path.join(RUN, "inputs", "navtrain_sample_inputs.json.gz")))["tokens"]
toks = sorted(inp)
N = len(toks)
tix = {t: i for i, t in enumerate(toks)}
logs = sorted({inp[t]["log_name"] for t in toks})
lix = {l: i for i, l in enumerate(logs)}
cid = np.array([lix[inp[t]["log_name"]] for t in toks])
NL = len(logs)
cnt_log = np.bincount(cid, minlength=NL)

COLS = {"NC": "no_at_fault_collisions", "DAC": "drivable_area_compliance", "DDC": "driving_direction_compliance",
        "TTC": "time_to_collision_within_bound", "C": "comfort", "EP": "ego_progress", "PDMS": "score"}


def read_csv(path):
    with open(path, newline="") as fh:
        return {r["token"]: r for r in csv.DictReader(fh) if r["token"] != "average"}


SC = lambda lab: os.path.join(SEL, "scores", lab, lab + ".csv")
DG = lambda lab: os.path.join(SEL, "diag", "scores", lab, lab + ".csv")
ARM_CSV = {"RT_s0": SC("r8sel_ema48852_s0"), "RT_s1": SC("r8sel_ema48852_s1"),
           "LEGAL_s0": DG("r8sel_diag_ema48852_LEGAL_s0"), "LEGAL_s1": DG("r8sel_diag_ema48852_LEGAL_s1"),
           "PRIOR": SC("r8sel_ref_PRIOR_ha0p"), "STOP": SC("r8sel_ref_STOP"), "CV": SC("r8sel_ref_CV")}
V = {}
for k, p in ARM_CSV.items():
    t = read_csv(p)
    assert set(t) == set(toks) and all(t[x]["valid"] == "True" for x in toks), f"{k}: token set / validity"
    V[k] = {m: np.array([float(t[x][c]) for x in toks]) for m, c in COLS.items()}


def seam(name):
    z = np.load(os.path.join(BR, f"seam_{name}.npz"), allow_pickle=False)
    idx = np.array([list(z["token"]).index(t) for t in toks]) if list(z["token"]) != toks else np.arange(N)
    return {"poses": z["poses"][idx].astype(np.float64), "knots": z["knots"][idx].astype(np.float64)}


SEAM = {"RT_s0": seam("R8_RT"), "RT_s1": seam("R8_RT_s1"), "LEGAL_s0": seam("R8_LEGAL"), "LEGAL_s1": seam("R8_LEGAL_s1"), "PRIOR": seam("PRIOR_ha0p")}


def rows(name):
    out = {}
    for ln in open(os.path.join(BR, f"rows_{name}.jsonl")):
        if ln.strip():
            r = json.loads(ln)
            out[r["token"]] = r
    return out


ROWS_L0, ROWS_L1, ROWS_P = rows("R8_LEGAL"), rows("R8_LEGAL_s1"), rows("PRIOR_ha0p")
v0 = np.array([np.hypot(*inp[t]["ego_statuses"][-1]["ego_velocity"]) for t in toks])
a0 = np.array([ROWS_P[t]["a0"] for t in toks])
cmd_vec = np.array([inp[t]["ego_statuses"][-1]["driving_command"] for t in toks])
CMD_NAMES = ["left", "straight", "right"]
cmd = np.argmax(cmd_vec[:, :3], axis=1)
assert (cmd_vec[:, 3] == 0).all() and (cmd_vec.sum(1) == 1).all()
human = np.array([inp[t]["human_future_poses"] for t in toks], np.float64)          # [N,8,3] ego frame t0, 0.5..4.0 s
sel0 = np.array([ROWS_L0[t]["diag"]["sel_idx"] for t in toks])
sel1 = np.array([ROWS_L1[t]["diag"]["sel_idx"] for t in toks])
SPEED_EDGES = [0.0, 2.0, 8.33, 13.89, 1e9]
SPEED_NAMES = ["[0,2)", "[2,8.33)", "[8.33,13.89)", ">=13.89"]
band = np.digitize(v0, SPEED_EDGES[1:-1])


def pdms_formula(m):
    """MEASURED on every banked CSV (max |err| 2.2e-16): this scorer's PDMS = NC * DAC * (5 TTC + 2 C + 5 EP) / 12.  DDC is REPORTED but is NOT in
    the score (a formula that multiplies it fails on 27/2089 PRIOR tokens, DDC in {0, 0.5})."""
    return m["NC"] * m["DAC"] * (5 * m["TTC"] + 2 * m["C"] + 5 * m["EP"]) / 12.0


# ------------------------------------------------------------------------------------------------------------ bootstrap
rng0 = np.random.default_rng(SEED)
DRAWS = rng0.integers(0, NL, (B_BOOT, NL))          # common random numbers for every cell


def boot_cell(d, mask=None, scale=100.0, unit="x100"):
    """Mean of d over the mask (ratio of sums), 95 % percentile CI over log-cluster draws; cells with no token in a draw are dropped."""
    mask = np.ones(N, bool) if mask is None else mask
    s = np.bincount(cid, weights=d * mask, minlength=NL)
    c = np.bincount(cid, weights=mask.astype(float), minlength=NL)
    num, den = s[DRAWS].sum(1), c[DRAWS].sum(1)
    ok = den > 0
    bs = num[ok] / den[ok]
    n = int(mask.sum())
    out = {"n": n, "n_logs": int((c > 0).sum())}
    if n == 0:
        out.update({f"mean_{unit}": None, f"ci95_{unit}": None, "boot_valid_draws": 0})
        return out
    lo, hi = np.percentile(bs, [2.5, 97.5])
    out.update({f"mean_{unit}": round(scale * float(d[mask].mean()), 3), f"ci95_{unit}": [round(scale * float(lo), 3), round(scale * float(hi), 3)],
                "boot_valid_draws": int(ok.sum())})
    return out


def boot_contrib(d, mask, scale=100.0):
    """Contribution of the masked tokens to the OVERALL mean (sum over mask / N), denominators = all tokens of the drawn logs."""
    s = np.bincount(cid, weights=d * mask, minlength=NL)
    num, den = s[DRAWS].sum(1), cnt_log[DRAWS].sum(1)
    bs = num / den
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"n": int(mask.sum()), "contrib_x100": round(scale * float(d[mask].sum() / N), 3),
            "ci95_x100": [round(scale * float(lo), 3), round(scale * float(hi), 3)]}


def ci_width(ci):
    return ci[1] - ci[0]


# =================================================================================================================== CONTROLS
ctl = {}
# C1 PRIOR - PRIOR = 0 exactly
d_pp = V["PRIOR"]["PDMS"] - V["PRIOR"]["PDMS"]
c1 = boot_cell(d_pp)
ctl["C1_PRIOR_minus_PRIOR"] = {"result": c1, "pass": bool(c1["mean_x100"] == 0.0 and c1["ci95_x100"] == [0.0, 0.0] and float(np.abs(d_pp).max()) == 0.0)}
# C2 means on the 2,089 tokens (x100)
want = {"RT_s0": 63.40, "RT_s1": 62.73, "PRIOR": 73.84, "STOP": 67.57, "CV": 15.80}
mm = {k: round(100 * float(V[k]["PDMS"].mean()), 3) for k in V}
ctl["C2_means_x100"] = {"recomputed": mm, "brief": want, "n": N,
                        "pass": all(abs(mm[k] - w) <= 0.0051 + (0.0051 if k == "RT_s1" else 0) for k, w in want.items())}
# C3 the PDMS formula reproduces the scorer's own per-token score from the six sub-scores
ctl["C3_pdms_formula_vs_csv_max_abs"] = {k: float(np.abs(pdms_formula(V[k]) - V[k]["PDMS"]).max()) for k in V}
ctl["C3_pdms_formula_vs_csv_max_abs"]["pass"] = bool(max(v for k, v in ctl["C3_pdms_formula_vs_csv_max_abs"].items()) < 1e-9)
# C4 the seam poses are the poses the scorer was given (hooks agent_poses), for RT_s0 and PRIOR
hk = {}
for lab, key in (("r8sel_ema48852_s0", "RT_s0"), ("r8sel_ref_PRIOR_ha0p", "PRIOR")):
    h = json.load(open(os.path.join(SEL, "scores", lab, lab + "_hooks.json")))["pdm_score_calls"]
    mx = max(float(np.abs(np.array(c["agent_poses"]) - SEAM[key]["poses"][tix[c["token"]]]).max()) for c in h)
    hk[key] = {"n_calls": len(h), "max_abs_pose_diff": mx}
ctl["C4_seam_equals_scored_agent_poses"] = {**hk, "pass": all(v["max_abs_pose_diff"] < 1e-9 and v["n_calls"] == N for v in hk.values())}
# C5 token-shuffled pairing: asked to WIDEN the gap CI.  Reported under BOTH estimators, because under a log-cluster ratio-of-sums bootstrap a
#    global token shuffle only re-allocates log sums and need not widen (it widens iff Cov_logs(RT_l, PRIOR_l) > Var_logs(PRIOR_l)/2).
def iid_width(d, B=2000, seed=11):
    r = np.random.default_rng(seed)
    bs = np.array([d[r.integers(0, N, N)].mean() for _ in range(B)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return 100 * float(hi - lo)


d_rt = V["RT_s0"]["PDMS"] - V["PRIOR"]["PDMS"]
paired = boot_cell(d_rt)
paired_w = ci_width(paired["ci95_x100"])
widths, iid_sh = [], []
for sd in range(20):
    perm = np.random.default_rng(1000 + sd).permutation(N)
    dsh = V["RT_s0"]["PDMS"] - V["PRIOR"]["PDMS"][perm]
    widths.append(ci_width(boot_cell(dsh)["ci95_x100"]))
    if sd < 5:
        iid_sh.append(iid_width(dsh))
lm_rt = np.bincount(cid, weights=V["RT_s0"]["PDMS"], minlength=NL) / cnt_log
lm_pr = np.bincount(cid, weights=V["PRIOR"]["PDMS"], minlength=NL) / cnt_log
iid_paired = iid_width(d_rt)
ctl["C5_shuffled_pairing_widens_CI"] = {
    "log_cluster_bootstrap": {"paired_ci_width_x100": round(paired_w, 3), "shuffled_median_of_20": round(float(np.median(widths)), 3),
                              "shuffled_min": round(float(np.min(widths)), 3), "widens": bool(np.min(widths) > paired_w)},
    "token_iid_bootstrap_(pairing_sensitivity_only,_not_a_valid_interval)": {"paired_ci_width_x100": round(iid_paired, 3),
                                                                          "shuffled_median_of_5": round(float(np.median(iid_sh)), 3),
                                                                          "widens": bool(np.min(iid_sh) > iid_paired)},
    "token_level_corr_RT_vs_PRIOR_pdms": round(float(np.corrcoef(V["RT_s0"]["PDMS"], V["PRIOR"]["PDMS"])[0, 1]), 4),
    "log_level_corr_of_means_RT_vs_PRIOR": round(float(np.corrcoef(lm_rt, lm_pr)[0, 1]), 4),
    "log_level_var_ratio_PRIOR_over_RT": round(float(lm_pr.var() / lm_rt.var()), 3),
    "reading": "AS WRITTEN (widens under the cluster estimator) this control FAILS: the log-cluster ratio-of-sums bootstrap sees only per-log sums, and pairing "
               "matters through the log-level covariance, which is weak here (see log_level_corr). It widens under the token-iid bootstrap, which proves the "
               "pairing carries token-level signal; that estimator is shown ONLY for that purpose and is not a valid interval (ignores log clustering).",
    "pass_token_iid_variant": bool(np.min(iid_sh) > iid_paired), "pass": bool(np.min(widths) > paired_w)}   # "pass" = AS WRITTEN (cluster estimator): FALSE, reported loudly
# C6 human future sanity: mean along-track speed over 4 s tracks v0 (a correct frame/units reading)
hx4 = human[:, 7, 0] / 4.0
ctl["C6_human_future_sanity"] = {"corr_v0_vs_human_x4_over_4s": round(float(np.corrcoef(v0, hx4)[0, 1]), 4),
                                 "median_human_x4_over_4s_minus_v0": round(float(np.median(hx4 - v0)), 3), "pass": bool(np.corrcoef(v0, hx4)[0, 1] > 0.8)}

# ================================================================================================================= Q1 where
Q1 = {"estimator": "independent log-cluster bootstrap (cluster = log_name, 30), B 2000, seed 7, paired on tokens, ratio-of-sums per cell; "
                   "answers 'another draw of LOGS', NOT 'another training run' / 'another inference run'",
      "headline": {}, "speed_band": {}, "command": {}, "speed_x_command": {}, "subscore_flip_counts": {}, "failure_class_exclusive": {},
      "failure_signature_top": []}
gap = {k: V[k]["PDMS"] - V["PRIOR"]["PDMS"] for k in ("RT_s0", "RT_s1", "LEGAL_s0", "LEGAL_s1")}
gap["RT_avg"] = 0.5 * (gap["RT_s0"] + gap["RT_s1"])
TOT = float(gap["RT_s0"].sum())
Q1["headline"] = {"RT_s0_minus_PRIOR": boot_cell(gap["RT_s0"]), "RT_s1_minus_PRIOR": boot_cell(gap["RT_s1"]),
                  "RT_avg_s0s1_minus_PRIOR": boot_cell(gap["RT_avg"]), "LEGAL_s0_minus_PRIOR": boot_cell(gap["LEGAL_s0"]),
                  "LEGAL_s1_minus_PRIOR": boot_cell(gap["LEGAL_s1"]), "RT_s0_minus_LEGAL_s0": boot_cell(V["RT_s0"]["PDMS"] - V["LEGAL_s0"]["PDMS"]),
                  "RT_s0_minus_STOP": boot_cell(V["RT_s0"]["PDMS"] - V["STOP"]["PDMS"]), "PRIOR_minus_STOP": boot_cell(V["PRIOR"]["PDMS"] - V["STOP"]["PDMS"]),
                  "per_arm_mean_x100": mm}


def cell_row(mask):
    r = boot_cell(gap["RT_s0"], mask)
    if r["n"]:
        r["prior_mean_x100"] = round(100 * float(V["PRIOR"]["PDMS"][mask].mean()), 3)
        r["rt_s0_mean_x100"] = round(100 * float(V["RT_s0"]["PDMS"][mask].mean()), 3)
        r["gap_s1_x100"] = round(100 * float(gap["RT_s1"][mask].mean()), 3)
        r["gap_legal_s0_x100"] = round(100 * float(gap["LEGAL_s0"][mask].mean()), 3)
        r["share_of_total_gap"] = round(float(gap["RT_s0"][mask].sum() / TOT), 4)
        r["contrib_to_overall_gap_x100"] = round(100 * float(gap["RT_s0"][mask].sum() / N), 3)
        r["frac_tokens_rt_worse"] = round(float((gap["RT_s0"][mask] < -EPS).mean()), 4)
        r["frac_tokens_rt_better"] = round(float((gap["RT_s0"][mask] > EPS).mean()), 4)
        if r["n"] < 30 or r["n_logs"] < 5:
            r["UNDERPOWERED"] = f"n={r['n']}, logs={r['n_logs']}: interval is not decision-grade"
    return r


for b, nm in enumerate(SPEED_NAMES):
    Q1["speed_band"][nm] = cell_row(band == b)
for c, nm in enumerate(CMD_NAMES):
    Q1["command"][nm] = cell_row(cmd == c)
for b, nm in enumerate(SPEED_NAMES):
    for c, cn in enumerate(CMD_NAMES):
        Q1["speed_x_command"][f"{nm} x {cn}"] = cell_row((band == b) & (cmd == c))

# --- sub-score flips PRIOR -> RT_s0
SUBS = ["NC", "DAC", "TTC", "C", "EP", "DDC"]      # DDC last: reported by the scorer but NOT part of PDMS
EP_TOL = 0.05
for m in SUBS:
    dm = V["RT_s0"][m] - V["PRIOR"][m]
    tol = EP_TOL if m == "EP" else EPS
    worse, better = dm < -tol, dm > tol
    same = ~(worse | better)
    Q1["subscore_flip_counts"][m] = {
        "rule": ("EP: |delta| > %.2f counts as a change" % EP_TOL) if m == "EP" else ("any decrease/increase; DDC is NOT in this scorer's PDMS (informational)" if m == "DDC" else "any decrease/increase"),
        "mean_subscore_delta_x100": round(100 * float(dm.mean()), 3),
        "RT_worse": {"n": int(worse.sum()), "mean_token_pdms_gap_x100": round(100 * float(gap["RT_s0"][worse].mean()), 3) if worse.any() else None,
                     "contrib_to_overall_gap": boot_contrib(gap["RT_s0"], worse)},
        "same": {"n": int(same.sum())},
        "RT_better": {"n": int(better.sum()), "mean_token_pdms_gap_x100": round(100 * float(gap["RT_s0"][better].mean()), 3) if better.any() else None,
                      "contrib_to_overall_gap": boot_contrib(gap["RT_s0"], better)}}
Q1["subscore_flip_counts"]["_note"] = ("NC / DAC / TTC / EP are NOT independent: in this scorer EP is zeroed on 100 % of NC=0 / DAC=0 tokens and TTC on most NC=0 tokens, "
                                       "so the flip rows OVERLAP and their contributions do not add -- see failure_class_exclusive for the additive decomposition.")

# discrete transitions PRIOR -> RT_s0 for the multipliers and TTC
Q1["transitions_prior_to_rt_s0_counts"] = {}
for m in ("NC", "DAC", "TTC"):
    c_ = collections.Counter((float(a), float(b)) for a, b in zip(V["PRIOR"][m], V["RT_s0"][m]))
    Q1["transitions_prior_to_rt_s0_counts"][m] = {f"{a:g}->{b:g}": n_ for (a, b), n_ in sorted(c_.items())}

# --- exclusive, additive failure classes (hierarchical on the multiplier product)
mult = {k: V[k]["NC"] * V[k]["DAC"] for k in ("PRIOR", "RT_s0")}
dm_ = mult["RT_s0"] - mult["PRIOR"]
cls = np.empty(N, object)
dd = gap["RT_s0"]
for i in range(N):
    if dm_[i] < -EPS:
        cls[i] = "A_RT_lost_multiplier"
    elif dm_[i] > EPS:
        cls[i] = "B_RT_fixed_multiplier"
    elif mult["PRIOR"][i] < EPS:
        cls[i] = "C_both_multiplier_zero"
    elif abs(dd[i]) <= EPS:
        cls[i] = "F_exact_tie"
    elif dd[i] < 0:
        cls[i] = "D_graded_loss_multipliers_intact"
    else:
        cls[i] = "E_graded_gain_multipliers_intact"
for k in sorted(set(cls)):
    m = cls == k
    r = boot_contrib(dd, m)
    r["mean_token_gap_x100"] = round(100 * float(dd[m].mean()), 3)
    r["share_of_net_gap"] = round(float(dd[m].sum() / TOT), 4)
    Q1["failure_class_exclusive"][k] = r
Q1["failure_class_exclusive"]["_sum_of_contribs_x100"] = round(sum(v["contrib_x100"] for k, v in Q1["failure_class_exclusive"].items() if not k.startswith("_")), 3)
Q1["failure_class_exclusive"]["_total_gap_x100"] = round(100 * TOT / N, 3)
# sub-split of class A by which multiplier(s) went down; of class D by which graded term(s) went down
sigs = collections.OrderedDict()
for i in range(N):
    if cls[i] == "A_RT_lost_multiplier":
        key = "A:" + "+".join(m for m in ("NC", "DAC") if V["RT_s0"][m][i] < V["PRIOR"][m][i] - EPS)
    elif cls[i] == "D_graded_loss_multipliers_intact":
        key = "D:" + "+".join([m for m in ("TTC", "C") if V["RT_s0"][m][i] < V["PRIOR"][m][i] - EPS] + (["EP"] if V["RT_s0"]["EP"][i] < V["PRIOR"]["EP"][i] - EP_TOL else [])) or "D:small"
        if key == "D:":
            key = "D:EP<=0.05"
    else:
        continue
    sigs.setdefault(key, []).append(i)
for key, ix in sorted(sigs.items(), key=lambda kv: dd[kv[1]].sum()):
    m = np.zeros(N, bool)
    m[ix] = True
    r = boot_contrib(dd, m)
    r.update(signature=key, mean_token_gap_x100=round(100 * float(dd[m].mean()), 3), share_of_net_gap=round(float(dd[m].sum() / TOT), 4))
    Q1["failure_signature_top"].append(r)

# additive classes WITHIN each command: contribution of the class to the OVERALL gap (sums over commands to the class row above)
Q1["failure_class_by_command_contrib_x100"] = {}
for c, cn in enumerate(CMD_NAMES):
    row = {}
    for k in sorted(set(cls)):
        m = (cls == k) & (cmd == c)
        r = boot_contrib(dd, m)
        row[k] = {"n": r["n"], "contrib_x100": r["contrib_x100"], "ci95_x100": r["ci95_x100"]}
    Q1["failure_class_by_command_contrib_x100"][cn] = row

# ================================================================================================================== Q2 plan error
Q2 = {"note": "NavSim poses at 0.5..4.0 s in the ego frame at t0 (x fwd, y left). Human = logged ego future from the inputs file. "
              "Thresholds below are committed here, post-hoc, and descriptive."}
P = {k: SEAM[k]["poses"] for k in SEAM}
IDX2, IDX4 = 3, 7                                   # 2.0 s, 4.0 s


def arc_len(p):                                     # cumulative chord length through the origin: [N,8]
    seg = np.diff(np.concatenate([np.zeros((p.shape[0], 1, 2)), p[:, :, :2]], 1), axis=1)
    return np.cumsum(np.hypot(seg[..., 0], seg[..., 1]), axis=1)


S_RT, S_PR, S_HU, S_LG = arc_len(P["RT_s0"]), arc_len(P["PRIOR"]), arc_len(human), arc_len(P["LEGAL_s0"])
lose = dd < -EPS
win = dd > EPS
ties = ~(lose | win)


def stat(x, m):
    if not m.any():
        return None
    return {"n": int(m.sum()), "median": round(float(np.median(x[m])), 3), "mean": round(float(x[m].mean()), 3),
            "p10": round(float(np.percentile(x[m], 10)), 3), "p90": round(float(np.percentile(x[m], 90)), 3)}


feat = {"dx2_RT_minus_PRIOR_m": P["RT_s0"][:, IDX2, 0] - P["PRIOR"][:, IDX2, 0], "dx4_RT_minus_PRIOR_m": P["RT_s0"][:, IDX4, 0] - P["PRIOR"][:, IDX4, 0],
        "dy2_RT_minus_PRIOR_m": P["RT_s0"][:, IDX2, 1] - P["PRIOR"][:, IDX2, 1], "dy4_RT_minus_PRIOR_m": P["RT_s0"][:, IDX4, 1] - P["PRIOR"][:, IDX4, 1],
        "dh4_RT_minus_PRIOR_rad": P["RT_s0"][:, IDX4, 2] - P["PRIOR"][:, IDX4, 2],
        "dx2_RT_minus_HUMAN_m": P["RT_s0"][:, IDX2, 0] - human[:, IDX2, 0], "dx4_RT_minus_HUMAN_m": P["RT_s0"][:, IDX4, 0] - human[:, IDX4, 0],
        "dx2_PRIOR_minus_HUMAN_m": P["PRIOR"][:, IDX2, 0] - human[:, IDX2, 0], "dx4_PRIOR_minus_HUMAN_m": P["PRIOR"][:, IDX4, 0] - human[:, IDX4, 0],
        "dy4_RT_minus_HUMAN_m": P["RT_s0"][:, IDX4, 1] - human[:, IDX4, 1], "dy4_PRIOR_minus_HUMAN_m": P["PRIOR"][:, IDX4, 1] - human[:, IDX4, 1],
        "arc4_RT_over_PRIOR": S_RT[:, IDX4] / np.maximum(S_PR[:, IDX4], 0.05), "arc4_RT_over_HUMAN": S_RT[:, IDX4] / np.maximum(S_HU[:, IDX4], 0.05),
        "arc4_PRIOR_over_HUMAN": S_PR[:, IDX4] / np.maximum(S_HU[:, IDX4], 0.05), "arc4_LEGAL_over_PRIOR": S_LG[:, IDX4] / np.maximum(S_PR[:, IDX4], 0.05)}
Q2["feature_distributions"] = {k: {"RT_loses": stat(v, lose), "RT_wins": stat(v, win), "tie": stat(v, ties), "all": stat(v, np.ones(N, bool))} for k, v in feat.items()}


def ade(p, q):
    return np.hypot(p[:, :, 0] - q[:, :, 0], p[:, :, 1] - q[:, :, 1]).mean(1)


ade_h = {k: ade(P[k], human) for k in ("RT_s0", "PRIOR", "LEGAL_s0")}
ade_h["STOP"] = np.hypot(human[:, :, 0], human[:, :, 1]).mean(1)
cvp = np.array([inp[t]["cv_poses"] for t in toks], np.float64)
ade_h["CV"] = ade(cvp, human)
fde_h = {k: np.hypot(P[k][:, 7, 0] - human[:, 7, 0], P[k][:, 7, 1] - human[:, 7, 1]) for k in ("RT_s0", "PRIOR", "LEGAL_s0")}
Q2["human_proximity"] = {
    "note": "L2 to the logged human future is NOT the metric; included to show whether the plan that scores worse is also farther from the human",
    "ade_to_human_m": {k: boot_cell(v, scale=1.0, unit="m") for k, v in ade_h.items()},
    "RT_minus_PRIOR_ade_to_human_m": boot_cell(ade_h["RT_s0"] - ade_h["PRIOR"], scale=1.0, unit="m"),
    "RT_minus_PRIOR_fde4s_to_human_m": boot_cell(fde_h["RT_s0"] - fde_h["PRIOR"], scale=1.0, unit="m"),
    "RT_closer_to_human_than_PRIOR_frac_all": round(float((ade_h["RT_s0"] < ade_h["PRIOR"]).mean()), 4),
    "RT_closer_to_human_than_PRIOR_frac_losers": round(float((ade_h["RT_s0"][lose] < ade_h["PRIOR"][lose]).mean()), 4),
    "RT_closer_to_human_than_PRIOR_frac_winners": round(float((ade_h["RT_s0"][win] < ade_h["PRIOR"][win]).mean()), 4)}


# lateral PATH deviation of the RT knots from the PRIOR polyline (+ its end ray): is the RT PATH the prior's path (so the loss is progress), or a different path?
def dist_to_polyline(pt, poly):                         # pt [N,2]; poly [N,K,2] vertices; min distance to the segments
    best = np.full(pt.shape[0], np.inf)
    for k in range(poly.shape[1] - 1):
        a, b = poly[:, k], poly[:, k + 1]
        ab = b - a
        l2 = (ab ** 2).sum(1)
        t = np.where(l2 > 0, ((pt - a) * ab).sum(1) / np.where(l2 > 0, l2, 1.0), 0.0)
        t = np.clip(t, 0, 1)
        proj = a + t[:, None] * ab
        best = np.minimum(best, np.hypot(*(pt - proj).T))
    return best


def extended_polyline(knots):                          # origin + knots + a 200 m extension along the last positive-length segment
    n = knots.shape[0]
    V_ = np.concatenate([np.zeros((n, 1, 2)), knots], 1)
    seg = V_[:, 1:] - V_[:, :-1]
    ln = np.hypot(seg[..., 0], seg[..., 1])
    pos = ln > 0
    last = np.where(pos.any(1), 7 - np.argmax(pos[:, ::-1], 1), 0)
    u = np.where(pos[np.arange(n), last][:, None], seg[np.arange(n), last] / np.where(pos[np.arange(n), last], ln[np.arange(n), last], 1.0)[:, None], np.array([[1.0, 0.0]]))
    return np.concatenate([V_, (V_[:, -1] + 200.0 * u)[:, None]], 1)


kn_rt, kn_pr = SEAM["RT_s0"]["knots"], SEAM["PRIOR"]["knots"]
kn_lg = SEAM["LEGAL_s0"]["knots"]
poly_pr = extended_polyline(kn_pr)
path_dev = np.stack([dist_to_polyline(kn_rt[:, k], poly_pr) for k in range(6)], 1).max(1)       # knots 0.5..4 s
path_dev_leg = np.stack([dist_to_polyline(kn_lg[:, k], poly_pr) for k in range(6)], 1).max(1)
Q2["path_vs_progress"] = {
    "definition": "max over the 6 knots (0.5..4 s) of the distance from the plan knot to the PRIOR polyline extended by its end ray: ~0 means the plan "
                  "follows the prior's PATH and only differs in progress along it",
    "path_dev_RT_vs_PRIOR_m": {"RT_loses": stat(path_dev, lose), "RT_wins": stat(path_dev, win), "all": stat(path_dev, np.ones(N, bool))},
    "frac_losers_path_within_0p3m": round(float((path_dev[lose] < 0.3).mean()), 4),
    "frac_losers_path_within_1m": round(float((path_dev[lose] < 1.0).mean()), 4),
    "frac_losers_path_beyond_2m": round(float((path_dev[lose] > 2.0).mean()), 4),
    "RT_path_equals_LEGAL_path_check_max_dev_m": None}
# RT keeps LEGAL's path by construction: verify, to justify using LEGAL as the path source
Q2["path_vs_progress"]["RT_path_equals_LEGAL_path_check_max_dev_m"] = round(float(max(
    dist_to_polyline(kn_rt[:, k], extended_polyline(kn_lg)).max() for k in range(8))), 6)

# exclusive plan-error mode for losers (committed thresholds: 0.5 m on the 4 s knot, lateral wins if its |dy| exceeds |dx|)
dx4, dy4 = feat["dx4_RT_minus_PRIOR_m"], feat["dy4_RT_minus_PRIOR_m"]
mode = np.empty(N, object)
for i in range(N):
    if max(abs(dx4[i]), path_dev[i]) < 0.5:
        mode[i] = "near_identical_lt0.5m"
    elif path_dev[i] >= abs(dx4[i]) and path_dev[i] >= 0.5:
        mode[i] = "steers_differently(path)"
    elif dx4[i] <= -0.5:
        mode[i] = "slower_than_prior"
    else:
        mode[i] = "faster_than_prior"
Q2["plan_mode_losers"] = {}
for k in sorted(set(mode)):
    m = lose & (mode == k)
    r = boot_contrib(dd, m)
    r["frac_of_losers"] = round(float(m.sum() / max(lose.sum(), 1)), 4)
    r["n_all_tokens_with_mode"] = int((mode == k).sum())
    r["mean_gap_all_tokens_with_mode_x100"] = round(100 * float(dd[mode == k].mean()), 3) if (mode == k).any() else None
    r["share_of_net_gap_from_losers_in_mode"] = round(float(dd[m].sum() / TOT), 4)
    Q2["plan_mode_losers"][k] = r
# plan mode x failure class cross-tab for losers (counts)
xt = collections.defaultdict(lambda: collections.Counter())
for i in np.where(lose)[0]:
    xt[cls[i]][mode[i]] += 1
Q2["plan_mode_x_failure_class_losers_counts"] = {k: dict(v) for k, v in xt.items()}
# mode for ALL tokens by class -> gives the base rate of 'slower' etc. among non-losers
Q2["plan_mode_all_tokens_counts"] = dict(collections.Counter(mode))
# net gap by plan-geometry bin over ALL tokens (not only losers): where does the learned stage lose / gain?
pd_edges = [0.0, 0.3, 1.0, 2.0, 1e9]
pd_names = ["path_dev<0.3m", "0.3-1m", "1-2m", ">=2m"]
pb = np.digitize(path_dev, pd_edges[1:-1])
Q2["gap_by_path_dev_bin_all_tokens"] = {nm: cell_row(pb == i) for i, nm in enumerate(pd_names)}
lr = np.log(np.maximum(feat["arc4_RT_over_PRIOR"], 0.05))
ar_edges = [-0.5, -0.15, 0.15, 0.5]       # log ratio of 4 s arc length RT/PRIOR
ab = np.digitize(lr, ar_edges)
ar_names = ["RT arc <0.61x prior", "0.61-0.86x", "0.86-1.16x (similar)", "1.16-1.65x", ">1.65x prior"]
Q2["gap_by_arc4_ratio_bin_all_tokens"] = {nm: cell_row(ab == i) for i, nm in enumerate(ar_names)}
Q2["gap_by_mode_all_tokens"] = {k: cell_row(mode == k) for k in sorted(set(mode))}
Q2["mode_by_command_all_tokens_counts"] = {cn: dict(collections.Counter(mode[cmd == c])) for c, cn in enumerate(CMD_NAMES)}
Q2["gap_by_mode_within_command"] = {cn: {k: cell_row((mode == k) & (cmd == c)) for k in sorted(set(mode))} for c, cn in enumerate(CMD_NAMES)}
# vs HUMAN: is the losing plan slower or faster than what the human actually did?
dxh = feat["dx4_RT_minus_HUMAN_m"]
hmode = np.where(dxh <= -1.0, "slower_than_human_gt1m", np.where(dxh >= 1.0, "faster_than_human_gt1m", "within_1m_of_human"))
Q2["plan_vs_human_4s_along_track_losers"] = {}
for k in ("slower_than_human_gt1m", "within_1m_of_human", "faster_than_human_gt1m"):
    m = lose & (hmode == k)
    r = boot_contrib(dd, m)
    r["frac_of_losers"] = round(float(m.sum() / max(lose.sum(), 1)), 4)
    r["base_rate_all_tokens"] = round(float((hmode == k).mean()), 4)
    r["mean_gap_all_tokens_with_label_x100"] = round(100 * float(dd[hmode == k].mean()), 3)
    Q2["plan_vs_human_4s_along_track_losers"][k] = r
Q2["plan_vs_human_x_failure_class_losers_counts"] = {c_: dict(collections.Counter(hmode[(cls == c_) & lose])) for c_ in ("A_RT_lost_multiplier", "D_graded_loss_multipliers_intact")}
Q2["human_future_has_no_lead_agent_information"] = ("The inputs file carries ego_status + the logged ego future only; agent tracks live in the Thor metric cache. "
                                                    "'Accelerates into the lead' vs 'brakes and is hit' is therefore NOT resolvable here -- only slower/faster than the prior and the human (UNVERIFIED mechanism).")
# per speed band: progress ratio RT/PRIOR at 4 s (losers vs all)
Q2["arc4_RT_over_PRIOR_by_speed_band"] = {nm: {"all": stat(feat["arc4_RT_over_PRIOR"], band == b), "losers": stat(feat["arc4_RT_over_PRIOR"], (band == b) & lose)}
                                          for b, nm in enumerate(SPEED_NAMES)}
# anchor: index only (anchor table is not local)
sel_dist = collections.Counter(sel0)
top_anchor = [a for a, _ in sel_dist.most_common(8)]
Q2["anchor"] = {"note": "refcv8 anchor TABLE is not on this machine, so anchors are compared by INDEX only (UNVERIFIED geometry); the prior has no anchor",
                "n_distinct_sel_idx_s0": len(sel_dist), "n_distinct_sel_idx_s1": len(collections.Counter(sel1)),
                "s0_s1_same_anchor_frac": round(float((sel0 == sel1).mean()), 4),
                "s0_s1_same_anchor_frac_losers": round(float((sel0[lose] == sel1[lose]).mean()), 4),
                "top_anchors_s0": [{"sel_idx": int(a), "n": int(sel_dist[a]), "mean_gap_x100": round(100 * float(dd[sel0 == a].mean()), 3),
                                    "mean_v0": round(float(v0[sel0 == a].mean()), 2)} for a in top_anchor]}
# inference-seed self-consistency of the plan
dseed = np.hypot(P["RT_s0"][:, IDX4, 0] - P["RT_s1"][:, IDX4, 0], P["RT_s0"][:, IDX4, 1] - P["RT_s1"][:, IDX4, 1])
Q2["inference_seed_plan_disagreement_4s_m"] = {"losers": stat(dseed, lose), "winners": stat(dseed, win), "all": stat(dseed, np.ones(N, bool))}

# ================================================================================================================ Q3 concentration
Q3 = {}
order = np.argsort(dd)                                  # most negative first
neg_sum = float(dd[dd < -EPS].sum())
pos_sum = float(dd[dd > EPS].sum())
Q3["net"] = {"n": N, "sum_gap": round(TOT, 4), "sum_negative": round(neg_sum, 4), "sum_positive": round(pos_sum, 4),
             "frac_tokens_worse": round(float(lose.mean()), 4), "frac_tokens_better": round(float(win.mean()), 4), "frac_tokens_tie": round(float(ties.mean()), 4)}
lor = {}
for q in (0.01, 0.05, 0.10, 0.20, 0.50):
    k = int(round(q * N))
    lor[f"worst_{int(q * 100)}pct"] = {"n": k, "sum_gap": round(float(dd[order[:k]].sum()), 4), "share_of_NET_gap": round(float(dd[order[:k]].sum() / TOT), 4),
                                         "share_of_total_loss": round(float(dd[order[:k]].sum() / neg_sum), 4)}
Q3["lorenz"] = lor
# bootstrap of worst-10 % shares: resample LOGS, take their tokens, recompute
tok_by_log = [np.where(cid == l)[0] for l in range(NL)]
sh_net, sh_loss = [], []
for b in range(B_BOOT):
    ix = np.concatenate([tok_by_log[l] for l in DRAWS[b]])
    x = dd[ix]
    ko = int(round(0.10 * len(x)))
    xs = np.sort(x)[:ko]
    tot_b = x.sum()
    neg_b = x[x < -EPS].sum()
    sh_net.append(xs.sum() / tot_b if abs(tot_b) > 1e-9 else np.nan)
    sh_loss.append(xs.sum() / neg_b if neg_b < -1e-9 else np.nan)
sh_net, sh_loss = np.array(sh_net), np.array(sh_loss)
Q3["worst10pct_share_ci95"] = {"share_of_net_gap": [round(float(np.nanpercentile(sh_net, 2.5)), 3), round(float(np.nanpercentile(sh_net, 97.5)), 3)],
                               "share_of_total_loss": [round(float(np.nanpercentile(sh_loss, 2.5)), 3), round(float(np.nanpercentile(sh_loss, 97.5)), 3)],
                               "estimator": "log-cluster bootstrap re-ranking tokens inside each draw"}
# the PDMS values behind the loss: how many losers are 'collapse to 0' events
coll = (V["RT_s0"]["PDMS"] < 1e-9) & (V["PRIOR"]["PDMS"] > 0.2)
Q3["collapse_tokens_RT_pdms_zero_prior_gt_0p2"] = {"n": int(coll.sum()), "frac_of_all": round(float(coll.mean()), 4), "sum_gap": round(float(dd[coll].sum()), 4),
                                                    "share_of_total_loss": round(float(dd[coll].sum() / neg_sum), 4)}
# per-log table
per_log = []
for l in range(NL):
    m = cid == l
    per_log.append({"log": logs[l], "n": int(m.sum()), "prior_mean_x100": round(100 * float(V["PRIOR"]["PDMS"][m].mean()), 2),
                    "rt_s0_mean_x100": round(100 * float(V["RT_s0"]["PDMS"][m].mean()), 2), "gap_mean_x100": round(100 * float(dd[m].mean()), 2),
                    "gap_s1_mean_x100": round(100 * float(gap["RT_s1"][m].mean()), 2), "share_of_net_gap": round(float(dd[m].sum() / TOT), 4),
                    "mean_v0": round(float(v0[m].mean()), 2)})
Q3["per_log"] = sorted(per_log, key=lambda r: r["share_of_net_gap"], reverse=True)
lm = np.array([r["gap_mean_x100"] for r in per_log])
Q3["logs_with_negative_mean_gap"] = {"n_logs_negative": int((lm < 0).sum()), "n_logs": NL,
                                     "n_logs_negative_s1_replicate": int(sum(1 for r in per_log if r["gap_s1_mean_x100"] < 0)),
                                     "n_logs_negative_among_logs_with_n_ge_30": [int(sum(1 for r in per_log if r["n"] >= 30 and r["gap_mean_x100"] < 0)),
                                                                                int(sum(1 for r in per_log if r["n"] >= 30))]}
sh = np.array([r["share_of_net_gap"] for r in Q3["per_log"]])
Q3["top_logs_share_of_net_gap"] = {f"top{k}_logs": round(float(sh[:k].sum()), 4) for k in (1, 3, 5, 10)}
Q3["top_logs_share_of_net_gap"]["token_share_of_those_logs"] = {f"top{k}_logs": round(float(sum(r["n"] for r in Q3["per_log"][:k]) / N), 4) for k in (1, 3, 5, 10)}
# leave-one-log-out range of the mean gap
loo = [100 * float((TOT - dd[cid == l].sum()) / (N - (cid == l).sum())) for l in range(NL)]
Q3["leave_one_log_out_mean_gap_x100"] = {"min": round(min(loo), 3), "max": round(max(loo), 3), "full": round(100 * TOT / N, 3), "n_logs": NL}
# cluster-effect test: between-log dispersion vs token-permutation null
def disp(x):
    s = np.bincount(cid, weights=x, minlength=NL)
    mu = x.mean()
    return float((cnt_log * (s / cnt_log - mu) ** 2).sum())


obs = disp(dd)
rp = np.random.default_rng(99)
nulls = np.array([disp(rp.permutation(dd)) for _ in range(2000)])
Q3["cluster_effect_permutation_test"] = {"statistic": "sum_l n_l (mean_l - mean)^2 on token gaps", "observed": round(obs, 4),
                                          "null_median": round(float(np.median(nulls)), 4), "null_p95": round(float(np.percentile(nulls, 95)), 4),
                                          "p_value_one_sided": round(float((1 + (nulls >= obs).sum()) / 2001), 4), "B": 2000,
                                          "reading": "p small => the gap varies BY LOG more than chance token assignment explains (cluster effect); the gap can still be everywhere AND clustered"}

# ================================================================================================================ Q4a gating (exact, offline)
Q4 = {"note": "A gate that chooses PRIOR or RT per token is evaluated EXACTLY offline (both arms are already scored per token). "
              "Triggers use only inference-admissible inputs: model plan, causal prior (v0/a0/yaw-rate at t0), v0, driving command, anchor index."}
sc_rt, sc_pr = V["RT_s0"]["PDMS"], V["PRIOR"]["PDMS"]
best = np.maximum(sc_rt, sc_pr)
Q4["bounds_x100"] = {"always_RT": round(100 * float(sc_rt.mean()), 3), "always_PRIOR": round(100 * float(sc_pr.mean()), 3),
                     "ORACLE_per_token_best_of_two_NOT_computable_at_inference": round(100 * float(best.mean()), 3),
                     "oracle_headroom_over_PRIOR": boot_cell(best - sc_pr), "oracle_headroom_over_RT": boot_cell(best - sc_rt)}
def make_triggers(key):
    """Oracle-free trigger values for one inference seed (key in RT_s0 / RT_s1): larger => fall back to the prior.  Each feature is offered with BOTH signs
    (the sign is not guessed); every one uses only the model plan + causal prior + admissible t0 ego scalars + the command."""
    pk = SEAM[key]["poses"]
    kn = SEAM[key]["knots"]
    dx2_, dx4_ = pk[:, IDX2, 0] - P["PRIOR"][:, IDX2, 0], pk[:, IDX4, 0] - P["PRIOR"][:, IDX4, 0]
    dy4_ = pk[:, IDX4, 1] - P["PRIOR"][:, IDX4, 1]
    pdv = np.stack([dist_to_polyline(kn[:, k], poly_pr) for k in range(6)], 1).max(1)
    arc = arc_len(pk)[:, IDX4] / np.maximum(S_PR[:, IDX4], 0.05)
    base = {"abs_dx2_plan_vs_prior": np.abs(dx2_), "abs_dx4_plan_vs_prior": np.abs(dx4_), "plan_faster_than_prior_dx4": dx4_,
            "abs_dy4_plan_vs_prior": np.abs(dy4_), "path_dev_from_prior_polyline": pdv, "mean_l2_plan_vs_prior_8poses": ade(pk, P["PRIOR"]),
            "abs_log_arc4_ratio_plan_over_prior": np.abs(np.log(np.maximum(arc, 0.05))), "v0": v0, "prior_decel_minus_a0": -a0,
            "command_is_straight": (cmd == 1).astype(float) + 1e-6 * rr0.random(N), "command_is_turn": (cmd != 1).astype(float) + 1e-6 * rr0.random(N)}
    out = {}
    for k_, v_ in base.items():
        out["+" + k_] = v_
        if not k_.startswith("command_is"):
            out["-" + k_] = -v_
    return out


rr0 = np.random.default_rng(3)
TRIG = {key: make_triggers(key) for key in ("RT_s0", "RT_s1")}
TRIG["RT_s0"]["+seed_disagreement_4s(needs 2 forwards)"] = dseed
TRIG["RT_s0"]["-seed_disagreement_4s(needs 2 forwards)"] = -dseed
GRID = np.linspace(0.0, 1.0, 21)


def gated_curve(score_rt, score_pr, trigger):
    o = np.argsort(-trigger, kind="stable")             # largest trigger first -> falls back first
    rank = np.empty(N, int)
    rank[o] = np.arange(N)
    out = []
    for q in GRID:
        fb = rank < int(round(q * N))
        out.append(float(np.where(fb, score_pr, score_rt).mean()))
    return np.array(out)


# 6-fold log-grouped CV: pick the fallback fraction q on the TRAINING logs, convert to a threshold (a quantile of the trigger on TRAIN), apply to the held-out logs
FOLDS = 6
fold_of_log = {l: i % FOLDS for i, l in enumerate(np.argsort(-cnt_log, kind="stable"))}      # size-balanced round robin
fold = np.array([fold_of_log[c] for c in cid])


def cv_gate(score_rt, score_pr, trigger):
    pred_fb = np.zeros(N, bool)
    qs = []
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        trt = trigger[tr]
        best_q, best_v = 0.0, -1.0
        for q in GRID:
            if q == 0:
                v = score_rt[tr].mean()
            else:
                thr = np.quantile(trt, 1 - q)
                v = np.where(trt >= thr, score_pr[tr], score_rt[tr]).mean()
            if v > best_v + 1e-12:
                best_q, best_v = q, v
        qs.append(round(float(best_q), 2))
        if best_q > 0:
            thr = np.quantile(trt, 1 - best_q)
            pred_fb[te] = trigger[te] >= thr
    return pred_fb, qs


Q4["gates"] = {}
for nm in TRIG["RT_s0"]:
    row = {}
    for key, scr in (("RT_s0", V["RT_s0"]["PDMS"]), ("RT_s1", V["RT_s1"]["PDMS"])):
        if nm not in TRIG[key]:
            continue
        tr_ = TRIG[key][nm]
        cur = gated_curve(scr, sc_pr, tr_)
        fb, qs = cv_gate(scr, sc_pr, tr_)
        cvv = np.where(fb, sc_pr, scr)
        row[key] = {"in_sample_best_fallback_frac": float(GRID[int(np.argmax(cur))]), "in_sample_best_gated_mean_x100": round(100 * float(cur.max()), 3),
                    "heldout_6fold_log_grouped": {"chosen_q_per_fold": qs, "fallback_frac_realised": round(float(fb.mean()), 4),
                                                  "gated_mean_x100": round(100 * float(cvv.mean()), 3),
                                                  "gain_over_always_RT": boot_cell(cvv - scr), "gain_over_always_PRIOR": boot_cell(cvv - sc_pr)}}
        if key == "RT_s0":
            row[key]["in_sample_curve_x100_at_q_0_to_1_step_0.05"] = [round(100 * float(x), 2) for x in cur]
    Q4["gates"][nm] = row
Q4["gates_multiplicity_note"] = ("%d trigger variants were tried; picking the best of them is a selection the 6-fold CV (which protects only the threshold) does not "
                                 "correct for -- the best row is a HYPOTHESIS for a pre-registered re-test on other logs / navtest, not a result." % len(Q4["gates"]))
# controls for the gate machinery
rr = np.random.default_rng(5)
rand_curves = np.array([gated_curve(sc_rt, sc_pr, rr.random(N)) for _ in range(200)]).mean(0)
lin = sc_rt.mean() + GRID * (sc_pr.mean() - sc_rt.mean())
cc = gated_curve(sc_rt, sc_pr, TRIG["RT_s0"]["+abs_dx2_plan_vs_prior"])
oc = gated_curve(sc_rt, sc_pr, sc_pr - sc_rt)
Q4["gate_controls"] = {"q0_equals_always_RT": bool(abs(cc[0] - sc_rt.mean()) < 1e-12), "q1_equals_always_PRIOR": bool(abs(cc[-1] - sc_pr.mean()) < 1e-12),
                       "random_trigger_max_abs_dev_from_straight_line_x100": round(100 * float(np.abs(rand_curves - lin).max()), 3),
                       "oracle_trigger_max_x100": round(100 * float(oc.max()), 3), "best_of_two_x100": round(100 * float(best.mean()), 3)}
Q4["gate_controls"]["pass"] = bool(Q4["gate_controls"]["q0_equals_always_RT"] and Q4["gate_controls"]["q1_equals_always_PRIOR"]
                                   and Q4["gate_controls"]["random_trigger_max_abs_dev_from_straight_line_x100"] < 0.3
                                   and abs(Q4["gate_controls"]["oracle_trigger_max_x100"] - Q4["gate_controls"]["best_of_two_x100"]) < 0.01)


# ---- policy gates learned per cell by CV (choose RT or PRIOR for each cell on the training logs; apply to the held-out logs)
def cv_cell_policy(cellid, scr, ncell):
    pol_used = []
    gated = np.zeros(N)
    for f in range(FOLDS):
        tr, te = fold != f, fold == f
        pol = np.zeros(ncell, bool)                       # True => PRIOR
        for c_ in range(ncell):
            m = tr & (cellid == c_)
            pol[c_] = bool(m.any() and sc_pr[m].mean() > scr[m].mean())
        pol_used.append([int(x) for x in pol])
        gated[te] = np.where(pol[cellid[te]], sc_pr[te], scr[te])
    return gated, pol_used


Q4["cell_policy_cv"] = {}
for nm, cid_, ncell, names in (("by_command", cmd, 3, CMD_NAMES), ("by_speed_band", band, 4, SPEED_NAMES),
                               ("by_speed_x_command", band * 3 + cmd, 12, [a + "|" + b for a in SPEED_NAMES for b in CMD_NAMES])):
    r_ = {}
    for key, scr in (("RT_s0", V["RT_s0"]["PDMS"]), ("RT_s1", V["RT_s1"]["PDMS"])):
        g_, pol = cv_cell_policy(cid_, scr, ncell)
        full_pol = np.array([bool(sc_pr[cid_ == c_].mean() > scr[cid_ == c_].mean()) if (cid_ == c_).any() else False for c_ in range(ncell)])
        in_s = np.where(full_pol[cid_], sc_pr, scr)
        r_[key] = {"cells": names, "policy_per_fold_(1=fallback_to_PRIOR)": pol, "in_sample_policy_(1=PRIOR)": [int(x) for x in full_pol],
                   "in_sample_gated_mean_x100": round(100 * float(in_s.mean()), 3), "heldout_gated_mean_x100": round(100 * float(g_.mean()), 3),
                   "heldout_gain_over_always_RT": boot_cell(g_ - scr), "heldout_gain_over_always_PRIOR": boot_cell(g_ - sc_pr)}
    Q4["cell_policy_cv"][nm] = r_
# fixed, single-hypothesis cell gates (not tuned): listed so the cell shares of Q1 can be read as lever sizes
cell_gates = []
for b, nm in enumerate(SPEED_NAMES):
    m = band == b
    cell_gates.append({"gate": "fallback to PRIOR when v0 in " + nm, "n": int(m.sum()), "gated_mean_x100": round(100 * float(np.where(m, sc_pr, sc_rt).mean()), 3),
                       "gain_over_always_RT": boot_cell(np.where(m, sc_pr, sc_rt) - sc_rt)})
for c, nm in enumerate(CMD_NAMES):
    m = cmd == c
    cell_gates.append({"gate": "fallback to PRIOR when command = " + nm, "n": int(m.sum()), "gated_mean_x100": round(100 * float(np.where(m, sc_pr, sc_rt).mean()), 3),
                       "gain_over_always_RT": boot_cell(np.where(m, sc_pr, sc_rt) - sc_rt)})
m = (cmd == 1)
cell_gates.append({"gate": "HYBRID: PRIOR on command=straight, RT on left/right", "n": int(m.sum()), "gated_mean_x100": round(100 * float(np.where(m, sc_pr, sc_rt).mean()), 3),
                   "gain_over_always_RT": boot_cell(np.where(m, sc_pr, sc_rt) - sc_rt), "gain_over_always_PRIOR": boot_cell(np.where(m, sc_pr, sc_rt) - sc_pr),
                   "s1_replicate_gated_mean_x100": round(100 * float(np.where(m, sc_pr, V["RT_s1"]["PDMS"]).mean()), 3),
                   "s1_gain_over_always_PRIOR": boot_cell(np.where(m, sc_pr, V["RT_s1"]["PDMS"]) - sc_pr)})
Q4["cell_gates"] = cell_gates

# ---- the ONE measured longitudinal intervention that already exists: RT = LEGAL's path with its progress pulled toward the prior by (1-g)
#      (g = 0.96 at 0.5 s ... 0.58 at 2 s ... 0.91 at 6 s).  RT - LEGAL by how far LEGAL's own progress sat from the prior's:
lr_leg = np.log(np.maximum(S_LG[:, IDX4] / np.maximum(S_PR[:, IDX4], 0.05), 0.05))
leg_bin = np.digitize(lr_leg, [-0.15, 0.15])
PULL = {"what": "RT keeps LEGAL's path and moves its progress a fixed fraction (1-g) toward the prior's; an existing, scored, partial 'longitudinal scale-down'",
        "bins_by_LEGAL_arc4_over_PRIOR": {}, "by_command": {}}
for i_, nm_ in enumerate(["LEGAL slower than prior (<0.86x)", "similar (0.86-1.16x)", "LEGAL faster than prior (>1.16x)"]):
    m_ = leg_bin == i_
    PULL["bins_by_LEGAL_arc4_over_PRIOR"][nm_] = {"s0": boot_cell(V["RT_s0"]["PDMS"] - V["LEGAL_s0"]["PDMS"], m_), "s1": boot_cell(V["RT_s1"]["PDMS"] - V["LEGAL_s1"]["PDMS"], m_)}
for c_, nm_ in enumerate(CMD_NAMES):
    PULL["by_command"][nm_] = {"s0": boot_cell(V["RT_s0"]["PDMS"] - V["LEGAL_s0"]["PDMS"], cmd == c_), "s1": boot_cell(V["RT_s1"]["PDMS"] - V["LEGAL_s1"]["PDMS"], cmd == c_)}
PULL["note"] = "bins use LEGAL's own plan (the thing being pulled); a bin is a post-hoc cut. Reads the SIGN of a longitudinal pull-to-prior where LEGAL over/under-drives."
Q4["measured_partial_longitudinal_pull_RT_minus_LEGAL"] = PULL

# the measured longitudinal / geometry reference points on the (PRIOR -> LEGAL) segment
Q4["measured_points_x100"] = {"PRIOR(alpha=0)": mm["PRIOR"], "RT_s0 (LEGAL path, progress blended 0.42..0.04 toward prior)": mm["RT_s0"],
                              "LEGAL_s0 (alpha=1, no retime)": mm["LEGAL_s0"], "RT_s1": mm["RT_s1"], "LEGAL_s1": mm["LEGAL_s1"], "STOP": mm["STOP"],
                              "reading": "RT - LEGAL is ~0 on average (see headline RT_s0_minus_LEGAL_s0), so the existing progress re-timing neither fixes nor causes the gap; "
                                         "the gap is already present in LEGAL, i.e. in the model's residual on both axes."}

# ===================================================================================================================== write
res = {"label": LABEL, "inputs": {"run_dir": RUN.replace("\\", "/"), "n_tokens": N, "n_logs": NL, "speed_edges_mps": SPEED_EDGES[:-1] + ["inf"],
                                  "command_counts": {nm: int((cmd == c).sum()) for c, nm in enumerate(CMD_NAMES)},
                                  "speed_band_counts": {nm: int((band == b).sum()) for b, nm in enumerate(SPEED_NAMES)}},
       "controls": ctl, "Q1_where": Q1, "Q2_plan_error": Q2, "Q3_concentration": Q3, "Q4a_gating": Q4}


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "PRIOR_GAP_DIAG.json"), "w", encoding="utf-8") as fh:
    json.dump(clean(res), fh, indent=1, allow_nan=False)
print("controls (pass flags; C5 is expected FALSE as written, see its reading):", {k: (v.get("pass") if isinstance(v, dict) else v) for k, v in ctl.items()})
print("wrote", os.path.join(OUT, "PRIOR_GAP_DIAG.json"))
