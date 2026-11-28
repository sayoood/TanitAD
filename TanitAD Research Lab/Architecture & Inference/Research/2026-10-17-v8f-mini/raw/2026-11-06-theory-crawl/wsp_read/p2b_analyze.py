"""P2b analysis of P2B_PERWINDOW.npz against CLAIMS_P2.md (stamped 22:14:27Z). Cluster bootstrap over clips (episodes), B 2000, seed 0, windows in the band.
Writes P2B_RESULT.json. Verdict of P2 as written is R_tot on the 3-8 m/s band (point estimate); thresholds >= 5 holds, <= 2 fails, between INCONCLUSIVE."""
import json, sys
import numpy as np

HERE = "E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/wsp_read"
TAG = sys.argv[1] if len(sys.argv) > 1 else "P2B"
Z = np.load(f"{HERE}/{TAG}_PERWINDOW.npz", allow_pickle=True)
clip = Z["clip"]; v0 = Z["v0"]; band = Z["band"]
BANDS = ["v<3", "3-8", "8-15", "v>=15"]
NORM = lambda k: Z["norm." + k]
VAL = lambda k: Z["val." + k]
GK = lambda s: Z["gk." + s]
rng = np.random.default_rng(0)
B = 2000

def boot_ratio(num, den, mask, stat=lambda x: x.mean()):
    """ratio stat(num)/stat(den) over windows in mask; episode-cluster bootstrap CI."""
    idx = np.where(mask)[0]
    cl = clip[idx]; uc, inv = np.unique(cl, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uc))]
    pt = float(stat(num[idx]) / stat(den[idx]))
    rs = []
    for _ in range(B):
        pick = rng.integers(0, len(uc), len(uc))
        sel = np.concatenate([groups[p] for p in pick])
        rs.append(stat(num[idx][sel]) / stat(den[idx][sel]))
    lo, hi = np.quantile(rs, [0.025, 0.975])
    return {"point": pt, "ci95": [float(lo), float(hi)], "n_windows": int(len(idx)), "n_episodes": int(len(uc))}

def boot_stat(x, mask, f):
    idx = np.where(mask)[0]
    cl = clip[idx]; uc, inv = np.unique(cl, return_inverse=True)
    groups = [np.where(inv == i)[0] for i in range(len(uc))]
    pt = float(f(x[idx])); rs = []
    for _ in range(B):
        pick = rng.integers(0, len(uc), len(uc))
        rs.append(f(x[idx][np.concatenate([groups[p] for p in pick])]))
    lo, hi = np.quantile(rs, [0.025, 0.975])
    return {"point": pt, "ci95": [float(lo), float(hi)]}

R = {"tag": TAG, "claims_sha_prefix": "afb9785f", "n_windows": int(len(v0)), "bands": {}}
rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
for bi, bn in enumerate(BANDS):
    m = band == bi
    if m.sum() < 5:
        continue
    r = {"n": int(m.sum()), "n_episodes": int(len(np.unique(clip[m])))}
    # ---- verdict statistic and companions
    r["R_tot_z"] = boot_ratio(NORM("s_tot.ego_total_z"), NORM("s_tot.vision_total_z"), m)
    r["R_tot_raw_units"] = boot_ratio(NORM("s_tot.ego_total_raw"), NORM("s_tot.vision_total_raw"), m)
    r["R_tot_z_rms"] = boot_ratio(NORM("s_tot.ego_total_z"), NORM("s_tot.vision_total_z"), m, stat=lambda x: np.sqrt((x ** 2).mean()))
    r["R_tot_z_median"] = boot_ratio(NORM("s_tot.ego_total_z"), NORM("s_tot.vision_total_z"), m, stat=lambda x: np.median(x))
    r["R_res_O_f8z_over_vision_O"] = boot_ratio(NORM("s_res.ego_O_f8z"), NORM("s_res.vision_O_z"), m)
    r["R_res_total_ego3_over_vision_total"] = boot_ratio(NORM("s_res.ego_total_z"), NORM("s_res.vision_total_z"), m)
    r["R_res_ST_f8z_over_vision_ST"] = boot_ratio(NORM("s_res.ego_ST_f8z"), NORM("s_res.vision_ST_z"), m)
    r["R_code_over_ego_res_O"] = boot_ratio(NORM("s_res.code_O"), NORM("s_res.ego_O_f8z"), m)
    r["R_nav_over_ego_res_O"] = boot_ratio(NORM("s_res.nav_O"), NORM("s_res.ego_O_f8z"), m)
    r["R_ck_over_ego_res_O"] = boot_ratio(NORM("s_res.ck_O"), NORM("s_res.ego_O_f8z"), m)
    r["R_lim_over_ego_res_O"] = boot_ratio(NORM("s_res.lim_O"), NORM("s_res.ego_O_f8z"), m)
    r["R_vision_O_over_ego_res_O"] = boot_ratio(NORM("s_res.vision_O_z"), NORM("s_res.ego_O_f8z"), m)
    r["R_codeLON_over_code_res_O"] = boot_ratio(NORM("s_res.code_lon_O"), NORM("s_res.code_O"), m)
    # mean norms (absolute) per scalar and group
    r["mean_norms"] = {k.replace("norm.", ""): float(np.mean(Z[k][m])) for k in Z.files if k.startswith("norm.") and not k.startswith("norm.ctrl")}
    # ---- W1: the prior's net acts on a0, not on v0
    g_net, g_res, g_tot = GK("s_net"), GK("s_res"), GK("s_tot")
    ds_base = g_tot - g_net - g_res
    r["W1"] = {"median_d_s_net_d_v0": float(np.median(g_net[m, 0])), "median_abs_d_s_net_d_v0": float(np.median(np.abs(g_net[m, 0]))),
               "median_d_s_net_d_a0": float(np.median(g_net[m, 1])), "median_d_s_base_d_v0": float(np.median(ds_base[m, 0])),
               "median_d_s_base_d_a0": float(np.median(ds_base[m, 1])), "median_d_prior_d_a0": float(np.median(g_net[m, 1] + ds_base[m, 1])),
               "median_d_prior_d_v0": float(np.median(g_net[m, 0] + ds_base[m, 0])),
               "median_d_s_res_d_v0": float(np.median(g_res[m, 0])), "median_d_s_res_d_a0": float(np.median(g_res[m, 1])),
               "median_d_s_tot_d_v0": float(np.median(g_tot[m, 0])), "median_d_s_tot_d_a0": float(np.median(g_tot[m, 1]))}
    # ---- components
    s_tot, s_base, s_net, s_res = VAL("s_tot"), VAL("s_base"), VAL("s_net"), VAL("s_res")
    prior = s_base + s_net
    r["components"] = {"mean_v0": float(v0[m].mean()), "mean_s_tot": float(s_tot[m].mean()), "mean_s_base": float(s_base[m].mean()), "mean_s_net": float(s_net[m].mean()),
                       "mean_s_res": float(s_res[m].mean()), "rms_s_res": rms(s_res[m]), "rms_s_net": rms(s_net[m]), "std_s_tot": float(s_tot[m].std()),
                       "std_s_res": float(s_res[m].std()), "std_prior": float(prior[m].std()),
                       "var_share_residual": float(s_res[m].var() / s_tot[m].var()), "var_share_prior": float(prior[m].var() / s_tot[m].var()),
                       "cov_share_residual": float(np.cov(s_res[m], s_tot[m])[0, 1] / s_tot[m].var()),
                       "rms_s_tot_minus_v0": rms((s_tot - v0)[m]), "rms_base_minus_v0": rms((s_base - v0)[m]), "rms_prior_minus_v0": rms((prior - v0)[m]),
                       "rms_res": rms(s_res[m]), "corr_stot_v0": float(np.corrcoef(s_tot[m], v0[m])[0, 1]), "corr_sres_stot_minus_v0": float(np.corrcoef(s_res[m], (s_tot - v0)[m])[0, 1]),
                       "R2_of_stot_on_prior": float(1 - ((s_tot - prior)[m] ** 2).sum() / ((s_tot[m] - s_tot[m].mean()) ** 2).sum()),
                       "R2_of_stot_minus_v0_on_prior_minus_v0": float(1 - ((s_tot - prior)[m] ** 2).sum() / (((s_tot - v0)[m] - (s_tot - v0)[m].mean()) ** 2).sum())}
    # ---- ablations: effect on plan speed s (m/s)
    ab = {}
    for k in Z.files:
        if k.startswith("abl."):
            d = Z[k][m] - s_tot[m]
            ab[k[4:]] = {"rms": rms(d), "mean_abs": float(np.mean(np.abs(d))), "median_abs": float(np.median(np.abs(d))), "mean_signed": float(d.mean())}
    r["ablation_effect_on_s"] = ab
    def ratio_ab(g1, g2, mode):
        return ab[f"{g1}.{mode}"]["rms"] / ab[f"{g2}.{mode}"]["rms"]
    r["ablation_rms_ratios"] = {mode: {"ego_over_vision": ratio_ab("ego", "vision", mode), "code_over_vision": ratio_ab("code", "vision", mode), "code_over_ego": ratio_ab("code", "ego", mode),
                                       "nav_over_vision": ratio_ab("nav", "vision", mode)} for mode in ("zero", "shuffle_band", "shuffle_global")}
    R["bands"][bn] = r

# ---------------------------------------------------------------- verdict and claim checks (band 3-8)
b = R["bands"]["3-8"]
rt = b["R_tot_z"]["point"]
verdict = "HOLDS" if rt >= 5 else ("FAILS" if rt <= 2 else "INCONCLUSIVE")
def crossing(ci, pt):
    return {"ci95": ci, "crosses_5": ci[0] < 5 <= ci[1], "crosses_2": ci[0] < 2 <= ci[1]}
chk = {"P2_verdict_as_written": verdict, "R_tot_z": b["R_tot_z"], "R_tot_z_threshold_crossing": crossing(b["R_tot_z"]["ci95"], rt)}
w1 = b["W1"]
chk["W1_net_not_level_carrier"] = {"claim": "median |d s_net/d v0| <= 0.15 and median d s_net/d a0 in [-2.0,-0.6]; median d(base+net)/d a0 in [0,1.3]",
                                   "measured": {"median_abs_dv0": w1["median_abs_d_s_net_d_v0"], "median_da0": w1["median_d_s_net_d_a0"], "median_prior_da0": w1["median_d_prior_d_a0"]},
                                   "holds": bool(w1["median_abs_d_s_net_d_v0"] <= 0.15 and -2.0 <= w1["median_d_s_net_d_a0"] <= -0.6 and 0.0 <= w1["median_d_prior_d_a0"] <= 1.3)}
rr = b["R_res_O_f8z_over_vision_O"]
chk["W2_residual_ego_over_vision_in_[0.08,1.3]"] = {"measured_O_level": rr, "measured_total": b["R_res_total_ego3_over_vision_total"], "holds_O_level": bool(0.08 <= rr["point"] <= 1.3),
                                                    "below_2_O_level": bool(rr["point"] < 2)}
rc = b["R_code_over_ego_res_O"]
chk["W3_code_over_ego_in_[0.35,3.1]"] = {"measured": rc, "holds": bool(0.35 <= rc["point"] <= 3.1)}
R["checks_3_8"] = chk
json.dump(R, open(f"{HERE}/{TAG}_RESULT.json", "w"), indent=1)

# ---------------------------------------------------------------- print a compact digest
def fmt(d): return f"{d['point']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]"
for bn, r in R["bands"].items():
    print(f"== band {bn} n={r['n']} eps={r['n_episodes']}")
    print("  R_tot_z", fmt(r["R_tot_z"]), "| raw units", fmt(r["R_tot_raw_units"]), "| rms", fmt(r["R_tot_z_rms"]), "| median", fmt(r["R_tot_z_median"]))
    print("  R_res O-level", fmt(r["R_res_O_f8z_over_vision_O"]), "| total", fmt(r["R_res_total_ego3_over_vision_total"]), "| ST", fmt(r["R_res_ST_f8z_over_vision_ST"]))
    print("  code/ego(res,O)", fmt(r["R_code_over_ego_res_O"]), "nav/ego", fmt(r["R_nav_over_ego_res_O"]), "ck/ego", fmt(r["R_ck_over_ego_res_O"]), "lim/ego", fmt(r["R_lim_over_ego_res_O"]), "vision/ego", fmt(r["R_vision_O_over_ego_res_O"]))
    print("  W1", {k: round(v, 3) for k, v in r["W1"].items()})
    c = r["components"]
    print("  comp", {k: round(v, 3) for k, v in c.items()})
    print("  abl rms (m/s)", {k: round(v["rms"], 3) for k, v in r["ablation_effect_on_s"].items()})
    print("  abl ratios", {m: {k: round(x, 2) for k, x in v.items()} for m, v in r["ablation_rms_ratios"].items()})
print(json.dumps(chk, indent=1)[:3500])
