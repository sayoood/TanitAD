#!/usr/bin/env python3
"""P6-HN step 0 -- DATA gates run BEFORE the pre-registration is hashed. They test the join, not the census:
   no distance to the teacher and no hard-negative count among proximate proposals is computed here."""
import json, sys
import numpy as np
z = np.load(sys.argv[1])
N = len(z["pdms"]); R = {}
R["n_sets"] = int(N); R["n_logs"] = int(len(np.unique(z["keys_log"])))
R["shape_proposals"] = list(z["proposals"].shape); R["shape_teacher"] = list(z["teacher"].shape)
R["any_nan"] = {k: bool(np.isnan(z[k]).any()) for k in ("proposals", "teacher", "nc", "dac", "ep", "ttc", "c", "ddc", "pdms")}
R["unique_keys"] = int(len({(a, b, int(c), int(d)) for a, b, c, d in zip(z["keys_log"], z["keys_token"], z["keys_step"], z["keys_rank"])}))
R["rank_counts"] = {int(r): int((z["keys_rank"] == r).sum()) for r in np.unique(z["keys_rank"])}
R["step_counts"] = {int(r): int((z["keys_step"] == r).sum()) for r in np.unique(z["keys_step"])}
R["label_version_counts_in_sets"] = {int(r): int((z["set_label_version"] == r).sum()) for r in np.unique(z["set_label_version"])}
R["ckpt_step_counts_in_sets"] = {int(r): int((z["set_ckpt_step"] == r).sum()) for r in np.unique(z["set_ckpt_step"])}
# G3 PDMS==0 <=> NC*DAC==0 (and NC, DAC are exactly {0,1}?)
nc, dac, pdms = z["nc"], z["dac"], z["pdms"]
R["G3_nc_values"] = sorted(set(np.unique(nc).round(3).tolist()))[:10]
R["G3_dac_values"] = sorted(set(np.unique(dac).round(3).tolist()))[:10]
zero_pdms = pdms == 0; zero_gate = (nc * dac) == 0
R["G3_pdms0_equals_ncdac0_all"] = bool((zero_pdms == zero_gate).all())
R["G3_n_disagree"] = int((zero_pdms != zero_gate).sum())
# G2 order alignment between the SETS' own navsim_dac.violation and the labels' dac, with a within-set permutation control
nav = z["set_navsim_dac"]; ok = ~np.isnan(nav)
R["G2_nav_defined_frac"] = float(ok.mean())
lab0 = (dac == 0); nav1 = (nav > 0)
agree = ((lab0 == nav1) & ok).sum() / ok.sum()
rng = np.random.default_rng(20260927)
ag_perm = []
for _ in range(20):
    perm = np.stack([rng.permutation(64) for _ in range(N)])
    nav_p = np.take_along_axis(nav, perm, axis=1); okp = ~np.isnan(nav_p)
    ag_perm.append(float((((lab0 == (nav_p > 0)) & okp).sum()) / okp.sum()))
R["G2_dac_agreement_labels_vs_sets_navsim_dac"] = float(agree)
R["G2_dac_agreement_permuted_control_mean"] = float(np.mean(ag_perm)); R["G2_dac_agreement_permuted_control_max"] = float(np.max(ag_perm))
# 2x2 table for transparency
R["G2_table_labdac0_vs_navviol"] = {"lab0_nav1": int((lab0 & nav1 & ok).sum()), "lab0_nav0": int((lab0 & ~nav1 & ok).sum()),
                                    "lab1_nav1": int((~lab0 & nav1 & ok).sum()), "lab1_nav0": int((~lab0 & ~nav1 & ok).sum())}
# G6 trajectory frame sanity: first pose near the ego origin, forward positive x for proposals AND teacher (descriptive of the frame only)
R["G6_proposal_first_pose_x_median"] = float(np.median(z["proposals"][:, :, 0, 0])); R["G6_teacher_first_pose_x_median"] = float(np.median(z["teacher"][:, 0, 0]))
R["G6_proposal_last_pose_x_median"] = float(np.median(z["proposals"][:, :, -1, 0])); R["G6_teacher_last_pose_x_median"] = float(np.median(z["teacher"][:, -1, 0]))
print(json.dumps(R, indent=1))
json.dump(R, open(sys.argv[2], "w"), indent=1)
