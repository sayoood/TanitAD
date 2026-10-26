import json
from pathlib import Path
x = json.loads(Path("C:/Users/Admin/refcv8_pf/refcv9_r9pf/result/R9PF_X_ATTRIB.json").read_text(encoding="utf-8"))
c = x["cells"]
def g(tag, arm, s="s0"):
    r = c[tag][arm][s]
    return r
o0, s0, t0, r0 = g("R9PF_fs0", "ORACLE"), g("R9PF_fs0", "SOFT"), g("R9PF_fs0", "TAU"), g("R9PF_fs0", "R9")
f = lambda q: f"{q['delta']:+.4f} [{q['lo']:+.4f}, {q['hi']:+.4f}]"
conf = c["R9PF_fs0"]["decision_vs_r_star_S6_compliant"]["s0"]["confusion_rows_rstar_cols_decision"]
txt = (
"**RULE ZERO -- the failed bars leave the next arm, and the attribution that picks it was RUN (EXPLORATORY, post-hoc on eval139, "
"result/R9PF_X_ATTRIB.json; T1 open loop; paired episode-cluster bootstrap, B 2000 seed 0; inference seed 0 shown, seed 1 within the same CIs; "
"NOT a bar, NOT a claim).**\n\n"
f"1. **The loss is the DECISION, not the profiles.** ORACLE (R9PF_fs0's own profiles with the GT regime -- inadmissible, attribution only) "
f"vs H2CN_DEPLOY8 on S6 compliant: ADE {f(o0['ADE_S6_vs_H8'])} m, MD {o0['MD_rate']:.3f} vs {x['H8_rates']['s0']['MD_rate']:.3f}, "
f"{o0['n_rows_worse']} family rows separated worse (all three fit seeds: 0). The registered argmax decision gets {conf[0][0]} / {sum(conf[0])} GT-decel "
f"and {conf[2][2]} / {sum(conf[2])} GT-accel windows right ({conf[0][1]} decel and {conf[2][1]} accel called cruise; {conf[1][0] + conf[1][2]} / {sum(conf[1])} "
"cruise windows called decel/accel); OOF decel recall on the calibration split is 0.52-0.55 at a false-decel rate of 0.07-0.08 (x_tau_fit.py).\n"
f"2. **Calibrated tau (prereg s1 beside, run now) makes it WORSE:** fs0 tau 0.25 -> MD {f(t0['MD_vs_H8'])} but FD {f(t0['FD_vs_H8'])}, "
f"{t0['n_rows_worse']} rows worse. A threshold only moves along the same weak classifier's ROC.\n"
f"3. **The soft decode removes the guard failure.** SOFT = sum_r p(r) dS[r] on the registered heads, no refit: fs0 MD {f(s0['MD_vs_H8'])}, "
f"FD {f(s0['FD_vs_H8'])} vs H2CN_DEPLOY8; family rows separated worse 13 -> {s0['n_rows_worse']} (fs1 1, fs2 2 / 1); PA-2-like holds at every "
"fit seed; PA-1b-like holds for fs0 (seed 0) and fs1 (both seeds). It still misses PA-1a on FD (+0.022 to +0.026, separated above) -- not a pass.\n\n"
"**Next arm (cheapest first; each needs a Master-Mind seal BEFORE it runs -- prereg s5 says the constrained stage-2 objective is "
"'pre-registered first'):**\n"
"* **R9PF-SOFT2 (0 GPU, ~15 CPU-min):** stage 2 refit with the loss evaluated through the decode it is used with -- L1 of s_P + sum_r p(r) dS[r] "
"against s_H, plus the regime CE -- same split, grid, folds and fit seeds; decode = the posterior-weighted profile; bars PA-1a/1b/2/G/R as sealed. "
"This is prereg s5's 'constrained stage-2 objective' aimed at the measured term (decision errors), with K-CONSTZ / K-SHUF carried.\n"
"* **The decision itself (the dominant lever):** a better regime classifier needs either more clean calibration clips (V-C; availability "
"UNVERIFIED -- the blocker is a corpus check) or the lon decision trained in the trunk with richer supervision (Phase B; ~240 Thor-h ESTIMATED -- "
"the blocker is the PI's compute decision D-R9-INMODEL Q3).\n"
"**Blocked on:** the Master Mind's seal for R9PF-SOFT2 (prereg s5); the PI for Phase B; a corpus availability check for V-C.")
Path("C:/Users/Admin/refcv8_pf/refcv9_r9pf/code/next_r9pf.json").write_text(json.dumps({"text": txt}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(txt)
