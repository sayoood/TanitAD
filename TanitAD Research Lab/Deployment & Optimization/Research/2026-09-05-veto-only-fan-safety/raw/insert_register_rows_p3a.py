#!/usr/bin/env python3
"""Register rows for the 200-step veto arms and the offset-head attribution.

Anchored on the `| H-VETO-FAN-1 |` row inserted earlier; refuses on a missing or
duplicated anchor rather than silently no-op'ing.
"""
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
PATH = os.path.join(REPO, "Project Steering", "GOALS_AND_CLAIMS.md")
CR, LF = chr(13), chr(10)
CRLF = CR + LF
ST, NE, W, ED, X, ARR, ELL = (chr(11088), chr(9940), chr(9888), chr(8212), chr(215),
                              chr(8594), chr(8230))
TH, D_ = chr(952), chr(916)

ROWS = """| D-RL-VETO-DOSE200-1 | {ST}{ST} **THE VETO ALONE MAKES refcv3's EMITTED FAN MEASURABLY MORE FEASIBLE {ED} `fan_peak_g_mean` **-0.093 / -0.116 g**, REPLICATED ACROSS THREE RUNS, CLEARING BOTH FLOORS {ED} AND IT COSTS A LITTLE FEASIBILITY ON THE SELECTED PATH.** MEASURED 2026-09-05 (`veto200`: reward weights ALL 0.0, `veto_enabled=True`, `w_anchor` 1.0, lr 1e-5, **200 steps**, two_scalar noise, G 4, batch 2; 214.7 s and 150.8 s on the dev-box 4060, 0 pod-hours). Readout: the SAME fixed **120 EVAL windows** (`Random(1234)`), 79 episodes, 36 lead windows, paired episode-cluster bootstrap n_boot 4,000. `veto_rate_mean` **0.0897 (s0) / 0.0980 (s1)** {ED} the constraint channel really fired. **Guards:** G1 `ctrl_null` `veto_rate_mean` **0.0000 exactly**; G2 every BEFORE readout **bitwise identical**, max abs diff **0.000e+00** over 120 w {X} 65 metrics; G3 both seeds present. {ST} **`fan_peak_g_mean` clears all three hurdles** {ED} separated at both seeds, same sign, 4-5{X} the seed-replicate floor (0.0229), and the zero-information arm drifts it **+0.134 the OTHER way**, so the drift cannot explain it; the direct veto-minus-`ctrl_null` contrast is **-0.227 g**, separated. Level: **4.1809 {ARR} 4.088 / 4.065 g, -2.2 to -2.8 %**, from a channel carrying NO reward at all. {NE} **AND THE REPLICATE KILLED FOUR METRICS A SINGLE SEED WOULD HAVE SHIPPED**, including the one the previous package proposed shipping on: `top32_infeasible` reads **-0.0064 (s0) vs -0.0011 (s1)** against a seed floor of 0.0053 {ED} not reproducible across training runs; same for `top8_kamm_over`, `top32_envelope`, `top8_infeasible`. {NE} **THE COST IS REAL AND IS REPORTED:** `sel_peak_g` **WORSENED** and clears the same floors (**+0.0155 / +0.0080 g** on a base of 0.1947, i.e. **+4 to +8 % on the path the car drives**), with selector agreement **1.000 / 0.992** {ED} the fan moved under a selector that never changed its pick, so the selected candidate itself got more aggressive. A safety claim quoting only `fan_peak_g_mean` would be true and misleading. {ST} **A THIRD run was already banked:** `ctrl_const` (same seed, same flags, hours earlier, `veto_rate` 0.0896826 vs 0.0896924 {ED} GPU non-determinism, not a config difference) reads `fan_peak_g_mean` **-0.0859**; across the three runs the metric is **-0.0859 / -0.0929 / -0.1158**, every one negative. But `top32_infeasible` spans **-0.01431 to -0.00105 over those same three runs, TWO OF WHICH SHARE A SEED** {ED} so this rig's run-to-run noise is not only a SEED effect, and `H-ESTIM-SEED-1`'s replicate requirement needs to be read as *runs*, however they differ | **SUPPORTED (MEASURED 2026-09-05)** {ED} `fan_peak_g_mean` only; every other feasibility metric is WITHIN-NOISE. SPEC {SEC}7's committed SUCCESS text needs `fan_peak_g_mean` **and** one of `top32_infeasible`/`sel_infeasible`, so the formal PRIMARY exit at this dose is **FAIL** | `TanitAD Research Lab/Deployment & Optimization/Research/2026-09-05-veto-only-fan-safety/RESULT.md` {SEC}5; `raw/veto_verdict_veto200.json` (all 57 metrics, both floors, the contrast), `raw/run/s{{0,1}}/veto200/arm_summary.json`, `raw/run/s0/ctrl_null/arm_summary.json` |
| D-REFC-OFFSET-FEAS-1 | {ST}{ST} **refcv3's FAN INFEASIBILITY IS NOT IN THE VOCABULARY {ED} THE OFFSET HEAD MANUFACTURES IT, AN 8.56{X} BLOW-UP IN PEAK FRICTION LOAD.** MEASURED 2026-09-05 at **ZERO GPU** (`stack/scripts/bank_vs_fan_feasibility.py`): the FROZEN anchor bank (`core.decoder.anchors`, a fixed `[128, 8, 2]` path set read straight out of the checkpoint) scored by the SAME `fan_safety.score_paths`, on the SAME 240 EVAL windows, with the same `v0` and lead track as the emitted fan. **`envelope` 0.0156 {ARR} 0.8877** (+0.8721) {MID} **`kamm_over` 0.1953 {ARR} 0.8406** {MID} **`peak_g` 0.4808 {ARR} 4.1131 g (+3.6323, 8.56{X})** {MID} `contact` 0.0239 {ARR} 0.0249 {MID} mean \\|emitted - bank\\| per waypoint over the 2 s prefix **9.29 m**. {W} **Read the `off_reach` row before concluding the offset is gratuitous:** the bank is **77.8 %** off-reach and the emitted fan only **10.8 %**, because the bank is a FIXED path set that mostly does not match the window's own speed {ED} adapting it IS the offset head's job. The finding is that **the adaptation is paid for in the friction envelope, at 8.56{X}**. {ST} **Three consequences.** (1) The RL veto moves `fan_peak_g_mean` by **-0.10 g against a +3.63 g blow-up** {ED} it addresses **~2.7 %** of the gap; RL post-training of the offset head is the right SURFACE and the wrong SIZE of instrument. (2) The high-value build is a **FEASIBILITY-AWARE OFFSET** {ED} a reparameterisation or penalty keeping the ROLLED path inside the envelope, applied where the offset is produced rather than after it. (3) It explains `D-RL-FANSAFE-1`'s paradox with no RL panel at all: `oracle_sel` (best-ADE) is the LEAST drivable candidate (0.1105 vs `os` 0.0865) because matching the human closely is exactly what large offsets buy. {W} LIMIT: `peak_g` is the finite-difference load on free waypoints and under-reports by 1.21-1.85{X} {ED} **on BOTH sides**, so the RATIO is the robust quantity and the levels are lower bounds | **SUPPORTED (MEASURED 2026-09-05)** {ED} opens a feasibility-aware-offset work item for Arch+Inference | `{ELL}/2026-09-05-veto-only-fan-safety/RESULT.md` {SEC}6, `raw/bank_vs_fan_feasibility.json`, `raw/fan_bank_base_240w.npz`, `stack/scripts/bank_vs_fan_feasibility.py` |
""".replace("{ST}", ST).replace("{NE}", NE).replace("{W}", W).replace("{ED}", ED) \
   .replace("{X}", X).replace("{ARR}", ARR).replace("{ELL}", ELL) \
   .replace("{SEC}", chr(167)).replace("{MID}", chr(183))

raw = open(PATH, "rb").read()
is_crlf = CRLF.encode() in raw
s = raw.decode("utf-8").replace(CRLF, LF)
if "D-REFC-OFFSET-FEAS-1" in s:
    raise SystemExit("[register] rows already present")
lines = s.split(LF)
idx = [i for i, l in enumerate(lines) if l.startswith("| H-VETO-FAN-1 |")]
if len(idx) != 1:
    raise SystemExit("[register] expected exactly one `| H-VETO-FAN-1 |` row, found %d" % len(idx))
new_rows = [r for r in ROWS.split(LF) if r.strip()]
lines[idx[0] + 1:idx[0] + 1] = new_rows
out = LF.join(lines)
open(PATH, "wb").write((out.replace(LF, CRLF) if is_crlf else out).encode("utf-8"))
print("[register] %d rows inserted after H-VETO-FAN-1 (%s)"
      % (len(new_rows), "CRLF" if is_crlf else "LF"))
