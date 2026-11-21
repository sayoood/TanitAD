#!/usr/bin/env python3
"""Final register rows: the dose-response, the T1 guard, and the 0-training product.

Also resolves `H-VETO-FAN-1` from OPEN to its committed exit.
"""
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
PATH = os.path.join(REPO, "Project Steering", "GOALS_AND_CLAIMS.md")
CR, LF = chr(13), chr(10)
CRLF = CR + LF
ST, NE, W, ED, X, ARR, ELL, PCT = (chr(11088), chr(9940), chr(9888), chr(8212), chr(215),
                                   chr(8594), chr(8230), "%")
MID, SEC = chr(183), chr(167)

ROWS = """| D-RL-VETO-DOSE2K-1 | {ST} **THE VETO'S FAN GAIN IS A SHORT-DOSE EFFECT: AT 2,000 STEPS NOTHING SURVIVES, AND THE FAN'S QUALITY STARTS TO GO.** MEASURED 2026-09-05 (`veto2k`: identical to `veto200` but `steps` 200 {ARR} **2,000**; 2,133.6 s / 2,106.4 s; `veto_rate_mean` **0.0912 / 0.0916**, so the constraint fired at the same rate throughout; guards G1/G2/G3 pass on the same 120 EVAL windows). **PRIMARY-FAIL with NOTHING quotable as a lever**: `fan_peak_g_mean` **-0.0078 / -0.0484** against a seed-replicate floor of **0.0407**; `top32_infeasible` -0.0048 / -0.0008 (floor 0.0040); `fan_envelope` -0.0035 / -0.0015 (floor 0.0019). Beside the 200-step arm this is a **DOSE-RESPONSE, not a null**: `fan_peak_g_mean` reads **-0.0929 / -0.1158 (floor 0.0229, QUOTABLE)** at 200 steps and **-0.0078 / -0.0484 (floor 0.0407, not)** at 2,000. The supporting T0 readouts agree in a second voice: at 2 k `R3` (sel-ADE 2 s) **+0.0121 / +0.0136 SEPARATED** and `R_ORACLE` (oracle-in-fan {ED} fan QUALITY) **+0.0178 / +0.0207 SEPARATED**, where at 200 steps `R3` was +0.0033 **ns**; confidence mass on infeasible candidates rises too (`mass_rank_infeasible` +0.0129 / +0.0155 vs a 0.0026 floor). {ST} **Read as a control law:** a pure constraint channel has NO ranking term to trade against, so once it has pushed the fan off the violating region there is nothing left to optimise and the trust region plus the optimizer's own drift (`D-RL-VETO-EXPLICIT-1` / RETRACTION #27) take over. Running BOTH doses was pre-registered; one dose alone would have produced either an unrepeatable win or an unexplained null. {W} **The 2 k arm's WORSENINGS are NOT ATTRIBUTABLE**: the dose-matched zero-information control was queued and then deliberately dropped once no positive 2 k claim existed to floor (a floor is a hurdle for a positive claim), and those GPU-minutes went to the T1 read {ED} recorded in `raw/chain_after_arms2.sh`, not left as a gap | **SUPPORTED (MEASURED 2026-09-05)** | `{ELL}/2026-09-05-veto-only-fan-safety/RESULT.md` {SEC}7; `raw/veto_verdict_veto2k.json`, `raw/run/s{{0,1}}/veto2k/arm_summary.json` |
| D-RL-VETO-T1-1 | {NE} **THE VETO IS 14{X} GENTLER THAN THE COMPOSED-REWARD STAGE AND STILL REGRESSES DRIVING: EVERY TRAJECTORY-DERIVED T1 FAMILY IS SEPARATED WORSE.** MEASURED 2026-09-05, `taniteval/tools/paired_openloop.py`, **T1 (self-action OPEN loop {ED} never a closed-loop claim)**, `s0/veto200` vs the banked refcv3-40284 base over the shared `ha0` floor: **`void: false`**, all six gates pass, **4,823 shared windows / 141 episodes, 0 dropped on either side**, paired episode-cluster bootstrap (cluster = clip) n_boot 2,000, power adequate. **ADE** `ade_m` **+0.0362** [+0.0261, +0.0460] (base 0.4419, **+8.2 {PCT}**) {MID} `fde_m` +0.0623; **LONGITUDINAL** `accel_mae` **+0.3349** [+0.2719, +0.4052] on a base of 0.6806 (**+49 {PCT}**, the largest single movement) {MID} `speed_mae` +0.0779 {MID} `along_mae` +0.0290; **LATERAL** `heading_mae` +0.2912{DEG} {MID} `yaw_rate_mae` +0.0191 {MID} `cross_mae` +0.0154; **TACTICAL** `traj_lon_correct` **-0.0889** [-0.1145, -0.0655] {MID} `traj_lat_correct` -0.0025 **ns**. All separated except the last. {ARR} **The ADE guard FAILS.** For scale, the full composed-reward arm was **+0.5014 m (+113 {PCT})** on the same corpus (`D-RL-FANSAFE-4`), so the veto is ~14{X} gentler and still a regression {ED} and the axis is the same one the T0 readout flagged (`sel_peak_g` up while the fan's mean came down): a constraint pushed off the violating region takes the SELECTED path with it. {W} The `TAC_declared_*` (6) and `STR_route_*` (3) rows reading **0.0000 [0, 0]** are **STRUCTURAL ZEROS**, never "no harm": those heads are in `forbidden_prefixes` and frozen, so identical inputs give identical outputs {ED} an identity (`H-ECHO-4` class). {W} **ONE SEED**: `s1/veto200`'s T1 was still rolling; every row is necessary-not-sufficient under `H-ESTIM-SEED-1` and the artifact carries an explicit `_LIMIT` field. The same arm's **T0** sel-ADE reads +0.00328 (s0, ns) vs +0.01817 (s1, sep), a 0.0149 floor on a DIFFERENT tier {ED} not transferable, but evidence that this arm's ADE effect is seed-sensitive | **SUPPORTED (MEASURED 2026-09-05)** {ED} one seed; finish with `raw/read_t1_families.py --s0 {ELL} --s1 {ELL}` | `{ELL}/2026-09-05-veto-only-fan-safety/RESULT.md` {SEC}11; `raw/run/paired_s0-veto200_vs_base.{{json,md}}`, `raw/run/eval/refcv3-40284-s0-veto200.{{json,md}}`, `raw/t1_families_veto200.json` |
| D-REFC-KINGATE-1 | {ST}{ST} **THE DELIVERABLE: A TOP-2 KINEMATIC GATE MAKES refcv3's DRIVEN PATH 31 {PCT} LESS ENVELOPE-VIOLATING AT NO MEASURABLE ADE COST {ED} ZERO TRAINING, ZERO NEW PARAMETERS, ZERO NEW PERCEPTION.** MEASURED 2026-09-05 (`stack/scripts/rl_fan_rerank_probe.py`, **480 EVAL windows / 138 episodes**, 139 lead windows, paired episode-cluster bootstrap n_boot 4,000; the model is never retrained and the SAME 128 emitted candidates are re-ranked). `gate2` = keep the model's own **top-2 by `sel_score`** (so the semantic/tactical ranking the network learned is preserved) and choose between them by **`feasibility + comfort`**: selected-path `envelope` **0.1062 {ARR} 0.0729 (-31 {PCT} relative, SEPARATED)**, `peak_g` **0.1815 {ARR} 0.1459 g (-20 {PCT})**, and `ade_m` **0.4742 {ARR} 0.4705, delta +0.0037 in the MODEL's favour and NOT separated** {ED} no measurable ADE cost. It changes the pick on **48 {PCT}** of windows, so it is not a rounding artifact of rarely intervening. The frontier is measured: `gate4` -41 {PCT} envelope for -0.0230 m ADE (ns), `gate8` -45 {PCT} for -0.0452 m (sep), `kin_only` -45 {PCT} for -0.0655 m (sep). {NE} **ADMISSIBILITY: the kinematic score reads NO SCENE INPUT** {ED} `feasibility` and `comfort` are functions of the candidate's own waypoints only (no lead, no obstacle track, no ego state), so the gate is deployable under the vision-only-at-inference rule with no new perception. `reward_full` is reported for contrast because it is NOT: it needs the lead track and destroys ADE (0.9474). {ST} **The `oracle` row explains why this works**: the fan's best-ADE candidate is the LEAST drivable one (`sel_envelope` **0.1437** vs the model's 0.1062 and the gate's 0.0729), so an ADE-neutral gate buys feasibility by discarding candidates that were never worth their ADE {ED} `D-RL-FANSAFE-1` reproduced from the other side. {W} Quoted from a QUARANTINED artifact whose `lambda_sweep` block alone is withdrawn (RETRACTION #30); the selection-rule numbers touch none of the withdrawn operand and are re-confirmed by the corrected re-run | **SUPPORTED (MEASURED 2026-09-05)** {ED} the combined gate-on-veto'd-checkpoint measurement is queued (`raw/fan_rerank_veto200s0.json`) | `{ELL}/2026-09-05-veto-only-fan-safety/RESULT.md` {SEC}8; `stack/scripts/rl_fan_rerank_probe.py`, `raw/fan_rerank_WITHDRAWN_wrong_lambda_operand.{{json,log}}`, `raw/fan_bank_base_240w.npz` |
""".replace("{ST}", ST).replace("{NE}", NE).replace("{W}", W).replace("{ED}", ED) \
   .replace("{X}", X).replace("{ARR}", ARR).replace("{ELL}", ELL).replace("{SEC}", SEC) \
   .replace("{MID}", MID).replace("{PCT}", PCT).replace("{DEG}", chr(176))

raw = open(PATH, "rb").read()
is_crlf = CRLF.encode() in raw
s = raw.decode("utf-8").replace(CRLF, LF)
if "D-REFC-KINGATE-1" in s:
    raise SystemExit("[register] final rows already present")
lines = s.split(LF)
idx = [i for i, l in enumerate(lines) if l.startswith("| D-REFC-OFFSET-FEAS-1 |")]
if len(idx) != 1:
    raise SystemExit("[register] expected exactly one D-REFC-OFFSET-FEAS-1 row, found %d" % len(idx))
new_rows = [r for r in ROWS.split(LF) if r.strip()]
lines[idx[0] + 1:idx[0] + 1] = new_rows

# resolve H-VETO-FAN-1
j = [i for i, l in enumerate(lines) if l.startswith("| H-VETO-FAN-1 |")]
if len(j) == 1:
    old = "**OPEN " + ED + " pre-registered 2026-09-05, arms running; result rows follow in the same package**"
    new = ("**RESOLVED " + ED + " FAILURE against the committed SUCCESS text (2026-09-05).** SUCCESS "
           "needed a negative quotable `fan_peak_g_mean` AND one of `top32_infeasible`/"
           "`sel_infeasible` AND no `ade_m` regression beyond the replicate floor. Only the "
           "FIRST holds: `top32_infeasible` is WITHIN-NOISE (-0.0064/-0.0011 vs a 0.0053 seed "
           "floor) and T1 `ade_m` regresses **+0.0362 m, separated**. At 2,000 steps nothing is "
           "quotable at all. " + ST + " The fan IS safer (`fan_peak_g_mean` -0.098 g, three runs); "
           "the DRIVEN path is not " + ED + " and the thing that does make the driven path safer "
           "today is `D-REFC-KINGATE-1`, at zero training")
    if old in lines[j[0]]:
        lines[j[0]] = lines[j[0]].replace(old, new)
        print("  [ok]   H-VETO-FAN-1 resolved")
    else:
        print("  [warn] H-VETO-FAN-1 status not matched verbatim; left unchanged")

out = LF.join(lines)
open(PATH, "wb").write((out.replace(LF, CRLF) if is_crlf else out).encode("utf-8"))
print("[register] %d final rows inserted (%s)" % (len(new_rows), "CRLF" if is_crlf else "LF"))
