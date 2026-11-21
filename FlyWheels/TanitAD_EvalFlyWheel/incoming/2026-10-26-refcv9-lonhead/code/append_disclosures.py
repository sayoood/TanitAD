"""append_disclosures.py -- adds the deviations / disclosures and the two post-hoc DESCRIPTIVE diagnostics (LABEL_LEVEL_DIAG, TRANSFER_DIAG) to
result/LONHEAD_VERDICT.{json,md} without touching any bar, word or number lh_bars.py wrote.  NEW file."""
import json
from pathlib import Path
R = Path(__file__).resolve().parent.parent / "result"
v = json.loads((R / "LONHEAD_VERDICT.json").read_text(encoding="utf-8"))
lab = json.loads((R / "LABEL_LEVEL_DIAG.json").read_text(encoding="utf-8"))
tr = json.loads((R / "TRANSFER_DIAG.json").read_text(encoding="utf-8"))
DEV = [
 "D1 one label file per head serves both LHS inference seeds: eval z_tac and the fed v0 are bit-identical across seeds 0/1 (asserted, max |diff| 0.0 on all 4,754 keys), as SELFSHARP's s1 roll reused its s0 labels.",
 "D2 LHS_FACT8 was produced TWICE and asserted bit-equal on S6 at both seeds: (a) the full SHIP-R8 Amendment-5 forward (code/sub_fx_lhs = ship_r8_sub_fx with a 3-line diff: the trace source; outputs in fwd_lhs/, nothing written into ship_r8_sub_fx) and (b) the light composer (SF.compose + RS.retime_plan, registered deploy table) that reproduces the LANDED FACTORED_DEPLOY8 bit for bit from the SELFSHARP traces at both seeds (plan8 / pos8 / S6 pos; COMPOSER_VALIDATION.json, with a mutation that must and does differ).",
 "D3 DER_FACT8 (seed 0 only, as the prereg rolls it) comes from the light composer; its compliance row and E1 / E3 are NOT computed because the SHIP-R8 forward reads BOTH inference seeds.",
 "D4 G's BASE-os = the LANDED OSONLY_DEPLOY8 S6 positions, asserted == the os-only plan staged (lane AND l2_common.apply_stage) and reproducing the K_NONE literals (0.46948, 0.13955) s0 / (0.47105, 0.13739) s1.",
 "D5 A when both F-bars pass would need a reading of 'the frontier bar that passes'; coded as (F1 and A1) or (F2 and A2). Moot: neither F-bar passed at either seed.",
 "D6 the shared gpu_lock returned ok:true while its own gate field read ok:false (free RAM 5.92 GB < its 8.0 GB rule; no other python compute app, GPU 3.5 GB used by desktop apps). Rolls ran one at a time under the lock and completed 4,754/4,754 each (2,091 / 2,022 / 1,999 s).",
 "D7 the roll logs (logs/roll_*.log) are written by the reused upstream roll code and carry 12-char clip-id prefixes: they are EXCLUDED from the manifest and must not be landed.",
 "D8 K-HEAD's v0 = the z_tac bank's fed v0_in (== extras v0 exactly, max |diff| 0.0)."]
v["deviations_and_disclosures"] = DEV
v["post_hoc_descriptive"] = {"NOT_A_BAR": "post-hoc, not pre-registered; read for the NEXT prereg only", "label_level": lab, "label_to_plan_transfer": tr,
  "reading": ("LH's argmax IS a better decel DECISION at the label level (GT-decel windows forced to a decel class 0.4961 vs native 0.4163; GT-cruise windows "
              "forced to decel 0.1288 vs 0.1358; n 639 / 1,856, seed 0), yet the composed plan does not move (MD 0.3912 vs 0.3897, FD 0.1638 vs 0.1503). "
              "The label -> plan transfer is leaky both ways: on GT-decel windows whose label is NOT decel the plan still decelerates 0.357 (LHS) / 0.450 "
              "(FACTORED), and on GT-cruise windows labelled decel it decelerates 0.577 / 0.456. The windows LH adds to 'decel' are ones the os-only plan "
              "already decelerates on more often (base-os 0.763 vs 0.692), so the marginal planner gain is small, and its decel labels on cruise convert "
              "to braking more often. => the binding constraint is the progress the forced class implies, not the class (prereg s5's committed branch).")}
v["next_lever"] = {"lever": "class-conditioned progress head on the fresh split (prereg s5 committed branch for 'head better, F1 and F2 FAIL')",
                   "blocked_on": "a Master-Mind-sealed pre-registration (s5: 'pre-registered first'); 0 GPU until sealed",
                   "design_inputs_measured_here": ["label-level decel TPR/FPR of LH vs native (LABEL_LEVEL_DIAG.json)",
                                                   "label->plan transfer table (TRANSFER_DIAG.json)",
                                                   "rate-matched LH P(decel) threshold: TPR 0.4695 / FPR 0.1175 vs native argmax 0.4163 / 0.1358 (dTPR +0.053 [-0.021, +0.127], dFPR -0.018 [-0.062, +0.028]; episode bootstrap) -- a label-source arm, separate from the progress head"]}
(R / "LONHEAD_VERDICT.json").write_text(json.dumps(v, indent=1, default=str), encoding="utf-8")
md = (R / "LONHEAD_VERDICT.md").read_text(encoding="utf-8").replace("ZZLHVERDICTENDZZ\n", "")
add = ["## Beside: LHS_FACT8 vs FACTORED_DEPLOY8 on the four families (s0, S6 compliant; not a bar)",
       "ADE +0.0779 m [+0.0228, +0.1389] SEP worse; FDE +0.2226 SEP worse; LONGITUDINAL speed MAE +0.0419 / along MAE +0.0854 / accel MAE +0.0182 / along |final| +0.2441 "
       "all SEP worse; LATERAL all unseparated; TACTICAL traj_lon_correct -0.0228 [-0.0395, -0.0061] SEP worse, traj_lat_correct unseparated; vs BASE-os no row separated worse (G PASS). "
       "STRATEGIC and distance-keeping: n/a (no route label / no lead track in the banked dumps; n = 0).", "",
       "## Post-hoc DESCRIPTIVE (not pre-registered, not a bar) -- why a better-ranking head did not move the plan",
       v["post_hoc_descriptive"]["reading"], "",
       "## Next lever", f"{v['next_lever']['lever']} -- BLOCKED on: {v['next_lever']['blocked_on']}.", "",
       "## Deviations and disclosures"] + [f"* {d}" for d in DEV] + ["", "ZZLHVERDICTENDZZ", ""]
(R / "LONHEAD_VERDICT.md").write_text(md + "\n".join(add), encoding="utf-8")
print("ok")
