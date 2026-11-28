# Currency audit — eval packages since 2026-09-03 outside Architecture & Inference Research

`Produced 2026-09-19 by a read-only sub-agent of stream E4 (leaderboard currency) — 140 tool uses,
443,250 tokens, ~25 min. Saved by the EvalFlyWheel orchestrator from the agent's completion report
because E4 was PAUSED (weekly usage 87 %) and the agent's transcript file on disk was EMPTY (0 B) — this
file is the only durable copy. Evidence class of every row here: INHERITED (the sub-agent's reading)
until E4 re-verifies it against the named artifact. Nothing was edited or staged by the sub-agent.
Tree at the time: branch agent/arch-inf-20260803 @ 37645fc; other agents were writing during the read.`

## Headline (sub-agent's words, condensed)

- **No TanitAD model has an EPDMS/PDMS score anywhere in scope.** The five NavSim packages quote only
  published/inherited numbers for other teams' agents, and none gives a harness SHA.
- **Both `2026-09-04-*closedloop*` folders are closed loop ON PERCEPTION ONLY**: a gsplat renderer of ONE
  NuRec scene, run in-process (`"transport": "inproc"`), NOT through AlpaSim's gRPC runtime. The PI's
  2026-09-02 definition is met (the plan moves a simulated vehicle that drives the next render), but
  neither doc stamps a T-tier, and on that scene no model beats a straight line at constant speed on ADE.

Path roots: **TR** `taniteval\results\` · **BE** `TanitAD Research Lab\Benchmarks & Evals\Research\` ·
**BEI** `…\Benchmarks & Evals\Implementation\incoming\` · **DO** `…\Deployment & Optimization\Research\` ·
**DE** `…\Data Engineering\Research\` · **TD** `…\Tools & DevEnv\Research\` ·
**AI** `…\Architecture & Inference\Implementation\incoming\`. † = confirmed committed in HEAD 37645fc.

## Package table (61 in scope; the sub-agent's classification)

| package | kind | arm(s) + step | tier · loop (doc's words) | grid | estimator | headline number(s) | four families? | primary raw artifact | superseded / retracted? |
|---|---|---|---|---|---|---|---|---|---|
| TR\2026-09-04-refcv3-closedloop\ | ARM-EVAL (closed + matched open loop) | refcv3 @40,284; refc-base @29,999; flagship-v1 @29,999 | "closed loop on perception", no T-number · CLOSED (perception only); open half OPEN | closed 435 paired windows / 9 clusters (9 starts of ONE clip), scene 00040136-…, `--condition empty`; open 190 (170 paired) | paired episode-cluster bootstrap, 2,000 | closed `ade_0_2s` refcv3 − refc-base **+0.2185 [−0.6198, +1.1586]**; `yawrate_err_rads` +0.0074 [+0.0027, +0.0119] (refc-base better); `manoeuvre_plan_eq_logged` −0.0736 [−0.1309, −0.0244] (refc-base better); refcv3 open→closed 0.8431 → 2.8755 m (3.41×) | all five blocks; distance-keeping UNAVAILABLE n=0; STRATEGIC UNAVAILABLE on this scene | raw\V3_refcv3_vs_refc-base_empty.json†, raw\V3_refcv3_vs_flagship_empty.json, raw\OL_*.json; refcv3 ckpt on Thor (OFF-REPO) | verdict "refcv3 drives closed-loop" superseded the same day by closedloop-floor §4.2 and MODEL_REGISTRY §1.12 |
| TR\2026-09-04-closedloop-floor\ | CONTROL-OR-FLOOR (+ re-read of both panels as margins) | floors `cl_ha0`, `cl_ha`, `cl_ha0_ext`; refcv3 @40,284, refc-base @29,999, flagship-v1 @29,999 | no T stamp · CLOSED (perception, non-reactive); §5 OPEN | closed 435–437 / 9 clusters / 1 scene; open 170 | episode-cluster bootstrap over 9 starts, paired | closed `refcv3 − cl_ha0` +0.1307 [−0.6213, +0.8809]; `refc-base − cl_ha0` −0.0897 [−0.4029, +0.1994]; `flagship-v1 − cl_ha0` **+7.0745 [+5.1151, +9.0440]**; open `refcv3 − cl_ha0_ext` **+0.3558 [+0.0224, +0.6981]** (floor wins) | yes per family; distance-keeping n=0; floors' manoeuvre/route heads n=0 by construction | FLOOR_SUMMARY.json†, raw\metrics\{V3,HQ,OL,FL}_*.json | re-qualifies the 2026-08-03 +7.1642 headline |
| TR\2026-09-04-refcv3-smoothness\ | ARM-EVAL (smoothness) | refcv3 @40,284 vs refc-base @29,999 (+flagship-v1, GT, controls) | "T1 open loop (self-action)" · OPEN | 170 paired / 1 clip; 4,438 of 4,823 / 141 eps | window means, ratios, **no CI** | jerk ×2.80, dκ/ds ×1.44, zig-zag ×1.77; oracle-in-vocab ADE 0.9433 vs 0.4369 (CV line 0.6780) | no | raw\FINAL_smoothness.json† | no |
| TR\RESULT-refcv3-40284-stratified.md | ARM-EVAL (vs controls) | refcv3 @40,284 `os` vs `ha`, `ha0` | "T1 · OPEN LOOP" | 4,823 / 141; manoeuvre 1,583/103; straight 3,240/139 | paired episode-cluster, 2000, seed 0 | `os − ha` manoeuvre **+0.2530 [+0.2007, +0.3058]**; straight/constant +0.0882 [+0.0701, +0.1062] | LON/LAT/TAC; STRATEGIC + distance-keeping absent WITHOUT a reason | TR\refcv3-40284-stratified.json† | no ("incomplete", not retracted) |
| TR\paired-openloop-refav1-vs-refcv3-20260903T204638Z.json | ARM-EVAL, cross-model read **VOID** | refav1 `cl` (ckpt_ep2, step 1000) vs refcv3 `os` (ckpt_30000); floor `ha0` | T1 · OPEN | 120 shared / 20 eps; instants {1.0, 2.0} s | paired, 2000, seed 0 | `void: true` — "refav1:cl is a CONSTANT-VELOCITY plan on 100.0% of the shared windows"; refcv3:os − ha0 `ade_m` −0.2045 [−0.3997, −0.0178] (sep) | ADE/LON/LAT/TAC; STRATEGIC REFUSED; no distance-keeping | this JSON†; dumps OFF-REPO | VOID (D-REFAV1-PAIRED-READ-VOID) |
| BE\2026-09-03-paired-openloop-refav1-vs-refcv3\ | prose for the row above (VOID) | as above | T1 matched · OPEN | 120 / 20 | as above | as above | STRATEGIC refused (adapters disagree on `route_label`) | raw\paired_table.md | VOID |
| BE\2026-09-03-refav1-kinematic-contract-lateral\ | INSTRUMENT | refav1 banked dump arms `ol`/`cl`/`ha` | no stamp · OPEN | 140 / 20 | episode-cluster 2,000; paired | `h_c1_wb` LAT curved 0.0600 [0.0427, 0.0800] vs shipped 0.7156; paired −0.6556 [−0.8701, −0.4349] | no | raw\kin_contract.json | no |
| BE\2026-09-03-refcv3-arm\ | INSTRUMENT | random-init fixture | "NOT a refcv3 result" | 42 windows | — | one anchor on 42/42 while `trivial_frac` 0.0000 | n/a | raw\refcv3_arm_fixture.json | no |
| BE\2026-09-03-refcv3-epoch-conclusions\ | T0 in-training monitor | refcv3 series to 25,650 | "T0" | 160 windows, 55 evals | segmented fits | "`eval_traj` is an ORACLE-ANCHOR-SELECTED error"; "is not an ADE" | no | raw\metrics.jsonl | its `DEFAULT_TIERS` lacks `ha0` claim contradicted by 2026-09-03-refcv3-arm |
| BE\2026-09-05-nav-compliance-metric\ | INSTRUMENT (+ early smoke) | refcv4b @9,500 `os`; `ha0`, `ha0_ext`, `ha` | "T1* … EARLY-TRAINING DIAGNOSTIC" · OPEN | 1,860 / 141; informative 141 / 22 | paired | plan-compliance with TRUE command 0.390 [0.303, 0.472]; nav-SHUFFLE Δ +0.099 [+0.041, +0.167]; `g_str` Δ 0.0000 [0, 0] | STRATEGIC block only | raw\navcomp_refcv4b-9500.json† | no |
| BE\2026-09-06-b1-agent-join\ / -b1-train-join\ | label artifacts | — | "Tier N/A" | 139/141 eval clips | — | "139 / 141 B1 EVAL clips joined" | n/a | raw\b1eval_agents.jsonl.xz† (train: `.meta.json` only) | no |
| BE\2026-09-07-refcv5-v2-comparison\ | ARM-EVAL (+ refcv4b control) | refcv5-v2 @40,284 (roll seeds 0/1) vs refcv4b @40,284; `ha`/`ha0`/`ha0_ext` | "T1 (self-action open loop) on every number" · OPEN | 4,823 / 141 | paired, 2000, seed 0 | raw only: ADE 0.3079 [0.2795, 0.3390]; `os − ha0_ext` +0.0205 [0.0043, 0.0390]; `os − ha` +0.0082 [−0.0082, 0.0267]; `refcv5-v2 − refcv4b` +0.0114 [0.0059, 0.0172]; HEADLINE FAIL. refcv4b `os` 0.2975 (A40) vs **0.2965** (RTX 4060) | yes; STRATEGIC route_acc 0.7708 but "route_pred identical under nav-shuffle 1.0" | raw\refcv5-v2_vs_refcv4b.{json,txt}†, raw\refcv5-v2-seed1_vs_refcv4b.json† | RETRACTION 2026-09-11: pairing refcv4b's +0.0101 with +0.0205 retracted (use +0.0091 [−0.0055, +0.0254]); 18,472 never-trained params (zerograd row) |
| BE\2026-09-07-refcv5-v2-inference-floor\ | CONTROL (inference-seed floor) | refcv5-v2 ckpt_15000, seeds 0/1/2 | "T1 … MID-TRAINING" · OPEN | 4,823 / 141 | episode-cluster 2,000 | `os − ha0_ext` +0.11342 / +0.11288 / +0.11269 (range 0.00073); controls range 0.00000 | partial | raw\floor_3seed.json† | header PARTIAL overridden by §4 COMPLETE |
| BE\2026-09-09-navsim-split-reconciliation\ | PROTOCOL (literature) | external only | none | — | — | see NavSim section | n/a | raw\search_log.md | "≤ 45.0" cap later superseded |
| BE\2026-09-10-navhard-split-stamp\ | PROTOCOL | external | "navhard (v2, two-stage, pseudo-closed-loop)" | — | — | see NavSim section | n/a | raw\search_log.md | PDM-Closed "51.3" reading superseded 09-13 |
| BE\2026-09-13-pdmc-stage2-provenance\ | PROTOCOL | external | "external NAVSIM-v2 EPDMS — not a TanitAD tier" | — | — | see NavSim section | n/a | raw\stage_subscores.md | "DrivoR (base, no TOAD) 54.6" label corrected 09-15 |
| BE\2026-09-15-epdms-stage1-fingerprint\ | PROTOCOL (+ fingerprint instrument) | external | same | — | — | see NavSim section | n/a | raw\s1s2_fingerprint.json† | caveated by 09-17 (the fix can move EP) |
| BE\2026-09-17-b1-agent-join-3d\ | label artifact | — | none | 905,512 agents / 139 clips | — | "`center_z` is the cuboid CENTRE" | n/a | TanitAD-artifacts\b1-agent-join-3d-20260917\ (ONE place, OFF-REPO) | no |
| BE\2026-09-17-navsim-fix151-mechanism\ | PROTOCOL | external | same | — | — | see NavSim section | n/a | raw\search_log.md | "Nothing is retracted" |
| BE\2026-09-18-ladder-l1-row14\ | PROTOCOL (admissibility ladder) | — | quotes EVAL_DOCTRINE: T1 = "self-action OPEN loop" | — | — | no EPDMS | n/a | raw\search_log.md | no |
| BE\2026-09-18-refcv6-metric-family-coverage\ | INSTRUMENT (audit) | 15 refcv6 tiny runs, 1–40 steps | "in-training MONITOR at T0" | 16 windows | — | "174 distinct keys, ZERO map to any of the 16 required criteria" | no | raw\refcv6_family_coverage.json | partly overtaken by w0–w3 |
| BE\2026-09-19-w0 … w3 (JSON only) | INSTRUMENTS | real refcv6 ckpt / one-step model | "_NOT_A_CAPABILITY_CLAIM" | 9 / 999 / 144 windows | paired | w0 "loads STRICTLY"; w1 `paired_ha_minus_ha0.ADE −0.3245 [−0.5745, −0.0119]`; w2 "THE JOIN WAS NEVER BROKEN"; w3 `ha0` headway 25.2809 (n 75), `os` n=0 | w1: LON/LAT/TAC present, STRATEGIC unavailable | raw\w{0..3}_*.json (w1†) | no |
| BEI\2026-09-03-refcv3-arm-UNVERIFIED\ | rescued draft | — | "Do not import it, do not quote" | — | — | none | n/a | refcv3_arm.py draft | superseded by `taniteval\tools\refcv3_arm.py` |
| BEI\2026-09-05-rung-a1-harness\ | INSTRUMENT | — | — | — | — | two `ha0_ext` kinematics diverge up to 1.86 m at 6 s | n/a | raw\integrator_divergence.txt | no |
| DO\2026-09-05-collision-projection\ | T0 readout of a banked fan | refcv3 @40,284 fan | "T0 / T1-adjacent … never a driving claim" | 240 / 121 × 128 | bootstrap 4000 | `fan_contact` 3.428 % → structural 0.000000 % | no | raw\contact_projection_panel*.json | §8 "2.11× selection gap" inference RETRACTED 2026-09-05 |
| DO\2026-09-05-feasible-decode\ | ARM-EVAL (zero-training decode lever) | refcv3 @40,284 `os`: base vs `proj07` (+`projoff`) | "T1 = self-action OPEN loop" | 4,823 / 141 | paired, 2000, floor `ha0` | `ade_m` +0.0012 [0.0002, 0.0024] (sep); `LAT_yaw_rate_mae_radps` 0.2176 → 0.0427 (−0.1749 [−0.2563, −0.1088]); envelope 0.0865 → 0.0000 | yes | raw\paired_proj07_vs_base.json†; dumps OFF-REPO | SPEC claim retracted (D-FEASDEC-SPECERR-1); headline not |
| DO\2026-09-05-kinematic-gate\ | T0 selection rule | refcv3 frozen: model vs `gate2` (+`gate1`) | "T0 throughout" | 2 × 400 / 70 eps | paired 4,000 | gate2 `sel_envelope` 0.1075 → 0.0775 (−0.0300 sep) / 0.1100 → 0.0900 (−0.0200 sep); `ade_m` +0.0064 / −0.0099 (ns) | yes at T0 | raw\kingate_base.json† | corrects D-REFC-KINGATE-1; T1 leg NOT run |
| DO\2026-09-05-refc-rl-readiness\ | DESIGN-PREREG (+cost) | frozen refcv3 | "T0 or instrument probe" | — | — | "RUNNABLE-WITH-FIXES"; 0.862 s/step, peak 1.78 GB | n/a | raw\preflight.json | no |
| DO\2026-09-05-rl-generator-collisions\ | T0 RL post-training | `coll200` s0/s1 vs base; `ctrl0`, `ctrl_null` | "T0 on the emitted fan" | 240 / 121 | paired + replicate/zero-info floors | `fan_contact` Δ s0 −0.001693 [−0.003646, −0.000260] (sep), s1 ns → FAILURE vs committed SUCCESS text | no ("T1 … cancelled") | raw\coll_verdict_coll200.json† | motivating top32 gain never real; endpoint retired |
| DO\2026-09-05-swept-collision\ | INSTRUMENT | refcv3 base fan | "not a driving number" | 240 × 128 × 5 | — | 289 candidates, 27.7 % of lead windows | n/a | raw\swept_vs_point.json | no |
| DO\2026-09-05-veto-only-fan-safety\ | ARM-EVAL (RL; T0 + T1) | `veto200` s0/s1, `veto2k`, `ctrl_null`; `gate2` on base | "T1 (self-action OPEN loop)" | T1 4,823 / 141; T0 240 / 121 | paired + seed-replicate and `ctrl_null` floors | T1 `ade_m` +0.0362 / +0.0475 (floor 0.0113) → FAILURE; `gate2` `sel_envelope` 0.1062 → 0.0729 (−31 %), `ade_m` +0.0037 ns | yes at T1 | raw\run\paired_s{0,1}-veto200_vs_base.json†, raw\t1_families_veto200.json | one re-rank file WITHDRAWN (RETRACTION #30) |
| DO\2026-09-09 / 09-10 i1 prescreens | DESIGN-PREREG (literature) | — | none | — | — | GuideFlow scorer "+15.9 EPDMS" (no split named); 2601.00844 success rates | n/a | raw\search_log.md | no |
| DO\2026-09-13 … 09-18 i1 value-geometry screens (4 pkgs) | OTHER (representation diagnostics) | frozen refcv5-v2 trunk tokens | "Tier: none" | 17–139 clips | clip-cluster bootstrap | ρ 0.4223 [0.3943, 0.4498] vs bar 0.50 → PARTIAL; 09-15 VOID (twice); 09-17 VOID-POWER; 09-18 ID-TRUNK +0.2088 [+0.1329, +0.2782] | n/a | raw\vgeo*.json | 09-15 VOID; 09-17 VOID-POWER |
| DE\2026-09-03-steer-curvature-interface\ | INSTRUMENT | refav1 `ol` replay, action-units ON/OFF | none | 140 | as kin-contract | 0.7156 → 0.0600 m curved LAT | no | raw\interface_summary.json | no |
| DE\2026-09-13-qwen-drive-usage-review\ | external perception teacher | Qwen-Drive-1.0-4B | "no driving tier" | 4 nuPlan demo frames | none | 54/57 GT boxes; map mIoU 0.799; occupancy IoU 0.389 | n/a | raw\demo_control_metrics.json† | no |
| DE\2026-09-16-256x1024-cache\ | eval-cache build | — | none | 139 eval clips | — | caches 8.069 / 11.398 GB | n/a | manifests in-repo; caches ONE PLACE (TanitAD-artifacts) | no |
| DE\2026-09-18-refcv6-eval-parity-clean\ | SPLIT | — | none | 139 → **128** (64/64) | — | "QUOTED for a held-out read is 128 clips, never 139" | n/a | raw\parity_clean_report.json† | narrowed to 124 by a0 for SAM3-map evals |
| DE\2026-09-19-a0-clean124-split\ | SPLIT | — | none | 124 = 62/62 | — | 139 − 11 in parity TRAIN − 4 without a SAM3 map | n/a | raw\a0_clean124_stamp.json† | no |
| DE\2026-09-19-a1-per-half-occupancy-floor\ | CONTROL (trivial "all drivable") | — | none | halfA 62 / 675 frames; halfB 62 / 679 | no CI | `corpus_floor` 0.3412; halfA 0.3466957; halfB 0.3388181 | n/a | raw\a1_per_half_floor.json† | no |
| TD\2026-09-07-thor-checkout-drift-a13\ | eval provenance | refav1-b1-v72-ep3-speed | none | 130-file import closure | — | "a re-run from today's repo is not a replication of that arm" | n/a | raw\A13_DRIFT_TABLE.json | brief's premise stale |
| AI\2026-08-30-t1-first-v7-read\ | ARM-EVAL | v7-TINY `emao14_30k`, `o14fut30k`, `emao14_30k_tauramp` @30k, `cl` vs `ha` | "T1 (PRIMARY)"; says "closed-loop arm" → **OPEN** under 2026-09-02 | 40 eps / 6,924 windows | episode-cluster | `ade_dense_m` cl/ha 14.069/13.879; 14.293/14.116; 13.864/13.806 | partial; strategic UNAVAILABLE | raw\t1_*.json† | "closed-loop" wording superseded by 2026-09-02 ruling |
| AI\2026-08-30 … 09-11 (11 pkgs: action-divergence, drift-decomposition, k1-overlap, provenance-rescue, v7-labels-consumer, mme11-o1-read, trunk-anchor, v7-eval-exclusion, label-vocab-audit, refcv5v2-zerograd-heads, agentgt-anchor-repair, untrained-param-gate) | diagnostics / instruments | various | T0-DIAGNOSTIC or none | — | — | e.g. zerograd: "18,472 parameters never trained"; untrained-param-gate: `scorer.goal_point` untrained and exactly zero in THREE arms | n/a | per package | none superseded except as noted in them |

Left out (neither score a model nor define an eval split/floor): Data Engineering — agent-join-into-batch,
refcv5-labels-and-strategy, control-relevant-clip-value, kairos-curation-fourth-route,
qwen-drive-perception-sample, posted-speed-limit-supplier, sam3-only-road-map, speed-sign-reader-selection,
flywheel-negatives, kitscenes-map-speed-values, cot-sign-values, c3-source-transfer-price,
c4-hf-quota-check, d-sync-divergence-resolution, deferred-merges, nuplan-posted-limit-coverage;
Tools & DevEnv — refcv4b-pace-regression, backlog-drift.

## The two `*closedloop*` folders — what they are

Harness `stack\experiments\alpasim-gsplat\closedloop_drive.py` (closed) / `openloop_drive.py` (open) →
`cl_metrics.py`, on Thor: render(ego pose) → f-theta canonicalise → policy → waypoints → pure-pursuit →
kinematic bicycle → new ego pose → render. gsplat on ONE NuRec 3DGS volume (scene
`00040136-e651-4abd-991d-0655ccda9430`), 9 starts × 50 ticks. **Not AlpaSim's runtime** (`"transport":
"inproc"`, no `--addr`). Closed arm x/y/yaw from a planar bicycle; z/roll/pitch copied from the nearest
logged pose; controller reads only the 0.5 s waypoint; commanded acceleration executed as 0.4·a. "empty"
condition: no constructed lead, agents replayed/scripted, no map, no collision response. ⚠️ **Naming trap:**
the suite's `ha0_ext` = constant (a0, κ0) (`refav1_arm.hold_ext_controls`) — that is the `cl_ha` law, not
`cl_ha0_ext` (constant a0 + constant yaw rate, CTRA). Read: **CLOSED on perception only, non-reactive, one
scene, within-sim relative**; the PI-definition mapping is in `Project Steering/CLOSED_LOOP_FEASIBILITY_2026-09-03.md`.

## NavSim packages — every EPDMS/PDMS number they contain (all EXTERNAL agents; no harness SHA anywhere)

- 09-09 split reconciliation: navhard cluster "23.1-45.0 · 38.0 · 36.9" (GuideFlow, IDOL, RAP-DINO); "NAVSIM v2"
  12k scenarios: 89.3 Latent-WAM · 86.1 DriveVLA-W0 · 85.1 Epona · 84.8 World4Drive (paper names no split);
  "ours, unstamped 55.5 · 56.3" = DriveFuture and DrivoR, INHERITED.
- 09-10 navhard split stamp: navhard PDM-Closed 51.3 · Latent TransFuser 23.1 · MLP 12.7 · Constant Velocity
  10.9; navtest 84.8 / 85.1 / 86.1 / 86.4 / 89.3; Drive-HWM 86.4 EPDMS; "93.8 PDMS is NAVSIM v1".
- 09-13 PDM-Closed stage-2 provenance: PDM-Closed navhard two-stage **51.3 (paper v2, 2025-08-27) vs 56.6
  (v3, "Snapshot from 03/2026")**; v3 Table 2: **CV 11.4 · Ego MLP 14.1** · LTF 25.1 · NavFormer 34.1 · LTFv6
  31.9 · RAP 39.6 · ZTRS 48.1 · GuideFlow 51.5 · SimScale 53.2 · DrivoR 54.5 · PDM-C 56.6; DrivoR 54.6 in TOAD
  Table 6; DrivoR + TOAD 56.3; "not a devkit commit — GitHub releases stop at v2.1.2".
- 09-15 Stage-1 fingerprint (pre-/post-#151): PDM-Closed 51.3 / 56.6; DrivoR base 45.3 / 48.3; +65k 52.3;
  +134k 54.6; +134k+TOAD 56.3; RAP-DINO 36.9 / 39.6; RAP 89.1 PDMS (v1); DriveFuture 55.5 and CLOVER 48.3
  UN-STAMPABLE; GuideFlow 27.1.
- 09-17 fix #151: WA-JEPA "91.7 EPDMS is navtest" (baselines 90.1 / 90.4); "no linked fix commit".
⇒ **ORCHESTRATOR NOTE:** the published navhard reference-agent values (CV 11.4, Ego MLP 14.1, PDM-C 56.6 on
the post-fix v3 snapshot; CV 10.9 / MLP 12.7 / PDM-C 51.3 pre-fix) are exactly the independent cross-check
stream E1 needs for its harness validation on navhard — and the pre/post-#151 split means OUR devkit SHA
decides which column our CV number belongs in.

## (a) Results a leaderboard should carry, with the best primary artifact

1. Closed loop (perception only, n = 1 scene): `TR\2026-09-04-closedloop-floor\FLOOR_SUMMARY.json`;
   arm-vs-arm `TR\2026-09-04-refcv3-closedloop\raw\V3_refcv3_vs_refc-base_empty.json`.
2. Open loop on the same NuRec scene: `FLOOR_SUMMARY.json` (`OL_*`), `…\raw\OL_refcv3_vs_refc-base.json`.
3. refcv3 @40,284 T1 stratified: `TR\refcv3-40284-stratified.json`.
4. refcv5-v2 vs refcv4b @40,284 T1 (FAIL): `BE\2026-09-07-refcv5-v2-comparison\raw\refcv5-v2_vs_refcv4b.json`
   (+ seed1) — carry the zerograd caveat and refcv4b's 0.2975 / 0.2965 spread across rigs.
5. Inference-seed floor: `BE\2026-09-07-refcv5-v2-inference-floor\raw\floor_3seed.json`.
6. refcv3 decode variants and RL arms, T1: `DO\2026-09-05-feasible-decode\raw\paired_proj07_vs_base.json`;
   `DO\2026-09-05-veto-only-fan-safety\raw\run\paired_s{0,1}-veto200_vs_base.json`.
7. T0 only, separate non-driving section: `DO\…kinematic-gate\raw\kingate_base.json`;
   `DO\…rl-generator-collisions\raw\coll_verdict_coll200.json`.
8. Single-arm margin only (cross-model read VOID): refcv3 @30,000 `os − ha0` from the TR paired JSON.
9. Early-training diagnostic, do not rank: `BE\…nav-compliance-metric\raw\navcomp_refcv4b-9500.json`.
10. v7-TINY T1 floor read, relabelled OPEN: `AI\2026-08-30-t1-first-v7-read\raw\t1_*.json`.
11. Note only (ratios, no CI): `TR\2026-09-04-refcv3-smoothness\raw\FINAL_smoothness.json`.

## (b) Contradictions with `Project Steering/MODEL_REGISTRY.md` (for the Master Mind — E4 does not edit the registry)

1. The v7 T1 block (~line 4722) is headed "T1 — THE PROGRAMME'S FIRST CLOSED-LOOP CAPABILITY READ ON THE v7
   LINE" — contradicts the 2026-09-02 ruling and the registry's own §1.12 relabel; still calls `strategic`
   UNAVAILABLE an instrument gap (the package corrected that the same day: val40 carries v7 labels on only
   6 of 40 clips); points to Thor raws although in-repo copies exist.
2. §1.12 banner (line 1274): "neither refav1 nor refcv3 has been through that harness" — contradicted by
   `2026-09-04-refcv3-closedloop` and by the registry's own next paragraph (line 1288).
3. §1.12 calls it "the AlpaSim/NuRec panel" — the rollouts record `"transport": "inproc"`; those closed-loop
   numbers also sit under a heading tagged "SELF-ACTION OPEN LOOP … [TIER T1]" with no T2 stamp.
4. §4.6 quotes refcv4b `os` 0.2975 without a caveat; the registry flags it itself near line 5193; the
   2026-09-07 control reads 0.2965.
5. §4.7's header says "NO RESULT EXISTS AND NONE EVER WILL · SUPERSEDED by §4.8" while its body (~3001–3008)
   carries the refcv5-v2 tables (numbers match raw).
6. Outside the registry: `EVAL_DOCTRINE.md` lines 15–16 and 30 still say "published NO closed-loop number";
   RETRACTION_LOG #14 retracted that sentence and names EVAL_DOCTRINE among the five files it was written into.
7. Absent from the registry (omissions): feasible-decode, veto200 T1 regression, kinematic-gate, coll200,
   the stratified +0.2530, the VOID paired read.

## (c) Readability notes

No unreadable package. `TanitAD Research Lab\Benchmarks & Eval\Research\` does not exist (the only
`Benchmarks & Eval\` is the repo-root one holding LEADERBOARD.md). JSON/code-only packages: w0–w3, a0, a1,
v7-eval-exclusion, qwen-drive-perception-sample, refcv5-labels-and-strategy, refcv4b-pace-regression;
k1-overlap has prose only. `TR\refcv3-40284-stratified.json` must be read as UTF-8 (fails cp1252).
