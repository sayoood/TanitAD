# SPEC_P1P2 -- pre-registration of the candidate-fan probe (P1) and the nav-withhold probe (P2) on refcv7-r101-s0 step 30,000 / navhard

Stream D6 (refcv8 data audit), written 2026-10-04 BEFORE any P1 or P2 number exists. To be registered by the Master Mind (sha256 + time) before any GPU
job of this SPEC is queued or launched. Nothing in this file is tuned on P1/P2 data; every threshold is a literal; every outcome of both probes has a
pre-committed reading in section 6. Evidence classes are stamped in the result as MEASURED (ours + artifact path).

Tier stamp for every number the probes produce: NavSim open-loop benchmark (T1-family, never closed loop), zero-shot PhysicalAI-AV B1 -> nuPlan cameras,
non-parity; device CUDA bf16 ("as_trained" precision), checkpoint `D:/refcv7_eval_kit/ckpt/ckpt_30000.pt` md5 `ac4e4fab87e35b20de24d8d94910d910`.

Why these probes (Rule Zero): the D6 anatomy (RESULT.md) attributes navhard DAC-zero (1,563 scenes) and NC-zero (889 scenes) to plan geometry and speed, but the
LEVER is unidentified for 790 clean-path DAC-zero scenes (LATERAL-DRIFT 346, ON-ROUTE 243, OVER-STEER 170, NO-RECOVERY 31) and for the 843 front-collision NC
scenes. The D5 inventory (NS-4 / NS-5 / gap G1) names the discriminating experiment: does the 117-anchor fan hold a clean candidate, and does the pick take it.
P2 isolates whether the nav input causes the premature-turn and under-turn associations measured in RESULT s4.

## 1. What is exported (the instrument)

An OPT-IN flag `--export-fan <dir>` in a COPY of the NavSim bridge (`D6_navsim_failure_anatomy/code/bridge_fix/`; the files under
`navsim/code/` are not edited). With the flag absent the bridge is bit-identical to the base (section 8, test T-DEFAULT). With it present, for every scene the
bridge additionally writes (from the SAME forward, no extra model call):

* `cands_knots` [181, 8, 2] = `out["r7_candidates"]` (117 anchor-fan members, then 64 WTA proposals; ego frame; knots at 0.5, 1, 1.5, 2, 3, 4, 5, 6 s);
* `cands_poses` [181, 8, 3] = each member through the bridge's own `knots_to_navsim` (the conversion that produced the banked seam plans);
* `sel_idx` (the E9 pick, an index into the first 117), `sel_idx_base`, `sel_score_v3` [117], `reach_keep` [117], `r7_score` [181] (the refcv7 disentangled
  scorer's aggregate), `r7_sel_idx` (the scorer's own argmax over the 181 after its reach / ceiling mask), `r7_pick_is_wta`, `traj_r7` poses (the scorer's pick).

If `r7_candidates` is absent from the output the export aborts the run (no silent fall-back to the 117-fan alone).

## 2. P1 -- token sets (files in `raw/spec_tokens_*.txt`, sha256 in `raw/spec_token_sets.json`; built by `code/d6_make_spec_sets.py` from the banked A1-30k scores)

| set | definition | n | sha256 (first 16) |
|---|---|---|---|
| S1 | A1-30k DAC = 0 AND the PDM-Closed reference passes DAC (a clean path exists) | 1,136 | 1fa25eac7f33329a |
| S1b | A1-30k DAC = 0 AND the reference also fails | 427 | 41a3a603b81f5ce8 |
| S2 | A1-30k NC = 0 (overlaps S1 / S1b in 249 scenes) | 889 | 1f9af3d3a94ff1be |
| Cpass | 200 random (numpy seed 20261004) of the 3,116 scenes whose four A1 multipliers (NC, DAC, DDC, TLC) are ALL exactly 1 | 200 | 33aec3e86adbb5af |
| P1_all | union of the four | 2,403 | 54639ba15da179e6 |
| K1fail | 100 random (seed 20261005) of S1, used for the reproduction control | 100 | 2898d1cf987c88b2 |

Primary set = S1 for the DAC question, S2 for the NC question. S1b is a floor read (see s6). Strata reported: stage, geometry class (RESULT s2 literals), command,
v0 band, max-speed known / unknown.

## 3. P1 -- definitions

Candidate universe per scene: U181 = 117 fan + 64 WTA. Also reported separately: FAN117 (the D5 formulation) and WTA64. The banked pick = `sel_idx` (E9).

**Scoring of a candidate** (CPU, NavSim-2 devkit `C:/Users/Admin/navsim-crun/devkit`, the SAME `PDMSimulator` / `PDMScorer` objects as the official run; code
`code/d6_fan_score.py`, written and unit-tested BEFORE the GPU export; its path is exact for DAC / DDC and approximate for NC, see Tier B):

* **Tier A (map-only, all 181 + STOP):** DAC and DDC (and TLC from the same call) of each candidate, batched in one `score_proposals` call per scene (no environment needed).
* (Tier A and Tier B are evaluated in the SAME `score_proposals` call: DAC / DDC / TLC do not depend on the environment, which K5 checks as batch-vs-single.)
* **Tier B (NC, approximate):** NC / TTC of each candidate against the SINGLE reactive environment simulated for the banked pick (`traffic_agents_policy.simulate_environment(pick)`),
  one batched call. This is an approximation (the IDM agents would react to each candidate differently); it is labelled `NC_approx`.
* **Tier C (exact confirmation):** the candidate with the highest `r7_score` among those that are clean at Tier A+B is re-scored through the exact path (`pdm_score`, its own
  reactive environment, human-penalty filter as official); its eight sub-scores are banked. `confirmed_clean` iff its four multipliers are all 1.

**Clean candidate** (pre-registered, two levels):
* `CLEAN_DAC` = DAC = 1 AND DDC = 1 (Tier A; exact for these terms).
* `CLEAN_FULL` = `CLEAN_DAC` AND NC = 1 AND TLC = 1, with NC from Tier B; reported as `CLEAN_FULL_approx`, and as `CLEAN_FULL_confirmed` for the Tier-C candidate.
* Not required: EP / TTC / LK / HC / EC (all eight sub-scores equal to 1 is neither feasible nor the question; a clean candidate that stops is clean). A secondary read
  `GOOD` = `CLEAN_FULL_approx` AND ego_progress >= the banked pick's ego_progress is reported for S2 only (it separates "slower" from "evasive" clean candidates).
* DDC = 0.5 and NC = 0.5 are NOT clean (only exact 1).

**The pick-takes-it read.** Because every member of S1 / S2 is a failure of the pick by construction, "the pick took a clean candidate" is read on the POPULATION, with the
200 random all-pass scenes as the unbiased remainder: Horvitz-Thompson weights 1 for every S1 / S1b / S2 scene (the union is a census of the failures) and 3,116 / 200 = 15.58 for each
Cpass scene. The population of the read is therefore {scenes with A1 DAC = 0, or NC = 0, or all four multipliers = 1}; scenes outside it (DDC-only or TLC-only zeros, partial 0.5 values) are excluded and said so. Per level L in {CLEAN_DAC, CLEAN_FULL_approx}:

* `F_L(S)` = share of scenes of S whose candidate universe contains >= 1 L-clean candidate ("fan-contains-clean");
* `PTI_L` = weighted P(pick is L-clean | universe contains an L-clean candidate) over S1 u S1b u S2 u Cpass ("pick-takes-it");
* `RND_L(S)` = mean over scenes of S of (n_clean / 181): the expected cleanliness of a UNIFORMLY RANDOM pick (the random control; an oracle gap is quoted only beside it);
* `RANK_L(S)` = for scenes with >= 1 clean candidate, the rank (1 = best) of the best-ranked clean candidate under `r7_score` within U181 and under `sel_score_v3 | reach_keep` within FAN117;
  reported as median and as the shares in the top 1 / 3 / 5 / 10;
* `R4_L(S)` = share of scenes of S where the refcv7 scorer's own pick `traj_r7` (NOT the deployed E9 pick) is L-clean;
* for S2 only: `SLOWER(S2)` = among scenes with a CLEAN_FULL_approx candidate, the share where the best-ranked clean candidate has a smaller 4-s travelled distance than the pick.

Intervals: log-cluster bootstrap over the 76 navhard logs, B = 2000, seed 0, percentile 95 % (question answered: another draw of LOGS; blind to training and inference variance;
the inference-variance evidence quoted beside it is the banked seed-1 overlap, Jaccard 0.86 stage 2).

## 4. P1 -- controls (all must hold or the affected read is reported INCONCLUSIVE, not dropped)

* **K1 reproduction.** On Cpass u K1fail (300 scenes), `cands_poses[sel_idx]` equals the banked R7_A1 seam plan to max abs <= 1e-4 m (pose units) on >= 99 % of scenes, and the Tier-C exact
  re-score of that member reproduces the banked eight sub-scores exactly. (The export must not change the plan; CUDA same-seed determinism is the banked K0 control.) Below 99 % -> every
  P1 number is reported with the reproduction rate beside it and marked INCONCLUSIVE.
* **K2 known-clean token.** The lexicographically first token of Cpass: its universe MUST contain >= 1 CLEAN_FULL candidate (its pick is clean). In aggregate F_CLEAN_FULL(Cpass) = 100 %,
  F_CLEAN_DAC(Cpass) = 100 % (the pick is a member). Any shortfall -> the clean detector is broken.
* **K3 STOP through the same path.** The STOP plan (all-zero poses) is appended as candidate 182 to the Tier-A batch on every scene. Its DAC, DDC, TLC must equal the banked STOP floor
  frame values (`raw/floors/navhard_two_stage`) on >= 99.9 % of STAGE-2 scenes (no human filter there); stage-1 scenes are reported separately because the official path applies the human-penalty filter to them and the map-only tier does not; its NC via Tier C (exact, own environment) must equal the banked STOP NC. This is the "scored through the same path" control.
* **K4 negative control (mutation).** Every candidate of Cpass shifted laterally by +30 m: F_CLEAN_DAC must be <= 5 % -- the clean detector is not vacuous. (A dry run on 6 synthetic scenes showed +8 m is NOT decisive -- 5 of 6 scenes still had drivable footprint at +8 m, wide intersections / carparks -- while +30 m and +60 m left 0 of 181 candidates clean on the 3 scenes tried; the magnitude was fixed on those synthetic scenes before registration, never on P1 data.)
* **K5 batch-vs-single.** 100 random (scene, candidate) pairs scored singly equal the batched Tier-A DAC / DDC exactly.
* **K6 parity of the replay.** The 117 fan + 64 WTA export contains `sel_idx` within the first 117 on 100 % of scenes, and `argmax(sel_score_v3 | reach_keep) == sel_idx` (the bridge's own `derived_ok`) on >= 99.5 %.

## 5. P2 -- the nav-withhold probe

Token set P2_all = every navhard scorer token whose NavSim driving command is LEFT or RIGHT (2,312; both stages). Strata: S_prem = command L/R and the route centreline is straight inside the
4-s look-ahead (the "premature turn" association; 724 scenes); S_turn = command L/R and the route turns inside the horizon (1,588). Route features exactly as RESULT s2 (heading change >= 20 degrees over
D = clip(4 v0, 10, 40) m). Files `spec_tokens_P2_*.txt`: P2_all c63000d51b9ef885, P2_S_prem 81c2c83c652a33c8, P2_S_turn 77ce4263b3baa24f, P2_K1 (300 random, seed 20261005) d6b08f504f874286.

Arms (same checkpoint, same device and precision, same frames / max-speed / ego inputs / inference seed 0; ONLY the nav input differs):

| arm | nav input fed | purpose |
|---|---|---|
| R7_A1 (banked) | the NavSim command as banked | baseline = the banked 30k rows; re-run on P2_K1 only, as control K7 |
| R7_NAVOFF | `nav_cmd = None` (the model's nav-ZERO intervention; exists in the base bridge) | does the model need the command at all |
| R7_NAVFOLLOW | the token forced to "follow" (index 0) on every scene | removes the premature turn if the command causes it |
| R7_NAVFLIP | LEFT <-> RIGHT swapped (STRAIGHT unchanged) | positive control: if the nav input is USED, the plan must turn the other way on S_turn |

NAVFOLLOW and NAVFLIP are new arms of the COPY bridge only (`bridge_fix/refcv7_bridge.py`, entries in `ARMS7` with `nav_override`; absent from the base).
Metrics per arm, stratum and stage (exact devkit re-score through `code/d6_rescore.py`'s path; the reactive environment as official): DAC0 rate, NC0 rate, command-compliance rate
(RESULT s4 definition), geometry-class shares (RESULT s2 classifier). Paired against the banked A1 on the same tokens. Floor: the banked A1 vs A1_s1 paired difference on the same tokens (the inference-seed floor).
Controls: **K7** A1 re-run on P2_K1 reproduces the banked plans (>= 99 % exact) and sub-scores; **K8** NAVFLIP changes the plan's 4-s heading sign on >= 50 % of S_turn scenes
(else the nav input is not used on this surface -- itself a P2 finding, reading O4); **K9** the identity: scoring an arm with its own banked-equivalent plans reproduces the banked sub-scores (as K1).

## 6. Reading rules (pre-committed; thresholds are literals)

Resolved effect (P2): |paired delta in a rate| >= 3.0 percentage points AND the log-cluster 95 % CI of the paired delta excludes 0 AND |delta| > 2 x |banked A1 - A1_s1 paired delta on the same tokens|.

**P1, on S1 (DAC, clean-path scenes), universe U181, level CLEAN_DAC; read together with F_CLEAN_FULL_approx:**

| outcome | points to |
|---|---|
| F >= 0.70 AND PTI < 0.5 x F | SELECTION is the dominant defect: the model generates a clean path and does not take it. Levers: (a) if R4 >= 0.5 x F the free lever is `refcv7_select` at inference (the model's own scorer already prefers clean candidates -- no retraining); (b) else retrain the selector / scorer with DAC- and NC-aware targets (SEL-1 family); (c) inference-time drivable gate from the BEV map head. |
| F <= 0.30 | GENERATION is the dominant defect: the fan / WTA do not cover a clean path. Levers: anchor-set coverage conditioned on the drivable area, WTA training, L2 time-localised nav (generation conditioning), map head quality (MAP rows). Selection levers cannot help. |
| 0.30 < F < 0.70 | MIXED: report F as the selection ceiling and 1 - F as the generation floor; both lever families are funded in proportion. |
| RANK median > 30 of 181 (best clean candidate buried) | the selector is blind to the clean path (score uncorrelated with cleanliness): (b) or (c) above, not a re-weighting. |
| RANK top-5 share >= 0.5 | a mild re-ranking (a DAC-aware term in the existing scorer) is sufficient. |

Class-level: the same table per geometry class with n >= 50; n < 50 -> UNDERPOWERED, no reading. S1b is a floor: a high F on S1b would mean the PDM-Closed reference is beatable by the fan on scenes it fails (reported, no lever).

**P1, on S2 (NC):** F_CLEAN_FULL_approx(S2) in the same three bands, plus: SLOWER(S2) >= 0.80 -> the clean candidates are mostly SLOWER than the pick: a speed commit
(selector speed bias / L3 speed input / L1 longitudinal tactical labels); SLOWER < 0.50 -> clean candidates are as fast or faster (evasive / earlier-lateral): perception of the lead (box) and
lateral generation; in between -> both. R4 on S2 read as above.

**P1 population read:** `PTI_CLEAN_FULL_approx` against `RND_CLEAN_FULL_approx` and the oracle `F`: report all three together (the oracle-gap control). A pick that is not better than uniform random among the clean-containing scenes
(PTI <= RND + 0.05) means the selector contributes nothing at the clean/not-clean boundary.

**P2:**

| outcome | points to |
|---|---|
| NAVFOLLOW or NAVOFF lowers DAC0 on S_prem by a RESOLVED effect | the premature turn is caused by the command firing before the geometry: lever L2 (time-localised nav / announce only junction turns), ceiling = the measured drop x the S_prem share of DAC-zero. |
| no resolved change on S_prem, and NAVFLIP resolves a change in compliance / WRONG-SIDE on S_turn (K8 holds) | the nav input is used but the premature turn is not nav-caused: vision / tactical (L1 dense labels incl. "follow" windows), not L2. |
| NAVOFF raises DAC0 or the ROUTE-FOLLOWING share on S_turn by a RESOLVED effect | the command helps the turn; the residual under-turn is a turn-EXECUTION deficit: L1 (dense tactical turn labels) and nav-conditioning strength. |
| K8 fails (NAVFLIP does not flip the turn on >= 50 % of S_turn) | the nav input is not used on this surface (consistent with the banked NAV-3): the nav pathway itself is the defect; premature turn and under-turn CANNOT be attributed to nav; lever L2 = make the pathway matter (time-localised token, stronger conditioning). |
| none of the above resolved | no nav lever from this probe; record as a NEGATIVE for L2 on NavSim and keep L1 / L3. |

Every conclusion cites its stratum n and estimator. A P2 effect is a difference between model forwards of ONE trained arm under input interventions (inference variance floor named above); it is not a
training-seed claim (H-ESTIM-SEED-1 stays open) and is not promoted to a lever effect for retraining without a replicate arm.

## 7. Cost and run plan (MEASURED unless marked)

* GPU (blocked behind the battery chain, lock job `refcv7-milestone-step50400`, acquired 13:20): model load 4.3 s (banked manifest) + 0.39-0.45 s/scene (banked R7_A1 navhard stream): P1 2,403 scenes ~ 17 min + K-controls;
  P2 3 arms x 2,312 + 300 = 7,236 forwards ~ 50 min. Total ~ 1.1 h of lock time. Queue order: P1 first, then P2. Both via `battery/code/with_gpu_lock.py` (waits; never touches a held lock).
* CPU scoring afterwards (3 processes, 0.5 GB each): P1 ~ 2,403 scenes x (Tier A+B ~1.5 s + Tier C ~2 s) ~ 2.4 h/3 = ~50 min; P2 ~ 6,936 exact scorings x ~2 s ~ 3.9 h/3 = ~1.3 h. (ESTIMATED from the measured fan-bench 1.0-1.3 s/token and 1.6-7 s/exact scoring under contention.)
* Disk: fan sidecar ~ 16 KB/scene (float32 base64 JSON line) ~ 40 MB.

## 8. Instrument hygiene (checked before the GPU job is queued)

* The base package files are read-only for this stream; the copy and the base blob sha of every copied file are listed in `code/bridge_fix/BASE_BLOBS.txt`.
* **T-DEFAULT:** `code/bridge_fix/test_default_identical.py` (22 checks, 1 mutation control, already green at the time of writing) -- (i) the diff base vs copy is INSERT-only (no base line deleted or changed; `boot7.py` and `gpu_lock.py` are byte-identical copies; the `--gpu-lock-job` check is an inserted monkeypatch of `gpu_lock.held_by`, inert unless the flag is given); (ii) the `.base` files equal the live package files byte for byte; (iii) AST: every use of the export is under `if EXPORT_FAN[...]`, every new flag use is gated by an `if` on that flag; (iv) runtime import of the copy: `ARMS7` = base + the two new arms, every base arm unchanged, `EXPORT_FAN` off, `knots_to_navsim` reproduces 20 banked seam poses exactly, `export_fan_from_out` on a synthetic `out` built around a real banked row returns the banked poses at `sel_idx` and refuses an `out` without `r7_candidates`. A CPU one-scene forward with the flag on vs off compares `traj` bit-for-bit when memory allows
  (the GPU job's first act, K1, is the full-scale version).
* **Lock protocol:** the queued job runs `with_gpu_lock.py --job refcv8-d6-p1p2 -- <launcher>`; the child verifies the lock is the wrapper's (job name + wrapper pid) via the new `--gpu-lock-job` check (the battery lock record carries no token).
* Launch: PowerShell `Start-Process -WindowStyle Hidden` with its own launcher script (`code/launch_p1p2.ps1`; GPU queue `code/run_p1p2_chain.py` under `with_gpu_lock.py`, CPU stage `code/run_p1p2_cpu.py`), only AFTER the Master Mind confirms this SPEC is registered. Nothing is queued before that. The chain refuses to run unless the lock file names `refcv8-d6-p1p2` AND records the pid of an ANCESTOR of the chain (dry-run tested with a temp lock: accepted; foreign job refused; right job with a non-ancestor pid refused). The analysis scripts `code/d6_p1_analyze.py` and `code/d6_p2_analyze.py` implement s3-s6 and were smoke-tested on SYNTHETIC inputs only (outputs tagged SYNTHETIC-TEST).

## 9. What this SPEC does not claim

No P1 / P2 number exists at the time of writing. P3 (the zero-GPU perfect-fix oracles) is NOT part of this registration; its numbers are reported separately in RESULT (they size the
ceilings, they do not decide the reading rules above, and the rules were fixed before they were read). The probes' conclusions are readings of ONE checkpoint (step 30,000); step 50,400 is not covered.
