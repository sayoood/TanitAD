# S0 of the auxiliary drivable-area head: label store, gate, build (2026-10-12)

PI decision D-AUX-2 (2026-10-12): build the label store now. Design settled in `SPEC_AUX_DRIVABLE_HEAD.md` section 3; not redesigned here.
All numbers are **MEASURED** on the dev box, CPU only, from the artifacts named. Tier: label construction, no model, no driving claim.

## Files

| file | what |
|---|---|
| `refe/aux_label_store.py` | the builder: path B, resumable by key, `--workers N`, `--shard i/n`, `logmap` / `build` / `finalize` |
| `refe/test_aux_label_store.py` | `ZZTEST_PASS 16`: synthetic-map oracle + RED mutations + store behaviour |
| `raw/2026-10-12-aux-label-store/s0_gate.py` | the gate (`--mode pre` before the build, `--mode post` on the finished store) |
| `raw/2026-10-12-aux-label-store/s0_gate_pre.json` | the pre-build gate result (80 samples, 6 regression arms) |
| `raw/2026-10-12-aux-label-store/s0_gate_pre.log` | the gate's per-sample console log |
| `raw/2026-10-12-aux-label-store/log_map.json` | the log -> map table (1,192 logs; provenance per log) |
| `raw/2026-10-12-aux-label-store/HANDOFF.md` | state of the background build and what to verify when it ends |

## Inputs (all on the dev box; nothing fetched from a pod)

| input | where | size |
|---|---|---|
| train bank, `origin_world` on every row | `D:/Projects/TanitAD-artifacts/_rlfan_work/bank_full/targets_rank0.jsonl` (extracted earlier from the pod-final archive by another stream) | 103,037 rows = 103,037 distinct (log, token), 1,191 logs, all rank 0 / step 0 |
| nuPlan DBs for the train logs | `D:/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits` (flat dir + `trainval/`) | 463 of the 1,192 navtrain logs |
| maps | `D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0` | all 4 |
| token universe | `splits/navtrain.yaml` | 103,288 tokens, 1,192 logs |
| navtest list | `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-estimator-and-route-leak/raw/cluster_maps/navtest.json` (tracked) | 12,146 tokens, 136 logs |

**Coverage: 103,139 of 103,288 train tokens (99.856 %)** = 103,037 bank rows + 102 yaml tokens that only a local DB holds (their origin is read from the
DB `ego_pose`). The 149 tokens left have neither a bank row nor a local DB; `missing_tokens.txt` in the store lists them.
SPEC section 7 asks for coverage of the *training scenes* >= 99.9 %: against the **103,037-row train bank (what the trainer iterates) it is 100.000 %**; against the 103,288-token yaml it is
99.856 %, 45 tokens short of 99.9 %, and `PREREG_AUX_DRIVABLE_DRAFT.md` line 144 writes "expect 103,288", which this store cannot meet without the missing source below. (The yaml has 251 tokens with
no log in the bank, as `PREREG_RL_S1.md` says; the local DBs recover 102 of them.)

The log -> map table is the one input that did not exist for 729 of the 1,192 logs (no local DB). It is derived, not guessed:
`infer_map_name` scores each map by the share of six sampled ego centres that lie inside a drivable polygon of that map. MEASURED: **462 / 462 agreement with
the DB's own `map_version` on every log that has a DB; on the 729 inferred logs the best map scores 1.0 and the runner-up 0.0 on every one, 0 ambiguous**
(`log_map.json` summary). nuPlan's four maps occupy disjoint coordinate regions, which is why a wrong map scores 0.

Origin source cross-check (the build repeats it): DB `ego_pose` vs bank `origin_world` over 33,704 rows = **0 m, 8.9e-16 rad** (heading uses pyquaternion's
formula, `(q0 q3 - q1 q2)`, not the textbook sign: `test_yaw_matches_pyquaternion_not_the_textbook`).

## Pre-build gate (`s0_gate_pre.json`, seed 20261012)

80 samples: 64 bank rows (stratified over the 4 maps, one per log; boston 16 / pittsburgh 18 / las-vegas 26 / singapore 20) + 16 DB-origin frames with no bank row.
Path A = `NuPlanScenario` (64) or, for the 16 frames the scenario builder refuses, the devkit's own `get_ego_state_for_lidarpc_token_from_db` + the DB's map.
Bars were fixed before the first run (SPEC's where it has one).

| criterion | bar | result | pass |
|---|---|---|---|
| AB: path A vs B, mask and SDF | 0 differing cells | **0** of 80 x 20,480 x 2 | yes |
| ORIGIN: rear-axle pose A vs B | <= 1e-6 m, <= 1e-9 rad | **0.0 m, 4.4e-16 rad** | yes |
| CORNER: label vs `navsim_dac` exact, 13 shifts x 64 targets, 136,448 corners | >= 98.0 % | **98.605 %** bilinear [98.31, 98.90]; 98.474 % nearest | yes |
| VERDICT: whole proposal, 832 proposals | >= 95 % | **97.596 %** bilinear [95.79, 99.04]; 97.356 % nearest | yes |
| TARGET: unshifted rank-0 targets inside the label | >= 90 % | **98.44 %** (63 of 64; NAVSIM's exact rule: 64 of 64) | yes |
| LEAK: universe vs navtest | 0 tokens, 0 logs | **0 / 0** over 103,139 tokens and 1,192 logs; yaml vs navtest token overlap 0 | yes |

Intervals are 4,000-draw percentile bootstraps over the 64 samples (the cluster), seed 0.

⚠️ **The SPEC's 20-sample corner figure (99.12 % bilinear) is not reproduced at n = 64**: 98.605 % with an interval whose upper end (98.90 %) is below it. The 20 were
one frame per log from the low-speed `r0_shard0` shard (v0 max 13.1 m/s); whether that sample difference explains the gap is UNVERIFIED. The bar (98.0 %) is cleared with margin; the SPEC's headline number should be
quoted as the 64-sample one. The label over-flags violations: 39.2 % of shifted proposals vs 37.0 % by NAVSIM's exact rule (+2.2 points, 18 of 832).

**Deliberate-regression arms, judged by the same verdict function (all six FAIL the gate):**

| arm | tripped | corner / verdict / target |
|---|---|---|
| y axis flipped | AB, CORNER, VERDICT, TARGET | 72.0 % / 42.2 % / 50.0 % |
| x/y swapped | AB, CORNER, VERDICT, TARGET | 50.7 % / 42.2 % / 10.9 % |
| wrong map for the log (cyclic) | AB, CORNER, VERDICT, TARGET | 13.6 % / 37.0 % / 0.0 % |
| label of the frame 0.5 s later (10 DB rows) | AB, CORNER, VERDICT, TARGET | 96.4 % / 92.9 % / 87.5 % |
| padding removed | **AB only** | 98.6 % / 97.6 % / 98.4 % (unchanged) |
| clip 12.7 m -> 6 m | **AB only** | 98.6 % / 97.6 % / 98.4 % (unchanged) |

⚠️ Padding and clip change the SDF magnitude far from the boundary, not the sign at footprint corners, so **the fidelity criteria cannot see them**; only the cell-for-cell
AB check (and `test_aux_label_store.py`, which pins them analytically) does. Do not read "fidelity passed" as "the SDF values are right".

## Findings worth a second look

1. **Degenerate-grid guard is an equivalent mutant (documented, not hidden).** The prototype's `signed_distance` had no guard for a window with no drivable cell.
   `scipy.ndimage.distance_transform_edt` of a no-background array returns the distance to the array border, and with `PAD_M = CLIP + 0.5` that is always >= the clip, so
   the prototype already read +-127 there. The guard I added makes that independent of the pad; `test_degenerate_guard_is_equivalent_at_current_pad` pins both facts.
2. **int8 wrap guard.** `quantize_sdf` refuses a clip above 12.7 m (254 LSB would wrap int8 silently); RED-tested.
3. **Label speed here is 21 ms median** (mean 23 ms) per warm frame, against the SPEC's 38.5 / 67.8 ms: the box was quieter. The SPEC's cost range was an upper bound.
4. `navtrain.yaml` tokens must be quoted strings: an all-digit token is parsed as an int by YAML and loses its leading zeros. The loader now raises on a non-string token.

## Not done / not claimed

* The 729 inferred-map logs have no independent path A (no DB). Their evidence is the 462 / 462 inference agreement, the perfect score margin, and the per-key QC (share of
  rank-0 target points inside the label) that the build stores; `--mode post` reports that QC split by map source.
* Navtest evaluation labels (path A on 12,146 frames) are not built here (not requested).
* The head, the trainer wiring and the G-LIVE items of SPEC section 7 other than alignment and no-leakage are not part of S0.
