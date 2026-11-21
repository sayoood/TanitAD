> **DRAFT -- NOT REGISTERED, NOT RUN.** Written 2026-09-28 ~02:00 Berlin for the coordinator's review. Once
> registered, it is appended to `eval/SPEC_NAVTEST.md` verbatim below Amendment 7 and its git blob is recorded in
> `eval/raw/SPEC_PREREG_HASH.txt` BEFORE any fresh-token model output exists.

> **AMENDMENT 8 -- (registration time from `date`), BEFORE any model output on the confirmation tokens exists.**
> **Seen so far, EXPLORATORY, on the 1,123 proxy tokens (W3's 200 + Amendment 5's 923; the defect was FOUND on them,
> so none of it is admissible for adoption):**
> - The planner's goal input (`refe/planner.py:449-474` `_goal_for` -> `code/augment_routes.py:46-70`
>   `_route_with_lane_rank` -> DriveRL `goal_position_utils.py:512` `route_goal_positions`) has NO guard for a route
>   that does not reach the ego.
>   - On two logs the scenario's own `route_roadblock_ids` do not contain the ego's roadblock; the nearest route
>     roadblock is 373.5 m / 421.1 m ahead (`raw/2026-09-28-goal-clamp/goal_trace.json`).
>   - `_route_start_index` (`driverl_runtime_map_features.py:881`) takes the nearest roadblock at any distance.
>   - The projection (`goal_position_utils.py:555-566`) clamps to the route's first point at any distance, so the goal
>     becomes "route start + 30 / + 60 m", 349-480 m ahead. The re-derivation reproduces the cached goals exactly.
> - Census of the ego's distance to its own route polyline over the 1,123 (`route_cover_census.json`, CPU, no model):
>   median 0.29 m, p90 1.54 m; > 5 m: 60 tokens, > 10 m: 58, > 20 m: 53 (9 logs), > 50 m: 45, > 100 m: 27, > 300 m: 19.
> - PDMS of the live recipe (snapshot 015 + Amendment 7's repair) falls with that distance:
>
>   | ego-to-route distance | tokens | PDMS |
>   |---|---|---|
>   | < 2 m | 1,035 | 79.6 |
>   | 2-10 m | 30 | 65.6 |
>   | 10-20 m | 5 | 79.5 |
>   | 20-50 m | 8 | 56.4 |
>   | 50-100 m | 18 | 58.9 |
>   | 100-300 m | 8 | 41.2 |
>   | >= 300 m | 19 | 2.2 |
>
> - On the 20 tokens with goal p2 > 300 m, the WHOLE fan drives 57.6 m in 4 s against a GT of 12.6 m.
>   - The executed plan scores 2.08 PDMS with the repair ON, and 0-14 under every M6 / M6b arm.
>   - These tokens cost the 1,123-token mean 1.2-1.5 PDMS in every condition (`garbage_tokens_pdms.json`).
> - A CPU re-decode of those 20 tokens with the goal replaced (`goal_clamp_probe.json`; the untouched-goal control
>   reproduces the GPU picks 20/20) scored:
>
>   | goal on the 20 tokens | PDMS |
>   |---|---|
>   | untouched (control) | 2.08 |
>   | clamped to 219 m | 27.85 |
>   | clamped to 150 m | 35.35 |
>   | straight-route goal from the ego, max(v0, 5 m/s) x 12 s | 88.38 |
>
> - The training side carries the same defect at negligible frequency: the live bank has 9 of 33,704 r0 rows with goal
>   p2 > 300 m.
>
> **Sanitisation under test (ONE, fixed now; its threshold was read off the 1,123-token census above, before any
> confirmation token was read):**
> - **Trigger:** the ego's distance to the route polyline `_goal_for` builds (the 100-point polyline, segment
>   distance) exceeds **D = 20 m**.
>   - Why 20 m: about five lane widths, so unambiguously off the route.
>   - Tokens at 10-20 m score like on-route tokens (79.5, n = 5).
>   - The 2-10 m band (65.6, n = 30) is a DIFFERENT question (lane choice on a covering route) and is left alone.
> - **Below D:** nothing changes, bit for bit.
> - **Above D:** the goal is re-derived by the SAME `route_goal_positions` (horizon 12 s, min speed 5 m/s, 2 points),
>   on a **fallback route built from the ego's OWN lane** instead of the scenario route:
>   1. Candidates: the lanes and lane connectors within 10 m of the ego (`map_api.get_proximal_map_objects`).
>   2. The start is the candidate with the best `_route_edge_anchor_score` (distance + 5 x heading error).
>   3. The route extends through `outgoing_edges` until it is >= 150 m long. At each fork it takes the successor whose
>      exit heading change is the most counter-clockwise for command LEFT, the most clockwise for RIGHT, and the
>      smallest in magnitude for STRAIGHT and UNKNOWN.
>   4. It is resampled with `_fit_route_polyline`.
> - If no lane lies within 10 m (off-map), the fallback is the straight-route goal along the ego heading:
>   p2 = max(v0, 5) x 12 s, p1 = p2 / 2.
> - **Inputs, all admissible at inference:** the map, the ego pose, v0 at t0, and the benchmark's driving command,
>   which NAVSIM gives every agent. Never GT, never future ego.
> - **Why it is not optimistic by construction on turns:**
>   - The fallback follows the MAP's lane geometry and the COMMAND. It does not follow the logged future.
>   - On a turn it produces the turn only if the lane graph and the command say so.
>   - A wrong command or an ambiguous fork gives a WRONG goal, and that cost is in the statistic.
>   - The exploratory straight-route result (88.38) is the special case where all 20 tokens were STRAIGHT on straight
>     roads. It is NOT the number this amendment tests.
>
> **Confirmation tokens (FRESH):** every navtest token OUTSIDE the 1,123 on which the trigger fires, by the CPU census
> over the full 12,146 (`route_cover_census_navtest_full.json`, computed with no model output): **422 tokens in
> 32 logs** (commands L/S/R/U: 112 / 266 / 44 / 0).
> - The full census triggers on 475 of the 12,146 tokens; 53 of those are the selection tokens, which are excluded. The list is banked in `amendment8_fresh_set.json`.
> - The same census over the 1,123 reproduces the first census exactly (0 of 1,123 differ).
> - Distance bands of the fresh set: 122 at 20-50 m, 103 at 50-100 m, 171 at 100-300 m, 26 at > 300 m.
> - The fresh set contains 156 LEFT or RIGHT tokens, so the lane-following fallback is tested on turns, not only on the straight roads it was found on.
> - **Why a second, stricter read.** The defect is a property of a log's route, so 6 of the 32 logs also hold selection tokens; only 279 fresh tokens sit in the 26 logs that hold NO selection token.
> - **Estimator:** the paired log-cluster bootstrap (10,000 resamples, percentile 95 %, seed 20260927). It is computed TWICE, both committed now:
>   - **PRIMARY:** over all 422 tokens, clustered by their 32 logs;
>   - **FRESH-LOG:** over the 279 tokens of the 26 selection-free logs.
> - 32 and 26 clusters are enough for this estimator; the M6 / M6b analyses used 136. No smaller design is needed. If a harness run drops tokens, the design stays as written and the valid-token count is reported; fewer than 20 logs in either read is NOT PROVEN.
>
> **Snapshot:** after epoch 15 (md5 `d7c59f4f2fbcbde3e2dec8f67d63a7e7`), Amendment 7's repair ON, rule v1, in both arms.
>
> **Measurement:** the unchanged pipeline (`eval/refe_navtest_seam.py`, planner forward on GPU when the PI allows it)
> writes two seams on the confirmation tokens: goal as today (OFF) and goal sanitised (ON). Both are scored by the
> unchanged harness.
>
> **Validity gates, all must pass before the statistic is read:**
> - (a) on every confirmation token the trigger fires (census re-checked in the run);
> - (b) a unit test with a mutation arm:
>   - a covering route leaves the goal bit-identical;
>   - the recorded 373.5 m case is replaced;
>   - LEFT and RIGHT forks pick opposite successors on a synthetic Y junction;
>   - deleting the trigger check must go RED;
> - (c) the OFF seam reproduces the pipeline's own pick and score on every token;
> - (d) every harness run PASSes with every token valid;
> - (e) nothing but the goal differs between the arms' inputs.
>
> **Statistic:** per token PDMS(ON) - PDMS(OFF), mean over the confirmation tokens, with the estimator and CI named in
> the design paragraph. It answers "another draw of episodes" only.
>
> **Decision, both outcomes committed now:**
> - **ADOPT** iff BOTH reads' CI lower bounds are > 0 (PRIMARY and FRESH-LOG) AND no longitudinal or lateral family component separates adversely (a
>   named interval entirely on the worse side).
>   - The sanitisation then becomes part of REFe's planner goal path for every evaluation from then on: a declared
>     test-time change in `refe/planner.py`, recorded in MODEL_REGISTRY §14.1.
>   - Earlier points keep their values; sanitised values are reported beside them where a seam allows.
> - **REFUTED** iff the PRIMARY upper bound is < 0.
> - Otherwise **NOT PROVEN**, and nothing is adopted.
> - The four metric families (families6; strategic UNAVAILABLE in NAVSIM by design) are reported for both arms. Unlike
>   Amendment 7 they CAN differ here, because the goal changes positions.
>
> **Reported, not gating:**
> - the 150 m clamp;
> - the straight-route-only fallback;
> - the sanitised seam on the 53 triggered tokens of the 1,123 (exploratory, selection tokens);
> - the NAVSIM sub-score deltas.
>
> **The final full navtest (Thursday):**
> - The score of record uses the pipeline as it stands at the moment the run starts.
>   - If this amendment has read ADOPT by then, the final score uses the sanitised goal and ALSO reports the
>     unsanitised score beside it on the same run.
>   - If it has not been read, or reads NOT PROVEN / REFUTED, the final score uses today's goal path, and the sanitised
>     score is reported beside it as a non-gating readout.
> - The triggered tokens are named in the final report in either case.
> - ⛔ D, the fallback rule and the 10 m lane radius are NOT tuned after any confirmation token is read.
>
> **Training side (next model version, NOT a live change):** the same guard belongs in the bank builder
> (`build_targets.py`'s goal path).
> - A row whose ego lies > D from its route gets the SAME fallback goal. Today 9 of 33,704 r0 rows have p2 > 300 m.
> - It is a declared recipe change: a `--declare-change` identity key, a G-DVB entry in `launch_gate.py`, and the
>   launch-gate PASS token.
> - It keeps train and test goal definitions identical. Without it the model would still see ~0.03 % of rows with a
>   far goal at train time and none at test time: harmless at that frequency, but not symmetric.
> - The live run is not touched.
