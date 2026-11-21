# W-TRANSP is not harness-identical on navtest: root cause, proposed fix, test (2026-11-12)

Scope: instrument diagnosis for the void TF-N1 read (`../run_void/RESULT_N1_VOID.md`). It applies to nothing frozen and was not applied to
`C:/Users/Admin/n1_run`. Data used: N1-T rows only (`arms.N1T`, which is the control's own scope), the banked harness CSV
`refe_p4_teacher_navtest.csv`, the harness's own metric cache (`F:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest`) and the
nuPlan DB. **No N1-C, N1-W, N1-M, CERT or human value was opened, and no arm of any seam was scored in this diagnosis** (the PDM-Closed
reference trajectory is a per-token quantity that belongs to no arm). Scripts and raw outputs: `diag/`.

## 1. Answer in one paragraph

W-TRANSP and the harness feed the same scorer (NAVSIM v1.1 `PDMSimulator` + `PDMScorer`, same code) **different scenery**. The harness reads a
stored `MetricCache` built from NAVSIM's `NavSimScenario`, whose tracked objects come from the OpenScene annotations. W-TRANSP rebuilds the
"cache" in-process from the **nuPlan DB** (`NuPlanScenario`). The two object sets agree on identity and box size, but their positions differ by
about 1 to 15 cm. Two consequences, and both are the symptoms seen:
1. **EP.** PDM-Closed is an IDM follower, so its trajectory, and therefore its progress `raw0` (the EP normaliser), shifts with the lead's position.
   `EP = raw_plan / max(raw0, raw_plan)`. EP is therefore different by exactly the ratio of the two `raw0` values on every token where the plan is
   slower than PDM-Closed (EP < 1) and PDM-Closed reacts to an object.
2. **NC / TTC flips.** The plan's collision and time-to-collision tests run against the same objects, at centimetre separation. 7 tokens sit on the
   boundary.

It is **not** a float32 issue in the normaliser, not a progress-reference bug, not time sampling, and not interpolation.

## 2. Evidence (all MEASURED 2026-11-12)

**(a) Which tokens differ in EP** (N1-T rows against the banked CSV, 12,146 tokens; `diag/an1.py`):

| ego_progress of the banked row | tokens | fraction with |dEP| > 1e-6 |
|---|---|---|
| 0 to 0.01 | 657 | 0.15 % |
| 0.01 to 0.9 | 4,736 | 47.7 % |
| 0.9 to 0.9999 | 2,651 | 40.5 % |
| 0.9999 to 1.0001 (plan at least as fast as PDM-Closed) | 4,102 | 0.27 % |

The two ends are untouched. That is the signature of the normaliser: EP = 1 never uses it, EP = 0 is a multiplier zero. Both signs occur
(+1,864 / -1,479), median |dEP| 9.9e-5, 99th percentile 2.0e-2. For tokens with 0 < EP < 1 the ratio EP_w / EP_h has median 1.000009 and 1st/99th
percentile 0.9949 / 1.0053.

**(b) The ratio is the normaliser, to 2e-8.** For 10 randomly drawn EP-differing tokens (seed 1) I scored the PDM-Closed trajectory alone
(`diag/diag_norm.py`: `simulate_proposals` + `score_proposals` on `[pdm_traj]`, reading `_progress_raw[0]`), once from the harness cache and once
from W-TRANSP's rebuild:

    max over the 10 tokens | raw0_cache / raw0_Wtransp  -  EP_Wtransp / EP_harness | = 2.3e-8

so EP_w / EP_h equals raw0_h / raw0_w, i.e. the whole EP difference is the PDM-Closed progress. On 10 tokens that do not differ, raw0 agrees to
8e-7 or better (relative). `diag/diag_norm.json`.

**(c) Why raw0 differs: the PDM-Closed trajectory moves, and the objects are different** (27 tokens: the 7 flips, the 10 EP-differing, 10
non-differing; `diag/diag_cache_compare.json`):

| group | ego state |diff| | PDM-Closed trajectory |diff| (m) | object box bounds |diff| (m) | object token sets |
|---|---|---|---|---|
| 7 flips | <= 2.0e-7 | 1.5e-7 to 1.7e-2 | 0.034 to 0.135 | equal at all 51 steps |
| 10 EP-differing | <= 3.0e-7 | 5.3e-4 to **0.546** | 0.021 to 0.145 | equal |
| 10 non-differing | <= 3.6e-7 | 7e-10 to 7.0e-5 | 0.019 to 0.109 | equal |

The ego state, centerline length and drivable map are identical. **Only the objects differ**, on every token, by centimetres; where the IDM
follows one of them the PDM-Closed trajectory moves by up to half a metre, where it does not it stays within 1e-4 m.

**(d) The objects differ at t = 0, as a near-rigid shift.** Token `2b482f615de15efb`, iteration 0, same `track_token`, 12 objects
(`diag/diag_probe3.py`): 11 dynamic agents differ by (+0.009, -0.012) m with a spread of about 2 mm, not along their velocity (the implied time
offset is inconsistent in sign and size, so it is not a timing difference), and the one static object differs by (-0.0009, -0.0006) m. Box
length and width are identical to 0. The ego timestamps are identical (1631823407500117 us).

**(e) The code paths differ at the source.**
* Harness cache: `navsim/planning/scenario_builder/navsim_scenario.py:182-195` (`NavSimScenario.get_tracked_objects_at_iteration` reads
  `scene.frames[i].annotations` through `annotations_to_detection_tracks`, `navsim_scenario_utils.py:42-82`; ego from `ego_status.ego_pose`,
  `navsim_scenario.py:171-174`; `_database_interval = 0.5`, line 50). The harness stores the result with `MetricCache.dump` (lzma pickle).
* W-TRANSP: `eval/instruments/transp_score.py:354` calls `refe/navsim_pdm_targets.py:87-96` (`metric_cache_parts`), which runs the same
  `MetricCacheProcessor` methods on a `NuPlanScenario` built at `refe/navtrain_scenarios.py:260-273` from the nuPlan DB `lidar_box` table
  (`subsample_ratio = SIM_HZ/20 = 0.5`, `database_interval 0.1`, measured).
* The processor code itself (`_interpolate_gt_observation`, `compute_metric_cache`, `metric_cache_processor.py:91-235`) is identical in both
  runs; only the `scenario` argument differs.

**(f) The 7 flips.** `2b482f615de15efb`, `76615d386ce6519e`, `9ef1d618435d5e53`, `66ccc68c8f755683`, `b66db24cee7957a2`, `b966c10ea361593f`,
`c678ab21b707538f` (NC flips: `2b48...`, `b966...`, `b66d...`; TTC flips: `9ef1...`, `c678...`, `66cc...`, `7661...`; `c678ab21b707538f` was not in the MM's list). All seven have object bounds that differ by 3 to 14 cm
and equal token sets. Which box edge decides each flip was not isolated (it needs the per-step collision geometry, which is arm-level and so was
left alone).

**Not isolated (UNVERIFIED):** why the OpenScene-derived and DB-derived object positions differ by about 1 cm at t = 0. Candidates are the
OpenScene annotations being stored in the ego frame (and re-projected with the OpenScene ego pose) against the DB's global `lidar_box`
coordinates, or float32 storage of the ego-frame boxes. It does not matter for the fix, because the fix removes the DB route.

## 3. What is excluded

* Not the plan: DAC, DDC and C agree on all 12,146 tokens, and EP agrees to 1e-6 on every EP = 1 token. The plan simulation is identical.
* Not a normaliser formula difference: `transp_score.py:174-181` is `_aggregate_scores`'s rule (`m > 5.0`), and the ratio test in (b) closes to 2e-8
  with the harness's own `raw0`.
* Not float32 in W-TRANSP: its sub-scores are float64 and the differences are signed both ways with a heavy tail (0.5 m in the trajectory).
* Not ego state, centerline or map (c).

## 4. Proposed fix (hold copy `transp_score_a1fix.py`, diff `transp_score_a1fix.diff`; NOT applied to the run tree)

Add `--metric-cache-root`. When given, `parts` is the harness's own cache, loaded exactly as `navsim.evaluate.pdm_score` reads it
(`trajectory, ego_state, observation, centerline, route_lane_ids, drivable_area_map`), instead of `PT.metric_cache_parts(sc)`:

    parts = PT.metric_cache_parts(sc) if not a.metric_cache_root else load_cache_parts(a.metric_cache_root, lg, tok)

The scenario `sc` is still built for the human future and the lead track (DB-based descriptive quantities); the ego state it uses differs from
the cache's by at most 3.6e-7. Navtest only (navtrain has no cache). The transparent arm and the families then use the harness's scenery too.
**Status: the scoring function is tested (section 5); the `transp_score.py` command-line path is a three-line change that I did not run,
because running it means scoring a seam on navtest.** Run the section 5 test and one `--limit` smoke on a non-N1 seam before any use.

## 5. Test (`wtransp_cache_parts_check.py`; result `smoke/wtransp_check.json`)

Scores a banked **non-N1** seam (the human seam of `eval/raw/e0_human_sub200`, W3's exported human future, 200 navtest tokens) through
W-TRANSP's own `score_arms`, twice, against the banked harness CSV `refe_e0b_human_sub200.csv`:

| arm | parts from | max |d| EP | max |d| PDMS | tokens over 1e-6 |
|---|---|---|---|---|
| fix | harness metric cache | 1.0e-12 | 4.3e-13 | **0 / 200** |
| deliberate regression | nuPlan DB rebuild (today's W-TRANSP) | 1.03e-2 | 4.3e-3 | **54 / 200 (27 %)** |

NC, DAC, DDC, TTC, C: 0.0 in both. The regression arm reproduces the void run's rate (27.5 % of tokens with an EP difference) on a different
seam, which is an independent confirmation of the cause. The test returns rc 0 only if the fix arm passes AND the regression arm exceeds 1e-6
somewhere (so a test that cannot tell the two routes apart fails). This is a navtest read of the human seam, which is a reference arm and not an
N1 arm; it is disclosed in the draft (section 6).

## 6. What this changes for the programme

Any earlier W-TRANSP navtest number (TF-1, TF-2 transparent comparisons, E2) carries the same 1 to 15 cm scenery difference. Tracked EP and
PDMS from W-TRANSP are within ~1e-3 of the harness on most tokens and 0.1 or more on ~10, so they are fine for ranking and not admissible as
"harness-identical". Register this on `INSTRUMENTS.md` (W-TRANSP: control "tracked == pinned W3 harness" MEASURED FAIL on navtest; fix =
`--metric-cache-root`).
