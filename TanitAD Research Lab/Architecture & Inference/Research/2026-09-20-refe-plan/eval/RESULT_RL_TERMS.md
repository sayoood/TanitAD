# RESULT RL-S1 part 2: the `rl_terms` labeller (D, K, L) and G-REWARD-ID (2026-10-12)

**What this is:** `refe/rl_terms_label.py` writes the per-proposal inputs `rl_reward.py` needs and the PDM labels do not carry. **No model is trained or scored.**
Evidence classes: MEASURED = ours, artifact path given; ESTIMATED / UNVERIFIED are marked. Code under test: `refe/rl_terms_label.py` sha256 `0fa4bf81...` (recorded in
`eval/raw/rl_terms/identity_navtest.json`), `eval/rl_terms_identity.py` sha256 `8d4d90f7...`.

## 0 · Verdict

**G-REWARD-ID PASSES, with zero disagreements, on two independent populations** (MEASURED):

| population | proposals | `g>0 & DAC=0` | `g>0 & DAC=1` | `g=0 & DAC=0` | `g=0 & DAC=1` |
|---|---|---|---|---|---|
| navtest, 998 scenes, final model's 64 hypotheses, NAVSIM scorer run in-process | 63,872 | **8,467** | **0** | **0** | **55,405** |
| real on-policy proposals (432 sets, ckpt-4933 dumps), production CLIs + side-file join | 27,648 | **5,735** | **0** | **0** | **21,913** |

(`eval/raw/rl_terms/identity_navtest.json` `.hypotheses_64.exact_g_gt_0`; `eval/raw/rl_terms/identity_onpolicy.json`. Adding the human plan: 64,870 proposals, same zeros.)
`D` is exactly `0.0` on all 55,405 + 21,913 compliant proposals and positive on every failing one (on-policy: 5,734 of 5,735; the one exception has `gmax` = 4e-5 m, its D is below 5e-6 and the 5-decimal rounding of the side file writes 0.0; `gmax` keeps it, which is why the join gate reads `gmax`, not `d`).

**Controls, same run, same scenes (MEASURED, `identity_navtest.json .controls`):**
- C0 vendored-free executed states == the real `PDMScorer._states` **bit for bit on 998/998 scenes** (max abs 0.0).
- C1 our bbox corners / centres == navsim `state_array_to_coords_array`: max abs 9.3e-10 m. Pacifica constants match `get_pacifica_parameters()`.
- C2 per STEP: `(g>0)` == `PDMScorer._ego_areas[..., NON_DRIVABLE_AREA]` on 2,659,670 steps (90,877 non-drivable): **0 mismatches**.
- C3 the scorer's DAC in this run == the census DAC from NAVSIM's real navtest cache: 64,870 / 64,870.
- C4 K inputs: offset == `Point.distance(centerline.linestring)` (max abs 0.0); heading error == `PDMPath.interpolate(project)` (7.6e-14 deg); INTERSECTION switch == `dam.is_in_layer` (0 mismatches);
  `multiple_lanes` == the scorer's `_ego_areas[..., MULTIPLE_LANES]` (0 mismatches).

**Deliberate-regression arms (what a mutation of the labeller would do; all must break the identity, all do):**

| arm | missed failing proposals (of 8,467) | false alarms (of 55,405 compliant) |
|---|---|---|
| R1 the 8-pose plan instead of the executed states | 2,621 (31.0 %) | 1,608 (2.9 %) |
| R2 the untracked 41-step interpolation of the plan (no LQR) | 2,406 (28.4 %) | 1,723 (3.1 %) |
| R3 bbox centre instead of the 4 corners | 7,295 (86.2 %) | 5 |
| R4 min over corners instead of max | 8,315 (98.2 %) | 0 |

P5 is therefore not a nicety: 28 to 31 % of the fan's DAC failures exist only on the tracked states (consistent in size with the 29 % the review measured on the deployed student, a different population).

## 1 · Findings that change how the terms are used (each read from source or measured)

1. **The DAC polygon set in OUR harness (NAVSIM v1.1 @ 3e8291b) is not the review's list.** `_calculate_ego_area` fails a step when fewer than 4 corners lie in at least one polygon typed
   `ROADBLOCK / INTERSECTION / DRIVABLE_AREA / CARPARK_AREA`; `from_simulation` adds the ROADBLOCK, INTERSECTION and CARPARK_AREA polygons and **not** the ROADBLOCK_CONNECTOR polygon (only its interior lanes, typed
   LANE_CONNECTOR). Review 3.2(b) / 4.1 quote the v2 devkit. The labeller selects by type name from the scene's own `PDMDrivableMap` (`DAC_LAYERS`), so it follows the scorer. Three mutations pin it.
2. **`g > 0` is the identity; a tolerance is not.** 19 of 8,467 failing hypotheses (0.22 %) have g <= 1 mm; `g > 1e-3` would call them compliant. Consumers must test `gmax > 0`, never a tolerance. D for those is ~1e-6.
3. **`PDMPath` interpolates with scipy `interp1d` on a 2-D array, which returns the FIRST of duplicated knots; `np.interp` returns the last.** The PDM-Closed centreline repeats vertices with a heading jump;
   `np.interp` was **0.66 deg** off at 3 of 2,665 points on the first scene. `CenterLine` reimplements the `interp1d` rule (`_interp_linear`), pinned against scipy on duplicate-laden data and by a mutation.
4. **The vendored simulator differs from the real one by ~3e-6.** `navsim_dac.simulated_states` was not bit-equal to `PDMScorer._states` (max abs 3.3e-6). `executed_states` therefore runs the front half of `score_plans`
   itself (textually pinned to `navsim_pdm_targets.py` by a test), and C0 is bit-exact.
5. **D carries real tie-breaking signal where PDMS cannot (P3).** Of 998 navtest scenes: 577 have all 64 hypotheses compliant (D identically 0), 11 have all 64 failing (D varies in all 11), 410 are mixed.
   Among failing hypotheses D has median 0.109, p90 0.391, max 0.746 (gmax median 0.43 m, p90 1.59 m, max 7.14 m); 14.9 % have D < 0.01.
6. **On the 432 on-policy sets the top of the reward is tied in most scenes** (`onpolicy_summary.json`): 318 scenes have several proposals tied at the maximum PDMS, 317 still tied under the core reward (D + guard barely moves
   the argmax: it differs from the PDMS argmax in 1 scene), 103 under R2. K and L break about two thirds of the ties; the R2 pick gives up at most 0.021 PDMS against the best (mean 0.0003; 5 scenes above 0.01), so P4 holds.
   `np.argmax` takes the first of exact ties; read the tie counts, not the argmax-changed counts (232 of 432), as the measure.

## 2 · K and L on real data (MEASURED; rates, not validations)

| quantity | navtest, 998 scenes (hypotheses) | on-policy, 432 sets |
|---|---|---|
| ego starts on a LANE (L is inert otherwise) | 664 (66.5 %) | 286 (66.2 %) |
| label (i) human lane change | 11 (1.1 %) | 13 (3.0 %) |
| label (iii) route needs a lane change | 12 (1.2 %) | 5 (1.2 %) |
| label (i) OR (iii) fires | 19 (1.9 %) | 13 (3.0 %) |
| `lc_executed` per proposal | 1.10 % | 2.33 % |
| `dither` per proposal | 6.59 % | 7.92 % |
| mean K (suppressed / raw) | 0.144 / 0.152 | median 0.119 / 0.135 |
| mean D | 0.0209 | 20.7 % of proposals have D > 0 |
| mean abs contribution to R of lam_d D / lam_k K / lam_l L | | 0.0065 / 0.0089 / 0.0043 (PDMS 0.63, guard 0.034) |

A third of scenes start in a connector or off any lane polygon, where `lc_executed` is 0 and the label is False by construction. The label (i) rates sit around the review's 2.7 % pilot (9 of 336, interval 1.4 to 5.0 %): on-policy 3.0 % is inside it; navtest 1.1 % (Wilson interval about 0.6 to 2.0 %, hand arithmetic) overlaps it.
The review's own warning stands: K and L cannot move PDMS; they act only on ties and on the lateral family.
**UNVERIFIED:** nothing here checks `dither` or `lc_executed` against human judgement. 6.6 to 7.9 % of proposals carry the dither flag at the 0.15 m/s dead band; that is a definition read off the data, not tuned, and it is
a candidate to review before R2 uses it at weight 0.05.

## 3 · Interpretations the review left open (flagged; each recoverable from the record)

1. **K and the lane-change label.** Review 3.2(c) says K is suppressed in intersections and when a lane-change label fires; `rl_reward.lane_consistency` has no label argument. Applied at labelling:
   when `label_fires`, the INTERSECTION mask is OR-ed with the scene flag (k1 = k2 = 0) and the straddle fraction is zeroed. **`k` is the suppressed value (the one the EPO loader reads); `k_raw` is stored too.**
   13 of 432 on-policy scenes differ. If the Master Mind wants unsuppressed K, either the loader reads `k_raw` or the sets are relabelled.
2. **Straddle** `k3` = fraction of the 41 steps with the scorer's MULTIPLE_LANES rule true and not in an intersection.
3. **dither:** sign changes of the lateral velocity, lateral = signed offset to the PDM-Closed centreline, velocity over 0.5 s, |v| < 0.15 m/s ignored, >= 2 changes.
4. **lc_executed** = label (i)'s rule on the executed rear-axle poses at 0.5 s (a pose in another LANE of the start roadblock, final pose > 2.0 m from the start-lane baseline).
5. **label (iii)** is the definition only the review gave: the start lane's outgoing edges (lane connectors, then the lanes they feed) reach neither of the next two route entries while an adjacent lane's do.
6. **Not implemented:** the optional inner-margin term of 3.2(b); the slow-lead permission (ii), which `rl_reward` has no input for.

## 4 · Contract with the consumer

`scorer_finetune.py` (ac4d3b8) globs `rl_terms_*.jsonl` and validates each line with `_rl_terms_record`. The labeller writes `<out>/rl_terms_<set file name>` (the pdm side files are `pdm_<same name>`), key
`[log, token, step, rank]` + `ckpt_step`, and the test extracts `_rl_terms_record` from `scorer_finetune.py` by AST and runs the labeller's record through it (plus a negative control, so the validator is live).
**No schema change needed.** The record carries extras the loader ignores: `gmax`, `k_raw`, `label_version`, `labels`, `plans`. `eval/rl_terms_join_identity.py` joins `gmax` to `pdm_targets.dac`,
so G-REWARD-ID can be re-run on the pod from side files alone.

## 5 · Cost (MEASURED, one process, warm, 22 scenes, dev box)

Per scene: `metric_cache_parts` 0.47 s, `executed_states` 0.15 s, `scene_geometry` + lane context 0.04 s, `terms_from_states` 0.68 s = **1.34 CPU-s** (`score_plans` for reference: 0.28 s).
**ESTIMATED:** ~51,000 scenes = ~19 CPU-hours, about 1.2 h on 16 workers; pod CPUs may differ. The PDM relabel and this one both rebuild `metric_cache_parts`; a fused pass would save ~0.5 s per scene (not built).
`terms_from_states` is the largest term and has not been optimised.

## 6 · What is not done

- The pod chain script and the RL-train / RL-val partition (out of scope; the partition waits on the open base checkpoint).
- Navtrain tokens: the identity ran on navtest (in-process scorer) and on the dev box's own ckpt-4933 on-policy dumps, **not** on the ~51,000-scene navtrain dump. Run `rl_terms_join_identity.py` on the pod output as the post-label gate.
- A navtrain-token label-rate run (review 4.2 follow-up) was not repeated; the rates above are navtest and the 432 dev-box sets.

## 7 · Reproduce

```
python eval/rl_terms_identity.py --n 1000 --workers 14          # navtest identity + controls + regression arms (10 min; relaunches itself in the navsim venv)
bash eval/rl_terms_onpolicy_local.sh 'onpolicy_r0_m5w[0-2].jsonl' 150 12     # pdm relabel + rl_terms CLI + join on local on-policy sets
python eval/rl_terms_onpolicy_summary.py --dir eval/raw/rl_terms/onpolicy_local --out eval/raw/rl_terms/onpolicy_summary.json
python refe/test_rl_terms_label.py                              # ZZTEST_PASS 39 (1 suite + 37 RED mutations + coverage)
python refe/rl_terms_label.py --sets <dir> --out <dir> --workers 16 [--served-only] [--limit N] [--shard I/N] [--terms DKL]
```

## 8 · Deliverable manifest

| artifact | where | copies |
|---|---|---|
| `refe/rl_terms_label.py` (labeller + CLI) | repo, staged | 1 (repo working tree) |
| `refe/test_rl_terms_label.py` (`ZZTEST_PASS 39`) | repo, staged | 1 |
| `eval/rl_terms_identity.py`, `eval/rl_terms_join_identity.py`, `eval/rl_terms_onpolicy_summary.py`, `eval/rl_terms_onpolicy_local.sh` | repo, staged | 1 |
| `eval/raw/rl_terms/identity_navtest.json` + `.log`, `identity_onpolicy.json`, `onpolicy_summary.json` | repo, staged | 1 |
| `eval/raw/rl_terms/onpolicy_local/` (3 `pdm_*` + 3 `rl_terms_*` side files, logs; 2.3 MB) | repo, staged | 1 |
| this file | repo, staged | 1 |

Nothing lives only on a pod. Nothing is committed or pushed. No existing file was edited.
