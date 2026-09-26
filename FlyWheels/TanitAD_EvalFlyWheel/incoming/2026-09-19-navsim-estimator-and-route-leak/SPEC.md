# SPEC — the NavSim interval: a LOG-CLUSTER bootstrap of the OFFICIAL aggregate (pre-registered)

**Stream** E3 → W2 of the eval-suite build · **Written** 2026-09-19, AFTER the structural census
(`raw/split_census.json`, `raw/two_stage_*.json`, `raw/key_gap_within_log.json` — counts only) and
**BEFORE any interval was computed on any real score.** No NavSim interval exists in the programme
at the time of writing; the only real NavSim scores on disk are E1's/E2's warmup runs, whose split
cannot carry an interval under this SPEC (§5). Blob hash → `raw/SPEC_PREREG_HASH.txt`.
**Implementation** `taniteval/adapters/navsim_ci.py` · **Tests** `taniteval/tests/test_navsim_ci.py`
· **Gate** `CRITERIA_REGISTRY.json → benchmarks.navsim.GATE_estimator_cluster_unit` (evaluated by
`tools/criteria_check.py`).

## 1. What the interval answers — and the two questions it does not

It answers **"would another draw of LOGS like these give this score?"** (evaluation-set variance).
It is structurally blind to **training** variance (`H-ESTIM-SEED-1`: a zero-lever replicate cleared a
paired bootstrap on 14.3 % of cells) and to **inference** variance of a sampling planner (refav1's
seed floor ≈ 0.30 m). ⇒ For any claim that a LEVER moved EPDMS, a separated paired interval is
**necessary, not sufficient**: the claim also needs a replicate arm read against the same interval.
Every emitted record carries this sentence (`question_answered`).

## 2. The measured structure it is built on (MEASURED, `raw/split_census.json`)

Two independent derivations of every stage-1 scene list — the devkit's own `filter_scenes`
(source text executed verbatim from the pinned blob) and a re-derivation — agree **token for token
on all 7 local splits**. Read controls: 147/147 log pickles read, 0 failures, 75,122 key frames.

| split | stage-1 scenes | stage-2 | mapping keys | `log_name` (segments) | nuPlan drives | cities | scenes/log min·med·max |
|---|---|---|---|---|---|---|---|
| **navhard_two_stage** | 450 | 5,462 | 225 | **76** | **34** | 4 | 2 · 4 · 30 |
| **navtest** | 12,146 | — | — | **136** | **44** | 4 | 8 · 70 · 568 |
| warmup_two_stage | 16 | 204 | 8 | **7** | 7 | 3 | 2 · 2 · 4 |
| private_test_hard_two_stage | 140 (YAML) | 1,732 (YAML) | `null` (private) | **not observable**: 0/140 tokens in the local test metadata (control: navhard 450/450 found) | — | — | — |
| test (OpenScene standard, CONTROL) | 5,044 | — | — | 138 | 44 | 4 | 3 · 27 · 223 |

**Overlap — where the dependence lives:**
* **Within a log, scenes overlap heavily.** navtest scenes are 14-frame windows at a stride of ONE
  key frame: consecutive scenes share a median **13 of 14 frames**, **92.6 %** of consecutive pairs
  share ≥ 1 frame, and each key frame sits in **5.55 scenes** on average (frame-reuse factor).
  navhard: a mapping key's `orig`/`prev` tokens are adjacent frames (Δ = 1 frame = 0.5 s, **225/225**)
  sharing **11 of 12** frames; the two-frame EC term couples them explicitly.
* **The control reads its known value:** the OpenScene standard `test` split (`frame_interval: null`)
  has frame-reuse factor **1.0000** and **0** of 158,789 within-log pairs sharing a frame — the
  docs' *"NavSim splits overlap; OpenScene standard splits do not"*, confirmed by measurement.
* **Across logs, nothing is shared.** Segments of one nuPlan drive: **0** time-overlapping adjacent
  pairs, **0** shared tokens; minimum gap **11.5 s** (navhard median 109.5 s, navtest 61.0 s).
* **Stage 2 never leaves its log.** Every synthetic scene's own `scene_metadata.log_name` equals its
  key's stage-1 log: **5,462/5,462** (navhard), **204/204** (warmup); its
  `corresponding_original_scene` is the stage-1 scene's final (t = +4 s) frame in every case.
  Every stage-2 token belongs to exactly one key (0 in more than one).
* navhard keys within one log do NOT share frames (gap ≥ 16 frames = 8 s), but they are close in
  time: median gap 40 s, **19.5 %** ≤ 12 s (`raw/key_gap_within_log.json`) — same road, same traffic.

## 3. The cluster unit — PRE-REGISTERED

**Primary unit: `log_name`** (the OpenScene log segment; NavSim's own `metric_cache.log_name`).
* It is the SMALLEST unit that contains every measured shared-frame dependence (within-log window
  overlap, the orig/prev EC coupling, every stage-2 scene) and it shares nothing with any other
  unit (§2). It is the analog of the programme's episode (a contiguous recording).
* **Rejected: scene token** — 5.55 scenes per frame on navtest; resampling them as independent is the
  `overlapping_holdout_se` class (a narrower interval that looks valid). **Rejected: mapping key**
  — keys of one log are 8–40 s apart on the same road; frame-disjoint is not independent.
* **Sensitivity unit: `nuplan_drive`** (the log name without its `_NNNNN_NNNNN` segment suffix).
  Segments of one drive share vehicle, day and route minutes apart — a soft dependence the census
  cannot rule out. It is computed beside the primary on every record that has ≥ 8 drives.

## 4. The statistic — the OFFICIAL aggregate, reproduced (PUBLISHED-CODE @0a380a9 / v1.1@3e8291b)

The interval is on the number the devkit publishes, never on a proxy:
* **Two-stage EPDMS** (`EPDMS_v2_navhard_two_stage`, `…_warmup_two_stage`,
  `…_private_test_hard_two_stage`): `run_pdm_score.py::calculate_individual_mapping_scores`
  L242-290 — per mapping key `(orig, prev)`:
  `c_k = ( s1(orig) · wavg(stage-2 "now" scores) + s1(prev) · wavg(stage-2 "prev" scores) ) / 2`,
  with `wavg` = `calculate_weighted_average_score` L221-239 over the SceneAggregator weights
  (stage-1 weight 1.0; stage-2 Gaussian σ² = 0.1 normalised within the group, uniform fallback);
  headline = `pd.DataFrame(all_group_scores).mean()` — a **skip-NaN mean over keys**.
* **Single-stage** (`EPDMS_v2_navtest_single_stage` — `run_pdm_score_one_stage.py` L293;
  `PDMS_v1_navtest` — v1.1 `run_pdm_score.py` L144): a skip-NaN mean of per-token `score`.
* ⭐ **Why resampling logs reproduces the official aggregation exactly:** each unit's contribution
  depends only on rows of ONE log (§2: stage 2 never leaves its log), and the official aggregate
  over any set of units is the skip-NaN mean of their contributions. So the statistic on a resample
  = the devkit's own aggregation applied to the resampled logs. Verified on the devkit's own code
  in four fixtures incl. physically duplicated logs (`raw/devkit_aggregation_reference.json`).
* ⛔ The published CSV **drops `weight`** (L422) — the stage-2 weights cannot be recovered from it.
  An interval needs the pre-CSV per-token frame (E1's `*_final_scores_frame.csv` carries it).
* **NaN (failed-scene) semantics are the devkit's:** a failed unit is SKIPPED, which changes the
  denominator. Every record reports `n_units_nan`; a paired comparison with different failure
  patterns is REFUSED (the arms would be scored on different scene sets — RG-13).

**Worked example (the literal the tests pin, derived on paper):** s1(orig) = 0.8; now-group
scores [1.0, 0.5] with weights [0.75, 0.25] → 0.875; s1(prev) = 0.6; prev-group [0.4, 0.2] with
[0.5, 0.5] → 0.3; `c = (0.8·0.875 + 0.6·0.3)/2 = (0.7 + 0.18)/2 = 0.44`.

## 5. The estimator

| parameter | value | source |
|---|---|---|
| resampling | clusters (`log_name`) with replacement, `n_clusters` draws per replicate | `taniteval/ci.py` `_draws` (reused, not re-implemented) |
| statistic per replicate | skip-NaN mean of the drawn clusters' unit contributions, with multiplicity | §4 |
| B | **2000** | brief; `taniteval.ci.DEFAULT_N_BOOT` |
| interval | percentile, α = 0.05 | as `taniteval.ci` |
| seed | 0 (recorded) | |
| **floor** | **n_clusters ≥ 8** | RG-14 (`tools/release_gate.py:85` `EPISODE_CLUSTER_FLOOR = 8`) |
| ceiling | n_clusters ≤ the split's `log_name` count (navhard 76, navtest 136, warmup 7) | census; MORE clusters than logs ⇒ a finer unit was resampled ⇒ FAIL |
| point | the full-sample official aggregate; must equal the devkit's summary row to ≤ 1e-9 when supplied, else REFUSED | §4 |

**Per split (consequence of the floor):** navhard ✅ (76 logs; drives 34 ✅) · navtest ✅ (136; 44 ✅)
· **warmup ⛔ NEVER** (7 logs < 8 — confirms `D-BENCH-PORT`) · **private_test_hard ⛔** (log
identities and per-scene scores are not released; the leaderboard server scores it) · navtrain —
a training split, never scored.

**Paired form (two arms, identical units):** the SAME resampled clusters in every replicate
(`taniteval.ci.paired_episode_cluster_bootstrap`, reducer = the §4 statistic); refused unless both
arms cover the identical units with identical failure patterns. **Decision predicate, pre-registered:
`separated` = the paired CI excludes 0 under `log_name` AND under `nuplan_drive`** (conjunction; no
tunable threshold). If the drive level has < 8 drives the record says `drive_level: CANNOT-RULE`
and any claim is stated as "separated at the log level only". And per §1 a lever claim additionally
needs a replicate arm.

## 6. Refusals (each is `{status: UNAVAILABLE, reason, n}` — a work item, never a pass)

n_clusters < 8 · point ≠ official value · all units failed · a replicate drew only failed units ·
paired arms with different units or failure patterns · unknown protocol / aggregation · a unit with
no cluster id (never invented).

## 7. What the gate admits (registry v2.10.0, `tools/criteria_check.py`)

PASS = `estimator.cluster_unit == "log_name"` AND an interval with `estimator ∈
{navsim_log_cluster_bootstrap, paired_navsim_log_cluster_bootstrap}`, `resample_unit == log_name`,
`8 ≤ n_clusters ≤ max_logs(split)`, `aggregation` = the protocol's official one, `n_boot ≥ 2000`.
REFUSED (work item) = an honest `{UNAVAILABLE, reason, n}`. **FAIL** = anything else: a scene-token /
mapping-key / episode-cluster / `overlapping_holdout_se` interval, n < 8, n > the split's logs, an
unnamed aggregation, a missing `cluster_unit`, no interval block.

## 8. Tests committed in advance, with the outcome each must show

| id | test | must read |
|---|---|---|
| T-ZERO | identical contributions (0.75) over 10 logs | lo = hi = 0.75, se = 0.0 **exactly** |
| T-TWO | 2 logs (all 1.0 / all 0.0), floor overridden to 2 | every replicate ∈ {0.0, 0.5, 1.0}; bootstrap mean 0.5 ± 0.03; lo = 0.0, hi = 1.0 |
| T-SE | 10 logs × 20 identical-within-log scenes, log means 0.1…1.0 | bootstrap SE = 0.0908 ± 8 % (analytic `0.28723/√10`) |
| T-MUT | the same with SCENE-TOKEN resampling (code mutation) | **RED**: SE ≈ 0.020 (fails T-SE by > 4×), `n_clusters` 200 > 10 ⇒ admissibility FAIL, checker violation |
| T-FLOOR | 7 logs | UNAVAILABLE, n = 7, reason names the floor 8 |
| T-KEY | the §4 worked example | 0.44 |
| T-DEVKIT | the F2 fixture (10 logs, 40 keys, NaN / missing-row / zero-weight / duplicate cases) | the devkit's own value 0.2886492611337752 (banked) |
| T-REPRO | E1's real CV warmup frame (`…/A1_final_scores_frame.csv`) + the warmup mapping | the devkit's `extended_pdm_score_combined` **0.1853562745165113** to ≤ 1e-12; bootstrap REFUSED (7 logs) |
| T-PAIR | paired arms with different failure patterns | REFUSED |

**Both outcomes committed:** a T-REPRO or T-DEVKIT failure means the aggregation is wrong and **no
NavSim interval ships** until it is fixed. A mutation that does not go RED means the test is
vacuous and must be rebuilt before landing. If the drive-level sensitivity disagrees with the
log-level interval on a real run, the conjunction rule decides — it is not re-litigated post hoc.

## 9. Known limitations (stated, not fixed here)

Percentile intervals are anti-conservative at small cluster counts (n ≈ 8–15); the floor is
RG-14's and is not moved here. The interval inherits every property of the devkit's scoring
(IDM-reactive background traffic in both stages; human agent undefined on two-stage splits — E1).
