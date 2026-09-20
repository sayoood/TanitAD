# SPEC — E1: NavSim v2 harness validation on `warmup_two_stage` (PRE-REGISTERED)

**Written 2026-09-19 11:10 local (Europe/Berlin), BEFORE any scoring run.** Nothing below was
adjusted after a number was seen. Later additions go in RESULT.md, marked post-hoc.

## 0. The harness — "the SHA plus any patch is part of the column"

| component | pin | local modification (pre-existing, found 2026-09-19, NOT made by E1) |
|---|---|---|
| navsim devkit | `0a380a9063d7162ec93d0f51e9990ebac585f720` | `navsim/common/dataclasses.py`: HEAD blob `a358d07…` → worktree blob `596cb7d…` (diff sha256 `d5fe71fe…`). A `_PosixSafeUnpickler` that maps `pathlib.PosixPath` → `PurePosixPath` in `Scene.load_from_disk`, dated 2026-08-28 in its own comment. Runtime-relevant (synthetic-scene loading). |
| nuplan-devkit | `ce3c323af01c0d7ec5672f7832ef53f9c679aab0` | `setup.py`: HEAD `f6ddafc…` → worktree `3ad18c6…`; adds `'docs','docs.*'` to `find_packages(exclude=…)`. Packaging only, no runtime effect. |
| E1 patch (planned) | — | ⛔ none in place. Any Windows defect is fixed by a **monkeypatch in `code/navsim_win.py`**, applied before the official, unmodified script runs. Predicted need (from source): `MetricCacheLoader._load_metric_cache_paths` (`navsim/common/dataloader.py:316`) takes the token as `cache_path.split("/")[-2]`, while nuPlan's `save_cache_metadata` writes `str(WindowsPath)` (backslashes) ⇒ **predicted `IndexError` on Windows**. If it does not fail, no patch is applied and that is recorded. |

Python 3.9.25; hydra 1.2.0; numpy 1.23.4; pandas 2.3.3; shapely 2.0.7 (navsim venv). CPU only.

## 1. The split — expected counts, derived TWICE from config before running (MEASURED)

`config/common/train_test_split/warmup_two_stage.yaml` + `scene_filter/warmup_two_stage.yaml`, parsed
by `code/count_splits.py` (raw: `raw/split_counts.json`). Two independent derivations agree:

| quantity | from `reactive_all_mapping` | from the scene-filter lists | agree |
|---|---|---|---|
| logs | — | 7 `log_names` | — |
| stage-1 tokens (16-hex) | 8 entries × (now, prev) = **16** | `tokens` = **16** | ✅ identical sets |
| stage-2 tokens (17-hex) | 102 pairs × 2 = **204** | `reactive_synthetic_initial_tokens` = **204** | ✅ identical sets |

navhard (for §7 pricing): 76 logs, 225 entries → **450** S1, 2,731 pairs → **5,462** S2 — both
derivations agree, which verifies the Lab's INHERITED "450 + 5,462" from the local yaml.

## 2. Arms

| id | agent | runner | traffic | purpose |
|---|---|---|---|---|
| **A1** | `constant_velocity_agent` | official two-stage `run_pdm_score.py` | as configured (see §5) | **the floor; the primary EPDMS result** |
| **A2** | `human_agent` | official two-stage `run_pdm_score.py` | as configured | privileged log replay — see prediction P-HUM |
| **A2b** | `human_agent` | official `run_pdm_score_one_stage.py`, `traffic_agents=reactive`, `include_synthetic_scenes=false` | IDM (same policy as A1's stage 1) | the human's stage-1 numbers + control C1 |
| **A1b** | `constant_velocity_agent` | as A2b | as A2b | cross-runner control C6 |
| **M1** | `human_agent` | as A2b + `scorer.config.human_penalty_filter=false` | as A2b | **mutation**: gives C1 teeth; counts how often the filter fires |
| **R1** | `constant_velocity_agent` | as A1, fresh process | as A1 | replicate: determinism / order-invariance control C7 |
| ~~ego_status_mlp~~ | — | — | — | **SKIPPED by rule**: no trained checkpoint (`*.ckpt`/`*.pth` absent under `C:\Users\Admin\navsim`, and no `*ego*mlp*`/`*ego_status*` under `D:\Projects\TanitAD-artifacts` or `~\.cache`; two locations probed). Not trained (brief). |

Worker: `sequential` (1 process). Metric cache at `C:\Users\Admin\navsim\exp\metric_cache_warmup_two_stage`.

## 3. Predictions, both outcomes committed

**P-HUM (the human agent is undefined on stage 2).** MEASURED from data before running: all 204
synthetic pickles carry exactly **4 frames, `num_history_frames=4`, `num_future_frames=0`**
(`code/probe_data.py`). `HumanAgent.compute_trajectory` returns `scene.get_future_trajectory(8)`,
which indexes `frames[3..11]` (`dataclasses.py:367-371`). ⇒ predicted: **204/204 stage-2 rows
`valid=False`** in A2, and the two-stage aggregation for A2 fails or is NaN.
* If confirmed → the human is a **stage-1-only** reference on this split; its combined EPDMS is
  reported **UNDEFINED**, never imputed. (Consistent with [N2]'s navhard table, which has no Human row.)
* If refuted (stage-2 rows valid) → my reading of the data or code is wrong; I find out what
  trajectory it emitted **before** using any A2 number.

**E1 (ordering, stage 1).** EPDMS_S1(CV, A1) **<** EPDMS_S1(human, A2b). This is an expectation,
not a devkit identity. If CV ≥ human on stage 1, it is reported as such and investigated
per token before any number is released (a log-replay reference with the human filter should
not lose to kinematic extrapolation on a hard split).

**E2 (ordering, EP).** EP_S1(CV) < EP_S1(human). Expectation only — v2 EP is NOT zeroed for
violating proposals (unmasked numerator; `NAVSIM_PROTOCOL.md` §2.3), so no devkit guarantee.

**E3.** A1 combined EPDMS ∈ [0, 1] and finite; A1 has 220/220 valid rows.

## 4. Controls that must read KNOWN values — derived from source; any failure is a HARNESS DEFECT and outranks every result

**C1 — human-filter identity (stage 1, A2b).** `pdm_score.py:171-219`: on ORIGINAL scenes, any
sub-metric the human scores `== 0` is set to `1` on the agent. For the human agent the agent
trajectory IS the human trajectory (precondition P1), so a binary metric reads 1 whether or not the
filter fired. ⇒ on **every** valid stage-1 token: **DAC = TLC = TTC = LK = 1.0 exactly**, and
**NC ∈ {0.5, 1.0}, DDC ∈ {0.5, 1.0}** (0.5 is not filtered — the rule is `== 0`).
EP and HC are **never** filtered and are not constrained: in the human call EP is
self-normalised (one proposal ⇒ `pdm_scorer.py:232-237` gives 1.0) and HC is 1.0 because
`human_past_trajectory` is not passed (`pdm_score.py:184-192` → `pdm_scorer.py:663`).
* **P1 (precondition, checked independently):** `HumanAgent`'s trajectory
  (`Scene.get_future_trajectory`, `dataclasses.py:357-384`) equals `metric_cache.human_trajectory`
  (`metric_cache_processor.py:288-311`) on all 16 tokens; max |Δpose| reported.
* **Teeth (M1):** with the filter OFF, count `n_fired` = stage-1 tokens where the human reads 0 on
  any filterable term. **If `n_fired = 0`, C1 is VACUOUS on this split** (the human violated
  nothing) and is reported as vacuous, never as a pass of the filter.

**C2 — scene-type classification.** The cache labels a scene SYNTHETIC by `len(token) == 17`
(`metric_cache_processor.py:317`) — a heuristic that also gates the human filter. Independent
check: every cached entry's `scene_type` must match its membership in the yaml lists (`tokens` ⇒
ORIGINAL, `reactive_synthetic_initial_tokens` ⇒ SYNTHETIC). 220/220.

**C3 — counts (no success over an empty set).** Cache: exactly 16 ORIGINAL + 204 SYNTHETIC = 220
entries whose token set **equals** the yaml sets; zero missing, zero unused. Scorer (A1): 16 S1 +
204 S2 per-token rows, **all `valid`**, plus 3 summary rows. A zero anywhere is a FAIL.

**C4 — per-token EPDMS identity.** For every valid row, the CSV `score` must equal
`NC·DAC·DDC·TLC · (5·EP + 5·TTC + 2·LK + 2·HC + 2·EC) / 16` recomputed from the CSV's own
sub-metric columns, |Δ| ≤ 1e-9. The formula is taken from `docs/metrics.md` (independently
authored from `compute_final_scores`). This is also a regression test of the real historical
defect, navsim issue #151 (2025-09-29): `multiplicative_metrics_prod`/`weighted_metrics` not
re-derived after the human filter.

**C5 — aggregate identity.** The three summary rows (`extended_pdm_score_stage_one/_two/_combined`)
recomputed by my own code from the per-token rows, the yaml mapping, and Gaussian weights
recomputed from `endpoint_*` / `start_point_*` with σ² = 0.1 (`scene_aggregator.py:18,34-45`),
|Δ| ≤ 1e-9. Inputs captured by an instrumentation hook on `compute_final_scores` (returns its
output unchanged).

**C6 — cross-runner identity (A1 vs A1b, stage 1).** Same agent, same cache, same IDM policy, same
`pdm_score()` ⇒ NC, DAC, DDC, TLC, EP, TTC, LK, HC **bit-identical** per token. EC: identical on
"now" tokens; on "prev" tokens the one-stage runner leaves EC NaN and re-weights /14
(`run_pdm_score_one_stage.py:177-189`) — a known, predicted divergence, not a defect.

**C7 — determinism / order invariance (A1 vs R1).** No RNG exists in the scoring path
(`grep random` over `evaluate/`, `planning/simulation/`, `traffic_agents_policies/`,
`planning/metric_caching/`, `run_pdm_score.py`, `dataloader.py`: **0 files**), but token order
comes from `list(set(...))` and so varies with the per-process string-hash seed. ⇒ every
per-token value and all three summaries must agree **exactly** (max |Δ| = 0.0), compared by
token, never by row position.

**C8 — external reference (secondary, INHERITED).** The HF warmup leaderboard row
`baseline_constant_velocity` = **18.5356** (`NAVSIM_PROTOCOL.md` §6.3, retrieved 2026-08-23, not
re-verified — the brief forbids downloads). `docs/submission.md`: local warmup results *"should
match"* HF. Reading: |100·A1 − 18.5356| ≤ 0.01 ⇒ external reproduction. Otherwise the delta is
reported with its candidate causes (server devkit version unknown — e.g. the 2025-09-29 human-filter
fix and the v2.2 EPDMS update post-date the warmup opening). ⛔ A C8 miss alone is NOT a harness
defect, because the server's version is unknown; C1–C7 decide that.

## 5. Tier / loop stamps

* **Tier T1-family** — the agent's own plan is executed by LQR + kinematic bicycle; no recorded
  future is fed back. **Loop OPEN** for stage 1 (PI ruling 2026-09-02: the plan is fixed, the ego is
  never re-queried). **Stage 2: loop status UNRULED** under the 2026-09-02 vocabulary (a one-shot
  re-perception of a 3DGS-rendered perturbed start). Not ruled here.
* ⚠️ **Correction to the brief's premise, from source:** the official two-stage runner instantiates
  `cfg.traffic_agents_policy.reactive` for **both** stages (`run_pdm_score.py:77-79, 132-134`) =
  `NavsimIDMTrafficAgents` (`navsim_IDM_traffic_agents.yaml`). **Background vehicles are
  IDM-REACTIVE**, not log-replay; pedestrians/static objects follow the log; the **ego remains
  non-reactive**. This does not change the ego's loop status, but "non-reactive logged agents"
  must not be written for this protocol.

## 6. Estimator

⛔ **Warmup NEVER carries a CI** (D-BENCH-PORT, PI 2026-08-29: *"7 log-groups < the RG-14 floor of
8"*; the 7 is re-MEASURED here from the scene filter). Every artifact carries
`estimator.interval = {status: UNAVAILABLE, reason, n}` and `estimator.cluster_unit` UNAVAILABLE.
Point estimates only, with per-stage n.

## 7. navhard pricing (ESTIMATED, from measurement)

Measure wall-clock and peak RSS of (a) cache build and (b) A1 scoring, per stage-1 and per stage-2
scene where separable; extrapolate linearly to 450 S1 + 5,462 S2. Excludes download time
(31 GB sensors + synthetic pickles, not on disk) and the one-off map load. Mark ESTIMATED with the
per-scene measurement it scales.

## 8. Budget and stop rules

Cache ≤ 2 h wall-clock (stop and report beyond). 1 worker process. System free RAM is sampled every
2 s by the wrapper and the run aborts itself if it falls below 3,000 MB. GPU and the live training
process (PIDs 16996/21724) are never touched. Nothing is downloaded.
