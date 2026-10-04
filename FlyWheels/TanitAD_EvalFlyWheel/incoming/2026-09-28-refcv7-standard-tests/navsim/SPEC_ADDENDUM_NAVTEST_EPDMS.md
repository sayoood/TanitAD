# SPEC ADDENDUM — refcv7 on NAVSIM v2 EPDMS, `navtest`, ONE stage — PRE-REGISTERED

**Written 2026-10-04 ~02:15 Berlin, BEFORE any refcv7 EPDMS number exists and before any scorer has been
launched by this addendum's tooling.** The sha256 of this file and its UTC write time are in
`raw/SPEC_ADDENDUM_NAVTEST_EPDMS_SHA256.txt`. Any later change is a dated amendment *below §11*, made before
the numbers it touches, with its own hash line. This file is an ADDENDUM to `SPEC.md` (same package) and binds
only the work it names; it changes no bar, no primary arm and no floor of `SPEC.md`.

**Authority and place in the SPEC.** `SPEC.md` §4 scores navtest with NAVSIM **v1.1 PDMS** over the full 12,146
tokens (`BAR-R7-N1`, SPEC_REFCV7 §3). **That bar is untouched and is never replaced by anything here.** The
NAVSIM v2 metric EPDMS was scored only on `warmup_two_stage` and `navhard_two_stage`. This addendum adds the
EPDMS on **navtest** — the column most NAVSIM-v2 method papers quote beside navhard — as a **SECONDARY READ
with NO bar**, by RE-SCORING plans that are already banked. No model forward pass is run.

## 0. Facts this addendum is built on (MEASURED 2026-10-04 ~01:55–02:10 Berlin by this stream, unless marked)

| # | fact | evidence |
|---|---|---|
| F1 | A **complete v2 navtest metric cache exists**: 136 log directories, **12,146** `metric_cache.pkl` (counted with `find`), `CACHE_DONE.json` says `token_set FULL_SPLIT`, 12,146 stage-one / 0 stage-two, `tokens_sha256` `8d42fef68542f095d0e03894fdaf38babfdc89f3fd7cdc26080c3a6009cb4108`. I recomputed `CACHE_MANIFEST.json` sha256 `b02d1a833f205035421e5ce5b180a8d41657bd7d234dd5866e2aa3df6ca04706` and the metadata CSV sha256 `c6b9c4bf6807da78e1770f564fb469d0eef3dfab5b77f5553a01acc07f1bf840` — both equal `CACHE_DONE.json`'s. (Built by W8, `2026-09-26-navtest-single-stage`; the per-entry lzma integrity claim is INHERITED from `CACHE_DONE.json`, not re-verified here.) | `C:/Users/Admin/navsim-crun/exp/metric_cache_navtest_v2/{CACHE_DONE.json,CACHE_MANIFEST.json,metadata/metric_cache_navtest_v2_metadata_node_0.csv}` |
| F2 | The four step-5,000 navtest seams `R7_A1`, `R7_A1_s1`, `R7_CEILDECL_d`, `PRIOR_ha0p` each hold **12,146 unique tokens**, finite `poses [12146,8,3]`, `sampling [8,0.5]`, `source = precomputed` on every row; their token set has sha256 (sorted, newline-joined) **`8d42fef6…4009` = the cache's `tokens_sha256`**; and **every row's `fingerprint` equals the v2 one-stage export's fingerprint (12,146 / 12,146 on each of the four seams)**. ⇒ the banked seams are consumable by the v2 one-stage scorer UNCHANGED (no re-bridge, no re-export). | `raw/milestones/step5000/bridge_navtest/seam_*.npz` vs `C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navtest_single_stage/navsim_agent_inputs.json` (sha256 `a41795496e00cb62ec7719781b744e2338c09340b2d55834be2f12a7328ade11`, `EXPORT_DONE.json`) |
| F3 | The runtime devkit's scoring-side files are byte-equal (after CRLF→LF) to upstream `autonomousvision/navsim@0a380a9063d7162ec93d0f51e9990ebac585f720` (post-#151): `run_pdm_score_one_stage.py` `caf57612…`, `evaluate/pdm_score.py` `51ce6f73…`, `pdm_scorer.py` `5ce74081…`, `scene_aggregator.py` `dfe75cbb…`, `metric_cache.py` `509f5bc3…`, `scorer/pdm_scorer.yaml` `eeaef5f7…`, `default_run_pdm_score.yaml` `96225a52…`, `docs/metrics.md` `e766aa9e…` | `C:/Users/Admin/navsim-crun/devkit` vs `D:/Archive/devbox-C/navsim/devkit` @0a380a9 |
| F4 | The scoring plumbing is the NavSim suite's promoted v2 scorer: `taniteval/taniteval/bench/navsim/{profiles.py,scoring.py}` profile `navtest_single_stage` (runner `pdm_score_one_stage`, `traffic_agents=non_reactive`, logs from `D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test`), the E1 wrapper `devkit_side/navsim_win.py` (sha256 `b955d362…`), and the lookup seam agent `devkit_side/tanitad_seam_agent.py` (sha256 `fdb54c49…`, **byte-identical** to E2's). | files as named |
| F5 | **No full-split v2 floor exists.** W8's full-split run (CV, STOP, HUMAN on this cache) was RAM_GUARD_ABORTed on all three arms, 2026-09-27 05:20–05:29 Berlin (`status FAILED, retryable`); no `verdict_full.json`. The floors are therefore scored here, on the same tokens. (Absence checked in W8's package and its run directory only; UNVERIFIED elsewhere.) | `2026-09-26-navtest-single-stage/raw/{full_waiter.log,full_cli_20260927T052007.log}` |
| F6 | W8's whole-log smoke (3 logs, 218 tokens; `_scratch`, a SUBSET, not claim-bearing) banked per-token STOP rows: headline 0.579805532200227. Used here as a known-value control (§6, KE2). | `taniteval/results/bench/_scratch/navsim_v2/navtest_single_stage/20260926T135821Z-navsim_v2-none-ce7ac0/scores/STOP.csv` |

## 1. Question

For the refcv7-r101-s0 milestone checkpoints (step 5,000 now; 30,000 and 50,400 as they complete), what is the
**official NAVSIM v2 EPDMS on the full navtest split (one stage)** of the banked plans, against STOP / CV /
HUMAN and the model's own prior — and does the sign of "model vs STOP" agree with the v1.1 PDMS reading?

## 2. Computability (the finding the PI asked for) — YES, with no forward pass and no new cache

Evidence (devkit `0a380a9`, runtime copy `C:/Users/Admin/navsim-crun/devkit`):

* The one-stage runner scores one query per original navtest frame with an agent's trajectory:
  `navsim/planning/script/run_pdm_score_one_stage.py` L39–126 (`run_pdm_score`), agent call L89–92,
  `pdm_score(...)` L94–101; token set = scene-loader tokens ∩ metric-cache tokens L79; background traffic L60–66.
* The agent is a **lookup**: `tanitad_seam_agent.TanitADSeamAgent.compute_trajectory` returns the banked
  `poses` row of the scorer token after a fingerprint check — it runs no model (F4; F2 proves every fingerprint
  matches the v2 export).
* The extra inputs EPDMS needs beyond PDMS are all in the **v2 metric cache** (`navsim/planning/metric_caching/
  metric_cache.py` L29–49: `past_human_trajectory`, `human_trajectory`, `drivable_area_map`, `centerline`,
  `route_lane_ids`, `observation`, `future_tracked_objects`, `map_parameters`) — and that cache exists (F1).
* Nothing blocks it except **RAM and wall-clock**: ESTIMATED ≈ 45–50 min per arm at one worker
  (W8's linear scaling of its smoke, `2026-09-26-navtest-single-stage/RESULT.md` §3; to be MEASURED by this
  stream's smoke, §6), peak RSS ESTIMATED 2–2.5 GB.

## 3. What is scored

**Tokens:** the full navtest split, 12,146 tokens, 136 logs, from the single cache F1. A different count is
REFUSED (§6 G1/G2). **No subset is ever a result.**

**Per milestone `step<N>` (arms with a banked navtest seam, scored from the SAME file the v1.1 PDMS scorer read):**

| arm | seam | role |
|---|---|---|
| `R7_A1` | `raw/milestones/step<N>/bridge_navtest/seam_R7_A1.npz` | the model AS BUILT (filter ON) |
| `R7_A1_s1` | `…/seam_R7_A1_s1.npz` | inference-seed replicate — the floor every read is held against |
| `R7_CEILDECL_d` | `…/seam_R7_CEILDECL_d.npz` | DIAGNOSTIC only (ceiling applied at the E9 argmax); never "refcv7's number" |
| `PRIOR_ha0p` | `…/seam_PRIOR_ha0p.npz` | the model's own kinematic prior, model-free |

**Floors (checkpoint-independent; scored ONCE, shared by every milestone, on the identical cache and tokens):**
`STOP` (all-zero plan for every token; seam built by the suite's own `seams.make_stop_seam` from the full v2
export), `CV` (the devkit's `constant_velocity_agent`), `HUMAN` (the devkit's `human_agent` — PRIVILEGED
logged future, context only, never a floor to beat).

**Not scored, with reason:** the 200-token navtest diagnostics (`R7_FILTOFF`, `R7_VMAXOFF`, `R7_VMAXORACLE`:
a random subset has no adjacent pair, so the one-stage aggregator `pd.concat([])`s on nothing —
`run_pdm_score_one_stage.py` L222 — and a subset is never claim-bearing because EC pairs adjacent scored
tokens); `R7_FILTOFF_d` (withdrawn by SPEC A1); the camera ablations (warmup only).

## 4. The metric, exactly

* **Definition (primary source):** `docs/metrics.md` L14–38. Multipliers NC, DAC, DDC, TLC; weighted EP (5),
  TTC (5), LK (2), HC (2), EC (2); `filter_m(agent, human) = 1.0` when the human scores 0 on `m`, else the
  agent's value. Weights: `scorer/pdm_scorer.yaml` L8–12; human filter ON: L27.
* **Implementation (devkit 0a380a9):** per-token metrics `pdm_scorer.py` L169–182, aggregation L223–251;
  the human penalty filter `evaluate/pdm_score.py` L171–219; two-frame extended comfort pairs a token with
  its adjacent previous token in the same log (start-time gap ≤ 0.55 s) `run_pdm_score_one_stage.py` L129–156,
  L200–227 (`SceneAggregator.aggregate_scores(one_stage_only=True)`, `scene_aggregator.py` L79–90); final
  per-token score with EC's weight zeroed when EC is NaN (denominator 14 instead of 16) L159–197.
* **Headline:** the `score` column's `average_all_frames` row = `pdm_score_df[score_cols].mean(skipna=True)`
  (`run_pdm_score_one_stage.py` L283–299), read as TEXT from the devkit CSV and ×100. ⛔ The `pdm_score` column
  is NEVER the EPDMS (EC masked out, /14) and is refused by name (`summarize.headline_value`).
* **Setting:** devkit `0a380a9` = post-#151; `traffic_agents=non_reactive` (log replay), passed explicitly;
  `PYTHONHASHSEED=1`, `OMP_NUM_THREADS=2`, `CUDA_VISIBLE_DEVICES=-1`. Published navtest-EPDMS rows mix
  pre-/post-#151 implementations and no banked primary states their traffic policy (W8 RESULT §6 E1,
  INHERITED): our number is comparable **only** to post-#151 non-reactive rows, and publication-style
  comparison is NOT a purpose here.
* ⚠ **EPDMS is not PDMS on a new column.** The v2 cache observes 4.0 s of agent tracks, the v1.1 cache 5.0 s,
  and single-observation objects are placed at every step ("ghosts") — so NC/TTC inputs differ between the two
  protocols for the same plan (W8 RESULT §2 and §6 E5, INHERITED, not re-verified). Cross-protocol numbers are
  never subtracted; the per-token cross-check of §6 KE7 is a plan-identity diagnostic only.

## 5. Statistics, estimator, seed floor — and what each interval answers

* **Per arm:** `taniteval/adapters/navsim_ci.py` (blob `ecffa36c6b608a2b92ccc9b6f0f34cc21bb2306a`)
  `log_cluster_bootstrap`, aggregation `single_stage_token_mean`, unit `log_name` (**136 clusters**, floor 8),
  **B = 2000, seed 0**, 95 % percentile interval.
* **Paired differences:** `paired_log_cluster_bootstrap` (same resampled logs in every draw), same B/seed/unit —
  for `R7_A1` − {STOP, CV, HUMAN, PRIOR_ha0p, R7_A1_s1, R7_CEILDECL_d}; and `PRIOR_ha0p` − STOP, `STOP` − `CV`.
* ⭐ **Every interval names its question.** The log-cluster bootstrap answers *"would another draw of LOGS give
  this score?"* only. **Inference variance** is read from `R7_A1_s1`. **Training variance is UNTESTED** (one run,
  `H-ESTIM-SEED-1`), so no read here may be called a lever effect.
* **Inference-seed floor:** `F = | EPDMS×100(R7_A1) − EPDMS×100(R7_A1_s1) |` on the full split (n = 1 replicate).
  For a paired difference `d` (×100): **SEPARATED** iff its paired interval excludes 0 **and** `|d| > 2F`;
  **NOT PROVEN** iff the interval excludes 0 but `|d| ≤ 2F`; **NOT SEPARATED** otherwise. `HUMAN` is context
  (privileged) and gets no classification word.
* **This is a SECONDARY READ. There is no bar.** A result here neither passes nor fails `BAR-R7-N1`, and an
  EPDMS reading never overrides the PDMS reading of the same checkpoint.
* **Four metric families.** The plans are unchanged, so LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC for
  `R7_A1` and `PRIOR_ha0p` on navtest are the already-banked `raw/milestones/step<N>/families_navtest_*.json`
  (plan-only, scorer-independent); this addendum adds the EPDMS's own nine components per arm (NC, DAC, DDC,
  TLC, EP, TTC, LK, HC, EC), their zero-rates, a split by driving command and by v0 speed band, and the
  stop fraction (4 s endpoint < 1 m, from the seam) beside every bar-like row.

## 6. Guards and controls (a failure REFUSES the headline; nothing partial is ever written as a result)

| id | check | pass |
|---|---|---|
| G1 | count guard (the suite's `score_arm`) | `Number of successful scenarios` == **12,146**, failed == 0, CSV token rows == valid rows == 12,146, exactly ONE summary row `average_all_frames` and no other protocol's rows, `agent_calls.seam == 12,146` and no other call class |
| G2 | token identity | the CSV's token set has sha256 `8d42fef6…4009`; 136 distinct logs under the W2 map |
| G3 | cache identity | `CACHE_DONE.json` `tokens_sha256` and `CACHE_MANIFEST.json` sha256 equal §0 F1's values at launch AND at summary time |
| G4 | seam identity | the seam's sha256 is recorded at launch and re-read at summary time (equal); its token-set sha256 equals G2's; the seam is the SAME file the v1.1 scorer read |
| G5 | definition | KE1 below holds on every token row; the `average_all_frames` score equals the skipna mean of the token scores within 1e-12 |
| KE1 | formula control, derived independently of the devkit | every token row: `score == NC·DAC·DDC·TLC · (5EP+5TTC+2LK+2HC+2EC)/16`, EC NaN ⇒ `(5EP+5TTC+2LK+2HC)/14`; max \|Δ\| ≤ 1e-9. The test pins the formula on hand-computed LITERALS (`0.71875`, `0.678571428…`, `0.359375`), never on an expression over the code |
| KE2 | **known value, banked** | STOP, scored through THIS driver on W8's 218 whole-log tokens (first 3 yaml logs, F6) against the FULL cache, reproduces W8's banked per-token cells (score and all nine components) \|Δ\| ≤ 1e-9 and headline 0.579805532200227. If it does not, NOTHING from this driver is quoted until the difference is explained |
| KE3 | determinism | the same arm scored twice on the same tokens is cell-identical (smoke: `R7_A1` step 5,000, one whole log) |
| KE4 | subset ↔ full consistency | the smoke's whole-log rows equal the same tokens' rows in the later full-split run of the same arm (\|Δ\| ≤ 1e-9); recorded when that run lands |
| KE5 | HUMAN analytic | on every `HUMAN` row NC = DAC = TTC = TLC = LK = 1 and DDC ∈ {0.5, 1} (W8's smoke: 0 exceptions) |
| KE6 | guards go RED | tests with a deliberate wrong count, a swapped token set, a `pdm_score`-only CSV and a changed seam byte each REFUSE |
| KE7 | plan-identity diagnostic (REPORTED, not a gate) | for the same arm and token, v2 DAC ≥ v1.1 DAC (the human filter is one-sided), equal except where the human's DAC is 0; the count of tokens with v2 DAC < v1.1 DAC is reported and is expected to be **0** — a non-zero count means the two scorers did not see the same plan |

## 7. Resource rules (binding for the tooling this addendum ships)

* The dev box's RAM is the binding resource (≈ 3–5 GB free at writing; a NavSim runner, a CPU bridge and a
  battery job are live). **One scorer process at a time.** A scorer is started only when
  `FreePhysicalMemory ≥ 9.0 GB` **and** `FreeVirtualMemory ≥ 6.0 GB` (commit headroom) on **5 consecutive
  samples 30 s apart**, re-read before every arm and every retry. A smoke below the full-split gate is allowed
  only for ≤ 50 tokens (peak RSS ≈ 0.5 GB, W8 MEASURED 484–515 MB on 218 tokens) and only after the same two
  numbers are read and logged.
* The wrapper's RAM guard is set to **4,000 MB** (every other registered job's floor is 3,000 MB), so this job
  is the one that yields; an abort with rc 3 is *retryable* and retried after a fresh gate.
* The tooling runs at BELOW_NORMAL priority, never touches a GPU (`CUDA_VISIBLE_DEVICES=-1`) or the GPU lock,
  and writes nothing under `raw/milestones/step<N>/` (outputs: `raw/navtest_epdms/`; waiter state/log only in
  `raw/milestones/epdms_waiter.{json,log}`).
* Disk is scarce (C: ≈ 3 GB, D: ≈ 4.7 GB free at writing): the per-arm pre-aggregation dump stays on C: scratch,
  is deleted after a PASS, and no arm starts with < 1.5 GB free on the scratch drive.
* The scorer's `thread-id` UUIDs (random per worker, not PhysicalAI clip ids) are replaced by `<thread-uuid>` in
  the logs banked into the package so the landing scan does not refuse them; the original log's sha256 is
  recorded beside it.

## 8. Stamps

Tier: **NavSim open-loop benchmark** (one query on a real logged frame; the plan is fixed and propagated by LQR
+ kinematic bicycle at 10 Hz over 4 s; background traffic non-reactive log replay), **zero-shot**
(PhysicalAI-AV B1 → nuPlan cameras, 3-camera stitch), **non-parity** data, ⛔ **never closed loop**. **Launch
tree: the speed ceiling does not reach the emitted plan** (SPEC_REFCV7 §26.1; this SPEC A1) — `R7_A1` is the
model AS BUILT, not as declared. Evidence class **MEASURED (ours + artifact path)** for every EPDMS number
written by this tooling; the devkit's definition is **PUBLISHED** (`docs/metrics.md`); W8-derived facts are
marked **INHERITED**. Device and precision of each model arm are read from its seam manifest and printed beside
its score.

## 9. Outputs

`code/score_navtest_epdms7.py` (plan · score · summarize), `code/epdms_waiter7.py` (detached, event-driven),
`tests/test_score_navtest_epdms7.py`; per arm `raw/navtest_epdms/{floors|step<N>}/<ARM>/` (`*.devkit.csv`
unmodified, `*.counts.json`, sanitised score log, wrapper manifest, `ARM_DONE.json`); per milestone
`raw/navtest_epdms/step<N>/summary_navtest_epdms.json` (schema `navtest-epdms7/1`); smoke and controls under
`raw/navtest_epdms/smoke/` and `raw/navtest_epdms/controls/` stamped **VALIDATION ONLY**; waiter state
`raw/milestones/epdms_waiter.json`, log `raw/milestones/epdms_waiter.log`.

## 10. Prediction (not a bar) and both outcomes, committed before any number

**P1.** At every milestone, `sign( EPDMS(R7_A1) − EPDMS(STOP) )` equals `sign( PDMS(R7_A1) − PDMS(STOP) )`
(at step 5,000 the PDMS Δ is **+3.7774**, paired interval [+1.58, +5.86], `raw/milestones/step5000/
summary_navtest.json`).
* **If P1 holds:** reported as "the PDMS ordering carries to EPDMS" with the §5 classification word; nothing else changes.
* **If P1 fails (EPDMS(R7_A1) ≤ EPDMS(STOP)):** it is reported plainly as a **disagreement between the two
  protocols on the same plan**, the automatic component table (Δ vs STOP for NC, DAC, DDC, TLC, EP, TTC, LK, HC,
  EC and the multiplier zero-rates) names the term(s) that flip it, and the next lever is that term — the
  summary states it in the same artifact (Rule Zero: a refutation is a waypoint). It does NOT amend
  `BAR-R7-N1`.
* Either way the rows are written with the classification words of §5 and the training-variance caveat.

## 11. Amendments

**A1 — 2026-10-04 02:03 Berlin (00:03Z), clerical, BEFORE any scorer ran:** the header above says the file was written "~02:15 Berlin"; the measured write time is **02:00:31 Berlin (00:00:31Z)**, as recorded in the first line pair of `raw/SPEC_ADDENDUM_NAVTEST_EPDMS_SHA256.txt`. No definition, arm, estimator, guard, control or rule changes. This amendment's own hash line is the second entry of that file.

**A2 — clerical, BEFORE any scorer ran:** A1 above states "02:03 Berlin (00:03Z)" as its own write time; that was typed, not measured. The measured write time of A1 is the second hash entry of `raw/SPEC_ADDENDUM_NAVTEST_EPDMS_SHA256.txt` (**00:00:40Z = 02:00:40 Berlin**). The hash entries, not the prose times in this file, are the record of when each version existed. No definition, arm, estimator, guard, control or rule changes.

**A3 — implementation record, BEFORE any full-split EPDMS score exists (only the VALIDATION-ONLY smoke of section 6 has run).**
No definition, arm, estimator, guard, control, statistic, or classification rule above changes. What the shipped tooling does where
the text above is silent or was approximate, stated so that nobody has to read the code to know:

1. **STOP seam.** Built by the suite's own `taniteval.bench.navsim.seams.make_stop_seam`, on a doc-shaped dict whose fingerprints come
   from `raw/navtest_epdms/inputs/token_meta.json` — a compact (2.3 MB) projection of the pinned full v2 export (sha256 `a4179549…ade11`,
   refused if it differs; token set and the 136 logs re-checked against the cache's pinned token hash). Control KE2 shows this STOP
   reproduces W8's banked STOP rows cell for cell.
2. **Smoke gate.** The full-split gate is unchanged (`FreePhysicalMemory >= 9.0 GB` AND `FreeVirtualMemory >= 6.0 GB`, 5 consecutive samples
   30 s apart, wrapper floor 4,000 MB). A <= 50-token smoke is admitted at `FreePhysicalMemory >= 4.0 GB` and `FreeVirtualMemory >= 4.0 GB` with
   a 3,500 MB wrapper floor (the box sat at 3.9–4.9 GB for hours; a 0.5 GB process leaves it above every other registered job's 3,000 MB
   floor). The two numbers are read and logged on the first line of every run.
3. **`separated`.** Beside the pre-registered log-level criterion of section 5 the summary prints the estimator's own `separated`
   (`navsim_ci.paired_log_cluster_bootstrap`: the log_name interval AND the coarser nuplan_drive interval, when >= 8 drives exist). The
   classification word uses the pre-registered log-level criterion only.
4. **Scratch and locks.** A full-split PASS deletes its scratch directory (every deliverable is banked first); the per-arm lock lives in
   scratch, not in the package. Raw `*.score.log` copies carry `<thread-uuid>` in place of the devkit's random worker ids (section 7).
5. **KE4** is computed automatically by `summarize` for every arm that has a whole-log smoke (`R7_A1` at step 5,000, `STOP`, `HUMAN`, `CV`).
6. **Record of the smoke controls (VALIDATION ONLY, never a result):** `raw/navtest_epdms/controls/CONTROLS_SMOKE.json`, recomputed from the
   banked smoke artifacts by `score_navtest_epdms7.py controls`.

*(Further amendments: each must be dated, written before the scores it touches, and carry its own hash line.)*
