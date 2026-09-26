# RESULT — W8: `navsim_v2 --split navtest_single_stage` (NAVSIM v2 EPDMS, ONE stage, navtest)

**EvalFlyWheel W8, 2026-09-26. PI (chat): *"wire the navtest single-stage split."*** Pre-registration:
`PREREG.md` (sha256 chain in `raw/PREREG_HASH.txt`: original 15:33, A1 15:53, A2 16:26 Berlin — each written
before the data it governs existed).

## 0. Headline

| | status | evidence |
|---|---|---|
| **The split RUNS** | ✅ **WIRED** — `python -m taniteval.bench navsim_v2 --ckpt none --split navtest_single_stage` dispatches to the devkit's **ONE-STAGE runner**, the `:228` refusal is lifted **for this split only** (unknown splits still REFUSE), and a subset run went **COMPLETE end-to-end** (preflight → export → STOP seam → 3 arms scored → artifacts → criteria → schema-valid summary) | MEASURED: `raw/smoke218_run/` (run `_scratch/…/20260926T135821Z-navsim_v2-none-ce7ac0`, 218/218 valid × 3 arms) |
| **Tests** | ✅ **35 new** in `taniteval/tests/test_bench_navsim_navtest_single_stage.py` (34 pass + 1 skip = the full-split live dry-run, skipped with `NO_TREE` until the 12,146 cache exists); related bench/leaderboard/report/NavSim suites **467 passed / 10 skipped** after the last change; the **FULL taniteval suite: 1,888 passed / 34 skipped / 0 failed** (670 s, finished ~16:48 Berlin — measured BEFORE the `--reuse-scored-arms` addition, which only the 20 related suites re-verified; `stack/` untouched by this change) | pytest (tanitad venv), 2026-09-26 |
| **Smoke validation vs the independent v1.1 run** | ⚠️ **23 / 24 pre-registered checks PASS; C-NC[CV] FAILED as pre-registered (1 / 218)** — then **diagnosed to an INPUT difference and closed by a discriminating experiment** (§2): the v2 NC/DAC **function** reproduces W3's v1.1 values **exactly on 654/654 token×arm pairs** when fed v1's inputs | `raw/cross_protocol_smoke218.json`, `raw/counterfactual_smoke218_{CV,STOP,HUMAN}.json`, `raw/diag_937ca624/` |
| **Full split (12,146 tokens)** | ⛔ **NOT SCORED — BLOCKED ON THE SHARED BOX**, by design: (1) the 12,146-token v2 **metric cache** is building RAM-gated (4 of 34 groups, 1,153 pickles at 16:39) and has been **waiting since ~16:22 with 3.6–8.5 GB available < the 9 GB floor** the brief set (other sessions: refcv6 battery `reproduce_inrun_eval.py` 5.0 GB, a `train_research.py` 3.3 GB, a NavSim job 1.8 GB); (2) the brief schedules the full floors **after the refcv6 FINAL** (~19:30). An **agent-free waiter is running** that does exactly that, then runs every pre-registered check (§5) | `raw/full_waiter.log` (`ZZW8…ZZ` markers), `raw/cache_full.driver.log` |

## 1. What was wired (all in the suite, built ON the two-stage code — the two-stage path is unchanged)

* **`profiles.py`** — `SPLITS["navtest_single_stage"]`: protocol `EPDMS_v2_navtest_single_stage`, **stages 1**,
  12,146 tokens / 0 / 136 logs, devkit split `navtest`, **logs from D:** (`D:/Archive/devbox-C/navsim/data/openscene/navsim_logs/test`, all 136 present) — ⛔ the C: dir holds navhard's 76 and `dataloader.py:34-36` skips an absent log **silently**; **`traffic_agents=non_reactive`** (the runner's own default, `default_common.yaml:22`, passed explicitly); cache `C:/Users/Admin/navsim-crun/exp/metric_cache_navtest_v2` verified by `CACHE_DONE.json`; **no synthetic fields**. `runner_for()` / `check_runner()` bind stage count → runner (`pdm_score_one_stage` / `pdm_score`) and REFUSE a mismatch; `read_single_stage_yaml()` refuses a filter with two-stage content; `token_to_log()` (W2's map, verified == the yaml); `verify_cache(expected=…)` for subsets; `preflight(tokens=…)` checks every needed log on the NAMED dir.
* **`scoring.py`** — `overrides_for()` (single-stage: `navsim_log_path`, `traffic_agents`, no `synthetic_*`, optional subset); the launched `--script` comes from `runner_for`; the count guard excludes the PROTOCOL's declared summary rows and REFUSES a CSV carrying another protocol's rows.
* **`single_stage.py`** (new) — the one-stage run: arms CV + STOP (mandatory) + HUMAN (a legal REFERENCE here — every token is an original frame; still refused on two-stage splits), headline = the devkit's **`average_all_frames`** `score` read as TEXT, `per_stage` = ONE block, controls C4 (formula) + C_AVG (average row == skipna token mean) + counts, interval = W2's **`navsim_log_cluster_bootstrap`** with the SINGLE-STAGE aggregation (`single_stage_token_mean`) over `log_name` clusters, paired deltas + paired log-cluster bootstrap vs STOP and CV, four families on **all** tokens (every navtest token has a logged human future). Model arms / `--ckpt` REFUSED with the unblock named (§6).
* **`artifacts.py`** — `build_arm_artifact_single_stage()` (same adapter + E1 gate evaluator + mutations; `protocol.harness_fix151 = post`).
* **`export.py` + `devkit_side/export_agent_inputs.py`** — a single-stage export mode in **marked `# --- W1 ADDITION` blocks**: stripping them reproduces E2's origin **byte-for-byte** (pinned; `PROVENANCE.json` updated). A subset export is keyed by its token sha256 and never reused for the full split.
* **`cache_build.py`** (new) + **`devkit_side/write_cache_metadata.py`** (new) — W3's per-log-group, resumable, RAM-gated cache recipe promoted onto v2: `Win32_PerfFormattedData_PerfOS_Memory.AvailableMBytes` ≥ 9,000 MB for 5 consecutive samples before a group starts, the wrapper's own guard at the same floor so a running group **YIELDS** (rc 3, retried, not counted as a failure); the metadata CSV written by the devkit's own `save_cache_metadata`, verified on CONTENT (rows, files, patched-loader parse, **every entry lzma-decompressed** — a yield can truncate a pickle — sampled unpickle for scene type / log / token / human trajectory), plus `CACHE_MANIFEST.json` (per-file sha256) and `CACHE_DONE.json`.
* **`cli.py`** — `--tokens-file` (single-stage subset; **forces `_scratch/`**, never claim-bearing: EC pairs adjacent scored tokens, so a subset's per-token score is not the full split's), `--metric-cache` (subset runs only) and **`--reuse-scored-arms RUN_DIR`** — an arm that PASSED in a previous attempt is adopted instead of re-scored, after an identity check that REFUSES on any mismatch (runner, traffic policy, devkit sha, patch set by raw blob, cache path + CACHE_DONE token/manifest sha256, token set by value, export sha256, STOP seam byte-equal; 1 positive control + 4 mutations pinned). It exists because RAM-guard aborts killed two navhard attempts at ~80 % of an arm (W7) and this box is at 3.6–11 GB available today.

**Stamps (every run):** tier **T1-family**; loop `single_stage: OPEN (PI ruling 2026-09-02)`; background traffic **NON-REACTIVE log replay** (`stamps.reactive_background_traffic = false`); ego non-reactive.

## 2. The smoke — numbers (all MEASURED; a SUBSET, `_scratch`, **not claim-bearing**)

`smoke218` = every token of the first 3 navtest logs (yaml order; A1). Devkit `0a380a9` (post-#151), one-stage
runner, non-reactive. EPDMS ×100 from the `average_all_frames` row:

| arm | EPDMS | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **HUMAN** (privileged reference) | **95.13** | 1.0 | 1.0 | 1.0 | 1.0 | 0.914 | 1.0 | 1.0 | 0.973 | 0.832 | 57.1 s |
| **STOP** (floor) | **57.98** | 0.977 | 0.950 | 1.0 | 1.0 | 0.380 | 0.927 | 0.954 | 0.587 | 0.130 | 59.4 s |
| **CV** (floor) | **26.40** | 0.667 | 0.560 | 0.819 | 1.0 | 0.863 | 0.638 | 0.904 | 0.986 | 0.462 | 56.8 s |

* n = 218 tokens / 3 logs per arm; 34 tokens per arm have no adjacent earlier frame (EC NaN → /14).
* **STOP > CV** (the directional prediction holds): paired Δ **+31.58** points, per-token W/T/L 141 / 23 / 54.
* **Interval: REFUSED** on the smoke — 3 `log_name` clusters < the RG-14 floor of 8 (estimator `navsim_log_cluster_bootstrap`); the full split has 136.
* **Four families** (OUR instruments, all 218 tokens, agent vs the logged human future):
  LONGITUDINAL **PARTIAL** (distance-keeping UNAVAILABLE: no lead block) — CV speed MAE 1.366 m/s, along MAE 2.355 m; STOP 5.131 / 11.416;
  LATERAL CV **OK** (cross 1.277 m, heading 8.48°, curvature 0.018 1/m, yaw-rate 4.57°/s), STOP **PARTIAL** (heading/curvature/yaw-rate undefined for a stationary plan — refused with counts);
  TACTICAL **OK** (CV lateral-decision acc 0.61 / κ 0.0, longitudinal 0.321 / κ 0.0, goal-point error 7.67 m);
  STRATEGIC **UNAVAILABLE** (reason below — ⚠ escalation E3). **HUMAN reads exactly 0.0 error on every geometry term** — the control that must read a known value.
* `tools/criteria_check.py`: **rc 0, 0 violations** on all 3 artifacts; E1's four NavSim gates PASS/declared; their deliberate mutations all RED.
* **HUMAN 95.13 is NOT comparable to the published 90.3** (that row is `fix151: pre`, basis INFERRED — and a 218-token subset besides).

### Pre-registered checks on the smoke (`raw/cross_protocol_smoke218.json`)

| check | CV | STOP | HUMAN |
|---|---|---|---|
| C-COUNT (218 valid, 0 failed, 1 summary row) | ✅ | ✅ | ✅ |
| C-FORMULA (max \|Δ\|) | ✅ 1.1e-16 | ✅ 1.1e-16 | ✅ 2.2e-16 |
| C-AVG (\|Δ\|) | ✅ 0.0 | ✅ 0.0 | ✅ 0.0 |
| **C-DAC** exact | ✅ 0/218 | ✅ 0/218 | ✅ 0/218 |
| **C-NC** exact (E-T0 = 0 tokens) | ⛔ **FAILED 1/218** | ✅ 0/218 | ✅ 0/218 |
| C-TTC v2 ≥ v1 | ✅ 0 | ✅ 0 | ✅ 0 |
| C-DDC v2 ≥ v1 | ✅ 0 | ✅ 0 | ✅ 0 |
| C-HUMF (NC=DAC=TTC=TLC=LK=1, DDC∈{0.5,1}) | — | — | ✅ 0 exceptions |
| K-NEG (v2 CV vs v1 STOP, DAC) / K-MUT (1 flipped cell) | ✅ 85 ≥ 1 / ✅ exactly 1 | | |

A second smoke sample — W3's random `sub200`, whose one-stage AGGREGATION crashed (§4) — was checked on its
PRE-AGGREGATION per-token rows: **C-DAC / C-NC / C-TTC / C-DDC / C-HUMF all 0 violations on 200 × 3**
(`raw/cross_protocol_sub200_preagg.json`).

EP (normalisation changed; descriptive only): CV v2 0.863 vs v1 0.217 — the v1 masked-EP effect E1 predicted; STOP 0.380 vs 0.351; HUMAN 0.914 vs 0.916.

### The C-NC failure — diagnosed, then closed (RULE ZERO)

Token `937ca624cc2658a6`, CV: v1.1 NC **0.5** (a static-object collision), v2 NC **1.0**.
1. **Both caches' observations** (`code/diag_observation_objects.py`, `raw/diag_937ca624/`): v1 holds **5 objects at all 41 steps with one constant footprint** (1 VEHICLE, 4 GENERIC_OBJECT) that v2 holds at none; common objects' footprints are identical (max |Δ| 0.0 m on a second token).
2. **The log** (independent source, `ghost_objects_log_check.json`): each of the 5 is observed **only at +5.0 s**. Mechanism: the metric cache's observation window is **5.0 s in v1.1 and 4.0 s in v2** (v1 `metric_cache_processor.py:101` vs v2 `:109`), and an object observed exactly once in its window is placed at EVERY step (`StateInterpolator` start == end; v1 `:165-166`, v2 `:177-178`) — a **ghost**. CV's straight 36 m path meets a GENERIC_OBJECT ghost → NC 0.5. ⛔ **PREREG §2 missed this input difference.**
3. **The discriminating experiment** (`code/counterfactual_v2fn_v1inputs.py`): each arm's own plan re-scored by the **v2 function on the v1.1 cache's inputs** reproduces W3's v1.1 **NC and DAC exactly on 218/218 for CV, STOP and HUMAN** (TTC never below v1); the **control** (v2 function on v2 inputs) reproduces our CSV **exactly** (the re-scoring path is the scorer's). On the failing token: v1 inputs → **0.5**, v2 inputs → **1.0**. ⇒ **the NC and DAC functions are unchanged; the difference is the input.**
4. Ghosts run **both ways** (an object seen once inside 0–4 s but again at +4.5/+5.0 s is a v2 ghost), so NC/TTC are two-sided wherever observations differ — per-token classes on smoke218: IDENTICAL 9 / V1_SUPERSET 28 / V2_EXTRA 181 (cross-tree read control 418/418 identical). **AMENDMENT A2** pre-registers class-aware NC/TTC rules + the function-identity test for the full split; the smoke's C-NC stays **FAILED as recorded**.

## 3. Projection for the full split (ESTIMATED by linear scaling of the MEASURED smoke)

| step | measured on the smoke | full split (12,146 tokens) |
|---|---|---|
| metric cache | 0.733 s/scene alone (sub200), 1.04 s/scene under contention (full build, group 0: 325 in 339 s) | **2.5–3.5 h of compute at 1 worker**, plus RAM-gate waits (currently open-ended) |
| scoring, per arm | 56.8–59.4 s for 218 tokens; per-token loop 0.219–0.228 s; `pdm_score` 0.102–0.103 s; peak RSS 484–515 MB | **≈ 45–50 min per arm** (12,146 × 0.22 s + log loading + aggregation); **≈ 2.3–2.5 h for CV+STOP+HUMAN** sequential; peak RSS ESTIMATED ~2–2.5 GB (W3's v1 full-split analog: 2.2 GB) |
| post-checks (A2) | pipeline test on the smoke: ~6 min | E-T0 ~25 min, v1 digest ~35 min, classification ~100 min, counterfactual (discrepancy ∪ every 25th ≈ 500+ tokens) ~10 min/arm |

## 4. Harness defects found and fixed on the way (each MEASURED, each pinned)

1. **The C: log trap** (the brief's): `preflight` hard-coded C:'s `navsim_logs/test` (76 logs). Now per-profile `logs_dir` + every needed log checked — pinned by `test_the_c_log_dir_would_silently_score_a_subset…`.
2. **One-stage aggregation crashes on a subset with no adjacent tokens** (devkit, `run_pdm_score_one_stage.py:222`, `pd.concat([])`): W3's random sub200 has no adjacent pair. Not a full-split risk (0.5 s frames); smoke switched to whole logs (A1).
3. **`scoring.py` counted pre-aggregation LINES, not records**: the dump's `ego_simulated_states` cells span lines (24,600 lines for 200 records), so AGGREGATION_FAILED never fired and a fully-scored arm read as FAIL. Now csv records — pinned by `test_a_multiline_preaggregation_dump_is_counted_in_records_not_lines`.
4. **`criteria_check.py` crashed under a cp1252 parent** (`⛔` at print → rc 1, NO json): `contract.run_criteria` now runs it with `PYTHONUTF8=1`. (Exposed by the existing offline twin failing in this shell.)
5. **W5's report renderer requires string `stamps.loop` values** (`benchreport/render.py:160`); the smoke's first report failed on a bool. Loop values are strings now; the flag lives beside it (`stamps.reactive_background_traffic`). Re-render on a patched copy: `verify: PASS (133 numbers, 0 errors, 0 coverage gaps)`.

## 5. What runs next without me (agent-free) — and what to read

* **`code/navtest_full_after_final.py`** (PID 2052, relaunched 16:54 with the start gate + reuse below; log `raw/full_waiter.log`; the first instance's log is `raw/full_waiter.attempt1_superseded.log`): waits until **(a)** `metric_cache_navtest_v2/CACHE_DONE.json` certifies the yaml token set, **(b)** `ZZFINALV2ENDZZ` follows the latest `ZZFINALV2STARTZZ` in `C:/Users/Admin/ev6_battery/raw/final_v2.log`, **(c)** no `run_battery.py` is alive; then a sustained window of **≥ 11.5 GB available** (the 9 GB floor + the scorer's own ESTIMATED ~2.5 GB peak — the wrapper's guard counts our footprint too, so starting AT the floor would yield on itself); then the full CLI run (CV, STOP, HUMAN; wrapper floor 9 GB; up to 6 attempts, each retry adopting the arms that already PASSED via `--reuse-scored-arms`) and the pre-registered post-checks → **`raw/verdict_full.json`** (`validated_under_A2`, plus the original C-NC/C-TTC reported beside it). Its post-pipeline was tested end-to-end on the smoke (`raw/verdict_smoke218_pipelinetest.json`).
* **The cache builder** (`raw/cache_full.driver.log`) — resumable: re-running the same command skips finished groups.
* ⚠ If either process dies with this session, relaunch (cwd `D:/Projects/TanitAD/taniteval`):
  `python -m taniteval.bench.navsim.cache_build --split navtest_single_stage --out ../FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-navtest-single-stage/raw/cache_full --label navtest_v2 --logs-per-process 4 --min-avail-mb 9000 --load sample`
  and `python ../FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-navtest-single-stage/code/navtest_full_after_final.py`.

## 6. Escalations

* **E1 → W4 (leaderboard)** — the `EPDMS_v2_navtest_single_stage` column **mixes 8 `pre` / 6 `unverified` / 4 `post` rows**; our number is **post-#151** (`summary.provenance.harness.fix151 = "post"`, caveat `FIX151_POST_ONLY_COMPARABLE` on every arm) and is comparable **only to the 4 post rows** (DiffusionDrive 84.5, DDv2 87.5, Latent-WAM 89.3, DriveFuture 89.9) — and even those only if they were scored with **non-reactive** traffic, which **no banked primary states** (searched 2605.09701, 2512.07745, 2604.03581, 2506.04218). `Human (logged) 90.3` is pre-fix/INFERRED and **not a target**. The renderer does not read `provenance.harness.fix151` for OUR rows today — they must land in the post table. The 0E.4 "TanitAD — any arm: NOT TARGETED" line needs revisiting once the full-split floors exist.
* **E2 → Master Mind** — the full split is blocked on the shared box (§0); nothing to decide unless the 9 GB floor should be relaxed for a 0.5 GB process (the brief set it; I did not move it). Single committer: see `LANDING_READY.txt`.
* **E3 → W2 (adapter)** — the STRATEGIC refusal text on NavSim artifacts reads *"PhysicalAI-AV carries NO map, NO lane graph …"* — true of PhysicalAI, **wrong for NavSim** (which has maps and a route); a NavSim-specific reason is needed (true-but-wrong-for-the-reader class). Pre-existing, not introduced here.
* **E4 → whoever wires model arms on navtest** — refused here with the unblock: a `model_arms.DEFAULT_BANKS['navtest_single_stage']` entry + a stage-1-only join of W3's 32-shard bank (`D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/frame_bank`) through the suite's bridge.
* **E6 → Master Mind (register)** — `REGISTER_ROWS.md` carries 5 rows + 1 retraction candidate for `GOALS_AND_CLAIMS.md` / `RETRACTION_LOG.md`. ⛔ Not edited in place: this worktree's register blob (`740e4a59…`) is OLDER than the tip's (`baeaf548…`), and editing the stale base would revert the tip's newer rows on commit.
* **E5 → EvalFlyWheel** — **PREREG §2 was wrong about NC's inputs** (the 5.0 → 4.0 s observation window + single-observation ghosts). Retraction-log candidate, class: *"identical function text read as identical metric — an input pipeline changed underneath it."*

## 7. Deliverable manifest

| artifact | where |
|---|---|
| wiring (profiles, benchmark dispatch, scoring, export, artifacts, plans, cli, contract) | `repo:taniteval/taniteval/bench/{navsim/profiles.py, navsim/benchmark.py, navsim/scoring.py, navsim/export.py, navsim/artifacts.py, navsim/plans.py, cli.py, contract.py}` (staged) |
| single-stage run + cache builder | `repo:taniteval/taniteval/bench/navsim/{single_stage.py, cache_build.py}` (staged) |
| devkit-side | `repo:taniteval/taniteval/bench/navsim/devkit_side/{export_agent_inputs.py, write_cache_metadata.py, PROVENANCE.json}` (staged) |
| tests | `repo:taniteval/tests/test_bench_navsim_navtest_single_stage.py` (new), `repo:taniteval/tests/test_bench_suite_navsim_offline.py` (literal file list +1) (staged) |
| PREREG + hash chain, RESULT, REGISTER_ROWS, LANDING_READY | `repo:FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-26-navtest-single-stage/` (staged) |
| analysis code | `repo:…/code/{cross_protocol_check.py, e_t0_from_cache.py, e_obs_digest.py, e_obs_classify.py, counterfactual_v2fn_v1inputs.py, diag_observation_objects.py, navtest_full_after_final.py}` (staged) |
| smoke evidence | `repo:…/raw/` (smoke token sets, cache-build records, both smoke runs' records, fixtures, E-T0/E-OBS/counterfactual/cross-check JSONs) (staged) |
| smoke metric caches (200, 218 tokens) | `C:/Users/Admin/navsim-crun/exp/metric_cache_navtest_v2_smoke{200,218}` (+ `CACHE_DONE.json`, `CACHE_MANIFEST.json`) — **one place, off-repo, rebuildable in ~4 min each** |
| the 12,146-token cache (IN PROGRESS) | `C:/Users/Admin/navsim-crun/exp/metric_cache_navtest_v2` — **one place, off-repo**; rebuildable (§5) |
| smoke run directories | `D:/Projects/TanitAD/taniteval/results/bench/_scratch/navsim_v2/navtest_single_stage/{20260926T135050Z-…-2e5ea3 (sub200, FAILED in aggregation), 20260926T135821Z-…-ce7ac0 (smoke218, COMPLETE)}` — **worktree only, untracked** (`_scratch`, consumers skip it); their load-bearing records are copied into `raw/smoke200_run/`, `raw/smoke218_run/`, `raw/smoke218_fixtures/` |
| exports | `C:/Users/Admin/navsim-crun/exp/tanitad_bench/exports/navtest_single_stage__subset_*` — off-repo, rebuilt on demand (content-keyed) |
