# RESULT — E1: the programme's first locally computed EPDMS (NavSim v2, `warmup_two_stage`)

**Harness:** `autonomousvision/navsim@0a380a9063d7162ec93d0f51e9990ebac585f720` (2025-10-27) — on the
**post-#151** side (MEASURED, two probes: README changelog `[2025/09/29]` + the fix's code at
`navsim/evaluate/pdm_score.py:194-219`) — plus three PRE-EXISTING local modifications and one E1
reader patch (§4). **Every number below: tier T1-family; loop OPEN for stage 1, UNRULED for stage 2
(2026-09-02 vocabulary); background vehicles IDM-reactive in both stages.** Intervals: **warmup
NEVER carries one** (D-BENCH-PORT: 7 log groups < the RG-14 floor of 8; the 7 re-MEASURED);
**navhard does** (76 log clusters) under W2's settled log-cluster bootstrap — §5b.

## 1. Findings

0. ⭐⭐ **navhard: our CV EPDMS is 11.4816 — the official leaderboard's `baseline_constant_velocity`
   value to the digit (Δ 0.0000), with 5,912/5,912 scenes valid — and all 19 published terms are
   reproduced exactly at the paper's printed precision under truncation.** First NavSim number of
   ours with an interval: **11.48 [8.25, 14.50]** (§5b). Getting there required diagnosing and
   patching a devkit assert that otherwise kills the whole run (§5b), and the run before it was lost
   to my own RAM guard — both are written up rather than smoothed over.
1. ⭐ **The harness reproduces the official HF warmup leaderboard.** Constant velocity, official
   two-stage runner: **combined EPDMS 0.1853562745 → 18.535627 ×100** vs the leaderboard's
   `baseline_constant_velocity` **18.5356** (INHERITED, `NAVSIM_PROTOCOL.md` §6.3, retrieved
   2026-08-23): **Δ +0.000027**, i.e. identical at the leaderboard's printed precision.
   MEASURED `raw/A1/devkit_2026.09.19.12.24.20.csv`; control C8 `raw/controls.json`.
2. **Every control derivable from source reads its known value, except one pre-registered
   strictness (C7).** C1 human-filter identity PASS and **not vacuous** (the filter fired on 3/16
   tokens); C2, C3, C4 (5 CSVs, 488 rows), C5, C6 PASS. C7 **FAILED as written** (exact 0.0 was
   pre-registered; observed 1.1e-16 in 4 summary cells — float summation order; all 220 per-token
   rows bit-identical). Table §2.
3. ⛔ **The human agent is UNDEFINED on any two-stage split — the official runner crashes.**
   MEASURED: all 204 synthetic scenes carry `num_future_frames = 0` (4 frames); `HumanAgent` indexes
   `frames[3..11]` → **204/204** stage-2 `IndexError` (`dataclasses.py:371`) →
   `AssertionError: Invalid interval nan` (`scene_aggregator.py:62`, caught) → **uncaught**
   `TypeError: Could not convert [array([nan…])…] to numeric` → exit 1, **no CSV**
   (`raw/A2/A2.log`). The human is a **stage-1-only** reference; its combined EPDMS is reported
   UNDEFINED, never imputed. (Consistent with [N2] v3 Tab. 2, which has no Human row.)
4. ⚠️ **Three premises of the brief are wrong, from source + measurement:**
   * background traffic is **IDM-reactive in both stages** (`run_pdm_score.py:77-79, 132-134` →
     `navsim_IDM_traffic_agents.yaml`), not "non-reactive logged agents"; the ego is still non-reactive;
   * the human reference exists on **stage 1 only** (finding 3);
   * *"CV must score EP well below the human"* does not hold for v2 EP: **0.8445 vs 0.8564**
     (Δ 0.0119, stage 1, same runner). v2 normalises the UNMASKED progress against PDM-Closed; the
     large CV/human EP gap in [N1] Tab. 1 (19.4 vs 87.5) is a v1 masked-EP effect (`NAVSIM_PROTOCOL.md` §2.3).
5. **Windows defect in the devkit reader, MEASURED:** the unpatched `MetricCacheLoader` raises
   `IndexError` on this cache (`dataloader.py:316` splits on `/`; nuPlan writes backslash paths).
   Fixed by a separator-only patch (unit-tested 6/6, end-to-end 220/220 tokens). **Any consumer of
   this cache on Windows (E2) needs it.**
6. **The TanitEval artifacts pass `criteria_check.py` with 0 violations** (T1, IN_SCOPE, 8 work
   items each, every one a reasoned refusal) — but ⛔ **`criteria_check.py` never evaluates
   `benchmarks.navsim`**; its four BLOCKING gates are checked only by E1's evaluator (all PASS or
   declared N/A; 5/5 deliberate mutations go RED). §5.

## 2. Numbers (all MEASURED; EPDMS read from `score`, never `pdm_score`)

**Constant velocity — OFFICIAL two-stage runner (A1), 220/220 valid**

| | EPDMS | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stage 1 | **0.460289** | 0.71875 | 0.625 | 0.90625 | 1.0 | 0.844523 | 0.6875 | 1.0 | 1.0 | 0.875 | 16 |
| stage 2 | **0.334129** | 0.675315 | 0.638856 | 0.804840 | 0.875 | 0.759839 | 0.675315 | 0.614656 | 0.995833 | 0.720810 | 204 |
| **combined** | **0.185356** | | | | | | | | | | 8 groups |

**Stage 1 on the one-stage runner** (`run_pdm_score_one_stage.py`, `traffic_agents=reactive`, n = 16;
its 8 "prev" tokens have no adjacent earlier frame, so EC's weight is dropped (/14) — predicted in
SPEC C6, which is why CV reads 0.465343 here vs 0.460289 above):

| arm | EPDMS | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
|---|---|---|---|---|---|---|---|---|---|---|
| CV (A1b) | 0.465343 | 0.71875 | 0.625 | 0.90625 | 1.0 | 0.844523 | 0.6875 | 1.0 | 1.0 | 0.875 |
| **human, filter ON (A2b)** | **0.951255** | 1.0 | 1.0 | 1.0 | 1.0 | 0.856415 | 1.0 | 1.0 | 1.0 | 1.0 |
| human, filter OFF (M1, mutation) | 0.872014 | 1.0 | 0.9375 | 1.0 | 1.0 | 0.856415 | 1.0 | 0.875 | 1.0 | 1.0 |

**Human, two-stage (A2): UNDEFINED** — runner exit 1, no CSV (finding 3).
Pre-registered orderings: E1 CV 0.465 < human 0.951 ✅; E2 EP 0.8445 < 0.8564 ✅ (small); E3 ✅.

**Controls** (`raw/controls.json`, `code/verify_controls.py`; every expectation derived independently of the devkit code that produced the value):

| id | what must hold | result |
|---|---|---|
| C1 | human, stage 1: DAC=TLC=TTC=LK=1.0 exactly; NC, DDC ∈ {0.5, 1} | **PASS** 16/16. P1 precondition: agent vs cache human trajectory max \|Δpose\| **1.24e-9**. Teeth (M1): raw human DAC=0 on 1 token, LK=0 on 2 → filter fired on **3/16**; every changed cell is exactly 0→1 (0 unexplained) |
| C2 | pickled `scene_type` matches yaml membership | **PASS** 220/220 (16 ORIGINAL + 204 SYNTHETIC) |
| C3 | cache = CSV = yaml token sets; scorer 16 + 204 valid + 3 summaries | **PASS** |
| C4 | per-token `score` = NC·DAC·DDC·TLC·(5EP+5TTC+2LK+2HC+2EC)/16 (docs/metrics.md) | **PASS** A1, R1 (220 rows each), A1b, A2b, M1 (16 each; 8 with EC dropped /14); max \|Δ\| 1.1e-16 |
| C5 | summaries recomputed from rows + Gaussian weights (σ²=0.1) recomputed from endpoints | **PASS** max \|Δ\| 1.1e-16 (weights and all 3 summaries) |
| C6 | A1 stage 1 vs A1b: 8 metrics bit-identical; EC NaN only on "prev" tokens | **PASS** (0.0; 8 NaN-EC tokens = the 8 prev tokens) |
| C7 | rerun (hash seed 2 vs 1, evaluation order verified to differ): max \|Δ\| = 0.0 | ⛔ **FAIL as pre-registered**: 4 summary cells differ by **1.1e-16** (DAC/EP stage-2 means in the stage_two and combined rows); **0 of 220 per-token rows differ**; the three EPDMS scores are identical. Mechanism: row order follows `list(set(...))`, and the weighted means sum in that order (`raw/C7_diagnosis.json`). Not relabelled post hoc. |
| C8 | local CV combined ×100 vs HF warmup LB 18.5356 (INHERITED) | **reproduced**, Δ +0.000027 |

## 3. Cost, and the navhard price (ESTIMATED)

MEASURED (1 process, sequential worker, C: runtime, cache written to the D:-backed canonical path):
metric cache **551.8 s** wall for 220 scenes, peak RSS **382.5 MB**, per scene **2.43 s** (S1 mean) /
**2.30 s** (S2 mean); first scene 4.3 s (map load). CV two-stage scoring **534.6 s** for 220 tokens,
**2.21 s/token** mean (incl. loading), peak RSS **508 MB**. (`raw/navhard_price.json`)

**navhard_two_stage** (counts MEASURED from the local yaml, two derivations agree: **450 S1 + 5,462
S2, 76 logs**) — ESTIMATED by linear scaling: cache **3.80 h**, CV scoring **3.63 h**, human stage 1
≈ 0.35 h ⇒ **≈ 7.8 CPU-h sequential; ≈ 3.9 h wall at 2 pooled workers** (RAM, not cores, is the
limit: ~0.4–0.5 GB/process against ~1–2 GB of headroom above the 3 GB floor). Excludes downloads,
extraction, copies, imports, map loads.

## 4. The harness column (what "0a380a9 + patches" means here)

⚠️ **Every blob id below carries its HASH SPACE, because this package contains both and they differ
on a CRLF file.** `git hash-object` (and anything run inside a repo) NORMALISES CRLF→LF; raw-byte
hashing does not. The normalising id is therefore **blind to a pure line-ending change**, so for
"did this file change at all" the raw-byte id is the stronger probe. Mixing the two inside one record
produces a guaranteed false "mismatch" for the next reader (W1 found exactly that in the promoted
copy of this table on 2026-09-20).

* PRE-EXISTING (found, not made, by E1): `dataclasses.py` PosixPath unpickler
  (**git-normalised** blob `596cb7d`; raw-byte id differs — both are correct, they answer different
  questions), the venv `fcntl.py` flock shim (**sha256** `75184a48…`), nuPlan `setup.py` (packaging only).
* E1 patch 1 — the `MetricCacheLoader` separator fix: in-process monkeypatch for all warmup scoring
  (`code/navsim_win.py`), and the same fix on the **C: copy** for process-pool children
  (`code/patches/apply_dataloader_patch.py`; **raw-byte** blobs `0a62e6a0` → `b850e162`). ⚠️ Its
  first application was silently wrong (a heredoc ate one backslash: the regex matched only `/`);
  the end-to-end test against the real cache caught it before any run used it.
* E1 patch 2 — the IDM degenerate-path assert (§5b), C: copy only
  (`code/patches/apply_idm_degenerate_path_patch.py`; **raw-byte** blobs `1924b7c1` → `b95bcc7f`;
  the same file reads `214ef5ee` under `git hash-object` because it is 100 % CRLF).
* **Runtime deviation:** a byte-verified C: mirror, because the D:-resident venv was I/O-starved by
  the concurrent navhard download (`import numpy` 23,350 ms vs 324 ms). Verification: PLAN.md
  Deviation 1, `raw/devkit_copy_verify.json`, `raw/data_mirror_verify_*.json`.

## 5. Artifacts and gates

`raw/artifacts/A1_constant_velocity_agent_warmup_two_stage.json`,
`raw/artifacts/A2_human_agent_warmup_two_stage.json` — built with `taniteval/adapters/navsim.py`
(unmodified) + gap fills; protocol `EPDMS_v2_warmup_two_stage`, devkit SHA, estimator
`{UNAVAILABLE, reason, n}` + `cluster_unit` UNAVAILABLE, modality/setting, tier T1, loop stamps.

| check | A1 (CV) | A2 (human) |
|---|---|---|
| `tools/criteria_check.py` | IN_SCOPE, T1, **0 violations**, 8 work items | same |
| navsim.score / variant / submetrics / ego_inputs | PRESENT ×4 | score REFUSED (undefined two-stage), rest PRESENT |
| navsim.route_leak_check | REFUSED (arm consumes no route input) | same |
| GATE estimator_unit · modality_label · cross_protocol | **PASS ×3** | **PASS ×3** |
| GATE ego_enforcement | NOT_APPLICABLE_DECLARED (not vision-only, not claimed) | same |
| E1 gate evaluator mutations (5) | **all RED** | all RED |

**Four families** (OUR instruments, not NavSim's; CV's 16 stage-1 plans vs the logged human future,
dt 0.5 s, K = 8, frame check verified; MEASURED, T1-family, loop OPEN, no CI, n = 16 windows):
LONGITUDINAL speed MAE **0.893 m/s** (bias +0.381), along-track MAE **1.267 m** (bias +0.939),
progress ratio 1.064; distance-keeping UNAVAILABLE (no lead block — work item). LATERAL cross-track
MAE **1.066 m**, heading **7.48°**, curvature **0.0155 1/m**, yaw rate **4.47 °/s**. TACTICAL
trajectory-derived lateral/longitudinal decisions computed (executed-vs-executed, NOT
selected-vs-executed). STRATEGIC UNAVAILABLE (NavSim has a map but never scores the route). Stage 2 has
no human future, so no geometry family exists there. The human artifact's families are trivially exact
(pred = GT) and are a reference, not a result.

## 5b. navhard_two_stage (priority 2) — the blocker, its cause, and the patch

**Cache** (built overnight by the orchestrator's 1-worker C:-only variant of `code/run_navhard.sh`,
5,187 s): content-verified by E1 on 2026-09-20 — **5,912 entries = 450 ORIGINAL + 5,462 SYNTHETIC**,
token sets equal the yaml, 0 scene-type mismatches ⇒ `CACHE_DONE.json` written
(`raw/navhard/cache/verify_cache_navhard.json`).

**The blocker (N1 try 1, rc 1 after 4,144 s, NO CSV).** MEASURED chain:
`AssertionError: Agent's baseline does not intersect the agent itself`
(`navsim_idm_agent_manager.py:84`) on **166 of 5,462 stage-2 tokens (3.04 %), 0 of 450 stage-1** →
each becomes an invalid all-NaN row → `AssertionError: Invalid interval nan`
(`scene_aggregator.py:62`) → caught at `run_pdm_score.py:382` → **uncaught**
`TypeError: Could not convert [array([nan…])…] to numeric` → exit 1. The 68-minute rollout was
**lost**: the only dump hung off `compute_final_scores`, which never runs on that path.

**Root cause, MEASURED** (`raw/navhard/diag/idm_assert_diag.json`): the offending IDM agent has
consumed its whole baseline, so `get_path_to_go()` is a **zero-length** LineString
(`path_to_go_length_m = 0.0`, path start == agent position). Under the pinned **shapely 2.0.7** a
zero-length line buffered with `cap_style=flat` is an **EMPTY** polygon (round caps would give a
disc; `distance(box, empty) = NaN`) — so the agent can never appear in its own path's intersection
set and the assert *must* fire. ⇒ **a different trigger from the warmup human agent** (there: 0
future frames); only the downstream fragility — one invalid row kills the entire run — is shared.

**Patch** (`code/patches/apply_idm_degenerate_path_patch.py`; diff + blobs in
`raw/navhard/idm_degenerate_path_patch.{json,diff}`), applied to the **C: copy only**, never the D:
devkit: the **assert condition alone**, so an agent with nothing ahead falls through to the devkit's
own *"Free road case: no leading vehicle"* branch (which the code already selects, since an empty
intersection set makes `size > 1` False). Verified in both directions
(`raw/navhard/diag/patchcheck_COMPARE.json`): **3/3 previously-failing tokens now score; 4/4 healthy
tokens bit-identical; zero drift.** The warmup arms are unaffected — **0 IDM assertions in every
warmup log** (A1/R1/A1b/A2b/M1 and the cache build). ⛔ The patch is part of the navhard column.

**Durable fix for the class:** the wrapper now banks the per-token frame **before** the aggregation
(`create_scene_aggregators` hook → `*_PRE_AGGREGATION.csv`), so an analysis-time failure can never
again cost the rollout. ⚠️ That CSV has no `ego_simulated_states`, so it is readable but NOT
re-aggregatable; W1's suite adds a worker_map-seam dump (CSV **+** pickle) that `taniteval.bench
reaggregate` can replay. **Try 1 left neither** — VERIFIED, not assumed: no CSV and no pickle
anywhere, `pdm_score_calls: 0` (its work ran in a pool child), and the only dump then hung off
`compute_final_scores`. ⇒ nothing to recover; no hand-rolled recovery attempted.

### ⭐⭐ navhard CV: the harness reproduces the OFFICIAL LEADERBOARD EXACTLY, and carries an interval

Suite run `20260920T082848Z-navsim_v2-none-06e257` (W1's `taniteval.bench`, one worker, my C: runtime
+ both E1 patches), **CV: 5,912/5,912 successful, 0 failed, 5,338 s**:

| | EPDMS ×100 | vs published |
|---|---|---|
| **combined** | **11.4816** | leaderboard `baseline_constant_velocity` **11.4816** → **Δ 0.0000** (INHERITED, `NAVSIM_PROTOCOL.md` §6.2); paper [N2]v3 Tab. 2 prints **11.4** |
| stage 1 | 28.9588 | — |
| stage 2 | 34.2464 | — |

**95 % interval (the programme's FIRST NavSim CI): 11.48 [8.25, 14.50]**, SE 1.53 — W2's settled
**log-cluster bootstrap** (clusters = OpenScene `log_name`, 76 clusters, 225 mapping-key units,
B = 2000, seed 0), which re-derives the aggregate and **reproduces the devkit's own value exactly**
(`abs_diff 0.0`, tol 1e-9). Drive-level sensitivity (34 `nuplan_drive` clusters): [9.10, 14.71].
⚠️ It answers *"would another draw of LOGS give this score?"* only — blind to training and to
inference variance (H-ESTIM-SEED-1). `raw/navhard/interval_CV_navhard.json`.
⚠️ My first call passed a ROUNDED official value and the estimator **refused** rather than bootstrap a
different aggregate (`|diff| 8.4e-08 > 1e-9`) — the guard working; the rerun used the CSV's full
precision.

**All 19 published terms (9 sub-metrics × 2 stages + EPDMS) are reproduced EXACTLY at the paper's
printed precision under TRUNCATION (19/19; only 9/19 under rounding)** — `raw/navhard/compare_published_CV.json`:

| | NC | DAC | DDC | TLC | EP | TTC | LK | HC | EC |
|---|---|---|---|---|---|---|---|---|---|
| ours S1 | 88.89 | 42.89 | 70.67 | 99.33 | 77.53 | 87.33 | 78.67 | 97.11 | 60.44 |
| [N2]v3 S1 | 88.8 | 42.8 | 70.6 | 99.3 | 77.5 | 87.3 | 78.6 | 97.1 | 60.4 |
| ours S2 | 83.21 | 59.14 | 76.53 | 98.06 | 71.38 | 81.14 | 47.98 | 97.17 | 61.98 |
| [N2]v3 S2 | 83.2 | 59.1 | 76.5 | 98.0 | 71.3 | 81.1 | 47.9 | 97.1 | 61.9 |

⚠️ **Pre-registration reported as written:** SPEC_NAVHARD committed `|100·EPDMS − 11.4| ≤ 0.05`, which
assumes ROUNDING; that test reads **FALSE** (Δ 0.0816). The paper truncates — established here on 19
terms, and consistent with `NAVSIM_PROTOCOL.md` §6.2's note about the leaderboard rows — and against
the 4-decimal leaderboard value the agreement is **exact**. The tolerance was mis-specified, not the
result; I report both rather than move the goalpost.

**Cross-run identity:** W1's standalone `w1_stage1_check.py` on the suite CSV reads
`identical_to_e1 = True` — the 450 stage-1 rows from my *aborted* run and from the suite's completed
run agree exactly, across two different runner code paths and two processes.

**⭐ The STOP floor is the one that matters, not CV.** Same run, `STOP` seam, 5,912/5,912 valid:
**EPDMS 29.8532 ×100** (S1 56.3162 · S2 50.6846) — a car that simply stops scores **2.6×** the
constant-velocity floor, because the four multiplicative terms (NC, DAC, DDC, TLC) reward doing
nothing while only EP punishes it. ⇒ ⛔ **"beats CV" is not evidence of driving on navhard**; the
bar any TanitAD arm must clear is **STOP ≈ 29.85**, with CV 11.48 far below it. (Neither is in
[N2]v3 Table 2 — the paper publishes no stopped-agent row — so this is ours, MEASURED, and it
changes how our own future numbers should be read.)

**navhard reference table** (all MEASURED; EPDMS ×100; T1-family; S1 loop OPEN, S2 UNRULED):

| arm | S1 | S2 | combined | n valid | note |
|---|---|---|---|---|---|
| CV (official) | 28.96 | 34.25 | **11.4816 [8.25, 14.50]** | 5,912/5,912 | = leaderboard 11.4816 |
| STOP (floor seam) | 56.32 | 50.68 | **29.8532** | 5,912/5,912 | the floor that actually binds |
| human, stage 1 only (one-stage runner, filter ON) | **93.48** | undefined | undefined | 450/450 | NC=DAC=TLC=TTC=LK=**1.0 exactly**, DDC 99.89, EP 84.24, HC 97.56, EC 86.67 |

**Controls on navhard** (`raw/navhard/controls_navhard.json`): **C1 PASS on all 450 human tokens** —
and this is where its `{0.5, 1.0}` branch finally fired (DDC), which never happened on warmup's 16;
P1 max |Δpose| 1.49e-09. **C3 PASS** (450 + 5,462, all valid). **C4 PASS** (per-token EPDMS identity,
CV and human). **C8 reproduced** (Δ 8e-06 against the leaderboard's 4-dp value). ⛔ **C5 FAILED as
pre-registered**: the summary identity holds at **6.6e-12**, but my independently recomputed Gaussian
weights differ by **2.46e-09**, just over the 1e-9 tolerance I committed — float accumulation over
5,462 stage-2 rows where warmup had 204. Same class as C7: a tolerance that was right at n=220 and
too tight at n=5,912. Reported as written; not relabelled.

⛔ **A FALSE REFUSAL, found by W2 and fixed here — the failure mode I should have caught myself.**
The first navhard artifact declared `estimator.interval = {status: UNAVAILABLE, reason: "the v2.9.0
gate admits only UNAVAILABLE"}` while carrying the real interval nested under `summary_interval`, and
`cluster_unit` as a *refusal whose reason was the single token* `"log_name"`. A false refusal reads
exactly like an honest one: the checker would have filed a MEASURED interval as a work item and the
number would have stayed invisible. **Root cause: I read the registry as a fresh process reads it —
`git show HEAD:…CRITERIA_REGISTRY.json` serves 2.9.0, because the SETTLED gate (2.10.x) has lived in
the working tree STAGED AND NEVER COMMITTED.** The gate was never blocking anything; a staged gate is
in force for nobody. Re-emitted: `cluster_unit = "log_name"` (string) and the real interval promoted
into `estimator.interval` — `raw/navhard/artifacts/N1_suite_constant_velocity_agent_navhard_two_stage.json`
now reads `status: OK`, lo/hi **8.25 / 14.50 ×100**, 76 clusters, and `criteria_check` scores it
**0 violations with `GATE navsim.estimator_unit` = ok**.

### navhard stage 1 already reproduced the PUBLISHED column before the full run finished — 8/8 sub-metrics

E1's second navhard attempt (with the IDM patch) scored **0 agent failures across 5,576 scenes** —
the patch holds at scale — but **my own RAM guard aborted it at 5,576/5,912** (system available dipped
to 2,634 MB for 3 samples while this process held 828 MB). Because that run used the `sequential`
worker, the per-token hooks recorded in-process, so **all 450 stage-1 rows survive**
(`raw/navhard/N1/N1_hooks.json`) — the insurance paid off even though the run died.

The published stage-1 column is the mean over exactly those 450 tokens, so it is directly comparable
(MEASURED, `raw/navhard/stage_one_vs_published_interim.json`; PUBLISHED = [N2] arXiv **2506.04218v3**
Table 2, CV column, library key `2506.04218`):

| | NC | DAC | DDC | TLC | EP | TTC | LK | HC |
|---|---|---|---|---|---|---|---|---|
| **ours ×100** (n=450) | 88.89 | 42.89 | 70.67 | 99.33 | 77.53 | 87.33 | 78.67 | 97.11 |
| published (S1) | 88.8 | 42.8 | 70.6 | 99.3 | 77.5 | 87.3 | 78.6 | 97.1 |
| Δ | +0.09 | +0.09 | +0.07 | +0.03 | +0.03 | +0.03 | +0.07 | +0.01 |

⭐ **All 8 agree under TRUNCATION to the paper's printed 1 dp (8/8); only 4/8 agree under rounding** —
so the paper truncates, which is also what `NAVSIM_PROTOCOL.md` §6.2 observed for the leaderboard rows.
⇒ **the harness reproduces the published navhard CV stage-1 column exactly at its printed precision**,
independently of the two-stage aggregation. The combined EPDMS (and its interval) comes from the
suite re-run in flight; the guard's `sustain` default is now 60 samples (2 min) so a 6-second dip
caused by another process cannot kill an 80-minute run again.

## 6. Integration asks (escalated in the report headline)

1. **E3 — `tools/criteria_check.py` evaluates no `benchmarks.navsim` criterion or gate** (it iterates
   families / hygiene / leak guards only). ✅ It does now. ⚠️ The reference implementation E1 wrote
   for it (`navsim_gates` + `gate_mutations`) **now survives ONLY in W1's promoted copy**,
   `taniteval/taniteval/bench/navsim/artifacts.py:89,124` — see §7.
2. ✅ **CLOSED (E3, registry 2.10.0):** `EPDMS_v2.multipliers` now reads the literal
   `["NC","DAC","DDC","TLC"]`, pinned twice. ⚠️ My finding was true of **HEAD** (2.9.0 still omits
   TLC) — see ask 9: the fix is staged, not committed.
3. ✅ **CLOSED (E3/W2): adapter gaps G1–G6, 9/9 verified against our real `CV.devkit.csv`.** My six
   local fillers are DELETED and this package now calls the adapter (`submetrics_from_row(...,
   stage="one")` resolves all nine terms, 0 missing, TLC = 1.0). I also deleted my own NavSim-gate
   evaluator: `criteria_check` evaluates those gates now, and my copy had gone stale **in the
   opposite direction** — it required `interval.status == UNAVAILABLE` and would have FAILED a
   correct artifact. A second checker that lags the first is worse than none.
4. **E2 — reading this cache on Windows needs the loader patch; the fast runtime is the C: mirror**
   (sent to the orchestrator at 12:15). The cache entries record
   `map_root = C:/Users/Admin/navsim-crun/data/maps`, re-opened by the IDM policy at scoring time.
5. **Register owner** — `GOALS_AND_CLAIMS.md` not edited by E1 (shared file): rows to add for
   finding 1 (harness reproduces the LB), finding 3 (human undefined on two-stage splits) and C7.
6. **PI** — stage-2 loop status is UNRULED under the 2026-09-02 vocabulary; the C8 reference is
   INHERITED (a one-request leaderboard re-read would make it PUB-LB).
7. ✅ **WITHDRAWN — the gate was never blocking.** I read it from HEAD (2.9.0); the settled gate is
   in the working tree. My artifact's refusal was therefore **false**, and is fixed (§5b). The real
   item is ask 9.
10. ⚠️ **A second, measured cost of the same cause, and it is mine:** my `build_artifacts.py`
   restructure deleted `short_row`, `navsim_gates` and `gate_mutations`, which W1 had promoted.
   With nothing committed there is **no history**, so the pre-edit file is unrecoverable and W1's
   copies (`taniteval/taniteval/bench/navsim/artifacts.py:56,89,124`) are the only survivors. I am
   not restoring them (W1's `ORPHANED_PROMOTIONS` control goes RED on restoration, and two of the
   three were deleted *because* they had gone stale). ⭐ The deletion was correct; the
   irreversibility was not — and the two costs today (the false-refusal artifact, these three
   functions) have one cause, ask 9.
9. ⛔ **The settled registry + adapter live in the WORKING TREE, STAGED AND NEVER COMMITTED.** Every
   fresh process — mine, a CI job, anyone's `git show HEAD:` — sees 2.9.0: TLC missing from the
   multipliers, the cluster unit "unsettled", the gate admitting only a refusal. That is how a
   correct measurement came to declare itself unavailable. A staged gate is in force for nobody;
   this needs committing, and it is the one item here with a blast radius beyond NavSim.
8. **W1** — `devkit_patches()` needs my IDM patch entry (sent, and reported landed as entry 6), the
   suite's report hook wrote `REPORT_FAILED` on this run (`raw/benchreport.log`), and both hash
   spaces should be carried per patch entry, labelled.
