# RESULT — E1: the programme's first locally computed EPDMS (NavSim v2, `warmup_two_stage`)

**Harness:** `autonomousvision/navsim@0a380a9063d7162ec93d0f51e9990ebac585f720` (2025-10-27) — on the
**post-#151** side (MEASURED, two probes: README changelog `[2025/09/29]` + the fix's code at
`navsim/evaluate/pdm_score.py:194-219`) — plus three PRE-EXISTING local modifications and one E1
reader patch (§4). **Every number below: tier T1-family; loop OPEN for stage 1, UNRULED for stage 2
(2026-09-02 vocabulary); background vehicles IDM-reactive in both stages.** No CI anywhere:
warmup NEVER carries one (D-BENCH-PORT: 7 log groups < RG-14 floor of 8; 7 re-MEASURED).

## 1. Findings

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

* PRE-EXISTING (found, not made, by E1): `dataclasses.py` PosixPath unpickler (blob `596cb7d`),
  the venv `fcntl.py` flock shim (sha256 `75184a48…`), nuPlan `setup.py` (packaging only).
* E1: the `MetricCacheLoader` separator fix — in-process monkeypatch for all warmup scoring
  (`code/navsim_win.py`), and the same fix applied to the **C: copy** for process-pool children
  (`code/patches/apply_dataloader_patch.py`, `raw/navhard/dataloader_copy_patch.{json,diff}`). ⚠️ Its
  first application was silently wrong (a heredoc ate one backslash: the regex matched only `/`);
  the end-to-end test against the real cache caught it before any run used it.
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

Four families (OUR instruments on the 16 stage-1 trajectories vs the logged human; frame check
verified): present in the artifacts; stage 2 has no human future, so no geometry family exists there.

## 6. Integration asks (escalated in the report headline)

1. **E3 — `tools/criteria_check.py` evaluates no `benchmarks.navsim` criterion or gate** (it iterates
   families / hygiene / leak guards only). E1's `code/build_artifacts.py::navsim_gates` + its
   mutation arms are a ready reference implementation.
2. **E3 — `CRITERIA_REGISTRY.json` `benchmarks.navsim.variants.EPDMS_v2.multipliers` = [NC, DAC, DDC]
   — TLC is missing** (devkit, docs and the adapter all have 4).
3. **E3 — adapter gaps G1–G6** (`build_artifacts.py` docstring): long stage-suffixed devkit columns,
   `estimator.cluster_unit`, `protocol.ego_status_enforcement`, `protocol.sensor_set/setting`,
   `protocol.navsim_protocol/devkit_sha`, `protocol.corpus`/`controls`.
4. **E2 — reading this cache on Windows needs the loader patch; the fast runtime is the C: mirror**
   (sent to the orchestrator at 12:15). The cache entries record
   `map_root = C:/Users/Admin/navsim-crun/data/maps`, re-opened by the IDM policy at scoring time.
5. **Register owner** — `GOALS_AND_CLAIMS.md` not edited by E1 (shared file): rows to add for
   finding 1 (harness reproduces the LB), finding 3 (human undefined on two-stage splits) and C7.
6. **PI** — stage-2 loop status is UNRULED under the 2026-09-02 vocabulary; the C8 reference is
   INHERITED (a one-request leaderboard re-read would make it PUB-LB).
