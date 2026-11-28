# SPEC — W3: NAVSIM **v1.1** `navtest` harness, reference-agent reproduction, frame bank (PRE-REGISTERED)

**Written 2026-09-19 ~15:55 local, BEFORE any scoring run** (no `score_*`/CSV exists in `raw/`
at this time; the only executions so far are an import check and read-only metadata probes).
Its git blob hash is recorded in `raw/SPEC_PREREG_HASH.txt` when staged. Build plan row **W3**
(`…/2026-09-19-eval-suite-build/BUILD_PLAN.md` §2). PI decision (relayed): NAVSIM v1 data on
**D: only**. Owner: EvalFlyWheel W3.

## 0. Facts this SPEC is built on (MEASURED 2026-09-19 unless marked)

| fact | evidence |
|---|---|
| v1.1 code = `autonomousvision/navsim` **v1.1 branch @ `3e8291bfa89ff247231e0227778840cd0a036896`** (setup.py `version="1.1.0"`), unpacked at `D:/Archive/devbox-C/navsim/navsim-3e8291b…/` | `setup.py:15`; zip beside it |
| `navtest` = **12,146 tokens over 136 logs**, `num_history_frames 4`, `num_future_frames 10`, `has_route true` | `…/train_test_split/scene_filter/navtest.yaml` (parsed; 12,146 unique) |
| all **136/136** navtest log pickles present in `D:/…/data/openscene/navsim_logs/test` (147 files) | directory census |
| runtime: C: venv `C:/Users/Admin/navsim-crun/venv` (py 3.9.25, torch **2.0.1+cpu**, numpy 1.23.4, shapely 2.0.7, hydra 1.2.0) with `PYTHONPATH` = the v1.1 tree ⇒ `navsim.__file__` = `D:\Archive\devbox-C\navsim\navsim-3e8291b…\navsim\__init__.py` (the venv's own navsim 2.0.0 is a `.pth` APPENDED after site-packages, so PYTHONPATH wins); `nuplan` = `C:/Users/Admin/navsim-crun/nuplan-devkit` @ **`ce3c323` = tag `nuplan-devkit-v1.2`**, exactly v1.1's pin (`requirements.txt:1`) | import probe, `git describe` |
| ⛔ v1.1 has the SAME Windows loader expression E1 fixed in v2: `MetricCacheLoader._load_metric_cache_paths` → `cache_path.split("/")[-2]` (`navsim/common/dataloader.py:180`) | source read; defect to be VERIFIED on a real v1.1 cache (C7) before the patch is applied |
| v1.1 background agents are **NON-reactive (logged)**: "background actors follow their recorded future trajectories" (`docs/metrics.md`); `pdm_score.py:83-140` scores 2 proposals (PDM-Closed, agent) against the cached interpolated GT observation | source |
| v1.1 PDMS = `NC·DAC·(5·EP + 5·TTC + 2·C)/12`; DDC weight 0 (`pdm_scorer.py:38-49`, `docs/metrics.md`); the ONLY headline column is **`score`** (`PDMResults`, `dataclasses.py:557-568`; v1.1 has no `pdm_score` column) | source |
| EP = raw progress / max compliant raw progress, **≡ 1 for every compliant proposal when that max ≤ 5 m** (`pdm_scorer.py:165-173`) | source |
| the official runner drops failed tokens from the mean (`run_pdm_score.py:144`, `mean(skipna)`) ⇒ a count guard is mandatory | source |

## 1. Question

Does this box's v1.1 harness reproduce the NAVSIM paper's own reference-agent PDMS on `navtest`
(the independent cross-check that validates the v1 harness, as the warmup leaderboard did for
v2)? Then: where do the **STOP** and **CV** floors sit on `navtest`, and — queued behind the
GPU gap — does refcv4b clear them?

## 2. Protocol `PDMS_v1_navtest` (fixed now)

* Metric cache: **official, unmodified** `navsim/planning/script/run_metric_caching.py`,
  `train_test_split=navtest`, `worker=sequential`, cache path
  `D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/metric_cache_navtest` (**D:**).
* Scoring: **official, unmodified** `navsim/planning/script/run_pdm_score.py`
  (`default_run_pdm_score`, `train_test_split=navtest`, `worker=sequential`), one arm per run.
* Env: `NUPLAN_MAP_VERSION=nuplan-maps-v1.0`, `NUPLAN_MAPS_ROOT=D:/Archive/devbox-C/navsim/data/maps`,
  `OPENSCENE_DATA_ROOT=D:/Archive/devbox-C/navsim/data/openscene`, `NAVSIM_EXP_ROOT=D:/…/exp/w3_navtest_v1`,
  `PYTHONHASHSEED=1`, `CUDA_VISIBLE_DEVICES=-1`, OMP/MKL/OPENBLAS threads 2.
* Wrapper `code/navsim_v1_win.py`: asserts the imported tree, applies at most ONE behaviour
  patch (the loader separator, E1's function, only if C7 shows the defect), observation-only
  hooks, RAM guard (floor 3,000 MB).
* **Headline = mean of the `score` column over the valid token rows** (= the devkit's own
  `average` row), ×100. Never `pdm_score`. Tier **T1-family**, loop **OPEN** (one query, plan
  fixed, LQR + bicycle 10 Hz × 4 s), background **non-reactive (logged)**.
* Estimator: **no interval** until W2 registers the log-cluster bootstrap (navtest 136 logs ≥
  RG-14's floor of 8, so it becomes admissible on registration) —
  `{status: UNAVAILABLE, reason, n}`.

## 3. Arms

| arm | agent | declared inputs | status |
|---|---|---|---|
| **CV** | devkit `ConstantVelocityAgent` | `ego_velocity[t0]` | runs |
| **HUMAN** | devkit `HumanAgent` (privileged: the logged future) | scene future | runs |
| **STOP** | `code/w3_agents_v1.py::StopAgent` — 8 × `(0, 0, 0)` | nothing | runs (floor, MANDATORY per the suite contract) |
| **EGO_MLP_s{0,1,2}** | devkit `EgoStatusMLPAgent` + the **published** checkpoints `autonomousvision/navsim_baselines/ego_status_mlp/ego_status_mlp_seed_{0,1,2}.ckpt` (6,518,594 / 6,518,530 / 6,518,530 B; HF API listing 2026-09-19) | v, a, cmd at t0 | ⛔ **BLOCKED — the weights are not on the box and a download needs an explicit approval.** Never trained here. |
| refcv4b `A1_ego_cmd` / `A2_vision_pure` / `A3_ego_nocmd` / `A4_blind_ego_cmd` | E2's bridge functions (`tanitad_navsim_bridge.py`, imported, not copied) over the navtest frame bank → v1.1 seam agent | as E2's arms | **QUEUED behind W1's GPU-gap launcher** (PI: our-model inference waits for a GPU gap) |

## 4. Pre-registered expectations — reference reproduction (the harness validation)

**Primary source:** arXiv **2406.15349v2** (31 Oct 2024), library key `2406.15349`
(sha256 `d3bc66d3…0403c`, re-verified 2026-09-19), **Table 1 (p. 7)** and **Table 3 (p. 9,
"NAVSIM 1.1 Leaderboard")**, re-read today from the PDF:

| arm | NC | DAC | TTC | Comf. | EP | **PDMS** | source |
|---|---|---|---|---|---|---|---|
| Constant Velocity | 68.0 | 57.8 | 50.0 | 100 | 19.4 | **20.6** | Tab. 1; Tab. 3 PDMS 20.6 |
| Ego Status MLP | 93.0 | 77.3 | 83.6 | 100 | 62.8 | **65.6** | Tab. 1 |
| Ego Status MLP (3 seeds) | | | | | | **66.4 ± 0.9** | Tab. 3 |
| Human | 100 | 100 | 100 | 99.9 | 87.5 | **94.8** | Tab. 1 |

Secondary, INHERITED (`NAVSIM_PROTOCOL.md` §6.1, HF `navtest` leaderboard retrieved
2026-08-23): CV **20.6517**; Ego Status MLP **66.3989 ± 0.9406** (3 seeds).

**Verdict rule (fixed now), per arm, PDMS and each sub-score, all ×100:**
* **REPRODUCED** — rounded to the table's printed precision it equals the table
  (0.1 for the paper; 4 dp for the INHERITED leaderboard values).
* **CLOSE** — |Δ| ≤ 0.5 on PDMS: harness plausible, the source of Δ must be named
  (devkit version of the table, data version, token set) before any model row is quoted.
* **NOT REPRODUCED** — |Δ| > 0.5: ⛔ no model number is scored on navtest until diagnosed.
* **BAR-W3-1 (the headline): CV PDMS REPRODUCED against Tab. 1/3 (20.6).** The
  leaderboard's 20.6517 is the sharper secondary test (a 4-dp match would make this the v1
  analogue of E1's warmup 18.5356 match).
* **BAR-W3-2: HUMAN PDMS REPRODUCED against Tab. 1 (94.8)**, sub-scores at 0.1.
* **BAR-W3-3 (blocked): EGO_MLP** — mean over the 3 published seeds REPRODUCED against
  Tab. 3 66.4 (and the LB 66.3989 at 4 dp); the ± is reported as BOTH the population and the
  sample std, because the source does not say which (whichever matches is recorded, not chosen).
  Tab. 1's 65.6 is expected to equal ONE of the seeds at 0.1 (it is a single run in the paper).

## 4b. ⛔ AMENDMENT A1 (2026-09-20 08:36 UTC) — the printed-precision convention is UNSETTLED

**Written BEFORE any navtest CSV existed** (`find raw -name "*_navtest*.csv"` = **0** at
2026-09-20T08:36:12Z; the metric cache was ~44 % built and no arm had been scored). Trigger: E1
MEASURED that the sibling paper [N2] (arXiv 2506.04218v3) **TRUNCATES** its printed values —
8/8 of its Table 2 CV S1 sub-metrics match under truncation, only 4/8 under rounding. §4's rule
was written around ROUNDING, and the two conventions disagree on values like 20.58 (→ 20.6
rounded, 20.5 truncated).

**What the evidence says for THIS paper** (`raw/print_convention_probe.json`, `code/convention_probe.py`):

| test | reading |
|---|---|
| internal: Table 2's three printed seeds 83.3 / 84.0 / 84.4 → sample std **0.5568**, text prints "± 0.56" | **ROUNDING** (truncation prints 0.55) ⚠️ the paper's std came from the TRUE seeds |
| Tab. 3 CV **20.6** vs the INHERITED leaderboard's 20.6517 | **TRUNCATION** (rounding gives 20.7) |
| Tab. 3 TransFuser **83.9** vs 83.8822 | ROUNDING |
| Tab. 3 Ego-MLP **66.4** vs 66.3989 | ROUNDING |
| Tab. 3 LTF ± **0.6** vs 0.552 | ROUNDING |
| Tab. 3 LTF 83.5 / TransFuser ± 0.4 / Ego-MLP ± 0.9 | uninformative (both conventions agree) |

⇒ **UNSETTLED**: 3 ROUNDING / 1 TRUNCATION over the leaderboard cells, and that source is a
moving target two years newer than the paper, so a lone disagreement is as likely to be a
different run as a different convention. ⛔ No convention may be assumed.

**The amended rule** (replaces §4's single-reading verdict; `_verdict` in the suite plugin is the
ONE implementation, imported by `analyze_navtest.py`):

* read the measured value at the table's printed precision BOTH ways — **round-half-up** and
  **truncate** — and print both in every cell;
* **REPRODUCED** only when **both** readings give the published digits;
* **REPRODUCED_UNDER_ROUNDING** / **REPRODUCED_UNDER_TRUNCATION** when exactly one does, with the
  other reading beside it. ⛔ Never reported as a bare REPRODUCED, and never chosen because it is
  the flattering one;
* **CLOSE** when neither reading matches but |Δ| ≤ 0.5; **NOT REPRODUCED** otherwise — unchanged.

The bars themselves (BAR-W3-1/2/3) are unchanged; only how a printed cell is compared.

## 5. Pre-registered controls (each must read a known value)

| id | control | expected |
|---|---|---|
| C1 | per-token `score` recomputed as `NC·DAC·(5·EP+5·TTC+2·C)/12` from the same row | max \|Δ\| ≤ 1e-12, every row, every arm |
| C2 | count guard: log `successful == n_expected`, `failed == 0`; CSV = n token rows + 1 `average` row, all `valid` | exact; a violation REFUSES the run |
| C3 | token set of every full-run CSV == the navtest yaml's 12,146 | set equality |
| C4 | determinism: CV re-scored on the smoke tokens with `PYTHONHASHSEED=2` | per-token rows bit-identical; averages \|Δ\| ≤ 1e-12 |
| C5 | **navtest's own construction rule** (paper §3.1, p. 5): scenes with CV PDMS > 0.8 or human PDMS < 0.8 were REMOVED | 0 tokens with CV > 0.8 and 0 with HUMAN < 0.8 **if** the split was built with this scorer and data. Any violation is reported with its tokens and read as evidence that the split was built with another scorer/cache version — a finding, not a harness failure |
| C6 | STOP shares the human's t0 state, and every navtest scene passes the human at t0 (C5) | STOP NC ≥ 0.98, DAC ≥ 0.98, TTC ≥ 0.95 (a prediction from source, not an identity) |
| C7 | the unpatched v1.1 loader on a cache written by v1.1 on THIS box | raises `IndexError` (predicted: nuPlan writes `str(WindowsPath)`); patched loader maps n cached tokens → n distinct keys, each == its parent directory name |
| C8 | import provenance | `navsim.__file__` under the v1.1 tree, `nuplan` @ `ce3c323`; recorded per run, a mismatch REFUSES |
| C9 | metric-cache count | 12,146 cached, 0 failed (the metadata CSV row count, not the log line) |

## 6. Pre-registered STOP / CV floor expectations (the "EP ≤ 5 m may favour stopping" question)

Reasoning from source (§0): STOP gets NC, DAC, TTC ≈ 1 (C6), EP ≈ 0 wherever the best compliant
progress > 5 m — which navtest's CV > 0.8 removal rule should make the common case — and C = 1
only when braking from v0 stays within the comfort bounds (`min_lon_accel −4.05 m/s²`). Hence
STOP ≈ (5 + 2·C̄ + 5·EP̄)/12 with a floor of **5/12 = 41.7** wherever NC·DAC·TTC = 1.

* **Prediction: STOP PDMS ∈ [38, 60], point ≈ 45.** Committed inequalities:
  **STOP > CV (20.6)** and **STOP < Ego Status MLP (65.6)**.
* Both outcomes committed in advance: if STOP > CV (predicted), **CV is not a floor on
  navtest** and every navtest model row must be read against STOP; if STOP ≥ 65.6, a
  do-nothing plan beats a published learned baseline — a protocol finding to escalate.
* Decomposition reported: fraction of tokens with max compliant progress ≤ 5 m (EP forced to 1),
  STOP's per-term means, and STOP − CV paired per token (W/T/L).

## 7. Pre-registered model bars (refcv4b, queued — scored only after BAR-W3-1 passes)

* **BAR-W3-M1:** refcv4b `A1_ego_cmd` navtest PDMS **> max(STOP, CV)**, paired on the identical
  12,146 tokens (mean Δ, W/T/L, per-log). ⛔ No interval until W2's estimator is registered.
* **BAR-W3-M2 (stretch):** A1 > the published Ego Status MLP (65.6 Tab. 1 / 66.4 Tab. 3) — the
  minimum for "vision adds value over a blind learned baseline" on this protocol.
* Diagnostics beside them (E2's ladder): A2 vision-pure, A3 no command, A4 frames-blind.
* Zero-shot, non-parity corpus, perception-free; never trained on navtest (forbidden).

## 7b. ⛔ AMENDMENT A2 (2026-09-21T07:35:12Z) — a zero-training post-processing arm, registered AFTER BAR-W3-M1 FAILED and BEFORE any smoothed number exists

**State at this timestamp (MEASURED):** `raw/A1S*`, `raw/HUMANS*` and any smoothed seam are
**absent** (a directory probe returned 0 at 07:34:54Z). No smoothed plan has been scored.

**Why this arm, from the A1 diagnosis (MEASURED on the full split):**
* BAR-W3-M1 **FAILED**: A1 **58.9738** vs STOP 61.8202 (CV 20.6517), n = 12,146. BAR-W3-M2 therefore
  fails too (A1 < 65.6). Both stand as written; nothing below changes them.
* The deficit is the multiplicative `NC·DAC` gate: A1 zeroes **3,828 / 12,146** tokens (DAC 3,224,
  NC 903) against STOP's 732; on the **8,318** tokens A1 does not zero it scores **86.11 vs 64.58**.
* A1's plans are **jittery**: the median total |Δheading| along the 8 poses is **26–32°** against a
  NET change of 3–6°, and **87–93 %** of plans flip the sign of their heading change (human
  37–62 %). On the tokens A1 fails DAC it also **under-turns** (net 5.74° vs the human's 10.05°).
* Those two mechanisms predict DIFFERENT outcomes for a smoother — jitter is removable after the
  fact, under-turning is not — which is what makes this a discriminating experiment and not a
  search for a better number.

**Arm `A1S`:** A1's own seam, each plan replaced by a **least-squares cubic in time** for x(t) and
y(t) separately, **anchored at the origin** (p(0) = 0, the ego's pose at t0), fitted to the 8 poses
with equal weight; heading = atan2(ẏ, ẋ) of the fit, except where the fitted speed is below
**0.5 m/s** (the four-families harness's own `min_ds_mps`) where the model's heading is kept.
Degree **3** is fixed a priori as the lowest degree that can represent a lane change (an S in y(t)).
⛔ **No parameter is tuned on navtest, and the arm is scored ONCE.**

**Controls (each must read a known value, on the 200-token subset `raw/A1_sub200_tokens.json`):**
* `HUMANseam` — the logged human future passed through the SEAM path, unsmoothed, must reproduce
  the devkit HumanAgent's per-token scores on those tokens **exactly** (the seam path's own control
  at a scale beyond KX's 20 tokens);
* `HUMANS` — the logged human future through the SAME smoother must stay within **1.0 PDMS** of
  HUMAN raw on the same tokens (94.1195). A smoother that damages the human's own paths makes any
  A1S reading uninterpretable, so a failed control VOIDS the A1S verdict rather than reversing it.

**Bar BAR-W3-M1S:** `A1S > max(STOP, CV)` on the full split, paired, same estimator. **Outcomes
committed now:**
1. **PASS** → the deficit was predominantly jitter. A1S is reported as a **post-processed arm**
   (refcv4b + a fixed cubic smoother), never as refcv4b's raw score.
2. **FAIL, DAC zeros fall** → jitter is a partial cause; the residual is under-turning, a
   model-side lever (curvature following) that needs training → a named PI decision.
3. **FAIL, DAC zeros do not fall** → jitter is not the mechanism; the smoother is refuted as a lever.

## 7c. ⛔ AMENDMENT A3 (2026-09-21T08:00:16Z) — the frames-blind diagnostic (E2's A4), scoped to where A1 fails, registered BEFORE it runs

**State at this timestamp:** no A4 seam, score or directory exists (probe = 0). §7 already
pre-registered A4 as a diagnostic beside BAR-W3-M1; this amendment fixes its **token set**, its
**questions** and what each answer means, before any A4 number exists.

**What forced it (MEASURED, `raw/A1_by_driving_command.json`, `raw/gate_decomposition_A1S.json`):**
* A2's pre-registered smoother arm **A1S FAILED with outcome 3** — DAC zeros ROSE (3,224 → 3,277):
  the lateral error is **not jitter**. The smoother is refuted as a lever.
* Split by NavSim `driving_command`, refcv4b **beats STOP under STRAIGHT (63.96 vs 62.97,
  n = 8,070)** and loses only under LEFT (47.24 vs 58.78) and RIGHT (52.03 vs 60.76). Under LEFT it
  turns as much as the human (net-heading ratio 1.07); under RIGHT 0.70; and under **STRAIGHT on
  a curving road** (the human turns > 10°) it turns **0.149×** as much as the human — it does not
  follow road curvature when told to follow the lane.

**Token set (fixed now, from ground truth + the command ONLY, never from a model output):**
`raw/A4diag_tokens.json` — every STRAIGHT-command token whose logged path turns > 10° net (771)
∪ the 200-token random subset = **957 tokens**. Arm `A4_blind_ego_cmd` exactly as E2 defines it
(every frame one constant grey; ego + command as A1). Scored once, paired against A1 on the same
tokens.

**Questions and committed readings:**
1. **Curvature following on the 771 curve tokens** (median model/human net-heading ratio, positions
   only): **A4 ≈ A1 (|Δratio| < 0.05)** ⇒ the frames carry **no** road-curvature information into
   the plan on NavSim — the lever is **perception / domain transfer**, model-side, needs training.
   **A1 − A4 ≥ 0.05** ⇒ vision informs curvature, but too weakly — the lever is the planner's use
   of it. **A4 > A1 + 0.05** ⇒ vision actively HURTS curvature following.
2. **Is vision helping or hurting overall?** Paired PDMS `A1 − A4` on the 200 random tokens with the
   registered estimator, reported with its interval whatever its sign.
⛔ Neither answer changes BAR-W3-M1 (FAILED) or BAR-W3-M1S (FAILED, outcome 3).

## 8. Frame bank (CAM_F0/L0/R0 only)

* Stitch = E2's `build_frames.py` geometry, **imported, not re-derived** (`rig_key`,
  `build_map`, `sample`, `build_scene`), output `PHYSICALAI_WIDE120_256x640`, per scene
  `u8 [4, 256, 640, 3]`, t0 last.
* Built per verified shard (receipt row `COMPLETE_VERIFIED`, sha256 == HF ETag): extract ONLY the
  needed F0/L0/R0 jpgs to a D: scratch dir, build, then delete the jpgs; archives kept.
  Unique frames stored once (navtest windows overlap by up to 3 frames); per-token index.
* **KB1:** ≥ 6 navtest scenes rebuilt by E2's `build_frames.py` run UNMODIFIED as its own
  process → per-scene `sha256(arr.tobytes())[:16]` **bit-exact** vs this bank's reconstruction.
* **KB2:** every stored frame `mean_px ≥ 1.0`; per-token count guard (built + refused ==
  navtest tokens whose shards are verified); every refusal named with its reason.
* **KB3:** a token is built only if all 4 × 3 source jpgs were extracted (missing ⇒ refused).

## 9. Cost (priced from the smoke, then ESTIMATED)

Smoke ≤ 20 tokens (5 from each of 4 navtest logs in 4 different map locations where
available) for the cache, CV, HUMAN, STOP; per-scene wall time and RSS recorded; full-run cost =
per-scene × 12,146 at ONE worker. HEAVY runs (full cache, full scoring, frame bank) start only
after `C:/Users/Admin/tanitad-caches/a8-occupancy-5k-20260919/run/summary.json` exists.
Smoke numbers are never compared with the paper (a 20-token subset is not navtest).
