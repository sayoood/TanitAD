# RESULT — W3: the NAVSIM **v1.1** `navtest` harness, its controls, and the STOP floor

**Stream** EvalFlyWheel W3 · **2026-09-19/20** · **Protocol** `PDMS_v1_navtest` · **Harness**
`autonomousvision/navsim` **v1.1 branch @ `3e8291bfa89ff247231e0227778840cd0a036896`** (setup.py
`version="1.1.0"`), imported from `D:/Archive/devbox-C/navsim/navsim-3e8291b…` through
`PYTHONPATH`; `nuplan-devkit` `ce3c323` = tag `nuplan-devkit-v1.2` (v1.1's own pin) ·
**Pre-registered** `SPEC.md`, git blob `48562058f6be8e729ec344762f224e76d4a0d152`, staged
**2026-09-19T13:41:38Z with `raw/` empty** (`raw/SPEC_PREREG_HASH.txt`) — before any scoring run.

Every number below is **MEASURED by this stream** unless marked. Tier **T1-family**; loop
**OPEN** (single query, plan fixed, LQR + kinematic bicycle at 10 Hz over 4 s); background
traffic **NON-REACTIVE (logged)** — ⛔ unlike v2, which is IDM-reactive. Headline column
**`score`** (v1.1 has no `pdm_score` column at all).

## 1. Findings

⭐ **Headline: the harness reproduces the NAVSIM v1 reference on navtest** — CV's five
sub-scores match Table 1 exactly and its PDMS is **20.6517**, the leaderboard's number to 4 dp;
the human is **CLOSE** (94.55 vs 94.8) with the whole gap in one term; and a **do-nothing plan
scores 61.82**. ⛔ The CV cell shows the paper truncated **that cell**; the paper's convention
overall stays **UNSETTLED** (§3.1) — three other cells read as rounding.

⭐⭐ **Headline 2 (2026-09-21): refcv4b scores 58.9738 on the full navtest split and FAILS
BAR-W3-M1** — it beats CV by +38.3 but not the do-nothing floor (61.82); BAR-W3-M2 fails too
(§3.6). **It beats STOP on every STRAIGHT-command scene (63.96 vs 62.97, two thirds of navtest)
and loses the whole margin at commanded turns** (−3.51 of the −2.85). Two zero-training levers
were run and **both eliminated with pre-registered outcomes**: plan jitter (a fixed smoother made
it WORSE, §3.7) and "vision is not helping" (frames-blind collapses to 23.3, vision is worth
**+34.7 PDMS**, §3.8). What remains is model-side: refcv4b **sees the road but does not commit to
the curve** — 60.8 % of its plans on curving STRAIGHT-command roads barely turn. Inference is
**deterministic** (200/200 plans bit-identical across two launches), so no inference variance
hides behind any of these numbers.

1. **The v1.1 runtime needed no new venv, and it is asserted rather than trusted.** The existing
   C: venv (py 3.9.25, torch 2.0.1+cpu, numpy 1.23.4, shapely 2.0.7, hydra 1.2.0) satisfies
   v1.1's `requirements.txt`, and its `nuplan-devkit` is exactly v1.1's pin. `PYTHONPATH` beats
   the venv's own navsim 2.0.0 (a `.pth` appended after site-packages) — and every run REFUSES
   unless `navsim.__file__`, the script's `__file__` and `nuplan.__file__` are inside the expected
   trees (control C8, recorded in each run's manifest). All v1 **data** is on **D:** (PI).
2. ⛔ **v1.1 carries the SAME Windows loader defect E1 fixed in v2, and it is now proven on a real
   v1.1 cache.** `MetricCacheLoader._load_metric_cache_paths` keys the cache by
   `cache_path.split("/")[-2]` (`navsim/common/dataloader.py:180`) while nuPlan writes
   `str(WindowsPath)`: **20/20 metadata rows are backslash paths**, the unpatched loader raises
   `IndexError`, and E1's separator-only replacement maps **20/20 tokens to their own path, every
   file present** (`raw/C7_loader_smoke20.json`). A 5-test unit pin shows the two function bodies
   are the same code except the token expression, with the defect exercised
   (`tests/test_v1_loader_patch.py`).
3. ⭐ **ON navtest A DO-NOTHING PLAN SCORES 61.82 — THREE TIMES CONSTANT VELOCITY (20.65) AND
   WITHIN 3.8 POINTS OF THE PUBLISHED BLIND LEARNED BASELINE (65.6 / 66.4).** Paired on the
   identical 12,146 tokens: **STOP − CV = +41.17**, W/T/L 8,607 / 727 / 2,812, log-cluster CI
   **[+39.50, +42.84], separated**; STOP is ahead in every t0-speed band. The mechanism is the
   protocol: NC 97.40 · DAC 96.53 · TTC 96.40 make **5/12 = 41.67 points nearly free**, the LQR's
   braking coast earns a median **5.41 m** of real progress, and on **8.28 %** of tokens the best
   compliant progress is ≤ 5 m so **EP ≡ 1 by rule**. ⇒ **CV (published 20.6) is not a floor on
   navtest**; every navtest row must carry its STOP floor (§3.3).
4. **The navtest frame bank is BUILT and verified end to end: 12,146 / 12,146 tokens.**
   All **32 camera shards** (127,882,665,618 B, every one sha256 == its HF ETag) streamed once:
   **53,616 jpgs** extracted → **17,872 unique frames** stitched → **12,146 scenes**, with
   **0 frame failures, 0 partial tokens, 0 carry-over** (no navtest log spans a shard boundary)
   and **17 distinct camera rigs**. `bank_finalize.py` then re-read and **re-hashed every token's
   `[4,256,640,3]` stack**: 12,146 verified, **0 bad, 0 missing, PASS**
   (`raw/frame_bank_final.json`, per-shard `raw/frame_bank_shards_navtest.json`).
   Cost 5.96 h (extract 3.92 + stitch 2.04), size **8.18 GiB** — the unique-frame layout is
   **2.7× fewer frames** than the 48,584 frame-slots a per-scene bank would hold, and it avoids
   ~36 GiB of exFAT cluster waste and 24,292 files. Pixels are E2's: KB1 pinned 20/20 scenes
   bit-exact against `build_frames.py` run unmodified as its own process — per-scene sha256, the
   full array, the `src` mask, rig key, observed fraction and mean pixel
   (`raw/KB1_frame_bank_vs_e2.json`).
5. **refcv4b is RUNNING on navtest — on CPU, through the gap launcher, banking resumable parts.**
   The bridge scores through E2's own `declare` / `pack_frames` / `knots_to_navsim`, and
   `run_model_dev` is pinned **bit-identical** to E2's `run_model` on CPU
   (`tests/test_bridge_device.py`). ⛔ **CORRECTION (2026-09-20T20:4xZ, MEASURED on the live
   launch): I wrote that the queued arm was "WAITING … the launcher re-checks every 60 s for up
   to 8 h". It never waited, and on `--device auto` it never could.** W1's launcher is documented
   (`taniteval/taniteval/bench/gpu_gap.py:8-10`, code at `:230-256`) as **`auto` = ONE probe →
   `cpu`** (event `NO_GAP_CPU`) and **`cuda` = probe every `interval_s` up to `max_wait_s`
   (default 8 h) → then ALSO `cpu`** (event `GAP_WAIT_TIMEOUT`). The 8 h re-check belongs to
   `cuda`, and I attached it to `auto`. The live launch emitted
   `[gpu-gap] NO_GAP_CPU {'reasons': ['GPU memory used 1906 MiB >= limit 1024 MiB']}` and was on
   CPU **1.2 s later** — the artifact said so the whole time
   (`raw/bridge_navtest/seam_A1_ego_cmd.manifest.json → device_gate`), and my prose did not.
   ⭐ **And the correction found a real hole, which is now closed.** Either fallback lands on the
   CPU path **without the CPU gate's own condition ever being evaluated** — and that condition is
   the arbitration's: *CPU inference beside a LIVE training process is REFUSED unless the operator
   names the override*. So `--device auto --gpu-gap` would have run beside a live trainer. The
   bridge now **re-asks the gate as if `--device cpu` had been typed** and refuses on a no; the
   live run records `[bridge] no gap -> CPU, and the CPU gate re-checked: explicit --device cpu
   and no training process is alive`. Reported to W1 (in code freeze) as an integration ask, not
   edited in their file.
   ⭐ **The pass is now crash-survivable, which is what makes a ~10 h CPU run admissible at all.**
   It banks an atomic resume part every 200 tokens, streams every row to `rows_A1_ego_cmd.jsonl`,
   re-checks the brief's 3 GB RAM floor **between parts** (sustained, never on one sample), runs
   at `BELOW_NORMAL`, and resumes from whatever is on disk — so a kill costs ≤ 200 tokens, a
   partial pass is still **scorable** on the tokens it holds, and **a GPU gap opening later
   finishes the remainder on CUDA** (the seam then records `mixed_device: true`; a mixture is
   flagged, never implied). Pinned by `tests/test_bridge_resume.py` (11 tests: order, bit-identical
   one-part-vs-three, an unreadable part recomputed rather than counted done with a same-breath
   control, and the RAM sustain with its **mutation** control — `sustain=1` must read the opposite).
6. **The suite plugin is landed and validates.** `taniteval/taniteval/bench/plugins/navsim_v1.py`
   implements `run_benchmark(ctx)`; its `summary.json` passes W1's schema **and** the cross-field
   contract on the smoke run, with both deliberate-regression arms going RED (drop the STOP floor;
   set the headline column to `pdm_score`) — `tests/test_plugin_summary_contract.py`, 6 tests.
7. **Every artifact passes `criteria_check.py` with 0 violations** (registry 2.10.0) for CV,
   HUMAN and STOP — after two gaps were closed: the `navsim.ego_enforcement` gate needed a
   **mechanism + structured evidence per arm** (the devkit's own agent source; StopAgent reads
   nothing), and the harness's **undefined** lateral metrics on a zero-length plan needed an
   explicit refusal carrying the harness's own counters (`n_steps_heading 0`,
   `excluded_below_min_ds 160`) instead of a bare `null`.
8. ⚠️ **The binding constraint on this box is MEMORY, not CPU.** A sibling agent's CPU training
   smoke (`refc_v3_train.py --device cpu --batch 4 --image-hw 416 1024`) committed **33.9 GB** and
   drove system-available memory from 13 GB to **419 MB**; the RAM guard correctly aborted my
   first full-cache pass (exit 3, `min_system_available_mb 1379`). The cache is now built **one
   log per process**, waiting for ≥ 3.5 GB before each log and retrying a guard abort — so a
   sibling's spike costs one log, not the pass.

## 2. What is MEASURED (the 20-token smoke and the controls)

**Arms, ×100** (`raw/{CV,HUMAN,STOP}_smoke20/*.csv`; 20 tokens from 4 logs in 4 map locations —
⛔ a subset, never comparable to the paper):

| arm | NC | DAC | TTC | C | EP | DDC (w = 0) | **PDMS** |
|---|---|---|---|---|---|---|---|
| CV | 85.0 | 70.0 | 80.0 | 100.0 | 33.29 | 75.0 | **37.62** |
| **STOP** | 95.0 | 100.0 | 100.0 | 80.0 | 22.56 | 100.0 | **61.48** |
| HUMAN | 100.0 | 100.0 | 100.0 | 100.0 | 81.10 | 100.0 | **92.12** |

**Controls** (each must read a known value; SPEC §5):

| id | control | result |
|---|---|---|
| C1 | per-token `score` == `NC·DAC·(5·EP+5·TTC+2·C)/12` | **max \|Δ\| 0.0** on every row of every arm (4 runs × 20) |
| C2 | count guard: log `successful == 20`, `failed == 0`, CSV 20 valid rows + 1 `average` | **PASS** ×4 |
| C3 | CSV token set == the requested set | **PASS** (0 missing, 0 unexpected) |
| C4 | determinism: CV re-scored with `PYTHONHASHSEED=2` | per-token cells **bit-identical**; `average` row \|Δ\| **5.55e-17** (row ORDER differs — `list(set(...))`, E1's C7 mechanism) |
| C5 | navtest's own construction rule (paper §3.1: CV > 0.8 and human < 0.8 were removed) | **0 / 20 violations** either way (max CV 0.799, min human 0.830) |
| C6 | STOP prediction NC/DAC/TTC ≥ 0.98/0.98/0.95 | DAC 1.00 ✅, TTC 1.00 ✅, **NC 0.95 ❌** (1 token: an at-fault collision while stopped, where CV also scores 0 and the human passes) — reported as failed-as-written |
| C7 | the v1.1 loader on a real v1.1 cache | unpatched **IndexError**; patched **20 → 20** distinct keys, each == its parent dir, every file present |
| C8 | import provenance | navsim/nuplan/script all inside the pinned trees, recorded per run |
| C9 | metric-cache count (smoke) | 20 expected / 20 metadata rows / 0 missing / 20 backslash rows |
| KX | the export == what the scorer handed the agents | CV and HUMAN poses **max \|Δ\| 0.0** on 20/20; STOP poses exactly 0 |
| KB1 | frame bank vs E2's builder | **20/20 bit-exact** (sha16, array, src, rig key, observed_frac, mean_px) |
| KB2/KB3 | content + completeness | every stitched frame `mean_px ≥ 1.0`; a token is built only with all 4 × 3 jpgs (99 extracted, 33 unique frames, 0 failures) |

**Cost, MEASURED, ONE worker, BELOW_NORMAL priority** (`raw/cache_smoke20/*_hooks.json`):
metric cache **1.47 s/scene** (median 1.14; the first scene of a new map costs 2.3–19.1 s of map
load), peak RSS **450 MB**; scoring **0.06–0.075 s/token** (`pdm_score` itself 0.027 s), peak RSS
**≈ 410–440 MB**. ⇒ **full navtest ESTIMATED: cache ≈ 5–6.5 h, each scoring arm ≈ 15–25 min.**
⚠️ The devkit's own `cache_data` first builds a SceneLoader over **all 136 logs** (2.1 GB RSS,
~2 min) — the reason for the per-log mode (§1.8).

**The four families are CHEAP at navtest scale, MEASURED on a synthetic win of the exact shape** (12,146 windows × 8 poses, 136 log clusters, `n_boot=2000`): `scenes_to_win` **3.1 s**, `four_families_block` **6.5 s**, all three geometry families OK with a log-cluster CI attached (STRATEGIC stays UNAVAILABLE with its reason). So chain step S3b costs seconds, not hours — and the adapter's own frame guard REFUSED the first synthetic draft whose speeds and displacements disagreed (rel err 0.471 > 0.35), which is the guard doing its job on a fabricated input.

**Tests: 48 passing** — `test_v1_loader_patch.py` (5) + `test_w3_agents.py` (8) +
`test_reconstruct_rows.py` (3) in the NAVSIM venv; `test_plugin_summary_contract.py` (6) +
`test_bridge_device.py` (2) + `test_frame_bank_selection.py` (6) +
`test_device_gate.py` (11) + `test_published_table.py` (7) in the TanitAD venv. The recovery from an aggregation crash is pinned
against a REAL devkit CSV: the rows streamed by the wrapper rebuild all 20 tokens and the
`average` row **bit for bit** (< 1e-12).

**The suite device gate (orchestrator arbitration 2026-09-20) runs through W1's shared gate.**
`plugins/navsim_v1.py::device_decision` now CALLS `taniteval.bench.gpu_gap.device_gate`
(MEASURED live: `gate_source` reads that name in every run dir) and keeps its composition of W1's
`probe_training_pids` only as the fallback if that export is ever withdrawn. ⚠️ The call shape is
pinned by a test because it bit once: W1's gate takes the device **string** plus keyword
arguments, not the args namespace — passing the namespace raised `ValueError`. W3 passes a
`queue_hint`, so W1's refusal text carries the exact command that queues this run. The rules: `--device auto|cuda` → wait for a
GPU gap; `--device cpu` → allowed only with no training process alive; `--device cpu` beside a
live trainer → **REFUSED** unless `--accept-training-box-load` (or
`W3_ACCEPT_TRAINING_BOX_LOAD=1`), and the flag **and the accepted process** are written to
`bench_run.json: device.gpu_gap` (MEASURED through the CLI: `raw/cli_dryrun/`; W1's `record`
block is preserved verbatim, incl. `override_flag` and `training_process_definition`).
`code/run_bridge_navtest.py` calls the same function and exits **rc 3** on a refusal — except
under `--gpu-gap` on `auto|cuda`, where "no gap **yet**" is the launcher's cue to WAIT, which is
what that flag means. ⚠️ "training process" is W1's single definition
(`gpu_gap.TRAINING_SCRIPT_RE`) — W3 does not carry a second one. MEASURED on this box: `auto` and
`cuda` both REFUSE while the GPU holds 3.1 GB, **`cuda` + the override still refuses** (the
override accepts CPU load only, never the card), and `cpu` is allowed with no trainer alive.

**The published reference numbers have ONE production source.** They were heading for three copies (the plugin's `PUBLISHED`, the analysis's own table, W4's banked JSON); the analysis now IMPORTS the plugin's table by path (pure stdlib at module level, so it loads in the NAVSIM venv too) and keeps no fallback copy — `test_published_table.py` pins all of them against literals read from the banked PDF and fails if a private copy of a published number reappears in the analysis.

**The `PDMS_v1_navtest` summary row is now MEASURED, not declared** (W1's
`profiles.summary_row_shape`): **exactly 1 summary row, token verbatim `average`** — read out of
four independent real v1.1 CSVs on this box (CV / HUMAN / STOP / a CV replicate; each 20 hex token
rows + one non-hex row whose token cell is `'average'`, no whitespace), cross-checked against the
line that writes it, `run_pdm_score.py:145` `average_row["token"] = "average"` (v1.1 @ `3e8291b`).
Two details for the consumer: the CSV's first column is the unnamed pandas index, and the summary
row's `valid` cell is the AND over all token rows (`:146`) — a guard signal, not decoration.
Evidence: `raw/W1_summary_row_shape_navsim_v1.json`.

**Reuse of a banked scoring run is keyed on the ARM and re-checked on the file** (ruling (a)/(b)):
the key is `(arm runner, devkit agent class, metric cache)` — `AGENT_CLASS` has three distinct
values, so no arm can inherit another's CSV — and `recheck_csv` re-runs the count guards **on the
file as it is now** (token rows == valid == 12,146, exactly **1** `average` row — v1.1 has one,
not v2's three `extended_pdm_score_*` — token set equality, and the C1 PDMS identity ≤ 1e-12).
A freshly scored CSV goes through the same re-check before it is copied into the run dir. Every guard has an arm that must go RED:
the original loader expression on a Windows path, an unknown seam token, a fingerprint mismatch,
a non-finite seam, the v2 `/14` denominator, a dropped STOP floor, a `pdm_score` headline, a
tampered bank sha, a shard whose sha256 ≠ its HF ETag, a token with 11 of its 12 jpgs.

**The plugin is wired into W1's CLI end to end** (`raw/cli_dryrun/`):
`python -m taniteval.bench navsim_v1 --ckpt none --split navtest --arms CV,STOP,HUMAN --dry-run`
writes a run dir whose `bench_run.json` **validates with 0 errors** (protocol `PDMS_v1_navtest`,
12,146 scenes / 136 logs, devkit `3e8291b` + the one in-process patch, tier/loop stamps) and
REFUSES with a count: *"the navtest metric cache … is not complete: 1 metadata csv, 19 rows,
763 pickles, expected 12146"*. ⚠️ That 19-row CSV is itself a finding: **each per-log caching
process overwrites the metadata CSV with its own log's rows**, so the cache is only loadable
after the final full pass — which is why the driver always runs one and guards on 12,146 rows.

## 2b. ⛔ The published table's printed-precision convention is UNSETTLED — and the rule was amended BEFORE the number

E1 MEASURED that the sibling paper [N2] (arXiv 2506.04218v3) **TRUNCATES** its printed values
(8/8 of its Table 2 CV S1 sub-metrics match truncating, 4/8 rounding). W3's §4 verdict rule was
written around ROUNDING, so the convention had to be settled — or declared unsettled — before any
navtest number was read. `code/convention_probe.py` → `raw/print_convention_probe.json`:

| test on arXiv 2406.15349v2 | implies |
|---|---|
| **internal**: Table 2's printed seeds 83.3/84.0/84.4 → sample std **0.5568**; the text prints "± 0.56" | **ROUNDING** (truncation prints 0.55) |
| Tab. 3 CV **20.6** vs the INHERITED leaderboard 20.6517 | **TRUNCATION** (rounding gives 20.7) |
| Tab. 3 TransFuser **83.9** vs 83.8822 · Ego-MLP **66.4** vs 66.3989 · LTF ± **0.6** vs 0.552 | ROUNDING ×3 |
| Tab. 3 LTF 83.5 · TransFuser ± 0.4 · Ego-MLP ± 0.9 | uninformative (conventions agree) |

⇒ **UNSETTLED** (3 ROUNDING / 1 TRUNCATION on a leaderboard that is a moving target two years
newer than the paper; the one internal test says ROUNDING). **SPEC AMENDMENT A1**, recorded
2026-09-20T08:38:24Z with **0 navtest CSVs in `raw/`** (`raw/SPEC_PREREG_HASH.txt`, blob
`ec36eeb3…`): every cell now prints **both** readings — round-half-up and truncate — and counts as
REPRODUCED only when **both** give the published digits; one-sided agreement is reported as
`REPRODUCED_UNDER_ROUNDING` / `REPRODUCED_UNDER_TRUNCATION` with the other reading beside it,
never as a bare REPRODUCED and never chosen for being the flattering one. The bars are unchanged;
only how a printed cell is compared. One implementation (`plugins/navsim_v1.py::_verdict`), which
the analysis imports — pinned by a test that fails if the vocabulary is copied.

⭐ **And W3's own CV number will be EVIDENCE about the convention, not just a subject of it.** The
CV row is the same quantity in both directions: the devkit's own agent, the same split, the same
scorer. If the measured CV lands in **[20.65, 20.70)** the paper's printed 20.6 can only be a
truncation (rounding would print 20.7) — which would corroborate E1's [N2] finding with a PRIMARY
measurement instead of the INHERITED leaderboard; if it lands in **[20.55, 20.65)** both
conventions print 20.6 and the cell stays silent on the question. Either way the reading is
reported as measured, and the convention question is answered by where the number falls — not by
which answer would be more convenient.

## 3. ⭐ THE FULL-SPLIT RESULT — 12,146 tokens, 136 logs, one worker

All three arms: **12,146 successful / 0 failed**, CSV 12,146 token rows + exactly 1 `average` row,
C1 PDMS identity **max |Δ| = 0.0** on every row of every arm, C3 token set equality, loader patch
applied. Scoring wall 1,222 s / 1,051 s / 1,127 s. Interval = **W2's registered log-cluster
bootstrap** over the **136 logs** (it reproduces the official aggregate on the full sample in every
case).

| arm | NC | DAC | TTC | C | EP | DDC (w=0) | **PDMS** | CI95 (log-cluster) |
|---|---|---|---|---|---|---|---|---|
| **CV** | 68.0183 | 57.8380 | 50.0329 | 100.0000 | 19.4370 | 79.907 | **20.6517** | [19.20, 22.22] |
| **STOP** | 97.3983 | 96.5256 | 96.4021 | 69.4385 | 30.9994 | 99.593 | **61.8202** | [60.70, 63.08] |
| **HUMAN** | 100.0000 | 100.0000 | 100.0000 | 99.9012 | 86.9629 | 98.827 | **94.5514** | [93.79, 95.23] |

### 3.1 BAR-W3-1 (CV) — **PASS, with the convention named**

Against the paper's Table 1 (p. 7), under AMENDMENT A1's dual reading: **NC, DAC, TTC, Comf and EP
are all REPRODUCED** (68.0 / 57.8 / 50.0 / 100 / 19.4 — both readings give the published digits).
The PDMS cell is the decisive one:

* measured **20.6517** (exactly: 20.651651538606658);
* published Tab. 1 / Tab. 3 **20.6** → rounding reads **20.70**, truncation reads **20.60**
  ⇒ **REPRODUCED_UNDER_TRUNCATION**;
* INHERITED HF leaderboard **20.6517** at 4 dp → rounding reads **20.6517**, truncation reads
  **20.6516** ⇒ **REPRODUCED_UNDER_ROUNDING**.

⭐ **THE CELL IS SETTLED; THE PAPER'S CONVENTION IS NOT.** One underlying number — 20.65165… —
explains both printed forms: **this cell was TRUNCATED** in the paper (rounding to 1 dp would have
printed 20.7) and **ROUNDED** on the leaderboard (20.6517). W3's value fell in the discriminating
window [20.65, 20.70); in [20.55, 20.65) the cell would have been silent.

⛔ **That is a statement about ONE CELL, and it does not license "the paper truncates."** W3's own
probe (`raw/print_convention_probe.json`) returns **UNSETTLED** and still does: the three other
informative cells — TransFuser **83.9** vs 83.8822, Ego-MLP **66.4** vs 66.3989, LTF ± **0.6** vs
0.552 — all read as ROUNDING, and the one test internal to the paper (its own printed seeds → std
0.5568, printed "± 0.56") reads as ROUNDING too. A paper that truncated everywhere would have
printed 83.8, 66.3 and 0.5. **One cell demonstrably truncated is not a convention.**

What the measurement DID change is the evidence class of a single cell, from *print vs an
INHERITED leaderboard value* to *print vs a value W3 measured*: the remaining three cells compare
the 2024 print against a live leaderboard retrieved 2026-08-23, so they may record a different
RUN rather than a different rule — unexplained either way. ⚠️ And the cell's own conclusion rests
on one stated assumption: that the paper's CV row is the same underlying quantity as W3's and the
leaderboard's. It is well supported (five sub-scores match Tab. 1 at 1 dp, the PDMS matches the LB
at 4 dp, and the paper states Tab. 3 IS that leaderboard) but it is an assumption, not a
measurement — had the paper's own CV been e.g. 20.6499, both conventions print 20.6 and the cell
would say nothing.

⇒ **For any OTHER cell of this paper, AMENDMENT A1 still applies unchanged**: print both readings
and never assume. The amendment is what made this result reportable without picking a side.

### 3.2 BAR-W3-2 (HUMAN) — **CLOSE, not reproduced, and the gap is one term**

PDMS **94.5514** vs published **94.8**: rounding 94.60, truncation 94.50, neither reads 94.8, and
|Δ| = 0.25 ≤ 0.5 ⇒ **CLOSE**. NC, DAC, TTC and Comfort all **REPRODUCED** exactly (100 / 100 / 100 /
99.9). The whole gap sits in **ego progress: 86.9629 vs 87.5, Δ −0.54 ⇒ NOT REPRODUCED**. EP is the
only sub-score normalised against a *computed* quantity — PDM-Closed's own progress in the metric
cache — so a small difference in the cached planner output moves it while every other term, which
depends only on the logged scene, matches exactly. ⛔ Reported as a FAILED cell; not smoothed.

### 3.3 ⭐ STOP on navtest — a do-nothing plan scores **61.82**, three times CV

**STOP − CV = +41.17 PDMS points**, paired on the identical 12,146 tokens, W/T/L **8,607 / 727 /
2,812**, log-cluster CI **[+39.50, +42.84] — separated**. STOP is ahead in **every** t0-speed band
(0–1 m/s 59.94 vs 51.42 · 1–4 71.53 vs 29.76 · 4–8 63.14 vs 9.79 · >8 48.31 vs 12.84).

The mechanism is the protocol, measured rather than argued:

* a stopped ego keeps **NC 97.40 · DAC 96.53 · TTC 96.40**, so **5/12 = 41.67 points are nearly
  free** before any progress is made;
* **EP is not 0**: the LQR coasts while braking — STOP's median progress is **5.41 m** against
  PDM-Closed's **20.19 m** — and on the **1,006 / 12,146 (8.28 %)** tokens where the best compliant
  progress is ≤ 5 m, **EP ≡ 1** by rule (`pdm_scorer.py:165-173`): STOP's EP there is **93.94**
  against **25.32** elsewhere;
* comfort is the only term that punishes it, and only at speed: **100 %** below 4 m/s, **77.05 %**
  at 4–8, **0.12 %** above 8 m/s.

⚠️ **My pre-registered range was WRONG, and the direction matters**: SPEC §6 committed STOP ∈
[38, 60] with a point estimate of ≈ 45; the measured **61.82 is above the range**. The two committed
*inequalities* both hold — **STOP > CV** (+41.17) and **STOP < the published Ego-Status-MLP**
(61.82 < 65.6 / 66.4) — but the band did not contain the answer, because I under-estimated both the
braking progress and the ≤ 5 m regime. Recorded as a failed prediction, not re-fitted.

⇒ **CV (published 20.6) is not a floor on navtest.** A plan that reads nothing and does nothing
scores 61.82, within **3.8 points** of the published blind learned baseline. Every navtest row must
carry its STOP floor.

### 3.4 Controls at full scale

| control | result |
|---|---|
| C1 PDMS identity | **max \|Δ\| = 0.0** over 3 × 12,146 rows |
| C2 count guard | 12,146 successful / 0 failed / 1 `average` row, all three arms |
| C3 token set | 0 missing, 0 unexpected, all three arms |
| C5 the split's own construction rule (CV ≤ 0.8, human ≥ 0.8) | **8 / 12,146** CV above 0.8 and **9 / 12,146** human below — **0.07 %**. The rule essentially holds; the handful of violations say the split was filtered with a slightly different scorer or cache build, which is also the most likely home of the human EP gap in §3.2 |
| C6 STOP prediction (NC ≥ .98, DAC ≥ .98, TTC ≥ .95) | **2 of 3 FAILED as written**: NC 97.40 ✗, DAC 96.53 ✗, TTC 96.40 ✓ — a stopped ego is hit slightly more often than the "it starts where the human starts" argument implied |
| C9 cache | 12,146 pickles = the split's tokens exactly; the metadata CSV written by the devkit's own `save_cache_metadata` and verified on content (12,146 rows, all files present, loader maps 12,146 → 12,146) |
| four families (OUR instruments) | LONGITUDINAL / LATERAL / TACTICAL **OK** on all three arms with log-cluster CIs; STRATEGIC UNAVAILABLE with its reason |
| `criteria_check.py` (registry 2.10.3) | **0 violations** on all three artifacts (6 / 6 / 9 reasoned work items) |

**Separations** (log-cluster paired bootstrap, 136 clusters): HUMAN − STOP **+[31.49, 33.89]**,
HUMAN − CV **+[72.29, 75.45]**, STOP − CV **+[39.50, 42.84]** — all separated. ⚠️ The estimator
answers *"would another draw of LOGS say this?"* only; it is blind to training and inference
variance by construction (`H-ESTIM-SEED-1`). For these three arms that is the whole question —
none of them is trained, and all are deterministic.

### 3.5 ⚠️ PROVISIONAL — refcv4b on the first 200 navtest tokens, paired (the BAR is decided on the full split)

**Evidence class MEASURED** · tier **T1-family** · loop **OPEN** (the planner consumes its own
actions; this is not closed loop) · background **non-reactive (logged)** · estimator **W2's
registered log-cluster bootstrap**, 200 tokens over **93 logs**.
⛔ **This does NOT decide BAR-W3-M1.** The pre-registered bar is read on the full 12,146-token
split; this is the first resume part of the live pass, read early because the parts exist. Against
the paper it is not even admissible — the analyzer **suppresses every published verdict** under a
token restriction (`raw/analysis_navtest_sub200.json → restricted`, pinned by
`tests/test_analyze_restrict.py`). The floors are **not re-scored**: they are the banked full-split
per-token CSVs restricted to the same 200 tokens, so every arm is read on identical scenes.

| arm | PDMS ×100 | NC | DAC | EP | TTC | C | 95 % CI (±) |
|---|---|---|---|---|---|---|---|
| CV (constant velocity) | 21.8222 | 68.50 | 56.00 | 19.45 | 49.50 | 100.00 | 0.0395 |
| STOP (do nothing) | 62.5812 | 97.00 | 99.00 | 30.59 | 96.50 | 66.00 | 0.0260 |
| **refcv4b `A1_ego_cmd`** | **57.9689** | 88.75 | 74.50 | **51.20** | 81.50 | **100.00** | 0.0587 |
| HUMAN (log replay) | 94.1195 | 100.00 | 100.00 | 85.89 | 100.00 | 100.00 | 0.0134 |

**Paired (same 200 tokens, log-cluster bootstrap):**
* `A1 − CV` = **+0.3615**, CI95 ±0.0599 → **separated**. refcv4b beats constant velocity, clearly.
* `A1 − STOP` = **−0.0461**, CI95 ±0.0590 → **NOT separated**, and the point estimate is below zero.
⇒ **Provisionally the bar is NOT met**: `A1 > max(STOP, CV)` fails on the STOP side, which is the
side that matters, because on navtest v1 a do-nothing plan scores **three times CV** (§3.3).

⭐ **AND THE MEAN HIDES THE SHAPE — THE LEVER IS NAMED AND MEASURED.** Head to head against STOP,
refcv4b **WINS 113 of 200** (W/T/L **113/15/72**) with a **median of +9.1722**, while its **mean is
−4.6122**. PDMS_v1 multiplies by `NC·DAC`, so a single non-compliant token is a **zero**, not a
penalty, and a minority of zeros outweighs a majority of wins:

| | refcv4b | STOP |
|---|---|---|
| mean per-token `NC·DAC` gate | **0.6625** | 0.9600 |
| tokens zeroed by NC or DAC | **67 / 200** (NC 21, DAC 51) | **8 / 200** |
| mean PDMS on the **133** tokens refcv4b does **not** zero | **87.1713** | 63.5858 |

⇒ **On the scenes it does not zero, refcv4b beats a do-nothing plan by +23.6 PDMS.** The entire
deficit is the multiplicative gate, and within it **drivable-area compliance (51 tokens) dominates
at-fault collisions (21)**.

⚠️ **Is that a driving defect or a bridge bug?** A cheap discriminator says **driving**: the DAC
failures are spread over **39 of the 93 logs** (at most 5 in any one log, so not a per-log data
defect) and they are **speed-gated** — **0.0 % below 1 m/s** (n=23), 16.4 % at 1–4, **34.1 %** at
4–8, **35.0 %** above 8 m/s. A frame, sign or units error in the seam would not switch itself off
at low speed, so this reads as **driving**, not plumbing.
⛔ **CORRECTED (2026-09-21) — I first wrote that "the median *planned* progress on a zeroed token is
32.07 m against 18.81 m", and concluded "a planner that commits to long fast paths".** Those two
numbers are index **[0]** of the hook's `progress_raw_m`, which is **`[PDM-Closed, agent]`**
(`code/navsim_v1_win.py:199`) — the **reference** planner's progress, not refcv4b's. With the
right index:

| median progress | refcv4b's own plan | PDM-Closed reference | refcv4b / reference |
|---|---|---|---|
| tokens refcv4b **fails** DAC (n=51) | **23.21 m** | 32.07 m | **0.750** |
| tokens it complies (n=149) | 14.59 m | 18.81 m | 0.808 |

⇒ The failing tokens are **faster scenes** (the reference itself goes 1.7× further), and in them
refcv4b plans **relatively SHORTER**, not longer. **The "over-long plan" reading is refuted; the
failure is the path's SHAPE at speed** — heading/curvature, i.e. the LATERAL family — which is
exactly what the full split's four-family artifact measures directly (curvature, heading and
yaw-rate error). Class of the error, logged in COMMS: *an array index quoted without its
semantics* — the same family as a tensor without its units.

⭐ **And the corrected hypothesis has already met its first test** — a per-token geometric read of
refcv4b's 4 s pose against the logged human pose (`code/decompose_gate.py::lateral_at_horizon`,
pinned against analytic targets in `tests/test_decompose_gate.py`; independent of the suite
adapter), `raw/gate_decomposition_A1sub200_sub200.json`:

| median at 4 s vs the human | heading error | cross-track | FDE |
|---|---|---|---|
| tokens refcv4b **fails** DAC (n=51) | **10.51°** | **1.722 m** | 3.761 m |
| tokens it complies (n=149) | 3.35° | 0.377 m | 3.541 m |

⇒ **The endpoint distance is the same (FDE 3.76 vs 3.54 m) while the cross-track error is 4.6×
and the heading error 3.1×.** The failing plans are not too long or too short — they are
**displaced sideways and mis-headed**, and an FDE/ADE table would show nothing at all. Cross-track
grows monotonically with t0 speed (0.09 → 0.31 → 0.65 → 1.19 m across the four bands). This is the
binding four-families rule earning its keep on the first model reading: **the lever is LATERAL,
and ADE cannot see it.** Still n=200 and PROVISIONAL; the full split re-runs the same script.

**Artifacts** — `raw/A1sub200_navtest/` (CSV, hooks, rows, manifest, counts; **PASS**, 200/200
scored, 0 failed, **C1 max |Δ| = 0.0** so the PDMS formula reproduces on every row),
`raw/analysis_navtest_sub200.json`, `raw/A1_sub200_tokens.json` (the selection + token→log map),
`raw/bridge_navtest/seam_A1_sub200.npz` (+ manifest; `partial: true`, `n_requested: 12146`).
The seam came from the live pass's own `part_00000` via **`--assemble-only`** — no model, no GPU,
60 s of scoring — which is the recovery path proven by `tests/test_bridge_assemble_only.py`.

### 3.6 ⭐ THE FULL SPLIT — refcv4b `A1_ego_cmd` on 12,146 navtest tokens: **BAR-W3-M1 FAILS**

**Evidence class MEASURED** · tier **T1-family** · loop **OPEN** · background **non-reactive
(logged)** · 12,146 tokens / 136 logs · CPU, one worker, 10.51 h bridge + 20.4 min scoring ·
**PASS** on every scoring guard (12,146/12,146 scored, 0 failed, **C1 max |Δ| = 0.0**) · seam
`raw/bridge_navtest/seam_A1_ego_cmd.npz` (all 12,146 rows on CPU, `mixed_device: false`; the gate
record carries `NO_GAP_CPU` → `cpu_regate: CPU, no training process is alive`).

| arm | PDMS | NC | DAC | EP | TTC | C | DDC¹ |
|---|---|---|---|---|---|---|---|
| CV | 20.6517 | 68.02 | 57.84 | 19.44 | 50.03 | 100.00 | 79.91 |
| STOP | **61.8202** | 97.40 | 96.53 | 31.00 | 96.40 | 69.44 | 99.59 |
| **refcv4b A1** | **58.9738** | 91.35 | **73.46** | **52.37** | 82.54 | 98.02 | 89.21 |
| HUMAN | 94.5514 | 100.00 | 100.00 | 86.96 | 100.00 | 99.90 | 98.83 |

¹ DDC carries weight 0 in PDMS_v1; reported, never scored.

**The pre-registered bars (SPEC §7), reported as written:**
* ⛔ **BAR-W3-M1 — FAIL.** `A1 > max(STOP, CV)`: 58.9738 < 61.8202. Paired on identical tokens
  with W2's registered estimator: `A1 − CV` = **+0.3832** (log level [+0.3585, +0.4086], drive
  level [+0.3489, +0.4215]) → **separated**; `A1 − STOP` = **−0.0285**, log level
  [−0.0497, −0.0077] (excludes 0), drive level [−0.0567, +0.0029] (does not) → **not separated
  under the registered log-AND-drive conjunction**. No reading of either interval puts A1 above
  STOP; the best case is a tie.
* ⛔ **BAR-W3-M2 (stretch) — FAIL.** A1 < the published Ego Status MLP (65.6 Tab. 1 / 66.4 Tab. 3,
  arXiv 2406.15349, library key banked).
* The 200-token provisional reading (§3.5) **replicated at 60× the n**: 57.9689 → 58.9738, and
  the same shape below.

**Where the 2.85 points go — the shape replicates** (`raw/gate_decomposition_A1.json`):
* Head to head against STOP, refcv4b **wins 6,680** of 12,146 (W/T/L **6,680 / 1,312 / 4,154**),
  **median +4.5319**, mean −2.8464.
* The whole deficit is the `NC·DAC` gate: refcv4b zeroes **3,828** tokens (DAC **3,224**, NC 903)
  against STOP's **732**. **On the 8,318 it does not zero: 86.114 vs 64.577.**
* DAC failure is **speed-gated**: **2.0 %** below 1 m/s (n=1,439), 17.7 % at 1–4, 33.5 % at 4–8,
  **38.0 %** above 8 m/s; spread over **119 of 136 logs** (not a per-log defect).
* It is **LATERAL**: on DAC-failing tokens the 4 s heading error is **11.21°** vs 2.97°, the
  cross-track **1.539 m** vs 0.283 m (**5.4×**), while FDE is 4.22 vs 2.86 m (1.47×).
  Progress (named columns): refcv4b **21.30 m** vs the PDM-Closed reference **30.16 m** on failing
  tokens, 14.22 vs 17.25 m on compliant ones — faster scenes, relatively SHORTER plans.
* **The four families say the same thing from the suite's own instrument**
  (`raw/artifact_A1_navtest.json`, criteria check **0 violations**, registry 2.10.3):
  | family | refcv4b A1 | CV (for scale) |
  |---|---|---|
  | longitudinal — speed MAE / bias | 1.030 m/s / **−0.634** | 1.285 / +0.179 |
  | lateral — heading MAE | 8.08° | 8.48° |
  | lateral — **yaw-rate MAE** | **11.93 °/s** | 4.54 °/s |
  | lateral — **curvature MAE** | **0.0912 1/m** | 0.0200 1/m |
  | lateral — cross-track MAE | 0.580 m | 1.081 m |
  | tactical | OK (trajectory-derived, T1) | OK |
  | strategic | UNAVAILABLE — NavSim scores no strategic decision (the adapter's `navsim_specific_reason`) | same |
  refcv4b is **closer to the human path than CV** (cross-track 0.58 vs 1.08 m) and **slower**
  (−0.63 m/s bias), but its **curvature error is 4.6× and its yaw-rate error 2.6× CV's** — CV is
  straight by construction, refcv4b's paths WIGGLE.
* **The wiggle, measured directly on the seam:** the median total |Δheading| along a plan is
  **26–32°** against a net change of 3–6°, and **87–93 %** of plans flip the sign of their heading
  change (the human: 4.6–11.2° total, 37–62 %). On DAC-failing tokens refcv4b also **under-turns**
  (net 5.74° vs the human's 10.05°, 200-token read).
* ⚠️ **A second, smaller mechanism, and it belongs to the bridge's conversion, not only the
  model:** **176** plans carry |heading| > π (up to ±12.4 rad) and **332** have a heading more than
  30° off their own direction of travel. E2's `knots_to_navsim` fits an **interpolating**
  not-a-knot spline through the model's knots and takes the tangent as heading, so erratic knots
  become looping paths and spinning headings. Those plans fail DAC **74 %** / NC **58 %** of the
  time — but they are only **4–6 % of all DAC failures**; **94.4 %** occur on plans whose heading
  is consistent. Flagged to E2 as a finding, not as a bug in its declared method.

⭐ **Rule Zero — the next lever, and it is already running.** The arithmetic names it: the deficit
is the gate, the gate is DAC, DAC is lateral, and the lateral error is at least partly **jitter**.
Jitter is the one component removable **without training**, so the cheapest experiment that could
still make this arm pass is a fixed smoother — pre-registered as **SPEC AMENDMENT A2 (§7b)** at
**07:35:12Z, before any smoothed number existed**, with both controls and all three outcomes
committed. §3.7 is its result.

### 3.7 ⛔ SPEC §7b — the zero-training smoother arm `A1S`: **FAILS, outcome 3 — jitter is NOT the mechanism**

Pre-registered at **07:35:12Z** (staged 07:35:13Z, blob `0d2ed0c4…`) with no smoothed number in
existence; scored once; nothing tuned.

**Controls — both PASS:**
* `HUMANseam` (the logged human future through the SEAM path, unsmoothed, 200 tokens × 6 columns)
  reproduces the devkit HumanAgent to **max |Δ| = 6.7 × 10⁻⁸** — PDMS 94.1195 = 94.1195. ⚠️ The
  SPEC said "exactly"; it is exact to **float32** (the seam stores float32 poses), not bit-for-bit,
  and I report it that way rather than rounding it into a 0.
* `HUMANS` (the human path through the SAME smoother): **93.6131** vs 94.1195, Δ **−0.51** — inside
  the pre-registered 1.0 tolerance (136 tokens moved by ~0.005, one zeroed by DAC). The smoother
  is not destructive, so the A1S reading is interpretable.

**The arm** (12,146 tokens, PASS, C1 0.0; `raw/A1S_navtest/`, `raw/gate_decomposition_A1S.json`):

| | PDMS | NC | DAC | EP | TTC | C | zeroed (NC/DAC/either) |
|---|---|---|---|---|---|---|---|
| A1 | 58.9738 | 91.35 | 73.46 | 52.37 | 82.54 | 98.02 | 903 / 3,224 / 3,828 |
| **A1S** | **58.1502** | 91.12 | **73.02** | 51.97 | 81.81 | 97.83 | 930 / **3,277** / 3,925 |

* `A1S − STOP` = **−0.0367**, log [−0.0581, −0.0156], drive [−0.0648, −0.0055] → **separated**:
  the smoothed arm is significantly WORSE than doing nothing. `A1S − CV` = +0.3750, separated.
* Against A1 per token: A1S wins **5,136** and loses **959** — yet its mean falls **−0.82**,
  because the wins are small (comfort, progress) and the losses are gate ZEROS: **154** DAC
  failures fixed, **207** created.
* On the DAC-failing tokens the lateral error is **unchanged** (heading 11.54° vs 11.21°,
  cross-track 1.525 vs 1.539 m).
⇒ **Pre-registered outcome 3: FAIL, and DAC zeros did not fall — jitter is not the mechanism; the
smoother is REFUTED as a lever.** The wiggle is real, but it is not what puts the car off the road.

**What the refutation localised, at zero compute** (`raw/A1_by_driving_command.json`; NavSim
command one-hot order left/straight/right/unknown, verified by E2's `tests/test_command_order.py`;
net heading from POSITIONS only, so the spline-heading pathology cannot touch it):

| NavSim command | n | A1 | STOP | HUMAN | A1 DAC0 | turns: model / human net heading | same sign |
|---|---|---|---|---|---|---|---|
| **STRAIGHT** | 8,070 | **63.96** | 62.97 | 94.90 | 21.1 % | **0.149** (n = 755 human turns > 10°) | 76.7 % |
| LEFT | 2,501 | **47.24** | 58.78 | 94.39 | **39.8 %** | 1.070 (n = 2,169) | 90.6 % |
| RIGHT | 1,575 | **52.03** | 60.76 | 93.00 | 33.4 % | 0.699 (n = 1,249) | 84.0 % |

⭐ **refcv4b BEATS the do-nothing floor on two thirds of navtest** — every STRAIGHT-command scene —
and the whole loss is in the turns. Two distinct defects:
1. **It does not follow road curvature under a follow command**: where the logged path turns
   > 10° on a STRAIGHT command, refcv4b turns **0.149×** as much.
2. **At commanded turns it turns the right AMOUNT but in the wrong PLACE**: LEFT net heading ratio
   **1.07**, same sign 90.6 %, and still **39.8 %** DAC failure — the turn's geometry (where and
   how tight), not its size. RIGHT is under-turned (0.70).
Both are model-side. Which part of the model — does vision carry the road geometry at all on
NavSim frames? — is exactly what E2's pre-registered frames-blind arm answers, so it is the next
lever, and it is **running** (SPEC AMENDMENT A3, §7c, registered 08:00:16Z before any A4 number).

### 3.8 SPEC §7c — E2's frames-blind arm `A4`: **vision HELPS by +34.7 PDMS**, and the pre-registered curvature metric MISREAD it

`A4_blind_ego_cmd` = every frame one constant grey, ego + command exactly as A1 (E2's registered
deliberate regression). **957 tokens** (771 STRAIGHT-command curve tokens + the 200 random, 14 in
both), CPU, 2.83 s/scene; scoring **PASS** (957/957, C1 0.0). Registered at **08:00:16Z**
before any A4 number existed.

**Question 2 (pre-registered) — is vision helping or hurting overall?** Paired with W2's estimator
(`raw/paired_A1_minus_A4.json`):
| token set | A1 | A4 (blind) | STOP | `A1 − A4` | log CI | drive CI |
|---|---|---|---|---|---|---|
| 200 random | 57.9689 | **23.2941** | 62.5812 | **+0.3467** | [+0.2659, +0.4256] | [+0.2640, +0.4194] |
| 771 curves | 39.9708 | **17.9247** | 57.9411 | **+0.2205** | [+0.1648, +0.2727] | [+0.1561, +0.2663] |
⇒ **Vision helps — separated at both levels, on both sets.** Without it the model collapses to CV's
level (DAC 41.5 % on the random set).

**Question 1 (pre-registered) — does vision carry road curvature?** On the 771 curve tokens the
pre-registered statistic, the median model/human net-heading ratio, reads **A1 0.1487, A4 0.6664**:
`A4 > A1 + 0.05`, whose committed reading is *"vision actively HURTS curvature following."*
⛔ **That is the reading as registered, and it is WRONG — the same tokens refute it:**
| on the 771 curve tokens | A1 (vision) | A4 (blind) |
|---|---|---|
| turns the WRONG way (ratio < 0) | **23.3 %** | 41.6 % |
| over-turns (> 1.5×) | 13.6 % | 36.8 % |
| barely turns (\|ratio\| < 0.25) | **60.8 %** | 12.7 % |
| follows the curve (0.5–1.5×) | 11.4 % | 13.9 % |
| **drivable-area compliance** | **55.12 %** | 28.02 % |
The blind arm turns HARD in near-random directions; a median of a SIGNED ratio lands inside that
large positive tail and reads "turns more" as "follows better". The quantity that matters on the
same tokens — staying on the road — is **55 % with vision against 28 % blind** (STOP 96 %).
**Class of the error, logged in COMMS:** a location statistic over a mixed-sign ratio, registered
without a control that a random-direction arm must fail. The diagnostic's pre-registered reading
is reported as written and marked **REFUTED BY ITS OWN TOKENS**; no bar depended on it.

⇒ **What the diagnostic actually localised:** refcv4b **sees the road** — vision halves its
wrong-way turning and doubles its drivable-area compliance on curves — but it **does not commit to
the curve**: 60.8 % of its curve plans barely turn. It plans timidly where the road bends, and at
commanded turns it turns the right amount in the wrong place (§3.7).

**The deficit, attributed** (`A1 − STOP` = −2.8464 ×100): commanded turns (**33.6 %** of navtest)
contribute **−3.5075**; STRAIGHT scenes contribute **+0.6611**. The turns carry MORE than the
whole loss. ⛔ **A trap, stated so nobody books it as a win:** the admissible-at-inference policy
"STOP at every commanded turn, refcv4b elsewhere" reads **62.4813** — it clears BAR-W3-M1
arithmetically, and it is **REFUSED**: it was selected on the test split, and it passes by exploiting
PDMS_v1's reward for not moving (§3.3), which is gaming the metric, not driving. **BAR-W3-M1 is
gameable by a command-gated stop, and the bar should say so before anyone else finds it.**

### 3.9 Inference variance — MEASURED to be zero on this rig

CLAUDE.md's third variance question — *would another INFERENCE run say this?* — was open for a
decoder that runs `steps=2`. The A1 bridge was re-launched on the 200 random tokens a day after the
first pass (`raw/bridge_navtest_replicate/`): **200 / 200 plans bit-identical, poses AND knots,
max |Δ| = 0.0** (`raw/A1_inference_replicate_sub200.json`). On CPU the arm is deterministic, so the
intervals above answer the episode (log) question and there is no inference floor to clear.
⚠️ **Training variance is NOT addressed** — there is one checkpoint and no replicate training
run; `H-ESTIM-SEED-1` applies to any claim that a TRAINING lever moved these numbers. W3 trains
nothing, so W3 makes no such claim.

## 4. Blocked, with what unblocks each (Rule Zero)

| # | lever | blocked on | unblock |
|---|---|---|---|
| 1 | the CV / HUMAN reproduction verdict | ✅ **DONE** — BAR-W3-1 PASS (CV, convention named), BAR-W3-2 CLOSE (HUMAN) | — |
| 2 | **Ego-Status-MLP (SPEC BAR-W3-3)** | a **download approval**: `autonomousvision/navsim_baselines/ego_status_mlp/ego_status_mlp_seed_{0,1,2}.ckpt`, **6,518,594 / 6,518,530 / 6,518,530 B** (HF API listing 2026-09-19; sha256 per file known). These are the checkpoints the v1.1 leaderboard row 66.4 ± 0.9 was populated with | a PI/user approval to download them to D:; then `run_v1.py score --arm EGO_MLP:<ckpt>` — the devkit's own agent, **no training** |
| 3 | refcv4b on navtest (BAR-W3-M1: A1 > max(STOP, CV)) | ✅ **DONE — FAIL** (58.9738 vs 61.8202, §3.6); BAR-W3-M2 FAIL. Zero-training levers run and ELIMINATED: jitter (A1S, §3.7, pre-registered outcome 3), bridge/seam path (HUMANseam exact to float32), heading pathology (≤ 6 % of DAC failures), inference variance (deterministic, §3.9); vision confirmed HELPING (+34.7, §3.8) | ⛔ **what remains needs TRAINING — a PI decision:** curvature commitment on curving roads under a follow command (0.149× the human's turn, 60.8 % barely turn) and the geometry of commanded turns (LEFT turns the right amount in the wrong place, DAC fail 39.8 %). Cheapest discriminating runs that need no training but ~10 h CPU each or a GPU gap: E2's A3 (no command) on the LEFT/RIGHT tokens, and A2 (vision-pure) — not started, because they attribute rather than repair |
| 4 | a faster cache | the ONE-worker rule. MEASURED: 1.47 s/scene on one worker; the box has 24 logical CPUs and the devkit ships `worker=ray_distributed` | an orchestrator/PI decision to raise the worker count |
| 5 | a BAR-W3-M1 that cannot be gamed | the bar as written is passed by "STOP at every commanded turn, refcv4b elsewhere" (**62.4813**, refused, §3.8) | an orchestrator/PI decision to add a no-stop-at-turns (or EP-floor) clause to the model bar before the next model is scored |

## 5. Where everything is (see the manifest in the report)

Code `code/` (wrapper, driver, agents, export, analysis, artifacts, frame bank, bank finalize,
bridge, chain), tests `tests/` (4 files, 21 tests), evidence `raw/` (smoke runs, controls,
artifacts + criteria, chain log), the CLI plugin at
`taniteval/taniteval/bench/plugins/navsim_v1.py`, and the D:-resident data artifacts under
`D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/` (metric caches, `inputs/`, frame banks).
