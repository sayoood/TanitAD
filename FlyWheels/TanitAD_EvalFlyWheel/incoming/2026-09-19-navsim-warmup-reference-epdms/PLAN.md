# PLAN — E1 (executed order; deviations marked)

| # | step | command / file | status |
|---|---|---|---|
| 0 | read devkit + our protocol/adapter/registry; derive controls from source | SPEC.md §4 | done 11:10 |
| 1 | pre-register | `SPEC.md` (written before any scoring) | done 11:10 |
| 2 | metric cache, warmup | `bash code/run_warmup_reference.sh cache` → `C:\Users\Admin\navsim\exp\metric_cache_warmup_two_stage` | see below |
| 3 | verify cache by content (C2, C3, loader defect) → **`CACHE_DONE.json`** (atomic) | `code/verify_cache.py --write-marker` | — |
| 4 | score A1 (CV, two-stage), R1 (replicate), A2 (human, two-stage), A1b/A2b (one-stage, IDM), M1 (filter-off mutation) | `bash code/run_warmup_reference.sh <arm>` | — |
| 5 | controls C1, C3–C8 | `code/verify_controls.py` | — |
| 6 | TanitEval artifacts + `tools/criteria_check.py` + my NavSim-gate evaluator | `code/build_artifacts.py` | — |
| 7 | navhard price (ESTIMATED from step 2/4 per-scene timings) | `code/price_navhard.py` | — |
| 8 | **priority 2 (coordinator, PI-authorised):** navhard cache + official CV/human scoring + published cross-check | see below | in flight |

## navhard (priority 2) — what actually ran

| # | step | outcome |
|---|---|---|
| 8.1 | pricing from warmup per-scene costs | ESTIMATED 3.80 h cache + 3.63 h CV (raw/navhard_price.json) |
| 8.2 | mirror verify + 76 logs copied, sha256 | 0 mismatches |
| 8.3 | cache (orchestrator's 1-worker C: variant of `run_navhard.sh`, 5,187 s) | 5,912 entries |
| 8.4 | **E1 content-verified the cache** (the chain had not) → `CACHE_DONE.json` | 450 + 5,462, sets == yaml, 0 type mismatches |
| 8.5 | N1 try 1 (overnight) | ⛔ rc 1 in the aggregation, no CSV; **verified unrecoverable** |
| 8.6 | diagnose → zero-length `path_to_go` ⇒ empty flat-cap buffer (shapely 2.0.7) | `raw/navhard/diag/idm_assert_diag.json` |
| 8.7 | patch the C: **copy** (assert condition only) + verify effective AND inert | 3/3 fixed, 4/4 bit-identical |
| 8.8 | N1 try 2 (mine, sequential) | 0 agent failures in 5,576 scenes; ⛔ **my RAM guard aborted at 95 %** (another process's dip); all 450 stage-1 rows survived in the hooks |
| 8.9 | **stage-1 vs published**, from those banked rows | **8/8 sub-metrics match [N2]v3 Tab. 2 under truncation** |
| 8.10 | guard policy `sustain` 3 → 60 samples; W1 asked to re-promote | done |
| 8.11 | N1 try 3 **through W1's suite** (`taniteval.bench navsim_v2 … --arms CV,STOP --ram-floor-mb 2400`) | in flight; carries pre-agg dump + settled interval + artifact + gates |
| 8.12 | human stage-1 arm (N2b), controls, published comparison, artifact | after 8.11 (one worker) |

## ⚠️ Deviation 1 — the runtime moved to a verified C: mirror (11:40–12:03)

**Why (MEASURED):** the navsim venv, devkit, data and exp dir all live on the external exFAT D:
(`C:\Users\Admin\navsim` is a junction). With the navhard sensor download (38.3 GB) writing to D:,
`import numpy` took **23,350 ms** from the D: venv vs **324 ms** from a C: venv; cache attempt 1
was still importing after **469 s** (194 MB RSS, 3.7 s CPU in 8 min).

**What the mirror is** (`C:\Users\Admin\navsim-crun`), and how each part was verified:
* venv — `uv pip install --offline` of the exact 200 pins of `uv pip freeze` of the D: venv, from
  uv's C: cache (**nothing downloaded**; `--offline` forbids network); `raw/navsim_venv_freeze.txt`.
* devkit + nuplan-devkit — robocopy of the D: worktrees; **every git blob re-hashed**: navsim 198/199
  equal HEAD, the 1 diff = the pre-existing patched `dataclasses.py` blob `596cb7d`; nuplan 1321/1324,
  diffs = pre-existing `setup.py` blob `3ad18c6` + 2 non-runtime files byte-identical to the D:
  worktree (git eol normalisation). `raw/devkit_copy_verify.json`.
* the pre-existing `fcntl.py` shim (not visible to `pip freeze`; found by diffing site-packages) —
  copied, sha256 `75184a48…`.
* data — 4 city maps + index json, the 7 warmup logs, 204 synthetic pickles: **218 files sha256-verified,
  0 mismatches** (`raw/data_mirror_verify_warmup.json`, `raw/data_mirror_verify_lasvegas.json`).
* **outputs** (cache, CSVs) stay on the canonical D:-backed `C:\Users\Admin\navsim\exp` that E2 polls.
* every run's manifest asserts WHICH tree was imported (`imported_from`).

Attempt log: #1 aborted by my RAM guard on a transient dip caused by other processes (my RSS 194 MB)
→ guard changed to "sustained 3 samples or < 2,000 MB once" (`raw/cache_attempt1_aborted_ram_guard/`);
#2 failed in 26 s: nuPlan's `GPKGMapsDB` opens a dummy layer of EVERY map in the index at init, so
Las Vegas is required though no warmup scene uses it (`raw/cache_attempt2_missing_lv_map/`); #3 = the run.
