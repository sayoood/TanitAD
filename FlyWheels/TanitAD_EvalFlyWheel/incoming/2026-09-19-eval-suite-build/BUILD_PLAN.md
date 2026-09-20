# BUILD PLAN — the one-command TanitAD benchmark suite, NAVSIM v1 + v2, nuScenes, a self-regenerating leaderboard, and high-quality visual reporting

**Commissioned by the PI in chat, 2026-09-19:** *"build what is missing for a one-command suite,
NavSim v1, nuScenes, and a leaderboard that regenerates itself. I want you also to add high quality
visualizations and reporting to the eval suite."*
**PI decisions the same turn:** (1) NAVSIM v1 download **YES, D: drive only**; (2) navhard inference
for our models **waits for a GPU gap**; (3) **no submission** to any official leaderboard until the PI
approves. **Owner:** TanitAD_EvalFlyWheel. **Schema:** `TANITAD_PROGRAMME.md` §3 per package.

## 0. What already exists (MEASURED 2026-09-19 — build ON it, never beside it)

| asset | where | state |
|---|---|---|
| NAVSIM v2 devkit `0a380a9` (post-#151) + Windows patches | C: runtime `C:/Users/Admin/navsim-crun` (venv, devkit, maps, openscene) | ✅ validated: CV warmup combined EPDMS **18.535627** = HF LB **18.5356** |
| E1 wrapper (reader separator patch, hooks, counters) | `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-19-navsim-warmup-reference-epdms/code/navsim_win.py`, `run_*.sh`, `build_artifacts.py` | ✅ unit-tested |
| E2 bridge (ckpt → declared inputs → seam → official scorer; STOP/CV/ECHO floors; navhard-ready frame builder; controls) | `…/2026-09-19-navsim-refcv4b-bridge/code/` (12 files), `tests/` (3) | ✅ refcv4b scored on warmup S2 |
| navhard_two_stage data | archives D: `D:/Archive/devbox-C/navsim/data/navsim-v2/` (sha256 == HF ETag); 3-cam working copy C: `…/navsim-crun/data/openscene/navhard_two_stage` (`EXTRACT_DONE.json ok`) | ✅; metric cache building after A8 (agent-free waiter) |
| NAVSIM v1.1 code | `D:/Archive/devbox-C/navsim/navsim-3e8291bfa89ff247231e0227778840cd0a036896/` (v1.1.0, pinned 2025-06-05) | ✅ unpacked, NOT installed |
| navtest camera shards (32, **127,882,665,618 B**) | downloading → `D:/Archive/devbox-C/navsim/data/openscene-v1.1/openscene_sensor_test_camera/`; receipt `…/2026-09-19-navhard-download/raw/receipt_navtest_camera.json` | ⏳ sha256-verified per shard as it lands |
| OpenScene test metadata (`navsim_logs/test`) + nuPlan maps | D: `…/navsim/data/openscene/navsim_logs/test`, `…/data/maps` | ✅ |
| TanitEval adapter / criteria | `taniteval/adapters/navsim.py` (build_artifact …), `products/P7-TanitEval/CRITERIA_REGISTRY.json` v2.9.0, `tools/criteria_check.py` | ⚠️ checker does NOT evaluate `benchmarks.navsim`; `EPDMS_v2` multipliers omit TLC |
| internal T1 panel (four families, episode-cluster bootstrap) | `taniteval/tools/refcv3_arm.py`, `four_families` | ✅ REF-C incl. refcv6 |
| leaderboard | `Benchmarks & Eval/LEADERBOARD.md` (hand-maintained, ~1,700 lines) + E4's audit `…/2026-09-19-leaderboard-currency/raw/audit_eval_packages_since_0903.md` | ⚠️ no External Field, not regenerable |
| published results (INHERITED until re-read) | Lab packages `…/Benchmarks & Evals/Research/2026-09-1{0,3,5,7}-*` (navhard: CV 11.4 · Ego-MLP 14.1 · LTF 25.1 · PDM-C 56.6 · DrivoR 54.5 / +TOAD 56.3, post-#151); registry `benchmarks.navsim.*` (v1 perception-free ladder LAW 83.8 · World4Drive 85.1 · Epona 86.1 · Drive-JEPA 89.0) | ⚠️ re-read from banked PDFs |

## 1. THE CONTRACT every package builds against (so six streams interoperate)

**One command:**
```
python -m taniteval.bench <benchmark> --ckpt <path|none> --split <split> [--arms A1,STOP,CV,...] [--device auto|cpu|cuda]
    benchmark ∈ {navsim_v2, navsim_v1, nuscenes_ol, internal_t1}
python -m taniteval.leaderboard build          # regenerates the leaderboard from results on disk
python -m taniteval.report <run_dir>           # (re)renders one run's report; bench calls it at the end
```
**Run directory** `taniteval/results/bench/<benchmark>/<split>/<run_id>/` (small files in git; anything
> 20 MB off-repo with its sha256 in the manifest):
* `bench_run.json` — `run_id`, `utc`, `git_head`, `ckpt {path, sha256, registry_key}`, `benchmark`,
  `protocol` (closed set, e.g. `EPDMS_v2_navhard_two_stage`, `PDMS_v1_navtest`, `nuScenes_OL_L2_stp3`),
  `devkit {repo, sha, patches[]}`, `split {name, n_scenes, n_logs}`, `arms[]` (each: `name`, `kind` ∈
  {model, floor, reference}, `declared_inputs`), `device`, `wall_s`, `claim_bearing` (false for nuScenes).
* `scores/<arm>.csv` — the devkit's own per-token output, unmodified.
* `artifacts/<arm>.json` — TanitEval artifact (`taniteval/adapters/navsim.py::build_artifact` for NavSim).
* `criteria/<arm>.txt` — `tools/criteria_check.py` output.
* `summary.json` — per arm: headline (`score` column, NEVER `pdm_score`), every sub-metric, per stage,
  per log, the paired deltas vs **STOP** and **CV** (both MANDATORY floors on every NavSim run), and
  the interval **only if** the estimator admits it (else `{status: UNAVAILABLE, reason, n}`).
* `report/index.html` + `report/fig/*.{svg,png}` — W5.

**Binding rules the suite enforces, not just documents:** tier + loop stamp on every number (NavSim:
T1-family, S1 loop OPEN, S2 UNRULED, IDM-reactive traffic); four families reported or refused per
family with reason + n; protocol tag from the closed set and never two protocols in one column;
STOP + CV floors on every NavSim run; ego-status enforcement evidence (declared-input manifest +
mutation); **submission commands are REFUSED** unless `--pi-approval <decision-id>` names a recorded PI
decision (none exists: PI 2026-09-19 *"dont submit now until I approve"*); GPU use only through the
**gap launcher** (no training process alive AND GPU memory used < 1 GB, re-checked every minute; it
backs off the moment either changes) — PI 2026-09-19 *"wait for a gap in the gpu"*.

## 2. WORK PACKAGES (priority order; each owns its files EXCLUSIVELY)

| id | package dir (`FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-20-…`) | owns (exclusive write) | deliverable |
|---|---|---|---|
| **W1** | `suite-core-navsim-v2` | `taniteval/taniteval/bench/**` (new), `taniteval/tests/test_bench_*.py` | the CLI + run-dir contract; NAVSIM v2 end to end by PROMOTING E1/E2 code (not rewriting it); STOP/CV/ECHO floors; the GPU-gap launcher; the submission refusal; `internal_t1` wrapping `refcv3_arm.py`; acceptance: `python -m taniteval.bench navsim_v2 --ckpt none --split warmup_two_stage --arms CV,STOP` reproduces **18.5356** and E2's STOP **0.3009** bit-for-bit from a clean shell |
| **W2** | `estimator-and-gates` (E3 resumed) | `taniteval/adapters/navsim.py`, `products/P7-TanitEval/CRITERIA_REGISTRY.json`, `tools/criteria_check.py`, `tools/tests/test_criteria_check.py` | (a) `criteria_check.py` EVALUATES `benchmarks.navsim` (all four blocking gates) with a mutation arm per gate that must go RED; (b) `EPDMS_v2` multipliers gain TLC; (c) the log-cluster bootstrap (navhard 76 logs, navtest ≥ 8) reproducing the official two-stage aggregation, analytic literal tests + RED mutation; (d) the route-leak check at source (E3 Part B); (e) a `navsim_v1` benchmark block (PDMS_v1: NC·DAC·(5EP+5TTC+2C)/12, v1.1 SHA `3e8291b`) |
| **W3** | `navsim-v1-navtest` | its package; the v1 runtime under `D:/Archive/devbox-C/navsim/` (**D: ONLY — PI**) | v1.1 runtime (first try the existing C: v2 venv with `PYTHONPATH` → the v1.1 tree, asserting `navsim.__file__`; only if incompatible build a venv **on D:**); navtest metric cache **on D:**; CV / human / ego-status-MLP PDMS reproduced against the NAVSIM paper's own table (banked `2406.15349` — re-read it); the navtest frame bank (CAM_F0/L0/R0 only, built from the verified shards, extracted per shard and cleaned up — exFAT has 1 MiB clusters); plugs into W1's CLI as `navsim_v1`; refcv4b on navtest queued behind the GPU-gap launcher |
| **W4** | `leaderboard-generator` (E4 resumed) | `Benchmarks & Eval/LEADERBOARD.md`, `taniteval/taniteval/leaderboard/**`, `products/P7-TanitEval/benchmarks/published_results.json` (new) | `published_results.json`: every external number with protocol tag, library key, table/page, modality, ego-status use, harness version — re-read from banked PDFs (bank missing ones); the generator writes a marker-delimited section of `LEADERBOARD.md` (`<!-- BENCH:BEGIN -->…<!-- BENCH:END -->`, hand-written content outside the markers untouched) + `Benchmarks & Eval/leaderboard.html`; one table PER protocol; our rows from `summary.json`s with floors, tier/loop, CI-or-UNAVAILABLE; the "5 October position" block; E4's currency audit folded in; acceptance: delete the section, run `build`, get it back byte-identical |
| **W5** | `visual-reporting` | `taniteval/taniteval/report/**`, `taniteval/tests/test_report_*.py` | publication-quality per-run HTML reports + figures (load the `dataviz` skill FIRST): headline vs floors and published bars per protocol; sub-metric breakdown per stage; per-log and per-speed-band panels; paired win/tie/loss vs CV and STOP; four-family panels; a **failure gallery** of the worst scenes: BEV on the nuPlan map (lanes/drivable from the local maps, agent boxes) with our plan vs human vs CV vs STOP, beside the stitched 3-camera frame with the plan projected, text overlay (command, decoded manoeuvre, failing sub-metrics) — the programme's standing viz standard (camera projection + metric BEV inset + text overlay); leaderboard charts for W4 (rank bars per protocol, params-vs-score scatter, our row highlighted); light + dark safe palette; build and test it against E1/E2's EXISTING run outputs as fixtures |
| **W6** | `nuscenes-ol-harness` (E5 resumed) | `taniteval/adapters/nuscenes_planning.py` (+ tests), its package | L2 / collision @1/2/3 s in BOTH conventions (ST-P3 / UniAD, verbatim to pinned reference code), `claim_bearing: false` by API, literal analytic tests + RED mutation; the nuScenes → our-frame input adapter (CAM_FRONT pinhole → the cylindrical frame with its observed mask, never a resize); plugs into W1 as `nuscenes_ol`; published rows (banked) into W4's `published_results.json` via a patch file; the **PI ACTION LIST** (registration steps + minimal files + sizes, D: landing path) |

**Order of landing:** W1 first (the contract others plug into) but W2–W6 start in parallel against §1
and E1/E2's existing outputs. Integration requests go to the EvalFlyWheel orchestrator, never into a README.

## 3. Constraints (all packages)

CPU only unless through W1's GPU-gap launcher · never touch a training process · A7 trains on this
box overnight and reads D: — keep D: I/O modest (except W3, whose D:-only placement is the PI's
explicit decision) · weekly usage is a hard ceiling: no sub-agents, bank incrementally · stage exact
paths, never commit/push · four families, tier/loop, estimator rules as in `CLAUDE.md` · every number
carries its evidence class · never submit.
