# BRIEF E1 — NavSim harness validation: the programme's FIRST real EPDMS numbers

**From** EvalFlyWheel orchestrator, 2026-09-19 (session "Start TanitAD_EvalFlyWheel: NavSim/nuScenes
+ leaderboard", commissioned by the PI in chat). **Schema** `Project Steering/TANITAD_PROGRAMME.md` §3
(SPEC / PLAN / tests / code / raw / RESULT / COMMS). **Package** = this folder.

## Operating rules (binding)

1. STAGE, NEVER PUSH. `git add` every deliverable into the working tree when you finish — EXACT FILE
   PATHS ONLY, never a directory (the shared index holds other agents' staged work). Do NOT
   `git commit`, do NOT `git push`, do NOT switch branches. Verify: a NEW file must appear in
   `git ls-files --stage -- <path>`; a MODIFIED tracked file needs its index blob
   (`git ls-files --stage -- <path>`) to equal `git hash-object <path>` — require both to be 40 chars,
   else report INCONCLUSIVE. Re-verify at the END of your turn (the index moves under you).
2. END WITH A DELIVERABLE MANIFEST — every artifact and WHERE it lives (`repo:<path>` /
   `devbox:<path>`). Mark anything that exists in only ONE place.
3. ESCALATE INTEGRATION in your report's headline. Never write "please merge" into a README.
4. QUOTE ONLY PRIMARY SOURCES. Model facts from `Project Steering/MODEL_REGISTRY.md` or raw eval JSON;
   published numbers from banked PDFs (`TanitAD Research Lab/Library/`, cite the library key).
5. FAIL LOUD, REPORT HONESTLY. UNVERIFIED beats a guess. Evidence class on EVERY number: MEASURED
   (ours + artifact path) · PUBLISHED (library key + table) · INHERITED · ESTIMATED · HYPOTHESIS.
- Research banking: cite a paper ⇒ bank it: `python tools/kb_add.py <arxiv-id> --tag <topic> --note
  "<finding>" --cited-by "<your RESULT path>"` (files into `TanitAD Research Lab/Library/`; the old
  "Research Hub" path is DEAD). Verify with `python tools/kb_add.py --verify`.
- Keys.txt is git-ignored — never commit, print or copy tokens.

## Environment facts (MEASURED 2026-09-19 unless marked)

- Repo `D:\Projects\TanitAD` (exFAT, canonical). ⛔ Never read/write any `G:\Meine Ablage\…` path.
- TanitAD venv `C:\Users\Admin\venvs\tanitad\Scripts\python.exe`; set `PYTHONPATH=D:/Projects/TanitAD/stack`
  and assert `tanitad.__file__` is under `D:/Projects/TanitAD` (an editable install can silently serve G:).
- NavSim venv `C:\Users\Admin\navsim\venv\Scripts\python.exe` (Python 3.9.25). Devkit
  `C:\Users\Admin\navsim\devkit` **@ 0a380a9** (`git -c safe.directory=* rev-parse HEAD`), nuplan-devkit
  `C:\Users\Admin\navsim\nuplan-devkit`. Env vars in `C:\Users\Admin\navsim\env.sh`. `C:\Users\Admin\navsim`
  is a junction to `D:\Archive\devbox-C\navsim`.
- Data: `C:\Users\Admin\navsim\data\openscene\warmup_two_stage\` (sensor_blobs, synthetic_scene_pickles,
  openscene_meta_datas, synthetic_scenes_attributes.csv); OpenScene test metadata under
  `…\data\openscene\navsim_logs\` and `…\openscene-v1.1\`; maps `…\data\maps` (nuplan-maps-v1.0 layout).
- The 2026-08-29 smoke (`C:\Users\Admin\navsim\smoke2.log`) ran `run_create_submission_pickle.py` with the
  CV agent: 7 logs loaded, **16** first-stage and **204** second-stage scenes. No local score was ever
  computed. `C:\Users\Admin\navsim\smoke_assert.py` refuses "success over an empty set" — reuse it.
- Dev box 16 cores / 31.8 GB RAM. ⛔ The RTX 4060 is OCCUPIED by a live training run
  (`refc_v3_train.py … a8-occupancy-5k`, PIDs 16996/21724, ~6 GB RSS). Do NOT use the GPU, do NOT touch
  that process, cap yourself at ≤ 4 worker processes and ~5 GB RAM, and never let system free RAM fall
  below ~3 GB (`Get-CimInstance Win32_OperatingSystem` → FreePhysicalMemory). A sibling stream (E2)
  will also run the NavSim scorer later — keep your worker count modest.
- Windows console is cp1252: `PYTHONIOENCODING=utf-8` / `PYTHONUTF8=1` for anything printing unicode.
- ⛔ Never read `$?` through a pipe; assert on the ARTIFACT (row counts, files) not exit codes.

## Goal

Produce REAL, locally computed **EPDMS** numbers on the official NavSim v2 harness (pinned SHA) for
the devkit's reference agents on `warmup_two_stage`, and leave behind the **metric cache** that the
TanitAD agent (stream E2) will be scored against. No NavSim number exists in the programme today; the
harness must be validated on known agents BEFORE any TanitAD number is trusted.

## Steps (priority order)

1. **Read first**: devkit `docs/metrics.md`, `docs/splits.md`, `docs/submission.md`, `docs/cache.md`,
   `docs/traffic_agents.md`; `navsim/planning/script/run_metric_caching.py`, `run_pdm_score.py`,
   `run_pdm_score_from_submission.py`, `run_pdm_score_one_stage.py`; configs
   `…/config/common/train_test_split/warmup_two_stage.yaml` and `…/scene_filter/warmup_two_stage.yaml`,
   `…/config/common/agent/*.yaml` (constant_velocity_agent, human_agent, ego_status_mlp_agent,
   transfuser_agent), `…/config/common/worker/*.yaml`. Ours: `products/P7-TanitEval/benchmarks/NAVSIM_PROTOCOL.md`,
   `taniteval/adapters/navsim.py` (read_epdms, submetrics_from_row, estimator_refusal,
   navsim_native_family_refusals, build_artifact), and `products/P7-TanitEval/CRITERIA_REGISTRY.json →
   benchmarks.navsim` (blocking gates). ⛔ Read `score`, NEVER `pdm_score` (H-EVAL-7).
2. **Write SPEC.md BEFORE running**: expected ordering (CV < human), both outcomes committed, and the
   controls that must read known values (step 4).
3. **Metric cache** for `warmup_two_stage` at `C:\Users\Admin\navsim\exp\metric_cache_warmup_two_stage`,
   using a worker that works on Windows (`sequential` or `single_machine_thread_pool` with ≤ 4 workers;
   ray may not). When complete, write `C:\Users\Admin\navsim\exp\metric_cache_warmup_two_stage\CACHE_DONE.json`
   with n cached scenes per stage, wall-clock, devkit SHA and the exact command — **stream E2 polls for
   this marker; do not rename it**. Assert non-empty counts. If a Windows-only defect needs a patch, do
   NOT edit devkit files in place: write the patch into `code/` and apply it through a copy or a
   monkeypatch wrapper, and record it — *the SHA plus any patch is part of the column*.
4. **Score reference agents** with the official two-stage scorer: `constant_velocity_agent` (floor) and
   `human_agent` (log replay — the privileged reference); `ego_status_mlp_agent` ONLY if a trained
   checkpoint exists locally (do not train). Per agent: EPDMS from `score`, ALL sub-metrics (NC, DAC,
   DDC, TLC, EP, TTC, LK, HC, EC), stage-1 / stage-2 / aggregate, and counts. **Controls that must read
   known values** — find what the devkit guarantees and assert it: e.g., the human-filter
   (`filter_m = 1.0 if the human also violated m`) implies specific values for the human agent; counts
   must equal the split size; a CV agent must score EP well below the human. Any control that fails is
   a harness defect and outranks every other result.
5. **Artifacts**: one TanitEval artifact per agent via `taniteval/adapters/navsim.py::build_artifact`
   (or the closest function; ⛔ do not edit that file — stream E3 owns it; if it lacks something, write
   the wrapper in your `code/` and name the gap) with protocol `EPDMS_v2_warmup_two_stage`, devkit SHA,
   estimator `{status: UNAVAILABLE, reason, n}` — ⛔ **warmup NEVER carries a CI** (D-BENCH-PORT: 7 log
   groups < the RG-14 floor of 8), modality/setting labels, tier/loop stamps (below). Run
   `python tools/criteria_check.py` (find its CLI) over each and record PASS/FAIL per gate.
6. **Price navhard from a measurement**: record wall-clock and RAM of cache + scoring per scene, and
   extrapolate to navhard_two_stage (450 stage-1 + 5,462 stage-2 per the Lab — INHERITED; verify
   against the local scene_filter yaml if possible). Mark it ESTIMATED with the per-scene measurement
   it scales.

## Tier / loop stamps (binding vocabulary)

NavSim scores the agent's OWN planned trajectory via a 4 s LQR + bicycle-model rollout against
**non-reactive** logged agents, with perception fixed at t0 → **tier T1-family**, **loop OPEN** under
the PI's 2026-09-02 ruling for stage 1. Stage 2 re-renders a perturbed start (3DGS) — a one-shot
re-perception; its loop status is **UNRULED** under the 2026-09-02 vocabulary: state that, do not rule it.

## Do NOT

download anything (name what is missing and stop that sub-step) · use the GPU · touch the live
training process · edit devkit files in place · edit `taniteval/adapters/navsim.py`,
`CRITERIA_REGISTRY.json`, `tools/criteria_check.py` (owned by stream E3) or `LEADERBOARD.md` (E4).

## Deliverables (this folder)

`SPEC.md` (pre-registered) · `PLAN.md` · `code/` (exact commands/scripts, e.g. `run_warmup_reference.sh`,
any patch) · `raw/` (devkit score CSVs copied in, logs, counters, artifact JSONs, criteria_check
output, timing) · `RESULT.md` (findings FIRST; evidence class + tier/loop on every number; the navhard
cost estimate) · `COMMS.md` (decisions asked/made, integration asks). Stage exact paths; manifest at
the end of your report. Budget: CPU only; stop and report if the cache would exceed ~2 h wall-clock.
