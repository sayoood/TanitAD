# EvalFlyWheel session plan — 2026-09-19 · "NavSim/nuScenes + leaderboard"

**Commissioned by** the PI (Sayed) in chat, 2026-09-19: *"start the new session for the eval fly
wheel agent … Start TanitAD_EvalFlyWheel: NavSim/nuScenes + leaderboard"*.
**Charter** `Project Steering/AGENT_CHARTERS.md` §5 (owns P7 TanitEval, the binding criteria, the
LEADERBOARD, community benchmarks — *"NavSim, nuScenes … MANDATORY"*).
**Deadline in view** 2026-10-05 first final evaluation — `Project Steering/Master Plan.md` §2 Phase 2
names *"reproducible gate ladder, leaderboard"* as part of the P7 package.

## Where the programme stood at session start (MEASURED this session unless marked)

| item | state | evidence |
|---|---|---|
| NavSim protocol spec + criteria gates | ✅ done — `CRITERIA_REGISTRY.json` v2.9.0 `benchmarks.navsim`: official column **navhard_two_stage EPDMS @ SHA**; blocking gates estimator_unit / ego_enforcement / modality_label / cross_protocol | registry read 2026-09-19 |
| NavSim devkit | ✅ installed at `C:\Users\Admin\navsim` (junction → `D:\Archive\devbox-C\navsim`), devkit **@ 0a380a9**, venv Python 3.9.25, `navsim` + `nuplan` import | `git rev-parse HEAD`, import probe 2026-09-19 |
| NavSim data on disk | ✅ `warmup_two_stage` (204 synthetic scenes), OpenScene **test** metadata, nuPlan maps v1.1 | `ls`/`du` 2026-09-19 |
| a real EPDMS number | ⛔ **none, for any agent** — the 2026-08-29 session produced CV **submission pickles** only, never a local score | `C:\Users\Admin\navsim\exp\tanitad_cv_smoke2\…` |
| NavSim frame bank in our geometry | ✅ 204/204 scenes, `rig_clean` 176×624 + `wide` 256×640 (DataFlyWheel, commit `d17d947`) | package `2026-08-28-navsim-corpus-adaptation` |
| navhard_two_stage bytes | ⛔ not downloaded — archives **12,804,089,546 + 25,521,966,184 + 211,118,643 B = 38.54 GB** (HEAD `X-Linked-Size`, MEASURED 2026-09-19; the earlier "31 GB" is the docs' figure and does NOT match the archives) | HEAD requests to `huggingface.co/datasets/OpenDriveLab/OpenScene/…/navsim-v2/` |
| nuScenes | ⛔ no bytes; a HUMAN must register + accept the ToU (agents must not); OL planning is **inadmissible as a criterion** (H-EVAL-6) and SKIP-claim-bearing (D-BENCH-PORT) | `stack/tanitad/data/nuscenes.py` docstring; register |
| dev-box GPU | ⛔ OCCUPIED by a live `refc_v3_train.py` run (a8-occupancy-5k) — this session uses **CPU only** | `nvidia-smi` + process list 2026-09-19 |

## Streams (priority order — a PRIORITY ORDER, not a dependency chain)

| # | stream | package | owns (exclusive write) |
|---|---|---|---|
| E1 | NavSim harness validation: first real EPDMS on `warmup_two_stage` for the devkit reference agents + the metric cache | `2026-09-19-navsim-warmup-reference-epdms/` | the metric cache dir, its package |
| E2 | TanitAD → NavSim bridge: refcv4b scored by the official harness, ego+cmd arm **paired with a vision-pure arm** | `2026-09-19-navsim-refcv4b-bridge/` | its package |
| E3 | NavSim estimator pre-registration (log-cluster bootstrap) + the `route_leak_check` at source | `2026-09-19-navsim-estimator-and-route-leak/` | `taniteval/adapters/navsim.py`, `products/P7-TanitEval/CRITERIA_REGISTRY.json`, `tools/criteria_check.py`, `tools/tests/test_criteria_check.py` |
| E4 | LEADERBOARD currency for 5 Oct + the External Field section | `2026-09-19-leaderboard-currency/` | `Benchmarks & Eval/LEADERBOARD.md` |
| E5 | nuScenes open-loop planning harness, protocol-verbatim, both conventions; external rows; the measured blocker | `2026-09-19-nuscenes-planning-harness/` | new `taniteval/adapters/nuscenes_planning.py` + its tests |

**Named blockers (PI):** (1) the navhard_two_stage download — 0.21 GB scene pickles (enough for the
reference agents' official-protocol EPDMS) and 38.3 GB sensors (needed only for our camera agent);
(2) nuScenes registration + ToU acceptance (human-only).
