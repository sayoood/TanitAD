# Colab compute-unit usage log

One row per metered job (TANITAD_PROGRAMME.md: "log every metered use"). CU = the live `consumptionRateHourly` read
by `colab/ccu_info.py` while assigned x the assigned time; the account balance lags and rounds, so it is not the
per-job evidence. Account: fambouzouraa@gmail.com. Balance before the first row: 200 CU.

| date (Berlin) | job | GPU | assigned | rate CU/h | ~CU | launched by | evidence |
|---|---|---|---|---|---|---|---|
| 2026-09-27 16:57 | GPU survey: smoke | L4 | 44 s | 1.54 | 0.02 | orchestrator (research-lab session) | `raw/2026-09-27-gpu-survey/survey-l4.json` |
| 2026-09-27 16:58 | GPU survey: smoke | G4 | 42 s | 8.9 | 0.10 | orchestrator | `raw/2026-09-27-gpu-survey/survey-g4.json` |
| 2026-09-27 16:59 | GPU survey: A100 assign refused (503, no capacity) | A100 | 0 s | - | 0 | orchestrator | `raw/2026-09-27-gpu-survey/survey-a100.json` |
| 2026-09-27 17:00 | GPU survey: smoke | T4 | 65 s | 1.07 | 0.02 | orchestrator | `raw/2026-09-27-gpu-survey/survey-t4.json` |
| 2026-09-27 17:49 | relay proof: Colab-Secret probe (no reply in 30 s: headless sessions cannot read Colab Secrets); pull not run | CPU | 63 s | 0.16 for 2 assignments (~0.08 each) | 0.003 | orchestrator | `raw/2026-09-27-hf-relay/relay-proof.json` |
| 2026-09-27 17:54 | signed-link proof: snap_epoch016.pt pulled with no token on the VM, md5 verified | CPU | 44 s | 0.08 | 0.001 | orchestrator | `raw/2026-09-27-hf-relay/signed-proof.json` |
| 2026-09-27 19:01 | NAVSIM-on-Colab P0, session B: setup (public data + relay pull) + G-ENV, G-H, G-S seam, G-P, G-REP on snapshot 016 sub200 (`navtest/p0_session.ps1`) | L4 | ~778 s (17:01:27Z-17:14:25Z; the script's own stop did not run -- stopped by hand and checked on the server, `raw/2026-09-27-navtest-p0/navtest-p0-l4/manual_stop.txt`) | 1.54 (read 1.62 = L4 + the concurrent 0.08 CPU session A) | 0.33 | navtest P0 agent | `raw/2026-09-27-navtest-p0/navtest-p0-l4/session_record.json` |
| 2026-09-27 18:53 | NAVSIM-on-Colab P0, session A: setup (public data + relay pull) + G-H, G-H' (VM-built cache + 3 floors), G-E6 (64 harness runs, 2 workers), the G-H levers (determinism rerun, float probe) on snapshot 016 sub200 (`navtest/p0_session.ps1`) | CPU | ~3,310 s (~16:53:50Z-17:49:05Z; stopped by a watcher, since the script's own stop had the defect found in session B; server then read `assignments: []`, rate 0) | 0.08 | 0.074 | navtest P0 agent | `raw/2026-09-27-navtest-p0/navtest-p0-cpu/session_record.json`, `.../manual_stop.txt` |
