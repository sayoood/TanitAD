# TanitAD Program Report — 2026-07-09 21:50 (report series #1, D-025)

> Written by the MVP orchestrator. Series cadence: 07:57 / 12:57 / 17:57 daily + on demand.
> Everything here is measured; preview-grade numbers are labeled.

## 1. Program position (Phase 0, day 5 of ~42)

| Edge (Goal 2) | Evidence to date | Grade |
|---|---|---|
| Planning mechanism (H1) | D2 **PASS** at 17% training: imagined consequences rank real actions 0.872 (P1) / 0.940 (P4) vs 0.5 chance | measured, preview |
| Inference efficiency (H5) | decision tick **15.07 ms / 1.08 GB** fp32 un-optimized (~66 Hz vs 10–20 Hz req.); ONNX parity ≤1.2e-5; 261 M vs competitors' 15–32 B on leaderboard | measured |
| Inherent safety / self-knowledge (H9/H11/H15) | D8 paired-weather: 16/23 same-scene pairs show raised imagination error under degradation (p≈0.047, 22% training); SC-01 live 3-seed: OKRI **36.0±3.8 vs 12.8±0.6**, LOPS **0.834±0.008 vs 0** (scripted archetypes, real physics) | measured, preview/baseline |
| Data efficiency (H3/H7) | transition-operator rank ≈31–43 vs 2048 readout (2× spectral studies → D-021); 44 h training corpus total | measured (spectral), thesis pending slope experiments |

## 2. Compute fleet — live status at writing

| Resource | Doing | State |
|---|---|---|
| pod1 (A6000) | main training p0-sB01 | **step ~14,350/30,000**; pace degraded again (~100/h) — see §4 incident; **gate eval on the step-14k ckpt RUNNING now** (D1/D2/D3+spectral), results pushed tonight |
| pod2 (A40) | bake-off arms baseline vs K-step=4 | relaunched 21:45 **with new timing instrumentation**; warm-started step 9,000; first instrumented rows expected ≤30 min |
| local 4060 | (CARLA rendering attempt FAILED — 12 h shader-compile wedge, killed) | free for eval bursts; D8 15k re-run queued |
| Colab | validated (T4 33 s cold-to-done) | on-demand for agents |

## 3. Agent knowledge transfer — the transparency table (new standing section)

| Agent (last run) | Produced | Transferred to main stream? |
|---|---|---|
| Benchmarks & Eval (Thu 06:43) | **LAL-v2 metric** + SC-01 audit + leaderboard upgrade (navhard SOTA, EPDMS mapping, competitor block) | ✅ LAL-v2 **integrated** into `stack/tanitad/eval/metrics.py` (suite emits v1+v2; 188 tests); audit numbers on LEADERBOARD |
| DataEng (Tue+extra) | PhysicalAI **R1 selection** (1,926/2,000 urban clips, tool + report); WorldModel-Synthetic **license verified ungated** (264k long-tail clips) | 🔶 R1 tool intake **queued for triage** (next 24 h); D-022 adoption proposal comes to Sayed after the ego-pose probe |
| Opponent Analyzer (Fri) | **SC-04 Stop-Arm scenario** spec+oracle (11 tests); SC-13 catalogued (Avride NHTSA, 16 crashes) | 🔶 SC-04 intake **queued for triage** (next 24 h); SC-13 in scenario DB |
| Tools & DevEnv (Mon+extra) | Colab CLI end-to-end validation; test-suite I/O profile (Drive hydration −30 s/run); CARLA graphics root-cause + pod recipe | ✅ Colab pattern documented for all agents; hydration pinned; recipe feeds the render-pod decision |
| Production & Optimization (Sat #1) | ONNX export parity (≤1.2e-5, no blockers); epcache hardening | ✅ epcache fix **integrated** (+legacy fallback, 178→190 tests); TRT fp16 queued Sat |
| Orchestrator screening | full-hub digest → K-step/RoPE/ZOD/NuRec routed into backlogs | ✅ K-step lever **implemented & running as an arm right now** |

## 4. Incidents & improvements (honest)

- **Trainer crawl (3rd occurrence):** pattern = fast after restart (750/h) → decay → crawl (~100/h) over 1–2 h. My tar-I/O explanation fits occurrence #1 but NOT #3 (no bulk I/O since). **Stopped guessing: per-step `data_s`/`step_s` timing now in every log row** (deployed to the arms tonight; pod1 gets it at next checkpoint-safe restart). The numbers will name the bottleneck (data-path vs compute vs memory pressure); options come to Sayed with evidence.
- **Workflow hardening (D-025):** loop on a cron drumbeat (:13/:41 hourly), reports on their own clock (07:57/12:57/17:57, chat+push+this file series), zombie background tasks cleaned, local-CARLA wedge killed and marked failed-for-now (windowed attempt tomorrow; else render-pod).
- Stall-watchdog live on pod1 (auto-restart on 20-min full stalls; the slow-crawl mode evades it — instrumentation is the answer there).

## 5. Next steps (ordered)

1. **Tonight:** 14k gate results (push + file update) → D2 regression verdict vs 0.872/0.940; first instrumented arm rows → crawl diagnosis.
2. Tomorrow AM: stop-arm + R1 intake triage; D8 matched-pairs re-run vs 14k ckpt (4060); Friday agents 06:43; morning report 07:57.
3. K-step arm interim D2 comparison (~24 h of arm training).
4. DINOv3 arm-B feature precompute (behind arms); BC-baseline reference arm behind that.
5. 30k finish (ETA slipped to ~July 11–12 by the crawls — recovered pace would pull it back).

## 6. Decisions required from Sayed

| Decision | Default if silent |
|---|---|
| none blocking tonight | — |
| D-021 latent-dim doctrine (confirm/reject) | default governs (keep 2048, keep measuring) |
| D-022 WorldModel-Synthetic adoption | comes with pose evidence |
| render-pod recreation (if local windowed CARLA fails tomorrow) | stays on nullrhi + local retry until you decide |
