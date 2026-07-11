# TanitAD Program Report — 2026-07-11 17:57 series (compiled 18:30, D-025)

## 1. Main training (pod1)
**Step 25,750/30,000 (86 %)**; 3-min sampled pace 1,000 steps/h; trainer PID 10h13m uptime (no
restarts today); watchdog alive, watchdog log empty. **HONEST FLAG:** a ~2 h crawl episode
(16:05→18:00 produced only ~50 net steps) escaped the watchdog's <50-steps/15-min threshold —
borderline crawl, recovered on its own. **30k ETA slips ~20:15 → ~22:15** if pace holds; the
flagship gate eval follows immediately (D1 as route-resampled mean±CI).

## 2. Pod2 arms
**REF-A COMPLETE — full 30k run finished (~4.5 h wall-clock, feature-level speed dividend):**
loss 1.19→**0.465**, pred 0.080, **roll 0.076** (K=4 rollout healthy), SigReg 3.44 active,
adapter_std 0.86 monotonically rising (no collapse), 60 checkpoints. The frozen-DINO reference
finished training BEFORE our main model reaches 30k. Earlier K-step A/B arms already banked
(D-027). Pod2 now idle → REF-A comparison probes are the next launch (gate-matched vs main 30k).

## 3. Experiments/evals since 13:05 report
- **REF-A 30k training complete** (above) — comparison probes pending tonight.
- **REF-B rev1 built, tested, merged** (`e616b23`): full 4-layer E2E, 260.7 M (−0.82 % vs main,
  test-pinned), 204 tests green, verified on pod3.
- **REF-B rev2 (Sayed's review) in flight:** strategic layer → real transformer d384×4 with
  route-derived nav commands + own aux loss (fixes a REAL defect — strategic trained on constant
  `follow`), tactical 4→6, encoder 27→25. Agent was cut by the 16:30 usage window, resumed 17:57.
- **Pod3 provisioned:** repo + env + 193-test parity; comma val cache complete (15.9 GB).
- **H16 banked** (Sayed's active-depth-interrogation idea): dossier + F1–F3 falsifiers + ledger row.

## 4. Agents & transfer
Orchestrator logged Sayed's confirmations (`7cc210f`: D-028 accepted, D-022 numbering gap fixed).
Data-Eng backlog: NEW semantic/strategic-label dataset survey (Sayed directive — nuPlan, DriveLM,
CoVLA, L2D, Talk2Car, Bench2Drive; license-checked, one Phase-1 ingest recommendation). No new
discipline-agent commits since W30. My intake debt (stop-arm, R1) still open.

## 5. Master-Plan position & the four edges
Phase 0 day 7/42 — all three architecture arms exist as of TODAY (main training, REF-A trained,
REF-B built). Data-efficiency edge: 261 M vs 15–32 B field + REF-A comparison arriving. Safety
edge: D8 matched-pairs positive, H9 barrier shipped, H16 pipeline defined. Inference edge:
15.07 ms tick / fp16-safe verdict standing. Compliance: REGULATION_TRACE ongoing. Tonight's 30k
gates + REF-A comparison close the Phase-0 core evidence cycle ~2 weeks early.

## 6. Next steps (ordered)
1. **Sayed: expand pod3 volume to 300 GB** (see §7 — data blocked on quota).
2. REF-A comparison probes on idle pod2 (launch at next drumbeat).
3. D1 route-resampled-CI instrument into the gate runner BEFORE the 30k eval (~22:15).
4. 30k flagship gates → Phase-0 results report (late evening).
5. REF-B rev2 completes → Sayed look → training GO (data ready once quota fixed).
6. PhysicalAI cache rsync pod1→pod3 in the post-30k idle window.
7. ~22:50 usage window: workflow resumes (stack audit / literature / scenario scaling).

## 7. Decisions/actions required from Sayed
- **ACTION (only you can): pod3 volume quota is ~50 GB, not the planned 300 GB** — bulk writes
  fail since 14:55 ("Disk quota exceeded"; the stalled download was this, not the network).
  RunPod console → pod 69.30.85.16 → Edit/Volume → **300 GB**. Default if not done: REF-B data
  staging stays blocked (34 GB of 88 GB banked; wget resumes automatically after expansion).
- REF-B training GO (after rev2 look) — default: hold until your word, per agreement.
- No other decisions pending; D-021/D-022/D-027 defaults govern.

## 8. Incidents & improvements (honest)
- **Pod3 quota mis-sizing undetected for 3.5 h:** wget stalled silently at 14:55; I mis-read it
  as a network stall and restarted the download twice before finding "Disk quota exceeded". Cost:
  ~3.5 h data-staging delay. Improvement: provisioning checklist now includes an explicit
  `dd`-write quota probe, not just `df` (which shows the cluster, not the quota).
- **pkill self-match, third occurrence:** my download-restart ssh contained "wget" in the relaunch
  string → pkill killed my own session. The documented lesson (separate ssh calls) exists since
  the trainer incidents — I violated it under time pressure. No damage.
- **Pod1 crawl episode escaped the watchdog threshold** (borderline ~50 steps/15 min): ETA honesty
  restored (22:15, not 20:15). If a second episode occurs before 30k, I apply Sayed's standing
  stall remedy (restart from checkpoint) without further ask.
- **REF-B rev2 usage-window cut** — resumed; the 22:50 window is reserved for workflow resumes.
