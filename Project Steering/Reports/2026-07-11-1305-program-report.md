# TanitAD Program Report — 2026-07-11 13:05 (D-025 series)

## 1. Main training
Step **23,700/30,000 (79%)**, measured **1,000 steps/h** (3-min sample), 0 watchdog events since
morning restart. **30k ETA ~19:30 tonight** → decision-grade gate evaluation immediately after.
Timing rows: data path = 67% of step time even at full pace (post-30k optimization #1, quantified).

## 2. D1 "regression" RESOLVED — protocol, not model
Probe-capacity discriminator (14k-frozen vs 23.5k, route-held-out ADE@1s):
ridge α1 7.56→7.51 · α10 8.54→**7.02** · α100 12.60→9.47 · MLP 11.19→**8.40**.
Later checkpoint decodes BETTER at every capacity → the 21k gate number (11.52 m) was
**route-split sensitivity** of the D1 protocol, not information loss. Action: D1 becomes a
route-resampled mean±CI in the gate runner before the 30k eval. Artifact:
`/workspace/experiments/d1_probe_capacity.json` (pod1), script `stack/scripts/d1_probe_capacity.py`.

## 3. Since last report
REF-A stage 1 done (490 eps → 54 GB DINOv2-B/14 feature grids; DINOv3 gated — fallback recorded);
D-027 (K-step) adopted with config-flip guard; timing instrumentation live on both pods.

## 4. Agents & transfer
Overnight: daily report, orchestrator W29 (6/6 healthy, $0/$80), 21k gate preview (flag resolved
today). Sat 06:44: production agent (TRT fp16). Orchestrator triage debt: stop-arm + R1 intakes.

## 5. Master-Plan position
Phase 0 day 7/42: D2 PASS strengthening, D1 protocol-hardened, D3 lever proven+adopted, scenario
suite live (real physics, multi-seed), efficiency baselined (15 ms/1.08 GB), REF-A staged, REF-B
planned, acquisition pipeline ahead. 30k closes the Phase-0 core evidence cycle ~2 weeks early.

## 6. Next steps
1) D1 CI instrument → gate runner; 2) REF-A trainer (A40, tonight, rollout_k=4);
3) **30k flagship evaluation tonight**; 4) workflow resumes + architecture panel;
5) intake triage; 17:57 report.

## 7. Decisions required
None new. D-027 auto-executes post-30k; D-021/D-022 defaults; remote-control re-toggle pending
(pushes undeliverable — this file series + chat are the record).

## 8. Incidents (honest)
Monitor died with app restart (re-armed); workflows usage-window-bound (3 interrupted passes);
pod2 idle ~2 h between precompute and REF-A build (on me); D1 scare cost half a day, bought a
permanently better instrument.
