# TanitAD Program Report — 2026-07-12 01:30 (D-025 series)

## 1. Main training (pod1)
Step **26,950/30,000 (90%)**, 3-min sample **0 steps/h — CRAWLING at the memory cap**
(cgroup 61/62 GB). Trainer + OOM-guard both ALIVE; loss healthy (-1.40). No crash since the guard
(20 restarts earlier tonight, none new). **This is the OOM-vs-thrash tension made visible:** the
guard's page-cache eviction prevents the silent OOM kill but starves the data path → near-zero
throughput at the cap. **30k ETA is now unbounded at this pace.** The 26,950 checkpoint is saved
and is 90%-trained on the full recipe — decision-useful for the flagship gates NOW (see §7).

## 2. Reference arms (pod2)
- **REF-A pool-adapter 30k: DONE.** ADE@1s probe (comma-val, d1_probe protocol): ridge_a10 **17.01**.
- **REF-A grid-adapter 30k: DONE.** ADE@1s probe: ridge_a10 **20.22** — grid is WORSE than pool.
- **Main comma-only control (step ~9k):** ridge_a10 **11.95**; main mixed-corpus reference 7.0–8.5.
- **REF-B: NOT yet launched** — blocked on pod3 comma data (no loss/D2 numbers yet).
- **PhysicalAI dataset build on pod2: COMPLETE** (401 train + 100 val) — 2nd self-sufficient
  full-mix node, built from origin, pod1 untouched.
No D2 (direction-accuracy) numbers this cycle — these are D1-style decodability probes only.

## 3. Experiments/evals since 17:57 report
- **Grid-adapter probe REFUTES the adapter hypothesis:** keeping DINO's spatial token layout made
  decoding WORSE (20.2 vs pool 17.0), not better. The frozen-encoder deficit looks REAL, not an
  adapter artifact — first directional H4 evidence (from-scratch encoder > frozen DINO at this
  scale/task). Decisive read needs main@30k.
- **Comma-only main control** (main@9k ridge_a10 11.95) — the apples-to-apples anchor for REF-A.
- **Replay app first live session:** main@8.5k ADE 11.46 m vs REF-A@30k 7.60 m (main undertrained —
  fair fight is 30k).
- **Shipped:** `--grpc-only` single-port serving, replay tutorial, rr_log legend groundwork.

## 4. Agents & knowledge transfer
Recent commits all MVP-loop (replay app, TanitResim, Alpamayo encoder research, resolution &
recipe-dataset backlog items). No new discipline-agent worktree commits since W30/W31; orchestrator
W31 narrative clock ran ahead of wall-clock again (flagged, not acted). Intake debt persists:
stop-arm gate + R1 selection still un-triaged.

## 5. Master-Plan position & four edges
Phase 0 **day 8/42**. All three architecture arms now EXIST (main training, REF-A trained ×2
adapters, REF-B built & tested). **Efficiency edge:** 261 M vs 15–32 B field; REF-A comparison
maturing. **Safety edge:** D8 matched-pairs positive, H9 barrier, H16 dossier. **Inference edge:**
15 ms tick / fp16-safe. **Compliance edge:** REGULATION_TRACE ongoing. New capability tonight:
data-mix-as-recipe proven (rebuilt full corpus on pod3 AND pod2 without pod1) → reproducibility
asset for the paper.

## 6. Next steps (ordered)
1. **pod1 crawl decision (§7)** — the critical-path blocker.
2. TanitResim (building on Opus) → deploy pod2, send proxy URL.
3. Three-arm probe comparison at the chosen main checkpoint (26,950 or 30k).
4. REF-B launch on pod3 comma-extraction complete; full-mix REF-A after DINO precompute.
5. 07:57 report; then finally the deferred workflow resumes.

## 7. Decisions required from Sayed
- **pod1 crawl (only real decision):** the 26,950 ckpt is 90%-trained on the full recipe and the
  gates are identical at 27k vs 30k. **DEFAULT (my recommendation): if the crawl hasn't cleared by
  ~03:30, run the flagship decision-grade gate evaluation on the 26,950 checkpoint and label it
  "≈30k (90%)"** — hours of near-zero-throughput crawl are not worth 10% more steps for a
  necessary-not-sufficient open-loop gate. I will NOT change the data recipe mid-run (would
  contaminate the comparison). Override if you want the literal 30k.
- Everything else on standing defaults (REF-B/REF-A-full-mix proceed; D-021/022/027 govern).

## 8. Incidents & improvements (honest)
- **pod1 OOM-vs-thrash tension unresolved at root:** the guard trades crash-death for crawl-slowness
  because the working set genuinely exceeds the 62 GB cap near end-of-run. A proper fix (LRU-aware
  eviction, or a smaller-but-recipe-identical mmap window) is post-30k work; tonight it limps.
- **Comma re-download stalled silently at 44 GB** (2nd time) → restarted with aggressive retry
  (--tries=100, read-timeout). The HF-tar path is fragile; the pod2 rebuild-from-origin is the
  more robust pattern and is what unblocked pod2.
- **Dual-sink .rrd bug** (serve+record → empty file) and **rerun viewer version-lock** cost a
  confused hour on the app demo; both documented, TanitResim (single-port HTTP) sidesteps the proxy
  issue entirely.
- **Workflow resumes (audit/literature/scenario) slipped a 2nd night** behind data recovery —
  honestly overdue; first priority once pods settle.
- **Fable-5 usage limit** killed the first TanitResim agent (0 commits); relaunched on Opus.
