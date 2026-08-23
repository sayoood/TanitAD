# PhysicalAI-AV R1 selection (measured) + WorldModel-Synthetic-Scenarios license — 2026-07-09

**Agent:** Data Engineering (Tuesday). **Budget:** 1 iteration, ~1.5 h wall, 1 web search + 1 web fetch.
**Hardware:** local RTX 4060 host / venv `C:/Users/Admin/venvs/tanitad` (CPU-only work — offline
scoring + one HEVC decode). **Cost:** $0 (no cloud, no gated re-download).

> Note on dating: the real wall-clock (per session context + git file mtimes + the Colab-burst note) is
> **2026-07-09**. Prior DataEng notes carried narrative-ahead "W2" dates (…-07-14); this note uses the
> true date. LAST_RUN in STATE moves to 2026-07-09 accordingly (P8 honesty over cosmetic continuity).

## 0. Consumed since last run
- **Monday (Tools&DevEnv):** Colab burst compute is **LIVE** — `ssh tanitad-pod2 'colab run --gpu T4 …'`,
  33 s cold-to-done, $0 free tier (`Tools&DevEnv/Implementation/colab_burst/README.md`). Pattern noted for
  future bake-off arms / probe fits that exceed the 4060. Not needed this run (work was CPU-bound).
- **Orchestrator (2026-07-08):** `DATASET_LANDSCAPE.md` gained the ranked acquisition queue; **PhysicalAI-AV
  R1 (500→2,000) promoted to rank #1**, "DataEng Tue, this week" — this run executes it.
- **Opponent Analyzer:** `SCENARIO_DATABASE.md` SC-02/05/06 carry data-source rows I advance below (joint duty).

## 1. Experiment (G-H) — PhysicalAI-AV R1 urban selection from cached egomotion

**Question.** R0 selected 500 urban clips by scoring 30 egomotion chunks (motion-proxy urban score with
hard driving gates). Can R1=2,000 be reached from the **already-cached** egomotion, or does it need new
gated downloads (the expensive part)?

**Method.** New tool `physicalai_r1.py` (intake pkg `2026-07-09-physicalai-r1-selection/`) reuses the
*exact* R0 scorer `_urban_score_from_egomotion` (single source of truth — R1 scores comparable to R0),
scores every clip in the 30 cached zips offline, selects top-urban round-robin over countries. No token,
no network.

**Measured result** (`C:/Users/Admin/tanitad-data/physicalai/r1/R1_REPORT.json`):

| Quantity | Value |
|---|---|
| Cached egomotion chunks | 30 (1.2 GB) |
| Clips scored (0 errors) | **2,850** (≈95/chunk, not 100 — chunks vary) |
| Clips passing the driving gate (`urban_score>0`) | **1,926 (67.6 %)** |
| Gate failures | 924 — **all speed-band** (`mean_v∉[2,14] m/s`); stop-fraction & distance gates never bind |
| **R1=2,000 reachable from cache?** | **NO — 1,926 < 2,000 (74 short)** |
| Urban-score percentiles (passing) | p50 1.38 · p75 1.76 · p90 1.97 · p95 2.12 · p99 2.44 |
| Selection geo-coverage | 23 countries (France 168 / Italy 143 / Portugal 141 / US 134 / …) |
| Selection hour coverage | all 24 h present; day-peak 12–13 h, night 0–5 h thin but non-empty |
| Mean speed / mean urban-score (selected) | 8.12 m/s / 1.39 |

**Verdict vs expectation.** The backlog assumed R1=2,000 was a scale-up on the pod. Measured: we are
**96 % of the way there from data already on disk** — R1 needs only **~1–2 additional egomotion chunks**
(each ~95 clips × 67.6 % ≈ 64 passing) to clear 2,000. This shrinks the R1 acquisition from "score dozens
of new chunks" to "one small top-up fetch," and is the falsifiable, measured answer G-H asks for.

**Camera-fetch cost (the real ingest cost).** The 1,926 clips live in the **same 30 chunks R0 already
needed**. Camera cost is **per-chunk** (whole ~2 GB zip downloaded, only selected members extracted, zip
deleted): 30 × 2 GB = **~60 GB** to materialise all 1,926 mp4s — *identical bandwidth to R0's 500-clip
fetch* (R0 already listed all 30 as `camera_chunks_needed`). → **3.85× the clips for the same 60 GB.**
Actionable fetch-plan rule: **extract ALL gate-passing clips per already-downloaded chunk**, never re-pay
per-chunk bandwidth for a subset. On the pod (datacenter net) this is ~1 h; the zips were deleted post-R0
(disk discipline) so the 60 GB is a re-download, not incremental.

**Real-bytes contract check (G-D2 / G-E).** `physicalai.build_episode` on a real R0 clip →
`frames [199,9,256,256] uint8`, `actions [199,2]`, `poses [199,4]`, all finite/aligned, **6.5 s/clip**
(single-thread CPU decode + focal-crop). Steer ∈ [−0.01, 0.23] rad, accel ∈ [−1.6, 2.7] m/s², v ∈
[4.5, 16.7] m/s — physically sane. Raw mean-abs inter-frame delta ≈ 0.023 (this is *not* the thresholded
A8 `frame_change_fraction`; a proper A8 sweep on R1 stays backlog). Same pipeline serves R1 (same corpus,
same loader) → R1 clips will load under the identical contract.

## 2. WorldModel-Synthetic-Scenarios — license VERIFIED + a loader caveat (rank #2, backlog P0.1)

Fetched the HF card `nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios`:

- **License = OpenMDW-1.1, UNGATED.** Not the confidential NVIDIA AV Dataset License that firewalls the
  real PhysicalAI-AV. OpenMDW-1.1 (Linux Foundation; NVIDIA's standard open license for Cosmos/Nemotron/
  Isaac GR00T) grants unrestricted royalty-free use/redistribution/modification, only condition = retain
  the license + origin notices; **outputs are unrestricted**. → **preliminarily public-claimable**, same
  class as Cosmos-Drive-Dreams (CC-BY-4.0). *But reclassifying the public-claim firewall is a data-strategy
  decision (D-018) → proposed as **D-022**, default = keep firewalled to comma+Cosmos until Sayed/legal
  confirms.*
- **Size / format:** 264,000 clips, ~1,467 h, **8.3 TB**, 4K@24 fps H.264; 4-cam forward rig + 7-cam 360°
  fisheye; per-cam VLM captions + clip metadata (weather/time/surface/region).
- **Scenario families (exact card proportions):** cut-in 32.9 % · vehicle–pedestrian 21.1 % · lane-change
  12.9 % · pedestrian 12.4 % · **weather-degradation 9.2 %** · nudging 8.8 % · **emergency-vehicle 2.7 %**.
- **⚠ Loader caveat (corrects the backlog's "near-zero cosmos mirror" assumption):** the card advertises
  RGB + captions + scene metadata but **does not list ego pose / actions**. The Cosmos loader derives
  steer/accel from per-frame 4×4 `vehicle_pose`; if this corpus ships no poses, actions must come from an
  **IDM (H7)** or it is video-only — **NOT** a near-zero mirror. Pose availability is the gating question
  before spending loader effort. Next step: `huggingface_hub` file-listing probe on one clip's parquet
  set to confirm/deny a pose field (no full download; ~minutes).

## 3. Scenario DB joint duty (D-020 §5) — data-source rows advanced

- **SC-02** (ghost cut-through, occluded pedestrian): license was the blocker — now **verified ungated
  (OpenMDW-1.1)**; pedestrian material = 12.4 % ped + 21.1 % vehicle–pedestrian ≈ **33 % of 264 k clips**.
  Row moves from "license check pending" → "license clear; pose/action availability UNVERIFIED (gating)".
- **SC-05** (degraded-visibility): adds **weather-degradation 9.2 % ≈ 24 k clips** as a second synthetic
  source complementing the already-sourced Cosmos weather variants → strengthens the SC-05 paired D8 set.
- **SC-06** (emergency-vehicle) — **fills a documented public-data gap**: the entry said "thin publicly";
  the corpus has an **emergency-vehicle family 2.7 % ≈ 7 k clips** (visual light-pattern proxy, audio out
  of scope P0). Row moves catalogued → data-source identified (pending the same pose caveat).

## 4. Hypotheses / decisions
- **H7 / H4** — no status change (P8): R1 yield is a data-availability measurement, not a hypothesis test;
  WorldModel-Synthetic-Scenarios (once loadable) is H6/H15/D9 long-tail + H4 diversity material. Ledger
  evidence row added.
- **D-022 proposed** (data-strategy, D-018 ESCALATE): reclassify WorldModel-Synthetic-Scenarios (OpenMDW-1.1)
  as public-claimable, i.e. widen the firewall beyond comma2k19 + Cosmos-DD. Default if no answer: hold —
  keep firewalled, internal training use only. Nothing is blocked by holding.

## 5. Recommendations (actionable)
1. **Top-up fetch, not a re-score:** grab 2 more egomotion chunks on the pod (~4 GB), re-run
   `physicalai_r1.py --target 2000` → R1 clears 2,000. Then camera-fetch the (≤32) chunks extracting **all**
   gate-passing clips (~64 GB, ~1 h pod).
2. **Confirm the pose field of WorldModel-Synthetic-Scenarios before any loader work** — it decides whether
   it's a near-zero cosmos mirror or an IDM-gated (H7) ingest.
3. **Loader `selection_path` param** (small follow-up intake) so `physicalai.py` can load `r1_selection.parquet`
   without touching R0 provenance.
