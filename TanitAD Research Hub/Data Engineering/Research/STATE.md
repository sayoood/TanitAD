# STATE — Data Engineering

LAST_RUN: 2026-07-09 (Tuesday agent)
QUALITY: full (G-A…G-C, G-E, G-H, G-D1, G-D2 met; intake pkg 3✓ standalone, no stack files touched).
  Shared-file rows (G-D ledger, joint SCENARIO duty, PROJECT_STATE §5) **DEFERRED — see HANDOFF**: a
  concurrent agent was actively writing SCENARIO_DATABASE / HYPOTHESIS_LEDGER / PROJECT_STATE (mtimes
  within 90 s of my commit) — deferred to avoid clobbering live work (repo-advances-mid-session rule, P8).
  All deferred content is captured verbatim below and in the research note; nothing is lost.

## This run (2026-07-09)
- **PhysicalAI-AV R1 selection MEASURED** (backlog P0.0 / DATASET_LANDSCAPE rank #1; intake pkg
  `incoming/2026-07-09-physicalai-r1-selection/`, 3 tests ✓). New tool `physicalai_r1.py` reuses the exact
  R0 urban scorer and scores the **30 already-cached egomotion chunks** offline (no token/network):
  **2,850 clips scored (0 errors) → 1,926 pass the driving gate (67.6 %)**. **R1=2,000 is 74 short from
  cache** → needs ~1–2 more egomotion chunks. Gate failures (924) are **all speed-band**. Camera fetch =
  the same 30 chunks as R0 (~60 GB) but **3.85× clips for identical bandwidth** (per-chunk cost) → fetch
  rule: extract ALL gate-passing clips per downloaded chunk. Artifacts `.../physicalai/r1/{r1_selection.parquet,R1_REPORT.json}`.
- **Episode-contract PASS on real bytes (G-D2/G-E):** `physicalai.build_episode` on a real R0 clip →
  `[199,9,256,256]` u8 frames, actions `[199,2]`, poses `[199,4]`, finite/aligned, 6.5 s/clip; steer/accel/v
  physically sane. Same pipeline serves R1.
- **WorldModel-Synthetic-Scenarios license VERIFIED** (rank #2 / backlog P0.1): **OpenMDW-1.1, UNGATED**
  (Linux-Foundation permissive; NVIDIA's Cosmos/Nemotron license) → *preliminarily public-claimable*. 264 k
  clips / 8.3 TB / 4K@24; families incl. weather-deg 9.2 %, emergency-veh 2.7 %, ped ~33 %. **⚠ card lists
  NO ego pose/actions** → "near-zero cosmos-mirror" assumption at risk; confirm a pose field before loader
  work (else IDM/H7 or video-only). DATASET_LANDSCAPE + KB updated.
- Note: `2026-07-09-physicalai-r1-selection-and-worldmodel-scenarios-license.md`.

## Next (backlog, priority order)
1. **R1 top-up (2 chunks) on pod** → clear 2,000; then camera-fetch (≤32 chunks, ~64 GB, ~1 h) extracting
   ALL gate-passing clips per chunk; build epcache AFTER the 30k trainer finishes (cgroup).
2. **WorldModel-Synthetic-Scenarios pose probe** — `huggingface_hub` file-listing on one clip's parquet set
   to confirm/deny a pose field. Decides loader path (cosmos-mirror vs IDM/H7). ~minutes, no full download.
3. **Loader `selection_path` param** (small follow-up intake) so `physicalai.py` loads `r1_selection.parquet`
   without touching R0 provenance.
4. Steering-ratio calibration log (H7 binding artifact) on real Chunk_1; A8 harness on real Chunk_1.
5. Zenseact ZOD pilot loader (real-CAN #2, H4 arm-B, EU/night).

## HANDOFF — DEFERRED shared-file rows (concurrent-write race; orchestrator/next run to merge)
Deferred to avoid clobbering a live editor. Apply these verbatim when the files are quiescent:

**PROJECT_STATE.md §5 session-log row (newest):**
| 2026-07-09 (Thu) | Data Engineering agent | **PhysicalAI-AV R1 selection measured** (backlog P0.0 / rank #1): tool `physicalai_r1.py` (intake, 3✓) reuses the R0 scorer on the 30 CACHED egomotion chunks — **2,850 scored → 1,926 gate-pass (67.6 %); R1=2,000 is 74 short from cache → ~1–2 more chunks**; camera fetch = same 30 chunks as R0 (~60 GB) but **3.85× clips/GB** (extract-all rule). Episode-contract PASS on real clip (`[199,9,256,256]`, 6.5 s). **WorldModel-Synthetic-Scenarios license = OpenMDW-1.1 UNGATED** (D-022 proposed: widen public firewall — default hold) — 264 k clips, emergency-veh 2.7 %/weather 9.2 %/ped 33 %, **⚠ no ego pose on card** (loader caveat). Advances SC-02/05/06 data rows. | `.../Data Engineering/Research/2026-07-09-physicalai-r1-selection-and-worldmodel-scenarios-license.md`, `.../Implementation/incoming/2026-07-09-physicalai-r1-selection/`, `DATASET_LANDSCAPE.md` |

**HYPOTHESIS_LEDGER.md change-log row:**
- 2026-07-09: Data Eng — H7/H4 data-availability delta (no status change, P8): PhysicalAI-AV **R1 yield
  measured** (1,926 urban clips already reachable from cached egomotion; +64 GB camera to materialise 3.85×
  R0's clips). WorldModel-Synthetic-Scenarios (OpenMDW-1.1, ungated) identified as H6/H15/D9 long-tail +
  H4-diversity source *conditional on a pose field* (card lists none → IDM/H7 or video-only). See
  `Data Engineering/Research/2026-07-09-*.md`.

**SCENARIO_DATABASE.md data-source rows (joint duty D-020 §5):**
- SC-02 (occluded-ped): PhysicalAI-WorldModel-Synthetic-Scenarios **license VERIFIED ungated (OpenMDW-1.1)**;
  ped material ≈33 % of 264 k; ⚠ pose/action availability UNVERIFIED (gating) → catalogued → *data-source
  identified (license clear, pose caveat)*.
- SC-05 (degraded-visibility): + weather-degradation family 9.2 % ≈ 24 k clips (second synthetic source
  complementing Cosmos weather variants).
- SC-06 (emergency-vehicle): **fills the documented public-data gap** — emergency-vehicle family 2.7 % ≈ 7 k
  clips (visual light-pattern proxy; audio out of scope P0) → catalogued → *data-source identified (pose caveat)*.

**DECISIONS.md — new proposed entry (D-018 ESCALATE, data-strategy):**
- **D-022 (proposed)** — reclassify PhysicalAI-WorldModel-Synthetic-Scenarios (OpenMDW-1.1) as
  public-claimable (widen firewall beyond comma2k19 + Cosmos-DD). Default if no answer: HOLD — keep
  firewalled, internal training use only. Nothing blocked by holding.
