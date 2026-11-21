# INTAKE — Semantic/strategic-label survey + L2D taxonomy probe & contract-map spec

- **Package:** `Data Engineering/Implementation/incoming/2026-07-11-semantic-label-survey/`
- **Author agent / date:** Data Engineering agent (Tuesday), 2026-07-11 (pm run)
- **Proposed target:** research + spec only for now. If L2D is adopted (recommended), the loader lands
  as a *separate* Phase-1 package `stack/tanitad/data/l2d.py` (Cosmos-mirror + D-016); `l2d_contract_map.py`
  becomes its instruction/action mapping helper. Nothing in this package needs to enter `stack/` yet.
- **Hypothesis / WP served:** H7 (data flywheel) / H12 (command conditioning) / REF-B strategic head
  supervision (Sayed directive 2026-07-11, backlog P1 #2d).

## What & why (≤10 lines)

REF-B rev2's strategic head trains on comma2k19 route-geometry pseudo-labels that are ~all `follow`
(highway-dominated). This package surveys + ranks datasets with rich semantic strategic/behavior labels
and recommends **ONE** Phase-1 ingest. Headline: **L2D (`yaak-ai/L2D`, Apache-2.0)** — probed on real HF
bytes — carries **4,219 distinct compositional nav commands** (96 % distance / 74 % speed-limit / 61 %
road-class tokens) **co-registered with real ego actions** (`action.continuous`-3 + `action.discrete`-2),
`waypoints`-10, and 6 surround cams, over 100k eps / 26.5 M frames. It is the only surveyed corpus that
is real + L1/L2-labeled + action-co-registered + camera-present + **public-claimable**. Recommendation:
ingest a *filtered streaming slice* (non-`follow` maneuver tail) as REF-B's strategic supervision.

## Evidence & tests

- **Measured (real HF bytes):** `probe_l2d_taxonomy.py` ran against `yaak-ai/L2D` `meta/` + one data shard
  (truststore; no 90 TB clone), ~3 min, $0. Raw result committed: `l2d_taxonomy_result.json`
  (total_episodes 100000, total_frames 26466954, fps 10; action dims 3/2; waypoints dim 10; 4,219 tasks;
  token coverage 4039/3133/2577 of 4,219). Reproducible with the HF token in `Keys.txt`.
- **Standalone tests:** `tests/test_l2d_contract_map.py` — **10 passed (0.14 s)** on venv
  `C:/Users/Admin/venvs/tanitad`, offline (synthetic rows in the measured L2D schema). Covers the 3→2
  action channel algebra, fail-loud on wrong dims, nav-command classification on **verbatim measured
  instructions**, the compositional "earliest decisive maneuver" rule, roundabout override, the
  comma-vs-L2D label-entropy gap metric, and contract-row assembly incl. the dict-wrapped instruction form.
- **No stack file touched.** `l2d_contract_map.py` is numpy-only, no video decode.

## Risk & rollback

- Blast radius if integrated: none yet — research + spec. The eventual L2D loader is additive
  (new `stack/tanitad/data/l2d.py`), gated on the §4 falsifier (action-sign decode + front-cam FOV
  after D-016). If the falsifier fails, L2D drops to EVAL-only and nuPlan becomes the train pick.
- Rollback: delete the package; no `stack/` state depends on it.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** integrate / integrate-with-changes / defer / reject
- **Date / by:**
- **Reason & notes:**
- **Integrated as:**
