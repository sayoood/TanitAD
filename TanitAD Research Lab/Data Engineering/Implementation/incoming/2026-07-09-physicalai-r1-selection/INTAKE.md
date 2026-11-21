# INTAKE — PhysicalAI-AV Stage R1 selection tool

- **Package:** `Data Engineering/Implementation/incoming/2026-07-09-physicalai-r1-selection/`
- **Author agent / date:** Data Engineering agent (Tuesday), 2026-07-09
- **Proposed target:** `stack/scripts/physicalai_r1.py` (sibling of `physicalai_r0.py`)
- **Hypothesis / WP served:** H7 (data flywheel) / H4 arm-B corpus / DATASET_LANDSCAPE rank #1 (backlog P0.0)

## What & why (≤10 lines)

Scales the R0 urban selection (500 clips) toward R1 (2,000) per Sayed's 2026-07-08 directive
("leverage all high-quality data"). Reuses the EXACT R0 urban scorer (`_urban_score_from_egomotion`,
single source of truth) so R1 scores are comparable to R0; scores every clip in the cached egomotion
zips (offline — no HF token, no network), selects top-urban round-robin over countries, and writes a
loader-compatible `r1_selection.parquet` + `R1_REPORT.json` with a camera-fetch cost plan.
**Measured (30 cached chunks): 2,850 clips scored, 1,926 pass the driving gate (67.6%)** — 74 short of
2,000, so R1 needs ~1–2 more egomotion chunks. Camera fetch = the SAME 30 chunks as R0 (60 GB) but
yields 3.85× the clips (per-chunk cost, not per-clip). Note: `2026-07-09-physicalai-r1-selection-and-worldmodel-scenarios-license.md`.

## Evidence & tests

- Tests included: `tests/test_physicalai_r1.py` — **3 passed (2.1 s)** on author machine (venv
  `C:/Users/Admin/venvs/tanitad`). Synthetic egomotion in the real schema (timestamp/vx/vy/curvature);
  covers gate logic (urban vs highway vs parked), loader-compatible selection schema, and
  reachability/cost reporting.
- Measured numbers (real bytes): 30 cached chunks → 2,850 scored / 1,926 gate-pass (all 924 failures are
  speed-band); urban-score p50/p90/p99 = 1.38 / 1.97 / 2.44; 23 countries in the 1,926-clip selection.
  Episode-contract PASS verified separately on a real R0 clip via `physicalai.build_episode`
  (`[199,9,256,256]` u8, actions `[199,2]`, poses `[199,4]`, finite; 6.5 s/clip). Artifacts:
  `C:/Users/Admin/tanitad-data/physicalai/r1/{r1_selection.parquet,R1_REPORT.json}`.

## Risk & rollback

- Blast radius if integrated: additive — one new script under `stack/scripts/`; imports the existing
  `physicalai_r0._urban_score_from_egomotion`; no change to `stack/tanitad/`. The `physicalai.py` loader
  currently hardcodes `r0/r0_selection.parquet`; to LOAD an R1 selection it needs a small `selection_path`
  parameter (separate follow-up; NOT in this package — this package only produces the selection + plan).
- Rollback: delete the script; no state in `stack/` depends on it.

---

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)

- **Verdict:** **reject-with-reason — superseded by the canonical R0-derived corpus**
- **Date / by:** 2026-08-03 · orchestrator daily sweep (age at adjudication: **25 days**)
- **Reason & notes:**
  - ⚠️ **This is NOT a quality rejection.** The package's tests were re-run here and **3 passed /
    6.24 s** (`venvs/tanitad`); the scorer reuse is correct and the measurement (2,850 scored /
    1,926 gate-pass / 23 countries) is sound work. It is rejected on **fit to the corpus the
    programme actually built**, which changed under it while it waited 25 days.
  - **What overtook it.** The canonical train corpus is
    `physicalai-train-e438721ae894` — **2,376 episodes**, skip-hash `f09e44db` — and
    `MODEL_REGISTRY.md` §rebuild traces its provenance to `build_pai_cache.py`,
    `physicalai_r0.py fetch-camera` and `rebuild_pai_rolling.py`. **R1 is nowhere in that chain.**
    R1 targeted 2,000 and reached 1,926; the corpus that exists is larger *and* is the one every
    banked arm was trained on.
  - **Absence checked at more than one location** (the two-probe rule): `grep -rn
    "r1_selection\|physicalai_r1"` over `stack/` **and** `Project Steering/` returns **zero
    hits** — 25 days on, nothing references it. Conversely `r0_selection.parquet` is hardcoded at
    **three** live call sites (`stack/tanitad/data/physicalai.py:282,401,463`) and is named in
    `stack/tanitad/lake/filtering.py:72` as the thing the **parity key derives from**.
  - **The follow-up it needs is the one thing that cannot be done.** The INTAKE is explicit that
    loading an R1 selection requires a `selection_path` parameter on the loader. That loader is
    on the parity path. CLAUDE.md: *"Parity is sacred … anything that re-selects episodes breaks
    cross-arm comparability and must be refused."* Integrating a selector whose only use requires
    editing the parity path — for a selection smaller than the one in production — buys nothing
    and risks the comparability of every banked arm.
  - **Nothing is lost by rejecting.** The reusable half is
    `_urban_score_from_egomotion`, which already lives in `stack/scripts/physicalai_r0.py` and is
    imported, not copied. The R1-specific half is round-robin selection + a camera-fetch cost
    plan whose artifacts were written **only** to `C:/Users/Admin/tanitad-data/physicalai/r1/`
    — one dev box, never banked. That stranding is itself the finding here.
  - **If a future non-parity corpus arm (R2 / an external corpus) is opened**, re-open this
    package rather than rewriting it: the scorer-reuse design is right, and only the target size
    and the loader hook need to change. Filed, not deleted — the directory stays in `incoming/`
    with this verdict attached.
- **Integrated as:** n/a — nothing copied into `stack/`.
