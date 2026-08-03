# INTAKE — v2-line clean-val: column-semantics contract + balancing selector

- **Package:** `Data Engineering/Implementation/incoming/2026-08-02-v2-clean-val-selector/`
- **Author:** Data Engineering agent (weekly run), 2026-08-02, branch `agent/data-engineering-20260802`
- **Backlog item:** `Project Steering/BACKLOG.md` **A3** (C64 option B) — the stated blocker was
  *"establish column semantics before any selector"*.
- **Proposed target in `stack/`:** `stack/tanitad/data/pool_columns.py` +
  `stack/scripts/clean_val_select.py` (both additive; nothing existing is modified).

## What & why (≤10 lines)

A3 could not proceed because `v2_pool_scored.parquet`'s columns had no contract: `lk` was read as
a rate (it is a frame COUNT) and produced `needed_in_val = 48330` for a 600-clip split, and
`stopped/city/hw` were called "not clean 0/1" (they are per-window FRACTIONS). `pool_columns.py`
is that contract as code — every semantic derived from the emitting scorer and backed by an
identity that fires on corrupt data. `clean_val_select.py` then builds the split, and reports the
three things the prior pass conflated: COUNT feasibility (per axis set), DISTRIBUTIONAL balance
(per metric family), and POOL DEPTH (how much of each cell it eats). All 34 semantic checks pass
on the real 18,988-row pool. **The selector's design changed because of what it measured** — see
Evidence.

## Evidence & tests

- `pytest tests -q` → **27 passed, 0 skipped** (standalone; synthetic fixtures, no parquet needed).
  Every identity has a corruption test — a check that cannot fail is not a check.
- `python make_results.py` (~6 s, dev-box CPU, $0) regenerates every artifact here.
- **MEASURED, real bytes** (`selector_comparison.json`):
  - semantics: **34/34 checks PASS** on 18,988 rows; `nlab ≡ 179` for every clip, so the `lk`
    misread was a pure scale error (rank-preserving) — corrected `lk_rate` train **0.4500** vs
    remainder **0.6718**, which reproduces the corpus design's "lane_keep 45.0 %".
  - **the pool has 18,988 rows but 18,987 unique clips** — one clip is registered under two
    chunks (`32ad1a3a-…`), and it IS in the v2 selection. Both figures have been quoted; neither
    doc said which was which.
  - **A3's "6.77× headroom ⇒ FEASIBLE" is an artifact of a one-axis view.** Headroom at n=600:
    junction only **6.77×** → +has_turn **4.05×** → +speed **1.07×** → +has_brake **0.77×
    (infeasible)**.
  - **Cell-matching does not deliver balance**: the quota design leaves max |d| **0.3997**
    (10/13 axes over the 0.10 bar). Greedy covariate balancing reaches **0.0094**; the shipped
    hybrid (quota + balance) reaches **0.0532** with the cell match intact (L1 0.0047).
  - **Why matching cannot fix it**: inside each matched cell the remainder is still skewed vs
    train — median |d| **0.359**, p90 0.915, max 2.351 over 10 cells × 9 features.
  - ⛔ **v1 contamination**: a v2-only-clean 600-draw contains **62 clips of v1's TRAIN** and 24 of
    v1's VAL ⇒ the shipped manifests exclude the whole parity selection. C64 at clip granularity:
    **256 of v1's 600 val clips (42.7 %) are inside v2corpus's training selection** — a different
    statistic from C64's 21/40 eval episodes; do not merge them.
- **Shipped artifacts:** `v2_clean_val_manifest.json` (**n=400**, sha256 `abe041db72a045b3…`,
  max |d| 0.0409, max cell census 69.4 %) and `v2_clean_val_manifest_n300.json` (n=300, max |d|
  0.0158, census 52.8 %). Both carry the disjointness proof, the balance table per family, and
  the per-cell pool-depth cost.

## Risk & rollback

- **Additive only.** No `stack/` file is touched by this package; nothing in the training or eval
  path changes. Rollback = delete the two modules.
- ⛔ **This is a PLAN + an instrument, not a corpus change.** Freezing a v2-line val is a PI
  decision (`V2_CLEAN_VAL_PLAN.md` §"Neither is authorised … without the PI"). The manifests are
  candidates.
- **Known limits, stated rather than papered over:** (1) balancing matches MOMENTS — max KS vs
  train is still 0.15–0.19, above the 0.057 critical value, so the split is mean-balanced, not
  distributionally exchangeable, and no draw from this remainder can be (it is the residue of a
  quota selector); (2) disjointness is provable at **clip** granularity only — PhysicalAI-AV ships
  no session/drive id and its egomotion clock is clip-local microseconds, so an L2D-style
  time-overlap test cannot be run; (3) 600 clips is **not available** once v1's parity corpus is
  excluded (headroom 0.95) — 400 is the largest size that keeps every axis balanced and leaves
  ~31 % of the binding cell for a future split.

## ORCHESTRATOR VERDICT (filled by the MVP stream — do not pre-fill)
