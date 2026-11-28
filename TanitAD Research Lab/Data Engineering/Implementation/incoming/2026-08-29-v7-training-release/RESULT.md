# tanitad-v7-training-corpus — the PI-authorized training release

**Date** 2026-08-29 · **Owner** DataFlyWheel (P2) · **Status** RELEASED (HF private)
**Authorization** PI, verbatim: *"Let take the current state as base line and release the
data set as required for training our next model v7f, refcv3, refav1 and refd. The usage
of the corresponding data to the alpamayo and tactical_/strategic labeled data is
mandatory in order to complete the training corpus."* Start + finish notified to the
Master Mind as instructed; start notice sent as PI-aligned before the build.

## What was released

| component | content | verification |
|---|---|---|
| `labels/` | **4,719 clips / 26.2 h** v7 ground truth (blob `121a8d93`, = `D-LABEL-GT` pin at HEAD `9b440d6`) | 0 exclusion violations · 0 coherence defects · 68 turn-suppressions carried |
| `alpamayo/` | full augmentation: **23,644 rows / 5 tasks** (`records.parquet`) + selection manifest — **MANDATORY per PI** | row/clip/task counts asserted |
| `egomotion/` | all 4,719 per-clip provider parquets, tar **1,967 MB** | tar member count == corpus |
| `index/clip_to_chunk.parquet` | camera retrieval map (chunk + split per clip) | covers 4,719/4,719 |
| `index/front_wide_cy.parquet` | ⭐ **per-clip cy + rig** — **rig B is the MAJORITY: 2,723 vs 1,996 A** (medians 753.9 / 540.7); a geometric-centre crop is ~215 px wrong for most of the corpus | coverage 4,719/4,719, clusters match the known two-rig split |
| `tools/` | `pull_camera.py` + `pull_egomotion_range.py` (HTTP-range zip readers) | camera (~47 GB) ships as tool+index, not bytes |
| `MANIFEST.json` | corpus id `a48251e89c7a8603`, per-file sha256, per-consumer notes, **5 mandatory trainer conditions**, exclusions ledger (EMPTY at release, append-with-reason policy) | sha `7ebd88cb16d0c956` |

**Destination**: `https://huggingface.co/datasets/Sayood/tanitad-v7-training-corpus`
(**PRIVATE**, per the standing augmented-sets rule). Push verified by content:
remote listing complete + MANIFEST and labels re-downloaded and sha256-matched.

## The five mandatory trainer conditions (in MANIFEST, binding for all four consumers)

1. **Mask the 8 `NOT_YET_EXTRACTABLE` classes** — else a full-vocab head trains 8 empty classes.
2. **Weight the strategic skew** (FOLLOW_ROUTE ~64–74 % of `g_str`).
3. **`disputed` tokens** (contested-turn suppressions, ungrounded CoT) — mask or down-weight, never clean positives.
4. **Untimed CoT tokens**: `t_nominal_s` = band midpoint (PI convention); `time_basis` separates placed-by-convention from placed-by-evidence.
5. **`nav_command` is an ORACLE INPUT** (ego-future) — training input only, never a target, never in vision-only eval.

## Parity note (binding)

This corpus (`alpamayo ∩ egomotion`, UUID-keyed) is a **new, separately identified
corpus** — id `a48251e89c7a8603`. It does **not** re-select the canonical parity corpus
`physicalai-train-e438721ae894` / skip-hash `f09e44db`, which remains authoritative for
all cross-arm parity comparisons.

## Deliverable manifest

* HF: `Sayood/tanitad-v7-training-corpus` (private) — 11 files, 1,996 MB
* repo: this package (`raw/MANIFEST.json`, `raw/DATACARD.md`, `raw/front_wide_cy.parquet`, `code/` builders) — staged
* local build: `C:/Users/Admin/tanitad-wt/_s2build/release/tanitad-v7-training-corpus/`
* register: `D-LABEL-GT` (ground truth) — release rides the pinned blob unchanged
