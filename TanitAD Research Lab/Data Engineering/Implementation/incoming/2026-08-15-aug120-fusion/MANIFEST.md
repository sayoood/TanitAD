# Deliverable manifest — aug120 PH1 fusion, 2026-08-15

Result doc: `AUG120_FUSION_RESULT.md`. Forward scope: `NEXT_4472_BUILD_INPUTS.md`.
⛔ **Nothing here lives in only one place** except where the "single copy" column says so.

## Artifacts

| artifact | where it lives | single copy? |
|---|---|---|
| **201 fused PH1 records** (`<clip_id>.json`) + `_summary.json` + `_batch_accounting.json` + `_label_sources.json` — 204 files, 4.26 MB | **HF `Sayood/tanitad-ph0-aug120/fused_aug120/`** (far-side verified: 204/204, 0 missing/extra/size-mismatch, 6/6 byte round-trip) · working copy in the session scratchpad | no — HF is the durable home; the per-record payload is deliberately NOT in git (it is a derived label set of that dataset's own class, beside `fused_w120val/`) |
| `_summary.json` (aggregate counts) | `repo:TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-15-aug120-fusion/raw/fused_aug120_summary.json` · HF | no |
| `_batch_accounting.json` (per-batch in→out) | `repo:…/raw/fused_aug120_batch_accounting.json` · HF | no |
| `_label_sources.json` (per-clip v2/sam3 source tag + policy) | `repo:…/raw/fused_aug120_label_sources.json` · HF | no |
| `aug120_coverage.json` (todo list, union ids, the 115 no-sam3 clips, batch_00184's 8) | `repo:…/raw/aug120_coverage.json` | **yes — repo only** |
| **Run driver** `aug120_fuse_run.py` (merge policy + assertions + accounting) | `repo:…/code/aug120_fuse_run.py` | **yes — repo only** |
| Pull/verify/push helpers (`hf_pull_labels.py`, `hf_pull_ego.py`, `hf_push_fused.py`, `analyze_coverage.py`) | `repo:…/code/` | **yes — repo only** |
| **Fuser change**: `--missing-sam3-ok`, absent-marker, not_computable degradation, loud refusal | `repo:stack/scripts/ph1_fuse.py` | **yes — repo only** |
| **Pipeline fix**: `--n` passed to `ph0_sam3.py` | `repo:stack/scripts/aug120_pipeline.py` | **yes — repo only** |
| **Tests** (+2 pinning the named partial; + the missing scripts-dir `sys.path` insert) | `repo:stack/tests/test_ph1_fuse.py` | **yes — repo only** |

## Git status

All repo paths above are **staged, not committed** (`git add`, verified with
`git ls-files --cached`). No commit, no push, no branch switch. Branch: `agent/arch-inf-20260803`.

⛔ **No gated source bytes entered git** — the staged artifacts are derived labels and ids
(`clip_id` UUIDs) only. No video, no cache, no frames. Payload moved: ~34 MB down (label JSON,
ego npz, `records.parquet`, corpus listing), 4.3 MB up.

## Integration needed (escalated, not filed in a README)

1. **`stack/scripts/ph1_fuse.py` changed** — the fuser now **refuses** a partial SAM3 leg unless
   the operator names it. Any existing caller that fuses a v2 set with incomplete SAM3 will now
   fail loudly instead of emitting unmarked records. That is intended; callers must pass
   `--missing-sam3-ok REASON`.
2. **One GPU pod for ~30 min** closes the perception gap (115 clips, §9 of the result doc). This
   is the only compute this stream needs.
3. **`fused_w120val/` carries 4 unmarked SAM3-absent records** — a correction would re-baseline
   the published 175/41/56. PI decision.
4. **Runbook §6.11 understates the gap**: it names `batch_00184` (8 clips); the true figure is
   **115 of 201**. `STOP_2026-08-15_RESUME_RUNBOOK.md §4.3a` should be updated to say fusion is
   done and the SAM3 re-run is what remains.
