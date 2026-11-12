# FREEZE_BUILD1 (A5 build order items 1 and 2: `s1_read_v2.py`, `vdac_stage0.py`)

UTC: 2026-11-12T16:01Z (freeze written after the last test run; no real model output was read: the only real-data computation is the S1a regression identity). CPU only, `CUDA_VISIBLE_DEVICES=-1`.
Source of the frozen reader and cards: git mirror tip `52114a00585a1e8c2bf01f6bd0ed11d2ab584c3d`. Evidence class of every line: MEASURED (this build, artifact paths below) unless marked.
`PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. A5 draft status unchanged: DRAFT, not frozen; these two tools are what its section 7 items 1 and 2 ask for.

## 1. Files and sha256

| file | sha256 | lines |
|---|---|---|
| `PK/eval/s1_read_v2.py` | `796aea21506e27d1646f5b7aa71b08089ced8c6d92a5b02429d35431692b09f4` | 539 |
| `PK/eval/test_s1_read_v2.py` | `26117b589cb1b337211b7fd61557cd728dc0078944ed4e8ff2e34b8058880ccb` | 565 |
| `PK/refe/vdac_stage0.py` | `04a3d5f4dde23dd669167c18a6090cb16a37c3a680ee3351729b4fd8b51dc2b7` | 518 |
| `PK/refe/test_vdac_stage0.py` | `524a879b46cd94b16f1c414f94a5c0cf5463e4be81b2727ab807dbbad5edf10c` | 494 |
| frozen S1a reader (imported, unchanged, sha asserted at import by both tools) `PK/raw/2026-11-09-scorer1p6-prereg/vdac/s1a_read/s1_read.py` | `58aa3a77b976a6fb4b08b4bc5126d572e8be8ef1fc37926d9c9ce0983eed718e` | 663 |
| S1a inputs manifest (original) | `e9068856f117103571bdda6d771eeafb844ffc55c8959f5ebb8c467ad55a00f0` (the repo's `READER_FREEZE.md` lists `70b8a8fa...` for it: that is the manifest at the first freeze, before the 15:26 fix; the file at the tip hashes as shown. UNRECONCILED, immaterial to the identity: the identity is on the input FILES' sha256) | |
| the S1a-run manifest used here (paths re-pointed to this build's downloads) `raw/2026-11-12-vdac-a5-build1/s1a_regression/inputs_manifest_v2_run.json` | sha256 in `s1_read_v2.json` -> `manifest_sha256` `33db4608abdbe348f61f6a2b3356310658f8e269cf91589faabca3e75564a8a4` | |
| **in-sample frame list** | **NOT FROZEN HERE, see section 4.** The rule is frozen in `vdac_stage0.py`'s `draw_list`; a list built from the label-store universe as a stand-in is banked as `raw/2026-11-12-vdac-a5-build1/INDICATIVE_store_universe_list/stage0_frames.tsv`, sha256 `9d74a184d4f0d0be98dea6cc7a012bcda3735cefddf172e5d3f7cdc81a00733d` (INDICATIVE ONLY: its candidate pool is all 103,139 stored keys, not the P6 60,000-set bank, so it may contain frames the head never trained on) | 2,048 rows |

## 2. What changed relative to the frozen reader (and what did not)

`s1_read_v2.py` imports the frozen module and calls its functions (`frame_stats`, `p1_block`, `q_block`, `verdict`, `diff_block`, `boot_weights`, `city_iou_read`, ...); nothing is copied; a test asserts object identity. Reader changes, exactly the A5 draft 2.3 list:
1. manifest-driven arm table (names no longer hardcoded; each trained arm names its own SHUF and its own initial head; the frozen `verdict()` is applied per trained arm with that arm's own C-SHUF and its own y-flip as C-FLIP); a manifest with no `arms` key (S1a's) is read exactly as v1 reads it;
2. paired blocks X minus REF on S1-P1 and S1-Q through the same log resamples; X-LIFT = both lower 95 % bounds > 0 (the draft's definition);
3. HC grid: cell 1.0 arms ([N,80,64]). S1-P1 on the 2 m reference grid (mean of the four sigmoids, mask = mean > 0.5, strict), S1-Q with the 1 m mask (EDT at cell 1.0, bilinear at 1 m), the 1 m IoU vs the 1 m label mask reported;
4. an arm may carry a supplied `in_sample_iou` (a sha-verified file, Stage 0's output) so the frozen `verdict()` splits S1-FIT / S1-WEAK.
Added on top (reporting only, does not alter any v1 output): `x_row`, the draft section 4 rows X-GO / X-COMP / X-BOUND / X-LIFT / X-NULL / INVALID / INCOMPLETE applied mechanically to the per-arm numbers. NOT tested against real HB / HC data (none exists).

## 3. Results

| check | result | artifact |
|---|---|---|
| **S1a regression identity** (v2 on the S1a inputs, re-downloaded from HF `Sayood/tanitad-refe-scorer1p6`, all 6 npz + export json sha256 equal to `inputs_manifest.json`; the 5 cards equal their frozen sha256; H48 list `2ae13935...`) | **213 leaf values compared, 0 mismatches, exact float equality and type**: S1-P1 0.7801999608707509 [0.7587085552669882, 0.7974453536337924]; S1-Q 0.8332411521287983 [0.7950626583525341, 0.8690684744479363]; per city LV 0.8914 / BOS 0.7394 / PIT 0.7234 / SG 0.6960; bands, C-SHUF 0.4468, C-FLIP 0.7056, true-field AUC 0.9408, sigma_b, near-boundary MAE/RMS, all-48 sensitivity, controls, verdict **INCOMPLETE** all reproduced. The tables markdown is identical to the repo's `s1a_read_tables.md` byte for byte. Excluded from the comparison: `tool`, `wall_s`, `reader_sha256`, `manifest_sha256`, `audit` paths (all expected to differ); v2-only keys: `per_arm`, `paired`, `frozen_v1_reader_sha256`. A second run through the explicit v2 arm-table manifest (same files) gives the same 213 values, 0 mismatches. | `raw/2026-11-12-vdac-a5-build1/regression_identity_result.txt`, `regression_identity.py`, `s1a_regression/s1_read_v2.json`, `s1_read_v2_tables.md` |
| `test_s1_read_v2.py` | **36 tests OK** = the 18 frozen v1 tests ported (bodies unchanged, both v1 mutation arms `no_reweight` / `frame_resample` still RED) + 18 new (arm table, per-arm controls, paired delta point / CI / lift / self-delta exactly zero, the `unpaired_delta` mutation, x_row, supplied in-sample, 1 m pooling in probability space, 1 m = 2 m on an upsampled head, 1 m perfect head, refusals, sha refusal of a changed frozen reader) | `test_results_s1_read_v2.txt` |
| `test_vdac_stage0.py` | **22 tests OK** (synthetic label store and banks): draw rule vs an independent computation, purity under row shuffle, ineligible rows (step, rank, no label, H48 / SFT) never drawn, duplicates refused, the verifier refuses 8 defects, 0.85 read mechanically on both sides (0.85 / 0.8500001 -> FIT, 0.8499999 -> WEAK), known IoUs 0.882 (FIT) and 0.829 (WEAK), C0-FLIP blind instrument -> INVALID, C0-ID failure -> INVALID, C0-FLOOR 0.125, reweighting known value 1 - 0.4 x 3731/12146, input refusals, CLI make-list + read incl. `--dry-run` writes nothing, doctored list with the right sha still refused | `test_results_vdac_stage0.txt` |
| mutations (each must go RED) | 5 source-level mutations of the tools all RED; plus the built-in regression arms | `mutation_results.txt` |
| `make-list` dry-run and run on the REAL store metadata (all shards re-hashed, 1,192 logs, city counts equal the manifest's 67,593 / 12,925 / 10,807 / 11,814) | the draw rule runs on real keys: 2,048 frames from 635 distinct logs; the excluded set (H48 union SFT-24) is 48 logs, i.e. the SFT-24 is a subset of H48 | `INDICATIVE_store_universe_list/stage0_frames.json` |

## 4. Not done, or decided differently from the draft's wording (read before relying on either tool)

1. **The real frame list cannot exist off the pod.** The draft draws "labelled bank rows" of P6's 60,000-set bank, and those rows live only on the pod (`s1/HANDOFF.md`: "the bank rows live on the pod"). `make-list` takes the bank's (log, token, step, rank) rows as a keys file, and must be run there (CPU, seconds) BEFORE any forward; its output carries the sha256 the draft wants frozen. The freeze of item 2 is therefore the rule + the tool, and the list sha256 is to be recorded from the pod run. The indicative list above is NOT a substitute (a frame outside the bank would be out-of-sample).
2. **The tool does not run the model.** The forwards (2,048 in-sample, 256 C0-ID, the seed-0 initial head on the 1,705 + 1,502 frames) are a separate producer, `--stage0-*` in the trainer (READINESS item S0; it owns the `fwd()` closure and needs the `aux_drv.pt` loader, H6). This tool consumes exactly the reader's npz schema and refuses anything else. That producer is NOT built.
3. **S1-F: the 1,705-frame-subset branch is not built.** If the rebuilt seed-0 head fails the 99.9 % sign check against `head_pred_step0.npz`, `s1f` stops and says so instead of reading the subset.
4. The draft leaves open and these tools fix (each reported in the outputs): the 256 C0-ID frames = the 256 lowest sha256('<log>|<token>') keys of the banked `head_pred_update10000.npz` rows (keys only); C0-ID IoU = pooled [0,40) IoU over those frames, not city-reweighted; the 1 m head's SDF channel is block-mean pooled (reported near-boundary MAE only, not deciding); HB's initial-head arm for S1-F can point at HA's initial-head files (HB at step 0 reads P6's ctx bit-for-bit, draft 2.1) but the draft does not say so.
5. `READER_FREEZE.md`'s manifest sha (`70b8a8fa...`) differs from the tip's `inputs_manifest.json` (`e9068856...`), see section 1.

## 5. Scratch and deletions (CLAUDE.md rule, literal paths)
Scratch `F:/tanitad_scratch/vdac_build1_scratch/` (124 MB: HF downloads 117 MB, a 6 MB keys file, cards / sources 0.4 MB, mutation copies 0.1 MB) deleted at the end of the job; every byte is re-creatable (HF `Sayood/tanitad-refe-scorer1p6` + the git tip). Logged in the hand-back.
