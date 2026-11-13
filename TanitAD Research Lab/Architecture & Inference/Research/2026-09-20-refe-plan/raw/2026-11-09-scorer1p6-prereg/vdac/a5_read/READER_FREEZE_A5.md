# READER_FREEZE_A5 (written BEFORE any HB / HC arm output exists on the dev box; none was opened)

UTC: 2026-11-13. Repo tip `e262e2cce34a9b1a30e273787309bd73e9107304` (read with `git show`; the frozen files below were exported byte-exact with `core.autocrlf=false`).
PK = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`. Interpreter: `C:/Users/Admin/venvs/tanitad/Scripts/python.exe` (Python 3.13.5, numpy 2.5.1; the one POD_RUN_PLAN step F names; the S1a READER_FREEZE.md names none, so UNVERIFIED that S1a used the same).

## 1. Hashes (sha256)
| file | sha256 |
|---|---|
| PK/eval/s1_read_v2.py (frozen) | 796aea21506e27d1646f5b7aa71b08089ced8c6d92a5b02429d35431692b09f4 (equals PREREG 7.1) |
| PK/raw/2026-11-09-scorer1p6-prereg/vdac/s1a_read/s1_read.py (frozen v1, via `S1_READ_V1`) | 58aa3a77b976a6fb4b08b4bc5126d572e8be8ef1fc37926d9c9ce0983eed718e (equals PREREG 7.1) |
| PK/eval/test_s1_read_v2.py (36 tests) | 26117b589cb1b337211b7fd61557cd728dc0078944ed4e8ff2e34b8058880ccb |
| a5_read/test_a5_reader_hbonly.py (new, 7 tests) | f1092441178e4bf52da56410141c0f1c37578fbb59d677c63d562c632b5b5730 |
| a5_read/manifest_a5.json (both pairs) | 1cf776d95c758726d582a5fa87de9866b44bd8426e9f0a5cedaa98c87070a869 |
| a5_read/manifest_a5_hb.json (HB read) | bd1c6f7effa9018c744d6d933e8f0f2996f8f39e43986958a78fedf4b704a4ab |
| a5_read/manifest_a5_hc.json (HC read) | ae10e96fbed673b001684ed327903281b4e2d4b69693eed9c5dc088905484d82 |
| a5_read/make_manifests_a5.py (generator) | 4dfe12d1b94ae01cb3d70310006a1f481ad4193e9a2d5291c0f0dcac90378e89 |
| a5_read/fill_manifest_sha.py | 0b6510a9f334e88d342fb0beeed3a6aa378940b8b941cf2299577bb5abe05720 |
| a5_read/a5_hc_s1f_subset.py (wrapper, NOT part of the frozen reader) | af12970ea8cc2c5b60e6d1503354c7675d401645c829e1ba9f4040b581807444 |

The three manifests carry `PENDING` sha256 sentinels for the arm files (5 per new arm, plus HB's `hb_init_b` which is HA's fixed value). The frozen `load_real` REFUSES a sentinel (sha mismatch), so an unfilled manifest fails closed. The manifest that is READ is the filled copy; its sha256 lands in the verdict's `manifest_sha256` field.

## 2. Per-file sha256: how it is handled (precedent followed)
S1a `inputs_manifest.json` and Stage 0 `manifest_s1f_v2.json` FIX every file's sha256 in the manifest; their note: the five HF npz were "verified equal to the LFS sha256 of Sayood/tanitad-refe-scorer1p6 at download". Same here: the HA rows are the fixed Stage 0 values; the arm rows are filled at pull time from an INDEPENDENT source (HF LFS sha256 of the npz pushed under `vdac/<arm lower>`; the pod `sha256sum` for `scores/head_pred_step0.npz`, which the egress does not push) with `fill_manifest_sha.py`, which hashes bytes only and refuses a mismatch. No array is opened by it.

## 3. Regression identity (PREREG 7.4)
* What the banked build compared: v2 on the S1a inputs vs `s1a_read.json`: banked `regression_identity_result.txt`: 213 leaf values, **0 mismatches**, verdict INCOMPLETE equal.
* **Re-run on the real inputs today: NOT DONE.** The S1a / Stage 0 input files (export npz, ha-s0 / ha-shuf npz, HA init heads) are no longer on the dev box. Looked in: `F:/tanitad_scratch/` (no vdac_s0, no vdac_s1a_scratch), `E:/Projects/TanitAD-artifacts/` (only `s1_read_cache/.../head_pred_step0.npz` of other arms), `E:/Archive`, `C:/Users/Admin/qland`, and a name `find` over `F:/` (nothing). Re-pulling needs HF (forbidden for this task). The manifests name the HA files at `F:/tanitad_scratch/a5read/pulled/ha/...`; the MM must re-pull them (their sha is fixed, the reader checks it) before the first read.
* What WAS re-checked: the banked artifacts with the banked comparison tool: `regression_identity.py` on `raw/2026-11-12-vdac-a5-build1/s1a_regression/s1_read_v2.json` vs `s1a_read.json`: 213 compared, 0 mismatches (`evidence/regression_identity_recheck_banked.txt`). The banked `s1_read_v2.json` carries `reader_sha256` 796aea21... = today's reader. This checks the banked pair; it is not a fresh execution of the reader on real data. Per PREREG 7.4 a failing identity on the MM's re-run after the re-pull means INVALID.

## 4. Tests and mutation arms (CPU, `CUDA_VISIBLE_DEVICES=-1`, `PYTHONIOENCODING=utf-8`, `S1_READ_V1` set)
* `test_s1_read_v2.py`: Ran 36 tests, OK, 9.2 s (`evidence/test_s1_read_v2_rerun.txt`); includes the v1-ported mutation tests.
* `test_a5_reader_hbonly.py`: Ran 7 tests, OK (`evidence/test_a5_reader_hbonly.txt`).
* Mutation arms, each with the mutation DISABLED in a scratch copy of the tool (the v1 copy re-hashed and its sha put into the v2 copy), full test file re-run; each goes RED:

| arm | disabled at | RED test | evidence |
|---|---|---|---|
| no_reweight | v1 `reweight=(mutate != "no_reweight")` -> `reweight=True` | `test_city_reweighting_known_value_and_mutations_go_red` | evidence/mutation_no_reweight.txt (2 failures: this one plus the sha-pin test, which fails only because the scratch v1 was changed) |
| frame_resample | v1 `if mutate == "frame_resample":` -> `if False:` | same test | evidence/mutation_frame_resample.txt (same 2) |
| unpaired_delta | v2 `if mutate == "unpaired_delta":` -> `if False:` | `test_mutation_unpaired_delta_goes_red` | evidence/mutation_unpaired_delta.txt (1 failure) |

## 5. Read commands (POD_RUN_PLAN step F; one command = one arm with its controls and its paired delta)
Pull first (`HF_HUB_DISABLE_XET=1`, tanitad venv) into `F:/tanitad_scratch/a5read/pulled/<arm lower>/` keeping pod-relative names; export the tip's PK tree (cards, h48_logs, `s1a_read/`, in_sample json) to `F:/tanitad_scratch/a5read/code/<PK>/`; fill the sha256; then:
```
python fill_manifest_sha.py --manifest manifest_a5_hb.json --shas shas_hb.json --out manifest_a5_hb.filled.json
export CUDA_VISIBLE_DEVICES=-1 PYTHONIOENCODING=utf-8
export S1_READ_V1="<PK>/raw/2026-11-09-scorer1p6-prereg/vdac/s1a_read/s1_read.py"
C:/Users/Admin/venvs/tanitad/Scripts/python.exe "<PK>/eval/s1_read_v2.py" --manifest manifest_a5_hb.filled.json --out <read_dir>/hb-s0
# HC, only when A5HC ran:
python fill_manifest_sha.py --manifest manifest_a5_hc.json --shas shas_hc.json --out manifest_a5_hc.filled.json
C:/Users/Admin/venvs/tanitad/Scripts/python.exe "<PK>/eval/s1_read_v2.py" --manifest manifest_a5_hc.filled.json --out <read_dir>/hc-s0
S1_READ_V2_DIR="<PK>/eval" C:/Users/Admin/venvs/tanitad/Scripts/python.exe a5_hc_s1f_subset.py --manifest manifest_a5_hc.filled.json --init-key hc_init_a --trained HC-s0 --out <read_dir>/hc-s0-s1f
```
`manifest_a5.json` (both pairs in one read) works only when BOTH arms exist; use the per-arm manifests when one arm is NOT RUN. `make_manifests_a5.py <dir>` regenerates all three.

## 6. UNVERIFIED / limits
1. HB alone: the frozen reader runs with only the HB pair (synthetic tests `test_hb_only_manifest_reads_with_its_pair_and_no_hc`, `test_pair_naming_an_arm_that_is_not_in_the_table...`). It does NOT run if the manifest names HC rows whose files are absent (load_real hashes every file in `files`: FileNotFoundError) or a pair naming an absent arm (`ValueError` in `run_read`): hence the per-arm manifests. Not verified on real files (load_real needs the real cards, store and export).
2. HC's `init_b` absent: PREREG 7.5's UNVERIFIED is ANSWERED: the frozen reader does NOT accept an init arm covering only the 1,705 held-out frames: `s1_read_v2.py` load_real, `if not got.all(): raise SystemExit("REFUSED: arm ... does not cover the export's N keys")` (test `test_load_real_refuses_a_partial_init_arm_source_level`). So `manifest_a5_hc.json` has NO HC init arm (HC-s0 verdict INCOMPLETE, S1-F unread, as the frozen test `test_perfect_1m_head_reads_one_on_the_reference_grid_and_the_1m_grid` shows), and HC's S1-F on the 1,705 frames comes from the wrapper `a5_hc_s1f_subset.py` (frozen `run_read` on the subset). Design choice for the MM to confirm: the bootstrap is redrawn over the subset's logs. The wrapper's real-data `main()` is untested (only `s1f_on_subset` is).
3. RISK found by a synthetic test: S1-P1 is city-reweighted; a subset that lacks a city leaves it NaN (S1-F not readable; the wrapper flags `p1_defined: false`). Whether the 1,705 held-out frames cover all 4 cities is UNKNOWN to me.
4. `in_sample` file: the repo copy hashes 1547bd30...; the Stage 0 manifest said 5ab0c72c.... I used the repo copy (its value 0.7864227917638612 equals `s1f_read_v2.json`). Why they differ: UNVERIFIED.
5. The HA input files and all A5 pulled paths do not exist yet. The PREREG 7.5 precondition (HB `head_pred_step0` equals HA `head_init0_a` on mask sign at >= 99.9 % of cells) is a read-time check not done here.
6. Arm directory names `hb-s0`, `hb-shuf`, `hc-s0`, `hc-shuf` follow POD_RUN_PLAN "<arm lower>"; the score file names follow step F verbatim.
