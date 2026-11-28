# FREEZE_BUILD2 (A5 build order, the forward producer: `vdac_stage0_forward.py`, its tests, `vdac_stage0_pod.sh`, `POD_PLAN.md`)

UTC: 2026-11-12T16:33Z (written after the last test run, which ran on exactly the files hashed below). CPU only, `CUDA_VISIBLE_DEVICES=-1`, interpreter `C:/Users/Admin/venvs/tanitad` (Python 3.13.5, torch 2.11.0+cu128, numpy 2.5.1). No real model output, no real data and no pod were touched.
Source of everything imported: git mirror tip `2207ef03eeae8d048c7d02ef8c8410fe08a7d101` (exported read-only into scratch; `PK` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`). Evidence class of every line: MEASURED (this build, artifact paths below) unless marked.
A5 draft status unchanged: DRAFT, not frozen. This build closes READINESS items S0 (the producer the Stage 0 consumer needs) and H6's `aux_drv.pt` loader for the HA head (`aux_drv/1`); HB's `aux_drv/2` is not built.

## 1. Files and sha256

| file | sha256 | lines |
|---|---|---|
| `PK/refe/vdac_stage0_forward.py` | `e6f4911c1484a88e9013837f4e07016de7d5fbc44ece8455e70541a3a9666bba` | 644 |
| `PK/refe/test_vdac_stage0_forward.py` | `e5733a2b6433035ea719412eee5c574769f0e790582ef4ecedac82453571caa1` | 681 |
| `PK/refe/vdac_stage0_pod.sh` | `bea6279f6b6ff48e274526b89c4b4ad5b01eb3f80f4e88b767cf2e14990b281d` | 73 |
| `PK/raw/2026-11-12-vdac-a5-build2/POD_PLAN.md` | `0bb6ed59ac29f5ed21e299a60192a77b36230ebd702e90fd94535a6cbf6ef679` | |
| `.../mutation_arms.py` / `mutation_results.txt` / `test_results_vdac_stage0_forward.txt` / `ha_s0_argv_MODE_argv.txt` | `f4887cdf...` / `83c6d5b6...` / `879aad28...` / `76f424b6...` | |
| the consumer, imported unchanged (build 1) `PK/refe/vdac_stage0.py` | `04a3d5f4dde23dd669167c18a6090cb16a37c3a680ee3351729b4fd8b51dc2b7` = FREEZE_BUILD1's value; tree copy re-hashed equal to the tip blob | 518 |

**No pinned file was edited.** `scorer_finetune.py` (`e3e08ac0...`), `model.py` (`e6011cd8...`), `train.py` (`eafd03e5...`), `ckpt_io.py` (`240de348...`), `aux_drivable_head.py` (`66dff7da...`), `launch_gate.py` (`3eebe84f...`), `gate_checks.py` (`959f83c2...`): sha256 prefixes of the scratch copies equal the tip blobs (checked at freeze). The hold directory contains only the files above (new files); `scorer_finetune.py` is imported for `build_parser()` only. A test pins that the tool has exactly one text writer (the bank-keys TSV) and writes only into `--out-dir`.

## 2. Test results (`test_results_vdac_stage0_forward.txt`)

| check | result |
|---|---|
| `test_vdac_stage0_forward.py` | **33 tests OK** (73 s) |
| `test_vdac_stage0.py` (the consumer's own suite, same scratch tree, unchanged) | **22 tests OK** (re-run at the end) |
| `bash -n vdac_stage0_pod.sh`; `MODE=plan` | clean; the printed commands are the ones in POD_PLAN 3.2 |

What the 33 tests assert (synthetic: a TINY `REFe` built with the real class, the REAL 2,440,450-parameter head class, the consumer's synthetic label store `Fx`, a synthetic bank-shaped dataset with decoy rows at step 1 / rank 3):
* **loader round trip:** a head saved in the trainer's `aux_drv/1` format loads with every tensor bit-for-bit equal (`torch.equal`, not allclose); 11 loader refusals (format, feed, head config, update count, store sha, shuffled head, strict-load key, data-event facts x2, untrained head == initial head, wrong aux sha, absent file).
* **seed-0 rebuild:** two builds with the global RNG moved in between are bit-identical; equal to `AUX.build_head(seed 0)`; differs from a seed-1 build; a run whose seed is not 0 is refused.
* **config reproduction:** the REAL HA-s0 argv (`ARM=HA-s0 MODE=argv bash vdac_pod.sh` at the tip, saved as `ha_s0_argv_MODE_argv.txt`) parses with the trainer's own parser (seed 0, 10,000 updates, depth 2, tap visual, feed off, w 0.1, vitl16, `--require-pdm`); refusals for an empty / unparsable / feed-on / shuffled / split-half argv; the argv must equal the namespace recorded in `model_sft_A.pt` (a `--batch` or `--lr` difference is refused) and its `--out` must be the run dir.
* **forward:** the written predictions equal an INDEPENDENT forward (a plain batch-1 fp32 loop with separate head objects) to 1e-4 and differ from the initial head's by more than 1e-3; batch 2 vs 9 agree; two runs are bit-identical; the run directory is never written; outputs are never overwritten.
* **acceptance by the consumer:** `vdac_stage0.py read` on the written files gives its own C0-ID control `ok = True` (n = 6 frames) and 32 in-sample frames; `s1f --dry-run` accepts the rebuilt initial head (sign agreement 1.0, `"ok": true`) and builds its v2 manifest from the written files; the tool re-validates every file it writes with the consumer's own `load_pred`.
* **frames:** frame-list sha mismatch refused (wrong, short, and the C0-ID list); a doctored list with a correct declared sha is refused by the consumer's `verify_list`; a frame with no bank row refused before the forward; the trainer's `tr_idx` / `ho_idx` filters and the h48x first-row rule reproduced on a fake bank; `bank-keys` output feeds the consumer's `make-list` and yields the SAME frame-list sha as the fixture's own bank, and is refused when the row count differs from the run's data event `train_sets`.
* **GPU guard:** `nvidia-smi` parsing; busy => REFUSED by default; refused BEFORE any bank or model is built; checked again right before the forward (idle at start, busy after the banks load => refused); unverifiable => refused; `--dry-run` never initialises CUDA (`torch.cuda.is_initialized()` False) and reports what a real run would do.

## 3. Mutation arms (`mutation_results.txt`, `mutation_arms.py`): every one RED

| arm (`VDAC_FWD_MUTATE`) | what it breaks | how it went RED |
|---|---|---|
| `skip_head_load` | the head weights are not loaded | the in-tool guard refuses ("bit-identical to a freshly built seed-0 head in ALL 43 tensors") |
| `skip_head_load,no_guard` | not loaded AND the guard off: predictions equal the initial head's | the check's reproduction control fails (rc 3, C0-ID vs the banked file) |
| `y_flip_output` | writes the y-flipped prediction (schema-valid) | independent-forward comparison: max abs diff 1.69 |
| `misalign_keys` | keys rotated by one against the predictions (schema-valid) | independent-forward comparison: max abs diff 1.95 |
| `init_seed1` | the initial head built with seed 1 | S1-F reproduction against the pod's seed-0 step 0 fails (rc 3) |
| `skip_list_sha` | the frame-list sha check skipped | a wrong declared sha is accepted -> the refusal check fails |
| `skip_gpu_guard` | the busy-GPU refusal skipped | a busy GPU is accepted -> the refusal check fails |

Not a mutation arm, by design: processing order. The forward is order- and batch-invariant (tested), so reversing the order is not a regression.

## 4. Not done, or decided differently (read before relying on this)

1. **Nothing ran on real data or on the pod.** `real_banks` / `real_context` copy the trainer's construction (`scorer_finetune.py:1736-1760, 1771-1776`) and are exercised only through their pure helpers; the first real execution is `bank-keys` on the pod (POD_PLAN risk 2).
2. **`fwd()` is replicated, not imported** (it is a closure inside `main()`; importing it would need an edit to the pinned trainer, which the task forbids). The replicated part is the head-relevant path: `model(img, ego, goal, calib, score_extra=cxy)` under bf16 autocast, the scoring decoder's own context tap, `aux.predict`. Parity with the trainer's dumps is the C0-ID / S1-F control, not a code comparison.
3. **bf16 sign flips can fail C0-ID without a tool defect** (POD_PLAN risk 1); the tool uses the trainer's batch size (16) and index order to give it the best chance, and reports the exact agreement and max logit difference.
4. **The task's order "S1-F, then `read`" is reversed** (`s1f` needs `read`'s `in_sample_iou.json`).
5. **`head_pred_step0.npz` has no earlier recorded sha256**; its sha is a RUNTIME record (POD_PLAN 3.2).
6. **The sibling pause wrapper (`vdac_stage0_pause_p7.sh`) is specified as an edit list, not built or tested** (POD_PLAN 3.1).
7. HB's `aux_drv/2` loader, the 1,705-frame-subset S1-F branch and HC's 1 m heads are not built (unchanged from FREEZE_BUILD1 section 4 and READINESS).
8. The estimate in POD_PLAN section 4 rests on one INHERITED number (0.066 s per frame forward, an assumption in the A5 draft); no timing of the h48x forward exists in the repo.

## 5. Scratch and deletions (CLAUDE.md rule, literal paths)
Scratch `F:/tanitad_scratch/vdac_build2_scratch/` (git-tip exports of `refe/`, `eval/`, the frozen reader and `splits/`: about 10 MB; test copies; results). Every byte is re-creatable from the git tip or from the hold directory. Deletion is logged in the hand-back.
