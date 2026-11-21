# HANDOFF LC (the LoRA arm for the stage-5 driver) -- 2026-11-07. Agent LC (Sonnet), dev box, CPU only, nothing launched, nothing staged, nothing committed.

MEASURED = ours on the dev box. UNVERIFIED = needs box 2, the pod, or a real L checkpoint. Frozen and untouched: everything under `code/`, `stage4/`, `stage5/`, `audit/`, `tests/`.

## STATE (what exists, all NEW, under `motion_c/t2v8f/`)
* `code_lc/t2v8f_lora_merge.py`: the standalone fold (`fold_lora`, `parse_lora`, `model_digest`, `build_pin_for`, `check_lora_encoder_block`, `preflight`, `lora_fold_active`). Training-side expression op for op:
  `bf16( fp32(W) + (fp32(B) @ fp32(A)) * (alpha/r) )`, CPU, q and v of 24 blocks. Imports only torch (+ `t2a_v8f_encoder` lazily).
* `code_lc/t2v8f_stage5_cell.py` (copy): `arm5_from_manifest` -> `(name, ckpt_file, encoder)`, `install5(R, name, ckpt_file, encoder=None)`; kind `dinov3_lora_merged` wraps the ORIGINAL `build_arm` in `lora_fold_active`
  (class `InLoopEncoder` swapped for a subclass for one build; `AutoModel.from_pretrained` hooked inside its `__init__`: fold after load, BEFORE `.to(device)`; the digest asserted in the hook and again on the finished encoder).
  Frozen kind: the unchanged wrapper, no LoRA import. Unknown kind: `ENCODER_UNSUPPORTED` (SystemExit).
* `code_lc/t2v8f_stage5.py` (copy): `check_arm5` accepts `dinov3_lora_merged` via `check_lora_encoder_block` (named: LORA_BAD_FIELD, LORA_SOURCE_MISMATCH, LORA_PIN_INCONSISTENT); unknown kinds still `ENCODER_UNSUPPORTED`;
  preflight (off `--stub`) dry-runs the fold of both seeds on box 2's CPU (`MERGED_SHA_MISMATCH` / `LORA_BASE_MISMATCH` named before any cell); `LORA_MODULE_NOT_IN_OVERLAY`; result keys `arm_encoder`, `arm_merged_qv_sha256` for LoRA arms only.
* `code_lc/make_stage5.py` (copy): `--encoder lora --lora-ckpt-s0 P --lora-ckpt-s1 P [--snapshot DIR]`; folds on this CPU, writes `arm5.encoder` (r, alpha, targets, blocks, model_id, base sha, source ckpt sha per seed,
  lora tensors sha, merged q/v digest + 48 per-tensor digests per seed, fold description), ships `code/t2v8f_lora_merge.py`. Frozen build: unchanged.
* `code_lc/check_l_ckpt.py`: format + fold check of a REAL L checkpoint (see its docstring); does not fetch.
* `tests_lc/lc_common.py`, `tests_lc/ref/` (verbatim tip copies: `v8f_lora_encoder.py` blob 749fd497, the two builder files), `tests_lc/test_t2v8f_lc.py` (see TEST STATUS).

## ANSWERS
* Head weights: `V8FMiniT2Policy` IGNORES the `lora` key (and `opt`, `encoder_levers`): `t2a_v8f_policy.py:101` `torch.load` of the whole blob, `:102` `V8FMiniConfig(**ck["cfg"])`, `:106` `load_state_dict(ck["model"])` (strict, B's layout).
  So the L blob ships unsplit as `pkg/ckpt/LC_s<k>.pt`; no split file needed. (Blob ~ heads + opt + 12.6 MB lora ESTIMATED.)
* Trainer save: `v8f_mini_train.py:2659-2675` `save()` writes `blob["lora"], blob["encoder_levers"] = enc.lora_state_dict(), enc.built_levers()` beside `model`/`opt`; `lora_state_dict()` = `{meta: {r, alpha, targets, blocks, dropout, seed, weights_sha256}, tensors: {<backbone prefix>.layer.i.attention.{q,v}_proj.lora_{A,B}}}`.
  HF 5.15 names carry a `model.` prefix (`backbone.model.layer.0...`); the fold matches by suffix so any prefix works.

## TEST STATUS (MEASURED, dev box CPU, real DINOv3-L bf16, real banked cylinder frame `refcv5cmp/data/eval/01be5919-...v2ep.pt` raw frame 0)
`logs_lc/pytest_lc_final.log`: **18 passed in 619.8 s** (`tests_lc/`: 16 in `test_t2v8f_lc.py` = 8 checks + 8 RED arms; 2 in `test_t2v8f_lc_identity.py` = identity + 1 RED). After a wording-only edit of 3 message strings: `logs_lc/pytest_lc_after_wording.log` 4 passed (the manifest and frozen-path tests).
Earlier runs: `pytest_lc_run1_restorecheck_defect.log` (my own test asserted `"from_pretrained" not in vars(AutoModel)`; transformers defines it there, so that check failed; every guard under test was refused as designed; the check now compares with the pre-call value), `pytest_lc_run2.log` 14 green, `pytest_lc_identity_run1.log` 2 green, `pytest_lc_frozen_build.log` 2 green.
* (a) B = 0: folded q/v digests equal the frozen weights' (48 of 48), and the folded `InLoopEncoder` tokens equal the frozen `InLoopEncoder` tokens BYTE FOR BYTE. RED: the non-zero LoRA presented as the identity fails.
* (b) B ~ N(0, 0.01^2): exactly 48 tensors change, every other parameter identical, tokens move (rel-L2 vs frozen 0.0335, `probes/smoke.log`). RED: the "no change" expectation fails.
* (c) epsilon = 0: T2-side merged tokens equal the training side's `merged_encoder().pooled` byte for byte (max abs 0.0, rel-L2 0.0; same CPU, same frame, batch 1, bf16), and the folded weights digest equals the training side's `merge()` (48 of 48). RED `skip_fold`: fold = no-op fails the byte equality.
  The card's tau_device 1e-3 (rel-L2) is the bar for a DIFFERENT device (box 2's GPU): UNVERIFIED there.
* (d) six faults refused BY NAME: fold skipped (`MERGED_SHA_MISMATCH (after the fold`), fold skipped + in-hook check bypassed (`... (on the finished encoder)`), one wrong per-tensor pin (the tensor `layer.5.attention.v_proj.weight` is named), wrong blob (`HASH_MISMATCH_LORA_SOURCE`),
  fp32 / non-CPU model (`FOLD_WRONG_PLACE`), malformed `lora` (`LORA_TENSOR_SET`, `LORA_META_MISMATCH`); `InLoopEncoder` and `AutoModel.from_pretrained` restored afterwards. RED: guards disabled.
* manifest rules: 8 planted faults + `LORA_SOURCE_MISMATCH` each NAMED; unknown kinds (`dinov3_lora`, None, `frozen`) still `ENCODER_UNSUPPORTED` in the driver AND the cell. RED: faults not applied.
* (e) frozen path = code/: `check_manifest5` equal to code/'s on both REAL EGAIN manifests + 11 planted faults (only the text of ENCODER_UNSUPPORTED differs; its code and all other reasons are equal); `cell_order5`, `arm_names`, `passes5`, `NAME_RESERVED`, `REQUIRED_*` equal;
  a frozen manifest through `install5` leaves `InLoopEncoder` (class and `__init__`) untouched and imports no LoRA module; `make_stage5.build` (frozen, EGAIN a10 checkpoints) gives the same MANIFEST5 as `stage5/EGAIN_a10` except the sha256 of the two code files (both `code_lc/` copies);
  THREE fresh processes on the stage-4 rehearsal fixture (kit scene 7c72937c, T2): stage 4's cell C0C_s0, code/'s stage-5 cell C0ID_s0, code_lc's stage-5 cell C0ID_s0 give equal plans and extra bit for bit. RED: C0's seed-1 checkpoint as seed 0 differs.
* (f) the builder, `--encoder lora`, on the REAL stage-4 tar with synthetic L blobs (REAL trained heads of two banked checkpoints + a synthetic `lora`): 7 tar members, `check_manifest5 == []`, pins equal the training side's `merge()` digests recomputed independently, code files LF-only and equal to `code_lc/`. The CLI refuses `--ckpt-s*` with `--encoder lora` and the converse.
* `check_l_ckpt.py`: PASS on a synthetic blob (format + fold; its merged digest equals the builder's pin), FAIL by name on a B checkpoint (no `lora` / `encoder_levers`); `logs_lc/check_l_ckpt_synthetic_{fold,nofold}.json`.

## UNVERIFIED (each needs box 2, the pod, a GPU, or a real L checkpoint)
1. A REAL L checkpoint was never read: all blobs are synthetic `lora` entries in the trainer's own key format (names taken from the training module's `lora_state_dict()`), on real heads. `check_l_ckpt.py` is the first-contact check; a format difference in the real file shows there.
2. Cross-CPU digest: the pin is a function of this CPU's fp32 `B @ A`; box 2's CPU may differ in the last bit of a few weights (ESTIMATED under one bf16 weight in 5e7 in expectation, not measured). The driver preflight (off `--stub`) and `check_l_ckpt.py --expect-merged-sha256` refuse by name rather than run another encoder.
3. GPU numerics: the heads were trained on UNMERGED LoRA tokens on the pod's GPU; at T2 they read the merged encoder on box 2's GPU. On one CPU the merged and training-side merged tokens are identical (c); merged vs unmerged and CPU vs GPU are the card's tau_device question (1e-3) and are not measured here.
4. The driver preflight fold on box 2 (two blobs + the model, ~30 s ESTIMATED from this box) and the per-cell fold (~2 to 3 s ESTIMATED per cell x 248): wall ~1.7 h becomes ~+10 min ESTIMATED. A `--stub` rehearsal of an LC package through the whole driver (`all --stub --inproc`) was NOT run: the stub encoder has no weights, so the cell skips the fold there (stated in a code comment); the fold is covered by the real-weights tests above.
5. Blob size: the real L blob carries `opt`; LC adds the lora Adam state: footprint ESTIMATED larger by ~40 MB per seed; `footprint5` computes it from the real file sizes at build time.
6. Records carry no LoRA stamp (record keys unchanged on purpose: the pinned reader reads them); the pin lives in MANIFEST5 (`arm5.encoder`) and `STAGE5_RESULT.json` (`arm_encoder`, `arm_merged_qv_sha256`).
7. `tests_lc/ref/dinov3_fp8_encode_ship.py` is a verbatim tip copy and holds one word from the banned-word list (not editable without breaking the verbatim pin).
8. Scratch ended at 3.4 GB (limit 3 GB; the final full run rebuilt the fixtures): all moved to `E:/Projects/TanitAD-artifacts/scratch_lc_t2v8f/_aside_20261107/` (PI cleanup). One of my own marker files (`logs_lc/run1_FINISHED.txt`) was removed with `rm` before I applied the move-aside rule; its content (rc=1, the restore-check defect) is quoted above.

## NEXT (exact)
1. Master Mind, after the BPC build: `code_lc/` holds the three changed files + `t2v8f_lora_merge.py` + `check_l_ckpt.py`. Build the LC package with `python -I code_lc/make_stage5.py build --arm LC --encoder lora --lora-ckpt-s0 <L_s0> --lora-ckpt-s1 <L_s1>` (the overlay then ships `code_lc/t2v8f_stage5*.py` and `t2v8f_lora_merge.py`). `code/` stays as the running MC package was built.
2. Fetch the first real L checkpoint (private HF `Sayood/tanitad-lorapod-ckpts`, `HF_HUB_DISABLE_XET=1`), run `python -I code_lc/check_l_ckpt.py <file> --expect-sha256 <ledger line>`; then on box 2 the same with `--expect-merged-sha256 <MANIFEST5 digest>`.
3. A one-cell canary on box 2 first (HANDOFF_S5 plan), then `stage5_chain.sh LC_<tag>`.

## MANIFEST (artifact paths under E:/Projects/TanitAD-artifacts/dagland/motion_c/t2v8f/; each exists in ONE place; nothing staged, committed or pushed)
| artifact | sha256 (first 16) |
|---|---|
| code_lc/t2v8f_lora_merge.py | 9ca6100d91f9145d |
| code_lc/t2v8f_stage5_cell.py | b674ca2f2f7dd2ea |
| code_lc/t2v8f_stage5.py | 28db8f371ac22408 |
| code_lc/make_stage5.py | e2660efb2f460aa8 |
| code_lc/check_l_ckpt.py | 3550430a3ec123f6 |
| tests_lc/lc_common.py | f7f4ba3f03952806 |
| tests_lc/test_t2v8f_lc.py | 928e067ae7e6057c (after the wording edit) |
| tests_lc/test_t2v8f_lc_identity.py | 2945bd94d1636f01 |
| tests_lc/ref/{v8f_lora_encoder,pull_encode_pooled,dinov3_fp8_encode_ship}.py | dcbcf6cb, a7dee388, 308f36c9 (verbatim tip a9da6e8f; training module blob 749fd497) |
| logs_lc/ (pytest logs, check_l_ckpt json), HANDOFF_LC.md | this folder |
Unchanged and verified: code/t2v8f_stage5.py 1fb1af10, code/t2v8f_stage5_cell.py 7fb89e5d, code/make_stage5.py 38e6d4fc.
Scratch (moved aside): `E:/Projects/TanitAD-artifacts/scratch_lc_t2v8f/` (tip copies of the training module, test and prereg, probes, `_aside_20261107/`).

## SEED-0-ONLY READ (item 5; NOT built). The pinned `code/read_t2_dominance.py` (sha 7fd65ffd...) needs BOTH seeds
Where: `SEEDS = (0, 1)` (:75); `decide()` raises `need both seeds` (:136-137) and names DOMINANT-T2 only when every per-seed block is DOMINANT (:141); `arm_cells` lays out `T2|NAME_s{0,1}` / `RR|NAME_s{0,1}` (:438);
`g2e_check` is handed `NAME_s0` and `NAME_s1` (:386); the G3 plan reads `NAME_s0` and `NAME_s1` (:735); `arrays_for_seed` loops `for s in SEEDS` (:484, :761). The stage-5 driver is two-seed too (`passes5`, `check_arm5`: `arm5.arms == [NAME_s0, NAME_s1]`).
A FIFTH pin (a new file beside the fourth, `read_t2_dominance.py` itself untouched, new `PINS_*.txt` line) would need:
1. `--arm-seeds 0,1` (default) | `0`: `SEEDS` becomes a per-run tuple threaded through the six sites above; the driver/builder get the matching `--seeds 0` (one cell arm, cells `T2|NAME_s0`, `RR|NAME_s0`, 124 cells).
2. A seed-0-only run NEVER prints DOMINANT-T2: outcome `SEED0-ONLY` with the per-seed block for s0, `n_seeds: 1`, and the sentence "seed 0 only, not a counted DOMINANT"; `OUTCOMES` gains that name; a RED arm (mutate: remove the cap -> a DOMINANT s0 block yields DOMINANT-T2) must go RED.
3. IDENTITY: with the defaults (`--arm-seeds 0,1`, arm C0C, stage 4) a fresh read of the real pulls equals the read of record field for field (the `compare_dominance_reads.py` / `logs/q_identity.json` form AGENT Q used), plus a differential on the 62-scene fixtures.
UNVERIFIED: whether other sites count cells by 2 arms (grep `SEEDS`, `_s1`, `EXPECT` in the reader; I read only the lines above).
