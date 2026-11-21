# BUILD_IREP: the I-rep (inference floor) of bar E clause 3, as an instrument

Finished 2026-10-17 by a TanitAD build agent (Master Mind brief). **Nothing here has run on a pod, no GPU was touched, and no SAFEOBJ state exists** (no R0d): every verdict below is a FIXTURE verdict and every `irep_abs_delta` in a log is a fixture number.
Evidence classes: **M** measured (log path given) · **I** inherited (another document, not re-run here) · **E** estimated · **U** unverified. Python: `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe -I -B -X utf8`; tests run from Git bash (`C:/Program Files/Git/bin/bash.exe`), CPU only, one child process at a time.
Companion: `PLAN_IREP.md` (the exact command sequence, the VRAM finding, the cost, the gates).

## 1. The gap, and what was built

`PREREG_SAFEOBJ.md` §7 registers an inference floor, "the I-rep of `PREREG_RL_S1.md` §3 (eval batch 8 vs 32, shuffled order, non-deterministic kernels) on R0d-B u1000", and bar E clause 3 uses it. **Nothing produced it** (a repo grep found it only in prereg text; it was registered for RL-S1 and never built). The consumer existed: `navtest_E.py irep --a --b --out`. This package is the producer side.

| file | what |
|---|---|
| `make_variant.py` | builds `fp_forward_irep.py` from the ORIGINAL `navtestE/fp/fp_forward.py` (md5-pinned `82bf2151…`) by 5 exact-string hunks, each anchored on text that must occur exactly once; `--check` re-derives it |
| `fp_forward_irep.py` (+ `.diff`) | the I-rep variant of the navtest forward: `--eval-batch N`, `--token-order-seed S`. **82 lines added, 0 removed.** On the default path the only edit is one `if irep_active(a)` that is False |
| `irep_tool.py` | `drive` (dev-box chunk loop, resumable), `expected` (the independent hash of ONE scorer state), `seams` (`navtest_E.build_seams` for one state), `make` (the guard, then the consumer, then an independent recompute) |
| `vram_estimate.py` | the VRAM probe (§4), CPU only, meta device |
| `test_irep.py` | 29 checks + 10 RED mutations |
| `PLAN_IREP.md` | the sequence, cost, gates, failure handling |
| `logs/` | the full-form test log, the VRAM log, the identity proof, the determinism-flag grep, the audit trail of development runs |

## 2. Evidence (MEASURED, ours; Git bash, `driverl-eval` venv, no pod, no GPU, no network)

* **Full form: `ZZTEST_PASS 11  (1 unmutated suite of 29 checks, 10 RED mutations, 3 read-only files unchanged)  total 466 s`, `ZZEXIT rc=0 seconds=467`.** Log: `logs/ZZTEST_irep.log`. (11 = the unmutated suite + 10 mutations.) The header pins the original (`fp_forward.py` md5 `82bf21514f919ee4eafab72690004e34`), the variant (sha256 `218b115a…`) and the tool.
* **At the defaults the variant is byte-identical to the original** (task 1). The original `fp_forward.py` and `fp_forward_irep.py` were each run as child processes on the same 40-token fixture with `run … --repeat-first 4`; **the two output npz files have the same sha256 `1e77fba49abad5b61de0c4bf86f10478dc721da56628dcd710929198d7502d78`, 82,746 bytes, 7 arrays** (`logs/identity_proof.json`), the same stdout, the same return code, and the model was called 44 times at batch 1 in sorted order in both (the 40 tokens + the 4-token IC5 repeat). The clock is frozen by a runner so that the zip headers and `meta_json.seconds` are reproducible: the comparison is of FILES, not of arrays alone (`identity:defaults_file_byte_identical`; the arrays and `meta_json` are also compared separately). `meta_json` gains an `irep` record only when a run used a non-default option. The default-path function body is also checked verbatim against the original by AST (`variant:diff_is_a_pure_addition_and_the_default_function_body_is_verbatim`).
* **Batch 8 / 32 with a shuffled order is aligned token for token** (task 1). Against the original batch-1 read of the same tokens: the `token`/`log` arrays are identical; every row of `traj`, `logits_BASE` and `logits_R0d_B` is within float noise of the SAME token's row (max 2.4e-7, 8.9e-8, 1.4e-6 at batch 8; 4.8e-7, 8.9e-8, 1.9e-6 at batch 32) and **≥ 8.5e4 times closer than the nearest OTHER token's row** (margins ×5.3e5, ×1.2e5, ×1.1e5 at batch 8; ×2.7e5, ×1.2e5, ×8.6e4 at batch 32). The model's own call log (written by the stand-in model, independent of anything the variant reports) shows batch sizes [8,8,8,8,8] and [32,8], and the token ids in call order equal `RandomState(seed).permutation(40)` exactly. At batch 1 with a shuffled order the restored arrays equal the original's **bit for bit** (`identity:batch_1_with_a_shuffled_order…`).
* **The 10 RED mutations**, each asserted to go RED where it exists to and to stay GREEN at the controls (`logs/ZZTEST_irep.log`):

| mutation (task 3) | goes RED at | stays GREEN at (the controls) |
|---|---|---|
| `order_not_restored` (rows written at their processing position) | `rows_are_per_token_aligned…` (row i is **1.07** from its own token's batch-1 row), `batch_1_with_a_shuffled_order…` | batch sizes, processing order, all `defaults_*` |
| `batch_option_ignored_but_still_reported` (the loop runs at 1, the record still says 8) | `forward_ran_at_the_requested_batch_sizes` (the model saw forty calls of batch 1), `meta_records_the_run…` | alignment, processing order, all `defaults_*`, batch-1 shuffle |
| `shuffle_ignored` | `processing_order_is_the_seeded_permutation`, `meta_records_the_run…` (order hash), `batch_1_with_a_shuffled_order…` | alignment, batch sizes, `defaults_*` |
| `one_rig_for_the_whole_batch` (a plausible batching bug: every row gets the first token's calibration) | `rows_are_per_token_aligned…` (row i is 3.3e-4 off) | batch sizes, processing order, `defaults_*`, batch-1 shuffle |
| `default_path_changed_output_perturbed` (+1e-7 in the ORIGINAL loop) | `defaults_file_byte_identical`, `defaults_arrays_and_meta_equal` | alignment, batch sizes, processing order, batch-1 shuffle |
| `default_path_changed_default_seed_is_0` | the four `defaults_*` checks (the defaults now shuffle) | alignment, batch sizes, processing order |
| `default_path_changed_default_batch_is_2` | the four `defaults_*` checks | alignment, batch sizes, processing order |
| `irep_from_two_different_states` (the same-state gate removed) | `guard:two_different_scorer_states_are_refused` (without it the tool wrote `irep_abs_delta` 0.497 for R0d-B vs a different scorer) | `legit_pair_passes…`, `identical_token_order_seeds…` |
| `identical_seeds_accepted_both_layers_removed` | `guard:identical_token_order_seeds_are_refused` | `legit_pair_passes…`, `two_different_scorer_states…` |
| `expected_hash_not_compared` | `guard:a_hash_other_than_the_independently_computed_one…` | `legit_pair_passes…`, `two_different_scorer_states…` |

  ⚠ **The first full run caught a mutation that stayed GREEN** (`logs/dev_full_before_seed_layers_fix.log`: `identical_seeds_accepted` removed only `seeds_a != seeds_b`). Cause: two reads with the same seed are refused TWICE (the seeds, and the permutation hashes, which are equal for equal seed and chunk size). That is defence in depth, not a missing check; the mutation was changed to remove both layers (named `…_both_layers_removed`) and a source comment says so. Same lesson as navtestE's "triple guard": a single removal of a redundant layer is not detectable end to end, by design.
* **The consumer's blind spot is real and is now closed** (task 3, fourth mutation). `navtest_E.py irep` alone, handed the tables of read A (R0d-B) and a read of a DIFFERENT scorer state, returns a well-formed `{"irep_abs_delta", "unit": "PDMS x100"}` (`guard:two_different_scorer_states_are_refused` calls it directly and asserts it accepts them). That number is the training-seed contrast under the inference floor's name, and bar E clause 3 would compare against it without complaint. `irep_tool.py make` refuses it before the consumer is called.
* **The chain, end to end on the fixture:** `drive` (2 chunks, resumable, refuses mixed settings on resume) → fake harness → `make`: the consumer's value equals the value recomputed independently from the in-memory tables to 1e-9 (fixture values 2.0183 for batch 8 vs 32, 0.497 for the mutant); the file is accepted by `navtest_E.read_irep`, the reader `build` uses; `irep.check.json` lists every gate and the diagnostics (`pick_changes_a_vs_b`, `logit_maxabs_a_vs_b`).
* **The pod's entry point is covered:** `serve --no-gate --eval-batch 8 --token-order-seed 1` on a chunk directory produces `out/chunk_000.npz` equal to the `run` output of the same logs **bit for bit** (6 arrays), writes `.sha256`/`.done`, deletes the chunk's frames, and records `ic5_forward.skipped` (an IC5 repeat would re-batch the tokens: it is skipped in I-rep runs, a decision of this build).
* **Real artifacts, not only fixtures:** (i) `irep_tool.py seams` on a copy of the REAL free-point dry chunk (a real CPU forward of 2 tokens, `freept/work/dry/out/chunk_000.npz`) reproduces the live picker's `seam_BASE.npz` and `seam_R0B.npz` **bit for bit** (poses and tokens) and refuses a chunk with no `irep` record; (ii) the guard's harness parser accepts the REAL harness artifacts of the free-point dry run (`counts.json` with its `SEAM:C:\…\seam_C0A.npz`, `status.json`, the CSV) and refuses the same artifacts under a wrong arm string or against another label's seam.
* **No determinism flag anywhere** (`logs/determinism_flags_grep.log`): `deterministic|float32_matmul_precision|allow_tf32|cudnn.benchmark|use_deterministic_algorithms` = 0 hits in `fp_forward.py`, `model.py`, `ckpt_io.py`, `load_dinov3.py` and the variant, with same-breath controls that hit (`no_grad` 1, `eval_batch` 3, `scaled_dot_product_attention` 1). So "non-deterministic kernels" is satisfied by leaving the library defaults, which is what the original does.
* **Read-only trees untouched** (**M**): `sha256sum` of all 36 files under `navtestE/` and `outcome/` before and after are identical; no `__pycache__` was created there; the test also re-hashes `fp_forward.py`, `navtest_E.py` and the reader at its end. The free-point package was read, never written.

## 3. Which fixture is real and which is synthetic (the "built, tested, unreachable from its caller" rule)

* **REAL:** the original `fp_forward.py`; the variant as a child process with its real argparse, ZipCache, cv2 JPEG decode, Taps hooks, IC2; `navtest_E.py irep` and `navtest_E.read_irep`; `navtest_E.build_seams` on the free-point dry chunk; the free-point dry run's seam/counts/status/CSV artifacts; the real A40 chunk metas (cost); the real `REFe` forward (VRAM probe, meta device).
* **SYNTHETIC (labelled):** the model (a ~10 k-parameter stand-in with the worker's interface; its scorer head can add a function of the BATCH SIZE, standing in for a kernel that is not batch-invariant, so that the pipeline has a signal to carry: it says nothing about the real model's I-rep); the 40 JPEG tokens; the harness (`fake_harness`: sel_score picks, PDMS from a fixed table: the real `score_e.py` was NOT run); the seam/points/status/counts files of the fixture reads (mimics of the real formats, whose real counterparts the guard's parser was checked against).
* ⚠ **NOT EXERCISED: the REAL model at batch > 1.** The brief asked for tiny synthetic CPU states, and a real ViT-L CPU forward is ≈ 60 s/token (**M** `freept/work/smoke1.log`). `PLAN_IREP.md` gate **I-G5** gives the exact 2-token real-model CPU smoke (batch 2, order swapped, compared to the free-point's own batch-1 forward); it is **U** until someone runs it, and it should run before the pod does. The real `forward` takes `calib [B, 4, 16]`, folds the camera axis into the batch and gathers per-sample rigs, so batch > 1 is the code's design; that is reading, not measurement.
* ⚠ **NOT EXERCISED: the harness (`score_e.py`) on I-rep seams, the pod, the shipper's upload of the variant.** Inherited from the free-point run (**I**).

## 4. The VRAM finding (task 2)

**A ViT-L scorer forward at batch 32 does not fit the dev box's 8 GB, and the margin is not close.** **M** (`logs/vram_estimate.log`; the REAL `REFe` forward executed on the meta device with a dispatch-mode live-byte tracker that first reads its two analytic controls exactly; fp32, `no_grad`, 4 cameras of 512×960 = 1,925 tokens per image):

| eval batch | images | model peak (weights 1.29 GB + activations) | ~nvidia-smi (×1.15 + 0.4 GB, **E**) |
|---|---|---|---|
| 1 | 4 | 1.71 GB | 2.4 GB |
| **8** | 32 | 4.69 GB | 5.8 GB |
| **32** | 128 | 14.91 GB | **17.5 GB** |

The peak is inside the trunk's MLP, and at batch 32 it is accounted for exactly: the fc1 output 4.04 GB + the GELU output 4.04 GB, three residual-stream-sized tensors (the block input the caller still holds, the mid-block `x`, the LayerNorm output the `Sequential` call still holds) 3 × 1.01 GB, the patch-embed output 1.01 GB (a local the backbone keeps for the whole loop), and two 0.755 GB image tensors (the device copy and its normalised copy) = **13.62 GB, the sum of the twelve biggest live tensors the tracker lists** (`logs/vram_estimate.log`, last lines). A hand-built bound (weights + 2 stream + 2 MLP hidden + image = 12.1 GB) misses the 2.8 GB of the other live tensors; the tracker follows the real lifetimes. With the math SDPA the peak is 18.6 GB at batch 8 and 70.6 GB at batch 32 (the score matrix); which kernel CUDA picks was derived from the package's own training-memory measurement, not by touching the GPU (`PLAN_IREP.md` §1). The card is shared: 4,229 MiB used early in the session, 1,582 MiB at 15:40 (8,188 MiB total). **Batch 32 never fits; batch 8 fits only when the other tenant leaves ≥ 6 GB; so the registered "dev box" cannot hold the registered batches. Fallback proposed: the A40 slot between R0d and R1d, both reads there** (≈ 17.5 GB of 48 GB). The guard refuses a pair of reads from two machines (`one_checkpoint_and_one_machine`), so the dev box cannot be used for one read only.

## 5. Cost (task 2; from the free-point's measured per-token numbers)

A40 forward **0.60 s/token (M**, `freept/work/out/chunk_000.npz`: 1,521 tokens in 918.3 s; `chunk_001.npz`: 1,440 in 866.3 s) ⇒ ≈ 2.0 h per read at the free-point rate; the upload of the 10.06 GB of frames is the binding constraint at **1.07 MB/s (M**, `freept/work/ship.log`, four chunks, 4.96 GB in 4,642 s) ⇒ **≈ 2.6 h per read, ≈ 5.2 h for the two**, plus 0.371 h of harness per read (**I**) on the dev box, of which the first overlaps the second upload: **≈ 5.7 h wall, 0 A40 training-hours.** Details and the unbuilt one-upload variant: `PLAN_IREP.md` §4.

## 6. Decisions made in the build (each is reversible; none was in the brief)

1. **Shuffling is per chunk** (the worker sees the frames chunk by chunk). A "shuffled order" across all 12,146 tokens would need all frames at once. Stated in `PLAN_IREP.md` open point 8.
2. **`np.random.RandomState(S).permutation(n)`**, not `default_rng`: RandomState is NumPy's frozen legacy stream, so the pod's NumPy and the dev box's give the same order (the recorded `order_sha256` lets anyone check).
3. **IC5 is skipped in I-rep runs** (it re-runs the first n tokens, i.e. re-batches them, measuring the very thing this run varies); IC2 (the replicated scorer path equals the model's own score) stays, per token.
4. **Seeds 1 and 2** for reads A and B: the prereg names none.
5. **The guard compares the expected hash for read A only**; B is tied to A by the same-state gate. Each gate then has one unique job, which is what lets the mutations tell them apart.
6. **`make` runs the consumer unchanged, in its own process, and recomputes the value itself**; a disagreement beyond 1e-9 is a refusal.

## 7. Open points (each blocks or changes the read; none is decided here)

1. **Dev box vs A40** (§4): *Master Mind.*
2. **Gate I-G5 (real model, batch 2, CPU) has not been run.** The one unexercised path. Cheap (≈ 2–4 min CPU, no GPU).
3. **Two uploads** (5.2 h of mostly-idle A40): the one-upload worker is not built.
4. **The navtest-level I-rep is a different object from the RL-S1 §3 one** (that was defined on RL-val lookups). Bar E's file carries PDMS ×100 on navtest, so this one is the coherent producer; an RL-val I-rep is not built.
5. **One draw** of |Δ| (two reads); a third read is not registered.
6. **The variant upload to the pod is manual** (`sobj_ship.py` ships the original `fp_forward.py`); untested against a pod.
7. **`fp_score.py` hard-codes `SP`** (PLAN_E open point 7).

## 8. Manifest

See `MANIFEST.md` (every file with its sha256; written after the last edit).
