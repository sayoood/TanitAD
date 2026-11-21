# NOTES — compose_fix: the perturbation's generator is now independent of the swap's

Base: landed tip `e7ea20d1287599409d1758971ded62b2fc4ef375`. Nothing committed, pushed or git-staged (the lander takes `stage/`). Work tree: `F:/TanitAD-scratch/compose_fix/tree` (pristine export beside it: `.../pristine`).

## The defect (MEASURED 2026-10-28, root cause proven by the previous agent; reproduced here)

`DaggerSwap._seed_of` (`stack/tanitad/data/v8f_dagger.py:653`) and `EgoPerturb._seed_of` (`stack/tanitad/data/v8f_mini_data.py`, was :680) were the same formula `(seed * 1_000_003 + step) % 2**62`. The registered
arm A has `--dagger-seed 0 --ego-perturb-seed 0`, so the two private generators were one generator, and both take their selection uniform from row 0 of `torch.rand(k, B)`. A sample eligible for both was swapped
iff `u < p_swap` and perturbed iff `p_swap <= u < p_ego`: P(perturbed | not swapped) = (0.3 - 0.1) / 0.9 = 0.222, not 0.3 (box 2: 19161 / 76898 = 0.2492).
Reproduced on the trainer's synthetic world with the new tests on the UNMODIFIED code: 3522 / 15931 = 0.2211 and 3570 / 15953 = 0.2238 at 0.1 / 0.3, 0 / 3362 at 0.5 / 0.5, the two selection-uniform rows exactly equal, and a
200-step trainer run at the registered shape reading ego_perturb_n / ego_perturb_eligible = 1561 / 6345 = 0.2460 (box 2: 0.2492). See RESULTS.md for the logs.
Why the old tests missed it: their fixtures used UNEQUAL seeds (`dseed=3, eseed=7`).

## The fix (minimal; the non-composed path is bit-identical)

Only the PERTURBATION moves; the DAgger swap's generator is untouched (its counts in the 200-step run are identical before and after: 720 / 7380).

| file : line (fixed tree) | change |
|---|---|
| `stack/tanitad/data/v8f_mini_data.py:595` | new constant `EGO_PERTURB_COMPOSED_STREAM = 1` (+ the 4-line comment at :591-594 stating the defect) |
| `stack/tanitad/data/v8f_mini_data.py:625,628` | `EgoPerturb.__init__(.., stream=0)`; `self.stream = int(stream)` |
| `stack/tanitad/data/v8f_mini_data.py:694-699` | `_seed_of`: `base = (seed * 1_000_003 + step) % 2**62`; **stream 0 returns `base` (the original formula, bit for bit)**; a non-zero stream returns `int.from_bytes(sha256(f"ego_perturb_stream/{stream}/{base}")[:8], "little") % 2**62` |
| `stack/tanitad/data/v8f_mini_data.py:640-641` | `built_levers()` gains `ego_perturb_stream` ONLY when the stream is non-zero, so a stream-0 record (every non-composed run, the golden's `levers_ego`) is key-for-key the pre-change record |
| `stack/tanitad/data/v8f_mini_data.py:660-661` | `law_problems()`: a negative stream is illegal |
| `stack/scripts/v8f_mini_train.py:508-510` | `EgoDaggerCompose.__init__`: `if self.on: self.ego.stream = D.EGO_PERTURB_COMPOSED_STREAM` (flag off: the perturber is not touched) |
| `stack/scripts/v8f_mini_train.py:513` | `EgoDaggerCompose.built_levers()` gains `ego_perturb_stream` |
| `stack/scripts/v8f_mini_train.py:556-558` | **G-DVB read-back**: `bad_ego_dagger` refuses a stage whose BUILT perturbation stream is not `EGO_PERTURB_COMPOSED_STREAM` (flag named) or not 0 (flag absent) — a composed stage without separation can no longer pass the launch gate |
| `stack/scripts/v8f_mini_train.py:1739` | `levers["ego_perturb_stream"] = comp.ego.stream` — the run record (selfcheck.json, summary.json `levers`): **0 off / 1 composed** |
| `stack/scripts/v8f_mini_train.py:1792` | composed checkpoints carry `ego_perturb_stream` (composed arms only; a flag-off checkpoint keeps its key set) |
| `stack/scripts/v8f_mini_train.py:1772-1773` | resume guard: a composed checkpoint with no/another stream (i.e. one written by e7ea20d1) is REFUSED (`ZZRESUME-REFUSEDZZ`) — the same argv hash would otherwise let a run switch generators mid-run |
| `stack/scripts/v8f_mini_train.py:499-503` | docstring: why |

### Tests (`stack/tests/test_v8f_ego_dagger_compose.py`; 26 -> 36 tests)

* `:420-` section 7 (new): `test_equal_seeds_every_sample_eligible_for_both_...` (3 params: 0.5/0.5 seed 0, 0.1/0.3 seed 0, 0.1/0.3 seed 3; asserts its own power), `test_the_swaps_and_the_perturbations_selection_uniforms_are_not_one_stream_...`
  (a spy on `torch.rand` reads the rows actually drawn; control: an unseparated perturber reads EXACTLY equal), `test_seed_of_stream_zero_is_the_original_formula_...`, `test_the_trainer_run_at_the_box2_shape_...`
  (TR.main, equal seeds 0/0, 0.1/0.3, vmax 8, batch 64, 200 steps, ego share 0.3 +- 0.02 with a power guard N >= 4725), `test_the_run_record_names_the_perturbation_stream_...`,
  `test_gdvb_deliberate_regression_a_perturbation_that_is_not_on_the_declared_stream_is_refused` (2 params + a clean control), `test_a_composed_checkpoint_from_before_the_stream_separation_is_not_resumed_...`.
* Existing tests edited (3 lines, all consequences of the new spec, none loosened): the golden test's "levers only GAIN the flag fields" set now includes `ego_perturb_stream` (`ego_dagger_compose_golden.json` itself is
  UNCHANGED and still passes); `test_composition_overlap_...` builds its reference perturber with `stream=D.EGO_PERTURB_COMPOSED_STREAM` (the composed perturbation is the perturbation ON THE COMPOSED STAGE'S STREAM);
  `test_the_single_applying_lines_exist_exactly_once` gains 5 anchors for the new mutants.

## Consequences the caller must act on

1. **Every composed arm trained on e7ea20d1 (box 2 arm A) drew the perturbation from the swap's stream and is NOT reproducible with this fix.** Its checkpoint is refused by the resume guard; re-run from the init checkpoint.
2. The swap's selection and the DAG-OACC-style (swap-only) and ORC-style (perturbation-only) runs are bit-identical to e7ea20d1 (golden + `_seed_of` test) — those results are unaffected by this fix.
3. Land `stage/` (3 files) on a tip that still has e7ea20d1's versions of them; `diff.patch` is against e7ea20d1.

## Side observation (NOT part of the fix; MEASURED, tool `tools/global_rng_check.py`)

The trainer also seeds the GLOBAL RNG every step with the same formula (`v8f_mini_train.py:1833`, `torch.manual_seed(a.seed * 1_000_003 + step)`), and the model's teacher-forcing draws read it
(`v8f_mini.py:741,754`, float32 `torch.rand(B)`). With `--seed` equal to `--dagger-seed` those draws share a Mersenne stream with the swap's selection, but MEASURED they are not related: corr(u, v) in [-0.02, 0.06]
over 5 (seed, step) pairs, P(v < 0.5 | u < 0.1) = 0.48-0.64 (n 25-42), never equal (float32 vs float64 draws consume the stream differently). A float64 global draw WOULD be identical to the private generator's.
I read two sites only; whether any other float64 global draw exists in the loss is UNVERIFIED. No change made.

## Log provenance

`logs/new_green.log` is an INTERIM run: its 3 failures were a bug in my own new statistical test (it asserted `perturbed <= both-eligible & not swapped`, but a perturbed sample need only be perturbation-eligible); fixed, re-run in
`logs/new_green2.log`, and the final selection re-run on the final files in `logs/green_new.log` / `logs/red_on_old.log` (the pipeline). `logs/new_red_on_old.log` is the first RED run (same 10 failures, test file before one rename).
