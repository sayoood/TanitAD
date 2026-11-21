# refcv8 eval determinism: root cause and fix (pre-gate blocker B4), 2026-10-12

> Base: `d014ff2fd695c306abf769af0572a17e8d713991` (the pre-gate's gated commit). Everything below is **MEASURED (ours)**
> on the **dev box, CPU** (Windows, torch from `C:/Users/Admin/venvs/tanitad`), on a local `git archive d014ff2` export
> (`tanitad.__file__` asserted per run). Thor was not used. Tier: **none of this is a driving metric.** It is an
> instrument / model-semantics fix, measured on the `_refcv8_rig` model (real `RefCV3Model`, real decoder + refcv8
> seams, small trunk, random init) built with the **provisional A2 recipe as the trainer's own `_pin_refcv8` pins it**
> from `argv_PROVISIONAL.json` (sha256 `d1fa48b2…`): `n_alloc 32, alloc_emit, emit_start 1000, w_subscore 0.5,
> lat_prior_dropout 0.3, v9_cons, speed_input label`, r8 seed 20261004.

## Headline

| | |
|---|---|
| **Mechanism** | Two **EVAL-time** draws took their **values** from the decoder's persistent `R8Generator` (a dedicated `torch.Generator`, seeded once, advancing on every call), which no `torch.manual_seed` and no `torch.random.fork_rng` restores: **(1)** the allocated candidates' source-anchor choice, `refcv8_conditioning.py:399` (`select_source_anchors`, called at `refc.py:2509`); **(2)** the allocated candidates' sampler noise `eps_x`, `refc.py:3024` (in `_sample`). Neither is gated on `self.training` (the other r8 draws are: `refc.py:2350-2353`, `:2568-2570`). So the same checkpoint, batch and inference seed gave a different answer on every call. |
| **Bisect** | each site alone leaves outputs differing (select fixed → 12 differ, `eps_x` fixed → 14 differ); **both fixed → 0**; re-creating the generator before each call → 0 (`raw/out/r8_bisect.json`). |
| **Reproduces the gate** | the gate's **own** `launch_gate._forward_out` (`launch_gate.py:4763-4783`), twice, at d014ff2: **14 of 150 outputs differ** with emission off (the launch state at step 0) = exactly the pre-gate's 18-key list (`thor_dir/raw/digest_trace_d6.json`, `d7_omp1`) less 4 of the 8 `layer_u0_hat[i]` (the rig decoder has 4 layers); 28 of 150 with emission on. **refcv7 control: 0 of 126** (the defect is r8-specific, as the gate's history implies). `raw/out/gate_probe_matrix.jsonl` |
| **Fix** | at eval the two draws take their values from a **fresh generator seeded from the device's GLOBAL RNG STATE — read, never consumed —** + the r8 seed + a per-site salt (`R8Generator.eval_generator / rand_eval / randn_eval`, `refcv8_conditioning.py:173-213` new). The persistent training stream is **still advanced by the identical draw** (value discarded). Training path textually unchanged. |
| **RED → GREEN** | new `stack/tests/test_refcv8_eval_determinism.py`: **d014ff2: 11 failed / 2 passed** (the reproducibility tests fail for the right reason: `14` / `28` / gate-control `14` outputs differ; the API tests fail on the missing API; the 2 that pass are the refcv7 control and "a different seed differs", both true before and after). **Fix: 13 passed + 1 CUDA test** (skips without a GPU; see the GPU note). `raw/out/pytest_OLD_d014ff2_det.log`, `raw/out/pytest_FIX_det.log` |
| **Train-mode equivalence** | real d014ff2 module vs the fix, **cross-tree, same thread settings**: TRAIN fwd+bwd → 3 EVAL forwards → TRAIN fwd+bwd, emission on and off: **0 of 87 outputs, 0 of 344 grads differ, loss equal, `R8Generator` state equal** after every phase, global RNG end state equal (`raw/out/compare_cases.json`). The second train step runs AFTER eval forwards, so the in-run eval cadence moves the training trajectory exactly as it did before. |
| **Second defect fixed (the next G-EVAL failure)** | trainer-vs-loader `out.r8_emit_keep` (pre-gate d6/d7): a fresh refcv8 decoder was born with `r8_emit_live = True` (`refc.py:2476`) regardless of `emit_start`, while the loader schedules the gate's `{"step": 0}` checkpoint (`launch_gate.py:4808`) at step 0 → **off** (`refcv7_loader.py:517-519`). On the rig the never-scheduled build vs the loader-scheduled copy differ on **15 of 150** outputs incl. `r8_emit_keep` and the emitted `traj` (`raw/out/emit_probe_old.log`); on Thor's warm-started batch only the mask differed. Fix: the switch is **born at the schedule's step-0 value** (`emit_start <= 0`). `train()` re-applies the schedule before every training forward (`refc_v3_train.py:10572`; no model forward between the build at `:8894` and the loop at `:10535`), so training is unchanged. |

## What this does to things already MEASURED

* **Warm-start / I-2 identity (refcv7 vs refcv8 at step 0, emission off): NOT affected.** The base candidates draw
  their noise from the global stream (`refc.py:3023`) and never see the extras' draws: base columns of every
  per-candidate output are identical d014ff2 vs fix, and with emission off the emitted plan (`traj`) does not move
  (`compare_cases.json` `eval_old_vs_new_base_columns_or_noncandidate_differ`; RED-arm test asserts `traj` stable).
* **refcv8 in-run eval rows (ladder, de-risk smoke).** The in-run eval runs after the step's schedule
  (`refc_v3_train.py:10851` after `:10572`). By code reading (magnitude on real arms **UNVERIFIED**, no ladder data read):
  * **emission OFF** (step < `emit_start`): the emitted plan is independent of the extras' draws, so plan-derived rows
    (ADE etc.) were not touched by this defect. Rows computed over **all** candidates were: on the rig
    `goal_score_absmean` (a mean over all candidates) moves.
  * **emission ON** (step ≥ `emit_start`, and **tiny ladder arms with `emit_start` 0**, `refcv8_conditioning.py:133-134`):
    the emitted plan depends on the extras' noise, which came from the persistent stream.
  * **Is noise inflated? No, and the fix does not remove any.** Either way the row is one draw of the same sampler; the
    sampler is stochastic at eval by design (`refc.py:2995-3005`). The fix makes that draw a **function of the
    inference seed** (a different seed still differs: 30 / 32 outputs, `compare_cases.json`). What the defect DID break:
    (a) re-evaluating a checkpoint could not reproduce an in-run row on an emitting arm (the loader's fresh generator vs
    the run's advanced one), so in-run vs post-hoc differences on emitting arms include sampler noise; (b) an
    **inference-seed replicate** (CLAUDE.md's third variance) could not hold the extras' noise fixed; (c) G-EVAL's
    trainer-vs-loader comparison was blind.
* **Training trajectories: unchanged** (bit-identical above). ⚠️ A pre-existing property this fix deliberately keeps:
  in-run evals ADVANCE the training `R8Generator` (`refcv8_train.py:553`'s tf coin is drawn at eval too, plus the two
  sites), so a run's training draws depend on its `--eval-every` / `--eval-batches`. Changing that would change
  training; it is a separate owner decision.

## Scope, gaps, and what was NOT shown

* **Rig, not the full A2 model.** The full A2 build needs Thor data files at build time (`_pin_trainer_cfg` opens the
  extrinsics JSON, `refc_v3_train.py:2332`). The rig carries the real decoder/refcv8 code path with the A2 r8 recipe
  pinned by the trainer; trunk, decoder depth (4 vs 8 layers), anchors (20 vs 117) and perception (`FakeBranch`) differ.
  The re-run of G-EVAL on Thor at the new commit is the confirmation; until then "G-EVAL's control passes on A2" is
  **UNVERIFIED** (it is MEASURED on the rig, with the gate's own function).
* **CUDA branch** of `eval_generator` (reads `torch.cuda.get_rng_state(dev)`, the generator `eps_b` draws from):
  exercised once by `test_eval_generator_on_cuda_reads_the_device_state_and_never_consumes_it` — see the GPU note.
* **G-EVAL still needs B2 (kit remap) and B3 (this argv's own smoke config)** from the pre-gate; this package closes B4
  and the `r8_emit_keep` follow-on only.
* ⚠️ **CPU bits depend on the OMP thread count.** A first cross-tree comparison (d014ff2 at default threads vs the fix
  at `OMP_NUM_THREADS=4`) read 41/87 outputs and 243/344 grads different; the same comparison at equal threads read 0/0.
  Any CPU bit-identity claim must hold the thread count fixed (the gate's control compares within one process, so it is
  not exposed).
* ⚠️ **GPU note (disclosure).** The coordinator asked for CPU only via `CUDA_VISIBLE_DEVICES=`. MEASURED: on this
  Windows box an **empty** `CUDA_VISIBLE_DEVICES` does **not** hide the GPU (`torch.cuda.is_available()` → True); only
  `-1` does. Every model and forward here is CPU by construction, and the refcv8 suite touches only TF32 backend flags,
  but the new CUDA test **ran once on the RTX 4060** (64 floats, ~1 s) before I caught it; later runs used `-1`.

## Test evidence

| run | tree | result | log |
|---|---|---|---|
| `test_refcv8_eval_determinism.py` | d014ff2 | **11 failed, 2 passed** (RED, reasons above) | `raw/out/pytest_OLD_d014ff2_det.log` |
| `test_refcv8_eval_determinism.py` | fix | **14 passed** (incl. the CUDA test, see GPU note) | `raw/out/pytest_FIX_det.log` |
| all `tests/test_refcv8*.py` + `test_launch_gate_refcv8.py` | fix without the emit-default change | 570 passed, 2 skipped | `raw/out/suite_refcv8_fix_without_emit_default.log` |
| same files | fix (final) | **572 passed, 2 skipped** (= the 559 below + 13 new; the CUDA test was added after this run collected; 0 regressions) | `raw/out/suite_refcv8_fix.log` |
| same files | d014ff2 (baseline) | **559 passed, 2 skipped** (both skips: `REFCV8_FULL_FIT` 5-min acceptance, `test_refcv8_det_nms_acceptance_full.py:101`) | `raw/out/suite_refcv8_d014ff2.log` |

Repro: `raw/code/` (`repro_lib.py` builds the rig with the trainer-pinned A2 r8 config; `r8_bisect.py`, `gate_probe.py`,
`dump_case.py` + `compare_cases.py`, `emit_probe.py`; env `CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=<tree>/stack;<tree>/taniteval
R8DET_TREE=<tree>`).

---

## Master Mind verification (2026-10-12, independent of the agent's scripts)

Re-run by the Master Mind before landing. `raw/mm_verify/` holds its own scripts, not the agent's repro code.
* **RED/GREEN, re-run:** `test_refcv8_eval_determinism.py` on d014ff2 plus the 9cc6ace gate fix gives **11 failed, 2
  passed, 1 skipped**. On the fix it gives **13 passed, 1 skipped**. The skip is the CUDA-only test; this run used
  `CUDA_VISIBLE_DEVICES=-1`.
* **Cross-tree training identity, own script** (`xtree_dump.py`, one process per tree, `OMP_NUM_THREADS=4`):
  - Sequence: TRAIN, then EVAL at seeds 1234, 777 and 1234, then TRAIN twice.
  - Three modes: emission forced off, forced on, and **`sched`**. In `sched` the build is fresh with `emit_start` 1,
    and the trainer's own `apply_emit_schedule` runs before each step. This covers the emit-default change: the old
    build is born `True`, the fix is born `False`.
  - Result: **every TRAIN phase in every mode reads 0/87 outputs and 0/344 grads different, with loss, r8 generator
    state and global RNG state equal.**
* **Eval reproducibility under seed 1234 twice:** old **14** (off/sched) and **28** (on) outputs differ; fix **0** in
  all modes. A different seed (1234 vs 777) still changes 30–32 outputs in both trees.
* **Control:** train0 vs train1 grads differ on 110 of 344 tensors, so the comparator is not blind.
* **Consequence for the running ladder** (code d014ff2-era): training is bit-identical under the fix, so its readings
  transfer to the launch commit. Only in-run eval rows that draw the extra candidates are not reproducible by
  re-evaluation, as the agent's RESULT says.
