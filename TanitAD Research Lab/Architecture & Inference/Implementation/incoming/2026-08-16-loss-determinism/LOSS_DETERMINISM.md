# `v6_loss_step` is now reproducible from its own generator — and the culprit was a single un-seeded `torch.randn`

**Date** 2026-08-16 · **Branch** `agent/arch-inf-20260803` · **Base HEAD** `655ce40`
**Tier** N/A — this is a *loss-mechanics* result, not a capability claim. No model number is quoted or moved.
**Evidence class** MEASURED (ours) unless stamped otherwise. Box: dev box, CPU-only, torch `2.11.0+cu128`.
**Instrument** `stack/tests/test_loss_determinism.py` (40 tests, all green).

---

## 0. The one-paragraph version

Two identical `v6_loss_step` calls with the **same `generator`** returned different numbers.
I reproduced it, localised it **per loss term**, and the entire discrepancy is in **`o6`** —
`SigReg` drew its M slice directions with a bare `torch.randn`, i.e. from the **global** RNG,
which the passed `generator` does not cover. It is the **only** such draw in the whole loss
path; that claim is now proved dynamically, not by grep. A `generator=` is threaded through
`SigReg` → `position_relaxed` → `o6_sigreg_loss` → `v6_loss_step` as an **opt-in**
`sigreg_generator`, so the default — the code v6F S-W is training from on Thor — is
**bit-identical**, proved against a content-anchored reference. With the switch on, the total,
every per-term tensor **and every parameter gradient** agree bit-exactly across all four stages.

---

## 1. Reproduced — and quantified per term (BEFORE)

MEASURED, `train()` mode, minimal dev build, `generator=torch.Generator().manual_seed(11)`
passed identically to both calls. Raw: `scratchpad/repro_before.json`, script `repro_nondet.py`.

| stage | total A | total B | bit-equal | rel. move | which terms moved |
|---|---|---|---|---|---|
| **S-W** | 3.379698 | 3.384279 | ❌ | 0.1355 % | **`o6` only** |
| **S-T** | 1.691426 | 1.691426 | ✅ | 0 | — |
| **S-S** | 0.832642 | 0.832642 | ✅ | 0 | — |
| **S-J** | 5.903766 | 5.908347 | ❌ | 0.0776 % | **`o6` only** |

Per term, S-W and S-J (identical values — the same `o6` call):

| term | A | B | bit-equal | rel. move |
|---|---|---|---|---|
| `o1` | 0.450948 | 0.450948 | ✅ | 0 % |
| `o2` | 1.111233 | 1.111233 | ✅ | 0 % |
| `o3` | 0.737363 | 0.737363 | ✅ | 0 % |
| `o5` | 1.022733 | 1.022733 | ✅ | 0 % |
| **`o6`** | **0.057419** | **0.062000** | **❌** | **7.98 %** |
| `s1`, `seam`, `t1` (S-J) | — | — | ✅ | 0 % |

⚠️ These are the **weighted** `terms[...]` tensors that get summed into the total — `o6` is
already `× o6_sigreg = 0.1`, so the raw Epps-Pulley statistic is ~0.574. The unweighted value
is what `log["o6_sigreg"]` reports; do not compare the two columns across documents.

⚠️ **These are not the PI's exact numbers and should not be quoted as if they were.** The PI
measured S-W **3.9301 vs 3.9227** with `o6` **0.046874 vs 0.039470 (18.7 %)**; my synthetic
build and batch differ, so the absolute values differ. What reproduces exactly is the
**mechanism, the affected term, and the order of magnitude** — and the seed-to-seed spread
below (**19.1 %**) brackets the PI's 18.7 %, i.e. their two calls happened to land further
apart on the same distribution than mine did.

**S-T and S-S are clean for a structural reason, not by luck:** `V6LossWeights().for_stage()`
sets `o6_sigreg = 0` there, so the defect cannot reach them. That is asserted, not assumed —
`test_o6_is_live_in_exactly_the_stages_this_file_claims`.

---

## 2. The enumeration — proved, not listed (C74)

⛔ C74's failure mode is *a list whose every entry is verified but whose completeness never
was*. So the sweep for "every source of randomness in the loss path" was done **three
independent ways**, and only the third is load-bearing.

**(a) Static, over the modules `V6Stack` actually imports.** `encoder.py`, `predictor.py`,
`readout.py`, `tactical.py`, `metric_dynamics.py`, `sigreg.py`, `v6.py`,
`scripts/train_v6_staged.py`, `scripts/train_stage_a.py` — every `torch.randn/rand/randint/
randperm/normal/bernoulli/multinomial/*_like`, every in-place `.normal_/.uniform_/.random_/
.bernoulli_`, every `nn.Dropout`/`F.dropout`/`dropout=`.

**(b) Call-graph disambiguation of the two near-misses.**

- `metric_dynamics.py:576` — `torch.rand_like(v)` under `shortcut_dropout` (default **0.1**,
  active in `train()`) is a genuine un-seeded draw, but it lives in **`UnicycleStepReadout`**
  (class opens line 472). `V6Stack` builds `StepDisplacementReadout` (`v6.py:1866`, class
  opens line 195). **Not in this path.** ⚠️ This is the one that would have been a false
  positive in a grep-only sweep.
- `flagship_v15.py` `ego_dropout` / `goal_dropout` (both 0.5, un-seeded) — `v6.py` imports
  **only `SPEED_SCALE`** from that module, inside `_lift3`. Not in this path.
- `v6.py:402-403` (`sample_cell_block_mask`, O3's mask) and `train_stage_a.py:164-165`
  (`sample_random_deltas`, O1's counterfactual) already take `generator=` and `v6_loss_step`
  already passes it. ✅ Correct before this turn.
- `v6.py:525/533` — the episode sampler, uses `self.gen`; a dataset, not the loss.

**(c) ⭐ Dynamic, and this is the actual proof.** `torch`'s whole RNG surface (11 module
functions, 8 in-place methods, 6 dropout functionals) is monkeypatched to record any call made
**without** `generator=`, and `v6_loss_step` is run in `train()` mode **with every lever on** —
`selector="goal"`, `anchor_goal="snap_lat"`, `goal_factored`, `goal_multilabel`,
`goal_cat_args`, `plan_wta_eps=0.1`, `sigreg_free_dims=4`, `o3_band_rows=1`,
`o3_mode="context"`, `o5_mode="linear-decay"`, `lambda_plan/w_select/w_anchor` all 1.0 →
**11 live terms in S-J**. This walks the graph rather than sampling it, and it sees draws
inside `torch.nn` that no grep of our repo could.

**Result, all four stages, all levers:** exactly **one** entry —

```
torch.randn @ stack/tanitad/models/sigreg.py:70 (_forward_fp32)
              | dirs = torch.randn(d, self.n_slices, device=z.device, dtype=z.dtype)
```

**(d) And an enumeration-free cross-check.** Fork `torch.random.get_rng_state()` around the
call and compare bytes. Whatever the graph contains, if the state did not move, nothing read
the global stream. BEFORE: moved in S-W and S-J, unmoved in S-T and S-S — exactly matching (c).

The dynamic enumerator is **kept as a test** (`test_the_ENUMERATION_of_unseeded_draws_is_EMPTY`),
so a sixth, seventh or eighth loss term that adds an un-seeded draw fails here with its
`file:line`, instead of silently re-creating this defect.

---

## 3. The fix

`stack/tanitad/models/sigreg.py` — new module-level `sample_directions(d, n_slices, like,
generator=None)`, and `generator` threaded as a **keyword-only** argument through
`SigReg.forward` → `SigReg._forward_fp32` and through `position_relaxed`.

```python
if generator is None:
    return torch.randn(d, n_slices, device=like.device, dtype=like.dtype)   # ⛔ the incumbent, literally
if generator.device.type == like.device.type:
    return torch.randn(d, n_slices, device=like.device, dtype=like.dtype, generator=generator)
return torch.randn(d, n_slices, device=generator.device, dtype=like.dtype,
                   generator=generator).to(like.device)
```

The third branch exists because `torch.randn(device='cuda', generator=<cpu gen>)` **raises** —
and a CPU `torch.Generator` feeding CUDA activations is the normal case on a pod. Drawing on
the generator's device and moving is the pattern `sample_cell_block_mask` already uses, and it
has the side benefit that a run's O6 stream does not depend on which device it lands on.

⚠️ **Two minimal edits in `stack/scripts/train_v6_staged.py` — flagged loudly, another agent
owns that file's spectrum/gate path.** They are additive and touch nothing else:

1. `o6_sigreg_loss(sigreg, z, free_dims=0, *, generator=None)` — forwards to `position_relaxed`.
2. `v6_loss_step(..., sigreg_generator: torch.Generator | None = None, ...)` — one new
   keyword-only parameter, passed at the single `o6_sigreg_loss` call site.

No other line of that file changed. `git diff` is 4 hunks, all inside `o6_sigreg_loss`, the
`v6_loss_step` signature, its docstring, and the O6 call.

### ⛔ Why it is a SEPARATE parameter and not just `generator`

The trainer **already passes `generator=gen`** (`train_v6_staged.py:1955`, seeded `a.seed + 1`),
while the live global stream is seeded `a.seed`. Threading `generator` into `SigReg`
unconditionally would therefore change the direction sequence of the **running v6F S-W job** —
`o6` would move at every step and the run would stop reproducing the loss it was trained with.
That is precisely the default-must-not-change constraint. So reproducibility is **opt-in per
caller**, and `sigreg_generator=None` is the incumbent, byte-for-byte.

### ⚠️ Use a SEPARATE generator object, not the same one

`generator` also feeds `sample_random_deltas` (O1) and `sample_cell_block_mask` (O3). Sharing
one stream **re-couples the terms**: switching O3 off changes how many draws precede O6, so
`o6` moves for a reason that has nothing to do with the ablation — the exact confound this
parameter exists to remove. Documented at the parameter.

**Unchanged by construction:** every other `SigReg` / `position_relaxed` caller —
`tanitad/models/dynamics_encoder.py:384`, `tanitad/train/flagship_losses.py:371-374`,
`tanitad/train/train_worldmodel.py:366` — omits the new keyword and keeps the global draw.

### ⚠️ Remaining exposure, named rather than left implicit

The three callers above are **still non-reproducible in-process**, and they now *can* be fixed
with a one-word change each (the plumbing is in place). I deliberately did not touch them: they
belong to the flagship / REF-A / world-model trainers, not to `v6_loss_step`, and silently
widening the blast radius into files another stream may be editing is how a reconciliation
becomes a merge conflict. **If anyone plans an in-process A/B on those losses, pass
`generator=` at those call sites first** — and note they carry *additional* un-seeded draws that
this turn did not enumerate, because the enumeration above was run on the v6 graph:
`flagship_losses.py:216/242/267` (ego-, nav- and future-action dropout) and
`flagship_v15.py:442/446/459/474` (ego- and goal-dropout, both p=0.5) are un-generatored by
inspection. ⚠️ That is a **static** read, not the dynamic proof used for v6 — treat it as a
starting list, not a complete one, and re-run the enumerator against that graph.

---

## 4. Proof (AFTER)

Raw: `scratchpad/after_fix.json`, script `after_fix.py`. All in `train()` mode.

**(1) Bit-exact with the same `sigreg_generator` seed — total AND every per-term tensor,
all four stages.** Minimal build **and** all-levers build (11 terms in S-J). Every
`rel_pct = 0.0000 %`, every `torch.equal` **True**.

| build | S-W | S-T | S-S | S-J |
|---|---|---|---|---|
| minimal | 3.376587 ✅ | 1.691426 ✅ | 0.832642 ✅ | 5.900655 ✅ |
| all levers | 3.313521 ✅ | 11.530195 ✅ | 0.832642 ✅ | 15.676358 ✅ |

**(2) ⭐ The gradient is reproducible too, not just the scalar.** An ablation harness compares
**updates**, not printed losses; a reproducible scalar over a non-reproducible gradient would
still make every A/B noise. Bit-exact per parameter tensor, S-W and S-J
(`test_the_GRADIENT_is_reproducible_too_not_just_the_scalar`).

**(3) The negative controls — four of them, because without these the suite passes vacuously.**

| control | expectation | MEASURED |
|---|---|---|
| different `sigreg_generator` seeds | `o6` **must** still move — the directions are resampled per call *specifically* to prevent adversarial anisotropic collapse; a fix that froze them would pass every test above and break SIGReg | `o6` 0.052838 → 0.062929, **19.10 %**; totals move 0.2989 % (S-W) / 0.1710 % (S-J); every other term bit-equal ✅ |
| the **default** path (`sigreg_generator=None`) | **must still be non-reproducible** — if this ever passes, the incumbent moved | `o6` 0.059172 vs 0.057286, 3.19 % ✅ |
| re-passing one **live** generator object | must advance it (a harness that hoisted the generator out of its loop must not silently get frozen directions) | `o6` differs ✅ |
| the RNG-state watcher / the enumerator | must be **capable of firing** | both fire on the default path, and the enumerator names `models/sigreg.py` ✅ |

**(4) ⛔ The default did not move — against a CONTENT-anchored reference, never `HEAD` (C75).**
`_side_by_side` walks each file's own history for the newest revision lacking the change
marker. Resolved this turn:

| file | marker | reference commit |
|---|---|---|
| `stack/tanitad/models/sigreg.py` | `sample_directions` | **`7507176`** |
| `stack/scripts/train_v6_staged.py` | `sigreg_generator` | **`a4a449b`** |

- `test_SIGREG_default_is_bit_identical_to_the_PRE_CHANGE_module` — old vs new `SigReg`
  (parameter-free, so directly comparable) under an identical global seed, at
  `(slices, beta, free_dims)` = (8, 1.0, 0), (13, 0.7, 0), (8, 1.0, 5) — covering **both**
  `position_relaxed` branches. `torch.equal` on the statistic, **and** `torch.equal` on the
  global RNG state afterwards: the stream is consumed to the same value *and in the same
  amount*, because a different draw **count** desynchronises everything downstream.
- `test_v6_loss_step_default_is_bit_identical_to_the_PRE_CHANGE_trainer` — the pre-change
  trainer vs the current one with the parameter omitted, all four stages, total + every term +
  the log key set. Both run against the **current** `tanitad.models.v6`, so the model is held
  fixed and only the loss code varies — any difference could only be mine.
- Neither guard skipped (a skip would be honest but would prove nothing); both resolved real
  references, and `test_NEGATIVE_CONTROL_the_no_change_guard_CAN_fail` shows the comparison
  bites rather than the reference resolution quietly returning the same object.

### ⭐ A defect the negative control caught, in the instrument itself

The enumerator's `_site()` reported **itself** as the culprit — `_site` is the deepest in-repo
frame at the moment it walks the stack, so the one real hit came back labelled
`test_loss_determinism.py:163` instead of `sigreg.py:70`. The positive test still *passed*;
only the diagnostic was wrong, which is worse, because it would have sent the next reader to
the wrong file. This is **CLAUDE.md's monitor-whose-filter-matches-its-own-echo trap in a new
costume** — keep the observer disjoint from the observed. Fixed by excluding the instrument's
own frames, and pinned by
`test_NEGATIVE_CONTROL_the_enumerator_can_fire_AND_NAMES_THE_RIGHT_FILE`, which now asserts
both that it fires **and** that it does not name itself.

---

## 5. What this makes possible — and the cheapest experiment that uses it

**Before:** any in-process A/B on the v6 loss was confounded by an RNG resample worth
**7.98 % of the `o6` term** between two same-`generator` calls, and **19.10 %** across two
`sigreg` seeds. (⚠️ The per-term values quoted throughout are the **weighted** `terms["o6"]`,
i.e. already multiplied by `o6_sigreg = 0.1`; the raw statistic is in `log["o6_sigreg"]`.)
Carried to the total, the MEASURED moves on this build were **0.1355 %** (S-W) and
**0.0776 %** (S-J) same-seed, and **0.2989 %** / **0.1710 %** across seeds. Small in absolute
terms, and *not obviously smaller than the effect a single-term ablation is trying to resolve*.
Worse, it was **unpaired**: the two arms drew different directions, so the noise did not
cancel, and it propagated into the **gradients**, which is what an ablation actually consumes.

**Now:** with a fixed `sigreg_generator`, two arms differing only in one lever share the exact
same slice directions, the same O1 counterfactual deltas and the same O3 mask. Any difference
in the total, in a per-term tensor, or in a parameter gradient is **attributable to the lever
and to nothing else**. This is the precondition for the PI's question *"do our masking /
regularisation measures actually help?"* — it does not answer it, it makes it answerable.

**⚠️ Two things this does NOT do,** stated so nobody over-reads it:
1. It does not change any trained model, any banked number, or the v6F run. The default is
   byte-identical and the live job is untouched.
2. Determinism ≠ significance. A same-seed A/B is now *attributable*, but a single seed is
   still one draw; a claim about whether a measure **helps** still needs multiple seeds and a
   paired estimator (episode-cluster bootstrap), and still carries the four metric families.

**Cheapest discriminating experiment — 0 GPU, seconds, and it is the right first one.**
A **paired single-batch gradient probe** on S-W: fixed batch, fixed `generator` and
`sigreg_generator`, two arms `o3_masked = 1.0` vs `0.0`, and read (a) the cosine between the
two parameter-gradient vectors and (b) the per-module gradient-norm delta on
`stack.masked_cells` and on the shared trunk. That answers *"does the masking term move the
parameters it claims to move, and does it move the trunk in a different direction than the
rest of the objective?"* — which is the precondition for "does it help", and which was
**unmeasurable before this turn** because the O6 resample perturbed the gradient of every
shared parameter between the two arms. The same probe with `o6_sigreg = 0.1` vs `0.0` is the
regularisation half. ⛔ Per the brief I did **not** run these.

The capability version — a short **paired** S-W training A/B at matched steps, ablating O3 and
then O6, read on the four metric families — is the follow-on and needs a GPU decision from the
PI. It is not blocked on anything here.

---

## 6. Suite

`PYTHONUTF8=1 python -m pytest -q -p no:cacheprovider` from `stack/`.
Baseline at HEAD (INHERITED from the brief): **3282 passed / 0 failed / 17 skipped / 2 xfailed**.

`tests/test_loss_determinism.py` alone: **40 collected, 40 passed** (MEASURED).

**Run 1 — 3319 passed / 0 failed / 17 skipped / 2 xfailed** (714.95 s, exit 0).
⚠️ **That is 3282 + 37, not + 40, and the arithmetic is the point.** This run was launched
before the three gradient tests were added to the file, so it covers 37 of the 40. What it does
establish — and it is the thing that matters most — is that **`skipped` and `xfailed` are
unchanged and `failed` is 0**, i.e. no pre-existing test changed status under the sigreg edit.
The 3 uncovered tests passed standalone.

**Run 2 — 3322 passed / 5 failed / 17 skipped / 2 xfailed** (641.78 s).
⚠️ **The 5 failures are a CONCURRENCY ARTIFACT, and that is established, not assumed:**

- All 5 **pass in isolation** immediately afterwards — `pytest tests/test_e_ag1_anchor_floor.py
  tests/test_v6_selector.py` → **32 passed**. Both endpoints of the interval are green.
- The truncated failure reason was `Imp…` = **ImportError**. During run 2 a sibling agent was
  editing `stack/scripts/train_v6_staged.py` and `stack/tanitad/models/v6.py`: the trainer's
  import line now reads `SpectrumAccumulator, o6_rank_verdict` from `tanitad.models.v6`, and
  **neither symbol exists in `HEAD`'s `v6.py`** (`grep -c` → 0 at HEAD, 2 in the worktree).
  pytest imported the pair mid-write and got a torn read.
- Run 1 already contained the complete sigreg change and had **0 failures**.
- ⇒ the failures bracket a window of active editing by another stream and are not attributable
  to this change. **None of the 5 is one of my 40.**

**The arithmetic reconciles exactly.** Run 2 executed 3322 + 5 = **3327** tests
= 3282 (baseline) + **40** (mine) + **5** (the sibling's new `test_e_wc2_sigma_star.py` cases;
that file went 58 → 63 `def test_` between `HEAD` and the worktree). The two "5"s are
unrelated — the sibling's 5 new tests passed; 5 *different* tests hit the torn import.

**Run 3 — ⭐ 3343 passed / 0 FAILED / 17 skipped / 2 xfailed** (565.04 s, exit 0), on a settled
tree with the final 40-test file. `skipped` and `xfailed` are **identical to baseline**; the
growth over 3282 is entirely tests added by concurrent streams during the session (the
SigReg-gate-power agent's new `stack/tests/test_o6_spectrum_power.py` alone contributes 17) plus
this file's 40. **This is the green run against the final state, in a single measurement.**

⭐ All of this is recorded rather than quietly re-run, because "+40" was the number I would
have **predicted** for run 1 and **3319** is what was measured. Writing a predicted suite total
down as if measured is exactly the failure class this programme logs.

---

## 7. Deliverables

| path | what | state |
|---|---|---|
| `stack/tanitad/models/sigreg.py` | `sample_directions` + keyword-only `generator` on `SigReg.forward` / `_forward_fp32` / `position_relaxed`; the defect and its measurement recorded in the module docstring | **staged** |
| `stack/scripts/train_v6_staged.py` | ⚠️ **2 minimal additive edits** — `o6_sigreg_loss(..., *, generator=None)` and `v6_loss_step(..., sigreg_generator=None)` + its call site. | **staged**, but see ⚠️ below |
| `stack/tests/test_loss_determinism.py` | 40 tests: bit-exactness (total, per term, gradients, both builds), 4 negative controls, the dynamic RNG enumerator, the RNG-state watcher, 2 content-anchored no-change guards | **staged** |
| this document + `raw/{repro_before,after_fix}.json` + `code/{repro_nondet,repro_maximal,after_fix}.py` | `.../incoming/2026-08-16-loss-determinism/` | **staged** |

### ✅ RESOLVED — committed as `142ce34`, and HEAD is self-consistent

The orchestrator committed this work while run 3 was in flight:
**`142ce34` "Loss determinism: ONE un-seeded draw made every in-process ablation unmeasurable —
`sigreg.py:70`"**. Verified after the fact, because a partial landing here would have been
worse than none — `test_loss_determinism.py` calls `v6_loss_step(..., sigreg_generator=...)`,
so committing the tests **without** the trainer hunk would have left HEAD broken:

| check | result |
|---|---|
| `sigreg_generator` in `HEAD:stack/scripts/train_v6_staged.py` | **4** ✅ |
| `sample_directions` in `HEAD:stack/tanitad/models/sigreg.py` | **2** ✅ |
| `HEAD:stack/scripts/train_v6_staged.py` blob | **`1a30bdc`** — byte-identical to the blob I staged ✅ |

⚠️ That commit was a **whole-index** commit and therefore also carries a sibling stream's work
(`e_ag1_anchor_floor`, `e_wc2_sigma_star`, `taniteval/v0_antiecho`, `V0_ANTIECHO.md`). Recorded
here so the provenance is not mysterious later; it is the pattern CLAUDE.md permits only after
listing the index, and the result is coherent.

The index and worktree have since moved **past** HEAD for `train_v6_staged.py`
(`1a30bdc` → `2b0e2af` → `267bd28`) as the SigReg-gate-power agent layers its spectrum work
(`SpectrumAccumulator`, `o6_rank_verdict`, `--spectrum-accum`, `--spectrum-ci-reps`) on top.
**The two changes do not collide** — different regions, and all four of my hunks survive in the
worktree. Nothing further is needed from this stream.

⭐ Worth noting for the record: their new code already cites this one — it gives the spectrum
bootstrap *"a DEDICATED generator … so switching the CI on cannot consume the global stream and
move the run's loss (the exact failure the loss-determinism stream just fixed on the SigReg
side)"*. The fix generalised before it was even committed.
