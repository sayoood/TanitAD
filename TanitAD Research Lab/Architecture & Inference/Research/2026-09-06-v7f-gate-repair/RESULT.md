# v7f gate repair — two live gate defects fixed, with deliberate-regression proofs

**Date** 2026-09-06 · **Agent** gate-repair · **Branch** `agent/arch-inf-20260803`
**Evidence class** MEASURED (ours; dev-box RTX 4060 / CPU) unless stamped otherwise.
**Tier** n/a — these are INSTRUMENT repairs, not capability claims. No arm's number moves.

---

## Summary

| defect | what ruled before | what rules now | tests failing on the UNFIXED tree |
|---|---|---|---|
| **D1** `o6_rank_verdict` | `effective_rank` (p ∝ σ, amplitude) | `participation_ratio` (p ∝ σ², energy) | **7 of 9** |
| **D2** panel baselines | `present[0]` — whoever is listed first | a **named** `BASELINE`, three-valued | **9 of 9** |

⭐ **Total: 16 of 18 new tests FAIL on the unfixed tree.** The 2 that pass on both are the
**PREMISE** (the inverting representation exists — a statement about `spectrum_report`, which did
not change) and the **CONTROL** (a healthy arm must still PASS — a guard that only ever fails is
not a guard). Both are labelled as such in the test names.

---

## D1 — the collapse gate ruled on the statistic that inverts

### The defect

`stack/tanitad/models/v6.py::o6_rank_verdict` **computed** `participation_pass` and put it in its
output dict, then **never read it**. All three clauses ruled on `effective_rank`, and the PASS
message even quoted the amplitude floor. The function's own `statistic_note` said *"COLLAPSE is an
energy question — decide on participation"* — the note was right and the code did not follow it.

### The deliberate regression (MEASURED)

Built with `stack/tests/test_o6_verdict_rules_on_participation.py`'s own builders,
n=1100, d=1024, admissible ceiling 1024:

| arm | top-1 energy | `effective_rank` (σ) | `participation_ratio` (σ²) |
|---|---|---|---|
| **COLLAPSED** | **0.549** | **769.09** | **3.31** |
| HEALTHY | 0.142 | 659.69 | 31.27 |

⛔ **The collapsed representation reads a HIGHER effective rank than the healthy one.** It clears the
old absolute floor of 64 by **12×** while holding 55 % of its energy in a single direction. This is
the same shape the v7f board reports for `postrain30k_freeze` (lowest by participation, highest by
effective_rank), reproduced synthetically so it can be pinned.

At the fast test scale (n=260, d=256, jackknife block 26 → 10 clusters), the **same two readings**
give opposite verdicts:

```
OLD (effective_rank)  retention = 1.0815  ci=[1.0564, 1.1068] -> PASS
NEW (participation)   retention = 0.1577  ci=[0.1261, 0.1943] -> FAIL
```

Run against HEAD's own `v6.py`, the unfixed gate returns `status=PASS pass=True` for the
55 %-energy arm. **That is the false pass, executed.**

### What changed, and what deliberately did not

* **Clause 2 (RETENTION) now rules on `participation_ratio`**, CI-based, against the phase-start
  reference. ⭐ This is the *structurally sound* place for the energy statistic: `cur` and `ref` are
  the same run, same corpus, same ambient `d`, same pooling — so the ratio is immune to the
  corpus/dimension sensitivity that makes the **absolute** participation number unquotable.
* `spectrum_report` now emits **`participation_ratio_ci95`** (same leave-one-cluster-out jackknife,
  via a new `_pr_from_gram`). ⚠️ It is computed with `reps=0` **on purpose**: the bootstrap block is
  the only consumer of the RNG, so adding this key leaves a live run's sampling stream bit-for-bit
  unchanged. Its coverage stamp is labelled `INHERITED` — 0.85/0.867 was measured for
  `effective_rank`, not for this functional, and quoting it bare would be the scope error.
* **`effective_rank` is still reported on every branch, under its existing key names**, so the
  banked series stays comparable and no downstream reader breaks. It is marked non-ruling *inside
  the record* (`effective_rank_is_ruling: False`, `ruling_statistic`,
  `effective_rank_retention_DIAGNOSTIC`). Renaming the keys would have been the more satisfying
  edit and the more destructive one.
* ⛔ **No silent fallback.** A reading with no `participation_ratio` is INCONCLUSIVE, never a verdict
  computed from the inverting statistic.

### ⚠️ Clause 3 (the absolute floor) — REPORTED, NOT RULING. **This is a PI decision, surfaced.**

Both candidate floors are currently inadmissible, for two *different* reasons:

| floor | status | why |
|---|---|---|
| `O6_RANK_FLOOR = 64` on `effective_rank` | **wrong statistic** | it is the inversion above; it produces false PASSES *and* false FAILS |
| `O6_PARTICIPATION_FLOOR = 8.56` | **right statistic, wrong number** | reproduced by **no** live instrument; moves **3.51×** on episode diversity alone at fixed encoder/n/d (`O6_PARTICIPATION_REFERENCES`); the constant's own docstring already carries the standing instruction *"DO NOT FAIL AN ARM ON THE PARTICIPATION CLAUSE"* |

⇒ The absolute clause now fires **only** when the caller passes **both** `participation_floor` and
`participation_reference` (a string naming the matched corpus/n/d). A floor without a reference is
INCONCLUSIVE, not a FAIL — that unnamed-threshold shape is what produced champ30k's recorded FAIL
against a number from a different `d`.

⭐ **This makes the gate rule on the right statistic without inventing a threshold.** A criterion
that cannot rule must say INCONCLUSIVE — the principle `rank_gate_capacity` already enforces one
level up in the same file.

> **PI DECISION — recommended default: KEEP the absolute clause DISARMED.**
> Arming it needs a participation floor re-measured at **matched corpus and matched `d`** (z_op is
> d=2048; the DINOv3 reference column is d=1024, so they are not comparable in either direction).
> The alternative — retire `O6_PARTICIPATION_FLOOR` from live gates entirely and rely on retention —
> is defensible and cheaper. **What is not defensible is the status quo ante**, where an absolute
> clause fired on the statistic that inverts.

### Blast radius handled

* `stack/scripts/train_v6_staged.py` — both call sites (`:5148`, `:7347`) get the repair for free.
  The `STAGE_GATE_SPEC` **prose** at `:616` still said *"FLOOR: pooled effective_rank >= 64"* and has
  been corrected; leaving it would have been the "prose lied to us" failure this repo has a rule for.
  ⚠️ `O6_spectrum` is `reported`, not `required`, at every stage, so **no launch behaviour changes**.
* `stack/tests/test_o6_spectrum_power.py` — two tests **pinned the defect** and were rewritten with
  the reason recorded in the docstring. Three sibling guard tests still pass unchanged, which is the
  evidence the gate still fires on a genuine collapse and still reaches a real PASS on a healthy arm.
* `stack/tests/test_x4_layer_spectrum.py` — its pin's contract now allows the *ruling* keys to move
  (pinning them would pin the defect) while still requiring every diagnostic key to match.

### ⛔ A separate finding: that pin test has been silently SKIPPING

`_pre_x4_module()` loaded a historical `v6.py` under a bare module name. `v6.py` uses **relative
imports**, so it raised *"attempted relative import with no known parent package"*, which the bare
`except Exception: return None` swallowed — and the test then skipped saying **"git could not supply
a pre-X4 revision"**. Git had supplied it perfectly; the *loader* was broken. A guard that cannot
fail is decoration, and this one also **misdirected the reader to the wrong subsystem**. Fixed by
loading it as `tanitad.models.v6_pre_x4` with `__package__` set, and the skip reason is now honest.

---

## D2 — the relative gate's baseline was "whoever is listed first"

### The defect

`full_panel.py` and `rolled_predict.py` both took `present[0]` as the reference arm and ruled the
rest with `present[1:]`. Three failures, none of which announces itself:

1. **Verdicts flip with run order.** Reorder the arms — or simply run before `rdw8`'s checkpoint has
   landed — and a different arm becomes the baseline, *while the printed header still said
   `vs rdw8`*. The table asserted a comparison it had not made.
2. **The first arm never got a verdict at all**, and a missing rejection is indistinguishable from a
   pass to anything reading the JSON.
3. **`None` is falsy.** The renderer was
   `'REJECTED' if v['REJECTED_by_kill_gate'] else 'passes gate'` — so the moment a "cannot rule"
   state exists, it renders as **passes gate**.

### The deliberate regression (MEASURED)

Same three arms, two orders, frozen pre-repair logic
(`stack/tests/test_panel_baseline_named.py::_OLD_POSITIONAL_frozen`):

| order | baseline taken | `rdw8` verdict | `o8w1p0` verdict |
|---|---|---|---|
| `[rdw8, o7w1p0, o8w1p0]` | `rdw8` | **no row at all** | REJECTED |
| `[o8w1p0, o7w1p0, rdw8]` | `o8w1p0` | ruled → "passes gate" | **no row at all** |

⛔ **The arm that was REJECTED in one order becomes the BASELINE in the other.** The repaired
`relative_verdicts` returns dicts that compare **equal** across both orders.

### The fix

New module **`stack/tanitad/eval/panel_gate.py`** — in the stack, so it is covered by `pytest -q`
and cannot drift per panel:

* `select_baseline(present, baseline)` — by NAME; refuses an empty/positional baseline outright.
* `relative_verdicts(...)` — **one entry per present arm, the baseline included**, three-valued.
  Nothing in it indexes `present`, so order-invariance is by construction rather than by care.
* `render_verdict(...)` — three-valued; `None` renders **NO VERDICT**, never a pass.

An arm that is its own baseline, and every arm in a run whose baseline is absent, reads
**NO VERDICT**. ⛔ **No substitute baseline is ever promoted** — absolute per-arm numbers are
order-free and still stand, and the artifact now carries `baseline_arm` / `baseline_present` /
`baseline_is_positional` so a reader never has to infer the reference from row order. The delta
table's header is derived from `BASELINE`, so it can no longer name an arm it did not compare to.

Also hardened: the pose/target-populating guard (`if arm == present[0]`) is now a named
`pose_anchor` with an **assertion** that every arm produced identical per-clip row counts — the guard
is only order-free if that holds, and it was being assumed.

---

## Row 3 (≈0-motion readout) — GPU blocked; zero-GPU narrowing done

**GPU status: BLOCKED, not skipped.** Dev-box RTX 4060 at **100 % util, 5153/8188 MiB**, four
sibling `refav1_arm.py` arms live. Per the brief I polled with a check command (an armed
`nvidia-smi` + process-count monitor, `arms=0` as the release condition) rather than a sleep-retry
loop, and did not touch the sibling queues. No slot freed within this turn.

### What I established without a GPU (MEASURED, by source read)

**The T1 emission path carries no multiplicative scale constant anywhere:**

| stage | file | what it does to scale |
|---|---|---|
| readout | `metric_dynamics.py:212` `StepDisplacementReadout.forward` | `LayerNorm → Linear → GELU → Linear(→3)`. Raw output, **no scale, no output activation** |
| rollout | `metric_dynamics.py:233-244` `rollout_decode` | stacks Δposes, `accumulate_se2` — **no scale** |
| speed | `taniteval/taniteval/rollout.py:280-284` `dense_speed_profile` | `norm(Δp)/dt` — **no scale** |

Combined with the brief's input-side finding (`rollout.py:71` `SPEED_SCALE = 10.0` matches
`flagship_v15.py:85`'s trained 10.0):

⇒ **A `v/30`-vs-`v/10` style error cannot be introduced by the T1 harness on either the input or the
output side.** Row 3's stated hope — *"if it is a scale error, every T1 number is recoverable by
re-analysis with zero retraining"* — **is not supported by the eval code path.**

⚠️ **Scoped honestly: this rules out the HARNESS, not the TRAINER.** A readout supervised against a
wrongly-scaled target would produce the same ≈0 signature, and I did not close that path.

### ⛔ A trap I hit and caught, worth recording

My first attempt to quantify the corpus speed from `taniteval/results/windows_flagship-30k.pt`
computed `norm(diff(gt))/0.1` and got **63.85 m/s (230 km/h)**. `gt` is `[881, 4, 2]` at **`wp_steps`
spacing, not 10 Hz ticks** — the same "correct formula, wrong units" class as the cylindrical-FOV and
`alat`/curvature traps. Caught by the implausibility of the value, not by the code.

⇒ **And I am therefore NOT quoting a corpus speed against the board's `speed_bias −10.381`.** That
figure belongs to **v7-tiny**; `windows_flagship-30k` is a **different arm**. Dividing one arm's bias
by another arm's mean speed is exactly the cross-arm error this repo has a rule for. The
dead-vs-mis-scaled discrimination therefore stays **OPEN**, and needs v7-tiny's *own* val speed.

### The cheapest next probe, and its blocker

**Zero GPU:** load the banked v7-tiny checkpoint on CPU and read `step_readout_op`'s final `Linear`
weight/bias norms. If the weight term is ≈0 relative to the bias, the readout is a **constant
emitter** — which would make `echo_index 0.0000` vacuous exactly as the board suspects, and would
mean **no re-analysis can recover the T1 numbers**.

⛔ **BLOCKED ON THE ARTIFACT, and I verified this rather than assumed it:** `taniteval/results`
holds **29** dumps (control read non-zero) and **none** is a v7/v7-tiny arm; the locally reachable
checkpoints are refav1 (`refav1_eval_slice/ckpt*`) and phase-0 flagship
(`tanitad-wt/stack/experiments/p0-*`). **What unblocks it: the path to a banked v7-tiny checkpoint.**

---

## Deliverable manifest

All paths repo-relative, staged in the working tree on `agent/arch-inf-20260803`.

| artifact | path |
|---|---|
| D1 fix | `stack/tanitad/models/v6.py` (`o6_rank_verdict`, `_pr_from_gram`, `_cluster_interval_er`, `spectrum_report`) |
| D1 prose fix | `stack/scripts/train_v6_staged.py` (`STAGE_GATE_SPEC` O6 criterion text) |
| D1 tests | `stack/tests/test_o6_verdict_rules_on_participation.py` (new) |
| D1 pins updated | `stack/tests/test_o6_spectrum_power.py`, `stack/tests/test_x4_layer_spectrum.py` |
| D2 fix (shared) | `stack/tanitad/eval/panel_gate.py` (new) |
| D2 panels | `TanitAD Research Lab/Architecture & Inference/Research/2026-08-19-simwam-analysis/code/full_panel.py`, `.../rolled_predict.py` |
| D2 tests | `stack/tests/test_panel_baseline_named.py` (new) |
| this report | `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-v7f-gate-repair/RESULT.md` |

**Suite:** `84 passed, 0 failed` over the affected set
(`test_o6_spectrum_power`, `test_o6_verdict_rules_on_participation`, `test_panel_baseline_named`,
`test_subspace_sigreg`, `test_participation_floor_provenance`, `test_rank_gate_abort`,
`test_x4_layer_spectrum`).
⚠️ `test_x4_layer_spectrum.py::...UNCHANGED...` is deselected in that run: its git-history walk over
every `v6.py` revision exceeded **15 minutes** on the G: mount and was stopped. That cost is
pre-existing (the walk runs before the loader), but it is now a *live* test rather than a silent
skip, and its runtime should be bounded before it lands in a routine `pytest -q`.
