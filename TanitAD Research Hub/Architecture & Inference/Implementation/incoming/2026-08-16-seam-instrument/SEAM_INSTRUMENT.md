# F-16 — THE X2 BAND-SEAM INSTRUMENT. Verify, never repair.

**Date:** 2026-08-16 · **Branch:** `agent/arch-inf-20260803` · **Author:** arch-inf subagent (F-16 stream)
**Closes:** `DIAGRAM_CONFORMANCE.md`:131 — *"a dedicated X2 seam instrument (per-seam paired-bootstrap CI in the registry row) is ⬜ **NOT BUILT**"* — and `…:196` (X2 🟨 PARTIAL) and fix-list row **F-16**.

> ⛔ **ESCALATION — INTEGRATION NEEDED, TWO ITEMS.**
> 1. **The instrument is BUILT, TESTED and VALIDATED, but it has never been run on a real v6 arm, because NOTHING IN THE PROGRAMME BANKS THE 60-STEP PLAN.** `V6Stack.emit` is the only place it exists and `train_v6_staged.py` saves only checkpoints (three probes, §7). Six lines in the eval/dump path close this; the exact snippet is in §8. Until then the S-T/S-S gate's `X2_seam` row can only ever read *not-run*.
> 2. **`STAGE_GATE_SPEC`'s `X2_seam` owner was a bare estimator name** (`"taniteval/ci.py (PAIRED bootstrap only)"`) — an estimator is not an owner, and the row named no instrument because none existed. **Edited in this stream** to name `taniteval/tools/seam_probe.py`, and an `X2_seam` **criterion** was added (S-T had none). Both are data-only edits to `STAGE_GATE_SPEC`; `reported` probes are never adjudicated, so behaviour is unchanged. **Another agent is live in that file** — flagging rather than assuming.

---

## 1. Why this instrument exists, and what it must be able to do

The binding diagram cell reads: *"ONE 60-step (a,κ)@10 Hz rollout; seam discontinuity-free **by construction**; seam metrics **verify, never repair**."* The audit found the construction half ✅ in code — `PLAN_STEPS=60`, `DT=0.1`, ONE `unicycle_rollout`, and `V6Config.split_bands` returning **views** (*"materialising it as a copy is how a 'seam' gets invented"*) — and the **verification half absent**. The trainer logs `plan_ade_0_2s` / `plan_ade_2_6s`; **two band ADEs cannot see a discontinuity**, because an arm can have identical band errors and still jump at the boundary.

⚠️ **The point of this build is FALSIFICATION.** An instrument that can only confirm the architecture is the C13 family (*"instruments structurally unable to report the answer they are cited for"*) and is worth nothing here. Three mechanisms make this one able to fail, and all three are MEASURED below, not asserted: a three-valued verdict that can return INCONCLUSIVE, a boundary scan that measures the rule's own false-positive rate on the same data, and a seam-injection validation that shows it detecting the defect it hunts.

## 2. THE NULL

> **H0** — the band boundary is **exchangeable** with every other step boundary of the same rollout: the discontinuity at the 2 s edge is drawn from the same distribution as the within-band step-to-step discontinuities.

⚠️ **"non-zero" is not "seam".** Every real control sequence has non-zero step-to-step differences everywhere. A seam is a boundary whose discontinuity is an **OUTLIER against the within-band null**. Every statistic here is therefore a **contrast between the seam boundary and the within-band boundaries of the SAME window** — which is also why the estimator is the **paired** episode-cluster bootstrap.

**The boundary is DERIVED, never a literal.** `V6Config.band_slice("op") = slice(0, 20)` and `band_slice("tac") = slice(20, 60)`, so the edge sits between plan step 19 and step 20 ⇒ boundary index **20**, t = 2.0 s. `seam.seam_boundary_of()` recomputes it from the band spec and **refuses a gap or an overlap** — the same refusal `V6Config.__post_init__` makes, because *a gap or an overlap here IS the stitched-trajectory defect*. Pinned by `test_seam_boundary_is_the_v6_band_edge_not_a_literal`.

**The statistic.** For a per-step scalar channel `x[0..59]`, the order-*m* discontinuity at boundary *i* is the *m*-th finite difference anchored at `i-1` (`|np.diff(x, n=m)|[i-1]`): `d1` level jump, `d2` slope jump, `d3` curvature jump. Channels: **`a`** (m/s²), **`kappa`** (1/m) from the emitted controls; **`wp_x`**, **`wp_y`** (m) from the integrated waypoints; **`err`** (m) — the per-step ADE — when a plan target is present. Per window: `d_seam = D_m(20)`, `null_ref = MEAN of D_m(i), i ≠ 20`, `excess = d_seam − null_ref`, plus a distribution-free mid-**rank** (H0 = 0.5 exactly) and a **top-1** indicator (H0 = `1/(n_null+1)`).

**Two nulls, because exchangeability is an assumption.** `global` = every other boundary; `local` = `|i − 20| ≤ 5`, which removes a smooth index trend to first order. **A row counts as a seam only when BOTH nulls fire.** A global-only firing is an index trend until shown otherwise. This is not theoretical — see §6.2.

**The estimator, delegated verbatim.** `taniteval.ci.paired_episode_cluster_bootstrap` for the contrast (both arms on the same resampled episodes) and `episode_cluster_bootstrap` for the rank/top-1/SE. ⛔ `overlapping_holdout_se` is used nowhere and the string does not appear in either source file (pinned by test): it narrows 1.107–3.100× **and** biases the point estimate bidirectionally, **up to a sign flip on paired deltas** — and the seam contrast *is* a paired delta, i.e. exactly the shape that lesson (`ctx→tactical` +0.0439 → true +0.0148) was measured on.

## 3. THE TIER

| block | tier | why |
|---|---|---|
| **continuity** (the seam test) | **T1**, `tier_invariant: true` | it consumes the emitted 60-step plan and **nothing else** — no recorded future actions, no future frames, no ground truth. There is no teacher-forcing channel for it to be contaminated by, so the same numbers come out at T0 and T1. Stamped T1 (the primary tier) rather than left unstamped. |
| **bands** (per-band ADE) | **inherits the dump's declared tier** | it compares the plan against the GT future, which is capability-adjacent. **An undeclared tier is a hard CLI refusal** — an un-tiered number is exactly what `EVAL_DOCTRINE.md` forbids (pinned by `test_cli_refuses_a_dump_with_no_tier`). |

Per-band ADE is reported **separately, never pooled** (§4b: *"report the bands SEPARATELY. A pooled 0–6 s number cannot show the 2 s seam"*). The pooled 0–6 s row exists for cross-arm comparability and is labelled as such. ⚠️ The tac−op band delta is **horizon growth, not a discontinuity**, and the record says so on its own line.

## 4. THE PRE-COMMITTED OUTCOMES

Registered here **before** any v6 arm is scored, with both branches committed.

**Materiality floor.** `floor = k · scale`, `scale` = the within-band mean step, `k = 1.0` by default ⇒ *"the boundary step is more than **twice** a typical within-band step"*. Scale-free by construction (one bar serves m/s², 1/m and metres), CLI-declarable, and stamped into every record so the reader always sees the bar the verdict was read against. The verdict is read on the **scale-normalised** interval (the identical interval — both arms divided by the same constant, so the bootstrap is equivariant — but off the programme's 4-dp physical display resolution, which for κ ≈ 1e-2 is coarse enough to decide a verdict by rounding).

| verdict | condition | what it means for the architecture |
|---|---|---|
| ⛔ **SEAM** | the paired CI's **lower** bound exceeds the floor, **under both nulls** | **THE "BY CONSTRUCTION" CLAIM IS FALSIFIED** at that channel/order. It is REPORTED to the PI and the diagram/code re-examined. ⛔ **No loss term may be added in response.** The instrument names the channel and the order, which is the whole diagnostic value: a control-space seam and a position-space seam are different defects (§6.1). |
| ✅ **NO_MATERIAL_SEAM** | no material excess **and** MDE@80 % ≤ floor **and** ≥ 8 episode clusters | **THE CLAIM SURVIVES A TEST IT COULD HAVE FAILED.** This is a POSITIVE result, not a silence — the record carries the MDE, so the claim reads *"no seam larger than X of a within-band step, at 80 % power, on n episodes"*. |
| 🟨 **INCONCLUSIVE** | CI covers 0 but MDE@80 % > floor, **or** < 8 episode clusters, **or** the CI is at float64 resolution | the instrument could NOT have seen a material seam. **Not a pass.** Raise n or report the bound. |
| ⬜ **DEGENERATE** | the channel is identically zero at every boundary | the expected reading for a **zero-init emission head** (a = κ = 0 ⇒ the CV straight rollout has no discontinuities to rank). **Not a pass** — this is the specific vacuous-pass this branch exists to prevent, and S-W's emission head is exactly this until S-T trains it. |

All four are reachable and none is decorative — pinned by `test_every_verdict_in_the_space_is_reachable_none_is_decorative`.

## 5. THE POWER — MEASURED, at val40 scale

**MEASURED 2026-08-16, this box, 880 windows / 40 episodes / n_boot 2000**, on the self-test's genuine single-rollout arm (declared synthetic OU controls at val40 dimensions, integrated by the programme's own `unicycle_rollout`). Artifact: `raw/seam_selftest_val40scale_n2000.json`.

| MDE at 80 % power, in multiples of a typical within-band step | value |
|---|---|
| across all 30 rows (5 channels × 3 orders × 2 nulls) | **0.0137 – 0.1510**, median **0.0717** |
| control channels (`a`, `kappa`) | **0.0622 – 0.0759** |
| position channels (`wp_x`, `wp_y`) | **0.0146 – 0.1182** |

⇒ **At the canonical 40-episode / 881-window scale the instrument resolves a seam of ≈ 1.4 %–15 % of a typical within-band step at 80 % power**, against a materiality bar of 100 %. The bar is ~7–70× the resolution, so a `NO_MATERIAL_SEAM` at this n is a well-powered null by a wide margin, and the record says so per row (`power.powered_for_material_seam`).

⚠️ **THE HONEST LIMIT, stated rather than buried: power in CONTROL space depends on the emitted plan's temporal coherence.** A trained planner emits temporally coherent controls (that is precisely what `DiffusionProposalGenerator`'s correlated noise exists to produce), and the numbers above assume it. Against a **white-noise** control sequence — which is what the emission MLP produces at random init, since its last layer maps one feature to all 120 outputs through independent rows — the within-band null is already as large as any control-space stitch could make it, and **no instrument can see a control-space seam**. The power block reports this per row rather than hiding it, and the **position channels retain power regardless**, which is why both spaces are always reported.

## 6. VALIDATION — it detects the defect it hunts

### 6.1 Seam injection: two independently-rolled bands concatenated

Two distinct stitch defects, because they fail in different spaces and a single test would miss one.

| arm | construction | headline | confirmed rows (both nulls) | effect size |
|---|---|---|---|---|
| `genuine` | ONE OU control sequence, ONE `unicycle_rollout` | ✅ **NO_MATERIAL_SEAM** | *none* | — |
| `stitch_controls` | tactical band from an **independent** control sequence, integrated by ONE rollout (position stays continuous) | ⛔ **SEAM** | `a/d1`, `a/d2`, `kappa/d1`, `kappa/d2` | `a/d1` excess **+2.5864 [+2.4086, +2.7715]** × scale (**+0.6339 m/s²**) |
| `stitch_rollout` | two **independent** rollouts, the second re-based at its own origin (the classic position jump) | ⛔ **SEAM** | `wp_x/d1..3`, `wp_y/d1..3`, `err/d1..3` | `wp_x/d1` excess **+19.2750 [+18.8688, +19.6993]** × scale (**+18.5688 m**) |

⭐ **Each defect fires in exactly the right channel and nowhere else:** the control stitch leaves `wp_*` clean (one integrator ⇒ continuous position) and the rollout stitch leaves `a`/`kappa` clean (the controls are the genuine ones). That cross-pattern is the strongest evidence available that the instrument is measuring what it claims to.

### 6.2 The false-positive calibration (the C13 defence, measured)

The identical rule applied at **all 59 boundaries**, `n_boot 400`. Artifact: `raw/seam_scan_calibration.json`.

| arm / channel | FPR at the 58 non-seam boundaries | hotspot | seam's rank |
|---|---|---|---|
| genuine / `a` | **0.0** | 54 | 25 / 59 |
| genuine / `wp_x` | **0.0** | 1 | 19 / 59 |
| stitch_controls / `a` | **0.0** | **20** | **0** ⭐ |
| stitch_controls / `wp_x` | **0.0** | 1 | 19 / 59 |
| stitch_rollout / `a` | **0.0** | 54 | 25 / 59 |
| stitch_rollout / `wp_x` | **0.0** | **20** | **0** ⭐ |

⇒ the rule fires at the injected seam and **at none of the other 58 boundaries in any arm**; on the seam-free arm the band edge ranks mid-pack (19th and 25th of 59), which is what H0 predicts. A `SEAM` verdict at boundary 20 is therefore meaningful against a measured 0 % reference rate.

### 6.3 The real `V6Stack.emit` path

The "by construction" claim is a claim about **CODE**, not about weights, so it is checked on the code (`test_the_real_emit_produces_ONE_rollout_whose_bands_are_VIEWS`, `test_the_real_emit_path_carries_no_seam_at_the_band_edge`, CPU, tiny geometry, `plan_steps` kept at 60):

* `emit` returns `[B, N, 60, 2]` controls **and** waypoints — ONE rollout;
* `split_bands` returns **views** (`op.data_ptr() == wp.data_ptr()`) that reassemble to the identical tensor by `torch.equal` — bands are slices, not copies, exactly as the diagram claims;
* the probe finds **no confirmed position seam** in the emitted waypoints of the real path.

⚠️ This validates the **plumbing and the construction**, not a trained arm. It is not a capability claim and is not offered as one.

### 6.4 Why the bands are never pooled — demonstrated, not asserted

`--dump-b` scores two arms on the SAME windows with the paired estimator. Arm B here is arm A with **only the tactical band replaced** (10 episodes / 80 windows, n_boot 200):

| band | paired delta A−B | separated |
|---|---|---|
| `ade_0_2s` | **+0.0000 [+0.0000, +0.0000]** | no — the operative band is **bit-identical** |
| `ade_2_6s` | **−13.0344 [−16.1844, −9.9047]** | yes |
| `ade_0_6s_pooled` | −8.6896 [−10.7896, −6.6031] | yes |

⇒ the pooled row reports **−8.69 m** for a pair of arms whose first band does not differ **at all**. That single number describes neither band. Pinned by `test_cli_paired_arms_show_WHY_the_bands_are_never_pooled`.

## 7. TWO DEFECTS FOUND — in this instrument's own statistic, during construction

Both were found by running the instrument against its own null, and both would have produced **false seams on seamless rollouts**. Recording them because the failure class matters more than the fix.

1. ⛔ **A MEDIAN null reference is BIASED UPWARD under H0.** The first build used the per-window **median** of the within-band boundaries — the "obviously robust" choice. It is not admissible: for the right-skewed `|Δ^m x|` distribution `E[median of n draws] < E[draw]`, so `E[d_seam − median(null)] > 0` **with no seam present**. MEASURED on the genuine arm (144 windows / 12 episodes): `a/d1` **+0.0387 [+0.0174, +0.0587]** and `kappa/d1` **+0.0028 [+0.0018, +0.0039]** — intervals **separated from zero on a trajectory containing no seam**. The shipped reference is the **mean**, which is exactly unbiased under exchangeability. ⛔ Pinned by `test_the_MEDIAN_reference_is_BIASED_under_H0_which_is_why_it_is_not_used`, which recomputes both and **asserts the median version is still biased** — so the pin cannot silently stop protecting anything.
   *Class: a robustness choice that changes the estimand.* Same family as the `overlapping_holdout_se` lesson — an estimator whose centre is not the quantity you meant.
2. ⛔ **A cluster bootstrap over very few episodes reports a CONFIDENTLY WRONG SE, and it looks like precision.** MEASURED at 2 episodes × 2 windows on i.i.d. data: the bootstrap SE collapsed to **0.019× the within-band scale** — an 80 %-power MDE of **0.054×**, which would have licensed a "well-powered null" off **four windows**. The reason is combinatorial — resampling `n` clusters with replacement admits only `C(2n-1, n)` distinct resamples (**3** at n=2, 10 at n=3, 35 at n=4), so the 2.5th percentile is supported on a handful of values. The instrument now refuses a clean bill below **8 clusters** (6 435 distinct resamples) and returns INCONCLUSIVE with the reason.
   *Class: a probe reporting the wrong scope* — the `df`-on-a-pod family. An under-covering interval does not look broken; it looks like a very good result.

⚠️ **A third, non-defect finding that travels with every reading:** on the genuine arm, `wp_x/d1` against the **global** null reads **−0.0176** and `err/d1` reads **−0.1096 [−0.1524, −0.0712]** — separated, with no seam present. That is an **index trend** (step displacement and accumulated error both drift across the horizon), and it vanishes under the local null / at order ≥ 2. This is why both nulls are mandatory and why a global-only firing is never called a seam.

## 8. What could NOT be run, and exactly what closes it

⛔ **There is no banked v6 60-step plan anywhere in the repo, so the instrument has produced NO real-arm numbers.** Three independent probes:

1. `grep torch.save|np.savez` over `train_v6_staged.py` / `gate_emitters.py` / `run_gate.py` → the only save is the **checkpoint** (`{"stack": …, "opt": …}`). Nothing banks `emit()`'s output.
2. `find` for `*plan*.pt` / `*emit*.pt` / `*v6*.pt` → only REF-C fans and the rung-1 planner dump.
3. Shape inspection of the candidates: `fan_emitted_refc-xl-30k.pt` is `fan (881, 256, 4, 2)` with `wp_steps` of length **4** — a 4-waypoint, 2 s plan **entirely inside the operative band**, so it has no 2 s boundary to test; `r1planner_compact_K185.pt` carries `dense_de`/`fed_actions`/`psi`, no control sequence.

⇒ **This is stated plainly rather than demonstrated on something else.** The instrument was not pointed at a REF-C fan to manufacture a number; a plan with no tactical band cannot answer the X2 question.

**What closes it — six lines wherever `emit` is already called (zero extra GPU):**

```python
out = stack.emit(z_op, e_g_tac, v0)              # no GT, no future actions
torch.save({"controls":  out["controls"].cpu(),   # [B, N, 60, 2]
            "waypoints": out["waypoints"].cpu(),  # [B, N, 60, 2]
            "sel": out["sel_score"].argmax(-1).cpu() if "sel_score" in out
                   else torch.zeros(len(v0), dtype=torch.long),
            "gt": plan_target.cpu(),              # [B, 60, 2], optional
            "eid": eids, "tier": "T1", "arm": "<run>@<step>",
            "plan_steps": 60, "dt": 0.1,
            "op_band_s": [0.0, 2.0], "tac_band_s": [2.0, 6.0]}, path)
```

then

```
python taniteval/tools/seam_probe.py --dump <path> --out <run>_x2_seam.json
```

⚠️ **Run it on an S-T (or later) checkpoint, not S-W.** In S-W the planner is absent and the emission head sits at its zero-init, so every control is exactly (0, 0) and the probe correctly returns **DEGENERATE** — which is the right answer and not a pass.

## 9. How to run it

```
# the validation, no GPU / no data / no checkpoint — exits non-zero if the
# instrument fails to detect an injected seam
python taniteval/tools/seam_probe.py --self-test

# a real arm
python taniteval/tools/seam_probe.py --dump <emit_dump.pt> --arm v6-st@30k \
    --tier T1 --out <out>.json                       # scan on by default

# two arms on the SAME windows -> per-band PAIRED deltas (never quadrature)
python taniteval/tools/seam_probe.py --dump A.pt --dump-b B.pt --out ab.json
```

Knobs that change a verdict are all declarable and all stamped: `--materiality-k`, `--local-halfwidth`, `--orders`, `--candidate {winner,all,<int>}`, `--seam` (an override is recorded as an override and does **not** answer the X2 question), `--n-boot`, `--seed`.

## 10. Tests

`stack/tests/test_v6_seam_probe.py` — **34 tests, all passing** (14 s, CPU, no GPU). Groups: the geometry is derived from v6 and refuses a band gap · the finite-difference stencil against hand numbers · the estimator is `taniteval.ci`'s **by monkeypatched call-count identity**, not a lookalike · `overlapping_holdout_se` reachable from nowhere · verify-never-repair as a **source** property (no torch import, no gradient, nothing named like a loss) · H0 unbiasedness + the median-bias regression pin + the too-few-clusters guard · **fires on both injected seams** and the scan's argmax lands on the injected boundary · does **not** fire on a genuine rollout and returns a well-powered null · all four verdicts reachable · per-band never pooled + paired cross-arm · the real `V6Stack.emit` path (ONE rollout, bands are views) · candidate/winner handling · six CLI refusals.

**Suite baseline — MEASURED by this stream, not inherited.** ⚠️ The brief quoted *"3572 passed / 0 failed"* and the orchestrator later retracted it as an inherited figure, warning that the suite might be carrying **9 pre-existing failures**. This stream ran the full suite **itself, on this tree, before its first edit**:

```
3574 passed, 7 skipped, 2 xfailed, 10 warnings in 440.63s   [exit code 0]
```

and a **second, independent** full-suite record banked by another live stream 3 minutes later reads:

```
3574 passed, 7 skipped, 2 xfailed, 10 warnings in 436.24s   [exit code 0]
```
*(`TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-08-16-sam3-dtype-fix/raw/pytest_full_suite.txt`)*

⇒ **two independent measurements on this tree agree: 3574 passed / 0 failed / 7 skipped / 2 xfailed.** The "9 failed / 3563 passed / 3572 collected" figure is not reproducible here and is internally inconsistent (3563 + 9 + 7 + 2 = **3581**, not 3572). The most likely explanation is a **transient tree state**: this repo has ≥3 agents editing concurrently, and a moment where one stream's source edit has landed but its test has not (or vice versa) produces exactly that shape. ⚠️ **No `git stash` was used to recover a baseline** — stashing a tree with live sibling agents would have destroyed their uncommitted work.

**AFTER, measured on the finished tree** (`raw/pytest_after_final.txt`):

```
3658 passed, 7 skipped, 2 xfailed, 10 warnings in 443.61s   [exit code 0]
```

⚠️ **3658 − 3574 = +84, and only 34 of those are mine.** The rest arrived from the ≥3 sibling streams that added test files during this window (chiefly the F-18 agent-slot-decoder stream). Two consecutive full runs 8 minutes apart read **3655** then **3658** with no edit of mine in between the second pair beyond my own +2 — **the tree is moving under every measurement here, and any global count is a snapshot, not a property of a change.** That is exactly why the attribution below is the claim, and the global count is only context.

**This stream's delta, attributable independently of any global count:**
| | |
|---|---|
| new tests added | **+34**, all passing (`tests/test_v6_seam_probe.py`, 14 s, CPU) |
| new failures introduced | **0** — 0 failed in both the before and the after run |
| existing files touched | exactly one — `stack/scripts/train_v6_staged.py`, **data-only** inside `STAGE_GATE_SPEC`. Its four covering test files re-run green: `test_v6_staged.py` + `test_v6_chain.py` + `test_v6_stage_revalidation.py` + `test_gate_emitters.py` → **163 passed** |
| everything else | **new files**, imported by nothing but this stream's own test |

⛔ **No claim is made that "the suite is green because of this work"** — it was green before and after, both times measured here, and the credit/blame for anything else in that window belongs to the streams that made it.

## 11. Deliverable manifest

| artifact | where | notes |
|---|---|---|
| the statistics module | `repo:taniteval/taniteval/seam.py` | numpy-only; the null, the estimator delegation, the verdict, the power |
| the CLI | `repo:taniteval/tools/seam_probe.py` | ⚠️ the brief said `stack/taniteval/tools/…`; **`taniteval/` is a REPO-ROOT package**, not under `stack/` — placed at the real path, matching `t1_eval.py` / `eval_four_families.py` |
| the tests | `repo:stack/tests/test_v6_seam_probe.py` | 32 tests, in the suite the gate reads |
| this writeup | `repo:TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-16-seam-instrument/SEAM_INSTRUMENT.md` | |
| validation record (880 windows / 40 episodes / n_boot 2000) | `repo:…/2026-08-16-seam-instrument/raw/seam_selftest_val40scale_n2000.json` | the §5 power and §6.1 injection numbers |
| scan calibration record | `repo:…/2026-08-16-seam-instrument/raw/seam_scan_calibration.json` | the §6.2 FPR table |
| suite baseline, measured **before** the first edit | `repo:…/2026-08-16-seam-instrument/raw/pytest_baseline_pre_edit.txt` | 3574 passed / 0 failed, exit 0 — the retraction of the inherited "9 failed" figure (§10) |
| suite after, measured on the finished tree | `repo:…/2026-08-16-seam-instrument/raw/pytest_after_final.txt` | 3658 passed / 0 failed, exit 0 |
| gate wiring | `repo:stack/scripts/train_v6_staged.py` — **WORKING TREE, DELIBERATELY NOT STAGED** | `STAGE_GATE_SPEC` S-T/S-S `X2_seam` **owner** now names the instrument; S-T gains an `X2_seam` **criterion**. Data-only; `reported` probes are never adjudicated. ⚠️ **Not staged on purpose:** the **F-18 agent-slot-decoder stream is live in the same file** (`STAGE_MAY_INTRODUCE["S-T"] += "agent_slots."`, `build_stack_from_args`, and `stack/tanitad/models/v6.py`). `git add <file>` stages the whole blob, so staging it here would sweep a sibling's in-progress work into this stream — the exact hazard `CLAUDE.md`'s git-hygiene section records (twice). The three hunks are **in the working tree on this branch** and are listed above so the integrator can see exactly what is mine. |

*Nothing lives on a pod or in a worktree; the one unstaged item is named above with its reason. No GPU was used: every number here is CPU, and Thor was not touched.*

**The three `train_v6_staged.py` hunks that belong to THIS stream** (everything else in that file's diff is F-18's):
1. `STAGE_GATE_SPEC["S-T"]["owners"]["X2_seam"]` → `taniteval/tools/seam_probe.py …`
2. `STAGE_GATE_SPEC["S-T"]["criteria"]["X2_seam"]` → **new key** (S-T had a reported probe with no criterion)
3. `STAGE_GATE_SPEC["S-S"]["owners"]["X2_seam"]` → same owner rename

---

### Appendix — what a registry row should carry

`X2_seam | <arm> | T1 | boundary 20 (t=2.0 s) | verdict | worst confirmed row (channel/order) | excess ×scale [lo, hi] | MDE@80% | n windows / n episodes | paired episode-cluster bootstrap | scan FPR`

⛔ and never without the verdict's **power**: a bare "no seam" is the C13 shape this whole document exists to avoid.
