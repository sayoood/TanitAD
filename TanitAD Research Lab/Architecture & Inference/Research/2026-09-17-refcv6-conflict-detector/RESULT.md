# refcv6 §6 — the gradient-conflict detector: built, controlled, and MEASURED to be half-blind

**2026-09-17 · Architecture & Inference · dev box, CPU only · no GPU-minute, no download, no pod hour**

⚠️ **Base commit `e95af00`.** The branch tip advanced to `24065b6` (tactical
wiring, flywheel negatives, the 3-D agent join) while this was being built, and
`stack/scripts/refc_v3_train.py` is touched by BOTH. MEASURED: the three-way
merge of that file onto `24065b6` is **CLEAN** — `git merge-file` exit 0, zero
conflict markers — because their hunks are at base lines 1534 / 2111 / 4722 /
4772-4780 / 5534 and none of them is a region this work edits. No other file
here is touched by either side.

---

## ⛔ HEADLINE — TWO DECISIONS FOR THE PI

**1. The pre-registered statistic CANNOT fire on the pre-registered mutation, and
this is algebra, not an implementation defect.**
`PREREG_BEV_CAPACITY_COMPETITION.md` §4 asks for `conflict(t) = cos(g_traj, g_aux)`
and then demands *"a MUTATION arm that must go RED: … scale the aux loss by 30× …
and the conflict detector must fire."* **A cosine is invariant under positive
scaling.** Scaling `g_aux` by 30 is exactly the transformation the cosine
quotients out, so no implementation of `cos(g_traj, g_aux)` can see it.
MEASURED on the real model: a 30× aux moved the cosine by **3.4 × 10⁻⁸**
(relative 5.4 × 10⁻⁷ — float rounding), and a *binary-exact* 32× rescale left it
**bitwise identical**, which proves the residual is rounding and not sensitivity.

⇒ The instrument as shipped **does** fire, because it reports two scale-SENSITIVE
channels beside the cosine, from the same two gradients at no extra backward:
`ratio = |g_aux|/|g_traj|` and `proj = ⟨g_traj,g_aux⟩/|g_traj|²`. Both read
**exactly ×30**. `E-DEC-18`'s measured mechanism was magnitude ("aux ~0.3–0.9
against an objective ~0.03, i.e. 10–30×") — a **magnitude** defect, which is why
an **angle** statistic was always going to miss it.

⭐ **This amends pre-registered criterion `B2`**, which reads *"the median
`conflict(t)` over steps 1–500 of `C2` is < −0.05, and its sign is stable (≥ 80 %
of the first 500 steps negative)"*. §5 below shows that on a 40-step worked
example the **deliberate 30× defect** reaches median cos **−0.0515** (barely past
the −0.05 line) with **75 %** of steps negative (**under** the 80 % line) — i.e.
**B2 as written would have called the known defect REFUTED**, while `ratio`
separated by **21×** and was never ambiguous. ⇒ **PI decision needed:** add a
magnitude arm to B2, or accept that B2 measures direction only.

**2. The prereg's cost estimate was optimistic by roughly a factor of two.**
It budgets *"one extra backward over the trunk — no extra arm, no extra GPU-day."*
MEASURED (§6): the pre-registered `probe` mode costs **+72 % to +104 %** of step
time and the overhead **rises with trunk size**. ⇒ **PI decision needed:**
`--conflict-every N` (overhead ÷ N, but a sparse median is a different
statistic), or `--conflict-mode subtract` (+26 % to +62 %, still bit-identical,
but its planning side is the whole planning objective and not `L_traj`).

Everything else asked for is delivered and green: three analytic controls exact,
per-group reporting, flag ON by default for refcv6 arms, and the training step
**bit-identical with the flag on *as well as* off**.

---

## 1. What was built

| file | what |
|---|---|
| `stack/tanitad/train/grad_conflict.py` | the detector: `cosine_stats`, `GradientConflictDetector`, the three controls, the grouping, the flag |
| `stack/scripts/refc_v3_train.py` | wiring: `--conflict-detector/-mode/-every`, three seams, the step-loop call, `cd_*` in `metrics.jsonl`, the detector block in `config.json` |
| `stack/tests/test_refcv6_grad_conflict.py` | 30 tests — controls, mutation table, groups, bit-identity, guard mutations |
| `stack/tests/test_refcv6_conflict_wiring.py` | 11 tests — the same instrument on the REAL `RefCV3Model` + WP-D BEV head, and the trainer's refusals |
| `.../2026-09-17-refcv6-conflict-detector/code/` | `worked_example.py`, `bench_overhead.py` |
| `.../raw/` | `worked/{1x,30x}/metrics.jsonl`, `worked/summary.json`, `overhead.json` |

**Logged and reported, never a stop rule.** Nothing in the module halts, clips,
re-weights or projects a gradient. The single refusal is at *start*: the probe
will not run if its own controls miss (prereg §4, *"THE PROBE REFUSES TO RUN IF
ANY MISSES"*), implemented as `self_check()` and called once at the first
supervised step.

---

## 2. The three analytic controls — EXACT, not to a tolerance

MEASURED on the real `RefCV3Model` trunk (`core.encoder.`, 36 tensors /
77,528 params) with the real trajectory loss and the real WP-D BEV aux head
(`raw/worked/*/metrics.jsonl`, first line of each):

| control | analytic target | READ | evidence |
|---|---|---|---|
| `cos(g_traj, g_traj)` | **exactly +1.0** | **`1.0`** | MEASURED, `==`, 12/12 seeds |
| `cos(g_traj, −g_traj)` | **exactly −1.0** | **`-1.0`** | MEASURED, `==`, 12/12 seeds |
| detached aux — reported conflict | **exactly 0.0** | **`0.0`** | MEASURED |
| detached aux — raw quotient | **NaN** (0/0, never a smoothed 0) | **`NaN`** | MEASURED |
| detached aux — `\|g_aux\|` on θ_trunk | **exactly 0.0** | **`0.0`** | MEASURED |

Three things make that exactness real rather than lucky:

* **The denominator is `sqrt(a*b)`, never `sqrt(a)*sqrt(b)`.** For the self
  control both sums are the same float `S`; IEEE-754 `sqrt` is correctly rounded,
  so `sqrt(S*S) == S` exactly. Written the other way the quotient misses `1.0`
  for **46.8 %** of values (MEASURED: 93,641 of 200,000 uniform draws), and
  `test_mutation_lossy_denominator_is_CAUGHT` re-introduces that spelling over 12
  seeds and watches the `+1` control refuse it.
* **No epsilon anywhere.** A zero-norm side yields `NaN` + `degenerate`, so
  "orthogonal" and "absent" can never collide. The reported `conflict` is `0.0`
  **only** when `degenerate` is set *and* `norm_aux == 0.0`, both logged beside it
  — a 0.0 in the log always carries its receipt. ⚠️ This reconciles the prereg's
  two clauses ("must read exactly 0.0" *and* "report NaN → refuse, never 0 by
  accident") by reporting **both numbers**; flagged here rather than resolved
  silently in either direction.
* **The detached control is checked, not trusted.** `controls()` builds it from a
  probe scalar outside the trunk; the tests repeat it against a model whose aux
  head genuinely reads `feat.detach()`, and get the same three readings.

⭐ **A fourth control nobody has to configure.** The `heads` group reads
`cos = 0.0` with **both** norms non-zero (`|g_traj| = 78.008`, `|g_aux| = 6.074`,
`dot = 0.0` exactly): the planning and perception losses touch disjoint head
parameters, so their dot product there is structurally zero. It is a standing
check that the two losses are the two losses we think they are.

⚠️ MEASURED and worth recording: the identities hold in **float32 as well as
float64** (12/12 seeds each), because the denominator makes them structural
rather than precision-dependent. `float64` buys precision in the per-step
reading, not the controls.

---

## 3. ⛔ THE 30× MUTATION — the table

Real model, **identical parameters**, aux scaled in both the detector's copy and
the loss the arm trains on:

| channel | 1× | 30× | 32× (binary-exact) | verdict |
|---|---|---|---|---|
| `cos` | −0.063267506 | −0.063267472 | **−0.063267506 (bitwise equal to 1×)** | ⛔ **BLIND** |
| `ratio` = \|g_aux\|/\|g_traj\| | 0.14436 | **4.33064** | 4.61935 | ✅ **×30.0000002** |
| `proj` = ⟨g_traj,g_aux⟩/\|g_traj\|² | −0.009133 | **−0.273988** | −0.292254 | ✅ **×29.99998** |
| \|g_aux\| | 0.8143 | **24.4286** | 26.0564 | ✅ ×30 |

and on the two-head toy, where the 30× arm crosses the line that matters:

| channel | 1× | 30× |
|---|---|---|
| `ratio` | 2.0659 | **61.9758** |
| `proj` | −0.04642 | **−1.39255** ← *the aux gradient now more than cancels the planner's own descent direction* |

`proj ≤ −1` is the interpretable red line: at 1× the aux erodes 4.6 % of the
planning gradient along its own direction; at 30× it reverses it. That is the
reading the detector exists to show, and it is **not** the cosine.

⇒ **The mutation fires on the instrument as shipped. It does not fire on the
pre-registered statistic.** Said loudly, per brief, rather than shipped quietly.

---

## 4. Per parameter group

"The trunk is in conflict" and "the last stage is in conflict" imply different
fixes, so the pooled number is never the only one logged. Groups come from the
backbone's **own** naming, positionally (a substring rule would file half of
every ResNet stage under `stem`), and cover both trunks refcv6 can build:
`RefCModel`/`refc` → `stem` + `stage_stages0..3`; timm ResNet → `stem` +
`stage_layer1..4` + `fuse`.

Real model, step 0, 1× arm (`raw/worked/1x/metrics.jsonl`):

| group | n params | `cos` | `ratio` | `proj` |
|---|---|---|---|---|
| **trunk (pooled)** | 77,528 | **−0.06327** | 0.1444 | −0.00913 |
| `stem` | 408 | −0.06334 | 0.1451 | −0.00919 |
| `stage_stages0` | 1,184 | **−0.08646** | 0.1527 | −0.01320 |
| `stage_stages1` | 3,680 | −0.06555 | 0.1366 | −0.00895 |
| `stage_stages2` | 14,528 | −0.06805 | 0.1404 | −0.00956 |
| `stage_stages3` | 57,728 | **−0.03660** | 0.1458 | −0.00534 |
| `heads` | 46,233 | **0.0** (structural control) | 0.0779 | 0.0 |

The gradient here already reads non-uniformly: the **earliest** stage is in
2.4× more conflict than the **deepest** one. On the real refcv6 trunk that is
precisely the "shrink the head" versus "split the late stages" fork, and the
pooled number alone cannot tell them apart. ⚠️ EVIDENCE CLASS: this is a
**smoke-model** reading on a synthetic corpus — it demonstrates the instrument
resolves a per-stage difference, it says nothing about refcv6's real trunk.

`metrics.jsonl` carries, per step: `cd_cos`, `cd_conflict`, `cd_ratio`,
`cd_proj`, `cd_gn_traj`, `cd_gn_aux`, `cd_degenerate`, `cd_n_params`,
`cd_plan_side`, `cd_ms`, plus `cd_<group>_{cos,ratio,proj,gn_traj,gn_aux}`.
⛔ **Unrounded**, merged after the trainer's own 5-dp rounding pass (the
`_grad_probe_row` convention): a real −1e-8 conflict rounded to 5 dp reads 0.0,
which is exactly the misreading this instrument exists to prevent.

---

## 5. Worked example — a tiny two-head smoke, 40 steps, 1× against 30×

`code/worked_example.py`, real `RefCV3Model` + WP-D BEV head, synthetic corpus,
CPU, SGD(1e-3, momentum 0.9), clip 10.0. Raw: `raw/worked/{1x,30x}/metrics.jsonl`.

| statistic | 1× arm | **30× arm (the mutation)** | separation |
|---|---|---|---|
| `cos` at step 0 (same parameters) | −0.0632675064 | −0.0632674723 | **none (3.4e-8)** |
| `ratio` at step 0 | 0.1444 | **4.3306** | **×30.0** |
| median `cos` over 40 steps | −0.01341 | **−0.05149** | ×3.8 |
| ≥ negative fraction of steps | 55 % | **75 %** | +20 pp |
| median `ratio` | 0.2396 | **5.0759** | **×21.2** |
| median `proj` | −0.00080 | **−0.16630** | **×208** |
| fraction of steps with `proj < −1` | 0 % | **5 %** | — |

Read it in the order the mechanism runs: **at step 0 the cosine is identical**,
because the parameters are identical and the cosine cannot see the scale. It
separates *later*, and only because the 30× aux has dragged the trunk somewhere
else — the cosine is a **lagging** indicator of a scale defect; `ratio` and
`proj` are **immediate** ones.

⛔ **And the calibration that matters for `B2`:** the 30× arm — a *deliberately*
introduced, historically *measured* defect — lands at median cos **−0.0515**
against B2's **< −0.05** threshold and at **75 %** negative against B2's
**≥ 80 %** requirement. **B2 as written would have returned REFUTED on a known
defect.** Its own failure twin then reads *"competition is real but invisible
early — the detector is worthless and the expensive gate stays"*, which would be
the wrong conclusion drawn from the right rule. ⚠️ EVIDENCE CLASS: 40 steps on a
smoke model, not 500 steps on a refcv6 arm — this is a **warning about the
threshold's margin**, not a measurement of `C2`.

---

## 6. MEASURED overhead

`code/bench_overhead.py` → `raw/overhead.json`. Real `RefCV3Model` + BEV aux,
CPU, batch 2, torch 2.11.0, 4 threads. Three trunk sizes × **3 interleaved
replicates** × median of 15 steps after 4 warm-up. ⚠️ The dev box is shared
tonight; three *sequential* passes over the same configuration measured +80 %,
+93 % and +84 %, so the modes are cycled **inside** each replicate and every
replicate is kept in the banked file rather than averaged away.

| trunk params | step, detector off | `probe` (pre-registered) | `reuse` | `subtract` |
|---|---|---|---|---|
| 77,528 | 0.0239 s | **+71.7 %** `[+42.8 … +85.8]` | +26.4 % | +26.3 % |
| 1,573,800 | 0.0385 s | **+83.8 %** `[+79.7 … +89.2]` | +44.2 % | +53.1 % |
| 6,949,968 | 0.1065 s | **+104.2 %** `[+101.2 … +106.1]` | +67.8 % | **+62.2 %** |

**The answer to "cheap enough to run every step?" is NO at face value.** The
pre-registered statistic doubles the step at the largest trunk measured and the
overhead **rises with trunk size**; refcv6's trunk is 21.8 M (spec §9) to ~42 M
(resnet101), so it will be at least that.

* EVIDENCE CLASS: **MEASURED** to 6.95 M params on CPU. **EXTRAPOLATED** above
  that. **UNVERIFIED** on GPU — the arithmetic (a training step is ~⅓ forward,
  ⅔ backward, so 2 extra backwards ≈ +130 %) says the shape carries over, but
  nobody has measured it and I have not spent a GPU-minute to find out.
* The levers, both wired: `--conflict-every N` divides the overhead by `N`
  (⚠️ and changes the statistic — a median over sparse steps is not the
  pre-registered median); `--conflict-mode subtract` costs one backward instead
  of two and is **still bit-identical**, but answers against `L_total − L_aux`
  rather than `L_traj` and stamps `cd_plan_side=total_minus_aux` on every row so
  the two can never be silently compared.

⭐ `subtract` **dominates** `reuse`: same cost, and the training step stays
bit-identical because it *reads* `.grad` and recovers `g_plan = .grad − g_aux` by
linearity instead of writing a re-summed gradient. `reuse` survives only as the
literal reading of the prereg's "one extra backward" accounting.

---

## 7. The flag, and bit-identity

`--conflict-detector auto|on|off`, **`auto` = ON for refcv6 arms** (`refcv6`
block present *and* a live perception weight) **and off otherwise** — the
`refcv6: null` discipline, so "baseline" and "absent" stay distinguishable.

| claim | proof | result |
|---|---|---|
| flag off → **nothing constructed** | `for_model` returns `None`; no module, no parameter list, no RNG draw | MEASURED |
| flag off → **no `metrics.jsonl` key** | the row is `{}`; log schema identical to the pre-detector trainer | MEASURED |
| flag off → **bit-identical** on a fixed seed | 5 steps, SGD + momentum + clip, `torch.equal` on **every** state-dict tensor | MEASURED |
| flag **ON** (`probe`) → **also bit-identical** | same comparison; `autograd.grad` does not accumulate into `.grad` | MEASURED |
| flag ON (`subtract`) → **also bit-identical** | same comparison; it reads `.grad`, never writes it | MEASURED |
| `reuse` → **NOT** bit-identical | `allclose(rtol=1e-5)` but `torch.equal` false — asserted in both directions | MEASURED |
| the comparison **can fail** | a 1.0001× mutation of the loss makes it fail | MEASURED |

### ⭐ And the same claim at the TRAINER level, against the tip's own trainer

`refc_v3_train.py` from **e95af00** (the branch tip, which has no detector at
all) versus this one, same argv, `--seed 0 --smoke --synth-episodes 2 --steps 6`:

| artifact | result |
|---|---|
| `ckpt.pt` — 640 tensors, **393,362 elements** | **0 bitwise different** |
| `metrics.jsonl` | **byte-identical**, and carries **no `cd_*` key** |
| `config.json` | identical except the `--out` path inside `argv`; **no `conflict_detector` block** on a non-refcv6 arm |

⇒ an arm with the detector on and an arm with it off are **the same arm**, and a
non-refcv6 arm is byte-for-byte the pre-detector trainer. That is stronger than
the brief required and is the reason this instrument can never be blamed for a
refcv6 result.

### ⛔ A defect this end-to-end run found, and the refusal it produced

The first smoke run with `--conflict-detector on` on a non-refcv6 arm **built the
detector, printed "ON", stamped `conflict_detector.enabled=true` in
`config.json` — and emitted zero `cd_*` rows for the entire run**, because no
perception term had a live weight and there is therefore no second gradient.
That is the `--w-agent` defect verbatim: a run whose record claims an instrument
it never read. It now **refuses**, before `config.json` is written and before a
batch is loaded (exit 1, no output directory), naming every perception weight it
found: `bev=0.0, map=0.0, box3d=0.0`. ⭐ Two-sided in the test: `auto` must NOT
refuse, or the guard would break every non-refcv6 arm in the repo.

---

## 8. ⛔ Guards proven by MUTATION, not inspection

| guard | the defect re-introduced | caught? |
|---|---|---|
| `+1` control | `sqrt(a)*sqrt(b)` denominator (fails on 46.8 % of values, MEASURED) | ✅ over 12 seeds; the exact spelling passes **12/12**, the lossy one fails on ≥ 1, and `self_check` then **raises** |
| detached control | an un-detached "detached" control (`Tensor.detach` neutered) | ✅ `ok=False`, `norm_aux > 0`, `cos` not NaN |
| bit-identity check | a 1.0001× loss scale | ✅ `AssertionError: … differs` |
| empty/frozen trunk | prefix matching nothing; trunk frozen | ✅ two **distinct** messages |
| loss with no graph | `la.detach()` | ✅ refuses |
| `subtract` before the backward | called with every `.grad is None` | ✅ refuses |
| "on" with no perception weight | the real trainer, real argv | ✅ `SystemExit`, and `auto` still runs (two-sided) |
| 30× invariance argument | a binary-exact 32× rescale | ✅ bitwise equal — the argument is checked, not asserted |

⚠️ **A defect this work found in itself, MEASURED:** the trunk lives at
`encoder.` on `RefCModel` but at **`core.encoder.`** on the trainer's own
`RefCV3Model`. A hard-coded prefix selected **zero** parameters and would have
logged `NaN` for an entire run while `config.json` said the detector was on. The
refusal caught it, and it is now resolved from the model
(`resolve_trunk_prefixes`) with a regression test — this is why that refusal is
not a warning.

---

## 9. What this does NOT show

* **Nothing about refcv6's real trunk.** Every number here is a smoke model on a
  synthetic corpus. The instrument is proven; the readings are not a result.
* **No GPU measurement.** CPU only, per the night's constraint.
* **The map and box losses are not yet in `compute_losses_v3`.** The detector's
  perception side lists `bev`, `map` and `box3d` and sums whatever has a live
  weight; today only `bev` exists, so the `map`/`box3d` arms of the sum are
  **declared and untested**. They are listed deliberately rather than added
  later: the alternative is a detector that silently ignores the head it was
  built for.
* **`C2`, `B1`, `B2` are not run.** This is the instrument, not the panel.

---

## 10. Reproduce

```
cd stack
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD python -m pytest \
    tests/test_refcv6_grad_conflict.py tests/test_refcv6_conflict_wiring.py \
    tests/test_refcv6_no_session_paths.py -q
CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD python \
    "../TanitAD Research Lab/Architecture & Inference/Research/2026-09-17-refcv6-conflict-detector/code/worked_example.py" \
    --out <dir> --steps 40
```

`45 passed` — 30 + 11 + the 4 session-path guards. ⚠️ Run pytest with the repo
root **and** `taniteval/` on `PYTHONPATH`; without them six `test_refcv6_tactical.py`
tests fail on `ModuleNotFoundError: taniteval`, which is a **pre-existing path
issue on this branch, not this work** (79 pass with the path set).
