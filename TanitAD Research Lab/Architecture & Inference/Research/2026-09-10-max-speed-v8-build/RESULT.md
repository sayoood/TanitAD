# The v8 `speed_max_input` block — BUILT from the labels we already hold, at full coverage

**2026-09-10 · Architecture & Inference · `agent/arch-inf-20260803`**

⛔ **TIER: NONE, and that is not a dodge.** No capability is claimed here. Every number
below is a **census over a label blob**, a **cross-check against an independently
authored reference**, or a **mutation proof on a refusal path**. T0 is a world-model
diagnostic and T1 is self-action open loop; neither describes counting fields in a
JSONL. Stamping this T0/T1 would be a category error. The arm that *would* carry a
tier is pre-registered and **not run** — there is no pod.

---

## THE ANSWER IN ONE LINE

> **BUILT, at 4,572/4,572 train (100 %) and 147/147 eval (100 %)** — a pure transform
> of `g_tac.goals.SPEED_BAND`, no sign detector, no new labels, no re-selection of
> clips. The 20-step smoke **fed 112/112 windows** (not the `0/2` case), the loader's
> refusal and the new config-stamp refusal are each **proven by mutation**, and the
> ladder is verified against an **exact-rational ladder authored from the km/h
> integers** that re-derives the published 57.5 % rounding defect without being told
> it.

⭐ **This is the PI's ruling executed, and it CLEARS the bar that killed the previous
supplier.** The sign-reading supplier covered 31/4,572 train and **0/147 eval** — with
zero eval coverage the ON and OFF arms receive identical input on every eval window and
the lever is unmeasurable. This one is at **100 % on both**.

---

## 1 · ⭐ THE LADDER — and which field is the ceiling, which the floor

**The ladder is PINNED and this build did not re-pin it**
(`stack/tanitad/refs/max_speed_input.py:239`). VALUES come from road law; MEMBERSHIP
came from the corpus; ⛔ **no step is a quantile.**

| km/h | 20 | 30 | 50 | 70 | 80 | 100 | 120 | 130 |
|---|---|---|---|---|---|---|---|---|
| **m/s** (`k/3.6`, **full precision**) | 5.555555555555555 | 8.333333333333334 | 13.88888888888889 | 19.444444444444443 | 22.22222222222222 | 27.77777777777778 | 33.333333333333336 | 36.11111111111111 |

* ⭐ **CEILING = `SPEED_BAND.v_hi_ms`, snapped UP** — `q = min{s : s ≥ v}`. That is what
  a posted limit *is*: the smallest real sign that admits the speed observed. It is
  the **only** value the model-facing block encodes. Above the top step there is no
  larger real sign, so the value is the top step and `over_ceiling` is True —
  **a real limit can be exceeded**; the raw ego-derived value cannot be, because it
  *is* the max. **MEASURED: 28/4,572 train clips (0.61 %) sit over the ceiling**,
  reproducing the module's own pinned figure.
* ⭐ **FLOOR = `SPEED_BAND.v_lo_ms`, snapped DOWN** — `f = max{s : s ≤ v}`. This is the
  PI's *"logic of minimal speed"*: the band's own minimum, on the same ladder, in the
  mirror direction. Below the bottom step there is no smaller real sign, so the floor
  is the bottom step and `under_floor` is True (**1,649/4,572 train, 50/147 eval**).

⚠️ **ALREADY SETTLED, NOT RE-DERIVED — the "30 km/h floor" premise FAILS.** The
ladder's bottom step is **20 km/h**, and `quantize_up` never returns below its first
step, so **every clip receives a ceiling and `n_null_ceiling` is 0** — a fully stopped
ego (`v_hi = 0.00 m/s`) snaps to 20 km/h. A synthetic 30 km/h floor was never needed.

⚠️⚠️ **THE FLOOR IS RECORDED, NOT WIRED, AND THAT IS DELIBERATE.** The model-facing
block is `MAX_SPEED_DIMS = 3` = `(value_norm, over_ceiling, valid)` — there is no floor
slot. Widening it would change the seam, break the parameter pin in
`test_max_speed_wiring.py`, and make refcv6 a **multi-lever** arm, which is exactly what
a single-lever pre-registration forbids. So the floor ships in the *record* (where it
stratifies a result and documents the band) and reaches the *model* not at all. Pinned
by `test_the_FLOOR_IS_RECORDED_BUT_NOT_WIRED_so_refcv6_stays_single_lever`.

### 1.1 The shipped distribution

**MEASURED (ours)**, `raw/build_census.json`:

| bucket (km/h) | 20 | 30 | 50 | 70 | 80 | 100 | 120 | 130 |
|---|---|---|---|---|---|---|---|---|
| **ceiling**, train | 809 | 932 | **1593** | 620 | 165 | 229 | 138 | 86 |
| **floor**, train | 2451 | 1134 | 465 | 141 | 194 | 117 | 47 | 23 |
| **ceiling**, eval | 23 | 26 | 52 | 25 | 5 | 9 | 6 | 1 |

⛔ **The 38.1 % at ≤ 30 km/h is the channel's known residual defect, registered in
advance** (`PREREG …:§4`): snapping UP from a stopped ego reports the LOWEST limit
where a map would say 50, so the channel **can teach "slow ego ⇒ low limit"**. That is
why the arm's SHUFFLE and WITHHOLD controls are mandatory.

---

## 2 · ⛔⛔ WHAT THIS CHANNEL IS — declared, three times, mechanically

`speed_max_input.v_max_ms` **is** `g_tac.goals.SPEED_BAND.v_hi_ms` = **max of the
ego's OWN realised speed over `[t0+2 s, +6 s]`** (`s2_geom_emit_v7.py:309-311`,
`v.max()` over the band). Provenance **`ego-future`**. Quantization does not launder
it — bin + `v0` recovers **R² 0.9702** of the raw ego future.

⭐ **The PI has decided to use it, and that decision is CONSISTENT with the programme's
existing position rather than a new exception.** refcv5-v2 already feeds an oracle nav
command and declares `nav_cmd_derivation: "v7.2 nav_command token (oracle, provenance
ego-future; allow_oracle_nav=True)"`. The standing rule is that the nav command is an
**INPUT simulating the vehicle's nav system, not a training signal**; max speed is the
same class, standing in for a **speed-limit service**.

⛔ **THEREFORE THE REQUIREMENT IS DECLARATION, NOT REFUSAL — and it is enforced in
three places, none of them a comment:**

| # | mechanism | site |
|---|---|---|
| 1 | every record carries `oracle: true`, `provenance: "ego-future"`, and a `derivation` naming the source field and window | `build_v8_speed_max_labels.build_block` |
| 2 | the blob is readable **only** through the oracle gate | `v7_labels.oracle_max_speed` |
| 3 | ⭐ `config.json` carries **`speed_max_derivation`** and the run **REFUSES TO START** without it | `refc_v3_train._assert_speed_max_stamp` (NEW) |

**The stamp, verbatim, read from the smoke run's own `config.json`:**

```
v7.2 g_tac.goals.SPEED_BAND.v_hi_ms (oracle, provenance ego-future: max of the ego's
OWN REALISED speed over [t0+2 s, +6 s]; floor from .v_lo_ms; allow_oracle_nav=True).
INPUT standing in for a speed-limit service, never a training signal.
```

⛔ **NO CAPABILITY CLAIM MAY BE CREDITED TO THIS CHANNEL WITHOUT THAT STRING BESIDE
IT.** Written into the pre-registration as an admissibility condition, not a courtesy.

---

## 3 · COVERAGE — both releases, with same-breath positive controls

**MEASURED (ours)**, `raw/build_census.json` + `raw/verify_ladder_independent.json`:

| | train | eval |
|---|---|---|
| records | 4,572 | 147 |
| **records with a `speed_max_input` block** | **4,572 (100.000 %)** | **147 (100.000 %)** |
| records declaring `units` | 4,572 | 147 |
| records with `units` MISSING | **0** | **0** |
| `n_null_ceiling` | **0** | **0** |
| over the 130 km/h top step | 28 (0.61 %) | 0 |
| under the 20 km/h bottom step | 1,649 | 50 |
| ⭐ **CONTROL** — `clip_id` present (must be > 0) | **4,572** | **147** |
| ⭐ **CONTROL** — `SPEED_BAND` present (must be > 0) | **4,572** | **147** |

⛔ **The zeros above are claims about the CONTENT, not about the probe**, because the
two same-breath controls read 4,572/4,572 and 147/147 through the same loader in the
same pass. A blob that could not be opened would have failed both.

⭐ **THE COVERAGE BAR IS CLEARED ON BOTH RELEASES**, which is precisely what the
previous supplier could not do (31/4,572 train, **0/147 eval** — an unevaluable arm).

---

## 4 · ⛔⛔ THE LADDER CROSS-CHECK — independently authored, and its mutation goes RED

`code/verify_ladder_independent.py` **imports nothing from `max_speed_input`**. It
types the ladder from **road law as km/h integers**, converts with the exact rational
`Fraction(18, 5)` over the JSON token's own decimal text, and compares **in km/h
space** while the builder snaps **in float m/s space**. Two spaces, two authors.

> ⛔ *A cross-check must be derived independently of the value it checks. Re-running
> the producer's own derivation and finding agreement measures **determinism, not
> correctness**.*

| | train | eval |
|---|---|---|
| ceiling bucket matches the exact ladder | **4,572 / 4,572** | **147 / 147** |
| floor bucket matches | **4,572 / 4,572** | **147 / 147** |
| `over_ceiling` / `under_floor` flags match | 4,572 / 4,572 | 147 / 147 |
| mismatches | **0** | **0** |
| records carrying an m/s bucket (**must be 0**) | **0** | **0** |

### 4.1 ⭐⭐ The regression arm re-derives the published defect without being told it

The historical failure (MEASURED 2026-09-07, recorded in `enable_max_speed`'s own
docstring) was **not** a bad snap — it was **quantizing twice**: ship
`v_max_bucket_ms` rounded to 4 dp, then re-snap the shipped bucket.
`round(50/3.6, 4) = 13.8889` is strictly **greater** than the true step
`13.888888…`, so it climbs a rung.

```
MUTANT2_double_quantize_moved       2631 / 4572  =  57.5459 %
MUTANT2_moves                       50->70: 1593 · 20->30: 809 · 100->120: 229
```

⭐ **That reproduces the published `2,631 / 4,572 (57.5 %)` and its exact three-way
breakdown, from an instrument that shares no code with the producer.** Only the three
steps whose 4-dp rounding lands *above* the true value move — 20, 50, 100 — and the
other five do not, which is the mechanism, not a coincidence.

⇒ ⛔ **This build emits the bucket ONLY as an INTEGER km/h.** There is no m/s bucket to
round, so the defect has no surface to live on, and the consumer's own content
assertion (`round(quantized_ms * 3.6) == v_max_bucket_kmh`,
`refc_v3_train.py:1574-1584`) compares an integer with an integer.

### 4.2 ⚠️ The honest half — one regression arm is INERT, and it is reported as inert

`MUTANT1` (snap the **raw** value against a 4-dp-rounded ladder) moved **0** clips on
both splits. It is **not** a passed mutation and is not counted as one: the sliver it
could move is ~1.2e-5 m/s wide, and `v_hi_ms` is emitted at 2 decimal places
(`s2_geom_emit_v7.py:311`), so nothing can land in it. It is kept because it
**localises** the defect — the failure was never in the snap. ⛔ The verdict rides on
MUTANT2 alone; a gate resting on MUTANT1 would be green forever.

---

## 5 · ⛔ THE REFUSALS — both proven by mutation, neither weakened

`code/smoke_and_mutations.py`, `raw/smoke_and_mutations.json`. **Four arms, all PASS.**
⛔ Every judgement reads a **file**; none reads an exit code.

| arm | what it does | verdict |
|---|---|---|
| **A — ON** | real v8 labels, 20 steps | **PASS** |
| **B — OFF** | the single-lever twin | **PASS** |
| **M1 — block removed** | `speed_max_input` stripped from all **4,572** records | **PASS (refused)** |
| **M2 — stamp removed** | `speed_max_derivation` deleted from the config dict | **PASS (refused)** |

**M1** — the loader's guard at `refc_v3_train.py:1596`, **untouched**:

> `[v3] ⛔ --max-speed-input: NOT ONE of this split's 112 windows receives a
> speed_max_input value. … Refusing rather than feeding nothing.`

⛔ **The guard was not "fixed".** It passes on the real build because real data is
supplied. Its teeth were re-proven by removing the data, not by removing the guard.

**M2** — the NEW guard, `_assert_speed_max_stamp`:

> `[v3] ⛔ --max-speed-input is ON but this run's config.json would carry no
> speed_max_derivation. … Refusing to start rather than banking an arm whose record
> cannot say what fed it.`

Both refusals wrote **no `config.json` and no `summary.json`** — the artifact is the
evidence, not the status code.

### 5.1 ⭐ The mutation that nearly proved nothing, and the check that caught it

⚠️ **MEASURED, and it is the sharpest thing in this package.** M2's first version
wrote the mutant trainer to `%TEMP%`. It died at `import refb_labels`
(`refc_v3_train.py:74` resolves siblings relative to its own path) — **before reaching
the guard**. It exited non-zero and wrote no `config.json`, so a refusal check testing
only *"rc != 0 and no artifact"* would have scored it **PASS and certified a guard that
had never run.** Only the **message assertion** separated *"refused"* from *"crashed on
the way to the refusal"*. The mutant now runs beside the real trainer and is deleted in
a `finally`; `ModuleNotFoundError` count in its log is **0**.

---

## 6 · THE 20-STEP SMOKE — asserted on content, not on "it ran"

```
[v3] v7.2 labels: 4572 records, joined 4/4 episodes = 100.0 %
[v3] max_speed_input (quantized): 4572/4572 clips carry a ceiling,
     112/112 windows fed, 28 over the 130 km/h top step (md5=fa89ea55dfce68403eb30300e57852ab)
```

⭐ **112/112 windows fed.** ⛔ This is the assertion that separates the run from the
`0/2 clips carry a ceiling, 0/2 windows fed` mutant, which **trained happily** — the
`tac_goal` zero-gradient failure exactly. A run can exit 0, write every artifact, and
have fed nothing.

| assertion | read from | value |
|---|---|---|
| windows fed > 0 | trainer census | **112 / 112** |
| loss rows | `metrics.jsonl` | **20** |
| loss all finite | `metrics.jsonl` | **yes** |
| ⛔ loss **NOT constant** | `metrics.jsonl` | **20 distinct values**, 118.4463 … 169.5285 |
| `config.json.max_speed_input` | artifact | `true` |
| `config.json.speed_max_derivation` | artifact | present, all 4 required tokens |
| `max_speed_stats.train.window_ceiling_frac` | artifact | **1.0** |
| OFF twin carries **no** stamp | artifact | `null` |

### 6.1 ⛔ THE GAP THE SMOKE COULD NOT SEE — the EVAL split, closed separately

⚠️ **The 20-step smoke uses `--synth-episodes` and therefore has no `--eval-cache`,
so `e_ds.enable_max_speed(...)` is NEVER CALLED by it.** A green smoke and an
untested eval path — the shape of defect this programme keeps paying for: *a thing
that is built and stamped and never actually exercised.* ⛔ And the eval split is the
half that decides whether the arm is **evaluable at all**: the previous supplier died
exactly there (31/4,572 train but **0/147 eval**), which a train-only check cannot see.

⇒ `code/verify_eval_split_feeds.py` drives the trainer's **real**
`V3Dataset.enable_max_speed` — unbound, over the same `SimpleNamespace` rig
`test_max_speed_wiring.py` uses — against the **real v8 blobs** through the **real**
`load_v7_labels`. Nothing is stubbed but the frame store.

| | eval | train |
|---|---|---|
| clips | 147 | 4,572 |
| windows | 147 | 4,572 |
| ⭐ **windows receiving a ceiling** | **147** | **4,572** |
| `window_ceiling_frac` | **1.0** | **1.0** |
| clips over the top step | 0 | 28 |
| shipped value range (RAW m/s) | 0.0 … 35.54 | 0.0 … **37.80** |

⭐ **37.80 m/s = 136.1 km/h reproduces the module docstring's independently recorded
corpus maximum** (*"the corpus's maximum `v_hi` is 37.803 m/s = 136.1 km/h"*) — a
second, unplanned cross-check against a number this build never consulted.

⛔ **The values are shipped RAW, asserted explicitly** (`values_are_RAW_not_on_the_ladder`):
if the loader ever started quantizing, the ladder would be applied twice — the exact
double-quantization defect §4.1 reproduces. ⭐ A **vacuity control**
(`CONTROL_more_than_100_windows`) guards against a probe over an empty index, which
would satisfy several of the other checks and mean nothing.

---

## 7 · ⭐ THE BLOCKER THIS TURN REMOVED (RULE ZERO — it was not in the brief)

⛔ **`--synth-episodes` + `--v7-labels` died on a bare
`ValueError: invalid literal for int() with base 10: 'synth-000'`.** The synthetic
corpus stamped a **string** `episode_id` while the label join is
`{stable_episode_id(clip_id): label}` looked up as `int(e.episode_id)`. ⇒ **No CI or
local smoke of ANY v7-label channel — nav, tac-goal, nav-args, max-speed — was
possible**, and the crash read like a broken flag rather than a corpus that
structurally cannot carry a label.

⭐ **Independent corroboration that this belonged in production:**
`test_refc_v3_nav_from_v7.py:325` had already been forced to hand-roll the exact same
capability as a monkeypatch shim, because the production code could not do it.

**Fixed, additively:** `_synth_episodes(..., clip_ids=...)` stamps
`stable_episode_id(clip_id)`; `train()` supplies the first N clip ids when
`--v7-labels` is given; `config.json` records `synth_clip_ids_from_labels` so such a
run can never be mistaken for a trained arm; the join now **refuses by name** instead
of raising `ValueError`; and it **refuses to reuse a clip id** (two episodes sharing
one stable id would collapse into one label while the join count still read 100 %).
⛔ Default `None` keeps the historical string id **exactly**, so no existing path moved.
⚠️ **The frames stay synthetic** — this makes the JOIN real, not the perception, and a
test pins that misreading shut.

### 7.1 ⛔⛔ I BROKE THREE TESTS DOING IT, AND ONLY ONE CHECK FOUND THEM

⚠️ **MEASURED 2026-09-10, and it is the sharpest process lesson in this package.** My
first version passed `clip_ids=synth_clip_ids_from_labels` **unconditionally**. Three
tests in `test_refc_v3_save_before_eval.py` monkeypatch `_synth_episodes` with a shim
of the OLD signature, so they died on
`TypeError: _train_eps() got an unexpected keyword argument 'clip_ids'`.
**The shims were right; my call site was wrong.**

⭐ **How they were nearly missed, and what actually caught them.** The full suite reported
**74 failures**, in 19 files, none obviously related. The reasoning that would have
dismissed them — *"these are pre-existing, they're in files about loss determinism and
git trees"* — was **true of 71 of them and false of 3**. Three independent lines of
evidence all pointed the right way and none of them was sufficient:

* **positional** — the same failure block at the same offset in a pristine-HEAD run;
* **direct** — the same files failing identically at HEAD;
* ⭐ **static — "does this failing file reference anything I changed?"**, run over all
  19 with a discriminating control (my own files must read non-zero: 58 / 7 / 21).

⛔ **Only the static check flagged the two files that mattered**, and the head-to-head
on exactly those two then read **WT 8 failed / PR 5 failed** — a real regression, in a
haystack of 74 pre-existing failures. After the fix: **5 / 10 in BOTH trees, identical
test names.**

⇒ **The fix is at the CALL SITE, not in the shims:** the kwarg is passed only when it
carries something, so a new optional argument never reaches a caller that did not ask
for it. That closes the whole class instead of the two instances that happened to be
found. Pinned by
`test_a_two_arg_shim_for_synth_episodes_STILL_WORKS_no_kwarg_leaks`, which asserts on
the trainer's **source at the call site** (a test that only called `_synth_episodes`
directly would pass either way and pin nothing) and carries a same-breath control
proving the guard is not vacuous.

### 7.2 ⛔⛔ RETRACTED IN THE SAME TURN: "the other 71 failures reproduce at pristine HEAD"

⚠️ **I wrote that, and I had not established it.** Two defects in my own instrumentation,
both of the class this file keeps documenting:

1. ⛔ **MY COMPARISON ARTIFACT WAS TRUNCATED AND READ LIKE A COMPLETE ONE.** The
   working-tree side of the nine-file comparison ran as `pytest … -rf 2>&1 | tail -25`,
   so its `FAILED` list was **cut to 24 lines while its own tally said 41 failed**. The
   pristine side was not piped and captured **63 of 63**. The per-file table I then built
   compared **a truncated list against a complete one**, and it "showed" pristine failing
   more in six files — an artifact of my own pipe. ⭐ **The discriminating control was
   free and internal: the tally line (41) disagreeing with the line count (24).** Same
   family as `raw/NOISE_FLOOR.md`, which crashed mid-write on a cp1252 error and reads
   like a finished document.
2. ⛔ **THE PRISTINE TREE WAS NOT A CLEAN BASELINE.** It was built with
   `git archive HEAD stack taniteval tools`, so every test needing a repo-root path
   (`Project Steering/`, research packages, data files) **errors or skips there and only
   there** — 10 errors and 5 skips the working tree does not have. Its aggregate counts
   are therefore **not comparable** to the working tree's at all.

⇒ **WHAT ACTUALLY STANDS, and it is enough:**

* ⭐ **The only failing files that reference anything I changed are the two I ran
  head-to-head**, with complete output on both sides and the files run in the same
  invocation: after the fix, **5 failed / 10 passed in BOTH trees, identical test
  names.** That is like-for-like and unaffected by either defect above.
* ⭐ **The static check stands**: 0 of the other 17 failing files reference
  `max_speed` / `speed_max` / `_synth_episodes` / `synth_clip_ids` / `refc_v3_train`,
  with a discriminating control reading **58 / 7 / 21** on the files that must.
* ⭐⭐ **Within one tree the counts are not merely stable, they are INVARIANT TO
  INVOCATION COMPOSITION** — which eliminates the alternative explanation I had
  entertained (*"these tests are flaky or order-dependent"*) and leaves the truncation
  as the whole cause. MEASURED, three different invocations:

  | invocation | `test_ema_tau_ramp` | `test_drift_levers` | `test_grad_budget_honesty` | `test_o1_detach_encoder` | `test_mktree_commit` |
  |---|---|---|---|---|---|
  | those 3 files alone (×3 consecutive runs) | **3** | **3** | **1** | — | — |
  | 5 files together | **3** | **3** | **1** | **4** | **6** |
  | the full 7,409-test suite | **3** | **3** | **1** | **4** | **6** |

  ⇒ per-file comparison is a **sound method**; it was my *artifact* that was unsound,
  not the approach. ⭐ And the 5-file run was checked with the control I should have run
  the first time: **tally 17 failed = 17 `FAILED` rows present.**

⛔ **What I do NOT claim:** that all 71 remaining failures were verified to reproduce at
HEAD. They were not. The admissible statement is the static one — **none of them touches
anything this stream changed** — and that is what any reader should quote.

---

## 8 · WHAT THIS CHANGES IN THE REGISTER

* ⭐ **`D-VOCAB-REACH-4` is DISCHARGED for the label half.** It said the channel needs
  the v8 label release. The release now exists, at 100 % on both splits, in the repo.
* ⭐ **The 2026-09-10 census's §6.1 finding — "no builder is pre-registered, and that
  is the finding" — is SUPERSEDED BY A PI RULING, not by an argument.** The census was
  right that the *sign* supplier is unevaluable (0/147 eval). The PI chose the
  SPEED_BAND supplier instead and required declaration. Nothing in the census is
  retracted; its verdict on route 1 stands.
* ⚠️ **The `ChannelExclusion` in `max_speed_input.py:148` is UNCHANGED and still
  correct.** It excludes `v_max_ms` from an **RL rollout** on this corpus. This build is
  a **supervised INPUT** with a declared train/deploy mismatch — a different question,
  and the exclusion's `permanent=False` still holds.
* ⚠️ **A layer statement, so it cannot rot:** *"PhysicalAI-AV publishes no speed-limit
  feature"* is **TRUE**. *"The programme has no speed-limit ceiling"* is now **FALSE at
  the augmented v8 label layer** — 4,572 + 147 records carry one. What remains true is
  that its **provenance is ego-future, not a sign and not a map**.

---

## 9 · THE ARM — argv for a pre-registered single-lever panel

⛔ **NOT LAUNCHED. There is no pod.** `raw/ARM_ARGV.md` carries the two lines;
`raw/arm_argv.json` is the machine-readable pair.

**BASE** = refcv5-v2's argv **verbatim** (`config.json` md5
`a03a3ebc46691fa70b5fc55c4efb2ef4`, three byte-identical copies), per
`Project Steering/REFCV6_ARM_FACTS.md:99-101`. **ARM** = BASE + `--max-speed-input`.

**MEASURED (ours)**, `raw/assert_single_lever.json` — the parsed-namespace diff:

```
n_shared_keys 109 · n_equal 107
changed  {"max_speed_input": {"base": false, "arm": true}}
EXEMPTED_run_identity_keys  {"out": {...OFF, ...ON}}
CHECK_not_vacuous · CHECK_only_the_expected_key_changed
CHECK_the_lever_actually_moved · CHECK_no_stale_exemptions      all true
```

⭐ **Three of those four checks exist because the obvious one is not enough.**
*"No unexpected differences"* is satisfied by **no differences at all**, so the lever
must be shown to have MOVED; a diff over two nearly-empty namespaces reports "one
difference" and means nothing; and an `--ignore` entry that did not differ would
silently absolve a real lever later. ⛔ `out` is exempted **by name** and printed in
the report — the default ignore-list is empty so every exemption is deliberate.

---

## 10 · ⛔⛔ THE SINGLE-LEVER PRECONDITION NOBODY WOULD THINK TO CHECK

`--max-speed-input` needs the v8 blob. **If the ON arm reads v8 and the OFF arm reads
v7.2, the namespaces differ in THREE keys** — `max_speed_input`, `v7_labels`,
`eval_labels` — and the panel is **not single-lever**. That is the `--v2` conflation
again: *ten levers on two axes, result non-attributable.*

⇒ **BOTH ARMS READ THE v8 BLOB**, and that is sound only because v8 is a **PURE
SUPERSET** of v7.2 — **asserted, not assumed** (`code/verify_v8_is_superset.py`):

| | train | eval |
|---|---|---|
| clip sequence identical (parity) | ✅ | ✅ |
| records adding **exactly** `speed_max_input` | **4,572 / 4,572** | **147 / 147** |
| `_provenance` a superset, old sub-keys byte-equal | 4,572 | 147 |
| `schema_version` + `vocab` **unchanged** | 9,144 / 9,144 | 294 / 294 |
| ⭐ **untouched-key comparisons EQUAL** | **91,440** | **2,940** |
| keys removed / unexpectedly changed | **0** | **0** |
| **CONTROL** — shared keys per record (vacuity guard) | 22 of 22 | 22 of 22 |

⚠️ **It is a DESCENDANT check, not a difference check.** *"v8 differs from v7.2"* is
not the question — of course it does. Whether it differs **only** where intended is.
**A revert also differs.**

⚠️ **CONSEQUENCE, stated plainly:** the E16 OFF twin is **not** bit-identical to the
banked refcv5-v2 run, which points at v7.2. Its `v7_labels` manifest md5 will differ,
so it must be **run**, not borrowed.

---

## DELIVERABLE MANIFEST

| artifact | where | only one place? |
|---|---|---|
| `RESULT.md` — this verdict | `repo:<pkg>/` | **staged, uncommitted** |
| `code/verify_ladder_independent.py` — exact-rational ladder + 2 regression arms | `repo:<pkg>/code/` | **staged, uncommitted** |
| `code/smoke_and_mutations.py` — the 4-arm smoke, asserted on artifacts | `repo:<pkg>/code/` | **staged, uncommitted** |
| ⭐ `code/verify_v8_is_superset.py` — the single-lever precondition (§10) | `repo:<pkg>/code/` | **staged, uncommitted** |
| ⭐ `code/assert_single_lever.py` — parsed-namespace diff with 4 checks | `repo:<pkg>/code/` | **staged, uncommitted** |
| ⭐ `code/verify_eval_split_feeds.py` — the EVAL split the smoke cannot reach (§6.1) | `repo:<pkg>/code/` | **staged, uncommitted** |
| `raw/verify_eval_split_feeds.{json,txt}` | `repo:<pkg>/raw/` | **staged, uncommitted** |
| `raw/ARM_ARGV.md` + `raw/arm_argv.json` — the two lines, machine-readable | `repo:<pkg>/raw/` | **staged, uncommitted** |
| `raw/build_census.{json,txt}`, `raw/verify_ladder_independent.{json,txt}`, `raw/verify_v8_is_superset.{json,txt}`, `raw/smoke_and_mutations.{json,txt}`, `raw/assert_single_lever.{json,txt}` | `repo:<pkg>/raw/` | **staged, uncommitted** |
| ⭐ `stack/scripts/build_v8_speed_max_labels.py` — THE BUILDER | `repo:stack/scripts/` | **staged, uncommitted** |
| ⭐ `stack/tests/test_speed_max_derivation_stamp.py` — 42 tests | `repo:stack/tests/` | **staged, uncommitted** |
| `stack/scripts/refc_v3_train.py` — stamp + guard + synth-join fix | `repo:stack/scripts/` | **staged, uncommitted** |
| `stack/tests/test_refc_v3_nav_from_v7.py` — shim tolerates the new kwarg | `repo:stack/tests/` | **staged, uncommitted** |
| ⭐ **the v8 label release**, both splits | `repo:TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-10-v8-speed-max-label-release/raw/` | **staged, uncommitted** |
| `Project Steering/PREREG_E16_MAX_SPEED_INPUT_REFCV6.md` | `repo:Project Steering/` | **staged, uncommitted** |

`<pkg>` = `TanitAD Research Lab/Architecture & Inference/Research/2026-09-10-max-speed-v8-build/`

⛔ **Nothing here lives only on a pod or only in a worktree.** The M2 mutant trainer was
written beside the real one and deleted in a `finally`; its absence is asserted.

## INTEGRATION ITEMS (escalated, not filed in a README)

1. ⛔ **The pre-registration needs a PI go and a GPU.** The arm is ready; nothing else
   in it is blocked.
2. ⚠️ **`GOALS_AND_CLAIMS.md` needs `H-E16-1` registered** with the declaration
   condition from §2 attached, so no later reader can quote a result without it.
3. ⚠️ **`WIRING_DIFF.md` from the 2026-09-06 package is STILL UNAPPLIED and still has
   no owner.** ⭐ This build **raises** its priority — the census had lowered it on the
   grounds that there was no admissible value to feed through that seam. There now is
   a value, at full coverage.
