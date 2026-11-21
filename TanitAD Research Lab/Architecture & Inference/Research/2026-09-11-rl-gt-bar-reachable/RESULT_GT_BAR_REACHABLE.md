# V2's >=GT truncation is now REACHABLE — and the reward audit's `clean` on a gameable reward is fixed

**Architecture & Inference FlyWheel, 2026-09-11 (Europe/Berlin).** Dev-box RTX 4060, no spend.
Continues `…/2026-09-10-rl-pilot-cold-start-allowance/RESULT_P_RC21_2K_ARM.md`.

⛔ **TIER: T0, training-side.** Every run number below is a training-side readout on a NON-PARITY
corpus, **one seed**, **no paired episode-cluster bootstrap**. ⛔ Nothing here is a capability
claim; that requires **T1** (`EVAL_DOCTRINE.md`). ⛔ `overlapping_holdout_se` is not used anywhere.

---

## 1. ⭐ THE FLAG EXISTS, AND THE MASK IS REACHED — proven by mutation

`rl_pilot_refc21.py` now carries **`--gt-bar`**, default **OFF**. The path it opens is:

```
--gt-bar  ->  PostTrainConfig.use_gt_bar=True
          ->  score_gt_bar(spec, ctx, n_steps)        # posttrain.py  (NEW)
          ->  make_refcv3_sample_fn(..., gt_bar_fn=)  # refcv3_adapter.py (NEW param)
          ->  extras["gt_bar"]  ->  rl_objective(gt_bar=)
          ->  composite_advantage(gt_bar=)  ->  advantage.gt_bar_mask   (already existed)
```

⛔ **The existence of a flag is not evidence that it reaches anything** — that is the whole reason
this item exists. An AST census of this same wiring once read **0 suspects on BOTH the fixed and
the broken trainer**. So the proof is a MUTATION, banked in
`raw/MUTATION_PROOF_GTBAR.json`.

### 1.1 The two analytic identities, through the real caller wiring

Neither target needs any knowledge of what the reward computes.

| arm | `theta.grad` | `frac_above_bar` | the identity |
|---|---|---|---|
| bar OFF (control — the instrument must be live) | **+33.523860931396484** | — | non-zero, else nothing below can discriminate |
| bar BELOW every achievable reward | **+33.523860931396484** | **1.0** | identity mask ⇒ **bit-identical** to the no-bar gradient |
| bar ABOVE every achievable reward | **exactly 0.0** | **0.0** | admits nothing ⇒ with `w_intra = 0` the loss loses every dependence on `logp` |

### 1.2 The REAL scorer masks PARTIALLY — a switch would not

Sweeping the logged trajectory's own along-track reach through `score_gt_bar`:

| GT reach (m) | 2.0 | 4.0 | 6.0 | 8.0 | 10.0 | 12.0 |
|---|---|---|---|---|---|---|
| `frac_above_bar` | 1.0 | 0.6667 | 0.6667 | 0.3333 | 0.0 | 0.0 |

⭐ Monotone in the bar, and **strictly between 0 and 1 in the middle** — the mask really zeroes
*some* anchors. A mask that only ever admits all or nothing is a switch, not a truncation.
At a partial bar the inter-anchor advantage contains **both** a zeroed entry and a surviving
positive one (asserted in `test_the_REAL_scorer_ZEROES_SOME_anchors_at_a_bar_in_between`).

### 1.3 The deliberate-regression arm — the emitter removed, and it goes RED

`raw/MUTATION_PROOF_GTBAR.json` writes a byte-copy of `refcv3_adapter.py` with the single line
`extras["gt_bar"] = bar` replaced by `pass`, imports it as its own module, and re-runs all three
checks. ⛔ The real file is never written to (`adapter_untouched: true`, both md5s recorded).

| check | FIXED | MUTANT |
|---|---|---|
| `identity_admit_all_equals_no_bar` | `true` | **`RAISED: ValueError`** |
| `identity_admit_none_is_exactly_zero` | `true` | **`RAISED: ValueError`** |
| `real_scorer_masks_PARTIALLY` | `true` | **`RAISED: ValueError`** |

⭐ `MUTANT_FAILS_ALL_THREE: true`. The `ValueError` is `rl_objective`'s own refusal — *"use_gt_bar
=True but no gt_bar was handed to rl_objective"* — i.e. **the historical defect, reintroduced,
now fails loudly instead of running a V1 arm under a V2 name.**

**And the same mutation applied to the REAL file** (`tanitad/rl/refcv3_adapter.py`, applied,
measured, reverted, md5-verified): `pytest tests/test_rl_gt_bar_reaches_the_caller.py` went
**3 failed / 8 passed**, back to **12 passed** on revert. The test file is provably not vacuous.

### 1.4 Two guards that make the bar mean something

* ⛔ **The bar and the candidates are scored by ONE `RewardSpec` object.** The pilot now passes
  `spec=` into `run_posttrain` and hands the SAME instance to `score_gt_bar`.
  ⚠️ **`test_rl_gt_bar_gradient_path.py::test_run_posttrain_REFUSES_a_spec_that_disagrees_with_the_bars_config`
  was RED on the tip** — the test existed, the guard did not (`grep` for its message: 2 hits, both
  in the test file, 0 in `posttrain.py`). It is implemented now, and the accepting direction is
  asserted too so it is not a blanket refusal.
* ⛔ **The horizon must match.** `score_gt_bar` refuses a GT whose `S` differs from the fan's:
  `_progress` normalises by `(S-1)*dt`, so a bar over a different horizon is a threshold on a
  *different reward function* and looks entirely plausible.
* ⛔ **The pilot refuses to report success on a bar that never applied**: `--gt-bar` with
  `frac_above_bar_mean is None` is a `SystemExit`, and the sampler's own `emits_gt_bar` attribute
  is asserted against the parsed flag before training starts. *Assert on the object, not the argv.*

---

## 2. THE BAR-ON RUN

⛔ **TIER: T0, training-side.** ⛔ **These deltas are NOT a capability claim and NOT a lever
verdict.** One seed, no paired episode-cluster bootstrap, NON-PARITY corpus. They are reported as a
**training-side diagnostic** — what the truncation did to the optimisation — and nothing more.

### 2.1 ⛔ THE CONTROL FIRST — the bar-OFF arm reproduces the banked run EXACTLY

Without this the pair proves nothing: any change elsewhere in the tree could account for the whole
difference. `p1-grpo-2k-repro` re-ran the banked `p1-grpo-2k` with the same seed, steps, batch and
reward, bar OFF, through the **modified** library.

| quantity | banked 2026-09-10 | repro 2026-09-11 | identical |
|---|---|---|---|
| `final_loss` | 0.4329703450202942 | 0.4329703450202942 | **yes** |
| `veto_rate_mean` | 0.14399609375 | 0.14399609375 | **yes** |
| R1 before / after | +0.6427 / +0.6691 | +0.6427 / +0.6691 | **yes** |
| R2 before / after | 21.335 % / 20.924 % | 21.335 % / 20.924 % | **yes** |
| R3 before / after | 0.654 m / 2.1015708355853957 m | 0.654 m / 2.1015708355853957 m | **yes** |

⭐ `raw/COMPARE_ARMS.json` → **`CONTROL_REPRODUCES_EXACTLY: true`**. Default OFF is unchanged.

### 2.2 The pair differs by exactly one key

`differs_by` over the recorded config/outcome fields is
**`['final_loss', 'frac_above_bar_mean', 'use_gt_bar', 'veto_rate_mean']`** — `use_gt_bar` is the
only *setting*; the other three are outcomes of it. `noise_mode`, `w_anchor`, `steps`,
`train_mode_forward` are identical, and the BEFORE readouts are **bit-identical on all three
metrics** (the bar touches training only).

### 2.3 ⭐ The mask was REACHED, and it bit hard

**`frac_above_bar_mean = 0.1505`** — averaged over 2,000 steps, only **15.05 %** of anchors scored
at least the human's own score under the same reward, so the truncation zeroed the positive
inter-anchor advantage on roughly **85 %** of them. `use_gt_bar: true` in the run record.
⇒ This is the field that read **`null`** on every previous P-RC21 arm.

| | bar OFF (control) | bar ON | note |
|---|---|---|---|
| `use_gt_bar` | false | **true** | |
| `frac_above_bar_mean` | **null** | **0.1505** | the mask ran |
| `final_loss` | 0.432970 | **2.728485** | truncating ~85 % of positives changes the objective's scale — not a quality number |
| `veto_rate_mean` | 0.143996 | 0.136008 | |
| ΔR1 fan reward | +0.026417 | **+0.123764** | 4.7× |
| ΔR2 fan collision | **−0.410 pp** | **+0.990 pp** | ⚠️ the bar arm moves the WRONG way here |
| ΔR3 sel-ADE | **+221.18 %** | **+42.26 %** | ⚠️ still a large drift; **5.2× smaller** |

⚠️ **READ ΔR3 CAREFULLY AND DO NOT CALL IT A FIX.** +42.26 % is still far outside the pilot's own
+10 % drift guard. What is measured is that the arm the whole P-RC21 line has been fighting —
*the reward pays for drift* — drifts **5.2× less** with V2's truncation on. ⛔ One seed. ⛔ The R2
column moves the opposite way, and a two-metric split is exactly the shape that a replicate arm
exists to adjudicate. ⇒ **the next experiment is SEEDS, not windows, and not a bigger sweep.**

### 2.4 ⭐ And the P1 arm is now RULABLE — for the first time in the line

`reward_audit_exercising` on both `default` arms: **verdict `clean`, `dead_components: []`**.
The banked run's only audit was `INCONCLUSIVE` on `['collision','headway']`, which made *every* P1
number inadmissible. The empty-context audit is unchanged (`INCONCLUSIVE`) and kept beside it.

### 2.5 The regression arm went RED in the real pipeline, not only in a test

`p2-reg-2k-sep` (`--reward hackable`, 300 steps) now records **`verdict: FLAGGED`** with
`tied_at_max: ['bullet_straight', 'shaky', 'teleport']` on **both** the empty and the exercising
context — the same arm that recorded `clean` on 2026-09-10. ⭐ The deliberate-regression reward is
caught by the run record itself.

---

## 3. ⭐ THE REWARD AUDIT: `clean` on a gameable reward is fixed, and the scope is narrow

### 3.1 What was wrong, stated honestly

P2's reward is `HACKABLE_WEIGHTS` (progress-only), run at the pilot's `dt = 0.5`. The audit
returned **`verdict: clean`, `flagged: False`, `dead_components: []`** while its own scores read:

| probe policy | score |
|---|---|
| `bullet_straight` | **1.5** |
| `teleport` | **1.5** |
| `shaky` | **1.5** |
| `frozen` | 0.0 |
| `spinner` | −0.01041 |
| *(sane reference)* | *1.5* |

⚠️ **The audit was not broken.** Its `verdict` reports component **LIVENESS** — did every weighted
term vary enough to influence the ranking — and nothing was constant (`progress` spans −0.0104 to
1.5), so `clean` was **correct by that definition**. ⭐ **The gap is that liveness is not
gameability**: `progress` is clamped at `hi = 1.5`, so at `dt = 0.5` every policy fast enough
SATURATES and the *ranking* collapses while the *term* still varies. Liveness looks at the term;
separation looks at the ranking the term produces.

⚠️ It is also a **dt-dependent** defect, which is why it survived: at the module default
`dt = 0.1` the same spec is already FLAGGED through the winners path. Only the pilot's `dt` exposes it.

### 3.2 The check

`audit_reward` now computes the panel's own maximum and the set of entries tied at it
(`tie_atol = 1e-9`, named once). **Two or more panel policies at the maximum ⇒ FLAGGED**, with a
distinct `NOT SEPARABLE` reason and a new `tied_at_max` field reported on **every** branch.

⛔ The tie is taken on the **panel's** maximum, not the reference's — and that is the choice that
makes the check safe rather than a blanket refusal. MEASURED at both dt values:

| spec | ctx | panel max | panel entries at it | verdict |
|---|---|---|---|---|
| `default` dt 0.1 | empty | 1.45 | `['bullet_straight']` | INCONCLUSIVE (unchanged) |
| `default` dt 0.5 | empty | 1.45 | `['bullet_straight']` | INCONCLUSIVE (unchanged) |
| `default` dt 0.1 | exercising | 0.349503 | `['spinner']` | **clean** (unchanged) |
| `default` dt 0.5 | exercising | 1.090211 | `['shaky']` | **clean** |
| `hackable` dt 0.1 | empty | 1.5 | `[bullet, teleport]` | FLAGGED (was already) |
| **`hackable` dt 0.5** | empty | **1.5** | **`[bullet, shaky, teleport]`** | **FLAGGED (was `clean`)** |

⚠️ **The boundary is deliberate.** At `dt = 0.5` the default reward's `bullet_straight` scores
**exactly** the reference's 1.45. One degenerate tying the reference is the `margin` policy's
business (*"ties are not failures, strictly beating is"*); this check is about the reward being
unable to rank **two probes** apart, which no margin can express.

### 3.3 Both directions, and an isolation mutation

* **RED on `hackable`** — `test_DELIBERATE_REGRESSION_the_hackable_reward_at_dt_0p5_IS_NOW_FLAGGED`
  asserts `verdict == "FLAGGED"`, `tied_at_max == ["bullet_straight", "shaky", "teleport"]`, and
  ⭐ **`winners == []`** — so the winners path CANNOT be what flagged it, which is what proves the
  separation check is doing the work.
* **Green on `default`** — the existing `test_default_reward_is_not_flagged` still passes, and a
  new test pins `tied_at_max == ['bullet_straight']` at both dt values on the empty context.
* ⭐ **Isolation mutation:** on the programme's one green audit (`default` + exercising ctx),
  widening ONLY `tie_atol` to 1.0 — no reward, no context, no panel change — flips it to FLAGGED
  with `winners == []` and `dead_components == []`. Neither other branch can respond to `tie_atol`.
* **Source mutation:** replacing `if len(tied) >= 2:` with `if False:` in `audit.py` took
  `tests/test_rl_audit.py` from **33 passed** to **2 failed / 31 passed**; reverted and re-verified.

⛔ **Every expectation is a LITERAL**, never an expression over the code under test. The five panel
scores above are pinned as literals in `test_MEASURED_the_hackable_reward_at_the_pilots_dt_TIES_THREE_POLICIES`.

---

## 4. ⭐ AND THE AUDIT CAN NOW RULE ON P1 AT ALL — the thing its own verdict asked for

P1's banked verdict was **INCONCLUSIVE**, with its own reason: *"weighted component(s)
`['collision','headway']` were CONSTANT across the whole panel … supply a context that exercises
them (obstacles / lead_path / gt_traj) before trusting any verdict."*
⇒ ⛔ **no P1 verdict was admissible, with or without the bar.**

`audit.exercising_ctx(n_steps, dt)` builds that context, and ⭐ **every number in it is DERIVED
from the panel, not chosen**:

* `obstacle_x` = the **midpoint** between the sane reference's reach and `bullet_straight`'s —
  beyond the careful policy, inside the reckless one's path;
* `lead_path` offset = `lead_len_m + target_time_gap_s * v_ref` = the standoff `_headway` itself
  is written against. **No new constant, nothing tunable.**

⭐ **The cross-check that makes it trustworthy:** at the module default `dt = 0.1` this derivation
reproduces the hand-found literals **`[[40.0, 0.0]]` and `24.5` EXACTLY** — values found
empirically, by hand, months earlier and living in `tests/test_rl_audit.py::rich_ctx`. An
independently authored derivation landing on the number already in the tree is evidence;
re-running the producer's own arithmetic would have measured determinism.

⚠️ And it **must** scale with `dt`: at `dt = 0.5` the reference reaches **100 m**, so the banked
literal 40.0 would sit UNDER it — which is historical wrong-placement #1 (`frozen` legitimately
wins and the audit flags a reward that is fine). The derived value at `dt = 0.5` is **200.0**.

`run_posttrain` now writes **both** audits into every run record: `reward_audit` (empty context,
byte-unchanged so no banked record changes meaning) and **`reward_audit_exercising`** — the one to
quote. On `default` at `dt = 0.5` it reads **`dead_components: []`, verdict `clean`**.

---

## 5. ⛔ WHAT THIS DOES NOT LICENSE

1. ⛔ **The P1 deltas are still not a result.** One seed, no paired episode-cluster bootstrap, T0.
   ⛔ A separated CI would not be enough either: on this programme's rigs an arm with **zero levers
   moved** has read "separably worse", and a stochastic planner adds a third variance the same
   interval is blind to. The next step for a *claim* is **seeds**, not windows.
2. ⛔ **T0 is not driving performance.** Any capability claim needs T1.
3. ⚠️ **The sampler is still a surrogate.** `refcv3_adapter` samples a Gaussian on the emitted
   offset, not the decoder's true reverse-diffusion density. DDv2 attaches the policy gradient to
   the denoising step distribution itself. Parity with DDv2's estimator remains **UNVERIFIED**.
4. ⚠️ **The separation panel is fixed and finite.** "No two tie" bounds the claim; it does not
   prove a reward is unhackable.
5. ⚠️ **`final_loss` 0.433 → 2.728 is not a quality signal.** Zeroing ~85 % of the positive
   inter-anchor advantage changes what the loss *is*; the two numbers are not comparable.

---

## 5b. ⚠️ THE SAME DEFECT, FOUND IN PASSING, IN V2'S *SECOND* CODE-ONLY INGREDIENT

`D-DDV2-PORT-1` names two mechanisms that landed 2026-09-05: the >=GT bar **and the two-scalar
exploration** (`PostTrainConfig.noise_mode="two_scalar"`, DDv2's released sampler,
`diffusiondrivev2_model_rl.py:646-654`).

**MEASURED 2026-09-11 with a same-breath control:** `grep -c noise_mode rl_pilot_refc21.py` = **0**
while `grep -c PostTrainConfig rl_pilot_refc21.py` = **4** — the file was read, and the token was
genuinely absent. ⇒ the pilot ran, and would have kept running, on the **`multiplicative`** default.

⚠️ **Scope it honestly: `two_scalar` was NOT unreachable from the programme** — `grep -rl` finds it
in `stack/scripts/rl_refcv3_min.py` (the Deployment FlyWheel's driver). It was unreachable from
**this** caller, which is the one the P-RC21 line runs.

`--noise-mode {multiplicative,additive,two_scalar}` now exists, **default `multiplicative`** so
every banked arm reproduces, pinned by
`test_the_pilot_ALSO_EXPOSES_noise_mode_and_it_defaults_to_the_banked_value`.
⛔ **No two-scalar arm was run.** Changing it is a LEVER and needs its own pre-registration;
exposing it makes the arm *possible*, which is all this deliverable claims.

⚠️ **Currency note, stated rather than assumed:** the three arms in §2 were produced by the pilot at
md5 **`18af2c925be35e80fc456412581d1641`**, i.e. BEFORE `--noise-mode` was added. It cannot have
affected them — the flag's default is the value that was previously hardcoded, and each run's own
`config.json` records `noise_mode: "multiplicative"`. The repo copy is re-shipped and md5-verified
in the manifest below.

---

## 5c. ⭐ WHAT STILL BLOCKS THE V2 RL STAGE — and the next arm, named

⭐ **The PLUMBING no longer blocks it.** Both of V2's code-only ingredients are reachable from the
production caller, the bar demonstrably fires (`frac_above_bar_mean` 0.1505), and the arm's reward
audit can rule for the first time (`clean`, `dead_components: []`).

What blocks a **V2 RL RESULT**, in cost order:

1. ⛔ **ONE SEED. This rig cannot distinguish a lever from its own noise floor at n = 1.** MEASURED
   elsewhere in this programme: an arm with **zero levers moved** read "separably worse" on 5 of 9
   family metrics, and a stochastic planner adds a third variance the episode-cluster bootstrap is
   blind to. ⇒ **THE NEXT ARM IS `--gt-bar` AT A SECOND SEED plus a bar-OFF replicate at the same
   seed** — four 2,000-step runs, ~14 min total on the dev-box 4060, **zero spend**. Only then is
   the R3 split (drift 5.2× smaller) or the R2 split (collision 1.4 pp worse) readable as anything.
   ⚠️ Not a bigger window sweep: more episodes narrow an interval that is answering the wrong
   question.
2. ⛔ **T0 is not driving.** Even a clean seeded pair is a training-side diagnostic. The capability
   read is **T1** (`taniteval/tools/t1_eval.py`), four families, paired episode-cluster bootstrap.
3. ⚠️ **The sampler is still the surrogate** (`H-DDA-3`): a Gaussian on the emitted offset, not the
   decoder's reverse-diffusion density. `--noise-mode two_scalar` now makes the *released* DDv2
   sampler reachable, which is the cheap half; the per-step chain objective is the expensive half
   and needs a retrained base.
4. ⚠️ **`--noise-mode two_scalar` has no pre-registration.** The flag exists; the LEVER needs a
   both-outcomes spec before an arm.

---

## 6. Deliverable manifest

⛔ Everything below lives in **at least two places** unless the row says otherwise. Repo paths are
relative to `D:\Projects\TanitAD` (the canonical tree since 2026-09-10). Staged, **not committed**.

### Code changed (repo, staged)

| path | what | md5 |
|---|---|---|
| `stack/scripts/rl_pilot_refc21.py` | `build_argparser()` extracted; **`--gt-bar`** (default OFF); **`--noise-mode`** (default `multiplicative`); `spec=` passed to `run_posttrain`; `gt_bar_fn` wired; two refusals (`emits_gt_bar` vs the parsed flag, and a bar-ON run with `frac_above_bar_mean is None`) | `bcfce77160458c452522821ec61953ed` |
| `stack/tanitad/rl/posttrain.py` | **`score_gt_bar()`**; the spec-commensurability guard (tested but never implemented before); `tied_at_max` + **`reward_audit_exercising`** in the run record | `374d41c07fd67c96cc1908194e6ad2c2` |
| `stack/tanitad/rl/audit.py` | the **separation check** (`tie_atol`, `tied_at_max`, `NOT SEPARABLE`); **`exercising_ctx()`** | `d40df0e8807614a3a80602babf0dfc41` |
| `stack/tanitad/rl/refcv3_adapter.py` | **`gt_bar_fn=`** parameter, `extras` assembly, `emits_gt_bar` attribute; three-tuple default preserved | `587ec6399d2bfef6e907f40f5e5a511c` |

### Tests (repo, staged)

| path | what |
|---|---|
| `stack/tests/test_rl_gt_bar_reaches_the_caller.py` | **NEW, 13 tests** — the caller-level mutation proof, both analytic identities, the partial-mask assertion, three deliberate-regression arms, the pilot's flag surface, the pilot's own batch shapes |
| `stack/tests/test_rl_audit.py` | **+9 tests** — the separation check both directions, the isolation mutation, the exercising context and its cross-check |
| `stack/tests/test_refc_cold_start_allowance.py` | ⭐ **a LITERAL option-surface pin that FIRED, exactly as intended.** It asserts the pilot's complete flag list to catch a new flag that could set `anchors.v0_conditioned` or waive the cold-start allowance. Adding two flags broke it; both were examined (each touches the ADVANTAGE or the SAMPLER, downstream of the anchor bank that reads `v0_conditioned` at `refc.py:1715`) and the literal was extended, never relaxed. The substring guard that does the real work is untouched. **26 passed.** |

### Test evidence

* **The RL set — 24 files, `test_rl_*.py`: 399 passed, 0 failed** (CPU).
* **The blast radius — every test file that imports `tanitad.rl`, 30 files: 493 passed, 5 failed.**
  ⛔ All 5 are in `test_refc_v3_agent_gt_head_drop.py` and are **PRE-EXISTING**, proven by blob
  comparison rather than asserted: `stack/scripts/refc_v3_train.py`,
  `stack/tests/test_refc_v3_agent_gt_head_drop.py` and `stack/tanitad/rl/refc_adapter.py` are each
  **byte-identical to `HEAD`** (40-char blobs on both sides, equal), and the failures are its own
  mutation anchors no longer matching `refc_v3_train.py`. ⚠️ Tally control: the summary line says
  `5 failed` and the output contains exactly **5** `FAILED` rows.
* ⚠️ **Two collection errors elsewhere in `stack/tests` are also PRE-EXISTING and unrelated**:
  `test_refa_v1_dk_hook.py` / `test_metric_decode_refusal.py` import `DistanceKeepingSpec`, which
  `grep -c` finds **0 times in both the worktree and `HEAD:stack/tanitad/refs/refa_v1.py`**.
* ⚠️ **The FULL `stack` suite (~7.4 k tests) did NOT complete inside this session — it BLOCKED at
  ~30 % with zero failures recorded, and the cause is environmental, not a regression.** Stated as
  an incomplete measurement rather than reported as a pass. ⛔ **Blocked, not slow — measured, not
  assumed:** the pytest process's `UserModeTime` was **4710937500 at two samples eight minutes
  apart**, i.e. it burned no CPU at all in between, and its only child was `git.exe`. Meanwhile
  **22 live `git.exe` processes** sat on the box, all carrying an editor integration's signature
  (`-c core.quotepath=false -c safe.directory=* -c core.fsmonitor=false`) rather than any of this
  session's. ⇒ the shared-index contention `CLAUDE.md` documents (a `read-tree` taking 13 minutes
  under ~97 concurrent git processes), reached through the suite's git-subprocess tests. Two
  earlier attempts blocked at the same place. The run was stopped by explicit PID.
  ⛔ **The artifact, read after the kill rather than inferred:** the log ends at `[ 30%]`, carries
  **`ZZSUITE-EXIT-127ZZ`** (the kill, not a clean exit), **0** `FAILED` rows, **0** `F`/`E` marks in
  the progress dots, and **NO tally line at all** — so there is no "N passed" to quote and none is
  quoted. ⚠️ The task notification for the waiter that watched this file reported *exit code 0*;
  that is the WAITER's status, not the suite's. Same family as `cmd | tail` reporting `tail`'s
  status — the exit code belongs to the wrong process, and only the artifact settles it.

### Research package (repo, staged)

| path | what |
|---|---|
| `…/Research/2026-09-11-rl-gt-bar-reachable/RESULT_GT_BAR_REACHABLE.md` | this document |
| `…/code/prove_gt_bar_reachable.py` | the mutation proof (writes the JSON below) |
| `…/code/compare_arms.py` | the arm comparison, control-first |
| `…/code/run_rc21_gtbar.sh` · `…/code/wait_then_run_gtbar.sh` | the chain and its GPU waiter |
| `…/raw/MUTATION_PROOF_GTBAR.json` | gradients, the sweep, the mutant's three failures, both md5s |
| `…/raw/COMPARE_ARMS.json` | `CONTROL_REPRODUCES_EXACTLY: true`, `THE_MASK_WAS_REACHED: true` |
| `…/raw/runs/{p1-grpo-2k-repro,p1-grpo-2k-gtbar,p2-reg-2k-sep}/` | `pilot_summary.json`, `config.json`, `config_contract.json`, both readouts, `summary.json`, `run.log` |
| `…/raw/chain_gtbar.out` · `…/raw/waiter_gtbar.out` | the chain's own markers, including every gate |

### Live copies (NOT the repo — named so nothing is stranded)

| where | what | also in repo? |
|---|---|---|
| `C:/Users/Admin/tanitad-rlrun/stack/` | the run tree; the four changed modules + two test files re-shipped and **md5-verified identical** to the repo | yes |
| `C:/Users/Admin/tanitad-rlrun/run_rc21_gtbar.sh`, `wait_then_run_gtbar.sh` | the chain scripts | yes (`…/code/`) |
| `C:/Users/Admin/tanitad-data/rl-pilot/p1-grpo-2k-{repro,gtbar}/`, `p2-reg-2k-sep/` | full run dirs **including `ckpt_after.pt` (~400 MB each)** | ⛔ **JSON + logs only.** The three `ckpt_after.pt` files exist in ONE place; they are re-derivable from the chain (3–4 min each on the 4060) but are not banked |

### ⛔ Escalation — for the Master Mind, not a README

1. ⛔ **A register row asserted a fix that was never in the code.** `D-GTBAR-SPEC-DIVERGENCE` read
   **CLOSED** — *"`run_posttrain` now refuses a divergent explicit spec"* — while the guard existed
   **only in its test**, which was RED on the tip. `GOALS_AND_CLAIMS.md` is corrected in place and
   the guard is implemented. ⚠️ It is the same failure as the missing `--gt-bar`: *a thing described
   as built, that nothing called.* Worth a sweep of other rows whose evidence is a test name.
2. **`GOALS_AND_CLAIMS.md` carries four new/corrected rows** (`E-GTBAR-REACHED-FROM-CALLER`,
   `D-REWARD-AUDIT-LIVENESS-NOT-GAMEABILITY`, `D-RL-AUDIT-WAS-UNRULABLE`, and the correction above).
3. **A pre-registration is owed before any two-scalar arm** — the flag exists, the lever does not
   have a committed both-outcomes spec.
