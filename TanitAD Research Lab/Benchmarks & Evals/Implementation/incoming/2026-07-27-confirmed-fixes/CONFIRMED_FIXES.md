# Four confirmed defects, fixed — with the test that would have caught each

**Date:** 2026-07-27 (local, Europe/Berlin). **Host:** dev box only (CPU). pod1 (training v2corpus),
pod2, pod3 and the eval pod were **not touched** — no GPU was needed for any of this.

**Evidence classes:** `MEASURED` (ours + artifact path) · `PUBLISHED` (cited) · `INHERITED` (another
agent/doc, NOT re-verified) · `ESTIMATED` · `HYPOTHESIS`.

🔒 PhysicalAI-AV is gated-confidential: no clip UUID or raw content appears in this folder.

**Suites:** `stack` **1123 → 1135** passed (7 skipped) · `taniteval` **488 → 514** passed. Both green.

---

## 0. VERDICT IN ONE BOX

> | # | defect | status | what it unblocks |
> |---|---|---|---|
> | **3** | the H2 "chance" comparator scored **1.7259x chance** | **FIXED + RE-SCORED + REPORT CORRECTED** | the ~52 GB corpus expansion can now be booked against a real baseline |
> | **2** | the head's own reachability clamp was never applied to the candidates | **FIXED, default ON for v1.5** | **3.58x cheaper** per-candidate compute, at a MEASURED cost of exactly zero |
> | **4** | `blind_baseline` returned `CIRCULAR` on three CLEAN targets | **FIXED + 3 TARGETS RE-ADJUDICATED** | three retired decision problems are admissible again |
> | **1** | `imagine_probes` has no candidate axis | **MADE LOUD + a real per-candidate roll added** | E-V5-1's negative is formally over-determined; the repair is wired, not just named |
>
> ### ⚠️ Three things this work found that the brief did not know
>
> **1. FIX 3 is bigger than "the ego arm".** With a comparator that IS chance, **every learned arm
> separates above it on BOTH surfaces** — not only `head_ego`. The brief's *"the correction does not
> rescue the image arms (0/24)"* is **VOID**: it came from `owed_chance_baseline.py` rows 71/74,
> which scored the **wrong target** (`C["Y"]`, the trigger label, instead of `1 - EX[:,1]`) and — row
> 74 — the **wrong score array** (`arm in k` matched `head_img` against `s__head_img_ego__*`). Both
> rows report the identical `AP_a = 0.00398`, which is the tell. Re-derived correctly here.
>
> **2. What still stands, stated before anything is claimed.** *"Adding images destroys the working
> ego head"* (3.74× → 1.59×) is **UNTOUCHED** — it is an arm-vs-arm comparison and no chance
> comparator enters it. **Only the comparator moved:** no arm's AP changed by more than 1e-5. And the
> **pre-registered H2 verdict stays `UNDERPOWERED`** — it is decided on paired RECALL
> (`h2c_eval.py:399-414`), which never touches this comparator.
>
> **3. FIX 2 must NOT be defaulted on for flagship v4, and is not.** The zero-change property was
> measured on **REF-C-XL's** emitted fan. v4's own fan geometry was never dumped, and v4 carries a
> standing invariant that the selector must never truncate the candidate set. `V4Config` therefore
> overrides `sel_reach_clamp = False` with the reason in the code. 🔴 **ESCALATION below.**

---

# FIX 3 — the "chance" comparator was 1.7259x chance *(priority 1: it gates ~52 GB)*

### The verification

MEASURED, reproduced from the committed `scores_heldout.npz` before anything new was computed
(`code/fix3_rescore_chance.py`, fidelity gate; `raw/fix3_chance_comparator.json`):

| | trigger surface | `NOT_T_seen` surface |
|---|---|---|
| published "chance" AP | **0.005269** | **0.046040** |
| true base rate | 0.0030527 | 0.0327620 |
| **inflation** | **1.7259x** | **1.4053x** |
| a genuinely random ranking (24 seeds) | 0.003172 | 0.033260 |

**Mechanism.** `h2c_eval.py:138` builds the comparator as `chance = np.zeros_like(y)` on the
documented belief that *"a constant score has AP exactly equal to the base rate WITHIN EACH DRAW"*.
That holds only if ties COLLAPSE. `h2c_stats.average_precision` ranked with
`np.argsort(-s, kind="mergesort")` — a **stable** sort, which on an all-tied score returns the
**identity permutation**, i.e. **row order**. `h2c_eval.py:85` lays the rows out as *[every
left-camera row, then every right-camera row]*, and the left camera carries the larger share of
positives. **The "constant score" was the ranker "fire the left camera everywhere".**

The old docstring claimed to compute *"the step-interpolated AP that
`sklearn.average_precision_score` computes"*. **It did not** — sklearn collapses ties. The docstring
even reasons correctly that row-order tie-breaking flatters heavily-tied scores and that this is
"the safe direction"… and never asks what happens to the score that is tied **everywhere**, which is
the one number the entire above-chance test rests on. **C13 class: the failure mode was named, then
not applied to the guard.**

Direction of the damage: the comparator is **HARDER** than chance ⇒ every `ΔAP − chance` was
**understated** and every null biased toward *"not separated"*. For a control, toward the desired
verdict.

### The repair

| where | what |
|---|---|
| **`taniteval/taniteval/rank_metrics.py`** (new) | `average_precision(y, s, ties="collapse"\|"row_order")` — collapse is the default and is sklearn-exact; `chance_ap` (= the base rate); `random_ranking_ap`; `comparator_audit`; **`assert_chance_comparator`**, which REFUSES any baseline sold as chance whose AP is not the base rate. No `force` argument: the person who wants to waive it is the person whose null depends on waiving it. |
| `…/2026-07-26-h2-classifier/scripts/h2c_stats.py` | `average_precision` now **delegates** to the package. The pre-fix body is kept as `_legacy_average_precision` — reproduction only — with the retraction written into its docstring. |
| `…/2026-07-26-h2-classifier/scripts/h2c_eval.py` | calls `assert_chance_comparator` **before** using the comparator and stores the audit in the results JSON. The assumption now fails loudly instead of being documented. |
| `…/2026-07-26-h2-classifier/H2_CLASSIFIER.md` | corrected in place: a CORRECTION box at the top, both `ΔAP vs chance` tables corrected beside the retracted originals, §6.1a / §7.1 / §8.1 amended, and **amendment A4** added. The auto-generated retracted table is left unedited as the exhibit — *no number in that document is typed by hand, including a wrong one.* |

### The corrected numbers

Paired episode-cluster bootstrap, B = 2000, unit = clip (322 clusters), `taniteval.ci._draws`.
Comparator: constant score, ties collapsed ⇒ **AP = base rate exactly in every draw** (seed-free).
Corroborated by 3 true-random-ranking seeds carried through the full paired bootstrap.

**Trigger surface** (306 positives / 100,238 camera-frame pairs):

| arm | AP (**unchanged**) | published ΔAP | **corrected ΔAP** | CI95 | verdict | true-random seeds |
|---|---|---|---|---|---|---|
| `head_img_ego` | 0.007948 | +0.002679 | **+0.004895** | [+0.000706, +0.028013] | ✅ above | 3/3 |
| `head_img` | 0.005506 | +0.000237 | **+0.002453** | [+0.000387, +0.009624] | ✅ above | 2/3 |
| `head_ego` | 0.012758 | +0.007489 | **+0.009705** | [+0.001378, +0.033221] | ✅ above | 3/3 |
| `heur_decel` | 0.007672 | +0.002406 | **+0.004619** | [+0.000295, +0.015898] | ✅ above | 2/3 |
| `heur_speed` | 0.001987 | −0.003282 | **−0.001066** | [−0.001604, −0.000597] | **below** | 3/3 |

**`NOT_T_seen` surface** (1,642 positives / 50,119 frames, 101 positive clips):

| arm | AP (**unchanged**) | published ΔAP | **corrected ΔAP** | CI95 | verdict | true-random seeds |
|---|---|---|---|---|---|---|
| `head_ego` | 0.122630 | +0.076590 | **+0.089868** | [+0.053051, +0.137277] | ✅ above | 3/3 |
| `head_img_ego` | 0.052046 | +0.006006 | **+0.019284** | [+0.008323, +0.040387] | ✅ above | 3/3 |
| `head_img` | 0.049144 | +0.003103 | **+0.016382** | [+0.005652, +0.042374] | ✅ above | 3/3 |

⚠️ **Read the size, not the sign.** On the trigger surface these are separations of **0.002–0.010 AP
against a 0.0031 base rate**, every interval touching its own lower edge. *Above chance* ≠ *useful*:
`head_img_ego − heur_decel` is still **+0.00027 [−0.00633, +0.01462]** — two million parameters and a
frozen 87 M encoder still do not beat *"rank frames by how hard the ego is already braking"*.

### The test that would have caught it

`taniteval/tests/test_rank_metrics.py` — **10 tests**, every guard driven with input designed to make
it fail:

- `test_constant_score_under_row_order_ties_is_NOT_chance` — front-loaded positives + a constant
  score: the legacy policy must still exhibit the defect (>1.3x base) **and** the repaired one must
  return the base rate to 1e-12. If the first assertion ever stops failing under `row_order`, the
  test is pinning nothing and says so.
- `test_the_bias_direction_is_toward_the_null` — across four front-loading fractions the defect
  **always inflates**. It is bias, not noise.
- `test_guard_REFUSES_the_row_order_comparator` — the guard must **fire** on the exact input that
  shipped, name TIE-HANDLING and ROW ORDER, quote how far off it is, **and then accept** the
  repaired comparator (a guard that always fires is not a guard).
- `test_guard_REFUSES_an_informative_comparator_that_is_not_constant` — the second failing mode,
  with a message that must **not** blame tie handling.
- `test_collapsed_ties_match_sklearn` — cross-checked against `sklearn` at 1 / 3 / 17 / 1000 distinct
  score values, i.e. from fully-constant to nearly-untied.
- plus: random rankings must **bracket** the base rate; perfect/inverted rankings bracket the metric;
  `row_order` stays reachable and must **differ** from the default (or the fix did nothing); shape
  mismatch raises rather than broadcasting.

### What it unblocks

🔴 **The ~52 GB gated camera re-download can now be booked against a baseline that is not broken.**
It was about to be bought to re-measure against a comparator running 1.73x hot. The expansion is
**still the right call** — the pre-registered verdict is unchanged at `UNDERPOWERED` — but the
arithmetic behind "what would 4x the positives buy" now starts from real numbers.

---

# FIX 2 — the free reachability clamp *(priority 2: 3.58x, zero risk)*

### The verification

MEASURED on `taniteval/results/fan_refc-xl-30k.pt` — REF-C-XL @30k, the emitted 256-candidate fan
with its real per-candidate logits, on the **881 canonical val windows / 40 episodes**
(`…/2026-07-27-percandidate-labels/raw/t1_clip_fansize.json`, INHERITED; **re-verified here** by
`taniteval/tests/test_reach_clamp_committed_windows.py`, which recomputes every figure from the dump):

| quantity | value |
|---|---|
| candidates removed | **72.08 %** |
| windows with an empty survivor set | **0.00 %** |
| ADE-oracle survives | **100 %** |
| **paired Δ ADE** (episode-cluster bootstrap, B=2000) | **0.0000 [0.0000, 0.0000]** |
| windows where the pick moves | **0 of 881** |
| miss@2m | 0.0159 → 0.0159 |
| ⇒ per-candidate compute | **3.58x cheaper** |

**The vocabulary is blameless.** `furthest_point_sample` returns `pool[chosen]`, so all 256 anchors
are bitwise identical to real human windows. The excess is the **unbounded offset head**: MEASURED
max candidate mean speed **171.5 km/h**, p99 **159.6**, against a val GT max of **132.4** — all three
re-derived from the dump in the test. **Clamp the refinement; do not touch the vocabulary.**

Corroborating on a different surface (INHERITED, `raw/t2b_clip_x_rule.json`): on the *rule* scorer
over 3,360 windows the same clamp moves the pick from **9.7073 → 3.8950 m (−59.9 %)** while raising
the composite (0.8067 → 0.8480) and leaving the ADE-oracle at 0.5906 either way. Inert where the
scorer is good, decisive where it is not.

### The repair

`stack/tanitad/models/flagship_v15.py`:

- `candidate_mean_speed(fan, horizon_s)` — `‖wp_last‖ / horizon_s`. **Deliberately named apart from
  `terminal_speed`**, which is the *instantaneous* `(wp[-1]−wp[-2])/0.5 s` used by the VTARGET
  aspiration term. Two metrics under one name has already cost this program days.
- `reachability_mask(fan, v0, accel_max=2.5, horizon_s=2.0)` — free-standing, so a caller can prune
  **before** spending per-candidate compute without instantiating a head. The band is
  `v0 ± a_max·T` = **the head's own `sel_accel_max`**, already used on the goal at
  `flagship_v15.py:139,455`. Nothing is tuned on held-out error, and the band is **2x wider than the
  kinematic bound** on a mean speed — so 72.08 % is a statement about the offset head, not a tight band.
- `V15Config.sel_reach_clamp: bool = True` — **default ON, flag to disable.**
- `FlagshipV15Head.select` masks the **argmax only**. The returned `sel_score` stays **unmasked and
  bit-identical**, so `v15_losses` sees exactly the tensor it saw before and **no `-inf` can reach a
  cross-entropy**. Training risk is zero by construction, not by hope. A row with an empty survivor
  set keeps its whole fan: an unreachable-everywhere window is a measurement failure, not a licence
  to return no plan. Telemetry: `reach_frac_candidates_clipped`, `reach_frac_windows_empty`.

### The test that would have caught it

`taniteval/tests/test_reach_clamp_committed_windows.py` — **5 tests on the committed windows**:
`test_the_clamp_is_free_on_the_committed_windows` asserts 0.7208, oracle 1.0, `Δ = lo = hi = 0.0`,
881 windows / 40 episodes, miss@2m unchanged, and 3.5 < speedup < 3.7.

⚠️ **`Δ = 0` is only evidence if the instrument CAN move the pick.** So
`test_the_mask_CAN_move_the_pick_so_the_zero_is_evidence` drives the identical code at
`accel_max ∈ {0.5, 0.2, 0.05}` and requires that the pick **does** move and the ADE gets **strictly
worse** — if a tight band ever *helped*, the clamp would be a tuned selector rather than physics.
Without this, the headline is the C13 class: a guard that cannot fire.

`stack/tests/test_flagship_v15.py` — **5 unit properties**, including
`test_the_clamp_MOVES_the_pick_when_the_top_candidate_is_unflyable` (a 47 m/s candidate that wins the
argmax; the clamp must delete it), `test_a_window_with_no_reachable_candidate_keeps_its_whole_fan`
(every candidate at 100 m/s → fall back, no crash, no `-inf`), and
`test_the_clamp_never_touches_the_SUPERVISED_score` (loss terms bit-identical with the clamp on and off).

### 🔴 ESCALATION — the clamp is OFF on flagship v4, deliberately

`V4Config.sel_reach_clamp = False`, overriding the inherited default, because:

1. the zero-change property is **MEASURED on REF-C-XL's fan and UNMEASURED on v4's** — the same
   study's surface B (`v5_v4_windows_reduced.pt`) has v4's per-candidate errors but **no fan geometry
   and no usable score ranking**, so only the coverage side was ever computable for v4;
2. v4 carries a standing invariant — *"`q` MUST NOT EXIST in the deployment path… no masking, no
   `-inf`, no top-k"* — that exists to keep a measurement arm which cost **+0.21…+5.82 m** out of the
   deployed selector. A physical band is a different object from a score-based truncation, but
   *"different in kind"* is an argument, and this program settles those with measurements.

**To flip it:** dump v4's fan geometry + logits on the same 881 windows, re-run
`t1_clip_and_fansize.py`, and enable only if the paired Δ is again 0.0000 with the oracle surviving.
Pinned by `test_the_reachability_clamp_is_OFF_on_v4_until_it_is_measured_there`, so inheritance
cannot smuggle the default across a surface it was not measured on.

### What it unblocks

The candidate-conditioned imagination of FIX 1 costs one predictor rollout **per candidate**. At 256
candidates that is the reason it was never built. **The clamp removes 72.08 % of them for a MEASURED
zero change**, so the roll costs **3.58x less** — the two fixes were designed to compose, and
`imagine_candidates(..., keep=reachability_mask(...))` is the composition.

---

# FIX 4 — a firewall that returned `CIRCULAR` on clean targets *(priority 3: actively dangerous)*

### The verification

MEASURED, reproduced exactly (`code/fix4_readjudicate_situations.py`,
`raw/fix4_situation_readjudication.json`): all three situation targets in
`…/2026-07-26-situation-classifier/artifacts/sc_results.json` were `CIRCULAR` — i.e. **INADMISSIBLE,
"any score on it measures the lookup, not the model"** — while the same record's `context_leaks = 0`
and `blind_skill_over_majority ≈ 0` said the context carries nothing.

| target | blind | majority | real | skill | route to the wrong verdict |
|---|---|---|---|---|---|
| `roundabout` | 0.9970 | 0.9970 | 0.9864 | 0.0 | **`blind ≥ 1 − eps` fires on the MAJORITY CLASS ITSELF**, at a positive rate of **0.0030** |
| `intersection` | 0.9743 | 0.9743 | 0.8194 | 0.0 | **`vision_buys_nothing` compares ACCURACIES** |
| `lane_change` | 0.9787 | 0.9788 | 0.9193 | **−7.6e-05** | same — and the blind head is **below** the floor |

A recall-seeking rare-event model **must** lose an accuracy comparison to *"always predict
negative"*. That clause could not fail in the model's favour, so it was never a test.

### The repair

`taniteval/taniteval/blind_baseline.py` — every clause of the C13 class, *a test that cannot fail is
not a test*:

1. **`deterministic` disarmed when degenerate** — `blind ≥ 1 − eps` is meaningless when the majority
   class already clears the same bar.
2. **`vision_buys_nothing` disarmed when degenerate** — the accuracy comparison is admissible only
   when the real model itself clears the accuracy floor (`real ≥ majority + SKILL_EPS`); otherwise it
   cannot distinguish *"vision buys nothing"* from *"accuracy is the wrong statistic"*.
3. **A `leak_test_degenerate` bound** — `max_possible_blind_skill = 1 − majority`; below `SKILL_EPS`
   the `LEAKY` test cannot fire at any observed value (M8 class, a proof for a bounded metric).
4. **The verdict falls back to BALANCED ACCURACY** (macro per-class recall), whose floor is
   `1/n_class` at **any** imbalance — the exact property raw accuracy lacks. Reported in every record
   under `degeneracy_audit`, and `statistic` names which scale decided the verdict.
5. **New verdict `REFUSED`** when even that is undefined: not admissible for registration (a problem
   the firewall cannot adjudicate must not slip through) but **not `CIRCULAR`** either. Its `_read`
   says *UNADJUDICATED* and points at the rare-event route (`taniteval.rank_metrics`).
6. **Variable arity** — `blind_option_baseline(option_context, group, is_chosen, eid)`: a shared
   per-option scorer with a **masked softmax over each decision point's own option set**, so K = 2
   and K = 7 are the same problem to it. Floors reported both ways: `chance = mean_i 1/K_i` (the
   leak-relevant one, and the contrast the S1 audit says was never intervalled) and the best fixed
   option index, chosen on TRAIN and applied out of fold. `blind_conditioning_baseline` now **REFUSES
   loudly** when handed a varying `n_options`, naming the right tool — padding to fixed arity is a
   strictly weaker attack, i.e. a **lower bound** on the leak, the wrong direction for a firewall.

### The re-adjudication

| situation | published | **repaired** | statistic | balanced blind | balanced real |
|---|---|---|---|---|---|
| `lane_change` | CIRCULAR | ✅ **CLEAN** | balanced_accuracy | 0.4999 | **0.6069** |
| `roundabout` | CIRCULAR | ✅ **CLEAN** | balanced_accuracy | 0.5000 | **0.5570** |
| `intersection` | CIRCULAR | ✅ **CLEAN** | balanced_accuracy | 0.5000 | **0.7288** |

Every published number reproduces (blind 0.9787/0.9970/0.9743, majority 0.9788/0.9970/0.9743, real
0.9193/0.9864/0.8194) on the same held-out frames, the same context construction and the same
subsample rule. The balanced numbers say what raw accuracy hid: **the blind head is a constant
predictor** (0.500 = exactly its floor) while the image model carries **real** balanced skill.
`vision_buys_nothing` was not merely degenerate — **it was backwards.**

⚠️ **This does not say the situation heads work.** It says the circularity firewall was not the
instrument that retired them. The pre-registered AP-based `− head_ego` contrast is untouched.

⚠️ **The situation-classifier stream DIAGNOSED this correctly on 2026-07-26** and wrote it into its
own record (`MDE_AUDIT: "DEGENERATE — the CIRCULAR branch cannot fail on this target"`) and into a
🔴 escalation. The record still said `verdict: CIRCULAR`, and the module still shipped the defect.
**That is the "escalate integration, don't write it into a doc" failure class — a correct diagnosis
living next to the wrong verdict for a day.** The escalation is now marked CLOSED in
`SITUATION_CLASSIFIER.md` with the re-adjudication table.

### The tests that would have caught it

`taniteval/tests/test_blind_baseline_rare_events.py` — **13 tests**, each reconstructing a real
failure and each checked in **both** directions:

- `test_a_rare_target_with_a_NOISE_context_is_no_longer_called_CIRCULAR` — the context is **pure
  noise**, so any `CIRCULAR` is definitionally wrong. It first pins that the degeneracy is **real**
  (`maj ≥ 1 − eps`, `max_possible_blind_skill < SKILL_EPS`) — otherwise the test proves nothing.
- `test_a_recall_seeking_real_model_no_longer_forces_CIRCULAR` — a model at 60 % recall / 18 % FPR:
  HIGH recall, LOWER accuracy than the majority predictor. Asserts the accuracy inversion is
  reproduced **before** asserting the verdict.
- `test_the_repair_still_catches_a_REAL_leak_on_a_rare_target` — a rare target that IS a lookup of
  its context must still come back `CIRCULAR`. **Disarming the degenerate tests must not disarm the
  firewall**, or FIX 4 traded a false positive for a false negative.
- `test_a_balanced_target_is_still_decided_on_RAW_accuracy` — no behaviour change where accuracy was
  never broken.
- `test_fixed_class_entry_point_REFUSES_a_variable_arity_problem` — hand it an S1-shaped problem; it
  must refuse, say "LOWER BOUND", and name `blind_option_baseline` — **and still accept** a genuinely
  fixed-arity declaration.
- `test_option_baseline_CATCHES_a_leak_the_padded_attack_would_miss`, plus arity-exactness, the
  `mean_i 1/K_i` chance floor, one-option refusal, and exactly-one-chosen validation.

The 12 pre-existing `test_blind_baseline.py` tests are **unchanged and green** — including all three
`CIRCULAR` cases, so the real `route_target = _NAV_TO_ROUTE[nav_cmd]` post-mortem still fires.

### What it unblocks

Three decision problems are admissible again. More importantly the firewall is safe to point at any
rare-event target — which is most safety-relevant targets in this program.

---

# FIX 1 — `imagine_probes` has no candidate axis *(priority 4)*

### The verification

MEASURED (`…/2026-07-27-percandidate-labels/raw/t4_imagination_conditioning.json`, INHERITED;
re-verified here as an executable invariant): `imagine_probes` returns **32 tokens at `n_anchors`
64 and at 256** — `n_probes × |imag_read|` = 8 × 4, with **no dependence on `n_anchors` anywhere**.
`probes` is `[M, K, 2]`, a vocabulary **shared across the batch**
(`pr = probes.unsqueeze(0).expand(...)`), and `V15Decoder._decode` emits every candidate from the
**same `kv`**. So the 32 tokens are **identical for all 256 candidates**.

⇒ **The imagination path structurally cannot rank candidates, and E-V5-1's imagination-scoring
negative is over-determined — the experiment could not have worked.** A function that silently
returns the same thing for every candidate is the same class as a guard that cannot fail.

### The repair — both halves

**Loud (the mandatory half).** In `stack/tanitad/models/flagship_v15.py`:

- `IMAGINATION_HAS_CANDIDATE_AXIS = False` and `IMAGINATION_TOKEN_AXES = ("batch", "probe x
  read_step", "state_dim")` — the absence is a **named constant**, not a paragraph.
- `imagination_token_count(cfg)` — takes the **config**, so "the answer does not depend on
  `n_anchors`" is something you can ask and assert.
- **`assert_candidate_axis(x, n_candidates, name, axis)`** — the reusable guard, failing in **both**
  degenerate ways because they are different bugs with the same silent symptom: no axis of that
  length, **or** an axis that is CONSTANT along it.
- `imagine_probes(..., n_candidates=...)` raises `NoCandidateAxis` naming the measurement and
  pointing at the real roll. The request E-V5-1 implicitly made now fails loudly.

**Real (the repair).** `imagine_candidates(predictor, states, actions, cand_actions, read, v0n,
keep=None)` → `[B, N, |read|, S]` with a genuine candidate axis: one frozen-predictor rollout **per
candidate**, mechanism byte-identical to `imagine_probes` (1-step head, slide the window, append the
next action, v1 speed channel **held** at the observed `v0` — leakage-safe). `keep` rolls only the
surviving candidates and leaves the rest **zero**, which is where FIX 2 pays: **3.58x less compute**.
The mask is an **input, never a hidden default** — a caller must not rank on zeroed rows.

⚠️ `cand_actions` must come from the fan via the inverse of the corpus's own action definition
(`traj_to_actions`, the v5 stream — INHERITED, not re-verified here). Wiring that inversion into the
head's conditioning path is the remaining step and is **not** done here; what is done is that the
axis now exists, is tested, and its absence can no longer be assumed away.

### The test that would have caught it

`stack/tests/test_flagship_v15.py` — **6 tests**:
`test_imagination_token_count_is_INVARIANT_to_n_anchors` (the measured fact as an executable
invariant: `{32}` at both 64 and 256); `test_imagine_probes_output_is_IDENTICAL_for_every_candidate`
(broadcasts the tokens across a 256-candidate fan the way the decoder does, and the guard **must
refuse** — "CONSTANT along it"); `test_asking_imagine_probes_for_a_candidate_axis_RAISES` (**and** the
ordinary call must still work, or the guard is just a wall);
`test_assert_candidate_axis_fires_on_BOTH_degenerate_shapes`;
`test_imagine_candidates_HAS_a_real_candidate_axis`; and
`test_imagine_candidates_rolls_only_the_REACHABLE_candidates` (asserts only 4 of 12 rows are rolled,
masked rows come back **zero not stale**, and an all-empty mask neither crashes nor fabricates latents).

### What it unblocks

E-V5-1's negative can be **formally retired as over-determined** rather than re-argued, and the next
imagination-selection experiment has a path that can actually work — at 3.58x less compute than the
naive one, because of FIX 2.

---

# Retraction-log rows — DRAFTED, **NOT FILED**

⚠️ `Project Steering/RETRACTION_LOG.md` is a shared append-only file and I did not edit it. Paste
these four rows; the classes are named so the log teaches, not just corrects.

```
| 2026-07-27 | H2_CLASSIFIER.md §0.2 / §0.4 / §6 / §7.1 / §8.1 | "⛔ NO ARM IS ABOVE CHANCE" and
"neither image arm clears chance at all" | RETRACTED. The comparator was not chance: an all-tied
constant score ranked by a STABLE argsort = ROW ORDER (all left-camera rows first). AP 0.005269 vs
base 0.0030527 = 1.7259x (trigger); 0.046040 vs 0.032762 = 1.4053x (NOT_T_seen). Corrected: EVERY
learned arm separates above chance on both surfaces; heur_speed separates below. No arm's AP moved
(<1e-5) — ONLY the comparator did. The pre-registered UNDERPOWERED verdict (paired recall) and
"images destroy the ego head" (arm-vs-arm) are both UNCHANGED. |
ROOT-CAUSE CLASS: **C13-COMPARATOR — a baseline sold as "chance" that was never tested against
chance.** Sibling of "a guard that cannot fail": here, a FLOOR that was never checked to be the
floor. Whenever a null is adjudicated against a constructed comparator, MEASURE THE COMPARATOR
FIRST. Enforced by `taniteval.rank_metrics.assert_chance_comparator`. |

| 2026-07-27 | h2c_stats.average_precision docstring | "this is the step-interpolated AP that
`sklearn.average_precision_score` computes" | FALSE. sklearn COLLAPSES ties; that implementation
broke them by row order. The same docstring correctly analysed the risk of row-order ties for
heavily-tied scores and concluded it was "the safe direction" — while never applying the analysis to
the score that is tied EVERYWHERE. |
ROOT-CAUSE CLASS: **C1-EXTENDED — a metric NAME is not a metric DEFINITION, and neither is a CITED
REFERENCE IMPLEMENTATION.** "Equivalent to <library>" is a testable claim; test it (we now do, at
1/3/17/1000 distinct score values). |

| 2026-07-27 | …/2026-07-26-situation-classifier/artifacts/sc_results.json (all 3 situations) |
verdict CIRCULAR / admissible=false | RETRACTED -> CLEAN on all three. Two degenerate routes in
taniteval.blind_baseline: (a) `blind >= 1-eps` is cleared by the MAJORITY CLASS at a 0.0030 positive
rate; (b) `vision_buys_nothing` compares ACCURACIES, which a recall-seeking rare-event model must
lose to "always predict negative". Balanced accuracy: blind 0.4999/0.500/0.500 (exactly its floor)
vs real 0.6069/0.5570/0.7288 — the clause was BACKWARDS, not merely degenerate. |
ROOT-CAUSE CLASS: **C13-INVERTED — a guard that cannot PASS.** A control biased toward firing is as
broken as one that cannot fire, and more expensive: it retires admissible work. SECONDARY CLASS:
**DIAGNOSED-BUT-NOT-FIXED** — the owning stream measured this exactly right in its own MDE_AUDIT and
escalated it, and the record still shipped the wrong verdict for a day. A correct diagnosis next to
an uncorrected verdict is not a fix. |

| 2026-07-27 | E-V5-1 imagination-selection negative | read as evidence about imagination-based
selection | OVER-DETERMINED, not evidence. `imagine_probes` returns 32 tokens invariant to n_anchors
(verified at 64 and 256) and IDENTICAL for all 256 candidates: there is no candidate axis, so the
experiment could not have worked whatever imagination does. |
ROOT-CAUSE CLASS: **SILENT-INVARIANCE — a per-candidate quantity that is constant across
candidates.** Same class as a guard that cannot fail: it returns a clean-looking number that cannot
carry the information the experiment needs. Before reading ANY per-candidate result, assert the
candidate axis exists AND varies (`flagship_v15.assert_candidate_axis`). |
```

---

# Deliverable manifest

**Everything is in the repo working tree and STAGED (`git add`). I committed nothing and pushed
nothing.** Nothing produced by this task lives in only one place — no pod was used.

> ⚠️ **A sibling commit swept four of my files mid-session — the CLAUDE.md whole-index hazard, third
> occurrence.** `2dc2795` ("PSEUDO-SIMULATION…", 02:38 local) committed the **entire index**, which
> at that moment contained `taniteval/taniteval/rank_metrics.py`,
> `taniteval/taniteval/blind_baseline.py`, `taniteval/tests/test_rank_metrics.py` and
> `taniteval/tests/test_reach_clamp_committed_windows.py` — none of them that stream's work, all of
> them under a message about pseudo-simulation.
>
> **VERIFIED SAFE:** the committed blob hashes are **identical** to my working tree, and
> `git show HEAD:…/blind_baseline.py` contains the full repaired API (`REFUSED`, `FirewallRefused`,
> `balanced_accuracy`, `blind_option_baseline`). Nothing was lost or truncated. Recorded because the
> *provenance* is now wrong: four files carry a commit message describing someone else's experiment.
> Everything else from this task is staged and uncommitted.
>
> Verify with `git ls-files --stage`, not a scoped `git status --short` — a scoped status shows these
> four as clean and reads as if they were never produced.

| artifact | what it is | where it lives |
|---|---|---|
| `CONFIRMED_FIXES.md` | this document | **repo** |
| `code/fix3_rescore_chance.py` | FIX 3 re-scoring driver; fidelity-gated (refuses to quote a new number until every committed one reproduces) | **repo** |
| `raw/fix3_chance_comparator.json` | ⭐ FIX 3 raw output — both surfaces, both comparators, per-arm intervals, the predecessor audit's two defects, the arm-APs-unchanged proof | **repo** |
| `raw/fix3_chance_comparator.log` | its run log | **repo** |
| `code/fix4_readjudicate_situations.py` | FIX 4 re-adjudication driver (same context construction and subsample rule as `sc_eval.py:249-268`) | **repo** |
| `raw/fix4_situation_readjudication.json` | ⭐ FIX 4 raw output — published vs repaired verdict per situation, with every published number reproduced | **repo** |
| **`taniteval/taniteval/rank_metrics.py`** | **NEW** — tie-honest AP, `chance_ap`, `random_ranking_ap`, `comparator_audit`, `assert_chance_comparator` | **repo** |
| `taniteval/tests/test_rank_metrics.py` | **NEW** — 10 tests (FIX 3) | **repo** |
| `taniteval/tests/test_reach_clamp_committed_windows.py` | **NEW** — 5 tests on the 881 committed windows (FIX 2) | **repo** |
| `taniteval/tests/test_blind_baseline_rare_events.py` | **NEW** — 13 tests (FIX 4) | **repo** |
| `taniteval/taniteval/blind_baseline.py` | degeneracy audit, balanced-accuracy fallback, `REFUSED`, `FirewallRefused`, `balanced_accuracy`, `blind_option_baseline`, variable-arity refusal | **repo** (modified) |
| `stack/tanitad/models/flagship_v15.py` | `sel_reach_clamp` (default ON), `candidate_mean_speed`, `reachability_mask`, `reach_band`; `NoCandidateAxis`, `assert_candidate_axis`, `imagination_token_count`, `imagine_candidates`, `imagine_probes(n_candidates=…)` refusal | **repo** (modified) |
| `stack/tanitad/models/flagship_v4.py` | `sel_reach_clamp = False` override + the reason + the escalation | **repo** (modified) |
| `stack/tests/test_flagship_v15.py` | +11 tests (FIX 2 × 5, FIX 1 × 6); `test_selection_uses_the_returned_score` re-pointed at the flag | **repo** (modified) |
| `stack/tests/test_flagship_v4.py` | +1 test pinning the v4 default OFF; the no-truncation guard now asserts its own precondition | **repo** (modified) |
| `…/2026-07-26-h2-classifier/H2_CLASSIFIER.md` | **corrected in place** — CORRECTION box, both tables corrected, §6.1a / §7.1 / §8.1 amended, amendment **A4** | **repo** (modified) |
| `…/2026-07-26-h2-classifier/scripts/h2c_stats.py` | delegates to `taniteval.rank_metrics`; pre-fix body kept as `_legacy_average_precision` | **repo** (modified) |
| `…/2026-07-26-h2-classifier/scripts/h2c_eval.py` | calls `assert_chance_comparator` before using the comparator | **repo** (modified) |
| `…/2026-07-26-situation-classifier/SITUATION_CLASSIFIER.md` | escalation #1 marked **CLOSED** with the re-adjudication table | **repo** (modified) |

**Not regenerated, and deliberately so:** `artifacts/h2c_results.json` and `artifacts/c12_fix.json`
are left at their pre-fix values, and the auto-generated retracted table in `H2_CLASSIFIER.md` is
left unedited. They are the **exhibit** the retraction refers to, and re-running `h2c_eval.py` (now
repaired) regenerates them from the same `scores_heldout.npz` at any time. The corrected numbers live
in `raw/fix3_chance_comparator.json`, which is where every corrected quote in the document points.

**Reproduction, end to end (dev box, CPU, ~10 min):**

```
python "…/2026-07-27-confirmed-fixes/code/fix3_rescore_chance.py" \
    --h2c    "…/2026-07-26-h2-classifier" \
    --scores "…/2026-07-27-rung1-planner-and-owed-controls/raw/h2clf_scores" \
    --out    "…/2026-07-27-confirmed-fixes/raw"
python "…/2026-07-27-confirmed-fixes/code/fix4_readjudicate_situations.py" \
    --sc  "…/2026-07-26-situation-classifier" \
    --out "…/2026-07-27-confirmed-fixes/raw"
cd stack     && pytest -q          # 1135 passed, 7 skipped
cd taniteval && pytest -q          # 514 passed
```

---

# Escalations — raised here, not buried

1. 🔴 **The ~52 GB H2 corpus expansion is now decidable against a real baseline** and should be
   booked. The pre-registered verdict is unchanged at `UNDERPOWERED`; what changed is that the
   comparator it will be re-measured against is no longer running 1.73x hot.
2. 🔴 **Flip `V4Config.sel_reach_clamp` only after measuring it on v4's own fan** (§FIX 2). The
   missing input is a v4 fan-geometry + logits dump on the same 881 windows — cheap, and it converts
   an argument into a measurement.
3. 🔴 **`owed_chance_baseline.json` rows 71 and 74 are VOID** (wrong target, and row 74 the wrong
   score array). Anything downstream quoting *"0/24 seeds"* for the image arms on `NOT_T_seen` must
   be repointed at `raw/fix3_chance_comparator.json`.
4. ⚠️ **Four retraction rows are drafted above and NOT filed.** `RETRACTION_LOG.md` is shared; the
   owner should paste them.
5. ⚠️ **Commit `2dc2795` carries four of this task's files under a pseudo-simulation message** (see
   the manifest note). Content is verified identical and nothing is lost; the provenance is what is
   wrong. Third occurrence of the whole-index sweep this repo's `CLAUDE.md` already documents twice.
6. ⚠️ **`traj_to_actions` still lives only in the v5 stream's `incoming/` code** — the candidate →
   action-sequence inversion that `imagine_candidates` needs. Promoting it into `stack/` is the
   remaining step to make candidate-conditioned imagination usable from the head, and it is exactly
   the "unmerged instrument" shape that has cost this program 10 days before.
