# PRE-REGISTRATION — the `--w-tac-goal` weight sweep (D-TACGOAL-2 / PI queue item 10)

**Registered 2026-09-10, BEFORE any weighted arm produced a single log row.**
⭐ The registration is timestamped by an artifact, not by a claim: it was written while the
control arm `A_w0` was at **step 180**, and the check run immediately AFTER writing it — Thor
clock **`2026-09-10T23:35:55Z`** — reads `A_w0` at **step 200 of 400** with `C_w0p05`,
`D_w0p5` and `E_w5p0` returning **`No such file or directory`** for `metrics.jsonl`. ⛔ No
weighted arm had produced one byte when the rule below was fixed. ⛔ Nothing here may be edited
after the data is read; a post-hoc finding needs its own pre-registration.

⚠️ **THIS IS A CALIBRATION, NOT A CAPABILITY TEST.** It does not test whether the 22-token
tactical head helps — that is `H-TACGOAL-1`, already pre-registered at
`…/2026-09-06-label-wiring/PREREG_D-TACGOAL-1.md`, and it is **not re-litigated here**. This
package answers one narrower question the PI asked to be **measured rather than guessed**:
*at what weight does `tac_goal_tok_head` actually learn without displacing what already works?*

⛔ **NO TIER STAMP, NO ADE, NO FOUR-FAMILY EVAL TABLE — and that is deliberate, not an
omission.** No arm here is evaluated; every number is a **training-loss diagnostic** read off
the trainer's own `metrics.jsonl`. A tier stamp on a non-eval would be a false strength claim,
and the four-family rule binds *evals*. ⛔ Consequently **nothing in this package licenses any
statement about driving performance, traffic-light behaviour, or tactical goal quality.**

---

## 1. The rig, stated before the result

| | |
|---|---|
| trainer | `stack/scripts/refc_v3_train.py` — **the refcv6 arm-C trainer itself** |
| params | **108,257,502** (`config.json:param_breakdown.total`), head **11,286** |
| corpus | `physicalai-b1-w120-256x640cyl` + `s2_labels_v7.2_train.jsonl.gz` (**4,572** records, md5 `0ff902130ce76886b8a925eceed9e3a5`) — refcv5-v2's own corpus. `v2_parity.parity = True`, checked by the trainer. No episode re-selected. |
| steps | **400**, `--warmup 20` (5 %, matching refcv5-v2's 2000/40284 **fraction**) + cosine to 0 |
| seed | **0**, every arm |
| host | Thor, verified idle before launch |

⛔ **NOT the v7-tiny rig the brief named, and the reason is structural:**
`stack/scripts/train_v6_staged.py` contains **0** occurrences of `--w-tac-goal` against a
same-read positive control of **232** `add_argument` calls. The head under test does not exist
in that trainer, so a v7-tiny sweep would have measured nothing. Banked:
`raw/rig_census.json`.

## 2. Arms — one lever, nothing else

`A_w0` (flag absent) · `A_w0_rep` (A_w0's flags, A_w0's seed, run again, **zero levers moved**)
· `B_w0p005` 0.005 · `C_w0p05` 0.05 · `D_w0p5` 0.5 · `E_w5p0` 5.0.

⛔ **`A_w0_rep` is not optional.** A separated result from a one-seed arm is **necessary and not
sufficient**; the replicate is the floor every "the lever moved it" statement is read against.
⚠️ **Its honest scope, stated now:** it varies **nothing**, so it measures the rig's
**nondeterminism** floor (dataloader ordering, cuDNN/atomics), which is a **LOWER BOUND** on
the training-variance floor. A different-seed replicate would be a strictly larger floor and is
named in §5 as the first thing that would change the reading.

## 3. Controls that must read a known value — or the panel is INCONCLUSIVE

1. ⛔ `A_w0` must read `gp_tac_goal_tok_head_grad_abs_sum` **exactly 0.0** at **every** logged
   step, with `n_grad_none` **2 of 2** — the term is *absent from the graph*, not multiplied by
   zero. If it reads anything else, the probe is not measuring what it claims and **no weight
   is recommended.**
2. `gp_tac_goal_tok_head_n_params` must read the literal **11,286**.
3. Two extra no-information controls ride along free, and must read **exactly 0.0** in **every**
   arm: `core.decoder.offset_head` (6,160) and `scorer.goal_point` (1,026), both independently
   MEASURED untrained in `REFCV6_ARM_FACTS.md` §5. A probe that made them non-zero would be
   reporting something other than gradient.
4. Every weighted arm must read `n_grad_none` **0 of 2**.

## 4. ⭐ THE DECISION RULE, committed now

A weight **w** is **ADMISSIBLE** iff all three hold:

* **(a) THE HEAD LEARNS.** `grad_abs_sum` on `tac_goal_tok_head` is **> 0.0 at every logged
  step**, and `n_grad_none` is 0 of 2.
* **(b) THE PRIMARY DOES NOT MOVE BEYOND THE FLOOR.** `|Δ tail-5 traj|` against `A_w0` is
  **≤** `|A_w0_rep − A_w0|` on the same statistic. ⛔ Read against the replicate, never
  against zero.
* **(c) THE NEW SURFACE DOES NOT OUT-SHOUT THE TWO THAT ALREADY EXIST.** The contribution
  `w × tac_goal` is **≤** the trainer's own tactical-auxiliary contribution
  `0.025 × (lat + lat_tac + lon + lon_tac)` — i.e. `MANEUVER_WEIGHT = 0.1` as actually spent
  (`refc_train.py:80,90,91`; the `/2.0` split at `refc_v3_train.py:2495-2496`).

**RECOMMENDED = the LARGEST admissible w** — the strongest learning signal inside the band.

**Committed in advance, both directions:**

* If the admissible band is **non-empty**, the recommendation is its largest member, reported
  with its ratio to the primary and to the tactical budget, and handed to the PI **as a
  measurement plus a recommendation — never as a settled value.**
* If the band is **EMPTY** — every weight that trains the head also moves the primary beyond
  the floor — that is reported as a **FAILURE of the additive framing**, and the next lever is
  named and, if it is cheap, executed in the same run. The pre-declared next levers, in order:
  **L1** a weight between the largest that passes (b) and the smallest that passes (a), by
  bisection; **L2** a **warm-up ramp** on `w` so the head is supervised only after the primary
  has settled; **L3** the `/3.0` re-split of `MANEUVER_WEIGHT` across three tactical surfaces,
  which is ⛔ **a separate arm, not a tweak**, because it changes `lat`/`lon` pressure and
  breaks pairing with the banked arm.
* If **control 1 fails**, the panel is **INCONCLUSIVE** and no weight is recommended at all.

⛔ **In every branch the value remains the PI's to ratify.** This package hands over a curve and
a recommendation; it does not close queue item 10.

## 5. What will NOT transfer, stated before the numbers

* **The step budget.** 400 steps is **1.0 %** of refcv6's 40,284. This measures **early-training
  displacement**, which is not final-quality displacement. A weight that is admissible here can
  still be wrong at 40 k, and the reverse.
* **Training-seed variance.** The replicate shares A_w0's seed (§2).
* **Anything about whether the head HELPS.** That is `H-TACGOAL-1` and needs its own arm, its
  own eval, its four families and its tier stamp.

What **does** transfer, and is the reason this rig was chosen over the one the brief named: the
**architecture, the parameter count, the head width, the corpus, the label join and the loss
combination are identical to refcv6 arm C**, because this *is* arm C's trainer.
