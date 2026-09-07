# ERRATUM-1 to `PREREG_REFCV5_V2_LANDING.md` — the arm carries a lever that receives NO GRADIENT

**Raised 2026-09-07, hours after the prereg, while refcv5-v2 is still training.**
⛔ **NO CRITERION CHANGES.** `BAR-REFCV5V2-1` and `-2` stand exactly as written. This erratum
corrects the prereg's **description of the arm**, which was wrong in two places. It takes the
erratum form rather than an edit because the correction was written **after** seed-0 data
existed, and a pre-registration that is quietly rewritten after data is not a pre-registration.

## 1. ⛔ `--tac-goal-tok-head` IS PASSED, BUILT, AND RECEIVES NO GRADIENT

§1 lists `--tac-goal-tok-head` among the arm's levers. That is **true about the flag and false
about the lever.**

**MEASURED** (gradient probe on the real model, the trainer's own loss, one backward, under the
live argv; `…/Research/2026-09-07-v7-vocab-reach-census/raw/gradreach_live.json`):

| module | tensors | grads that are `None` | `grad_abs_sum` | verdict |
|---|---|---|---|---|
| `tac_goal_tok_head` | 2 | **2 of 2** | **0** | ⛔ **NOT WIRED** |
| `tac_goal_head` | 2 | 0 | **14.368** | ✅ GRADIENT REACHES |

`tac_goal_tok_head_built` is **`true`** — it is constructed, it is rollable, and in the live
config it carries **11,286 parameters that cannot learn**.

⭐ **THE NUANCE THAT MATTERS, AND THAT A SUMMARY LOSES: THERE ARE TWO HEADS, AND ONLY ONE IS
DEAD.** The factored `tac_goal_head` trains normally. ⇒ **the arm is NOT tactically blind**; what
is inert is the **22-token TOKEN head** specifically. Do not restate this as "refcv5-v2's tactical
head does not train".

⛔ **CONSEQUENCE FOR THE LANDING:** any claim attributing a refcv5-v2 improvement to *"the
22-token tactical vocabulary"* is **INADMISSIBLE** — those parameters never learned. The arm's
real levers against refcv4b are `--sel-refined`, `--sampler ddim`, the P14 fan ranking,
`--anchor-v0-conditioned` and `--nav-from-v7`. ⚠️ The launch record's phrase *"the 22-token
tactical vocabulary on a provably rollable head"* is exactly the trap: **rollable and trained are
different claims**, and only the first was ever demonstrated.

⚠️ **This CONFIRMS the already-registered `D-TACGOAL-TRAINER-SEAM-OPEN`; it does not discover
it.** What is new is (a) the **gradient** evidence rather than a code reading, and (b) that the
**live run is carrying the dead parameters right now**. Two sibling documents headline
*"supervised head"* / *"GAP CLOSED"* — true of the head, false of the trainer.

⭐ **WHY BOTH EXISTING GUARDS WERE GREEN, which is the transferable part.**
`assert_seams_are_built` asks *"is it built?"* — and it **is**. `effective_weights_stamp_v3`
enumerates **declared loss weights**, and this head **has no weight flag**, so it produces **no
row at all**. ⇒ **an instrument that enumerates weights cannot see a head that has no weight.**
Same family as the estimator rules and `e4af94f`: the guard is correct, and its **question is
narrower than the claim being hung on it**. Neither guard was wrong; both were asked the wrong
thing. Now pinned by `stack/tests/test_built_heads_receive_gradient.py`, which requires every
built module to show `p.grad is not None` or sit on a **literal** `KNOWN_UNWIRED` list that must
shrink.

## 2. ⛔ `--max-speed-input` IS NOT MERELY "NOT PASSED" — IT IS UNRUNNABLE ON v7.2

§1 says max speed *"belongs in its own arm as a single lever"*. The **attribution** argument
there is correct and stands. The **readiness** implication is wrong.

**MEASURED:** `speed_max_input` is present on **0 of 4,572** v7.2 train records and **0 of 147**
eval records, and the trainer **refuses the flag** rather than training on nothing. ⇒ it needs
**the v8 label blob, not a flag flip**, and `max_speed_cond_built` reads **`False`** in the live
probe.

⛔ So "the next arm carries `--max-speed-input`" is **not** a scheduling decision that can be
taken tonight; it is **gated on the v8 labels reaching the trainer's corpus**. Any plan that
omits this is planning an arm that cannot start. *(The flag itself is real and tested —
`a1d52e6`, `test_max_speed_input.py`, `test_max_speed_wiring.py`. The GAP IS THE DATA, and
"`max_speed_input.py` is dead code" remains FALSE.)*

## 3. What does NOT change

* **The bar.** `BAR-REFCV5V2-1` (`os - ha0_ext` separated **and** a 0.10 relative margin) and
  `BAR-REFCV5V2-2` (`os - ha`) are untouched. Neither depended on the token head.
* **The decision not to restart the run.** It is strengthened, not weakened: restarting to wire a
  head would add a lever mid-flight and cost ~20 h of training, and the seam's fix is **blocked on
  the `MANEUVER_WEIGHT` budget, which is an owner/PI call already registered.**
* **The three-variance discipline** of §4, and the inference-floor measurement in flight.

*Evidence: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-07-v7-vocab-reach-census/`
(`RESULT.md`, `raw/gradreach_live.json`, `raw/census.json`, `raw/MUTATION_LOG.md`).
Pin: `stack/tests/test_built_heads_receive_gradient.py`.*
