# T1 on v7: the first runs, and exactly what they can and cannot say

**Written** 2026-08-26 20:05 · **Author** Master Mind · **Tier** T1 (the doctrine's
primary tier) · ⚠️ **Corpus: NON-PARITY.** These are pipeline results, not
comparable numbers. The parity run is armed on Thor.

---

## 1. What now works

T1 executes end-to-end on a v7 arm: **STRICT load, 256×640 cylindrical geometry,
`grounding.step['op']` decoder, 40 episodes × 173 windows in 202 s.**

| | 4 episodes | **40 episodes** |
|---|---|---|
| S-rate (masked) | 0.0 | **0.0351** |
| lag | UNAVAILABLE | −0.15 ⚠️ *n=4 of 40 episodes with signal* |
| four families | — | 3 of 4 populated |

⭐ **3.5 % closed-loop S-curve reproduction sits in the doctrine's banked regime
(~5 % closed-loop, against 97.9 % open-loop).** A different architecture reproducing
the action-echo finding is a meaningful sanity check on the pipeline — **not** a
comparison, because the corpus is non-parity.

---

## 2. ⛔ What the numbers may NOT be used for — the instrument says so itself

`four_families` returns **`_longitudinal_claim_admissible: False`**, and the rule
travels with it (Sayed, 2026-08-16):

> *"A LONGITUDINAL number emitted while `_longitudinal_claim_admissible` is False is
> a **fidelity diagnostic, NOT a longitudinal capability result**, and must not be
> presented as one."*

⚠️ **I was one step from breaking that.** The block reports `speed_mae 10.94 m/s`,
`speed_bias −10.62 m/s`, `heading_mae 95.6°` — and my drafted reading was *"at T1
the model does not drive at all."* **The gate stopped it**, and the gate is right:
the anti-echo condition (beat hold-v0, separated) is **undischarged**, so those
numbers are unverified against the baseline that would give them meaning.

⚠️ **And my first diagnosis of WHY was also wrong.** I inferred *"the model is being
fed no speed"* from `speed_bias ≈ −(corpus mean speed)`. **It is not:**
`roll_closed_grounding` receives `v0`. What lacks `v0` is the **DUMP** — inspected
directly, a record holds only `g`, `cl`, `ws`, `eid`, `clip_index`. The rollout had
the speed; the **scorer** cannot build its hold-v0 baseline. ⇒ **A missing value in
an artefact is not a missing value in the computation, and the difference is the
whole claim.**

---

## 3. The three gaps to a COMPLETE T1, each named by the tool

| gap | consequence | fix |
|---|---|---|
| **`v0` absent from the dump** | anti-echo undischarged ⇒ **no longitudinal capability claim, ever, from this dump shape** | write `v0` per window into the dump — a code change in `t1_eval`'s dump writer |
| **no lead-agent track** | `distance_keeping` UNAVAILABLE — *"pass `lead=`"* | `taniteval/tools/build_lead_block.py` **exists**; build it row-aligned to the stride-1 grid |
| **strategic** | UNAVAILABLE — *"a world-model FIDELITY pass does not traverse the hierarchy"* | needs map-derived option sets (`strategic_gt.py` → `taniteval.strategic_optionset`). ⚠️ The tool states explicitly that **a route label read off the ego's own future yaw is NOT a substitute** — it cannot tell whether the map admitted a choice. |

⇒ **88.7 % of the programme's oracle gap is longitudinal, and the longitudinal
family is exactly the one that cannot yet produce an admissible number.** That is
the single highest-value fix in the eval stack.

---

## 4. What the armed Thor run will and will not deliver

The chain evaluates `postrain30k` and `postrain30k_freeze` at T1 on the **registered
parity corpus** when the crossed cell lands. ⚠️ **It will hit the same three gaps**,
so it delivers:

- ✅ the programme's **first parity T1 numbers** — S-rate, lag, lateral, tactical;
- ✅ a **trained-vs-frozen T1 comparison** on those families;
- ⛔ **no admissible longitudinal claim**, and **no strategic family**.

⚠️ **Stated in advance so the partial result is not later presented as a full one**,
and so the T1 comparison is not mistaken for the crossed cell's pre-registered T0
criterion (drift < 0.45 **and** held-out `nrmse` ≤ 0.893), which it does not replace.
