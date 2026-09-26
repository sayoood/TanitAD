# ⛔ The P panel as specified cannot produce its own PRIMARY metric — found before any card was spent

**Agent:** TanitAD_TrainingFlyWheel · **2026-09-20** · **GPU used: ZERO** (chain paused at the PI's
request; this is a source-and-artifact read).

> ## ⛔ RETRACTED IN PART — read this before §3 (2026-09-20, same day)
>
> **§0–§2 (the gap) STAND and were verified independently at source by the Master Mind.**
> **§3's REMEDY IS WITHDRAWN.** Ruling: do **NOT** add `--eval-window-dump`; **re-score the
> banked checkpoints** instead (`arm_box_score.py`, CPU, any checkpoint, emitting near_forward
> AND all_360 with n).
>
> **Why §3 was wrong, and it was wrong on its load-bearing word.** I called the flag *"free"*.
> My four tests prove the dump does not change **TRAINING**; they cannot make the **argv
> IDENTICAL** to A8's, and §3.1 asks for *identity*, not equivalence. And the argument I missed
> settles it: **A8 is finished and can never acquire the flag retroactively**, so a re-score has
> to happen anyway to give the baseline a near-forward number — adding the flag would create a
> **SECOND route to the same quantity**, present on the new arms and absent on the baseline.
> That is the cross-**METHOD** version of the cross-**POPULATION** error this document exists to
> prevent: the same mistake, one level up. It would also have been insufficient alone, since
> `P1`/`P3`/`P5` carry the verdicts and share the gap, so the flag would have had to go on every
> arm — changing the whole panel relative to the baseline it chains behind.
>
> ⭐ **A further correction to §1's `n`, MEASURED after this file was written**
> (`raw/bar_unit_probe.json`): the bar's **129 near-forward pairs** come from **24 windows**,
> which come from **21 DISTINCT EPISODES** — and one of those 24 windows overlaps another by
> **6 of its 8 frames** (same-episode frame gaps 2, 20, 89). So any pair-level interval is
> pseudo-replication, and a window-level one is still an under-correction: **the unit is the
> EPISODE**. `box_quality.summarise(..., eid=...)` already clusters that way.
>
> ⭐ **AND THE BAR ITSELF HAS BEEN RE-READ** (2026-09-20, by the canonical tool). `box_quality.py`
> on A8 `ckpt_5000`, `--n 24`, CPU, controls green, reports near-forward
> **L1 5.9539 m** (|dx| 2.9826, |dy| 2.9713), **CI95 [4.8922, 7.1084]** — episode-clustered,
> n = 129 pairs / **24 windows / 21 episodes**, n_boot 2000. The banked **6.06** (|dx| 3.09) is
> **superseded**: two independent implementations agree on 5.95 to four decimals, and the
> discrepancy is confined to **|dx| alone** (0.107) while |dy| and the whole all-360 population
> match. ⛔ The CAUSE is unseparated — a different checkpoint or n behind the banked read, a
> transcription, or a probe-side bug later fixed — and nobody is asserting one.
> ⚠️ Read honestly in BOTH directions: **6.06 sits INSIDE the new interval**, so this is a better
> estimate of the same quantity, not a refutation; and **nothing is rescued** — `meets_bar` is
> false either way, 5.95 being ~3× the 2 m target. Every `6.06` below should be read as `5.95
> [4.89, 7.11]`.
>
> §5's durable fix (the population label inseparable from the number) is unaffected and stands.

---

## 0. The finding in one line

**`PREREG_PERCEPTION_BOX_QUALITY` judges every arm on the NEAR-FORWARD centre error, but the P
arms' configuration emits only `eval_box3d_centre`, which is the 360° aggregate.** The floor and
the verdicts would therefore live on **different populations**, and the two differ by roughly
**2×** on the banked read — so this is not a rounding matter, it is a cross-population comparison.

## 1. The three probes that establish it

| # | probe | result |
|---|---|---|
| 1 | what `eval_box3d_centre` measures | `agent_slots.py:577` — `acc["centre"] += (pb[:,:2] - tb[:,:2]).abs().sum()` over **every matched row `r`**, then `/ n["centre"]`. **No population filter.** ⇒ it is the `all_360` number |
| 2 | what the prereg judges on | `PREREG…:22-23` — near-forward `x ∈ [0,60], |y| ≤ 16` **6.06 m** (⚠️ re-read as **5.95 [4.89, 7.11]**, see banner), **"⭐ < 2 m — the primary bar"**; all-matched 12.04 m "reported, secondary". `P1`/`P3`/`P5`'s SUPPORTED clauses are all written on **near-forward** |
| 3 | can the arms produce near-forward? | `box_quality.match_pairs` needs per-window pred/target pairs ⇒ `--eval-window-dump`. The P arms copy **A8's argv verbatim**, and A8's 82 tokens contain **no dump flag at all** |

⇒ With the panel as specified, `P0`/`P0b` produce **no near-forward number**, so the floor can only
be computed on the secondary population, while `P1`/`P3`/`P5` are graded on the primary one.

## 2. Why this is a blocker and not a nuisance

The floor's whole purpose is the sentence *"no arm may be called SUPPORTED by a margin smaller than
the floor"*. That comparison is only meaningful when the floor and the margin are **the same
measurement**. A near-forward improvement of 0.4 m judged against an all-360 floor of 0.25 m is not
a comparison at all — it is the wrong-scope error this programme keeps logging, with the scope
being the **population**. `P3` makes it sharper still: it *changes the target population*, so an
all-360 floor is doubly unfit to judge it.

## 3. ⛔ WITHDRAWN — the fix I proposed, and why it was NOT free (see the banner)

**Add `--eval-window-dump` to `P0` and `P0b`.**

⭐ **It is provably inert for training**, and the proof is already banked — I built it earlier this
session when I found and fixed the dump's train-mode defect. `stack/tests/test_eval_window_dump_mode.py`:

* `test_the_dump_pass_runs_in_EVAL_mode`
* `test_the_dump_pass_does_not_touch_the_tactical_prior`
* `test_the_dump_changes_NOTHING_about_training`
* `test_the_dump_rows_REPRODUCE_the_aggregate_eval_row` (the dump and the aggregate agree)

The dump pass runs under `_RngIsolated(device, None)` — a fork **without reseeding** — so every RNG
stream is left exactly where it was found, and the model is restored to `train()` afterwards. The
training trajectory is therefore identical with and without the flag.

⇒ Adding it changes the *argv* relative to A8 but **not the experiment**. §3.1's *"A8's exact
flags"* is satisfied in the sense that matters (the same run), and the alternative — a floor on a
population nobody grades against — is strictly worse.

⚠️ **A7 already carries `--eval-window-dump`**, so this also makes the P arms consistent with the
panel they chain behind.

## 4. ⚠️ What I have NOT established, stated rather than guessed

* **The dump's wall-clock cost is UNMEASURED.** A8 runs **5** evals (`--eval-every 1000`,
  `--eval-batches 500`); A7 runs **1**. If the dump is a second pass over the eval batches, P0/P0b
  could pay it five times. ⛔ I will not quote a figure I have not measured. The cheapest way to
  get it: read the gap between consecutive `elapsed_s` rows around an eval in **A7-IN-s0**'s
  metrics once that arm reaches step 2,000 — A7 has the dump, so the cost is already being paid
  there and can be read for free.
  *(A8's own row gaps at eval steps were 36–55 s, but its per-step distribution is skewed —
  median 4.76 s against a mean of 7.65 — so a clean eval/step split from that file is not
  trustworthy, and I am not using it.)*
* Whether the PI/MM prefer to instead **re-score the checkpoints** in a separate eval pass (which
  keeps A8's argv byte-identical, at the cost of extra GPU afterwards). That is a real alternative
  and it is their call, not mine.

## 5. What I changed, and what I did NOT

**Changed** — `stack/scripts/p_floor.py` now carries the population label *inseparably* from the
number: `HEAD_POPULATION = "all_360 (SECONDARY) - NOT the near-forward primary the verdicts use"`
appears in the JSON **and** on the rendered line, pinned by
`test_the_headline_carries_its_POPULATION_in_json_and_text`. A reader cannot see the figure without
seeing which population it belongs to.

⛔ **NOT changed** — the pre-registration, and the arms' flags. Adding a flag to a pre-registered
arm is a specification decision, and this one is the PI's/Master Mind's. The panel is paused, so
there is no time pressure to decide it badly.

## 6. Manifest

| artifact | state |
|---|---|
| `stack/scripts/p_floor.py` | population label + guard, staged |
| `stack/tests/test_p_floor.py` | 14 passed bare |
| `P_PANEL_DUMP_GAP.md` | this file |
