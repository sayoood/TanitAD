<title>LEDGER A5 - benchmarks and evaluation (driving, embodied)</title>

# LEDGER — A5 · Benchmarks + evaluation

`APPEND-ONLY. Charter DAILY_RESEARCH_CHARTER.md §3. Opened 2026-09-01.`
`Opened because TRACKS.md recorded A5 as "partial - no dedicated DEEP" since the charter began.`

---

## Entry 2026-09-01-01 - WorldRoamBench: trajectory-level metrics HIDE action-following failure

**Source:** `arXiv 2606.31672`, "WorldRoamBench: An Open-World Benchmark for Long-Horizon Stability of
Interactive World Models" (Xu, Sui, Gao, Shi, Yang, Liu, Sun, Sun, Pan, Jiang, Xu, Fan, Gao, Li,
Chen; submitted 2026-06-30, revised 2026-07-06).
**Evidence class:** PUBLISHED lib 2606.31672 **abstract-only - DECLARED.** Banked this pass (the only
new bank of the day).

> *"existing benchmarks evaluate action following only at trajectory level and ignore memory and
> interaction physics"*

Four dimensions: **(i) Action** - a per-frame action metric "exposing failures hidden by trajectory";
**(ii) Vision** - a segment-based drift metric "capturing non-monotonic mid-sequence collapse missed
by start-vs-end comparisons"; **(iii) Physics** - controllability-gated scoring over mechanics, optics
and 3D consistency; **(iv) Memory** - scene memory via transition-localized 3D point-cloud
reconstruction, subject memory via tracking + VLM reasoning.

Over **10+ open/closed-source models: "None reliably satisfies all dimensions."**

⭐⭐ **What it changes for us - an independent line arrives at our action-echo finding.**
"Failures hidden by trajectory[-level metrics]" is precisely `EVAL_DOCTRINE` §1.12 MEASURED:
open-loop S-curve reproduction **97.9 %** vs hold-action **0.0 %** vs closed-loop **~5 %**. A further
independent line now holds that trajectory-level open-loop scoring conceals action-following failure.

⚠️ **Two instruments we do not have:** (a) a per-frame action-fidelity metric; (b) a **segment-based**
drift metric. ⛔ **Our drift is currently read start-vs-end, which by their construction cannot see
mid-sequence collapse** - so every drift number we hold may be a lower bound.

**Risk:** built for open-world *interactive* world models, not driving; the physics/memory dimensions
may not port. **Abstract-only: no degradation curves or per-model numbers were readable, so nothing
here may decide a GPU-day yet.**

**Pre-registerable experiment (0 GPU, proposed unranked):** port ONLY the segment-based drift metric
into `taniteval` and re-score the banked v7 rollout dumps. **Committed outcome:** monotone drift =>
start-vs-end is adequate and the instrument stands; **non-monotone mid-sequence collapse => every
drift number we hold is a lower bound and the instrument must change.**

`Next in this track: full text for the segment-metric definition; then the per-frame action metric.`


---

## Entry 2026-09-02-01 — WorldRoamBench full text: the drift metric is now portable, and non-monotonicity is MEASURED

`PUBLISHED lib 2606.31672 · FULL TEXT READ 2026-09-02 (was abstract-only 2026-09-01) · unblocks backlog FS-1`

### The segment-based drift formulation, exactly

Partition the rollout into **N = 10** windows; take the **best-quality** and **worst-quality** windows;
report the **relative percentage change** between them.

> *"D = 0.05 means the worst 10% window of the video is 5% below the best 10% window in average score;
> unlike a fixed start/end comparison, this localizes the strongest dip wherever it occurs, so transient
> mid-rollout collapses that are later masked by recovery still contribute to D."*

Relative rather than absolute *"keeps drift comparable across models with different score magnitudes"*.
Instantiated on aesthetic and imaging score sequences to give two lower-is-better metrics.

### ⭐ The non-monotonicity FS-1 was written to test is MEASURED, not hypothesised

- *"A subset of models additionally show transient mid-rollout collapse, dipping sharply mid-sequence
  before partially recovering; **a start-vs-end comparison would treat them as stable**, but our
  segment-based best-vs-worst drift still localizes and penalizes the dip."*
- `minWM`: *"selective collapse, with its aesthetic score crashing after frame 150 even as imaging stays flat"*.
- `Matrix-Game 3.0`: *"progressive, accelerating decline driven by compounding autoregressive error accumulation"*.
- `HY-World 1.5` best: aesthetic drift **13.85**, imaging drift **15.91**. `Yume 1.5` *"starts strong yet
  accumulates 23.01 imaging drift"*.
- *"low drift and high average quality are correlated but not identical"* — frame-averaged quality cannot
  substitute for a stability measure.

### ⭐⭐ Second independent confirmation of our action-echo finding, now with numbers

> *"Models with high trajectory alignment (trajectory score above 85) can exhibit **below 65% per-frame
> strict action accuracy**, revealing a hidden failure mode in which a model eventually drifts."*

⇒ Three independent lines now agree that trajectory-level open-loop scoring conceals action-following
failure: ours (S-curve 97.9 % open-loop / 0.0 % hold-action / ~5 % closed-loop), Alpamayo, and this.
**Guideline S-1 (T1 is the burden of proof) strengthens.**

### ⚠️⛔ SCOPE LIMIT — this changes what FS-1 may port

Their drift is computed over **image-quality** scores (aesthetic / imaging). **Ours is a trajectory/latent
quantity.** ⇒ **Port the segment-based best-vs-worst FORMULATION, not the metric.** Quoting their
13.85 / 23.01 against our drift numbers would be the `df` / `step_s` scope error in a new costume.

⇒ FS-1 is **unblocked and executable at 0 GPU**: N=10 segments, best-vs-worst, relative change, over the
banked v7 rollout dumps. **Committed outcome unchanged: monotone ⇒ the instrument stands; non-monotone
⇒ every drift number we hold is a LOWER BOUND and the instrument must change** — including the MM-E1 EMA
read (0.4531 -> 0.36-0.40) currently deciding a recipe.

`Next in this track: C4 leaderboard numbers landed 2026-09-02 (see LEDGER_C1_releases.md C4 entry) but`
`are NOT admissible into a comparability table until backlog row 3 (portfolio) and D-EPDMS-FAM land.`
