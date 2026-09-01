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
