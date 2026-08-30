# MM-E15 — ⭐⭐ THE HORIZON LADDER IS MISALIGNED WITH THE LABEL BANDS BY 1–2 ORDERS OF MAGNITUDE

**Measured** 2026-08-30 · Master Mind · **Tier** T0-DIAGNOSTIC (a structural
property of the recipe, not a driving claim) · **Register** D-HORIZON-LADDER.

## The question I was not asking

I opened the v7.2 label blob to answer a wiring question — does the incumbent's
`_check_block` work on a v7 record. It does not, for reasons already filed. But the
`bands` field answered a much larger question that nobody had asked.

## What the labels say, and it is constant across the whole corpus

**MEASURED** on the canonical `s2_labels_v7.2_train.jsonl.gz`
(md5 `0ff902130ce76886b8a925eceed9e3a5`), **all 4,572 records**:

| band | value | distinct values across 4,572 records |
|---|---|---|
| `bands.operative_s` | `[0.0, 2.0]` | **1** |
| `bands.tactical_s` | `[2.0, 6.0]` | **1** |
| `bands.strategic_s` | `[8.0, 30.0]` | **1** |

⭐ These are **forward-looking horizon ranges**, not offsets bracketing `t0`, and I did
not have to assume that — the record's own args corroborate it independently:

```
a_str.args.by_time_s   n=1,624   min 8.0   p25 8.0   median 12.4   p75 19.8   max 30.0
a_str.args.within_m    n=1,624   min 0.2   median 73.4             max 507.1
```

The strategic action's referent lands **exactly inside `strategic_s`** — **100.0 % of
1,624 records**, median **12.4 s** ahead. *(Sample: `PREPARE_TURN_R_FOLLOW_ROUTE`,
"106.5 m / 11.1 s ahead, not yet begun".)* A field agreeing with a band it was not
derived from is what makes the reading trustworthy — the cross-check that settled the
cylindrical FOV.

⚠️ **BUT THE SAME CHECK DOES NOT CONFIRM THE TACTICAL BAND, AND I WILL NOT LET THE
STRATEGIC RESULT CARRY BOTH.** `a_tac` has no `by_time_s` — only `within_m` (present
on all 4,572) and `v_target_ms`. The only available proxy is a ratio **I constructed,
not a field the record provides**:

```
within_m / v_target_ms   n=4,067   p10 6.1   median 6.8   p90 12.4     inside [2,6]: 0.0 %
```

⇒ **0 % inside the band.** ⚠️ The likelier reading is that my proxy is wrong, not the
band: `v_target_ms` is the speed to *reach*, not the current speed, so an ego
decelerating into the manoeuvre covers `within_m` faster than the ratio implies — and
a median of 6.8 s against a far edge of 6.0 s is **adjacent, not disjoint**. But
"likelier" is not measured, so the honest status is:

* `strategic_s` — **CONFIRMED independently** (100 % of 1,624).
* `tactical_s` — **band read from the schema only, NOT independently confirmed.**

⭐ **The ladder conclusion is robust to this either way**, which is why the finding
survives the failed check: whether the tactical referent is 2–6 s (schema) or ~6.8 s
(proxy), it is **2.5×–8.5× beyond the 0.8 s rollout** and 10×–34× beyond effective
reach. The proxy, if anything, makes the gap **larger**. ⇒ the uncertainty moves the
number, not the verdict.

### ⭐⭐ SETTLED PROPERLY — the corpus carries its own event times

`manoeuvre_sequence` carries **explicit timestamps** (`t_start_s`, `t_end_s`,
`dyaw_deg`, `radius_m`, `v_min_ms`), so the proxy was never needed. ⛔ **SCOPE, AND IT IS LOAD-BEARING: every count in this document is TRAIN-ONLY**
(`s2_labels_v7.2_train.jsonl.gz`, 4,572 records / 3,464 manoeuvres). The
`V72_MANIFEST` `band_coverage` block counts **train+eval** (4,719 / 3,580). MEASURED
split: train 53 unassigned + eval 1 = **54**. ⇒ the two documents legitimately differ
(53 vs 54, 440 vs 453) and **neither is wrong**. ⚠️ This is the manifest note's own
lesson one level down: a count must name not only its **convention** but its
**corpus**. Quoting across the two without checking scope manufactures a fifth and
sixth "right answer".

**MEASURED over
all 4,572 TRAIN records / 3,464 manoeuvres:**

```
strategic by_time_s equals SOME manoeuvre t_start_s   962 / 1,255   (76.7 %)
t_start_s   p10 0.0   p25 3.3   MEDIAN 12.5   p75 22.6   p90 29.0   max 33.7
```

⭐ **The `a_str.args.by_time_s` field IS the manoeuvre's start time** on 76.7 % of
records where both exist — so the strategic label points at a real, timestamped event.
And the median manoeuvre start (**12.5 s**) matches the median `by_time_s` (**12.4 s**)
from an independent field. ⇒ **the central claim is now carried by the corpus's own
event timing, not by my reading of a schema constant**: the events these labels
describe are typically **~12.5 s ahead**, which is **15.6× beyond the 0.8 s rollout**
and **~62× beyond effective reach**. That is MM-E15's number, re-derived.

⚠️ **And the tactical band is populated after all** — my `within_m / v_target_ms` proxy
was simply the wrong instrument, as suspected:

| band | manoeuvres starting in it | share |
|---|---|---|
| operative `[0,2)` | 687 | 19.8 % |
| tactical `[2,6)` | 463 | **13.4 %** |
| ⚠️ **GAP `[6,8)`** | **176** | **5.1 %** |
| strategic `[8,30]` | 1874 | 54.1 % |
| ⚠️ **beyond 30 s** (`t >= 30.0`) | **264** | **7.6 %** |

⛔ **THE BANDS ARE NOT A PARTITION — THERE IS A HOLE AT `[6,8)` AND NO UPPER
CATCH-ALL.** 440 manoeuvres (**12.7 %**) start outside every band.

### ⚠️ A DISCREPANCY I RAISED AND THEN REFUTED MYSELF — RECORDED BECAUSE THE SHAPE RECURS

I first wrote that `bands.unassigned_manoeuvres` (**53 records, 53 entries**)
disagreed by **~8×** with the 440 that a literal band reading implies, and called it
an open question gating §3.1. ⛔ **That was wrong, and measuring it took two minutes.**
The two fields count **different things**:

```
of the 53 unassigned entries:  in the [6,8) hole 53   >30 s 0   inside some band 0
```

**All 53 land in the hole, none anywhere else.** And the example shows why —
`manoeuvre_sequence` has `t_start_s 5.3 → t_end_s 7.7` (starts in tactical, ends in
the hole) while the unassigned entry is `t_start_s 6.0 → t_end_s 7.7`: **a fragment.**

⭐ ⇒ **The builder SPLITS manoeuvres at band edges** and records the portion falling
in the gap. `unassigned_manoeuvres` tracks *straddling fragments in `[6,8)`*; my 440
counted *whole manoeuvres by start time*, 264 of which start beyond 30 s. Neither
number is wrong; they are not the same quantity, and my "disagreement" assumed they
were. ⭐ **This also shows the `[6,8)` gap is DESIGNED** — a deadband deliberate enough
that the builder wrote accounting for it — not the oversight I took it for.

✅ **AND THE >30 s ASYMMETRY IS ANSWERED — correctly, and from the loader rather than
from anyone's intent.** `emit_one` reads each track with
`max_s = RAW_T0_S + LOOKAHEAD_S + 5.0` = **35 s**, deliberately 5 s past the strategic
edge, so that a manoeuvre *straddling* 30 s still has the poses to compute its clipped
`dyaw` **at** the edge — without the margin, `dyaw_between(a, 30.0)` runs off the end
of the track and the strategic upper boundary is ill-defined. ⭐ My measured
`t_start_s` **max of 33.7 sits inside that 35 s window**, which confirms the margin is
live rather than merely intended. ⇒ manoeuvres starting past 30 s are a **by-product
of making the 30 s edge well-defined**, not content the label claims.

⇒ **The asymmetry is correct as designed:** `[6,8)` is a hole **INSIDE** the horizon
that no layer owns — a real hierarchy blind spot, hence the accounting; `>30 s` is
**OUTSIDE** the horizon — beyond what the label describes, not a gap within it.
Different things, and only the first is a defect.

⚠️ **One boundary case survives:** **9 manoeuvres start at exactly `30.0`** (train 8 +
eval 1). A start-time census counts them strategic (`t <= 30`); the emitter's overlap
test is strict (`min(end,30) > max(start,8)`), so a zero-width intersection at the
closing edge is empty and they are claimed by nothing.

⛔ **AND I COMMITTED THE SAME DEFECT WHILE DOCUMENTING IT — recorded because that is
the point.** This document quoted **264** over-horizon manoeuvres in the table above
and **256** in the reconciliation below, both train-only, both mine, and I did not
notice. MEASURED:

```
t >= 30.0 : 264      (my band census: tuple (30.0, 1e9) tested a <= t < b)
t >  30.0 : 256      (my per-split census: strict >)
t == 30.0 :   8      the difference — and EXACTLY the boundary case
264 - 8 = 256  ✓
```

⭐ Two censuses, one investigation, two conventions, **neither named** — inside a
write-up whose central lesson is that a count must name its convention. The boundary
case the DataFlyWheel found independently in the emitter is the *same* 8 that
separates my own two numbers. ⇒ this failure is not carelessness, it is **how easy the
convention is to leave implicit**: `>=` and `>` were both the obvious thing to write,
ten minutes apart. Any over-horizon count in this programme must carry its operator.

⚠️ **And a fourth count of "the same thing":** the emitter's own census finds **332**
manoeuvres overlapping no assignable band, against my 440 by start-time and its 53
reported unassigned. Three different questions, three right answers — 268 start past
30 s, 55 overlap only the phantom operative band, 9 sit on the 30.0 edge (overlapping,
so the sets are not disjoint). ⭐ That this keeps happening is itself the lesson: with
bands, **"how many are unassigned" is not one question**, and any count must state
which convention it used.

⚠️ One more fact for anyone training on these labels: **mean 0.76 manoeuvres per
record** — roughly a quarter of records describe **no manoeuvre event at all**.

## What the model reaches — MEASURED from `o1ctrl30k/config.json` on Thor

```
dt 0.1 s · window 6 (= 0.6 s context) · horizons [1,2,4] (= 0.1 / 0.2 / 0.4 s)
o5_k 8  (= 0.8 s, the LONGEST trained rollout) · o1_k 4 (= 0.4 s) · in_channels 9
```

⚠️ And the *effective* reach is shorter than the nominal one. **MM-E10 measured the
predictor to collapse beyond h=1**: `max|h1−h2| = 1.211` against
`max|h2−h4| = 0.00179` — **~680× smaller**. It emits one long-horizon guess for
everything past one tick, so meaningful imagination is **≈ 0.2 s**.

## ⭐⭐ The ladder, side by side

| level | label referent | token labels? | longest rollout (`o5_k`) | shortfall | vs *effective* ~0.2 s |
|---|---|---|---|---|---|
| operative | 0–2 s | ⛔ **NONE** | 0.8 s | covers **40 %** of the band | 10 % |
| tactical | 2–6 s | `a_tac`/`g_tac` | 0.8 s | **2.5× short of the NEAR EDGE** | **10×** |
| strategic | 8–30 s (median 12.4) | `a_str`/`g_str` | 0.8 s | **15.5× short at the median** | **62×** |

⛔ **NO ROLLOUT IN THE v7 RECIPE REACHES EVEN THE NEAR EDGE OF THE TACTICAL BAND.**

⭐⭐ **AND THE THIRD COLUMN IS THE SHARPEST FORM OF THE FINDING.** MEASURED
independently by me over all 4,572 records: the top-level keys are `a_tac`, `g_tac`,
`a_str`, `g_str` and **there is no operative field of any kind — 0 records carry
`a_op`, `g_op` or any operative token** — while `bands.operative_s [0,2]` is declared
in every record. ⇒

> **The ONLY band the predictor can partially reach is the one with NO token
> supervision. Every band that HAS token labels lies beyond its reach.**

⚠️ **This is NOT a missing layer, and it must not be reported as one.** The operative
level *is* supervised — by trajectory regression (the `ade_dense_m` / `fde_last_m`
family the T1 harness scores), which is **continuous, not categorical**. A hierarchy
whose lowest level emits a trajectory has nothing to put a discrete token on. So the
absence of `a_op` is **expected and correct**, and `bands.operative_s` is a statement
of the layer's temporal *scope*, not a promise of token labels.

⛔ **What IS a live trap** — and it is the DataFlyWheel's catch, not mine: a consumer
computing per-layer supervision from `bands` reads the operative layer as **declared
but empty**, and would conclude the layer is untrained. Same family as the
`glob(...)[0]` defect fixed today: **a structure asserting a property it does not
provide.** ⇒ documentation fix, in the manifest, not a code change.

## What this does and does not license

⚠️ **It does NOT say the heads cannot work.** A classifier can predict "turn right in
~12 s" from present visual evidence without imagining 12 s of future; people do it.
The bands are label *validity* windows, not required prediction horizons.

⛔ **It does say the MECHANISM THE THESIS ATTRIBUTES TO THEM IS UNAVAILABLE.** The
programme's claim is multi-hierarchy reasoning where **each planner predicts via
imagination** (PI, three-planner directive: the strategic planner gets its *own*
predictor on a strategy-only latent subspace). Selection-by-imagined-consequence
requires the imagination to reach the consequence. At 0.2–0.8 s against a 2–6 s
tactical referent and a 12.4 s strategic one, **the tactical and strategic heads are
currently feed-forward classifiers on the present latent — not planners.** That is a
horizon-ladder defect in the architecture, not a defect in the labels.

⭐ **AND IT RETRODICTS A RESULT WE ALREADY HAVE.** The first T1 read found S-rate
straddling its own control on all three arms (0.2807 / 0.0702 / 0.1404 against
0.2105 / 0.1579 / 0.2105) and we declared it noise, correctly, on the strength of two
T0-indistinguishable arms landing on opposite sides. **A strategic readout whose
referent sits 62× beyond the model's effective imagination is *expected* to read at
chance.** This does not prove the mechanism — it removes the puzzle.

## ⛔⛔⛔ CORRECTION 2026-08-31, PI-FLAGGED — I MEASURED AGAINST THE WRONG TARGET

**The PI: *"we said clearly that the prediction must be up to 6 seconds, this was
already included in v1.7 etc."* They are right, it is binding, and it is documented.**
Everything below this section measured our rollout against the trainer's `--o5-k`
**default of 20**. That was the wrong reference and it **understated the gap**.

**§4b, "the binding 6-second horizon — the clause that reshaped v6"** (`Project
Steering/Reports/2026-08-15-2200-campaign-science-addendum.md` §1.5), verbatim:

> *"Every planned trajectory spans up to 6 s — covering BOTH the operative and the
> tactical horizon in one kinematically consistent rollout."*
> *"a 60-step control sequence (a, κ) @10 Hz integrated through ONE unicycle rollout
> 0→6 s — never two stitched trajectories."*
> **"Emission heads scale k=20 → k=60."**

⛔ **So `k=20` is the SUPERSEDED value, and §4b says so explicitly.** My recommendation
to "restore `--o5-k` to its default 20" pointed at the number the programme had already
retired. The binding target is **k=60 / 6.0 s**, and it is already in the code:
`v6.py:153` `PLAN_STEPS = 60`, `:155` `HORIZON_S = PLAN_STEPS * DT # 6.0 s`.

⭐ **AND THE THREE BANDS ARE NOT MY DISCOVERY — THEY ARE THIS SPEC.** §4b states
*"0–2 s is the operative band, 2–6 s the tactical band"*. The label bands I measured
**implement a decided design**, they do not reveal one. Presenting them as a finding
was wrong; what is genuinely new is only the *gap* between them and what we trained.

### ⛔⛔ THE REAL NUMBER: WE TRAIN 1.7 % OF THE BINDING HORIZON

MEASURED on the O1 arm's own snapshots, **all 8, steps 5,000 → 22,500**:

```
step     |W1|      |W2|       |W4|      | Δ from previous snapshot
5000    3.7283   0.026154   0.026113   |   --        --          --
7500    6.0255   0.026154   0.026113   | 2.897e+00  0.000e+00  0.000e+00
...
22500   7.7516   0.026154   0.026113   | 3.203e-01  0.000e+00  0.000e+00
```

⛔ **The h=2 and h=4 heads are BIT-IDENTICAL across 17,500 steps — Δ exactly
`0.000e+00`.** Not "small gradient": **zero gradient**. They have never been updated.
This is stronger than MM-E14's "untrained" and it holds **with `w_o1_ctrl 1.0` in
force**, so it is a property of the wiring, not of any objective.

⇒ `--horizons 1 2 4` is effectively `[1]`. The **actually trained** prediction horizon
is **h=1 = 0.1 s**:

| | horizon | vs the binding 6.0 s |
|---|---|---|
| §4b requirement | 6.0 s (k=60) | — |
| nominal rollout `o5_k 8` | 0.8 s | **13 %** |
| `o5_k 20` (the superseded value) | 2.0 s | 33 % |
| **actually trained (only h=1 gets gradient)** | **0.1 s** | **1.7 %** |

⚠️ **AND THE PROGRAMME ALREADY SUSPECTED THIS, a week before I measured it.**
`PLAN_TO_THE_GOAL_2026-08-24.md:159`: *"The 0.6 s horizon may be too short for actions
to matter at all."* That note now has a mechanism and a worse number — it is not 0.6 s,
it is **0.1 s**. ⭐ And the same failure class was caught once before: the 2026-08-12
report records that inheriting `plan.max_horizon = 20` *"would have made the 6 s horizon
structurally untrainable"*. It was caught there and missed here.

⚠️ **Scope, so this is not over-claimed:** §4b binds the *planned trajectory* and the
*emission heads* to 6 s / k=60. Whether the world-model rollout term `o5_k` must equal
60 is a design question I am not settling here. The defensible statement is narrower and
sufficient: **a world model trained to imagine 0.1 s cannot support a planner required
to roll to 6 s**, and no objective on top can supply reach the trunk was never trained
to have.

⇒ **The `o5k20` arm as prepared is aimed at the retired number and should not run as
specified.** Corrected recommendation in §"recipe consequence" below.

## ⛔ AND THE LADDER IS SHORTER THAN IT WAS DESIGNED TO BE — WE SHORTENED IT
*(⚠️ superseded by the correction above: the reference here is the `--o5-k 20` default,
not the binding 6 s. Kept because the three-flags-below-default pattern still stands.)*

**MEASURED on Thor's authoritative stack** (`/home/nvidia/TanitAD/stack/scripts/
train_v6_staged.py`, the tree these checkpoints were trained with — MM-C12
discipline; the off-Drive mirror agrees line-for-line):

```
:6971   ap.add_argument("--o5-k", type=int, default=20)     every arm ran  8
:6829   ap.add_argument("--o1-k", type=int, default=10)     every arm ran  4
```

⭐ **`o5_k` default 20 at `dt 0.1` is 2.0 s — EXACTLY the operative band's far edge
`[0.0, 2.0]`.** That is not a coincidence to wave at: the code's default was set to
span the operative horizon the labels define. **We ran 8 — 40 % of it.** `o1_k`
likewise: designed 1.0 s, run 0.4 s.

⛔ **THIS IS THE THIRD INSTANCE OF ONE PATTERN IN THIS CAMPAIGN, AND THE PATTERN IS
THE REAL FINDING:**

| flag | code default | every arm ran | found by |
|---|---|---|---|
| `--w-o1-ctrl` | **1.0** | **0** | MM-E10 → MM-E11 |
| `--o5-k` | **20** (2.0 s) | **8** (0.8 s) | here |
| `--o1-k` | **10** (1.0 s) | **4** (0.4 s) | here |

Each one alone reads as a sensible economy on a ~19 M tiny rig. Together they mean
**the arms we have been drawing architectural conclusions from were never running the
intended recipe** — and the aggregate was never priced. ⚠️ I am not claiming the
reductions were wrong: a deliberate speed choice is legitimate. I am claiming that
*"the predictor collapses past h=1"* and *"the strategic readout is at chance"* were
measured on a configuration that is **below its own design point in three places at
once**, and that has to be stated wherever those results are quoted.

⭐ **THE CHEAP, IMMEDIATE ACTION: restore `--o5-k 20`.** One flag, one variable, tiny
rig, ~8.6 h — and it moves operative-band coverage from **40 % to 100 %** without any
architectural change. It is the obvious arm to queue behind MM-E11.
⛔ **Do NOT touch the live `o1ctrl30k` run** (step 10,400/30,000, `w_o1_ctrl 1.0`
confirmed in force). MM-E11 is a one-variable pre-registration and must finish as
registered; the `o5_k` restoration is the NEXT arm, not an amendment to this one.

## The recipe consequence — and the cheap version

The naive fix is to extend the rollout: tactical needs `o5_k ≥ 20` (2 s), strategic
`≥ 80` (8 s) at `dt 0.1`. ⛔ That is 10× the current rollout compute and, per MM-E10,
would extend a predictor that already collapses past one tick — **more of a rollout
that does not work.**

⭐ **THE HIERARCHY IS THE ANSWER, AND THIS REFRAMES WHAT IT IS FOR.** We have treated
the hierarchy as abstraction of **content** (operative trajectory / tactical
manoeuvre / strategic route). The bands say it must equally be abstraction of
**TIME**: each level should roll out at **its own dt**. A strategic predictor
stepping at ~1 s reaches 12.4 s in ~12 steps — the same rollout budget the operative
level spends on 1.2 s. That is exactly the PI's *"strategic gets its own predictor on
a strategy-only latent subspace"*, and this measurement gives it a **quantity**: the
temporal ratios the levels must span are **1 : ~3 : ~15**, read off the label bands
rather than chosen.

## ⚠️ Limits, stated

* The ladder comparison is **arithmetic on two MEASURED sets** (label bands; trainer
  config). The claim that this *causes* the weak strategic readout is **HYPOTHESIS** —
  it is consistent with the T1 S-rate result and with MM-E10, and it is not tested.
* The discriminating experiment is a **temporally-strided predictor** at one level
  (tactical, `dt_eff` 0.5 s, reach 4 s) against the incumbent, one variable, on the
  tiny rig — pre-registration to follow.
* `by_time_s` is present on **1,624 of 4,572** records (35.5 %); the quantiles are
  over those. The bands themselves are on **all 4,572**.
* ✅ **`tactical_s` RESOLVED** — the band holds 13.4 % of manoeuvre starts. The
  earlier 0 % came from `within_m / v_target_ms`, a ratio I constructed; the corpus's
  own `t_start_s` was the right instrument and was there all along. ⭐ The sequence is
  the point: the strategic check passed, I tested the tactical one anyway, it failed,
  I recorded the failure rather than shipping one verified level as two — and then the
  real instrument settled it. Had I not run the failing check I would not have gone
  looking for `manoeuvre_sequence`.
* ✅ **The 53-vs-440 "disagreement" was mine and is RETRACTED** — the fields count
  different things (straddling fragments in the `[6,8)` deadband vs whole manoeuvres
  by start time). ⭐ The lesson is the one the programme keeps relearning: **two
  numbers about "the same thing" that differ by ~8× are usually two different
  things.** I nearly sent it to the builder as a defect; the two-minute check is what
  a discrepancy is owed before it becomes someone else's work.
* ⚠️ **Genuinely open, and narrow:** the 264 manoeuvres starting beyond 30 s are not
  tracked as unassigned, while the `[6,8)` hole is. Probably correct — beyond the
  horizon is not the same as unassigned within it — but the asymmetry wants one line
  from the builder before supervision is counted.
* ⚠️ **Mean 0.76 manoeuvres/record**: ~a quarter of records carry no manoeuvre event,
  so any per-layer supervision count must be reported against the records that
  actually have one, never against 4,572.
