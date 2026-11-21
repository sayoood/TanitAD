# RESULT — E-DATA-HORIZON-1

`2026-09-02 · Research Lab · Data Engineering · 0 GPU · raw/horizon_budget.json`

**Tier: N/A** — a corpus/windowing measurement, not a model read. No T0/T1 stamp
applies; the four-metric-families rule binds *evals* and does not bind here.
Stated so the omission is not mistaken for one.

**Headline.** The gate's question is answered, and the answer **inverts the
blocker**: the time-shifted O11 control *is* affordable at the settings the queued
arm will use — and **affordability was the wrong question**. At the largest FREE
shift the "counterfactual" steering still correlates **r = 0.744** with the true
action: a negative that is mostly the positive. Separately, two published facts
about the corpus turn out to be wrong — the episodes are **~199 frames, not 120**,
and the strategic band reaches **18 s, not 10**.

---

## F1 — ⛔ THE EPISODES ARE ~199 FRAMES, NOT THE 120 THE TRAINER'S OWN DOCSTRING ASSUMES

`stack/scripts/train_v6_staged.py:2864-2890` builds a published limit table on
**T = 120**, and is candid that the figure is INHERITED —

> *"the 120-frame figure is INHERITED (`V6_TRAINER_DESIGN.md §3.6`, consistent
> with the `-w120-` cache name …)"*

⇒ **a cache NAME was used as a frame count.** Two independent lines refute it.

### Line 1 — the banked arm configs settle it for the TRAIN corpus, exactly

Every arm's `config.json` banks `o4.o4_n`, the realised training-window count
(**MEASURED**, ours; `raw/horizon_budget.json` → `arm_configs`, md5 per file):

| arm | `max_horizon` | `o4_n` (windows) |
|---|---|---|
| `postrain30k` · `k8clip05p30k` · `rdw8p30k` | 20 | **415,002** |
| `k60clip05p30k` | 60 | **319,002** |

```
415,002 − 319,002 = 96,000 = 40 steps × 2,400 episodes   exactly
```

Raising `max_horizon` by 40 removes exactly 40 windows **per episode**
(`_contract.py:120`), so the difference being *exactly* 40 × 2,400 pins the
contributing-episode count at **E = 2,400** — and then **both** counts give the
same mean independently:

```
415,002 / 2,400 = 172.9175 = T̄ − 6 − 20   ⇒  T̄ = 198.9175
319,002 / 2,400 = 132.9175 = T̄ − 6 − 60   ⇒  T̄ = 198.9175   ← same value, two sources
```

⇒ **T̄ = 198.92 for the train corpus.** At T = 120 the same arithmetic predicts
`(120−6−20) × 2,400 = 225,600` windows against the banked **415,002** — **off by
1.84×**. The docstring's premise is refuted by artifacts the programme already
holds.

⭐ A free corollary from the exactness: since the difference is *precisely*
40 × 2,400, **every one of the 2,400 episodes contributed windows at both
horizons**, so every episode satisfies `T ≥ 6 + 60 + 1 = 67`.

### Line 2 — the episode bytes agree, on the build this pass may touch

**MEASURED** (ours; 24 banked `*.v2ep.pt` of `physicalai-val-w120-256x640cyl`,
`raw/horizon_budget.json` → `episodes`):

```
T : min 200 · max 207 · mean 201.17 · distinct {200, 201, 207} · n = 24
geometry : 256x640 cylindrical, f_ref 305.577, codec png, n_stack 3
```

⚠️ The train build lives on Thor and was **not** touched; Line 1 is what carries
the train claim, and it needs no access. The two lines are kept as separate
evidence and agree in family (val mean 201.17 vs train mean 198.92).

*(This package does not claim to know what `w120` does denote — only that it is
not the episode length.)*

## F2 — ⛔ THE STRATEGIC BAND IS REACHABLE TO ~18 s, NOT 10 — AND THE CORPUS IS NOT THE BINDING CONSTRAINT

`reachable_strategic_ticks`'s docstring table says K = 6 is where *"the corpus is
exhausted"* (0 windows/episode), and that only the band's bottom edge (K = 4, 8 s)
is reachable, at a **64 %** window cost. With `max_horizon = 20 K` and
`window = 6`, MEASURED (`raw/horizon_budget.json` → `window_budget_t_max`, val
build; train column derived from T̄ = 198.92):

| K | horizon | `max_horizon` | val win/ep | train win/ep | zero-window eps (val) | docstring said |
|---|---|---|---|---|---|---|
| 1 | 2 s | 20 | 175.17 | 172.92 | 0/24 | 94 |
| 3 | 6 s | 60 | 135.17 | 132.92 | 0/24 | 54 |
| 4 | 8 s | 80 | **115.17** (−34 %, not −64 %) | 112.92 | 0/24 | 34 |
| 5 | 10 s | 100 | 95.17 | 92.92 | 0/24 | 14 |
| **6** | **12 s** | 120 | **75.17** | 72.92 | **0/24** | ⛔ **0 — "the corpus is exhausted"** |
| 7 | 14 s | 140 | 55.17 | 52.92 | 0/24 | — |
| **9** | **18 s** | 180 | 15.17 | 12.92 | 0/24 | — |
| 9.75 | 19.5 s | 195 | 0.25 | — | 23/24 | — |

⭐ **The consequence is strategic, not clerical.** MM-E15's median manoeuvre start
is **12.5 s** and the `strategic_s` band is **[8, 30)**. K = 6 (12 s) and K = 7
(14 s) are both reachable with every val episode still contributing — so the
corpus **straddles the median manoeuvre start**. `V7_LAUNCH_GATE.md` P4 says
*"Strategic (8–30 s) is still unreached"*; that stands as a statement about the
**recipe**, but the accompanying reason — that the band is corpus-limited — does
not hold below ~18 s. **The binding constraints are compute (MM-E15's 10× rollout)
and the predictor's collapse past one tick, not the data.**

⚠️ **Two scope limits, stated rather than discovered later.**
1. `reachable_strategic_ticks` refuses on the **shortest** episode, not the mean.
   The train build's minimum is **unmeasured** (F1 bounds it only at `T ≥ 67`),
   and the mean 198.92 being below the val minimum 200 proves some train episodes
   *are* shorter. ⇒ **any K above ~5 needs the train build's minimum measured
   first** — a pod-side one-liner, proposed as backlog row **L-12**.
2. **The CODE is correct and is not implicated.** The function is deliberately
   parameterised on `episode_frames` and receives the real value at the call site
   (`:5122-5124`), so a live guard computes the true table. It is the DOCSTRING's
   worked example — and anything quoting it — that is wrong.
   *Root-cause class: a true measurement quoted outside its scope (`df` / Thor
   `free` / `step_s` family), the scope here being a different cache build.*

## F3 — ⚠️ THE k=60 ARM'S WINDOW SET SHIFTED BY 23 %, AND THE INSTRUMENT THAT CHECKS FOR CONFOUNDS CANNOT SEE IT

⭐ **Credit where it is due:** the MM-E19 attribution package **did** record this
axis — its §2 table lists `max_horizon` 20 → 60 and `o4_n` 415,002 → 319,002,
both correctly marked *"derived"*. This finding is not that it was missed.

**It is that the automated check misses it, and the consequence was never stated.**

`max_horizon` is derived (`train_v6_staged.py:5053` `need_k = max(o1_k, o5_k)`;
`:5076`), and **`args.max_horizon` is `None` on all four banked arms** — so an
args-level diff reports them **identical on that key**. MM-E19's own
`one_variable_check_args_diff` (`…/2026-08-31-mm-e19-k60-horizon/raw/
mm_e19_read_step30000.json`) indeed does **not** list it, while the value the
windowing actually uses is **20 vs 60**. The prose caught what the instrument
cannot.

**What was never stated is the size and the shape of the change:**

* **−23.1 %** training windows (415,002 → 319,002 — MEASURED, banked configs);
* window START positions truncated from `t ∈ [0, ~173)` to `t ∈ [0, ~133)` — the
  k=60 arm **never sees the last ~23 % of any episode as a window start**. That is
  a **distributional** shift, not only a smaller sample.

⚠️ It is `o5_k`'s mechanical consequence, so it is **not a free variable to
control** — no arm can hold it fixed while changing `o5_k` without an explicit
`--max-horizon`. What it changes is what the experiment **means**: *"we lengthened
the rollout"* is also *"we shrank the training set by 23 % and re-sliced it
earlier in every episode."*

⭐ **And it bears directly on the finding MM-E19 elevates.** The decomposition's
headline is that the **scene** spread rose **1.71×** under k = 8 → 60 while action
response fell — *"the model got better at the scene while getting worse at the
action."* A 23 %-smaller, temporally-truncated window set is a **candidate
contributor to a change in scene statistics**, and it is not controlled.
⛔ **The k=8/clip-0.5 control does not cover this axis** — its banked config reads
`max_horizon = 20`, `o4_n = 415,002`, i.e. the incumbent windowing. It isolates
`clip` and `init_from` cleanly; the windowing shift stays welded to `o5_k`.
*(HYPOTHESIS, not a measured cause — this package measures the exposure, not its
effect on the spread. The cheap discriminating test is proposed as L-10.)*

## F4 — ⛔ THE GATE'S QUESTION, ANSWERED — AND AFFORDABILITY WAS THE WRONG QUESTION

**The available future-action horizon EQUALS `max_horizon`, exactly.**
`_contract.py:135` slices `ep.actions[t+w : t+w+max_horizon]`, and `:120`'s
`t_max = T − window − max_horizon` guarantees the slice is always **full** — never
short, never padded. So the gate's conditional resolves **by regime**:

| regime | `o5_k` | `max_horizon` | spare beyond the rollout |
|---|---|---|---|
| the k=60 arm | 60 | 60 | ⛔ **0** — the gate's worry is exactly right here |
| ⭐ **the queued O11 re-run** (`postrain30k` + O11) | 8 | **20** | **12**, and against `o11_k = 4` the free shift budget is **s ≤ 16** |

⇒ **At the settings the arm will actually use, the time-shifted control needs no
new bytes, no regrouped batches and no windowing change.** The gate's
*"neither is small"* is too pessimistic for the arm it was written about.

### ⛔ But the free range is exactly the range where the negative is not a negative

A same-clip time-shifted negative at shift `s` is **the ego's own action `s` steps
later**, and the action is strongly autocorrelated. **MEASURED**, pooled over 24
episodes (`raw/horizon_budget.json` → `action_autocorrelation_r_by_lag`; n = 4,444
pairs at lag 16):

| shift s | seconds | steer r | accel r |
|---|---|---|---|
| 4 | 0.4 | 0.978 | 0.902 |
| 8 | 0.8 | 0.920 | 0.780 |
| **16 — the free budget** | 1.6 | ⛔ **0.744** | 0.461 |
| 26 | 2.6 | 0.500 | ~0.16 |
| 41 | 4.1 | 0.250 | ~0.09 |
| 55 | 5.5 | 0.100 | ~0.08 |

Interpolated thresholds (`min_decorrelating_shift_steps`): steer |r| ≤ 0.75 at
**s ≈ 15.7**, ≤ 0.50 at **s ≈ 26.1**, ≤ 0.25 at **s ≈ 40.5**, ≤ 0.10 at
**s ≈ 54.6**. Accel decorrelates ~1.8× faster and is **not** the binding channel;
**steering is**.

⇒ At the largest free shift the "counterfactual" steering retains **74 %**
correlation with the true action. An InfoNCE negative that is mostly the positive
**pulls the loss toward its floor and reads as action-blindness that is not
there** — precisely the failure `train_v6_staged.py:3255-3260` already documents
for `randperm` fixed points, arriving through a different door. Shipping the
control on the free budget alone would have manufactured a **false negative** on
the programme's most contested question.

### ⇒ The design target, as a number

**`max_horizon ≥ o11_k + s*`**, with `s*` chosen from the table and *stated in the
arm's spec*:

| target | s\* | required `max_horizon` | train windows | cost vs 415,002 |
|---|---|---|---|---|
| steer \|r\| ≤ 0.50 | 26 | **30** | 391,002 | **−5.8 %** |
| ⭐ steer \|r\| ≤ 0.25 | 41 | **45** | 355,002 | **−14.5 %** |
| steer \|r\| ≤ 0.10 | 55 | 59 | 321,402 | −22.6 % |

**Recommendation:** run the time-shifted control at **`--max-horizon 45`**
(steer |r| ≤ 0.25). The cost is 14.5 % of windows; it is strictly cheaper than the
grouped same-clip batching the gate lists as the alternative, which additionally
changes the sampling distribution and so raises a parity question this does not.
⚠️ It is still a **windowing change** and must be declared in the arm's spec
rather than inherited silently — which is exactly the defect F3 documents.

## F5 — ⚠️ THE ANTI-RE-SELECTION GUARD IS ARMED IN ONLY ONE STAGE

The episode drop-out census and refusal — the check that stops a longer horizon
from silently re-selecting the corpus, and the reason PI decision D4's parity
reasoning holds — lives at `train_v6_staged.py:5120-5151`, inside
**`if w_stage.w_s1_multi:`**.

Every v7-tiny 30k arm ran stage **S-W** with `w_s1_multi` off (banked configs), so
**the census never ran for any of them**. No harm occurred — F1's exact
`40 × 2,400` arithmetic proves all 2,400 episodes contributed at both horizons.
But the protection is **stage-conditional**, and an arm raising `--max-horizon`
outside `w_s1_multi` — including the F4 recommendation above — gets only the
all-or-nothing `if not len(ds_train)` check at `:5102`, which fires solely when the
corpus yields **zero windows in total**.

⇒ Proposed as backlog row **L-11**: hoist the census out of the `w_s1_multi`
branch so it runs for every stage, printing unconditionally and refusing on
`n_drop > 0`.

---

## What this changes

1. **`V7_LAUNCH_GATE.md` §"The O11 re-run design (#4)"** — the REQUIRED control is
   viable and costed (**`--max-horizon 45`, −14.5 % windows**); the binding
   constraint is action autocorrelation, not storage.
2. **`V7_LAUNCH_GATE.md` P4** — the strategic band is reachable to ~18 s; the
   corpus-limit reason for "unreached" does not hold below that.
3. **MM-E19's scope sentence** — the k=60 arm's window set is 23 % smaller and
   temporally truncated, the running control does not cover it, and the automated
   args-diff structurally cannot see it.
4. **`train_v6_staged.py:2864-2890`** — the docstring's worked table is wrong by
   ~1.65× and should be re-derived at T̄ ≈ 199 or reduced to a formula.

## Limits, stated

* The train corpus's **minimum** episode length is unmeasured (bounded only at
  `T ≥ 67`). Reachability beyond K ≈ 5 needs it, because the guard refuses on the
  shortest episode, not the mean. Proposed as **L-12**.
* The autocorrelation is a **corpus** property, reported pooled without a
  per-episode interval. It is used to pick a design threshold, not to support a
  comparative claim, so no estimator is quoted — per the rule that an interval
  must carry its estimator, none is offered rather than a wrong one.
* F3's relevance to MM-E19's scene-spread rise is a **HYPOTHESIS**. This package
  measures the exposure; it does not measure its effect.
