# PRE-REGISTRATION — is refav1's left/right turn asymmetry real?

*Written 2026-09-05 BEFORE any arm ran on the wider panel. Both outcomes are
committed below. The power target is derived from the measured seed floor and is
stated before the achievable n was known to bind.*

**Question (PI, direct).** `wk15` = `ccos` + `W_KAPPA = 15.11245` is the interior
optimum of a closed ladder (ADE 1.3272 -> 0.8934, best curvature 0.030982 and
heading 15.2704 of any arm, parity with `ha0_ext`). Its **turn recall is
asymmetric**: **0.0 LEFT (n = 11)** against **0.5 RIGHT (n = 8)**, where
`W_KAPPA = 0` reads 0.3636 / 0.75. ⛔ Until this is settled, `wk15` may NOT be
quoted as a lateral fix (`M30` §3).

---

## §1 — THE POWER TARGET, derived before looking

MEASURED input, and the only one: the inference-seed floor on the tactical
lateral family is **0.0750 absolute** (`TAC_traj_lat_correct`, paired
episode-cluster bootstrap, 40 windows / 8 episodes,
`.../2026-09-05-refav1-cost-geometry/RESULT.md` §4).

⭐ **A recall on `n` trials lives on a grid of spacing `1/n`.** For a floor of
0.0750 to be *representable at all*, the instrument needs `1/n <= 0.0750`, i.e.
**n >= 14**. At exactly 14 the floor is ONE countable window, so any candidate
effect is confounded with a single window flipping. Requiring the floor to span
at least **two** countable steps gives `2/n <= 0.0750`:

> ⭐ **TARGET: n >= 27 windows per direction. ADOPTED: n_L >= 30, n_R >= 30.**

⛔ **This is what makes `M30`'s "far too thin to call" precise rather than
rhetorical.** The banked panel has `n_L = 11` (granularity **0.0909**) and
`n_R = 8` (**0.1250**) — **both COARSER than the 0.0750 floor itself**. A panel
whose smallest representable change is larger than the noise floor cannot
resolve that floor, so no per-direction recall claim can be made on it in either
direction. *(Same family as `H-ESTIM-SEED-1`: a statistic whose question is
narrower than the claim hung on it — here, one whose RESOLUTION is coarser than
the floor it is being compared against.)*

**Cluster requirement.** The decision estimator is the paired episode-cluster
bootstrap over the 8 eval episodes; its resolution is bounded by the number of
CLUSTERS carrying each stratum, never by the window count (`RETRACTION_LOG` #17:
a stratum with few clusters is a POWER LIMIT, not a negative).

> ⭐ **TARGET: >= 5 of the 8 episodes must carry each direction.**

**The effect size that would call it.** ⛔ NOT 0.0750 — that is a *pooled
accuracy* floor and the wrong statistic for a per-direction rate. The panel
measures its OWN per-direction floor by running the same arm at `--plan-seed 0`
and `--plan-seed 1`:

```
floor_dir = max over dir in {L, R} of | recall_dir(seed 0) - recall_dir(seed 1) |
```

⛔ **AMENDMENT 0, written 21:50 BEFORE any wide arm ran, and it makes this test
STRICTER, not weaker.** The form above has a hole I found by running the reader
on the banked seed pair: the two seeds there agree **exactly** on both turn
strata, so the measured `floor_dir` is **0.0000** — and a floor of zero makes
condition 1 satisfiable by *any* non-zero gap, which is precisely the "a
separated CI is necessary and not sufficient" failure this whole SPEC exists to
avoid, re-entering through the floor's own back door. One seed pair agreeing is
one draw, not a demonstration that the true floor is zero. ⇒ **The floor used in
the decision is**

```
floor_dir = max over dir of max( |lo_dir| , |hi_dir| )      # the paired bootstrap's
                                                            # CI, not its point
floor_used = max( floor_dir , 1/n_L , 1/n_R )               # never finer than the
                                                            # instrument's own step
```

i.e. *how large could the seed effect plausibly be*, floored at the smallest
change the recall can even represent (**0.0333** at n = 30). Both halves can only
raise the bar.

**The asymmetry is REAL only if ALL THREE hold:**

1. `|recall_R - recall_L| > floor_dir` at **BOTH** seeds;
2. the **SIGN** of `recall_R - recall_L` **agrees** across the two seeds;
3. the n and cluster targets of this section are met.

Any other combination is **NOT ESTABLISHED** and is reported as such.

### §1.1 The higher-powered secondary statistic, registered here

The recall is capped by the GT turn count. A strictly higher-powered read of the
same question is **goal-conditioned curvature retention**: over ALL windows whose
DECODED goal token is `TURN_L` or `TURN_R`, the fraction whose plan realises
`max|kappa| > 0.06` — i.e. tracks the goal's canonical `GOAL_KAPPA_TURN = 0.08`
rather than being crushed by the charge. It uses every turn-GOAL window, not only
the GT-turn ones, and it is the quantity the ladder actually moves.

⚠️ **It is a DIFFERENT claim from turn recall and is reported separately**, never
substituted for it. Its own seed floor is measured the same way.

---

## §2 — THE PANEL. Selection rule, stated before it ran

⛔ **PARITY: the corpus is NOT re-selected.** The episode set is the SAME 8
eval-slice episodes (`C:/Users/Admin/refav1_margin/p4/fp8`, labels blob md5
`aa12c948f062181c3297265b51526ec5`). Enrichment is by **stratified sampling of
windows within that fixed episode set**, over the loader's own 535-window grid.

**MEASURED census (GT poses ONLY — no arm output is read, so running it before
this SPEC was committed cannot leak the outcome). Tool `raw/census.py`.**

| stratum | n windows | episodes carrying it |
|---|---|---|
| `lane_keep` | 294 | 8 |
| `turn_left` | **144** | **6** — {0, 1, 2, 5, 6, 7} |
| `turn_right` | **97** | **6** — {0, 2, 3, 4, 6, 7} |
| **total** | **535** | 8 |

⭐ **The census reproduces the banked stride-16 panel EXACTLY — LK 21 / TL 11 /
TR 8 of 40** — the same-breath control that this GT pipeline is the arm tool's
own and not a re-derivation.

⚠️ **Turn definition.** The recall's labeller is the programme's canonical gate
`tanitad.refs.refc_tactical.factor_from_kinematics` (v1 branch, `kappa=None`): a
turn is `|dyaw| > YAW_TURN_RAD = 0.15 rad` over the 2 s horizon, with `turn_l` =
`dyaw > +0.15` and `turn_r` = `dyaw < -0.15`. **The stratum is defined by the
SAME gate the recall is scored with**, so the selection and the metric cannot
disagree. *(The `|kappa| > 4e-2` crossover is the ADE-stratum definition of
`RESULT.md` §7.2; it is recorded per window here but is NOT the recall's class
definition, and the two must not be conflated.)*

**SELECTION RULE (deterministic, published before the run):**

1. Strata from GT only, by the gate above.
2. Quota: **30 `turn_left` + 30 `turn_right` + 15 `lane_keep` = 75 windows.**
3. Allocate each stratum's quota across the episodes that carry it by
   **round-robin, largest remaining availability first**, so no single episode
   can dominate a direction.
4. Within an (episode, stratum) cell, take the requested count **evenly spaced in
   `t`** across that cell's available indices — maximum spacing, to minimise the
   frame overlap between neighbouring windows.
5. Ordering by ascending `(episode, t)`; the list is written to JSON and its
   sha256 recorded in the run record.

⚠️ **The windows OVERLAP by construction** (the grid steps 0.2 s; a window spans
0.8 s of observation and a 2 s future). Overlap is exactly why the decision
estimator resamples **EPISODES**, and why §1's cluster target exists.
⛔ **The resulting ADE / four-family numbers are NOT comparable to the banked
40-window panel** — this is a deliberately turn-enriched panel on a different
window grid. Cross-panel comparison is refused; only WITHIN-panel arm-to-arm
deltas are quoted.

---

## §3 — WHAT IS ALREADY SETTLED AT ZERO GPU (mechanism, P4)

Run before the panel, on synthetic pairs and on the BANKED dumps. Tools
`raw/mirror_assert.py`, `raw/census.py`, `raw/corpus_asym.py`,
`raw/speed_confound.py`, `raw/goal_asym.py`, `raw/goal_scale.py`, `raw/mech.py`.
Every assertion carries a same-breath control that must read the other way.

1. **The goal vocabulary is EXACTLY antisymmetric.**
   `canonical_controls("TURN_L").kappa == -canonical_controls("TURN_R").kappa`
   bit-for-bit (`max|kL + kR| = 0.000e+00`) at every `v0` and every `lon` token,
   with the accel channel EXACTLY equal; `NUDGE_*` and `LANE_CHANGE_*` likewise.
   Controls: kappa is not identically zero (0.08000); `LANE_KEEP` differs.
2. **The curvature charge is EXACTLY sign-invariant.**
   `W_KAPPA * mean(kappa^2)` over 2000 random profiles reads
   `max|c(+k) - c(-k)| = 0.000e+00` at every ladder rung (0, 0.05, 15.11245,
   151.1245). Control: the charge is not constant (std 5.7e-05).
3. **`_clip` and the Kamm cap are sign-symmetric**, plain and at `mu = 0.7`, with
   controls proving both actually bite (2533/5000 steps clipped;
   `max|k_kamm - k_plain| = 0.18360`).
4. **The iCEM noise pool is zero-mean and sign-balanced** at both seeds (mean
   4.3e-10 against 4*SE 1.9e-02; n(+) 20045 vs n(-) 19955 of 40000).
5. **The labeller is exactly mirror-symmetric** — mirroring `y` swaps
   `turn_left <-> turn_right` on 3000/3000 synthetic windows, leaves the
   longitudinal label untouched, and negates `dyaw` to float32 epsilon
   (2.4e-07 rad = 1.6e-06 of the 0.15 rad gate).
6. **The corpus IS asymmetric — but not in the way that would explain it.**
   GT-left turns happen at **v0 median 1.886 m/s**, GT-right at **5.189 m/s**
   over all 535 windows (MWU p = 2.5e-06), a 2.75x speed gap; and because the
   turn gate is on `dyaw ~= kappa * v * T`, a left turn needs **3.62x** the
   curvature and pays **13.1x** the `W_KAPPA` charge to be LABELLED a turn.
   ⛔ **That hypothesis is REFUTED by its own test:** pooled by speed band on the
   banked panel `wk15` reads SLOW **0.2000** / FAST **0.2222** — flat — while by
   direction it reads LEFT **0.0000** / RIGHT **0.5000**; and in the SLOW band
   itself right is 2/2 while left is 0/8. **Recorded as refuted, by me, in the
   same breath as it was proposed.**
7. **The floors are symmetric on the same windows** — `ha0_ext` reads
   **0.7273 L / 0.7500 R**, `ha` 0.6364 / 0.7500, `ol` 0.7273 / 1.0. Whatever the
   effect is, it is in the PLANNER, not in the corpus or the labeller.
8. **The decoded goal token is IDENTICAL across all seven banked arms (40/40),
   including across the two plan seeds**, so the goal head is not the seed's
   doing. Its own recall is near-symmetric: **left 6/11 = 0.5455**, **right
   5/8 = 0.6250**.
9. ⭐⭐ **THE EFFECT LIVES IN CURVATURE RETENTION, AND ON THE BANKED PANEL IT IS
   STARK.** Of windows whose DECODED goal is `TURN_L`, `wk15` tracks the goal at
   full `|kappa| > 0.06` on **0 of 9**; of those whose goal is `TURN_R`, on
   **9 of 13**. ⛔ **CONTROL at `W_KAPPA = 0`, same goals, same seed: `TURN_L
   9/9` and `TURN_R 13/13`** — a complete symmetry. The split is created by the
   charge and does not predate it.
10. ⭐ **AND THE GOAL TERM'S OWN SCALE IS SYMMETRIC** — median goal decision
    `TURN_L` **0.98185** vs `TURN_R` **0.97965** (ratio **1.002**, MWU p = 0.79;
    control `LANE_KEEP` reads **0.00000**). So the split is NOT "the left goal is
    worth less"; the full-`kappa` plan's goal cost is 0.00001 median in BOTH
    directions, and the charge it pays is **0.09672 in both**.
11. ⇒ **The split is produced INSIDE the stochastic iCEM search.** On the 9
    `TURN_R` windows that kept full curvature, the arm returned the SAME plan as
    the `W_KAPPA = 0` arm to five decimals — i.e. **no cheaper crushed candidate
    was found there** — while on all 9 `TURN_L` windows one was found and won.
    ⚠️ Those 9 rows carry `d_goal = 0.00000` **by construction** and are therefore
    UNINFORMATIVE about what a crushed right plan would have cost. They must not
    be read as "the goal objected".

⇒ ⛔ **Every deterministic component is provably sign-symmetric, and the observed
split is produced inside a stochastic search. That is precisely the condition
under which `H-ESTIM-SEED-1` binds: a separated-looking result whose only
untested cause is the INFERENCE SEED. §4's seed pair is therefore not a
formality — it is the discriminating experiment.**

### §3.12 AMENDMENT, written 2026-09-05 21:30 — BEFORE any wide arm ran

⛔ **§3.9's 0/9 vs 9/13 fails this SPEC's OWN cluster criterion, and I am
recording that against myself rather than quoting the split as settled.** The
reader (`raw/turn_asym_read.py`) prints each stratum's cluster count, and on the
banked panel the decoded-goal strata are **`TURN_L` n = 9 in only 2 EPISODES**
and **`TURN_R` n = 13 in 4 episodes** — both below §1's `>= 5 clusters` target.
⇒ The retention split is a **striking observation with an effective n of 2 and 4
clusters**, not a decision-grade result, and §5's outcomes are decided on the
WIDE panel only. *(This is `RETRACTION_LOG` #17 again, and it is the reason the
reader was built to print clusters beside every rate instead of only n.)*

⚠️ Two more numbers the reader surfaced on the banked panel, both of which
belong here because they bound what any wide-panel result can mean:

* **`ol` — the T0 arm, which has no planner at all — is itself L/R uneven**
  (0.7273 L vs 1.0000 R, contrast **+0.2727 [+0.0000, +0.5000]**, not
  separated). So a small directional gap is present in the *corpus-plus-labeller*
  before any planner acts, and only a gap materially larger than the floor arms'
  is attributable to the cost.
* **The reader reproduces the banked pooled seed delta EXACTLY** —
  `TAC_traj_lat_correct` **−0.0750 [−0.1750, +0.0000]**, the same 0.0750 quoted
  in `RESULT.md` §4 — which is the same-breath control that this tool is reading
  the programme's estimator and not a re-implementation of it.

### §3.13 ⛔⛔ AMENDMENT 2, 21:45 — THE BANKED SPLIT IS **STRUCTURALLY** UNATTRIBUTABLE, NOT MERELY THIN

MEASURED, `raw/retain_drivers.py`: on the banked 40-window panel the decoded
goal tokens live in **COMPLETELY DISJOINT EPISODES** —

| decoded goal | n | episodes |
|---|---|---|
| `TURN_L` | 9 | **{1, 6}** |
| `TURN_R` | 13 | **{0, 2, 4, 7}** |
| **carrying BOTH** | — | ⛔ **NONE** |

⇒ **On that panel, *"the goal is `TURN_L`"* and *"the window is in episode 1 or
6"* are THE SAME VARIABLE.** The 0/9 vs 9/13 split is therefore *exactly* as
consistent with **"episodes 1 and 6 are hard"** as with any left/right claim, and
no estimator can separate them — the episode-cluster bootstrap least of all,
since resampling episodes is precisely what it does. ⛔ **The design is
DEGENERATE, not merely small**, and that is a stronger and more honest statement
of the defect than "n is thin".

⚠️ A single continuous variable — the goal cost of the full-`kappa` plan — splits
retained from crushed at 0.8636 against direction's 0.8182, but the values are
0.00001 on 20 of 22 windows and 0.00013 on 2, so that "threshold" is two points
of overfit on 22 and **is not offered as a mechanism.** The episode confound
above is structural; this one is arithmetic noise, and they must not be quoted
with the same weight.

⭐ **THE WIDE PANEL WAS ALREADY BUILT TO BREAK THIS, and the numbers say by how
much.** Because §2's rule round-robins each stratum across every episode that
carries it, **4 of the 8 episodes carry BOTH GT directions — {0, 2, 6, 7}, with
20 `turn_left` and 18 `turn_right` windows inside them** (the banked stride-16
panel manages 3 episodes with 3 + 4).

> ⭐ **REGISTERED NOW, BEFORE THE RUN — the WITHIN-EPISODE contrast is the
> primary attribution statistic**, computed only over the episodes carrying both
> directions: `mean over those episodes of (recall_R − recall_L) INSIDE the
> episode`, with the episode-cluster bootstrap over that episode set. It removes
> the episode confound by construction, and a directional effect that survives it
> is attributable to direction in a way that a pooled contrast never can be.
> ⚠️ It is reported BESIDE the pooled contrast of §1, never instead of it, and
> §5's outcome conditions are evaluated on **both**: if the pooled contrast and
> the within-episode contrast disagree, the answer is **outcome B (NOT
> ESTABLISHED)**, because a disagreement is the confound speaking.

---

### §3.14 AMENDMENT 3 — the ONE learned component inside the search, named from source

The §3.1-4 audit covers everything **deterministic**. It leaves exactly one place
a directional bias can genuinely live, and it is not the cost:

* `refa_v1.py:2484-2485` — **`seed_pool = modes[1:]`**. The imitation
  `proposal` head emits `cfg.proposal_k` modes; the score head ranks them,
  mode 0 becomes the named `proposal` baseline and **every other mode is
  injected into iCEM's ITERATION-0 population** (`refa_v1_plan.py:291-293`).
  That pool is **learned**, so nothing forces it to be sign-balanced.
* Everything else in iteration 0 is symmetric by construction: the mean starts
  at **zero** (`prev_elites` is never passed by `refav1_arm.py`, so windows do
  not inherit each other's mean — there is no ORDER effect), `init_var = 1.0`
  on both channels, and `colored_noise` is zero-mean and sign-balanced (§3.4).

⇒ **HYPOTHESIS, named before the run and NOT tested by this package:** *the
proposal head's mode set is directionally biased, so at `W_KAPPA = 0` the goal
term can still drive the CEM to either direction, while under a curvature charge
only a direction already present in the seed pool survives the search.* It
predicts exactly the observed pattern (symmetric at `W_KAPPA = 0`, split at
15.11) without any asymmetric term in the cost.

⚠️ **The instrument to settle it does not exist yet** — the seed pool's own
signed curvature is not written to the decisions sidecar. ⛔ **Adding it now
would edit `refa_v1.py` under a sibling stream's live arms, so it is registered
as the next work item rather than hacked in mid-flight.** If §5 returns outcome
A, this is the first thing to measure and it is a one-field dump change.

### §3.15 ⭐ A FREE TEST OF THE LAST STANDING EXPLANATION — registered before it lands

The sibling cost-geometry stream has an arm running that I did not ask for and
that happens to be the cleanest available probe of "it is the stochastic search":
**`wk15_ladder` = `wk15` + `--seed-kappa-ladder 0.002,0.005,0.01,0.02,0.04`, same
`W_KAPPA = 15.11245`, same `--plan-seed 0`, same 40 windows.**

⭐ **The ladder is SIGN-SYMMETRIC BY CONSTRUCTION** — `refa_v1.py:2643-2646`
loops `for sgn in (1.0, -1.0)` and the validator **refuses a negative entry**
precisely because *"a negative magnitude is the sign, which this ladder adds for
you"*. So that arm hands iteration 0 an **exactly balanced** set of left AND
right curvature candidates at five magnitudes.

> ⭐ **REGISTERED PREDICTION, before the record exists.** If the asymmetry is the
> SEARCH failing to find the left candidate, handing it perfectly symmetric left
> candidates should **raise `turn_left` recall** relative to `wk15`'s 0.0. If
> `turn_left` recall stays at **0.0** while symmetric candidates were on the
> table, the search-failure explanation is **weakened**, and the cause sits in
> the COST LANDSCAPE — i.e. in the world model's goal-latent geometry, the one
> thing this package has not been able to audit from source.

⚠️ **Scope it honestly, twice.** (a) The ladder tops out at **0.04**, below the
goal's `GOAL_KAPPA_TURN = 0.08`, so a ladder-seeded plan cannot satisfy the
`|kappa| > 0.06` RETENTION criterion — this read is on **RECALL** only, where
0.04 at 5 m/s over 2 s gives `dyaw ≈ 0.4 rad`, comfortably over the 0.15 gate.
(b) It is **one arm at one seed on the degenerate 40-window panel** (§3.13), so
it is a **direction-of-travel** reading and can never be the verdict. §5's
outcomes remain decided on the wide panel alone.

### §3.16 ⛔⛔ AMENDMENT 4, written 00:55 WHILE THE ARMS RAN AND BEFORE ANY `cl` OUTPUT EXISTED — the eval slice's CLUSTER CEILING, and a baseline the recall must be read against

The decoded goal token is an **INPUT** to the planner — `lat_head(intent)` from
vision + nav + `v0`, identical across all seven banked arms (40/40) **including
across plan seeds**. Reading it is panel characterisation of exactly the kind
§1's cluster criterion demands; nothing here touches `cl`, a cost or a plan.
Tool `raw/goaldecode_probe.py`, CPU only.

**1. ⛔ THE RETENTION STATISTIC (§1.1) IS UNATTRIBUTABLE ON THE WIDE PANEL TOO.**

| decoded goal | n | episodes |
|---|---|---|
| `TURN_L` | 13 | **{1, 6}** |
| `TURN_R` | 35 | **{0, 2, 4, 7}** |
| carrying **both** | — | ⛔ **NONE** |

The degeneracy of §3.13 **reproduces**, and it is not an artefact of my sampling:
**the head only ever decodes `TURN_L` in the left-heavy episodes 1 and 6.**
⇒ §1.1's retention read is **reported as UNATTRIBUTABLE on this panel and not
used to decide anything.** ⭐ The RECALL statistic is unaffected — its strata are
GT-defined and **4 episodes carry both directions**.

**2. ⛔ THE WITHIN-EPISODE CONTRAST HAS 4 CLUSTERS, BELOW §1's `>= 5` TARGET —
and I never set a cluster target for it, which was an omission.** Registering it
now: **if the within-episode contrast fails to separate, that is a POWER LIMIT,
not a negative** (`RETRACTION_LOG` #17).

**3. ⭐⭐ THE GOAL HEAD IS ITSELF ASYMMETRIC, AND IT IS UPSTREAM OF THE PENALTY.**

| goal head alone, wide panel | value |
|---|---|
| goal recall **LEFT** | **0.3000** [0.0000, 0.6333], n = 30 / 6 eps |
| goal recall **RIGHT** | **0.6667** [0.3125, 0.9375], n = 30 / 6 eps |
| pooled contrast R − L | **+0.3667 [−0.3433, +0.8525]** — **NOT separated** |
| wrong-DIRECTION rate | GT-left **9/30 = 0.3000** vs GT-right **2/30 = 0.0667** — **4.50x** |

*(Control: a stratum against itself reads exactly +0.0000. Control: the GT strata
on the selected windows read 30 / 30 / 15, matching the panel file.)*

⛔ **This matters for attribution: at `W_KAPPA = 0` the plan IS the goal's
canonical seed on 21/22 turn-goal windows (§4.3), so the plan's turn recall is
BOUNDED ABOVE by the head's.** A plan-recall gap is therefore **partly
INHERITED**, and "the penalty costs LEFT turns first" cannot be read off the
plan's gap alone.

> ⭐ **REGISTERED, before any `cl` output exists:** the plan's per-direction
> recall is reported **beside the goal head's** on the same windows, and only the
> INCREMENT over the head is attributed to the cost. If the plan's contrast
> separates while the head's does not, that increment is the cost's; if neither
> separates, the panel is **cluster-limited** and the answer is **outcome C**.

**4. ⛔⛔ THE CEILING, AND IT IS THE CORPUS — NOT THE PANEL SIZE.** The
within-episode breakdown shows *why* nothing separates:

| episode | goal recall R | goal recall L | R − L |
|---|---|---|---|
| 0 | 0.8333 (n=6) | 0.0000 (n=5) | **+0.8333** |
| 2 | 1.0000 (n=5) | 0.0000 (n=5) | **+1.0000** |
| **6** | 0.0000 (n=2) | 1.0000 (n=5) | ⛔ **−1.0000** |
| 7 | 1.0000 (n=5) | 0.0000 (n=5) | **+1.0000** |
| **mean** | — | — | **+0.4583 [−0.5000, +1.0000]**, not separated |

**Three episodes say right, one says left, and the bootstrap over four clusters
cannot choose.** ⇒ **On this eval slice, left/right turn behaviour varies MORE
BETWEEN EPISODES than between directions**, and *every* per-direction contrast —
the head's, the plan's, anything's — is bounded by 8 episodes of which 4 carry
both directions and one flips sign.

> ⛔ **CONSEQUENCE, stated before the result so it cannot be read as an excuse:
> settling the left/right asymmetry needs MORE EPISODES, not more windows.** I
> raised n per direction from 11/8 to 30/30 and it did not raise the number of
> CLUSTERS, which is what the decision estimator actually consumes. That is a
> corpus request, not a compute request, and it is the honest unblock.

⚠️ **This does NOT license reading a null as a result.** If §6 returns
**outcome C**, the correct report is *"the eval slice cannot resolve an effect of
this size"* — with this section as the reason — and **not** *"there is no
asymmetry"*.

---

## §4 — THE ARMS, in priority order

Two at a time is the measured dev-box ceiling. A killed run still yields value,
because round 1 alone answers the PI's question about `wk15`.

| round | arms | answers |
|---|---|---|
| **1** | `wk15_wideA` (`--plan-seed 0`), `wk15_wideB` (`--plan-seed 1`) | the panel's own per-direction seed floor **and** `wk15`'s asymmetry against it |
| **2** | `ccos_wideA` (`W_KAPPA = 0`, seed 0), `ccos_wideB` (seed 1) | the contrast that makes it a claim about the PENALTY rather than about `wk15` |

All arms: `--cost-metric ccos`, `--cost-weights 0,<W_KAPPA>,64.2971504241507`,
`--no-navshuf --no-lead-block`, argmax goal rule, vocabulary **v7.0 /
`GOAL_KAPPA_TURN = 0.08` untouched** — so the arms are comparable
window-for-window with each other, and with nothing else.

---

## §5 — BOTH OUTCOMES, COMMITTED

**A — the asymmetry is REAL** (all three §1 conditions hold).
⇒ `wk15` may be quoted as a lateral fix **on ADE, curvature MAE and heading MAE
only**, always with the per-direction recall stated beside it. `M30` §3's caveat
**stands and hardens** into a named defect. The mechanism of §3.9-11 is the work
item, and the next lever is a **direction-aware search** (a seed pool carrying
both signs of the goal's canonical curvature), not another weight.

**B — the asymmetry is NOT ESTABLISHED** (the gap is inside the panel's own seed
floor, or its sign flips between seeds).
⇒ The per-direction recall claim is **retracted as unresolvable at the n
available**, `M30` §3's caveat is **lifted**, and `wk15` may be quoted as a
lateral fix. The retraction is logged with its class.

**C — UNDERPOWERED** (an n or cluster target of §1 is missed).
⇒ **INCONCLUSIVE, reported as such; NEITHER A NOR B may be quoted.** A confident
null from an underpowered panel is worse than no panel, because it will be
quoted. ⛔ This outcome is a legitimate result of this SPEC and is not a failure
of the run.

⚠️ **A fourth possibility is registered so it cannot be presented as a surprise:
the two seeds may disagree so much that the SEED FLOOR ITSELF exceeds the L/R gap
in both directions.** That is outcome B by the rule above, and it would make the
interesting number the floor, not the gap.
