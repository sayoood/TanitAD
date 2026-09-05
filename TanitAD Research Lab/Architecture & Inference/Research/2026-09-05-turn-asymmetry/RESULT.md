# RESULT — is refav1's left/right turn asymmetry real?

**STATUS: ROUND 1 LANDED. ⭐⭐ THE VERDICT IS OUTCOME A — THE ASYMMETRY IS REAL**
(§6, all four pre-registered conditions met, both inference seeds).
⛔ **But the CAUSAL half is still open:** the increment over the goal-head
baseline does not clear this panel's own seed floor, so *"the curvature penalty
causes it"* is NOT yet established. Round 2 (`ta_ccos_s0/s1`, `W_KAPPA = 0`, the
same 75 windows) is RUNNING and is the arm that adjudicates it — see §6.5.

*Read `SPEC_TURN_ASYMMETRY.md` first: it carries the power derivation, the panel
rule, the three amendments I wrote against myself before the run, and BOTH
outcomes.*

---

## §1 — THE FIRST ANSWER IS ARITHMETIC AND IT NEEDED NO GPU

⭐⭐ **AN 11-WINDOW RECALL CANNOT REPRESENT THE 0.0750 SEED FLOOR. IT MOVES IN
STEPS OF 0.0909.**

A recall on `n` trials lives on a grid of spacing `1/n`. The banked ladder panel
carries `n_L = 11` and `n_R = 8`:

| stratum | n | granularity `1/n` | vs the 0.0750 seed floor |
|---|---|---|---|
| `turn_left` | 11 | **0.0909** | ⛔ **coarser than the floor** |
| `turn_right` | 8 | **0.1250** | ⛔ **coarser than the floor** |

⇒ On that panel the instrument's smallest representable change is **larger than
the noise it is being compared against**, so a per-direction recall claim is
inadmissible **in either direction** — including the claim that the asymmetry is
absent. This is what makes `M30` §3's *"far too thin to call"* precise rather
than rhetorical, and it sets the target: `2/n <= 0.0750` ⇒ **n >= 27**.

⭐ Same family as `H-ESTIM-SEED-1` with the object swapped: there the
estimator's **question** was narrower than the claim hung on it; here the
statistic's **resolution** is coarser than the floor it is compared against.

---

## §2 — ⛔⛔ AND THE BANKED PANEL IS **STRUCTURALLY** UNATTRIBUTABLE, NOT MERELY THIN

MEASURED, `raw/retain_drivers.py`. On the banked 40-window panel the DECODED
goal tokens live in **completely disjoint episodes**:

| decoded goal | n | episodes |
|---|---|---|
| `TURN_L` | 9 | **{1, 6}** |
| `TURN_R` | 13 | **{0, 2, 4, 7}** |
| carrying **both** | — | ⛔ **NONE** |

⇒ *"the goal is `TURN_L`"* and *"the window is in episode 1 or 6"* are **the same
variable** there. Any split between them is exactly as consistent with
*"episodes 1 and 6 are hard"* as with a left/right effect, and the
episode-cluster bootstrap — which resamples episodes — is the estimator least
able to separate them. **The design is degenerate, not merely small.**

⚠️ **This applies to my own most striking number and I am recording it against
myself:** §5's `0/9 vs 9/13` retention split sits entirely inside that
degeneracy, at an effective cluster n of **2 and 4**, both below the SPEC's
`>= 5` target.

---

## §3 — THE PANEL THAT FIXES IT, WITHOUT TOUCHING PARITY

⛔ **No episode is re-selected.** The enrichment is a stratified sample of
WINDOWS over the loader's own 535-window grid on the SAME 8 eval-slice episodes.

**GT-only census** (`raw/census.py`; reads ego poses only, so it could not leak
any outcome):

| stratum | n of 535 | episodes |
|---|---|---|
| `lane_keep` | 294 | 8 |
| `turn_left` | **144** | 6 — {0, 1, 2, 5, 6, 7} |
| `turn_right` | **97** | 6 — {0, 2, 3, 4, 6, 7} |

⭐ **The census reproduces the banked stride-16 panel EXACTLY — LK 21 / TL 11 /
TR 8 of 40** — the same-breath control that this GT pipeline is the arm tool's
own and not a re-derivation.

**The built panel** (`raw/build_panel.py`, `raw/panel_turn75.json`, sha256
`c765fd4f9b0722b1…`): **30 `turn_left` + 30 `turn_right` + 15 `lane_keep` = 75
windows over 8 episodes.**

| target (SPEC §1) | required | achieved |
|---|---|---|
| n per direction | >= 27 | **30 / 30** — MET |
| episode clusters per direction | >= 5 | **6 / 6** — MET |
| recall granularity `1/n` | <= 0.0375 | **0.0333** — MET |
| episodes carrying BOTH directions | as many as possible | **4** — {0, 2, 6, 7}, **20 left + 18 right windows inside them** (the banked panel manages 3 episodes with 3 + 4) |
| in-band v7.2 label windows | — | **21** (banked panel: 8; the control that this join count is the arm tool's) |
| nav coverage | — | **75 / 75** |

⇒ The **within-episode contrast** (SPEC §3.13) — the statistic that removes the
episode confound by construction — is available on **5× the data** the banked
panel offers, and it is registered as the **primary attribution statistic**.

⛔ **The panel's ADE and four-family numbers are NOT comparable to the banked
40-window panel.** It is deliberately turn-enriched on a different window grid;
only within-panel arm-to-arm deltas are quotable.

### §3.1 ⭐ THE WIDE PANEL'S FLOOR BASELINE, MEASURED BEFORE ANY PLANNER RAN

`ha`, `ha0`, `ha0_ext` and `ol` are **pure kinematics on the recorded actions** —
no model, no GPU, no planner — so the panel's corpus-plus-labeller directional
baseline can be fixed up front, and it cannot leak the outcome (nothing here
reads `cl`). Tool `raw/floor_preview.py`, which calls the **arm tool's own**
`hold_action_controls` / `hold_ext_controls` / `paths_from_controls` /
`gt_waypoints` rather than a re-implementation.

| arm (no planner) | recall LEFT | recall RIGHT | pooled R − L | **within-episode R − L** |
|---|---|---|---|---|
| `ha` | 0.7000 [0.5667, 0.8333] | 0.6667 [0.4231, 0.8571] | **−0.0333** | **−0.1917 [−0.4000, +0.0500]** |
| `ha0` (constant velocity) | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| **`ha0_ext`** (the M11 integrator floor) | **0.7333** [0.5667, 0.8667] | **0.7000** [0.4996, 0.8571] | **−0.0333** | **−0.1917 [−0.4000, +0.0500]** |
| `ol` (T0) | 0.8333 [0.6333, 1.0000] | 0.9333 [0.7388, 1.0000] | +0.1000 | +0.0000 [−0.3000, +0.3000] |

Controls, all in the same table: GT against itself reads **exactly 1.0000 /
1.0000**; the panel's GT strata read **30 / 30 / 15**, matching the panel file
exactly; and `ha0`'s **exactly 0.0000 / 0.0000** is a *structural* zero — a
constant-velocity plan cannot turn — with `ha0_ext` reading **0.7333 / 0.7000**
non-zero on the very same windows as its same-breath control.

⇒ ⭐ **On this panel the corpus-plus-labeller baseline is ≈ 0 pooled and
SLIGHTLY LEFT-FAVOURING within episodes.** So a POSITIVE `recall_R − recall_L`
from a planner arm would be running **against** its own floor, not with it —
which makes such a gap more notable, not less. ⚠️ And it settles in advance the
objection *"the panel just has easier right turns"*: on the floors, it does not.

---

## §4 — THE MECHANISM AUDIT: NOTHING DETERMINISTIC CAN PRODUCE IT

22 assertions, `raw/mirror_assert.py` / `.txt`, **each with a same-breath control
that must read the other way**.

| component | assertion | value | its control |
|---|---|---|---|
| goal vocabulary | `canonical_controls("TURN_L").kappa == -canonical_controls("TURN_R").kappa`, at every `v0` and every `lon` token | **`max\|kL + kR\| = 0.000e+00`** | kappa not identically zero (**0.08000**); `LANE_KEEP` differs |
| ... its accel channel | EXACTLY equal | `max\|aL - aR\| = 0.000e+00` | — |
| `NUDGE_*`, `LANE_CHANGE_*` | antisymmetric | `0.000e+00` | — |
| curvature charge | `W_KAPPA * mean(kappa^2)` sign-invariant, n = 2000 profiles, at rungs **0 / 0.05 / 15.11245 / 151.1245** | **`0.000e+00`** | the charge is not constant (std 5.7e-05) |
| `_clip` | sign-symmetric | `0.000e+00` | the clip BITES: 2533/5000 steps |
| Kamm cap `mu = 0.7` | sign-symmetric | `0.000e+00` | the cap BITES: `max\|k_kamm - k_plain\| = 0.18360` |
| iCEM noise pool | zero-mean, both seeds | mean **4.3e-10** vs 4·SE 1.9e-02 | not degenerate (std 0.9487) |
| ... and sign-balanced | n(+) **20045** vs n(−) **19955** of 40000 | \|diff\| 90 vs 4·sqrt(n) 800 | — |
| labeller | mirroring `y` swaps `turn_left <-> turn_right` | **3000 / 3000** | the mirror changes 2423 of 3000 labels |
| ... longitudinal label | untouched by the mirror | identical | — |

⚠️ One assertion reads **not exactly zero and I am reporting the number rather
than the verdict**: `dyaw` is negated by the mirror to **2.384e-07 rad**, i.e.
float32 `atan2` rounding — **1.6e-06 of the 0.15 rad turn gate**. Symmetric to
float32 epsilon, not bit-exactly.

**And the floors are symmetric on the same windows** (`raw/turn_asym_read.py`):
`ha0_ext` **0.7273 L / 0.7500 R**, `ha` 0.6364 / 0.7500. ⇒ whatever the effect
is, it is in the **planner**, not the corpus and not the labeller.

⚠️ **`ol` — the T0 arm, which has no planner at all — is itself L/R uneven**
(0.7273 / 1.0000, pooled contrast **+0.2727 [+0.0000, +0.5000]**, not separated).
So a small directional gap exists before any planner acts, and only a gap
materially larger than the floor arms' is attributable to the cost.

### §4.1 The one learned component inside the search, named from source

`refa_v1.py:2484-2485` — **`seed_pool = modes[1:]`**: the imitation `proposal`
head's non-top modes are injected into iCEM's **iteration-0 population**
(`refa_v1_plan.py:291-293`). That pool is **learned**, so nothing forces it to be
sign-balanced. Everything else in iteration 0 is symmetric by construction — the
mean starts at **zero** (`prev_elites` is never passed by `refav1_arm.py`, so
there is **no window-order effect**), `init_var = 1.0` on both channels, and the
noise is sign-balanced.

**HYPOTHESIS, named before the run:** the proposal mode set is directionally
biased, so at `W_KAPPA = 0` the goal term can still drive the CEM either way,
while under a curvature charge only a direction already present in the seed pool
survives. It predicts the observed pattern with **no asymmetric term anywhere in
the cost**.

### §4.2 ⛔⛔ AND IT IS REFUTED — BY THE CHEAPEST POSSIBLE FACT, AT ZERO GPU

I said the instrument did not exist. It did — the features are **cached**, so
`encode` + the proposal head is a few small matmuls that run on **CPU**.
`raw/seedpool_probe.py` runs the same three lines `plan()` runs, in the same
order, over all 75 wide-panel windows:

> ⛔ **`cfg.proposal_k = 1` on this checkpoint, so `modes[1:]` is EMPTY and the
> seed pool is NEVER INJECTED.** `refa_v1.py:2485` reads
> `seed_pool = modes[1:] if modes.shape[0] > 1 else None`; the guard fires,
> iCEM receives `seed_pool=None`, and `refa_v1_plan.py:291` skips the injection
> entirely. **A learned, possibly-biased candidate set cannot explain an
> asymmetry it never contributes.**

Control that this is a real read and not a dead probe: the mode-0 statistics
below come from the **same `modes` tensor** and are non-degenerate, and
`proposal_k` is printed from the loaded config. ⚠️ The vacuous pool sections are
**skipped, not printed as zeros** — a 0.0000 from an empty array is the absence
of a measurement, and printing it as a number is how an absent instrument gets
quoted as a result.

**And the one directional thing that IS in the search leans the WRONG WAY.**
Mode 0 — the named `proposal` baseline, which competes but is not the pool:

| GT stratum | mean signed peak `kappa` of the proposal | n |
|---|---|---|
| `turn_left` | **+0.07040** [−0.17710, +0.31790] | 30 |
| `turn_right` | **+0.11740** [−0.09510, +0.27690] | 30 |
| `lane_keep` | +0.12640 [−0.04790, +0.28400] | 15 |

It leans **LEFT (+) on every stratum, including GT-right** — so it does not track
the GT direction, and it points the **opposite** way to the right-favouring
retention it would have to explain. ⚠️ Its median `|kappa|` is **0.32691**
against `kappa_max = 0.2`, i.e. `_clip` clamps the proposal on most windows;
sign balance n(+) **47** vs n(−) **28** of 75.

⇒ ⭐⭐ **EVERY NAMED MECHANISM IS NOW REFUTED — the vocabulary, the cost, the
clip, the Kamm cap, the noise pool, the labeller, the corpus/speed route, the
goal term's scale, the PROPOSAL-MODE seed pool (which does not exist) and the
proposal's own lean (which is backwards).**

### §4.3 ⛔⛔ CORRECTION TO §4.2, FOUND BY THE LAUNCHED ARM'S OWN BANNER — AND IT MAKES THE CONCLUSION STRONGER

⚠️ **I wrote "the seed pool is EMPTY". That is true of `modes[1:]` and FALSE of
`seed_pool`: I read ONE contributor to a variable and reported the variable.**
The correction came from the very next arm's startup line:

```
[seed-pool] seed_kappa_ladder=None -> the SHIPPED pool
            (proposal modes + THE DECODED GOAL'S CANONICAL CONTROLS)
```

`refa_v1.py:2571-2599` seeds the **decoded goal's own canonical control** into
iteration 0 on every window, unconditionally — and the source says why, from a
measurement:

> *"MEASURED 2026-09-02: without this the planner returned `hold_v0` on 24/24
> windows against a TURN goal at BOTH residual-init scales. `colored_noise` is
> zero-mean over time and its mean is seeded only by the injected candidates, so
> a SUSTAINED curvature is unreachable unless some candidate carries it — and no
> baseline carries curvature."*

⇒ **On every `TURN_L`-goal window a full `kappa = +0.08` candidate IS in the
iteration-0 population, by construction. The search does not have to find it —
it is handed it.**

**MEASURED confirmation, zero GPU, banked dumps** — `|kappa|` realised EXACTLY
0.080000 (= `GOAL_KAPPA_TURN`) over the 22 turn-goal windows:

| arm | exact 0.080000 | `TURN_L` | `TURN_R` |
|---|---|---|---|
| `ccos_argmax` (`W_KAPPA` 0) | **21 / 22** | **8 / 9** | **13 / 13** |
| `ccos_seed1` (the seed replicate) | **21 / 22** | **8 / 9** | **13 / 13** |
| **`wk15`** | 9 / 22 | **0 / 9** | 9 / 13 |
| `wk151` | 1 / 22 | 0 / 9 | 1 / 13 |
| `cos_wk` (shipped) | 0 / 22 | 0 / 9 | 0 / 13 |

Control that 0.08 is not an argmax artefact: across all `ccos_argmax` turn
windows the realised `|max kappa|` takes exactly **two** distinct values —
**0.0800** and **0.2000** (the clip). At zero charge **the plan simply IS the
canonical goal seed**, in both directions and identically under both plan seeds.

⛔⛔ **THIS KILLS "THE SEARCH FAILED TO FIND THE LEFT CANDIDATE" OUTRIGHT.** On
all 9 `TURN_L`-goal windows the full `+0.08` candidate was **in the population
and LOST ON COST** — `wk15` evaluated it and preferred a cheaper one. The
mechanism is the **cost comparison**, on a cost that is provably sign-symmetric
(§4 table) with a symmetric goal decision range (ratio **1.002**, §5.2).

⇒ ⭐ **The last standing explanation is therefore NARROWER and SHARPER than
"the stochastic search":** on those particular windows the *modelled rollout* of
a crushed candidate lands closer, in centred cosine, to the imagined goal than
the curvature differential costs — **a PER-WINDOW property of the world model's
latent geometry, not a per-direction one.** Whether that per-window set splits by
direction beyond chance is exactly what §6 measures.

⚠️ **And it re-reads §5.4.** The ladder did not "help the search find" anything:
it **added cheaper candidates to a comparison that was already being lost**,
which is precisely why it removed the right turns too.

→ **ROOT-CAUSE CLASS of my error: a true statement about ONE CONTRIBUTOR to a
variable, reported as a statement about the variable.** Same family as the
`df` / cgroup / `step_s` traps — a correct reading quoted outside its scope —
with the scope being *which assignment site you looked at*. ⭐ What caught it was
**reading the launched arm's own banner instead of trusting my source walk** —
the same discipline that saved §5.4, where `plan_cfg` said `None` while the
banner said the ladder had run.

---

## §5 — WHAT THE BANKED PANEL ACTUALLY SHOWS (read with §2's caveat throughout)

### 5.1 A hypothesis of mine, proposed and REFUTED in the same turn

**The corpus IS asymmetric.** GT-left turns run at `v0` median **1.886 m/s**
against right's **5.189** over all 535 windows (MWU p = **2.5e-06**;
`raw/corpus_asym.py`). Because the turn gate is on `dyaw ≈ kappa·v·T`, a left
turn needs **3.62×** the curvature and pays **13.1×** the `W_KAPPA` charge to be
*labelled* a turn (`raw/speed_confound.txt`). That is a clean, quantitative
mechanism — **and it is wrong.**

| `wk15`, banked panel | value |
|---|---|
| recall, SLOW half of GT turns | **0.2000** |
| recall, FAST half | **0.2222** |
| recall, LEFT | **0.0000** |
| recall, RIGHT | **0.5000** |
| inside the SLOW band alone | LEFT **0/8**, RIGHT **2/2** |

⇒ Pooled by speed the recall is **flat**; pooled by direction it is not. **The
speed confound does not explain the gap and is logged as refuted.**

### 5.2 The effect is curvature RETENTION, and the charge creates it

`raw/goal_asym.py`, `raw/goal_scale.py`, `raw/mech.py`.

| `W_KAPPA` | tracks the goal at full `\|kappa\| > 0.06`, goal = `TURN_L` | goal = `TURN_R` |
|---|---|---|
| **0** (control, same goals, same seed) | **9 / 9** | **13 / 13** |
| **15.11245** (`wk15`) | **0 / 9** | **9 / 13** |
| 151.1245 | 0 / 9 | 1 / 13 |
| shipped `cos` 0.05 | 0 / 9 | 0 / 13 |

⭐ **The `W_KAPPA = 0` control is a complete symmetry**, so the split is created
by the charge and does not predate it.

**And it is NOT "the left goal is worth less":**

| quantity | `TURN_L` | `TURN_R` | control |
|---|---|---|---|
| median goal decision range | **0.98185** | **0.97965** (ratio **1.002**, MWU p = 0.79) | `LANE_KEEP` **0.00000** |
| goal cost of the full-`kappa` plan | 0.00001 | 0.00001 | — |
| curvature charge at the full goal | **0.09672** | **0.09672** | — |

The decoded goal token is **identical across all seven banked arms (40/40),
including across the two plan seeds**, so the goal head is not the seed's doing;
its own recall is near-symmetric (**left 6/11 = 0.5455**, right 5/8 = 0.6250).
⇒ **The split is produced inside the stochastic iCEM search.** On the 9 retained
`TURN_R` windows the arm returned the **same plan as the `W_KAPPA = 0` arm to
five decimals** — no cheaper crushed candidate was FOUND there — while on all 9
`TURN_L` windows one was found and won.

⚠️ Those 9 rows carry `d_goal = 0.0000` **by construction** and must not be read
as *"the goal objected"*.

⭐ **The sign is never wrong:** **32 / 32** retained plans across the ladder
curve the way their goal asked. The failure is magnitude, not direction — the
crushed left plans sit at `kappa` 0.0115–0.0338 against the goal's 0.0800.

### 5.3 The banked within-episode contrast, at the n it has

| arm | pooled R − L | **within-episode R − L** (3 episodes, 3 L + 4 R) |
|---|---|---|
| `wk15` | +0.5000 [+0.2000, +0.8333] | **+0.8333 [+0.5000, +1.0000]** |
| `ccos_argmax` (`W_KAPPA` 0) | +0.3864 [−0.0278, +0.8571] | +1.0000 [degenerate] |
| `ha0_ext` (FLOOR) | +0.0227 [−0.5000, +0.3750] | **−0.3333 [−1.0000, +0.0000]** |
| `ol` (FLOOR, T0) | +0.2727 [+0.0000, +0.5000] | +0.0000 |

⚠️ **Directionally consistent, and NOT quotable.** Three episodes with 3 left and
4 right windows is below every target in SPEC §1, and the `ccos_argmax` interval
is **degenerate** (all three episodes give the same value, so the bootstrap has
no spread to report). ⭐ The one genuinely informative line is that the **floor**
leans the *other* way within episodes (−0.3333), which is why the wide panel is
worth its GPU.

---

## §5.4 ⭐ THE REGISTERED FREE TEST RESOLVED — and it fired the branch I registered, for a reason I had not

`wk15_ladder` = `wk15` + `--seed-kappa-ladder 0.002,0.005,0.01,0.02,0.04`, **same
`W_KAPPA = 15.11245`, same `--plan-seed 0`, same 40 windows**. The prediction was
written into `SPEC` §3.15 **before the record existed**.

⭐ **The ladder is sign-symmetric, and the arm's OWN banner proves it applied** —
verified from the log rather than from `plan_cfg`, which does not carry it
(`seed_kappa_ladder` is a `plan()` kwarg, not a `PlanConfig` field, and reading
`plan_cfg` alone would have said `None` and voided this whole section):

```
[seed-pool] seed_kappa_ladder=(0.002, 0.005, 0.01, 0.02, 0.04) 1/m
            -> 10 EXTRA iteration-0 candidates (both signs).
            Radii 500 m, 200 m, 100 m, 50 m, 25 m
```

| arm (all `ccos`, seed 0, 40 windows) | recall L | recall R | n_pred L | n_pred R | TAC lat kappa |
|---|---|---|---|---|---|
| `ccos_argmax` (`W_KAPPA` 0) | 0.3636 | 0.7500 | 9 | 9 | 0.3795 |
| **`wk15`** (`W_KAPPA` 15.11) | **0.0000** | **0.5000** | 0 | 4 | 0.2611 |
| **`wk15_ladder`** (+ 10 symmetric candidates) | **0.0000** | **0.0000** | **0** | **0** | **0.0000** |
| `combined` (`W_KAPPA` 0 + `kamm_mu` 0.7) | 0.3636 | 0.7500 | 8 | 9 | 0.4148 |

**REGISTERED PREDICTION:** *if the asymmetry is the SEARCH failing to find the
left candidate, `turn_left` recall should RISE above 0.0; if it stays at 0.0 with
symmetric candidates on the table, the search-failure explanation is WEAKENED.*

⇒ **`turn_left` stayed at 0.0000, so the registered branch fired.** ⛔ **But it
fired for a reason the prediction did not anticipate, and that weakens the
inference — which I am reporting rather than banking the branch:** `turn_right`
collapsed from 0.5000 to **0.0000** as well. **The arm emitted no turns in either
direction** (confusion matrix entirely in the `lane_keep` column, `TAC lat kappa`
exactly 0.0000). ⚠️ **An arm that never turns cannot tell you which direction it
prefers**, so as a *directional* probe this test is largely VOID.

⭐ **What it IS informative about is the COST, and there it is decisive and
symmetric.** The ladder offered cheap sub-threshold curvature: at `kappa = 0.04`
the charge is `15.11245 x 0.04^2 = 0.0242` against the full goal's `0.09672`, and
at 0.02 it is `0.0060`. **The search took the cheap compromise on every window,
in both directions**, and the `|dyaw| > 0.15` gate then reads the result as
`lane_keep`. That is the cost-geometry reading confirmed — *the charge dominates
the goal well below the goal's own curvature* — and it is **sign-neutral**.

⭐⭐ **AND IT ADDS ONE GENUINELY NEW FACT, which is the most important line in
this section:** `wk15`'s right-turn survival is **FRAGILE**. A third draw at the
**same cost, same weights and the same plan seed**, differing only in the
iteration-0 candidate set, moved the split from **0.0 / 0.5** to **0.0 / 0.0**.
⇒ the "9 of 13 retained" is a property of **which candidates the search happened
to have**, not a stable property of the direction — which is exactly the class of
instability the two-seed pair was registered to measure, arriving from a third
direction.

⚠️ **Scope, as registered:** the ladder tops out at 0.04 against
`GOAL_KAPPA_TURN = 0.08`, so it can never satisfy the `|kappa| > 0.06` retention
criterion and this read is on **recall only**; and it is **one arm at one seed on
the degenerate 40-window panel** (§2), so it is direction-of-travel and never the
verdict. §5's outcomes remain decided on the wide panel alone.

---

## §6 — THE WIDE PANEL, TWO SEEDS — ⭐⭐ **OUTCOME A: THE ASYMMETRY IS REAL**

**All four pre-registered conditions MET.** Reader `raw/turn_asym_read.py`, which
prints the verdict mechanically; raw at `raw/turn_asym_read_round1.txt`, records
and logs at `raw/arms/`.

### 6.1 The numbers

**Power (SPEC §1): `turn_left` n = 30 / 6 clusters, `turn_right` n = 30 /
6 clusters, granularity 0.0333 — ALL MET.**

| arm | recall LEFT | recall RIGHT | **R − L** | sep? |
|---|---|---|---|---|
| **`ta_wk15_s0`** (`--plan-seed 0`) | **0.0000** [0.0000, 0.0000] | 0.4333 [0.1934, 0.6552] | **+0.4333 [+0.1667, +0.6539]** | **YES** |
| **`ta_wk15_s1`** (`--plan-seed 1`) | **0.0000** [0.0000, 0.0000] | 0.5000 [0.2219, 0.7419] | **+0.5000 [+0.1874, +0.7407]** | **YES** |
| `GOAL_HEAD` (baseline, an INPUT) | 0.3000 [0.0000, 0.6333] | 0.6667 [0.3125, 0.9375] | +0.3667 [−0.3297, +0.8377] | no |
| `ha0_ext` (FLOOR, no planner) | 0.7333 | 0.7000 | **−0.0333** | no |
| `ha` (FLOOR) | 0.7000 | 0.6667 | −0.0333 | no |
| `ol` (FLOOR, T0) | 0.8333 | 0.9333 | +0.1000 | no |

**The panel's own seed floor** (paired, same windows, `s0` vs `s1`):
`turn_left` **+0.0000 [+0.0000, +0.0000]**, `turn_right` +0.0667 [+0.0000,
+0.1935]. Per amendment 0 the floor used is the CI's **reach**, floored at the
instrument's step: **`floor_used = max(0.1935, 0.0333) = 0.1935`.**
*(Control: the pooled `TAC_traj_lat_correct` seed delta reads +0.0267 [+0.0000,
+0.0741], consistent with the banked 0.0750. Control: an arm against itself reads
exactly +0.0000.)*

| SPEC §1 condition | result |
|---|---|
| 1 — `\|gap\| > floor` at **both** seeds | **MET** — 2.24× and 2.58× the floor |
| 2 — **sign agrees** across seeds | **MET** — both `+` |
| 3 — n and cluster targets | **MET** — 30/30, 6/6 |
| 4 — **within-episode** contrast agrees and clears the floor | **MET** |

### 6.2 ⭐ It survives the confound that killed the banked panel

The within-episode contrast (SPEC §3.13), over the **4 episodes carrying both GT
directions**, 20 left + 18 right windows inside them:

| arm | within-episode R − L |
|---|---|
| **`ta_wk15_s0`** | **+0.4250 [+0.1250, +0.7000] SEPARATED** |
| **`ta_wk15_s1`** | **+0.5083 [+0.2000, +0.8167] SEPARATED** |
| `ha0_ext` (FLOOR) | **−0.1917** [−0.4000, +0.0500] |
| `ol` (FLOOR) | +0.0000 [−0.3000, +0.3000] |

⇒ **The plan's gap runs AGAINST its own floor, not with it** — the floors lean
slightly *left*-favouring within episodes. ⛔ The banked panel could not do this:
its two turn-goal strata were episode-disjoint (§2). *(Control: a stratum against
itself reads exactly +0.0000.)*

### 6.3 ⭐⭐ AND THE LEFT RECALL IS A STRUCTURAL ZERO

`turn_left` recall is **0.0000 at BOTH seeds**, `n_pred = 0`, `never_predicted:
["turn_left"]` — **the planner does not emit a single left turn on any of the 75
windows**, across all **6** left-carrying episodes. The GT-left row of the
confusion matrix is `[30, 0, 0]`: every one of the 30 left windows is planned as
`lane_keep`. ⚠️ That is why the seed floor on left is exactly 0.0000 and why no
seed can move it — it is a structural zero with full cluster support, not a noisy
rate.

### 6.4 The decisive cell: it is the COST, not the goal head

`raw/head_vs_cost.py` / `_s0.txt` / `_s1.txt`. Restricted to windows where the
head decoded the **correct** turn token:

| seed | stratum | n | clusters | plan recall | median realised `kappa` | full `\|k\| > 0.06` |
|---|---|---|---|---|---|---|
| s0 | **GT-LEFT** | 9 | ⚠️ **2** | **0/9 = 0.0000** | **+0.02066** | **0/9** |
| s0 | GT-RIGHT | 20 | ⚠️ 4 | 13/20 = 0.6500 | −0.08000 | 18/20 |
| s1 | **GT-LEFT** | 9 | ⚠️ **2** | **0/9 = 0.0000** | **+0.02496** | **0/9** |
| s1 | GT-RIGHT | 20 | ⚠️ 4 | 15/20 = 0.7500 | −0.08000 | 19/20 |

Per episode, `ta_wk15_s0`: the head decodes left correctly in episodes **1
(4/5)** and **6 (5/5)**, and in **both** the plan produces **0/5**.
⇒ **Where the head is right about the direction, the plan still loses every left
turn and keeps 18–19 of 20 right turns.** Combined with §4.3 — the full `+0.08`
candidate is *handed* to the search and loses on cost — the loss is in the
**cost comparison**.

⚠️ **Both cells sit at 2 and 4 clusters, below §1's `>= 5` target**, and are
reported with that stated rather than discovered afterwards. The part with full
cluster support is §6.3's structural zero.

### 6.5 ⛔ WHAT IS **NOT** ESTABLISHED — the causal half, and it is still open

**The pre-registered claim that was tested is *"the asymmetry is real"*. It is.
The claim that the CURVATURE PENALTY causes it is a DIFFERENT claim and this
panel does not settle it.**

The increment over the goal head (SPEC §3.16 — the only part attributable to the
cost on the pooled statistic):

| seed | plan gap | head gap | **increment** | vs `floor_used` 0.1935 |
|---|---|---|---|---|
| s0 | +0.4333 | +0.3667 | **+0.0667** | ⛔ **below** |
| s1 | +0.5000 | +0.3667 | **+0.1333** | ⛔ **below** |

⇒ **On the pooled statistic the increment does NOT clear the floor.** §6.4's cell
says the cost is doing the work and §6.5's arithmetic says the pooled evidence
cannot separate the cost's contribution from the head's. ⭐ **Those two readings
are in tension, and the arm that adjudicates them is round 2** — `ta_ccos_s0/s1`,
`W_KAPPA = 0` on the same 75 windows, launched 01:33 and 01:43. If `ccos` reads
symmetric there, the penalty causes it; if `ccos` reads the same 0.0 / high split,
the penalty does **not**, and the defect is upstream.

⚠️ **The RETENTION statistic (§1.1) is NOT used.** It reads 0/13 vs 0.8000 (s0)
and 0.8286 (s1), separated — but its strata are the **episode-degenerate decoded
goal tokens** (`TURN_L` 13 windows / **2** episodes, `TURN_R` 35 / 4), exactly as
amendment 4 registered before the run. It is reported and decides nothing.
*(Control: **57/57** retained plans across both seeds curve the way their goal
asked — the sign is never wrong; the failure is magnitude.)*

### 6.6 The four families, both seeds (the binding rule)

⛔ Not comparable to the banked 40-window panel — a different, turn-enriched
window grid (§3).

| arm | LON speed MAE | LON along MAE | LAT heading MAE | LAT curv MAE | LAT cross MAE | TAC lat kappa | TAC lon kappa |
|---|---|---|---|---|---|---|---|
| `cl` s0 | 0.9527 | 0.8217 | **20.5113** | **0.041744** | 0.6718 | **0.1404** | −0.0806 |
| `cl` s1 | 0.9499 | 0.8112 | **19.4058** | **0.040262** | 0.6481 | — | −0.0243 |
| `ha0_ext` | **0.3312** | 0.8994 | 31.9865 | 0.100662 | 0.6765 | — | — |
| `ol` (T0) | 0.0895 | 0.7460 | 30.1803 | 0.107853 | 0.6518 | — | — |

⭐ **`wk15` still wins the LATERAL family outright on this panel** — heading MAE
**19.4–20.5** against the floor's **31.99**, curvature MAE **0.0403–0.0417**
against **0.1007** — and still loses the **LONGITUDINAL** family outright
(**0.95** against **0.33**), which is `M27` §5's named blocker, unchanged.
⚠️ **STRATEGIC: UNAVAILABLE** (no route label on this slice) and
**distance-keeping: REFUSED** (no lead block passed), each declared with its
reason and n rather than silently dropped. **TAC longitudinal recall is
`accelerate` 0.0000 at both seeds** — the longitudinal vocabulary defect, again.

### 6.7 ⛔ THE CONSEQUENCE, per SPEC §5 outcome A

* **`wk15` may be quoted as a lateral fix on ADE, curvature MAE and heading MAE
  ONLY — and the per-direction recall must be stated beside it, every time.**
* **`M30` §3's caveat STANDS and HARDENS into a named defect:**
  ⛔ **refav1's planner emits ZERO left turns. Not "fewer" — zero, on 30 GT-left
  windows across 6 episodes, at both inference seeds, while emitting 13–15 right
  turns on the matched stratum.**
* ⛔ **The next lever is NOT a direction-aware seed pool** — that is refuted
  (§4.3): the goal's own canonical `+0.08` control is already in the iteration-0
  population on every left-goal window, and it **loses on cost**. The lever is
  the **cost comparison itself**, and round 2 says whether `W_KAPPA` is the term
  responsible.

---

## §7 — INSTRUMENT WORK, all default-identical

1. **`--window-list` on `refav1_arm.py`.** A stride cannot enrich a stratum.
   Explicit `(episode NAME, t)` pairs — never indices, which are meaningful only
   against one loader construction. Omitting it is **bit-identical to every
   banked arm**, pinned by 17 tests including a rewrite of the legacy stride
   expression to compare against. It **REFUSES** an off-grid, unknown-episode or
   duplicated window **by name** rather than silently thinning a panel.
   Preflight against the real loader: the stride path returns exactly **40**, the
   list path exactly **75**.
2. ⛔ **`--help` exited 1 with `UnicodeEncodeError` on a default cp1252
   console** — **nine** pre-existing help strings carry a marker, and argparse
   writes the whole help text in one call. Every running arm was fine (the queue
   scripts set `PYTHONIOENCODING=utf-8`), so it surfaced only when an operator
   asked for help or mistyped a flag. ⭐ **Sibling of `M30` §4's `%` defect one
   layer down:** that broke argparse's **formatter**, this one the **stream**,
   and fixing the formatter did not fix the command. Markers now degrade
   (`backslashreplace`) instead of dying; on a utf-8 stream **nothing changes**.
   Verified both ways, plus a CLI typo now exits 2 with no traceback.
3. **`test_A6_adding_ha0_moves_no_existing_arm` was RED on a clean HEAD
   checkout.** It asserted **equality** against exactly four arms, so the
   legitimate addition of `ha0_ext` (the M11 integrator floor) broke it. An
   equality there does not test *"no existing arm moved"* — it tests *"no arm was
   ever added"*, which every correct addition must break. Made a subset
   assertion plus a stamp check. **58 tests green.**

---

## §8 — DELIVERABLE MANIFEST

| artifact | where it lives |
|---|---|
| `SPEC_TURN_ASYMMETRY.md` (power target, panel rule, 3 amendments, both outcomes) | repo: this package |
| `RESULT.md` (this file) | repo: this package |
| `raw/census.py` / `.txt` / `census_rows.json` | repo: this package |
| `raw/build_panel.py` / `.txt`, `raw/panel_turn75.json` | repo: this package |
| `raw/mirror_assert.py` / `.txt` (22 symmetry assertions) | repo: this package |
| `raw/corpus_asym.py` / `.txt`, `raw/speed_confound.py` / `.txt` | repo: this package |
| `raw/goal_asym.py` / `.txt`, `raw/goal_scale.py` / `.txt`, `raw/mech.py` / `.txt` | repo: this package |
| `raw/retain_drivers.py` / `.txt` (the episode-confound finding) | repo: this package |
| `raw/turn_asym_read.py` (the verdict reader) | repo: this package |
| `raw/ta_queue.py` (the GPU-gated launcher) | repo: this package |
| `--window-list`, `select_windows`, `_survive_a_narrow_console` | repo: `taniteval/tools/refav1_arm.py` |
| `test_refav1_window_list.py` (17 tests, new) | repo: `stack/tests/` |
| `test_refav1_kin_contract.py` (the stale equality, repaired) | repo: `stack/tests/` |
| arm records `rec_ta_*.json` + dumps | `C:/Users/Admin/refav1_margin/p4out/` — **OFF-REPO, SINGLE COPY until §6 lands** |
