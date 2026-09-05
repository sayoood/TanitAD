# RESULT — is refav1's left/right turn asymmetry real?

**STATUS: PART 1 (zero GPU) COMPLETE AND BANKED. PART 2 (the wide panel, two
inference seeds) is RUNNING.** ⛔ **Until §6 is filled, the answer to the PI's
question is OUTCOME C — INCONCLUSIVE — and neither "real" nor "not real" may be
quoted.** `SPEC_TURN_ASYMMETRY.md` §5 makes that a legitimate result of this
package, not a failure of it.

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

⇒ **HYPOTHESIS, named before the run:** the proposal mode set is directionally
biased, so at `W_KAPPA = 0` the goal term can still drive the CEM either way,
while under a curvature charge only a direction already present in the seed pool
survives. It predicts the observed pattern with **no asymmetric term anywhere in
the cost**. ⛔ Its instrument does not exist yet (the seed pool's signed
curvature is not dumped) and adding it would edit `refa_v1.py` under a sibling
stream's live arms — registered as the next work item, a one-field dump change.

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

## §6 — THE WIDE PANEL, TWO SEEDS *(pending — this is the section that decides)*

*To be filled from `rec_ta_wk15_s0.json` / `rec_ta_wk15_s1.json` (round 1) and
`rec_ta_ccos_s0/s1.json` (round 2), read by `raw/turn_asym_read.py`, which prints
the verdict against SPEC §1's four conditions mechanically.*

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
