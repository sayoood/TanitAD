# Is the SHARED sitclf re-tune deployable — and what does it cost?

**Date** 2026-08-04 · **Stream** sitclf event-recall deployability · **Substrate** dev box,
**0 pod GPU-h** — no pod was touched. **Pre-registration** `./PRE_REGISTRATION.md`, hash-pinned at
`7a62ca201603c7732b0bae0d56d13840c3979171` **before any number below was read** and re-verified at
the end (§11). **Tables** `./TABLES.md`, generated from the JSONs by `render_tables.py` — nothing
retyped.

⛔ PI ruling 2026-08-03 honoured throughout: **labels may use ego; INFERENCE IS VISION-ONLY.** Every
arm reads frozen v1 camera latents and nothing else. The ego block appears only as the
LONGITUDINAL/LATERAL **stratification** variable, in its **causal** rebuild, with the quarantined
legacy block run beside it so the difference is measured (§8).

---

## 0. ⭐ THE ANSWER

**Yes — one part of it, and it is not the part that was banked.**

| | |
|---|---:|
| ⛔ **NOT deployable** | the **+0.3333 event-recall** headline. It is **NOT a skill measurement** |
| ✅ **Deployable** | `lead_s: 3.0 → 1.0` at the **deployed window W = 8**, for **0 extra parameters** |
| **what it buys** | `lane_change` alarm precision at the deployed 3 s horizon **0.0228 → 0.0348**, i.e. **chance-lift 0.8835 → 1.3453** — from **worse than chance** to **1.35× chance**, paired Δ **+0.0119 [+0.0016, +0.0236] SEPARATED**, and its permuted-feature null earns **−0.0063, not separated** |
| **what it costs** | `intersection` **LATERAL-turning −0.1515 [−0.3590, −0.0208] SEPARATED WORSE**; the declared warning horizon drops **3 s → 1 s**; a **label/substrate rebuild** (the trainer has no `--lead`); every banked lead-3.0 sitclf number becomes incomparable |
| **the bigger cell** | `(W = 32, L = 1.0)` buys **2.1607× chance** (`+0.0330 [+0.0150, +0.0520] SEP`) for **+1,152 params** and a different named cost: `intersection` **LONGITUDINAL-decelerating −0.1840 [−0.3515, −0.0381] SEP WORSE** |

**Pre-registered verdict, applied literally: `SELECTION_ARTEFACT` + `FROZEN_IS_BELOW_CHANCE`.**
Both fired. §1 and §2 are why, and they are the most important part of this report.

---

## 1. ⭐ THE CONTROL THAT FIRED — the deployed head is FAR BELOW CHANCE on event recall

`C-CHANCE` was pre-registered as "200 uniform-random score columns at the same 5 % budget", with the
consequence written in advance: *"if `FROZEN`'s event recall is at or below the chance floor, the
headline is not 'a re-tune helps', it is 'the deployed setting is broken and nearly anything beats
it'."* It fired, and by a margin nobody expected.

| situation | **chance event recall** | 95 % band | `FROZEN` | the "winning" cell |
|---|---:|---|---:|---:|
| `lane_change` | **0.8088** | [0.7368, 0.8772] | **0.1404** | 0.4912 |
| `intersection` | **0.8183** | [0.7792, 0.8528] | **0.2294** | 0.3160 |
| `roundabout` ⛔ | **0.8509** | [0.7297, 0.9459] | **0.2973** | 0.8108 |

**Every arm in the parent's headline table — including the winner — is dramatically BELOW a
uniform-random alarm set.** `FROZEN` warns 8 of 57 `lane_change` onsets; a coin warns ~46.

**The mechanism, and it is arithmetic, not a bug.** At a 5 % *global* rank budget with a 5 s
look-back, a random alarm lands in some onset's window with probability ≈ 1 − 0.95⁵⁰ ≈ 0.92. So
`event_recall` at a fixed global budget is **maximised by DISPERSING alarms**, and a classifier that
correctly concentrates its alarms on the clips where the manoeuvre happens is *penalised for doing
so*. Moving from `L = 3.0` to `L = 1.0` disperses the alarms. **That is what the +0.3333 measured.**

⚠️ **This is the generalisation of the warning in my brief.** A sub-chance operating point was known
to hide behind an AP-lift of 1.2926. It also hides behind an event-recall *delta*, which is worse,
because a delta looks like an improvement rather than a level.

⭐ **Cross-checked two ways.** The 200-draw simulation and an **exact hypergeometric expectation**
computed by the promoted `stack` function agree to **0.0025 / 0.0013 / 0.0011** on the three
situations (`verify_chance_floor.json`). Neither number is this stream checking its own JSON.

---

## 2. ⭐ THE DECISIVE CONTROL — the SAME gain appears on PERMUTED FEATURES

The pre-registered `C-NULL-*` control re-runs every cell on the **clip-permuted feature** substrate,
where the camera carries no information about the drive at all.

| `lane_change` | Δ event recall — REAL | Δ event recall — **NULL twin** |
|---|---|---|
| `CELL_w8_L1.0` | +0.0877 [+0.0196, +0.1637] **SEP** | **+0.0877 [+0.0196, +0.1698] SEP** |
| `CELL_w32_L1.0` | +0.3509 [+0.2105, +0.4912] **SEP** | **+0.1579 [+0.0204, +0.2909] SEP** |
| `intersection` · `CELL_w8_L1.0` | **−0.0087** [−0.0494, +0.0321] | **+0.1991 [+0.1375, +0.2594] SEP** |

**`CELL_w8_L1.0`'s event-recall gain is +0.0877 with features and +0.0877 without them — the same
number to four decimals.** On `intersection` the null *beats* the real arm. The event-recall delta
between two `(W, L)` cells is therefore a measurement of a **structural property of the cell** — how
dispersed its top-5 % alarms are — that survives destroying the features entirely.

⇒ ⛔ **The +0.3333 must not be quoted as evidence that the tactical layer gets better input.**
The pre-registered `SELECTION_ARTEFACT` label fires exactly as written.

⚠️ **The precision axis behaves in the opposite way, and that is the sign that it IS a skill
measure:** the same NULL twins earn **−0.0070** and **−0.0179** on alarm precision @5 s, neither
separated. **The gain that survives feature permutation is on recall; the gain that does not survive
it is on precision.** Only the second one is skill.

---

## 3. ⭐ WHAT IS REAL — the operating point, with precision, recall and BOTH denominators

`lane_change` · **2,849 alarms fixed for every arm by construction** (`C-BUDGET` verified: identical
`n_alarm` across all 52 columns) · 1,472 positives · 57 onsets in 55 clusters.

| cell | **precision @3 s** | **chance-LIFT** | recall | **n_alarm** (prec. denom.) | **n_pos** (recall denom.) | tp | params |
|---|---:|---:|---:|---:|---:|---:|---:|
| `W8, L3.0` ⬅ **DEPLOYED** | **0.0228** | **0.8835** ⛔ **worse than chance** | 0.0442 | 2,849 | 1,472 | 65 | 387 |
| **`W8, L1.0`** ⭐ | **0.0348** | **1.3453** | 0.0673 | 2,849 | 1,472 | **99** | **387** |
| `W32, L1.0` | **0.0558** | **2.1607** | 0.1080 | 2,849 | 1,472 | **159** | 1,539 |

Paired, on the same 2,000 episode-cluster draws:

| cell | Δ precision @3 s | Δ precision @3 s — **NULL twin** | Δ precision @5 s |
|---|---|---|---|
| `W8, L1.0` | **+0.0119 [+0.0016, +0.0236] SEP** | −0.0063, not separated | +0.0123 [+0.0016, +0.0243] **SEP** |
| `W32, L1.0` | **+0.0330 [+0.0150, +0.0520] SEP** | −0.0144, not separated | +0.0295 [+0.0003, +0.0561] **SEP** |

⇒ **At the same alarm budget, the deployed setting fires 65 useful alarms out of 2,849 and the
one-knob change fires 99; `W32, L1.0` fires 159.** The deployed head is *below chance* on
`lane_change`; both candidates are above it; and neither gain is reproduced by the permuted-feature
null. **That is the deployable result.**

⚠️ **It does NOT show up as a ranking-metric win.** The frame-level TACTICAL AP-lift on the deployed
label moves 1.2926 → 1.3371 (`+0.0444 [−0.0653, +0.2316]`, **not separated**) for `W8, L1.0` and
→ 1.4746 (`+0.1820 [−0.0471, +0.4839]`, **not separated**) for `W32, L1.0`. **The gain is
concentrated at the top of the ranking, which is where the system actually operates, and AP
integrates it away.** Both facts are reported; neither is dropped.

⚠️ **Honest labelling of what changed after the pre-registration.** My decision rule's D2 was
written on **event recall**. My own `C-CHANCE`/`C-NULL` controls showed that metric to be
dispersion-gamed, so the actionable analysis moved to the **precision** axis. That axis was already
in the pre-registration as **D4** and the re-ordering was anticipated by the `C-CHANCE` clause — but
it is a **re-ordering of pre-registered criteria after seeing a control fire**, not a blind
confirmation, and it is labelled as such rather than presented as the registered primary.

---

## 4. THE FIVE DEPLOYABILITY CRITERIA, scored as written

| | | `W8, L1.0` | `W32, L1.0` |
|---|---|---|---|
| **D1** | a single fixed `(W, L)` | ✅ | ✅ |
| **D2** | `lane_change` Δ event recall separated above | ✅ +0.0877 (pre-specified, uncorrected) — ⛔ **but its NULL twin earns the identical +0.0877** | ✅ +0.3509, **survives Holm** — ⛔ null twin +0.1579 SEP |
| **D3** | `intersection` NOT separated below | ✅ −0.0087, n.s. | ✅ +0.0866 (⛔ null +0.1558, so not claimable either) |
| **D4** | `lane_change` precision not separated below | ✅ **+0.0123 SEP ABOVE**, null-clean | ✅ **+0.0295 SEP ABOVE**, null-clean |
| **D5** | positive in **both** folds | ✅ **+0.0909 / +0.0833** — the most stable cell in the grid | ⚠️ +0.5454 / +0.0833 — same sign, **6.5× imbalance** |

⛔ **`SELECTION_ARTEFACT` fires for both** on the event-recall axis, per the pre-registered rule.
✅ **`DEPLOYABLE_WITH_COST` holds for both on the precision axis**, with the costs named in §5.
⛔ **`FROZEN_IS_BELOW_CHANCE` fires** and is reported in addition, per §5 of the pre-registration.

### Multiplicity, pre-registered (the parent had none)

| situation | cells separating above frozen, uncorrected | **surviving Holm–Bonferroni (24, α = 0.05)** |
|---|---:|---:|
| `lane_change` | 8 of 24 | **5** — `w4_L2.0`, `w16_L1.0`, `w32_L1.0`, `w32_L2.0`, `w32_L3.0` |
| `intersection` | 9 of 24 | **4** — `w16_L2.0`, `w8_L2.0`, `w32_L2.0`, `w32_L1.0` |

⚠️ **`CELL_w8_L1.0` does NOT survive Holm** (p = 0.00750 against a threshold of 0.00263). It is
admissible only because it was **pre-specified in §2 of the pre-registration**, outside the family —
and, as §2 above shows, on a metric that turned out to be artefactual anyway. Its **precision** gain
is what carries it.

### The oracle — and this one really is an upper bound

`C_ORACLE_POOLED` maximises the **pooled** delta, i.e. the same statistic the arms are judged by, so
it bounds every fixed-cell arm by construction. That is the repair of the parent's `C-ORACLE-PS`,
which maximised a *per-fold* statistic while the headline was pooled and therefore bounded nothing.

| situation | oracle Δ | **NULL oracle Δ** | winner's-curse share |
|---|---:|---:|---:|
| `lane_change` | +0.3509 | **+0.1579** | **45 %** |
| `intersection` | +0.0866 | **+0.1991** | **230 %** — the null exceeds the real oracle |
| `roundabout` ⛔ | +0.5135 | +0.0541 | 11 % |

**Prediction P-3 landed:** grid-max selection has a large winner's curse on this `n`.

### The honest re-tune procedure — and it is not a setting either

`RETUNE_SEL` selects one **shared** cell per fold on the SEL split by the **event** criterion — the
arm the parent explicitly named as missing (§10.1 of `PER_SITUATION_HORIZON.md`).

| arm | fold 0 | fold 1 | one cell in both? | `lane_change` | `intersection` |
|---|---|---|---|---|---|
| `RETUNE_SEL` | `W8, L2.0` | `W32, L1.0` | ⛔ **NO** | +0.2456 [+0.1250, +0.3704] **SEP** | **−0.0779 [−0.1350, −0.0171] SEP WORSE** |
| `NULL_RETUNE_SEL` | `W32, L1.0` | `W32, L1.0` | ✅ yes | +0.1579 **SEP** | +0.1558 **SEP** |

Three things at once: the real procedure **does not converge on one cell**, so it is not a shippable
setting; it **regresses the decision-grade situation**, separated; and **its own null converges on a
single cell and separates on both situations**. The selector had **5 and 7** `lane_change` onsets in
the two SEL splits — that `n` is the whole story, and it is reported rather than hidden.

---

## 5. ⭐ THE COST, PER BINDING FAMILY — never pooled

Full tables in `TABLES.md` §5, on the **causal** ego block with the legacy block run beside it.

| family | `W8, L1.0` | `W32, L1.0` |
|---|---|---|
| **TACTICAL** | `lane_change` AP-lift +0.0444 **n.s.** but operating-point precision **+0.0119 SEP**; `intersection` +0.0673 **n.s.** | `lane_change` +0.1820 **n.s.**, precision **+0.0330 SEP**; `intersection` −0.0576 **n.s.** |
| **LONGITUDINAL** ⚠️ *not computable as stated* (target-speed / headway / TTC need a **predicted path**; this arm emits a per-frame probability) — reported stratified by regime, n = 11,931…42,508 rows per stratum | `intersection` **steady +0.1240 [+0.0361, +0.3196] SEP BETTER**; all other strata n.s. | ⛔ **`intersection` decelerating −0.1840 [−0.3515, −0.0381] SEP WORSE** — the safety-critical regime |
| **LATERAL** ⚠️ same (heading / curvature / yaw-rate / cross-track need a predicted path) | `intersection` **straight +0.1008 [+0.0167, +0.1986] SEP BETTER** and ⛔ **turning −0.1515 [−0.3590, −0.0208] SEP WORSE** | `lane_change` **turning +0.1911 [+0.0034, +0.4269] SEP BETTER**; `intersection` turning −0.2222, n.s. |
| **STRATEGIC** | ⛔ **UNAVAILABLE** — no route/goal/map label exists on PhysicalAI-AV (settled at five probes: no map, lane graph, junction annotation or route signal; `egomotion` is clip-local metres with no GNSS). n = 56,979 rows | same |

⭐ **This is why the four families are binding.** `intersection` looks *neutral* pooled
(−0.0087 event recall, +0.0024 precision, both n.s.) and is **not** neutral inside: `W8, L1.0` is
separated **better on straight** and separated **worse on turning** — at intersections, which is
where turning is the manoeuvre. A pooled number would have shipped that invisibly.

### The non-metric costs

| cost | `W8, L1.0` | `W32, L1.0` |
|---|---|---|
| ridge parameters | **387 → 387 — zero change** | 387 → **1,539 (+1,152)** |
| history window | 0.8 s, unchanged | 0.8 s → **3.2 s** |
| declared warning horizon | **3 s → 1 s** — the tactical layer is told *later* | same |
| implementation | ⚠️ **`stack/scripts/sitclf_train.py` exposes `--win` but has NO `--lead`** — the lead is baked into the substrate's `y`. A lead change is a **label/substrate rebuild**, not a flag | same |
| comparability | ⛔ every banked lead-3.0 sitclf number becomes incomparable to the new head | same |

⚠️ **The horizon cost is real and is not a metric.** A 1 s warning horizon on a lane change may be
too late to be actionable regardless of precision. Nothing in this study measures that; it is a
tactical-layer requirement, and it is the PI's call, not mine.

---

## 6. PREDICTIONS — scored against what was written before measuring

| # | prediction | outcome |
|---|---|---|
| **P-1** | `FIXED_L1` separates on `lane_change` event recall, band +0.05…+0.12 | ✅ **LANDED** — +0.0877, in band — and **worthless**, because §2 shows the null twin earns the same |
| **P-2** | the best single fixed cell buys **less** than `C-GLOBAL`'s +0.3333 | ❌ **FALSIFIED** — `CELL_w32_L1.0` alone buys **+0.3509**, *more* than the two-cell mixture. The fold disagreement did not help `C-GLOBAL`; it cost it |
| **P-3** | the pooled oracle earns > +0.05 on permuted features | ✅ **LANDED** — +0.1579 / +0.1991 / +0.0541 |
| **P-4** | `FROZEN`'s `lane_change` event recall is **above** the chance floor | ❌ **FALSIFIED, and it is the finding** — 0.1404 against a floor of **0.8088**, and below the floor's 2.5th percentile on all three situations |
| **P-5** | `RETUNE_SEL` lands below the best fixed cell and may not separate | ⚠️ **HALF** — it landed below (+0.2456 < +0.3509) but did separate; the part I did not predict is that it **picks two different cells** and **regresses `intersection`, separated** |

**2 landed, 2 falsified, 1 half.** P-4's falsification is worth more than the other four combined.

---

## 7. ⭐ THE CODE CHANGE — the floor now travels with the number

The near-miss in §1 was possible because the promoted yardstick published `event_recall` with **no
reference level**. That is fixed in `stack`, additively:

`stack/tanitad/eval/sitclf_deploy.py::event_anticipation_report` now also returns
**`chance_event_recall`** (an exact hypergeometric expectation over a uniform alarm set of the same
size — not a simulation), **`chance_alarm_precision_h_max`/`_deploy`**, **`event_recall_vs_chance`**
and **`alarm_precision_lift_h_max`/`_deploy`**. A report can no longer quote the headline without
the floor, because they are keys of the same dict.

Four tests were added (`stack/tests/test_sitclf_deploy.py`, **43 → 47** in that file):

1. the floor keys **must** be present — "a headline with no floor is how a sub-chance operating
   point hides";
2. the analytic floor **matches a 400-run uniform simulation** to < 0.02;
3. ⭐ a **dispersed** score beats a **concentrated** one on event recall at the identical budget —
   the defect itself, pinned as a test so it cannot be forgotten;
4. a useless score lands at a precision **lift** of ≈ 1.0, so `< 1.0` is a genuine below-chance
   verdict.

Nothing existing in `stack/` was modified; every change is an addition.

---

## 8. The substrate defects, handled as instructed

**The ego block.** Every stratum number in §5 is on the **causal** rebuild
(`sitclf_b4_substrate.ego_causal.npz`); the quarantined legacy block was run beside it. On the
powered situations the shift is small — `intersection` LATERAL-turning moves **−0.1515 → −0.1470**,
a **3.0 %** change, and `lane_change` LATERAL-straight moves 12.0 %. **No conclusion here depends on
it.**

⚠️ **And it reconciles the parent's "2.1× change in a point estimate" rather than contradicting it.**
The parent measured `intersection` LATERAL-turning moving −0.0282 → −0.0604 on `PS_SEL`. The shift
is **roughly constant in absolute terms (~0.005–0.03)**; its *ratio* is dominated by how small the
denominator is. Both statements are true; the ratio is the misleading one to quote alone.

**The label provenance.** The situation labels are derived from ego dynamics
(`stack/tanitad/data/situations.py`), but the head's input window and the label's evidence window are
**disjoint**, so this is **same-source privileged access, not a future leak** — and no arm here reads
ego at inference in any case.

**Reproducibility, newly measured.** `C-FID-GRID` shows the parent's grid is **not bit-reproducible
on a re-run**: max abs diff **1.967e-06** across 25 cells, because `torch.svd_lowrank(niter=4)` on
CUDA is not run-to-run deterministic. Consequence measured, not assumed: **2 alarm-set rows move
across 25 cells × 3 situations.** `eval_rows` and `y_lead3` are bit-identical. Every real arm in this
report uses the parent's **banked** columns verbatim, so nothing here depends on the drift — but the
parent's `C-FID-PARENT` "bit-identical" claim is a statement about a *banked artefact*, not about
re-running the pipeline, and should be read that way.

---

## 9. Powered situations only, and which

| situation | positive clusters | status |
|---|---:|---|
| `intersection` | **216** | ✅ powered — decision-grade |
| `lane_change` | **55** | ✅ powered |
| `roundabout` | **37** | ⛔ **`UNDERPOWERED_C_POW`** (bar 40) — **no verdict**, bar not lowered |

⛔ `roundabout` is reported in every table **with its `n`** and decides nothing. It is worth one
line: its `CELL_w8_L1.0` shows the largest apparent gain in the whole study (event recall
0.2973 → 0.7297, TACTICAL +0.8586 SEP) — and it is **the only situation whose best arm reaches the
chance floor** (0.8108 against 0.8497). ⛔ **That is exactly the shape that gets misquoted as a win,
which is what the bar exists to prevent.**

---

## 10. ⭐ WHAT I DID **NOT** DO — stated plainly

1. **I did not re-litigate the per-situation null.** `NO_PER_SITUATION_GAIN_EXISTS` stands.
2. **I did not re-fit anything on new data.** Every real number is a re-analysis of the parent's
   banked out-of-fold columns over the same **500 clips**. `FIXED_L1` was pre-specified but **not
   blind** — it came from the parent's per-lead sweep on this same substrate. **There is no
   out-of-sample confirmation in this report**, and a decision to ship should want one.
3. **I did not change any deployed constant.** `lead_s = 3.0`, `WIN = 8` remain deployed. §0 is a
   characterisation of what the change would buy and cost.
4. **I did not lower `roundabout`'s bar**, and I did not issue it a verdict.
5. **I did not retrain the head at `lead_s = 1.0`.** The candidate's scores come from the parent's
   grid, which trained a *ridge* at each lead; a deployed transformer head at lead 1.0 is **not
   measured here**. ⚠️ B4 measured the 129-param ridge beating every transformer arm on
   `intersection`, so the ridge is the right floor — but that is INHERITED, not re-verified by me.
6. **I did not measure whether a 1 s warning is actionable** for the tactical layer. That is the
   decisive question for shipping and this study cannot answer it.
7. **I did not run any closed-loop or trajectory metric.** LONGITUDINAL and LATERAL are reported as
   regime-stratified decision quality, with the reason and the `n`, because this arm emits a
   probability and not a path.
8. **I did not touch a pod, and I did not push.**
9. **I did not edit `RETRACTION_LOG.md`** — §1's finding is a *new* measurement, not a correction of
   a recorded claim, and the parent's report never asserted the event-recall gain was skill; it
   banked it and flagged it as needing a powered study. §11 escalates it instead.

---

## 11. Verification and test suite

| check | result |
|---|---|
| pre-registration hash, re-verified after the run | see `MANIFEST.md` — must equal `7a62ca201603c7732b0bae0d56d13840c3979171` |
| `C-FID-QB` — every headline re-derived by the **promoted** `stack` function | **PASS**, 12 cells, **0 mismatches** |
| `C-FID-FLOOR` — analytic floor vs 200-draw simulation | **PASS**, |Δ| ≤ **0.0025** on all three situations |
| `C-BUDGET` — identical `n_alarm` across all 52 columns | **PASS** on all three situations |
| `stack` test suite | see `MANIFEST.md` for the exact counts |

---

## 12. What this redirects

1. ⛔ **Stop quoting `event_recall` deltas at a fixed global budget as evidence of skill.** They are
   dispersion measurements. The floor now ships inside the function (§7) so the mistake is harder to
   repeat; **the parent's `+0.3333` row should carry this caveat wherever it is cited.**
2. ⭐ **The deployed sitclf is BELOW CHANCE on `lane_change` at its own operating point**
   (precision-lift **0.8835**). That is a bug-class finding about the *deployed* system, independent
   of any re-tune, and it is the strongest single number in this stream.
3. ⭐ **`lead_s = 1.0` at `W = 8` is the cheapest real improvement available: 0 parameters,
   0.8835 → 1.3453 chance-lift, null-clean.** It needs (a) a substrate rebuild at lead 1.0, (b) a PI
   ruling on whether a 1 s horizon is actionable, and (c) acceptance of a separated
   `intersection`-turning regression.
4. **An out-of-sample confirmation is the missing evidence.** Everything here is one 500-clip
   substrate; the corpus for a second one does not exist in this study.
5. **The ceiling is still the trunk.** Window, motion basis, head capacity, horizon and now
   operating point have each moved this metric by less than the gap to a competent classifier.
   BACKLOG **B5** (video-pretrained trunk) is unchanged and reinforced.
