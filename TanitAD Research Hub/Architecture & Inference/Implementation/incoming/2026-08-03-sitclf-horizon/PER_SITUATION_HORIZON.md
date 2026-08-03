# Does a PER-SITUATION `(window, lead_s)` beat the single frozen setting?

**Date** 2026-08-03 · **Stream** sitclf per-situation horizon · **Substrate** dev box, **0 pod
GPU-h** — no pod was touched; `tanitad-new` (v5f) and `tanitad-pod4` (v1arch) were left training.
**Pre-registration** `./PRE_REGISTRATION.md`, written and staged **before** any number below was
read. **Tables** `./TABLES.md`, generated from the JSONs by `render_tables.py` — nothing retyped.

⛔ PI ruling 2026-08-03 honoured throughout: **labels may use ego; INFERENCE IS VISION-ONLY.** Every
arm reads frozen v1 camera latents and nothing else. The ego block appears only as the
LONGITUDINAL/LATERAL **stratification** variable, and this run uses the **causal** rebuild (§5).

---

## 0. ⭐ THE ANSWER — NO

**A per-situation `(window, lead_s)` does not beat the single frozen setting.** Not on the
decision-grade situation, not on the other powered one, and not on either of the two questions the
finding decomposes into.

| situation | pos. clusters | **Q-A verdict** (train-horizon) | **Q-B verdict** (deploy-horizon) |
|---|---:|---|---|
| **`intersection`** ⬅ decision-grade | **216** | **`NO_PER_SITUATION_GAIN_EXISTS`** | **`NO_PER_SITUATION_GAIN_EXISTS`** |
| `lane_change` | 55 | `NO_EFFECT_ABOVE_MDE` | `NO_PER_SITUATION_GAIN_EXISTS` |
| `roundabout` | **37** ⛔ | `UNDERPOWERED_C_POW` — **no verdict** | `UNDERPOWERED_C_POW` — **no verdict** |

On `intersection`, the out-of-fit per-situation arm lands **−0.1280 [−0.3126, +0.0291]** against the
frozen setting — the wrong sign, not separated, on a study whose median resolving power is 0.098
AP-lift. ⛔ **I did not weaken a threshold or re-scope an outcome.** My registered prediction (§6 of
the pre-registration: `NO_PER_SITUATION_GAIN_EXISTS` on `intersection`, ~65 %) came in — and the two
places I was wrong are in §4.

### And the mechanism, which is sharper than the verdict

**The per-situation selection criterion is noise.** Across the 6 fold×situation slots, `PS-SEL`
chose **6 different** `(W, L)` cells and **never the same cell twice for the same situation**:

| arm | distinct `(W, L)` chosen across 6 slots |
|---|---|
| `PS-SEL` | **6 of 6** — `(1,1)` `(1,5)` `(4,4)` `(4,5)` `(32,1)` `(32,5)` |
| `C-ORACLE-PS` ⛔ cheats | **6 of 6** — and it disagrees with `PS-SEL` on every slot |

A stable per-situation optimum would pick the same cell for the same situation in both folds. It
never does. That is *why* the arm cannot win, and it is visible without any interval.

---

## 1. ⭐ WHAT DID WIN — and it is not per-situation

The pre-registration did not anticipate this, and it is the actionable result of the study.

**On `lane_change`'s event yardstick, a SHARED re-tune beats the frozen setting by more than 3×, at
the same alarm budget and with BETTER precision.**

| arm | event recall | onsets warned | **n_alarm** (fixed) | alarm prec. @5 s | @3 s | median lead | Δ recall vs FROZEN |
|---|---:|---:|---:|---:|---:|---:|---|
| `FROZEN` (W8, L3.0) | 0.1404 | **8 / 57** | 2,849 | 0.0470 | 0.0228 | 3.65 s | — |
| **`C-GLOBAL`** (W32/W1, L1.0) | **0.4737** | **27 / 57** | 2,849 | **0.0653** | **0.0432** | 3.30 s | **+0.3333 [+0.1500, +0.5000] SEP** |
| `PS-SEL` | 0.1228 | 7 / 57 | 2,849 | 0.0267 | 0.0137 | 2.30 s | −0.0175 |

Three-and-a-third times the onsets warned, **for the same 2,849 alarms**, with precision up on both
horizons and the lead essentially unchanged. And its own control is clean: **`NULL_C-GLOBAL` earns
exactly `+0.0000 [−0.0678, +0.0656]`** against `NULL_FROZEN` on the clip-permuted substrate, so this
is not the selection procedure manufacturing a gain.

⇒ **The frozen `(W = 8, lead_s = 3.0)` IS suboptimal for `lane_change` — the fix is a different
SHARED setting, not a per-situation one.** Both folds' global selection chose **`L = 1.0 s`**, the
*shortest* lead tested.

⚠️ **This is `GAIN_IS_SELECTION_NOT_PER_SITUATION` in substance**, and the pre-registration named
that outcome in advance: *"re-tune the shared `(W, L)` once; do not build per-situation machinery."*

⚠️ **It does NOT replicate on `intersection`** — `C-GLOBAL` there is +0.0520, not separated. So this
is a `lane_change` fact, not a programme-wide one, and it is reported as such.

---

## 2. ⭐ THE CONTROL THAT TURNED OUT TO BE DEFECTIVE — and I found it by looking

`C-ORACLE-PS` was registered as *"the decisive control … the upper bound on any per-situation
gain"*. **It is not an upper bound, and it fires under the null.** Two independent demonstrations,
both from this run:

**(a) It is beaten by arms it is supposed to bound.** On `roundabout`, `PS-SEL` (lift **3.3687**)
exceeds the oracle (**2.4835**). On `intersection`, the fixed cell `CELL_w4_L2.0` (**1.6899**)
exceeds it (**1.4300**). The cause is structural: the oracle maximises the **per-fold** AP-lift
while the headline is the **pooled** AP-lift over both folds, and **AP is not decomposable across
folds** — two per-fold-optimal columns can pool worse than one fixed cell.

**(b) It earns a margin on pure noise.** Post-hoc (`oracle_winners_curse.py`, ⚠️ **not**
pre-registered, and it changes no verdict): the identical test-fold argmax run on the
**clip-permuted** substrate, where no per-situation gain can exist —

| situation | NULL oracle vs NULL frozen | REAL oracle margin | share that is winner's curse |
|---|---|---:|---|
| `lane_change` | **+0.2900 [+0.0553, +0.8104] SEP** | +0.8528 | **34 %** |
| `roundabout` | **+0.1089 [+0.0312, +0.2453] SEP** | +0.0239 | **4.54× the real margin** |
| `intersection` | +0.1048 [−0.0371, +0.2586] | **−0.1674** | the real oracle is *below* frozen |

⇒ **`lane_change`'s `NO_EFFECT_ABOVE_MDE` verdict rests on an oracle whose firing is a third
selection bias.** The verdict stands as recorded — ⛔ I have not re-read it — but the report must
say that the "a gain exists but is not selectable" reading is weaker than the label suggests.

⭐ **And it makes `intersection` HARDER, not softer.** There, a test-fold-*cheating* per-situation
selection lands **−0.1674 [−0.3439, −0.0122] SEPARATED BELOW** the frozen setting, while the same
procedure on pure noise lands *above* it. The decision-grade null is the strongest number in the
study.

---

## 3. Q-A — the full 5×5 grid, and the multiplicity that kills the two survivors

Evaluation label frozen at `lead_s = 3.0`; `L` moves only the head's training target; every cell
paired against `FROZEN` on identical rows (`valid(5.0) ∧ hist_ok(32)`).

### `intersection` — 5,889 pos / 50,975 rows, 216 positive clusters. MDE 0.168 widest / **0.098 median**

| arm | AP-lift | vs FROZEN | skill over its own permuted-feature null |
|---|---:|---|---|
| **`FROZEN`** (W8, L3.0) | **1.5974** | — | **+0.5994 SEP** |
| `PS-SEL` | 1.4693 | −0.1280 [−0.3126, +0.0291] | +0.5227 SEP |
| `C-GLOBAL` | 1.5562 | −0.0412 [−0.1952, +0.0804] | +0.4938 SEP |
| `C-ORACLE-PS` ⛔ | 1.4300 | **−0.1674 [−0.3439, −0.0122] SEP WORSE** | — |
| `CELL_w4_L2.0` | 1.6899 | +0.0925 [+0.0115, +0.1800] SEP | — |
| `CELL_w4_L3.0` | 1.6342 | +0.0369 [+0.0073, +0.0686] SEP | — |

⚠️ **Two of 25 cells separate above frozen — and NEITHER survives multiplicity.** At α = 0.05 over
25 comparisons the expected number of false positives is **1.25**, and the two survivors sit at
`p(δ>0) = 0.9880` and `0.9940` against a Bonferroni requirement of **0.999**. ⛔ **They are reported,
and they are not claimed.** The pre-registration did not register a multiplicity correction for the
grid; that is a defect in my design, disclosed here rather than exploited.

*(Directionally they are still interesting: both are `W = 4` — a **shorter** 0.3 s window than the
deployed 0.7 s — and `L ∈ {2, 3} s`. That is the same direction the sibling stream's horizon decay
predicts. It is a hypothesis for a powered study, not a result.)*

### `lane_change` — 1,472 pos / 56,979 rows, 55 positive clusters. MDE 0.287 / 0.120

**Zero of 25 cells separate above frozen.** `FROZEN` lift 1.2926; `PS-SEL` −0.0132 [−0.1081,
+0.0871]; `PS-SEL` − `C-GLOBAL` −0.0829, not separated.

### `roundabout` — 37 positive clusters ⇒ ⛔ `UNDERPOWERED_C_POW`, no verdict (bar 40)

Its `PS-SEL` point estimate is **+0.9091** with an interval of **[−0.6232, +2.9841]** — a MDE of
1.389, ten times `intersection`'s. ⛔ **The bar was not lowered**, and this row is exactly why it
exists: an unbounded interval with a large centre is the shape that gets misquoted as a win.

---

## 4. ⭐ WHERE MY REGISTERED PREDICTION WAS WRONG

§6 of the pre-registration committed two predictions. One landed; **one was falsified, cleanly.**

| prediction | outcome |
|---|---|
| `NO_PER_SITUATION_GAIN_EXISTS` on `intersection` (~65 %) | ✅ **landed** |
| *"On Q-B I predict both powered situations select `L = 5.0`"* — because a head trained to fire earlier should win a budget-matched event-recall race | ❌ **FALSIFIED.** Both powered situations select the **SHORTEST** lead: `lane_change`'s best cell is `W32, L1.0` and `C-GLOBAL` chose `L = 1.0` in both folds; `intersection`'s best is `W16, L2.0` |

The reasoning was wrong in an instructive way: I assumed "trained at a longer lead ⇒ fires earlier ⇒
covers more onsets". What actually happens is that a longer training lead makes the target **more
diffuse** — the positive set grows from 633 to 2,692 frames on `lane_change` — so the head spends its
fixed 5 % budget on a blurrier region and covers *fewer distinct* onsets. Sharper target, better
event coverage, at equal cost.

---

## 5. P4 — the B4 substrate's ego block, audited over ALL 500 clips and rebuilt

**Status: the defect is confirmed, sized on the whole substrate, fixed by a sidecar, and the
substrate is stamped.** The sibling stream established this from **clip 0 alone**; this is the full
audit (`rebuild_causal_ego.py` → `ego_leak_audit.json`).

The banked `E` block is **`LEGACY_LEAKY`** — a **bit-exact** match to
`kinematics(..., causal_pre=False)` (max abs diff **0.000e+00** over all 99,477 rows) and
**0.3055** away from the causal rebuild.

| channel | mean abs change | p99 | max | % of the channel's own scale | rows changed > 1 % |
|---|---:|---:|---:|---:|---:|
| `v` | 0.000000 | 0.000000 | 0.000000 | **0.00 %** | 0.0 % |
| `alon_pre` (m/s²) | 0.030873 | 0.176098 | 0.611039 | **4.56 %** | **73.7 %** |
| `omega_pre` (rad/s) | 0.003020 | 0.020819 | 0.074957 | **3.26 %** | **52.1 %** |

⭐ **And the number nobody had — what it does to the CONSUMER.** The only thing that reads `E` is
`sitclf_deploy.regime_strata`, i.e. the LONGITUDINAL/LATERAL stratum boundaries:

| stratum | rows (causal) | rows (legacy) | **reassigned** |
|---|---:|---:|---:|
| `longitudinal.steady` | 50,987 | 51,027 | **1,960 (1.97 %)** |
| `longitudinal.accelerating` | 28,254 | 28,276 | 1,108 (1.11 %) |
| `longitudinal.decelerating` | 20,236 | 20,174 | 852 (0.86 %) |
| `lateral.straight` / `.turning` | 59,871 / 39,606 | 59,840 / 39,637 | 1,035 (1.04 %) |
| `longitudinal.low_speed_lt8` / `.cruise_ge8` | 66,608 / 32,869 | identical | **0** — `v` is not a `*_pre` channel |

**Which arms were scored on the leaky block:** ⛔ **none, as model inputs** — ego is not a legal
inference input, so no arm in B4, in the sibling temporal study, or in this one reads `E`. What was
affected is **stratum membership** for ≤ 1.97 % of rows in the LONGITUDINAL/LATERAL families of the
sibling stream's `results_four_families.json` and B4's. A paired within-stratum contrast stays valid;
the boundaries were simply not the causal ones.

**Action taken** (`MANIFEST.md` §3): a **new** `…/sitclf_b4_substrate.ego_causal.npz` sidecar with a
provenance blob, and an `ego_block_defect` **quarantine stamp** written into the substrate's own
`.meta.json`. ⛔ **The 410 MB substrate was deliberately NOT rewritten** — that would have broken the
bit-reproducibility this study's own `C-FID-PARENT` depends on. This run's four-family tables use the
**causal** block, and re-run the legacy one beside it so the difference is measured, not asserted.

---

## 6. The controls, and why the null is interpretable

| control | result | what it licenses |
|---|---|---|
| **C-FID-PARENT** | **PASS** — reproduces the sibling stream's banked lead-3.0 row **bit-identically** on all three situations (`lane_change` AP 0.02841, `roundabout` 0.03822, `intersection` 0.16607, and both interval bounds) | this pipeline IS the one that produced the banked table |
| **C-FID-RIDGE** | max abs diff **0.0e+00** at every window vs `stack`'s `sitclf.ridge_scores` | the λ-shared speed path is the banked estimator, not a re-derivation |
| **C-POW** (written to disk before any score was read) | `intersection` **216** · `lane_change` **55** · `roundabout` **37** ⛔ | roundabout gets **no verdict**; the bar was not lowered |
| **C-SEL-NULL** | `intersection` −0.0514 · `lane_change` −0.0709 · `roundabout` +0.0557 — **none separated** | the per-situation selection procedure does **not** manufacture a gain |
| **C-SEL-NULL is NOT vacuous** | on the permuted substrate the argmax still chose **5 distinct cells across 6 slots** | it had real variation to exploit and did not profit — unlike the E-SEL stream's `C-shuffled` leg, which was uniform-random by construction |
| **Q-B null controls** | `NULL_C-GLOBAL` vs `NULL_FROZEN` event recall: `lane_change` **+0.0000**, `intersection` −0.0390, neither separated | §1's `+0.3333` is not a selection artefact |
| **C-ORACLE-PS** | ⚠️ **DEFECTIVE — see §2** | its non-firing on `intersection` is still strong; its firing elsewhere is worth little |
| **INDEPENDENT VERIFIER** | `verify_event_yardstick.py`: the Q-B yardstick re-derived from the banked scores by the **promoted `stack` function** — **12/12 arm×situation cells, 0 mismatches** | two implementations agree; the JSON does not merely agree with itself |

---

## 7. P2 — PRECISION **WITH** RECALL, AND BOTH DENOMINATORS (binding)

*The shape never to repeat: `brake_stop 0.026 → 0.503, a free win` — wrong denominators, no
precision (0.2340 → 0.1711, 380 fires for 153 true), not separated.*

| situation | arm | **precision** | recall | **n_alarm** (prec. denom.) | **n_pos** (recall denom.) | tp | base rate | prec.-lift |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `intersection` | `FROZEN` | **0.2546** | 0.1102 | 2,549 | 5,889 | 649 | 0.11553 | 2.204 |
| `intersection` | `PS-SEL` | **0.2346** | 0.1016 | 2,549 | 5,889 | 598 | 0.11553 | 2.031 |
| `lane_change` | `FROZEN` | **0.0228** | 0.0442 | 2,849 | 1,472 | 65 | 0.02583 | 0.883 |
| `lane_change` | `PS-SEL` | **0.0137** | 0.0265 | 2,849 | 1,472 | 39 | 0.02583 | 0.530 |
| `lane_change` | `C-GLOBAL` | **0.0432** | 0.0836 | 2,849 | 1,472 | 123 | 0.02583 | 1.671 |

Full table for every arm in `TABLES.md` §3. Note `FROZEN`'s precision-lift on `lane_change` is
**0.883 — BELOW 1.0**, i.e. at its own operating point the deployed setting is *worse than chance*
on that situation, which is a finding the AP-lift of 1.2926 hides completely. `C-GLOBAL` moves it to
**1.671**, and `PS-SEL` makes it **worse (0.530)**.

---

## 8. THE FOUR BINDING FAMILIES — per family, never pooled

Full tables in `TABLES.md` §5 (`results_four_families.json`), on the **causal** ego block, with the
legacy block re-run beside it.

| family | status here | reason, with its `n` |
|---|---|---|
| **TACTICAL** | ✅ **COMPUTED** for every arm × situation | situation anticipation *is* the tactical decision. AP-lift + paired CI + **the operating point with precision, recall and both denominators** + median anticipation lead |
| **LONGITUDINAL** | ⚠️ **not computable FOR THIS ARM CLASS** — reported as decision quality **stratified by longitudinal regime** (decelerating / steady / accelerating / low-speed / cruise) | target-speed accuracy and headway/time-gap/TTC need a **predicted path**; a situation classifier emits a per-frame probability. ⭐ **The instrument now EXISTS** — `taniteval.lead_metrics.distance_keeping(paths, leads, …)` landed today (`49e2229`) and closed this hole **for trajectory arms**. It does not apply here, and the reason is the arm's output type, not a missing metric. n = 50,975 (`intersection`) / 56,979 (`lane_change`) rows |
| **LATERAL** | ⚠️ same — stratified by lateral regime (straight / turning) | heading, curvature, yaw-rate and cross-track error need a predicted path. Same n |
| **STRATEGIC** | ⛔ **UNAVAILABLE** | no route/goal/map label exists on PhysicalAI-AV — settled at five probes (no map, lane graph, junction annotation or route signal; egomotion is clip-local metres with no GNSS). n = 50,975 / 56,979 rows |

---

## 9. P3 — THE TRAINER GAP

### ⚠️ First, a correction: the gap is not what the brief says it is

*"The situation classifier still has no promoted trainer"* is **FALSE AT HEAD.**
**`stack/scripts/sitclf_train.py` exists** — 245 lines, landed in commit **`49e2229`** the same day
the sibling stream wrote that sentence, and it **already exposes `--win` for exactly this study**
(its docstring says so verbatim). It refuses `--features ego` with a non-zero exit, enforcing the
vision-only ruling in code.

**Root-cause class: absence found at ONE location is not absence** — the operating standard's own
rule, and the claim was true when written. Probed three ways: `ls stack/scripts | grep -i sit`,
`git log -- stack/scripts/sitclf_train.py`, and reading the module.

### What would actually have to be built — and it is NOT parameters

⛔ **The result is null, so per the pre-registration's `NO_PER_SITUATION_GAIN_EXISTS` branch: do NOT
build it.** Costed anyway, in advance of any future positive, in `trainer_gap_cost.json`:

| item | where | why it is not just a flag | param cost |
|---|---|---|---:|
| per-situation **WINDOW** | `stack/scripts/sitclf_train.py` | `causal_window()` is called **once** and its `keep` mask filters `X`, `Y`, `V` and `clip` together (`:172-174`). Three windows ⇒ three different `keep` masks ⇒ the single-array contract breaks: it needs **three heads**, each with its own row set, plus an emit contract for three score columns | **−336 … +1,152** on the ridge floor |
| per-situation **LEAD** | the **substrate builder**, not the trainer | the trainer has no `--lead`; the lead is baked into the substrate's `y` `[N, S]` at ONE `lead_s`. Three leads ⇒ three label sets | **0** |

⭐ **Capacity is not what makes this expensive.** B4 MEASURED that on `intersection` the **129-param**
ridge (lift 1.677) beats **every** transformer arm (best 1.486 at 423,172 params), so the capacity
axis for this head is the **ridge**: a per-situation window costs **−336 to +1,152 params**, inside
the band of accepted levers (**+897 / +385 / +128**). *(Three per-situation transformer heads would
cost **+833,282** — over 3× the **+272,001** lever this programme rejected — but B4 says the
transformer is not the arm.)*

**The real cost is a contract change**: the tactical layer must be told which horizon each channel
carries, and every banked sitclf number becomes incomparable.

---

## 10. ⭐ WHAT I DID **NOT** DO — stated plainly

1. **I did not run a Q-B-criterion selection.** `PS-SEL` selects on the **Q-A** criterion (SEL-fold
   AP-lift against the frozen lead-3 label) for *both* questions, because §4 of my pre-registration
   defines the arm without naming a per-question criterion. So on Q-B, `PS-SEL` is not the arm that
   question deserves. ⛔ **I did not add one after seeing that `C-GLOBAL` wins** — that is exactly
   the post-hoc arm addition the parent pre-registration forbids. It is a work item.
2. **I did not register a multiplicity correction** for the 25-cell grid (§3). The two surviving
   cells are reported and not claimed.
3. **I did not re-select the deployed constants.** `lead_s = 3.0` and `WIN = 8` remain deployed. §1
   is a *characterisation* of what a shared re-tune would buy on `lane_change`; acting on it needs
   its own powered, pre-registered study.
4. **I did not touch `roundabout`'s bar.** 37 clusters, no verdict, reported with its `n`.
5. **I did not rewrite the 410 MB B4 substrate** — sidecar + quarantine stamp instead (§5).
6. **I did not re-verify the sibling stream's still-frame or subspace results.** They are INHERITED
   here and load-bearing for nothing I claim.
7. **I did not edit `RETRACTION_LOG.md`.** The sibling's citation escalation is **already logged** as
   `R-2026-08-03-cite` (checked, not assumed). My own stale-absence correction (§9) is escalated in
   the report instead, because the file was concurrently staged-modified by another agent and the
   git-hygiene rule makes a contended edit the wrong move.
8. **No new eval was run on the parity cache.** Everything is a within-substrate paired contrast on
   the 500-clip B4 rebuild; absolute APs are **not** comparable to the banked parity table.

---

## 11. What this redirects

1. ⛔ **Do not build per-situation head machinery.** The pre-registered null fired on the
   decision-grade situation, and the selection criterion is demonstrably noise (§0).
2. ⭐ **Do pre-register a powered study of a SHARED re-tune**, specifically `lead_s` at the short
   end. Two independent signals point there: `lane_change`'s +0.3333 SEP event-recall gain at
   `L = 1.0` (§1), and the sibling stream's monotone `intersection` decay (+0.982 → +0.378 over
   1→5 s). ⚠️ Both powered situations preferred the **shortest** lead, which is the opposite of what
   I predicted and worth understanding before it is bought.
3. **Fix `C-ORACLE-PS` before reusing the template** (§2): an oracle must be evaluated on rows it did
   not select on, or it is a control that fires under the null.
4. **The ceiling is still the trunk.** Nothing downstream of the frozen v1 encoder has moved this
   metric in three consecutive studies — window, motion basis, head capacity, and now horizon. That
   is BACKLOG **B5** (video-pretrained trunk), unchanged and reinforced.
