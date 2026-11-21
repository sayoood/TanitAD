# RESULT — the VISION-ONLY decoded lead-gap head for refav1's distance-keeping cost (`D-REFAV1-DK-DECODED`)

**Evidence class: MEASURED (ours).** Rig: **dev-box RTX 4060 only** — ⛔ neither named
GPU was touched (the A40 is running refcv5's training, Thor a sibling's eval; the 4060
showed no python compute before the run).
**Pre-registration:** `SPEC.md`, committed **`c9f736a`** *before any head was fit and before
any number below existed* (verified in HEAD by blob comparison and by content marker).

Raw: `raw/decode_panel.json`, `raw/reprice_oracle_only.json`, `raw/reprice_decoded.json`,
`raw/diagnosis_recalibration.json`, `raw/operating_point_sweep.json`,
`raw/near_range_lever.json`, `raw/bank_meta.json`, `raw/bar3_windows*.json`, the logs and
the seven scripts in `code/`.

⛔ **TIER.** The decode panel and every re-pricing are **representation / label arithmetic** —
no tier, because nothing is rolled. The A/B ladder is **T1** (self-action open loop).

---

## 0. The one-line answer

⛔ **NO — refav1 cannot yet keep distance from vision alone, and the pre-registered bar says
so: BAR-2 FAILED at sensitivity 12/21 = 0.5714 against a committed 0.80 (16/21 = 0.7619 after a
post-hoc bias correction — still a miss).**
⭐ **But the decode is real and the failure is localised to one number.** The head beats the
raw-pixel floor by a separated paired margin (**+0.4375 [+0.2639, +0.5815]**), and at
**matched specificity 0.8841** it reproduces **16 of the oracle's 21** re-ranking flips
against a raw-pixel floor's **13** and a no-perception constant's **11**.
⭐ **And on the real checkpoint the vision-only arm DOES act: BAR-3 PASSES.** The planner's
constant-velocity degeneracy broke on **3 of 29** windows from vision alone, against the
oracle's **10**, with the LATERAL family **bit-identical** on every arm — so refav1's
acceleration channel is now priced by something it can see. **It retains 30 % of the ceiling.**
⭐⭐ **The remaining gap to the ceiling is two measured quantities.** (1) The head's error is
larger than the decision it feeds: MAE **11.13 m** against a mean shortfall of **9.10 m** —
though on the near range where the barrier lives the SAME head reads MAE **3.41 m**. (2) The
**arming gate** is the bigger single loss: the vision presence head **missed 10 of 29 real
leads** (`gate_fn` 10, `gate_fp` 0), because its decode is almost entirely clip identity.

---

## 1. ⛔ THE CONTROL THAT CHANGED THE READING — the ceiling is partly a SPEED fact

Before any decoded number is quotable, one control had to be run that the ceiling result did
not carry: **what does a gap predictor with NO PERCEPTION AT ALL score on the 21/21
re-ranking test?**

The barrier is `gap < d0 + tau·v0`. Hand it a **constant** gap and the firing decision
collapses to a **speed threshold** — and the violating windows are the fast ones (16.44 m/s
mean against 8.00 m/s in the adequate block, the predecessor's own measurement).

| arm (identical procedure, training-fold-only median debias) | sensitivity | specificity |
|---|---|---|
| **`oracle_label`** — the published CEILING | **21/21 = 1.0000** | **69/69 = 1.0000** |
| ⭐ **`field` — the decoded vision head** | **16/21 = 0.7619** | 61/69 = 0.8841 |
| `pix` — RAW-PIXEL FLOOR | 13/21 = 0.6190 | 61/69 = 0.8841 |
| ⛔ **`constant` — NO PERCEPTION AT ALL** | **11/21 = 0.5238** | 64/69 = 0.9275 |

⇒ ⛔ **A constant reproduces 11 of the oracle's 21 flips.** So *"the term re-ranks 21/21"* is
a correct statement about the term that is **about half a statement about `v0`**, and the
incremental value of the gap on this decision is at most ten windows. That is a correction to
the CEILING's interpretation, not to its arithmetic — `D-REFAV1-DK-COST`'s 21/21 reproduces
exactly (§3).
⭐ **And it is what makes the decoded number readable**: at the *identical* specificity
0.8841, `field` 16 > `pix` 13, and `constant`'s 11 comes at a *more permissive* bar for
itself (0.9275). The decode is worth **+5 windows over no perception and +3 over raw pixels**.

---

## 2. BAR-1 — DECODE: **PASS**

Cross-fitted over the **141-clip v7.2 EVAL corpus**, 5 clip-disjoint folds, label-free fold
assignment (sha256 of `clip_id`). ⛔ Stage-A channel PCA, stage-B row PCA, the ridge λ
(clip-grouped inner CV) and the fit-mean all fit **inside the training folds only**; every
scored prediction is **out-of-fold**. Bank: **141 clips / 14,237 rows / 3,991 LEAD rows**.

| cell | n rows / **clusters** | d | ⭐ `field` | 95 % CI | `pix` FLOOR | `constant` | shuffle | **paired `field − pix`** |
|---|---|---|---|---|---|---|---|---|
| **`gap_all_lead`** | 3,991 / **85** | 128 | **+0.4040** | [+0.2481, +0.5183] | **−0.0335** | **+0.000000** | +0.1542 | **+0.4375 [+0.2639, +0.5815] EXCLUDES 0** |
| **`gap ≤ 30 m`** | 2,383 / **64** | 128 | **+0.6586** | [+0.5621, +0.7298] | −0.0568 | **+0.000000** | **−0.0066** | **+0.7154 [+0.6096, +0.8209] EXCLUDES 0** |
| `present` | 7,120 / **128** | 128 | +0.1894 | [+0.0627, +0.3044] | −0.1393 | **+0.000000** | **+0.1702** | +0.3287 [+0.1844, +0.4831] EXCLUDES 0 |

⭐ **PASS on all three committed criteria**: the paired `field − pix` CI excludes zero, the
constant control reads **exactly +0.000000**, and the within-clip shuffle is below `field`.

⭐ **The `≤ 30 m` cell is the one M84 measured, and it is much stronger here: +0.6586 against
M84's +0.3632**, with **within-clip +0.7199 against a within-clip shuffle of −0.0066** —
i.e. essentially all of it tracks the lead **as it moves**, none of it is clip identity.
(M84 was the TRAIN corpus; this is EVAL. Two corpora, and the numbers are not interchangeable.)

⚠️ **The honest readable quantity on `gap_all_lead` is TRUE − SHUFFLED = 0.4040 − 0.1542 =
+0.2498**, because that cell's within-clip shuffle is **not** zero. On the `≤ 30 m` cell the
shuffle is −0.0066 and no such deduction applies.

⛔⛔ **AND THE PRESENCE HEAD IS ALMOST ALL CLIP IDENTITY — this is the M84 agent-count lesson
repeating, and it must not be buried.** `present` reads +0.1894 with a separated paired
delta, but its **within-clip skill is NEGATIVE (−0.1787)** and its within-clip shuffle reads
**+0.1702**, so **TRUE − SHUFFLED = +0.0192**. ⇒ **the vision-only ARMING GATE is weak**, and
any claim resting on it is weak with it.

**Freeze proven, not asserted:** 407 trunk tensors, **0** sha256 mismatches,
**max |delta| = 0 exactly**, before and after.
**Label-free alignment proven, with a control that has power:** 14,237 rows cross-checked,
true join **3.20e-05 m/s**, and a deliberate **one-step MIS-JOIN control reads 1.24 m/s max /
0.095 m/s mean**. ⚠️ The first version of that check looked for a `speed` key the v2ep **does
not have**, so it silently never ran and reported `0.000e+00` — *a vacuous check reads exactly
its own pass value.* It was repaired and given the mis-join control before any number here.

⭐ **THE HEAD IS ONE DOT PRODUCT, AND THE COLLAPSE IS VERIFIED NUMERICALLY.** The two-stage-PCA
+ ridge pipeline collapses exactly to `gap_hat = (pool(_last_state) * V).sum() + bias` per
fold. Checked on every fold before the bundle is written: **max relative error 2.74e-07 over
15 folds**, and a relative error above 1e-4 is a refusal — a head the planner evaluates
differently from the head that was scored is two heads.

---

## 3. BAR-2 — RE-RANKING REPRODUCTION: ⛔ **FAIL**

**The oracle reproduction control passes exactly first**, which is what makes the rest
readable: **90 LEAD windows, 21 in violation, 21/21 flips — `reproduces_published: true`**,
recovering `D-REFAV1-DK-COST`'s published counts on the banked 141-episode 21109 panel.

| | committed bar | ⭐ **the PRE-REGISTERED head** | best POST-HOC (§4) | |
|---|---|---|---|---|
| **SENSITIVITY** | ≥ 0.80 (≥ 17 of 21) | **12/21 = 0.5714** | 16/21 = 0.7619 | ⛔ **MISS, both** |
| **SPECIFICITY** | ≥ 0.70 (≥ 49 of 69) | 62/69 = 0.8986 | 61/69 = 0.8841 | ✔ pass, both |

⛔ **BAR-2 FAILS as written, and it is reported as written — with the PRE-REGISTERED arm's
number first.** The head as specified reads **12/21 = 0.5714**. The 16/21 column is the head
*after* the training-fold-only median bias correction of §4, which was derived **after** seeing
this failure and is therefore post-hoc; it is quoted because it is the honest best this head can
do, not because it is the registered result. **Neither clears the bar**, so the verdict is the
same either way.

⇒ ⛔ **DECODABILITY WAS NECESSARY AND NOT SUFFICIENT (`C131`)** — stated in exactly those
words, as the SPEC committed. A gap that decodes at +0.4040 against a −0.0335 pixel floor
still does not price the plan well enough to reproduce the oracle's re-ranking.

---

## 4. THE DIAGNOSIS — a SHRINKAGE, and its unsafe half

A refutation is a waypoint. The mechanism is measured, not guessed
(`raw/diagnosis_recalibration.json`):

* regressing truth on prediction gives **truth = 1.362 + 0.896 · pred**, spread ratio
  **0.724** ⇒ **the head's spread is shrunk toward the fit mean**, which is optimal for
  squared error and **exactly wrong for a one-sided barrier**;
* ⛔ **bias on gaps < 30 m: +8.36 m** (it thinks the lead is further than it is ⇒ the term
  **under-fires**) against **−7.61 m** on gaps ≥ 30 m. **The unsafe half is the near half.**

⭐ **THE LEVER, RUN:** a per-fold linear recalibration `g' = α + β·g_hat` with α, β fit on the
**training folds only**. It moved sensitivity **12/21 → 14/21** at unchanged specificity, and
the median-residual debias (§3) reaches **16/21**. It costs R² (**0.4040 → 0.3925**) exactly
as it should — it trades squared error for spread, which is what the barrier needs.
⛔ **It does not clear the bar.**

---

## 5. ⭐ THE WHOLE TRADE-OFF, because one operating point is not a characterisation

`raw/operating_point_sweep.json`. The offset for quantile `q` is the `q`-quantile of the
**training-fold** residual; the scored fold is never consulted. **Every arm is swept through
the identical procedure, the constant included.**

| q | `field` sens / spec | `pix` sens / spec | `constant` sens / spec |
|---|---|---|---|
| 0.10 | 0.0476 / 0.9855 | 0.0000 / 1.0000 | 0.0000 / 1.0000 |
| 0.25 | 0.3333 / 0.9855 | 0.3810 / 0.9855 | 0.3810 / 0.9710 |
| ⭐ **0.50** | **0.7619 / 0.8841** | 0.6190 / 0.8841 | 0.5714 / 0.8116 |
| 0.75 | 0.7619 / 0.5942 | 0.9048 / 0.4493 | 0.9048 / 0.4493 |
| 0.90 | 0.9048 / 0.3623 | 0.9524 / 0.2174 | 0.9524 / 0.2464 |

⛔ **NO arm clears BAR-2 at ANY operating point** — not `field`, not `pix`, not `constant`.
⭐ **`field` vs `constant` at MATCHED specificity** — the statistic that is not confounded by
where each arm sits on its own curve:

| specificity | `field` | `constant` | ⭐ advantage |
|---|---|---|---|
| 0.9855 | 0.3333 | 0.0000 | **+0.3333** |
| **0.8841** | **0.7619** | 0.3810 | ⭐ **+0.3810** |
| 0.5942 | 0.7619 | 0.5714 | +0.1905 |
| 0.3623 | 0.9048 | 0.9048 | +0.0000 |

⇒ **the gap decode carries real, separated information for the firing decision in the useful
region, and none at all once the operating point is permissive enough that everything fires.**
⚠️ **No point on this curve is "the answer."** Choosing one needs a safety criterion the
programme has not committed to (how many missed distance-keeping events are worth how many
spurious decelerations). Reporting the curve is the honest output; picking the point that
clears a bar **after** seeing the data is the goalpost move the rules forbid.

---

## 6. ⭐ THE NEXT LEVER, RUN — and the control that stops it being oversold

The pooled MAE (**11.13 m**) exceeds the decision margin (**9.10 m** mean shortfall), but it
is pooled over gaps out to 80 m that the barrier never touches. On the `≤ 30 m` cell the same
head reads **MAE 3.41 m** — well inside the margin. So: *would a correctly gated near-range
head clear the bar?* (`raw/near_range_lever.json`)

| arm | window set | sensitivity | specificity |
|---|---|---|---|
| ALL-RANGE head | all LEAD | 16/21 = 0.7619 | 61/69 = 0.8841 |
| pixel floor | all LEAD | 13/21 = 0.6190 | 61/69 = 0.8841 |
| constant CONTROL | all LEAD | 11/21 = 0.5238 | 64/69 = 0.9275 |
| ⭐ **NEAR head** | **NEAR ≤ 30 m** *(ORACLE-GATED CEILING)* | **14/16 = 0.8750** | 29/39 = 0.7436 |
| ⛔ **pixel floor** | NEAR | **13/16 = 0.8125** | 29/39 = 0.7436 |
| ⛔ **constant CONTROL** | NEAR | **12/16 = 0.7500** | 30/39 = 0.7692 |
| ALL-RANGE head | NEAR *(attribution control)* | 12/16 = 0.7500 | 34/39 = 0.8718 |

⭐ A perfectly gated near-range head **would** clear BAR-2 (0.8750 ≥ 0.80, 0.7436 ≥ 0.70).
⛔⛔ **BUT THE RAW-PIXEL FLOOR CLEARS IT TOO (0.8125 / 0.7436), AND THE CONSTANT NEARLY DOES
(0.7500 / 0.7692).** ⇒ **on the near subset the bar is not a discriminating instrument** —
restricting to ≤ 30 m removes precisely the windows a constant gets wrong. **The two-stage
design is therefore NOT established by this measurement**, and it must not be quoted as
"the near-range head passes". This is the specificity lesson one level up: a *subset* can be
easy the same way an operating point can be.
⚠️ The NEAR rows assume a **perfect near/far gate**, which does not exist. They are a ceiling
on a design, never a vision-only capability number.

---

## 7. BAR-3 — THE PLANNER A/B ON THE REAL CHECKPOINT: ⭐ **PASS**

Step-21,109 checkpoint, **same seed, same 29-window list, `--dk-gap-source` the only flag
moved**. The window list is the ORACLE's own LEAD windows in the 17 episodes carrying a
violating one — **identical for every arm**, so it is a measurement choice and never an input.

| arm | gate | gap | `cl` bit-identical to `ha0` | **Δ vs control** | const-vel frac | LATERAL bit-identical | `ha0` unchanged |
|---|---|---|---|---|---|---|---|
| **`off`** (`w_dk = 0`) — PARITY CONTROL | — | — | **26/29 = 0.8966** | — | 0.8966 | — | — |
| **`oracle_label`** — the CEILING | ORACLE | ORACLE | **16/29 = 0.5517** | ⭐ **−10** | 0.5517 | ✔ **True** | ✔ |
| **`decoded_gap`** — gap decoded | ORACLE | **DECODED** | **20/29 = 0.6897** | ⭐ **−6** | 0.6897 | ✔ **True** | ✔ |
| ⭐ **`decoded`** — VISION ONLY | **DECODED** | **DECODED** | **23/29 = 0.7931** | ⭐ **−3** | 0.7931 | ✔ **True** | ✔ |

⭐ **PASS on both committed criteria.** Every armed arm's exact-equality count moves **below**
the unarmed control's, and the **LATERAL family is bit-identical** on every arm — the
same-breath proof the term touched only the longitudinal channel — with `ha0` (which does not
plan) unchanged throughout.

⚠️ **The SPEC's phrasing was "moves off 1.0000", and that number was the predecessor's control
value on a 4-window dev-box slice.** On this 29-window panel the unarmed control is **0.8966**,
not 1.0000, so the criterion is read in its intended form — *does the armed arm differ from the
unarmed control?* — with both counts stated. **The goalpost is not moved; the control's own
value is simply reported rather than assumed.**

⭐⭐ **THE LADDER IS MONOTONE AND ATTRIBUTABLE, ONE VARIABLE AT A TIME:**

| step | what changed | windows lost |
|---|---|---|
| `oracle_label` → `decoded_gap` | the GAP: oracle label → vision head | **4 of 10** |
| `decoded_gap` → `decoded` | the GATE: oracle label → vision head | **3 more** |

⇒ ⛔ **the fully vision-only arm retains 3 of the oracle's 10 structural effects — 30 % of the
ceiling.** Both halves of the perception problem cost real windows, and **the GATE costs
proportionally more than the GAP** (it removes 3 of the 6 the decoded gap still had).

⛔ **AND THE GATE'S FAILURE MODE IS MEASURED, NOT INFERRED.** The `decoded` arm's plan-time
record reads **`n_armed` 19 of 29**, with **`gate_fp` = 0 and `gate_fn` = 10**: the vision
presence head **never armed on a non-lead, and missed 10 of the 29 real leads**. That is the
`present` cell's "almost all clip identity" (TRUE − SHUFFLED **+0.0192**) showing up
operationally — conservative, and blind to a third of the leads.

⭐ **THE END-TO-END CROSS-CHECK — the planner's head IS the scored head.** The arm re-encodes
`feats` inside the window loop; the bank read the fp8 cache in a separate process, days apart
in code path. On all **29** windows of both decoded arms the two gaps agree to
**max |Δ| = 8.141e-04 m**. ⇒ *"the head is in the loop"* is a measurement here, not an argument.

**Plan-time provenance, banked in every dump** (`goal_rule.distance_keeping_cost`): head
sha256 `89fc642bf63532db…`, version `dkgap-linear-v1+recal`, `ckpt_step` 21109, `fit_corpus`
`v7.2-eval-141`, `fit_scheme` `5fold-crossfit-clip-disjoint-seed0`, `n_fit_rows` 3465,
`n_fit_clips` 69, `gap_source`, `vision_only`, `is_ceiling` — and the label-free alignment
proof at plan time, `speed_check_max_mps = 1.12e-06`. ⇒ **a decoded run can never be read as
the oracle ceiling.**

⛔⛔ **NO ADE / HEADWAY / TTC CLAIM IS MADE.** n = 29 windows over 17 episodes carries no family
claim, and refav1's planner **samples**: its inference-seed floor is **≈0.30 m ADE** and no arm
here was run at ≥ 3 inference seeds. The quantities above are **exact-equality counts under a
fixed seed with one flag moved** — identities about this set of runs, not estimates.
⚠️ These counts are also not comparable to the banked full-panel figures: a 29-window subset
consumes the sampler's RNG stream differently from the 282-window panel. The comparison is
**internal to this ladder**, which is why all four arms were re-run.

---

## 8. BAR-4 — THE PROVENANCE GUARD: ⭐ **PASS, PROVEN BY MUTATION**

`stack/tanitad/refs/refav1_lon_cost.py` gains a **CLOSED** gap-source vocabulary. Before
today the field was a free-form string: **any** value was accepted and stamped into the dump
verbatim, so a typo or an optimistic label would have been **banked as fact**.
⛔ **The fix for "the decoded head does not exist yet" was never to relax the refusal — it is
to EXTEND the vocabulary and make every member carry what it is.**

| source | gate | gap | vision-only | ceiling | needs head |
|---|---|---|---|---|---|
| `oracle_label` | ORACLE | ORACLE | ✘ | ✔ | ✘ |
| `decoded_gap` | ORACLE | **DECODED** | ✘ | ✔ | ✔ |
| ⭐ `decoded` | **DECODED** | **DECODED** | ✔ | ✘ | ✔ |
| `unit-test` | — | synthetic | ✘ | ✘ | ✘ |
| `unset` | — | — | ✘ | ✘ | ✘ |

**`stack/tests/test_refav1_dk_gap_source.py` — 26 tests, all passing.** Each guard comes in a
PAIR: the bad input **raises**, and with the guard's own datum **mutated** the *same* bad
input is **accepted** — which is what proves the refusal came from the guard under test and
not from an unrelated error on the way past it. *(An AST census once read 0 suspects on BOTH
the fixed and the broken trainer; inspection is not evidence.)*

Guards, all committed in the SPEC before they existed:
1. an **unrecognised** source raises — mutation: widening `GAP_SOURCES` accepts it;
2. **armed + `unset`** raises — mutation: bypassing `__post_init__` accepts it *and the cost
   then happily prices it*, which is the danger;
3. a **decoded source without provenance** raises — mutation: flipping `needs_head` accepts it;
4. ⛔ an **ORACLE source CARRYING head provenance** raises — the more dangerous direction,
   because an oracle arm stamped with a head reads like a vision result;
5. a **short or blank sha256** raises (*the 2026-09-04 blob-comparison hole in stamp form: a
   "verified" comparison that passes on two failed reads*);
6. the arm **refuses a head whose `ckpt_sha256`/`ckpt_step`/`d_state`/geometry** disagree with
   the running checkpoint, and refuses a clip with **no out-of-fold head** rather than
   guessing a fold (which would leak that clip's own labels into its own prediction);
7. **parity**: declaring a source does not move a single number (`torch.equal`), with a
   same-breath control that must differ — so no banked comparison is confounded by its label.

---

## 9. ⛔ THE ADMISSIBILITY CHECK, ANSWERED

> *Could any input this head sees AT INFERENCE have been computed from something the LABEL was
> derived from?*

**Answer: NO for the gap value; and the question has a SECOND half that must be answered
separately, where the answer is YES for two of the three rungs.**

* **The head's only input** is `model._last_state(model.encode(feats))` — where `feats` is the
  cached DINOv3 patch-token window, i.e. **camera pixels and nothing else**. No ego state, no
  privileged channel, no future. It is taken from the **same `feats` tensor the very next line
  hands to `plan()`**, so it is the planner's own latent, not a second pipeline.
* **The label** is the B1 block's `gap0_m`, derived from **`obstacle.offline` 3-D cuboid
  tracks**. Those cuboids are **not an input to the trunk at any stage** — refav1 is trained on
  cached camera features and actions — so the head cannot be reading its label's own source.
  *(This is the check that dissolved the sitclf anomaly, applied here rather than assumed.)*
* ⛔ **THE SECOND HALF — THE ARMING GATE.** `oracle_label` and `decoded_gap` take their
  LEAD/NO_LEAD gate from the block and are therefore **NOT vision-only, however good their gap
  is.** Only the `decoded` rung is, and the module stamps exactly that (`vision_only` is True
  for `decoded` alone).
* ⚠️ **`v0` enters the COST (not the head)** — `s*(v) = d0 + tau·v` and the candidate
  integration. That is the measured ego speed at cycle time, **admissible under the PI's
  binding ruling of 2026-09-02**, and it is named here rather than left implicit — especially
  because §1 shows how much of the firing decision it carries.
* ⛔ **No closing rate was smuggled in.** The head predicts **gap and presence only**. M84
  measured the rate does not decode (+0.0061 [−0.0406, +0.0513]; the explicit temporal
  difference recovers +0.0145, still spanning zero), and a rate that *did* decode would be a
  new result needing its own pre-registration, not a quiet upgrade.

---

## 10. THE SUITE — a CONTROLLED comparison, not a bare "green"

The identical 35-file `refa_v1*` / `refav1*` subset, run twice against the same corpus, on the
same box, differing only in which source files were on disk:

| | files | result |
|---|---|---|
| **BASELINE** — the predecessor's staged (pre-edit) `refav1_lon_cost.py` + `refav1_arm.py`, restored from their **index blobs** | 35 | **403 passed, 1 skipped, 0 failed** |
| **POST-EDIT** — this agent's versions, + the new guard file | 36 | **429 passed, 1 skipped, 0 failed** |

**403 + 26 new = 429, exactly.** ⇒ no pre-existing test changed status; every new pass is a
new test.

⚠️ **Four tests DID fail on the first post-edit run, and they were the correct behaviour.**
`DistanceKeepingSpec(w_dk=1.0)` with no declared source now RAISES — which is BAR-4's committed
guard #2. The four call sites were arithmetic tests; they now declare `gap_source="unit-test"`,
which is itself a vocabulary member stamped **non-deployable** so a dump carrying it can never
be read as a run.

⚠️ **The predecessor's three DK files are STAGED BUT NOT IN `HEAD`** (correct under the
operating standard's *stage, never commit*). The baseline therefore had to be reconstructed
from **index blobs** (`git cat-file blob`), not from `HEAD` — `git show HEAD:<path>` returns
*"exists on disk, but not in 'HEAD'"* for all three, which reads exactly like a lost commit
and is not one.

---

## 11. WHAT IS STILL MISSING — named, with what unblocks each

1. ⛔ **THE HEAD'S ERROR IS LARGER THAN THE DECISION IT FEEDS.** MAE **11.13 m** against a
   **9.10 m** mean shortfall. This one number is the whole remaining gap to the oracle ceiling.
   *Unblocked by:* a better near-range head — and §6 says the obvious two-stage version is
   **not** established, because the pixel floor clears the same bar on the same subset.
2. ⚠️ **A FINER POOLING WAS THE OBVIOUS NEXT LEVER, AND A ZERO-GPU CONTROL JUST
   WEAKENED HALF OF IT.** The bank pools 16x40 -> **8x20** to match M84 exactly,
   halving the VERTICAL resolution -- and vertical image position is the classic
   monocular range cue, so "re-bank at 16x20" looked obvious. ⛔ But the head's
   weight map says otherwise: its vertical energy peaks in band 4 of 8 at **2.28x
   uniform**, and **the PRESENCE head -- whose decode is almost entirely clip
   identity -- has an essentially IDENTICAL vertical profile (cosine 0.9870)**.
   ⇒ the vertical structure is *where vehicles appear in this rig's frame*, not a
   range-specific cue. ⭐ **AZIMUTH is where the gap head is actually distinctive**
   (peak bin 3.68x uniform vs the presence head's 2.32x; centre two bins hold
   **0.327** of the energy vs **0.231**). ⇒ if a finer pooling is tried, the
   evidence points at AZIMUTH, not vertical -- and it needs its own
   pre-registration either way. *(Descriptive only: a weight map locates a linear
   functional's mass; it does not establish what the trunk encodes.)*
3. ⛔ **THE VISION-ONLY ARMING GATE IS WEAK** — `present` is +0.1894 raw but **+0.0192 after
   the within-clip shuffle**, and its within-clip skill is **negative**. Any deployable
   `decoded` arm rests on this. *Unblocked by:* a real detection head, not a ridge on pooled
   tokens.
4. ⛔ **CLOSING RATE — still absent** (M84 §4/§4b, including the refuted cheap fix). The cost
   prices a GAP and cannot price CLOSING. *Unblocked by:* the representation work item a
   sibling owns, not by anything here.
5. ⚠️ **THE HEAD IS FIT INSIDE THE EVAL CORPUS.** Cross-fitting makes it admissible — no
   clip's own labels ever enter its own prediction — but a deployed head would be fit on the
   TRAIN corpus. *Unblocked by:* the train fp8 cache, which is not on this box.
6. ⚠️ **NO ADE CLAIM IS MADE.** refav1's planner samples and its inference-seed floor is
   **≈0.30 m ADE**; nothing here was run at ≥ 3 inference seeds, so the readable quantities are
   the **exact-equality counts** and the **const-velocity fraction** only.

---

## 12. FILES

| path | what |
|---|---|
| `stack/tanitad/refs/refav1_lon_cost.py` | the closed gap-source vocabulary, `GapHeadProvenance`, the both-ways stamp check |
| `stack/tests/test_refav1_dk_gap_source.py` | **26 tests, every guard proven by MUTATION** (new) |
| `stack/tests/test_refav1_lon_cost.py`, `stack/tests/test_refa_v1_dk_hook.py` | 6 call sites now declare `gap_source="unit-test"` |
| `taniteval/tools/refav1_arm.py` | `--dk-gap-head` / `--dk-present-thr`, the in-loop decode from the planner's own `feats`, the head verification, the gate confusion matrix, `decoded_gap`/`decoded` in `--dk-gap-source` |
| `…/Research/2026-09-06-refav1-decoded-gap/` | `SPEC.md`, this document, `raw/`, `code/` |
