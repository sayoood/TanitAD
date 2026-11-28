# E-S1 — THE S1 CLIMB-OUT: implemented at **0 parameters**, and the hole it has to climb out of is **5.5×–19× smaller than anyone thought**

**Date:** 2026-08-04 (Europe/Berlin) · **Stream:** arch-inf (D-SEL, S1) · **GPU cost:** ~2 min of
an idle Jetson Thor across four inference dumps. **No training launched. No training pod touched.
Nothing committed, nothing pushed.**

**Pre-registration:** `PREREG_S1_CLIMBOUT.md` (this directory), content-pinned **before** the first
statistic — `raw/prereg_pin.json`, git blob **`8dc68daf8cf6a086dba5782045af9e35f1241b85`**,
sha256 `39eb43a1…`. ⚠️ It could not be `git add`ed first because the repo mount was down (§9); the
blob id is a deterministic function of content, so `git ls-files -s` **must** print that same id
once staged. That is the falsifiable object, and it is checkable by anyone.

**n = the canonical 881 windows / 40 episodes**, both arms. Estimator: **paired episode-cluster
bootstrap**, unit = episode, `n_boot = 2000`. ⛔ `overlapping_holdout_se` never called.

---

## 0. LEAD — the three things the brief asked for, up front

| question | answer |
|---|---|
| **Is the climb-out testable without training?** | **Partly, and the honest split was registered BEFORE measuring (§2 of the prereg).** The *staleness/readout* half is testable at 0 training and **was measured** (§3). The *supervision* half is **not** — a monotone re-weighting of a banked scalar has the identical argmax, so it can only reproduce the incumbent. Bounding it needs the refined pass's query features (a GPU dump, still 0 training); **measuring** it needs a retrain. |
| **What does it cost in parameters?** | **EXACTLY ZERO**, proven not asserted: `param_breakdown` is unchanged for `sel_score_emitted`, `sel_ce_reach`, both together, and the full climb-out preset. `refc_config()` stays at **104,191,577**; the D-SEL preset stays at **+385**. An all-off build is byte-identical (state_dict keys **and** values) and **bit-identical in the forward** across 6 flag combinations. |
| **What is the headline?** | ⭐ **S1 ranks on the single worst-but-one readout the decoder can produce, and 82 % (base) / 95 % (XL) of its deficit is a TIMESTEP TOKEN.** Pinning the emitted-fan readout to the token `loss_cls` actually supervises is **−1.6771 m** (base) / **−1.3243 m** (XL), separated, for **0 parameters**. The hole S1's CE must climb out of drops from **+0.8372 m → +0.1525 m** (base) and **+0.9147 m → +0.0480 m** (XL). |

⛔ **And the registered verdict, which is NOT that headline:** `S1_IS_DEAD_AFTER_ALL` **FIRED**. No
supervised ranker over the banked deployable readouts beats the incumbent selector out-of-episode —
every one is separated **worse** — and S1b at its default token is separated **adverse**. The
headline above is **POST-HOC** and is labelled that way everywhere it appears. It does not un-fire
the death; it says the next pre-registration should be about the token.

---

## 1. WHAT WAS IMPLEMENTED (P1) — two zero-parameter distribution matches

The brief's diagnosis was that *the supervision target and the evaluation distribution disagree*.
Tracing it in source found **three** objects where there should be one — the object that is
**SCORED**, the object that is **SUPERVISED**, and the object that is **EMITTED**:

```
_decode(kv, cond, x_in, t) -> (conf OF x_in, offset that improves x_in)
loop:  x = x_in + off                      # and refined := conf(x_in)
```

⇒ with `steps = 2`: `logits = conf(anchors)`, `refined = conf(X₁)`, **`anchor_traj = X₂`**. The
shipped ranker is **two** denoise passes stale, S1's refined ranker is **one** pass stale, and **the
trajectories that actually leave the decoder are scored by no head at all.**

| id | flag | what it makes agree | params |
|---|---|---|---|
| **S1b** | `sel_score_emitted` (+ `sel_score_emitted_t`) | the **scored** object becomes the **emitted** object: one extra conf-only pass on `X_k`, its offset **discarded** so `anchor_traj` is bit-unchanged | **0** |
| **S1c** | `sel_ce_reach` | the **CE's normalising set** becomes the **argmax's** set: today a full-fan softmax while the selector solves a 26–28 %-sized problem (73.76 % / 72.08 % of the fan is unreachable, never selected, and deleting it moves ADE by **exactly 0.0**). The target moves with the support, so the CE is never asked to put mass on a class its own softmax masked out | **0** |

**Why this is the minimal change.** Neither adds a weight, a head, or an input. S1b reuses the
decoder's existing `conf_head` and the existing `time_embed` table; S1c changes only which columns a
softmax normalises over. The alternatives were rejected in source: re-supervising `loss_cls` would
change the published control arm, and giving the refined readout its own head is a capacity change
that C34's rule forbids before attribution.

### 1.1 The admissibility gate — the D-SEL standard, re-run

`raw/verify_s1_identity.json`, on Thor, against the **pre-edit file loaded from a backup path**
(real bytes, not a remembered claim):

| check | result |
|---|---|
| `param_delta` for `{score_emitted}`, `{ce_reach}`, both, and the full climb-out preset | **0, 0, 0, 0** |
| `refc_config()` total / D-SEL preset delta | **104,191,577** ✔ registry · **+385** ✔ unchanged |
| all-off `state_dict` keys equal / values equal | **True / True** |
| forward **bit-identical** across `{}`, `sel_refined`, `sel_reach_clamp`, `graft_cons`, `seam_clamp`, and all three together | **True for every output, every combination** |
| S1b: `anchor_traj`, `offset`, `anchor_logits` bit-unchanged; `refined_logits` changed | **True** |
| S1b: `prefinal_logits` is bit-identically the readout S1 ships today | **True** |
| S1c: masked CE loss finite, all grads finite, target always a survivor, support < 1.0 | **True** |

**Tests (new, 12, all passing):** `stack/tests/test_refc_select_s1_climbout.py`. The trainer runs the
climb-out arm end to end and the banner reports `n_selection_params: 0` with
`ce_support: "reachable_only"`.

### 1.2 Two fail-louds, both earned by measurement

* `--sel-ce-reach` **without** `--sel-reach-clamp` is refused at parse time — without S2 there is no
  survivor mask, so the flag would be **silently inert** while `config.json` recorded it as ON. That
  is the D-TAC1 `tactical_speed_input` failure exactly.
* `--sel-score-emitted` **without** `--sel-refined` is refused for the same structural reason **and
  a measured one**: with S1 off the ranked score is `conf`, so the flag cannot reach the argmax —
  and at frozen weights the emitted readout is **0.9924 m worse** (§3.3). S1b must never ship
  without the supervision that is supposed to repair it.
* `--sel-score-emitted` with the token left at `-1` prints a **loud warning carrying the measured
  numbers**, because the default is the worse half of a 5.5× difference (§3.4).

---

## 2. THE PRE-REGISTRATION (P2) — three-sided, with direction predicates

`PREREG_S1_CLIMBOUT.md`, staged content-pinned before measuring. It honours both escalations that
this stream inherited:

* **E-SEL escalation 1** (*"§6.3 has no branch for separated-adversely"*): **every** two-sided
  quantity here has a **three-sided** table — `TARGET-EXISTS` / `TARGET-MARGINAL` /
  `NOT-SEPARATED` / **`ADVERSE`**, and the same for `C-lon` and `S1b`.
  ⭐ **It was needed: every single registered row came back `ADVERSE`.** Unlike E-SEL and
  S3_DEPLOYABLE, this experiment's decision table **could express what happened.**
* **S3_DEPLOYABLE escalation 1** (*"the trigger fired literally while both controls beat the
  score"*): every trigger reads *separated **AND** the delta favours the treatment*, implemented as
  a `_sep_dir()` predicate that returns **both bits** and is applied mechanically in `adjudicate()`.
  No row is chosen by hand.
* **"Do not repeat a control that cannot fail":** `C-shuffled` is declared **VACUOUS BY
  CONSTRUCTION** in the code (`can_fire: False`) and is never load-bearing. Five controls that
  **can** fire were designed instead, and **two of them did**.

---

## 3. THE MEASUREMENT (P3) — 0 GPU-days, on the banked fans

**Protocol:** a per-candidate scorer `s_i = w·φ_i` fit by listwise softmax CE against the oracle
index, **restricted to the S2-reachable survivors**, evaluated **leave-one-episode-out** (folds
asserted disjoint per fold). Headline = **selection ADE@2s**, paired against the shipped selector.
⛔ ρ is a secondary diagnostic only, and only on the reachable subset — S3_DEPLOYABLE §3 measured
that ρ over the full candidate axis is disconnected from selection for *every* score.

### 3.1 The registered panel — `refc-base-30k`, 881 windows

| feature set | role | LOEO ADE@2s | paired vs shipped | sep | `rank_acc` | `frac_2x` | C-leak |
|---|---|---|---|---|---|---|---|
| shipped (`logits`) | incumbent | **0.4728** | — | — | 0.3292 | 0.4109 | — |
| **A-shipped** | ⚠️ degenerate control | 0.4728 | +0.0000 | no | 0.3292 | 0.4109 | 0.0000 |
| **A-refined** | ⚠️ degenerate control | 1.3100 | +0.8372 [+0.6915,+0.9939] | yes | 0.0681 | 0.8695 | 0.0000 |
| **B-both** | **S1's marginal** | 0.5667 | **+0.0940** [+0.0507,+0.1410] | **yes ADVERSE** | 0.2849 | 0.4960 | −0.0010 |
| **C-lon** | the longitudinal lever | 0.6864 | **+0.2137** [+0.1459,+0.2858] | **yes ADVERSE** | 0.2304 | 0.5539 | −0.0010 |
| **D-lon+scores** | both | 0.5124 | **+0.0396** [+0.0045,+0.0720] | **yes ADVERSE** | 0.2894 | 0.4699 | −0.0020 |
| **E-cv** | ⚠️ zero-param control | 0.8149 | +0.3421 | yes | 0.2860 | 0.5096 | 0.0000 |
| **A-emitted** (S1b) | ⚠️ degenerate control | 2.3024 | +1.8296 | yes | 0.0352 | 0.9285 | 0.0000 |
| **B-both+emitted** | | 0.6574 | +0.1846 | yes ADVERSE | 0.2599 | 0.5437 | −0.0005 |
| **D-all** | | 0.5128 | +0.0400 | yes ADVERSE | 0.2894 | 0.4711 | −0.0032 |

### 3.1b The registered panel — `refc-xl-30k`, 881 windows: **every row replicates**

| feature set | LOEO ADE@2s | paired vs shipped | sep | `rank_acc` | C-leak |
|---|---|---|---|---|---|
| shipped (`logits`) | **0.4714** | — | — | 0.3110 | — |
| A-shipped ⚠️ control | 0.4714 | +0.0000 | no | 0.3110 | 0.0000 |
| A-refined ⚠️ control | 1.3861 | +0.9147 [+0.7755,+1.0633] | yes | 0.0647 | 0.0000 |
| **B-both** | 0.6404 | **+0.1689** [+0.1068,+0.2443] | **yes ADVERSE** | 0.2247 | −0.0130 |
| **C-lon** | 0.9582 | **+0.4868** [+0.3844,+0.5925] | **yes ADVERSE** | 0.1635 | −0.0017 |
| **D-lon+scores** | 0.5135 | **+0.0421** [+0.0014,+0.0812] | **yes ADVERSE** | 0.2611 | −0.0020 |
| E-cv ⚠️ control | 0.8158 | +0.3443 | yes | 0.2633 | 0.0000 |
| A-emitted ⚠️ control | 1.8437 | +1.3722 | yes | 0.0681 | 0.0000 |
| B-both+emitted | 0.6389 | +0.1675 | yes ADVERSE | 0.2281 | −0.0192 |
| D-all | 0.5141 | +0.0426 | yes ADVERSE | 0.2622 | −0.0038 |

**Controls, XL:** C-monotone **0.471439 == 0.471439** and **1.386093 == 1.386093** (exact) ·
C-permuted-target **3.0119** vs the survivor floor **2.9599**, paired **+0.0520 [−0.0610,+0.1750]
NOT separated** ✅ no leak · C-identity dev 3.87e-05 · C-oracle-floor dev 5.06e-05 · C-cv **0.8158**.

⇒ **`S1_IS_DEAD_AFTER_ALL` fires on BOTH arms.** Two scales, 128 and 256 anchors, same direction,
same controls.

⭐ **Every fitted ranker is separated WORSE than the incumbent — including ones whose feature set
CONTAINS the incumbent's own score.** And `C-leak` is **−0.001 to −0.003 m**: in-sample beats LOEO
by 1–3 mm, so **this is not overfitting.** A 2–6-parameter linear model on 860 training windows
cannot overfit, and the measurement says it did not.

### 3.2 THE CONTROLS — two of the five that could fire, did

| control | result | reading |
|---|---|---|
| **C-monotone** | A-shipped **0.472772 == 0.472772**, A-refined **1.310019 == 1.310019** | ✅ the LOEO plumbing, the reachability restriction and the argmax are all exact. A single-feature fit is a monotone transform and reproduces its incumbent **bit-for-bit** |
| **C-permuted-target** | 2.6898 vs the survivor floor 2.7816, paired **−0.0918 [−0.2527,+0.0718] NOT separated** | ✅ **no leak.** ⚠️ **This control FAILED on its first run** and the failure was mine — see §3.5 |
| **C-leak** | −0.001 to −0.003 m | ✅ fires if the effect were fitting noise; it is not |
| **C-cv** | 0.8149 / 0.8158 | ⭐ **FIRED, and reproduces S3_DEPLOYABLE's argmax rows to 4 dp** from an independent fit — a zero-parameter deployable score still beats every fitted ranker except the incumbent |
| **C-reproduce / C-oracle-floor** | dev 2.78e-05 / 2.13e-05 | ✅ 0.4728 and 0.1914 reproduced |
| ~~C-shuffled~~ | — | ⛔ declared vacuous in code, reported, never load-bearing |

### 3.3 S1b — the emitted-fan readout, and its own control in the SAME forward

| | base | XL |
|---|---|---|
| controls: fan / gt / eid / `prefinal` == E-SEL's `refined_logits` | **bit-identical, all** | **bit-identical, all** |
| `prefinal` = conf(X₁, t=2) — what S1 ranks on today | **1.3100** | **1.3861** |
| `emitted` = conf(X₂, t=2) — S1b at the default token | **2.3024** | **1.8437** |
| paired `emitted − prefinal` | **+0.9924** [+0.7848,+1.1888] ✅ sep **ADVERSE** | **+0.4576** [+0.2289,+0.6852] ✅ sep **ADVERSE** |
| argmax agreement / score correlation | 0.1691 / 0.9038 | — / 0.7302 |

⇒ **`S1b-ADVERSE`, both arms.** My registered prediction was `S1b-HYGIENE`; **I was wrong, and in
the direction I explicitly enumerated rather than one the table could not express.**

### 3.4 ⭐⭐ POST-HOC — THE DENOISE-DEPTH DOSE-RESPONSE, AND WHERE THE DEFICIT ACTUALLY LIVES

*(Added after the registered result. It moved no threshold and adjudicated no branch. It exists
because `S1b-ADVERSE` made "why does scoring what you emit make it worse?" the obvious next
question, and the answer was one config integer away.)*

`loss_cls` supervises `conf_head` on exactly one distribution: **the raw anchor vocabulary at
timestep token t=0.** So the token is part of the distribution — and it turns out to be most of it.

**`raw/s1_dose_response_refc-{base,xl}-30k.json`** — selection ADE@2s over the reachable survivors,
881 windows:

| readout | what it scores | supervised? | **base** ADE / `rank_acc` (×chance) | **XL** ADE / `rank_acc` (×chance) |
|---|---|---|---|---|
| `logits` | conf(**anchors**, t=0) | ✅ by `loss_cls` | **0.4728** / 0.3292 (42.1×) | **0.4714** / 0.3110 (79.6×) |
| `prefinal` | conf(X₁, **t=2**) | ❌ | 1.3100 / 0.0681 (8.7×) | 1.3861 / 0.0647 (16.6×) |
| `emitted` | conf(X₂, **t=2**) | ❌ | 2.3024 / 0.0352 (4.5×) | 1.8437 / 0.0681 (17.4×) |
| ⭐ **`emitted_t0`** | conf(X₂, **t=0**) | trajectory ❌, **token ✅** | **0.6253** / 0.1975 (**25.3×**) | **0.5194** / 0.2463 (**63.0×**) |

| paired contrast | base | XL |
|---|---|---|
| ⭐ **`emitted_t0 − emitted`** | **−1.6771** [−1.9430,−1.4091] ✅ sep | **−1.3243** [−1.6301,−1.0263] ✅ sep |
| **`emitted_t0 − prefinal`** (vs what S1 ranks on today) | **−0.6847** [−0.8317,−0.5369] ✅ sep | **−0.8667** [−1.0177,−0.7259] ✅ sep |
| **`emitted_t0 − logits`** (the hole left to climb) | **+0.1525** [+0.0819,+0.2378] | **+0.0480** [+0.0288,+0.0683] |

⇒ **The refined readout's collapse is dominated by an UNSUPERVISED TIMESTEP TOKEN, not by the
denoised trajectory.** Both arms, monotone in depth, and the recovery is **91.7 %** (base) / **96.5 %**
(XL) of the whole `emitted → logits` gap for **zero parameters and zero extra compute**.

⭐ **What it changes: the size of the job.** S1's CE currently has to climb out of **+0.8372 m**
(base) / **+0.9147 m** (XL). Ranking on `conf(X₂, t=0)` instead makes that **+0.1525 m** /
**+0.0480 m** — **5.5× / 19× smaller**, and on XL an *unsupervised* readout of the emitted fan lands
within **4.8 cm** of the shipped selector. **That is a materially different experiment from the one
D-SEL registered**, and it is the thing to pre-register next.

⚠️ **It is still not a win.** `emitted_t0` is separated **worse** than shipped on both arms, and on
every four-family row (§5). It changes the *prior* on the retrain, not the *verdict*.

### 3.5 ⚠️ A CONTROL FAILED FIRST, AND THE FAILURE WAS MINE

`C-permuted-target` first reported **14.9108** against a floor of **2.7816** — apparently a
catastrophic control failure. It was not a leak: I had permuted the **reachability mask** along with
the features, so the argmax landed outside the true survivor set and the honest comparator became
the **full-fan** uniform floor (14.5426), not the survivor-restricted one. 14.91 vs 14.54 is the
control **passing** against the wrong yardstick. Fixed by permuting features only; the comparator
now reads 2.6898 vs 2.7816, **not separated**. ⇒ **a control judged against the wrong floor is not a
control**, and the fix is written next to the code so it cannot recur.

---

## 4. ⭐⭐ THE FINDING WITH THE LONGEST REACH — IT IS THE OBJECTIVE, NOT THE INFORMATION

*(POST-HOC. Decides no registered branch.)*

Every registered fit lost — **including feature sets containing the incumbent score**, and **without
overfitting**. That leaves two readings which license opposite decisions: *no information*, or
*wrong objective*. The registered objective is a **listwise softmax CE against the oracle index** —
**exactly what `refc_train.loss_rcls` optimises**. So the discriminating test is to hold the
features, folds and survivors fixed and swap the objective for the quantity actually wanted:
**expected ADE under the score's own softmax**.

| feature set | ADE under **CE** (registered) | ADE under **soft-ADE** | paired `softADE − CE` | paired `softADE − shipped` |
|---|---|---|---|---|
| A-shipped (degenerate) | 0.4728 | 0.4728 | +0.0000 | +0.0000 |
| **B-both** | 0.5667 | **0.4694** | **−0.0974** [−0.1417,−0.0573] ✅ **sep BETTER** | **−0.0034** [−0.0122,+0.0047] ⛔ **not separated** |
| **D-lon+scores** | 0.5124 | **0.4751** | **−0.0373** [−0.0597,−0.0148] ✅ **sep BETTER** | +0.0023 [−0.0201,+0.0231] ⛔ not sep |
| **C-lon** | 0.6864 | 0.6872 | +0.0008 [−0.0050,+0.0076] ⛔ not sep | +0.2145 ✅ sep worse |

**And on XL, same direction, larger:**

| feature set | ADE under **CE** | ADE under **soft-ADE** | paired `softADE − CE` | paired `softADE − shipped` |
|---|---|---|---|---|
| **B-both** | 0.6404 | **0.4733** | **−0.1670** [−0.2390,−0.1070] ✅ **sep BETTER** | +0.0019 [−0.0057,+0.0096] ⛔ not sep |
| **D-lon+scores** | 0.5135 | **0.4675** | **−0.0461** [−0.0690,−0.0239] ✅ **sep BETTER** | **−0.0040** [−0.0300,+0.0216] ⛔ not sep |
| **C-lon** | 0.9582 | 0.9612 | +0.0030 [−0.0080,+0.0151] ⛔ not sep | +0.4898 ✅ sep worse |

⇒ **Swapping the objective recovers essentially the entire deficit** (−0.0974 against a +0.0940
deficit) and moves the fitted ranker from *separated worse* to *statistically indistinguishable*
from the incumbent. **The information was there; the objective was throwing it away.**

⚠️ **And it still does not beat the incumbent** (−0.0034, not separated). So:
* the registered verdict stands — **there is no free target**;
* but the *reason* the registered fits lost is now MEASURED, and it is a property of the objective
  D-SEL's S1, S3, S5 and S6 **all** depend on;
* `C-lon` is unmoved by the objective ⇒ purely longitudinal candidate geometry genuinely carries no
  selection signal on this fan. That row is a clean null, not an artefact.

### 4.1 ⭐ AND THE RECOVERY IS LONGITUDINAL — the family that actually matters

`raw/s1_objective_families_refc-base-30k.json`, paired `softADE − CE`, same windows, per family:

| family | metric | **B-both** | sep | **D-lon+scores** | sep |
|---|---|---|---|---|---|
| **LONGITUDINAL** | `speed_abs_err_mps` | **−0.1102** [−0.1584,−0.0672] | ✅ **better** | **−0.0357** [−0.0587,−0.0119] | ✅ better |
| | `along_abs_err_m` | **−0.1004** [−0.1465,−0.0587] | ✅ **better** | **−0.0351** [−0.0573,−0.0125] | ✅ better |
| | `speed_signed_err_mps` | +0.2376 | ✅ (bias back up) | +0.0885 | ✅ |
| | `along_signed_err_m` | +0.2249 | ✅ | +0.0831 | ✅ |
| **LATERAL** | `cross_abs_err_m` | −0.0038 | no | −0.0092 | ✅ better |
| | `heading_abs_err_deg` | −0.0191 | no | −0.0659 | ✅ better |
| | `curvature_abs_err_1pm` | −0.0010 | ✅ better | −0.0005 | ✅ better |
| | `yaw_rate_abs_err_degps` | −0.0765 | ✅ better | −0.1163 | ✅ better |

**XL, same table, larger and unambiguous:** `speed_abs` **−0.1816** [−0.2599,−0.1159] ✅ better ·
`along_abs` **−0.1691** [−0.2429,−0.1066] ✅ better · `cross_abs` −0.0086 · `heading` −0.0510 ·
`curvature` −0.0017 · `yaw_rate` −0.1420 (all four lateral separated better, but **an order of
magnitude smaller than the longitudinal pair**).

⇒ **The objective swap repairs the LONGITUDINAL axis specifically** — `speed_abs` and `along_abs`
move ~0.10 m/(m/s) while every lateral metric moves ≤ 0.08 and two of four are not separated at all.
**87.6–89.9 % of the selection gap is longitudinal**, so this is the axis a ranker has to move, and
it is the one the registered objective was damaging. ⚠️ Note the mirror-image `signed` rows: the CE
fit was buying a *bias* reduction at the cost of *absolute* error, and the soft-ADE fit gives the
bias back to recover the absolute. That trade is invisible in ADE and is exactly what the
four-family rule exists to expose.

⛔ **What this does NOT license:** *"replace `loss_rcls` with a soft-ADE loss and S1 will work."*
This is a linear model over 2–6 features at frozen weights. It is a warning with a measured
magnitude, and the correct response is a **pre-registered** arm, not a patch.

---

## 5. THE FOUR METRIC FAMILIES — per family, never pooled (binding)

Grid **derived**, never assumed: `wp_steps [5,10,15,20] × 0.1 s → dt = 0.5 s`
(`four_families.infer_dt`). A hard-coded 0.1 s inflates speed ×5 and accel ×25 (R-2026-08-03-c).
⛔ An ADE horizon sweep is one row of five.

### 5.1 The registered arms, paired vs shipped (base, 881 windows)

| family | metric | **B-both** | sep | **C-lon** | sep | **S1b-emitted** | sep |
|---|---|---|---|---|---|---|---|
| **LONGITUDINAL** | `speed_abs_err_mps` | **+0.1073** | ✅ worse | **+0.0564** | ✅ worse | **+2.0094** | ✅ worse |
| | `speed_signed_err_mps` | −0.2678 | ✅ (bias down) | −0.0644 | ✅ | −2.1858 | ✅ |
| | `along_abs_err_m` | **+0.0963** | ✅ worse | **+0.0455** | ✅ worse | **+1.8523** | ✅ worse |
| | `along_signed_err_m` | −0.2535 | ✅ | −0.0747 | ✅ | −1.9916 | ✅ |
| **LATERAL** ⛔ guard-rail | `cross_abs_err_m` | +0.0043 | no | **+0.2306** | ✅ **worse** | **+0.0776** | ✅ worse |
| | `heading_abs_err_deg` | +0.0188 | no | **+1.9919** | ✅ **worse** | **+1.3967** | ✅ worse |
| | `curvature_abs_err_1pm` | **+0.0010** | ✅ **worse** | **+0.0069** | ✅ **worse** | **+0.0209** | ✅ worse |
| | `yaw_rate_abs_err_degps` | **+0.0820** | ✅ **worse** | **+2.4304** | ✅ **worse** | **+3.4227** | ✅ worse |

⇒ **B-both fails the LATERAL guard-rail as well as ADE** (curvature and yaw-rate separated worse).
⇒ **The `speed_signed` / `along_signed` improvements are a BIAS trade, not a win** — every fitted
ranker reduces the signed over-prediction while making the *absolute* error worse. Reporting
`speed_bias` alone would have read as a longitudinal gain; that is exactly why the rule requires the
signed **and** absolute rows together.

### 5.2 The CEILING (oracle-in-fan − shipped) — reproduces E-SEL §5.1 to 4 dp

`speed_abs` **−0.2688** [−0.3436,−0.1987] ✅ · `along_abs` **−0.2781** [−0.3515,−0.2090] ✅ ·
`cross_abs` **−0.0334** ✅ (small) · `heading` −0.1334 no · `curvature` **+0.0013** no (wrong sign) ·
`yaw_rate` **+0.1212** no (wrong sign).

⇒ **The LATERAL guard-rail is tight by construction and that is measured:** a *perfect* reranker of
this fan buys **0.033 m** of cross-track and moves heading, curvature and yaw-rate **not at all**.
Any separated lateral regression is therefore a real fault — and three arms have one.

### 5.3 Per-family status, with the reason and the n where not computable

| family | status | n |
|---|---|---|
| **ADE** | ✅ MEASURED, all arms | 881 |
| **LONGITUDINAL** — speed / along-track | ✅ MEASURED, all arms, signed **and** absolute | 881 |
| **LONGITUDINAL** — distance-keeping (headway / time-gap / TTC) | ⛔ **NOT COMPUTED. Reason: `taniteval/lead_source.py` is absent on the only reachable host, and the repo mount was in a whole-mount READ-FAILURE state for this entire run (§9).** Thor's val epcache carries `frames_u8/actions/poses/maneuvers` only — **no obstacle tracks**. ⚠️ This is a **RUN, not a work item**, the moment either is reachable: coverage is ~270 of these exact 881 windows. **The brief asked for this family specifically and it is the one thing I could not deliver.** | **0** |
| **LATERAL** | ✅ MEASURED, all four metrics, all arms | 881 |
| **TACTICAL** — goal/anchor selection | ✅ MEASURED (`rank_acc`, `sel_gap`, `frac_sel_2x_worse`) — the half D-SEL exists to move | 881 |
| **TACTICAL** — manoeuvre decision + confusion | ⛔ UNAVAILABLE: a fan bank stores no decoded manoeuvre logits | **0** |
| **STRATEGIC** | ⛔ UNAVAILABLE: no route/goal label in a fan bank, and the decode used `nav_mode='follow_constant'` so the route input was never exercised (the C6 confound, inherited deliberately — changing it would move the baseline every contrast is paired against) | **0** |

---

## 6. WAS I RIGHT? — the registered prediction, scored

| I predicted (pinned in advance) | outcome |
|---|---|
| **B-both: `S1-NOT-SEPARATED`** | ❌ **`ADVERSE`** — separated, and worse. Right that there is no target; **wrong about the direction.** |
| **C-lon: `LON-LEVER-NULL`** (~55 % confidence) | ❌ **`ADVERSE`** — separated worse, on ADE *and* on both families. |
| **S1b: `S1b-HYGIENE`** | ❌ **`S1b-ADVERSE`** — +0.9924 m (base) / +0.4576 m (XL), separated. |
| **Overall: §4.4 fires, S1 reported DEAD** | ✅ **correct** |

**So: I got the conclusion right and the direction of every component wrong.** The thing that saved
the adjudication is that the pre-registration had an `ADVERSE` row for each — the escalation E-SEL
filed and S3_DEPLOYABLE reproduced. **The lesson generalises: I under-predicted how often a lever is
actively harmful rather than merely useless.** Three for three here.

**And the thing I did not predict at all:** that the deficit is mostly a **timestep token** (§3.4)
and that the registered *objective* is itself a measured liability (§4). Both came out of following
an adverse result rather than filing it.

---

## 7. 🔴 ESCALATIONS

1. **→ PI / arch-inf — the S1 arm should be re-pre-registered around the TOKEN, and it is cheap.**
   The `dsel-s1only` arm as §7 of the parent prereg defines it ranks on `conf(X₁, t=2)` — MEASURED
   the second-worst readout the decoder can produce. The same arm with
   `--sel-score-emitted --sel-score-emitted-t 0` starts **5.5× / 19× closer** to the incumbent at
   **identical parameter count and identical compute**. ⛔ I did not launch anything; a GPU-day is
   the PI's call. **Recommended arm:**
   `--sel-refined --sel-score-emitted --sel-score-emitted-t 0 --sel-reach-clamp --sel-ce-reach --labels v21`.
2. **→ arch-inf / D-SEL — `loss_rcls` is a MEASURED liability, not just an unexamined choice** (§4).
   Swapping objective recovers −0.0974 m of a +0.0940 m deficit on frozen weights. **S1, S3, S5 and
   S6 all depend on that CE.** It needs its own pre-registered arm before any of them is funded.
3. **→ eval-tools / ops — `taniteval/lead_source.py` exists in ONE place (the repo) and the repo was
   unreachable for this entire run.** Every LONGITUDINAL distance-keeping number in the programme
   currently depends on a single Google-Drive-backed checkout. Thor holds no copy. ⇒ **the
   distance-keeping instrument should be mirrored to Thor**, the same way the val cache already is.
4. **→ ops — the repo mount failure is DIAGNOSED, not mysterious** (§9). `GoogleDriveFS`'s Dokan
   layer logs `GetInfo request with new, unhandled info class: 77` in a loop and then returns
   `ERROR_INVALID_FUNCTION` for **every** read while directory listing keeps working. It is a
   client-side driver fault, not a network outage; the recovery is a Google Drive restart. **I did
   not restart it** — other streams may be mid-write on the same mount, and that is the PI's call.

---

## 8. WHAT I DID NOT DO — plainly

* ⛔ **Did not train anything, launch any arm, or touch `tanitad-new` / `tanitad-pod4`.** Cost was
  ~2 min of GPU on an idle Thor (0 % before and after) plus ~15 min of its CPU.
* ⛔ **Did not test the SUPERVISED climb-out.** Registered in §2 of the prereg **before** measuring:
  a monotone re-weighting of a banked scalar cannot test it. Bounding it needs the refined pass's
  query features (a GPU dump, 0 training); measuring it needs a retrain.
* ⛔ **Did not compute LONGITUDINAL distance-keeping** — the one family the brief singled out.
  Reported per family with the reason and n = 0 (§5.3). Not a pass.
* ⛔ **Did not compute the manoeuvre-confusion or STRATEGIC families** — UNAVAILABLE with reason and
  n = 0.
* ⛔ **Did not run `pytest -q` on the full suite.** The repo mount was unreadable and Thor's venvs
  carry no pytest (the two-venv rule). The 12 new tests were executed on Thor via a minimal shim
  (`raw/run_tests_noPytest.py`) — **12 passed, 0 failed** — but **the authoritative full-suite count
  is UNVERIFIED for this change** and must be run in the repo before anything is committed.
* ⛔ **Did not commit and did not push.** ⚠️ **And could not `git add`** — see §9.
* ⚠️ **Did edit Thor's `~/TanitAD` checkout** (`refc.py`, `refc_train.py`, plus new files under
  `stack/scripts/` and `stack/tests/`). Originals backed up to `thor:~/_s1_backup/*.bak`; every edit
  is reproduced by the idempotent `s1_patch_refc.py`, which is the single source of truth for the
  diff and is banked in the manifest.
* ⚠️ **Did not re-open E-SEL-0 or S3**, and did not run `refc-small-30k` (Thor holds no small
  checkpoint — unchanged since E-SEL).

---

## 9. ⛔ THE DELIVERY BLOCKER — the repo mount, and why nothing is staged

**The Google Drive mount holding the repo entered a whole-mount READ-FAILURE state mid-edit and
stayed there for the entire session.** Directory listing works; **every file read fails**, including
`CLAUDE.md` and `.git`, so `git` itself reports *"not a git repository"*.

* MEASURED: `[Errno 22] Invalid argument` (POSIX) / *"Unzulässige Funktion"* = `ERROR_INVALID_FUNCTION`
  (Win32) on every read, across four path shapes (`G:\…`, `\\?\G:\…`, Python, .NET).
* MEASURED, and this is the diagnosis: `%LOCALAPPDATA%\Google\DriveFS\Logs\drive_fs.txt` ends in a
  loop of `file_info_handler.cc:182 GetInfo request with new, unhandled info class: 77` and then
  stops. **A Dokan driver fault, not a network outage.**
* `G:` does not appear in `Get-Volume` (it is a Dokan virtual mount), and `GoogleDriveFS` is alive.

⇒ **Two edits to `refc.py` landed before the mount died and their current on-disk state is
UNVERIFIED.** Everything else is **reproducible by one command** —
`python s1_patch_refc.py <refc.py> <refc_train.py>`, idempotent, refusing to apply against an
unexpected anchor. **Nothing was lost; nothing is only in my context.** But the
AGENT_OPERATING_STANDARD's *"stage, never push"* could not be honoured, and that is the single
outstanding item.

**RECOVERY, in order, once the mount reads again:**
1. `git status --short -- stack/tanitad/refs/refc.py stack/scripts/refc_train.py` — if the two
   partial edits are present, `git checkout` both files to HEAD first, so the patcher applies to a
   known state.
2. `python <scratchpad>/s1_patch_refc.py "stack/tanitad/refs/refc.py" "stack/scripts/refc_train.py"`
3. copy the five new files from the manifest into the repo, `cd stack && pytest -q`, then
   `git add` **and verify with `git ls-files --cached <path>`** (an `add` exit code is not evidence).

---

## 10. DELIVERABLE MANIFEST

⚠️ **Everything below lives in TWO places — the dev-box scratchpad bundle
`…/scratchpad/S1_CLIMBOUT_BUNDLE/` and `thor:~/s1_climbout/` — and in NEITHER is it in the repo.**
That is the §9 blocker, not a choice. ⭐ **Nothing is single-disk**: every result JSON and every
augmented bank was pulled back off Thor. `RECOVER_INTO_REPO.sh` in the bundle puts the whole package
into the repo and stages it in one command, and verifies the staging with `git ls-files --cached`
rather than trusting an `add` exit code.

| artifact | scratchpad (dev box) | Thor | repo |
|---|---|---|---|
| **the source patch (single source of truth for the diff)** | `s1_patch_refc.py` | `~/s1_climbout/s1_patch_refc.py` | ⛔ **NOT STAGED** → `stack/scripts/` is not its home; the diff belongs applied to `stack/tanitad/refs/refc.py` + `stack/scripts/refc_train.py` |
| **pre-registration** (blob `8dc68daf…`) | `PREREG_S1_CLIMBOUT.md` | `~/s1_climbout/PREREG_S1_CLIMBOUT.md` | ⛔ → `…/incoming/2026-08-03-s1-climbout/` |
| **tests (new, 12, all passing)** | `test_refc_select_s1_climbout.py` | `~/TanitAD/stack/tests/` | ⛔ → `stack/tests/` |
| **the 0-GPU probe** | `refc_s1_climbout_probe.py` | `~/TanitAD/stack/scripts/` | ⛔ → `stack/scripts/` |
| **the GPU dump (S1b + the t=0 readout)** | `refc_s1_dump_emitted.py` | `~/s1_climbout/` | ⛔ → `stack/scripts/` |
| **the dose-response probe** | `refc_s1_dose_response.py` | `~/TanitAD/stack/scripts/` | ⛔ → `stack/scripts/` |
| **the objective four-family probe** | `refc_s1_objective_families.py` | `~/TanitAD/stack/scripts/` | ⛔ → `stack/scripts/` |
| **identity/capacity gate** | — | `~/s1_climbout/verify_s1_identity.py`, `raw/verify_s1_identity.json` | ⛔ |
| **prereg content pin** | — | `~/s1_climbout/raw/prereg_pin.json` | ⛔ |
| **results, 881 windows, 2 arms** | — | `~/s1_climbout/raw/s1_climbout_probe_refc-{base,xl}-30k.json` | ⛔ |
| **dose-response results, 2 arms** | — | `~/s1_climbout/raw/s1_dose_response_refc-{base,xl}-30k.json` | ⛔ |
| **objective-families results, 2 arms** | — | `~/s1_climbout/raw/s1_objective_families_refc-{base,xl}-30k.json` | ⛔ |
| **augmented banks (`emitted` + `emitted_t0` + `prefinal`)** | — | `~/s1_climbout/raw/fan_emitted{,_t0}_refc-{base,xl}-30k.pt` | ⛔ |
| **test transcript (shim runner)** | `run_tests_noPytest.py` | `~/s1_climbout/` | ⛔ |
| Thor originals (pre-edit) | — | `~/_s1_backup/*.bak` | ⚠️ **THOR ONLY** — restorable, not program work |

---

## 11. EVIDENCE CLASS

| claim | class |
|---|---|
| both flags cost **exactly 0** parameters; `refc_config()` = 104,191,577; D-SEL preset still +385 | **MEASURED** — `raw/verify_s1_identity.json`, `param_breakdown` on a meta build, pinned by test |
| all-off byte-identity (keys **and** values) and forward bit-identity across 6 flag combinations | **MEASURED** — same file, diffed against the pre-edit file loaded from `~/_s1_backup/refc_pre.py` |
| `refined` scores `x_in` while `anchor_traj = x_in + off` | **MEASURED (source)** — `refc.py` `_decode` + the denoise loop |
| B-both **+0.0940** [+0.0507,+0.1410]; C-lon **+0.2137**; D-lon+scores **+0.0396**, all separated ADVERSE (base) — and **+0.1689 / +0.4868 / +0.0421**, all separated ADVERSE (XL) | **MEASURED** — `raw/s1_climbout_probe_refc-{base,xl}-30k.json` |
| S1b `emitted − prefinal` **+0.9924** (base) / **+0.4576** (XL), separated | **MEASURED** — same + `refc-xl` |
| ⭐ `emitted_t0 − emitted` **−1.6771** / **−1.3243**, separated; residual hole **+0.1525** / **+0.0480** | **MEASURED** — `raw/s1_dose_response_refc-{base,xl}-30k.json`. ⚠️ **POST-HOC** |
| ⭐ soft-ADE recovers **−0.0974** [−0.1417,−0.0573] (base) / **−0.1670** [−0.2390,−0.1070] (XL) of B-both's deficit; still **not separated** vs shipped on either arm | **MEASURED** — `objective_diagnostic` block, both arms. ⚠️ **POST-HOC** |
| ⭐ that recovery is LONGITUDINAL: `speed_abs` **−0.1102** / **−0.1816**, `along_abs` **−0.1004** / **−0.1691** (base / XL), all separated better, with every lateral metric an order of magnitude smaller | **MEASURED** — `raw/s1_objective_families_refc-{base,xl}-30k.json`. ⚠️ **POST-HOC** |
| C-monotone exact; C-permuted-target not separated from the survivor floor; C-cv reproduces S3_DEPLOYABLE's 0.8149 | **MEASURED** — `controls` block |
| the CEILING four-family rows reproduce E-SEL §5.1 to 4 dp | **MEASURED** — `families._ceiling_oracle_minus_shipped` |
| E-SEL-0's +0.8372 / +0.9187 and the 8.7× / 16.6× chance figures | **INHERITED** — `ESEL_VERDICT.md`; the +0.8372 row is **re-derived here** and reproduces exactly |
| *"87.6–89.9 % of the selection gap is longitudinal"* | **INHERITED** — `ESEL_VERDICT.md` §5.1; its component rows ARE reproduced here (§5.2) |
| *"the oracle gap is ~92 % irreducible"* | **INHERITED** — a prose note in `MODEL_REGISTRY.md` §4.1, not a results JSON. Load-bearing and flagged as such by the parent prereg |
| *"a supervised climb-out will/won't work"* | **NOT MEASURED** — out of this experiment's reach by construction, registered as such in advance |
| full-suite `pytest -q` count for this change | ⛔ **UNVERIFIED** — the repo was unreadable and Thor has no pytest. 12/12 new tests pass under a shim. |
