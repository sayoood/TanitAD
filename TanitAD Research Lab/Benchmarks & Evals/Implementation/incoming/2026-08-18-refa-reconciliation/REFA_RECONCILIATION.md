# REF-A vs C104 — THE TENSION IS AN ARTEFACT OF THE COMPARISON, AND "READABLE ≠ USABLE" IS NOW MEASURED

**Date** 2026-08-18 · **Branch** `agent/arch-inf-20260803` · **Zero GPU** — every number below comes from
banked per-window dumps and banked label blocks already in the repo.
**Estimator** paired episode-cluster bootstrap over the 40 val episodes (`taniteval/ci.py`), 2 000 draws,
seed 0. ⛔ Never `overlapping_holdout_se`.
**Tier** ⛔ **EVERYTHING HERE IS T0.** See §1.1 — the brief's premise that driving ADE is T1 is **wrong**,
and that correction changes which resolution applies.

---

## 0. THE ANSWER, IN FOUR SENTENCES

1. **The two numbers were never comparable, and the "91× vs 5.1×" framing is inadmissible as stated** —
   not on tier grounds (both are T0), but because the two comparisons **do not share an arm, do not share
   a quantity, and do not share a trainable-parameter budget**. Three independent non-comparabilities, each
   established from a primary source in §2.
2. ⭐ **But the underlying question is real and I answered it rather than dissolving it.** I built and ran
   the driving-side counterpart of C104's own `lead_gap` rung — **E-RECON-1**, pre-registered with both
   outcomes committed in the instrument's docstring before the run. **Result: the pre-registered NULL
   (O2).** REF-A's paired deficit vs the flagship is **+1.7150 m on windows with a lead vehicle** and
   **+1.7295 m on windows without one** — contrast **−0.0146 [−0.5988, +0.5551], not separated**, and the
   longitudinal members' point estimates go the **wrong way for the "readout helps" hypothesis**.
3. ⭐⭐ **And a new, decision-grade LONGITUDINAL result fell out of it, on the exact quantity C104
   measures.** On distance-keeping the flagship is **statistically indistinguishable from ground truth**
   on all three metrics, while **both frozen-DINOv2 arms are CI-separated from ground truth on all
   three, in the unsafe direction** (REF-A min-TTC **−5.82 s [−9.34, −2.06]** vs human; flagship
   **−0.16 s [−1.10, +0.71]**). ⚠️ Read with the confound in §4.3 — distance-keeping is a function of
   the same predicted path as ADE, so it is a **consequence**, not an independent test.
4. ⇒ **"Readable ≠ usable" stops being candidate resolution #3 and becomes a MEASUREMENT.** C104 remains
   true as a statement about linear decodability of a frozen representation. It is **not evidence about
   driving**, and §7 registers the one experiment that could still make it so.

⚠️ **What this does NOT do.** It does not show the encoder is fine, and it does not show C104 is wrong.
It shows the encoder gap is **unattributed**, because §3 found REF-A and the flagship differ in **at
least four things, not the two the registry claims** — and nobody has isolated any of them.

---

## 1. WHAT THE BRIEF GOT WRONG, ESTABLISHED FIRST BECAUSE IT REDIRECTS EVERYTHING ELSE

### 1.1 ⛔ The REF-A driving number is **T0**, not T1 — so the tier resolution FAILS in the direction stated

The brief's candidate #1: *"A linear readout is T0 … Driving ADE is T1. `EVAL_DOCTRINE` forbids comparing
across tiers."* **The second half is false, MEASURED from source.**

`taniteval/taniteval/rollout.py:144-153`, the docstring of the function that produced every window in
`driving_*.json`:

> **PC2: THIS SURFACE IS NOT A HIERARCHY SURFACE, AND NOW SAYS SO.** `rollout_decode` takes no
> `intent`/`ctx`/`nav` and **is fed the expert's true future actions**, so the number it produces is a
> **world-model fidelity** decode of a known control sequence.

and the mechanism, `rollout.py:177-182`:

```python
fa = torch.stack([ep.actions[t + window:t + window + fwd_k] for t in ch]).to(device)
...
wp_full, _ = rollout_decode(model.predictor, states, aw, fa, step_readout, fwd_k)
```

`fa` is the **recorded future action sequence**. The dict carries `actions_source="expert_future"`.
⇒ By `EVAL_DOCTRINE.md`'s own definition — *"T0 | teacher-forced: predictor consumes recorded future
actions"* — **2.1675 and 0.4271 are T0 numbers.**

**Consequence, and it cuts against the easy answer:** since C104 is also T0, **the doctrine does not
forbid this comparison on tier grounds.** The tier resolution is *not* available. The comparison fails
for three other reasons (§2), which is a stronger and more useful finding than "wrong tier".

### 1.2 ⚠️ A NAME COLLISION that made §1.1 hard to see — `tier0` means two different things

`driving_refa-dinov2.json#block` reads `taniteval.driving/tier0`. That `tier0` is the **METRIC-SUITE**
tier (sparse 4 waypoints vs the dense 20-step path), stated in the same file's `surface` field:

> *"4 waypoints 0.5 s apart (tier-0). The dense 20-step path is … discarded at rollout.py:94 → jerk,
> comfort bounds, curvature PROFILE and decel-onset lead time are **tier-1** (suite E2)."*

It is **not** `EVAL_DOCTRINE`'s T0. The two vocabularies collide on the same string, and here they
happen to agree — which is worse than disagreeing, because it hides the collision. Same class as the
`E-ENC` collision that `PREREG_ENCODER_EXPERIMENTS.md §0` had to fix.
⇒ **Escalated (§8): the doctrine tier must be a distinct field, not inferred from a suite-tier string.**

### 1.3 ⛔ The §6 rank table — where 2.1675 and 0.4271 sit side by side — carries **NO tier stamp at all**

`MODEL_REGISTRY.md:2772-2795`. The binding rule (`CLAUDE.md`) says *"A registry row or report quoting a
number without its tier stamp is incomplete."* The single most-quoted table in the programme is
incomplete by its own standard, and that is exactly why the brief inherited "REF-A ADE is T1".

---

## 2. THE THREE NON-COMPARABILITIES — why "91× vs 5.1×" cannot be set against each other

### 2.1 They do not share an arm — C104's "ours" is **v6**, REF-A's opponent is **v1**

| | C104 (registry §12.1) | REF-A comparison (registry §2.1, §6) |
|---|---|---|
| the "ours" side | **`v6F-SW-30k` snapshots** — `MODEL_REGISTRY.md:3615`, verbatim: *"**Substrate:** frozen `v6F-SW-30k` snapshots"* | **`flagship4b-speedjerk-30k`** (flagship v1), `MODEL_REGISTRY.md:2789` |
| params quoted | *"our encoder's **87.3 M**"* (`:3627`) | encoder + readout **87,121,280** — MEASURED, constructed on CPU, bit-exact against `MODEL_REGISTRY.md:175` |
| line | v6 successor line | v1, trained 2026-07 |

**These are different models from different architecture generations.** The 91× belongs to a v6-vs-DINOv2
readout; the 5.1× belongs to a v1-vs-REF-A driving comparison. ⚠️ **The `87.3 M` figure has no
`param_report` I could locate and does not match the v1 encoder's MEASURED 87.12 M — INHERITED, flagged.**
The load-bearing fact is not the param count but the substrate line, which is quoted verbatim above.

### 2.2 They do not share a quantity — and **two of C104's three rungs have no driving counterpart at all**

| C104 rung | what the driving eval does with it |
|---|---|
| **`ego_v0`** (0.717 vs 0.052, 13.7×) | ⛔ **SUPPLIED AS AN INPUT TO BOTH ARMS.** D-A3 made `v0` the 3rd action channel; `rollout.py:80` appends `poses[last, 3:4] / SPEED_SCALE` under `speed_input`, and **both** `refa-4brain-speed-30k` and the flagship carry `--speed-input` (registry `:1925`, `:1910`). **An encoder's ability to read `v0` from pixels is irrelevant to an arm that is handed `v0`.** |
| **`lead_gap`** (0.450 vs 0.005, **91×**) | 🟥 **THE DRIVING EVAL REFUSES THE ENTIRE FAMILY.** `driving.py:608`: *"no lead-agent state exists (`lead_state` is a None stub)"*, surfaced in every `driving_*.json` as `refused.headway_ttc_distance_keeping`. |
| **`lead_closing`** (0.017 vs 0.000) | same refusal |

⇒ **The headline 91× is measured on a quantity the driving benchmark explicitly declines to score, and
the 13.7× on a quantity the driving arms are given for free.** That is the core of the artefact. §4
removes the first half of it by building the missing family.

### 2.3 They do not share a trainable budget — MEASURED, and it is **116.9 M params**

Constructed on CPU (torch 2.11), replicating each trainer's own build path:

| | flagship `flagship4b-speedjerk-30k` | REF-A `refa-4brain-speed-30k` |
|---|---:|---:|
| **TOTAL TRAINABLE** | **277,404,073** | **160,514,460** (**57.9 %**) |
| frozen params in graph | 0 | 0 (DINOv2's 86,580,480 is **not in the graph at all**) |
| total model | 263,442,838 — bit-exact vs `MODEL_REGISTRY.md:175` | 154,462,611 |

Evidence class **MEASURED** for every row except the flagship's `aux_accel` head (528,897,
ESTIMATED-FROM-REGISTRY-ARITHMETIC — its source is uncommitted per `MODEL_REGISTRY.md:178`).
⇒ **A frozen arm losing to a fully-trained one with 1.73× the trainable capacity is, on its face, as
much a capacity result as an encoder result.** Nothing published isolates the two.

---

## 3. ⛔ THE REGISTRY'S "EXACTLY TWO DIFFERENCES" IS WRONG — THERE ARE AT LEAST FOUR

`MODEL_REGISTRY.md:1897-1899`: *"the flagship and REF-A differ in **exactly two things**: (1) the encoder,
(2) the SIGReg target (`pred_only` vs `full_relaxed`)."*

MEASURED from source, on the trainer that **actually produced the arm**
(`stack/experiments/reset-speed4b/refa_train_plus.py`, per registry `:1926`):

| # | difference | size | evidence |
|---|---|---|---|
| 1 | **encoder** | trained ViT 87.12 M vs frozen DINOv2 86.58 M + a **0.197 M** adapter | MEASURED, CPU-constructed |
| 2 | **SIGReg target** — *and it is two changes, not one* | `full_relaxed` adds a term on the encoded **states** **and** routes both terms through `position_relaxed`, so `--sigreg-free-dims 64` **has no effect at all on the REF-A path** | `stack/tanitad/train/flagship_losses.py:366-376` |
| 3 | 🟥 **h15 imagination — ABSENT FROM REF-A ENTIRELY** | **22,055,683 trainable params** the flagship has and `RefAModel` does not | MEASURED; `refa.py:179-239` has no such module; added flagship-side at `train_flagship4b.py:657-658` |
| 4 | 🟥 **grounding depth** | flagship `HierarchicalGrounding` op/tac/str **13,432,338**; as-trained REF-A a flat single pair **4,477,446** | MEASURED; `metric_dynamics.py:320`, `refa_train_plus.py:440` |

⛔ **And the test that is supposed to pin the two-difference claim does not cover this arm.**
`stack/tests/test_refa_flagship_parity.py:150-155` compares `train_flagship4b` against **`refa_train4b`** —
not `refa_train_plus.py`. Its `SHARED_BRAINS` tuple (`:53-54`) omits `imagination`, so difference #3 is
structurally invisible to it. ⚠️ **The registry's mechanism narrative (`:1899`, and `config.py:466`)
describes a code path the published arm never ran.**

### 3.1 ⭐ Is the SIGReg confound isolable? **YES — and it is NOT inert under a frozen encoder**

This was the brief's starred sub-question. **Answer: refuted by measurement, in the direction that makes
the confound real.** A gradient probe (CPU, `RefAModelPlus`, temporal adapter, `full_relaxed` state term
alone) gives:

```
pred_only                  loss=0.56829   |grad| adapter=1.068545e+02   predictor=7.388654e+02
full_relaxed_state_term    loss=0.74078   |grad| adapter=8.930440e+01   predictor=0.000000e+00
```

The state term deposits a **large gradient directly on the trainable adapter and zero on the predictor** —
an independent training pressure, not a re-derivation of `pred_only`. Mechanism: in REF-A
`states = adapter(standardizer(feats))` (`refa.py:241-247`), so SIGReg acts on **the adapter's output
distribution** instead of the flagship's ViT output. `refa.py:15` records the choice — *"SigReg only on
PREDICTOR outputs"* — as a design decision, not a structural necessity.

**Cost to isolate:** ⛔ **not a flag on any trainer.** `train_flagship4b.py:71` and `refa_train4b.py:73`
hard-code `SIGREG_VARIANT` as module constants; `refa_train_plus.py` never imports `flagship_loss` at all
and hard-codes `loss_sig = model.sigreg(z_pred_all)` at `:169`, with **no sigreg argument anywhere** in its
30-flag argparse block. ⇒ an **~8-line source edit** to `compute_losses_plus`, then a training run. §7
prices it.

---

## 4. CLOSING THE ADE-ONLY DEFECT — the four families, paired, for the first time

⛔ The H4 verdict (D-A5) is argued **entirely on ADE**, which the binding rule (Sayed, 2026-08-02) makes an
**INCOMPLETE result**. The data to fix it was banked and needed zero GPU.

**Both gates passed before any number was emitted** (`code/refa_vs_flagship_families.py`):
* **ALIGNMENT** — `eid`, `gt`, `cv`, `speed`, `head_deg`, `wp_steps` all **bit-identical** across the three
  arms' dumps ⇒ the paired estimator is legal. 881 windows / 40 episodes.
* **REPRODUCTION** — all 48 banked headline means (3 arms × 16 metrics) reproduced from the dumps,
  **max abs diff 4.9e-05**. ⛔ Exit codes are not evidence; this is.
* ⭐ **Independent cross-check:** the instrument's `refa-dynin-30k − flagship` ADE delta comes out
  **+2.6200 [+2.0945, +3.2570]**, reproducing the registry's own §6 published paired delta **exactly**.

### 4.1 REF-A DINOv2 4B **minus** flagship v1 — **[TIER T0]**, paired episode-cluster bootstrap, n = 881 / 40

| family | metric | REF-A | flagship | Δ (REF-A − flag) | CI95 | separated |
|---|---|---:|---:|---:|---|:--:|
| TRAJECTORY | `ade_0_2s` | 2.1675 | 0.4271 | **+1.7404** | [+1.4870, +1.9870] | ✅ |
| TRAJECTORY | `fde_2s` | 3.2803 | 0.9075 | +2.3728 | [+1.9494, +2.7961] | ✅ |
| TRAJECTORY | `miss_2m` | 0.6129 | 0.0454 | +0.5675 | [+0.5023, +0.6349] | ✅ |
| **LONGITUDINAL** | `long_abs_2s_m` | 3.0993 | 0.8412 | **+2.2581** | [+1.8455, +2.6648] | ✅ |
| **LONGITUDINAL** | `speed_mae_mps` | 1.7754 | 0.4710 | **+1.3044** | [+1.1196, +1.4901] | ✅ |
| **LONGITUDINAL** | `progress_abs_err_m` | 3.1009 | 0.8370 | +2.2638 | [+1.8516, +2.6775] | ✅ |
| **LATERAL** | `lat_abs_2s_m` | 0.5776 | 0.2369 | +0.3407 | [+0.2032, +0.4930] | ✅ |
| **LATERAL** | `heading_mae_2s_deg` | 5.0346 | 6.6062 | **−1.5716** | [−7.1424, +2.5419] | ⛔ **no** |
| **LATERAL** | `heading_med_2s_deg` | 1.2317 | 1.2742 | **−0.0425** | [−0.3661, +0.5455] | ⛔ **no** |
| **LATERAL** | `heading_exceed_5deg` | 0.2043 | 0.0874 | +0.1169 | [+0.0509, +0.1818] | ✅ |
| **LATERAL** | `pathgeom_crosstrack_m` | 0.2450 | 0.1110 | +0.1341 | [+0.0848, +0.1845] | ✅ |
| **LATERAL** | `curv_sign_agree` ↑ | 0.8664 | 0.9535 | −0.0870 | [−0.1192, −0.0572] | ✅ |

*Signed diagnostics (`long_signed`, `speed_bias`, `progress_signed`, `lat_signed`) are reported as two
point values and **deliberately not paired** — closer-to-zero is not smaller, and a difference of means
does not answer it. Same rule that excludes biases from `driving.PAIRED`.*

⭐ **The deficit is not uniform across families, and ADE alone hid that.** It is **overwhelmingly
LONGITUDINAL** (speed MAE ×3.8, along-track ×3.7), and on **two lateral metrics — heading MAE and the
heading MEDIAN, which registry R5 names the honest reducer for a heavy-tailed quantity — REF-A and the
flagship are statistically indistinguishable.** On heading MAE REF-A is *nominally better*.
⇒ *"REF-A cannot drive"* is too coarse. **REF-A cannot regulate SPEED; its path shape is not separated
from the flagship's on the honest heading reducer.**

### 4.2 REF-A dyn-in (the H4 final answer) minus flagship — same construction

Every longitudinal and trajectory row separated and larger (ADE +2.6200, speed MAE +1.9081
[+1.5076, +2.3878], along-track +3.6950 [+2.8901, +4.6652]); **heading MAE +0.1171 [−3.0188, +2.6311]
and heading median +0.1983 [−0.2277, +0.7930] again NOT separated.** Full table in
`raw/refa_vs_flagship_families.json`.

### 4.3 ⭐ LONGITUDINAL **distance-keeping** — the family that was "refused", now built

⛔ **`driving.py:608`'s *"no lead-agent state exists"* is a STALE BLOCKER.** `taniteval/taniteval/lead_source.py`
(the `obstacle.offline` → `win["lead"]` wiring) landed, and a **val40 lead block** was built and
row-verified against these exact dumps. I re-verified alignment rather than inheriting it:
**881 block rows = 881 dump rows, episode partition identical, speed correlation 1.0** (a 0.0018 m/s max
difference is expected — the block recomputes `v0` on the registered clip clock, realised spacing
~0.1007 s). **LEAD 270 / NO_LEAD 551 / NO_LABEL 60.**

**[TIER T0] paired episode-cluster bootstrap, n_used ≈ 218–240 windows in 19 episode clusters:**

| arm | headway_min (m) vs GT | time_gap_min (s) vs GT | min_TTC (s) vs GT |
|---|---|---|---|
| **flagship v1** | **−0.0801** [−0.2412, +0.1497] ⛔ **not sep** | **−0.0102** [−0.0356, +0.0209] ⛔ **not sep** | **−0.1566** [−1.1020, +0.7065] ⛔ **not sep** |
| **REF-A DINOv2** | **−1.4180** [−2.3983, −0.3815] ✅ sep | **−0.3152** [−0.5797, −0.0881] ✅ sep | **−5.8223** [−9.3414, −2.0561] ✅ sep |
| **REF-A dyn-in** | **−1.9295** [−3.4288, −0.5172] ✅ sep | **−0.4959** [−0.9334, −0.1241] ✅ sep | **−5.5187** [−9.5892, −2.1061] ✅ sep |

direct: **REF-A − flagship** headway **−0.9819** [−1.9782, −0.0594] ✅ · time-gap **−0.2944**
[−0.5603, −0.0616] ✅ · min-TTC **−3.9094** [−6.7124, −1.4463] ✅ — all separated, all in the unsafe
direction. REF-A's paths **close on the lead in 136 of 247 windows against the human's 103**.

⚠️ **THREE CAVEATS THAT TRAVEL WITH THIS TABLE.**
1. ⛔ **This is NOT an independent test of encoder usability.** Distance-keeping is computed from the
   same predicted path as ADE, so an arm with 5× lower ADE has lead-relative quantities closer to GT
   *by construction*. Read it as **the longitudinal deficit propagating into a safety-shaped quantity**,
   which is what makes it worth reporting — not as a second, confirming experiment. **The properly
   controlled statistic is §5.**
2. **TTC is dt-dependent.** These use the banked waypoint spacing **dt = 0.5 s**; TTC scales as 1/dt
   through the closing rate (`lead_metrics.py:134`). **Not comparable to a dense-path TTC at dt = 0.1 s.**
   Headway and time-gap are dt-invariant and are comparable.
3. **Censoring:** 111/247 (REF-A) and 113/228 (flagship) windows never close and are censored at
   `TTC_CAP_S = 30 s`. `n_closing` is quoted beside every mean, never the mean alone.

### 4.4 TACTICAL and STRATEGIC — **UNAVAILABLE, with reason and n**, as clause 5 requires

| family | status | reason | n | what closes it |
|---|---|---|---|---|
| **TACTICAL** | 🟥 UNAVAILABLE | the tier-0 dumps carry waypoints only; **no decoded manoeuvre / tactical goal / anchor was persisted for EITHER arm**, so a confusion matrix cannot be built for either side | **0** | a decode-traversing pass over the same 881 windows for both arms — **GPU, not free** |
| **STRATEGIC** | 🟥 UNAVAILABLE | same, **plus** PhysicalAI-AV carries no map/lane-graph/route signal, **and** the flagship's route head is an exact bijection of the nav it is fed (369/369, 81/81, score 1.0000 — *cited from `CLAUDE.md`'s binding goal-input rule; **INHERITED**, not re-verified here*) — a persisted strategic decision would be scoring an **echo** | **0** | a route/goal signal that is not the ego's own future path — **blocked on corpus, not compute** |

⇒ **The H4 verdict rests on two of four families.** It is not overturned by that — the longitudinal
evidence is overwhelming — but it must stop being presented as complete.

---

## 5. ⭐ E-RECON-1 — THE DISCRIMINATING EXPERIMENT, PRE-REGISTERED AND EXECUTED

**The question no banked number could answer:** does DINOv2's 91× `lead_gap` readability show up in the
driving of the arm DINOv2 actually powers?

**Design.** The difference of paired differences, resampling episodes **inside one joint draw** (⛔ not two
intervals differenced — that is the quadrature error in a new costume):

```
contrast = mean_LEAD(REF-A − flagship) − mean_NO_LEAD(REF-A − flagship)
```

**Both outcomes were committed in the instrument's docstring before the run**
(`code/refa_lead_rung.py`, §"PRE-REGISTRATION"): **O1** contrast < 0 separated ⇒ the readout advantage
*is* usable; **O2** contrast ≈ 0 ⇒ **readable ≠ usable, demonstrated on the rung itself**; **O3** contrast
> 0 separated ⇒ worse where the scene is interactive. Registered in advance: *no outcome can prove C104
wrong* — it decides **relevance**, not correctness.

### 5.1 Result — **O2**, and the longitudinal point estimates go the WRONG WAY for O1

| stratum | REF-A ADE | flagship ADE | paired Δ | CI95 |
|---|---:|---:|---:|---|
| **LEAD** (n = 270, 21 ep) | 2.0587 | 0.3437 | **+1.7150** | [+1.2273, +2.1929] |
| **NO_LEAD** (n = 551, 33 ep) | 2.1874 | 0.4579 | **+1.7295** | [+1.4313, +2.0274] |

| family | metric | contrast (LEAD − NO_LEAD) | CI95 | separated | direction |
|---|---|---:|---|:--:|---|
| TRAJECTORY | `ade_0_2s` | **−0.0146** | [−0.5988, +0.5551] | ⛔ no | — |
| **LONGITUDINAL** | `long_abs_2s_m` | **+0.0989** | [−0.8768, +0.9802] | ⛔ no | ⚠️ **wrong way for O1** |
| **LONGITUDINAL** | `speed_mae_mps` | **+0.0914** | [−0.2996, +0.4583] | ⛔ no | ⚠️ **wrong way for O1** |
| **LONGITUDINAL** | `progress_abs_err_m` | +0.0954 | [−0.8808, +0.9772] | ⛔ no | ⚠️ wrong way |
| LATERAL | `lat_abs_2s_m` | −0.1664 | [−0.3662, +0.0349] | ⛔ no | p(Δ>0) = 0.050 |
| LATERAL | `pathgeom_crosstrack_m` | −0.0657 | [−0.1426, +0.0146] | ⛔ no | p = 0.051 |
| LATERAL | `heading_mae_2s_deg` | −11.4835 | [−25.8568, +0.4733] | ⛔ no | p = 0.032 |

⇒ **On the LONGITUDINAL family — the family C104's `lead_gap` and `lead_closing` rungs belong to — the
presence of a lead vehicle does not shrink REF-A's deficit by any detectable amount, and the point
estimates move slightly the wrong way.**

### 5.2 The registered confound control — it is **not** a speed artefact

LEAD windows are slower and more urban, so "there is a lead" could stand in for "the ego is slow". The
contrast was therefore also computed **within speed bands** (`< 5`, `5–12`, `≥ 12` m/s). **Not separated
in any band**, on either ADE or speed MAE (`lo` +0.1481 / +0.4573 · `mid` −0.0826 / −0.0857 · `hi`
−0.3265 / −0.1063). Strata below 30 windows are marked UNPOWERED, never quoted.

### 5.3 ⚠️ POWER — registered in advance, and stated as a bound, never as "no effect"

The LEAD arm has **21 episode clusters**, and the bootstrap resamples episodes. The honest reading is a
**bound**, computed against REF-A's own total deficit:

| metric | total deficit | most-negative CI bound | **excluded: a lead-presence benefit larger than** |
|---|---:|---:|---|
| `ade_0_2s` | 1.7404 | −0.5988 | **34.4 % of the deficit** |
| `speed_mae_mps` | 1.3044 | −0.2996 | **23.0 % of the deficit** |
| `long_abs_2s_m` | 2.2581 | −0.8768 | **38.8 % of the deficit** |

⇒ **A benefit smaller than ~23–39 % of the deficit is NOT excluded.** What is excluded is the large
effect the 91× ratio would lead one to expect. ⛔ **This is "not separated at n = 21 clusters", never
"no effect exists".**

### 5.4 ⚠️ The lateral rows are consistently negative and I am **not** claiming them

Three lateral contrasts sit at p(Δ>0) = 0.032–0.051 — all just outside separation — and for the **dyn-in**
arm two of them *are* separated (`lat_abs` −0.2928 [−0.5852, −0.0135]; `pathgeom_crosstrack` −0.1017
[−0.2022, −0.0013]). That is a real pattern: **the frozen arm's LATERAL deficit shrinks where a lead is
present.** But (a) it is the *lateral* family, not the family C104's rungs belong to; (b) seven metrics
were tested per arm with no multiplicity control; (c) it is equally explained by LEAD windows being
lateral-easier. ⇒ **Recorded as an observation for a future pre-registration, not as a finding.**

---

## 6. THE BRIEF'S SIX CANDIDATE RESOLUTIONS — adjudicated

| # | candidate | verdict |
|---|---|---|
| 1 | **Tier mismatch** | ⛔ **FAILS AS STATED.** Both sides are **T0** (§1.1). The comparison is inadmissible for other reasons — and the brief's own premise needed correcting, which is the more useful result. |
| 2 | **Trainable capacity** | ✅ **CONFIRMED AND QUANTIFIED — 160,514,460 vs 277,404,073 (57.9 %), a 116.9 M gap.** MEASURED, CPU-constructed, bit-exact against the registry's own flagship figures. **Unisolated.** |
| 3 | **Readable ≠ usable** | ⭐⭐ **PROMOTED FROM HYPOTHESIS TO MEASUREMENT.** §5: the pre-registered null, on C104's own rung, with a stated power bound. This is the answer. |
| 4 | **The SIGReg confound** | ✅ **ISOLABLE — and NOT inert under a frozen encoder** (MEASURED gradient probe, §3.1: the state term puts \|grad\| 8.93e+01 on the adapter, 0 on the predictor). ⛔ Not a flag on any trainer — an ~8-line source edit plus a run. |
| 5 | **Estimator hygiene** | ✅ **AUDITED — and the registry's own flag is partly false**, see §8.2. |
| 6 | **I-JEPA leak / `CONTAMINATED`** | ✅ **RESOLVED, and they are UNRELATED**, see §6.1. |

### 6.1 What `.CONTAMINATED-<ts>` meant — **not a leak, and nothing to do with ADE**

MEASURED from the code that writes the suffix, `taniteval/taniteval/efficiency.py:1730-1741`:

```python
dirty = [p for p in precisions if not out.get(p, {}).get("contamination_check", {}).get("valid")]
if dirty:
    dest = res_dir / f"eff_{key}.CONTAMINATED-{_t.strftime('%Y%m%d-%H%M%S')}.json"
    out["QUARANTINED"] = f"GPU was NOT exclusive during {dirty}; ..."
```

It is a **GPU-exclusivity quarantine on the inference-LATENCY panel**. `valid` comes from `_gpu_state()`
(`efficiency.py:134-145`), sampled before and after every timed block.
`eff_refa-dinov2.CONTAMINATED-20260720-215641.json` records the co-tenant explicitly:
`other_compute_detail = ["1270125, 17156 MiB"]`, `util_pct 98.0`.

* ⛔ **It is NOT the I-JEPA val leak** (R8/§2.2) — a different defect on a different axis. The quarantined
  files contain **no val split, no episode list, no ADE re-measurement**; top-level keys are
  `QUARANTINED / ckpt_step / fp32 / key / model / protocol`.
* **11 unique quarantined files** (22 including the pod-rescue mirror). ⚠️ `WAVE1_E_REPORT.md:127` says
  14 — that is a report, not primary, and 14 could not be reproduced.
* **The deployed flagship was NOT hit** — `eff_flagship-30k.json` is `valid` on all three precisions; the
  quarantined arm is `flagship-nospeed`.
* ⚠️ **The bias is not monotone and it flipped a verdict.** For `refa-dinov2` the contaminated run is
  *faster* and reports `meets_10hz_p99 = true` where the clean run reports **false**;
  `eff_flagship-nospeed.CONTAMINATED-*` runs *slower*. **These files are wrong in both directions** — the
  same shape as the `overlapping_holdout_se` bidirectional bias.
* **No registry or retraction class owns it** (zero hits for `CONTAMINATED`/`QUARANTIN` in
  `MODEL_REGISTRY.md`) — because the code **prevented publication**. A caught error, not a retracted claim.
  ⚠️ Not to be confused with `RETRACTION_LOG.md:624` *"A CONTAMINATED ANCHOR"*, which is a genuine
  train/val leak class — **same word, unrelated mechanism**.

### 6.2 REF-A I-JEPA — worse than "leaked": **it never entered the driving block at all**

Probed three paths. There is **no `refa-ijepa.json` and no `driving_refa-ijepa.json`** anywhere; only
`diag_`, `imag_`, `plan_`, `eff_` artifacts exist, all with `block = None`, `estimator = null`. Its
published *"fwd-ADE 3.194 vs DINOv2 3.796 @15 k"* are **training-side** numbers, not a TanitEval val
statistic. ⇒ The arm has **neither a heldout nor a decision-grade ADE**. "Unusable because leaked"
understates it: **there is no canonical-val number to be unusable.**

---

## 7. ⭐ WHAT WOULD SETTLE WHAT REMAINS — **E-RECON-2**, pre-registered

**What is now settled:** the two measurements are not in tension (§2), and the readout advantage is not
detectably usable on its own rung (§5).
**What is NOT settled, and is now the sharpest open question in H4:** REF-A's deficit is **unattributed**
across **four** differences (§3), and the registry attributes it to the encoder by default.

⛔ **Not a duplicate of any live sibling stream.** Checked against both in the working tree:

| sibling | what it answers | why E-RECON-2 is not it |
|---|---|---|
| `…/Research/2026-08-18-encoder-localisation/` (the stage-by-stage readout ladder: raw DINOv2 → adapter → predictor latent) | **WHERE in REF-A's stack the information is lost** | it cannot say **whether restoring it would improve driving**, and it cannot separate the encoder from the other three differences |
| `…/Research/2026-08-18-pretrained-encoder-arm/` (**E-XENC-1** — fusing an external pretrained encoder **into v6**) | **does adding DINOv2 help the v6 line** | opposite direction: it adds a foreign encoder to a trained stack; E-RECON-2 **removes trainability** from a known-good one |

E-RECON-2 takes the localisation stream's framing as given and is designed for the gap both leave open:
**attribution of REF-A's measured deficit across the four differences of §3.**

### E-RECON-2 — **THE FROZEN-FLAGSHIP ARM** (priority 1, the only cell that separates encoder from capacity)

**Arm.** Take `flagship4b-speedjerk-30k`'s **own trained encoder**, **freeze it**, and retrain
adapter + full 4-brain from scratch on the canonical 2 376-episode corpus (parity key `e438721ae894`,
skip-hash `f09e44db`), **with h15 and 3-level grounding present** — i.e. REF-A's *trainability* with the
flagship's *representation* and the flagship's *architecture*.

**Why it is the discriminator.** It holds the representation at a known-good one and varies only
frozen-vs-trainable. It is the only single arm that separates candidate 2 from candidate 3.

**Both outcomes committed in advance:**

| outcome | reading | what it does to the programme |
|---|---|---|
| **P1** — lands near REF-A (ADE@2s within the REF-A CI, paired Δ vs flagship separated and > +1.0 m) | **the deficit is TRAINABILITY, not the encoder** | D-A5's *"frozen-encoder ceiling"* is really a *frozen-anything* ceiling; C104's encoder gap is **not** the lever, and the v6 encoder work loses this as a motivation |
| **P2** — lands near the flagship (paired Δ vs flagship not separated, or < +0.5 m) | **the deficit IS the representation** | C104's readout gap becomes the leading explanation, and E-XENC-1 gains a strong prior |
| **P3** — lands between, separated from both | partial attribution | report the **fraction** of the 1.7404 m deficit each side explains; no single-cause claim |

**Cost.** One training run. ⭐ **Run it to 5 k steps first, not 30 k** — the REF-A milestone ladder is
banked at 5 k with a decision-grade value (`driving_overfit_refa-dynin-5k.json#headline.ade_0_2s` =
3.8307 [3.2216, 4.4916]), so a 5 k cell is **immediately comparable, paired, on the same 881 windows**,
at ~1/6 the cost. Escalate to 30 k only if 5 k lands in P3.

**Riders on the same run family, in priority order** (so a killed stream still yields value):
* **R-A (cheapest, ~8 lines):** REF-A with `full_relaxed` SIGReg — isolates registry difference #2,
  the one the registry names and nobody has tested. The `position_relaxed` routing means this also
  restores `--sigreg-free-dims`, so **state both sub-changes** or it is a two-lever cell.
* **R-B:** REF-A **+ h15 imagination + 3-level grounding** — isolates differences #3 and #4 together
  (+35.5 M trainable). ⚠️ Two levers in one cell; split only if R-B moves the number.

**Registered in advance for all cells:** four metric families per family, never pooled; paired
episode-cluster bootstrap on the same 881 windows; **T-tier stamped**; and the LEAD/NO_LEAD contrast of §5
re-run, since a cell that closes the deficit *without* closing the contrast would mean something different
from one that closes both.

⚠️ **Registered limitation:** every E-RECON-2 cell is **T0**. It cannot make a driving-capability claim.
A T1 read requires `taniteval/tools/t1_eval.py` on the resulting checkpoints — costed separately, and it
is where the deficit could behave differently, since §1.12 measured open-loop lateral skill to be an
**action echo** that collapses 97.9 % → ~5 % closed-loop.

---

## 8. ⛔ REGISTRY CORRECTIONS REQUIRED — escalated, not written into a README

These are **not** for me to apply to `MODEL_REGISTRY.md` (agents do not edit it unilaterally); each is
stated with its primary source so the owner can act.

### 8.1 Structural

| # | site | correction |
|---|---|---|
| C1 | `:1897-1899` | *"differ in **exactly two things**"* → **at least four** (§3). And the mechanism narrative describes `refa_train4b`'s code path, which **the published arm never ran** (`:1926` names `refa_train_plus.py`). |
| C2 | `test_refa_flagship_parity.py:150-155`, `:53-54` | the parity pin tests the **wrong trainer** and its `SHARED_BRAINS` omits `imagination` ⇒ **difference #3 is invisible to the test that exists to catch it.** |
| C3 | `:2772-2795` (§6 rank table) | **no T-tier stamp**, while the binding rule requires one on every row. It is **T0** (§1.1). This is what propagated "REF-A ADE is T1" into a brief. |
| C4 | `driving.py:608`, and every `driving_*.json#refused` | *"no lead-agent state exists"* is a **STALE BLOCKER** — `lead_source.py` + `val40_lead_block.npz` exist and attach row-for-row (§4.3). Same class as the `obstacle.offline` stale-absence sweep. |
| C5 | §2.1, §12.1 | **§2.1 has no Params row at all**, and §12.1's *"our encoder's 87.3 M"* has no `param_report` behind it and does not match the v1 encoder's MEASURED 87,121,280. |

### 8.2 Estimator hygiene — the audit, and where the registry contradicts itself

| site | published | class | decision-grade form |
|---|---|---|---|
| `:1927` ADE@2s full-set 2.1675 | ✅ | **DECISION-GRADE** | `driving_refa-dinov2.json#headline.ade_0_2s` |
| 🔴 `:1927` **FDE 3.2619** | **BANNED split-mean, UNLABELLED** | — | exists: `#headline.fde_2s` = **3.2803** [2.8524, 3.7084] |
| 🔴 `:1927` **miss 0.6245** | **BANNED split-mean, UNLABELLED** | — | exists: `#headline.miss_2m` = **0.6129** [0.5442, 0.6803] |
| §2.3 milestone curve 3.755→3.694→3.016→2.920 | all **heldout** | — | all four exist as `driving_overfit_refa-dynin-*.json`: 3.8307 / 3.7818 / 3.1138 / 3.0471, ecb. ✅ **monotonicity survives — the "not overfitting" verdict holds** |
| §2.3 RMSE 6.21 / 1.54 / 94.2 % | point-only | — | raw says **6.2278 / 1.4677 / 94.74 %** — the registry's digits are wrong |

⚠️ **§2.1 and §6 publish different FDE and miss for the same arm** (3.2619 vs 3.2803; 0.6245 vs 0.6129)
because §2.1 quotes the banned split-mean without saying so.

⛔ **And the registry's own flag at `:1972-1973` is partly FALSE.** It says *"Only the ADE@2s row has both
forms; the other five rows have **no decision-grade value published anywhere**."*

| row | registry claim | reality |
|---|---|---|
| ADE@0.5s / 1s / 1.5s | no DG anywhere | ⚠️ **half-true** — full-set points **1.3159 / 1.9005 / 2.4748** exist in two independent artifacts; the **interval** is genuinely missing |
| **FDE@2s** | no DG anywhere | 🔴 **FALSE** — **4.7642 [3.92, 5.7824]**, ecb — **printed in the same document at `:2791`, 817 lines below** |
| **miss@2m** | no DG anywhere | 🔴 **FALSE** — **0.7412 [0.6822, 0.7946]**, ecb — **also at `:2791`** |

⇒ The named remedy (*"a re-emission from `driving_refa-dynin-30k.json`"*) is correct for the three ADE
horizons and **unnecessary** for FDE/miss. **A stale absence-claim living inside an estimator-hygiene
flag** — the same class the `CLAUDE.md` absence rule exists for.

### 8.3 Integration escalations

1. ⭐ **The four-family panel of §4 should replace the ADE-only argument in D-A5 and §2.1/§2.3.** The
   verdict does not change; its completeness does.
2. ⭐ **`taniteval.driving` should consume `lead_source` and stop emitting the stale refusal** — the
   wiring exists, the block exists, and §4.3 shows it works. That is a code change in `taniteval/`, owned
   by the eval stream.
3. **The doctrine tier needs its own field** in the driving block, distinct from the suite-tier string
   (§1.2).

---

## 9. DELIVERABLE MANIFEST

| artifact | where it lives | notes |
|---|---|---|
| `REFA_RECONCILIATION.md` (this) | `repo:TanitAD Research Hub/Benchmarks & Eval/Implementation/incoming/2026-08-18-refa-reconciliation/` | — |
| `code/refa_vs_flagship_families.py` | same, `code/` | the paired four-family instrument; alignment + reproduction gates |
| `code/refa_lead_rung.py` | same, `code/` | E-RECON-1; **pre-registration in the docstring, written before the run** |
| `raw/refa_vs_flagship_families.json` | same, `raw/` | §4.1–4.2, plus lead/speed strata |
| `raw/reproduction_gate_rows.json` | same, `raw/` | all 48 rows, max abs diff 4.9e-05 |
| `raw/refa_lead_rung.json` | same, `raw/` | §4.3, §5 |

**Nothing lives in only one place** — all six are in the repo working tree and staged. **No pod, no
worktree, no GPU was used.** Inputs consumed (all pre-existing, unmodified):
`taniteval/results/windows_{refa-dinov2,flagship-30k,refa-dynin-30k}.pt` ·
`taniteval/results/driving_*.json` ·
`…/2026-08-04-distance-keeping-arms/raw/val40_lead_block.npz`.

**No file under `stack/` or `taniteval/` was modified**, so the suite's state is unchanged by this package.

---

## 10. EVIDENCE-CLASS LEDGER

| claim | class |
|---|---|
| every number in §4 and §5 | **MEASURED (ours)** — artifact paths in §9, both gates passed |
| trainable-param counts, the four differences, the SIGReg gradient probe | **MEASURED (ours)** — constructed on CPU, bit-exact against `MODEL_REGISTRY.md:175` |
| `.CONTAMINATED` semantics | **MEASURED** — from `efficiency.py:1730-1741` and the JSON's own `QUARANTINED` field |
| teacher-forcing of the driving block | **MEASURED** — `rollout.py:144-153`, `:177-182` |
| C104's rung values, substrate, and the 91× | **PUBLISHED (cited)** — `MODEL_REGISTRY.md:3615-3628` |
| *"our encoder's 87.3 M"* | ⚠️ **INHERITED** — no `param_report` located; does not match the v1 encoder's MEASURED 87,121,280 |
| flagship `aux_accel` = 528,897 | **ESTIMATED-FROM-REGISTRY-ARITHMETIC** — source uncommitted per `:178` |
| REF-A step-time / wall-clock | 🟥 **UNVERIFIED** — probed four paths, recorded nowhere; E-RECON-2's cost is therefore given in **steps**, not hours |
| §5.4's lateral pattern | **OBSERVATION, not a finding** — no multiplicity control |
| E-RECON-2's outcome map | **HYPOTHESIS**, pre-registered |
