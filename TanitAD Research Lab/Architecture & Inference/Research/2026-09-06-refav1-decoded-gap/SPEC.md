# SPEC — the VISION-ONLY decoded lead-gap head for refav1's distance-keeping cost

**`D-REFAV1-DK-DECODED` · hypothesis `H-REFAV1-DK-DECODED-1`.**
⛔ **PRE-REGISTERED. Written and committed BEFORE any head was fit and before any
number in the PASS/FAIL tables below existed.** The only facts consulted first are
label statistics needed to size the run (the eval block's state census) and the
already-published M84 / `D-REFAV1-DK-COST` results, both of which are cited, not
re-derived.

---

## 0. The question, in one sentence

`D-REFAV1-DK-COST` proved refav1's distance-keeping cost fires (21/90), re-ranks
(21/21) and breaks the constant-velocity degeneracy — **but its `gap0` is an
ORACLE LABEL, so the whole result is a CEILING.** This spec asks: **does the same
cost still work when its gap comes from VISION ALONE?**

---

## 1. Schema

```yaml
hypothesis: H-REFAV1-DK-DECODED-1     # registered in GOALS_AND_CLAIMS.md this turn
one_variable: gap_source              # oracle_label -> decoded_gap -> decoded
held_constant: [ckpt, seed, corpus, window_grid, w_dk, tau_target_s, d0_m,
                lead_speed_closure, cost_metric, cost_weights, planner_config]
controls: [constant_only, raw_pixel_floor, within_clip_shuffle,
           deliberate_regression_unstamped_source, adequate_window_specificity]
splits:
  units: CLIPS (episode-disjoint), never rows
  scheme: 5-fold CROSS-FITTING over the 141 v7.2-EVAL clips, seeded, label-free
          fold assignment (sha256 of clip_id). Every scored prediction is
          OUT-OF-FOLD: the head that produced it never saw that clip's labels.
  tuning: PCA basis AND ridge lambda fit INSIDE the training folds only, lambda by
          clip-grouped 5-fold CV within those folds. The scored fold is scored,
          never tuned on.
tier: none for the decode panel (a representation/probe measurement, nothing is
      rolled). T1 (self-action open loop) for the planner A/B.
```

⚠️ **Why cross-fitting and not one FIT/SCORE split.** A deployed head would be fit
on the TRAIN corpus and applied to EVAL. That corpus's fp8 cache is not on this
box (only `refav1-eval141`), and the named GPUs are busy, so the head is fit
*inside* the eval corpus. Cross-fitting is then the discipline that makes this
admissible: no clip's own labels ever enter its own prediction, and **every** one
of the 141 clips still gets a prediction, so the planner A/B is not restricted to
half the panel. ⛔ It is a MEASUREMENT device, not the deployment story, and the
result must say so.

---

## 2. The arms — exactly one variable moves per step

| arm | lead PRESENT gate | `gap0` value | what it isolates |
|---|---|---|---|
| `w_dk = 0` | — | — | ⛔ the PARITY CONTROL. Must be bit-identical to the pre-2026-09-06 shipped path. |
| `oracle_label` | ORACLE (block `state == LEAD`) | ORACLE (block `gap0_m`) | the published CEILING (`D-REFAV1-DK-COST` §6) |
| `decoded_gap` | ORACLE | **DECODED** | the GAP decode alone |
| `decoded` | **DECODED** | **DECODED** | ⭐ the DEPLOYABLE VISION-ONLY arm |

Everything else is held: same checkpoint (step 21,109), same planner seed, same
window grid (`--window-stride 40`), `w_dk = 1e-5`, `tau = 1.5`, `d0 = 5.0`,
`LEAD_SPEED_STEADY`.

---

## 3. THE BARS — both outcomes committed here, before the data

### BAR-1 — DECODE (necessary, not sufficient)

Scored on the pooled OUT-OF-FOLD rows, clip-cluster bootstrap, `n` and `d` printed.

* ⭐ **PASS** if **all three** hold:
  1. paired **`field − pix` CI EXCLUDES ZERO** on `gap0_m` (LEAD rows), and
  2. the `constant` control reads **exactly +0.000000**, and
  3. the within-clip **shuffle** control's skill is **below** `field`'s, so the
     decode is not clip identity.
* ⛔ **FAIL** otherwise — and a FAIL here ends the arm: a head that does not beat
  raw pixels has added nothing, and the decoded cost must not be run.

### BAR-2 — RE-RANKING REPRODUCTION (the decisive one)

Zero-GPU re-pricing on the **same 282-window panel** and the **same two
constant-acceleration baselines** the oracle used, where both surviving
regularisers are identically zero so **every flip is attributable to the DK term
alone**.

Population: the oracle's **21 violating** windows (fire) and its **69 adequate**
windows (do not fire).

* ⭐ **PASS** if **both**:
  1. **SENSITIVITY ≥ 0.80** — the decoded gap fires *and* flips `a = −1.5` above
     `a = 0` on **≥ 17 of the oracle's 21** violating windows; **and**
  2. **SPECIFICITY ≥ 0.70** — the decoded gap does **not** fire on **≥ 49 of the
     oracle's 69** adequate windows.
     ⛔ Criterion 2 is not optional and is the control that makes criterion 1
     mean anything: a decoder that under-predicts every gap fires everywhere and
     reproduces 21/21 **trivially**.
* ⛔ **FAIL** if either misses ⇒ **decodability was NECESSARY AND NOT SUFFICIENT
  (`C131`)**, stated in exactly those words, with the next lever named.

### BAR-3 — THE PLANNER A/B (structural, exact-equality counts)

On the real step-21,109 checkpoint, same seed, `--dk-gap-source` the only flag
moved.

* ⭐ **PASS** if **both**:
  1. the **constant-velocity fraction moves off 1.0000** on the armed decoded
     arm (an exact-equality count under a fixed seed, not an estimate); **and**
  2. the **lateral family is BIT-IDENTICAL** to the control — same-breath proof
     that the term touched only the longitudinal channel.
* ⛔ **FAIL** if the decoded arm is bit-identical to the `w_dk = 0` control on
  every window ⇒ the decoded gap never crossed the barrier where the planner
  could act on it.

⛔⛔ **THE FLOOR THAT DECIDES WHAT COUNTS AS AN EFFECT.** refav1's planner
SAMPLES; its **inference-seed floor is ≈0.30 m ADE** (`D-REFAV1-SEED-GOAL-MISMATCH`;
the predecessor's own A/B moved ADE **+0.2303 m** and reported it as **NOT an
effect**). ⇒ **No ADE / headway / TTC difference is claimed without ≥ 3
inference-seed replicates, and any difference below that floor is NOT an effect.**
The readable quantities at this `n` are the **exact-equality counts** and the
**const-velocity fraction**, which are identities about a pair of runs, not
estimates.

### BAR-4 — THE PROVENANCE GUARD, PROVEN BY MUTATION

⛔ The cost tool must **REFUSE** — not silently accept — a gap whose source it
cannot name. Committed in advance:

1. an **unrecognised** `gap_source` raises;
2. `gap_source = "unset"` with `w_dk > 0` raises (an armed term must declare its
   source);
3. a decoded arm whose head bundle's **sha256 does not match its stamp** raises;
4. a head bundle fit against a **different checkpoint** than the arm is running
   raises;
5. every dump carries `gap_source`, the head bundle sha256, the head version and
   the checkpoint identity, so a decoded run can never be read as the oracle
   ceiling.

⭐ **Each guard is proven by MUTATION, not by inspection** — the guard is removed
/ the stamp is corrupted and the failure branch is shown to be REACHABLE. *(An AST
census once read 0 suspects on BOTH the fixed and the broken trainer; inspection
is not evidence.)*

---

## 4. ⛔ THE ADMISSIBILITY CHECK — asked here, answered in the result

**BINDING (Sayed 2026-08-03): labels may use ego and GT; inference is VISION
ONLY.** The question that must be answered explicitly, not assumed:

> *Could any input this head sees AT INFERENCE have been computed from something
> the LABEL was derived from?*

The head's only input is `model._last_state(model.encode(feats))` where `feats`
is the cached DINOv3 patch-token window — **camera pixels, nothing else**. The
label is the B1 block's `gap0_m`, derived from **3-D cuboid tracks**
(`obstacle.offline`) which are **not** an input to the trunk at any stage. The
answer must be stated with both halves named in the result, and the second
half — *the ARMING GATE* — must be checked too: the `oracle_label` and
`decoded_gap` arms take their LEAD/NO_LEAD gate from the block and are therefore
**NOT vision-only**; only the `decoded` arm is.

---

## 5. ⛔ WHAT THIS SPEC REFUSES TO DO

* ⛔ **It does not relax the cost tool's source refusal — it EXTENDS the
  vocabulary.** A source the tool does not recognise stays refused (BAR-4).
* ⛔ **It does not let the head smuggle in a CLOSING RATE.** `M84` measured the
  rate does not decode (`field` +0.0061 [−0.0406, +0.0513]; the explicit
  temporal-difference fix recovers +0.0145, still spanning zero). The head
  predicts **gap and presence only**. If a rate were found to decode, that is a
  **new result needing its own pre-registration**, not a quiet upgrade.
* ⛔ **It does not touch `tau` or `d0`.** `tau = 1.5` was chosen by the
  predecessor's leakage-immune EXCESS sweep on 8,339 rows; re-choosing it here
  would move two variables.
* ⛔ **It does not touch the LATERAL channel.** `wk15` is settled (curvature MAE
  0.03098 vs the straight floor's 0.040083).
* ⛔ **It does not use the `|dyaw| > 0.15` gate** — the human fails it 3/9.
* ⛔ **It does not touch the A40 (refcv5 training) or Thor (a sibling's eval).**
  Dev-box RTX 4060 only, and only while it shows no python compute.

---

## 6. Rig

| | |
|---|---|
| box | dev-box RTX 4060 (8.2 GB), `OMP_NUM_THREADS=6` |
| checkpoint | `C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt`, step 21,109 |
| features | `C:/Users/Admin/tanitad-data/refav1-eval141/refav1-fp8-eval/` (141 clips, `float8_e4m3fn [101,640,1024]`) |
| episodes | `C:/Users/Admin/tanitad-data/refav1-eval141/eps/` (141 `.v2ep.pt`) |
| labels | `C:/Users/Admin/tanitad-wt/_lead_b1/b1_eval_lead_block.npz` (29,556 rows / 147 clips) |
| pooling | 16x40 -> 8x20 (20 azimuth bins, 6 deg/bin) — **identical to M84**, so the two panels are comparable |
| freeze | trunk fingerprinted before and after; `max abs delta` must be **0 exactly** |
