# The reachability band is a COMPUTE lever, not a selection lever — swept, not assumed

**Asked:** *"implement c for the better selector without retrain."*
**Answer:** the band does not give one. It gives a **4.37× decode saving at zero selection change**,
which is worth shipping — but it is a different claim and must not be reported as the other one.

**Evidence class: MEASURED** — `taniteval/tools/calibrate_reachability_selector.py`, post-hoc on
already-banked fans. **No retraining, no new decodes**, so it cannot be confounded by a different run.

---

## 1. The sweep (REF-C-XL, 881 canonical-val windows × 256 anchors, ckpt_step 29999)

Baseline, shipped ranking (`logits`): **ADE 0.4714** · oracle-in-fan **0.1640** · **selection gap 0.3075**
— i.e. **65 % of this arm's error is selection loss**, which is the headroom any selector must come out of.

| `accel_max` | survivors | decode saving | **oracle survives** | rows whose selection changed | ADE | paired Δ vs baseline |
|---|---|---|---|---|---|---|
| 0.5 | 16.7 | 15.30× | **0.8502** ⛔ | 54 | 0.4770 | **+0.0055** [−0.0139, +0.0272] not sep |
| 1.0 | 31.5 | 8.14× | 0.9682 ⛔ | 2 | 0.4698 | −0.0017 [−0.0042, 0.0] not sep |
| 1.5 | 45.4 | 5.64× | 0.9977 ⛔ | 0 | 0.4714 | 0.0000 |
| **2.0** | **58.6** | **4.37×** | **1.0000** ✅ | **0** | 0.4714 | 0.0000 |
| 2.5 | 71.5 | 3.58× | 1.0000 ✅ | 0 | 0.4714 | 0.0000 |
| 4.0 | 106.4 | 2.41× | 1.0000 ✅ | 0 | 0.4714 | 0.0000 |
| 6.0 | 145.4 | 1.76× | 1.0000 ✅ | 0 | 0.4714 | 0.0000 |

Estimator: **paired episode-cluster bootstrap** over the fan's own 40 episodes, n_boot 2000.

**Reading.** Above 1.5 m/s² the band prunes only anchors the ranking had already rejected — the
selected index is *literally the same integer* on 881/881 windows, so Δ is 0.0000 **by construction**,
not by measurement. Below it, the band starts deleting the oracle, and at 0.5 m/s² the ADE gets
**worse**. There is no band where reachability improves the shipped ranking.

⭐ **The new number: `accel_max = 2.0` is the tightest safe band.** Oracle survival is still exactly
1.0000 and selection is still exactly unchanged, at **4.37×** instead of the 3.58× at the
standard 2.5. That is a free 22 % compute saving, and it is the first time the boundary was
*located* rather than assumed — E-SEL-0 tested one band.

---

## 2. ⚠️ The trap this sweep walks into, and how to read past it

Run the same sweep on the fan's **`refined_logits`** (S1's discarded last-denoise-pass confidence)
and the band appears to be a large win:

| `accel_max` | ADE | paired Δ | separated |
|---|---|---|---|
| — (baseline) | **1.3901** | — | — |
| 0.5 | 0.7955 | **−0.5946** [−0.7220, −0.4790] | **yes** |
| 1.0 | 1.2539 | −0.1362 [−0.2314, −0.0691] | yes |

⛔ **That is REPAIR, not IMPROVEMENT.** The refined ranker starts at 1.3901 — roughly **3×** worse
than the shipped 0.4714 — and the best the band drags it back to is **0.7955, still far worse than
doing nothing**. A −0.5946 quoted without its baseline reads as the largest selection win in the
programme; it is the opposite.

The tool's verdict string was changed to say this, because the first version would have printed
*"reachability IS a selection lever"* for exactly this arm.

**Generalisation for the estimator rules:** *a separated favourable delta against a WEAK arm is not
evidence about the lever.* Always ask what the best available baseline is, not only what this arm's
own baseline is. Same family as the C6 confound (a decoder compared on its marginal).

---

## 3. Cross-checks that make this admissible

* **Replicates `ESEL_VERDICT.md` exactly** on numbers computed independently here: XL refined
  **1.3901**, shipped **0.4714**, oracle **0.1639**, and shipped − refined = **0.9187** — matching the
  banked paired delta **+0.9187** [+0.7778, +1.0669] to 4 dp.
* **Internal consistency of the two fan files.** `taniteval/results/fan_refc-xl-30k.pt` and the
  esel-verdict `fan_refined_refc-xl-30k.pt` were produced on different hosts (`/root/models` vs
  `thor6`) yet give `logits` ADE **0.4714** and oracle **0.1640 / 0.1639** on the same 881 windows —
  so they are the same windows and the comparison is clean.
* ⚠️ **One known instrument nit.** At `accel_max = 2.5` on the refined ranker this sweep changes
  **1** row of 881 (Δ −0.0040, not separated) where `ESEL_VERDICT` records *exactly* 0.0000. The
  verdicts agree; the difference is a single tie-break and is recorded here rather than smoothed over.

---

## 4. ⛔ Two things that must not be conflated

1. **Stage.** `decode_saving_x` above is measured on the **DECODED fan (S2)**. The published
   pre-decode **anchor** figure (S2b, `be2da04`: **2.78×**, 881/881 identical) is a *different stage*
   with different survivor counts. Neither number may be quoted for the other.
2. **Anchor dependence.** Reachability is a property of **the anchor set and `v0`**, not of the model.
   A rebuilt or re-fit anchor set changes every survivor count in the table above. Re-measure; never
   inherit.

---

## 5. What this means for the request

* **Ship S2b as a compute lever** at `accel_max = 2.0` (4.37× decoded, oracle survival 1.0000,
  selection unchanged on 881/881). It is already implemented behind
  `RefCConfig.sel_anchor_prefilter` with its guard, and 22 tests pin it.
* **The "better selector without retrain" is still open.** The 0.3075 selection gap (65 % of ADE) is
  untouched by any geometric band, because the band and the ranking agree wherever the band bites.
  A selector that closes it has to carry information the ranking does not already have —
  which is what the D-SEL retrain arm was pre-registered to test.
* This is consistent with the live v5f curve: `sel_gap@2s` **0.3092** at step 12,000, essentially the
  same fraction of the error. The gap is structural, not an artefact of one arm.
