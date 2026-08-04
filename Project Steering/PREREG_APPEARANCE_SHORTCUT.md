# PRE-REGISTRATION — the appearance-shortcut audit (D-APPEAR)

**Written 2026-08-03 BEFORE any PhysicalAI number was produced.** Both outcomes are committed
here in advance, per the operating standard's rule 5 (*settle conflicts with the cheapest
discriminating experiment, pre-registered with both outcomes committed in advance*).

Extends `…/incoming/2026-08-03-latent-bottleneck/LATENT_BOTTLENECK.md` §0.0, whose headline is
**MEASURED on comma2k19 highway only**:

> a single 32×32 grayscale **still frame** through an rbf ridge reads `speed` at **R² +0.6642**,
> against the 18,432-feature 800 ms learned latent window's **+0.7145** — **93 %** — and
> **1.75×** the best motion-only arm. All ten linear pure-difference arms sit at the null.

⚠️ Highway is the **easiest** case for an appearance shortcut: near-constant road furniture, a
narrow speed distribution, little manoeuvre variety. The claim is a HYPOTHESIS at programme scale
until it is measured where the furniture stops co-varying with speed.

---

## 1. The primary question (P1)

**Does the still frame keep ~93 % of the learned latent's `speed` read on PhysicalAI-AV — a
corpus that is NOT highway-dominated?**

**Primary statistic** (fixed now):

```
RATIO = R²_speed( pix32_centre_rbf )  /  R²_speed( v1_window )
```

both on the same held-out windows, same split, same ridge recipe, encoder-matched
(`v1_speedjerk_ckpt.pt` step 29999 — the SAME frozen encoder both corpora were probed with).

**comma2k19 reference value: 0.6642 / 0.7145 = 0.9296.**

| pre-registered outcome | decision rule (fixed in advance) | what it means |
|---|---|---|
| **OUTCOME S — SHORTCUT SURVIVES** | `RATIO ≥ 0.70` on PhysicalAI **and** `pix32_centre_rbf` separates from its own shuffled control | the shortcut is **not** corpus-specific. The programme-scale claim stands and P2/P3 become load-bearing. |
| **OUTCOME C — CORPUS-SPECIFIC** | `RATIO ≤ 0.40` | the shortcut is a comma2k19-highway artifact. **The programme-scale claim is WITHDRAWN** and `LATENT_BOTTLENECK.md` §0.0's last paragraph must be corrected to say so. |
| **OUTCOME P — PARTIAL** | `0.40 < RATIO < 0.70` | appearance is a real but weaker route off-highway. Report the ratio and the stratification; make no programme-scale claim either way. |
| **VOID** | `v1_window` does **not** separate on `speed` against its shuffled control, or the empirical null arm is not reproduced | the panel cannot answer the question; report VOID, do not read the ratio. |

**Admissibility gate, fixed now:** the run is admissible only if (a) the `NULL_train_mean` arm
reproduces the train-mean floor, (b) `v1_window` separates on `speed`, and (c) the shuffled control
of every quoted arm does not separate.

### 1a. Stratification — committed in advance, because a pooled number hides the regime

Report the ratio **per speed bin** (`[0,1)`, `[1,3)`, `[3,6)`, `[6,10)`, `[10,15)`, `[15,∞)` m/s)
and **per manoeuvre** (`refb.MANEUVER_CLASSES`). ⚠️ The corpus stream measured the tactical lossy
rate as strongly speed-dependent (**38.2 % at 1–3 m/s → 1.8 % at 10–15 m/s**), so a pooled ratio
is not admissible on its own. Bins with `n < 100` held-out windows or `< 5` held-out episodes are
reported as **UNPOWERED**, not as evidence.

### 1b. The frame-order-shuffle control — committed in advance

Shuffle the **order of the 9 window positions** (same frames, same marginal, order destroyed),
per window, refit. If `R²_speed` is unchanged the read uses **no temporal order at all**.
Decision rule: `|ΔR²| ≤ 0.05` ⇒ order-free.

---

## 2. P2 — does it re-explain the cross-rig collapse?

The measured collapse is frozen-v1 speed **R² +0.930 → −2.465** across the camera rig
(`…/incoming/2026-07-22-idm-proof/results.json`), currently attributed to **camera geometry**.
PhysicalAI AV front-wide has two rigs (cy ≈ 543 rig A / cy ≈ 755 rig B) and a geometric-centre crop
is ~215 px wrong for rig B — so geometry is a live competing explanation with independent support.

**An appearance shortcut and a geometry shift both predict a cross-rig drop.** The pre-registered
discriminators, with their opposite predictions written down now:

| # | contrast | **geometry** predicts | **appearance shortcut** predicts |
|---|---|---|---|
| G1 | horizon row of the cached frames, rig A vs rig B | an offset — the crop is misaligned in **this** substrate | no offset required |
| G2 | cross-rig drop of a **motion-energy** arm vs a **still-frame** arm | both drop **similarly** — same pixel grid, same shift | the **still-frame** arm drops **much more**; motion energy is a physical flow→speed proxy that should transfer modulo scale |
| G3 | a **synthetic vertical shift** of held-out frames, swept to the rig-B-equivalent offset | reproduces a large part of the collapse | reproduces little of it |
| G4 | **within-rig** held-out (A→A, B→B) vs **cross-rig** (B→A, A→B) | within-rig should be fine for both | within-rig fine, cross-rig collapses for appearance-carrying arms |

**Decision rule:** if G1 shows **no** horizon offset in this substrate **and** G3 fails to reproduce
the collapse, geometry is **not sufficient** here and the appearance reading gains. If G2 shows the
motion arm dropping as hard as the still arm, the appearance reading does **not** gain.
⚠️ If the contrasts point in different directions, the registered answer is **"cannot separate"** —
that is an admissible outcome and must be reported as such rather than resolved by preference.

---

## 3. P3 — does it threaten the scenario classifier?

Situation labels are a deterministic function of the ego pose track (`stack/tanitad/data/situations.py`
and `scripts/emit_situation_labels.py:54-62`). The PI's binding ruling makes **`head_img`** the only
deployable arm. If a still frame reads speed at ~93 % of the latent, `head_img` may be scoring
**appearance → speed → the ego-derived label** rather than perceiving the situation.

**Pre-registered arms** (all on the banked sitclf substrate, same clip clusters):

| arm | features | what it is |
|---|---|---|
| `img_latent` | frozen v1 2048-d | the deployable arm's substrate |
| `img_still32` | one 32×32 grey still frame (1024) | pure appearance, no temporal order |
| `ego_speed_true` | 1 — the true ego speed | the **leak ceiling** through the speed channel |
| `speed_from_appearance` | 1 — speed **predicted from a still frame** | ⭐ the shortcut path itself |
| `img_latent_resid` | `img_latent` scored on label residual after removing `ego_speed_true` | what vision adds **beyond** the speed channel |
| `*_shuf` | the permuted-feature null for each | the control every arm is read against |

| pre-registered outcome | rule | meaning |
|---|---|---|
| **THREATENED** | `speed_from_appearance` reaches ≥ 50 % of `img_latent`'s AP-lift on a situation | a large part of the "vision-only" score is the appearance→speed shortcut; escalate to the sitclf stream |
| **NOT THREATENED** | `img_latent_resid` keeps ≥ 70 % of `img_latent`'s AP-lift | vision carries situation information beyond the speed channel |
| **MIXED** | otherwise | report both numbers, claim neither |

⛔ **A separate stream owns sitclf. This audit MEASURES and ESCALATES; it does not edit sitclf files
and does not change any sitclf verdict.**

---

## 4. P4 — the screen becomes a rail

The §6 latent screen (jitter ratio ≤ 2×, derivative corr > 0.50, derived-accel R² > +0.50 /
speed-read σ ≤ 0.28 m/s) is implemented as a repo instrument with tests, so no future arm can be
launched on an unscreened latent. ⚠️ The **σ ≤ 0.28 m/s / ~21×** figure is estimator-specific
(optimal 9-point Savitzky-Golay); the older *σ ≲ 0.1 m/s, ~47×* was a 2-point centred difference.
**The estimator travels with the number, in the code and in the output.**

No gate threshold is swept in this run; the thresholds are inherited from §6 and re-calibrated only
by adding a second screened latent to the record (which this run does).

---

## 5. Estimator and protocol — fixed now

* **Paired episode-cluster bootstrap**, B = 2000, `tanitad/eval/ap_ci.py`
  (`paired_stat_episode_cluster_bootstrap`). ⛔ Never `overlapping_holdout_se`.
* Episode-disjoint split `i % 3 == 0` held out; inner split of TRAIN only for hyper-parameter
  selection; exact dual ridge over the whole α path with the exact-mean sentinel; skill gate 0.01;
  one-SE tie-break. **Identical recipe to the comma panel** — the only things that change are the
  corpus and the substrate.
* Every arm is read against **its own shuffled control**, never against another arm's interval.
* **Four metric families** reported per family, never pooled. Where a family cannot be computed,
  the reason and the `n` are stated per family.

## 6. What this run will NOT establish — written in advance

* It cannot establish anything about **v5f's wide-FOV cylindrical substrate** unless a banked v5f
  latent for these clips is found; if it is not, that is reported as NOT DONE, not narrowed away.
* It cannot re-open the `long_accel` verdict (OUTCOME V), which is pre-registered elsewhere.
* The corpus here is the **r0 500-clip PhysicalAI build**, cache keys
  `physicalai-train-14231cd29c74` / `physicalai-val-bb543bdf7836` — **NOT** the canonical training
  corpus `physicalai-train-e438721ae894`. This is a probe, not a training arm; no episode is
  re-selected for any training arm. The distinction is stated with every number.
