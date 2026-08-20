# E-TRUNK-3 — the environment information was NEVER ACQUIRED, and it is not collapse

`MEASURED (ours; dev-box CPU dual ridge)` · **T0-DIAGNOSTIC** ·
pre-registered in `PREREG_E_TRUNK_3.md` (**committed `548548f` before any ladder
number was computed**) · 5,617 frames · 130 episodes · episode-disjoint 5-fold ·
scale-normalised dual ridge · episode-cluster bootstrap · **0 GPU, 0 retraining**.

---

## 1. Result

Identical frames, identical folds, identical probe. **Only the checkpoint moves.**

| step | % of S-W | participation ratio | top-8 share | `lead_gap_m` R² | `left_occ` AUC | `right_occ` AUC |
|---|---|---|---|---|---|---|
| **2000** | 6.7 % | 3.29 | 0.836 | **−0.0093** [−0.0948, +0.0537] | .5554 | .6286 |
| 16000 | 53 % | 6.94 | 0.765 | −0.0147 [−0.0421, +0.0019] | .5340 | .5911 |
| 18000 | 60 % | 6.78 | 0.772 | −0.0190 [−0.0502, +0.0010] | .5344 | .5930 |
| 20000 | 67 % | 4.90 | 0.806 | −0.0176 [−0.0454, +0.0001] | .5321 | .5890 |
| — | reference | **40.77** | **0.348** | **+0.3792** (`dino_pooled`) | **.8312** | **.8236** |

**Controls (checkpoint-independent, computed once):** `C-EGO` `lead_gap_m`
**+0.334**, `ego_speed` **1.0000**; `C-PIXEL` **+0.002** / −0.051.

**Falsifiers, all passed:** (1) every cache carried the **same keys in the same
order** — the runner refuses otherwise; (2) controls are constant by
construction, which (1) licenses; (3) the `C-EGO`→`ego_speed` **identity map
read 1.0000**.

## 2. ⭐ Verdict against the committed decision rule

The rule, quoted from the pre-registration: *"decodability is **flat at ~0 from
step 2000** ⇒ **WORLD B — NEVER ACQUIRED**. No counterweight will help. **Route 1**
(frozen strong encoder + trained predictor) becomes the primary recommendation,
and REF-D's frozen-prior bet is directly supported."*

`lead_gap_m` is **−0.0093 at step 2000 and −0.0176 at step 20000** — below zero
throughout, never separated, no early peak to decay from. ⇒ **WORLD B.**

⛔ **This is NOT the "prediction objective destroyed the content" story.** There
was no content to destroy at 6.7 % of training, and none appeared later.

## 3. ⛔ The collapse framing is RETIRED — third correction in this investigation

The pre-registration committed a row for exactly this case: *"decodability falls
while participation ratio **rises** ⇒ the deficit is **not** dimensional
collapse; drop the collapse framing entirely."*

**Participation ratio RISES across the ladder: 3.29 → 6.94 → 6.78 → 4.90**
(non-monotone, with a dip back at 20 k). Top-8 share **falls** 0.836 → 0.806.
The representation is becoming **less** concentrated while decodability stays
pinned at zero.

⇒ **"Collapse" is the wrong word and is withdrawn.** The correct statement is
**the representation is not concentrated — it is uninformative about the
environment, and always was.** Spreading a representation out does not make it
see. *(Retraction chain in this investigation: collapse-onto-ego → refuted by
probe; "2.3 of 2048 dimensions" → inadmissible, C128; "collapse" as the framing
→ retired here.)*

⚠️ The anisotropy-vs-reference statement from `E_TRUNK_2` §3.2 **still stands as
a measurement** (PR 4.90 vs 40.77) — what is withdrawn is the causal reading
that concentration is the *mechanism* of the deficit.

## 4. ⛔ What this does NOT settle

* **World B does not exonerate the objective.** It is equally consistent with
  *"this objective never induces perception"* and with *"2,376 episodes is too
  little to learn perception from scratch"*. Those separate **only** by training
  a different objective on the same corpus — not by any probe.
* ⚠️ **Step 2000 is already 6.7 % through S-W.** Content present at
  initialisation and destroyed within 2 k steps would be invisible here. The
  earliest available cache **bounds** the claim.
* ⚠️ **The ladder is 4 rungs, not the 9 the pre-registration first claimed** —
  five caches were pruned to metadata only, and the amendment is recorded in
  that document. **1 early anchor + 3 late rungs**: the LOST/NEVER-ACQUIRED
  question separates, but *when* anything happened between 2 k and 16 k does not.
* **Linear probe, one seed, one corpus.** Non-linearly encoded content reads as
  absent.
* **T0-DIAGNOSTIC.** Nothing here is a driving claim.

## 5. What follows

1. ⭐ **Route 1 is the committed recommendation** — a frozen strong encoder with
   a trained predictor. This is DINO-WM's actual recipe, V-JEPA 2-AC's shape, and
   **REF-D's existing design**. The programme does not need a new bet; it needs
   to stop making the one that is failing.
2. ⛔ **Do not spend on Route 2/3 as the primary fix.** An aux perception head or
   an anti-collapse term counteracts a *destruction* mechanism, and the ladder
   says there is nothing being destroyed. *(An aux head may still be worth
   having — but as a way to ACQUIRE, not to protect, and that is a different
   experiment with a different justification.)*
3. **The S-W trunk should still finish 30 k.** It costs nothing extra, and a
   clean 30 k checkpoint is the honest baseline any replacement must beat.
   ⚠️ **S-T should not launch on this trunk** until the PI rules, because S-T
   trains a tactical layer *on the frozen S-W trunk* — i.e. it would build on
   exactly the representation measured here.
4. **The cheapest next discriminator**, if the objective-vs-data question matters
   for the paper: train the *same* encoder with an added perception target on the
   *same* corpus. If it then decodes, the corpus was sufficient and the objective
   was the limit. That is the only experiment that separates the two, and it is
   worth pre-registering before it is run.

## 6. Manifest

| artifact | where |
|---|---|
| pre-registration (with its amendment) | `…/simwam-analysis/PREREG_E_TRUNK_3.md` |
| this result | `…/simwam-analysis/E_TRUNK_3_LADDER.md` |
| `e_trunk3_ladder.json` (every point, CI, spectrum, λ edges) | `…/simwam-analysis/raw/` |
| `e_trunk3_ladder.py` (falsifiers enforced in code) | `…/simwam-analysis/code/` |
