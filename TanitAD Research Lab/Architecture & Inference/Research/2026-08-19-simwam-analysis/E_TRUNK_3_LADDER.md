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

## 5. ⛔ THE CONSEQUENCE I ATTACHED TO THIS RESULT IS RETRACTED (2026-08-20, PI challenge)

**The MEASUREMENT stands. The RECOMMENDATION does not.** The pre-registration's
World-B branch said *"Route 1 — freeze a strong pretrained encoder — becomes the
primary recommendation."* I wrote that consequence without checking two things
that were already in the registry, and both refute it:

1. ⛔ **It contradicts a STANDING PI DECISION.** `D-003` (2026-07-05):
   *"Main track = from-scratch 4-brain latent world model; **frozen-encoder is a
   comparison arm, not a hedge to adopt** … the from-scratch arm is what makes
   the data-efficiency claim disruptive."* A pre-registered consequence may not
   quietly overturn a PI decision.
2. ⛔ **REF-A IS THE COUNTEREXAMPLE, AND IT IS ALREADY MEASURED.**
   `refa-4brain-speed-30k` — frozen DINOv2-B/14 — scores ADE@2s
   **2.1322 ± 0.1821**, *"does not beat CV"*, ran 14 k = 2.05 → 30 k = 2.14 →
   **plateaued**, and the registry's own reading is *"not overfitting — it is at
   a capability ceiling."* **flagship > REF-A by a paired +2.6200 m
   [2.0945, 3.2570].**

⭐ **THE INFERENCE THIS BREAKS IS THE LOAD-BEARING ONE.** A frozen DINOv2 trunk
almost certainly has far better environment decodability than what §1 measures
for v6 — and it **drove 2.62 m WORSE**. ⇒ **Decodability does not translate into
driving.** E-TRUNK-2/3 measured a real property of the v6 latent at **T0**; the
step from that property to *"therefore freeze the encoder"* is exactly the tier
crossing `EVAL_DOCTRINE` forbids, and the programme has already run the
experiment that refutes it.

⚠️ **What survives.** *"The v6 operative latent does not linearly expose
environment properties, and did not at any measured step"* is unaffected — it is
a T0 statement about a representation and it is measured. What is withdrawn is
every architectural prescription derived from it.

⚠️ **What is NOT established either.** REF-A used frozen **DINOv2-B/14 at 224 px,
16×16, d=768**; the reference arm here is **DINOv3 ViT-L/16 at 256×640, 16×40,
d=1024**. REF-A's ceiling is measured for *its* configuration, not for every
frozen encoder. The burden sits on any new frozen proposal to show it is not
REF-A again — it is not discharged by pointing at a probe.

## 5b. ⭐ SIGReg IS ON, IS WORKING, AND THAT CHANGES THE DIAGNOSIS

MEASURED from the live run's own config: **`w_o6 = 0.1`**, `sigreg_slices = 512`,
and O3 masking active (`o3_mode = action`, `o3_blocks = 2`, `w_o3 = 1.0`). So the
programme's anti-collapse and physics-teaching machinery is **present and
engaged**, and any framing that implied otherwise was wrong.

⭐ **And it is demonstrably working**: participation ratio **rises** 3.29 → 6.94
and top-8 share **falls** 0.836 → 0.806. SIGReg is preventing dimensional
collapse exactly as designed.

⇒ **The real finding is sharper than "collapse":** an anti-collapse regulariser
guarantees the dimensions are *USED*, never that they are *INFORMATIVE*. A
variance/isotropy constraint **can be satisfied by noise**. The v6 latent is
spread out **and** uninformative — which is not a failure of SIGReg, and not
collapse, but the limit of what an unsupervised predictive objective plus an
isotropy prior can be expected to induce on its own.

⚠️ **One config fact flagged for the PI, NOT diagnosed as a defect:**
`sigreg_free_dims = 0`, and `position_relaxed`'s docstring says that
*"reproduces plain SIGReg on the full latent"*, the configuration the §B.3
relaxation exists to replace (*"the diagnosed step-21k regression mechanism"*).
Whether v6's operative latent has reserved ego-motion channels for the exemption
to protect is a **design question I have not verified**, so this is reported as
an open item, not a diagnosis.

## 6. What follows

1. ⛔ **NO architectural recommendation follows from this document.** See §5.
   The measurement is T0; REF-A already showed a better-decoding frozen trunk
   driving 2.62 m worse. The next decision needs a **T1 comparison**, not a probe.
2. **What a probe CAN legitimately do** is act as an early screen inside an arm
   that is already committed — it caught this at 67 % of one stage, at zero GPU,
   from banked artifacts. That is its role, and it is a real one.
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
