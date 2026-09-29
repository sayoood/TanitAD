# PRE-REGISTRATION — REF-F: a contrastive state–action selector (CLM transfer)

**Status: DRAFT, written 2026-09-29, before any REF-F code, cache or checkpoint exists.**
**Freeze rule.** The PI approves the document, and its SHA-1 is written to `PREREG_REFF_CONTRASTIVE_SELECTOR.sha1` before the first REF-F checkpoint is written. Any later change is an amendment with a date and a reason, never an edit.
**Design:** `TanitAD Research Hub/Architecture & Inference/REFF_CONTRASTIVE_SELECTOR_PLAN.md`. **Statistics:** CLAUDE.md rules. Full-set means; episode-cluster bootstrap B = 2000, **paired** for any two arms, never the deprecated split-mean; four metric families per panel; learning-curve exponents only with window, R² ≥ 0.80 and n.

---

## 0. The claim under test

A CLM-style dual encoder can pick a driving trajectory at least as well as the programme's best flat planner, and much faster:
- a frozen vision backbone plus a small trainable state head, carrying ego kinematics;
- a small action encoder over prior-residual trajectories;
- a calibrated cosine score over cached candidates.

It is also a better *selector* over that planner's own candidates than the planner is.

---

## 1. Fixed before any data is seen

| item | fixed value |
|---|---|
| train data | parity train `physicalai-train-e438721ae894`, 2,376 episodes, skip-hash `f09e44db`, asserted in-process |
| val data | canonical `physicalai-val-0c5f7dac3b11` (40 episodes / 881 windows) **and** the 600-episode / 13,198-window deployment; never mixed, and the 600 set is the decision set because val-40's minimum detectable effect is 0.06–0.16 m |
| windows | window 8, K = 20 future steps at 10 Hz, waypoints 0.5/1/1.5/2 s, metric BEV ego frame (TanitEval protocol) |
| action | 2 s residual over the CTRA prior (v0, yaw rate, `ax_fd`), decomposed into lateral and longitudinal offset profiles |
| vocabulary | farthest-point sampling on the parity train split only: K_lat = 64 × K_lon = 64 = 4,096 entries; time-weighted L2; frequency prior stored with it |
| backbone arms | F-own (frozen v1 encoder, the default); F-vid and F-img (frozen foundation encoders, checkpoints named in an amendment before their caches are built) |
| state input | `z[8]` + last-stack patch tokens pooled to 8×8×768 + ego (v0, `ax_fd`, 8-step speed history, yaw rate); ego-history dropout 0.2 with a learned null row |
| losses | stage 1 `L_nce` (in-batch, bidirectional, mask ε = 0.25 m ADE@2s); stage 2 adds synthetic near-miss negatives; stage 3 `L_voc` (soft targets, σ = 0.5 m) + `L_cost` + `L_ref`, with 40 % stage-1 replay |
| optimisation | AdamW, weight decay 0, `lr = 2e-3·sqrt(1024/width)·sqrt(batch/1024)`, OneCycle 10 % warm-up, batch 2,048, 20 epochs, patience 5, τ₀ = log(1/0.07), e^τ ≤ 100 |
| seeds | 3 per arm (`seed_everything` + seeded DataLoader); intervals pool over episodes, not seeds; the seed spread is reported |
| selection rule | argmax of the vocabulary softmax (default); for contrastive-score ranking (the H-F4 arm and verifier mode) `e^τ⟨s,a⟩ + β̂ log p̂(a)`, with β̂ fitted on the train holdout from the grid {0, 0.25, 0.5, 1, 2} (the toy gave β* = 0.5) and open candidates taking the prior of their nearest vocabulary entry; hysteresis γ = 0 in open loop |

---

## 2. Negative controls, run first

If any of these fails in the stated way, the panel is void.

| id | control | expected | if violated |
|---|---|---|---|
| NC1 | **state permutation:** state embeddings shuffled across windows within the eval set | ADE degrades to ≥ the frequency-prior-only baseline | leakage or ego-channel dominance; stop and find it |
| NC2 | **random vocabulary:** entries sampled, not farthest-point sampled | worse than or equal to the pre-registered vocabulary | the vocabulary construction is doing nothing; report it |
| NC3 | **kNN retrieval** in the same frozen feature space | reported; REF-F should be separated-better | a tie means the heads add nothing beyond the backbone, and that is the finding |
| NC4 | **ego-only** (no images) and **image-only** (no ego history) REF-F | ego-only ≈ the no-vision ego-status ceiling (0.5735 on val-40) | a large ego-only score signals copycat risk; report the transition-window split |

---

## 3. Hypotheses: thresholds and both outcomes, committed now

| id | primary measurement (val-600 decides; val-40 reported) | CONFIRM if | then | REFUTE if | then |
|---|---|---|---|---|---|
| **H-F1** | paired Δ ADE@2s, REF-F standalone − REF-C-base | upper CI95 bound < +0.02 m | REF-F becomes the fast System-1 reference arm | lower CI95 bound > +0.02 m (separated-worse beyond the margin) | move to retrieve-then-re-rank. Between the two: INCONCLUSIVE, reported as such |
| **H-F2** | steady-window speed MAE, paired vs hold-v0; transient (brake/accelerate) windows separately | steady: upper CI95 bound of (REF-F − hold) < +0.02 m/s, **and** transient: separated-better than hold-v0 | the residual/prior parameterisation is adopted programme-wide | steady windows separated-worse than hold-v0 | the steady-window loss is not a parameterisation problem |
| **H-F3** | verifier over REF-C-XL's 256-candidate fan: gap recovery = (pick_REF-C − pick_REF-F) / (pick_REF-C − oracle) | ≥ 0.25, with the paired ADE improvement's lower CI95 bound > 0 | REF-F is the v6 selector candidate (review D2b) | improvement not separated from 0 | selection needs early interaction or cost targets |
| **H-F4** | KL(selected-class frequency ‖ ground-truth-class frequency) and steady-window ADE, InfoNCE-only argmax vs corrected (log-prior) vs vocabulary softmax | InfoNCE-only has the larger KL **and** is separated-worse on steady-window ADE than either correction | the correction is mandatory for contrastive selectors in this programme | no separated difference | the corpus prior does not bias selection enough to matter; drop the correction |
| **H-F5** | full REF-F tick on Jetson Thor (cached backbone, heads, 4,096 + 300 candidates, controller), ≥ 1,000 ticks on held-out windows, bf16 | p95 ≤ 50 ms (within-run; ±13 % drift noted) | ≥ 50 ms of the 100 ms budget remains for System 2 and the envelope | p95 > 50 ms | profile; the backbone must shrink or cache more |
| **H-F6** | four-family panel at 1 / 3 / 10 / 30 / 100 % of parity-train hours, REF-F vs REF-C at matched fractions (same episode subsets, nested) | REF-F's relative ADE loss at 10 % **and** at 3 % is separated-smaller than REF-C's | first measured evidence toward the mission's data goal | not separated at either fraction | frozen-backbone heads give no data-efficiency edge here |
| **H-F7** | H-F1's measurement per backbone, paired F-vid or F-img vs F-own | a foundation arm separated-better on ADE@2s and non-inferior on the other three families | foundation encoders are viable decision substrates; settles BACKLOG B5 | F-own non-inferior | keep the programme's own encoder |
| **H-F8** | closed loop on NuRec/AlpaSim, paired vs REF-C-base: pass rate, collisions, jerk | pass rate non-inferior (paired, n ≥ 30 scenes) and jerk not separated-worse | REF-F is a closed-loop reference | pass rate separated-worse | strengthen stage 3 (DAgger). With n < 30 the verdict is INCOMPLETE, whatever the point estimate |

**Four families.** Every panel reports LONGITUDINAL (target-speed bias and MAE; distance keeping on the LEAD windows), LATERAL (heading, curvature, yaw rate, cross-track), TACTICAL (selected vs executed lateral × longitudinal class, from the factorised vocabulary) and STRATEGIC. STRATEGIC is reported UNAVAILABLE, with reason "no route or goal source in PhysicalAI-AV" and n = 0, until a route-bearing corpus is used.

---

## 4. Stopping rules

- If NC1 fails, stop and diagnose before any other result is read.
- If V0 shows oracle-in-vocabulary ADE@2s on val-600 above REF-C-XL's pick (0.47-class), the standalone mode (H-F1) is not run until the refinement head and open candidates are in. Verifier mode (H-F3) still runs.
- No arm is tuned on val. Temperature, β and γ are fit on a held-out 10 % of *train* episodes.

---

## 5. What this is not

- **Not a test of the hierarchy.** REF-F is flat by default. Its tactical and strategic *roles* get their own tests in the review's plan (Δ4, Bet B).
- **Not a strategic-level result.** The corpus has no route information.
- **Not a safety claim.** Candidate pruning with agent states runs only as an oracle-perception simulation test until a vision range head exists.
- **Not a claim about CLM-8B itself.** The optional pilot that uses the released model as an offline teacher has its own pre-registration if the PI approves it.

---

## 6. Deliverable manifest (M0)

| artifact | where |
|---|---|
| this pre-registration (DRAFT) | `repo:Project Steering/PREREG_REFF_CONTRASTIVE_SELECTOR.md` |
| design and plan | `repo:TanitAD Research Hub/Architecture & Inference/REFF_CONTRASTIVE_SELECTOR_PLAN.md` |
| research stream | `repo:TanitAD Research Hub/Architecture & Inference/Research/2026-09-29-reff-prior-art-and-theory.md` |
