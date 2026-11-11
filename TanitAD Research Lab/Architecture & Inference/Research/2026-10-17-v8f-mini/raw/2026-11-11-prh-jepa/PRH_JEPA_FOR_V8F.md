# The Platonic Representation Theorem (Balestriero & LeCun, arXiv:2610.39302) and what it changes for TanitAD and v8F (2026-11-11)

**Source:** Balestriero & LeCun, *Proving and Improving the Platonic Representation Hypothesis with JEPAs*, arXiv:2610.39302v1, 29 Oct 2026, CC BY 4.0. Banked as library key `2610.39302`, sha256 `083cc422…a6838`, 3,166,374 B. Every paper number below is PUBLISHED and read from the primary (HTML v1).

**Evidence classes.**
* Claims about our code: MEASURED from source, with `file:line`, in the read tree `F:/TanitAD-scratch/egowm/tree` (`stack/...`).
* Claims about what this means for v8F: HYPOTHESIS until a pre-registered read settles them.
* Nothing here is a driving result, in any tier.

## 1. What the paper proves (and its limits)

1. **A JEPA is the spectral embedding of its positive-pair graph (Prop. 1).** Take the identity predictor, squared error, and whitening anti-collapse (`E h = 0`, `E hhᵀ = I`). The minimisers are then exactly `Q·Φ_π` with `Q ∈ O(k)`: the top-k eigenvectors of the normalised adjacency `K_π` of the graph the positives draw (Laplacian eigenmaps). They are unique up to rotation when `μ_k > μ_{k+1}`. Loss minimum: `2 Σ_{i≤k}(1 − μ_i)`.
2. **Platonic Representation Theorem (Thm. 1).** Any number of modalities (or runs) with the same positive-pair distribution learn the same embedding up to one rotation each.
3. **Kernel mismatch bounds misalignment (Thm. 2).** With different positives, the RMS error after the best rotation is `≤ (2√2/δ)·min(√k‖K₁−K₂‖₂, ‖K₁−K₂‖_F)`, with δ the eigengap. Alignment lives in the leading, well-gapped directions. MEASURED in the paper: best-rotation correlation **0.70 on the first principal direction, ~0.1 by the 60th** (Fig. 6).
4. **Bad views carry nuisance (Thm. 3, Prop. 3).**
   * *Bad views*: two corruptions of one observation, such as crops, masks, or spans of one caption.
   * Bad views spend `r` embedding directions on the modality's own nuisance whenever a nuisance direction keeps more correlation than the world's k-th direction (`β₁ > α_k`). No rotation can align those directions.
   * *Good views*: two observations of one world state, with independent nuisances. Good views spend none.
   * **DINOv2/v3 are bad-view JEPAs by this definition**: their positives are crops of one photograph.
5. **Next-token prediction is a bad-view JEPA (Prop. 4).** Its anti-collapse term is the log-partition, and it is identified only up to `GL(k)`, not a rotation.
6. **Empirical results.**
   * Good views give **42× fewer samples** for the same alignment as next-token prediction on the same corpus (21.6M vs 900M).
   * A 109M text JEPA scores WIT mutual-10NN **.184**, ahead of Qwen2.5-72B at **.154**.
   * **Trimming** (each view keeps only what the observations share) is the knob that matters: whole captions lose a quarter (.185 vs .246). Mask, view count, batch and width move it ≤ .016.
   * **Gradient clipping** prevents most SIGReg collapses: 3 of 36 default runs collapsed without it, abruptly, between epochs 20 and 60.
   * A **scaling law** in the good-view fraction q and samples M predicts unseen runs within seed noise (RMSE .0093 vs .0096).
   * **k pairs** fix the rotation (orthogonal Procrustes).
7. **Limits the authors state, which matter for us.**
   * The analysis does **not cover a learned, action-conditioned predictor**, which is exactly what a world model is (§2.1).
   * It assumes a finite world, injective renderings and population optima.
   * Uniqueness is proved for second-order anti-collapse, not for SIGReg's full-distribution constraint.
   * Good *image* views were tested only on MSR-VTT frames. The image-side gain is small: mutual-10NN **.039 → .043** with good image views alone, against **.039 → .059** with good text views alone.
   * The measured gains are cross-modal **alignment and retrieval**, never control or planning.

**The single translation for TanitAD:** *what a JEPA latent contains is set by which pairs we call positive, and by its eigengaps. It is not set by scale and not by the seed.* That is a design lever we have not been pulling deliberately.

## 2. Where v8F sits in this theory (MEASURED from source)

| v8F component | what it is in the paper's terms | consequence |
|---|---|---|
| frozen **DINOv3-L/16**, pooled 4×10 (paper v3 §v8F) | a **bad-view JEPA** (crop positives) | Its features spend directions on per-image nuisance: viewpoint, lighting, texture. Viewpoint is *signal* for O (the paper itself says a world model must track its viewpoint), but nuisance for T and S. |
| shared **adapter** | **affine** in the layer-normalised features: `adapter(f)+pos` (`v8f_mini.py:1119-1121`); the pooled path is `Linear(LayerNorm(f))+pos` (`v8f_mini.py:1123-1125`) | For an affine encoder on frozen features, Prop. 1's whitened optimum has a **closed form**: the top canonical directions between the two views' features. It needs no gradient, and it is **seed-free**. |
| WM auxiliary: H-JEPA on the adapter token-mean, `z1 = proj(mean_tokens(adapter))` (`v8f_mini.py:889-902`), SIGReg weight `wm_beta = 0.08` and IDM `wm_gamma = 100` (`v8f_mini.py:354-355`, after H-JEPA's DROID values, `hjepa.py:435-439`) | a **predictor** JEPA with action conditioning: **outside the theorem** | The theory speaks to the *representation* the WM reads, not to its dynamics. The paper's λ = 0.02 belongs to a different recipe (invariance + SIGReg); our 0.08 is not thereby wrong. |
| gradient clipping | `--clip-grad` default **1.0** (`v8f_mini_train.py:286`), applied at `:2778` and `:3422` | Already in place, so the paper's main collapse fix is present. A collapse **monitor** (effective rank of z1 per level) is not known to exist: work item W3. |
| clocks: WM levels step at 0.2 / 0.6 / 3.0 s (paper v3 §4B "World-model planning") | positives at a **time lag Δ** define a temporal kernel | **The lag of the positive pair sets which state the spectral embedding keeps**: what persists over Δ stays, and what changes faster is dropped. This gives a theorem-backed construction of "the slower clock abstracts", which the 4B architecture so far asserts by down-sampling (paper v3, property (1)). |
| H-JEPA planner: `argmin ‖ẑ − g‖²` over **all** latent dims (paper v3, eq. world-model planning) | latent distance in a space whose tail directions are **not identified** (Thm. 2, Fig. 6) | HYPOTHESIS H-PRH-PLAN: the latent-distance cost spends weight on directions that are neither predictable nor seed-stable. That could be part of why *"a first planner read … did not beat holding the action"* (paper v3 §4B Status). It is testable with zero training (lever L3). |
| tactical code: softmaxes over 5 lat / 7 lon, sigmoid goals (paper v3, eq. tactical code) | softmax heads are next-token-like, `GL(k)`-identified | Harmless at the interface, because O reads the code (probabilities), not T's hidden state. No action. |
| seeds (`H-ESTIM-SEED-1`) | Thm. 1: same positives ⇒ same embedding up to rotation | Gives a new instrument for **training variance in representation space**: CCA / Procrustes between replicate arms, beside the metric floors. This is the PRH-SEED card, running now. |
| PhysicalAI-AV **multiple cameras**; Thor rig with **4 cameras** | two cameras at one instant = **good image views** of one world state, independent viewpoint nuisance | The paper's own missing piece ("scaling good image views is left to future work"). Our corpus has them natively. Today we read one camera (`camera_front_wide_120fov`). |
| Alpamayo **CoT text** in the v7.x label release (`provenance: "vlm-cot"`) | a text modality of the same world states | One rotation could join a text JEPA to the vision latent (captioning / goal language) with **k pairs** (Prop. 2). Research only; low priority. |

## 3. Levers, ranked (cheapest discriminating first; Rule Zero: each failed read names the next)

**L1 — PRH-SEED probe (RUNNING, CPU, zero training).**
* Card: `theory/prh/PREREG_PRH_SEED.md` (frozen before data).
* It asks whether M2b_s0 and M2b_s1's trained adapters agree beyond random-init adapters (P1), and whether they lean toward the closed-form temporal-CCA ("SLOW", Δ = 1 s) subspace rather than PCA (P2).
* Outcomes O1 / O2 / O3 each commit a next step: L2 or a supervised-subspace read.
* Cost: minutes of dev-box CPU.

**L2 — Closed-form spectral adapters, one per clock (the headline v8F lever).**
* Construction: for each level ℓ with clock τ_ℓ ∈ {0.2, 0.6, 3.0} s, take the top-d_ℓ canonical directions between `nm_t` and `nm_{t+τ_ℓ}` of the frozen DINOv3 token-mean (same clip). That is Prop. 1's population optimum for a linear encoder with temporal good views.
* Properties by construction:
  * whitened;
  * ordered by predictability μ_i;
  * deterministic, so no seed variance enters through the representation;
  * cheap: one pass over the train-split features.
* Use: as the WM latent basis (ladder arm), and as the S-level context input. The slowest directions should carry place, lighting and weather, which is what S's detected-context posteriors read (paper v3 §4B, the strategic level).
* ⚠️ For O the slow subspace is the wrong one: O keeps the spatial grid (paper v3: *"never a global pool"*). L2 is for the pooled WM path and the upper levels only.
* Cost: a box-2 ladder slot; the PI's call. Pre-register after L1 reads.

**L3 — Predictability-weighted planner cost (zero training, ladder planner test).**
* Re-score the existing LP planner with the latent cost weighted by μ_i, or truncated at the eigengap, in a basis fit on the train split.
* H-PRH-PLAN predicts that the weighted cost beats the full-256 cost **and** the hold-action floor more often.
* Both outcomes committed in its card.
* Needs: L1's per-direction spectrum, and the LP harness (`theory/ladder_plantest/`, currently re-running w3par).

**L4 — Good image views from cameras at one instant.**
* Use front-wide + front-tele trimmed to their overlap ("trimming is the knob"), then the cross cameras at their overlaps.
* Prop. 3 predicts this removes viewpoint and lighting nuisance from the T/S latent.
* **Blocked** on two things:
  1. a read-set change (program-wide features 6 → 7; the pinning test `test_physicalai_feature_readset.py` must move with it);
  2. a DINOv3 extraction pass over a second camera (ESTIMATED: the same order of GPU time as the front-wide cache).

  Both are PI decisions.

**L5 — Privileged-label kernels (admissible under the 2026-08-03 ruling: labels may use ego; inference is vision-only).**
* Define positives as windows from *different clips* whose ego-derived tactical labels (manoeuvre / lon class / constraint bucket) match, or whose future ego trajectories are close.
* The privileged signal then shapes **which pairs are positive**, and never enters at inference.
* ⚠️ Disjointness rule: the **situation classifier's labels must not define the kernel of any trunk that feeds T's goal path**. Otherwise the goal path learns the classifier's partition: the back-door the 2026-08-03 ruling forbids, by a new route. Tactical labels are fine; situation labels are excluded.
* Cost: one training arm. After L2.

**L6 — Text alignment of the CoT (research lab, no slot).**
* A good-view text JEPA over multiple CoT descriptions per clip, joined to the vision latent by one Procrustes rotation fit on k pairs.
* Payoff: an interpretable tactical readout and a language-goal interface.
* Only one CoT per clip exists today. More would be generated text (private HF).

## 4. Work items opened (0-GPU)
* **W1** — write the L3 card once L1 has read.
* **W2** — write the L2 card once L1 has read (it states the ladder argv and the bars).
* **W3** — collapse monitor: log the effective rank (participation ratio) of z1 per WM level on every eval dump, so an abrupt collapse (the paper's 20–60-epoch signature) is visible.
* **W4** — register `H-PRH-SEED-1` (pre-registered, running) and `H-PRH-PLAN` (hypothesis, no card yet) in `GOALS_AND_CLAIMS.md`.

## 5. What this paper does NOT license
* Any claim that v8F's WM will "converge to the world" by scale: the theorem excludes action-conditioned predictors.
* Any driving claim: every lever above must earn its place in T1 with the four families (STRATEGIC reported PAUSED).
* Treating DINOv3 as "the world's representation": by the paper's own definition it is a bad-view JEPA.
* Treating the paper's numbers as transferable. Its alignment numbers are cross-modal retrieval on captions; none is a statement about control.
