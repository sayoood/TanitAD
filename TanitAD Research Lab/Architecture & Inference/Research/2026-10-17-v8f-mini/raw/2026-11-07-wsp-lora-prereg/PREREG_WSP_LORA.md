# PREREG_WSP_LORA: a weight-space read of the LoRA deltas of RENC and of the LoRA arm L (Master Mind, 2026-11-07 06:58 dev-box clock, from `date`; the PI asked for it in chat on 2026-11-07; written before any RENC encoder exists and before any L checkpoint is read)

## 1. Why
* The weight-space method (Takehi, Teoh, Zhang and Khattab, 2026-11-02, a blog post, not peer-reviewed; Library key `takehi2026weightspace`) read v8F's planner correctly. It located the crawl as shrunken kinematic gains. But edits to the planner cannot add information (T2), and both edits read HARMS.
* The levers that can move the frontier now act on the encoder: RENC (`PREREG_T2_RENC.md`, frozen prefix `425d3496`) and L (`AMENDMENT_V8F_LORA_4.md`, frozen `626aa1e5`). Both are q, v LoRAs, rank 32, alpha 32, on the 24 blocks of DINOv3-L.
* This card turns the method into a diagnostic for those levers. It says, from the weights, **where** each delta acts and **on which inputs**, before any forward pass of the adapted encoder.

## 2. Objects
* For block l in 1..24 and projection p in {q, v}: the delta ΔW_lp = (α/r) · B_lp · A_lp, with α = r = 32, and the frozen weight W_lp.
* **Base statistics, the only data the claims may use:** Σ^real_lp and Σ^render_lp, the covariances of the inputs to projection p of block l in the FROZEN encoder (no LoRA). They are computed on the held-out-fold real and rendered cylinders of the RENC data job, at the GAP48 ticks. No adapted encoder runs to make them.
* L and RENC are read on the same frames, the 48 NuRec scenes' real and rendered cylinders. For L, the fold is irrelevant, because L never saw these scenes.

## 3. The weight claims (computed from Section 2 only, written to `WSP_LORA_CLAIMS_<arm>.json` and hashed into this card's log before any forward pass of the adapted encoder)
* **W1, where.** The relative delta energy ρ_lp = ‖ΔW_lp‖_F / ‖W_lp‖_F. The **carrier set K** is the smallest set of (l, p) whose ρ² sums to at least 50 % of the total.
* **W2, rank.** For each ΔW_lp, the number of singular values that hold 90 % of its energy.
* **W3, selectivity.** The input-weighted delta energy e^D_lp = tr(ΔW Σ^D ΔWᵀ) / tr(W Σ^D Wᵀ), for D in {real, render}, and the ratio σ_lp = e^render_lp / e^real_lp.
* **The claims, one per arm:**
  * For RENC (E′_A and E′_B): (a) K holds at most 12 of the 48 (l, p); (b) σ > 1 on at least 75 % of K, so the delta acts more on render inputs than on real ones, which is what its keep term asks for.
  * For L, at its first HF checkpoint (step 1000) and at its last: the same W1 and W2, plus the median σ over K. That median is the weights' prediction of whether L's change reaches renders: σ ≥ 0.5 predicts the T2 token-use gate 5(b) of `DRAFT_PREREG_T2_DOM_LORA.md` can pass, and σ < 0.5 predicts it fails.

## 4. The checks (run only after the claims file is hashed)
* **C1, where, by ablation (RENC).** On the held-out pairs, the gain is G = median cos(P(E′(r)), P(E(x))) − median cos(P(E(r)), P(E(x))). Zeroing the LoRA on K must remove at least 50 % of G. Zeroing it on the complement of K must remove less than 50 %.
* **C2, selectivity, by measurement (RENC and L).** For each (l, p) in K, the measured ratio of ‖ΔW h‖ / ‖W h‖ on render inputs to that on real inputs is compared with σ_lp. The check holds if the measured ratio is above 1 exactly where σ_lp > 1, on at least 75 % of K.
* **C3, for L only, the prediction against the gate.** When L's gate 5(b) is read, its outcome is compared with the median-σ prediction of W3. That is reported as a hit or a miss. It changes nothing in the gate.

## 5. Outcomes, committed now
* **Per arm: WSP-LORA HOLDS** if every claim checked for that arm matches (C1 and C2 for RENC; C2 for L, then C3 when gate 5(b) is read). **PARTIAL** if some match. **FAILS** if none match. Each claim is reported as hit or miss.
* **What it sends, committed:**
  * If RENC passes M-RENC but fails G-KEEP, and K is small with σ > 1 on it (claims (a) and (b) hold), the next RENC variant restricts the LoRA to K under its own card. Fewer parameters, aimed where the weights say the render change lives.
  * If L's step-1000 median σ < 0.5, it is reported before L's T2 read as the weights' prediction that L's gain will not reach renders. The RENC recipe applied on top of L's merged encoder becomes the planned step for L's T2 read.
* This card decides no driving outcome, and it changes no bar of another card.

## 6. Scope
* T0-class: these are encoder weights and activations, with no driving metric. The estimator for C1's G is a scene-cluster bootstrap over the held-out scenes, B 2,000, seed 0. One adaptation seed per arm.
* L's checkpoints are read only after `DRAFT_PREREG_T2_DOM_LORA.md` is frozen. The weights carry no driving number, and the order keeps the draft card free of anything L shows.
* Compute: CPU on the dev box, or box 2 when it is free. Never the pod while L trains.

## 7. Log (dated, append-only after the freeze)
