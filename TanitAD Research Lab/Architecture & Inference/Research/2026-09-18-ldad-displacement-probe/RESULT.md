<title>LDAD transfer probe — the frozen trunk's displacement carries LONGITUDINAL action; the LATERAL read is VOID on my own control</title>

# `E-AI-LDAD-0`: Delta-JEPA read in full, and its mechanism tested on our frozen trunk. **LON → `DZ-TRUNK`** (R² 0.33 vs a pixel floor ≈ 0). **LAT → VOID-LEAK**: the shuffle control failed, and the defect is in my SPEC

**2026-09-18 · Research Lab (LAB-RUN-015) · Architecture & Inference · Band A2 full-text read + a 0-GPU transfer test**
⛔ **Tier: none.** A linear probe on frozen features. No rollout, no driving claim.
⛔ **Pre-registered:** `SPEC.md` + `code/ldad0_probe.py` sha256 pinned in `raw/prereg_pin.json` **before** the run.

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⭐⭐ **Delta-JEPA `2606.31232` (FULL TEXT, the debt left by 09-17's failed extraction): decoding the executed action from the latent *displacement* `Δz = z_{t+1} − z_t` beats decoding it from concatenated endpoints on all four tasks**: +4.07 / +1.07 / **+12.60** / +0.67 pp planning success (Two-Room / Reacher / Push-T / OGB-Cube, 3 seeds, Table 2). λ = 0 **nearly collapses** (Fig. 3); best λ = 50. Their Fig. 6 test (fix `z_t`, swap the action, measure `ẑ_{t+1}(a) − ẑ_{t+1}(0)`) is **exactly our P-2 action/scene-ratio problem**, stated as a training objective. | PUBLISHED lib `2606.31232` (full text, pypdf; 13 pp) |
| **F2** | ⚠️ **Its limits, in its own numbers:** 4 toy control tasks, ViT-Tiny, 50 epochs, **3 seeds**. The Reacher margin over Sub-JEPA is **+0.33 ± 0.50/2.40**, i.e. within noise. The encoder is trained **end-to-end**, and the loss acts on the *encoder's* geometry. ⇒ it does **not** transfer as-is to a frozen trunk (AI13-2). | PUBLISHED |
| **F3** | ⭐⭐⭐ **`DZ-TRUNK` on LONGITUDINAL: the frozen refcv5-v2 trunk's displacement decodes ego acceleration over 0.8 s at R² 0.3297 [0.2384, 0.3957].** That is +0.3279 [0.2344, 0.3953] over the pixel-displacement floor and **+0.2760 [0.1777, 0.3607] over the start frame alone**. All controls pass: C-const **0.0000**, C-oracle **0.9999**, C-shuffle **0.0311 [−0.0122, 0.0669]** (≤ 0.05, and its CI upper < 0.3297). n_fit 4,514 pairs, d 704, scored 1,922 pairs / 40 clips. | MEASURED `raw/ldad0.json` |
| **F4** | ⛔ **LATERAL is `VOID-LEAK` as written.** C-shuffle read **0.3705 [0.3045, 0.4077]** against a bar of ≤ 0.05. ⚠️ **The cause is my control, not the trunk:** the shuffled displacement `z_rand − z_t` still contains `−z_t`, and yaw rate is **readable from the scene alone** — `latent_start_only` R² **0.8847 [0.787, 0.9295]** (the road's curvature is visible). So the shuffle does not remove the scene term for a target the scene determines. Descriptively LAT is `DZ-SCENE` (`latent_dz − latent_start_only` **−0.8095**), but no verdict is claimed. | MEASURED |
| **F5** | ⚠️ **The pixel floor reads ≈ 0 on both targets** (LON 0.0018, LAT 0.0004; λ selected at the maximum 1e6). The floor is **linear on 8×8-pooled pixels**, and a pixel *difference* is not optical flow. ⇒ it is a **weak** floor, and "beats pixels" here means *beats a linear read of coarse pixels*, nothing stronger. | MEASURED |

**Verdicts (SPEC §4):** LON → **`DZ-TRUNK`**. LAT → **VOID-LEAK** (control failed; defect in the SPEC's control design, reported not repaired).

## 1 · What it means

* **For LDAD on our stack:** the longitudinal half of the action is **already linearly present in the frozen displacement**, and it is *not* present in the scene alone. ⇒ an LDAD lever for us is **predictor-side**: supervise the **predicted** displacement `ẑ_{t+1} − z_t` to decode the action. The encoder cannot move, so an encoder-side LDAD cannot act. This targets P-2 (h=1 action/scene ratio 0.004) head-on, because P-2 is a property of the **predictor's response to the action**, and the frozen displacement shows the signal is there to be matched.
* **Combined with VGEO-5 (today, Deploy):** the frozen trunk carries **place identity** (P2 +0.2088) and **longitudinal displacement** (here), but **not path length ordering beyond pixels** (P1r-s TIE). Consistent picture: the trunk encodes *where* and *how fast things change*; metric geometry needs a learned readout.
* **Combined with IDOL `2605.31476`** (abstract-only, banked today): an inverse-dynamics step on adjacent **predicted** BEV latent states is reported as the bridge from prediction to planning on NAVSIM. That is a second, driving-domain, independent line for the same mechanism.

## 2 · Five-dimension analysis (Delta-JEPA + this probe)

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐⭐ | P-2 (action/scene ratio), LR14-5 (flow predictor, which reads the same ratio), backlog row 13 (additive motion latent), AI13-2. |
| **CONSEQUENCE** | Adds a **third candidate lever for P-2** next to LR14-5 (flow matching) and row 13, and the cheapest one: one auxiliary loss, no architecture change. |
| **COMBINATION** | Row 13's `z_t + m_t ≈ z_{t+1}` and LDAD are the same idea from two sides: row 13 makes displacement a *latent*, LDAD makes it *action-decodable*. ⭐ **Run them as one ablation pair.** |
| **CHANCES / RISKS** | **Upside:** cheap, attributable, and the frozen-trunk signal (F3) is already measured. **Risks:** (a) the action decoder could satisfy itself from the **scene** term (F4's lesson) — the loss must take **displacement only**, never `z_t`; (b) the lateral action is scene-dominated in our corpus, so LDAD may buy LON only; (c) an action-reconstruction loss uses ego as a **training label** only (admissible), and must never become an inference input. |
| **EXPERIMENT** | **`E-AI-LDAD-1` (proposed AI18-1, 2 v7-tiny arms + replicate):** add `L_LDAD = ‖D(ẑ_{t+1} − z_t) − a_t‖²` to the O5 predictor. **Committed:** success iff h=1 action/scene ratio rises **≥ 3×** at equal-or-better o5 loss **and** the replicate arm does not reproduce the rise. Controls: λ=0 arm (must equal baseline within the replicate floor), a `D([ẑ_{t+1}, z_t])` concat arm (Delta-JEPA's own ablation), constant predictor. **And the 0-GPU pre-step AI18-2:** re-run this probe with a **scene-residualised** lateral target (regress LAT on `latent_start_only`, probe the residual) so the lateral read gets a control that can pass. |

## 3 · Stopping condition (Rule Zero)

**(3a) for LON, (3b) for LAT.** LON cleared its committed bar. LAT's control failed, and the failure is **located** (F4): the next lever, AI18-2, is 0 GPU and specified, and is **not** run here because it would be a post-hoc repair of a pinned SPEC. AI18-1 needs **v7-tiny GPU arms**. The 4060 was occupied all session (two `python.exe` compute contexts), which is a named blocker.

`Deliverables: SPEC.md · RESULT.md · code/ldad0_probe.py · raw/{prereg_pin.json, ldad0.json, ldad0.log, search_log.md}`
