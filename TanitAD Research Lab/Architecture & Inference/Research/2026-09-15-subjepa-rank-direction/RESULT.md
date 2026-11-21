<title>Sub-JEPA read (debt D-9) — we ran it as a rank-RAISER; its authors credit rank REDUCTION</title>

# Sub-JEPA `2605.09241`, full text: our sub32c/sub64c arms tested a mechanism the paper does not claim, and two September primaries disagree on which way rank should move

**2026-09-15 · Research Lab (LAB-RUN-013) · Architecture & Inference · discharges debt D-9's surviving half (pre-committed rotation item 3) · touches H-RANK-5 / H-RANK-15 / D-V7F-NONCOLLAPSE / row 9**
⛔ **Tier: none.** Literature + register reconciliation; no training.

---

## 0 · Findings first

| # | finding | class |
|---|---|---|
| **F1** | ⭐⭐⭐ **Sub-JEPA credits its gains to REDUCING effective rank, not raising it.** *"the effective-rank reduction from LeWM to Sub-JEPA is strongly and directly correlated with planning improvement across all four environments"*. The mechanism claim: a full-space isotropic Gaussian prior *"forces the latent space into unnecessarily high rank"*. The rank used is Roy–Vetterli entropy over **covariance eigenvalues** (σ², Eq. 9) on N = 2,000 frozen-encoder embeddings. | PUBLISHED lib `2605.09241` v1 (full text) |
| **F2** | ⛔⭐⭐ **Our register ran Sub-JEPA for the opposite purpose.** H-RANK-5: *"Estimator CONDITIONING is a lever (… dims d via Sub-JEPA)"* → REFUTED because **sub32c / sub64c VAL participation 3.33 / 3.56 ≈ lewm 3.50**. That refutes *our* hypothesis (Sub-JEPA raises our participation). **It says nothing about Sub-JEPA's own claim**, which predicts rank goes *down* and quality goes *up*. Whether sub32c / sub64c were better on *quality* than lewm is **not in the register** (searched: `GOALS_AND_CLAIMS.md`, 5 rows mention them, all rank-only). | INHERITED register rows + this read |
| **F3** | ⚠️ **The paper's "consistently … very clear margins" does not survive its own table on 2 of 4 tasks.** Six seeds, mean ± std: Two-Room **95.00 ± 2.76 vs 84.33 ± 4.23** ✅; OGB-Cube **76.33 ± 5.99 vs 67.33 ± 5.01** ✅; Reacher **84.00 ± 4.00 vs 82.67 ± 4.42** (inside one std); PushT **89.00 ± 5.33 vs 84.67 ± 6.53** (inside one std). K is chosen per environment *"on a held-out validation set"*, and **PushT collapses to 28.00 ± 5.22 at the K = 32 used elsewhere**. The rank-quality "correlation" rests on **n = 4 environments**. | PUBLISHED Tables 1–2 |
| **F4** | ⭐⭐ **A second September primary points the other way on a neighbouring quantity.** `2609.03565` (IDM + state alignment): LeWM's higher temporal straightening *"is associated with a substantially lower effective transition dimension"* (**r₉₅ 17 vs 32.8**), and the higher-dimension model plans better (**TwoRoom 100 vs 87 %, OGB-Cube 87 vs 74 %**, 3 seeds, no std). ⚠️ This is **transition-subspace** dimension, not state-embedding rank. So the two papers do not contradict on one quantity; they disagree on whether "lower-dimensional latent" is the virtue. | PUBLISHED lib `2609.03565` v1 (HTML full text) |
| **F5** | ⭐ **One concession worth carrying.** Sub-JEPA's frozen projections beat trainable ones because *"learned projections can align with directions that reduce the effective strength of the regularizer"*. Random un-orthogonalised frozen projections read **13.33 % on PushT**. **The anti-collapse term is only as good as its immunity to encoder co-adaptation.** That is the same failure shape as our O1 margin line (a learned margin gamed by the representation). | PUBLISHED Table 3 |

## 1 · What this means for our gate (stated narrowly, because the evidence is thin)

* Our collapse criterion `D-V7F-NONCOLLAPSE` is **participation (σ²), VAL-side, ≥ 8.56**, and H-REFAV1-COLLAPSE-2b already says it is **necessary, not sufficient**. **F1 does not attack the floor as a collapse check.**
* ⛔ **What F1 does attack is any reading of participation as "higher is better" above the floor**, including ranking arms by participation or treating a participation rise as progress. H-RANK-12 already refused that as progress; F1 is the first external primary that predicts the sign could be *negative*. F4 predicts positive on a different quantity. **n = 4 and n = 4 environments, toy control, not driving.** ⇒ **Rank above the floor is unordered for us until measured on our own arms.**
* ⚠️ **Scope:** Roy–Vetterli entropy (Sub-JEPA) and participation ratio (ours) are both σ²-spectrum statistics but are not the same number. H-RANK-15 already showed one arm can be worst on one rank statistic and best on another. Quantity-matching is part of the experiment below.

## 2 · Five-dimension analysis

| dim | |
|---|---|
| **RELEVANCE** ⭐⭐ | The v7 rank gate; row 9 (anti-collapse design space); the trunk-freeze (AI13-2); debt D-9. |
| **CONSEQUENCE** | H-RANK-5's REFUTED row needs a scope line: *"refutes Sub-JEPA as a participation-raiser; Sub-JEPA's own rank-down / quality-up claim is UNTESTED on our arms."* No gate threshold moves. |
| **COMBINATION** | ⭐ **Three independent lines now say rank and quality can run opposite:** IWM / NextLat (row 9: *"rank down, quality up"*), our own H-REFAV1-COLLAPSE-2b (flagship v1 had the highest rank ever measured and no environment interpretation), and Sub-JEPA F1. Against them: `2609.03565` F4 (on transition dimension). ⚠️ **And under our frozen trunk none of it is a lever**: Sub-JEPA regularises the *encoder*, so it is the eighth encoder-side mechanism on the AI13-2 list. What survives the freeze is the *diagnostic* reading, not the regulariser. |
| **CHANCES / RISKS** | **Upside:** the sub32c / sub64c checkpoints may already hold the test (quality vs lewm) at 0 GPU. **Risks:** toy tasks with low intrinsic dimension (Two-Room carries most of the effect); per-environment K tuning; n = 4. |
| **EXPERIMENT** | **`E-AI-RANKDIR-1` (proposed AI15-1, 0 GPU):** over every v7-tiny / v7f arm holding **both** a VAL-side participation read **and** a quality read from the same checkpoint (o5 loss, T0 fwd_ade, and the hold-action margin), compute Spearman ρ(participation, quality) **restricted to arms at or above the 8.56 floor**, plus the same with Roy–Vetterli entropy (quantity-matched to Sub-JEPA). **Controls:** a shuffled arm-label permutation (must read ρ ≈ 0); report n arms. **Committed:** ρ ≤ −0.3 on both statistics ⇒ participation is declared **collapse-only** and arms are never ranked by it; \|ρ\| < 0.3 ⇒ uninformative above the floor, same ruling; ρ ≥ +0.3 ⇒ Sub-JEPA's direction does not transfer and the current reading stands. **Fewer than 6 arms ⇒ no verdict, say so.** |

## 3 · What this changes (≤3)

1. ⭐⭐ **Add a scope line to H-RANK-5** (F2). The sub32c / sub64c result is not evidence about Sub-JEPA's mechanism.
2. ⭐ **Rank above the 8.56 floor is unordered until `E-AI-RANKDIR-1` reads.** No arm is preferred for higher participation.
3. ⭐ **Record Sub-JEPA on AI13-2's encoder-side list (eighth), and F5's co-adaptation concession on row 33** (O1 margin rescue): a learned margin shares the failure its frozen-projection ablation measured.

## 4 · Manifest

| artifact | location |
|---|---|
| RESULT | `repo:TanitAD Research Lab/Architecture & Inference/Research/2026-09-15-subjepa-rank-direction/RESULT.md` |
| Search log | `repo:…/raw/search_log.md` |
| Primaries | lib `2605.09241` (full text via local extraction, cited-by updated), `2609.03565` (banked today) |
