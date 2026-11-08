# MATH_AUDIT: living paper v1.6 (2026-11-08)

Method: each equation and worked number re-derived by hand or in Python from its stated factors; primary artifacts opened only where a measured number was in doubt. Labels: ERROR (wrong as written, fixed in place), OVERSTATED (claim outruns its argument or cited condition, wording narrowed), OK-CHECKED (re-derived, holds). No measured number was changed. Edit numbers (A..) refer to `EDIT_LOG.md`.

## ERROR
| # | where | finding | action |
|---|---|---|---|
| E1 | §3.12 and §7.16 | "Selection degrades as the fan deepens (frozen trunk K = 8/32/64 → 0.5772/0.5173/0.5319 ...) ... the direction agrees". The primary `w7_gate_k{8,32,64}.json` (`w7_selected_ade`) reads 0.577245 / 0.517281 / 0.531908: the series improves 8 to 32, worsens 32 to 64, and ends 0.0453 m better at 64 than at 8. It does not show degradation, and the "direction agrees" sentence is false. Also "the predicted monotone divergence ... is what we observe" has no single-trunk sweep behind it. | Reworded in §3.12 (A, Model A paragraph and Model B paragraph) and §7.16 (heading, "direction agrees", "measured degradation"); the Model A refutation now rests on fixed-N tail evidence (mean error-rank 132.3 of 256, flat low-cost stratum), which is unchanged. |
| E2 | §11.6 | "300 samples × 30 iterations would cost 325 s per MPC tick" at the measured 160 ms per candidate. 300 × 30 × 0.160 s = 1,440 s. 325 s does not reproduce (it implies about 2,031 candidates). Derived figure, not a measurement; no primary artifact found. | Replaced by the 1,440 s arithmetic with the 325 s flagged UNVERIFIED. The conclusion ("a batch job") holds a fortiori. The measured 21.5 s and 3.5 s are untouched. |
| E3 | §14.7 | "proportionally 10–100× above every published operating point" with 12.5 % against data2vec ≈ 1 % and BYOL ≈ 0.1 %: 12.5 and 125, not 10 to 100. | Reworded to about 12 to 125 times; the two reference fractions are PUBLISHED-derived and not re-derived here. |
| E4 | §3.11 (1) | "A strategic roll over 15 s is three steps" holds only at 0.2 Hz (5 s step); the table gives 0.2 to 0.5 Hz (seven to eight steps at 0.5 Hz). | Qualified (0.2 Hz end; about seven at 0.5 Hz; 30 s is six steps at 0.2 Hz). |
| E5 | §11.4 | E(K) = ε(λ^K − 1)/(λ − 1) is stated for λ ≥ 1 but is 0/0 at λ = 1. | Limit K·ε added. |

## OVERSTATED
| # | where | finding | action |
|---|---|---|---|
| O1 | §3.3 (i) | "Regret linear in T, so per-level short horizons pay a strictly smaller bound at each level". True per level, but linear bounds add back to the undivided bound at equal constants, so the factorisation alone buys no bound. The theorem (Cui et al. Thm 4.2) was not re-read against a banked primary (not found in `library.json`). | Condition stated; UNVERIFIED against primary. |
| O2 | §3.3 (ii) | Product-to-sum sample complexity needs the level components to combine independently. | Condition added. |
| O3 | §4 (1) | "Latent models win precisely when ..." (iff) and "pixel prediction is the degenerate k = n endpoint" from a bound stated for the rank-k factorisation. | Narrowed to "the bound favours"; pixel reading marked an analogy. |
| O4 | §4 (2) | 0.05 to 0.11 is the per-step frame-change fraction (as in §3.2, A8), not the action's share of the change; "dominant fraction" was not what was measured. | Wording names the measured quantity; the premise is pointed at §7.3's causal panel. |
| O5 | §3.12 Model B | P(i* in D_δ) → 1 needs the cost on D_δ to lie below the cost's infimum off D_δ; "both terms → 0 on D_δ" alone does not give it. | Condition stated as an assumption. |
| O6 | §3.12 generalisation | "Widening makes a degenerate cost worse" is a Model B statement; Model A says it helps. | Scoped to Model B. |
| O7 | §3.12 Model A | The "45 % of the gain" uses ρ of the Gaussian copula; a Spearman 0.45 maps to ρ = 2 sin(0.45π/6) = 0.467 (46.7 %). Small, but the text equated the two. Expected mean rank 128 is 128.5 for ranks 1..256. | Mapping and 128.5 stated. |
| O8 | §11.4 | "Strategic pays only K = 4 of compounding": assumes the same per-step ε at every Δt; a coarser step carries a larger ε. | Marked an upper bound on the saving. |
| O9 | §13.1 | "(u, η) is a non-collapsed optimum": a predictable scene-encoding η would lower O5 by removing its irreducible noise term, so (u, η) is a solution that satisfies every term, not shown to be the minimiser. §14.1 reference changed to match. | Narrowed. |
| O10 | §3.11 (2) | The isolation matrix G blocks the upper level's loss from the lower level's parameters. It says nothing about the converse edge (lower loss onto upper parameters), which the PI's rule needs at 0.0 and which `detach_upper` controls; its default is False (`v8f_mini.py:215`, read from the ladder export). The by-construction wording overstated what holds for v8F runs, M2b included. | Scope paragraph added. |

## OK-CHECKED
* §3.2 SIGReg statistic: the displayed T_{n,β} is the d = 1 Henze-Zirkler / Epps-Pulley form; Cramér-Wold step correct (standard-normal projections give the isotropic Gaussian).
* §3.12 Model A identities: E[e | c] = ρc for standardised bivariate normal, E[e_{i*}] = −ρ a_N, oracle −a_N, random 0, captured fraction exactly ρ, a_N ≍ √(2 ln N); finite-m range m/N = 2/256 = 0.0078 to 32/256 = 0.125.
* §3.13: PR in [1, D], PR = 1 for rank one, D for isotropic; retention 6.94 / 4.53 = 1.532.
* §3.14 (inserted): knots 30 × 0.2 s = 6 s, knots 1 to 10 = 2 s; strides 1, 3, 15 give steps 0.2, 0.6, 3.0 s; reach 10 × 0.2 = 2, 10 × 0.6 = 6, 6 × 3.0 = 18 s; F = 1 + max(10, 30, 90) = 91; T_3 = 4 gives 12 s, T_3 = 2 gives 6 s; V_l mask gives zero for steps without a target; the cost sums in Eqs. 4 to 6 match the review's reading of H-JEPA.
* §4 measured corroboration (R² 0.997, 99 % in about 22 dims, effective rank about 35) consistent with the abstract's 22 to 35; spectrum claims not re-fit here.
* §5.3 bootstrap definition; T_heldout = Σ_w (m_w/8h) e_w with Σ ω_w = 1; bias = |W|·Cov_w(ω_w, e_w); full-set point estimate.
* §11.3: 640 × 1024 × 4 = 2,621,440 (about 2.6e6) inputs to 2K = 40 outputs (K = 20); N·d·K = 640 × 1024 × 30 = 19,660,800 (1.97e7); about five orders above 2K.
* §11.4 table K = T/Δt: 6/0.2 = 30, 6/0.6 = 10, 6/1.5 = 4.
* §11.5(iii): J(u*) ≤ min over baselines holds trivially for the exact-J argmin over C ∪ B; §11.5(iv) states the limit.
* §11.6: 2553/16, 5129/32, 10519/64 ms = 159.6, 160.3, 164.4 ms per candidate (linear); 300 × 10 × 1.31 MB = 3.93 GB; 640 × 1024 × 2 B = 1.31 MB.
* §13.2(a) 41.3 / 0.088 = 469 (468 within rounding of 0.088); (d) 20.228 / 5.756 = 3.51.
* §14.7: unrolled EMA formula, memory horizon 1/(1 − 0.996) = 250; 250/2000 = 12.5 %, 250/30000 = 0.83 %; drift −11.7 / −19.6 %, cos −27.6 / −46.9 %, seed spreads 0.0358 / 0.0357 / 0.0042 all recompute.
* §7.22 numbers: each copied from `RESULT_PLANTEST_body.md`; four spot-checked against `RESULT_PLANTEST.json` (1.4227509 rounds to 1.4228; HOLD 0.3296; TRUE minus HOLD 1.0932; TD3 minus FLAT1 0.0302).
* Replicate false-positive rate (CLAUDE.md: 4 of 42 = 9.5 %): not quoted anywhere in the paper (grep for "of 42", "14.3 %", "false-positive rate" found nothing); nothing to fix. Line 3230 of the tip says §5 does not yet state H-ESTIM-SEED-1; left as it is.

## Not checked / UNVERIFIED
* Cui et al. Thm 4.2 and the O(T·√(Σσ² + C(k)/√n)) bound against the primary (not banked under a key I could find). The wording is narrowed; the theorem itself is unverified.
* §3.14's "O 10 Hz": the cache step is 0.2 s (5 Hz) and AUDIT_MODEL gives 60 points at 0.1 s from 30 knots; I left the stated 10 Hz. §3.11's scope sentence repeats it.
* §13.2(c): "0.02612 / 0.02614 ... identical to five significant figures" quotes two different values; reads as per head, not confirmed. The 10^-3 initialisation vs norm 0.026 implies about 680 entries per head; not checked.
* §13.2(b) (Δ = −0.1251, t = −2.72), §11.5(ii) lag-1 autocorrelation figures, C101 35.8 %, FROST-Drive/DINO-WM/BYOL/data2vec PUBLISHED figures: not re-derived.
* §14.2 to §14.6 and §14.8 to §14.10 numbers, §5.1/5.2 blast-radius percentages (CLAUDE.md quotes them identically), §12 and §13.3 to §13.19: outside the audit list or not reached within the context budget.
* §3.14 not edited (inserted verbatim); its HZ-style claims about H-JEPA were read against `REVIEW_HJEPA_CONSISTENCY.md`, not the PDF.
