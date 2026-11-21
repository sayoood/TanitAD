import Mathlib

/-!
# T1 and T2 of the v8 theory in Lean 4: when a hierarchy dominates, and when it cannot

* T1 (multi-level prediction): the composed one-step bound, its attainment, the bound of a level that
  jumps `k` steps, and the exact criterion under which the jump bound is below the composed bound.
  The solver version checked instances (VC-A1..A5); the induction on the step count was a paper step.
* T2 (a decision before the trajectory): the exact risk split in an inner-product space, optimality of
  the mean under squared error (a selector that reads only the context cannot beat it), and channel (b):
  under a hit-or-miss criterion with separated modes every single plan hits at most one mode, while a
  decided plan hits its mode. The solver version checked VC-B1..B4 on the real line.

Theorems about model classes. Nothing here is a measurement and nothing is a statement about driving
performance. Every premise (step error, jump error, mode separation, selection accuracy) is a quantity
that has to be measured before a conclusion is drawn for a model.
-/

namespace V8Proofs.Hier

open scoped RealInnerProductSpace

/-! ## T1. Multi-level prediction against composed one-step prediction -/

/-- Geometric sums unroll from the front. -/
theorem geom_succ (L : ℝ) (n : ℕ) :
    ∑ i ∈ Finset.range (n + 1), L ^ i = 1 + L * ∑ i ∈ Finset.range n, L ^ i := by
  rw [Finset.sum_range_succ', Finset.mul_sum]
  simp only [pow_succ', pow_zero]
  ring

/-- T1, composed bound: an error that grows by at most the factor `L` per step and gains at most `ε`
per step is at most `ε * (1 + L + ... + L^(H-1))` after `H` steps (`= ε * (L^H - 1)/(L - 1)` for
`L ≠ 1`, `= ε * H` for `L = 1`). The induction on the step count, machine-checked. -/
theorem compose_bound (L ε : ℝ) (hL : 0 ≤ L) (e : ℕ → ℝ) (h0 : e 0 ≤ 0)
    (hstep : ∀ t, e (t + 1) ≤ L * e t + ε) (H : ℕ) :
    e H ≤ ε * ∑ i ∈ Finset.range H, L ^ i := by
  induction H with
  | zero => simpa using h0
  | succ n ih =>
    have h1 := hstep n
    have h2 : L * e n ≤ L * (ε * ∑ i ∈ Finset.range n, L ^ i) :=
      mul_le_mul_of_nonneg_left ih hL
    rw [geom_succ]
    nlinarith [h1, h2]

/-- T1, the composed bound is attained: the bound itself satisfies the recursion with equality. -/
theorem compose_bound_attained (L ε : ℝ) (t : ℕ) :
    ε * ∑ i ∈ Finset.range (t + 1), L ^ i = L * (ε * ∑ i ∈ Finset.range t, L ^ i) + ε := by
  rw [geom_succ]
  ring

/-- T1, jump bound: a level that predicts `k` steps at once with error `εk` has, after `m` jumps
(horizon `m * k`), error at most `εk * (1 + L^k + ... + (L^k)^(m-1))`. -/
theorem jump_bound (L εk : ℝ) (hL : 0 ≤ L) (k : ℕ) (f : ℕ → ℝ) (h0 : f 0 ≤ 0)
    (hstep : ∀ j, f (j + 1) ≤ L ^ k * f j + εk) (m : ℕ) :
    f m ≤ εk * ∑ j ∈ Finset.range m, (L ^ k) ^ j :=
  compose_bound (L ^ k) εk (pow_nonneg hL k) f h0 hstep m

/-- The geometric sum over `m * k` steps factors into the sum over `m` jumps times the sum over one jump. -/
theorem geom_factor (L : ℝ) (k m : ℕ) :
    ∑ i ∈ Finset.range (m * k), L ^ i
      = (∑ j ∈ Finset.range m, (L ^ k) ^ j) * ∑ i ∈ Finset.range k, L ^ i := by
  induction m with
  | zero => simp
  | succ m ih =>
    rw [add_one_mul, Finset.sum_range_add, ih, Finset.sum_range_succ, add_mul]
    congr 1
    rw [Finset.mul_sum]
    apply Finset.sum_congr rfl
    intro i _
    rw [pow_add, ← pow_mul, mul_comm m k]

theorem jump_sum_pos (L : ℝ) (hL : 0 ≤ L) (k m : ℕ) (hm : 0 < m) :
    0 < ∑ j ∈ Finset.range m, (L ^ k) ^ j := by
  obtain ⟨n, rfl⟩ : ∃ n, m = n + 1 := ⟨m - 1, by omega⟩
  rw [Finset.sum_range_succ']
  have h : 0 ≤ ∑ j ∈ Finset.range n, (L ^ k) ^ (j + 1) :=
    Finset.sum_nonneg (fun j _ => pow_nonneg (pow_nonneg hL k) (j + 1))
  simp only [pow_zero]
  linarith

/-- T1, the criterion: at every horizon `m * k` (`m ≥ 1`) the jump bound is below the composed
one-step bound exactly when the jump error is below the composed bound of ONE jump:
`εk < ε * (1 + L + ... + L^(k-1))`. The horizon drops out. -/
theorem jump_beats_compose_iff (L ε εk : ℝ) (hL : 0 ≤ L) (k m : ℕ) (hm : 0 < m) :
    εk * ∑ j ∈ Finset.range m, (L ^ k) ^ j < ε * ∑ i ∈ Finset.range (m * k), L ^ i
      ↔ εk < ε * ∑ i ∈ Finset.range k, L ^ i := by
  have hS := jump_sum_pos L hL k m hm
  rw [geom_factor]
  have e : ε * ((∑ j ∈ Finset.range m, (L ^ k) ^ j) * ∑ i ∈ Finset.range k, L ^ i)
      = (ε * ∑ i ∈ Finset.range k, L ^ i) * ∑ j ∈ Finset.range m, (L ^ k) ^ j := by ring
  rw [e]
  exact ⟨fun h => lt_of_mul_lt_mul_right h hS.le, fun h => mul_lt_mul_of_pos_right h hS⟩

/-- T1 for `L = 1` (no expansion): the jump level wins exactly when `εk < k * ε`. -/
theorem jump_beats_compose_iff_one (ε εk : ℝ) (k m : ℕ) (hm : 0 < m) :
    εk * (m : ℝ) < ε * ((m : ℝ) * (k : ℝ)) ↔ εk < (k : ℝ) * ε := by
  have hm' : (0 : ℝ) < (m : ℝ) := by exact_mod_cast hm
  have e : ε * ((m : ℝ) * (k : ℝ)) = ((k : ℝ) * ε) * (m : ℝ) := by ring
  rw [e]
  exact ⟨fun h => lt_of_mul_lt_mul_right h hm'.le, fun h => mul_lt_mul_of_pos_right h hm'⟩

/-! ## T2. A decision before the trajectory -/

section Split

variable {ι V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

/-- T2, exact risk split for one group: with `μ` the weighted mean of the outcomes (stated without a
division: the weighted deviations from `μ` sum to zero), the weighted squared error of ANY plan `c` is
the scatter around `μ` plus the total weight times the squared distance from `μ` to `c`. -/
theorem risk_split_one (s : Finset ι) (w : ι → ℝ) (y : ι → V) (μ c : V)
    (hμ : ∑ i ∈ s, w i • (y i - μ) = 0) :
    ∑ i ∈ s, w i * ‖y i - c‖ ^ 2
      = ∑ i ∈ s, w i * ‖y i - μ‖ ^ 2 + (∑ i ∈ s, w i) * ‖μ - c‖ ^ 2 := by
  have key : ∀ i, w i * ‖y i - c‖ ^ 2
      = w i * ‖y i - μ‖ ^ 2 + 2 * ⟪w i • (y i - μ), μ - c⟫ + w i * ‖μ - c‖ ^ 2 := by
    intro i
    have e : y i - c = (y i - μ) + (μ - c) := by abel
    rw [e, norm_add_sq_real, real_inner_smul_left]
    ring
  have hcross : ∑ i ∈ s, ⟪w i • (y i - μ), μ - c⟫ = 0 := by
    rw [← sum_inner, hμ, inner_zero_left]
  have h1 : ∑ i ∈ s, w i * ‖y i - c‖ ^ 2
      = ∑ i ∈ s, (w i * ‖y i - μ‖ ^ 2 + 2 * ⟪w i • (y i - μ), μ - c⟫ + w i * ‖μ - c‖ ^ 2) :=
    Finset.sum_congr rfl (fun i _ => key i)
  rw [h1, Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, ← Finset.sum_mul, hcross]
  ring

/-- T2, non-dominance under squared error: with non-negative weights no single plan beats the mean. So a
selector that reads only what the flat model reads cannot gain under squared error. -/
theorem mean_is_optimal (s : Finset ι) (w : ι → ℝ) (hw : ∀ i ∈ s, 0 ≤ w i) (y : ι → V) (μ c : V)
    (hμ : ∑ i ∈ s, w i • (y i - μ) = 0) :
    ∑ i ∈ s, w i * ‖y i - μ‖ ^ 2 ≤ ∑ i ∈ s, w i * ‖y i - c‖ ^ 2 := by
  rw [risk_split_one s w y μ c hμ]
  exact le_add_of_nonneg_right (mul_nonneg (Finset.sum_nonneg hw) (sq_nonneg _))

/-- T2, exact risk split over groups (modes): for a plan `c` that is constant on each group and `μ γ`
the weighted mean of group `γ`, total error = within-group scatter + between term. -/
theorem risk_split {κ : Type*} [DecidableEq κ] (s : Finset ι) (g : ι → κ) (w : ι → ℝ) (y : ι → V)
    (μ c : κ → V) (hμ : ∀ γ, ∑ i ∈ s.filter (fun i => g i = γ), w i • (y i - μ γ) = 0) :
    ∑ i ∈ s, w i * ‖y i - c (g i)‖ ^ 2
      = ∑ i ∈ s, w i * ‖y i - μ (g i)‖ ^ 2 + ∑ i ∈ s, w i * ‖μ (g i) - c (g i)‖ ^ 2 := by
  have hmaps : ∀ i ∈ s, g i ∈ s.image g := fun i hi => Finset.mem_image_of_mem g hi
  rw [← Finset.sum_fiberwise_of_maps_to hmaps (fun i => w i * ‖y i - c (g i)‖ ^ 2),
    ← Finset.sum_fiberwise_of_maps_to hmaps (fun i => w i * ‖y i - μ (g i)‖ ^ 2),
    ← Finset.sum_fiberwise_of_maps_to hmaps (fun i => w i * ‖μ (g i) - c (g i)‖ ^ 2),
    ← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro γ _
  have hg : ∀ i ∈ s.filter (fun i => g i = γ), g i = γ := fun i hi => (Finset.mem_filter.mp hi).2
  have h1 : ∑ i ∈ s.filter (fun i => g i = γ), w i * ‖y i - c (g i)‖ ^ 2
      = ∑ i ∈ s.filter (fun i => g i = γ), w i * ‖y i - c γ‖ ^ 2 :=
    Finset.sum_congr rfl (fun i hi => by rw [hg i hi])
  have h2 : ∑ i ∈ s.filter (fun i => g i = γ), w i * ‖y i - μ (g i)‖ ^ 2
      = ∑ i ∈ s.filter (fun i => g i = γ), w i * ‖y i - μ γ‖ ^ 2 :=
    Finset.sum_congr rfl (fun i hi => by rw [hg i hi])
  have h3 : ∑ i ∈ s.filter (fun i => g i = γ), w i * ‖μ (g i) - c (g i)‖ ^ 2
      = (∑ i ∈ s.filter (fun i => g i = γ), w i) * ‖μ γ - c γ‖ ^ 2 := by
    rw [Finset.sum_mul]
    exact Finset.sum_congr rfl (fun i hi => by rw [hg i hi])
  rw [h1, h2, h3]
  exact risk_split_one _ w y (μ γ) (c γ) (hμ γ)

end Split

/-! ## T2, channel (b): an acceptance region that is not convex -/

section Modes

variable {V : Type*} [NormedAddCommGroup V]

section Lin

variable [NormedSpace ℝ V]

/-- The averaged plan `(1 - p) • a + p • b` misses every outcome within `r` of the mode centre `a`
when `p * ‖a - b‖ > tol + r`. -/
theorem mean_plan_misses_mode_a (a b y : V) (p r tol : ℝ) (hp : 0 ≤ p) (hy : ‖y - a‖ ≤ r)
    (h : tol + r < p * ‖a - b‖) : tol < ‖y - ((1 - p) • a + p • b)‖ := by
  have e : a - ((1 - p) • a + p • b) = p • (a - b) := by
    rw [sub_smul, one_smul, smul_sub]
    abel
  have h1 : ‖a - ((1 - p) • a + p • b)‖ = p * ‖a - b‖ := by
    rw [e, norm_smul, Real.norm_eq_abs, abs_of_nonneg hp]
  have h2 := dist_triangle a y ((1 - p) • a + p • b)
  rw [dist_eq_norm, dist_eq_norm, dist_eq_norm, norm_sub_rev a y] at h2
  linarith

/-- The same for the other mode: missed when `(1 - p) * ‖a - b‖ > tol + r`. -/
theorem mean_plan_misses_mode_b (a b y : V) (p r tol : ℝ) (hp : p ≤ 1) (hy : ‖y - b‖ ≤ r)
    (h : tol + r < (1 - p) * ‖a - b‖) : tol < ‖y - ((1 - p) • a + p • b)‖ := by
  have h' := mean_plan_misses_mode_a b a y (1 - p) r tol (by linarith) hy
    (by rwa [norm_sub_rev b a])
  have e : (1 - (1 - p)) • b + (1 - p) • a = (1 - p) • a + p • b := by
    rw [sub_sub_cancel, add_comm]
  rwa [e] at h'

/-- VC-B3 and VC-B4 in any normed space: when both conditions hold the averaged plan misses every
outcome of both modes, while the decided plan (the centre of the outcome's mode) hits whenever
`r ≤ tol`. -/
theorem decided_hits_mean_misses (a b : V) (p r tol : ℝ) (hp0 : 0 ≤ p) (hp1 : p ≤ 1) (hr : r ≤ tol)
    (ha : tol + r < p * ‖a - b‖) (hb : tol + r < (1 - p) * ‖a - b‖) :
    (∀ y, ‖y - a‖ ≤ r → ‖y - a‖ ≤ tol ∧ tol < ‖y - ((1 - p) • a + p • b)‖) ∧
    (∀ y, ‖y - b‖ ≤ r → ‖y - b‖ ≤ tol ∧ tol < ‖y - ((1 - p) • a + p • b)‖) :=
  ⟨fun y hy => ⟨le_trans hy hr, mean_plan_misses_mode_a a b y p r tol hp0 hy ha⟩,
   fun y hy => ⟨le_trans hy hr, mean_plan_misses_mode_b a b y p r tol hp1 hy hb⟩⟩

end Lin

/-- No single plan hits an outcome of each mode when the modes are further apart than
`2 * tol + 2 * r`: a plan that hits both forces `‖a - b‖ ≤ 2 * tol + 2 * r`. -/
theorem single_plan_cannot_hit_both (a b c y y' : V) (r tol : ℝ) (hya : ‖y - a‖ ≤ r)
    (hyc : ‖y - c‖ ≤ tol) (hyb : ‖y' - b‖ ≤ r) (hyc' : ‖y' - c‖ ≤ tol) :
    ‖a - b‖ ≤ 2 * tol + 2 * r := by
  have h1 := dist_triangle a y c
  have h2 := dist_triangle a c b
  have h3 := dist_triangle c y' b
  rw [dist_eq_norm, dist_eq_norm, dist_eq_norm, norm_sub_rev a y] at h1
  rw [dist_eq_norm, dist_eq_norm, dist_eq_norm] at h2
  rw [dist_eq_norm, dist_eq_norm, dist_eq_norm, norm_sub_rev c y'] at h3
  linarith

variable {ι : Type*}

/-- Under separated modes the hits of ANY single plan `c` all lie in one mode. `mode i` says which
mode outcome `i` belongs to (`true`: centre `a`; `false`: centre `b`). -/
theorem flat_hits_one_mode (s : Finset ι) (mode : ι → Bool) (y : ι → V) (a b c : V) (r tol : ℝ)
    (hmode : ∀ i ∈ s, ‖y i - (if mode i = true then a else b)‖ ≤ r)
    (hD : 2 * tol + 2 * r < ‖a - b‖) :
    (∀ i ∈ s, ‖y i - c‖ ≤ tol → mode i = true) ∨ (∀ i ∈ s, ‖y i - c‖ ≤ tol → mode i = false) := by
  by_contra hcon
  rw [not_or] at hcon
  obtain ⟨h1, h2⟩ := hcon
  push Not at h1 h2
  obtain ⟨i, hi, hic, him⟩ := h1
  obtain ⟨j, hj, hjc, hjm⟩ := h2
  have him' : mode i = false := by simpa using him
  have hjm' : mode j = true := by simpa using hjm
  have hi' := hmode i hi
  have hj' := hmode j hj
  rw [him'] at hi'
  rw [hjm'] at hj'
  simp at hi' hj'
  have h := single_plan_cannot_hit_both a b c (y j) (y i) r tol hj' hjc hi' hic
  linarith

open scoped Classical in
/-- Count form: the number of outcomes a single plan hits is at most the size of the larger mode. -/
theorem flat_hit_count_le (s : Finset ι) (mode : ι → Bool) (y : ι → V) (a b c : V) (r tol : ℝ)
    (hmode : ∀ i ∈ s, ‖y i - (if mode i = true then a else b)‖ ≤ r)
    (hD : 2 * tol + 2 * r < ‖a - b‖) :
    (s.filter (fun i => ‖y i - c‖ ≤ tol)).card
      ≤ max (s.filter (fun i => mode i = true)).card (s.filter (fun i => mode i = false)).card := by
  rcases flat_hits_one_mode s mode y a b c r tol hmode hD with h | h
  · refine le_max_of_le_left (Finset.card_le_card ?_)
    intro i hi
    rw [Finset.mem_filter] at hi ⊢
    exact ⟨hi.1, h i hi.1 hi.2⟩
  · refine le_max_of_le_right (Finset.card_le_card ?_)
    intro i hi
    rw [Finset.mem_filter] at hi ⊢
    exact ⟨hi.1, h i hi.1 hi.2⟩

open scoped Classical in
/-- A decided plan (the centre of the SELECTED mode `sel i`) hits every outcome whose mode was selected
correctly, when `r ≤ tol`. -/
theorem decided_hit_count_ge (s : Finset ι) (mode sel : ι → Bool) (y : ι → V) (a b : V) (r tol : ℝ)
    (hmode : ∀ i ∈ s, ‖y i - (if mode i = true then a else b)‖ ≤ r) (hr : r ≤ tol) :
    (s.filter (fun i => sel i = mode i)).card
      ≤ (s.filter (fun i => ‖y i - (if sel i = true then a else b)‖ ≤ tol)).card := by
  refine Finset.card_le_card ?_
  intro i hi
  rw [Finset.mem_filter] at hi ⊢
  refine ⟨hi.1, ?_⟩
  rw [hi.2]
  exact le_trans (hmode i hi.1) hr

open scoped Classical in
/-- T2, dominance through channel (b). Outcomes of one context fall into two modes further apart than
`2 * tol + 2 * r`, each outcome within `r ≤ tol` of its mode centre. If the selector is right on more
outcomes than the larger mode holds, the decided plan hits strictly more outcomes than EVERY single
plan. If it is not, this channel gives nothing: the premise is the selection accuracy against the
share of the larger mode, a quantity to measure. -/
theorem hierarchy_dominates_flat (s : Finset ι) (mode sel : ι → Bool) (y : ι → V) (a b : V)
    (r tol : ℝ) (hmode : ∀ i ∈ s, ‖y i - (if mode i = true then a else b)‖ ≤ r) (hr : r ≤ tol)
    (hD : 2 * tol + 2 * r < ‖a - b‖)
    (hsel : max (s.filter (fun i => mode i = true)).card (s.filter (fun i => mode i = false)).card
      < (s.filter (fun i => sel i = mode i)).card) (c : V) :
    (s.filter (fun i => ‖y i - c‖ ≤ tol)).card
      < (s.filter (fun i => ‖y i - (if sel i = true then a else b)‖ ≤ tol)).card :=
  lt_of_le_of_lt (flat_hit_count_le s mode y a b c r tol hmode hD)
    (lt_of_lt_of_le hsel (decided_hit_count_ge s mode sel y a b r tol hmode hr))

end Modes

end V8Proofs.Hier
