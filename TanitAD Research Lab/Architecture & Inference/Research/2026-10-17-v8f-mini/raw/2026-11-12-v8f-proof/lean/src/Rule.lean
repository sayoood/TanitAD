import Mathlib
import V8proofs.Hier

/-!
# T3, T7 and T9 of the v8 theory in Lean 4: rules by construction, imagination, weight averaging

* T3 (rules): a stop-line shield keeps its invariant for EVERY sequence of proposals, over any number of
  steps (the solver version, VC-C1, checked one step); the fallback is always feasible; a safe proposal
  is passed unchanged; masked selection over any finite candidate set; the penalty bound; and the
  residual risk: a shield on a predicted line violates the true line at most as often as the predicted
  line lies beyond the true one.
* T7 (imagination): with imagined and true returns within `G`, an imagined gain above `2 G` is a real
  gain, the candidate with the best imagined value is within `2 G` of every other one, the `2 G` is
  attained, a gain above `G` alone is not enough, and `G` follows from the step error of the model
  through the composed bound of T1.
* T9 (weight space): the error of a member average equals the mean member error minus the mean member
  scatter, in any inner-product space, so the average is never worse than the member mean; and for a
  head that is linear in its weights a weight average is the output average.

Theorems about model classes. Nothing here is a measurement and nothing is a statement about driving
performance. Every premise (the braking law, the gap `G`, the envelope of a predicted line) is a
quantity that has to be measured before a conclusion is drawn for a model.
-/

namespace V8Proofs.Rule

/-! ## T3: a shield keeps a rule for every proposal -/

/-- A braking-distance function for the step length `dt` and the speed decrement `q` per step. -/
structure Brake (B : ℝ → ℝ) (dt q : ℝ) : Prop where
  dt_pos : 0 < dt
  q_pos : 0 < q
  zero : B 0 = 0
  nonneg : ∀ x, 0 ≤ x → 0 ≤ B x
  step : ∀ v, 0 < v → B v = v * dt + B (max (v - q) 0)

/-- The braking fallback: the speed after one step of full braking. -/
noncomputable def vb (q v : ℝ) : ℝ := max (v - q) 0

/-- The shielded speed for the proposal `u` in the state `(d, v)`: the proposal (never below the
braking fallback) if the braking distance from it still fits, else the braking fallback. -/
noncomputable def shield (B : ℝ → ℝ) (dt q d v u : ℝ) : ℝ :=
  if B (max u (vb q v)) ≤ d - v * dt then max u (vb q v) else vb q v

theorem vb_nonneg (q v : ℝ) : 0 ≤ vb q v := le_max_right _ _

/-- The fallback is always feasible: braking from a state that satisfies the invariant satisfies it
after the step. -/
theorem fallback_feasible {B : ℝ → ℝ} {dt q : ℝ} (hB : Brake B dt q) {d v : ℝ} (hv : 0 ≤ v)
    (hinv : B v ≤ d) : B (vb q v) ≤ d - v * dt := by
  rcases hv.eq_or_lt with h0 | hpos
  · subst h0
    have hq := hB.q_pos
    have e : vb q 0 = 0 := by
      unfold vb
      exact max_eq_right (by linarith)
    rw [hB.zero] at hinv
    rw [e, hB.zero]
    linarith
  · have e := hB.step v hpos
    unfold vb
    linarith

/-- One shielded step keeps the invariant for ANY proposal `u`, keeps the speed non-negative and
keeps the remaining distance non-negative. -/
theorem shield_step {B : ℝ → ℝ} {dt q : ℝ} (hB : Brake B dt q) {d v : ℝ} (hv : 0 ≤ v)
    (hinv : B v ≤ d) (u : ℝ) :
    B (shield B dt q d v u) ≤ d - v * dt ∧ 0 ≤ shield B dt q d v u ∧ 0 ≤ d - v * dt := by
  have hfb := fallback_feasible hB hv hinv
  have hvb := vb_nonneg q v
  have hd : 0 ≤ d - v * dt := le_trans (hB.nonneg _ hvb) hfb
  refine ⟨?_, ?_, hd⟩
  · unfold shield
    split_ifs with h
    · exact h
    · exact hfb
  · unfold shield
    split_ifs with h
    · exact le_trans hvb (le_max_right _ _)
    · exact hvb

/-- The state `(remaining distance, speed)` after `n` shielded steps under the proposals `u`. -/
noncomputable def run (B : ℝ → ℝ) (dt q : ℝ) (u : ℕ → ℝ) (d0 v0 : ℝ) : ℕ → ℝ × ℝ
  | 0 => (d0, v0)
  | n + 1 => ((run B dt q u d0 v0 n).1 - (run B dt q u d0 v0 n).2 * dt,
      shield B dt q (run B dt q u d0 v0 n).1 (run B dt q u d0 v0 n).2 (u n))

/-- The rule holds at every step, for every sequence of proposals: the braking distance fits into the
remaining distance, the speed is non-negative, and the line is never passed. -/
theorem shield_forever {B : ℝ → ℝ} {dt q : ℝ} (hB : Brake B dt q) {d0 v0 : ℝ} (hv : 0 ≤ v0)
    (hinv : B v0 ≤ d0) (u : ℕ → ℝ) (n : ℕ) :
    B (run B dt q u d0 v0 n).2 ≤ (run B dt q u d0 v0 n).1 ∧ 0 ≤ (run B dt q u d0 v0 n).2
      ∧ 0 ≤ (run B dt q u d0 v0 n).1 := by
  induction n with
  | zero =>
    simp only [run]
    exact ⟨hinv, hv, le_trans (hB.nonneg _ hv) hinv⟩
  | succ n ih =>
    obtain ⟨h1, h2, _⟩ := ih
    have h := shield_step hB h2 h1 (u n)
    simp only [run]
    exact ⟨h.1, h.2.1, h.2.2⟩

/-- Minimal intervention: a proposal that is not below the braking fallback and whose braking distance
fits is passed unchanged. -/
theorem shield_keeps_safe_proposal (B : ℝ → ℝ) (dt q d v u : ℝ) (hu : vb q v ≤ u)
    (hs : B u ≤ d - v * dt) : shield B dt q d v u = u := by
  unfold shield
  rw [max_eq_left hu, ite_eq_left hs]

/-- A penalty of weight `lam` only bounds the violation; it does not remove it. -/
theorem penalty_bound {J V lam J0 Jinf : ℝ} (hlam : 0 < lam) (h : J + lam * V ≤ J0)
    (hJ : Jinf ≤ J) : V ≤ (J0 - Jinf) / lam := by
  rw [le_div_iff₀ hlam]
  nlinarith [h, hJ]

/-- Masked selection over any finite candidate set: if anything is admissible, there is an admissible
candidate whose score is at least that of every admissible candidate. -/
theorem masked_argmax {ι : Type*} (s : Finset ι) (adm : ι → Prop) [DecidablePred adm]
    (score : ι → ℝ) (h : ∃ i ∈ s, adm i) :
    ∃ b ∈ s, adm b ∧ ∀ i ∈ s, adm i → score i ≤ score b := by
  obtain ⟨i0, hi0, ha0⟩ := h
  have hne : (s.filter adm).Nonempty := ⟨i0, Finset.mem_filter.mpr ⟨hi0, ha0⟩⟩
  obtain ⟨b, hb, hmax⟩ := (s.filter adm).exists_max_image score hne
  rw [Finset.mem_filter] at hb
  exact ⟨b, hb.1, hb.2, fun i hi ha => hmax i (Finset.mem_filter.mpr ⟨hi, ha⟩)⟩

/-- The residual risk is perception. If the shield never lets the car pass the predicted line
(`x i ≤ dhat i`), the true line is passed at most as often as the predicted line lies beyond it. -/
theorem violations_le_misses {ι : Type*} (s : Finset ι) (x dhat dtrue : ι → ℝ)
    (hshield : ∀ i ∈ s, x i ≤ dhat i) :
    (s.filter (fun i => dtrue i < x i)).card ≤ (s.filter (fun i => dtrue i < dhat i)).card := by
  apply Finset.card_le_card
  intro i hi
  rw [Finset.mem_filter] at hi ⊢
  exact ⟨hi.1, lt_of_lt_of_le hi.2 (hshield i hi.1)⟩

/-! ## T7: when an imagined gain is a real gain -/

/-- If imagined and true values differ by at most `G`, an imagined gain above `2 G` is a real gain. -/
theorem imagined_gain_is_real {ι : Type*} (J I : ι → ℝ) (G : ℝ) (h : ∀ i, |J i - I i| ≤ G)
    (a b : ι) (hg : 2 * G < I a - I b) : J b < J a := by
  have ha := abs_le.mp (h a)
  have hb := abs_le.mp (h b)
  linarith [ha.1, ha.2, hb.1, hb.2]

/-- Choosing by imagined cost loses at most `2 G` against every other candidate. -/
theorem imagined_choice_regret {ι : Type*} (J I : ι → ℝ) (G : ℝ) (h : ∀ i, |J i - I i| ≤ G)
    (a : ι) (hmin : ∀ i, I a ≤ I i) (b : ι) : J a ≤ J b + 2 * G := by
  have ha := abs_le.mp (h a)
  have hb := abs_le.mp (h b)
  have hab := hmin b
  linarith [ha.2, hb.1]

/-- On any finite non-empty candidate set there is a choice (the best imagined one) within `2 G` of
every candidate in true cost. -/
theorem exists_choice_within_two_gaps {ι : Type*} (s : Finset ι) (hs : s.Nonempty) (J I : ι → ℝ)
    (G : ℝ) (h : ∀ i ∈ s, |J i - I i| ≤ G) : ∃ a ∈ s, ∀ b ∈ s, J a ≤ J b + 2 * G := by
  obtain ⟨a, ha, hmin⟩ := s.exists_min_image I hs
  refine ⟨a, ha, fun b hb => ?_⟩
  have h1 := abs_le.mp (h a ha)
  have h2 := abs_le.mp (h b hb)
  have h3 := hmin b hb
  linarith [h1.2, h2.1]

/-- The `2 G` is attained: two candidates with equal imagined cost whose true costs differ by `2 G`. -/
theorem two_gaps_attained (G : ℝ) (hG : 0 ≤ G) :
    ∃ (J I : Bool → ℝ), (∀ i, |J i - I i| ≤ G) ∧ (∀ i, I true ≤ I i)
      ∧ J true = J false + 2 * G := by
  refine ⟨fun i => cond i G (-G), fun _ => 0, ?_, ?_, ?_⟩
  · intro i
    cases i
    · show |(-G) - 0| ≤ G
      exact le_of_eq (by rw [sub_zero, abs_neg, abs_of_nonneg hG])
    · show |G - 0| ≤ G
      exact le_of_eq (by rw [sub_zero, abs_of_nonneg hG])
  · intro i
    exact le_refl _
  · show G = -G + 2 * G
    ring

/-- The stronger twin is false: an imagined gain above `G` (and below `2 G`) can be a real loss. -/
theorem one_gap_not_enough (G : ℝ) (hG : 0 < G) :
    ∃ (J I : Bool → ℝ), (∀ i, |J i - I i| ≤ G) ∧ G < I true - I false ∧ J true < J false := by
  refine ⟨fun i => cond i (G / 2) G, fun i => cond i (3 * G / 2) 0, ?_, ?_, ?_⟩
  · intro i
    cases i
    · show |G - 0| ≤ G
      exact le_of_eq (by rw [sub_zero, abs_of_pos hG])
    · show |G / 2 - 3 * G / 2| ≤ G
      have e : G / 2 - 3 * G / 2 = -G := by ring
      exact le_of_eq (by rw [e, abs_neg, abs_of_pos hG])
  · show G < 3 * G / 2 - 0
    linarith
  · show G / 2 < G
    linarith

/-- The gap of a summed cost is at most `K` times the summed state error, when each step cost moves
by at most `K` times the state error of that step. -/
theorem return_gap_le (H : ℕ) (c chat e : ℕ → ℝ) (K : ℝ)
    (h : ∀ t ∈ Finset.range H, |chat t - c t| ≤ K * e t) :
    |∑ t ∈ Finset.range H, chat t - ∑ t ∈ Finset.range H, c t|
      ≤ K * ∑ t ∈ Finset.range H, e t := by
  rw [← Finset.sum_sub_distrib, Finset.mul_sum]
  exact le_trans (Finset.abs_sum_le_sum_abs _ _) (Finset.sum_le_sum h)

/-- `G` from the step error of the model: with a one-step error `ε`, an expansion factor `L` and a
step cost that is `K`-Lipschitz in the state, the imagined return over `H` steps is within
`K * Σ_t ε (1 + L + ... + L^(t-1))` of the true one (T1 feeds T7). -/
theorem model_gap_from_step_error (L ε K : ℝ) (hL : 0 ≤ L) (hK : 0 ≤ K) (e c chat : ℕ → ℝ)
    (H : ℕ) (h0 : e 0 ≤ 0) (hstep : ∀ t, e (t + 1) ≤ L * e t + ε)
    (hc : ∀ t ∈ Finset.range H, |chat t - c t| ≤ K * e t) :
    |∑ t ∈ Finset.range H, chat t - ∑ t ∈ Finset.range H, c t|
      ≤ K * ∑ t ∈ Finset.range H, ε * ∑ i ∈ Finset.range t, L ^ i := by
  refine le_trans (return_gap_le H c chat e K hc) ?_
  refine mul_le_mul_of_nonneg_left ?_ hK
  apply Finset.sum_le_sum
  intro t _
  exact V8Proofs.Hier.compose_bound L ε hL e h0 hstep t

/-! ## T9: averaging members, in output space and in weight space -/

section Soup

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] {ι : Type*}

/-- The summed squared error of the members equals their scatter around the member average plus
`K` times the squared error of the average (`K` = number of members). -/
theorem ensemble_identity (s : Finset ι) (a : ι → V) (abar y : V)
    (hbar : ∑ i ∈ s, (a i - abar) = 0) :
    ∑ i ∈ s, ‖a i - y‖ ^ 2 = ∑ i ∈ s, ‖a i - abar‖ ^ 2 + (s.card : ℝ) * ‖abar - y‖ ^ 2 := by
  have h := V8Proofs.Hier.risk_split_one s (fun _ => (1 : ℝ)) a abar y (by simpa using hbar)
  simpa using h

/-- The member average is never worse than the member mean (squared error, any dimension). -/
theorem ensemble_le_member_sum (s : Finset ι) (a : ι → V) (abar y : V)
    (hbar : ∑ i ∈ s, (a i - abar) = 0) :
    (s.card : ℝ) * ‖abar - y‖ ^ 2 ≤ ∑ i ∈ s, ‖a i - y‖ ^ 2 := by
  rw [ensemble_identity s a abar y hbar]
  have hsc : 0 ≤ ∑ i ∈ s, ‖a i - abar‖ ^ 2 := Finset.sum_nonneg (fun i _ => by positivity)
  linarith

end Soup

/-- For a head that is linear in its weights, a weighted average of weights gives exactly the same
weighted average of outputs. -/
theorem linear_head_soup {W O : Type*} [AddCommGroup W] [Module ℝ W] [AddCommGroup O] [Module ℝ O]
    (f : W →ₗ[ℝ] O) {ι : Type*} (s : Finset ι) (lam : ι → ℝ) (θ : ι → W) :
    f (∑ i ∈ s, lam i • θ i) = ∑ i ∈ s, lam i • f (θ i) := by
  simp [map_sum, map_smul]

end V8Proofs.Rule
