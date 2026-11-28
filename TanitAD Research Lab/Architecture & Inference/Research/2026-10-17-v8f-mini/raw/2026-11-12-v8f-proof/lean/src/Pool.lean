import Mathlib

/-!
# Group P in Lean 4: what a pooled state keeps and loses

Machine-checked version of group P of `fv/FV_SPEC_V5_POOL.md` (v8 proof programme), lifted from
"4 cells, one real feature per cell" (solver) to:

* any number of cells (`ι` any type; sums over a finite set of cells),
* vector features and vector states (`E`, `F` real vector spaces),
* any decoder (any function of the state) and any real target of the object position,
* pooling over any set of cells with any weight (mean state, block state, grid state are special cases),
* a permutation statement for ANY per-cell map (not only a linear one).

Reading: a state pooled over a set of cells holds how much is inside the set and nothing about where.
Nothing here is a measurement and nothing is a statement about driving performance.
-/

namespace V8Proofs.Pool

/-! ## Scalar facts (VC-P1, VC-P2 and the mutated twins) -/

/-- VC-P1 (pair bound): two scenes with equal state and targets `y1`, `y2` cost every decoded value `g`
at least half the squared target difference. -/
theorem pair_bound (y1 y2 g : ℝ) : (y1 - y2) ^ 2 / 2 ≤ (y1 - g) ^ 2 + (y2 - g) ^ 2 := by
  nlinarith [sq_nonneg (y1 + y2 - 2 * g)]

/-- VC-P2: the pair bound is attained at the midpoint. -/
theorem pair_bound_attained (y1 y2 : ℝ) :
    (y1 - (y1 + y2) / 2) ^ 2 + (y2 - (y1 + y2) / 2) ^ 2 = (y1 - y2) ^ 2 / 2 := by
  ring

/-- M-P1 is false: the pair bound without the factor 1/2 does not hold. -/
theorem stronger_pair_bound_false :
    ¬ ∀ y1 y2 g : ℝ, (y1 - y2) ^ 2 ≤ (y1 - g) ^ 2 + (y2 - g) ^ 2 := by
  intro h
  have h1 := h 1 0 (1 / 2)
  norm_num at h1

/-- M-P4 is false: no risk floor above the variance 1/4 for two positions. -/
theorem two_position_floor_tight : ¬ ∀ g : ℝ, (26 : ℝ) / 100 ≤ ((1 - g) ^ 2 + g ^ 2) / 2 := by
  intro h
  have h1 := h (1 / 2)
  norm_num at h1

/-- M-P5 is false: the two-position floor 1/4 does not hold for four positions (the floor there is 3/16). -/
theorem four_position_floor_tight : ¬ ∀ g : ℝ, (1 : ℝ) / 4 ≤ ((1 - g) ^ 2 + 3 * g ^ 2) / 4 := by
  intro h
  have h1 := h (1 / 4)
  norm_num at h1

/-- VC-P1 for vector targets (for example a position in the plane). -/
theorem pair_bound_vec {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] (y1 y2 g : V) :
    ‖y1 - y2‖ ^ 2 / 2 ≤ ‖y1 - g‖ ^ 2 + ‖y2 - g‖ ^ 2 := by
  have h := parallelogram_law_with_norm ℝ (y1 - g) (y2 - g)
  have e : (y1 - g) - (y2 - g) = y1 - y2 := by abel
  rw [e] at h
  nlinarith [mul_self_nonneg ‖y1 - g + (y2 - g)‖, h]

/-! ## Risk floor: equal states over a set of positions leave only the variance (VC-P4, VC-P5, n positions) -/

section Risk

variable {ι : Type*}

theorem sum_sq_shift (K : Finset ι) (t : ι → ℝ) (c g : ℝ) :
    ∑ k ∈ K, (t k - g) ^ 2
      = ∑ k ∈ K, (t k - c) ^ 2 + 2 * (c - g) * ∑ k ∈ K, (t k - c) + (K.card : ℝ) * (c - g) ^ 2 := by
  have key : ∀ k, (t k - g) ^ 2 = (t k - c) ^ 2 + 2 * (c - g) * (t k - c) + (c - g) ^ 2 := by
    intro k; ring
  have h1 : ∑ k ∈ K, (t k - g) ^ 2
      = ∑ k ∈ K, ((t k - c) ^ 2 + 2 * (c - g) * (t k - c) + (c - g) ^ 2) :=
    Finset.sum_congr rfl (fun k _ => key k)
  rw [h1, Finset.sum_add_distrib, Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const,
    nsmul_eq_mul]

theorem sum_centered_eq_zero (K : Finset ι) (hK : K.Nonempty) (t : ι → ℝ) :
    ∑ k ∈ K, (t k - (∑ j ∈ K, t j) / (K.card : ℝ)) = 0 := by
  have hn : (K.card : ℝ) ≠ 0 := by
    exact_mod_cast (Finset.card_pos.mpr hK).ne'
  have hc : (K.card : ℝ) * ((∑ j ∈ K, t j) / (K.card : ℝ)) = ∑ j ∈ K, t j := by
    field_simp
  rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul, hc, sub_self]

/-- The variance of the target over the positions `K` is a floor for every constant decoded value. -/
theorem variance_is_floor (K : Finset ι) (hK : K.Nonempty) (t : ι → ℝ) (g : ℝ) :
    ∑ k ∈ K, (t k - (∑ j ∈ K, t j) / (K.card : ℝ)) ^ 2 ≤ ∑ k ∈ K, (t k - g) ^ 2 := by
  have h := sum_sq_shift K t ((∑ j ∈ K, t j) / (K.card : ℝ)) g
  rw [sum_centered_eq_zero K hK t, mul_zero, add_zero] at h
  rw [h]
  exact le_add_of_nonneg_right (mul_nonneg (Nat.cast_nonneg _) (sq_nonneg _))

/-- Zero skill from equal states. `st k` is the state of the scene whose object is at position `k`.
If the state is the same for all positions in `K`, then EVERY decoder `dec` of the state pays at least
the variance of EVERY target `t` over `K` (object equally likely at each position of `K`). -/
theorem zero_skill_of_equal_states {S : Type*} (K : Finset ι) (hK : K.Nonempty) (st : ι → S)
    (hst : ∀ k ∈ K, ∀ k' ∈ K, st k = st k') (dec : S → ℝ) (t : ι → ℝ) :
    ∑ k ∈ K, (t k - (∑ j ∈ K, t j) / (K.card : ℝ)) ^ 2 ≤ ∑ k ∈ K, (t k - dec (st k)) ^ 2 := by
  obtain ⟨k0, hk0⟩ := hK
  have hrw : ∑ k ∈ K, (t k - dec (st k)) ^ 2 = ∑ k ∈ K, (t k - dec (st k0)) ^ 2 :=
    Finset.sum_congr rfl (fun k hk => by rw [hst k hk k0 hk0])
  rw [hrw]
  exact variance_is_floor K ⟨k0, hk0⟩ t _

end Risk

/-! ## Cells, scenes, pooled states -/

section Cells

variable {ι : Type*} [DecidableEq ι]
variable {E F : Type*} [AddCommGroup E] [Module ℝ E] [AddCommGroup F] [Module ℝ F]

/-- The per-cell adapter: a linear map `A` plus a constant `b i` per cell (bias and position term). -/
def cell (A : E →ₗ[ℝ] F) (b : ι → F) (x : ι → E) (i : ι) : F := A (x i) + b i

/-- A scene: background `u` plus an object of strength `s` in cell `k`. -/
def scene (u : ι → E) (s : E) (k : ι) : ι → E := fun i => u i + (if i = k then s else 0)

/-- A state pooled over the cells `K` with weight `w` (mean state: all cells, weight 1/n; block state: one block). -/
def pool (w : ℝ) (K : Finset ι) (A : E →ₗ[ℝ] F) (b : ι → F) (x : ι → E) : F :=
  w • ∑ i ∈ K, cell A b x i

/-- The grid state: one adapted feature per cell. -/
def grid (A : E →ₗ[ℝ] F) (b : ι → F) (x : ι → E) : ι → F := fun i => cell A b x i

theorem cell_scene (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E) (k i : ι) :
    cell A b (scene u s k) i = cell A b u i + (if i = k then A s else 0) := by
  simp only [cell, scene]
  split_ifs with h
  · rw [map_add]; abel
  · simp

theorem sum_cell_scene (K : Finset ι) (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E) (k : ι) :
    ∑ i ∈ K, cell A b (scene u s k) i = (∑ i ∈ K, cell A b u i) + (if k ∈ K then A s else 0) := by
  simp only [cell_scene, Finset.sum_add_distrib, Finset.sum_ite_eq']

/-- An object inside the pooled set shifts the pooled state by `w • A s`, wherever it is; outside: nothing. -/
theorem pool_scene (w : ℝ) (K : Finset ι) (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E) (k : ι) :
    pool w K A b (scene u s k) = pool w K A b u + (if k ∈ K then w • A s else 0) := by
  unfold pool
  rw [sum_cell_scene, smul_add]
  by_cases hk : k ∈ K
  · rw [ite_eq_left hk, ite_eq_left hk]
  · rw [ite_eq_right hk, ite_eq_right hk, smul_zero]

/-- VC-P3 / first half of VC-P9 (any number of cells, vector features): moving the object inside the
pooled set leaves the pooled state equal, for every background, weight and adapter. -/
theorem pool_move_inside (w : ℝ) (K : Finset ι) (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E)
    {i j : ι} (hi : i ∈ K) (hj : j ∈ K) :
    pool w K A b (scene u s i) = pool w K A b (scene u s j) := by
  rw [pool_scene, pool_scene, ite_eq_left hi, ite_eq_left hj]

/-- Second half of VC-P9: moving the object across the border of the pooled set changes the pooled
state by `w • A s`, independent of the background. -/
theorem pool_move_across (w : ℝ) (K : Finset ι) (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E)
    {i j : ι} (hi : i ∈ K) (hj : j ∉ K) :
    pool w K A b (scene u s i) - pool w K A b (scene u s j) = w • A s := by
  rw [pool_scene, pool_scene, ite_eq_left hi, ite_eq_right hj, add_zero]
  abel

/-- VC-P6 (any number of cells, vector features): the grid state reads the object position linearly;
the read `G i` differs by `A s` between "object in cell i" and "object in cell j", for every background. -/
theorem grid_scene_diff (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E) {i j : ι} (hij : i ≠ j) :
    grid A b (scene u s i) i - grid A b (scene u s j) i = A s := by
  have h1 := cell_scene A b u s i i
  have h2 := cell_scene A b u s j i
  rw [ite_eq_left rfl] at h1
  rw [ite_eq_right hij, add_zero] at h2
  show cell A b (scene u s i) i - cell A b (scene u s j) i = A s
  rw [h1, h2]
  abel

/-- With a live adapter (`A s ≠ 0`) the grid states of two object positions differ. -/
theorem grid_separates (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E) {i j : ι} (hij : i ≠ j)
    (hs : A s ≠ 0) : grid A b (scene u s i) ≠ grid A b (scene u s j) := by
  intro h
  have hd := grid_scene_diff A b u s hij
  rw [h, sub_self] at hd
  exact hs hd.symm

/-- M-P6 is false: without a live adapter direction (`A s = 0`) the grid state does not see the object. -/
theorem grid_blind_without_live_adapter (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E) (i j : ι)
    (hs : A s = 0) : grid A b (scene u s i) = grid A b (scene u s j) := by
  funext c
  show cell A b (scene u s i) c = cell A b (scene u s j) c
  rw [cell_scene, cell_scene, hs]
  simp

/-- A state made of several pooled states (one per block `c`, pooled over `Ps c` with weight `w c`). -/
def blockState {κ : Type*} (w : κ → ℝ) (Ps : κ → Finset ι) (A : E →ₗ[ℝ] F) (b : ι → F)
    (x : ι → E) : κ → F := fun c => pool (w c) (Ps c) A b x

/-- VC-P9, general: two object positions that no block separates give the same block state. -/
theorem blockState_move_inside {κ : Type*} (w : κ → ℝ) (Ps : κ → Finset ι) (A : E →ₗ[ℝ] F)
    (b : ι → F) (u : ι → E) (s : E) {i j : ι} (h : ∀ c, i ∈ Ps c ↔ j ∈ Ps c) :
    blockState w Ps A b (scene u s i) = blockState w Ps A b (scene u s j) := by
  funext c
  show pool (w c) (Ps c) A b (scene u s i) = pool (w c) (Ps c) A b (scene u s j)
  rw [pool_scene, pool_scene]
  by_cases hi : i ∈ Ps c
  · rw [ite_eq_left hi, ite_eq_left ((h c).mp hi)]
  · rw [ite_eq_right hi, ite_eq_right (fun hj => hi ((h c).mpr hj))]

/-- VC-P9, general: a block that separates the two positions reads the move, by `w c • A s`. -/
theorem blockState_move_across {κ : Type*} (w : κ → ℝ) (Ps : κ → Finset ι) (A : E →ₗ[ℝ] F)
    (b : ι → F) (u : ι → E) (s : E) {i j : ι} (c : κ) (hi : i ∈ Ps c) (hj : j ∉ Ps c) :
    blockState w Ps A b (scene u s i) c - blockState w Ps A b (scene u s j) c = w c • A s :=
  pool_move_across (w c) (Ps c) A b u s hi hj

/-- Zero skill inside the blocks: if no block separates the positions in `K`, every decoder of the whole
block state pays at least the variance of every target over `K`. -/
theorem location_zero_skill_blocks {κ : Type*} (w : κ → ℝ) (Ps : κ → Finset ι) (A : E →ₗ[ℝ] F)
    (b : ι → F) (u : ι → E) (s : E) (K : Finset ι) (hK : K.Nonempty)
    (hsep : ∀ k ∈ K, ∀ k' ∈ K, ∀ c, k ∈ Ps c ↔ k' ∈ Ps c) (dec : (κ → F) → ℝ) (t : ι → ℝ) :
    ∑ k ∈ K, (t k - (∑ j ∈ K, t j) / (K.card : ℝ)) ^ 2
      ≤ ∑ k ∈ K, (t k - dec (blockState w Ps A b (scene u s k))) ^ 2 :=
  zero_skill_of_equal_states K hK (fun k => blockState w Ps A b (scene u s k))
    (fun k hk k' hk' => blockState_move_inside w Ps A b u s (hsep k hk k' hk')) dec t

/-- VC-P8, second half: the grid state sees the motion in the cell the object enters ... -/
theorem grid_sees_motion_enter (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E) {i j : ι}
    (hij : i ≠ j) : grid A b (scene u s j) j - grid A b (scene u s i) j = A s :=
  grid_scene_diff A b u s (Ne.symm hij)

/-- ... and in the cell it leaves. -/
theorem grid_sees_motion_leave (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E) {i j : ι}
    (hij : i ≠ j) : grid A b (scene u s j) i - grid A b (scene u s i) i = -(A s) := by
  rw [← neg_sub, grid_scene_diff A b u s hij]

/-! ## The mean state (the present level-1 world-model state before its projection) -/

section Mean

variable [Fintype ι]

/-- The mean state: all cells, weight 1/n. -/
noncomputable def meanState (A : E →ₗ[ℝ] F) (b : ι → F) (x : ι → E) : F :=
  pool ((Fintype.card ι : ℝ)⁻¹) Finset.univ A b x

/-- VC-P3, n cells: moving the object leaves the mean state equal. -/
theorem mean_move (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E) (i j : ι) :
    meanState A b (scene u s i) = meanState A b (scene u s j) :=
  pool_move_inside _ _ A b u s (Finset.mem_univ i) (Finset.mem_univ j)

/-- Each object shifts the mean state by the same amount, wherever it is. -/
theorem mean_scene (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E) (k : ι) :
    meanState A b (scene u s k) = meanState A b u + (Fintype.card ι : ℝ)⁻¹ • A s := by
  unfold meanState
  rw [pool_scene, ite_eq_left (Finset.mem_univ k)]

/-- VC-P8, first half: an object that moves from cell `i` to cell `j` on a constant background changes
nothing in the mean state, so a copy predicts the next mean state exactly. -/
theorem mean_blind_to_motion (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E) (i j : ι) :
    meanState A b (scene u s j) - meanState A b (scene u s i) = 0 :=
  sub_eq_zero.mpr (mean_move A b u s j i)

/-- VC-P4 / VC-P5 for n cells, any decoder, any target of the position: the position of the object is
not readable from the mean state. Risk of every decoder ≥ variance of the target (object equally
likely in every cell), for every background, strength, adapter and position term. -/
theorem location_zero_skill_mean [Nonempty ι] (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E)
    (dec : F → ℝ) (t : ι → ℝ) :
    ∑ k, (t k - (∑ j, t j) / (Fintype.card ι : ℝ)) ^ 2
      ≤ ∑ k, (t k - dec (meanState A b (scene u s k))) ^ 2 := by
  have h := zero_skill_of_equal_states (Finset.univ : Finset ι) Finset.univ_nonempty
    (fun k => meanState A b (scene u s k)) (fun k _ k' _ => mean_move A b u s k k') dec t
  rwa [Finset.card_univ] at h

/-- The same as a skill: 1 - risk / variance ≤ 0 whenever the target varies. -/
theorem location_skill_le_zero_mean [Nonempty ι] (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E)
    (dec : F → ℝ) (t : ι → ℝ)
    (hvar : 0 < ∑ k, (t k - (∑ j, t j) / (Fintype.card ι : ℝ)) ^ 2) :
    1 - (∑ k, (t k - dec (meanState A b (scene u s k))) ^ 2)
        / (∑ k, (t k - (∑ j, t j) / (Fintype.card ι : ℝ)) ^ 2) ≤ 0 := by
  have h := location_zero_skill_mean A b u s dec t
  rw [sub_nonpos, le_div_iff₀ hvar, one_mul]
  exact h

/-- Several objects, one after the other, at the positions of the list (repeats allowed). -/
def sceneList (u : ι → E) (s : E) : List ι → (ι → E)
  | [] => u
  | k :: ks => scene (sceneList u s ks) s k

/-- VC-P7, general: the mean state of a scene with objects of equal strength depends on their number only. -/
theorem mean_sceneList (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E) (ks : List ι) :
    meanState A b (sceneList u s ks)
      = meanState A b u + (ks.length : ℝ) • ((Fintype.card ι : ℝ)⁻¹ • A s) := by
  induction ks with
  | nil => simp [sceneList]
  | cons k ks ih =>
    simp only [sceneList, List.length_cons, Nat.cast_add, Nat.cast_one]
    rw [mean_scene, ih, add_smul, one_smul, add_assoc]

theorem mean_depends_on_count_only (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E)
    (ks ks' : List ι) (h : ks.length = ks'.length) :
    meanState A b (sceneList u s ks) = meanState A b (sceneList u s ks') := by
  rw [mean_sceneList, mean_sceneList, h]

theorem mean_count_diff (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E) (ks ks' : List ι) :
    meanState A b (sceneList u s ks) - meanState A b (sceneList u s ks')
      = ((ks.length : ℝ) - (ks'.length : ℝ)) • ((Fintype.card ι : ℝ)⁻¹ • A s) := by
  rw [mean_sceneList, mean_sceneList, sub_smul]
  abel

/-- VC-P7: with a live adapter the mean state separates different object counts. -/
theorem mean_separates_counts [Nonempty ι] (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E)
    (ks ks' : List ι) (hlen : ks.length ≠ ks'.length) (hs : A s ≠ 0) :
    meanState A b (sceneList u s ks) ≠ meanState A b (sceneList u s ks') := by
  intro h
  have hd := mean_count_diff A b u s ks ks'
  rw [h, sub_self] at hd
  have hn : (Fintype.card ι : ℝ) ≠ 0 := by
    exact_mod_cast (Fintype.card_ne_zero : Fintype.card ι ≠ 0)
  have hl : (ks.length : ℝ) - (ks'.length : ℝ) ≠ 0 := by
    intro h0
    exact hlen (by exact_mod_cast sub_eq_zero.mp h0)
  rcases smul_eq_zero.mp hd.symm with h1 | h1
  · exact hl h1
  · rcases smul_eq_zero.mp h1 with h2 | h2
    · exact hn (inv_eq_zero.mp h2)
    · exact hs h2

/-- M-P7 is false: the mean state does not separate counts when strengths differ; two objects of
strength `s` read the same as one object of strength `s + s`. -/
theorem mean_confuses_two_with_double (A : E →ₗ[ℝ] F) (b : ι → F) (u : ι → E) (s : E) (i j k : ι) :
    meanState A b (scene (scene u s i) s j) = meanState A b (scene u (s + s) k) := by
  rw [mean_scene, mean_scene, mean_scene, map_add, smul_add]
  abel

end Mean

end Cells

/-! ## Any per-cell map: a permutation of the cells leaves the sum (and so the mean) equal -/

/-- Control c3 of SPATIAL stage 1 as a theorem. `a` is ANY per-cell map (LayerNorm followed by a linear
layer is one), `p i` a per-cell additive term. Reordering the cell contents leaves the pooled sum equal. -/
theorem sum_perm_invariant {ι X G : Type*} [Fintype ι] [AddCommMonoid G] (a : X → G) (p : ι → G)
    (σ : Equiv.Perm ι) (x : ι → X) :
    ∑ i, (a (x (σ i)) + p i) = ∑ i, (a (x i) + p i) := by
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib, Equiv.sum_comp σ (fun i => a (x i))]

/-- Every read-out of the pooled sum is blind to the cell order. -/
theorem readout_perm_invariant {ι X G R : Type*} [Fintype ι] [AddCommMonoid G] (a : X → G)
    (p : ι → G) (σ : Equiv.Perm ι) (x : ι → X) (f : G → R) :
    f (∑ i, (a (x (σ i)) + p i)) = f (∑ i, (a (x i) + p i)) := by
  rw [sum_perm_invariant]

end V8Proofs.Pool
