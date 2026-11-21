# SPEC addendum A2 — a TRAIN-fitted linear re-scorer over the model's own selection terms (pre-registered BEFORE it is computed)

**Written 2026-10-04 after SPEC §5's levers and A1's W1/W3 were scored on EVAL seed 0 (all FAILED their bar:
every |ΔADE| ≤ 0.13 m on any class while the oracle gap is ~2.0 m on GT-turn windows) and BEFORE any number of
this addendum existed.** sha256 + time in `raw/SPEC_SHA256.txt`. POST-HOC relative to SPEC.md; reported in its own
section, never merged into SPEC §5's verdict.

**Why this is the next lever (Rule Zero).** Every fixed rule re-weights ONE term; the measured gap needs the score
re-weighted across ALL its terms at once. The cheapest experiment that answers *"is the information to pick a better
candidate already in the model's inference-time outputs?"* — without retraining the model — is a linear
conditional-logit re-scorer fitted on the TRAIN windows. If it clears the bar, it is shippable as-is (a 16-parameter
inference-time head) and names the recipe lever (train the selector on these terms); if it does not, the information
is not in these outputs and the lever is a training change.

**Model.** For window w and candidate c (within `reach_keep`): score `u_wc = θ · φ_wc`, φ standardised with TRAIN
mean/std. Fitted by maximising `Σ_w log softmax_c(u_wc)[oracle_w]` (oracle = argmin ADE over the window's
`reach_keep` candidates; windows whose GT has < 2 valid slots excluded) + L2 penalty α‖θ‖², α ∈ {1e-3, 1e-2, 1e-1, 1,
10} chosen by 5-fold EPISODE-grouped cross-validation on TRAIN (criterion: mean held-out ADE of the argmax pick).
Emitted pick = argmax_c u_wc within `reach_keep`.

**Features φ (16, all inference-time outputs of this model or geometry of its own fan; nothing label-side):**
1 sampler confidence (`refined_base`), 2 tac8-lat term, 3 tac8-lon term, 4 behaviour term, 5 nav-compliance term,
6 E9 goal graft (`s_e9 − s_core`), 7 `s_e9` z-scored within the window, 8 `1[dir(c) == tactical side]`,
9 `p_side(c)` (tactical probability of the candidate's side), 10 `1[dir(c) == nav side]` (0 on follow clips),
11 `|θ(c)|` (rad), 12 candidate progress at 6 s / max(6·v0, 1 m), 13 candidate peak lateral acceleration proxy
`max_t v_t² |κ_t|` on the 8-slot geometry (m/s²), 14 ADE8 to the residual prior path, 15 Σ_τ∈{2,4,6 s}
‖c(τ) − ĝ_tac(τ)‖ (m), 16 `1[c is the E9 pick]`.

**Arms (all fitted on TRAIN only, scored on EVAL seed 0, replicated on seed 1):** X1 = all 16 features;
X2 = X1 without the nav features (5, 10) — the attribution of the oracle nav; X3 = X1 without the goal-head feature
(15). **The reported A2 lever = X1** (fixed now, not chosen on EVAL). Bar = SPEC §5's four parts, unchanged.

**Controls.** (a) A shuffled-target control: the same fit with each TRAIN window's oracle index replaced by a
uniformly random `reach_keep` index must NOT beat V0 on EVAL (ΔADE CI must contain 0 or be > 0). (b) The fit's
TRAIN in-sample and cross-validated ADE are both reported (an in-sample gain that does not survive CV is overfit).
