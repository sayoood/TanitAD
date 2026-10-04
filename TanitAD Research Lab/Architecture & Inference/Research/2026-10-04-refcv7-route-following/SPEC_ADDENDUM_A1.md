# SPEC addendum A1 — three further inference-only selection levers (pre-registered BEFORE they are computed)

**Written 2026-10-04 after the first EVAL chain numbers were read (SPEC.md §4 tables, eval_s0) and BEFORE the
TRAIN capture finished, i.e. before any number of SPEC §5's levers or of the levers below existed.** sha256 + time
in `raw/SPEC_SHA256.txt`. These levers are POST-HOC relative to SPEC.md (motivated by what eval_s0 showed) and are
reported in their own section, never merged into SPEC §5's verdict.

**What motivated them (eval_s0, MEASURED, open-loop):** the fan contains a direction-correct candidate on 100 % of
GT-turn windows, the oracle is the E9 pick on only 24 % of them (median oracle rank 3 of 117), E9's score
correlates with candidate quality (mean per-window Spearman ≈ 0.72 on turns; random control ≈ 0), the nav-compliance
term is ~4 % of the score spread and changes 1–2 picks of 291 informative windows, and the dominant term that
prefers the pick over the oracle is the sampler's own confidence. ⇒ the loss is in the TOP of a ranking that is
right in the bulk; three cheap levers that act on exactly that are:

| id | rule (within `reach_keep`; ties → lowest index) | grid (fitted on TRAIN) |
|---|---|---|
| W1 | **minimum-Bayes-risk pick**: `p = softmax(s_e9 / T)` over the top-k candidates by `s_e9`; emit `argmin_c Σ_j p_j · ADE8(c, c_j)` (ADE8 = mean L2 over the 8 slots between two candidate paths) | T ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 1, 2} × k ∈ {8, 16, 117} |
| W2 | **goal-point consistency at 4 s and 6 s**: `s' = s_e9 − λ · (‖c(4 s) − ĝ(4 s)‖ + ‖c(6 s) − ĝ(6 s)‖)` with ĝ the model's OWN tactical goal head `g_tac` at τ = 40 / 60 ticks (E9 already uses its τ = 20 point); metres | λ ∈ {0.02, 0.05, 0.1, 0.2, 0.5, 1, 2} |
| W3 | **E9 goal-graft scale**: `s' = blend(s_core, k · graft)` through the SAME seam clamp (1.0) | k ∈ {0, 0.5, 2, 5, 10} |
| W4 | W1 applied to W2's score (both fitted parameters re-fitted jointly on the W1 × W2 grid) | W1 grid × W2 grid |

* **Admissibility (W2):** `g_tac` is a model OUTPUT computed at inference from the trunk features and the refcv6
  condition (nav one-hot, fed ceiling, v0, a0) — the `goal_provenance` stamp in `config.json` records no
  situation-classifier input. It carries the same oracle-nav caveat as everything else in this package.
* **Fit / report / bar:** exactly SPEC §5 — per rule the TRAIN-best grid point by mean ADE over all TRAIN windows;
  the reported A1 lever is the TRAIN-best of W1–W4; EVAL seed 0 with the paired episode-cluster bootstrap vs V0;
  the same four-part bar (turn ADE and turn-correct up with CI excluding 0; straight ADE point ≤ +0.05 m and CI upper
  ≤ +0.10 m; all-window ADE not worse; replicate on seed 1). A lever that misses any part FAILED.
* W2/W4 need `g_tac`, captured by the additive `run_route.py` change (passes `eval_s0g`, `train_s0g`, `reel_s0`,
  `eval_s1`); control: `eval_s0g` / `train_s0g` fans must be bit-identical to `eval_s0` / `train_s0` (same per-window
  seed) — otherwise W2 is scored on its own pass and the mismatch is reported.
