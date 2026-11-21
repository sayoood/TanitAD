# PREREG_HSGATE -- a per-window gate between the clean-split H2cn progress and the self-sharpened (factored) progress, fitted on the FRESH split (Master Mind, 2026-10-26 ~06:36 CET)

Sealed BEFORE the fresh SELFSHARP trace exists and before any gate fit, pick, rate, cell or bar. The committed FAIL-L1 lever of SHIP-R8
Amendment 4c (d3a9347b): *"a learned per-window gate between H2cn progress and the self-sharpened progress, fitted on the fresh split"*.

## 0. What is known (disclosed, MEASURED; T1 self-action open loop on eval139, S6 compliant, paired episode-cluster bootstrap over EPISODES)
* `H2CN_DEPLOY8` (A4 PASS, 2da6a63): FD 0.0603 vs RETIME 0.1369 (separated), MD 0.5336 vs 0.4695 (not separated).
* `FACTORED_DEPLOY8` (A5 PASS, aea2580; SSGUARD REPLICATED 160f583): the self-sharpened progress cuts MD (-0.08 separated vs BASE).
* A4c (`out_4c/BARS_4C.json` 744502f0): feeding the lon softmax INTO the progress regression moves along the same MD/FD trade-off
  (MD -0.0908 sep., FD +0.0436 sep.), and a refit with SCRAMBLED logits also moves it (MD -0.0282 sep., FD +0.0140 sep.) -> bars VOID.
  Two lessons are binding here: (i) any refit shifts the operating point, so attribution is tested against a RATE-MATCHED MIX, not a
  label derangement; (ii) a progress head scored on its own fit rows is in-sample, so the gate's labels use OUT-OF-FOLD H2cn predictions.

## 1. The arm `HSGATE8`
* Per window: emitted positions = `FACTORED_DEPLOY8`'s if `g(x) = 1`, else `H2CN_DEPLOY8`'s -- both registered constructions reused
  VERBATIM (A4 stage `f293b7bd...6f32`; A5 composition + the registered per-frame RETIME table). Both share the native PATH; only the
  progress differs.
* `x` (inference-available only; no GT, no future ego, no situation-classifier output): softmax of the 6 LON6 logits (`l2_common.lon6`
  of `nav_true.tacv6_lon_logits`) (6); one-hot of their argmax (6); v0/10 (1); `d = (arc_S6(P_F) - arc_S6(P_H)) / max(arc_S6(P_H), 1 m)`
  at the last S6 slot (1); `d * v0/10` (1). 15 features, standardised on FIT rows.
* `g(x) = 1[sigmoid(w.x + b) >= 0.5]`, logistic regression, L2 on w, float64, LBFGS, seed 0.
* **Label (FIT rows only):** `y = 1[ADE_S6(P_F) < ADE_S6(P_H)]` against the human future `G`, where `P_H` uses **out-of-fold** H2cn
  progress: the registered H2cn recipe (S2 + S6 heads, the REGISTERED seed-0 wd / epochs) refit on 4 of 5 episode-grouped folds and
  applied to the 5th (`folds_of(eid, 0)`), and `P_F` = the fresh SELFSHARP roll's plan composed + staged exactly as A5.
* **Fit:** lambda in {1e-3, 1e-2, 1e-1, 1, 10, 100} by 5-fold episode-grouped CV on FIT rows (a DIFFERENT fold seed, 1, from the OOF
  folds), criterion = mean staged ADE_S6 of the gated pick on held-out folds; final fit on all FIT rows. Exactly ONE configuration is scored.
* **FIT** = the L1C FRESH split, S6 compliant rows (7,147 as `l1c_fit_score.load_fresh` loads them), joined to the fresh SELFSHARP trace
  (`F:/refcv9_ssfresh/bank/trace_selfsharp_fresh_s0.npz`, inference seed 0). **SCORE** = eval139, the landed `H2CN_DEPLOY8` and
  `FACTORED_DEPLOY8` packages, inference seeds 0 and 1 (never pooled). The gate is frozen and its sha256 registered by the Master Mind
  before any SCORE-side output.

## 2. Controls (a failed control voids the bars)
* **K-ID0 / K-ID1:** `g = 0` everywhere reproduces `H2CN_DEPLOY8`'s positions bit-for-bit; `g = 1` reproduces `FACTORED_DEPLOY8`'s.
* **K-FRESH-SS:** the fresh trace is FINAL with 10,456 windows; the applied lon logits are one-hot +-20 on the label on every window; raw
  argmax == label on every window. **K-JOIN:** every FIT row joins exactly one trace window (clip index, ws).
* **K-OOF:** every FIT row's `P_H` comes from a fold model whose training episodes exclude that row's episode (asserted from the fold
  bookkeeping), and the all-rows refit reproduces the registered stage's predictions bit-for-bit (the recipe is the registered one).
* **K-EVALFEAT:** the eval-side features are built from the same extras keys K-LABSRC validated (`F:/refcv9_ssfresh/bank/KLABSRC.json`).

## 3. Bars (S6 compliant, paired vs `H2CN_DEPLOY8` on the SAME draw, B 2000 seed 0; PASS = G1-G4 at BOTH inference seeds)
* **G1** MD separated BELOW `H2CN_DEPLOY8`.  **G2** FD NOT separated above `H2CN_DEPLOY8`.
* **G3** SHIP-R8 (1)-(7)+(1b) on `HSGATE8` -> OVERALL PASS (Amendment 4's clauses, arm swapped).
* **G4 (attribution):** against `MIX_r`, the per-window EXPECTATION of a gate that picks F at HSGATE8's own SCORE-side rate `r` without
  looking (indicator `r*I_F + (1-r)*I_H`, deterministic): MD(HSGATE8) - MD(MIX_r) separated below OR FD(HSGATE8) - FD(MIX_r) separated
  below, and neither separated above. Without G4 a PASS on G1-G3 is reported as **"rate effect, not attributable"**.
* Beside: the compliance row (vs H2CN / CAP8 / RETIME), the gate's F-rate per surface, E1 / E3, ADE and the four families vs
  `H2CN_DEPLOY8` and vs `FACTORED_DEPLOY8`; A4d's fit-seed floor beside every H2cn-relative cell.

## 4. Tier / estimator / deployment note
T1 self-action OPEN LOOP on logged eval139 frames (not driving performance); paired episode-cluster bootstrap over EPISODES only; one
refcv8 training seed; one gate fit; inference seeds 0 / 1 never pooled. **Deployment note:** HSGATE8 needs the self-sharpened decode
at inference (a second planner pass, or an in-model factored decode) -- its latency against the 100 ms budget is UNMEASURED and is a PI
decision before any ship.

## 5. Committed next steps
* **PASS (G1-G4)** -> HSGATE8 joins the shipped-stage choice for the PI (with the latency item); refcv9 design input: the progress
  head routes on the lon decision.
* **G1-G3 PASS, G4 FAIL** -> the gate learns a mixing RATE, not windows: these features cannot tell which progress is right; no further
  post-hoc progress stages on refcv8 -- the lever moves to refcv9 (a progress head trained with the lon decision as a training target on a
  clean split), and the PI chooses between H2CN_DEPLOY8 / H2CN_CAP8 / FACTORED_DEPLOY8 on their four-family rows.
* **G2 FAIL** -> operating-point trade: next = the same gate with its threshold chosen on FIT rows under an FD cap (FD_FIT(gate) <=
  FD_FIT(H2cn-OOF)), pre-registered first.
* **G1 FAIL** -> name it; H2CN_DEPLOY8 stands.
