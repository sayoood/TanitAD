# PREREG_LHA0_ARM -- DRAFT, NOT SEALED (LHA0 diagnostic agent for the Master Mind, 2026-10-26). No sha256 file exists for this document on purpose.

Status: **a draft for the Master Mind to edit, decide on and seal.** Nothing in it has been run. In particular NO eval-side output of an LH-A0 head exists
(the head is not fitted or frozen anywhere; LH's discipline "freeze and register before any eval-side output" is kept: the diagnostic touched FRESH rows only, plus
metadata-only reads of the eval poses).

## 0. Why this arm may exist (and the decision it needs)

`PREREG_LHA0_DIAG.md` (sha256 `72ed889d...`, `result/LHA0_DIAG.json`) returned **GO** by its sealed bar: dAUC(B - A) = **+0.0241 [+0.0172, +0.0320]**
(A = [z_tac, v0/10] 0.6699, B = A + a0 + psi_dot0 0.6940; fresh-split OOF, LH's folds, lambda 100). Five fold-seed replicates 0.0227 - 0.0249, all >= 0.020.

**Read the GO together with these, all in the same JSON (none moves the verdict; all move how much weight it bears):**
* **C4 (a0, psi_dot0 permuted WITHIN episode) = +0.0174 [+0.0107, +0.0247] over A -- 0.0026 below the VOID bar and its CI excludes 0.** 72 % of B's gain is reproduced by
  features that carry only the episode's a0 / psi distribution. The window-specific increment is **B - C4 = +0.0067 [+0.0002, +0.0130]** (replicates +0.0045 .. +0.0072; two of five seeds have a CI that includes 0).
  It would not clear +0.020 by itself.
* **The gain is not robust across the head family:** lambda 1000: +0.0143 [+0.0104, +0.0190] (below the bar); lambda 10: +0.0237; the LH-recipe **MLP: +0.0038 [-0.0067, +0.0130]** (CE worse).
* **The forward-difference leak contrast cannot discriminate here:** C5 - B = +0.0009 [-0.0000, +0.0018]; corr(a0 backward, a0 forward) = 0.974 (the dataset's speed channel is smooth). It is not evidence of "no leak".
* **POST-HOC (not pre-registered, `result/LHA0_POSTHOC.json`): a0 ALONE is a far stronger decel predictor than anything in the head.** Raw `-a0`, no model: AUC **0.7699**; the one-feature C3 head 0.7656
  [0.7373, 0.7949]; both are **+0.100 [+0.042, +0.161] above A** and +0.076 above B. Equal-weight fusion A+C3 adds nothing over a0 alone (-0.0006 [-0.038, +0.034]). The ordering holds in every v0 band
  (v0 in [0,3): 0.885 vs A 0.541; [3,8): 0.744 vs 0.717; [8,15): 0.766 vs 0.694; >= 15: 0.797 vs 0.598). So the B recipe (ridge over 513 standardised dims, 6-class LON6 target) leaves most of a0's information unused.

**Decision needed (Master Mind / PI):**
1. build the LH-A0 head literally as B (Design 1 below), expected head gain +0.024 AUC, which -- by the LONHEAD precedent (LH +0.045 AUC over native; F1 and F2 both FAIL at s0 and s1) -- is **unlikely to move the downstream F bars**; or
2. pre-register an a0-first design (Design 2) whose upside is the +0.10 the post-hoc row shows, which the GO does NOT license; or
3. both, Design 2 as the first rolls.

## 1. Design 1 -- LH-A0 (the arm the GO licenses)

* **Input:** refcv8 EMA-48,852 frozen `z_tac` (512-d) + v0/10 + **a0** + **psi_dot0** (the last two standardised on the fit rows, exactly as arm B of the diagnostic).
  a0 = (v[k] - v[k-1]) / dt, psi_dot0 = wrap(yaw[k] - yaw[k-1]) / dt, k = ws (anchor rule exact: max|v0_in - pose speed| = 0.0 on all 10,456 fresh and all 4,754 eval windows), per-clip dt of the cache's clock sidecar.
  PI ruling 2026-10-18: IMU-measurable ego state at t0, causal (backward difference only); **curvature is not an input**; a0_fwd is never an input.
* **Target / fit rows / folds / recipe:** identical to LH (`lh_fit.py`, AMENDMENT1 rows = the 292 lead clips, 8,178 rows, 290 episodes, `folds_of` seed 0, multinomial ridge-logistic).
  lambda: freeze **100** as LH did, **or** select on the sealed LH grid by the same OOF-CE criterion (the diagnostic shows lambda 1000 shrinks the gain to +0.0143, so the choice must be committed BEFORE any eval read).
  New file `lh_a0_fit.py` = `lh_fit.py` + the two extra columns; `lh_fit.py` itself is not edited. Freeze one `LHA0_FROZEN.pt` (+ sha256) with `predict_logits6`, registered by the Master Mind before any eval-side output.
* **Controls (frozen the same way):**
  * `DER-A0`: the same recipe on episode-deranged targets (seed 0) with the same four feature groups.
  * `A0PERM` (new): the frozen LH-A0 scored with a0 / psi_dot0 PERMUTED WITHIN EACH EVAL EPISODE (shared permutation, `default_rng(0)`) -- the eval-side image of C4; must NOT reproduce LH-A0's gain.
  * `LH` / `LHS_FACT8` (existing): the comparator -- the question is whether a0 / psi_dot0 move the frontier BEYOND LH.
* **Eval-side requirement (what must exist on eval139) and the check that was made (metadata only, no frames, Thor not touched):**
  * poses (x, y, yaw, v): **the eval139 episode caches (`*.v2ep.pt`) are NOT in `E:/refcv8_eval_kit`** (they live at `/home/nvidia/data/refcv6-b1-416x1024-eval139` on Thor; the kit holds sidecars, manifest md5s, windows, v9 labels and the ckpt).
    The same fields are on the dev box as the eval dump's `decisions/ep*.npz['ep_poses']` [199, 4] for all 139 episodes (`C:/Users/Admin/refcv8_eval_archive/battery/step48852ema_pf/dump_s0`);
    on the 300 fresh episodes that array equals `_scan_meta`'s `poses[n_stack-1:]` exactly. Eval anchor rule: **k = ws, max|v0_in - ep_poses[ws,3]| = 0.0 over all 4,754 windows** (k = ws -1 / +1: 0.445 / 0.497).
  * dt: `E:/refcv8_eval_kit/data/refcv8_clip_clock_sidecar_ts33.jsonl` covers **136 / 139** eval clips. **3 clips = 102 windows (71 of the 2,495 K-HEAD union windows) have no sidecar row.**
    Draft rule: use nominal dt = 0.1 s for them and disclose it; fresh evidence that this is immaterial: the nominal-dt arm Bnom is identical to B (AUC 0.6940 both; dAUC vs A +0.0241). (Alternative: exclude those clips, as G3 does for tactical targets -- MM's call.)
  * ready-made table: `result/lha0_features_eval139.npz` (4,754 rows, keys (e_i, ws), sha12 only; a0, a0_05, a0_fwd (contrast only), psi_dot0, dt (NaN for the 3 clips); a0 mean -0.034 / sd 0.75, psi_dot0 sd 0.086; no row lacks a k-1 / k-5 frame (min ws 7); the only NaN rows are the 102 no-dt windows of the 3 clips above; forward != backward on 97.2 % of finite rows).
  * nothing else changes: a0 / psi_dot0 enter ONLY the head; roll, planner, trunk, nav and goal paths are untouched (labels-only roll), so the goal path stays information-disjoint from the situation classifier.
* **Deployment caveat (state it in the sealed text):** the diagnostic's a0 is a backward difference of the DATASET's egomotion speed, not a real IMU; a deployed measurement is noisier. A noise-robustness row (a0 + N(0, sigma^2), sigma from a CAN/IMU spec the PI supplies) belongs beside the arm.

## 2. Stages, mirroring PREREG_LONHEAD_C s2 (CPU stages on the dev box; GPU rolls under the shared `gpu_lock.py`)

1. `lh_a0_fit.py` -> `LHA0_FROZEN.pt`, `DERA0_FROZEN.pt` (fresh rows only; K-REPRO analogue: the A-feature refit reproduces LH_FIT_INFO's OOF CE 1.0917272119).
2. **K-HEAD-A0 (go / no-go, descriptive):** LH-A0 decel AUC on the eval139 K-HEAD union (same population: 3,668 S6 -> 3,251 compliant -> 639 decel / 1,856 cruise = 2,495, 136 episodes; reproduced in this diagnostic by `lha0_diag.py prep` at 0.7305),
   renormalised exactly as K-HEAD. **If its point AUC <= LH's 0.7305 the rolls are NOT run** (HEAD-NOT-BETTER-THAN-LH).
3. Labels: LH-A0 / DER-A0 argmax on all 4,754 roll windows (eval z_tac and a0 / psi are bit-identical across inference seeds 0 / 1, so one label file serves both) -> `labels_{LHAS,DERA0}.npz`.
4. Rolls (`l2_roll.py --arm oracle --labels-npz`, one-hot +-20 on the LON6 columns, os-only, per-frame feed, EMA-48,852; ~35 min each): **LHAS s0, LHAS s1, DERA0 s0** (+ **A0PERM s0** if the Master Mind wants the attribution control on the roll side; it is label-only, so it is one more roll, not new machinery).
5. Compose and stage EXACTLY as `FACTORED_DEPLOY8` (A5: `RT.eval_polyline(RT.polyline_geometry(P_os8), arc(P_sharp8))` + the registered per-frame RETIME table) via `code/sub_fx_lhs/run_lhs_fact8.sh` with the traces swapped; verdict via `lh_bars.py` extended with the LHAS arms.

## 3. Proposed bars (Master Mind to set; paired episode-cluster bootstrap B 2000 seed 0 over EPISODES, same inference-seed draw)

* F1 / F2 / G / A exactly as LONHEAD (vs `FACTORED_DEPLOY8`, vs `H2CN_DEPLOY8`, vs BASE-os; attribution vs DER-A0).
* **H (new, the question this arm exists to answer):** `LHAS_FACT8` vs `LHS_FACT8`, same windows: MD separated BELOW with FD not separated above (a0 buys decels at no false-brake cost), or the reverse frontier direction pre-declared.
* **A' (new attribution):** `LHAS_FACT8` minus `A0PERM_FACT8` separated in the same direction as H.
* PASS = ((F1 or F2) and G and A at s0) with H reported beside, or H as the single bar if the Master Mind chooses to ask only the incremental question. REPLICATED iff the same bar holds at s1.
* Tier: T1 self-action OPEN LOOP on logged eval139 frames; one refcv8 training seed; one head fit; inference seeds never pooled; the interval answers "another draw of EPISODES" only.

## 4. Committed outcomes (to be sealed with both sides written in advance)

* **K-HEAD-A0 <= 0.7305:** a0 / psi_dot0 do not carry over to eval139; no rolls; report the fresh-vs-eval gap.
* **PASS (+ REPLICATED):** a0 / psi_dot0 are admissible, measured, decision-relevant inputs; SHIP-R8 amendment on `LHAS_FACT8` needs a PI go; refcv9 trains the lon head with them as first-class inputs.
* **Head better, F1 / F2 / H FAIL:** the decel decision is not the binding constraint for the factored composition even with the ego-dynamics input -> the class-conditioned progress head (LONHEAD s5) is the lever.
* **A' FAIL:** the gain is the episode-level a0 / psi distribution (C4's channel), not the per-window value: report it; no ship.

## 5. Design 2 -- NOT covered by the GO; needs its own diagnostic + pre-registration before any compute

The post-hoc rows say a0 is the dominant single decel signal and z_tac adds nothing to it for decel-vs-cruise. The speed-profile consumer needs the 6-way class, though, and a0 cannot separate FOLLOW from CRUISE or
BRAKE_TO from CREEP / HOLD. A candidate: P(decel set) from an a0-first monotone / low-dimensional head (inputs a0, v0, psi_dot0 only, target = the K-HEAD plan class or the LON6 decel set, NOT the 6-way softmax renormalised),
the within-set split from LH's z_tac head. Cheapest discriminating experiment (fresh OOF, no roll): that head's decel AUC vs `-a0` alone (0.7699) and vs A, with the same C4 / forward / constant controls,
before any GPU is spent. **Caveat to carry:** a label-level AUC gain did not translate downstream for LH (LONHEAD FAIL_F), so Design 2 also needs a roll-side bar committed in advance.

## 6. Files this arm would add / reuse

New: `lh_a0_fit.py`, `lh_a0_klabel.py` (not needed: labels unchanged), `k_head_a0.py`, `chain_lha0.sh`, `post_lha0.sh`, `lh_bars.py` extension. Reused unedited: `lh_fit.py`, `lh_klabel.py`, `k_head.py`, `l2_common.py`, `sub_fx_lhs/*`, `gpu_lock.py`.
Diagnostic inputs this draft relies on: `result/LHA0_DIAG.json`, `result/LHA0_POSTHOC.json`, `result/LHA0_EVAL_SIDE_CHECK.json`, `result/lha0_features.npz`, `result/lha0_features_eval139.npz`.
