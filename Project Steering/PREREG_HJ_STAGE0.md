# PREREG_HJ_STAGE0 — rung 0 floors, RA-0 collapse diagnostic, RA-1 refav1-HJ mini

**Committed:** 2026-10-17 (Europe/Berlin), BEFORE any of the data below exists. Plan:
`Project Steering/PLAN_V8F_REFAV1_HJEPA_2026-10-17.md`. Primary: H-JEPA `lib:2610.06805`.
Both outcomes of every hypothesis are committed here; a missed bar is reported FAILED and the named next lever runs
(RULE ZERO). Estimator everywhere: **paired episode-cluster bootstrap, B = 2,000** (`taniteval/taniteval/ci.py`),
and each read names which variance it answered (episodes / training seed / inference seed).

---

## H-V8F-FLOOR-1 — does the deployed REF-C driver beat the learned kinematic floor? (stream S1, 0 GPU)

* **Arms:** `os` = refcv7-50,400 from the banked dumps
  `E:/refcv7_eval_kit/battery/step50400_a6text/dump_s{0,1}` (inference seeds 0, 1).
* **Floors:** `P10` = an MLP on the 8 t0 kinematic features of `probe_te.py::fit_mlp_kin`, unchanged; also `damp50`
  and `kd_x` (the 30-step definition, named in the output).
* **Grid:** eval139, S2 (0–2 s) and S6 (0–6 s), every window of the banked dumps; T1 open loop.
* **P10 fit:**
  * primary: TRAIN windows, kinematics + 6 s futures only, pulled read-only from Thor's refcv8 train view;
  * fallback: **grouped 5-fold CV inside eval139**, folds by episode, every hyper-parameter fit on the fit folds only. The output states which was used.
* **Known-value controls:**
  * `P10 − P10` = +0.0000 [0, 0];
  * P10 refit on the refav1 grid reproduces 6 s ADE **2.9257 ± 0.02 m** (same code, same features).
* **Bar:** `os − P10` ADE 0–6 s < 0, CI excluding 0, at **both** inference seeds. Report the four families (STRATEGIC = PAUSED, R6 ruling) and the "acts" check (mean |a|, zero-accel fraction).
* **Outcomes:**
  * **POS** ⇒ the margin is the programme's first measured "vision helps over learned kinematics" quantity. It is registered and quoted with its estimator.
  * **NEG** ⇒ the prior is v8F's critical lever: rung α runs arm α1 first, and every LEADERBOARD "beats doing nothing" row is re-read against P10.
* **Exploratory only, never a lever claim:** a late fusion `α·os + (1−α)·P10`, with α fit by the same grouped CV.

## H-HJ-COLLAPSE-1 — is our action-deafness H-JEPA's slow-feature / advection collapse? (stream S2, RA-0, CPU)

* **Model:** refav1 `refav1-b1-v72-ep3-speed`, ckpt step **21,109** (read from `ckpt['step']`), dev-box copy
  `C:/Users/Admin/refav1_eval_slice/ckpt_ep3/ckpt.pt` (md5 `1189bc02…`).
* **Data:** 40 of the 141 v7.2 EVAL episodes, chosen by the first 40 sorted clip ids (no other selection). Eval fp8 cache `C:/Users/Admin/tanitad-data/refav1-eval141/refav1-fp8-eval`.
* **Metrics:**
  * **ρ** = E_episode[tr Var_t(z_t | episode)] / tr Var(z_t), over the adapter latent and over the operative predictor's k-step predictions. This is the within-episode variance share, paper Eq. 10.
  * **Action sensitivity** `sens_k(c)` = mean over windows of ‖ẑ_k(a with channel c swapped from another episode) − ẑ_k(a)‖₂ / ‖ẑ_k(a) − z_t‖₂. It is computed for c ∈ {all, a, steer, v} and k ∈ {1, 5, 10} operative steps. The swap donor episode is chosen by a fixed permutation (seed 0).
* **Controls (must read their known values):**
  * a swap with the window's **own** actions gives sens **0.0 exactly**;
  * an identity-predictor stub gives sens **0.0**;
  * a predictor stub that adds the action embedding linearly gives sens **> 0**. This shows the probe can see a non-zero.
* **Outcomes:**
  * **O-COLLAPSE:** steer-sensitivity at k = 10 **< 0.05**, and v-sensitivity ≥ 5× steer-sensitivity. ⇒ The mechanism is consistent. The IDM lever (α3, RA-1 A1) is the primary next step.
  * **O-NOT:** steer-sensitivity ≥ 0.10. ⇒ H-JEPA's mechanism does not explain our deafness. Next lever: **action-inferable-from-history (copycat, `2010.14876`)**; remedy = history-action dropout, tested on the tiny rig.
  * **0.05–0.10:** inconclusive. Run both levers.

## H-HJ-SENS-1 / H-HJ-K-1 / H-HJ-COST-1 — the refav1-HJ mini on pooled frozen DINOv3 (stream S4, RA-1)

* **Data:**
  * Pooled frozen DINOv3 ViT-L/16 tokens at 4×10 (primary) and 8×20 (secondary), fp8, built by `pull_encode.py` + pooling.
  * **600 TRAIN clips** first, then all 4,572.
  * Eval: the 141 EVAL clips, re-encoded with the same pooling.
  * Integrity: bit-exact re-encode on 5 clips; train/eval clip-id disjointness asserted.
* **Arms** (all with the same 6 s trajectory readout over [level latents, P10 features], P10 as the residual prior, **training seeds 0 and 1**):
  * `R0` — direct readout from the pooled tokens, no WM (the frozen-encoder ceiling);
  * `A0` — 2-level H-JEPA (Δt 0.2 s, s₂ = 3 ⇒ 0.6 s), SIGReg (T,N,D) only;
  * `A1` — A0 + IDM at both levels, a 2×2 grid β ∈ {0.04, 0.08} × γ ∈ {50, 100} at seed 0, with the best cell at seed 1;
  * `A2` — best A1 + level 3 (s₃ = 3 ⇒ 1.8 s);
  * `A1x` — best A1 with **time-shuffled targets**: a deliberate regression that must lose to A1 on K.
* **Bars:**
  * **SENS:** A1 level-1 steer-sensitivity (definition above, k = 10) ≥ 0.10 and separated from A0 at both seeds.
  * **K:** `best(R0, A1, A2) − P10` ADE @ 6 s < 0, separated at both seeds, \|Δ\| > 2× the training-seed floor \|A1_s0 − A1_s1\|; event strata reported (GT turn onset, stop, lead braking).
  * **COST:** −Spearman(level-2 distance to the +6 s latent, time) > the level-1 value, paired over episodes.
* **Outcomes:**
  * SENS PASS + K PASS ⇒ frozen + adapter suffices; build refav2 (RA-2).
  * SENS PASS + K FAIL ⇒ the frozen ceiling is reached; next arm = the same with **LoRA r 32** (T-B).
  * SENS FAIL ⇒ raise γ to 200, IDM at level 2 only, then a time-repulsion floor (|λ_sim| ≤ 0.06), in that order.
  * R0 alone passing K while A1 does not ⇒ the WM objective is hurting the readout. Report it, and move the WM to v8F's chassis question.
  * **A1x not losing to A1 ⇒ the instrument is broken; nothing else is read until it is fixed.**
