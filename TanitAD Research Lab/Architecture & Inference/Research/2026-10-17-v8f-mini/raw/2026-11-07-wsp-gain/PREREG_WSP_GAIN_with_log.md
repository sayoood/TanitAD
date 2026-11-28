# PREREG_WSP_GAIN: a weight-space edit that restores the kinematic base's longitudinal gains in v8F's residual (Master Mind, 2026-11-06, written before any edit is built or scored)

## 1. The case ([M], `wsp_read/WSP_READ.md`, `P2B_RESULT.json`, `P2B_s1_RESULT.json`; C0, eval-141 tick 0, 586 windows / 70 episodes at 3–8 m/s)
* **The base carries the speed level with the physics gains.** v8F's plan is the non-learned kinematic base (kd_x) plus a frozen prior net plus the learned residual head. The base's gains are d s/d v0 = 1.00 and d s/d a0 = 1.90.
* **The learned parts shrink those gains:**
  * to 0.81 / 1.27 at s0 and 0.76 / 1.11 at s1;
  * and they shift plan speed by about −0.35 m/s.
* **The prior explains R² 0.87 / 0.82 of plan speed.**
* **This is T1′'s acceleration shrinkage, located in the weights** (`THEORY_CRAWL_AMEND_1.md`, frozen `9d39ac53`).
* **What P1 measured at T2:** under-acceleration against the human of −1.25 [−1.69, −0.83] at s0 and −0.62 [−1.30, +0.20] at s1 (post-hoc, `P1B_RESULT.json`).
* **Where the edit idea comes from [P].** The idea of removing an obstacle by editing one direction, without training, is Takehi et al., 2026-11-02 (Library `takehi2026weightspace`, a blog post, unreviewed).

## 2. The arm (one lever: restore the base's longitudinal response inside the learned residual)
* **E-GAIN(α), on C0_s for s in {0, 1}:** a closed-form edit of the LAST linear layer of O's residual head (`head_o`, the coefficients of the longitudinal channel only). The lateral coefficients are bit-for-bit unchanged.
* **The edit has three parts:**
  * **w:** the ridge-regression direction (λ = 1e-3 · trace of the hidden covariance) that decodes the measured a0 and v0 from `head_o`'s input hidden h;
  * **u:** the coefficient vector that changes plan speed at +2 s by exactly 1 m/s through the fixed basis;
  * **ΔW = α · u · (Δg_a · w_a + Δg_v · w_v)ᵀ and Δb = α · u · 0.35 · κ.** Here Δg = (base gain − measured model gain), and κ makes the band-mean shift exactly the measured −0.35 m/s.
* **Fitted on TRAIN windows only:** a fixed random subset of 20,000 windows with v0 in 3–8 m/s from the 4,512 trained clips, seed 0. No eval-141 window is used to fit, and the measured gains used are the train subset's own.
* **Doses:** α ∈ {0.5, 1.0}, fixed now.
* **Admissibility:**
  * the inputs are unchanged: vision plus measured ego history, the PI's ruling;
  * no logged-future or oracle channel;
  * no situation-classifier output reaches the goal or the plan.

## 3. Reads, in order
1. **Mechanism check (T0-class, CPU):** the edited model's d s/d a0 and d s/d v0 on eval-141 at 3–8 m/s are within 10 % of α-restored targets. Otherwise the edit is not the lever named, and nothing is read.
2. **T1-kin on eval-141 (dev-box CPU, zero training):**
   * the instrument and rows of the LF / A-series reads: the four families, M1, the paired episode-cluster bootstrap B 2,000 seed 0, both seeds, against C0 and against P10;
   * the excess-closure share as `share_closed`.
3. **T2 (box 2, one job, safe mode):** only if the T1-kin read is not HARMS, i.e. it does not have more separated-worse guard rows than C0.
   * The edited composite replaces C0C in the stage-4 driver, on the same 48 R-SCN scenes and the same starts.
   * Read by the pinned dominance read with the arm name changed, needing a re-pin that takes the arm as an argument (the same re-pin `DRAFT_PREREG_T2_DOM_LORA.md` needs).

## 4. Predictions and outcomes, committed now
* **The theory's prediction (T2, frontier) [H]:** E-GAIN is an admissible downstream lever. It lands on the admissible chain, moving M1 the right way and paying in guard rows (DK-VIOL, TAC-LON, ADE6 at α = 1.0).
  * **The prior for DOMINANT-T2 is LOW [E].** A DOMINANT would contradict T2 and count as a result all the same.
* **Outcome table:**

| T1-kin read | next, executed in the same turn |
|---|---|
| MECH-FAIL (read 1 misses) | the edit is wrong, not the theory: name the defect and rebuild the edit; no T2 stage |
| HARMS (more separated-worse guards than C0) | the frontier point is logged in P3's chain; no T2 stage; the perception lever (LoRA) stays the route |
| MOVES-M1, guards hold (no new separated-worse row) | the T2 stage on box 2; DOMINANT-T2 there is the scale signal, with video |
| NO-EFFECT (M1 not separated) | the shrinkage is not the crawl's lever at T1-kin; logged; the LoRA route stays |

## 5. What is pinned before the build
This card's sha256 is in `FROZEN_SHA_PREREG_WSP_GAIN.txt`. The edit code, its tests (a mutant edit with α = 0 must reproduce C0 bit for bit, and a lateral coefficient change must turn a test RED) and the train subset's window ids are pinned in a dated log entry below, before any eval-141 window is scored.

## 6. Log (append-only, dated)
* 2026-11-07 00:42 (dev-box clock, from `date`), Master Mind. FOURTH PIN of `t2v8f/code/read_t2_dominance.py`: sha256 `7fd65ffdf62ae74f5609e3ac259588826bb84b51e3a3a84968b299332a3f35f9` (AGENT Q; third pin `d6664718` kept as `read_t2_dominance.pin3.py`). It adds `--arm NAME` (default C0C) and `--arm-stage N` (default 4), plus `--arm-result`, `--arm-root`, `--arm-exclusions`, `--arm-overlay` and `--arm-gate-log` for a stage beyond 4. The union of R-SCN exclusions over every stage read leaves every arm. **Identity, binding: with the defaults it reproduces the read of record exactly** (`logs/q_identity.json`, identical: true; decision, gates, both seeds' 19 rows, seed gap, r_scn and the md equal; only the script sha, the argv and the append-only card's whole-file sha differ, the frozen prefix intact). Tests: 14 new and 57 existing green; 80 of 80 mutations RED on the working copy. The full-suite close-out is in flight (`logs/Q_FINISHED.txt`). Entered before any stage-5 cell exists.

### 2026-11-07 PIN (WSP-EDIT agent). Written after the edit was built and fitted on TRAIN windows only, before any eval-141 window was scored through an EDITED checkpoint. The text above this log is unchanged (sha256 92d61dcb... verified at the start of the run).
* **One earlier touch of eval-141, disclosed:** 64 eval-141 windows (3-8 m/s) were passed through the UNEDITED C0 s0 to check that the measuring code reproduces P2b (`P2B_PERWINDOW.npz`): plan speed max abs difference 2.4e-6 m/s, finite-difference gains equal the autograd gains to 2.5e-4 (v0) and 4.3e-3 (a0) median abs. No edited checkpoint existed then.
* **Code (sha256), folder `dagland/motion_c/theory/wsp_edit/`:** `wsp_edit.py` ad4cda1b..., `wsp_measure.py` f326427a..., `wsp_common.py` a30ad73a..., `fit_and_write.py` 6d71f5ce..., `build_train_ids.py` aed4698a..., `train_measure.py` fc617ca0..., `tests/test_wsp_edit.py` f6bc0836..., `run_mutations.py` 73664703..., `read1_eval.py` 21aab748..., `wspedit_read.py` 3801f17b..., `read2_chain.sh` 6af8a1ec... (full hashes in `raw/PIN_HASHES.txt`).
* **Tests:** 9 passed (`logs/pytest_wsp_edit.log`). They cover: alpha = 0 reproduces C0 byte for byte (real C0 s0 state dict); at alpha = 1 the lateral rows `head_o.2` nb:2nb and every other tensor are byte-identical and the input state dict is not mutated; four mutants of the edited state (one lateral weight, one lateral bias, an all-rows edit, an unrelated tensor) turn the lateral test RED; the kappa rule gives a band-mean shift of exactly 0.35 alpha on the fit windows; the wrong-u-sign mutant fails read 1 on a synthetic world where the correct edit passes. 9 of 9 deliberate mutations of the edit code turn a test RED (`logs/mutations.log`, "MUTATIONS ALL RED"). On the REAL models and 192 TRAIN windows (`raw/MUTANT_REAL_U_SIGN.json`) the wrong-u-sign mutant FAILS read 1 at both seeds (alpha 1.0: dS/dv0 0.626 / 0.588, dS/da0 0.679 / 0.558 against targets 1.0 / 1.899).
* **Fit set:** `raw/train_window_ids.txt` sha256 `948a2a1d68e91764965baa408fe5aab309ff5dd85955e8be7f740abae108812b`: 20,000 TRAIN windows with v0 in [3, 8), seed 0 (`np.random.default_rng(0).choice` over the pool of 87,942, sorted), 2,588 of the 4,512 trained clips; the canonical 306,566-window list was rebuilt and matched its banked sha `e144adcd...`. Train features: `F:/TanitAD-scratch/speedread/cache/train` (byte-equality with box 2 `cache_full` UNVERIFIED); train join rebuilt by the trigger probe (not compared with box 2's `join_train.npz`, UNVERIFIED).
* **Fitted values (`raw/FIT_RESULT.json`, sha256 a62f10e0...; train-subset medians):**
  * s0: model dS/dv0 0.7705, dS/da0 1.2080; base 0.9995, 1.8989; **dg_v +0.2290, dg_a +0.6908**; m -0.1221, **kappa +1.3489**; ridge decode R^2 of v0 0.547, of a0 0.919; train mean s_res -0.380.
  * s1: model dS/dv0 0.7536, dS/da0 1.1310; **dg_v +0.2459, dg_a +0.7679**; m -0.3410, **kappa +1.9743**; decode R^2 of v0 0.521, of a0 0.937; train mean s_res -0.410.
  * Edited checkpoints (`v8f-wspedit/M2b_s{s}_wspedit_a{05,10}/ckpt_last.pt`, sha256 prefixes): s0 a05 9e86ef02, a10 6d90f4d0; s1 a05 fab455d9, a10 ee80e090.
* **Readings I fixed (the Master Mind may overrule; none moves a number):**
  1. h is the INPUT of head_o's last linear layer (post-GELU, 512-d), because the edit acts on that layer. u = r_res / (r_res . r_res), so r_res . u = 1 exactly. Delta-g uses train-subset MEDIANS. kappa = 1 - m / 0.35 with m the fit-window mean of h . (dg_a w_a + dg_v w_v), so the edit's band-mean plan-speed shift over the fit windows is exactly +0.35 alpha m/s (it undoes the learned parts' -0.35 m/s); 0.35 is the card's literal.
  2. **Read 1:** target = (1 - alpha) g_C0 + alpha g_base, both measured on the eval-141 3-8 m/s windows (586, the P2b set) by central finite differences (eps 0.05) with the ego chain rebuilt. Pass = relative error <= 10 % for dS/dv0 AND dS/da0 in all four (seed, alpha) cells; any miss is MECH-FAIL for the arm. A second target, g_C0(eval) + alpha * dg(train), is reported beside.
  3. **Train-side foreknowledge, disclosed:** on the 192 train windows of the mutant test, the alpha 1.0 edit reached dS/dv0 0.911 (s0) and 0.897 (s1) against the target 1.0, and dS/da0 1.777 and 1.781 against 1.899. So s1 sits at 10.3 % on dS/dv0, at the edge of the 10 % rule. The cause is the ridge decode of v0 from h (R^2 0.52 to 0.55), which the card's recipe fixes.
  4. **Read 2 outcome rules (card section 4):** per dose, HARMS if at either seed a non-M1, non-STRATEGIC row is separated-worse in WSP - RIC0, or separated-worse in WSP - P10 while not separated-worse in RIC0 - P10 (a new separated-worse row); else MOVES-M1 if M1 of WSP - RIC0 is separated-better at BOTH seeds; else NO-EFFECT (a note when separated-better at exactly one seed). RIC0 is C0 on the `reindexed` scene with the same flags. STRATEGIC is PAUSED (PI item 25): its rows are listed beside, never decisive. The deciding scene is `reindexed`; two doses and two seeds are read, and a counted outcome names its dose and says so.
* 2026-11-07 01:32 (dev-box clock, from `date`), Master Mind. **READ 2 (T1-kin reindexed, eval-141; paired episode-cluster bootstrap B 2,000 seed 0): HARMS in all four cells** (`wsp_edit/raw/READ2.json`, `RESULTS_READ2.md`). M1 moves separated-better in every cell (alpha 0.5: -0.0095 / -0.0139; alpha 1.0: -0.0132 / -0.0146; share_closed 0.178 / 0.233 / 0.247 / 0.246), and every cell adds separated-worse guard rows (ADE6, DK-VIOL, along-track, cross-track, curvature and heading MAE at all four; TAC-LON and speed/accel MAE at alpha 1.0). By section 4: the frontier point is logged in P3's chain, there is NO T2 stage, and the perception lever (LoRA) stays the route. This is the outcome section 4 predicted from the frontier argument (T2) before the edit was built. STRATEGIC PAUSED (PI item 25).
