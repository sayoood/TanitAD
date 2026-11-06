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
