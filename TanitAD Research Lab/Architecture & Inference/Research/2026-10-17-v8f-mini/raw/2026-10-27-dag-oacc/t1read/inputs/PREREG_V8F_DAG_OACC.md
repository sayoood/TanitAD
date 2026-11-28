# PREREG_V8F_DAG_OACC: OACC's acceleration term inside DAG's fine-tune recipe

Written 2026-10-27 by the Master Mind. It comes after `RESULT_V8F_DAG_HACC` (COSTS POSITION) and **before any DAG-OACC weight value, gate,
run or row**. Claim row `H-V8F-DAG-OACC-1`.

**Why it runs.** `RESULT_V8F_DAG_HACC`'s committed next, verbatim: *"OACC's acceleration-matching term added to DAG's fine-tune recipe (box 2,
PI go), under its own PREREG"*.
* Plain DAG against P10 at T1-kin (`RESULT_V8F_DAG_REREAD.md` §1) reads separated-worse on the longitudinal rows accel MAE (+0.807 / +1.444)
  and DK-VIOL (s0 +0.0353).
* The zero-training post-map cleared every longitudinal row against P10 and failed KEEP at s0 (+0.1264 [+0.0522, +0.2004] m against DAG).
* OACC on M2b (`RESULT_V8F_OACC.md`, open-loop tier, s0 only) moved accel MAE by −0.5663 m/s² (separated) and failed the same KEEP bar
  (upper bound +0.1262 m).
* No arm has had the trained term **and** own-state supervision together. That is this arm.

**PI go, 2026-10-27, verbatim:** *"yes queue it"*, in answer to *"Queue DAgger iteration 2, then DAG-OACC, then FLAT-DAG on box 2, each
through its launch gate?"*

## 1. The arm (ONE lever)

* **DAG-OACC_s** = DAG_s's training argv **verbatim** (`v8f-dagger/M2b_s{s}_dag/argv.json`: `--init-ckpt` M2b_s, 3,000 steps, lr 1e-4,
  warmup 100, `--p-gt-start` / `--p-gt-end` 0.2, `--ego-perturb-p 0`, `--dagger-p 0.1`, the iteration-1 bank `dagger_bank/s{s}` with the same
  sha256, `--dagger-seed s`) **plus `--w-acc W`**, and nothing else.
* **Its control is DAG_s itself**, the banked run. The pair differs in exactly one flag. No new control is trained.
  * Run-to-run noise is read against F (F/2 = 0.0604 m), as KEEP was for DAG-HACC.
* **W is OACC's rule, verbatim** (`PREREG_V8F_OACC.md` and its Amendment 1, R1–R3), with **B = DAG**:
  * $w_{acc} = 0.5\cdot\mathcal{L}_{traj}(DAG_{s0}) / \mathcal{L}_{acc}(DAG_{s0})$, over HBLEND's fit windows (141 TRAIN clips, 9,582 windows);
  * $\mathcal{L}_{traj}$ is the trainer's own trajectory term (R1), and $\mathcal{L}_{acc}$ is OACC's term at unit weight (R2);
  * rounded to 6 significant figures (R3); one value for both seeds, as OACC.
  * The value is computed **once, by the box-2 `waccoacc` stage on DAG_s0's checkpoint**, and is written into both argvs. The result quotes
    that value.
  * ⚠ **Disclosed alternative, not used:** OACC's M2b value, 42.9311. The rule names B's checkpoint, and B here is DAG.
* ⛔ The weight rule, the mask, the bars and the seed order do not change after any DAG-OACC number is seen. The weight is never re-tuned.

## 2. The read

At T1-kin with DAG's decisive settings (m = 10, causal bucket, dead-reckoned nav, logged scene, self-action OPEN loop). Eval-141, paired
episode-cluster bootstrap, B = 2,000, per seed. F = 0.1207 m. Plain arms, no post-map.

| read | definition | bar |
|---|---|---|
| **PRIMARY-ACC** | accel MAE, DAG-OACC − DAG | < 0 and separated, at both seeds |
| **KEEP** | ADE6, DAG-OACC − DAG | upper bound ≤ +F/2 = +0.0604 m, at both seeds |
| **LON-CLEAR** | every LONGITUDINAL row of Read A, DAG-OACC − P10 | none separated-worse, at both seeds |
| **Read A** | `PREREG_V8F_DAG_REREAD.md` §2 verbatim (rows, D1 / D2 / D3, crawl rows, readers, estimator), arm = DAG-OACC | as there |
| beside | DAG-OACC − DAG on every Read-A row; the four families at T0; S-POS against CTRV; R-T0 (T0 ADE6, DAG-OACC − DAG) | descriptive |

## 3. Controls (each must pass, or the read is REFUSED)

* **C-O1, one lever:** DAG-OACC's `argv.json` equals DAG's except `--w-acc` and the output paths. The bank sha256 is identical.
* **C-O2, the lever runs:** the trainer's launch gate shows the acceleration term finite and non-zero, and a non-zero gradient on O's head
  from it (OACC's G-LIVE entry). The run's record carries the built w_acc, equal to the argv's.
* **C-O3, the weight:** the box-2 value equals an offline recomputation on DAG_s0's banked fit plans (`daghacc/plans/fitplans_DAG_s0.npz`)
  to within 1e-4 relative. A mismatch is REFUSED; it is never resolved by picking one.
* **C-O4:** Read A's C1–C4, re-run on the new arm.

## 4. Outcomes, committed now

Applied in this order: REFUSED → NO EFFECT → COSTS POSITION → DOMINANT → LONGITUDINAL FIXED → NOT FIXED.

| outcome | condition | committed next |
|---|---|---|
| **REFUSED** | a control fails | fix and re-run |
| **NO EFFECT** | PRIMARY-ACC fails at ≥ 1 seed | the accel error is not reachable through O's loss on DAG's states at this capacity. Next: OACC's NO EFFECT next, an O-capacity read, under its own PREREG |
| **COSTS POSITION** | PRIMARY-ACC passes; KEEP fails at ≥ 1 seed | not adopted. The trade has then appeared on three routes. Next: the same term restricted to the plan's along-track component, so that lateral smoothing cannot cost position in curves (HYPOTHESIS), under its own PREREG |
| **DOMINANT** | PRIMARY-ACC and KEEP pass; Read A's D1, D2 and D3 at both seeds | the dominance claim at T1-kin over the kinematic floor, on all four families, at both seeds, is registered for DAG-OACC. Signal the PI. Evidence videos are re-rendered on DAG-OACC. FLAT-DAG stays the attribution read |
| **LONGITUDINAL FIXED** | PRIMARY-ACC and KEEP pass; LON-CLEAR at both seeds; a lateral or crawl row still separated-worse | the term is the trained longitudinal lever and is adopted. It composes with DAgger iteration 2: next is DAG2's recipe plus this W (DAG2-OACC), read on Read A, as `PREREG_V8F_DAGGER2`'s FIXED / PARTIAL rows commit |
| **NOT FIXED** | PRIMARY-ACC and KEEP pass; a longitudinal row still separated-worse at ≥ 1 seed | next: HACC's rule re-fitted on DAG-OACC (`PREREG_V8F_DAG_HACC`'s recipe, zero training), read with KEEP against DAG-OACC |

## 5. Queue and reporting

* **Box 2**, second of three (after DAGGER2's chain, before FLAT-DAG): the `waccoacc` stage on DAG_s0, then both gates, then train and T0-eval
  s0, then s1 (ESTIMATED ~2 h).
* **Dev-box CPU:** the 2 T1-kin rollouts (DAG's are banked) and the reader.
* Per seed: delta [lo, hi], separated, arm means and n, with the tier stamp. STRATEGIC is reported; it is not paused for v8F.
* Package `raw/2026-10-27-dag-oacc/`. Result `RESULT_V8F_DAG_OACC.md`. The register row is updated in the same landing.
