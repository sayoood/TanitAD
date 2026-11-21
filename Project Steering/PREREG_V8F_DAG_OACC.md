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

## Amendment 1 (A1-2026-10-27): before any DAG-OACC T1-kin rollout or value

Written 2026-10-27 by the Master Mind.

**State when written.**
* Both DAG-OACC runs are trained on box 2 and pulled to the dev box, with their T0 evals (`ZZV8FMINI-DONE` 21:51).
* No T1-kin rollout of DAG-OACC has run.
* No DAG-OACC metric value, T0 or T1-kin, has been opened by the Master Mind, the reader or the chain. The reader's dry run hashed the
  pulled files and did not read them.

**Why.** The reader's cell enumeration (`raw/2026-10-27-dagoacc-amend1/enumerate_cells_before.out`) shows two defects in §4. First, the table
leaves **255 of 524,288 cells** unplaced. Second, two clauses each admit two readings. This amendment fixes both before any row exists. Rows,
bars, controls, the order, the weight rule and W = 49.6969 do not change.

1. **The table is made complete.** LONGITUDINAL FIXED's condition reads: *PRIMARY-ACC and KEEP pass; LON-CLEAR at both seeds; not
   DOMINANT.*
   * The unplaced cells are cells where PRIMARY-ACC, KEEP and LON-CLEAR hold, no lateral or crawl row is separated-worse, and DOMINANT fails
     because D1 or D3 is short or a TACTICAL or STRATEGIC row is separated-worse.
   * In each of them the lever did its longitudinal job and the arm is not dominant, so they join LONGITUDINAL FIXED.
   * The result names, per seed, which of DOMINANT's clauses failed.
   * Clause (a) asked whether a lateral or crawl row had to be worse at one seed or at both. It then decides nothing and is dropped.
2. **DOMINANT needs the crawl rows to pass.** DOMINANT reads: *PRIMARY-ACC and KEEP pass; Read A's D1, D2 and D3 at both seeds; and Read
   A's crawl rows pass (L2 and M1 not separated-worse at either seed).*
   * Reason: this table's own LONGITUDINAL FIXED row names *"a lateral or crawl row still separated-worse"* as a state that is not dominant.
     Read A's D2 excludes M1, so as written a cell with M1 separated-worse reads DOMINANT before LONGITUDINAL FIXED is reached.
   * The programme's goal is dominant driving, and a launch crawl is a driving defect.
   * A cell with D1–D3 held and the crawl open therefore reads LONGITUDINAL FIXED, reported as *"D1–D3 held, crawl open"*.
   * Read A's own table (`PREREG_V8F_DAG_REREAD` §5) calls that state DOMINANT, CRAWL OPEN. This PREREG does not register a dominance claim
     while the crawl is open.
3. **LONGITUDINAL FIXED's committed next.** As written it named DAG2-OACC, *"as `PREREG_V8F_DAGGER2`'s FIXED / PARTIAL rows commit"*.
   * DAGGER2 read **HARMS** (`RESULT_V8F_DAGGER2.md`, landed at `d64750bd`, after this PREREG was written). HARMS is neither FIXED nor
     PARTIAL, so the condition of that route is not met.
   * The committed next reads: *the term is adopted and DAG-OACC becomes the base; next is the lever of the row still open.*
     * If the open rows are crawl rows (L2, curvature, yaw-rate, M1; the list `PREREG_V8F_DAG_REREAD` §5 uses), the lever is DAG-ORC
       (`PREREG_V8F_DAG_ORC.md`), re-based on DAG-OACC by an amendment of its own PREREG before any DAG-ORC data.
     * Otherwise, the failing family's lever, under its own PREREG.
4. **The reader codes this amendment.** Its enumeration must read **0 unplaced** cells, every row reached, and decide() equal to an
   independent table on every cell. The unamended outcome is computed beside, as a descriptive row that decides nothing.
