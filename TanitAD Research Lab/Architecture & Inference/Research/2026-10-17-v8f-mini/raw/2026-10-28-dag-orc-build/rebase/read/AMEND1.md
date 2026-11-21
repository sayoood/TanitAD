
## Amendment 1 (A1-2026-10-27): re-based on DAG-OACC, before any DAG-ORC run (launch precondition 4)

Written 2026-10-27 ~23:10 by the Master Mind.

**State when written.**
* No DAG-ORC run, gate or row exists.
* The PI's go of 2026-10-27, verbatim, was *"Yes, queue DAG-ORC on box 2 after the SPEEDREAD pull, through its launch gate."* It was
  asked as ~3 h of box 2. The dev-box waiter that would have started the DAG-based chain was stopped by the Master Mind before it fired
  (`box2_ship/logs/dagorc_waiter.out`). Nothing reached box 2.

**Why.** Launch precondition 4 (§10): *"If a read adopts another base (for example DAG-OACC) before A's gate, this PREREG is not launched as
written."*
* `PREREG_V8F_DAG_OACC` read **LONGITUDINAL FIXED** under its Amendment 1. The read is `raw/2026-10-27-dag-oacc/read/DAGOACC_READ.json`
  (sha256 `9e6bd4f6…`), with 60 / 60 controls PASS. The unamended table gives the same word.
* Its committed next adopts DAG-OACC as the base and sends the open crawl rows to this arm, re-based on DAG-OACC.
* The open rows at both seeds, DAG-OACC − P10, all separated-worse: yaw-rate MAE, curvature MAE, L2, M1.

### 1. The arms, re-based (the COMPOSE form)

| arm | what | role |
|---|---|---|
| **A** | DAG-OACC_s's training argv **verbatim** (`v8f-dagoacc/M2b_s{s}_dagoacc/argv.json`: init M2b_s, 3,000 steps, `--dagger-p 0.1` on the iteration-1 bank, `--w-acc 49.6969`), plus the perturbation at §1's law (`--ego-perturb-p 0.3 --ego-perturb-vscale 0.5 1.0 --ego-perturb-vmax 8.0 --ego-perturb-adrop 1.5 --ego-perturb-yawrate-std 2.0 --ego-perturb-seed <s>`), plus `--ego-perturb-with-dagger`, with its own `--out` | decisive |
| **CTRL** = DAG-OACC_s | **banked, reused.** Checkpoints `67265ec1…` (s0) / `90b01474…` (s1); T1-kin dumps `v8f-dagoacc-t1/` | the one-lever control: A minus the perturbation |
| **REP** | DAG-OACC_s's argv with `--seed 10` / `--seed 11` (init M2b_s0 / M2b_s1), with its own `--out`. Trained new | the replicate floor RF |
| **B** | A with `--ego-perturb-p 0.1` | the dose fallback. Trained **only on A's HARMS, with its own PI go** |

* **Why this form, and not "a newly trained control" from DAG-OACC's checkpoint.**
  * Precondition 4 asked for a new control because DAG2-0 is valid only for DAG. Here the base's own banked run is the control. It
    differs from A in the lever alone, the way DAG-OACC itself was read against DAG.
  * So no control is retrained, and only REP is new.
  * It also removes the adoption handicap that §4 disclosed. A no longer starts behind its base: A and DAG-OACC are trained the same way
    from M2b, except for the perturbation.
  * The alternative was rejected. It fine-tunes 3,000 GT-only steps from DAG-OACC's checkpoint, with a new control and a new replicate:
    6 runs, ~4.3 h, outside the ~3 h asked. It would carry DAG2-0's measured handicap (+0.27 / +0.13 m ADE6 against DAG).
* **The composition is a build item and a LAUNCH BLOCKER.**
  * The trainer refuses `--dagger-p > 0` together with `--ego-perturb-p > 0` (`v8f_mini_train.py`, the `PREREG_V8F_DAGGER` §4
    exclusivity, written for attribution inside the DAGGER arm).
  * A new **default-off** flag, `--ego-perturb-with-dagger`, lifts that refusal for an argv that names it. Semantics, fixed now:
    * the swap is drawn first;
    * the perturbation is then drawn, at its own p and from its own seeded generator, **only on samples the swap did not replace**;
    * no sample is ever both swapped and perturbed.
  * With the flag off, every existing argv must build and train bit-identically (K-BUILD).
  * Proofs before any gate, per the CLAUDE.md rule of 2026-09-10:
    * a G-DVB entry for the flag;
    * G-LIVE on a composed batch: both levers reach the loss gradient, and the ego probe's p = 0 control reads exactly 0.0;
    * a mutation (delete the perturbation's apply on the composed path ⇒ RED);
    * the run record carries `dagger_n / dagger_eligible` in 0.1 ± 0.02, `ego_perturb_n / ego_perturb_eligible` (non-swapped eligible) in
      0.3 ± 0.02, and an overlap count of **exactly 0**. REP reads ego 0 exactly and swap 0.1 ± 0.02.

### 2. What every section now reads

* **"DAG" means DAG-OACC, and "CTRL" means DAG-OACC,** in §3, §4, §6 and §7. This covers R-KEEP, R-NOBASE, the domain D_s and the beside
  rows.
* **R-ADE and R-KEEP now share one contrast**, A − DAG-OACC. R-KEEP keeps its point bar (≤ +F/2 = 0.0604 m), and R-ADE keeps "not
  separated-worse". The same holds for R-NEW and R-NOBASE. Both coincidences are disclosed; no row is dropped.
* **The crawl set C** is the set of rows separated-worse for DAG-OACC − P10. At both seeds that is yaw-rate MAE, curvature MAE, L2 and M1.
* **The domain D_s** is every Read-A row not separated-worse for DAG-OACC − P10, with **no exempt rows**.
  * accel MAE and DK-VIOL were exempt because *"they belong to DAG-OACC"*. DAG-OACC has now cleared them, so they are guarded here.
* **The floors.**
  * RF0 is unchanged: P-0 is frozen at `raw/2026-10-27-dag-orc-read/P0_RF0.json`, sha256 `1608d70b…`.
  * RF_r = max over s of |d_r,s(REP − DAG-OACC)|.
  * F and bar_r are as in §3.
* **K-PAIR.**
  * argv(A_s) equals argv(DAG-OACC_s) except `--out`, the six perturbation flags at §1's values and `--ego-perturb-with-dagger`.
  * argv(REP_s) equals argv(DAG-OACC_s) except `--out` and `--seed`.
  * The bank sha256 is DAG-OACC's.
* **K-INIT:** M2b_s, as recorded in DAG-OACC's launch-gate token.
* **K-BUILD:** against `aeb54468`, the commit DAG-OACC trained at, under §6's two conditions. The new flag's hunks pass a bit-identity
  test with the flag off.
* **K-RC:** the reader, with the floors at 0 and DAG-OACC in the base slots, reproduces `DAGOACC_READ.json`'s Read-A table for
  DAG-OACC − P10 with 0 differences, as well as DAG2's rows as before.

### 3. Outcomes (§7's table, the same conditions in the same order, re-based)

* **FIXED / ADOPTED:** adopt X as the base and re-run Read A on X. DOMINANT there is the dominance claim. ("Then X with DAG-OACC's lever"
  is already done.)
* **FIXED / NOT ADOPTED** and **PARTIAL / NOT ADOPTED:** DAG-ORC-SWAP cannot apply, because A already keeps the swap on. Next is **B (p
  0.1) under this table**: a lower dose that may keep the base. It needs its own PI go.
* **HARMS** of A ⇒ B, as written. HARMS of B ⇒ NEXT-CRAWL.
* **NEXT-CRAWL** (§8) runs on DAG-OACC's states.
  * N0 already read NOT NAVMOVE.
  * COMPOSE is done: DAG-OACC − DAG is not separated-better on the rows of C.
  * So its branch is **LAUNCH-INTENT**, with DAG-OACC's recipe in place of "DAG init".
* **Every other row is unchanged.** The 524,288-cell enumeration stays complete and exclusive, because no condition changed, only its
  base.

### 4. Cost and queue

* A + REP is 4 runs, ≈ 2.8 h. MEASURED basis: DAG-OACC's two runs took 41 and 42 min with their T0 evals (`chain_v8f_dagoacc_r2.out`).
  This is within the ~3 h of the PI's go.
* No bank stage: the iteration-1 bank is reused at DAG-OACC's sha256.
* B is not queued.
* Dev-box CPU: the T1-kin rollouts of A and REP at both seeds. CTRL's and P10's runs are banked.
