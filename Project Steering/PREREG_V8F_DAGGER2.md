# PREREG_V8F_DAGGER2: DAgger iteration 2, on DAG's own states, aimed at the crawl rows

Promoted 2026-10-27 by the Master Mind from the draft banked at `5eaa0232` (`raw/2026-10-27-dagger2-draft/PREREG_V8F_DAGGER2.md`). It is
written after `RESULT_V8F_DAG_REREAD` and `RESULT_V8F_DAG_HACC`, and **before any DAGGER2 bank, run or row**. Claim row `H-V8F-DAGGER2-1`.

**Why it runs.** `PREREG_V8F_DAGGER`'s ROBUST row, registered before any DAgger data, commits it: *"If the crawl beside rows (M1, L2) still
fail, next is DAgger iteration 2 (DAG's own states), under its own read"*. Read A measured M1 and L2 separated-worse at both seeds. Read A's
committed next sends the crawl rows here.
* ⚠ **Disclosed change from the draft.** The draft's run condition (*"only crawl rows failing"*) was narrower than that registered clause and
  does not fire, because accel and DK-VIOL also failed. The registered clause governs. The longitudinal rows are handled by `PREREG_V8F_DAG_OACC`
  (DAG-HACC's committed next), so this arm's decisive set is the crawl rows only.

**PI go, 2026-10-27, verbatim:** *"yes queue it"*, in answer to *"Queue DAgger iteration 2, then DAG-OACC, then FLAT-DAG on box 2, each
through its launch gate?"*

**What it is.** Everything is `PREREG_V8F_DAGGER.md` with Amendments 1 and 2, **verbatim**: the bank, the matched route point, sample
construction, controls K-B0…K-B5 and COVERAGE, the reader's controls, the estimator and the tiers. Only the changes below differ.

## The changes

1. **Generator (§1):** `M2b_s{seed}_dag/ckpt_last.pt` (sha256 from `v8f-dagger/M2b_s{seed}_dag/SHA256SUMS.txt`), PLAIN, with the same
   rollout configuration: m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, executor `plan`. The same 4,512 TRAIN clips and
   306,566 windows; ticks 1–2. Each seed's bank comes from that seed's DAG.
2. **Arms (§4):**
   * **DAG2** = DAG_s{seed} fine-tuned 3,000 steps with `--dagger-p 0.1 --dagger-bank <DAG2 bank> --dagger-bank-sha256 <its sha>
     --dagger-seed <seed> --ego-perturb-p 0 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2`, with `--init-ckpt` = DAG_s{seed}.
     Every other flag is DAG's argv.
   * **DAG2-0** = the same argv with `--dagger-p 0`.
   * Both seeds, each through the trainer's launch gate.
3. **The replicate floor (§5):** no same-argv replicate of DAG2-0 is trained. RF is fixed at DAGGER's measured values: lat 0.0067 / 0.0043, lon
   0.0251 / 0.0139. Bars are max(2F, 2·RF), inclusive: lat 0.0248 / 0.0248, lon 0.0502 / 0.0278. ⚠ It is an estimate from another init with
   the same recipe, disclosed as such.
4. **The decisive rows**, at T1-kin with DAG's decisive settings. Readers: Read A's (`PREREG_V8F_DAG_REREAD` §2), unchanged. Plain arms (no
   post-map).
   * **The crawl set C:** the rows that read separated-worse for plain DAG − P10 at that seed (`RESULT_V8F_DAG_REREAD.md` §1):
     * s0: curvature MAE, L2, M1;
     * s1: curvature MAE, L2, M1, yaw-rate MAE.
   * **R-CRAWL:** every row of C reads **not separated-worse** for DAG2 − P10 at its seed.
   * **R-MOVE:** every row of C is **separated-better** for DAG2 − DAG2-0 at its seed.
5. **Guards** (each at both seeds):
   * R-TLAT and R-TLON (DAG2 − DAG2-0) not separated-worse;
   * R-ADE (T1-kin ADE6, DAG2 − DAG2-0) not separated-worse;
   * R-T0 point ≤ +F/2 = 0.0604 m;
   * no Read-A row that was not separated-worse for DAG − P10 becomes separated-worse for DAG2 − P10. Accel and DK-VIOL are exempt from this
     clause, because they were already worse and belong to DAG-OACC; they are reported.
6. **Beside (never decisive):**
   * DAG2 − DAG on every Read-A row;
   * the four families DAG2 − DAG2-0 at T0 and T1-kin (O-RECOVER's instruments);
   * DAG2 with HACC's rule re-fitted by `PREREG_V8F_DAG_HACC`'s recipe, read on Read A, if the fit inputs exist;
   * S-POS against CTRV.

## Outcomes, committed now

| outcome | condition | committed next |
|---|---|---|
| **FIXED** | R-CRAWL and R-MOVE at both seeds; guards held | adopt DAG2 as the base. Read A is re-run on DAG2 and on DAG2 with DAG-OACC's lever when it exists; a DOMINANT read there is the dominance claim |
| **PARTIAL** | R-MOVE at both seeds but R-CRAWL fails somewhere; or R-CRAWL and R-MOVE at one seed only; guards held | adopt DAG2 as the base if R-MOVE holds at both seeds; report. The residual crawl is then a launch-intent defect (O's v0 at tick 1, `R76`). Next: the launch-recovery target of `PREREG_V8F_HACC_MC` §7 on DAG2 states, under its own PREREG |
| **NO EFFECT** | R-MOVE fails at both seeds; guards held | the residual crawl is not an own-state-distribution defect. Next: as PARTIAL, on DAG states |
| **HARMS** | a guard fails at ≥ 1 seed | DAG2 is not adopted; DAG stays the base. Next: as PARTIAL, on DAG states |
| **BLOCKED** | COVERAGE < 0.5 | the rejoin-target form (DAGGER §7) |
| **REFUSED** | any control fails | fix and re-run |

The table is complete:
* REFUSED and BLOCKED are decided first;
* then the guards: if one fails, the outcome is HARMS;
* with the guards held, R-CRAWL and R-MOVE decide FIXED, PARTIAL or NO EFFECT.

## Queue

**Box 2**, first of three (then DAG-OACC, then FLAT-DAG):
1. bank s0 and bank s1, about 1.1–1.5 h per seed;
2. gate and train DAG2 s0, DAG2-0 s0, DAG2 s1 and DAG2-0 s1, each through its gate (ESTIMATED ~4–5 h);
3. the T0 evals.
**Dev-box CPU:** the 8 T1-kin rollouts and the reader.
⛔ No bar, row or routing changes after any DAGGER2 row is seen.
