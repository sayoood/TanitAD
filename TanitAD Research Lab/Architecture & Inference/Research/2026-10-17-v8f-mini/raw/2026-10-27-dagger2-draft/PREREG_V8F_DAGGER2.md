# PREREG_V8F_DAGGER2 (DRAFT): DAgger iteration 2, on DAG's own states

Drafted 2026-10-27 by the Master Mind. It was written **before** any row of `PREREG_V8F_DAG_REREAD.md` (`bdf8d561`) was seen, and is banked
in `raw/2026-10-27-dagger2-draft/`. Claim row `H-V8F-DAGGER2-1` (created when the draft is promoted).

**When it runs.** It runs if and only if `PREREG_V8F_DAG_REREAD` Read A reads one of:
* **DOMINANT, CRAWL OPEN**;
* **MARGIN, NOT DOMINANT** with only crawl rows failing (L2, curvature, yaw-rate, M1).

Under DOMINANT, or MARGIN with a non-crawl row failing, it is parked. Box 2 needs the PI's go.

**What it is.** Everything is `PREREG_V8F_DAGGER.md` with Amendments 1 and 2, verbatim (bank, matched route point, sample construction,
controls K-B0…K-B5, reader controls, estimator, tiers), **except the six lines below**. The difference is the generator and the init: the bank
is rolled out by DAG, the policy whose remaining failure Read A measured, and the fine-tune starts from DAG.

## The six changes

1. **Generator (§1):** `M2b_s{seed}_dag/ckpt_last.pt` (sha256 from `v8f-dagger/M2b_s{seed}_dag/SHA256SUMS.txt`), PLAIN, with the same
   rollout configuration: m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, executor `plan`. The same 4,512 TRAIN clips and
   306,566 windows; ticks 1–2.
2. **Arms (§4):**
   * **DAG2** = DAG_s{seed} fine-tuned 3,000 steps with `--dagger-p 0.1 --dagger-bank <DAG2 bank> --dagger-seed <seed> --ego-perturb-p 0`,
     with the same `--lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2` and the init `--init-ckpt` = DAG_s{seed}.
   * **DAG2-0** = the same argv with `--dagger-p 0`.
   * Both seeds, through the trainer's launch gate.
3. **The replicate floor (§5):** no same-argv replicate of DAG2-0 exists or is trained. RF is fixed now at DAGGER's measured values,
   per seed: lat 0.0067 / 0.0043, lon 0.0251 / 0.0139 (`DAG_READ.json`). Bars are max(2F, 2·RF), inclusive: lat 0.0248 / 0.0248, lon 0.0502
   / 0.0278. ⚠ RF is an estimate taken from another init with the same recipe; it is disclosed as such.
4. **The decisive rows (§5).** These replace R-TLAT / R-TLON as the target, because the heads already cleared:
   * **R-CRAWL:** every Read-A row that was separated-worse for DAG − P10 at a seed must read **not separated-worse** for DAG2 − P10 at that
     seed (same reader, same row, same mask). M1, if it failed, is in this set.
   * **R-MOVE:** on each such row, DAG2 − DAG2-0 is separated-better at that seed.
5. **Guards (§5):** each must hold at both seeds:
   * R-TLAT and R-TLON (DAG2 − DAG2-0) not separated-worse;
   * R-ADE (T1-kin ADE6) not separated-worse;
   * R-T0 point ≤ +F/2 = 0.0604 m;
   * no Read-A row that passed for DAG becomes separated-worse for DAG2 − P10.
6. **Outcomes (§7):**

| outcome | condition | committed next |
|---|---|---|
| **FIXED** | R-CRAWL and R-MOVE at both seeds; guards held | adopt DAG2 as the base; re-run Read A on DAG2 (a DOMINANT read there is the dominance claim) |
| **PARTIAL** | R-CRAWL at one seed, or R-MOVE without R-CRAWL; guards held | report; iteration 3 is NOT run automatically. The crawl is then a launch-intent defect (O's v0 at tick 1, `R76`), and the next lever is the HACC-MC §7 launch-recovery target on DAG states, under its own PREREG |
| **NO EFFECT** | R-MOVE fails at both seeds; guards held | the residual crawl is not an own-state-distribution defect. Next: as PARTIAL |
| **HARMS** | a guard fails at ≥ 1 seed | DAG2 is not adopted; DAG stays the base. Next: as PARTIAL |
| **BLOCKED** | COVERAGE < 0.5 | the rejoin-target form (DAGGER §7) |
| **REFUSED** | any control fails | fix and re-run |

## Queue and cost (ESTIMATED)

* **Box 2:** bank s0, bank s1 (1.1–1.5 h per seed, DAGGER §8), then 4 runs of 3,000 steps each. That is ~6–7 h in all, by analogy with
  DAGGER's chain (01:14 → 05:55 for the four runs).
* **Dev-box CPU:** 8 T1-kin rollouts (~5 min each), the fresh-age T0 evals, and the reader.
* **Code:**
  * the generator already takes `--ckpt`, so the only new code is the chain's re-parametrisation and a reader whose R-CRAWL / R-MOVE rows
    read Read A's failing set from `READ_DAG_REREAD`'s JSON;
  * the reader is proved by mutation (a mutant that drops a failing row from the set must go RED).
