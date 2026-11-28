# PREREG_V8F_ROLLHIST, Amendment 7 (DRAFT, for the MM to freeze; written before any RHp3 bank step, run or number)

Card `PREREG_V8F_ROLLHIST.md` (781eea87) and Amendments 1 to 6 all stand (Amendment 1 87928487, 2 15004dc9, 3 3c176fac, 4 192f1cff, 5 5ebf1f05; 6 as frozen). The commit of record stays **97fe0a56**, and no `stack/` file changes. This amendment adds ONE arm per seed, named in the card in advance, and the checker, chain and read files that run it. It moves no cell, bar, floor, estimator or tier.

## 1. What the card already says, and what was measured (MEASURED, ours)
* Card 3.4, row **ROLLHIST-PARTIAL**: "share p = 0.3 on the SAME bank (stage B only, K-B2 holds, about 0.27 pod-h per seed, the cheapest next arm; p = 0.3 is the DAgger-1 mapping not yet tried on this line)". The arm below is that row, not a post hoc choice.
* The two-seed read of record is **ROLLHIST-PARTIAL** (`read/out/read/RESULT_ROLLHIST.json`, sha256 5613818e5748b8ee38fa6c88165bf5699dd549f04128c4af8f068643e0d99c41):
  * P1: LATERAL, TACTICAL, STRATEGIC BETTER; LONGITUDINAL MIXED; none WORSE.
  * P2 fails; P3 fails (the ROLL name of RH is DEGRADES under both R0Hb20 and R0C10XHb20); P4 fails.
  * G-T0 FAILS at seed 0 and HOLDS at seed 1.
* The RH arms trained at `--dagger-p 0.1`. RULE ZERO: a PARTIAL is a waypoint; the next arm is the one the card named.

## 2. The arms (RHp3_s0, RHp3_s1)
| | RH_sk (the finished arm) | **RHp3_sk (new)** | B0_sk (the comparator, EXISTING) |
|---|---|---|---|
| argv | `/root/rollhist/runs/RH_sk/argv.json` | the RH_sk argv with **exactly two tokens changed: `--dagger-p 0.3` and `--out /root/rollhist/runs/RHp3_sk/train`** | `/root/rollhist/runs/B0_sk/argv.json` |
| bank | `/root/rollhist/bank/sk`, bank.json sha256 s0 e5e44c01...662cf, s1 bde6566e...279e6 | **the same bank, the same `--dagger-bank-sha256`** (no bank step) | none (`--dagger-p 0`) |
| init, steps, seed, dagger-seed, lr, batch, w-acc, readout token causal_g1x, kappa 0.2 | as the argv | **identical** | as the argv |
| code | 97fe0a56 | 97fe0a56 | 97fe0a56 |
* The dose is the literal 0.3 (the DAgger-1 mapping of the card). **No other dose is admitted**: the gate refuses 0.5, 0.25, 0.1, 1.0 and every other value for an RHp3 arm.
* **No new B0.** B0_s0 and B0_s1 are the comparators; their argv files and checkpoints are only read. B0 is the same model on the same windows, so RHp3 minus B0 uses the PARTIAL read's own B0 dumps.
* Stage A is untouched: each RHp3 arm initialises from its own `UE_sK.A` (K-B2).

## 3. The read (card section 3, on RHp3)
* **P1 (lever)**: RHp3 minus B0 under readout C\* (`R0C10XHb20`), all four families, the cells of 3.2 and the cell verdict rule of 3.2, F_c = max(abs(RHp3_s1 - RHp3_s0), abs(B0_s1 - B0_s0)) per cell under the same form (the card's floor with the treated arm's own seed spread). Estimator: paired episode-cluster bootstrap, B 2,000, seed 0 (`taniteval/ci.py` through the frozen readers). It answers "another draw of EPISODES"; the second seed answers "another TRAINING run"; inference is deterministic. Tier: T1-kin/hb0.2 (the executor of record, Amendment 2 item 1); G-T0 at T0.
* **P2, P3, P4, G-T0**: as 3.3 and 3.2, with RHp3 in RH's place and B0 as before (the P2 attribution control reads B0's ACC pair, which is the finished B0 rollouts).
* **Beside, no bar, decides no name**: RHp3 minus RH under `R0Hb20` and under `R0C10XHb20`, per family (the dose p = 0.3 against p = 0.1 on the same bank). One-seed floors and the UE.A references are the PARTIAL read's (`out/read/refs_*`, `STRAT_FLOOR_UEA.json`), reused read only.
* **Outcome names**: the card's table 3.4 with its precedence (BLOCKED, then HARMS, then SEED-DISAGREE, then the rest in table order) and Amendment 2's gaps G1 and G2 and its straddle rule, reading RHp3 where the card reads RH. A one-seed read prints the same name prefixed PROVISIONAL. Names that mention RH (REPAIRS-K-UNNEEDED, REPAIRS) are read as "RHp3 does".
* **The one-line answer to "did p = 0.3 repair P2 / P3?"** (printed by the reader as `P03:`): `p = 0.3 REPAIRS both P2 and P3` | `repairs P2 but NOT P3` | `repairs P3 but NOT P2` | `repairs NEITHER P2 nor P3`, with RHp3's P2 and P3 against the PARTIAL read's RH P2 and P3 (both False) beside. P2 and P3 are the card's, read pooled over the two seeds as the reader of record reads them.
* **What is committed in advance about G-T0**: the PARTIAL read already failed G-T0 at seed 0. A larger dose moves the policy further from the logged history, so G-T0 failing at BOTH seeds is a real possibility. By the card's own precedence that reads **ROLLHIST-HARMS** whatever P1 shows. It is reported as written and is not softened.

## 4. Cost (ESTIMATED from measured arm times, `rh_chain.log`)
* Measured chain wall per arm on the LoRA pod (selfcheck, gate mint, 3,000 steps, post gates): RH_s0 15.9 min, B0_s0 15.1 min, RH_s1 15.9 min, B0_s1 15.2 min (the trainer's own 3,000 steps: 569 s for RH_s0). RH costs about 0.8 min over B0 for the swap at p = 0.1.
* RHp3: **about 0.27 pod-h per seed (ESTIMATED; 0.26 to 0.30 allowing the swap overhead to triple at p = 0.3), about 0.54 pod-h for both, no bank step.** This is the card's "about 0.27 pod-h per seed".
* Dev-box read (CPU, free): 12 RHp3 rollouts at about 6.4 min each (c-EXEC's 6.2 to 6.6 min per rollout, ESTIMATED) = about 77 min, plus the T0 evals and the readers. B0 and RH dumps are reused: nothing re-rolled.
* The marker protocol and the GPU queue lock are the RH chain's (the chain holds `/root/gpu_queue.lock` to its end).

## 5. Next lever per outcome (committed in advance; the first two are the card's)
| name (RHp3) | next |
|---|---|
| ROLLHIST-REPAIRS-K-UNNEEDED | as the card: the integrated candidate with the causal handover and without readout K; register the base as `UE.A + ROLLHIST` at p = 0.3 |
| ROLLHIST-REPAIRS | as the card: the candidate keeps K beside; DAgger iteration 2 only if P4's gap is above floor on two families |
| ROLLHIST-PARTIAL (at p = 0.3) | p = 0.3 is spent and the card names no lever after it. **The MM pre-registers one before any run.** Candidates, labelled post hoc until frozen: a 9,000-step stage B at p = 0.3 (about 0.8 pod-h, ESTIMATED from 0.282 s/step); DAgger iteration 2 from RHp3's own states (about 2.3 pod-h); the recovery label (`H-ST-RECOVERYLABEL-1`) |
| ROLLHIST-NO-EFFECT | the card's row, p = 0.3 now done: the 9,000-step stage B at p = 0.3 |
| ROLLHIST-HARMS | the card's row: localise with the beside UE.B if it exists, else a bank under readout K or the recovery label |
| ROLLHIST-SEED-DISAGREE | the card's row: a third seed pair, back to the PI |
| ROLLHIST-BLOCKED | no number; fix and re-gate |
RULE ZERO: whichever row fires, the turn ends with the next arm's argv ready, not with the verdict.

## 6. Launch gate and reachability (the class of Amendments 4 to 6)
* **The checker is a new file, `dscale10k_check_rhp3.py`** (the Amendment-5 checker 69ff1a0f plus the kind `RHp3`; patch tool `patch_rhp3.py`, every edit anchored and counted). `dscale10k_check_rh.py` is untouched, so RH and B0's tokens stay valid. What it pins, and why each needed changing for p = 0.3:
  1. `rh_argv_problems` (C-ARGV) compared an RH argv to the template (`--dagger-p 0.1`) and refused any difference in `--dagger-p`. For RHp3 `--dagger-p` is an allowed difference **and must read exactly `['0.3']`**; every other flag is still pinned to the template (the "fifth changed flag" rule is unchanged: RHp3 differs from the template only in the arm's own out/init/bank/sha/declarations and the dose).
  2. `RH_ARMS`, `required_checks`, the pair, bank-generator and G-DAGGER branches keyed on kind `"RH"` now take kind in `("RH", "RHp3")`; the run-time token check likewise.
  3. NOT changed: `rollhist_pair_check.py` (stock; it requires `--dagger-p > 0` for RH and `0` for B0, no dose), `rh_gate_record.py` (keyed on the tag prefix `RH`, seed from `_sK`), `rh_evidence.py`, `rh_selfcheck.py`. The checks that pin `RH_s0` or `0.1` and were found by reading the source: C-ARGV (changed), the arm-name tables (changed), nothing else.
* **Gate tests** (`test_dscale10k_check_rhp3.py`, the 40 existing tests unchanged plus the RHp3 tests): see `pytest_log_p3.txt`. Mutation arms that must be refused: `--dagger-p` 0.5, 0.25, 0.1, 1.0, 0.30, 0, abc; p = 0.3 plus any second changed flag (lr, steps, an added flag, a removed flag, batch, a data path, kin mode, seed, w-acc, dagger-seed, p-gt, ego-perturb); an RH argv at 0.3 under the RH name; RH's argv under the RHp3 name; a moved B0; a wrong token; a missing token; an edited argv at run time. A checker with the dose pin removed must fail at least one of them (`pytest_log_p3_mutant.txt`).
* **Reachability** (each hop named): the flag `--dagger-p 0.3` is in the argv written by `rhp3_argv.py` (the trace: `rhp3_argv.py` -> `runs/RHp3_sk/argv.json` -> `chk_arm_rhp3.sh mint` -> `dscale10k_check_rhp3.py` C-ARGV -> the token binds the argv sha256 -> `chk_arm_rhp3.sh run` refuses without it -> the trainer's `--dagger-p` -> `make_dagger` -> `DaggerSwap(p=0.3)` -> `bad_dagger` reads it back from the BUILT swap). The trainer's own `--selfcheck-only` (`dagger_probe`: swap reaches the loss, every p = 0 control exactly 0.0) runs on THIS argv at the arm.
* **Before the launch** (Amendment 6's lesson, applied): the MM runs `DRY_MINT_PLAN.md` on the pod, the arm's own mint by hand on the real bank, and reads `gate/checks.json` in full.
* Class of the three pre-train refusals of Amendments 4 to 6: environment paths that only a pod mint exercises. This arm adds none: it uses the same `cd $DS`, the same bank, the same tree.

## 7. Files this amendment adds (sha256 in `MANIFEST_SHA256.txt`)
Pod side (copied by the MM to `/root/rollhist/`): `rh_p03_chain.sh`, `rhp3_arm.sh`, `chk_arm_rhp3.sh`, `rhp3_argv.py`, `rh_p03_marker_watch.sh`, `launch_p03.sh`, and `study/dscale10k_check_rhp3.py`. Dev box: `read/rh_read_p03_chain.sh` and `.cmd`; edits (each with a `.bp03` copy) to `rh_common.py`, `rh_roll.py`, `rh_t0.py`, `rh_strat.py`, `rh_read.py`, `dry_check.py`, all inert unless `RH_AKIND=RHp3` is set (proved by `dry_check.py`: the default read's pairs are equal to the pre-edit copy).
