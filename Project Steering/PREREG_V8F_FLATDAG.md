# PREREG_V8F_FLATDAG: does the hierarchy beat flat once BOTH get own-state supervision?

Promoted 2026-10-27 by the Master Mind from the draft banked at `5eaa0232` (`raw/2026-10-27-flatdag-draft/PREREG_V8F_FLATDAG.md`), which was
written **before** any row of `PREREG_V8F_DAG_REREAD` was seen. This promotion comes **before any FLAT-DAG bank, run or row**. Claim row
`H-V8F-FLATDAG-1`.

**Why it runs.** The draft's condition: it runs if and only if `PREREG_V8F_DAG_REREAD` Read B-2 reads ESTABLISHED.
* B-2 read **ESTABLISHED under Amendment 1** of that PREREG: DAG − FLAT at T1-kin −0.596 [−0.738, −0.457] / −0.351 [−0.470, −0.229] m
  (`RESULT_V8F_DAG_REREAD.md`).
* ⚠ That amendment is post-data and control-only (C5's cut is shown by the checkpoint config and FLAT's exact-zero oracle identity). It is
  disclosed here because this arm's run condition rests on it.
* DAG-0 − FLAT is not separated at either seed, so B-2's margin came with DAgger. This arm asks whether it needs the hierarchy.

**PI go, 2026-10-27, verbatim:** *"yes queue it"*, in answer to *"Queue DAgger iteration 2, then DAG-OACC, then FLAT-DAG on box 2, each
through its launch gate?"*

**The question.** B-2 compares DAG, which had a DAgger fine-tune, with FLAT, which had none. This arm gives FLAT the same treatment: its own
bank and the same recipe. The DAG − FLAT-DAG contrast then varies only the hierarchy's interface.

## 1. The arms

* **FLAT-DAG** = `M2b_s{seed}_flat` (sha256 `8496934a…` / `6fdfc9b0…`, `v8f-mini-flat/M2b_s{seed}_flat.pt`) fine-tuned 3,000 steps with DAG's
  recipe:
  * `--init-ckpt` FLAT_s{seed}, `--dagger-p 0.1 --dagger-bank <FLAT bank> --dagger-bank-sha256 <its sha> --dagger-seed <seed>
    --ego-perturb-p 0 --lr 1e-4 --warmup 100 --p-gt-start 0.2 --p-gt-end 0.2`;
  * every other flag is FLAT's argv, and FLAT's `--o-interface cut` is kept.
* **FLAT-DAG-0** = the same argv with `--dagger-p 0`.
* **The FLAT bank** = `PREREG_V8F_DAGGER` §1–2 with Amendments 1 and 2, verbatim, with FLAT_s{seed} as the generator: PLAIN, the same rollout
  configuration (m = 10, causal bucket, dead-reckoned nav, logged scene, periodic, executor `plan`), the same 4,512 TRAIN clips and ticks
  1–2, and the same controls K-B0…K-B5 and COVERAGE ≥ 0.5. Each seed's bank comes from that seed's FLAT.
* ⚠ **Build check, a launch blocker (CLAUDE.md 2026-09-26).** The trainer's `--o-interface cut` and `--dagger-*` must compose. In the same
  gate run, G-DVB has to show both the cut (zero gradient on `e_code`; O invariant to S and T) and the swap reached (share 0.1 ± 0.02).
  Until that is shown, nothing is trained, and the outcome is BLOCKED with the failing check named.

## 2. Rows

All at T1-kin with DAG's decisive settings. Paired episode-cluster bootstrap, B = 2,000, per seed. F = 0.1207 m. Plain arms.

| row | definition | bar |
|---|---|---|
| **PRIMARY H-T1** | ADE6, DAG − FLAT-DAG | ≤ −2F = −0.2415 m, separated, at both seeds |
| **families** | Read A's rows of `PREREG_V8F_DAG_REREAD` §2, DAG − FLAT-DAG | none separated-worse, for the claim to read "beats" |
| H-T1-HEADS | R-TLAT / R-TLON, DAG − FLAT-DAG (T's heads exist in both; FLAT's O does not read them) | beside |
| D-FLAT | ADE6, FLAT-DAG − FLAT-DAG-0: does DAgger help flat too? | beside, with DAGGER's R-ADE / R-T0 guards reported |
| S-POS | against CTRV, both arms | beside |

## 3. Controls (each must pass, or the read is REFUSED)

* **C-F1:** each bank passes K-B0…K-B5, with its header naming that seed's FLAT checkpoint (sha256).
* **C-F2, the pair rule:** FLAT-DAG and FLAT-DAG-0 differ only in `--dagger-p`, the bank flags and the output paths.
* **C-F3, the cut survives the fine-tune:** FLAT-DAG's checkpoint config reads `o_interface: cut`, and its oracle − own identity reads
  0 [0, 0] on the eval windows, as FLAT's did.
* **C-F4:** Read A's C1–C4, re-run on the new arms.

## 4. Outcomes, committed now

Applied in this order: REFUSED → BLOCKED → FLAT BETTER → HIERARCHY BEATS FLAT → NOT ESTABLISHED.

| outcome | condition | committed next |
|---|---|---|
| **REFUSED** | any control fails | fix and re-run |
| **BLOCKED** | the build check fails, or COVERAGE < 0.5 at ≥ 1 seed | the failing check is named; for COVERAGE, the rejoin-target form (DAGGER §7) |
| **FLAT BETTER** | ADE6 DAG − FLAT-DAG separated-worse at ≥ 1 seed | report; the hierarchy costs at T1-kin; the PI decides the architecture |
| **HIERARCHY BEATS FLAT** | PRIMARY H-T1 at both seeds, and no family row separated-worse | the first hierarchy-over-flat claim of v8F, at T1-kin: register it and signal the PI |
| **NOT ESTABLISHED** | otherwise | B-2's margin was DAgger's, not the hierarchy's. The interface stays the defect; next is MIXO's committed T-side arm, under its own PREREG |

## 5. Queue and reporting

* **Box 2**, third of three (after DAG-OACC): bank s0 and s1, then the four gates, then the four runs and their T0 evals (ESTIMATED ~6–7 h).
* **Dev-box CPU:** the 4 new T1-kin rollouts (DAG's are banked) and the reader.
* Per seed: delta [lo, hi], separated, arm means and n, with the tier stamp. STRATEGIC is reported; it is not paused for v8F.
* Package `raw/2026-10-27-flatdag/`. Result `RESULT_V8F_FLATDAG.md`. The register row is updated in the same landing.
* ⛔ No bar, row or routing changes after any FLAT-DAG row is seen.
