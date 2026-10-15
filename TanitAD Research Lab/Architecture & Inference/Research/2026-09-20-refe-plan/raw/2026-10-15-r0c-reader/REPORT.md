# R0c mechanical reader: maps an R0c arm directory to ONE outcome of PREREG_R0C §4, or to AMBIGUOUS

**Master Mind, 2026-10-15 ~23:35 Berlin.** Built by a Sonnet agent and verified by the MM. It lands **before R0c's first step**: R0c has been queued since 20:57:25Z and has no number yet.

## What it is
| file | role |
|---|---|
| `r0c_read.py` | Reads the arm's monitor, kill and trainer records. It evaluates PREREG_R0C §4 under every reading that has no committed answer, and names an outcome only when all readings agree. It never picks. It parses the five §4 rows from the prereg at run time and pins its sha256 (`cab2bb1c…`). It prints `ZZR0C_OUTCOME <O1..O5\|AMBIGUOUS>`. stdlib only. |
| `r0c_ba_read.py` | **GAP 0.** The chain writes no B − A interval: its u1000 read has B − A0 and A − A0, but B − A as a point only. This writes `rlval_BvsA_update<N>.json` using the chain's own `rl_val_eval` functions and the same estimator. |
| `test_r0c_read.py` | **197 = 179 checks + 18 RED mutations.** Re-run by the MM on this tree: `ZZTEST_PASS 197`. |
| `build_realdata.py`, `realdata/` | R0's and R0b's banked outputs, rebuilt as arm directories, two ways: `*_banked` (against C0, as recorded) and `*_ownA` (re-judged against the arm's own A, the R0c control rule). |

## Validation on real data (MEASURED, RL-val selection tier, one training seed per arm)
- **All four directories reproduce the recorded kill:** K3 at checks (250, 500).
- **All four read AMBIGUOUS (MISSING), which is correct.** R0 and R0b anchor EPO to the BASE, so their `epo_kl` is base→B and cannot decide O3 vs O4. The reader refuses to label them, rather than misreading base→B as A→B. This is the negative control for the `data.epo.ref == "armA"` requirement.
- **Plumbing check (MM; not a result):** on a copy of `R0_ownA` with only `data.epo.ref` relabelled to `armA`, the reader under the A1 policy resolves to a **single** outcome (O3, at mean `epo_kl` 0.1185 ≤ 0.2). This shows the policy path decides end to end. R0's KL is base→B, so this number means nothing about R0c.
- **R0's B − A against its own A**, paired log-cluster bootstrap, 10,000 resamples, seed 20260927, ×100 (`realdata/R0_ownA/rlval_BvsA_update*.json`):

  | check | PDMS | TTC | EP |
  |---|---|---|---|
  | u250 | +0.61 [+0.12, +1.12] | −1.23 [−1.83, −0.62] | +3.06 |
  | u500 | +0.95 [+0.43, +1.48] | −0.63 [−1.35, +0.06] | +3.10 |
  | u1000 | +1.15 [+0.54, +1.75] | −1.00 [−1.73, −0.30] | +3.44 |

  These are the base-anchored numbers that R0c's M1 is predicted to move: B − A TTC rising toward −0.17.

## Its two integration requests, both answered in the same commit
1. **Q1–Q8 open, plus the silent states.** They are answered **before any R0c number** in `eval/PREREG_R0C_A1.md`. Machine form: `eval/r0c_policy_A1.json`, sha256 `34a8e169…`.
2. **GAP 0.** Fixed **at read time** without touching the running chain: `PREREG_R0C_A1.md` §1 runs `r0c_ba_read.py` at u250/500/750/1000 on R0c's dumps, ~9 s each, deterministic.

## Families
- **LONGITUDINAL** (EP, TTC, C) and **LATERAL** (DAC, DDC): B − A paired CIs at every check.
- **TACTICAL:** N/A for this scorer-only arm.
- **STRATEGIC:** PAUSED (PI 2026-09-19).
