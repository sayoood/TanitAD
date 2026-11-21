# PREREG_R0C — Amendment A1: the eight open readings of section 4, the read procedure, and the silent states

**Status. IN FORCE 2026-10-15 ~23:30 Berlin (21:30Z), written by the Master Mind BEFORE R0c's first step.**
- R0c has been queued since 20:57:25Z (`ZZRLS1_QUEUED`) behind the P2 seed replicate. It holds no GPU and has produced no number.
- This file **amends** `eval/PREREG_R0C.md` (sha256 `cab2bb1c…`), which stays byte-identical. The mechanical reader pins that sha256.
- **Machine form:** `eval/r0c_policy_A1.json`. Its sha256 is printed by `r0c_read.py` at the read and recorded in the result.
- **Nothing here changes:**
  - a bar, an estimator or a kill rule (K0–K6, K-MON);
  - the argv (`eval/rls1c_arms.json`), the code, or the queue.

It resolves readings, fixes the read procedure, and names the states the table is silent on.

## 0. Why

The mechanical reader `raw/2026-10-15-r0c-reader/r0c_read.py` (197 checks: 179 checks plus 18 RED mutations) evaluates an arm under **every** reading of section 4 that has no committed answer. It names an outcome only when all readings agree. On R0c it found:
- **eight open clauses (Q1–Q8);**
- **GAP 0:** the chain never writes a B − A interval. Its u1000 read gives B − A0 and A − A0 with intervals, but B − A only as a point. **O1 and O2 cannot be read from the chain's own outputs.**

Without this amendment most reachable states read AMBIGUOUS.

## 1. Read procedure (closes GAP 0, without touching the running chain)

After the arm ends (finished or killed), on the dev box from R0c's HF egress, or on the pod only after the chain has exited (never beside training):

1. For each check N ∈ {250, 500, 750, 1000} whose dumps exist:
   ```
   python raw/2026-10-15-r0c-reader/r0c_ba_read.py --a <arm>/scores/scores_update<N>_A.npz --b <arm>/scores/scores_update<N>_B.npz --pdm <pdm_val> --terms <terms_val> --out <arm>/rlval_BvsA_update<N>.json
   ```
   - It uses the same estimator as every RL-val read: paired log-cluster bootstrap, 10,000 resamples, 95 %, seed 20260927.
   - It is deterministic and takes ~9 s per check on CPU.
2. Then run `python raw/2026-10-15-r0c-reader/r0c_read.py <arm> --policy eval/r0c_policy_A1.json --out <arm>/R0C_READ.json`.
3. **The reader's outcome IS the outcome.** Before the R0C_READ.json is quoted, the policy sha256 it prints must equal the one this commit lands.

## 2. Answers (committed before any R0c number)

| Q | the open clause | answer | why |
|---|---|---|---|
| **Q1** | g4 on RL-val (`rl_val_eval` lists g4, the longitudinal-bias rows, as `not_computed`) | **vacuous** at RL-val | "Blocking" would make O1 unreachable by construction, and a registered outcome cannot intend that. g4 is **carried, not waived**: it is computed and binding at O1's own next lever (the full-navtest endpoint after the replicate). |
| **Q2** | which `epo_kl` is "the mean at the breach" | **second_breach_window**: mean of the logged `train.epo_kl` rows in (first breached check, second breached check] | These are the 250 updates over which the breach persisted and K3 fired. All five readings are still **reported** in R0C_READ.json as a sensitivity row; only this one decides O3 vs O4. |
| **Q3** | an NC fall inside a g5 breach | **g5** (read as O3/O4, not O5) | `PREREG_RL_S1.md` section 5 (K0–K6) defines **no separate NC/DAC bound** (re-read at `159c862`). g5 is "TTC and NC may not fall by more than 0.5 vs the arm's control". O5's "NC / DAC bound" clause has no referent, so it cannot fire. DAC is reported, never gating, at RL-val. |
| **Q4** | which check O2's "lower bound ≤ 0" refers to | **either_le0**: LB ≤ 0 at u500 OR u1000 | O1 needs LB > 0 at **both**. O2 is its complement among g5-holding arms, which removes the gap where LB > 0 at one check only. |
| **Q5** | does O2 ("safe") need g3 | **g3_also**: g3 holds at every check | "Safe" means every guard held. A flat arm with a g3 breach is not "safe but flat" (state S2 below). |
| **Q6** | the +0.25 floor of `PREREG_RL_S1` section 4 (1) on RL-val | **not_applicable** at RL-val | That floor is defined on FULL navtest over two R − C pairs. At the RL-val tier, O1 needs LB > 0 at u500 and u1000. The +0.25 bar is applied **unchanged** at O1's full-navtest endpoint. |
| **Q7** | scope of "no longitudinal / lateral component separates adversely" | **all** six sub-scores (NC, DAC, EP, TTC, C, DDC) | Any sub-score whose B − A interval lies entirely below 0 at u1000 blocks O1. This is the strictest scope: an adoption candidate may not trade a sub-score. |
| **Q8** | a K-MON stop (the monitor died) | **not_O5** ⇒ **INVALID** | K-MON is the chain's fail-closed stop on a dead monitor, i.e. an **instrument failure, not a training kill**. The arm has no outcome. It is re-run after the monitor fault is fixed. |

## 3. The states the table is silent on (the reader returns NONE / OVERLAP / MISSING), named now

| state | what the reader shows | read as | next lever |
|---|---|---|---|
| **S1** | no kill; g5 breached at one or more checks, never two consecutive | **NOT ADOPTABLE, NOT SAFE-FLAT** | the replicate arm (same flags, other seed) decides whether the breach is the rig's run-to-run floor (`H-ESTIM-SEED-1`). Every breached check is reported. |
| **S2** | no kill; g5 holds; g3 breached at isolated checks; LB not > 0 at both checks | **NOT SAFE-FLAT** | as S1. O2's `--epo-beta 0.03` pair is **not** taken. |
| **S3** | g3–g5 hold (or g3 isolated); LB > 0 at u500 and u1000; a side clause fails (an isolated g3 breach, or an adverse sub-score under Q7) | **O1-BLOCKED** | not a candidate. Escalate to the PI with the blocking clause named; the replicate is NOT launched automatically. |
| **S4** | K3 fired on a g3-only or mixed g3/g5 pair (g5 not breached at both checks of the pair) | **O5 (other kill)** | K3 is a section 5 kill. The g3 telemetry (pick concentration, `rls1_checks.py` `pick_ok`/`share_ok`) is reported. Next lever: per section 5. |
| **S5** | OVERLAP: two kill kinds on record that no timestamp orders | **O5** | a training kill is never overridden by a guard reading. Both are reported. |
| **S6** | MISSING: a needed input is absent (e.g. no `rlval_BvsA_update<N>.json`; `data.epo.ref` ≠ `armA`) | **INVALID read, not an outcome** | re-produce the input (section 1) and re-read. If `data.epo.ref` ≠ `armA`, the arm did not test the R0c hypothesis at all, which is a launch-gate defect to escalate. |

## 4. Reported in every outcome (never gating)

- **The second control:** B vs C0's arm A (`rlval_vs_C0.json`). The reader proves that the outcome is identical whether it is present or absent.
- **LONGITUDINAL:** EP, TTC, C. **LATERAL:** DAC, DDC. Each is a B − A paired CI at every check.
- **TACTICAL:** N/A on this scorer-only arm (a frozen fan with no decision output; decision metrics are full-navtest R2 endpoints).
- **STRATEGIC:** **PAUSED** (PI 2026-09-19), never a pass.
- **Tier:** the RL-val **selection tier**: the true NAVSIM v1.1 PDMS of the scorer's pick on 3,000 sets, open-loop table lookup. It is **not** T1 and **not** an endpoint.
- **Seeds:** one training seed per arm, so a separated interval is **necessary, not sufficient** (`H-ESTIM-SEED-1`).
