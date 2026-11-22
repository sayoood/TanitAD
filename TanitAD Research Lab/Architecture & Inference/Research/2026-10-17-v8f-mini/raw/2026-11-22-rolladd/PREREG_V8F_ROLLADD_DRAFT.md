# PREREG_V8F_ROLLADD (DRAFT, MM 2026-11-22): own rolled states ADDED to the batch, the logged samples kept whole

## FREEZE RECORD
* DRAFT. It is frozen after the zero-training build (FACTS, gold test, reachability, mutants, the add census) and before any RA training step, pod census or number. No RA number exists.
* Code: a new commit on `agent/v8f-rollhist-cdb-20261120` on top of cf27dbb7 (`--dagger-mode {swap,add}`, default `swap` = cf27dbb7 bit for bit, proven by a gold test). RH, RL and B0 stay at their own commits.
* Running the arms needs a PI go with the cost (section 7).

## 0. Why (the measured case, quoted)
* ROLLHIST line, tier T1-kin/hb0.2 with G-T0 at T0, paired episode-cluster bootstrap B 2,000 seed 0. Four arms train stage B on their own rolled history, and every one costs the T0 forecast on logged history. RH and RL fail G-T0 at s0 only; RHp3 and RHK fail at both seeds. Meanwhile every one buys own-action gains at T1. RL - B0 under C\*: STRATEGIC -1.80 / -3.50 / -5.29 m at the 30 / 50 / 80 m marks, yaw rate -5.31 deg/s, ade_tac -0.368 m (`2026-11-22-reclab/read/MM_VERDICT_RECLAB.md`).
* RECLAB read MECH-NO: the speed-consistent label does not lower the T0 cost against RH (RL - RH at T0 ADE6: s0 -0.0024 [-0.0273, +0.0247], s1 -0.0156 [-0.0376, +0.0058]). The RECLAB card commits the next lever for MECH-NO: **keep the logged samples whole and ADD the own states**, which separates interference from replacement.
* MEASURED from source at cf27dbb7: `DaggerSwap.apply` (`v8f_dagger.py` :733) selects rows `(u[0] < p) & eligible` and `swap_rows` (:570) overwrites those rows' ego tensors in place. So each selected logged window loses its logged sample for that step. In the RL train logs, 15,411 of 192,000 rows (8.0 %) were swapped at s0 and 15,380 at s1 (`pod_rl/RL_s*/train_log.jsonl`).

## 1. Hypothesis (HYPOTHESIS, committed before data)
**H-V8F-ROLLADD-1 (replacement).** The T0 cost of own rolled history comes from replacement. The swap removes the logged sample of every selected window, which is exactly the input-target pair T0 scores. If the same own states are appended and the logged samples stay, the logged mapping is trained as in B0 and the own-state training is kept. So at RH's bank and dose, RA - RH at T0 ADE6 < 0, G-T0 holds, and RH's T1 gains stay.

**The named alternative (interference).** The own states teach a mapping that pulls the outputs on logged history, through the shared trunk, whatever else is in the batch. Then RA - RH at T0 is about 0 (MECH2-NO) and G-T0 fails as before.

What `add` changes against RH, and nothing else:
* The selection is the same: the same generator, the same `u`, and so the same own records.
* The N logged rows that the swap would overwrite stay in the batch, and the N swapped rows are appended after them.
* Per step, the logged share is B / (B + N), about 0.93 at the RL rates, against (B - N) / B, about 0.92, under the swap. The own share is about 0.07 against 0.08. So the mixture is nearly unchanged; what differs is that the replaced logged samples are kept.
* The arm sees B + N rows per step instead of B, about 8 % more samples per step.

## 2. Arms
| arm | what | argv against RH_sK | init |
|---|---|---|---|
| **RA_s0** (the lever) | RH's stage B with the own states appended | exactly `--dagger-mode add` added, `--out` changed; the RH bank `/root/rollhist/bank/s0` (sha256 e5e44c01...) | UE_s0.A |
| **RA_s1** (replicate) | as above at seed 1 | the same; `/root/rollhist/bank/s1` | UE_s1.A |
| comparators, no new cost | RH_s0/s1, B0_s0/s1 (banked runs and dumps) | | |

* The label is the human one (RH's), so RA - RH has one lever.

## 3. Reads and bars (T1-kin/hb0.2; G-T0 and MECH2 at T0; ROLLHIST card section 3 binds unless changed here)
* Executor, windows, estimator, cells, the cell and family verdict rules and F: as in the ROLLHIST card's sections 3 and 3.2 and Amendment 8. Estimator: paired episode-cluster bootstrap B 2,000. It answers "another draw of EPISODES", and the two seeds answer "another TRAINING run".
* **P1 (lever, decides):** RA - B0 under readout C\* (`R0C10XHb20`), all four families, STRATEGIC in full.
* **G-T0 (guard, decides):** RA - B0 at T0 ADE6. The upper interval bound must be at most +0.0705 m, per seed.
* **MECH2 (mechanism, decides the next lever only):** RA - RH at T0 ADE6, paired, per seed. MECH2-YES needs the interval below 0 at both seeds.
* **ATTR (beside, no bar):** RA - RH under `R0C10XHb20`, four families; also P2, P3 and P4 as the ROLLHIST card defines them, on RA.
* **Lever-acts check (known values):** in the train log, the appended row count per log interval equals the selected count, and is greater than 0 for RA. Under the default `swap` it is exactly 0, and the batch keeps B rows. In the selfcheck probe, the first B rows of the add batch equal the logged batch bit for bit, and the appended rows equal the swap's rows bit for bit.

## 4. Outcomes, committed in advance, mutually exclusive, two seeds (a one-seed read prints the same name prefixed PROVISIONAL)
| name | condition | reading | next lever |
|---|---|---|---|
| **ROLLADD-REPAIRS** | G-T0 holds at both seeds; in P1, at least one family BETTER at both seeds and none WORSE at either | the swap's replacement was the T0 cost; own rolled history is admissible in add mode | the dose with the larger T1 gains: p 0.3 in add mode (stage B only, same bank) |
| **ROLLADD-HARMS** | G-T0 fails at both seeds, or a family WORSE at both | own rolled history still costs when nothing is replaced | MECH2-YES (replacement is part of the cost): add mode at p 0.05. MECH2-NO (interference): the kinematic label, R2 of the RECLAB card: the yaw-rate and acceleration targets of the swapped sample blend from the own history into the human's, under the speed label's feasibility envelope |
| **ROLLADD-PARTIAL** | G-T0 holds at one seed only, or holds at both while P1 shows no family BETTER at both seeds | part of the cost is replacement | MECH2-YES: add mode at p 0.05, to find the dose that holds G-T0 with the gains. Otherwise as HARMS with MECH2-NO |
| **ROLLADD-SEED-DISAGREE** | a family BETTER at one seed and WORSE at the other, or the decisive cells straddle the floor | not decidable on two seeds | a third seed pair, which needs a UE_s2.A stage A: back to the PI |
| **ROLLADD-BLOCKED** | a gate check fails, or the lever-acts check fails | the arm does not launch, or its number is not read | fix and re-gate |

Precedence: BLOCKED, then HARMS, then SEED-DISAGREE, then the rest in table order.

## 5. Launch gate and reachability
* G-DVB: a `dagger_mode` entry in FLAG_CHECKS, read back from the built swap. A non-default mode is refused at `--dagger-p 0`, and so is any stage that `add` does not support (the builder enumerates them).
* G-LIVE: `--selfcheck-only` on each arm's own argv and real bank. The chain checker mints the token. G-SUITE runs fresh on the new commit.
* Reachability: the argv to the appended rows to the loss, `file:line` per hop. Mutants must turn RED: no append, appending the logged rows instead of the swapped ones, and replacing instead of appending under `add`.
* Pod dry mint with the real banks before the launch.

## 6. Cost (ESTIMATED; the build's census sets the step cost)
* Pod: RA_s0 + RA_s1 stage B at about 1.08 x RH's 569 s, plus selfchecks, about 0.4 pod-h of GPU in all; G-SUITE on the pod CPU, about 0.6 h, no GPU. A projection above 0.6 pod-h of GPU goes back to the PI.
* Dev box: 12 rollouts and the T0 reads, about 2.5 h wall (RL's read took about 2 h 20 min, 09:44Z to 12:05Z).

## 7. Decisions for the PI
1. The run go (about 0.4 pod-h of GPU plus the dev-box reads).
2. Queue: behind the other session's current run on the LoRA pod, through the `/root/N1DW_READY` marker protocol.
