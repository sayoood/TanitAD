# PREREG_V8F_RECLAB (FROZEN 2026-11-22, before any RL step, pod census or number): the DAgger swap's label on own states starts at the own speed

## FREEZE RECORD (MM, 2026-11-22)
* Frozen after the zero-training build (three passes, `FACTS_RECLAB.md`, `FACTS_RECLAB_2.md`, `FACTS_RECLAB_3.md`) and the one-shard construction census, before any RL training step, pod census, rollout or read. No RL number exists.
* Code: a new commit on `agent/v8f-rollhist-cdb-20261120` on top of 97fe0a56 (files and sha256 in `CHANGED_FILES.txt`). RH and B0 stay at 97fe0a56; the gold test proves `--dagger-label human` reproduces 97fe0a56's swap bit for bit.
* Running the arms needs a PI go with the cost (section 8).

## 0. Why (the measured case, quoted, not re-measured)

* Three arms on the ROLLHIST line, all MEASURED, tier T1-kin/hb0.2 with G-T0 at T0, paired episode-cluster bootstrap B 2,000 seed 0 (`F:/TanitAD-scratch/rollhist/read/out/read*/GT0_ROLLHIST*.json`):

| arm | bank, dose | G-T0: arm - B0, T0 ADE6 [interval], s0 | s1 | label |
|---|---|---|---|---|
| RH | C\*, p 0.1 | +0.0503 m [+0.0060, +0.0929] fail | +0.0217 m [-0.0122, +0.0596] pass | ROLLHIST-PARTIAL |
| RHp3 | C\*, p 0.3 | +0.1124 m [+0.0337, +0.1991] fail | +0.0939 m [+0.0309, +0.1628] fail | ROLLHIST-HARMS |
| RHK | K, p 0.1 | +0.0641 m [+0.0341, +0.0962] fail | +0.0474 m [+0.0194, +0.0753] fail | ROLLHIST-HARMS |

  The guard bar is an upper interval bound of at most +F_T0 = +0.0705 m (INHERITED, one seed pair; card section 3.2).
* At T1 the same arms buy real own-action gains (RHp3 - B0: STRATEGIC -2.01 m, ade_tac -0.41 m, yaw rate -8.37 deg/s; RHK - B0: STRATEGIC -1.14 m). The T0 cost grows with the dose and does not depend on the bank's readout (`MM_VERDICT_RHK.md`, "What it says").
* What all three share is the LABEL. `swap_rows` re-anchors the own poses by the rigid motion that maps the own NOW onto the LOGGED NOW at r' (`v8f_dagger.py` `reanchor_rows` :161, in `swap_rows`), and the scene and nav channels 0-2 and 4 stay logged. So the own POSE offset is erased from every input: NOW sits on the human's path with the human's heading, and the human label at r' agrees with the inputs in position and heading. What the re-anchoring keeps is the own KINEMATICS: the own history's speed (`p10f`, `kin3`), while the label `gt` (`v8f_mini_data.py:681`) starts at the human's speed. Only `gt` and `s_h` are derived from the future waypoints (`FACTS_RECLAB.md` section 1).
* So `H-ST-RECOVERYLABEL-1`'s pose recovery (a blend back to the path from the own pose) does not fit this swap: it would tell the model to rejoin a path that every input shows it is on, a label no input predicts. Its speed rule (start at the own speed, rate-limited toward the human's) does fit, and it is the part this card takes.

## 1. Hypothesis (HYPOTHESIS, committed before data)

**H-V8F-RECLAB-1.** The T0 cost of own rolled history is a speed conflict in the label. In a swapped sample the own history says "this car moves at v_own", and the label says "it moves at v_h from now on". On logged history, continuing one's own speed is exactly the mapping T0 scores. A label that starts at the own speed and rejoins the human's speed at a feasible rate along the human's path keeps that mapping. So, at the same bank and dose as RH, the speed-consistent label lowers the T0 cost (RL - RH at T0 ADE6 < 0) and keeps RH's T1 gains.

Two named risks:
* **R1, the label moves the target.** Along the path, the label lags or leads the human by the integrated speed deficit. If T1 reads the human's timing as the target, longitudinal cells (along_mae, speed_mae) can worsen while T0 improves.
* **R2, the conflict is elsewhere.** The own history also differs in yaw rate and acceleration, which this label leaves alone. If the T0 cost lives there, MECH reads no.

## 2. Arms

| arm | what | argv against RH_sK | init |
|---|---|---|---|
| **RL_s0** (the lever) | RH's stage B with the speed-consistent label | exactly `--dagger-label speed_v1` added, `--out` changed; the RH bank `/root/rollhist/bank/s0` (sha256 e5e44c01...) | UE_s0.A, as RH_s0 |
| **RL_s1** (replicate) | as above at seed 1 | the same two tokens; the RH bank `/root/rollhist/bank/s1` | UE_s1.A |
| comparators, no new cost | RH_s0/s1 and B0_s0/s1 (banked runs and dumps) | | |

* Code: a new commit carrying `--dagger-label`; default `human` reproduces 97fe0a56's swap bit for bit (gold test). RH and B0 stay at 97fe0a56; the gold test is what makes RL - RH a one-lever contrast.
* Semantics (MM design, `BRIEF_BUILD_RECLAB3.md`), for matcher-ok records only; off-route, end-edge and non-finite records are dropped exactly as today, and the eligible index is equal under both flag values:
  * human step speeds u_h[k] from the human pose polyline's own chords (rows r', r'+1, ... in the human's ego frame at r');
  * label step speeds: u_l[0] = the own speed the model reads (`p10f[:, -1, 0]`); each step moves toward u_h[k] by at most A * 0.2 s with A = 3.0 m/s2 (0.75 of A_MAX 4.0, `feasible_decode.py`), and follows u_h once merged;
  * label point k = the human polyline at the cumulative label arc; after the merge the label is the human's path shifted along itself;
  * refused (today's label kept) when the arc runs past the clip's logged path, is not non-decreasing, or fails V1's feasibility checker; identity (u_l equal to u_h at every step) gives the human label bit for bit;
  * `s_h` re-derived from the label; every other label unchanged.
* **Pre-freeze construction check (MEASURED, no training, no RL number):** a first form took u_h from the store's speed column while placing points on the pose polyline. On bank/s0 shard 0 (372,791 ok records) it relabelled 67.4 % of ok records (tick 1: 84.9 %, tick 2: 45.2 %), and 10,689 of 11,378 refusals in a 40,000-record sample were acceleration violations from mixing the two speed sources at 0.99 A_MAX (`REAL_SHARD_CENSUS.json`, `REFUSAL_DIAG.json`). The form above uses one source and a margin. Its census on the same shard (`REAL_SHARD_CENSUS_2.json`, MEASURED): relabelled 54.2 % of ok records (tick 1: 38.8 %, tick 2: 73.7 %), identity 43.1 % (the own speed is within one 0.6 m/s rate step of the human's, so the label equals the human's), refused 2.7 % (path end 4,719; checker 5,479, of which curvature 4,294 and acceleration 1,250, terms overlapping); |v_own - u_h| p50 0.77 m/s, p95 6.22 m/s; the label's shift along the path at 6 s p50 0.035 m, p95 6.33 m. So the lever acts on about half of the own states and is metres-scale only in the tail.

## 3. Reads and bars (all at `T1-kin/hb0.2`, G-T0 at T0; the ROLLHIST card's section 3 binds unless changed here)

* Executor, windows, estimator, cells, cell and family verdict rules, F: as ROLLHIST card sections 3 and 3.2 and Amendment 8, unchanged. Estimator: paired episode-cluster bootstrap, B 2,000 (`taniteval/ci.py`); it answers "another draw of EPISODES"; the two seeds answer "another TRAINING run"; inference is deterministic.
* **P1 (lever, decides):** RL - B0 under readout C\* (`R0C10XHb20`), all four families, STRATEGIC in full.
* **G-T0 (guard, decides):** RL - B0 T0 ADE6, upper interval bound <= +0.0705 m, per seed.
* **MECH (mechanism, decides the next lever only):** RL - RH at T0 ADE6, paired, per seed. MECH-YES when the interval sits below 0 at both seeds.
* **ATTR (beside, no bar):** RL - RH under `R0C10XHb20`, four families; P2, P3, P4 as the ROLLHIST card on RL.
* **Lever-acts check (known value):** from the trainer log, the share of swapped samples whose plan label differs from the human's is > 0 for RL and exactly 0 for RH's argv on the new code.

## 4. Pre-launch census (zero training, on the RH banks, from the trainer's `--selfcheck-only`)

Per tick: ok-relabelled, ok-identity, ok-refused (by reason); p50 / p95 of |v_own - u_h[1]| and of |L[30] - s[30]| (the label's shift along the path at 6 s); the relabelled share of ok records. The pod census on the full banks (both seeds, all shards) decides; the one-shard census in section 2 is one of four shards of s0. **RECLAB-BLOCKED** if the relabelled share of ok records is below 0.5 at either seed (the lever would act on too few samples to test the hypothesis). The one-shard dev-box census of the frozen form is `REAL_SHARD_CENSUS_2.json` (section 2).

## 5. Outcomes, committed in advance, mutually exclusive, two seeds (a one-seed read prints the same name prefixed PROVISIONAL)

| name | condition | reading | next lever |
|---|---|---|---|
| **RECLAB-REPAIRS** | G-T0 holds at both seeds; P1 at least one family BETTER at both seeds, none WORSE at either | the label was the T0 cost; own rolled history is admissible | the dose that bought the larger T1 gains: p 0.3 under the speed-consistent label (stage B only on the same bank, about 0.27 pod-h per seed) |
| **RECLAB-HARMS** | G-T0 fails at both seeds, or a family WORSE at both | own rolled history still costs under this label | if MECH-YES: the label helps but not enough: extend it to the rest of the own kinematics (a yaw-rate and acceleration blend from the own history into the human's, under the same feasibility envelope); if MECH-NO: the label is not the cause: keep the logged samples whole and ADD the own states (own-state samples appended to the batch instead of swapped in), which separates interference from replacement |
| **RECLAB-PARTIAL** | G-T0 holds at one seed only, or holds at both while P1 shows no family BETTER at both seeds | part of the cost is the label | if MECH-YES: p 0.05 under the speed-consistent label (half the dose) to find the dose that holds G-T0 with the gains; else as HARMS with MECH-NO |
| **RECLAB-SEED-DISAGREE** | a family BETTER at one seed and WORSE at the other, or the decisive cells straddle the floor | not decidable on two seeds | a third seed pair needs a UE_s2.A stage A: back to the PI |
| **RECLAB-BLOCKED** | census below 0.5, a gate check fails, or the lever-acts check reads 0 for RL | the arm does not launch, or its number is not read | fix and re-gate; if the census is low: report the refusal reasons and widen the label to the refused class first |

Precedence: BLOCKED, then HARMS, then SEED-DISAGREE, then the rest in the order of the table.

## 6. Launch gate and reachability

* The flag's G-DVB entry lives in `v8f_mini_train.py` FLAG_CHECKS (ROLLHIST card section 4.1); `--selfcheck-only` on each arm's own argv and real bank is G-LIVE; the chain checker mints the token. G-SUITE fresh on the new commit.
* Reachability (`FACTS_RECLAB_3.md` section 2, file:line at the new commit): `--dagger-label` `v8f_mini_train.py:243` -> `load_dagger_bank` `:606` -> `DR.label_bank` `v8f_recovery.py:333` (`speed_label` :287, per ok record :359) -> `bank.attach_recovery` `v8f_mini_train.py:607` / `v8f_dagger.py:285` -> gather `v8f_dagger.py:339` -> `swap_rows` `:611-618` (the one install line `gt[rows[hm]] = label` :617; `s_h` by `arc_from_label` `v8f_recovery.py:255`, :618) -> `DaggerSwap.apply` counter `:761` -> loss `v8f_mini.py:1182-1190` (ade, fde, acc), `:632` and `:1215` (`s_h`). MEASURED on the shard: 202,076 labels differ from the human's under `speed_v1`, 0 attached under `human`, eligible index equal (372,791 both). Mutations RED, 12 of 12 (`mutants/MUTANTS_RESULT.json`), among them: no install line; the human speed in place of the own; refusal skipped; speed jump unclipped; the rate at A_MAX with no margin.
* Pod dry mint with the real banks before the launch (memory: dry-mint first; a dry step's rc 0 proves nothing).

## 7. Cost (ESTIMATED)

* Pod: RL_s0 + RL_s1 stage B, about 0.27 pod-h each (RH's measured wall), G-SUITE and dry mint about 0.3 pod-h: about 0.9 pod-h in all. A projection above 1.1 pod-h (+20 %) goes back to the PI.
* The full-bank census comes from each arm's `--selfcheck-only` in the pod chain, before any training step (label_bank about 20 s per shard). Dev box: the reads (12 rollouts and the T0 reads, about 5 h wall, as RHK).
* F: at most 1.5 GB during the build; the reads reuse the banked RH and B0 dumps.

## 8. Decisions for the PI

1. The run go (about 0.9 pod-h plus the dev-box reads).
2. Queue: after the other session's current training on the LoRA pod, through the `/root/N1DW_READY` marker protocol.
