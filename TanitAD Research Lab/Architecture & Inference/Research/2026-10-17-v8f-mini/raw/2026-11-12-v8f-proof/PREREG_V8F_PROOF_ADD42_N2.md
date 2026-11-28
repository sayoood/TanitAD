# PREREG_V8F_PROOF_ADD42_N2 : the refit is read per arm type as note N1 says; the dose step for the type that missed the coverage band

Note to card `PREREG_V8F_PROOF_ADD42.md` and note `PREREG_V8F_PROOF_ADD42_N1.md`. Frozen after the four fits of note N1 were read (controls of the fits only: coverage, selected step, shift) and before any rollout of note N1. No rollout with a head of note N1 exists, and no outcome number of the rule with these heads is known. Tier: T0 for the fits (heads on frozen states of training clips; no model is trained), T1-kin for the rollouts (kinematic rollout on the model's own actions, logged frames, the lead does not react); never driving performance. STRATEGIC: NOT COMPUTED (n = 0; the arms of this study carry no strategic layer and the reader has no strategic instrument).

## 1. The four fits of note N1 (MEASURED, T0, pod CPU, 56 s to 64 s each)

Rows: fit 21,060, select 5,267, check 5,698. Tool `qhead_fit_n1.py` (sha256 57e1242713e0b42e...), records in `/workspace/v8proof/qhead_run_n1/heads` on the pod, done 2026-11-19T04:05:47Z (pod clock).

| checkpoint | selected step (of 19,164) | shift c | coverage fit | coverage select | coverage check | h5 (band 0.15 to 0.25 on check) |
|---|---|---|---|---|---|---|
| LS_s0 | 250 | -0.1421 m | 0.1437 | 0.2001 | 0.1794 | PASS |
| LS_s1 | 1,000 | +1.2670 m | 0.0381 | 0.2001 | 0.1885 | PASS |
| LSD_s0 | 750 | +2.3126 m | 0.0355 | 0.2001 | 0.1086 | FAIL (below the band) |
| LSD_s1 | 250 | -0.5951 m | 0.1219 | 0.2001 | 0.1876 | PASS |

Stated plainly: the selection stops very early (step 250 to 1,000), so the pinball loss on unseen clips is lowest near the start of the fit; at two checkpoints the shift is above 1 m, so there the level of the quantile comes from the shift more than from the fit; the coverage on the fit clips falls to 0.04 at two checkpoints, which is the same mismatch between fitted and unseen clips that failed the first run. Three of four checkpoints land in the band on clips that entered neither fit nor selection.

## 2. Reading rule (note N1 section 4 as written; nothing is changed)

Note N1 section 4: "per checkpoint; outside it the arm of that checkpoint is not named". The names of the card need both seeds of an arm type.

* **LS type** (`LSQE_s0`, `LSQE_s1`): both checkpoints are inside the band. The two rollouts run and the names of note N1 section 5 are read for the LS type.
* **LSD type** (`LSDQE_s0`, `LSDQE_s1`): `LSD_s0` is outside the band. The type is not named under note N1. Its two rollouts are not run and no outcome number of the LSD type under note N1 is computed, also not for `LSDQE_s1` by itself.
* **Tool change, stated.** The names tool of the build (`qhead_names_n1.py`) withheld every name when any of the four checkpoints failed h5, which is stricter than the note. Before any rollout it is changed to the rule of the note: h5 gates per arm type; h1, h4, h6, h7, h8 gate every name as before; h2 and h3 are read on the arms of the types that are read. The readers are called with the pairs of the read types only. The test of the global h5 gate is replaced by tests of the per-type rule (a type with a checkpoint outside the band gets no name and no outcome number; the other type is read).

## 3. The dose step of note N1 section 6, for the LSD type only

Note N1 section 6 commits the dose step when h5 fails for an arm type. It is run for the type that failed. The LS type is read once, under note N1, and is not fitted again: a second fit of the LS type would be a second draw at the same names.

* **Arms:** `LSDQF_s0`, `LSDQF_s1`: the head and loss of the card, the fit of note N1 (selection every 250 steps on the select clips, constant shift from the select clips), on states of all 4,572 training clips of the manifest in the trainer's order: fit clips 0 to 3,047, select clips 3,048 to 3,809, check clips 3,810 to 4,571 (two thirds, one sixth, one sixth, as in note N1). Thresholds as for the LSD arms of the card (0.4 and 0.3).
* **Volume, named before it runs (ESTIMATED from MEASURED 57 s and 156 MB for 1,200 clips):** states of 4,572 clips for the two LSD checkpoints: about 4 min of one GPU slot and 0.6 GB per checkpoint, 1.2 GB on the pod volume together; the lead read and the label pack for the same clips on the CPU. Never a fourth GPU job; 20 GB of the container memory limit stay free.
* **Controls:** those of note N1, with h5 on the new check clips; and
  * d1: the label pack built for all 4,572 clips equals the banked pack of the 1,200 probe clips row for row on manifest positions 600 to 1,799 (largest absolute difference 0);
  * d2: the states of positions 600 to 1,799 in the new files equal the banked 1,200-clip states of the same checkpoint within 1e-4 of the root mean square of the state, and the same comparison with the clip order shifted by one clip must fail;
  * d3: the record of each head lists its three clip sets, and no check clip enters fit, selection or shift (h8 on the new sets).
* **Names:** those of note N1 section 5 for the LSD type, written with the suffix `-QF`.

## 4. What follows (committed before the data)

* LS type named under note N1: card section 8 applies to the LS type.
* LSD type: h5 passes at both checkpoints under the dose step: the rollouts run and the names are read. h5 fails again at 3,048 fit clips: the state of the LSD checkpoints does not carry a transferable 0.20 quantile for this head; the LSD type keeps the calibrated map, and no further fit variant is run without a new card.
* Both types end without a name that cuts the unsafe rate: the lever "trained low quantile on the frozen state" is put down for these checkpoints, and the next lever for distance keeping is the one in the training queue (the level term of note ADD37_N6 with the consistency term, read on the model's own plans by card ADD40).

## 5. Limits

One fit per checkpoint and seed; the coverage bar is read on 200 check clips (note N1) and on 762 check clips (dose step); the LS type is read at 800 fitted clips and the LSD type at 3,048, so the two types are not compared with each other under one fit size; the tool change of section 2 was made after the fit controls were seen and before any rollout existed.
