# RULE1S10K_DEFS.md -- tool set of card PREREG_V8F_PROOF_ADD59 (RULE1S10K)

Tier: T1-kin (kinematic rollout on the model's own actions); never driving performance. Nothing here was run on a pod; checks done on the dev box: `bash -n` on the four shell scripts, `ast.parse` of the reader, a text check of run names, and a text check that the pair list of the reader equals the pair list of the pod read (16 pairs). A `DRY=1` print of the queue (no write) showed the fifteen commands.

## Files
`pod_rule1s10k_queue.sh`, `chain_rule1s10k.sh`, `podread_rule1s.sh`, `wait_podread_rule1s.sh`, `read_rule1s10k.py`, `RULE1S10K_REUSED_FILES.txt` (tab-separated: pod path under the integ10k folder, then the key of the json), this file.

## Run map (wrapper `v8f_t1_eval_reader_rule_rz.py`, sha256 prefix fa7207d72311b5cf; models UE_s0, UE_s1; reader seed = model seed)
Run folders are `t1/<name>_s<k>`; every run gets `--readout` and `--readout-log`; rule runs also `--rule-log`. No `--tau`, `--b-max`, `--a-max`, `--b-lim`, `--slope` (not read from the environment in this queue).

| run | form (argv) | `--readout` | `--replan-steps` | extra argv | seeds |
|---|---|---|---|---|---|
| `I10R0BkID` | `R0` | backward | 5 | none | 0 only (control r1) |
| `I10RLcBk5` | `RLc` | backward | 5 | rule log | 0, 1 |
| `I10RQ50Bk5` | `RR` | backward | 5 | rule log, `--reader` TK16_s<k>, `--qmap` QMAP_LR16_s<k>_q50.json | 0, 1 |
| `I10R0Bk10` | `R0` | backward | 10 | none | 0, 1 |
| `I10RLcBk10` | `RLc` | backward | 10 | rule log | 0, 1 |
| `I10RQ50Bk10` | `RR` | backward | 10 | as RQ50Bk5 | 0, 1 |
| `I10RLcZ5` | `RLc` | causal_z | 5 | rule log | 0, 1 |
| `I10RQ50Z5` | `RR` | causal_z | 5 | as RQ50Bk5 | 0, 1 |

Order: control, comparison (marker `r1scmp`, output `small_rule1s/CMP_R0BkID.txt`), then per model seed the five backward runs (both seeds), then the four ZERO runs. Markers `START_RULE1S10K.txt`, `RULE1S10K_STOP.txt`, `RULE1S10K_DONE.txt`; wait for `$WAIT_DONE` / `$WAIT_STOP` (defaults `BRAKE10K_DONE.txt` / `BRAKE10K_STOP.txt`) in the same folder.
Reused (not rerun): `I10R0B5`, `I10R0Z5`, `I10R0`, `I10RLc`, `I10RQ50`, `I10R0M5`, `I10RLcM5`, `I10RQ50M5`.

## Pairs (A minus B, U/C/F and the four families, pooled over the two model seeds; spread arm I10R0)
- CY: `I10R0B5:I10R0Bk10`, `I10RLcBk5:I10RLcBk10`, `I10RQ50Bk5:I10RQ50Bk10`
- RU at 1.0 s: `I10RLcBk5:I10R0B5`, `I10RQ50Bk5:I10R0B5`; beside, at 2.0 s: `I10RLcBk10:I10R0Bk10`, `I10RQ50Bk10:I10R0Bk10`
- GB: `I10RLcBk5:I10RLc`, `I10RQ50Bk5:I10RQ50`
- RB: `I10R0Bk10:I10R0`, `I10RLcBk10:I10RLc`, `I10RQ50Bk10:I10RQ50`
- ZERO beside: `I10RLcZ5:I10R0Z5`, `I10RQ50Z5:I10R0Z5`, `I10RLcZ5:I10RLc`, `I10RQ50Z5:I10RQ50`
Forms: NORULE = R0 arms, LABEL = RLc arms, READER = RQ50 arms. Names (card section 5) come from U only.

## Controls: where each is computed
- r1: the queue (`cmp_dumps.py`, marker `r1scmp`) writes `CMP_R0BkID.txt`; the reader checks `CMP files=141 differing=0`. The queue stops (`R1-RERUN-DIFFERS`) if it fails.
- r2: reader, from the readout log, the result file (`"replan_steps"`) and the rule log (`replan_steps`, ticks 0 to 5 for the 5-step rule runs).
- r3: reader, from the readout log (`max_abs_speed_minus_last_step` <= 1e-6 in every new run; `max_abs_acceleration_read` <= 1e-6 in the causal_z runs).
- r4: reader, first N steps of `cl` and `fl` of the dumps against the banked run (RLcBk5, RLcZ5 vs RLcM5; RQ50Bk5, RQ50Z5 vs RQ50M5; R0Bk10, RLcBk10, RQ50Bk10 vs R0, RLc, RQ50; largest difference 0.0), and the first-tick inputs `g_src`, `c_src`, `v0`, `v_L` of the 2,399 windows in the rule logs (<= 1e-6).
- r5: reader, 141 dump files in each of the fifteen dumps; 2,399 windows, 642 lead windows from the headway reader.
- r6: reader, sha256 of the 16 result files against `RULE1S10K_REUSED.json`; the key names must equal the expected 16.
- r7: reader controls of the headway and the family reader.
- r8: reader, checkpoint sha256 prefix of the run record; reader and map sha256 of the rule log of the reader-fed runs equal those of the banked `I10RQ50` rule log at the same seed.
- r9: queue (sha256 prefix check before the first rollout), chain (local file) and reader (`--wrapper`, the pod build copy).
- e1: `podread_rule1s.sh`, first step, as in the earlier pod reads.

## REUSED json layout (copy of `HANDOVER10K_REUSED.json`)
`{"card": "PREREG_V8F_PROOF_ADD59", "control": "r6", "files": {"<key>": "<sha256 hex>", ...16 entries...}, "what": "sha256 of the result files of the reused runs, recorded before the queue of the card started"}`; the keys are the second column of `RULE1S10K_REUSED_FILES.txt`; the sha256 is that of the file in the first column under `/workspace/v8proof/integ10k/`. On the pod the reader resolves a key under `<podread>/root/` through the `pull` links.

## What was not checked
No rollout, no reader, no pod access; the wrapper was not run with a rule form together with `--readout` (never run before). The reader was not run (no numpy job): only parsed. Names of npz keys (`reader_sha256`, `qmap_sha256`, `tick`, `g_src`, ...) are taken from the earlier readers and from the wrapper source; whether the banked `I10RQ50` rule logs carry `reader_sha256` and `qmap_sha256` was not opened (the wrapper source writes them).

## Places where I guessed
1. Run folder names `I10R0Bk10` etc.; control name `I10R0BkID` as in the task; reader-fed form argv `RR` as in the cycle queue.
2. The comparison marker is `r1scmp`; no earlier file of the folder holds it (grepped).
3. Order of the rollouts (backward runs of both seeds first, ZERO last): my choice for priority.
4. Wait loop copied from the brake queue (28,800 s limit; it proceeds also after a STOP marker of the previous queue, as that queue does). Chain limit 18 h, waiter limit 15 h: my values.
5. The queue refuses (`NO-BANKED-...`) unless every banked run it compares with has `T1_DONE` on the pod (eight runs, two seeds); the card does not ask for it.
6. r2: for the 10-step rule runs the ticks are reported, not required. Row counts of rule logs are reported in r4, not gated.
7. r4: banked reference of `I10R0BkID` is not compared beyond r1 (card lists no first-block check for it).
8. r8: the model-checkpoint part covers all fifteen runs, the reader/map part the six reader-fed runs.
9. Names of the RB step: `RULE1S10K-<form>-BACKWARD-COSTS-AT-2S`, `-BACKWARD-HELPS-AT-2S`, `-BACKWARD-NEUTRAL-AT-2S` (the card writes the last two as `-HELPS-AT-2S` and `-NEUTRAL-AT-2S`). "Interval lies above/below 0" is `interval[0] > 0` / `interval[1] < 0`; CUTS names also need both per-seed values below 0 as the card says; RAISES, BELOW, ABOVE and the RB names use the pooled interval only (as the card words them).
10. The reader floor 0.0196 is applied to the U cell only (a U floor); every READER line prints "inside/outside the reader floor", and the READER names carry the remark in a separate field (`name_floor_notes`), not inside the name.
11. Levels of the three pinned 1.0 s runs (R0M5, RLcM5, RQ50M5) are not printed: they are in no pair of the card, so the headway reader does not list them.
12. Control r9 in the reader checks only the sha256 of the pod build copy of the wrapper; the card names no tool-check output file for this card.
13. Output paths of the pod read (`e1r`, `read_rule1s`) and the files pulled by the waiter.

## Things in the patterns that looked wrong
- The brake queue's wait loop does not distinguish DONE from STOP (a stopped previous queue lets the next one start). Kept as is, listed above.
- The chain pattern pulls `I10RRx_*` runs although no later queue makes them; dropped here.
- `podread_*.sh` expect the lead to push the pod read script, the reader and the REUSED json into the podread folder beforehand; no chain step does it.
