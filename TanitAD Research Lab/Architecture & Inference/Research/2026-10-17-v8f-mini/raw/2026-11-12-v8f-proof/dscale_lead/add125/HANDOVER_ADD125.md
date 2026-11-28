# HANDOVER_ADD125 — the union probe at equal passes (card ADD125, DSCALE-LEAD-EQP)

Card: `PREREG_V8F_PROOF_ADD125.md` 9dfbc7f38e665b06, frozen 2026-11-23 before any tool of the card and before any run. Tier T0: a probe on cached tokens, not driving performance. The run needs the go of the PI and is run by the session that owns the pod, under its lock and after its own queue.

## 1. Kit

No new run kit. The kit of card ADD123 as it is installed on the pod is used unchanged:

| file | sha256 (first 16) |
|---|---|
| `pod_dscale_lead_2x.sh` | a01c5ca8cb43a2c5 |
| `tok_probe_u_ck.py` | 5888527b5c37003b |
| `TREE_PINS_2X.txt` | ad681c67505bfed3 |

Please compare these three hashes on the pod before the start.

## 2. Settings

As in `HANDOVER_KIT_ADD123.md` section 2 (KIT, TREE, TRACKS and the defaults), with four differences, all through the environment:

```bash
export OUT=/root/dscale_lead/out_eqp      # a NEW folder (not out_2x)
export STEPS_U2=61568                     # 4.0 passes at 15,392 steps per pass
export TRAIN_LIMIT_S=6600                 # time limit of the one training process
export MAX_H=1.5                          # stop rule of the card: a smoke projection above 1.5 h goes back to the PI
```

Inside `$OUT` the script names the run folder `runQ` as before; the card's arm names are `P16E_sK` = `out_eqp/runQ/TK16_sK`. The states at 19,164 and 28,746 steps are written as before and are not read for a name. The message of a smoke stop still carries the words of the older bar in its name (`ESTIMATE-ABOVE-0.9H`); the number it compares with is `$MAX_H`.

## 3. Order of work

1. `DRY=1 bash $KIT/pod_dscale_lead_2x.sh <stage>` for the stages first, with the four settings exported, and read that the lines print 61568 steps, the 1.5 h bar and the new folder.
2. The stages in the order of card ADD123: `precheck`, `cpucheck`, `smoke`, `run`, `extract`, `checklog`.
3. Expected: about 0.070 s per step (MEASURED on the run of card ADD123: 2,687 s for 38,328 steps), so about 1.2 h for the training process and about 1.3 pod-h of lock.
4. Back to the fork: the output folder without the checkpoints as one tar (`out_eqp_nockpt.tar.gz`) with its sha256, the content as for card ADD123 (`precheck/`, `SMOKE_ESTIMATE.json`, the log, `markers/`, `states/` with its three folders, `checklog/`, `DSCALE_2X_DONE.txt`, `runQ/r1_batches.json`, `runQ/TRAIN_DONE.json`, `runQ/CK_RECORD.json`).

## 4. Read

The reader of this card is built by the fork after the go and before any value is opened; it runs on the fork pod against the banked runs of 2026-11-20 and of card ADD123. No value of the new run is opened before its controls e0 to e6 pass and are logged. The log check of the older card will again report its fall clause on short tail batches; that clause is not a control of this card.
