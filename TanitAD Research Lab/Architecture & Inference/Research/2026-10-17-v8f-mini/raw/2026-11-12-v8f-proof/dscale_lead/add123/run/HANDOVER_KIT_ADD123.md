# HANDOVER_KIT_ADD123: running card PREREG_V8F_PROOF_ADD123 (DSCALE-LEAD-2X) on the pod of the tree owner session

Tier: T0 (probes on cached tokens of logged frames; it says what the tokens allow, not what a model does with them). No model is trained. Nothing is deleted or moved by the kit.
Go of the PI: 2026-11-22, "Yes, after the seeds" (the two model seeds of that session have to have ended first). Budget: about 0.85 pod-h of lock time (training process ESTIMATED 0.70 h, a 200-step smoke, twelve extracts).
Stop rules: the smoke projection of the training process above 0.9 h (exit 4, text `ESTIMATE-ABOVE-0.9H`), or the training process above 3,300 s (text `TRAIN-ABOVE-TIME-LIMIT`). Either stop goes back to the PI, with `$OUT/DSCALE_2X_STOP.txt` and `$OUT/SMOKE_ESTIMATE.json`.
The kit is the kit of card ADD45 (`HANDOVER_KIT.md`, `HANDOVER_KIT_N3.md`) with three new files; the study tree, the banked probe tool and every input are those of that run.

## 1. What to copy to the pod

You already hold the banked kit (card ADD45) with the study tree. Check the banked files by sha256 (first 16) before use; the new pins file checks them in full:

| file | sha256 (first 16) | state |
|---|---|---|
| `ship/dscale_tree.tar.gz` (unpacked to `<KIT>/tree`) | 24392737b179e1de | banked, as before |
| `ship/p10_prior.pt` | 14a062a91078266a | banked, as before |
| `tok_probe_u.py` | 61073f5f7753c8fe | banked, UNCHANGED (never edited, never copied under another name) |
| `dscale_precheck.py` | e1e829be8e69262c | banked |
| `dscale_clockcheck.py` | a0575682288f98a9 | banked |
| `dscale_tokens_same.py` | 01d133312ed785ab | banked |
| `dscale_evalview.py` | 5d50a2d07411cee2 | banked |
| `STUDY_EVAL_MD5.tsv` | 0c804c23593d0b50 | banked |
| `STUDY_TRAIN_MD5.tsv` | 32659dcdf9210323 | banked |
| `tok_probe_u_ck.py` | 5888527b5c37003b | NEW: the saving tool |
| `pod_dscale_lead_2x.sh` | a01c5ca8cb43a2c5 | NEW: generated from the banked `pod_dscale_lead_run.sh` (90f4cdc236840c56) by `make_2x_script.py` |
| `TREE_PINS_2X.txt` | ad681c67505bfed3 | NEW: the banked pins plus `tok_probe_u_ck.py` (11 pins) |

New files come from `F:/TanitAD-scratch/v8proof/dscale_lead/add123/run/` on the dev box; put the three beside the banked kit files in `<KIT>/` (the old `pod_dscale_lead_run.sh` and `TREE_PINS.txt` stay there, unused by this run).
`tok_probe_u_ck.py` finds `tok_probe_u.py` beside itself (or in the folder given by env `TP_DIR`).

## 2. Variables (same as the banked kit; OUT is new)

```
export KIT=/root/dscale_lead/kit TREE=/root/dscale_lead/kit/tree OUT=/root/dscale_lead/out_2x      # OUT must be NEW (not out1)
export P10=$KIT/p10_prior.pt
export TRACKS=<native tracks .pt that covers all 14,512 union clips>                              # as in the run of card ADD45
```
Everything else as in `HANDOVER_KIT.md` section 2 (defaults for PY, CACHE, EVAL_CACHE, the union files, LOCK, THREADS, the memory floor). New defaults: `MAX_H=0.9`, `TRAIN_LIMIT_S=3300`, `PINS=$KIT/TREE_PINS_2X.txt`. `DRY=1 bash $KIT/pod_dscale_lead_2x.sh <stage>` prints every command and runs nothing.

## 3. Stages, in this order

1. `bash $KIT/pod_dscale_lead_2x.sh precheck`  CPU only. The prechecks of the banked kit with their planted failures (shared evaluation id, shifted clock, swapped evaluation ids), in the NEW output folder (control q0). A stop here: send `$OUT/precheck/` back before anything else.
2. `... cpucheck`  CPU only, no lock: two steps on the union and two with the corpus list.
3. `... smoke`  200 steps of the four arms under the lock with the banked tool; projection = seconds / 200 x 38,328 in hours for the ONE training process; above 0.9 h: stop, exit 4.
4. `... run`  folder `runQ`: the arms `TK16_s0` to `TK16_s3` (the card's `P16Q_s0` to `P16Q_s3`) at 38,328 steps, ONE GPU process under the lock, through `tok_probe_u_ck.py --ck-steps 19164,28746` under `timeout 3300`. It writes beside each `ckpt_last.pt` the files `ckpt_step19164.pt` and `ckpt_step28746.pt`, and `runQ/CK_RECORD.json`.
5. `... extract`  per arm three extracts under the lock: the last state into `$OUT/states/runQ`, step 19,164 into `$OUT/states/runQ_s19164`, step 28,746 into `$OUT/states/runQ_s28746` (twelve in all).
6. `... checklog`  CPU; writes `$OUT/DSCALE_2X_DONE.txt` (first line `DSCALE_2X_DONE <UTC time>`; sha256 of the tools and of the twelve checkpoints).

Every stage refuses under the memory floor, is resumable through markers, never overwrites a folder, and writes `$OUT/DSCALE_2X_STOP.txt` with the reason on a stop. The log is `$OUT/dscale_2x.log`. Do not touch a pod that trains: the kit takes the lock of the other session and waits for it.

## 4. What comes back to the coordinator fork

A tar of `$OUT` WITHOUT any `.pt` file, for example (run in the folder that holds `$OUT`; the exclude keeps the checkpoints on the pod):
`tar --exclude='*.pt' -czf out_2x_nockpt.tar.gz out_2x`
It must hold: `precheck/`, `SMOKE_ESTIMATE.json`, `dscale_2x.log`, `markers/`, `states/` (all three folders `runQ`, `runQ_s19164`, `runQ_s28746`), `checklog/`, `DSCALE_2X_DONE.txt`, `runQ/r1_batches.json`, `runQ/TRAIN_DONE.json`, `runQ/CK_RECORD.json`, and per arm `runQ/<ARM>/train_log.jsonl` and `train_meta.json`.
Put it under `F:/TanitAD-scratch/v8proof/dscale_lead/add123/back/` (or name another dev-box path) with its sha256. The checkpoints stay on the pod (their sha256 are in the done file).

## 5. State of the checks

Run on the dev box, CPU only, on tiny synthetic data and on fakes of the interpreter and of flock (results: `helper_pytest.out`): the saving tool (last weights with and without it bit-equal; a step file bit-equal to the last weights of a run stopped at that step with the same schedule function; the files load through `load_probe`; a step not below the total refused with exit 3; `CK_RECORD.json` keys), the generator (stops on a missing or doubled anchor), the script (dry run on every stage creates no file; the smoke bar and the time limit stop it with their reasons; the arguments of the training and of the twelve extracts; the done file).
Not exercised anywhere: the real study tree under the saving tool (the tests use a stand-in head and lead term, which the tool never touches), the GPU path, the pod's lock and its `timeout`, the union files. Tier T0 throughout: no model is trained, and nothing is deleted or moved.

## 6. Checks by the coordinator fork after the build (fork pod, CPU only, 2026-11-22)

* Test files on the fork pod (Linux, its pytest environment): `test_ck_tool.py` 7 passed; `test_2x_script.py` 49 passed against the banked source 90f4cdc236840c56 (two test-harness lines were made portable for Linux first: the run bit of the stand-in programs and the interpreter path; the tools are unchanged).
* The saving tool under the real study tree (the shipped tree, the corpus cache as stand-in for the union, two arms, 6 steps, CPU): with and without the saving tool the last weights are equal bit for bit and the log records are equal; the step files carry their step and arm; `tok_probe_u.py extract --ckpt <step file>` reads a step file and its state record carries the step.
* Smoke margin: the banked smoke of 2026-11-20 read 15.69 s for 200 steps of four arms, which projects to 0.835 h for this run, 7 per cent under the bar of 0.9 h; the banked four-arm run itself went at 0.0653 s per step (0.70 h for this run). A smoke that reads slower than about 16.9 s stops the kit with exit 4. That stop is a stop of the card's rule, not a fault: send `SMOKE_ESTIMATE.json` back, the question then goes to the PI.
* Inherited from the banked script and left as it is: the smoke stage does not check that the seconds read from `TRAIN_DONE.json` are a number.
* Not exercised anywhere: the GPU path, the lock and `timeout` of the pod of the tree owner session, the union files with the saving tool.

## 7. The read

The reader (`dscale_lead/add123/read/read_dscale2x.py`) runs on the pod of the coordinator fork, where the read of card ADD45 ran; it reads the controls q0 to q6 first and opens names and values only after they are logged. The row that reports the card is read by the tree owner session before it is landed.
