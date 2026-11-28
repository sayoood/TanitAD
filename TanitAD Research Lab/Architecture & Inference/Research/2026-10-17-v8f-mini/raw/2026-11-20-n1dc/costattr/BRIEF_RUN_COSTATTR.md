# Brief: run the frozen N1DC cost-attribution card (dev box CPU; zero training)

You are a TanitAD build agent working for the Master Mind (MM). Read `Project Steering/AGENT_OPERATING_STANDARD.md` from the mirror tip first:
`git -C C:/Users/Admin/tanitad-push show $(git -C C:/Users/Admin/tanitad-push ls-remote origin agent/arch-inf-20260803 | cut -f1):"Project Steering/AGENT_OPERATING_STANDARD.md"`.

Rules:
* Stage only: no commit, no push.
* No GPU. No edits to protected files (launch_gate.py, stack/tanitad/models/__init__.py, anything under E:/Projects/TanitAD/stack, the sha-pinned v8f_t1_eval.py).
* No secrets, and no IP, port or pod id in any file.

## The card (FROZEN; do not edit it)
`F:/TanitAD-scratch/n1dc/costattr/PREREG_V8F_N1DC_COSTATTR.md`, sha256 295db8eb (check it first). Sections 2-5 are your spec: arms, read, outcomes, controls.

## Where things are
* The guard chain that ran successfully is `F:/TanitAD-scratch/n1dc/read/eval141/`. Read it; do not change it.
  * Its scripts: `run_n1dc141.sh`, `n1dc_dumps.py`, `n1dc_read141.py`, with the first-run fix in `fix/n1dc_read141.diff`.
  * Inputs: tree `tree/`, checkpoints `ckpts/`, hb0.2 dumps `dumps141_hb02/`.
  * The result to reproduce for K-GUARD: `out/RESULT_N1DC_EVAL141.json`.
* Reuse its dump step for U exactly. It is the same command as the `hb_dump_N1DC_s{s}` steps, with the checkpoint and the tag changed. Read the step lines in `run_n1dc141.sh` and the dump lines in `out/n1dc141_chain.log`.
  * Tag the U dumps so they cannot collide with the existing ones, for example `N1DC_s{s}U`.
  * Write them under `F:/TanitAD-scratch/n1dc/costattr/dumps141_hb02/`, never inside `eval141/`.
* The reader: a new `costattr_read.py` in your folder. It imports the same instrument functions as `n1dc_read141.py` (`navrb2_read`, `strat1_read`, `obd_read`, `obd_geom`, `strat1_lib`), loads with the K-FL-HB rule, and computes P1, P2 and the five controls of card section 5.

## Pod (it is TRAINING another session's job: `pscp`, `ls`, `cat` and `sha256sum` only)
* Run `source C:/Users/Admin/lorapod/podenv.sh`.
* Record `pssh "sha256sum /root/strat1/n1dc/runs/N1DC_s0_zuse/train/ckpt_last.pt /root/strat1/n1dc/runs/N1DC_s1_zuse/train/ckpt_last.pt"`.
* Pull each file with `pscp "$POD_TARGET:/root/strat1/n1dc/runs/N1DC_s{s}_zuse/train/ckpt_last.pt" F:/TanitAD-scratch/n1dc/costattr/ckpts/N1DC_s{s}_zuse/`. They are 87.7 MB each.
* ⛔ Never touch `/root/gpu_queue.lock`, `/root/N1DW_READY`, `/root/ruled`, any process, or anything you did not create.

## Order
1. Check the card sha. Then pull and run K-COPY.
2. Run the two dumps, one process at a time:
   * gate each on dev-box free commit >= 5.5 GB (`F:/TanitAD-scratch/strat1/read/seeds/wait_commit.py 5.5 60`);
   * use 2 threads, `CUDA_VISIBLE_DEVICES=-1`, and `TMP`/`TEMP` under your folder.
   * Each dump takes about 4.5 min. Run each in the background and check on it; never wait more than about 4 min in one call.
3. Run the read. It takes about 2 min.
4. Write `RESULT_N1DC_COSTATTR.json` and `READ_COSTATTR.txt`. If any control fails, set status INVALID and print no label.

## Disk
F: has about 5.3 GB free, and other sessions use it. Check `(Get-PSDrive F).Free` before the pull and before each dump; below 4 GB, stop and report.

When the result is written, delete your own `tmp/` and the two pulled checkpoints. Both are re-creatable: the pod keeps the source, and you checked its sha. Delete by literal absolute path and log each deletion in `F:/TanitAD-scratch/n1dc/DELETIONS.log`. Keep the U dumps; the MM banks them.

## Deliverables in `F:/TanitAD-scratch/n1dc/costattr/`
* the reader and any driver;
* the result and the read text;
* `HANDOFF_COSTATTR.md` (at most 600 tokens);
* `MANIFEST_SHA256.txt`.

Hand back in at most 200 words: the four labels, every control's value, and the paths.
