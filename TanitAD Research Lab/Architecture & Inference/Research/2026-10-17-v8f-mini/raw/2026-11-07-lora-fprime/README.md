# F-prime for the LoRA card (B-prime eval-141 on box 2, 2026-11-07)

* `out/`: the identity eval (banked M2b_s0 re-scored on box 2) and B-prime s0 / s1, by the invocation of record (`v8f_mini_box2_queue.cmd:219-232`), with `FPRIME.json` / `FPRIME.txt` from `compute_fprime.py` (banked in `raw/2026-11-07-t2-pair-fprime-prep/bpeval/`).
* **F-prime = 0.12701 m, 2F-prime = 0.2540 m.** The identity eval passes all 8 checks.
* `bpeval_run.out`: the driver's log. The driver ended `PULL-FAILED` after all three evals had ended rc 0 on box 2. Its pull step looked for `out/` one level below the folder `sftp get -r` had filled; its dry-run stub had created that extra level, so the dry run could not see it. The 12 files were compared by sha256 with box 2 (`pull_042100.local.sha`, `pull_042100.box2.sha`, all equal) and moved by hand. `compute_fprime.py` then ran unchanged.
* `dk/`: distance keeping for B-prime (`dk141.py`, banked in `raw/2026-11-07-dk141/`), T1-class open loop.
* `AMENDMENT_V8F_LORA_4_LOG.md`: the card log as of this read, with the F-prime entry (the copy cited in report v42 is the earlier state).
