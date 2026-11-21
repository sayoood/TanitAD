# AMENDMENT 1 to PREREAD_V8F_TOPK_PAIR (control 5) and PREREAD_V8F_CONSORC (control 6): the memory guard only (Master Mind, 2026-11-10 15:38)

Written at 15:38:52 (file time), recorded in `AMENDMENT_1_MEMORY.sha256` at 15:39:22, before the pair read's PARITY_DAGOACC_s1 and OB-CTL_s0 arms and before any CONSORC arm ran. PARITY_DAGOACC_s0 of the pair read was already read (S 0.239); nothing in this amendment depends on a read number.

## What changes
The two memory guards, and nothing else. Frozen: 10 GB free before each arm, 7.5 GB before each chunk (6 h / 1 h limits). Amended: **4 GB free before each arm, 3 GB before each chunk** (same limits). Every other control, decode, estimator, band, outcome rule and trigger stands as frozen.

## How
`theory/consorc/mem_amend.py` loads the frozen script file unchanged (its sha256 still equals the card's `.sha256` line; the wrapper prints it) and replaces only the module's `wait_free` thresholds: a call with gb >= 10 becomes 4.0, any other call becomes min(gb, 3.0). Runner: `theory/consorc/run_amend1.sh` (pair read PARITY_DAGOACC_s1 + OB-CTL_s0 into `topk/out/TOPK_PAIR_s1ctl.json`, then the CONSORC sha check, smoke + controls, full read).

## Why
* The frozen guard's stated reason was "so the planner gates' 6 GB guard keeps its margin". That margin does not exist: the dev box sat at 4-6 GB free from 07:17 to 15:40 (the PI's desktop apps and another session's scorer job hold most of the 31.8 GB), and the planner gates failed units on their own 6 GB guard while no read was running (`ladder_plantest/run_lp.log`, e.g. 14:40:18 w2A p1 C0_s0 rand). The 10 GB guard blocked both reads for 8.4 h and protected nothing.
* MEASURED footprint of one pair-read arm (`topk/out/run.log`, PARITY_DAGOACC_s0): free fell from >= 10 GB at the start to 9.1 GB at the first chunk and 8.8-9.2 GB through the arm, about 1.2 GB, for 106 s of forward passes plus the bootstrap. A CONSORC arm runs 11 decodes per window and is ESTIMATED larger; its chunk guard pauses it at 3 GB.
* Log: 15:39:27 pair read started (wrapper printed the frozen sha256 8f60e3c8...). About 15:39:40 the header stamp was corrected from "15:45" to the file time; the text above it is unchanged, and the sha256 file was rewritten for this file only.
* Cost to the planner gates: a unit that starts while a read arm runs waits on its own guard (limit 1 h) or fails and is re-run (the run is resumable: a unit whose npz exists is skipped). No planner result changes.
