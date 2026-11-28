# HANDOFF: TOKPROBE tool (card PREREG_V8F_PROOF_ADD27), dev box, 2026-11-15

State: DONE on the dev box (CPU, synthetic data). Nothing touched the pod, box 2, git or a GPU; nothing staged or committed. No number of the card was computed. Tier T0; STRATEGIC: PAUSED (PI item 25). Folder `F:/TanitAD-scratch/v8proof/tokprobe/`. All decisions beyond the card: `TOKPROBE_DEFS.md` (written before the tool was run).

## Files and sha256 (re-hash before use; the two marked POD go to the pod)
| file | what | sha256 |
|---|---|---|
| `tok_probe.py` **POD** (to `/workspace/v8proof/probe/tok_probe.py`) | `train`, `extract`, `check-log`; LF | 42eeb809e8d1bce00baf7e8620686fbcc6051e92a0edbad4a00912ad21e5b24e |
| `pod_tokprobe_run.sh` **POD** (run by the lead) | memory guard, finished-marker guard, train, extract x4, p3 short run, DONE/STOP; LF; `DRY=1` | b42983e09a5e0c7393c23c6ecf6b500d861b7cf16160fa816e7b83edc59e5611 |
| `TOKPROBE_DEFS.md` | decisions, each with file:line of the tree | (not a pod file) |
| `test_tok_probe.py` | 14 tests | (not a pod file) |

The sha256 of `tok_probe.py` and `pod_tokprobe_run.sh` above are at hand-off; the pod script writes the pod-side sha256 of `tok_probe.py` into `TOKPROBE_DONE.txt`. No other file is needed on the pod (the tree `/workspace/v8proof/tree_leadsup` and the cache exist).

## Verified (`python -m pytest -q -s test_tok_probe.py --basetemp F:/TanitAD-scratch/v8proof/tokprobe/tmp/pt/<new>`: 14 passed, 103 s)
* p1: first-step gradients non-zero on the per-token network and the head, exactly 0.0 on `ctrl` (outside the graph), also in the log of the short real run; the mutation that replaces the one `# LOSS-LINE` makes the check read 0.0 on the probe parameters and fail (module rebuilt from mutated source).
* p2: planted tokens (gap and closing speed planted along fixed directions, 500 steps, held-out clips): R2 gap 0.955, closing 0.947 (bar 0.9). Both quantities are planted (the card says "a known gap"; the closing speed is read from five frames of a gap-only plant much less well and was not tested).
* p3: the flag's own function (`shuffle_closing`) on the planted set: held-out closing R2 -0.39 (bar at most +0.02), gap R2 still 0.93, unshuffled 0.95; on the trainer data path the flag permutes the finite closing labels among the labelled rows, the item reads the permuted array, and a short CLI run writes `TK16SH_s0`.
* p4: `forward(self, tokens)` is the only signature (a second argument raises); labels replaced by noise give identical outputs; extract run twice with `D.lead_sup_label_arrays` replaced by a noise function: never called, outputs byte-equal.
* p5: `check-log` goes red on a rising loss, on a zero gap count and on an unfinished log (synthetic logs); the short real run (4 arms, 400 steps, window 100) passes. For a real run the window is 1,000.
* The four probes differ (seeds give different weights, same shapes) and see the same batches (a spy on `step_probe` records an identical batch hash per step for all four); the same seed on the same batches gives identical weights.
* Data path: the dataset comes from `TR.build_data` + the three lines of TR:2819-2821; the parsed default argv gives lr 3e-4, wd 1e-2, warmup 200, clip 1.0, batch 64, steps-mult 4, exclude 540:600, seed 0; `LRSchedule(3e-4, 200, 19164)` endpoints checked.
* Extract: files load with `leadsup_read.load_states`; `leadsup_read.check_records` passes (shas, applied-to-pulled-states, one checkpoint sha across the four records); the reader's own head forward (`row_outputs`, `load_head`) on the stored `Z` equals the stored gap, closing speed and presence logit within 1e-4; `history_rows` equals `pod_lead_apply.history_rows`; a changed head tensor fails the reader's record check. Dtypes as `extract_states.py` (float16 / int32 / int16).
* Pod script: `DRY=1` prints the 7 commands; a finished marker refuses (exit 3, no STOP written); a missing `tok_probe.py` writes `TOKPROBE_STOP.txt`; a full non-dry run on the dev box with the synthetic world (`DEVICE=cpu TRAINER_ARGV=... EVAL_RANGE=0:2 TRAIN_RANGE=0:6`, outputs in `tmp/e2e/`) wrote the 50 state/lead files for the five arms and `TOKPROBE_DONE.txt`.

## Not verified
* **Nothing ran on real data or on a GPU.** The real cache, join, labels and the 19,164-step count are untested here; `--expect-steps 19164` refuses a run whose computed total differs, and `train_meta.json` carries `train_sha256`, `train_window_list_sha256`, label sha256 and counts to compare with LSD_s0's `summary.json`.
* The LayerNorm on the tokens (`--input-norm ln`, DEFS B6) is my decision; raw token scale was not seen. Set `--input-norm none` in the script's train line if the lead prefers the card's literal wording.
* `leadsup_read_n1.py` was not run on a TK arm: its arm list and `TYPES` are `LS/LSH/LSD`. The files are name-generic for `arm_files`/`check_records`, but the type grouping, the paired difference "probe arm vs LSD of the same seed index" and the floor need a small extension by the reader's author or the lead (no number is computed by this tool).
* No resume: a stopped train is rerun.

## Pod commands (the lead runs them; nothing was run)
1. Push `tok_probe.py` and `pod_tokprobe_run.sh` to `/workspace/v8proof/probe/`; compare sha256 on both sides.
2. `cd /workspace/v8proof/probe && DRY=1 bash pod_tokprobe_run.sh` (read the 7 commands), then `nohup bash pod_tokprobe_run.sh > /workspace/v8proof/runs/tokprobe.log 2>&1 &` (after the rollouts of card ADD26; the script refuses under 20 GB container free memory and over a `TOKPROBE_DONE.txt`).
3. Done = `/workspace/v8proof/runs/tokprobe/TOKPROBE_DONE.txt`; stop = `TOKPROBE_STOP.txt` (reason inside; a previous STOP is moved to `TOKPROBE_STOP.prev.*`). Pull from `/workspace/v8proof/probe/states/`, per arm `<ARM>_{eval,train}.npz(+.json)` and `<ARM>_{eval,train}_lead.npz(+.json,+.head.npz)` for TK16_s0 TK16_s1 TK64_s0 TK64_s1 and TK16SH_s0 (p3), 10 files per arm; logs `runs/tokprobe/train/<ARM>/train_log.jsonl`, `train_meta.json`, `TRAIN_DONE.json` (p5 per arm; train exits 2 and the script records `p5_main_run=FAIL` but still extracts).
4. p5 can be re-read on any copy: `python tok_probe.py check-log --run-dir <train dir>`.

## Estimate of pod memory and time (ESTIMATED; basis stated)
* Host memory: ClipStore LRU `1.5e9 // (n_max * 40 * 1024 * 2)` clips (about 1.5 GB), three prefetched batches of 84 MB (feats [64,16,40,1024] fp16), CUDA context, join arrays: about 6 to 8 GB. Extract: train state of TK64 = 121,166 x 2,560 x 2 B = 620 MB, 1.2 GB at the concatenation. Far under the 20 GB floor and the 50 GB container cap. GPU: one batch is 168 MB as fp32 plus the LayerNorm and hidden activations, freed arm by arm: under 2 GB.
* Time: train is data-bound. Measured here: 0.091 s per batch of 64 for `ds.item` + `collate` (warm LRU, synthetic T=60, one thread) and 0.34 s for one TK64 step on 2 CPU threads (about 17 GFLOP; a GPU takes a few ms per arm). 19,164 steps x about 0.1 to 0.15 s = about 32 to 48 min including clip loads (precedent: the full trainer took 0.185 s/step on the RTX 4000 Ada, `pod/POD_TRAIN_RECIPE.md` line 72, a superset of this work). Extract: the eval states of 141 clips took 10 s on the CPU for `extract_states.py` (`s1states/G_s0_eval.npz.json`); 1,200 train clips about 90 s plus the head and the npz write, four arms: about 10 min. p3: 2,000 steps about 5 min. Total about 55 to 70 min (card: about one hour).

## Scratch (nothing deleted; deletes are refused here; all regenerable)
`F:/TanitAD-scratch/v8proof/tokprobe/tmp/` 297 MB: `pt/a`, `pt/b`, `pt/c` (pytest basetemps, safe to remove), `e2e/` (the dev-box run of the pod script: 5 arms of state files, safe to remove), `bench.py`, `bench/` (timing script and its synthetic world). Nothing written on C:.

## Lead addendum (2026-11-15)
pod_tokprobe_run.sh was extended by the lead with the sensitivity pass of note ADD27_N1 (SENS=1: the same four arms with no token normalisation, after the done marker of the main pass); its sha256 on the pod is e939d9703c3a61b0f9722e4b7ac16807166f1850b315ab06d8b5b05003c6c5ed. pod_tokprobe_wait.sh (lead) starts the job after the marker of the DISTRULE queue.
