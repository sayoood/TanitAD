# HANDOFF: FINEGRID-R (card ADD43), dev box, 2026-11-19

State: built. Tests of the reader and the pod script ran (numpy only); everything that needs torch is NOT RUN (dev-box memory ruling). No pod, no git, no number of the card. Tier T0; STRATEGIC: NOT COMPUTED (n = 0). Decisions: `FINEGRID_R_DEFS.md`. Folder `F:/TanitAD-scratch/v8proof/finegrid_r/`, footprint 3 MB.

## Files (sha256 prefix)
tok_probe_gr.py 89efd1567b2d | pod_finegrid_r_run.sh 0d05c18f08d5 | finegrid_r_read.py 7c7a4dfae8f2 | run_pod_read_r.sh 85918c965426 | run_checks_pod.py 11e526e745af | controls_devbox_r.json 4c4ce4ba8a7a (placeholder, fails closed) | make_tok_probe_gr.py af39a427a0f1 | tests: test_finegrid_r_read.py (21 passed), run_mutations_gr.py, MUTATIONS_GR.json (13 of 13 red, suite green before and after).

## Pod commands, in order
1. Copy tok_probe_gr.py, pod_finegrid_r_run.sh, run_checks_pod.py to `/workspace/v8proof/finegrid/` (compare sha256). Nothing of the pilot is overwritten.
2. `CUDA_VISIBLE_DEVICES=-1 /workspace/venv/bin/python run_checks_pod.py --tmp /workspace/v8proof/finegrid_r/checks_tmp --controls-out /workspace/v8proof/finegrid_r/controls_devbox_r.json` (needs a new folder, about 70 MB; last line `ZZ CHECKS-POD DONE n_pass 0`). NOT RUN yet.
3. `DRY=1 bash pod_finegrid_r_run.sh` (32 commands), then `nohup bash pod_finegrid_r_run.sh > /workspace/v8proof/runs/finegrid_r.log 2>&1 &`. Done = `runs/finegrid_r/FINEGRID_R_DONE.txt`, stop = `FINEGRID_R_STOP.txt`.
4. Reader: put finegrid_r_read.py and controls_devbox_r.json into `rk/F:/TanitAD-scratch/v8proof/finegrid_r/`, then `run_pod_read_r.sh` (header says how). Out: `rk/out_read_r/RESULT_FINEGRID_R.json`, `VERDICT_FINEGRID_R.md`.

## Expected (ESTIMATED)
Time 3 to 5 pod-h (card says 2 to 3). Volume about 1.5 GB (eval states only; the pilot's 7.65 GB held train-probe states). 16 arms in one process fit the GPU: a few GB (batch and layer-norm copies dominate; arms train one after another).

## REVIEW
1. The unchanged tool gives the two grids different sampler chunks (48 at 40 tokens, 41 at 160 tokens): control r1 would fail. Lever `--sampler-chunk-like pool8x20` on process A and TK16SH. The new 4 by 10 arms then use another window order than banked TK16 and TK64.
2. p5 parts were already in the log; no log change. Train log gets a sibling `r1_batches.json`.
3. Two-level bootstrap: one shared episode draw per replicate (crossed design); independent-draw form beside.
4. 30 slot waits (45 s gaps) add about 25 min.
5. Pilot note N1 guard (1e-4 from a bar) is printed, not applied.

## Not verified
Everything with torch: tool syntax only parsed; g1, r1 log, chunk lever, p5 in the tool, g3, g4 (all in run_checks_pod.py); `run()` of the reader on real files (its parts are tested; the g7 TK path is the pilot's code); pod script only against a stub tool.
