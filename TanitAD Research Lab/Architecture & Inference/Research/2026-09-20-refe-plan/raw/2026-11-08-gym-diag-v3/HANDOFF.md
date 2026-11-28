# HANDOFF M-GYM-3 diag (2026-11-08). State: Q1 diagnosed and written; Q2 and Q3-constant-speed and ttc counterfactual are RUNNING detached. Do not wait alive.

Draft deliverable: `M_GYM_3_DIAG.md` (Q1 complete except counterfactual numbers; Q2 and the Q3 constant-speed block say PENDING). `PROPOSED.md` draft (items 2, 5, 6 need numbers). Scripts and Q1 raw outputs are in this folder; run outputs in `F:/tanitad_scratch/mgym3/diag/` (new scratch, mine; delete under CLAUDE.md rule when finished and copied; log size).

## Jobs (all CPU, CUDA off, max 4 workers, all started by this agent; do not kill anything else)
* Q2 chains: `run_q2_A.cmd` (pilot 0,12,14,15) DONE (`q2_A.done`); `run_q2_B.cmd` (stall 1,2,4,8) and `run_q2_C.cmd` (stall 14,16,18,19) running; each writes `q2_<sampler>_<idx>.jsonl` (+ `.log`), `q2_B.done` / `q2_C.done` when finished. A stall state takes about 5 min.
* Queue `job_queue.py` (pid 55724, status in `queue.status`) launches, at most 4 of my jobs at once: `cf_ttc.py` for stall 7, 10, 11, 15 (-> `cf_stall<idx>.json`) then `q3_const.py` for pilot 4, 5, 6 and stall 9 (-> `q3_<sampler><idx>.json`). It never writes QUEUE_DONE correctly; judge completion by the 8 json files and by no remaining python processes whose command line contains `pilot_v3_diag`.

## Exact next steps for the fresh agent (Sonnet, ~40k tokens)
1. Check files: `ls F:/tanitad_scratch/mgym3/diag/{q2_*.jsonl,cf_*.json,q3_*.json,*.done}`. If a job crashed read its `.log` tail, fix, rerun that single state only.
2. Q2: `python q2_agg.py F:/tanitad_scratch/mgym3/diag > q2_agg.out` (default python is fine, it only reads json). Read the per-cause precision (lead, conflict, red; also sole-cause, falsely-excused count, stop-policy scenes versus arm-set members). Put the table and n into `M_GYM_3_DIAG.md` Q2, replacing PENDING. Truth definitions are in the script (strict 1.15 m ego-width corridor; lead not receding vlon<2; conflict object ahead (lx>0) entering the strict corridor within 4 s; red connector on route and in the corridor). Also weight by the stored cause split (main: lead 555, conflict 1,131, red 115; stall: lead 657, conflict 846, red 600) in the stored precondition counts.
3. CF: `python cf_agg.py F:/tanitad_scratch/mgym3/diag`. Insert into Q1: for each state, number of arms hard under ttc off / brake3 / brake6 versus baseline (stall 7: 18/24; stall 10: 24/24; stall 11: 5/24; stall 15: 13/24), whether any real collision remains, whether stall 10 becomes clean. Say whether the stall-10 miss disappears (then it is purely the predicate) or turns into a collision (then genuine).
4. Q3: `python q3_agg.py F:/tanitad_scratch/mgym3/diag`. Fill the constant-speed block: return versus target (1.5 m/s crawl, 0.3, 0.5, 0.7, 0.9, 1.0 x limit), loss from q_free versus other terms, q_stall fired count at crawl (expect 0 by the code rule). State plainly whether any slow target matches the 0.9-1.0 target.
5. Update `PROPOSED.md` items 2, 5, 6 with the numbers. Keep the statement that a fresh seeded set of 20 states is required. Remove the word DRAFT and the PENDING markers; fix the typo `cf_stallstall10` in M_GYM_3_DIAG.md (file is `cf_stall10.json`).
6. Copy the small final outputs (q2_agg.out, cf_*.json, q3_*.json, q2_*.jsonl if under ~5 MB total, tr_*.jsonl) into this folder, then delete `F:/tanitad_scratch/mgym3/diag` by literal path after verifying the copies, logging path and size in the hand-back. Do not touch `F:/tanitad_scratch_mgym2`.
7. Hand back at most 300 words plus paths and a manifest.

## Facts already settled (see the draft; do not redo)
* Stall 7: 6 of 24 arms avoid the event; oracle (sweep1.3, 6.689) beats best clean arm (pdm0, 4.356) because a late hard event only zeroes remaining reward. Log-replay vehicle `5a528b` cuts across the lane while the ego is 3 m/s faster than the human.
* Stall 10: all 24 arms ttc at cycle 6; constant-velocity predicate fires on an oncoming car 10 m away (gap 10.5 m, lateral 1.8 m).
* Main 10: on-centreline footprint overshoots the drivable union by 0.254 m (map seam, s 18.5-23 m). Main 19: 24/24 arms return exactly 0.0 (1.1 cm over slack).
* Hard-event counts: 10 of 80 cells have a hard event in some arm, 4 of 80 in all arms.
* q_stall: free-road crawl cannot fire it (needs v < 0.5 for 20 ticks); the stop policy fires it 67/169 (main) and 37/220 (stall).
* Reward: persistent crawl pays 7-15 times less than PDM; first-action sweep is a weak probe; main 6 reversal comes from q_comf, q_osc and q_cons in the PDM continuation.
