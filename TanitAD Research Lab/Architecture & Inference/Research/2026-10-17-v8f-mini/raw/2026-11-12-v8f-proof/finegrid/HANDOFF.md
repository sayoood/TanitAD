# HANDOFF: FINEGRID tool (card PREREG_V8F_PROOF_ADD41), dev box, 2026-11-18

State: built and tested on the dev box (CPU, synthetic data). Nothing touched the pod, box 2, Thor or git; nothing staged. No number of the card is computed (the reader ran only on the banked TK arms, control g7). Tier T0; STRATEGIC: NOT COMPUTED (n = 0). Folder `F:/TanitAD-scratch/v8proof/finegrid/`. Decisions: `FINEGRID_DEFS.md` (REVIEW items: T0 tree, T6 row-major token order, R4 "each mean").

## Pod files (push to `/workspace/v8proof/finegrid/`, compare sha256 on both sides)
| file | sha256 |
|---|---|
| `tok_probe_g.py` | 7fe801655b91a1ae26f53c7d6c7c01454844cdd55ec9785bc0b3175a5fdf5820 |
| `pod_finegrid_run.sh` | 86fca71df32bd6a251d43f476afa28ab258d615858810a26c84f953e2f7638e4 |
Nothing else is needed on the pod (tree `/workspace/v8proof/tree_leadsup`, cache `/workspace/v8proof/data/cache_pod`, python `/workspace/venv/bin/python` exist). `finegrid_read.py` stays on the dev box.

## Pod steps (the lead runs them)
1. `cd /workspace/v8proof/finegrid && DRY=1 bash pod_finegrid_run.sh` : 8 commands (g2, train, 4 extracts, p3 train, p3 extract).
2. `nohup bash pod_finegrid_run.sh > /workspace/v8proof/runs/finegrid.log 2>&1 &` . The script itself waits for a slot: fewer than 3 GPU compute processes AND container free memory above 28 GB on two probes 45 s apart, polling every 60 s for at most 10 h, then `FINEGRID_STOP.txt` (never a fourth GPU job). Then g2 on the CPU, train (four arms, one process, `--grid pool8x20 --input-norm ln --expect-steps 19164`), extract (eval 0:141, train 600:1800), p3 (F16SH_s0, 2,000 steps) and its extract. Log: `/workspace/v8proof/runs/finegrid/finegrid.log` (UTC time-stamped).
3. Done = `/workspace/v8proof/runs/finegrid/FINEGRID_DONE.txt` (carries the sha256 of the tool and of the five checkpoints); stop = `FINEGRID_STOP.txt` (reason inside). A finished marker refuses a second run (exit 3).
4. Pull: `/workspace/v8proof/finegrid/states/` (per arm F16_s0 F16_s1 F64_s0 F64_s1 F16SH_s0: `<ARM>_{eval,train}.npz(+.json)` and `<ARM>_{eval,train}_lead.npz(+.json,+.head.npz)`; ESTIMATED 7.65 GB in all, F64 train states 2.5 GB each; the same size lands on the dev box, F: had 26 GB free), `runs/finegrid/G2_GRIDS.json`, `runs/finegrid/train/<ARM>/train_log.jsonl` (four arms), `runs/finegrid/p3/F16SH_s0/train_log.jsonl`, both `train_meta.json`.
5. Read (dev box, about 10 min; needs the pulled state files in one folder, head outputs may sit in the same folder):
   `python finegrid_read.py --in-f <folder> --g2 <G2_GRIDS.json> --controls-json F:/TanitAD-scratch/v8proof/finegrid/controls_devbox.json --train-log-dir <folder with train/ and p3/ logs as <ARM>/train_log.jsonl, comma list> --out-dir <dir>`
   (`--in-tk` and `--tk-states-dir` default to the banked head outputs and to `moved_from_c/tokstates`.) Outputs `RESULT_FINEGRID.json`, `VERDICT_FINEGRID.md`. Exit 0 = every control passed (names written), 2 = a control failed (no name, outcome differences withheld), 3 = an input record failed.

## Verified on the dev box (counts in the hand-back; files `MUTATIONS_G.json`, `out_tk_repro/TK_REPRO_G7.json`, `controls_devbox.json`)
* g1 default path: bit-identical probe weights, step-by-step losses, and extract outputs against the banked `tokprobe/tok_probe.py` (same seeds and steps).
* g3 planted 160 tokens read above 0.9 on held-out clips; g4 gradients non-zero on the per-token network and head, exactly 0.0 on the control parameter at 160 tokens; g5 shape refusals, arm/grid refusals, checkpoint records 160 tokens seen; g6 forward path takes the tokens only, noise labels change nothing, p5 on a short real-path run.
* Mutations (temp copies of the tool): loss line removed, 4 by 10 grid under the 8 by 20 flag, token-count refusal removed, checkpoint records 40 tokens, checkpoint declares 40 tokens, extract reads the 4 by 10 key, arm/grid refusal removed, g2 bar always passes, control parameter in the graph: each turns the suite red, the unmutated control is green.
* Pod script: DRY, finished-marker refusal, slot wait stops after the limit on 3 GPU jobs and on 20 GB free (fake `nvidia-smi` and cgroup files), full run on the synthetic world on CPU (states, markers, log).
* Reader: planted better F16 reads READS-GAP (and READS-CLOSING); equal arms FLAT; worse arm WORSE; a gain under the floor or missing the 10 % MAE clause FLAT; a failed control writes no name and no difference. g7: the real reader on the banked TK arms reproduces M1, M2, M4 (eval and train probe clips) with max absolute difference 0.0.

## Not verified / doubtful
* Nothing ran on real data or a GPU. The token order of the 8 by 20 grid (row-major) is assumed; g2 prints the column-major reading beside.
* The dev-box tree `v8proof_c/leadsup/tree_patch` was missing; tests use a rebuilt copy (pins equal). The pod's `tree_leadsup` is the one the tool runs against.
* Pod volume of the five state sets is about 7.65 GB against the card's "under 8 GB".

## Scratch (nothing deleted; deletes are refused here; all regenerable)
`F:/TanitAD-scratch/v8proof/finegrid/tmp/` is 7.2 GB, over the 300 MB cap: `pt/` is 7.1 GB (pytest basetemps of about 15 test and mutation runs; the 160-token synthetic worlds and checkpoints are large). A delete was refused here, so it is left. SAFE FOR THE LEAD TO REMOVE: `tmp/pt/` entirely (all regenerable). Keep `tmp/tree_leadsup` (39 MB tree copy the tests need). Also `mut/` (400 KB mutated copies), `t/`, logs. F: had 18 GB free after the runs (26 GB before): below the 20 GB floor of a pinned run, so remove `tmp/pt/` first. Nothing written on C:.
