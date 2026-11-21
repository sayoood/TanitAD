# X_s0 (the 10,000 data-scale clips alone): staged, NOT launched

Card: `datascale/card/CARD_DATASCALE_TRAIN.md`, arm table row `X_s0` (10k set only, stage A 19,164, stage B 3,000, seed 0 / dagger-seed 0), section 6 DATA-HURTS row.
Status 2026-11-18: chain files written on the LoRA pod and dry-run (32 of 32 scenarios PASS, `DS_FAKE=1` on a fake root, CPU only, fake roots deleted by the script). Nothing was trained and the GPU was not touched. **The MM asks the PI before the launch.**

## Why X_s0 (the question it answers)
DATA-HURTS read: TACTICAL costs at both seeds. HYPOTHESIS: dilution (VLM goal bits masked on the 10k clips). X_s0 trains on the 10k clips alone, so its TACTICAL cells against B_s0 show what the new clips teach by themselves; U between B and X would read as a mixture. The goal-class F1 read (`datascale/diag/RESULT_DIAG.md`) came out NOT-DILUTION as pre-committed (neither group's macro-F1 difference is separated at either seed), so X_s0 is the arm that can still separate "the 10k labels teach the wrong tactical decisions" from "the mixture dilutes".

## Launch command (the MM, after the PI says yes). Its own ssh command, never inside an && list.
```
source C:/Users/Admin/lorapod/podenv.sh
pssh "nohup setsid bash /root/dscale10k/blockx/queue_blockx.sh < /dev/null > /root/dscale10k/blockx.log 2>&1 &"
```
* `queue_blockx.sh` waits for the line `ZZ STRAT1 NAVRB2 DONE ZZ` or `ZZ STRAT1 NAVRB2 FAIL ZZ` at the start of a line in `/root/strat1/logs/navrb2_queue.log` (the sentinel written by `/root/strat1/navrb2/navrb2_chain.sh`), then `exec flock /root/gpu_queue.lock bash /root/dscale10k/blockx/blockx.sh`.
* End line in `/root/dscale10k/blockx.log`: `ZZ DSCALE10K BLOCKX DONE ZZ` or `ZZ DSCALE10K BLOCKX FAIL ZZ <why>`. A file `/root/dscale10k/chains/STOP` ends it between steps (the same STOP file the other DSCALE chains read).
* Stop before launch by not running the command. Stop after: `touch /root/dscale10k/chains/STOP`.

## What the chain does (forked from block3.sh; `/root/dscale10k/blockx/FORK_NOTES.md` lists every change)
1. `blockx_prep.sh` (CPU, inside the chain, under the GPU lock): `xview.py` (cache view), `gen_argv_x.py` (argvs, already written), dry build of X_s0.A (writes `merged/join_X.npz`), `init_x.py` (G-HYG / C-INIT against B_s0, refuses if the pod tree is not the one G-SUITE ran on), `assemble_evidence_x.py`, mint of the stage-A token, `verify_token_x.py` (token plus 4 tampered copies must be refused).
2. Stage A: 19,164 steps, post-training zero-grad gate (registered parameters equal B_s0's, every one has Adam state), bank to HF through the banker.
3. DAgger bank from X_s0's own stage-A checkpoint over its own train windows (`bank4.sh`, 4 shards, coverage at least 0.5), stage-B argv with the arm's own `--init-ckpt` / bank / bank sha256, trainer `--selfcheck-only`, stage-B token, stage B 3,000 steps (`--w-acc 49.6969 --detach-upper --dagger-p 0.1 --dagger-seed 0`, the OBD-CTL recipe), post gate, banker. The mirror of block 1/2's U_s0 and U_s1 stage B, flag for flag (the X stage-B argv is U_s0.B.tmpl with only the seven data flags changed).

## Data (what X_s0 reads)
| input | path on the pod | note |
|---|---|---|
| cache | `/root/data/cache_x_dscale` (a view built by `xview.py`) | **`/root/data/cache_10k` itself has no `train/` folder**: its 10,000 clips sit in 42 shard folders `s*/train/clips`. MEASURED (read-only count): 10,000 shard files, 0 duplicates, 10,000 manifest rows, 0 missing, 0 extra. The view is 10,000 symlinks plus `manifest.jsonl` (= `in/manifest_10k.jsonl`) and `meta.json` (the corpus cache's, as the union view does). |
| v9 release | `in/labels/v9_labels_datascale10k.npz`, md5 `395221440e3a3c20f17dd381297af4f0` | the checker's accepted role `tenk` (`dscale10k_merge.TENK_V9_MD5`) |
| lead / speed / tracks | `in/labels/lead_datascale10k.npz`, `speed_sidecar_datascale10k.jsonl`, `native_tracks_datascale10k.pt` | the 10k files of the merge inputs |
| join | `merged/join_X.npz` (written by `--fit-join`) | |

## Cost (pod: RTX 4000 Ada, one arm at a time)
MEASURED rates from the block-1/2 logs (`/root/dscale10k/block1.log`, `block2.log`, `runs/*/*/summary.json`):
* Stage A: 19,164 steps in 3,607.5 s (U_s0) and 3,609.2 s (U_s1), 0.188 s per step. The step time does not depend on the clip count (B_s0 stage A: 3,648 s from the chain log). X_s0 stage A: **about 1.00 h**.
* Stage B: 570.4 s (U_s0), 570.5 s (U_s1), 603 s (B_s0 from the chain log): **about 0.16 h**.
* Bank plus stage-B selfcheck plus mint (chain-log wall between "stage A done" and "run ... stage B"): B_s1 3,427 s for 306,566 windows, U_s1 6,738 s for 985,048 windows.
ESTIMATED total for X_s0: a two-point fit of those two walls (fixed part 1,931 s, 4.88 ms per window) at about 678,482 windows (U minus B windows, which is the 10k clips' own windows) gives a bank step of **about 1.46 h**. Total **about 2.6 pod-h** (1.00 + 1.46 + 0.16), plus the CPU prep (dry build, init record; not timed here) and the banker's HF upload. The MM's earlier figure was about 1.5 pod-h; the bank step is what raises it. If the PI wants a cheaper first look: stage A alone is about 1.0 h, but the deciding read is the stage-B model.
Disk: the chain's gate needs 60 GB free on `/root` (probe 2026-11-17 23:30Z: 105 GB free); the sweep deletes banked bank shards and selfcheck scratch under chainlib rules.

## Read afterwards
The read for X_s0 is the block-2 reader's pairs `X_s0 - B_s0` (and `X_s0 - U_s0`) at stage B, on the same 141 episodes, plus the diagnostics of `datascale/diag/diag_read.py` (goal F1, country / platform) with X_s0 added. A one-seed arm: it is a diagnostic of the DATA-HURTS cause, never a second lever claim, and its floor is the B_rep pair.

## UNVERIFIED (first real run will show; each fails loud and stops before training)
1. `init_x.py` calls `gate_items.build_init(argv, "X_s0")`; whether that function accepts an arm name outside the block-1..3 list was not exercised (it builds the model and data on CPU).
2. The dry build of X_s0.A on the real X view (CPU, RAM about as for U_s0's dry build) was not run (brief: dry runs only on a fake root).
3. **C-JOIN-EQ for X** is a rule I added in the blockx fork of the checker: the X join must equal the union join (U_s0's) on every 10k clip, bit for bit. If `--fit-join` output is not per-clip identical across the two sets the mint refuses, which is the right outcome but needs an MM decision on the rule (diff in `FORK_NOTES.md`).
4. The G-LIVE reach evidence for X_s0 is INHERITED from `gate/probe2/UE_s0.A.json` (its `tenk` record and the corpus control: same code, same release). No GPU probe runs for X (the probe needs a corpus batch that an X-only argv does not have). The MM may instead require a fresh probe.
5. G-EVAL / G-CKPT / G-SUITE reuse the block-1 items, valid only while the pod tree equals the block-1 G-SUITE tree (`init_x.py` checks and refuses otherwise).
6. Banker and `chainlib.py sweep` take the arm name as given; X_s0 tags were exercised only by the fake-root dry run.
