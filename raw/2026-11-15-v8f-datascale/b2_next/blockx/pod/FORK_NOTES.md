# blockx: what differs from block3 (written by fork_x.py and make_study_x.py; the block-3 files were only read)

| file | forked from | change |
|---|---|---|
| `gen_argv_x.py` | `argv/gen_argv2.py`, `block3/gen_argv_b3.py` | writes `argv/X_s0.A.json` and `argv/X_s0.B.tmpl.json` from U_s0.A / U_s0.B.tmpl: the seven data flags point at the 10k-only files and the X cache view, `--out runs/X_s0/{A,B}`; nothing else changes |
| `xview.py` | `union_view.py` | builds `/root/data/cache_x_dscale/train` (10,000 symlinks into the shard folders, manifest, meta) |
| `study/dscale10k_check.py` | `block3/study/dscale10k_check.py` | two X-only edits: G-LIVE kinds for the X arm = tenk only; C-JOIN-EQ for the X arm = the X join equals the union join on the X clips (diff printed by `make_study_x.py`) |
| `init_x.py` | `init_b3.py` | arm X_s0, record `gate/init_x.json`, imports the blockx checker |
| `assemble_evidence_x.py` | `assemble_evidence_b3.py` | init item `init_x`; reach = the tenk record and control of `gate/probe2/UE_s0.A.json` (INHERITED); no `ue_steps` |
| `chk_armx.sh` | `chk_arm3.sh` | blockx checker; X_s0 takes the union-release extras; join slots as above |
| `verify_token_x.py` | `verify_token3.py` | blockx checker; X_s0 in the union extras |
| `blockx_prep.sh` | `block3_prep.sh` | arm X_s0; adds the cache view and argv steps; no GPU reach probe |
| `blockx.sh` | `block3.sh` | stage A 19,164, `do_arm X_s0 yes` (bank, selfcheck, stage-B token, stage B 3,000), X_s0 in the sweep, no S_U check, sentinels `ZZ DSCALE10K BLOCKX ... ZZ` |
| `queue_blockx.sh` | `queue_block3.sh` | waits for `ZZ STRAT1 NAVRB2 (DONE\|FAIL) ZZ` in `/root/strat1/logs/navrb2_queue.log`, then `flock /root/gpu_queue.lock` |
| `fake_leaf_x.sh`, `dryrun_blockx.sh` | `fake_leaf3.sh`, `dryrun_block3.sh` | stage A 19,164 / stage B 3,000 fake steps; 32 scenarios (order, rerun skip, failures at prep / stage A / stage B / mint, STOP, disk guard, queue sentinel incl. a C0REP line that must not release the wait) |

Header comments inherited from block 3 still say "block 3" in places; the code paths are the X ones (dry run scenario 9 checks the code lines).
