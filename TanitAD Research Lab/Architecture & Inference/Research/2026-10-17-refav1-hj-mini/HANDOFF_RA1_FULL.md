# HANDOFF — RA-1 at the full corpus (PREREG_RA1_FULL), 2026-10-18 ~11:50

**State: pre-registered and landed (bc0d19a, 2061c3f). The data build is about to start on box 2; nothing has been read.**

## What is running, and where (dev box, background bash; each writes a log and EXITS on any failure)

| chain | log | ends with |
|---|---|---|
| `box2_ship/ship_trainall.sh`: mp4 + timestamps for 3,972 + 2 clips, egomotion zip and intrinsics → `D:\tanitad\data\trainall_stage` | `box2_ship/logs/ship_trainall.log` | `ZZSHIP-DONE` / `ZZSHIP-FAIL` |
| `box2_ship/chain_ra1_full.sh` (banked: `code/box2_ops/`): waits for the transfer and an idle box 2, then runs `ra1_full_box2.py` as ONE SAFE-MODE job (~22 h) | `box2_ship/logs/chain_ra1_full.out`; box 2 `D:\tanitad\logs\ra1full_build.log` | `ZZRA1FULL-BUILD-DONE` / `ZZRA1FULL-STOP` |
| `box2_ship/chain_ra1_full_runs.sh`: waits for BUILD-DONE, refreshes box 2's code from the REMOTE tip (6 key files sha-checked), then runs phase 1 → select → [extend] → phase 2 → read → [phase 3 if K FAIL] | `box2_ship/logs/chain_ra1_full_runs.out`; results in `box2_ship/logs/ra1_full/` | `ZZRA1FULLRUNS-DONE … outcome: …` / `ZZRA1FULLRUNS-STOP` |

* **Build exit codes:**
  * 10: C0 control mismatch, a STOP rule (see the prereg's reading 0);
  * 11: the disk guard;
  * 12: an mp4 sha mismatch (re-ship that chunk);
  * 13: the build;
  * 14: the encode;
  * 15: verify.
* **Resuming:** the build is resumable through `D:\tanitad\data\ra1_full_state.json`. To resume, rerun the job (`D:\tanitad\jobs\ra1full_build.cmd` via `launch.ps1`). The runs queue skips any run whose `eval.json` exists.
* **Busy marker:** `box2_ship/BOX2_BUSY.txt` exists while a chain holds box 2. Another session (A1a, `local_ede73180…`) shares box 2, and it was told to check the marker.
* **Dev-box copies:** the staging inputs (~60 GB) are still on the dev box in `C:\Users\Admin\tanitad-data\refav1-train4572-256_stage`. The dev box's C: has ~29 GB free. Delete them only after `ZZRA1FULL-BUILD-DONE`; until then they are the re-ship source.

## When the runs chain ends: the remaining work

1. Read `box2_ship/logs/ra1_full/full1.json` and `full_final.json`, plus `full_final3.json` if phase 3 ran. The outcome is one of INSTRUMENT_BROKEN, K_PASS or K_FAIL, and the readings are in the prereg.
2. Write `RESULT_RA1_FULL.md` with:
   * the selection table by budget for R0 and A1 (holdout, and eval-141 as non-deciding);
   * the memorisation row;
   * the gate;
   * K for the K arm;
   * the four families from each run's `eval/eval.json` `families` block: STRATEGIC PAUSED; distance-keeping NOT COMPUTABLE (the pooled cache has no lead agent);
   * the matched-budget comparison with 540 clips (`raw/ra1_es/`).
3. Bank `box2_ship/logs/ra1_full/runs/**` (no `.pt`) under `raw/ra1_full/runs/`, plus `ctl_result.json`, `ra1_full_state.json` and the telemetry. Update the `H-RA1-FULL-1` row in `GOALS_AND_CLAIMS.md` in the same landing. Land with qland:
   * a hold with `stage/`, `mods/` (append-only) and `mods-rw/`;
   * `bash /c/Users/Admin/qland/land_hold.sh hold_X`;
   * if `ZZABORT-TIP-NOT-REMOTE`, fast-forward `refs/heads/agent/arch-inf-20260803` to the remote by CAS, after `merge-base --is-ancestor`;
   * verify with `git ls-remote`.
4. Rule Zero: if the outcome is INSTRUMENT_BROKEN, the next lever is Addendum D's A1a at the full corpus. A phase-3 K FAIL means the LoRA r 32 compute decision goes to the PI.

## Traps met on this stream

* The trainer trains on EVERY clip in its cache. That is why there is a separate `cache_full` root, and `test_queue_full_stages_read_their_own_cache_root_and_prior_and_nothing_else_does` pins it.
* Never copy a `.cmd` to box 2 while a queue invocation runs there: cmd reads batch files by byte offset.
* The heredoc in the Bash tool collapses `\\`. Write Windows paths with the Write tool and `chr(92)`.
