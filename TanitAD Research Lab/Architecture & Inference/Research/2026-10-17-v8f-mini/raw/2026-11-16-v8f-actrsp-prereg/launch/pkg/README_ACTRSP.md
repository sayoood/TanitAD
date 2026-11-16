# ACTRSP: box-2 launch package (world-model action response: `--wm-rsp-w 1.0`)

Card: `PREREG_V8F_ACTRSP.md` (frozen, sha256 `ce2d02df...`). Code: branch `agent/v8f-actrsp-20261116`, commit `a13f6f16983010d2a96e14c16aef8273626fcd73` (base ACTSENS `81cee6c1`). Fork of the ACTSENS package (names ACTSENS_* became ACTRSP_*; the checker is the build agent's `actrsp_check.py`).
**Nothing here has touched box 2.** It ran only over stub transports and a fake box folder (`tests/DRYRUN_RESULT.log`).

## Gate status on 2026-11-16
B-CAL (card section 6) TRIGGERED on both chassis and both channels (MEASURED, `../bcal/*.json`): the chain must not start until the MM freezes the slope-rule amendment and the amended trainer is landed.

## Order of the chain (card section 3/5)
1. ship to own tree `D:\tanitad\actrsp\repo` (trainer, data, windows, v8f_mini, hjepa + import closure, raw blobs from the commit, needles checked).
2. pilot `PILOT_RSP_s0` (200 steps of RSP_s0's argv; the same job gates the full-length RSP_s0 and RSP_s2 argvs); 72 h rule; stop rules from the card: s/step > 1.0 or max_memory_allocated > 6.5 GB.
3. runs: RSP_s0, RSP_s2, NULL_s4, RSP_s1, RSP_s3, NULL_s5, each with disk, GPU-idle, argv pin, `--selfcheck-only` gate (C3 identity, BUILT-weight read-back), per-run token, training under the token, post check (C5 Adam state, C2 rsp-from-log), bank, cleanup of its own gate dir.

## Launch (MM; READY, NOT RUN)
```
export ACTRSP_COMMIT=<40-hex sha of the landed ACTRSP commit>
bash ../launch_actrsp.sh      # detach via WMI as for ACTSENS; copy this folder to C:/Users/Admin/actrsp_pkg first and blob-check it
```
State: `C:/Users/Admin/actrsp_state`. Banks: `C:/Users/Admin/t2v8f_pulls/actrsp_<tag>/`. Sentinels in their own files: `STOP.txt` (`ZZ ACTRSP-STOP`), `DONE.txt` (`ZZ ACTRSP-DONE`), `ARM_REFUSED_<tag>.txt`.
Changes from ACTSENS: no `r35_live_terms.json` (C3 is the trainer's own step-0 identity, `live.c3`); `hjepa.py` joins the export list; `--wm-rsp-w 1.0` must be named exactly once in an RSP argv and absent from a NULL argv (`argv_rsp_problems`); `--amp` is off in every argv.
