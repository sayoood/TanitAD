# EGOWM: box-2 launch package (world-model speed channel, --wm-speed kin / now)

Card: `theory/egowm/PREREG_V8F_EGOWM.md` (frozen). Choices: `theory/egowm/IMPL_CHOICES_EGOWM.md`. Frozen argvs: `theory/egowm/argvs_egowm.txt`, copied token for token into `frozen/argvs_egowm.json` (sha256 pinned in `egowm_check.py`).
Tree commit: `1e1b9fc4ef33974ac82ee57f812a86490910ab76` (`theory/egowm/EGOWM_COMMIT.txt`), branch `agent/v8f-egowm-20261109`, mirror `C:/Users/Admin/tanitad-push`. Template: `theory/otband/box2/` (read, never edited), with the wm-only gate of `theory/ladder/box2/`.
**Nothing here has touched box 2.** It was exercised only over stub transports and a fake box folder (evidence class at the end).

## Order of the chain
1. **Wait for the predecessor.** The OTBAND chain's `C:/Users/Admin/otband_state/DONE.txt` (first bytes `ZZ OTBAND-DONE`) must exist and `BOX2_BUSY.txt` must be absent. If `C:/Users/Admin/otband_state/STOP.txt` (first bytes `ZZ OTBAND-STOP`) exists, this chain stops at once with `predecessor-stopped` (rc 12) and takes nothing on the box. STOP wins over DONE.
2. **ship** (own tree `D:\tanitad\egowm\repo`, never `D:\tanitad\repo`, never the OTBAND tree): the four patched files plus the import closure of the trainer (relative imports followed; `v8f_wm_windows.py` named explicitly), raw blobs from the commit, needles checked.
3. **pilot**: 200 steps of E-KIN_s0 (its argv plus `--steps 200`, `--out` under `D:\tanitad\egowm\pilot_out`). The same job gates the full-length E-KIN_s0 and E-NOW_s0 argvs (both speed modes read back from a real selfcheck). Projection over 5 runs; the 72 h rule needs an amendment file if exceeded.
4. **runs**: E-KIN_s0, E-NOW_s0, E-KIN_s1, E-NOW_s1, E-KIN_rep_s2. Each: disk, GPU idle (below 3,500 MiB, no python.exe), argv pin, selfcheck gate, per-run token, training under the token, post check, bank with sha256 against box 2's copy, cleanup of the run's own gate dir.

## What the gate reads back (`egowm_check.gate_problems`)
`declared_vs_built == PASS`; `argv_sha`; `steps_total == 19164`; data identity; `levers.wm_speed` equals the mode and `levers.wm_speed_readback.built` shows that mode on the planted ramp, 3 channels of width 32, cumulative strides 1/3/15, zero weights at build, no bias, and the built windows return the measured speed; `levers.wm_ladder_readback` (rollout k 10/10/6, context 3, strides 3 5, pooled wm-only windows, the latent cache of the same seed as the start state, none missing); `levers.wm_only` (trainable parameters equal the WM's, count = ladder base 2,143,298 + 96); `info.wm_only` (start state sha256, applied, 0 mismatches, WM re-initialised under the run's seed); G-LIVE (every term finite, controls exactly 0.0, every WM group has gradient); planner identity probe; no recover/dagger/ego-perturb.
Post check: summary steps, checkpoint step, every non-WM tensor bit-equal to M2b's, the three speed-channel tensors present and non-zero.
Seeds 0 and 1 use their own M2b init and latent cache; E-KIN_rep_s2 uses M2b_s0's.

## Launch (Master Mind, from a session background task)
```
export EGOWM_COMMIT=1e1b9fc4ef33974ac82ee57f812a86490910ab76
export EGOWM_SSH="$(command -v ssh)" EGOWM_SFTP="$(command -v sftp)"      # absolute paths; PATH is never searched; EGOWM_DRY must be unset
[ export EGOWM_AMENDMENT=<file> ]                                          # only if the pilot projects above 72 h and the PI amended
bash E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/egowm/box2/egowm_chain.sh
```
State: `C:/Users/Admin/egowm_state` (`EGOWM_STATE`). Banks: `C:/Users/Admin/t2v8f_pulls/egowm_<tag>/`. Sentinels live in their own files only: `STOP.txt` (`ZZ EGOWM-STOP`), `DONE.txt` (`ZZ EGOWM-DONE`), `ARM_REFUSED_<tag>.txt` (`ZZ EGOWM-ARM-REFUSED`). Resume after a stop: inspect, then `EGOWM_RESUME=1`.
Environment names: `EGOWM_COMMIT/SSH/SFTP/STATE/PULLS/AMENDMENT/RESUME/OTBAND_STATE/GPU_IDLE_MIB/WAIT_POLLS/WAIT_POLL_S`.

## Failure behaviour
A gate, pin or token refusal of one full run writes `ARM_REFUSED_<tag>.txt` and the chain goes on with the next independent run; the chain then ends with STOP, never DONE. Every other failure (ship, pilot, 72 h rule, training, post check, disk, pull, transport, timeout, box busy) stops the chain. Exit codes as in the OTBAND package; 12 also covers a predecessor that is not ended or ended in STOP.

## Isolation
Box 2 is a person's desktop. The chain acts only on `D:\tanitad\egowm\` and `D:\tanitad\runs\v8f_egowm\`, plus the shared read-only inputs named in the argvs. Deletions are the run's own gate and pilot dirs, by literal path, after the dev box proved the bank (`BANK_VERIFIED`), logged (path, bytes). Checkpoints are never deleted.

## Evidence class
* MEASURED (this package): `tests/UNIT_RESULT.log`, `tests/DRYRUN_RESULT.log`, `tests/SHIPCHECK_RESULT.log`.
* UNVERIFIED until the first real run: the real selfcheck key names and values (the checker reads the keys of commit 1e1b9fc4 as found in the trainer source; the stand-in trainer writes the same names), the WM parameter count 2,143,298 + 96, `S_steps_per_epoch`, how box 2 behaves under the GPU and python.exe rules, the latent cache paths on box 2, `train_log.jsonl` carrying `step` and `sec` columns for the pilot timing.
