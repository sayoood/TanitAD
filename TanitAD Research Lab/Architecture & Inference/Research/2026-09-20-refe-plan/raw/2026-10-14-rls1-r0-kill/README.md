# RL-S1 arm R0: KILLED by K3 (retroactively), and the live monitor that never ran

Banked 2026-10-14 ~20:35 Berlin by the Master Mind from the A40 pod (`/workspace/data/rls1/`). Pre-registration: `eval/PREREG_RL_S1.md`.

## What happened (MEASURED, files in this folder)

* R0 = RL-core arm (BCE + EPO, reward `PDMS - 0.15 D + g2`, `reward_cfg` in `rlval_vs_C0.json`), seed 0, 1,000 updates. Launch gate PASS (`token_verify.json`),
  trained 06:24Z-09:46Z, trainer rc 0.
* `replay.log`: **`ZZRLS1_REPLAY_KILL R0 K3 g3/g5 bound breached at update250 and update500`**. `kill_replay.json`: g3 held at both checks; **g5 failed at
  both** (TTC fell by more than the 0.5-point bound versus the control; NC held). K3 = two consecutive breaches ⇒ stop the arm (PREREG section 5).
* The chain then refused every further arm (`rls1_chain.log`, 10:07:21Z): "the Master Mind decides before any further arm".

## The RL-val read (selection tier, NOT an endpoint)

`rlval_vs_C0.json` (tool `rl_val_eval/1`): the true NAVSIM v1.1 PDMS of the scorer's pick on 3,000 RL-val sets over 75 logs; R0's final checkpoint
against C0's final checkpoint. Paired log-cluster bootstrap, 10,000 resamples, seed 20260927. It answers "another draw of LOGS" only: one training seed
per arm; training and inference variance are not measured.

| quantity (x100) | C0 | R0 | R0 - C0 |
|---|---|---|---|
| PDMS | 90.06 | 91.22 | **+1.16 [+0.59, +1.74]** (1,231 better / 450 worse / 1,319 tied) |
| EP | 83.06 | 86.54 | +3.47 |
| TTC | 96.67 | 95.67 | **-1.00** (g5 bound 0.5) |
| DDC | 98.17 | 97.28 | -0.88 |
| NC | 99.30 | 99.00 | -0.30 (within g5) |
| DAC | 97.67 | 97.93 | +0.27 |
| pick / fan-mean path | 0.925 | 0.986 | longer picks |
| shortest-quintile pick share | 0.405 | 0.306 | |
| unsafe picks / 3,000 | 171 | 191 | +20 |

At update 250, PDMS was already +0.60 and the TTC fall already past the bound (`kill_replay.json`).

## The live monitor never ran (instrument defect, leading explanation OOM)

* `monitor.log` is **0 bytes for C0 and R0** (`monitor_evidence.txt`). `rls1_checks.py`'s `say()` flushes every line, so even R0's step-0
  `ZZRLS1_K3_LIVE` line, written right after the first check, was never reached.
* Monitor exit: **C0 rc 0, R0 rc 137 (SIGKILL)** (`rls1_chain.log`). On a C arm the monitor never loads the RL-val lookup. On an R arm it does,
  beside the trainer.
* The pod's memory cgroup records **`oom_kill 2`** (`memory.oom_control`), and `max_usage_in_bytes` equals the 71 GB limit. That counter includes
  page cache (TRAPS: cgroup v1 counts reclaimable cache), so it is supporting evidence only.
* ⇒ **Leading explanation (MEASURED in parts, NOT proven): the monitor is OOM-killed when it loads the RL-val lookup beside the trainer.** The time of
  death is not recorded. The fix must measure the monitor's peak RSS and make the chain FAIL CLOSED: an R arm whose monitor is dead is running without
  its kill rules.
* Cost: R0 trained to update 1,000 (3.4 A40-h) instead of stopping at the update-500 check (~1.7 A40-h).

## Where the rest lives

* The full arm: HF private `Sayood/tanitad-refe-rls1-r0` (egress `ZZEGRESS_OK arm_R0 27 files 2783596210 B`); the pod's
  `/workspace/data/rls1/arms/R0/` until the verified-copy cleanup.
* `sft.jsonl`, the score dumps and the checkpoints are not copied here (sizes); they are in the HF repo above.
