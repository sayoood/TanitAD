# HANDOFF -- step-40,000 chain (armed 2026-10-19 03:54, EvalFlyWheel)

PKG = `C:/Users/Admin/qland/hold_r8batt/stage/FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-10-15-refcv8-milestone-battery`. Unlanded qland hold, NOT git-added (local override). The Master Mind lands it.

## State
* `sh code/chain_40k_r8.sh` running detached: MSYS pid **1152752** (Windows sh.exe 45004), pidfile `E:/refcv8_eval_kit/chain/chain_40k.pid`, out `chain/chain_40k.out`.
  Markers (opaque ZZC40...ZZ): `E:/refcv8_eval_kit/chain/chain_40k.log`. First two lines present: START, SNAPWAIT (snap40k pid 1027887 alive, train.log last ckpt 34,500 at 03:30; snapshot expected ~18:20).
* Stages: **a** wait for snap40k DONE + md5 re-verified 4 ways -> **b** `extract_ema_r8.py --expect-step 40000` + chain re-asserts step on the provenance artifact + EMA gate -> **c** EMA battery tag `step40000ema_unc`
  (batt root `C:/Users/Admin/refcv8_eval_archive/battery`, seeds 0,1, `--no-g0`, PROVISIONAL) -> **f** raw: metrics pull, `chain_g0_r8.sh 40000`, wrapper probe (non-blocking), `chain_battery_r8.sh 40000` (pass 1 only), tag `step40000_unc`.
  **CPU lane** (background, memory-gated, runs as each GPU stage lands): clamp_posthoc read -> compliant_reread read (BOTH needed, in this order) -> 3 frozen-gain forwards -> heading_drift, EMA tree then raw tree, then `render` x3 and `summarize_40k_r8.py` -> `raw/step40000/READ_40K.md`.
* G0 policy = the 30k one (EMA: no G0, PROVISIONAL; raw: G0 then battery even on FAIL-as-coded if M1 detected, PROVISIONAL).
* Terminal markers: `ZZC40ENDZZ` (all done; line says ema=ok|fail raw=ok|fail) or `ZZC40ABORTZZ` (stage a or b; the line names it). Failure markers: SNAPFAIL, SNAPDEAD, MD5BAD, MD5INCONCLUSIVE, EXTRACTFAIL, EMAGATEFAIL, EMAFAIL, RAWINCOMPLETE, COMPLFAIL, FWDCTRLFAIL, LANESKIP.
* The chain NEVER writes `step40000ema_unc/battery_summary.json`; only `run_battery_r8.py` does, on a complete pass (the retime train-fit chain keys on it). Terminal EMA failure = `ZZC40EMAFAILZZ`; the train-fit then waits its cap.

## How to check / resume
`tail E:/refcv8_eval_kit/chain/chain_40k.log`; alive = `kill -0 $(cat E:/refcv8_eval_kit/chain/chain_40k.pid)`. Per-stage logs: `chain/c40_extract.out`, `chain/battery_run_step40000ema_unc.log`, `chain/c40_g0.out`, `chain/c40_battery.out`, `raw/step40000/logs/lane_<tree>.log`.
Resume = re-run `nohup sh code/chain_40k_r8.sh < /dev/null > E:/refcv8_eval_kit/chain/chain_40k.out 2>&1 &` (stages with a done artifact are skipped; one instance only). Do not edit `chain_40k_r8.sh` while it runs.
Not in the chain (owed, need the PI's/MM's call): strict-cap (brake 0) sensitivity and `assemble` (both are wired to the 30k PRIMARY tree and would overwrite COMPLIANT_REREAD.json), the refcv7-matched-phase comparison, K-draw at 40k.
Tests: `code/test_chain_40k_r8.py` 19 passed (`raw/step40000/pytest_test_chain_40k_r8.log`); the 39,500 arm runs the REAL `extract_ema_r8.py`.

## GPU priority (no mechanism exists; measured from the scripts)
`gpu_lock.py` = O_EXCL file + smi rule, no queue, no priority file; every waiter polls ~60 s and the first to see it free wins. This chain takes the lock exactly like them (no faster poller: a fast-poll acquirer was
considered and NOT built -- it would race other agents' jobs). It cannot preempt. Queue ahead (ESTIMATED from kdraw_queue.log, seed roll 33 min): kdraw ends ~06:30 -> writes `/c/lgs/RELEASE_R7` -> `r7_after_kdraw.sh` launches
chain_r7_matched_phase (3 refcv7 batteries ~3.5 h each) + chain_r7on_50400 (1) + K2K4 at +8 min: ~15 GPU-h serial => lock held until ~21:00-22:00. **So YES: at 18:20 a refcv7 battery or K2K4 will very likely hold the card; the 40k EMA battery
starts at the next release (18:40-22:00), later if it loses a 60 s-poll race; it cannot starve past the end of that finite queue.** Only chain_retime_trainfit already yields to 40k (`b40_pending`, cap 4 h per seed).
To give 40k priority at the START of stages (not preempt): the three chains are NOT running yet, so the MM could add before their `gpu_acquire` the trainfit's predicate, extended to the whole 40k chain:
`b40() { ls /e/refcv8_eval_kit/ckpt/*40000*.pt >/dev/null 2>&1 || return 1; l=/e/refcv8_eval_kit/chain/chain_40k.log; grep -q 'ZZC40STARTZZ' $l 2>/dev/null || return 0; awk '/ZZC40STARTZZ/{e=0} /ZZC40(END|ABORT)ZZ/{e=1} END{exit e}' $l; }` then `while b40 && [ $n -lt $CAP ]; do sleep 120; done` (cap ~6 h) in
chain_r7_matched_phase.sh `battery_stage`, chain_r7on_50400.sh and queue_k2k4_r8.sh. A battery already running still finishes first. Kdraw is running and cannot be edited.
