# HANDOFF -- brake-locus CPU re-roll (written 2026-10-19 21:15 +0200; the roll is NOT done, do not wait for it)

## State
* Built (staged in this package, nothing committed): `code/brake_locus_reroll_r8.py` (select / roll / status / next / smoke-report), `code/brake_locus_reroll_after_40k.sh` (waiter), `code/brake_locus_reroll_analyze_r8.py` (analysis, selftest passes, exercised end to end on a SYNTHETIC bank that was deleted), seals `PREREG_BRAKE_LOCUS_AMENDMENT3.{md,sha256}` (hedge arms H1/H2) and `..._AMENDMENT4.{md,sha256}` (F2/S0 readout + roll order).  `brake_locus_r8.py` is unedited (its seal check knows A1-A2 only; the new tools check the original + A1-A4).
* Smoke PASS (`raw/brake_reroll/SMOKE_REPORT.json`): fan [149,8,2], all design fields present; selected candidate == emitted plan (max 0.0156 m = fp16 storage); sel_idx == argmax of the blended score over reach_keep; N and F2 equal force_lon's banked arms to 4e-7 m; S0id == N bit-for-bit; S0x moves the pick; H1 puts the forced joint (id 1 = LANE_KEEP x CRUISE) x4 into the allocation.  Cost MEASURED: base native forward 14.6 s, every extra arm 7.5 s per window (CPU shared with the 40k battery).
* Plan (`select`, seed 20261020, `raw/brake_reroll/plan_sha12.json`, raw ids in `E:/refcv8_eval_kit/brake_locus_scratch/plan_raw.json`): 1,383 windows (FB 493, CT 250, DC 100, ALONG 690; FB250/CT100/DC100 carry phase B).  Stages in priority order A (3.7 h) -> BH (2.8) -> BF (0.7) -> BS (1.7) -> AL (3.3) = 12.3 h ESTIMATED; the 2026-10-20 15:00 stop will probably cut AL.  `E` needs only A; the hedge arms only BH.
* Waiter PID **1667477** (nohup, detached, log `E:/refcv8_eval_kit/chain/brake_reroll.log`, markers ZZBLR...ZZ).  Starts on the step-40k ship end line / `raw/ship_r8/SHIP_R8_step40000ema_unc.json`; else after 2026-10-20 06:00 if no 40k CPU lane; never starts a chunk after 15:00 or once chain_final.log shows a marker beyond its wait state; >= 5 GB free commit per chunk.  Branch tests with stub env vars all passed.
* Check: `kill -0 1667477`; `tail E:/refcv8_eval_kit/chain/brake_reroll.log`; `python -X utf8 code/brake_locus_reroll_r8.py status` (env below).  Resume after a stop: re-run the waiter (set `BLR_DEADLINE_STOP='<later +0200>'` if past 15:00 and the final chain is done).

## Analysis command (run when status shows A and BH complete; PARTIAL is stamped)
```
cd <PKG>/code && CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=4 PYTHONIOENCODING=utf-8 \
 PYTHONPATH='C:/lgs/Lfb47a58376/stack;C:/lgs/Lfb47a58376/taniteval;<PKG>/code' \
 C:/Users/Admin/venvs/tanitad/Scripts/python.exe -X utf8 brake_locus_reroll_analyze_r8.py --tag full
```
Writes `raw/brake_reroll/BRAKE_REROLL.{json,md}`.  READ `controls_R6` FIRST: selected==emitted, S0id==N, S0x moved, N vs force_lon, `R6_rerolled_false_decel` vs banked 0.2656 beside the N2 floor.  The CPU draws are NOT the banked GPU seeds (draw0 = offset 0, draw1 = offset +1).

## Reading rules (all sealed before any number; do not move)
* **E / label**: `fan_cells`, X = 2.91 m FDE@6 s, reach survivors; E < 0.3 CANDIDATE, >= 0.5 SELECTION, else neither; label per original section 5 with HEAD test = banked FAIL.  Sensitivities all149 / base117 / ADE 1.21 m in the JSON.  E_ctrl on GT-cruise windows the re-rolled N does not brake.
* **ALONG-RAND** (A1 A3): dR = r_os - r_RAND per draw, paired bootstrap; SELECTOR-ADDS-VALUE iff lo > 0 at both draws; SELECTION-SUPPORTED iff not; else SEED-SPLIT; same for g*; MED beside; best-of-k bounds ORC.  os above RAND is the EXPECTED outcome.
* **H1 / H2** (A3): dFD = FD_N - FD_X (weights FB250 493/250, CT100 1363/100); NO-EFFECT iff its CI covers 0 (checked first); HEDGE-WORKS iff dFD >= 0.0873 (50 % of 0.1746) with lo > 0 and missed-decel (DC100) not separated worse; else MIXED; stamp ABOVE_N2_FLOOR.
* **F2 / S0a / S0b** (A4): on FB-N windows a_X = c_X - c_N2; CLOSES iff >= 0.50 with lo > 0, NOT-CLOSES iff hi < 0.25; gates (S0id==N, S0x moved, Fbrake raises CT100 decel, lo > 0) else VOID; mechanical combination in the JSON.

## Open / UNVERIFIED
* The force_lon roll (`E:/refcv8_eval_kit/force_lon_scratch/roll_full.json`) holds 20 / 100 windows and no process was alive 2026-10-19 20:45 (not touched, not killed).  My F2 is the first complete F2 run.
* R7 (refcv7 fan) arm dropped (design: optional, loader compatibility UNVERIFIED).  CPU is not bit-equal to the GPU dumps; only N2 / N_s1 measure the draw floor; one training seed.
* AL windows' `sel_idx` etc. come from native draws only; no rule uses seed-0-of-GPU.  Integration: land the 3 code files + this raw dir into the repo with the usual scoped commit.
