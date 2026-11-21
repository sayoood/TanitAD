# PREREG LEVELHEAD (arm CL): option (a), the level-from-image head (written 2026-11-05 before any CL run; frozen before launch)

**PI decision (verbatim, 2026-11-05):** *"Do c then a, contibue autonmously until scalabiliry readiness is achieved"*. Option (c)
stopped on named blockers (`raw/2026-11-05-motion-c-crawllead/C_STATUS.md`: T2 re-perception, A40). This card is (a), row K2's
committed next arm (addendum 4), on the allocated box 2. Claim row: H-LEVELHEAD-1.

## 1. Hypothesis
O's crawl (own speed < half the logged speed at the matched route point while that speed is >= 3 m/s) persists because O sets its
speed level from its own executed history; a vision-only estimate of the speed the scene affords, fed to O at every tick, gives O
a level that does not decay with its own history. Evidence it can exist: one cached frame reads the logged speed at R² 0.767 over
all windows (MOTIONPROBE F1, TRAIN, clip folds), a scene-context prior, not motion.

## 2. The arm
* **CL_s{0,1}** = C0's argv (`v8f-recover/M2b_s{s}_recft_C0/argv.json`: init `M2b_s{s}_dagoacc`, 3,000 steps, lr 1e-4, batch 64,
  pool4x10, recover-p 0) + `--level-head on --w-level W`, nothing else (argv audit: exactly these tokens differ from C0's, plus
  `--out`). The lever is built in `v8f_mini.py` / `v8f_mini_train.py` (default off = today's model bit for bit), its G-DVB and
  init checks in the trainer's gate.
* **W** is fixed by formula before any run: W = (0.1 / 0.9) × L_rest / L_level, with L_rest the total loss without the level
  term and L_level the unweighted level loss, both measured at step 0 of a real-data selfcheck on the C0 s0 init with the head on
  (box-2 W probe, trainer key `level_head_probe.w_level_for_10pct_share`), so the level term is a tenth of the total, itself
  included, at the start. The value is written into
  `FROZEN_W.txt` and this card's freeze line before launch.
* Head: mean of the current tick frame's adapted tokens → MLP → v_hat; Huber(v_hat, logged v(t)); O reads e_level(v_hat.detach()/10)
  at every tick, train and inference alike, ADDED to O's ego conditioning token (zero-init, so the step-0 forward equals C0's init
  bit for bit; an appended token would change the Reader's query mean, as the build's test shows; build `motion_c/build/`,
  model a3634276, trainer 9ffdf1b4). Vision only at inference (ego, nav, limit, T's code refused by a build guard).
* Base and replicate for the contrasts: C0 and C0REP (reused, finished).

## 3. Reads (T1-kin, the RECOVER-NEXT reader's functions; paired episode-cluster bootstrap B = 2000, seed 0, eval-141)
Per seed, three scene modes, each with its composite (`--smooth sg72 --feasible kbox`), on the RI tool (bit-identical to the
FRAMEFREEZE tool on `logged` and `hold_now`):
| mode | arm | its base |
|---|---|---|
| `reindexed` (**deciding**) | RICL, RICLC | RIC0, RIC0C (landed b07982c) |
| `logged` | CL, CLC | C0, C0C |
| `hold_now` | FZCL | FZC0 |

## 4. Committed rules (all evaluated on `reindexed` unless named)
* **P (primary):** M1 crawl share closed by CL vs C0, share = (C0 − CL)[M1] / (C0 − P10)[M1], >= 0.5 at both seeds, with
  CL − C0 on M1 separated-better.
* **G (guards):** no non-crawl row of CL − C0 separated-worse; ADE6 CL − C0 <= +0.0604 m (upper end of the interval).
* **F (floor):** |CL − C0| on M1 exceeds |C0REP − C0| on M1 (logged; the rig's training-variance floor).
* **D (counted DOMINANT):** composite CLC − P10 on `reindexed`: D1, D2 (no row of 19 separated-worse) and D3 (four families,
  STRATEGIC included) at both seeds, with P and G holding. Only this counts; it is the PI's "dominant driving result".
* **LEAK-BORNE:** if the `logged` share minus the `reindexed` share exceeds half the `logged` share at either seed, the row is
  labelled LEAK-BORNE whatever else it reads (DESIGN_C.md section 4).
* Frame-freeze (hold_now) is reported beside, not deciding (a frame-reading lever is expected to move under a freeze).

## 5. Outcomes and the next arm, committed now (Rule Zero: no outcome ends at its own verdict)
* **D holds:** counted DOMINANT. Signal the PI for scale (data and size), with the video evidence the PI asked for.
* **P holds, G or D fails:** next arm CL-W2 (W / 2), the same card otherwise; then the composite's failing row is the target.
* **P fails, CL − C0 on M1 separated-better:** next arm **CLR** = CL + RAPR-P01's recovery recipe (a-priori law, p 0.1; the
  recovery family with the smallest guard harm), because the recovery fine-tune lacked a level to recover TO and CL supplies it.
* **P fails, M1 not moved:** the head's level does not reach O's decision: next is the head read-out check (does v_hat track the
  logged speed on eval at T0? R² and the M1-window R²) and, if it does, O's use of it (attribution read with e_level zeroed at
  inference); if it does not, a lead-aware head (lead presence and distance from the frame, CRAWL-LEAD: half the crawl sits
  behind a lead).
* **LEAK-BORNE:** reported as the camera-clock effect, never as a result; next as for "P fails".

## 6. Variance named
The bootstrap answers another draw of EPISODES only. Training variance: the C0REP floor (rule F). Inference: the planner is
deterministic at eval (no sampling), so no inference seed applies.

## 7. Freeze
**FROZEN W = 0.2027** (2026-11-05 launch: `cl_prereg/FROZEN_W.txt`; the W probe's level_head_probe.w_level_for_10pct_share = 0.20274419640036745, selfcheck sha256 0d274fc5bff68d98fbc620331e89e3fa94c493c648db11614cda854bb4ef496a, step-0 L_without_level 20.45931339263916, L_level 11.21243953704834; the level term is a tenth of the total at step 0)
