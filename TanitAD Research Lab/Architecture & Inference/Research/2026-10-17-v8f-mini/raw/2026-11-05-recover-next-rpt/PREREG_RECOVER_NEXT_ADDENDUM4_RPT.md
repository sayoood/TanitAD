# PREREG_RECOVER_NEXT addendum 4: RPT, the per-tick deviation shape, its read and its committed next arm (frozen 2026-11-05 02:51:53, before any RPT run, checkpoint or rollout exists)

Why this exists. Addendum 3 row K (RAPR-P01's read, landed `05f9dea`) closed the one-offset-per-window recovery recipe and made the
next lever a PI decision. The decision item (`raw/2026-11-04-recover-rowk-decision/DECISION_ROWK.md`, landed `ff09d2c`) set out (a)
the level-from-image auxiliary and (b) a per-tick deviation recipe; **the PI chose (b) on 2026-11-04** ("(b) per-tick recipe"). This
addendum commits RPT's arm, read and outcome rows BEFORE any RPT data. Every read rule, bar, control and row of the card and of
addenda 1 to 3 is unchanged. State at freeze (MEASURED): box 2 idle (`C:/Users/Admin/box2_ship/BOX2_BUSY.txt` absent), no `M2b_s*_recft_RPT` directory in `E:/Projects/TanitAD-artifacts/v8f-recover`, the PI's box-2 go for RPT not given; the per-tick code landed at `eccdabd1ce63` (sha256 trainer `bda468fe…`, `v8f_recover.py` `0257ccb9…`, `v8f_mini_data.py` `209da7a6…`, checked at freeze against that commit); the reader's `ARM_SPEC` RPT reproduces R's read as its known answer; the results writer `gen_rpt_results.py` (sha256 `5959fdb1…`) and its 21 mutation cases (all PASS) are banked in this folder with the argv, the instruments and the launcher.

## The arm
RPT = R's frozen argv (`raw/2026-11-02-recover-ft-launch/box2_files/argv/R_{0,1}.train.txt`: the empirical TRAIN-bank deviation law, p 0.3, rejoin
target, the seed's own bank and bank sha256) with `--out ...\M2b_s{s}_recft_RPT` and **`--recover-dev-shape pertick`** appended;
2 runs, seeds 0 / 1, the same init checkpoints as R. Argv `argv/RPT_{0,1}.train.txt` (64 tokens; differs from R's in the
`--out` value and the two appended tokens, checked by test). The shape (trainer `stack/scripts/v8f_mini_train.py` and
`stack/tanitad/data/v8f_recover.py` at `eccdabd1ce63`): each selected sample draws ONE record from the bank as R does (same random
stream, same record index), and instead of R's single offset (dv, da, dom) at the clock row j applied to every history frame, the
history frame k receives the record's own-minus-logged deviation AT THAT ROW (rows j − 9 ... j, DT 0.2 s); the pose history, the yaw
history and the route-distance channel are re-integrated from the per-row speed and yaw-rate offsets; the lateral and heading
offsets (dy, dpsi) and the target are R's. The NOW row of every profile equals R's offset bit for bit (pinned by test), so RPT and
R differ in the shape of the deviation over the history window and in nothing else. Records whose per-row deviation is not finite
are dropped and counted (reported, not deciding).

## The read
RPT is read exactly as R, RTL, RP01, RAPR and RAPR-P01: the card's section 3 rules with R := RPT (PRIMARY, GUARDS, FLOOR, HARMS),
the composite RPTC = RPT + `--smooth sg72 --feasible kbox` vs P10 (Read A: D1, D2, D3 over four families, STRATEGIC included), the
frame-freeze control (`--scene hold_now`), 20 controls, T1-kin, paired episode-cluster bootstrap B = 2,000, seed 0, the same 141
episodes / 2,399 windows and the same comparators (C0, C0rep, FZC0, C0C, reused). The reader needs an `ARM_SPEC` entry for RPT with
a known answer before the read (instrument work).

## Committed next arm (rows in this order; the first that matches decides)
| # | outcome | next arm |
|---|---|---|
| M | the read is refused, or a control fails, or a composite is not computed | addendum 1 row B: no new box-2 arm; report the refusal, fix the instrument, re-read. |
| I | RPTC DOMINANT at both seeds and counted (frame-freeze ROBUST or PARTIAL) | card row 1 / addendum 1 row A: evidence videos and the PI scale signal; a third seed pair of RPT on fresh inits. |
| N | RPTC DOMINANT at both seeds and not counted | no videos and no scale signal; report the frame-freeze reading as the blocker; next as row K2. |
| J | RPT's GUARDS PASS at both seeds (ADE6 RPT − C0 point ≤ +0.0604 m and no separated-worse non-crawl row) and RPTC is not DOMINANT | **RPT-P05**: RPT's argv with `--recover-p 0.5`, 2 runs (the guards leave room, and the closure-need estimate asks for more closure). |
| K1 | the GUARDS fail at either seed, and RPT's M1 share closed exceeds R's (0.2198 / 0.1318) at BOTH seeds | **RPT-P01**: RPT's argv with `--recover-p 0.1`, 2 runs (the shape adds closure at R's dose; RP01's dose is the one that removed R's ADE6 harm). |
| K2 | the GUARDS fail at either seed, and RPT's M1 share closed does not exceed R's at both seeds | the per-tick shape adds no closure at the dose measured: next arm is option (a), the level-from-image auxiliary of the decision item, as a PI-decision item with its costs; it does not start without the PI's decision and box-2 go. |

## Reported, not deciding
* RPT against R (same law, same dose, constant shape): M1 share closed, ADE6 X − C0 and the 19 rows, per seed, side by side (two
  separate paired reads against C0; there is no paired RPT − R interval, and none is claimed).
* The per-tick table's record counts (kept, dropped for a non-finite row) per seed, and its NOW-column identity with R's table.
* The four metric families of RPT − C0 and of RPTC − P10, all 19 rows, per seed, never pooled.
* Against the decision item's closure-need estimate (about 0.68 / 0.60, post hoc, not a rule): where RPT's closure falls.

## What this does NOT do
* It does not authorise a box-2 job: RPT and every next arm need the PI's explicit go (box 2 runs one job at a time, SAFE MODE).
* It does not change which outcome is a success: only a counted DOMINANT is one.
* RPT-P05 and RPT-P01 argv are written from RPT's argv by changing one token (p), checked by token count, each with a reader
  `ARM_SPEC` entry and a known answer before its read.

ESTIMATED (written before any number; not results): P(M) 0.05, P(I) 0.08, P(N) 0.02, P(J) 0.25, P(K1) 0.25, P(K2) 0.35. Basis: the
bank law at p 0.3 failed its guards with constant offsets (R: ADE6 +0.1892 / +0.0977 m and seven / two separated-worse rows); a per-row
profile moves the history less at its older rows whenever the deviation grows toward NOW, so its closure and its harm may both be
smaller than R's; the per-tick oracle (HLVL) closes 0.599 to 0.633 without training, which this arm does not reproduce. Each 2-run arm
is about 1.3 h on box 2 (MEASURED for RAPR-P01: 78 min including both gates); the read about 30 min on the dev box (MEASURED for
RAPR-P01: 20:10 to 20:40).
