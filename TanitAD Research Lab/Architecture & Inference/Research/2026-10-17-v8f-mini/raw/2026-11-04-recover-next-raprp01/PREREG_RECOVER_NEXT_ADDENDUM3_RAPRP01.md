# PREREG_RECOVER_NEXT addendum 3: the read of RAPR-P01 and its committed next arm (frozen 2026-11-04, before any RAPR-P01 run, checkpoint or rollout exists)

Why this exists. Addendum 2 (sha256 a292eac5...) row E selected RAPR-P01 after the RAPR read (landed `cd6ce52d`): RAPR's frozen argv
with `--recover-p 0.1`, 2 runs, seeds 0 / 1 (argv `argv/RAPRP01_{0,1}.train.txt`). The card's section 4 table and its combiner are
written for RAPR, so no committed row covers RAPR-P01's outcomes. This addendum commits them BEFORE the data. State at freeze
(MEASURED): box 2 is idle (lock file moved aside 2026-11-04 15:54:25 after the RAPR pull), no RAPR-P01 chain has been started, and the
PI's box-2 go for RAPR-P01 has not been given. Every read rule, bar, control and row of the card and of addenda 1 and 2 is unchanged.

## The read
RAPR-P01 is read exactly as RTL, RP01 and RAPR: the card's section 3 rules with R := RAPR-P01 (PRIMARY, GUARDS, FLOOR, HARMS), the
composite RAPRP01C = RAPR-P01 + `--smooth sg72 --feasible kbox` vs P10 (Read A: D1, D2, D3 over four families, STRATEGIC included),
the frame-freeze control (`--scene hold_now`), 20 controls, T1-kin, paired episode-cluster bootstrap B = 2,000, seed 0, the same 141
episodes / 2,399 windows and the same comparators (C0, C0rep, FZC0, C0C, reused). The reader needs an `ARM_SPEC` entry for RAPRP01 with
a known answer before the read (instrument work).

## Committed next arm (rows in this order; the first that matches decides)
| # | outcome | next arm |
|---|---|---|
| M | the read is refused, or a control fails, or a composite is not computed | addendum 1 row B: no new box-2 arm; report the refusal, fix the instrument, re-read. |
| I | RAPRP01C DOMINANT at both seeds and counted (frame-freeze ROBUST or PARTIAL) | card row 1 / addendum 1 row A: evidence videos and the PI scale signal; a third seed pair of RAPR-P01 on fresh inits. |
| N | RAPRP01C DOMINANT at both seeds and not counted | no videos and no scale signal; report the frame-freeze reading as the blocker; next as row K. |
| J | RAPR-P01's GUARDS PASS at both seeds (ADE6 RAPR-P01 − C0 point ≤ +0.0604 m and no separated-worse non-crawl row) and RAPRP01C is not DOMINANT | **RAPR-P02**: RAPR's frozen argv with `--recover-p 0.2`, 2 runs. The guards hold at 0.1 and fail at 0.3, so the dose between them is the cheapest lever on the closure. |
| K | the GUARDS fail at either seed | the one-offset-per-window recovery recipe is closed at the laws and doses measured (bank law at p 0.3 and 0.1, a-priori law at p 0.3 and 0.1). Next arm: a PI-decision item between (a) card row 4's level-from-image auxiliary (architecture change) and (b) a per-tick deviation recipe (a new trainer lever, with its own launch-gate entry, motivated by report v37 §7.36.5's untested hypothesis). The Master Mind brings both with cost estimates; neither starts without the PI's decision and box-2 go. |

## Reported, not deciding
* RAPR-P01's M1 share closed against RP01's (0.1473 / 0.1342, the bank law at the same dose) and against RAPR's (0.4447 / 0.3175, the same law at p 0.3), per seed.
* ADE6 RAPR-P01 − C0 against RP01's (+0.0017 / +0.0008 m) and RAPR's (+0.3504 / +0.3202 m), per seed.
* The four metric families of RAPR-P01 − C0 and of RAPRP01C − P10, all 19 rows, per seed, never pooled.

## What this does NOT do
* It does not authorise a box-2 job: RAPR-P01 and every next arm need the PI's explicit go (box 2 runs one job at a time, SAFE MODE).
* It does not change which outcome is a success: only a counted DOMINANT is one.
* RAPR-P02's argv must be written from RAPR's frozen argv by changing one token (p), checked by token count, with a reader `ARM_SPEC` entry and a known answer before its read.

ESTIMATED (written before any number; not results): P(M) 0.05, P(I) 0.05, P(N) 0.01, P(J) 0.29, P(K) 0.60. Basis: at p 0.1 the bank law kept
ADE6 inside its guard and still failed on DK-VIOL; the a-priori law at p 0.3 failed on ADE6 and four longitudinal rows. Each 2-run arm is
about 1.3 h on box 2 (MEASURED for RAPR: about 2,250 s training per run plus about 2 min of launch gate); the read about 25 min on the dev box (MEASURED for RAPR: 15:54:45 to 16:18:05).
