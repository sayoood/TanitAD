# PREREG_RECOVER_NEXT — R-APR, plus the read rules of the committed HARMS arms (frozen 2026-11-03, before any of these runs exist)

Tier T1-kin (self-action open loop on logged cameras; not closed loop). Estimator: paired episode-cluster bootstrap,
B = 2,000, seed 0, over the 141-episode / 2,399-window eval lane (answers "another draw of episodes" only; the
training-variance check is the existing C0rep_s floor). Reader: `recover_read/code/recover_read.py`, extended only by an
`--r-arm <ARM>` option that substitutes the run `M2b_s{s}_recft_<ARM>` for `M2b_s{s}_recft_R` (every rule, bar,
control and row unchanged; a known answer must hold: `--r-arm R` reproduces `raw/RECOVER_READ.json` exactly).

## 1. What the RECOVER-FT read gave (MEASURED, `recover_read/raw/RECOVER_READ.json`, the only basis of this card)
Label **HARMS** (H-RECOVER-FT-1): M1 crawl share R − C0 separated-better at both seeds (−0.0117 / −0.0078) but share
closed 0.220 / 0.132 (bar 0.5); ADE6 R − C0 +0.189 (s0, separated) / +0.098 m (s1), beyond the +0.0604 m guard; frame-freeze
PARTIAL. Composite RC = R + S + kbox vs P10: D1 and D3 pass at both seeds, **D2 fails on M1 crawl share alone**
(+0.048 / +0.057) — NOT DOMINANT. Replicate floor C0rep − C0: ADE6 +0.306 m (s0, separated) / +0.007 m (s1).

## 2. Arms (box 2, one job at a time, in this order; init, data, steps, every other flag = R_s's shipped argv)
| order | arm | change vs R_s | committed by |
|---|---|---|---|
| 1, 2 | RAPR_0, RAPR_1 | `--recover-dev-law apriori` (the law fixed in RECOVER-FT section 3 before any data), bank flags dropped | this card (RECOVER-FT section 8 names it as the NOT-MOVED arm) |
| 3, 4 | RTL_0, RTL_1 | `--recover-target logged` (no rejoin blend) | RECOVER-FT section 8, HARMS |
| 5, 6 | RP01_0, RP01_1 | `--recover-p 0.1` | RECOVER-FT section 8, HARMS |
Comparators reuse the existing C0_s, C0rep_s and P10 rollouts. Argv: `recover_next/argv/`.

**Disclosed post-data choice.** RAPR is added and put first AFTER the R read: what blocks DOMINANT is M1 alone (D1
has ~0.7 m slack over its bar), and the a-priori law is heavier on the slow side (dv ~ U[−3, 0] m/s) where the crawl
lives. The choice of WHICH pre-specified arm to run is post-data; its law and every reading rule below are not.

## 3. Read (per arm X in {RAPR, RTL, RP01}; the RECOVER-FT section 6 / 7 rules with R := X)
Rollouts per seed: X logged, X hold_now (frame-freeze), XC = X + `--smooth sg72 --feasible kbox`; all `--smooth sg72`.
* **Label of X** by the RECOVER-FT label table (HARMS / NOT MOVED / INSIDE-FLOOR / RECOVERS / ARTEFACT / PARTIAL).
* **Composite of X**: DOMINANT iff Read A of XC vs P10 holds at BOTH seeds (D1 ADE6 separated-better with point
  <= −0.2415 m; D2 no row of 19 separated-worse; D3 a separated-better row in each of the four families, STRATEGIC
  included), counted only if X's frame-freeze reading is ROBUST or PARTIAL.
* **Attribution (Q1, committed HARMS question):** blend harmed iff RTL's guards PASS at both seeds; dose harmed iff
  RP01's guards PASS at both seeds; both / neither otherwise. Reported with RTL's and RP01's M1 share closed.
* **RAPR secondary:** M1 share closed by RAPR exceeds R's (0.220 / 0.132) at each seed.

## 4. Committed next arm per outcome (Rule Zero)
| outcome | next arm |
|---|---|
| any XC DOMINANT, counted | evidence videos + the PI scale signal; a third seed pair of X on fresh inits. |
| RAPRC fails D2 on M1 alone and RAPR's M1 closure exceeds R's at both seeds | dose up: RAPR at `--recover-p 0.5` (2 runs, ~1.3 h). |
| RAPRC fails D2 on a non-M1 row | the a-priori law with whichever target / dose the attribution says does not harm (RTL or RP01 setting). |
| RAPR's M1 closure does not exceed R's at either seed | the lever is not the size of the training deviation: the level-from-image auxiliary (a head regressing the logged speed from the current frame tokens, fed back as a feature; RECOVER-FT section 8), as a PI-decision item (architecture change). |

## 5. ESTIMATED (not results)
P(any XC DOMINANT, counted) ~0.12 (RAPR ~0.10). At s0 any X − C0 difference is read against a training floor of
ADE6 0.306 m / M1 0.036 (MEASURED above); a single-seed s0 contrast below that is not a lever effect.
Box-2 cost: RAPR ~0.7 h per run (no table build); RTL / RP01 ~2 h per run (gate table build ~42 min + start build
~42 min + training). Whole chain ~9.5 h. Dev-box read ~2 h. Each box-2 start needs the PI's go.
