# Amendment 1 to card PREREG_V8F_PROOF_ADD72 (AGEDOSE10K): the wrapper of record refuses a pre-roll below 30 steps; the shape of the state part is read at 6 and 12 s from banked runs

Written by the coordinator fork after the queue of the card stopped at its first rollout and before any value of this card exists (time: the line of this amendment in `TIMES_NOTE.md`; hash in the sidecar file). The card as frozen (sha256 in its sidecar) stays on record.

## 1. What stopped

`AGEDOSE10K_STOP.txt`: `ROLLOUT-RC 1 I10R0KP5Hb20_s0`, two seconds after its start; the wrapper's own line: "REFUSED: --preroll-steps 5 is not a multiple of --replan-steps 10". The wrapper of record of card ADD70 chains whole 30-step windows (a pre-roll has to be a multiple of 30 steps and of the replan cycle; its text at the refusal: one pre-roll segment is the 30-step logged path of the rule's arc tables). No dump, no tick file and no value exists. The card was written without that property of the wrapper checked: an error of the card, found by the wrapper's refusal. Sections 2, 3a and the candidate sentence of section 5 of the card cannot be run with the wrapper of record and are not run. Ages of 1, 2 and 4 s need a wrapper generation with partial segments; that is named as a possible later card and is not opened here.

## 2. What is read instead (no rollout, no GPU; CPU read on the pod)

The question of the card on the older side, with banked runs of card ADD70: does the state part of the cost of age grow from 6 s to 12 s of own driving, or is it complete by 6 s?

* Runs, banked and used as they are: `I10R0KHb20` (fresh), `I10R0KP30Hb20` (6 s), `I10R0KP60Hb20` (12 s), `I10R0Hb20` (spread run), both model seeds.
* Windows: the P = 60 grid of card ADD70 (`longroll/grid_p60.json`, 284 windows; each is a window of the P = 30 grid). The 6 s run and the fresh run are restricted to these windows.
* Split per age P in 30, 60 as in section 3b of the card (tool of card ADD71 part B with its note N2): POSE(P), STATE(P), total, share, offsets.
* Shape of the state part for the four cells of section 3c (TRACK, ALONG, SPEED, HEADING), S(P) = STATE part at age P on the windows where the cell has a value at both ages, two-level bootstrap as in the card:
  * `AGEDOSE10K-STATE-<CELL>-PLATEAU` when the interval of S(60) minus 1.5 times S(30) excludes 0 below;
  * `AGEDOSE10K-STATE-<CELL>-COMPOUNDS` when that interval excludes 0 above;
  * `AGEDOSE10K-STATE-<CELL>-BETWEEN` otherwise;
  * `AGEDOSE10K-STATE-<CELL>-NO-STATE-COST` when S(30) on these windows is not above 0 with its interval excluding 0.
  Why 1.5, fixed now: a state part that grows in proportion to the age doubles from 6 to 12 s; one that is complete by 6 s stays. The bar is the midpoint. Reported without a bar: S(30), S(60), their ratio, POSE and total at both ages, the share of POSE, the offsets.

## 3. Controls (a control that does not pass: no name; logged before a repair)

* b1, b2, b3 of card ADD71 once; b5 and b6 at each age on the 284 windows; the planted start offset of note N4 of card ADD70.
* b4 at each age: POSE plus STATE equals the total to 1e-9.
* b4-bank: the same tool run for the pair `I10R0KP30Hb20` minus `I10R0KHb20` on the full P = 30 grid returns the total of the pair AGE-K of the re-read of card ADD70 to 1e-9 on every cell (this checks run names, tick files and the cell path of the K pair against a banked read; its parts are written beside and name nothing).
* g1: every window of the P = 60 grid is a window of the P = 30 grid, and the three runs hold the same 284 windows in the same order after the restriction.
* d1: on planted arrays the shape statistic returns PLATEAU for equal state parts, COMPOUNDS for a doubled one, BETWEEN for 1.5 times, NO-STATE-COST for a state part of 0 at 6 s.

## 4. What follows, fixed now (replaces section 5 of the card; nothing is adopted)

* PLATEAU on TRACK: the state part does not compound between 6 and 12 s; what grows there is the carried pose, which a kinematic rollout cannot repair or fairly score. Acceptance check C5 at 6 s then covers the state side at 12 s, and a training arm on own rolled history needs rolled histories of 6 s, not longer, for the state side.
* COMPOUNDS on TRACK: the state part grows with the age; a training arm has to roll as long as it is to be used, and C5 gets a second age (12 s) in the candidate specification.
* BETWEEN: both ages are reported and C5 stays as it is.
* The names on ALONG, SPEED and HEADING are reported beside and decide nothing by themselves.

## 5. Known at this time, and limits

Known: the totals at both ages (tactical track error higher by 14.8 m after 6 s and by 34.8 m after 12 s under readout K on their own grids; pose offsets of about 6 m and 21 m) and the split at 6 s of the pair with the logged route (card ADD71 part B). Not known: any POSE or STATE part at 12 s, and the split at 6 s on the 284 windows. With the totals known this is not a blind read of the totals; the parts are unread.

Limits: the 284 windows are those with 12 s of clip before t0, another population than the 1,271; nothing is said about ages below 6 s; the candidate block of the card is not run and the hold of the label-fed rule on the unsafe rate stays measured at 6 s (row 129); the headway instrument is not used here (it is not qualified on the P = 60 grid). Two model seeds; a separated interval answers another draw of episodes and is necessary, not sufficient.
