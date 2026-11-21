# Card PREREG_V8F_PROOF_ADD70 (LONGROLL10K): the rollout at an age of 6 to 12 s and 12 to 18 s on the bounded executor; no training

Card of the v8F proof study, written by the coordinator fork after the read of card ADD67 (row 122), while the rollouts of cards ADD68 and ADD69 are on the pod and unread, before the wrapper of this card exists and before any of its runs (time of the freeze: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). Tier of every rollout number: T1-kin (kinematic rollout on the model's own actions; logged camera frames; the lead does not react); never driving performance. Executor: heading bound 0.2 1/m in every run (the tier of record by the PI's decision of 2026-11-20, row 125; stamp T1-kin/hb0.2, study wrapper).

## 1. What is known (MEASURED)

* Every T1-kin number of the study is read on windows of 6 s: three plans of the model at the 2.0 s cycle, started from the logged state. The programme's bands are 2 s (operative), 6 s (tactical) and 6 to 30 s (strategic); nothing has been rolled beyond 6 s.
* With six plans inside 6 s (the 1.0 s cycle) the speed climbs and the unsafe rate rises under every readout tried on the executor without a heading bound (rows 108 to 116). The errors rise with the share of its own acceleration the model is handed at the replan tick (row 119). So a cost per replan exists; whether it also appears with six or nine plans at the 2.0 s cycle is not measured.
* The inference-side candidate (readout K, the lead rule, its second look) with the heading bound reads better than the convention on all four families on the 6 s windows (rows 121, 122). Whether that holds at a higher rollout age is not measured.
* The strategic head's route goal degrades on the model's own rollout inside 6 s (row 120).

## 2. Question

Q1. Does a window read worse when the model has already driven the 6 s before it on its own actions (rollout age 6 to 12 s) than when it starts from the logged state, for the convention, for readout K and for the two candidates? Q2. Does the candidate's advantage over the convention hold at that age? Q3 (diagnostic). Is a cost at that age the cost of the route input held for 6 s, or of the own-motion state?

## 3. Mechanism

A new parameter of the study wrapper, `--preroll-steps P` (wrapper `v8f_t1_eval_reader_rule_rzgfhp.py`, made from the wrapper of card ADD67 by `make_longroll10k.py`; with P = 0 it is that wrapper bit for bit). With P above 0 every window (clip, t0) of the grid is started P steps earlier: the state is built at t0 - P, rolled for P / 10 plans on the model's own actions by the same planner with the same rule, readout, second look and heading bound (the pre-roll; its rows go to separate log files), and then the pinned tool rolls the window's 30 steps as always, with its sample-and-hold ages restarted as for a fresh window. Every path of the window is written in the LOGGED ego frame at t0, so the pose error carried out of the pre-roll counts as error, and ground truth, labels and every reader stay those of window (clip, t0). The route input stays dead-reckoned from the start of the pre-roll, as the tool does it. The pinned tool is not edited. Grids: the pinned grid restricted to the windows with t0 - P at or above 3 (`make_longroll_grid.py`): expected 1,271 windows at P = 30 and 284 at P = 60 (the pinned grid has 2,399; starts 3, 7, ..., 67 or 71 per clip).

## 4. Arms

Line models `UE_s0` and `UE_s1` (10k line, code cdbff971), reader seeds as in card ADD67, 2.0 s cycle, `--nav deadreckon --scene logged --schedule periodic --limit causal`, `--kappa-max 0.2` in every run. Run names carry the prefix `I10` on the pod.

| form | parameters | P = 30 | P = 60 | fresh reference (banked, card ADD67) |
|---|---|---|---|---|
| CONV | no rule, `--readout pinned` | `R0P30Hb20` | `R0P60Hb20` | `R0Hb20` |
| K | no rule, `--readout causal_g --acc-gain 0 --path-exec` | `R0KP30Hb20` | `R0KP60Hb20` | `R0KHb20` |
| LABEL-CAND | label-fed rule with closing speed, readout K, `--rule-every 5` | `RLcKF5P30Hb20` | `RLcKF5P60Hb20` | `RLcKF5Hb20` |
| READER-CAND | reader-fed rule at map level 0.50, readout K, `--rule-every 5` | `RQ50KF5P30Hb20` | `RQ50KF5P60Hb20` | `RQ50KF5Hb20` |

Diagnostic arms for Q3 (form K, both seeds, `--nav logged`, on the P = 30 grid): `R0KNlP30Hb20` (P = 30) and `R0KNlHb20` (P = 0). The logged route is taken from the ego's own future path and is optimistic by construction; these two runs are read against each other only.

Controls at model seed 0: `R0LrID` (new wrapper, P = 0, CONV, full grid), `R0LrCarry` (P = 30, `--preroll-exec logged`, `--nav logged`, CONV) and `R0LrFresh` (P = 0, `--nav logged`, CONV, P = 30 grid). In all: 16 + 4 + 3 rollouts.

## 5. Reads

All pairs are paired on identical windows (the P grid); a banked run is restricted to those windows by (clip, t0). Estimators as in cards ADD46 and ADD67: headway shares U (unsafe), C (contact), F (falling behind) by the paired episode-cluster bootstrap pooled over the two model seeds; the ten family cells by the family reader (two-level bootstrap, seeds then episodes, B = 2,000); a cell is BETTER or WORSE only if its interval excludes 0 and the absolute difference exceeds the spread of the fresh convention run `R0Hb20` over the two model seeds on the same windows; INSIDE-SPREAD and OPEN as before. Family rule as in card ADD67 (LONGITUDINAL: speed and along error; LATERAL: heading and yaw-rate error; TACTICAL: the 6 s track error; STRATEGIC: the route goal at 30, 50 and 80 m at the third tick, instrument of card ADD66; a decision between routes is not scored, n = 0).

Named pairs at P = 30:
* AGE pairs (pre-rolled minus fresh, same form): AGE-CONV `R0P30Hb20` - `R0Hb20`; AGE-K `R0KP30Hb20` - `R0KHb20`; AGE-LABEL-CAND `RLcKF5P30Hb20` - `RLcKF5Hb20`; AGE-READER-CAND `RQ50KF5P30Hb20` - `RQ50KF5Hb20`.
* AT-AGE pairs (both pre-rolled): K-AT-AGE `R0KP30Hb20` - `R0P30Hb20`; LABEL-CAND-AT-AGE `RLcKF5P30Hb20` - `R0P30Hb20`; READER-CAND-AT-AGE `RQ50KF5P30Hb20` - `R0P30Hb20`.
* NAV pair (Q3): AGE-K-NAVLOGGED `R0KNlP30Hb20` - `R0KNlHb20`.

Beside, descriptive, naming nothing: the same pairs at P = 60 (few windows); the AT-AGE pairs on the fresh runs restricted to the same windows and the difference of the two differences with its interval; the rule under K at age (`RLcKF5P30Hb20` - `R0KP30Hb20`, `RQ50KF5P30Hb20` - `R0KP30Hb20`) for U, C, F; the pose, heading and speed offsets at t0 from the pre-roll logs per form; levels of U, C, F per run and seed; counts of scored and lead windows on each grid.

## 6. Names and consequences (fixed here)

Per named pair `<P>` and family: `LONGROLL10K-<P>-<FAMILY>-BETTER | WORSE | MIXED | NOT-SEPARATED`; for the unsafe rate `LONGROLL10K-<P>-U-LOWER | HIGHER | NOT-SEPARATED` (interval excludes 0 or not).

* AGE-CONV with a family WORSE or U HIGHER, and neither candidate's AGE pair with a family WORSE or U HIGHER: the convention's errors grow with rollout age and the candidate's do not; this goes into the candidate specification (recommendation A of decision card D16), and reporting the read at age 6 to 12 s beside the 6 s windows is proposed to the PI and the other session.
* A candidate's AGE pair with a family WORSE or U HIGHER: the model's reading of its own rolled history costs at 2.0 s too. The AGE read then is the acceptance check for the training line of recommendation C: an arm trained on its own rolled history must read NOT-SEPARATED or BETTER on every family and U NOT-SEPARATED or LOWER where the line models read WORSE or HIGHER here.
* LABEL-CAND-AT-AGE and READER-CAND-AT-AGE with no family WORSE, at least one BETTER and U not HIGHER: the candidate's advantage holds at age 6 to 12 s (stated per form). A family WORSE at age: the specification states that limit.
* AGE-K-NAVLOGGED NOT-SEPARATED on a family where AGE-K is WORSE: that cost is the held route input (a route supply during the rollout is then the lever, not the state); WORSE in both: the own-motion state.
* All AGE pairs NOT-SEPARATED on every family and U: no cost of rollout age is measured at 6 to 12 s on 1,271 windows and two model seeds; stated as such, with the intervals.
A pre-rolled window that reads BETTER than a fresh one is reported as written and decides nothing. Adopting a rollout age for the programme's convention is a decision of the PI and the other session; this card measures.

## 7. Controls (all must pass; else nothing is named)

* e1: environment record of the pod read (as in card ADD67).
* l1: `check_preroll.py` on the pod: PASS (frame mapping, plan mapping, grid restriction, order of the two rollout calls on a stub, log split; each with a planted wrong arm).
* l2: `R0LrID` at seed 0 equals the dump of `R0Hb20` at seed 0 of card ADD67, 141 files, 0 differing.
* l3 (state carry): `R0LrCarry` against `R0LrFresh` on the P = 30 grid: with the logged motion as pre-roll the window must read as a fresh one. The tolerance is fixed before the queue by the smoke on a two-clip mini grid (the first two clips of the grid order; a known-value check, no outcome): d = the largest absolute difference of any dump array there; tolerance = 10 x d, at least 1e-6 m and at most 0.05 m. If d exceeds 0.005 m the instrument is not qualified: the card stops, the cause is found and written as a note before anything else runs.
* l4: window counts: the grid tool's counts equal 1,271 and 284 or the card's count is replaced by the tool's with the per-clip rule checked (starts at or above 3 + P); every P run holds exactly its grid's windows; the restriction of a banked run returns the same (clip, t0) set; rule-log and tick-log rows = windows x 3 in every run.
* l5: the pre-roll logs: plans pre-rolled = P / 10 per window in every P run; median position offset above 0 for the plan executor; for `R0LrCarry` the largest position offset at or below the tolerance of l3.
* l6: reader identity: with the restriction set to all windows the reader reproduces the pair `R0Hb20` - `R0` of card ADD67 (ten cells and U, C, F) to 1e-9.
* l7: hashes: pinned tool, wrapper, parent wrapper, the reused result files, checkpoints, reader and map files.
* l8: readout logs: `kappa_max` 0.2 and frames counted in every run; form, readout, rule interval and replan steps as declared; no pre-roll row in a file a reader consumes (row counts of l4), pre-roll rows present in their own files.
* l9: the strategic reader's rows: three ticks per window on each grid.

## 8. Limits

T1-kin with logged frames: at a rollout age of 6 s the model's pose differs from the pose the camera frame was taken from (the offsets are reported); the scene and the lead do not react; no re-perception. The P grids hold the later windows of each clip only, so levels are not comparable with rows read on the full grid; every named pair is paired on identical windows. Two model seeds: a separated interval is necessary, not sufficient, and the spread rule stands in for a replicate. P = 60 has 284 windows and names nothing. The route input is held from the start of the pre-roll. STRATEGIC is the route goal only. Nothing here is a statement about driving.
