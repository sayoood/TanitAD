# PREREG_V8F_PROOF_ADD73 (HANDSRC10K): which input of the state hand-over at the first scored tick carries the state part of the cost of rollout age

Card of the coordinator fork (v8F proof study). Written after rows 129 to 131 and after the wrapper generation of this card was built and reviewed, and before any rollout with it exists, the pod check of the wrapper included (time: the line of this card in `TIMES_NOTE.md`; hash in the sidecar file). No training. Tier of every number: T1-kin/hb0.2 under a study wrapper (kinematic rollout on the model's own actions with the heading bound 0.2 1/m; logged camera frames; the lead does not react); never driving performance. Models: the two 10k line checkpoints `UE_s0`, `UE_s1` of the other session (code cdbff971), INHERITED.

## 1. Question and what is known

Known (MEASURED, rows 130, 131): after 6 s of own driving under readout K the tactical track error of a window is higher than fresh by 14.8 m on the 1,271 windows of the P = 30 grid; 7.33 m of it is the carried pose (the fresh track moved rigidly to the own pose) and 7.47 m the own motion state (the aged track minus that moved track); the state part does not compound from 6 to 12 s.

Not known: where the state part enters. At the first tick of the scored window the model reads, of its own driving: the speed group (speed, acceleration, the speed-bucket input), the path group (pose x, y, heading and the yaw-rate channel), and the route channel held from before the pre-roll. At the later ticks of the window it reads what it executed inside the window, as a fresh window does. The design note `integ10k/HANDOVER_SRC_DESIGN.md` lists every state input with its source.

## 2. Runs (6 rollouts, pod of this fork, one GPU job at a time; about 30 minutes)

P = 30 grid of card ADD70, 2.0 s cycle, readout K (`--readout causal_g --acc-gain 0 --path-exec`), form R0 (no rule), `--kappa-max 0.2`, `--preroll-steps 30`, model seeds 0 and 1, wrapper `integ10k/v8f_t1_eval_reader_rule_rzgfhqs.py` (written by `integ10k/make_handover_src.py` from the wrapper of record of card ADD70; one more parameter `--t0-state`, which replaces, at the first scored tick only and only in the batch the model is handed, one group of the state by what a fresh window at the same t0 reads; the pre-roll, the executor, the own pose and the later ticks are not changed):

* `I10R0KP30LgHb20`: `--t0-state logged` (everything the model reads at that tick is the logged state, the route channel included);
* `I10R0KP30SlHb20`: `--t0-state speed_logged` (the speed group logged, the rest own);
* `I10R0KP30PlHb20`: `--t0-state path_logged` (the path group logged, the rest own).

Banked and used as they are: `I10R0KP30Hb20` (own: the wrapper of record), `I10R0KHb20` (fresh), `I10R0Hb20` (spread run). The three modes are oracles: after own driving a logged state does not exist at deployment. The arm locates where the cost enters; it proposes no readout and adopts nothing.

## 3. Readings and names

With F the fresh track, D the fresh track moved rigidly to the own pose at t0 (the same for every mode: the pre-roll is not touched), A_own and A_M the aged tracks of the own run and of mode M, per cell on the windows where the cell has a value in all tracks used, two-level bootstrap of the family reader (model seeds and episodes, B 2,000):

* state part of mode M: A_M minus D; removed by mode M: A_own minus A_M (the moved track cancels).
* Names for M in LOGGED, SPEED-LOGGED, PATH-LOGGED and four cells, TRACK (`TACTICAL:ade_tac_m`), ALONG (`LONGITUDINAL:along_mae_m`), SPEED (`LONGITUDINAL:speed_mae_mps`), HEADING (`LATERAL:heading_mae_deg`):
  * `HANDSRC10K-<M>-<CELL>-CARRIES` when the interval of (A_own minus A_M) minus one half of (A_own minus D) excludes 0 above (the mode removes at least half of the state part);
  * `HANDSRC10K-<M>-<CELL>-PART` when not CARRIES and the interval of A_own minus A_M excludes 0 above;
  * `HANDSRC10K-<M>-<CELL>-WORSE` when the interval of A_own minus A_M excludes 0 below;
  * `HANDSRC10K-<M>-<CELL>-NOT-SEPARATED` otherwise;
  * `HANDSRC10K-<M>-<CELL>-NO-STATE-COST` when A_own minus D is not above 0 with its interval excluding 0 on these windows.
* Reported without a bar: the state part of every mode with its interval for these four cells and for the cross-track, yaw-rate and decision error rate cells, the removed share, and the same over the first replan segment alone (steps 1 to 10), where only the first tick has acted.
* The four families for the pairs mode minus own at age, when the family reader of card ADD71 can be run on the new runs with its controls passing: LONGITUDINAL with the headway read (unsafe rate U, contact, falling behind), LATERAL, TACTICAL, STRATEGIC by the instrument of card ADD66; names `HANDSRC10K-<M>-VS-OWN-<FAMILY>-BETTER|WORSE|MIXED|NOT-SEPARATED` and `HANDSRC10K-<M>-VS-OWN-U-LOWER|HIGHER|NOT-SEPARATED` by the rules of card ADD71. If that reader cannot be run on these runs, the row says for each family what is read and what is missing, with the reason; the four cell names above stand by themselves.

## 4. Controls (a control that does not pass: no name is written, the fault is logged before a repair)

* Pod check of the wrapper before the queue (`integ10k/check_handover_src.py`): with `own` the new wrapper returns the dumps of the parent (largest difference 0.0); with `logged` the tick-0 plan equals the fresh plan (1e-6) and the first replan segment, in the own frame at t0, equals the fresh track (1e-3 m); the two partial modes differ from `own` in their own column group only; `--t0-state logged` without a pre-roll is refused. If one of these does not pass, no rollout of the card starts.
* c0, on all 1,271 windows and both seeds, from the dumps: under `logged` the first replan segment of the aged track equals the moved fresh track D (largest difference at most 1e-3 m). This checks the wrapper on the full grid and the premise of the split of card ADD71 (the plan a fresh state gives, executed from the own pose, is the rigidly moved fresh track).
* c1: the pre-roll offsets of the three new runs equal those of the banked own run (largest difference 0.0).
* c2: each run's pre-roll log names its mode; in the log line of the first tick the group that is not replaced differs by exactly 0 from the own batch (speed for `path_logged`; positions and headings for `speed_logged`) and the replaced group differs.
* Split tool: b1 to b3, b5, b6, the identity POSE plus STATE equals total (1e-9) for the own run, the banked total of the pair AGE-K of the re-read of card ADD70 returned for the own run (1e-9), the planted start offset of note N4 of card ADD70; and h1: on planted arrays the name rule returns CARRIES for a mode that removes 0.8 of the state part, PART for 0.2, NOT-SEPARATED for 0, WORSE for a mode that adds, NO-STATE-COST for a state part of 0.

## 5. What follows, fixed now

* LOGGED CARRIES on TRACK: the state part of the cost of age enters at the hand-over of the state at one tick. Then the partial modes say which group: SPEED-LOGGED CARRIES names the speed group, PATH-LOGGED CARRIES the path group; both PART or the two not adding up to LOGGED names the route channel or an interaction, reported as such. The group that carries the cost is what an arm trained on its own rolled history has to be made robust to, and it is handed to the other session for its card; an inference-side arm on the hand-over of that group may then be opened as its own card.
* LOGGED PART or NOT-SEPARATED on TRACK: the state part is not the first tick's hand-over; it enters at the later ticks, which read what was executed inside the window from the own pose, or it is not separable by this arm. That is reported as written, and no hand-over arm follows from this card.
* A mode WORSE on a cell: handing that group from the logged state while the rest is own makes the plan worse there (a mismatch between groups); reported as written.
* Whatever the names: nothing here is a deployable readout.

## 6. Limits stated now

* Two model seeds are not a replicate study; a separated interval answers another draw of episodes and is necessary, not sufficient. The readout does not sample.
* The partial modes recompute the curvature column from one group's yaw rate and the other group's speed; the residual of that recomputation against the stored columns is logged and reported.
* In the two partial modes the route channel stays own; only `logged` replaces it.
* One setting (readout K, no rule, held route), one age (6 s).
