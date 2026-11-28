# PREREG_V8F_PROOF_ADD39_N1 : does the lead explain any of the plan's speed error? A read on existing rollouts before the next build

Note to card `PREREG_V8F_PROOF_ADD39.md`. Frozen after the reads of the card (row H-V8F-PROOF-74: `LEAD-INPUT-SCRATCH-UNUSED`, `-FLAT`, `-NO-COST`) and before the read of this note is written or run: no number of it exists. The card and its names stay as written. Tier: T0 for this read (the plan of the first tick of each window, made from the logged state on logged frames; a diagnostic, never driving performance). STRATEGIC: PAUSED (PI item 25).

## 1. Why

Section 6 of the card commits, for `LEAD-INPUT-SCRATCH-UNUSED`, another place of entry for the lead read (its own token for the plan heads). That is a build plus arms. Two facts make it doubtful that any entry can help: the lead read is computed from the same camera tokens the plan heads attend to, and the input layer of LCS learned weights of a visible size that change nothing. One read on dumps that exist can say whether there is anything for an entry to gain. It is run first; the committed build follows or is set aside by the rule of section 4, stated here before the read.

## 2. The read (arms LS_s0, LS_s1; no arm is trained)

* Windows: the lead windows of the headway read (label lead valid at the window start; 642 windows of 56 episodes expected; the count is printed). One row per window: the first tick, where the model starts from the logged state.
* Target y: the plan's along-track position at 2.0 s minus the logged along-track position at 2.0 s (m; above 0 = the plan is faster than the driver). Beside: the same at 4.0 s.
* Inputs, all known at the window start: B = ego speed at the start (and its square); L = label lead gap at the start, label closing speed over the last 1.0 s, and their products with the ego speed (time gap terms); V = the lead head's own read at the start frame from the camera (presence logit, gap, closing speed; arm's own head).
* Statistic: out-of-fold R2 (5 folds by episode, ridge with the penalty chosen on the training folds from 0.01, 0.1, 1, 10, 100 times the mean diagonal). GL = R2(B + L) minus R2(B). GV = R2(B + V) minus R2(B). Two-level bootstrap (seeds, then episodes), B 2,000, rng 0.

## 3. Names (both seeds needed; "above 0" = each seed above 0 and the lower bound of the two-level interval above 0)

* `PLAN-ERROR-LEAD-READABLE`: GL above 0 and GV above 0. The plan leaves speed error that the lead explains and that the model's own lead read also explains.
* `PLAN-ERROR-LEAD-NOT-READ`: GL above 0, GV not. The label lead explains error that the camera read does not.
* `PLAN-ERROR-NOT-LEAD`: GL not above 0.

## 4. What follows (committed before the read)

* `PLAN-ERROR-LEAD-READABLE`: an entry can gain something; the build of section 6 of the card (own token) is done next, with GV as the size to expect.
* `PLAN-ERROR-LEAD-NOT-READ`: the precision of the lead read is the limit; the build of section 6 is set aside with this reason, and the finer lead read stays the open decision of the PI (row H-V8F-PROOF-56).
* `PLAN-ERROR-NOT-LEAD`: at the window start the lead does not explain the plan's speed error; an entry for a lead read cannot improve the plan there, and the build of section 6 is set aside with this reason. Distance keeping then rests on what acts on the model's own gap: the rule at inference and planning by imagination (cards ADD36, ADD37).

## 5. Controls (all must pass; else nothing is named)

* d1 (known zero): with the rows of L (and of V) permuted across windows inside each seed, GL (GV) does not lie above 0.
* d2 (known positive): with y itself plus noise of the size of y's spread added to L as one more column, GL lies above 0.
* d3: the window count and the episodes equal those of the headway read of the same arms (printed); every window is used once.
* d4 (frame alignment of the camera read): on the same windows the head's gap read correlates with the label gap at the start frame more than at the frames 5 steps earlier and 5 steps later (both seeds), and the correlation at the start frame is above 0.3.
* d5: the plan and the logged track of the first tick are in one frame: at step 1 the logged along-track position divided by 0.2 s equals the start speed to within 0.5 m/s in at least 0.95 of the windows.

## 6. Limits

Linear fits on a few inputs: a lead effect of another form is not excluded by a flat read. The first tick only: the read says nothing about error that builds up after the model has left the logged track, which no camera input from logged frames can show. Two seeds, one corpus.
