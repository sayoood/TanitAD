# Note N1 to card ADD77 (CLOSEREAD): the run named in control c3, and the episode unit

Written before any tool of the card has run on the pod and before any dump exists. No bar, no arm, no name and no form of a control changes.

## 1. What was found

Review of the kit against the pod (file listings and array names only, no value of any read):

1. Control c3 of the card names the fresh run `I10RQ50KHb20`. No run of that name exists on the pod. The fresh runs of form RR with the q50 map and the heading bound are `I10RQ50Hb20` (pinned readout, rule at replan ticks) and `I10RQ50KF5Hb20` (readout K, second look every five ticks). The card took the name from the list of banked runs of card ADD76 and dropped the `F5`.
2. The card counts score frames "in at least 20 episodes" (control c7) and its intervals resample evaluation episodes. The fit tool takes the episode of a clip from a map file; without the file it takes each clip as its own episode. Measured with `make_episode_map.py` on the banked dump folder (index columns only): 141 clips, 141 episode ids, one per clip. The episode unit of the readers of rows 126 to 134 is therefore the clip, and the map says the same as the tool's fallback; the map is still written and given, so the unit is read from the banked dump and not assumed. (The first form of this note, frozen at 13:08:03Z with hash 0131c9daa75a3c92, said the clips belong to fewer episodes. That was not measured and is not so. Nothing had run in between.)

## 2. What is fixed here

1. Control c3 reads the rule log of `I10RQ50KF5Hb20` of both seeds (file `ticks/rule_log_I10RQ50KF5Hb20_s<k>.npz`, the rows of the replan ticks; the rows of the second look are in another file and are not read). The comparison, the rows and the bar of c3 stay as the card has them: presence and gap read after the q50 map at the same clip and frame, largest difference at most 1e-3, on every row with the window in the clip, with rows of the first replan tick among them. Why this run stands for the named one: the reader input of the rule at a tick is a function of the cached tokens of the logged frames and of the reader alone; it does not depend on the readout or on a second look. `I10RQ50Hb20` would serve the same way.
2. Before the dump, `make_episode_map.py` checks that the rule log of each seed records the hash of the reader file and of the map file that the dump and the fit tool are given, and that its form is RR. A difference is a refusal.
3. The episode of a clip is the episode id of its windows in the episode files of the banked dump folder of `I10RQ50KF5Hb20_s0` (the id that the readers of rows 126 to 134 resample). `make_episode_map.py` writes the map and refuses when a clip has two ids, when a clip of the join has no window, or when the clip order of the join differs from the order of the pinned grid. The run script refuses to fit without the map.

## 3. The target, as the source has it (added before any run)

Section 2 of the card takes the labels from "the label source of the study wrapper". Read in source (`v8f_t1_eval_rule.py`, class `LabelSource`, method `read`): its closing speed is the least-squares slope of the label gap over the trailing run at the frame (at most five frames, steps of at most 3.0 m, at least three frames; 0.0 for a shorter run). It is not a relative speed of the track taken from the lead file, as the sentence on limits in section 7 of the card says. The tool follows section 2: the target is the closing speed that the label-fed rule of row 134 was given. Two consequences, stated before any value: (a) control c2 compares two readings of the same label gap over the same five frames, so it fixes sign, unit and frame clock and is expected to pass by a wide margin; it is not a comparison of two independent quantities; (b) a cell that reads READS-CLOSING reads a one-second slope of the gap, which is what the rule consumes.

Interpretations of the kit, confirmed against the card before the run: control c4 is run at history 25 (the card names no history); the estimate of arm N is the mean of its three nets (the card lists seeds 0, 1, 2 and one name per arm and history); the pooled r is taken over the rows of both reader seeds together.

## 4. What this note does not do

It does not touch the form of any control. If c3 does not pass on the run of this note, the card is reported with c3 failed; the read is not repeated against another run to make it pass.
