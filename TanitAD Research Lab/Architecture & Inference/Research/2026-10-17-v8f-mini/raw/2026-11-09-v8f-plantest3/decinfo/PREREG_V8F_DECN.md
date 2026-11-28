# PREREG_V8F_DECN: is the weak speed and yaw-rate readout limited by the 53 fit episodes? (Master Mind, 2026-11-09, written before any DECN code or run)

## Why
`PREREG_V8F_DECINFO.md` read speed as NOT-READ-FROM-VISION at 53 fit episodes: every stage of the vision chain stays at or below R2 0.498, against a bar of 0.70 (`VERDICT_DECINFO.md`). Its committed trigger is the same probe at full n on box 2 after the ladder. PLANTEST3's decoder was fitted on those same 53 episodes, so a sample-limited readout would also limit PLANTEST3's NO-GAIN. This card reads the dependence on n now, without waiting for box 2: train clips are streamed read-only (sftp, one clip at a time, the `collect_latents_train.py` pattern), adding no GPU or RAM load on box 2, and every computation is on the dev box's CPU.

## Objects
* Held-out set, unchanged: the PLANTEST3 TUNE inner split's 17 held-out episodes, 238 windows, 16 frames per window. No SCORE window is read.
* Fit sets (one change: the number of fit episodes):
  | N | episodes |
  |---|---|
  | **N53** | the 53 TUNE fit episodes (the reference, reproduced) |
  | **N194** | N53 plus the 141 train clips of `wmact/train_stream/clips141.json` |
  | **N1053** | N194 plus 859 further train clips |
  N1053's further clips are selected once from the train manifest by the lowest sha256 of (clip id + salt `decn-v1`), excluding every eval clip. Each train clip contributes 17 windows of 16 frames, as `collect_latents_train.py`.
* Inputs: R0 (M2b z1, s0 and s1), R2 (pooled adapter token, s0 and s1), R1 (ladder z1, every checkpoint banked at the time of the run; those finished after this card are marked as added).
* Targets: speed and yaw rate as DECINFO. On train clips, speed comes from the clip's logged speed; control 1 ties it to the eval target.

## Readers
Those of DECINFO for vector rungs: ridge (1e-3 times the sample count) and the MLP of `diag_mlp.py` with early stopping on 20 % of the fit episodes, at one frame and at four frames. Grid rungs are not read here: they were not above the pooled token in DECINFO, and their storage scales badly. Details in `theory/decinfo/decn/IMPL_CHOICES_DECN.md`, with its sha256, before any held-out row is read.

## Read (T0, no interval; pooled held-out R2; differences under 0.05 carry no weight)
Per input, N and target: the best held-out R2 over the readers (optimistic, stated as such). Deciding input set: R0 and R2 at both M2b seeds. Deciding N: N1053. Bars: speed 0.70, yaw rate 0.50. Rise = R2 at N1053 minus R2 at N53, on the same input and reader. LONGITUDINAL, LATERAL, TACTICAL: no metric here (a readout R2 only); STRATEGIC: PAUSED (PI item 25).

## Committed outcomes (per target)
* **N-LIMITED:** R0 or R2 clears the bar at N1053 at both seeds. The small fit was the limit.
* **RISES:** below the bar at N1053, and the rise is at least 0.10 at both seeds on the same input.
* **FLAT:** the rise is below 0.10 at either seed on every deciding input.

## What each outcome triggers (committed now, Rule Zero)
* **N-LIMITED:** the decoder is refitted on N1053 and PLANTEST3 is rerun on FLAT1 DEC-END (plain and shrink), SCORE windows, both M2b seeds. Its NO-GAIN at N53 is reported with this limit stated.
* **RISES:** the full 4,572-clip probe on box 2 (the DECINFO trigger), as one job after the ladder, before any conclusion about the vision chain.
* **FLAT:** more data does not lift the read at this scale. The measured-speed world-model arm (`theory/egowm/`) goes first, and the full-n probe is dropped from the box-2 queue unless the PI asks for it.

## Controls (the read is INVALID if one fails)
1. **Target tie:** on the 982 TUNE eval windows, the logged-speed source used for train clips and the eval target agree: R2 at or above 0.99 and mean absolute difference at or below 0.10 m/s.
2. **Reproduction:** at N53, R0 ridge at one frame reproduces DECINFO's value within 0.01 at both seeds, for speed and yaw rate.
3. **Shuffled targets:** whole target tracks permuted across fit episodes read held-out R2 at or below 0.05, at every N, input and reader.
4. **Planted signal:** a planted linear target is recovered at held-out R2 at or above 0.80 at N1053; zeroed features read at or below 0.02.
5. **Split:** no eval clip is in the train additions, no SCORE window is read, and the held-out set is identical across N (asserted; a mutation goes red).
6. **Clip integrity:** each pulled clip's md5 matches box 2's manifest; a mismatch stops the run.

## Log
* 2026-11-09 07:36:50 frozen: prefix 4470 bytes, sha256 cfa5bb1fdda36a77efd6bf203232dbf5cf0d15f08a2189ad908e2436509619e5 (everything above this section). No DECN code or run existed at this time.
* 2026-11-09 amendment 1, chosen after seeing control 3 fail and the first-run values. Control 3 and its bar (0.05) do not move. A reader row whose shuffled control reads above 0.05 leaves the deciding maximum, in place of invalidating the read. 10 of 216 rows failed, all MLP; none is the maximum of a deciding cell, so no deciding value moves. Reason: the post-hoc diagnostic `decn/null_shuffled.json` (8 fresh permutations on each of the ten rows, 80 fits) reads mean -0.126, sd 0.125, maximum +0.160, 3 of 80 above 0.05, every row's mean below 0. The first run stays published as INVALID as run. Also: the full-n probe of the RISES trigger runs streamed read-only to the dev box on the deciding inputs (N4565), in place of a box-2 job after the ladder; its bars and readers are the card's.
