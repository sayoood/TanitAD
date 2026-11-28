# Card ADD110 (FALLATTR10K), note N1: the start line of the queue and the driver of the read

Written by the lead on 2026-11-22 after the queue of the card started (pod clock 08:24:43Z) and before any rollout of the card finished. No value of the card exists at the time of this note; no control, bar, name or measure of the card is changed by it.

## 1. What was found

The queue of the gap kit (`fallattr/gap/pod_fallattr10k_queue.sh` dc6f6c6405092fe3) writes the first line of its start marker `FALLATTR10K_START.txt` as: seconds since the epoch, an ISO time, the process id. The driver of the read kit (`fallattr/read/podread_fallattr10k.sh` cb98f2223f6c31de) takes the first token of that line and hands it to `date -d` to get the queue start for control a4. `date -d` refuses a bare number of seconds, so the driver as built would stop with `START-MARKER-UNREADABLE` before the reader runs. The two kits were built by two helpers and each kit's tests pass; the mismatch is between them and shows only with the real marker. It was found by reading the real start line right after the queue start.

## 2. What is changed

One line of the driver, in a new file `fallattr/read/podread_fallattr10k_r1.sh` (5d3865fbbac86f26), written by `fallattr/make_podread_r1.py` (56516b8e8f9d7ab9), which asserts that the old line is there once and that every other line is equal. The new line takes the first token as the queue start when it is a number and stops otherwise. Checked by the lead: `bash -n` passes; the file differs from the built driver in line 45 only; the real start line gives 1795335883 and a line that starts with an ISO time is refused.

Not changed: the reader `read_fallattr10k.py` (84ca638404d482c9), in which every control and every value of the card is computed; it receives the queue start as an argument and accepts a number of seconds. The queue, the gap files, the tests of both kits and both test outputs of the lead are not touched. The built driver stays in the kit as it is.

## 3. Control a4

As coded, a4 compares the file times of the lead's two test outputs on the pod with the queue start and reads their last lines; that stays as it is, with the true queue start.

By the letter, a4 says that the tests of every kit of this card pass in the lead's run on the dev box before the first rollout starts. That held for both kits as they were then (lead runs logged in `TIMES_NOTE.md` at 08:17:35Z: 25 passed, 54 passed). The one driver line of this note was changed after the first rollout had started, and the test file of the read kit does not run the changed driver (it reads the built driver's text). So for the driver of the read, the time clause holds for the file as built and not for the changed line. The row reports a4 with both readings beside it, and the other session reads the row before it is landed (fork rule of row 136).

## 4. Rehearsal before the queue start (for the record)

Before the queue start the lead ran the reader's controls stage on the pod with the four arm names pointed at banked, already open rollouts of card ADD106 (`fallattr/rehearse_reader.py`, `fallattr/rehearse_run.sh`): a0, a1, a3, a4, a5 passed, a2 failed as it has to (other gap files), and the levels equal the banked levels of card ADD106 with a largest difference of 0.0. After the queue start and before any rollout finished, the names stage was run once on that stand-in payload (banked rollouts only): it ran through and gave 14 name slots. The stand-in names say nothing about this card and are not kept.

## 5. Time of the queue

The card estimated 1.2 to 1.6 pod-hours with two rollouts at a time. The queue as built runs the eight rollouts one after the other (about 2.4 hours by the times of card ADD106), inside the stop rule of 3 pod-hours from the queue start, which is not changed.
