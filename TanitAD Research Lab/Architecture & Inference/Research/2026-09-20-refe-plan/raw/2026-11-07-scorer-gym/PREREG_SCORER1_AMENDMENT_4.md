# AMENDMENT 4 to PREREG_SCORER1 (2026-11-08, Master Mind): the lane-label condition for S1L

**Written before any arm trained and before Stage 0 was read.** At this moment Stage 0's dump is still running and no `stage0.json` exists. No bar, estimator, arm definition or endpoint changes.

## 1. Why
Amendment 3 condition 3 asked that u400 be shown to have trained on the banked SFT-4 lane labels. That cannot be shown (`raw/2026-11-08-s1l-overlay/provenance.md`): u400 trained on `train_teacher_lane` (2,109 sets, INHERITED from `PREREG_SFT4` Amendment 4), and that directory was lost with the pod and never banked. S1L would train on a regenerated draw of the same recipe (seed 20261004).

The PI's words, 2026-11-08, on the replacement the Master Mind proposed: *"yes accept the replacement condition for the lane arm"*.

## 2. The replacement
Amendment 3 condition 3 and the "Labels" condition of Amendment 2 are replaced, for S1L only, by:

> **C0, C0r, S1 and S1L all train on the same regenerated lane draw, and the result says so.**

Made checkable before launch:
1. All arms read one `--teacher-lane-train` directory.
2. Its sha256 manifest (file names and hashes) is written before the first arm and re-checked at the start of every arm. A row count alone is not enough: `scorer1_chain.sh:144` checks only the count, so this check must be added with the S1L landing.
3. The result states: u400, the shared warm start, trained on a different draw of the same recipe that no longer exists; the lane comparison is between arms on the new draw; nothing is claimed about continuity with u400's own lane training.

## 3. What is unchanged
S1L still starts only if:
1. The lane-truth control relabelled on the Linux machine class after P7 ends passes **G-L1 as written** (99 % of proposals within 1 mm). If it fails, S1L does not run, the reason is reported, and the lane read stays CONDITIONAL.
2. That same pipeline supplies the E2 lane truth.
3. The S1L code and gate work is landed and the launch gate passes, including the real trainer preflight on the L40 (gate G3 read 0.125 on the dev box for u400's native 7-row head and must be read there).

The added compute is about 3.0 L40-hours (ESTIMATED, Amendment 3).
