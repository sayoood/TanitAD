# PREREG RI-C0: the base under reindexed frames (zero training; written 2026-11-05 before any RI rollout exists)

**Question.** T1-kin feeds the logged frames on the human's clock. How much of the C0 base's T1-kin behaviour, and of its M1
crawl excess against the kinematic floor P10, depends on that clock? `--scene reindexed` (`motion_c/ri/`, DESIGN_C.md section 3)
shows each row after t0 the logged frame where the human was at the model's OWN along-track position (never a later frame).

**Runs.** RIC0_s{0,1} (C0 checkpoints `v8f-recover/M2b_s{0,1}_recft_C0/ckpt_last.pt`, reader flags, `--scene reindexed`) and
RIC0C_s{0,1} (the same with `--feasible kbox`). Reused, read only: C0_s*, C0C_s*, FZC0_s* arrays of the RECOVER-NEXT reader
(`recover_next/read/raw/arrays/`). Tier T1-kin (self-action open loop on logged cameras; reindexed is a longitudinal-only
approximate re-perception, not T2). Estimator: paired episode-cluster bootstrap, B = 2000, seed 0 (gapdecomp.decompose, the
RECOVER-NEXT reader's functions loaded by file path); it answers another draw of EPISODES only.

**Readings (all 19 rows reported, four families, STRATEGIC included).**
1. Frame-clock effect: RIC0 − C0 (19 rows) per seed.
2. M1 excess under RI: (RIC0 − P10)[M1] against (C0 − P10)[M1]; ratio r = RI excess / logged excess per seed.
3. Dominance under RI: Read A of RIC0C − P10 (D1, D2, D3), beside the logged C0C − P10.

**Committed outcomes (both written before data).**
* **CLOCK-SENSITIVE**: RIC0 − C0 separated on M1 crawl share at both seeds. Then logged-frame T1-kin numbers of frozen-feature
  arms carry a camera-clock effect, every (c) / (a) arm is decided on RI, and the RI C0 / C0C rows become their reference.
* **CLOCK-INSENSITIVE**: M1 not separated at either seed and |r − 1| < 0.25 at both. Then the base does not use the frame clock
  (consistent with frozen features carrying no motion, §7.31), and logged-frame reads of frozen-feature arms stand; (c) and (a)
  are still decided on RI, because their levers can read the clock.
* Otherwise **MIXED**, reported per seed.
No training, no box 2, no eval-141 data shapes any training: this read decides the read protocol only.
