# Brief: the NuRec scoring-alignment audit, then the stage-2 package (AGENT F; CPU only; do NOT launch)

**Rules.**
* Everything in `BRIEF_T2V8F_STAGE1_E.md` binds: the preamble, CPU only, no ssh, never commit or stage, never delete, LF, banned
  words, ≤ 150k context with a `HANDOFF_F.md` at about 130k.
* The card `t2_prereg/PREREG_T2_CRAWL.md` is FROZEN (`2c4c6557`, `FROZEN_SHA.txt`). ⛔ Never edit it. A change is a dated
  amendment file beside it, written only by the Master Mind; propose one in your hand-back if needed.
* Stage-1 rollouts are PLUMBING ONLY: use them to test the instrument, never to read H1 or H2.

**Read first:**
* `../t2_stage1_bank_pkg/STAGE1_READ.md`: gates, render gap, and the open item.
* `stage1/pull_box2/PLUMBING_SCORE.json` and `stage1/pull_box2/pull/plumbing/`.
* `HANDOFF_E.md` and `STAGE1_LAUNCH.md`.

## Item 1 (priority): why does P10 read about 36° heading MAE on the two stage-1 starts?
Every arm on both starts reads heading MAE about 30 to 45°, yaw-rate MAE about 110 to 190 °/s and 6 s ADE 4.6 to 7.8 m. That includes
P10, which never reads the camera. Find out which of these it is:
* **(a) Real difficulty.** The starts are near standstill, where heading is ill-defined. On T1-kin eval-141, find P10's rows on
  near-standstill windows: the T1 reader's own output, from banked dumps, not recomputed. Do these starts read the same?
* **(b) A scorer or alignment defect on NuRec scenes:**
  * the logged-future alignment (rig trajectory timestamps, 10 Hz against the 30 Hz camera, the f0 offset);
  * the frame convention (world against rig, yaw sign);
  * a heading computed on standstill steps that the T1 reader masks.

How to discriminate:
* **Per-start, per-knot breakdown.** Feed the scorer the LOGGED track as the "plan"; every row must read about 0. A playback of the
  logged plan should score about 0 against itself. If it does not, that is the defect.
* **Cross-check.** Compute the same rows with the T1 reader's path on the same logged track.
* **Mutation.** Shift the logged future by one tick; the self-score must leave about 0.

Deliver `AUDIT_NUREC_SCORE.md` with the verdict (a) or (b) and the evidence.
* If (b): fix it at source with a test, plus a mutation that goes RED, and re-score the stage-1 pull. Every existing test must stay
  green.

## Item 2 (only if item 1 reads (a), or (b) is fixed and pinned): the stage-2 package
* **Scenes.** The 42 launch scenes from `t2_scenes/raw/SCREEN_TABLE.json`, with starts per card section 3 and its 6b clamp.
  * Exact bytes come from `t2_scenes/raw/DOWNLOAD_MANIFEST.json` (`files`). Disk copies are on `F:/TanitAD-scratch/t2_scenes/` and
    `C:/Users/Admin/TanitAD-scratch/t2_scenes/`; verify by hash.
* **Box 2 disk.** D: read 35.58 GB free after stage 1, and the rule keeps ≥ 10 GB free. If the 42 scenes plus the package do not fit
  with that margin, plan two batches and NAME the blocker: freeing batch-1 copies on box 2 is a deletion, which is the PI's decision.
* **Jobs, one process:** preflight (as stage 1), then the deciding rollouts. That means C0_s0/_s1, P10, ZERO and BLIND under T2 and RR,
  plus DAGOACC_s0/_s1 beside.
* **Build, rehearse, document.**
  * Build with the stage-1 builder pattern.
  * Rehearse on CPU with the stub renderer and a planted missing scene.
  * Write `STAGE2_LAUNCH.md`: commands, sizes, ESTIMATED wall time from stage 1's MEASURED 345 s for 28 cells, and the pull and score
    steps.
* ⛔ Do not launch.

## Hand-back (≤ 300 words)
The audit verdict with its evidence; any fix with tests and mutations (n passed); the stage-2 tar path, size, sha256 and box-2
footprint; any batch plan and blocker; open items; and a deliverable manifest.
