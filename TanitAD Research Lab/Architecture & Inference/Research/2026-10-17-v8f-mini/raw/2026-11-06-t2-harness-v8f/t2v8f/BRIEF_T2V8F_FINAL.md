# Brief: the last T2 kit items, split across two agents with no file overlap (CPU build; do NOT launch)

**Rules.** Everything in `BRIEF_T2V8F_E3E6.md` (same folder) binds unchanged: the preamble, CPU only, no network, no box 2, never commit,
never delete, LF line endings, banned words, ≤ 150k context with a handoff at about 130k.

**Starting point.** `HANDOFF_E4E6.md`, sections FINDINGS and LEFT. The Master Mind re-ran `tests/`: 12/12 PASS. Every existing test must
still pass at the end.

**Two agents run at once.** Each owns ONLY the files listed for it. Each writes its own pins file and its own handoff. Neither edits a
file of the other or of E1 to E3, except as stated.

## AGENT A: E4 and the stage-1 package
**Owns:**
* `code/t2v8f_run.py` (new)
* `code/t2v8f_kit.py` (extend)
* `stage1/` (new)
* `STAGE1_LAUNCH.md`
* `tests/test_t2v8f_e4.py`
* `PINS_E4.txt`
* `HANDOFF_A.md`

**Tasks:**
1. **The entry script, `t2a_run`-style.** Register C0_s0/_s1, DAGOACC_s0/_s1, P10, ZERO and BLIND in `cd.EXTRA_ARMS`.
   * DAG-OACC comes from C0's `argv.json` init. If that checkpoint is absent, mark the arm "beside, not built".
   * BLIND uses the banked training-mean tokens, computed on CPU from the train-cache statistics, with their sha recorded.
   * Starts come from `motion_c/t2_scenes/raw/SCREEN_TABLE.json`: t_c − 4 s and t_c − 2 s for launch scenes, `default_rng(1)` for
     guards.
   * Conditions T2 / RR / logged, and `objects` / `empty`.
   * Results are written per rollout in the scorer's input format. Read it from `code/t2v8f_score.py`; do not edit that file.
2. **Nav.** Build the T1 tool's nav (the join column of `v8f_mini_data.load_join`, from the v9 rows) for a scene from its logged route,
   using the v9 builder's nav functions. Test equality on a banked eval window, and add a mutation.
   * Scenes outside the v9 release: state exactly how their nav is built, and test that the construction equals the v9 one on a banked
     clip.
3. **The stage-1 package** (`stage1/`, item 3 of LEFT): code plus checkpoints, no scenes.
   * It runs the GPU gates (render gap, T5 determinism, actor falsifier) and a plumbing run of every arm on the 2 scenes on box 2.
   * Results go to a pull folder, with done / fail markers.
   * Rehearse it end to end on CPU with the stub renderer.
   * `STAGE1_LAUNCH.md` gives the exact launch steps (see the memory files `box2-second-computer.md` and
     `devbox-drive-letter-and-detached-jobs.md`) and the byte size.

## AGENT B: the remaining E5 rows and the outcome logic
**Owns:**
* `code/t2v8f_score.py` (extend)
* `code/t2v8f_lead.py` (new)
* `tests/test_t2v8f_e5b.py`
* `PINS_E5B.txt`
* `HANDOFF_B.md`

**May change:** `code/t2a_v8f_policy.py`, ADDITIVELY ONLY, so that the policy also returns S's checkpoints per tick. Its existing outputs
and the G1 bit-for-bit tests must stay unchanged.

**Tasks:**
1. **DK rows on scenes.** Build a lead block from `sequence_tracks.json` with `taniteval/lead_source.lead_block`, the function that
   built the banked block (FINDINGS 2). Pin it: on a banked eval clip, feed the same tracks and check it reproduces the banked block bit
   for bit. Add a mutation.
2. **S-POS@30/50/80** (FINDINGS 3). This is S-head checkpoint error against the logged-route checkpoints, minus the CTRV baseline's error,
   exactly as the T1 heads reader computes it.
   * The logged-route checkpoints are built with the v9 builder's checkpoint functions from the scene's rig trajectory.
   * A start with less than the horizon's arc of logged route ahead gets NaN at that horizon, with its n counted.
   * Pin it against the reader on the banked dump.
3. **The bootstrap over scenes** (`eid` = scene index; `taniteval/ci.py`, B 2000, seed 0) for X1 and X2.
4. **A unit test of `outcome()`** against card section 6. Cover every branch with synthetic contrasts, and add a mutation per branch.
