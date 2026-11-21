# Brief: the T2 entry script and the stage-1 box-2 package (AGENT C; CPU build and rehearsal; do NOT launch)

**Rules.** Everything in `BRIEF_T2V8F_E3E6.md` (same folder) binds unchanged: the preamble, CPU only, no box 2, never commit, never
delete, LF line endings, banned words, ≤ 150k context with a handoff at about 130k.

**Exception: one download only.** You may download the two mp4 files named in item 3, using the T2 scene agent's download code and
rules: the HF token read in place and never printed, `HF_HUB_DISABLE_XET=1`, release `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`
pinned at commit `5fe7a149…`. Nothing else may be downloaded.

**Read first:**
* `HANDOFF_A.md`, its design for `t2v8f_run.py` and the stage-1 steps;
* `HANDOFF_E4E6.md`;
* the card `motion_c/t2_prereg/PREREG_T2_CRAWL.md`, sections 3, 4 and 6b (the start clamp, the guard starts, BLIND, the mp4 files).

**Another agent (B) is editing at the same time:** `code/t2v8f_score.py`, `code/t2v8f_lead.py`, and `code/t2a_v8f_policy.py`
(additively). Import these files and never edit them.

**You own:**
* `code/t2v8f_run.py`
* `code/t2v8f_kit.py` (only `starts_for_scene` and anything the entry needs)
* `stage1/`
* `STAGE1_LAUNCH.md`
* `tests/test_t2v8f_run.py`
* `PINS_C.txt`
* `HANDOFF_C.md`

## Items
1. **Starts per card 6b.** For a launch scene, a start is max(t_c − Δ, f_min) for Δ ∈ {4, 2} s, and two starts on the same frame
   collapse to one. A guard scene gets two starts from `default_rng(1)`. Test the count and the clamp, with a mutation.
2. **`t2v8f_run.py`.** Follow the `t2a_run.py` pattern in `HANDOFF_A.md`: a dispatcher for the v8F arms in `cd.run_rollout`.
   * It covers the conditions T2, RR and logged, and the agent modes `objects` and `empty`.
   * It writes per-rollout outputs in the scorer's input format, and then a scoring call (`t2v8f_score.py`).
   * Rehearse it on CPU with the stub renderer on 2 scenes: every arm, both conditions, end to end through the scorer.
3. **Stage-1 package (`stage1/`).** Code plus the checkpoints C0_s0/_s1 and DAGOACC_s0/_s1, and the two mp4 files of scenes
   `00040136` and `7c72937c` (`camera_front_wide_120fov.mp4`, about 38 MB each).
   * It runs on box 2 against the 2 scenes and the renderer of the T2-A kit already there, and DINOv3-L at `D:\tanitad\hf`. Its first
     step is a preflight that checks all of these and refuses clearly if one is missing.
   * The GPU gates: T1's render gap (rendered frame at the logged pose against the real mp4 frame, pooled-token cosine) and T5
     determinism (one C0 start rendered twice).
   * The actor falsifier: `objects` draws the lead in 7c72937c, and a planted no-lead read is absent.
   * A plumbing run of every arm under T2 and RR. These rollouts are plumbing only, and the outputs say so.
   * Results go to a pull folder, with `STAGE1.done` / `STAGE1.fail` markers.
   * Safe mode: one job, no other process.
   * `STAGE1_LAUNCH.md` gives the exact copy and launch steps (memory `box2-second-computer.md`: launch via `launch.ps1`, `D:\tanitad`
     layout, `cmd &` and backslash quirks) and the package byte size.
   * Rehearse the package's control flow on CPU with the stub renderer, with the box-2 paths substituted.

## Hand-back (≤ 300 words)
Tests and rehearsal results with numbers, mutation count, the package contents and size, the exact launch steps, open items, blockers,
and a deliverable manifest.
