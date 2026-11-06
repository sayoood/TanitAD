# Brief: T2 kit for v8F, the rest of E4 to E6 and the stage-1 package (continuation; CPU build, do NOT launch)

**Rules.** Everything in `BRIEF_T2V8F_E3E6.md` (same folder) binds unchanged: the preamble, CPU only, no network, no box 2, never
commit, never delete, LF line endings, banned words, ≤ 150k context with a handoff at about 130k.

**Start from.** `HANDOFF_E3E6.md` (same folder), section "LEFT", in its order. E1 to E3 are built and gated, and the Master Mind re-ran
9/9 tests PASS (`tests/`).

## The four items, with three rulings from the Master Mind
1. **E4.** Register the arms in `EXTRA_ARMS`. Nav must be equivalent to the T1 tool's: use the T1 tool's own nav construction on the
   scene's logged route, not the kit's `nav_from_route`. Test it on a banked window. Also the BLIND mean, and the starts from
   `motion_c/t2_scenes/raw/SCREEN_TABLE.json`, which now exists with 42 launch and 20 guard scenes.
   * **Ruling, the DAG-OACC base:** use the checkpoint that C0's own `argv.json` names as its init
     (`E:/Projects/TanitAD-artifacts/v8f-recover/M2b_s{s}_recft_C0/argv.json`). If it is absent on this disk, name the path and mark
     DAG-OACC "beside, not built". It is a beside arm.
2. **E5, the scorer.** Reuse the reader's own functions for every row they compute.
   * **Ruling, the 8 rows without inputs.** Do not drop them; build them where the scene supplies the input. Each must equal the reader's
     function on a banked window, where a pin is possible.
     * **DK-VIOL, DK-HW, DK-TG:** build the lead block from the scene's `sequence_tracks.json` with the same definition as the pod-side
       join (`stack/scripts/build_obstacle_join.py`, grep the tip): lead = nearest in-lane vehicle ahead.
     * **S-POS@30/50/80:** build them from the logged rig trajectory (the human's route at 30 / 50 / 80 m of arc), exactly as the reader
       builds them from the eval join. A start with less than 80 m of logged route ahead gets NaN at that horizon, with its n counted.
     * **TAC-LAT and TAC-LON need the v9 tactical labels.** If the offline label builder runs on CPU from the scene's ego log and tracks
       (grep the tip for the v9 builder), build them. Otherwise report the TACTICAL family as NOT COMPUTED, with the reason and the n,
       as a work item.
3. **Gates T3 and T4**, each with a mutation that must go RED.
   * **T3:** ZERO equals P10 exactly, and P10 attributed against itself reads exactly 0.
   * **T4:** the scorer reproduces the T1 reader's M1 share and every row it shares with the reader, bit for bit, on the banked RIC0_s0
     dump.
4. **The stage-1 box-2 package and its CPU rehearsal with the stub renderer.** As in `BRIEF_T2V8F_E3E6.md` item 5, with the job's exact
   launch steps written into `STAGE1_LAUNCH.md`.
   * **Ruling, box-2 disk:** D: has 45.4 GB free. Stage 1 uses only the 2 scenes already on box 2 (from the T2-A kit), so the package
     ships code and checkpoints, not scenes. State its byte size.

## Hand-back (≤ 300 words)
Gate results with numbers, mutation count, the rows computed per family and the ones NOT COMPUTED with reasons, the stage-1 package
contents, size and launch steps, open items, blockers, and a deliverable manifest.
