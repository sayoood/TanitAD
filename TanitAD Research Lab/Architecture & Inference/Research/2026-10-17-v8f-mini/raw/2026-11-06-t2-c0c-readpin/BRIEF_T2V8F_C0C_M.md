# Brief: finish the composite arm and stage 4 (AGENT M, continuing AGENT L; dev-box CPU; no download; do NOT launch)

**Read first:**
* `t2v8f/HANDOFF_L.md`: state, exact dump paths, the gate design, next steps.
* `t2v8f/BRIEF_T2V8F_C0C_L.md`: the whole task. Its rules bind you unchanged.
* The frozen card `motion_c/t2_prereg/PREREG_T2_DOMINANCE.md`: sections 3, 4 and 6, and the 17:42 log entry in section 8. ⛔ Never edit the
  card.

## Done by L (MEASURED, do not redo)
* `code/t2v8f_c0c.py`: the C0C arm, registered by `install(R)`, with the RI tool loaded by path and sha-checked.
* `code/read_t2_dominance.py`: sha256 `df4e8636...efd24`, 12 tests in `tests/test_t2v8f_dominance.py`, G4 real.

## Items, in priority order (bank each before the next)
1. **Gate C0C-ID.** As designed in `HANDOFF_L.md` item 2: the tick-0 identity against the banked composite dump, the RI tool's own
   `model_planner(..., smooth="sg72", feasible="kbox")` as a second reference, the three RED arms, and off+off reproducing C0 exactly. Write
   `C0C_ID_RESULT.json` in the schema the read script takes (`--g1`).
2. **The read script, G2 per the card's 17:42 reading** (section 8, readings 1 to 3). Add parts (a) to (e) of G2 to
   `read_t2_dominance.py`:
   * (a) stage 1's `STAGE1_RESULT.json` `overall` == ALL_GATES_PASS. It lives at `stage1/pull_box2/pull/STAGE1_RESULT.json`. Check the cell
     code by package hashes: the stage-1 manifest's sha256 of the rollout code must equal what stages 2, 3 and 4 ran, or the gate fails and
     names the file.
   * (b) T0 and T4: the dev-box gate tests (`tests/test_t2v8f_gates.py`) are named and their last green log is cited.
   * (c) each stage's `gate` PASS.
   * (d) ZERO(T2) == P10(T2) on every window, exactly, recomputed from the stage-2 and stage-3 records.
   * (e) every deciding cell's `actor_attach.json` present, and the verdict tabulated per scene.

   Each part gets its RED arm in the tests. Keep L's 12 tests green. Then re-pin: write the new sha256 to `code/read_t2_dominance.py.sha256.txt`
   and name it in the hand-back; the Master Mind appends it to the card log. ⛔ No real rollout data from stage 2, 3 or 4 is read by this script
   or by you. Real pulls do not exist on the dev box yet; if one appears, do not open it.
3. **The stage-4 package** (L's brief item 3). Ship the RI tool as a pinned byte copy (76,704 B), as L found it is not on box 2. The scenes are
   already on box 2: `D:\t2v8f\scenes2` (42) and `C:\t2v8f\scenes3` (20, after stage 3 runs). Pull to `D:\t2v8f\pull4`.
4. **Rehearsal and mutations** (L's brief item 4).
5. **`STAGE4_LAUNCH.md`** (L's brief item 5). The waiter uses EXACT marker names (`Test-Path`, `grep -x`).
6. **Close-out.** The full suite runs detached through a finisher that writes `logs/M_FINISHED.txt`, ending in the fixed line
   `ZZ M-CHAIN-DONE ZZ`. Then write `PINS_M.txt`.

Hand off at about 130k context: write `HANDOFF_M.md` and hand back. The hand-back is ≤ 300 words, with a deliverable manifest.
