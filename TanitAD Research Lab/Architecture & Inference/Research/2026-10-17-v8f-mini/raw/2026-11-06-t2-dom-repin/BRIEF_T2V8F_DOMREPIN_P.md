# Brief: re-pin the dominance read with rule R-SCN (AGENT P; dev-box CPU; no download; do NOT launch)

Preamble: `Project Steering/AGENT_OPERATING_STANDARD.md` binds you. ⛔ Stage nothing, commit nothing, push nothing. End with a deliverable manifest.

**Read first:**
* `motion_c/t2_prereg/AMENDMENT_T2_SCENE_REFUSAL.md`: rule R-SCN.
  * The renderer refuses some scenes before any planner runs. Such a scene leaves the read for BOTH arms.
  * Rule 4: a scene excluded in ANY stage is out of every cross-stage pairing.
* `motion_c/t2_prereg/PREREG_T2_DOMINANCE.md`: section 4 (G3), and the log entry at 19:31 in section 8.
  * ⛔ Never edit any card in `t2_prereg/`.
* `code/read_t2_dominance.py`: pinned `6345d4c7...`.
* The stage-2 implementation of R-SCN, which is your template:
  * `code/scene_exclusions.py` (`83a5f647...`);
  * `code/read_stage2.py`, function `check_exclusions`;
  * `tests/test_scene_exclusions.py`.
* `HANDOFF_M.md` and `HANDOFF_O.md`: the read's contract and stage 4.
  * Stage 4's cells are T2 and RR × C0C_s0 and C0C_s1. Check the deciding/beside split in `code/t2v8f_stage4.py`.

## Items, in priority order (bank each in `HANDOFF_P.md` before the next)
1. **Add `--exclusions2`, `--exclusions3` and `--exclusions4` to `read_t2_dominance.py`.** Each takes an `EXCLUSIONS*.json` from
   `scene_exclusions.py`. G3 under R-SCN means all of the following, per stage:
   * the exclusions file is ATTRIBUTED, with no problems, and was built from THAT stage's result (`manifest_sha256` equal);
   * the stage's failed cells are exactly all the cells of its excluded scenes, deciding and beside;
   * every cell of every included scene is DONE;
   * the UNION of the excluded scenes over stages 2, 3 and 4 is removed from BOTH C0C and P10 before `pair_check` and before the rows
     are built;
   * `overall` INCOMPLETE is accepted only under these conditions.

   Without the three flags, the read behaves exactly as the current pin.
2. **`scene_exclusions.py` on stage-4 and stage-3 layouts.** It derives each scene's cells from the pull directories. Prove it on a
   synthetic stage-4 fixture (4 cells per scene) and a synthetic stage-3 fixture. Only if it needs a change: make it, keep its 16 tests
   green, and record its new sha256.
3. **G2(e).** An excluded scene has no `actor_attach.json`. Tabulate it as `EXCLUDED (R-SCN)` with its reason, never as a gate failure.
   The deciding cells of included scenes still need the file.
4. **Disclosure.** `DOMINANCE_T2.json` and `.md` list the excluded scenes per stage, with the reason, the cell counts and the starts
   lost.
5. **Tests.** Each must go RED:
   * a failed cell outside the excluded scenes;
   * an excluded scene with one DONE cell, or only part of its cells failed;
   * an UNATTRIBUTED file;
   * an exclusions file from another stage (its manifest sha);
   * any stage still pairing on an excluded scene;
   * a scene excluded in stage 4 only that is NOT removed from P10.

   Keep the dominance tests green (18, `tests/test_t2v8f_dominance*.py`), and O's stage-4 contract test too. If that contract test pins
   the read's sha, update ONLY that pin. Run the mutation check: delete each new guard and require a test to go RED.
6. **Re-pin.** Write the new sha256 to `code/read_t2_dominance.py.sha256.txt` and name it in the hand-back. The Master Mind appends it
   to the card log.
7. **Close-out.**
   * Run the full suite detached through a finisher: log `logs/pytest_full_P.log`, and `logs/P_FINISHED.txt` ending with the fixed line
     `ZZ P-CHAIN-DONE ZZ`.
   * ⚠️ C: has about 9.9 GiB free, so put pytest scratch on F:.
   * Write `PINS_P.txt`.

**Rules:**
* ⛔ No ssh, no box 2, no commit, stage or delete, no download, CPU only (`CUDA_VISIBLE_DEVICES=-1`), LF line endings.
* ⛔ No banned words in the docs: void, superseded, retracted, formerly, no longer, correction, withdrawn.
* ⛔ Do not open any real pull: `stage2/pull_box2_run2*`, `stage3/pull_box2`, `stage4/pull_box2`. Synthetic fixtures only.
* Hand off at about 130k context with `HANDOFF_P.md`.
* The hand-back is ≤ 300 words, with a deliverable manifest.
