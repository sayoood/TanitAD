# SLOWWM relaunch at the code-fix child (written, NOT executed; box 2 untouched)

Order: 1 land (Master Mind runs `land_slowwm_codefix.sh`) -> 2 fill the child sha -> 3 aside the ship marker -> 4 launch.sh -> 5 launch -> 6 read.

1. **Land.** `bash E:/Projects/TanitAD-artifacts/dagland/motion_c/theory/slowwm/codefix/land_slowwm_codefix.sh` (refuses unless origin and the local branch both equal f7e790ab, the pinned sha256 of both files match, and the tree differs from the parent in exactly 2 paths). Prints `CHILD_SHA=<40 hex>`. Verify yourself: `git -C C:/Users/Admin/tanitad-push ls-remote origin refs/heads/agent/v8f-slowwm-20261111` equals it.
2. **Record.** Replace `<CHILD_SHA>` in `codefix/AMEND_SLOWWM_3_CODE.md` (Write tool, new file or Edit), write `AMEND_SLOWWM_3_CODE.sha256`. Optionally run the suite step the STOPBAL amendment used (parent-side vs child-side wider suite) before launching; the diff is one model file, one new test.
3. **Move the ship done-marker aside (REQUIRED).** `done/ship.done` is keyed on the checker, not the commit: left in place the chain skips the ship and box 2 keeps the f7e790ab tree, and the pilot gate then refuses on `TIP_SLOWWM.txt`.
   ```
   mkdir -p C:/Users/Admin/slowwm_state/aside_codefix
   mv C:/Users/Admin/slowwm_state/done/ship.done C:/Users/Admin/slowwm_state/aside_codefix/ship.done.f7e790ab
   ```
   Leave `done/basis.done` and `basis/` in place (see COMMIT_BINDING.md rows 8-9 and the evidence). *Optional 2b, if a basis record stamped with the child is wanted:* also `mv done/basis.done aside_codefix/basis.done.f7e790ab` and `mv basis aside_codefix/basis.f7e790ab` (the basis stage re-runs on CPU, ~5 min on 11-11, needs 6 GB D: less existing basis bytes; afterwards compare `basis_sha256` of the new BASIS.json with the aside one: equal = identical basis). Never delete either.
   `STOP.txt` needs no move: `SLOWWM_RESUME=1` renames it itself to `STOP.<timestamp>.txt`. `runs/PILOT_SLOW_s0` holds only `argv.txt` and stage cmd files (no RUN.done); the gate rewrites it.
4. **launch.sh** (`C:/Users/Admin/slowwm_relaunch/launch.sh`): change one line, keep the rest (SLOWWM_RESUME=1, the STOPBAL dose-2 predecessor, which is DONE, so no wait):
   `export SLOWWM_COMMIT=<CHILD_SHA>` (was f7e790ab57e10a9004b04b206195a0ae5a5295b8). No `SLOWWM_AMENDMENT` yet (it is for the 72 h permit and must name `pilot_sha256=` and `commit=<CHILD_SHA>` once a pilot exists). Update the comment line to cite AMEND_SLOWWM_3.
5. **Launch** the way the previous relaunch was done (detached, `< NUL`, per the dev-box rule; not from this agent session). Expected `chain.log`: `ship was done...` is NOT printed (marker absent) -> `start ship` -> `done ship` -> `skip basis (done)` -> `start pilot`. Check `state/ship/<CHILD_SHA>/SHIP_RESULT.*.json` reads `ok: true`, `commit: <CHILD_SHA>`, and `moved: 1` (exactly one member differs from the f7e790ab tree: `stack/tanitad/models/v8f_mini.py`).
6. **Read the pilot gate.** It is the first GPU exercise of `built_basis` (and of whatever follows it in the gate): this fix is CPU-evidence only. If the gate fails again, read `D:\tanitad\slowwm\runs\PILOT_SLOW_s0\logs\gate.log` on box 2 for the next frame; a second device bug would be the same class (CPU-made tensor, GPU module) and the next patch is again a child of the child.

Nothing in the card, argvs, frozen files or package changes, so `PACKAGE_MANIFEST.sha256` stays valid. The README and `slowwm_common.sh` text mention f7e790ab; they are documentation strings, not checks (`slowwm_common.sh:73` message only).
