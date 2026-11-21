# HANDOFF: EGOWM box-2 chain (deliverable 4) NOT STARTED; deliverables 1 to 3 are in `egowm/read/` (see its README_READ.md)

Stopped here for context budget, not for a block. Nothing in `egowm/box2/` exists yet, nothing was shipped, nothing touched box 2.

## Exact next step
Copy `theory/otband/box2/` (the closest template: it already waits for a predecessor, pilots, applies the 72 h permit, runs a list of arms, and carries the relative-import closure fix in `otband_ship_tree.py::package_of / imports_of`) to `theory/egowm/box2/`, then:
1. Rename every OTBAND/otband token to EGOWM/egowm (scripts, sentinels `ZZ EGOWM-STOP`, `ZZ EGOWM-DONE`, `ZZ EGOWM-ARM-REFUSED`, state dir `C:/Users/Admin/egowm_state`, env names `EGOWM_COMMIT/SSH/SFTP/STATE`).
2. Predecessor wait: replace the ladder wait in `egowm_chain.sh` (`ladder_ended`) by "`C:/Users/Admin/otband_state/DONE.txt` or `STOP.txt` exists and starts with `ZZ OTBAND-DONE` / `ZZ OTBAND-STOP`" and keep the `BOX2_BUSY.txt` exclusion.
3. `egowm_check.py`: NEEDLES = the strings of the patched trainer/model/data files (`--wm-speed`, `bad_wm_speed`, `built_speed`, `reflect_integrate`, `anchored_speeds`, `with_v` / `ego=` in `v8f_wm_windows.py`); TABLE and RUN_ORDER = E-KIN_s0, E-NOW_s0, E-KIN_s1, E-NOW_s1, E-KIN_rep_s2 with argvs from `egowm/argvs_egowm.txt` (frozen JSON like `frozen/argvs_otband.json`; the pin compares token for token); tree commit `1e1b9fc4ef33974ac82ee57f812a86490910ab76` (`EGOWM_COMMIT.txt`); GPU idle limit 3,500 MiB; box root `D:\tanitad\egowm\` (own tree, never `D:\tanitad\repo`, never the OTBAND tree); import closure from `egowm_ship_tree.py` must contain `tanitad/data/v8f_wm_windows.py` (add to the `|` set as the ladder did with WINDOWS_REL) and follow `from . import x`.
4. Run: `--selfcheck-only` on real data, per-run token, `--wm-speed` read back (`levers.wm_speed`), banks `C:/Users/Admin/t2v8f_pulls/egowm_<tag>/` with sha256 against box 2's own copy; pilot = 200 steps of E-KIN_s0, stop if the projection of 5 runs exceeds 72 h (permit as OTBAND).
5. Tests: adapt `otband/box2/tests` (`fake_box.py`, `plant_trainer.py`, stubs, `test_otband_dryrun.sh`, `test_otband_check.py`). The stubs must keep the `*-STUB-TRANSPORT` marker and the chain must refuse a stub when `EGOWM_DRY` is not 1 and a non-stub when it is 1 (already the template's rule; keep the test for it). The dry run must also prove the predecessor wait (create stub `otband_state/DONE.txt` inside the test state dir, never the real one: point `EGOWM_OTBAND_STATE` at a test folder).
6. Dry run against stubs only; do not launch; do not connect to box 2.

Template sizes: `otband_check.py` 876 lines, `otband_ship_tree.py` 203, `otband_chain.sh` 80, stages 3 files, jobs 2 files, tests 7 files.
