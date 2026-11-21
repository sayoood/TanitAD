# BUILD_CKPT120_FIX3 -- dagger_probe x nav_s (H2 mint refusal)

Base: branch `agent/v8f-ckpt120-20261117` at `213250cee3c0a695c22f7ce823f4a7aabee6e311` (export checked: blob sha256 of `stack/scripts/v8f_mini_train.py` = a0372d9e...b570, equal to `git show 213250ce:`).
Staged by this agent: nothing committed, nothing pushed, nothing edited under E:/Projects/TanitAD/stack. `launch_gate.py` and `stack/tanitad/models/__init__.py` untouched.

## The defect (verified, not re-derived)
`stack/scripts/v8f_mini_train.py` `dagger_probe` (the function the G-DVB `--selfcheck-only` path calls; the MM's name `_dagger_probe` is `dagger_probe` in the source) admitted `M.NAV_S_KEYS` in its `stray` partition (:3220) but its `kept_moved` check (:3255 in 213250ce) still demanded `pert["nav_s"] is batch["nav_s"]`. `DG.swap_rows` (`v8f_dagger.py:577-583`) rebuilds `nav_s` channel 3, so every batch that carries `nav_s` (H2_s0, H2_s1, R2 donor arms) was refused. H1 arms carry no `nav_s`: unaffected.
Reproduced RED on the unpatched file, with the pod's exact text: `AssertionError: dagger probe: ['nav_s'] are not the same objects after the swap: targets, labels and kept inputs must not be touched` (`scripts\v8f_mini_train.py:3257`).

## Changes (all in `stack/scripts/v8f_mini_train.py`, patched line numbers; `DG.REBUILT` / `EGO_PERTURB_REBUILT` untouched)
* :3294 `kept_moved` excludes `M.NAV_S_KEYS` (the key is absent from a batch without `nav_s`, so that case is the old expression).
* :3297-3298 `if "nav_s" in batch:` runs the new clause and records `res["nav_s"]`. With no `nav_s` in the batch nothing is added to `res` (bit-identical, see test d).
* :3184-3216 new `_dagger_nav_s_check` (+ `changed_frames` :3218): same shape and dtype; any unselected row bit-identical; channels 0-2 and 4 bit-identical on every row; channel 3 of the selected rows equal (1e-6) to the nav rule recomputed in the probe from the logged `nav_s` and the bank record's own arc: a real frame (`EgoPerturb._hist_valid`) sharing NOW's token and validity, with an own kinematic row behind it, is `clip(nav_s[NOW,3] + own arc behind NOW / 100, 0, 3)`; NOW and every other frame as logged. This mirrors `swap_rows` :580-583 and is the same style as the probe's `rc` recompute. Record: `res["nav_s"] = {rows_changed, ch3_frames_changed, ch3_frames_rebuilt_by_rule, ch3_max_abs_err, token_validity_bit_identical, unselected_rows_bit_identical}`.

## Other sites (git grep at 213250ce: `EGO_PERTURB_KEPT|DG.KEPT|kept_moved|is not batch|same objects`)
* `ego_perturb_probe` :2961 and `recover_probe` :3112 keep the same-object check. Left alone on purpose: `bad_sfar` (:1871) refuses `--s-far`/`--nav-strat on` together with the ego perturbation and the recovery stage (they would move S's frame without re-expressing nav_s), so those probes never see a `nav_s` batch. Their `stray` lines already carry `NAV_S_KEYS`.
* `ego_dagger_compose_probe` (:3324+) checks `D.EGO_PERTURB_REBUILT` only; composition is also refused with nav_s by `bad_sfar`. No change.
* `swap_item` (`v8f_dagger.py:602`) already carries the rebuilt `nav_s` (test `test_v8f_ckpt120_navs_swap.py`, 3 passed).
* Gate checkers `stack/scripts/ckpt120_check.py` (and `ckpt120_reach.py`): no kept/rebuilt enumeration; they read `selfcheck.dagger_probe` only through the trainer (`ckpt120_check` tests 202 passed). Replays `g_box_overfit._build_model` / `refcv7_loader`: no `REBUILT`/`KEPT`/same-object site (grep empty). No other file needed a change.

## Tests: `stack/tests/test_v8f_ckpt120_dagger_nav_s.py` (new, 7 tests)
Uses the pod H2_s0 levers from `F:/TanitAD-scratch/strat1/ckpt120build/pod/ckpt120_argvs_dry.json` (`--s-far on --w-ckpt-far 0.3333 --s-far-lr-mult 10 --s-far-label-perm off --s-adapter-detach off --nav-strat on --nav-strat-variant real --detach-upper`) through the trainer's own parser and `make_config`, on the DAgger pack (trained checkpoint, bank, synthetic world) of `test_v8f_dagger_train.py`.
* (a) probe PASSES with nav_s and DAgger on, and records the counts. RED on unpatched (the pod message above), GREEN patched.
* (d) batch without nav_s: passes, no `nav_s` key in `res`, deterministic (all keys except `live_swapped`, whose gradient sums move in the 6th digit between two calls).
* (b) swap changes nav_s channel 0: RED (`nav_s channels 0-2 / 4 ... changed`). (c) swap changes nav_s on an unselected row: RED. Also NOW frame moved, a frame outside the nav rule changed, and swap leaving nav_s as logged: each RED (`nav_s channel 3 differs from the nav rule`).
* Mutation of the FIX itself (clause deleted): 6 of 7 go RED (only (d) stays green).
* Unpatched trainer: 7 of 7 fail (the unmutated pass at the head of each fails with the pod message); patched: 7 passed.

## Regression (CPU, patched vs unpatched export of 213250ce): 0 regressions
List (`evidence/testlist.txt`): test_v8f_ckpt120_dagger_nav_s 7 passed; test_v8f_ckpt120_navs_swap 3; test_v8f_ckpt120 38; test_ckpt120_check 202; test_ckpt120_eval 25 (+1 skipped); test_ckpt120_eval_e2e 5; test_ckpt120_row_alignment 5; test_strat1_token 18; test_v8f_dagger 57; test_v8f_dagger_train 95 (+2 skipped); test_v8f_ego_dagger_compose 35 passed, 1 failed; test_v8f_ego_perturb 80 (+2 skipped); test_v8f_recover 16; test_v8f_recover_pertick 32; test_v8f_recover_wiring 53; test_v8f_strat1 36. Logs: `evidence/reg_patched_clean.txt`.
The one failure, `test_v8f_ego_dagger_compose.py::test_flag_off_reproduces_the_unmodified_tip_bit_for_bit`, fails identically on the UNPATCHED export (`evidence/reg_base.txt`: 1 failed, 35 passed): a golden-digest test sensitive to the torch build on this box. Not caused by this patch. The full G-SUITE is the MM's, on the pod.

## Pins and re-mint
* No sha of `v8f_mini_train.py` is pinned in `launch_gate.py` or in a live profile file (grep at 213250ce: only banked historical `gate_token.json` / `tree_manifest.json` under raw/). The edit does not touch `launch_gate.py`, so the 6 profile fingerprints / 3 tests do not move from this patch.
* The CKPT120 gate token binds the commit, so a new commit means re-minting every arm's token. The pod `ckpt120_new10.sha256` list and the 19-entry diff in `ckpt120_gsuite.sh` name the changed files of 213250ce: the MM adds `stack/tests/test_v8f_ckpt120_dagger_nav_s.py` and the new `v8f_mini_train.py` sha256 there.
* The CKPT120 test file `test_v8f_ckpt120.py` snapshots `v8f_mini.py` (not the trainer); unchanged.

## Not verified
No pod/GPU run; the probe was exercised on the synthetic world only. On real H2 data the check demands exact agreement with the nav rule; a refusal there would name the frame count and max error.

## Deletions (own temp, literal paths)
Logged at the end of the run in the final report (see the handoff message).

## Deliverable manifest
| artifact | where |
|---|---|
| fix3.patch (git-apply-able on 213250ce, LF, checked with `git apply --check` then cmp) | F:/TanitAD-scratch/strat1/ckpt120fix3/fix3.patch |
| patched trainer | F:/TanitAD-scratch/strat1/ckpt120fix3/files/stack/scripts/v8f_mini_train.py |
| new test | F:/TanitAD-scratch/strat1/ckpt120fix3/files/stack/tests/test_v8f_ckpt120_dagger_nav_s.py |
| evidence logs | F:/TanitAD-scratch/strat1/ckpt120fix3/evidence/ |
All exist in ONE place (F: scratch); nothing is staged in the repo (the MM commits).
