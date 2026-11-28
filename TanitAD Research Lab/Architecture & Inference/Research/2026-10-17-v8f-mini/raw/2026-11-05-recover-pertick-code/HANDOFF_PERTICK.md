# HANDOFF_PERTICK -- `--recover-dev-shape {constant, pertick}` (tip 76a0d73c)

Nothing committed, pushed, landed, or launched. Patch = `stack_patch/stack/...` (LF); hashes in `SHA_CHANGES.txt`; supersets in `superset/`.

## What changed (file:line, in `stack_patch/`)
* `tanitad/data/v8f_recover.py`: docstring paragraph `THE SHAPE` :12; `DEV_SHAPES` :30, `PER_TICK_FIELDS` :38, `RECOVER_DEFAULTS["dev_shape"]="constant"`; `_om_profile` :49; table `cols_t` / `has_per_tick` / `ticks` :70-90, hash and `summary()` cover the per-tick columns only when present; `from_dagger_bank(per_tick=)` :141 + `_with_per_tick` :177; `RecoverPerturb(dev_shape=)` :235, `built_levers` :246, `problems()` :257/:274, `_draw` :304 (same stream, same index), `apply` guards the tick count :326 and keeps `last["dev"]` as NOW-row values plus `last["dev_t"]`, `_rebuilt` dispatches :347 (constant body untouched), `_rebuilt_pertick` :395 (anchor `PERTICK-PROFILE-INTAKE` :407).
* `scripts/v8f_mini_train.py`: FLAG_CHECKS :103; `--recover-dev-shape` :187; `RECOVER_DESTS` :690; `from_dagger_bank(per_tick=...)` :709; `make_recover` passes `dev_shape` :734; `bad_recover` loop :748 + pertick/apriori refusal :763; `recover_probe` new pertick clause (:1470s): history v == max(v + dv_t, 0) of the drawn profile, and flat profiles refused.
* Tests: new `tests/test_v8f_recover_pertick.py` (32 items) + `tests/v8f_recover_tip_snapshot.txt` (frozen tip module, sha256 pinned in the test); extended `test_v8f_recover_wiring.py` (ALL, defaults), `test_v8f_launch_intent.py`, `test_v8f_ego_dagger_compose.py` (dev_shape in the default-lever tuple).
* NOT touched: `launch_gate.py`, `ops/launch_gate_thor_env_failures_mapft.json` (no recover entry exists in either).

## How each test proves it
(a) constant == tip: snapshot module, `apply` outputs equal tensor by tensor on 6 step/seed/p/target combos, built lever = tip + one key, constant-table `content_sha256`, `from_dagger_bank` bytes/counts/log_v/summary. (b) constant rows == constant within 1e-9 (non-negative dv), plus the clamp case. (c) NOW column == constant law bit for bit, subset rows match the constant table exactly, unreal rows counted, independent per-record numpy recomputation <1e-12. (d) each history row's v, a, pose, nav3 vs numpy loops (worst err 1.8e-6). (e) argv -> make_recover -> built lever -> G-DVB both shapes, mismatch refused, every refusal (unknown, apriori, no per-tick columns, p 0), selfcheck-only, 4-step run. (f) probe passes, grad diff > 0, p=0 same object and exact 0.0 controls, pertick-vs-constant loss/grad differ. (g) mutation (profile -> NOW column) turns (d) RED (err 1.105) AND the gate probe RED (2.668): `logs/mutation_output.log`.

## Decisions the spec did not fix
1. NOW column of own v, a comes from the bank's float64 `kin3`, not `p10f` (float32): only that makes `dv_t[:, -1] == dv` bit for bit. History rows use `p10f` ch 0/1.
2. Per-row clamp: `dv_eff_t = max(v+dv_t,0)-v` per frame, so pertick differs from constant wherever the clamp binds at a different frame than NOW (test b2). With no clamp it equals constant.
3. Logged yaw rate row 0 := row 1 as for own; the `j >= F` rule is applied exactly as written (row j-F is checked real though not used).
4. Cumulative sums skip frames that are not real (`_hist_valid`).
5. Added the probe clause (not requested) so a gate pass proves the profile reaches the batch.
6. F mismatch (table vs batch) raises in `apply`; the launch probe hits it first.

## Unverified
* No real bank / real TRAIN clips: per-tick build and probe ran on the synthetic world only; real profile smoothness untested (a flat-profile bank would be refused by the probe).
* FULL SUITE (14,859 tests) is NOT finished at hand-back. One pytest process cannot hold it on this box (a 30-file batch reached 5 GB; two monolithic runs died at start with C: 7 GB free). Instead, DETACHED (WMI): `run_files.py new` runs every `test_*.py` of the patched tree in its own pytest process (log `logs/files_NEW.log`, ~85 min), then `chain_tip_and_compare.py` runs the UNMODIFIED tip on exactly the files that failed on NEW (`logs/files_TIP.log`) and writes `logs/COMPARE.txt` (NEW-ONLY must be empty; exit 0 = clean). Re-run `python compare_suites.py` any time for a partial read. Exports are `stack/` + `taniteval/` only, so data-dependent tests fail on both sides (pre-existing, e.g. test_argoverse2, test_secret_scan collection).
* Targeted on the patched tree (all green): test_v8f_recover 16, _recover_wiring 53, _recover_pertick 32, _fast_table 21 (+1 skipped), _launch_intent 36, _ego_dagger_compose 36 (`logs/targeted_new_*.log`). The other ~24 trainer-importing files are covered by `files_NEW.log`.
* Tip-export sha256 recorded; nothing re-read from E: (lags the tip).

## Result (appended 2026-11-05 after run 2)
The full suite finished: NEW-ONLY 0, TIP-ONLY 0, 318 failing on both trees before this change. Run 1 was invalid (a missing `--basetemp` parent). Details in `logs/README_SUITE.md`.
