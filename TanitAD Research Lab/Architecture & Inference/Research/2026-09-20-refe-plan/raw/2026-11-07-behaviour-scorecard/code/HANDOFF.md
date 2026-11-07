# HANDOFF -- streams BEHAV-CODE + BEHAV-F1 (metric code for BEHAVIOUR_SCORECARD.md), 2026-11-07

**State: DONE. Nothing running. Nothing staged, committed or pushed (per brief); folder is outside the repo, the MM lands it.**
`$C` = `E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/code/`. Python `C:/Users/Admin/venvs/driverl-eval/Scripts/python.exe`, `CUDA_VISIBLE_DEVICES=-1`. Scorecard sha256 `fac6c7f1...` verified, file not edited.

## BEHAV-F1 (AMENDMENT_1, A-10 + tier stamp + A-2): what changed
* **f1 now reproduces NAVSIM's at-fault logic from source.** `navsim_pdm/pdm_collision.py` = `get_collision_type` (NAVSIM `pdm_scorer_utils.py:12-64`), `is_agent_behind` (150 deg), `is_track_stopped`, `get_agent_relative_angle` (nuPlan `idm/utils.py`), bodies byte-identical to the sources (CRLF->LF only), sha-pinned in `behaviour_score._VENDOR_SHA` and tamper-tested. `navsim_pdm/nuplan_shim.py` is the thin adapter for the nuPlan types (adapter, not vendored). `_collisions` / `_classify_tick` follow `pdm_scorer.py:363-420` tick by tick: ego "stopped" and track "stopped" are `<= 5e-02`; the exemption (first not-at-fault contact exempts the track; a track overlapping at tick 0 is exempt) is implemented. Lane-less stand-in kept: `|cross-track| > 0.5 m`, output flag `lane_proxy = true` (rule level and `details`).
* **Tier stamp** is `T2 (re-perception closed-loop sim; MM ruling 2026-11-07)` everywhere (`TIER_STAMP`; also AMBIGUITIES.md, ROLLOUT_SCHEMA.md, regenerated `example/score.json`).
* **A-2:** output flag `agent_cls_defaulted` (true iff the rollout has agents but no `agent_cls`); suite aggregate carries `n_agent_cls_defaulted` and `tier_stamp` per block.
* New optional rollout key `ego_rear_axle_to_center` (default 1.461 m, nuPlan Pacifica): NAVSIM evaluates `is_agent_behind` at the rear axle, the schema carries box centres.
* Tests 51 -> **66** (15 new, all hand-built geometry, see `test_f1_*`): front-bumper hit on a mover = at fault; rear-end = not at fault (and the exemption holds inside the run); ego 0.04 m/s = stopped, 0.06 = not; stopped track = at fault (0.04) / front hit (0.06); lateral ct 0.3 = not at fault, 0.8 = at fault; exemption across runs and at tick 0; static object score 0.5; 150 deg tolerance; pins/tamper; **bodies == sources**; **3,000-point fuzz parity of the vendored classifier vs the REAL NAVSIM source run on real nuplan types (all 5 collision types hit)** -- both skip loudly if the sources are absent.
* **Deliberate-regression arm:** `superseded_a10_bearing_rule.py` (old +-45 deg / 0.005 rule) swapped in makes the A-10 scenario checks go RED (asserted by `test_f1_old_bearing_rule_turns_the_a10_scenarios_red`, with a green control on source) and, in `mutation_check.py`, kind `oldrule` on the `# MUT:f1_cls` line is KILLED by `test_f1_ego_stopped_is_speed_le_0p05_not_0p005`. neuter/delete of f1 still KILLED. **ALL KILLED** (14 rules x neuter/delete + f1 oldrule).

## Escalations (integration and decisions)
1. **MM -- d1 (not touched): the vendored `pdm_comfort_metrics.py` is NOT the F:/Archive NAVSIM source.** It comes from the refe-plan tree; 8 function bodies differ from `F:/Archive/devbox-C/navsim/.../scoring/pdm_comfort_metrics.py` (archive version takes `vehicle_parameters` and converts rear-axle to centre-state acceleration via `state_array_to_center_state_array`, and adds extended-comfort functions). d1 may not reproduce the archive NAVSIM. Decide which NAVSIM version is binding; then re-vendor d1 the same way as f1.
2. **MM -- A-10 judgement call:** source order puts STOPPED_EGO first, so a stopped ego (<= 0.05 m/s) hitting a **static object** is NOT at fault, unlike the amendment's one-line summary ("non-agent objects are at fault"). I followed source order. Say if a stopped ego must still fail on a non-agent.
3. **Sim session:** emit `agent_cls` (else `agent_cls_defaulted = true`), and `ego_rear_axle_to_center` if not the Pacifica 1.461 m. Otherwise `ROLLOUT_SCHEMA.md` unchanged. `lane_proxy` stays until a lane map exists.
4. **Real-data gate not yet run:** all tested on synthetic rollouts. First real rollout: score the human log against itself (Gate H) before any checkpoint. Until then f1 is source-faithful but untested on a sim rollout.

## Re-run
`cd $C && python test_behaviour_score.py` (66 tests, ~25 s) ; `python mutation_check.py` (~25 s) ; `python make_pin.py --check`. Score: `python behaviour_score.py score --rollout R.npz --human H.npz --out S.json`.

## Facts worth knowing
* Tests that use the ego as its own human reference assert on `raw_status` (a FAIL there is Gate H INFORMATIONAL by design).
* pytest is not installed in the driverl-eval venv; tests are plain functions with their own runner.
* TACTICAL executed-decision table is still A-14 (work item); nuPlan-seeded block cannot score c1-c4 until map lifting exists.
* Bootstrap: 10,000 draws, seed 20260927; scoring is bit-identical across processes.

## Deletions (rule 4: literal path, size) -- logged by BEHAV-F1
* `C:/Users/Admin/AppData/Local/Temp/claude/E--Projects-TanitAD/c216c849-5fe7-4d43-a3be-32429241f811/scratchpad/bf1/` (my generator, parity and splice scripts, `newtests.py`, `edit_docs.py`, `score_old.json`; all re-creatable, nothing unique): removed at the end of the job, sizes in the final report.
* Test and mutation temp dirs (`bv_*`, `mut_*`) are removed by the scripts' own `finally`. No `__pycache__` written (PYTHONDONTWRITEBYTECODE=1).
