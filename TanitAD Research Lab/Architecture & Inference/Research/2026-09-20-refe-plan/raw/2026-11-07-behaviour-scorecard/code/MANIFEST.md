# MANIFEST -- streams BEHAV-CODE + BEHAV-F1, 2026-11-07

**Everything below lives in ONE place only: `E:/Projects/TanitAD-artifacts/refe_behaviour_20261107/code/` (outside the repo, not staged, not committed, not pushed, per the brief). Nothing is on a pod or in a worktree. The MM lands it; a copy in git is the only banking.**
CODE_PIN.json sha256 = `b07aad570bbfd242fb0140961dff51144ca741db388eb551a52a9818993f5039` (pins all 22 files below except CODE_PIN.json, HANDOFF.md, MANIFEST.md; `python make_pin.py --check`). Scorecard sha256 `fac6c7f1...` (file unedited); AMENDMENT_1.md applied (A-10, A-2 flag, tier stamp).

| file | what | location |
|---|---|---|
| `ROLLOUT_SCHEMA.md` | exact keys, dtypes, shapes, units of `rollout.npz` and the human reference; CLI. BEHAV-F1: `ego_rear_axle_to_center`, `agent_cls_defaulted`, the new tier stamp | code/ |
| `behaviour_score.py` | per-scenario scorer: lead/gap/TTC derivation, rules a1..f2, Gate H, EPDMS passthrough, JSON out. BEHAV-F1: f1 = vendored NAVSIM classifier (`_collisions`, `_classify_tick`), `TIER_STAMP`, `agent_cls_defaulted` | code/ |
| `behaviour_suite.py` | suite aggregator + paired adoption estimator (section 4); BEHAV-F1: per-block `tier_stamp`, `n_agent_cls_defaulted` | code/ |
| `AMBIGUITIES.md` | 18 definitions (A-1..A-18); A-2, A-10 and the tier stamp now marked RULED with the implemented reading | code/ |
| `test_behaviour_score.py` | **66 tests** (51 + 15 `test_f1_*`/tier-stamp tests) on synthetic rollouts: per-rule positive control + deliberate-regression arm, lead geometry, determinism, suite/adoption, A-10 geometry, vendored-body == source, real-NAVSIM fuzz parity | code/ |
| `synth_rollouts.py` | synthetic rollout generator (also the reference emitter for the schema) | code/ |
| `mutation_check.py`, `mutation_check_result.json`, `MUTATION_RESULT.txt` | 14 rules x {neuter, delete} plus f1 `oldrule` (superseded bearing rule swapped back in): all killed, original untouched | code/ |
| `superseded_a10_bearing_rule.py` | the OLD f1 rule (+-45 deg, 0.005), kept ONLY as the deliberate-regression arm; not used by the scorer | code/ |
| `TEST_RESULT.txt` | output of the final test run (66/66) | code/ |
| `navsim_pdm/pdm_collision.py` | NEW: NAVSIM `get_collision_type` + nuPlan `is_agent_behind`, `is_track_stopped`, `get_agent_relative_angle`, byte-identical bodies, sha-pinned (sources: `F:/Archive/devbox-C/navsim/devkit/navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer_utils.py`; `C:/Users/Admin/dz/DriveZero/DriveRL/nuplan-devkit/nuplan/planning/simulation/observation/idm/utils.py`) | code/ |
| `navsim_pdm/nuplan_shim.py` | NEW: thin adapter for the nuPlan types (not vendored code), sha-pinned | code/ |
| `navsim_pdm/` (rest) | NAVSIM comfort + enums (d1), from the refe-plan tree (see HANDOFF escalation 1: differs from the F: archive version), `LICENSE` Apache-2.0 | code/ |
| `example/` | worked rollout.npz, human.npz, score.json (tailgating at 0.8 s; score.json regenerated with the new stamp) + `make_example.py` | code/ |
| `make_pin.py`, `CODE_PIN.json` | pin writer/checker; sha256 of every file | code/ |
| `HANDOFF.md`, `MANIFEST.md` | state, escalations, deletions; this table | code/ |
