# MANIFEST: S1c build package (2026-10-16)

**Everything below lives in ONE place: this dev box (`C:/Users/Admin/AppData/Local/Temp/`). Nothing is in the repo, staged, on a pod, or in a worktree. Integration (land -> ship-manifest -> ship -> launch) is the Master Mind's step; see `S1C_DESIGN.md` section 0.**

`PK` = `C:/Users/Admin/AppData/Local/Temp/s1c/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-16-s1c-build/`
`W`  = `C:/Users/Admin/AppData/Local/Temp/s1c_work/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/` (the patched tree; the 11 changed files are byte-identical to `PK/patched/`)
`B`  = `C:/Users/Admin/AppData/Local/Temp/s1c_base/.../2026-09-20-refe-plan/` (pristine `git archive` of origin/agent/arch-inf-20260803 @ `df9871e3`; origin is now `afbc8350`, no change under `refe/` `eval/`)

| artifact | what | location | copies |
|---|---|---|---|
| `S1C_DESIGN.md` | argv diff, budget reason, head-only answer, cost model, ordering file:line, dd, reachability, eval plan, open questions | `PK/` | single location |
| `PREREG_S1C_DRAFT.md` | question, arm, committed outcomes both ways with next levers, seed policy, estimator | `PK/` | single location |
| `PATCH_1..11_*.diff`, `patched/`, `PATCHES.json` | the 11-file change set against `df9871e3`; `git apply` proven byte-exact | `PK/` | `W` holds the same bytes |
| `REACHABILITY.json` | numbers read by the reachability proof (head Adam steps, G12 grads, w0 control, RED mutants) | `PK/` | regenerate: `S1C_EVIDENCE_OUT=<f> python refe/test_s1c.py` |
| `COST_MODEL.json`, `cost_model.py` | S1c cost from S1's nine arm logs | `PK/` | inputs: `D:/Projects/TanitAD-artifacts/s1_read_cache/arms/` (also on HF `Sayood/tanitad-refe-s1aux`) |
| `ARGV_DIFF.json`, `argv_diff.py` | S1c argv vs the banked H1-s0 argv | `PK/` | input: `.../H1-s0/PASS_d257647b7858.json` |
| `logs/` | `base_*.log` (before), `post_*`/`final/`/`final2/` (after), `podlike_test_s1c.log`, `s1c_run1-6.log` | `PK/logs/` | single location |
| `tools/` | `run_suites.sh`, `run_eval.sh`, `make_patches.py` | `PK/tools/` | single location |
| `HANDOFF.md` | record only (work complete) | `PK/` | |

Files changed (all under `.../2026-09-20-refe-plan/`): `refe/launch_gate.py`, `refe/gate_checks.py`, `refe/s1aux_checks.py`, `refe/s1aux_pod.sh`, `refe/test_s1c.py` (new), `refe/test_s1aux_chain.py`, `refe/test_aux_drivable_head.py`, `eval/aux_s1_readout.py`, `eval/test_aux_s1_readout.py`, `eval/aux_navtest_forward.py`, `eval/test_aux_navtest_forward.py`.

Not produced (named in the design): `SHIP_MANIFEST.json` (needs a landed commit), the natural-population K2 driver, an `N-u10k` control, the replicate arm.
