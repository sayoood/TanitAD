# I3: old-record rebuilds at the STAMPED query count. The named loaders are pinned; the exposure is in refcv6_loader

**Item:** refcv7 box head `BUILD.md` I3 (SPEC_REFCV7 A9 R4). Master Mind ruling: PATCH.
**Built on:** remote tip `37086c399cf209ad2bca8e0ffe19242c5e1e5f56`. The tip moved twice while I worked:
* to `2e123e19ca3d821ab81354de75175e2bb52bb0e4`, a SPEC doc correction;
* then to **`b711411d89e8ce62ac04a0097e415e7d06a7fba9`** (refcv7 box A17: `g_box_overfit.py`, `launch_gate.py`,
  `box_head_guard.py`, their tests, and the canonical refcv7 argv).

**All 18 dependency files of the new tests read SAME by blob at `b711411`** (list in §6), and both new paths are
still ABSENT there. The two new files were **re-run on a `git archive` of `b711411`: 17 passed**
(`raw/newtests_at_b711411.log`).
**Date:** 2026-09-27 (Europe/Berlin). CPU only, dev box, `OMP_NUM_THREADS=4`, ≤ 2 concurrent pytest processes.
**Author:** Architecture & Inference builder agent. Nothing staged, committed or pushed; the Master Mind lands.

---

## 0. Headline (escalation)

1. **None of the three loaders I3 named needed a code change. The premise in I3 is wrong for all three.**
   None of them re-parses a recorded argv itself:
   * **`closedloop_drive.py`**: its refcv3 arm calls `refcv3_arm.load_model` (`:395`). The stamped rule is already in that function.
   * **`bench/cli.py`**: its only `parse_args` is its OWN CLI (`:23`, `:317`). Every in-process model the
     suite loads goes through `bench/navsim/bridge.py::load_refcv4b` → `refcv3_arm.load_model` (`:246`, `:255`).
     `internal_t1` runs `refcv3_arm.py` as a subprocess.
   * **`seam_probe.py`**: it rebuilds **no model at all**. Its `build_parser` is its own CLI (`:508`, `:575`), and it
     reads emission dumps (`load_dump`, `:116`).
2. **MEASURED:** a synthetic **pre-A9 record** (argv silent on `--agent-queries`, stamps `seams.agents.queries: 100`,
   `refcv6_perception.n_queries: 100`) rebuilds at **100 / 100 and loads STRICT (0 missing, 0 unexpected)**.
   That holds through all three entry points (`refcv3_arm.load_model`, `closedloop_drive.RefCV3Policy`,
   `bench bridge.load_refcv4b`). A **post-A9 record** rebuilds at **300 / 300**, strict.
   **Two source mutations turn the new tests RED on all three entry points.** M1 reintroduces the pre-I3 trainer
   (resolver removed). M2 makes the loader ignore the box-head stamp.
3. **The landing is 2 NEW test files, and no existing file is modified.** That respects the coordinator's 2026-09-27
   closure freeze (`refcv3_arm.py`, `stack/tanitad/**`, `refc_v3_train.py`, `g_box_overfit.py`,
   `map_hires_overfit.py`, `closure_run.py` untouched). No new module is imported by any of them.
4. ⛔ **The REAL unpatched I3 exposure is `refcv6_loader.build_model`**, the refcv6 battery loader, which exists as
   3 in-repo copies. **MEASURED:** on a pre-A9 record, rebuilt against a post-A9 tree, it dies on **BOTH** heads.
   The agent head and the box head are each rebuilt at 300 against a checkpoint that holds 100. **The fix is
   BLOCKED**, and I have NOT worked around it:
   * the in-repo copy is under `stack/tanitad/` (frozen);
   * it is pinned byte-for-byte to the audit blob by `g_box_overfit.py` (frozen) and by a literal in
     `test_g_box_overfit.py`;
   * the other two copies are banked EvalFlyWheel / audit evidence.

   **Decision needed from the Master Mind** (§4). It fails LOUDLY, never silently.
   **MEASURED: in its default configuration it is not exposed.** The dev-box battery tree `C:/Users/Admin/ev6`
   still has `N_QUERIES_DEFAULT = 100`.
5. **A pre-existing failure at `37086c3`, unrelated to this change, and already FIXED at the current tip.**
   `test_g_box_overfit.py::test_TOY_the_loop_memorises_...` read AP 0.652 against a bar of 0.8. It gave the
   identical value `0.651569903443288` when run **alone** on `37086c3` without my files. **At `b711411` (A17) the
   same test passes alone: 1 passed** (`raw/g_box_overfit_TOY_at_b711411_omp4.log`).

---

## 1. What I3 said, and what the source says (tip `37086c3`; citations verified by grep)

| Loader named by I3 | Re-parses a RECORDED refc argv? | Its model path | Needs a patch? |
|---|---|---|---|
| `taniteval/tools/seam_probe.py` | **No.** `build_parser` `:508` / `parse_args` `:575` is its own CLI | **none**: it reads `.pt`/`.npz` emission dumps (`load_dump` `:116`); the self-test imports only `train_v58f_unicycle_head.unicycle_rollout` (`:378`) | **No**: there is no rebuild for the rule to act on |
| `taniteval/taniteval/bench/cli.py` | **No.** `build_parser` `:23` / `parse_args` `:317` is its own CLI; work goes to `_module_for` `:163` | navsim_v2 → `navsim/model_arms.py:174` → `bridge.load_refcv4b` `:246` → **`refcv3_arm.load_model`** `:255`; nuscenes_ol → `adapters/nuscenes_planning.py:2041` → the same `load_refcv4b`; internal_t1 → subprocess `taniteval/tools/refcv3_arm.py` (`internal_t1.py:29`, `:47`) | **No**: it already runs the stamped rule |
| `stack/experiments/alpasim-gsplat/closedloop_drive.py` | **No.** Its `ap.parse_args()` `:1001` is its own CLI | `RefCV3Policy` `:376` → **`refcv3_arm.load_model`** `:395`. `FlagshipV1Policy` `:304` builds `flagship4b_config()` `:312` (no argv, no slot head). `RefCPolicy` `:345` uses `refc_v12_cache.load_frozen` presets `:350` (no argv, no slot head) | **No**: it already runs the stamped rule |

**The stamped rule** (read, not edited):
* **agent head.** `refcv3_arm.rebuild_config` `:1024`-`:1029` calls `refc_v3_train.agent_queries_as_trained`
  (`:1509`). Precedence: an explicit argv `--agent-queries` wins, then the stamp `seams.agents.queries`, then
  `(None, why)`.
* **box head.** `refcv3_arm.rebuild_perception_branch` reads `n_queries` from the `refcv6_perception` stamp (`:1101`).

⚠️ **The no-stamp case, as refcv3_arm handles it** (the brief asked me to check this). When `agent_queries_as_trained`
returns `(None, …)`, refcv3_arm **keeps the parser default (300)** and writes nothing about it into `rebuilt_from`. It
then relies on its guards, which are `cross_check_config`'s `param_breakdown` and the strict load. The guards
**REFUSE** a record that was trained at another count; MEASURED in the red arms below, both messages. A record that
was trained at 300 with no stamp would load correctly. So the default is never *silently wrong*. It is, however,
*silently chosen*: the provenance does not say "parser default". **Proposed follow-up, not done because the file is
frozen:** append `; agent_queries: parser default (<why>)` to `src` in that case.

**Why I did not "implement the rule inside" the three loaders:** the coordinator's instruction assumed they re-parse
argv. They do not. Adding a copy of the rule to `closedloop_drive.py` would fork the loader its own docstring forbids
forking (`:379`-`:385`: *"The loader is IMPORTED, never forked"*). In `seam_probe.py` a copy would act on nothing.

## 2. What changed (the landing)

Two NEW files. No existing file is touched.

| Repo path | What it pins |
|---|---|
| `stack/tests/test_loader_stamped_queries.py` (11 tests) | `refcv3_arm.load_model` and `closedloop_drive.RefCV3Policy` (through the policy's own constructor): pre-A9 → 100/100 strict; post-A9 → 300/300 strict. Three red arms per entry point. Plus a mutation-tested AST census that `seam_probe.py` rebuilds no model. |
| `taniteval/tests/test_bench_stamped_queries.py` (6 tests) | `bench/navsim/bridge.load_refcv4b` (the bench's one in-process model loader): the same pre/post/red-arm set. Plus the same census on `bench/cli.py`. |

**The synthetic records** are built through the trainer's OWN `build_parser` + `_pin_trainer_cfg` + `RefCV3Model` +
`build_perception_branch`, with `args.agent_queries` set to the count "the parser default was then". The argv is
**silent on `--agent-queries`**, as refcv6-r101-s0's is. Both heads are built:
* the agent head via `--agents head`;
* the box head via `--w-box3d 1.0`, which requires `--trunk timm` (resnet18, no pretrained weights) and `--agent-join`.

The geometry is 9 channels, 256x640 and the v3 horizons, because `RefCV3Policy` and `load_refcv4b` both assert it.
**Pre-A9 records omit the stamp fields that A9 / A14 / A6 / A7 added:** `presence_loss`, `presence_prior`,
`deep_supervision`, `vis1`, `query_select`, `bev_source`, `planner_crop_m`. Their `n_queries_default_upstream` is 100.

**All expectations are literals.** The built counts are read from the loaded query tables
(`core.agent_head.queries.shape[1]`, `_perception.box_dec.queries.shape[1]`), not from a config field.

**The three red arms per entry point:**
* **`pre_i3_tree` fixture.** Every `refc_v3_train.py` loaded under it lacks `agent_queries_as_trained`: an
  `importlib.util.spec_from_file_location` hook strips it after exec. Already-loaded `refcv3_arm` copies re-resolve
  their trainer. Everything is restored at teardown.
  * Without `param_breakdown` in the record, the pre-A9 record is refused at the **strict load**. The test asserts
    `DISAGREE ON SHAPE`, `size mismatch for core.agent_head.queries` and `torch.Size([1, 100,` / `torch.Size([1, 300,`.
  * With `param_breakdown`, it is refused **first by the cross-check** (`CONTRADICTS the rebuilt model`).
* **Box stamp without `n_queries`.** This is the behaviour of a loader that ignores the stamp. The branch is rebuilt
  at 300 and refused on `_perception.box_dec.queries`, and the test asserts the agent head is **not** named.

## 3. Test evidence (MEASURED; logs in `raw/`)

All commands ran with `PYTHONPATH=C:/lgt/i3tree/stack;C:/lgt/i3tree/stack/scripts;C:/lgt/i3tree/taniteval`,
`PYTHONIOENCODING=utf-8 PYTHONUTF8=1 OMP_NUM_THREADS=4`, and `C:/Users/Admin/venvs/tanitad/Scripts/python.exe`.
`tanitad` and `taniteval` were verified to import from the clean tree. The clean tree is `C:/lgt/i3tree`, a
`git archive` of `37086c3` covering `stack taniteval tools`.

**3.1 The new files alone.** `pytest -q stack/tests/test_loader_stamped_queries.py` gives **11 passed**, and
`pytest -q taniteval/tests/test_bench_stamped_queries.py` gives **6 passed**. The combined 17 are the
`CONTROL_unmutated` row in `raw/mutations.log`.

**3.2 The same tests against the UNPATCHED logic go RED** (`raw/mutations.log`; tool `raw/scripts/mutate.py` +
`run_mutations.sh`). These runs used a SEPARATE scratch tree, `C:/lgt/i3mut`, with the source mutated and then
reverted. Both files were verified byte-identical to the tip blobs afterwards: `refc_v3_train.py` `203b437f…`,
`refcv3_arm.py` `c5862e71…`.

| Arm | Source change (the historical defect) | Result |
|---|---|---|
| CONTROL | none | **17 passed** |
| **M1** | `refc_v3_train.agent_queries_as_trained` renamed away (the pre-I3 tree) | **6 failed / 11 passed**. The pre-A9 test fails on all 3 entry points (`refcv3_arm`, `closedloop`, `bench`). The box-stamp red arm also fails on all 3, because its "agent head is not the mismatch" assertion is now violated. |
| **M2** | `rebuild_perception_branch` no longer reads the stamped `n_queries` (refcv6_loader's behaviour) | **3 failed / 14 passed**. The pre-A9 test fails on all 3 entry points. |

**3.3 The red-arm fixture leaves nothing behind.** This used a scratch probe, `raw/scripts/isolation_probe.py`, which
is NOT landed. It runs after the file in the same process. It asserts: no hooked importer; no stripped trainer in any
arm copy or in `sys.modules["refc_v3_train_for_arm"]`; nothing planted on `torch.ops`. It also checks that a pre-A9
record still loads at 100 through both entry points.
* **Full order: 12 passed.** `-k` order (red arms only, then the probe): **3 passed.**
* Two defects were found and fixed on the way. Each is shown by a mutation of the fixture that turns the probe RED:
  1. ⚠️ **A `hasattr` selector plants attributes on lazy namespace modules.** `hasattr(torch.ops, "_TRAINER")` is
     True because the lookup CREATES it, so the first fixture patched `torch.ops`
     (`raw/isolation_probe_MUTATED_hasattr_selector.log`). Arm copies are now selected by `__file__` read from
     `__dict__`.
  2. **A stripped trainer survived in `sys.modules["refcv3_arm_i3_under_test"]`** when a red arm was the last caller
     (`raw/isolation_probe_k_MUTATED_restore.log`). That name is now restored too.

**3.4 The covering suites, with and without the new file** (per-test outcomes diffed from JUnit XML):

| Suite (files) | WITH the new file | BASELINE (without) | Diff |
|---|---|---|---|
| **stack**: `test_loader_stamped_queries` + `test_refcv3_arm`, `test_v6_seam_probe`, `test_closedloop_floor`, `test_nav_known_channel`, `test_refcv7_box_head_wiring`, `test_g_box_overfit` | **163 passed, 1 failed** | **152 passed, 1 failed** | +11 passed (the new file); **0 outcomes changed** |
| **taniteval**: `test_bench_stamped_queries` + all 12 `test_bench_*`, `test_refcv3_arm_astar_geometry`, `test_refcv3_arm_equalize_as_trained`, `test_refcv6_perception_rebuild`, `test_map_hires_rebuild` | **225 passed, 26 failed, 17 errors, 2 skipped** | **219 passed, 26 failed, 17 errors, 2 skipped** | +6 passed (the new file); **0 outcomes changed** |

**The pre-existing failures, same set in both columns.** None touches a changed file; there are none.
* **stack: 1.** The `g_box_overfit` TOY failure (§0 item 5). It also fails standalone at `37086c3`
  (`raw/g_box_overfit_ALONE_omp4.log`, 1 failed / 17 passed), and it passes at `b711411`
  (`raw/g_box_overfit_TOY_at_b711411_omp4.log`).
* **taniteval: 43, all ENVIRONMENTAL.** The clean tree is a `git archive` of `stack taniteval tools` only.
  * 37 read `FlyWheels/…` banked artifacts that are not in it.
  * 2 read `products/P7-TanitEval/CRITERIA_REGISTRY.json`.
  * 2 run `git show HEAD:…` in a tree with no `.git`.
  * 2 are bench dry runs that refuse on the same missing FlyWheels inputs.

## 4. The real exposure: `refcv6_loader.build_model` (MEASURED; BLOCKED; decision needed)

**The mechanism** (read in `stack/tanitad/eval/refcv6_loader.py`; the audit / battery copies are byte-identical after
the provenance block):
* `parse_args` `:177`: `tr.build_parser().parse_args(argv)`, with no as-trained step, so the agent head is built at
  the parser default;
* `_build_model` `:280`: `PerceptionBranchConfig(w_map=…, w_box3d=…)`, with no stamp read, so the box head is built at
  the dataclass default.

This was correct on a pre-A9 tree, where both defaults were 100.

**The measurement** (`raw/refcv6_loader_exposure.log`, tool `raw/scripts/measure_refcv6_loader_exposure.py`, tree
`C:/lgt/i3tree`):

| Record | `refcv3_arm.load_model` | `tanitad.eval.refcv6_loader.build_model` (strict) |
|---|---|---|
| pre-A9 (100) | OK, built [100, 100], 0 missing / 0 unexpected | **RuntimeError**: `size mismatch for core.agent_head.queries` … `[1, 100, 256]` vs `[1, 300, 256]` **and** `size mismatch for _perception.box_dec.queries` … `[1, 100, 256]` vs `[1, 300, 256]` |
| post-A9 (300) | OK, built [300, 300] | OK, built [300, 300] |

**The copies** (blobs by `git rev-parse` at the tip):

| Copy | Blob | Status |
|---|---|---|
| `FlyWheels/TanitAD_EvalFlyWheel/incoming/2026-09-23-refcv6-standard-tests/battery/code/refcv6_loader.py` | `11808258…` | the battery; `launch_gate._EVAL_LOADER_REL` `:152` |
| `TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-box-head-audit/code/refcv6_loader.py` | `11808258…` | the audit |
| `stack/tanitad/eval/refcv6_loader.py` | `14450c78…` | the audit blob plus the vendor block, pinned by `g_box_overfit.AUDIT_LOADER_BLOB` `:521` and a test literal |
| dev box `C:/Users/Admin/ev6_battery/code/refcv6_loader.py` | `11808258…` | not in the repo |

**Who calls `build_model`, and when they would hit it.**
* `render_refcv6_map_video.py:1202` uses the battery copy (default `--battery-code` `:64`).
* `launch_gate.py:4244` (G-EVAL) rebuilds the launch's OWN post-A9 argv, so it is consistent and **not exposed**.
* `g_box_overfit.py` uses the vendored copy for `build_eval_dataset` only (`:646`), so it is **not exposed**.
* `render_refcv6_gt_validation.py` builds no model (`:422`), so it is **not exposed**.
* **The loader resolves its tree from `REFCV6_REPO`**, which defaults to `C:/Users/Admin/ev6` (`:67`). MEASURED on
  the dev box:
  * that tree's `agent_slots.py:212` reads `N_QUERIES_DEFAULT: int = 100`;
  * its trainer (9,771 lines read) has **no** `agent_queries_as_trained`.
  * ⇒ **In its default configuration the battery rebuilds refcv6 records correctly.** It breaks only if someone
    points `REFCV6_REPO` at a post-A9 tree and loads a pre-A9 record. For example, a refcv7-vs-refcv6 comparison run
    on the current tree. It then fails loudly on a strict-load RuntimeError.

**What would unblock it, by option:**

| Option | What it does | Blocked by / needs |
|---|---|---|
| **(a) ACCEPT + document** (recommended now) | Pre-A9 records go through `refcv3_arm.load_model` (patched, pinned above) or through the battery on its pre-A9 `ev6` tree. | Nothing: this is a documentation line |
| **(b) PATCH after the binding runs** | Apply the two lines of refcv3_arm's rule in `refcv6_loader`: `agent_queries_as_trained` after `parse_args`, and `n_queries=` from `config["refcv6_perception"]` at `:280`. Re-pin A11: `AUDIT_LOADER_BLOB`, the literal in `test_g_box_overfit.py`, and the vendor block. | The closure freeze on `stack/tanitad/` and `g_box_overfit.py`; an A11 contract decision (the vendored copy stops being "the audit's loader, byte for byte") |
| **(c) New battery loader version** | A new file beside the banked one, never an edit of it | The EvalFlyWheel owner's decision |

The test for (b) already exists in shape: the entry-point pattern above plus `raw/scripts/measure_refcv6_loader_exposure.py`.

## 5. Also checked (not exposed)

* **The v6 line.** `train_v6_staged.py` `--n-slot-queries` also defaults to `N_QUERIES_DEFAULT`. Its loader rebuilds
  from the FULL recorded args dict: `train_v6_staged.py:6369` banks `"args"`, and `v6_probe_trunk._run_args` `:65`
  reads it. So `n_slot_queries` comes from the record, not the default. ⚠️ UNVERIFIED whether any v6 record lacks
  the key; it would then take the default.
* **Other refc rebuilders all route through `refcv3_arm.load_model`:** `s1_pass.py:252`, `ddv2_rl_refcv5.py:117`,
  `render_refcv3_video.py`, `openloop_suite.py`. `g_box_overfit.py` builds fresh from the LAUNCH argv, with no
  checkpoint.

## 6. UNVERIFIED / limits

* **No real pre-A9 checkpoint was loaded.** refcv6-r101-s0 is not on this box, and Thor is off-limits. The synthetic
  record's stamp shape is **INHERITED** from the trainer source (`_seam_stamp`, `as_dict()`) and from
  `test_refcv7_box_head_wiring.py`'s `refcv6_like` fixture.
* **The tip moved `37086c3` → `2e123e1` → `b711411` during the work.** These 18 files read SAME by blob at
  `37086c3` and at `b711411`:
  `refcv3_arm.py`, `refc_v3_train.py`, `closedloop_drive.py`, `bridge.py`, `bench/cli.py`, `seam_probe.py`,
  `refcv6_perception_branch.py`, `agent_slots.py`, `refc_agents.py`, `refc_v3.py`, `refc.py`, `refcv6_loader.py`,
  `box3d_head.py`, `timm_trunk.py`, `stack/tests/conftest.py`, `taniteval/conftest.py`, `refav1_arm.py`, `t1_eval.py`.
  * The covering suites (§3.4) and the mutation / isolation runs used the `37086c3` tree.
  * Only the two new files were re-run at `b711411` (17 passed).
  * `b711411` changed `g_box_overfit.py`, `launch_gate.py`, `box_head_guard.py`, `test_g_box_overfit.py` and
    `test_launch_gate.py`. None of them is imported by the new tests.
* **Only the covering subsets were run, not the full suites.**
* **The `g_box_overfit` TOY failure:** it fails at `37086c3` and passes at `b711411`, both at `OMP_NUM_THREADS=4`.
  Which A17 change fixed it was not isolated.
* **The `ev6` tree is not a git checkout.** Its commit (the docstring says `287d72e`) is INHERITED; only its
  `N_QUERIES_DEFAULT` and trainer were read.

## 7. Deliverable manifest

Package dir (D:): `D:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-27-loader-stamped-queries/`

| Artifact | Where it lives | Only there? |
|---|---|---|
| `code/stack/tests/test_loader_stamped_queries.py` (blob `ed94ed79b8f15f4d331fc8648bf1ad170fda5dde`, LF) | package; copy in `C:/lgt/i3tree/stack/tests/` (scratch) | lands via LANDING_READY |
| `code/taniteval/tests/test_bench_stamped_queries.py` (blob `555e13ceaa6c7c24adb364af754673eb2e89b535`, LF) | package; copy in `C:/lgt/i3tree/taniteval/tests/` (scratch) | lands via LANDING_READY |
| `RESULT.md`, `LANDING_READY.txt` | package | yes, until landed |
| `raw/*.log`, `raw/*.xml` (pytest logs, JUnit, mutation / isolation / exposure logs) | package; originals in `C:/lgt/i3logs/` | package is the banked copy |
| `raw/scripts/{mutate.py, run_mutations.sh, measure_refcv6_loader_exposure.py, isolation_probe.py}` | package | yes, until landed |
| Scratch trees `C:/lgt/i3tree` (clean `37086c3`), `C:/lgt/i3mut` (mutation tree, reverted) and `C:/lgt/i3b711` (clean `b711411`) | C: only | disposable; nothing unique |
