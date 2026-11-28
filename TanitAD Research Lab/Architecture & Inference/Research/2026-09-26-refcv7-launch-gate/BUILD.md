# refcv7 launch gate: BUILD status (living file, banked 2026-09-27 ~10:00 Berlin; the evidence and verdicts are in RESULT.md)

**Owner:** the gate agent. Nothing is staged or committed by me. The Master Mind lands it via `LANDING_READY.txt`.
**Base:** tip `3cca805`, which is NEW-2 `cef9709` plus its Thor gate records.
- All nine repo files are NEW except `.gitignore`.
- The tip blob of `.gitignore` is `cd58f3b6`, unchanged at every tip since 49f10b4. The package file is that blob plus 4 lines.

## Repo files (`code/fix/<repo path>`)

**`stack/scripts/launch_gate.py`** is the gate itself.
- Subcommands: `run`, `finalize`, `verify`, `exec`, `bind`, `suite-bind`, plus the internal `check`.
- It mints an HMAC token bound to five things: the commit; the tree sha256 over `stack/` + `taniteval/`; the argv sha256; the data manifest (including the `gate:*` inputs); and the tau record's sha256.
- Checks: G-HYG, G-DVB, G-LIVE (+ G-LIVE-PRES, G-MAP), G-CLOCK, G-EVAL, G-CKPT, G-SUITE, G-SUITE-PINNED, G-MAP-OVERFIT, G-BOX-OVERFIT.

**`stack/scripts/closure_run.py`** (NEW, SPEC_REFCV7 A11) is the code-closure wrapper for a binding harness run. It is stdlib only. It records:
- every module in `sys.modules` whose file lies anywhere in the tree;
- tree `.py` files the process opened;
- the tree modules it looked for and probed, and the names it looked for and did not find;
- the data files opened under `--data-root`;
- `closure_sha256`, the argv sha, the exit status (a crash still writes its record), the PASS JSON sha, and the environment (torch, timm, CUDA, cuDNN).

**`stack/scripts/launch_gate_refcv7.py`** is the SPEC's file name. It pins the refcv7 profile.

**`stack/scripts/launch_gate_arms.py`** runs the deliberate-regression arms (CAUGHT / MISCREDITED / ESCAPED / NOT-APPLICABLE) and the token arms.

**`stack/ops/sup_refcv7.sh`** is the supervisor.
- It launches ONLY through `launch_gate.py exec`, after `verify` returns MATCH.
- It refuses `TRAIN_CMD`.
- It closes `200>&-` on every child.
- NEW: it unsets its manifest variables before sourcing, so a caller's exported `STEPS` or `TRAIN_CMD` never reaches them. This was MEASURED on Thor 2026-09-27.

**`stack/ops/launch_gate_thor_env_failures.json`** is the pinned Thor environment-failure list: 165 ENV ids + 1 FLAKY id with its cited record, in 33 files. Its sha256 is pinned in the profile.

**`stack/ops/runs.d/refcv7-r101-s0.argv.json`** is THE canonical launch argv: 140 tokens, `argv` sha256 `52c00af7…`.
- It also carries the `todo_box_head` and `todo_map_lift` blocks and `launch_prep`.
- The argv tokens are unchanged by the TODO blocks.

**`stack/tests/test_launch_gate.py`** has 126 tests: 126/126 on Thor (3cca805 tree, stage 8, 0 skipped).

**`.gitignore`** adds `launch_gate.key` and `*.gate.key`.

## What a refcv7 token requires

**Argv rules**, re-derived inside `finalize` so a stand-alone `finalize` cannot skip them:
- `--residual-prior ha0_ext_pose` + `--ego-history`. `off`, `ha0_ext` and `cv_yawrate` are REFUSED, each with a regression arm.
- The three selection mechanisms are ON and BUILT (`check_refcv7_required`).
- tau equals the banked record (10ca19db…): the file flag, the `{path, sha256, tau}` stamp in `config.json`, and the label content.
- NEW-2 levers, one admissible value each:
  - `--map-hires on`
  - `--bev-source map_hires_pool`
  - `--map-hires-decision-rule prior_corrected`
  - `--map-hires-x-max-m 100`
  - `--map-hires-y-half-m 30`
  - `--map-hires-grad-ckpt on`
  - `--bev-planner-crop-m 60 16`
  - `--w-map 0`
  - `--w-map-hires` > 0
- No DrivoR-T flag, prefix, or built module.

**⛔ OPEN ITEMS (`PROFILES["refcv7"]["open_items"]`).** While any item is open the verdict is at best INCOMPLETE, with each item named:
- **BOX-HEAD:** the R1–R4 flags, plus +R6 `--slot-dn-groups 5` if A10.1 decides it. The box-head builder names the flags; the Master Mind decides MAIN vs +R6.
- **MAP-LIFT:** one or two NEW-2 lift flags. The early non-binding G-MAP-OVERFIT MAIN failed at step 1,000 (lane 0.417, edge 0.076 vs 0.50; INHERITED from the Master Mind). Next is the prereg §9 lift lever.
- Closing an item = its flags land as `required_values` and in the canonical argv, and the item is deleted.

**G-LIVE** (Thor GPU smoke, the Master Mind's run):
- `ga_*` gradient reach, including `ga_mh_*`, in every logged row. The smoke runs at least 2 × `log_every` steps.
- The grad-unreachable probe runs AFTER the G-CKPT snapshot.
- G-LIVE-PRES: fraction of slots with σ ≥ 0.5 must be below 0.5 on BOTH slot heads.
- G-MAP per LOGGING_SPEC_MAP10: per-class gradients; 40 IoU keys + 40 raw keys + counts + 40 loss shares; the decision rule; the class-weights stamp.
- The Training Watch (`taniteval/tools/training_watch/build_watch_refcv7.py`) is built by the gate from the smoke's own `metrics.jsonl`. A non-zero exit is a FAIL.
- Cost: above 8.0 s/step (+25 % over 6.4) gives **PI-DECISION**, not FAIL.

**G-DVB (static):**
- The NEW-2 branch is built and supervised, and names its weights.
- The per-class × band eval keys are read from the trainer's OWN `_eval_row_from_acc` run on the BUILT model. This is new: the landed trainer declares no list.
- The spelling check runs at the DECLARED extent (`band_keys_for_rows(1000)`), not against the /2 `BAND_KEYS` constant.

**G-MAP-OVERFIT / G-BOX-OVERFIT:**
- The PASS JSON is re-judged against the prereg literals. For G-BOX-OVERFIT, criteria 1–6 are recomputed from each arm's FINAL row, and the record's pass flags never decide.
- **plus its closure record (A11).** Every blob is recomputed at the launch tree (git at the launch commit when a file is outside it). The closure_sha and argv_sha must be consistent, `binding: true`, exit 0, and no looked-for module may now resolve. No shadowing and no Python children. Every file the harness imports at module level must be recorded. Data is re-hashed on the launch host. torch/timm/CUDA/cuDNN must equal the launch host's, else FAIL.
- The PASS text says it covers the head IN ISOLATION; the launch model's wiring is G-LIVE's and G-DVB's job.
- **Verification runs ON THOR** (Master Mind 2026-09-27): env and data are launch-host facts. The box loader is vendored into `stack/` by the box builder.
- G-BOX-OVERFIT's schema is **PROVISIONAL** until the box package lands. It reads `g_box_overfit.py`'s record as it stands in `…/2026-09-27-refcv7-box-head/code/new/`.

**G-SUITE:**
- (a) The Thor full-suite verdict, bound by `suite-bind` to the commit and the CAND tree digest. Failures are admitted only if they are pinned ENV ids, or FLAKY with a cited record.
- (b) G-SUITE-PINNED: a `git archive` of the FULL launch commit, into which the gate builds a THROWAWAY repo (objects via `alternates`, LF bytes; never tanitad-push's shared index; no `GIT_*` and no `MKL_NUM_THREADS` in the test environment).
  - The repo is armed like a clone (`tools/secret_scan.py --install-hook`), and `SECRET_SCAN_FULL=1` is set.
  - Every pinned test must PASS or SKIP for a registered (test id, reason) pair. The 5-entry registry is IN the refcv7 profile (RESULT.md sec. 3).
  - Every guard's mutation arm must be CAUGHT.

## Evidence

**Thor, CPU only, in the fresh dirs `/home/nvidia/gate_lg_0508` and `gate_lg_0518`, tree = `git -c core.autocrlf=false archive 3cca805` + the gate files, md5-verified:**
- Unit tests: 126/126 (stage 8, the final files).
- The CANONICAL argv: G-HYG **PASS**, G-DVB **PASS**. The token is INCOMPLETE: the other checks are not run here, and the two open items are named.
- tiny_live: G-LIVE PASS and G-CKPT PASS (the probe-after-snapshot fix is confirmed).
- Arms on the canonical argv: 7/7 CAUGHT in stage 5; stage 6 adds `map_drivable_only_static`.
- Tiny arms: 11 CAUGHT + 8 NOT-APPLICABLE.
- The gate's self-mutation audit: see RESULT.md for the final count.

**Dev box, CPU:**
- Unit tests: 123/123.
- Self-mutation audit: 51/51 CAUGHT on the 05:15 snapshot. The final 56-mutation audit is recorded in RESULT.md.
- G-SUITE-PINNED rehearsal on the 3cca805 full archive: see RESULT.md.

## Named blockers (what a PASS still needs)

1. **BOX-HEAD / MAP-LIFT flags.** These are the open items above. Owners: the builders, then the Master Mind.
2. **The binding G-MAP-OVERFIT and G-BOX-OVERFIT runs** under `closure_run.py --binding`, on the final map and box paths. Owner: the Master Mind (Thor GPU).
3. **The G-LIVE / G-CKPT / G-EVAL GPU smoke** on the canonical argv. Owner: the Master Mind (Thor GPU).
4. **The skip registry for G-SUITE-PINNED (b):** 5 entries IN the profile (RESULT.md sec. 3). Landing is the Master Mind's approval.
5. **The Thor full-suite verdict at the LAUNCH commit**, stamped by `suite-bind`. Owner: the Master Mind.
