# refcv7 launch gate: RESULT (2026-09-27, the gate agent)

**Package:** `TanitAD Research Lab/Architecture & Inference/Research/2026-09-26-refcv7-launch-gate/`
**Binding spec:** `Project Steering/SPEC_REFCV7.md` §2 plus amendments A1–A11. A11 is §16, the code-closure binding, registered at `03e6ed9`.
**Base:** tip `3cca805` (NEW-2 `cef9709` + its Thor records). Nothing is staged or committed by me; the Master Mind lands the package via `LANDING_READY.txt`.

## Headline

⚠️ **ESCALATE INTEGRATION.** The gate is 9 repo files. Full files sit under `code/fix/`, with base and new blobs in `LANDING_READY.txt`. They are **not on the branch until the Master Mind lands them.**

**On THE canonical refcv7 argv at `3cca805`, G-HYG PASSES and G-DVB PASSES** (MEASURED, Thor CPU, `raw/2026-09-27-thor-3cca805/stage7_gate_lg_0928/`). The 9 NEW-2 lever FAILs seen on the pre-NEW-2 tip are gone.

Every other check is built, unit-tested (126/126 on Thor) and mutation-tested (the gate's own audit: 59/59 CAUGHT on Thor, on the exact banked files).

The token is **INCOMPLETE by construction** until these exist:

| still needed for a PASS | owner |
|---|---|
| **OPEN ITEM BOX-HEAD:** the R1–R4 flags, and +R6 if A10.1 decides it. The early G-BOX-OVERFIT failed both MAIN and +R6 (INHERITED, Master Mind 09:20) | box builder → Master Mind |
| **OPEN ITEM MAP-LIFT:** one or two lift flags, or a PI step-budget decision. The early G-MAP-OVERFIT failed at 1,000 steps | NEW-2 builder → Master Mind / PI |
| the BINDING G-MAP-OVERFIT and G-BOX-OVERFIT runs, each under `closure_run.py --binding` | Master Mind (Thor GPU) |
| the G-LIVE / G-CKPT / G-EVAL GPU smoke of the canonical argv | Master Mind (Thor GPU) |
| G-SUITE (a): the Thor full-suite verdict at the LAUNCH commit, stamped by `suite-bind` | Master Mind |
| G-SUITE-PINNED (b): approval of the 5-entry skip registry (§3), which is IN the profile | Master Mind (landing = approval) |

## 1. What the gate checks (the refcv7 profile)

**Token.** An HMAC token binds:
- the commit;
- a sha256 over `stack/` + `taniteval/`;
- the argv sha256;
- the data manifest, including every `gate:*` input and the tau record's sha256.

**Verdicts.** PASS, FAIL, INCOMPLETE or PI-DECISION. `finalize` re-derives the argv rules, so a stand-alone `finalize` cannot skip them.

**Argv rules:**
- `--residual-prior ha0_ext_pose` plus `--ego-history`. `off`, `ha0_ext` and `cv_yawrate` are refused.
- The three selection mechanisms must be ON and BUILT.
- tau is the banked record. `--nav-compliance-tau-file` must carry its `{path, sha256, tau}` stamp.
- The NEW-2 levers each have exactly one admissible value.
- No DrivoR-T flag, prefix or module.
- **OPEN ITEMS** (`PROFILES["refcv7"]["open_items"]`): BOX-HEAD and MAP-LIFT. Each is named in the token, and the verdict can be no better than INCOMPLETE while either is open.

**G-DVB (static):**
- The NEW-2 branch is BUILT.
- The per-class × band eval keys are read from the trainer's OWN `_eval_row_from_acc` run on the built model. The landed trainer declares no key list.
- The spelling check runs at the declared extent via `band_keys_for_rows(1000)`. Before this fix it compared against the /2 constant `BAND_KEYS`, and would have failed the real smoke.

**G-LIVE (GPU smoke):**
- `ga_*` (incl. `ga_mh_*`) in every logged row.
- The grad-unreachable probe runs AFTER the G-CKPT snapshot.
- G-LIVE-PRES reads the LAST layer of BOTH slot heads. A9 R2's `aux` layers are informative and never count as a head.
- G-MAP per LOGGING_SPEC_MAP10.
- The Training Watch is built by the gate from the smoke's own metrics. A non-zero exit is a FAIL.
- Above 8.0 s/step the verdict is **PI-DECISION**.

**G-MAP-OVERFIT / G-BOX-OVERFIT:**
- The PASS JSON is re-judged against the prereg literals. The box criteria 1–6 are recomputed from each arm's FINAL row, and the record's pass flags never decide. The box schema is PROVISIONAL until the box package lands.
- **plus its A11 closure record:**
  - every blob is recomputed at the launch tree (git at the launch commit for files outside it);
  - `closure_sha` and `argv_sha` must be consistent; the record must be `binding: true` with exit 0;
  - no looked-for module may now resolve;
  - no shadowing and no Python children;
  - every file imported at module level must be recorded;
  - data files are re-hashed;
  - torch/timm/CUDA/cuDNN must equal the launch host's, else FAIL.
- The PASS text says the overfit covers the head IN ISOLATION; wiring is G-LIVE's and G-DVB's job.
- Verification runs ON THOR.

**G-SUITE:**
- **(a)** The Thor full-suite verdict, bound by `suite-bind`. Failures are admitted only if they are pinned ENV ids, or FLAKY with a cited record.
- **(b)** G-SUITE-PINNED:
  - a full-commit `git archive` with a THROWAWAY repo inside it: `alternates` written as LF bytes, never tanitad-push's shared index, no `GIT_*` and no `MKL_NUM_THREADS`;
  - the repo is armed like a clone (`secret_scan.py --install-hook`), with `SECRET_SCAN_FULL=1`;
  - every pinned test must PASS or SKIP for a registered (id, reason) pair;
  - every guard's mutation arm must be CAUGHT;
  - the RAM floor abort is VERIFIED.

## 2. Evidence (MEASURED unless stated)

**Thor, CPU only.** Fresh dirs, trees md5-verified, `tanitad.__file__` asserted. Tree = `git -c core.autocrlf=false archive 3cca805` + the 8 gate files.

| run | result | artifact |
|---|---|---|
| unit tests, final files (stage 8) | **126 passed, 0 skipped** | `raw/2026-09-27-thor-3cca805/stage8_gate_lg_0944/out/unit/` |
| gate self-mutation audit, final files, 59 mutations (stage 8) | **59/59 CAUGHT** (baseline 126 passed) | `raw/2026-09-27-thor-3cca805/stage8_gate_lg_0944/out/gate_self_mutation.json` |
| **canonical argv: G-HYG / G-DVB** (stage 7) | **PASS / PASS** | `raw/2026-09-27-thor-3cca805/stage7_gate_lg_0928/out/model_r7/` |
| tiny rig: G-LIVE / G-CKPT (stage 7) | **PASS / PASS** (the probe-after-snapshot fix confirmed) | `…/out/tiny_live/` |
| arms on the canonical argv (stage 7) | **8/8 CAUGHT**: unwire_selection_term, required_on_missing, tau_differs, tau_file_missing, drivort_flag, residual_prior_ha0_ext, residual_prior_cv_yawrate, map_drivable_only_static | `…/out/arms_r7c/arms_result.json` |
| tiny-rig arms (stage 7) | **11 CAUGHT + 8 NOT-APPLICABLE**; token arms **6/6 CAUGHT** | `…/out/arms_tiny/arms_result.json` |
| **A11 dry run:** the REAL `map_hires_overfit.py --help` under `closure_run.py`, judged by `judge_closure` (stage 7) | **AS EXPECTED**: 34 modules (33 `stack/tanitad`); all 34 module-level files recorded; 0 outside the tree; env equals the host (torch 2.13.0+cu130, timm 1.0.29, CUDA 13.0, cuDNN 92000); only the two dry-run refusals | `…/out/closure_probe/probe.json` |
| self-mutation audit, 56 mutations (stage 6, before the last 3 fixes) | **56/56 CAUGHT** | `/home/nvidia/gate_lg_0518/out/gate_self_mutation.json` |

**Dev box, CPU:**

| run | result | artifact |
|---|---|---|
| unit tests (before the skip-registry test) | 125 passed | `raw/2026-09-27-devbox-3cca805/devbox_unit_full.log` |
| self-mutation audit, 51 mutations (05:15 snapshot) | **51/51 CAUGHT** | `…/selfmut_snap_0515/selfmut.json` |
| **G-SUITE-PINNED rehearsal r2** (the final gate on the full 3cca805 archive) | **33/33 pinned files ran, 0 failures.** The only skips were the 5 now registered (§3); `test_secret_scan.py` passed, including the full tracked scan and the hook tests. The guard mutation audit was then **ABORTED by the RAM floor** (3.88 GB < 4.0 at 09:43 Berlin: other agents' pytest suites and refav1/refe evals were running). The abort was **VERIFIED** (`stopped: true`) | `…/pinned_rehearsal_3cca805_r2/` |
| G-SUITE-PINNED rehearsal r1 | the guard mutation audit **13/13 CAUGHT** on the same archive. `test_secret_scan.py` timed out there; a verbose re-run of that file in the identical environment **passed in 123 s**, and r2 passed it | `…/pinned_rehearsal_3cca805/`, `…/rerun_secret_scan_3cca805/` |

**Caveat on the rehearsal.** No SINGLE rehearsal ran uninterrupted end to end. The two halves were each measured complete, in two runs on the same archive and commit: all pinned files (r2) and the guard mutation audit (r1). The dev box is shared and fell below the 4.0 GB floor both times. The real G-SUITE-PINNED runs at the launch commit.

## 3. The G-SUITE-PINNED (b) skip registry, IN the refcv7 profile

Each entry is a (test id, reason) PAIR; a reason admits ITS test only. Landing this is approval; to strike an entry, delete its tuple.

| test | reason | why it is admissible |
|---|---|---|
| `test_anchor_prefilter::test_prefilter_is_BIT_EXACT_on_every_candidate_it_decodes` | Windows-CPU BLAS gemm tilings | A per-backend claim. ⚠️ On Thor the same test is a PINNED ENV failure, so it passes on NEITHER host. |
| `test_anchor_prefilter::test_replicates_the_measured_survivor_counts_on_the_canonical_val` | `refc_anchors_small64.pt` absent | **KNOWN GAP.** The fixture is NOT in the repo at 3cca805: `cat-file -e` fails while a same-breath control resolves. The test runs nowhere from a git archive. |
| `test_kingate_contract::test_gate_k_1_must_be_reachable_…` and `::test_the_zero_flag_path_must_be_declared_bit_identical` | the T1 gate hook is not written yet | **KNOWN GAP** (Decisions/2026-09-05 M23/M24). The tests activate when the hook lands. **Owner: UNASSIGNED; the Master Mind assigns.** |
| `test_refa_v1_precision::test_cuda_bf16_autocast_tiny_step_has_finite_loss_and_grad_norm` | no CUDA device | The dev-box half is CPU-only by rule. |

## 4. Bugs found by measuring in this pass (each fixed, and each pinned by a test + a mutation)

1. **The RAM-floor abort did not kill its child.**
   - MEASURED: "ABORTED" was written at 03:53:54Z. The `check` child ran on for 8 minutes and overwrote that ERROR with a complete FAIL at 04:01:34Z.
   - Cause: `_kill_tree` swallowed every failure. A taskkill under memory pressure did nothing, and nobody checked.
   - Fix: `_stop_child` VERIFIES the stop (tree kill, then TerminateProcess/SIGKILL) and records each step. The r2 abort was verified `stopped: true`.
2. **The alternates file was written in text mode**, which gives CRLF on Windows, so git looked for `objects\r` and HEAD never resolved. It is now written as LF bytes; a GIT_DIR decoy must not move.
3. **`sup_refcv7.sh` read a caller's exported `STEPS` as the manifest's.** MEASURED on Thor, where my own runner's selector triggered it. The script now unsets its manifest variables before sourcing.
4. **`closure_run` flagged Debian's `/usr/lib/python3.12/sitecustomize.py`** as shadowing; it is a symlink into `/etc`. It also recorded its own source when linecache read it for a warning. Both are fixed.
5. **G-DVB / G-MAP against the LANDED NEW-2:** no key list, and the /2 band constant. See §1.
6. **Every gate child now gets `stdin=DEVNULL`.** A gate test must never read the operator's terminal.
7. **A self-audit re-reads its source for every mutation**, so editing mid-audit contaminated one run. It was killed; audits now run on frozen snapshots or in fresh Thor dirs.
8. ⚠️ **My own trap, disclosed.** A `pgrep -f` selection on Thor matched my own ssh shell and killed it. Only my shell and its child died; no bystander was affected. I had documented this trap myself and still repeated it. Selection now matches the exact interpreter path over `/proc`.

## 5. Box-harness integration

These went to the builder through the Master Mind:
- `--launch-argv` must accept the canonical OBJECT file.
- The argv hash uses `json.dumps` defaults; the gate accepts both forms.
- The audit loader is vendored into `stack/`, so closure verification stays on Thor.

## 6. The FLAKY pinned test: a one-line fix (proposed, not applied)

`stack/tests/test_flagship_v4.py::test_graft_tau_to_zero_makes_the_class_posterior_one_hot`: Thor measured 1/6 failing as-is and 0/6 when seeded.
- Proposed fix: insert `torch.manual_seed(0)` as its first statement.
- Until then it is admitted only through its cited FLAKY record.

## 7. Manifest

See `LANDING_READY.txt` for every path, with base and new blobs.

**Repo files (9, under `code/fix/`):**
- `stack/scripts/{launch_gate,launch_gate_refcv7,launch_gate_arms,closure_run}.py`
- `stack/tests/test_launch_gate.py`
- `stack/ops/{sup_refcv7.sh,launch_gate_thor_env_failures.json,runs.d/refcv7-r101-s0.argv.json}`
- `.gitignore`

**Package tools (22, under `code/`).**

**Evidence:**
- `raw/2026-09-27-thor-3cca805/` (stage 7 and stage 8);
- `raw/2026-09-27-devbox-3cca805/` (rehearsals, secret-scan re-run, 05:15 audit, unit log).

⚠️ **Single-copy items:** the Thor gate dirs `/home/nvidia/gate_lg_0{928,944}/` (their evidence is banked under `raw/`), and the dev-box scratch trees `C:/Users/Admin/lg0926/{work,c3c,snap_0515,thor_stage*}` and `C:/lgs/f3cca80520b`. All are regenerable from the package and the commit.
