# SFT-4R readiness (dev box, 0 GPU) -- 2026-10-16

Package: `raw/2026-10-16-sft4r-readiness/` under `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`. Base: branch tip `6c3d469c` (mirror and remote agreed at 22:50).
Nothing was launched, nothing touched on the pod, nothing staged, committed or pushed. Evidence classes: **MEASURED** = produced by this session, artifact path given; **INHERITED** = from the previous
agent's files / the chain header, not re-verified (pod facts are all INHERITED: this session may not touch the pod).

## 0. Headline

* **The runaway is found, fixed at its source, and pinned.** Root cause: `check_suite` exported `REFE_GATE_NESTED=1` and nothing ever read it, and `test_sft4r_gate.py` (a G-SUITE entry of the profile it tests)
  ran `launch_gate.py run --profile refe_sft4r` with all checks, which reaches G-SUITE, which runs `test_sft4r_gate.py` again. See section 2.
* **The real `refe_sft4r` G-SUITE now terminates on the dev box:** `launch_gate.py suite-prerun --profile refe_sft4r` PASS in 59 s, 4/4 scripts rc 0, 0 skips, peak 15 live descendants
  (MEASURED, `logs/e2e/suite_prerun_refe_sft4r.txt`; re-run on the final tree by the queue as `e2e_suite_prerun_refe_sft4r`, section 7).
* **INTEGRATION NEEDED (Master Mind):** (1) run `apply_all.py` on an LF tree of the branch tip, (2) land the six files, (3) append `AMENDMENT_3_DRAFT.md` to `eval/PREREG_SFT4R.md`
  (the code cites an "Amendment 3" that the tip's PREREG does not contain, section 8), (4) ship + run the pod sequence of section 9.

## 1. The change set (six files; `apply_all.py` re-applies it byte-exact, `SHA256.json` holds pre/post sha256)

| file (under `refe/`) | change | origin |
|---|---|---|
| `launch_gate.py` | the `refe_sft4r` profile (pinned argv, recipe, `data_expect`, G-EGR, `SFT4R_SUITE`) | previous agent (`patch/patch_launch_gate.py`); **+ `test_suite_nesting.py` in `SFT4R_SUITE`** (`patch/patch_suite_nesting.py`) |
| `gate_checks.py` | `scorer_expect_rows` (G-DVB data rows + the re-derived schedule) | previous agent (`patch/patch_gate_checks.py`); **+ `check_suite` refuses to nest** (`patch/patch_suite_nesting.py`) |
| `test_sft4r_gate.py` (new) | profile tests, 8 sections | previous agent; **section 7 `run` call restricted to `--only G-HYG,G-DVB,G-PREFLIGHT`, own `--key-file`, + section 8** (`patch/patch_test_sft4r_gate.py`) |
| `test_suite_nesting.py` (new) | the regression test with 5 mutation arms | this session |
| `sft4r_chain.sh` | v2 chain (section 6) | previous agent (no patcher existed; shipped as `post_p0/`); **+ precheck demands `refe/test_suite_nesting.py`** (`patch/patch_sft4r_chain_precheck.py`) |
| `sft4r_pod.sh` | `MODE=argv`, `rc=$?`, layout via `SFT4R_WS` | previous agent (`post_p0/`) |

Diffs: `diff/<file>.tip_to_final.diff` (everything vs the tip) and `diff/<file>.p0_to_final.diff` (only what THIS session added on top of the previous agent's version).
`apply_all.py <LF repo root> --dry-run` then without `--dry-run`: refuses a CRLF tree, refuses if the tip moved under a file (use `--patchers-only`, then compare to `diff/`), verifies all six post-image sha256.
MEASURED: applied to a fresh `git archive` of the tip it reproduces the tested tree with 0 differences (`C:/s4r/tools/treediff.py`, n_diff 0).

## 2. The recursion

**Symptom (Master Mind):** running the suite spawned ~80 python processes, RAM fell to 8.6 GB, killed by hand. (INHERITED)

**Call chain, quoted (the previous agent's tree "P0"; the tip's `gate_checks.py` has the same lines 42 earlier because P0 added `scorer_expect_rows` above them):**

1. `test_sft4r_gate.py:286` `rc = cli("run", "--profile", "refe_sft4r", "--args", ..., "--pkg", str(work / "nopkg"), ...)`: every check, no `--only`. The comment beside it said it "fails later, on the missing package/preflight". It does, but only AFTER G-SUITE.
2. `launch_gate.py:975` `cmd_run` -> `run_checks(...)` -> `launch_gate.py:688-689` (scorer branch) `if "G-SUITE" in want: emit(suite_check(prof, ...))` -> `launch_gate.py:841` `return C.check_suite(prof, ...)`.
3. `gate_checks.py:1335` (tip: 1293) `subprocess.run(argv, ...)` over `profile["suite"]`, which for `refe_sft4r` is `SFT4R_SUITE` (`launch_gate.py:165`) = `test_sft4r_chain.py`, **`test_sft4r_gate.py`**, `test_launch_gate.py --static --no-banked`.
   So step 3 runs step 1's file. Each level adds a `launch_gate.py` and a `test_sft4r_gate.py` (x2 processes on Windows: the venv launcher and the real interpreter).
4. `gate_checks.py:1332` (tip: 1290) set `REFE_GATE_NESTED="1"` on every suite script. **Nothing read it**: the only other mention is `gate_token.py:450`, the env-fingerprint ignore list. A guard variable that was set and never tested.

**Why the other profiles never did this:** the arm-table profiles (`refe_rls1`, `refe_aux_s1*`) refuse `run` with no `--arm-name` in `_profile_for` (`launch_gate.py:955`, "mints for ONE named arm") BEFORE any check runs.
`refe_sft4r` has no arm table (one arm: A and B share a pass), so nothing stopped it. The tip already knew the class and handled it by hand in three places (`test_launch_gate.py:1013, 1044, 1557`:
"G-SUITE excluded: it would recurse into this file"). The new test arrived without that care; the engine had no guard to catch it.

**Reproduction, bounded (MEASURED, `logs/repro/p0_test_sft4r_gate.txt`):** `python test_sft4r_gate.py` on P0 under `tools/bounded_run.py --max-children 14` was killed after 37.5 s at 17 live descendants, 0 survivors; the
tree dump at the kill shows `test_sft4r_gate.py -> launch_gate.py run -> test_sft4r_gate.py -> launch_gate.py run -> test_sft4r_gate.py -> launch_gate.py run -> test_sft4r_chain.py`.

**Fix, two layers:**

| layer | where | what |
|---|---|---|
| the call | `test_sft4r_gate.py` section 7 | `--only G-HYG,G-DVB,G-PREFLIGHT` (the pattern `test_launch_gate.py` uses), own `--key-file` under the test's temp dir (the default key file is CREATED on first use: a side effect on whichever box ran the test) |
| the engine | `gate_checks.check_suite` | the sentinel now carries the JSON list of suite entries running ABOVE the process (id = sha256[:16] of resolved script path + args, or of a raw argv). `check_suite` spawns NOTHING and returns FAIL when (a) an entry it is about to run is already on that list, or (b) the list is already `SUITE_NEST_MAX` = 2 deep. Legacy `REFE_GATE_NESTED=1` (still exported by `raw/*/run_suite_scripts.sh` and the pod timing scripts) reads as depth 1; an unreadable value FAILS CLOSED. |

One nested level is allowed on purpose: `test_sft4r_gate.py` section 6 calls `run_checks` with a STUB suite from inside the real `refe_sft4r` G-SUITE run (it is a suite entry), so a blanket "any nesting refuses" would break the gate on the pod. (`test_gate_suite_prerun.py` also nests synthetic suites but is in no profile's suite.)

**Regression test `test_suite_nesting.py` (88 checks, 9.9 s, MEASURED `logs/t1/test_suite_nesting.txt`).** A: in-process unit checks incl. every one of the 44 (profile, entry) pairs of the real profiles: re-entry is refused, an empty stack refuses nothing.
B: real spawns on a toy suite through `run_bounded` (timeout, descendant cap 16, kill by explicit PID only, and a FUSE inside the probe: the 5th execution of any probe stops without recursing, so even a broken engine is bounded by construction):

| experiment | guarded engine (MEASURED) |
|---|---|
| cycle: P1's suite contains P1 | ledger `[P1]`, inner `check_suite` FAIL "refuses to nest ... ALREADY RUNNING", outer PASS, peak 3 descendants |
| chain P1 -> P2 -> P3 | ledger `[P1, P2]`, P2's suite refused on depth, P3 never runs, peak 5 |
| legacy sentinel `1` | plain suite PASS, peak 3 |
| unreadable sentinel | refuses, spawns nothing, peak 1 |

C: MUTATION. The same judge against a temp copy of `gate_checks.py` with one line of the guard removed must go RED (MEASURED, all bounded: 0 watchdog kills, peak <= 11):

| mutation | judge | what the toy did |
|---|---|---|
| guard branch removed (`if False and refused`) | RED, 6 reasons | cycle ran to the FUSE (5 executions) |
| identity rule removed | RED, 2 | cycle ran twice (the depth cap caught the second) |
| depth cap removed | RED, 3 | chain ran P3 |
| sentinel not propagated (= the pre-fix `REFE_GATE_NESTED="1"`) | RED, 6 | cycle ran to the FUSE |
| unreadable sentinel fails open | RED, 1 | the garbage-sentinel suite spawned |

`test_sft4r_gate.py` section 8 adds the real-profile proof without any spawn: with the sentinel holding this file's own id, `check_suite(PROF)` returns FAIL "refuses to nest" while `subprocess.run` is replaced by a function that raises.

**Defence in depth (MEASURED, `logs/mut/engine_guard_only_test_bug_reintroduced.txt`):** P0's `test_sft4r_gate.py` (the unrestricted `run` call put back) against the patched engine: finite, `ZZTEST_PASS 149`, 67 s, peak 17 descendants, 0 kills.
The guard alone turns an infinite recursion into one extra level.

## 3. What the `refe_sft4r` profile checks (`launch_gate.py` `PROFILES["refe_sft4r"]`; `test_sft4r_gate.py` pins each with a negative twin)

| check | asserts |
|---|---|
| G-HYG | the scorer trainer (`main`, `epo_refusals`, `_g_probe`, `aux_s1_refusals`) reads no namespace attribute its parser does not declare |
| G-DVB | the **whole argv is pinned token for token** (`SFT4R_ARGV`, diffed against `MODE=argv bash refe/sft4r_pod.sh`); the EFFECTIVE recipe (lr 1e-4, warm-up 50, batch 8, accum 8, epochs 1.0, `max_updates` 0, eval-every 400, seed 0, `expreward`, lam 1.0, tau_s 0.1, lane weight 1.0) because a parser default can move under a pinned argv; the trainer's own refusals; no EPO, no aux head, no `--max-updates`; exact flag spelling (argparse accepts abbreviations); and from the trainer's own `data` event: `train_sets == 174,912`, `pdm_train == 174,912`, `pdm_train_stale == 0`, held-out 3,137, teacher-lane >= 2,109, and the schedule re-derived = 2,733 updates |
| G-PREFLIGHT | the trainer's `--preflight` verdict row consumed and bound into the token by sha256 |
| G-SUITE | `SFT4R_SUITE` (section 2) all PASS tokens, 0 skips |
| G-EGR | `hf_egress.py gate`'s JSON: main push verified by the remote listing AND the regression push refused BY THE SERVER; `egr_require_ckpt: false` (no checkpoint is pushed) |
| G-CLOCK | NA, reason recorded (no label clock of its own) |

Deliberate-regression arm `data_incomplete` (SFT-4's 146,185 of 174,912): G-DVB RED. The token binds the commit, tree digest, argv, data fingerprints and both gate input files; a regression arm can never mint.
KNOWN LIMIT (pinned in the test): a directory is fingerprinted by listing (relpath + size), so a SAME-SIZE rewrite of a label file is invisible to the token; the chain pins the PDM tars by sha256 instead.

## 4. The launch argv (MEASURED: `MODE=argv SFT4R_WS=/workspace bash refe/sft4r_pod.sh`; the registered recipe's other flags are parser defaults, asserted by G-DVB)

```
--ckpt /workspace/data/refe_runs/vitl16_navtrain10_grow_tau0.3/model_final.pt --targets /workspace/data/refe_navtrain/train_grow --images /workspace/data/navtrain_pixels
--calib /workspace/data/refe_navtrain/train_grow/calib_table.json --onpolicy /workspace/data/refe_navtrain/onpolicy/sets --heldout /workspace/data/refe_heldout/sets
--heldout-logs /workspace/data/refe_heldout/heldout_logs.txt --workers 3 --lr 1e-4 --b-mode expreward --lam 1.0 --tau-s 0.1 --lane-head --lane-weight 1.0
--teacher-lane-train /workspace/data/refe_sft2/train_teacher_lane --teacher-lane-heldout /workspace/data/refe_sft2/heldout_teacher_lane
--lane-labels-heldout /workspace/data/refe_sft2/heldout_lane --pdm-labels-heldout /workspace/data/refe_sft2/heldout_pdm
--pdm-labels-train /workspace/data/refe_sft2/train_pdm --require-pdm --out /workspace/data/refe_sft4r
```
(one line in the file; wrapped here.) Trainer: `refe/scorer_finetune.py`. Parity: the canonical corpora are used as registered, nothing is re-selected.

## 5. Chain v2 (`sft4r_chain.sh`, the previous agent's rewrite): what each change fixes (read from the diff; INHERITED for the pod facts)

| v1 at the tip | v2 |
|---|---|
| no launch gate (BINDING since 2026-09-26) | stage `gate` mints the `refe_sft4r` token and VERIFIES it (never greps); verified AGAIN right before the trainer |
| `PDM train` completeness had no expected count; held-out lane check was `> 0` | `--expect 174912` / `3137`, held-out lane and teacher-lane `== 3137` (SFT-4 launched on 146,185 of 174,912 because "complete" was never computed) |
| hard-coded `/workspace/refe-sft4r/run`, no tree check | shipped package `/workspace/refe-plan-sft4r`, tree == `SHIP_MANIFEST.json` (the token binds the tree, so must the chain); stage markers carry the manifest sha |
| no lock, no wait: it would start on a busy pod | one chain at a time (`flock`); `wait_clear`: no other chain's lock held AND the GPU idle on 3 consecutive polls (an unreadable `nvidia-smi` counts as busy) |
| HF auth by `HF_TOKEN` (the pod has none) | `hf_egress.py --token-file`, read in place |
| `sft4r_pod.sh run` exit status was the `echo`'s (always 0): "RUN EXIT" could never be non-zero | `rc=$?` kept, `exit $rc` |
| no post-run check | `run_check`: `ZZSFT_DONE`, `updates == 2733`, `train_sets == 174912`, both models + states written |
| no disk probe | real 6 GB `dd` write test (never `df`: the volume is a per-pod quota) |
| a killed run could be re-launched silently | `trainstart` marker: a started-and-unfinished run REFUSES a re-run (no resume in `scorer_finetune.py`) until the out dir is archived |
| `sft4r_relabel.sh` implied | not run: the labels exist; the chain AUDITS them read-only |

`bash -n` on both scripts: OK (MEASURED). Only the two helpers (`gate_is`, `last_exit`) have an executable test (`test_sft4r_chain.py`, 20 checks); the stages that need the pod (GPU wait, HF pushes, preflight) have never run end to end:
**UNVERIFIED by this session; `SFT4R_STOP_AFTER=<stage>` is the chain's own dry run, section 9.**

## 6. Test table: every test that names an edited file, FULL form, short path tree `C:/s4r/p`, watchdog on every run

Forms: **gate** = the environment `check_suite` gives a script (sentinel = JSON stack holding the script itself; `1` on the unpatched tree). **alone** = no sentinel (how a person runs it; the form the runaway started from).
Every run: `bounded_run.py` (timeout, <= 20 live descendants, re-run once at 40 if exceeded and recorded, kills by explicit PID), `CUDA_VISIBLE_DEVICES=""`, `driverl-eval` venv, `-X utf8`.
A failure is re-run alone (serial), then on the unpatched tree of the same commit (`C:/s4r/b`, `git archive` of `6c3d469c`) to separate PRE-EXISTING from REGRESSION. `peak procs` = live descendants (a venv python counts 2).

<!--TEST_TABLE_BEGIN-->
**Generated 2026-10-16 23:36 by `tools/run_all.py`: 33 (test, form) pairs, 9 NOT clean.** Raw: `logs/SUMMARY.txt` (one line per run), `logs/q/<test>.<tree>.<form>.txt`.
Verdict words: PASS | PASS-ON-SERIAL-RERUN (failed under 4-way contention, passed alone) | PRE-EXISTING (fails on the unpatched tip tree too) | REGRESSION (passes unpatched) | NEW-TEST-FAILS.

| test | form | patched (rc / token / s / peak procs) | verdict | unpatched, if it failed |
|---|---|---|---|---|
| test_launch_gate--full | gate | rc 0 / `ZZLG_FULL_PASS 690` / 530.0 s / 8 | **PASS** |  |
| test_p6_chain | gate | rc 0 / `ZZP6CHAIN_PASS 341` / 510.2 s / 9 | **PASS** |  |
| test_gate_suite_prerun | gate | rc 0 / `ZZTEST_PASS 144` / 421.2 s / 17 | **PASS** |  |
| test_p2rep_chain | gate | rc 0 / `ZZTEST_PASS 353` / 237.6 s / 6 | **PASS** |  |
| test_p6_levers | gate | rc 0 / `ZZP6_PASS 145` / 344.7 s / 5 | **PASS** |  |
| test_rls1_chain | gate | rc 0 / `ZZTEST_PASS 512` / 137.1 s / 7 | **PASS** |  |
| test_s1c | gate | rc 1 / `NONE` / 97.0 s / 6 | **PRE-EXISTING (fails unpatched too)** | rc 1 AssertionError: (1, "-10-16T21:33:29 ZZS1AUX_UPSTREAM_NO_DONE rls1 (WA |
| test_aux_s1_wiring | gate | rc 1 / `NONE` / 67.1 s / 2 | **PRE-EXISTING (fails unpatched too)** | rc 1 AssertionError: the unmutated scorer_finetune.py FAILS its aux wiring  |
| test_aux_g12_bf16 | gate | rc 1 / `NONE` / 5.1 s / 2 | **PRE-EXISTING (fails unpatched too)** | rc 1 AssertionError: Invalid device id |
| test_s1aux_chain | gate | rc 1 / `NONE` / 63.0 s / 6 | **PRE-EXISTING (fails unpatched too)** | rc 1 AssertionError: ('data', 'navtest list', '-s1 H2-s1 N-s2 H2-s2 H1-s1 H |
| test_rl_r0c | gate | rc 0 / `ZZTEST_PASS 26  (1 unmutated suite + 25 ` / 80.5 s / 1 | **PASS** |  |
| e2e_suite_prerun_refe_sft4r | gate | rc 1 / `NONE` / 70.0 s / 13 | **NEW-TEST-FAILS (no unpatched counterpart)** |  |
| test_launch_gate--static | gate | rc 0 / `ZZLG_STATIC_PASS 645` / 46.5 s / 4 | **PASS** |  |
| test_launch_gate--static | alone | rc 0 / `ZZLG_STATIC_PASS 645` / 55.9 s / 4 | **PASS** |  |
| test_sft4r_gate | gate | rc 0 / `ZZTEST_PASS 152` / 20.3 s / 3 | **PASS** |  |
| test_sft4r_gate | alone | rc 0 / `ZZTEST_PASS 152` / 21.6 s / 3 | **PASS** |  |
| test_suite_nesting | gate | rc 0 / `ZZTEST_PASS 88` / 16.2 s / 13 | **PASS** |  |
| test_suite_nesting | alone | rc 0 / `ZZTEST_PASS 88` / 15.1 s / 13 | **PASS** |  |
| test_p2r_chain | gate | rc 0 / `ZZTEST_PASS 139` / 7.6 s / 5 | **PASS** |  |
| test_p2r_chain | alone | rc 0 / `ZZTEST_PASS 139` / 7.3 s / 5 | **PASS** |  |
| test_rl_s1b | gate | rc 0 / `ZZTEST_PASS 18  (1 unmutated suite + 17 ` / 19.6 s / 1 | **PASS** |  |
| test_p6_cpu_probe | gate | rc 0 / `ZZTEST_PASS 111` / 148.2 s / 8 | **PASS** |  |
| test_p6_overlap | gate | rc 0 / `ZZTEST_PASS 54` / 1.9 s / 3 | **PASS** |  |
| test_rls1_queue | gate | rc 0 / `ZZTEST_PASS 134` / 517.4 s / 8 | **PASS** |  |
| test_rls1_queue | alone | rc 0 / `ZZTEST_PASS 134` / 519.6 s / 7 | **PASS** |  |
| test_sft4r_chain | gate | rc 0 / `ZZTEST_PASS 20` / 1.9 s / 5 | **PASS** |  |
| test_sft4r_chain | alone | rc 0 / `ZZTEST_PASS 20` / 1.9 s / 5 | **PASS** |  |
| test_hf_egress | gate | rc 1 / `NONE` / 0.4 s / 1 | **PRE-EXISTING (fails unpatched too)** | rc 1 ZZTEST_FAIL 1/13 |
| test_hf_egress | alone | rc 1 / `NONE` / 0.5 s / 1 | **PRE-EXISTING (fails unpatched too)** | rc 1 ZZTEST_FAIL 1/13 |
| test_pdm_label_complete | gate | rc 0 / `ZZTEST_PASS 13` / 2.4 s / 1 | **PASS** |  |
| test_pdm_label_complete | alone | rc 0 / `ZZTEST_PASS 13` / 0.2 s / 1 | **PASS** |  |
| test_gate_gpu_probes | gate | rc 1 / `NONE` / 4.2 s / 1 | **PRE-EXISTING (fails unpatched too)** | rc 1 AssertionError: Invalid device id |
| test_gate_gpu_probes | alone | rc 1 / `NONE` / 4.1 s / 2 | **PRE-EXISTING (fails unpatched too)** | rc 1 AssertionError: Invalid device id |

<!--TEST_TABLE_END-->

Previous agent's baseline (INHERITED, different tree state, `C:/Users/Admin/sft4r_ready/logs/baseline/SUMMARY.txt`): pre-existing there too were `test_hf_egress` (same cause), `test_s1aux_chain` (missing fixture file, since supplied), `test_p2rep_chain` rc 127 and `test_rls1_queue` ("returned before S1 was over", a timing assertion).

## 7. What this does NOT prove

* **Nothing ran on the pod, and `launch_gate.py run` has not minted a real `refe_sft4r` token anywhere**: it needs the pod's data, the preflight JSON and the G-EGR JSON. On the dev box the REAL profile cannot mint by design (`test_sft4r_gate.py` section 6, `v5`).
* The pod-side facts (labels present with 174,912 / 3,137 rows, HF token file provisioned, `/workspace/refe-plan-sft4r` layout, `teacher_env.sh`) are INHERITED from the chain v2 header.
* Windows only. `test_suite_nesting.py` has POSIX branches (`start_new_session`, `os.kill`) that run only if a kill is needed; they were not exercised. `psutil` is optional there: without it the descendant cap is UNVERIFIED and the probe FUSE is the bound.
* CUDA was hidden for every run (0-GPU task), so `test_gate_gpu_probes`' CUDA probe cannot have run on a GPU; see its row.
* The `suite-prerun` evidence JSON in `logs/e2e/` is signed with a throwaway dev-box key: it is NOT valid for `run --suite-evidence` on the pod and must not be shipped.

## 8. PI / Master Mind decisions still needed

1. **Pre-registration text (Master Mind lands, PI can overrule):** append `AMENDMENT_3_DRAFT.md` to `eval/PREREG_SFT4R.md` BEFORE launch. The code cites an Amendment 3 that does not exist at the tip, and the PREREG still says 2,335 updates / 9.3 h (the lost run's 146,185-label epoch); the profile pins 2,733 updates (~10.9 h ESTIMATED). No lever moves.
2. **The pod:** `wait_clear` will block while any other chain (`refe_p2`, `refe_p2rep`, `rls1`, `s1aux`, `refe_p6`) holds its lock or the GPU is busy. Priority between those and the ~11 h SFT-4R run is a PI scheduling call, not made here.
3. **Standing, not new:** HF Pro allowance for the PDM-label push (`--max-push-bytes 2e9` is enforced by `hf_egress.py`); deploying an ADOPT result is the PI's call (PREREG_SFT4R "Next arm, by outcome").
4. **Optional (Master Mind):** the guard protects every profile, but `test_suite_nesting.py` is only in `SFT4R_SUITE`. Adding it to the common `SUITE` would put 10 s in every profile's G-SUITE and change every profile's suite binding (pre-run evidence), so it was not done.

## 9. The exact command sequence (NOT run)

Dev box (an LF tree of the branch tip: the mirror or a `git archive`; `$REPO` = its root, `$PK` = `$REPO/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan`):
```
python "$PK/raw/2026-10-16-sft4r-readiness/apply_all.py" "$REPO" --dry-run     # all six post-image sha256 must be reported equal
python "$PK/raw/2026-10-16-sft4r-readiness/apply_all.py" "$REPO"
cd "$PK/refe" && python test_suite_nesting.py && python test_sft4r_gate.py      # each prints ZZTEST_PASS; ~10 s and ~40 s
# land through the coordinated lander (scoped_commit / land.sh) -> <40-hex commit>
python refe/launch_gate.py ship-manifest --pkg "$PK" --commit <40-hex> --out "$PK/SHIP_MANIFEST.json"
python refe/launch_gate.py suite-prerun --profile refe_sft4r --pkg "$PK" --out <suite_prerun.json>   # OPTIONAL, CPU only, ~1 min: ZZSUITE-PRERUN-PASSZZ on the shipped bytes
```
Ship the package root (`refe/ code/ splits/ SHIP_MANIFEST.json`, plus the `raw/2026-10-04-sft4-smoke/` logs and `splits/navtrain.yaml` that the precheck lists) to the pod at `/workspace/refe-plan-sft4r` by tar/scp.
The pod has no git credentials; do not paste its address or the HF token anywhere.

Pod (the chain does gate -> preflight -> train itself; do NOT run `launch_gate.py run` by hand):
```
# dry runs, each stops after the named stage (precheck | inputs | egr | pdm_push | dd | preflight | gate; chain.log shows "stopping after <stage>"):
SFT4R_STOP_AFTER=precheck nohup setsid bash /workspace/refe-plan-sft4r/refe/sft4r_chain.sh > /workspace/data/refe_sft4r/chain.out 2>&1 < /dev/null &
#   then inputs (read-only audit); egr and pdm_push make real HF pushes; dd writes 6 GB; preflight and gate need the GPU idle (wait_clear)
# the run:
nohup setsid bash /workspace/refe-plan-sft4r/refe/sft4r_chain.sh > /workspace/data/refe_sft4r/chain.out 2>&1 < /dev/null &
# watch:  tail -f /workspace/data/refe_sft4r/chain.log        success = ZZSFT4R_CHAIN_DONE, failure = "REFUSED: stage=<stage>: <why>" or ZZSFT4R_CHAIN_FAIL
```
Re-running after a refusal resumes at the first stage without a marker. A run that STARTED training and did not finish refuses a re-run until `mv /workspace/data/refe_sft4r /workspace/data/refe_sft4r.killed_<k>` (no resume exists).

## 10. Manifest

| artifact | where | only in ONE place? |
|---|---|---|
| this package (patchers, post-images, diffs, SHA256.json, apply_all.py, tools, evidence, logs, READINESS.md, AMENDMENT_3_DRAFT.md, HANDOFF.md) | `E:/Projects/TanitAD/TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-16-sft4r-readiness/` (NOT staged) | **yes, plus the working copy `C:/s4r`** |
| tested patched tree | `C:/s4r/p/` (equals `apply_all.py` output; n_diff 0) | working copy |
| unpatched tip export | `C:/s4r/b/` (`git archive 6c3d469c`) | reproducible |
| raw logs of every run | `C:/s4r/logs/` (copied to `<package>/logs/`) | both |
