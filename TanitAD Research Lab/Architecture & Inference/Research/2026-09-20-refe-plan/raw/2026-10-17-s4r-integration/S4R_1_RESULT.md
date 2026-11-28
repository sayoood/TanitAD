# S4R-1 + SO-A16 -- SFT-4R readiness integrated onto the tip, the open defect root-caused, the token-file path made a required env var (dev box, 0 GPU) -- 2026-10-17

Base: branch `agent/arch-inf-20260803`, tip `3b8c39f5bbea1c5a11668d888a046b875fee7358` (read-only mirror `C:/Users/Admin/tanitad-push`). Nothing was staged, committed or pushed; nothing touched on the pod.
Evidence classes: **MEASURED** = produced by this session, artifact path in this directory; **INHERITED** = from the 2026-10-16 readiness package, not re-verified.

## 0. Headline

* **INTEGRATION NEEDED (Master Mind): land the 9 files of `paths.json` (8 under `refe/` + `eval/PREREG_SFT4R.md`) and this record directory.** They are full-content post-images built from the PRISTINE tip by `build_final.py`
  (every patcher asserts its anchor; `SHA256_BUILD.json`). Per-file diffs against the tip: `diff/`.
* The 2026-10-16 patch set **rebased cleanly except one anchor**: `patch_gate_checks.py` named `scorer_dvb_rows(ns, data)`; `e2d4422` (safety-objective) made it `scorer_dvb_rows(ns, data, profile)`. Rebased copy: `patch/patch_gate_checks.py`.
  `launch_gate.py` (`daa4568`, `e2d4422`) took `patch_launch_gate.py` and `patch_suite_nesting.py` unchanged. MEASURED: `build_final.py` rc 0 on a `git archive` of the tip.
* **The open defect is root-caused and fixed (section 2).** It is NOT a flake and NOT the nesting guard misfiring: `test_sft4r_gate.py` section 6 was not independent of the ambient G-SUITE nesting depth.
* **`sft4r_chain.sh` now takes the HF token-file path from the REQUIRED env var `SFT4R_HF_TOKEN_FILE`** (no default path in source, fail closed, first line of the precheck). The chain, its tests and its diffs now carry no token-file path, so
  they are banked in full (`refe/sft4r_chain.sh`). The 3 files of `raw/2026-10-16-sft4r-readiness/NOT_BANKED.sha256` stay withheld in THAT package; their content is superseded by this one.
* **Pod command changes:** `SFT4R_HF_TOKEN_FILE=<the pod's HF token file> nohup setsid bash /workspace/refe-plan-sft4r/refe/sft4r_chain.sh > ... &` (value is the file's path; never a token).

## 1. SO-A16 (`test_launch_gate.py:339` bare `bash`)

`posix_bash()` (new, in the test) resolves a REAL POSIX bash by absolute path, skipping the WSL launcher wherever it sits on PATH (System32 / WindowsApps), then Git for Windows' own `usr/bin/bash.exe`; none found => `ZZTEST_SKIP` with a loud reason.
MEASURED: unit-checked under four PATHs (normal; WSL first; WSL only; nothing). **Full form, NO shim, tip + this one change: rc 0, `ZZLG_FULL_PASS 719`, 1435 s, 0 skips** (`logs/e2e/test_launch_gate_full_noshim_tip_plus_SO-A16.log`; 719 is the tip's own count, `raw/2026-10-17-safety-objective/BUILD.md`).
Shimmed (`PYTHONPATH=C:/so/shim`) and unshimmed runs on the FINAL tree: rows `test_launch_gate_full_*` in section 6.
**Other bare-`bash` calls (listed, NOT fixed here):** `code/finalize_mini.py:156` (`["bash", ...]`, a dev-box data script), `raw/2026-10-15-p2rep-gsuite-pod/wc_probe.py:38` (banked probe). In the tests, the `shutil.which("bash")` family resolves by PATH order, so it is right
under Git bash and WRONG in a process whose PATH has System32 first (a WMI-detached process; MEASURED below): `test_p2rep_chain.py:49`, `test_p2r_chain.py:279`, `test_p6_chain.py:537-584`, `test_p6_cpu_probe.py:196,463,476`, `test_rls1_chain.py:746,877,1171,1222,1247`,
`test_rls1_queue.py:49`, `test_rl_safeobj.py:953,986`, `test_s1aux_chain.py:71,773,792,1218-1248`, `test_s1c.py:602`. `test_sft4r_gate.py` and `test_sft4r_chain.py` (both edited here) got the same resolver.

## 2. S4R-1(b): "`test_sft4r_gate.py` exits rc 1 INSIDE the refe_sft4r G-SUITE but passes standalone"

**Root cause (MEASURED, reproduced): `test_sft4r_gate.py` section 6 calls `L.run_checks(<profile with a STUB suite>)` IN-PROCESS. That reaches `gate_checks.check_suite`, which reads the ambient `REFE_GATE_NESTED` (the stack of suite entries above this
process) and by design (`SUITE_NEST_MAX = 2`) lets a suite script run exactly ONE nested suite. When the ambient stack is already 2 deep the stub suite is REFUSED -- `G-SUITE refuses to nest: this process is already 2 suite level(s) deep (limit 2)` -- G-SUITE reads FAIL where the test
requires PASS, and the file exits rc 1 after 132 checks, ~3 s in.**
* Why only "inside G-SUITE": the 2026-10-16 harness's "gate" form put a depth-1 stack on the `suite-prerun` job (`tools/run_all.py:stack_env`), but `suite-prerun` is NOT a suite entry, so every real suite child ran at depth 2. Standalone the ambient stack was empty, and the first e2e (which passed, 59 s) ran before the queue's gate form existed
  (INFERRED from the table: it has no sentinel row). Signature match with the readiness table row `e2e_suite_prerun_refe_sft4r`: rc 1, ~3.3 s, `ZZTEST_PASS` not seen, the other three entries rc 0.
* Reproduction: `tools/emul_child.py <refe> test_sft4r_gate.py --shim --outer` (child spawned exactly as `check_suite` does: argv `[sys.executable, script]` without `-X utf8`, `cwd=refe`, env + `PYTHONIOENCODING` + `REFE_GATE_NESTED`) -> **rc 1, `ZZTEST_FAIL after 132 checks`, the quoted refusal** (`logs/repro/emul2_depth2_before_fix.log`).
  Same child at depth 1 (the real flow) -> rc 0, `ZZTEST_PASS 152` (`logs/repro/emul1_depth1.log`; both on the rebased tree BEFORE this fix, with the bash shim on, because the bare-bash defect below was not yet fixed). Ruled out by the same instrument: `-X utf8` absent in the child, `cwd`, and the sentinel's JSON form.
* **Fix: the TEST, not the engine.** The guard did its job; section 6's subject is run -> evidence -> token -> verify on a miniature of the pod. Section 6 now pops the ambient sentinel for its in-process stub-suite runs and restores it before section 7
  (sections 7-8 deal with the real ambient stack; section 8 and `test_suite_nesting.py` still prove the real suite refuses to nest). After: depth-2 child **rc 0, `ZZTEST_PASS 153`** (`logs/repro/emul_depth2_after_fix.log`, no shim); the real `suite-prerun --profile refe_sft4r` PASSES both clean AND with the harness's depth-1 outer sentinel
  (4/4 entries rc 0, 0 skips, 60 s each; `logs/e2e/`).
* **Second defect found in the same file, the same class as SO-A16:** `test_sft4r_gate.py:81` ran the pod script with a BARE `bash`. On THIS box today it fails standalone at check 10 (rc 127, `/bin/bash: C:\...\sft4r_pod.sh: No such file`; `logs/repro/sg_alone_bare_bash.log`). The readiness runs on 2026-10-16 ~23:00 did not
  see it. MEASURED timestamps: `System32\bash.exe` last written 17.10 09:47, `WindowsApps\bash.exe` alias created 09:55, the NVIDIA SDK Manager WSL distro registered 10:04 -- i.e. AFTER the readiness queue (23:11-23:36 on 16.10). INFERRED (not proven): that is why a bare `bash` worked then and does not now.
  Fixed with the same absolute-path resolver; a box with no POSIX bash now FAILS LOUD at that line (the cross-check cannot be skipped).

## 3. S4R-1(c): the token-file path (`patch/patch_sft4r_chain_tokenenv.py`)

`HFTOK=${SFT4R_HF_TOKEN_FILE:-}`; the precheck's FIRST statement after `STAGE=precheck` is `[ -n "$HFTOK" ] || refuse "SFT4R_HF_TOKEN_FILE is not set: ..."` (before `chain armed`, the manifest, the lock's work). The v2 default path and the header comment that named it are gone.
`test_sft4r_chain.py` (20 -> 31 checks) RUNS the chain's precheck with a stub `flock` and an empty pod root: unset -> rc 1 + the refusal and no stage marker; empty -> the same; set to a missing file -> the variable is accepted and a LATER check refuses (no `SHIP_MANIFEST.json`);
two MUTATIONS of a temp copy (a default path put back; the check removed) both stop refusing on the variable, so the test can go RED. The chain source, the patchers, the tests and the diffs were grepped: none names the path (MEASURED; the old value is assembled from pieces in the patcher).
Cost: `bash -n sft4r_chain.sh` OK. **UNVERIFIED (as in the readiness package): the stages that need the pod (GPU wait, HF pushes, preflight) have never run end to end.**

## 4. S4R-1(d): `eval/PREREG_SFT4R.md` Amendment 3

APPENDED (`raw/2026-10-17-s4r-integration/amendment3_append.md`, folded from `AMENDMENT_3_DRAFT.md` with the tip reference updated and one operational item added: the required env var). MEASURED: the original 15,187 bytes (sha256 `28f5a7b4...`) are an exact PREFIX of the new file
(19,049 bytes, sha256 `d7af485d...`). **There is no `.sha256` sidecar for `PREREG_SFT4R.md` and nothing in code pins its hash** (grep over `*.py *.sh *.json *.sha256 *.txt`), so no pin was changed. The PI can still overrule the amendment. It moves no lever (2,733 updates is the registered epoch on the complete labels).

## 5. What this does NOT prove

* Nothing ran on the pod; `launch_gate.py run` has not minted a real `refe_sft4r` token anywhere (needs the pod's data, the preflight JSON and the G-EGR JSON).
* `SHIP_MANIFEST.json` at the tip is stale (it predates several refe files); regenerate it after landing (`launch_gate.py ship-manifest --commit <40-hex>`). The chain's precheck now also lists `refe/test_suite_nesting.py`.
* `build_final.py` cannot be re-run from the repo alone: it reads the previous agent's v2 chain from `post_p0/` of the readiness package, which was withheld for naming the token path (it exists on the dev box at `E:/Projects/TanitAD/.../raw/2026-10-16-sft4r-readiness/post_p0/`). The post-images ARE banked (`refe/`).
* Windows only; the POSIX branches of the new helpers (`shutil.which("bash")`) were not exercised. CUDA was NOT hidden in these runs (the readiness harness hid it, which is what made its GPU-probe rows fail with `Invalid device id`).
* The dev box also ran a long CPU job (the P2 full table, 3 workers) throughout: every run was one test at a time at below-normal priority, with a free-RAM guard (>= 8 GB) in `tools/queue.py`. Times are therefore ~2.7x the readiness package's 4-way times.
* Environment hazard found while queueing: a WMI-detached process has neither Git for Windows' dirs on PATH nor `HOME`; the chain tests then fail inside `last_exit` (`grep | tail | cut`) -- MEASURED, `logs/qfinal_attempt1/`. `tools/queue.py` restores both (as the 2026-10-16 harness did).

## 6. Test table: every test that names an edited file, FULL form, one at a time

<!--TEST_TABLE_BEGIN-->
**Generated 2026-10-17 23:43 by `tools/post_queue.py`: 22 runs, 3 not clean.** Raw: `logs/qfinal/SUMMARY.txt` (patched, one line per run), `logs/qbase/SUMMARY.txt` (baseline re-run of each failure). Each test ran ALONE, at below-normal priority, CUDA not hidden, no bash shim except the row named `_shim`. Tree = the 2026-10-17 final set on a `git archive` of tip `3b8c39f`.

| test (queue id) | patched: rc / token / s | baseline (pristine tip + SO-A16), if it failed | verdict |
|---|---|---|---|
| test_sft4r_chain | rc 0 / `ZZTEST_PASS 31` / 5 s / skips 0 |  | **PASS** |
| test_suite_nesting | rc 0 / `ZZTEST_PASS 95` / 8 s / skips 0 |  | **PASS** |
| test_sft4r_gate | rc 0 / `ZZTEST_PASS 153` / 11 s / skips 0 |  | **PASS** |
| test_p2r_chain | rc 0 / `ZZTEST_PASS 139` / 7 s / skips 0 |  | **PASS** |
| test_p6_overlap | rc 0 / `ZZTEST_PASS 54` / 1 s / skips 0 |  | **PASS** |
| test_rl_s1b | rc 0 / `ZZTEST_PASS 18  (1 unmutated suite + 17 RED mu` / 11 s / skips 0 |  | **PASS** |
| test_rl_r0c | rc 0 / `ZZTEST_PASS 26  (1 unmutated suite + 25 RED mu` / 46 s / skips 0 |  | **PASS** |
| test_rls1_chain | rc 0 / `ZZTEST_PASS 522` / 166 s / skips 0 |  | **PASS** |
| test_s1c | rc 0 / `ZZTEST_PASS 220  (2 skipped, not counted)` / 117 s / skips 2 | rc 0 / ZZTEST_PASS 220  (2 skipped, not counted) | **PRE-EXISTING (fails on the baseline too)** |
| test_aux_s1_wiring | rc 0 / `ZZTEST_PASS 20  (1 unmutated suite + 19 RED mu` / 129 s / skips 0 |  | **PASS** |
| test_s1aux_chain | rc 1 / `NONE` / 58 s / skips 0 | rc 1 / AssertionError | **PRE-EXISTING (fails on the baseline too)** |
| test_aux_g12_bf16 | rc 1 / `NONE` / 17 s / skips 0 | rc 1 / AssertionError: the G12 bf16 checks FAIL on the real source: ['h1cuda/preflight_ran_to_a_G | **PRE-EXISTING (fails on the baseline too)** |
| test_gate_gpu_probes | rc 0 / `ZZTEST_PASS 16  (incl. the CUDA probe on this ` / 3 s / skips 0 |  | **PASS** |
| test_p6_cpu_probe | rc 0 / `ZZTEST_PASS 111` / 137 s / skips 0 |  | **PASS** |
| test_p6_levers | rc 0 / `ZZP6_PASS 156` / 212 s / skips 0 |  | **PASS** |
| test_p2rep_chain | rc 0 / `ZZTEST_PASS 353` / 191 s / skips 0 |  | **PASS** |
| test_rl_safeobj | rc 0 / `ZZTEST_PASS 61  (1 unmutated suite + 60 RED mu` / 345 s / skips 0 |  | **PASS** |
| test_gate_suite_prerun | rc 0 / `ZZTEST_PASS 144` / 609 s / skips 0 |  | **PASS** |
| test_p6_chain | rc 0 / `ZZP6CHAIN_PASS 341` / 780 s / skips 0 |  | **PASS** |
| test_rls1_queue | rc 0 / `ZZTEST_PASS 254` / 922 s / skips 0 |  | **PASS** |
| test_launch_gate_full_noshim | rc 0 / `ZZLG_FULL_PASS 719` / 441 s / skips 0 |  | **PASS** |
| test_launch_gate_full_shim | rc 0 / `ZZLG_FULL_PASS 719` / 343 s / skips 0 |  | **PASS** |
<!--TEST_TABLE_END-->
