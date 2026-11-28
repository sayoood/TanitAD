# R0c chain wiring: `RLS1_ARMS="R0c"` through `rls1_pod.sh`, built and proved offline at zero GPU

Research Lab agent (Arch+Inference, Sonnet), 2026-10-15. Base: the tip tree `066ca5a` (code plus fixtures, scratch copy `tw2`) with the R0c build's `PATCH_1..7` and `test_rl_r0c.py` applied (`../NOTES.md`). **Nothing is launched, staged or committed**; the only files written on E: are in this folder. Evidence class of every number below: **MEASURED (ours, CPU, Git Bash on the Windows dev box, logs in `logs/`)** unless a line says otherwise. No pod, no A40, no network.

## 1. Headline

* **Built and proved:** R0c is a registered arm of `refe/rls1_pod.sh`, maps to gate profile `refe_rls1c`, emits **exactly** `../R0c_ARGV.txt`'s argv (token for token against the new `eval/rls1c_arms.json`; sha256 `f588a6cb...`; R0's chain argv rebuilt from the same code hashes to R0's banked `24451b62...`), is read against **its own arm A** (the draft's rule), follows the generic "a killed arm refuses" rule, and an unknown or misspelled arm is still refused at the precheck. `test_rls1_chain.py` **392 -> 512**; the R0c arm-loop scenarios of `test_rls1_queue.py` add 33 checks (section 2). Every new behaviour has a check and a RED mutation (section 6).
* **INTEGRATION NEEDED (Master Mind):** land `../PATCH_1..7` + `../test_rl_r0c.py` **and** `PATCH_8..12` here as ONE change, then regenerate `SHIP_MANIFEST.json` at the landing commit (`launch_gate.py ship-manifest`): `rls1_pod.sh` is under `refe/`, so the tree digest moves and no earlier token stays valid (by design); `eval/` is outside the digest. Apply order is numeric, not glob order (`PATCH_10` sorts before `PATCH_2`): `for i in 1 2 3 4 5 6 7; do patch -p1 < raw/2026-10-15-r0c-build/PATCH_${i}_*.diff; done; cp raw/2026-10-15-r0c-build/test_rl_r0c.py refe/; for i in 8 9 10 11 12; do patch -p1 < raw/2026-10-15-r0c-build/chain/PATCH_${i}_*.diff; done`. ⚠ **Do not use a plain `git apply` on a Windows checkout with `core.autocrlf=true`**: it rewrites every target in CRLF (MEASURED here: all 609 lines of `rls1_pod.sh` came out CRLF; the chain's own test raises `CRLF` and bash on the pod would choke on `\r`). `patch -p1` or `git -c core.autocrlf=false apply` keep LF; the stack was proved with the latter on a fresh tip copy, and `rls1_pod.sh`, `test_rls1_chain.py`, `test_rls1_queue.py`, `eval/rls1c_arms.json`, `eval/RLS1_POD_RUNBOOK.md` are byte-identical to the working copy (`cmp`).
* **Still a PI decision:** running R0c (about 3.5 A40-h, beyond RL-S1b's authorised budget, `PREREG_R0C_DRAFT.md` section 5). Nothing here starts it.
* **`launch_gate.py` and `gate_checks.py` are NOT changed.** The chain needed nothing the R0c patches lack: `--arm-name R0c` is a choice (`launch_gate.py:872`), the profile is registered (`:216-217`), and the real gate accepts the chain's argv (section 4, hop 8). `test_launch_gate.py --static` stays at **659** (658 before PATCH_1..7).

## 2. Counts, before and after (torch 2.11 `venvs/tanitad` unless stated; each suite run from `refe/` as `PYTHONIOENCODING=utf-8 python -u test_x.py`)

BEFORE = a fresh copy of the tip tree, untouched. AFTER = the same tree with `PATCH_1..12` applied from this folder and `test_rl_r0c.py` copied in (`r0f`, LF-preserving apply).

| suite | BEFORE (tip) | AFTER (patched + wired) | note |
|---|---|---|---|
| `test_rl_r0c.py` | (does not exist) | **26** | also **26** under torch 2.7.1 (`venvs/driverl-eval`) |
| `test_rl_s1b.py` | 18 | **18** | also 18 / 18 under torch 2.7.1 |
| `test_rl_epo.py` | 42 | **42** | |
| `test_launch_gate.py --static` | 658 | **659** | +1 is PATCH_2/3 (the registered-EPO loop now covers `epo_ref`), not this wiring |
| `test_aux_s1_wiring.py` | 20 | **20** | |
| **`test_rls1_chain.py`** | **392** | **512** | +120, see below |
| `test_rls1_queue.py` (full file) | 101 | **134** | `--only-rls1c` alone: **33** (`ZZTEST_PARTIAL_RLS1C 33`) |
| `test_s1aux_chain.py` | 385 | **385** | |
| `test_p2rep_chain.py` | 353 | **353** | |
| `test_p6_chain.py` | **341**\* | **341**\* | see "p6" below |
| `test_flock_rw.py` (scans `rls1_pod.sh`) | 59 | **59** | |

**The +120 in `test_rls1_chain.py`:** 2 (`write_sh` exec bit and its RED twin) + 51 (`r0c_behaviour` probes run under bash: argv 6, profile 9, control-read block 11, kill refusals 6, precheck arm names 19) + 41 (RED mutations: 17 mutations caught by both a static and a behaviour check, 7 by one) + 11 (2e: the profile the chain picks, 4 calls of the REAL `launch_gate.py refusals` CLI, 6 REAL token mint/verify cases) + 3 (2f: the planned-update check and the trainer's own parser and EPO refusals on the chain's argv) + 5 (the REAL `rls1_pod.sh` run in the sandbox: 4 unknown or misspelled arms and `R0c` passing the arm check) + 7 (`rls1_checks` monitor and replay judge `R0c` as an R arm).
**Two pre-existing lines of `test_rls1_chain.py` had to move:** `check_queue`'s `known` argv set (it refuses an invented RL-S1b argv; `R0c` is now known) and the mutation "R0b / R1b are not registered" (its anchor was the old `REGISTERED_ARMS` string).
**p6:** \* `ZZP6CHAIN_PASS 341` both times, **with `TMP`/`TEMP` on E:** the test writes a 2 GB `dd` file into the system temp dir and C: had 1.1-2.0 GB free (100 % full), so with the default temp dir it fails at `assert ... "dd write test 2 GB ok"` on the UNTOUCHED tip and on the patched tree alike (`logs/before_tip_clean/p6_chain_t211.log`, `logs/after_patched_wired/p6_chain_t211.log`): an environment fault, identical before and after, not a result. The passing runs are `logs/before_tip_clean/rerunE_p6_chain_t211.log` and `logs/after_patched_wired/rerunE_p6_chain_t211.log`.
**`test_rls1_queue.py` full file:** the R0c build's NOTES called it "NOT RUNNABLE on Windows" (`couldn't create signal pipe, Win32 error 5` inside `kmon_scenarios`). On this run the **unpatched** tip file **passed 101** (434 s), so that failure is intermittent, not structural. The patched full file also **passed: 134 = 101 + 33** (rc 0, 511 s, `logs/after_patched_wired/rls1_queue_t211.log`); it is one sample of an intermittent harness, so a red run there should be re-run before it is believed.

## 3. What changed (`PATCH_8..12`, all made against the PATCHED tree so they stack on `PATCH_1..7`)

| patch | file | change |
|---|---|---|
| `PATCH_8_rls1_pod.diff` | `refe/rls1_pod.sh` (+14 / -2) | `REGISTERED_ARMS` += `R0c`; `arm_profile` `R0c) echo refe_rls1c`; `arm_args` case `R0c)`; the control-read block (`RD=C0` named, `REFDIR` forced to its own arm A); a loud `ZZRLS1_R0C_NO_SECOND_CONTROL` log in the 9d read; the precheck's required-file list gains `refe/test_rl_s1b.py refe/test_rl_r0c.py` (the gate's suite; absent = a refusal after the queue wait and the preflight); header comments |
| `PATCH_9_test_rls1_chain.diff` | `refe/test_rls1_chain.py` (+311 / -3) | `check_rls1c`, `control_runner`, `r0c_behaviour`, `r0c_mutations`, sections 1b / 2e / 2f, sandbox and monitor cases, **`write_sh` now `os.chmod(path, 0o755)`** with `write_sh_violations` (RED twin) |
| `PATCH_10_test_rls1_queue.diff` | `refe/test_rls1_queue.py` (+82) | `rls1c_checks` (7 arm-loop scenarios), `rls1c_all` (+ 8 mutations), `--only-rls1c` partial mode |
| `PATCH_11_new_eval_rls1c_arms.diff` | `eval/rls1c_arms.json` (NEW) | the argv source of truth, in the shape of `rls1b_arms.json`, generated from `../R0c_ARGV.txt` and self-checked (empty `refe_rls1c` refusals, both sha256 relations) |
| `PATCH_12_runbook.diff` | `eval/RLS1_POD_RUNBOOK.md` | the R0c paragraph (section 7 here) and the updated test counts in its header |

## 4. Reachability, `file:line` at each hop (patched tree; line numbers are those of `rls1_pod.sh` etc. after PATCH_8)

`RLS1_ARMS="R0c"` -> trainer argv -> gate profile -> token verify -> trainer. Each hop names the check that fails if it is cut.

| # | hop | where | proved by |
|---|---|---|---|
| 1 | the arm name is accepted | `rls1_pod.sh:230` precheck `arm_registered "$ARM" \|\| refuse` -> `:93` -> `:46` `REGISTERED_ARMS=... R0c`; the same check again in the loop (`:481`) | `r0c_behaviour` group `arms` (4 lists accepted, 15 unknown or misspelled refused with `REFUSED: unknown arm`); sandbox run of the REAL script (5); queue `unknown_arm` |
| 2 | the package carries the gate's suite | `:233` required files `refe/test_rl_s1b.py refe/test_rl_r0c.py` | `check_rls1c` `R0C_SUITE_FILE_NOT_REQUIRED` (mutation 22); the existing `REQUIRED_FILE_ABSENT` loop checks both exist |
| 3 | argv | `:484` `ARGS="$(arm_args "$ARM")"` -> `:185` `arm_args` -> `:197` `R0c) seed=0; epo="... --epo-ref armA"` -> `:200` the `echo` | group `argv`: tokens equal `eval/rls1c_arms.json` (53 tokens); sha256 `f588a6cb...`; R0 rebuilt by the same function hashes to `24451b62...`; R0c minus the lever with the directory swapped back IS R0; the REAL gate accepts it |
| 4 | the profile | `:485` `PROFILE="$(arm_profile "$ARM")"` -> `:85-92`, `:88` `R0c) echo refe_rls1c` | group `profile` (9 arm/list combinations, including that C0, C1, R0, R0b keep theirs); queue `order`: `launch_gate.py run --profile refe_rls1c --arm-name R0c` |
| 5 | the pre-registered budget | `:488` `rls1_checks.py updates-check --args "$ARGS" --n-sets "$NTRAIN"` | 2f: `RC.updates_check(argv, 16000)` is OK (1,000 updates) |
| 6 | the reference the monitor and the replay read against | `:491-497` control block: `:493` `RD=C0`, `:497` `REFDIR="$ARMDIR/scores"` (after the generic `:495`); the R0b guard `:499-501` is untouched and is not reached by R0c | group `control` (11 arm/profile/C0-C1-present cases, **R0c -> `arms/R0c/scores` with C0's dumps present AND absent; R0 -> C0's when present; R0b refused without C0**); queue `order`, `no_c0`, `mixed` |
| 7 | the argv reaches the TRAINER'S parser and loss | preflight `:505` `scorer_finetune.py $ARGS --preflight`; train `:529`; `scorer_finetune.py:575` `--epo-ref` (choices `EPO_REFS` `:179`) -> `:591` `epo_refusals` -> `:340` value check -> `:798` `epo_term(epo_reference(a.epo_ref, sx0, sxA), ...)` -> `:780` -> `epo_loss` `:218`, `:226` `logpi0 = epo_policy(sx_ref, ...)`; the BUILT reference is written to the data event `:706` (`"ref": epo_reference_built(a.epo_ref)`) | 2f: the trainer's OWN `build_parser()` reads `armA` from the chain's R0c argv and `base` from R0's, `epo_refusals` is empty for both; queue `order`: the preflight and the trainer calls both end in / contain `--epo-ref armA`; `../test_rl_r0c.py` proves flag -> loss (26) |
| 8 | the gate | `:512` `launch_gate.py run --profile "$PROFILE" --arm-name "$ARM" ... --args "$ARGS"` -> `launch_gate.py:872` (`--arm-name` choices include `RLS1C_ARM_CLASS`) -> `:216-217` `PROFILES["refe_rls1c"]` -> `:375` `must_name` (`--epo-ref` NAMED) -> G-DVB `gate_checks.py:646` (`epo_ref: built`), `:658` (`ref` in `SCORER_EPO_KEYS`), `:814` | 2e: the REAL CLI `launch_gate.py refusals --profile refe_rls1c --arm-name R0c` on the chain's argv is `ZZGATE-CLEAN`; R0's argv under R0c's name is refused (`--epo-ref`); the original profile refuses R0c's argv, and `R0c` is not one of its arms |
| 9 | the token is verified, not grepped | `:204-208` `token_ok` -> `launch_gate.py:830` `cmd_verify` -> `gate_token.verify_token` (profile, argv sha256, tree, data); called at `:515`, `:518`, `:527` | 2e: a token MINTED by `GT.mint(profile="refe_rls1c", argv=<the chain's R0c argv>)` verifies; R0's argv, `--epo-ref base`, R0's arm directory, the old profile, and an R0 token for R0c all FAIL |
| 10 | the trainer starts on that same string | `:529` `scorer_finetune.py $ARGS` after `:527` `token_ok \|\| refuse` | the existing `TRAIN_WITHOUT_TOKEN` static check (unchanged) + queue `order` |
| 11 | monitor, replay, reads, egress | `:541` `rls1_checks.py monitor --arm "$ARM" --refdir "$REFDIR"`; `:562` replay; `:572-582` reads (`rlval_vs_A0.json`, and `rlval_vs_C0.json` or `:581` the loud marker); `:598` egress to `Sayood/tanitad-refe-rls1-r0c`; `:605` `ZZRLS1_ARM_DONE` | queue `order` / `no_c0` (the exact `rl_val_eval.py` calls); chain 7 checks that `RC.monitor` / `RC.replay` judge `R0c` as an R arm (K3, K4, K6; K5 stays R2's) |
| 12 | a kill or refusal stops the chain | `:483` (a re-run of a killed arm refuses before the preflight) and `:603` (after the evidence is banked) key on `$ARM`, never on a name | group `kill` (R0c and R0b, three states each); queue `kill`, `kmon`, `rerun_after_kill` |

## 5. The read rule (quoted, not invented)

`../../2026-10-15-rls1-adam-dose/PREREG_R0C_DRAFT.md` section 2: *"its own arm A is the same-pass control, C0 is the banked second control"*, and under guards: *"the control for every delta is the arm's own A; RL-val is the selection tier"*; section 4: reads at u250 / u500 / u750 / u1000, RL-val, **B - A and B - base**, paired log-cluster bootstrap 10,000 / seed 20260927. `launch_gate.py` (`RLS1C_*` comment) says the same: *"R0c is read against its own arm A (the same-pass control) and C0 (banked)"*.

Implemented as:
* **Own arm A, always.** The K3 / K4 monitor (`--refdir`), the kill replay and `rlval_vs_A0.json` (`A0` = base, `CTRL` = arm A, `R` = arm B with `--base A0`: B - A and B - base) all read against `arms/R0c/scores`, **even when C0's dumps are on the pod**. Without the override line R0c would inherit R0's behaviour (read against C0 when its dumps exist: `${ARM:1:1}` happens to give `C0`), which is NOT the draft's rule; mutation 14 / queue mutation 1 prove the override is what holds it.
* **C0 as the second control.** `rlval_vs_C0.json` (`C` = C0's arm A, `R` = arm B, `--base C`) is produced when `arms/C0/scores/scores_update1000_A.npz` exists.
* ⚠ **DECISION for the Master Mind (not made silently): a missing C0 does NOT refuse the run.** The draft calls C0 the *second* control and the pre-registered deltas are against arm A, so the run goes on and `chain.log` carries `ZZRLS1_R0C_NO_SECOND_CONTROL ... rlval_vs_C0.json was NOT produced ... restore C0's scores (HF Sayood/tanitad-refe-rls1-c0) and run rl_val_eval.py`. If you want it hard, the one-line alternative is a precheck refusal on the missing file for `R0c` (cheap, before the queue wait); the RL-S1b rule (R0b/R1b refused without their control) is untouched and the opposite choice was made there because C0 IS their registered control.

## 6. RED mutations (each is a copy of the chain / helper with ONE property broken; the matching check must say so)

`test_rls1_chain.py` (24, `r0c_mutations`; the three the brief named are #1, #4-6 and #8):

| # | mutation (a copy of the chain / helper with ONE property broken) | static catcher (check_chain) | behaviour catcher (r0c_behaviour group: violation) |
|---|---|---|---|
| 1 | the R0c argv line is deleted | `R0C_NO_ARGV` | `argv: R0C_NO_ARGV` |
| 2 | R0c is not registered | `R0C_NOT_REGISTERED` | `arms: R0C_REFUSED_AT_PRECHECK` |
| 3 | R0c is registered under a typo (R0C) | `R0C_NOT_REGISTERED` | `arms: MISSPELLED_ARM_ACCEPTED` |
| 4 | R0c is judged under the original profile | `R0C_NO_PROFILE` | `profile: R0C_WRONG_PROFILE` |
| 5 | R0c is judged under the RL-S1b profile | `R0C_NO_PROFILE` | `profile: R0C_WRONG_PROFILE` |
| 6 | R0c has no profile line (falls through to refe_rls1) | `R0C_NO_PROFILE` | `profile: R0C_WRONG_PROFILE` |
| 7 | C1 is judged under refe_rls1b whenever R0c is listed | - | `profile: R0C_WRONG_PROFILE` |
| 8 | --epo-ref armA is dropped from R0c's argv | `R0C_NO_EPO_REF_ARMA` | `argv: R0C_ARGV_NOT_THE_JSON` |
| 9 | --epo-ref armA is dropped (the real gate says so too) | - | `argv: R0C_REFUSED_BY_THE_REAL_GATE` |
| 10 | --epo-ref base is named instead of armA | `R0C_NO_EPO_REF_ARMA` | `argv: R0C_ARGV_NOT_THE_JSON` |
| 11 | a second lever rides along (--epo-lam 0.25) | `R0C_EXTRA_LEVER:--epo-lam` | `argv: R0C_NOT_R0_PLUS_ONE_FLAG` |
| 12 | R0c runs with seed 1 | `R0C_WRONG_SEED` | `argv: R0C_ARGV_NOT_THE_JSON` |
| 13 | R0 is no longer the banked token's argv | - | `argv: R0_ARGV_NOT_THE_BANKED_TOKEN` |
| 14 | R0c is read against C0's dumps when they are there (the own-A override is deleted) | `R0C_NOT_READ_AGAINST_ITS_OWN_A` | `control: R0C_CONTROL` |
| 15 | the own-A override runs BEFORE the generic assignment and is overwritten | `R0C_NOT_READ_AGAINST_ITS_OWN_A` | `control: R0C_CONTROL` |
| 16 | C0 is not named as R0c's second control (static only: ${ARM:1:1} happens to give C0 as well) | `R0C_SECOND_CONTROL_NOT_NAMED` | - |
| 17 | R0c falls under the RL-S1b control guard (refused against its own arm A) | `RLS1B_GUARD_EDITED` | `control: R0C_CONTROL` |
| 18 | R0b / R1b lose their guard (a missing control is silently replaced by the arm's own A) | `RLS1B_GUARD_EDITED` | `control: CONTROL_DRIFT` |
| 19 | no loud marker when C0's banked dumps are missing | `R0C_SILENT_WITHOUT_SECOND_CONTROL` | - |
| 20 | the end-of-loop kill refusal skips R0c | `KILL_RULE_ARM_GATED` | `kill: R0C_KILL_DOES_NOT_REFUSE` |
| 21 | the restart-after-kill refusal skips R0c | `KILL_RULE_ARM_GATED` | `kill: R0C_KILL_DOES_NOT_REFUSE` |
| 22 | the gate's suite files are not required before the queue wait | `R0C_SUITE_FILE_NOT_REQUIRED` | - |
| 23 | arm names are matched as substrings | - | `arms: MISSPELLED_ARM_ACCEPTED` |
| 24 | the precheck no longer validates the arm names | `ARM_NAMES_NOT_REFUSED_TWICE` | `arms: R0C_NO_PRECHECK` |

`test_rls1_queue.py --only-rls1c` (8, `rls1c_all`; each runs the whole arm loop against the fakes, so these are BEHAVIOUR catches):

| mutation | scenario that goes RED |
|---|---|
| R0c is read against C0's dumps whenever they are on the pod (override line deleted) | `order`, `mixed` |
| R0c is judged under the original profile | `order` |
| R0c's argv loses `--epo-ref armA` | `order` (preflight AND trainer call) |
| the loud marker for a missing second control is gone | `no_c0` |
| the second control (C0) is never read | `order` |
| a killed R0c does not stop the chain | `kill`, `kmon` |
| a killed R0c re-trains on a re-run | `rerun_after_kill` |
| a misspelled arm reaches the preflight | `unknown_arm` |

`write_sh` without `os.chmod` -> `NO_CHMOD_X` (`write_sh_violations`; on Windows the `chmod` CALL is recorded because the OS stores no exec bit; off Windows the shim is also run directly).
Three chain-file mutations have a static catcher only: #16 (`RD=C0` unnamed: behaviourally an *equivalent mutant*, `${ARM:1:1}` is `0` -> `C0` anyway, so only the text check holds the explicit naming), #19 (the loud marker; the BEHAVIOUR catch for it is the queue test's `no_c0`, above) and #22 (the suite files; the sandbox package fills every required file with an empty one).

## 7. Runbook: running R0c on the pod (this is also in `eval/RLS1_POD_RUNBOOK.md`, PATCH_12)

**Not launched: a PI decision (about 3.5 A40-h).** R0c is R0 with ONE lever, `--epo-ref armA`. The pod address is never written into a file.

*Before the start* (all light, none of it touches the GPU):
1. The landed package with a **regenerated `SHIP_MANIFEST.json`** is in `/workspace/refe-plan-rls1` (`python refe/launch_gate.py ship-manifest --pkg <package root> --commit <40-hex> --out <package root>/SHIP_MANIFEST.json`). The precheck now also requires `refe/test_rl_s1b.py` and `refe/test_rl_r0c.py` and refuses in seconds if they are missing.
2. `ls /workspace/data/rls1/stage/` still shows the heavy markers of the earlier runs: `base plan fancmp dbs tlane bank dump_val_s0of1 dump_train_s{0..3}of4 label_val_s0of1 label_train_s{0..3}of4`. A missing one makes the chain REDO that stage (the dump alone is about 3.3 GPU-h). UNVERIFIED: I cannot see the pod.
3. `ls /workspace/data/rls1/arms/C0/scores/scores_update1000_A.npz` (the second control; otherwise restore from HF `Sayood/tanitad-refe-rls1-c0`).
4. `HF_TOKEN` (write-scoped) is exported in the shell that starts it, never on a command line. No `stage/arm_R0c_killed` from an earlier attempt.

*The command* (detached, like every RL-S1 start):

```
RLS1_ARMS="R0c" nohup setsid bash /workspace/refe-plan-rls1/refe/rls1_pod.sh > /workspace/data/rls1/stdout.log 2>&1 < /dev/null &
```

No override is needed: `RLS1_ARMS="R0c"` is the whole selection (the default list is unchanged). **Dry run first (recommended):** add `RLS1_STOP_AFTER=gate_R0c`; the chain queues, runs the A40 preflight, mints and VERIFIES the PASS token, and stops before the trainer. (`RLS1_STOP_AFTER=queue` stops as soon as the queue clears.)

*What it waits on:* the same QUEUE as every RL-S1 run, so it may be started while the pod is busy. After the cheap precheck (P2 over `ZZP2R_DONE`, the token, the arm names, the files) it holds only `chain.queue` (`ZZRLS1_QUEUED <why>` every 10 min) while ANY chain is ahead: P2; S1 (its `chain.lock`, the S1 H arms); the P2 seed replicate (P2-s1) once PAST its wait; P6 once past its wait (`ZZP6_WAIT_OK`). So **the order "S1 H arms, then P2-s1 with P6 beside it, then R0c" is automatic.** When nothing is ahead it takes `chain.lock`, settles `RLS1_SETTLE_S` (20 s) and needs the GPU idle on 3 polls; finished stages are skipped; then the arm: preflight -> launch gate (`refe_rls1c`, `--arm-name R0c`) -> token verified -> trainer + watcher + kill monitor -> RL-val reads -> egress to the private repo `Sayood/tanitad-refe-rls1-r0c` (created by the first push).

*What to look for in `/workspace/data/rls1/chain.log`:* `arm R0c: preflight PASS, launch gate PASS (token verified)`; `arm R0c: trainer pid/pgid ... (control: .../arms/R0c/scores)` (**a control path ending anywhere else is a wiring fault: stop it**); `arm R0c: kill monitor READY`; at the end `ZZRLS1_ARM_DONE R0c`, `ZZRLS1_DONE arms: R0c`. In `arms/R0c/`: `rlval_vs_A0.json` (B - A, B - base), `rlval_vs_C0.json`, `kill_checks.jsonl`, `monitor.log`. A pre-registered kill (K3 / g5 ..., or K-MON) STOPS the chain after the evidence is banked and writes `stage/arm_R0c_killed`; a re-run refuses until the Master Mind removes that marker or drops the arm. The outcomes O1..O5 and each next lever are `PREREG_R0C_DRAFT.md` section 4; a separated interval on one training seed is necessary, not sufficient (`H-ESTIM-SEED-1`): the replicate comes before any claim.

## 8. UNVERIFIED (and what would verify it)

* **Nothing ran on a pod or a GPU.** The chain was executed under Git Bash with shimmed `flock` / `nvidia-smi` / `ps` and a fake python for the trainer, monitor, watcher and gate (the existing `armloop_run` harness), plus the REAL gate / token / trainer-parser code. Linux-only behaviour (the real `flock`, cgroup reads, `nvidia-smi`) is shimmed, as it already was for every earlier arm.
* **A real `launch_gate.py run --profile refe_rls1c` (mint) was not run**: it needs the pod's `preflight.jsonl` and key. Exercised instead: `refusals` (CLI), `GT.mint` + `verify` through the chain's own `token_ok`, `profile_refusals`, the G-DVB check (`../NOTES.md`). The R0c `--preflight` on the A40 (the step-0 bit-equality `sxA == sxB`, so G11's `kl_tol 0.0` holds under `armA`) stays **INHERITED** from R0's banked preflight (`../NOTES.md` section 8).
* **Pod state is unknown to me:** the stage markers, C0's dumps, `launch_gate.key`, which package version is shipped, whether `Sayood/tanitad-refe-rls1-c0` is still there. Section 7 lists the `ls` checks that settle each in seconds.
* **The kill rules are the registered ones, unchanged.** The draft's analysis-level outcome table (O1..O5: PDMS B - A lower bounds, `epo_kl` thresholds) is NOT machine-enforced by the chain; only K0-K6 / g3-g5 are.
* `test_rls1_queue.py`: only the R0c part was added and run alone; the rest of that file is untouched code (its 13 mutation anchors still hit the wired chain exactly once: `logs/anchors_check.txt`). The full-file pass (134) is one sample of that intermittent harness.
* A SHIP_MANIFEST regeneration and the actual landing are the Master Mind's.

## 9. Findings for the Master Mind

1. **A real wiring trap, found by reading the arm loop, not by the draft:** with no change, `RLS1_ARMS="R0c"` would have been read against C0's dumps (R0's rule, because `RD=C${ARM:1:1}` happens to give `C0`) instead of its own arm A. The override line is what makes the draft's rule true; it is pinned statically and by behaviour (control-read block run with C0 present and absent) and by the whole-loop scenarios.
2. **`git apply` on this Windows box corrupts line endings** (`core.autocrlf=true`): CRLF in every patched file. The R0c build's NOTES recommended `patch -p1`; keep to it (or `-c core.autocrlf=false`).
3. **`write_sh` exec bit:** fixed in `test_rls1_chain.py` (and therefore in `test_rls1_queue.py`, which imports it). **`test_s1aux_chain.py:62` has the SAME latent defect** (its own `write_sh` has no `chmod`; `test_p2rep_chain.py:66` was already fixed on 2026-10-15). Not touched (outside this brief): on the pod its shims would fail with rc 126 exactly as described. One-line fix, same comment.
4. **The precheck did not require the gate's suite files** (`test_rl_s1b.py` was never in the list, although `launch_gate.RLS1B_SUITE` runs it: by that definition an R0b start from a package without it would be refused at G-SUITE after the queue wait and the preflight - inferred from the code, not run). Both suite files for `refe_rls1c` are now required up front (this also covers `test_rl_s1b.py` for R0b / R1b).
5. **`${ARM:1:1}` naming:** the original `RD=C<second character>` rule only works for arms named `R<digit>...`; R0c relies on it nowhere now (explicit `RD=C0`), but any future arm that does not follow the naming still needs its own line (that is what `RLS1B-INSERTION-POINT-3` is for).
6. The R0c build's note that `test_rls1_queue.py` "cannot run on Windows" should read "fails intermittently in `kmon_scenarios` (Win32 error 5 creating a signal pipe)"; it passed 101 on the unpatched tip here.

## 10. Deliverable manifest

All files are in the E: working tree of this checkout, **unstaged** (the brief's deliberate exception to "stage, never push": the stale index would sweep reverts into the next commit). The scratch trees I built on C: (`Temp/r0b` clean tip, `Temp/r0f` patch-stack proof tree, `Temp/r0w_base`) were deleted after the runs (C: was 100 % full); `Temp/r0w` (the working copy with every edit, CRLF in the seven files PATCH_1..7 touched because I applied them with plain `git apply`; do not take files from it, take the patches) and `Temp/r0w_logs` remain.

| artifact | where |
|---|---|
| `NOTES_CHAIN.md` (this file) | `repo(unstaged):TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/raw/2026-10-15-r0c-build/chain/` |
| `PATCH_8_rls1_pod.diff`, `PATCH_9_test_rls1_chain.diff`, `PATCH_10_test_rls1_queue.diff`, `PATCH_11_new_eval_rls1c_arms.diff`, `PATCH_12_runbook.diff` | same folder (**one place only on E:**; `Temp/r0w` on C: of this PC holds the same edits, which is not a second safe copy) |
| `logs/` (BEFORE and AFTER suite outputs, summaries, anchors check, mutation table) | `.../chain/logs/` |
