# YP7 CHANGES: the P7 package made launch-ready WITH the yaw lever (AMENDMENT YAW), 2026-11-07

Nothing staged, committed or pushed. The pod was never touched. The dev-box GPU lock was never touched. All tests ran with `CUDA_VISIBLE_DEVICES=-1` under `C:/Users/Admin/venvs/driverl-eval`.

## Base and verification
- Tip used: `b17b549232a80f72db28c7214468ec65eb443fe1` (`origin/agent/arch-inf-20260803`, fetched into `C:/Users/Admin/tanitad-push`). YT started at `a521c07`.
- **None of YT's 7 files changed on the tip.** The 4 modified files (`train.py`, `gate_checks.py`, `test_launch_gate.py`, `test_p7_head_wiring.py`) have, at the new tip, exactly the tip blobs YT recorded in `YT_CHANGES.md`; the 3 new files are absent. `git diff a521c07..b17b549 -- <PK>` is empty (the two commits touch only `GOALS_AND_CLAIMS.md` and `2026-10-17-v8f-mini/raw/`). No rebase was needed.
- Work tree: `C:/Users/Admin/pyp7` = a raw-blob export of the tip's whole PK (11,230 files, 0 differ from the tip, checked by `git hash-object --no-filters`), then YT's 7 files (blob-checked against `YT_tree/`), then YP7's edits. YT's `pxyaw` was NOT used as the base: it holds only `code/ eval/ refe/ splits/` (no `raw/`, no `SHIP_MANIFEST.json`, so the suite and the ship-list check cannot run on it) and carries a different `refe/p7_ship_list.json` than the tip.

## What YP7 changed (priority order of the brief)
1. `SPEC_P7.md`: MM's amendment inserted VERBATIM (byte-checked) as a new section directly after section 6 and before `## Deliverable manifest` (the manifest stays the file's footer).
2. `refe/launch_gate.py`, profile `refe_p7`:
   - `--ego-yaw-table` is a **required flag** (value any) and a **data flag**. The table is built on the pod, so its sha256 cannot be registered; it rides the launch token's data manifest (`fingerprint` = file sha256, re-measured by `verify` before every launch), exactly as the pod-built label store does, and the trainer writes it into the checkpoint meta and run identity (YT). The profile carries no `ego_yaw_sha256` pin (YT's optional pin key stays unused, and a test pins that).
   - `P7_SUITE_EXTRA` gains `{"script": "test_ego_yaw.py", "args": ["--fast"], "token": "ZZEGOYAW_PASS"}` (last entry).
   - `refe_p6` and the earlier profiles are untouched (P6's profile digest test passes).
3. `refe/p7_pod.sh`: new **stage 5b** `yaw` between the reserve stage and the metric-cache stage, so it runs after the P6 wait and the ship-list check, on the reserve-free bank the trainer reads (`CBANK` = `train_grow_cmd_3c_p7`), and before the gate.
   - `ego_yaw_table.py build --split navtrain --yaml $PKG/splits/navtrain.yaml --db-root $DATA/navtrain_dbs --out $DATA/refe_navtrain/ego_yaw_navtrain.json.gz`, never `--allow-partial` (the builder refuses rc 4 and writes nothing when any asked token has no DB row).
   - then `check-bank --table $YAWT --bank-dir $CBANK`; rc 0 AND `ZZYAWTABLE_BANK_PASS ... missing_rows=0` or REFUSE. The check runs on EVERY chain start (seconds); the build is skipped by a keyed marker (builder bytes + split file + DB count) and re-done when the table file is gone.
   - the trainer ARGS gain `--ego-yaw-table $YAWT` (after `--w-field 0.0`); the memory run, the gate's `--args`, the preflight and the trainer all carry it.
   - `P7_STOP_AFTER=yaw` is accepted; the package-file list requires `ego_yaw_table.py`, `ego_yaw.py`, `test_ego_yaw.py`.
   - Refusal texts contain none of the FD200 words (`curl`, `sleep "`, `tar -x`, `dd if=`, `apt-get`, `uv pip`, `find -exec`).
4. `refe/test_p7_chain.py` (81 -> 110 checks): ARGS pin, stage order (reserve filter < yaw build < check-bank < cache < gate), 8 new static rules each with a mutant (`YAW_ALLOW_PARTIAL`, `YAW_FD200`, `YAW_BUILD_ARGS`, `YAW_CHECK_NOT_ON_THE_TRAINERS_BANK`, `YAW_COVERAGE_NOT_100` x2, `YAW_NOT_IDEMPOTENT`, `YAW_BUILD_NOT_CHECKED`, `YAW_STAGE_MISSING`, the ARGS drop), sandbox arms: build refused (partial) / table covering < 100 % of the bank / second start skips the build but re-checks / a table swapped for a partial one is caught / a deleted table is rebuilt / builder missing from the package; 3 behavioural mutants (coverage verdict ignored, `--allow-partial`, flag not passed to the trainer). The sandbox's builder is a stub that writes a REAL table; `check-bank` is the REAL tool.
5. `refe/test_p7_integration.py` (221 -> 225): IV suite-tail pin extended; a new IV check that `refe_p7` requires the flag and carries it as a data flag while no other profile does; a new IV check that no sha pin is registered; mutants M26 (flag not a data flag), M27 (not required), M28 (suite entry dropped) all RED (`--mutations` run, log in `logs/`).
6. `refe/test_p7_head_wiring.py` (YT's file, 2 more edits): `P7_ARGS` carries the flag (a refe_p7 argv without it is now refused, as designed); S2's required-flag set includes `--ego-yaw-table`.
7. `refe/ego_yaw_table.py` (YT's file): `check-bank` now also prints `ZZYAWTABLE_BANK_PASS|FAIL rows= distinct_tokens= missing_rows=` (the chain greps it; rc semantics unchanged); `refe/test_ego_yaw.py` tests the CLI both ways.
8. `refe/test_ego_yaw.py` (YT's file) **made pod-safe**: its fast block `chk_loader` contained a git-mirror comparison that printed `ZZTEST_SKIP` when the mirror is absent. On the pod there is no mirror, and G-SUITE FAILS on any skip (`gate_checks.check_suite`), so the pod gate would have failed while the dev-box pre-run (mirror present) passed. The comparison moved verbatim into the local-only SLOW check `loader_vs_tip`. `--fast` now prints no `ZZTEST_SKIP` (0 lines, MEASURED).
9. `refe/p7_ship_list.json`: re-derived with a FRESH traced run at the final bytes (see results below).

## Decisions and caveats the MM should know
- **Order of the last two steps is derive THEN pre-run.** `refe/p7_ship_list.json` is inside the gate's tree digest (`.json` is a tree suffix), so a pre-run taken before the derive is bound to a tree that no longer exists. A sanity pre-run (`suite_prerun_PRE.*`) ran first only to find failures early; the evidence to use is `suite_prerun_FINAL.*`, taken on the exact final bytes.
- **The pre-run is bound to the SHIP_MANIFEST's commit (`22b40687...`, the tip's manifest) because `_commit_of` prefers the manifest.** MM regenerates `SHIP_MANIFEST.json` at ship time for the landed commit; the tree digest then still matches (the manifest is outside the tree roots) but the commit does not, so re-run `suite-prerun` after landing if the pod gate needs `--suite-evidence` bound to the landed commit (about 13-15 min).
- **No `control` run in the chain.** `ego_yaw_table.py control` (bit-equality of vx/vy/ax/ay against `EgoState` and the bank's ego[0..3]) needs the nuPlan devkit and maps and was run by YT on the dev box on 12 logs (`evidence/control_navtrain.json`); the join code is the same on the pod. The chain does the two checks the brief names (build at 100 % of the asked tokens, check-bank at 100 % of the bank).
- The pod's table build reads 1,192 SQLite DBs once (CPU, no GPU), after P6 has released its lock. Its duration is UNVERIFIED on the pod (YT's dev-box build of 539 logs: see `evidence/build_navtrain.log`).
- Navtest consequence from the amendment is unchanged and out of scope here: P7's navtest reads feed the navtest table through `refe/ego_yaw.py`.

## Files changed or new, with the tip blob (= the land base guard) and the final blob

Tip: `b17b549232a80f72db28c7214468ec65eb443fe1` (`agent/arch-inf-20260803`, fetched into the mirror `C:/Users/Admin/tanitad-push`). Paths are relative to `TanitAD Research Lab/Architecture & Inference/Research/2026-09-20-refe-plan/`. Blob = `git hash-object --no-filters`.

| file | change | tip blob (base guard) | final blob | by |
|---|---|---|---|---|
| `SPEC_P7.md` | MODIFIED | `f1fd04c2e5f0a652c611b89b485f5b22d0af9ca5` | `cf689338ca6ae364516270821152efdc29d8c785` | YP7 |
| `refe/ego_yaw.py` | NEW | `(absent at the tip)` | `5bac0b0da6a75244a0a4b024c95045d7fe9076d7` | YT (unchanged by YP7) |
| `refe/ego_yaw_table.py` | NEW | `(absent at the tip)` | `97cf698edaaa772a6c9a4baf13b80877e8890f31` | YT + YP7 |
| `refe/gate_checks.py` | MODIFIED | `30f7efa749db3b59ff14873905c351d13115f675` | `84d7dfb3cefe2c97b9ea72e20a5f6d7f4182c265` | YT (unchanged by YP7) |
| `refe/launch_gate.py` | MODIFIED | `b00b1aef669b73236d398317fc10d134c06b7a43` | `e705ced75ec19d35dc907f69fdc183691f15025c` | YP7 |
| `refe/p7_pod.sh` | MODIFIED | `a71a4be26af21ea86e8a01330a0e17915bc20538` | `4428ddc4e656c7b4ab6b95a49e77f7b792fdf272` | YP7 |
| `refe/p7_ship_list.json` | MODIFIED | `7da916ade3fd39f5c29aaba6b8540c3f55b68d3e` | `cc284fc04a332ff99a4173eb9aae5f4d317c941c` | YP7 |
| `refe/test_ego_yaw.py` | NEW | `(absent at the tip)` | `5a61e43f7735434e861a14a1535882e8696371e6` | YT + YP7 |
| `refe/test_launch_gate.py` | MODIFIED | `2a9a45098a72d2887cfb5eab089f35fcc7c71648` | `ae18f396ad878d40fedfa672908538a7126cc25f` | YT (unchanged by YP7) |
| `refe/test_p7_chain.py` | MODIFIED | `3fc62cc7996370465179d3697d8bc21ea8083aa6` | `e17b4c60bb74f45c7cc1313c5cc8cab7928f0875` | YP7 |
| `refe/test_p7_head_wiring.py` | MODIFIED | `51cea17c6927fafc7b0c9ee13554a1893e04d6b6` | `48e8ef5a68574f0150b2cd9b44126b0ba4bb2cd8` | YT + YP7 |
| `refe/test_p7_integration.py` | MODIFIED | `2c16e5544ce3e8c895348a81f61632ea20044377` | `7efa505fd48dc81d93898052cf154244f2bcc2b2` | YP7 |
| `refe/train.py` | MODIFIED | `0b2170eb5c1357471a06152374f4b463a86e2c3c` | `0e115c0d4d3e8ea60d613035114d2f8f64db672f` | YT (unchanged by YP7) |

Mirror of every row above: `E:/Projects/TanitAD-artifacts/yawrate_wp/YP7/tree/<path>`. 13 files.
Unexpected changed files (not in the expected list): none. Expected but unchanged: none. Tip files missing from the work tree: 0.
`refe/planner.py` and everything under `eval/` are byte-identical to the tip (not in the table).

## Pipeline results

```
YP7 RESULT (generated 2026-11-07T03:01:30Z)

pre-run A (sanity, before the derive): suite-prerun PASS in 1220s -> E:/Projects/TanitAD-artifacts/yawrate_wp/YP7/suite_prerun_PRE.json (profile refe_p7, tree 2729a395a5bc, commit 22b406875ae8) | ZZSUITE-PRERUN-PASSZZ
ship-list derive:     outside raw/2026-10-13-aux-head/PREREG_BLANKS_PROPOSED.md  via=static  readers=refe/test_s1aux_chain.py |   outside raw/2026-10-18-r0e-pick/PREREG_R0E_PICK.md  via=static+trace  readers=refe/test_launch_gate.py,test_launch_gate.py |   outside raw/2026-10-22-p7-session-reserve/raw/src/navtrain_token_table.tsv.gz  via=trace  readers=test_session_reserve.py
ship-list check:    ZZP7SHIP_OK files=296 scripts=106 closure=adf9d8f49e4a
FINAL suite-prerun: suite-prerun FAIL in 1330s -> E:/Projects/TanitAD-artifacts/yawrate_wp/YP7/suite_prerun_FINAL.json (profile refe_p7, tree b1ebeb46ee0d, commit 22b406875ae8) | ZZSUITE-PRERUN-FAILZZ
full test_ego_yaw.py (local, incl. SLOW + mutation harness): (absent)
ship list: files=296 scripts=106 closure=adf9d8f49e4a trace=run now; in the list: {'ego_yaw_table.py': {'in_scripts': True, 'in_files': True}, 'ego_yaw.py': {'in_scripts': True, 'in_files': True}, 'test_ego_yaw.py': {'in_scripts': True, 'in_files': True}}; suite=['test_p2r_chain.py', 'test_launch_gate.py', 'test_p6_levers.py', 'test_p6_chain.py', 'test_p7_head_wiring.py', 'test_session_reserve.py', 'test_p7_integration.py', 'test_p7_chain.py', 'test_ego_yaw.py']
changed/new files: 13; unexpected: []; expected-but-unchanged: []
```
