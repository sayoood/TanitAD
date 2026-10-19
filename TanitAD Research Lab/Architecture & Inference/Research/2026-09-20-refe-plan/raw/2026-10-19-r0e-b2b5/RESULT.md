# R0E-B2B5 -- RESULT (2026-10-19): launch-gate profile + pod chain for R0e-id / R0e-pick / R1e-pick

Built on tip `1e62e02`, then **rebuilt on `6400a47`** (HNSCREEN-FP-REQUIRED; branch tip now `807a29c`, which changes none of my files). Base blobs I built on: `refe/launch_gate.py` `ec7a21baf23c5f014e3c606393da676ed7a4a1f5`, `refe/test_launch_gate.py` `d4bd6f3c85f4eba92413de602b69bdeb7396ee35`; the other four edited files are unchanged between the tips. Rebuild method: `git merge-file` 3-way (mine / `1e62e02` blob / `6400a47` blob), **0 conflicts** on both files (the tip's hunks are inside `hnscreen_refusals` and `hnscreen_static`, mine are elsewhere), then every number below was re-measured on the merged tree. Every number is **MEASURED** on the dev box, CPU only (`CUDA_VISIBLE_DEVICES=-1`), by the test named. Nothing was pushed or committed; no token was minted; no pod, ssh or GPU was touched.

## HEADLINE: integration escalations (read these first)

1. **B9 -- the third launch blocker, NOT built: R0e-id's "K3 observe-only" (PREREG_R0E_PICK section 4).** `refe/rls1_checks.py:738` refuses `--observe-only` for any arm but R1o, and `test_rl_safeobj.py` pins `OBS="--observe-only"` count == 1 and `CH_OBS_is_observe_only_for_R1o_alone`. As built, **R0e-id runs with K3 LIVE**. Exact fix: allow `a.arm in ("R1o", "R0e-id")` at `rls1_checks.py:738`, set `OBS="--observe-only"` for `R0e-id` in the `OBS` block of `rls1_pod.sh`, and move the three pins. It touches the kill monitor, so it needs its own regression cycle; not folded in here.
2. **No ID-reader gate exists for "R0e-pick only after R0e-id reads ID-PASS"** (PREREG section 5). The chain registers all three arms; the order is operator-sequenced (`RLS1_ARMS="R0e-id"`, then `"R0e-pick"`). Proposed B10: a chain refusal that reads R0e-id's bank before R0e-pick starts.
3. **`SHIP_MANIFEST.json` must be regenerated at landing** (the pod precheck now requires `eval/rls1e_arms.json` and `refe/test_rl_pickcon.py`; the manifest is not edited here).
4. **`RLS1_R1O_LINEAGE=e` must be exported for the R1o run**, else R1o is the SAFEOBJ line (gated on R0d and R1d), exactly as before. Any other value than `d` / `e` is refused at the precheck.
5. **Disk (acknowledging the coordinator's note, no reply needed):** C: was at 0 bytes free at 10:30-10:36 and it broke runs of mine (`Errno 28` writing a lane log; base `test_p6_chain` failed its own "dd write test 2 GB"; both re-run, see the table). Every staged file was re-verified by sha256 and size against `tree/` and `tree_b5/` after the space came back. Still on C: and not mine to delete: three runaway task-output files of about 5 GB each under `Temp\claude\` (`E--Projects-TanitAD\c216c849-...\tasks\bm5nhc0pv.output` 5,121 MB, `...\tasks\bb2x8d9m6.output` 5,047 MB, `D--Projects-TanitAD\7bf339a1-...\tasks\b00zzvgfg.output` 5,123 MB), `Temp\p2f` 7.0 GB, `Temp\lg_full_*` about 8 GB and 1,594 `tmp*` directories made in 3 hours. In `par18/r0eb/` only `tree/` is the bank source: `tree_b5/` is a byte-identical test copy and `base/` a pristine tip export, both removable once this is landed.

## What was built

**B2 -- gate profile `refe_rls1e`** (`refe/launch_gate.py`; `refe_rls1`'s runner and checks, its own arm table).
- Arms and registered lever: `R0e-id` T 0.1 (seed 0, to u1000), `R0e-pick` T 0.01 (seed 0), `R1e-pick` T 0.01 (seed 1, Stage 2 only on O1), `R1o` (seed 1, no lever).
- `arm_must_name` for the three T arms = the five `--epo-con*` flags **plus `--epo-con-temp`**, so the omitted lever (parser default `None` inherits `--epo-temp` 0.1 and silently reproduces R0d) is refused. R0e-id's T equals that default, so only the must-name rule can see its omission. `arm_epo_registered` holds the value per arm (a wrong T is refused).
- **Lineage rule** (`rls1e_lineage_refusals`, new `lineage: "rls1e"` key): with the lever pair removed, own paths swapped back to `/arms/R0d` and the seed read as R0d's, the argv must equal `arms.R0d.argv` **token for token**; `lever_tokens_not_in_R0d` is exactly {`--epo-con-temp`, `<T>`, own paths} (+ `1` for R1e-pick's seed). It is the only rule that sees `--lane-weight 0.5`, `--workers 4`, `--max-updates 250` (not an identity argv: the cosine schedule runs over `min(total, max_updates)`) or `--epo-con-feed carve` (the NEXT arm's lever).
- **R1o forbids the lever** (`arm_forbid_flags`, new key; the trainer's own `epo_con_refusals` refuses it too, defence in depth).
- G-SUITE for the profile = `RLS1D_SUITE` + `test_rl_pickcon.py`. No other profile carries the new keys (E1 asserts it).
- The registered argvs are a **constant in the gate** (the token binds `refe/`, `code/`, `splits/` only, not `eval/`), cross-checked token for token against `eval/rls1e_arms.json` and `eval/rls1d_arms.json` by the tests.

**B5 -- the pod chain** (`refe/rls1_pod.sh`).
- `REGISTERED_ARMS` += `R0e-id R0e-pick R1e-pick`; `arm_profile` -> `refe_rls1e`; `arm_args` = R0d's epo string + `--epo-con-temp 0.1 | 0.01 | 0.01`.
- Each e arm is read against **its own arm A** (`REFDIR=$ARMDIR/scores`); the seed-matched control is C0 (seed 0) / C1 (seed 1); `ZZRLS1_R0E_NO_SECOND_CONTROL` is logged when its dumps are not on the pod.
- `RLS1_R1O_LINEAGE=e`: R1o is judged under `refe_rls1e` and the S gate runs `safeobj-gate --dirs $R/arms/R0e-pick $R/arms/R1e-pick` (R1o refuses unless BOTH read bar S = PASS). Default `d` keeps the SAFEOBJ lines verbatim.
- Precheck file list += `refe/test_rl_pickcon.py`, `eval/rls1e_arms.json`.

**New files:** `eval/rls1e_arms.json` (schema `rls1e_arms/1`: argv, argv_sha256, self-check per arm), its builder `raw/2026-10-19-r0e-b2b5/build/make_rls1e_arms_json.py`, and `refe/test_rls1e_chain.py` (pod chain: argv per arm == the arms file token for token, profile choice, R1o gate dirs, precheck list; 16 RED mutations).

**Deliberate-regression arms** (each makes the gate REFUSE, asserted in `rls1e_static()` of `test_launch_gate.py`, sections E1-E7, and each makes EVERY check FAIL with nothing started): omit `--epo-con-temp`; wrong T (0.02); extra lever token (`--lane-weight 0.5`); a MOVED token (`--workers 4`, named as MOVED, alone); `--epo-con-feed carve`; `--max-updates 250`; duplicated lever; abbreviation `--epo-con-t`; `--epo-temp` named; R1o carrying `--epo-con-temp` (profile rule AND trainer rule); two flags reordered. Three MUTATIONS prove each rule is what refuses its arm (M1 lineage off -> the extra lever walks through; M2 must-name off -> R0e-id's omitted lever is CLEAN; M3 forbid off -> only the trainer still speaks).

## Dry gate checks (CLI `launch_gate.py refusals`, no token, no GPU; re-run on the `6400a47`-based tree) -- `ref/logs/dry_check.json`

| argv / profile / arm | rc | verdict |
|---|---|---|
| **R0e-pick registered argv, `refe_rls1e`, R0e-pick** | 0 | **CLEAN (PASS)** |
| R0e-id / R1e-pick / R1o registered argvs, own arms | 0 | CLEAN |
| **R0d registered argv, `refe_rls1e`, R0e-pick** | 5 | **REFUSED**: requires `--epo-con-temp` NAMED ("a flag whose default reproduces the old arm must be named") |
| R0d argv under R0e-id | 5 | REFUSED (same rule) |
| R0d argv under `refe_rls1d`, R0d (unchanged profile) | 0 | CLEAN |
| R1o + `--epo-con-temp 0.01` | 5 | REFUSED (trainer: lever set but `--epo-con` off) |
| R0e-pick minus the lever / T 0.02 / plus `--lane-weight 0.5` | 5 / 5 / 5 | REFUSED: must-name / "resolves to 0.02" / "it ADDS --lane-weight 0.5 (a second lever)" |

Only the `refusals` stage is dry-runnable here: G-LIVE needs real data and a GPU, so a full `run` cannot PASS on the dev box. The `con_temp_from_arms_file` reader of `eval/safeobj_outcome.py` returns R0e-id 0.1, R0e-pick 0.01, R1e-pick 0.01 and raises `ValueError` for R1o (no constraint).

## Base vs mine (per test; same venv, same flags)

**Gate suite on the `6400a47` tip (base = pristine export, mine = rebased tree):**

| test | base | mine |
|---|---|---|
| `test_launch_gate.py --static --no-banked` | ZZLG_STATIC_PASS 957 | **1148** (+191 = `rls1e_static()` E1-E7, 0 skips) |
| `test_launch_gate.py` (full) | ZZLG_FULL_PASS 1002 (377 s) | **1193** (417 s); `refe_hnscreen` and every older profile green |

(At the old tip `1e62e02` the same pair read 926 / 971 base and 1117 / 1162 mine: +191 both times. The tip's HNSCREEN-FP change adds 31.)

**Tests that read an edited file.** Rows marked R were **re-run on the rebased tree**; the rest were run before the rebase on a tree whose `launch_gate.py` differed only by the tip's `hnscreen_refusals` hunk, and are shown with their pre-rebase base number (the tip changes nothing they pin: `rls1_pod.sh` and the four other files are byte-identical between the tips).

| test | base | mine | note |
|---|---|---|---|
| `refe/test_rls1e_chain.py` (NEW) R | -- | 936 | 1 unmutated + 16 RED mutations |
| `test_rl_safeobj.py` R | 61 | 61 | pins edited |
| `test_rls1_chain.py` R | 522 | 522 | |
| `test_rls1_queue.py` R | **rc 1 twice** (below) | **254** (758 s) | see note |
| `test_rl_r0c.py` R | 26 | 26 | `--arm-name` choices snippet pin kept |
| `eval/test_safeobj_gate.py` R | 155 | 155 | |
| `test_rl_pickcon.py` / `test_rl_s1b.py` / `test_carve_feed_gate.py` R | 20 / 18 / 10 | 20 / 18 / 10 | |
| `test_suite_nesting.py` R | 102 | **110** | +8: it loops `PROFILES` x suite entries, so the new profile adds its checks |
| `test_sft4r_gate.py` R | 153 | 153 | |
| `test_hnscreen_chain.py` R | 16 | 16 | the `refe_hnscreen` profile |
| `test_p6_levers.py` R | 199 | 199 | base re-run after its first run died on ENOSPC |
| `test_flock_rw.py` | 59 | 59 | |
| `test_gate_gpu_probes.py` | 14 (SKIP_CUDA) | 14 (SKIP_CUDA) | GPU check did not run on either |
| `test_sft4r_chain.py` | 31 | 31 | |
| `test_s1c.py` / `test_aux_s1_wiring.py` | 220 / 20 | 220 / 20 | |
| `test_aux_g12_bf16.py` | rc 0, no token | rc 0, no token | prints a "needs a GPU" notice on both |
| `test_p2rep_chain.py` / `test_s1aux_chain.py` / `test_p2r_chain.py` | 353 / 256 / 139 | 353 / 256 / 139 | |
| `test_gate_suite_prerun.py` | 144 | 144 | |
| `test_p6_chain.py` | 351 (re-run after ENOSPC; the first base run failed its "dd write test 2 GB") | 351 | |
| `eval/test_hn_screen_read.py` | rc 1 | rc 1 | **identical environmental failure on both**: `F_adapter_dir_missing (C:\Users\taniteval; set TANITEVAL_DIR ...)` |

**Queue test note.** `test_rls1_queue.py` is timing-sensitive on this box. On mine it passed 3 of 4 runs: 254 at 07:30, 254 at 10:51, 254 on the rebased tree; the one failure (09:31, "kill monitor READY" absent) fell in the C:-full window. Base failed **both** of its runs (10:03 at `kmon_scenarios`, 11:01 at `rls1d_checks`), at two different assertion sites that both look for the same "kill monitor READY" log line. The supervise code path is unchanged by my edits (the `rls1_pod.sh` diff is additive), so the base failures are a pre-existing race, not a regression of mine and not a pass of base's. INFERENCE for the cause (disk pressure for the first failures; a monitor-READY race for the last): not isolated.

## Edited files (blob sha1 at the `6400a47` tip vs new) -- also `out/EDITED_BLOBS.md`; all LF, 0 CR

| file | tip blob | new blob |
|---|---|---|
| `refe/launch_gate.py` | ec7a21baf23c5f014e3c606393da676ed7a4a1f5 | 2132b029ebf8f40574f91ee8910ce1ae510ea76e |
| `refe/test_launch_gate.py` | d4bd6f3c85f4eba92413de602b69bdeb7396ee35 | 2086fb124b0e71d3afe9fdf82cd5f386e677303a |
| `refe/rls1_pod.sh` | c1a14189cd15ede449618643c6e9fff2041e9985 | 95399d20c1bdae97ede1d0c525840a668c3cdf83 |
| `refe/test_rls1_chain.py` | c12dcdb1b3d7c1ffc1b2c69c3bc84372ffb78af7 | 3fba5b19b1167c320766c5a205b9c77a4c03bdd1 |
| `refe/test_rls1_queue.py` | 6c40ecbf3183ede95dcaf1016cb2815c5b53c186 | 0f3363dd6ffd961c5764aff15808ac4d90b9a501 |
| `refe/test_rl_safeobj.py` | ff1b2fcdc3fff1d84a1c7b4f460cbd29411e1bf7 | 0eae9540bf0d6cb053d6391febe4f23899c4b46e |
| `eval/rls1e_arms.json` (new) | -- | 857507d1184a58cb1264e76d8a48023d0098f1a8 |
| `refe/test_rls1e_chain.py` (new) | -- | 7831a3e6b0225e96aeb6a340d67c67a97d7fc32f |
| `raw/2026-10-19-r0e-b2b5/build/make_rls1e_arms_json.py` (new) | -- | 563d020c3236e23b7328c79fcabfac3bc526d98e |

The three older test files changed **pins only** (the `REGISTERED_ARMS` literal, the `safeobj-gate --dirs` count 1 -> 2 with the R0d/R1d line pinned exactly once, the registered-arms list). Three snippet pins in other tests constrained the source edits and were kept: the `--arm-name` choices text (`test_rl_r0c`), `"R1o": ("--epo-con",)}` occurring once in `launch_gate.py` (`test_rl_safeobj` mutation G1), and the `safeobj-gate --dirs` count in `rls1_pod.sh`. `eval/test_safeobj_outcome.py` (731 s) was not rerun; its `rls1_pod.sh` substring pins (`monitor died`, `monitor not READY`, `rlval_vs_A0.json`, the `--scores A0=... --base A0` string, and the absence of `rlval_B_vs_ownA`) were verified present by `grep -cF` (1 / 1 / 8 / 1 and 0).

## Proposed register rows (for the Master Mind to apply; I edited neither file)

- **BACKLOG**: `R0E-B2B5 | DONE 2026-10-19 | launch-gate profile refe_rls1e + pod chain for R0e-id / R0e-pick / R1e-pick (B2 + B5); gate suite 1002 -> 1193 | OPEN: B9 (R0e-id K3 observe-only), B10 (no ID-reader gate before R0e-pick), SHIP_MANIFEST regen`.
- **GAC** `D-REFE-R0E-B2B5` (2026-10-19): the launch gate now refuses, with a named reason each, R0e-pick / R0e-id / R1e-pick argvs that omit `--epo-con-temp`, name a wrong T, or add any second lever token, and R1o carrying the lever; the registered argvs equal R0d's token for token minus the lever. MEASURED: `test_launch_gate.py` full 1002 -> 1193, 0 skips; dry `refusals` CLEAN on the four registered argvs and REFUSED on R0d's argv under the R0e-pick profile. Claim scope: **reachability of the gate rule, not training correctness**; B9 and B10 are open.

## Deliverable manifest

Staged (not pushed, not committed) under `C:/Users/Admin/par18/r0eb/out/`: `paths.json` (`[repo path, abs source, "raw"]`), `EDITED_BLOBS.md`, `files/` (the 9 files above, byte-identical to `tree/`, verified by sha256 and size), `pkg/RESULT.md` (this file). Trees: `tree/` (mine, the bank source), `tree_b5/` (identical copy used by the test lanes), `base/` (pristine `6400a47` export for the five changed files, `1e62e02` elsewhere). Logs: `ref/logs/` (`mine_t2*`, `base_t2*`, `mine_b5r3*`, `base_b5r3*`, `mine_q*`, `base_q*`, `dry_check.json`). Helpers, not for landing: `dry_check.py`, `spot_con_temp.py`, `run_lane.py`, `bank.py`, `blobcheck.py`, `rebase/` (the three-way merge inputs and outputs).
