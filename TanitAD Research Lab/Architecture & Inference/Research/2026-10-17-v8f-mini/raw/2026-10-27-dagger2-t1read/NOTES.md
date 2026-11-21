# dag2t1: the dev-box CPU T1-kin read of PREREG_V8F_DAGGER2 (BUILT AND SELF-TESTED, NOT LAUNCHED), 2026-10-27

> **UPDATE 2026-10-27 (`AMEND1.md`)**: PREREG_V8F_DAGGER2 Amendment 1 (`A1-2026-10-27`, `1446333d`) placed the 4 unplaced cells, so section 6 item 1 below is RESOLVED and the PREREG pin in this file (`1d4f4fbb`, sha `87d6c77d...`) is superseded by `1446333d`, sha `c4239e45...` (staged at `inputs/PREREG_V8F_DAGGER2.md`). Self-test now 196/196, mutants 68/68 (see `AMEND1.md` for what is not run under the blinding). The rest of this file is the pre-amendment build record.

Nothing was launched or trained, no rollout and no smoke run was made, box 2 was not contacted, nothing was committed, staged in git or pushed. Everything is under `E:/Projects/TanitAD-artifacts/dag2t1/`. Evidence classes: MEASURED (file) / INHERITED / UNVERIFIED.

PREREG (binding, changed nothing): `Project Steering/PREREG_V8F_DAGGER2.md` at `1d4f4fbb`, sha256 `87d6c77dc3362bfb65af1e8966c913b957143f88d9c04b3867576f4bba423e39` (the pinned copy on E: is byte-identical to the tip blob). It inherits PREREG_V8F_DAGGER (+ Amendments 1, 2) and adds: the crawl set C, R-CRAWL, R-MOVE, the guards R-TLAT / R-TLON / R-ADE (DAG2 - DAG2-0), R-T0, the "no Read-A row newly separated-worse vs DAG - P10" clause, the fixed replicate floor RF, and the outcome table.

## 1. What is here

| file | role |
|---|---|
| `dag2_t1_chain.sh` (+ `dag2_t1_chain.cmd`, detached WMI start) | ONE chain, zero training, zero GPU. For each run in the PREREG order (DAG2 s0, DAG2-0 s0, DAG2 s1, DAG2-0 s1): wait until box 2's pull is COMPLETE by content, then two plain CPU T1-kin rollouts (`--limit causal`, decisive; `--limit logged`, K4 only) and the fresh-age T0 eval; then the bank gate, the manifest and the reader. Re-parametrised from `dagt1/dag_t1_chain.sh` by `mk_dag2_helpers.py` (`chain.diff` is the change, 277 diff lines) |
| `dag2t1_pullcheck.py` | content check of one pulled run / one iteration-2 bank (derived, `pullcheck.diff`) |
| `dag2t1_manifest.py` | writes `runs.json` for the reader; imports DAG's sha-pinned `dagt1_manifest.py` and `dagt1_pullcheck.py` unchanged; subcommands `manifest verify initsha dagbase treetip ids parity` |
| `dag2_read.py` | the reader: composes the EXISTING readers, re-derives nothing (section 3). Writes `v8f-dagger2-t1/read/DAG2_READ.json` |
| `readA_code/{rr_lib,readA,build_report}.py` | sha-verified copies of `dagreread/code/` (Read A's driver code). SINGLE-COPY: not in the repo at `1d4f4fbb` |
| `inputs/b1_eval_lead_block_k30.npz` | 4 MB, sha-verified copy (`7a879baa...`) of the repo blob; the earlier default lived in a session scratchpad |
| `mk_manifest.py`, `MANIFEST.txt` | the inventory with sha256 and the single-copy marks |
| `dag2t1_selftest.py`, `selftest_output.txt`, `mutation.json` | the self-test and the mutation proof (section 5) |
| `dry_run_real_config.txt`, `preflight/`, `repro/` | the evidence runs (section 4) |

## 2. Install and launch (nothing to install into the repo; the tool tree is DAG's)

Preconditions (the chain checks each and STOPs, never waits on, a wrong one): C: >= 2048 MB free (start AND before every rollout; PREREG section 8) and E: >= 1500 MB free (start; 200 MB per rollout). **Today C: has ~0.45 GB and E: ~1.5 GB free (MEASURED `df -m` at 14:18; E: is at its own 1500 MB start gate), so a real launch STOPS until the PI frees C:** (PI-only deletes; `dry_run_real_config.txt` was produced with `DAG2T1_TEST_CFREE_MB=3000`). The chain never deletes anything.

Dry run (prints every guard, path, pull state and the EXACT command of every rollout, eval and the reader; touches nothing):
```
bash dag2_t1_chain.sh --dry-run
```
Real launch, detached, with the box-2 stop watcher (a WMI process does not inherit the shell, so the environment goes in the command):
```
$si = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{CreateFlags=[uint32]0x08000200}
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine='cmd.exe /c set DAG2T1_B2_OUT=C:/Users/Admin/box2_ship/logs/chain_v8f_dag2.out&& E:\Projects\TanitAD-artifacts\dag2t1\dag2_t1_chain.cmd'; ProcessStartupInformation=$si}
```
Markers (last wins): `E:/Projects/TanitAD-artifacts/dag2t1/logs/chain.out`: `ZZDAG2T1-DONE <n>/8 rollouts, reader rc <rc>` or `ZZDAG2T1-STOP <reason>`. Reader rc: 0 outcome written (`v8f-dagger2-t1/read/DAG2_READ.json`), 2 REFUSED (a control failed; nothing written; `logs/dag2_read.out` lists it), 3 UNPLACED (section 6, item 1), 4 BLOCKED (coverage < 0.5).
Waits: <= 40 h for pulls (`DAG2T1_MAX_WAIT_H`), <= 12 h for the banks (`DAG2T1_BANK_WAIT_H`); the chain is a bash script, so waiting costs no model tokens. ESTIMATED compute once a pull lands: about 5 min for the rollout pair (two lanes) and 5 min for the fresh T0, from DAG's `dagt1/logs/chain.out` (same tool tree, same CPU); the reader took 61 s on the repro.

Outputs go ONLY to `E:/Projects/TanitAD-artifacts/v8f-dagger2-t1/` (DAG's `v8f-dagger-t1/` is read, never written; a mutant that points OUT there is killed). Estimated 1.3 GB (DAG's were 1.3 GB, MEASURED).

## 3. How the reader is composed (nothing re-derived)

* **Layer G** = `C:/Users/Admin/dagread/dag_read.py` `read()` (sha `1cf56c7c...`), slots DAG := DAG2, DAG0 := DAG2-0. It supplies R-TLAT, R-TLON, R-ADE (the lateral / longitudinal heads and ADE rows, DAG2 - DAG2-0), R-T0, every K-control, the bank controls, COVERAGE (>= 0.5 or BLOCKED), S-POS vs CTRV. Two controlled patches, both AST-extracted and re-presented to the reader's own check: `g_argv` (the iteration-2 init is `M2b_s<S>_dag`, not `M2b_s<S>`; the check is presented the M2b tail and the record is not mutated) and `g_replicate` (RF fixed: the ORC0 slot must be DAG2-0's own entry and the RF row must read exactly 0.0; RF := the PREREG constants; bar = max(2F, 2RF), inclusive). `BANKED_M2B_SHA` := DAG's checkpoint shas (`d186fd0b...` s0, `1d12ec2e...` s1).
* **Layer A** = Read A's own functions (`readA.rows_block`, `t1kf_read_a1`, `build_report.collect/judge/verdict/outcome`) called once per pair: DAG2 - P10 (R-CRAWL, newly-worse), DAG2 - DAG2-0 (R-MOVE), and DAG - P10 (the newly-worse baseline, from DAG's banked dumps).
* **Composition** (`compose()`, pure, the part the self-test drives with synthetic rows): per seed R-CRAWL = no C row separated-worse for DAG2 - P10; R-MOVE = every C row separated-better for DAG2 - DAG2-0; guards = R-TLAT, R-TLON, R-ADE not separated-worse (DAG2 - DAG2-0), R-T0 point <= 0.0604, newly-worse clause held (accel MAE and DK-VIOL exempt but REPORTED when newly worse). C is s0 {curvature MAE, L2 curvature MAE, M1 crawl share}, s1 {the same + yaw-rate MAE}. The outcome table is coded in the PREREG's order: REFUSED, BLOCKED, then HARMS if any guard fails at either seed, then FIXED / PARTIAL / NO EFFECT.
* RF bars (lat 0.0248 / 0.0248, lon 0.0502 / 0.0278) are computed and compared with the PREREG's printed bars (control RF-BARS) and reported beside the outcome (`clears_dagger_bar_beside`). **They have no consumer in the DAG2 table** (section 6, item 2).

## 4. MEASURED (ours, with the artifact)

* Static check: 15 sha256 pins re-hashed, every PREREG outcome-table cell verbatim in the pinned text (`dry_run_real_config.txt`; `python dag2_read.py --static-check`).
* **RC-DAG**: layer A re-computes DAG's banked Read A table from DAG's banked dumps with ZERO differences (19 rows x 2 seeds) (`repro/REPRO_DAG_vs_READ_A.json`). So the composition is DAG's reader, not a re-implementation.
* The PREREG crawl set equals the banked separated-worse crawl rows per seed (self-test C3, from `dagreread/raw/READ_A.json`).
* Preflight on the real pulled iteration-2 banks (s0, s1) and DAG2 / DAG2-0 s0: the patched DAGGER bank and run-record controls PASS (45 controls, 0 failed); coverage 0.773 (s0) / 0.764 (s1) so it does not BLOCK (`preflight/PREFLIGHT.json`). The patched ARGV control REFUSES the iteration-1 DAG record (init M2b) (self-test L9).
* The pull check COMPLETEs the real pulled `M2b_s0_dag2` and `M2b_s0_dag20` and both banks, and refuses every deliberate wrong case (self-test P).
* `bash dag2_t1_chain.sh --dry-run` on the real configuration ends `ZZDAG2T1-DRY-OK` (rc 0); s1 shows INCOMPLETE because box 2 has not pulled it yet.
* **Self-test: 141/141 checks as expected (logic 109, pull check 21, chain dry-run scenarios 11) and 53/53 mutants killed (37 of `dag2_read.py`, 6 of `dag2t1_pullcheck.py`, 10 of `dag2_t1_chain.sh`), unmutated control green** (`selftest_output.txt`, `mutation.json`; 262 s on this box). Two mutants survived the FIRST run and were fixed in the test, not the code: a redundant defence (the `--dagger-p` arm clause was masked by a second summary.json clause, and the chain's PREREG pin by dag2_read's own pin of the same file); the tests now make the first line of defence the thing under test.

## 5. Self-test and mutation proof (`dag2t1_selftest.py`; plain script, no pytest, CPU, no rollout)

Logic (outcome table: one case per outcome plus all 16 crawl/move combinations against a hand-written expected table; HARMS per guard and per seed; RF bars; the crawl set per seed; the newly-worse clause with its two exemptions; the two layer-G patches on stubs and on REAL records), the pull check on synthetic runs and on the real pulled pair, and the chain dry-run (run order, arm names, tags, paths, the C: and E: gates at 1500 / 2047 / 2048 MB, tampered PREREG, wrong tree, the removed box-2-fresh switch). Mutation proof: each mutant is ONE textual defect in a COPY of `dag2_read.py`, `dag2t1_pullcheck.py` or `dag2_t1_chain.sh`; it is KILLED iff a check goes RED; an anchor that does not occur exactly once is an ANCHOR ERROR; the unmutated control must be green. Per-mutant result: `selftest_output.txt`, `mutation.json`.

## 6. PREREG issues found while building (ESCALATE before any DAG2 row is read; the PI decides, I changed nothing)

1. **The outcome table is not literally complete.** Over (R-CRAWL_0, R-CRAWL_1, R-MOVE_0, R-MOVE_1) with all guards held, 16 combinations: NO EFFECT 4, FIXED 1, PARTIAL 7, and **4 match no row**: R-MOVE holds at exactly ONE seed AND R-CRAWL fails at that same seed (e.g. R-MOVE_0 and not R-CRAWL_0, with R-MOVE_1 false; NO EFFECT needs R-MOVE false at both seeds, PARTIAL needs either R-MOVE at both seeds or a seed where both hold). The reader reports these as `UNPLACED` (rc 3), invents no outcome word and names no committed next. The PREREG needs an amendment that places them, before any row is read. The s0 pair is already pulled but no row has been read.
2. **The RF bars have no consumer.** The guards are "not separated-worse"; no cell compares a delta with a bar. The reader reports `clears_dagger_bar_beside` and nothing decides on it.
3. **Disk.** C: ~0.45 GB < the 2048 MB gate; E: ~1.5 GB free vs ~1.3 GB of outputs (E: gate 1500 MB). This directory is 4.9 MB apparent (limit 50 MB) but ~75 MB ALLOCATED on exFAT (1 MiB clusters, ~80 small files); I deleted nothing.
4. **Single-copy**: Read A's driver code (`dagreread/code/{rr_lib,readA,build_report}.py`) is not in the repo at `1d4f4fbb` (MEASURED: `git ls-tree -r` over the tip's 49,511 paths has no `rr_lib.py` or `readA.py`, and its only `build_report.py` of that lineage is a different file, sha 15d11d6c vs the pinned c4e94920); the staged copies in `readA_code/` are the only second copy. By contrast `dag_read.py` (sha 1cf56c7c) IS in the repo, byte-identical, at `.../2026-10-26-dagger-read/dag_read.py`.

## 7. NOT BUILT / UNVERIFIED

* **Layer G + layer A end to end on REAL DAG2 rollouts is UNVERIFIED**: no DAG2 rollout exists. What is MEASURED is the static, preflight, repro and synthetic-composition coverage above.
* The T0 four-family paired beside rows and the HACC re-fit are NOT BUILT (the former has no instrument; the latter is declared `NOT RUN` in the output).
* TAC / DK arm-mean columns are not carried (None) in the beside rows.
* DAG's SMOKE mode was not carried (it needs a 87 MB stand-in checkpoint and two torch processes on a nearly full C:).
* The s1 pair is not pulled (box 2), so s1 pullcheck / bank gates were only exercised on synthetic fixtures and the s1 banks.
