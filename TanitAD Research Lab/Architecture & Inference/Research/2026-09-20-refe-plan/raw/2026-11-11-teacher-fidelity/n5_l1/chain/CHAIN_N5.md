# CHAIN_N5: the TF-N5 navtest chain (built and tested, NOT launched)

Card: `../PREREG_TF_N5_DRAFT.md` (draft). Run tree (kept for the launch): `F:/tanitad_scratch/n5nt/run` (124 MB, copies, never links). Hold copy of every code file: this folder. CPU only, dev box, at most 2 processes, detached, marker-based, resumable (`n5run/markers/<stage>.done|.FAIL`, `ZZN5_CHAIN STOP`).

## Tree layout (copied from the pinned TF-N4 A1 run tree `F:/tanitad_scratch/n4nt/run_a1`, which was not modified)
| dir | content |
|---|---|
| `w/`, `repo/`, `n2/` | A1's harness package, taniteval repo, TF-N1/N2 analyzers, byte-identical, EXCEPT `w/pk/eval/instruments/families_run.py` = the TIP version (sha256 `44c62807f125...`, MEASURED: git `f938dca6` and `origin/agent/arch-inf-20260803` give the same bytes) |
| `n4/` | pinned A1 files the chain imports: `n4_geom.py` (`1966bb6d`), `n4_repair.py` (`1a4be60a`), `n4_nt_build.py` (`e74ebf9f`), `n4_common.py`, `fam_ci.py` (`8f7ada4e`) |
| `n5/` | `n4_repair_l1.py` (`7af0d3dd`), `n5_env.py`, `test_n4_repair_l1.py`, `fixture_real_instances.json` (byte-identical to the hold's), the NEW `n5_nt_chain.py`, `n5_nt_build.py`, `n5_nt_analyze.py`, `n5_make_pins.py`, `test_n5_chain.py`, `N5_PINS.json` |
| `inputs/` | A1's inputs plus `inputs/n4d/` (banked N4-D plans, seam, harness CSV, build report, A1 derive rows) and `inputs/n5/residual25.json` (the 25 split 5 guard-blocked / 20 non-guard from `ANATOMY_25` cause (b); the 25 equal the banked report's `unrepaired_dac0.N4D`, checked) |

## Stages and argv (a card flag is an argv flag: the build reads the parameters back from the object the argv built)
`n5_nt_chain.py --run-dir R --card C --card-sha256 S --pins-sha256 P --ledger L --k 47 --procs 2` (L1 flags default to the card; any other value is REFUSED).
| stage | what | est. wall (ESTIMATED from A1) |
|---|---|---|
| guards | card sha, ledger `TF-N5:<sha>:E3` n = 47 (TF-N4-A1 n = 46 present, nothing higher), pins sha + every pinned file + REQUIRED pinned + no unpinned .py/.json under w/ repo/ n2/ n4/ n5/ + metric-cache listing digest + **pinned `families_run.py` is the TIP sha** + F: >= 20 GB + `reach` (read-back AND behavioural signature of the three built repair objects) | 5 s |
| reach_real | the two real navtrain fixtures: zero-correction identity, sign +1 certifies, sign -1 does not, determinism | 25 s |
| derive (2 shards) | builder tracked score of the N2-R plan on all 12,146 tokens (C-DACMAP); L1 on the 116 DAC-0 tokens (N5-R all; N5-A / N5-X on the 25; each run twice for C-DET); C-DERIV 200 tokens | 0.9 to 1.1 h (A1 derive 1.1 h with 3 arms) |
| build | merge: C-COVB, C-DACMAP, C-DERIV, C-LABELEGO, C-DET, C-SET, C-CERT-B, C-SEAMFN, C-UNCH, C-REACH; plans + seams | 15 min |
| unch | C-UNCH on the files on disk | 10 min |
| score_A_X | N5-A and N5-X FULL 12,146 seams, 2 processes in parallel, through `score_p4_a1.py` (the pinned W3 harness); PASS + 12,146 rows | 1.2 to 1.7 h |
| read | `n5_nt_analyze.py read`: controls first (exit 1 before any N5-A aggregate), then the card bars | 2 min |
| families | transp rows (N4D, N5A, human) -> famprep -> **rows check** -> families_run `--per-window` (N4D, N5A) -> **tokens/logs check** -> fam_ci paired (A = N4D, B = N5A) -> famread; sub-markers `families_rows/run/ci` | 40 to 50 min |
Total ESTIMATED 3.5 to 4.5 h (A1: 7.3 h with three sequential seams). RAM: each worker is gated on FreeVirtualMemory > 10,240 MB (as A1); **free virtual memory read 11.1 GB at build time with the RJ5 re-roll running, so the second worker may wait** (`--min-free-virt-mb` is the knob, not mine to lower). F: 22.4 GB free at build time; the chain waits (not fails) below 20 GB; its own output is about 0.4 GB.
Resume: re-issue the same launch; finished stages are skipped, `derive` resumes per token. A `.FAIL` marker must be removed by the MM after the cause is fixed.

## Controls and where each is implemented
Build side (`n5_nt_build.py`): C-PIN (guards), C-REACH (guards + reach_real), C-DERIV, C-LABELEGO, C-DACMAP, C-UNCH (on files, count of L1-run tokens == 25), C-DET (stronger than the card: EVERY builder DAC-0 token run twice, plus N5-A == N5-R on the 25), C-CERT-B, plus two additions: **C-SET** (builder DAC-0 set == banked N4-D build's 91 + 25; L1-run set == the card's residual split) and **C-SEAMFN** (`ND.to_navsim` reproduces the banked N4-D seam bit for bit on all 12,146 tokens, so the changed rows are made by the function that made the banked ones). Read side (`n5_nt_analyze.py`): C-COV, C-BUILD, C-UNCH, C-PIN (+ families TIP), C-CERT, C-ID-C (full set + the 200 sentinels), C-EST, C-REG. Mutation arms and their tests: `test_n5_chain.py` (header lists them). Test count: see `test_run_n5_chain.log` (53 chain tests + 8 L1 tests = 61 passed).

## At the freeze (MM)
1. Copy the frozen card to `F:/tanitad_scratch/n5nt/run/n5/PREREG_TF_N5.md`; its sha256 goes to `--card-sha256`.
2. Append the ledger entry `TF-N5:<sha>:E3` n = 47 to `inputs/ledger_K.json` (re-read the max first).
3. If ANY pinned file changed since this hand-off, re-run `python n5/n5_make_pins.py --run-dir R --tip <sha>` (N4_RUN=R, navsim venv) and use the new pins sha; the template pins sha is in `launch.cmd.template`.
4. `n5_nt_chain.py ... --check-only` (guards only, 3 s), then launch with `launch.cmd.template` through WMI.
Never `--mutate`: the chain cannot pass it and the build refuses it without `--smoke-allow-mutation`.

## Smoke (navtrain and synthetic only; no navtest N5 arm was scored)
`smoke/`: 7 real navtrain DAC-0 instances (DEV30) through the REAL build CLI (`derive --route navtrain`, `reach --real`, `merge`, `unch`): N5-A certified 5/5 (L1 set of 5), N5-X 0/5, N5-R 7/7, every build control PASS; the same with `--mutate cert_off --smoke-allow-mutation`: C-CERT-B and C-UNCH RED. Families end to end on 300 real banked A1 rows (N2R vs N4D, tip `families_run.py` --per-window -> `fam_ci.py` -> `famread`): ran, JSON carries 300 tokens / 300 logs; the A1-tree `families_run.py` on the same rows writes no tokens and `fam_ci` REFUSES it ("JSON carries no tokens/logs"), and the chain's check catches it right after the first families_run job.
