# DAG-ORC (re-based on DAG-OACC) dev-box T1-kin chain -- NOTES

Status: BUILT AND REHEARSED, NOT LAUNCHED. Nothing committed or pushed (stage-never-push); everything lives only on E:
under `E:/Projects/TanitAD-artifacts/dagorc_t1/`. CPU only, no model turns, one torch process at a time, OMP/MKL 8.

## What it does (in order)
1. Waits, by script polling (60 s), on ONE variable: the box-2 chain log `DORT1_B2_OUT`
   (default `C:/Users/Admin/box2_ship/logs/chain_v8f_dagoaccorc_r2.out`; the r1 log basename is refused in real mode,
   `dagorct1_chain.sh:54`). A run is read only when its section (from its LAST `<run> pulled to <path> (checkpoint sha256 <64hex>)`
   line) holds a PASSing `run record` line AND a PASSing `K-REACH` line (`dagorct1_lib.py parse_boxlog/run_state`).
   K-REACH FAIL, record not PASS, `ZZDAGOACCORC-CONTROL-FAIL`, a later box-2 STOP, a pull path other than `$ART/<run>`, or
   a path ending `_r1_kreach_stop` => FAIL => chain STOP, run never rolled out or read.
2. Pull check by content (`pullcheck`): SHA256SUMS.txt aged >= 3 s, 8 names, every file re-hashed, summary done,
   checkpoint sha256 equal in THREE places (the log's pull line, SHA256SUMS.txt, a re-hash of the file), run identity
   (seed A s / REP 10,11; steps 3000; dagger-p 0.1; w-acc 49.6969; bank sha; init sha; A: ego-perturb-p 0.3 + composition flag
   + seed; REP: p 0, no flag; token PASS naming the run, 40-hex commit). Refuses a checkpoint equal to DAG-OACC's or M2b's,
   a dir ending `_r1_kreach_stop`, and a token binding r1 commit `e7ea20d1...` (+ env `DORT1_REFUSE_COMMITS`).
3. Per run, sequentially: T1-kin causal (decisive), T1-kin logged (K4 only), fresh-age T0 eval (K4 comparator). 12 steps,
   about 50-60 min total (DAG-OACC / DAG2 chains' walls: ~5 min, ~5 min, ~1.5 min). Checkpoint re-hashed before and after
   every step; done records carry ckpt sha, tool digest, command sha256.
4. After the 4 runs: box-2 DONE marker (must follow the last pull line), every pull re-verified, manifest written
   (`runs.json`, A / REP / CTRL slots; CTRL = banked DAG-OACC), then the PRE-REGISTERED reader
   `dag_orc_read.py` (sha256 2e5f9f17...) in real-run mode -> `read/DAGORC_READ.json`. Markers (anchored, line start):
   `ZZDAGORCT1-DONE <n>/12 steps, reader rc <rc>` / `ZZDAGORCT1-STOP <why>`.
5. Tier/estimator stamp for anything quoted from it: T1-kin = self-action OPEN loop (never "closed loop"); T0 = teacher-forced
   diagnostic; paired episode-cluster bootstrap B=2000. A separated CI is necessary, not sufficient (replicate arm REP is the floor).

## Launch (only after the r2 chain is running; it also tolerates an absent log and waits)
```
powershell -NoProfile -Command "([wmiclass]'Win32_Process').Create('cmd /c E:\Projects\TanitAD-artifacts\dagorc_t1\dagorct1_chain.cmd') | Select-Object ReturnValue, ProcessId"
bash E:/Projects/TanitAD-artifacts/dagorc_t1/wake_dagorct1.sh            # waker: exit 0 DONE | 4 STOP | 5 timeout | 6 chain dead without marker ; --status = one-shot
```
Optional env goes into the `.cmd` (a WMI process does not inherit the shell): `DORT1_OUT`, `DORT1_ART`, `DORT1_EXPECT_COMMIT`
(40-hex fixed trainer commit every token must bind), `DORT1_COMPOSE_PKG` / `DORT1_EVIDENCE_COMMIT`, `DORT1_B2_OUT`.
Resume = launch again (finished steps are skipped by their done records; an incomplete step is MOVED to `_incomplete_*`).

## r2 changes requested mid-task, and what answers each
| requested | where |
|---|---|
| chain log path ONE variable, default r2 log | `dagorct1_chain.sh:45` (`B2OUT`); waker `wake_dagorct1.sh:8`; r1 basename refused `:54` |
| every wait keyed on pull + PASS record + K-REACH PASS | `wait_run` `:324`, `lib parse_boxlog/run_state` |
| refuse dirs ending `_r1_kreach_stop` | `lib REFUSED_SUFFIX:51` in `run_state`, `pullcheck`, `verify_manifest`; dry run lists such dirs as ignored |
| checkpoint sha vs the r2 log's pull line (not only SHA256SUMS.txt) | `pullcheck --expect-sha` (3-way), `chain wait_run` |
| pinned trainer commit for the harness = variable | `DORT1_EXPECT_COMMIT`; `commit_gate` `:305` (one launch commit across the 4 tokens, never r1's) |
| which code does the rollout import; does the trainer change touch it | answer below; gate runs on the r2 commit after the first pull and STOPs if touched |

### Does the trainer change reach the T1-kin rollout?  MEASURED (runtime trace, smoke, 2026-10-28)
The three programs (`v8f_t1_eval.py`, `v8f_mini_eval.py`, `taniteval/tools/t1_eval.py`, loaded by path) import **45 `.py` files from the pinned
tool tree** (tree digest `6a4b47c8...` = the tree that rolled DAG-OACC out; the chain never replaces it; trace in
`smoke/trace_*.json`). Compared with the **r1 build `e7ea20d1`** (`smoke/ROLLOUTCODE_RUNTIME_vs_r1_e7ea20d1.out`): 44 identical, 1 differs --
`stack/tanitad/data/v8f_mini_data.py`, changed top-level name `EgoPerturb` only, which the harness never references (it reaches 44 other names
of that file; TOUCHED = []). `stack/scripts/v8f_mini_train.py` is not imported by the harness at all. `v8f_mini.py` (model) identical.
=> for r1 the harness is unchanged. The r2 fix is expected to edit the same two files; the chain re-runs this gate on the r2 launch commit
(`lib rolloutcode --gate v8f_mini.py v8f_mini_data.py`, static closure, over-approximate) and STOPs if a harness-reachable name of either differs.
Note the static closure also lists `refs/refc.py` and `refs/refc_v3.py` as differing: that is repo drift between the tool tree and the commit,
not loaded at runtime (absent from the 45), and not gated.

## Verbatim from DAG-OACC / DAG2 chains vs different
Copied in structure (re-implemented where noted): the `.cmd`/WMI shape (`speedread_run/run_speedread.cmd`); guards 0a-0c
(`dagoacc_t1_chain.sh:98-162` -> `dagorct1_chain.sh:93-190`); `pin_check` (`:184` -> `:184`); `cmin_wait`/`emin_wait`
(`:205/:209` -> `:101/:102`); the rollout function and done records (`:245` -> `t1run :246`); `b2_done_wait`/`verify_pulls`
(`:281/:294` -> `:357/:368`); the K4 logged lane and fresh-T0 command (`dag2_t1_chain.sh:210-290` -> `t1cmd :204`, `t0cmd :210`, `t0run :274`).
Rollout argv: NOT retyped. `argv/*.tmpl` were derived programmatically and PROVEN on startup (dry run and real): causal template rendered with
DAG-OACC s0's parameters == `v8f-dagoacc-t1/argv/DAGOACC_s0_m10_causal_heads.argv` (32 tokens, byte for byte); logged and fresh-T0 templates
== the commands in `dag2t1/logs/chain.out` for DAG2 s0 (`tmpl_proof :217`). The only argv difference between lanes is `--limit`.
Different from DAG-OACC's chain: 4 runs not 2 (A s0, REP s0, A s1, REP s1, in the box-2 chain's order); three steps per run (adds logged and
fresh T0); the wait is keyed on a log that carries run-record and K-REACH lines; identity checks per arm; rollout-code gate; K-BUILD record;
one-instance lock + `chain.out` rotation (`:79-86`) so the waker never sees a stale marker; floor-aware E: gate (below); reader invoked in
real mode with slots A, CTRL, REP (B not queued); box-2 chain files pinned on first sight in `<OUT>/PINS_BOX2.txt` (`b2_pin :170`) because
the r2 chain does not exist yet.

## K-BUILD (the reader refuses the run without it)
Built from the four launch-gate tokens' commit and git objects in `C:/Users/Admin/tanitad-push`. sha256 and hunks: MEASURED. Hunk-to-flag
attribution: ATTESTED at file level by the compose package's flag-off bit-identity evidence, bound to the launch bytes (the package's tested-file
sha256 list must equal the launch blobs). r1 worked example (`kbuild` on the r1 build): 30 hunks, all behind `--ego-perturb-with-dagger`, default
off, bit-identity PASS, accepted by the reader's own `k_build`. **DEPENDENCY for r2:** the fix must ship its own re-run evidence (RESULTS.md with
sha256 of the r2 files, >= 26 passed, M13 killed, all >= 19 mutants killed, `logs/compose_run2.log`), at `DORT1_COMPOSE_PKG` (default = the r1 path)
in a commit the mirror holds (`DORT1_EVIDENCE_COMMIT` if later than the launch commit). Otherwise K-BUILD is written `UNVERIFIED: ...` and the
reader refuses A. The chain prints the early K-BUILD status right after the first pull, so this is visible hours before the end.

## Evidence this was tested (and what was not)
* `dagorct1_selftest.py`: 103/103 checks as expected (`selftest/SELFTEST.out`): constants equal the reader's; the real r1 log reads A s0 FAIL; 17 synthetic
  log arms (pass, record FAIL, K-REACH FAIL, awaiting, wrong path, `_r1_kreach_stop` path, arm/seed confusion, re-pull, CRLF, stale DONE...); 25 pull-dir
  mutations (each identity clause, refused commit, 3-way sha, incomplete sums, suffix); K-BUILD positive + 8 negatives against the reader's own `k_build`;
  manifest vs the reader's `slot_needs`; `verify_manifest` positive and 8 negatives (incl. the shape the real smoke wrote, `n_windows` 8).
* REAL smoke (`smoke/SMOKE_RUN.out`): DAG-OACC s0's checkpoint, same templates, `--max-windows 8` for causal and logged (n_windows 8), fresh T0 in full:
  rc 0 x3, `ZZDAGORCT1-SMOKE-OK`. Not a read.
* Real-mode rehearsals in throwaway worlds (`selftest/REHEARSAL_*.out`): (1) the real r1 K-REACH FAIL log => STOP before any checkpoint is touched;
  (2) synthetic PASS log + fake checkpoint => wait, 3-way pullcheck with A identity, commit gate, rollout-code gate, K-BUILD early status all ran,
  then STOP at the first rollout ("pickle data was truncated"): a failed rollout is never read.
* `--dry-run` (`DRY_RUN.out`): 12 commands printed, r1 A s0 pull shown as WRONG (r1 commit refused), reader pre-flight PASS.
* NOT verified: the positive `manifest --write` path on real A / REP outputs (they do not exist; verified only on a synthetic world); the r2 box-2 log
  line shapes beyond what the r1 log showed (the parser is keyed on the r1 chain's exact printer; `b2_pin` + a 15-min AWAITING timeout STOP, not guess,
  if they change); the smoke was run before the final K-BUILD threshold edit (lib only; rollout path untouched; pins rewritten, dry run re-run after).

## Risks to know before launch
* **E: capacity.** E: had ~2.5 GB free. New data: ~340 MB per run (2 x 141 MB dumps + ticks + fresh T0) = ~1.4 GB, plus 4 pulls of ~100 MB, and box 2's own
  1 GiB pull floor. The chain's floor-aware gate (200 MB + 1024 MB + 100 MB x pending pulls) WAITS (up to 2 h, never deletes) rather than starve a pull; it can
  therefore sit waiting at the end. Safer: set `DORT1_OUT=F:/TanitAD-scratch/dagorc_t1_out` in the `.cmd` (F: ~200 GB free); then copy `runs.json`, `read/`
  and `done_*.txt` (a few MB) to E: after the reader finishes. The dumps are scratch.
* C: ~7 GB free (gate >= 2 GB); the chain writes nothing on C:.
* Another agent's pytest (2 GB, CPU torch) was running during the smoke; it is not a rollout, but check for any torch rollout before launch.
* A box-2 pull that lands while a rollout is running is handled (pull dir is only read after the log says PASS and the sums are >= 3 s old).
