# PRIVGEOM-CLEAN: Operational Amendment 1 (OPS1), the extraction-completion bug in the read controller

**STATUS: DRAFT** (prepared 2026-10-20 for the Master Mind, who seals it with `ops1/seal_amendment_ops1.py`; a DRAFT operational amendment re-pins nothing: `pgc_seal.check_seal` refuses it)
Sealed at: NOT SEALED
Governs: the stage `code/pgc_read_chain.py` (the extraction phase of the read) and the seal check itself. Amends: `PREREG_PRIVGEOM_CLEAN.md`, sha256 `148ebf0f2162b43ba7a1c29243fceac5900fe6b88ace3cae04fc27bdf29c6cf9` (UNCHANGED; that file, its `.sha256` and `SEALED_AT.txt` are not rewritten).

## 1. This amendment is OPERATIONAL ONLY

No rule, bar, estimator, gate, control, constant, seed, stage order or analysis code changes. The sealed prereg's call rule, C-NULL, C-POWER, A-ADMIT, the primary arm (LGc) and every number in it stand as sealed. Of the **@@N_UNCHANGED@@ files besides the three below that the seal pins** (`SEALED_AT.txt` `code_sha256`), every one is byte-identical to its sealed pin: this is asserted by the generator of this file and re-checked by the seal check (a differing byte in any of them is refused, `ops1/test_ops1` part C).

## 2. What happened (MEASURED, 2026-10-20; artifacts under `run_clean/`)

- The sealed read stopped at 01:09:44 with `ZZPGC_STOP extraction worker crashed 4 times` (`run_clean/read.log`, `run_clean/CHAIN_STOP.txt`). Preflight had passed (`stages.log`: SEAL HOLDS, branch table clean, select OK: 76 logs).
- **The extraction itself COMPLETED: `run_clean/scalars/` holds 76 / 76 `.npz`, no `.lock`, no `.failed`; `worker1.log` ends `WORKER_DONE pid 68968 logs_done 42`, `worker2.log` ends `WORKER_DONE pid 65328 logs_done 34` (42 + 34 = 76), every log `fail 0`.**
- **Cause (verified, not assumed).** On Windows the venv `python.exe` that `subprocess.Popen` starts is a launcher whose child interpreter has a DIFFERENT pid. `pg_extract.py work` writes `WORKER_DONE_<its own os.getpid()>.txt`; the controller looked for `WORKER_DONE_<Popen.pid>.txt` (`pgc_read_chain.py` old line 163), so every clean exit counted as a crash and was respawned. The respawns found nothing to do, exited rc 0 and were counted as crashes too; after `MAX_RESPAWN` = 3 the controller stopped. Pids:

| worker (read.log) | Popen pid (launcher) | interpreter pid = name of its done record | logs_done |
|---|---|---|---|
| worker1, start 21:31:40 | 57416 | 68968 | 42 |
| worker2, start 21:33:10 | 65900 | 65328 | 34 |
| worker2 respawn 1, 01:07:44 | 67540 | 50000 | 0 |
| worker2 respawn 2, 01:08:44 | 51064 | 55176 | 0 |
| worker1 respawn, 01:09:44 | 67380 | 44192 | 0 |

  No Popen pid appears among the five done-record names. A direct probe on this machine (`Popen` of the same venv `python.exe`) gave Popen pid 10740 and interpreter pid 45952 (a child of the launcher).
- **A second defect of the same root cause, found on the way.** The controller freed a dead worker's `.lock` only if the lock's recorded pid equalled `Popen.pid`; the lock records the interpreter's pid, so on a REAL crash the lock would never have been freed and the replacement worker could never take that log.

## 3. The change (three pinned files re-pinned; nothing else)

| file | sealed sha256 (in force today) | sha256 after this amendment |
|---|---|---|
@@PINROWS@@

- `pgc_read_chain.py` (diff `ops1/pgc_read_chain.py.diff`, sha256 `@@D_CHAIN@@`): (a) a worker that exits rc 0 is FINISHED iff a `WORKER_DONE_*.txt` written at or after that worker's own start exists under ANY pid and is not already claimed by another worker (one record finishes one worker), OR all 76 logs are extracted with no lock held and no `.failed`; a nonzero rc, or rc 0 with logs missing and no record, is still a crash (respawn, then stop after 3); (b) a dead worker's lock is freed iff its recorded owner pid is no longer a live `pg_extract.py` process; (c) a start with 76 / 76 already extracted spawns NO worker and goes straight to the join. The join, gate, fit and analyze stages, their order, the `stage()` calls and every constant are untouched.
- `pgc_seal.py` (diff `ops1/pgc_seal.py.diff`, sha256 `@@D_SEAL@@`): **this is the one change that needs the Master Mind's judgment.** Before this amendment `check_seal` had no way to accept a changed pinned file: amendments were hash-checked only, so ANY edit to a pinned file, `pgc_read_chain.py` included, was refused by every stage including preflight. There is no seal-respecting way to ship the fix without teaching the check one thing, narrowly: a SEALED operational amendment (`PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS<k>.md` + `.sha256`) may re-pin a file listed in `REPINNABLE` (the read controller, the seal tool, `PINS.json`, `pgc_pins.py`, `build_package.py`, the three launch scripts), **never** analysis code (`pg_*`, `pgc_lib`, `pgc_admit`, `pgc_run`, `pgc_coverage`, `pgc_design`, ...), a frozen suite, `test_pgc_seal.py` or `ref/`; only from the pin in force (a chain starting at the SEALED pin); the plain `_AMENDMENT<k>` series cannot re-pin; a DRAFT-status operational amendment re-pins nothing. The sealed prereg, its section-13 table and `SEALED_AT.txt` stay byte-identical: the pin in force is the sealed pin carried forward through the sealed amendments. The widening of what an amendment may do is real; its authority rests on the Master Mind sealing this file.
- `PINS.json`: regenerated by `pgc_pins.py` (unchanged); two lines differ (`ops1/PINS.json.diff`, sha256 `@@D_PINS@@`); `pgc_pins.py --check` reads OK.

## 4. The 76 extracted outputs are KEPT, and the extractor is unchanged

- `code/pg_extract.py` sha256 **`@@PEX@@`**, unchanged and still the sealed pin. No extraction is redone.
- Manifests (sha256 of every kept file, hashed as bytes; nothing was opened as data): `ops1/scalars_npz_sha256.txt` (76 `.npz`; the manifest file's own sha256 `@@MAN_NPZ@@`), `ops1/scalars_timing_sha256.txt` (76 `.timing.json`; `@@MAN_TIM@@`), `ops1/worker_done_sha256.txt` (the 5 done records; `@@MAN_WD@@`).
- The five `WORKER_DONE_*.txt` stay in `run_clean/scalars/`; they pre-date the relaunch, so the fixed controller ignores them (a record counts only if written at or after the worker's own start).

## 5. Nothing after the extraction has run, and no label has been read (state at the time of writing)

`run_clean/` holds `CHAIN_PID.txt`, `CHAIN_STOP.txt`, `read.log`, `scalars/`, `seal_proofs/`, `stages.log`, `subset_clean.json`, `worker1.log`, `worker2.log` and nothing else: no `join/`, `gate_report.json`, `gate_token.json`, `join_labels_sub.npy`, `fit*.log`, `fits/` or `results/`; `stages.log` records only the preflight. No join, gate, fit, C-POWER, C-NULL or call has run; no label has been opened by any analysis stage; no scalar `.npz` was parsed while preparing this amendment (they were hashed as raw bytes only). The preparation was CPU only (no GPU, no pod, no lock).

## 6. Tests (CPU, throw-away directories, planted processes; no real extraction, no label)

`code/test_ops1.py` (unpinned evidence, sha256 `@@TESTOPS@@`; fixture `code/fake_pg_extract_ops1.py`, sha256 `@@FAKE@@`), **34 tests, all OK**, plus the pinned `test_pgc_seal.py`, **24 tests, still all OK against the changed `pgc_seal.py`**:
- **RED on the current bytes** (`ops1/orig/pgc_read_chain.py`, sha256 `b17b96ad...`): a launcher-pid mismatch with clean completion is counted as a crash and the run stops with `extraction worker crashed 4 times`, with the same `respawn 1/3 ... 4/3` signature as the real stop; with 76-analog logs already extracted the old controller still starts 2 workers and stops the same way.
- **GREEN on the fix**: the same scenario finishes both workers and reaches `ZZPGC_EXTRACT_DONE`; a relaunch with everything extracted (and 5 older done records present) spawns 0 worker processes and reaches `ZZPGC_EXTRACT_DONE`; all logs extracted with no record also counts.
- **RED arms on the fix**: rc 0 with logs missing and no done record is still a crash (stops `crashed 4 times`); a nonzero rc is a crash even with a done record present, and its stale lock is freed so the replacement completes; one done record finishes ONE worker; a record older than the worker's start finishes nothing.
- **MUTATION arms** (one guard disabled in a temp copy; the matching RED scenario is then passed wrongly): the rc guard, the logs-extracted guard, the one-record-one-worker claim, the start-time guard, the stale-lock freeing, the skip-when-complete; and in the seal: the STATUS guard, the REPINNABLE guard, the pin-in-force chain guard.
- **Seal RED arms**: an edited operational file without an amendment; a DRAFT amendment; an edited-after-hash amendment; a wrong `.sha256`; a `to` that is not the file on disk; a `from` that is not the pin in force; an analysis file in the block; the plain amendment series; a malformed block. **Mirror of the real package**: the new bytes are refused without a sealed amendment and accepted with it, the sealed prereg and `SEALED_AT.txt` hash identically to the real ones, and a one-byte edit of `pgc_lib.py` is still refused with the amendment sealed.

## 7. Resume behaviour (verified read-only on the real files; the chain was NOT launched)

A relaunch of the fixed controller: refuses on any `PGC_*` variable and on a live chain (none: `CHAIN_PID.txt` pid 60272 is dead; no `pg_extract.py` process); runs preflight (the seal check incl. this amendment, the branch table, `select`, which rewrites `subset_clean.json` byte-identically: checked in memory, True); sees `counts() = (76, [], [])`; logs `extraction already complete (76/76): no worker spawned`; logs `ZZPGC_EXTRACT_DONE 76/76`; proceeds to join, gate (Phase A label-free, then Phase B), fits, analyze, exactly as sealed. `CHAIN_STOP.txt` is deleted by the controller at start (as written in the sealed version): copy it aside first (the launch note in `HANDOFF_OPS1.md` does).

## 8. Disclosures (not fixed here; none is on this stop's path)

1. If an extraction ever writes a `.failed` marker, the controller's `p.terminate()` signals the launcher only, and its child interpreter may keep running. Not changed (not part of this stop).
2. `run_clean/seal_proofs/` was produced by the OLD checker and was not re-run; the new checker's refusals are proven by part B and C of `test_ops1.py`.
3. The sealed prereg's section-13 table still lists the OLD shas of `pgc_read_chain.py`, `pgc_seal.py` and `PINS.json`, by design (it is sealed text); this amendment is the record of the change. Anyone reading the table must read this amendment with it.
4. The staged repo copy of the package must carry these bytes, this amendment and its `.sha256`, and `ops1/`; `build_package.py` is unchanged and was not re-run.

## 9. How the Master Mind seals this (one command; nothing is hand-written)

1. Review `ops1/*.diff` (about 235 lines in all).
2. `python -B ops1/seal_amendment_ops1.py` (CPU, label-free): refuses unless this file reads DRAFT, no `.sha256` exists and every `to` below equals the file on disk; flips the status line to SEALED, writes `PREREG_PRIVGEOM_CLEAN_AMENDMENT_OPS1.sha256`, runs `pgc_seal.check_seal(inputs=True)`, and rolls everything back if that fails.
3. Then the launch command in `HANDOFF_OPS1.md`.

<!-- OPS_REPIN_BEGIN -->
```json
@@REPIN_BLOCK@@
```
<!-- OPS_REPIN_END -->
