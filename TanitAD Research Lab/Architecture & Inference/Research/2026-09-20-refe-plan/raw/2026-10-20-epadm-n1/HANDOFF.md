# HANDOFF EPADM-N1 (2026-10-20 ~02:42). The simulation is RUNNING detached; the report, tables, RESULT.md and staging are chained behind it. No model turn is needed to finish them.

**State.** `SIM_DESIGN_N1.md` hashed 02:37:09 (sha256 `f45ef24c...`, `SIM_DESIGN_N1.sha256` beside it) BEFORE the first run. Code in `code/`: `epadm_n1.py` (the statistic, `N1Undefined`, `n1_verdict`), `epadm_sim_n1.py` (simulator), `epadm_sim_s0.py` (byte copy of S0BANK, unedited), `test_n1.py` (39 planted-data tests, pass), `mutation_check_n1.py` (32 arms, all CAUGHT before the run; re-run in the finish step). Base X split built and PC9 passed (`work/sim/basex_n1.json`). Tip pinned `c3a5c329...` (`TIP_SHA_N1.txt`; frozen gate, R1, C1, adapter, fixture are byte-equal to the copies read).

**Job.** Two chains started 02:38:55 via WMI (PIDs 32332 and 62124): `work/chain1.sh` (C-GS117-A, X-GS41-A, C-GS41-P) and `work/chain2.sh` (X-GS117-A, C-GS41-A, C-GS117-P), each job followed by its PC5 repro. CPU estimate ~15 min total, ~8-10 min wall on 2 processes. When chain2 ends it waits for `work/logs/chain1.done`, then runs `work/finish.sh` (PC8, PC3, `report`, `make_tables_n1.py` -> `work/tables.md`, the two pytest suites, the mutation harness, `make_result_n1.py` -> `RESULT.md`, `stage.py` -> `out/` + `out/paths.json`).

**Check.** DONE MARKER: `work/logs/finish.done` (and `ZZFINISH_DONE` in `work/logs/chain2.log`). Progress: `tail -n 5 work/logs/chain1.log work/logs/chain2.log` (one line per finished condition).

**Exact next step (a fresh agent, Haiku/Sonnet, CPU only, ~10k tokens):**
1. If `finish.done` exists: read `RESULT.md` (generated; its "Plain statement" and tables are the result) and `out/paths.json`; check `controls_ok` in `work/sim/sim_results.json`; hand back at most 250 words: the table of A1-r1 FR, A2-N1 FR and N1 DAC power x1.25 with Wilson CIs per base and geometry against the bars, which cells meet every bar, and the manifest. Quote the DQ3 branch line of RESULT.md verbatim; add no interpretation beyond it.
2. If a job died: find it in the logs, delete only its `work/sim/replicates_<job>.npz`/`run_<job>.json`, and re-run it with `bash -c '. work/args.sh; runjob <job>'` (same offset 0); then run `bash work/finish.sh`. At most 2 python processes at once.
3. If `controls_ok` is False the run is VOID: report which control failed; do not read a bar. Repair by a new revision file (`SIM_DESIGN_N1_R2.md`, hashed before any re-run), never by editing the hashed design.

**Do not:** edit `SIM_DESIGN_N1.md`, change a margin / bar / definition / generator, re-run with other salts, take the GPU lock, ssh a pod, commit, push, `git add`, or edit any register. Delete only `work/` scratch of this package. C: has ~0.1 GB free (D: 0): keep every file small.

**Disclosure for RESULT.md:** a 20-replicate plumbing smoke (run/repro/report) was executed after the hash and its scratch deleted; its rates were not used. Seen in the smoke only (confirm in T3's R-NAN counts): on Base C the NC head had no observed unsafe mass at the changed picks in any of the 60 replicate-seed bases, so NC cannot refuse there.
