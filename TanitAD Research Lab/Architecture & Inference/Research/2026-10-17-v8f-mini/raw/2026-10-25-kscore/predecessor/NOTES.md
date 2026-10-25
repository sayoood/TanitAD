# KSCORE predecessor wait: fail-closed classifier ported from FLAT's chain

**Fault.** `chain_v8f_kscore.sh` (tip 9a6551d5) stopped on ANY `ZZV8FMINI-STOP` in `chain_v8f_flat.out`. A FLAT run that ends on its own verdict would leave box 2 idle for hours.
**Fix.** One function `after_decision <log> [<guard log>]` -> PROCEED (rc 0) / HOLD (rc 1) / WAIT (rc 2), used by the wait loop and by `--classify-predecessor`. It sits before every statement that
touches box state (no side effect). Default guard log: `v8fflat_guard.out` (FLAT's own). DONE anywhere wins; the LAST stop line decides; box-level evidence (`ZZGUARD-STOP`, `ZZCRASH`,
`ZZUNREACHABLE`, `ZZTIMEOUT`, all printed at column 0 by v8f_guard.sh / waitjob.sh, verified in source) in the chain log OR the guard log HOLDs whatever the message says; any unknown text HOLDs.

## FLAT's stop texts (chain_v8f_flat.sh: 43 `stop "<msg>"` calls = 42 distinct texts, + 3 direct `echo ZZV8FMINI-STOP` lines, + 1 that only `--dry-run` prints)
| class | decision | texts |
|---|---|---|
| RUN | PROCEED | `<job> rc=<n>`, n >= 1, job = `v8fflat<MMDDHHMM>_(gateflat\|t_M2b_s[01]_flat\|e_M2b_s[01]_flat)` (FLAT's gate / train / eval jobs) |
| VERDICT | PROCEED | `flat read` (reader refused, rc 2) |
| GATE | PROCEED | `gate FLAT assertions` (nothing was trained) |
| BOX: transport (11) | HOLD | `<job> wait=<n> (2 = CRASH/reboot ...)`, `box 2 unreachable`, `ship <job>`, `launch <job>`, `pull <path>`, `ship the code archive`, `ship refresh job`, `launch refresh`, 2x `cannot hash box 2's ...`, `no fresh telemetry ...`; and `<refresh job> rc=<n>` |
| BOX: state (5) | HOLD | `guard stop`, `box 2 <file> != tip <sha>`, `box 2 v9 <file> md5 != ...`, `<phase>: box 2's M2b_s<n> checkpoint sha256 ... != the banked run's ...`, `M2b_s<n>_flat's checkpoint changed during the evals ...` |
| BOX: marker/chain (5, incl. 3 direct) | HOLD | `busy marker present`, `busy marker taken while arming`, `the predecessor's end does not clear the box` (FLAT's own held predecessor), `write the gate token`, `v8f_guard.sh does not know this chain's markers` |
| pre-flight (21) | HOLD | tip / staging not as FLAT expects, box never asked to run FLAT code: `remote tip ... not in the mirror`, `archive`, `extract the reader`, `the tip's queue has no <stage> stage` / `prints no gateflat line`, `the tip has no <file> / strata file / B1 lead block`, 3x `built_levers does not record`, 2x `v8f_mini.py has no`, 3x `the tip's trainer ...`, 4x `stage ...`, `the reader cannot use the M2b side (check-inputs)` |
| dry-run only | HOLD | `no queue at the tip` |
Anything else (another chain's `mixo read`, `gate X assertions`, `gate s0 assertions`, `t_x rc=1`, a 2-seed-out-of-range job, `rc=`, `rc=0`, a negative rc) HOLDs.

## Tests (test_v8f_kscore.py): 122 passed (28 existing + 94 new), 104 s; FLAT's chain absent: 1 skip (the source-drift test), rest pass
79 table rows (every FLAT text rendered, each decision asserted by running the chain's own `--classify-predecessor` on a synthetic log); 5 evidence overrides x 3 proceeding stops x 2 log locations;
DONE / WAIT / empty / CRLF / last-stop-wins; the wait loop run end to end with ssh/scp stubbed (DONE, run, verdict, gate PROCEED and die at `box 2 unreachable` releasing the marker; thermal, timeout, unknown
HOLD and take no marker; WAIT -> DONE proceeds); source-drift test (FLAT's `stop "..."` set == the table, job names, tag format, log names; runs once FLAT lands at `<v8f-mini>/code/box2_ops/`, or via `KSCORE_FLAT_CHAIN`).
**Mutations (all go RED)**: unknown text PROCEEDs -> in-test mutant detected, and applied in place the suite shows 73 failed; evidence ignored -> 8 failed; refresh rc let through -> 5 failed;
RUN_LEVEL_RE widened to any job name -> detected on every wrong-chain row; verdict branch removed -> 2 failed. Rig restored byte-identical after each.

## Decisions for the lead (I did not make them)
1. **Deviation from the MIXO-era predicate:** FLAT's refresh-job `rc=<n>` HOLDs (the MIXO-era regex was job-name-agnostic and let it through). The refresh is a `tar` into box 2 (no model code), so its failure is the box's. One regex alternation to revert.
2. **Pre-flight stops HOLD (21 texts), as in the MIXO-era table.** A FLAT pre-flight failure at night therefore leaves box 2 idle until a human acts. These are local tip/staging checks that say nothing about box health; if you want them to PROCEED, add a PREFLIGHT class (KSCORE re-checks its own tip anyway).
3. Negative rc (native crash) HOLDs; positive rc of any size PROCEEDs.
4. OACC (KSCORE's successor) cannot copy these regexes: KSCORE's run stop reads `<job> rc=<n> (the read's rc 2 = ...)` and its gate stop `gate s<seed>[_perm] assertions`; the new `the predecessor's end does not clear the box` text HOLDs by default.
5. The drift test is skipped until FLAT lands (not yet at the tip); the job names are pinned from the scratch-tree draft of FLAT's chain.
