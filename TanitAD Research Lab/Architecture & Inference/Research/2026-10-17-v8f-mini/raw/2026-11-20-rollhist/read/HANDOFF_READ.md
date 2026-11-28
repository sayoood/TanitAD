# HANDOFF_READ (ROLLHIST read package; STAGED, chain NOT started)
Card PREREG_V8F_ROLLHIST.md (781eea87) + Amendment 1 (87928487); code of record 43319227. Folder `F:/TanitAD-scratch/rollhist/read/`. Nothing staged, committed or pushed. No IP, port or pod id written.

## State
* `tree/` = `git archive 43319227 stack taniteval`, `tree_blobcheck.txt` 3,407 of 3,407 blobs, 0 bad, 0 extra (hb wrapper sha256 74fd8445 is in the commit).
* **c-EXEC PASS** (`C_EXEC.json`): UE_s0.A (sha 2fd4b747, pulled from HF) x R0C10XHb20, study wrapper rzgfh vs the tier of record, 141 files, 1,269 arrays, **0 differing**; bound acted on 50,946 of 287,880 frames. CPU wall about 6.2 min per rollout.
* **FINDING for the MM:** the card's `--inner v8f_t1_eval_reader_rule_rzg` cannot compose with `v8f_t1_eval_hb.py` as written (hb refuses: rzg imports the tool inside main(); and rzg's load_pinned would drop hb's patch). The package uses the adapter `code/integ10k/v8f_t1_eval_reader_rule_rzgi.py` (rzg unedited, tool imported once). c-EXEC passes with it.
* Stage-A references: `REFS_FOUND.json`. R0Hb20 and R0KHb20 dumps exist (ADD67, rzgfh). R0C10Hb20 and R0C10XHb20: result JSON only (ADD68, controls pass), no dump on this box. R0G50Hb20 and R0G0Hb20 on UE.A: none (ADD63 ran the old executor).
* Reader `rh_read.py`: selftest PASS (K-SELF, K-PLANT, K-FLOOR, K-SEEDS, K-HARMS, mutation other-seed arm). `DRY_CHECK.json` PASS. Real-format dry run (`DRY_RESULTS.json`): see its K-SELF and K-PLANT lines.

## Launch (MM says go AND the four pod checkpoints exist)
```
powershell -c "$s=([wmiclass]'Win32_ProcessStartup').CreateInstance(); $s.CreateFlags=0x08000200; $r=([wmiclass]'Win32_Process').Create('cmd.exe /c F:\TanitAD-scratch\rollhist\read\rh_read_chain.cmd < NUL','F:\TanitAD-scratch\rollhist\read',$s); 'pid='+$r.ProcessId+' rv='+$r.ReturnValue"
```
One-seed provisional read: set `RH_SEEDS=0` (or 1) in the .cmd first. Resumable (`out/steps/*.ok`); log `out/rh_read_chain.log`, last line `ZZ ROLLHIST-READ DONE|FAIL ZZ`; result `out/read/RESULT_ROLLHIST.json`.
Steps: gates (C_EXEC pass, blobcheck, reader selftest) -> pull 4 ckpts by the pod's own sha256sum -> 24 rollouts -> K-REPLAY -> G-T0 (4 T0 evals) -> headway + family readers -> STRATEGIC -> `rh_read.py`.

## Cost (ESTIMATED from MEASURED walls: c-EXEC rollouts 373 s and 397 s; T0 eval 84 s; headway 25 s, families 14 s, STRATEGIC 80 s on the stand-in dry run)
24 rollouts x ~6.6 min = ~2.6 h; 4 T0 evals ~6 min; K-REPLAY ~2 min; readers ~2 min; pulls a few min. **About 2.8 to 3.0 h CPU on the dev box** (one seed pair about 1.4 h). Optional references: 4 rollouts (UE.A x R0G50Hb20, R0G0Hb20) = ~0.45 h, only for one-seed P2 D1/D2 floors. Do not run beside another CPU chain.

## Controls, with values (all MEASURED, ours)
c-EXEC 0 of 1,269 arrays differ (PASS). K-REPLAY 0 of 576 differ (stand-in UE_s0.A; the chain re-runs it on RH_s0). K-SELF on real dumps: 273 pair cells, max abs D = 0.0, outcome NO-EFFECT. K-PLANT on real dumps: a 3 deg rotation of RH's `cl` tracks about their start point leaves speed, yaw-rate and curvature cells at <= 2e-15 and moves heading (+1.49 deg), cross (+1.24 m) and ade_tac (+1.05 m) in P1 and P4. G-T0 dry: UE_s0.A T0 ADE6 mean 1.9671 (equals FACTS E3), guard delta 0.0. Mutation (other seed's arm): the provenance check goes RED.

## Caveats (said, not hidden)
* The card's table does not name every combination (e.g. exactly one family BETTER with P2 and P3 passing; two families BETTER with G-T0 not holding at one seed only). The reader prints `OUTSIDE-TABLE` for those, never an invented name.
* P2 pass = RH not worse on every family AND the B0 attribution control (ACC pair WORSE on at least two of three families). One-seed floors come from ADD68's pod read (INHERITED).
* BLOCKED is raised by provenance (a rollout whose recorded checkpoint sha256 is not its arm's) or an optional `--blocked` JSON; the pod's coverage record is not read by this package.
* `decide()` reads SEED-DISAGREE as: opposite-sign seed differences each above F_c, or a decisive cell whose pooled interval excludes 0 with |D| <= F_c. That reading of "straddle the floor" is the MM's to confirm.
