# HANDOFF_READ_R2 (ROLLHIST read package, R2; Amendment 2 sha256 15004dc9)
Folder `F:/TanitAD-scratch/rollhist/read/`. Nothing staged, committed or pushed. No IP, port or pod id written. Main read chain NOT started; pod untouched.

## Refs chain (running, detached)
`rh_refs_chain.sh/.cmd`, launched by WMI (CreateFlags 0x08000200) 02:36Z-local, wrapper pid 36568. Resumable (`out/steps_refs/*.ok`, `T1_DONE`). Log `out/rh_refs_chain.log`; last line `ZZ ROLLHIST-REFS DONE ZZ` or `ZZ ROLLHIST-REFS FAIL ZZ <why>`.
Order: pull UE_s1.A (HF `Sayood/tanitad-v8f-dscale10k-runs` dataset repo, sha256 f13df8ce026eed31e0d9d1df72de3e7b7977fefa6da3def2b411b7d74dbd4ebc = the HF LFS oid; pulled, checked OK), then UE_s0.A and UE_s1.A x R0G50Hb20, R0G0Hb20, R0C10Hb20, then UE_s1.A x R0C10XHb20. First rollout was at 2,399/2,399 (327 s) at 02:45. Estimate ~46 min total (ESTIMATED), done about 03:25.
Form of record, tree `read/tree` 43319227, CPU 2 threads, one rollout at a time, each gated on commit >= 5.5 GB and F: >= 6 GB.

## Reader R2 (`rh_read.py`; R1 kept as `rh_read_r1.py` sha256 195f96a2727eb51b...)
Items 2, 3, 4, 6 implemented. `SELFTEST_R2.json` ALL_PASS: R2-G1, R2-G2, R2-STRADDLE-FLIPS (both names printed), R2-STRADDLE-KEEPS, R2-P2-ATTRIBUTION, R2-PROVISIONAL(+NAMES), each with a regression arm (rule switched off via REG) that reads differently; old selftests still pass. `DRY_CHECK.json` PASS (40 checks).
Reading choices for the MM to confirm:
* Straddle flip is applied to every named cell in every pair (P1, P2, P4), not P1 only.
* G1 is read literally: no G-T0 condition. G2 ignores P4.
* P2 "at both seeds" is read from the pooled pair families (a cell needs both seeds on one side).
* SEED-DISAGREE from seeds of opposite sign beyond the floor is kept (R1 rule), after HARMS.

## Wiring
`rh_read_chain.sh`: refuses to start unless the refs log ends `ZZ ROLLHIST-REFS DONE ZZ` or `RH_REFS_SKIP=1`; adopts c-EXEC's UE_s0.A x R0C10XHb20 dump as `out/t1/I10UEA_R0C10XHb20_s0`; runs headway + family readers over UE.A (ACC pair uses ADD67's R0KHb20 dumps) into `out/read/refs_hw`, `refs_q3`; `rh_read.py` takes them as `--refs-*` (F_U,c MEASURED) and ADD68's JSON as `--add68-*` (printed beside as `floor_ADD68_inherited`); P2 reference D1/D2/ACC on UE.A is in `P2_reference_UE_A`.
OPEN: one-seed read STRATEGIC family has no stage-A tick-2 floor, so a one-seed read prints NOT-DECIDABLE until `--strat-floor` data exists (hook `strat_floor` in `decide`, not wired to a file). The headway/q3 ref readers are untested on real UE.A dumps (run after the refs finish).

## Main chain launch line (after refs DONE and the four pod checkpoints exist; MM says go)
```
powershell -c "$s=([wmiclass]'Win32_ProcessStartup').CreateInstance(); $s.CreateFlags=0x08000200; $r=([wmiclass]'Win32_Process').Create('cmd.exe /c F:\TanitAD-scratch\rollhist\read\rh_read_chain.cmd < NUL','F:\TanitAD-scratch\rollhist\read',$s); 'pid='+$r.ProcessId+' rv='+$r.ReturnValue"
```
Deletions: none (`DELETIONS.log` untouched).
