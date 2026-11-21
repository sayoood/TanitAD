# READ FIX 2 (MM, 2026-11-21T06:36:41Z): the one-seed read's reader spread is F_U,c (Amendment 2 item 6)

* The provisional seed-0 read ran:
  * all 12 rollouts;
  * K-REPLAY (576 arrays, 0 differing);
  * the T0 evals, G-T0, and the refs readers (controls PASS).
* It stopped at `headway`: control h2_planted_regression FAILED with `spread_max` NaN, verdict INSIDE-SPREAD. The chain passed `--spread-arm I10RH_R0Hb20 --spread-seeds 0`, and a one-seed spread does not exist.
* Amendment 2 item 6 (FROZEN 01:29Z, sha256 15004dc9) runs the one-seed read "against F_U,c". F_U,c = abs(UE_s1.A - UE_s0.A) under R0C10XHb20 (item 5). `rh_read.py` already takes its one-seed floor from the references (lines 56-68). The chain did not pass that floor to the headway and family readers.
* Fix:
  * `rh_read_chain.sh` sets `SPREAD`: with one seed it is `--spread-arm I10UEA_R0C10XHb20 --spread-seeds 0,1` (the stage-A dumps already in out/t1); with two seeds, as before, the RH arm's own spread.
  * sha256 a2a4e24f → 87b6eb82; the copy is kept as `rh_read_chain.sh.bfix1`.
  * The failed reader output was moved, not deleted, to `out/read/hw_failed_spreadnan_s0/`.
* Before this fix the MM had seen only the control's own numbers and the G-T0 guard record. G-T0 is a frozen step whose output the chain prints; this fix does not touch it. No RH - B0 family cell was read.
* The two-seed read (of record) is unchanged by this fix.
