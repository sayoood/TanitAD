# BPC stage-5 read (2026-11-07): the route term is RT-WITHIN-FLOOR, narrowly

* **Card:** `PREREG_T2_ATTRIB.md`, frozen `f7d2cfa1`. `PREREG_T2_ATTRIB_with_log.md` is the card with its dated log; the last entry is this read.
* **Arm:** BPC is the composite of B′, which is M2b retrained on the PI's GPU pod from a cache re-encoded there. It ran on box 2 from 04:29 to 06:03 with the stage-5 driver `1fb1af10…`, the same bytes MC ran. `pull5_BPC/` is the whole pull: 1,027 files, 44,462,390 bytes, equal on both sides.
* **Reads:**
  * `EXCLUSIONS5.json`: R-SCN ATTRIBUTED, the same 14 scenes.
  * `score_dom/`: the pinned read (pin 4) of BPC against P10. NOT-DOMINANT at both seeds.
  * `pair_BPC_MC/` and `RT_VERDICT.json` (from `rt_verdict.py`): the route term RT = BPC − MC by the card's rule.
  * `pair_C0C_BPC/`: C0C − BPC beside.
* **Verdict, RT-WITHIN-FLOOR.**
  * At s0, RT is not separated on M1, ADE6 or along-track MAE.
  * At s1 all three are separated, BPC worse: M1 +0.0510, ADE6 +0.4634 m, along-track +0.4014. Each stays below MC's seed floor (0.0574, 0.5089, 0.4390), with margins of 0.0064, 0.046 and 0.038.
  * So LC − C0C is admissible as the mechanism row, beside LC − (the B′ control).
* **Not here:** the overlay tar `t2v8f_stage5_overlay_BPC.tar` (174 MB, it holds the checkpoints) stays on E:. Its sha256 is in `t2v8f_stage5_overlay_BPC.tar.sha256`.
* **Driver:** `bpc_read.sh` ran the whole read in one pass.
