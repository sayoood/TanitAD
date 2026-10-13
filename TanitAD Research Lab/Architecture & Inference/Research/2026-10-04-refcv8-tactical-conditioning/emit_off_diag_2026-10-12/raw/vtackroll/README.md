# V-TACk-roll (L2 deliberate regression) descriptive read, 2026-10-13 ~07:55Z. POST-HOC; L2 is scored at I-0.

V-TACk-roll = V-TACk + `--r8-roll-targets tac` (tactical targets rolled by one row: same gradient magnitude, zero
information). Thor batch-8 dumps (md5 above), eval139, paired episode-cluster bootstrap, sampler seeds 0 / 1.

* **The scramble took effect (the control reads its known value).** The lateral tactical head (`p_lat`, argmax over
  1,112 windows) predicts class 0 (lane keep) for **1,112 / 1,112** windows (V-TACk: 975 / 33 / 104 over classes 0 /
  6 / 7). `lat3_f1_v9` 0.7244 → **0.3149**, constraint MAEs +0.04 / +0.05 / +0.11 (all separated worse).
* **B-REG (SPEC_WPB_LADDER §6: "V-TACk-roll's route gain over V-R8 is ≤ F") reads as holding:** against V-R8,
  `dir_correct_turn` −0.075 (ns), `head15_all` +0.005 / +0.003 (ns). No route gain.
* `gs_trunk_proj_tac_v6` last-50 % median **0.0003** (V-TACk 0.0042, V-R8 0.0001): scrambled targets carry little
  projected gradient.
* **V-TACk's costs vanish with the information:** against V-TACk, `speed_mae_0_2s` −0.031 / −0.045 and drivable IoU
  +0.016 (separated better), `off_drivable` −0.005 / −0.022. The 100× INFORMATIVE tactical signal is what cost speed and
  off-drivable in V-TACk.
* ⚠ **CAVEAT for every "the R8 tactical head is better than V0's" statement in this programme's tiny-rung reads:** V0's
  `p_lat` ALSO predicts class 0 for 1,112 / 1,112 windows. V0's `lat3_f1_v9` 0.3149 / `lat3_acc_v9` 0.8954 /
  `lat3_acc_gtplan` 0.8645 are exactly V-TACk-roll's, to 4 dp. They are the constant-majority-class values. The R8
  head's gain is therefore over a DEGENERATE V0 head at this rung (3,000 steps, resnet34), not over a working one.
