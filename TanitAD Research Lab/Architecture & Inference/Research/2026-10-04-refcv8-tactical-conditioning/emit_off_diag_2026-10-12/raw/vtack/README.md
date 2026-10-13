# V-TACk (L2 treatment) descriptive read, 2026-10-13 ~02:45Z. POST-HOC; L2 is scored at I-0.

V-TACk = V-R8 with `--w-tac-v6 100` (V-R8: 1.0); k = 100 from SIZE_K.json. Thor batch-8 dumps (md5 above), eval139,
paired episode-cluster bootstrap, sampler seeds 0 / 1. Every delta is read against the R8 recipe's OWN run-to-run
spread (V-R8r − V-R8, same flags, training seed 1 vs 0), because the episode bootstrap cannot see training variance.

* **Premise (B-PREM) reading:** `gs_trunk_proj_tac_v6` readings at steps 500 / 2000 / 2500: −0.0023 / −0.0022 /
  +0.0107. The median over the last 50 % of steps is **0.0042**, below the 0.01 premise (V-R8: 0.0001). Raising the
  weight 100× moved the tactical term's share of the trunk gradient about 40×, but not to the premise. This reads
  PREMISE-UNMET; the formal scorer decides at I-0.
* **Beyond the replicate spread (both seeds, separated, |Δ| larger than V-R8r − V-R8): all WORSE.**
  `cons_mae_lat` +0.058, `cons_mae_lon` +0.061, `cons_mae_spd` +0.092 (replicate ≤ 0.007); `speed_mae_0_2s`
  +0.030 / +0.042 (replicate +0.007 / +0.010); `off_drivable` +0.029 / +0.032 (replicate −0.002 / +0.004).
* **Within the replicate spread (not attributable to the weight):** turns (cross +1.9 / +2.0 m vs a replicate
  +3.3 / +2.95 m), `dir_correct_turn`, all-window cross and ADE, lane IoU +0.013 (replicate −0.054).
* No family improved beyond the replicate. `lat3_f1_v9` is ns (−0.037).
