# S0 read: true-raster ceiling of a drivable-area check on P6 (navtrain FIT sets, zero GPU)

**Outcome: S0-GO** (controls valid: True). R_max = **0.3476** (V0). Card bars: GO >= 0.33, NEAR [0.28, 0.33), STOP < 0.28.

> IN-SAMPLE CAVEAT (card S0): the FIT sets are in-sample for P6, so P6 fails on them less often than on navtest (PDMS-0 picks 1.01 % in-sample vs 3.06 % on E1); R is a ratio of gains and is read as a CEILING ESTIMATE, not a navtest value.

Evidence class: MEASURED (ours; artifacts below), open-loop selection read on navtrain, no navtest token read.

## Counts (before the read)

```
{
 "pack_sets": 8489,
 "pack_logs": 101,
 "sets_in_label_store": 8489,
 "logs_in_label_store": 101,
 "pack_has_pdm_all": true,
 "heldout_flagged_sets_in_population": 1705
}
```

Bootstrap: paired log-cluster, 10000 draws, seed 20260927, 101 logs. paired log-cluster bootstrap over the pack's logs (sel_common.log_boot_idx draws); it answers 'another draw of logs' only; R CI = percentile of per-draw gain_rule / gain_O-DAC.

P6's pick PDMS x100 on these sets: 94.491; best-of-64: 98.102. Footprint points outside the raster extent: 0.00000; proposals with no point inside: 0.

## Arms (gain = PDMS x100 over the sets minus P6's pick; R = gain / gain(O-DAC))

| arm | gain [95 % CI] | R [95 % CI] | picks changed |
|---|---|---|---|
| O-DAC | +0.5861 [+0.2825, +0.9985] | 1 (reference) | 54 |
| V0 | +0.2037 [+0.0226, +0.4398] | 0.3476 [0.0653, 0.5120] | 123 |
| V25 | +0.1642 [+0.0557, +0.3088] | 0.2801 [0.1455, 0.3933] | 47 |
| Phi-GT0.5 | +0.0564 [-0.1887, +0.3575] | 0.0962 [-0.6158, 0.3852] | 3117 |
| Phi-GT1.0 | -0.1267 [-0.3478, +0.1345] | -0.2161 [-1.1280, 0.1434] | 3816 |
| Phi-GT2.0 | -0.2517 [-0.4553, -0.0427] | -0.4294 [-1.3912, -0.0468] | 4427 |

R_max (V0 and the three Phi-GT rules; V25 is reported, not in the max) = 0.3476, arm V0; CI of the max over those four arms [0.0665, 0.5121] (the point decides, per the card).

## Controls

```
{
 "zero_all_drivable": {
  "ok": true,
  "n_changed": {
   "V0": 0,
   "V25": 0,
   "Phi-GT0.5": 0,
   "Phi-GT1.0": 0,
   "Phi-GT2.0": 0
  },
  "delta_sum": {
   "V0": 0.0,
   "V25": 0.0,
   "Phi-GT0.5": 0.0,
   "Phi-GT1.0": 0.0,
   "Phi-GT2.0": 0.0
  }
 },
 "pick_reproduces": {
  "ok": true,
  "n_sets": 8489,
  "n_equal": 8489,
  "note": "the pack records no pick; recomputed pick from the pack's logits via sel_common.pick_from_logits(torch, lane=False)"
 },
 "odac_ge_all": {
  "ok": true,
  "O-DAC": 0.5861387683640578,
  "rules": {
   "V0": 0.2037350695269986,
   "V25": 0.16419471981303826,
   "Phi-GT0.5": 0.05635834948616913,
   "Phi-GT1.0": -0.12666167727992902,
   "Phi-GT2.0": -0.2516728828829328
  }
 },
 "yflip_below_true": {
  "ok": true,
  "R_max_true": 0.34758845605048994,
  "R_max_flip": -5.487249894166709,
  "per_rule_flip_below_true": {
   "V0": true,
   "Phi-GT0.5": true,
   "Phi-GT1.0": true,
   "Phi-GT2.0": true
  }
 }
}
```

## Reported, not deciding

```
{
 "R_yflipped_field": {
  "V0": -5.487249894166709,
  "Phi-GT0.5": -14.447229619577804,
  "Phi-GT1.0": -12.268861531750385,
  "Phi-GT2.0": -10.168085581285586
 },
 "R_stored_sdf_field": {
  "V0": 0.3477080347265494,
  "V25": 0.2515274117562786,
  "Phi-GT0.5": 0.0548005777280525,
  "Phi-GT1.0": -0.25369413303193417,
  "Phi-GT2.0": -0.36412834929152776
 },
 "alignment": {
  "proposals_dac0": 68257,
  "share_dac0_with_d_lt0": 0.840734283663214,
  "share_dac1_with_d_lt0": 0.036285441826881586,
  "p6_pick_dac0_sets": 61
 }
}
```

## Inputs

```
{
 "pack": "F:/tanitad_scratch/p6fit/fit_pack8.npz",
 "pack_sha256": "9a6f7f1b568ab2b20589cb309fb19eaa684a22b60853d9107eacecaaffe92d7c",
 "label_store_sha256": "4a17db673b1548ba35724ab848a7dc551e4b0bc1e49c57ae93e6357454bf287a",
 "card_sha256": "5594c2ae777a53ed4a8a873ca374c12943eadc74385cb07671c2ac90a3292ae4",
 "sel_common": "E:/Projects/TanitAD-artifacts/refe_selection_probe_20261107/sel_common.py",
 "sel_common_sha256": "50727023525c86b2be503db5c7f84d56de9b2be98f260aab700e4c9438f27d23"
}
```

Elapsed 79.0 s. Method: `s0_read.py` (this folder), DESIGN_VDAC_P6.md 5.1 verbatim, pose-only.
