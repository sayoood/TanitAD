# PREREG_VDAC_P6_A3 (FROZEN): the H48 held-out log list

**Status: FROZEN by the Master Mind, by sha256, 2026-11-11, BEFORE any S1 training and before any H48 read** (drafted by the S1 build agent; A2 section 6, card section 1). The --epochs ruling below is the MM ruling of 2026-11-11.
It amends `PREREG_VDAC_P6.md` (sha256 `5594c2ae...`) in one point only: the card said "list and sha256 written into this card at freeze"; this is that list.

## The list

- File: `h48_logs.txt`, 48 log names, one per line, sorted ascending, LF, trailing newline.
- **sha256 `2ae13935a3ead7f30b015f33b0bebc7a35e70eb2a4194161f8cbb66ac89b8da8`**.
- Shipped to the pod as `splits/h48_logs.txt` (the package) and copied to `/workspace/data/refe_vdac/h48_logs.txt`; the trainer, the launch gate (profile `refe_vdac_p6`, `VDAC_H48_SHA256`) and the chain each re-hash the bytes.
- Built by `h48/build_h48.py` (identical bytes shipped as `refe/vdac_build_h48.py`) from the label store `E:/Projects/TanitAD-artifacts/aux_drivable/label_store` (store sha256 `4a17db673b15...`, asserted by the script).

## The rule (DESIGN_VDAC_P6 section 3.2, applied literally)

H48 = the 24 logs of `PK/splits/sft_heldout_logs.txt` (file sha256 `b814998deeb40a9d7839cce2807f8f282ede76c7f2c5ccce97adc064b27f4902`)
+ for each of SG, BOS and PIT, the **8 lowest `sha256(log name, utf-8)` hex digests** among that city's navtrain logs **in the label store** outside the 24.

**Where the city per log is recorded:** the label store's `manifest.json`, `files["shards/<log>.aux1"].map` (written by the store build from nuPlan's map name):
`us-nv-las-vegas-strip` = LV, `us-ma-boston` = BOS, `us-pa-pittsburgh-hazelwood` = PIT, `sg-one-north` = SG. (`aux_drivable/log_map.json` carries the same fact: 1,192 logs, 463 from the nuPlan DB, 729 inferred, 0 ambiguous.) Frames per log = the shard's `n` (labelled frames in the store).

## Counts (MEASURED on the store manifest, `h48/h48_report.json`)

| city | logs in H48 | labelled store frames | of which the 24 SFT logs | of which the 24 extra logs |
|---|---|---|---|---|
| LV | 15 | 1,013 | 15 logs | 0 |
| BOS | 12 | 534 | 4 logs | 8 logs, 277 frames |
| PIT | 12 | 923 | 4 logs | 8 logs, 728 frames |
| SG | 9 | 1,566 | 1 log | 8 logs, 1,323 frames |
| **all** | **48** | **4,036** | 24 logs, 1,708 frames (the card's F3 number, reproduced) | 24 logs, 2,328 frames |

The 24 SFT logs are already outside the trainer's training index (the P6 bank's held-out logs); the 24 extra logs are excluded from the head's labels by `--aux-drv-exclude-logs` (their rows stay in the scorer's bank; the head's label for them reads `valid = 0`, exactly 0 loss). The trainer MEASURES the number of labelled frames this removes from the bank it holds and refuses if any H48 row still reaches the loss (`aux_excl` block of the data event).

## Disclosed deviation from S1c (MM ruling 2026-11-11, travels with every S1 number)

The head draws its frames from **P6's 60,000-set bank** (the scorer's training index, minus P6's held-out logs), not from S1c's own bank, with `--epochs 2` named so that `--max-updates 10000` is the stop that binds (10,000 x batch 8 = 80,000 draws = 1.33 epochs of that bank). The number of distinct logs and distinct frames in that pool is NOT derivable off the pod (the bank rows live on the pod). The chain prints it BEFORE training: `ZZVDAC_G19_G20 <arm> ... bank {'sets': ..., 'distinct_logs': ..., 'distinct_frames': ...}` (measured on the training index by `vdac_probes.alignment_gate`, from the preflight). The run check asserts updates == 10,000 (HA-SHUF 2,000) and prints `ZZVDAC_UPDATES`.

## Disclosures that travel with every S1 number (from the card, unchanged)

All H48 logs are in P6's own training set: held-out for the HEAD, in-sample for P6's trunk and scorer. HA-SHUF's key map is a derangement of ALL 103,139 store keys, so under that control arm an H48 frame's label can be paired with another frame's image; the exclusion is keyed on the image's log, and HA-SHUF is the arm that must fail S1 anyway.

## The list (log, city, labelled frames, origin)

```
2021.06.09.14.50.36_veh-26_00832_00905  LV  11  SFT24
2021.06.09.18.23.43_veh-35_03110_03179  LV  35  SFT24
2021.06.14.14.03.45_veh-38_01624_01811  LV  11  SFT24
2021.06.14.14.25.15_veh-26_04417_04531  LV  12  SFT24
2021.06.14.16.32.09_veh-35_04516_04698  LV  19  SFT24
2021.06.14.17.26.26_veh-38_05455_05749  LV  117  SFT24
2021.06.14.18.13.35_veh-26_01537_01717  LV  30  SFT24
2021.06.14.18.13.35_veh-26_04547_04710  LV  89  SFT24
2021.06.14.18.42.45_veh-12_01345_01523  LV  12  SFT24
2021.06.14.18.42.45_veh-12_04233_04472  LV  43  SFT24
2021.06.14.19.22.11_veh-38_02275_02455  LV  96  SFT24
2021.06.23.14.54.32_veh-16_00862_01000  LV  18  SFT24
2021.06.23.15.56.12_veh-16_00839_01285  LV  123  SFT24
2021.07.09.20.26.06_veh-35_02793_03289  LV  161  SFT24
2021.07.09.20.59.12_veh-38_03292_04331  LV  236  SFT24
2021.08.09.17.55.59_veh-28_00021_00307  BOS  15  extra-8
2021.08.09.17.55.59_veh-28_00960_01031  BOS  7  extra-8
2021.08.17.13.10.50_veh-08_00122_00295  PIT  30  SFT24
2021.08.17.13.15.12_veh-45_00819_00884  PIT  60  extra-8
2021.08.17.13.15.12_veh-45_02304_02650  PIT  106  extra-8
2021.08.17.14.32.33_veh-08_00521_01051  PIT  133  extra-8
2021.08.17.14.45.12_veh-42_01866_01999  PIT  79  SFT24
2021.08.17.16.48.45_veh-43_01439_01665  PIT  78  extra-8
2021.08.17.17.17.01_veh-45_00762_01166  PIT  117  extra-8
2021.08.17.17.17.01_veh-45_02098_02251  PIT  47  extra-8
2021.08.17.18.54.02_veh-45_00016_00304  PIT  102  extra-8
2021.08.24.20.03.01_veh-45_00269_00428  PIT  47  SFT24
2021.08.24.20.03.01_veh-45_00687_00787  PIT  39  SFT24
2021.08.31.12.21.30_veh-40_00661_00762  BOS  20  SFT24
2021.08.31.13.27.52_veh-40_00688_00750  BOS  23  extra-8
2021.08.31.14.40.58_veh-40_01268_01618  BOS  122  extra-8
2021.08.31.18.15.54_veh-40_00579_00980  BOS  143  SFT24
2021.09.14.18.43.41_veh-45_02296_02477  PIT  85  extra-8
2021.09.15.14.00.15_veh-28_01953_02255  BOS  73  SFT24
2021.09.15.14.27.22_veh-39_00473_00568  BOS  30  extra-8
2021.09.15.14.27.22_veh-39_01420_01480  BOS  17  extra-8
2021.09.15.14.50.05_veh-28_00389_00508  BOS  43  extra-8
2021.09.15.15.02.19_veh-39_00105_00203  BOS  20  extra-8
2021.10.01.15.32.11_veh-28_00025_00097  BOS  21  SFT24
2021.10.05.04.03.05_veh-50_01003_01426  SG  102  extra-8
2021.10.05.04.03.05_veh-50_01466_01790  SG  243  SFT24
2021.10.05.06.57.40_veh-50_00665_00857  SG  169  extra-8
2021.10.05.07.10.04_veh-52_00689_01322  SG  369  extra-8
2021.10.05.07.49.39_veh-52_00934_01406  SG  303  extra-8
2021.10.11.05.34.05_veh-50_00838_00947  SG  29  extra-8
2021.10.11.05.34.05_veh-50_00971_01251  SG  103  extra-8
2021.10.11.08.31.07_veh-50_01184_01318  SG  137  extra-8
2021.10.11.08.31.07_veh-50_01365_01539  SG  111  extra-8
```
