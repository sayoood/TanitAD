# RESULT_PRH_SEED_V2 (2026-11-11 10:55:17 UTC)

**Outcome: O2 PLATONIC-OTHER** with metric CC32 (P1=True, P2=False; controls all pass: True)

Evidence class: MEASURED, ours; artifact `RESULT_PRH_SEED_V2.json`. Calibration: `CALIB_PRH_SEED_V2.json` sha256 4a3ca969aa740a4ce0d6f44f031852a2828c1aa9fb99b9a073878ea71e405678

| S | point | CI95 |
|---|---|---|
| S(TR0,TR1) | 0.9938 | [0.9933, 0.9941] |
| S(RI0,RI1) | 0.9769 | [0.9749, 0.9782] |
| S(TR0,SLOW) | 0.9510 | [0.9468, 0.9536] |
| S(TR1,SLOW) | 0.9515 | [0.9475, 0.9541] |
| S(RI0,SLOW) | 0.9424 | [0.9378, 0.9451] |
| S(RI1,SLOW) | 0.9431 | [0.9390, 0.9455] |
| S(TR0,PCA) | 0.9974 | [0.9972, 0.9976] |
| S(TR1,PCA) | 0.9974 | [0.9971, 0.9975] |
| S(TR0,DMIX2) | 0.9944 | [0.9939, 0.9948] |
| S(TR0,TR0) | 1.0000 | [1.0000, 1.0000] |

| diff | point | CI95 | separated, positive |
|---|---|---|---|
| P1: S(TR0,TR1)-S(RI0,RI1) | 0.0169 | [0.0155, 0.0186] | True |
| P2a_s0: S(TR0,SLOW)-S(RI0,SLOW) | 0.0087 | [0.0048, 0.0127] | True |
| P2b_s0: S(TR0,SLOW)-S(TR0,PCA) | -0.0464 | [-0.0505, -0.0438] | False |
| P2a_s1: S(TR1,SLOW)-S(RI1,SLOW) | 0.0084 | [0.0048, 0.0122] | True |
| P2b_s1: S(TR1,SLOW)-S(TR1,PCA) | -0.0458 | [-0.0498, -0.0432] | False |

## Controls

```json
{
 "C-SELF": {
  "S": 1.0,
  "pass": true
 },
 "C-ROT": {
  "S_rotated": 0.9937678059300208,
  "S_plain": 0.9937678059300069,
  "absdiff": 1.3877787807814457e-14,
  "pass": true
 },
 "C-SHUF": {
  "S": -0.0031880207262925675,
  "ci95": [
   -0.006402149233481493,
   0.0003359507347770776
  ],
  "S(RI0,RI1)": 0.9769078196357555,
  "mutual_10NN": 0.006898907103825138,
  "chance": 0.006830601092896175,
  "pass": true
 },
 "D-MIX2": {
  "S(TR0,DMIX2)": 0.9944036926147561,
  "S(TR0,TR1)": 0.9937678059300069,
  "delta": 0.0006358866847492317,
  "ci95": [
   0.00014984030005321113,
   0.0011367835623962623
  ],
  "pass": true
 },
 "C-ZC": {
  "s0": [
   {
    "clip": "01be5919-6e61-45ba-a193-5f447efb4b51",
    "T": 101,
    "max_abs_diff": 9.5367431640625e-07,
    "ok_equal": true
   },
   {
    "clip": "060531e1-9935-482e-8c25-2abf37bdc41b",
    "T": 101,
    "max_abs_diff": 1.9073486328125e-06,
    "ok_equal": true
   },
   {
    "clip": "0fa2b231-435e-4700-9138-30f333dc5d6e",
    "T": 101,
    "max_abs_diff": 9.5367431640625e-07,
    "ok_equal": true
   }
  ],
  "s1": [
   {
    "clip": "01be5919-6e61-45ba-a193-5f447efb4b51",
    "T": 101,
    "max_abs_diff": 2.384185791015625e-07,
    "ok_equal": true
   },
   {
    "clip": "060531e1-9935-482e-8c25-2abf37bdc41b",
    "T": 101,
    "max_abs_diff": 2.384185791015625e-07,
    "ok_equal": true
   },
   {
    "clip": "0fa2b231-435e-4700-9138-30f333dc5d6e",
    "T": 101,
    "max_abs_diff": 1.7881393432617188e-07,
    "ok_equal": true
   }
  ],
  "pass": true
 }
}
```

## Beside (no bar)

* TR0_vs_TR1: Procrustes R2 0.240, mutual 10-NN 0.889 (chance 0.0068); profile along SLOW mean dirs 1-16 0.930, 17-64 0.736, 65-256 0.605 (256 values in the JSON)
* RI0_vs_RI1: Procrustes R2 0.073, mutual 10-NN 0.873 (chance 0.0068); profile along SLOW mean dirs 1-16 0.898, 17-64 0.702, 65-256 0.546 (256 values in the JSON)
