# RESULT_RBSTACK

Card PREREG_RBSTACK.md sha256 bf4d5395bb481cbeaca552bd66a987071e772655b11f8a1d1a8bda2e22c72901. Tier T0 decodability, NOT driving performance. Evidence class MEASURED (ours).
Estimator: paired episode-cluster bootstrap, B 2,000, seed 0, on per-window correctness; answers another draw of episodes only. Probe variance: seeds 0 and 1.

## Reading: **STACK-WEAK**  (by seed: {'0': 'STACK-WEAK', '1': 'STACK-WEAK'})

Controls: {'K-REPRO': 'PASS', 'K-NEST': 'PASS', 'K-SHUF': 'PASS', 'K-V': 'PASS'}

| seed | arm | accuracy [95% CI] |
|---|---|---|
| 0 | EDV | 0.6623 [0.6364, 0.6867] |
| 0 | EDV_SHUF | 0.6282 [0.6011, 0.6550] |
| 0 | ED_STR | 0.6389 [0.6093, 0.6671] |
| 0 | V | 0.6255 [0.5971, 0.6537] |
| 0 | S | 0.4075 [0.3682, 0.4472] |
| 0 | ED | 0.6406 [0.6122, 0.6696] |
| 1 | EDV | 0.6665 [0.6398, 0.6913] |
| 1 | EDV_SHUF | 0.6273 [0.5994, 0.6535] |
| 1 | ED_STR | 0.6389 [0.6093, 0.6671] |
| 1 | V | 0.6295 [0.5997, 0.6579] |
| 1 | S | 0.4075 [0.3682, 0.4472] |
| 1 | ED | 0.6406 [0.6122, 0.6696] |

| seed | contrast | delta | CI | separated |
|---|---|---|---|---|
| 0 | EDV-ED | +0.0217 | [+0.0012, +0.0434] | True |
| 0 | EDV-ED_STR | +0.0234 | [+0.0029, +0.0445] | True |
| 0 | EDV_SHUF-ED | -0.0125 | [-0.0215, -0.0035] | True |
| 0 | EDV_SHUF-ED_STR | -0.0107 | [-0.0198, -0.0021] | True |
| 0 | V-S | +0.2180 | [+0.1808, +0.2549] | True |
| 0 | EDV-V | +0.0368 | [+0.0080, +0.0633] | True |
| 0 | ED-S | +0.2331 | [+0.1906, +0.2748] | True |
| 1 | EDV-ED | +0.0259 | [+0.0049, +0.0484] | True |
| 1 | EDV-ED_STR | +0.0276 | [+0.0075, +0.0496] | True |
| 1 | EDV_SHUF-ED | -0.0133 | [-0.0223, -0.0051] | True |
| 1 | EDV_SHUF-ED_STR | -0.0116 | [-0.0210, -0.0026] | True |
| 1 | V-S | +0.2220 | [+0.1858, +0.2577] | True |
| 1 | EDV-V | +0.0370 | [+0.0088, +0.0638] | True |
| 1 | ED-S | +0.2331 | [+0.1906, +0.2748] | True |

n windows 9236 / episodes 395; majority rate 0.4075.
EDV, EDV_SHUF, ED_STR fit on strided labelled fold-train windows (probe features exist for those only); ED and S are the dense RBJK arms.

K-REPRO: {"ED": {"mine": 0.6406453009961022, "rbjk": 0.6406453009961022, "pass": true}, "S": {"mine": 0.4075357297531399, "rbjk": 0.4075357297531399, "pass": true}, "n_windows": {"mine": 9236, "rbjk": 9236}, "pass": true}

## Beside: 5-way lat_cls, probe seed 0, no bar

```
{
 "population": {
  "windows": 12201,
  "episodes": 426
 },
 "paired": {
  "EDV-ED": {
   "delta": -0.0064748791082698265,
   "lo": -0.026,
   "hi": 0.013,
   "separated": false,
   "n_windows": 12201,
   "n_episodes": 426
  },
  "EDV-ED_STR": {
   "delta": -0.004098024752069551,
   "lo": -0.0226,
   "hi": 0.0145,
   "separated": false,
   "n_windows": 12201,
   "n_episodes": 426
  },
  "EDV_SHUF-ED": {
   "delta": -0.012212113761167154,
   "lo": -0.0209,
   "hi": -0.003,
   "separated": true,
   "n_windows": 12201,
   "n_episodes": 426
  },
  "EDV_SHUF-ED_STR": {
   "delta": -0.009835259404966878,
   "lo": -0.0184,
   "hi": -0.0011,
   "separated": true,
   "n_windows": 12201,
   "n_episodes": 426
  },
  "V-S": {
   "delta": 0.026637160888451805,
   "lo": 0.0043,
   "hi": 0.0493,
   "separated": true,
   "n_windows": 12201,
   "n_episodes": 426
  },
  "EDV-V": {
   "delta": 0.024997950987623918,
   "lo": 0.0075,
   "hi": 0.0413,
   "separated": true,
   "n_windows": 12201,
   "n_episodes": 426
  },
  "ED-S": {
   "delta": 0.05810999098434555,
   "lo": 0.0287,
   "hi": 0.0857,
   "separated": true,
   "n_windows": 12201,
   "n_episodes": 426
  }
 },
 "arms": {
  "EDV": {
   "accuracy": {
    "point": 0.7091222030981067,
    "lo": 0.6839,
    "hi": 0.7338,
    "n_windows": 12201,
    "n_episodes": 426
   },
   "pred_counts": [
    8967,
    2101,
    1133,
    0,
    0
   ]
  },
  "EDV_SHUF": {
   "accuracy": {
    "point": 0.7033849684452094,
    "lo": 0.6783,
    "hi": 0.7273,
    "n_windows": 12201,
    "n_episodes": 426
   },
   "pred_counts": [
    8986,
    2282,
    933,
    0,
    0
   ]
  },
  "ED_STR": {
   "accuracy": {
    "point": 0.7132202278501762,
    "lo": 0.6873,
    "hi": 0.7379,
    "n_windows": 12201,
    "n_episodes": 426
   },
   "pred_counts": [
    8813,
    2424,
    964,
    0,
    0
   ]
  },
  "V": {
   "accuracy": {
    "point": 0.6841242521104828,
    "lo": 0.6558,
    "hi": 0.7124,
    "n_windows": 12201,
    "n_episodes": 426
   },
   "pred_counts": [
    9149,
    2085,
    967,
    0,
    0
   ]
  },
  "S": {
   "accuracy": {
    "point": 0.657487091222031,
    "lo": 0.6244,
    "hi": 0.6914,
    "n_windows": 12201,
    "n_episodes": 426
   },
   "pred_counts": [
    12201,
    0,
    0,
    0,
    0
   ]
  },
  "ED": {
   "accuracy": {
    "point": 0.7155970822063765,
    "lo": 0.6892,
    "hi": 0.7409,
    "n_windows": 12201,
    "n_episodes": 426
   },
   "pred_counts": [
    8892,
    2382,
    927,
    0,
    0
   ]
  }
 },
 "reading_this_seed_for_reference_only": "STACK-SILENT",
 "rbjoint_reference": {
  "ED_acc": 0.7155970822063765,
  "RAW8_acc": 0.6841242521104828
 },
 "feature_replay_check": {
  "per_fold_agreement_with_rbjoint_RAW8": [
   1.0,
   1.0,
   1.0,
   1.0,
   1.0
  ],
  "note": "diagnostic of the re-extracted features; drift from thread counts can flip a few windows"
 },
 "controls_seed": {
  "K-NEST": "PASS",
  "K-SHUF": "PASS",
  "K-V": "PASS"
 }
}
```

## Stage B (held-out D1-HO windows)

NOT-YET: the NAVRBD D1-HO windows are the 10k data-scale RB grid windows (5,736 / 497 under the card, 7,594 / 545 in the NAVRBD count). Their pooled 8x20 DINOv3 features and join/ED rows are not on this machine (strat1/navrbd/in is absent; the cached pooled grids here cover only the train4572 and eval-141 clips), so stage B needs the pod-side cache, and the card forbids pulling from the pod and re-selecting episodes. It never changes the section 3 reading.
