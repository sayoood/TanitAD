# DIAG_CONTROLS_VOE1 (POST HOC; not part of the card; the INVALID verdict stands as written)
Source: `RESULT_VOE1.json` (control rows only) and a re-score of the real-vs-real control at larger shifts. No M2b manipulation row is read here.

## (a) Family-wise false-fail rate of the controls as committed (from the run's own 20 calibration draws)
* Control 1 (12 cells, either side): per-cell separated share 3.333%; 7 of 20 calibration draws had at least one of the 12 cells separated; independent-cells expectation 33%.
* Control 2 (12 cells, NOTICES-like = separated above 0.5): per-cell share 3.333%; 8 of 20 draws had at least one; independent-cells expectation 33%.
* A permuted draw has no signal by construction, so these are the rates at which a clean pipeline fails the control as worded ('at any level', 12 cells each). Cells within a calibration draw are independent (own coin flips), real cells are not, so the real rate is lower than the independent figure but not zero.

## (b) Control 1 cells (shift 1 = the card's 0.2 s)
| cell | accuracy [interval] | reference 2.5 / 97.5 % | flag |
|---|---|---|---|
| M2b_s0/L1 | 0.497 [0.489, 0.506] | 0.492 / 0.509 | - |
| M2b_s0/L2 | 0.488 [0.461, 0.516] | 0.472 / 0.525 | - |
| M2b_s0/L3 | 0.492 [0.467, 0.518] | 0.475 / 0.526 | - |
| M2b_s1/L1 | 0.504 [0.495, 0.513] | 0.492 / 0.507 | - |
| M2b_s1/L2 | 0.487 [0.460, 0.515] | 0.473 / 0.527 | - |
| M2b_s1/L3 | 0.502 [0.475, 0.528] | 0.474 / 0.524 | - |
| RI_s0/L1 | 0.509 [0.498, 0.521] | 0.487 / 0.512 | - |
| RI_s0/L2 | 0.503 [0.475, 0.529] | 0.475 / 0.526 | - |
| RI_s0/L3 | 0.469 [0.444, 0.493] | 0.478 / 0.526 | S-below |
| RI_s1/L1 | 0.498 [0.485, 0.510] | 0.487 / 0.511 | - |
| RI_s1/L2 | 0.494 [0.469, 0.519] | 0.475 / 0.521 | - |
| RI_s1/L3 | 0.514 [0.488, 0.539] | 0.473 / 0.527 | - |

## (c) Control 2 cells (random-init, accuracy above 0.5 needed for NOTICES)
| cell | n | accuracy [interval] | reference 97.5 % | rank (above) | label |
|---|---|---|---|---|---|
| RI_s0/FREEZE/L1 | 1725 | 0.355 [0.333, 0.376] | 0.528 | 1.000 | BLIND |
| RI_s0/FREEZE/L2 | 1725 | 0.396 [0.374, 0.416] | 0.534 | 1.000 | BLIND |
| RI_s0/FREEZE/L3 | 1725 | 0.494 [0.475, 0.514] | 0.519 | 0.765 | BLIND |
| RI_s0/ACTION-SWAP/L1 | 1976 | 0.391 [0.356, 0.423] | 0.534 | 1.000 | BLIND |
| RI_s0/ACTION-SWAP/L2 | 1976 | 0.380 [0.351, 0.409] | 0.535 | 1.000 | BLIND |
| RI_s0/ACTION-SWAP/L3 | 1976 | 0.392 [0.359, 0.424] | 0.537 | 1.000 | BLIND |
| RI_s1/FREEZE/L1 | 1725 | 0.366 [0.345, 0.388] | 0.529 | 1.000 | BLIND |
| RI_s1/FREEZE/L2 | 1725 | 0.424 [0.401, 0.447] | 0.524 | 1.000 | BLIND |
| RI_s1/FREEZE/L3 | 1725 | 0.511 [0.495, 0.526] | 0.515 | 0.115 | BLIND |
| RI_s1/ACTION-SWAP/L1 | 1976 | 0.510 [0.479, 0.540] | 0.528 | 0.260 | BLIND |
| RI_s1/ACTION-SWAP/L2 | 1976 | 0.525 [0.494, 0.558] | 0.527 | 0.060 | BLIND |
| RI_s1/ACTION-SWAP/L3 | 1976 | 0.550 [0.515, 0.582] | 0.533 | 0.005 | NOTICES |

## (d) Real-vs-real with the later copy 0.4 to 1.0 s later (a systematic bias would persist across shifts)
| cell | shift 1 (card) | shift 2 | shift 3 | shift 4 | shift 5 |
|---|---|---|---|---|---|
| M2b_s0/L1 | 0.497 - | 0.495 - | 0.491 - | 0.493 - | 0.494 - |
| M2b_s0/L2 | 0.488 - | 0.501 - | 0.471 S-below | 0.502 - | 0.481 - |
| M2b_s0/L3 | 0.492 - | 0.492 - | 0.504 - | 0.518 - | 0.488 - |
| M2b_s1/L1 | 0.504 - | 0.503 - | 0.500 - | 0.499 - | 0.504 - |
| M2b_s1/L2 | 0.487 - | 0.492 - | 0.486 - | 0.494 - | 0.486 - |
| M2b_s1/L3 | 0.502 - | 0.509 - | 0.503 - | 0.537 S-above | 0.497 - |
| RI_s0/L1 | 0.509 - | 0.518 S-above | 0.512 - | 0.518 - | 0.518 - |
| RI_s0/L2 | 0.503 - | 0.527 S-above | 0.510 - | 0.532 S-above | 0.526 - |
| RI_s0/L3 | 0.469 S-below | 0.504 - | 0.505 - | 0.493 - | 0.481 - |
| RI_s1/L1 | 0.498 - | 0.489 - | 0.491 - | 0.491 - | 0.492 - |
| RI_s1/L2 | 0.494 - | 0.498 - | 0.500 - | 0.504 - | 0.497 - |
| RI_s1/L3 | 0.514 - | 0.495 - | 0.482 - | 0.491 - | 0.482 - |
