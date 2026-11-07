# RESULT_VOE1: does each level notice an impossible future? (violation of expectation on real driving video; tier T0)
Card `../PREREG_V8F_VOE1.md` (frozen prefix 5,822 bytes, sha256 `555c5a38...`, unedited). Definitions: `IMPL_CHOICES_VOE1.md` (sha256 `848a2c8c`). Models: M2b s0 and s1, zero training; controls: random-init world models. 1,976 eval-141 windows (141 episodes), onset t0 = window frame 4, 1.6 s = 8 cache steps scored. **Tier T0** (a world-model diagnostic on cached latents, teacher-forced one-step prediction, never a driving claim). Every number below is computed by `run_voe1.py` into `RESULT_VOE1.json` (run 2026-11-07 15:12:28 to 2026-11-07 15:16:00, 214 s).

## 0. Controls (checked before any row is read)
**Run status: INVALID** -- control 1: separated from 0.5 at ['RI_s0/L3']; control 2: random-init NOTICES at ['RI_s1/ACTION-SWAP/L3']

| control | rule | result | pass |
|---|---|---|---|
| 1 real-vs-real | the second copy from the window 0.2 s later: not two-part separated from 0.5 at any level (M2b s0/s1, RI s0/s1; 12 cells) | points 0.469 to 0.514; separated: RI_s0/L3 | FAIL |
| 2 random-init WM | RI_s0/RI_s1 (every wm.* parameter re-initialised) not NOTICES on FREEZE and ACTION-SWAP at any level (12 cells) | NOTICES at: RI_s1/ACTION-SWAP/L3 | FAIL |
| 3 calibration | 20 extra permutation draws (1001-1020) scored as if real: at most 5 % of rows separated per family | ACC 52/1800 = 2.889%; NULL-ACC 30/1800 = 1.667%; DIFF 51/1800 = 2.833%; RVR 14/360 = 3.889%; AP-CHANCE 6/360 = 1.667%; AP-NULL 3/360 = 0.833% | PASS |
| 4 synthetic test | `test_voe1_synth.py`: toy constant-velocity model NOTICES on planted FREEZE and TELEPORT, action-blind toy BLIND on ACTION-SWAP; 13 tests | 13 passed; test file and core hash equal the log's: True | PASS |

Control 1 cells (accuracy of 'the later copy scores higher', two-part rule, either side):

| cell | n | accuracy [interval] | separated either side |
|---|---|---|---|
| M2b_s0/L1 | 1976 | 0.497 [0.489, 0.506] | False |
| M2b_s0/L2 | 1976 | 0.488 [0.461, 0.516] | False |
| M2b_s0/L3 | 1976 | 0.492 [0.467, 0.518] | False |
| M2b_s1/L1 | 1976 | 0.504 [0.495, 0.513] | False |
| M2b_s1/L2 | 1976 | 0.487 [0.460, 0.515] | False |
| M2b_s1/L3 | 1976 | 0.502 [0.475, 0.528] | False |
| RI_s0/L1 | 1976 | 0.509 [0.498, 0.521] | False |
| RI_s0/L2 | 1976 | 0.503 [0.475, 0.529] | False |
| RI_s0/L3 | 1976 | 0.469 [0.444, 0.493] | True |
| RI_s1/L1 | 1976 | 0.498 [0.485, 0.510] | False |
| RI_s1/L2 | 1976 | 0.494 [0.469, 0.519] | False |
| RI_s1/L3 | 1976 | 0.514 [0.488, 0.539] | False |

Reproduction (MEASURED): the unmodified path (per-clip frame bank + adapter + wm.z1) vs `latents.pt` `Z1`, 1,976 windows: max abs difference 3.81e-06 (s0), 2.86e-06 (s1) on values up to 4.9; logged actions 0.0. Random-init: RI re-initialised 2,007,874 of 2,007,874 wm.* parameters (133 tensors: proj, act_embed, level encoders, macro-encoders, predictors, idm heads); RIs (beside) re-initialised 1,974,466 (the stack), trained wm.proj kept; chassis (adapter, pos) trained in both.

**INVALID: no label table is printed. The failing controls above are the finding; the run is not read.** Raw rows stay in `RESULT_VOE1.json`.
