# The G-BOX-OVERFIT toy reads A17's decayed snapshot (addendum to `LANDING_READY_A17`)

**Box-head builder (Architecture & Inference), 2026-09-27.** Landing: `LANDING_READY_A17_TOY.txt`, one repo file
(`stack/tests/test_g_box_overfit.py`) ON TOP OF `LANDING_READY_A17.txt`. Its base is the blob A17 declares NEW for
that path (`2a582320…`), so A17 lands first. `BUILD.md` is deliberately NOT touched here: it is a package file of the
A17 list, and changing it now would change what that list lands.

## 1. What NEW-2 found

On a clean 2ac0bfb extraction on the dev box, `test_TOY_the_loop_memorises_and_the_memory_zeros_arm_cannot` FAILED
deterministically: ap2m 0.651569903443288 (bar 0.8), prec 0.8125, 600-step loss mean 5.5058 at step 1,200. It PASSED
on Thor in the Master Mind's gate (aarch64, `OMP_NUM_THREADS=4`).

## 2. Why: the intra-op thread count picks one draw from a trajectory that never settles (MEASURED)

`code/tools/toy_repro.py` runs the tree's OWN `_Toy` and `G.run_arm` at the test's arguments (lr 1e-3, 1,200 steps)
and logs every 100 steps. The toy decoder has no dropout and the eval rule draws no random numbers, so the logging
cadence never changes the training trajectory. Main arm, seed 0, at the TIP (constant lr):

| platform | threads | ap2m @ 1,200 | prec | the test |
|---|---|---|---|---|
| dev box (x86, torch 2.11.0+cu128, MKL) | 16 (its default) | 1.0000 | 1.0000 | PASS |
| dev box | 4 | **0.651569903443288** | **0.8125** | FAIL -- NEW-2's numbers to the last digit (600-step loss mean 5.5058) |
| dev box | 1 | 0.1717 | 0.4375 | FAIL |
| Thor (aarch64, gate env) | 4 | 1.0000 | 1.0000 | PASS -- the gate's reading |
| Thor | 1 | 0.1450 | 0.2708 | FAIL |

The trajectory shows the mechanism. At a CONSTANT lr, AP@2 m swings between 0.07 and 1.00 across the 100-step
readings of steps 700-1,200. The 1-thread run, for example, reads 1.00, 0.89, 0.96, 0.93 at steps 800-1,100 and 0.17 at
1,200. The median centre error sits at 0.55-2.57 m against the 2 m match threshold. **The loop memorised in every run;
the read-out never settled.** Floating-point reduction order (the thread count, x86 vs aarch64) decides which side of the
2 m edge step 1,200 lands on. It is not a seed bug: every seed is fixed. It is A17's failure (a constant-lr final
snapshot) in miniature (`raw/toy/devbox_tip_*.json`, `thor_tip_*.json`).

## 3. The fix: read the decayed snapshot, as the harness now does -- and prove it on both platforms

A17 is in `run_arm`: the toy's 1,200 steps decay over 1,080-1,200 to factor 0. The test now:

- reads MAIN through `_memorised_failures`. It REQUIRES the final row's `lr_factor == 0.0` (the decayed snapshot),
  ap2m and rec both > 0.8, and presence below step 0's. The bars are unchanged.
- reads memory_zeros through `_blinded_arm_failures`: ap2m < 0.5 and the harness verdict `failed as required`,
  unchanged.
- **RED arm 1 -- the defect the toy exists for:** `_LeakyToy`, a memory_zeros arm that still reads each frame's memory
  and therefore trains exactly as MAIN does. Both checks must fire, as literals: `["ap2m", "verdict"]`. The verdict must be
  `VOID (a must-fail arm passed its criteria)`.
- **RED arm 2 -- the read-out it relies on:** a loop whose lr never decays (`lr_factor` forced to 1) leaves the final
  snapshot at factor 1.0, and the check names A17 FIRST. 20 steps suffice, because only the snapshot's factor is at
  stake; the real schedule's 20-step run reads 0.0 in the same test.
- No thread pinning: the Master Mind preferred a criterion that holds on every reduction order to one pinned draw.

## 4. Evidence: 1 / 4 / 16 threads x seeds 0-2 x both platforms (MEASURED)

The A17 code: tip 37086c3 (whose `stack/` tree is 2ac0bfb's, `cea3d328…`) plus `LANDING_READY_A17`. Each run uses the
test's arguments, and each cell is judged by the ADDENDUM test's own `_memorised_failures` / `_blinded_arm_failures`,
IMPORTED from the test module (`code/tools/toy_grid_judge.py`). Table: `raw/toy/toy_grid_18cells.json`; all 36 runs:
`raw/toy/toy_grid_judge_a17.json`.

| platform | threads | seed | MAIN ap2m | MAIN rec | MAIN centre p50 (m) | memory_zeros ap2m | memory_zeros verdict |
|---|---|---|---|---|---|---|---|
| dev box (x86_64) | 1 | 0 | 1.0000 | 1.0000 | 0.106 | 0.0090 | failed as required |
| dev box (x86_64) | 1 | 1 | 1.0000 | 1.0000 | 0.105 | 0.0129 | failed as required |
| dev box (x86_64) | 1 | 2 | 1.0000 | 1.0000 | 0.209 | 0.0038 | failed as required |
| dev box (x86_64) | 4 | 0 | 1.0000 | 1.0000 | 0.112 | 0.0090 | failed as required |
| dev box (x86_64) | 4 | 1 | 1.0000 | 1.0000 | 0.110 | 0.0129 | failed as required |
| dev box (x86_64) | 4 | 2 | 1.0000 | 1.0000 | 0.192 | 0.0022 | failed as required |
| dev box (x86_64) | 16 | 0 | 1.0000 | 1.0000 | 0.172 | 0.0072 | failed as required |
| dev box (x86_64) | 16 | 1 | 1.0000 | 1.0000 | 0.075 | 0.0087 | failed as required |
| dev box (x86_64) | 16 | 2 | 1.0000 | 1.0000 | 0.192 | 0.0038 | failed as required |
| Thor (aarch64, gate env) | 1 | 0 | 1.0000 | 1.0000 | 0.088 | 0.0110 | failed as required |
| Thor (aarch64, gate env) | 1 | 1 | 1.0000 | 1.0000 | 0.091 | 0.0031 | failed as required |
| Thor (aarch64, gate env) | 1 | 2 | 1.0000 | 1.0000 | 0.104 | 0.0171 | failed as required |
| Thor (aarch64, gate env) | 4 | 0 | 1.0000 | 1.0000 | 0.128 | 0.0110 | failed as required |
| Thor (aarch64, gate env) | 4 | 1 | 1.0000 | 1.0000 | 0.116 | 0.0129 | failed as required |
| Thor (aarch64, gate env) | 4 | 2 | 1.0000 | 1.0000 | 0.101 | 0.0022 | failed as required |
| Thor (aarch64, gate env) | 16 | 0 | 1.0000 | 1.0000 | 0.128 | 0.0110 | failed as required |
| Thor (aarch64, gate env) | 16 | 1 | 1.0000 | 1.0000 | 0.082 | 0.0129 | failed as required |
| Thor (aarch64, gate env) | 16 | 2 | 1.0000 | 1.0000 | 0.109 | 0.0139 | failed as required |

**MAIN memorised in 18 of 18 cells; memory_zeros failed as required in 18 of 18.** MAIN ap2m is 1.0000 in every
cell, and its median centre error is 0.075-0.209 m,
far below the 2 m edge (the constant-lr snapshots sat at 0.55-2.57 m). memory_zeros ap2m is
0.0022-0.0171. The trajectories DO differ across
thread counts (seed 0's final 100-step loss mean on the dev box: 2.40 / 2.17 / 2.26 at 1 / 4 / 16 threads), yet the
decayed read-out agrees in every cell. That is the difference between measuring memorisation and reading one draw.

The same check applied to the constant-lr TIP runs (`raw/toy/toy_grid_judge_tip.json`) refuses every MAIN run, and
names A17 first -- including the two that would have passed the old test (dev box 16 threads, Thor 4 threads, both
1.0000). A read-out that is right by luck is refused for the reason it is unreliable.

**The test file itself** (blob `cb6812e16fe0…`, 23 tests = A17's 21 + the 2 red arms). Every run asserted `tanitad`
imported from its own tree:

| platform | environment | selection | result |
|---|---|---|---|
| Thor (aarch64) | gate env, `OMP_NUM_THREADS=4` (the gate's reading) | whole file | **23 passed in 96.74s** (`raw/toy/thor_pytest_toy_omp4.log`) |
| Thor | gate env, `OMP_NUM_THREADS=1` | `-k TOY` (4: the A14 one-frame toy + the 3 TOY tests) | 4 passed, 19 deselected in 88.42s |
| Thor | gate env, `OMP_NUM_THREADS=16` | `-k TOY` | 4 passed, 19 deselected in 141.88s |
| dev box (x86_64) | venv default (16 threads) | whole file | 23 passed in 222.55s (`raw/toy/pytest_devbox_default.log`) |
| dev box | `OMP_NUM_THREADS=4` (NEW-2's failing setting) | `-k TOY` | 4 passed, 19 deselected in 137.01s |
| dev box | `OMP_NUM_THREADS=1` | `-k TOY` | 4 passed, 19 deselected in 111.82s |

The Thor runs used `tree_toy` = tip 2ac0bfb + the A17 blobs + this file (`raw/toy/thor_import_check.log`: `tanitad`
from the tree, test blob `cb6812e1…`). ⚠️ **Runtime:** RED arm 1 adds one 1,200-step toy run and RED arm 2 adds two
20-step runs -- about +30-50 s per run of the file.

## 5. Files

- `code/a17_toy/stack/tests/test_g_box_overfit.py`: the full file for the landing (`code/tools/patch_toy_test.py`
  made it from A17's file).
- `code/tools/toy_repro.py` (the instrument), `code/tools/toy_grid_judge.py` (applies the TEST's own check functions,
  imported from the tree, to every grid run), `code/tools/write_addendum_landing.py` (the landing list: base = A17's NEW
  blob; package files listed explicitly), `code/tools/toy/*.sh` (the Thor and dev-box launchers).
- `raw/toy/`: every grid JSON and log, the judge's table, and the pytest logs.
