
## Amendment 1 (A1-2026-10-27): the ego vector takes the data's own width (22 dims), before any SPEEDREAD number

Written 2026-10-27 ~23:20 by the Master Mind.

**State when written.**
* The pull is done: 4,572 TRAIN clips, each md5-verified (`ZZSPEEDREAD-PULL-DONE`, 23:08).
* The §8 run has stopped twice in its first stage, `extract`. Neither stop wrote a number:
  1. `No module named taniteval`. This was a run-mechanics fix only: a wrapper puts the exported tree's `taniteval/` on `sys.path` under
     `-I`. The instrument did not change.
  2. On the first batch of 128 TRAIN windows: `ValueError: could not broadcast input array from shape (128,22) into shape (128,23)`.
* **What that second stop opened.** One batch of 128 TRAIN windows was decoded into the read-out's input vector, and then the stage
  raised.
  * Its only output is the pre-allocated scene memmap, partly filled. It is moved aside unread (`speedread/stop2/`).
  * No eval batch was opened. Nothing was summarised, fitted, selected or printed, and no target value was looked at.

**The defect.** §3 counted the ego vector as *"23 dims (p10f 8, nav 5, limit 5, limit_ok 1, frame_ok 4)"*, and the instrument hard-coded
23. On the real cache, `limit` is **4 wide**, so the vector T reads is 22 dims. The 5 was a wrong count of a key the PREREG selects by name
(`READ_KEYS` and `INFERENCE_INPUT_KEYS`), not a design choice. The synthetic self-test built a 5-wide `limit` and so could not see it.

### 1. What changes

* **§3 Ego** reads: *"Ego: the whole `limit` at its data width, 22 dims on the real cache (p10f 8, nav 5, limit 4, limit_ok 1, frame_ok 4)."*
  * The instrument allocates the ego array at the first batch's own width.
  * It records `d_ego` in `extract.json`.
* **§3 MLP** reads *"(512 + d_ego) → 256 → 256 → 5"*, which is 534 on the real cache. The code already sized the MLP from the design
  matrix (`mlp_build(X.shape[1], …)`). Only its docstring said 535.
* **§9** reads 27 tests. The two new cases are `test_extract_sizes_ego_from_the_data[4]` and `[5]`.
  * Each runs `cmd_extract` end to end on a stub data layer with a `limit` of that width.
  * Each asserts that the stored ego equals `window_vectors` and that `d_ego` is recorded.

### 2. What does not change

The keys read, the features, the scene vector, the PCA, the λ grid, the MLP's configuration, the folds, the PRIMARY rule, every control and
§5's decision table stay as written. No entry of `limit` is dropped and none is added. The amendment changes how the vector's length is
counted, not what goes into it.

### 3. Proofs, before the run restarts

| check | result | file |
|---|---|---|
| self-test, fixed instrument | **27 passed** | `raw/2026-10-27-speedread-a1/pytest_patched.txt` |
| the new test against the promoted instrument (`d5545751…`) | **2 failed**: `[4]` on the broadcast error, `[5]` on the missing `d_ego` | `mutation_old_instrument.txt` |
| `mutate_speedread.py`, fixed instrument | **19 / 19 KILLED**, 0 survived, 0 not applied, baseline green before and after | `mutation_results_a1.json` |

* The instrument's sha256 changes from `d5545751…` to **`fd5ce0bea37c3e0e608144e041f54a337fd4378b38fb312df65787bc864be4fe`**.
* The run script pins the new sha256 before it restarts.
