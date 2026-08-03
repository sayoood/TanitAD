# INTAKE — Thor P6: the optimisation + precision gate ON REAL TRAINED WEIGHTS (2026-08-03)

**Research note:** `../../Research/2026-08-03-thor-real-weight-precision-gate.md`
**Runbook annotation:** `../../THOR_DEPLOYMENT_RUNBOOK.md` § "ANNOTATION 2026-08-03 — RE-RUN ON REAL
TRAINED WEIGHTS" (inserted before §4).
**Host:** `tanitad-thor` (thor6), aarch64 sm_110, torch 2.13.0+cu130, TRT 10.13.3.9,
`~/venvs/tanitad-edge`, `OMP_NUM_THREADS=6`. Thor repo sha `4954544`. Cost **$0**.

## Why

Every number in the runbook's §1/§2/§3 was measured on a **randomly-initialised** `WorldModel` fed
`torch.randn` — no `torch.load` anywhere in the five Thor scripts. Latency survives that; numerics do
not. This package re-runs the whole thing on **real checkpoints and real held-out frames**, and
closes backlog item **O1** (the four-family deployment gate).

## Harnesses (each is standalone; each banks JSON incrementally)

| script | what it measures | runtime |
|---|---|---|
| `thor_c2_real_weights.py` | Q1 latency (real + random control, both geometries) · Q2 TRT fp16/fp32 numerics on **real encoded states + real expert actions** · Q3 encoder bf16 on **real frames** · Q4 MHA fastpath × opset 17/18 on **real weights** | ~9 min, 6 engine builds |
| `thor_c2b_random_control.py` | the random-weight control stage 4 dropped on a batch mismatch — random vs real weights on **identical `torch.randn` inputs**, so weights are the only variable | ~90 s |
| `thor_c3_four_family_gate.py` | ⭐ **O1** — fp32 eager vs (bf16 encoder + TRT-fp16 dynamic-batch predictor), identical windows, ADE + four families + paired episode-cluster bootstrap, with a negative control before any score | ~6 min |
| `thor_c4_rescore_corrected.py` | 0-GPU rescore of the banked windows through the **corrected** `four_families`, reporting pre-fix and corrected side by side | ~20 s |
| `thor_c5_encoder_export.py` | **O2-pre** — does the REAL encoder export at the deployed 176x624 after the readout fix, with ORT parity, and is v1's tiling path untouched | ~2 min |

## Raw results

| file | headline |
|---|---|
| `thor_c2_real_weights.json` | latency within **0.1–2.1 %** of a same-session random control ⇒ **weight-independent**; TRT-fp16 b1 **1.17676 ms** vs published 1.168 (+0.7 %); fastpath rel-err **5.56e-08** at all 4 cells |
| `thor_c2b_random_control.json` | replicates the published random-weight row (**1.193e-3 → 1.652e-3, growth 1.385×** vs published 1.41e-3 → 1.80e-3, 1.3×); real weights on the same input: growth **2.056×** |
| `thor_c3_four_family_gate.json` | paired **ΔADE 0–2 s = −0.0004 m** [−0.0009, +0.0001]; tactical agreement **0.9942**; strategic **1.0000**; LONGITUDINAL separates on `speed_bias` / `accel_mae` |
| `thor_c4_rescore_corrected.json` | the dt negative control (**12.4565 vs 62.9789 m/s = 5.0559×**) and every corrected level |
| `thor_c5_encoder_export.json` | ✅ **O2-pre CLOSED** — the encoder exports at 176×624 in 1.7 s, no adaptive pool in the graph, ORT rel-err **1.51e-05 / 1.77e-05** on real v5f weights, at both fastpath settings |

## Code changes staged in the repo (NOT on Thor only)

| path | change |
|---|---|
| `taniteval/taniteval/four_families.py` | **dt is now derived, never assumed.** `_seq_geometry(wp, dt)`; `longitudinal/lateral(..., dt)`; new **`infer_dt(win)`** reading the window's `wp_steps`×`dt_s` contract with a provenance string and no silent guessing; `all_families(..., prefer_dense=True)` uses the true 10 Hz dense path when present; `MIN_DS_M` → **`MIN_DS_MPS = 0.5 m/s`** so the gate scales with cadence (identical at 10 Hz); every family carries `dt_s` and `all_families` emits a `_grid` provenance block |
| `taniteval/tests/test_four_families_dt.py` | **NEW — 12 tests, all passing.** A constant-velocity path must report its own speed on any grid; the ×5 / ×25 factors; `infer_dt` never guesses silently; heading/curvature/positions stay dt-invariant; the `min_ds` gate scales; headings are degrees |
| `stack/tanitad/models/readout.py` | **O2-pre.** `AdaptiveAvgPool2d` on a non-tiling grid is unexportable and blocked the encoder at 176×624. New `_adaptive_avg_matrix(n_in, n_out)` materialises PyTorch's own bins as a constant averaging matrix; the non-tiling route becomes two matmuls. Buffers are **`persistent=False`** so no state_dict key is added. **The tiling route (v1 @256px) is not touched at all.** |
| `stack/tests/test_readout_onnx_pool.py` | **NEW — 13 tests, all passing.** Matches `F.adaptive_avg_pool2d` at 11×39 / 11×42 / 13×41; pins that the bins **OVERLAP** (11→4 = 3/4/4/3, not a partition — the first draft asserted the partition and failed); the tiling path is **bit-identical** to `AvgPool2d`; the matrices stay out of the state_dict; and the module round-trips through ONNX + onnxruntime with parity |

⚠️ **The corrected `four_families.py` was also pushed to `thor:~/TanitAD/taniteval/taniteval/` so the
rescore ran against the same code that is staged here.** Thor's checkout is otherwise at `4954544`
and drifts; re-sync before any further run there.

## Verification performed (not exit codes — artifacts)

* Both checkpoints loaded with **STRICT** `load_state_dict` (v5f: 0 missing / 0 unexpected) and the
  step field read as **29999** / **1000**.
* **Val integrity by LOADING every clip, not by size.** `ep_00028.pt` is truncated (92.3 MB against
  the cohort's ~117 MB) and unreadable; it is **named and excluded**, and the run is stamped
  `decision_grade_absolute: False`. It cannot be re-pulled — the HF transfer repo
  `Sayood/tanitad-transfer-2026-08` carries only the 256×640cyl v2ep val. **The paired delta is on
  identical windows in both arms and is unaffected.**
* **Negative control before any score** in `thor_c3`: the engine differs from eager (5.9e-4) **and**
  responds to its inputs (zeroing the actions moves the output 0.232 relative). A wired-through or
  input-ignoring engine would have made every family delta 0.
* **Negative control for the instrument** in `thor_c4`: the derived speed is checked against the
  ego's own recorded `poses[:,3]`.
* Parity: the CLEAN `physicalai-val-0c5f7dac3b11` split, skip-hash `f09e44db`, and **each arm scored
  at its own trained raster** — v1 at 256×256, v5f never scored for accuracy.
* TRT bindings bound **by name** (`states`, `actions`, `z_next`), never by index.

## Escalations — these need an owner, they are not filed and forgotten

1. ⭐ **PI DECISION.** The O1 gate's LONGITUDINAL falsifier fired at effect sizes of **0.12–0.58 %**.
   A **materiality threshold must be pre-registered beside the separation test** (proposed: 1 % of
   the fp32 level per metric). Until it is, the gate as written says **DO NOT SHIP** and is reported
   that way. This is the single blocked item.
2. 🔴 **To the owner of the runbook §3 retraction:** the MHA-fastpath mechanism has now failed to
   reproduce at **10 cells across two weight distributions**. The 2026-07-08 "ONNX-clean" retraction
   rests on it and should be revisited.
3. 🔴 **To every consumer of `four_families`:** absolute rate numbers in the hub are wrong by the
   §6 factors — notably the 2026-08-02 REF-C Thor panel (`speed_mae 3.0609 → 0.6122 m/s`). Ranks and
   paired deltas are unaffected. A sweep of published four-family tables is a work item.
4. **`predictor_fp16.plan` as shipped is batch-1 static** — confirmed on real weights at 242 % of
   budget serialised vs 56 % batched. The dynamic 1–8 engine built by `thor_c3` is the template.
5. ⛔ **Corpus durability, again:** the 256 px val cache exists on exactly one disk and one clip
   arrived truncated. Runbook §11 says push it to HF as a dataset the day it is built.
6. ✅ **O2 is now UNBLOCKED and nobody owns it.** O2-pre is closed (the encoder exports at 176×624
   with ORT parity ~1.8e-05), so the next step is to build the encoder TensorRT engine and compare
   it against bf16 autocast at **30.23 ms**. Its falsifier is already written in the backlog: engine
   ≤ bf16 ⇒ keep autocast and close the item.

## Reproduce

```bash
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export OMP_NUM_THREADS=6          # ⛔ not optional — torch spawns ~113 threads/process
export PYTHONPATH=$HOME/TanitAD/stack:$HOME/TanitAD/stack/scripts:$HOME/TanitAD/taniteval:/usr/lib/python3.12/dist-packages
python thor_c2_real_weights.py    # -> ~/thor_c2_real_weights.json     (banks per stage)
python thor_c2b_random_control.py # -> ~/thor_c2b_random_control.json
python thor_c3_four_family_gate.py# -> ~/thor_c3_four_family_gate.json + ~/thor_c3_windows.pt
python thor_c4_rescore_corrected.py   # 0 GPU; needs the corrected four_families.py on the box
python thor_c5_encoder_export.py  # needs the fixed stack/tanitad/models/readout.py on the box
```
⚠️ `thor_c4` and `thor_c5` read the FIXED modules. Sync `taniteval/taniteval/four_families.py` and
`stack/tanitad/models/readout.py` to Thor first and clear `__pycache__`, or they will silently run
the old code — Thor's checkout is at `4954544` and drifts.
