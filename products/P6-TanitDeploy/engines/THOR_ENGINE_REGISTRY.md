# Thor engine registry — provenance, hashes, rebuild recipes

`Created 2026-08-23 by the TanitAD_DeployFlyWheel to close census escalation E1:
every TensorRT engine existed on exactly ONE disk and is gitignored. Binaries
stay on the box — a .plan is device- and TRT-version-bound and is not a portable
artifact. What lives in git is the RECIPE, the DESCRIPTOR and the HASH.`

⚠️ **A recipe is durable; a copy is not.** But a recipe is only durable if its
*inputs* are: these engines rebuild from checkpoints in `thor:~/models/`, and
those must themselves be on HF or the recipe rebuilds nothing. Tracked as
backlog **D-02**.

## Build environment (all MEASURED 2026-08-23, `ssh tanitad-thor`)

| fact | value |
|---|---|
| host | `thor6`, Jetson Thor, sm_110, aarch64, `Linux-6.8.12-tegra` |
| TensorRT | **10.13.3.9+cuda13.0** (`libnvinfer-*` apt packages) |
| builder binary | `/usr/src/tensorrt/bin/trtexec` |
| ⛔ python bindings | **NOT importable** from `tanitad-edge` or `tanitad-train` — `trtexec` CLI is the only build route |
| torch | 2.13.0+cu130 (identical in both venvs) |
| disk | `/dev/nvme0n1p1` 937 G, 26 % used |

⚠️ **An engine built under a different TensorRT or on a different SM will not
load.** Every rebuild re-stamps this table; a mismatch is the first thing to
check when a `.plan` refuses to deserialise.

## Registry

### ⭐ DEPLOYABLE

| engine | sha256 | source | profile | intent | verified |
|---|---|---|---|---|---|
| `~/trt_deploy/predictor_v1_intent_dyn1-9_fp16.plan` (174.7 MB) | `f8dc0cc60b94bb16b2c53dac4ac400938ec6f9fce24af1553bb32052e0f475b2` | `flagship-v1-speedjerk` **step 29999**, STRICT load | dynamic **1..9** (opt 9) | ✅ 256 | rel-err **3.666e-4** @b1 / **3.673e-4** @b9; `intent_is_live` **0.0522**; `row_independence` 6.52e-4 |
| `~/trt_deploy/predictor_v5f_intent_dyn1-9_fp16.plan` (174.7 MB) | `8f4e51d12c393c8e056ae974f2fd9b0a33d315c7f4efa8ab706e8861d9d85413` | `v5f` **step 1000**, 176×624 deployed geometry | dynamic **1..9** | ✅ 256 | rel-err 8.49e-4 / 6.06e-4; `intent_is_live` 0.0470 |

⛔ **Deployable ≠ shipped.** Both still sit behind the four-family gate, whose
`F_LONGITUDINAL` fires and whose materiality threshold is unratified
(backlog **D-01**).

### ⛔ CONTROL / SUPERSEDED — retained for provenance, never for use

| engine | sha256 | why not |
|---|---|---|
| `~/trt_deploy/predictor_v1_dyn1-9_fp16.plan` | `ca5a6e1140c90a3e2e2ee15dd662acac11af3ed5c6063a11d4c7bb7ee37c178f` | **no intent input** — kept ONLY as the control arm for retraction R-2026-08-03-e. An intent-less engine computes the UNCONDITIONED prediction: **3.81 m** mean tactical regret @K20, 60 % of selections flipped |
| `~/trt_deploy/predictor_v5f_dyn1-9_fp16.plan` | `01018511df25ea91d1f6033bef57fdcffc4e8f049a8ba71feb61a893fbff7e27` | intent-less; superseded by the row above |
| `~/trt_c3/pred_dyn_fp16.plan` | `fd8a408088c37388a4c0203dc91984ea71735f84fba43cb13f30bfb597ac783d` | real weights but **dynamic 1..8** — cannot serve the 9-candidate fan (`n_maneuvers = 9`) |
| `~/trt/predictor_fp16.plan`, `gate_*.plan`, `fixed_*.plan` (2.6 GB) | — | ⛔ **built from a RANDOMLY-INITIALISED model** — the building script contains no `torch.load` and no `load_state_dict`. Batch-1 static. Any number taken from these describes noise |
| `~/trt_c2/*` (4.8 GB) | — | superseded; see the name/content warning below |

⚠️ **NAMES THAT ASSERT DISTINCTIONS THE BYTES DO NOT CARRY.** In `~/trt_c2`,
`v1_pred_b1_fp16.onnx`, `v1_pred_b1_fp32.onnx`, `v1_256x256_b1.onnx`,
`fp_17_0.onnx` and `fp_17_1.onnx` are **one byte-identical file under five
names** (md5 `8b6efc611e80549b487faea8ba98faac`), and `fp_18_0/1.onnx` are
another such pair. This is correct behaviour — fp16 is a `trtexec` **build**
flag, not an export property — but a reader who trusts the filenames will
believe an fp16 export exists that does not. **Verify by content.**

## Rebuild recipe

```bash
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=$HOME/TanitAD/stack:/usr/lib/python3.12/dist-packages
python $HOME/TanitAD/stack/scripts/build_predictor_trt.py \
  --ckpt $HOME/models/flagship-v1-speedjerk/ckpt.pt \
  --out  $HOME/trt_deploy/predictor_v1_intent_dyn1-9_fp16 \
  --max-batch 9 --fp16 --intent-dim 256
```

~38–40 s per engine, ~174 MB out. The builder
(`repo:stack/scripts/build_predictor_trt.py`) is checked in and exports with
`set_fastpath_enabled(False)`, opset 17, dynamic min/opt/max shapes, then
deserialises and verifies.

⛔ **`--intent-dim` is NOT optional for the flagship.** ⛔ **v5f additionally needs
`--v2-subframe 176x624`** — without it the STRICT load *refuses* the checkpoint
(`encoder.pos` 429 vs 256) rather than quietly resizing. That refusal is the
geometry guard working.

⛔ **And the caller must batch**: `propose_and_score(..., batch_fan=True)`. A
batch-9 engine driven by a serialised loop measures **worse** than a batch-1
engine (272.8 vs 265.7 ms) — shipping the engine without the caller is a
regression.

## Files rescued into git alongside this registry

| file | what |
|---|---|
| `descriptor_predictor_v1_intent_dyn1-9_fp16.json` | full descriptor: ckpt, step, shapes, exact `trtexec` command, verify block |
| `descriptor_predictor_v5f_intent_dyn1-9_fp16.json` | same, v5f |
| `_THOR_MANIFEST_rescued_2026-08-23.md` | the original `thor:~/trt_deploy/MANIFEST.md`, verbatim |

## Verification

`sha256sum ~/trt_deploy/*.plan ~/trt_c3/*.plan` on Thor must reproduce the hashes
above. A mismatch means an engine was rebuilt — re-stamp the row and the build
environment table; do not assume the recipe still describes the file.
