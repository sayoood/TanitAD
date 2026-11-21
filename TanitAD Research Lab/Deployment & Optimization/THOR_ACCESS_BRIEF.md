# Thor access brief — for the Production & Optimization agent

**Everything needed to work on the Jetson Thor independently. All facts MEASURED 2026-08-02.**

---

## 1. Connect

```bash
ssh tanitad-thor          # alias already in ~/.ssh/config on the dev box
```

| | |
|---|---|
| host | `192.168.178.93` (local LAN, WiFi) |
| user | `nvidia` |
| key | `~/.ssh/tanitad_pod` |
| hostname | `thor6` |
| sudo | password is the Jetson factory default; ⚠️ **not interactive over ssh** — pipe it: `printf 'PASSWORD\n' \| sudo -S -p '' <cmd>`. Ask the PI for it; do not hardcode it in committed scripts |

⚠️ **On Windows use native OpenSSH** (`/c/Windows/System32/OpenSSH/ssh.exe`), not MSYS ssh — MSYS
deadlocks under subprocess pipes against busy hosts and looks like an outage.

⚠️ **WiFi drops happen.** A `Connection timed out` is usually transient — retry before concluding
the board is down. `uptime` distinguishes a drop from a reboot.

## 2. ⛔ THE TWO-VENV RULE (PI directive, non-negotiable)

| venv | use case |
|---|---|
| `~/venvs/tanitad-edge` | **use case 1** — optimised inference, open-loop, closed-loop (AlpaSim), all deployment optimisation |
| `~/venvs/tanitad-train` | **use case 2** — training |

**Never mix them; never install into the system python.** Always name the interpreter explicitly:
`~/venvs/tanitad-edge/bin/python`, never bare `python3`.

Rationale: edge work pulls TensorRT/ONNX/compiled stacks that conflict with training deps, and a
broken shared env would take both workstreams down on a device with no fast rebuild path.

## 3. Hardware and stack (MEASURED)

| | |
|---|---|
| GPU | **NVIDIA Thor, `sm_110` (Blackwell)**, 20 SMs |
| memory | **122 GB UNIFIED** (CPU+GPU share it) — 131.9 GB reported by torch |
| disk | 937 GB, ~880 free |
| OS | L4T **R38.4.0** (JetPack 7 line), aarch64, Ubuntu 24.04, python 3.12.3 |
| torch | **2.13.0+cu130**, `cuda_available True` (edge venv) |
| TensorRT | **10.13.3.9** — `trtexec` at `/usr/src/tensorrt/bin/trtexec` |
| ONNX | `onnx 1.22.0`, `onnxruntime 1.28.0` (**CPU only** — no aarch64 GPU wheel exists; TRT is the GPU path) |
| uv | 0.12.1 (in the edge venv) |
| docker | usable **without sudo**; compose v2.40.3 |

⚠️ **TensorRT python bindings are SYSTEM packages** — there is no aarch64 TRT wheel matching an L4T
runtime. Reach them from the venv with:

```bash
PYTHONPATH=/usr/lib/python3.12/dist-packages ~/venvs/tanitad-edge/bin/python ...
```

⚠️ **Unified memory**: `torch.cuda.max_memory_allocated()` reported 1.2 GB while system RSS was
**11.5 GB**. Size deployments from `/proc/meminfo`, not the allocator.

## 4. What is already on the box

| path | contents |
|---|---|
| `~/TanitAD` | full repo clone (public GitHub, current) |
| `~/models/rollout-recovery` | **3.0 GB** — RR-20, RR-CTL, refc-base-e1f-junction (public HF) |
| `~/models/refc-base`, `refc-xl`, `refc-base-e1b-clsft`, `flagship-v4.2b` | pulling — ⚠️ these repos are **GATED**; pass `token=open('~/.hftok').read().strip()` or you get a 401 |
| `~/.hftok` | HF token, mode 600 |
| `~/trt/*.plan` | built TensorRT engines |
| `~/alpasim` | AlpaSim clone, protos compiled, wizard runnable |
| `~/thor_*.py`, `~/thor_*.json` | the profiling harnesses and their raw results |

⛔ **No validation data yet.** The val caches live on RunPod pods (currently stopped). Any
four-family accuracy work needs them moved first — this is the single biggest gap.

## 5. Run the stack

```bash
cd ~/TanitAD/stack
PYTHONPATH=$HOME/TanitAD/stack:$HOME/TanitAD/taniteval ~/venvs/tanitad-edge/bin/python -c \
  "from tanitad.models.fourbrain import WorldModel; print('ok')"
```

⚠️ **Geometry must be applied through the trainer's own seam** or the encoder raises — its
positional embedding is sized for the declared frame:

```python
from train_flagship_v4 import resolve_v2_frames    # scripts/ must be on sys.path
ns = SimpleNamespace(frame_h=256, frame_w=640, frame_hfov=120.0,
                     projection='cylindrical', v2_subframe='176x624', f_ref=None)
resolve_v2_frames(ns, cfg, label='yourjob')
cfg.speed_input = True                                    # v5 trains the speed-input trunk
cfg.predictor = dataclasses.replace(cfg.predictor, action_dim=3)
```

## 6. ⛔ ONNX export — the mandatory line

```python
torch.backends.mha.set_fastpath_enabled(False)   # BEFORE export. Not optional.
```

`nn.MultiheadAttention` fuses to `aten::_native_multi_head_attention`: opset 18 **fails loudly**,
opset 17 **exports a silently wrong graph** (rel-err **0.726**). With the fastpath off, both opsets
give **7.9e-7**. It changes eager output by 5.1e-7, so leaving it on permanently is safe.

🔴 This **retracts** the 2026-07-08 "no unexportable ops / MHA fine" finding **for torch 2.13**.
**Re-verify ONNX parity per torch version — never inherit it.**

## 7. Measurement discipline

- warmup ≥10 iters, then **`torch.cuda.synchronize()` around every timed region** (CUDA is async;
  unsynchronised timing measures kernel *launch*)
- report **p50 AND p99**, never a mean — the 100 ms budget is a deadline, so the tail is the spec
- ⛔ **an accuracy delta beside every speed delta**, and accuracy means the **four families**
  (longitudinal, lateral, tactical, strategic), never ADE alone
- validate engines **in ONE process with ONE model** — a comparison across two random inits is not
  a comparison (a gate "failed" identically at fp32 and fp16 for exactly this reason; identical
  error across precisions is the signature of a wiring/test bug)
- bind TRT tensors **by name** (`states`, `actions`, `z_next`), never by index

## 8. Current baseline to beat

| stage | best measured |
|---|---|
| encoder | **27.8 ms** (bf16 autocast, 6.76× over fp32) |
| predictor 1-step | **1.168 ms** (TRT-fp16; 3.62× eager, 2.93× over the free CUDA graph) |
| **full tick** | **≈51.2 ms** = **5.33×** vs the 272.56 ms fp32 baseline, **51 %** of the 100 ms budget |

⭐ **Per-stage, never global**: bf16 is **6.76× on the encoder** and **0.86× — a LOSS — on the
predictor** (tensors too small to repay autocast). One blanket `.half()` costs ~14 %.

Thermals: GPU ~1976 mW, junction 61.3 → 61.9 °C over 39 min, **no throttling**.

## 9. Where the work is queued

**`Production & Optimization/THOR_DEPLOYMENT_RUNBOOK.md`** — full procedure, 10 learnings, and the
**O1–O14 backlog, each with its falsifier** so experiments can run independently. P0 items:
four-family gate on the TRT engine (needs val data), encoder engine, measured end-to-end tick.

Raw artifacts: `Architecture & Inference/Implementation/incoming/2026-08-02-thor-deployment-profile/`

## 10. Do-not

- ⛔ don't install into the system python or cross the two venvs
- ⛔ don't run training on the edge venv (or vice versa)
- ⛔ don't quote a speedup without its accuracy delta
- ⛔ don't trust exit codes or package counts as completion evidence — read the log line that says
  it finished (~2 h were lost polling `dpkg -l` mid-transaction while the install had completed)
- ⛔ don't publish to HF or delete anything without the PI
