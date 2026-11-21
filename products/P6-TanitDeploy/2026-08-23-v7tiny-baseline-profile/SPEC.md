# E-DEPLOY-1 — v7-tiny (champ30k) baseline profile on Thor

`Work package per TANITAD_PROGRAMME.md §3. Owner: TanitAD_DeployFlyWheel.
Written 2026-08-23 BEFORE any measurement was taken. Both outcomes are stated
below and neither is the "hoped-for" one.`

## What & why

P6 TanitDeploy has **no measured baseline for the current model line**. Every
latency/memory number the Deployment & Optimization folder holds was taken in
the *pod era* on *v1/v5f* checkpoints (2026-08-02/03). The successor line is
v6/v7-tiny, the fleet is Thor + the dev-box RTX 4060, and `champ30k` — the
first collapse-free trunk (19,337,289 params, 30k steps, finished 2026-08-23
12:55 Thor-local) — is the first checkpoint of that line worth profiling.

This work package establishes **the template for every later TanitDeploy run**:
an arm grid, admissible probes only, a numerical-fidelity control paired with
every dtype change, and a raw JSON that is the quotable layer.

⛔ **This is a PROFILE, not a deployment.** Charter §4: *a quantization without a
paired eval is not a deployment*. A dtype arm here ships with a **numerical
deviation** measurement only; the four-family paired accuracy eval is a separate
work package and is explicitly OUT OF SCOPE. No arm in this package may be
described as "deployed", and no speedup may be quoted without its deviation
number attached.

## Target & environment (all MEASURED 2026-08-23, not inherited)

| fact | value | how verified |
|---|---|---|
| host | `thor6`, Jetson Thor, aarch64, `Linux-6.8.12-tegra` | `ssh tanitad-thor` |
| GPU | `NVIDIA Thor`, compute capability 11.0 | `torch.cuda.get_device_capability` |
| torch | `2.13.0+cu130`, CUDA build 13.0, `cuda_available=True` | `deploy_env_probe.py` |
| conv2d on CUDA | **works** (finite output, abs_mean 0.475) | real conv2d, not `import torch` |
| fp16 / bf16 matmul | both work, finite | real matmul |
| Triton | **3.7.1 PRESENT** (unlike the dev box) | import |
| TensorRT | `10.13.3.9+cuda13.0` system pkgs + `/usr/src/tensorrt/bin/trtexec`; **python bindings NOT importable from either venv** | `dpkg -l`, import probe |
| venvs | `~/venvs/tanitad-edge` (inference) and `~/venvs/tanitad-train` — **identical torch**; `onnx`/`onnxruntime` only in `-edge` | import probe both |
| GPU idle at launch | no `python`/`train_*` process at all | full `ps -eo args` grep, not a top-N sample |

## The subject

`~/v7tiny/champ30k/ckpt.pt` (133,268,876 B) + `config.json`, produced by
`~/champ.sh` → `scripts/train_v6_staged.py --stage S-W`, 19,337,289 params,
`elapsed_s` 14774.7, `summary.json: {"done": true}`, `gate_verdict`
`INCONCLUSIVE`.

⚠️ **`gate_verdict: INCONCLUSIVE` is recorded here as a fact about the subject.**
It bears on whether this checkpoint is worth *shipping* — a question owned by the
Master Mind / EvalFlyWheel, not by this package. Profiling an INCONCLUSIVE
checkpoint is still correct: the profile characterises the **architecture's
inference cost**, which is a property of the config, not of the weights' quality.

## Hypotheses (pre-registered, both outcomes)

| id | hypothesis | outcome A | outcome B |
|---|---|---|---|
| **H-DEPLOY-2** | Thor's batch-8 saturation (measured on the v6 *trainer*) also holds for the v7-tiny **inference** tick — throughput ticks/s is flat or sublinear from batch 4→8 | **SUPPORTED**: windows/s at b8 ≤ 1.3× that at b4 ⇒ the deploy playbook keeps small batches | **REFUTED**: windows/s at b8 > 1.3× b4 ⇒ the saturation figure is a *training* property and must not be quoted for inference; the deploy batch ceiling is re-opened |
| **H-DEPLOY-3** | fp16 autocast gives a material latency win on this model at batch 1 | **SUPPORTED**: median tick latency fp16 ≤ 0.85 × fp32 | **REFUTED**: > 0.85 × ⇒ this model is not matmul-bound at b1; the first optimisation lever is elsewhere (kernel-launch overhead ⇒ CUDA graphs), and an fp16 deployment would buy memory only |
| **H-DEPLOY-4** | Reduced precision is numerically safe on this stack at inference — no NaN/Inf, and plan-waypoint deviation stays small | **SUPPORTED**: all outputs finite AND max abs deviation on `plan` ≤ 1e-2 m ⇒ dtype arms are eligible for a paired four-family eval | **REFUTED**: any non-finite, or deviation > 1e-2 m ⇒ this stack needs per-module dtype policy before any quantization ladder is attempted; report which tensor broke |

⚠️ H-DEPLOY-4's threshold is a **screening** threshold on synthetic input, not an
accuracy criterion. Passing it does NOT license an accuracy claim — it licenses
*running the paired eval*. Failing it stops the ladder.

## Method

- **Arms**: dtype ∈ {fp32, fp16-autocast, bf16-autocast} × batch ∈ {1, 4, 8}.
  9 arms, one process, one model instance moved between dtypes via autocast
  (weights stay fp32 — this is the autocast ladder rung, not a weight cast).
- **Input**: `synthetic_train_batch(stack, batch=B, seed=0)` — the trainer's own
  fully-shaped stand-in, so shapes are the real ones. **The same seed for every
  arm**, so a deviation between arms is a dtype effect and nothing else.
- **Forward**: `V6Stack.forward(frames, actions, v0)` under `torch.no_grad()` and
  `model.eval()` — one call = **one inference tick** = one hierarchy pass over a
  6-frame window.
- **Timing**: `torch.cuda.synchronize()` on both sides, `time.perf_counter`,
  **20 warmup + 50 timed** iterations per arm; report median and p90, and print
  `n` on every panel.
- **Memory**: ⛔ **`torch.cuda.max_memory_allocated()` ONLY** — reset per arm.
  `free`, `tegrastats`, `mem_get_info` and `VmRSS` are inadmissible on Thor
  (CLAUDE.md: they lie in both directions, measured 2026-08-03).

## Tests / controls that must read a known value

A panel with no control cannot fail, and a result from an instrument that cannot
fail is not evidence.

1. **TIMER CONTROL** — a fixed `4096×4096` fp32 matmul is timed with the same
   harness. It must land in a plausible band (>0.1 ms, <1 s) and its fp16
   counterpart must be **faster**. If the timer cannot resolve a difference on an
   op that is *definitionally* matmul-bound, no latency claim in this file is
   admissible.
2. **MEMORY CONTROL** — allocate a known `256 MiB` tensor and confirm
   `max_memory_allocated()` rises by ≥ 256 MiB. A probe that does not move for a
   known allocation is disqualified (this is exactly how the Thor `free` trap
   was caught).
3. **DETERMINISM CONTROL** — the same arm run twice with the same seed must
   produce bit-identical `plan` output in fp32. If it does not, the deviation
   numbers in H-DEPLOY-4 measure noise, not dtype.
4. **PARAM CONTROL** — the instantiated model's parameter count must equal the
   `19,337,289` recorded in `summary.json`. A mismatch means the config was not
   faithfully reconstructed and **the whole run is void**.
5. **LOAD CONTROL** — `load_state_dict` must report **zero** missing and zero
   unexpected keys. Profiling a randomly-initialised model would produce a
   perfectly valid-looking latency table that describes nothing we have.

## Success criteria

The package is DONE when `raw/thor_v7tiny_profile.json` exists containing all 9
arms plus all 5 controls **passing**, and `RESULT.md` states a verdict for
H-DEPLOY-2/3/4 with each number carrying its artifact path and evidence class.

**A failed control voids the run** — the correct output in that case is a
RESULT.md saying so, not a table of numbers with a caveat.

## Out of scope (named so it is not silently skipped)

- Paired four-family accuracy eval of any dtype arm → next work package.
- ONNX export / TensorRT engine build for v7-tiny → next work package
  (`trtexec` exists; python bindings do not — that shapes the design).
- INT8/FP8 → blocked on the paired-eval harness, by charter.
- The dev-box RTX 4060 comparison arm → separate, and must not be pooled with
  Thor numbers.
