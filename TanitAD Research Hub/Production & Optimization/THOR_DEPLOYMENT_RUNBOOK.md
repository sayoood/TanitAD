# Thor deployment runbook + optimisation backlog (2026-08-02)

**Everything here is MEASURED on `thor6`** — NVIDIA Thor (Blackwell **sm_110**), aarch64, L4T
R38.4.0, torch 2.13.0+cu130, TensorRT **10.13.3.9**, at v5f's deployed geometry **176×624 / 117°
cylindrical**, 263.58 M params, 100 ms (10 Hz) budget.

---

## 1. THE RESULT

| configuration | tick p50 | vs budget |
|---|---|---|
| fp32 eager (baseline) | **272.56 ms** | ⛔ 273 % |
| bf16 encoder + CUDA-graph predictor | 98.63 ms | 98.6 % |
| ⭐ **bf16 encoder + TRT-fp16 predictor** | **≈51.2 ms** | ✅ **51 %** |
| | ⭐⭐ **5.33×** | ~2× headroom |

**Per stage:**

| stage | eager | optimised | gain |
|---|---|---|---|
| encoder | 187.8 ms fp32 | **27.8 ms** bf16 autocast | **6.76×** |
| predictor (1 step) | 4.23 ms | **1.168 ms** TRT-fp16 | **3.62×** |
| 20-step roll | 81.7 ms | **23.4 ms** (20 × 1.168) | 3.49× |

ⓘ The 5.33× lands almost exactly on the A40 stream's **5.35×** (138 → 18.75 ms) — independent
replication of the achievable ceiling on different silicon with a different lever mix.

## 2. ✅ PRECISION VALIDATION — "output without opt must equal output with opt"

**Single process, one model, export → build → compare.** Error measured at 1 step **and** after a
20-step recursive roll (the operating condition — E-CR measured that rollout error compounds
3.50 → 80.77, so a 1-step check alone would be negligent):

| engine | rel-err 1 step | rel-err after 20 steps | growth | verdict |
|---|---|---|---|---|
| **TRT fp32** | **3.05e-4** | **4.39e-4** | 1.4× | ✅ **PASS** |
| **TRT fp16** | **1.41e-3** | **1.80e-3** | 1.3× | ✅ **PASS** |
| CUDA graph | **0.0** | **0.0** | — | ✅ bit-exact |

⛔ **DO NOT READ THIS AS "ERROR DOES NOT COMPOUND."** *(Corrected 2026-08-02 by adversarial
verification; the earlier ⭐ headline here made exactly that claim and it is withdrawn.)*

Every row above was measured on a **randomly-initialised model fed `torch.randn`** — no `torch.load`,
no `load_state_dict`, in any of the five Thor scripts. **Quantisation error is a function of the
TRAINED weight and activation distribution**; outlier channels are the entire difficulty, and a
random network has none. A rel-err on noise says almost nothing about a rel-err on the flagship.

🔴 **Our own programme measured the opposite on real weights.** Paper §7.10: the encoder's
un-normalised post-pool `readout_head` collapses to cosine **0.566** under weight+activation INT8,
and rolled out 20 steps on 880 held-out windows it costs **+0.0215 m ADE@2s — past the
pre-registered 0.02 m falsifier — with the degradation ratio growing 27× from 0.5 s to 2 s.**
That is compounding, on the model we actually ship.

⚠️ The CUDA-graph "bit-exact, rel-err exactly 0.0" row is **near-tautological**: a *static*
`torch.randn` input replayed through a graph must reproduce itself. The aliasing hazard the test
exists to catch requires **varying** inputs across replays. **UNVERIFIED as a hazard test.**

⇒ **Correct reading of the table: an ARCHITECTURE READ — the export/build path is numerically
sound and the kernels do what they claim. It is NOT a deployment precision gate.**

⛔ **STILL REQUIRED BEFORE FLEET DEPLOYMENT:** this is a **numerics** gate on random weights. The
**four-family accuracy gate on real windows with a trained checkpoint** (their 95.3 %
decision-agreement bar) has **NOT** run — Thor has no val data yet. A quantisation that preserves
ADE while degrading manoeuvre κ or route accuracy is exactly the silent regression the binding
rule exists to catch.

## 3. ⚠️ THE BUG THAT MADE THIS NECESSARY — MHA exports SILENTLY WRONG

**Did we use ONNX? Yes — and it is where the only real defect was found.**

| export path | result |
|---|---|
| opset 17, legacy, MHA fastpath ON | ⛔ **rel-err 0.726 — SILENTLY WRONG, no error raised** |
| opset 18, legacy, MHA fastpath ON | ⛔ fails loudly: `aten::_native_multi_head_attention` unsupported |
| ⭐ **fastpath OFF, opset 17 or 18** | ✅ **rel-err 7.9e-7 — PARITY** |

**THE FIX — one line, mandatory before any export:**

```python
torch.backends.mha.set_fastpath_enabled(False)
```

`nn.MultiheadAttention` fuses into `aten::_native_multi_head_attention`, which opset 18 rejects
outright and **opset 17 exports as a wrong graph without complaint**. Disabling the fastpath
decomposes it into exportable ops. It changes eager output by only **5.1e-7**, so it is safe to
leave on permanently.

🔴 **This RETRACTS an inherited claim.** The Production & Optimization run of 2026-07-08 recorded
*"no unexportable ops — MHA/FiLM/causal-triu all fine — falsifier did not fire"*. On **torch 2.13**
that is **false**. Root-cause class: *a passing check on an old toolchain re-asserted on a new one
without re-running it.* ⇒ **ONNX parity must be re-verified per torch version, never inherited.**

### ⚠️ ANNOTATION 2026-08-03 (Production & Optimization) — §3's MECHANISM DID NOT REPRODUCE

*Added, not deleted: the original text stands and this is the measurement beside it.*
`Research/2026-08-03-thor-candidate-fan-and-engine-graph.md` §3/§5 ·
`Implementation/incoming/2026-08-03-thor-b1-fan/`

**MEASURED on the same box, same torch 2.13, same model, 6 cells** — predictor × opset {17, 18} ×
fastpath {ON, OFF}, plus 2 encoder cells:

| cell | fused MHA op in graph | ORT rel-err vs eager | engine median |
|---|---|---|---|
| predictor op17 fastpath **ON** | **no** | **3.6e-07 / 4.32e-07** | 1.154 ms |
| predictor op17 fastpath OFF | no | 3.7e-07 / 4.41e-07 | 1.187 ms |
| predictor **op18 fastpath ON** | **no** | **4.32e-07** | — |
| predictor op18 fastpath OFF | no | 4.41e-07 | — |

⇒ **The flag is inert on this path** (identical node counts, 1223 both), **opset 18 with the
fastpath ON does NOT fail**, and the 0.726 does not reappear. `nn.MultiheadAttention` modules ARE
present (predictor 10, encoder 12, model 41), so "no MHA here" is not the explanation.

**The likelier cause of the 0.726 is the one §5 learning #8 already names:** the superseded
`thor:~/thor_trt_accuracy.json` (18:13) reports fp32 **0.72824** and fp16 **0.72818** — near-identical
across precisions, *"the signature of a wiring/test bug, not a precision problem"* — and learning #9
records a sibling failure in the same session (a gate compared the engine to a **different random
model**). ⇒ 🔴 **the retraction §3 applies to our 2026-07-08 "ONNX-clean" claim rests on a mechanism
that does not reproduce, and its owner should revisit it.**

⚠️ **Fairness:** the script behind `thor_trt_gate.json` is **not on disk** (transient heredoc — only
the 2026-08-03 scripts reference `set_fastpath` anywhere on the box), so what it exported cannot be
inspected. The non-reproduction is scoped to what was re-run.

✅ **Keep the `set_fastpath_enabled(False)` line in §4 anyway** — 5.1e-7 in eager, 1.6 % in engine
latency; cheap insurance against a mechanism not fully mapped.

✅ **§1's headline survives the check that mattered:** the corrected-graph engine measures
**1.187 ms** against the published **1.168 ms** (**1.6 %**), so the 5.33× and the 51.2 ms are
admissible — they had simply never been re-timed after the correction.

### 🔴 ANNOTATION 2026-08-03 — TWO CORRECTIONS TO §6's PRICING

1. ⭐⭐ **O1/O3 understate the tick: every Thor measurement rolls ONE candidate, the deployed
   `TacticalSelector` loops over NINE** (`config.py:95`, `fourbrain.py:571`). MEASURED: 9 candidates
   **serialised** through the shipped **batch-1** `predictor_fp16.plan` = **243.84 ms (244 % of
   budget)**; through a **batch-9** engine = **56.13 ms (56 %)**. ⇒ **O9 is not P2-structural, it is
   a deployment requirement worth 4.3×**, and the shipped engine must be rebuilt with a batch-9 or
   dynamic profile.
2. **O4's saving is ~2× overstated.** `thor_ksweep.json` is an **eager** sweep (4.107–4.308 ms/step);
   at TRT-fp16 K20→K10 saves **11.7 ms**, not 23 (23.36 ms is the whole K20 TRT roll). At the
   batch-9 engine's 1.294 ms/step the whole fan is 25.9 ms and halving K saves **12.9 ms** of a
   56 ms tick.
3. ⛔ **O8 is CLOSED on its own falsifier** — full-roll graph **1.02×** vs per-step (bar was <1.1×).
4. ⛔ **O2 IS BLOCKED and the blocker is not TensorRT.** The encoder does not export to ONNX at the
   deployed **176×624** at all: `SymbolicValueError: adaptive_avg_pool2d, output size that are not
   factor of input size` — at **both** fastpath settings. The 2026-07-08 "encoder exports clean" row
   was measured at **256×256** and is **geometry-conditional and false for the shipping geometry**.
   ⇒ new item **O2-pre**: fix the shape-derived pooling in `stack/`, with an export test **at the
   deployed geometry**.

## 4. DEPLOYMENT PROCEDURE ON THOR

**Environment** (PI rule — two venvs, never mixed):
* `~/venvs/tanitad-edge` — inference / optimisation / AlpaSim closed-loop
* `~/venvs/tanitad-train` — training

⚠️ TensorRT's python bindings are **system** packages; there is no aarch64 TRT wheel matching an
L4T runtime. Reach them with `PYTHONPATH=/usr/lib/python3.12/dist-packages`.

```bash
# 0) one-time
sudo apt-get install -y tensorrt libnvinfer-bin python3-libnvinfer python3-libnvinfer-dev
~/venvs/tanitad-edge/bin/pip install onnx onnxruntime      # no aarch64 GPU wheel: CPU ORT for parity checks only

# 1) export  ⛔ the fastpath line is NOT optional
python - <<'PY'
import torch; torch.backends.mha.set_fastpath_enabled(False)
torch.onnx.export(wrapped_predictor, (states, actions), "predictor.onnx",
                  input_names=["states","actions"], output_names=["z_next"],
                  opset_version=17, dynamo=False)
PY

# 2) build (36 s, 166 MB)
/usr/src/tensorrt/bin/trtexec --onnx=predictor.onnx --saveEngine=predictor_fp16.plan \
                              --fp16 --skipInference

# 3) GATE — never skip. Same process, same weights, 1-step AND 20-step roll.
#    Bind by NAME ('states','actions','z_next'), never by index.
```

**Runtime shape:** bf16 autocast on the encoder · TRT-fp16 engine for the predictor · CUDA-graph
capture where TRT is not used. ⛔ **Never** blanket-cast the model: bf16 is **6.76× on the encoder**
and **0.86× (a LOSS) on the predictor**.

## 5. LEARNINGS — what generalises beyond this device

1. ⭐⭐ **Per-stage, never global.** Precision helps the encoder 6.76× and *hurts* the predictor
   0.86×. One blanket `.half()` would have cost ~14 % on the dominant stage.
2. ⭐ **Match the lever to the bottleneck.** Graph capture on the *encoder* bought 1.09 %; on the
   *predictor* it bought 24 %. Compute-bound stages need arithmetic levers, launch-bound stages
   need capture.
3. ⭐ **Low-precision gain scales with GEMM size.** FP8: **1.21×** at our 8×2048×2048 predictor
   shape vs **1.97×** at 4096³. Published 4-bit/8-bit speedups are measured at LLM scale; a
   small-tensor many-step model lives at the other end of that curve.
4. **The bottleneck moves after every lever.** encoder (69 %) → roll (71 %) → encoder again (54 %).
   Re-measure after each change; never plan two levers ahead.
5. **Levers compose** — measured 98.63 ms vs additive projection 101.25 (−2.6 %), matching the
   4060's 0.4 %. Plan additively, verify occasionally.
6. ⛔ **`torch.compile` is not viable** — failed on the 4060 (no Triton) *and* Thor (InductorError,
   gcc/libcuda). Two platforms, two causes, same verdict: **manual capture + TensorRT**.
7. ⚠️ **Exit codes and package counts are not evidence.** The apt install completed at 15:59 while
   `dpkg -l` mid-transaction still read 0; ~2 h were lost polling the wrong signal. **Read the
   artifact that owns the answer.**
8. ⚠️ **A comparison across two random inits is not a comparison.** The first gate "failed"
   identically at fp32 and fp16 because it compared the engine to a *different* random model.
   Identical error across precisions is the signature of a **wiring/test bug**, not a precision
   problem.
9. **Thor is thermally uncommitted**: GPU ~1976 mW, junction 61.3 → 61.9 °C over 39 min, no
   throttling over 180 s sustained. ~2 W is an automotive-plausible envelope.
10. **Unified memory lies to the allocator.** `max_memory_allocated` said 1.2 GB; system RSS was
    **11.5 GB**. Size deployments from RSS.

---

## 6. EXPERIMENT BACKLOG FOR THE OPTIMISATION AGENT

Ordered by expected value. Each states its falsifier so the agent can run independently.

### P0 — blocking for deployment

| # | experiment | falsifier |
|---|---|---|
| **O1** | **Four-family accuracy gate on the TRT-fp16 engine**, real windows, trained ckpt, paired episode-cluster bootstrap. Needs val data on Thor. | any family degrades beyond CI, or decision-agreement < 95.3 % ⇒ fp16 does not ship |
| **O2** | **TRT engine for the ENCODER** — now the larger stage again (27.8 of 51.2 ms) | engine ≤ bf16 autocast ⇒ keep autocast, close the item |
| **O3** | **End-to-end tick with both engines**, measured not projected | measured > 1.1 × the 51.2 ms projection ⇒ levers stopped composing |

### P1 — high value

| # | experiment | falsifier |
|---|---|---|
| **O4** | **K 20→10 four-family gate** (−23 ms at TRT speed, −41 ms eager) | tactical/strategic degrade ⇒ K stays 20 |
| **O5** | **INT8 PTQ** via TRT calibration + four families | > 1 % on any family ⇒ stop at fp16 |
| **O6** | **NVFP4 via TRT ModelOpt** on the encoder (⛔ eager torch has no fp4 cast path; TRT owns it) | < 1.3 × over fp16 ⇒ not worth the toolchain at our tensor sizes |
| **O7** | **`nvpmodel` power modes** — ~62 °C junction means headroom is unexplored | no gain, or thermals exceed automotive envelope |
| **O8** | **One engine for the whole 20-step roll** (loop inside the graph) vs 20 invocations | < 1.1 × ⇒ per-step invocation is already optimal |

### P2 — structural

| # | experiment | falsifier |
|---|---|---|
| **O9** | **Batched multi-candidate rolls** (`imagine_candidates`) — moves GEMMs rightward on the size curve and could flip the NVFP4/INT8 verdict | per-candidate cost does not amortise |
| **O10** | **Encoder resolution ablation** 176×624 vs 128×448 — four families + latency | accuracy loss beyond CI |
| **O11** | **Multi-camera scaling** — 122 GB and 2 W suggest room for 2–3 cameras | tick exceeds budget, or thermals rise |
| **O12** | **Orin-class port** — same pipeline on a smaller target; memory becomes binding where it is not here | fp16 engine does not fit |
| **O13** | **DLA offload** of encoder layers (Thor has dedicated accelerators, entirely unexplored) | unsupported ops force GPU fallback |
| **O14** | **Sparsity (2:4 structured)** on the predictor — Blackwell has native support | accuracy loss, or no speedup at our sizes |

### Standing rules for this agent

1. ⛔ **Every speed delta carries an accuracy delta** (their G-P2) — and accuracy means the **four
   families**, never ADE alone.
2. ⛔ **Re-verify ONNX parity per torch version.** §3 is why.
3. **Re-measure the bottleneck after every lever** — it has moved three times.
4. **Report p50 AND p99** with warmup + `cuda.synchronize()`.
5. **A negative result is a result.** Four of today's most useful findings were negatives.

## Evidence class

| claim | class |
|---|---|
| all latencies, memory, thermals, rel-errs | **MEASURED (ours)**, `2026-08-02-thor-deployment-profile/*.json` |
| MHA fastpath export bug + fix | **MEASURED (ours)** — bisected ORT-vs-eager in one process |
| A40 5.35×, 95.3 % bar, ONNX-clean claim | **INHERITED** — Production & Optimization runs #4/#5; the ONNX-clean claim is **RETRACTED for torch 2.13** |
| 51.2 ms tick | **MEASURED per stage, PROJECTED in composition** — O3 measures it end-to-end |
| NVFP4/INT8/DLA outlooks | **HYPOTHESIS** — O5/O6/O13 exist to test them |


---

## 11. ⛔ CORPUS DURABILITY RULE (added 2026-08-02, after a real loss)

**What happened:** four RunPod pods were shut down. Model weights were pushed to HF and 283 small
artifacts to git — but the **validation caches were left on pod volumes**, because they were too
large to relay in the remaining runway. The pods were then **TERMINATED, not stopped**, and the
val cache plus every optimizer state went with them.

**Recoverable:** all model weights (HF), all results/logs/configs (git), and — by luck rather than
design — the canonical parity val, which happened to also exist on a still-running training pod
(603 clips, sha-verified against the committed manifest).

**Unrecoverable:** optimizer states (so RR-20 / RR-CTL can be EVALUATED from their HF weights but
never RESUMED), and the eval pod's derived caches.

⇒ **THE RULE: a corpus or cache that exists on exactly ONE disk is a pending loss.**

1. **Any val/eval cache must live in at least two places** — push it to HF as a **dataset** the day
   it is built, not the day the pod is shut down. Weights were handled this way and survived;
   caches were not, and did not.
2. **Distinguish STOP from TERMINATE explicitly in every shutdown note.** Stop preserves the
   volume; terminate does not. Writing "safe to stop" is not writing "safe to terminate", and the
   reader will not supply that distinction for you.
3. **If a full checkpoint cannot be moved, move the model-only extract AND state what the extract
   cannot do.** A model+grounding extract (1.1 GB from 3.3 GB) evaluates fine and **cannot resume
   training**. Say that at the time, not afterwards.
4. **Derived caches deserve a rebuild recipe, not just a copy.** Anything rebuildable from HF plus
   a committed script is not really lost — record the recipe next to the artifact so the next
   reader knows which category it is in.
