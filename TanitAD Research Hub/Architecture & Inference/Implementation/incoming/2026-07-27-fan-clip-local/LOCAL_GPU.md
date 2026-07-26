# LOCAL_GPU — what the dev box can run, MEASURED

**Date:** 2026-07-27. **Evidence class:** MEASURED (ours, dev box). **Tier:** CONFIRMED.
**Raw:** `raw/fc_localgpu.json`. **Probe:** `code/fc_localgpu.py`.

Several recent program results were recomputable with no GPU or on a frozen checkpoint, yet ran on
pods and queued behind training. This note says what no longer has to.

---

## 1. The machine

| | |
|---|---|
| host | `FREEDOM2035`, Windows 11 Pro |
| GPU | **NVIDIA GeForce RTX 4060, 8.00 GiB, SM 8.9, 24 SMs** |
| stack | torch **2.11.0+cu128**, Python **3.13.5**, venv `C:/Users/Admin/venvs/tanitad` |
| fp32 throughput | **7.56 TFLOPS** (8192³ matmul, 145.5 ms) |
| largest single fp32 allocation | **7.5 GiB** |

**Method.** Batch size pushed until `torch.cuda.OutOfMemoryError`, with the real `stack/tanitad`
modules at their committed default configs. No checkpoint, no corpus, no pod.

⚠️ **Windows/WDDM spills instead of raising.** An allocation past 8 GiB frequently does **not** OOM
— the driver pages into shared host memory and the step completes **~28× slower**
(v4 head, batch 128: 10.60 GiB "peak", 6804 ms vs 245 ms at batch 32). `max_memory_allocated` counts
those bytes, so a naive sweep reports capacity that does not exist. Every row whose peak exceeds
0.95 × VRAM is flagged `spilled_to_host` and **excluded** from `max_batch_in_vram`. **Any local
capacity claim in this program must apply that filter or it is fiction.**

⛔ **`torch.compile` is unusable here** — no Triton on Windows: inductor fails and dynamo-cudagraphs
is ~20× slower. Everything below is eager. Use a manual `torch.cuda.CUDAGraph` if a hot loop ever
needs it.

---

## 2. ✅ FITS — run it here

| workload | capacity (in VRAM) | peak | time |
|---|---|---:|---:|
| **Re-scoring a cached fan / bootstrap analysis** | unbounded at our sizes | — | **CPU only** |
| ↳ *this stream:* 3 REF-C fans × 16 bands **+** v4's fan × 2 scorers × 16 bands, each with a **B = 2000 paired episode-cluster bootstrap** | — | — | **26.9 s, CPU** |
| ↳ *this stream:* every committed v5 bar + 3 registry headlines reproduced | — | — | **0.44 s, CPU** |
| **`taniteval` full suite** (`pytest -q`) | — | — | **449 passed, 59.1 s, CPU** |
| **`stack` full suite** (`pytest -q`) | — | — | **1129 collected, exit 0** (1 skipped), CPU |
| **Head-only TRAINING** — the real v4 planner head (10.57 M params, `states [B,8,2048]`) | **batch 32** | 2.73 GiB | 246 ms/step |
| **Head INFERENCE / fan generation** — emits `anchor_traj [B,256,20,2]` | **batch 1024** | 5.62 GiB | 4.67 s |
| **One predictor imagination step** (91.36 M operative predictor, window 8, `state_dim` 2048) | **batch 2048** | 1.74 GiB | 485 ms |
| ↳ at the E-V5-3 reference shape, batch 256 | — | 1.10 GiB | **68 ms** (pod A40: **25.2 ms** → **2.7× slower, and it fits**) |
| **Camera ENCODE, no-grad** — full `flagship4b` ViT, 9-ch 256 px, windows of 8 | **batch 32** (= 256 frames) | 3.82 GiB | 2.04 s → **7.96 ms/frame** |
| **Linear probes** (e.g. the 2,049-parameter probe that beat a 2.17 M head) | trivial | ≪1 GiB | seconds |
| **AdamW state for a 286.34 M model** (weights + grad + 2 moments, no activations) | **fits** | **5.35 GiB** | — |

ⓘ **Two parameter counts, both correct.** `flagship4b_config()` builds **263,440,533** model
parameters here; the registry's **286.34 M** is the canonical total *including the grounding heads*,
which `config.py` documents as living **outside** the `WorldModel` under separate checkpoint keys.
The 5.35 GiB row is measured at the larger figure.

⭐ **The consequence worth acting on.** `881 windows × 256 candidates × 20 steps` = 4.51 M
predictor-step-candidates. At batch 2048 that is **≈ 18 GPU-minutes on this 4060** — i.e. **the whole
E-V5-1 imagination sweep is a dev-box job**, if the per-window `states` were staged. They are not
(28.9 MB in fp16). See `FAN_CLIP_LOCAL.md §8`, escalation 2.

---

## 3. ❌ DOES NOT FIT — or fits only nominally

| workload | measured outcome |
|---|---|
| **Full-model TRAINING** (`flagship4b`, 263.44 M model params + grounding heads ≈ 286 M) | encoder fwd+bwd fits only to **batch 4** (6.36 GiB, 764 ms). Batch 8 **spills** (11.36 GiB, 16.6 s — 22× slower). A 30 k-step run at batch 4 is ~6.4 h of *pure* step time, at a batch the program has never trained at. **Not a viable training host.** |
| **`flagship4b_reduced` training** (53.0 M) | fits to **batch 8** (3.78 GiB, 359 ms). Batch 16 nominally fits (7.25 GiB) but takes **3963 ms** — 11× the time for 2× the batch, i.e. already thrashing. **Practical limit: batch 8.** |
| **Full-model rollouts on real frames** | the *compute* is fine (§2); the **inputs are not here** — see §4. |
| **600-episode encode** | GPU compute alone ≈ **16 min** (119 k frames × 7.96 ms). **Blocked by the corpus, not the card** — see §4. This is a correction: it is not a hardware limit. |
| **Anything at fp16 on the v4 selector** | prohibited independently — Bar A measured that 256 candidates separate by less than fp16's ULP. Keep fp32. |

---

## 4. ⛔ THE REAL LIMIT IS PARITY, NOT VRAM

**The dev box's episode cache is keyed `14231cd29c74`. The canonical parity corpus is
`physicalai-train-e438721ae894` (2376 episodes, skip-hash `f09e44db`).** They are not the same
selection. A sibling agent caught this today and correctly refused to use the local cache; so does
every experiment in this directory.

**Ruled out locally — by parity, at any VRAM:**

- any **training arm** meant to be compared cross-arm (it would break comparability at step 0);
- any **eval on real frames** — `driving_*`, `corridor_*`, closed-loop, gate D1/D2/D3;
- the **600-episode re-adjudication** and any n > 40 power upgrade;
- **encoding the corpus** to produce new `states` / latents;
- anything that **re-selects episodes** at all. *Parity is sacred; refuse it.*

**Admissible locally — the shape of every experiment in this directory:**

> the input is an **artifact already committed to this repo** (a per-window dump, a cached fan, a
> frozen checkpoint), and the computation **re-reduces or re-scores it**. No episode is opened.

**Also admissible:** anything on data that is *not* the parity corpus at all — synthetic inputs,
capacity probes like this one, unit tests, and externally-sourced corpora (which carry their own
provenance and are not claimed to be parity-comparable).

---

## 5. Standing recommendations

1. **Run BOTH suites locally before every commit.** `taniteval` 449 tests in 59 s and `stack`
   1129 tests, both green on this box, no pod. There is no reason for either to queue.
2. **Re-scoring and interval work belongs here, not on a pod.** Bar A, E-V5-1's post-hoc, E-V5-3's
   whole cost curve and this stream's 96-band sweep are all CPU jobs of well under a minute.
3. **Stage per-window `states` alongside per-window errors.** 28.9 MB in fp16 moves fan generation
   and every head-side ablation off the pod queue entirely (§2). This is the single highest-leverage
   change to how the program stages artifacts.
4. **Never quote a local capacity without the spill filter** (§1). It is the dev-box analogue of
   *"never judge pod disk with `df`"* — the tool reports a number that looks like success.
