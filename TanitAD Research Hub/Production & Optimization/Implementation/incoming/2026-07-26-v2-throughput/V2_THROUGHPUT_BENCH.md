# Can we accelerate `flagship-v2corpus-30k`? — measured answer

**Date:** 2026-07-26 · **Author:** performance-engineering agent · **PI question:** *"Any chance to
accelerate the training of the v2 corpus?"*

**Answer: not safely, and not on the evidence available. DON'T RESTART.**

* The **prime suspect — `--grad-checkpoint` off — is dead**: it needs **42.4 GiB** at the production
  micro-batch against pod1's **48 GiB**, and where it *does* fit (micro-batch 8) it **lost 3/3
  repeats**. The bursty 22↔100 % GPU trace was never checkpoint overhead.
* The only candidate that might win, **`32 × 2`**, measured **1.18× / 1.40× / 0.84×** across three
  repeats on a **shared** eval pod — **too noisy to authorise a restart.** It needs an exclusive
  30-minute rerun before anyone spends a GPU-day on it.
* The only **parity-clean** lever, `--workers`, is capped by Amdahl at **1.13×** (~7.6 h).
* And `batch × accum` is **not** a free knob here: two losses (`decorr` = v2 LEVER B, and SIGReg)
  are estimated **across the micro-batch**, and `16 × 4` is a deliberate cross-arm constant.

Details and numbers below.

> Evidence classes used throughout: `MEASURED (ours + artifact)` · `PUBLISHED` ·
> `INHERITED (not re-verified)` · `ESTIMATED` · `HYPOTHESIS`.
> Nothing below is INHERITED — every load-bearing number was re-measured from a primary source.
> **pod1 was never touched beyond read-only inspection. Nothing was committed or pushed.**

---

## 1. Baseline — re-verified from the primary source, not from the brief

MEASURED (`tanitad-pod:/workspace/experiments/flagship-v2corpus-30k/train_log.jsonl`,
read-only, 2026-07-26T02:04Z):

| step | `step_s`/50 | `data_s`/50 | data share |
|-----:|------------:|------------:|-----------:|
| 7250 | 10.63 | 1.23 | 11.6 % |
| 7400 | 10.71 | 1.28 | 12.0 % |
| 7550 | 10.62 | 1.24 | 11.7 % |
| 7700 | 10.70 | 1.28 | 12.0 % |

**10.65 s/step, data 11.6–12.1 %** — the brief's figures confirmed. Note `t_step` **encloses**
`t_data` in the trainer (`train_flagship4b.py:465-492`), so compute ≈ **9.40 s/step**.

Live invocation re-read from `/proc/699284/cmdline` (the actual python child is PID **699286**):
no `--rollout-k`, no `--staged-levers`, no horizon overrides. `config.json` confirms
`needed_fut` = 16 indices, `max_horizon` 20, `rollout_k` 12, encoder 9ch/256px/d768/depth12,
**286,339,251** trainable params.

---

## 2. Devices — the two cards are NOT interchangeable

MEASURED (`nvidia-smi`, both pods):

| | pod1 (`tanitad-pod`) | eval pod (`tanitad-eval`) |
|---|---|---|
| GPU | **RTX A6000**, 47.99 GiB | **A40**, 44.42 GiB |
| SM clock under load | **1875 MHz** (max 2100) | **1665–1725 MHz** (max 1740) |
| Power | 208 W / 300 W | **296–310 W / 300 W**, `SW_POWER_CAP` active (`0x4`) |
| ECC | **Disabled** | **Enabled** |

Two consequences, both load-bearing:

1. **The A40 has 3.6 GiB LESS memory than pod1.** So an OOM on the A40 is not automatically an OOM
   on pod1 — memory verdicts are therefore given as a **fitted model** (§5), not as raw OOMs.
2. **The A40 power-throttles under sustained load; pod1's A6000 does not** (it sits at 208 W
   because the loader keeps it only ~56 % busy). So **absolute s/step does not transfer** —
   only ratios measured on the same device in the same session do.

### Fidelity check — the bench really is the production workload
MEASURED: bench trainable params **286,339,251** = pod1's `config.json` **exactly**; bench peak
reserved at the production config **14.293 GiB** vs pod1's nvidia-smi **15,288 MiB** (≈14.2 GiB
torch-reserved + ~0.6 GiB CUDA context). Geometry, horizon plan, `rollout_k`, and the whole `--v2`
lever block are copied from `train_flagship4b.py`. Frame *values* are synthetic; every shape,
dtype and code path is production, and compute here is value-independent.

---

## 3. Measurement hazards found (three passes were invalidated before a number was trusted)

Recorded because each would have produced a confidently wrong recommendation:

1. **Single-process sweeps leak memory after an OOM.** A caught `torch.cuda.OutOfMemoryError`
   keeps a traceback referencing the frames that hold `frames`/`fut`/activations, so
   `del model; empty_cache()` cannot reclaim them. Every config after the first OOM reported
   ~42 GiB already "in use" and OOMed spuriously. → **one process per config** (`run_sweep.sh`).
2. **The eval pod is NOT free.** Another agent's IDM jobs (`idm2_encode.py`, then
   `idm2_diag_curve.py`, `idm2_v2.py`) occupy the GPU at 85–90 % util intermittently. The *same*
   baseline config measured **10.86 s/step** on a verified-idle GPU and **18.48 → 26.07 s/step**
   under contention. **Their jobs were left running and untouched.**
3. **The A40 drifts under sustained load** (step times climbed 14.9 → 22.0 s within one run).

→ Final timing design: **interleaved ABBA, 3 repeats**, all configs in one session so contention
and drift hit every arm alike; ratios are read, absolutes are not. Memory, by contrast, is
contention-insensitive and perfectly reproducible (14.293 GiB on two independent runs).

**How badly this matters — the same lever flipped sign.** Measured against the same baseline,
`gc_off_8x8` looked **1.16× FASTER** while the co-tenants were running at 85–90 % util and varying,
and **0.95× (slightly slower)** once the co-tenant load settled to a stable ~25 %. The first
reading also implied gradient checkpointing was *faster* than no checkpointing at matched
micro-batch, which is physically impossible — that was the tell. **The final timing table below is
taken only from the stable-contention session.** Anyone re-running this must re-check
`nvidia-smi --query-compute-apps` first: this pod is shared.

---

## 4. Sweep results

All arms hold **effective batch 64**. Interleaved ABBA, stable-contention session, A40.
**Read the ratio column, not the seconds** — the absolute s/step carries the co-tenant's load
(§2, §3). Peak memory is from the fresh-process probes and *is* absolute.

### 4.1 Headline table

| config | batch × accum | grad-ckpt | s/step (A40, contended) | speedup vs baseline | GPU util | peak reserved | fits pod1 (48 GiB)? |
|---|---|---|---|---|---|---|---|
| **`gc_on_16x4` (production baseline)** | 16 × 4 | on | 14.43 / 17.71 / 17.01 | **1.000×** | 89–99 % | **14.29 GiB** | ✅ (running now) |
| **`gc_on_32x2`** | **32 × 2** | on | 12.28 / 12.69 / 20.24 | **1.18× / 1.40× / 0.84×** — ⚠️ **NOT ESTABLISHED, see §4.2** | 92–100 % | 25.51 GiB | ✅ |
| `gc_off_8x8` | 8 × 8 | **off** | 15.16 / 17.86 / 20.44 | **0.83–0.99×** (3/3 repeats: no win) | 91–99 % | 23.51 GiB | ✅ |
| `gc_off_16x4` | 16 × 4 | **off** | — | — | — | **42.35 GiB** (fitted; 38.65 reached before OOM) | ❌ **~5.6 GiB margin — refuse** |
| `gc_on_64x1` | 64 × 1 | on | — | — | — | **38.60 GiB** (fitted; 38.08 before OOM) | ⚠️ thin |
| `gc_off_32x2` / `gc_off_64x1` | — | off | — | — | — | 80.2 / 155.9 GiB (fitted) | ❌ impossible |

Reference: the **one clean, uncontended** baseline measurement was **10.857 s/step**, against pod1's
**10.65 s/step** — a 1.02× device agreement that validates the whole rig.

### 4.2 ⚠️ The `gc_on_32x2` speedup is NOT established — the third repeat reverses it

Within-repeat paired ratios vs the baseline:

| repeat | execution order | baseline | `32×2` | ratio | co-tenant state |
|---|---|---|---|---|---|
| 0 | base first, `32×2` last | 14.427 | 12.276 | **1.175×** | 1 co-tenant, ~25 % util, stable |
| 1 | `32×2` first, base last | 17.711 | 12.692 | **1.395×** | 1 co-tenant, load rising |
| 2 | base first, `32×2` last | 17.011 | 20.243 | **0.840×** | **co-tenants went 1 → 4 mid-repeat** |

**Spread 0.84× – 1.40×. That is not a measurement, it is a rumour.** The pooled median
(**1.2475×**) is arithmetically real but pools three different machines-in-effect, and this program's
standard is explicit: *a claim that decides a GPU-day must be MEASURED*. **1.2×  for `32×2` does not
clear that bar and must not be quoted as if it did.**

What survives the noise (direction, not magnitude): `32×2` was the fastest arm in 2 of 3 repeats and
posted the two lowest absolute times of the whole session (12.28, 12.69); its own times were the
*most stable* of any config until the repeat-2 surge. The mechanism (fewer, larger micro-batches →
less per-micro overhead, higher util) is sound and matches the brief's hypothesis. But the size of
the win is **unknown to within a factor that spans "worth 19 h" and "actively harmful."**

**To certify it:** rerun `--interleave gc_on_16x4,gc_on_32x2 --repeats 4` in an **exclusive** window
on the eval pod (~30 min, GPU otherwise idle). Until then `32×2` is a HYPOTHESIS with a plausible
mechanism, not a MEASURED lever.

`gc_off_8x8`, by contrast, **lost in all three repeats** (0.952 / 0.992 / 0.832). A lever that never
wins under any contention condition is a safe negative — and it is corroborated by the
contention-free memory verdict (§5). **That verdict is solid.**

*(Absolute times drift upward across repeats — baseline 14.4 → 17.7 → 17.0, `gc_off_8x8`
15.2 → 17.9 → 20.4 — as co-tenant load grew. This is exactly why only within-repeat ratios are
quoted, and why even those failed in repeat 2.)*

### 4.2b Profile of one full optimizer step (baseline config)

MEASURED, `torch.profiler` (CPU+CUDA), one complete `16 × 4` step — raw table in
`profile_gc_on_16x4.txt`, top rows by self-CUDA time:

| op | self CUDA | share | # calls |
|---|---|---|---|
| **`aten::copy_`** | **2.804 s** | **22.99 %** | **48,432** |
| `aten::mm` | 2.713 s | 22.24 % | 18,539 |
| `aten::addmm` | 1.338 s | 10.97 % | 7,561 |
| `aten::add_` | 1.147 s | 9.41 % | 33,984 |
| `aten::native_layer_norm_backward` | 0.668 s | 5.48 % | 3,369 |

Two things jump out, and both support the report's conclusions:

1. **`aten::copy_` is the single biggest CUDA consumer (23 %), and it also burns 25.9 % of CPU
   total (5.52 s).** The trainer moves **~4.2 GB per optimizer step** host→device
   (66.1 MB/sample × 64) through a DataLoader with **no `pin_memory`** and `.to(device)` calls with
   **no `non_blocking`** (§8) — i.e. synchronous, pageable copies that cannot overlap. This
   promotes the §8 `pin_memory` item from an ESTIMATED ~2–3 % nicety to **the most clearly
   identified inefficiency in the step**, and it is a 2-line change. *(Caveat: `copy_` also covers
   internal device-side copies and grad-checkpoint stashing, so not all 23 % is recoverable —
   sizing it is exactly what the next run's A/B should measure.)*
2. **The kernel counts are enormous** — 48 k copies, 34 k `add_`, 18.5 k `mm` **per step**. A step
   is thousands of small launches, which is precisely the per-micro-batch-overhead regime that makes
   *fewer, larger* micro-batches (`32 × 2`) attractive and *more, smaller* ones (`8 × 8`) a loss —
   exactly the ordering §4.1 measured. It also explains the oscillating 22↔100 % utilisation without
   needing grad-checkpointing as the culprit.

Note what is **not** in the top rows: no unfused attention (`need_weights=False` is already set,
§8), and the GEMMs that dominate are `ampere_bf16_s16816gemm_*` — bf16 tensor-core kernels, i.e.
autocast is doing its job.

### 4.3 What this says about the hypothesis

The brief's mechanism was right and its remedy was wrong. **Per-micro-batch overhead is real** —
halving the micro-batch count (4 → 2) buys 1.18–1.40×, and *doubling* it (4 → 8, the only way to fit
`gc_off`) costs ~3–5 %. But the overhead is **not** grad-checkpoint recompute: removing that
recompute entirely (`gc_off_8x8`) is a **wash**. The lever is the number of micro-batches, and the
bursty 22↔100 % GPU trace on pod1 is the loader (§6) plus the 4 per-micro-batch `.item()` syncs,
not checkpointing.

---

## 5. Memory — this is what actually kills the prime suspect

Activation memory is linear in micro-batch. Fitting `peak_alloc = fixed + slope × micro_batch` to
the MEASURED points (`gc_on` mb8 = 8.766, mb16 = 13.028; `gc_off` mb8 = 23.429 GiB):

* **fixed = 4.50 GiB** — cross-checks against 286.34 M params × 16 B (fp32 weights + grads + AdamW
  m,v) = **4.27 GiB**. The model is sound.
* **slope `gc_on` = 0.533 GiB/sample**; **slope `gc_off` = 2.366 GiB/sample** (**4.44×**).

| micro-batch | `--grad-checkpoint` ON | OFF |
|---|---|---|
| 8  | 8.77 GiB | **23.43 GiB** (MEASURED, fits) |
| **16 (production)** | **13.03 GiB** (MEASURED) | **42.35 GiB** (fitted) — **OOM on A40; ~5.6 GiB margin on pod1** |
| 32 | 21.55 GiB | 80.20 GiB |
| 64 | 38.60 GiB | 155.90 GiB |

**Verdict on the prime suspect:** turning `--grad-checkpoint` **off at the production micro-batch
of 16 needs ~42.4 GiB** against pod1's 47.99 GiB. That is a ~5.6 GiB margin *before* reserve/
fragmentation overhead, on a job that must survive **~58 more hours** unattended. One fragmentation
spike OOM-kills a run that has already burned days. **Not acceptable.** It is only feasible at
**micro-batch 8** (23.4 GiB) — which is a different config with its own cost (§7).

`64×1` needs **38.6 GiB** allocated (MEASURED before it OOMed on the A40). It would *fit* on pod1
with ~9 GiB nominal headroom, but reserved-vs-allocated overhead pushes it past ~42 GiB. Same
unattended-risk objection, and §4 shows it buys nothing anyway.

---

## 6. The loader levers are capped by arithmetic, not by tuning

MEASURED: step 10.65 s, data-wait 1.25 s. If the loader were made **perfectly free**:

```
10.65 s  ->  9.40 s   =  1.133x   (11.7 % of wall clock, 7.8 h of the remaining 66 h)
```

**1.133× is the hard ceiling for `--workers`, `--v2-lru`, and every other input-side change
combined.** MEASURED on pod1 (read-only): 128 CPUs at load ~5; trainer main RSS 3.2 GB + 8 workers
10.3 GB = 13.5 GB against a **62 GB cgroup limit**. Raising `--workers` to 16 would put it near
~24 GB — still under the `--guard-limit-gb 45` sweep threshold, so it is *feasible*. But it cannot
exceed 1.133×, so `--v2-lru` was not swept separately: it is bounded by the same ceiling and cannot
pay for a restart on its own.

*(Caveat on more workers: the guard (`finetune_traj.start_cache_guard`) drops clean page-cache via
`posix_fadvise(DONTNEED)` when **cgroup** usage — which includes page cache — crosses 45 GB. More
workers means more resident payload and more frequent sweeps, whose cost is cold re-reads. So the
realised gain would land below the 1.133× ceiling, not at it.)*

**`--workers` is the only lever here that is scientifically free** — it changes *when* windows are
fetched, never *which* ones or how they are grouped (labels are derived deterministically from
poses; no augmentation). Parity is untouched.

---

## 7. Why `batch × accum` is NOT a free trade — two batch-coupled losses

The brief assumed constant `batch × accum` is "optimization-equivalent in expectation but not
bit-identical". **That is too generous.** Two terms in `flagship_loss` are estimated **across the
micro-batch** and therefore change when the micro-batch changes:

* **`ego_decorr_loss`** (`tanitad/train/decorr.py:55-76`, the estimator at :74, v2 LEVER B / H25, weight 0.05) is the
  **mean squared Pearson correlation** between the pooled latent and the fed ego, estimated over
  the micro-batch: `corr = (zs.T @ es) / b`. For uncorrelated inputs `E[r²] ≈ 1/b`, so the penalty
  carries a micro-batch-dependent **noise floor** — a bias in the very lever this run exists to
  measure:

  | micro-batch | `E[r²]` noise floor | vs production mb16 |
  |---|---|---|
  | **8** (the only memory-safe way to drop checkpointing, §5) | 0.125 | **2.00× worse** |
  | **16 (production)** | 0.0625 | 1.00× |
  | 32 | 0.0312 | 0.50× |
  | 64 | 0.0156 | 0.25× |

  Note the trap: the *only* config that lets us switch gradient checkpointing off within memory
  (micro-batch 8) is also the one that **doubles** the decorr noise floor. The speed lever and the
  science pull in opposite directions here.
* **SIGReg** (`tanitad/models/sigreg.py`) is a sliced distributional test over
  `z.reshape(-1, D)`; its effective sample count scales with the micro-batch (milder, since tokens
  already enlarge N, but non-zero).

*(No BatchNorm anywhere — LayerNorm/RMSNorm only, and BatchNorm is explicitly banned in the
inference path — so that particular failure mode is absent.)*

**Consequence:** changing `batch × accum` mid-run does not merely reorder floating-point
reductions; it **changes the effective strength of an encoder-ego decorrelation lever that is
under active study**, at step 7.7 k of 30 k, producing a run whose first quarter and last
three-quarters were trained under different regularization. Any later "LEVER B did/didn't work"
read would be confounded. This program has already been burned by exactly this class of confound
(the "strategic choice is a ~2 % lever" retraction).

*(Minor, same direction: `drop_last=True` discards up to `batch_size − 1` windows per epoch, so 64×1
also drops up to 63 windows/epoch vs 15 at 16×4.)*

### 7.1 `16 × 4` is a deliberate cross-arm constant, not an incidental default

MEASURED from `Project Steering/MODEL_REGISTRY.md` — **every** flagship arm was launched with
`--batch-size 16 --accum 4`:

| arm | registry line | batch × accum |
|---|---|---|
| `flagship4b-phase0-30k` (no-speed control) | :121 | 16 × 4 |
| `flagship4b-speedjerk-30k` (**deployed v1**) | :140 | 16 × 4 |
| `flagship4b-v3enc-30k` | :345 | 16 × 4 |
| `flagship-v4.2-30k` | :739 | 16 × 4 — *registry's own words:* **"eff batch 64 (batch 16 × accum 4, matching v1)"** |
| `flagship-v2corpus-30k` (this run) | :812 | 16 × 4 |

v4.2's entry states the match to v1 as a **distinguishing-lever decision**. So `16 × 4` is a
constant the program deliberately holds across arms, and the micro-batch is the level at which the
two batch-coupled regularizers above are estimated. **Changing it for `v2corpus` alone would break
exactly the matching v4.2 was configured to preserve** — and it would do so a quarter of the way
into the run, so the arm would not even be self-consistent.

This is the decisive objection to the one lever that actually works (§4). It is a
comparability argument, not a parity-of-corpus one: the sacred corpus and skip-hash are untouched
by `batch × accum`.

---

## 8. Levers checked and found already optimal / not worth it

* **Fused attention — already on.** `nn.MultiheadAttention(..., need_weights=False)` in *both*
  `encoder.py:33` and `predictor.py:44`, so PyTorch already dispatches to the fused SDPA kernel.
  (Had `need_weights` been left at its `True` default this would have been the single biggest
  lever in the program — it is worth stating explicitly that it is **not** available, because it
  was checked, not assumed.)
* **`pin_memory` / `non_blocking` — genuinely absent, and the profile says this is the best
  code-level lead.** `train_flagship4b.py:356-358` builds the DataLoader with no `pin_memory`, no
  `prefetch_factor`, and the `.to(device)` calls carry no `non_blocking`. At 66.1 MB/sample that is
  **~4.2 GB of synchronous pageable H2D copy per optimizer step** — and §4.2b measures
  **`aten::copy_` as the single largest CUDA-time op at 23 %** (plus 25.9 % of CPU total). Not all of
  that is H2D, so the recoverable share is unknown, but this is the one place where a **2-line
  change** targets the biggest measured line item. Belongs to the *next* run, not a mid-flight
  restart — and it should be A/B'd, not assumed.
* **`torch.compile`** — plausible on a ViT and available on Linux pods (unlike the dev box), but a
  recompilation-risk change to a 4-brain model 7.7 k steps into an unattended 66 h run. Next run.

---

## 9. Restart economics

`ckpt.pt` **auto-resumes** (`train_flagship4b.py:394-400`: `step = int(ck["step"]) + 1` at :399 — note the brief cited 385-393, the block is actually 394-400 in the current revision) — verified.

**But "current" ≠ "at the live step".** MEASURED: `ckpt.pt` holds **step 7000** (written
2026-07-25T23:59:44Z) while the live step was **7700**. `--ckpt-every 1000`, so:

| restart timing | steps discarded | hours lost |
|---|---|---|
| at an arbitrary moment | up to 1000 | up to **2.96 h** |
| right now (was step 7700) | 700 | **2.07 h** |
| **immediately after a `[ckpt] saved at step N` line** | ~0 | **0.00 h** |

Plus ~0.1 h process overhead (model build + 3.4 GB ckpt load + worker spawn + warmup).
**If anything is ever restarted, it must be done in the minutes after a checkpoint write.**

Break-even, from step 7700 (22,300 steps remaining, 66.0 h), restarting right after a checkpoint:

| speedup | new ETA | net hours saved |
|---|---|---|
| 1.03× | 64.0 h | 1.8 h |
| 1.05× | 62.8 h | 3.0 h |
| 1.10× | 60.0 h | 5.9 h |
| **1.133× (loader ceiling)** | **58.2 h** | **7.6 h** |
| 1.20× | 55.0 h | 10.9 h |
| 1.30× | 50.7 h | 15.1 h |

*(One more restart cost, applying to any restart including a crash-resume: the resume path restores
model/optimizer/step but **not the DataLoader iterator**, and `torch.manual_seed(args.seed)` runs
at startup — so the shuffle order restarts from the top rather than continuing.)*

---

## 10. Recommendation

### **DON'T RESTART `flagship-v2corpus-30k`. Let it finish at ~66 h.**

Three independent reasons, in descending order of how solid the evidence is:

1. **The prime suspect is dead on memory, and that verdict is airtight.** `--grad-checkpoint` off
   needs **42.4 GiB at the production micro-batch against pod1's 48 GiB** (§5) — a ~5.6 GiB margin
   before fragmentation, unattended, for ~58 h. At the only micro-batch where it *does* fit (8), it
   **lost all three repeats** (0.83–0.99×). There is no version of this lever that wins.
2. **The one lever that might win is not measured well enough to spend a GPU-day on.**
   `32 × 2` posted 1.18× and 1.40× — and then **0.84×** when the shared eval pod picked up three
   more tenants (§4.2). A number whose range spans "saves 19 h" and "costs 10 h" cannot authorise a
   restart. **This is the honest state of the evidence, not a hedge.**
3. **Even if it were certified, it is the wrong change for THIS arm.** `16 × 4` is a deliberate
   cross-arm constant (§7.1 — v4.2's registry entry says *"matching v1"*), and the micro-batch is
   the level at which `decorr` (**v2 LEVER B / H25**) and SIGReg are estimated (§7). Switching at
   step 7.75 k of 30 k would leave the arm internally inconsistent and would give `v2corpus` — the
   arm that exists to isolate the **v2 corpus** — a second distinguishing lever.

**What to do instead:**

* **Let it run.** ETA ~66 h from step 7 750; nothing safe is being left on the table.
* **Certify `32 × 2` cheaply and separately.** ~30 min on an **exclusive** eval-pod window:
  `--interleave gc_on_16x4,gc_on_32x2 --repeats 4`. That is the pre-registered experiment; both
  outcomes are useful and neither risks the running job.
* **Bank it for the next launch, not this one.** From step 0 a real `32 × 2` win is worth *more*
  (full 30 k steps) and costs *nothing*, because a fresh arm sets its own constant. Pair it with
  the §8 code items (`pin_memory=True` + `non_blocking=True`, ~2–3 %; evaluate `torch.compile`).

**If the PI overrides and wants the hours anyway,** the restart is only defensible under all four
conditions:
* certify the speedup on an idle GPU **first** — restarting on the current numbers is a coin-flip;
* fire it in the **minutes after a `[ckpt] saved at step N` line** (else up to 2.96 h is discarded — §9);
* use `--batch-size 32 --accum 2` (**not** `64 × 1`: 38.6 GiB MEASURED, too thin on a 48 GiB card,
  and **not** `--grad-checkpoint` off: 42.4 GiB, §5) — and take `--workers 16` in the same launch,
  since it is free (§6);
* record the micro-batch change in `MODEL_REGISTRY.md` as a **second distinguishing lever** for the
  arm, so no later report reads its ADE as a clean corpus comparison.

**On the brief's prime suspect:** `--grad-checkpoint` off is **dead**, and not for a subtle reason —
it needs **42.4 GiB at the production micro-batch against pod1's 48 GiB** (§5). At the only
micro-batch where it fits (8), it is **0.95–0.99×**, i.e. a **wash to slightly slower**: the
recompute it saves is paid back by doubling the number of micro-batches. The bursty 22↔100 % GPU
trace was **not** grad-checkpoint overhead.

**On the loader:** `--workers`/`--v2-lru` are capped at **1.133×** by arithmetic (§6) and cannot
justify a restart on their own — but `--workers 16` is the one change that is *scientifically free*,
so if a restart happens for any other reason, take it in the same launch.

---

## 10.1 Pre-registered follow-up: certify `32 × 2` (cheapest discriminating experiment)

Registered here **before** it is run, both outcomes committed in advance:

* **Cost:** ~30 min, eval pod, **exclusive** (verify `nvidia-smi --query-compute-apps` is empty first).
* **Command:** `--interleave gc_on_16x4,gc_on_32x2 --repeats 4 --timed-steps 5 --warmup-steps 2`
* **Read:** the **4 within-repeat paired ratios**, not the pooled median.
* **Decide in advance:**
  * **All 4 ratios ≥ 1.15× and spread < 0.15** → the lever is real. **Still do not restart
    `v2corpus`** (§7.1 / §10 reason 3) — instead make `32 × 2` the default for the **next** flagship
    launch and record it in `MODEL_REGISTRY.md`.
  * **Any ratio < 1.05×, or spread ≥ 0.15** → the repeat-2 reversal was not an artifact; the lever is
    unreliable on this hardware. Drop it and keep `16 × 4` everywhere.
* **What would change the "don't restart" call:** nothing in this experiment alone. Only a
  PI decision that wall-clock outranks cross-arm comparability would, and §10's four conditions
  apply if so.

---

## 11. Deliverable manifest

All paths are in the repo working tree; **nothing was `git add`ed, committed, or pushed.**

Base dir: `TanitAD Research Hub/Production & Optimization/Implementation/incoming/2026-07-26-v2-throughput/`

| artifact | file | what it holds |
|---|---|---|
| this report | `V2_THROUGHPUT_BENCH.md` | sweep, memory model, ETA math, recommendation |
| bench script | `bench_v2_throughput.py` | real `WorldModel` + real `flagship_loss`; `--interleave` ABBA mode, per-config device telemetry |
| per-process runner | `run_sweep.sh` | one process per config (the OOM-leak fix, §3) |
| raw sweep JSON | `results_interleaved.json` | 3 repeats × 3 configs, per-step times, util/clock/power samples |
| raw memory-probe JSON | `results_memory.json` | 5 fresh-process probes incl. both OOM verdicts |
| profile (table) | `profile_gc_on_16x4.txt` | `torch.profiler` CPU+CUDA op table, one full baseline step (§4.2b) |
| profile (JSON) | `results_profile.json` | same run's top-25 ops, machine-readable |

**Nothing was `git add`ed, committed, or pushed** (checked with `git status --short`: my only entry
is this untracked directory; the other modified files belong to sibling agents).

Pod-side scratch (eval pod `tanitad-eval`): `/root/v2bench/` — bench, synced stack at
`/root/v2bench/stack`, raw logs `tight.log` / `memprobe.log` / `profile.log`. **Scratch only; the
repo copies above are authoritative.** Nothing was written to pod1, pod2 or pod3.

### Reproducing
```bash
# on a pod with the stack synced to /root/v2bench/stack
PYTHONPATH=/root/v2bench/stack PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 bench_v2_throughput.py --out results.json \
  --interleave gc_on_16x4,gc_off_8x8,gc_on_32x2 --repeats 3 --timed-steps 4
```
**First check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` — this pod is shared,
and a co-tenant silently flipped the sign of one lever in this very study (§3).**
