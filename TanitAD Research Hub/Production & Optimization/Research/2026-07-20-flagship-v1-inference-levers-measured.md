# Flagship v1 inference levers — PROJECTED becomes MEASURED

- **Agent / task:** inference-lever *measurement* (a sibling agent owns the Orin/Thor desk
  research — **no Jetson number is claimed anywhere in this note**).
- **Run (exact, from the campaign log):** smoke 22:51:56 → full sweep 22:54:48–23:13:04 → tail
  replicates 23:13:04–23:17:21 UTC 2026-07-20; fan block 01:51:32–01:55:56 and shared-encoder fan
  01:58:26– UTC 2026-07-21. Note written 2026-07-21.
- **Hardware / cost:** `tanitad-eval` **NVIDIA A40**, batch 1, **exclusive** — the run took the
  pod-wide advisory device lock (`gpu_lock.sh acquire inference-levers`) and held it across the
  **whole campaign**, not per script. **~30 min GPU total, $0.** All four GPUs were occupied when
  this work started; the job blocked on the *device lock* for 30 min (22:21→22:51) and nothing was
  measured until the eval GPU was genuinely idle. A latency benchmark that shares a GPU is not a
  measurement.
- **Model:** `flagship-30k` = **`flagship4b-speedjerk-30k`, step 29,999** — v1 as deployed
  (registry §1.2), **not** the `phase0-30k` no-speed ablation control.
- **Corpus:** physicalai val (`physicalai-val-0c5f7dac3b11`) — the val the accuracy panel uses.
- **Artifacts (all in-repo):**
  `taniteval/results/eff_levers_flagship-30k.json` (16 variants × 2 precision blocks, stage
  breakdown, k-sweep, strided-head block) · `taniteval/results/eff_levers_tail_flagship-30k.json`
  (tail replicates) · `taniteval/results/eff_levers_fan_flagship-30k.json` and
  `…/eff_levers_fan_sharedenc_flagship-30k.json` (planning-fan cost) ·
  harness `taniteval/taniteval/efficiency.py` (levers section) ·
  `taniteval/tests/test_efficiency_levers.py` (23 tests) · `taniteval/levers_tail.py` ·
  `taniteval/levers_fan.py`.
- **Baseline improved on (not re-derived):** `taniteval/results/eff_flagship-30k.json`.
- **Contamination:** every published block has `contamination_check.valid == true`, with
  exclusivity sampled **before, after and between every variant**. Zero build errors.

---

## 0. The lever table — fp32, the strict reference block

Reference = the eager tick measured in the same block, same protocol, same warm GPU.
`Δ ADE` is `ade_0_2s` recomputed on the identical 32 val windows (`bench._suite`'s definition).

| lever | what it does | tick p50 → | p99 → | ×p50 | max abs dev | Δ ADE | 10 Hz @p99 | class |
|---|---|---|---|---|---|---|---|---|
| — | **eager baseline** | **100.29** | **113.98** | 1.00 | — | — | **NO** | MEASURED |
| **L1a** `graph_step` | 1 graph/step, replayed 20× | 57.33 | 57.63 | 1.75 | **0.0 m** | **0.0** | YES | MEASURED |
| **L1b** `graph_rollout` | ONE graph over all 20 steps | **57.18** | **57.45** | **1.75** | **0.0 m** | **0.0** | **YES** | MEASURED |
| **L1c** `graph_fulltick` | ONE graph over encode+rollout | 57.34 | 57.66 | 1.75 | **0.0 m** | **0.0** | YES | MEASURED |
| **L1d** `compile_rollout` | `torch.compile(reduce-overhead)` | **52.89** | **53.12** | **1.90** | 3.8e-6 m | 3e-8 | YES | MEASURED |
| **L1e** `compile_cudagraphs` | `torch.compile(backend=cudagraphs)` | 58.24 | 58.54 | 1.72 | 0.0 m | 0.0 | YES | MEASURED |
| **L2** `enc_cache` | rolling 1-frame encoder cache | 95.11 | 102.67 | **1.05** | 1.9e-6 m | 1.5e-8 | no | MEASURED |
| **L4** `enc_cache_graph` | L1b + L2 | 33.27 | 33.42 | 3.01 | 1.9e-6 m | 1.5e-8 | YES | MEASURED |
| **L7** `drop_horizons` | compute only the k=1 head | 100.47 | 112.26 | **1.00** | **0.0 m** | **0.0** | no | MEASURED |
| **L7+L1b** `drop_horizons_graph` | pruned + graph | 56.62 | 56.90 | 1.77 | **0.0 m** | **0.0** | YES | MEASURED |
| **L3-ctl** `autocast16_eager` | `torch.autocast(fp16)` | 112.14 | 131.64 | **0.89** | 0.024 m | −1.4e-4 | no | MEASURED |
| **L3** `fp16_eager` | `model.half()` fp16 weights | 98.47 | 104.03 | **1.02** | 0.024 m | −4.6e-4 | no | MEASURED |
| **L3′** `fp16_fp32acc` | fp16 net, fp32 SE(2) accumulate | 101.17 | 114.46 | 0.99 | **0.013 m** | 1.3e-4 | no | MEASURED |
| **L4** `fp16_graph_rollout` | L1b + L3 | 24.76 | 24.78 | 4.05 | 0.024 m | −4.6e-4 | YES | MEASURED |
| **L4** `fp16_enc_cache_graph` | L1b + L2 + L3 | 19.18 | 19.19 | 5.23 | 0.024 m | −6.6e-5 | YES | MEASURED |
| **L4** **`all_levers`** | **L1b + L2 + L3 + L7** | **18.75** | **18.76** | **5.35** | 0.024 m | −6.6e-5 | **YES** | MEASURED |

tf32 block (the A40 deployment default), same protocol: eager **94.50 / 102.48** (still misses
at p99) → `graph_rollout` **40.26 / 40.49** (2.35×) → `compile_rollout` **36.42 / 36.57** (2.59×)
→ `enc_cache_graph` **28.16 / 28.17** (3.36×) → **`all_levers` 18.76 / 18.82** (5.04×).

> `all_levers` lands at **18.75 ms (fp32 block)** and **18.76 ms (tf32 block)** — identical,
> because once the weights are fp16 and the rollout is captured, the TF32 switches have nothing
> left to act on. That agreement is a free internal consistency check on the whole sweep.

**Headline 1 — the tick. The composed tick is 18.76 ms at p99 = 53.3 Hz, against a 100 ms budget.
v1 meets 10 Hz at p99 with 5.3× headroom, for a 3.4 mm mean waypoint shift and Δ ADE −0.07 mm.**

**Headline 2 — the planner. An 8-candidate imagine-and-select fan costs 20.82 ms p50 / 23.72 ms
p99, not the 723 ms the "8 × 20 steps" arithmetic predicts (§6b). A 32-candidate fan costs
28.41 ms. CEM-style planning over this world model is not latency-blocked.**

---

## 1. What a row means

One **planning tick** = `encode(8-frame window) → 20 sequential operative-predictor steps →
per-step metric Δpose via the step readout → SE(2) dead-reckoning`. That is the path that
produces the ADE@2s the leaderboard scores. It is **not** the 2026-07-18 *decision tick*
(`encode(1 frame) + select_K9`), which excludes the rollout entirely and was measured on a
different GPU, checkpoint and corpus.

Every row uses the baseline panel's protocol **unchanged**: batch 1, W = 8, k = 20 @ 10 Hz,
per-iteration `torch.cuda.Event` timing bracketed by `torch.cuda.synchronize()`, 30 warmup
iterations discarded, **200 timed iterations**, both TF32 switches moved together, host→device
copy excluded. Variants differ **only** in the intervention named.

Every row also carries an **equivalence block** — the decoded trajectory over the same 32 real
val windows compared against the eager reference of the *same* precision block, plus `ade_0_2s`
recomputed with `bench._suite`'s definition (test-pinned). A speed row without an accuracy row
is not publishable; a fast wrong answer is worthless.

⚠️ **The equivalence ADE is not the heldout ADE.** Reference `ade_0_2s` on these 32 windows is
**0.1929 m**, because they come from 2 val episodes. v1's decision-grade number remains
**0.4522 ± 0.0312** (registry). The equivalence block measures *change*, and Δ is what it
licenses — never the level.

---

## 2. L1 — CUDA-graph the rollout: **CONFIRMED**, and it settles the 10 Hz question

**100.29 → 57.18 ms p50, 113.98 → 57.45 ms p99, bit-identical output.** The eager tick MISSES the
10 Hz budget at p99; the graphed tick meets it with 42 % headroom. In the tf32 block the same
lever is worth 2.35×. This is the first optimised variant of the planning tick that exists as a
measurement rather than a projection (registry gap **R12** — closed).

**But not 2.57×, and the reason matters.** The 2026-07-18 figure was one *isolated* batch-1
predictor forward, where launch overhead is the entire cost. Inside a real tick the rollout's own
GPU work is not negligible: the rollout stage goes **95.03 → 28.73 ms** for 20 steps, i.e.
**4.75 → 1.44 ms/step**. The graph removes **3.31 ms/step of launch overhead out of 4.75** — a
**3.31× rollout speedup** — which is then diluted to 1.75× end-to-end by the eager encoder still
sitting in front of it.

### 2.1 The CPU round-trips are **not** the cost — an assumption refuted

| variant | p50 | difference |
|---|---|---|
| `graph_step` (20 replays, 19 host round-trips) | 57.33 ms | — |
| `graph_rollout` (1 replay, 0 round-trips) | 57.18 ms | **0.147 ms**, i.e. **0.0077 ms/step** |

Both replay the *same* kernels; the only difference is that `graph_step` returns to the CPU 19
times and copies each Δpose out of the private pool. That costs **7.7 µs per step**. The
expectation going in was that removing the round-trips would be a large second win. It is not —
**essentially the entire L1 gain is per-kernel launch elimination.**

This is *good news for deployment*: the far simpler single-step capture is worth 99.7 % of the
full-rollout capture, so a runtime that cannot capture a 20-iteration Python loop loses almost
nothing. `graph_fulltick` (encoder inside the graph too) is likewise **57.34 ms — no gain over
`graph_rollout`**, because the encoder is compute-bound and has nothing to reclaim.

### 2.2 `torch.compile` on Linux **beats** manual capture — the opposite of the dev-box finding

| path | fp32 p50 | tf32 p50 | bit-identical? |
|---|---|---|---|
| manual `torch.cuda.CUDAGraph` | 57.18 | 40.26 | **yes, exactly 0.0** |
| `torch.compile(mode="reduce-overhead")` | **52.89** | **36.42** | no — 3.8e-6 m / 6.1e-4 m |
| `torch.compile(backend="cudagraphs")` | 58.24 | 41.16 | yes, 0.0 |

On the Windows dev box `reduce-overhead` was unavailable (`TritonMissing`) and the
`cudagraphs` backend was **~20× slower**. On the pod (Linux, torch 2.8.0+cu128, Triton 3.4.0)
both work, and inductor's fusion buys a further **7.5 % (fp32) / 9.5 % (tf32)** over manual
capture — it removes launches *and* fuses elementwise work the graph merely replays.

**Recommendation: manual capture is still the deployment default.** It is bit-identical, has no
Triton dependency, no compile-time, and no risk of a silent recompile on the vehicle. Take
`reduce-overhead` only where the extra ~4 ms is needed and a 3.8e-6 m deviation is acceptable.

### 2.3 Numerical equivalence — exact

Every manually captured variant (`graph_step`, `graph_rollout`, `graph_fulltick`,
`drop_horizons_graph`, `compile_cudagraphs`) returns **max abs deviation 0.0 m, Δ ADE 0.0,
cosine 1.0** on all 32 windows. Not "within float noise" — *identical*. Expected: a graph replays
the same kernels in the same order.

**The standing worry that capture is blocked was wrong.** The rollout allocates ~38 tensors per
tick (`metric_dynamics.py:241-242`, the window-shift `torch.cat`s) and the predictor rebuilds its
causal mask every call (`predictor.py:112`). Neither blocks capture: allocations *inside* a
capture are served from the graph's private pool and replay at the same addresses. Capture
succeeded first time on every variant, with **zero build errors**. The real hazard is the
opposite one — a capture that succeeds and silently returns *stale* numbers — which is why the
test suite pins that a graph fed a different input returns a correspondingly different output.

Consequence: **hoisting the causal-mask allocation is UNTESTED here and not worth doing.** Inside
a graph the allocation costs nothing at replay; it would only pay in the eager path, which is the
path this note recommends against.

### 2.4 The tail — and a correction

| variant (fp32, 5 replicates × 200 iters) | p50 spread | p99 spread | tail ratio p99/p50 | worst iteration |
|---|---|---|---|---|
| `eager` | 1.55 % | 1.21 % | **1.038** | **107.2 ms** |
| `graph_rollout` | 0.57 % | 0.47 % | **1.0041** | **58.4 ms** |
| `graph_fulltick` | 0.14 % | 0.18 % | 1.0042 | 58.0 ms |

The graph **collapses the within-run tail ratio from 3.8 % to 0.4 %** and halves the run-to-run
p99 spread. For a control loop that is the point: the graphed tick's *worst observed iteration
out of 1,000* is 58.4 ms, comfortably inside budget, whereas the eager tick's is 107.2 ms —
already over.

⚠️ **The reported "26 % run-to-run p99 spread" did not reproduce.** In these five back-to-back
replicates the eager p99 ranged 102.15–103.39 ms, a spread of **1.21 %**, not 26 %. The 26 %
comes from `eff_repeatability.json` (p99 107.93–135.76 across 5 reps), taken in a session where
a second arm was loaded and freed between replicates. Both runs record
`gpu_exclusive: true`, so the difference is not a neighbour process — most likely allocator /
memory-state churn from the model swap. **Treat the 26 % as a property of that session, not of
the arm**, and do not quote it as v1's tail behaviour. What *is* reproducible, and what the
deployment argument needs, is the ratio collapse above.

---

## 3. L2 — encoder cache: the **projection was wrong by 4.5×**

The baseline JSON projects `plan_step_ms_if_encoder_cached = 84.74 ms` from stage arithmetic
(tick − encode_window + encode_1frame). Measured end-to-end, with the rolling cache implemented
and its bookkeeping included:

| | projected | **measured** | actual saving |
|---|---|---|---|
| `enc_cache` (fp32) | 84.74 ms | **95.11 ms** | **5.2 ms**, not the 23.4 ms the stage arithmetic implies |

**L2 as a standalone lever is refuted: 1.05×, and it still misses 10 Hz.** The reason is exactly
the effect the baseline's own `stage_sum_note` warns about — in the full tick the encoder's large
kernels run *while the CPU is already racing ahead launching the rollout's tiny ones*. Removing
7/8 of the encoder therefore does not remove 7/8 of the encoder's isolated time; it mostly
uncovers launch latency that was previously hidden.

**But L2 is worth a lot once L1 has run.** With the rollout captured, the CPU is no longer
running ahead, so the encoder cost becomes real and removable:

| | tick | saving from adding L2 |
|---|---|---|
| `graph_rollout` | 57.18 ms | — |
| `enc_cache_graph` | **33.27 ms** | **23.9 ms** |

So the levers are **not additive — they are sequenced**. L2 is worth 5 ms before L1 and 24 ms
after it. Anyone reading the baseline's projection in isolation would have deployed the cache
first and got 5 % instead of 41 %.

Equivalence: **1.9e-6 m** max deviation, Δ ADE 1.5e-8 m. Not bit-identical, and that is expected
rather than a bug — per-frame encoding runs the ViT at batch 1 while `encode_window` runs it at
batch 8, and cuBLAS picks different reduction orders. 1.9 µm on a 30 m trajectory is float noise.

---

## 4. L3 — fp16 weights, and the orthogonality claim tested on the A40

**Stage-level, which is where the answer lives:**

| stage | fp32 | fp16 weights | speedup |
|---|---|---|---|
| `encode_window` (8 frames) | 28.13 ms | **7.38 ms** | **3.81×** |
| `rollout` (20 steps) | 95.03 ms | 94.04 ms | **1.01×** |
| `rollout` under a CUDA graph | 95.03 ms | 28.73 ms | **3.31×** |

**The 2026-07-18 orthogonality claim is CONFIRMED and now quantified on the A40:** precision is
the *encoder* lever (2.1–3.8×), the graph is the *rollout* lever (3.3–3.5×), and **precision does
literally nothing for the rollout (1.01× fp32-block, 0.97× tf32-block)**. A launch-bound
dependent chain cannot be helped by making its arithmetic cheaper.

End-to-end this makes fp16 look worthless on its own — `fp16_eager` is **1.02×** — for the same
sequencing reason as L2: the encoder's time is hidden behind rollout launches. After L1 it is
worth **32 ms** (57.18 → 24.76).

**Autocast is confirmed slower, within a single block:** `autocast16_eager` **112.14 ms, 0.89×** —
*slower than fp32 eager*, and it also has the highest peak activation (374 MB vs 76 MB) because
every op materialises a cast copy. The baseline's cross-block `amp16 > tf32` observation was not
an artifact. **fp16 weights, never autocast.**

**Accuracy cost of fp16** (the composed rows): max abs deviation **0.024 m**, mean waypoint shift
**3.4 mm**, shift at 2 s 4.5 mm, **Δ ADE −6.6e-5 m** (i.e. 0.07 mm *better* on these windows —
noise, not an improvement). For scale, the 2026-07-17 fp16 policy accepted 3.9 cm mean shift on
the K=9 decision task; the trajectory rollout is an order of magnitude tighter than that bar.

A free refinement: **`fp16_fp32acc` halves the deviation** (0.0241 → 0.0127 m) by doing only the
SE(2) dead-reckoning in fp32. About half of fp16's error is the accumulator, not the network —
fp16's spacing at 30 m is ~0.03 m. The accumulate is ~20 trivial ops, so **keep the SE(2)
accumulation in fp32 in any deployed fp16 build**; it costs nothing measurable and buys 2× on
the deviation.

---

## 5. L4 — composition, and the 10 Hz verdict

| stack | fp32 p50 / p99 | × | meets 10 Hz @p99 |
|---|---|---|---|
| eager | 100.29 / 113.98 | 1.00 | **no** |
| + L1b graph | 57.18 / 57.45 | 1.75 | yes |
| + L2 cache | 33.27 / 33.42 | 3.01 | yes |
| + L3 fp16 | 19.18 / 19.19 | 5.23 | yes |
| + L7 pruned heads | **18.75 / 18.76** | **5.35** | **yes — 53.3 Hz** |

**The composed tick meets 10 Hz at p99 with 5.3× headroom.** Note the p50/p99 gap essentially
vanishes (18.75 → 18.76 ms): once everything is captured, the tick is deterministic.

Composition here is **not additive** — the 2026-07-18 "levers compose additively" result does not
generalise from a 1-step select to a 20-step rollout. Each of L2, L3 and L7 is worth ~nothing
standalone (1.05×, 1.02×, 1.00×) and a great deal after L1, because L1 is what stops the CPU
launch stream from hiding everything behind it. **Order matters: capture first, then precision
and caching.** Peak activation also drops 76 MB → 9.2 MB, which is the number that matters on an
embedded target.

---

## 6. L5 — rollout length. **LATENCY ONLY**

### (a) Truncating the k=1 roll

| k | eager fp32 | graphed fp32 | eager tf32 | graphed tf32 |
|---|---|---|---|---|
| 20 | 99.74 | 57.25 | 93.68 | 40.41 |
| 10 | 56.94 | 42.78 | 48.86 | 27.82 |
| 5 | 36.81 | 35.60 | 25.84 | 21.51 |

### (b) Striding with the **already-trained** k=2 / k=4 heads (no retraining needed)

| head | predictor calls for 2 s | eager fp32 | graphed fp32 | × vs k=1 eager |
|---|---|---|---|---|
| k=1 | 20 | 98.56 | 57.53 | 1.00 |
| k=2 | 10 | 58.36 | 42.88 | 1.69 |
| k=4 | 5 | 36.92 | 35.69 | 2.67 |

(tf32: 94.15 / 49.12 / 26.54 eager, 40.42 / 27.81 / 21.53 graphed — up to 3.55×.)

⚠️ **No accuracy verdict may be read from either table.** (a) shortens the horizon, so it predicts
something different. (b) reaches the *same* 2 s horizon using heads that already ship in the
checkpoint — but the 13 M step readout was calibrated on **0.1 s** transitions and would be
decoding 0.2 s / 0.4 s ones, so the decoded trajectory is invalid until the readout is
recalibrated against a **frozen** predictor. Both tables supply the latency side only.

Note also that striding buys much less *after* the graph (k=4 graphed 35.69 vs k=1 graphed 57.53
= 1.61×, against 2.67× eager): once launches are gone, the remaining cost is real arithmetic that
striding cannot remove. **L1 is the lever; striding is a fallback if L1 is unavailable on the
target runtime.**

---

## 6b. The planning fan — imagine-and-select is **affordable**, and the 723 ms arithmetic is wrong

The arithmetic that makes CEM look impossible — *8 candidates × 20 steps ≈ 723 ms* — assumes the
fan costs **8 sequential ticks**. It does not. A K-candidate fan is **one rollout at batch K**: the
same 20 sequential predictor calls, each doing K× the work *per kernel*. On a launch-bound pass
that is nearly free. (Same mechanism as 2026-07-18's "batch dilutes the graph's gain", read in the
other direction.)

In a real planner all K candidates also share **one observation history** and differ only in the
action sequence, so the encoder runs **once** and only the rollout fans out. `fan_shared_encoder`
measures exactly that tick (encode at batch 1, broadcast the state, fan the rollout to K); it is
**bit-identical to `all_levers` at K=1** by construction, and the measurement confirms it
(18.78 vs 18.78 ms).

Measured, fp32, 50 iters, `eff_levers_fan_sharedenc_flagship-30k.json` (`valid: true`):

| K | eager | `all_levers` (re-encodes K windows) | **`fan_shared_encoder`** (the planner tick) | p99 | ms / candidate | naive K× |
|---|---|---|---|---|---|---|
| 1 | 95.16 | 18.78 | **18.78** | 21.96 | 18.78 | — |
| 2 | 124.29 | 20.03 | **19.37** | 21.93 | 9.69 | 37.55 |
| 4 | 171.01 | 21.96 | **19.58** | 22.11 | 4.90 | 75.10 |
| **8** | 282.69 | 26.46 | **20.82** | **23.72** | **2.60** | **150.20** |
| 16 | 486.72 | 41.03 | **28.65** | 31.72 | 1.79 | 300.41 |
| 32 | 924.59 | 55.27 | **28.41** | 30.55 | 0.89 | 600.81 |

**An 8-candidate imagine-and-select fan costs 20.82 ms p50 / 23.72 ms p99 — not 723 ms.** That is
**2.04 ms more than a single tick**: the marginal cost of an extra candidate is **~0.3 ms**.
A **32-candidate** fan costs **28.41 ms**, still under a third of the 10 Hz budget.

The 723 ms figure comes from multiplying a single tick by K. That multiplication is the error: a
fan is not K ticks, it is one tick with K× the work *inside each already-launched kernel*, and the
kernels were never arithmetic-saturated to begin with.

Two further reads from the same table:

1. **The graph's advantage evaporates as K grows** (1.96× at K=1 → 1.02× at K=16 in the companion
   `graph_rollout` sweep) because batching dilutes launch-boundedness — the 2026-07-18 mechanism,
   confirmed across a full sweep. The composed stack does *not* evaporate: at high K the dominant
   cost becomes the **encoder**, which the cache and fp16 remove. **At batch 1 the lever is the
   graph; at fan scale the lever is the encoder.**
2. **Sharing the encoder is worth more the wider the fan** — 0 ms at K=1, 5.6 ms at K=8, 26.9 ms at
   K=32 — so a planner implementation that naively re-encodes per candidate throws away most of
   the headroom. Encode once, broadcast the state.

⚠️ Latency only. Which candidate a planner selects, and whether selection improves driving, are
closed-loop questions this note does not touch.

---

## 7. L7 — dropping the unused horizon heads: refuted standalone, marginal composed

`horizons=(1,2,4)` means `predictor.forward` computes three state-dim readouts per call and the
rollout discards two, 20× per tick. Pruning to the consumed head is bit-identical (**0.0 m**) —
and worth **nothing on its own: 100.47 vs 100.29 ms, 0.998×**. The isolated rollout stage does
move (95.03 → 90.19 ms, 5.1 %), but that saving is again hidden behind the launch stream.
Composed after the graph it is worth **0.56 ms** (57.18 → 56.62), and inside `all_levers`
**0.43 ms**. Keep it — it is free and exact — but it is a rounding correction, not a lever.

---

## 8. Corrections to prior claims

| claim | status after this run |
|---|---|
| "predictor CUDA-graph 2.57×" transfers to the rollout | **partly.** The *rollout stage* gets 3.31×; the *tick* gets 1.75× (fp32) / 2.35× (tf32). The dilution is the eager encoder, not the graph. |
| `plan_step_ms_if_encoder_cached = 84.74 ms` | **REFUTED.** Measured 95.11 ms. Stage arithmetic over-estimates the saving 4.5×. |
| "the levers compose additively" (2026-07-18, 1-step select) | **does not generalise.** On the 20-step rollout the levers are *sequenced*: L2/L3/L7 are worth ~1.0× before L1 and 1.7–3.0× after it. |
| `torch.compile` is not the graph route (Windows) | **platform-specific.** On the Linux pod `reduce-overhead` is the *fastest* variant measured (1.90×), beating manual capture — but it is not bit-identical. |
| "~40 allocating `cat`s / mask rebuild block capture" | **REFUTED.** Zero build errors; allocations inside a capture come from the graph's private pool. |
| v1's p99 swings 26 % run to run | **did not reproduce** (1.21 % here). Session property, not arm property — see §2.4. |
| flagship v1 misses 10 Hz | **true only for the eager tick.** Optimised: 18.76 ms p99 = 53 Hz. |

**Bounds.** Batch 1, single stream, A40, one checkpoint, 200 iterations × 5 replicates, 32
equivalence windows from 2 val episodes. Latency is architecture- not weight-determined, so these
hold across training progress but **not** across architectures. Δ ADE is an equivalence delta on
32 windows, not a heldout re-score — a lever with a *real* accuracy cost would still need the
canonical 40-episode harness. No Jetson/Orin claim is made: A40 numbers do not transfer to Orin,
whose launch path and memory bandwidth differ, and the sibling desk-research task owns that.

---

## 8b. Cross-reference with the Orin/Thor desk-research note (same date, same tree)

`2026-07-20-orin-thor-deployment-and-inference-levers.md` computes a **memory-bandwidth floor**
for the rollout: 7.31 GB of weight traffic per tick in fp32, 3.66 GB in fp16. Against the A40's
696 GB/s that is a floor of **10.5 ms fp32 / 5.3 ms fp16**. Two checks against what was measured
here:

- **Measured graphed rollout (fp32): 28.73 ms — 2.7× above its bandwidth floor.** So the capture
  removes the launch overhead but does *not* take the rollout to the memory wall; roughly two
  thirds of what remains is still something other than weight streaming. Consistent with that
  note's "launch-bound with headroom" reading, and it bounds how much a further scheduling trick
  could possibly buy on this device.
- The two notes reach **different lever orderings, for different devices, and that is not a
  conflict.** On the A40 the order is unambiguously *graph first, precision second* (fp16 alone
  1.02×; after the graph it is worth 32 ms). The Orin note argues precision comes first there
  because Orin's arithmetic is ~7× slower, so the compute-bound encoder dominates and launch
  overhead is a smaller *fraction* of a larger total. **Nothing measured here can settle the Orin
  ordering** — A40 numbers do not transfer, and no attempt is made to transfer them. What this run
  does establish, and what should carry across, is the *structural* finding: **these levers are
  sequenced, not additive**, so any port must re-measure the ordering rather than assume it.

---

## 9. Recommendations

- **A1 (deploy, decisive).** Capture the operative rollout as a CUDA graph. Bit-identical, free,
  and it is the difference between missing and meeting 10 Hz at p99. Use **manual
  `torch.cuda.CUDAGraph`** — `graph_step` (single-step capture, replayed) is worth 99.7 % of the
  full-rollout capture, so even a runtime that cannot capture the whole loop gets the win.
- **A2 (deploy, sequenced after A1).** Then add the encoder cache (−24 ms) and fp16 weights
  (−32 ms), in that order, and keep the SE(2) accumulation in fp32. Composed: **18.76 ms p99**.
  Do **not** deploy L2/L3 first and conclude they do not work.
- **A3 (do not).** Autocast — measurably slower and 5× the activation memory. `bf16` remains
  rejected on the 2026-07-17 accuracy evidence.
- **A4 (registry).** §1.2's planning-tick row and gap **R12** can both be closed with the numbers
  above. R12's prediction that whole-rollout capture would beat 2.57× is **not** what happened —
  worth recording, since it was reasoned from the launch-bound diagnosis and the diagnosis was
  right about the *mechanism* but wrong about the *magnitude*.
- **A5 (accuracy owner).** Only the equivalence delta is measured here. If fp16 is deployed, a
  single canonical 40-episode re-score at fp16 would convert Δ ADE −6.6e-5 m from "unchanged on
  32 windows" to a decision-grade statement.
- **A6 (next lever).** The remaining 18.76 ms is **real arithmetic, not overhead** — the captured
  rollout dominates it, and even in fp32 a captured rollout still costs 28.73 ms of pure GPU work.
  (No exact split is quoted: isolated stage timings provably do not sum to the end-to-end tick on
  this workload — that is the whole §3 finding — so decomposing 18.76 ms into stage numbers would
  repeat the mistake this note corrects.) Further gains need TRT/INT8 on the encoder or a shorter
  serial chain (§6, gated on readout recalibration), not another scheduling trick.
- **A7 (planning — this changes a design decision, not just a budget).** The "8 candidates ×
  20 steps = 723 ms" arithmetic that made imagine-and-select look unaffordable is **wrong by 35×**:
  the measured 8-candidate planner tick is **20.82 ms**, and the marginal candidate costs
  **~0.3 ms**. Whoever owns the planner should size the fan on this table, not on multiplication —
  and should implement it as **one batched rollout with a shared encoder pass**, since re-encoding
  per candidate is what actually costs (5.6 ms at K=8, 26.9 ms at K=32). The open question for
  CEM is no longer latency; it is whether selection improves closed-loop driving, which the
  closed-loop harness owns.
