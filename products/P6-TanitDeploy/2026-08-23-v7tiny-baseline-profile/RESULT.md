# RESULT — E-DEPLOY-1: v7-tiny (champ30k) baseline profile on Thor

`2026-08-23 · TanitAD_DeployFlyWheel · pre-registered in SPEC.md before any
measurement. Evidence class MEASURED (ours) unless stamped otherwise. Every
number below resolves to a raw JSON in raw/.`

## Headline

**The batch-1 deployment lever on Thor is CUDA graphs, not precision.** Graph
replay takes the v7-tiny tick from **13.020 ms → 7.840 ms (1.66×)** and the
output is **bit-identical** — so it needs no paired accuracy eval, because
bit-identity is a stronger guarantee than any eval could supply. fp16 autocast,
by contrast, is **21 % SLOWER than fp32 at batch 1** and moves the plan.

**All 5 pre-registered controls passed.** `raw/thor_v7tiny_profile.json →
all_controls_pass: true`.

| hypothesis | verdict |
|---|---|
| **H-DEPLOY-2** batch-8 saturation holds for inference | **SUPPORTED** — in all four configurations |
| **H-DEPLOY-3** fp16 wins at batch 1 | **REFUTED** — fp16 is 1.21× *slower* at b1 |
| **H-DEPLOY-4** reduced precision numerically safe (≤1e-2 m screen) | **REFUTED as written — and the test was MIS-SPECIFIED by me.** See §4 |

---

## 1. The measured grid

`raw/thor_v7tiny_profile.json` · Thor, torch 2.13.0+cu130, `tanitad-edge` venv ·
n = 50 timed + 20 warmup per arm · memory = `torch.cuda.max_memory_allocated()`
ONLY · p90 ≈ median throughout, so run-to-run noise is negligible.

| batch | dtype | median ms | p90 ms | windows/s | peak MiB | plan dev vs fp32 (max abs) |
|---|---|---:|---:|---:|---:|---:|
| 1 | fp32 | **12.529** | 12.570 | 79.8 | 171.6 | — (reference) |
| 1 | fp16 | 15.214 | 15.258 | 65.7 | 183.2 | 8.663e-02 m |
| 1 | bf16 | 15.257 | 15.317 | 65.5 | 183.1 | 2.618e+00 m |
| 4 | fp32 | 27.582 | 27.649 | 145.0 | 386.8 | — |
| 4 | fp16 | 19.857 | 19.938 | 201.4 | 432.3 | 5.519e-01 m |
| 4 | bf16 | 19.759 | 19.862 | 202.4 | 432.1 | 2.698e+00 m |
| 8 | fp32 | 47.844 | 47.948 | 167.2 | 672.0 | — |
| 8 | fp16 | 32.889 | 32.967 | 243.2 | 762.4 | 3.455e-01 m |
| 8 | bf16 | 32.992 | 33.069 | 242.5 | 762.0 | 4.936e+00 m |

### The tick decomposes almost perfectly into fixed + marginal cost

Least squares on the three fp32 arms: **t(B) = 7.45 ms + 5.05 ms × B**,
**R² = 0.99999** (residuals 0.031 / −0.055 / 0.023 ms). A batch-independent
**7.45 ms** sits under every tick. That constant is the whole optimisation
target at batch 1 — and §3 shows what it is made of.

## 2. H-DEPLOY-2 — SUPPORTED (batch-8 saturation holds for inference)

Pre-registered bar: windows/s at b8 ≤ 1.3 × b4.

| configuration | b4 win/s | b8 win/s | ratio | verdict |
|---|---:|---:|---:|---|
| fp32 eager | 145.0 | 167.2 | **1.15** | ≤1.3 ✅ |
| fp16 eager | 201.4 | 243.2 | **1.21** | ≤1.3 ✅ |
| fp32 graph | 168.3 | 179.7 | **1.07** | ≤1.3 ✅ |
| fp16 graph | 254.9 | 279.8 | **1.10** | ≤1.3 ✅ |

The trainer-era instinct transfers: **doubling batch 4→8 buys 7–21 % throughput
and costs ~74 % more memory** (386.8 → 672.0 MiB fp32). Note the *earlier* step
is where batching pays — b1→b4 is 1.82× — so the deploy default is **batch ≤ 4**,
not "batch as large as fits".

## 3. H-DEPLOY-3 — REFUTED, and the refutation names the real lever

Pre-registered bar: fp16 median ≤ 0.85 × fp32 at batch 1. **Measured 1.21×** —
fp16 is *slower*. The pre-registered outcome B said this would mean *"the model
is not matmul-bound at b1; the first optimisation lever is elsewhere
(kernel-launch overhead ⇒ CUDA graphs)"*. That prediction was then tested.

### CUDA graphs — `raw/thor_v7tiny_plan_and_cudagraph.json`

| batch | dtype | eager ms | graph ms | speedup | saved ms | bit-identical | win/s |
|---|---|---:|---:|---:|---:|---|---:|
| 1 | fp32 | 13.020 | **7.840** | **1.66×** | 5.18 | ✅ **yes** | 127.6 |
| 1 | fp16 | 15.238 | **5.674** | **2.69×** | 9.56 | ✅ yes | 176.2 |
| 4 | fp32 | 27.516 | 23.761 | 1.16× | 3.75 | ✅ yes | 168.3 |
| 4 | fp16 | 19.775 | 15.693 | 1.26× | 4.08 | ✅ yes | 254.9 |
| 8 | fp32 | 48.390 | 44.518 | 1.09× | 3.87 | ✅ yes | 179.7 |
| 8 | fp16 | 32.854 | 28.597 | 1.15× | 4.26 | ✅ yes | 279.8 |

Graph capture **succeeded on every arm**, and replay is bit-identical to eager
in every arm (`replay_vs_eager_max_abs` = 0.0, all six).

**The saving is a near-constant 3.75–5.18 ms regardless of batch** — exactly the
signature of a fixed per-tick launch cost, and it accounts for **70 %** of the
7.45 ms fixed term the affine fit found. At b1 that fixed cost is 60 % of the
whole tick, which is why the lever is worth 1.66× there and only 1.09× at b8.

⚠️ **Bit-identical is a claim about replay vs its OWN eager arm.** Graphed fp16
is bit-identical to eager fp16 — it does **not** inherit fp32's accuracy. The two
statements must never be merged.

## 4. H-DEPLOY-4 — REFUTED as written, and MY PRE-REGISTRATION WAS DEFECTIVE

Measured fp16 plan deviation 8.663e-02 m > the 1e-2 m screen ⇒ **REFUTED**.
I am recording that verdict, and then recording why it should not be acted on.

**Defect 1 — the threshold was set on the wrong scale.** I picked 1e-2 m without
first measuring the trajectory's extent. The waypoints reach **56–121 m**
(`waypoints_absmax`), so the fp16 deviation is **1.5e-3 to 5.6e-3 relative** —
0.15–0.56 %. A screen calibrated on the real reference would have been a
*relative* bar. This is precisely the failure the programme's own estimator rule
warns about (*thresholds calibrated on REAL references, never synthetic*).

**Defect 2 — the tensor I screened carries no learned behaviour.** Verified by
content (`raw/thor_v7tiny_plan_and_cudagraph.json → plan_contents`):

| plan field | shape | content |
|---|---|---|
| `a` (accel command) | [1,8,60] | **480/480 exactly zero** |
| `kappa` (curvature) | [1,8,60] | **480/480 exactly zero** |
| `controls` | [1,8,60,2] | **960/960 exactly zero** |
| `waypoints` | [1,8,60,2] | absmax 56.13, **480/960 zero** (all lateral) |

champ30k is a **stage S-W** run: `champ.sh` sets `--w-o1-ctrl 0 --w-o1-fact 0
--w-o1-scene 0 --w-o2 0 --w-o3 0`, so the planner was never trained and its
zero-init emission head still emits exactly zero. The "8-candidate fan" is
therefore **8 identical straight lines at constant v₀** — `waypoints` is
`unicycle_rollout(0, 0, v₀, dt)` and nothing else. **A precision screen run on
that tensor measures the integrator's arithmetic, not the network's.**

⇒ **H-DEPLOY-4 must be re-registered against a checkpoint whose planner is trained,
with a RELATIVE bar.** Until then no precision verdict for this stack is
admissible. The trunk deviations below *are* meaningful, because those tensors
are learned.

### Where the deviation is actually born — `raw/thor_v7tiny_mechanism.json`

| stage | fp16 max abs | fp32 ref absmax | bf16 max abs |
|---|---:|---:|---:|
| `z_op` (trunk latent) | 2.696e-02 | 8.57 | 6.602e-02 |
| `z_tac` | 4.488e-03 | 2.61 | 1.512e-02 |
| `z_str` | 4.644e-03 | 2.73 | 1.985e-02 |
| `plan.feat` | 1.514e-03 | 1.88 | 1.188e-02 |
| `plan.a`, `plan.kappa` | **0.0** | **0** | **0.0** |
| `plan.waypoints` | 8.663e-02 | 56.1 | 2.618e+00 |

⚠️ **Do not quote the `max_rel` column from the raw JSON as "the error is 324 %".**
It is dominated by near-zero reference elements against a 1e-6 clamp. The
admissible statistic here is max-abs against the tensor's own scale — naming the
statistic, not just the quantity.

### The mechanism, tested — `raw/thor_v7tiny_integrator_v2.json`

Because `a` and `kappa` are bit-identical zeros in both precisions, a waypoint
difference can only be born *inside* `unicycle_rollout`. Confirmed by calling it
standalone with dtype-varied controls:

| control dtype | waypoints out dtype | max abs dev | in ULP of out dtype | reproduces end-to-end? |
|---|---|---:|---:|---|
| fp16 | float16 | 4.169e-02 m | **0.67 ULP** | 0.48× of 8.66e-02 |
| bf16 | bfloat16 | 2.301e+00 m | **4.60 ULP** | **0.88×** of 2.618 |

The rollout **inherits the autocast dtype from its control inputs** and then
accumulates 60 steps of travelled distance in it; the deviation is 0.7–4.6 ULP of
the *output* dtype at ~65 m, and grows monotonically along the horizon
(fp16 t₀ 7.2e-04 → t₅₉ 1.2e-02 m).

⭐ **The fix is proven, not proposed**: upcasting the controls to fp32 before the
rollout returns a **bit-identical** fp32 result — `fix_upcast_before_rollout.
bit_identical: true` for **both** fp16 and bf16. Cost is ~zero: the integrator is
60 steps of elementwise arithmetic, invisible next to the network.

⚠️ The standalone drew a different v₀ (10.93 m/s vs the model's ≈9.36), so the
reproduction ratio is order-of-magnitude, not exact. The ULP counts and the
bit-identical fix do not depend on that.

### ⛔ RETRACTED IN-FLIGHT — my own first integrator probe

`raw/thor_v7tiny_integrator_v1_DEFECTIVE.json` measured **exactly 0.0** deviation
and appeared to refute the integrator hypothesis. It was wrong: it passed **fp32**
zeros for the controls, and autocast does not cast `cumsum`/`sin`/`cos` inputs, so
the rollout stayed fp32 — **the probe held fixed the very variable it existed to
vary.** Kept in `raw/` under its `_DEFECTIVE` name rather than deleted, because a
probe that reads a clean zero for the wrong reason is the most dangerous artifact
in the folder. Root-cause class: *control-variable not actually varied* — the same
family as a constant-control that was never wired.

## 5. What this licenses, and what it does not

**Licensed now (bit-identical, no eval needed):**
- Enable **CUDA graph capture** for the v7-tiny inference tick on Thor. 1.66× at
  b1, 1.09–1.26× elsewhere, output bit-identical, capture succeeded on all arms.
- Deploy default **batch ≤ 4**; b8 buys ≤ 21 % throughput for ~74 % more memory.

**NOT licensed:**
- ⛔ **No fp16/bf16 deployment claim.** fp16 is slower at b1, and its accuracy is
  unmeasured on a trained planner. Charter §4 stands: *a quantization without a
  paired eval is not a deployment*.
- ⛔ **bf16 is the worse of the two** here — 30× the fp16 deviation for no latency
  gain. That converges with the programme's independent 4060 result (bf16 decision
  agreement 67.2 %, 47.7 cm shift) and with G5 in the census. Two independent
  measurements now point the same way.
- ⛔ **These are not driving numbers.** champ30k's `gate_verdict` is
  `INCONCLUSIVE` and its planner is untrained; this profiles the *architecture's
  inference cost at 19.3 M params*, nothing about driving quality.

## 6. Reproduction recipe

```bash
scp thor_profile_v7tiny.py tanitad-thor:/home/nvidia/
ssh tanitad-thor "cd ~/TanitAD/stack && PYTHONPATH=/home/nvidia/TanitAD/stack \
  ~/venvs/tanitad-edge/bin/python ~/thor_profile_v7tiny.py \
  --out ~/v7tiny_profile.json --batches 1 4 8 --warmup 20 --iters 50"
```
Scripts are in `code/`. Preconditions: Thor GPU idle (verify with a full
`ps -eo args` grep for `python`, not a top-N sample), `~/v7tiny/champ30k/`
present. Each script re-derives the model from the run's own `config.json` and
aborts if the param count or state-dict does not match.

## 7. Deliverable manifest

| artifact | location | only one place? |
|---|---|---|
| Pre-registration | `repo:products/P6-TanitDeploy/2026-08-23-v7tiny-baseline-profile/SPEC.md` | no |
| 9-arm profile raw | `repo:…/raw/thor_v7tiny_profile.json` | no (also `thor:~/v7tiny_profile.json`) |
| Stage-deviation + first graph raw | `repo:…/raw/thor_v7tiny_mechanism.json` | no |
| Plan contents + graph grid raw | `repo:…/raw/thor_v7tiny_plan_and_cudagraph.json` | no |
| Integrator probe v2 raw | `repo:…/raw/thor_v7tiny_integrator_v2.json` | no |
| ⛔ Defective probe v1 (kept as evidence) | `repo:…/raw/thor_v7tiny_integrator_v1_DEFECTIVE.json` | no |
| Profiler + 3 probe scripts | `repo:…/code/` | no (also `thor:~/`) |
| champ30k checkpoint | `thor:~/v7tiny/champ30k/` | 🔴 **YES — one disk** |

🔴 **champ30k exists only on Thor.** 133 MB, 4.1 h of compute, and it is the first
collapse-free trunk. Escalated in COMMS.md — the owning agent should push it to HF
(`Sayood/`) as the other arms were.
