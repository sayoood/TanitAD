# Thor vs A40 — v6F S-W training speed, MEASURED

**2026-08-16 · answers the PI's question: *"measure the speed of training and compare it to the A40."***
Arm: `v6F-SW-30k`, config E, **336,542,025 params**, strict resume from step 6,250 of 30,000.

---

## 1. The number

| | marginal s/step | window | evidence |
|---|---|---|---|
| **Thor** (Jetson Thor, `tanitad-thor-wifi`) | **27.18** | 100 steps, 6300 → 6400 (3 rows) | MEASURED — `~/experiments/v6F-SW-30k/train_log.jsonl` |
| **A40** (the machine that trained 0 → 6300) | **20.46** | its last 100 steps, **matched width** | MEASURED — same file, earlier process segment |
| **ratio** | **1.329× — Thor is 32.9 % SLOWER** | | |

**Projected remainder (23,600 steps): Thor 7.42 days · the A40 would have needed 5.59 days.**

⭐ **The rate is stable, and that is measured rather than hoped.** Three statistics with different
startup exposure and different window widths agree to **0.5 %**: marginal over 50 steps **27.21**,
marginal over 100 steps **27.18**, cumulative including startup **27.32**. The first reading was
taken at 2 rows and reported as provisional; the 3-row reading moved it by **0.03 s/step**.

### Why these are the right two numbers, and not the obvious ones

⚠️ **`step_s` in this trainer is a CUMULATIVE MEAN, not a per-step time.**
`train_v6_staged.py:1406` computes `(now − t0) / (step − start_step)` and writes a `step_s_note`
naming its own divisor. On a resume the first rows are inflated by startup, and across a whole run
the value hides any drift. Both sides here are therefore **marginal**, from the same identity:

```
T(S) = step_s(S) · S            S = steps THIS process ran
marginal[S1,S2] = (T(S2) − T(S1)) / (S2 − S1)
```

⚠️ **The A40's own headline figure would have understated it.** Its lifetime cumulative is
**17.46 s/step**, but its marginal over its last 300 steps is **19.68** and over its last 50 is
**20.20** — the A40 was **13 % slower at the end than its own average**, because the `o5` rollout
grows during training. Comparing Thor's marginal to the A40's *cumulative* would have reported a
**1.56×** deficit instead of the true **1.33×**. Thor's rows sit at steps 6300–6400, so the A40's
matched band is its own 6300–6400 — same steps, same loss schedule, same `o5_k`.

⛔ **The step numbers OVERLAP and cannot be used to separate the machines.** `train_log.jsonl` is
appended across processes: the banked A40 log ends at **6300** and the Thor resume starts at
**6250**. A `step > 6250` filter silently computes a rate *across two machines*. The segmentation
here is on the note's own divisor resetting — the producer, not the step. This is written up as
**RETRACTION_LOG C68**; two of my own drafts had the defect before it was caught.

## 2. The resume is sound — a free continuity check

| step | loss | `o1_factual_ade` | `gnorm` | `o5_loss` |
|---|---|---|---|---|
| A40 6200 | 2.8722 | 2.3064 | 723.2 | 0.21300 |
| A40 6250 | 2.1450 | 1.3875 | 515.2 | 0.20251 |
| A40 6300 | 3.2959 | 3.0941 | 562.6 | 0.21146 |
| **Thor 6300** | **2.4341** | **1.7207** | **488.8** | **0.20063** |

Thor's step-6300 loss sits **inside the A40's own step-to-step spread** (2.145 – 3.296), and
`o5_loss` — much the slowest-varying term — is continuous at the bottom of the A40's band.
⚠️ These are **different batches** (the data order does not replay across a resume), so this is
**evidence against corruption, not proof of bit-identity**. The bit-level claim is the separate
strict-load check: 573 tensors, `load_state_dict(strict=True)` OK.

## 3. ⭐ The number is "Thor AS CONFIGURED", not "Thor" — and there is an unused lever

MEASURED on the live box while it trained:

| probe | reading |
|---|---|
| `nvpmodel -q` | **mode 1 = "120W"** (mode **0 = MAXN** exists and is unused) |
| `gpu-gpc-0` (GPU core) | cur **1386 MHz** / max **1386 MHz** — **already at its ceiling** |
| `bwmgr` (EMC / memory bandwidth) | cur **3200 MHz** / max **4266 MHz** — **25 % unused** |
| EMC sampled 5× over 5 s | `3200000000` every time — **pinned, not oscillating** |
| `power.draw` | **30.75 – 33.50 W** of a **120 W** budget |
| `tj-thermal` | **58.5 °C** — nowhere near throttling |
| `%Cpu` idle | 99.4 % — the trainer is not CPU-starved |

**The GPU core is already maxed, so the only headroom is memory bandwidth** — and a 336 M
transformer at batch 8 on 20 SMs with `--grad-checkpoint` ON is precisely the workload where EMC
binds, because gradient checkpointing trades compute for *more* memory traffic (recompute plus
re-read of activations).

⚠️ **This is C14's family and it must be labelled.** C14's lesson is *"before recording a limit, ask
whether the instrument could have reported a LARGER value."* Quoting **27.18 s/step as "Thor's
speed"** would record **our own power configuration as the hardware's capability**. Until the lever
below is tried, the honest statement is: *Thor is 32.9 % slower than the A40 **in nvpmodel 1 with
EMC at 3200/4266**.*

### 🟥 BLOCKED ON THE PI — one command

`jetson_clocks` pins clocks to max **within the current nvpmodel**; it signals no process, restarts
nothing, and is reversible. `/sys/class/devfreq/bwmgr/min_freq` is `root:root rw-r--r--` and a
non-root write is denied; the `nvidia` user is in the `sudo` group but **`sudo` requires a password**,
so I cannot run it autonomously. *(Checked before attempting — a blind `sudo` inside a piped ssh
would have hung on the password prompt and looked exactly like a stall.)*

```bash
ssh tanitad-thor-wifi 'sudo jetson_clocks --store ~/jetson_clocks.before.conf && sudo jetson_clocks && cat /sys/class/devfreq/bwmgr/cur_freq'
```

Expected: `cur_freq` goes **3200000000 → 4266000000**. Revert with
`sudo jetson_clocks --restore ~/jetson_clocks.before.conf`.

⛔ **`nvpmodel -m 0` (MAXN) is NOT recommended unattended** — on some Jetson platforms it prompts for
a reboot, and a reboot kills a 336 M training run. That stays a deliberate PI decision.

⚠️ **The experiment must not average through the change.** The clean "before" window is already
banked (6300 → 6400, 27.18 s/step over 100 steps). The interval containing the change **must be
DISCARDED**, and
the first admissible "after" interval is the one whose *both* endpoints are logged after it — the
same C68 discipline as the machine boundary. A ready-to-run script that records the exact step and
both `/sys` snapshots is at `code/thor_clocks_experiment.sh`.

## 4. Caveats stated rather than buried

1. ~~One interval.~~ **RESOLVED** — now two intervals / three logged points, agreeing to 0.5 %
   (see §1). The A40 side moved slightly too (20.20 → **20.46** on the matched 100-step window),
   which is why the ratio settled at 1.329× rather than 1.347×: **both sides must be re-cut to the
   same width when the width changes**, and the estimator does that rather than holding one side
   fixed.
2. **Batch and workers are already right for Thor** and were not the lever: CLAUDE.md records that
   Thor saturates at **batch 8** (throughput flat across a 6× batch range) and that each dataloader
   worker costs **~8.6 GB host RAM**. The live command runs `--batch 8`.
3. **`--grad-checkpoint` is a second, larger lever and is NOT tested here.** Turning it off would cut
   memory traffic materially, but it needs a restart and a memory-ceiling check — and on Thor only
   `torch.cuda.max_memory_allocated()` is admissible for that (`mem_get_info`, `free`, `tegrastats`
   and `VmRSS` all lie, in both directions). Separate experiment, separate pre-registration.
4. **Durability gap, expected but worth naming:** `--save-every 250` from 6250 means the first new
   checkpoint lands at step **6500** (~00:55 UTC). Until then every step past 6250 exists in GPU
   memory only; 6250 itself is safe on HF.

## Deliverable manifest

| artifact | where it lives |
|---|---|
| this document | `…/incoming/2026-08-15-v6-thor-resume/THOR_VS_A40_TRAINING_SPEED.md` |
| `thor_measure_sstep.py` (the marginal estimator, refuses across producers) | `…/incoming/2026-08-15-v6-thor-resume/code/` |
| `thor_clocks_experiment.sh` (E-THOR-CLK, ready to run) | `…/incoming/2026-08-15-v6-thor-resume/code/` |
| the raw log both sides come from | `thor:~/experiments/v6F-SW-30k/train_log.jsonl` ⚠️ **single disk until step 6500** |

## Escalation

🟥 **One PI command unblocks a measured 25 % memory-bandwidth headroom on a 7.45-day run.** If EMC is
the binding constraint, the payoff is days. If it is not, the experiment costs one command and closes
the question — and either way the 27.21 figure stops being ambiguous between *"Thor's speed"* and
*"the power mode Thor happened to boot in."*
