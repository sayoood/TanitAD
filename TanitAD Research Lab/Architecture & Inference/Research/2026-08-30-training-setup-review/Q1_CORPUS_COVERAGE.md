# PI Q1 — "does training contain ALL 26 h of driving?" → **YES**, and "0.29 epochs" is the wrong frame

**Date:** 2026-08-30 · **Owner:** TanitAD_TrainingFlyWheel · **Tier:** T0 training-side
**Evidence class:** MEASURED (ours) — simulation of the **actual sampler code**,
episode lengths read from the **actual caches**. ⛔ Nothing inherited, including
the 173 windows/episode I was asked to re-derive.

---

## 0. Verdict

| question | answer |
|---|---|
| Does the run see all 26 h? | ⭐ **YES — EPISODE coverage is 100.00 % in every configuration tested.** Every clip is drawn, ~25–50× over. |
| Is "~0.29 epochs" right? | ⚠️ **The arithmetic is right; the FRAME is wrong.** There is no epoch — see §1. |
| Is there a real coverage weakness? | ✅ **Yes, but a small one and not the one feared** — a tail of under-covered episodes on B1 at batch 8 (§3). |

## 1. ⛔ "EPOCHS" DOES NOT APPLY: the sampler draws i.i.d. WITH REPLACEMENT

`train_v58f_unicycle_head.make_sampler:368-378` — the v6 sampler — picks
`eps_per_batch` episodes with `rng.randrange(len(ep_ids))`, then each window with
`rng.randrange(len(pool))`. **It is not a shuffled epoch loop.** Consequently
`steps × batch / n_windows` is an **expected-draws ratio**, not a coverage
fraction: N draws from M items never touch N distinct items. Coverage is
coupon-collector, `1 − (1 − 1/M)^N`.

⇒ Quoting "0.29 epochs" invites the reading *"we see 29 % of the data"*. **We see
100 % of the episodes and ~96 % of the frames.**

## 2. The three coverage units — and why only one answers the PI

MEASURED inputs: episode length **199 frames** (median over 25 sampled episodes
from `physicalai-train-14231cd29c74` **and** the val cache; min 197);
`window = 6` (`V6Config().predictor.window`); `eps_per_batch = 4`; 30,000 steps.
⚠️ `max_h` is **stage-derived** (`train_v6_staged.py:5070`), so both plausible
values are reported rather than one being asserted canonical.

| corpus / config | draws / windows | EPISODE | WINDOW | FRAME mean | FRAME p1 | FRAME **min** |
|---|---|---|---|---|---|---|
| **max_h=20 → 173 win/ep** | | | | | | |
| parity, batch 8 (Thor live) | 240k / 411k = 0.58× | **100.00 %** | 44.00 % | 98.18 % | 93.84 % | 87.44 % |
| parity, batch 16 (default) | 480k / 411k = 1.17× | **100.00 %** | 68.59 % | 99.05 % | 96.98 % | 94.47 % |
| B1, batch 8 | 240k / 815k = 0.29× | **100.00 %** | 25.43 % | 96.41 % | 86.93 % | ⚠️ **64.82 %** |
| B1, batch 16 | 480k / 815k = 0.59× | **100.00 %** | 44.26 % | 98.19 % | 93.47 % | 84.92 % |
| **max_h=60 → 133 win/ep** | | | | | | |
| parity, batch 8 | 240k / 316k = 0.76× | **100.00 %** | 53.14 % | 98.59 % | 95.48 % | 90.45 % |
| parity, batch 16 | 480k / 316k = 1.52× | **100.00 %** | 77.79 % | 99.21 % | 97.99 % | 96.98 % |
| B1, batch 8 | 240k / 627k = 0.38× | **100.00 %** | 31.70 % | 97.26 % | 90.95 % | 85.43 % |
| B1, batch 16 | 480k / 627k = 0.77× | **100.00 %** | 53.12 % | 98.59 % | 95.48 % | 91.96 % |

Simulated WINDOW coverage tracks the coupon-collector analytic to **within 0.3 pp**
in every row — the simulation is doing what the arithmetic says.

⭐ **WHY WINDOW COVERAGE IS THE LEAST MEANINGFUL NUMBER.** Windows are built at
**stride 1** (`data/_contract.py:118-121`, `t_max = frames − window − max_horizon`),
so window `t` and `t+1` share all but one frame and each window spans
`window + max_h` frames. Missing a window does **not** mean missing the driving in
it. That is why 25 % window coverage still yields **96 % frame coverage**.

## 3. The real weakness, and it is a TAIL not a mean

On **B1 at batch 8** (the configuration the live Thor run would use), the
worst-covered episode sees only **64.8 %** of its frames and the 1st percentile is
**86.9 %**. Every episode is *visited*, but a small tail is thinly sampled.

⇒ If the PI's requirement is read strictly as *"every episode substantially
covered"*, the lever is **batch 16 rather than more steps**: it lifts B1's worst
episode 64.8 % → 84.9 % and p1 86.9 % → 93.5 % at the same step count.
⚠️ **But batch 16 is not free on Thor** — the standing measurement is that
throughput is FLAT across a 6× batch range because the 20 SMs saturate at batch 8,
so a larger batch buys coverage and costs memory, not time. That trade is the
per-host table's business (PI Q2), not this section's.

## 4. ⛔ WHAT THIS DOES **NOT** ESTABLISH — the caveat that could overturn it

This simulates **`make_sampler`**, the *uniform* sampler. The live run may use
**`InteractionSampler` / `StratifiedEpisodeSampler`** with **O4 weighting**
(`--o4-alpha`), which makes window draws **non-uniform by design**. Under
weighting, low-weight windows are drawn less and coverage of *those* windows falls
below every number above.

⇒ **These figures are the UNIFORM CONTROL — an upper bound on coverage for any
weighted sampler.** The next measurement is the same simulation driven by the
live arm's actual weights. ⛔ Until that is run, no claim may be made about the
coverage of a weighted arm, and I am making none.

⚠️ Also assumed uniform: 199 frames/episode (measured 197–199 on 25 episodes, so
the spread is small but not zero). A short-episode tail would lower the minimum.

## 5. Consequence for the scaled run

⛔ **Do not increase step count to "reach one epoch".** The premise is false —
there is no epoch, and episode coverage is already total. Steps should be chosen
by convergence, not by a division whose frame does not apply.
⚠️ The only coverage-motivated change worth considering is **batch 16 on hosts
where it is free**, and that is a throughput question with a per-host answer.
