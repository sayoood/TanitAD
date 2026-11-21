# E1f feasibility — the junction-restricted buffer is BUILDABLE, and its risk is 20.7 %

**MEASURED 2026-07-28**, pod3, CPU only (756 KB file, no GPU, nothing else running).
This is a **feasibility probe, not an experiment** — no arm was trained and no outcome is claimed.

## Why check this now

E1d MEASURED that the closed-loop primary's two halves behave completely differently: **junction
recovery is cheap and monotone** (separated-better at α=0.20 for +6.5 % open-loop cost) while
**overall-corridor recovery is expensive and barrier-crossing**. The pre-registered fallback if the
`lam_replay` axis closes is therefore to change **what** is supervised, not how heavily — a
junction-restricted buffer.

⚠️ **That plan was, until now, an assumption: that the buffer can be filtered to junction states at
all.** If it could not, the fallback would have collapsed the moment E1e-B returned — after the GPU
was already spent. Checking it while B trains costs nothing and removes that risk.

## What the buffer actually contains (MEASURED)

`/workspace/e1b/mined_buffer.pt`, md5 `a32cfe9bfea4b1b5c196d3bb7f71fa5f` (the same file E1c, E1e-A and
E1e-B all assert at train time):

- **3,537 records**, **362 distinct episodes**
- per-record keys: `dlat`, `dpsi`, `ep_path`, `episode_id`, `k`, `k_cross`, `lat_k`,
  `recovery_target`, `slice_start`, `v0`
- meta: `K = 185`, `corridor_halfwidth_m = 1.75` — matching the evaluator's rollout config

⭐ **`dpsi` (heading change) is present**, which is precisely how the evaluator defines a junction
(`--junction-deg 10.0`).

**Units settled by measurement, not assumption:** |dpsi| max is **0.8272**, and **0 records** exceed
10 in raw units ⇒ `dpsi` is in **radians** (0.8272 rad ≈ 47.4°). Distribution:

| stat | \|dpsi\| |
|---|---|
| min | 0.0052 |
| median | 0.0632 |
| p90 | 0.3156 |
| p99 | 0.6501 |
| max | 0.8272 |

## The answer

**At the evaluator's own junction threshold (10° = 0.1745 rad): 733 of 3,537 records survive =
20.7 %.** So the filter is buildable, uses the *same* threshold the primary metric uses, and needs no
new mining run.

## 🔴 The risk this surfaces, to be pre-registered before any GPU

**A 4.8× smaller buffer means 4.8× more reuse per record.** E1c/E1e draw `4000 steps × cl-batch 16 =
64,000` closed-loop samples; against 3,537 records that is ~18× reuse per record, against 733 it
becomes **~87×**. That is a memorisation regime, and it is the same failure that bounded GATE-1
("only ~13–22 real junction episodes ⇒ leave-3-out held-out Δ≈0").

⇒ **E1f's pre-registration must therefore fix, in advance:**
1. a **held-out-by-episode** check on the junction subset (the 733 records span 362 episodes — the
   per-episode count must be reported, not just the record count);
2. whether to **shorten the run** or **lower `cl-batch`** so reuse stays comparable to E1c's ~18×,
   since otherwise a "win" could be memorisation rather than transfer;
3. the honest alternative that **the junction gain E1d measured may not survive restriction at all** —
   α-interpolation moved a model trained on *everything*, which is not the same object as a model
   trained only on junctions. **E1f is a new experiment, not a confirmation of E1d.**

## Status

**NOT LAUNCHED and NOT YET PRE-REGISTERED.** E1f only becomes the live next step if E1e-B returns
BOUND; if B delivers a success point, this file is superseded and the `lam_replay` result is the
deliverable. Recorded now so the fallback is costed rather than assumed.
