# P2-s1: the training-seed replicate of P2 (pre-registration)

> **IN FORCE on landing.** Written 2026-10-14, **before any replicate step exists**: no chain, no token, no run directory, no checkpoint.
> After landing, nothing below changes without a dated amendment written before the read it affects.
>
> **Ruling.** PI, 2026-10-14, verbatim: *"Replicate p2"*.
>
> **Why the replicate is owed.**
> - `eval/PREREG_P2.md` registers it on ADOPT: *"A training-seed replicate (H-ESTIM-SEED-1) is then the next arm, before the lever is claimed"* and *"ADOPT: the seed replicate, then P6"*.
> - Amendment 1 replaced only the NO-GO clause, so this clause still binds.
> - P2 was ADOPT (R1) and GO (R2: +0.57 PDMS [+0.02, +1.18] over u400; `eval/RESULT_P2.md`).
> - A lower bound of +0.018 from one seed per arm is exactly the case H-ESTIM-SEED-1 names: **necessary, not sufficient**.
>
> Evidence classes: **MEASURED** (ours, artifact named) · **INHERITED** · **ESTIMATED** · **UNVERIFIED**.

## 1 · The arm

**The argv.** P2's training `ARGS` (`refe/p2r_pod.sh`, the string that P2's launch-gate token bound) **character for character**, with exactly two changes:
- `--out` → the replicate's own run directory;
- `--seed 1` (P2 ran the parser default, seed 0).

**Everything else is identical:**
- the init checkpoint;
- the carved on-policy bank (174,912 served sets);
- the complete `train_pdm` labels;
- `navtrain_pixels`;
- the step count (1,209), the schedule and the precision;
- the hardware: the same A40 pod, 11 vCPU.

The chain **verifies** P2's inputs on the pod and never rebuilds them.

**What the seed moves.** `--seed` reaches `torch.manual_seed` (`refe/train.py:1125`) and both epoch samplers (`:1262`, `:1500`). So it changes:
- the initial draw of every freshly initialised parameter;
- the data order.

The trainer is also not bit-reproducible across launches (MEASURED on another trainer of this programme: a same-seed rerun diverges). **What this arm measures is therefore the total run-to-run variance, which is the variance a lever claim must clear.**

**Launch.** Only on a verified launch-gate PASS token bound to this argv. The gate must refuse an argv without `--seed 1`, with P2's `--out`, or with any other lever changed (deliberate-regression arms).

**Output.** A new private HF repo, `Sayood/tanitad-refe-p2-joint-pdm-s1`. P2's repo is never written.

**Order on the pod (PI):** P2 ✅ → RL-S1 → S1 → **P2-s1** → P6.

**Cost:** about **21 A40-h** (MEASURED for P2: 42 half-hourly egress ticks), plus a 5.3 h read on the dev box (MEASURED for u400 and for P2).

## 2 · The reads (dev box; the R2 instrument unchanged)

The harness, the token file and the seam settings are those of R2 (`eval/P2_READOUT_RUNBOOK.md` §3):
- `eval/stage2_sft.py` + `eval/stage3_full.py`;
- the pinned 12,146-token file (md5 `3a8b3653…`), 136 logs;
- rule `navsim_v1`, A7 repair ON, goal fix `pdm_route`.

| read | what | estimator |
|---|---|---|
| **RR1** | P2-s1 vs SFT-4 B u400, under the **identical rule** of `PREREG_P2_AMENDMENT_1.md` §4: lo > 0 AND no PDMS sub-score adverse | paired log-cluster bootstrap over the 136 logs, 10,000 resamples, seed 20260927 |
| **RR2** | **the seed floor:** Δ_seed = PDMS(P2-s1) − PDMS(P2-s0) on the same 12,146 tokens, paired per token. P2-s0 is P2's banked table, `raw/p2_readout/p2_full/arm/refe_stage2_p2_full.csv`, md5 `bed5687f…` | the same estimator |
| RR3 | the four families: P2-s1 vs u400 and P2-s1 vs P2-s0, as R2 reports them. STRATEGIC is PAUSED (PI 2026-09-19) and unavailable in NAVSIM | as R2 |

**The floor is defined now:** **F = |mean Δ_seed|**, together with its interval.
- One pair of seeds is **one draw** of the run-to-run difference. F is reported as a one-draw estimate, never as a variance.
- The gates of R2 apply unchanged (harness PASS 12146/12146, 0 seam misses, frame control equal to u400's, checkpoint sha256). A failed gate makes the read **INVALID**, never a pass.

**Tier.** T1 = NAVSIM v1.1 PDMS: the open-loop plan pseudo-simulated against logged agents. It is self-action open loop, not closed loop.

## 3 · What each outcome decides (committed now)

| outcome | definition | P2's lever claim | what follows |
|---|---|---|---|
| **REPLICATED** | RR1 passes (lo > 0, no adverse sub-score) **and** \|Δ_seed\| < the smaller of the two arms' Δ vs u400 | **supported** for "P2's recipe raises PDMS over u400", with the confounds of amendment §6 (labels, un-detached ctx and joint LoRA are not separated) | the lever is recorded at the mean of the two seeds' Δ; SFT-4R remains the arm that separates the confounds |
| **REPLICATED, FLOOR-BOUND** | RR1 passes, but \|Δ_seed\| ≥ the smaller Δ vs u400 | **not supported as a magnitude**; the direction holds on both seeds | report both Δs and F; any P2-derived lever is quoted as "direction replicated, size within run-to-run noise" |
| **NOT REPLICATED** | RR1 fails | **not supported**: P2's +0.57 lies within run-to-run variance | the next lever is the NOT PROVEN row of amendment §4: **SFT-4R** (scorer-only on complete labels), then a seed replicate of the better arm |
| **INVALID** | any R2 gate fails | — | fix the instrument and re-read; no outcome is recorded |

**What the replicate does NOT decide.**
- **P6's launch.** P6 is a from-scratch, paper-faithful run with its own registration (`eval/PREREG_P6.md`). The PI's order places it after this arm.
- Its job for P6 is to supply **F**, the floor P6's lever claim must clear (`PREREG_P6.md`, "Training variance"). The S1 no-head seeds are the fallback floor only if this arm fails to produce a valid read.
- **Which checkpoint is "P2".** P2-s0 remains P2 of record; P2-s1 is its replicate. Neither is selected by these reads.

## 4 · Rule Zero

A NOT REPLICATED outcome names its next lever above, and that lever is run. It is never a stopping point.
