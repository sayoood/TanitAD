<title>search log - long-horizon BPTT stability</title>

# raw/search_log.md — E-ARCH-LHB-1, 2026-09-01

Every external claim in `RESULT.md` was read from a **banked PDF**, not from a search-result
summary. That distinction is load-bearing here: see the retraction note in §3 below.

## Queries run

| # | query | outcome |
|---|---|---|
| Q1 | world model training long rollout backpropagation through time gradient instability clipping schedule | HIT — surfaced Looped World Models `2606.18208`, InfinityDrive `2412.01522`, RDR-adjacent work |
| Q2 | Dreamer V3 imagination horizon BPTT stability gradient clipping recurrent state space model | HIT — DreamerV3 Nature paper; confirmed H=15 and AGC |
| Q3 | TD-MPC2 latent rollout horizon training gradient stability horizon 3 short rollout terminal value | HIT — TD-MPC2 `2310.16828`; SimNorm + terminal-value bootstrap |
| Q4 | "progressive" horizon curriculum world model train K=1 gradually increase rollout decode-free latent dynamics | HIT — located the true source of the curriculum claim (see §3) |

## Papers banked this pass (`tools/kb_add.py`, tag `long-horizon-bptt`)

| key | title | role |
|---|---|---|
| `2310.16828` | TD-MPC2: Scalable, Robust World Models for Continuous Control | H=3; terminal value; SimNorm |
| `2301.04104` | Mastering Diverse Domains through World Models (DreamerV3) | H=15; AGC(0.3); lambda-returns |
| `2606.18208` | Looped World Models | K-curriculum; truncated BPTT; rho(A-bar)<1 |
| `2412.01522` | InfinityDrive: Breaking Time Limits in Driving World Models | 16->32->64->128 curriculum (driving) |
| `2608.25017` | Rollout-Decoded Reconstruction for Long-Horizon Prediction in Latent World Models | alpha_e ramp, epochs 2->5 |

`2310.16828` and `2301.04104` were **already banked**; the tool updated tags/citations rather than
re-downloading (correct behaviour — the "re-downloading the same paper per stream" failure mode
`kb_add.py` exists to prevent).

## ⚠️ §3 — one relay was caught and discarded, and it is worth recording

Q1's **synthesised search-result summary** attributed the progressive-horizon-curriculum
("begins with K=1 ... gradients must back-propagate through K × T shared-parameter applications")
to no specific paper, and the two most prominent links returned were `2512.09929` and
`2606.18208`. Fetching **`2512.09929`** for the details returned, explicitly:

> "no information regarding the specific technical parameters you requested ... no gradient-specific
> safeguards or horizon curriculum schedules"

⇒ Had the summary been quoted as-is, `RESULT.md` would have cited **the wrong paper** for its
central schedule. The claim was re-run as Q4, traced to **`2606.18208` §3.5.4**, and only then
banked and quoted from the PDF. This is the `PUBLISHED-SECONDARY` failure the banking rule exists
to stop, caught in flight. No number in `RESULT.md` comes from a search summary.

## Named EMPTY searches — the absence claim

The headline finding is an **absence**: no primary back-propagates >= 60 untruncated steps through
a shared-parameter latent transition. Probed at four independent angles, all empty:

| angle | what was looked for | result |
|---|---|---|
| E1 | reference-class latent WMs (Dreamer/TD-MPC/DINO-WM lineage) with rollout horizon >= 60 | EMPTY — the class caps at H=3 (TD-MPC2) and H=15 (DreamerV3) |
| E2 | driving-specific long-horizon WMs training through the full window | EMPTY as *full-chain* — InfinityDrive reaches 128 frames but by **curriculum + clip-resetting autoregressive rollout**, not one backprop chain |
| E3 | rollout-loss / exposure-bias literature (scheduled sampling, professor forcing, pushforward) | EMPTY — all ramp or truncate; none reports a stable untruncated deep chain |
| E4 | explicit TBPTT-vs-full-BPTT stability results at depth >= 60 | EMPTY — Looped WM truncates at `mu_bwd = ceil(mu_rec/2)` precisely to bound this |

⛔ **Standing instruction:** do NOT re-run E1–E4 without a new trigger (a new paper, or a reviewer
challenging the absence). Add to the `LAB_BACKLOG` row-39 standing empty-search register.

## What was NOT done, and why

- **No GPU run.** `nvidia-smi` showed no python compute on the 4060, so the GPU was available —
  but ASK-1 is a literature question and a local run would not have answered it. The *experiment*
  this implies (curriculum arm vs clip-0.5 arm) is a Training-FlyWheel arm, not a Lab arm, and is
  escalated rather than run here.
- **No re-analysis of the k=60 run's own logs.** The prereg already contains the per-step gnorm
  table; re-deriving it would add nothing and risks quoting a second, drifting copy.
