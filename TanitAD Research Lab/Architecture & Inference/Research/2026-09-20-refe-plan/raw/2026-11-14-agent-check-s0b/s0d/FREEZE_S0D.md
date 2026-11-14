# FREEZE_S0D: R-sim, the agent LIST through NAVSIM's own scorer. Frozen by the MM before any result exists (2026-11-14)

## Why S0-D
- **The committed successor.** The card's X-NULL row lists, as successor (2), an agent-LIST head with the R-sim readout. The S0B-READOUT row commits R-sim directly.
- **What came before:**
  - S0-B (`READ_S0B.md`): the pose-overlap readout captures 0.43 of the oracle even on TRUE agents.
  - S0-C (`run_s0c/`): the learned fusion is NULL.
- **What R-sim changes.** It replaces the readout with NAVSIM's real `PDMScorer`, at-fault logic included, run on agent lists that a perception head could supply.
- **Build record:** `FREEZE_S0D_DRAFT.md` (definitions, hook file:line, cost) and `HANDOFF.md`, both in this folder.
- **What has been seen:**
  - The builder's smoke on 2 held-out tokens printed counts, timings and the identity control only.
  - A 6-set assembly self-test was deleted unread.
  - **No R-sim gain, capture or precision number exists.**

## Pins (sha256, MEASURED by the MM)
| file | sha256 |
|---|---|
| `s0d_rsim.py` | `66d787153424efef526972f790ea7f06a928eabfdbe574efa71f6cb69320d39e` |
| `s0d_hook.py` | `b2d953f6d9aae22e1f47d987658ea8301c013ad32a96e9c9db020365e58af345` |
| `test_s0d.py` (13 tests; 12 mutations RED) | `b18550591fa21af95d0bc995f03a9d49998cec692529676b0f4bd2222d49fb40` |
| `../s0b_readout.py` (unchanged, as `FREEZE_S0B.md`) | `f08e24846eab63556310605801b514b01e384b411f0d91021f34cd728b80ce80` |
| `../covered_sets.txt` | `4e61d655fb523b14b968c573546eebbd80d577a5fd82f10d52aa52cfe71c0607` |

## Arms

| arm | role | definition |
|---|---|---|
| **R-sim-true** | identity control | must equal truth exactly: flags, raw nc/ttc, and per-set gain = the oracle |
| **R-sim-t0** | the single-frame ceiling | dynamic agents present at t0 frozen at their t0 box, velocity 0 |
| **R-sim-t0v** | MM-added | pose frozen at t0, TRUE t0 velocity kept, so NAVSIM's stopped-track test (`pdm_scorer_utils.py:45`, ≤ 0.05 m/s) sees the real classification |
| **R-sim-cv** | the privileged-velocity bound | t0 box plus t0 velocity, constant |
| **C-SIM-NONE** | control | dynamic agents removed; its NC flags must be a per-proposal subset of R-sim-true's |

In every arm, statics are NAVSIM's own.

## Outcome rows (committed now)
The held-out covered sets decide (1,343 sets, 13 logs), on the point estimate. c = gain / O_NC+TTC on the same sets, by the S0-B code path.

| row | condition | next |
|---|---|---|
| **INVALID** | R-sim-true ≠ truth, or C-SIM-NONE not monotone | fix and re-run |
| **S0D-GO** | c(R-sim-t0) ≥ 0.25 | an agent-LIST head (boxes, single frame) on P6's trunk with the R-sim readout: S1, which needs PI pod time. Its cost is in the card's section 7 frame and is re-estimated for a list head before any ask |
| **S0D-STOPFLAG** | c(t0) < 0.25 and c(R-sim-t0v) ≥ 0.25 | the same head plus a per-agent moving/stopped attribute, labelled from lidar_box velocity. S1 with that attribute; PI pod time |
| **S0D-MOTION** | c(t0) < 0.25, c(t0v) < 0.25, and c(cv) ≥ 0.50 | motion is what carries the gain. A motion-conditioned input or a velocity head; **Q-D goes to the PI**, as an architecture decision |
| **S0D-NULL** | otherwise | agent perception through the scorer does not reach the oracle from what a head can plausibly supply. **The agent check closes at Stage 0.** The lever moves to the card's last successor: a **hard-negative fine-tune of P6's NC/TTC heads** on its confident misses. That needs P6 forwards on navtrain images, so it runs on the pod after P7 |

## Run (dev box, one process, CUDA off, venv driverl-eval; nothing keyed, navtrain only)
1. `test_s0d.py` and `test_s0d.py --mutations` must give 13 OK and 12 RED.
2. `s0d_rsim.py --out <run>`, held-out column only.

**ESTIMATED cost:** 0.8 to 1.6 CPU-h, peak RSS about 0.5 GB. Launched detached (WMI, no-window flags).
