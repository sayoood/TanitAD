<title>search log - terminal value vs fan-scoring</title>

# raw/search_log.md — E-LAB-DEPLOY-0901, 2026-09-01

This package is **measurement-led, not literature-led**: the primaries were banked by today's
Architecture package and are cited from there rather than re-searched. What follows is the run
log and the provenance of every non-measured statement.

## Literature actually used (all PRIMARY, all banked, none re-downloaded)

| key | used for | banked by |
|---|---|---|
| `2310.16828` TD-MPC2 | Planning Horizon **H = 3**; *"bootstrapping return estimates beyond horizon H with a learned terminal value function"* | today's Arch package, tag `long-horizon-bptt` |
| `2301.04104` DreamerV3 | Imagination horizon **H = 15**; lambda-returns past prediction horizon T = 16 | idem |

⛔ **No search-result summary is quoted anywhere in `RESULT.md`.** The one relay encountered
today was caught and discarded in the Arch package — see its `raw/search_log.md` §3.

## Run log

| step | what | outcome |
|---|---|---|
| 1 | `nvidia-smi --query-compute-apps` | no python compute on the 4060 → GPU available (desktop/browser processes only). Thor and pods untouched throughout. |
| 2 | Located venv `C:\Users\Admin\venvs\tanitad` | system python 3.14 has **no torch**; the venv has `torch 2.11.0+cu128` |
| 3 | **Verified CUDA with a real `conv2d`**, not `import torch` | `conv2d ok (4, 8, 30, 30)` — per CLAUDE.md, cuBLAS can work while cuDNN/conv is broken |
| 4 | Read `…/2026-08-31-rollout-depth-latency/raw/rollout_depth_4060.json` for the exact config | dim 384 / depth 4 / heads 6 / **tokens 16** / fp16 — copied verbatim so C1 is a true reproduction |
| 5 | Wrote `code/breadth_vs_depth_bench.py` **with both outcomes committed in its docstring**, saved before execution | pre-commitment is in the shipped file |
| 6 | Run A: 40 reps / 12 warmup | headline 6.65×; **C1 ratio 1.55 — failed its bar**; K=8 row +63 % vs the prior package |
| 7 | Diagnosed as rep-count/contention noise, **not** accepted as the answer | patched reps to be env-driven (`BVD_REPS`) |
| 8 | Run B: 120 reps / 30 warmup | headline **5.94×**; C1 clean at K>=3, reproduces the prior package within ~7 % |
| 9 | Both JSONs banked | the corrected-away run is kept on the record, not deleted |

⭐ **Step 6→8 is the finding-behind-the-finding.** A 40-rep run produced a *confident, plausible,
wrong-by-12 %* headline and a control that quietly failed. It was caught only because C1 had a
**pre-committed external value to agree with** (the 2026-08-31 package on an identical config).
Without that anchor, 6.65× would have shipped. This is the CLAUDE.md ridge-probe lesson —
*the only thing that catches a manufactured result is a control that must read a known value* —
reproduced in a latency costume.

## Named EMPTY searches

None. **No web search was run for this package** — the question is a property of our own
hardware and the two horizon values were already banked. Recording this explicitly so the
absence of a search log is not later read as a lost one.

## What was NOT done, and why

- ⛔ **No quality comparison.** A trained terminal value head on our latent does not exist; the
  benchmark head is randomly initialised and used **only** to price its forward pass. Any
  accuracy number from it would be meaningless, so none is reported.
- **No CUDA-graph variant.** The 2.57× is cited from the July package, not re-measured; adding
  it here would have conflated two levers in one arm.
- **No Thor run.** `k60p30k`/fleet discipline: the Lab does not touch Thor or pods. The Thor
  re-measurement is escalated as a proposed row instead.
- **No sweep past N=960.** The saturation point (~N≈32) was already inside the swept range, so
  extending it would not change the finding.
