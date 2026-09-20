# E9 — does `taniteval/tools/t1_eval.py` run on a refcv6 checkpoint at 416 × 1024?

**Commissioned by** the Master Mind brief of 2026-09-19 (task 3). **Answer: NO — `t1_eval.py`
is UNWIRED for refcv6, by construction.** The REF-C T1-family harness is `refcv3_arm.py`,
and **it RUNS** on the same checkpoint (rate below).
**Evidence class:** MEASURED (ours), 2026-09-19. CPU only, because the GPU belongs to A8. The
checkpoint is A3's `C:/Users/Admin/tanitad-caches/a3-heldout-20260919/run/ckpt.pt` (refcv6
`--size tiny`, resnet34, `--image-hw 416 1024`, 2,000 steps on halfA), scored on
`v2ep-eval124clean-416x1024cyl-halfB`.

## 1. `t1_eval.py` — three refusals, each read from its own output

| attempt | what it said (verbatim) | log |
|---|---|---|
| 1 | `rollout mode needs --dump-dir (or use --analyze-only)` | `raw/e9_t1eval_attempt.log` |
| 2 | `rollout mode needs --head or --grounding-readout (or use --analyze-only)` | `raw/e9_t1eval_attempt2.log` |
| 3 (`--grounding-readout`) | `… ckpt.pt has no 'model'+'grounding' keys — the adapter path needs a flagship trunk checkpoint (v1/v4/v5f shape). Keys: ['model', 'opt', 'step']` | `raw/e9_t1eval_attempt3.log` |

⚠️ Attempt 3 also printed `[geometry] t1_eval: DEPLOYED (unchanged) - 256x256px, f_ref 266.00,
pinhole`. **The geometry auto-detection does not recognise a refcv6 run either:** it knows v6/v7
checkpoints only. Had the load gone further, it would have rolled a 416 × 1024 cylindrical model
on the deployed pinhole frame.

⇒ **Why it is unwired, and why that is correct rather than a bug:** `t1_eval.py` rolls an
**action-conditioned predictor**. It decodes latents to actions through a readout (`--head`, a
`UnicycleStepReadout`, or `--grounding-readout`) and feeds them back. REF-C (refcv3 → refcv6) is
a **one-shot anchor planner with no action input** (`refc_v3.py` `forward(frames, nav_cmd, v0,
…)`), so it has no loop for this tool to roll. Wiring it would mean inventing an action interface
that the model does not have.

## 2. The harness that IS wired: `taniteval/tools/refcv3_arm.py` (W1, commit `4000946`)

| | |
|---|---|
| command | `refcv3_arm.py --ckpt <A3> --config <A3 config> --episodes <halfB> --labels s2_labels_v8_eval.jsonl.gz --device cpu --episodes-n 2 --window-stride 40 --lru 2 --n-boot 50 --dump-dir …` |
| result | **RC 0**: 9 windows / 2 episodes; surfaces `os`, `ha`, `ha0`, `ha0_ext` plus the nav-shuffle; LON / LAT / TAC emitted with paired intervals; STRATEGIC unavailable (PI-off, item 25) |
| **rate** | **173 s wall for 9 windows ⇒ ≤ 19.2 s/window on CPU, ALL-IN**. That includes model load, corpus build and analysis, so the marginal per-window cost is lower and not separated out here. ⛔ **The GPU rate is NOT MEASURED** (A8 holds the card) |
| tier stamp the tool itself prints | `os tier=T1*`, with the note *"the ruling OPEN (does EVAL_DOCTRINE admit as T1 a model that consumes NO actions…)"*. The harness flags its own tier question, and that question is the PI's |
| not a capability claim | the planner has 0.38 of an epoch; `os` ADE 2.9098 m vs `ha` 0.1946 m on 9 windows only shows **the chain runs** |
| log | `raw/e9_refcv3arm_rate.log` |

## 3. What downstream numbers must be relabelled

* `PREREG_REFCV6_DEVBOX_PREPARATION.md` §3.1 **A2**'s criterion *"a `t1_eval.py` invocation at
  416 × 1024 … its s/window is recorded"* **cannot be met by `t1_eval.py` for any REF-C arm**. The
  admissible reading is: the REF-C T1-family harness (`refcv3_arm.py`) runs at 416 × 1024, at
  **≤ 19.2 s/window on CPU**, all-in.
* ⇒ §7's A2 branch *"T1 harness not wired at 416 ⇒ C8 is live, A3–A6 are trainer-side reads"*
  **does NOT fire**, provided A3–A6 are read through `refcv3_arm.py`. **Any refcv6 number stamped
  "T1 via `t1_eval.py`" is impossible and must be relabelled.**
* The `T1*` stamp on `os` for a model with **no action input** is an open doctrine question the
  tool raises itself. It is escalated, not decided here.
