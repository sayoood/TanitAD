# MM-E11 — ⛔ **O1-INERT. The pre-registered lever is REFUTED, and the prereg already said what that means.**

**Measured** 2026-08-31 · Master Mind · **Tier T0-DIAGNOSTIC** · raw
`actdiv_o1ctrl.json` md5 `72606e6a01984d7f2e446da00c7f07db` (verified by read-back
against Thor) · 24 val clips, 144 windows, 8 action variants drawn by ROLL.

## The question, and the answer

MM-E10 measured the predictor to be **action-deaf**. `--w-o1-ctrl` — the term the code
itself documents as *"the ANTI-ACTION-ECHO measure"*, `default=1.0`, which **every arm
in the campaign ran at 0** — was the cheapest possible test of the diagnosis: turn on
the objective whose documented job is exactly this defect.

| arm | h1 action_spread | h1 scene_spread | **h1 ratio** | C0 |
|---|---|---|---|---|
| `postrain30k` (incumbent, banked) | 0.00139 | 0.23434 | **0.00595** | PASS |
| `o1ctrl30k` (`--w-o1-ctrl 1.0`) | 0.00055 | 0.23176 | **0.00236** | PASS |

**Fold-change: 0.40×.** The pre-registered `O1-WORKS` criterion required a **≥10×
rise** (to ≥0.06). The ratio did not rise — it **more than halved**.

⇒ ⛔ **VERDICT: O1-INERT.** Turning on the anti-action-echo objective did **not** make
the predictor action-sensitive.

## Why the panel is admissible

* **C0 IDENTITY PASSES**: feeding the same actions twice gives spread **exactly 0.0**.
  The forward is deterministic, so 0.00055 is real signal-absence, not numerical noise.
* **C1 SCENE REFERENCE IS LARGE** (0.23176), and within 1 % of the incumbent's 0.23434.
  ⇒ the model is *not* degenerate, and the two arms are being measured on the same
  scale — the denominator did not move, so the ratio change is a numerator change.
* ⛔ **h2/h4 ARE NOT REPORTED**, per the amendment made at step ~23,000 before any
  number existed. Those heads take **exactly zero gradient** (measured: bit-identical
  across all 8 snapshots), so their values are initialisation noise. Reporting them is
  what made MM-E10's h2/h4 rows meaningless and forced MM-E14's retraction.

## ⚠️ What this does NOT license

* **Not "O1 harms the model".** A 0.40× move on a **single seed**, with no measured
  seed band at 30k, is *unresolved* in magnitude (the MM-E8 lesson). What is resolved
  is the **direction**: `O1-WORKS` demanded a 10× rise and the value fell, and no
  plausible seed band converts a fall into a 10× rise.
* ⛔ **The anti-gate does not rescue this arm.** The prereg committed in advance that a
  T0 prediction regression would be *expected* and not grounds for rejection — but it
  also stated the limit: *"a T0 regression with NO action-ratio gain is a dead arm, not
  a de-confounded one."* There is no gain here. The escape clause was written before
  the data and it does not apply.

## ⭐ The consequence, pre-committed before the arm ran

The pre-registration's `O1-INERT` row says it, and it is now in force:

> ⛔ **the defect is NOT the missing objective — it is architectural (how actions enter
> the predictor), and the next lever is the conditioning path itself, not another loss
> weight.**

That is the value of having written it down first: the result is a redirect rather than
a disappointment, and nobody has to argue about what it means now that a number exists.
Two loss-weight levers have now been tried against action-deafness (O1 off→on) and the
**objective family is exhausted**; actions enter through `act_emb` → FiLM `cond`, and
that path is where the next experiment belongs.

## ⭐ SECOND RESULT — the 6 s horizon costs 2.4×, not 7.5×

The k=60 timing probe (200 steps, same build, marginal rate from `step_s_interval` —
the field the trainer itself labels *"THIS is the live-monitor field"*, never the
cumulative `step_s`):

```
step  50  2.9755      step 150  2.8858
step 100  2.9113      step 200  2.9087       mean (excl. warmup) 2.9019 s/step
```

| | s/step | 30k steps | peak CUDA |
|---|---|---|---|
| `o5_k 8` (0.8 s) — the whole campaign | 1.21 | ~10 h | — |
| **`o5_k 60` (6.0 s) — the §4b requirement** | **2.90** | **24.2 h** | **6.27 GB** |

⭐ **2.4× slower, not the 7.5× flag arithmetic implies** — because the rollout is not
the dominant cost; the encoder pass does not change with k. ⚠️ And the repo's own prior
estimate for k=8 (*"~3-4× the k=1 ~1.0 s/step"*) overshot by ~3×, which is exactly why
this was measured rather than extrapolated: **200 steps cost 0.4 % of an arm and removed
the guess in both directions.**

⛔ **No OOM risk at this scale.** The banked A40 OOM at k=60 (37.97 / 44 GiB) was the
**190 M config-E** predictor; this rig is 18.25 M and peaks at **6.27 GB**. A number
that was true of a different model was about to be inherited as a constraint on this one.

⚠️ **The cost that is NOT wall-clock:** `[v6] windowing: window 6 + max_horizon 60 …
a LONGER horizon yields FEWER windows per episode` — the corpus yields **319,002**
windows at k=60. The addendum priced this at *"~43 % of the windows (94 → 54 per
120-frame episode)"*. The 6 s horizon buys reach and pays in sample count; that trade
must be stated wherever a k=60 arm is compared to a k=8 one.

## Deliverable manifest

| artifact | md5 |
|---|---|
| `raw/actdiv_o1ctrl.json` | `72606e6a01984d7f2e446da00c7f07db` |
| `raw/k60_timing_log.jsonl` | `927eaec9f2bfa108a3f2db74e56963bc` |
| pre-registration (incl. the pre-read amendment) | `Project Steering/PREREG_O1_CTRL.md` |
