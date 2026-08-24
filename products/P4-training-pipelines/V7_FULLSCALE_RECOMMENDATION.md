# v7 full-scale — the recommendation, with its evidence and its gates

**Status:** RECOMMENDATION for the PI. Not a decision, not adopted.
**Author:** Master Mind, 2026-08-24.
**Tier:** every number below is **T0-DIAGNOSTIC**. None of it is a driving claim
(`EVAL_DOCTRINE.md`); no arm here has a planner or a closed loop.
**Sources:** `MODEL_REGISTRY.md` §13.0 and the raw panels under
`TanitAD Research Lab/Architecture & Inference/Research/2026-08-19-simwam-analysis/raw/`.
Claims live in `Project Steering/GOALS_AND_CLAIMS.md` (E-DEC-16 … E-DEC-20).

---

## 1. The recommendation in one line

**Train v7 on the PLAIN two-term objective at parity scale, for as many steps as
the budget allows. Add no auxiliary term. Do not grow the encoder yet. Hold the
frozen-initialisation decision for `splitp30k`.**

## 2. The config

```bash
--stage S-W --require-parity \
--v2-cache <physicalai-train-e438721ae894-w120-256x640cyl> \
--enc-dim 128 --enc-depth 3 --enc-heads 4 \
--readout-grid 4 --readout-grid-w 8 --readout-dim 64 \
--pred-dim 256 --pred-depth 3 --pred-heads 4 \
--window 6 --horizons 1 2 4 \
--w-o5 1.0 --w-o6 0.1 --o5-form l1 --o5-k <1 or 8 — SEE GATE A> \
--sigreg-subspaces 32 --sigreg-slices 512 --spectrum-accum 43 \
--w-o1-ctrl 0 --w-o1-fact 0 --w-o1-scene 0 --w-o2 0 --w-o3 0 \
--w-o7-distill 0 --w-o8-pixel 0 --w-o9-ema 0 --w-o10-psg 0 \
--steps <max affordable, >= 30000> --batch 8
```

## 3. Why: the one thing that worked, and the twenty that did not

**E-DEC-19.** `rdw8p30k` — this exact objective, parity corpus, 30k steps, **no
teacher, no external target, nothing auxiliary**:

| | tiny (2k, 130 clips) | **parity (30k)** | vs raw-pixel floor (paired) |
|---|---|---|---|
| participation val / held24 | 3.80 / 3.62 | **25.58 / 26.96** | — |
| predictor cos h=1 (z) | 0.0541 (3.99) | **0.6224 (30.55)** | — |
| `n_agents` | −1.0407 | **−0.0180** (t 10.06, 24/24) | **+0.1976 (t 10.21, 24/24)** |
| `lead_gap_m` | −0.3290 | **+0.0063** | −0.0001 (t −0.01) |

Predictor **11.5×** the tiny arm and **3.3× the best figure the programme has ever
measured under any objective**; first trained representation of ours to clear the
raw-pixel floor on scene content.

**Against that, every auxiliary objective we built — ~20 arms across six
families — failed the raw-input floor, destroyed the predictor, or both:**

| family | arms | outcome |
|---|---|---|
| O1 counterfactual separation | plain · stop-grad · stop-grad+detach · 3× | predictor to noise in all four (E-DEC-15) |
| O2 / O3 masked-cell | — | ego below the constant control (E-DEC-5b) |
| O7 DINOv3 distillation | w = 1 / 10 / 50 | best env at w=50, ego destroyed, gate-rejected (E-DEC-11) |
| O8 raw pixels | w = 1 | low-pass target, ego regressed (E-DEC-10) |
| O9 EMA masked latent | naive · DMT-JEPA neighbour | `n_agents` −4.36 / −2.97 (E-DEC-13) |
| O10 PSG, our own cuboids | w = 0.03 / 0.1 / 1 / 3 · encoder-only | predictor dead at every weight; never beats pixels (E-DEC-18b/c) |

**PSG deserves its own line because it was the strongest candidate and its
failure is the most instructive.** At w=0.03 the term sits at loss-parity with
O5, **ego is statistically unchanged (t −0.26)**, environment moves from far
below raw pixels to level with them — **and the predictor is still exactly
zero**. Encoder-only PSG (`--psg-enc-only`, no gradient to the predictor at all)
kills it too. ⇒ **a term does not have to touch the predictor to destroy it**, and
PhyLatent's shared-head-on-predictions is exonerated: the damage is what the term
does to the **encoder**.

⚠️ **Not a general law.** I formed "decodability and predictability are in
tension" from this and refuted it the same hour: across **19 arms on the same 24
clips, r(cos, `n_agents`) = +0.3479, t ≈ 1.53 — not separable from zero**
(`tension.json`). The −0.91 anticorrelation exists only inside the PSG weight
sweep, which is one knob's dose-response. **PSG's failure is specific to PSG.**

## 4. Why NOT to grow the encoder yet — this contradicts the current plan

`V7_FULLSCALE_PLAN.md:104` names the **encoder** as the first scaling axis
(`--enc-dim 128 → 384+`, `--enc-depth 3 → 8+`), on 15.5× parameter headroom. Our
only two parity-scale arms argue against spending there first, and they are
comparable because **`d_op` is 2048 in both**:

| | `scale1` | `rdw8p30k` |
|---|---|---|
| encoder | 256 × 6 (**4×** params) | 128 × 3 |
| readout | old 4×4, dim 128 | 4×8×64 |
| batch | 4 | 8 |
| `d_op` | **2048** | **2048** |
| predictor cos h=1 (z) | 0.6090 (15.65) | **0.6224 (30.55)** |
| `n_agents` | **+0.0263** | −0.0180 |

**Four times the encoder bought nothing at 30k on parity.** At this step budget we
look **data/step-limited, not capacity-limited**. ⚠️ Batch and readout shape also
differ, so this is **suggestive, not decisive** — and it is a reason to sequence
capacity *after* the step/data answer, not a reason to freeze the encoder size
forever.

⚠️ It also shows **E-DEC-2's readout geometry is not what produced the parity
gain** — `scale1` used the OLD 4×4 readout and reads *better* `n_agents`. E-DEC-2
was always a claim about **ego**, and it stays one.

## 5. ⭐ The finding that reframes the decision — E-DEC-20

All on the **same 24-clip ENV set**, so the columns are comparable:

| | `splitfrz` (2k) | **`splitfrz10k`** | `rdw8p30k` | frozen DINOv3 |
|---|---|---|---|---|
| predictor cos h=1 (z) | 0.1872 (7.99) | **0.0007 (0.40)** | **0.6224 (30.55)** | — |
| `n_agents` | +0.4035 | **+0.4156** (t 10.65, 24/24) | −0.0180 | +0.2754 |
| `n_agents` vs pixel floor | +0.6191 (t 36.27) | **+0.6313 (t 36.67)** | +0.1976 (t 10.21) | — |
| speed / `d_ego` | +0.2465 / +0.2878 | +0.2594 / +0.2939 | +0.1482 / +0.0963 | +0.4081 / +0.3238 |

**The frozen distilled encoder is the best scene-content carrier the programme
has** — above `rdw8p30k` *and* above frozen DINOv3 on the same clips, **36σ over
raw pixels** — and it **holds and improves over 5× more training** (+0.4035 →
+0.4156), with ego intact. Freeze CONTENT-verified at both 2k and 10k (encoder
byte-identical to the init, **max|Δ| = 0** over 41 tensors).

⛔ **And its predictor collapses to noise between 2k and 10k** (z 7.99 → 0.40).
Since the encoder provably did not move, **the trainable readout+predictor found a
degenerate solution on top of a fixed feature field.** Freezing preserves
**content** and does **not** preserve the **predictor**.

⇒ **Two arms, each winning one axis, neither winning both.** That is why the
config above leaves the initialisation open: **`splitp30k`** (frozen distilled
encoder, parity, 30k, running) is the arm that tests whether both are obtainable
at once, and it is the most informative number outstanding.

## 6. Gates before the full spend

| gate | question | cost | status |
|---|---|---|---|
| **A** | Does `--o5-k 8` transfer to parity? **Both of our best arms ever ran `--o5-k 1`** — the setting the tiny four-point curve calls a correctness bug. `ok8p30k` = `rdw8p30k` with one flag changed. | ~25–35 h (8× rollout) | **armed on Thor** behind `splitp30k`, md5-verified, watcher self-match-safe |
| **B** | DATA or STEPS? `rdw8s30k` (30k on 130 clips) vs `rdw8p30k`. If DATA, the step budget above is wasted without corpus expansion and the Data FlyWheel's priority changes. | running | lands ~18:00 today |
| **C** | Does the frozen part give content **and** predictor at parity? `splitp30k`. | running | ~10.8k/30k |

**I would not sign a long v7 run before B and C read out.** A is a
cheaper-or-equal question and can run in parallel.

## 7. What is explicitly NOT claimed

* **No driving claim.** All T0. `C131`: flagship v1's open-loop 0.4271 → closed-loop 1.7318, a 4.05× divergence.
* **We have never beaten frozen DINOv3** (`C138`, paired: speed −0.1251, t −2.72). It reads `n_agents` **+0.2754** where our best *trained* parity arm reads **−0.0180**. Only the **frozen distilled** arm exceeds it (+0.4156), and that arm has no working predictor. Under **D-003** frozen stays a comparison arm, not a hedge to adopt.
* **`rdw8p30k`'s environment rows are IN-SAMPLE** — 130 of 130 lead clips are inside the parity train corpus, 0 in val (MEASURED pod-side). A valid contrast; not a generalisation claim. Ego rows are held out for both arms.
* **`n_agents` −0.0180 is still marginally below a constant predictor**, and on that clip set the pixel floor (−0.2156) is itself below the constant control — so clearing the floor is **the weaker of the two bars** and must not be quoted as the stronger.
* **Batch is not null** — `rdw8b8` moves `n_agents` −1.0407 → −0.1099 (t 2.75, 22/24), ~86 % of the distance to `rdw8p30k`, at 2k. It is a substantial lever for **environment** and a minor one for the **predictor** (0.0933 vs 0.6224). I claimed it was ruled out for free; that is withdrawn (`C148`).
* **Tiny-arm results do not license capability claims** (H-SCALE-2). Four conclusions of mine from 2026-08-23/24 were re-opened by E-DEC-19, and `--o5-k 8` is itself a tiny-arm result — hence Gate A.
