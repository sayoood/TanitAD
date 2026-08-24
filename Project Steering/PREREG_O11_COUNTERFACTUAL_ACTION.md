# PRE-REGISTRATION — O11-CF: can an objective that *cannot* be minimised action-blind make the predictor action-conditioned?

**Registered** 2026-08-24, **before** the arm is launched · **Tier** T0-DIAGNOSTIC
· **Author** Master Mind · **Both outcomes committed below in advance.**

---

## 1. The finding this tests a fix for

**E-DEC-30 (MEASURED, 444 windows, 3 arms, positive control passing on all
three).** Normalised against each arm's own response to a 10 % latent nudge, a
**251 % change to the entire action tensor** moves the prediction by:

| arm | action pathway, as % of the latent pathway |
|---|---|
| `rdw8p30k` | **8.5 %** |
| `splitp30k` | **2.2 %** |
| `scale1` | **4.2 %** |

Flipping a hard left into a hard right moves `rdw8p30k` by **1.1 %**. And
`nrmse` — the number the whole Gate-B/Gate-C census ranks arms on — is unchanged
to four decimals under an action shuffle (**0.7845 → 0.7845**).

**The mechanism (§3.2 of the E-DEC-30 report).** O5 trains ẑ_{t+k} ≈ z_{t+k}.
Over 0.6 s the scene at t+k is overwhelmingly determined by the scene at t and
only marginally by the ego's command, so **the loss-minimising solution is to
ignore the action.** The predictor is doing exactly what it was asked.

⇒ The lever is the **objective**, not capacity and not steps. That is the claim
under test.

---

## 2. The arms — a matched pair, ONE thing changed

`ok8p30k` (Gate A, running, 22,600/30,000 at time of registration) is the
control. The treatment is byte-identical except for the O11 flags.

```
--out <run> --stage S-W --v2-cache physicalai-train-e438721ae894-w120-256x640cyl
--frame-h 256 --frame-w 640 --patch 16 --enc-dim 128 --enc-depth 3 --enc-heads 4
--pred-dim 256 --pred-depth 3 --pred-heads 4 --readout-grid 4 --readout-grid-w 8
--readout-dim 64 --window 6 --horizons 1 2 4 --o1-k 4 --o5-k 8 --d-tac 128
--d-str 64 --steps 30000 --batch 8 --v2-lru 64 --log-every 200 --save-every 2500
--seed 0 --spectrum-accum 43 --sigreg-slices 512 --o5-form l1
--sigreg-subspaces 32 --w-o5 1.0 --w-o6 0.1 --w-o1-ctrl 0 --w-o1-fact 0
--w-o1-scene 0 --w-o2 0 --w-o3 0
```

| arm | added flags |
|---|---|
| `ok8p30k` (**control**, already running) | — |
| `o11p30k` (**treatment**) | `--w-o11-cf 1.0 --o11-k 4 --o11-negs 3 --o11-tau 1.0` |

⚠️ **Parity is preserved**: same corpus (`e438721ae894`), same seed, same steps,
same batch. ⛔ **The control must NOT be re-run** — it is the same binary and the
same seed, and re-running it would spend 8 h of the only GPU we have to reproduce
a number we already hold.

**Why `--o11-k 4`, not 8:** the action's influence on the scene is largest early
and is swamped by autocorrelation later — which is the whole mechanism in §1. A
contrastive taken at the end of the roll would be the weakest possible version of
the test.

**Why `--o11-negs 3`:** the no-information floor is `ln 4` = 1.3863; three
negatives give a 25 % chance rate, well separated from a real signal, at the cost
of 3 extra rollouts of length 4 (≈ ×1.6 on the rollout, which is itself a
fraction of the step). ⛔ **`o11_loss` is NOT comparable across different
`--o11-negs`** — the floor moves with it. `o11_excess` is the comparable
quantity, and `test_the_floor_MOVES_with_n_neg_so_losses_are_not_comparable_across_it`
pins that.

---

## 3. The primary read, and both outcomes committed in advance

**Primary instrument:** `actchan.py` (the E-DEC-30 panel, banked at
`TanitAD Research Lab/Architecture & Inference/Research/2026-08-24-action-conditioning-and-heldout/code/`),
re-run on `o11p30k` at 30k **with the identical windows, clips and seed** used for
the control.

**Primary metric:** `d_out(shuffle_all) / d_out(latent +10 % control)` — the
action pathway as a fraction of the latent pathway. Control value: **8.5 %**.

| outcome | reading | what we do |
|---|---|---|
| ⭐ **CONFIRMED** | ratio **≥ 50 %** AND `o5_loss` within **+10 %** of the control's | The objective is the lever. O11-CF goes into the v7 full-scale recipe, and every arm ranked on `nrmse` is re-read with the shuffle control beside it. |
| ⚠️ **PARTIAL** | ratio in **[20 %, 50 %)** with `o5_loss` not degraded | Real but insufficient. Sweep `--w-o11-cf` (0.3 / 3.0) and `--o11-k` before committing full-scale GPU. |
| ⛔ **DEGENERATE** | ratio rises **but `o5_loss` degrades > 10 %** | This is the ẑ = f(z) + λa solution the term's own docstring warns about — perfect action separation, useless prediction. **O11-CF is wrong for this rig and is not carried forward.** |
| ⛔ **REFUTED** | ratio **< 20 %** with `o11_excess` ≈ 0 throughout training | The objective is NOT the lever. The predictor cannot be made action-conditioned by asking it to be, and the deficit is architectural (the action pathway's capacity or where it enters). **Next probe becomes the action-injection site, not the loss.** |

**Secondary reads, recorded but not decisive:**

* `o11_excess` trajectory over training — it must rise **above 0** and stay
  there. A run whose `o11_excess` never leaves 0 has an inert term, which is a
  *bug* reading, not a *result* reading, and must be diagnosed before the arm is
  scored.
* `o5_loss`, effective rank, and the constant-predictor `nrmse` floor — O11 must
  not be bought with collapse.
* `nrmse` **and** `nrmse_SHUFFLED` side by side (`nrmse_shuf.py`). The control
  reads 0.7845 / 0.7845. A CONFIRMED arm must separate them.

---

## 4. Falsifiability guards — what would make this test *invalid* rather than negative

1. ⛔ **An inert term.** If `o11_loss` sits at exactly `ln 4` for the whole run,
   the term never fired. `test_o11_counterfactual.py` pins that an untrained
   predictor reads the floor **exactly** (verified: 1.3862943649 vs 1.3862943611),
   so a *flat* floor reading is the expected START, not the expected END.
2. ⛔ **A `randperm` fixed point** would make a row's "counterfactual" the TRUE
   action and drag the loss to the floor — reading as failure that is not there.
   The call site uses a **cyclic shift** (a derangement by construction) and a
   test pins the distinction.
3. ⛔ **The tie-credit defect.** `pick_acc` credits ties at chance; the naive
   `argmax == target` reads **1.0000 for a completely action-blind predictor**.
   Pinned by test. A `pick_acc` of exactly 1.0000 early in training should be
   treated as a defect report, not a triumph.
4. ⚠️ **Scope.** This is **T0**. It says what the representation contains and how
   it responds, never that a planner drives better. A T1 claim requires
   `taniteval/tools/t1_eval.py`.
5. ⚠️ **The four metric families still bind** for any capability claim built on
   this arm; O11 is a WM diagnostic and does not substitute for them.

---

## 5. Cost and scheduling

* **~8.2 h** on Thor at ≈ 0.98 s/step × 30,000, plus ≈ 60 % on the rollout
  fraction for the three counterfactual rolls.
* ⛔ **Launches only after Gate A (`ok8p30k`) completes** — never add GPU load to
  a box that is training. Gate A was at 22,600/30,000 when this was registered.
* Code is **shipped and md5-verified on Thor**
  (**`24994d5ce95d736028c913dd7cfd31c1`**, O11 marker present, `--w-o11-cf` knob
  present, and **both** call sites carrying the knobs — see below). Thor has no git
  credentials; this arrived by file-ship, as every Thor-side change must.

⛔ **A BUG THE DRY-RUN CAUGHT, RECORDED BECAUSE IT WOULD HAVE INVALIDATED THIS
TEST SILENTLY.** The first ship patched only the TRAINING call site, so the
`--dry-run` path ran on the signature DEFAULTS: `--o11-negs 3 --o11-k 4` dry-ran
as **n_neg = 1, k = 6**. It was visible only because the term logs its own
parameters back. **A dry-run whose hyper-parameters differ from the run it exists
to de-risk certifies a configuration nobody is going to train** — the wrong-scope
family again. Fixed; the shipped file now carries the knobs at **both** call
sites (`grep -c "o11_k=int(getattr"` = **2**, verified on Thor), and the dry-run
reproduces `o11_n_neg: 3`, `o11_at_step: 4`, `o11_loss` 1.3862943649 against
`ln 4` = 1.3862943611, `pick_acc` 0.25 = chance, finite gnorm. Thor has no git credentials; this arrived by file-ship, as every
  pod-side change must.
