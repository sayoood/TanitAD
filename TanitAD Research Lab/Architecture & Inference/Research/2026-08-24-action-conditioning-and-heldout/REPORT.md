# E-DEC-30 / E-DEC-31 — the predictor does not read its actions, and the environment numbers were never held out

**Date** 2026-08-24 · **Tier** T0-DIAGNOSTIC (a WM diagnostic, never "driving
performance") · **Evidence class** MEASURED (ours; dev-box RTX 4060) ·
**Author** Master Mind

> ⛔ **Reader's warning.** This report **retracts a claim I made to the PI about
> four hours earlier** (C151) and **corrects a defect in the instrument that
> produced its own headline number**. Both are recorded here rather than
> quietly fixed, because the retraction is the finding.

---

## 1. What the PI asked, and where this lands

> *"continue working until we solved the collapse, environment learning in the
> wm in both encoder and predictor and prove that it is a good base to learn the
> physics of driving"* — and, sharpening it: *"if the vehicle in front
> decelerates, the ego vehicle must react on it… but we can do this if we are
> sure that the representation is not collapsing and is rich including the
> prediction of the representation."*

| mandate | status after this report |
|---|---|
| **(1) collapse** | ✅ **SETTLED.** Rank 3.80 → 25.58; five arms beat the constant-predictor floor. Unchanged by this report. |
| **(2) environment in encoder AND predictor** | ⛔ **REOPENED.** The encoder number was in-sample; the held-out re-read is pending a fixed instrument (§4). |
| **(3) a good base for driving physics** | ⛔ **ANSWERED NEGATIVE, with a mechanism and a fix.** The predictor is insensitive to its action input (§3). A representation whose prediction does not move when you change the command cannot answer *"what happens if I brake?"* |

---

## 2. E-DEC-28b/28c — the control that started it

The action-shuffle control (committed in advance, both outcomes pre-registered)
rolls the **identical** window with the action sequence drawn from a random other
time in the same clip.

| target (k=6, `rdw8p30k`) | `z_t` baseline | true actions | **shuffled actions** |
|---|---|---|---|
| `n_agents` | +0.0994 | +0.1529 (t 11.03) | **+0.1492 (t 10.96)** |
| `lead_range_m` | +0.0023 | +0.0342 (t 12.62) | **+0.0360 (t 13.20)** |
| `lead_closing` (k=3) | −0.0256 | +0.0019 (t 31.62) | **+0.0019 (t 31.68)** |

And the same shuffle leaves the **latent-space** metric — the number the whole
Gate-B/Gate-C census ranks arms on — unchanged to four decimals:

| arm | nrmse (true) | nrmse (**shuffled**) | ‖d−d_shuf‖/‖d‖ |
|---|---|---|---|
| `rdw8p30k` | 0.7845 | **0.7845** | **0.0077** |
| `splitp30k` | 0.8683 | 0.8680 | 0.0495 |
| `scale1` | 0.8200 | 0.8199 | 0.0084 |

⇒ The ẑ advantage I had reported as *"the first positive physics signal"* is
**temporal smoothing**. It also explains the anomaly flagged and unexplained in
that same report — ẑ scoring *above* the encoded-future ceiling: an averaging
extrapolator is smoother than the noisy true future, so it probes better.
**Retracted as C151.**

---

## 3. E-DEC-30 — which action channel conditions the predictor?

⛔ **The scope error I nearly shipped on top of the retraction.** Reading §2 I
wrote *"the predictor is action-blind"*. It is **not that claim**: the action
tensor is **3-dimensional** — `action_dim = 3  # [steer, accel, v0/10]`
(`flagship_v15.py:101`) — and the probe does `torch.cat([aa, vv], -1)`, so the
shuffle replaced **steer and accel only** and left **v0 at its true value**: the
one channel the programme MEASURED to be worth **3.73 → 0.83 m** fwd_ade.
**C137 retracted a programme-wide action-blind claim once already, for a defect
of exactly this family.** Hence the full factorial.

**Design.** Identical windows, identical rollout, only the action tensor differs.
Two numbers per cell, and the first is what makes the second readable:

* `d_in` = ‖a′ − a‖ / ‖a‖ — **how different the input actually is**
* `d_out` = ‖ẑ′ − ẑ‖ / ‖ẑ − z_last‖ — the **C137-corrected** response, normalised
  by the predictor's OWN delta, never by an arm property

⭐ **Positive control** (without which the panel is unreadable): perturb the
**latent window** by 10 % Gaussian, keep the action true. A predictor that
responds to the latent but not the action is genuinely insensitive; one that
responds to neither is dead and the panel says nothing.

**444 windows · 3 arms · 12 clips.**

| condition | `d_in` | `rdw8p30k` | `splitp30k` | `scale1` |
|---|---|---|---|---|
| shuffle steer+accel | 2.32 | 0.0062 | 0.0472 | 0.0080 |
| shuffle v0 | 0.33 | 0.0118 | 0.2123 | 0.0045 |
| **shuffle all three** | **2.51** | **0.0150** | **0.2247** | **0.0109** |
| zero steer+accel | 0.76 | 0.0058 | 0.0547 | 0.0062 |
| zero v0 | 0.43 | 0.0620 | 0.4437 | 0.0151 |
| **negate steer+accel** (hard left → hard right) | 1.52 | 0.0114 | 0.1019 | 0.0110 |
| `scale100` (**the C137 probe form**) | 75.37 | 0.5084 | 2.1479 | 0.7793 |
| **[control] latent +10 % noise** | 0.10 | **0.1768** | **10.4510** | **0.2571** |

**Normalised against each arm's own positive control** — the only fair reading,
because `splitp30k`'s denominator is much smaller:

| arm | 251 % action change, as % of a 10 % latent nudge |
|---|---|
| `rdw8p30k` | **8.5 %** |
| `splitp30k` | **2.2 %** |
| `scale1` | **4.2 %** |

⇒ **Uniform across all three arms: the action pathway is one to two orders of
magnitude weaker than the latent pathway.** The positive control passes
everywhere, so this is insensitivity, not a dead predictor.

### 3.1 C137 is reproduced, and its scope is narrowed

C137's own metric (`a × 100`) reads **0.5084** here — its measurement **stands**,
and the arms *are* responsive at that input. But ×100 is an input **75× larger in
norm than a real action**; that response is the tail of a saturating
nonlinearity. In the **operating regime** — a real shuffle, a real sign flip —
the response is **0.6–1.5 %**.

⇒ **A sensitivity probe must perturb at the magnitude the model will actually
meet.** A response measured only at 75× the operating input is not evidence about
operation. Same family as `df` on a pod, or the pinhole FOV formula on a
cylindrical projection: a true number quoted outside the regime where it applies.

### 3.2 Why the objective produces this — a design finding, not a bug

O5 trains ẑ_{t+k} ≈ z_{t+k}. Over a 0.6 s horizon the scene at t+k is
overwhelmingly determined by the scene at t and only marginally by the ego's
commanded action. **The loss-minimising solution is therefore to ignore the
action** — it is a low-variance nuisance input, and extrapolation captures most of
the variance. The predictor is doing exactly what it was asked to do.

⇒ **Neither a bigger predictor nor more steps fixes this. The objective does.**

### 3.3 The fix, implemented and tested: O11-CF

`stack/scripts/train_v6_staged.py` — `o11_counterfactual_action_loss`, an
InfoNCE over actions: roll the identical states with the true future actions and
with `n_neg` counterfactual ones taken from other batch elements, and require the
true-action rollout to be the one that matches the observed future.

⭐⭐ **The property that makes it the right instrument: an action-independent
predictor scores EXACTLY `ln(1 + n_neg)` and cannot do better.** Every logit is
equal, the softmax is uniform, the loss sits precisely at the no-information
value. **The C149 constant-predictor floor is inside the loss rather than bolted
onto a panel.** `o11_excess` = floor − loss is ≤ 0 only for an action-blind
predictor.

Verified end-to-end on a synthetic batch: at weight 0 the term is fully inert and
`o5_loss` is bit-identical; switched on, an **untrained** predictor reads
`o11_loss` **1.3862943649** against `ln 4` = **1.3862943611**, `o11_excess`
**−0.0**, `pick_acc` 0.25 = chance. **The instrument reports the known answer on
a case whose answer is known.**

⚠️ **It ADDS to O5 and must never replace it.** O11 alone is minimised by
ẑ = f(z) + λa for large λ — perfect action separation, useless prediction. O5
keeps the prediction accurate; O11 forces the accuracy to be action-dependent.
Both are logged so the degenerate solution is visible rather than inferred.

⚠️ **Two silent-failure guards are pinned by test** (`test_o11_counterfactual.py`,
6 tests):
* the negatives use a **cyclic shift**, not `randperm` — a `randperm` fixed point
  makes that row's "counterfactual" the TRUE action, dragging the loss toward the
  floor and reading as action-blindness that is not there;
* `pick_acc` credits **ties at chance**. The obvious `argmax == target` reads
  **1.0000 for a completely action-blind predictor**, because tied logits make
  argmax return index 0 — which *is* the target. **That is the C149 shape inside
  the term written to prevent C149**, and it was caught only because the test
  demanded the control read chance *exactly*.

---

## 4. E-DEC-31 — the held-out corpus, and the defect it exposed

**Every environment number in this campaign was IN-SAMPLE**: all 130 labelled
clips sit inside the parity train set. The join now fixes that.

**Built** (23.6 s): 124 held-out val episodes · 23,164 frames · 762,204 agent
boxes · reader-verified · md5 `66efaa94fcc58b7fc4f57734545b103c`. Five clips
skipped for having no `obstacle.offline` (the documented 2.5–3.1 % rate).
Geometry matches the training corpus (256×640 png, 201 frames) and the overlap
with the in-sample corpus is **ZERO**.

### 4.1 The first held-out read was an artefact, and its own control said so

| target | `splitp30k` | `rdw8p30k` | pixels | **frozen DINOv3** |
|---|---|---|---|---|
| `n_agents` **in-sample** | **+0.3881** | +0.0777 | — | **+0.2754** |
| `n_agents` **held-out (pre-fix)** | **−0.3657** | −0.0580 | −0.0810 | **+0.0114** |

⛔ **DINOv3 collapsed too.** It trained on neither corpus, so it cannot have
memorised ours. **A control that moves with the treatment means the measurement
changed, not the model** — so the drop was not read as a result.

**The cause, found by probing rather than assuming:** the probe wrote
`ag.get(i, [])`, so an **unlabelled frame became `n_agents = 0`** — a missing
value silently wearing the costume of a real one. **This is the C150 defect
exactly.** It was invisible in-sample and severe held-out:

| corpus | labelled of first 100 frames | clips < 50 labelled | **fake zeros** |
|---|---|---|---|
| in-sample | 100.0 | 0 | **0.00 %** |
| held-out | 95.1, **min 0** | **7 of 124** | **4.90 %** |

Seven panel clips are mostly unlabelled and read as *"zero agents in view"*. In
LOEO, when one is the held-out clip, the probe must predict a constant 0 from
real imagery — R² goes sharply negative, for every column including DINOv3.

**Fixed:** `spatial_targets` now returns a labelled-mask, unlabelled rows are
dropped rather than zero-filled, clips below 80 % coverage are dropped, and **the
count dropped is printed** — an aggregate that does not report what it compared is
the vacuous-freeze-check defect in another costume. `envpred.py`'s `n_agents` is
NaN-masked to match (its lead targets already were).

**Control re-run confirms the fix is inert where the defect was absent:**
in-sample reads `2400/2400 frames labelled (100.00 %), 24/24 clips kept, 0
dropped`.

⚠️ **The held-out environment verdict is therefore PENDING, not negative.** It
will be reported when the fixed instrument returns on both splits. What is
already established is that the pre-fix held-out numbers are inadmissible.

---

## 5. Deliverable manifest

| artifact | where it lives |
|---|---|
| `o11_counterfactual_action_loss` + call site + 4 CLI knobs | `stack/scripts/train_v6_staged.py` (repo) |
| 6 pinning tests | `stack/tests/test_o11_counterfactual.py` (repo) |
| C151 retraction | `Project Steering/RETRACTION_LOG.md` (repo) |
| E-DEC-30 panel | `raw/actchan.json` + `code/actchan.py` (repo) |
| E-DEC-28b/28c | `raw/actionshuf.json`, `raw/nrmse_shuf.json` (repo) |
| join provenance | `raw/val130_join.meta.json` (repo) |
| held-out corpus + labels (23,164 frames, 7.13 GiB of source chunks) | scratchpad `sp2/` — **too large for the repo; the meta sidecar and the exact rebuild command are banked instead** |

**Rebuild command for the join** (deterministic, 23.6 s, no download needed once
the chunks are cached):

```bash
python stack/scripts/build_obstacle_join.py --corpus <val130-heldout> --episodes 200 --selection <r0_selection_v2.parquet> --hf-cache <hfcache> --no-download --out val130_agents.jsonl
```

---

## 6. What follows

1. **Read the fixed-instrument panel on both splits** — the held-out environment
   verdict for mandate (2).
2. **Train the O11-CF arm** on the v7-tiny rig once Gate A frees Thor
   (`ok8p30k` at 22,600/30,000, 0.98 s/step, healthy). **Pre-registered read:**
   `d_out(shuffle_all)/d_out(control)` rises from **8.5 %** toward ≥ 50 %, with
   `o5_loss` not degrading. Both outcomes committed in advance: if O11 rises while
   O5 degrades, that is the ẑ = f(z) + λa degenerate solution and the term is
   wrong for this rig.
3. ⛔ **Every arm ranked on `nrmse` must be re-read with the action-shuffle
   control beside it.** The Gate-B/Gate-C census ranked arms on a number that is
   unchanged when the actions are replaced with noise.
