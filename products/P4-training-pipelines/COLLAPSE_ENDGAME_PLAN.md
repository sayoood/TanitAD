# The collapse / decodability challenge — state, remaining gaps, and the plan to finish

**Status:** briefing + plan for the PI. Written 2026-08-24 by the Master Mind.
**Tier:** every number is **T0-DIAGNOSTIC**. Nothing here is a driving claim.
**Sources:** `MODEL_REGISTRY.md` §13.0–§13.0d, `GOALS_AND_CLAIMS.md` E-DEC-7…E-DEC-22,
Paper §13.1–§13.12, raw panels under
`TanitAD Research Lab/Architecture & Inference/Research/2026-08-19-simwam-analysis/raw/`.

---

## 1. The problem, stated precisely

The programme called this "collapse", but it is **three separable failures**, and
conflating them is why it stayed open so long:

| # | failure | the question it asks |
|---|---|---|
| **P1** | **Rank collapse** | does the latent occupy its dimensions? |
| **P2** | **Decodability** | can the scene be read out of the latent? |
| **P3** | **Prediction** | can the next latent be predicted from actions? |

⭐ **They are not the same problem and they do not move together.** A latent can be
full-rank and carry nothing (§13.1's `z = (u, η)` derivation — ego plus isotropic
noise satisfies every rank criterion exactly). A latent can be highly decodable and
un-predictable. And **ego decodability is free**: a frozen *randomly initialised*
encoder reads the best speed of any arm in the programme (+0.3552), so no ego
number is evidence that an objective worked.

---

## 2. What is now SETTLED

### 2.1 ⭐ The empirical severity was largely a SMALL-DATA artefact

Every "collapse" conclusion the campaign drew was measured at **2 000 steps over
130 clips**. At full scale, with the *same architecture and the same two-term
objective*, the problem substantially recedes:

| | 2k / 130 clips | 30k / parity |
|---|---|---|
| participation (val / held-out) | 3.80 / 3.62 | **25.58 / 26.96** |
| predictor `nrmse` vs constant floor | 0.9988 (at the floor) | **0.7845 (beats it)** |
| `n_agents` vs raw-pixel floor | −0.7367 (t −20.45) | **+0.1976 (t 10.21, 24/24)** |

⛔ **Four separate conclusions of ours were re-opened by this**: the degeneracy's
practical severity, "a frozen part is the only thing that works", "freezing is
necessary but not sufficient", and the frozen-field predictor collapse.

### 2.2 ⭐ Only four arms in the programme's history predict at all

The predictor metric had a permutation null but **no constant-predictor floor**
(**C149**). With the floor added and all 30 finished arms re-scored:

| verdict | n | arms |
|---|---|---|
| **beats a constant** | **4** | `rdw8p30k` 0.7845 · `scale1` 0.8200 · **`splitp30k` 0.8416** · `champ30k` 0.9348 |
| ≈ a constant | 16 | every healthy 2 000-step arm |
| **worse** than a constant | 11 | all four O1 arms (to 22.7) · all five PSG arms (to 32.4) |

**Every arm that beats the floor is 30 000 steps on the parity corpus; every such
arm beats it; no shorter arm ever does.** Two do it at batch 4, so batch is not the
operative variable for prediction.

### 2.3 ⭐ The best representation we have, and what it cost

`splitp30k` — frozen distilled encoder, parity, 30k — is the **first arm to clear
the predictor floor and carry scene content simultaneously**:

| metric | `rdw8p30k` | **`splitp30k`** | frozen DINOv3 |
|---|---|---|---|
| predictor `nrmse` vs floor | **0.7903** ✅ | **0.8416** ✅ | — |
| `n_agents` | −0.0180 | **+0.3881** (t 28.21, 24/24) | +0.2754 |
| `n_agents` vs raw-pixel floor | +0.1976 | **+0.6037 (t 30.86)** | +0.4911 |
| `lead_gap_m` | **+0.0063** | −0.0940 (t −6.22) | +0.0294 |
| participation | **25.58 / 26.96** | 6.38 / 7.63 | — |

### 2.4 ⛔ Twenty arms across six auxiliary-objective families all failed

O1 counterfactual separation (4 forms) · O2/O3 masked-cell · O7 DINOv3
distillation (3 weights) · O8 raw pixels · O9 EMA masked latent (2 forms) ·
O10/PSG from our own cuboids (5 weights + encoder-only). **Each failed the
raw-input floor, destroyed the predictor, or both.** The most instructive: PSG at
w=0.03 costs ego *nothing* (t −0.26) and still zeroes the predictor, and
encoder-only PSG — which sends the predictor no gradient at all — kills it too.
⇒ **a term does not have to touch the predictor to destroy it.**

---

## 3. What is still OPEN — ranked by how much it should worry us

### G1 ⛔ `lead_gap_m`: NO arm beats the raw-pixel floor, at any scale

`rdw8p30k` is *level* with pixels (t −0.01); `splitp30k` is *below* them
(t −3.91). **The two environment targets dissociate**, so `n_agents` alone may
never be reported as "environment". This is the clearest remaining hole in the
decodability story and it has never been diagnosed.

⛔⛔ **RESOLVED WHILE WRITING THIS PLAN — THE TARGET IS DEFECTIVE (C150), NOT THE
REPRESENTATION.** MEASURED over all 25,790 frames of the corpus every `lead_gap_m`
number was computed on: **4.8 %** of frames have no lead and take the **80.0
default**, and that 4.8 % carries **59.9 % of the total variance**. So the metric
mostly measures *"is there a lead at all"*, not range. And **80.0 is not a sentinel
outside the data** — real leads reach **180.84 m**, so a no-lead frame is
numerically indistinguishable from a genuine lead at 80 m.

⇒ **"no arm beats the raw-pixel floor on `lead_gap_m`" is now suspect as a
statement about representations**: raw pixels are plausibly good at the binary
presence question that dominates the variance. **G1 is therefore an INSTRUMENT
task, not a training task**, and the readout-row sweep below is **suspended** until
the target is split — it would have tuned a representation against a broken target
and been uninterpretable either way.

**The fix (cheap, no GPU):** score **two** targets — `lead_present` (binary) and
`lead_range_m` (continuous, **lead-present frames only**, no sentinel) — each with
its own constant control, raw-pixel floor and **n**. Then re-read the four
floor-beating arms. Only if range *still* fails does the geometry hypothesis below
become worth testing.

⚠️ **The suspended hypothesis, kept for when the target is sound.** `lead_gap_m` is
a **range** regression.
In a cylindrical projection, range is carried by the **vertical** position and
apparent size of the lead vehicle — i.e. by readout **rows**. E-DEC-2 fixed
`--readout-grid-w` 4 → 8 (columns, = azimuth) and recovered ego. **Nobody has ever
varied `--readout-grid` (rows).** With 4 rows the readout has ~11° of vertical
resolution per cell, which may simply be too coarse to encode range.
*(The competing explanation this section originally flagged — "check the label
distribution before running anything" — is what found C150. It is kept as the
record of why that check came first.)*

### G2 ⛔ Every environment number is IN-SAMPLE

MEASURED pod-side: **130 of 130** labelled clips are inside the parity training
corpus; **0** are in the val cache. So `n_agents` **+0.3881** is measured on clips
the model trained on. The contrast between arms is valid; **a generalisation claim
is not available at all**.

⇒ **This is the single biggest credibility gap in the whole result.**

⭐ **FEASIBILITY AND COST ARE NOW SETTLED (E-DEC-27), metadata-only, no bytes
fetched.** The `obstacle.offline` labels live on HF as 3,146 chunks totalling
147.2 GiB — but only the chunks holding val clips are needed, the clip→chunk table
**survives in-repo** (`…/2026-07-24-v2-corpus-50h-balanced/r0_selection_v2.parquet`),
and `build_obstacle_join.py` already downloads exactly the chunks a clip list
requires:

| | |
|---|---|
| **130 held-out val clips** | **92 chunks → 3.58 GiB** |
| all 256 available | 146 chunks → 5.67 GiB |
| new code needed | **none** |

⇒ the job is `build_obstacle_join.py --selection <parquet>` over a val clip list.
⛔ **NOT started: HF quota is a hard ceiling and a 3.58 GiB pull is spend — the
PI's call.** ⚠️ Only 256 of the 600 val clips are in the table (ample; the current
corpus is 130), and the C150 split must be applied to the new corpus from the
start rather than retrofitted.

### G3 ⛔ Teacher dependence

`splitp30k`'s advantage comes from an encoder distilled from **DINOv3**. It buys
capability, **not independence**. The standing preference for no pretrained labels
is **untouched**, and all six teacher-free families failed. ⚠️ Note also **C138
stands**: we have never beaten DINOv3 in a paired test; `splitp30k` exceeds it on
`n_agents` only, on in-sample clips.

### G4 The two gates in flight (both land today)

* **Gate B — data or steps?** `rdw8s30k` (30k on 130 clips) vs `rdw8p30k`
  (30k on parity). Pre-registered and binary: **beats the floor ⇒ steps suffice
  and we buy GPU-hours; does not ⇒ data is required and corpus expansion becomes
  the Data FlyWheel's top priority.** Lands ~17:30.
* **Gate A — `--o5-k 8` vs `1`?** `ok8p30k`, one flag changed from `rdw8p30k`.
  Lands ~21:20 (measured `step_s` 0.9704 — **not** the 25–35 h I first estimated).
  Early read at 2 500 steps: `rel_scale` 0.0432, `nrmse` 0.9949 — training
  normally, no catastrophic mode. ⚠️ That **cannot pass** the arm.

### G5 Nobody has run past 30 000 steps

All four floor-beating arms are **exactly 30k**. The steps-return curve is
**unmeasured**, and it is precisely what a training budget decision needs.

### G6 Capacity is untested at the right scale

`scale1`'s 4× encoder bought nothing at 30k on parity **at matched `d_op` 2048** —
but if the regime is data-limited, capacity *cannot* show. **Sequence capacity
after Gate B**, not before.

### G7 T0 only

No planner, no closed loop. Under the tier doctrine nothing here may be presented
as driving performance. C131: flagship v1's open-loop 0.4271 → closed-loop 1.7318.

---

## 4. The plan to finish

### Phase 0 — close the gates (today, running, no action needed)
Gate B ~17:30 · Gate A ~21:20. Both pre-registered with outcomes committed.

### Phase 1 — close the two scientific holes (this is what "excellent" requires)

**P1.1 — the held-out environment set (G2).** Extract `obstacle.offline` agent
labels for ~130 parity **val** clips and rebuild the probe corpus there. Until this
exists, **every content number we have is in-sample** and the result cannot be
published as a generalisation claim. ⭐ **This is the highest-value item in the
whole plan** and it is a data job, not a GPU job — it can run fully in parallel
with training.

**P1.2 — REPAIR `lead_gap_m` (G1). Done in part: the diagnosis landed while this
plan was being written and the answer is C150 — the target is defective.** The
remaining work is an instrument change, not an experiment: split it into
`lead_present` (binary) and `lead_range_m` (lead-present frames only), give each
its own constant control and raw-pixel floor, and re-read the four floor-beating
arms. **0 GPU.** The readout-row sweep is suspended until that says whether range
is genuinely undecodable.

**P1.3 — the teacher-free frozen part (G3).** The only route that satisfies the
standing preference. All six auxiliary families are closed, so the honest next
candidate is **not another loss term** — it is to ask whether a frozen part
initialised *from our own 30k parity encoder* (`rdw8p30k`) reproduces
`splitp30k`'s content gain without any pretrained model. That is a bootstrapping
test and it is cheap.

### Phase 2 — the v7 seed run, which doubles as the steps-return curve (G5)

⭐ **One arm answers a scientific question and produces the model.** Launch on Thor
the moment Gate A frees it:

```bash
--stage S-W --require-parity \
--v2-cache <physicalai-train-e438721ae894-w120-256x640cyl> \
--init-from <distill_init.pt> --freeze-encoder \
--enc-dim 128 --enc-depth 3 --enc-heads 4 \
--readout-grid 4 --readout-grid-w 8 --readout-dim 64 \
--pred-dim 256 --pred-depth 3 --pred-heads 4 \
--window 6 --horizons 1 2 4 \
--w-o5 1.0 --w-o6 0.1 --o5-form l1 --o5-k <GATE A> \
--sigreg-subspaces 32 --sigreg-slices 512 --spectrum-accum 43 \
--w-o1-ctrl 0 --w-o1-fact 0 --w-o1-scene 0 --w-o2 0 --w-o3 0 \
--w-o7-distill 0 --w-o8-pixel 0 --w-o9-ema 0 --w-o10-psg 0 \
--steps 60000 --batch 8 --save-every 5000
```

* **`--steps 60000`** — double the current best. Score at 30k *and* 60k to get the
  first steps-return curve the programme has ever had.
* **frozen distilled init** — Gate C's measured trade-off (§5b of the
  recommendation), adopted with its dependency named.
* **no auxiliary term** — twenty arms across six families say so.
* **encoder unchanged** — capacity is sequenced after Gate B (G6).
* the run now carries the **predictor-health monitor**, so a catastrophic
  miscalibration shows at step ~50 rather than after 17 hours.

### Phase 3 — capacity, decided by evidence not by plan

If Gate B says **steps**, scale the encoder next and re-test at matched `d_op`.
If Gate B says **data**, capacity waits and the corpus becomes the priority.

---

## 5. What this plan does NOT promise

* **No driving claim.** Everything is T0. The closed loop is a separate programme.
* **It does not solve the teacher-free problem.** P1.3 is a test, not a solution,
  and six families have already failed.
* **It does not fix `lead_gap_m`.** P1.2 is a diagnosis with one candidate lever.
* **Until P1.1 lands, the content result is in-sample.** That limit must appear
  beside every `n_agents` number we quote, including in the paper.
