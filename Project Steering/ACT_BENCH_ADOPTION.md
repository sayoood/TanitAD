# ACT-Bench: the adoption decision

**Written** 2026-08-26 · **Author** Master Mind · **Source** arXiv **2412.05337**,
banked (`sha256 560190990e9f…`), read from the **PDF**, not from the abstract or an
aggregator — so the numbers and definitions below are **PUBLISHED-PRIMARY**.

⚠️ This item sat in the library tagged *"queued for the action-conditioning review"*
for days and was cited three times in this campaign as something we *would* adopt.
This closes it, and the answer is **not** the one I assumed.

---

## 1. What ACT-Bench actually measures

> Generate the **future video** conditioned on a **commanded trajectory**, then use
> an **independent trajectory estimator** to recover the trajectory *from the
> generated video*, and score **ADE / FDE** against the command.

Their contribution over GenAD is precisely that the estimator is **public and
independent of the world model** — GenAD's relied on a non-public evaluator, which
is why its numbers were not reproducible. They also ship a baseline world model
(**Terra**) and a dataset pairing nuScenes context video with future trajectories.

⭐ **The methodological principle is the important part:** *action fidelity must be
measured by an estimator that is independent of the model being evaluated, applied
to the model's own generations.*

---

## 2. ⛔ It is NOT runnable on our architecture, and the reason is structural

**TanitAD has no latent→video decoder.** Our world model predicts a 2048-d latent;
ACT-Bench's estimator consumes **video frames**. There is no adapter that makes a
latent trajectory estimable by a visual-odometry model.

⇒ **We cannot produce an ACT-Bench number, and we should stop saying we will
"adopt" it as though it were a configuration step.** ⚠️ **That is a real capability
gap, not a scheduling one**, and it belongs to **P7 (TanitEval)** as a scoping input:
*community action-fidelity benchmarks assume a generative video world model; ours is
a latent one.*

---

## 3. ⭐ What it DOES give us, and it is not nothing

**Our probe methodology already satisfies ACT-Bench's principle**, independently
arrived at. Every action/ego panel in this campaign recovers the conditioning signal
from the model's own output using an **independent ridge** — never the model's own
head, and never its training loss:

| ACT-Bench | ours |
|---|---|
| independent trajectory estimator | independent RFF+ridge probe |
| applied to the model's generations | applied to `ẑ_{t+k}` |
| scored against the command | scored against the ego state / action |
| ADE / FDE | within-clip r vs a **measured null** |

⇒ **The alignment is worth recording** — after nine retractions in a day it matters
that the *method* matches the published standard even where the *benchmark* does
not run. ⚠️ And on one axis we go further than ACT-Bench: **they report ADE/FDE with
no null distribution.** This campaign's central lesson (C162/C163/E-DEC-62) is that
a statistic without its measured null is uninterpretable — at n=7 our band cell read
p 0.143, at n=30 it read 0.067, and a six-draw version would have published a
discovery that thirty draws refute.

---

## 4. The decision

| | |
|---|---|
| ⛔ **Do NOT** | schedule ACT-Bench as an eval task, or cite it as a yardstick we will report against |
| ✅ **DO** | record it as the reference definition of *action fidelity*, and cite its **independent-estimator** principle as the standard our probes already meet |
| ⚠️ **Escalate** | *"we cannot be measured on any community action-fidelity benchmark without a video decoder"* — a **PI-level scoping fact** for P7 and for any claim about comparability with GAIA / Vista / Terra |

⚠️ **The honest correction to my own earlier statements:** I wrote three times that
we should "adopt ACT-Bench as an external yardstick", most recently hours ago, and
treated its non-adoption as laziness. **It was never adoptable.** Reading the PDF
instead of the abstract is what settled it — the same PRIMARY-over-SECONDARY rule
the research-banking policy exists to enforce, applied to a paper we had already
banked and still not read.
