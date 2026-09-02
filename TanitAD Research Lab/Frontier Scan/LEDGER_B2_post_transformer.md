<title>LEDGER B2 - LLM architecture and post-transformer</title>

# LEDGER — B2 · LLM architecture + post-transformer

`APPEND-ONLY. Charter DAILY_RESEARCH_CHARTER.md §3.`
`⚠️ Opened 2026-09-01. TRACKS.md linked this ledger from 2026-08-31 but THE FILE DID NOT EXIST -
a broken link in the growing artifact. Recorded rather than silently created.`

---

## Entry 2026-09-01-01 - ⛔ CORRECTION: the "25 % budget" claim attributed to 2601.22156 does not exist

**Source:** `arXiv 2601.22156`, "Hybrid Linear Attention Done Right: Efficient Distillation and
Effective Architectures for Extremely Long Contexts" (Chen, Thai, Zhou, Zhang, Shen, Wang, Xiao, Han,
Liu; submitted 2026-01-29). **Evidence class:** PUBLISHED lib 2601.22156, abstract read.
**Discharges the pass-1 debt "banked but NOT read".**

Proposes **HypeNet**, a hybrid RNN + softmax-attention architecture with a **HyPE** hybrid position
encoding and modifications for length generalisation.

⛔ **The efficiency claim is about TRAINING DATA, not compute or memory budget.** The conversion
pipeline requires *"just 2.3B tokens, less than 0.01% of their pre-training data"*, against
*">10B tokens"* for prior distillation methods. **No 25 % compute/memory figure appears.**

⇒ `TRACKS.md` and injected backlog row **I-2** both carried "its 25 %-budget claim is RELAYED".
That claim is now **UNSUPPORTED-AS-STATED** and is struck. *(Root-cause class: a RELAYED number
carried as settled - the same class as register debts D-1 and D-2, both also corrected today.)*

⭐ **The paper is more useful than the claim it was filed under, in a different direction.**
I-2 was scoped as *"state-space / linear-attention predictors as the O5 predictor at matched params"* -
i.e. **train one and compare**. This paper's actual contribution is **CONVERSION**: distil an
already-trained transformer into a hybrid linear-attention model for a very small token budget.

**Why that matters here specifically:** we have a **trained v7 predictor** and a **fixed fleet**.
Converting is a categorically cheaper experiment than retraining at matched params, and it is the
method the literature actually supports. ⇒ **Recommend re-scoping I-2 from "train" to "convert".**

**Chances / risks:** ⭐ upside - a matched-quality long-context predictor at lower rollout cost, which
is the O5 depth question. ⚠️ risk - the result is demonstrated on **language** at extreme context;
our predictor is a short-horizon visual latent model, so **transfer is a hypothesis, not an
inheritance**. The 2.3B-token figure is a language-pretraining fraction and **does not translate into
our units**; do not quote it as our conversion cost.

**Pre-registerable experiment (proposed unranked):** convert the v7-tiny predictor to a HypeNet-style
hybrid at matched params; score rank + decodability against the unconverted control. **Committed
outcome:** if the converted arm does not match the control's rank within CI, conversion does not
transfer to short-horizon visual latents and I-2 closes on the cheap path.

`Next in this track: full text - the conversion recipe and what it assumes about the source model.`


---

## Entry 2026-09-02-01 — HALO full text: the method is confirmed, the payoff regime is not ours

`PUBLISHED lib 2601.22156 · FULL TEXT READ 2026-09-02 (abstract-only 2026-09-01) · injected row I-2`

**HALO** (*Hybrid Attention via Layer Optimization*) converts pre-trained Transformers into RNN-attention
hybrids. The load-bearing step is **attention-layer selection** — which layers to leave unconverted:

> *"our objective is to identify which attention layers are most important for modeling recall abilities
> and leave them unconverted."*

Conversion costs **2.3B tokens, <0.01 % of pre-training data**, vs *">10B tokens"* for prior methods
(their Table 1). Pipeline is multi-stage (initialisation + staged distillation), with some stages
training-free. Produces **HypeNet**, converted from the Qwen3 series, using a **HyPE** position encoding
for length generalisation.

### ⛔ The regime caveat that downgrades I-2's expected payoff

The efficiency case is made at **128K and 1M** context — *"For 1M context length, the Qwen3 model runs out
of GPU memory"* — and hybrids *"enjoy significant inference speedups over Transformer-based models"*
precisely *"in long-context scenarios"*. **Our v7 predictor operates at K in {8, 60, 300} steps.**

The paper also concedes: *"distilled hybrid models typically underperform those trained from scratch"*.

⇒ **The method is confirmed and the motivation is downgraded in the same read.** Yesterday's re-scope
("CONVERT the trained v7 predictor rather than TRAIN a Mamba at matched params") stands as the correct
*method*; today adds that the *reason to want it* may not transfer to our context lengths at all.
**This is the honest shape of a Band-B transfer question answered against us.**

### Consequence

- **I-2 stays open but its priority should fall** pending P-6's benchmark.
- ⭐ **Run backlog P-6 FIRST.** Its pre-committed read — *"SSM wins only if ms/step is flat in K AND lower
  in absolute terms at K=60; flat-but-slower refutes"* — is exactly the regime test this paper leaves open
  for us, and it is 0-GPU then 4060.
- ⭐ **One idea transfers independently of the SSM question:** selecting components by *recall importance*
  rather than uniformly. That is a general compression heuristic and may apply to our deployment work.
- ⛔ Adopting HALO on the strength of a 1M-context result would be the `INHERITED` failure the operating
  standard bans.
