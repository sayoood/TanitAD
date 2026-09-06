# RESULT — H-TL-1 precondition: the headroom is real, and it is a RE-RANKING problem

**2026-09-05 · TanitAD_TrainingFlyWheel · Tier T0 · ⚠️ NON-PARITY pilot corpus (15 val episodes,
120 windows) · eval-mode deterministic · paired episode-cluster bootstrap, 4000 reps.**
**Evidence class: MEASURED (ours)** — `raw/htl1_headroom.json`, runner `code/htl1_headroom.py`.

---

## 0. Why this ran before any reasoner was trained

H-TL-1 claims *"reasoned priors beat the current heads on selection quality"*. ⛔ **That
experiment can only show something if choosing a DIFFERENT anchor would help.** If the scorer were
already at the best anchor the fan contains, no prior — reasoned or otherwise — could move the
metric, and a trained reasoner would return a null that says nothing about reasoning. So the
ceiling is measured first. *(The same discipline that killed the reach-lever and the
anchor-vocabulary lever earlier this session for zero GPU.)*

## 1. The numbers

| quantity | value | what it is |
|---|---|---|
| WORST anchor in fan | 49.963 m | — |
| **MEAN over fan** | **16.049 m** | the chance floor — picking at random |
| **SELECTED (scorer)** | **0.654 m** | what we ship |
| **ORACLE (best in fan)** | **0.250 m** | the ceiling any prior can reach |

* **skill = MEAN − SELECTED = +15.394 m — the scorer already earns 95.9 % of everything available
  over chance.** ⭐ It is emphatically not inert; this is the control that makes the rest readable.
* **HEADROOM = SELECTED − ORACLE = +0.404 m, which is 61.8 % of the error we ship.**
  Paired 95 % CI **[+0.267, +0.570] — SEPARATED.**

⇒ **H-TL-1 is worth running.** The remaining error is dominated by *choosing the wrong member of a
fan that already contains a much better one*.

## 2. ⭐ THE FINDING THAT CHANGES THE DESIGN: it is a top-5 re-ranking task

Where does the best anchor sit in the scorer's own ranking?

| | |
|---|---|
| median rank of the best anchor | **3 of 128** |
| top-1 | 32.5 % |
| **top-5** | **75.0 %** |
| top-10 | 83.3 % |

⭐⭐ **The scorer puts the right answer in its top 5 three quarters of the time and then picks
wrongly two times in three.** The geometry is already solved; what is unsolved is a *choice among a
handful of near-equivalent trajectories* — which is precisely a **semantic** judgement ("which of
these yields to the cyclist?"), not a geometric one.

**Three consequences, all of which make the design cheaper and sharper:**

1. **The reasoner does not score 128 anchors. It re-ranks ~5.** Compute on the tact collapses;
   the 300–500 ms budget stops being tight.
2. **H-TL-1's metric changes.** Not ADE over the fan — **top-1 accuracy within the top-5
   shortlist**, with the current scorer's 32.5 % as the baseline to beat and 75.0 % as the ceiling
   the shortlist allows.
3. ⚠️ **A prior over all 128 anchors is the wrong instrument** for this. The existing
   `*_to_anchor` ports emit exactly that, so the port is right but the *task* framing was too broad.

## 3. What is NOT established

⛔ That a language/reasoning prior *can* make that choice — only that the choice is available and
worth 0.404 m. ⛔ Anything about parity (this is the pilot corpus, 15 episodes). ⛔ Anything at T1.
⚠️ ORACLE here is *best-by-ADE*, which is a **trajectory** criterion; the four metric families may
rank differently, and a reasoner optimising ADE-oracle could still pick an unsafe member. The
top-5 re-rank must be scored on all four families before it is called an improvement.

## 4. Corrections this run forced

* ⚠️ **128 anchors, not 117.** My design docs said 117 (from refcv4b's `anchors.pt [117,8,2]`);
  this checkpoint's fan is **128**. Every anchor-count figure in v2/v3 is corrected to 128.
* ⚠️ **The checkpoint load needed care.** The current config registers a buffer
  `decoder.anchor_controls [128,2]` that this checkpoint predates. Diagnosed rather than
  suppressed: **exactly one missing key, it is a BUFFER not a parameter, zero unexpected keys, and
  the state-dict delta is exactly 256 = 128 × 2.** `strict=False` is safe here and the loader now
  *asserts* all three facts rather than assuming them — a bare `strict=False` would hide a real
  architecture mismatch.

---

## ⛔ CORRECTION 2026-09-05 — I asserted a gap refcv5 had already closed

**I wrote that REF-C has no agent/map grounding, "verified on a read of the full 154,975-char
source".** The read was real and the count was right — **and the claim was wrong**, because I
probed `stack/tanitad/refs/refc.py` (the SHIPPED v2.1/v4 line) and asserted it of **refcv5**, the
thing this design extends. `REFCV5_DESIGN_PLAN.md` contains **bev ×161, agent ×73, map ×54,
obstacle ×35, cross-att ×14**.

⚠️ **This is `CLAUDE.md`'s own rule — "absence found at ONE location is not absence" — and I broke
it while quoting a character count as if that made the probe authoritative.** A verified read of
the wrong artifact is still the wrong artifact. *(Same family as the DE-C152 epcache error: every
number right, the artifact wrong.)*

### What refcv5 actually already specifies, and what TanitLang must attach to

| WP | mechanism | why it matters for the language layer |
|---|---|---|
| **WP-3 `E-DDA-1`** | waypoint-indexed **grid-sample attention on the PV map** — sampled AT the candidate waypoints, with a permutation test that the attention map **moves when the candidate moves** | ⭐⭐ this produces **candidate-conditional scene features** — exactly the substrate a PER-ANCHOR rationale needs. TanitLang does not need to build it |
| **WP-6 `E-AGT-1`** | **agent tokens**, `cross_agent_layer` behind a zero-init gate; controls `shuffled_agent_tokens`, `raw_pixel_detection_floor` | the referential grounding for "the cyclist on the right". ⚠️ And the note that matters: *"obstacle.offline is a TRAIN-TIME label. At inference the tokens come from the trunk. An arm that reads the join at inference is REFUSED, not fixed."* — the vision-only rule, already enforced |
| **WP-7 `E-DDA-2b`** | the **four-family sub-metric selector**, replacing `argmax(conf)`, generator weights FROZEN | ⭐⭐⭐ **this IS H-TL-1.** Its success criterion — *"the selection gap closes with a CI excluding 0"* — is the gap I measured |

### ⇒ The corrected position

**TanitLang is not a new interface on REF-C. It is a PRODUCER for WP-7's selector, reading WP-3's
candidate-conditional features and WP-6's agent tokens.** The v3 doc invented an `AnchorReasoner`
with its own cross-attention; refcv5 already builds the per-candidate attention it needs.

**And my headroom result is WP-7's ceiling, measured:**

* selection headroom **0.404 m**, paired CI **[+0.267, +0.570] SEPARATED** — WP-7 has something to
  close;
* the best anchor sits at **median rank 3 of 128**, **top-5 75.0 %, top-1 32.5 %** ⇒ **WP-7's task
  is a top-5 re-rank**, not a search over 128;
* the scorer already earns **95.9 %** of what is available over chance — so WP-7 is refining a
  working selector, not replacing a broken one.

⚠️ **And refcv5 already anticipated my caveat**: WP-7 states *"ADE-only reading refused in advance.
Four families always."* My headroom is **ADE-only** and therefore a **precondition, not a result** —
the same 0.404 m must be re-measured on all four families before any re-ranker is called an
improvement. An ADE-oracle re-rank could pick an unsafe member.
