# YES — the programme's only T1 capability read was decoded through a RANDOM PROJECTION

**ArchInf FlyWheel · 2026-09-06 · 0 GPU** (CPU only. The A40 is running refcv5 — not touched.
Thor not touched. Dev-box RTX 4060 not used: all of P1–P2 is checkpoint forensics plus mutation
tests.)
Package: `TanitAD Research Lab/Architecture & Inference/Research/2026-09-06-metric-decode-hazard/`

---

## 0. THE ANSWER, IN ONE LINE

⛔⛔ **OUTCOME 1. A published metric-space number WAS decoded through the untrained readout — and
not one number but the whole of `D-T1-V7-READ`, the programme's only T1 capability read.** The
hazard is **not** an accounting defect.

The chain, each link MEASURED and each with a control that had to read a known value:

| link | MEASURED | control |
|---|---|---|
| the readout never trained | 9 of 9 banked v7-tiny arms read `AT_INIT`; `net.0` LayerNorm bit-exactly ones/zeros; `max\|w\|/init-bound` **0.9999981** on `net.1.weight` | a REAL flagship readout banked at **step = 1** reads **1.0078** — ONE optimizer step crosses the bound |
| the same readout in every arm | `step_readout_op` takes **3 distinct fingerprints** across 9 arms | the trained control `predictor_op.heads.1.weight` takes **9** |
| the evals decoded through it | **7 banked T1 artifacts** record `"decoder": "grounding.step['op']"` on a v6/v7 `stack` checkpoint | **2,616 JSON files scanned, 0 unreadable**; every other artifact carrying that string is flagship/v5f lineage, where the readout really trained |
| the register quotes those metres | **3 rows**, and the exposure is exactly the predecessor's E1 | four unswept T1 artifacts checked in both bare and comma-formatted number forms: **no register row quotes them** |

⭐ **The single sharpest fact:** `emao14_30k`, `emao14_30k_tauramp` and `o14fut30k` — the three arms
`D-T1-V7-READ` and `D-V7F-T1-NO-PAIRED-CI` compare — carry a **BIT-IDENTICAL** `step_readout_op`.
Three arms that trained to three completely different predictors were scored through **one** random
projection. Any difference between them in metres is not a difference in their world models.

⇒ **P2 shipped a REFUSAL, not a retrain.** Every path that can emit a metre through an unfitted
readout now raises, proven two-sided on the real `.pt` files, and proven *wired* by executing each
path rather than by reading its source.

---

## 1. P1 — THE CONSUMER TRACE, AND ITS CONTROLS

### 1.1 Five production sites read `V6Stack.step_readout_op` — the hazard row named three

Traced from source in a clone verified byte-identical to `HEAD` (blob comparison with the
40-character shape check, so a mount outage reads INCONCLUSIVE and never MATCH). Second probe:
the same trace by class name (`StepDisplacementReadout`) and by attribute (`step_readout`, no
`_op`), which is what separated the two lineages below.

| # | site | what it emits | in the hazard row's list? |
|---|---|---|---|
| 1 | `models/v6.py` `V6Stack.roll_consistency` | `accumulate_se2` waypoints — **metres** | ✅ |
| 2 | `tanitad/eval/v6_probe_trunk.py` `V6Grounding.step["op"]` | the eval seam | ✅ |
| 3 | `scripts/probe_saliency_p9.py` `builtin_targets` | speed **m/s**, yaw **rad/s** | ✅ |
| 4 | **`taniteval/tools/t1_eval.py` `--grounding-readout`** (through #2) | **ADE/FDE/heading/speed in metres** | ⛔ **MISSING** |
| 5 | **`scripts/stage_a_probes.py`** (through #2) | `decode_transitions(...)` — **metres** | ⛔ **MISSING** |

⛔ **The two the row missed are exactly the two that PUBLISHED.** `probe_saliency_p9` appears in the
register only inside the hazard row itself; `roll_consistency` appears in **zero** rows (its `0.7164`
and `+5.9787` figures live in a v6.py docstring and are W7/flagship-era, not register claims — both
searched and absent). Site 4 produced every exposed number. Site 5 is reachable and has produced
**zero** banked artifacts on a v6/v7 stack — a first-time-hazard, not a retraction.
⇒ the banner in `train_v6_staged.py::GRADREACH_EVAL_HAZARDS` is corrected to name all five.

⚠️ **AND ONE APPARENT CONSUMER IS NOT ONE — scope, stated because it would have inflated the
radius.** `scripts/compare_arms.py` also does `step_readout = gr.step["op"]`, but its `gr` is
`HierarchicalGrounding` loaded from a **flagship** checkpoint's `"grounding"` key, i.e. a
**separately trained** readout. Same expression, different module. Grepping the *phrase* would have
added it; reading the *construction* removed it.

**Completeness of the trace, and the one bypass that remains.** `taniteval/taniteval/loaders.py`
routes **only** flagship (`HierarchicalGrounding`) and refa-plus (`ck['step_readout']`) — both
separately trained — so the ~25 `taniteval` modules that consume a `step_readout`
(`bench`, `rollout`, `closedloop`, `hierarchy`, `eval_four_families`, `render_openloop_video`, …)
are **flagship-lineage and not exposed**. `actdiv_anchored.py` and `transition_probe.py` call
`load_trunk_auto` but bind the grounding as `_g` and never touch `.step` (verified: `.step[` count
**0** in both), so they are unaffected by the refusal.
⚠️ **The bypass I did not close, stated rather than discovered later:** `V6ProbeTrunk` exposes
`.stack`, so `world.stack.step_readout_op(...)` reaches the module directly. That is a deliberate
act, not an accidental one — every *existing* direct-access site (`probe_saliency_p9`) is guarded —
but it is a hole, and closing it would mean guarding the module's own `forward`, which would fire
during TRAINING at O1 > 0, i.e. exactly when the readout is legitimately being fitted.

### 1.2 The readout is at initialisation — three independent evidences

`raw/p1_readout_forensics.json`, all 9 banked v7-tiny checkpoints.

| evidence | what it reads | why it is not an estimate |
|---|---|---|
| **LayerNorm identity** | `net.0.weight` **all exactly 1.0**, `net.0.bias` **all exactly 0.0**, in **9 of 9** arms | `nn.LayerNorm`'s own init. A bit-exact identity, robust to stored dtype |
| **Linear init signature** | `net.1.weight` `max\|w\|/b` **0.9999981–0.9999995**, `std/(b/√3)` **1.00021–1.00028**; `net.3.weight` **0.9992220–0.9997801** / **0.99693–1.01248** | `nn.Linear` draws `U(-b, b)`, `b = 1/√fan_in`. A fresh draw sits **just below** the bound; training crosses it |
| **the run's own record** | `w_o1_ctrl / w_o1_fact / w_o1_scene = (0.0, 0.0, 0.0)` in **9 of 9** `config.json` | `stage_a_losses` is the only gradient path, and the trainer guards it on exactly those three |

**Fingerprint census — the control that makes this a measurement rather than a reading:**

| tensor | distinct md5 across 9 arms | reads |
|---|---:|---|
| `step_readout_op.net.1.weight` | **3** | 6 arms share one · 2 share another · `rdw8p30k` alone |
| `step_readout_op.net.3.weight` | **3** | the same partition, co-varying perfectly |
| ⭐ **CONTROL `predictor_op.heads.1.weight`** | **9** | one per arm — the reader reads real per-arm bytes |

⚠️ **One check of mine came back negative and I am not going to dress it up.** A bit-exact
comparison against `StepDisplacementReadout(2048)` constructed **standalone** at seeds 0–3 matched
**none** of the 9 arms. That does **not** contradict the predecessor's *"bit-identical to a fresh
seed-0 init"*: the RNG stream depends on construction ORDER, and theirs built the whole `V6Stack`.
My instrument answered a slightly different question; the three evidences above and the 3-vs-9
census are the ones that carry the claim.

### 1.3 The artifacts say, in their own words, which decoder ran

`raw/p1_consumer_trace.json` · `raw/p1_artifact_sweep.json`. Scanned **2,616** JSON files;
**0 unreadable** (stated, because "no matches" from a file that could not be opened is a claim about
the search, not the content).

**7 artifacts record a `grounding.step['op']` decode, and all 7 are v6/v7 `stack` checkpoints:**

| artifact (repo path under `TanitAD Research Lab/Architecture & Inference/…`) | ckpt | md5 |
|---|---|---|
| `Implementation/incoming/2026-08-30-t1-first-v7-read/raw/t1_emao14_30k.json` | `/home/nvidia/v7tiny/emao14_30k/ckpt.pt` | `632fb91e…` |
| `…/t1_emao14_30k_tauramp.json` | `…/emao14_30k_tauramp/ckpt.pt` | `d1af4e24…` |
| `…/t1_o14fut30k.json` | `…/o14fut30k/ckpt.pt` | `45cffdd6…` |
| `Implementation/incoming/2026-08-27-t1-parity-first/raw/t1_postrain30k.json` | `…/postrain30k/ckpt.pt` | `54dec7c8…` |
| `…/t1_postrain30k_freeze.json` | `…/postrain30k_freeze/ckpt.pt` | `52be525b…` |
| `Research/2026-08-24-action-conditioning-and-heldout/raw/t1_nonparity40_rdw8p30k.json` | `v7tiny_rdw8p30k/ckpt.pt` | — |
| `…/t1_v7_smoke.json` | `v7tiny_rdw8p30k/ckpt.pt` | — |

Each carries `rollout_provenance.decoder.kind = "grounding.step['op']
(StepDisplacementReadout)"` and `.source = "the checkpoint's OWN metric decoder"`.
⚠️ **That `source` line reads like a provenance guarantee and is only a source.** It is the sentence
that made the defect invisible: the decoder *was* the checkpoint's own — the checkpoint's own
**untrained** one.

⭐ **The scope control that stopped this over-claiming.** Five further artifacts in the repo carry the
same string — `w7_gate_k8/k32/k64.json`, `w7_full_gate.json`, `w7_w4r_k32_gate.json`,
`rung1_v4_readout_swap.json`, `budget_report.json` — and every one is **flagship / v4 / v5f / v58f**
lineage, where `step_readout` is separately trained. **They are not exposed.** *(`rung1_v4_readout_swap`
is a nice historical echo: it re-initialised that readout at random ON PURPOSE, as an ablation.)*

### 1.4 What an at-init readout actually emits — scoped, and NOT quoted as "what T1 saw"

`raw/p1_readout_forensics.json`, `raw/p1_emission_vs_corr.json`. The readout's first layer is a
**LayerNorm over the concatenated pair**, so the input scale is normalised away by construction —
the emitted magnitude is a property of the weights, not of the data. Verified: inputs scaled by
`1e-3 / 1 / 1e3` give **bit-identical** output statistics.

* Banked `emao14_30k` readout on synthetic latents, `k = 20`, `dt = 0.1`: **0.2775 m/step**
  (implied **2.78 m/s**), end displacement **3.17 m** over 2.0 s.
* **Invariant to how much consecutive latents correlate**: sweeping `ρ = corr(z_t, z_hat)` from
  **0.0 to 1.0** moves it only **2.767 → 2.789 m/s**. So "the predictor restates rather than
  transports" does not explain the emission — the readout does.
* ⭐ **SEED-ARBITRARY, which is the property that makes the number inadmissible.** Decoding the same
  latents through other-seed inits moves the endpoint by **2.12 / 3.23 / 5.26 m** with endpoint
  cosine **0.729 / 0.286 / −0.571**. **CONTROL: the same readout twice differs by exactly 0.0.**
  ⇒ change one arbitrary integer and every T1 metre changes, with nothing about the trained model
  changing.

⚠️ **SCOPE, BINDING.** Those magnitudes are on **synthetic Gaussian latents**, not on the real
banked rollouts (which live on `thor:/home/nvidia/t1dumps/`, not reachable from this box). The
artifact's own `under_progress 0.9925` is **more** degenerate than my 0.83, so **2.78 m/s is an
illustration of the mechanism and an upper bound — it must not be quoted as the emission T1 saw.**

---

## 2. P1 — THE E1 ROWS, CLASSIFIED PER ARTIFACT

⛔ **Every number below was decoded through a readout MEASURED to be at initialisation.**
Register hits found by searching each artifact value in **both bare and comma-formatted** form, with
a same-breath non-zero control (`H-ESTIM-SEED-1` → 33) and a must-be-zero control (0).

| row | numbers, and the artifact each comes from | verdict |
|---|---|---|
| ⛔ **`D-T1-V7-READ`** | ade **14.069**/**13.879**, fde **26.297**, LAT_heading **94.63°**, LON_speed **10.703**, o14fut ade **14.293** — all from `t1_emao14_30k.json` / `t1_o14fut30k.json`, whose own `decoder` field is `grounding.step['op']`. The artifact's `ade_dense_m.mean` is **14.0688**, i.e. the register's 14.069 IS this file | **FALSE RESULT as a capability read.** The row's own hedges do not save it: it presents these as *"THE PROGRAMME'S FIRST T1 CAPABILITY READ … a FLOOR"*, and a floor decoded through a random projection is not a floor of the model |
| ⛔ **`D-V7F-T1-NO-PAIRED-CI`** | its **correction half**: cross-track **1.0722 / 1.1704**, heading **−0.4957**, speed **−0.1974**, tauramp ade **13.8645** | **the correction half is FALSE**; ⭐ its **structural half stands untouched** — `paired_decision_grade` and `paired_legacy` are `{}`, which I re-verified directly in all five reachable artifacts. That is a fact about the JSON, not about the decoder |
| ⛔ **`H-ARCH-ACTINS`** | the "~1 %" closed-loop-vs-hold-action gap (`emao14_30k` ade **+1.37 %**, fde **+0.64 %**) — arithmetic on the same metres | **the EVIDENCE is void; the HYPOTHESIS is not.** ⭐ And it does not need re-establishing: `MM-E10` already resolved it in **latent space**, a route that never touches the readout |
| ⭐ **`D-V7F-READOUT-DEAD`** *(not in the swept E1 — I add it)* | `speed_bias_mps` **−10.381**, `under_progress` **0.9925**, holdv0 delta **10.2192** | ⭐ **CONFIRMED, NOT INVALIDATED.** Its claim is *about the readout* — "the readout emits ≈0 ego motion". Measuring an untrained module through itself is the legitimate case, and this turn supplies the mechanism it was missing |
| ✅ **`M-WPSTEPS-UNITS`** *(also unswept; also mine)* | mentions **−10.381** | **NOT exposed.** It cites the figure only to **withhold** a derived claim. A row that declines to derive from a number is not exposed by that number being wrong |

⇒ **E1 = 3 exposed rows, exactly as swept.** My extension found two more rows *touching* the
numbers and **neither is a new exposure**. And four further T1 artifacts (`postrain30k`,
`postrain30k_freeze`, `rdw8p30k`, `v7_smoke`) have **no register row quoting them** — every apparent
hit was a 2-decimal collision with an unrelated REF-A/REF-C row, none matched at 4 decimals.

### 2.1 ⛔⛔ THE SPECIFIED REMEDIATION IS INFEASIBLE — I WENT AND LOOKED

Both the sweep and `D-P1-EXPOSURE-SWEEP-82` prescribe: *"re-analyse the banked
`thor:/home/nvidia/t1dumps/*/dumps/` rollouts with a readout fitted at eval time."*
⛔ **That cannot be done.** MEASURED on Thor (reachable; **idle** — GPU 1 %, load 0.24 — so this is
not a compute blocker): each arm's dump directory is **2.7 MB, 40 `ep*.npz`**, and every file holds

`g [173, 20, 2]` · `cl [173, 20, 2]` · `ha [173, 20, 2]` · `v0 [173]` · `ws [173]` · `eid` · `clip_index`

— i.e. **the readout's OUTPUT and nothing else.** Its INPUT, the per-step latent transition
`(z_prev, z_hat)`, was never dumped. **A readout fitted at eval time has nothing to be applied to.**
⇒ the remediation is a **T1 RE-ROLL with latent dumping**, not a re-analysis: a `--dump-latents`
path in `t1_eval` plus ~11 min GPU per arm on an idle Thor. Cheap, but a new work package with its
own pre-registration — named here rather than smuggled in.

### 2.2 ⭐⭐ WHAT THE DUMPS *CAN* STILL SETTLE — AND IT IS THE SHARPEST NUMBER IN THIS TURN

`raw/p1_between_arm.py` / `.json`, 0 GPU, on Thor's own CPU. The three T1-read arms carry a
**bit-identical** `step_readout_op`. So: how much of their decoded trajectory is the ARM?

| control / measurement | value |
|---|---:|
| ⭐ **CONTROL — each arm's ADE recomputed from its dump** | `emao14_30k` **14.0688** · `tauramp` **13.8645** · `o14fut30k` **14.2926** — reproducing the register's 14.069 / 13.8645 / 14.293 **exactly**, so these dumps ARE the published artifact |
| ⭐ **CONTROL — GT vs GT across arms** | **0.0000000000** on all three pairs: identical corpus, identical window grid, paired comparison valid |
| each arm vs GROUND TRUTH (`cl`) | mean **14.0753 m** |
| ⛔ **each arm vs EACH OTHER (`cl`)** | **1.4169 / 2.4521 / 2.3892**, mean **2.0861 m** |
| **between-arm / arm-vs-GT** | **0.1482** |
| the deltas the register publishes (`cl − ha`) | **+0.1895 / +0.0588 / +0.1766 m** |

⛔⛔ **THREE ARMS THAT TRAINED TO THREE DIFFERENT WORLD MODELS AGREE WITH EACH OTHER 6.7× MORE
CLOSELY THAN ANY OF THEM AGREES WITH THE ROAD** (14.0753 / 2.0861 = **6.75×**) — and the per-arm
effects the register reports (**0.0588–0.1895 m**) are **11.0–35.5× SMALLER** than that between-arm
spread, and **74–239× smaller** than the error itself. `D-T1-V7-READ`'s headline ranking
(`emao14_30k` **14.0688** beating `o14fut30k` **14.2926**) is a **0.2238 m** difference read against
a **2.0861 m** spread.

⚠️ **SCOPE, and I am stating it because the causal step is the tempting over-claim.** The three arms
share more than the readout — the same corpus, the same window grid, the same recipe family. So
this measures that the decode is **dominated by something the arms share**, and the bit-identical
readout is the only thing they share *exactly*; it does not, alone, prove the readout is the cause.
⭐ **What it does establish without any causal step is the one that matters**: the published effects
sit an order of magnitude inside the spread produced by arms sharing an unfitted decoder, so they
are **unresolvable at this scale** — which is `D-V7F-T1-NO-PAIRED-CI`'s "no interval of any kind"
given a magnitude for the first time.

### 2.2b ⛔⛔ AND THE EXPOSED METRES ARE IN `MODEL_REGISTRY.md` — WHICH OUTRANKS THE REGISTER

The E1 sweep covered `GOALS_AND_CLAIMS.md` **only**. Scanning **1,390 `.md` files (0 unreadable)**
for each figure finds them in **six** further documents, and the first is the one that matters:

⛔ **`Project Steering/MODEL_REGISTRY.md`** — *"the ONLY quotable source for model facts"* per
`CLAUDE.md` — carries the full three-arm table
`| arm | ade_dense_m cl / ha | fde_last_m cl / ha | LAT_cross_mae_m cl / ha | LAT_heading_deg cl / ha |`
with `emao14_30k` **14.069 / 13.879 · 26.297 / 26.131 · 1.310 / 1.124 · 94.63 / 95.13**,
`o14fut30k` **14.293 / 14.116 · …**, `emao14_30k_tauramp` **13.864 / 13.806 · …**, followed by
*"IN EVERY DISTANCE METRIC, FOR EVERY ARM, THE CLOSED-LOOP ARM IS WORSE…"*.
Plus `Project Steering/Reports/2026-09-02-2300-v7-readiness.md`,
`Implementation/incoming/2026-08-30-t1-first-v7-read/RESULT.md`,
`Research/2026-08-30-t1-floor-action-sensitivity/RESULT.md`,
`Research/2026-09-03-v7f-design/raw/EVIDENCE_TABLE.md` and
`Benchmarks & Evals/Research/2026-08-30-anti-echo-control-precedent/RESULT.md`.

⇒ **a defect the register now flags is still quotable from the source that outranks the register.**
⛔ **WORK ITEM FOR THE MASTER MIND, not done by me** — agents do not edit `Project Steering/*.md`
beyond their own appended register row.

⭐ **VERIFIED ON THE LIVE FILE.** The mount refused `MODEL_REGISTRY.md` on **7 of 7** reads
(`Errno 22`) in an outage window; retried after a same-breath control (`CLAUDE.md` 77,180 B, five
consecutive reads) and the live file (**443,136 B**) carries every figure, **1 occurrence each**,
with control `ade_dense_m` = 1. ⚠️ The **six-document list** comes from a 1,390-file `.md` scan over
a local clone, so it is **INHERITED for the five non-registry documents** and MEASURED for the
registry itself.
⚠️⚠️ **AND A NEAR-MISS WORTH LOGGING: an earlier LIVE scan returned EMPTY for every figure AND for
its own control.** That is a mount-outage zero — *"0 hits is a claim about the SEARCH, not the
content"* — and the only reason it was not written up as "the numbers are not in the steering docs"
is that the control was in the same breath and read zero too.
⚠️ Three further `14.293` hits (`2026-07-26-program-harvest/H4_CONTRADICTIONS.md`,
`PROGRAM_HARVEST.md`, `2026-07-26-v2-throughput/V2_THROUGHPUT_BENCH.md`) are **3-decimal collisions
from July, unrelated to this T1 read** — stated so the radius is not inflated.

### 2.3 ⭐ What re-establishes each

1. **`D-T1-V7-READ`** — a **T1 re-roll with latent dumping** (§2.1), then a readout **fitted on
   held-out windows**, restated as *latent decodability*, not driving skill. Or re-run T1 on an arm
   trained at `--w-o1-ctrl > 0`.
2. **`D-V7F-T1-NO-PAIRED-CI`** — compute the paired CIs **in the same pass** as that re-decode, so
   deltas and intervals come from one admissible decode. Its structural half needs nothing.
3. **`H-ARCH-ACTINS`** — already answered by `MM-E10`'s latent-space action-divergence probe.
   ⛔ Nothing here re-opens `MM-E10`: it reads latents, never the readout.

⚠️ **AND A CONSEQUENCE THE SWEEP DID NOT DRAW.** `V7_LAUNCH_GATE` **P1** — *"no v7 arm has ever
beaten its own hold-action control"* — is stated over these same metres, and `D-V7F-GATE-IS-NOT-EMA`
carries it forward as the v7f gate. **A gate criterion resting on a random projection cannot
adjudicate anything.** ⛔ I am **not** amending it: that is a PI/gate decision. It is named here as
a blocker with its exact remedy (item 1 above), which is the cheapest experiment that could still
make the gate readable and it needs **no GPU**.

---

## 3. P2 — THE REFUSAL, PROVEN BOTH WAYS AND PROVEN WIRED

⭐ **A refusal, not a retrain, and not switching O1 on** — that would change the recipe every banked
arm was trained under, and the arms are the comparison basis. *(If the readout should be trained, it
is a pre-registered future arm: `--w-o1-ctrl > 0` at v7-tiny costs one 30 k run and its own control.)*

### 3.1 What shipped

* **`models/v6.py`**: `UntrainedMetricReadout` (a **third** claim, deliberately distinct from
  `FrozenExternalViolation` = PROVENANCE and `GradUnreachableViolation` = LOSS GRAPH, checked at
  LAUNCH; this one is about MEASUREMENT, at EVAL time, where no launch preflight runs at all),
  `metric_readout_status`, `assert_metric_readout_trained`.
  Verdicts: **`AT_INIT` · `TRAINED` · `CONTRADICTION` · `UNKNOWN`** — only `TRAINED` decodes.
* **`CONTRADICTION` is a designed state, not a leftover**: the weights moved *and* the record says
  nothing could reach them. One of the two facts is then wrong, which is the worst case to publish
  through, so it refuses.
* **`UNKNOWN` refuses too.** *"I could not check"* is exactly where a plausible number is most
  dangerous. It is a loud, fixable failure with a named opt-in.
* Wired at **five** sites; `--allow-untrained-readout` at **three** tools
  (`t1_eval`, `stage_a_probes`, `probe_saliency_p9`) so a diagnosis of the readout ITSELF can still
  run — `D-V7F-READOUT-DEAD` did exactly that, legitimately, and a guard that leaves a live tool
  unrunnable is the shape that gets deleted.
* ⭐ **The artifact now says whether its own decoder was ever fitted**:
  `rollout_provenance.decoder.readout_training_status` in every `t1.json`. An artifact is opened in
  isolation far more often than its run record — which is why *"the checkpoint's OWN metric
  decoder"* was able to read as a guarantee for months.

### 3.2 ⛔ THE MUTATION PROOF — on the REAL banked files (`raw/p2_mutation.json`)

| case | result |
|---|---|
| **A · NEGATIVE — the 9 banked v7-tiny `step_readout_op`** | ⛔ **REFUSED, 9 of 9**, verdict `AT_INIT` |
| **B · MUTATION — the same tensors after ONE AdamW step** at the arm's own `lr 1e-4, wd 0.05` | ⭐ **PASSED, 9 of 9**, verdict `TRAINED`, `max_ratio` **1.00639** |
| **B′ · the same mutation asked WITH the arm's own `O1 = 0` record** | ⛔ `CONTRADICTION` — the guard reporting a real inconsistency, not a false positive |
| ⭐ **D · POSITIVE CONTROL ON A REAL ARTIFACT** — 4 flagship v4 `grounding['step.op.*']` readouts, banked at **step = 1** | ⭐ **PASSED**, verdict `TRAINED`, `max_ratio` **1.00783** |
| **C · END-TO-END through `t1_eval`'s own decode expression** `grounding.step["op"]`, on `v7tiny_emao14_30k` | flag **off** ⇒ ⛔ **REFUSED**; flag **on** ⇒ passes **and records** `verdict: AT_INIT` with all three evidences |

⭐ **D is the half that stops this guard being deleted, and it is a REAL artifact rather than a
synthetic mutation**: a genuinely trained readout from a *different lineage*, saved after a single
optimizer step, passes. So the discriminator is not "v7 vs everything else" — it is "fitted vs not".

**Detection floor, MEASURED and carrying its `n`:** `max|w|` sits ~`1/n` below the init bound, so the
floor scales with the tensor — flip at relative weight move **3e-5** for `n = 65,536`, **1e-5** for
`n = 524,288` and for `n = 2,097,152` (`net.1` at the banked geometry). ⚠️ The boundary near 1e-6 is
**draw-dependent**, so the test brackets it rather than pinning a knife-edge. The margin that matters:
one AdamW step reads **1.0064**, some **three orders of magnitude** past the floor.

### 3.3 ⭐⭐ AND THE TEST THAT IT IS ACTUALLY CALLED — the predecessor's lesson, applied

`stack/tests/test_metric_decode_refusal.py`, **25 tests**, split by claim:

* **§1–2 CORRECTNESS** (13): fires at init, on every seed; does not fire after 1/2/5 real AdamW
  steps; `max|w|/bound` is the load-bearing statistic; the record channel; `CONTRADICTION`;
  `UNKNOWN`; the escape hatch returns the verdict; tiny tensors are skipped **because measured**
  (`net.3.bias`, n = 3, reads `std_ratio 1.583` at a genuine init — testing it would refuse correct
  readouts at random).
* **§3 WIRING** (7), by **spying on the guard and EXECUTING each path**, not by reading source:
  `V6Grounding.step` · `roll_consistency` · `probe_saliency_p9.builtin_targets` ·
  `load_trunk_auto` threading the opt-in · each with its **non-firing control** on a trained readout.
  Plus a source assertion for `t1_eval` (running it needs a corpus) pinning the pass-through **and**
  the artifact stamp, each with a same-breath token that must also be present.
* ⛔ **`test_WIRING_the_guard_has_callers_OUTSIDE_its_own_test_file`** — the failure this whole
  family exists for: `assert_frozen_external` was fully built, pinned in both directions, and called
  by **nothing** outside its own test file, so the trap it guarded happened anyway.

### 3.4 ⚠️ TWO DEFECTS THE BUILD ITSELF PRODUCED, BOTH CAUGHT BY A CONTROL

1. ⛔ **THE MEMO ANSWERED A DIFFERENT QUESTION FROM THE ONE ASKED.** The status is memoised (a full
   pass over ~2.1 M weights, and `roll_consistency` runs inside an MPC inner loop). Keyed on the
   MODULE alone, a first call carrying an `O1 = 0` record cached `CONTRADICTION`, and the next call
   **without** the record read that back instead of `TRAINED`. **This is the programme's own scope
   error in a cache's costume**, and only the mutation arm's two-sided reading exposed it. Fixed:
   the memo is keyed on the record. Pinned by `test_the_MEMO_is_keyed_on_the_record_…`.
2. ⚠️ **`except Exception` LET A `SystemExit` THROUGH AND TURNED A CORROBORATOR INTO A HARD
   DEPENDENCY.** `_run_args` **exits** when a checkpoint travelled without its config; the record is
   only corroborating, but the load then died. Caught by an existing test, not by me.

### 3.5 What the guard deliberately does NOT refuse, and the measurement behind it

`V6Stack.emit`'s **two internal** `roll_consistency` calls pass the opt-in explicitly, with the
reason inline: one is the P7-validated fallback **variance** (never its metre value,
permutation-invariant so it cannot select), the other an MPC **regulariser** at `mpc_w_consist`
default **0.0**. **MEASURED: zero register rows quote a roll-consistency value from a v6/v7 arm.**
Refusing there would break a live planner to protect a number nobody publishes. An **external**
caller still refuses.

---

## 4. THE SUITE, AS A CONTROLLED COMPARISON

Two trees, so the control could not be contaminated by editing the tree it ran in.

| | tree | state |
|---|---|---|
| **BASELINE** | `C:\Users\Admin\tanitad-mdhaz-base` | ⭐ **blob-verified against git HEAD for all 8 touched paths**, with the 40-character shape check — and **re-verified after HEAD advanced twice mid-turn** (`63e2609` → `1964d51`): still identical, so no sibling's work is being compared away |
| **AFTER** | `C:\Users\Admin\tanitad-mdhazard` | the same tree plus this turn's edits |

The diff surface between them is **exactly** the 7 edited files plus 1 new test file — verified by a
full recursive `diff -rq`, so nothing else can be moving the numbers.

| | failed | passed | skipped | xfailed | errors | wall |
|---|---:|---:|---:|---:|---:|---:|
| **BASELINE** | **65** | 6,395 | 121 | 2 | **51** | 1,077.5 s |
| AFTER (first run) | 68 | 6,417 | 121 | 2 | 51 | 1,035.7 s |
| **AFTER (final)** | **65** | **6,420** | 121 | 2 | **51** | 818.8 s |

* ⭐⭐ **FAILURE-ID SET DIFF, FINAL: EMPTY IN BOTH DIRECTIONS.** `comm -13` (new) = **0**,
  `comm -23` (newly fixed) = **0**, over **116** ids on each side. The diff is over the IDs, not
  the counts, so an equal-count swap could not hide in it.
* ⭐ **The `passed` delta is +25 and is accounted for EXACTLY**:
  `test_metric_decode_refusal.py` collects **25**. 6,395 + 25 = **6,420**. No existing test was
  quietly deleted or skipped to make the numbers work.
* ⛔⛔ **THE FIRST AFTER RUN HAD 3 REGRESSIONS AND THEY WERE THE GUARD DOING ITS JOB** —
  `test_p9_saliency.py`'s three target tests build a FRESH tiny stack (at init by construction) and
  ask `builtin_targets` for the speed/yaw targets, which decode metres. ⭐ **The controlled
  comparison is the only thing that found them:** every targeted run on the files I had edited was
  green. Fixed by making the opt-in explicit where the test interrogates the SUPPORT GRAPH rather
  than a metric value, and by auto-allowing `--synthetic` — not a loophole: that arm is random
  weights AND random frames by construction and the script already stamps it `quotable: NONE`.
* ⚠ **The baseline is NOT green, and that is why the verdict is the set diff.** Both trees are
  `stack/`-only copies without most repo-root data, so a large block of environment-dependent tests
  fails or errors in **both** columns and cancels out. Full table:
  `raw/SUITE_BASELINE_VS_AFTER.md`.
* ⚠ The wall-clock differs only because the runs used different `OMP_NUM_THREADS` (2 vs 4) and
  did not overlap; it carries no signal.

---

## 5. EVIDENCE CLASSES, TRAPS MET, ARTIFACTS

Every number above is **MEASURED (ours)**, on CPU, in `C:\Users\Admin\tanitad-mdhazard`, with the
import origin re-checked (`tanitad.__file__` → the clone, not the G: editable install).

⚠️ **The G: mount flapped hard.** Reads of the two predecessor `RESULT.md` files, `git rev-parse`
and `git cat-file` each failed for long stretches while listings kept working. Every read was made
through a **retry loop with a same-breath control** and every absence claim waited for a control
success. The banked T1 JSONs were finally read from a full local clone and md5-anchored.
⚠️ `git ls-files` (index-based, positive) was used to enumerate; **no absence claim rests on an
empty git result**.

| artifact | what it holds |
|---|---|
| `raw/p1_readout_forensics.py` / `.json` | the 9-arm fingerprint census, the 3-vs-9 control, the LayerNorm identity, the emission and its scale-invariance, the seed-arbitrariness table |
| `raw/p1_consumer_trace.py` / `.json` | the banked T1 artifacts' own decoder provenance + the register number scan in both number forms |
| `raw/p1_artifact_sweep.py` / `.json` | all 2,616 JSONs, classified by CHECKPOINT LINEAGE (the flagship/v6 split) |
| `raw/p1_extend_e1.py` / `.json` | the four unswept T1 artifacts vs the register — a clean negative |
| `raw/p1_emission_vs_corr.json` | emitted motion vs latent autocorrelation |
| `raw/p2_calibrate.py` / `.json` | the two-sided calibration incl. the real flagship step-1 readouts |
| `raw/p2_mutation.py` / `.json` | the mutation proof A/B/B′/C/D on the real banked files |
| `raw/SUITE_BASELINE_VS_AFTER.md` | the controlled suite comparison |
