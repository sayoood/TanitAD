<title>Cosmos-Reason1-7B vs Cosmos-Reason2-8B — controlled head-to-head on route labeling</title>

# Cosmos-Reason1-7B vs Cosmos-Reason2-8B — the route-labeling head-to-head

**Author:** Data Engineering (agent run, PI request). **Date:** 2026-07-20. **Status:** measured, both arms
run fresh back-to-back on one pod. **Pod:** `tanitad-eval` (A40 46 GB, torch 2.8.0+cu128, transformers 5.14.1).
**Decides:** which VLM labels the TanitDataset v1 corpus, and which mints the VLM-pending slots of the frozen
[v3 goal vocabulary](../../Architecture%20%26%20Inference/V3_GOAL_VOCABULARY_V1.md).
**Anchors:** [pilot note](2026-07-20-cosmos3-vlm-pilot.md) · [VLM survey](2026-07-19-vlm-augmentation-survey.md) ·
[TANITDATASET_V1_STRATEGY §4/§8](../TANITDATASET_V1_STRATEGY.md)
**Artifacts:** `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-20-vlm-reason1-vs-reason2/`
· harness `stack/scripts/vlm_model_compare.py` + `stack/scripts/vlm_compare_score.py`

---

## TL;DR — the verdict is "neither, for ROUTE"

**Neither model may mint the `ROUTE` slot.** On 200 windows over 40 val episodes, identical prompt, same GPU,
back to back:

- **Cosmos-Reason2-8B finds turns and cannot say which way.** Turn-detection recall **76.8 %** (63/82) —
  but conditional on committing to left/right its hit rate is **57.1 %** (36/63), 95 % CI **[0.400, 0.745]**
  (episode-cluster bootstrap, 19 episodes) — **the interval contains 0.5**. The bias-free test is worse:
  Fisher exact on {GT left|right} × {said left|right} gives **odds ratio 1.32, p = 0.78**. Its direction call
  carries **no information about the true direction**.
- **Cosmos-Reason1-7B says the right thing when it speaks, and almost never speaks.** Direction hit rate
  **78.6 %** (22/28), CI **[0.645, 0.923]** — **excludes 0.5**; Fisher **OR 12.44, p = 0.006**. But it detects
  only **34.2 %** of turns (28/82); it answers `straight` on 89/89 true-straight windows *and* on 53 of 82 true
  turns.
- **Overall 3-class accuracy is a statistical tie**: 64.9 % vs 69.0 %, paired delta **−0.041, CI [−0.116, +0.030]**
  (paired episode-cluster bootstrap, 38 episodes), McNemar **p = 0.21**. Picking on headline accuracy would pick
  nothing.
- **Reason2's errors are correlated, which is worse than noise.** Conditional on detecting a turn it is right
  **73.7 %** of the time on left turns and **32.0 %** on right turns — *below chance on right turns*. A labeler
  with a systematic left bias injects a directional prior into the corpus; random noise would at least average out.

**Action:** `ROUTE` stays **kinematic** (`refb_labels.route_from_future_v21`), exactly as `VTARGET` and `LONMODE`
already did after the 48-clip pilot. **The pattern is now three-for-three: every slot the ego track can compute,
the VLM gets wrong.** For the non-geometric slots the VLM is still the right tool and **Reason2 is the better
engine** (0 parse failures, 0 ROUTE enum violations, 21 % fewer generated tokens, ~9 % faster, correct
abstention behaviour).

---

## 1. Design — and the one confound we deliberately did not fix

Both arms ran **on the same pod, over the same 200 windows, with the same prompt version, back to back**.
pod3 holds 400 banked Reason2 records, but they were produced on a *different* val build
(`physicalai-val-f1b378f295ae`, 80 episodes) than the eval pod's canonical 40-episode
`physicalai-val-0c5f7dac3b11`; comparing across builds would confound model with data. Both arms were therefore
re-run fresh, and pod3's numbers used only as a consistency check (§6).

| | value |
|---|---|
| Windows | episodes 0–39, `stride 40`, `t0 0` → **200 windows / 40 episodes** (5 per episode, as the pod3 recipe predicts) |
| Pass | **A only** for every accuracy number — Pass A is *not* given the numeric future ego track, so its ROUTE is independent evidence. Pass B ROUTE is downstream of our kinematics by construction and is refused entry to the statistics (the scorer raises on `--pass B`) |
| Prompt | **`vlmroute-2026-07-20-a`**, byte-identical for both arms, imported from `vlm_route_labels.py` |
| Decoding | greedy (`do_sample=False`), `max_new_tokens=1000`, 8 images @448², HF cache **offline** for both |
| Ground truth | kinematic **v2.1** `refb_labels.route_from_future_v21`, evaluated on the pod and stamped into every record |
| GT distribution | 171 valid (**89 straight / 52 left / 30 right**) + **29 unknown** (8 `no_arc`, 21 `gray_zone`) |

> ⚠️ **The prompt is a real confound and we did not remove it.** `vlmroute-2026-07-20-a` was authored *for
> Reason2*. The brief's rule was that a Reason1 parse failure would be a **result** (formatting failure vs
> reasoning failure) and not something to quietly fix. In the event **Reason1 parsed 200/200** — so no adapted
> third arm was needed, and the headline arms are the identical-prompt ones. The confound did bite, but in the
> *opposite* direction from the one we guarded against: §5 shows Reason2 **copying the prompt's own example
> sentence** into its evidence field on 16 % of its turn calls, which is prompt contamination of the *favoured*
> model.

**One loader fix, no prompt change.** `vlm_route_labels.Cosmos` loads with `device_map=` (needs `accelerate`,
absent on `tanitad-eval`) and hard-prefers `Qwen3VLForConditionalGeneration`. Reason1-7B is `qwen2_5_vl` and
Reason2-8B is `qwen3_vl`; forcing one class onto both is simply wrong. The harness resolves the class from each
checkpoint's own config via `AutoModelForImageTextToText` and loads CPU→cuda. Nothing the model is *shown* or
*asked* changed.

**Fleet etiquette — and where it failed.** Two `taniteval.efficiency` latency benchmarks were running on the
eval pod when this started; we waited for `ALLDONE` on both (the 17 GB Reason2 pull was also held until after,
so 16 GB of IO could not perturb the p95/p99 they were measuring). pod1/pod2/pod3 were never touched beyond
read-only pulls. **That was not enough:** the sibling agent's benchmark *campaign* continued past the `ALLDONE`
token with further scripts, and our runs overlapped them. **`ALLDONE` in a log marks one script finishing, not
the end of an agent's GPU campaign** — the lesson that produced `gpu_lock.sh`. Exact occupancy, for
reconciliation against the quarantined benchmark files (all 2026-07-20 UTC):

| block | start | end | GPU |
|---|---|---|---|
| Reason1 smoke (3 windows) | 21:17:12 | 21:17:33 | busy |
| Reason2 smoke (3 windows) | 21:18:35 | 21:18:57 | busy |
| **Reason1 Pass A** (200) | **21:19:17** | **21:35:41** | busy |
| **Reason2 Pass A** (200) | **21:35:41** | **21:49:52** | busy |
| *(idle)* | 21:49:52 | 21:55:47 | **free** |
| **Reason1 Pass B** (50) | **21:55:47** | **22:17:54** | busy |
| **Reason2 Pass B** (50) | **22:18:00** | **22:51:2x** | busy |
| *(released — no further GPU work)* | 22:51 | — | **free** |

The 17 GB Reason2 download ran ~21:08–21:09 and was **IO only, no GPU**. A queued ROUTE-order probe was
cancelled unrun at 22:12 to shorten the blocking window (§8.1).

## 2. The headline numbers

| | **Cosmos-Reason1-7B** | **Cosmos-Reason2-8B** |
|---|---|---|
| arch / weights resident | `qwen2_5_vl` · 15.49 GiB | `qwen3_vl` · 16.34 GiB |
| **turn-detection recall** (GT turn → says any turn) | **34.2 %** (28/82) | **76.8 %** (63/82) |
| **direction accuracy given detected** | **78.6 %** (22/28) | **57.1 %** (36/63) |
|   ⤷ episode-cluster bootstrap 95 % CI | **[0.645, 0.923]** ✅ excludes 0.5 | **[0.400, 0.745]** ❌ contains 0.5 |
|   ⤷ Fisher exact, direction ⟂ truth | **OR 12.44, p = 0.006** ✅ | **OR 1.32, p = 0.78** ❌ |
| 3-class accuracy over answered | 65.3 % (n = 170) | 69.0 % (n = 171) |
| 3-class accuracy **over all** GT-valid | **64.9 %** | **69.0 %** |
| straight recall | 100 % (89/89) | 92.1 % (82/89) |
| abstention `unknown` on GT-valid windows | 0 | **0** |
| `u_turn` emitted | 0 | 0 |
| **parse failure rate** | **0 %** | **0 %** |
| **ROUTE enum violation** | **1.0 %** (2 — `curve_left`, `curve_right`) | **0 %** |
| `road_geometry` enum violation | 0 % | 2.0 % |
| truncation at `max_new_tokens` | 0 | 0 |
| slot fill (`ROUTE`) | 100 % | 98.5 % |
| slot fill (`sees_junction_ahead`) | 92.5 % | 100 % |
| prompt tokens / window (mean) | 2068 | 1733 |
| generated tokens / window (mean, p95) | 116.1 · 150 | **91.5 · 100** |
| generation seconds (median) | 4.21 | **3.86** |
| wall clock s/window incl. I/O | 4.86 | **4.20** |
| peak VRAM | **16.02 GiB** | 16.94 GiB |
| model load | 5.5 s | 5.4 s |

**Confusion matrices** (rows = kinematic v2.1, columns = model answer; `n/a` = parse/enum failure):

| Reason1 | left | straight | right | n/a | | Reason2 | left | straight | right | unknown |
|---|---|---|---|---|---|---|---|---|---|---|
| **GT left** (52) | **14** | 35 | 3 | 0 | | **GT left** (52) | **28** | 14 | 10 | 0 |
| **GT straight** (89) | 0 | **89** | 0 | 0 | | **GT straight** (89) | 6 | **82** | 1 | 0 |
| **GT right** (30) | 3 | 18 | **8** | 1 | | **GT right** (30) | **17** | 5 | **8** | 0 |

Read the bottom-right cell of each: on true **right** turns, Reason2 answers **left 17 times and right 8 times**.

## 3. The paired statistics

Both arms saw the same windows, so every comparison is **paired**. Intervals are the **episode-cluster
bootstrap** (`taniteval/taniteval/ci.py`, 2000 draws, clustered on episode); the retired
`overlapping_holdout_se` is not used anywhere here, and no two independent intervals were combined in quadrature.
`delta = Reason1 − Reason2`.

| metric | n (windows / episodes) | Reason1 | Reason2 | delta | 95 % CI (paired ep-cluster boot) | separated? | McNemar exact |
|---|---|---|---|---|---|---|---|
| 3-class accuracy over all | 171 / 38 | 0.649 | 0.690 | −0.041 | **[−0.116, +0.030]** | **no** | b=8, c=15, **p = 0.210** |
| turn detected | 82 / 21 | 0.342 | 0.768 | −0.427 | **[−0.538, −0.321]** | **yes** | b=1, c=36, **p < 1e-6** |
| direction correct over *all* GT turns | 82 / 21 | 0.268 | 0.439 | −0.171 | **[−0.293, −0.060]** | **yes** | b=1, c=15, **p = 0.00052** |

Single-arm intervals, same estimator: Reason1 acc3 **0.649 [0.523, 0.767]**, Reason2 **0.690 [0.573, 0.804]**.

**Inter-arm agreement** (with each other, not with kinematics): Cohen's **κ = 0.430** over all 200 windows
(72.5 % raw agreement), **κ = 0.456** over the 171 GT-valid windows. The two models are only moderately
correlated — they are not two draws of the same labeler, and the disagreement is systematic, not noise.

*Note on the third row.* Direction correctness is scored over **all** GT turns rather than over each arm's own
detected subset, because a detection-conditioned denominator differs per arm and is therefore **not pairable**.
That row is the honest paired comparison; the *conditional* hit rates (78.6 % vs 57.1 %) are reported per-arm
with their own intervals in §2, never as a paired delta.

## 4. Why Reason2 "wins" the aggregate and still loses the argument

Reason2's higher 3-class accuracy (a tie statistically, but it is the higher point estimate) is **bought by
attempting more turns on a left-heavy corpus**, not by knowing which way the car goes.

| | Reason1 | Reason2 | kinematic truth |
|---|---|---|---|
| predicted-left share of its turn calls | **60.7 %** (17/28) | **72.9 %** (51/70) | 63.4 % (52/82) |
| recall on GT **left** | 26.9 % | 53.8 % | — |
| recall on GT **right** | 26.7 % | **26.7 %** | — |
| accuracy given detected, GT **left** | 82.4 % | 73.7 % | — |
| accuracy given detected, GT **right** | 72.7 % | **32.0 %** | — |

Reason1's error is **symmetric** — it misses left and right turns at the same rate (26.9 % vs 26.7 % recall) and
is roughly equally accurate on both when it commits. Reason2's error is **asymmetric**: it recovers twice as
many left turns as right turns and, on the right turns it does flag, it names the wrong direction two times out
of three. That is exactly the failure mode a training corpus must not absorb, because it is a *bias*, not noise.

**Worked example — `ep_00002`, three consecutive windows.** Kinematic v2.1 says `right` at t = 40, 80, 120
(net heading −8.1°, −13.8°, −15.6°; peak κ = 0.022 m⁻¹). Reason2 answers **`left` with
`route_confidence: 0.99` all three times**, with evidence *"the ego vehicle approaches a T-junction and turns
left, as seen in the future frames where the road ahead ends and the view shifts to face a cross street."*
Reason1 answers `straight` on all three — wrong, but not confidently wrong in the opposite direction.

## 5. Two defects the aggregate hides

**5.1 Reason2 copies the prompt's own example into its evidence field.** The Pass-A prompt illustrates
`route_evidence` with *"the road ahead ends at a T-junction and frames 4-5 show the view swinging to face a
cross street"*. Reason2 reproduces the fragment **"swinging to face a cross street" on 13/200 windows — 16 % of
its 80 turn calls**. Reason1: **0/200**. Reason2 emits 146 unique evidence strings out of 200; Reason1 emits 195.
The `route_evidence` field is supposed to be the audit trail that lets a human verify a label. On Reason2 it is
partly a template, so **it cannot be trusted as the verification hook** the strategy doc assumes it is.

**5.2 Reason2's confidence is a constant.** It emits `route_confidence: 0.99` on **195/200** windows (Reason1:
`0.9` on 181/200, with real spread 0.6–1.0). Neither is calibrated, but Reason2's is degenerate — including on
every one of the `ep_00002` wrong-direction calls above. **Confidence-thresholded auto-accept, which §7.4 of the
strategy assumes, is not available on Reason2.**

**5.3 (in Reason2's favour) its abstentions are correct.** Reason2 answered `unknown` exactly 3 times — all
three on `ep_00025`, stopped at a red light, where the kinematic labeler *also* returns unknown (`no_arc`). Zero
abstentions on GT-valid windows. That is precisely the behaviour the vocabulary's R3 rule asks for, and Reason1
never abstains at all (it produced 1 unparseable reply and 2 enum violations instead).

**5.4 Reason1's 2 enum violations are semantic, not syntactic.** Both were `curve_left` / `curve_right` —
tokens lifted from the *`road_geometry`* enum into `ROUTE`. Under our own definition (`straight` *"INCLUDING
following a bend or curve in that road"*) the intended answer was `straight`, and the kinematic label agrees in
both cases. So Reason1's only format failures are a model correctly perceiving a curve and reaching for the
wrong slot's vocabulary — a **prompt** defect, not a reasoning defect.

**5.5 Difficulty is confounded with clip position — read the strata with care.** Accuracy *rises* as the
available future *shrinks* (Reason1 49.3 % → 90.3 % from 4 future frames to 1; Reason2 60.3 % → 87.1 %). This is
a **base-rate artifact, not skill**: windows at t = 160 have both fewer future frames and a shorter kinematic
horizon, so 32 % of them are turns versus 55 % at t = 0. The one signal worth keeping is that Reason1's turn
recall collapses to **15 %** on the long-horizon (4-future-frame) windows where Reason2 holds **72.5 %** — the
long-horizon turn is the case Reason1 simply cannot see.

## 6. Consistency check against pod3's banked Reason2 run

pod3 holds 400 Pass-A Reason2 records on the 80-episode val build. Re-scored from the raw rows
(`/workspace/vlm_crossval.json`) with the same conventions used here:

| quantity | pod3 banked run (400 windows, 80 eps) | this run (200 windows, 40 eps) | verdict |
|---|---|---|---|
| direction accuracy given detected | **52.68 %** (59/112) | **57.1 %** (36/63) | ✅ replicates — both at chance |
| turn-detection recall | **80.6 %** (112/139) | **76.8 %** (63/82) | ✅ consistent |
| 3-class accuracy over all | 73.7 % (338 GT-valid) | 69.0 % | ✅ consistent |

**The chance-level direction result replicates across two different val builds. Treat it as CONFIRMED.**

⚠️ **One banked figure does not reproduce.** The circulating **"89.3 % turn recall"** for Reason2 could not be
recovered from `vlm_crossval.json` under any denominator tried (v2.1 turns 80.6 %; `tight_transient` only
81.0 %; ≥2/3/4 future frames 79.8/76.9/76.1 %; v2 labels 76.5 %; junction-flag variants 80.6–81.3 %). The
*direction* figure 52.7 % reproduces to four digits from the same rows with the same code path, so the scoring
convention is not the difference. **Per CLAUDE.md, the raw rows win: Reason2's turn recall on the pod3 build is
80.6 %, not 89.3 %.** Whatever produced 89.3 % should be found and corrected wherever it is quoted.

**Determinism cross-check:** on `ep_00000` t = 0/40/80 this run reproduced pod3's `route_evidence` strings
*verbatim*, including the wrong `left` at t = 80. Same model, same prompt, greedy decoding → same output on two
pods and two builds.

## 6b. Pass B — schema adherence over the FULL v3 vocabulary

The second half of the question is *which model mints the VLM-pending slots*. Pass B was run for both arms on a
**deterministic 50-window sub-sample** (`--window-stride 4` over the same manifest, so it stays episode-balanced
rather than truncating to a prefix), same prompt, same pod, `max_new_tokens=2200`. **Pass B is scored for schema
behaviour ONLY** — it is shown the numeric future ego track, so nothing here may be read as accuracy, and
`STRATEGIC.ROUTE` in this table is *not* a route measurement.

| | Reason1-7B | Reason2-8B |
|---|---|---|
| JSON parse rate | **88 %** | **74 %** |
| **truncated at 2200 tokens** | **0 %** | **26 %** ❌ |
| generated tokens / window (mean) | 668 | **1022** |
| generation seconds (median) | 23.6 | 23.4 |
| mean slot **violation** rate (of answered, 32 slots) | 3.4 % | **0.76 %** ✅ |
| mean **informative** rate (non-`unknown`, over all windows) | 69.4 % | 56.8 % |
| …renormalised by answered rate | 79.7 % | 76.7 % |

**Reason2's lower parse rate is entirely our token budget, not a defect.** It generates ~1.5× the output and
hits the 2200-token ceiling on 26 % of windows; Reason1 never does. Budget **≥3500 output tokens** for Reason2
on the full-vocabulary pass and this gap should close — *untested, and it must be tested before any bulk run.*

**On the slots it does complete, Reason2 is markedly more disciplined.** Reason1's violations concentrate in two
slots: `STRATEGIC.ODD` (**81 %** violations — it emits `construction_zone`, `yield_merge`, `[]`, `none`, i.e. it
borrows from the free ODD list and neighbouring enums) and `TACTICAL.RULECTX` (**14 %** — dotted compound tokens
like `justified_deviation.construction_zone`). Reason2's worst slots are `SCENARIO.road_geometry` and
`STRATEGIC.ODD` at ~11 %. This is the same shape as the pilot's finding: **word-token slots are near-perfect;
the failures are where a slot invites composition.**

**One unresolved difference.** Reason2 answers `none`/`unknown` far more often on the lead-vehicle and
traffic-light observation slots — `OBS.distance_bucket` 16 % vs 56 % informative, `OBS.lead_lane` 16 % vs 40 %,
`OBS.relative_motion` 16 % vs 46 %, `TACTICAL.TACPOINT` 0 % vs 32 %, `OBS.light_state` 0 % vs 12 %. Whether that
is Reason2 being appropriately conservative or Reason1 hallucinating cannot be settled without ground truth —
**but the pilot independently flagged Reason1 as over-detecting lead vehicles (88 % `present`, "likely
over-detection")**, which makes conservatism the more likely reading. **UNRESOLVED — needs the human-verified
gold slice §5 of the strategy doc already calls for.**

**Both models are 0 % informative on `TACTICAL.INTERACT` and ~0 % on `TACTICAL.SIGNAL`.** These are not model
failures — a forward camera cannot see a blinker, and the vocabulary already assigns `SIGNAL` to `human/can`.
Stop asking the VLM for them.

## 7. Verdict

**For the `ROUTE` slot: neither. Mint it kinematically.**
`refb_labels.route_from_future_v21` already emits `left/straight/right/unknown` with an explicit validity flag
and a graded target, at zero GPU cost, on every window we own. Nothing in this comparison suggests a VLM adds
information to it — one model is at chance on direction and the other declines to answer 66 % of turns.
This is the same conclusion the 48-clip pilot reached for `VTARGET` and `LONMODE`, and it is now a rule worth
writing down: **if the ego pose track determines the label, the pose track owns the label.**

**For the corpus-labeling engine overall: Cosmos-Reason2-8B**, on format discipline and cost, *not* on accuracy —
Pass A: 0 parse failures, 0 ROUTE enum violations, correct abstention on genuinely ambiguous windows, 21 % fewer
output tokens, ~9 % faster per window, at +0.9 GiB VRAM. Pass B: **4.5× lower slot-violation rate** across the 32
categorical vocabulary slots (0.76 % vs 3.4 %). This also matches NVIDIA's own migration guidance and the
strategy doc's §4 choice. Two conditions attach:
1. **Raise the Pass-B output budget to ≥3500 tokens** before any bulk run — at 2200 Reason2 truncates on 26 % of
   windows and that, not the model, is what drags its parse rate to 74 %.
2. Deploy it **only on slots the kinematics cannot compute** — scene tags, sign reads, lead state, `VSOURCE`
   justification, `RISK`, `LIGHTSTATE`, `TACPOINT`, CoC narrative — **without** relying on `route_confidence` as
   a filter or `route_evidence` as an audit trail (§5.1, §5.2), and **without** asking it for `INTERACT` or
   `SIGNAL` at all (§6b).

**Confidence in each finding**

| finding | status |
|---|---|
| Reason2's direction call is at chance | **CONFIRMED** — replicates across two val builds; CI contains 0.5; Fisher OR 1.32, p = 0.78 |
| Reason2 detects turns far better than Reason1 | **CONFIRMED** — paired CI [−0.538, −0.321], McNemar p < 1e-6 |
| Reason1's direction call is informative when it commits | **CONFIRMED but low-n** — 28 windows, 15 episodes; CI [0.645, 0.923], Fisher p = 0.006 |
| Overall 3-class accuracy is a tie | **CONFIRMED** (as a *failure to separate*, not as equality) — CI [−0.116, +0.030] |
| Reason2 has a systematic **left** bias | **CONFIRMED on this corpus** — 32.0 % accuracy on detected right turns vs 73.7 % on left |
| The left bias is caused by enum ordering in the prompt | **SUGGESTIVE / UNTESTED** — see §8 |
| Reason2 templates its `route_evidence` | **CONFIRMED** — 13/200 verbatim prompt-example reuse, 0/200 for Reason1 |
| Throughput ranking (Reason2 faster) | **CONFIRMED but note contention** — a co-tenant GPU job overlapped Reason1's first ~30 windows; the *median* generation time (4.21 vs 3.86 s) is the contention-robust comparison and it agrees with the token counts |

## 8. What would have to change — the cheap decisive next test

1. **Swap the enum order (≈15 min, 200 windows). BUILT, QUEUED, THEN CANCELLED — run it first.**
   `ROUTE_ENUM` lists `left` before `right` and the definition line reads `left / right = …`. If Reason2's bias
   follows the ordering when swapped, it is a **prompt artifact and fixable**; if the bias stays left, it is a
   model property and ROUTE is closed for Reason2 permanently. This single run converts the last SUGGESTIVE row
   above into a decision. The arm is **implemented and tested** —
   `vlm_model_compare.py --route-order right_first`, two auditable substitutions that stamp a distinct
   `prompt_version` (`…-a-rswap`) so a swapped record can never be pooled with the headline arms. It was queued
   on `tanitad-eval` and **cancelled unrun** to release the GPU to a blocked sibling agent; no swapped record
   exists. Cost to close: ~15 min on any free GPU.
2. **Two-stage route prompt.** Reason1 proves the direction signal is extractable from these frames (OR 12.4).
   Ask *"does the vehicle leave the road it is on? yes/no"* first, then re-present the frames for direction only.
   Reason2's detection (76.8 %) plus Reason1-grade direction would be a usable labeler; the current single-shot
   prompt gets neither.
3. **Give the model the geometry it is being asked to read.** All the frames are forward camera at 448²; a
   metric BEV inset (the standing TanitEval viz convention) would make "which way did the road go" a *reading*
   task rather than an *inference* task. Untested here.
4. **Do not re-open ROUTE without one of the above.** A third model (Qwen3-VL-8B, Cosmos-Reason2-32B) on the
   same broken prompt would measure the prompt, not the model.

## 9. Reproduce

Everything needed is in the repo; the pod is not required to re-derive a single number.

```bash
# score the committed records (no GPU, no pod, numpy only)
python stack/scripts/vlm_compare_score.py \
  --out "TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-20-vlm-reason1-vs-reason2" \
  --arms reason1,reason2 --json /tmp/results.json

# regenerate the raw records on a free GPU pod
scp stack/scripts/{refb_labels,vlm_route_labels,vlm_model_compare}.py <pod>:/root/vlm_compare/
ssh <pod> 'cd /root/vlm_compare && PYTHONPATH=/root/vlm_compare HF_HUB_OFFLINE=1 \
  python3 vlm_model_compare.py --out out --val /root/valdata/physicalai-val-0c5f7dac3b11 \
  --episodes 0-39 --stride 40 --passes A --model nvidia/Cosmos-Reason1-7B --tag reason1'
```

The `windows.json` manifest is built **once** and reused by every arm — that is what makes the comparison
paired. Deleting it and rebuilding reproduces it exactly (it is a pure function of the val build, the episode
range, the stride and `t0`).
