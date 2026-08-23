<title>Production semantic labeling with Cosmos-Reason2-8B — prompt v2, the enum-order probe, and the scenario corpus</title>

# Production semantic labeling with Cosmos-Reason2-8B

**Author:** Data Engineering (agent run, PI request). **Date:** 2026-07-21.
**Status:** Phases 1 and 2 **complete and measured**. Phase 3 **running unattended on the pod** — val at 33/400 windows when this was written, train sample queued behind it. Everything measured is staged in the repo; `bash stack/scripts/pod_ops/pull_vlm_records.sh` rescues the rest at any time.
**Pod:** `tanitad-pod3` (A40 46 GB, torch 2.8.0+cu128, transformers 5.14.1), GPU lock held as `vlm-production` for the whole campaign.
**Times are Europe/Berlin (UTC+2); the pod clock is UTC.**
**Decides:** whether `road_geometry` / `scenario_tag` can carry the intersection / roundabout / merge capability metrics
[TanitEval v2](../../Benchmarks%20&%20Eval/TANITEVAL_V2_METRIC_SUITE.md) §3.5 says have no ground truth today.
**Anchors:** [head-to-head](2026-07-20-cosmos-reason1-vs-reason2-headtohead.md) · [TANITDATASET_V1_STRATEGY](../TANITDATASET_V1_STRATEGY.md) ·
[V3 goal vocabulary](../../Architecture%20&%20Inference/V3_GOAL_VOCABULARY_V1.md)
**Artifacts:** `TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-21-vlm-production-semantic/`
· harness `stack/scripts/vlm_semantic_labels.py` · scorer `stack/scripts/vlm_semantic_score.py` · tests `stack/tests/test_vlm_semantic.py`

---

## TL;DR

1. **The enum-order probe is answered, and the answer closes ROUTE for good.** Listing `right` before `left`
   (and swapping the definition gloss) did **not** move the left bias: left share of turn calls **74.5 % → 66.7 %**
   on a corpus that is **48.2 % left**, recall on true right turns **0.2069 in both arms — bit-identical**, Fisher
   **OR 0.882 → 0.735** (p 1.0 → 0.74, no information either way), direction-accuracy CI **[0.268, 0.702]** and
   **[0.250, 0.683]** — both contain chance. Inter-arm **κ = 0.852**. **The bias is a property of Cosmos-Reason2,
   not of our prompt's ordering.** The head-to-head's last SUGGESTIVE row is now CONFIRMED, in the direction that
   means ROUTE may never be minted from this model.
2. **But enum order is not inert, and that is a new finding.** Listing `right` first cost **8.9 points of
   turn-detection recall** (78.6 % → 69.6 %, paired episode-cluster CI **[+0.035, +0.153]**, separated). Order does
   not decide *which way* the model says, but it does decide *how often it commits at all*. Any categorical slot we
   mint from this model inherits that sensitivity, so it has to be measured — not assumed away — on the slots we
   actually keep.
3. **Two of the four prompt fixes worked; two failed, and both failures taught us more than the successes.**
   Removing the in-prompt example killed evidence contamination — **13.1 % of turn calls (8/61) → 0 of 27**, with
   unique-string rate 0.735 → 0.930. But **raising the token budget made truncation worse, not better**
   (32.5 % at 2200 → **61.5 % at 3500**, median Pass-B generation 24 s → 135 s) — the model spends whatever
   ceiling it is given — and **switching `route_confidence` from a float to a discrete band did not create
   calibration**, it relabelled the constant (`0.99` on **200/200** → `high` on **97/100**, `low` never once). §2.
4. **The most consequential finding of the run: Pass B echoes our own kinematics, in a new slot.** Pass B's
   "fabricated" event times are our *own* `future_track` onset copied to the decimal (11.9 → 11.9, 15.4 → 15.4,
   11.0 → 11.0). Pass A, which never sees that block, is **100 % compliant**. The doctrine that quarantines Pass
   B's ROUTE therefore extends to event times *and to `road_geometry`* — so the scenario strata are now taken from
   **Pass A**, the independent instrument, with Pass B's kept beside them marked contaminated. §2.
5. **Truncation costs one free-text block, and the truncation RATE is a misleading headline.** The reply is
   emitted correctly in schema order and only the last free-text field runs away, so a salvage parser recovered
   **7/7** strict-parse failures in Phase 1 and takes the usable rate to **100 %**. On the production run, at a
   **75 %** truncation rate, `SCENARIO` / `STRATEGIC` / `TACTICAL` / `OBSERVATIONS` are all recovered on **14/14**
   windows — **only the `COC` narrative is lost** (5/14). Judge this by block survival, never by the rate. §2.
6. **The frame ablation answers "more past and future" with a split verdict.** More **future** pays: the reference
   plan was silently delivering only **2.8 of its 5** future frames (our clips are ~20 s, so its 15 s/20 s offsets
   mostly do not exist), and an early-weighted schedule delivers **5.0** and lifts turn-detection recall
   **0.750 → 0.857** for +23 % prompt tokens and **+4 %** wall clock. More **past** pays nothing: 3 → 5 history
   frames costs +17 % tokens for *identical* recall. And resolution is not a free saving — dropping to the stored
   256 px costs **21 points** of turn detection for a 3 % time gain, even though 448 px is pure upscaling. §3.
7. **A blocking data fact, found while setting up, that changes what this run can deliver.** The val epcache on the
   free GPU is **not** the canonical TanitEval val build, and it is not a superset of it: exact pose-fingerprint
   matching says **only 8 of the 40 canonical val episodes are present on pod3**. `split_clips` permutes with
   `torch.randperm(len(clips))`, so a 200-clip discovery and a 500-clip discovery draw different val sets. The
   episode map is shipped; the other 32 episodes need the canonical epcache on a free GPU. **Escalated — §7.**
8. **Can we build the intersection / roundabout metrics on these labels? NOT YET — and the blocker is sampling,
   not label quality.** The labels are schema-clean (Pass-A parse failure and enum violations **0 %**, event times
   **0 % fabricated**, Pass-B usable **100 %** after salvage, informative rate up 0.554 → **0.819**), and the
   geometry call reproduces at **κ = 0.776** across frame plans. But the eventful strata are too thin to score:
   over the full 400-window val run `junction` projects to **20–53** (the two samples disagree 2.7×, so it must be
   re-counted, not projected), while **`roundabout`, `merge` and `fork` are absent, not merely thin** — against
   TanitEval v2's own **n ≥ 30** floor. Use the labels to **stratify**; do not publish a capability number. §6
   lists the five things that would change the answer and ships the human audit sheet for the one that matters
   most.

---

## 0. What ran, and when

| block | start (Berlin) | end | windows | prompt |
|---|---|---|---|---|
| enum-order probe, arm `as_written` | 07:36 | 07:52 | 200 | v1 |
| enum-order probe, arm `right_first` | 07:55 | 08:08 | 200 | v1 `-rswap` |
| `p1_v1` — the v1 baseline, Pass A+B | 08:08 | 08:41 | 40 | v1 @ 2200 |
| `p1_v2` — v2a, Pass A+B | 08:41 | **09:04 (stopped at n=13)** | 13 | v2a @ 3500 |
| `ab_base` — frame ablation reference | 09:04 | 09:14 | 100 | v2b, Pass A |
| `ab_dense_early` · `ab_wide_cheap` · `ab_dense_hist` | 09:14 | 09:51 | 3 × 100 | v2b, Pass A |
| `ab_base_randenum` — enum-order on `road_geometry` | 09:51 | 10:02 | 100 | v2b, randomized |
| `p1_v2b` — Pass-B validation | 10:02 | **10:08 (stopped at n=3)** | 3 | v2b @ 3500 |
| **Production: val build, 80 episodes** | **10:08** | in flight | 400 | v2b, `dense_early`, @1200 |
| Production: stratified train sample | queued | — | ≤600 | v2b, `dense_early`, @1200 |

`p1_v2` was **stopped by explicit PID at n=13**: at 103 s/window (it truncates, and a truncated window generates
to the ceiling) the remaining 27 windows would have cost ~46 min of GPU to refine a negative result that was
already unambiguous, and that budget was worth more to production. Its records were checked for a partial write
after the kill (13/13 intact).

### Two operational traps this run hit, both worth writing down

**1. Killing a supervised job ADVANCES its supervisor.** The val run was stopped by explicit PID to pick up a
reordered manifest. That is the documented-safe way to stop a job — but the job was the body of a `for` loop in a
supervisor script, so the loop saw it "finish", logged `val_full_DONE`, and immediately started the *next* item
(the train sample). For ~30 s two labeling jobs raced for one GPU, and `prod.log` now contains a **false**
`val_full_DONE` at 10:27. Caught before either had loaded weights; the train job was killed by explicit PID and
re-queued behind val. **Kill the SUPERVISOR first, then the job — or expect the supervisor to move on.** (This is
adjacent to the standing `pkill -f` trap but is not the same one: every kill here was by explicit PID.)

**2. A manifest ordering choice can silently defeat its own purpose.** The val manifest was made *t-major* so a
partial run would cover all 80 episodes rather than a prefix of episodes. It did — but the first block was
`t = 0`, and `t = 0` is precisely the offset with **no TanitEval counterpart** (an eval window is keyed by its
start and needs 8 frames of history), as well as the least representative window (no history frames, maximum
future). Re-ordered to `t = 40, 80, 120, 160, 0`; the 24 records already produced at `t = 0` are kept and the run
resumed. **"Cover everything early" and "cover something useful early" are different orderings.**

Two CPU-only jobs also ran alongside (the stratified-window pose walk). **The first one cost 3× throughput on the
GPU job** — it took 7 cores and starved the VLM's image-preprocessing and tokenisation, dropping the labeler from
3.9 s/window to 13 s/window at 20 % GPU utilisation. Re-run under `taskset -c 0,1` + `OMP_NUM_THREADS=1` it was
invisible, and *faster* (72 s vs 320 s, warm page cache). **A CPU-heavy job on a GPU pod is not free even when it
never touches the GPU** — `nice` alone does not fix it, because the contention is for cores the GPU job needs to
keep the device fed.

### Where the wall clock actually goes (measured, and it is not where it looked)

Per-window wall time was profiled against `gen_seconds` on 13 consecutive Pass-A+B windows:
**median wall 32.8 s, median generation 31.9 s — non-generation overhead is 0.8 s.** Episode `torch.load` is
0.48 s and the whole CPU-side image path (`to_pil` + `apply_chat_template`, 8 images) is **0.13 s**. So there is no
pipeline inefficiency to chase; every second is the model generating.

**What the profile did expose is that truncation is expensive as well as damaging.** The windows that hit the
2200-token ceiling took **87 s each against a 25 s median — 3.5×** — because a truncated reply is one that
generated to the cap. That reframes defect 1: raising the budget does not only rescue lost records, it also
*raises the cost of the worst windows*, so the real fix has to be a **terser prompt**, with the larger budget as
insurance. This is why v2 deletes two asks outright and adds an explicit brevity rule rather than simply buying
more tokens. §2 measures whether that worked.

Image-token cost by frame plan, measured on the real processor (relevant to §3): 8 images at 448 px =
**1593 prompt tokens**; the same 8 at the stored **256 px = 537**; 11 at 256 px = 735; 11 at 448 px = 2187. The
frames are stored at 256², so 448 is pure upscaling — it buys **3× the image tokens for zero extra information**.

---

## 1. The enum-order probe — BUILT, QUEUED, CANCELLED, and now RUN

The head-to-head left exactly one finding SUGGESTIVE: *"the left bias is caused by enum ordering in the prompt."*
The arm existed (`vlm_model_compare.py --route-order right_first`, two auditable substitutions stamping a distinct
`prompt_version`), was queued on `tanitad-eval`, and was cancelled unrun to release the GPU. This run executes it.

**Design.** 200 windows over **val episodes 40–79 of a build the head-to-head never touched** — so the probe is not
fitted to the 200-window set that produced the finding. Same model, same greedy decoding, same 8 frames, back to
back on the same GPU, shared window manifest (paired by construction).

| | `as_written` (`left` first) | `right_first` (`right` first) |
|---|---|---|
| prompt version | `vlmroute-2026-07-20-a` | `vlmroute-2026-07-20-a-rswap` |
| GT valid / GT turns | 170 / 56 | 170 / 56 |
| **left share of its turn calls** | **0.7447** (35 L / 12 R) | **0.6667** (26 L / 13 R) |
| *(GT left share of turns)* | *0.4821* | *0.4821* |
| **recall on GT right turns** | **0.2069** | **0.2069** |
| recall on GT left turns | 0.5556 | 0.4444 |
| accuracy given detected, GT left | 0.7143 | 0.6316 |
| accuracy given detected, GT right | 0.2609 | 0.3000 |
| direction accuracy given detected | 0.4773 (21/44) | 0.4615 (18/39) |
| ⤷ episode-cluster 95 % CI | **[0.268, 0.702]** ❌ contains 0.5 | **[0.250, 0.683]** ❌ contains 0.5 |
| Fisher OR / p (direction ⟂ truth) | **0.882 / p = 1.00** | **0.735 / p = 0.74** |
| turn-detection recall | **0.7857** | **0.6964** |
| 3-class accuracy over all | 0.7765 | 0.7765 |
| parse failures / ROUTE enum violations | 0 / 0 | 0 / 0 |
| generated tokens (mean) · seconds (median) | 91.5 · 3.77 | 92.2 · 3.75 |

**Paired tests** (episode-cluster bootstrap, `delta = as_written − right_first`):

| metric | n (win/eps) | as_written | right_first | Δ | 95 % CI | separated | McNemar |
|---|---|---|---|---|---|---|---|
| 3-class accuracy over all | 170 / 37 | 0.7765 | 0.7765 | 0.0000 | [−0.042, +0.039] | no | p = 1.00 |
| **turn detected** | 56 / 14 | 0.7857 | 0.6964 | **+0.0893** | **[+0.035, +0.153]** | **yes** | p = 0.0625 |
| direction correct over all GT turns | 56 / 14 | 0.3750 | 0.3214 | +0.0536 | [−0.036, +0.137] | no | p = 0.45 |

Inter-arm **Cohen's κ = 0.852** (93.5 % raw agreement over all 200 windows).

### The verdict, in three parts

**(a) The left bias is the model's, not ours.** If the bias were positional, swapping the list would have swapped
the bias. It did not: the model still called left on two thirds of its turns on a corpus that is under half left,
and — the cleanest single number here — **its recall on true right turns is 0.2069 in both arms, the identical
6 of 29 windows**. Swapping the enum did not let it see a single extra right turn. It changed which *left* calls
it was willing to make, and those went to `straight`, not to `right`: `n_pred_left` fell 35 → 26 while
`n_pred_right` rose only 12 → 13.

**(b) ROUTE is now closed on model evidence, not on prompt suspicion.** Both arms' direction-accuracy intervals
contain chance and both Fisher odds ratios sit at ~1. The head-to-head's recommendation (*"a third model on the
same broken prompt would measure the prompt, not the model"*) is discharged: the prompt was varied and the model
did not move. **ROUTE stays kinematic. `route_from_future_v21` remains the only admissible route ground truth.**

### A third independent measurement retires the circulating "89.3 %"

The head-to-head could not recover Reason2's widely-quoted **89.3 %** turn-detection recall from the raw rows
under any denominator it tried (it measured **80.6 %** on pod3's banked run and **76.8 %** on the eval pod), and
asked that the figure be found and corrected wherever it is quoted. **This probe is a third independent
measurement on a fourth window set: 78.6 %** (`as_written`, 200 held-out windows, 40 episodes).

| measurement | corpus | turn-detection recall |
|---|---|---|
| head-to-head, eval pod | `physicalai-val-0c5f7dac3b11`, 200 win | **76.8 %** |
| head-to-head, pod3 banked re-score | `physicalai-val-f1b378f295ae`, 400 win | **80.6 %** |
| **this probe** | `physicalai-val-f1b378f295ae` eps 40–79, 200 win | **78.6 %** |
| the circulating figure | — | 89.3 % ❌ not reproducible |

Three measurements on three window sets cluster at **77–81 %**; none approaches 89.3 %. **Two live documents still
quote it** — `TanitAD Research Hub/Benchmarks & Eval/TANITEVAL_V2_METRIC_SUITE.md` §2.3 (*"A VLM may be used to
DETECT that a route event occurred (89.3 % agreement)"*) and
`TanitAD Research Hub/Architecture & Inference/V35_DESIGN.md` C5 (*"IS-A-TURN 89.3 % is a good event detector"*).
**Both should be corrected to ~78 % (range 77–81 % across three measurements).** Flagged rather than edited: both
are other agents' live documents. The *qualitative* claim they rest on — that the VLM is a competent event
detector and an incompetent direction reader — survives unchanged.

**(c) The unexpected half: order changes the model's willingness to commit.** An 8.9-point drop in turn detection
with a CI that excludes zero is not noise. The mechanism is visible in the confusion: with `right` listed first
the model retreats to `straight`. **This is the finding that generalises beyond ROUTE** — every categorical slot we
*do* mint is selected from a list we wrote in some order, and this says the order moves the abstention rate. It
does not say the order moves the *content*.

### (d) So we measured it on the slot we actually keep — and then chose not to randomize

ROUTE's answer does not transfer to `road_geometry` for free, so a fifth arm (`ab_base_randenum`) re-ran the
100-window ablation manifest with **every enum permuted per window** (deterministic seed, recorded on the record;
`unknown`/`none` pinned last so abstention behaviour is not itself confounded).

| | `as_written` | `randomized` |
|---|---|---|
| **`road_geometry` agreement with `as_written`** | — | **κ = 0.804** (91 % raw) |
| `road_geometry` answered (informative) | **100/100** | 97/100 |
| Pass-A parse failure | **0 %** | 3 % |
| turn-detection recall | 0.750 | 0.714 |
| 3-class accuracy over all | 0.7816 | 0.7586 (McNemar p = 0.625, not separated) |
| left share of turn calls | 0.667 | 0.700 |

**`road_geometry` is largely insensitive to enum order (κ = 0.804), the left bias persists under randomization
(0.700), and randomizing costs 3 % of answers and 3 % of parses for no measured gain.** The brief's conditional —
*"if order flips the bias, every categorical enum is suspect and you must randomize"* — **is not met**, so
**production runs `as_written`**. The randomized path stays implemented and tested, because the *next* model may
behave differently and the arm now costs 10 minutes to re-run.

---

## 2. The four prompt defects — before and after

**Two of the four fixes worked. Two did not, and one of the failures was instructive enough to change the
design.** All arms below run on the same held-out window manifest (val eps 40–79).

**First: every defect replicated on this independent slice**, which is what makes the "after" numbers mean
something. On the 200-window `as_written` probe arm (prompt v1): `route_confidence` = a constant on **200/200**
(values seen: 0.98/0.99/1.00), evidence prompt-copy on **13.1 %** of turn calls (8/61), unique evidence strings
**73.5 %** — against the head-to-head's 195/200, 16 %, and 146/200. On the 40-window v1 Pass-A+B arm: Pass-B
truncation **32.5 %**, strict JSON parse **67.5 %**, against 26 % and 74 %.

| | **v1** (`vlmroute-2026-07-20-a`) | **v2a** (`vlmsem-…-a`, 3500 tok) | **v2b** (`vlmsem-…-b`, `ab_base`) | verdict |
|---|---|---|---|---|
| n | 40 (A+B) · **200 (A only)** | **13** ⚠️ stopped early | **100** (A only) | |
| **① Pass-B truncation** | 0.325 | **0.615** | *(§3 validation arm)* | ❌ **worse** |
| ① strict JSON parse | 0.675 | 0.385 | — | ❌ worse |
| ① usable incl. salvage | 1.00 | **1.00** | — | ✅ (new mechanism) |
| ① Pass-B tokens (mean) | 1141 | 2403 | — | ❌ |
| ① Pass-B generation (median s) | 24.4 | **135.0** | — | ❌ 5.5× |
| **② confidence modal share** | **1.00** (`0.99` on 40/40; 200/200 on the big arm) | 1.00 (`high` 13/13) | **0.97** (`high` 97, `medium` 3) | ❌ **not fixed** |
| ② distinct answers given | 1 | 1 | 2 | ❌ still degenerate |
| **③ evidence copy, turn calls** | **0.131** (8/61) · 0.231 (n=13) | 0.000 (n=2 turns) | **0.000 (0/27 turns)** | ✅ **fixed** |
| ③ evidence unique rate | 0.735 (200-win) · 0.875 | 0.923 | **0.930** | ✅ |
| **④ enum order** | — | — | — | ✅ answered in §1 |
| Pass-A `route_event_time_s` in offered set | 0.975 | 1.000 | **1.000** | ✅ |
| Pass-A `geometry_event_time_s` in offered set | *(not asked)* | 1.000 | **1.000** | ✅ |
| Pass-A parse failure / enum violation | 0 / 0 | 0 / 0 | **0 / 0** | ✅ |
| Pass-B mean slot violation | 0.004 | 0.013 | — | — (both tiny) |

**Confirmation on the plan production actually uses.** The table's v2b column is `ab_base` (the same frames as v1,
so the comparison is clean). Re-measured on **`dense_early`**, the plan §3 selects, over its own 100 windows:
evidence prompt-copy **0.0 % on 30 turn calls**, unique-string rate **0.959**, ROUTE enum violations **0 %**,
Pass-A parse failure 2 %, `route_event_time_s` and `geometry_event_time_s` **fabricated 0.0 %** (98 % in the
offered set, 2 % null — the two parse failures). Confidence: `high` **96**, `medium` 1, `low` **1**, missing 2 —
so `low` does exist, on 1 window in 100, which is not a usable signal but is worth recording as the one case where
the band was not simply a constant.

### ① Truncation — raising the budget made it worse, and that is the useful result

**v2a truncated on 61.5 % of windows at 3500 tokens where v1 truncated on 32.5 % at 2200.** The model spends
whatever ceiling it is given. Median Pass-B generation went from 24.4 s to **135 s**, which is why this arm was
stopped by explicit PID at n=13 — the finding was already unambiguous and the remaining 27 windows would have
cost ~46 min of GPU to refine a fix that was already being replaced.

**The raw replies say exactly what happens**, and it is not what "truncation" suggests: the object is emitted
**correctly, in schema order**, and then one free-text field near the end turns into an essay
(*"…the lack of any significant events or interactions with other vehicles or road features suggests that the
driving task is relatively simple…"*). So:

**Fix A — the salvage parser (`vlm_semantic_labels.salvage_json`).** Because the failure is a *tail* failure, a
truncated record still contains an intact `SCENARIO` block — the exact block the scenario metrics are blocked on.
Closing the open braces at the last safe boundary recovered **7 of 7** strict-parse failures on the real v2a
records, **all 7 with a usable `SCENARIO.road_geometry`**, and both `SCENARIO` and `STRATEGIC` every time. Applied
retroactively it takes both arms to **`json_usable_rate` = 1.00** (v1: 13 rows recovered, v2a: 8). The strict
parse stays in `parsed`; the salvage lands in `parsed_salvaged` with `parse_mode`, and every rate that uses it
reports `n_rows_from_salvage` so a reader can subtract them. What truncation still costs is the **tail** blocks —
`OBSERVATIONS` (lead state, sign reads) and `COC` — never the scenario strata.

**Fix C — and this is the one that settles the budget question exactly.** Pool every Pass-B reply across all three
arms that **actually completed** (`truncated = False` and a strict parse) and look at how many tokens they needed:

| completed Pass-B replies (n = 34, v1 + v2a + v2b) | tokens |
|---|---|
| min | 513 |
| median | 618 |
| p90 | 723 |
| p95 | 967 |
| **max** | **1146** |

| candidate cap | completable replies it would have truncated |
|---|---|
| 1000 | 1 / 34 (2.9 %) |
| **1200** | **0 / 34 (0.0 %)** |
| 1500 · 2000 · 2200 · 3500 | 0 / 34 |

**The distribution is bimodal: a reply either finishes under ~1150 tokens or it never terminates at all.** Every
token of budget above ~1200 is spent exclusively on replies that were going to run away regardless. So the correct
production budget is **1200 — lower than v1's 2200 and a third of v2a's 3500** — and the original recommendation
("budget ≥3500 before any bulk run", head-to-head §7.1) is **refuted**: it was the exact wrong direction. At 1200 a
runaway window costs ~47 s instead of 136 s (**2.9× faster**) and loses nothing, because `salvage_json` recovers
its `SCENARIO` block anyway.

**Production therefore runs `--max-new-b 1200`.** The `p1_v2b` validation arm was stopped by explicit PID at n=3
once this table existed: measuring v2b's truncation rate at n=20 is strictly worse than measuring it at n=400 on
the production run itself, which is the run we care about and which is now the validation.

⚠️ **And that validation immediately caught a slip in the reasoning above — recorded because it is the kind of
mistake this project keeps paying for.** The 34-completion table was measured entirely on the **`base`** frame
plan, but production runs **`dense_early`**. More future frames means more scene to describe, and the truncation
rate on the first production windows is **75 %**, not the ~30 % the phase-1 arms showed. *The frame plan and the
token budget are coupled, and the plan was chosen on Pass-A evidence while the budget was measured on a different
plan.*

**The cap itself is still right, and the block-level survival is why.** Under `dense_early` the replies that
complete still do so in **572–771 tokens** — the distribution is bimodal exactly as before, so 1200 is still
generous for anything that terminates. And "75 % truncated" turns out to name a much smaller loss than it sounds:

| Pass-B block, first 14 production windows | recovered |
|---|---|
| `SCENARIO` (road_geometry, scenario_tag, event times, scene axes) | **14/14** |
| `STRATEGIC` | **14/14** |
| `TACTICAL` | **14/14** |
| `OBSERVATIONS` (lead vehicle, sign reads, lane info, traffic light) | **14/14** |
| `COC` (the free-text narrative) | 5/14 |
| **usable rate (strict + salvage)** | **14/14 = 100 %** |

**The runaway happens inside the last block, so truncation costs the CoC narrative and nothing else.** Report
block-level survival, not the truncation rate — the rate is alarming and nearly meaningless on its own. (`COC`
is the one field the brief lists that this run under-delivers; it is a "WHY" sidecar for failure forensics, not
an input to any metric.)

**Fix B — v2b cuts the free text instead of buying more of it.** `route_evidence` deleted from Pass B (Pass B's
ROUTE is inadmissible anyway, so its justification was pure cost), `odd_flags` deleted (a free *list*, when
`STRATEGIC.ODD` is the enum'd twin), every remaining free-text value capped at 20 words in the shared rules
block, `critical_agents`/`sign_reads` capped at 2 entries, and an explicit *"STOP after the closing brace."*
⚠️ Because the cap lives in the **shared** rules block, a Pass-A prompt is **not** byte-identical between -a and
-b — only its task block is. Records must not be pooled across versions; the per-arm `prompt_A.txt` written into
every run directory is the authoritative record of what was asked.

### ② Confidence — the discrete band did not fix it either

v1 emitted `route_confidence: 0.99` on **200/200** (values seen: 0.98, 0.99, 1.00). v2 asks for a discrete band
with an explicit instruction naming when `low` is required, and on **100 windows** it emitted `high` **97 times**
and `medium` 3 times — **`low` never once**. Modal share 0.97; the field is degenerate under both encodings.
**Changing the response format does not create calibration; it relabels the same constant.**

The band is not entirely inert — 3-class route accuracy is 0.786 on `high` (n=84) against 0.667 on `medium`
(n=3) — but n=3 makes that unquotable, and a field that answers `high` 97 % of the time cannot threshold
anything at any n.

**Recommendation: treat confidence as absent.** It must not gate auto-accept (as `TANITDATASET_V1_STRATEGY` §7.4
assumes) and no consumer should filter on it. It is retained in the schema only because it costs ~2 tokens and a
later model may fill it honestly; every reader must mark it non-informative. If the production run reproduces a
modal share ≥ 0.90, **delete the field** rather than carrying a constant that looks like evidence.

### ③ Evidence contamination — fixed

Removing the in-prompt example sentence took verbatim prompt reuse from **13.1 % of turn calls** (8/61, the
200-window v1 arm) to **0.0 %** — and the v2b measurement is properly powered: **0 of 27 turn calls, 0 of 100
windows**, zero copied fragments of any length. Unique-string rate rose 0.735 → **0.930**. The replies also do
what the structural instruction asked and cite the frame they read (*"At t+2.0 s, the ego vehicle continues
straight on the highway…"*). **`route_evidence` is now usable as the human audit hook the strategy doc assumes
it is** — which matters, because with no ground truth for the scenario slots the audit hook is a large part of
what we have (§6).

### ⚠️ The finding that changed the design: Pass B echoes our own kinematics

v2a's event-time compliance looked like fabrication — `SCENARIO.scenario_event_time_s` was **outside the offered
frame-offset set on 69.2 %** of windows, `geometry_event_time_s` on 38.5 %. It is not fabrication. Lined up
against the numeric future-ego-track summary Pass B is given, the values are **our own kinematic onset, echoed to
the decimal**:

| window | our `future_track.route_event_time_s` | Pass B `geometry_event_time_s` | Pass B `scenario_event_time_s` | **Pass A** `geometry_event_time_s` |
|---|---|---|---|---|
| ep_00043 t0 | 11.9 | **11.9** | **11.9** | 15.0 |
| ep_00044 t0 | 15.4 | **15.4** | **15.4** | 2.0 |
| ep_00045 t0 | 11.0 | **11.0** | **11.0** | 15.0 |
| ep_00046 t0 | 11.3 | **11.3** | **11.3** | 15.0 |
| ep_00047 t0 | 14.4 | 15.0 | **14.4** | 15.0 |

**Pass A — which is never shown that block — is 100 % compliant** (13/13 for v2a, 39/40 for v1; every value is
one of the offsets actually offered).

This is the Pass A / Pass B doctrine reasserting itself in a new slot. The head-to-head established that Pass B's
ROUTE is downstream of our kinematics and inadmissible; **the same is now measured for event times, and by the
same argument it must extend to `road_geometry`** — "junction" is an easy read off *"the ego turns left 78 deg
beginning at t+11 s"*. A stratum that will be used to judge a model against our kinematic metrics must not itself
be derived from those kinematics, or the evaluation is circular.

**So the converter's primary `road_geometry` and `geometry_event_time_s` are taken from PASS A**, and Pass B's are
emitted beside them as `*_passB_CONTAMINATED` with `passB_is_downstream_of_our_kinematics: true`. No prompt change
was needed — the independent instrument already existed and was already being asked. `scenario_tag` has no Pass-A
twin yet and therefore inherits the contamination; **adding `scenario_tag` + its event time to Pass A is the
single highest-value next iteration of this prompt** (§9).

## 3. The frame ablation — Sayed asked for more past and future; here is what it actually bought

**Design.** One shared 100-window manifest (val eps 40–79, 40 episodes), one prompt (v2b), **Pass A only** — so the
plans differ in nothing but the frames. Pass A is the right arm for this: it is the *independent* pass, it is
cheap (≈5 s/window against Pass B's 24–135 s), and it carries the only slots that can be scored at all.

⚠️ **What can and cannot be scored.** There is no ground truth for `road_geometry` or `scenario_tag` — producing it
is the point of the run — so a frame plan cannot be shown to label the *scene* better. What can be measured:
**turn-detection recall** against the kinematic v2.1 label (an event detector, not a direction reader), format
discipline, **cross-plan label stability**, and cost. Anything else would be a claim without an instrument.

| | `base` (v1's schedule) | **`dense_early`** | `wide_cheap` | `dense_hist` |
|---|---|---|---|---|
| future offsets asked | 2, 5, 10, 15, 20 | **1, 2, 3, 5, 8, 12, 16, 20** | 1, 2, 3, 5, 8, 12, 16, 20 | 2, 5, 10, 15, 20 |
| history offsets asked | −3, −1.5, 0 | −3, −1.5, 0 | −3, −1.5, 0 | **−3, −2, −1, −0.5, 0** |
| image edge px | 448 | 448 | **256** | 448 |
| **future frames actually delivered (mean)** | **2.8** (max 4) | **5.0** (max 7) | 5.0 | 2.8 |
| images per window (mean) | 5.4 | 7.6 | 7.6 | 7.0 |
| prompt tokens (mean) | 1983 | 2445 | **1442** | 2318 |
| generated tokens (mean) | 124.1 | 140.9 | 165.2 | 132.5 |
| **generation seconds (median)** | **4.87** | 5.07 | 4.92 | 5.11 |
| peak VRAM (GiB) | 17.00 | 17.13 | **16.80** | 17.13 |
| Pass-A parse failure | **0.00** | 0.02 | 0.05 | 0.01 |
| **turn-detection recall** | 0.750 | **0.857** | 0.643 | 0.750 |
| 3-class accuracy over all | 0.7816 | 0.7816 | 0.7586 | 0.7701 |
| `road_geometry` κ vs `base` | — | 0.776 | 0.557 | **0.827** |

**Finding 1 — `base`'s future schedule was designed for clips we do not have.** Our episodes are ~199 frames
(19.9 s), and the labeling windows sit at t ∈ {0, 40, …, 160}. So `base`'s 15 s and 20 s offsets mostly **do not
exist**: it asks for 5 future frames and **delivers 2.8**. `dense_early`, by concentrating offsets early
(1, 2, 3, 5, 8, 12, 16, 20), delivers **5.0**. Nearly half of the reference plan's future schedule was being
silently dropped by `pick_frames`, and no previous run reported it because only frames that exist are ever offered.

**Finding 2 — the denser early future is worth its cost.** Turn-detection recall **0.750 → 0.857 (+10.7 pp)** for
**+23 % prompt tokens, +4 % generation time and +0.13 GiB VRAM**. Overall 3-class accuracy is unchanged (0.7816
both, McNemar p = 1.00, 2 vs 2 discordant) — as expected, because the extra detections still get their *direction*
wrong; direction is closed (§1). **Detection is the capability the VLM actually has, and it is the one that
improved.**

**Finding 3 — resolution is not a free saving, and this one is counter-intuitive.** `wide_cheap` is
`dense_early`'s exact frame schedule at the **stored** 256 px instead of the upscaled 448 px. It cuts prompt
tokens 41 % (2445 → 1442) — and buys almost nothing in wall clock (5.07 → 4.92 s median, ~3 %), because
generation dominates prefill. What it costs is large: **turn-detection recall 0.857 → 0.643 (−21 pp)**, Pass-A
parse failures 2 % → 5 %, and the `road_geometry` call agrees with the 448 px arm at only **κ = 0.557**.

The counter-intuitive part: **448 px is pure upscaling — it adds no information to a 256² source — and it still
measurably improves the model's reading of the scene.** The binding constraint is the vision tower's patch grid,
not the source resolution: more patches means finer spatial tokens even over interpolated pixels. *Do not
"optimise" this pipeline by feeding frames at their native resolution.*

**Finding 3b — denser HISTORY buys nothing, and that is worth knowing because it was half the ask.** Sayed asked
for more past *and* future scenes. Going from 3 to 5 history frames costs **+17 % prompt tokens and +5 %
generation time** and returns **identical turn-detection recall (0.750 both)**, slightly worse direction accuracy
(0.381 vs 0.429, noise), slightly worse 3-class accuracy (0.7701 vs 0.7816), and a `road_geometry` call that
barely moves (**κ = 0.827** vs `base`, the highest agreement of any pair here). **The evidence is in the future
frames**, exactly as the two-pass design assumed — the history block's job is to establish the ego state, and 3
frames already do it. The past half of "past and future scenes" is answered: *do not spend tokens there.*

**Finding 4 — labels are only moderately stable to the frame plan, and that is a caveat on everything in §6.**
`road_geometry` (Pass A) agrees at **κ = 0.776 / 89 % raw** between `base` and `dense_early`, and only
**κ = 0.557 / 78 %** between `dense_early` and `wide_cheap`. So roughly **1 window in 9** changes its geometry
label when the frames change, at fixed prompt and fixed model. With no ground truth, this cross-plan agreement is
one of the few reliability numbers available, and it says the labels are *reproducible-ish*, not deterministic.

**Decision: `dense_early` (448 px).** Chosen on the delivered-frame count and turn-detection recall, at a cost of
+4 % generation time. Not chosen on scene-label quality — that cannot be measured here, and the report says so.

## 4. What is extracted, and the one new kind of number we allow

The provenance split is unchanged and binding: **kinematics own `VTARGET` / `LONMODE` / `LATMANEUVER` / `DYN` /
`HEADWAY` / `ROUTE`; the VLM owns the WHY.** Pass A stays independent (history + future *frames*, no numeric future
ego track) so its answers can cross-check ours; Pass B adds the numeric summary for interpretation and is refused
entry to every agreement statistic.

**Two asks were deleted, not merely re-budgeted.** `TACTICAL.INTERACT` and `TACTICAL.SIGNAL` were ~0 % informative
for *both* models in the head-to-head, for a structural reason — a forward camera cannot see a blinker, and the
vocabulary already assigns `SIGNAL` to `human/can`. v2 stops asking and stamps `not_asked: ["INTERACT","SIGNAL"]`
on every record, so a consumer can tell *"the model declined"* from *"we stopped asking"*. This also buys back
output tokens for the fields that matter.

**The one new class of number: event time offsets.** This is the change that makes the corpus useful for the
metric suite at all. An intersection traverse is 5–10 s, a roundabout 8–20 s, a merge 5–15 s
([TanitEval v2 §3.5](../../Benchmarks%20&%20Eval/TANITEVAL_V2_METRIC_SUITE.md)) — but our planning window is **2 s**.
A 2 s window inside a roundabout is kinematically indistinguishable from a constant-radius curve, which is exactly
why §6.3 refuses intersection capability at a 2 s horizon. **The VLM's future frames reach 20 s, so for
long-horizon scenario labels it is not a fallback — it is the correct instrument.** v2 therefore asks for
`geometry_event_time_s`, `geometry_event_end_time_s` and `scenario_event_time_s`, so a window at *t* can be tagged
with an event that resolves at *t + 12 s*.

This does not violate the "never ask for a number it cannot measure" rule, and the distinction is load-bearing:
**the model must COPY one of the frame offsets it was shown.** It is a selection from a shipped list, exactly like
every categorical slot — not a measurement. The scorer enforces it: an offset that is not in that window's
`future_offsets_s` is counted as **fabricated**, and `vlm_labels_to_lake.py` **drops it to `null`** rather than
passing it downstream. The 48-clip pilot's finding (band edges fabricated on 48 % of clips) is what this rule is
made of, and metres and m/s remain forbidden.

**Coarse lead state, and a refusal written into the data.** `stack/tanitad/lake/enrich.py:61-65` stubs
`lead_state` as `{"present": None, "gap_m": None, "closing_speed_ms": None, "ttc_s": None, "_pending": True}`, and
that stub is why headway/TTC metrics were refused. The VLM's `ENUMS_OBS` can supply `lead_lane`,
`distance_bucket`, `relative_motion` — **enough to STRATIFY a metric, not enough to compute a TTC.**
`vlm_labels_to_lake.py` fills the categorical fields, leaves `gap_m` / `closing_speed_ms` / `ttc_s` as `None`, and
attaches `_metric_fields_unavailable` explaining why. A downstream consumer asking this corpus for headway in
seconds is refused **by the data**, not by a sentence in a document that nobody re-reads.

**Two output shapes, because the consumers disagree.** `enrich.py`'s sidecars are per **episode** (weather does not
change mid-clip, so scene axes aggregate — and the converter reports the per-axis agreement share, because an
episode whose windows split 3/2 on `weather` has not measured the weather). TanitEval's scenario strata are per
**window**, keyed `(episode, t)`, because one clip contains both a straight stretch and a junction. The converter
writes both from the same records and refuses to blur them.

## 5. Coverage — stated as a percentage, and what it does and does not license

**The denominators, from primary sources.** Canonical train corpus `physicalai-train-e438721ae894`:
**2,376 episodes / 406,099 windows** (MODEL_REGISTRY §1). TanitEval's eval corpus: **881 windows / 40 episodes**
on `physicalai-val-0c5f7dac3b11` at stride 8 (TANITEVAL_V2_METRIC_SUITE). The prior VLM corpus was
**595 records ≈ 0.15 %** of the train windows, which is why it was ruled eval-only.

### What the run targets

| target | windows | of what | %  |
|---|---|---|---|
| pod3 val build, **all 80 episodes**, stride 40 | 400 | that build's 80 episodes | **100 % of episodes**, 5 windows each |
| ⤷ joinable to the **canonical** eval episodes | 8 eps × 4 windows = **32** | 881 eval windows | **3.6 %** (20 % of eval *episodes*) |
| stratified train sample | ≤600 | 406,099 train windows | **≤0.148 %** |

**The canonical-val number is the one that matters and it is small, for the reason in §7** — only 8 of the 40
canonical val episodes exist on the pod that had a free GPU. The `t = 0` window of each episode has no TanitEval
counterpart (an eval window is keyed by its *start* and needs 8 frames of history), so 4 of each episode's 5
windows join; `vlm_labels_to_lake.py` emits `taniteval_window_start = t − 8` so nobody re-derives that off-by-eight.

**Two ordering changes so that a partial run is still a useful corpus** — this campaign is hours long and a
report should not depend on it finishing:
- The **val manifest is now t-major**: the first 80 windows cover *every* episode at `t=0`, then every episode at
  `t=40`, and so on. Episode-major order would have given a stopped run the first N episodes completely and the
  rest not at all — the opposite of "full coverage of the val episodes first".
- The **train manifest is deterministically shuffled** (seed 1234), so any prefix is a uniform subsample of the
  stratified draw. Measured: at n=300 of 600 the maximum stratum-share drift is **3.8 pp**, and the prefix still
  touches 281 distinct episodes.

### What the stratified train sample is, and why it is not uniform

A uniform draw over this corpus spends most of its budget re-confirming that a straight road is straight. The
sampler walks every episode's pose track (frames stay on disk via `mmap`; **21,393 candidate windows** at stride
20 over all 2,376 episodes, ~6 min CPU) and fills an **equal quota per stratum**:

| stratum | candidates | sampled (600) | enrichment |
|---|---|---|---|
| `turn_left` | 2,977 | 164 | 2.0× |
| `turn_right` | 2,802 | 155 | 2.0× |
| `sharp_turn` | 3,541 | 212 | 2.1× |
| `brake` | 3,278 | 163 | 1.8× |
| `accel` | 3,826 | 139 | 1.3× |
| `stop_approach` | 1,656 | 107 | 2.3× |
| `launch_from_stop` | 905 | 83 | 3.3× |
| `high_speed` | 5,034 | 108 | 0.8× |
| `steady` | 14,289 | 298 | 0.7× |

Windows with a valid kinematic turn are **27.0 %** of the corpus and **53.2 %** of the sample.

⚠️ **The first attempt at this was wrong and is worth recording.** Inverse-frequency *weighting* plus a greedy
sort simply drains the rarest stratum first: it gave **380 of 600** windows to `launch_from_stop` (the rarest, 905
candidates) and **1** to `high_speed` (5,034 candidates). `high_speed` is where the flagship's dominant
longitudinal failure lives, so that sample would have been *worse than uniform* for the thing most in need of
labels. Explicit per-stratum quotas fixed it. The candidate walk is cached, so re-sampling with a different budget
costs seconds rather than the 6-minute pose walk.

### What the production run measured, at the point this note was written

The val run was **in flight** at 30 windows / 24 episodes when this was written; it continues unattended and the
train sample is queued behind it. Everything below re-derives from `val_full.jsonl` with no pod.

| | production (`dense_early`, `--max-new-b 1200`, n=30) | v1 baseline (n=40) |
|---|---|---|
| Pass-A parse failure · ROUTE enum violation | **0 % · 0 %** | 0 % · 0 % |
| Pass-A evidence unique-string rate | **1.000** | 0.875 |
| Pass-A `route_event_time_s` / `geometry_event_time_s` **fabricated** | **0.0 % / 0.0 %** | — |
| Pass-B strict JSON parse | 0.467 | 0.675 |
| **Pass-B usable (strict + salvage)** | **1.000** (16 rows recovered) | 1.000 (13 recovered) |
| Pass-B truncation | 0.533 | 0.325 |
| Pass-B mean slot violation (30 slots) | 0.024 | 0.004 |
| **Pass-B mean informative rate** | **0.819** | 0.554 |
| Pass-A confidence band | `high` **30/30** — modal 1.00 | `0.99` 40/40 |

**The informative rate is the number worth noticing: 0.554 → 0.819.** The terser prompt plus the salvage means a
far larger share of slots come back with an actual token rather than `unknown`/`none`/missing — which is the whole
point of running the pass. The cost is a slightly higher slot-violation rate (0.4 % → 2.4 %), still small, and
driven mostly by the vocabulary bleed §6.1b describes.

**And confidence is now degenerate on a third independent sample** (30/30 `high`). Three encodings, three
degenerate distributions. §9 item 2 stands: delete the field.

### What this coverage licenses

- ✅ **Eval-side stratification and failure forensics** on the episodes it covers — scenario strata, lead-state
  buckets, sign reads, the WHY sidecars.
- ✅ **A rare-event probe set** for training diagnostics: 600 windows enriched 2× on turns is a far better lens
  than 600 uniform windows, even at 0.148 % coverage.
- ❌ **NOT training-grade labels.** 0.148 % is not coverage, and the previous corpus was ruled eval-only at
  essentially the same fraction. Nothing here changes that; a bigger record count must not be read as a bigger
  mandate.
- ❌ **NOT a drop-in for the TanitEval v2 scenario strata on the canonical corpus** — 3.6 % of the eval windows
  until §7 is resolved.

## 6. Are the scenario labels good enough to build intersection / roundabout metrics on?

**Short answer: the labels are good enough to STRATIFY with, and the corpus is not big enough to SCORE with.
The binding constraint is not label quality — it is that the events are rare and the sample is small.**

### 6.1 What can be checked with no ground truth, and what it says

Scenario slots have no stored dataset column — producing one is why this run exists — so *nothing below is
accuracy*. These are the properties that are measurable without truth:

| property | measurement | reading |
|---|---|---|
| **schema adherence** | Pass-B mean slot violation **0.4 %** (v1) / 1.3 % (v2a) over 30 slots; Pass-A ROUTE enum violations **0 %** | ✅ the model selects from our vocabulary, it does not invent tokens |
| **event-time validity** | Pass A **100 %** of `geometry_event_time_s` and `route_event_time_s` are offsets we actually offered (n=113) | ✅ (Pass B's are contaminated — §2) |
| **reproducibility under frame plan** | `road_geometry` κ = **0.776** (base ↔ dense_early, 89 % raw); κ = 0.557 at 256 px | ⚠️ ~1 window in 9 changes label |
| **format failures** | Pass-A parse failure 0 % (base) / 2 % (dense_early) / 5 % (256 px) | ✅ |
| **slot bleed** | `road_geometry` answered with a ROUTE token (`left`/`right`) on **3 of 100** windows | ⚠️ now rejected to `unknown` by the converter and counted |
| **kinematic co-occurrence** | P(kinematic turn \| VLM says junction/roundabout/merge/fork) ≈ 0.67 | weak positive — and a junction traversed straight is a *correct* label with no turn signature |
| **event detection** | turn-detection recall **0.750–0.857** depending on frame plan | ✅ the VLM is a competent event detector, as three prior measurements also found |

### 6.1b The Pass-A-primary choice is a trade, and the audit sheet is what settles it

Sourcing `road_geometry` from Pass A buys **independence** — §2 measured Pass B parroting our own kinematics — but
it is not free, and the honest statement is that we cannot yet show it is also *more accurate*:

- Pass A's geometry is a **secondary ask inside a route-focused prompt**; Pass B's is the core of a
  scene-interpretation prompt. Prior belief should favour Pass B on quality and Pass A on independence.
- **Both slots bleed neighbouring vocabularies.** Pass A answers a ROUTE token (`left`/`right`) in the geometry
  slot on ~3 % of windows; Pass B answers `intersection` — a `road_type` token — in it. Both are now rejected to
  `unknown` and counted rather than passed into the strata, but a rejected answer is a lost window either way.
- On the first production windows the two passes visibly disagree: `ep_00010 t=0` is Pass A `junction` vs Pass B
  `curve_right`; `ep_00009 t=0` is Pass A `right` (rejected) vs Pass B `junction` on a window the kinematics call
  a right turn.

**We chose independence because the alternative is circular** — a stratum derived from our kinematics cannot be
used to judge a model against those same kinematics — and because a contaminated label is wrong in a *correlated*
way, which is worse than being wrong at random. But this is exactly the question a human gold slice answers in an
hour, and the audit sheet ships both columns side by side (`SHIPPED_geometry` from Pass A,
`passB_geometry_CONTAMINATED`) so the verdict can be read off directly. **If Pass B turns out materially more
accurate, the right response is to move `road_geometry` into Pass A's ask properly (§9 item 1) — not to go back
to the contaminated source.**

**A worked example of why the geometry call is worth having even though ROUTE is not.** `ep_00047 t=0`: the
kinematic label is a **right** turn of −77.95°. Pass A says ROUTE `left` — wrong, the §1 bias exactly — but calls
`road_geometry: junction` with band `high` and Pass B tags it `unprotected_turn`. **The model reads the scene
correctly and names the direction wrongly. Those are separable capabilities**, and this run's whole premise is
that we take the one it has and keep computing the one it does not.

### 6.2 The blocker: the strata are too thin to score

TanitEval v2's own rule R3 marks any stratum under **30 windows** low-confidence. Two independent samples of the
shipped (Pass-A) `road_geometry`, and what they project onto the 400-window val run:

| geometry | ablation, 100 windows | **production, first 30 windows / 24 episodes** | projected / 400 | ≥30-window floor |
|---|---|---|---|---|
| `straight` | 67 | 17 | 227–268 | ✅ |
| `curve_right` | 10 | 4 | 40–53 | ✅ |
| `curve_left` | 12 | 3 | 40–48 | ✅ |
| **`junction`** | **5** | **4** | **20–53** | ⚠️ **borderline** |
| **`roundabout`** | **1** | **0** | **0–4** | ❌ |
| **`merge`** | 0 | 0 | 0 | ❌ |
| **`fork`** | 0 | 0 | 0 | ❌ |
| `unknown` (incl. 2 rejected out-of-enum answers) | — | 2 | — | — |

**The two samples disagree on `junction` by 2.7×** (5 % vs 13.3 %), which at n=4 is exactly what small numbers do;
the honest range is **20–53 windows over the full val run**, i.e. straddling the floor. **`roundabout`, `merge` and
`fork` are not borderline — they are absent**, and no scoring convention fixes an empty cell.

So: an **intersection** stratum is plausibly buildable on the full val run and must be re-counted when it finishes
(not projected — the disagreement above is the reason). A **roundabout or merge** capability metric is not
buildable at this coverage at all. And the blocker is *sampling*, not *labeling* — which is the good version of
this news, because sampling is fixable and label quality would not have been.

**The fix is already half-built: the stratified TRAIN sample.** It is enriched 2× on turns (53.2 % of windows
carry a valid kinematic turn against 27.0 % in the corpus), so its eventful-geometry rate should be markedly
higher than val's — and it draws from **21,393 candidate windows over 2,376 episodes**, so the pool is there.
Filling a 30-window `roundabout` stratum is a matter of enlarging that draw, and the candidate walk is cached so
re-sampling costs seconds.

### 6.3 Verdict

**NOT YET — and here is precisely what is missing.**

| # | what is missing | status | cost to close |
|---|---|---|---|
| 1 | **n ≥ 30 per eventful stratum.** `junction` ~20, `roundabout` ~4, `merge`/`fork` 0 on val. | the binding blocker | enlarge the stratified draw; the candidate pool (21,393 windows) already exists and is cached |
| 2 | **A human-verified gold slice.** Nothing above is accuracy; schema-cleanliness and self-consistency cannot tell us whether `junction` means a junction. | **shipped as `audit_*.tsv`** — stratified toward eventful geometries, with the model's own evidence sentence and the kinematic label beside each row | a person, ~1 h for 60 rows |
| 3 | **The canonical val join.** 8 of 40 episodes, 3.6 % of the 881 eval windows. | §7, escalated | ~35 min of GPU on `tanitad-eval`, or a 4.7 GB HF relay |
| 4 | **A `scenario_tag` that is not conditioned on our own kinematics.** `road_geometry` moved to Pass A; `scenario_tag` has no Pass-A twin yet. | §9 item 1 | ~4.9 s/window — add it to Pass A |
| 5 | **A stability budget.** ~1 window in 9 changes its geometry label when the frame plan changes. | measured, unresolved | either fix the frame plan (done: `dense_early`) and quote κ as a known noise floor, or majority-vote across plans at 3× cost |

**What can be done today, without any of the above:** use the labels as a **stratification and forensics layer** —
split existing metrics by `road_geometry`/`scenario_tag`, pull the `curve_left`/`curve_right`/`straight` strata
(which do clear n ≥ 30), and use the CoC/evidence sidecars for failure analysis. **What must not be done:**
publish an intersection or roundabout *capability number*. The strata that name those events do not have the n,
and no scoring convention fixes an empty cell.

---

## 7. ESCALATION — the val-build join, and why it blocks 32 of 40 canonical episodes

**This needs a decision; it is not a note-to-self.**

The corpus every TanitEval and MODEL_REGISTRY number is computed on is `physicalai-val-0c5f7dac3b11` — 40
episodes, resident on `tanitad-eval`. The only val epcache on the GPU that was free is
`physicalai-val-f1b378f295ae` — 80 episodes, on `tanitad-pod3`. **They are different draws, not nested subsets.**
`split_clips` (`stack/tanitad/data/physicalai.py:505-513`) permutes with `torch.randperm(len(clips))`, so the val
membership depends on the clip-count bound: the canonical build came from a 200-clip discovery (160 train / 40 val),
pod3's from a 500-clip discovery (400 train / 100 val, 80 materialised — `DONE` says `{"episodes": 80}`).

**Measured overlap: 8 of 40.** By exact pose fingerprint — the tuple of
`(t, net_dyaw_deg, arc_m, peak_kappa, clip_len)` from `route_from_future_v21` at all five stride-40 windows of an
episode — 8 canonical episodes appear in pod3's build. That is exactly the 8 expected from independent draws
(40 × 80/400). The map is shipped as `val_build_episode_map.json` and those 8 episodes' labels can be joined to
canonical val windows today.

**pod3 cannot rebuild the canonical val.** Its R0 root `/workspace/pai_build` yields only **4** clips from
`discover_r0_clips` (verified), so no `--episodes` bound reproduces cache key `0c5f7dac3b11`. Pods cannot SSH each
other. **Two options, both cheap, one decision needed:**
1. Run this harness on `tanitad-eval` once its efficiency sweep and the queued REF-C eval finish (~35 min of GPU
   for 200 windows of Pass A+B over the canonical 40).
2. Push the 40-episode epcache (~4.7 GB) `tanitad-eval` → HF → `tanitad-pod3` (~2 min of transfer at the measured
   118 MB/s, per the standing pod-to-pod recipe), and finish it here.

Until one of those happens, **the scenario strata this run produces cover 8/40 = 20 % of the canonical val
episodes**, and the rest of the corpus is joinable to *training* (the canonical 2376-episode parity build) but not
to the published eval windows.

---

## 8. Reproduce

Every number in §1–§3 re-derives from the committed records. **No pod, no GPU, no 4.4 GB val build.** Verified:
the probe verdict above was recomputed from the repo alone (200 windows / 40 episodes, left share 0.7447 vs
0.6667, recall-on-right 0.2069 in both arms).

```bash
D="TanitAD Research Hub/Data Engineering/Implementation/incoming/2026-07-21-vlm-production-semantic"

# §1 the enum-order probe verdict
python stack/scripts/vlm_compare_score.py --out "$D" \
  --arms r2_as_written,r2_right_first --json /tmp/probe.json

# §2 the prompt before/after   §3 the frame ablation
python stack/scripts/vlm_semantic_score.py --out "$D" --arms p1_v1,p1_v2 --json /tmp/prompt.json
python stack/scripts/vlm_semantic_score.py --out "$D" \
  --arms ab_base,ab_dense_early --compare --json /tmp/frames.json

# §6 the two shapes the rest of the stack consumes, + the human audit sheet
python stack/scripts/vlm_labels_to_lake.py --jsonl "$D/val_full.jsonl" \
  --windows-out "$D/scenario_strata_val.jsonl" --sidecars-out "$D/sidecars_val"
python stack/scripts/vlm_semantic_score.py --out "$D" --arms val_full \
  --audit-sheet "$D/audit_val.tsv" --audit-n 60

# rescue whatever is on the pod right now (idempotent, safe mid-run)
bash stack/scripts/pod_ops/pull_vlm_records.sh
```

The window manifests are pure functions of (val build, episode range, stride, `t0`); deleting and rebuilding one
reproduces it exactly, which is what makes every arm paired.

---

## 9. What to change next, in priority order

1. **Add `scenario_tag` (and its event time) to PASS A.** This run proved Pass B parrots anything our conditioning
   block hands it, which is why `road_geometry` and `geometry_event_time_s` now come from Pass A. `scenario_tag`
   has no Pass-A twin, so it is the one strata field still sourced from the contaminated pass. Pass A costs
   ~4.9 s/window against Pass B's 24–135 s, so the fix is nearly free. **Highest-value change to the prompt.**
2. **Delete `route_confidence_band` if production reproduces a modal share ≥ 0.90.** Two encodings, two
   degenerate distributions; a constant that looks like evidence is worse than an absent field.
3. **Never raise a token ceiling as a fix for truncation on this model — LOWER it, and judge it by BLOCK
   SURVIVAL, not by the truncation rate.** It spends what it is given: 2200 → 32.5 % truncation, 3500 → 61.5 %,
   median generation 24 s → 135 s. The completed-reply distribution is bimodal (**max 1146 tokens over 34
   completions**, 572–771 under `dense_early`), so a **1200** cap loses nothing that would have finished and cuts
   a runaway from 136 s to 47 s. The head-to-head's "budget ≥3500 before any bulk run" is refuted — it was
   backwards. And the rate itself is a poor headline: at 75 % truncation the production run still recovers
   `SCENARIO`/`STRATEGIC`/`TACTICAL`/`OBSERVATIONS` on **14/14** windows and loses only the CoC narrative.
4. **Choose the frame plan and the token budget TOGETHER.** They are coupled and this run treated them
   separately: the plan was picked on Pass-A evidence, the budget measured on a different plan, and the truncation
   rate then more than doubled in production (30 % → 75 %). The completion-token distribution barely moved, so the
   cap survived — but the next ablation should measure Pass-B verbosity per plan, not just Pass-A cost.
5. **Extend the salvage to the head-to-head's banked records.** `vlm_compare_score.py` still counts a truncated
   Pass B as a parse failure; the head-to-head's "Reason2 parse rate 74 %" is therefore a *strict* rate, and its
   usable rate is almost certainly ~100 %. The comparison's verdict does not change (it was decided on Pass A),
   but the number is quoted in two documents and should carry the distinction.
6. **A human-verified gold slice.** Shipped as `audit_*.tsv` — stratified toward the eventful geometries, with the
   model's own evidence sentence and the kinematic label beside each row. Nothing else can convert "schema-clean
   and self-consistent" into "correct" (§6).

---

## 10. Deliverable manifest

**Everything below is STAGED in the working tree, not committed and not pushed.**

| artifact | where | note |
|---|---|---|
| `vlm_semantic_labels.py` — production harness, prompt `vlmsem-2026-07-21-b`, frame plans, stratified sampler | `repo:stack/scripts/` | `--prompt-version v1` still runs the frozen v1 prompt |
| `vlm_semantic_score.py` — scorer / QA / audit-sheet, **no pod, no GPU** | `repo:stack/scripts/` | reads both record layouts; re-salvages legacy records from `raw` |
| `vlm_labels_to_lake.py` — per-window strata rows + episode sidecars + the salvage parser | `repo:stack/scripts/` | stdlib only; the metric-field refusal lives here |
| `pull_vlm_records.sh` — one-command rescue from the pod | `repo:stack/scripts/pod_ops/` | idempotent, safe mid-run |
| `test_vlm_semantic.py` — **26 tests**, the suite's first VLM coverage | `repo:stack/tests/` | full suite **637 passed / 2 skipped**, green |
| This note | `repo:TanitAD Research Hub/Data Engineering/Research/` | |
| Records, manifests, scored JSON, prompts, episode map, `INTAKE.md` | `repo:…/Implementation/incoming/2026-07-21-vlm-production-semantic/` | see `INTAKE.md` for the file-by-file guide |
| `train_candidate_census.json` — kinematic stratum census of all **21,393** candidate windows over the canonical 2,376-episode train corpus | same directory | makes re-sampling the train draw a pod-free operation |
| **Rescued** legacy corpus (400 Pass-A + 160 Pass-B) that had lived **only** on `tanitad-pod3:/workspace` | same directory | different schema (no inline kinematic GT, no token counts) — kept because a pod is not storage |
| **Production records, in flight** | `tanitad-pod3:/root/vlmprod/{valfull,trainstrat}/` | ⚠️ **the only artifacts that exist in one place.** `bash stack/scripts/pod_ops/pull_vlm_records.sh` rescues them at any point, including mid-run |

**GPU lock.** Held as `vlm-production` on `tanitad-pod3` since 07:35 Berlin and **still held** for the running
production job. Release with `/root/vlmprod/gpu_lock.sh release vlm-production` once `PROD_ALL_DONE` appears in
`/root/vlmprod/prod.log` — not before, or a sibling agent will land on a busy GPU.

**No HF token was read.** The model was already cached at `/root/hf/hub/models--nvidia--Cosmos-Reason2-8B` and
every run used `HF_HUB_OFFLINE=1`, so the campaign never needed a credential. No PhysicalAI-AV frame or crop left
the pod; only labels and JSONL entered the repo.
