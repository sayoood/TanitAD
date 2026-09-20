# D-DAC-HUMAN-ZERO-1, separated: **the rule reads ONE channel, and the human drives over painted road markings**

**Date** 2026-09-20 · **Owner** DataFlyWheel · **Asked by** Master Mind · **0 GPU, CPU only, read-only**
⛔ **The current rule's numbers stay exactly as landed.** Every variant below is a MEASUREMENT of
what that rule *would* read on the same windows. Nothing here is chosen, tuned or recommended, and
no redefinition is proposed without its own pre-registration.

## Reproduced first

On the same 736 held-out windows, through the harness's own loaders
(`taniteval.tools.s1_pass.Corpus`), my per-window `dac` equals the loader's **and** the landed
`roundtrip_halfB_A8cfg.json` value on **736 of 736** windows. Zero rate **0.4457** — the landed
44.6 %. A characterisation of a number I could not reproduce would describe my own pipeline, so
this runs first and the script stops if it fails.

## The three causes, separated

| | share of the 328 zeroed windows | what it is |
|---|---|---|
| ⭐ **the rule reads one channel of nine** | **264 of 328 (80.5 %)** are zeroed *only* by cells whose mass is **road surface** | the map calls the cell "lane / road line", "crosswalk" or "arrow / text"; `dac_from_drivable` reads the **"drivable" channel alone**, so paint on the road reads as **not drivable** |
| **the map says nothing there** | 754 of 4,431 violating samples (17.0 %) are "seen, no map class" ≥ 0.5 | under-segmentation — but **near the ego**, not at range (748 of those 754 sit in 0–15 m) |
| **genuine off-map driving** | **33 of 328 (10.1 %)** have any explicitly off-road cell (sidewalk / edge / hatched ≥ 0.5) — **4.5 % of all 736 windows** | the only candidates for the human actually leaving the drivable surface, and most are kerb-adjacent crawls in narrow streets (see the renders) |

**The mechanism in one line:** a 41-tick trajectory samples **164** cells; **3.7 %** of those cells
read below the drivable threshold; the rule zeroes the window if **any single one** does.

## 1. Map quality — under-segmentation at RANGE is refuted

Violations are **most** frequent in the **near** band, not the far ones. Counts alone would say the
opposite of the rate, because a 4 s trajectory spends most of its ticks close to the ego — so the
denominator is reported with every count.

| band | admissible samples | violating | **violation rate** | mean drivable fraction under the footprint | A1 `p_drivable_given_seen` (halfB, whole grid) |
|---|---|---|---|---|---|
| 0–15 m | 53,871 | 2,982 | **5.54 %** | 0.928 | 0.354 |
| 15–30 m | 32,970 | 905 | **2.74 %** | 0.954 | 0.352 |
| 30–45 m | 14,092 | 426 | **3.02 %** | 0.947 | 0.331 |
| 45–60 m | 5,443 | 118 | **2.17 %** | 0.953 | 0.312 |

⭐ Two readings, both worth keeping. **(a)** The far bands are the *best* behaved, so "under-segmentation
at range" is not the cause. **(b)** Under the footprint the map reads **0.93–0.95** drivable against A1's
**0.31–0.35** over the whole grid — a **2.7× enrichment**, which is what a map that works looks like
when the car is on the road. The map is not broken; the *read-out* of it is one channel wide.

**Where the violating cells are, by what the map actually says there** (4,431 samples):

| the cell's own mass | samples | share |
|---|---|---|
| **road surface** (drivable + lane line + crosswalk + arrow/text) ≥ 0.5 | **2,822** | **63.7 %** |
| explicitly off-road (edge + hatched + sidewalk) ≥ 0.5 | 804 | 18.1 % |
| "seen, no map class" ≥ 0.5 | 754 | 17.0 % |
| no class reaches 0.5 | 51 | 1.2 % |

argmax class of the violating cells: lane / road line **1,443**, crosswalk **1,244**, seen-no-class
755, sidewalk / verge 532, drivable 354 (mixed cells below 0.5), arrow / text 103.

⛔ **63 windows are zeroed by the ego's own footprint at tick 0** — the recorded car's position at
the moment the window opens, where it is on the road by construction. 46 of those 77 tick-0 samples
are road-surface cells.

⚠️ **Seen-fraction abstention cannot help:** only **15 of 4,431** violating samples sit on a cell with
a seen fraction below 0.75 (median seen fraction **1.000**).

## 2. Rule strictness — the zero-rate curve

Same 736 windows, same map, one rule change at a time. ⛔ **No winner is picked here.**

| rule | human zero-rate |
|---|---|
| **V0 — the current rule** (any corner, any tick, drivable < 0.50) | **44.57 %** |
| V1 — ≥ 2 corners at the same tick | 19.57 % |
| V2 — ≥ 2 consecutive ticks | 33.42 % |
| V2b — ≥ 4 consecutive ticks | 23.23 % |
| V3 — threshold 0.25 instead of 0.50 | 26.49 % |
| V4 — abstain unless the cell's seen fraction ≥ 0.75 | 44.43 % |
| V5 — abstain unless the cell's seen fraction ≥ 0.90 | 44.29 % |
| ⚠️ P1 — *definition change*: road surface (drivable + paint) < 0.50 | 8.70 % |
| ⚠️ P2 — *definition change*: explicit off-road (edge + hatched + sidewalk) ≥ 0.50 | 4.48 % |
| ⚠️ P1 + V1 | 5.43 % |
| ⚠️ P1 + V2 | 6.93 % |
| ⚠️ P2 + V2 | 3.26 % |

⭐ **Strictness and definition are different axes, and the data separates them.** Every
strictness-only knob (V1–V5) leaves the rate at **19.6 % or above**; the two **definition** changes
(P1, P2) reach **8.7 % / 4.5 %** on their own. A knob that only asks for *more* evidence of the same
one-channel read cannot fix a read that is looking at the wrong channels.

⚠️ **P1 and P2 are PROPOSALS, not findings.** Each would need its own pre-registration with both
outcomes committed, its own known-value control, and a statement of what it would then fail to
catch — P2, for instance, is silent wherever the map has no class at all, and would score a car on
an unlabelled surface as compliant.

## 3. Genuine off-map driving — what the renders show

`media/`, 14 PNGs, sha12 names only. Each has the camera frame at t0 (context, **not** projected),
the drivable fraction the rule reads, and the map's own class; every corner sample is drawn, green
where the rule is satisfied and red where it fires.

| stratum | what a human eye sees |
|---|---|
| **near-threshold** (1 violating sample, 4 clips) | the car drives down the middle of a wide drivable corridor; **one** corner clips a painted line at the last tick. `264a78e8fad1 t130`: the violating cell is **88 % lane line, 12 % drivable**, every other one of the 164 samples reads drivable = 1.00 |
| **median** (8–9 violating samples, 4 clips) | a crosswalk or a lane line crossed by the path; the corridor around it is fully drivable |
| **worst** (41–89 samples, 4 clips) | narrow streets with parked cars, where the mapped corridor is ~3 m wide and the 2.30 m car crawls with its left corners within ~0.4 m of the mapped edge. `7be92e6870d8 t48` is the clearest: the camera shows a one-lane street with parked cars and a pedestrian on the left pavement |
| **clean control** (dac = 1) | the whole footprint track sits inside white (drivable = 1) cells for all 164 samples |

⚠️ **The map alone cannot settle the 33 kerb-adjacent windows.** They are the honest candidates for
cause 3, and judging them needs the trajectory **projected into the camera** — deliberately not done
here, because an unverified projection overlay would mislead worse than none. The tools exist
(`RigCamera.from_extrinsics` per clip + `project_rig_to_frame`); it needs two known-value controls
first (the t0 footprint must land at the image's bottom edge; a far ground point must land on the
horizon row). Say the word and it is a short, zero-GPU follow-up.

## Controls — a scan that reads 0 proves nothing

Four synthetic maps under a straight 10 m/s trajectory, every variant applied to each
(`controls` in `raw/dac_anatomy.json`, all as specified):

| map | required | read |
|---|---|---|
| all drivable, seen — *the "entirely inside a hand-checked drivable region" case* | **1 under every variant** | ✅ 1 under all 12 |
| all unseen | 1 under every variant (no evidence) | ✅ 1 under all 12 |
| all sidewalk, seen | 0 under the current rule | ✅ 0 |
| all "seen, no map class" — the map saying **nothing** | 0 under the current rule | ✅ 0 — the finding's mechanism in one line |

Plus two live controls: my map read must equal the loader's channel and mask on **every** window
(asserted per window, 736/736), and the per-window `dac` must equal the landed value (736/736).
The three imported modules' md5s are recorded at start and end and were **unchanged** during the
run — the harness lives in the shared D: worktree.

## What this does and does not say about the D9 reward

The repair multiplies progress by this term with the human as the reference. On this corpus **that
reference is zeroed on 44.6 % of windows, and 80.5 % of those are painted road markings under the
car**. That is a property of the DAC read-out, not of the human's driving. ⛔ It does **not** follow
that any particular redefinition is correct; it follows that the term, as read today, cannot
calibrate the repair — which is what `PREREG_D9_REWARD_REPAIR`'s new gate asks to be established
before `H-DDV2RL-3` starts.

## Reproduce

```
python code/dac_anatomy.py --out-dir raw            # 192 s, CPU, 2 threads
python code/dac_render.py --raw raw --out media     # 14 PNGs
```

`raw/dac_anatomy.json` (every aggregate + the controls) · `raw/dac_windows.jsonl` (736 rows, one per
window, with all 12 variants) · `raw/dac_violating_samples.jsonl` (4,431 rows: tick, corner, x, y,
band, the cell's channel mass) · `raw/render_index.json`.

🔒 Clip ids appear only as sha12, in file names, titles and rows alike.

## ⭐ Appended 2026-09-20 — the camera projection, approved by the MM: **the 33 are boundary cases, not off-road driving**

Controls first, and they gate the renders: an overlay that is subtly wrong misleads worse than no
overlay. All four pass on **all 7 cameras** the 33 windows use (`raw/projection_controls.json`):

| control | bar | read |
|---|---|---|
| **C1** the ego's own footprint at t0 sits at the image bottom | front-corner rows ≥ 0.75 × 416 = 312 | **471.3 / 481.4** — below the bottom edge, as a bumper-level footprint must be |
| **C2** a ground point at 60 m is on the horizon row | monotone from below, ≤ 20 px | rows **281.7 → 236.9 → 224.2 → 216.1 → 212.3** against a horizon at **201.1**; Δ(60 m) = **11.1 px** |
| **C3** the declared 120° field reaches both edges | ±60° within 1 px of columns 0 and 1023 | **−0.5** and **1023.5** |
| **C4** +y (rig left) projects LEFT, asymmetry the mount can explain | ≤ 2·f·yaw + 2·f·y₀/d + 1 px = 9.2 px | +5 m → col 428.9, −5 m → col 601.7, asymmetry **7.64 px**, mount yaw **0.477°** |

⛔ **My first C4 was WRONG and it failed every camera.** It barred symmetry about the image centre
at 1 px — which assumes a mount at y = 0 with no yaw. The measured mount carries a **0.477° yaw**,
and 2·f·yaw alone is 8.1 px of the 7.64 px observed. The bar was an idealisation, not a known
value; restated against the mount's own numbers, it passes. *(The projection was never the
problem — reporting a control miss before touching the renders is exactly why it is run first.)*

### What the 33 windows are

They come from **7 clips**, and two supply **22 of the 33** — both narrow one-way streets with
parked cars, crawled at walking pace.

| | |
|---|---|
| off-road corner samples | **804** across the 33 windows |
| distance from such a corner to the nearest **mapped drivable** cell | median **0.429 m**, p90 0.479 m, **max 1.047 m** |
| ⭐ within **one cell (0.5 m)** of the corridor | **775 of 804 (96.4 %)**; 803 of 804 within two cells; **one** sample beyond 1.0 m |
| windows whose **worst** off-road corner is within one cell | **23 of 33** |
| the mapped corridor's width where they occur | median **3.00 m** — for a **2.297 m** car, i.e. **0.35 m of clearance per side** |
| the off-road corner's own lateral offset | **1.0–1.2 m** in 26 of 33 windows = the ego's own half-width (1.1485 m) |

⇒ **These are not excursions; they are the car's own body edge touching the far side of the
boundary cell.** At 0.5 m resolution the map cannot place a kerb more finely than the distance
almost every one of these samples sits at. `media_projected/` has one image per window (path
projected into the t0 camera + the map's class panel) so this is checkable by eye, not only by
statistic: the rails run down the roadway and the red marks ride the kerb line.

⚠️ **The honest residue.** One corner is 1.047 m past the mapped edge, and the corridor width
reads **0.00 m** at some x — places where the map has **no drivable cell at all across the whole
lateral extent**, which is a map gap rather than a kerb. Those are the only candidates left for
cause 3, and they are a handful of samples, not 44.6 % of windows.

## Appended 2026-09-20 — the blind labels scored against the key

The Master Mind labelled all 60 sheets blind and landed them (`c38fc6f`,
`adjudication/LABELS_MASTERMIND.csv`, blob `160576e1ab`) BEFORE the key left this session. Scored
here against the key whose sha256 was published in the pack before any label existed.

**The three label cells, kept apart** — `cannot-tell` is never folded into agreement or
disagreement:

| stratum | population | sampled | on-surface | over-boundary | cannot-tell |
|---|---|---|---|---|---|
| A — V0 fires, P1 does not | 264 | 20 | **20** | 0 | 0 |
| B — P1 fires, P2 does not | 31 | 20 | **20** | 0 | 0 |
| C — all three fire | 33 | 10 | **10** | 0 | 0 |
| D — none fires | 408 | 10 | **10** | 0 | 0 |

⛔ **Zero `cannot-tell` is a property of the adjudicator's tie-break, not of these images.** The
rule, fixed before the hard cases and applied to all 60: label on-surface / over-boundary whenever
the surface beneath the path is visible in ANY panel; reserve `cannot-tell` for when it is visible
in none. It must not be read as *"the camera always shows enough"*.

### The scoring, both readings

| candidate | agree | false-fire | false-pass | **committed reading (§5.4, over the 60)** | ⚠️ post-hoc, population-reweighted |
|---|---|---|---|---|---|
| **V0** (current rule) | 10 | 50 | 0 | **16.7 %** — does not clear | 55.4 % [LB 41.1 %] |
| **P1** (road surface) | 30 | 30 | 0 | **50.0 %** — does not clear | 91.3 % [LB 72.0 %] |
| **P2** (explicit off-road) | 50 | 10 | 0 | **83.3 %** — does not clear | **95.5 %** [LB 75.6 %] |

⛔ **A DEFECT IN THIS PRE-REGISTRATION, REPORTED AGAINST MYSELF.** §5.4's bar (≥ 95 % agreement,
neither error direction above 5 %) was written for a REPRESENTATIVE sample, but §5.1 specifies a
DISAGREEMENT-ENRICHED one: **50 of the 60 windows were selected *because* a rule fires there**. Over
such a sample the bar is unreachable by construction for any rule that ever fires, so *"no candidate
clears"* is partly a property of the sample design and cannot separate **a wrong rule** from **a
sample built to find firing**. The reweighted column is the quantity the bar was *meant* to express;
it is **post-hoc**, it is labelled so everywhere, and ⛔ **it adopts nothing**. Which reading governs
is an amendment for the Master Mind and the PI, to be recorded BEFORE it is used to adopt anything.

### What holds under either reading

⭐ **Sixty windows spanning every disagreement region produced ZERO `over-boundary` labels.** So in
this sample every firing of every candidate is a **false alarm**, and the candidates differ only in
how often they false-alarm — which is exactly their landed corpus rates: V0 **44.57 %**, P1
**8.70 %**, P2 **4.48 %**. With no positives in the sample, "agreement" reduces arithmetically to
"one minus the false-alarm rate"; that is what the data implies, not a rate chosen for comfort.

⚠️ **Where this design is blind:** stratum D (408 windows, none firing) was sampled **10**. That is
why the lower bounds above are wide — they let every unsampled window go the other way at the
one-sided 95 % limit. A corpus-level claim needs its samples spent there, not in the firing strata.

### Two defects in MY instruments, found by the adjudicator

1. ⛔ **The renderer has no depth test.** Ground points beyond a raised foreground object are drawn
   ON TOP of it, so a snow bank, kerb or parked car in the near field can make an on-road path look
   as though it runs over the obstacle (caught on one sheet, where panel D shows the car on the
   cleared carriageway). ⭐ The bias runs **toward over-boundary**, and the adjudicator still
   returned zero — so it cannot explain the result; it makes the zero **stronger**. The next pack
   must either depth-test the overlay or say in the sheet header that it does not.
2. ⚠️ **The blinding is PROCEDURAL, not cryptographic.** The pack carries the seed and the builder,
   and `dac_windows.jsonl` carries the verdicts, so the key is reconstructible from the commit —
   the pack's own `key_is_reproducible` says so. **The adjudicator could have derived it and did
   not**, and that sentence is what makes the discipline auditable rather than asserted.

### One inconsistency in the landed labels

`RESULT_MM_LABELS.md` reports **10** rows flagged BORDERLINE; the CSV it summarises carries **9** —
W18, W19, W25, W28, W43, W44, W50, W51, W60 (checked across every field after the label, not just
the note column, because 14 notes contain commas). The PI's spot-check plan reads *"12 checks on the
10 BORDERLINE rows"*; it should read 9, or a row is missing its flag. By stratum: B 6, C 1, D 2.

## Appended 2026-09-21 — the hunt for REAL over-boundary events: three routes, **no positives**

The adjudication returned 60/60 on-surface, so the pack held nothing to miss and no candidate can
be ranked on the error that matters. These three routes look for real departures by mechanisms
that do **not** consult the rules under test.

### Route 1 — the ego dynamics, a sensor the rules never touch

The corpus's **100 Hz egomotion** (az, vz, z, quaternion) carries genuine high-frequency content
(sample-to-sample |Δaz| reaching 5.3 m/s²), so it is not a smoothed solution and a kerb strike
would show. Each window's 4 s span is located through the corpus's own v2ep time grid
(`semantic_map_gt.episode_frame_times_us`), and the statistic is `max |az − rolling median(az,
0.5 s)|`. **The threshold was fixed before any overlap with a rule was looked at:** median +
6 × 1.4826 × MAD = **4.14 m/s²**.

| | |
|---|---|
| windows scored | **736** (0 without egomotion, 0 without a usable span) |
| distribution of the statistic | median **0.81**, p90 2.13, p99 3.99, **max 4.46 m/s²** |
| over threshold | **6 windows — but only 2 distinct events**, one per clip, the same shock falling inside four overlapping windows of one clip and two of the other |
| their vertical displacement | z de-trended peak-to-peak **0.032 m and 0.071 m**, against a corpus **median of 0.059 m** — at or below typical |
| ⭐ injection control | a 10 m/s², 60 ms pulse added to the quietest window lifts the statistic from **0.07 → 10.04**, far above the threshold ⇒ **the search could have found a strike** |
| windows anywhere above 8 m/s² (half the injected pulse) | **0** |

⇒ **Route 1 finds no mounting event.** Two ordinary road jolts sit at the top of a continuous
tail, with no vertical displacement to match; nothing in 736 windows comes within a factor of two
of a kerb strike's magnitude.
⚠️ **What route 1 cannot see, stated before the result:** a smooth roll onto a flush driveway, a
paved verge or a dropped kerb makes no shock at all.

### Route 2 — wide margin (⛔ not independent of the rules; reported as a curve)

| margin past the mapped drivable edge | ≥ 0.5 m | ≥ 0.75 m | ≥ 1.0 m | ≥ 1.25 m | ≥ 1.5 m |
|---|---|---|---|---|---|
| windows | 10 | 2 | **1** | 0 | 0 |

The widest is **1.047 m** — about two cells of a 0.5 m grid. No window is far enough out that a
read-out defect could not explain it.

### Route 3 — the sweep where a FALSE-PASS would hide

Stratum D (no rule fires) is **408 windows in 55 clips**, sampled **ten** times so far.
`media_sweep/` is a blind pack of **58 sheets**: one window from each of the 55 clips, plus the
**2 route-1 shock events** and the **1 route-2 widest-margin window**, mixed in under ids from the
same shuffle so a labeller cannot tell them apart. Key held off-repo; sha256 in `raw/sweep_pack.json`.

⛔ **Drawn BY CLIP FIRST — one window per clip — so n equals the cluster count.** The previous
pack counted windows as independent observations, and its most-flattered candidate rested on three
clips.
⛔ **Panel A now states the overlay is NOT depth-tested** (a ground point behind a raised object is
drawn over it). The first render put that sentence in a font glyph matplotlib does not have, which
would have shipped the one required disclosure as an empty box; it is plain ASCII now.

### What this means for the term

With **no positives anywhere in the 736**, a rule that simply always answers "on-surface" scores
**100 %** on this corpus — better than any candidate. ⇒ **This corpus cannot rank the candidates on
the error that matters.** It can only measure how often each one false-alarms, which the anatomy
already did (44.57 % / 8.70 % / 4.48 %). Ranking them needs either a corpus that contains real
departures, or a decision that a term validated only on its false-alarm rate is not the right
reference for a human-referenced reward. ⛔ Neither is decided here.

## The ruling (Master Mind, landed `5fbac40`) — recorded here, not restated as mine

**No candidate is adopted as the DAC definition.** The evidence supports one statement — the
candidates differ only in false-alarm rate (44.57 / 8.70 / 4.48 %) — and a term the reward depends
on cannot be defined on that alone, because **the degenerate optimum, a rule that never fires,
scores best on every measurement this corpus can make**.

⭐ **P2 may be used PROVISIONALLY as a LOW-NOISE PENALTY, explicitly not as a correctness
criterion.** A term false-firing on 4.48 % injects roughly ten times less label noise than one at
44.57 %. ⛔ That is a statement about **noise**, not **correctness**, and every claim resting on DAC
must say so in those words. The fork at the end of the previous section is **escalated to the PI
unchosen**, both branches as written. The register now carries the fact underneath it: on this
corpus a DAC penalty measures **map and read-out error, not driving error**, because in 736 windows
the ego never mounted anything.

## Pre-registered reading of the sweep — written BEFORE the labels exist

The Master Mind will label the 58 sheets blind and land the labels before the key is released.
`code/score_sweep.py` is written and **self-tested on synthetic labels now**
(`raw/score_sweep_selftest.json`) — a scorer first run at the moment the labels land is a
liability, so its branches are exercised in advance: it reports 55 clusters, keeps the mixed-in
three apart, treats `cannot-tell` as its own cell, and **refuses** a missing label or one outside
the rubric.

**The unit is the CLIP: n = 55, not 58 sheets.** The script asserts one clip per sweep sheet and
refuses to report a cluster count it cannot stand behind.

| outcome | what it means, committed now |
|---|---|
| **k = 0 over-boundary** | the sweep establishes a one-sided 95 % upper bound of **5.3 % of clips** on departures where no rule fires. ⛔ It does **not** establish that none exist. All three candidates remain unfalsified on misses, and the ruling's false-alarm-only reading stands unchanged |
| **k ≥ 1** | ⛔ the **first false-pass evidence this corpus can give**. A departure where NO rule fires falsifies the silence of V0, P1 **and** P2 alike — none of them fires in stratum D — and P2's provisional low-noise status must be re-read, because a rule that fires rarely *and* misses real departures is not merely quiet. With k = 1 the rate is 1.8 % of clips [UB 8.3 %] |
| **many `cannot-tell`** | a statement about what these sheets can show, **not** about the map's 0.5 m grid. ⚠️ And the adjudicator's tie-break leans toward on-surface, so a *low* count is unsurprising and is not evidence that the images are sufficient |

**The three mixed-in candidates are scored separately and never folded in** — two route-1 shock
events and one route-2 widest-margin window are a different population from a clip-first draw. ⛔ An
`over-boundary` among them would mean route 1 or 2 found a real event after all, and would revive
the ranking question the ruling closed on this corpus.

⛔ **Nothing is scored until the labels land.** The key stays off-repo until then; its sha256 is
already published in `raw/sweep_pack.json`.

## Appended 2026-09-21 — the sweep, scored: **k = 0**, and the pre-registered reading governs

Labels landed `3fb47e1` (`adjudication/LABELS_SWEEP_MM.csv`, blob `32d6b4260d`) BEFORE the key was
released, as committed. Scored with `code/score_sweep.py`, whose branches were exercised on
synthetic labels a turn earlier.

| population | unit | on-surface | over-boundary | cannot-tell |
|---|---|---|---|---|
| **the sweep** — stratum D, one window per clip | **n = 55 CLUSTERS** (not 58 sheets) | 54 | **0** | 1 |
| the three mixed-in, **never folded in** | 3 windows | 3 | **0** | 0 |

**k = 0 ⇒ the pre-registered reading applies exactly as written:** the sweep establishes a
one-sided 95 % upper bound of **5.3 % of clips** on departures where no rule fires, and ⛔ it does
**not** establish that none exist. All three candidates stay unfalsified on misses; the `aaf0879`
ruling is unchanged.

⭐ **The three mixed-in candidates all read `on-surface`** — the two route-1 vertical-shock events
and the route-2 widest-margin window. The ranking question the ruling closed stays closed: neither
physical route turned up a real departure, even under a blind eye that did not know why those
sheets were in the pack.

⚠️ **The tie-break caveat, carried verbatim:** *one cannot-tell in 58 is a property of the
adjudicator's rule, not evidence that the camera suffices* — judge whenever the surface is visible
in ANY panel, reserve cannot-tell for when it is visible in NONE.

**The combined picture: 118 windows — 60 disagreement-enriched, 58 swept by clip — produced ZERO
`over-boundary` labels.** Every firing of every candidate seen so far is a false alarm, and no rule
has yet been caught missing anything. ⛔ That is not evidence the rules are safe. It is evidence
this corpus cannot test them for missing.

### The BORDERLINE count, resolved exactly rather than reported as a mismatch

The CSV flags **7** rows (S10, S18, S28, S31, S36, S51, S56). The Master Mind's note discusses
**8**, and the eighth is **S53** — the `cannot-tell` row, treated as borderline in prose but
carrying no flag in its note. Both numbers are right about different sets; the PI's spot-check
needs the union.

⭐ Their *"five of the eight are tight turns whose inside corner is a kerb or island"* **checks out
against the file**: S10, S18, S28, S31, S51 — and every one of those five names the declared
**no-depth-test** artifact doing real work, the far arc drawing *over* the island because it lies
beyond it. Those five are the priority rows for a second adjudicator.

### S53 is not a one-off: the sheets' own evidence gap, measured

`code/footprint_visibility.py`, geometry only, over **both** packs (118 windows):

| | |
|---|---|
| sheets with **no corner visible in any panel** — no overlay at all | **2 of 118 (1.7 %)**: **S53** and **W46** |
| sheets with no **complete** footprint (the box never closes, though corners show) | **17 of 118 (14.4 %)**; median 33 ticks still show ≥ 1 corner |
| S53 specifically | the nearest footprint projects to row **436.5** against an image height of **416** — **20 px below the bottom edge**; path length 6.35 m |
| median ticks with a complete footprint | **31 of 41** |

⭐ **The two zero-overlay sheets are exactly the two rows that caused a counting anomaly in their
own packs** — W46 was named in prose but never flagged in pack 1; S53 is the lone cannot-tell in
pack 2. A 1.7 % failure has now produced an anomaly in **both** packs, which is what a systematic
gap looks like when the sample is small.

⇒ **Fix for the next pack, cheap and rule-independent:** the renderer already knows this from the
geometry, so a sheet whose footprint never enters the frame should SAY so on its face. Nothing
about that leaks: it depends on the camera and the path, never on a map or a rule.

### Rubric gap: a car park is neither carriageway nor kerb

S53 is a stationary manoeuvre in a snow-covered **car park**. The rubric names pavement, verge,
island and kerb as non-roadway and says nothing about a surface that is **trafficable but not
carriageway**, so both halves of the tie-break fail at once. `RUBRIC_v2_PROPOSAL.md` adds that cell
and leaves the definitional question where it belongs — whether the drivable-surface map is meant
to include car parks changes the label, and that is the PI's to rule, not mine to assume.

⚠️ It is also a warning about the corpus: if stationary car-park manoeuvres are common, a DAC term
will be scoring them.
