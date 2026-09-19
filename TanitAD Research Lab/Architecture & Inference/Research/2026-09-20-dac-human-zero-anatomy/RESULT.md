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
