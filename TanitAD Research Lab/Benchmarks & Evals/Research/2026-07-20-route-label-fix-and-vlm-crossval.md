# Route labels: the v2 defect, the v2.1 fix, and what a VLM says about both

*2026-07-20 · agent run · code STAGED not pushed · corpus: PhysicalAI-AV val, 80 episodes (~199 frames each)*

## TL;DR

1. **The strategic route label was mostly not a label.** v2 needed 15 s of future and looked ahead 25 s;
   the clips are ~20 s. 74 % of windows fell through the guard — and the guard **returned
   `ROUTE_STRAIGHT`**. "I cannot judge this" and "the road goes straight" were the same emitted class.
2. **v2.1 fixes it additively.** Coverage **26.0 % → 81.9 %**; genuine turns the strategic head can never
   learn **63.1 % → 8.9 %**. v2 stays byte-identical and callable, so shipped runs remain reproducible.
3. **The VLM (Cosmos-Reason2-8B, future frames only) is a good route-event DETECTOR and a useless
   direction reader.** IS-A-TURN agreement with v2.1 **89.3 %** — ⚠️ **RETRACTED, see the correction in
   §"The finding that matters"; the quotable figure is turn RECALL ~78 % (77–81 % across three
   independent measurements)**; left/right agreement on the turns they
   both flag **52.7 %**, i.e. chance, with a 2.24:1 left prior against the road's own 0.90:1. Use it to
   find and mask events, never as a direction label.
4. **18 overlay videos** show all four readings per frame (old / new / VLM / model) — a wrong label is
   visible at a glance.

---

## 1. The defect, measured

`stack/scripts/route_label_audit.py` (promoted from a pod-only script into the repo as the standing
regression harness), 80 val episodes, stride 20 → 800 windows.

| | v2 (shipped) | **v2.1** | v2.1 strict¹ |
|---|---|---|---|
| coverage (`valid`) | 26.0 % | **81.9 %** | 81.6 % |
| `unknown` emitted (honest gap) | 0.0 % | **18.1 %** | 18.4 % |
| turn labels | 12.8 % | **32.4 %** | 32.0 % |
| straight + valid | 13.2 % | 49.5 % | 49.6 % |

¹ `use_net_dyaw=False` — v2's strict junction-only semantics under the new horizon.

**Genuine turns** (referee: \|cumulative net heading\| ≥ 30° over the available future)

| | real turns | false-straight | masked | **unlearnable** |
|---|---|---|---|---|
| v2 | 203 | 3 (1.5 %) | 125 (61.6 %) | **63.1 %** |
| **v2.1** | 203 | 3 (1.5 %) | 15 (7.4 %) | **8.9 %** |
| v2.1 strict | 203 | 4 (2.0 %) | 17 (8.4 %) | 10.3 % |

v2's damage was almost entirely **masking**, not mislabeling — but the masked windows still *carried*
`ROUTE_STRAIGHT` in the emitted class, so any consumer reading the label file rather than the mask
(and the exports do) saw 74 % straight.

**Referee sensitivity** — the residual is a band, not a bug to tune away. The labeler's own net-heading
rule fires at 45°, so false-straights can only survive below it:

| referee ≥ | real | v2 false | v2.1 false | v2.1 masked |
|---|---|---|---|---|
| 20° | 250 | 3 | 3 | 21 |
| 30° | 203 | 3 | 3 | 15 |
| **45°** | 173 | 1 | **0** | 6 |
| 60° | 150 | 0 | 0 | 4 |

The 3 residual false-straights are all `ep_00069` windows in the 33–44° band — below the declared rule.

## 2. What v2.1 changes (`stack/scripts/refb_labels.py`, additive)

| Rule | Change | Why |
|---|---|---|
| R1 adaptive horizon | gate on **arc length travelled** (`MIN_ARC_ROUTE_M=20 m`), not step count | curvature is already per-metre, so tightness needs no rescaling; only *transience* does |
| R1b arc-anchored transience | concentration measured over `CONC_ARC_M=60 m` of road, and the gate is applied **only** above `TRANSIENCE_MIN_ARC_M=150 m` | a fixed 5 s sub-window is a different stretch of road at 5 vs 30 m/s, and its share → 1 as the horizon shrinks |
| R2 never straight | new sentinel `ROUTE_UNKNOWN = 3`, deliberately **outside** the 3-class CE range | an unmasked loss now raises instead of training a wrong class; `n_route` unchanged |
| R3 net heading in the decision | TURN iff (tight ∧ transient) **or** \|net_dyaw\| ≥ 45° | fixes the 479 m / 48° `ep_00069` case v2 called straight+valid |
| R4 graded target | `mean_curv = net_dyaw / arc` (+ `graded_route = tanh(·)`) | threshold-free and horizon-invariant; defined on windows the discrete label must mask |

**Bug caught while writing the tests:** v2's `net_dyaw` is the *wrapped endpoint difference*, which folds a
270° roundabout into −90° and flips its sign. v2.1 sums per-step wrapped deltas (`net_dyaw` cumulative,
`net_dyaw_wrapped` kept for cross-checking), so the graded target recovers 1/R to within 12 % from 5 s to
25 s of future at 6–30 m/s.

Tests: `stack/tests/test_refb_labels_v21.py`, 19 new. Suite **560 passed, 2 skipped** (541 before).

## 3. VLM cross-validation — Cosmos-Reason2-8B, two passes

`nvidia/Cosmos-Reason2-8B` (Sayed accepted the licence; Qwen3-VL architecture, BF16 ≈ 17 GB on the A40),
pod3. **400 windows, 80 episodes, 0 enum violations, 4.2 s/window.**

- **Pass A** = history frames + past-only ego-motion block + **future frames only**. No numeric future
  track. This is the only pass allowed into the agreement statistics; the cross-validator *raises* if a
  Pass B record tries to enter.
- **Pass B** = the same plus a plain-language reading of the numeric future track, for the rich
  scenario/tactical/observation/CoC schema. Its ROUTE is an echo of ours by construction and is excluded.

**Pass B parrots, exactly as predicted.** On `ep_00069` Pass A (frames only) says `straight`; Pass B, handed
"turns right 48° …", says `right` with 0.99 confidence. That is the whole reason for the split.

### Agreement

| | agreement | n |
|---|---|---|
| v2 **as emitted** (what a consumer reads) | 64.8 % | 400 |
| v2 where v2 says valid | 65.5 % | 139 |
| **v2.1 where v2.1 says valid** | **73.7 %** | 338 |

### The finding that matters: detection ≠ direction

| | agreement | n |
|---|---|---|
| **IS-A-TURN** (turn vs straight) | ⚠️ **89.3 %** — see correction below | 338 |
| **DIRECTION**, both call it a turn | **52.7 %** | 112 |
| direction on tight turns (R ≤ 60 m) | 52.7 % | 112 |

> ⚠️ **CORRECTION 2026-07-21 — the 89.3 % must not be quoted as detector quality. Two problems.**
>
> **(1) It is AGREEMENT, not RECALL.** Agreement counts every window where the VLM and v2.1 give the
> same turn/straight verdict — *including all the straight-straight matches*. On a corpus that is
> **~74 % straight**, a model that answered "straight" every single time would score ~74 % agreement
> while detecting **zero** turns. Quoting it as evidence of a "good event detector"
> (`V35_DESIGN.md:86`, `:406`) overstates it. The decision-relevant statistic is **recall on actual
> turns**, measured fresh in the Reason1-vs-Reason2 head-to-head: **76.8 % (63/82)**.
>
> **(2) The 89.3 % itself does not reproduce.** Re-scoring pod3's raw `vlm_crossval.json` rows gives
> **80.6 %** under every denominator tried. The 52.7 % direction figure reproduces to four digits from
> the *same rows with the same code*, so this is not a scoring-convention artefact. Per CLAUDE.md the
> raw rows win.
>
> **Quotable set:** direction is chance (52.7 % banked, **57.1 %** replicated fresh on a different val
> build, CI [0.400, 0.745] — contains 0.5); turn **recall 76.8 %** (63/82, fresh build) and **80.6 %**
> (112/139, pod3 banked build).
> Source: `Data Engineering/Research/2026-07-20-cosmos-reason1-vs-reason2-headtohead.md`.
>
> **AMENDED 2026-07-21 — two fixes to this block.** (a) The 80.6 % above was written as "turn
> *agreement*"; the head-to-head's own §6 table lists it as **turn-detection RECALL (112/139)**, in the
> same row as the 76.8 % (63/82). Both are recall — corrected here so the distinction this correction
> exists to make is not itself blurred. (b) A **third** independent measurement has since landed —
> **78.6 %** (enum-order probe, 200 held-out windows, 40 episodes). The three cluster at **77–81 %**;
> quote **~78 %**. Source:
> `Data Engineering/Research/2026-07-21-cosmos-reason2-production-semantic-labeling.md` §1.

53 of 112 shared turns are **opposite-direction** calls. 50 % is chance. The VLM's left:right prior is
**2.24** against the corpus's **0.90** — a model prior, not a road.

The disagreements are concentrated on the turns where the kinematics are *least* ambiguous (R = 6–13 m,
net heading 100–225°), at 0.99 confidence, with fluent evidence text — and the VLM flips direction
*within the same clip* (`ep_00015` t=40 "right", t=80 "left").

**Adjudicated against the camera, twice, on clips where the two sources disagree** — this matters, because
if the kinematics had the sign wrong the entire Part-1 conclusion would invert:

- `ep_00015` f080 (`still_ep15_f080_…png`): kin `LEFT`, VLM `left`, and the projected GT path curves
  visibly **left** into the T-junction the VLM itself describes. Three-way agreement including the image.
- `ep_00012` f110 (`still_ep12_f110_kinRIGHT-correct_VLM-straight-wrong.png`): kin `RIGHT` (net −45°),
  VLM **`straight` at 0.99** — and the green GT path curves visibly **right** off the bottom-right of the
  frame, with the BEV inset agreeing. The kinematics are right; the VLM is wrong.

**The kinematics own direction; the VLM does not.**

Where it *does* earn its keep: **93 windows v2 could not judge at all, the VLM called a turn** — v2.1's
adaptive read independently agrees on 38 (41 %).

### Pass B — the rich schema (the pilot's structure failure is fixed)
Pass B returns the full 5-section object (`SCENARIO` / `STRATEGIC` / `TACTICAL` / `OBSERVATIONS` / `COC`)
with every categorical field selected from an explicit enum shipped in the prompt. The 48-clip pilot got a
structured CoC on **3/48 (6 %)**; with the explicit JSON schema and a flat CoC object this run is at
**6/7 complete-structure** on the first samples — the remaining failure mode is output truncation at
`max_new_tokens=2200`, which yields an empty section list rather than prose (loud, not silent).
VTARGET/HEADWAY are **never asked for** (the pilot fabricated VTARGET band edges on 48 %); the VLM supplies
`VSOURCE`, the sign it actually read, and the qualitative headway bucket, and kinematics own the numbers.
Throughput 28 s/window vs Pass A's 4.2 s. **Pass B was still running at hand-off — see manifest.**

### Recommendation for v3 labels
- ROUTE **direction**: kinematic v2.1, provenance `kinematic`. Not VLM.
- ROUTE **event detection / mask**: VLM Pass A is a usable second opinion at **turn recall ~78 %**
  (77–81 % across three independent measurements — the **89.3 %** originally quoted here is retracted, see
  the correction in §"The finding that matters" and
  `Data Engineering/Research/2026-07-21-cosmos-reason2-production-semantic-labeling.md` §1) — good for
  flagging windows worth a human look, and for the `gray_zone` band the kinematics refuse.
- Before trusting VLM direction at all it needs: native-resolution frames (we feed a 256 px cache upscaled
  to 448), denser future sampling, and an explicit direction-calibration eval. **UNVERIFIED** whether any
  of that fixes it.

## 4. Validation videos

18 clips, standing viz standard (camera projection + metric BEV inset + HUD), 4 readings stacked per
frame — OLD v2 / NEW v2.1 / VLM Pass A + its evidence text / REF-C's own tactical + route argmax and ADE.
Rows colour by failure mode: **red** = says straight through a ≥30° turn, **amber** = disagrees with v2.1.

Covered: the 479 m drift (`ep_00069`), late-clip windows v2 could not label (every clip beyond f≈49 —
Sayed's f113 screenshot case), tight junction turns (R = 6–13 m), wide drifts, high-speed straights, and a
near-stationary clip that lands in the honest `no_arc` → `UNKNOWN` branch.

**What the videos show at a glance** (e.g. `ep_00015` f080, a T-junction left turn):
`OLD route v2: straight [MASKED→emitted straight]` in red, `NEW route v2.1: LEFT [valid·tight_transient]
net +161° arc 81 m` in green, `VLM: left conf 0.98` in green — **and the model's own route argmax:
`straight`**. REF-C, trained on the v2 labels, predicts "straight" while turning left through a junction.
That is the poisoned prior, on screen.

**Per-clip counters** (printed per video, stride 2): on `ep_00069` v2 shows `old_masked=64` and v2.1
`new_false=30` vs v2's `old_false=25` — v2.1 is *worse* on that one clip at frame level, because its higher
coverage commits `road_following` in the 30–45° band that v2 simply masked. Across all 800 audit windows
both sit at 3 false-straights; the clip-level number is that band, concentrated.

## 4b. Artifacts and how to reproduce

| What | Where |
|---|---|
| labeler v2.1 | `stack/scripts/refb_labels.py` (v2 untouched below it) |
| regression harness | `stack/scripts/route_label_audit.py` — `python scripts/route_label_audit.py --stride 20` |
| VLM labeler | `stack/scripts/vlm_route_labels.py --passes A\|B\|AB` |
| cross-validator | `stack/scripts/vlm_kin_crossval.py` |
| video renderer | `taniteval/taniteval/label_overlay.py` |
| tests | `stack/tests/test_refb_labels_v21.py` (19) |
| 18 videos | `Research/videos-2026-07-20/labels_ep_*.mp4` — **gitignored** (`*.mp4`), present on disk only |
| 5 stills + 2 result JSONs + Pass-A/B label archive | same dir, **committed** |

Pod-side (pod3): weights `HF_HOME=/root/hf` (container disk — `/workspace` is at ~431 G of its ~466 G
MooseFS quota, the 8B would not have fit there); run wrapper `/root/vlmrun.sh` (reads the HF token from
`Keys.txt`, never echoed); labels `/workspace/vlm_passA` (400, complete) and `/workspace/vlm_passB`
(in flight at hand-off). Re-archive Pass B when it finishes:
`ssh tanitad-pod3 "cd /workspace && tar -czf /tmp/vlm_passB.tgz vlm_passB"`.

**Pod hygiene note:** installing Qwen3-VL support upgraded `/workspace/venv` to **transformers 5.14.1** on
pod3. Nothing else was running there, and the label/render paths were exercised after the upgrade, but any
future pod3 job that pinned an older transformers should re-verify. **UNVERIFIED** for training paths.

## 5. Honest gaps

- The audit referee (\|net heading\|) is the ego track itself; it cannot separate a junction turn from a
  long road sweep. Read `false_straight` with the radius column.
- v2.1's 45° net-heading rule **contradicts** v2's design intent (a wide sweep is road-following). Pass A
  sides with v2 on `ep_00069` (`straight`). The rule is a named, switchable branch (`use_net_dyaw`) and the
  audit reports both readings; **the semantics are a decision for Sayed, not something the data settles.**
- Coverage is 81.9 %, not ~100 %: 9.9 % `no_arc` (clip tail / stationary — genuinely no future) and 8.3 %
  `gray_zone` (the honest ambiguity band). Both are now *labelled as unknown* rather than as straight.
- v2.1 is **not yet wired into any trainer.** `config.v2_labels` still selects v2. Wiring is a separate
  change with its own gate.
