# TanitAD — Program Report, 2026-08-12 05:54Z

**Covering the overnight campaign 2026-08-11 12:00Z → 2026-08-12 05:54Z.**
52 commits, all on `claude/tanitad-resumption-handoff-92zx39`, pushed. Every number below
carries its **evidence class**, its **eval tier**, and its **estimator**.

---

## 0. The one-paragraph version

The v5.8f campaign is **closed**. Its headline is not the ADE we set out to move — it is
that the **T1 action-closed loop diverges, a hold-action control beats it 22×, and the
divergence is ~99 % longitudinal: a runaway acceleration with a known sign**. The stage-A
repair is a large, separated win (−14.61 m paired) and still lands far short of driving
competence. Two independent instruments now agree the model **cannot see the vehicle in
front of it**, and last night's LF0 closed the cheap escape route: there is no
zero-training fix. That is the single most decision-relevant fact for v6, and it arrived
because the four-metric-family rule and the tier doctrine were enforced — ADE alone would
have shown a big number and hidden the mechanism entirely.

---

## 1. Fleet — fresh measurements (05:54Z)

| resource | state | note |
|---|---|---|
| **pod4** (A40, 46 068 MiB) | **IDLE** — 0 MiB, 0 % util | LF0 finished; torch repaired to 2.8.0+cu128, CUDA True, conv2d verified |
| **pod5** (A40) | **IDLE** — no trainer, no eval | T1 + four-families complete; `ff_rescore.log` + 8 JSONs banked |
| **local** | clean tree, all work pushed | 52 commits since last report |
| **workflows / agents** | **0 running** | 4 subagents completed and reported overnight |

⚠️ **Both pods are idle and burning money.** Nothing is blocked on compute — the next
GPU action is a PI decision (§5, D1/D2).

---

## 2. What we achieved — measured results

### 2.1 ⭐ T1 pseudo-closed-loop — the headline
**MEASURED 2026-08-11 23:27Z / analysed 08-12 00:10Z · TIER T1 = PRIMARY · 6 844 windows /
40 val episodes, stride 1 · episode-cluster bootstrap · `overlapping_holdout_se` used
nowhere.** Registry §1.14.

| arm | surface | tier | ADE dense (m) | LON along (m) | LAT cross (m) |
|---|---|---|---|---|---|
| `v5f-30k` | `cl` | **T1** | **23.9837** [21.442, 26.347] | 23.8965 | 0.9993 |
| `v5f-30k` | `ol` | T0 | 0.9397 [0.8162, 1.0679] | 0.8762 | 0.1947 |
| `v5f-30k` | `ha` | **T1** | 0.9597 [0.8361, 1.0879] | 0.8901 | 0.2072 |
| `stage-a-repaired` | `cl` | **T1** | **9.3697** [6.6822, 12.2576] | 9.2655 | 0.7446 |
| `stage-a-repaired` | `ol` | T0 | 0.3659 [0.2926, 0.4521] | 0.2990 | 0.1534 |
| `stage-a-repaired` | `ha` | **T1** | 0.4246 [0.3500, 0.5132] | 0.3487 | 0.1689 |

**Finding 1 — the repair wins decisively and separated.** Paired, same windows:
`cl` ADE **−14.6139 [−16.9319, −12.2010]**, `cl` LON speed MAE **−17.2064**, `ol` −0.5739,
`ha` −0.5351; all `p_delta_gt0` = 0.0. The repair targeted action-response gain
(0.27 → 0.971/0.966) and improved **exactly the axis it targeted**.

**Finding 2 — ⛔ the closed loop diverges and a hold-action control beats it 22×**
(0.4246 vs 9.3697 repaired; 0.9597 vs 23.9837 v5f). Within-arm paired `cl − ol`:
**+9.0039 [6.3659, 11.8487]** and **+23.0439 [20.5613, 25.3884]**, both separated.
⇒ **No closed-loop driving competence may be claimed for either arm.** T0 0.3659 vs T1
9.3697 on the same checkpoint and windows is a **25× gap** — the strongest evidence yet for
the tier doctrine, and precisely the failure it exists to expose.

### 2.2 ⭐ Four-family rescore — the mechanism
**MEASURED 08-12 00:55Z · 6 arms, 15 paired contrasts, `FF_EXIT=0`, same grid.**
Binding rule now **satisfied** on the T1 rows (`rule_satisfied: true`; only `strategic`
unavailable).

- **LONGITUDINAL — a runaway acceleration with a sign.** Speed MAE 9.7291 with **speed bias
  +9.3892** (bias ≈ **96 %** of the error ⇒ systematic over-speed, not scatter); along-track
  bias +9.0407, final bias **+18.5801 m**; accel MAE **19.0948 m/s² (>1.9 g — physically
  impossible)**; **ego progress ratio 1.7279** — the arm drives **1.73× the distance the
  human did**.
- **LATERAL — healthy, and not the problem.** Heading 3.8776°, yaw-rate 4.9188 °/s,
  curvature 0.0186 1/m, cross-track 0.7446 m. All four members present.
- **TACTICAL — longitudinal decision-making is AT CHANCE.** Lateral κ **0.3795** vs
  longitudinal κ **0.0405**; collapsed 5-way κ 0.1404 sits *between* and reports neither —
  **the direct measurement of the lat/lon-mixing softmax defect**, visible only because the
  family is reported factored.
- **TACTICAL goal-setting — direction right, distance wrong.** Goal bearing MAE 4.8098° vs
  goal range ratio **1.7584**, long-bias +18.5801 m vs lat-bias −1.2061 m. ⇒ **the model
  predicts the correct DIRECTION of travel and badly overestimates the DISTANCE.**
  ⚠️ Earlier phrasing ("knows where to go") was too loose and implied road understanding.
  This is bearing accuracy toward the *human's future position* — the model has **no
  representation of road or lane structure available to it at all** (§7b).
- **STRATEGIC — `n/a` with reason and n = 6 844** (no map, lane graph, or route signal in
  PhysicalAI-AV), per clause 5.

### 2.3 ⛔ LF0 — the decoded BEV does not read the lead gap
**MEASURED 08-12 08:30Z · zero-parameter geometric read · τ = 0.7 INHERITED from the P8
gate · 900 windows scanned, 129 labelled.**

**Reader sanity gate PASSED first** (GT reads rank-agree: `gt@1.0` ρ 1.0 / MAE 0.0144 m;
`gt@2.0` ρ 0.9596 / MAE 0.2829 m) — which is what makes this a statement about the latent.

| arm (1.5 m corridor) | R² | ρ | MAE (m) | n | **censored on labelled** |
|---|---|---|---|---|---|
| `gt@1.5` (truth) | 1.0 | 1.0 | 0.0 | 129 | 0 % |
| `enc@1.5` | −21.00 | 0.3826 | 26.85 | 24 | ⛔ **81.4 %** |
| `pred@1.5` | −16.12 | −0.7091 | 42.65 | 10 | ⛔ **92.3 %** |

**The finding is the censoring rate, not the R².** In 81.4 % / 92.3 % of windows where the
ground truth has a lead vehicle in the ego corridor, the decoded BEV shows **an empty lane**.
That rests on all 129 labelled windows. ⚠️ The R²/ρ sit on n = 24 and n = 10 and are **not
decision-grade**; `pred`'s ρ = −0.71 is **noise on n = 10, not an inverted signal**.

**⇒ RC1 REFUTED — there is no zero-training fix.** This is the **second independent test,
different instrument class** (zero-parameter geometric read vs P1's fitted probe: linear
R²(enc) ≤ 0, all transforms failed, MLP ceiling −0.334), same verdict. It is the concrete
consequence of P8's own stamped ~0.02 absolute-IoU limitation, **not a contradiction of it**.

### 2.4 Selection — W7 closed out
- **W7-FULL** (selector-free, 256/256 candidates, repaired trunk, refit head, fan oracle
  0.1273): **FAIL 3.3348** vs gate 0.4505. Mechanism identified: **winner's curse** — the
  argmin's error-rank is **132 of 256, the median**; top-m mean error flat at ~5.32 (the
  fan's own mean).
- **W7-PROG** (the one untried lever — `--w-prog` had been 0.0 in every run): pre-registered
  **PARTIAL**. Error-rank falls monotonically 132.3 → 130.31 → 126.69, but that is 2.2 % of
  the fan and gate ADE worsens. ⇒ **self-consistency selection retired as a headline route**;
  a goal-conditioned cost term is the pre-registered successor.

### 2.5 Representation — H-COTRAIN and SIGReg
- **H-COTRAIN REJECTED within the measured range.** Every probed physical variable became
  *more* decodable (curvature 0.213→0.513 enc, 0.225→0.704 pred; yaw 0.583→0.869),
  participation ratio **expanded 53 %** (4.53→6.94), P1 battery FAIL→PASS at 20k. Neither
  pre-registered CONFIRM condition fired. ⚠️ Scope: all milestones sit at/past the ramp, so
  the lowest reference is λ = 0.5, not 0 — the clean control is the mode-0 arm (= v6's S-W).
- **SIGReg VALIDATED** — retention **1.53×** against a ≥0.8× gate (you asked for this
  explicitly).

### 2.6 Other banked verdicts
- **P8 attempt-2 GATE PASS** — retention **0.932** at k=10, τ*=0.7 chosen on the encoded arm
  (gate can only get harder), **74×** lift over attempt-1. P4 permanence rides it: occluded
  recall ≥ visible (0.2178 vs 0.1881 enc). Limitation stamped: absolute IoU ~0.02 ⇒ the
  **ratio** is the quotable claim.
- **I4a — imagination is load-bearing**: intact 0.4011 / shuffled 1.2492 / **zeroed 7.6493**
  (19× collapse). zero ≫ shuffle ≫ intact ⇒ read as **content**, not a bias term.
- **W4r GATE PASS** — fan oracle 0.1273 on the repaired trunk, violations 0.
- **Consumer invalidation** — frozen selector 0.7933 → **4.4159** on repaired features ⇒
  v5.8f ships the **frozen-trunk** assembly (0.4815), and **you cannot repair a trunk and
  keep its planner**. That sentence *is* the staged-training argument for v6.

### 2.7 Shipped artifacts
- **v5.8f HF release PUBLISHED** — manifest-defined, md5 per file, MANIFEST uploaded last,
  hard refusal on a missing REQUIRED artifact.
- **Paper v0.9 → v1.0** (+882 lines): winner's-curse Gaussian-copula treatment, H-COTRAIN
  retraction in the paper's own voice, label-free admissibility algebra, 4-brain formalism,
  tier + four-family doctrine as method sections, Figure 3.
- **I1c belief reel** — camera │ decode(ẑ) │ belief ∩ truth, at the run's own τ*.
- **v6 architecture figure** (paper-quality SVG+PNG).

### 2.8 v6 — ready to train
**4 895 lines, 80 CPU tests green** under `-W error::UserWarning`.
`stack/tanitad/models/v6.py` · `scripts/train_v6_staged.py` · `tests/test_v6_staged.py` ·
`V6_TRAINER_DESIGN.md`. Params **87.89 M** shared-encoder / **120.74 M** per-layer;
E-ENC matched pair 118.11 M (**2.2 % gap** ⇒ a real matched comparison).
`assert_isolation()` is a real autograd probe, not a config assertion.
**S-W verdict: GO on code, blocked only on provisioning.**

---

## 3. Agent updates & knowledge transfer

| agent | produced | state |
|---|---|---|
| **v6 staged trainer** | `v6.py`, `train_v6_staged.py`, 80 tests, design note | ✅ **integrated** (`2b8d09e`) |
| **T1 adapter** | `--v2-val-cache` / `--grounding-readout` / `--window-stride`, `t1_summary.py`, 28 tests | ✅ **integrated** — produced §2.1 |
| **Paper update** | `TANITAD_PAPER.md` v1.0, Figure 3, Figure 2 fix | ✅ **integrated** |
| **v6 GO package** | `V6_GO_PACKAGE.md`, `PI_DECISIONS_2026-08-12.md`, BACKLOG §F | ✅ **integrated** |
| **Four-families** | `four_families.py` TACTICAL-from-trajectory, `ff_rescore.py`, 41 tests | ✅ **integrated** — produced §2.2 |

**Three agent findings that changed the plan:**
1. `plan.max_horizon` is 20 — inheriting it would have made the 6 s horizon *structurally
   untrainable*, failing while **looking like a corpus limitation**. v6 derives its own, and
   the loss is smaller than feared: **S-W and S-S are unchanged at 94 windows**; only
   S-T/S-J pay the −42.6 %.
2. **D6 (new)** — S-S's gate cannot pass before PH2, because its `required` tuple is
   `("STRATEGIC_family",)` and S2 is unwired ⇒ terminates at `pass: null`, and S-J then
   refuses.
3. `t1_eval.py` emitted TACTICAL `UNAVAILABLE` on **every T1 row the programme has ever
   produced**. One line (`tactical_from_traj=True, tier=t`) closed it at source.

---

## 4. Program position — the four edges

| edge | position | evidence grade |
|---|---|---|
| **Planning** | ⛔ **Weakest, and now precisely characterised.** Closed loop diverges; hold-action beats it 22×; longitudinal decisions at chance (κ 0.0405); selection-by-self-consistency retired | **MEASURED, decision-grade** (T1, 6 844 windows, episode-cluster bootstrap) |
| **Efficiency** | Sub-300 M held: v6 at 87.89 M of a 300 M budget | **MEASURED** (param count, dry-run) |
| **Safety / self-knowledge** | Mixed. P8 retention 0.932 and P4 permanence are real; but the model **cannot see the lead vehicle** (2 independent tests) and over-accelerates at 1.9 g | **MEASURED**; LF0 R²/ρ **not** decision-grade (n=10/24) — the **censoring rate** is |
| **Data efficiency** | Label-free trunk held throughout; SIGReg validated 1.53×; H-COTRAIN's feared collapse **refuted** | **MEASURED**, with the λ ≥ 0.5 scope caveat stamped |

**Against the Master Plan:** the hierarchy thesis is intact but **unproven at the strategic
level** — STRATEGIC has never been measured because the corpus cannot support it, and the
VLM PH0→PH2 pipeline is the programme's only route to it. That is why it was resequenced
ahead of v6, and it is now blocked on one decision (D7).

---

## 5. ⛔ Decisions required from you — with defaults

| # | question | recommended default | blocks |
|---|---|---|---|
| **D1** | S-W A40-hours before the first gate | **Authorise ≤12 A40-h to a step-500 re-cost**, four branches pre-registered now (<21 h proceed+re-cost · 21–35 proceed · 35–50 apply `--o5-k 10` first · **>50 STOP, that's a defect not a cost**) | ⛔ v6 launch |
| **D2** | which pod runs S-W | **Provision a dedicated A40.** Both current pods are free but have tracks; ship files md5-verified, never git | ⛔ v6 launch |
| **D7** | ⚠️ **CORRECTED 2026-08-12 — see §7a. Probably NOT a decision.** Only the 27B OOM claim was true; the 9B and gemma-4 both fail on the same **`swscaler` video-decode error**, which is environmental | **Do not pick a model yet.** Fix the decode error first; the 9B we already have may be fine. Re-run the smoke, then decide | ⛔ PH1 → STRATEGIC |
| **D3** | W5/E-H1 6 s baseline before or parallel to v6 | **PARALLEL** — the precursor binds the *claim*, not the *launch* | no |
| **D4** | 6 s window loss vs cache rebuild | **Accept** — smaller than feared, S-W unaffected; deferrable to S-T | no |
| **D5** | E-ENC shared (87.89 M) vs per-layer (120.74 M) | **Shared for the first S-W**; matched pair as a short arm, not a second 175–290 h run | no |
| **D6** | S-S gate cannot pass pre-PH2 | **Pre-registered amendment**: promote `S1_ade_8_30s` to required; STRATEGIC `n/a` with reason + n | before S-S |

*Seven further questions were **decided by evidence and not escalated** (`--o5-k 20`,
stopgrad uplink, `--o4-alpha 1.0`, isolation ON, no S-J up front, staging rests on field
evidence, keep SIGReg) — each with its reopen trigger in `PI_DECISIONS_2026-08-12.md`.*

---

## 6. Ordered next steps

**Immediately unblocked (0-GPU, no decision needed):**
1. **Lead block for the T1 dense grid** (`build_lead_block.py`) → closes LONGITUDINAL
   distance-keeping, *the* half where the failure lives. The T1 dump now carries
   `eid`/`clip_index`, so the join is clean; previously it would have been positional.
2. **Goal-conditioned selection cost** — pre-register and implement, the successor W7-PROG's
   PARTIAL points to.
3. **Training-side longitudinal lever** — LF0 forces this. Candidates in `JEPA_PHYSICS_SURVEY`
   §3: interaction-weighted sampling, masked-latent objectives, dense near-field loss
   shaping. All label-free in the trunk, all gated on the same frozen P1 battery.
4. **Suite hygiene** — 23 standing `pytest` failures, none from v6 (onnx absent ×1, a Windows
   basename assertion ×2, a suite-order dependence ×20). These are a live hazard to the
   "green before any commit" invariant.

**On D1+D2:** launch S-W, read `step_s` at step 500, re-cost, apply the pre-registered branch.
**On D7:** PH1 50-clip run → vocabulary mapping → PH2 → STRATEGIC becomes measurable.

---

## 7. Incidents — honestly

| incident | cost | fix / rule |
|---|---|---|
| **I broke pod4's CUDA twice.** `uv pip install -U accelerate`, then `compressed-tensors`, each silently pulled **torch 2.13.0+cu130** onto a CUDA-12.8 driver via the dependency closure | ~3 rounds | Install torch-dependent packages `--no-deps`; reinstall torch from the pinned index **last**; verify with a real CUDA **conv2d**, not `import torch`. In CLAUDE.md |
| **I misdiagnosed the resulting `CUDNN_STATUS_NOT_INITIALIZED`** as cu13/cu12 shadowing and purged the cuDNN torch actually needed | +1 round | A symptom read as its own root cause — same class as the `df` / Thor `free` / cgroup traps. In CLAUDE.md |
| **T1 died in `analyze()` after both arms rolled to completion** (`from taniteval import selgap`) | ~0 GPU — recovered | `--analyze-only` over the banked dumps recovered every number with zero recompute. Durable fix = preflight import probe. In CLAUDE.md |
| **LF0 took 5 launches**, 4 of them my own wiring errors — the worst being a copied namespace that would have run to completion scoring against a *different geometry* | ~1 h wall-clock | Copy the working reference (`p8_bev_reel`), don't re-derive; signatures cross-checked programmatically |
| **The kept-positions trap** — `batch_rasters` returns *kept positions*; `bool(keep[0])` is `bool(0)` = **False**, discarding every valid raster | caught by the gate | **The sanity gate did its job**: LF0 returned INADMISSIBLE rather than a false confirmation of P1. Two tests pin it |
| **A polling monitor matched its own echoed command** and reported a failure that never happened | 2 rounds | Emit an opaque marker, parse that. In CLAUDE.md |
| **My whole-index commits swept a sibling agent's in-progress files** 5× | nothing lost | The known CLAUDE.md hazard; agent re-staged and md5-verified each time |

**Distance-keeping was deliberately NOT delivered.** The labels are staged on pod5, but the
T1 dumps carried no clip identity, so any join would have been **positional** — exactly what
`build_lead_block.py` warns silently attaches another clip's traffic to every window. I made
the dump carry `eid`/`clip_index` instead. A missing metric is a work item; a fabricated one
is worse.

---

## 7a. ⚠️ Correction — the VLM arms (raised by the PI, 2026-08-12)

I wrote "all three arms unusable (text-only / OOM / broken 4-bit)". **Only the OOM was true.**
Re-read from `ph0_smoke.json` rather than from my own summary:

| arm | ACTUAL current error |
|---|---|
| `Qwen3.5-9B` | `BlockingIOError: [Errno 11]` — **`[swscaler] Failed initializing scaling graph`** |
| `Qwen3.5-27B-FP8` | `OutOfMemoryError` — 43.23 GiB on a 44.43 GiB card ✅ genuinely too big |
| `gemma-4-31B-it-qat-w4a16-ct` | **the same swscaler error** |

"Text-only" for the 9B was simply wrong — it is a **video-decode resource failure**, not a model
capability limit. "Broken 4-bit" for gemma was **STALE**: that was the `compressed-tensors`
ImportError, which I fixed hours earlier with a `--no-deps` install. And a
"Resource temporarily unavailable" on a scaling-context allocation is thread/FD exhaustion —
**environmental**, the same family as the torch-113-threads trap. The videos themselves are
well-formed (1 MB mp4s from the bridge).

**Root cause of the error: I carried a stale characterisation from an earlier smoke run into a
PI decision request** instead of re-probing. That is the "absence found at ONE location is not
absence" rule, broken inside the document that exists to escalate decisions.

## 7b. ⚠️ Correction — the BEV contains no road structure (raised by the PI, 2026-08-12)

Asked whether the extracted BEV includes road / corridor / lane boundaries. **It does not.**
From source (`bev_raster.py:79-82`), the raster is built from `obstacle.offline`, whose full
measured enum is **10 classes, ALL DYNAMIC AGENTS**: automobile · heavy_truck · bus ·
other_vehicle · trailer · person · rider · stroller · animal · protruding_object. No lane
boundary, no road edge, no drivable area, no junction — consistent with the settled fact that
PhysicalAI-AV ships no map data.

Two of my own phrasings were therefore misleading and are corrected:
1. **The "ego corridor" in LF0 is not perceived.** It is a hand-defined ±1.5 m band (grid
   columns 29–34) — my assumption about where the ego lane is. Now drawn dashed and labelled
   "ASSUMED" in the figure, and stamped in the artifact's `_no_map` field.
2. **"Knows where to go"** is bearing accuracy toward the human's future position, not road
   understanding (§2.2).

⇒ This *sharpens* the longitudinal story: the model has **no road-structure representation at
all**, and the one dynamic-agent channel it does have does not survive the decode well enough
to locate the lead vehicle. **Two independent gaps that I had been treating as one.**
Figure: `Paper/figures/lf0_bev_panels.svg` (+ generator, reads the artifact at render time).

## 8. Bottom line

v5.8f is closed and released; v6 is built, tested and GO on code. The programme's central
open problem is now **stated precisely rather than as a scalar**: the world model
over-accelerates into a runaway because it cannot perceive the vehicle ahead, its
longitudinal decisions are at chance, and no read-off of the existing latent recovers the
lead gap. Lateral is healthy. The next lever must be **training-side**, and the staged v6
ladder — a world model made stable under its own actions *before* a planner is attached — is
now motivated by measurement rather than by architecture taste.

**Three decisions (D1, D2, D7) unblock everything else. Both pods are idle until then.**
