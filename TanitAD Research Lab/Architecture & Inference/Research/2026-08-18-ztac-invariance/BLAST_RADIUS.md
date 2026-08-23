# BLAST RADIUS — every claim that assumes `z_tac` integrates a window

Companion inventory to `ZTAC_INVARIANCE.md`. Search was by **content**, not by hand-list, across
`Project Steering/`, `TanitAD Research Hub/`, `Paper/`, `stack/` docstrings and `README.md`.

## Evidence-class convention for this file

| tag | meaning |
|---|---|
| ⭐ **VERIFIED HERE** | I opened the file and read the line myself this session. |
| **SWEEP** | Located by a content sweep and reported with file:line. **Not individually re-opened by me.** Checkable in one command; treat as INHERITED until read. |

⚠️ The ⭐ rows are the ones that decide anything. I verified every row I use to argue for a paper
correction or an architecture change; I did **not** re-open all 34 sweep rows, and I say so rather
than implying a uniform standard I did not apply.

---

## Ground truth (⭐ VERIFIED HERE, all of it)

| fact | site |
|---|---|
| `encode_window` flattens `[B,W]` into batch | `stack/tanitad/models/v6.py:4324-4330` |
| `z_op = z_op_win[:, -1]` | `stack/tanitad/models/v6.py:4673` |
| `z_tac, _ = self.uplink_tac(z_op, own_tac)` | `stack/tanitad/models/v6.py:4686` |
| `adapter_tac` = pointwise MLP, **no time axis** | `stack/tanitad/models/v6.py:3977-3979` |
| `adapter_str` = same, one level up | `stack/tanitad/models/v6.py:3991-3993` |
| `isolate_uplink = True` (the autograd confound) | `stack/tanitad/models/v6.py:3229` |
| `_cut` → `.detach()` | `stack/tanitad/models/v6.py:4342`, applied at `:4354` (tac) and `:4377` (str) |
| **`PhiTac` exists** — causal TCN pool | `stack/tanitad/models/tactical.py:99` |
| **`v6.py` references `PhiTac` ZERO times**; imports only `FTac` | `grep -c` = 0; `stack/tanitad/models/v6.py:92` |
| `TacticalStage0` **uses** it: `z_tac = self.phi_tac(z_op)` | `stack/tanitad/models/tactical.py:487`, `:505` |
| flagship `TacticalHead` runs causal blocks over `[B,W,d]` | `stack/tanitad/models/fourbrain.py:332-339` |
| `predictor_op` gets the **full window** | `stack/tanitad/models/v6.py:4758`; mask `predictor.py:165-168` |
| BatchNorm **banned** in the inference path (no cross-frame coupling via batch stats) | `stack/tanitad/models/encoder.py:5` |

> ## ⛔ EVERY `v6.py` LINE NUMBER ABOVE IS STAMPED TO A FILE HASH, NOT TO A DATE — AND HERE IS WHY
>
> **MEASURED this session: `stack/tanitad/models/v6.py` GREW FROM 4,914 TO 5,154 LINES WHILE I WAS
> READING IT.** A sibling stream is editing it live. My first pass resolved `adapter_tac` to
> **3737**; a re-resolve 20 minutes later gave **3977** — a **+240-line shift** that silently
> invalidates every line citation taken before it.
>
> ⇒ All `v6.py` lines here are resolved against **`sha256 d1cd69d7…` (5,154 lines)**, recorded in
> `code/out.json` `meta.v6_py_sha256`, and the probe was **re-run against that exact state** (same
> numbers). ⇒ **Re-resolve by CONTENT (`grep -n "<symbol>"`), never by trusting these integers.**
>
> ⇒ **ROOT-CAUSE CLASS: C114's torn snapshot, in a CITATION costume.** C114 caught it as a false
> test failure; here it would have produced a report whose every file:line pointed at the wrong code
> — *the more authoritative-looking failure of the two.* ⭐ **A line number is a claim about a file
> state. Under live concurrency it must carry that state's hash or it is not evidence.**

⚠️ **The same rot is already in the source, which is how I noticed:** the in-code docstring at
`v6.py:2802-2810` cites `encode_window` at `v6.py:3844-3847` and the forward at `:4197`/`:4207` —
all three now wrong (actual: **4324-4330 / 4673 / 4686**). `F7_F8_CELLS.md:99-101` carries the same
stale numbers. **Worth fixing at source**, and an argument for citing symbols rather than lines.

---

## (a) INVALIDATED — the claim requires a window-integrating `z_tac`

### A1. The paper — highest blast radius

| # | site | quote (short) | note |
|---|---|---|---|
| 1 | ⭐ `Paper/TANITAD_PAPER.md:686` | `z_T = φ_T(sg[z_O(t−3..t)])` | ⛔ **THE WORST SITE.** `:695` calls these properties *"asserted by construction and then checked, not assumed"*; `:705` calls it *"The v6 contract"*. The `sg[·]` is real; the `(t−3..t)` is not. |
| 2 | ⭐ `Paper/TANITAD_PAPER.md:546` | `z_tac … = φ_tac(z_op(t−3..t))` | ⛔ FALSE of the shipped model |
| 3 | ⭐ `Paper/TANITAD_PAPER.md:697` | *"Abstraction by temporal down-sampling under a shrinking state"* | ⛔ **temporal half false, and never checked** — `assert_isolation` covers only property (2), the gradient matrix. Dimensional half (2048→512→256) SURVIVES. |
| 4 | ⭐ `Paper/TANITAD_PAPER.md:687` | `z_S = φ_S(sg[z_T window])` | ⛔ FALSE — the `φ_str` line |
| 5 | SWEEP `Paper/TANITAD_PAPER.md:547` | `φ_str(z_tac window)` | ⛔ same |
| 6 | SWEEP `Paper/TANITAD_PAPER.md:540` | *"temporal down-sampling as the abstraction mechanism"* | ⛔ temporal half only |

### A2. Design docs of record

| # | site | note |
|---|---|---|
| 7 | ⭐ `…/2026-08-07-hierarchical-wm-redesign/HIERARCHICAL_WM_REDESIGN.md:115` | `φ_tac: TCN/attn pool over z_op(t−3..t)` — the origin of the paper wording |
| 8 | SWEEP `…/HIERARCHICAL_WM_REDESIGN.md:116` | `φ_str: pool over z_tac window` |
| 9 | SWEEP `…/HIERARCHICAL_WM_REDESIGN.md:118` | *"Down-sampling in TIME is the abstraction mechanism"* |
| 10 | SWEEP `…/2026-08-07-hierarchical-wm-redesign/V18_BACKLOG.md:56` | `phi_tac`: temporal pool → **satisfied by `tactical.py`, not by `V6Stack`** |
| 11 | SWEEP `…/V5F_DATA_WIRING_AUDIT.md:50` | *"pooled z_op window + hindsight goal labels"* |
| 12 | SWEEP `…/V5F_DATA_WIRING_AUDIT.md:51` | *"pooled z_tac + 25 s corridor labels"* (`φ_str`) |

⚠️ Each of 7–12 is **TRUE of `tactical.py`/`TacticalStage0`** and false of `V6Stack`. The fix is a
**scope tag**, not a deletion.

### A3. Catalog T2 / diagram (already retracted as C115 — listed for completeness)

| # | site | note |
|---|---|---|
| 13 | ⭐ `…/2026-08-07-hierarchical-wm-redesign/V6_TRAINING_MEASURES.md:65` | **time-reversal half only.** ⭐ **Lane-mirror half SURVIVES, is built, and carries the catalog's whole stated justification.** |
| 14 | SWEEP `…/2026-08-16-diagram-conformance/DIAGRAM_CONFORMANCE.md:56` | half |
| 15 | SWEEP `…/DIAGRAM_CONFORMANCE.md:212` | half |

---

## (b) NEEDS A RULING — not invalidated, but reads wrong

| # | site | issue |
|---|---|---|
| 16 | SWEEP `…/V58F_FUSION.md:18` | **"temporal hierarchy" is named "the pillar".** State or clock? **PI call.** |
| 17 | SWEEP `…/V58F_FUSION.md:25` | pipeline diagram carries `φ_tac`/`φ_str` symbols + "(1 Hz)" reading as a state rate |
| 18 | SWEEP `…/V5F_ARCHITECTURE_REVIEW.md:55` | *"spatial belief + temporal abstraction"* — the second half is absent in v6 |
| 19 | SWEEP `…/V5F_ARCHITECTURE_REVIEW.md:49-52` | the v5f-vs-v6 contrast weakens to *"v6 adds a slow-clock predictor over a fast-clock state"* |
| 20-25 | SWEEP `HIERARCHY_VOCABULARY.md:125` · `V6_ARCHITECTURE_REVIEW.md:122` · `V6_TRAINER_DESIGN.md:42` · `Project Steering/Reports/2026-08-15-2200-campaign-science-addendum.md:145` · `stack/tanitad/models/v6.py:3893` · `Paper/figures/v6_architecture.svg:85` | six copies of `z_tac_{t+k} = P_T(…)`. **Mechanically honoured** — `predictor_tac` really rolls. Question is whether rolling an *instantaneous read* is "dynamics". `V6_TRAINER_DESIGN.md:227` already named this trap from the stride side: *"an identity map wearing a hierarchy's name"*. |
| 26-29 | SWEEP `DIAGRAM_CONFORMANCE.md:113`, `:54` · `Paper/figures/v6_architecture.svg:82` · `Project Steering/V6F_PLANNER_DESIGN.md:28` | the "2 Hz clock" verdicts. **Correct as evidenced** (they cite `stride_tac`), but a reader takes them to mean the *state* is sampled at 2 Hz. ⇒ annotate *"state support = 1 frame"*. |
| 30 | SWEEP `V6_TRAINING_MEASURES.md:79` (catalog **S1**) | *"on the T-layer's latent sequence"* + *"manoeuvre context"* — a sequence of independent per-frame reads. **A `z_str` claim.** |
| 31 | SWEEP `V6_TRAINING_MEASURES.md:64` (catalog T1) | *"tactical dynamics"* — same class as 20-25 |
| 32 | SWEEP `V6_TRAINING_MEASURES.md:55` | header assigns **manoeuvre** content to a latent that cannot see a manoeuvre |
| 33 | SWEEP `WM_PHYSICS_PROOF.md:130-132` | uses the `φ_tac` symbol. ⭐ Interestingly this leverage path is *strengthened*: v6's `z_tac` **is** "the encoded present". Symbol still needs correcting. |
| 34 | SWEEP `Project Steering/Mission Plan.md:40` | *"Tactical functions generally occur over a period of seconds"* — a **requirement**, not a claim. Not falsified; **currently unmet**. ⚠️ Agents may not edit this file — escalate only. |

---

## (c) UNAFFECTED — do not "correct" these

| group | sites | why |
|---|---|---|
| ⭐ **`TacticalStage0` / `PhiTac`** | `tactical.py:1`, `:10-13`, `:22-24`, `:96`, `:99-116`, `:184`, `:487`, `:505`; `train_tactical_stage0.py:38`, `:526` | **MEASURED here to integrate the window.** Its "down-sampling in TIME" docstring is TRUE of that file. Trained: registry §1.13b. |
| ⭐ **v1 flagship tactical** | `fourbrain.py:43`, `:268-276`, `:332-339` | causal transformer over the state window |
| **Every v1 tactical RESULT** | `MODEL_REGISTRY.md:133-134`, `:427`, `:1150`, `:2569`; `PROGRAM_OVERVIEW.md:328`, `:335`; `Paper:1493`, `:1730`, `:3196`, `:3203` | measure `fourbrain.py` |
| **The whole `ctx→tactical` seam family** | `README.md:269`; `PROGRAM_OVERVIEW.md:282`; `CLAUDE.md:37-39`; `Paper:1455`, `:1462`, `:1467`, `:1474`, `:3509`; `HPP1_UNBLOCK_REPORT.md:22-23`, `:89-91`, `:111-113`, `:122`; `ARCHITECTURE_WIRING_COMPARISON.md:266`, `:518`, `:558`; `Reviews/2026-07-25-…/R2_…:67`, `R3_…:101`; `V35_DESIGN.md:157`; `V4_FLAGSHIP_DESIGN.md:291`; `Progress Reports/2026-W33.md:168`; `Daily Reports/2026-07-18.md:8`, `:23` | a **conditioning** question in a **different model**. ⭐ Includes the corrected **+0.0148** seam in `CLAUDE.md`. |
| ⭐ **The operative path** | `Paper:535`, `:693`; `v6.py:4758`, `:4270-4276`; `predictor.py:165-168` | full window + real rollout. **This was my positive control and it fired.** |
| **Band / horizon / integrator** | `DIAGRAM_CONFORMANCE.md:92`, `:112`; `HIERARCHY_VOCABULARY.md:106-113`; `V6_ARCHITECTURE_REVIEW.md:21-23`, `:76`; `v6_architecture.svg:31`, `:141-147` | property of the **emitted control sequence** |
| **Vocabulary / decision shape** | `HIERARCHY_VOCABULARY.md:79-89`; `DIAGRAM_CONFORMANCE.md:114-117`; `FACTORED_GOAL_HEAD.md:120`, `:326`; `V6F_PLANNER_DESIGN.md:592-598`; `V6_ARCHITECTURE_REVIEW.md:86-88` | the shape of the decision **read off `z_tac` at the last tick** — the safe side of the distinction |
| **Imagine-and-select** | `Paper:308-309`, `:335`, `:437`, `:3163` | imagination runs on the **operative** predictor |
| **The four metric families** | — | measure emitted trajectories/decisions; none assumes a temporal `z_tac` |

### Already-correct statements — propagate, don't touch

| site | note |
|---|---|
| ⭐ `…/2026-08-16-x4-p9/X4_P9.md:16` | *"the uplink reads only the window's last frame"* — **written 2026-08-16, two days BEFORE C115, and already right.** Nobody propagated it. |
| ⭐ `stack/tanitad/models/v6.py:2796-2822` | the in-code finding (`time_reverse_window` docstring) |
| SWEEP `stack/tests/test_v6_t2_contrastive.py:363-382` | `test_z_tac_reads_the_last_frame_only` |
| ⭐ `Project Steering/RETRACTION_LOG.md:6288-6313` | C115 itself; `:6310-6311` is the standing instruction this survey answers |
| ⭐ `Project Steering/PREREG_TEMPORAL_LATENT.md:23`, `:30-31` | *"no arm in this programme has ever probed a representation that is temporal by construction"* — and `refc.py:1412-1422` keeps only the LAST frame's feature map |

⭐ **THE LAST-FRAME-ONLY DEFECT IS A RECURRING CLASS IN THIS CODEBASE, NOT A v6 ONE-OFF.**
REF-C (`refc.py:1412-1422`), `V6Stack.uplink_tac`, and `V6Stack.uplink_str` are three independent
instances. **Worth a standing check in the conformance audit rather than three separate retractions.**

---

## Out of programme — listed to close the search (SWEEP)

`Ressources/AD_TRANSFER_RESEARCH.md:172` (ALPS-4B, different repo) ·
`…/V3_HIERARCHICAL_PLANNING_DESIGN.md:201` (v3, *spatially* pooled, superseded) ·
`Ressources/Deep Think Analysis/Deep Think 3/4/9.md` (external ideation, never adopted).
