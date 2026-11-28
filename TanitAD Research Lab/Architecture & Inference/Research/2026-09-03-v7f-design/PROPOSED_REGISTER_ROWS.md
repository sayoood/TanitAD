<title>Proposed register rows — v7f design freeze and pre-registration (2026-09-03)</title>

# PROPOSED REGISTER ROWS for `Project Steering/GOALS_AND_CLAIMS.md`

`TanitAD Research Lab · Architecture & Inference FlyWheel · 2026-09-03 · 0 GPU`
`Proposed text, NOT applied. Statuses stay PROPOSED until the PI / Master Mind reads them — the`
`precedent is D-P2-LEAK-AUDIT ("the nine rows below are the FlyWheel's proposed text, applied`
`verbatim; statuses stay PROPOSED until the PI reads them").`

⚠️ **Ownership note.** This task's brief scopes the agent to `Project Steering/PREREG_V7F.md` and its
own package; `GOALS_AND_CLAIMS.md` is not mine to edit. CLAUDE.md's *"update the register in the same
turn"* rule and that scope are in tension, so the rows are proposed here **verbatim-ready** and the
conflict is escalated rather than resolved unilaterally.

⛔ **Rows 3 and 4 CORRECT standing instructions and should not wait for the rest.** Row 3 says a gate
in an active skill is inadmissible as written; row 4 removes a stated launch blocker.

---

## ROW 1 — the hypothesis (NEW, must exist before any v7f arm launches)

📋 **H-V7F-1 — A DINOv3-INITIALISED TRUNK, TRAINED UNDER A DISTILLATION-ANCHORED LOW-LR SCHEDULE, WITH
AN ENCODER-SIDE LATENT-DISPLACEMENT ACTION DECODER (LDAD ON `Δz`, TARGETS `(a, κ)`, `v` EXCLUDED) AS
THE SOLE NEW ANTI-COLLAPSE TERM, IS THE FIRST TanitAD RECIPE THAT MAKES THE PREDICTOR ACTION-SENSITIVE
AT THE PROGRAMME'S COMMITTED BAR WITHOUT DESTROYING THE TRUNK'S DYNAMIC DECODABILITY (PRE-REGISTERED
2026-09-03, ArchInf FlyWheel, 0 GPU; `Project Steering/PREREG_V7F.md` +
`TanitAD Research Lab/Architecture & Inference/Research/2026-09-03-v7f-design/`).** Status **OPEN /
PRE-REGISTERED, UNRUN.** **Success, committed in advance:** at the checkpoint selected by the two-key
rule (sensitivity filter, then held-out one-step embedding loss — *never* validation loss, because
val-loss selection picks action-collapsed checkpoints **17/36** times, `2608.06706`), the anchored
action-divergence read (`taniteval/tools/actdiv_anchored.py`, h = 1, TRAINER lift `v_last/10`, the
24-clip / 144-window banked actdiv population) returns verdict **SENSITIVE** — `F_sep ≥ 5 ×` the arm's
own 200-permutation null p95 on **both** axes, Spearman ρ ≥ 0.8 on both, `cos(m(+L), m(−L)) < 0` at
**every** level on both, and **`rel_to_scene(2σ) ≥ 0.0595`** (= 10 × the incumbent `postrain30k`'s
banked 0.005947, the programme's existing committed bar) — **AND** the Observer-Effect monitor's
frozen-feature LINEAR probe on the dynamic targets holds at **≥ 0.70 × its own step-0 value** (CI
excluding 0.70) — **AND** G-DRIVE at **T1** shows `cl − ha` ADE separated below 0 on the paired
episode-cluster bootstrap over the v7.2 labelled eval split, with `trivial_profile_fraction < 1.0` and
`baseline_won_frac` below the refav1 reference **0.7571**. **Failure, committed in advance:** (a)
`rel_to_scene(2σ) < 0.0595` at every step-stamped checkpoint ⇒ **the encoder-shaping family (LDAD /
SMWM / EB-JEPA-IDM) is REFUTED for TanitAD on a realised-motion action channel**, on the one
architecture that could host it, and the action line moves to the predictor-side family (ACID /
ActSWM hinge / AD-JEPA offset head), **all frozen-trunk compatible**; (b) monitor < 0.70 × step-0 ⇒
revert to frozen DINOv3 + a trained per-token adapter (`2602.18639`: 0.48 → 0.78 at 0 trunk compute);
(c) a VOID G-DRIVE read (arms bit-identical, or `trivial_profile_fraction == 1.0`) is stamped **VOID**,
never "no difference" (`D-REFAV1-PAIRED-READ-VOID`). **One variable: `trunk_policy`** ∈ {frozen, full,
anchored, lastk}; ⛔ **attribution is carried by the LADDER, never by v7f** — R0 (LDAD tautology test
at step 0, 0 GPU, can REFUTE the line before an arm is spent), R1 (`bptt_truncate`), R2 (λ), R3
(`trunk_policy`), each one-variable with its own deliberate-regression arm, and **if a rung is skipped
the prereg is void for the element that skipped it**. Tier: T0 for R0–R3, T1 for G-DRIVE.

---

## ROW 2 — the design decision (NEW)

⭐ **D-V7F-TRUNK — v7f's ENCODER IS DINOv3 ViT-B/16, TRAINABLE, ANCHORED — NOT FROZEN, NOT NAIVELY
FINE-TUNED, AND EXPLICITLY NOT LoRA (PI directive 2026-09-03 *"at v7 we ARE allowed to train the
trunk — consider the previous findings, taking DINO as init"*; design
`…/2026-09-03-v7f-design/DESIGN.md`).** The design is the middle the evidence actually supports, and
each third of it is measured: **(i) trainable**, because our own one-variable freeze cell is
DEGENERATE (`postrain30k_freeze` held-out nrmse **0.9301 vs 0.8115, +14.6 %**, E-DEC-64) *and* killed
the command channel (steer/accel ratio **0.00078** vs the trainable incumbent's **0.00561**, 7.2×
lower; paired factor **0.45 [0.28, 0.72]**, `H-LEAK-5` / `D-P2-LEAK-AUDIT` §0 finding 8) — freezing
made the action pathway **deader**, which is `GS-10`'s encoder-shaped-sensitivity hypothesis read
forwards; **(ii) anchored and low-LR**, because full fine-tuning corrupts what the trunk carries
(Observer Effect `2602.12218`: frozen linear probe ρ **0.91** → full fine-tune **0.05**, last-layer
0.65, a kinematic invariant **0.94 → −0.03**, damage concentrated in blocks B5–B10) and because
distillation **into** the trunk is what made the field's best full fine-tune work (Latent-WAM
`2603.24581` Table 4: 89.3 vs 88.0 for concatenated frozen features vs 88.3 for none); **(iii) NOT
LoRA**, because Latent-WAM Table 5 makes **Base-LoRA the WORST row (68.5)** against Base full **89.3**.
⭐ **The geometry is parameter-neutral:** config E's encoder is already 768×12, so DINOv3 ViT-B/16
(≈ 85 M) replaces it 1:1. ⛔ **Two things are NOT wired and are the run's blocking implementation
items:** there is **no DINOv3 weight loader into `ViTEncoder`/`ViT5Encoder`** (two probes: `grep` over
`stack/`, `tools/`, `taniteval/`, and PowerShell `Select-String` over the same trees — DINOv3 appears
only as O7's frozen teacher `train_v6_staged.py:931`, `dino_precompute.py`, `dinov3_fp8_encode_ship.py`)
and `--init-from` refuses a partial checkpoint (`train_v6_staged.py:7236-7247`); and the optimizer is
a **single flat AdamW over one `lr`** (`:6120`) with `--freeze-encoder` all-or-nothing (`:5964`), so
there is **no per-group / discriminative learning rate**. Recommended route (PREREG §10 D1-A): a
**seed-checkpoint converter** — full stack, encoder subtree = the real DINOv3 weights, everything else
fresh-init — plus `--trunk-lr-scale` / `--trunk-lr-warmup-steps` / `--w-trunk-anchor`. Evidence class:
MEASURED (ours, source read) + PUBLISHED-PRIMARY. **PROPOSED default; the PI closes.**

---

## ROW 3 — ⛔ CORRECTS A GATE IN AN ACTIVE SKILL

⛔ **D-GRANK-8.56-INADMISSIBLE — `TanitAD_ValidateAIDesign`'s G-RANK CRITERION *"participation ≥ 8.56
(frozen DINOv3, MEASURED on our frames)"* CONTRADICTS THE REGISTRY, AND THE REGISTRY WINS (found
2026-09-03 while writing `PREREG_V7F.md`; no new measurement).** `MODEL_REGISTRY.md` §13.3 and
`stack/tanitad/models/v6.py:1517-1552` record, through **`spectrum_report` itself** at n = 1440 in
every row, frozen DINOv3 ViT-L/16 patch tokens mean-pooled: **5.756** on the 12 physicalai-val clips
the 8.56 is sourced to, **20.228 ± 0.327** on the 130-clip corpus the 40.77 is sourced to, **20.516**
at full n = 5617 — *"Neither published number survives"*, the 3.51× spread is **episode diversity
alone** (H-RANK-23), sample size is not the confound (H-RANK-21 REFUTED), and the code carries
⛔ *"Until a matched-d, matched-corpus reference exists, DO NOT FAIL AN ARM ON THE PARTICIPATION
CLAUSE"*, pinned by `stack/tests/test_participation_floor_provenance.py`. ⇒ **The skill's gate would
pass or fail arms on a number no live instrument reproduces, at a different ambient dimension**
(`z_op` is d = 2048, the DINOv3 column d = 1024). **Proposed fix, applied in `PREREG_V7F.md` §6.1:**
G-RANK becomes a **matched-reference** gate — the arm's **val-side** participation (σ², never
`effective_rank`, C132) must exceed a frozen-DINOv3 reference measured on the **same corpus, the same
episode count and the same ambient `d`**, through `spectrum_report`, banked in the same run's `raw/`;
until that reference exists **G-RANK reports and does not refuse**. ⚠️ And C131 stands regardless:
**rank is NECESSARY, NOT SUFFICIENT** — flagship v1-era had the highest rank ever measured here and no
environment interpretation. **Work item: `.claude/skills/TanitAD_ValidateAIDesign/SKILL.md` §3 should
be amended** so the skill and the registry stop disagreeing. Root-cause class: *a number true for one
scope, quoted where that scope does not apply* — the `df` / `free` / cgroup / `step_s` family, this
time inside a gate that decides.

---

## ROW 4 — ⛔ REMOVES A STATED LAUNCH BLOCKER

✅ **D-R6-DOES-NOT-BIND-V7F — BACKLOG R6 (`--bptt-truncate` does not reach the O1 stage-A rolls) IS NOT
A v7f PRECONDITION, BECAUSE v7f RUNS O1 AT ZERO AND THE WHOLE O1 BLOCK IS GUARDED ON ITS WEIGHTS
(MEASURED from source 2026-09-03: `stack/scripts/train_v6_staged.py:3691`).** The line reads
`if w.o1_ctrl or w.o1_fact or w.o1_scene:` and the six `rollout_transitions` calls inside
`train_stage_a.stage_a_losses` (`stack/scripts/train_stage_a.py:274-320`) sit **inside** that block —
at `--w-o1-ctrl 0 --w-o1-fact 0 --w-o1-scene 0` they are never reached, so an untruncated stage-A roll
cannot exist. BACKLOG R6's *"owner needed before any k = 60 launch relies on the flag"* is therefore
scoped to **launches that turn O1 on**, and v7f's settled recipe keeps all three at zero (the
do-not-add list, `V7_RECIPE_AND_SCALEUP.md` §5.1). ⚠️ R6 stays open as a **correctness** item — the
one-line fix and its property test are still worth having — but it is not a v7f blocker. Evidence
class: MEASURED (ours, source read; no execution).

---

## ROW 5 — the horizon correction (PROPOSED)

⚠️ **D-V7F-BPTT-4 — `--bptt-truncate 15` IS AN INHERITED refav1 VALUE WITH NO PUBLISHED SUPPORT;
v7f's RECOMMENDED GRADIENT DEPTH IS 4, DECIDED BY A ONE-VARIABLE TINY ARM (PROPOSED 2026-09-03).**
`D-V7-WIRING`'s suggested v7f launch line carries `--bptt-truncate 15`, which is refav1's value.
**Every** banked primary truncates far shorter: V-JEPA 2-AC `2506.09985` §3.1 *"only differentiate the
predictor through one recurrent step"* (T = 2); What-Drives-Success `2512.24497` §C back-propagates
through the **last** prediction only, finds **K = 2** optimal in simulation and **K = 6** on DROID with
a decline beyond (Fig. 3b), and gives the mechanism (Remark 1: the K-step loss lowers the effective
Lipschitz constant along the rollout at the cost of one-step accuracy); EB-JEPA `2602.03604` K = 8 with
a Pareto at **4**. 15 is **7.5×** the largest published gradient depth; **4 is 2×** it. The **forward**
horizon stays `--o5-k 60` (the 6 s contract; the corpus' median manoeuvre is 12.5 s and k = 8 trains
0.8 s). ⇒ v7f rung **R1** runs `bptt_truncate ∈ {0, 2, 4, 15}` at `o5_k = 60`, with
**`bptt_truncate = 0` as the deliberate-regression arm** — the full chain that diverged at gnorm
**2.1e9** and was killed at 9,000, which must diverge again or the stability gate is void. ⚠️ Held
constant across the rung: the **window census**, which depends on `max_horizon`
(`train_v6_staged.py:3188-3214`) — an episode shorter than `window + max_horizon` yields ZERO windows,
so `o5_k` is fixed and each arm reports its census. Pre-registered fallback: `grad_norm > 1e3` on ≥ 3
of the last 20 rows ⇒ drop to 2; a non-finite gradient ⇒ restart from the last checkpoint under
`--precision fp32` (the `C-REFAV1-BF16-OVERFLOW` precedent).

---

## ROW 6 — the evaluation discipline, transferred (PROPOSED)

⭐ **D-V7F-EVAL-REFUSALS — EVERY v7f T1 READ REPORTS `trivial_profile_fraction`, `baseline_won_frac`
AND A PAIRED BIT-IDENTITY CHECK **BEFORE** ANY FAMILY ROW, AND CARRIES A `ha0` CONTROL BESIDE
HOLD-ACTION (PROPOSED 2026-09-03; the refav1 lesson transferred).** Three refusals, committed in
advance: ⛔ `trivial_profile_fraction == 1.0` ⇒ the read is **VOID** and no family row is reported
(`tools/straight_line_probe.py`; refav1's step-1,000 closed loop had `y ≡ 0` and constant speed on
**140/140** windows); ⛔ two arms bit-identical on the window grid ⇒ **VOID**, stamped as such and
**never** reported as "no difference" (`D-REFAV1-PAIRED-READ-VOID`: two checkpoints differing by a
full recipe were identical to 1e-9 on 140/140, max |Δ| exactly 0.000000 m); ⛔ `baseline_won_frac` not
below the refav1 reference **0.7571** ⇒ the planner has not departed from the CV baseline and no
planning claim is available. And the controls must include **`ha0`** — the straight line at `v0`,
the strongest trivial baseline — because hold-action drifts **0.12 m** even where the human drives
straight and **0.76 m** on curved windows while the straight line reads **0.035 / 0.496**, so
refav1's apparent "lateral planning gain over hold-action" was an artefact of a control weaker than a
trivial baseline (RETRACTION_LOG 2026-09-03 #2). ⚠️ **Instrument precondition:**
`C-REFAV1-KIN-CONTRACT-LAT` — the open-loop replay of the RECORDED `(a, κ)` misses the human laterally
by **0.716 m** on curved windows, more than the human's own excursion (0.496 m); whether the **v7** T1
adapter shares the defect is **UNVERIFIED** and must be checked before G-DRIVE, or no lateral row is
quotable.

---

## ROW 7 — the action-sensitivity gate, committed (PROPOSED)

📋 **G-ACT — v7f's ACTION-SENSITIVITY GATE IS THE ANCHORED (ZERO-ACTION) DISPLACEMENT AT
`rel_to_scene(2σ) ≥ 0.0595`, VERDICT SENSITIVE, WITH ITS SHUFFLED-ACTION NULL COMMITTED IN ADVANCE
(PROPOSED 2026-09-03).** Instrument `taniteval/tools/actdiv_anchored.py` (NEW, 34 unit tests),
metric `d_i(a) = ẑ_{t+h}(a_t ← a) − ẑ_{t+h}(a_t ← 0)` — the ActSWM / Delta-JEPA Fig. 6 / AD-JEPA form,
so the read is comparable to published effect sizes. Population: the banked actdiv windows verbatim
(24 clips of `physicalai-val-0c5f7dac3b11-w120-256x640cyl`, first 60 frames, **144 windows**), h = 1
(the only trained head), **TRAINER lift `v_last/10`** — ⛔ never the banked scripts' `v_first/30`,
which is `H-LEAK-1` and moves the legacy ratio **3–5×**. **PASS:** `F_sep ≥ 5 × p95` of the arm's own
200-permutation null on both axes AND Spearman ρ ≥ 0.8 on both AND `cos(m(+L), m(−L)) < 0` at every
level on both AND **`rel_pinned(2σ) ≥ 0.0595`** with its clip-bootstrap interval excluding 0.0595.
⛔ **THE DENOMINATOR IS PINNED to a named one-variable reference arm (v7f's is the R3 scratch-trunk
arm), NEVER the arm's own scene spread** — `PREREG_MM_E19_K60_HORIZON.md:74-84` **DEFECT 1**: the
k = 60 arm's `scene_factor` **1.7126** silently turned a written-down 10× bar into a **17.13×** one,
*"a criterion whose threshold is a function of the result is not a pre-registration"*, and it is
sign-blind the other way (an arm that genuinely **doubled** its sensitivity would report **+17 %** and
be written up INERT). **The bar's provenance:** 10 × the incumbent's banked **0.005947**
(`:61` HORIZON-WORKS). **Four co-primaries, never folded away** (`:90-105`): the action side alone;
the arm's **own** scene spread (it carried **73.7 %** of MM-E19's ratio fall); the **h ≥ 2 floor**
(~**1e-05** on *every* arm measured — k = 8, k = 60 and the incumbent — horizon-INDEPENDENT); and the
**training-window count** per arm (`o5_k` also sets `o4_n`: **415,002 → 319,002**, so the k = 60 arm
trained on **76.9 %** of the k = 8 arm's windows). ⛔ **A WORSE branch, pre-committed** (DEFECT 2, whose
outcome set admitted none and the arm fell **0.48×**, off the table): **G-ACT-WORSE** =
`rel_pinned(2σ)` below the incumbent's **0.0087** with the interval excluding it. ⛔ **An interval is
required and the instrument lacks one** — backlog **L-13**, *"a decision statistic without an interval
is a gap, not a virtue"*: `actdiv_anchored.py` must emit a clip-level bootstrap over its 24 clips
before G-ACT can be read. **Where we are (MEASURED):** `postrain30k` **0.0087**, `k8clip05p30k`
**0.0138**, `k60clip05p30k` **0.0032**, `rdw8p30k` **0.0181** — none material; the κ axis is a clean
linear map at ~10⁻³ gain and the `a` axis loses sign consistency beyond ~1σ on the three
postrain-recipe arms. **Shuffled-action control,
expected reading committed:** the 200-permutation null reads median **0.86–0.99**, p95 **1.01–1.28** in
F_sep units on the v7 population; ⛔ a measured p95 above **2.0** VOIDs the panel; on the realised read
the shuffled-action control must collapse the separation (refav1 reference 27.90 → ≤ 1.83). **VOID
conditions:** C0 identity ≠ 0, zero-model ≠ 0, `scene_spread ≤ 1e-6`, or all displacements identically
0. **Published priors for the jump:** ActSWM gap **0.002 → 0.592 → 0.760**; AD-JEPA post-hoc
**−0.002…+0.052 → 0.19–0.52**; Delta-JEPA `Δx`-from-`Δz` r **0.765 → 0.992**. **A FAIL is a result,
not a null run** — see H-V7F-1's committed failure branch.

---

## ROW 8 — decisions the PI must close (PROPOSED, defaults in bold)

📋 **D-V7F-DECISIONS — NINE OPEN DECISIONS FOR THE v7f FREEZE, EACH WITH A RECOMMENDED DEFAULT AND THE
COST OF ITS ALTERNATIVES (`PREREG_V7F.md` §10).** **D1** how DINOv3 becomes the init → **a faithful
`DINOv3ViTModel` trunk + a seed checkpoint that `--init-from` loads with zero missing keys** (the only
route under which the Observer-Effect step-0 control measures the published trunk). **D2** encoder
input → **`--newest-frame-only --in-channels 3`** (the only exact transfer; costs one extra
scratch-trunk reference arm because `encoder_input` must be identical across arms). **D3** tiny-rig
trunk size → **the same ViT-B/16 v7f will deploy** (the Observer Effect is a property of *those*
weights at that depth; a ladder on a different trunk does not transfer) — per-arm cost must be
**re-measured**, the ~19 M / ~17 min figure no longer applies. **D4** the encoder-shaping term →
**LDAD on `Δz`, decoding `(a_long, κ)`, `v` excluded** (the only candidate whose form was ablated
against the concat shape `GS-1` flags, and the only one with an in-paper λ optimum rather than a 300×
environment-specific range). **D5** ⛔ **PARAMETER BUDGET — the register must carry the PI's number:**
config E is **336.5 M** against the north star's **sub-300 M**; recommended **predictor 1024×12 →
1024×8 ⇒ ≈ 286 M** (ESTIMATED by the closed-form 12·d²/block; the real number comes from
`config.json`), at the cost that predictor depth is exactly the capacity that matters on a pretrained
trunk (DINO-world `2507.19468` Table 4: from-scratch 46.9/87.1/59.4 vs pretrained-and-fine-tuned
59.4/93.8/68.7) — so it must be its own one-variable tiny arm if taken; alternatives are raising the
north star (PI-only) or shrinking the trunk (⛔ inverts the design's premise: FROST-Drive
`2601.03460` shows fine-tuning **helps a weak trunk and hurts a strong one**). **D6** gradient depth →
**4** (Row 5). **D7** trunk schedule → **`trunk_lr_scale 0.1`, 2,000-step warmup, `w_trunk_anchor 1.0`
to a frozen copy of the same weights**; ⚠️ the anchor is not automatically benign (JEPA-x
`2608.24044`'s `DISTILL` control read **0.516** vs baseline 0.361), which is why the un-anchored
`full` arm is its one-variable control. **D8** reserve the `m_t` slot → **YES**, three lines, zero
cost (`D-M_T-SLOT`). **D9** slot and order → **PI**; R0 is 0-GPU and runnable on the 4060 without
touching Thor or the A40. ⛔ **Nothing is scheduled by this package.**

---

## ROW 9 — the launch blocker restated (PROPOSED, for visibility)

⛔ **D-V7F-PRECONDITIONS — v7f MAY NOT LAUNCH UNTIL EIGHT PRECONDITIONS ARE TRUE, AND THE FIRST IS
BACKLOG R8 (`PREREG_V7F.md` §11).** (1) **eval-clip exclusion** in the v7 trainer's B1 path — the 141
eval clips with pixels sit inside the 4,713-clip cache and `build_train_episodes` /
`build_v2_providers` carry no exclusion list (`D-V7-WIRING`, two probes); verification: the window
census of a `--require-parity` launch shows **0 eval clips**. (2) the DINOv3 **seed checkpoint** with
recorded provenance (`dinov3_model_id`, `sha256`, adapted/unadapted keys). (3) the **G-RANK matched
reference** measured (Row 3). (4) **R0 cleared** — LDAD is not tautological at step 0 (⭐ this can
REFUTE the whole line for 0 GPU, per `GS-2 (2)`; the action channel is realised motion, `E-DEC-57`
r 0.9664 corpus-wide). (5) **R1, R2, R3 cleared**, each one-variable with its regression arm FAILING,
verified by pairwise `--print-launch` diffs. (6) the **`m_t` slot** reserved. (7) the **T1 lateral
kinematic contract** verified on the *v7* adapter. (8) **H-V7F-1 applied** to this register.
⚠️ Out of v7f's scope and stated so: the **tactical CE term** on the factored v7.2 labels (BACKLOG R7
— the labels land in the batch and no loss reads them; a new loss term goes through
`TanitAD_ValidateAIDesign` as its own arm, never silently inside v7f) and the **strategic ARG slot
encoder** (BACKLOG R9).
