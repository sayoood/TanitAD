<title>Proposed register rows — anchored actdiv on refav1 (2026-09-03)</title>

# PROPOSED REGISTER ROWS for `Project Steering/GOALS_AND_CLAIMS.md`

`TanitAD Research Lab · Architecture & Inference FlyWheel · 2026-09-03`
`Proposed text, NOT applied. Status stays PROPOSED until the PI/Master Mind reads it —`
`the precedent is D-P2-LEAK-AUDIT ("the nine rows below are the FlyWheel's proposed text,`
`applied verbatim; statuses stay PROPOSED until the PI reads them").`

⚠️ **Ownership note.** `CLAUDE.md` binds *"any session that asserts, supports, or refutes a claim
updates `GOALS_AND_CLAIMS.md` IN THE SAME TURN"*, while this task's brief scopes the agent to
read-only outside its own package. The rows are therefore proposed here in full, verbatim-ready,
and the conflict is escalated rather than resolved unilaterally. **Row 1 CORRECTS a standing
claim** and should not wait.

---

## ROW 1 — the verdict (CORRECTS a standing hypothesis)

✅ **H-REFAV1-LAT-INSENSITIVE — REFUTED. refav1's step-1,000 predictor IS sensitive to the lateral
action: the anchored displacement response to κ is 2,298× (fp32 incumbent) and 11.8× (clean epoch)
its own measured permutation null, PERFECTLY LINEAR in κ and PERFECTLY ANTISYMMETRIC — so the
curvature-≡-0 plan on 140/140 windows is a PROPERTY OF THE PLANNER'S COST, NOT OF THE WORLD MODEL
(MEASURED 2026-09-03, dev-box CPU, 0 GPU; `Research/2026-09-03-anchored-actdiv-refav1/RESULT.md`
§4, raw `…-and-transition-probe/raw/actdiv_anchored_refav1_step1000_{fp32,ep2}.json` +
`…-anchored-actdiv-refav1/raw/actdiv_anchored_refav1_step1000.json`).** The Delta-JEPA Fig. 6 /
ActSWM anchored read `d(a) = ẑ_{t+1}(a) − ẑ_{t+1}(0)`, at h = 1 on the T1 read's own **140 windows**
(20 eval-slice episodes, `k_loader = 30`, stride 10; loader REUSED from `refav1_arm.py:483`), with
the lateral channel at ±{0.02, 0.05, 0.1} with a = 0, and a ∈ ±{0.5, 1.5} m/s² with lateral = 0.
⛔ **UNITS, recorded after the run on a source check:** that channel is a **STEER ANGLE in rad**,
not a curvature — `physicalai.py:621` writes `arctan(wheelbase·curvature)` into it and
`refav1_loader.py:264` labels it `kappa` (this is `C-STEER-CURVATURE-INTERFACE`, committed
independently 2 min after this run finished). **No number changes:** separation is invariant under
a monotone reparametrisation, σ is measured in the same channel (so 0.1 ≈ 2.1σ is exact), and
`tan δ ≈ δ` to 0.34 % over this range, so linear-in-steer is linear-in-curvature to the three
figures quoted. Equivalent true curvatures at the register's UNVERIFIED L ≈ 2.9 m are ≈ 0.0069 /
0.0173 / 0.0346 rad/m (R ≈ 145 / 58 / 29 m). Verdict space = `tac`
(the planner's own cost space, `_tac_field`); **all three spaces agree on both checkpoints**.
κ-axis `F_sep` **5,962.49** vs null p95 2.595 (incumbent) and **29.91** vs 2.545 (clean epoch);
a-axis 3,831.33 vs 3.696 and 102.29 vs 3.235. Exact full-field mean-displacement norms scale as
**1 : 2.500 : 5.005** and **1 : 2.498 : 4.980** against κ levels 1 : 2.5 : 5 (Spearman ρ = +1.000
both), with cos(m(+L), m(−L)) = −0.9956 … −0.9999. **Every control at its known value:** C0
identity and the zero-model control **exactly 0.0**; the zero anchor exactly 0.0; scene spread
4.2492 / 0.089146; the 200-permutation label null measured at median ≈ 1.0, p95 2.5–3.7; and the
**shuffled-ACTION control collapses the realised κ separation 27.90 → ≤ 1.83 and 16.21 → ≤ 1.86**.
⛔ **Leak check (D-P2-LEAK-AUDIT / H-LEAK-1) is clean and INERT here:** both step-1,000 configs
carry `speed_channel: false` (verified in `ckpt_ep2\config.json` and the incumbent's `ckpt['cfg']`),
so `a_dim 2 → a_in_dim 2` and **no speed value enters the predictor**; the tool never builds the
channel itself but calls `RefAV1.augment_actions`, which divides by the model file's own
`SPEED_SCALE_MPS` = 30.0 (`refa_v1.py:90`) and refuses a pre-widened action — pinned by two tests
including a source-level guard. **What IS small is κ RELATIVE to a:** ‖m(κ=0.1)‖/‖m(a=1.5)‖ =
**0.00329** (incumbent) and **0.02514** (clean epoch) on the exact full field, at matched ≈2σ levels
(σ = [a 0.7426, κ 0.047565]) — the lateral channel moves the imagination 1/300th to 1/40th as far
as the longitudinal one. ⭐ **THIS IS THE MISSING HALF OF `C-STEER-CURVATURE-INTERFACE`.** That row's mechanism — *"a planner
proposal of a real 12.5 m turn is imagined by the predictor as a 36 m one, so turning looks nearly
free of consequence, the cost surface goes flat in κ, and the search returns the straight line"* —
**requires a predictor that responds to its lateral input**; against a deaf predictor a ×2.9
mis-scaling would be irrelevant. This read, pre-registered and executed before that row existed, is
the measurement that the predictor responds. ⇒ **world model hears it (here) + planner mis-speaks
it by ×2.9 (that row) = one explanation of the flat plan, measured from both ends.**
Consequences: **(i)** the amendment to `D-REFAV1-STEP1000-READ` that names
*"refav1's imagination is insensitive to the lateral action"* is **CORRECTED — the imagination is
not insensitive**; **(ii) the next probe is the PLANNER'S COST SURFACE**, whose candidate mechanism
is now quantitative AND compounding: the cost scores `1 − cos` to the goal in this same `tac` space
and adds an explicit `0.05·κ²` penalty, so a lateral candidate that is intrinsically ~300× less
potent than a matched longitudinal one, *further under-actuated ×2.9 at the boundary while the
penalty is charged at the full proposed κ*, can be dominated outright — ⚠️ **NOT measured here**;
the probe should evaluate `_cost_chunk`'s four terms separately at h = 10 in BOTH conventions (raw,
and with `κ → arctan(L·κ)` at the model boundary), which quantifies the defect and its repair in
one panel;
**(iii)** the P2-family framing still holds on MAGNITUDE, not on structure: the per-dim response is
**0.4 %** and **1.3 %** of the scene std against the programme's 5.95 % material bar, i.e. the same
*structured-but-immaterial* picture the v7 arms read; **(iv)** `C-REFAV1-KIN-CONTRACT-LAT` (the 0.716 m lateral replay miss) is **RESOLVED by
`C-STEER-CURVATURE-INTERFACE` as the same unit defect** (0.7156 → 0.0600 m under
`κ = tan(steer)/2.9`) — quoted INHERITED; this package did not contribute to it; **(v)** both
checkpoints read here are now **SUPERSEDED** — `ep2` was retired at step 6,850 and the epoch
restarted as `ep3` with `speed_channel: true` (`D-REFAV1-EP3-SPEED`); this read stays valid as
banked evidence about them, and that row's independent proof that `operative.act.0.weight` is
`[1024, 2]` in **both** banked checkpoints **corroborates** the speed-channel finding above from a
third direction (config, code path, and now weight shape).
**Reproducibility:** a second, independently launched pass over both checkpoints reproduces the
first **BIT-FOR-BIT** — every F, null, norm, cos and ρ identical to every printed digit, C0 and the
zero-model control exactly 0.0 in both — so the verdict is not a seed or harness artefact (same
code, same box: this is determinism, NOT an independent implementation).
⭐ **And the arms' ARCHITECTURES ARE IDENTICAL — the parameter gap is the frozen EMA teacher:**
rebuilding both configs gives `ema_targets=True` total **182,455,604 = student 175,164,468 + EMA
path 7,291,136**, and `ema_targets=False` total **175,164,468**, exactly the incumbent.
`_EmaTargetPath` (`refa_v1.py:855`, built at `:1052`) is a frozen deepcopy of `adapter`,
`tac_queries`, `tac_pool`, `strategic.read` with every parameter `requires_grad=False`, moved only
by `ema_update()` at training time, and **the diagnostic's forward path never touches it**. The
whole-`cfg` diff between the two checkpoints is **exactly one field, `ema_targets: False → True`**.
⚠️ **Limits stamped:** h = 1 (0.2 s, one operative step) — the planner rolls h = 10, and that read
is a GPU follow-up **not done**; step 1,000 of a 21,109-step epoch; and the arms are still not
one-variable as a RECIPE (EMA targets + bf16 + TF32 moved together; the incumbent's fp32 half is
INHERITED from the register's description, as its checkpoint carries no `args` block to verify).
Tier **T0-DIAGNOSTIC**; the four metric families are REFUSED on the record in the JSON (a
latent-space probe has no trajectory to score).

## ROW 2 — the instrument (H-GS8-1, refav1 half) and the paired-read follow-through

⭐ **D-REFAV1-IMAGINATION-DISCRIMINATES — THE ANCHORED DISPLACEMENT INSTRUMENT SEPARATES THE TWO
refav1 RECIPES THAT PRODUCE BIT-IDENTICAL DRIVING, BY 199× (MEASURED 2026-09-03, dev-box CPU,
0 GPU; same raw as ROW 1).** `D-REFAV1-PAIRED-READ-VOID` measured the fp32 incumbent and the clean
epoch producing trajectories identical to **0.0 m on 140/140 windows** and concluded that *"the
discriminating instrument at low steps is the IMAGINATION and the COST SURFACE, not the deployed
trajectory."* On the **same 140 windows** that is now MEASURED, not argued: κ-axis `F_sep` 5,962.49
vs 29.91 (**199×**), a-axis 3,831.33 vs 102.29 (**37×**), latent scene spread (`tac`) 4.2492 vs
0.089146 (**48×**), max absolute displacement 0.23172 vs 0.041338 (**5.6×**). `F_sep` is
**scale-invariant** — asserted by an output-scale control that multiplies every prediction by c and
requires F, ρ and cos unchanged — so the 199× is structural, not a units artefact. It is also not
harness noise: an independent second pass reproduced every one of these numbers bit-for-bit, and
the arms' **student architectures are identical** (the 7,291,136-parameter gap is the frozen EMA
teacher, which this read never touches), so the 199× is attributable to the training recipe
(`ema_targets` + bf16 + TF32) and to nothing else in the model. ⇒ **A refav1
recipe comparison at low step counts is admissible on the anchored read and is NOT admissible on a
T1 planning number**, until `baseline_won_frac` falls and the trivial-profile fraction drops below
1.0. ⚠️ **This row does NOT say which checkpoint is better.** Two readings are open and this package
adjudicates neither: (a) the clean epoch's field is 48× more compressed with a 199× less consistent
action response — a worse imagination; or (b) the incumbent's very large F reflects a
near-deterministic, low-within-variance response that may itself be degenerate. That belongs to
`2026-09-02-refav1-target-space-collapse` and `2026-09-02-refav1-ema-inflation`, for which the
48× scene-spread collapse measured here is direct evidence.
**H-GS8-1 (instrument half) is SUPPORTED on refav1:** anchored on its own zero-action prediction
the response is STRUCTURED (linear, sign-consistent, far above a measured null) while remaining
IMMATERIAL in magnitude (0.4 % / 1.3 % vs the 5.95 % bar) — exactly the distinction the banked
`actdiv` magnitude ratio cannot draw. Instrument validated by **34 CPU tests**
(`stack/tests/test_actdiv_anchored.py`), of which the load-bearing ones read known values through
the REAL predictor path on a random-init tiny `RefAV1`: an action-blind model (`operative.act[0]`
zeroed) reads **exactly 0.0** with `ss_between = ss_within = 0`; a planted **κ-only** response reads
LAT-INSENSITIVE-REFUTED; and a planted **accel-only** response reads **LAT-INSENSITIVE-CONFIRMED** —
without that last test the instrument could only ever refute the hypothesis, never confirm it.

## ROW 3 — the stranding (operational, for the standard)

⚠️ **C-GS89-STRANDED — THE COMPLETED GS-8/GS-9 RUNS, INCLUDING BOTH refav1 READS, EXISTED ONLY IN A
WORKTREE FOR ~90 MINUTES AND WERE ONE `rm -rf` FROM GONE (2026-09-03).** The agent that built
`actdiv_anchored.py` and `transition_probe.py` finished four checkpoint reads, the v7
`--actdiv-compat` pass and the transition probe (`run_*.status`: all `rc=0`, 03:47–04:52 Berlin),
wrote them to `C:\Users\Admin\tanitad-wt\_gs89_pkg\`, and was killed by a model limit before
copying them into the repo — the repo package held **only `SPEC.md` and two empty directories**.
17 files (~730 KB) have been copied in and staged. ⇒ Reinforces AGENT_OPERATING_STANDARD rule 1,
and adds a sharper form of it: **the deliverable manifest must be written incrementally, as each
artefact lands, not at the end of the turn** — a turn that ends early ends before its manifest.
Two artefacts from that rescue are **banked and unread**: the GS-9 transition probe
(`raw/transition_probe_v7.json`, §4 of that package's RESULT is still *pending*) and the v7
`--actdiv-compat` read.

---

## Applying these

1. `Project Steering/GOALS_AND_CLAIMS.md` — append ROW 1 (correcting the
   `D-REFAV1-STEP1000-READ` amendment's lateral clause), ROW 2, ROW 3.
2. The amendment to `D-REFAV1-STEP1000-READ` currently reads *"**H-REFAV1-LAT-INSENSITIVE**
   (hypothesis, P2 family): refav1's imagination is insensitive to the lateral action"*. That
   clause is now **REFUTED by its own pre-registered test** and needs the correction inline, not
   only a new row further down.
3. `RETRACTION_LOG` — no retraction is required: the hypothesis was correctly stated, correctly
   pre-registered, and correctly tested. It simply came out the other way. The instrument was
   built to be able to confirm it and did not.
