# PAIRED_OPENLOOP — the cross-model **open-loop** comparison, and the six ways it goes wrong

**Instrument:** `taniteval/tools/paired_openloop.py` · **pin:** `stack/tests/test_paired_openloop.py`
(23 tests) · **owner:** Benchmarks & Evals FlyWheel · **written:** 2026-09-03 · **compute: 0 GPU**
(it reads banked dumps and nothing else).

It answers the PI's question of 2026-09-02 — *"compare the open and closed loop performance of both
refav1 and refcv3"* — for the half that is measurable today: **the open-loop half**. It consumes the
dumps produced by `refav1_arm.py` and `refcv3_arm.py`; it does not roll either model and it never
touches Thor or a pod.

---

## 0. ⛔ THE VOCABULARY, FIRST, BECAUSE IT DECIDES WHAT THIS TOOL IS

**PI ruling, 2026-09-02** (`Project Steering/VOCABULARY.md`, `t1_eval.py`, commit `32e319149`):

> *"The open loop performance corresponds to the fact that the AI model … is not controlling the
> vehicle in the world. The fact that the predictor is consuming the output of the planner of the
> world-model-based system is ALSO OPEN LOOP because the trajectory of the model is not affecting the
> new ego data (this will be fed from the eval ego data). Closed loop means the trajectory is
> controlling the vehicle in the simulation, e.g. AlpaSim or a real test vehicle."*

⇒ refav1's `cl` and refcv3's `os` are **BOTH OPEN LOOP**. Both are handed the same recorded ego data;
neither controls anything. ⭐ **That is exactly why they are comparable at all** — a question that was
previously open is now settled by the ruling, not by an argument. ⛔ The words *"closed loop"* never
describe an arm in this tool's output, and `test_the_record_and_the_markdown_never_say_CLOSED_LOOP`
pins it.

---

## 1. What the tool computes, and what makes the statistic admissible

`D-HF-COMPARABILITY` (`Project Steering/GOALS_AND_CLAIMS.md`) binds the claim:

> **the DIFFERENCE OF EACH ARM'S MARGIN OVER THE SAME TRIVIAL FLOOR, PER FAMILY, with a paired
> episode-cluster bootstrap over the shared `eid` set** — `cl − ha0` for refav1 against `os − ha0`
> for refcv3, ⛔ **never `cl` against `os` as levels**.

So the tool emits, per metric:

| column | what it is |
|---|---|
| `A abs` / `B abs` / `floor` | FULL-SET pooled means on the shared windows (the brief asks for absolutes; they are printed, they are not the claim) |
| `A−floor`, `B−floor` | each arm's **margin over `ha0`**, paired episode-cluster bootstrap |
| `(B−f)−(A−f)` | **the cross-model statistic**, paired over the same episodes |

⭐ **And the algebraic identity is STATED, not hidden.** When the floor is verified bit-identical on
both sides (control C2), `(os − ha0) − (cl − ha0)` **is** `os − cl`, exactly. The margin framing is
still what gets reported — it is what stays interpretable when the floor is *not* identical, and it
is what the register binds — but a reader is owed the identity rather than an implication that two
different quantities were computed. `test_the_cross_statistic_equals_the_level_difference_when_the_
floor_is_shared` pins both the equality and the sentence.

**Four families, separately, never pooled** (Sayed, binding 2026-08-02): LONGITUDINAL / LATERAL /
TACTICAL / STRATEGIC, plus ADE/FDE as **one row of four families and never "the result"**. The
family verdict is `SPLIT` when the two models separate in opposite directions on different metrics —
⛔ the tool does not manufacture an overall winner.

---

## 2. ⭐ THE CONTROL THAT PROVES THE WINDOW KEY — this is the whole job

The two dumps come from different tools, different caches (**DINOv3 fp8 tokens** vs **v2ep pixels**)
and different GT functions (`metric_dynamics.gt_ego_waypoints` vs `refb_labels.waypoint_targets`).
They are joined on `(clip_id, RAW 10 Hz frame of the window origin)`:

| side | `ws` is | RAW origin | provenance |
|---|---|---|---|
| refav1 | a **cache index** `t` (5 Hz view) | **`2·ws`** | `refav1_arm.gt_waypoints` opens `f0 = 2 * t`; `join_lead_block` keys on `(clip_id, RAW frame 2t)`. **Not in the manifest** — derived from source, then proven below |
| refcv3 | a **PROVIDER index** (`t + window − 1`) | **`ws + (n_stack − 1)`** | read from the manifest's own `corpus.frames.provider_to_raw_frame_offset` (`v2_dataset.py:36-38`) |

⛔ **A frame-offset argument cannot be settled by reading code. It is settled by a measurement:**
the two dumps' **GT waypoints and `v0`** must agree on the windows they claim to share. If they do
not, the key is wrong and every number downstream compares different moments in different clips —
so the tool **refuses**.

**MEASURED 2026-09-03**, `t1_dump_ep2` × the stride-1 refcv3 dump, **120 shared windows over 20
episodes** at {1.0, 2.0} s (and again on a 12-window dry run against a different refcv3 dump):

```
GT  max|A−B| = 0.000e+00 m      (expected 0)
v0  max|A−B| = 0.000e+00 m/s    (expected 0)
```

Two independent pipelines landing **bit for bit**. That is the strongest evidence available that the
key is right, and it is why every table in this tool can be read as being about the same moments.

### 2a. The grids are matched in INTEGER RAW FRAMES, never in seconds

refav1 dumps `dt 0.2 s, K 10` → RAW offsets `2,4,…,20`. refcv3's `2s` grid is model slots
`5,10,15,20` → `0.5,1.0,1.5,2.0 s`. The intersection is `{10, 20}` raw frames = **{1.0, 2.0} s**, and
both sides are **index-selected** onto it. ⛔ No resampling, no interpolation. Matching `0.2·k`
against `0.5·j` in floating point is how two grids silently "agree"; integers cannot.

Two consequences the record states rather than buries:

* **the common grid is COARSER than either dump's own.** Every LON/LAT rate is computed on it for
  both arms *and for GT*, so the comparison is fair — but ⛔ **the levels must not be quoted against a
  single-arm read on a finer grid.**
* `four_families._seq_geometry` **prepends the ego origin as step 0**, so a grid whose first instant
  differs from its spacing would make step 0 mean something different from the rest. The tool asserts
  `uniform` **and** `first_step_equals_spacing` and refuses otherwise. Here: first `1.0 s`, spacing
  `1.0 s` ✅.

### 2b. ⚠️ THE STRIDE TRAP, and it is the one that actually bites

refav1's window origins land on RAW `2t` with `t` stepping by the dump's stride; refcv3's on
`ws + 2` with its own stride. **A refcv3 dump at `--window-stride 5` shares ZERO windows with a
refav1 dump at stride 10** — every other gate passes and the intersection is simply empty. MEASURED:
refav1's origins here are `≡ 6 (mod 20)`; a stride-5 refcv3 dump reaches only `≡ 2 (mod 5)`.

⇒ **the refcv3 side of a paired read must be rolled at `--window-stride 1`** (or at a stride that
provably covers the other side's origins). The tool refuses an empty intersection with the modulus
arithmetic printed, so the diagnosis is in the error rather than in someone's head.

---

## 3. ⭐ THE FLOOR — the anchor of the whole table, and three controls on it

`ha0` is constant velocity at the measured `v0` (`a = 0, κ = 0`). It depends on **nothing but `v0`
and the integrator**, which is why it is the ONE arm bit-comparable across two architectures — and
why the action-unit defect cannot touch it: *zero is zero in either unit*.

**The banked refav1 dumps predate the `ha0` arm.** They are not therefore unusable: `ha0` is the one
arm recoverable **without the model**, so the tool DERIVES it from the dumped `v0` through the
programme's own integrator — `refav1_arm.hold_v0_controls` → `refav1_arm.paths_from_controls` →
`refa_v1_plan.unicycle_paths`. ⛔ Not a re-implementation; a second copy would be a second convention.

Three controls, each printing **EXPECTED beside MEASURED**:

| # | control | expected | MEASURED (12-window dry run) |
|---|---|---|---|
| **C1** | the floor reads the **no-information value** `x = v0·t, y = 0` | `0.0` exactly | `dx 7.629e-06 / 1.526e-05 m`, **`|y| 0.000e+00 m`** |
| **C2** | the floor is **bit-comparable across the two architectures** | `0.0` exactly | `1.526e-05 m` |
| **C3** | the DERIVED floor against the DUMPED one, **SAME SIDE** | `0.0` | `1.526e-05 m` |

⚠️ **C3 must be a same-side check.** With one side derived and the other dumped, a cross-side
comparison measures exactly what C2 measures — **one control reported as two**. The tool enforces
the same-side form and says why in the record.

The residual `1.5e-05 m` is float32 round-tripping through the integrator, ~10⁻⁶ relative at this
corpus's speeds. It is reported, not rounded away.

---

## 4. ⛔ THE SIX GATES (`D-HF-COMPARABILITY`), each a refusal with its measured value

| gate | condition | how it is enforced |
|---|---|---|
| **G1** | different window grids, unasserted | the common instants are derived in **integer RAW frames**; a grid sharing < 2 instants, or a non-uniform one, is refused |
| **G2** | refcv3's **oracle-selected** `traj` against refav1's arm | `oracle_sel` / `cl_oraclegoal` are refused **as model arms**, by name, before anything is read |
| **G3** | a cross-tier comparison | the two arms' tier stamps must match; the **OPEN tier ruling** (does the doctrine admit a model that consumes NO actions at T1?) travels onto the record as `UNRULED` |
| **G4** | either arm degenerate | the trivial profile **and** the selection profile run BEFORE any family row; a degenerate arm makes the read **VOID, not negative** |
| **G5** | the unrepaired steer→curvature reading | `action_units` is read from each manifest and reported. ⭐ It affects `ha` **only** — `ha0` is exactly zero, which is 0 in either unit |
| **G6** | no shared floor | `ha0` must exist or be derivable, and must pass C1–C3 |

### 4a. Why the profiles print first, and why VOID ≠ negative

MEASURED 2026-09-03 (`D-REFAV1-PAIRED-READ-VOID`): a full refav1 read shipped *"lateral planning
already beats holding"* — every number correctly computed — while **every one of the 140 `cl` plans
was a straight constant-speed line**. A family table answers *how far off*; it can never answer
*what shape*. And the selection profile exists because the trivial profile is **blind to an anchor
model's characteristic degeneracy**: a model can pick ONE anchor on every window while every
trajectory it emits is distinct, so `trivial_frac` reads `0.0000` on a maximally degenerate arm.
Both gates are pinned by test.

⚠️ **`identical_to` is a 1e-9 m comparison of raw arrays and cannot resolve two arms built through
different arithmetic** (a float32 integrator against a float64 reference; two forwards at different
batch sizes — REFCV3_ARM.md §2.8 measures that cross-call floor at `5.96e-07 m`). A zero there is not
proof that two arms differ. **The VOID gate does not depend on it** — it is decided on
`constant_velocity_frac`, a property of one arm's own shape.

---

## 5. The four families, and how STRATEGIC exists here at all

A trajectory-only dump leaves STRATEGIC **UNAVAILABLE** by construction: `four_families`'s decision
family needs `route_pred`/`route_gt` in the window dict, and `t1_eval`'s whitelist never forwards
them. This tool reaches them through each dump's **decisions sidecar**, which both tools write with
the **same key names** (`route_label`, `route_pred_nav_{true,shuffled,zero}`, `lat_label`, …).

Two things had to be got right before those rows are admissible:

1. ⭐ **The floor for a classifier is not `ha0`.** `ha0` is a trajectory and has no head, so a margin
   against it is undefined. The no-information value for a categorical row is the **MAJORITY-CLASS
   rate on exactly those windows** — and like `ha0` it is **shared across both models by
   construction**, because it is a property of the LABELS alone. That is what carries the
   margin-over-floor framing into the two decision families.
2. **The two tools do not label the same windows — and on `route_label` they DISAGREE.** MEASURED on
   the 120-window read: refav1's sidecar carries a `route_label` on **40** and refcv3's on **105**;
   on the **35** both label, **5 DISAGREE** (`1->2` x2, `2->1` x3). `lat_label`/`lon_label` are
   **identical on all 40**, and `nav_cmd`/`nav_valid` are identical too. ⇒ the tool **scopes** each
   declared row to the windows BOTH sides label and prints the counts; a coverage difference is an
   honest restriction, but a genuine *disagreement* REFUSES the row rather than scoring two different
   questions. See §13.1 — it is a repo defect, not a model result.

### 5a. The nav-echo controls — without them the strategic row is inadmissible

Nav is an **INPUT**. Flagship v1's route head scored **1.0000** as an exact bijection of the nav it
was fed. So the tool emits, per head and per model, the paired `true − navshuffled` and
`true − navzero`, plus the cross difference of those:

* **`nav_shuffled`** withholds the **pairing** (the marginal is preserved exactly — the model still
  sees a plausible token everywhere): *is the model using **this** window's nav?*
* **`nav_zero`** withholds the **signal**: *what happens when the oracle nav is not there* — i.e.
  **deployment**.

⛔ They are not interchangeable and a shuffle cannot stand in for a zero (BACKLOG R39). ⚠️ And for
refcv3 the nav-zero arm is a **LOWER BOUND** on nav dependence: at the core, "no nav" collapses onto
`one_hot(0) = follow` rather than being removed (REFCV3_ARM.md §2.8).

---

## 6. Statistical power, stated rather than implied

The bootstrap resamples **episodes**, because windows inside one clip are strongly dependent. It can
only ever resample the episodes actually present. Below **10** episodes the tool stamps
`power.adequate = false`, prints a warning, and adds to *what this does not establish*:

> ⛔ NOT evidence of NO effect where a row reads 'not separated' … below 10 episodes such a row is
> UNDERPOWERED — a statement about the sample, not about the models.

⛔ `overlapping_holdout_se` appears nowhere: it biases the **point estimate** bidirectionally
(−6.67 % to +11.69 %), up to a **sign flip**. And two overlapping single-arm CIs are **not** a null
result while two disjoint ones are **not** the paired test — only the paired difference interval
decides. The record carries that reading rule.

---

## 7. Invocation

```bash
python taniteval/tools/paired_openloop.py \
  --a-dump <refav1 dump dir> --a-name refav1 --a-arm cl --a-extra cl_navshuf --a-extra ha \
  --b-dump <refcv3 dump dir> --b-name refcv3 --b-arm os --b-extra os_navzero --b-extra os_navshuf \
  --floor ha0 --n-boot 2000 --seed 0 \
  --out taniteval/results/paired-openloop-refav1-vs-refcv3-<UTC>.json \
  --md  <package>/RESULT.md
```

* `--a-extra` / `--b-extra` add arms that are **reported beside** the model arm (its own controls);
  only `--a-arm` and `--b-arm` enter the cross-model statistic.
* ⭐ **`--b-extra os_navzero` is not optional in practice**: `os_navzero − ha0` is the
  **deployment-relevant** margin, because refcv3's v7.2 nav token is an ORACLE (provenance
  ego-future) that will not exist at deployment.
* The refcv3 dump must be rolled at **`--window-stride 1`** — see §2b.

---

## 8. What this tool does NOT do, and who owns it instead

| not here | owner |
|---|---|
| the refcv3 single-arm table on `ckpt_30000` | `refcv3_arm.py` → `taniteval/results/refcv3-30k-openloop-*.json` |
| the refav1 single-arm table, the planner block, the WM T0 diagnostic | `refav1_arm.py` |
| **distance-keeping / headway / time-gap / TTC** (the second half of LONGITUDINAL) | each side's own lead-block join — ⛔ **a WORK ITEM here, not a pass** |
| the video, the HF card, the leaderboard, the binding-KPI suite | other streams |
| ⛔ **any closed-loop number** | nobody — it requires AlpaSim or a real vehicle and does not exist |

## 9. Known gaps

1. **Distance-keeping is absent from this table.** `refav1_arm.join_lead_block` and
   `refcv3_arm.lead_block_common_grid` each join the banked B1 lead block on their own grid; the
   paired tool does not re-derive it. Reading it here would need a third common-grid join. Stated as
   a WORK ITEM in `what_this_does_not_establish`, never as a silent drop.
2. **The refav1 side is limited by what has been rolled.** `cl` costs ~7.3 min/episode at the
   default search, so the intersection is bounded by the refav1 dump's episode list, not by refcv3's.
3. **The tier ruling is OPEN** (does the doctrine admit as T1 a model that consumes NO actions?).
   The tool requires the two stamps to *match* and carries the ruling as `UNRULED`; the
   margin framing means the numbers are correct whichever way the PI rules.
4. `--floor` is settable but only `ha0` has the bit-comparability property. Any other floor voids the
   register's condition 6, and the tool does not pretend otherwise.

---

## 10. ⛔ THE CLOSED-LOOP HALF — what this tool is NOT, and where it lives

The PI asked for **open AND closed** loop. This tool is the **open-loop half**, and the record it
emits carries a `closed_loop` block plus a dedicated RESULT.md section so the open-loop table is never
presented as the whole answer.

* **This tool cannot measure closed loop** — it reads banked `[N, K, 2]` trajectories scored against
  recorded GT, and nothing in a dump can become a closed-loop number because the world never
  responded.
* ⭐ **But the harness EXISTS**, so "closed loop" here is a **NOT-YET-RUN**, not a NOT-POSSIBLE.
  **VERIFIED at source 2026-09-03**: `stack/experiments/alpasim-gsplat/closedloop_drive.py:518-533`
  steps a kinematic bicycle from the model's own `(steer, accel)` —
  `dyaw = v/WHEELBASE*tan(steer)*DT`, `T_new = T_ego @ D`, `v += accel*DT` — and then **re-renders**
  the next observation from the resulting ego pose (`transport.render(T_ego @ Ts_cam, ...)`). The
  next observation is a consequence of the model's own output: the PI's definition, met.
* It has already produced a published panel (flagship v1 vs REF-C base, 9 starts x 50 ticks on Thor,
  437 paired windows, paired episode-cluster bootstrap).

Pinned by `test_the_record_carries_a_CLOSED_LOOP_section_marked_NOT_MEASURED`.

## 11. ⚠️ THE CORPUS-IDENTITY CONTROL — two arms named "b1-v72" are not thereby the same corpus

`--a-run-config` / `--b-run-config` point at each run's OWN `config.json`; the control reads the
parity block and the train cache from them and reports `VERIFIED_IDENTICAL` / `DIFFER` /
**`UNVERIFIED`** with the fields it found and the fields it did not.

**MEASURED 2026-09-03 on the real runs — `UNVERIFIED`:**

* refcv3's `config.json` records `v2_parity: {parity: false, checked: false, corpus_key: null,
  clips_present: 4572}` — **the run is NON-PARITY by its own record**;
* refav1's trainer publishes **no parity block at all**.

⭐ **What IS established, far more strongly than any config field, is the EVAL side**: the window-key
proof shows the two dumps' GT and `v0` are bit-identical on every shared window, so both arms are
scored on the same moments of the same clips. ⛔ **What is NOT** is (a) that the two runs TRAINED on
the same episodes and (b) that either run's train split is disjoint from the eval clips. Both are
properties of the RUNS, not of this adapter, and neither can be read from a dump — the probe that
settles them is an episode-id set intersection between each run's train cache and the eval split
(`REFCV3_ARM.md` GATE 3).

## 12. ⛔ THE FINAL refcv3 CHECKPOINT IS `ckpt.pt`, AND THE STEP MUST BE ASSERTED

`REFCV3_ARM.md` §3.1 names `ckpt_40284_FINAL.pt`. **That file is never written.**
`refc_v3_train.py:103` has `MILESTONES = (5000, 15000, 20000, 30000)` and 40,284 is not among them;
the final save comes from the `or step == args.steps` branch at `refc_v3_train.py:1191`, which writes
plain **`ckpt.pt`** — which is ALSO the rolling checkpoint.

⇒ **assert the step after loading rather than trusting the path**:

```bash
python -c "import torch,sys; d=torch.load(sys.argv[1],map_location='meta',mmap=True);            assert d['step']==40284, d['step']; print('step OK', d['step'])" <run>/ckpt.pt
```

The tool carries this, the stride-1 requirement and the `os_navzero` requirement in the record's
`refire_preconditions`, so they travel with the artifact rather than living in someone's memory.

## 13. ⭐ TWO FINDINGS THE FIRST REAL RUN PRODUCED, both about the REPO rather than the models

1. **`route_label` DISAGREES between `refav1_arm.py` and `refcv3_arm.py`** on the same
   `(clip_id, RAW frame)`: over 120 shared windows refav1 labels 40 and refcv3 labels 105; on the 35
   both label, **5 disagree** (`1->2` x2, `2->1` x3). ⇒ the STRATEGIC family is **REFUSED**, with its
   reason, its `n` and the confusion counts. **WORK ITEM: reconcile the two derivations.** Until then
   the programme has no cross-model strategic row.
2. **`ha` is not a shared floor, and now it is a number.** Over the same floor and the same 120
   windows, `(refcv3:ha - ha0) - (refav1:ha - ha0)` = **-0.6404 m ADE [-0.9856, -0.3355], separated**.
   The two `ha` arms differ in BOTH the action-unit convention (refcv3 declares `steer`; the refav1
   dump carries no `action_units` field at all) AND the hold rule, so the gap is not attributable to
   either alone — which is exactly why `ha0`, exactly zero in either unit, is the floor.
