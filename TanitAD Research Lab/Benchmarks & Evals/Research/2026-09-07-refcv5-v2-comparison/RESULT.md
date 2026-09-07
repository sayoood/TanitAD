# refcv5-v2 vs refcv4b — ONE COMMAND, and the CONTROL that proves it works

**Benchmarks & Evals · 2026-09-07 · T1 (self-action open loop) on every number**

⭐ **DID THE HARNESS REPRODUCE refcv4b's PUBLISHED NUMBERS? — YES on everything the
verdict rests on, with ONE marginal cell that flipped and one arithmetic assumption of my
own that the instrument caught and I have retracted below.**

`refcv5-v2` was training on the A40 throughout (step ~2,600 of 40,284). ⛔ **That GPU was
not touched.** Every roll here ran on the **dev-box RTX 4060** — one full 141-episode roll,
25 min 20 s, 1,529 s of forward.

---

## 1. THE ONE COMMAND

```sh
bash "C:/Users/Admin/refcv5cmp/compare.sh" \
     --ckpt   C:/Users/Admin/refcv5v2_final/ckpt_40284_FINAL.pt \
     --config C:/Users/Admin/refcv5v2_final/config.json \
     --tag    refcv5-v2
```

Canonical source: `scripts/` in this directory. `C:/Users/Admin/refcv5cmp/` is the
off-Drive working copy — **G: cannot RUN the stack** (Errno 22 mid-import). Step 1 of the
command re-syncs the code from the repo and writes `_SYNC_MANIFEST.json` (per-file md5,
674 files), so the run can always say **which bytes** it executed.

Add `--skip-roll` to re-analyse a landed roll at **zero GPU cost**. Proven: `--analyze-only`
reproduced the roll-time analysis **bit-identically on CPU** (`os` 0.2363000000 both ways).

---

## 2. ⛔ THE CONTROL — published (A40) beside reproduced (dev-box RTX 4060)

Both read the same checkpoint, md5 **`99b573e8277d94a5e3bfbf630cb4d751`** (verified locally
*and* against the pod's own `pod_md5.txt`), on the **141-episode / 4,823-window B1 v7.2
EVAL grid**. Grid identity was asserted before any number: `n_windows`, `n_episodes`,
`dt_s`, `horizon_steps`, `gt_key` all MATCH, and **eid + window origins + GT are
bit-identical to the banked refcv3 @40,284 dump**.

### 2.1 The headline arms

| arm | PUBLISHED (A40) | REPRODUCED (RTX 4060) | |
|---|---:|---:|---|
| `os` | 0.2975 | **0.2965** | −0.0010 |
| `ha` | 0.2996 | **0.2996** | ✅ **EXACT** |
| `ha0` | 0.6723 | **0.6723** | ✅ **EXACT** |
| `ha0_ext` | 0.2874 | **0.2874** | ✅ **EXACT** |
| `os_navshuf` | 0.3013 | 0.3006 | −0.0007 |
| `os_navzero` | 0.3928 | **0.3926** | −0.0002 |

⭐ **`os` 0.2965 and `os_navzero` 0.3926 are EXACTLY the Thor re-roll values** recorded in
`…/2026-09-07-refcv5-v2-compose/README.md` §5.6. The RTX 4060 is a **third independent
hardware reading the same two numbers**, which settles the open item: the third-decimal
`os` difference is **cross-hardware argmax tie-breaking over 117 anchors**, not an
instrument defect. A number carries its roll.

### 2.2 Every metric of every model-free arm, across all four families

**66 of 66 rows bit-EXACT** for `ha` / `ha0` / `ha0_ext` — ADE, speed MAE, speed bias,
target-speed accuracy, along MAE, distance-keeping n / headway / time-gap / TTC, heading,
yaw-rate, curvature MAE, cross-track, curvature step counts, mask exclusions, tactical
accuracy and κ, goal FDE, goal bearing. ⇒ **the surface, the grid, the join and the
analysis path are IDENTICAL.**

Model arms (`os` / `os_navshuf` / `os_navzero`): largest continuous deviation **0.0416**
(an `os_navzero` mean-min-TTC of 25.39 s → 25.43 s, 0.17 %); window tallies move by at most
5 windows of 4,823.

### 2.3 ⛔ The paired margins — the cells the verdict rests on

| cell | PUBLISHED | REPRODUCED | verdict |
|---|---|---|---|
| **`os − ha`** | **−0.0021 [−0.0178, +0.0154]** not sep. | **−0.0032 [−0.0185, +0.0142]** not sep. | ✅ AGREE |
| **`os − ha0_ext`** | **+0.0101 [−0.0050, +0.0273]** not sep. | **+0.0091 [−0.0059, +0.0260]** not sep. | ✅ AGREE |
| `os − ha0` | −0.3748 [−0.4319, −0.3203] sep. | −0.3758 [−0.4331, −0.3212] sep. | ✅ AGREE |
| `os_navzero − ha0_ext` | +0.1054 [+0.0874, +0.1241] sep. | +0.1052 [+0.0876, +0.1238] sep. | ✅ AGREE |
| `ha − ha0` | −0.3727 [−0.4346, −0.3161] | **−0.3727 [−0.4346, −0.3161]** | ✅ **bit-identical** |
| `os − ha` speed MAE | +0.0368 [+0.0197, +0.0557] sep. | +0.0359 [+0.0191, +0.0547] sep. | ✅ AGREE |
| **`os − os_navshuf`** | −0.0038 [−0.0076, **+0.0001**] **not sep.** | −0.0041 [−0.0078, **−0.0004**] **sep.** | ⛔ **DISAGREE** |

**13 of 14 verdict-bearing cells agree.** ⛔ The one that does not is
`paired_os_minus_navshuf.ade_m`: the published interval's upper bound sat at **+0.0001 m —
one tenth of a millimetre from zero** — and the 1.1 mm cross-hardware shift carried it
across. **I am reporting this, not adjusting it.** It is also a live demonstration of the
brief's own rule: the harness classifies that cell **`⛔ NOT YET MEASURED — separated but
FRAGILE`** (its interval reaches back to within 25 % of the point estimate) and refuses to
call it a direction. Neither reading of it changes any published conclusion — `os_navshuf`
is a nav control, not a bar.

### 2.4 The cross-checkpoint control

```
refcv4b − refcv3 @40,284, same 4,823 windows / 141 episodes
  PUBLISHED   −0.1444 [−0.1647, −0.1227]  separated
  REPRODUCED  −0.1455 [−0.1655, −0.1240]  separated      ✅ same verdict, intervals overlap
  MODEL-FREE CONTROL  ha   delta −0.0000000000  [0, 0]   2 of 4,823 windows differ at float32
  MODEL-FREE CONTROL  ha0  delta  0.0000000000  [0, 0]   0 of 4,823 windows differ
```

The 0.0011 gap is the same cross-hardware `os` offset as §2.1. ⚠️ The published record
claims `ha` differs on **0/4,823**; across hardware it differs on **2/4,823** at float32,
with the paired delta still exactly 0 to 10 dp.

### 2.5 Other published values reproduced

* **`anchor_acc` 0.5275, chance 1/117 = 0.008547** — the corrected full-grid value, exactly.
  ⛔ The contaminated 0.0993 and the stale `1/128` string appear nowhere in this harness.
* **STRATEGIC** (which `refcv3_arm.py`'s own `four_families` reports UNAVAILABLE — this
  harness computes it from the decisions dump instead of dropping it):
  route acc **0.7791 [0.7248, 0.8318]** vs published 0.7786 [0.7205, 0.8324]; κ **0.4864**
  vs 0.4852; **n 3,622 / 128 eps — exact**; nav echo index **0.6422** vs 0.6405 (computed
  through `_ROUTE_TO_NAV = {0:1,1:0,2:2}`, never the raw index comparison that produced the
  2026-09-06 type error); `route_pred` identical under nav-shuffle **1.000000** — the
  structural identity, exactly.
* **LATERAL, masked**: curvature MAE **0.008150** vs published 0.008097; straight-line floor
  `ha0` **0.006802 EXACT**; echo control `ha0_ext` **0.003712 EXACT**; ratio **1.198 — refcv4b
  tracks the road WORSE than a plan that never steers**, the published shape defect,
  reproduced. `excluded_below_min_ds` 1,146 of 13,558+1,146 steps at `min_ds = 0.25 m`.
* **Per-class recalls** (vacuity gate): `turn_left` 0.813 (n 251), `turn_right` 0.889
  (n 396), `brake_stop` 0.539 (n 631), `accelerate` 0.506 (n 538) — no class is vacuous.

### 2.6 ⚠️ RETRACTION — my own instrument check was wrong, and the instrument caught it

The first version of the constant-arm check asserted `|curvature_bias_1pm| == curvature_mae_1pm`
for `ha0`, reasoning that a straight-line arm's curvature error is exactly `−κ_gt`. **That is
false** and it FAILED on the first real panel (MAE 0.006802 vs bias 0.001672): `κ_gt` is
**signed**, so `MAE = mean|κ_gt|` while `bias = −mean(κ_gt)`, and left and right turns cancel
in the second. Equality holds only on a corpus that turns one way.

Replaced with two checks that are actually closed-form, and both now PASS:
* the **real masked estimator** run over `ha0`'s own paths returns `max|κ| == 0.0` **exactly**
  over 13,713 valid pairs (the analytic pin: a straight line has exactly zero curvature on
  every grid);
* `|bias| ≤ MAE`, which is an identity.

The wrong version is recorded in the source rather than deleted.

---

## 3. THE VERDICT MACHINE — proven by running it on refcv4b

The full harness, end to end on the dev-box refcv4b roll:

```
[0]  INSTRUMENT ................ ALL PASS (5/5)
[0b] ANTI-ECHO GATE 1 .......... vs ha0_ext  0/4 slots pass, relative margin by slot
                                 [−0.0047, −0.0747, −0.0716, −0.0035]  ← NEGATIVE at every horizon
                                 vs ha       0/4 slots pass
                                 vs ha0      4/4 slots pass
[0c] VACUITY GATE .............. any_vacuous = False

BAR-REFCV5V2-1  os − ha0_ext  +0.0091 [−0.0055, +0.0254] not separated  ⇒ ⛔ FAIL
BAR-REFCV5V2-2  os − ha       −0.0032 [−0.0179, +0.0133] not separated  ⇒ ⛔ FAIL
BAR-REFCV5V2-3  arm delta (informational)                               ⇒ not the bar

MULTIPLICITY: 4 of 9 cells separated = 44.4 % vs a pure replicate's 14.3 % (6/42)
```

⇒ **the harness independently re-derives refcv4b's published conclusion — "beats refcv3 by
−0.1455 m separated, and still only TIES the do-nothing baselines"** — and writes the FAIL
itself. The bar is `refcv5_compare.py::BAR_PRIMARY`, registered 2026-09-07 while refcv5-v2
was at step ~2,600, and it converges with the sibling compose agent's independently
pre-registered §5.5.

`selftest_verdict.py` proves by **mutation** (not inspection) that all 12 branches are
reachable: PASS on a real win, FAIL on a tie, FAIL on a loss, FAIL on a sub-margin win,
REFUSAL of a panel missing `ha0_ext`, REFUSAL of a grid mismatch, REFUSAL of an empty dump
as a mount flap, and both instrument-failure branches. **A verdict machine that can only say
PASS is a rubber stamp; one that can only say FAIL is a wall; from the outside on a single
run they look identical.**

---

## 4. THE FOUR-FAMILY TABLE IT EMITS

Per arm (`os`, `ha`, `ha0`, `ha0_ext`), **never pooled**, each family with its estimator and n:

```
ARM os   TIER T1   ADE 0.2965 m [0.2697, 0.3280]          ← one row of four, never the result
  LONGITUDINAL  speed_MAE m/s · bias · target_speed_acc@0.5 · along_MAE m
                distance-keeping/TTC: status · n of N windows · mean_headway_min m ·
                mean_time_gap_min s (n) · mean_min_TTC s (n_closing) · ⚠️ censoring note
  LATERAL       heading deg · yaw_rate deg/s · cross_track m
     ⭐ CURVATURE_MAE 1/m  |  straight-line floor (ha0)  |  ratio  →  ABOVE / BELOW the floor
        MASKED estimator: n_steps · excluded_below_min_ds (min_ds m)
  TACTICAL      status · n · lateral_decision acc/κ · longitudinal_decision acc/κ
                per-class recall with support (the vacuity gate)  · goal FDE m · bearing MAE
  STRATEGIC     route_acc [CI] · κ · n / eps · chance 1/3 · nav_echo_index ·
                route_pred identical under nav-shuffle
                (as_declared_by_refcv3_arm: UNAVAILABLE + reason + n, kept beside the
                 computed value — a family is never silently dropped OR silently replaced)
```

Every separated cell then prints which variance question its interval answered:

```
episode draw  : ANSWERED — paired episode-cluster bootstrap, cluster = episode, n_boot 2000
training run  : ⛔ NOT MEASURED — one seed per arm (H-ESTIM-SEED-1 OPEN; replicate FP 6/42)
inference run : CLOSED BY CONSTRUCTION (refcv4b) / ⛔ NOT MEASURED (refcv5-v2, --sampler ddim)
```

⭐ The sampler flag is read out of the run's **own `config.json`**, so refcv5-v2's stochastic
DDIM sampler is stamped automatically. **This gap is new and did not exist for refcv4b.**

---

## 5. What this harness WILL and WILL NOT be able to say about refcv5-v2

> **WILL:** whether refcv5-v2's own selected 2 s path, on 4,823 held-out windows from 141
> episodes, beats the do-nothing controls `ha` and `ha0_ext` by a margin that survives
> resampling the EPISODES **and** clears a 10 % relative bar at every horizon slot — reported
> per family (longitudinal, lateral incl. masked curvature against the straight-line floor,
> tactical with per-class recalls, strategic route prediction), with the refcv4b arm delta
> beside it as context.
>
> **WILL NOT:** anything about **driving**. This is T1 open loop — no simulator, no vehicle,
> errors never compound, the world never reacts. It cannot say whether a margin survives a
> second **training** seed (one seed per arm), nor — new for refcv5-v2 — whether it survives a
> second **inference** draw, because `--sampler ddim` is stochastic at inference where refcv4b
> was deterministic by construction. It cannot say anything about **route-free** operation
> (both arms are ORACLE-NAV). And a PASS is **entry to the register, not a published result**.

---

## 6. Deliverable manifest

| artifact | where | only one place? |
|---|---|---|
| `scripts/compare.sh` — the one command | `repo:…/2026-09-07-refcv5-v2-comparison/scripts/` | no (also `C:/Users/Admin/refcv5cmp/`) |
| `scripts/refcv5_compare.py` — four families, gates, bars, verdict | same | no |
| `scripts/selftest_verdict.py` — 12 mutation cases | same | no |
| `scripts/repro_check.py` — the published-vs-reproduced control | same | no |
| `scripts/sync_repo.py`, `scripts/arm.sh` | same | no |
| `raw/REPRO_CHECK.{txt,json}` — the control, field by field | `repo:…/raw/` | no |
| `raw/DRYRUN_refcv4b.{txt,json}` — the full panel + verdict | `repo:…/raw/` | no |
| `raw/refcv4b_devbox.json` + `raw/refcv4b_devbox.log` — the roll | `repo:…/raw/` | no |
| `raw/refcv4b-40284-devbox4060-dump.tar.gz` — per-window baseline dump, md5 `c638603ead015d5547b8e88f0148e159` | `repo:…/raw/` | **was local-only; now banked** |
| `raw/_SYNC_MANIFEST.json` — md5 of all 674 executed files | `repo:…/raw/` | no |
| off-Drive run root (14 GB of episodes junctioned, not copied) | `C:/Users/Admin/refcv5cmp/` | yes, by design — rebuildable |

### Test state on the dev box
`taniteval/tests/test_four_families_curvature_analytic.py` **14/14**; the estimator/family
suite (`curvature_analytic`, `four_families_dt`, `ci`, `lateral`, `estimator_closeout`,
`eval_four_families_tool`) **113 passed, 5 skipped, 0 failed**. The 4 initial
`estimator_closeout` failures were a **mirror-completeness** defect — top-level
`taniteval/*.py` were not synced — and were fixed by making the mirror faithful, never by
excusing the failure.

### Open work items (named, not skipped)
* **GATE 2 / GATE 2b** of `assert_not_echoing` need a live model in the loop. Without GATE 2b
  the verdict is `STRUCTURAL_ONLY`, which is **not** an anti-echo pass.
* **Second training seed** — the only thing that answers `H-ESTIM-SEED-1`.
* **Second inference draw** for refcv5-v2 — new, and cheap: it is a re-roll of the same
  checkpoint with a different sampler seed.

---

## 7. THE LANDING SNAG, CLOSED ON THE LIVE CONFIG — `n_anchors` is RESOLVED, never defaulted

<!-- SECTION7-N-ANCHORS-RESOLUTION-2026-09-07 -->

**Benchmarks & Evals · 2026-09-07 · ⛔ ZERO GPU on the A40 (refcv5-v2 was training at step
~16,850 / 40,284 throughout). Everything below ran on the dev-box CPU / RTX 4060.**

The harness was built and validated against **refcv4b**, which has neither `--tac-goal-tok-head`
nor `--sel-refined`. refcv5-v2 has both, and its `config.json` was reported as carrying **no key
named `n_anchors` anywhere in its 206 leaves**. The question put to me was whether the harness
then **fails at the worst moment** or **silently falls back to a default**, reintroducing the
stale `1/128` denominator on the one arm that most needs to be right.

### 7.1 ⭐ WHAT IT ACTUALLY DID — ESTABLISHED BY RUNNING IT, NOT BY READING IT

**Neither.** The answer is a third thing, and it splits by path.

| path | what happens TODAY when the bank size is absent | evidence class |
|---|---|---|
| **ROLLOUT** (`compare.sh` step 2 → `refcv3_arm.load_model`) | It never looks for a config key called `n_anchors` at all. It rebuilds the model **through the trainer's own `build_parser` + `_pin_trainer_cfg` on the recorded `argv`**, so the bank size comes from `--n-anchors 117`. **MEASURED on the LIVE refcv5-v2 config: rebuild OK, `cfg.core.anchors.n_anchors = 117`, model builds (108,257,502 params), `core.decoder.anchors [117, 8, 2]`, and `cross_check_config` PASSES on all six keys including the `param_breakdown` that carries `tac_goal_tok_head = 11286`.** | MEASURED · `raw/LIVE_REFCV5V2_RESOLUTION.json` |
| **ROLLOUT, with `--n-anchors` stripped from argv** | ⛔ **The rebuild SILENTLY DEFAULTS to the v3 config's `128`** (`args.n_anchors = None`). It is **caught** — but by the **`param_breakdown` cross-check**, whose message is a ten-key parameter diff; with `param_breakdown` absent it is caught by a `state_dict` **size mismatch**. Both are true refusals **pointing at the wrong field**. | MEASURED · both messages reproduced in `raw/SELFTEST_N_ANCHORS_RED.log` |
| **ANALYSIS** (`--analyze-only` on a dump whose `manifest.json` has no `model.n_anchors`) | ⛔ **It does NOT fail and does NOT default: it emits `chance: None`, REFUSES the selection profile, and completes**, writing a full result JSON with every other number intact. | MEASURED · a real 4,823-window re-analysis, 29 s, CPU |
| **COMPARISON** (`refcv5_compare.py`, before this change) | ⛔ **Blind.** It read `n_anchors` **nowhere** (0 hits, against a same-breath control of 17 `def `s) and **never printed the anchor-selection row at all** — so a `chance: None` upstream would have produced a panel and a VERDICT with the TACTICAL family's goal/anchor-selection half **silently absent**. | MEASURED |

⇒ The real exposure was **not** a fabricated `1/128`. It was (a) a refusal that **names the wrong
field**, and (b) a four-families **silent drop** downstream. Both are now closed.

### 7.2 WHERE THE BANK SIZE COMES FROM, AND WHY

⭐ **The CHECKPOINT TENSOR is authoritative.** `core.decoder.anchors` is a **persistent buffer**
(`refc.py:1384`), the deployed selection is an argmax over exactly that tensor, and `anchor_acc`'s
denominator **is** its first dimension. Everything else *describes* it. Reading it is cheap —
**0.03 s via `torch.load(mmap=True)` on a 428 MB checkpoint, MEASURED**.

⛔ **And the fact was never missing — it was under another name.** The live config carries it
**three times**: `/anchors/shape[0] = 117`, `/anchors/controls_shape[0] = 117`, and
`argv --n-anchors 117`. *(Absence found at one location is not absence; here the search was for
the NAME.)*

`resolve_n_anchors()` — one in `taniteval/tools/refcv3_arm.py`, one in `scripts/refcv5_compare.py`
— takes every witness, requires **all present witnesses to AGREE**, and:

* **unresolvable** ⇒ REFUSAL **naming every field probed** and saying why there is no default;
* **two witnesses disagreeing** ⇒ REFUSAL naming **both**;
* **the rebuilt config disagreeing with the witnesses** ⇒ REFUSAL naming **`--n-anchors`**, in
  front of the `param_breakdown` and `state_dict` checks that used to speak first.

These witnesses are **independently authored** (a saved tensor, a builder's recorded shape, an
operator's flag), which is what makes the cross-check worth running — re-deriving a value from its
own producer measures determinism, not correctness.

### 7.3 THE NESTED READS

Every fact is read from its **nested** path and **corroborated against `argv`**; a contradiction is
a REFUSAL, and an agreed absence is a corroborated `False` — never a default.
(`config_facts()` in `scripts/refcv5_compare.py`.)

| fact | read from | live refcv5-v2 | refcv4b |
|---|---|---|---|
| `sel_refined` | `/selection/sel_refined` + `--sel-refined` | **True** | False |
| `sel_score_emitted` | `/selection/sel_score_emitted` + `--sel-score-emitted` | **True** | False |
| `sampler_ranks_the_fan` | `/selection/sampler_ranks_the_fan` | **True** | NOT DECLARED |
| `tac_goal_tok_head` | `/seams/tac_goal_tok_head/built` + `--tac-goal-tok-head` | **True** | False (`/seams` is `null`) |
| `goal_str` | `/seams/goal_str` + `--goal-str` | True | True |
| `anchor_v0_conditioned` | `/anchors/v0_conditioned` | True | True |
| `anchor_control_units` | `/anchors/control_units` | `alat` | `alat` |
| `sampler` | `/seams/sampler` + `--sampler` | **`ddim`** | absent |
| ⛔ `n_anchors` | `/anchors/shape[0]`, `/anchors/controls_shape[0]`, `argv`, **and the ckpt tensor** | **117** | 117 |

`sampler_is_stochastic` now reads `/seams/sampler` **and** `argv` and refuses on a contradiction —
it was argv-only, which is one flag away from stamping a stochastic run "deterministic".

⭐ **NEW IN THE PANEL — the TACTICAL family's anchor-selection half**, which the comparison used to
omit entirely (a missing metric is a WORK ITEM, not an excuse):

```
anchor_selection   acc 0.5275 [0.4883, 0.5664]  chance 1/117 = 0.008547  lift 61.72x  n 4823
    bank size from checkpoint:model['core.decoder.anchors'].shape[0] | distinct 51 of 117,
    modal 67 (0.489), entropy_ratio 0.452
```

### 7.4 ⛔ THE CONTROL — the refcv4b re-validation, AFTER the change

| the landing this harness must keep reproducing | published here (§2) | re-run 2026-09-07 after the change |
|---|---|---|
| model-free rows bit-exact | **66 of 66** | **66 of 66, `REPRO_CHECK.json` md5-IDENTICAL** (`031784a48ea6c1c2761d9ac4be0aa974`) |
| `refcv4b − refcv3` (os) | −0.1455 [−0.1655, −0.1240] sep. | **−0.1455 [−0.1655, −0.1240] sep.** |
| trivial tie `os − ha` | −0.0032 [−0.0179, +0.0133] not sep. | **−0.0032 [−0.0179, +0.0133] not sep.** |
| trivial tie `os − ha0_ext` | +0.0091 [−0.0055, +0.0254] not sep. | **+0.0091 [−0.0055, +0.0254] not sep.** |
| BAR-1 / BAR-2 | ⛔ FAIL / ⛔ FAIL | **⛔ FAIL / ⛔ FAIL** |
| paired cells agreeing with the A40 publication | 13 of 14 | **13 of 14** (same marginal `os − os_navshuf`) |

⭐ **`diff` of the whole rendered panel, before vs after: exactly 2 insertions per arm block and
nothing else** — the two new `anchor_selection` lines. The control is not merely "still passing";
it is **byte-identical except for the rows that were added**.

Pinned suites, dev box: `stack/tests/test_refcv3_arm.py` **32 passed**; the estimator/family set
(`refcv3_ablations`, `refcv3_ha0_ext_shared`, `refcv3_route_nav_alignment`,
`four_families_curvature_analytic`, `ci`, `estimator_closeout`, `eval_four_families_tool`,
`refcv3_arm_astar_geometry`) **118 passed, 1 skipped**.

### 7.5 ⭐ THE MUTATION PROOF — the refusal is REACHABLE, and the test is not vacuous

`scripts/selftest_n_anchors.py`, **20 cases, every expectation a LITERAL in the case table** (the
number `117`, the string `0.008547`, the field names) — ⛔ never the shape *"if missing: expect a
refusal; else: expect a pass"*, which is an expected value of whatever the code does.

* **GREEN: 20 of 20 PASS** (`raw/SELFTEST_N_ANCHORS.log`).
* **RED: reintroduce the historical defect** — a `resolve_n_anchors` that returns the rebuilt
  config's number and refuses nothing — and **10 of 20 go RED**, i.e. every arm-tool case
  (`raw/SELFTEST_N_ANCHORS_RED.log`). The 10 compare-side cases stay green, correctly: the
  regression was installed in the arm tool only.
* ⭐ **The discriminating pair.** Cases **B2/B3** do not ask *"did it refuse?"* — the pre-fix code
  refused too. They require the refusal to **name `n_anchors`** and to **NOT** contain
  `param_breakdown` (B2) or `size mismatch` (B3). On the regressed copy both assertions fail with
  the exact pre-fix messages, which is the proof that a *"did it refuse"* test would have been
  green on the defect.
* **B4** is the other direction: a config whose `/anchors/shape` **lies** (`[128, 8, 2]` while
  argv, the build and the weights all say 117) must be REFUSED. It is; on the regressed copy the
  call **returns**.

End to end, at the harness level (`raw/E2E_REFUSAL_AND_RESOLUTION.log`), with both expectations
written down before running:

```
E2E-1  dump manifest WITHOUT model.n_anchors, no --new-config, no --new-ckpt
       EXPECTED  refusal naming all five witness fields, exit 1
       OBSERVED  "n_anchors IS UNRESOLVABLE - every witness is absent: ..."      exit 1
E2E-2  the SAME stripped dump, plus --new-ckpt
       EXPECTED  resolves 117 from the TENSOR, chance 1/117 = 0.008547, exit 0
       OBSERVED  chance 1/117 = 0.008547, "bank size from checkpoint:...anchors.shape[0]"  exit 0
```

⚠️ E2E-2 also shows what the panel does **not** do: the arm's own `selection_profile` stays
**REFUSED** and says so. Resolving the denominator for the panel does not repair an upstream
refusal, and the row prints that in words.

### 7.6 ⛔ THE SECOND SNAG, FOUND WHILE CLOSING THE FIRST — THE ROLL WAS UNSEEDED

Chasing "what else will bite" turned up a second landing defect, and it is not cosmetic.

**MEASURED:** `refc.py:1926` draws `eps = torch.randn_like(x0_n)` **at eval**, deliberately — the
source says so in as many words (*"a loop whose output cannot move is not a sampler"*) — and
`taniteval/tools/refcv3_arm.py` called `torch.manual_seed` **nowhere**. ⇒ two runs of the landing
command on the **same** refcv5-v2 checkpoint would have produced **different numbers**, with **no
record of which draw** produced the banked one. refcv4b never exposed it: its decoder zeroes its
noise outside training, so the harness was validated on the one arm that cannot show the bug.

**Fixed in the same turn.** `--infer-seed` (defaults to `--seed`) seeds torch / numpy / random
before the rollout and is **recorded in the dump manifest** beside `sampler` and
`stochastic_at_inference`, so a dump now says which inference draw it is. `compare.sh` passes it,
so the replicate is one flag:

```sh
bash compare.sh --ckpt <refcv5-v2 ckpt> --tag refcv5-v2-seed1 --infer-seed 1
```

⭐ **The control that proves it cannot move the banked landing:** every other eval-time draw in
this stack is behind `self.training` (`refc.py:2223 / 2998 / 3075`, `refc_v3.py:1429`) and the arm
calls `.eval()`, so a deterministic arm makes **no draws at all**. Asserted, not assumed —
**refcv4b re-rolled at `--infer-seed 0` and `--infer-seed 7` is BIT-IDENTICAL: 4 of 4 episode
dumps by md5, and `os` / `ha` / `ha0` / `ha0_ext` ADE exact to 10 dp** — with the manifest
stamping `sampler: "none", stochastic_at_inference: false` on both
(`raw/SEED_INVARIANCE_refcv4b.log`).

### 7.7 IS IT READY FOR THE REAL CHECKPOINT?

**Yes — both snags are closed, and what remains is an experiment, not a defect.** The bank size
resolves **117** off the live config with **no checkpoint at all**; the rebuild + cross-check pass
on the live config today; `--new-ckpt` makes the tensor the primary witness the moment the file
exists; and the roll is now seeded and stamped. ⚠️ What is left is that refcv5-v2's `--sampler
ddim` **is** stochastic, so the panel stamps `inference_run: ⛔ NOT MEASURED` until a **second roll
at another `--infer-seed`** is done — one 25-minute dev-box roll, now a single flag. A one-seed
separated CI remains **necessary, not sufficient**, and on a sampler arm an effect smaller than the
seed spread is not an effect.

### 7.8 Deliverable manifest (this section)

| artifact | where |
|---|---|
| `taniteval/tools/refcv3_arm.py` — `resolve_n_anchors()` wired into `load_model` before any expensive work; `--infer-seed` seeding the rollout and recorded in the manifest | repo (also `C:/Users/Admin/refcv5cmp/repo/`, `C:/Users/Admin/tanitad-wt/`) |
| `scripts/refcv5_compare.py` — `config_facts()`, `resolve_n_anchors()`, `ckpt_n_anchors()`, `anchor_selection_block()`, `--new-ckpt`/`--base-ckpt` | repo `…/2026-09-07-refcv5-v2-comparison/scripts/` (also `C:/Users/Admin/refcv5cmp/`) |
| `scripts/compare.sh` — passes `--new-ckpt` and `--infer-seed`, and the refcv4b reference config/ckpt when present | same |
| `raw/SEED_INVARIANCE_refcv4b.log` — refcv4b bit-identical across two inference seeds | repo `…/raw/` |
| `scripts/selftest_n_anchors.py` — the 20 mutation cases | same |
| `raw/SELFTEST_N_ANCHORS.log` (green) · `raw/SELFTEST_N_ANCHORS_RED.log` (deliberate regression) | repo `…/raw/` |
| `raw/E2E_REFUSAL_AND_RESOLUTION.log` | repo `…/raw/` |
| `raw/LIVE_REFCV5V2_RESOLUTION.json` — what the harness resolves off the live config today | repo `…/raw/` |
| `raw/RECHECK_refcv4b_devbox_vs_refcv4b.txt` · `raw/RECHECK_REPRO_CHECK.json` | repo `…/raw/` |
