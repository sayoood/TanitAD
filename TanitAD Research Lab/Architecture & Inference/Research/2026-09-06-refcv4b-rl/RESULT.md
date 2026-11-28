# RL post-training on refcv4b — the primary readout now EXISTS, and it fails a broken fan

**status: INSTRUMENT LANDED · ARM PRE-REGISTERED · ⛔ NOT RUN.**
⛔ **No RL arm has executed on refcv4b. No RL result exists.** Both GPUs are occupied (a
sibling's refcv5 ~44 h training on the A40; an eval on Thor) and nothing here touched
either. Everything below is **0 GPU**.

**date:** 2026-09-06 · **stream:** RL post-training (Arch + Inference FlyWheel) ·
**pre-registration:** `SPEC.md` in this package (`E-DDA-3c`, hypothesis `H-DDA-5`).

---

## 0. The answer in one paragraph

The RL rung's **primary endpoint did not exist**: `B2` of the 2026-09-05 package records
that `fan_floor@k`, fan-collision-vs-replay and diversity are missing and are *"a hard
blocker on the primary readout [that] must land before any arm"*, and a content-verified
scan of **1,180 files (0 unreadable)** confirms it. It exists now — `fan_floor.py`,
`rl_fan_floor.py`, **31/31 tests**, run end-to-end on a **real 240-window × 128-candidate
banked fan**. Building it surfaced two defects the endpoint would have carried into every
future RL result, and one of them is measured rather than argued: ⭐⭐ **a fan destroyed by
a deliberate 90 % mode collapse reports `fan_floor@128` +9.614186 — a large POSITIVE
"gain" — while its diversity falls 90 %.** ⇒ `fan_floor@k` quoted alone would have
**passed a knowingly-broken policy with a wide margin**, so the instrument now **refuses**
to return a floor verdict without diversity on the same windows. The arm itself is
pre-registered with both outcomes, a seed replicate, and three deliberate regressions —
and it is ⛔ **still gated on `G-REWARD`, which is measurably unreachable as written** and
needs a PI/Master-Mind ruling on its population, not a goalpost move by me.

---

## 1. What was blocking, and what is blocking now

| # | blocker | before | after this turn |
|---|---|---|---|
| **B2** | `fan_floor@k` / fan-collision / diversity **do not exist** | ⛔ hard blocker on the primary readout | ✅ **CLOSED** — §2, §3 |
| **B4** | prefix tables are refcv3 literals | closed for refcv4b by the sibling | ✅ unchanged (refcv5 still needs the `agent_gate` ruling) |
| **B1** | ⛔⛔ `G-REWARD` unreachable as written | OPEN | ⛔ **STILL OPEN — PI / Master Mind.** §5 |
| **B5** | both GPUs busy | OPEN | ⛔ **STILL OPEN** |
| **B6** | ⭐ **no refcv4b FAN dump exists** | *not previously named* | ⛔ **NEWLY NAMED** — §4 |

---

## 2. The instrument, and the two defects it is built around

`stack/tanitad/rl/fan_floor.py` · `stack/scripts/rl_fan_floor.py` ·
`stack/tests/test_rl_fan_floor.py` — **31/31 green**, **106/106** across the RL suite
(`test_rl_fan_floor` + `test_rl_rewards` + `test_rl_advantage` +
`test_rl_refc_adapter_robust`).

**Why the FLOOR is the endpoint at all.** DD-v2's published effect is on the **raw fan
before any selector** (Tab. 3, 20 trajectories): PDMS **@1 93.5 → 94.9 (+1.4)**, **@5 84.3
→ 91.1**, **@10 75.3 → 84.4 (+9.1)**, diversity **42.3 → 30.3 (−28 %)**. The gain is
**6.5× larger at the fan's floor than at its top**. A readout on the selected path — which
is what every REF-C eval has reported — is structurally blind to it.

### 2.1 ⛔⛔ Defect A — `fan_floor@k` is MAXIMISED BY MODE COLLAPSE

Replace every candidate by the fan's own mean and the floor rises to meet the top while
the fan stops being a fan. **MEASURED, not argued** (§3). Consequence, enforced in code
rather than in a convention: `floor_verdict` **raises** if diversity was not measured on
the same windows, and returns `COLLAPSE-SUSPECT` when the floor rises while diversity
falls > 30 %. ⭐ DD-v2's own stage sits in that band (−28 %) — which is exactly why it is a
**named outcome** and not a failure: the trade must be *visible and priced*.

### 2.2 ⛔⛔ Defect B — `@k` is an ORDER STATISTIC over K, and this programme has already
lost a decision to that

The 2026-09-05 retraction (*"the RL arm's objective is the 2.11× selection gap"*,
root-cause class **A BEST-OF-N STATISTIC READ AS A SKILL GAP**) leaves a binding rule: any
min/max-over-N quantity is reported beside the **same statistic from a RANDOM selector at
the same N**. `summarise_fan` therefore computes `rand_best@k` **unconditionally**;
`assert_equal_k` **refuses** a paired comparison across fans of different K; `fan_quantile`
is the K-free form.

⭐ **And the control is not decorative on real data.** On the banked refcv3 fan the two
curves **cross between k = 10 and k = 32**: the fan's 32nd-best candidate scores
**0.358781** while a *random* best-of-32 scores **0.585294**. They answer different
questions — and without both printed side by side, the first is exactly the number someone
reads as "headroom a better ranker could close".

---

## 3. ⭐⭐ MEASURED — the gate CAN fail a knowingly-broken policy, and the floor alone CANNOT

Run on the real banked fan **240 windows × 128 candidates**
(`fan_bank_base_240w.npz`, `…/2026-09-05-veto-only-fan-safety/raw/`), 0 GPU.
Deliberate regression: `collapse_fan(fan, 0.9)` — every candidate shrunk 90 % toward its
window's own mean path.

| half of the control | MEASURED | reads |
|---|---|---|
| diversity | `d_diversity` **−4.089131 m**, rel **−0.9000** | **FIRED** |
| floor | ⭐ `d_fan_floor@128` **+9.614186** | **FIRED** |

⭐⭐ **The second row is the finding.** A destroyed fan reports a **large positive floor
gain**. Had `fan_floor@k` shipped alone — which is how DD-v2's Tab. 3 is most naturally
read — this arm would have **passed**. The pair does not: `floor_verdict` classifies it
`COLLAPSE-SUSPECT`, and `SPEC.md` §5 makes diversity a **co-condition of PASS**, not a
footnote.

### 3.1 The refcv3 baseline the refcv4b arm will be read against (MEASURED, T0-fan)

Quality = the composed rule-based reward
(`5·progress + 5·headway + 2·comfort + 12·collision + 12·feasibility`, normalised) — DD-v2's
PDMS **structure** with its map half removed, since PhysicalAI has no map.

| metric | refcv3 @ 40,284 | `rand_best@k` (the control) |
|---|---|---|
| `fan_floor@1` | 0.605852 | 0.307374 |
| `fan_floor@5` | 0.578698 | 0.458830 |
| `fan_floor@10` | 0.542388 | 0.523787 |
| ⭐ `fan_floor@32` | **0.358781** | **0.585294** |
| `fan_floor@64` | 0.265387 | 0.598599 |
| `fan_floor@128` | 0.165178 | 0.605852 |
| `fan_diversity` | **4.543479 m** | — |
| `fan_endpoint_std` | 4.308519 m | — |
| `fan_collision_all` | 0.024870 | — |
| `fan_collision_top32` | 0.008984 | — |
| `fan_collision_top8` | **0.000000** | — |

⚠️ `fan_collision_top8 = 0.000000` beside `fan_collision_all = 0.024870` is
`D-RL-GEN-COLLIDES-1` in one line: **the selector's view is clean and the generator's is
not.** The RL stage acts on the second population; a `sel_contact`-style number cannot see
it.
⚠️ Tier **T0-fan** on every row. Evidence class **MEASURED (ours)**;
artifacts `raw/refcv3_fanfloor_240w.json`, `raw/refcv3_fanfloor_geom.json`.
⚠️ `fan_diversity` is **our definition** (mean pairwise RMS waypoint distance). DD-v2 does
not publish its formula, so **only the direction and relative change** are comparable to
their 42.3 → 30.3 — never the level.

### 3.2 ⚠️ Two limits found by RUNNING the control, disclosed rather than smoothed over

1. **The control's floor half is INERT on a bank with precomputed quality columns.**
   Collapsing the geometry cannot move a column computed before the collapse, so the first
   run reported diversity −0.90 and floor **0.000000** — and printed `OK`. The driver now
   **detects this and prints `PARTIAL` with the reason**; the complete control needs a
   quality recomputed from the waypoints (`--quality geom_feasibility`), which is what
   produced the table in §3.
2. ⚠️ **A units bug of mine, caught before it was quoted.** That geometry quality first
   used `rewards.kinematics`' default `DT_S = 0.1` on a fan banked on the **0.5 s grid** —
   inflating every acceleration **25×** and reporting `fan_floor@128 = −268.25` instead of
   **−13.87**. The driver now derives `dt` from the waypoint count and **states it in the
   provenance string**. Same family as the `alat`/`kappa` unit trap: a correct formula
   under the wrong units reads exactly like an answer.

---

## 4. ⛔ B6 — NEWLY NAMED: no refcv4b FAN dump exists

MEASURED 2026-09-06: every banked refcv4b artifact carries the **SELECTED** paths only.
The openloop dump schema is `g · os · ha · ha0 · os_navshuf · os_navzero · oracle_sel ·
v0 · ws · eid · clip_index` — **no candidate axis**. The only banked *fan* in the repo is
refcv3's `fan_bank_base_240w.npz`.

⇒ The instrument is ready and its input is not. Banking a refcv4b fan is **one inference
pass, ~10 min on the dev-box 4060**, and it is step 1 of `SPEC.md` §8. It is listed as a
blocker rather than done because the 4060 is at 100 %.

---

## 5. ⛔⛔ THE LAUNCH GATE — `G-REWARD`, and why I am not moving it

`D-RL-GREWARD-UNREACHABLE-1` (sibling, MEASURED): over **2,400 weightings** of the four
admissible ranking terms the **minimum achievable rate is 0.3840** against a ≤ **0.30**
ceiling — **unreachable by every admissible reward on this corpus**. Cause, measured on
the same 6,089 windows: `progress` fires on 95.70 %, `comfort` 95.70 %, `headway` 76.79 %,
`feasibility` 2.89 %, `robust_contact` 1.03 %, `collision` 0.13 % ⇒ the rate is decided by
terms that are near-symmetric between a human and a constant-velocity path. Corroborated
from the other direction: the **GRPO advantage is identically zero in 92 % of windows**.
**The windows the gate scores are overwhelmingly windows on which RL cannot learn either.**

⛔ **I am not re-specifying it and I am not launching past it.** The signal-window form
reads **0.0635 [0.0000, 0.3158]** — but on **n = 63 windows / 6 EPISODES**, **post-hoc**,
with an upper bound that **straddles the ceiling**. That is *necessary, not sufficient*,
and a gate re-specified after seeing which specification passes is a moved goalpost.

### 5.1 ⭐⭐ SO I RAN THE NEXT LEVER INSTEAD — and it REFUTES the "population" repair, and names the real defect

⛔ **PRE-REGISTERED BEFORE ANY RATE WAS COMPUTED** and stated in the probe's own docstring
(`raw/greward_power_curve.py`): the population is a **PROPERTY OF THE SCENE**
(`gt_time_gap_min_s`, the human's own minimum time gap, recorded before any candidate
exists); the threshold ladder **(∞, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0) s is fixed in
advance and every rung is reported**; and ⛔ **the probe selects nothing.**

MEASURED 2026-09-06, 0 GPU, on the banked per-window rows (**6,089 windows / 73
episodes**), repaired reward `progress 0.3 · collision 1.0 · headway 0.3`, episode-cluster
bootstrap `n_boot` 4,000:

| population | n_win | n_eps | **RATE** `hold ≥ human` [CI] | vs ≤ 0.30 | **MEAN GAP** `hold − human` [CI] |
|---|---|---|---|---|---|
| all | 6,089 | 73 | 0.4418 [0.3687, 0.5163] | **FAIL** | **−0.011239 [−0.017039, −0.005306]** |
| ≤ 5.0 s | 4,642 | 62 | 0.4550 [0.3738, 0.5426] | **FAIL** | −0.007177 [−0.013258, −0.001170] |
| ≤ 3.0 s | 3,121 | 52 | 0.4854 [0.3809, 0.5826] | **FAIL** | −0.009265 [−0.017359, −0.002833] |
| ≤ 2.5 s | 2,690 | 40 | 0.4866 [0.3702, 0.5957] | **FAIL** | −0.011259 [−0.021052, −0.003940] |
| ⭐ ≤ 2.0 s | 1,576 | 29 | ⭐ **0.5482** [0.4128, 0.6529] | **FAIL** | −0.012704 [−0.029915, −0.002008] |
| ≤ 1.5 s | 698 | 17 | 0.5201 [0.3754, 0.6226] | **FAIL** | −0.010984 [−0.029911, +0.001700] |

⭐ **CHANNEL CONTROLS, all three exactly 0.000e+00** against the banked panel: the
all-window rate (**0.441780259484316**), the all-window mean gap
(**−0.011239379601720929**) and the `frozen` rate (**0.04549187058630317**), plus the
per-term attribution reproducing to 1e-15. The rows and the composition are the banked
object.

⛔⛔ **FINDING 1 — THE "POPULATION" REPAIR IS REFUTED, IN THE OPPOSITE DIRECTION.** On a
scene-property ladder the rate does not fall toward 0.30; it **RISES MONOTONICALLY with
conflict, 0.4418 → 0.5482**, and **every rung's CI lies entirely above the ceiling**.
Restricting to lead-conflict scenes makes `G-REWARD` **harder, not easier**. The ≤ 2.0 s
rung carries **29 episodes — 5× the banked signal population's 6** — so this is not an
underpowered null.
⚠️ **SCOPE, stated so this is not over-read:** this does **not** reproduce or refute the
banked `0.0635 on 63 windows / 6 episodes` figure, whose population needs `robust_contact`
— a term these banked rows do not carry. It is an **independent, larger-n, pre-registered
test of the same repair IDEA**, and that idea does not survive it.

⭐⭐ **FINDING 2 — AND THIS IS THE ONE THAT MATTERS: `G-REWARD`'s DEFECT IS ITS STATISTIC,
NOT ITS POPULATION AND NOT ITS THRESHOLD.** Read the two right-hand columns together. At
**five of six rungs**, including all-windows on 73 episodes, the **RATE fails** while the
**MEAN GAP is SEPARATED IN THE HUMAN'S FAVOUR** — the reward *does* prefer the human, with
an interval excluding zero, on every population. The two statistics **disagree in sign**
because the distribution is skewed: **the human wins big on a minority of windows and
loses small on a majority**, and a rate cannot see magnitude.

⇒ That is why **2,400 weightings could not reach 0.30**: no reweighting of terms can make
a rate see a magnitude it is structurally blind to. `D-RL-GREWARD-UNREACHABLE-1`'s
measurement was right and its diagnosis — *"the gate's POPULATION is wrong"* — is the part
this refutes.

⭐ **FINDING 3 — the coordinate to repair, named.** Per-term weighted mean gap
(`hold_v0 − human`; positive = the trivial path is rewarded more):

| term | all windows | ≤ 3.0 s | ≤ 2.0 s | ≤ 1.5 s |
|---|---|---|---|---|
| ⭐ `progress` | −0.005600 | **+0.003593** | **+0.006368** | **+0.008251** |
| `headway` | −0.004818 | −0.011256 | −0.015899 | −0.013504 |
| `collision` | −0.000821 | −0.001602 | −0.003173 | −0.005731 |

⭐⭐ **`progress` FLIPS SIGN as the scene tightens.** In close following the human slows for
the lead and a constant-velocity path does not, so **`progress` PAYS hold-v0 for not
slowing down** — exactly where the safety question is live. `headway` correctly punishes
it harder there, and the two terms fight; the rate counts the fight's outcome per window
and loses the magnitudes. This is `D-RL-PROGRESS-COMPOSED-1` (*"`progress` binds in
isolation and not in the composed reward"*) seen from the rate side, and it is expressible:
`progress` is ego geometry, so a conflict-aware reference (e.g. normalising by the
*achievable* speed given the lead rather than by `v0 · horizon`) is a term the corpus can
compute.

⛔ **I SELECT NOTHING.** Adopting the mean-gap statistic after seeing that it passes would
be the goalpost move this whole entry exists to avoid. What I have added is that the PI's
question is now **decidable with evidence**, and that it is a **different question** than
the one escalated yesterday.

⇒ **ESCALATED, restated:** *`G-REWARD` compares two paths with a **RATE** on a distribution
where the informative windows are a skewed minority. Should the gate's statistic be the
rate, the mean gap, or a magnitude-aware rank test — and should `progress` be re-referenced
so it stops paying the trivial path in close following?* Owner: PI / Master Mind. 0 GPU.

---

## 6. Escalations raised here, not written into a doc for someone to find

1. ⛔⛔ **`G-REWARD`'s population** — §5. **Owner: PI / Master Mind. This is the launch
   gate.**
2. ⛔⛔ **THE ARM THAT WOULD RUN TODAY IS THE DELIBERATE REGRESSION.** MEASURED 2026-09-06
   from the source, not the design: **`refcv3_adapter.sample_offsets` has no control space
   at all** — it scales the **offset waypoints** (`scale = mean.abs() * cfg.noise_scale`),
   and `control` / `rollout_unicycle` / `a_lon` / `alat` appear **ZERO times** in that file
   (content-verified, 9,078 bytes, non-zero control on the same read); and
   `rl_refcv3_min.py` has **no fan-dump path** (0 flags matching `fan`/`bank`/`dump`).
   ⇒ launching the `rl` arm today runs **metre-space noise = `reg_metre`**, the
   pre-registered deliberate regression, and would table it as the hypothesis.
   ⚠️ **This corrects my own earlier wording in this package** ("two missing flags"): it is
   a missing **capability**, not a missing flag.
   ⭐ **So I built it, as a NEW module rather than an edit to a file this stream does not
   own:** `stack/tanitad/rl/control_space.py` + `stack/tests/test_rl_control_space.py`
   (**15/15 green**) — the DD-v2 two-scalar policy in control space, σ floors kept distinct
   (exploration **0.04**, likelihood **0.10**), clamped to the envelope, re-rolled through
   the programme's single `rollout_unicycle`. ⭐⭐ **Its central claim is tested against its
   own counterexample:** `envelope_violation` reads **exactly 0.0** on a control-space
   sample **even at 25× the published σ**, while the metre-space arm **does** violate — so
   *flyable by construction* is discriminating, not a tautology. ⛔ Stated limit: the clamp
   is a non-injective pushforward, so `logp` is the density of the two **scalars**, not of
   the trajectory — the same class of object as the published estimator, quoted as a
   **scale-perturbation** gradient and never as an exact policy gradient.
   ⇒ **ESCALATION, now one wiring line:** call `control_space.sample_control_space` from the
   RL stage's sampling path behind a recorded config field, and add a fan-dump path.
   **Owner: Training FlyWheel / Master Mind.**
3. **B6** — bank a refcv4b fan (~10 min, one inference pass). **Owner: whoever holds the
   4060 next.**
4. ⚠️ **`fan_safety.py` has no `--bank-fan` path either**; the refcv3 fan bank was produced
   by a sibling's ad-hoc script. The flag belongs in the tool. **Owner: Benchmarks/Eval.**

---

## 7. Manifest

| artifact | where it lives |
|---|---|
| this report | `…/2026-09-06-refcv4b-rl/RESULT.md` (repo) |
| the pre-registration | `…/2026-09-06-refcv4b-rl/SPEC.md` (repo) |
| ⭐ the instrument | `stack/tanitad/rl/fan_floor.py` (repo) |
| ⭐ the driver | `stack/scripts/rl_fan_floor.py` (repo) |
| ⭐ its tests, 31/31 | `stack/tests/test_rl_fan_floor.py` (repo) |
| refcv3 baseline panel | `…/2026-09-06-refcv4b-rl/raw/refcv3_fanfloor_240w.json` (repo) |
| the complete collapse control | `…/2026-09-06-refcv4b-rl/raw/refcv3_fanfloor_geom.json` (repo) |
| ⭐ the control-space sampler | `stack/tanitad/rl/control_space.py` (repo) |
| ⭐ its tests, 15/15 | `stack/tests/test_rl_control_space.py` (repo) |
| banked primary (already banked, `--cited-by` extended and content-verified) | `2512.07745`, sha256 `076ce47e0b0a323c9bb0072432e7f1219ccdcf645b8ff6db172f1bb9efb85023` |

Nothing lives in only one place. ⛔ **No arm has run; no RL result exists.**

### 7.1 ⚠️ SUITE STATE, reported as measured rather than as required

**My tests: 46/46 green** (`test_rl_fan_floor` 31 + `test_rl_control_space` 15), and
**106/106** across the pre-existing RL suite.

⛔ **But `pytest -q` on the whole `stack/` is NOT green on this branch, and I will not
imply otherwise:** **171 failed · 6,446 passed · 61 skipped · 28 errors** in 1,090 s.

⭐ **None of them is mine, and that is established positively rather than assumed:**
1. **Content-verified import isolation** — `fan_floor` is referenced by **exactly my three
   files** across **1,135 `.py` files read with 0 unreadable**; `control_space` likewise;
2. **zero** of the 199 `FAILED`/`ERROR` lines match `test_rl_`, `fan_floor` or
   `control_space` — they fall in **35 unrelated files** (SAM3, overlays, sitclf, xodr,
   perception floor, v6 ladder, refcv5 preflight …), three siblings' live streams;
3. the only two plausibly-linked tests were run individually: `test_runbook_commands`
   fails on `train_v6_staged.py`'s `--horizons` refusal (a sibling's file, staged by them),
   and `test_text_encoding_is_explicit` names **ten pre-existing offenders and none of
   mine**. The one test that reads `GOALS_AND_CLAIMS.md`
   (`test_withheld_bank.py`) is **12/12 green** after my edits.

⚠️ **Also disclosed:** the suite cannot be run from the repo root reliably — it died twice
at pytest's own `pyproject.toml` read with `OSError [Errno 22]` while the G: mount was
wedged, which reads exactly like a broken suite and is a broken mount. The mount was
recovered under the standing PI authorisation before the run above completed.
