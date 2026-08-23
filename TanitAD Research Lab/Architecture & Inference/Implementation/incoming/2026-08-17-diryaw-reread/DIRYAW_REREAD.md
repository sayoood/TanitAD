# `DIR_YAW_RAD` 0.15 → 0.10 — the re-read, and the premise it corrects

**MEASURED 2026-08-17 (ours)** · **0 GPU, CPU only** · `code/gate_operating_point.py` →
`raw/gate_operating_point.json` · dev box `/c/Users/Admin/venvs/tanitad`.

Two earlier passes already answered *"does a verdict move?"*
(`…/Benchmarks & Eval/…/2026-08-15-dir-yaw-gate-reread/`,
`…/Architecture & Inference/…/2026-08-16-dir-yaw-reread/`). This pass was asked for the two
things neither produced — **the provenance of 0.15**, and **the delta under the mandated paired
episode-cluster bootstrap** — and answering the first overturned the framing both passes shared.

---

## ⛔ The answer, first — the threshold should NOT be changed, and the reason is new

| | |
|---|---|
| **1. `DIR_YAW_RAD` is not an instrument constant. It is an alias of the TRAINING-LABEL constant.** | `hierarchy.py:169` → `refb_labels.YAW_TURN_RAD`. The eval gate and the manoeuvre head's own definition of "turn" are **the same number**. |
| **2. So the published sweep is ASYMMETRIC — it moves ONE of the two raters.** | `_gate_sensitivity` re-thresholds the *trajectory*; `man_dir` is passed in **fixed**. At 0.15 the two raters share a definition; at 0.10 they do not. |
| **3. ⇒ κ@0.10 for `maneuver_vs_trajectory` is NOT a fairer read of the same quantity.** | It is a different quantity, with a definitional mismatch introduced by the sweep itself. |
| **4. If 0.15 is mis-scaled — and it is, at 8.3× the human median — the defect is in the LABELS, not the gate.** | Fixing it means re-deriving `maneuver_label` and **retraining**. Re-thresholding the eval alone does not fix it; it hides it. |
| **5. Provenance: author-chosen at first authoring, rationale stated but never measured, never revisited.** | `35adfab`, 2026-07-11. `git log -S` returns **exactly one** commit — the one that introduced it. |

⭐ **New MEASURED number, with the mandated estimator** (canonical val, 881 windows / 40 episodes):
the band the move actually touches is **3.97 % of windows, CI95 [1.93 %, 6.82 %], m = 35** —
paired episode-cluster bootstrap, CI-separated from zero. Every κ envelope in the 2026-08-16 pass
pivots on this number and it had never carried an interval.

⚠️ **Verdict on the backlog item: the evidence does not support changing the threshold.** Not
because the criticism of 0.15 is wrong — it is right — but because 0.10 is not the fix, and a
unilateral eval-side change would silently re-base every published κ against a rater the models
were never trained to match.

---

## 1. What it is and what it gates — from source, not from prose

### 1.1 The alias chain

| step | file:line | content |
|---|---|---|
| the gate | `taniteval/taniteval/hierarchy.py:169` | `DIR_YAW_RAD = rl.YAW_TURN_RAD` |
| what `rl` is | `taniteval/taniteval/hierarchy.py:95` | `import refb_labels as rl` |
| the real constant | `stack/scripts/refb_labels.py:57` | `YAW_TURN_RAD = 0.15` |
| its only gate consumer | `taniteval/taniteval/hierarchy.py:209-215` | `_dir_of(net_yaw)` → `{L,S,R}` |
| its **label** consumer | `stack/scripts/refb_labels.py:107-108` | `cls[dyaw > YAW_TURN_RAD] = TURN_LEFT` |

⭐ **The second consumer is the finding.** `refb_labels.py` is not an eval module — it is the
**training-label** module. The same constant decides (a) which windows the manoeuvre head is
*taught* to call a turn, and (b) which windows the eval calls a turn when scoring it.

Horizons match exactly, so the two really are the same definition and not merely the same number:

* labels: `LABEL_HORIZON = 20` (`refb_labels.py:63`), on `wrap_to_pi(yaw[t+H] − yaw[t])`.
* gate: `GOAL_H = K_MAX = 20`, `DT = 0.1` (`hierarchy.py:113-114,110`), on
  `wrap_to_pi(fut[:, GOAL_H−1, 2] − pl[:, 2])` with `fut = poses[t+WIN : t+WIN+GOAL_H]` and
  `last = t+WIN−1` (`hierarchy.py:589,595`) ⇒ `fut[:, GOAL_H−1] ≡ poses[last+20]`.

### 1.2 The sweep is asymmetric — MEASURED from source

`hierarchy.py:949-950` calls `_gate_sensitivity(A["traj_net_yaw"], A["gt_net_yaw"], man_dir)`.
Inside (`:266-267`), each swept row computes

```
k = _k(md, _cls(tn, g))      #  md = man_dir, NEVER re-derived;  only tn is re-thresholded
```

and `man_dir = np.array([MAN2DIR[int(m)] for m in A["man_pred"]])` (`hierarchy.py:908`) is the
**model's prediction**, from a head whose turn/straight boundary was fixed at training time.

| swept quantity | both raters gated at `g`? | is a sweep meaningful? |
|---|---|---|
| `trajectory_vs_gt_kappa` | ✅ yes — `_cls(tn,g)` vs `_cls(gn,g)` | ⭐ **YES.** A genuine free-parameter sensitivity. **Publish it as a sweep, never as a point.** |
| `frac_gt_turning`, `distributions.{trajectory_dir, gt_dir}` | ✅ yes | **YES** — corpus properties. |
| **`maneuver_vs_trajectory_kappa`** | ⛔ **no** — `md` fixed at the training definition | ⛔ **NO.** Lowering `g` alone manufactures disagreement on the band `(g, 0.15]`. |
| `kappa_turn_subset` (same block) | ⛔ no, and **not swept at all** | ⛔ instrument gap — §4. |
| `commanded_route_vs_trajectory` | ⛔ no, and worse — see §1.4 | ⛔ **NO.** |

### 1.3 ⛔ For `--v2` arms there is NO gate value that makes the two raters commensurable

`classify_maneuver_v2` (`refb_labels.py:321-347`) **contains no net-yaw threshold at all**. Its
turn test is a **curvature** gate, `|dyaw/arc| ≥ CURV_TURN_MAN_PER_M = 1/R_TURN_M ≈ 0.01667 /m`
(`refb_labels.py:275,287`). Its own docstring states the intent verbatim: a gentle highway curve
stays `lane_keep` *"even when its net dyaw exceeds v1's 0.15 rad"* (`refb_labels.py:327-328`).

So for a v2-labelled arm the declaration rater thresholds **1/m** and the execution rater
thresholds **rad**. They are different physical quantities. The mismatch is **not removable by
choosing a better `DIR_YAW_RAD`** — it exists at 0.15 as much as at 0.10.

Which arms, from the registry (the only quotable source for model facts):

| arm | label family | source | matched at 0.15? |
|---|---|---|---|
| `flagship4b-speedjerk-30k` (**deployed v1**) | **v1** net-yaw | `MODEL_REGISTRY.md` §1.2 exact command — no `--v2`, no `--labels-v2` | ✅ **matched** |
| `flagship-v1arch-v2bal-30k` | **v1** net-yaw | §1.9: *"Every `v2_*` lever in its own `config.json` is `false`"* — `v2_labels` named in that list | ✅ **matched** |
| `flagship-v2corpus-30k` | **v2** curvature | §1.7 exact command carries `--v2`; §1.6 records `--v2` *"implies `--speed-input`, `--labels-v2`"* | ⛔ **mismatched at every gate** |
| `flagship4b-v3enc-30k` | **v2** curvature | §1.6.x exact command carries `--v2 --staged-levers` | ⛔ **mismatched at every gate** |

⚠️ **Consequence for the paper's headline tactical claim.** `TANITAD_PAPER.md:1868` quotes the
collapse κ **0.253 (v1)** → **0.0072 (v2corpus)**. The v1 side is rater-matched; the v2corpus side
is rater-mismatched. **The comparison is confounded with a change in the label definition**, and
the confound is not separable from banked data.
⚠️ **It does not follow that the collapse is an artifact.** The 2026-08-16 pass measured
v2corpus's `distributions.maneuver_dir = [8, 6, 404]` — **96.7 % of windows declared one class**,
and a curvature gate would push windows *toward* `lane_keep`/straight, not toward a 404-window
`right` pile. That degeneracy is a real defect. **Both are true**: there is a real degenerate head
AND an instrument mismatch stacked on it, and the banked numbers cannot apportion between them.
Evidence class for the confound: **MEASURED** (source + registry). For its magnitude:
**UNQUANTIFIED** — and it needs a GPU re-score under v2 labels to become quantified.

### 1.4 ⚠️ `commanded_route_vs_trajectory` was already mismatched at 0.15 — by a factor of 5.2

`route_n` is the route head's argmax. Route labels come from `route_from_future_v21`, whose turn
test is `NET_DYAW_TURN_RAD = π/4` = **0.785 rad** (`refb_labels.py:548`). The trajectory side is
gated at **0.15 rad**. That is a **5.2× definitional gap present at the published gate**, before
any re-read — on top of the cross-timescale gap the panel's own `_note` already flags (route is a
15–25 s heading, trajectory is 2 s). Its κ on the deployed-architecture panel is **0.0116**
(880 windows / 40 episodes, OOD-val q90). ⇒ **This number is not a coherence measurement and
should not be read as one at any gate.**

### 1.5 ⛔ A second, separate defect found in passing — `man_tgt` is hardcoded to v1

`hierarchy.py:592-593` computes the ground-truth manoeuvre label as
`rl.classify_maneuver(...)` — **the v1 net-yaw classifier, unconditionally**, with no branch on
the arm's `v2_labels`. It flows to `rec["man_corr_real"]` (`:639`) and thence to
`seam_ctx_to_tactical.maneuver_acc` (`:820`).

⇒ **For every `--v2` arm, `maneuver_acc` is scored against a label definition the head was never
trained to produce.** This is not a `DIR_YAW_RAD` question at all; the gate re-read merely
surfaced it. It lands on the `ctx→tactical` seam, which `CLAUDE.md` records as the programme's
one "load-bearing" hierarchy seam. **Escalated, not fixed here.**

---

## 2. Provenance of 0.15 — MEASURED from source and git

**It was chosen by the author of the label module at first authoring, with a stated rationale that
was never checked against data, and it has never been revisited.**

* `git log -S'YAW_TURN_RAD = 0.15' -- stack/scripts/refb_labels.py` returns **exactly one commit**:
  **`35adfab`, 2026-07-11**, *"refB: 4-layer E2E reference…"*. `--diff-filter=A` confirms the same
  commit created the file. ⇒ **introduced once, never edited.**
* That commit's own message ends *"No training launched — implementation only."* ⇒ no measurement
  preceded it.
* The rationale is written down (`refb_labels.py:24-25`), and it is a real argument, not a vendor
  default:
  > `YAW_TURN_RAD = 0.15 rad (~8.6 deg over 2 s ~ a deliberate turn/lane change onset; highway
  > curve drift at ~0.02-0.05 rad stays lane_keep)`

**Verdict on the three options in the brief.** Not *measured*; not a *default nobody chose*. It is
a **third class: reasoned-but-unverified** — an author's prior, stated explicitly, then frozen.
That is materially better than the SAM3 `confidence_threshold=0.5` case (a vendor default nobody
chose) because the reasoning is inspectable and falsifiable.

⚠️ **And it was falsified the first time anyone measured it.** The docstring predicts the bulk of
non-turn driving sits at ~0.02–0.05 rad. Measured on canonical val (§3): the median |net yaw| is
**0.0181 rad** — consistent with the prediction — but the gate then sits at **8.3× that median**
and admits only **21.7 %** of windows as turning. The prior was right about where lane-keeping
lives and wrong about where to put the boundary relative to it.

---

## 3. ⭐ The re-read, with the mandated estimator

`code/gate_operating_point.py`, 0 GPU. **Estimator: `taniteval/ci.py`
`episode_cluster_bootstrap` / `paired_episode_cluster_bootstrap`, 2000 draws, seed 0, resampling
unit = EPISODE.** `overlapping_holdout_se` is refused in code, not merely avoided.

### 3.1 Hard gate — what is reconstructible, verified not inherited

Banked `windows_*.pt` carries `pred/gt/cv/eid/speed/head_deg`. `head_deg` is
`net_heading_change_deg(ep.poses, last)` = `|wrap(poses[last+20,2] − poses[last,2])|·180/π`
(`stack/scripts/driving_diagnostic.py:139-142`, written at `taniteval/taniteval/bench.py:399`) —
**the same poses, horizon and wrap as the gate's own input** (§1.1). ⇒ `head_deg·π/180` **is**
`|gt_net|`. The **sign is lost to `.abs()`, so κ is NOT reconstructible**; every magnitude
quantity is.

Cross-checked against each panel's own `consistency.distributions.gt_dir`, produced by a different
function in a different file during the GPU pass:

| | |
|---|---|
| panels cross-checked | **6 / 6 identical** — 191 turning windows of 881, every panel |
| panels with no matching dump | 4 (all `-lf19`, 418 windows — tensors were never rescued) |

⚠️ **Two corrections to the 2026-08-15 pass's version of this check** (`DIR_YAW_GATE_REREAD.md` §4,
*"0.000000 — bit-exact, 5/5"*):
1. It is **6 panels, not 5** — `hier_refa-dynin.json` also matches.
2. On the **fractions** the gap is **2.78 × 10⁻¹⁷**, not 0 — one float64 ULP, because
   `1 − straight/n` and `(rad > g).mean()` are different arithmetic paths to the same rational.
   The invariant is the **integer count**, and that is what this pass compares. Nothing
   substantive changes; the check is simply stated in the units that can actually be exact.

### 3.2 The operating point — canonical val, 881 windows / **40 episodes**, 22 arms

*(a corpus property: `head_deg` verified bit-identical across all arms sharing the episode set)*

| quantity | point | CI95 | separated |
|---|---|---|---|
| frac turning @ **0.15** | **0.2168** | [0.1396, 0.2997] | — |
| frac turning @ **0.10** | **0.2565** | [0.1703, 0.3443] | — |
| ⭐ **paired Δ (0.10 − 0.15)** | **+0.0397** | **[+0.0193, +0.0682]** | ✅ **yes**, p(Δ>0) = 1.000 |
| ⭐ **band mass (0.10, 0.15]** | **3.97 %** | **[1.93 %, 6.82 %]** | **m = 35 of 881** |
| median \|net yaw\| | 0.0181 rad (1.04°) | — | **gate ÷ median = 8.3×** |
| p90 / p99 \|net yaw\| | 0.4611 / 0.8667 | — | — |

**Second corpus present in the dumps** — `refc-v12` smoke, **4 episodes / 88 windows**: band mass
**2.27 %**, CI95 [0.00 %, 4.55 %], **not separated**, m = 2, median 0.0232, gate ÷ median 6.48×.
⚠️ **4 episodes is below decision grade** (`hierarchy.MIN_EPISODES_FOR_CI`); quoted only to show
the band mass is small on every corpus with a dump, never as evidence on its own.

**Incidental robustness check.** The 881-window corpus appears under **two different episode-id
encodings** (`'0','1','10',…` and `'808464434',…`) across 22 + 3 arms. `head_deg` is
`torch.equal`-identical; point estimates are identical; CI95 bounds agree to **≤ 0.003**. The
bootstrap is insensitive to episode-id ordering at this n. ⚠️ Minor instrument hygiene item: two
eid encodings coexist in banked dumps and nothing declares which is canonical.

### 3.3 What this does to the 2026-08-16 envelope — it confirms it, with an interval

That pass proved v1-lf19's *"weak"* is not established at 0.10 because a crossing count of
**m\* = 4 (0.96 %)** already admits SUBSTANTIAL, and estimated m ≈ 2–6 by scaling. The scaling
ratio band-mass ÷ frac-above-gate is now measured on canonical val at
**0.0397 / 0.2168 = 0.183**, alongside the 2026-08-16 OOD-val figure of 0.398. lf19's own
`gt_dir` gives 2.87 % above 0.15, so m ≈ 418 × 0.0287 × [0.183, 0.398] ≈ **2.2 – 4.8 windows**.

⇒ **m\* = 4 sits inside the range.** The 2026-08-16 conclusion **stands**, now resting on a
bootstrapped band mass rather than a point estimate. Evidence class: the ratio is **MEASURED**
(two corpora); the transplant onto lf19 remains **ESTIMATED** — lf19's own tensors are gone.

### 3.4 ⛔ What could NOT be measured, and why — stated plainly

**No Δκ in this report carries a bootstrap, because no banked artifact permits one.** A paired
episode-cluster bootstrap on Δκ needs per-window `(man_dir, traj_net_yaw, gt_net_yaw, eid)`
aligned. Probed at four locations:

| location | result |
|---|---|
| `hier_v1arch_gateswept.json.xz` (the one swept panel) | summary fields only — `gate_sensitivity.per_gate` holds κ, no arrays |
| every `hier_*.json` / `*.json.xz` in the repo | `grep -rl traj_net_yaw` over all JSON/xz ⇒ **0 hits** |
| `taniteval/results/windows_*.pt` (29 dumps) | `head_deg` is magnitude-only; no `man_pred`, no signed yaw |
| `…/2026-08-15-dir-yaw-gate-reread/results/gate_reread.json` | summary only |

`hierarchy.py` **does** build these arrays in memory (`rec["gt_net_yaw"]` `:658`,
`rec["traj_net_yaw"]` `:730`, `rec["man_pred"]` `:638`) — the 2026-08-06 fix put them there — but
the panel writer never persists them. ⇒ **Work item, §4.** Until it lands, every published Δκ in
this programme is a **point estimate with no interval**, including the ones in the two prior
passes. That is a fact about the instrument, not about the finding.

---

## 4. Blast radius — every document and row whose number would move

### 4.1 Would NOT move (established, so nobody re-opens them)

| | why |
|---|---|
| `MODEL_REGISTRY.md` §1.9 κ **0.6033** (SUBSTANTIAL, n = 6382) | needs −0.203 to leave SUBSTANTIAL; largest downward move ever measured is −0.0659. **No registry row changes.** |
| `commanded_route_vs_maneuver.*` | **gate-free on both sides** — `route_n` is an argmax, `man_dir` a fixed 5→3 table (`hierarchy.py:168,908`). Confirms the 2026-08-15 correction to `RETRACTION_LOG` R-2026-08-06-yawgate. |
| `distributions.{route_follow, route_commanded, maneuver_dir}` | no `_dir_of` |
| the *magnitude* of the v1→v2corpus κ collapse (0.253 vs 0.0072) | a 35× gap; no gate move approaches it |

### 4.2 Would move, or is not established — the actionable list

| # | number | where it is published | status |
|---|---|---|---|
| 1 | `maneuver_vs_trajectory_kappa` @ 0.15, on every banked panel *(the count **16** is **INHERITED** from the 2026-08-15 pass and not re-enumerated here)* | `Paper/TANITAD_PAPER.md:1868`, `:3175`; `MODEL_REGISTRY.md` §1.9; `Project Steering/V5_FLAGSHIP_DEEP_REVIEW.md`, `V5_PLAN.md`, `PREREG_v5_cheapest_guard.md`, `PREREG_rollout_recovery.md`, `V1_RECONSTRUCTION_RISK_RESOLVED.md`, `LOOP_STATE.md`, 4 × `Project Steering/Reports/*` *(MEASURED — grep over `Project Steering`, `Paper`, `DataEng`, `TanitAD Research Hub`)* | ⚠️ **quotable at 0.15 ONLY, and only for v1-labelled arms.** §1.2/1.3: the sweep is asymmetric; for `--v2` arms it is mismatched at every gate. |
| 2 | `kappa_turn_subset` = **0.2005** (deployed arm) | `RETRACTION_LOG.md`; `…/2026-08-06-v1-defect-triage/`; `…/2026-07-26-e1c-heldout-gated-clsft/`; `…/2026-07-27-x1-latent-metric-probe/` | ⛔ **still unswept** — the instrument gap flagged 2026-08-15 **and** 2026-08-16 is **still open**. And it is the number sitting on a 0.2 line. |
| 3 | `commanded_route_vs_trajectory.kappa` = 0.0116 | the same panels | ⛔ **not a coherence measurement at any gate** (§1.4, 5.2× label gap). Should carry that caveat wherever quoted. |
| 4 | `seam_ctx_to_tactical.maneuver_acc` for **`--v2` arms** | `MODEL_REGISTRY.md` §1.6/§1.7 rows; `V5_FLAGSHIP_DEEP_REVIEW.md` | ⛔ **scored against v1 labels the head never learned** (§1.5). Separate defect, escalated. |
| 5 | the paper's *"decorative"* / the collapse | `TANITAD_PAPER.md:1868`, `:3175` | ✅ **survive 0.10** (2026-08-16, exact envelope). ⚠️ now additionally **confounded by the v1/v2 label change** (§1.3) — the words survive, the *attribution* is not clean. |
| 6 | `trajectory_vs_gt_kappa` 0.8260 → 0.7781 (Δ −0.0479) | `…/2026-08-06-v1-defect-triage/`, `RETRACTION_LOG.md` | ✅ **legitimately swept** (symmetric). **Publish the sweep, never the point.** No verdict ladder attaches to it. |

### 4.3 Already CLOSED since the two prior passes — do not re-do

⭐ **`verdict_stable` is FIXED.** Both prior passes escalated it as open. It landed in commit
**`5daa3d7` (2026-08-16)**, *"The kappa ladder was single-sourced in the wrong direction…"*, with
`taniteval/tests/test_gate_sensitivity.py` guarding it. `_gate_sensitivity` now tests
`four_families.KAPPA_VERDICT_LADDER` (`hierarchy.py:290`) and emits
`maneuver_consistency_band` / `_verdict` per gate. Work by `…/2026-08-16-verdict-stable-kappa/`.
⚠️ **But every banked panel predates it** — `hier_v1arch_gateswept.json.xz` still carries
`verdict_stable: true` computed on the retired κ ≥ 0.2 predicate. **A pre-2026-08-16
`verdict_stable` is not a statement about the published verdict.**

---

## 5. Recommendation to the PI — one decision, three work items

**DECISION (yours).** ⛔ **Do not change `DIR_YAW_RAD`.** Changing it eval-side alone does not
correct the mis-scaling, it desynchronises the eval from the labels and re-bases every published
κ against a rater no model was trained to match. The mis-scaling is real (**8.3×** the human
median, MEASURED) and its home is `refb_labels.YAW_TURN_RAD` — i.e. **the training labels**. The
honest options are (a) leave 0.15 and always publish the symmetric sweep beside it, or (b) re-derive
labels at a measured threshold and **retrain**, which is a GPU programme decision, not an
instrument tweak.

**WORK ITEMS (mine or the next agent's — all 0-GPU except the last).**

1. ⛔ **Persist the per-window gate inputs.** `hierarchy.py` builds `gt_net_yaw`, `traj_net_yaw`,
   `man_pred` and drops them at panel-write. Persisting them makes every future Δκ
   bootstrappable — and until then no Δκ in this programme has an interval (§3.4).
2. ⛔ **Sweep `kappa_turn_subset`.** Flagged 2026-08-15, re-flagged 2026-08-16, still open, and it
   is the one number on a boundary (0.2005).
3. ⛔ **Branch `man_tgt` on the arm's label version** (`hierarchy.py:592`) — or refuse to emit
   `maneuver_acc` when they disagree. Currently a v2 arm is silently scored on v1 labels (§1.5).
4. ⚠️ *(GPU, later)* Re-score `v2corpus` under **v2** labels to separate the degenerate head from
   the instrument mismatch in the paper's collapse claim (§1.3).

---

## Deliverable manifest

| artifact | repo path | state |
|---|---|---|
| `DIRYAW_REREAD.md` (this file) | `TanitAD Research Hub/Architecture & Inference/Implementation/incoming/2026-08-17-diryaw-reread/` | staged |
| `code/gate_operating_point.py` (0-GPU re-read; integer-count hard gate; paired bootstrap) | same dir | staged |
| `raw/gate_operating_point.json` (every number in §3, machine-readable) | same dir | staged |

**Nothing committed, nothing pushed. No GPU used, no pod contacted. No threshold changed in
shipped code. `MODEL_REGISTRY.md` not touched — no registry number moves (§4.1).**
