# H2 · E1 + E0 — PRE-REGISTRATION

**Written 2026-07-25, BEFORE any held-out number was computed.** Timestamp of this file precedes
the timestamps of `e1_heldout.json` / `e0_split.json` in the same folder; the driver scripts are
committed alongside so the whole thing is re-runnable.

Both outcomes are committed below. Nothing in this file may be edited after the first result run —
corrections go in the results doc as a marked amendment.

---

## Why a pre-registration at all

`H2_SUBSTRATE_AND_LABELING.md` §6.4 reports the `L1_gate` decision-relevance lift as
**2.22× [1.30, 3.14] at a 3.0 m conflict radius**, and **0.43× [0.24, 0.71] at 6.0 m** — both CIs
excluding 1.0, in opposite directions. **3.0 m was selected after seeing the sweep.** The substrate
agent flagged this on itself. It is the program's recorded "premature certainty from a chosen cut"
class (`CLAUDE.md`: the same learning-curve log yields −0.387 … −0.738 depending on the window).

E1 exists to decide that one thing, on data the sweep never touched, at one threshold, once.

---

## The held-out split — defined by data availability, not by results

`L1_gate` needs three artifacts per clip: `obstacle.offline` 3D tracks, `camera_intrinsics` +
`sensor_extrinsics`, and `egomotion`. Locally: 26 obstacle chunks, 30 calibration chunks,
197 egomotion chunks. The intersection of all four is **`{0036, 0170, 1860, 1864}`**.

**The sweep used chunks `0036` and `0170` (80 clips, `crux3.parquet`).**

> **HELD-OUT SET (pre-committed):** every local `obstacle.offline` chunk *other than* `0036` and
> `0170` for which calibration is available at run time. As of writing that is `{1860, 1864}` =
> **175 clips**; if the calibration pull for further chunks succeeds before the run, those chunks
> join the held-out set **and every one of them is reported**. No chunk may be dropped after its
> result is seen.

**Zero clip overlap with the sweep** (measured: `n_in_sweep = 0` for both chunks,
`h2e_probe_split.py`). 175 ≥ 40 episode-clusters → the decision-grade bar is cleared 4.4×.

---

## E1 — the STOP gate

**Frozen label**, character-for-character the substrate audit's `L1_gate` at
`d = 3.0 m`, implemented by importing its own code (`crux.py` → `clip_rig`, `project`, `in_frame`,
`in_model_crop`, `resample_tracks`; conflict geometry from `crux3.py`):

```
L1_gate(t) = 1  iff  ∃ agent a :
  (i)   proj_front_wide(a,t) ∉ CanonicalCrop(t)          [outside the 51.4° encoder crop]
  (ii)  proj_X(a,t) ∈ frame(X),  X ∈ {cross_left, cross_right}
  (iii) min_{h∈(0,4s]} ‖p_a(t+h) − p_ego_CV(t+h)‖ ≤ 3.0 m   [ego CONSTANT-speed continuation]
  (iv)  no agent satisfying (iii) lies inside CanonicalCrop(t)

response(t) = 1  iff  v(t+4.0 s) − v(t) ≤ −1.0 m/s
lift        = P(response | L1_gate=1) / P(response | L1_gate=0)
```

**Threshold: 3.0 m ONLY.** No sweep, no re-tuning, no alternative response definition, no
alternative hysteresis. Per-clip `(cx, cy)` and per-clip 6-DoF extrinsics throughout (two-rig
corpus: rig A cy≈543 / rig B cy≈755; a global-centre assumption is ~215 px wrong for rig B).

**Estimator:** paired episode-cluster bootstrap, `B = 2000`, seed 0, resampling **clips** with
replacement; both arms (`P(resp|+)`, `P(resp|−)`) recomputed inside the *same* draw so the
episode-level difficulty cancels. Resampling machinery imported from `taniteval/taniteval/ci.py`
(`episode_index`, `_draws`). **`overlapping_holdout_se` is not used anywhere.**

### Outcomes, committed in advance

| | Condition | Consequence |
|---|---|---|
| **PASS** | held-out lift CI **excludes 1.0 from above** at 3.0 m | `L1_gate` is decision-relevant. Proceed to E0. |
| **FAIL** | CI includes 1.0, **or** the lift is materially attenuated (point estimate < 1.5×, the substrate agent's own stated bar) | **`L1_gate` is not decision-relevant and H2 stops at the label.** Report plainly. **Do not re-sweep, do not re-tune, do not rescue.** A *different label design* (§6.7 of the substrate doc: `L1-occlusion`, `L1-lateral`) is the next step — never a tuned version of this one. |

### Mechanism check (reporting only — it can NOT change the verdict)

If the "3 m = collision course, >3.5 m = adjacent lane" story is right, the held-out lift as a
**continuous function of `d`** should decay smoothly and monotonically through ~3.5 m. A spiky
profile that happens to peak at 3.0 m indicates the threshold was fit to noise. **This profile is
descriptive; the PASS/FAIL verdict above is decided at 3.0 m and nowhere else.**

---

## E0 — the real scope (runs only if E1 PASSES)

**MEASURED premise:** the encoder input is **51.4°** (`calib.py` `F_REF=266`, 256 px ⇒
`2·atan(128/266)`), while `front_wide_120` sees **120.5°**. 57 % of the front camera's own field is
discarded before the model. `cross_left_120` spans ≈ +6.5°…+127.7°; our crop ends at ≈ +25.7°.

Recompute `L1_gate` against the **full 120.5° front field** and partition every positive event:

1. **RECOVERABLE-BY-CROP** — agent inside the full front frame but outside the 51.4° crop
   ⇒ **no second camera needed; widen the crop.**
2. **GENUINE OFF-FRONT RESIDUAL** — outside the full front frame entirely, visible only in
   `cross_left`/`cross_right` ⇒ **H2's true scope.**

Reported as counts and percentages, overall and split by **intersection** and **lane-change**
strata (kinematic detectors from `situ_full.py`, unchanged), each with an episode-cluster
bootstrap CI. Then the efficiency claim is re-derived **on the residual only**: residual need-rate,
implied cameras/frame, compute saving vs always-on-7 and always-on-3.

### Outcomes, committed in advance — **neither is a failure**

| | Condition | Consequence |
|---|---|---|
| **A** | large recoverable-by-crop share | **The cheapest capability win is a wider crop, not a second camera.** H2 narrows to a smaller, better-posed residual. This is to be reported *loudly*, not buried — it changes what we build. |
| **B** | small recoverable share | The workstream is well-posed as originally framed; the residual is genuinely off-front. |

⚠️ **This must not become an argument for the conclusion anyone expects.** The split is reported as
measured, in both directions, with the CI.

---

## Discipline binding this run

- Evidence class on every number: MEASURED (+ path/command) · PUBLISHED · INHERITED · ESTIMATED · HYPOTHESIS.
- CPU only. No GPU, no training, no model inference. No pod touched (pod1/pod2/pod3 all busy).
- Read-only access to `C:\Users\Admin\tanitad-data\physicalai\` + a ~2 MB calibration pull from HF.
- Per-clip `cy` and per-clip extrinsics mandatory (two-rig corpus).
- Estimator named on every interval; ≥40 episode-clusters; never `overlapping_holdout_se`.
- No `git add`, no commit, no push. The orchestrator stages.
