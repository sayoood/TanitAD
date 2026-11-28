<title>SPEC — anchored actdiv (GS-8) and the transition-level probe rung (GS-9), 2026-09-03</title>

# SPEC — E-BE-GS89-0903: anchored action-divergence + transition-level probing

`TanitAD Research Lab · Architecture & Inference · Benchmarks & Evals FlyWheel · 2026-09-03`
`Seeds: LAB_BACKLOG.md GS-8, GS-9, GS-10 · Frontier Scan/LEDGER_A2_jepa.md 2026-09-02-01/-02 ·`
`V7_LAUNCH_GATE.md P2, P5 · Library 2606.31232 (Delta-JEPA, full text read for the definitions).`
`Class: INSTRUMENT work package. 0 GPU — every checkpoint pass runs on the dev-box CPU`
`(the RTX 4060 is reserved for the T1 reads until ~05:30 Berlin; coordinator directive).`
`Tier: T0-DIAGNOSTIC throughout. Nothing here is a driving-performance claim.`

⛔ **Written BEFORE any banked checkpoint was read.** The thresholds in §4 and §7 are also
the `THRESHOLDS` dict inside `taniteval/tools/actdiv_anchored.py`; changing either after the
run is a retraction, not an edit.

---

## 1. What & why

The banked action-divergence instrument (`actdiv`, MM-E10; code
`…/2026-08-24-action-conditioning-and-heldout/probes/actdiv_local.py` and
`…/incoming/2026-08-30-action-divergence/code_actdiv_thor.py`) reads ONE scalar per horizon:
`std across 8 rolled action variants / std across scenes` — **0.00408 / 0.00416 / 0.00595** on the
three 30k arms (`GOALS_AND_CLAIMS.md` MM-E10 row; raw `actdiv.json`), 0.006165 on `k8clip05p30k`
(raw `…/2026-09-01-mm-e19-k8-attribution/raw/mm_e19_read_step30000_K8.json`). It is a
**magnitude** read. It cannot see **structure**: whether a given action moves the prediction in a
consistent direction, and whether a larger action moves it further.

Delta-JEPA (2606.31232, §*Action-Sensitive Latent Dynamics*, Figure 6) reads exactly that, and
uses it to show that a strong published model (LeWorldModel) fails it: *"keep the history
representation fixed … replace the final action with each candidate action … measure the
displacement relative to the zero-action prediction ẑ_{t+1}(a) − ẑ_{t+1}(0)"*; sensitive =
*"well-separated action-wise mean responses, with larger action magnitudes generally inducing
larger predicted shifts"*; insensitive = *"means remain concentrated near the origin and
substantially overlap"*.

The same paper probes **transitions** (§*Physical and State-Delta Probing*, Table 5): *"predict
state changes Δx_t = x_{t+1} − x_t from latent displacements Δz_t = z_{t+1} − z_t … split
train/test data by trajectory"*. Our L1–L3 ladder probes **states** only
(`V7_LAUNCH_GATE.md` P5: L1 participation, L2 `z_t → attributes` vs pixels/const/DINOv3, L3
`ẑ` vs `z_t`). The transition rung is where P5 — *does the predictor add anything over z_t?* —
lives, because a predictor that restates the scene has a Δẑ that carries no transition.

Two instruments, no training change, both surviving a frozen trunk (ledger entry
2026-09-02-02 §"Three levers that survive a frozen trunk").

## 2. Hypotheses (IDs) with the readings committed in advance

**H-GS8-1.** *Anchored on its own zero-action prediction, the v7 predictor's action response is
STRUCTURED (consistent direction, monotone in |a|) even though it is small in magnitude.*
The banked instrument cannot distinguish this from noise; this one can (§4). All four outcomes
are admissible results; none is a failure of the instrument.

**H-GS9-1.** *The encoder displacement Δz = z_{t+1} − z_t carries the ego transition (Δx_fwd,
Δyaw) beyond the raw pixel change, and the PREDICTOR's displacement Δẑ = ẑ_{t+1} − z_t adds
NOTHING over z_t about that transition (P5 at the transition level), with the action
contribution to Δẑ indistinguishable from zero (P2 at the transition level).* Both halves are
committed with their opposite (§7).

**GS-10 (carried, not testable here).** `o11p30k` is not on this box (two probes: a name search
and a listing of every ckpt file > 50 MB under `C:\Users\Admin`; it lives on Thor, which this
package must not contact). The mechanism hypothesis stays OPEN and the tool is ready for it.

## 3. GS-8 — definitions

Population: the banked `actdiv` windows, reproduced VERBATIM — the 24 sorted clips of
`physicalai-val-0c5f7dac3b11-w120-256x640cyl` (local copy
`C:\Users\Admin\tanitad-caches\mm-e19-assets-20260901\sp2\cache\physicalai-val-w120-256x640cyl`,
24/24 md5-listed in `MANIFEST.md5`), first 60 frames, 3-frame stacks oldest-first, W = 6
windows at `range(0, n, n // 5)` ⇒ **144 windows per arm**.

For window i with history (z_{t−5..t}, a_{t−5..t}) and horizon h:

    d_i(a) = ẑ^{(i)}_{t+h}(a_{t} ← a) − ẑ^{(i)}_{t+h}(a_{t} ← 0)          (the last action replaced)

Candidates: two axes — channel 0 (`kappa`: the corpus' road-wheel-angle proxy `atan(L·κ)`),
channel 1 (`accel`: a_long) — at ±{0.25, 0.5, 1, 2, 3}·σ of the 144 windows' own final actions,
the other channel at 0 (20 candidates + the zero anchor). Reported per candidate: m(a) =
mean_i d_i(a), ‖m(a)‖, and ‖m(a)‖ / scene_spread where scene_spread = mean-over-dims std of the
zero-action prediction across windows (the banked C1 denominator, so the magnitude is on the
banked scale). Across candidates:

* **F_sep** = (Σ_c N‖m_c − m̄‖² / (C−1)) / (Σ_c Σ_i ‖d_ic − m_c‖² / (C(N−1))) — one-way
  between/within ratio of the displacement vectors;
* **monotonicity** per axis — Spearman ρ between |level| and ‖m‖ over the axis' 10 candidates;
* **sign consistency** per level — cos(m(+L), m(−L)); a linear-ish response reads < 0;
* **|a| binning** — candidates pooled by |level| across both axes, ‖m‖ per bin, Spearman ρ.

Secondary (the task's literal read): each window under its OWN final action vs the zero anchor,
binned by |a| (σ-normalised), by κ and by a (quartiles), same between/within ratio; its
shuffled-action control FEEDS rolled actions (window i gets window i+j's action) and labels by
window i's original bin.

Also recomputed on the same windows: the banked `ratio_action_over_scene` VERBATIM.

**Controls that must read known values (else VOID):** C0 identity (same inputs twice → max|Δ|
== 0.0 exactly); C1 scene spread > 1e-6; shuffled labels within each window (200 permutations →
median, p95 of F_sep — the null band is MEASURED); zero model (the real predictor fed the zero
action for every candidate → max|d| == 0.0 exactly); output scale (× 10 on every prediction →
F_sep, ρ, cos and relative magnitudes unchanged to < 1e-6 relative); input scale (z ← 0.5z, 2z, a
real re-forward → the verdict class should survive; if not, stamped scale-fragile).

**Lift.** DEFAULT = the trainer's own contract: `train_v6_staged._lift3` with the checkpoint's
`cond_param` and v0 = the window's LAST speed over `flagship_v15.SPEED_SCALE` (10.0). The banked
scripts hardcode `SPEED_SCALE = 30.0` and feed the window's FIRST speed; `--actdiv-compat`
reproduces that construction. ⭐ Under `--actdiv-compat` the recomputed legacy ratio MUST
reproduce the banked values (postrain30k **0.005947**, k8clip05p30k **0.006165**) — that is the
known-value reproduction of the window selection and the encoder path. If it does not, the
new numbers are not on the banked population and the run is VOID.

**Horizons.** Only head 1 is trained (`tanitad/models/predictor.py`, MM-E14). The verdict is
read at h = 1; h = 2, 4 are reported under `untrained_heads` as INIT NOISE and are not quotable.

## 4. GS-8 — the pre-registered reading (h = 1)

| outcome | criterion |
|---|---|
| **VOID** | C0 ≠ 0, or zero-model ≠ 0, or scene_spread ≤ 1e-6, or all displacements identically 0 |
| **INSENSITIVE** | F_sep < 5 × p95 of the shuffled-label null — the LeWM picture: means overlap |
| **SEPARATED-NONMONOTONE** | F_sep ≥ 5 × p95, but Spearman ρ < 0.8 on either axis OR cos(m(+L), m(−L)) ≥ 0 at any level on either axis — the GS-8 committed read: *"separation but NOT monotonicity ⇒ 'the predictor uses its actions' is false in a way the binary control cannot see"* |
| **STRUCTURED-WEAK** | separated AND monotone AND sign-consistent, but ‖m(2σ)‖ / scene_spread < **0.0595** |
| **SENSITIVE** | all of the above AND ‖m(2σ)‖ / scene_spread ≥ **0.0595** |

The magnitude bar is the programme's existing committed one (`PREREG_MM_E19_K60_HORIZON.md`
HORIZON-WORKS = 10 × the incumbent's 0.005947), read at the 2σ level because the banked
instrument's rolled variants differ by ≈√2σ per channel on both channels — the 2σ single-axis
candidate is the closest analogue. Every level is reported so the choice is inspectable.

⚠️ **Interpretation committed in advance.** A deterministic, smooth predictor has a Jacobian, so
a *tiny* response can still be STRUCTURED (same direction for every window). STRUCTURED-WEAK on
a 0.005-ratio arm therefore means: *the action reaches the output through a consistent
linear-ish path whose gain is negligible* — a different diagnosis from INSENSITIVE (no
consistent path) and from SEPARATED-NONMONOTONE (a path that is not a function of |a|).
The three are different mechanisms and they route P2 differently (§8).

## 5. GS-9 — definitions

Corpus: the local HELD-OUT set `physicalai-val130-heldout` (129 clips; the E-DEC-59 /
latentmotion corpus), first 100 frames per clip, dataset-consistent stacked rows (stacked row
j = raw frames j..j+2, paired with pose j+2; no padded stacks), W = 6 windows ending at row t
with a pose at t+1 ⇒ 92 rows per clip, **n ≈ 11.9k transitions**. Split BY CLIP: 5-fold outer
(every clip scored once out-of-fold), 5-fold clip-grouped inner CV for λ on the fit clips only.

Targets over one tick (k = 1 = 0.1 s, the predictor's only trained horizon), ALL from `poses`
[x, y, yaw, v]: `dx_fwd`, `dy_left` (world displacement rotated into the ego frame at t),
`dyaw` (wrapped), `dv = v_{t+1} − v_t` — the speed column of the pose is the ONLY place v enters
this probe, and only as a target.

Inputs (features, all standardised on the fit split; intercept unpenalised):

| cell | what | role |
|---|---|---|
| `const` | ones | **must read skill 0.0000 EXACTLY** |
| `pixdelta` | newest-frame 8×20 gray, pix_{t+1} − pix_t (160-d) | raw-input floor |
| `act2` | [steer_t, accel_t] — the fed action WITHOUT v | echo reference |
| `z_t` | encoder state | the state rung, for marginals |
| `dz_enc` | z_{t+1} − z_t | Delta-JEPA Table-5 object |
| `dzhat_true` | ẑ_{t+1}(a_true) − z_t | the predictor's transition |
| `dzhat_zero` | ẑ_{t+1}(a ← 0) − z_t | the predictor's transition without its action |
| `dzhat_anch` | ẑ_{t+1}(a_true) − ẑ_{t+1}(0) | the anchored action response (= GS-8's d) |
| `z_t+dz_enc`, `z_t+dzhat_true` | concatenations | marginal of the transition over the state |

Metric per target: **skill = 1 − SSE(pred)/SSE(fit-split mean)** pooled over out-of-fold rows
(so `const` reads exactly 0), plus Pearson r; CI = clip-cluster bootstrap (1000 draws) of the
pooled ratio; marginals = PAIRED clip-resampled Δskill. Controls: global target shuffle (must sit
in the null band — reported, and it is expected slightly NEGATIVE out-of-fold), within-clip
shuffle (what survives is a clip-level cue; **transition-specific skill = real − within**).
`n` (fit per fold, score) and `d` are printed for every cell. `--mlp` adds a fixed-seed RFF +
ridge non-linear comparison with the same controls.

⛔ A negative here is a negative about the linear map only (CLAUDE.md linear-probe rule); the
verdicts below say so.

## 6. What is and is not run

| arm | local? | run |
|---|---|---|
| `postrain30k` | YES (`…\v7tiny_postrain30k\ckpt.pt`, md5 `a58585883c279633c799a6f6968cc4b2` = registry §13.9) | both tools |
| `k8clip05p30k` | YES (`…\v7tiny_k8clip05p30k\ckpt.pt`, md5 `2d744d6d2faa6e62d3b5eb2e9c2eecd6` = the k8 package's stated md5; ⚠️ NO registry row) | both tools |
| `o11p30k` | **NO** — not on this box (two probes) | skipped, never approximated |
| `splitp30k` | **NO** — not on this box (two probes) | skipped |
| `postrain30k_freeze` | **NO** — not on this box (two probes) | skipped |
| `k60clip05p30k`, `rdw8p30k` | yes, unrequested | GS-8 only, as EXTRA rows (the pass is ~1 min on CPU and both are in the same action-ratio story) |

Device: **CPU** for every reading (coordinator directive; the CPU encoder measured 0.013 s/frame
on postrain30k, so a GS-8 pass is ~1 min and a GS-9 pass a few minutes).

## 7. GS-9 — the pre-registered readings

"A beats B" = paired clip-bootstrap 95 % CI of skill(A) − skill(B) excludes 0 **and**
|Δskill| ≥ 0.02 (a minimal effect so a CI alone cannot fire on a 0.001 gap at n ≈ 12k).

| read | quantity | outcome if it holds | outcome if it does not |
|---|---|---|---|
| **R1 encoder transition** | `dz_enc` beats `pixdelta` on `dx_fwd` and `dyaw` | ENC-TRANSITION-CARRIED (the Table-5 property holds for our encoder) | ENC-TRANSITION-ABSENT — the encoder displacement is no better than pixel change for ego motion |
| **R2 transition-level L3** | `z_t+dzhat_true` beats `z_t` on ≥ 2 of 4 targets | L3-TRANSITION-PASS — the predictor's transition adds information about Δx beyond the current state | **L3-TRANSITION-FAIL** — consistent with P5 ("the predictor adds nothing over z_t") |
| **R3 action attribution** | `dzhat_true` beats `dzhat_zero` | ACTION-REACHES-TRANSITION; then if skill(`dzhat_anch`) ≈ skill(`act2`) (within 0.02) it is an ECHO of the fed action, not dynamics | **ACTION-ABSENT-AT-TRANSITION** — consistent with P2 at the transition level |
| **R4 time structure** | transition-specific skill (real − within-shuffle) ≥ 0.5 × real skill for the winning cells | the read is about TRANSITIONS | the read is a CLIP-LEVEL cue (scene class); flagged, and R1–R3 are re-read on the transition-specific column |

Controls that must read known values: `const` = 0.0000 exactly (else the run is VOID);
global shuffle inside [−0.05, +0.05] on every cell (else VOID); the floor is reported.

## 8. What each outcome says about P2 / P5 (committed)

* GS-8 INSENSITIVE on every 30k arm ⇒ P2 stands with a sharper statement: *not even a consistent
  direction* — the LeWM failure mode, which the ledger reframed as a field failure (2026-09-02-01).
  GS-8 STRUCTURED-WEAK ⇒ P2 is a GAIN problem, not a wiring problem: the path exists and is
  monotone; the levers that scale a gain (target construction P2(d), conditioning interface GS-3)
  are the ones to test, and "action-as-token" is not obviously needed. SEPARATED-NONMONOTONE ⇒
  the path is non-functional in |a| — P2(c) (representation/geometry) gains weight.
  SENSITIVE ⇒ the banked ratio under-read the arm; P2's evidence must be re-derived.
* GS-9 L3-TRANSITION-FAIL + ACTION-ABSENT ⇒ P5 and P2 are one phenomenon at the transition
  level (the launch gate's own reading: *"the predictor is not transporting the scene, it is
  restating it"*). L3-TRANSITION-PASS with ACTION-ABSENT ⇒ the predictor does transport something
  about the transition — from the history, not the action — which narrows P2 to the action path.
  ENC-TRANSITION-ABSENT ⇒ the frozen/distilled encoder's Δz does not carry ego motion beyond
  pixels, which is the transfer risk the ledger named (LDAD shapes the encoder; ours has nothing
  to shape) measured rather than argued.

## 9. Method notes and admissibility

* Every number: `MEASURED (ours; dev-box CPU; raw path)`. Registry facts cited from
  `MODEL_REGISTRY.md` rows or raw JSON only; where a registry row is MISSING (`k8clip05p30k`,
  `o11p30k`) that gap is reported as a conflict, not papered over.
* `tanitad.__file__` and every ckpt md5 are stamped into the JSON (MM-C12).
* The tools carry `--selftest` (synthetic predictors / planted structure) and CPU unit tests
  (`stack/tests/test_actdiv_anchored.py`, `stack/tests/test_transition_probe.py`) that assert
  every control reads its known value and that the shuffled controls collapse.
* Four metric families: REFUSED with the reason stamped (T0 world-model probe; no driving metric
  exists to family-ise). Not a completeness gap; a scope statement.

---

## 10. ADDED 04:10 Berlin, 2026-09-03 — refav1 (H-REFAV1-LAT-INSENSITIVE), committed BEFORE the run

`Coordinator target, highest priority after the tool works. MEASURED tonight (register amendment to`
`D-REFAV1-STEP1000-READ): on 140/140 eval windows refav1's step-1,000 closed-loop plan has curvature`
`EXACTLY 0 under every nav and goal condition — the planner's search cost looks flat in kappa.`

**Where the planner's cost lives (read from `refa_v1.py::plan`, `_cost_chunk`):** the candidate
controls are rolled `last_only` and the TERMINAL operative field is pooled into the TACTICAL QUERY
space by the model's own `tac_pool` (`_tac_field`, 64 queries × d = 65,536 dims), then scored by
`1 − cosine` against the goal, PLUS `0.02·jerk²`, PLUS **`0.05·κ²` — an explicit curvature
penalty** — plus a target-speed term. ⇒ If the WM term is flat in κ, the κ² penalty alone drives
κ → 0 exactly. That is the mechanism the hypothesis names.

**The read.** Anchored displacement at ONE operative step (h = 1, 0.2 s — CPU; the planner-horizon
h = 10 rollout is a GPU follow-up), on the SAME 140 windows the T1 read used (20 episodes of the
eval slice, `k_loader = op_steps = 30`, stride 10 on `t − (W−1)`), with the intent token from the
model's own brains under the loader's nav (held fixed while the action varies), and the speed
channel exactly as `augment_actions` derives it from the loader's `v0` (OFF in both configs —
the predictor sees (a, κ) only):

    d_i(a) = step(last_i, aug(a)_0, intent_i) − step(last_i, aug(0)_0, intent_i)

Candidates in PHYSICAL units (the coordinator's grid): κ ∈ ±{0.02, 0.05, 0.1} rad/m at a = 0, and
a ∈ ±{0.5, 1.5} m/s² at κ = 0 (10 candidates + the zero anchor). Read in THREE spaces: the tac-pooled
field (the planner's cost space — the VERDICT space), the full token field (640 × 1024; exact
per-candidate statistics from sums, permutation null on a fixed 8,192-coordinate subsample), and
the token-mean field (1,024-d; exact). Per axis: F_sep over that axis's candidates, its
label-permutation null (200), the level norms, cos(+L, −L). Plus the realised read: each window
under its OWN first action vs zero, and 2 ROLLED-action variants (the shuffled-ACTION control).

**Controls (must read known values, else VOID):** C0 identity == 0.0 exactly; zero model (the
real predictor fed the zero action for one candidate per axis) == 0.0 exactly; scene spread > 1e-6
(std across windows of the zero-action step).

**Pre-registered verdict (tac space, h = 1):**

| outcome | criterion |
|---|---|
| **VOID** | any control off its known value |
| **LAT-INSENSITIVE-CONFIRMED** | `F_sep(a-axis) ≥ 5 × its null p95` AND `F_sep(κ-axis) < 5 × its null p95` — the a-bins separate, the κ-bins sit at the within-bin floor |
| **LAT-INSENSITIVE-REFUTED** | `F_sep(κ-axis) ≥ 5 × its null p95` — the predictor DOES respond to κ consistently; the flat-κ plan is then a COST property (κ² penalty against a κ-response that does not move the goal cosine), not a WM insensitivity |
| **BOTH-INSENSITIVE** | neither axis clears 5× — the P2 picture; lateral is not special |

Cross-check: the same read in the full-field and token-mean spaces; a verdict that differs across
spaces is reported as SPACE-DEPENDENT with all three. Magnitude asymmetry reported as
‖m(κ = 0.1)‖ / ‖m(a = 1.5)‖ (per-dim RMS) in every space. Both checkpoints: the fp32 incumbent
(`refav1_eval_slice\ckpt\ckpt.pt`, config from `ckpt['cfg']`) and the clean epoch (`ckpt_ep2`,
EMA + bf16 training, `config.json` beside it; the `model` state dict is what `refav1_arm.load_model`
loads and what the T1 read used). Device: CPU (measured 0.67 s per window-step; ~27 min per
checkpoint; launched detached).

**What it says:** CONFIRMED ⇒ H-REFAV1-LAT-INSENSITIVE stands as a WM property and the fix is on the
predictor's κ path (P2 family), not in the planner. REFUTED ⇒ the fix is in the cost (the κ²
penalty / goal-cosine blindness to lateral change), and the register row must say the WM is not
the cause. BOTH-INSENSITIVE ⇒ refav1 at step 1,000 is action-deaf like the v7 arms (P2), and the
lateral finding is a special case of it.

## 11. Units clarification (recorded 04:00 Berlin, after the first v7 read and before any verdict was quoted)

The GS-8 magnitude bar compares the mean displacement with the banked C1 denominator, which is the
MEAN OVER DIMS of the per-dim scene std. The commensurate numerator is the PER-DIM RMS of the mean
displacement, ‖m‖/√S — the first tool version divided the L2 norm ‖m‖ (√S ≈ 45× larger at S = 2048)
by the per-dim std and would have read a 0.006-ratio arm as "material". The threshold (0.0595 at
2σ) is unchanged; the numerator's units were corrected to match its stated definition. The first
run's JSON stores the unscaled norms, so the corrected reading is recomputable from it.
