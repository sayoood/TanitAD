# REF-C v3 — implementation, training readiness, and what we changed from DiffusionDrive

`MEASURED (ours; preflight run 2026-08-21 on the dev box)` +
`PUBLISHED-PRIMARY` (**DiffusionDrive banked today**, `2411.15139`) ·
**T0 for every number here** · requested by the PI as the next model to train.

⭐ **HEADLINE: v3 is implementation-COMPLETE and its own preflight PASSES. It is
blocked on COMPUTE, not on code — and NOT on the VLM label pipeline.**

---

## 1. Readiness — verified, not asserted

| check | status |
|---|---|
| module `stack/tanitad/refs/refc_v3.py` | ✅ 27.9 KB |
| trainer `stack/scripts/refc_v3_train.py` | ✅ 24.2 KB |
| tests `stack/tests/test_refc_v3.py` | ✅ **16 pass** |
| design + pre-registration | ✅ `REFC_V3_DESIGN.md` (38 KB, edges E1–E12) + `PREREG_REFC_V3.md` (both outcomes committed) |
| ⭐ **`--preflight` (its own gate)** | ✅ **PASS** — run today |
| instrument prerequisite (L2a) | ✅ *"REFUSAL RETIRED 2026-08-18"*, `win["lead"]` wired |
| ⛔ **compute** | ⛔ **BLOCKED** — see §4 |

**Preflight output, 2026-08-21:**

```
params  core 60,882,074 · phi_tac 1,708,288 · str_goal_head 195 · gstr_cond 66,816
        tac_heads 9,234 · tac_latent_proj 262,656 · scorer 1,156 · TOTAL 62,930,419
freeze-history gate: pass=True   pooled_rel_move 0.0   history_grad_nonzero True
loss step OK: traj cls law route lat lon lat_tac lon_tac goal_tac sel_v3 …
```

⇒ **The whole hierarchy costs 2,048,345 params — 3.3 % on top of the 60.88 M
core** — and `pooled_rel_move 0.0` confirms the zero-init FiLM leaves the cascade
**bit-inert at init**, so the H arm starts identical to the F arm by construction.

## 2. ⭐ It is NOT blocked on the tactical/strategic label extraction

The PI expected this to be a dependency. **It is not.** v3's supervision is
**hindsight-geometric from ego poses**:

* `refb_labels.goal_tac_targets(poses, …)` → `g_tac` **[K, 4] = (x, y, heading,
  speed)** at {2, 4, 6} s, ego-frame, with a clamp + validity mask;
* trajectory / maneuver targets via `RouteV21Dataset` (v2.1 labels);
* `g_str` trained on the **LAN corridor** label.

⇒ **The VLM tact/str pipeline supplies v6's SEMANTIC token vocabulary. v3 needs
geometric goals, which we already derive.** Finishing the label extraction is
valuable for v6 and for the paper — **it is not on v3's critical path.**

⭐ And v3 is correct against the PI's 2026-08-03 goal ruling: the goal is
**PREDICTED** (a 195-param head off the strategic GRU), the LAN corridor is a
**training label only**, and **E12 forbids a supplied route at inference**.

## 3. What we changed from DiffusionDrive — and why each is justified

**The paper** (`2411.15139`, banked today): truncated diffusion over **anchored**
multi-mode priors, **2 denoising steps** (10× fewer than a vanilla diffusion
policy), ResNet-34 backbone, **88.1 PDMS** on NAVSIM navtest, **45 FPS** on a 4090.

| axis | DiffusionDrive | REF-C v3 | basis |
|---|---|---|---|
| truncated diffusion, **2 denoise steps** | ✅ | ✅ **kept** | the paper's core efficiency claim |
| anchored trajectory priors | ✅ | ✅ **kept** | |
| encoder | ResNet-34 | **~48 M small trunk** | MEASURED: *small's fan is at least as tight as base's at every matched K* |
| anchors | vocabulary | **128** (FPS, nested prefix of the 256 pool) | MEASURED: *"the selected-ADE knee is anchor count, not encoder scale"* |
| horizon | NAVSIM 4 s | **6 s, 8 slots** | PI's binding horizon spec |
| ⭐ hierarchy | **none** — flat planner | **strategic → tactical → operative goal cascade** | the programme's thesis |
| ⭐ decision surface | single trajectory scorer | **factored lat(3) + lon(3)** | MEASURED: the 5-way softmax mixes lat+lon and *"provably destroys the longitudinal decision"* |
| ⭐ selection | learned confidence | **`GoalDistanceScorer`** — distance to the *predicted* goal | SEL-1 refused a learned re-scorer (winner's curse); this is **candidate-independent**, error-rank FALLS with N |

⇒ **We keep the paper's two load-bearing mechanisms and replace the two our own
measurements refuted** (the 5-way collapse, the learned re-scorer).

## 4. ⛔ How we train it — and the real blocker

Registered plan: **Adam lr 1e-4, warmup 2000, cosine, batch 20, 30 k steps,
`--mode diffusion`, 2 denoise steps**, arms `v3-H` (hier) vs `v3-F` (flat),
config delta **derived and REFUSED if it is not exactly the registered lever
set** (C122's rule enforced at build).

| | |
|---|---|
| per 30 k run | **~7–9 h A40** (ESTIMATED; small 30 k MEASURED 7 h 10 m) |
| registered ladder | 2 arms × seed 0 → +4 runs if gates pass ≈ **2.0–2.5 A40-days** |
| where | *"one idle pod A40, sequential; ⛔ never a training pod, ⛔ never Thor mid-run"* |

⛔ **WE HAVE NO PODS.** Thor is the only compute and is 2 days from finishing
v6F S-W. So v3 is **provisioning-blocked**, and that is a PI/spend decision.

**Three routes, with what each costs:**

1. ⭐ **Wait for Thor (~2 days), then run there.** Free. Thor is slower than an
   A40 but v3 is 63 M params at batch 20 — comfortably within Thor. ⚠️ Needs the
   *"never Thor mid-run"* rule respected, i.e. after S-W completes.
2. **Provision one A40 pod** — ≈2.0–2.5 A40-days for the full registered ladder.
   Fastest to a decision-grade result.
3. **Dev-box RTX 4060** — 63 M at batch 20 likely fits in 8 GB with grad
   checkpointing, but ESTIMATED 3–4× slower than an A40 ⇒ ~25–35 h per arm.
   Viable for **seed-0 early read only**, not the ladder.

⚠️ **Route 1 is the honest default** — it costs nothing, and the early read
(2 arms × seed 0) is what decides whether the ladder is worth provisioning.

## 5. ⭐ Further improvements worth considering — as CANDIDATES, with classes

⚠️ **None of these is recommended for the first run.** The registered delta is
pinned and adding to it would void E-V3DOM-1. These are for *after* the early read.

| candidate | source | class | note |
|---|---|---|---|
| **feature-prediction auxiliary** on the ResNet trunk | LAW (`2406.08481`, banked) — *"SOTA nuScenes/NAVSIM/CARLA"* | `PUBLISHED` | LAW adds latent feature prediction as an **auxiliary** to a supervised planner — structurally the same shape as v3, so it grafts without changing the arm |
| **isolated attention mask** (future-video as *training-time* supervision, expert deleted at inference) | SimWAM (`2608.07468`, banked) | `PUBLISHED` | ⭐ the prior is a TRAINING cost, not a deployment cost — compatible with the sub-300 M rule |
| **external feature target** (predict frozen DINOv3 futures) | DeepSight (`2605.10564`, banked) | `PUBLISHED` | ⚠️ a teacher; changes the thesis, same objection as the frozen-encoder route |
| **more denoise steps** | SimWAM Tab. 10: 1 step collapses (68.9), 5 → 90.1, **10 → 90.3**, 20 → 90.2 | `PUBLISHED` | v3 runs **2** (DiffusionDrive's setting). Worth a cheap sweep — but SimWAM is flow matching, not truncated diffusion, so the transfer is **not automatic** |

## 6. ⚠️ The risk I would flag before launch

**v3 is a SUPERVISED arm trained on trajectory** — and the PI's own reminder is
the relevant precedent: *v1f and successors were trained supervised with
trajectory and learned to predict from ego dynamics purely.* A supervised
trajectory arm has already collapsed to ego once in this programme.

⭐ **v3 has a designed defence, and it is real:** **E11 refuses ego state into any
goal head** (pinned by test + the `goal_provenance` intervention audit), so
*"every goal head is a function of frames alone"*. And the pre-registration
carries a **negative control**: *shuffled-goal selection (permute ĝ across the
batch) must NOT beat* the real thing, *"else the H arm is flat-in-disguise —
experiment VOID"*.

⇒ **The echo failure mode is anticipated and instrumented.** That is the single
strongest readiness signal in this review — more than the tests passing.

## 7. What I did NOT verify

* **No forward pass on real data.** The preflight runs on `--synth-episodes`;
  `goal2s_err_m 9.945` is a synthetic number and means nothing about quality.
* **I did not read all 38 KB of `REFC_V3_DESIGN.md`** — the E1–E12 edge list was
  read via the module docstring and the trainer, not end to end.
* **The DiffusionDrive comparison is from the abstract, figures and headline
  numbers**, not a full read of the paper.
* **Thor-fit is ESTIMATED**, not measured. Batch 20 × 63 M on Thor's memory has
  not been probed.
