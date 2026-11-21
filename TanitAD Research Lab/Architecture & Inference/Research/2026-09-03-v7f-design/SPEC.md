<title>SPEC — v7f design freeze and pre-registration (2026-09-03)</title>

# SPEC — E-ARCH-V7F-1: the v7f pre-registration, in the `TanitAD_ValidateAIDesign` schema

`TanitAD Research Lab · Architecture & Inference FlyWheel · 2026-09-03 · 0 GPU · no pod / no Thor contact`
`Tier: this SPEC produces NO numbers. Every number it quotes is MEASURED (ours, path given),`
`PUBLISHED-PRIMARY (a banked PDF, lib:<key> + table), INHERITED (another agent's read, not re-verified)`
`or UNVERIFIED (flagged). The full evidence table is raw/EVIDENCE_TABLE.md.`

**The full pre-registration is `Project Steering/PREREG_V7F.md`.** This file is the machine-checkable
core the skill's preflight reads; the PREREG carries the gates, the tiny-rig plan, the launch line and
the DECISIONS REQUIRED.

---

## 0. Scope and the honest statement about "one variable"

v7f is a **full training run** and is therefore a **composition of several changes**. A composition is
not attributable, and the skill's one-variable rule cannot be satisfied by the run itself. This SPEC
resolves that the only way it can be resolved:

⛔ **Attribution is carried by the LADDER, never by v7f.** Each new element earns its place on a
one-variable v7-tiny arm with its own deliberate-regression arm, IN ORDER (R0 → R3 below). v7f is then
the composition of the elements that cleared their own rung, and `one_variable` for v7f names the ONE
element that has not been reduced to an already-adopted register decision: **`trunk_policy`**.

⚠️ If a rung is skipped, v7f's prereg is void for the element that skipped it, and the run may not be
described as testing it.

---

## 1. The SPEC block (skill §1)

```yaml
hypothesis: H-V7F-1               # PROPOSED in PROPOSED_REGISTER_ROWS.md; must be
                                  # applied to GOALS_AND_CLAIMS.md before launch
one_variable: trunk_policy        # {frozen | full | anchored | lastk}
held_constant:
  - corpus                        # physicalai-b1-w120-256x640cyl, 4713 clips,
                                  #   episode_uid_sha256 e8bfb98e06eb…, --require-parity
  - eval_exclusion                # the 141 eval clips with pixels EXCLUDED (BACKLOG R8)
  - labels                        # s2-geom-v7 (v7.2), md5 0ff90213… train / aa12c948… eval
  - nav                           # --nav-labels --nav-cond, all three layers
  - seed
  - steps                         # one full epoch (D-ONE-EPOCH)
  - batch
  - window                        # W = 6
  - o5_k                          # 60 (the 6 s contract)
  - bptt_truncate                 # fixed by rung R1 before the trunk arms run
  - w_ldad                        # lambda fixed by rung R2 before the trunk arms run
  - cond_param                    # omega_accel_v, entering at the PREDICTOR ONLY
  - encoder_input                 # identical across arms (see PREREG §4.3)
success: >
  At the checkpoint selected by the two-key rule (PREREG §7), the anchored action-divergence read
  (taniteval/tools/actdiv_anchored.py, h=1, trainer lift v_last/10, the 24-clip/144-window banked
  actdiv population) returns verdict SENSITIVE: F_sep >= 5x the arm's own 200-permutation null p95
  on BOTH axes, Spearman rho >= 0.8 on BOTH axes, cos(m(+L), m(-L)) < 0 at EVERY level on BOTH axes,
  and rel_pinned(2 sigma) >= 0.0595 with its clip-bootstrap interval excluding 0.0595.
  ⛔ rel_pinned = per-dim RMS of m(2 sigma) DIVIDED BY THE SCENE SPREAD OF THE NAMED R3
  scratch-trunk REFERENCE ARM --- never the arm's own scene spread (PREREG_MM_E19_K60_HORIZON.md
  DEFECT 1: "a criterion whose threshold is a function of the result is not a pre-registration";
  the k=60 arm's scene_factor 1.7126 silently turned a 10x bar into a 17.13x bar).
  Bar provenance: 10 x the incumbent postrain30k's 0.005947 (HORIZON-WORKS, :61); incumbent
  anchored 0.0087, best arm ever measured rdw8p30k 0.0181.
  --- AND the Observer-Effect monitor's frozen-feature linear probe on the DYNAMIC targets stays at
  >= 0.70 of its own step-0 value with the paired episode-cluster-bootstrap CI on the ratio excluding
  0.70 --- AND G-DRIVE (T1) reports cl - ha ADE separated below 0 with the paired episode-cluster
  bootstrap over the labelled eval split, with trivial_profile_fraction < 1.0 and baseline_won_frac
  below the refav1 reference 0.7571.
failure: >
  ANY of: (a) rel_pinned(2 sigma) < 0.0595 at every step-stamped checkpoint --- the encoder-shaping
  family (LDAD/Delta-JEPA, SMWM, EB-JEPA-IDM) is REFUTED for TanitAD on a realised-motion action
  channel, and the next line is predictor-side (ACID / ActSWM hinge / AD-JEPA offset head), which is
  frozen-trunk-compatible and does not need this run; (a-WORSE) rel_pinned(2 sigma) BELOW the
  incumbent's 0.0087 with the interval excluding it --- the trunk change REDUCED action sensitivity;
  reported as G-ACT-WORSE, never force-fitted onto "inert" (MM-E19 DEFECT 2: its outcome set admitted
  no worsening and the arm fell 0.48x, off the table entirely); (b) the Observer-Effect monitor falls
  below 0.70 x step-0 --- fine-tuning corrupts the trunk faster than the anchor holds it, and the
  design reverts to frozen DINOv3 + a trained per-token adapter (lib 2602.18639); (c) G-DRIVE returns
  a VOID read (arms bit-identical, or trivial_profile_fraction == 1.0) --- reported as VOID, never as
  "no difference" (D-REFAV1-PAIRED-READ-VOID).
co_primaries_never_folded_away:                # MM-E19's "how arms on this statistic are judged"
  - action_side_unnormalised                   # the scene side carried 73.7% of MM-E19's ratio fall
  - own_scene_spread
  - h_ge_2_floor                               # ~1e-05 on EVERY arm measured; horizon-INDEPENDENT
  - training_window_count_per_arm              # o5_k also sets o4_n: 415,002 -> 319,002 (76.9%)
  - clip_bootstrap_interval                    # backlog L-13; the gate reads the interval, not the point
controls:
  - constant_only                 # must read the no-information value EXACTLY
  - raw_input_floor               # 8x8 raw pixel difference, on every decodability panel
  - deliberate_regression         # per rung, listed in section 3
  - shuffled_action               # per-arm, 200 within-window permutations (MEASURED null)
  - endpoint_shuffled             # dz' = z_{pi(t)+1} - z_t (H-LEAK-3's control)
  - zero_model                    # the real predictor fed the zero action -> max|d| == 0.0 exactly
  - c0_identity                   # same inputs twice -> max|delta| == 0.0 exactly
splits:
  fit:  "B1 train partition, 4,572 clips minus the eval index; every hyper-parameter (lambda,
         PCA basis, ridge lambda, checkpoint selection) fitted here and NOWHERE else"
  val:  "carved from FIT by clip, episode-disjoint; used for checkpoint selection and lambda"
  test: "the v7.2 labelled eval split (141 clips with pixels); SCORED, NEVER TUNED ON"
```

---

## 2. Preflight refusals (the skill's §1 list, instantiated)

The run is REFUSED if any of these is true at launch time:

| # | refusal | check |
|---|---|---|
| 1 | `H-V7F-1` is not in `GOALS_AND_CLAIMS.md` | grep the register |
| 2 | two arms differ in more than `trunk_policy` | **diff the launch commands, not the intent** (C164) — `--print-launch` output of every arm, diffed pairwise |
| 3 | a control is missing from any panel | the panel's own JSON must carry `c0`, `zero_model`, `constant_only`, `pixel_floor`, `null_p95`, `n`, `d` |
| 4 | any hyper-parameter was selected on the scored split | the selection log names the split |
| 5 | **the eval clips are not excluded** (BACKLOG R8) | the window census of a `--require-parity` launch must show **0 eval clips** |
| 6 | `--bptt-truncate` >= `--o5-k`, or negative | trainer preflight (already implemented, D-V7-WIRING) |
| 7 | `--allow-any-labels` without `--s2-labels` | trainer preflight (already implemented) |
| 8 | the trunk init checkpoint's provenance is not recorded | the seed checkpoint must carry `{"dinov3_model_id", "sha256", "adapted_keys", "unadapted_keys"}` |
| 9 | O1 is not at zero **and** `--bptt-truncate` does not reach the stage-A rolls | BACKLOG R6; **MEASURED not to bind at `--w-o1-* 0`** — `train_v6_staged.py:3691` guards the whole O1 block on `if w.o1_ctrl or w.o1_fact or w.o1_scene:` |
| 10 | the participation reference is quoted as a bare scalar | **`O6_PARTICIPATION_FLOOR = 8.56` is NOT reproducible** (`v6.py:1519-1552`, registry §13.3) — see PREREG §6 G-RANK |

---

## 3. The rungs, in order — each one variable, each with its regression arm

| rung | one variable | deliberate-regression arm (must FAIL the rung's gate) | GPU |
|---|---|---|---|
| **R0** | *(none — a probe, not an arm)* LDAD tautology test at step 0 on the DINOv3-init trunk | the **constant-only** decoder, which must read the no-information value exactly; and the **raw-pixel-difference floor**, which LDAD must beat or it has added nothing the input did not carry | 0 GPU / minutes on the 4060 |
| **R1** | `bptt_truncate` ∈ {0, 2, 4, 15} at fixed `o5_k = 60` | `bptt_truncate = 0` — the full chain, which is the arm that **diverged at gnorm 2.1e9** and must diverge again, or the stability gate is void |
| **R2** | `w_ldad` (λ) ∈ {0, 1, 10, 50} at the R1-selected truncation | **λ = 0** — Delta-JEPA's own near-collapse arm (`2606.31232` Fig. 3), which must NOT clear G-ACT |
| **R3** | `trunk_policy` ∈ {frozen, full, anchored, lastk} | **`full`** — the naive full fine-tune, which the Observer-Effect monitor MUST trip (`2602.12218`: probe rho 0.91 frozen -> 0.05 full FT). If the monitor does not trip `full`, a PASS on `anchored` means nothing. Second regression arm: **`frozen`**, which must reproduce `postrain30k_freeze`'s dead command channel (steer/accel ratio 0.00078 vs the trainable incumbent's 0.00561, 7.2x lower) |
| **R4** | *(the composition)* v7f at scale | **v1.7 itself** on the L4 harness — banked 0.9785 open / **0.0430** closed, hold-action **0.0 %**. A harness that does not reproduce that collapse cannot be trusted to detect it in v7f |

---

## 4. Verification by content, never by exit code (skill §5)

- Every rung's artifact is asserted on **bytes**: the checkpoint exists, is > 50 MB, loads with
  `missing_keys == []`, and `config.json` stamps the flags the launch claimed.
- The `--print-launch` string of every arm is banked next to its result; the pairwise diff is the
  one-variable evidence.
- ⛔ **A generated bank is asserted on CONTENT** — sample rows, require non-zero, print the mean —
  before it is used (the all-zero-floor false-positive class).

## 5. Closing the loop (skill §6)

Same turn as the result: `GOALS_AND_CLAIMS.md` gets `H-V7F-1`'s status + evidence path; `RESULT.md`
carries evidence class and tier stamps; raw JSON under `raw/`; `MODEL_REGISTRY.md` gets a row if a
model version results; a reusable procedure becomes a skill.
