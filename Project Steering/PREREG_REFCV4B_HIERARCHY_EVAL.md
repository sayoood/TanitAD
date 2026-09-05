# PRE-REGISTRATION — refcv4b post-training HIERARCHY EVAL: eval-time ablations on the FINAL checkpoint, both outcomes committed

**Date:** 2026-09-05 (Europe/Berlin) · **Author/Stream:** Benchmarks & Eval FlyWheel · **Status:**
instruments DELIVERED and STAGED; ⛔ **no ablation has been run on any checkpoint later than step
9,500** (an EARLY-TRAINING DIAGNOSTIC smoke of the metric only, §8). The run this pre-registers is
**`refcv4b-b1-v72-40k`**, LIVE on pod `tanitad-refcv3` at the time of writing; nothing here touches
that pod. **Every ablation below runs AFTER training has finished, on the final checkpoint**, per the
PI: *"why we can not prove the claim of the hierarchy in refcv4, let it do it after finished
training."*

**"Fixed in advance" is made verifiable the D-SEL way:** the falsifiable object is the **git blob id of
this file at staging time**, recorded in the commit that lands it, before any post-training number
exists.

**Estimator, declared before any number.** `taniteval/taniteval/ci.py::paired_episode_cluster_bootstrap`,
resampling unit = **episode**, `n_boot = 2000`, `seed = 0`, every ablation arm PAIRED against the full
model on the SAME windows; single-arm rates by `episode_cluster_bootstrap`. ⛔ `overlapping_holdout_se`
is never called. **Every number carries its tier: T1 (self-action OPEN loop, arm `os`)** —
EVAL_DOCTRINE 2026-09-02: T1 supports open-loop trajectory-prediction claims and **never a claim that the
model drives**. No T2 (closed loop) is provisioned for this arm.

---

## 0. What is being tested, and the honest scope

**The hierarchy thesis, as refcv4b instantiates it (MEASURED from source, `DESIGN_REFCV4_NAV_WIRING.md`
§1 and `…/2026-09-04-refcv4b-seam-state/SEAM_STATE.md`, both re-pinned by content):** the strategic
level (`g_str`, `str_goal_head(ctx + nav_s + ego_s)`, supervised by the 30 s label under `--goal-str`)
conditions the tactical latent (`z_tac`, E4 FiLM) which reaches the operative decoder (E7
`target_latent`) and selection (E9 `goal_gate·scorer`); nav enters the measurement encoder (→
decoder / selector) and, through E13, `z_tac_raw + nav_t` and `ctx + nav_s`; the core's own 3-wide
kin3 heads feed the H19 anchor prior. The claim the PI wants tested is not "the model drives well" but
**"the behaviour follows the nav command in consistency with the strategic goals, and each wired seam
carries something."**

**What CAN be tested at eval time (each edge has a switch that leaves the rest of the forward
bit-identical):** E13 (nav → decisions), E4 (`g_str` → `z_tac`), E7 (`z_tac` → decoder port), E9
(goal → selection), H19 (kin3 prior → anchor confidence), E11' (ego → goal path), the nav channel
itself (zero / shuffle / flip), and the SELECTION SURFACE (`sel_refined`, `sel_score_emitted`; the
implementation audit `7c67b3d` MEASURED that selection is decided at the t=0 confidence on 201/201
windows).

**What CANNOT be tested (stated, not hidden):**
1. **The core's `route_head` readout.** `graft_route = False` (`refc.py:2341-2342`, `:1643-1644`):
   its output reaches nothing, so ablating it changes nothing by construction. A "no effect" there
   is a pin on the wiring, not a finding (C109 class). It is EXCLUDED from every table.
2. **Training-time counterfactuals.** An eval-time knockout measures the trained model's RELIANCE on
   an edge — a lower bound on what the edge bought in training and an upper bound on nothing. A
   zeroed FiLM / withheld nav is an input regime the model saw only at its own init (zero-init
   gates) or under dropout; the arm can be off-distribution. Each ablation therefore states the
   regime it creates and whether training ever produced it (§3, column "seen in training").
3. **Driving.** T1 is open loop. "Follows the command" here means the emitted 6 s path / selected
   anchor / goal bearing turn the commanded way; it does not mean the car would complete the turn.

---

## 1. Instruments (all staged; none may be modified between staging and the run without a new prereg)

| instrument | path | what it produces |
|---|---|---|
| the roll | `taniteval/tools/refcv3_arm.py` (2026-09-05: ego-state fed on v4 builds; nav-compliance sidecar; `--with-navflip`) | the dump (`ep*.npz` + `decisions/ep*.npz` + `manifest.json`) for arms `os ha ha0 os_navshuf os_navzero [os_navflip] [oracle_sel]` |
| four families + controls | `refcv3_arm.py::analyze_refcv3` → `t1_eval.analyze`, `four_families`, `refav1_arm.trivial_profile` | LONGITUDINAL / LATERAL / TACTICAL / STRATEGIC, per arm, paired |
| **nav-COMPLIANCE** | `taniteval/taniteval/nav_compliance.py` (+ `taniteval/tools/nav_compliance_report.py`) | plan / `g_str` / selected-anchor compliance with the TRUE command on informative windows, decided by the shuffle + zero (+ flip) deltas; strata; deferred-consistency; nav-blind floors; verdicts; seam reading |
| echo gate | `stack/tanitad/eval/echo_gate.py` (`echo_gate`, `source_ablation_test`, `assert_not_echoing`) | beat `ha` AND `ha0_ext` per horizon; wrong-scene / wrong-ego degradation; the deliberate-regression arm must FAIL it |
| criteria completeness | `tools/criteria_check.py` against `products/P7-TanitEval/CRITERIA_REGISTRY.json` v2.6.0 (`strat.nav_compliance`, `…_ctrl_shuffle`, `…_ctrl_zero`) | 0 violations or named work items |

**Surface:** the v7.2 EVAL split, 141 clips (`/root/data/eval`, labels md5 `aa12c948f062181c3297265b51526ec5`),
**window stride 5, grid 6s** (the 6 s path is the object; the 2 s grid cannot see a turn), the trainer's
own window contract. Windows whose 6 s future is end-clamped are skipped by the harness
(`horizon_beyond_episode`, 22.9 % at stride 5 on refcv3's dump) — n is reported after the skip.

---

## 2. Hypotheses (registered in `Project Steering/GOALS_AND_CLAIMS.md`, section D-NAVCOMP)

| id | hypothesis | decided by |
|---|---|---|
| **H-NAVC-1** | The final refcv4b's emitted plan FOLLOWS the nav command: plan-compliance drops under nav-shuffle AND nav-zero, separated, by ≥ 0.10 absolute each, on the imminent windows | `nav_compliance.readouts.plan.paired_true_minus_{shuffled,zero}` |
| **H-NAVC-2** | The strategic goal `g_str` FOLLOWS the nav command by the end of training (the zero-init `nav_to_str` edge opens): `g_str`-compliance drops under shuffle AND zero, separated, by ≥ 0.10 | `…readouts.gstr.paired_true_minus_{shuffled,zero}` |
| **H-NAVC-3** | The plan's compliance with the TRUE command clears the nav-blind ego-extrapolation floor `ha0_ext` on the same windows (CI lower bound above the floor's point rate) | `…readouts.plan.conditionings.nav_true` vs `…controls.ha0_ext` |
| **H-SEAM-1** | The E4/E7/E9 seam carries the goal into behaviour: ablating **E7** (`target_latent`) or **E9** (`goal_gate`) degrades the plan's four families (ADE@6 s worse, paired, separated, ≥ 2 % relative) AND lowers plan-compliance | §3 rows E7-off, E9-off |
| **H-H19-1** | The internal tactical seam carries something: **H19-off** degrades TACTICAL decision κ / anchor selection agreement (paired, separated) | §3 row H19-off |
| **H-CONS-1** | On DEFERRED windows (commanded turn beyond the 6 s plan, within 160 m) the plan HOLDS (rate ≥ 0.80) while `g_str` points the commanded way (rate ≥ 0.60, dropping under shuffle) — the PI's "consistency with the strategic goals" | `nav_compliance.consistency_deferred` |
| **H-SEL-1** | Scoring the EMITTED fan (`sel_refined + sel_score_emitted`, 0 params, eval-time switch) changes the pick on ≥ 10 % of windows and does not worsen ADE@6 s (paired, not separated the wrong way) | §3 row SEL-refined |

---

## 3. The ablations — each paired against the FULL model on the same windows

Every arm: the same dump grid, the same windows, the same GT; four families + nav-compliance + echo
gate + the controls `ha` / `ha0` / `ha0_ext` recomputed on the same windows (they are checkpoint-
independent and must read bit-identically across arms — an internal control).

| arm | mechanism (eval-time, forward otherwise bit-identical) | seen in training? | claim it tests | **REFUTES the hierarchy thesis if** | **SUPPORTS if** |
|---|---|---|---|---|---|
| **FULL** | the final checkpoint as trained | — | reference | — | — |
| **nav-ZERO** | `nav_cmd=None` (E13 skipped, core on the `follow` one-hot) — the existing `os_navzero` arm | partially (nav is 64 % `follow`) | nav is used at all | plan Δ_zero not separated or < 0.10 → nav is decoration (H-NAVC-1 false) | Δ_zero ≥ 0.10 separated on plan AND on `g_str` (H-NAVC-1, H-NAVC-2) |
| **nav-SHUFFLE** | token permuted across windows (`os_navshuf`) | no | the model uses THIS window's token, not the marginal | Δ_shuffle not separated → the nav_true rate is scene coincidence | Δ_shuffle ≥ 0.10 separated; changed subset follows the FED command |
| **nav-FLIP** (`--with-navflip`) | left ↔ right on every window | no | the sharpest form of the above | `follows_TRUE` under flip ≈ nav_true rate (token ignored) | `follows_FED` under flip high, `follows_TRUE` low |
| **g_str-ZERO** | `str_goal_head` output replaced by the straight-ahead constant `(1, 0, 0)` before E4/E7 (forward hook on `str_goal_head`) | at init only (zero-init FiLM) | the strategic goal conditions the tactical/operative levels (E4) | plan compliance / four families unchanged (paired Δ ≈ 0, not separated) → the strategic level conditions nothing | separated degradation of plan-compliance or LATERAL family |
| **g_str-SHUFFLE** | `g_str` permuted across windows (batch-level permutation of the head output) | no | the goal carries WINDOW-specific information downstream | as above | as above |
| **E7-OFF** | the hook returns `target_latent=None` (the decoder skips the FiLM: `refc.py tgt_film` on None) | at init only | the tactical latent conditions the decoder | four families / plan-compliance unchanged | separated degradation (H-SEAM-1) |
| **E9-OFF** | `model.goal_gate` set to 0.0 at eval (`blended = sel_score`) | at init (gate zero-init) | goal-distance selection improves the pick | `sel_idx` changes on < 5 % of windows OR ADE not separated | pick changes AND ADE@6 s worse when off (H-SEAM-1) |
| **H19-OFF** | `decoder.maneuver_to_anchor = None` (the kin3-derived 5-way prior removed from the confidence) | no | the model's own tactical prediction improves its selection | TACTICAL κ / anchor agreement unchanged | separated degradation (H-H19-1) |
| **EGO-ZERO** | `ego_state[:, 4] = 0` (keep = 0; the X15 regime) with v0 withheld at the core | yes (ego_dropout 0.5) | the ego channels are used (E11') | — (this is a robustness read, not a thesis test) | reported beside the echo gate |
| **SEL-REFINED** | `sel.refined = True`, `sel.score_emitted = True` (0 params) | no | the selection surface is the lever the audit named | pick unchanged on > 95 % of windows | pick changes ≥ 10 %, ADE not worse (H-SEL-1) |
| **DELIBERATE REGRESSION** | frames replaced by their scalar mean (`--ablate-frames` regime, the image-blind arm) — an echo BY CONSTRUCTION | no | the instruments can FAIL | **if this arm PASSES the echo gate or reads FOLLOWS_NAV with a high compliance, the panel is VOID** (PREREG_REFC_V4 §7 OUTCOME IV) | the echo gate FAILS it and its compliance reads the nav-blind floor |

**Controls that must read known values, before any row is read:** `ha0` compliance = 0.0000 exactly;
every nav-blind control's shuffle/zero delta = 0.0000 exactly; the GT control ≈ 1 on imminent windows;
the label time-base control reproduces `dyaw_deg` on the majority of clips; `ha`/`ha0`/`ha0_ext`
bit-identical across arms. **Any control off its value voids the panel** (the 2026-08-22 rule).

---

## 4. Success and failure, committed now

**SUCCESS for the hierarchy thesis** requires ALL of: H-NAVC-1, H-NAVC-2 (the goal follows nav, not only
the path), H-SEAM-1 (at least one of E7-off / E9-off separated), and the deliberate-regression arm
FAILED by the gate. Then the registry row reads *"nav reaches the strategic decision AND the seam
carries it into behaviour"*, and only then.

**FAILURE modes, each named in advance and each a result:**
- H-NAVC-1 holds but H-NAVC-2 fails → **PATH_FOLLOWS_WITHOUT_GOAL**: nav reaches the operative layer
  through the measurement encoder and NOT the strategic decision — the inversion
  D-REFCV4-NAV-WIRING measured in refcv3, reproduced in refcv4b. The strategic level is then not
  demonstrably in the loop, whatever the ADE says. *(This is the step-9,500 reading, §8.)*
- H-NAVC-2 holds but H-NAVC-1 fails → **SEAM_FAILURE_RIGHT_GOAL_WRONG_PATH**: E4/E7/E9 do not carry
  the goal into behaviour; a hierarchy defect, not a nav defect.
- Neither → the nav channel is decoration at every level; `--nav-from-v7` bought nothing.
- H-NAVC-3 fails while H-NAVC-1 holds → the model follows nav but LESS OFTEN than a constant-curvature
  extrapolation of its own state complies by coincidence: nav-following exists and is weak.
- E7-off / E9-off / H19-off all unseparated → the seams are inert at eval; the hierarchy is a
  training-time story at best, and the registry may not call it "live".

**Margins are absolute, fixed here:** 0.10 on compliance deltas, 2 % relative on ADE@6 s, κ deltas
separated. **A separated CI on a smaller margin is a real but unusable difference and is reported as
such**, never rounded up to support.

---

## 5. What may NOT be cited as support

- A high `nav_true` compliance rate on its own (it can be coincidence; §0, and the registry refuses it
  without its two controls).
- The old route metric (`strategic.conditionings.*.accuracy`): its label is a bijection of the fed
  token (D-REFAV1-ROUTE-LABEL-IS-THE-NAV, 141/141). It stays in the record as the echo diagnostic it
  is.
- T0 numbers (`oracle_sel`, `law`) for any capability claim.
- Any stride-1 / grid-2s re-analysis compared against these stride-5 / grid-6s numbers as levels.
- The step-9,500 smoke (§8) for anything but "the instrument runs and its controls read".

---

## 6. Splits, and nothing tuned on the scored split

The tolerances `tau_plan` / `tau_gstr` are derived from the **GT and the label only** (Youden-J of the
GT signal on informative vs follow windows) and reported with their TPR/FPR; no model output enters
the derivation. The nav-shuffle permutation is seeded (`--nav-shuffle-seed 0`). No hyper-parameter is
selected on any model number.

---

## 7. Deliverable manifest of THIS pre-registration

| artifact | where |
|---|---|
| this file | `repo:Project Steering/PREREG_REFCV4B_HIERARCHY_EVAL.md` |
| the metric | `repo:taniteval/taniteval/nav_compliance.py` + `repo:taniteval/tests/test_nav_compliance.py` |
| the harness additions | `repo:taniteval/tools/refcv3_arm.py` (additive: ego-state on v4 builds, sidecar keys, `--with-navflip`, the analysis block) |
| the standalone report | `repo:taniteval/tools/nav_compliance_report.py` |
| the registry criteria | `repo:products/P7-TanitEval/CRITERIA_REGISTRY.json` v2.6.0 + `repo:tools/tests/test_criteria_check.py` |
| the smoke evidence | `repo:TanitAD Research Lab/Benchmarks & Eval/Research/2026-09-05-nav-compliance-metric/` (RESULT.md + raw/) |

⚠️ **Escalation:** the ablation switches for `g_str`-zero/shuffle, E7-off, H19-off and SEL-refined are
described here as eval-time hooks; they are ONE function each on `RefCV3Model` / the decoder and are
NOT yet implemented as CLI flags of `refcv3_arm.py`. Implementing them is 0 GPU and must precede the
post-training run; it is the next work item of this stream (`--ablate {gstr_zero,gstr_shuffle,
e7_off,e9_off,h19_off,ego_zero,sel_refined}`).

---

## 8. The EARLY-TRAINING DIAGNOSTIC (step 9,500 of 40,284) — instrument validation only

Recorded so the reader can see the metric run on the real checkpoint and its controls read their
values. ⛔ **NOT a result about refcv4b**, which is 23 % trained here; nothing in §4 is decided by it.
See `…/2026-09-05-nav-compliance-metric/RESULT.md` for the numbers, artifact paths and the seam reading.
