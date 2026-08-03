# PRE-REGISTRATION — D-SEL: REF-C's defects are on the SELECTION surface, and rebuilding only that surface costs +385 parameters

**Date:** 2026-08-03 (Europe/Berlin) · **Author/Stream:** arch-inf (REF-C model stream; DIRECTION 1 of 3
of the PI's REF-C improvement directive) · **Status:** code + instruments **DELIVERED and STAGED**, no
training launched, **0 GPU spent**.

**"Fixed in advance" is made verifiable in §10.1, not asserted.** The D-TAC1 adversarial pass (R11)
refuted that pre-registration's *"thresholds fixed in advance"* claim by mtime alone — the prereg was
written 6 minutes AFTER its own probe JSON, so the claim was **INHERITED, not MEASURED**. Here the
falsifiable object is the **git blob id of this file at staging time**; no arm has run.

**Estimator, declared before any number.** `taniteval/taniteval/ci.py::paired_episode_cluster_bootstrap`,
resampling unit = **episode**, `n_boot = 2000`, on the canonical val set
`physicalai-val-0c5f7dac3b11`. Two arms on the same windows use the **paired** form, never a combination
in quadrature. ⛔ **`overlapping_holdout_se` is never called** — it is not a jackknife, it is not a valid
SE, and it BIASES the point estimate (mean-of-split-means, measured −6.67 % to +11.69 % over 27 arms,
bidirectional, up to a sign flip on paired deltas).

---

## 0. The claim under test

> **Every MEASURED defect of REF-C is on the SELECTION surface — on WHICH candidate is emitted, not on
> which candidates are proposed. Therefore the highest-value port from the flagship v5f lineage is not a
> new head, a new input, or more capacity: it is the flagship's own repair of REF-C's decoder, brought
> home, plus the three things that repair needs to be measurable.**

| # | defect | primary source (file / artifact) | class |
|---|---|---|---|
| D1 | the refined fan is ranked by the **UNREFINED** score | `stack/tanitad/refs/refc.py`, denoise loop (pre-D-SEL: `_, off = self._decode(...)`); repaired upstream by `flagship_v15.V15Decoder` ("THE SCORING FIX") | MEASURED (source) |
| D2 | the pick is >2× worse than the fan's best in **45.4 %** of windows (REF-C-**XL**; base 41.09 %) | `taniteval/results/scaleab_refc-base-30k_vs_refc-xl-30k.json` → `gap["refc-xl-30k"]`; duplicated in `taniteval/results/planfan_clips_summary.json` → `meta.arm_summary` | MEASURED |
| D3 | **72.08 %** of the emitted fan is not physically flyable — and deleting it is **exactly inert** | `…/incoming/2026-07-27-percandidate-labels/raw/t1_clip_fansize.json` (`surfaceA.source = fan_refc-xl-30k.pt`) | MEASURED |
| D4 | the grafts that reach the score are **UNCLAMPED**, while their norms are already LOGGED | `stack/scripts/refc_train.py` logs `graft_lat_norm` / `graft_lon_norm` / `conf_norm`; no actuator existed | MEASURED (source) |
| D5 | the **consequences** of candidates never reach the ranking | `flagship_v15.IMAGINATION_HAS_CANDIDATE_AXIS = False`; `…/2026-07-27-percandidate-labels/raw/t4_imagination_conditioning.json` | MEASURED |
| D6 | the manoeuvre head is **READOUT**-limited, not input-limited (`auc_lon_active` 0.7294 vs a ≥0.65 threshold fixed in advance) | `…/incoming/2026-08-03-dtac1-tactical-head/dtac1_probe_refc-base-30k.json` | MEASURED |
| D7 | the route can only **warp** the condition, never **choose**; `route_head(pooled)` is nav-blind by architecture | `RETRACTION_LOG.md` R-2026-08-03-l (`nav_passthrough_rate 0.0000`, logit std across navs exactly 0.0) | MEASURED |

D6 is the one that is *not* fixed here and is worth saying plainly: D-TAC1's factorised head (`+897`) and
F1 input arm (`+384`) already exist, default OFF. **D-SEL is their precondition, not their competitor** —
see §3.

---

## 1. The defect, as a chain of MEASURED facts — and what that chain does NOT license

1. REF-C-XL's selected ADE@2s is **0.4714** [0.3896, 0.5556] while the best trajectory in its own emitted
   fan is **0.1640** [0.1414, 0.1902]; `sel_gap` **0.3075** [0.2397, 0.3778], `frac_sel_2x_worse` **0.454**.
   *(`scaleab_refc-base-30k_vs_refc-xl-30k.json`.)* REF-C-base: 0.4728 / 0.1914 / 0.2813 / 0.4109.
2. The 0.16 m plan is therefore **already in the fan**. That is a ranking statement, not a coverage one.
3. The decoder ranks with the t=0 classifier score over post-denoise trajectories (source, D1).
4. 72.08 % of those trajectories are outside a bounded-acceleration band around `v0`; removing them leaves
   the ADE-oracle intact (100 %) and the selected ADE **bit-unchanged** (`as_trained_ade` 0.4714 ==
   `clipped_ade` 0.4714, paired Δ **exactly 0.0**), with **0.00 %** empty survivor sets.

⛔ **THE CHAIN DOES NOT LICENSE "THERE IS 0.31 m OF HEADROOM."** `MODEL_REGISTRY.md` §4.1 carries a
standing caveat in the opposite direction: *the oracle gap is ~92 % irreducible — stop quoting it as
available headroom*. REF-C v1.2's learned re-scorer, across **47 trained arms**, recovered at most **8.4 %**
of it, and its headline (0.46251 vs 0.47144) is **NOT separated** (+0.00893 [−0.0062, +0.0250]).

This pre-registration is written under that adverse prior, on purpose. R-2026-08-03-d exists because *"a
chain of MEASURED links does not make the conclusion MEASURED"*, and R-2026-08-03-dtac1 exists because
*"a mechanism that is real in the source is not thereby the binding constraint."* Both apply here.

**The one thing that is genuinely different in kind, and the whole reason to run:** v1.2 re-ranked a
**FROZEN** decoder *post hoc*, so it could only exploit information the frozen confidences already
carried. S1 puts the ranking objective **inside training**, so the refined readout is *shaped by* it.
That is a real distinction — and "different in kind" is an argument, which this programme settles with an
experiment, not with an argument. §6 is that experiment. §6.3 commits both outcomes.

---

## 2. The mechanism, in source

### 2.1 Why the ranking cannot currently improve — and why `argmax` is the reason

`argmax` has **no gradient**. In REF-C's loss the only consumers of the selected trajectory are `traj`
and `law_pred` (through `traj`), and both differentiate w.r.t. the **fan**, through a detached index.
So **nothing in REF-C's loss has ever differentiated w.r.t. the score the selector ranks on.** The
refined confidences were discarded, and the score that survived was trained only as a *vocabulary
classifier* (`anchor-cls CE` against the GT-nearest ORIGINAL anchor) — a different question from *which
refined trajectory is best*.

⭐ **This was confirmed empirically, by the tests, during implementation.** The first run of
`test_rank_grafts_are_gated_not_dead` reported `cons_gate.grad is None`: a zero-init graft on the ranked
score received **exactly zero** gradient. `compute_losses` now builds the ranked-score CE for **every**
lever that touches that score. `flagship_v15.v15_losses` states the identical mechanism for its own
`sel_gate`: *"supervising the score rather than the bare logits is also what gives the longitudinal gate
a gradient — `argmax` has none."*

⇒ **S1 is two changes that cannot be split**: rank on the refined confidence AND supervise it. Ranking on
an unsupervised readout ranks noise; supervising a readout nothing reads trains a dead head. The arm
therefore carries a **new loss term** (`REFINED_CLS_WEIGHT = 1.0`, the value `flagship_v15` uses), and
that is stated here rather than hidden behind a flag name.

### 2.2 Why the flagship's `cond_imagination` transfers only in ONE form

`flagship_v15` calls it "THE NOVEL PART": probe action sequences rolled through the frozen predictor,
imagined latents fed as conditioning. It does **not** transfer to REF-C, for two independent reasons:

* **No candidate axis.** MEASURED (`t4_imagination_conditioning.json`): `imagine_probes` returns 32
  tokens, invariant to `n_anchors`, **identical for all 256 candidates**. It can condition a decode; it
  can never rank one. REF-C's measured defect is ranking. (This is why E-V5-1's imagination-scoring
  negative is *over-determined* — the experiment could not have worked.)
* **No rollable predictor.** The flagship rolls a frozen 20-step latent predictor over ACTIONS. REF-C's
  world model is `law_head`: `[pooled, traj] → pooled_{t+5}` (`refc_train.LAW_AHEAD = 5`). It consumes a
  TRAJECTORY and emits a POOLED vector, so it **cannot be iterated** — there is no `fmap` to re-decode
  from. A probe roll is structurally unavailable.

The form that **does** transfer is `imagine_candidates`: one consequence **per candidate**. REF-C can
afford exactly that — `law_head` is a single MLP evaluation per candidate, and D3 makes it 3.58× cheaper.
That is S3, and its output has a real candidate axis, asserted at runtime.

### 2.3 Why the route must reach SELECTION, not only the condition

LAN E0 (`…/incoming/2026-08-03-lan-refc-e0/E0_refc-base_navcf_full.json`) measured that REF-C's nav
pathway is LIVE — `reachable.max_pairwise_mean_m = 0.24159` against a `control_max_disp_m = 0.0` — and
that supplying the **ORACLE** route makes **cross-track separated WORSE** (+0.0031 [+0.0001, +0.0063])
and **curvature separated WORSE** (+0.0013 [+0.0003, +0.0024]), while ADE is not separated. A pathway
that can only *warp every candidate* and never *choose among them* is exactly what that looks like.

⚠️ Quote `reachable.max_pairwise_mean_m`, **not** the top-level `max_pairwise_mean_m = 1.768415`: nav
index 3 is unreachable (`refc_eval.ROUTE_TO_NAV` maps 3 route classes onto nav {0,1,2}; row 3 is
untrained), and the top-level field overstates route sensitivity **7.3×**.

---

## 3. Three things this is NOT competing with

| existing lever | relationship to D-SEL |
|---|---|
| D-TAC1 F2 `factored_maneuver` (+897) — `lon_to_anchor`'s "effect on trajectories is UNTESTED" | **D-SEL is its precondition.** `lon_to_anchor` adds to a score that (a) is not the ranked object after denoising, (b) ranks over a fan that is 72 % unflyable, (c) is uncapped. S1+S2+S4 are what make an `lon_to_anchor` measurement mean anything. |
| D-TAC1 F1 `tactical_speed_input` (+384) | orthogonal (an INPUT lever). `ego_valid_channel` is its natural companion and is deliberately **kept out of the D-SEL preset** so neither arm is confounded by the other. |
| LAN `graft_lan` | LAN's param-free geometric `lan_gate` is already a route→**selection** pathway, and it is the **higher-prior** one (route coverage 0.8801 vs `nav_cmd`'s 0.2724). S5 tests the *readout* pathway, which is the lower-prior one — see §6.3. |

---

## 4. What is implemented and staged

`tanitad/refs/refc_select.py` (new) · `tanitad/refs/refc.py` · `scripts/refc_train.py` ·
`tests/test_refc_select.py` (new). All flags default OFF; an all-off build is **byte-identical**
(state_dict keys **and** values) and **bit-identical in the forward** to pre-D-SEL REF-C — verified
against `git show HEAD:stack/tanitad/refs/refc.py` across 6 flag combinations (`{}`, factored+F1, LAN,
grounded, refc1, imagination): `keys=True vals=True FORWARD-BIT-IDENTICAL=True` for every one.

| id | lever | mechanism | new params (base) |
|---|---|---|---|
| **S1** | `sel_refined` | keep the last denoise pass's confidence, rank on it, and supervise it (`cls_refined` CE against the GT-nearest REFINED trajectory) | **0** |
| **S2** | `sel_reach_clamp` | bounded-acceleration band on the candidates, ARGMAX only; masked off wherever ego-dropout withheld `v0` | **0** |
| **S3** | `graft_cons` | per-candidate consequence through `law_head` (under `no_grad`), projected by the decoder's OWN `feat_proj`, scored by its OWN `conf_head`, scale fixed by a param-free `layer_norm`; one zero-init gate | **1** |
| **S4** | `seam_clamp` | in-graph per-sample norm cap on the TOTAL graft per surface + sustained-saturation fail-loud | **0** |
| **S5** | `graft_route` | zero-init `Linear(N_ROUTE, n_anchors, bias=False)` from the strategic readout onto the ranked score | **384** |
| — | `ego_valid_channel` | X15: explicit "v0 is present" flag for the measurement encoder and (with F1) the tactical head | 128 (+384 with F1) |

**MEASURED capacity — the control, not an estimate.** `param_breakdown`, pinned by
`test_dsel_is_not_a_capacity_change`:

```
refc_config()        104,191,577
refc_select_config() 104,191,962      delta = +385  (+0.00037 %)
```

Same order as D-TAC1's F1 arm (+384) and **~1/700** of the +272,001 an earlier two-MLP tactical head cost
before its own capacity check caught it. `param_breakdown` reports `selection` as a **carve-out** of
`decoder`, so the table still sums to `total` exactly. C34's rule — *match capacity before attributing an
effect to information; report the marginal, not the total* — is satisfied by construction.

### 4.1 Architecture choices argued against their named alternatives

* **Reuse `feat_proj` + `conf_head` for S3 instead of a dedicated projection.** A dedicated `Linear(F, d)`
  would cost ~270 k on base. Reuse costs 0 and is *semantically* the right object: the consequence latent
  lives in the same `feat_dim` space as the conv-map tokens, and scoring it with `conf_head` asks the same
  "is this a good plan" question the decoder already learned. The coupling this creates is real, starts at
  exactly zero gradient (the gate is zero-init — pinned by test), and is ablatable to zero.
* **`cons_detach = True` (world model under `no_grad`).** The flagship's `_imagination_inputs` rolls under
  `no_grad` for the same reason. Proved by CONTRAST in the tests, not by reading the flag: with the LAW
  MSE removed, `law_head` receives **exactly 0.0** gradient when detached and **> 0** when not.
* **Two additive rank grafts, not one Linear over a concatenation.** Same argument the factored tactical
  seam makes: separable terms are individually ablatable; a concatenation is not.
* **The seam fail-loud is population-over-time, not a batch max.** The flagship's first version fired on
  `ratio.max()` and one sample of 64 could kill a run — MEASURED, it lost BOTH wide arms of a geometry
  validation at ~step 350 on arms training at or below their control (C51). D-SEL requires `mean > fail`
  **AND** `fail_frac` of the batch clamped **AND** `patience` consecutive steps, and resets on any break.
* **NOT ported, argued from source:** `imagine_probes` (§2.2) · `vision_rank` (measured on a FLAT 2048-d
  state entering a flat reader; REF-C's decoder cross-attends 64 spatial tokens and its tactical head reads
  a mean-pooled vector — porting it would be an unmeasured capacity CUT, the same evidence-transfer error
  `V4Config.sel_reach_clamp` refuses in the other direction) · `lambda_plan` (there is no warm-started
  trunk/planner boundary in REF-C; one optimizer, from scratch) · prior-corrected decoding at `τ > 0`
  (D-TAC1B: NOT separated on the 1232 representable windows, +0.0107 [−0.0418, +0.0665], and it costs
  precision 0.2340 → 0.1711).

### 4.2 Self-consistency controls, with fail-loud runtime guards

| control | what it refuses | where |
|---|---|---|
| **candidate-axis guard** | a consequence score that is CONSTANT along the candidate axis — i.e. one that ranks nothing. This is the exact silent failure `imagine_probes` shipped with for months. It is **the flagship's own guard object**, re-exported (`sl.NoCandidateAxis is flagship_v15.NoCandidateAxis`), not a second implementation. | `refc_select.consequence_scores`, every forward |
| **dead-parameter guard** | a D-SEL parameter with `grad is None` or zero after the first backward. A zero-init graft is *gated*; a zero-init graft with no gradient is *dead*, and the weight cannot tell them apart. | `refc_train.assert_selection_params_are_alive`, once per run, printed as `d_sel_gradients` |
| **seam saturation guard** | a graft sustained above the clamp for `patience` steps — where the graft's strength is a no-op and any strength sweep would read SATURATION as a finding | `refc_select.apply_seam_clamp` |
| **C6 guard** | `--graft-route --labels v1`, refused at parse time: that route target is `route_target(nav_cmd)`, a deterministic function of a model INPUT | `refc_train.train` |
| **S1 identity** | `steps == 0 ⇒ refined_logits IS anchor_logits` (identity, not approximate equality), so `--mode classifier` is provably a control arm | pinned by test |
| **empty-set fallback** | a window whose survivor set is empty keeps its whole fan — an unreachable-everywhere window is a measurement failure, not a licence to emit nothing | `refc.AnchoredDiffusionDecoder.forward` |
| **preflight banner** | prints every D-SEL axis + the measured `n_selection_params` + `s1_inert_because_classifier_mode` before step 0 | `refc_train.train` |

---

## 5. ⚠️ Instrument hazards that bind this experiment (read before reporting anything)

1. **R-2026-08-03-c: every published four-family ABSOLUTE rate before the fix is wrong by 5×–25×**
   (`DT_S = 0.1` hard-coded against a 0.5 s waypoint grid; speed /5, accel /25, curvature /8.36, heading
   /1.90). **Cross-arm ranks and paired deltas survive; absolute rates do not.** D-SEL reports paired
   deltas on the same windows, which is the surviving quantity — and re-states absolute rates only from a
   post-fix instrument run.
2. **R-2026-08-02-a: REF-C was once scored at the WRONG RASTER** (176×624 = 120 tokens vs the trained
   8×8 = 64). **XL crashed; base returned numbers silently.** Every D-SEL eval asserts the 256×256 square
   raster before it scores. "When two arms share a defect and only one crashes, the crash is the honest
   signal."
3. **`0.1640` / `45.4 %` are REF-C-XL's, not base's.** Base is 0.1914 / 41.09 %. The arm travels with the
   number or the number is not quotable.
4. **`dist_to_gt_traj_m` ≡ `cross_track_abs_m`** (byte-identical in `metrics_empty.json`). Four lateral
   metrics, not five. And *"the separation is ENTIRELY LATERAL"* is **RETRACTED** (R-2026-08-03-C); no
   argument in this document depends on it.

---

## 6. ⭐ THE CHEAPEST DISCRIMINATING EXPERIMENT — run BEFORE any GPU-day

**E-SEL-0 costs ~10 GPU-minutes and needs no training at all.** S1's mechanism can be tested on the
**already-trained** `refc-base-30k` / `refc-xl-30k` checkpoints, because the refined confidences the
decoder used to throw away are computable from the existing weights.

### 6.0 Negative controls, run FIRST and reported alongside (non-negotiable)

| control | what it establishes | "nothing is quotable" trigger |
|---|---|---|
| **C-identity** | with every D-SEL flag off, re-scoring the banked fan reproduces the published selected ADE **bit-for-bit** | any deviation > 1e-6 ⇒ the harness changed something else; STOP |
| **C-shuffled** | rank on a *permuted* refined-confidence vector | if shuffled ranking scores as well as refined, the refined readout carries no ranking information and S1's premise is dead |
| **C-raster** | assert 256×256 / 8×8 = 64 tokens before scoring | a mismatch ⇒ R-2026-08-02-a; refuse to score |
| **C-oracle-floor** | oracle-in-fan recomputed on the same windows | must reproduce 0.1640 (XL) / 0.1914 (base); if not, the fan dump and the published numbers disagree |

### 6.1 E-SEL-0 — is the DISCARDED refined confidence a better ranker than the one we ship?

**Threshold-free statistic**, computed on the 881 canonical val windows from the banked fans
(`taniteval/results/fan_refc-{base,xl}-30k.pt`): the paired ADE@2s difference between ranking with the
t=0 classifier score (shipped) and ranking with the **refined** confidence — plus `rank_acc` and
`frac_sel_2x_worse` under each.

⚠️ **This is a LOWER BOUND on S1, not S1.** The banked refined confidences were never trained as a
ranker; S1's claim is that *supervising* them makes them one. A null here does not falsify S1 — it bounds
how much of S1 is free. §6.3 encodes that asymmetry explicitly.

### 6.2 E-SEL-1 — does the CONSEQUENCE carry candidate-discriminating information?

On the same banked fans, with `refc-base-30k`'s own `law_head`, compute
`cons_i = law_head([pooled, fan_i])` for every candidate and score the **rank correlation** between
`−‖cons_i − z_{t+5}‖²` (the consequence's agreement with what actually happened next) and the candidate's
true ADE. Threshold-free: Spearman ρ, with its episode-cluster CI, against **C-shuffled** (the same
statistic on a permuted candidate axis).

This is the cheapest possible test of D5, and it answers a question no amount of training can un-ask:
*is there any candidate-discriminating signal in REF-C's world model at all?* If ρ is not separated from
the shuffled control, S3 is a mechanism without information and must not consume a GPU-day.

### 6.3 ⛔ BOTH OUTCOMES, COMMITTED IN ADVANCE — thresholds fixed HERE

| branch | trigger (fixed now) | what it means | what happens next |
|---|---|---|---|
| **S1 FREE WIN** | E-SEL-0 paired ΔADE@2s separated **and ≥ 0.02 m** better | the refined confidence was already the better ranker and we were discarding it | ship S1; the retrain arm becomes a confirmation, not a discovery |
| **S1 NEEDS TRAINING** | E-SEL-0 not separated, **and** C-shuffled clearly worse than refined (`frac_sel_2x_worse` gap ≥ 0.05) | the refined readout carries ranking information but was never optimised for ranking — exactly S1's claim | run the retrain arm (§7); this is the **expected** branch |
| **S1 DEAD** | E-SEL-0 not separated **and** C-shuffled indistinguishable from refined | the refined confidence carries no ranking information at all; S1's premise is false | **do not retrain.** Report it. The 92 %-irreducible caveat wins and D-SEL reduces to S2+S4 as hygiene |
| **S3 LIVE** | E-SEL-1 Spearman ρ separated from C-shuffled, `\|ρ\| ≥ 0.10` | the world model discriminates candidates; the `cond_imagination` port has information to work with | include S3 in the retrain arm |
| **S3 DEAD** | ρ CI includes the shuffled control | REF-C's LAW head is candidate-blind in practice | **drop S3 from the arm** and say so; do not reframe it as "needs training to emerge" |
| **S5 (registered as the LOWEST-prior lever)** | — | `route_head(pooled)` is nav-blind by architecture, its junction accuracy 0.7613 sits **BELOW** that scene's majority baseline 0.7806, and an image-free NAV_ECHO table beats it by 0.1724 | **I expect S5 to be NULL.** It is included because zero-init means training decides and it is the only test of the readout pathway. If the learned `route_to_anchor` norm stays at ~0, that is the answer and it gets reported as such |
| **INDETERMINATE** | E-SEL-0 separated but < 0.02 m | too small to justify a GPU-day on its own | tie-break on E-SEL-1: S3 LIVE ⇒ run the arm (S1+S2+S3+S4); S3 DEAD ⇒ ship S2+S4 only |

**Registered personal prediction:** *S1 NEEDS TRAINING* (E-SEL-0 not separated, C-shuffled clearly worse)
and *S3 LIVE at a small ρ, 0.10–0.25*. **If E-SEL-0 comes back a FREE WIN ≥ 0.02 m, I was wrong about
where the defect sits, and I will say so in the results file rather than reframe it.** If S1 comes back
DEAD, the 92 %-irreducible caveat was right and this whole document was an expensive way to confirm it —
which is also a result, and it gets written that way.

### 6.4 The command lines

```bash
# E-SEL-0 / E-SEL-1 — 0 training, on the banked fans (eval pod or Thor)
OMP_NUM_THREADS=6 python scripts/refc_sel_probe.py \
    --fan taniteval/results/fan_refc-base-30k.pt \
    --ckpt /root/models/refc-base-30k/ckpt.pt \
    --out  "TanitAD Research Hub/.../incoming/2026-08-XX-dsel-probe/" \
    --controls identity,shuffled,raster,oracle-floor
```
*(`refc_sel_probe.py` is NOT written by this stream — it is §9's escalation item. E-SEL-0/1 read only
banked artifacts and an existing checkpoint, so they are 0-GPU-day and can run on any free box.)*

---

## 7. IF a retrain is warranted — the arm, and how it is judged

Identical data, identical parity key `physicalai-train-e438721ae894`, skip-hash `f09e44db`, identical
optimizer/schedule/seed. **The ONLY differences are the named flags.**

| arm | flags | isolates |
|---|---|---|
| `refc-base-30k` | — (the published control) | — |
| `dsel-full` | `--sel-refined --sel-reach-clamp --graft-cons --seam-clamp 1.0 --labels v21` | the selection surface |
| `dsel-s1only` | `--sel-refined --labels v21` | S1 alone (the ranking objective) |
| `dsel-nocons` | `--sel-refined --sel-reach-clamp --seam-clamp 1.0 --labels v21` | `dsel-full − dsel-nocons` = S3's marginal |
| `dsel-route` | `dsel-full` + `--graft-route` | S5's marginal (only if S3 is LIVE; else it confounds two weak levers) |

⚠️ `dsel-s1only` vs `dsel-nocons` is **not** a clean S2+S4 estimate — S2 is MEASURED inert on ADE, so its
value is compute, and S4 only acts when a graft saturates. Both are reported as **telemetry**
(`reach_frac_*`, `seam_*`), not as ADE deltas. Claiming an ADE effect for S2 would contradict its own
measurement.

### 7.1 The four metric families — per family, never pooled

Binding rule (Sayed, 2026-08-02). Each family carries the **paired episode-cluster bootstrap** and its CI
on the **same windows** as the ADE it accompanies. ADE is **one row of five**.

| family | metric | pre-committed direction | role |
|---|---|---|---|
| **ADE** | `ade_0_2s`, `fde_2s` | improve or hold | headline, never alone |
| **LONGITUDINAL** | target-speed accuracy, `speed_bias` (**signed** — an arm can pass MAE while flipping sign), `along_mae_m`, headway/time-gap/TTC to the lead agent | improve; **88.7 % of the oracle gap is longitudinal** | primary |
| **LATERAL** | `cross_mae_m`, `heading_mae_deg`, **`curvature_mae_1pm`**, yaw-rate | ⛔ **GUARD-RAIL: must be NOT separated (no worse)** | a separated LATERAL regression **fails the arm even if ADE improves** |
| **TACTICAL** | manoeuvre-decision quality (selected vs executed, full confusion over the classes), goal/anchor selection: `rank_acc`, `sel_gap`, `frac_sel_2x_worse` | improve `rank_acc`; reduce `frac_sel_2x_worse` from 0.4109 (base) | **the family D-SEL exists to move** |
| **STRATEGIC** | route/goal-setting quality **with its majority-class baseline printed beside it, always** | S5 arm only | reported with `n` and reason if not computable |

Every per-class row carries **recall AND precision AND F1** (D-TAC1B's lesson: never quote a recall from
a rule whose entire mechanism is moving the decision boundary toward the rare class). Where a family
cannot be computed, it is reported **per family with the reason and the n** — never silently dropped.

### 7.2 Pre-committed success / failure for the retrain

* **Success:** TACTICAL `frac_sel_2x_worse` separated below control **AND** LATERAL not separated worse
  **AND** ADE not separated worse.
* **Failure:** LATERAL separated worse (guard-rail), **or** `frac_sel_2x_worse` CI includes 0.
* **RED FLAG — stop and audit, do not publish:** ADE separated better by **> 0.10 m**. That would be
  ~1/3 of the entire "92 %-irreducible" gap from +385 parameters, and the first hypothesis must be a leak
  (the reachability band reading a `v0` that ego-dropout withheld; the route graft on a circular label),
  not a win.

---

## 8. What this is NOT

* **Not a headroom claim.** The oracle gap is ~92 % irreducible by the registry's own standing caveat.
* **Not a fix for D6.** The tactical readout limit is D-TAC1's; D-SEL is its precondition.
* **Not `cond_imagination` as the flagship ships it.** The probe-vocabulary form is refused, from source.
* **Not an input change.** `ego_valid_channel` is implemented and staged but deliberately **excluded** from
  the D-SEL preset, so no selection result is confounded by an input change.
* **Not a VTARGET arm.** The flagship's set-speed conditioning is the one lever that does NOT transfer for
  want of a **label**, not for want of a mechanism — see §9.
* **Not launched.** No training was started; no pod was touched; 0 GPU spent.

---

## 9. 🔴 ESCALATIONS — these are requests, not notes in a README

An orthogonality instrument once sat unmerged for 10 days because the request lived in a README nobody
re-read. These are addressed to named owners.

1. **→ REF-C DATA stream: the VTARGET set-speed INPUT.** The flagship's `cond_vtarget` (23 non-uniform
   bands + a DROPPED slot) is the direct handle on the 88.7 % longitudinal share, and REF-C has **no
   target-speed input at all** (`refc1`'s `speed_cls` is an *output*). The model-side seam is a small,
   well-understood change; the blocker is a **leak-guarded label**. ⛔ It must NOT be minted from
   `fut_poses[:, +2s]` — that is the label the manoeuvre head is judged on, and feeding it as an input is
   the same family as the C6 confound and the REF-A I-JEPA leak. It needs the *free-flow aspiration*
   derivation (read over 10–20 s), which is the DATA stream's object. **I did not build a half-wired seam
   with no label**: a logit no label can ever train is a dead parameter that invites a shortcut.
2. **→ eval/tools stream: `scripts/refc_sel_probe.py`** (§6.4). Reads banked fans + an existing
   checkpoint; 0 GPU-days. It is the gate on everything in §7 and it is not this stream's file.
3. **→ orchestrator: `MODEL_REGISTRY.md` §4.2 cites a path that does not exist** —
   `taniteval/results/refc-small-30k.json`. The artifact is at
   `…/incoming/2026-07-22-refc-small-30k/refc-small-30k.json`. Already flagged independently at
   `published_numbers.jsonl:57` and still unfixed.
4. **→ orchestrator: the "8.4 % of the gap" figure cites a prose research note, not a JSON**
   (`published_numbers.jsonl:55`, `citation_resolves: NOT-CHECKED`) — and it is load-bearing for the
   92 %-irreducible caveat, which is in turn load-bearing for §6.3's thresholds.

---

## 10. Deliverable manifest

| artifact | path | state |
|---|---|---|
| selection-surface module (new) | `stack/tanitad/refs/refc_select.py` | repo, **staged** |
| model + config + decoder | `stack/tanitad/refs/refc.py` | repo, **staged** |
| trainer: flags, ranked-score CE, guards, banner | `stack/scripts/refc_train.py` | repo, **staged** |
| tests (new, 20) | `stack/tests/test_refc_select.py` | repo, **staged** |
| this pre-registration | `Project Steering/PREREG_D-SEL_REFC_SELECTION_SURFACE.md` | repo, **staged** |

**Nothing is committed and nothing is pushed** (AGENT_OPERATING_STANDARD rule 1). Nothing lives only on a
pod or only in a worktree. No checkpoint was produced and no pod was touched.

### 10.1 How "fixed in advance" is made VERIFIABLE here

⛔ **No self-hash is printed in this file, deliberately.** A document cannot contain its own sha256, so a
digest pasted here would be either wrong or computed over a different version — a second unfalsifiable
claim of exactly the shape R11 refuted in D-TAC1 (whose prereg mtime was 6 minutes *after* its own probe
JSON, making "fixed in advance" INHERITED rather than MEASURED).

The falsifiable claim is the **git object id of the staged blob**, which is fixed the moment this file is
staged and is recorded in the stream's report:

```bash
git ls-files -s -- "Project Steering/PREREG_D-SEL_REFC_SELECTION_SURFACE.md"
git hash-object   "Project Steering/PREREG_D-SEL_REFC_SELECTION_SURFACE.md"   # must match
```

Anyone reading a future results file can re-run those two commands. If §6.3's thresholds have moved since
staging, the ids will not match, and the "committed in advance" claim is void — which is the property a
hash is supposed to buy and an mtime never did.

---

## Evidence class

| claim | class |
|---|---|
| every param count, and the +385 delta | **MEASURED** — `param_breakdown` on a `meta`-device build, pinned by `tests/test_refc_select.py::test_dsel_is_not_a_capacity_change` |
| byte- and bit-identity when off | **MEASURED** — diffed against `git show HEAD:stack/tanitad/refs/refc.py`, 6 flag combinations |
| the rank grafts get no gradient without the ranked-score CE | **MEASURED** — first run of `test_rank_grafts_are_gated_not_dead` |
| oracle-in-fan 0.1640 / `frac_sel_2x_worse` 0.454 (**XL**); 0.1914 / 0.4109 (**base**) | **MEASURED** — `scaleab_refc-base-30k_vs_refc-xl-30k.json`, duplicated in `planfan_clips_summary.json` |
| reachability clamp 72.08 % / oracle 100 % / paired Δ **0.0** | **MEASURED** — `t1_clip_fansize.json` (REF-C-XL's fan) |
| `auc_lon_active` 0.7294 vs the ≥0.65 threshold | **MEASURED** — `dtac1_probe_refc-base-30k.json` |
| LAN nav sweep 0.2416 m; oracle route worse on cross-track/curvature | **MEASURED** — `E0_refc-base_navcf_full.json` |
| `IMAGINATION_HAS_CANDIDATE_AXIS = False`, 32 tokens for 256 candidates | **MEASURED** — `t4_imagination_conditioning.json` |
| `route_head` nav-blind; 0.7613 below the 0.7806 majority baseline | **MEASURED** — R-2026-08-03-l; `junction/OL_flagship_vs_refc_objects.json` |
| "the oracle gap is ~92 % irreducible" | **INHERITED** — `MODEL_REGISTRY.md` §4.1, whose own 8.4 % citation resolves to a prose note (§9.4). Load-bearing for §6.3 and flagged as such |
| "S1 differs in kind from the v1.2 re-scorer" | **HYPOTHESIS** — §6 is designed to decide it |
| E-SEL-0 / E-SEL-1 outcomes | **NOT YET MEASURED** — thresholds fixed above, both branches committed |
