# PRE-REGISTRATION — HiCAP: hierarchical contrastive action priors on a frozen VLM (REF-F, VLM-first)

**Status: DRAFT, written 2026-09-29, before any HiCAP code, cache or checkpoint exists. No HiCAP model has been trained.**
**Freeze rule.** The PI approves the document, and its SHA-1 is written to `PREREG_HICAP.sha1` before the first HiCAP checkpoint is written. Any later change is a dated amendment with a reason, never an edit. Thresholds below are *proposals* the PI may amend before the freeze.
**Design:** the paper `Paper/HiCAP/hicap.tex` (build: `latexmk -pdf hicap.tex`). **Extends** `PREREG_REFF_CONTRASTIVE_SELECTOR.md` (H-F1…H-F8), which is the special case of one level, one backbone, no imagination. **Statistics:** CLAUDE.md rules — full-set means; episode-cluster bootstrap B = 2000, **paired** for any two arms; never the deprecated split-mean; four metric families per panel; no learning-curve exponents (matched-step ratios only).

> ⚠️ **Which state this is written against.** The newest programme state lives in the PI's Drive (v7 corpus, vocabulary v7 / label release v8.1, REF-C line refcv3→refcv6), **not** on GitHub `main`. The reference arm is **refcv5-v2** (newest arm with results) / **refcv6 BASE** (pre-registered 2026-09-10, never launched). There is **no `refcv7`** in any document searched (PI: "take refcv6 as nearest"). The code for the vocabulary and label consumer is on the PI's unpushed branch `agent/arch-inf-20260803`; **this pre-registration cannot be executed from git until that lands.**

---

## 0. The claim under test

A frozen vision–language backbone plus a small trainable motion adapter and fusion stack, with a **hierarchical, prior-pruned, cached action tree** built on the programme's frozen v7 vocabulary, can make the driving decision (especially the longitudinal one) at least as well as the programme's best flat planner, at a fraction of the run-time cost, and remain deployable through a ≤ 300 M distilled student.

## 1. Fixed before any data is seen

| item | fixed value |
|---|---|
| train data | corpus `tanitad-v7-training-corpus` (id `a48251e89c7a8603`), 4,572 train clips; **non-parity** (does not re-select `physicalai-train-e438721ae894`); arm-vs-arm deltas only, never levels against parity arms |
| labels | v7.2 for the reference comparison (what refcv5-v2/refcv6 read); v8.1 fields (`tac_SIT`, `nav_30s`, `lane_change_text`, `speed_max_input`) are **evaluation strata or unused**; `speed_max_input` and `nav_30s` are oracle-stamped and not built into any arm |
| eval | T1 open-loop panel: 4,823 windows / 141 episodes; rows `os`, `os_navzero`, controls `ha`, `ha0_ext`, `ha0` |
| vocabulary | frozen v7: `a_str` 7 (5 populated), `a_tac.lat` 8 × `a_tac.lon` 8 (5 × 7 = 35 populated cells), `g_tac` 22 (multi-label, aux), nav 3 (input, stamped oracle); level 3 = two-stage residual code over a CTRA prior, per-cell codebooks (farthest-point + k-means on TRAIN residuals, time-weighted L2), K₃ ∈ {4k, 16k, 64k, 256k} per cell |
| inference inputs | frames (front-wide; further cameras when cached), ego state e_t = (v, a_lon, ψ̇, κ), nav command (stamped oracle; nav-zero is the deployment-honest row), optional audio; **no** situation-classifier output, **no** `tac_SIT` / `disputed` / `alpamayo.*` |
| backbones | R1 general Qwen3.5-4B; R2 driving-aware Qwen-Drive-1.0-4B VLM (paired with R1); R3 Qwen3-VL-2B or Cosmos3-Edge reasoner; audio arms Gemma-4-E2B (licence unverified) / Qwen3-Omni-30B-A3B (teacher). **Contamination:** 13/400 clean-val and 76/2,400 train clips are in NVIDIA's public training sample (MEASURED, RA §Appendix B); the canonical 40 val UUIDs must be recovered before any driving-aware arm is read |
| losses / optimisation | per-level soft-target CE (σ = 0.5 m), cost heads, refinement, consistency KL; three-stage curriculum with 40 % replay; AdamW, wd 0, lr = 2e-3·√(1024/width)·√(batch/1024), OneCycle 10 % warm-up, batch 2,048, τ₀ = log(1/0.07), e^τ ≤ 100; **3 seeds per arm plus a seed-1 replicate; bar = effect ≥ 10 % relative AND ≥ 3× replicate floor** |
| selection rule | argmin Σ w_k ĉ_k − λ log p̂ over pruned set K ≤ 32; w_k fixed a priori by the four-family policy; λ, β, α_ℓ, η, gate thresholds fitted on held-out **train** episodes, never on val |

## 2. Negative controls — run first; a panel is void if one fails in the stated way

| id | control | expected | if violated |
|---|---|---|---|
| NC1 | state permutation across windows | ADE ≥ frequency-prior baseline | leakage / ego-channel dominance: stop and diagnose before reading anything else |
| NC2 | random instead of designed codebook | not better than designed | the codebook construction does nothing |
| NC3 | k-NN in the same frozen feature space | HiCAP separated-better | heads add nothing beyond the backbone |
| NC4 | ego-only and image-only | ego-only ≪ full | copycat risk; report transition-window split |
| NC5 | nav-zero, nav-shuffled | `os_navzero` reported beside `os` | echo of the oracle nav |
| NC6 | flat tactical (no parent restriction); shuffled parent | tree ≥ flat control only if the restriction helps | restriction is decorative or leaks |
| NC7 | imagination-shuffled, action-shuffled | gain disappears | imagination is not what helps |
| NC8 | random camera/token gate at matched cost; config-invariance and gate-shuffle for any gate reading the previous tactical token | learned gate separated-better | cues carry no value / back door |
| NC9 | audio-null (bit-exact), audio-shuffled, bearing flip, dose-response, onset shift, audio-only label probe | effects vanish or flip as designed | synthetic artefact / label echo |

## 3. Hypotheses — thresholds and both outcomes, committed now

“Non-inferior”: upper 95 % bound of the paired ΔADE@2 s < +0.02 m. “Separated”: 95 % interval excludes 0. Verdicts: SUCCESS / FAIL / INCONCLUSIVE / MISSING_DATA (INCONCLUSIVE is not a pass; n < 30 scenes ⇒ INCOMPLETE).

| id | measurement | CONFIRM ⇒ | REFUTE ⇒ |
|---|---|---|---|
| **H-HC0 gate** | k-NN over frozen VLM tokens + motion tokens + v₀ vs hold-speed, non-steady windows, paired, **before any head is trained** | separated-better ⇒ frozen features carry decision signal; proceed | not separated ⇒ change backbone/adapter before any head training |
| **H-HC1 adapter** | paired ADE with ego channel zeroed, with vs without motion adapter | with-adapter separated-better by ≥ 10 % ⇒ needed | not separated ⇒ drop the adapter |
| **H-HC2 driving knowledge** | R2 vs R1, same heads, paired, four families; second surface off PhysicalAI | R2 separated-better on ≥ 1 family, non-inferior on rest, both surfaces ⇒ driving tuning helps | R1 non-inferior ⇒ keep general backbone; PhysicalAI-only gain = contamination-suspect |
| **H-HC3 tree vs flat** | beam-retrieved tree vs flat softmax over identical leaves | non-inferior with ≥ 10× fewer nodes touched ⇒ adopt tree | lower bound > +0.02 m ⇒ use flat + pruning |
| **H-HC4 prior correction** | KL(selected-class freq ‖ GT freq) and steady-window ADE: InfoNCE argmax vs β-corrected vs soft-target softmax | InfoNCE-only larger KL and separated-worse steady ADE ⇒ correction mandatory | no separated difference ⇒ drop it |
| **H-HC5 vocabulary scale** | K₃ sweep: oracle-in-set, tree-selected, flat-selected ADE | tree-selected non-increasing 4k→64k within margin while flat-selected worsens (paired, separated) ⇒ scale usable only with the tree | tree-selected worsens like flat ⇒ selection is the bottleneck; keep K₃ = 4k |
| **H-HC6 pruning** | R1 retention of logged futures; conformal coverage on episode-disjoint test; share of level-3 nodes removed | retention lower bound ≥ 0.99; mean coverage over ≥ 200 episode-disjoint calibration draws ≥ 1 − Σα_ℓ − 0.01 with ≤ 20 % of draws below nominal (measured on the val-40 proxy with 12 calibration episodes: mean coverage at nominal but 33–43 % of draws below it); ≥ 50 % pruned ⇒ ship as hard masks | retention < 0.99 ⇒ widen or demote to costs |
| **H-HC7 re-score** | candidate-conditioned late-interaction re-score vs dot product only | separated-better on ADE or longitudinal, non-inferior elsewhere ⇒ keep | not separated ⇒ drop the block |
| **H-HC8 imagination** | gated imagination vs same model without; headway/TTC on closing windows; reliability by horizon | separated-better longitudinally, NC7 removes ≥ ½ the gain, gate off when heads disagree ⇒ keep | no gain or NC7 doesn't degrade ⇒ remove |
| **H-HC9 student** | HiCAP-S (≤ 300 M) vs HiCAP-R: paired ADE, level-2 agreement, share meeting the margin condition | non-inferior, agreement ≥ 90 %, condition on ≥ 80 % of windows ⇒ sub-300 M met with guarantee | otherwise end-to-end student, guarantee lost |
| **H-HC10 elastic sensing** | **first E-C3**: oracle-camera value by manoeuvre on a ~30-chunk side-study (n ≥ 30 per stratum), paired, four families; then gate **G-V** (vision-only: FW embedding + own keep-alive) vs fixed-all at the peripheral budget and vs a random gate at matched cost; G-V per-class recall and wrong-run length | oracle camera beats front-only beyond the harm margin in ≥ 1 stratum AND G-V's paired CI excludes the harm margin in every family, mean cost ≤ 0.35× fixed-all, p95 ≤ 100 ms, separated-better than random, wrong runs ≲ 3 ticks ⇒ adopt G-V, tier only those strata | no stratum where an oracle camera helps ⇒ **retire elasticity, ship front-only** (no caches bought); weak G-V ⇒ PI ruling on ego cues (G-E) or hard speed/route tier rules. **G-T (previous tactical token) inadmissible** unless it passes config-invariance and gate-shuffle. **INCOMPLETE for cameras until caches exist** (321 GB front-tele … 2.55 TB all six) |
| **H-HC11 latency / precision** | ≥ 1,000 ticks on Thor: fast-path p95, VLM refresh p95 (1 camera), NVFP4 vs BF16 feature drift | fast-path p95 ≤ 25 ms, refresh p95 ≤ 50 ms, drift within margin ⇒ fits 100 ms tick | otherwise profile; shrink or cache more |
| **H-HC12 audio** | T1 audio-null == original embedding bit-for-bit; T2 event heads (class, bearing, distance) on injected synthetic events; T3 controls (null, shuffle, bearing flip, dose-response, onset shift, no-vehicle/no-siren, **audio-only label probe first**); T4 no regression on four families | T1 exact, T2 above baseline, flips and dose-response behave, probe at chance, T4 non-inferior ⇒ path wired and controlled (**not** a driving-skill claim; behaviour only in a closed-loop acoustic sim) | probe above chance ⇒ generator leaks the label, regenerate from world state; controls fail ⇒ shortcut, audio path unusable here |
| **H-HC13 data efficiency** | four-family panel at 3/10/30/100 % of train hours, nested episode subsets, vs reference at matched fractions (matched-step ratio only) | relative ADE loss at 10 % and 3 % separated-smaller ⇒ first evidence for the data goal | not separated ⇒ no edge |
| **H-HC14 closed loop** | NuRec/AlpaSim paired vs reference: pass rate, collisions, jerk, n ≥ 30 scenes | pass rate non-inferior, jerk not worse ⇒ closed-loop reference | separated-worse ⇒ strengthen stage 3; n < 30 ⇒ INCOMPLETE |

**Four families, per family, never pooled.** LONGITUDINAL (speed MAE, target-speed acc@0.5, along-track MAE, headway/time-gap/min TTC with `n_closing` and censored share) · LATERAL (heading, yaw rate, cross-track, **masked** curvature vs straight-line floor) · TACTICAL (lateral/longitudinal accuracy and κ of the factored decision, cell-selection accuracy, goal FDE/bearing) · STRATEGIC (route acc, κ, n; **UNAVAILABLE with reason where n = 0** — five usable strategic actions today, no map/route supplier).

## 4. Stopping rules
- NC1 fails ⇒ stop and diagnose before reading any other result.
- H-HC0 refuted ⇒ no head is trained on that backbone.
- No arm is tuned on val; temperature, β, λ, α_ℓ, η and gate thresholds are fitted on held-out TRAIN episodes.
- Any driving-aware arm (R2) is uninterpretable until the 40 val UUIDs are recovered and overlap is measured.

## 5. What this is not
- **Not a strategic-level result** (5 usable strategic actions; no route supplier except the ego's own future).
- **Not a safety claim** (agent-state pruning is an oracle-perception simulation until a learned range head exists).
- **Not a claim about the released CLM-8B model.**
- **Not a claim that audio helps on real roads** (no corpus carries audio; H-HC12 tests wiring and controls only).

## 6. Decisions the PI owns before the freeze
1. Does "no VLM, we stick to the Alpamayo labels as teacher signals" (recorded in a label context, `PI_DECISION_QUEUE` item 1, RELAYED/PENDING) also exclude a VLM as the *backbone*?
2. Is audio admissible at inference, given "inference only vision" — and are ego cues (G-E) admissible for the camera gate (a gate is a situation classifier in disguise; the vision-only gate was much weaker in the toy)?
3. Does "sub-300 M" apply to the deployed student only (proposed) or to the reference model too?
4. Recover the 40 canonical val clip UUIDs (contamination test).
5. Which branch to build on (main is stale; newest code is unpushed).
6. Approve/amend the thresholds in §3.

## 7. Deliverable manifest
| artifact | where |
|---|---|
| this pre-registration (DRAFT) | `repo:Project Steering/PREREG_HICAP.md` |
| paper (sources) | `repo:Paper/HiCAP/` (`hicap.tex`, `sec_*.tex`, `fig_arch.tex`, `app_vocab.tex`, `refs.bib`) |
| research streams | `repo:TanitAD Research Hub/Architecture & Inference/Research/2026-09-29-hicap/` (`P1`, `P2`, `P3`, `RA`, …) |
| measurements | same folder: `hicap_prior_mask_*`, `hicap_cost_model_*`; `../reff_pmi_toy*`, `../reff_v0_coverage*` |
