# Three parallel streams, adversarially verified (2026-08-02)

**PI directive:** *"What's our plan with refc optimization, IDM performance and validation, scenario
classification. All these things must run in parallel."*

**Method:** 6 agents — one surveyor and one **adversarial verifier** per stream, the verifier
instructed to *refute rather than agree*. 995 k subagent tokens, 299 tool uses, 30 min wall-clock.

⭐⭐ **Every one of the three verifiers refuted its own side's survey.** That is the result worth
having: had these three surveys been reported to the PI as-is, they would have carried a fabricated
defect, a false blocker, two non-executable plans, and a mis-stated n.

---

## STREAM 1 — REF-C optimisation

### 🔴 THE FINDING THAT MATTERS: our published REF-C numbers are CONFOUNDED

`taniteval/taniteval/refc_eval.py:172-194` says, verbatim:

> *"`route_input_exercised` False means the decoder saw ONE constant command for every window, so
> it was compared on its marginal — the 07-21 C6 confound. **Every REF-C number published before
> 2026-07-26 (base 0.4728, XL 0.4714) was collected that way.**"*

⇒ **REF-C base 0.4728 and XL 0.4714 — the numbers the programme quotes — were measured with the
route input effectively disabled.** This is the same C6 class that the operating standard already
warns about (*"the 'strategic choice is a ~2 % lever' refusal was CONFOUNDED, because REF-C
evaluates with nav_cmd=None"*). ⇒ **any REF-C ranking built on them is not admissible** until
re-collected with the route input exercised.

### Other verified corrections

| # | finding |
|---|---|
| ⛔ | **A fabricated "artifact inconsistency" was struck.** The survey claimed stage-times contradict the registry; the artifact already publishes the reconciliation (`stage_sum_note`: >100 % ⇒ stages overlap CPU-launch with GPU-execute). Registry and artifact **agree**. |
| ⛔ | **"LATERAL complete" is FALSE.** `four_families_vs_floors.json` **REFUSES the intervals** for `heading_mae_deg`, `yaw_rate_mae_degps`, `curvature_mae_1pm` (per-window form disagrees with `four_families` beyond tolerance). LATERAL is **2 of 5 with intervals**. |
| ⛔ | **Registry absence under-scoped ~6×:** e1b, e1c, e1d, e1e-A, e1e-B **and** e1f are ALL missing from `MODEL_REGISTRY.md`. `refc-base-e1f-junction` is already **public on HF** and on Thor — a higher-priority registry gap than the one reported. |
| 🟠 | **E1b primary rests on n = 6**, not 44 (44 is the guardrail's n). |
| 🟠 | **speed_bias "SEPARATED win" read the sign backwards** — under the file's own `floor - model` orientation REF-C is −0.0754, i.e. *loses*; what is true is that \|bias\| is smaller. And the flagship comparison was **unpaired**. |
| 🟠 | **The precision conclusion is not immune:** the three blocks were captured at GPU util **43 / 87 / 28 %**, and the "LOSS" ratios carry **no interval** (0.42 ms gap inside ~1 std). Direction supportable; point ratios not decision-grade. |
| 🟠 | **Four-family absence found at one location.** `driving_refc-base-30k.json` *does* carry LONGITUDINAL and LATERAL surrogates at n=881/40. What is genuinely absent for every REF-C arm is an `all_families()` panel and **any TACTICAL/STRATEGIC family**. |

## STREAM 2 — IDM performance and validation

**32 of 47 claims confirmed exact; the headline table is sound. The failures cluster in the plan.**

| # | finding |
|---|---|
| ⛔ | **Two plan steps are NOT executable.** The banked JSONs (`idm5_ensemble.json`, `compare_v3.json`, `arms_v3.json`) contain **zero per-window prediction arrays** — aggregates only. The claim "predictions are already persisted" is false; both steps need the head re-run over latents at `/root/idm2/lat`, which was **pod-only and is now gone**. So "0 GPU-h / 3 h" is wrong. |
| ⛔ | **"Channel ordering matches the physics without being fitted" is REFUTED** on like-for-like slices. |
| ⚠️ | The only local val is `C:\Users\Admin\tanitad-data\eval\comma2k19-val-…` (comma, no PhysicalAI). The one run that used it **re-encoded** latents, and its own `controls.encode_fidelity` warns comma speed R² reads **0.5545 vs 0.7590** on the pod substrate — *"a plausibility band, not an equality check"*. |

## STREAM 3 — scenario classification

| # | finding |
|---|---|
| ⛔ | **The survey's loudest claim — "no per-window camera-head scores are banked anywhere" — is FALSE**, and it committed the exact error it was policing. `HUB\2026-07-26-situation-classifier\artifacts\heldout_frames.npz` (39 MB, in repo since 2026-07-27) holds **per-frame held-out scores for all 10 arms** over 308,973 frames, plus checkpoints and `pca.npz`. |
| ⭐ | ⇒ **L0 is cheaper than planned.** The gold-set re-score does not need a re-run — per-frame scores already exist; only scene-truth labels for a sampled subset are missing. |
| ⚠️ | Path correction: the real root is `TanitAD Research Hub\Architecture & Inference\Implementation\incoming\`; a second, empty `Implementation\incoming` exists and misleads. |

---

## What this changes

1. 🔴 **REF-C's published ADEs are not admissible for ranking** until re-collected with the route
   input exercised (C6). This is the highest-priority correction in the batch.
2. ⭐ **Scenario-classification L0 got cheaper** — per-frame scores are already banked; the work is
   labelling, not inference.
3. ⛔ **The IDM plan needs rewriting** — its two cheapest steps depended on data that never existed
   in the repo and now cannot be regenerated from the terminated pods.
4. **Registry debt is 6 REF-C arms, not 1**, one of which is already published publicly.

## Evidence class

| claim | class |
|---|---|
| every refutation above | **MEASURED** by the verifying agent against named artifacts/line numbers, 2026-08-02 |
| the C6 route-confound quote | **MEASURED** — `refc_eval.py:172-194`, quoted verbatim |
| REF-C 0.4728 / 0.4714 being confounded | **MEASURED** — the file states it about those exact numbers |
| "L0 is cheaper than planned" | **MEASURED** (the npz exists) + **HYPOTHESIS** (that it suffices for the gold-set re-score) |
