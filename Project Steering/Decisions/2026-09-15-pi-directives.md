# PI directives — 2026-09-15 (Europe/Berlin)

Recorded verbatim by the Master Mind so that no agent or later session acts against them. Typos are the PI's own and are kept.

## 1. Corpus augmentation (morning)

> *"Ok, start now the augmentation of the complete training corpus, download what do you need, manage efficiently disk memory and
> push the final daraset to my hf account, the data set should include all augmentations, the already geberated alpamayo outputs,
> our already generated newest nav commands, tactical and strategic labels and the semantic maps..."*

> *"Did the augmention start, i want regular updates about the progress"*

**Status:**
- SAM3 corpus production has been running on Thor since 07:51. All other components are on `Sayood/tanitad-v7-training-corpus` (private).
- Progress updates are scheduled at 08:13, 12:13, 16:13 and 20:13, with silent health checks at 00:13 and 04:13.
- Evidence: `TanitAD Research Lab/Data Engineering/Research/2026-09-13-sam3-only-road-map/RESULT.md` §21.

## 2. refcv6 — clarify before any plan is approved (evening)

> *"before we approve any new plan, I must understand and clarify important things:*
> *\* I want to review and understand how was the diffusion polanner linked to the resnet trunk, how was the perception implemented and linked to the diffusion planner in both papers of the diffusion drive. Then compare it to our plan*
> *\* How can we proove that we can extract envrionment information mainly bounding box objects, the semantic maps and occupancy maps. Is there an option to link directly the planner to the trunk (I think this waht we did in the past)*
> *\* The nav command is a mandatory input for both opreative and tactical layers and must be considered in the selection and planning of tactical behaviors and also operative trajectory*
> *\* The tactical layer is responsible to learn the max speed and all tactical behaviros from our vocabulary using the gt labels in our data set*
> *\* please review and confirm the diffusion implementation in the planner as stated in the paper and let us review the number of hyopthgeses, the denoising process etc,..*
> *\* Let review how si the resnet mdoule is trained and used  at inference in compariso toi the dricve diffsuion papers and other AD works using this resnet architecture*
> *\* Prepare and validate the RL approach as described in the paper*
> *\* Use my computer for prepration and final design of refcv6 then I will provide a pod for heavy work*
> *\* Let make the things clear for refcv6 and then take care of refav1 then v7 falgship"*

### What this binds

| # | binding consequence |
|---|---|
| a | ⛔ `Project Steering/REFCV6_DESIGN_GROUNDED.md` (d4e8bc0) is a **proposal, not approved**. Nothing launches on it. |
| b | ⛔ **Nav command is a mandatory input to BOTH the operative and the tactical layers**, and must shape tactical behaviour selection and the operative trajectory. MEASURED context: refcv5-v2 barely uses nav (`D-REFCV5V2-NAV-USE-1`). |
| c | ⛔ **The tactical layer learns the max speed and every tactical behaviour of the vocabulary from the dataset's GT labels.** ⚠️ This is in tension with two earlier rulings. **2026-09-01:** *"at inference this data will be provided as input by the user like the nav command, thus these are not training labels, just input data"* (`TanitAD Research Lab/Data Engineering/Implementation/incoming/2026-09-01-v8-tacsit-release/V8_MANIFEST.json:121`). **2026-09-10:** *"regarding to max speed as input, stick to the labels we created in the data set with the logic of minimal speed etc..."* (`Project Steering/PREREG_E16_MAX_SPEED_INPUT_REFCV6.md:18-19`). The label is the ego's own realised maximum speed over [t0+2 s, +6 s] (`provenance: ego-future`). The reconciliation is put to the PI; it is not resolved silently. |
| d | Every DiffusionDrive claim is answered from the banked primaries: V1 2411.15139, V2 2512.07745, released code under `…/2026-09-05-diffusiondrive-v2-analysis/raw/ddv2_src/`. |
| e | Preparation and final design run on the **dev box (RTX 4060)**; heavy training waits for the **pod the PI will provide**. |
| f | Order of work: **refcv6 → refav1 → flagship v7**. |
