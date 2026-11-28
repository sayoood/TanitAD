# PREREG_V8F_PROOF_ADD37_N2 : IMAGACT, what the dry gate measured, and one added dose arm, before any arm trains

Note to card `PREREG_V8F_PROOF_ADD37.md` and note N1. Frozen after the dry gate of LA_s0 (selfcheck only, nothing trained) and before chain `actcons_a` has started: no LA, LAm or LA01 checkpoint exists and no response value of any fine-tune has been read. The card, its named arm (LA, weight 1.0), its names and its controls stay as written.

## 1. The dry gate of LA_s0 (MEASURED on the pod, real data, 15:12:50Z; `runs/actcons_z/actcons_auto.log`)

* PASS: declared equals built for every lever string; the live step is finite; 6,000 steps.
* Probe of the term on one real batch of 64 windows: term 1.324 m; 32 valid pairs of sample and start frame in 13 windows; gradient norm on the predictor 403,196 and on its action input 179.7; on the lead head 0.0, on the state projection 0.0, outside the predictor and its action input 0.0 in sum; the all-invalid control 0.0 with no graph. This is part of control a4 of the card for this arm.

## 2. What the gate shows about the weight

* The trainer clips the gradient to a norm of 1.0 over all parameters and logs norms of about 20 before clipping in the arms without this term (mean 20.6 and 23.2 over the train logs of the control fine-tunes LFT_s0 and LFT_s1, 19.7 for LS_s0). At weight 1.0 the term alone brings a gradient of about 400,000 on the predictor, about 20,000 times the rest; at weight 0.1 about 2,000 times. The term is linear in its residual, so this size does not fall as the residual falls.
* Consequence, stated before the data: in LA and LA01 the predictor is trained almost only by this term on every step that has a valid sample, and the clipping scales the gradient of every other parameter down on those steps. Both arms test whether the response can be learned; neither tests the term in balance with the other losses of the predictor.

## 3. Added dose arm (names nothing, as every dose arm of the card)

* LAm_s0, LAm_s1: the argv of LA with `--w-act-cons 0.0001`: the term's gradient is then about 40, about twice the rest (about 20). Same tree, same start checkpoints, 6,000 steps.
* Order of chain `actcons_a`: LA_s0, LA_s1, LAm_s0, LAm_s1, LA01_s0, LA01_s1 (the named arm first).
* Read exactly as LA and LA01 (response, imagination, families against LFT). In section 6 of the card, "the dose that reads best between LA and LA01" is read over LA, LAm and LA01. If LA reads `ACT-CONS-LOSES-IMAGINATION` or a `COST-<family>` and LAm does not, that is reported as written for LA and beside it for LAm; the named outcome of the card is LA's.

## 4. Limits

The gradient sizes are from one batch of the selfcheck and from the logged norms of other arms; the weight 0.0001 is set from them by the rule above and from no outcome.
