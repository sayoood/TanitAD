# thor:~/trt_deploy — the predictor engines that may actually be deployed

Built 2026-08-03 by `stack/scripts/build_predictor_trt.py` (checked in — the
2026-08-02 builder was a transient heredoc and is NOT on disk, which is why its
export could never be inspected).

| plan | arm | weights | batch profile | intent input | verified |
|---|---|---|---|---|---|
| ⭐ **`predictor_v1_intent_dyn1-9_fp16.plan`** — **THE DEPLOYMENT ENGINE** | flagship-v1-speedjerk | **REAL, step 29999, STRICT load** | dynamic **1..9** (opt 9) | ✅ **yes** (256) | loaded + executed at b1/b9, rel-err vs eager **3.66e-4 / 3.67e-4**; `intent_is_live_rel_change` **0.0522** |
| `predictor_v1_dyn1-9_fp16.plan` ⛔ control only | flagship-v1-speedjerk | REAL, step 29999 | dynamic 1..9 | ⛔ **no** | kept ONLY as the control arm for retraction R-2026-08-03-e |
| ⭐ **`predictor_v5f_intent_dyn1-9_fp16.plan`** | v5f (176x624 deployed geometry) | REAL, step 1000 | dynamic **1..9** | ✅ **yes** (256) | 40.0 s, 174.7 MB; rel-err **8.49e-4 / 6.06e-4**; `intent_is_live` **0.0470** |
| `predictor_v5f_dyn1-9_fp16.plan` ⛔ superseded | v5f | REAL, step 1000 | dynamic 1..9 | ⛔ no | intent-less; kept for provenance |

⚠️ **v5f needs `--v2-subframe 176x624`.** Without it the builder's **STRICT** load REFUSES the
checkpoint (`encoder.pos` 429 vs 256) instead of quietly resizing — that refusal is the geometry
guard working, not a bug.

⛔ **An engine without the `intent` input computes the UNCONDITIONED prediction.** MEASURED: it costs
**3.81 m** mean tactical regret at K=20 and flips 60 % of selections. The runtime now raises rather
than dropping the token silently.

md5/provenance: `thor_d1_batch9_engine.json` and `predictor_v1_intent_dyn1-9_fp16.json`.

## ⛔ SUPERSEDED — do not deploy

`~/trt/predictor_fp16.plan` (2026-08-02, 173.9 MB) is **batch-1 static** AND was
built from a **randomly-initialised model** — `thor_trt.py`, the script that made
it, contains no `torch.load` and no `load_state_dict` (probed 2026-08-03).
It is kept for provenance, not for use.

`~/trt_c3/pred_dyn_fp16.plan` is real-weight but **dynamic 1..8** — it cannot
serve the 9-candidate fan (`TacticalConfig.n_maneuvers = 9`).

## Rebuild recipe (runbook §11.4 — a derived artifact needs a recipe, not a copy)

```bash
export PATH=$HOME/venvs/tanitad-edge/bin:/usr/local/cuda/bin:$PATH
export PYTHONPATH=$HOME/TanitAD/stack:/usr/lib/python3.12/dist-packages
python $HOME/TanitAD/stack/scripts/build_predictor_trt.py \
  --ckpt $HOME/models/flagship-v1-speedjerk/ckpt.pt \
  --out  $HOME/trt_deploy/predictor_v1_intent_dyn1-9_fp16 \
  --max-batch 9 --fp16 --intent-dim 256      # ⛔ --intent-dim is NOT optional for the flagship
```
⛔ **And the caller must batch**: `propose_and_score(..., batch_fan=True)`. A batch-9 engine driven
by the serialised loop measures *worse* than a batch-1 engine (272.8 vs 265.7 ms).
~38 s per engine, 174 MB. The engines are NOT in git (too large) and are fully
rebuildable from the HF weights + this script.
