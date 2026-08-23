# flagship v1.6 — unfreezing the trunk under the trained v1.5 head (LP-FT)

**Date:** 2026-07-20 · **Author:** tools/dev-env agent · **Host:** `tanitad-pod2` (A40)
**Status:** 🟢 **TRAINING LAUNCHED** (PID 34401, run dir created 2026-07-20 20:01 pod-local) ·
**final results 🟥 PENDING — not yet measured. Every result row below marked PENDING is UNVERIFIED.**

> **Headline / escalation.** The capacity review is **decided and complete**: the v1.5 decoder is
> **not** the bottleneck — it is already **131 % of REF-C-XL's decoder** (29.76 M vs 22.70 M) and
> matches XL on *every* other hyperparameter, yet its oracle-in-fan is **1.87× worse** (0.3073 vs
> 0.1640). Decoder capacity and anchor count are therefore **kept exactly as `ab`**, and the whole
> budget goes into the trunk. v1.6 = the `ab` arm with the trunk unfrozen (LP-FT).
> **Integration need:** the run needs ~6–7 h of A40 wall-clock to reach step 6000; the decision-grade
> read (`eval_flagship_v16.py`, vendored bench, n=881) must be run on the FINAL checkpoint after it
> lands. **Do not evaluate v1.6 with `eval_flagship_v15.py`** — that reads the cached `states_val.pt`,
> which are stale the moment the encoder moves, and would silently score the OLD trunk.

---

## 1. Capacity review — Sayed's explicit ask, answered from source

Every number below was counted by instantiating the real modules
(`tanitad/models/flagship_v15.py`, `tanitad/refs/refc.py`), not read from prose.

### 1.1 The comparison table

| Model / component | Params | Decoder geometry | oracle-in-fan @2s |
|---|---:|---|---:|
| **v1.5 `ab` head — TOTAL** | **30,975,882** | — | **0.3073** |
| ├─ decoder | 29,761,033 | **d512 · 8 layers · 8 heads · ff_mult 4 · 2 denoise · 256 anchors** | |
| │  ├─ cross-attn layers (8×3,677,696) | 29,421,568 | **98.9 % of the decoder** | |
| │  └─ feat/traj/cond/time/conf/offset | 339,465 | | |
| ├─ imag_proj (conditioning **b**) | 1,066,496 | 2048→512 + pos + src-type | |
| ├─ state_proj (conditioning **a**) | 131,584 | 128→512 + pos | |
| └─ measurement | 16,768 | | |
| **REF-C-XL — decoder only** | **22,702,345** | d512 · **6 layers** · 8 heads · ff_mult 4 · 2 denoise · 256 anchors | **0.1640** |
| └─ cross-attn layers (6×3,677,696) | 22,066,176 | | |
| REF-C-base — decoder only | 8,634,505 | d384 · 4 layers · 8 heads · 128 anchors | — |
| REF-C-small — decoder only | 2,950,729 | d256 · 3 layers · 4 heads · 64 anchors | — |

*(The three figures quoted in the brief are confirmed exactly: v1.5 head ~31 M → 30,975,882;
REF-C-XL decoder 22.7 M → 22,702,345; REF-C-base decoder 8.6 M → 8,634,505.)*

### 1.2 The decisive observation

**v1.5's per-layer cross-attention block is 3,677,696 params — byte-identically the same block as
REF-C-XL's** (same `CrossAttnLayer`, same d=512 / 8 heads / ff_mult 4). v1.5 simply instantiates
**eight** of them where XL has **six**. So v1.5's decoder is *literally XL's decoder plus two extra
layers* (+7,355,392).

> **v1.5 has 131 % of XL's decoder capacity and produces a 1.87× worse proposal set.**
> Proposal quality does not track decoder size. It tracks whether the trunk was trained end-to-end.

### 1.3 "Were any XL decoder hyperparameters left on v1.5 defaults?" — **No.**

Field-by-field, `V15Config.decoder` vs `refc_xl_config().decoder`:

| Field | v1.5 | REF-C-XL | `DecoderConfig` default (= base) | Verdict |
|---|---|---|---|---|
| `d` | 512 | 512 | 384 | ✅ XL-matched (default overridden) |
| `n_heads` | 8 | 8 | 8 | ✅ |
| `layers` | **8** | 6 | 4 | ⬆ **exceeds XL** |
| `ff_mult` | 4 | 4 | 4 | ✅ |
| `aux_hidden` | 512 | 512 | 384 | ✅ XL-matched |
| `diffusion_steps` | 2 | 2 | 2 | ✅ |
| `noise_std` | 0.1 | 0.1 | 0.1 | ✅ |
| `n_anchors` | 256 | 256 | 128 | ✅ XL-matched |

Nothing was left on a base-scale default. v1.5 explicitly overrides every one of the dataclass
defaults (`d`, `layers`, `aux_hidden`, `n_anchors`) to XL's values, and goes *past* XL on depth.

### 1.4 Decision — keep the decoder as-is, spend everything on the trunk

I do **not** run a wider/deeper decoder or a larger anchor set as a variant. Four reasons:

1. **Capacity is not the binding constraint.** XL reaches 0.1640 with **76 %** of v1.5's decoder
   params. Adding capacity to the side of the system that is already larger than the thing beating
   it is not an evidence-based move.
2. **Anchors already match XL (256)** and are FPS'd over the *real* corpus waypoint pool
   (`v15_prep.py anchors`, the same procedure as `build_refc_anchors.py`), not the synthetic
   default. Raising anchor count only helps if *coverage* is the limit — but the fan already
   contains a 0.24–0.31 m plan; the deficit is in how good the refined proposals are, which is a
   feature-quality question, not a vocabulary-size question.
3. **The warm start forces the geometry.** v1.6 loads the `ab` head `state_dict` **strict**;
   changing `d`, `layers` or `n_anchors` would discard the 8 k-step linear-probe phase and break the
   LP-FT contract (§2). A decoder-capacity variant would require re-running the LP phase first.
4. **The remaining structural difference is a trunk question anyway.** v1.5's KV is a heterogeneous
   token set (8×16 = 128 state tokens + 8 probes × 4 reads = 32 imagination tokens = 160) versus
   XL's 64 conv-map cells. That is about *what the decoder attends to* — i.e. trunk features again.

**Noted counter-consideration (not acted on):** the two extra layers beyond XL could be mild
over-capacity for a *frozen* feature set at only 8 k steps. Testing an XL-exact 6-layer decoder is a
legitimate but **low-priority** variant, and it cannot reuse the `ab` warm start. It is recorded here
rather than run.

---

## 2. What v1.6 is — LP-FT, in the right order

v1.5 already performed the **linear-probe** phase (8 k head-only steps on the frozen trunk). v1.6 is
the **fine-tune** phase (Kumar et al., ICLR'22 — naive FT distorts good pretrained features; LP-then-FT
preserves them).

| Element | Setting | Why |
|---|---|---|
| Warm start | `ab` **`ckpt.pt` step 7999** (not `ckpt_best.pt`) | see §2.1 |
| Unfrozen | `encoder.blocks[8:12]` (last **4 of 12**) + `encoder.norm` + `readout` + **operative predictor** | brief; predictor is the proven mechanism (`a→ab` = −0.1355 m) |
| Frozen | `encoder.patch`, `pos`, `blocks[0:8]`, all other brains | keep early features |
| Trainable | trunk **119,812,736** + head **30,975,882** | 28.45 M encoder-side + 91.36 M predictor |
| Discriminative LR | head **1e-4**, trunk **1e-5** (1/10) | brief |
| Gradual unfreeze | trunk LR **0 for 500 steps**, then 200-step ramp, then cosine | head re-settles first |
| Batch / steps | **64 / 6000** | batch matches the `ab` LP phase exactly (like-for-like) |
| Labels | **v2.1, `use_net_dyaw=False`**, `ROUTE_UNKNOWN` masked never clamped | Sayed's ruling; `cond=ab` so goal tokens are off anyway |
| Loss | `v15_losses` verbatim | no separate trunk loss — the trunk adapts to the SAME planning objective |

### 2.1 Correction: warm start is `ckpt.pt`, not `ckpt_best.pt`

`ab`'s `ckpt_best.pt` is **step 7500** (in-training val ADE 0.5497). The **final** `ckpt.pt` at step
**7999** is better (0.5404) *and* is the checkpoint that carries the canonical control numbers
(0.5437 heldout, oracle 0.3073, frac_sel_2x 0.3178). `best` was only ever compared at 500-step eval
points, so the post-loop final eval never updated it. Since the brief requires **final checkpoints**
for a like-for-like control, v1.6 warm-starts from `ckpt.pt`.

### 2.2 The encoder-unfreeze forces a new data path (the real engineering cost)

v1.5 trained on **cached states** (`states_train.pt`, built once by `v15_prep.py states`) and
literally `del trunk.encoder`. The moment the encoder trains, those states are stale. v1.6 therefore
**re-encodes raw frames from the epcache every step**. `encode_window_ft()` runs
patch + pos + the 8 frozen blocks under `no_grad` (no graph, no activation retention) and only the
trainable tail with grad + checkpointing — so the frozen prefix costs compute but no memory.

Measured on pod2 (this is why batch 64 was affordable): **batch 16 → 1.05 s/step, 6.38 GB peak of
46 GB.** The feared "days not hours" cost did not materialise; batch 64 fits with large headroom.

### 2.3 Predictor gradient path

`flagship_v15.imagine_probes` is `@torch.no_grad`. With the predictor unfrozen the imagination
rollout must carry gradient, so v1.6 adds `imagine_probes_grad()` — the identical roll, with each
1-step predictor call `checkpoint()`ed so the 20-step BPTT over 8 probes costs ~1 step of activation
memory instead of 20.

---

## 3. Verification performed before launch

| Check | Result |
|---|---|
| pod2 idle, no competing GPU procs | ✅ 0 MiB used, 0 % util at start |
| Raw frames present for the encoder path | ✅ `_epcache/physicalai-train-e438721ae894` (2376 eps) + val (600) |
| Pod-vs-repo drift (the standing trap) | ✅ `v15_prep.py`, `fourbrain`, `encoder`, `predictor`, `readout`, `metric_dynamics`, `refc.py` **md5-identical**; `flagship_v15.py` differs **in docstring text only** (restated oracle figures) — code identical, so the `ab` ckpt loads strict |
| Corpus parity | ✅ trainer reports **406,099 train windows** — exactly the canonical `2376 eps / 406099 windows` |
| Unfreeze mask correct | ✅ `['encoder.blocks[8:12]', 'encoder.norm', 'readout', 'predictor']`, 119,812,736 trainable |
| Gradual unfreeze active | ✅ `lr_trunk: 0.0` through step 500; `canary_vs_base: 0.0` while frozen |
| End-to-end smoke (fwd/bwd/eval/canary/save) | ✅ passed |
| Real write capacity (dd, not df) | ✅ 390 MB/s, 253 T free — no quota trap |
| `eval_flagship_v16.py` runs (CONTROL mode, 4 eps) | ✅ loads head 7999 + frozen trunk, vendored bench returns the full jackknife+gate block; ADE@2s 0.5114 heldout / 0.5157 full-set on 88 windows — sane for `ab`, so the decision-grade eval will not fail at the end of a 5 h run |

---

### 3.1 Gradient audit — the unfreeze is provably LIVE ✅

A freeze mask that *looks* right but routes no gradient would make this whole experiment a no-op, so
the mask was audited directly (one fwd/bwd, per-module grad norms):

| Module | Intended | params w/ grad | grad norm |
|---|---|---:|---:|
| `encoder.patch`, `blocks[0]`, `[4]`, `[7]` | FROZEN | **0** | 0.0 |
| `encoder.blocks[8]` / `[9]` / `[11]` | trainable | 12 / 12 / 12 | 2.22 / 2.61 / 4.82 |
| `encoder.norm` | trainable | 2 | 27.2 |
| `readout` | trainable | 2 | 71.9 |
| **`predictor`** | trainable | **151** | **188.6** |
| `tactical_pred` | FROZEN | **0** | 0.0 |
| `head.decoder` | trainable | 123 | 172.4 |

**The predictor unfreeze is live** — `imagine_probes_grad` really does carry gradient through the
20-step checkpointed rollout, so "the proven mechanism" is not silently inert. The 8 predictor
tensors *without* grad are exactly `out_proj` (reserved, never called), `intent_proj` (intent=None
in v1.5/v1.6) and the unused horizon-2 / horizon-4 heads — the 1-step imagination roll never touches
them. Correct, not a defect. Grad magnitude rising toward the output (2.2 → 4.8 → 27 → 72) is the
expected profile for backprop into a frozen-prefix network.

### 3.2 Measured launch economics (the "days not hours" fear, retired)

| Quantity | Measured |
|---|---|
| Step rate @ batch 64 | **2.82 s/step** (50 steps in 141.2 s) |
| GPU memory @ batch 64 | **15.0 GB of 46 GB** (probe @ batch 16: 6.38 GB peak) |
| GPU utilisation | 100 % |
| ETA to step 6000 | **~4.7 h** compute + ~0.5 h eval overhead |
| Train windows | 406,099 (canonical) · val windows scored per eval: 881 |

Re-encoding frames with a partially-unfrozen encoder is **~4.6× the head-only cost**
(v1.5 `ab` ran 0.617 s/step at the same batch 64), not the order-of-magnitude penalty assumed.
This is what made batch-64 parity with the LP phase affordable.

## 4. Canary — world-model collapse guard

### 4.1 The canary harness is VALIDATED against the published trunk ✅

At step 0 (trunk still exactly the frozen v1) the inline canary over the full **881** windows reads:

> **canary_ade@2s = 0.4217** vs the published frozen-trunk **full-set 0.4271**
> (`MODEL_REGISTRY §1.2`; the 0.4522 in the brief is the *jackknife heldout* statistic of the
> same run — a different statistic of the same model, per the registry's number-hygiene note).

A **0.005 m** agreement on an independently-implemented path is a strong verification that the whole
v1.6 data chain — epcache frames → partially-unfrozen encoder → readout → operative predictor →
`grounding.step['op']` → SE(2) — reproduces the known-good trunk. It also means the canary is a
trustworthy *absolute* detector here, not merely a relative one.

The frozen trunk's own **operative rollout** (predictor under TRUE actions → `grounding.step['op']`
→ SE(2) accumulate) is the collapse detector. v1.6 computes it inline at every eval and, crucially,
**establishes its own step-0 baseline on the same harness** rather than assuming the published
0.4522 — a relative detector is what matters, and mixing harnesses would produce a fake alarm.
`canary_vs_base` is logged every eval. **A rising canary means back off `--lr-trunk`.**
The external cross-check is `eval_grounded_rollout_4b_speed.py` run on the saved trunk checkpoint
(v1.6 saves `{"model", "grounding", "head"}`, exactly the format that script expects).

---

## 5. Results

### 5.1 Baselines (the control — measured, not pending)

| Arm | oracle-in-fan @2s | frac_sel_2x_worse | ADE@2s (heldout) | source |
|---|---:|---:|---:|---|
| **`ab` (frozen trunk, control)** | **0.3073** | **0.3178** | **0.5437** | `flagship-v15-ab/metrics.json` + `flagship-v15-ab-ckpt.json` |
| REF-C-XL (canonical) | 0.1640 | — | 0.458 | brief / MODEL_REGISTRY |
| flagship v1 | — | — | 0.4522 | MODEL_REGISTRY §1.2 |

Gates: **G1** beat REF-C 0.458 · **G2** beat v1 0.4522 · **G3** miss@2m ≤ 0.10.
**Primary read = oracle-in-fan vs 0.3073.** If ADE moves but oracle does not, the experiment bought
ranking, not proposals, and did **not** do what it was designed to do.

### 5.2 v1.6 run — 🟥 PENDING

Run dir `tanitad-pod2:/workspace/experiments/flagship-v16-ab-ft/`
(`train_log.jsonl`, `config.json`, `ckpt.pt`, `ckpt_best.pt`, `metrics.json`), stdout `/tmp/v16_ab_ft.log`.

In-training eval = 881 val windows, stride 8, re-encoded through the live trunk (same window set
as the canonical harness, but a plain mean — not the 8-split jackknife, which only
`eval_flagship_v16.py` produces). Read the *trend*, and take the decision from §5.4.

| step | trunk LR | oracle@2s | frac_2x | ADE@2s | canary | canary−base |
|---|---|---:|---:|---:|---:|---:|
| **`ab` control (frozen, step 7999)** | — | **0.3073** | **0.3178** | 0.5404¹ | 0.4217² | 0 |
| 0 (v1.6 init = ab + live re-encode) | 0 | — | — | — | **0.4217** | 0 |
| 500 (head re-settled, trunk still frozen) | 0 → ramp | _pending_ | | | | |
| 1000 (first trunk movement) | 1e-5 | _pending_ | | | | |
| … every 500 to 6000 | | _pending_ | | | | |

¹ `ab` in-training eval (`metrics.json`, same estimator as this column); its canonical jackknife
heldout is **0.5437**. ² measured this session at v1.6 step 0, i.e. the frozen trunk.

**Steps 0–500 are a control-within-the-run:** the trunk LR is pinned at 0, so any movement there is
the head re-settling at the new LR, *not* the unfreeze. The unfreeze effect is everything after 500.

### 5.3 Exact launch command (reproducibility)

Launched on `tanitad-pod2`, PID **34401**, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
stdout `/tmp/v16_ab_ft.log` (deliberately **not** `/workspace` — pod2 swallows logs on death):

```bash
cd /workspace/TanitAD/stack/scripts && PYTHONPATH=/workspace/TanitAD/stack \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup setsid python3 -u train_flagship_v16.py \
 --poses-train /workspace/v15/poses_train.pt --poses-val /workspace/v15/poses_val.pt \
 --labels-train /workspace/v15/labels_train.pt --labels-val /workspace/v15/labels_val.pt \
 --train-cache /workspace/data/physicalai_phase0/_epcache/physicalai-train-e438721ae894 \
 --val-cache /workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11 \
 --trunk /workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt \
 --warm-head /workspace/experiments/flagship-v15-ab/ckpt.pt \
 --anchors /workspace/v15/anchors256.pt --probes /workspace/v15/probes8.pt \
 --cond ab --label-set v21 --unfreeze-enc-blocks 4 --unfreeze-predictor \
 --lr-head 1e-4 --lr-trunk 1e-5 --head-warmup 200 --trunk-warmup 500 --trunk-ramp 200 \
 --steps 6000 --batch 64 --workers 6 --eval-episodes 40 \
 --log-every 50 --eval-every 500 --save-every 500 \
 --out /workspace/experiments/flagship-v16-ab-ft > /tmp/v16_ab_ft.log 2>&1 < /dev/null &
```

### 5.4 Decision-grade eval — run on the FINAL checkpoint

```bash
# v1.6 (re-encodes val frames through the UNFROZEN trunk)
PYTHONPATH=/workspace/TanitAD/stack python3 eval_flagship_v16.py \
  --ckpt /workspace/experiments/flagship-v16-ab-ft/ckpt.pt \
  --poses-val /workspace/v15/poses_val.pt --labels-val /workspace/v15/labels_val.pt \
  --val-cache /workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11 \
  --anchors /workspace/v15/anchors256.pt --probes /workspace/v15/probes8.pt \
  --vendor /workspace/v15/evalsrc --key flagship-v16-ab-ft \
  --out /workspace/v15/results/flagship-v16-ab-ft.json

# the `ab` CONTROL through the IDENTICAL harness (--trunk => frozen-trunk mode).
# Should reproduce ~0.5437 heldout / 0.5366 full-set; if it does not, the frames
# path is the suspect, not v1.6.
PYTHONPATH=/workspace/TanitAD/stack python3 eval_flagship_v16.py \
  --ckpt /workspace/experiments/flagship-v15-ab/ckpt.pt \
  --trunk /workspace/experiments/flagship4b-speedjerk-30k/ckpt.pt \
  --poses-val /workspace/v15/poses_val.pt --labels-val /workspace/v15/labels_val.pt \
  --val-cache /workspace/data/physicalai_phase0/_epcache/physicalai-val-0c5f7dac3b11 \
  --anchors /workspace/v15/anchors256.pt --probes /workspace/v15/probes8.pt \
  --vendor /workspace/v15/evalsrc --key flagship-v15-ab-reencoded-control \
  --out /workspace/v15/results/flagship-v15-ab-reencoded-control.json
```

### 5.5 How to read the result (pre-registered, so it cannot be rationalised later)

| Outcome | Reading |
|---|---|
| oracle ↓ toward 0.1640 **and** canary flat | ✅ the hypothesis holds — frozen imitation-optimal features were the proposal bottleneck |
| ADE ↓ but **oracle flat** (~0.3073) | ❌ we bought **ranking**, not proposals. The experiment did **not** do what it was designed to do — say so plainly |
| canary rises materially above 0.4217 | ⚠️ the world model is being destroyed → **back off `--lr-trunk`** (and report it) |
| oracle flat **and** canary rises | ❌ worst case: destroying the world model without buying proposals → stop, revert to frozen `ab` |
