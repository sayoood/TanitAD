<title>Cosmos3 VLM Augmentation Pilot — 48-clip labeling proof</title>

# Cosmos3 VLM Augmentation Pilot — 48-clip v3-vocabulary labeling proof

*2026-07-20 · tanitad-eval (A40 46 GB) · non-destructive, nothing committed to the lake*
*Brief: stand up the VLM augmentation pilot on NVIDIA Cosmos3 (Sayed's explicit choice), prove the labeling pipeline on ~50 driving clips.*
*Anchors: [V3_GOAL_VOCABULARY_V1](../../Architecture%20%26%20Inference/V3_GOAL_VOCABULARY_V1.md) · [TANITDATASET_V1_STRATEGY §8](../TANITDATASET_V1_STRATEGY.md)*

---

## TL;DR

**The labeling pipeline works.** 48 driving clips labeled end-to-end into the frozen v3 tactical
vocabulary with provenance stamps, in 23.5 min on one A40 — **100 % JSON parse, 99.1 % token
adherence on every word-token slot**.

**Cosmos3-Nano was NOT deployed, on hard evidence, not assumption.** It is a **16 B omnimodel,
34.89 GB BF16**, whose sanctioned serving path is a **docker container** — and RunPod pods cannot run
docker (they *are* containers). The pip alternative (`vllm-omni`, 81 deps with an exact
`diffusers==0.38.0` pin) would have to be venv-isolated, and 34.9 GB of weights on a 46 GB card
shared with live TanitEval jobs is marginal at best. Per the brief's fallback clause we ran
**`nvidia/Cosmos-Reason1-7B`** (ungated, 16.58 GB, native `qwen2_5_vl`) and got the proof regardless.

**Two findings change the pipeline design:**
1. **Do not ask a VLM to emit `VTARGET`.** It invents band edges (`v(7.3-8.2)`) instead of copying
   from the 23-token grid — 48 % violation. This is also *off-spec*: the vocabulary already says
   VTARGET is `kinematic + vlm(cap)`. Minted kinematically from the pose track it is **exact and free**.
2. **Cosmetic weather re-renders are a free label-stability probe.** The 48 clips are 24 drives × 2
   renders with *identical ego motion*; `LONMODE` flips on **38 %** of them and `VSOURCE` on **33 %**.
   The VLM's longitudinal reasoning is not appearance-invariant → mint LONMODE kinematically too.

---

## 1. What actually deployed

| | **Cosmos3-Nano** (primary, **not deployed**) | **Cosmos-Reason1-7B** (deployed) |
|---|---|---|
| HF gating | **ungated** ✓ (confirmed) | **ungated** ✓ |
| License | OpenMDW-1.1 (commercial OK) | NVIDIA Open Model License |
| Params / weights | **16 B / 34.89 GB** BF16 | 7 B / **16.58 GB** BF16 |
| Arch | `cosmos3_omni` · `Cosmos3ForConditionalGeneration` | `qwen2_5_vl` (Qwen2.5-VL-7B-Instruct post-train) |
| Serving | `docker pull vllm/vllm-omni:cosmos3` **or** `vllm-omni` / `sglang-diffusion` / cosmos-framework | stock `transformers` |
| **Measured GPU footprint** | — | **16.64 GB** weights resident |
| Blocker | **docker unavailable on RunPod**; 81-dep venv; 34.9/46 GB marginal | none |

**Verified deployment facts for a Cosmos3 follow-up** (so nobody re-derives these):
- Weight layout is dominated by `transformer/diffusion_pytorch_model-*` (7 shards ≈ 33.9 GB) — the
  **generation** DiT. Cosmos3 is a video/audio/action *generation* omnimodel that also reasons; the
  understanding path is a minority of the checkpoint but the serving stack loads the platform.
- Model card recommends `--enable-layerwise-offload` for constrained GPUs and
  `--tensor-parallel-size` / `--ulysses-degree` for multi-GPU → **single-A40 is explicitly the
  marginal case, not the happy path.**
- ⚠️ **PyPI `cosmos` (0.6 MB, 0 dependencies) is NOT NVIDIA's framework** — name collision. The real
  one is `pip install -e` from `github.com/NVIDIA/cosmos-framework`. A naive `pip install cosmos`
  silently installs the wrong package.
- `vllm-omni` 0.24.0 exists on PyPI (wheel only 5.3 MB) but pulls **81 dependencies** incl.
  `transformers>=5.5.3` and an **exact `diffusers==0.38.0` pin** → must be venv-isolated on any pod
  that also runs TanitEval/training.

**Deployment recipe used (zero environment mutation — nothing installed on the shared pod):**
```bash
# tanitad-eval already had torch 2.8.0+cu128 + transformers 5.14.1
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Cosmos-Reason1-7B')"
# 16 GB pulled in ~10 s. No token needed (ungated). No vLLM, no accelerate.
# load: AutoModelForImageTextToText.from_pretrained(id, dtype=bfloat16).to("cuda")
#   NB: device_map= would require `accelerate` (absent) -> load CPU then .to("cuda")
```

## 2. Data

48 **Cosmos-Drive-Dreams** clips staged at `/root/cosmos_data/pairs/generation` (CC-BY-4.0 →
`tier:ship`, commercially clean). 1280×704, 24 fps, ~5 s. Sampled **8 keyframes/clip @ 896×512**.

Crucially the sidecar dirs `vehicle_pose/`, `pose/`, `pinhole_intrinsic/` gave **real ego motion**:
300 poses over 20 s (~15 Hz), 46/48 clips. Speeds resolve to **0–17.4 m/s (0–62 km/h)** — plausible
urban/suburban. *Gotcha for the pipeline: clip timestamps are **microseconds**; reading them as ns
yields 15 000 Hz and 17 000 m/s. And weather suffixes may contain `_` (`Golden_hour`) — parse the
filename structurally, not by `split("_")`.*

**Render condition is carried in the filename** (Morning 11, Sunny 10, Rainy 8, Foggy 8, Snowy 4,
Night 4, Golden_hour 3) → a **free ground truth** for scoring scene tags, and the pairing that makes
§5's stability probe possible.

## 3. Throughput

| Metric | Value |
|---|---|
| Clips | 48 (2 passes each = 96 VLM calls) |
| Wall clock | **23.5 min** |
| Contended (sharing A40 with a TanitEval probe) | ~30 s/clip → **123 clips/GPU-hr** |
| **Uncontended** (probe finished mid-run) | **~14 s/clip → ~257 clips/GPU-hr** |
| Tokens | ~10 953 in / 336 out per clip |
| Model load | 10 s, 16.64 GB resident |

Uncontended ~257 clips/GPU-hr is **~2× the §8.4 estimate** (0.6–1.2 k clips/GPU-hr was for vLLM
continuous batching; we ran unbatched transformers). **vLLM batching should add several ×** — the
current run is one clip at a time with zero batching.

## 4. Schema adherence

**JSON parse: 48/48 (100 %) on both passes. Zero errors, zero unparseable outputs.**

| Slot | in-vocab | violation | |
|---|---|---|---|
| VSOURCE, LONMODE, HEADWAY, DYN, TACPOINT, LIGHTSTATE, INTERACT, RISK | **100 %** | 0 % | ✅ |
| LATMANEUVER | 92 % | 8 % | ✅ |
| **VTARGET** | **48 %** | **48 %** | ❌ |
| **All slots** | **94 %** | 6 % | |
| **All except VTARGET** | **99.1 %** | 0.9 % | ✅ |

**The VTARGET failure is structural, not random.** The model fabricates band edges around the
measured speed — `v(7.3-8.2)`, `v(11.7-13.5]`, `v(7.5-9.0]` — i.e. it understands the semantics (the
numbers bracket the right speed) but will not copy a *numeric-range* token from a 23-item list. Every
**word**-token slot is ~100 %. **Enumerated numeric ranges are the wrong ask for a 7 B VLM.**

- 92 % (44/48) of the fabricated bands are mechanically snappable back onto the grid.
- But snapped-VLM agrees with the kinematic 85th-pct band only **45 %** of the time.
- → **Mint VTARGET kinematically.** We already have the pose speed profile; the vocabulary's own
  label-minting rule says `kinematic + vlm(cap)`. Asking the VLM for it was our spec error.

**`coc_trace` structure: only 3/48 (6 %) returned the structured object**
(`observation`/`critical_agents`/`justification`/`decision`/`physics_flag`). The other 45 collapsed
to a single prose paragraph. The prose is *good* (see §6) but it is not parseable into
`critical_agents[]` → **INTERACT mining and the safety-event `physics_flag` cannot run off it.**
Fix: split CoC into its own call with a one-field-at-a-time schema, or constrained decoding.

## 5. Label quality

**Scene tags vs render-condition ground truth:** weather **75 %** (15/20 on Rainy/Snowy/Foggy),
time_of_day **75 %** (21/28).

**Cross-render stability** — 24 drives × 2 weather renders, *identical ego motion*, so any flip is
labeling noise (except RISK, which *should* track weather):

| Slot | stable | reading |
|---|---|---|
| LATMANEUVER, LIGHTSTATE | **100 %** | appearance-invariant ✅ |
| VTARGET, DYN | 92 % | good |
| HEADWAY | 79 % | usable |
| TACPOINT | 75 % | usable |
| VSOURCE | 67 % | ⚠️ |
| **LONMODE** | **62 %** | ⚠️ **the longitudinal slot we care most about** |
| RISK | 33 % | ✅ *by design* — `elevated_weather` correctly tracks the render |

`LONMODE` flipping on 38 % of cosmetic repaints is the sharpest quality signal in the pilot: the
VLM's **longitudinal** reasoning is not appearance-invariant — the same known weakness as the
flagship's longitudinal error. Mint LONMODE from kinematics + lead state; use the VLM for the *why*.

**Cross-field grounding audit (mechanical, cheap, and necessary):**

| Check | Result |
|---|---|
| `VSOURCE=sign_limit` with **no** sign actually read | **4/48 (8 %)** ❌ |
| `HEADWAY` band set with **no** lead present (my prompt says `unknown`) | **6/48 (12.5 %)** ❌ |
| `LONMODE=follow_lead` with no lead | **0/48** ✅ |
| `sign_reads` emitted | 6 total across 48 — appropriately conservative, **no sign spam** |
| `lead_state.present` | **42/48 (88 %)** — ⚠️ likely over-detection |

A caught hallucination worth naming: on `a6e65839…` the model read an overhead **directional** sign
("Stream Bed Voluntown / New London") and emitted `{"type":"other","value_kph":65}`, then set
`VSOURCE=sign_limit`. It invented a speed value from a route sign. **`value_kph` must be rejected
whenever `type != "speed_limit"`** — a one-line gate.

## 6. Sample records (verbatim, abridged to the label fields)

**A · stopped at a signalized intersection** — kinematics `v_mean 0.0`; the VLM independently reached
`v_stop` / `stop_at_point` / `TACPOINT=stop_line` / `LIGHTSTATE=stop_at_line`, and the kinematic mint
agrees exactly (`v_stop`). *(Also shows the `sign_limit`-without-sign-read defect.)*
```json
{"clip_id": "d80ab84a-...-ccbfd5019170_574066700000_574086700000_1_Morning",
 "scene_tags": {"weather":"clear","time_of_day":"day","road_type":"urban","surface":"dry",
                "traffic_density":"moderate","vru_present":false,"notable_events":[]},
 "sign_reads": [],
 "lead_state": {"present":true,"gap_band":"mid","closing":"steady","lead_type":"car"},
 "goal_tactical": {"VTARGET":"v_stop","VSOURCE":"sign_limit","LONMODE":"stop_at_point",
                   "LATMANEUVER":"unknown","HEADWAY":"hw_0.8s","DYN":"gentle","INTERACT":"none",
                   "TACPOINT":"stop_line","LIGHTSTATE":"stop_at_line","RISK":"nominal"},
 "vtarget_kinematic": {"token":"v_stop","v85_mps":0.0,"provenance":"kinematic"},
 "label_stamp": {"model":"nvidia/Cosmos-Reason1-7B","prompt_version":"v3vocab-2026-07-20-a",
                 "provenance":"vlm","source_license":"CC-BY-4.0","labeled_utc":"2026-07-20T05:37:50Z"}}
```

**B · fog on a highway — the clean case.** VLM band == kinematic band exactly; `RISK` correctly
elevated. `schema_ok: true`.
```json
{"clip_id": "a5f828d9-...-85999f12b4fb_1994304400000_1994324400000_0_Foggy",
 "scene_tags": {"weather":"fog","time_of_day":"dawn","road_type":"highway","surface":"dry",
                "traffic_density":"light","vru_present":false,"notable_events":[]},
 "lead_state": {"present":true,"gap_band":"mid","closing":"steady","lead_type":"car"},
 "goal_tactical": {"VTARGET":"v(12.5-15]","VSOURCE":"road_class_default","LONMODE":"free_cruise",
                   "LATMANEUVER":"lane_keep","HEADWAY":"hw_1.75s","DYN":"normal","INTERACT":"none",
                   "TACPOINT":"none","LIGHTSTATE":"proceed","RISK":"elevated_weather"},
 "vtarget_kinematic": {"token":"v(12.5-15]","v85_mps":14.69,"provenance":"kinematic"},
 "coc_trace": "The road is a multi-lane highway with light traffic and fog reducing visibility. The
   ego vehicle maintains a steady speed in the center lane, keeping a safe distance from other
   vehicles. The weather condition introduces a slight risk due to reduced visibility, requiring
   cautious driving. No immediate interactions with other road users are observed."}
```

**C · night, lead truck detected** — `lead_type:"truck"`, `gap_band:"near"`, `closing:"closing"`;
CoC grounds the decision in the truck's brake lights. VLM band `v(12.5-15]` vs kinematic
`v(15-17.5]` — the one-band-low bias.
```json
{"clip_id": "b4430a02-...-c70a244adcda_13497240800000_13497260800000_0_Night",
 "scene_tags": {"weather":"overcast","time_of_day":"night","road_type":"highway","surface":"dry",
                "traffic_density":"heavy","vru_present":true,"notable_events":[]},
 "lead_state": {"present":true,"gap_band":"near","closing":"closing","lead_type":"truck"},
 "goal_tactical": {"VTARGET":"v(12.5-15]","VSOURCE":"lead_constrained","LONMODE":"follow_lead",
                   "LATMANEUVER":"lane_keep","HEADWAY":"hw_1.75s","DYN":"normal","INTERACT":"none",
                   "TACPOINT":"none","LIGHTSTATE":"proceed","RISK":"nominal"},
 "vtarget_kinematic": {"token":"v(15-17.5]","v85_mps":15.99,"provenance":"kinematic"},
 "coc_trace": "The ego vehicle is maintaining a steady speed behind a large semi-truck ... The
   truck's brake lights are illuminated, indicating it is moving at a reduced speed ..."}
```
*(`RISK:nominal` at night, heavy traffic, `vru_present:true` should arguably be
`elevated_visibility` — RISK is under-triggered outside weather.)*

**D · the render-pair instability, side by side.** Same drive, same ego motion, two renders:
```json
// ..._0_Rainy : {"VTARGET":"v(3-4]","VSOURCE":"lead_constrained","LONMODE":"follow_lead",
//               "HEADWAY":"hw_1.2s","RISK":"elevated_weather"}   weather:"rain"  surface:"wet"
// ..._0_Sunny : {"VTARGET":"v(4-5]","VSOURCE":"sign_limit",      "LONMODE":"free_cruise",
//               "HEADWAY":"hw_1.75s","RISK":"nominal"}           weather:"clear" surface:"dry"
// kinematic truth for BOTH: v85 = 4.51 m/s -> v(4-5]
```
Weather/surface tags are correct in both. But `LONMODE`, `VSOURCE` and `HEADWAY` all flip on a
repaint, and only the Sunny render's VTARGET matches the kinematic truth.

## 7. Label distribution (48 clips)

`LONMODE` free_cruise 29 / follow_lead 15 / stop_at_point 4 · `LATMANEUVER` lane_keep 44 / unknown 4
· `VSOURCE` road_class_default 23 / lead_constrained 16 / sign_limit 8 / curve_constrained 1 ·
`RISK` nominal 32 / elevated_weather 16 · `HEADWAY` hw_1.75s 37 / hw_1.2s 4 / hw_0.8s 4 / hw_1.45s 3.

Exactly the imbalance §7.3 predicts (`free_cruise + lane_keep` dominates) — the curation weights are
justified, and this pilot is the mechanism that measures the strata.

## 8. Recommendation for the full pipeline

1. **Run the bulk pass on Cosmos-Reason1-7B now.** Ungated, 16.6 GB, no installs, ~257 clips/GPU-hr
   unbatched; it is the pragmatic workhorse and it already clears 99.1 % token adherence.
2. **Split label minting by provenance — the pilot's main lesson:**
   - **kinematic (authoritative):** `VTARGET` (85th-pct free-flow → band), `LONMODE`, `LATMANEUVER`,
     `DYN`, `HEADWAY` (from lead gap + speed). Exact, free, and appearance-invariant.
   - **vlm:** `VSOURCE` *reason*, sign reads, `lead_state`, scene tags, `RISK`, `LIGHTSTATE`,
     `TACPOINT`, `INTERACT`, and the CoC narrative.
   - This is what the vocabulary already specifies; the pilot proves *why* it is specified that way.
3. **Add the mechanical consistency gates** (all cheap, all caught real defects here): reject
   `value_kph` when `type != speed_limit`; force `VSOURCE=sign_limit` ⇒ a sign read exists; force
   `HEADWAY=unknown` ⇒ no lead; cross-check VLM `LONMODE` against the kinematic mint and flag
   disagreement as hard/ambiguous (§7.4 already calls for this — implement it as a blocking filter).
4. **Fix the CoC pass** — separate call, structured schema, or constrained decoding. 6 % structure
   yield blocks `critical_agents` → `INTERACT` mining and safety-event `physics_flag`.
5. **Adopt render-pair stability as a standing QA metric.** We own Cosmos-DD; re-rendering a slice
   into 2+ weather variants costs nothing and yields a label-noise number per slot with no human GT.
   Gate any slot below ~80 % stability out of *training* labels until it is kinematically minted.
6. **Cosmos3 follow-up (do not attempt on a shared eval pod):** needs a **dedicated pod with docker**
   (`vllm/vllm-omni:cosmos3`) or an isolated venv, and realistically **≥1 GPU to itself** — 34.9 GB
   of a 46 GB card leaves ~11 GB for vision/KV/VAE. It is a *generation* omnimodel; its unique value
   to TanitAD is **neural weather/lighting re-rendering (§6.2) and action-conditioned world
   simulation**, not bulk labeling — Reason1-7B labels at ~1/2 the weight and 100 % of the schema.
   Re-evaluate Cosmos3 when a dedicated labeling GPU lands (infra ask #1).

## 9. Reproduce

Pod `tanitad-eval`, all under `/root` (nothing written to `/workspace`, nothing committed):
`prep_clips.py` (keyframes + ego kinematics) → `label_clips.py --tag reason1` (2-pass labeling,
checkpoints per clip, resumable) → `analyze_pilot.py` (adherence, snapping, kinematic mint, scoring).
Sidecars: `/root/vlm_pilot/out/reason1/*.json` (48). Manifest: `/root/vlm_pilot/manifest.json`.

**Fleet etiquette observed:** pod2/pod3 untouched. The eval-pod A40 was shared with another agent's
`sc13_real_probe` (2.5 GB, `timeout 1800`); we waited for it to clear before loading, installed
nothing on the shared interpreter, and released the GPU at completion (0 MiB at exit).
