# VLM + geometric strategic labeling — closing the strategic goal-fan gap (PI directive 2026-08-10)

**Status: DESIGN (this doc). Implementation is sequenced AFTER v5.8f core validation closes
(registry §1.14 + T1 rows + the WM-interpretability battery P8/P9), per the PI's own
sequencing. Runs on pod4 (the Alpamayo pod), over the EXACT clips of the augmentation set,
extending that dataset under the same clip_id + t0_us linkage, no raw imagery, HF public
gated.**

## 1. The idea (PI, paraphrased faithfully)

Leverage a performant VLM (Qwen-VL class, 8 B or bigger) + a sophisticated geometric/
algorithmic analysis of LONG-HORIZON INTEGRATED ego trajectories, feeding both the PAST and
the FUTURE as short video clips — the hindsight trick. Extract:
- **Scenario classification**: environment (illumination, weather, day/night), road
  information incl. lanes, description of agents + their behaviour, description of ego
  behaviour.
- **Domain classification**: highway / urban / roundabout / intersection / rural / …
- **Ego-relevant traffic signs**: traffic lights + state, speed limits + state, navigation
  signage incl. OCR of text (cities, locations) and its LINK to ego navigation.
- **Strategic goals + actions**: e.g. *goal: follow route toward <city>*; *action: change
  lane within ≤500 m to keep the left carriageway toward <city>* — derived by joint analysis
  of the current+future ego path and the environment.

This supplies the missing **strategic geometric goal fan supervision (g_str)** for
v5.8f/v1.9's E7.2 layer, and as a by-product the P2 nuisance labels (weather/illumination)
the interpretability battery needs.

## 2. Admissibility (binding rules, restated for this pipeline)

- **Labels-may-use-ego/future**: the VLM sees future clips — privileged, offline, fine.
  Inference-side models NEVER consume these labels as inputs; they are supervision targets
  and eval strata only.
- **Goal/situation disjointness**: scenario/domain classifications are STRATA AND AUX
  LABELS; the strategic GOAL fields must be derivable from path+signage geometry, never a
  function of the scenario classifier's output (the 2026-08-03 rule, applied at label time
  too: goal fields carry their derivation source tag `path|signage|vlm-fused` so any
  contamination is auditable).
- **Honest ceiling (PhysicalAI, settled)**: no GNSS, no map, no route GT. "Toward <city>"
  exists ONLY when read off navigation signage by OCR; otherwise the strategic goal
  degrades gracefully to corridor/lane-level intent (keep-left / exit-right / lane target)
  from hindsight geometry. The schema has explicit `evidence` and `abstain` fields — a
  hallucinated Paris is worse than no Paris.

## 3. Architecture — two engines, adversarially fused

**Engine A — geometric/algorithmic (deterministic, cheap, runs first):**
long-horizon integrated ego trajectory over the full clip (register_poses_to_time clock):
E7.1 corridor machinery (curvature-relative turn/follow + valid mask) + lane-change detector
(lateral displacement vs heading, the E4.1 LANE axis at strategic horizon) + speed-profile
events (E4.1 LON axis) + distance-to-event estimates (arc-length to the next
turn/merge/stop). Output: a structured GEOMETRIC SUMMARY per clip (machine-readable, unit-
carrying), which (a) seeds the VLM prompt, (b) gates its claims.

**Engine B — VLM, THREE PH0 arms (PI decisions 2026-08-11: "qwen 3.5 at least" + "add
[Gemma 4] as third PH0 arm"):** `Qwen/Qwen3.5-9B` (bf16 workhorse), `Qwen/Qwen3.5-27B-FP8`
(bigger-quality arm), and challenger `google/gemma-4-31B-it-qat-w4a16-ct` (official QAT
quant ~17 GB; the survey's benchmark leader at this class — MMMU-Pro 76.9 — but with NO
published OCRBench number, which is exactly what PH0's sign-OCR gate measures). All three
run the same 50 pilot clips; the measured sign-OCR precision + schema compliance +
wall-clock decide PH1's model on the PI's desk. Qwen3.5's native video input is
verified-PUBLISHED (survey: early-fusion multimodal, Video-MME 78.4; no separate VL line);
the runtime video-template check at PH0 remains mandatory for all three. All prefetching on
pod4. PH0 wall estimate rises ~+2 h for the third arm.**
two low-fps clips (PAST: t0−8 s → t0, FUTURE: t0 → t0+12 s; front camera, ~2 fps, 448 px)
+ the geometric summary + a STRICT JSON schema prompt. Two-pass protocol:
1. **Extract** — scenario/domain/signs/agents/ego-behaviour + proposed strategic goal and
   actions, each field with confidence and evidence pointer (frame index, sign text).
2. **Verify** — a second prompt shows the model its own claims against the geometric
   summary and demands per-claim CONFIRM/RETRACT (the adversarial-verify pattern from the
   programme's own playbook, applied inside the labeler).

**Fusion gate (deterministic):** a strategic action is emitted ONLY if Engine B's claim is
consistent with Engine A's geometry (e.g. "change lane left ≤500 m" requires a leftward
lane-change event in the hindsight path within the stated envelope, or explicit signage
evidence). Disagreements are banked as `disputed` rows — not dropped, not trusted.

## 4. Output schema (versioned, v0 draft)

Per clip (keyed `clip_id`, `t0_us`, `schema_version`):
`scenario{illumination, weather, daynight, road{type, lanes_visible, lane_ego}, agents[{class, position_rel, behaviour}], ego_behaviour}`,
`domain{class, confidence}`,
`signs[{kind: light|speed|nav|other, state, text_ocr, applies_to_ego: bool, evidence_frame}]`,
`strategic{goal{kind: route_to|keep_corridor|lane_target|none, target_text, source: signage|path|fused, confidence}, actions[{verb, envelope_m, deadline_s, reason, geometric_consistency: pass|disputed}]}`,
`geometric_summary{...engine A verbatim...}`, `_provenance{model_id, prompt_hash, pass2_verdicts}`.

## 5. Dataset & publication

Extends `Sayood/tanitad-alpamayo2-augmentation` (same clips, ~4,7xx): new
`vlm_strategic.parquet` + updated card (the Alpamayo tasks and this table cross-linked by
clip_id+t0_us; no raw imagery anywhere). Public, gated. VQA-bank style provenance: prompts
+ schema banked in-repo.

## 6. Phasing, cost, gates (pre-registered before each phase)

- **PH0 pilot (50 clips, ~2 h pod4):** taxonomy freeze + measured per-clip wall + a
  100-field human-check sample (PI eyeballs a sheet) → gate: sign-OCR precision ≥0.9 on the
  checked sample, strategic-action geometric-consistency rate reported (no threshold —
  baseline row); prompt/schema iterate here ONLY.
- **PH1 batch (~4,7xx clips):** measured pilot wall × N (budget estimate at pilot; VLM pass
  is the driver — expect 20–60 s/clip on A40 ⇒ 26–78 h; PI approves the spend from the
  pilot number, not from this guess).
- **PH2 integration:** g_str supervision stream for E7.2; domain-stratified four-family
  evals; P2 nuisance probes consume the weather/illumination fields; sign-conditioned goal
  experiments (v1.9).

## 7. Sequencing (binding, per the PI)

Blocked until: registry §1.14 (assembled v5.8f, T0+families) ∧ E1.4 T1 rows ∧ P8/P9 battery
runs. Pod4's current queue (straggler sweep → final pack → card → I1a join → E1.4
byte-close) completes first; PH0 slots after. Design/prompt/schema work (0-GPU) may proceed
in parallel without violating the sequencing — implementation agents are NOT to be spawned
until the block clears.

- [x] PH0 prereg written (`PREREG_PH0_VLM.md`, 2026-08-11 — gates + both
  outcomes + PH1 selection rule bound) · [ ] PH0 run · [ ] taxonomy frozen ·
  [ ] PH1 · [ ] PH2 wired
