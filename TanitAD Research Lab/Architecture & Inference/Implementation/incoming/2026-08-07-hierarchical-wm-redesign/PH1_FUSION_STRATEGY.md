# PH1 fusion — combining ego, VLM, SAM3 and Alpamayo into one aligned record

**PI, 2026-08-14:** *"Develop a sophisticated strategy to combine the strengths of the three
sources, ego, vlm and sam3 … take the data set which was augmented with Alpamayo and extend
this augmentation by VLM/ego/sam3 to extract the rich aligned and highly structured
information including scenario description, signs, our aligned tactical and strategic
vocabulary and environment perception."*

**Implementation: `stack/scripts/ph1_fuse.py`** (committed with this doc). One fused record
per clip, schema `ph1-fused-v1`, written incrementally, pushed to HF per batch beside the
v2/sam3 outputs it consumes.

---

## 1. The design principle: each source is AUTHORITATIVE for one thing

The strategy is not averaging — it is **jurisdiction**. Each source owns the layer it is
measurably best at, and the others are used to *corroborate*, never to overwrite.

| source | authoritative for | why (MEASURED) | trust class |
|---|---|---|---|
| **ego** (Engine A) | the metric spine: kinematics, route token, speed profile, situations windows | deterministic, from the corpus itself; not a model output | privileged — **labels only, never inference** |
| **SAM3** | pixels & geometry: boxes, masks (RLE), agent census, tracks | boxes self-validated by masks+scores; correct abstention verified; 16 119 dets/598 clips | vision — inference-admissible |
| **VLM** (Qwen3.5-9B) | symbols & semantics: scene class, sign kind/OCR/state, goal + action vocabulary | schema 50/50 all-valid, 0 violations; but its **grounding measured 2/23 vs SAM3** → B3 demoted | vision — inference-admissible |
| **Alpamayo-2** | an external VLA's independent opinion: trajectory, meta_action, auto_labeling, vqa, grounding_via_vqa | independent model family and prompt stack → its agreement is *evidence*, its disagreement is *signal* | external — labels only |

The one measured failure drove the one hard rule: **pixels come from SAM3, meaning comes
from the VLM, and the VLM's own boxes are diagnostic-only.** That split was decided from the
2/49 grounding result, and the fuser enforces it structurally — there is no code path that
promotes a VLM box to geometry.

## 2. Alignment — the three axes

1. **Temporal.** Everything is indexed to the 10 Hz egomotion clock. SAM3 already runs on
   `frames = stride grid ∪ VLM frames` (frame-exact by construction after the cross-check
   fix); Alpamayo records carry `t0_us` on the same clip clock. The fused record stores
   per-frame keys, never "approximately the same time" — the 0/8 cross-check confound was
   exactly an alignment-by-vibes bug and it is not allowed back.
2. **Spatial.** SAM3 boxes live in original 120° cylindrical pixels (256×640); ego lives in
   ego-frame metres. They meet through **track dynamics**, not through a calibration we don't
   have: per-track box-area growth + image-x drift + ego speed give approach/recede/crossing
   classification without inventing a depth we cannot verify.
3. **Semantic.** The VLM's closed vocabularies (scene, sign kind, goal, actions) are the
   join keys; SAM3 concepts map onto them (`traffic sign`↔sign, `traffic light`↔light,
   agent classes↔census). Alpamayo's `meta_action` maps onto our factored tactical
   vocabulary via a declared, versioned table inside the fuser.

## 3. The corroboration layer — where fusion earns its keep

Pairwise, independent checks. Each emits `{verdict, margin, n}` — never a silent merge.

| check | sources | rule (versioned in code) |
|---|---|---|
| **speed-sign ↔ ego** | VLM sign `kind=speed, text=v` + ego speed profile | corroborated if future `v_min/v_now` is consistent with the limit (±15 %) |
| **light-state ↔ ego stop** | VLM `state=red` + ego `stops>0` or `v_min<0.5 m/s` | red+stop ⇒ corroborated stop scenario; red+no-stop ⇒ **conflict, clip flagged** |
| **scene ↔ situations** | VLM `road_type/domain` + ego situations windows | `intersection` claims must co-occur with an ego intersection window |
| **goal evidence** | VLM `route_to` + its evidence sign + SAM3 sign on that exact frame | a `route_to` whose evidence sign SAM3 also sees is grounded; else provisional |
| **tactical 3-way** | ego events · VLM actions · Alpamayo meta_action | 2-of-3 majority per LAT and LON axis; the minority report is kept |
| **census ↔ scene** | SAM3 per-concept counts + VLM scene | `urban` with 0 agents over all frames ⇒ flagged (either empty road — fine — or a perception miss) |

**Disagreement is a feature.** Conflicted clips are not averaged into mush — they land in
`_conflicts` with both readings, and the conflict count feeds the curation weight (the same
QQT lever as O4 saliency: hard clips are the informative ones). This is also the audit
surface: a systematic conflict pattern is how the next B3-class defect gets found.

## 4. The vocabulary emission — g_str / g_tac, with the leak rules enforced

- **`g_str`** (11 strategic tokens): from VLM B4 `goal_kind`, corroborated against the ego
  route token; `route_to` requires grounded sign evidence (§3) to be emitted at full
  confidence.
- **`g_tac`**: **factored LAT × LON by design** (the 5-way mixed softmax is our named
  defect, not a template). LAT from {ego turning/lane-change events, VLM action direction,
  Alpamayo lateral meta-action} by 2-of-3; LON from {ego speed events, VLM verbs, Alpamayo
  longitudinal meta-action} by 2-of-3.

**Binding rules, enforced structurally in the fuser:**
1. **Labels may use ego; inference is vision-only.** Every field carries
   `src ∈ {ego, vlm, sam3, alpamayo}` and the record carries an explicit
   `inference_admissible` whitelist — the vision-only fields. Nothing ego-derived can be
   silently promoted to an inference input.
2. **The goal fields never contain situation-classifier output.** Situations live in the
   ego layer and the corroboration layer ONLY. `assert` in code, not convention — the
   goal/situation information-disjointness rule (2026-08-03) survives the fusion.
3. Sign OCR text is carried but marked `pending_g1_gate` — extraction yes, PH1 supervision
   only after the PI grades the 31-text sample.

## 5. Scenario description — composed, not asked for

A free-text scenario line is generated **deterministically from fused fields** (scene +
situations + census + dynamics), e.g. *"day, clear, urban 2-lane; ego 8.4 m/s decelerating
to stop; red light corroborated; 3 vehicles (1 approaching), 1 pedestrian; intersection
window 2.1–6.0 s"*. Deterministic composition means it can never contradict its own
structured fields — asking a model to summarise the record would reintroduce exactly the
consistency problem the schema exists to prevent. (Alpamayo's `auto_labeling` text is kept
alongside as the external description.)

## 6. What consumes this

| consumer | fields |
|---|---|
| **S-T / S-S supervision** (the hierarchy's goal/action heads) | `vocab.g_str`, `vocab.g_tac` (+confidence ≥ threshold, conflicts excluded) |
| **situation classifier** (`head_img`, vision-only) | scenario fields as *labels*; inputs remain image-only |
| **curation / O4-style weighting** | conflict count, corroboration rate, census density |
| **eval joins** (lead_gap etc.) | SAM3 tracks + ego, per frame |

## 7. Honest limits

- **No metric 3D from fusion**: without calibration to rectified depth, track dynamics give
  ordinal (approaching/receding), not metres. `obstacle.offline` (97.44 % coverage) remains
  the metric agent source where needed — a privileged label channel, same class as ego.
- **The two video regimes must not be pooled**: w120 256×640 cylindrical vs epcache 256²
  pinhole recall differ by construction; `geometry` is stamped in every record.
- **Alpamayo coverage is partial** (56 done / 257 with w120 video / 4 472 without w120
  cache); the Alpamayo layer is nullable and its absence is recorded, not imputed.
