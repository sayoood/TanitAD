# G1 sign-OCR review — verdict: UNVERIFIABLE AT PIPELINE FIDELITY; gate stays CLOSED

**Reviewer:** the orchestrator (PI delegated 2026-08-14: *"regarding ocr do the review on
your self without me"*). **Method:** for each of the 31 pre-registered pilot OCR texts, the
sign was cropped from the pilot videos using **SAM3's boxes** (the trustworthy geometry per
the measured role split) — one crop from the largest-area sign/light detection, one from the
second-largest — 4× LANCZOS upscale, all 54 tiles assembled into one labeled contact sheet
(129 077 B, md5-verified transfer) and read visually.

## The verdict

**0 of 31 rows could be verified.** Not one claimed text is legible in the best available
crop. The rows split into two honest subclasses:

| subclass | ~n | examples |
|---|---|---|
| **no sign visible in the crop at all** — sky, foliage, building walls, clouds | ~22 | row 1 (claimed "20": gray sky), row 2 ("100": trees), row 5 ("80": blank sky), row 10 ("35": uniform brown), row 31 ("11": storm clouds) |
| sign-like structure discernible, text illegible | ~9 | row 3 ("P": distant blue board), row 11 ("100": night gantry scene), row 26 ("SLOW DOWN": rural road), row 22 (Greek pharmacy: street pole scene) |

## What this means — three findings, in decreasing order of certainty

1. ⛔ **The G1 gate stays CLOSED.** `signs[].text` and `route_to` goals remain
   **extraction-only, excluded from PH1 supervision** (the fused records already enforce
   `pending_g1_gate`, so nothing downstream changes — the gate simply does not open).
   This is *not* a measured OCR failure: it is the stronger statement that the claims are
   **unfalsifiable at the fidelity the pipeline operates at** (448-wide bridged frames — the
   same frames the VLM itself read). Several claimed texts have the shape of VLM priors
   filling in a blurry board ("Kreuzberg", "APOTHEKE", "The Sea", "ΦΑΡΜΑΚΕΙΟ"); plausible,
   unverifiable, and therefore inadmissible as labels.

2. ⚠️ **A NEW finding that outranks the OCR question: SAM3's "traffic sign" detections are
   themselves suspect on this corpus.** The crops came from SAM3's own sign boxes — and
   ~two thirds contain *no sign at all* (sky, foliage, walls), including detections scoring
   0.87–0.94. Production banked **4 048** "traffic sign" detections on the 600-clip set;
   if this false-positive character generalises, the sign-pixel channel of the fusion needs
   a score/size threshold study before it is treated as authoritative for *signs*
   specifically (agent classes — car/pedestrian — were separately validated and are not
   implicated). Filed as a fusion-layer concern: the `census_vs_scene` and sign-association
   checks should carry a sign-class reliability flag until this is measured properly.

3. ⚠️ **Caveat on the reviewer's own channel, stated rather than hidden:** the review chain
   (256×640 → 448-wide bridge → crop → 4× upscale → JPEG → viewer) cannot resolve glyphs
   that span ~10–30 px at source. Rows in subclass 2 might be legible to a human at a
   monitor with the raw source. But subclass 1 — *no sign present in the box* — is not a
   resolution artifact and stands at any fidelity.

## The route to actually closing G1

The original camera frames exist at full resolution in the **nvidia/PhysicalAI-Autonomous-
Vehicles chunks** (`camera/camera_front_wide_120fov/*.zip`) — the same archives the
4 472-clip build needs. When the chunk-index build machinery lands (already scoped), a
G1 re-run on **native-resolution crops** becomes cheap: same 31 rows, same protocol, real
legibility. Until then the gate stays closed, and nothing in PH0/PH1 depends on it opening.
