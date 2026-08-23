# PH0 pipeline — validation verdict and the full extracted structure

**PI, 2026-08-13:** *"Did you validate the correctness and the semantic quality of the
results? Can we start the production, can you give the whole structure and information you
are extracting to review it."*

**Verdict: three of four channels VALIDATED, one FAILED — production may start with the
failed channel demoted, which is the role split already decided.** Everything below is
MEASURED on the 50-clip pilot (`/workspace/ph0_pilot50/`), not the 8-clip smoke.

---

## 1. The verdict per channel

| channel | engine | status | evidence |
|---|---|---|---|
| **Schema / structure** | B (Qwen3.5-9B, grammar-constrained) | ✅ **PASS** | **50/50 clips all-calls-valid; 0 parse failures, 0 violations, 0 retries.** Gate G2 ≥ 0.90 → measured **1.00**. |
| **Agent / object detection** | C (SAM3) | ✅ **PASS** | **1 383 detections / 50 clips**: car 703 · sign 292 · light 167 · pedestrian 130 · truck 57 · bus 22 · cyclist 12. Correct abstention verified (0 cars on an empty road, 16 trees on a treed one). |
| **Geometry / ego** | A (integrated path) | ✅ **PASS** | Deterministic; route token + speed profile + situations + pre-decision ego state present on every clip. |
| **Sign box grounding** | B (call B3) | ⛔ **FAIL** | **2 / 49 agree with SAM3** on the *exact same frame*. See §3 — this is now a clean result, not an artifact. |

**Semantic quality, MEASURED rather than asserted:** 49 signs read across 50 clips, 31 with
non-empty OCR text; kind distribution `other 24 · speed 16 · light 4 · nav 3 · yield 2`.
Goals `follow_main_road 45 · route_to 5`, and every `route_to` is backed by a nav sign index
(the abstention rule held — no invented cities). Throughput **48.9 s/clip median, 52.5 s p90**.

⚠️ **What is NOT validated and cannot be by me:** **G1, sign-OCR precision.** The
pre-registration requires the PI to grade a sample. 31 texts exist to grade. Until then
`route_to` goals and `signs[].text` are **provisionally admitted for production extraction
but not for PH1 supervision** — the labels get written, the gate decides later whether they
are used.

---

## 2. The full extracted structure — verbatim from a real pilot clip

```jsonc
{ "schema_version": "ph0-v2.2", "ego_prompt_mode": "past",
  "clip_id": "<uuid>",

  // ── ENGINE B / call B1 — SCENE (closed vocabulary) ──────────────────
  "scene": { "illumination": "day",      // day|dusk|night|dark
             "weather":     "clear",     // clear|rain|snow|fog|unclear
             "road_type":   "urban",     // highway|urban|rural|junction|unclear
             "domain":      "urban",     // +roundabout|intersection
             "lanes_visible": 2, "lane_ego": 1, "conf": "high" },

  // ── ENGINE B / call B2 — SIGNS (verbatim OCR, never invented) ───────
  "signs": { "n_signs": 1,
             "signs": [ { "kind": "speed",        // light|speed|nav|stop|yield|other
                          "text": "20",           // VERBATIM, "" if illegible
                          "state": "none",        // red|amber|green|none
                          "applies_to_ego": true } ] },

  // ── ENGINE B / call B3 — GROUNDING  ⛔ DIAGNOSTIC ONLY (§3) ─────────
  "grounding":    [ { "visible": true, "frame_idx": 0,
                      "bbox": [320,100,360,160] } ],   // Qwen 0–1000 space
  "grounding_px": [ [143,18,161,29] ],                 // → this frame's pixels

  // ── ENGINE B / call B4 — SYMBOLS (the g_str/a_str vocabulary) ───────
  "symbols": { "goal_kind": "follow_main_road",   // 11 g_str tokens
               "goal_evidence_sign": null,        // index into signs[] iff route_to
               "actions": [ { "verb": "hold_corridor", "direction": "none" } ],
               "conf": "med" },

  // ── ENGINE A — PRE-DECISION EGO STATE (past-only, 8 s window) ───────
  "ego_state": { "window_s": 8.0, "v_now_ms": …, "v_now_kmh": …,
                 "v_mean_ms": …, "v_min_ms": 5.96, "v_max_ms": 10.99,
                 "accel_1s_ms2": 0.18, "accel_3s_ms2": 0.16,
                 "yaw_rate_rad_s": -0.003, "net_dyaw_rad": -0.024,
                 "dist_travelled_m": 82.0,
                 "motion": "steady", "turning": "straight" },

  // ── audit trail (every call's PROMPT + RAW OUTPUT is banked) ────────
  "_calls": [ … ], "_all_valid": true, "_n_parse_fail": 0,
  "_n_violation_fail": 0, "_n_retried": 0,
  "_n_sign_dupes_dropped": …, "_n_action_dupes_dropped": …,
  "_ego_prompt_mode": "past", "_frame_wh": [448,179] }
```

Engine A additionally supplies, per clip, from the integrated ego path: `route{token,
token_valid, dist_m, arc_m, maneuver_dyaw_rad, graded_route}` · `lane_change_events[]` ·
`speed_events[]` · `speed_profile{v_t0, v_min_future, v_max_future, net_dv, stops}` ·
`peak_kappa_per_m` · **`situations{lane_change, intersection, roundabout, *_windows_s}`**
(frozen detectors, deliberately NOT routed into the goal prompt).

```jsonc
// ── ENGINE C — SAM3, one record per clip ───────────────────────────────
{ "clip_id": …, "frame_wh": [448,179], "n_frames_run": 6, "n_det_total": 61,
  "per_concept_hits": { "car": 12, "truck": 0, "bus": 0, "pedestrian": 0,
                        "cyclist": 0, "traffic light": 8, "traffic sign": 20 },
  "frames": { "<frame_idx>": { "n_det": 5, "det": [
      { "concept": "car", "score": 0.91,
        "box_xyxy": [x0,y0,x1,y1],        // ORIGINAL frame pixels
        "mask_area_px": 1234,
        "rle_rows": [[row,start,end], …]  // exact mask, JSON-bankable
      } ] } },
  "vlm_cross_check": [ { "vlm_box": …, "vlm_label": "speed", "frame_idx": 3,
                         "sam3_frame_idx": 3, "frame_aligned": true,
                         "n_sam3_signs_on_frame": 2,
                         "best_sam3_sign": {"iou":…, "sam3_box":…, "sam3_score":…},
                         "matched": false } ] }
```

---

## 3. The one failure, decomposed — and why it is now trustworthy

**2 of 49.** The earlier 0/8 was *my* confound (I compared frames up to ~3.5 s apart). With
the cross-check made frame-exact, the decomposition is:

- **26 of 49** — SAM3 saw **no** sign on that exact frame.
- **23 of 49** — **both engines saw a sign on the same frame**, and only **2** overlap.

So on the frames where both fire, they agree on *location* **2/23 ≈ 9 %**. Two independent
detectors disagreeing that badly means at least one grounding channel is unreliable, and
SAM3's boxes are the ones cross-validated by their own masks and confidence scores.

⇒ **B3 is demoted to diagnostic-only. SAM3 supplies all boxes and pixels; the VLM supplies
symbols and OCR.** This is exactly the role split already decided, and this measurement is
its evidence rather than a precaution. ⚠️ Note what is *not* claimed: this does not show the
VLM's sign **classification** is wrong — B2 and B3 are separate calls, and B2 passed schema
validation on every clip.

---

## 4. Production readiness

| requirement | state |
|---|---|
| schema stability at scale | ✅ 50/50, zero retries |
| resume-safety | ✅ `--resume` skips all-valid clips; incremental write after every clip |
| per-clip failure isolation | ✅ a bad clip lands in `failures.json`, never kills the batch |
| corpus bridged | ✅ **2400/2400 clips, 5.46 GB** (mp4 + ego) on pod5 |
| cost | **MEASURED 48.9 s/clip → ~64 h** for 4 729 clips (VLM), + SAM3 |
| ⛔ blocker | the 5.46 GB has **not** been transferred pod5 → pod4 |

**Production may start** on the val-cache clips already on pod4 while the transfer runs.
The one thing production must NOT do is treat `grounding` as ground truth.
