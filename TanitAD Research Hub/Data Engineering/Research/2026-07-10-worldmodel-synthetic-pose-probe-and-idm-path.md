# Data Engineering — 2026-07-10: WorldModel-Synthetic pose probe (the gating question, settled) + the IDM path it forces

**Run type:** scheduled Tuesday agent, isolated worktree `agent/data-engineering-20260710` (D-026).
**Loop:** 1 iteration. Budget used: 3 web searches + 1 card fetch + ~30 HF tree/API calls;
wall ~55 min; **cost $0** (local RTX-4060 host + web + one 14 MB HF download).
**QUALITY: full** — G-A…G-C, G-E, G-H, G-D1, G-D2 met; measured experiment with numbers;
intake pkg standalone-green; shared-file rows applied in-worktree (D-026 removes the concurrent-write race).

---

## 0. Headline (measured, decision-grade for the loader path)

**PhysicalAI-WorldModel-Synthetic-Scenarios ships NO ego pose / actions / CAN / calibration.**
Backlog **P0.1** ("pose probe") is closed with a *negative* result, confirmed two independent ways:

1. **File-tree probe** (`probe_worldmodel_synth.py`, 18.9 s, 15 clips sampled across all 5 families):
   every clip's directory contains **exactly** `video/` + `description/`; file extensions are **only
   `.mp4` and `.json`** (105 of each in the sample = 15 clips × 7 cameras); **0 pose/action/odom/calib
   file hits**; the `description/*.json` top-level keys are `{framerate, nb_frames, t2w_windows,
   metadata}` with `metadata = {weather, time_of_day, surface_type, region}` — **no pose key**.
2. **HF card fetch** (2026-07-10) states verbatim: *"No ego-vehicle pose, trajectory, actions,
   steering, or CAN signals are provided … only video and VLM-generated captions paired with
   environmental metadata,"* license **OpenMDW-1.1**, and **no companion dataset** for obtaining actions.

→ The prior working assumption ("near-zero cosmos-mirror" — reuse `cosmos_drive.poses_to_signals`
+ the action-conditioned 9-ch contract) is **falsified**. This corpus is **IDM/H7-gated** (needs a
trained inverse-dynamics head to pseudo-label actions) **or video-only** (self-supervised visual
pretraining / caption-conditioned generation). It does **not** enter the action-conditioned D-010
training mix until action labels exist.

## 1. What the corpus actually is (measured on real bytes)

Layout: `‹family›/‹clip_id›/{video,description}/‹cam›.{mp4,json}`, families =
`emergency · lanechange · nudging · pedestrian · weather_degradation` (≈264 k clips / 8.3 TB total,
per card). 7 cameras: `front_tele, front_wide, left_fisheye, rear_fisheye, rear_left, rear_right,
right_fisheye`.

Decoded one real clip (`emergency/…_011922/video/front_wide.mp4`) on the 4060:

| Property | Measured |
|---|---|
| download (single HTTPS stream, TLS-proxy) | **14.09 MB in 4.2 s** (~3.3 MB/s) |
| resolution | **3840 × 2160 (native 4K)** |
| container fps | **24.0** — the *real* rate (matches `description.framerate`; **not** a mux artifact, unlike Cosmos where 24 was) |
| frames | 462 → **19.25 s clip** |
| A8 `frame_change_fraction` @0.05 / @0.10 | **0.0248 / 0.0137** (this emergency/night stop-sign clip = low motion; cf. comma-real 0.053/0.012) |
| per-clip size (all 7 cams) | ≈118 MB; **front_wide alone ≈14 MB** |

`description/front_wide.json` (real): `framerate 24.0`, `nb_frames 462`, one `t2w_windows` entry with
a `qwen2p5_7b_caption` ("At night, the ego vehicle stops at a stop sign … waits for a police vehicle
to pass …"), `metadata {weather: Clear, time_of_day: Night, surface_type: Asphalt, region: Highway}`.

**Consequence:** the frames are real, 4K, cleanly decodeable, and (via D-016 focal-canon of
`front_wide` at nominal 120° HFOV) geometrically the **same task** as comma2k19/Cosmos — usable
*today* for video-only pretraining and for **scene sourcing** (the caption + scene-metadata index).
The only missing ingredient is the ego action/pose track.

## 2. Increment (G-E/G-D2) — a *video-only* loader that refuses to fabricate actions

Intake pkg `Implementation/incoming/2026-07-10-worldmodel-synthetic-pose-probe/` (**10 tests, standalone-green**;
0 new deps; `tanitad` importable). Two modules:

- **`probe_worldmodel_synth.py`** — the runnable gating experiment above (tree-navigation, not a full
  `list_repo_files` walk which hangs on ~3.7 M paths; downloads one description JSON; emits a JSON
  verdict). Artifact: `worldmodel_synth_probe.json`.
- **`tanitad_worldmodel_synth.py`** — loader for the `family/clip/{video,description}` layout:
  `discover_clips` (with `family`/`weather`/`time_of_day` filters), `parse_description`,
  `build_episode` (front_wide → D-016 focal-canon → D-015 9-ch stacks, reusing `stack_frames` +
  `focal_crop_resize`), `build_manifest` (scene index for the scenario duty), CLIP-level `split_clips` (I3).

  **The honesty design (P8), because there are no actions:**
  - `build_episode` emits contract-shaped `actions [T,2]` / `poses [T,4]` as an explicit **NaN sentinel**
    (`ACTION_SOURCE = "idm_pending"`) — never fabricated zeros. `assert_video_only_contract` asserts
    both the 9-ch frame contract **and** that actions/poses are all-NaN. NaN makes any action-conditioned
    trainer fail **loud** (NaN loss), never silently train on invented actions.
  - `CORPUS_META["actions"] is None` is **load-bearing**: `i7_task_identity(comma2k19, wms)` returns
    *not-identical* (mismatch on `actions`), so `MixedWindowDataset` / probe-fit admission **mechanically
    exclude** this corpus from the action-conditioned mix — the same mechanism that correctly rejected
    the 1-ch BEV adapter (test `test_i7_excludes_from_action_mix`). The frame geometry
    (`channels/image_size/f_eff_px`) **does** match comma (test `test_frames_geometry_matches_task`) so
    video-only pretraining shares the encoder.

This is the correct G-D2 shape: the loader ships an episode-contract test, **and** the contract test
encodes the honest limitation instead of papering over it.

## 3. The path this forces: IDM pseudo-labeling (H7) — and its sharpest caveat

With no actions, WMS becomes the canonical **VPT/IDM setting** (Baker 2022: train an inverse-dynamics
model on a small action-labeled corpus, pseudo-label a vast unlabeled video corpus). Our labeled
bridge already exists — comma2k19 (real CAN) + Cosmos-Drive-Dreams (pose-derived) — and the H7 latent-
action literature has surged (LAWM, Drive-JEPA, CLAW/DeFI in the KB; new this run below).

**But the literature's own warning lands directly on this corpus (actionable, honest):**
inverse-model errors *"accumulate precisely at distribution edges where world models most need reliable
supervision"* (video-WM survey / VPT lineage). **WMS is almost entirely distribution edges** — its
value is exactly the long tail (emergency, weather-degradation, pedestrian, nudging). So an IDM trained
on comma2k19 highway + Cosmos will be **least reliable exactly on the WMS clips we most want to label.**

**Recommendation (G-B), tied to H7:**
1. **Do not** pseudo-label WMS with a highway-only IDM and feed it to the trained mix. First **validate
   the IDM on held-out real long-tail actions** — Zenseact ZOD (real CAN, EU/night/winter — backlog P1)
   is the right validation corpus because it has the edge distribution *with* ground-truth actions.
   Falsifier: IDM steer/accel error on ZOD night/urban ≤ its error on comma highway ⇒ trust WMS labels;
   if it blows up on the edges ⇒ WMS stays video-only.
2. **Use WMS video-only now** where action fidelity is irrelevant: (a) self-supervised visual
   pretraining of the encoder on 4K long-tail scenes; (b) **D8 degraded-visibility OOD probe** — the
   `weather_degradation` family (≈9.2 %) is a *never-trained* held-out visual OOD source complementing
   Cosmos weather pairs for SC-05; (c) caption-conditioned scenario retrieval (the manifest).
3. Watch **DriveWAM** ("video generative priors enable scalable world-action modeling") — if a
   generative-prior IDM closes the edge-reliability gap, WMS graduates to the action mix.

## 4. Literature sweep (D-013; H7 now the operative lens)

- **DriveWAM** (arXiv 2605.28544) — *Video Generative Priors Enable Scalable World-Action Modeling*:
  the paradigm that turns action-unlabeled video into world-action training via generative priors →
  the most relevant route for WMS's video-only bytes. **Watch as the WMS-graduation trigger.** — H7/H5.
- **Latent-WAM** (arXiv 2603.24581) — *Latent World Action Modeling for E2E driving*: spatially-aware,
  dynamics-informed latent world representations for planning → external support for latent-space
  planning (H3) + latent action heads (H7). No status change (P8).
- **IDM-error-at-edges** (video-WM survey 2411.02914 / VPT lineage): the caveat in §3 — the single most
  decision-relevant literature point this run because it directly shapes the WMS ingestion plan.
- **Adjacent watch (models, not ingestible corpora):** `nvidia/omni-dreams-models` now **public on HF**
  (action-conditioned generative WM, multi-cam from actions) — Phase-1 sim-arm watch, mirrors the
  Tools&DevEnv AlpaSim note; **Ego-1K** (2603.13741, egocentric multiview) — not driving-AD, skip.

Consumed Monday (Tools&DevEnv 2026-07-09): CARLA render blocker root-caused (graphics-pod recipe,
not urgent); **pin `stack/` to Drive "Available offline"** kills ~30 s cold-I/O tax per agent run
(applies to me too); AlpaSim now clonable (Phase-1). No loader impact this run.

## 5. Scenario-DB data sourcing (joint duty D-020 §5) — refined, honestly

WMS advances the SC-02/05/06 **scene** rows but the earlier "data-source identified" language needs the
no-action caveat: **WMS supplies perception/OOD/video-only VALIDATION scenes, not closed-loop telemetry**
(the LAL/OKRI/LOPS oracles need ego actions, which WMS lacks). Rows updated:
- **SC-02** (occluded ped): WMS `pedestrian` family = held-out **perception/imagination** scene source
  (no ego response) → complements CARLA scripted-occluder for the closed-loop half.
- **SC-05** (degraded visibility): WMS `weather_degradation` (≈9.2 %) = a second **never-trained D8 OOD**
  visual source alongside Cosmos weather pairs.
- **SC-06** (emergency vehicle): WMS `emergency` family (≈2.7 %) **fills the documented public-data gap**
  as a visual scene source (light patterns); still no ego-response telemetry → stays `catalogued`.

## 6. Sources
- HF card + tree API: `nvidia/PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios` (probed 2026-07-10).
- DriveWAM https://arxiv.org/pdf/2605.28544 · Latent-WAM https://arxiv.org/abs/2603.24581 ·
  video-generation×WM survey https://arxiv.org/html/2411.02914v1 · VPT (Baker 2022, IDM pseudo-labeling).
- `nvidia/omni-dreams-models` (HF, public) · Ego-1K https://arxiv.org/abs/2603.13741.
- Repo: `probe_worldmodel_synth.py`, `tanitad_worldmodel_synth.py`, `tests/` (this intake pkg).
