# TanitAD Data Strategy (v1.0, 2026-07-06 — implements D-009/D-010/D-012)

**Goal.** A *first rich dataset* for Phase 0: real camera data with genuine urban semantic
distribution (intersections, pedestrians, traffic lights, dense interaction, weather/night), real
actions, and a clean path to scale — while keeping license exposure auditable and the public-claims
story defensible.

## 1. Corpus roles (who provides what)

| Corpus | Role | Semantic profile | Actions | License position |
|---|---|---|---|---|
| **PhysicalAI-AV (PRIMARY RICH, D-012)** | urban diversity backbone of Phase 0 training; H2 multi-view demo | 1 727 h, 25 countries, 2 500+ cities: intersections, pedestrians, lights, weather — the distribution comma2k19 lacks; access verified, `clip_index.parquet` enables scenario-filtered subsets | egomotion (poses → yaw-rate/accel) | **use now, resolve later** (D-012); every consuming experiment tagged `data:physicalai` |
| comma2k19 (BOOTSTRAP + PUBLIC ANCHOR) | real-CAN action grounding; ALL public open-loop numbers until licenses resolved | highway commute only — honest limitation | real CAN (steer, speed) | MIT — clean |
| NVIDIA synthetic corpora (SIM-DATA ARM, D-014): WorldModel-Synthetic-Scenarios + Cosmos-Drive-Dreams | pre-rendered long-tail: emergency, lane change, nudging, pedestrian, weather degradation (H6/H15/D9 training material) | targeted safety-critical scenarios; ungated | scenario egomotion | Cosmos-Drive-Dreams CC-BY-4.0 (publicly safe); scenarios corpus per its card |
| CARLA on RunPod (CLOSED-LOOP ARM, D-014; W31–32) | D5/D6 topology gates, G0.5 closed loop, occluder LOPS, off-expert perturbation rollouts | procedural towns; UE4/5 | exact | MIT/open |
| ~~MetaDrive~~ | retired per D-014 (packaging unmaintainable on modern Python) | — | — | — |
| nuScenes-mini | D8 OOD probes only (never trained on) | urban, 2 cities | ego pose | research |
| Own GoPro/smartphone + OpenDV/YouTube | Phase 1 H7 pseudo-labeling scale-up | uncontrolled diversity | via IDM (H7) | own / public video |

## 2. PhysicalAI-AV staged ingestion (the "first rich dataset")

- **Stage R0 — urban starter (this week):** filter `clip_index.parquet` for urban/interactive
  scenarios → **500 front-wide-camera clips (20 s @ high fps → 10 Hz, ≈ 2.8 h)**; convert to the
  episode contract (6-ch 2-frame stacks @ 256 px; actions from egomotion: yaw-rate + accel; poses
  from egomotion curves). Loader: harden `DataEng/AVDataSetLoader/Loading_NvidiaDataSet.ipynb` into
  `stack/tanitad/data/physicalai.py` (DataEng agent top implementation duty, Tuesday; MVP stream
  assists). Splits: by clip-session/geography (I3), never by frame.
- **Stage R1 — diversity set (week 2):** grow to **2 000 clips (≈ 11 h)** stratified by
  `clip_index` attributes (country, scenario tags, lighting); publish the achieved distribution in
  the data card.
- **Stage R2 — multi-view (weeks 3–4):** **500 clips with front+left+right+rear** for the G0.7
  modality-steering demo (H2); radar comes with them for Phase 1.
- Storage: off-Drive at `C:\Users\Admin\tanitad-data\physicalai\` locally; `/workspace/data/` on pods.

## 3. Training composition (Phase 0 targets, revisit after first mixed run)

- **~60 % PhysicalAI-AV urban** (R0→R1) — the semantic backbone
- **~25 % comma2k19 highway** — real-CAN grounding + the regime where inverse dynamics is cleanest
- **~15 % MetaDrive perturbation/occluder** once the sim arm is live (D-010; bake-off-gated)
- Validation: held-out routes/sessions **per corpus, real-only**; public numbers **comma2k19-only**
  until the license issue is resolved.
- **Semantic-coverage audit (new standing metric):** per corpus and per training mix, report the
  scenario-tag distribution (intersections, pedestrians, lights, night, rain, …) from `clip_index`
  metadata — the data card must show that "rich" is measured, not asserted (DataEng agent owns it).

## 4. License management (the "solve later" plan, made concrete)

1. **Tagging (now):** every experiment record lists its corpora (`data:` tags); LEADERBOARD rows
   carry the tag so exposure is auditable in one grep.
2. **Firewall (now):** public claims/demos/publications use comma2k19 + MetaDrive (+ own data) only.
3. **Resolution paths (decide at Phase-0 exit):** (a) seek NVIDIA permission/partnership — the
   Alpamayo ecosystem exists to be built on and internal-dev use appears aligned with its intent;
   (b) replicate headline results on open corpora (OpenDV/BDD100K via H7 pseudo-labels + own data);
   (c) commission own urban data collection (GoPro rig, Phase 1/2). Costs and lead times to be
   assessed by the DataEng agent before the Phase-0 report.

## 5. Flywheel outlook (Phase 1+)

comma2k19-trained inverse dynamics (H7) pseudo-labels OpenDV/YouTube/GoPro → data-efficiency slope
experiment (C2 headline) → continual-learning loop (H10) writes surprise episodes back into training.
