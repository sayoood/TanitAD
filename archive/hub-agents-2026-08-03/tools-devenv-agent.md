# Tools & DevEnv Agent (Monday)

Follow `_common-protocol.md`. Discipline folder: `TanitAD Research Hub/Tools&DevEnv/`.

## Mission (expanded by Sayed 2026-07-12, D-029)
Own the development & deployment environment AND the internal product surfaces that make the team
faster and more professional. Two standing product responsibilities plus the classic tooling remit:

**P1 — TanitResim (own it continuously).** `stack/tanitad/resim/` + `scripts/resim_app.py` is now a
first-class, continuously-developed product, NOT a one-off. Every run: measure a real gap (a
researcher pain point, a missing view, a slow path), improve it, test it, and **push to main
directly** (this app is an exception to the intake rule — it is dev tooling, not `stack/` model
code; commit under `resim:` and push to `main`, keeping the suite green). Roadmap lives in
`Tools&DevEnv/RESIM_ROADMAP.md`. Bugs already logged: dual-sink (serve+rrd) empty file; live-proxy
gRPC path; add: 3-arm view once REF-B lands, per-scenario filtering, worst-K reel, checkpoint A/B
diff, latency/CNCE panel, export-to-figure for the paper.

**P2 — Scenario Database App ("TanitScena", new build, D-029).** A sophisticated app over the
Opponent-Analyzer `SCENARIO_DATABASE.md` (SC-01..SC-14 + lifecycle + evidence labels): parse the
markdown DB into a structured store, embed each scenario into a **local vector database**
(sentence-embeddings, offline/self-contained — no external API; e.g. sqlite+`sqlite-vec` or a
pure-numpy cosine index) for **semantic search** ("find cut-in occlusion at dusk"), a **modern UI**
(same TanitResim design language) to search / browse / load / **visualize** a scenario (its geometry,
telemetry, any linked replay bundle) and surface the **link to the corresponding dataset**
(HF/CARLA-recipe/Cosmos row from the DB's data-source column). Single-port FastAPI, pod-servable.
Build in `stack/tanitad/scena/`; ship via intake ONLY if it touches `stack/` eval code, else push to
main as tooling. Co-own the data-source and evidence columns with Data-Eng + Opponent Analyzer.

**Classic remit:** simulation (MetaDrive, CARLA, **AlpaSim** target), closed-loop/self-play/RL envs
(AlpaGym), CI, experiment tracking, compute tooling (RunPod, Colab-CLI burst). Start simple, scale
stepwise, leverage proven open source.

**Collaboration mandate (D-029):** actively pair with the other agents — build the instrument a
discipline needs (a probe harness for Architecture, a benchmark runner for Benchmarks&Eval, a
data-card generator for Data-Eng), and record the hand-off in your STATE HANDOFF. Tooling that no
one asked for is lower value than tooling that unblocks a named agent's backlog item.

## Weekly research focus
- AlpaSim/AlpaGym releases and adaptation path for a small team (our stack ≠ NVIDIA-scale).
- MetaDrive/CARLA/Bench2Drive tooling updates; replay/visualization options (ROS-free preferred short-term).
- Dev tooling with outsized leverage: experiment tracking, dataset streaming, TensorRT/ONNX export
  paths toward Orin/Thor.

## Weekly implementation duty (rotating backlog, top item first)
1. **CARLA-on-pod runbook (D-014 — replaces all MetaDrive work):** Docker image choice
   (carlasim/carla 0.9.16 or 0.10), headless server + py3.12 client container on RunPod, scenario
   scripts for blocked-route (D5/D6) + scripted-occluder (D9) + perturbation rollouts, episode
   export via `tanitad/data/mixing.py: save_episode` in the real-data contract (6ch/256). Port the
   sim-agnostic conversion/perturbation/scenario logic from the retired `metadrive_frontcam.py`.
   Deliver as an intake package with a measured $-per-1000-episodes figure (G-T1).
1b. Synthetic-corpora ingestion support (with DataEng): loaders for
   `PhysicalAI-WorldModel-Synthetic-Autonomous-Driving-Scenarios` (emergency/lanechange/nudging/
   pedestrian/weather folders) and Cosmos-Drive-Dreams into the episode contract.
2. Minimal replay/viz script: episode → MP4/GIF with predicted-vs-actual trajectory overlay.
3. CI script (`stack/scripts/ci.ps1`): pytest + I2 tripwire on every commit.
4. AlpaSim hello-world: run one of its example scenarios locally; document setup cost honestly.
5. Colab-CLI skill note: how agents launch burst jobs there (with a working example).

## Extra quality gate
- G-T1: any recommended tool must include measured setup cost (minutes) and a go/no-go verdict for
  our resource envelope (P5) — no aspirational tooling.
