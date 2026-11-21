# Semantic / strategic-label dataset survey — sourcing REF-B's nav-command supervision

**Agent:** Data Engineering (Tuesday) · **Date:** 2026-07-11 (pm run) · **Branch:** `agent/data-engineering-20260711`
**Quality:** full (G-A…G-C, G-E, G-H, G-D1). Loop: 1 iteration, 8 web searches + 1 measured HF probe, ~1.1 h, $0.

## 0. Why this run (Sayed directive 2026-07-11, from the REF-B review)

REF-B rev2 (`38cf9ca`) added a **strategic transformer** with **route-derived nav commands**, but the
review flagged the supervision problem: **comma2k19 is highway-dominated — its route-geometry
pseudo-labels are ~all `follow`**, so the strategic/behavior layer trains on a near-constant signal.
Sayed's directive (backlog P1 #2d): *survey + rank datasets with RICH semantic strategic/behavior
labels* for Phase-1 strategic/tactical training AND for richer pseudo-label validation, and recommend
**one** Phase-1 ingest. Pod is off-limits (30k record run live) → this is a web + HF-API run, no GPU.
This complements this morning's focal-invariance run (`7588330`) — different work item, same day.

## 1. What "strategic label" means for us (taxonomy axis)

The strategic layer needs supervision on the **route/mission → maneuver → intention** hierarchy, not
perception boxes. I score each corpus on a **label-depth ladder**:

| Depth | Signal | Example | Use to us |
|---|---|---|---|
| L0 | geometry-only pseudo-label | curvature→{left,straight,right} | comma2k19 today (starved: ~all straight) |
| L1 | **nav command** | "turn right at the intersection", "exit roundabout, 1st exit" | strategic head target |
| L2 | **maneuver / behavior** | lane-change, U-turn, yield, creep-to-stop | tactical head target |
| L3 | **intention / free text / QA** | "slowing because the lead braked; child at curb" | pseudo-label *validation*, H12 bridge |

A corpus is valuable to REF-B iff it carries **L1+ labels co-registered with camera + ego actions**
(so the label supervises a policy, not just a caption). License class governs whether numbers can go
public (firewall: comma2k19 + Cosmos-DD only, per DATA_STRATEGY §4).

## 2. MEASURED experiment (G-H) — L2D taxonomy probe on real HF bytes

Top candidate from the sweep is **L2D** (`yaak-ai/L2D`, HF, LeRobot format). Rather than trust the card,
I probed the actual dataset sidecars + one data shard over the HF API (truststore; `meta/info.json`,
`meta/tasks.jsonl`, one `data/chunk-000/file-000.parquet` — no clone of the 90 TB corpus).
Tool: `Implementation/incoming/2026-07-11-semantic-label-survey/probe_l2d_taxonomy.py`; raw result
`l2d_taxonomy_result.json`. Hardware: dev box, network-only, **$0, ~3 min**.

**Measured facts (real bytes, not the card):**
- **Scale:** 100,000 episodes · **26,466,954 frames** @ 10 fps · vehicle = KIA Niro EV 2023 (single rig).
- **Cameras:** 6× surround RGB `3×1080×1920` (`front_left`, `left/right_forward`, `left/right_backward`,
  `rear`) + a rendered `map` view `3×360×640`. **No dedicated narrow front** — `front_left` is the front proxy.
- **Ego actions PRESENT and real:** `action.continuous` **dim 3**, `action.discrete` **dim 2** — the labeled
  bridge REF-B's tactical head needs (no IDM required, unlike BDD100K/OpenDV).
- **Strategic labels = L1, compositional, real:** **4,219 distinct task instructions**; every frame carries
  `task.instructions` + `task_index` + `task.policy` (EXPERT). Of the 4,219:
  - **96 %** (4,039) carry an explicit **distance** token ("for 0.7 km", "150 m"),
  - **74 %** (3,133) carry a **speed-limit** token ("observe the speed limit of 70 km/h"),
  - **61 %** (2,577) carry a **road-class** token (motorway/secondary/primary/residential/roundabout).
  - Verbatim samples: *"Make a U-turn and then go straight for 50 m."* · *"Go straight on the residential
    road for 150 m … and turn right at the intersection following the right before left rule …"* ·
    *"exit the roundabout using the first exit"* · *"reverse out and then go straight for 10 m"*.
- **Route geometry:** `observation.state.waypoints` **dim 10** (5 (x,y) pairs) per frame — a second,
  denser strategic supervision channel than the language.
- **Scene state:** per-frame `observation.state.{lanes,road,surface,max_speed,precipitation,conditions,
  lighting}` — free tactical/OOD conditioning + D8 stratifiers.
- **License:** **Apache-2.0** → **public-claimable** (only the 3rd such AV corpus after comma2k19/MIT and
  Cosmos-DD/CC-BY-4.0), and the first *real-world* one with L1 labels.

**Verdict.** L2D is a **direct fix** for the REF-B supervision gap: real expert driving, 30 German cities,
L1 nav commands **co-registered with ego actions + waypoints + camera**, Apache-2.0. It gives the
strategic head a **4,219-way** compositional command distribution where comma gives ~1.

**Honest caveats (P8).** (a) The instructions are **templated from map routing** (the "right before left"
phrasing is the German *Rechts-vor-links* rule), not free human narration — they are L1/L2, not L3
intention. That is exactly what the strategic/tactical heads want, but it is *not* a substitute for an
L3 reasoning corpus. (b) `front_left ≠` a canonical narrow front → **D-016 focal canonicalization is
mandatory** at ingest (this run's morning result: wrong intrinsics = ~10–15× encoder drift), and L2D
publishes a fixed KIA rig so per-clip intrinsics are constant/known. (c) `action.continuous` is dim 3
(steer/accel/brake or throttle/brake/steer) → needs a **3→2 mapping** to our `[steer, accel]` contract;
sign/units to be pinned on a real decode (the contract-map stub in the intake package pins the plan).
(d) 90 TB total → Phase-1 ingest must **stream a filtered slice**, not clone.

## 3. Ranked survey (G-D1: license · size · actions · label depth · cost-to-batch)

Score = (label depth L1+ × co-registered actions × camera compat) ÷ ingest cost, license as a public-claim
*tag*. All rows are **eval-or-train** classified for REF-B.

| # | Corpus | Label depth | Actions co-reg? | License (claim) | Size | Cost→batch | REF-B role |
|---|---|---|---|---|---|---|---|
| **1** | **L2D** (`yaak-ai/L2D`) | **L1+L2** (4,219 nav cmds, waypoints, U-turn/roundabout/lane prims) | **yes** (cont-3 + disc-2, real) | **Apache-2.0 (PUBLIC)** | 100k eps / 26.5 M fr / 90 TB | ~4–6 h (LeRobot parquet + video + D-016) | **TRAIN — the recommendation** |
| 2 | **nuPlan** (`motional`) | L1 (mission goal + route centerline, map-derived) | yes (ego + tracks) | academic-free / commercial-lic | 1,282 h / 16 TB subset | ~6 h | TRAIN/EVAL — strongest *route* signal, heavy, planning-centric |
| 3 | **CoVLA** (`turingmotors`, 2408.10845) | L2+L3 (frame captions of maneuvers + trajectories) | **yes** (actions + traj) | academic-only (NC) | 10k clips / 80 h | ~4 h | EVAL / pseudo-label validation (NC → no public claim) |
| 4 | **Bench2Drive** (`Thinklab-SJTU`) | L2+L3 (full-stack planning/behavior VQA) | yes (CARLA sim) | **Apache-2.0 (PUBLIC)** | 1k–10k clips (CARLA) | ~4 h (sim) | EVAL closed-loop — public, but sim; pairs with our CARLA arm |
| 5 | **DriveLM** (`OpenDriveLab`, ECCV24) | **L3** (graph VQA: perception→pred→plan) | via nuScenes/CARLA | code Apache-2.0 / **text CC-BY-NC-SA** | 196k keyframes / 5,134 CARLA routes | ~5 h | EVAL reasoning — richest L3, but NC text |
| 6 | **CoVLA/Talk2Car** (`KU Leuven`) | L1 (object-referral commands) | via nuScenes | nuScenes-derived (NC) | 11,959 cmds / 850 vids | ~4 h | EVAL command-grounding (H12), NC |
| 7 | **AUTOPILOT-VQA** (2607.08745) | L3 (behavior taxonomy) | — | (fresh, unverified) | — | — | **Benchmarks & Eval owns** (D-028 seam) — behavior taxonomy for our probe suite |
| — | **Intention-Drive** (2512.12302, new) | **L1→L3 hierarchy** (atomic cmd → abstract human intention) | benchmark | (2026 release) | — | — | WATCH — mirrors REF-B's strategic→intention ladder exactly |

## 4. Recommendation (G-B) — ONE Phase-1 ingest: **L2D, filtered streaming slice**

Ingest **L2D** as the Phase-1 strategic/tactical supervision corpus. It is the only surveyed corpus that
is simultaneously **(a) real-world, (b) L1/L2-labeled, (c) action-co-registered, (d) camera-present, and
(e) public-claimable (Apache-2.0)** — it fixes the exact comma-is-all-`follow` problem the REF-B review
raised, with numbers we can publish. Concretely:
1. **Phase-1 slice, not clone:** stream episodes whose `task.instructions` contain a **turn / roundabout /
   lane-change / U-turn** primitive (the non-`follow` tail comma lacks) — target ~2,000 episodes first,
   balanced across the 4,219 task classes and the `road`/`lighting` state. Bandwidth-bounded, ~tens of GB.
2. **Loader = Cosmos-mirror + D-016:** reuse the pose/contract code; `front_left` → `focal_crop_resize`
   to `F_REF=266` with the fixed KIA intrinsics; `CORPUS_META` byte-identical (D-017 I7) so it is
   admissible in the D-010 mix; map `action.continuous[3]→[steer,accel]` (sign/unit pinned on decode).
3. **Two supervision channels for REF-B:** (i) `task.instructions` → strategic-head nav-command target
   (replaces the geometry pseudo-label); (ii) `waypoints[10]` → route-conditioning / auxiliary.
4. **Validation use even before training:** decode L2D nav commands vs comma's route-geometry pseudo-labels
   → **measure the label-entropy gap** (comma ~1 effective class vs L2D 4,219) to quantify REF-B's
   starvation numerically. This is the cheap first experiment once the loader stub lands.

**Falsifier for the recommendation:** if a 200-episode decode shows `action.continuous` cannot be mapped
to a physically-sane `[steer,accel]` (sign/unit ambiguity unresolved) OR `front_left` FOV after D-016 lands
> ±25 % off `F_REF` (cos < 0.92 by this morning's curve), L2D drops to EVAL-only and nuPlan (#2) becomes
the train recommendation. Both are pre-registered in the backlog.

## 5. Literature / recency scan (D-028 mandatory)

- **Intention-Drive / "From Human Intention to Action Prediction"** (arXiv 2512.12302, Jan 2026): a
  benchmark with **hierarchical NL instructions from atomic commands to abstract human intentions** — the
  same strategic→intention ladder REF-B builds. WATCH as the eval target for the strategic head; not an
  ingestible training corpus yet. https://arxiv.org/abs/2512.12302
- **"Unveiling the Surprising Efficacy of Navigation Understanding in E2E AD"** (2604.12208): nav-command
  conditioning materially moves closed-loop scores → external support that REF-B's nav-command channel is
  worth the parameters.
- **AUTOPILOT-VQA** (2607.08745, Sayed-delivered): behavior taxonomy — routed to **Benchmarks & Eval**
  (D-028 benchmark seam); its taxonomy should define our strategic-label *class set* — flagged to them.
- No hypothesis status change (P8); external support for H7 (heterogeneous-video flywheel) and H12
  (command conditioning).

## 6. Artifacts

- Research note (this file).
- Intake pkg `Implementation/incoming/2026-07-11-semantic-label-survey/`: `probe_l2d_taxonomy.py`
  (measured, ran on real HF bytes), `l2d_taxonomy_result.json` (raw result), `l2d_contract_map.py`
  (the 3→2 action + front-cam + instruction→class mapping spec), `tests/test_l2d_contract_map.py`
  (offline, synthetic-row, **standalone-green**), `INTAKE.md`.
- `DATASET_LANDSCAPE.md`: new **Tier 1.5 — semantic/strategic-label corpora**; L2D row promoted with
  measured taxonomy. `KNOWLEDGE_BASE.md`, `BACKLOG.md`, `HYPOTHESIS_LEDGER.md` updated.
