# Data Engineering — 2026-07-14 — Cosmos-Drive-Dreams loader + AV landscape sweep

> Weekly Tuesday agent. Consumed Monday's Tools&DevEnv note (2026-07-13, MetaDrive front-cam — now
> superseded by D-014's synthetic-corpora pivot). Budget used: 4 web searches + 3 fetches, 1 loop
> iteration — well under the 25-search / 3-iteration / 2-h caps.

---

## 1. The gap this run closes (why a Cosmos loader, why now)

D-012 put PhysicalAI-AV into the pipeline "use now, license later". My own **2026-07-07 license
review (D-002)** then found the *real* PhysicalAI-AV sets are NVIDIA-AV-licence: internal-dev-only,
confidential, 12-month expiry → **excluded from every public claim**. **D-014** split the sim arm and
named the two ungated NVIDIA synthetic corpora as the training-mix data source. Of those,
**Cosmos-Drive-Dreams is CC-BY-4.0** — per the license review, *the* one AV asset we may render, train
on, **and cite publicly**. Yet it had **no loader**. So the public-claims story had a real anchor
(comma2k19, MIT) with honest coverage limits (highway commute) and **no rich public corpus**. This run
ships the loader, making the publicly-claimable corpus a first-class citizen of the mix. — impact:
D-014 / D-002 firewall / H7 / H4 — `stack/tanitad/data/comma2k19.py` license row, DATA_STRATEGY §4.

## 2. The corpus, grounded (RDS-HQ format)

`nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams` — 5 843 RDS-HQ 10-s clips + **81 802**
Cosmos-generated synthetic videos across **7 weathers** (Foggy / Golden_hour / Morning / Night / Rainy
/ Snowy / Sunny), the long-tail comma2k19 lacks. On disk (fields the loader needs): synthetic RGB mp4
at **30 fps**, per-frame **4×4 `vehicle_pose`** (`.npy`, ego-to-world), `pinhole_intrinsic` `[fx,fy,cx,
cy,w,h]`, HD map + LiDAR (unused for now). **Front camera = `front_wide_120fov`** — the *same 120° HFOV*
as PhysicalAI-AV front-wide, so D-016 canonicalization reuses the identical nominal focal and the crop
is angularly consistent with comma2k19. License **CC-BY-4.0** (confirmed on the HF card). — sources:
[HF dataset card](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams),
[cosmos-av-sample-toolkits](https://github.com/nv-tlabs/cosmos-av-sample-toolkits),
[Cosmos-Drive-Dreams paper (arXiv 2506.09042)](https://arxiv.org/abs/2506.09042).

## 3. Implementation increment (intake pkg — D-011)

`Data Engineering/Implementation/incoming/2026-07-14-cosmos-drive-dreams-loader/`
(`tanitad_cosmos_drive.py` + `tests/test_cosmos_drive.py` + `INTAKE.md`; proposed target
`stack/tanitad/data/cosmos_drive.py`). **9 tests pass (4.0 s); full stack 73✓/1s** — additive, 0 new deps.

- **No CAN here → derive from geometry.** The novel part vs comma2k19 (real CAN) / physicalai
  (egomotion parquet): `poses_to_signals(veh_to_world[N,4,4], dt)` reads yaw from the rotation block
  (stable at any speed), speed/accel from the smoothed position derivative, and steering as the
  bicycle-model road-wheel angle from path curvature `κ = yaw_rate / v`, **clipped** so a near-stationary
  frame cannot emit a spurious hard-lock (the `1/v` blow-up — unit-tested by `test_low_speed_steer_guard`).
- **Contract identity.** 30 Hz→10 Hz stride, D-015 3-frame/9-channel stacks, D-016 focal canon (120°),
  actions/poses aligned to the latest frame. `CORPUS_META` is **byte-identical to comma2k19** — a test
  asserts `cd.CORPUS_META == comma2k19.CORPUS_META` (D-017 I7 task-identity), which is *why* probes fit
  on one corpus are admissible on the other and `MixedWindowDataset` (D-010) accepts Cosmos into the
  real+sim mix (`test_admissible_in_mix`). — impact: D-014 mix arm, D-017, H4.
- **Splits CLIP-level (I3);** per-weather distinct `episode_id` so the 7 variants of one scene never
  leak across train/val.
- **Honest limitations (P8):** exact `vehicle_pose` glob + FLU/OpenCV axis order are from toolkit docs,
  **pod-verified** by `verify_real_clip()` (returns A8 + speed/steer/accel ranges for the data card)
  before any trained claim; synthetic pixels are **not** off-expert action-consequence rollouts (the
  `max_a` JEPA argument) — that job stays with CARLA-on-pod (D-014). Neither blocks integration.

## 4. AV landscape sweep (D-012 standing duty)

Created `Data Engineering/Research/DATASET_LANDSCAPE.md` (was missing) — 3 tiers, one row per corpus
with size / sensors / actions / **license class** / urban richness / **cost-to-first-batch** (G-D1).
Headlines: Tier-1 pipeline = comma2k19 (public anchor) + Cosmos-DD (public synthetic, this run) +
PhysicalAI-AV (gated, tagged) + WorldModel-Synthetic-Scenarios (candidate); Tier-2 real urban (BDD100K
→ H7 IDM, Zenseact ZOD = real-CAN #2 for H4, CoVLA → H12); Tier-3 scale-up (OpenDV-2K, source-derived,
Phase-1 flywheel). Next-sweep priorities recorded in the doc.

## 5. Literature deltas (H7 / hierarchy moat)

Systematic sweep for latent-action / inverse-dynamics pretraining since the last run surfaced a **surge**
directly on our H7 thesis — and, importantly, it strengthens the moat argument rather than the corpus:

- **Latent Action Pretraining through World Modeling (LAWM)** — self-supervised latent actions from
  **unlabeled** video via world modeling, then efficient finetune on labeled data. Direct external
  support for H7's "1000× data via IDM": our comma2k19-trained inverse dynamics is exactly the labeled
  bridge LAWM finetunes onto. — impact: H7 —
  [arXiv 2509.18428](https://arxiv.org/abs/2509.18428).
- **Drive-JEPA** — V-JEPA as the *efficient latent world model with built-in collapse prevention* for
  end-to-end driving. Same family as our H3 core (SigReg collapse-health rows); confirms JEPA-latent-WM
  is now the mainstream E2E-driving substrate → reinforces the Opponent-Analyzer finding that "world
  model" no longer differentiates; our moat is **hierarchy + efficiency + imagination + self-monitoring**.
  — impact: H3 / H7 / opponent context — [arXiv 2601.22032](https://arxiv.org/abs/2601.22032).
- **Hierarchical Latent Action Model (HiLAM, ICLR-2026 workshop)** — hierarchy applied to latent actions
  for actionless pretraining + cross-embodiment transfer. Adjacent to H1 (our hierarchy lift) meeting H7.
  — impact: H1×H7 — [arXiv 2603.05815](https://arxiv.org/abs/2603.05815).
- **CLAW** (continuous latent action WMs from unlabeled video) and **DeFI** (disentangled forward/inverse
  dynamics) — two more label-free-action mechanisms; note for the H7 IDM design (a flow/forward-consistency
  term, extending the LAOF delta from 2026-07-07). — impact: H7 —
  [CLAW arXiv 2606.04130](https://arxiv.org/abs/2606.04130).

**No hypothesis status upgraded (P8):** these are external *support*, not our measurements. H7 stays
`supported`; the binding evidence remains our own IDM steering-ratio calibration residual (named artifact,
still to be produced on real Chunk_1).

## 6. Actionable recommendations (tied to hypotheses / WPs)

1. **(Triage)** Integrate the Cosmos-DD loader → the D-010 bake-off can now run a *publicly-claimable*
   real+synthetic mix (comma2k19 + Cosmos-DD), not just real-only. Owner: MVP orchestrator.
2. **(DataEng, next)** Mirror the loader for **PhysicalAI-WorldModel-Synthetic-Scenarios** once its card
   license is verified — near-zero cost (shares pose/contract/decode code), adds the emergency/pedestrian/
   weather_degradation long-tail (H6/H15/D9 material).
3. **(Pod)** Run `verify_real_clip()` on 3 downloaded Cosmos clips → settle the FLU/OpenCV axis order +
   A8 consequence on real synthetic renders; publish the numbers in a data card. ~1 engineer-h.
4. **(H4, Bench&Eval link)** Add **Zenseact ZOD** as a second real-CAN corpus → makes H4 arm-B (frozen vs
   trained encoder) meaningful on an EU/night distribution comma2k19 lacks.

## 7. Self-critique (quality gates)

- **G-A** every §-claim carries a source link or repo path. ✅  **G-B** 4 actionable recs tied to
  D-010/H7/H4/D9. ✅  **G-C** KB updated (deltas, newest first). ✅  **G-D** ledger H7/H4 evidence
  changelog row added (no status change, P8). ✅  **G-E** 9 passing standalone tests + explicit pod next
  step (`verify_real_clip`). ✅  **G-D1** landscape rows carry license/size/actions/cost. ✅  **G-D2**
  loader ships the episode-contract test (`assert_contract(channels=9)`). ✅  **P8** synthetic≠off-expert
  and pose-convention uncertainty both stated, not hidden; no status upgraded on external evidence.
- **Gap (recorded):** real-bytes validation (`verify_real_clip`) not run in this unattended session —
  the exact `vehicle_pose` glob and axis order are asserted-by-doc, proven-by-fixture only until the pod
  step. No trained claim depends on it yet.
