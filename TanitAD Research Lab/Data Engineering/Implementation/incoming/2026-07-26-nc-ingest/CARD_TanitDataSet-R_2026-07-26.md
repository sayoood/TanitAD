---
license: mit
task_categories:
- robotics
tags:
- autonomous-driving
- world-model
- tanitad
- ego-driving
- camera
- webdataset
size_categories:
- n<1K
extra_gated_prompt: >-
  TanitDataSet-R is the research tier of TanitDataSet. This public mirror carries
  ONLY the redistributable (ship-tier) subset of R — today that subset is 100%
  comma2k19 (MIT), byte-identical to TanitDataSet-C. Non-commercial records, if
  and when they are ingested, are internal-only and will never appear here.
  Access is request-gated so the maintainer can see who is using it and notify
  consumers of corrections; the gate is an access log, NOT an additional license
  restriction — your rights are the upstream ones.
extra_gated_fields:
  Name: text
  Affiliation: text
  Intended use: text
---

# TanitDataSet-R — the research tier (public redistributable mirror)

> # ⚠️ Read this first
>
> **This repo is currently byte-identical to [TanitDataSet-C](https://huggingface.co/datasets/Sayood/TanitDataSet-C).**
>
> **R contributes ZERO additional records today.** Same 90 episodes, same 14
> shards, same `sha256`s, same 72/18 split, same single source (`comma2k19`, MIT).
> `R = C ∪ NC` is the *design*; today `NC = ∅` on the build box, because **no
> non-commercial source is downloaded or adapted yet**.
>
> **If you already have TanitDataSet-C, downloading this repo gains you nothing.**
> It exists so the research tier has a stable, correctly-scoped home to grow into —
> not because it currently holds more data.

**Status of the non-commercial tier — updated 2026-07-26.** Still **0 records**.
The blocker is not engineering: the nuScenes ingest path is now written and tested
end-to-end. It is **acquisition** — nuScenes requires a human to hold an account and
accept its Terms of Use, which is not something automation can do on a maintainer's
behalf. See *NC tier: what changed, and what still blocks it* below. Nothing on
this page implies value-add that does not exist.

---

## What R is *designed* to be, and what it *is*

| | designed | **as of this release (MEASURED)** |
|---|---|---|
| definition | `R = C ∪ NC` — one build, one schema, per-record `tier` stamp | ✅ machinery live |
| records | C's permissive records **+** non-commercial corpora | **90 — all of them C's** |
| NC records | nuScenes, BDD100K, A2D2, Argoverse 2, KITTI-360, ONCE, DRAMA, Rank2Tell | **0** — registered, none downloaded |
| NC value-add over C | maximum scene diversity, HD maps, semantic layers | **0 records, 0 GB, 0 %** |
| redistribution | C-subset ✅ · NC-subset ❌ never | this mirror is **100 % C-subset** |

### What this public mirror will and will not ever contain

`TanitDataSet-R` as an *internal* view may hold non-commercial records. **This
public repo carries only the redistributable subset of that view** — the export is
guarded by a hard scope assertion (`allowed_classes={"owned-safe"}`,
`require_commercial_ok=True`, `forbid_share_alike=True`) that refuses egress if a
single row falls outside it.

So the honest statement of what this repo is:

> **`Sayood/TanitDataSet-R` = the ship-tier subset of R.**
> Today the ship-tier subset *is* all of R, which *is* C.
> When non-commercial records land internally, **they do not land here** — this
> mirror stays the ship-tier subset, and the gap between this repo and internal R
> becomes real rather than zero.

Never re-push R wholesale. The guard enforces this in code; this note records
*why* the guard exists.

---

## NC tier: what changed 2026-07-26, and what still blocks it

**A licence correction, which matters to anyone planning around nuScenes.**
Our source registry recorded nuScenes as `CC-BY-NC-4.0` with `share_alike=False`.
**That was wrong.** nuScenes is **CC BY-NC-**SA**-4.0** — *ShareAlike*, i.e. copyleft.
Confirmed against the dataset authors' own paper
([arXiv 1903.11027](https://arxiv.org/abs/1903.11027): *"The nuScenes data is
published under CC BY-NC-SA 4.0 license…"*). The registry is fixed, and the
consequence is structural, not cosmetic:

- nuScenes routes to a **segregated copyleft shard** and may never be co-mingled
  with non-ShareAlike data;
- it can **never** enter TanitDataSet-C or any commercial artifact;
- **derivatives inherit NC + SA** — a model or label set built on it is itself NC/SA.

Two related notes for anyone reading licences in this space:
- the nuScenes **devkit code** is Apache-2.0, but **the data is not** — do not
  infer the data licence from the repo licence;
- the AWS Open Data registry entry for the nuScenes bucket carries a licence field
  reading *"Commercial"*. **That tag contradicts nuscenes.org and should not be relied on.**

**What is now ready.** A metadata-first nuScenes adapter (episodes, ego pose, 3D
agent tracks with instance ids, per-sample camera calibration, and cross-camera
frustum visibility), wired into the same guarded ingest path as every other source,
with 22 tests green against a schema-faithful fixture. Camera geometry is handled:
nuScenes CAM_FRONT (`fx≈1266` on 1600×900) is not square-croppable to our canonical
`f_eff=266` — the naive crop lands ≈360 — so it is rectified to exactly 266 with an
explicit unobserved-periphery mask rather than a silent 1.35× zoom.

**What still blocks it.** Acquisition requires a human account and Terms-of-Use
acceptance at nuscenes.org. Until a maintainer does that, **no nuScenes byte exists
on the build box, and this repo does not change.** We deliberately did **not**
substitute a third-party mirror.

---

## Contents (identical to TanitDataSet-C)

| | |
|---|---|
| **Episodes** | **90** (train 72 · val 18) |
| **Sources** | `comma2k19` (MIT) × 90 — **100 %** |
| **License classes present** | `owned-safe` × 90 |
| **Non-commercial (`nc-research`) rows** | **0** |
| **Gated / `refuse` / share-alike rows** | **0 / 0 / 0** |
| **Shards** | 14 tar (11 train + 3 val) |
| **Total size** | 15.93 GB |
| **Frame format** | `uint8 [T, 9, 256, 256]` |

### Record schema

- **`frames`** — `uint8 [T, 9, 256, 256]` — 3-frame RGB stack (9 = 3×RGB),
  canonicalized to `f_eff ≈ 266 px`.
- **`actions`** — `f32 [T, 2]` — `(steer, accel)`.
- **`poses`** — `f32 [T, 4]` — `(x, y, yaw, v)`.
- per-episode metadata incl. `sha256` of the frame blob and `build_params_hash`.

Shard layout, tar member convention, Parquet catalog and the loading snippet are
identical to TanitDataSet-C — see [its card](https://huggingface.co/datasets/Sayood/TanitDataSet-C)
for the full description and the `tarfile` / `pyarrow` examples.

---

## Provenance & verification

Re-verified **2026-07-26** over the actual payload bytes: **14/14 shards**,
**90 episodes**, **90/90 `sha256` re-verified** against the recorded frame-blob
digests, 0 duplicate ids, 0 train/val id overlap, 0 split/path mismatches,
catalog ↔ payload bijection exact. Distinct source corpora in the payload:
`{comma2k19: 90}` — **nothing else**. Distinct license classes: `{owned-safe: 90}`.
Gated / `refuse` / `nc-research` / share-alike rows: **0 / 0 / 0 / 0**.
Machine-readable in `LICENSE_VERIFICATION.json`.

---

## Honest limits

Everything in [TanitDataSet-C's honest-limits section](https://huggingface.co/datasets/Sayood/TanitDataSet-C#honest-limits--read-this-before-you-plan-around-it)
applies verbatim, because this is the same data. In particular:

1. **R adds nothing to C today** — 0 non-commercial records. The headline above is
   the single most important fact about this repo.
2. **90 episodes is seed-scale**, one source, US highway, forward camera only.
   No urban, intersection, VRU, night or adverse-weather coverage. **No map, no
   lane graph, no route topology, no traffic lights, no surround cameras.**
3. **L2D contributed 0 records** — no LeRobot-v3 → 9-channel adapter exists yet
   (~2–3 engineering days).
4. **PhysicalAI-AV is excluded and always will be**, from *both* tiers. It is
   `gated-confidential` — NVIDIA-gated, not redistributable, firewalled and
   recipe-only. The ingestor raises `PermissionError` on it, so it cannot become a
   record on any tier, research included. Nothing here is derived from it.
5. **Waymo Open / WOD-E2E and Waymax are refused outright** — a distinct `refuse`
   license class, worse than non-commercial, because their terms follow the
   trained weights into the model and vehicle operation. They never enter the lake
   on *any* tier, research included.
6. **Split is episode-level, not route-disjoint**; comma2k19 is one commute route
   re-driven, so train and val can share road segments.
7. **No semantic / VLM labels** in this release — frames, actions, poses and
   provenance only.
8. **Anonymization is inherited, not re-applied** — upstream comma2k19 MIT
   footage, nothing added.

---

## Which repo should I use?

| you want | use |
|---|---|
| commercially-clean data you may ship | **[TanitDataSet-C](https://huggingface.co/datasets/Sayood/TanitDataSet-C)** |
| maximum research diversity | **C, for now** — R has nothing extra yet |
| to track the research tier as it grows | watch this repo |

## Citation / attribution

Cite the upstream source; per-source license text ships in `NOTICE`.

```bibtex
@article{schafer2018commute,
  title  = {A Commute in Data: The comma2k19 Dataset},
  author = {Schafer, Harald and Santana, Eder and Haden, Andrew and Biasini, Riccardo},
  journal= {arXiv preprint arXiv:1812.05752},
  year   = {2018}
}
```

_Built by the TanitAD Phase-A lake pipeline. Provenance travels with the data._
