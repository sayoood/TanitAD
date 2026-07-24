# TanitDataSet — integrating the 2026-07-21 source research into the two-tier build

*Amends `TANITDATASET_V1_STRATEGY.md` rev-3. Does not replace it: the tier machinery, the schema
firewall and the VLM-engine choices all stand. This adds the sources and licence facts measured on
2026-07-21, one **new** enforcement class, and the augmentation strategy Sayed asked for.*

**Status:** design, staged. **Not implemented.** ⚠️ Background agents unavailable until the weekly API
limit resets (Jul 26, 00:00 Berlin), so this was written directly.

---

## 1. What rev-3 already has (unchanged, and it is the right frame)

- **R = C ∪ NC**: one build, one schema, a per-record `tier` stamp decides what may ship.
- Tiers: **`ship`** (`commercial_ok`) · **`ship-sa`** (owned-safe **and** share-alike → segregated
  copyleft shard, never co-mingled) · **`nc`**.
- **The firewall is mechanical, not policy**: `license_class` comes from a `SOURCE_REGISTRY`
  CONSTANT and is *never inferred*; `assemble_lake_record` **raises `PermissionError`** if a
  `gated-confidential` source reaches the lake.

## 2. ⛔ NEW enforcement class: `refuse` — three tiers were not enough

`nc` means *"usable for research, not shippable."* Two sources are **worse than that** and must never
be ingested at all, on any tier:

| source | why | class |
|---|---|---|
| **Waymax** | §2.e bars using it to train **any foundation model** | 🔴 `refuse` |
| **Waymo Open / WOD-E2E** | terms **follow the trained weights into vehicle operation** — the only licence surveyed that reaches our *end product*, not just our data | 🔴 `refuse` |

**Everything else in the survey constrains the DATA. These constrain the MODEL.** An `nc` tier cannot
contain that, because the contamination survives training. `SOURCE_REGISTRY` needs a fourth
`license_class` whose handler raises on ingest, exactly as `gated-confidential` does today.

Also `refuse`-adjacent, pending human review: **TLD** (gated, **no licence file**, provenance is
scraped YouTube + Honda LOKI/NC) and **LISA Vehicle Lights** (bare Google Drive, no terms).
**Unverified ≠ permissive.**

## 3. The linking mechanism — sources join through the SLOT SCHEMA, not through geography

These corpora share no roads, so they cannot be joined spatially. **They join through the frozen
vocabulary** (114 tokens / 18 slots): each source populates whichever slots it can, and a record is
the union with per-slot provenance. That is already the schema; what was missing is knowing *which
source fills which slot*, i.e. which of the three planner layers each one can actually supervise.

| source | tier | **strategic** (route / map / speed limit) | **tactical** (maneuver · indicator · agents) | **operative** (ego traj) |
|---|---|---|---|---|
| ⭐ **L2D** (`yaak-ai`) | `ship` | ✅ OSM class, lane count, **posted speed limit**, snapped waypoints; NL instruction **with metric distance on 96.74 %**; roundabout on **3,532 eps** | ✅ **ego blinker from CAN** — 18.36 % of frames, **79.9 % direction-predictive at 5.5–6.7 s lead** | ✅ + **drives reconstruct to p90 346 s** |
| **commaCarSegments** | `ship` (MIT) | ✗ | ✅ blinker via opendbc decode — **3,148 h / 188,883 segments** | ✅ |
| **comma2k19** | `ship` (MIT) | ✗ | ✅ blinker via opendbc (already in our lake) | ✅ |
| **ZOD** | `ship-sa` | ✗ | ✅ `turn_indicator_status` **100 Hz** — cleanest commercial ego-indicator found | ✅ |
| nuScenes (+ QA economy) | `nc` | ✅ map | ✅ `left_signal`/`right_signal` (CAN exp, 2 Hz) · agents | ✅ |
| nuPlan · Argoverse 2 | `nc` | ✅ HD map | ✅ agents | ✅ |
| CoVLA | `nc` | ✗ | ✅ `leftBlinker`/`rightBlinker`, 16.11 % of frames | ✅ |
| PREVENTION · VRSD | `nc` | ✗ | ✅ blinker/taillight *of observed vehicles* (VRSD host **dead**) | — |
| **PhysicalAI-AV** | 🔒 `gated` | ✗ **no geo exists** — `country` is the finest location | ✅ **`obstacle.offline`: 3D tracks on 96.90 % of our corpus** | ✅ (parity corpus) |
| Waymax · Waymo Open | 🔴 `refuse` | — | — | — |

**Two structural readings.**
1. **The commercial tier is now viable for all three layers.** L2D alone supplies strategic + tactical
   + operative under Apache-2.0; ZOD and comma add indicator volume. Before today, `C` had no map and
   no indicator source at all.
2. **Our own corpus is the *weakest* strategically and the *strongest* tactically** — no geo ever, but
   agent tracks on 96.90 % of it that we never read. PhysicalAI stays `gated`: usable for our internal
   models, never shippable, never publishable.

## 4. ⚠️ The rule that governs augmentation: **a derivative inherits the strictest input tier**

Synthetic augmentation does **not** launder a licence. This has to be enforced in code, not remembered:

```
tier(derivative) = strictest( tier(source_record), tier(generator_model), tier(conditioning_labels) )
```

Concretely: **Cosmos-Drive-Dreams renders of PhysicalAI clips are still `gated`.** We hold such
renders (`tanitad-eval:/root/vlm_pilot/frames/*_{Rainy,Foggy,Snowy,Golden_hour}`) and they are **not**
shippable, however synthetic they look. Same for any counterfactual rollout conditioned on `nc` maps.

## 5. Augmentation strategy — ranked by (value × licence-cleanliness) ÷ cost

| # | augmentation | what it buys | tier behaviour | cost |
|---|---|---|---|---|
| **A1** | ⭐ **Cross-source label transfer** — train a labeler where labels EXIST, apply where they don't | The only mechanism that moves supervision *between* corpora. **L2D has map + blinker, PhysicalAI has neither; PhysicalAI has agents, L2D's are unverified.** Train on the labelled side, infer on the other | derivative label inherits the **labeler's** source tier — a `ship` labeler produces `ship` labels | eng-days, no GPU |
| **A2** | **Temporal restride / drive-level windows** | Breaks the 20 s ceiling on L2D at **zero** generation cost — 90.8 % of episodes overlap; de-dup is a `groupby` on drives | unchanged | ~1 eng-day |
| **A3** | **Rare-stratum up-sampling** (rev-3 §7 already specifies this) | Directly targets measured weaknesses: 639/881 steady-cruise windows lose to hold-v0; roundabouts 8 windows→3,532 eps via L2D | unchanged | free |
| **A4** | **Photometric re-render** (weather/night/fog) | Robustness. ⚠️ **Our own OOD numbers are unimpressive**: cosmos **29.4 %** win-rate vs **49.7 %** in-distribution | inherits strictest | H100-class |
| **A5** | ⭐ **Counterfactual rollout** — same past, different ego action | The only thing that attacks the **~92 % aleatoric** oracle bound, because real data gives exactly ONE future per state and an action-conditioned WM wants many | inherits strictest | UNRESOLVED — the investigation died on the API limit |

**A1 and A2 are the ones to build first**: highest value, zero licence risk, no GPU, and neither
depends on a generator. A4/A5 remain open — and A5 is the one worth reviving on Jul 26.

## 6. What to do, in order

1. **Add `license_class = refuse`** + the ingest-time raise. Encode Waymax and Waymo Open. ~0.5 eng-day.
2. **Ingest `obstacle.offline`** (2–3 eng-days, 0 GPU-days, 12.4 GB, **zero parity impact**) — gated
   tier, unblocks tactical agent-conditioning for our own models. Gate it on the **1-eng-day,
   $0 lead-state information test** already specified in `DATA_STRATEGY_FOR_HIERARCHY.md` §3.
3. **L2D pilot** (3–5 eng-days, ~0.5 GPU-day) — the first source that makes tier `C` complete across
   all three layers. ⚠️ Two traps: **no camera intrinsics ship anywhere** (risk to our `f_eff = 266`
   assertion) and the sliding-window overlap **double-counts ~50 %** unless split on drives.
4. **A1 label transfer**, once 2 and 3 exist.
5. Revisit A4/A5 after Jul 26.

## 7. Corrections to the record

- The 2026-07-19 sensor survey says **nuScenes has no blinker — it does** (`vehicle_monitor.left_signal`/
  `right_signal` in the CAN expansion).
- **PhysicalAI has no geo of any kind** — NVIDIA's card states it, and `data_collection.parquet` has
  five columns of which `country` is the finest location. The GPS→OSM map-matching path is closed for
  the parity corpus.
- ⚠️ **The "our pods cannot render" conclusion (2026-07-09) is in doubt** — the eval pod was observed
  to carry Vulkan ICD + EGL vendor files. Needs a functional check; it gates CARLA, the only source of
  *other agents'* indicator state.
