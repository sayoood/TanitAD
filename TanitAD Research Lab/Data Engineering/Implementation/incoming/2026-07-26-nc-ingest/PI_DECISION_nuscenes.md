# PI decision — nuScenes ingest: **INGEST FREELY**

**Date:** 2026-07-26 · **Decided by:** Sayed (PI) · **Verbatim:** *"regarding nuScenes, ingest freely."*
**Context put to the PI:** `NC_INGEST_REPORT.md` + the three-option decision brief (A don't ingest ·
B ingest research-firewalled · C ingest freely · D legal advice first).

---

## 1. What was decided

**Option C — ingest and accept the risk.** nuScenes is **`CC-BY-NC-SA-4.0`** (copyleft; registry
corrected from a wrong `share_alike=False` — the second licence-from-short-name error after ZOD).

**The accepted risk, stated plainly so it is not rediscovered later:** it is legally unsettled whether a
world model **trained on** ShareAlike data is a **derivative work** of that data. If it is, weights trained
on nuScenes could be argued to inherit **CC-BY-NC-SA** — i.e. **non-commercial and copyleft**. The PI has
accepted this risk knowingly. *(No legal advice was obtained; option D was not chosen.)*

**What this decision does NOT change — these are licence terms, not risk tolerances:**
- **nuScenes may never enter `TanitDataSet-C`.** C is *defined* as `owned-safe` + `commercial_ok` +
  no-share-alike. NC data in a commercial tier would be a licence **violation**, not a risk. The export
  guard stays exactly as it is, and the 8 regression tests that enforce it stay green.
- **Redistribution stays off.** We ship **pointers + derived features, never source bytes** — unchanged.
- **The SA shard stays segregated** (`shards/nc-research/sharealike/nuscenes/`). That routing is what the
  corrected licence *requires*; it is not extra caution.

⇒ In scope of this decision: **ingest into the research tier R, and train research models on it freely.**

## 2. The mechanical blocker — needs the PI, ~2 minutes

**MEASURED:** `Keys.txt` contains **no nuScenes / Motional / nuTonomy credential** (0 matches), and there
is **no official `motional`/`nutonomy` org on HuggingFace** — all 12 hits are third-party mirrors, which
were **not** used.

nuScenes gates downloads behind **a human account + acceptance of its Terms of Use**. An agent cannot
create an account or accept terms on the PI's behalf — that is a hard boundary, independent of the
licensing decision above. **The PI's "ingest freely" resolves the *licence-risk* question; it does not and
cannot substitute for accepting nuScenes' Terms with a human account.**

**What unblocks it (either is fine):**
1. Register at `nuscenes.org`, accept the Terms, and place the credential/token in `Keys.txt`; **or**
2. Download `v1.0-trainval` metadata + the **map expansion** manually and drop them on a pod.

**The metadata-first path is the cheap one:** `~0.48 GB` answers the entire value question (lane graph,
`is_intersection`, YIELD stop-lines, roundabout counts) **before** any image pull.

## 3. Everything else is ready — one command after access

| component | state |
|---|---|
| `stack/tanitad/data/nuscenes.py` — adapter (ego pose, 3D tracks with instance ids, per-sample calibration, **cross-camera frustum visibility as a real projection**) | ✅ built, **22 tests green** |
| `stack/scripts/ingest_nuscenes.py` — driver on the existing lake contract | ✅ built |
| `SOURCE_REGISTRY` — `share_alike=True`, `commercial_ok=False`, tier `nc` | ✅ corrected |
| SA-shard routing + C-tier refusal | ✅ **8 regression tests**, each gate refusing independently |
| Geometry — `calib_r1.pinhole_rectify` (stranded 9 days) | ✅ folded into `calib.py`; `fx 1266.4 → 266.0` exactly |

## 4. What nuScenes buys — and the one thing it does not

**Buys:** a **routable lane graph** (`lane_connector`, connectivity, arcline, `is_intersection`, YIELD
stop-lines) — **the strategic-brain ground truth**, which is now *proven absent* from PhysicalAI-AV at five
independent probes and **unrecoverable via OSM** (our `egomotion` has no lat/lon). This unblocks **S1**
(branch selection), **S2** (lane selection) and **HP-4** (compositional generalization to unseen junction
topologies).

**Does not buy:** **traffic-light state** — nuScenes has static geometry only, no light state, ever
(that would need OpenLane-V2). And its roundabout content is **unpublished/unknown** — the metadata pull
settles it for `~0.48 GB`.

## 5. The residual strategic issue — separate from this decision

A proof built on nuScenes is **NC+SA and cannot ship**. That does not make it worthless — the 4-brain
proof was **already** research-tier, since PhysicalAI-AV is `gated-confidential`. But **if the proof is what
gets shown to investors or partners, it needs a publishable twin.**

**The free next check remains open and is worth doing regardless of this decision:**
**Cosmos-Drive-Dreams** (`cosmos_dd`) is **CC-BY-4.0, `owned-safe`, commercially usable, already loaded**,
7 cameras, 4D tracking, **HD map** — and its junction/roundabout content is **countable from cached
metadata for $0**. If it carries the topology, we get a **publishable** strategic-brain proof *in addition
to* the nuScenes one. Recommended, not blocking.
