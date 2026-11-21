# Gate-0 free-floor — PREREQUISITE CHECK: drivable-area geometry IS accessible to the driver ✅

The coordinator's mandated first step: *"is a drivable-area/road-boundary geometry accessible to the
driver at inference? If NOT, STOP and surface it as the blocker."* **Answer: MET — richly so.** Evidence
`gate0_prereq_probe.json` + `gate0_probe.py` (staged). Lock `gate0-freefloor`; pod left clean.

## Verdict: PREREQUISITE MET
The drivable-area geometry (the very geometry AlpaSim's `offroad` scorer uses) is a **`trajdata.VectorMap`
embedded in each scene's USDZ**, and the driver — running on the same pod with the same USDZ paths — can
load it directly:

```python
from alpasim_runtime.scene_loader import ArtifactSceneProvider
prov = ArtifactSceneProvider.from_path(SCENESET_DIR, smooth_trajectories=True)
ds   = prov.get_data_source(scene_id)          # scene_id arrives in DriveSessionRequest.debug_info
vm   = ds.map                                  # trajdata VectorMap
```

**Measured on the actual suite USDZs (the junction-failure scenes):**
| scene | category | lanes | road_edges | wait_lines (stop/yield) |
|---|---|---|---|---|
| 00169207 | intersection | 385 | 190 | 135 |
| 41c06176 | intersection | 147 | 130 | 43 |
| 3cc29c99 | roundabout | 130 | 139 | 27 |
| 6dcd2117 | roundabout | 149 | 175 | 45 |
| adb72a39 | roundabout | 472 | 393 | 180 |

Every junction scene carries **130–472 lane polygons + 130–393 road-edge polylines + wait-lines** — abundant
drivable-area/road-boundary geometry. (Maps load for all 38; probe checked the 5 junction scenes that matter.)

## The exact loadable API (implementation is fully de-risked)
- **Road edges (the boundary):** `vm.elements[MapElementType.ROAD_EDGE]` → `{id: RoadEdge}`, each with
  `.polyline` (a point sequence). **Lanes (the drivable surface):** `vm.elements[MapElementType.ROAD_LANE]`
  → lane polygons (point-in-polygon = on/off drivable). Helper `vm.get_closest_road_edge(xyz)` exists
  (needs a **3-vector**, not 2 — my first probe's only bug).
- **Ego pose in the map frame:** `ds.rig.trajectory` is a `utils_rs.Trajectory` with `.positions` **(202,3)**
  + `.rotation_matrices`/`.quaternions`/`.get_pose(t)` — the ego's world-frame SE(3) path, **the same frame as
  the vec_map** (both from the USDZ). The driver receives this pose stream **live** at inference via
  `submit_egomotion_observation(request.trajectory.poses)` (already parsed in `refc_driver.py`). → the driver
  can transform road edges/lanes (map frame) into its **rig-frame plan** at each `drive()`.
- Other agents for the collision term: `request.dynamic_states` (already received) + `ds.traffic_objects`.

So all four ingredients the floor needs — drivable surface, road boundary, ego pose for alignment, and agent
positions — are in hand at inference. **Nothing blocks Gate-0.**

## Implementation design (specified; the anchored structure makes it favorable)
REF-C is **anchored diffusion**: `AnchoredDiffusionDecoder` denoises **N=128 anchor trajectories** over
`steps=2` passes (`x = anchors + offset`, shape `[B,128,n_steps,2]`), then selects by a `conf_head`
(`tanitad/models/refc_rescorer.py`, `fourbrain.py`). Two training-free additions:
- **(a) cost-guidance** — build a rig-frame drivable-area energy `E(x)` = Σ max(0, signed-dist-outside-lane) +
  collision term vs `dynamic_states`; at each of the 2 denoise steps subtract `λ·∂E/∂x` from the anchor
  trajectories (nudge on-road). *Note:* for anchored diffusion the equivalent, lower-risk realization is
  **cost-guided selection** — pick `argmax(conf − λ·offroad_cost − μ·collision_cost)` over the 128 denoised
  anchors (hooks the existing rescorer machinery). Recommend running the selection floor first (robust), then
  the per-step gradient as the refinement.
- **(b) safety clamp** — if the selected plan still has a waypoint outside all lane polygons, override:
  project to the nearest in-lane point / fall back to the route-GT corridor.
Then run **REF-C-base WITH vs WITHOUT** on the 38-scene suite, paired per-category, focus intersection +
roundabout off-road. Set `eval.allow_aggregation_with_failed_rollouts: true` (the route-sanity trap).

## Status / recommendation
- ✅ **Prerequisite CLEARED with hard evidence** — the whole free-floor rung is unblocked.
- The guidance+clamp implementation + 2 paired 38-scene runs is the **next substantial, intricate chunk**
  (coordinate transforms, differentiable off-road cost, anchor re-ranking, then ~70 min of runs). It should be
  done as a **focused piece** — a rushed guidance with a frame/sign bug would yield a *misleading* Gate-0
  number, which the pre-registration's "report plainly, do not force a pass" explicitly guards against. I
  banked the cleared prerequisite + this de-risked design rather than half-build the guidance at the tail of a
  very long session.
- **Pre-registration stands:** off-road collapses toward ~0 (esp. intersections) → Gate-0 PASS, zero training;
  residual survives → that residual is the Gate-1 target. ⚠️ within-sim / ~3.2×-OOD throughout.

## Manifest
| artifact | where | status |
|---|---|---|
| `gate0_prereq_probe.json` + `gate0_probe.py` | repo (staged) | prerequisite evidence + loader probe |
| `gate0_prerequisite_NOTE.md` (this) | repo (staged) | verdict + API + implementation design |

Pod: lock released, GPU idle, no procs — clean.
