#!/usr/bin/env python3
"""Gate-0 API introspection: nail down the EXACT VectorMap lane-polygon API, the
ego-trajectory frame, and the drivable-area geometry, so the off-road cost can be
built without a frame/sign bug. Read-only, CPU-only (no model, no GPU)."""
import sys, json, glob, os, inspect
sys.path.insert(0, "/workspace/alpa-invest/alpasim/src/runtime")
import numpy as np
from alpasim_runtime.scene_loader import ArtifactSceneProvider

SSDIR = "/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets/986fec83193b1baf3d5121f09462e248"
if not os.path.isdir(SSDIR):
    # discover the scaled-suite sceneset instead
    cands = sorted(glob.glob("/workspace/alpa-invest/alpasim/data/nre-artifacts/scenesets/*"))
    print("SSDIR fallback candidates:", cands)
    SSDIR = cands[-1] if cands else SSDIR

prov = ArtifactSceneProvider.from_path(SSDIR, smooth_trajectories=True)
ids = sorted(prov.scene_ids)
print("discovered", len(ids), "scenes in", SSDIR)

# pick an intersection scene (00169207) if present, else first
target = None
for sid in ids:
    if sid.replace("clipgt-", "")[:8] == "00169207":
        target = sid; break
if target is None:
    target = ids[0]
print("TARGET scene:", target)

ds = prov.get_data_source(target)
print("data_source type:", type(ds).__name__, "attrs:", [a for a in dir(ds) if not a.startswith('_')])

vm = ds.map
from trajdata.maps.vec_map_elements import MapElementType
print("\n=== MapElementType enum ===")
for e in MapElementType:
    print(" ", int(e.value), e.name)

print("\n=== vm.elements keys ===")
for k, v in vm.elements.items():
    print("  key", k, "->", len(v), "elements")

# ---- Inspect a ROAD_LANE element (drivable surface / polygon) ----
lanes = vm.elements.get(MapElementType.ROAD_LANE, {})
print("\n=== ROAD_LANE sample ===")
lk = next(iter(lanes))
lane = lanes[lk]
print("lane id:", lk, "type:", type(lane).__name__)
print("lane attrs:", [a for a in dir(lane) if not a.startswith('_')])
for attr in ("center", "left_edge", "right_edge", "polygon", "as_polygon", "lane_center"):
    if hasattr(lane, attr):
        val = getattr(lane, attr)
        if callable(val):
            print(f"  lane.{attr} is CALLABLE")
            continue
        try:
            arr = None
            for sub in ("xy", "points", "positions"):
                if hasattr(val, sub):
                    arr = np.asarray(getattr(val, sub)); break
            if arr is None:
                arr = np.asarray(val)
            print(f"  lane.{attr}: type={type(val).__name__} shape={getattr(arr,'shape',None)}")
            if arr is not None and arr.size:
                print(f"     first pt {arr.reshape(-1, arr.shape[-1])[0]} last {arr.reshape(-1, arr.shape[-1])[-1]}")
        except Exception as ex:
            print(f"  lane.{attr}: type={type(val).__name__} (introspect err {ex!r})")

# get_closest_lane / point-in-lane helpers on vm
print("\n=== vm helpers ===")
print("vm attrs:", [a for a in dir(vm) if not a.startswith('_')])

# ---- ROAD_EDGE polyline (boundary) ----
from trajdata.maps.vec_map_elements import MapElementType as MET
edges = vm.elements.get(MET.ROAD_EDGE, {})
print("\n=== ROAD_EDGE sample ===")
ek = next(iter(edges)); edge = edges[ek]
print("edge attrs:", [a for a in dir(edge) if not a.startswith('_')])
pl = getattr(edge, "polyline", None)
if pl is not None:
    for sub in ("xy", "points", "positions"):
        if hasattr(pl, sub):
            a = np.asarray(getattr(pl, sub)); print(f"  edge.polyline.{sub} shape {a.shape} first {a.reshape(-1,a.shape[-1])[0]}"); break
    else:
        a = np.asarray(pl); print("  edge.polyline asarray shape", a.shape)

# ---- Ego trajectory frame ----
print("\n=== ego trajectory ===")
tr = ds.rig.trajectory
print("traj type:", type(tr).__name__, "attrs:", [a for a in dir(tr) if not a.startswith('_')])
pos = np.asarray(tr.positions)
print("positions shape:", pos.shape, "start:", pos[0], "end:", pos[-1])
# bounding box of lanes vs ego to confirm SAME frame
allpts = []
for lid, ln in list(lanes.items())[:50]:
    ctr = getattr(ln, "center", None)
    if ctr is not None:
        for sub in ("xy", "points", "positions"):
            if hasattr(ctr, sub):
                allpts.append(np.asarray(getattr(ctr, sub))[:, :2]); break
if allpts:
    allpts = np.concatenate(allpts, 0)
    print("lane-center bbox x[%.1f,%.1f] y[%.1f,%.1f]" % (allpts[:,0].min(), allpts[:,0].max(), allpts[:,1].min(), allpts[:,1].max()))
    print("ego bbox       x[%.1f,%.1f] y[%.1f,%.1f]" % (pos[:,0].min(), pos[:,0].max(), pos[:,1].min(), pos[:,1].max()))
    # distance from ego mid-point to nearest lane-center point (should be small -> same frame + on-road)
    mid = pos[len(pos)//2, :2]
    dmin = np.linalg.norm(allpts - mid, axis=1).min()
    print("ego-midpoint -> nearest lane-center dist: %.2f m (small => same frame, on-road)" % dmin)
print("\nPROBE_OK")
