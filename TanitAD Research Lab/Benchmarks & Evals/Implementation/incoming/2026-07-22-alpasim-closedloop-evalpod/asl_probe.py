#!/usr/bin/env python3
"""Probe a gate1 rollout.asl: event schema, camera frames, timestamps, alignment
to preds.jsonl. Decides the Gate-1 frame-extraction approach."""
import asyncio, glob, json, os
from collections import Counter, defaultdict
import numpy as np
from alpasim_utils.logs import async_read_pb_log

SCENE = "00169207"
LOGDIR = "/workspace/gate1_junc"
roll_dir = glob.glob(f"{LOGDIR}/rollouts/clipgt-{SCENE}-*/*/")[0]
asl = os.path.join(roll_dir, "rollout.asl")
session = roll_dir.rstrip("/").split("/")[-1]
print("ASL:", asl)
print("SESSION:", session)

# preds for this session
preds = [json.loads(l) for l in open(f"{LOGDIR}/preds.jsonl") if l.strip()]
mine = sorted([p for p in preds if p["session"] == session], key=lambda r: r["t"])
print(f"PREDS for session: n={len(mine)}")
if mine:
    print(f"  t range: {mine[0]['t']} .. {mine[-1]['t']}  (span {(mine[-1]['t']-mine[0]['t'])/1e6:.2f}s)")
    print(f"  first pred: x={mine[0]['x']:.2f} y={mine[0]['y']:.2f} yaw={mine[0]['yaw']:.3f} speed={mine[0]['speed']:.2f}")


async def main():
    field_counter = Counter()
    cam_frames = []          # (ts, logical_id, nbytes)
    ego_events = []          # (ts, npose, has_dyn)
    route_events = 0
    cam_intr = None
    n = 0
    async for e in async_read_pb_log(asl):
        n += 1
        set_fields = [f.name for f, _ in e.ListFields()]
        for f in set_fields:
            field_counter[f] += 1
        # camera images
        if e.HasField("image_observation") if _has(e, "image_observation") else False:
            io = e.image_observation
            ci = getattr(io, "camera_image", io)
            lid = getattr(ci, "logical_id", "?")
            nb = len(getattr(ci, "image_bytes", b""))
            ts = getattr(io, "timestamp_us", getattr(e, "timestamp_us", 0))
            cam_frames.append((ts, lid, nb))
        # available cameras (intrinsics)
        if _has(e, "available_cameras_return") and e.HasField("available_cameras_return"):
            for cam in e.available_cameras_return.available_cameras:
                if "front" in cam.logical_id.lower():
                    spec = cam.intrinsics
                    m = spec.WhichOneof("camera_param")
                    cam_intr = (cam.logical_id, m, spec.resolution_h, spec.resolution_w)
        if _has(e, "egomotion_observation") and e.HasField("egomotion_observation"):
            eo = e.egomotion_observation
            npose = len(eo.trajectory.poses) if _has(eo, "trajectory") else 0
            ego_events.append((getattr(e, "timestamp_us", 0), npose,
                               len(getattr(eo, "dynamic_states", []))))
        if _has(e, "route_observation") and e.HasField("route_observation"):
            route_events += 1

    print(f"\nTOTAL events: {n}")
    print("FIELD histogram (top 25):")
    for k, v in field_counter.most_common(25):
        print(f"   {k:40s} {v}")
    print(f"\nCAMERA frames logged: {len(cam_frames)}")
    if cam_frames:
        # try to filter to a single camera by most-common logical id
        lids = Counter(l for _, l, _ in cam_frames)
        print("  logical_ids:", dict(lids))
        ts = np.array([t for t, _, _ in cam_frames], dtype=float)
        print(f"  cam ts range: {ts.min():.0f} .. {ts.max():.0f} (span {(ts.max()-ts.min())/1e6:.2f}s)")
        print(f"  median dt between cam frames: {np.median(np.diff(np.sort(ts)))/1e3:.1f} ms")
    print("CAM intrinsics:", cam_intr)
    print(f"EGO events: {len(ego_events)}  ROUTE events: {route_events}")

    # alignment: for a few preds t, count cam frames with ts <= t (single cam)
    if cam_frames and mine:
        front = [t for t, l, _ in cam_frames if "front" in l.lower()]
        front = np.array(sorted(front), dtype=float) if front else np.array(sorted(ts))
        print("\nALIGNMENT (preds t -> #front cam frames with ts<=t):")
        for p in [mine[0], mine[len(mine)//2], mine[-1]]:
            c = int((front <= p["t"]).sum())
            print(f"   step t={p['t']} -> {c} frames available (need>=10)")


def _has(msg, field):
    try:
        return field in [f.name for f in msg.DESCRIPTOR.fields]
    except Exception:
        return False


asyncio.run(main())
