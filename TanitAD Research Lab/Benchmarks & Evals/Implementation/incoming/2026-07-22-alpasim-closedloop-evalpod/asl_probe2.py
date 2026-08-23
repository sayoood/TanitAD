#!/usr/bin/env python3
"""Probe v2: exact message structure of driver_camera_image / driver_request /
route_request / driver_ego_trajectory, and reconstruct the deque REF-C saw."""
import asyncio, glob, json, os
from collections import deque
from io import BytesIO
import numpy as np
from PIL import Image
from alpasim_utils.logs import async_read_pb_log

SCENE = "00169207"
LOGDIR = "/workspace/gate1_junc"
roll_dir = glob.glob(f"{LOGDIR}/rollouts/clipgt-{SCENE}-*/*/")[0]
asl = os.path.join(roll_dir, "rollout.asl")
session = roll_dir.rstrip("/").split("/")[-1]
preds = [json.loads(l) for l in open(f"{LOGDIR}/preds.jsonl") if l.strip()]
mine = sorted([p for p in preds if p["session"] == session], key=lambda r: r["t"])
print(f"session={session[:12]} preds={len(mine)}")


def descfields(msg):
    return [f.name for f, _ in msg.ListFields()]


async def main():
    printed = {"driver_camera_image": 0, "driver_request": 0, "route_request": 0,
               "driver_ego_trajectory": 0, "driver_return": 0}
    frames = deque(maxlen=24)
    drive_snaps = []       # (time_now_us, n_frames_in_deque)
    first_img_shape = None
    async for e in async_read_pb_log(asl):
        fields = descfields(e)
        # ---- inspect one of each interesting wrapper
        for key in printed:
            if key in fields and printed[key] < 1:
                sub = getattr(e, key)
                print(f"\n=== {key} fields: {descfields(sub)}")
                # dig one level
                for fn in descfields(sub):
                    val = getattr(sub, fn)
                    t = type(val).__name__
                    extra = ""
                    if hasattr(val, "ListFields"):
                        extra = f" -> {descfields(val)}"
                    elif isinstance(val, bytes):
                        extra = f" [{len(val)} bytes]"
                    print(f"      .{fn}: {t}{extra}")
                printed[key] += 1
        # ---- accumulate camera images into the deque (like the driver)
        if "driver_camera_image" in fields:
            ci = e.driver_camera_image
            # find the image bytes + logical id
            lid = getattr(ci, "logical_id", None)
            raw = getattr(ci, "image_bytes", None)
            # maybe nested under camera_image
            if raw is None and hasattr(ci, "camera_image"):
                inner = ci.camera_image
                lid = getattr(inner, "logical_id", lid)
                raw = getattr(inner, "image_bytes", None)
            if raw is not None:
                if first_img_shape is None:
                    arr = np.array(Image.open(BytesIO(raw)).convert("RGB"))
                    first_img_shape = arr.shape
                frames.append(1)
        # ---- driver_request = a drive() call
        if "driver_request" in fields:
            dr = e.driver_request
            tnow = getattr(dr, "time_now_us", None)
            if tnow is None:
                for fn in descfields(dr):
                    v = getattr(dr, fn)
                    if isinstance(v, int) and v > 1e9:
                        tnow = v
            drive_snaps.append((tnow, len(frames)))

    print(f"\nfirst decoded image shape (HWC): {first_img_shape}")
    print(f"\nDRIVE snapshots: {len(drive_snaps)}  (time_now_us, #frames_in_deque)")
    for i, (t, nf) in enumerate(drive_snaps):
        tag = ""
        if mine and t is not None:
            match = [p for p in mine if abs(p["t"] - t) < 1000]
            tag = f"  pred_match={'Y' if match else 'n'}"
        print(f"   drive[{i:2d}] t={t} nframes={nf}{tag}")
        if i >= 12 and i < len(drive_snaps) - 3:
            if i == 13:
                print("   ...")
            continue


asyncio.run(main())
