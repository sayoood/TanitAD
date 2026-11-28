"""Read the M2 rollout.asl → the camera model the renderer actually served (ftheta vs pinhole)."""
import asyncio, glob
from alpasim_utils.logs import async_read_pb_log

asls = glob.glob("/workspace/m2run/rollouts/**/rollout.asl", recursive=True)
print("asl:", asls[0])


async def main():
    n = 0
    found = False
    async for e in async_read_pb_log(asls[0]):
        n += 1
        if e.HasField("available_cameras_return"):
            for cam in e.available_cameras_return.available_cameras:
                if "front_wide" in cam.logical_id or "front:wide" in cam.logical_id:
                    spec = cam.intrinsics
                    m = spec.WhichOneof("camera_param")
                    print(f"\n[available_cameras_return] {cam.logical_id}  "
                          f"MODEL={m}  res={spec.resolution_h}x{spec.resolution_w}")
                    if m == "ftheta_param":
                        ft = spec.ftheta_param
                        print("NATIVE_FTHETA_CONFIRMED")
                        print("  principal:", ft.principal_point_x, ft.principal_point_y)
                        print("  reference_poly:", ft.reference_poly)
                        print("  angle_to_pixeldist (forward):", list(ft.angle_to_pixeldist_poly))
                        print("  pixeldist_to_angle (backward):", list(ft.pixeldist_to_angle_poly))
                        print("  max_angle:", ft.max_angle)
                    elif m == "opencv_pinhole_param":
                        print("BAKED_PINHOLE -- STOP")
                        print(spec.opencv_pinhole_param)
                    found = True
            if found:
                break
    print(f"\nASL_PROBE_DONE scanned={n} found={found}")


asyncio.run(main())
