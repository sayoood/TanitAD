"""Query the live renderer: is camera_front_wide_120fov ftheta or pinhole? (make-or-break)."""
import sys
import grpc
from alpasim_grpc.v0 import common_pb2, sensorsim_pb2, sensorsim_pb2_grpc

ch = grpc.insecure_channel("localhost:6011")
stub = sensorsim_pb2_grpc.SensorsimServiceStub(ch)
scenes = stub.get_available_scenes(common_pb2.Empty())
sid = list(scenes.scene_ids)[0]
print("scene_id:", sid)
cams = stub.get_available_cameras(sensorsim_pb2.AvailableCamerasRequest(scene_id=sid))
print("n_cameras:", len(cams.available_cameras),
      "logical_ids:", [c.logical_id for c in cams.available_cameras])
for c in cams.available_cameras:
    if "front_wide" in c.logical_id or "front:wide" in c.logical_id:
        spec = c.intrinsics
        which = spec.WhichOneof("camera_param")
        print("\n=== camera_front_wide_120fov ===")
        print("logical_id:", c.logical_id)
        print("resolution_hw:", spec.resolution_h, spec.resolution_w)
        print("CAMERA MODEL (oneof):", which)
        if which == "ftheta_param":
            ft = spec.ftheta_param
            print("NATIVE_FTHETA_CONFIRMED")
            print("principal_point:", ft.principal_point_x, ft.principal_point_y)
            print("reference_poly:", ft.reference_poly)
            print("pixeldist_to_angle_poly (backward):", list(ft.pixeldist_to_angle_poly))
            print("angle_to_pixeldist_poly (forward):", list(ft.angle_to_pixeldist_poly))
            print("max_angle:", ft.max_angle)
            print("linear_cde:", ft.linear_cde.linear_c, ft.linear_cde.linear_d, ft.linear_cde.linear_e)
        elif which == "opencv_pinhole_param":
            print("BAKED_PINHOLE_DETECTED -- STOP, A approach changes")
            print(spec.opencv_pinhole_param)
        else:
            print("OTHER MODEL:", which, spec)
        break
print("PROBE_DONE")
