import pickle, sys, math
import numpy as np
d = pickle.load(open(sys.argv[1], "rb"))
fr = d[0]
for c in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_B0"):
    cd = fr["cams"][c]
    R = np.asarray(cd["sensor2lidar_rotation"], float)
    fwd = R @ np.array([0.0, 0.0, 1.0])
    K = np.asarray(cd["cam_intrinsic"], float)
    yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
    hfov = 2 * math.degrees(math.atan(960.0 / K[0, 0]))
    print(c, "yaw_deg %.1f" % yaw, "fx %.1f" % K[0, 0], "hfov(1920px) %.1f" % hfov, "t", np.round(np.asarray(cd["sensor2lidar_translation"], float), 2))
