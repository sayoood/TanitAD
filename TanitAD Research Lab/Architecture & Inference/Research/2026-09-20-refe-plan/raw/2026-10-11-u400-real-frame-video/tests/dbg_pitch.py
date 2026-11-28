import sys, json, numpy as np
sys.path.insert(0, ".")
import u400_lib as L
log = sys.argv[1]
ld = L.LogData(log)
P = ld.img["CAM_F0"]["pose"]
pit, rol, zz = [], [], []
for r in P:
    Rm = L.Pose(r).Rm
    pit.append(np.degrees(np.arcsin(-Rm[2, 0]))); rol.append(np.degrees(np.arctan2(Rm[2, 1], Rm[2, 2]))); zz.append(r[2])
pit, rol, zz = map(np.array, (pit, rol, zz))
print("pitch deg: mean %.2f std %.2f min %.2f max %.2f | frame-to-frame |d| median %.3f p99 %.3f max %.3f" % (pit.mean(), pit.std(), pit.min(), pit.max(), np.median(np.abs(np.diff(pit))), np.percentile(np.abs(np.diff(pit)), 99), np.abs(np.diff(pit)).max()))
print("roll  deg: mean %.2f std %.2f min %.2f max %.2f | frame-to-frame |d| median %.3f p99 %.3f max %.3f" % (rol.mean(), rol.std(), rol.min(), rol.max(), np.median(np.abs(np.diff(rol))), np.percentile(np.abs(np.diff(rol)), 99), np.abs(np.diff(rol)).max()))
print("z frame-to-frame |d| m median %.4f p99 %.4f max %.4f" % (np.median(np.abs(np.diff(zz))), np.percentile(np.abs(np.diff(zz)), 99), np.abs(np.diff(zz)).max()))
ts = ld.img["CAM_F0"]["ts"]; d = np.abs(ld.img["CAM_F0"]["ets"] - ts)
print("ego_pose-vs-image ts |dt| us: max", d.max())
