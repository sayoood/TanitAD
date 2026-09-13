"""Build DEGRADED copies of Qwen-Drive's own demo frames, one defect at a time.

The demo frames are the only data on which the released weights are known-good AND
ground truth exists (occ / map / boxes, already label-mapped). Each variant
re-introduces ONE property of our 2026-09-11 PhysicalAI packing, with the geometry
kept EXACT (the calibration is rewritten to match every resampled image), so any
drop is the model's sensitivity to that property -- not a projection bug.

  A0   the demo as shipped                                   (reference)
  A0r  resampled to 896x512 through our remap, SAME K        (no-op control: must ~= A0)
  A1   CAM_R1 and CAM_L1 removed -> 6 cameras                (what we fed: 6 of 8)
  A2   every view re-rendered at f=313.7 px @896 (110 deg)   (our focal: objects 2.3x smaller)
  A3   A2 + principal point moved to v0=383.7                (our off-centre principal point)

nuScenes demo frames are passed through A0 only (their rig is the other training rig).
"""
import json, shutil, sys
from pathlib import Path
import numpy as np
import cv2

SRC = Path(sys.argv[1])          # qwen-drive/data/demo/perception
DST = Path(sys.argv[2])          # ctrl/
OW, OH = 896, 512


def remap_pinhole(img, K_src, K_dst, out_w, out_h):
    """Pinhole -> pinhole with the SAME optical centre and rotation (exact, no parallax)."""
    u, v = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))
    x = (u - K_dst[0, 2]) / K_dst[0, 0]
    y = (v - K_dst[1, 2]) / K_dst[1, 1]
    mx = (K_src[0, 0] * x + K_src[0, 2]).astype(np.float32)
    my = (K_src[1, 1] * y + K_src[1, 2]).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def copy_frame(fd, out, cams_keep=None, K_new=None):
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    fr = json.loads((fd / "frame.json").read_text())
    c = dict(np.load(fd / "calib.npz"))
    order = fr["cam_order"]
    keep = [i for i, cam in enumerate(order) if cams_keep is None or cam in cams_keep]
    new_order = [order[i] for i in keep]
    K_out = []
    for i in keep:
        cam = order[i]
        img = cv2.imread(str(fd / "images" / f"{cam}.jpg"))
        K = c["cam_intrinsic"][i].astype(np.float64)
        if K_new is None:
            shutil.copy2(fd / "images" / f"{cam}.jpg", out / "images" / f"{cam}.jpg")
            K_out.append(K)
        else:
            Kd = K_new(K, img.shape[1], img.shape[0])
            res = remap_pinhole(img, K, Kd, OW, OH)
            cv2.imwrite(str(out / "images" / f"{cam}.jpg"), res, [cv2.IMWRITE_JPEG_QUALITY, 95])
            K_out.append(Kd)
    np.savez(out / "calib.npz",
             cam_intrinsic=np.stack(K_out),
             sensor2lidar_rotation=c["sensor2lidar_rotation"][keep],
             sensor2lidar_translation=c["sensor2lidar_translation"][keep],
             lidar2ego=c["lidar2ego"])
    # rebuild content keeping (tag, image) pairs of kept cameras, plus the trailing instruction
    content = []
    items = fr["content"]
    j = 0
    while j < len(items):
        it = items[j]
        if "text" in it and j + 1 < len(items) and "image" in items[j + 1]:
            if items[j + 1]["image"] in new_order:
                content += [it, items[j + 1]]
            j += 2
        else:
            content.append(it)
            j += 1
    fr2 = dict(fr, cam_order=new_order, content=content)
    (out / "frame.json").write_text(json.dumps(fr2, indent=2))
    shutil.copy2(fd / "gt.npz", out / "gt.npz")
    if (fd / "lidar.npy").exists():
        shutil.copy2(fd / "lidar.npy", out / "lidar.npy")
    return new_order


def same_K_at_896(K, w, h):
    s = np.diag([OW / w, OH / h, 1.0])
    return s @ K


def ours_focal(K, w, h):
    return np.array([[313.69297711795, 0, 448.0], [0, 313.69297711795, 265.5], [0, 0, 1]])


def ours_focal_pp(K, w, h):
    return np.array([[313.69297711795, 0, 448.0], [0, 313.69297711795, 383.68436617467574], [0, 0, 1]])


frames = sorted(p for p in SRC.iterdir() if p.is_dir())
log = {}
for fd in frames:
    fr = json.loads((fd / "frame.json").read_text())
    kind = fr["dataset_type"]
    log[fd.name] = {"type": kind}
    copy_frame(fd, DST / "A0" / fd.name)
    if kind != "nuplan":
        continue
    copy_frame(fd, DST / "A0r" / fd.name, K_new=same_K_at_896)
    log[fd.name]["A1_cams"] = copy_frame(fd, DST / "A1" / fd.name,
                                         cams_keep={"CAM_F0", "CAM_R0", "CAM_R2", "CAM_B0", "CAM_L2", "CAM_L0"})
    copy_frame(fd, DST / "A2" / fd.name, K_new=ours_focal)
    copy_frame(fd, DST / "A3" / fd.name, K_new=ours_focal_pp)

# ⛔ CONTENT assertion on every generated image: a black or failed remap must not pass
bad = []
for img in DST.glob("A*/*/images/*.jpg"):
    a = cv2.imread(str(img))
    if a is None or float(a.mean()) < 3.0:
        bad.append(str(img))
print(json.dumps(log, indent=1))
print("VARIANT_IMAGES", len(list(DST.glob('A*/*/images/*.jpg'))), "BAD", len(bad), bad[:5])
