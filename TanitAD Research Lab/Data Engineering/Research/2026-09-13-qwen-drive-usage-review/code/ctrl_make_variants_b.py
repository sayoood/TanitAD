"""Second control pass: separate SCALE from BLACK BORDERS, which variant A2 confounded.

A2 re-rendered each demo view at f = 313.7 px @896. A 63.7 deg source cannot fill a
110 deg canvas, so A2 was ALSO 81.4 % black -- a side effect our v1 PhysicalAI packing
never had (its 110 deg views were full of real pixels). A2's collapse therefore does not
by itself say whether scale or blackness broke the model.

  A4  demo scale kept (K as A0r), everything outside A2's content window BLACK (81.4 %)
      -> blackness alone, at the correct scale
  A5  A0r with ONLY CAM_B0 masked like A4
      -> the v2 PhysicalAI design (B0 is ~82 % black), measured where GT exists
  A6  zoom IN by 2.40x (the v1 B0 tele focal 1732.3 px), a centred crop, NO black
      -> scale alone, in the direction that needs no padding
"""
import json, shutil, sys
from pathlib import Path
import numpy as np
import cv2

SRC = Path(sys.argv[1]); DST = Path(sys.argv[2])
OW, OH = 896, 512
WF, HF = 313.69297711795 / 721.0, 313.69297711795 / 732.4375   # A2's content window, as fractions


def K896(K, w, h):
    return np.diag([OW / w, OH / h, 1.0]) @ K


def remap(img, Ks, Kd):
    u, v = np.meshgrid(np.arange(OW, dtype=np.float32), np.arange(OH, dtype=np.float32))
    mx = (Ks[0, 0] * (u - Kd[0, 2]) / Kd[0, 0] + Ks[0, 2]).astype(np.float32)
    my = (Ks[1, 1] * (v - Kd[1, 2]) / Kd[1, 1] + Ks[1, 2]).astype(np.float32)
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def make(fd, out, mode):
    out.mkdir(parents=True, exist_ok=True); (out / "images").mkdir(exist_ok=True)
    fr = json.loads((fd / "frame.json").read_text()); c = dict(np.load(fd / "calib.npz"))
    Kout = []
    for i, cam in enumerate(fr["cam_order"]):
        img = cv2.imread(str(fd / "images" / f"{cam}.jpg"))
        K = c["cam_intrinsic"][i].astype(np.float64)
        Kd = K896(K, img.shape[1], img.shape[0])
        if mode == "A6":
            s = 1732.3 / Kd[0, 0]
            Kd = np.array([[Kd[0, 0] * s, 0, OW / 2], [0, Kd[1, 1] * s, OH / 2 + (Kd[1, 2] - OH / 2) * s], [0, 0, 1]])
        res = remap(img, K, Kd)
        if mode == "A4" or (mode == "A5" and cam == "CAM_B0"):
            m = np.zeros((OH, OW), bool)
            x0, x1 = int(round(OW / 2 - WF * OW / 2)), int(round(OW / 2 + WF * OW / 2))
            y0, y1 = int(round(Kd[1, 2] - HF * Kd[1, 2])), int(round(Kd[1, 2] + HF * (OH - Kd[1, 2])))
            m[y0:y1, x0:x1] = True
            res[~m] = 0
        cv2.imwrite(str(out / "images" / f"{cam}.jpg"), res, [cv2.IMWRITE_JPEG_QUALITY, 95])
        Kout.append(Kd)
    np.savez(out / "calib.npz", cam_intrinsic=np.stack(Kout), sensor2lidar_rotation=c["sensor2lidar_rotation"],
             sensor2lidar_translation=c["sensor2lidar_translation"], lidar2ego=c["lidar2ego"])
    shutil.copy2(fd / "frame.json", out / "frame.json"); shutil.copy2(fd / "gt.npz", out / "gt.npz")
    if (fd / "lidar.npy").exists():
        shutil.copy2(fd / "lidar.npy", out / "lidar.npy")


for fd in sorted(p for p in SRC.iterdir() if p.is_dir()):
    if json.loads((fd / "frame.json").read_text())["dataset_type"] != "nuplan":
        continue
    for mode in ("A4", "A5", "A6"):
        make(fd, DST / mode / fd.name, mode)
# ⛔ content assertions: the masked fraction must be what the docstring says
for mode in ("A4", "A5", "A6"):
    fr = []
    for img in (DST / mode).glob("*/images/*.jpg"):
        a = cv2.imread(str(img)); fr.append(float((a.max(axis=2) < 8).mean()))
    print(mode, "images", len(fr), "black-fraction mean", round(float(np.mean(fr)), 4), "max", round(float(np.max(fr)), 4))
