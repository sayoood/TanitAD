"""Scorer for native-camera maps: the coverage / visibility masks use the SAME cameras (and camera model) the map used.

Before: VIEWS hard-coded to the 7 virtual pinhole views and their calib.npz under the v2 sequence. A v6 map (native f-theta
cameras) scored with that mask would be restricted to ground the VIRTUAL rig sees -- the wrong denominator.
After: the views are the cls_<cam> keys the npz carries; calib.npz / frame.json come from SAM3MAP_ROOT (default the v2
sequence, i.e. unchanged for every banked arm); projection is camera_model.Camera (pinhole or f-theta), with the image
size from calib image_wh when present. Class 7 (sidewalk / verge) has no map-metric class and is not written.
"""
from pathlib import Path
B = Path(__file__).resolve().parent
s = (B / "sam3map_score.py").read_text(encoding="utf-8")


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


sub('import json, sys\n', 'import json, os, sys\n')
sub('import mapq_ours as mo  # noqa: E402\n',
    'import mapq_ours as mo  # noqa: E402\n'
    'sys.path.append(str(Path(__file__).resolve().parent)); sys.path.append(str(Path(__file__).resolve().parent.parent))   # camera_model beside / above\n'
    'import camera_model as CM  # noqa: E402\n')
sub('VIEWS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2"]\n',
    'VIEWS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2"]      # replaced in main() by the views the npz carries\n'
    'CALIB_ROOT = None                                                                  # SAM3MAP_ROOT (native sequences) or mo.V2\n\n\n'
    'def cameras(fd):\n'
    '    """The map\'s own views at this token, through camera_model (pinhole or f-theta), calib from CALIB_ROOT."""\n'
    '    cd = (CALIB_ROOT or mo.V2) / fd.parent.name / fd.name\n'
    '    c = np.load(cd / "calib.npz"); fr = json.loads((cd / "frame.json").read_text(encoding="utf-8"))\n'
    '    out = []\n'
    '    for cam in VIEWS:\n'
    '        i = fr["cam_order"].index(cam)\n'
    '        w, h = (int(c["image_wh"][i][0]), int(c["image_wh"][i][1])) if "image_wh" in c.files else (1920, 1080)\n'
    '        out.append(CM.Camera.from_calib(c, i, w, h))\n'
    '    return out\n')
sub('''    c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text(encoding="utf-8"))
    ii, jj = np.meshgrid(np.arange(200), np.arange(400), indexing="ij")''', '''    ii, jj = np.meshgrid(np.arange(200), np.arange(400), indexing="ij")''')
sub('''    Pg = np.c_[x, y, z]; seen = np.zeros(len(Pg), bool)
    for cam in VIEWS:
        i = fr["cam_order"].index(cam); K, R, t = c["cam_intrinsic"][i], c["sensor2lidar_rotation"][i], c["sensor2lidar_translation"][i]
        cc = (Pg - t) @ R
        front = cc[:, 2] > 0.5
        uv = (cc[front, :2] / cc[front, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] < 1080)
        seen[np.flatnonzero(front)[inside]] = True
    return seen.reshape(200, 400)''', '''    Pg = np.c_[x, y, z]
    return visible(fd, Pg).reshape(200, 400)''')
sub('''    c = np.load(fd / "calib.npz"); fr = json.loads((fd / "frame.json").read_text(encoding="utf-8"))
    seen = np.zeros(len(Pg), bool)
    for cam in VIEWS:
        i = fr["cam_order"].index(cam); K, R, t = c["cam_intrinsic"][i], c["sensor2lidar_rotation"][i], c["sensor2lidar_translation"][i]
        cc = (Pg - t) @ R
        front = cc[:, 2] > 0.5
        uv = (cc[front, :2] / cc[front, 2:3]) * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        inside = (uv[:, 0] >= 0) & (uv[:, 0] < 1920) & (uv[:, 1] >= 0) & (uv[:, 1] < 1080)
        seen[np.flatnonzero(front)[inside]] = True
    return seen''', '''    seen = np.zeros(len(Pg), bool)
    for C in cameras(fd):
        seen |= C.project_rig(Pg)[2]
    return seen''')
sub('''def main():
    c8 = sys.argv[1]; npz_dir = Path(sys.argv[2])
    files = sorted(npz_dir.glob("*.npz"))
    frames = [dict(np.load(f, allow_pickle=True)) for f in files]''', '''def main():
    global VIEWS, CALIB_ROOT
    c8 = sys.argv[1]; npz_dir = Path(sys.argv[2])
    files = sorted(npz_dir.glob("[0-9][0-9][0-9].npz"))
    frames = [dict(np.load(f, allow_pickle=True)) for f in files]
    VIEWS = sorted(k[4:] for k in frames[0] if k.startswith("cls_"))
    CALIB_ROOT = Path(os.environ["SAM3MAP_ROOT"]) if os.environ.get("SAM3MAP_ROOT") else None
    print(f"views {VIEWS}; calibration from {CALIB_ROOT or mo.V2}", flush=True)''')
(B / "sam3map_score.py").write_text(s, encoding="utf-8", newline="\n")
print("ok")
