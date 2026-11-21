"""Library for render_u400_video.py: geometry (DB poses + Caltech-distorted camera model), per-log data access (DB, raw camera zips, 20 Hz boxes,
nuPlan map), and the drawing primitives (front overlays, metric BEV, HUD).  Reuses render_final_video.py (R) for text, the BEV class, box
geometry, the turbo ramp and the quaternion helper.  No frame is ever synthesised: see render_u400_video.py for the video assembly.
"""
import argparse
import collections
import csv
import gzip
import json
import math
import multiprocessing as mp
import os
import sqlite3
import subprocess
import sys
import time
import zipfile

os.environ.setdefault("REFE_DRIVE", "E:")
import cv2
import numpy as np
import shapely
from shapely import STRtree

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(PKG, "raw", "2026-10-01-final-video"))
sys.path.insert(0, os.path.join(PKG, "refe"))
for _p in ("C:/Users/Admin/dz/DriveZero/DriveRL/nuplan-devkit", "C:/Users/Admin/dz/DriveZero/DriveRL/src"):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
import render_final_video as R   # noqa: E402
import calib_table as CT         # noqa: E402

DRV = os.environ["REFE_DRIVE"]
EXPORT = f"{DRV}/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
CSV = f"{DRV}/Projects/TanitAD/data/refe_navtest/score/refe_stage2_sft4_B_u400_navtest_full/refe_stage2_sft4_B_u400_navtest_full.csv"
DBDIR = f"{DRV}/Projects/TanitAD/data/nuplan/nuplan-v1.1/splits/test"
MINI = f"{DRV}/Projects/TanitAD/data/nuplan-camera/mini"
S4 = f"{DRV}/Projects/TanitAD/data/nuplan-camera/scenarios4"
FRAMES2HZ = f"{DRV}/Projects/TanitAD/data/refe_navtest/frames"
MAPS = f"{DRV}/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0"
PROPS = os.path.join(HERE, "u400_video_proposals.npz")
META = os.path.join(HERE, "video_meta.json")
RUNS = os.path.join(HERE, "video_runs.json")
SEG = os.path.join(HERE, "segments")
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")

W, H = 2560, 1440
FPS = 10
TOP = 40                                  # title bar
FRONT_XY = (0, TOP)                       # native 1920x1080 front camera
BEVP = (1920, TOP, 640, 1080)
PANELS = (("CAM_L0", "LEFT", 0), ("CAM_R0", "RIGHT", 480), ("CAM_B0", "REAR", 960))   # 480x270 each at y = 1150
PANEL_Y, PANEL_W, PANEL_H = 1150, 480, 270
HUD = (1440, 1120, 1120, 320)
TITLE = "REFe SFT-4 B u400  |  NAVSIM navtest PDMS 88.25 (full 12,146 tokens)  |  real camera frames"
CMDN = ["LEFT", "STRAIGHT", "RIGHT"]
V1_W = (5.0, 5.0, 2.0)
BEV_PPM, BEV_X0 = 12.0, -10.0
BEV_XMAX = BEV_X0 + BEVP[3] / BEV_PPM
BEV_YMAX = BEVP[2] / 2 / BEV_PPM
CAT_COL = {"vehicle": (225, 150, 70), "pedestrian": (70, 170, 255), "bicycle": (90, 225, 225), "traffic_cone": (60, 140, 255),
           "barrier": (130, 130, 140), "czone_sign": (130, 130, 140), "generic_object": (160, 160, 170)}
GREEN, WHITE, MAG = (90, 255, 90), (255, 255, 255), (255, 0, 255)


# ------------------------------------------------------------------------------------------------- geometry
class Pose:
    """one ego_pose row: rear-axle pose in the global frame (full 3-D)"""

    def __init__(self, row):
        x, y, z, qw, qx, qy, qz, vx, vy = [float(v) for v in row[:9]]
        self.T = np.array([x, y, z])
        self.Rm = R.quat_to_mat((qw, qx, qy, qz))
        self.yaw = math.atan2(self.Rm[1, 0], self.Rm[0, 0])
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        self.Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.speed = math.hypot(vx, vy)


class DBCam:
    """DB camera calibration: p_ego = Rm p_cam + t ; pixel = K . distort(p_cam / z) with the Caltech model (k1 k2 p1 p2 k3)"""

    def __init__(self, v):
        (self.fx, self.fy, self.cx, self.cy, self.k1, self.k2, self.p1, self.p2, self.k3, tx, ty, tz, qw, qx, qy, qz) = v
        self.R = R.quat_to_mat((qw, qx, qy, qz))
        self.t = np.array([tx, ty, tz])

    def to_cam(self, Pb):
        return (np.asarray(Pb, dtype=np.float64) - self.t) @ self.R

    def project(self, Pb, near=1.0, distort=True):
        """Pb (N,3) in the ego BODY frame -> (uv (N,2) in the native 1920x1080 image, ok (N,) bool)"""
        pc = self.to_cam(Pb)
        z = pc[:, 2]
        zs = np.maximum(z, 1e-6)
        x, y = pc[:, 0] / zs, pc[:, 1] / zs
        r2 = x * x + y * y
        if distort:
            rad = 1 + self.k1 * r2 + self.k2 * r2 ** 2 + self.k3 * r2 ** 3
            xd = x * rad + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
            yd = y * rad + self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y
        else:
            xd, yd = x, y
        uv = np.stack([self.fx * xd + self.cx, self.fy * yd + self.cy], 1)
        return uv, (z > near) & (r2 < 1.0)


def ground_to_body(pxy, pose_from, pose_to, pose3d=True):
    """ground points (N,2) in pose_from's yaw frame (z = ground under its rear axle) -> (N,3) in pose_to's BODY frame.
    The ONLY pose used is each frame's own high-rate ego_pose; no interpolation anywhere."""
    pxy = np.asarray(pxy, dtype=np.float64)
    c, s = math.cos(pose_from.yaw), math.sin(pose_from.yaw)
    G = np.stack([pose_from.T[0] + c * pxy[:, 0] - s * pxy[:, 1], pose_from.T[1] + s * pxy[:, 0] + c * pxy[:, 1],
                  np.full(len(pxy), pose_from.T[2])], 1)
    d = G - pose_to.T
    return d @ (pose_to.Rm if pose3d else pose_to.Rz)


def glob_to_ego2d(gxy, pose):
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    d = gxy - pose.T[None, :2]
    return np.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], 1)


def hermite(poses, per=8):
    """poses (8,3) x y heading on NAVSIM's 0.5 s grid (+ the origin) -> a smooth dense path (N,2) and per-point plan time (N,)"""
    pts = np.vstack([[0.0, 0.0, 0.0], np.asarray(poses, dtype=np.float64)])
    out, tt = [], []
    for i in range(len(pts) - 1):
        p0, p1 = pts[i, :2], pts[i + 1, :2]
        L = float(np.linalg.norm(p1 - p0))
        chord = (p1 - p0) / max(L, 1e-9)
        d0, d1 = np.array([math.cos(pts[i, 2]), math.sin(pts[i, 2])]), np.array([math.cos(pts[i + 1, 2]), math.sin(pts[i + 1, 2])])
        if L < 1e-3:
            d0 = d1 = chord
        else:
            if float(d0 @ chord) < 0.5:
                d0 = chord
            if float(d1 @ chord) < 0.5:
                d1 = chord
        t = np.linspace(0, 1, per, endpoint=(i == len(pts) - 2))
        h00, h10, h01, h11 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t, -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
        out.append(h00[:, None] * p0 + h10[:, None] * (d0 * L) + h01[:, None] * p1 + h11[:, None] * (d1 * L))
        tt.append(0.5 * (i + t))
    return np.vstack(out), np.concatenate(tt)


def ribbon_edges(path, half_w=0.95):
    return R.ribbon(path, half_w)


def agg_lane(logits):
    p = 1 / (1 + np.exp(-logits.astype(np.float64)))
    return p[..., 0] * p[..., 1] * (p[..., 2] * V1_W[0] + p[..., 3] * V1_W[1] + p[..., 4] * V1_W[2]) / sum(V1_W) * p[..., 6]


def runs_of(mask):
    """contiguous True runs of a bool array -> list of (i0, i1) inclusive"""
    out, i = [], 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def draw_poly3d(img, cam, Pb, col, th, near=1.5):
    uv, ok = cam.project(Pb, near=near)
    for a, b in runs_of(ok):
        if b > a:
            cv2.polylines(img, [np.round(uv[a:b + 1]).astype(np.int32)], False, col, th, cv2.LINE_AA)
    return uv, ok


FRUSTUM = ((np.array([0.0, 0.0, 1.0]), 0.8), (np.array([-1.0, 0.0, 0.95]), 0.0), (np.array([1.0, 0.0, 0.95]), 0.0),
           (np.array([0.0, -1.0, 0.95]), 0.0), (np.array([0.0, 1.0, 0.95]), 0.0))   # near plane + a +-43 deg pyramid (r < 0.95, where the lens polynomial is monotone)


def clip_ring_planes(pc, planes=FRUSTUM):
    """Sutherland-Hodgman of a closed ring (N,3) in the camera frame against half-spaces n.p > d (applied in sequence)"""
    pts = [np.asarray(p, dtype=np.float64) for p in pc]
    for n, d in planes:
        if not pts:
            break
        out = []
        m = len(pts)
        f = [float(p @ n - d) for p in pts]
        for i in range(m):
            a, b, fa, fb = pts[i], pts[(i + 1) % m], f[i], f[(i + 1) % m]
            if fa > 0:
                out.append(a)
            if (fa > 0) != (fb > 0):
                out.append(a + (fa / (fa - fb)) * (b - a))
        pts = out
    return np.array(pts) if len(pts) >= 3 else np.zeros((0, 3))


# ------------------------------------------------------------------------------------------------- per-log data
class LogData:
    def __init__(self, log, pose3d=True):
        self.log, self.pose3d = log, pose3d
        db = f"{DBDIR}/{log}.db"
        self.con = sqlite3.connect(f"file:{db}?mode=ro&immutable=1", uri=True)
        cal = CT.read_db(db, ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"))
        self.cam = {ch: DBCam(v) for ch, v in cal.items()}
        self.img = {}
        for ch in ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_B0"):
            rows = self.con.execute(
                "select i.timestamp, i.filename_jpg, e.x, e.y, e.z, e.qw, e.qx, e.qy, e.qz, e.vx, e.vy, e.timestamp from image i "
                "join camera c on c.token=i.camera_token join ego_pose e on e.token=i.ego_pose_token where c.channel=? order by i.timestamp", (ch,)).fetchall()
            self.img[ch] = {"ts": np.array([r[0] for r in rows], dtype=np.int64), "fn": [os.path.basename(r[1]) for r in rows],
                            "pose": np.array([r[2:11] for r in rows], dtype=np.float64), "ets": np.array([r[11] for r in rows], dtype=np.int64)}
        rows = self.con.execute("select l.token, l.timestamp, e.x, e.y, e.z, e.qw, e.qx, e.qy, e.qz, e.vx, e.vy from lidar_pc l "
                                "join ego_pose e on e.token=l.ego_pose_token order by l.timestamp").fetchall()
        self.lid_ts = np.array([r[1] for r in rows], dtype=np.int64)
        self.lid_tok = {bytes(r[0]).hex(): i for i, r in enumerate(rows)}
        self.lid_pose = np.array([r[2:11] for r in rows], dtype=np.float64)
        self.cats = {bytes(r[0]): r[1] for r in self.con.execute("select token, name from category")}
        self.zips = {}
        f0 = f"{MINI}/{log}_CAM_F0.zip"
        self.src = {"CAM_F0": (f0 if os.path.exists(f0) else f"{S4}/{log}_CAM_F0.zip", "real")}
        for ch in ("CAM_L0", "CAM_R0", "CAM_B0"):
            p = f"{S4}/{log}_{ch}.zip"
            self.src[ch] = (p, "real") if os.path.exists(p) else (f"{FRAMES2HZ}/{log}_{ch}.zip", "held")
        self.boxes = {}
        self.map = None

    def pose_img(self, ch, i):
        return Pose(self.img[ch]["pose"][i])

    def read_jpg(self, ch, name):
        zp = self.src[ch][0]
        if zp not in self.zips:
            self.zips[zp] = zipfile.ZipFile(zp)
        return cv2.imdecode(np.frombuffer(self.zips[zp].read(name), np.uint8), cv2.IMREAD_COLOR)

    def nearest(self, ch, t_us):
        ts = self.img[ch]["ts"]
        j = int(np.searchsorted(ts, t_us))
        j = min(max(j, 1), len(ts) - 1)
        return j - 1 if abs(int(ts[j - 1]) - t_us) <= abs(int(ts[j]) - t_us) else j

    def load_boxes(self, t0_us, t1_us):
        """DB lidar_box rows (20 Hz) for lidar_pc within [t0, t1] -> {lidar_idx: (N,8) x y yaw w l cat_id z height}"""
        cat_ids = {n: i for i, n in enumerate(CAT_COL)}
        q = ("select l.timestamp, b.x, b.y, b.yaw, b.width, b.length, t.category_token, b.z, b.height from lidar_pc l join lidar_box b on b.lidar_pc_token=l.token "
             "join track t on t.token=b.track_token where l.timestamp between ? and ? order by l.timestamp")
        by = collections.defaultdict(list)
        for ts, x, y, yaw, w, l, ct, z, h in self.con.execute(q, (int(t0_us), int(t1_us))):
            by[int(ts)].append((x, y, yaw, w, l, cat_ids.get(self.cats.get(bytes(ct), "generic_object"), 6), z, h))
        lt = {int(t): i for i, t in enumerate(self.lid_ts)}
        for ts, rows in by.items():
            self.boxes[lt[ts]] = np.array(rows, dtype=np.float64)

    def load_map(self, minx, miny, maxx, maxy, map_name):
        import pyogrio
        ver = sorted(os.listdir(f"{MAPS}/{map_name}"))[-1]
        gp = f"{MAPS}/{map_name}/{ver}/map.gpkg"
        # the gpkg is stored in lon/lat; nuPlan's own devkit reprojects to the `meta` projectedCoordSystem, which is what ego_pose/lidar_box use
        crs = str(pyogrio.read_dataframe(gp, layer="meta").set_index("key").loc["projectedCoordSystem", "value"])
        from pyproj import Transformer
        tf = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
        xs, ys = [minx - 130, maxx + 130], [miny - 130, maxy + 130]
        cx, cy = tf.transform([xs[0], xs[1], xs[0], xs[1]], [ys[0], ys[0], ys[1], ys[1]])
        bbox = (min(cx), min(cy), max(cx), max(cy))
        self.map = {}
        for key, layers in (("area", ("lane_groups_polygons", "intersections", "carpark_areas")), ("lanes", ("lanes_polygons", "gen_lane_connectors_scaled_width_polygons")),
                            ("cross", ("crosswalks",))):
            geoms = []
            for ly in layers:
                try:
                    df = pyogrio.read_dataframe(gp, layer=ly, bbox=bbox, columns=[]).to_crs(crs)
                    geoms += [g for g in df.geometry.values if g is not None and not g.is_empty]
                except Exception as e:     # noqa: BLE001  a missing layer is reported, never hidden
                    print(f"  WARNING map layer {ly}: {type(e).__name__}: {e}", flush=True)
            arr = np.array(geoms, dtype=object)
            self.map[key] = (arr, STRtree(arr) if len(arr) else None)


def view_clip(layer, pose, x0, x1, ymax):
    """geometries of a map layer clipped to the ego-frame box [x0,x1] x [-ymax,ymax], returned in the EGO yaw frame (list of shapely polys)"""
    arr, tree = layer
    if tree is None:
        return []
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    corners = np.array([[x0, -ymax], [x1, -ymax], [x1, ymax], [x0, ymax]])
    g = np.stack([pose.T[0] + c * corners[:, 0] - s * corners[:, 1], pose.T[1] + s * corners[:, 0] + c * corners[:, 1]], 1)
    box = shapely.Polygon(g)
    idx = tree.query(box, predicate="intersects")
    if len(idx) == 0:
        return []
    cl = shapely.intersection(arr[idx], box)
    M = np.array([[c, s], [-s, c]])
    off = -M @ pose.T[:2]
    ego = shapely.transform(cl, lambda p: p @ M.T + off)
    out = []
    for gm in ego:
        if gm.is_empty:
            continue
        out += [p for p in (gm.geoms if hasattr(gm, "geoms") else [gm]) if p.geom_type == "Polygon" and not p.is_empty]
    return out


# ------------------------------------------------------------------------------------------------- drawing
def bev_px(bev, xy):
    return np.stack([bev.x + bev.w / 2 - xy[:, 1] * bev.ppm, bev.y + bev.h - (xy[:, 0] - bev.x0) * bev.ppm], 1)


def fill_polys_bev(img, bev, polys, col, outline=None):
    for p in polys:
        ext = np.round(bev_px(bev, np.asarray(p.exterior.coords))).astype(np.int32)
        rings = [ext] + [np.round(bev_px(bev, np.asarray(r.coords))).astype(np.int32) for r in p.interiors]
        cv2.fillPoly(img, rings, col, cv2.LINE_AA)
        if outline is not None:
            cv2.polylines(img, rings, True, outline, 1, cv2.LINE_AA)


def draw_bev_base(canvas, bev, ld, pose, boxes):
    bx, by, bw, bh = BEVP
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (26, 26, 28), -1)
    if ld.map is not None:
        fill_polys_bev(canvas, bev, view_clip(ld.map["area"], pose, BEV_X0 - 2, BEV_XMAX + 2, BEV_YMAX + 2), (50, 50, 56))
        for p in view_clip(ld.map["lanes"], pose, BEV_X0 - 2, BEV_XMAX + 2, BEV_YMAX + 2):
            cv2.polylines(canvas, [np.round(bev_px(bev, np.asarray(p.exterior.coords))).astype(np.int32)], True, (74, 74, 84), 1, cv2.LINE_AA)
        fill_polys_bev(canvas, bev, view_clip(ld.map["cross"], pose, BEV_X0 - 2, BEV_XMAX + 2, BEV_YMAX + 2), (96, 96, 104))
    for r in range(10, int(BEV_XMAX), 10):
        a, b = bev.pt(r, -BEV_YMAX), bev.pt(r, BEV_YMAX)
        cv2.line(canvas, a, b, (46, 46, 52), 1)
        R.text(canvas, f"{r} m", (bx + 6, a[1] - 3), 0.4, (125, 125, 132), 1, False)
    cv2.line(canvas, bev.pt(BEV_X0, 0), bev.pt(BEV_XMAX, 0), (46, 46, 52), 1)
    if boxes is not None and len(boxes):
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        d = boxes[:, :2] - pose.T[None, :2]
        ex = c * d[:, 0] + s * d[:, 1]
        ey = -s * d[:, 0] + c * d[:, 1]
        keep = (ex > BEV_X0 - 6) & (ex < BEV_XMAX + 6) & (np.abs(ey) < BEV_YMAX + 6)
        names = list(CAT_COL)
        for b, x_, y_ in zip(boxes[keep], ex[keep], ey[keep]):
            col = CAT_COL[names[int(b[5])]]
            poly = bev.poly(R.box_corners(x_, y_, b[2] - pose.yaw, b[4], b[3]))
            cv2.fillPoly(canvas, [poly], col, cv2.LINE_AA)
            cv2.polylines(canvas, [poly], True, (20, 20, 22), 1, cv2.LINE_AA)


def draw_front(front, ld, cam, plan, pose_k, pose_c, pose3d, inp_marker):
    """plan: dict with 'paths' list of (N,2) in the plan frame k, 'nrm', 'order', 'pick', 'sel_edges', 'hum', 'ticks'"""
    ov = front.copy()
    # drivable area (NAVSIM's DAC polygons: lane groups + intersections + car parks) -- a labelled rendering aid
    if ld.map is not None:
        for p in view_clip(ld.map["area"], pose_c, 2.5, 60.0, 40.0):
            rings_uv = []
            for ring in [p.exterior] + list(p.interiors):
                xy = np.asarray(shapely.segmentize(shapely.LinearRing(ring.coords), 2.0).coords)[:-1]
                G = np.c_[xy, np.zeros(len(xy))]                                   # ground, in pose_c's yaw frame
                Pb = G @ pose_c.Rz.T @ (pose_c.Rm if pose3d else pose_c.Rz)         # -> pose_c's body frame
                rc = clip_ring_planes(cam.to_cam(Pb))
                if len(rc) < 3:
                    continue
                z = rc[:, 2]
                x, y = rc[:, 0] / z, rc[:, 1] / z
                r2 = x * x + y * y
                rad = 1 + cam.k1 * r2 + cam.k2 * r2 ** 2 + cam.k3 * r2 ** 3
                xd = x * rad + 2 * cam.p1 * x * y + cam.p2 * (r2 + 2 * x * x)
                yd = y * rad + cam.p1 * (r2 + 2 * y * y) + 2 * cam.p2 * x * y
                rings_uv.append(np.round(np.stack([cam.fx * xd + cam.cx, cam.fy * yd + cam.cy], 1)).astype(np.int32))
            if rings_uv:
                cv2.fillPoly(ov, rings_uv, (205, 190, 40), cv2.LINE_AA)
        front[:] = cv2.addWeighted(ov, 0.20, front, 0.80, 0)
    # selected ribbon
    ov = front.copy()
    le, re_ = plan["sel_edges"]
    Lb, Rb = ground_to_body(le, pose_k, pose_c, pose3d), ground_to_body(re_, pose_k, pose_c, pose3d)
    uvl, okl = cam.project(Lb, near=2.0)
    uvr, okr = cam.project(Rb, near=2.0)
    ok = okl & okr
    for i in range(len(le) - 1):
        if ok[i] and ok[i + 1]:
            cv2.fillPoly(ov, [np.round(np.array([uvl[i], uvl[i + 1], uvr[i + 1], uvr[i]])).astype(np.int32)], (80, 220, 90), cv2.LINE_AA)
    front[:] = cv2.addWeighted(ov, 0.30, front, 0.70, 0)
    # hypotheses, low -> high predicted PDMS
    for kk in plan["order"]:
        if kk != plan["pick"]:
            draw_poly3d(front, cam, ground_to_body(plan["paths"][kk], pose_k, pose_c, pose3d), R.turbo(0.15 + 0.85 * plan["nrm"][kk]), 3)
    draw_poly3d(front, cam, ground_to_body(plan["hum"], pose_k, pose_c, pose3d), (255, 255, 255), 4)
    sel = ground_to_body(plan["paths"][plan["pick"]], pose_k, pose_c, pose3d)
    draw_poly3d(front, cam, sel, (0, 0, 0), 11)
    draw_poly3d(front, cam, sel, GREEN, 6)
    uv, ok = cam.project(ground_to_body(plan["ticks"], pose_k, pose_c, pose3d), near=2.0)
    for j in range(len(uv)):
        if ok[j]:
            c = tuple(np.round(uv[j]).astype(int))
            cv2.circle(front, c, 9, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(front, c, 6, WHITE, -1, cv2.LINE_AA)
            if j % 2 == 1:
                R.text(front, f"{(j + 1) * 0.5:.0f} s", (c[0] + 12, c[1] - 8), 0.7, WHITE, 2)


def draw_bev_plan(canvas, bev, plan, pose_k, pose_c):
    def _g2(xy, pose):
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        xy = np.asarray(xy)
        return np.stack([pose.T[0] + c * xy[:, 0] - s * xy[:, 1], pose.T[1] + s * xy[:, 0] + c * xy[:, 1]], 1)

    def e(xy):
        return glob_to_ego2d(_g2(xy, pose_k), pose_c)

    for kk in plan["order"]:
        if kk != plan["pick"]:
            cv2.polylines(canvas, [bev.poly(e(plan["paths"][kk]))], False, R.turbo(0.15 + 0.85 * plan["nrm"][kk]), 1, cv2.LINE_AA)
    sel = bev.poly(e(plan["paths"][plan["pick"]]))
    cv2.polylines(canvas, [sel], False, (0, 0, 0), 8, cv2.LINE_AA)
    cv2.polylines(canvas, [bev.poly(e(plan["hum"]))], False, WHITE, 3, cv2.LINE_AA)
    cv2.polylines(canvas, [sel], False, GREEN, 4, cv2.LINE_AA)
    off = R.EGO_L / 2 - R.EGO_REAR
    car = bev.poly(R.box_corners(off, 0.0, 0.0, R.EGO_L, R.EGO_W))
    cv2.fillPoly(canvas, [car], GREEN, cv2.LINE_AA)
    cv2.polylines(canvas, [car], True, WHITE, 1, cv2.LINE_AA)
    for x_ in e(plan["ticks"]):
        cv2.circle(canvas, tuple(bev.pt(x_[0], x_[1])), 4, WHITE, -1, cv2.LINE_AA)


def bev_overlays_static(canvas, bev):
    bx, by, bw, bh = BEVP
    R.text(canvas, "BEV metric, ego frame (up = forward): agents + map = rendering aids", (bx + 8, by + 22), 0.46, (205, 205, 210), 1)
    # scale bar
    sx, sy = bx + bw - 80, by + bh - 28
    cv2.line(canvas, (sx, sy), (sx + int(10 * BEV_PPM), sy), WHITE, 3)
    cv2.line(canvas, (sx, sy - 5), (sx, sy + 5), WHITE, 2)
    cv2.line(canvas, (sx + int(10 * BEV_PPM), sy - 5), (sx + int(10 * BEV_PPM), sy + 5), WHITE, 2)
    R.text(canvas, "10 m", (sx + 22, sy - 9), 0.5, WHITE, 1)
    ly = by + bh - 88
    for j, (lab, col) in enumerate((("selected plan (u400)", GREEN), ("human future (logged)", WHITE))):
        cv2.line(canvas, (bx + 12, ly + 18 * j - 4), (bx + 44, ly + 18 * j - 4), col, 4, cv2.LINE_AA)
        R.text(canvas, lab, (bx + 52, ly + 18 * j), 0.45, (220, 220, 226), 1, False)
    for j, (nm, col) in enumerate((("vehicle", CAT_COL["vehicle"]), ("pedestrian", CAT_COL["pedestrian"]), ("bicycle", CAT_COL["bicycle"]), ("other", CAT_COL["generic_object"]))):
        xo = (12, 92, 190, 272)[j]
        cv2.rectangle(canvas, (bx + xo, ly + 44), (bx + xo + 14, ly + 56), col, -1)
        R.text(canvas, nm, (bx + xo + 20, ly + 54), 0.4, (210, 210, 216), 1, False)


def hud(canvas, d):
    hx, hy, hw, hh = HUD
    cv2.rectangle(canvas, (hx, hy), (hx + hw, hy + hh), (24, 24, 28), -1)
    x0, y0 = hx + 18, hy + 26
    R.text(canvas, "NAV COMMAND (agent input)", (x0, y0), 0.55, (200, 200, 205), 1, False)
    for j, nm in enumerate(CMDN):
        on = d["cmd"] == j
        xx = x0 + j * 150
        cv2.rectangle(canvas, (xx, y0 + 12), (xx + 140, y0 + 48), GREEN if on else (62, 62, 68), -1 if on else 1)
        R.text(canvas, nm, (xx + 12, y0 + 38), 0.6, (0, 0, 0) if on else (160, 160, 166), 2 if on else 1, False)
    R.text(canvas, f"ego speed {d['speed'] * 3.6:5.1f} km/h ({d['speed']:4.1f} m/s)", (x0, y0 + 80), 0.65)
    R.text(canvas, "speed: DB ego_pose of this very frame", (x0, y0 + 99), 0.4, (150, 150, 156), 1, False)
    R.text(canvas, f"token {d['token']}", (x0, y0 + 128), 0.62, (230, 230, 235))
    R.text(canvas, f"t = {d['t']:5.2f} s in clip    plan age {d['age']:.2f} s", (x0, y0 + 154), 0.58, (230, 230, 235))
    R.text(canvas, f"clip {d['run']}/{d['nruns']}   log {d['log'][:19]}..{d['log'][-11:]}", (x0, y0 + 180), 0.48, (170, 170, 176), 1, False)
    R.text(canvas, f"camera time {d['f0_ts_s']:.2f} s into the log   F0 jpg {d['f0_name']}", (x0, y0 + 200), 0.42, (140, 140, 146), 1, False)
    # scorer
    xm = hx + 530
    R.text(canvas, "SCORER (u400 head, predicted)", (xm, y0), 0.55, (200, 200, 205), 1, False)
    R.text(canvas, f"selected #{d['pick']}: predicted PDMS {100 * d['pred_sel']:5.1f}", (xm, y0 + 30), 0.62, GREEN)
    R.text(canvas, f"range of the 64: {100 * d['pred_min']:4.1f} .. {100 * d['pred_max']:4.1f}", (xm, y0 + 54), 0.5, (200, 200, 205), 1, False)
    for j in range(64):
        cv2.rectangle(canvas, (xm + int(j * 5.6), y0 + 64), (xm + int((j + 1) * 5.6), y0 + 78), R.turbo(0.15 + 0.85 * j / 63), -1)
    R.text(canvas, f"{100 * d['pred_min']:.0f}", (xm, y0 + 94), 0.42, (210, 210, 216), 1, False)
    R.text(canvas, f"{100 * d['pred_max']:.0f}", (xm + 335, y0 + 94), 0.42, (210, 210, 216), 1, False)
    R.text(canvas, "64 hypotheses: colour = predicted PDMS (this token)", (xm, y0 + 112), 0.42, (170, 170, 176), 1, False)
    # truth
    t = d["true"]
    R.text(canvas, "NAVSIM HARNESS (true score)", (xm, y0 + 146), 0.55, (200, 200, 205), 1, False)
    if t is None:
        R.text(canvas, "true per-token score: not banked", (xm, y0 + 178), 0.62, (150, 150, 255))
    else:
        bad = t["score"] < 0.5
        R.text(canvas, f"true PDMS {100 * t['score']:5.1f}" + ("   FAILURE" if bad else ""), (xm, y0 + 180), 0.8, (80, 80, 255) if bad else (230, 230, 235), 2)
        for j, nm in enumerate(("NC", "DAC", "EP", "TTC", "C", "DDC")):
            v = t["sub"][j]
            xx = xm + (j % 3) * 125
            yy = y0 + 196 + (j // 3) * 34
            cv2.rectangle(canvas, (xx, yy), (xx + 105, yy + 11), (62, 62, 68), 1)
            cv2.rectangle(canvas, (xx, yy), (xx + int(105 * v), yy + 11), GREEN if v > 0.99 else (60, 90, 255), -1)
            R.text(canvas, f"{nm} {v:.2f}", (xx, yy + 26), 0.45, (210, 210, 216), 1, False)
    # overlay key
    xk = hx + 930
    R.text(canvas, "KEY", (xk, y0), 0.55, (200, 200, 205), 1, False)
    for j, (lab, col) in enumerate((("selected plan", GREEN), ("human future", WHITE), ("64 hypotheses", (60, 160, 255)))):
        cv2.line(canvas, (xk, y0 + 20 + 22 * j), (xk + 30, y0 + 20 + 22 * j), col, 4, cv2.LINE_AA)
        R.text(canvas, lab, (xk + 38, y0 + 26 + 22 * j), 0.45, (220, 220, 226), 1, False)
    cv2.rectangle(canvas, (xk, y0 + 86), (xk + 30, y0 + 98), (205, 190, 40), -1)
    R.text(canvas, "drivable area", (xk + 38, y0 + 97), 0.45, (220, 220, 226), 1, False)
    R.text(canvas, "(map: rendering aid,", (xk, y0 + 118), 0.4, (170, 170, 176), 1, False)
    R.text(canvas, " not a model input)", (xk, y0 + 134), 0.4, (170, 170, 176), 1, False)
    # clip progress
    py = hy + hh - 22
    cv2.rectangle(canvas, (x0, py), (hx + hw - 20, py + 10), (62, 62, 68), 1)
    cv2.rectangle(canvas, (x0, py), (x0 + int((hx + hw - 20 - x0) * min(1.0, d["clip_prog"])), py + 10), GREEN, -1)
