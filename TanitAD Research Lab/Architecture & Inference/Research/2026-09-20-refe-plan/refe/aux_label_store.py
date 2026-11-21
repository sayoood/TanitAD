"""Auxiliary drivable-area label store (S0 of SPEC_AUX_DRIVABLE_HEAD.md, path B). CPU only, dev box.

LABEL (settled in SPEC_AUX_DRIVABLE_HEAD.md section 3; this module does not redesign it)
  * frame      : the rear-axle frame at t0 -- x forward, y LEFT -- the frame of every REFe pose and of NAVSIM's `rear_axle`.
  * polygons   : exactly `refe/navsim_dac.py:drivable_polygons` -- ROADBLOCK + INTERSECTION + CARPARK_AREA within 100 m of
                 the ego CENTRE (rear axle + 1.461 m along the heading). A footprint corner is drivable iff it is inside ANY
                 polygon, so a point-in-union raster is the exact label.
  * raster     : x in [-8, 72), y in [-32, 32), 0.5 m cells -> array [160, 128], index [ix, iy], cell centre
                 (X0 + (ix + .5) * RES, Y0 + (iy + .5) * RES).
  * stored     : the SIGNED DISTANCE to the drivable boundary, int8, unit 0.1 m, clipped to +-12.7 m. POSITIVE = inside the
                 drivable area, NEGATIVE = outside, boundary at 0 (half-cell corrected). |value| >= 2 LSB, so never 0.
                 mask = (sdf > 0). Computed on a grid padded by 13.2 m so a cell near the window edge sees boundaries that
                 lie outside the window.
  * key        : (log_name, token). Goal twins share it (train.py has no flip/crop/jitter).

BUILD PATH B (no NuPlanScenario): origin = the bank row's `origin_world` (rear-axle x, y, heading in the map frame) and the
log's map name. Where a token has no bank row but its nuPlan DB is held locally, the origin is read from the DB's own
`ego_pose` (rear axle) -- MEASURED identical to the bank's `origin_world` (0 m / 9e-16 rad over 33,704 rows).
The log -> map-name table is read from the DB where held, and otherwise INFERRED by `infer_map_name` (the ego centre lies in
a drivable polygon of exactly one map): MEASURED 462 / 462 agreement with the DB on the logs that have one.

STORE LAYOUT (one packed shard per log -- D: is exFAT with a ~1 MB allocation unit, so per-token files are not an option):
  <out>/shards/<log_name>.aux1    magic b"AUXSDF1\\n" | u32 LE header_len | header JSON | zlib(int8 [160,128] C-order) blobs
                                  header = {schema, log, map, raster{..}, entries:[{t:token, o:offset, l:len, s:"bank"|"db",
                                  p:[x,y,h], q:[targets_inside, targets_in_extent] | null}]}  (offsets from the blob region)
  <out>/manifest.json             counts + sha256 of EVERY shard + store_sha256 + coverage + provenance of the inputs
  <out>/run_meta_*.json  failures.jsonl  missing_tokens.txt  build.log (if redirected)

RESUME is by key: a shard is rewritten only when it lacks a requested token (or was built with another raster or another
map for that log); every written shard is atomic (tmp + os.replace).

Usage (driverl-eval venv: shapely + scipy + nuplan-devkit; export PYTHONIOENCODING=utf-8):
  python aux_label_store.py logmap   --bank B.jsonl --db-root R --maps-root M --out log_map.json
  python aux_label_store.py build    --bank B.jsonl --db-root R --maps-root M --yaml navtrain.yaml --navtest navtest.json \
                                     --logmap log_map.json --out STORE --workers 4 [--shard i/n]
  python aux_label_store.py finalize --out STORE
The build prints  `ZZAUXSTORE_DONE <stored_keys_in_store> <failed> new=<n> skipped=<n>`  as its LAST line.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

SCHEMA = "aux-drv-label-v1"
MAGIC = b"AUXSDF1\n"

# ---- the label spec (SPEC_AUX_DRIVABLE_HEAD.md section 3.1) ------------------------------------------------------
X0, X1, Y0, Y1, RES = -8.0, 72.0, -32.0, 32.0, 0.5            # rear-axle frame, x forward, y left
NX, NY = int(round((X1 - X0) / RES)), int(round((Y1 - Y0) / RES))                      # 160 x 128
SDF_CLIP_M, SDF_UNIT_M = 12.7, 0.1                             # int8: +-127 * 0.1 m
PAD_M = SDF_CLIP_M + 0.5                                       # the SDF is computed on a grid padded by this much (13.2 m)
MAP_RADIUS = 100.0                                             # == navsim_dac.MAP_RADIUS
REAR_AXLE_TO_CENTER = (4.049 - 1.127) / 2.0                    # == navsim_dac.REAR_AXLE_TO_CENTER (nuPlan Pacifica)
MAP_VERSION = "nuplan-maps-v1.0"
MAP_NAMES = ("us-ma-boston", "us-pa-pittsburgh-hazelwood", "us-nv-las-vegas-strip", "sg-one-north")
TARGET_POINTS = 20                                             # rank-0 bank target: 20 poses at 5 Hz


def raster_spec() -> dict:
    return dict(x0=X0, x1=X1, y0=Y0, y1=Y1, res_m=RES, nx=NX, ny=NY, sdf_clip_m=SDF_CLIP_M, sdf_unit_m=SDF_UNIT_M,
                pad_m=PAD_M, frame="rear_axle_t0 x_forward y_left", dtype="int8", index="[ix, iy] C-order",
                sign="positive=inside drivable, negative=outside", map_radius_m=MAP_RADIUS)


# ============================================================ raster core (numpy + shapely + scipy only) ===========
def grid_centres(pad_m: float = 0.0):
    p = int(round(pad_m / RES))
    ix = np.arange(-p, NX + p)
    iy = np.arange(-p, NY + p)
    x = X0 + (ix + 0.5) * RES
    y = Y0 + (iy + 0.5) * RES
    return np.meshgrid(x, y, indexing="ij")                                  # [NX+2p, NY+2p]


def to_global(x, y, pose):
    """Rear-axle-frame points -> map frame. pose = (x, y, heading) of the rear axle."""
    x0, y0, h0 = pose
    c, s = math.cos(h0), math.sin(h0)
    return x0 + x * c - y * s, y0 + x * s + y * c


def rasterize(polys, gx, gy) -> np.ndarray:
    """Point-in-union at every grid centre: the SAME test NAVSIM applies to each footprint corner (strict `contains`)."""
    import shapely
    fx, fy = gx.ravel(), gy.ravel()
    box = shapely.box(fx.min(), fy.min(), fx.max(), fy.max())
    flat = np.zeros(fx.shape, dtype=bool)
    for p in polys:
        if not shapely.intersects(box, p):
            continue
        flat |= shapely.contains_xy(p, fx, fy)
    return flat.reshape(gx.shape)


def signed_distance(inside_pad: np.ndarray) -> np.ndarray:
    """Metres, + inside / - outside, boundary at 0 (half-cell corrected), on the padded raster.

    DEGENERATE GRIDS. `scipy.ndimage.distance_transform_edt` of an array with NO background measures the distance to the
    array BORDER (MEASURED: edt(np.ones((5, 6))) holds 1.0 ... 7.07). With PAD_M >= SDF_CLIP_M every window cell is then at least
    a clip away from the border, so the prototype's unguarded code already read +-127 for a window with no drivable cell (or all
    drivable): the 0.5 m in PAD_M = CLIP + 0.5 is what makes that true. The explicit branches below make it independent of the
    pad (test_degenerate_guard_is_equivalent_at_current_pad pins both facts); every non-degenerate output is unchanged.
    """
    from scipy.ndimage import distance_transform_edt as edt
    if not inside_pad.any():
        return np.full(inside_pad.shape, -SDF_CLIP_M)
    if inside_pad.all():
        return np.full(inside_pad.shape, SDF_CLIP_M)
    d_in = edt(inside_pad)            # inside cells: distance (cells) to the nearest outside cell
    d_out = edt(~inside_pad)
    return np.where(inside_pad, d_in - 0.5, -(d_out - 0.5)) * RES


def quantize_sdf(sdf_m: np.ndarray) -> np.ndarray:
    """metres -> int8 in units of SDF_UNIT_M, clipped to +-SDF_CLIP_M. Refuses a clip that would wrap int8."""
    if SDF_CLIP_M / SDF_UNIT_M > 127.0 + 1e-9:
        raise ValueError(f"clip {SDF_CLIP_M} m / unit {SDF_UNIT_M} m = {SDF_CLIP_M / SDF_UNIT_M:.1f} LSB does not fit int8")
    return np.round(np.clip(sdf_m, -SDF_CLIP_M, SDF_CLIP_M) / SDF_UNIT_M).astype(np.int8)


def build_label(polys, pose):
    """-> (mask [NX, NY] bool, sdf_int8 [NX, NY]) in the rear-axle frame, from map polygons + the rear-axle pose."""
    px, py = grid_centres(PAD_M)
    gx, gy = to_global(px, py, pose)
    inside_pad = rasterize(polys, gx, gy)
    sdf = signed_distance(inside_pad)
    p = int(round(PAD_M / RES))
    return inside_pad[p:p + NX, p:p + NY].copy(), quantize_sdf(sdf[p:p + NX, p:p + NY])


def ego_centre(pose):
    x, y, h = pose
    return x + REAR_AXLE_TO_CENTER * math.cos(h), y + REAR_AXLE_TO_CENTER * math.sin(h)


def resolve_label(backend, logmap: dict, log: str, pose):
    """The label of one frame: polygons of THIS LOG's map around the ego centre, rasterised in the rear-axle frame."""
    map_name = logmap[log]["map"] if isinstance(logmap[log], dict) else logmap[log]       # KeyError = fail loud
    cx, cy = ego_centre(pose)
    return build_label(backend.polys(map_name, cx, cy), pose)


def target_probe(sdf_i8: np.ndarray, xy) -> tuple[int, int]:
    """QC: of the bank's rank-0 target points (ego frame), how many fall inside the extent and how many inside the label."""
    xy = np.asarray(xy, dtype=np.float64)
    fx = (xy[:, 0] - X0) / RES - 0.5
    fy = (xy[:, 1] - Y0) / RES - 0.5
    ix, iy = np.rint(fx).astype(int), np.rint(fy).astype(int)
    ok = (ix >= 0) & (ix < NX) & (iy >= 0) & (iy < NY)
    inside = int((sdf_i8[ix[ok], iy[ok]] > 0).sum())
    return inside, int(ok.sum())


def yaw_from_quaternion(qw, qx, qy, qz) -> float:
    """pyquaternion's `yaw_pitch_roll[0]` (what nuPlan's devkit calls to build the ego heading) -- NOT the textbook ZYX formula:
    pyquaternion has (q0 q3 - q1 q2). The textbook sign differs whenever roll/pitch are non-zero. Normalises like pyquaternion."""
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return math.atan2(2.0 * (qw * qz - qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


# ============================================================ map backends =========================================
class NuplanBackend:
    """Drivable polygons from the nuPlan map GPKGs, exactly `navsim_dac.drivable_polygons` (from a bare centre point)."""

    def __init__(self, maps_root: str, version: str = MAP_VERSION):
        self.maps_root, self.version, self._apis = maps_root, version, {}

    def api(self, map_name: str):
        if map_name not in self._apis:
            from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
            self._apis[map_name] = get_maps_api(self.maps_root, self.version, map_name)
        return self._apis[map_name]

    def polys(self, map_name: str, cx: float, cy: float, radius: float = MAP_RADIUS) -> list:
        from nuplan.common.actor_state.state_representation import Point2D
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer as L
        layers = [L.ROADBLOCK, L.ROADBLOCK_CONNECTOR, L.INTERSECTION, L.CARPARK_AREA]
        objs = self.api(map_name).get_proximal_map_objects(Point2D(cx, cy), radius, layers)
        return [o.polygon for layer in (L.ROADBLOCK, L.INTERSECTION, L.CARPARK_AREA) for o in objs[layer]]


def make_backend(spec: str, maps_root: str):
    if spec == "nuplan":
        return NuplanBackend(maps_root)
    mod, _, fn = spec.partition(":")                               # "module:factory" -> factory(maps_root) (tests use this)
    import importlib
    return getattr(importlib.import_module(mod), fn)(maps_root)


def infer_map_name(origins, backend, map_names=MAP_NAMES) -> dict:
    """Which map does a log live on? score[m] = share of the log's sampled ego centres that lie inside >= 1 drivable polygon
    of map m. nuPlan's four maps use disjoint coordinate regions, so a wrong map scores 0. Returns
    {"map": best|None, "scores": {...}, "ok": bool}; ok = best >= 0.6 and every other map <= 0.2 (else AMBIGUOUS -> not used)."""
    import shapely
    scores = {}
    for m in map_names:
        hit = 0
        for pose in origins:
            cx, cy = ego_centre(pose)
            hit += any(bool(shapely.contains_xy(p, cx, cy)) for p in backend.polys(m, cx, cy))
        scores[m] = hit / len(origins)
    best = max(scores, key=scores.get)
    rest = max(v for k, v in scores.items() if k != best)
    ok = scores[best] >= 0.6 and rest <= 0.2
    return {"map": best if ok else None, "scores": scores, "ok": bool(ok)}


# ============================================================ shard format =========================================
def _pack_header(log: str, map_name: str, entries: list) -> bytes:
    h = {"schema": SCHEMA, "log": log, "map": map_name, "raster": raster_spec(), "entries": entries}
    return json.dumps(h, separators=(",", ":")).encode("utf-8")


def write_shard(path: Path, log: str, map_name: str, items: list) -> tuple[int, str]:
    """items: [{t, blob, s, p, q}] -> writes <path> atomically; returns (bytes, sha256)."""
    entries, blobs, off = [], [], 0
    for it in sorted(items, key=lambda i: i["t"]):
        entries.append({"t": it["t"], "o": off, "l": len(it["blob"]), "s": it["s"], "p": it["p"], "q": it.get("q")})
        blobs.append(it["blob"])
        off += len(it["blob"])
    head = _pack_header(log, map_name, entries)
    data = MAGIC + struct.pack("<I", len(head)) + head + b"".join(blobs)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)
    return len(data), hashlib.sha256(data).hexdigest()


def read_shard(path: Path, with_blobs: bool = True) -> tuple[dict, dict]:
    """-> (header, {token: zlib blob}). Raises ValueError on a bad magic / truncated file."""
    with open(path, "rb") as f:
        if f.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"{path}: bad magic")
        (hl,) = struct.unpack("<I", f.read(4))
        head = json.loads(f.read(hl).decode("utf-8"))
        blobs = {}
        if with_blobs:
            base = len(MAGIC) + 4 + hl
            for e in head["entries"]:
                f.seek(base + e["o"])
                b = f.read(e["l"])
                if len(b) != e["l"]:
                    raise ValueError(f"{path}: truncated blob for {e['t']}")
                blobs[e["t"]] = b
    return head, blobs


def encode_sdf(sdf_i8: np.ndarray) -> bytes:
    if sdf_i8.dtype != np.int8 or sdf_i8.shape != (NX, NY):
        raise ValueError(f"expected int8 [{NX}, {NY}], got {sdf_i8.dtype} {sdf_i8.shape}")
    return zlib.compress(np.ascontiguousarray(sdf_i8).tobytes(), 6)


def decode_sdf(blob: bytes) -> np.ndarray:
    a = np.frombuffer(zlib.decompress(blob), dtype=np.int8)
    if a.size != NX * NY:
        raise ValueError(f"decoded {a.size} cells, expected {NX * NY}")
    return a.reshape(NX, NY)


def read_label(store: Path, log: str, token: str) -> np.ndarray:
    _, blobs = read_shard(Path(store) / "shards" / f"{log}.aux1")
    return decode_sdf(blobs[token])


def shard_matches_spec(head: dict, map_name: str) -> bool:
    return head.get("schema") == SCHEMA and head.get("raster") == raster_spec() and head.get("map") == map_name


# ============================================================ worker: one log -> one shard =========================
_CFG: dict = {}
_BACKEND = None


def _init_worker(cfg: dict):
    global _CFG, _BACKEND
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _CFG = cfg
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)   # polite to the neighbours
    except Exception:
        pass
    _BACKEND = make_backend(cfg["backend"], cfg["maps_root"])


def _db_origins(db_path: str) -> dict:
    """{token: (x, y, heading)} of every lidar_pc row of a nuPlan log DB (rear-axle ego_pose, devkit heading)."""
    import sqlite3
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT lower(hex(lp.token)), ep.x, ep.y, ep.qw, ep.qx, ep.qy, ep.qz "
                           "FROM lidar_pc AS lp JOIN ego_pose AS ep ON lp.ego_pose_token = ep.token").fetchall()
    finally:
        con.close()
    return {t: (x, y, yaw_from_quaternion(qw, qx, qy, qz)) for t, x, y, qw, qx, qy, qz in rows}


def process_log(task: dict) -> dict:
    """task = {log, map, bank: {token: (x, y, h, traj_xy|None)}, db: path|None}. Writes/extends <out>/shards/<log>.aux1."""
    t0 = time.perf_counter()
    cfg, backend = _CFG, _BACKEND
    log, map_name = task["log"], task["map"]
    out = Path(cfg["out"]) / "shards" / f"{log}.aux1"
    yaml_toks, nt_toks = cfg.get("yaml_tokens"), cfg.get("navtest_tokens") or frozenset()
    res = dict(log=log, map=map_name, new=0, skipped=0, failed=[], excluded_navtest=0, not_in_yaml=0, from_bank=0, from_db=0,
               xcheck_n=0, xcheck_max_m=0.0, xcheck_max_rad=0.0, file=None, bytes=0, sha256=None, rebuilt=False)
    # ---- the keys wanted for this log: bank rows first, then DB-held yaml tokens that have no bank row
    want = {}
    for tok, (x, y, h, traj) in task["bank"].items():
        want[tok] = ((x, y, h), "bank", traj)
    if task.get("db"):
        dbo = _db_origins(task["db"])
        for tok, (x, y, h, _tr) in task["bank"].items():              # origin cross-check, bank vs the DB's own ego_pose
            if tok in dbo:
                d = dbo[tok]
                res["xcheck_n"] += 1
                res["xcheck_max_m"] = max(res["xcheck_max_m"], math.hypot(d[0] - x, d[1] - y))
                res["xcheck_max_rad"] = max(res["xcheck_max_rad"], abs((d[2] - h + math.pi) % (2 * math.pi) - math.pi))
        if yaml_toks is not None:
            for tok, pose in dbo.items():
                if tok in yaml_toks and tok not in want:
                    want[tok] = (pose, "db", None)
    for tok in list(want):
        if tok in nt_toks:                                              # NEVER label a navtest frame
            res["excluded_navtest"] += 1
            del want[tok]
        elif yaml_toks is not None and tok not in yaml_toks:
            res["not_in_yaml"] += 1
            del want[tok]
    # ---- resume by key
    have = {}
    if out.exists():
        try:
            head, blobs = read_shard(out)
            if shard_matches_spec(head, map_name):
                meta = {e["t"]: e for e in head["entries"]}
                have = {t: dict(t=t, blob=blobs[t], s=meta[t]["s"], p=meta[t]["p"], q=meta[t].get("q")) for t in blobs}
            else:
                res["rebuilt"] = True                                    # other raster / other map for this log: recompute all
        except Exception:
            res["rebuilt"] = True
    dirty = False
    for t in [t for t in have if t in nt_toks]:                       # a stale shard must not carry a navtest frame
        del have[t]
        res["excluded_navtest"] += 1
        dirty = True
    todo = [t for t in want if t not in have]
    res["skipped"] = len(want) - len(todo)
    new_items = []
    for tok in todo:
        pose, src, traj = want[tok]
        try:
            _mask, sdf = resolve_label(backend, {log: {"map": map_name}}, log, pose)
            q = list(target_probe(sdf, traj)) if traj is not None else None
            new_items.append(dict(t=tok, blob=encode_sdf(sdf), s=src, p=[float(pose[0]), float(pose[1]), float(pose[2])], q=q))
            res["from_" + src] += 1
        except Exception as e:                                          # one bad frame must not lose the log
            res["failed"].append([tok, f"{type(e).__name__}: {e}"[:300]])
    res["new"] = len(new_items)
    if new_items or dirty:
        keep = dict(have)                                               # existing entries are kept (additive)
        for it in new_items:
            keep[it["t"]] = it
        if keep:
            res["bytes"], res["sha256"] = write_shard(out, log, map_name, list(keep.values()))
            res["file"] = out.name
        elif out.exists():
            out.unlink()
    elif res["rebuilt"] and out.exists():
        out.unlink()                                                    # untrustworthy (other map / raster) and nothing to replace it
    elif out.exists():
        res["file"] = out.name
    res["secs"] = time.perf_counter() - t0
    return res


# ============================================================ main-process helpers =================================
def free_physical_gb() -> float:
    if os.name == "nt":
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong), ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullAvailPhys / 2 ** 30
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


def sha256_file(path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_bank(paths, with_traj: bool = True) -> dict:
    """{log: {token: (x, y, h, traj_xy float32 [20,2] | None)}} -- only the fields the builder needs (the files are ~350 MB)."""
    out: dict = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                ow = r["origin_world"]
                tr = np.asarray([[q[0], q[1]] for q in r["traj"]][:TARGET_POINTS], dtype=np.float32) if (with_traj and r.get("traj")) else None
                out.setdefault(r["log_name"], {}).setdefault(r["token"], (float(ow[0]), float(ow[1]), float(ow[2]), tr))
    return out


def load_yaml_tokens(path) -> tuple[list, set]:
    import yaml
    with open(path, encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
    toks = list(cfg["tokens"] or [])
    bad = [t for t in toks if not isinstance(t, str)]
    if bad:                                  # an unquoted all-digit token is parsed as an int by YAML: its leading zeros are LOST
        raise TypeError(f"{path}: {len(bad)} tokens are not strings (e.g. {bad[0]!r}); quote them in the yaml")
    return list(cfg["log_names"] or []), {t.lower() for t in toks}


def load_navtest(path) -> tuple[set, set]:
    """(tokens, logs) of navtest from a json holding `token_to_log_name` (the repo's cluster_maps/navtest.json) or `tokens`+`token_log`."""
    d = json.load(open(path, encoding="utf-8"))
    t2l = d.get("token_to_log_name")
    if t2l is None:
        t2l = dict(zip(d["tokens"], d["token_log"])) if isinstance(d.get("token_log"), list) else d["token_log"]
    return {t.lower() for t in t2l}, set(t2l.values())


def index_dbs(db_root: str) -> dict:
    out = {}
    for pat in ("*.db", "trainval/*.db", "mini/*.db"):
        for p in glob.glob(os.path.join(db_root, pat)):
            out.setdefault(os.path.basename(p)[:-3], os.path.abspath(p))
    return out


def db_map_name(db_path: str) -> str:
    import sqlite3
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return con.execute("SELECT map_version FROM log").fetchone()[0]
    finally:
        con.close()


# ============================================================ subcommand: logmap ===================================
def cmd_logmap(a) -> int:
    bank = load_bank(a.bank, with_traj=False)
    dbs = index_dbs(a.db_root) if a.db_root else {}
    backend = make_backend(a.backend, a.maps_root)
    table, t0 = {}, time.time()
    if a.yaml:                                   # DB-held navtrain logs with no bank row: the DB is the only (authoritative) source
        ylogs, _ = load_yaml_tokens(a.yaml)
        for log in sorted(set(ylogs) & set(dbs) - set(bank)):
            table[log] = {"inferred": None, "scores": None, "k": 0, "map": db_map_name(dbs[log]), "src": "db", "infer_agrees": None}
    for i, log in enumerate(sorted(bank)):
        rows = sorted(bank[log])
        idx = sorted(set(np.linspace(0, len(rows) - 1, min(a.k, len(rows))).round().astype(int)))
        inf = infer_map_name([bank[log][rows[j]][:3] for j in idx], backend)
        ent = {"inferred": inf["map"], "scores": inf["scores"], "k": len(idx)}
        if log in dbs:
            ent.update(map=db_map_name(dbs[log]), src="db", infer_agrees=bool(inf["map"] == db_map_name(dbs[log])))
        elif inf["ok"]:
            ent.update(map=inf["map"], src="inferred")
        else:
            ent.update(map=None, src="ambiguous")
        table[log] = ent
        if i % 200 == 0:
            print(f"[logmap {i}/{len(bank)}] {time.time() - t0:.0f}s", flush=True)
    known = [v for v in table.values() if v["src"] == "db" and v["infer_agrees"] is not None]
    summ = dict(logs=len(table), from_db=sum(v["src"] == "db" for v in table.values()), inferred=sum(v["src"] == "inferred" for v in table.values()),
                ambiguous=sum(v["src"] == "ambiguous" for v in table.values()),
                infer_vs_db_agree=sum(v["infer_agrees"] for v in known), infer_vs_db_n=len(known),
                inferred_min_best_score=min([max(v["scores"].values()) for v in table.values() if v["src"] == "inferred"], default=None),
                inferred_max_runner_up=max([sorted(v["scores"].values())[-2] for v in table.values() if v["src"] == "inferred"], default=None))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"schema": "aux-drv-logmap-v1", "summary": summ, "logs": table}, indent=1), encoding="utf-8")
    print(json.dumps(summ))
    print(f"ZZAUXLOGMAP_DONE {summ['logs']} {summ['ambiguous']}")
    return 0 if summ["ambiguous"] == 0 and summ["infer_vs_db_agree"] == summ["infer_vs_db_n"] else 2


# ============================================================ subcommand: build ====================================
def parse_shard(s: str | None):
    if not s:
        return 0, 1
    i, n = s.split("/")
    i, n = int(i), int(n)
    if not (n >= 1 and 0 <= i < n):
        raise SystemExit(f"--shard {s}: need 0 <= i < n")
    return i, n


def cmd_build(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    si, sn = parse_shard(a.shard)
    free = free_physical_gb()
    if a.workers > 1 and free < a.min_free_gb:
        print(f"REFUSED: free physical memory {free:.1f} GB < --min-free-gb {a.min_free_gb} (shared host); resumable, retry later")
        return 3
    logmap = json.load(open(a.logmap, encoding="utf-8"))["logs"]
    bank = load_bank(a.bank)
    dbs = index_dbs(a.db_root) if a.db_root else {}
    yaml_toks = None
    yaml_n = None
    if a.yaml:
        _logs, yaml_toks = load_yaml_tokens(a.yaml)
        yaml_n = len(yaml_toks)
    nt_toks, nt_logs = load_navtest(a.navtest) if a.navtest else (set(), set())
    # DB-only logs (no bank row) still contribute their yaml tokens, when --yaml enables the DB-derived extras
    logs = sorted(set(bank) | ({l for l in dbs if l in logmap} if yaml_toks is not None else set()))
    skipped_unmapped = [l for l in logs if not (logmap.get(l) or {}).get("map")]
    logs = [l for l in logs if l not in skipped_unmapped]
    excluded_logs = [l for l in logs if l in nt_logs]
    logs = [l for l in logs if l not in nt_logs]
    for l in nt_logs:                                            # a shard of a navtest log left by an earlier run must go
        (out / "shards" / f"{l}.aux1").unlink(missing_ok=True)
    logs = [l for k, l in enumerate(logs) if k % sn == si]
    if a.limit_logs:
        logs = logs[:a.limit_logs]
    tasks = [dict(log=l, map=logmap[l]["map"], bank=bank.get(l, {}), db=dbs.get(l)) for l in logs]
    tasks.sort(key=lambda t: -len(t["bank"]))
    cfg = dict(out=str(out), maps_root=a.maps_root, backend=a.backend, yaml_tokens=frozenset(yaml_toks) if yaml_toks is not None else None,
               navtest_tokens=frozenset(nt_toks))
    tag = "all" if sn == 1 else f"{si}-of-{sn}"
    meta = dict(schema=SCHEMA, tag=tag, argv=sys.argv, started=time.strftime("%Y-%m-%d %H:%M:%S"), raster=raster_spec(),
                inputs=dict(bank=[dict(path=p, sha256=sha256_file(p), bytes=os.path.getsize(p)) for p in a.bank],
                            logmap=dict(path=a.logmap, sha256=sha256_file(a.logmap)),
                            yaml=dict(path=a.yaml, sha256=sha256_file(a.yaml), tokens=yaml_n) if a.yaml else None,
                            navtest=dict(path=a.navtest, sha256=sha256_file(a.navtest), tokens=len(nt_toks), logs=len(nt_logs)) if a.navtest else None,
                            db_root=a.db_root, db_logs_held=len(dbs), maps_root=a.maps_root,
                            maps_json_sha256=sha256_file(os.path.join(a.maps_root, "nuplan-maps-v1.0.json"))
                            if os.path.exists(os.path.join(a.maps_root, "nuplan-maps-v1.0.json")) else None,
                            code_sha256=sha256_file(__file__)),
                logs_in_run=len(tasks), bank_logs=len(bank), bank_tokens=sum(len(v) for v in bank.values()),
                unmapped_logs_skipped=skipped_unmapped, navtest_logs_excluded=excluded_logs, workers=a.workers)
    (out / f"run_meta_{tag}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"build: {len(tasks)} logs ({tag}), bank {meta['bank_tokens']} tokens / {len(bank)} logs, db held {len(dbs)}, "
          f"free mem {free:.1f} GB, workers {a.workers}", flush=True)

    totals = dict(new=0, skipped=0, failed=0, excluded_navtest=0, not_in_yaml=0, from_bank=0, from_db=0, xcheck_n=0)
    xmax_m = xmax_rad = 0.0
    fails = []
    t0 = time.time()

    def absorb(r, k):
        nonlocal xmax_m, xmax_rad
        for key in ("new", "skipped", "excluded_navtest", "not_in_yaml", "from_bank", "from_db", "xcheck_n"):
            totals[key] += r[key]
        totals["failed"] += len(r["failed"])
        xmax_m, xmax_rad = max(xmax_m, r["xcheck_max_m"]), max(xmax_rad, r["xcheck_max_rad"])
        fails.extend({"log": r["log"], "token": t, "error": e} for t, e in r["failed"])
        if k % 25 == 0 or k == len(tasks):
            el = time.time() - t0
            print(f"[{k}/{len(tasks)}] new {totals['new']} skipped {totals['skipped']} failed {totals['failed']} "
                  f"elapsed {el:.0f}s eta {el / k * (len(tasks) - k):.0f}s free {free_physical_gb():.1f} GB", flush=True)

    aborted = False
    if a.workers <= 1:
        _init_worker(cfg)
        for k, t in enumerate(tasks, 1):
            absorb(process_log(t), k)
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(a.workers, initializer=_init_worker, initargs=(cfg,))
        try:
            for k, r in enumerate(pool.imap_unordered(process_log, tasks, chunksize=1), 1):
                absorb(r, k)
                if free_physical_gb() < a.abort_free_gb:
                    print(f"ABORT: free physical memory {free_physical_gb():.1f} GB < --abort-free-gb {a.abort_free_gb}; resumable", flush=True)
                    aborted = True
                    break
            pool.close()
        finally:
            pool.terminate()
            pool.join()
    with open(out / f"failures_{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in fails:
            f.write(json.dumps(r) + "\n")
    meta.update(finished=time.strftime("%Y-%m-%d %H:%M:%S"), totals=totals, origin_xcheck=dict(n=totals["xcheck_n"], max_m=xmax_m, max_rad=xmax_rad),
                seconds=time.time() - t0, aborted=aborted)
    (out / f"run_meta_{tag}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    stored = 0
    if sn == 1 and not aborted:
        man = finalize(out)
        stored = man["counts"]["stored_keys"]
        print(f"manifest: stored {stored} keys in {man['counts']['logs']} shards, {man['counts']['bytes']} bytes, store_sha256 {man['store_sha256'][:16]}")
    else:
        stored = sum(len(read_shard(p, with_blobs=False)[0]["entries"]) for p in glob.glob(str(out / "shards" / "*.aux1")))
        print(f"shard run {tag} complete{' (ABORTED)' if aborted else ''}; run `finalize` once every shard run is done")
    print(f"ZZAUXSTORE_DONE {stored} {totals['failed']} new={totals['new']} skipped={totals['skipped']}"
          + (" ABORTED" if aborted else ""))
    return 4 if aborted else 0


# ============================================================ subcommand: finalize =================================
def finalize(out) -> dict:
    out = Path(out)
    files, stored, by_src, by_map, tin_frac = {}, 0, {"bank": 0, "db": 0}, {}, []
    keys_by_log = {}
    for p in sorted(glob.glob(str(out / "shards" / "*.aux1"))):
        head, _ = read_shard(Path(p), with_blobs=False)
        n = len(head["entries"])
        rel = "shards/" + os.path.basename(p)
        files[rel] = {"bytes": os.path.getsize(p), "sha256": sha256_file(p), "n": n, "map": head["map"]}
        stored += n
        by_map[head["map"]] = by_map.get(head["map"], 0) + n
        keys_by_log[head["log"]] = [e["t"] for e in head["entries"]]
        for e in head["entries"]:
            by_src[e["s"]] = by_src.get(e["s"], 0) + 1
            if e.get("q") and e["q"][1] > 0:
                tin_frac.append(e["q"][0] / e["q"][1])
    runs = []
    for p in sorted(glob.glob(str(out / "run_meta_*.json"))):
        runs.append(json.load(open(p, encoding="utf-8")))
    cov = None
    ymeta = next((r["inputs"]["yaml"] for r in runs if r.get("inputs", {}).get("yaml")), None)
    nmeta = next((r["inputs"]["navtest"] for r in runs if r.get("inputs", {}).get("navtest")), None)
    if ymeta and os.path.exists(ymeta["path"]):
        _l, ytoks = load_yaml_tokens(ymeta["path"])
        have = {t for ts in keys_by_log.values() for t in ts}
        miss = sorted(ytoks - have)
        (out / "missing_tokens.txt").write_text("\n".join(miss) + ("\n" if miss else ""), encoding="utf-8")
        cov = dict(yaml_tokens=len(ytoks), stored_in_yaml=len(have & ytoks), stored_not_in_yaml=len(have - ytoks),
                   missing=len(miss), coverage_frac=len(have & ytoks) / len(ytoks))
    leak = None
    if nmeta and os.path.exists(nmeta["path"]):
        nt_toks, nt_logs = load_navtest(nmeta["path"])
        have = {t for ts in keys_by_log.values() for t in ts}
        leak = dict(navtest_tokens=len(nt_toks), token_intersection=len(have & nt_toks), log_intersection=len(set(keys_by_log) & nt_logs))
    tf = np.asarray(tin_frac) if tin_frac else np.zeros(0)
    man = dict(schema=SCHEMA, written=time.strftime("%Y-%m-%d %H:%M:%S"), raster=raster_spec(),
               counts=dict(stored_keys=stored, logs=len(files), bytes=sum(v["bytes"] for v in files.values()), by_src=by_src, by_map=by_map),
               coverage_vs_navtrain_yaml=cov, leakage=leak,
               qc_rank0_target_points_inside_label=dict(n=int(tf.size), mean=float(tf.mean()) if tf.size else None,
                                                        share_ge_0p9=float((tf >= 0.9).mean()) if tf.size else None,
                                                        share_eq_0=float((tf == 0).mean()) if tf.size else None),
               store_sha256=hashlib.sha256("\n".join(f"{k}:{v['sha256']}" for k, v in sorted(files.items())).encode()).hexdigest(),
               files=files, runs=[{k: r.get(k) for k in ("tag", "started", "finished", "seconds", "totals", "origin_xcheck", "workers", "aborted", "inputs")}
                                  for r in runs])
    (out / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


def cmd_finalize(a) -> int:
    man = finalize(a.out)
    print(json.dumps(man["counts"]), f"store_sha256={man['store_sha256']}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = lambda p: (p.add_argument("--maps-root", default=os.environ.get("NUPLAN_MAPS_ROOT", "D:/Projects/TanitAD/data/nuplan-maps/nuplan-maps-v1.0")),
                        p.add_argument("--backend", default="nuplan", help="'nuplan' or 'module:factory' (tests)"))
    p = sub.add_parser("logmap")
    p.add_argument("--bank", nargs="+", required=True)
    p.add_argument("--db-root", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--yaml", default=None, help="navtrain.yaml: also map DB-held logs that have no bank row")
    p.add_argument("--k", type=int, default=6, help="ego centres sampled per log for the inference")
    common(p)
    p.set_defaults(fn=cmd_logmap)
    p = sub.add_parser("build")
    p.add_argument("--bank", nargs="+", required=True)
    p.add_argument("--db-root", default=None)
    p.add_argument("--yaml", default=None, help="navtrain.yaml: the token universe (also enables DB-derived extras)")
    p.add_argument("--navtest", default=None, help="navtest json (token_to_log_name): those tokens/logs are NEVER written")
    p.add_argument("--logmap", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--shard", default=None, help="i/n: this process builds the logs with sorted-index %% n == i")
    p.add_argument("--limit-logs", type=int, default=0)
    p.add_argument("--min-free-gb", type=float, default=6.0, help="refuse a multi-worker start below this free physical memory")
    p.add_argument("--abort-free-gb", type=float, default=2.5, help="stop (resumable) if free physical memory falls below this")
    common(p)
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("finalize")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_finalize)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
