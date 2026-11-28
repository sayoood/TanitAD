"""v9 label release reader — per-frame tactical actions + goals with constraints, per-frame nav, route checkpoint.

Release: ``v9_labels_<split>.npz`` + ``.manifest.json`` built by ``stack/scripts/build_v9_labels.py`` (WP-A, SPEC
``FlyWheels/TanitAD_DataFlyWheel/incoming/2026-10-04-v9-labels/SPEC.md``). The tensor contract the trainer and the
model receive is ``INTEGRATION.md`` in the same package.

⛔⛔ NO MUTABLE MODULE STATE. refcv7 trained ``LANE_CHANGE_L`` as a negative on 100 % of tactical windows because
``v7_labels._MEASURED_GEOMETRY_TOKENS`` was module state refilled by the eval-label load before the DataLoader
workers started (D1 F2). Here every function is pure over an immutable :class:`V9Release` passed in explicitly;
the module holds only constants (tuples). A train release and an eval release are two objects that cannot see each
other. Pinned by ``stack/tests/test_v9_labels.py``.

Key: ``(sid, k)`` — ``sid = stable_episode_id(clip_id)`` (the trainer's ``ep.episode_id``), ``k`` = the RAW row =
provider row + ``n_stack - 1``; a dataset window with start index ``t`` has NOW at ``k = t + window - 1 + n_stack - 1``
(= ``t + 9`` for window 8, n_stack 3). ``now_s = grid_start_s + k * dt_s`` is the trainer's own clock
(``refc_v3_train.py::_now_s``) and is stored per row, so a lookup by time is checked, never guessed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

SCHEMA = "tanitad.v9_labels/1"
IGNORE = -100
LAT9 = ("LANE_KEEP", "TURN_L", "TURN_R", "LANE_CHANGE_L", "LANE_CHANGE_R")
LON9 = ("HOLD", "CREEP", "STOP", "FOLLOW", "DECELERATE", "ACCELERATE", "KEEP")
LAT7 = ("LANE_KEEP", "LANE_CHANGE_L", "LANE_CHANGE_R", "ABORT_LC", "NUDGE_L", "NUDGE_R", "TURN_L", "TURN_R")
LON7 = ("FOLLOW", "CRUISE", "YIELD_MERGE", "BRAKE_TO", "CREEP", "HOLD", "ADAPT_SPEED_FOR_CURVE", "ACCELERATE")
GOAL22 = ("FOLLOW_LANE", "TURN_L", "TURN_R", "YIELD_FOR_TURN_L", "YIELD_FOR_TURN_R", "YIELD", "STOP_POINT",
          "SPEED_BAND", "CORRIDOR_OFFSET", "EVADE_IN_CORRIDOR", "OVERTAKE_VEHICLE", "MERGE", "GAP_TARGET",
          "REACT_ON_ONCOMING", "TAKE_EXIT_L", "TAKE_EXIT_R", "TRAFFIC_LIGHT_REACT", "TRAFFIC_LIGHT_REACT_RED",
          "TRAFFIC_LIGHT_REACT_YELLOW", "TRAFFIC_LIGHT_REACT_GREEN", "LANE_CHANGE_L", "LANE_CHANGE_R")
RC_VARIANTS = ("A30", "A50", "A80", "B")
#: the INPUT fields (SPEC §8). Everything else in the release is a TARGET or a diagnostic.
NAV_INPUT_FIELDS = ("nav_token", "nav_side_next", "nav_d_next_m", "nav_d_end_m", "nav_dyaw_next_deg",
                    "nav_args_valid", "nav_lookahead_m")
#: ⛔ never an input (they encode the ego's FUTURE speed — SPEC §5.2): reading them through window_inputs refuses
NAV_FORBIDDEN_AS_INPUT = ("nav_t_next_s", "nav_token_ttime")


class V9LabelError(RuntimeError):
    """A refused lookup or load (wrong release, missing key, clock mismatch)."""


@dataclass(frozen=True)
class V9Release:
    """An immutable loaded release. Construct only through :func:`load_v9_release`."""
    path: str
    md5: str
    split: str
    manifest_json: str                       # the manifest as JSON text (parse with :meth:`manifest`)
    rows: MappingProxyType                   # field -> read-only np.ndarray [N_rows]
    clips: MappingProxyType                  # sid -> (row0, n_rows, k0)

    def manifest(self) -> dict:
        return json.loads(self.manifest_json)

    @property
    def n_rows(self) -> int:
        return int(len(self.rows["k"]))

    def __reduce__(self):                     # pickle (DataLoader workers): rebuild read-only, same content
        return (_rebuild, (self.path, self.md5, self.split, self.manifest_json, dict(self.rows), dict(self.clips)))


def _ro(a):
    a = np.array(a, copy=True)
    a.setflags(write=False)
    return a


def _rebuild(path, md5, split, manifest_json, rows, clips):
    return V9Release(path, md5, split, manifest_json, MappingProxyType({k: _ro(v) for k, v in rows.items()}),
                     MappingProxyType(dict(clips)))


def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_v9_release(npz_path: str, manifest_path: str | None = None, *, expect_md5: str | None = None) -> V9Release:
    """Load a release. ``expect_md5`` (recommended in a launch): refuse a file whose bytes differ."""
    md5 = _md5(npz_path)
    if expect_md5 is not None and md5 != expect_md5:
        raise V9LabelError(f"[v9] ⛔ {npz_path}: md5 {md5} != expected {expect_md5}")
    if manifest_path is None:
        manifest_path = npz_path[:-4] + ".manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        man = json.load(f)
    if man.get("schema") != SCHEMA:
        raise V9LabelError(f"[v9] ⛔ manifest schema {man.get('schema')!r} != {SCHEMA!r}")
    if man.get("npz_md5") not in (None, md5):
        raise V9LabelError(f"[v9] ⛔ manifest names npz md5 {man.get('npz_md5')} but the file is {md5}")
    with np.load(npz_path, allow_pickle=False) as z:
        rows = {k[5:]: _ro(z[k]) for k in z.files if k.startswith("row__")}
        cl = {k[6:]: z[k] for k in z.files if k.startswith("clip__")}
    clips = {int(s): (int(r0), int(n), int(k0)) for s, r0, n, k0 in zip(cl["sid"], cl["row0"], cl["n_rows"], cl["k0"])}
    return V9Release(npz_path, md5, str(man.get("split")), json.dumps(man), MappingProxyType(rows),
                     MappingProxyType(clips))


# ------------------------------------------------------------------------------------------------- lookups
def row_index(rel: V9Release, sid: int, k: int) -> int:
    """Row of frame ``k`` (raw row) of clip ``sid``; refuses an unknown clip or a frame outside it."""
    c = rel.clips.get(int(sid))
    if c is None:
        raise V9LabelError(f"[v9] ⛔ clip sid {sid} is not in the {rel.split} release")
    r0, n, k0 = c
    j = int(k) - k0
    if not 0 <= j < n:
        raise V9LabelError(f"[v9] ⛔ frame k={k} outside clip sid {sid} (k0 {k0}, {n} rows)")
    r = r0 + j
    if int(rel.rows["k"][r]) != int(k):
        raise V9LabelError(f"[v9] ⛔ release row {r} holds k={int(rel.rows['k'][r])}, expected {k}")
    return r


def row_for_window(rel: V9Release, sid: int, t: int, *, window: int = 8, n_stack: int = 3) -> int:
    """Row of a dataset window's NOW (``V3Dataset.index`` entry ``(e_i, t)``): k = t + window - 1 + n_stack - 1."""
    return row_index(rel, sid, int(t) + int(window) - 1 + int(n_stack) - 1)


def row_for_now(rel: V9Release, sid: int, k: int, now_s: float, *, tol_s: float = 1e-6) -> int:
    """As :func:`row_index`, and REFUSE when the trainer's ``now_s`` differs from the release clock by > ``tol_s``."""
    r = row_index(rel, sid, k)
    d = abs(float(rel.rows["now_s"][r]) - float(now_s))
    if d > tol_s:
        raise V9LabelError(f"[v9] ⛔ clock mismatch on sid {sid} k {k}: release {float(rel.rows['now_s'][r]):.6f} "
                           f"vs trainer {float(now_s):.6f} (|d| {d:.2e} s > {tol_s})")
    return r


# ------------------------------------------------------------------------------------------------- decoding
def goal_bits(rel: V9Release, r: int) -> tuple:
    """(y[22], w[22]) float32 over :data:`GOAL22`; w = 0 is IGNORE. SPEED_BAND always carries w = 0 (dropped from the
    BCE; the continuous SPEED goal replaces it)."""
    y, w = int(rel.rows["goal_y"][r]), int(rel.rows["goal_w"][r])
    yy = np.array([(y >> i) & 1 for i in range(len(GOAL22))], np.float32)
    ww = np.array([(w >> i) & 1 for i in range(len(GOAL22))], np.float32)
    return yy, ww


def allowed_mask(bits: int, n: int) -> np.ndarray:
    return np.array([(int(bits) >> i) & 1 for i in range(n)], dtype=bool)


def window_targets(rel: V9Release, r: int, *, lat_variant: str = "a", ids: str = "v7") -> dict:
    """The TARGETS of one row (SPEC §3-4). ``ids='v7'`` maps onto the frozen 8-wide v7 heads (D-WPA-2: WP-B uses
    these); ``ids='v9'`` returns the native v9 classes. A class of -100 with a non-empty ``*_allowed`` mask is a
    PARTIAL label: the loss is ``-log sum_{c in allowed} p_c``."""
    if lat_variant not in ("a", "b"):
        raise ValueError("lat_variant must be 'a' (junction, default) or 'b' (heading)")
    R = rel.rows
    if ids == "v7":
        lat, lat_al, n_lat = int(R[f"lat_v7id_{lat_variant}"][r]), int(R[f"lat_allowed_v7_{lat_variant}"][r]), len(LAT7)
        lon, lon_al, n_lon = int(R["lon_v7id"][r]), int(R["lon_allowed_v7"][r]), len(LON7)
    elif ids == "v9":
        lat, lat_al, n_lat = int(R[f"lat_cls_{lat_variant}"][r]), int(R[f"lat_allowed_{lat_variant}"][r]), len(LAT9)
        lon, lon_al, n_lon = int(R["lon_cls"][r]), int(R["lon_allowed"][r]), len(LON9)
    else:
        raise ValueError("ids must be 'v7' or 'v9'")
    gy, gw = goal_bits(rel, r)

    def f(name):
        return float(R[name][r]) if name in R else float("nan")
    return {"lat": lat, "lat_allowed": allowed_mask(lat_al, n_lat), "lon": lon, "lon_allowed": allowed_mask(lon_al, n_lon),
            "goal_y": gy, "goal_w": gw,
            "lat_constraints": np.array([f(n) for n in LAT_CONSTRAINTS], np.float32),
            "lon_constraints": np.array([f(n) for n in LON_CONSTRAINTS], np.float32),
            "speed_goal": np.array([f(n) for n in SPEED_GOAL], np.float32)}


#: constraint vectors (units in INTEGRATION.md); NaN = undefined for this row (mask it in the loss)
LAT_CONSTRAINTS = ("lat_theta_deg", "turn_t_start_s", "turn_t_end_s", "turn_d_start_m", "turn_len_m", "turn_dyaw_deg",
                   "turn_r_arc_m", "turn_exit_x_m", "turn_exit_y_m", "turn_exit_psi_deg", "lc_t_cross_s", "lc_d_cross_m")
LON_CONSTRAINTS = ("lon_v_target_ms", "lon_t_reach_s", "lon_d_reach_m", "lon_t_start_s", "stop_x_m", "stop_y_m",
                   "lead_gap_m", "lead_tg_s", "lead_gap_min_m", "lead_tg_min_s")
SPEED_GOAL = ("v_a_ms", "v_end_ms", "v_lo_ms", "v_hi_ms")


def window_inputs(rel: V9Release, r: int, *, rc_variant: str) -> dict:
    """The INPUTS of one row (SPEC §8): the nav token + its spatial args and the chosen route checkpoint.
    ⛔ No time-to-turn and no time-rule token (they encode the ego's future speed). The caller applies the
    training dropouts (INTEGRATION.md): RC-absent rows and the nav-args-unknown flag."""
    if rc_variant not in RC_VARIANTS:
        raise ValueError(f"rc_variant must be one of {RC_VARIANTS}")
    R = rel.rows
    tok = int(R["nav_token"][r])
    valid = int(R["nav_args_valid"][r])

    def fz(name):                       # args are encoded only when valid; never 0 for "unknown" (D4)
        v = float(R[name][r])
        return v if (valid and np.isfinite(v)) else 0.0
    return {"nav_token": tok,
            "nav_args": np.array([fz("nav_d_next_m"), fz("nav_d_end_m"), fz("nav_dyaw_next_deg"),
                                  float(R["nav_side_next"][r]) if valid else 0.0,
                                  float(np.clip(R["nav_lookahead_m"][r], 0, 1000.0))], np.float32),
            "nav_args_valid": valid,
            **_rc(R, r, rc_variant)}


def _rc(R, r, v):
    ok = bool(np.isfinite(R[f"rc_{v}_valid"][r]) and int(R[f"rc_{v}_valid"][r]) == 1)
    xyz = np.array([float(R[f"rc_{v}_x"][r]), float(R[f"rc_{v}_y"][r]), float(R[f"rc_{v}_psi"][r])], np.float32)
    if not ok or not np.all(np.isfinite(xyz)):
        return {"rc": np.zeros(3, np.float32), "rc_valid": 0}
    return {"rc": xyz, "rc_valid": 1}
