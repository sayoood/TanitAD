"""B1 EVAL agent join -> **3-D**: the two columns the 2-D join dropped.

``SPEC_REFCV6_V2.md`` §6 asks the BOX head for ``(x, y, z, l, w, h, yaw)``.
MEASURED 2026-09-16, the banked eval join
(``TanitAD Research Lab/Benchmarks & Evals/Research/2026-09-06-b1-agent-join/``)
emits per agent exactly ``{cx, cy, yaw, l, w, occ, track_id, cls}`` -- **no z,
no h** -- so ``box3d_head`` masks both terms (``zh_mask`` all-False, ``n_z``
0). The cuboids themselves carry both: ``obstacle.offline`` has ``center_z``
and ``size_z`` on every row. This builder recovers them.

⛔⛔ **THIS IS NOT A REBUILD OF THE JOIN. IT IS AN ANNOTATION OF IT.**
The 2-D join is read line by line and each line is re-emitted with ``cz`` and
``h`` APPENDED to each agent dict. The key, the frame index space, the row
order, the agent order, every existing field and every existing byte are
therefore the banked artifact's own -- not reproduced, *carried*. A consumer
joining on ``(clip_id, frame)`` keeps working, and so does one that ignores
keys it does not know.

  ⭐ That is also why the correspondence control is an IDENTITY, not a
  similarity: :func:`strip_zh` of every emitted line must be **byte-identical**
  to the 2-D line it came from, on ALL lines (``C2`` below). A sampled check
  would have been the weaker instrument for a question that admits a proof.

## Why z survives the join unchanged (and is therefore recoverable exactly)

The join's geometry is ``rig@sample -> world -> ego@frame``: ``rig_to_world``
composes with the egomotion pose and ``bev_raster.ego_frame_agents`` applies a
**planar SE(2)** rotation+translation -- it writes columns 0 (x), 1 (y) and 2
(yaw) and "sizes and the occ column pass through" (``bev_raster.py:202-224``).
``center_z`` and ``size_z`` are invariant under every step between the parquet
and the join line. Recovering them re-derives NO geometry.

## WHICH SAMPLE -- the part that must be proven, not assumed

``world_agents_at`` picks, per track, the single sample nearest the frame time
within ``tol_s``. This builder picks with the join's OWN track structure
(``build_obstacle_join.clip_tracks``, imported, never re-implemented) and the
same rule, then reads z/h from the parquet row keyed on
``(track_id, timestamp_us)`` of that chosen sample. Three controls make the
choice checkable rather than asserted:

* ``C2`` the strip-identity above;
* ``C5`` per line, the picked ``track_id`` SEQUENCE must equal the line's own,
  element for element -- membership AND order, on all lines;
* ``C6`` the picked row's ``size_x``/``size_y``, rounded as the join rounds
  them, must equal the line's ``l``/``w``.

⚠️ The line's ``t_s`` is ROUNDED to 4 decimals by the 2-D builder, so a pick
could flip near a tie. It is not waved away, and it is not forbidden either:
MEASURED, some tracks carry two label samples 27-83 us apart, so ties EXIST and
a "no ties" gate would refuse the corpus for a reason that is not a defect.
``C7`` therefore measures the CONSEQUENCE -- for every pick inside the
rounding's own 1e-4 s reach it reads BOTH candidates' z/h and requires the
difference to be below the quantum the artifact is rounded to.

## THE FRAME CONVENTION -- the measured trap this builder exists to not repeat

``RETRACTION_LOG.md:15278`` (CLASS H sibling): *"PhysicalAI's tracked boxes
carry z about 1 m under the LiDAR ground; a display mask built from z - h/2
blanked the road in front of every vehicle."* So the convention is MEASURED
here against an independent geometric reference and the verdict is recorded:

    ``center_z`` is the cuboid **CENTRE**; the base is ``center_z - size_z/2``.

MEASURED over 945,091 ``obstacle_offline_b1eval`` cuboids: the base median is
-0.008 m (automobile), -0.077 m (bus), -0.049 m (heavy_truck), +0.010 m
(person) -- the road plane. Under the rival reading (``center_z`` already the
base) every car would float ``h/2`` = +0.75 m above the road. The retraction's
"about 1 m under" is exactly what the rival reading produces when a consumer
then subtracts ``h/2`` again: -0.008 - 1.512/2 = **-0.764 m**.

``C3`` (the road-plane control) and ``C4`` (the LiDAR cross-check) below are
that verdict as gates, and ``--mutate emit-base-as-centre`` is the mutation
that must turn them RED.

Usage (dev box, CPU/IO only)::

  set PYTHONPATH=<repo>/stack
  python stack/scripts/build_b1_agent_join_3d.py ^
      --join-2d      <repo>/.../2026-09-06-b1-agent-join/raw/b1eval_agents.jsonl.xz ^
      --obstacle-dir <data>/physicalai/labels/obstacle_offline_b1eval ^
      --bevgt-dir    <artifacts>/bev-lidar-gt-b1eval-20260913 ^
      --out          <artifacts>/b1-agent-join-3d-20260917/b1eval_agents_3d.jsonl.xz

Writes ``<out>.meta.json`` (provenance, the seven controls, the census) and
ends with one ``B1_JOIN3D_DONE {...}`` line. Every print is ASCII: this box is
cp1252 and a non-ASCII print has already truncated a banked artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import sys
import time
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_SCRIPTS), str(_SCRIPTS.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: the two keys appended to each agent dict. APPENDED, never interleaved --
#: that is what makes :func:`strip_zh` a byte-exact inverse.
CZ_KEY = "cz"
H_KEY = "h"

#: the join's own rounding, so a 3-D line is written in the 2-D line's units:
#: a metre COORDINATE like cx/cy -> 4 decimals; a metre SIZE like l/w -> 3.
CZ_ROUND = 4
H_ROUND = 3

#: ⛔ CLASSES THAT STAND ON THE ROAD. ``protruding_object`` is airborne BY
#: DEFINITION (MEASURED base median +0.861 m over 3,611 rows) and pooling it
#: with vehicles would move the control's own statistic -- the
#: ``ground_bottom_stats`` docstring says so, and this is that warning as code.
GROUND_STANDING_CLASSES: tuple[str, ...] = (
    "automobile", "bus", "heavy_truck", "other_vehicle", "trailer",
    "train_or_tram_car",
)

#: ⛔ THE TOLERANCE IS THE LIDAR ARTIFACT'S OWN, NOT ONE INVENTED HERE.
#: ``bev-lidar-gt-b1eval-20260913`` gates its own per-clip ground estimate with
#: ``content_checks.bands.ground_peak_z_m = [-0.2, +0.2]`` -- that band IS the
#: programme's standing statement of "this is the road plane" on this corpus,
#: measured from LiDAR returns. Reused verbatim (the ``LEAD_SPEED_TOL_MPS``
#: precedent in the 2-D builder), so the control cannot be tuned to pass.
ROAD_PLANE_TOL_M: float = 0.20

#: the mutation must beat the tolerance by at least this much, or the control
#: is too blunt to be worth running. h/2 for a car is 0.756 m = 3.8x.
MIN_MUTATION_SEPARATION: float = 2.0

#: the radius inside which a label base and a LiDAR ground estimate are
#: measuring the SAME piece of road. Half the P8 grid's 60 m reach.
NEAR_FIELD_M: float = 30.0

#: the rounding reach of the 2-D join's ``t_s`` (round(t, 4)): a pick can only
#: flip if two samples are equidistant to within 2 x 0.5e-4 s.
T_S_ROUNDING_REACH_S: float = 1.0e-4

#: ⛔ THE MUTATIONS. A control is not proven by reading it; it is proven by
#: reintroducing the defect it claims to catch and watching it go RED
#: (project memory: "Guards need mutation, not inspection"). Each entry names
#: the control it must turn red -- a mutation that turns NOTHING red is a
#: control-shaped hole, and the test asserts this mapping.
MUTATIONS: dict = {
    "none": None,
    #: flip centre-vs-base: emit cz - h/2 as if the parquet z were a base.
    "emit-base-as-centre": ("C3_road_plane", "C4_lidar_cross_check"),
    #: read the labels one 10 Hz frame late: a different agent SET.
    "shift-frame": ("C5_track_sequence_equals_the_line",),
    #: right agent set, WRONG pairing -- rotate the picks inside the attach
    #: loop only, AFTER C5 has compared the sequences. This is the failure C5
    #: structurally cannot see, and the reason C6 exists.
    "misattach": ("C6_picked_row_sizes_equal_the_line",),
    #: touch a field the 3-D join must only carry, never author.
    "perturb-cx": ("C2_strip_zh_is_byte_identical_to_the_2d_line",),
}
_MUTATIONS = tuple(MUTATIONS)


def sha12(clip_id: str) -> str:
    """The repo-safe spelling of a clip id (``semantic_map_gt.sha12``)."""
    return hashlib.sha256(str(clip_id).encode("utf-8")).hexdigest()[:12]


# ===========================================================================
# reading
# ===========================================================================
def open_lines(path):
    p = str(path)
    if p.endswith(".xz"):
        return lzma.open(p, "rt", encoding="utf-8")
    return open(p, "r", encoding="utf-8")


def open_out(path, compressed: bool):
    """``compressed`` is EXPLICIT, never sniffed -- ``<out>.part`` sniffs False
    while the reader sniffs the FINAL name and yields True (the 2-D builder's
    ``open_out`` docstring records that exact failure)."""
    p = str(path)
    if compressed:
        return lzma.open(p, "wt", encoding="utf-8", preset=6)
    return open(p, "w", encoding="utf-8")


def md5_of(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dumps_line(rec) -> str:
    """The 2-D builder's own serializer (``build_b1_agent_join.py:795``)."""
    return json.dumps(rec, separators=(",", ":"))


def strip_zh(rec) -> dict:
    """The inverse of the annotation: drop ``cz``/``h`` from every agent.

    Returns a NEW record; the input is untouched. :func:`dumps_line` of this
    must reproduce the 2-D line byte for byte -- control ``C2``.
    """
    out = dict(rec)
    out["agents"] = [{k: v for k, v in a.items() if k not in (CZ_KEY, H_KEY)}
                     for a in rec["agents"]]
    return out


# ===========================================================================
# the per-clip cuboid side
# ===========================================================================
def exact_t_s(clip_id: str, eps_dir: Path, ego_dir: Path, block=None):
    """⭐ THE 2-D BUILDER'S OWN ``t_s``, UNROUNDED -- the exact time it picked at.

    ⛔ WHY THIS EXISTS, MEASURED. The banked line carries ``t_s`` rounded to 4
    decimals. On the first smoke run that rounding was not harmless: 19 picks in
    400 lines sat within 1e-4 s of a tie, and in the worst of them the two
    candidate samples were **101 ms apart with 0.714 m between their z**. A
    rounding that can move a height by 0.7 m is not a detail, and no control
    built on the rounded time can resolve it -- so the time is recovered exactly
    instead, and the ambiguity stops existing.

    These are ``join_clip_raw``'s own two lines (``build_b1_agent_join.py``:
    ``ls.register_poses_to_time(poses[:, :2], ego.t, ego.x, ego.y)`` then
    ``reg["t_s"]``), and the duplication is admissible ONLY because ``C9`` below
    proves the result: ``round(t_exact[frame], 4)`` must equal the banked line's
    ``t_s`` on EVERY line. A drift between the two spellings turns that control
    red on real data.

    Returns ``None`` when the clip's pose record is missing (the caller then
    falls back to the rounded ``t_s`` and says so in the meta).
    """
    from build_b1_agent_join import read_raw_poses, _bootstrap_taniteval
    from build_obstacle_join import EgoTrack

    ep = Path(eps_dir) / (clip_id + ".v2ep.pt")
    gp = Path(ego_dir) / (clip_id + ".parquet")
    if not ep.is_file() or not gp.is_file():
        return None, "missing_pose_or_egomotion"
    _bootstrap_taniteval()
    from taniteval import lead_source as ls

    _cid, poses, _n = read_raw_poses(ep)
    ego = EgoTrack(load_parquet(gp))
    try:
        reg = ls.register_poses_to_time(poses[:, :2], ego.t, ego.x, ego.y)
        return np.asarray(reg["t_s"], dtype=np.float64), "register_poses_to_time"
    except Exception as e:                                          # noqa: BLE001
        # a clip that does not MOVE carries no positional signal -- the 2-D
        # builder's own recoverable case, recovered the same way.
        if "egistration" not in repr(e) or block is None:
            raise
        fb = block.get(clip_id)
        if fb is None or not all(i in fb for i in range(poses.shape[0])):
            return None, "registration_impossible_and_no_block_cover"
        return (np.array([fb[i][0] for i in range(poses.shape[0])],
                         dtype=np.float64), "lead_block_t0_s")


def load_parquet(path: Path):
    import pandas as pd
    return pd.read_parquet(path)


def load_lead_block(path):
    """``clip_id -> {frame: (t0_s, speed)}`` (the 2-D builder's own reader)."""
    if path is None or not Path(path).is_file():
        return None
    blk = np.load(str(path), allow_pickle=False)
    if not {"clip_id", "frame", "t0_s", "speeds"} <= set(blk.files):
        return None
    out: dict = {}
    for c, f, t, s in zip(blk["clip_id"], blk["frame"], blk["t0_s"], blk["speeds"]):
        out.setdefault(str(c), {})[int(f)] = (float(t), float(s))
    return out


def load_clip(obs_path: Path):
    """``(tracks, zh_lookup, n_rows)`` for one clip.

    ``tracks``: ``build_obstacle_join.clip_tracks`` VERBATIM -- the same object
    ``world_agents_at`` picked from when the 2-D join was written, including
    its ``reference_frame == 'rig'`` refusal.
    ``zh_lookup``: ``(track_id, timestamp_us) -> (center_z, size_z)``. Keying on
    the exact sample rather than on a re-derived index is what lets the picked
    row be checked against the banked line (``C6``) instead of trusted.
    """
    from build_obstacle_join import clip_tracks
    from tanitad.data.agent_cuboid_gt import check_cuboid_schema, read_clip_cuboids

    obs = read_clip_cuboids(obs_path)
    check_cuboid_schema(obs)                       # refuses a 2-D / non-rig table
    tracks = clip_tracks(obs)
    tid = np.asarray(obs["track_id"]).astype(str)
    tus = np.asarray(obs["timestamp_us"], dtype=np.int64)
    cz = np.asarray(obs["center_z"], dtype=np.float64)
    sz = np.asarray(obs["size_z"], dtype=np.float64)
    cls = np.asarray(obs["label_class"]).astype(str)
    lookup: dict = {}
    dups = 0
    for k in range(tid.size):
        key = (str(tid[k]), int(tus[k]))
        if key in lookup:
            dups += 1
            continue
        lookup[key] = (float(cz[k]), float(sz[k]), str(cls[k]))
    if dups:
        raise SystemExit(
            "[b1join3d] REFUSING %s: %d duplicate (track_id, timestamp_us) rows "
            "-- the z/h lookup would be ambiguous" % (sha12(obs_path.stem), dups))
    return tracks, lookup, int(tid.size)


def pick_for_frame(tracks, t_s: float, tol_s: float):
    """The join's own per-track nearest-sample rule, with the sample IDENTIFIED.

    Returns a list of
    ``(track_id, timestamp_us, l, w, tie_margin_s, dt_s, runner_up_timestamp_us)``
    in ``tracks`` order -- i.e. in the order ``world_agents_at`` emitted its
    rows, which is the order the banked line's ``agents`` are in.

    ``tie_margin_s`` is ``|dt_2nd| - |dt_1st|`` (``inf`` for a single-sample
    track) and ``runner_up_timestamp_us`` names the sample the pick would flip
    to. ⭐ Together they are what turns ``C7`` from an assumption ("the rounded
    ``t_s`` surely does not flip a pick") into a MEASUREMENT: when the margin is
    inside the rounding's reach, the caller reads BOTH candidates' z/h and
    bounds what a flip could do.
    """
    out = []
    for tr in tracks:
        d = np.abs(tr["t"] - float(t_s))
        j = int(np.argmin(d))
        if float(d[j]) > float(tol_s):
            continue
        margin, runner = float("inf"), None
        if d.size >= 2:
            order = np.argsort(d, kind="stable")
            margin = float(d[order[1]] - d[order[0]])
            runner = int(round(float(tr["t"][int(order[1])]) * 1e6))
        out.append((tr["tid"], int(round(float(tr["t"][j]) * 1e6)),
                    float(tr["l"][j]), float(tr["w"][j]), margin, float(d[j]),
                    runner))
    return out


# ===========================================================================
# the controls
# ===========================================================================
class ControlFailure(SystemExit):
    """A control that must read a known value did not. The build writes nothing."""


class BaseStats:
    """Running per-class base-face (``cz - h/2``) and height accumulator."""

    #: ⚠️ the rig z=0 road plane is FLAT and a road is not. Reporting the base
    #: only as one median would let a reader conclude "base == 0 everywhere",
    #: which is true near the ego and drifts with grade at range -- the
    #: true-but-wrong-for-the-reader family. So it is reported BY RANGE.
    RANGE_EDGES_M: tuple[float, ...] = (0.0, 20.0, 40.0, 60.0, float("inf"))

    def __init__(self):
        self.base: dict[str, list] = {}
        self.h: dict[str, list] = {}
        self.cz: dict[str, list] = {}
        self.per_clip_veh_base: dict[str, list] = {}
        #: ⛔ THE LIKE-FOR-LIKE HALF OF C4. The LiDAR ground estimate is a
        #: NEAR-FIELD statistic (the ground band is dense around the ego);
        #: comparing it with labels 60 m away measures the ROAD'S SHAPE, not
        #: the label frame. MEASURED: the worst clip of the corpus has 0
        #: vehicle boxes inside 20 m and 20 of them at 20-60 m on a descent,
        #: base -2.90 m -- a 2.9 m "disagreement" between two different places.
        self.per_clip_veh_base_near: dict[str, list] = {}
        self.by_range: dict[str, list] = {}

    @classmethod
    def range_bin(cls, r: float) -> str:
        e = cls.RANGE_EDGES_M
        for i in range(len(e) - 1):
            if e[i] <= r < e[i + 1]:
                hi = "inf" if e[i + 1] == float("inf") else "%g" % e[i + 1]
                return "%g-%s m" % (e[i], hi)
        return "out"

    def add(self, cls: str, cz: float, h: float, clip_id: str,
            rng_m: float | None = None):
        b = cz - h / 2.0
        self.base.setdefault(cls, []).append(b)
        self.h.setdefault(cls, []).append(h)
        self.cz.setdefault(cls, []).append(cz)
        if cls in GROUND_STANDING_CLASSES:
            self.per_clip_veh_base.setdefault(clip_id, []).append(b)
            if rng_m is not None:
                self.by_range.setdefault(self.range_bin(float(rng_m)), []).append(b)
                if float(rng_m) <= NEAR_FIELD_M:
                    self.per_clip_veh_base_near.setdefault(clip_id, []).append(b)

    def per_range(self) -> dict:
        out = {}
        for k in sorted(self.by_range, key=lambda s: float(s.split("-")[0])):
            b = np.asarray(self.by_range[k])
            out[k] = {"n": int(b.size),
                      "base_median_m": round(float(np.median(b)), 4),
                      "base_p10_m": round(float(np.percentile(b, 10)), 4),
                      "base_p90_m": round(float(np.percentile(b, 90)), 4),
                      "frac_within_0.2m": round(float((np.abs(b) <= 0.2).mean()), 4)}
        return out

    def per_class(self) -> dict:
        out = {}
        for c in sorted(self.base):
            b = np.asarray(self.base[c]); h = np.asarray(self.h[c])
            z = np.asarray(self.cz[c])
            out[c] = {
                "n": int(b.size),
                "cz_median_m": round(float(np.median(z)), 4),
                "h_median_m": round(float(np.median(h)), 4),
                "h_p10_m": round(float(np.percentile(h, 10)), 4),
                "h_p90_m": round(float(np.percentile(h, 90)), 4),
                "base_median_m": round(float(np.median(b)), 4),
                "base_p10_m": round(float(np.percentile(b, 10)), 4),
                "base_p90_m": round(float(np.percentile(b, 90)), 4),
                "ground_standing_class": c in GROUND_STANDING_CLASSES,
            }
        return out

    def ground_base(self) -> np.ndarray:
        got = [np.asarray(self.base[c]) for c in GROUND_STANDING_CLASSES
               if c in self.base]
        return np.concatenate(got) if got else np.zeros((0,))


def control_road_plane(stats: BaseStats) -> dict:
    """⛔ C3 -- A CONTROL THAT READS A KNOWN VALUE.

    A ground-standing vehicle's BASE must sit on the road plane. The road plane
    is rig ``z = 0`` (the LiDAR BEV GT's ``ground_plane``, reproduced from LiDAR
    returns at -0.035 m) and the tolerance is that artifact's own accepted band
    :data:`ROAD_PLANE_TOL_M`. Flipping centre-vs-base moves the statistic by
    ``h/2`` ~ 0.76 m for a car; that is the mutation this control must fail on.
    """
    b = stats.ground_base()
    med = float(np.median(b)) if b.size else float("nan")
    passed = bool(b.size) and abs(med) <= ROAD_PLANE_TOL_M
    return {
        "control": "C3 ground-standing base on the road plane",
        "statistic": "median(cz - h/2) over %s" % (GROUND_STANDING_CLASSES,),
        "reference": "rig z = 0 IS the road plane (bev-lidar-gt-b1eval "
                     "meta.ground_plane, reproduced from LiDAR returns at "
                     "-0.035 m)",
        "tol_m": ROAD_PLANE_TOL_M,
        "tol_source": "bev-lidar-gt-b1eval meta.content_checks.bands."
                      "ground_peak_z_m = [-0.2, +0.2] (adopted verbatim)",
        "n": int(b.size),
        "median_m": round(med, 5),
        "p10_m": round(float(np.percentile(b, 10)), 5) if b.size else None,
        "p90_m": round(float(np.percentile(b, 90)), 5) if b.size else None,
        "passed": passed,
    }


def control_lidar(stats: BaseStats, bevgt_dir: Path | None) -> dict:
    """⛔ C4 -- THE INDEPENDENT GEOMETRIC REFERENCE.

    The label side says "vehicle bases are at rig z ~ 0". A DIFFERENT sensor,
    processed by a different pipeline, says where the road is: the banked LiDAR
    BEV GT carries a per-clip ``stats.ground_peak_z_m`` measured from LiDAR
    returns in the same rig frame. Pairing them per clip is the frame check the
    retraction asks for -- "measure a label's frame against the sensor it will
    be drawn over before building on it".

    ⛔ VERIFICATION ONLY. The PI has ruled LiDAR is not a training target; no
    number computed here reaches the artifact's agents.
    """
    out = {"control": "C4 label base vs LiDAR-measured road plane (per clip)",
           "binding": "VERIFICATION ONLY -- LiDAR is not a training target (PI)",
           "reference": "bev-lidar-gt-b1eval-20260913 stats.ground_peak_z_m",
           "tol_m": ROAD_PLANE_TOL_M}
    if bevgt_dir is None or not Path(bevgt_dir).is_dir():
        out.update({"available": False, "passed": None,
                    "note": "no --bevgt-dir: the cross-check was NOT run"})
        return out
    ground = {}
    for p in sorted(Path(bevgt_dir).glob("*.bevgt.npz")):
        z = np.load(str(p), allow_pickle=False)
        m = json.loads(str(z["meta_json"]))
        ground[m["clip_sha12"]] = float(m["stats"]["ground_peak_z_m"])
    rows, unpaired = [], 0
    for cid, vals in stats.per_clip_veh_base.items():
        g = ground.get(sha12(cid))
        if g is None:
            unpaired += 1
            continue
        rows.append((sha12(cid), float(np.median(vals)), g,
                     float(np.median(vals)) - g))
    near = []
    for cid, vals in stats.per_clip_veh_base_near.items():
        g = ground.get(sha12(cid))
        if g is not None and len(vals) >= 5:
            near.append((sha12(cid), float(np.median(vals)) - g, len(vals)))
    if not rows:
        out.update({"available": True, "passed": False, "n_clips": 0,
                    "note": "no clip paired with a LiDAR reference"})
        return out
    d = np.abs(np.asarray([r[3] for r in rows]))
    worst = max(rows, key=lambda r: abs(r[3]))
    out.update({
        "available": True,
        "n_bevgt_clips": len(ground),
        "n_clips_paired": len(rows),
        "n_clips_unpaired": unpaired,
        "statistic": "|median(base of ground-standing vehicles) - LiDAR "
                     "ground_peak_z| per clip",
        "median_abs_m": round(float(np.median(d)), 5),
        "p90_abs_m": round(float(np.percentile(d, 90)), 5),
        "max_abs_m": round(float(d.max()), 5),
        "worst_clip_sha12": worst[0],
        "worst_clip_delta_m": round(float(worst[3]), 5),
        "frac_clips_within_tol": round(float((d <= ROAD_PLANE_TOL_M).mean()), 4),
        # ⭐ THE LIKE-FOR-LIKE STATISTIC: vehicles inside NEAR_FIELD_M, where
        # the label and the LiDAR ground estimate describe the same road.
        "near_field": ({
            "radius_m": NEAR_FIELD_M, "min_boxes_per_clip": 5,
            "n_clips": len(near),
            "median_abs_m": round(float(np.median(np.abs([r[1] for r in near]))), 5),
            "p90_abs_m": round(float(np.percentile(np.abs([r[1] for r in near]), 90)), 5),
            "max_abs_m": round(float(np.max(np.abs([r[1] for r in near]))), 5),
            "frac_clips_within_tol": round(float(
                (np.abs([r[1] for r in near]) <= ROAD_PLANE_TOL_M).mean()), 4),
        } if near else {"radius_m": NEAR_FIELD_M, "n_clips": 0}),
        # the GATE is the median: a handful of steep-grade clips must not be
        # able to fail a corpus-level frame question, and must not be able to
        # hide one either -- p90 and max are reported beside it.
        "passed": bool(float(np.median(d)) <= ROAD_PLANE_TOL_M),
    })
    return out


def control_rival_hypothesis(stats: BaseStats) -> dict:
    """⛔ C8 -- THE RIVAL READING, PRICED.

    Stating "z is the centre" is worth nothing unless the alternative was
    computed and is worse. If ``center_z`` were already the base, the same
    vehicles would stand ``median(cz)`` above the road.
    """
    got = [np.asarray(stats.cz[c]) for c in GROUND_STANDING_CLASSES
           if c in stats.cz]
    z = np.concatenate(got) if got else np.zeros((0,))
    hs = [np.asarray(stats.h[c]) for c in GROUND_STANDING_CLASSES
          if c in stats.h]
    h = np.concatenate(hs) if hs else np.zeros((0,))
    med_c = float(np.median(stats.ground_base())) if z.size else float("nan")
    med_b = float(np.median(z)) if z.size else float("nan")
    return {
        "control": "C8 the rival convention, computed",
        "n": int(z.size),
        "if_centre_base_is_cz_minus_h2_median_m": round(med_c, 5),
        "if_base_then_base_is_cz_median_m": round(med_b, 5),
        "h_median_m": round(float(np.median(h)), 5) if h.size else None,
        "verdict": ("CENTRE" if abs(med_c) < abs(med_b) else "BASE"),
        "separation_m": round(abs(med_b) - abs(med_c), 5),
        "retraction_reproduced_m": round(med_c - float(np.median(h)) / 2.0, 5)
        if h.size else None,
        "retraction_note": "RETRACTION_LOG.md:15278 reports boxes 'about 1 m "
                           "under the LiDAR ground'; that is what the BASE "
                           "reading produces when a consumer subtracts h/2 "
                           "from an already-base z",
    }


# ===========================================================================
# the build
# ===========================================================================
def build(join_2d: Path, obstacle_dir: Path, out_path: Path, *,
          tol_s: float, bevgt_dir: Path | None, mutate: str = "none",
          limit_lines: int | None = None, verbose: bool = True,
          eps_dir: Path | None = None, ego_dir: Path | None = None,
          lead_block: Path | None = None) -> dict:
    """Annotate the 2-D join with z/h and run every control. Returns the meta."""
    if mutate not in _MUTATIONS:
        raise SystemExit("[b1join3d] --mutate must be one of %s" % (_MUTATIONS,))
    t0 = time.time()
    block = load_lead_block(lead_block)
    exact_mode = eps_dir is not None and ego_dir is not None
    t_exact = None
    time_source = "line_t_s_rounded"
    c9_fail = 0
    c9_example = None
    c9_max_abs_dt = 0.0
    n_lines_exact = 0
    time_sources: dict = {}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    compressed = str(out_path).endswith(".xz")
    tmp = out_path.with_name(out_path.name + ".part")

    stats = BaseStats()
    cur_clip = None
    tracks = lookup = None
    n_lines = n_agents = n_zh = n_missing = 0
    n_clip_loads = 0
    clips: dict[str, dict] = {}
    c1_fail = c2_fail = c5_fail = c6_fail = 0
    c1_example = c2_example = c5_example = c6_example = None
    tie_min = float("inf")
    n_tie_inside_rounding = 0
    n_single_sample_tracks = 0
    tie_dz_max = tie_dh_max = 0.0
    tie_unresolved = 0
    tie_example = None
    dt_max = 0.0
    no_obstacle: set = set()

    with open_out(tmp, compressed) as fh, open_lines(join_2d) as src:
        for raw in src:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            if limit_lines is not None and n_lines >= limit_lines:
                break
            rec = json.loads(raw)
            cid = rec["clip_id"]

            # ---- C1: the serializer round-trips the BANKED bytes -----------
            # Without this, C2 could "pass" by both sides being wrong the same
            # way. C1 pins the serializer against the artifact as written.
            if dumps_line(rec) != raw:
                c1_fail += 1
                if c1_example is None:
                    c1_example = {"clip_sha12": sha12(cid), "frame": rec["frame"]}

            if cid != cur_clip:
                op = Path(obstacle_dir) / (cid + ".parquet")
                if not op.is_file():
                    no_obstacle.add(sha12(cid))
                    raise SystemExit(
                        "[b1join3d] REFUSING: clip %s is in the 2-D join but has "
                        "no obstacle parquet -- the 3-D join would silently lose "
                        "its lines and stop being row-identical" % sha12(cid))
                tracks, lookup, n_rows = load_clip(op)
                cur_clip = cid
                n_clip_loads += 1
                t_exact, time_source = (None, "line_t_s_rounded")
                if exact_mode:
                    t_exact, time_source = exact_t_s(cid, eps_dir, ego_dir, block)
                    if t_exact is None:
                        time_source = "line_t_s_rounded(" + time_source + ")"
                time_sources[time_source] = time_sources.get(time_source, 0) + 1
                clips.setdefault(sha12(cid), {"n_lines": 0, "n_agents": 0,
                                              "n_zh": 0, "n_rows": n_rows,
                                              "time_source": time_source})
                if verbose and n_clip_loads % 20 == 0:
                    print("[b1join3d] clip %d ... (%s)" % (n_clip_loads, sha12(cid)),
                          flush=True)

            t_s = float(rec["t_s"])
            if t_exact is not None and 0 <= int(rec["frame"]) < t_exact.size:
                te = float(t_exact[int(rec["frame"])])
                # ---- C9: the recovered time IS the banked builder's ---------
                d = abs(round(te, 4) - t_s)
                c9_max_abs_dt = max(c9_max_abs_dt, d)
                if round(te, 4) != t_s:
                    c9_fail += 1
                    if c9_example is None:
                        c9_example = {"clip_sha12": sha12(cid),
                                      "frame": rec["frame"],
                                      "recovered_rounded": round(te, 4),
                                      "line_t_s": t_s}
                t_s = te
                n_lines_exact += 1
            if mutate == "shift-frame":
                t_s = t_s + 0.1                      # one frame of the 10 Hz grid
            picked = pick_for_frame(tracks, t_s, tol_s)
            agents = rec["agents"]

            # ---- C5: membership AND order, element for element -------------
            got = [p[0] for p in picked]
            want = [a["track_id"] for a in agents]
            if got != want:
                c5_fail += 1
                if c5_example is None:
                    c5_example = {"clip_sha12": sha12(cid), "frame": rec["frame"],
                                  "n_picked": len(got), "n_line": len(want)}
                # cannot annotate a line whose agents we cannot identify
                if mutate == "none":
                    raise ControlFailure(
                        "[b1join3d] REFUSING at %s frame %d: the picked track "
                        "sequence (%d) is not the line's (%d) -- the row "
                        "correspondence is broken, and an annotated line would "
                        "attach the WRONG agent's height"
                        % (sha12(cid), rec["frame"], len(got), len(want)))

            if mutate == "misattach" and len(picked) > 1:
                picked = picked[1:] + picked[:1]     # AFTER C5, on purpose
            n_zh_line = 0
            for a, p in zip(agents, picked):
                tid, tus, l_m, w_m, margin, dt, runner = p
                if round(l_m, 3) != a["l"] or round(w_m, 3) != a["w"]:
                    c6_fail += 1
                    if c6_example is None:
                        c6_example = {"clip_sha12": sha12(cid),
                                      "frame": rec["frame"],
                                      "picked_lw": [round(l_m, 3), round(w_m, 3)],
                                      "line_lw": [a["l"], a["w"]]}
                if margin == float("inf"):
                    n_single_sample_tracks += 1
                else:
                    tie_min = min(tie_min, margin)
                    if margin < T_S_ROUNDING_REACH_S:
                        n_tie_inside_rounding += 1
                        # ⭐ price the flip instead of fearing it
                        alt = lookup.get((tid, runner)) if runner is not None else None
                        cur = lookup.get((tid, tus))
                        if alt is not None and cur is not None:
                            dz = abs(alt[0] - cur[0]); dh = abs(alt[1] - cur[1])
                            tie_dz_max = max(tie_dz_max, dz)
                            tie_dh_max = max(tie_dh_max, dh)
                            if tie_example is None:
                                tie_example = {
                                    "clip_sha12": sha12(cid),
                                    "frame": rec["frame"],
                                    "margin_s": round(margin, 9),
                                    "sample_gap_us": abs(runner - tus),
                                    "abs_dcz_m": round(dz, 8),
                                    "abs_dh_m": round(dh, 8)}
                        else:
                            tie_unresolved += 1
                dt_max = max(dt_max, dt)
                hit = lookup.get((tid, tus))
                if hit is None:
                    n_missing += 1
                    continue
                cz, h, cls = hit
                if mutate == "emit-base-as-centre":
                    cz = cz - h / 2.0               # pretend the parquet z is a base
                if mutate == "perturb-cx":
                    a["cx"] = round(float(a["cx"]) + 0.5, CZ_ROUND)
                a[CZ_KEY] = round(float(cz), CZ_ROUND)
                a[H_KEY] = round(float(h), H_ROUND)
                n_zh += 1
                n_zh_line += 1
                stats.add(str(a.get("cls", cls)), float(cz), float(h), cid,
                          rng_m=float(np.hypot(a["cx"], a["cy"])))

            n_agents += len(agents)
            line3d = dumps_line(rec)

            # ---- C2: the strip-identity, on EVERY line ---------------------
            if dumps_line(strip_zh(rec)) != raw:
                c2_fail += 1
                if c2_example is None:
                    c2_example = {"clip_sha12": sha12(cid), "frame": rec["frame"]}

            fh.write(line3d + "\n")
            n_lines += 1
            c = clips[sha12(cid)]
            c["n_lines"] += 1
            c["n_agents"] += len(agents)
            c["n_zh"] += n_zh_line

    # the WRITTEN bytes are re-read: a truncated write and a good one are not
    # distinguishable by exit code (the 2-D builder's assert_content docstring).
    # ⛔ under a mutation the re-read assertion must REPORT, not raise: the
    # whole point of the run is to collect which controls turned red.
    try:
        reread = assert_content(tmp, n_lines, compressed, raw_2d=join_2d,
                                limit_lines=limit_lines,
                                strict=(mutate == "none"))
    except ControlFailure as e:
        if mutate == "none":
            raise
        reread = {"refused": repr(e)[:300]}

    c3 = control_road_plane(stats)
    c4 = control_lidar(stats, bevgt_dir)
    c8 = control_rival_hypothesis(stats)
    controls = {
        "C1_serializer_roundtrips_the_banked_bytes": {
            "n_lines": n_lines, "n_fail": c1_fail, "example": c1_example,
            "passed": c1_fail == 0},
        "C2_strip_zh_is_byte_identical_to_the_2d_line": {
            "n_lines": n_lines, "n_fail": c2_fail, "example": c2_example,
            "scope": "ALL lines, not a sample", "passed": c2_fail == 0},
        "C5_track_sequence_equals_the_line": {
            "n_lines": n_lines, "n_fail": c5_fail, "example": c5_example,
            "scope": "ALL lines, membership AND order", "passed": c5_fail == 0},
        "C6_picked_row_sizes_equal_the_line": {
            "n_agents": n_agents, "n_fail": c6_fail, "example": c6_example,
            "passed": c6_fail == 0},
        # ⛔ C7 measures the CONSEQUENCE of a flip, not its possibility.
        # A first draft gated on "no pick is inside the rounding reach" and
        # REFUSED the corpus: MEASURED, some tracks carry two label samples
        # 27-83 us apart, so ties exist and always will. What matters is that
        # the two candidates carry the SAME height -- which is measurable, and
        # measured here, against the quantum the artifact is rounded to.
        "C7_a_t_s_rounding_flip_cannot_change_zh": {
            "reach_s": T_S_ROUNDING_REACH_S,
            "min_tie_margin_s": None if tie_min == float("inf") else round(tie_min, 9),
            "n_picks_inside_reach": n_tie_inside_rounding,
            "n_picks_inside_reach_unresolved": tie_unresolved,
            "n_single_sample_tracks": n_single_sample_tracks,
            "max_abs_dcz_over_ties_m": round(tie_dz_max, 8),
            "max_abs_dh_over_ties_m": round(tie_dh_max, 8),
            "bound_cz_m": 10.0 ** (-CZ_ROUND),
            "bound_h_m": 10.0 ** (-H_ROUND),
            "example": tie_example,
            "max_dt_s": round(dt_max, 6), "tol_s": tol_s,
            "passed": (tie_unresolved == 0
                       and tie_dz_max <= 10.0 ** (-CZ_ROUND)
                       and tie_dh_max <= 10.0 ** (-H_ROUND))},
        "C3_road_plane": c3,
        "C4_lidar_cross_check": c4,
        "C8_rival_convention": c8,
        "C9_recovered_time_is_the_banked_builders": {
            "exact_mode": exact_mode,
            "n_lines_on_exact_time": n_lines_exact,
            "n_lines": n_lines,
            "time_sources": time_sources,
            "statistic": "round(recovered t_s, 4) == the banked line's t_s",
            "n_fail": c9_fail, "example": c9_example,
            "max_abs_dt_s": round(c9_max_abs_dt, 8),
            "passed": (not exact_mode) or (c9_fail == 0
                                           and n_lines_exact == n_lines)},
    }
    gates = ["C1_serializer_roundtrips_the_banked_bytes",
             "C2_strip_zh_is_byte_identical_to_the_2d_line",
             "C5_track_sequence_equals_the_line",
             "C6_picked_row_sizes_equal_the_line",
             "C3_road_plane"]
    # ⛔ C7 only GATES when the rounded time is actually what the pick used.
    # With the exact time recovered there is no rounding to flip anything, and
    # gating on a quantity that cannot vary would be a control that reports
    # nothing (the C9/C14 family). It is still COMPUTED and reported.
    if not exact_mode:
        gates.append("C7_a_t_s_rounding_flip_cannot_change_zh")
    else:
        gates.append("C9_recovered_time_is_the_banked_builders")
    if c4.get("available"):
        gates.append("C4_lidar_cross_check")
    all_passed = all(bool(controls[g]["passed"]) for g in gates)
    # ⚠️ A meta that shows `passed: false` beside `all_controls_passed: true`
    # is TRUE AND WRONG FOR THE READER. Every control says whether it GATES,
    # and the one that does not says why in the same object.
    for k, v in controls.items():
        v["gated"] = k in gates
    if exact_mode:
        controls["C7_a_t_s_rounding_flip_cannot_change_zh"]["role"] = (
            "DIAGNOSTIC, NOT A GATE. The rounded t_s was NOT used -- the frame "
            "time was recovered exactly (C9) -- so this control prices the "
            "route NOT taken: how far the heights could have moved had the "
            "build read the line's 4-decimal t_s instead. `passed` here is the "
            "verdict on THAT hypothetical, and it is false on purpose.")

    meta = {
        "task": "B1 EVAL agent join -> 3-D (center_z, size_z) for refcv6 SPEC "
                "v2 section 6",
        "_evidence_class": "MEASURED (ours; artifact = the jsonl + this meta)",
        "builder": "stack/scripts/build_b1_agent_join_3d.py",
        "derived_from": {
            "join_2d": str(Path(join_2d).name),
            "join_2d_md5": md5_of(join_2d),
            "relation": "ANNOTATION, not rebuild: every line, key, field, agent "
                        "and byte of the 2-D join is carried; cz and h are "
                        "APPENDED to each agent dict",
        },
        "key": {"fields": ["clip_id", "frame"],
                "frame_index_space": "RAW v2ep index -- UNCHANGED from the 2-D "
                                     "join (this builder never computes a frame)",
                "also_emitted": "frame_idx (post-n_stack-trim), carried through"},
        "added_fields": {
            CZ_KEY: "cuboid CENTRE height in the rig frame, metres, +z UP, "
                    "z = 0 is the road plane (see convention below)",
            H_KEY: "cuboid height = obstacle.offline size_z, metres",
            "rounding": {CZ_KEY: CZ_ROUND, H_KEY: H_ROUND},
            "base_face": "cz - h/2  (NOT cz)",
        },
        "convention_verdict": {
            "center_z_is": "the cuboid CENTRE",
            "frame": "rig, +x fwd +y LEFT +z UP, origin rear axle on the road "
                     "plane (bev-lidar-gt-b1eval meta.frame_convention)",
            "verified_against": "the banked LiDAR BEV GT's own per-clip "
                                "ground_peak_z_m -- an independent sensor and "
                                "pipeline, VERIFICATION ONLY",
            "retraction_addressed": "RETRACTION_LOG.md:15278",
        },
        "args": {"obstacle_dir": str(obstacle_dir), "tol_s": tol_s,
                 "bevgt_dir": str(bevgt_dir) if bevgt_dir else None,
                 "eps_dir": str(eps_dir) if eps_dir else None,
                 "ego_dir": str(ego_dir) if ego_dir else None,
                 "lead_block": str(lead_block) if lead_block else None,
                 "mutate": mutate, "limit_lines": limit_lines},
        "time_base": {
            "mode": "exact (register_poses_to_time re-run)" if exact_mode
                    else "the banked line's t_s, rounded to 4 decimals",
            "why": "MEASURED on the first smoke run: the rounded t_s leaves 19 "
                   "picks per 400 lines within 1e-4 s of a tie, worst case two "
                   "candidate samples 101 ms apart with 0.714 m between their z",
            "sources": time_sources},
        "summary": {"n_lines": n_lines, "n_clips": len(clips),
                    "n_agents": n_agents, "n_agents_with_zh": n_zh,
                    "n_agents_without_zh": n_missing,
                    "zh_coverage": round(n_zh / n_agents, 6) if n_agents else None,
                    "n_clip_loads": n_clip_loads,
                    "wall_s": round(time.time() - t0, 1)},
        "per_clip": clips,
        "census_by_class": stats.per_class(),
        "census_by_range": {
            "note": "ground-standing vehicle BASE (cz - h/2) by ego "
                    "range. rig z=0 is a FLAT plane and a road is not: "
                    "read the near bins for the convention and the far "
                    "bins for what grade does to it",
            "bins": stats.per_range()},
        "controls": controls,
        "all_controls_passed": all_passed,
        "content_assertion": reread,
    }

    if mutate == "none" and not all_passed:
        failed = [g for g in gates if not bool(controls[g]["passed"])]
        tmp.unlink(missing_ok=True)
        print("B1_JOIN3D_REFUSED %s" % json.dumps(
            {"failed": failed, "controls": {g: controls[g] for g in failed}}),
            flush=True)
        raise ControlFailure("[b1join3d] REFUSING to write: controls %s FAILED "
                             "-- an unproven 3-D join is worse than none"
                             % (failed,))

    tmp.replace(out_path)
    meta["summary"]["md5"] = md5_of(out_path)
    meta["digest_scope"] = {"algo": "md5", "digest": meta["summary"]["md5"],
                            "scope": "compressed" if compressed else "plain",
                            "filename": out_path.name,
                            "declared_by": "builder",
                            "schema": "tanitad.join_digest_scope/1"}
    Path(str(out_path) + ".meta.json").write_text(
        json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def assert_content(out_path, n_expect_lines, compressed, *, raw_2d,
                   limit_lines=None, strict: bool = True) -> dict:
    """CONTENT assertion on the WRITTEN bytes, re-read and re-compared against
    the 2-D source line by line. A pre-allocated file of zeros, a truncated
    write and a good build are indistinguishable by size and exit code."""
    n_lines = n_boxes = n_zh = 0
    z = []
    mismatched = 0
    src = open_lines(raw_2d)
    fh = (lzma.open(str(out_path), "rt", encoding="utf-8") if compressed
          else open(str(out_path), "r", encoding="utf-8"))
    with fh, src:
        for line in fh:
            two = src.readline().rstrip("\n")
            line = line.rstrip("\n")
            rec = json.loads(line)
            if dumps_line(strip_zh(rec)) != two:
                mismatched += 1
            n_lines += 1
            for a in rec["agents"]:
                n_boxes += 1
                if CZ_KEY in a:
                    n_zh += 1
                    z.append(a[CZ_KEY])
    if n_lines != n_expect_lines:
        raise ControlFailure("[b1join3d] REFUSING: wrote %d lines, expected %d"
                             % (n_lines, n_expect_lines))
    if n_lines == 0 or n_boxes == 0 or n_zh == 0:
        raise ControlFailure("[b1join3d] REFUSING: the written join is EMPTY or "
                             "carries no z (%d lines, %d boxes, %d z)"
                             % (n_lines, n_boxes, n_zh))
    if mismatched and strict:
        raise ControlFailure("[b1join3d] REFUSING: %d of %d RE-READ lines do not "
                             "strip back to the 2-D line" % (mismatched, n_lines))
    za = np.asarray(z, dtype=np.float64)
    if not np.isfinite(za).all() or float(np.std(za)) == 0.0:
        raise ControlFailure("[b1join3d] REFUSING: z column is degenerate "
                             "(finite=%s std=%r)" % (bool(np.isfinite(za).all()),
                                                     float(np.std(za))))
    return {"n_lines": n_lines, "n_agent_boxes": n_boxes, "n_with_zh": n_zh,
            "reread_strip_mismatches": mismatched,
            "z_mean_m": round(float(za.mean()), 4),
            "z_std_m": round(float(za.std()), 4)}


# ===========================================================================
def build_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Annotate the banked B1 EVAL agent join with cz/h from "
                    "obstacle.offline, preserving its key and every byte")
    ap.add_argument("--join-2d", required=True,
                    help="the banked 2-D join (.jsonl or .jsonl.xz)")
    ap.add_argument("--obstacle-dir", required=True,
                    help="dir of <clip_id>.parquet obstacle.offline tables")
    ap.add_argument("--out", required=True, help="output .jsonl.xz (ARTIFACT dir)")
    ap.add_argument("--bevgt-dir", default=None,
                    help="banked LiDAR BEV GT dir for the C4 cross-check "
                         "(VERIFICATION ONLY -- never a training target)")
    ap.add_argument("--eps-dir", default=None,
                    help="dir of *.v2ep.pt RAW episode records. With --ego-dir "
                         "the frame time is RECOVERED EXACTLY instead of read "
                         "back from the line's 4-decimal t_s (see exact_t_s)")
    ap.add_argument("--ego-dir", default=None,
                    help="dir of <clip_id>.parquet egomotion tables")
    ap.add_argument("--lead-block", default=None,
                    help="b1_eval_lead_block.npz -- the time base for a clip "
                         "that does not move, where registration is impossible")
    ap.add_argument("--tol-s", type=float, default=None,
                    help="per-track nearest-sample tolerance; default = the "
                         "join's own bev_raster.DEFAULT_TOL_S")
    ap.add_argument("--mutate", default="none", choices=_MUTATIONS,
                    help="deliberately break the build to prove a control "
                         "reads a known value")
    ap.add_argument("--limit-lines", type=int, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    a = build_args(argv)
    from tanitad.data.bev_raster import DEFAULT_TOL_S
    tol_s = DEFAULT_TOL_S if a.tol_s is None else a.tol_s
    print("[b1join3d] join_2d=%s tol_s=%.4f mutate=%s"
          % (Path(a.join_2d).name, tol_s, a.mutate), flush=True)
    meta = build(Path(a.join_2d), Path(a.obstacle_dir), Path(a.out),
                 tol_s=tol_s,
                 bevgt_dir=Path(a.bevgt_dir) if a.bevgt_dir else None,
                 eps_dir=Path(a.eps_dir) if a.eps_dir else None,
                 ego_dir=Path(a.ego_dir) if a.ego_dir else None,
                 lead_block=Path(a.lead_block) if a.lead_block else None,
                 mutate=a.mutate, limit_lines=a.limit_lines)
    print("B1_JOIN3D_DONE %s" % json.dumps({
        "n_lines": meta["summary"]["n_lines"],
        "n_clips": meta["summary"]["n_clips"],
        "n_agents": meta["summary"]["n_agents"],
        "n_agents_with_zh": meta["summary"]["n_agents_with_zh"],
        "md5": meta["summary"].get("md5"),
        "all_controls_passed": meta["all_controls_passed"],
        "mutate": a.mutate}), flush=True)
    return 0 if meta["all_controls_passed"] else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
