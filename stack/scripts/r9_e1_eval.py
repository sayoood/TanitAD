#!/usr/bin/env python3
"""r9_e1_eval.py -- the refcv9 E1 DEV evaluator (REFCV9_DESIGN 4.3): box AP and near-field map precision for ONE arm, or the
PAIRED difference of two arms on the same windows. Tier T0 (label level); estimator: episode-cluster bootstrap (one draw of
EPISODES -- NOT a training-seed floor, which is C1 - C0, see ``taniteval.ci.e1_floor_read``).

It CONSUMES dumps, it does not run a model:
  * BOXES   ``--packs <perc_det.pkl>``  a list of ``{"sha12", "ws", "box3d": [pack]}`` (PERC8's dump format; a pack is
            ``tanitad.eval.detection_metrics.window_packs`` output). The window key is ``(sha12, ws)``, the episode (bootstrap
            unit) is ``sha12``.
  * MAP     ``--rast-dir <dir of e*_w*.npz>`` planes ``gt`` (uint8 class, 255 = unobserved) and ``psw`` (uint8 own sidewalk
            flag, >= 128 = flagged) at 10 cm; the near-field corridor |y| < 3 m, bands 0-5 / 5-10 / 10-20 / 20-40 m.
  * ``--windows <file>`` restricts BOTH to the listed clips (sha12; one per line, a JSON list, or a JSON dict holding the
    split under ``--split dev|hold``); raster names are ``e<episode>_w<ws>.npz`` and are restricted by ``--rast-names``.
  * MAP PRIMARY (REFCV9_DESIGN 4.3, M1): ``nf_swedge_prec_lv_*`` = corridor precision of the UNION of the sidewalk (``psw``) and
    edge (``pedge``) flags against the GT union {5 edge, 7 sidewalk}, on GT-seen AND lift-valid cells. ``--lift-valid`` supplies the
    ARM-INDEPENDENT mask (a sidecar ``.npz``, or the rasters of any arm that wrote an ``unobs`` plane); WITHOUT it NO ``nf_swedge_*`` key
    is emitted at all (absent = NOT-COMPUTABLE; never an unmasked number under a lift-valid name, and the row set of a run that does not
    ask for the primary is unchanged). The historical ``nearfield_prec`` (``nf_prec_*``) is SIDEWALK-ONLY: GT class 7, flag ``psw``,
    GT-seen cells.

Reuse, not re-implementation: ``greedy_rows`` / ``_ap`` / ``_band_of`` / ``BANDS_X_M`` / ``pr_equal_gate`` come from
``tanitad.eval.detection_metrics`` and the post-hoc NMS from ``tanitad.eval.detection_nms``. The two things that were SCRIPTS
(``nearfield_prec.py``, the PERC8 census ``census_one``) are pinned, importable copies here, held equal to the originals by
``tests/test_r9_e1_eval.py``.

THE P = R GATE is a number fitted on a FIT split (``--gate`` or ``--gates-json``). ``--gate-from-scored`` fits it on the
scored windows themselves, which is tuning on the scored split: it is allowed only as a declared deviation and the output
says so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

SCHEMA = "r9e1-eval-v1"
BAND_NAMES = ("0_20", "20_40", "40_60")                   # == detection_metrics.BANDS_X_M (asserted in the test)
NEARFIELD_BANDS = ((0, 5), (5, 10), (10, 20), (20, 40))   # metres ahead; the script's own
CORRIDOR_COLS = (270, 330)                                # |y| < 3 m at 10 cm / cell
SIDEWALK_CLASS = 7
EDGE_CLASS = 5                                            # GT "non-drivable edge" (MAPSEL: 0 nocls 1 drv 2 lane 3 xwalk 4 arrow 5 edge 6 hatch 7 sw)
EDGE_KEY = "pedge"                                        # the edge probability plane every dump writes (r9_e1_dump.write_raster)
# nearfield_per_file's per-band count row: the first 6 are MAPSEL's (``nearfield_prec.py``), the rest are the union / lift-valid counts
NF_BASE_COLS = 6
NF_COLS = 11                                              # 6 + [U_P, U_TP, L_P, L_TP, L_S] -- see nearfield_per_file_ex
LV_ROWS = NEARFIELD_BANDS[-1][1] * 10                     # 0-40 m at 10 cm: the rows a lift-valid mask is stored for
LV_SHAPE = (LV_ROWS, CORRIDOR_COLS[1] - CORRIDOR_COLS[0])
_HEX12 = re.compile(r"^[0-9a-f]{12}$")


def _import_ci():
    try:
        from taniteval import ci
    except ModuleNotFoundError:
        here = Path(__file__).resolve()
        for cand in (here.parents[2] / "taniteval", here.parents[1] / "taniteval"):
            if (cand / "taniteval" / "ci.py").is_file():
                sys.path.insert(0, str(cand))
                break
        from taniteval import ci
    return ci


# --------------------------------------------------------------------------------------------------- windows
def load_window_list(path, split=None) -> list:
    """sha12 list from a text file / JSON list / JSON dict (``split`` picks the key, case-insensitive; ``<split>_sha12`` and
    ``<split>_clips`` also accepted). Every entry is normalised to the first 12 lowercase hex chars; anything else REFUSES."""
    txt = Path(path).read_text(encoding="utf-8")
    try:
        obj = json.loads(txt)
    except ValueError:
        obj = [ln.strip() for ln in txt.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if isinstance(obj, dict):
        if split is None:
            raise SystemExit("[r9-e1-eval] %s is a JSON object: pass --split dev|hold" % path)
        want = {str(split).lower(), str(split).lower() + "_sha12", str(split).lower() + "_clips"}
        hit = [k for k in obj if str(k).lower() in want]
        if not hit:
            for v in obj.values():                                    # one level of nesting ({"splits": {"dev": [...]}})
                if isinstance(v, dict):
                    hit = [k for k in v if str(k).lower() in want]
                    if hit:
                        obj = v
                        break
        if len(hit) != 1:
            raise SystemExit("[r9-e1-eval] %s: split %r not found exactly once (keys %s)" % (path, split, sorted(map(str, obj))[:8]))
        obj = obj[hit[0]]
        if isinstance(obj, dict) and isinstance(obj.get("sha12"), list):    # E0i's DEV_HOLD_SPLIT.json: {"DEV": {"sha12": [...], ...}}
            obj = obj["sha12"]
        elif isinstance(obj, dict):
            obj = list(obj)
    out = []
    for x in obj:
        s = str(x).strip().lower()[:12]
        if not _HEX12.match(s):
            raise SystemExit("[r9-e1-eval] %s: %r is not a sha12" % (path, x))
        out.append(s)
    if not out or len(set(out)) != len(out):
        raise SystemExit("[r9-e1-eval] %s: empty or duplicate window list" % path)
    return sorted(out)


def load_records(pkl_path, clips=None) -> list:
    """[(sha12, ws, pack)] from a PERC8-format dump, restricted to ``clips`` (a set of sha12) when given."""
    with open(pkl_path, "rb") as f:
        det = pickle.load(f)
    keep = None if clips is None else set(clips)
    out = []
    for r in det:
        if keep is not None and str(r["sha12"]) not in keep:
            continue
        if r.get("box3d"):
            out.append((str(r["sha12"]), int(r["ws"]), r["box3d"][0]))
    return out


# --------------------------------------------------------------------------------------------------- per-window items
def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def census_one(pk, gate, nms_cfg=None):
    """PINNED COPY of PERC8's ``perc8_score.census_one`` (the P=R census). ``[tp, n_conf, n_pos, sum_cnt, n_det, n_ge2]`` for
    one window at ``gate``; ``sum_cnt / n_det`` is the boxes per detected object."""
    from tanitad.eval import detection_metrics as dm
    q = pk
    if nms_cfg is not None:
        from tanitad.eval import detection_nms as nms
        q = nms.nms_pack(pk, nms_cfg["radius_m"], nms_cfg.get("p_floor", nms.P_FLOOR))
    rows = dm.greedy_rows(q, dm.PR_DIST_M, None, min_conf=float(gate))
    tp = sum(1 for r in rows if r[1] == 1)
    n_conf = sum(1 for r in rows if r[1] >= 0)
    n_pos = int(q["pos"].sum())
    p = _sig(q["logit"])
    gx = np.asarray(q["gt_xy"])[np.asarray(q["pos"], bool)].astype(np.float64)
    sum_cnt = n_det = n_ge2 = 0
    if len(gx):
        d = np.sqrt(((np.asarray(q["xy"])[None, :, :].astype(np.float64) - gx[:, None, :]) ** 2).sum(-1))
        cnt = ((d <= dm.PR_DIST_M) & (p >= float(gate))[None]).sum(1)
        det1 = cnt[cnt >= 1]
        sum_cnt, n_det, n_ge2 = int(det1.sum()), int(len(det1)), int((det1 >= 2).sum())
    return np.array([tp, n_conf, n_pos, sum_cnt, n_det, n_ge2], np.float64)


def window_item(pk, thresholds, gate, nms_cfg=None) -> dict:
    """Everything one window contributes, as small arrays (the bootstrap pools these, it never re-matches)."""
    from tanitad.eval import detection_metrics as dm
    names = [b[0] for b in dm.BANDS_X_M]
    assert tuple(names) == BAND_NAMES, "detection_metrics.BANDS_X_M changed: %s" % names
    bid = {n: i for i, n in enumerate(names)}
    item = {"rows": {}, "ngt": np.zeros(1 + len(names))}
    pos_idx = np.nonzero(pk["pos"])[0]
    item["ngt"][0] = len(pos_idx)
    for g in pos_idx:
        b = dm._band_of(float(pk["gt_xy"][g, 0]))
        if b is not None:
            item["ngt"][1 + bid[b]] += 1
    for t in thresholds:
        rr = dm.greedy_rows(pk, float(t))
        a = np.array([[c, k, (bid[b] if b in bid else -1)] for (c, k, b, _s, _g) in rr if k >= 0], np.float64).reshape(-1, 3)
        item["rows"][float(t)] = a
        if abs(float(t) - 2.0) < 1e-9:
            # |dx| of the 2 m greedy TPs whose GT lies 0-40 m ahead (the B1 primary), and the TP class confusion
            item["dx"] = np.array([abs(float(pk["xy"][s, 0]) - float(pk["gt_xy"][g, 0])) for (_c, k, _b, s, g) in rr
                                   if k == 1 and 0.0 <= float(pk["gt_xy"][g, 0]) < 40.0], np.float64)
            item["cls"] = np.array([[int(pk["gt_cls"][g]), int(pk["cls"][s])] for (_c, k, _b, s, g) in rr if k == 1], np.int64).reshape(-1, 2)
    if gate is not None:
        item["census"] = census_one(pk, gate, None)
        item["census_band"] = census_band(pk, gate)
        if nms_cfg is not None:
            item["census_nms"] = census_one(pk, gate, nms_cfg)
            from tanitad.eval import detection_nms as nms
            q = nms.nms_pack(pk, nms_cfg["radius_m"], nms_cfg.get("p_floor", nms.P_FLOOR))
            item["rows_nms"] = np.array([[c, k, (bid[b] if b in bid else -1)] for (c, k, b, _s, _g) in dm.greedy_rows(q, 2.0) if k >= 0],
                                        np.float64).reshape(-1, 3)
            item["ngt_nms"] = float(q["pos"].sum())
    return item


def fast_ap(rows: np.ndarray, n_gt: float) -> float:
    """All-point AP, the arithmetic of ``box3d_head.ap_from_rows`` (stable sort on -conf); NaN when there is no GT."""
    if n_gt <= 0:
        return float("nan")
    o = np.argsort(-rows[:, 0], kind="stable")
    k = rows[o, 1]
    tp = np.cumsum(k)
    fp = np.cumsum(1.0 - k)
    rec = tp / n_gt
    prec = tp / np.maximum(tp + fp, 1e-12)
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def clip_item(items) -> dict:
    """Pool the window items of ONE clip into a single item (the bootstrap unit)."""
    out = {"ngt": sum(i["ngt"] for i in items), "rows": {}}
    for t in items[0]["rows"]:
        out["rows"][t] = np.concatenate([i["rows"][t] for i in items])
    for key in ("census", "census_nms", "census_band"):
        if key in items[0]:
            out[key] = sum(i[key] for i in items)
    out["dx"] = np.concatenate([i["dx"] for i in items]) if "dx" in items[0] else np.zeros(0)
    out["cls"] = np.concatenate([i["cls"] for i in items]) if "cls" in items[0] else np.zeros((0, 2), np.int64)
    if "rows_nms" in items[0]:
        out["rows_nms"] = np.concatenate([i["rows_nms"] for i in items])
        out["ngt_nms"] = sum(i["ngt_nms"] for i in items)
    return out


def census_band(pk, gate):
    """refcv9 B3: the P=R census's boxes-per-detected-object, BY RANGE. ``[n_bands, 2]`` = ``(sum_cnt, n_det)`` per range band of the
    GT's x (``detection_metrics.BANDS_X_M``), NO NMS, the SAME arithmetic as :func:`census_one` (``sum_cnt / n_det`` pooled over
    windows is the boxes per detected object of that band). Additive: nothing existing reads it."""
    from tanitad.eval import detection_metrics as dm
    names = [b[0] for b in dm.BANDS_X_M]
    out = np.zeros((len(names), 2), np.float64)
    gt = np.asarray(pk["gt_xy"])[np.asarray(pk["pos"], bool)].astype(np.float64)
    if not len(gt):
        return out
    p = _sig(pk["logit"])
    d = np.sqrt(((np.asarray(pk["xy"])[None, :, :].astype(np.float64) - gt[:, None, :]) ** 2).sum(-1))
    cnt = ((d <= dm.PR_DIST_M) & (p >= float(gate))[None]).sum(1)
    for g in range(len(gt)):
        b = dm._band_of(float(gt[g, 0]))
        if b in names and cnt[g] >= 1:
            out[names.index(b)] += (float(cnt[g]), 1.0)
    return out


def box_stats(clips: list, thresholds=(1.0, 2.0)) -> dict:
    """The scalar metrics of a list of clip items, POOLED over their windows (never a mean of per-window ratios)."""
    out = {}
    ngt = sum(c["ngt"] for c in clips)
    out["n_pos"] = float(ngt[0])
    for t in thresholds:
        rows = np.concatenate([c["rows"][float(t)] for c in clips])
        tag = {0.5: "0p5", 1.0: "1", 2.0: "2", 4.0: "4"}.get(float(t), str(t))
        out["ap%sm_all" % tag] = fast_ap(rows, ngt[0])
        for i, b in enumerate(BAND_NAMES):
            out["ap%sm_%s" % (tag, b)] = fast_ap(rows[rows[:, 2] == i], ngt[1 + i])
        # refcv9 B2's primary "AP@2 m 20-60 m" (REFCV9_DESIGN 4.3): ONE AP over the union of the 20_40 and 40_60 rows and their pooled GT,
        # never the mean of the two band APs (additive key, MM 2026-11-04)
        far = (rows[:, 2] == 1) | (rows[:, 2] == 2)
        out["ap%sm_20_60" % tag] = fast_ap(rows[far], ngt[2] + ngt[3])
    dx = np.concatenate([c["dx"] for c in clips])
    out["dx_0_40_mean"] = float(dx.mean()) if dx.size else float("nan")
    out["dx_0_40_median"] = float(np.median(dx)) if dx.size else float("nan")
    cm = np.concatenate([c["cls"] for c in clips])
    out["tp_class_acc"] = float((cm[:, 0] == cm[:, 1]).mean()) if len(cm) else float("nan")
    for key, tag in (("census", "noNMS"), ("census_nms", "NMS")):
        if key in clips[0]:
            tp, n_conf, n_pos, sum_cnt, n_det, n_ge2 = sum(c[key] for c in clips)
            out["prec_gate_" + tag] = float(tp / n_conf) if n_conf else float("nan")
            out["rec_gate_" + tag] = float(tp / n_pos) if n_pos else float("nan")
            out["boxes_per_object_" + tag] = float(sum_cnt / n_det) if n_det else float("nan")
            out["frac_objects_ge2_" + tag] = float(n_ge2 / n_det) if n_det else float("nan")
    if "census_band" in clips[0]:                      # refcv9 B3: boxes per detected object at P=R, NO NMS, BY RANGE (additive keys)
        cb = sum(c["census_band"] for c in clips)
        for i, b in enumerate(BAND_NAMES):
            out["boxes_per_object_noNMS_" + b] = float(cb[i, 0] / cb[i, 1]) if cb[i, 1] else float("nan")
            out["n_det_noNMS_" + b] = float(cb[i, 1])
    if "rows_nms" in clips[0]:
        out["ap2m_all_NMS"] = fast_ap(np.concatenate([c["rows_nms"] for c in clips]), sum(c["ngt_nms"] for c in clips))
    return out


def build_clip_items(records, thresholds, gate, nms_cfg):
    """records [(sha12, ws, pack)] -> (keys [(sha12, ws)], per-window items) in sorted key order."""
    recs = sorted(records, key=lambda r: (r[0], r[1]))
    return [(r[0], r[1]) for r in recs], [window_item(r[2], thresholds, gate, nms_cfg) for r in recs]


def pool_by_clip(keys, items):
    clips, ids = {}, []
    for (sha, _ws), it in zip(keys, items):
        clips.setdefault(sha, []).append(it)
    ids = sorted(clips)
    return [clip_item(clips[s]) for s in ids], ids


# --------------------------------------------------------------------------------------------------- near-field map
class LiftValidError(RuntimeError):
    """A lift-valid mask that is missing, malformed, or contradicted by a raster's own ``unobs`` plane. Never swallowed: a wrong
    mask would be read as a map result."""


class LiftValid:
    """The ARM-INDEPENDENT lift-validity mask the MAP PRIMARY is read on: ``{episode index e_i: bool [400, 60]}`` (True = the lift
    OBSERVED the cell; 0-40 m ahead at 10 cm, the corridor columns only). It is a property of the CLIP (the rig extrinsics -> the
    lift geometry ``valid.any(dim=1)`` -> ``lift_valid_to_fine``), not of the arm: every arm is scored on the SAME object.
    ``source`` and ``sha256`` (over the sorted masks) go into the output, so two reads on different masks are visible."""

    def __init__(self, masks: dict, source: str):
        self.masks = {}
        for k, v in dict(masks).items():
            v = np.asarray(v, dtype=bool)
            if v.shape != LV_SHAPE:
                raise LiftValidError("lift-valid mask of episode %s has shape %s, expected %s" % (k, tuple(v.shape), LV_SHAPE))
            self.masks[int(k)] = v
        if not self.masks:
            raise LiftValidError("an empty lift-valid mask set (%s)" % source)
        self.source = str(source)
        h = hashlib.sha256()
        for k in sorted(self.masks):
            h.update(("e%d:" % k).encode())
            h.update(np.packbits(self.masks[k]).tobytes())
        self.sha256 = h.hexdigest()

    def record(self) -> dict:
        lo = float(np.mean([m[:50].mean() for m in self.masks.values()]))
        return {"source": self.source, "sha256": self.sha256, "n_episodes": len(self.masks),
                "mean_observed_frac_0_5m": lo, "rows_m": LV_ROWS // 10, "cols": list(CORRIDOR_COLS)}


def _episode_index(name: str) -> int:
    m = re.match(r"^e(\d+)_w", str(name))
    if not m:
        raise LiftValidError("raster name %r is not e<episode>_w<ws>: no episode to look a lift-valid mask up by" % (name,))
    return int(m.group(1))


def harvest_lift_valid(files, source: str = "harvest") -> LiftValid:
    """Read the mask OFF the rasters of an arm that wrote an ``unobs`` plane (r9_e1_dump.write_raster, the C1 adapter): the lift's
    own per-clip validity, which does not depend on the arm. Every raster of one episode must agree (the geometry is per CLIP; a
    disagreement means the mask is NOT a clip property and is refused). Rasters without ``unobs`` are skipped; none at all is an error."""
    c0, c1 = CORRIDOR_COLS
    masks, n_seen = {}, 0
    for f in files:
        z = np.load(f)
        if "unobs" not in z.files:
            continue
        un = np.asarray(z["unobs"])
        if un.ndim != 2 or un.shape[0] < LV_ROWS or un.shape[1] < c1:
            raise LiftValidError("%s: unobs plane %s is smaller than the 0-40 m corridor" % (f, tuple(un.shape)))
        lv = ~un[:LV_ROWS, c0:c1].astype(bool)
        e = _episode_index(Path(str(f)).name)
        n_seen += 1
        if e in masks and not np.array_equal(masks[e], lv):
            raise LiftValidError("%s: the unobs plane of episode e%d differs from an earlier window of the same episode "
                                 "(%d cells): the lift validity is a per-clip geometry and must not move" % (f, e, int((masks[e] != lv).sum())))
        masks.setdefault(e, lv)
    if not masks:
        raise LiftValidError("no raster under %s carries an `unobs` plane: nothing to harvest a lift-valid mask from" % source)
    return LiftValid(masks, "%s (%d rasters, %d episodes)" % (source, n_seen, len(masks)))


def save_lift_valid(lv: LiftValid, path) -> None:
    np.savez_compressed(str(path), **{"e%d" % k: v for k, v in lv.masks.items()})


def load_lift_valid(path) -> LiftValid:
    """``path`` = a sidecar ``.npz`` (keys ``e<episode>``, bool [400, 60]) OR a directory of rasters of an arm with ``unobs`` planes."""
    p = Path(str(path))
    if p.is_dir():
        return harvest_lift_valid(sorted(str(q) for q in p.glob("*.npz")), "harvest:%s" % p.as_posix())
    if not p.is_file():
        raise LiftValidError("--lift-valid %s is neither a sidecar .npz nor a raster directory" % p)
    z = np.load(p)
    masks = {}
    for k in z.files:
        if not re.fullmatch(r"e\d+", k):
            raise LiftValidError("%s: key %r is not e<episode>" % (p, k))
        masks[int(k[1:])] = z[k]
    return LiftValid(masks, "sidecar:%s" % p.name)


# --------------------------------------------------------------------------------------------------- the FIT-split decision thresholds
THR_SCHEMA = "tanitad.r9_map_thresholds/1"
FIXED_LEVEL = 128                                         # the historical fixed cut: flag = plane >= 128 (what every unset run still does)


class ThresholdsError(RuntimeError):
    """A map-thresholds file that is missing, malformed, or cannot be applied to the read's bands. Never swallowed: a wrong cut would
    silently move every flag."""


class MapThresholds:
    """Per-plane, per-BAND uint8 cut levels (REFCV9_DESIGN 2.1 decision rule: "per-channel, per-band thresholds fitted on a FIT split of TRAIN
    clips; never on eval"), written by ``r9_map_fit_thresholds.py fit``. A flag is ``plane_u8 >= level``. ``planes`` = {plane name (``psw``,
    ``pedge``, ``pdrv``): [level per band]}, ``bands_m`` = [[lo, hi), ...] metres ahead, contiguous from 0. A plane the file does not carry keeps
    the fixed 128 (and :meth:`applied` says so). ``sha256`` is the FILE's: it is the protocol stamp the eval / floor / verdict compare."""

    def __init__(self, planes, bands_m, sha256, source, provenance=None):
        bands = [[float(b[0]), float(b[1])] for b in bands_m]
        if not bands or bands[0][0] != 0.0 or any(b[1] <= b[0] for b in bands) or any(bands[i][1] != bands[i + 1][0] for i in range(len(bands) - 1)):
            raise ThresholdsError("bands_m %s must be non-empty, start at 0 and be contiguous and increasing" % (bands_m,))
        if not planes:
            raise ThresholdsError("a thresholds file with no plane carries nothing")
        self.planes = {}
        for k, v in planes.items():
            if not isinstance(v, (list, tuple)) or len(v) != len(bands):
                raise ThresholdsError("plane %r: %s levels for %d bands" % (k, v if not isinstance(v, (list, tuple)) else len(v), len(bands)))
            lv = []
            for x in v:
                if isinstance(x, bool) or not isinstance(x, int) or not (1 <= x <= 255):
                    raise ThresholdsError("plane %r: level %r is not an integer in 1..255 (a uint8 cut; 0 would flag every cell)" % (k, x))
                lv.append(int(x))
            self.planes[str(k)] = lv
        self.bands_m, self.sha256, self.source, self.provenance = bands, sha256, source, provenance or {}

    @classmethod
    def load(cls, path):
        p = Path(str(path))
        if not p.is_file():
            raise ThresholdsError("--map-thresholds %s is not a file" % p)
        raw = p.read_bytes()
        try:
            obj = json.loads(raw.decode("utf-8"))
        except ValueError as e:
            raise ThresholdsError("%s: not JSON (%s)" % (p, e))
        if not isinstance(obj, dict) or obj.get("schema") != THR_SCHEMA:
            raise ThresholdsError("%s: schema %r != %r" % (p, obj.get("schema") if isinstance(obj, dict) else None, THR_SCHEMA))
        return cls(obj.get("planes") or {}, obj.get("bands_m") or [], hashlib.sha256(raw).hexdigest(), p.name, obj.get("provenance"))

    def level(self, plane, lo_m, hi_m):
        """The cut for ``plane`` over the read band [lo_m, hi_m). 128 for a plane the file lacks. A read band that straddles two fit bands
        with DIFFERENT levels refuses (the cut would be ambiguous); inside one band, or across bands that agree, it is that level."""
        if plane not in self.planes:
            return FIXED_LEVEL
        hit = [i for i, (a, b) in enumerate(self.bands_m) if a < hi_m and b > lo_m]
        if not hit or lo_m < self.bands_m[0][0] or hi_m > self.bands_m[-1][1]:
            raise ThresholdsError("read band %s-%s m is outside the thresholds' bands %s" % (lo_m, hi_m, self.bands_m))
        lv = {self.planes[plane][i] for i in hit}
        if len(lv) != 1:
            raise ThresholdsError("read band %s-%s m straddles fit bands %s with different %s levels %s: fit the thresholds on bands that nest the read's"
                                  % (lo_m, hi_m, [self.bands_m[i] for i in hit], plane, sorted(lv)))
        return lv.pop()

    def applied(self, bands, keys):
        """{"lo-hi": {plane: level}} for the read's bands -- what the flags were actually cut at (stamped into every output)."""
        return {"%d-%d" % (lo, hi): {k: self.level(k, lo, hi) for k in keys} for lo, hi in bands}

    def record(self, bands=None, keys=None):
        r = {"schema": THR_SCHEMA, "sha256": self.sha256, "source": self.source, "bands_m": self.bands_m, "planes": self.planes,
             "absent_planes_keep": FIXED_LEVEL}
        if bands is not None:
            r["applied"] = self.applied(bands, keys)
        return r


def load_map_thresholds(path):
    return MapThresholds.load(path)


def thresholds_binding(thr, rast_dir) -> dict:
    """A thresholds file is bound to the CHECKPOINT whose planes it was fitted on (``provenance.ckpt_md5``, from the FIT dump's dump.json): when the
    raster dir's own ``../dump.json`` names another checkpoint this REFUSES. No dump.json beside the rasters (or an unbound file) is reported as
    UNCHECKED -- never as bound."""
    want = (thr.provenance or {}).get("ckpt_md5")
    dj = Path(str(rast_dir)).parent / "dump.json"
    if not want:
        return {"checked": False, "why": "the thresholds file carries no provenance.ckpt_md5"}
    if not dj.is_file():
        return {"checked": False, "why": "no dump.json beside %s" % rast_dir}
    got = json.loads(dj.read_text(encoding="utf-8")).get("ckpt_md5")
    if not got:
        return {"checked": False, "why": "dump.json carries no ckpt_md5"}
    if got != want:
        raise ThresholdsError("the thresholds (%s) were fitted on checkpoint md5 %s, the rasters under %s come from checkpoint md5 %s: a cut is bound to "
                              "its own model" % (thr.source, want, rast_dir, got))
    return {"checked": True, "ckpt_md5": got}


def thresholds_sha(rec):
    return None if not rec else rec.get("sha256")



def nearfield_per_file_ex(files, stride=1, psw_key="psw", gt_class=SIDEWALK_CLASS, lift_valid=None, edge_key=EDGE_KEY, thresholds=None):
    """:func:`nearfield_per_file` plus the SIDEWALK+EDGE union counts and an ``info`` record.

    ``acc [n, 4, 11]`` per file / band: columns 0-5 are MAPSEL's, UNCHANGED (``[own flags, flags on GT sidewalk, flags within 0.5 m
    of one, GT sidewalk cells, GT sidewalk flagged, seen cells]``, sidewalk flag ``psw >= 128``, GT class ``gt_class``, GT-seen
    cells); columns 6-10 are the M1 map-primary counts, with
      * UNION flag ``f = (psw >= 128) | (pedge >= 128)`` -- the edge plane is used where the raster HAS one, else ``f`` is the
        sidewalk flag alone (``info["edge_plane"]`` says which), scored against the SAME GT union ``{edge 5, sidewalk gt_class}``;
      * 6 ``U_P`` = |f & seen|, 7 ``U_TP`` = |f & seen & GT union| (lift-INDEPENDENT);
      * 8 ``L_P`` = |f & seen & lift-valid|, 9 ``L_TP`` = |f & seen & lift-valid & GT union|, 10 ``L_S`` = |seen & lift-valid|
        -- NaN in every file when no ``lift_valid`` was given (a count that was not measured is not 0).
    A raster that carries its own ``unobs`` plane must AGREE with the supplied mask (0-40 m corridor) or this raises.

    ``thresholds`` (a :class:`MapThresholds`, default None) replaces the fixed 128 of BOTH flags with the file's per-plane, per-band level
    (``psw_key`` / ``edge_key`` planes; a plane the file lacks keeps 128). None => every flag is ``>= 128`` exactly as before, and the
    ``info`` record carries no ``thresholds`` key: an unset run is byte-identical to the one before this argument existed."""
    from scipy.ndimage import binary_dilation
    c0, c1 = CORRIDOR_COLS
    names, accs, cls_ = [], [], []
    n_edge = n_checked = 0
    for f in list(files)[::int(stride)]:
        z = np.load(f)
        if "gt" not in z.files:
            continue
        gt, psw = z["gt"], z[psw_key]
        has_edge = edge_key in z.files
        n_edge += int(has_edge)
        sw = gt == gt_class
        gtu = sw | (gt == EDGE_CLASS)
        edge = z[edge_key] if has_edge else None
        near_sw = binary_dilation(sw, iterations=5)          # within 0.5 m of a GT sidewalk cell
        name = Path(str(f)).name
        lvf = None
        if lift_valid is not None:
            e = _episode_index(name)
            lvf = lift_valid.masks.get(e)
            if lvf is None:
                raise LiftValidError("no lift-valid mask for episode e%d (raster %s) in %s" % (e, name, lift_valid.source))
            if "unobs" in z.files:                            # an arm that wrote its own validity must agree with the shared mask
                own = ~np.asarray(z["unobs"])[:LV_ROWS, c0:c1].astype(bool)
                if own.shape != lvf.shape or not np.array_equal(own, lvf):
                    raise LiftValidError("%s: its own `unobs` plane disagrees with the supplied lift-valid mask (%s) on %d of %d cells"
                                         % (name, lift_valid.source, int((own != lvf).sum()) if own.shape == lvf.shape else -1, lvf.size))
                n_checked += 1
        a = np.zeros((len(NEARFIELD_BANDS), NF_COLS))
        g9 = np.zeros((len(NEARFIELD_BANDS), 9))
        for i, (lo, hi) in enumerate(NEARFIELD_BANDS):
            sl = (slice(lo * 10, hi * 10), slice(c0, c1))
            t_sw = FIXED_LEVEL if thresholds is None else thresholds.level(psw_key, lo, hi)
            t_ed = FIXED_LEVEL if thresholds is None else thresholds.level(edge_key, lo, hi)
            p0 = psw[sl] >= t_sw
            p, g, seen, ns = p0, sw[sl], gt[sl] != 255, near_sw[sl]
            p = p & seen
            a[i, :NF_BASE_COLS] = [p.sum(), (p & g).sum(), (p & ns).sum(), g.sum(), (g & p).sum(), seen.sum()]
            fu = (((p0) | (edge[sl] >= t_ed)) if has_edge else p0) & seen
            a[i, 6], a[i, 7] = fu.sum(), (fu & gtu[sl]).sum()
            if lvf is None:
                a[i, 8:11] = np.nan
            else:
                lvb = lvf[lo * 10:hi * 10]
                fl = fu & lvb
                a[i, 8], a[i, 9], a[i, 10] = fl.sum(), (fl & gtu[sl]).sum(), (seen & lvb).sum()
            cls = gt[sl][p]
            for c in range(8):
                g9[i, c] = (cls == c).sum()
        names.append(name)
        accs.append(a)
        cls_.append(g9)
    n = len(names)
    info = {"n_files": n, "n_with_edge_plane": n_edge,
            "edge_plane": ("none: the union flag is the SIDEWALK flag alone" if n_edge == 0 else "all" if n_edge == n else "MIXED"),
            "n_unobs_checked": n_checked, "lift_valid": None if lift_valid is None else lift_valid.record()}
    if thresholds is not None:                                # only a thresholded read carries the stamp: an unset run's info is unchanged
        info["thresholds"] = thresholds.record(NEARFIELD_BANDS, (psw_key, edge_key))
    return (names, (np.stack(accs) if n else np.zeros((0, 4, NF_COLS))), (np.stack(cls_) if n else np.zeros((0, 4, 9))), info)


def nearfield_per_file(files, stride=1, psw_key="psw", gt_class=SIDEWALK_CLASS, lift_valid=None, thresholds=None):
    """PINNED COPY of the counting in MAPSEL's ``nearfield_prec.py`` (``F:/refcv9_mapsel/...mapedge/diag``): per raster file,
    per band, ``[own flags, flags on GT sidewalk, flags within 0.5 m of one, GT sidewalk cells, GT sidewalk flagged, seen
    cells]`` and the GT class histogram of the own flags. ``files`` is the script's ``sorted(os.listdir(R))`` order and the
    script's ``[::6]`` is ``stride=6``. A file without a ``gt`` plane is skipped, as in the script.
    Returns ``(names, acc [n, 4, 11], gtcls [n, 4, 9])``: columns 0-5 of ``acc`` are the script's, 6-10 the sidewalk+edge union /
    lift-valid counts of :func:`nearfield_per_file_ex` (which also returns the ``info`` record)."""
    return nearfield_per_file_ex(files, stride, psw_key, gt_class, lift_valid, thresholds=thresholds)[:3]


def nearfield_table(acc, gtcls) -> dict:
    """band -> the script's printed row: own flag rate, precision exact, precision within 0.5 m, recall, class histogram."""
    out = {}
    for i, (lo, hi) in enumerate(NEARFIELD_BANDS):
        P, TP, TPn, G_, TPr, S_ = [float(x) for x in acc[:, i, :NF_BASE_COLS].sum(0)]
        d = gtcls[:, i].sum(0)[:8]
        d = d / max(d.sum(), 1)
        out["%d-%d" % (lo, hi)] = {"flag_rate": P / S_ if S_ else float("nan"), "precision_exact": TP / max(P, 1),
                                   "precision_within_0p5m": TPn / max(P, 1), "recall": TPr / max(G_, 1),
                                   "gt_class_of_flags": [float(x) for x in d], "n_flags": P, "n_seen": S_}
        if acc.shape[2] >= NF_COLS and np.isfinite(acc[:, i, 10]).all() and S_:       # the mask's coverage of the GT-seen cells
            sl_ = float(acc[:, i, 10].sum())
            out["%d-%d" % (lo, hi)].update({"n_seen_lift_valid": sl_, "lift_invalid_frac_of_seen": 1.0 - sl_ / S_})
    return out


def nearfield_episode_of(name: str) -> str:
    m = re.match(r"^(e\d+)_w", name)
    return m.group(1) if m else name


def nearfield_stats(units) -> dict:
    """Pooled corridor precision per band over a list of per-episode count blocks ``(acc [4,6], ...)``."""
    acc = sum(u for u in units)
    out = {}
    for i, (lo, hi) in enumerate(NEARFIELD_BANDS):
        P, TP, TPn, G_, TPr, S_ = [float(x) for x in acc[i][:NF_BASE_COLS]]
        out["nf_prec_%d_%d" % (lo, hi)] = TP / P if P > 0 else float("nan")
        out["nf_prec05_%d_%d" % (lo, hi)] = TPn / P if P > 0 else float("nan")
    # refcv9 T1's primary "corridor precision 0-10 m" (REFCV9_DESIGN 4.3): the 0-5 and 5-10 bands POOLED (counts summed, not the mean of the
    # two precisions). Cells are masked only by GT-seen (gt != 255), never by lift validity, so lift-invalid cells are included (additive key)
    assert NEARFIELD_BANDS[:2] == ((0, 5), (5, 10)), NEARFIELD_BANDS
    P, TP = float(acc[0][0] + acc[1][0]), float(acc[0][1] + acc[1][1])
    out["nf_prec_0_10"] = TP / P if P > 0 else float("nan")
    # refcv9 M1's MAP PRIMARY (REFCV9_DESIGN 4.3 / 4.4): corridor precision of the sidewalk+edge UNION (flags) against the GT union {5, 7},
    # the 5-10 and 10-20 m bands POOLED by counts (never the mean of two precisions). ``nf_swedge_prec_*`` = GT-seen cells only (lift-INDEPENDENT,
    # includes the camera blind zone); ``nf_swedge_prec_lv_*`` = GT-seen AND lift-valid cells (MAP-2's "on lift-valid cells"). BOTH sets exist
    # only for a block measured WITH a mask (finite lift-valid columns): a legacy 6-column block, or one measured without a mask (NaN columns),
    # emits none of them -- so a run that never asked for the primary has exactly the rows it had (test_r9_e1_floor pins the row count).
    if acc.shape[1] >= NF_COLS and np.isfinite(acc[:, 8:11]).all():
        assert NEARFIELD_BANDS[1:3] == ((5, 10), (10, 20)), NEARFIELD_BANDS
        ratio = lambda tp, p: float(tp) / float(p) if p > 0 else float("nan")                                  # noqa: E731
        for tag, (cp, ctp) in (("", (6, 7)), ("lv_", (8, 9))):
            for i, (lo, hi) in enumerate(NEARFIELD_BANDS):
                out["nf_swedge_prec_%s%d_%d" % (tag, lo, hi)] = ratio(acc[i][ctp], acc[i][cp])
            out["nf_swedge_prec_%s5_20" % tag] = ratio(acc[1][ctp] + acc[2][ctp], acc[1][cp] + acc[2][cp])
    return out


# --------------------------------------------------------------------------------------------------- CLI
def _gate_from_args(a, records):
    if a.gate is not None:
        return float(a.gate), {"source": "--gate", "value": float(a.gate)}
    if a.gates_json:
        from tanitad.eval import detection_metrics as dm
        g, stamp = dm.load_head_gates(a.gates_json)
        return float(g["box3d"]), {"source": "--gates-json", "sha256": stamp["sha256"], "value": float(g["box3d"])}
    if a.gate_from_scored:
        from tanitad.eval import detection_metrics as dm
        r = dm.pr_equal_gate([r[2] for r in records])
        return float(r["gate"]), {"source": "SCORED WINDOWS (declared deviation: tuned on the split it is read on)", **r}
    raise SystemExit("[r9-e1-eval] no P=R gate: pass --gate <p> or --gates-json <fit-split gates> (or --gate-from-scored, declared)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--packs", required=True)
    ap.add_argument("--packs-b", help="a second arm's dump: adds the PAIRED difference (A - B) on the shared windows")
    ap.add_argument("--windows", help="clip sha12 list / split file")
    ap.add_argument("--split", default=None)
    ap.add_argument("--gate", type=float)
    ap.add_argument("--gates-json")
    ap.add_argument("--gate-from-scored", action="store_true")
    ap.add_argument("--nms-json", help="tanitad.det_nms/1 file: adds the with-NMS columns (BOX-1 / BOX-3 read WITHOUT)")
    ap.add_argument("--rast-dir")
    ap.add_argument("--rast-names", help="text file of raster stems to keep (e<ep>_w<ws>)")
    ap.add_argument("--rast-stride", type=int, default=1)
    ap.add_argument("--psw-key", default="psw")
    ap.add_argument("--lift-valid", help="the ARM-INDEPENDENT lift-validity mask for the MAP PRIMARY (nf_swedge_prec_lv_*): a sidecar .npz "
                                         "(keys e<episode>, bool [400, 60]) or a directory of rasters of an arm that wrote `unobs` planes; "
                                         "absent -> the _lv_ keys are not emitted")
    ap.add_argument("--lift-valid-save", help="write the mask that --lift-valid resolved to as a sidecar .npz (a harvest -> a sidecar)")
    ap.add_argument("--map-thresholds", help="tanitad.r9_map_thresholds/1 (r9_map_fit_thresholds.py fit): this arm's OWN FIT-split per-plane, per-band uint8 "
                                             "cuts, replacing the fixed psw / pedge >= 128 of the near-field flags (a plane the file lacks keeps 128). Needs "
                                             "--rast-dir. Unset -> the fixed 128 and an output with no `thresholds` key, exactly as before")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.map_thresholds and not a.rast_dir:
        raise SystemExit("[r9-e1-eval] --map-thresholds needs --rast-dir: thresholds with no rasters would be silently ignored")
    ci = _import_ci()
    clips = load_window_list(a.windows, a.split) if a.windows else None
    recs = load_records(a.packs, clips)
    if not recs:
        raise SystemExit("[r9-e1-eval] no windows left after --windows")
    gate, gate_rec = _gate_from_args(a, recs)
    nms_cfg = None
    if a.nms_json:
        from tanitad.eval import detection_nms as nms
        cfgs, _stamp = nms.load_head_nms(a.nms_json)
        nms_cfg = cfgs["box3d"] if "box3d" in cfgs else cfgs
    thr = (1.0, 2.0)
    keys, items = build_clip_items(recs, thr, gate, nms_cfg)
    cl, ids = pool_by_clip(keys, items)
    out = {"schema": SCHEMA, "tier": "T0 (label level)", "packs": str(a.packs), "windows_file": a.windows, "split": a.split,
           "n_windows": len(keys), "n_clips": len(ids), "gate": gate_rec,
           "definitions": {"ap": "tanitad.eval.detection_metrics greedy 2 m / 1 m rows, pooled AP, head output (NO post-hoc NMS) unless a column says NMS",
                           "dx_0_40": "|x_pred - x_gt| of 2 m greedy TPs whose GT is 0-40 m ahead", "tp_class_acc": "class of the 2 m greedy TPs (all slots)",
                           "boxes_per_object": "PERC8 census_one at the gate, 2 m, no NMS"},
           "estimator": "episode-cluster bootstrap over clips, B=%d seed=%d (answers another draw of EPISODES, not another TRAINING run)" % (a.boot, a.seed)}
    stat = lambda cs: box_stats(cs, thr)                                         # noqa: E731
    out["boxes"] = ci.stat_cluster_bootstrap(cl, ids, stat, n_boot=a.boot, seed=a.seed)
    if a.packs_b:
        recs_b = load_records(a.packs_b, clips)
        keys_b, items_b = build_clip_items(recs_b, thr, gate, nms_cfg)
        ia, ib, eid, rep = ci.align_windows(keys, items, keys_b, items_b)
        ka = sorted(set(keys) & set(keys_b))
        ca, ida = pool_by_clip(ka, ia)
        cb, idb = pool_by_clip(ka, ib)
        assert ida == idb
        out["paired_vs_B"] = {"alignment": rep, "boxes": ci.paired_stat_cluster_bootstrap(
            ca, cb, ida, lambda cs: box_stats(cs, thr), n_boot=a.boot, seed=a.seed)}
    if a.rast_dir:
        files = sorted(str(p) for p in Path(a.rast_dir).glob("*.npz"))
        if a.rast_names:
            keep = {ln.strip() for ln in Path(a.rast_names).read_text(encoding="utf-8").splitlines() if ln.strip()}
            files = [f for f in files if Path(f).stem in keep]
        try:
            lvm = load_lift_valid(a.lift_valid) if a.lift_valid else None
            thr_m = load_map_thresholds(a.map_thresholds) if a.map_thresholds else None
            thr_bind = thresholds_binding(thr_m, a.rast_dir) if thr_m is not None else None
            names, acc, gtcls, nfinfo = nearfield_per_file_ex(files, a.rast_stride, a.psw_key, lift_valid=lvm, thresholds=thr_m)
        except LiftValidError as e:
            raise SystemExit("[r9-e1-eval] lift-valid mask refused: %s" % e)
        except ThresholdsError as e:
            raise SystemExit("[r9-e1-eval] map thresholds refused: %s" % e)
        if not names:
            raise SystemExit("[r9-e1-eval] no raster with a gt plane")
        if nfinfo["edge_plane"] == "MIXED":
            raise SystemExit("[r9-e1-eval] %d of %d rasters carry a `%s` plane: one arm, one plane set" % (
                nfinfo["n_with_edge_plane"], nfinfo["n_files"], EDGE_KEY))
        if a.lift_valid_save:
            if lvm is None:
                raise SystemExit("[r9-e1-eval] --lift-valid-save needs --lift-valid")
            save_lift_valid(lvm, a.lift_valid_save)
        eps = [nearfield_episode_of(n) for n in names]
        ue = sorted(set(eps))
        units = [sum(acc[i] for i in range(len(names)) if eps[i] == e) for e in ue]
        out["nearfield"] = {"table": nearfield_table(acc, gtcls), "n_files": len(names), "n_episodes": len(ue),
                            "edge_plane": nfinfo["edge_plane"], "lift_valid": nfinfo["lift_valid"], "n_unobs_checked": nfinfo["n_unobs_checked"],
                            "definition": {"nf_prec_*": "SIDEWALK only: flag psw >= 128, GT class 7, GT-seen cells (MAPSEL nearfield_prec.py, the historical figure)",
                                           "nf_swedge_prec_*": "UNION: flag (psw >= 128) | (pedge >= 128) [sidewalk alone when the raster has no edge plane], "
                                                               "GT {5 edge, 7 sidewalk}, GT-seen cells (includes the camera blind zone)",
                                           "nf_swedge_prec_lv_*": "as nf_swedge_prec_* AND lift-valid cells only (the --lift-valid mask); every nf_swedge_* key is absent without a mask",
                                           "*_5_20": "the 5-10 and 10-20 m bands pooled by counts"},
                            "ci": ci.stat_cluster_bootstrap(units, ue, nearfield_stats, n_boot=a.boot, seed=a.seed)}
        if thr_m is not None:                                  # stamped only when set: an unset run's JSON is unchanged
            out["nearfield"]["thresholds"] = dict(nfinfo["thresholds"], binding=thr_bind)
            out["nearfield"]["definition"]["flag_rule"] = ("the flags are `plane_u8 >= level` at THIS arm's own FIT-split per-band level "
                                                           "(`thresholds.applied`), not the fixed 128 the *_prec definitions above name")
    Path(a.out).write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    print("wrote %s: %d windows / %d clips; AP@2m %.4f" % (a.out, len(keys), len(ids), out["boxes"]["ap2m_all"]["point"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
