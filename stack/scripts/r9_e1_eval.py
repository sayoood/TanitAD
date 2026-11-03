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
def nearfield_per_file(files, stride=1, psw_key="psw", gt_class=SIDEWALK_CLASS):
    """PINNED COPY of the counting in MAPSEL's ``nearfield_prec.py`` (``F:/refcv9_mapsel/...mapedge/diag``): per raster file,
    per band, ``[own flags, flags on GT sidewalk, flags within 0.5 m of one, GT sidewalk cells, GT sidewalk flagged, seen
    cells]`` and the GT class histogram of the own flags. ``files`` is the script's ``sorted(os.listdir(R))`` order and the
    script's ``[::6]`` is ``stride=6``. A file without a ``gt`` plane is skipped, as in the script.
    Returns ``(names, acc [n, 4, 6], gtcls [n, 4, 9])``."""
    from scipy.ndimage import binary_dilation
    c0, c1 = CORRIDOR_COLS
    names, accs, cls_ = [], [], []
    for f in list(files)[::int(stride)]:
        z = np.load(f)
        if "gt" not in z.files:
            continue
        gt, psw = z["gt"], z[psw_key]
        sw = gt == gt_class
        near_sw = binary_dilation(sw, iterations=5)          # within 0.5 m of a GT sidewalk cell
        a = np.zeros((len(NEARFIELD_BANDS), 6))
        g9 = np.zeros((len(NEARFIELD_BANDS), 9))
        for i, (lo, hi) in enumerate(NEARFIELD_BANDS):
            sl = (slice(lo * 10, hi * 10), slice(c0, c1))
            p, g, seen, ns = psw[sl] >= 128, sw[sl], gt[sl] != 255, near_sw[sl]
            p = p & seen
            a[i] = [p.sum(), (p & g).sum(), (p & ns).sum(), g.sum(), (g & p).sum(), seen.sum()]
            cls = gt[sl][p]
            for c in range(8):
                g9[i, c] = (cls == c).sum()
        names.append(Path(str(f)).name)
        accs.append(a)
        cls_.append(g9)
    n = len(names)
    return names, (np.stack(accs) if n else np.zeros((0, 4, 6))), (np.stack(cls_) if n else np.zeros((0, 4, 9)))


def nearfield_table(acc, gtcls) -> dict:
    """band -> the script's printed row: own flag rate, precision exact, precision within 0.5 m, recall, class histogram."""
    out = {}
    for i, (lo, hi) in enumerate(NEARFIELD_BANDS):
        P, TP, TPn, G_, TPr, S_ = [float(x) for x in acc[:, i].sum(0)]
        d = gtcls[:, i].sum(0)[:8]
        d = d / max(d.sum(), 1)
        out["%d-%d" % (lo, hi)] = {"flag_rate": P / S_ if S_ else float("nan"), "precision_exact": TP / max(P, 1),
                                   "precision_within_0p5m": TPn / max(P, 1), "recall": TPr / max(G_, 1),
                                   "gt_class_of_flags": [float(x) for x in d], "n_flags": P, "n_seen": S_}
    return out


def nearfield_episode_of(name: str) -> str:
    m = re.match(r"^(e\d+)_w", name)
    return m.group(1) if m else name


def nearfield_stats(units) -> dict:
    """Pooled corridor precision per band over a list of per-episode count blocks ``(acc [4,6], ...)``."""
    acc = sum(u for u in units)
    out = {}
    for i, (lo, hi) in enumerate(NEARFIELD_BANDS):
        P, TP, TPn, G_, TPr, S_ = [float(x) for x in acc[i]]
        out["nf_prec_%d_%d" % (lo, hi)] = TP / P if P > 0 else float("nan")
        out["nf_prec05_%d_%d" % (lo, hi)] = TPn / P if P > 0 else float("nan")
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
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
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
        names, acc, gtcls = nearfield_per_file(files, a.rast_stride, a.psw_key)
        if not names:
            raise SystemExit("[r9-e1-eval] no raster with a gt plane")
        eps = [nearfield_episode_of(n) for n in names]
        ue = sorted(set(eps))
        units = [sum(acc[i] for i in range(len(names)) if eps[i] == e) for e in ue]
        out["nearfield"] = {"table": nearfield_table(acc, gtcls), "n_files": len(names), "n_episodes": len(ue),
                            "ci": ci.stat_cluster_bootstrap(units, ue, nearfield_stats, n_boot=a.boot, seed=a.seed)}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    print("wrote %s: %d windows / %d clips; AP@2m %.4f" % (a.out, len(keys), len(ids), out["boxes"]["ap2m_all"]["point"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
