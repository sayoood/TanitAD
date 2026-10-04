"""refcv8 WP-C F4b -- a centre-distance NMS on the box heads' detections, with its re-fitted gate. OPT-IN, READ-ONLY.

Source of the fix: ``TanitAD Research Lab/Architecture & Inference/Research/2026-10-04-refcv7-route-following``
(RESULT.md sec. 4, ``code/analyze_box_nms.py``, ``raw/box_nms.json``) on refcv7-r101-s0 at step 50,400.

THE DEFECT (MEASURED, EVAL, box3d head at the in-run gate 0.2589): a detected object carries **2.12 boxes on average**
(66.5 % have >= 2, 46.8 % of the false positives sit within 2 m of a ground-truth box) and the duplicates' presence scores
differ by a median of 0.033: the slot decoder has no duplicate suppression and a FLAT presence score. D3 shows the GT
carries no such duplicates (0.34 % of boxes, no repeated track ids), so the duplicates are prediction-side.

THE FIX (inference only): score-ordered greedy suppression, all classes together, on the slots with ``p >= P_FLOOR``
(the flat tail below it, whose TP rate is ~0, is untouched in every arm): a slot is suppressed when its BEV centre is within
``radius_m`` of an already-kept, higher-scored slot. Radius fitted on TRAIN by AP@2 m (tie -> the milder, smaller radius):
**box3d 2.5 m, agent 3.0 m**; the operating gate moves WITH it (the TRAIN P = R gate re-fitted after NMS: 0.2145 / 0.1809).
EVAL (route package, paired episode-cluster CIs there): box3d AP@2 m 0.248 -> 0.349, F1 0.303 -> 0.392; agent AP@2 m
0.131 -> 0.300, F1 0.207 -> 0.368; boxes per detected object 2.12 -> 1.07 (box3d), 2.60 -> 1.01 (agent). ⚠️ At the OLD gate
0.2589 the NMS halves conf_ratio to 0.43 (out of the [0.5, 1.5] band): radius and gate are ONE operating point.

⛔ THE PLANNER MUST NOT CONSUME THIS. ``refc_agents.AgentTokenEmbed`` reads the RAW ``sigmoid(presence_logit)`` as a feature
and a soft token scale (``presence_hard`` False); an NMS'd or re-gated presence would be a distribution shift for a planner
trained on the flat scores. This module therefore (a) works on DETACHED per-window numpy packs / arrays only -- it holds no
reference to a tensor with a graph and never writes back; (b) never mutates its inputs (a new pack is returned); and (c) is
imported by NO module under ``tanitad/models`` or ``tanitad/refs`` -- pinned by ``tests/test_refcv8_det_nms.py``.

Scope notes: the suppression ignores the class (the measured arm); the Hungarian ``matched`` flags of a pack refer to the
ORIGINAL slots, so an NMS'd pack must not be fed to ``detection_metrics.summarise``'s AUROC -- only to the greedy-row
statistics (AP, census, duplicates), which is all this module emits.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib

import numpy as np

from tanitad.eval import detection_metrics as _det

__all__ = ["NMS_SCHEMA", "P_FLOOR", "R_GRID", "PER_SLOT_KEYS", "NMS_KEYS", "centre_nms_keep", "nms_pack", "nms_packs",
           "duplicate_stats", "ap2m", "fit_nms", "load_head_nms", "nms_census_keys", "nms_key_names"]

NMS_SCHEMA = "tanitad.det_nms/1"
#: NMS acts on slots with sigmoid(logit) >= P_FLOOR; the flat tail below is left untouched (route package, SPEC deviation).
P_FLOOR: float = 0.05
#: the radius grid the route package searched (centre distance, metres).
R_GRID: tuple = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
#: the per-slot arrays of a pack (``detection_metrics.window_packs``); everything else is per-GT and passes through.
PER_SLOT_KEYS: tuple = ("logit", "xy", "cls", "cls_corr", "matched", "exempt")
#: suffixes of the emitted keys, ``eval_{head}_nms_<suffix>``.
NMS_KEYS: tuple = ("radius_m", "gate", "n_conf", "tp", "n_pos", "prec", "rec", "f1", "conf_ratio", "conf_ratio_alarm",
                   "ap2m", "boxes_per_object", "frac_objects_ge2")


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


# --------------------------------------------------------------------------------------------------------- #
# the suppression                                                                                            #
# --------------------------------------------------------------------------------------------------------- #
def centre_nms_keep(p, xy, radius_m: float, p_floor: float = P_FLOOR) -> np.ndarray:
    """Boolean keep-mask over the slots. ``p`` [N] presence probabilities, ``xy`` [N, 2] BEV centres (metres).

    Slots with ``p < p_floor`` are ALWAYS kept. Of the others, in descending ``p`` (stable: ties keep the lower slot
    index), a slot is dropped iff its centre is within ``radius_m`` (inclusive) of an already-kept slot. Distances use
    ``math.hypot`` on float64 so the result is the route package's ``route_metrics.nms_keep`` bit for bit."""
    p = np.asarray(p, np.float64)
    xy = np.asarray(xy, np.float64)
    if p.ndim != 1 or xy.shape != (p.shape[0], 2):
        raise ValueError(f"p {p.shape} / xy {xy.shape}: want [N] and [N, 2]")
    r = float(radius_m)
    if not (math.isfinite(r) and r >= 0.0):
        raise ValueError(f"radius_m {radius_m!r} must be a finite number >= 0")
    hi = p >= float(p_floor)
    keep = ~hi
    cand = np.nonzero(hi)[0]
    order = cand[np.argsort(-p[cand], kind="mergesort")]
    kept = []
    for i in order:
        xi, yi = float(xy[i, 0]), float(xy[i, 1])
        for j in kept:
            if math.hypot(xi - float(xy[j, 0]), yi - float(xy[j, 1])) <= r:
                break
        else:
            kept.append(int(i))
    keep[kept] = True
    return keep


def nms_pack(pk: dict, radius_m: float, p_floor: float = P_FLOOR) -> dict:
    """A NEW pack with the suppressed slots removed from every per-slot array; ``pk`` is not modified."""
    keep = centre_nms_keep(_sig(pk["logit"]), pk["xy"], radius_m, p_floor)
    q = dict(pk)
    for k in PER_SLOT_KEYS:
        if k in q:
            a = np.asarray(q[k])
            if a.shape[0] != keep.shape[0]:
                raise ValueError(f"pack key {k!r} has {a.shape[0]} rows, the slot axis has {keep.shape[0]}")
            q[k] = a[keep]
    return q


def nms_packs(packs: list, radius_m: float, p_floor: float = P_FLOOR) -> list:
    return [nms_pack(pk, radius_m, p_floor) for pk in packs]


# --------------------------------------------------------------------------------------------------------- #
# statistics (greedy-row based, the programme's own matcher)                                                 #
# --------------------------------------------------------------------------------------------------------- #
def ap2m(packs: list) -> float:
    """AP at the 2 m centre-distance, all classes, all bands: ``detection_metrics``' own rows + ``ap_from_rows``."""
    rows = [r for pk in packs for r in _det.greedy_rows(pk, _det.PR_DIST_M, None)]
    n_pos = int(sum(int(pk["pos"].sum()) for pk in packs))
    return _det._ap(rows, n_pos)


def duplicate_stats(packs: list, gate: float) -> dict:
    """Boxes per detected object: per VIS-1 positive GT, the number of slots with ``p >= gate`` whose centre is within
    ``PR_DIST_M`` (2 m); statistics over the positives with >= 1 such slot. NaN when nothing is detected."""
    cnt = []
    for pk in packs:
        p = _sig(pk["logit"])
        gx = np.asarray(pk["gt_xy"])[np.asarray(pk["pos"], bool)].astype(np.float64)
        if not len(gx):
            continue
        d = np.sqrt(((np.asarray(pk["xy"])[None, :, :].astype(np.float64) - gx[:, None, :]) ** 2).sum(-1))
        cnt += ((d <= _det.PR_DIST_M) & (p >= float(gate))[None]).sum(1).tolist()
    cnt = np.asarray(cnt)
    det1 = cnt[cnt >= 1]
    if not len(det1):
        return {"n_detected_objects": 0, "boxes_per_object": float("nan"), "frac_objects_ge2": float("nan")}
    return {"n_detected_objects": int(len(det1)), "boxes_per_object": float(det1.mean()),
            "frac_objects_ge2": float((det1 >= 2).mean())}


def _census(packs: list, gate: float) -> dict:
    rows = [r for pk in packs for r in _det.greedy_rows(pk, _det.PR_DIST_M, None, min_conf=float(gate))]
    tp = sum(1 for r in rows if r[1] == 1)
    n_conf = sum(1 for r in rows if r[1] >= 0)
    n_pos = int(sum(int(pk["pos"].sum()) for pk in packs))
    prec, rec = _det._div(tp, n_conf), _det._div(tp, n_pos)
    f1 = 2 * prec * rec / (prec + rec) if (prec == prec and rec == rec and (prec + rec) > 0) else float("nan")
    cr = _det._div(n_conf, n_pos)
    lo, hi = _det.CONF_RATIO_BAND
    alarm = float("nan") if cr != cr else (0.0 if lo <= cr <= hi else 1.0)
    return {"n_conf": float(n_conf), "tp": float(tp), "n_pos": float(n_pos), "prec": prec, "rec": rec, "f1": f1,
            "conf_ratio": cr, "conf_ratio_alarm": alarm}


# --------------------------------------------------------------------------------------------------------- #
# the fit (TRAIN only)                                                                                       #
# --------------------------------------------------------------------------------------------------------- #
def fit_nms(train_packs: list, radii=R_GRID, p_floor: float = P_FLOOR) -> dict:
    """Fit the radius AND the operating gate on TRAIN packs of ONE head.

    The radius is the grid value with the highest TRAIN AP@2 m (a tie keeps the smaller, milder radius); the gate is the
    TRAIN P = R point re-fitted AFTER that NMS (``detection_metrics.pr_equal_gate``). Returns the full grid, so the choice is
    auditable, plus the ``none`` row (no NMS) the gain is read against. Nothing here reads an EVAL pack."""
    grid = {}
    base = _det.pr_equal_gate(train_packs)
    grid["none"] = {"radius_m": None, "train_ap2m": ap2m(train_packs), "train_pr_gate": base["gate"],
                    "train_prec_at_gate": base["precision"]}
    for r in radii:
        q = nms_packs(train_packs, r, p_floor)
        g = _det.pr_equal_gate(q)
        grid[f"{float(r):g}"] = {"radius_m": float(r), "train_ap2m": ap2m(q), "train_pr_gate": g["gate"],
                                 "train_prec_at_gate": g["precision"]}
    cands = [(v["train_ap2m"], -v["radius_m"], v) for k, v in grid.items() if k != "none"]
    best = max(cands, key=lambda t: (t[0], t[1]))[2]          # highest AP; tie -> smaller radius
    return {"radius_m": best["radius_m"], "gate": best["train_pr_gate"], "train_ap2m": best["train_ap2m"],
            "p_floor": float(p_floor), "grid": grid, "n_windows": len(train_packs)}


# --------------------------------------------------------------------------------------------------------- #
# the config file + the OPT-IN readout                                                                       #
# --------------------------------------------------------------------------------------------------------- #
def load_head_nms(path) -> tuple[dict, dict]:
    """Read ``tanitad.det_nms/1`` -> ``({head: {"radius_m", "gate", "p_floor"}}, stamp)``.

    ⛔ REFUSES another schema, a head missing / unknown (one entry per head of ``detection_metrics.HEADS``, no silent
    fallback to an un-NMS'd head), a radius outside [0, 20] m, a gate outside (0, 1), a ``p_floor`` outside (0, 1). The
    stamp carries the file's sha256 (the trainer writes it to config.json)."""
    p = pathlib.Path(path)
    if not p.is_file():
        raise ValueError(f"detection NMS file {str(p)!r} does not exist")
    raw = p.read_bytes()
    d = json.loads(raw.decode("utf-8"))
    if d.get("schema") != NMS_SCHEMA:
        raise ValueError(f"{p.name}: schema {d.get('schema')!r} != {NMS_SCHEMA!r}")
    h = d.get("heads")
    if not isinstance(h, dict):
        raise ValueError(f"{p.name}: no `heads` object")
    if set(h) != set(_det.HEADS):
        raise ValueError(f"{p.name}: NMS entries for {sorted(h)} but the heads are {sorted(_det.HEADS)}")
    out = {}
    for hd in _det.HEADS:
        e = h[hd]
        try:
            r, g = float(e["radius_m"]), float(e["gate"])
            pf = float(e.get("p_floor", P_FLOOR))
        except (KeyError, TypeError, ValueError, AttributeError) as ex:
            raise ValueError(f"{p.name}: head {hd!r} needs numeric radius_m and gate ({type(ex).__name__}: {ex})") from None
        if not (math.isfinite(r) and 0.0 <= r <= 20.0):
            raise ValueError(f"{p.name}: radius_m {e['radius_m']!r} for {hd} must be in [0, 20] m")
        if not (math.isfinite(g) and 0.0 < g < 1.0):
            raise ValueError(f"{p.name}: gate {e['gate']!r} for {hd} must be a probability in (0, 1)")
        if not (math.isfinite(pf) and 0.0 < pf < 1.0):
            raise ValueError(f"{p.name}: p_floor {e.get('p_floor')!r} for {hd} must be in (0, 1)")
        out[hd] = {"radius_m": r, "gate": g, "p_floor": pf}
    return out, {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(), "schema": NMS_SCHEMA,
                 "heads": {k: dict(v) for k, v in out.items()}, "provenance": d.get("provenance")}


def nms_key_names(head: str) -> list:
    return [f"eval_{head}_nms_{k}" for k in NMS_KEYS]


def nms_census_keys(packs: list, head: str, cfg: dict | None = None) -> dict:
    """The readout of ``head`` AFTER its NMS, at its re-fitted gate, under NEW keys ``eval_{head}_nms_*``
    (:data:`NMS_KEYS`). ``cfg=None`` returns ``{}`` -- a default run emits no new key and computes nothing. The declared
    ``eval_{head}_*`` keys are untouched and the packs passed in are not modified. NaN = undefined, never 0."""
    if cfg is None:
        return {}
    if head not in _det.HEADS:
        raise ValueError(f"head {head!r} not in {_det.HEADS}")
    if head not in cfg:
        raise ValueError(f"no NMS entry for head {head!r} in {sorted(cfg)}")
    e = cfg[head]
    q = nms_packs(packs, e["radius_m"], e.get("p_floor", P_FLOOR))
    c = _census(q, e["gate"])
    dup = duplicate_stats(q, e["gate"])
    vals = {"radius_m": float(e["radius_m"]), "gate": float(e["gate"]), **c, "ap2m": ap2m(q),
            "boxes_per_object": dup["boxes_per_object"], "frac_objects_ge2": dup["frac_objects_ge2"]}
    return {f"eval_{head}_nms_{k}": vals[k] for k in NMS_KEYS}
