#!/usr/bin/env python3
"""refcv8 WP-C: RE-FIT the three opt-in perception operating points on TRAIN for ANY checkpoint.

refcv7's F1 (per-class map thresholds), F4 (per-head presence gates) and F4b (centre-distance NMS radius + its gate) were
fitted ONCE, on ONE checkpoint (refcv7-r101-s0 step 50,400), and are bound to its class weights -- the loaders refuse them
for another model. refcv8 needs its own. This tool turns the TRAIN-DIAG outputs of the diagnostics harness
(``2026-10-04-refcv7-map-box-diagnostics/code/run_diag.py --split train_diag``: ``*.packs.pkl`` and ``*.acc.pt``) of a NEW
checkpoint into the three JSON files the stack already loads:

    --box-train-packs  train.packs.pkl   -> --out-gates  (tanitad.det_presence_gates/1)   F4
                                         -> --out-nms    (tanitad.det_nms/1)             F4b   (needs --nms)
    --map-train-acc    train.acc.pt      -> --out-map    (tanitad.map_hires_class_thresholds/1)   F1

⛔ TRAIN ONLY. No EVAL pack or histogram is accepted here: the fit objective is the pooled TRAIN IoU (map, one-vs-rest
threshold on ``logit(p_hat_c)`` over the 4,000-bin histogram, ties keep the first / lowest edge) and the TRAIN P = R point
(``detection_metrics.pr_equal_gate``); the NMS radius is the TRAIN-AP@2 m argmax, the gate is re-fitted AFTER it.

Every file it writes is re-READ with the stack's own loader before the tool reports success (the artifact, not the exit
code, is the evidence), and carries the checkpoint, the fit set, the source files' md5 and this tool's sha256.

ACCEPTANCE (``tests/test_refcv8_refit_perception_thresholds.py``, needs the banked evidence): fed the refcv7 TRAIN-DIAG
outputs of step 50,400 it reproduces the SHIPPED files -- the 8 map thresholds and their TRAIN IoUs bit-identically, the
two gates to full precision, and the route package's NMS radii (2.5 / 3.0 m) and gates (0.2145 / 0.1809 at 4 dp).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HIST_LO, HIST_HI, HIST_NB = -20.0, 20.0, 4000        # the diagnostics' score histogram (diag_metrics.HIST_*)
CLASS_KEYS = ("nocls", "drivable", "lane", "crosswalk", "arrow", "edge", "hatched", "sidewalk")


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _md5(path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------------------- #
# F1 -- the map thresholds from a TRAIN score histogram                                                      #
# --------------------------------------------------------------------------------------------------------- #
def hist_edges() -> np.ndarray:
    """Lower bin edges in logit space (len ``HIST_NB``) -- ``diag_metrics.hist_edges``."""
    return HIST_LO + (HIST_HI - HIST_LO) / HIST_NB * np.arange(HIST_NB)


def best_threshold(neg, pos) -> tuple[int, float]:
    """``(lower-edge index, IoU)`` of the mask ``{bin >= i}`` that maximises IoU against the positives; first index on
    ties. ``neg`` / ``pos``: counts of ground-truth-negative / -positive cells per bin (``diag_metrics.best_threshold``)."""
    neg = np.asarray(neg, np.float64)
    pos = np.asarray(pos, np.float64)
    inter = np.cumsum(pos[::-1])[::-1]
    pred = inter + np.cumsum(neg[::-1])[::-1]
    g = float(pos.sum())
    u = pred + g - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        j = np.where(u > 0, inter / u, 0.0)
    i = int(np.argmax(j))
    return i, float(j[i])


def refit_map_thresholds(hist_phat, class_weight) -> dict:
    """``hist_phat`` [8, bands, 2, NB] or [8, 2, NB] (``[..., 0, :]`` GT-negative, ``[..., 1, :]`` GT-positive counts of
    ``logit(p_hat_c)``). Returns the fields of ``tanitad.map_hires_class_thresholds/1``."""
    h = np.asarray(hist_phat)
    if h.ndim == 4:
        h = h.sum(axis=1)
    if h.shape != (len(CLASS_KEYS), 2, HIST_NB):
        raise ValueError(f"histogram shape {h.shape}, want [8, (bands,) 2, {HIST_NB}]")
    w = [float(v) for v in np.asarray(class_weight, np.float32).reshape(-1)]
    if len(w) != len(CLASS_KEYS) or not all(math.isfinite(v) and v > 0 for v in w):
        raise ValueError(f"class_weight {w}: want 8 finite positive values")
    edges = hist_edges()
    taus, ious = [], []
    for c in range(len(CLASS_KEYS)):
        if float(h[c, 1].sum()) <= 0:
            raise ValueError(f"class {CLASS_KEYS[c]!r} has no ground-truth positive cell in the TRAIN histogram: "
                             f"its threshold is not fittable -- refusing rather than shipping an arbitrary one")
        i, iou = best_threshold(h[c, 0], h[c, 1])
        taus.append(float(edges[i]))
        ious.append(iou)
    return {"tau_phat_logit": taus, "tau_phat_prob": [float(1.0 / (1.0 + math.exp(-t))) for t in taus],
            "class_weight_values": w, "train_iou_at_tau_phat": ious}


# --------------------------------------------------------------------------------------------------------- #
# F4 / F4b -- gates and NMS from TRAIN packs                                                                 #
# --------------------------------------------------------------------------------------------------------- #
def refit_gates(packs_by_head: dict) -> dict:
    """The TRAIN P = R gate per head (``detection_metrics.pr_equal_gate``)."""
    from tanitad.eval import detection_metrics as det
    gates, pr, npos = {}, {}, {}
    for hd in det.HEADS:
        if hd not in packs_by_head or not len(packs_by_head[hd]):
            raise ValueError(f"no TRAIN packs for head {hd!r}")
        g = det.pr_equal_gate(packs_by_head[hd])
        if not (g["gate"] == g["gate"] and 0.0 < g["gate"] < 1.0):
            raise ValueError(f"head {hd!r}: P = R gate {g['gate']!r} is not a probability -- no positive was ranked")
        gates[hd] = float(g["gate"])
        pr[hd] = [float(g["precision"]), float(g["recall"])]
        npos[hd] = int(g["n_pos"])
    return {"gates": gates, "train_precision_recall_at_gate": pr, "train_n_pos": npos}


def refit_nms(packs_by_head: dict, radii=None) -> dict:
    from tanitad.eval import detection_metrics as det
    from tanitad.eval import detection_nms as nms
    heads = {}
    for hd in det.HEADS:
        if hd not in packs_by_head or not len(packs_by_head[hd]):
            raise ValueError(f"no TRAIN packs for head {hd!r}")
        f = nms.fit_nms(packs_by_head[hd], radii=nms.R_GRID if radii is None else radii)
        heads[hd] = f
    return heads


# --------------------------------------------------------------------------------------------------------- #
# file writers (the shipped schemas)                                                                         #
# --------------------------------------------------------------------------------------------------------- #
def _prov(a, extra: dict) -> dict:
    p = {"run": a.run, "checkpoint_step": a.checkpoint_step, "checkpoint_md5": a.checkpoint_md5,
         "fit_split": "TRAIN", "fit_set": a.fit_set, "tool": "stack/scripts/refit_perception_thresholds.py",
         "tool_sha256": _sha256(__file__)}
    p.update(extra)
    return p


def gates_file(fit: dict, a, src: Path, n_windows: int) -> dict:
    return {"schema": "tanitad.det_presence_gates/1",
            "definition": "gate g on sigmoid(presence_logit) at the P = R point of the pooled greedy 2 m rows "
                          "(detection_metrics.pr_equal_gate) on TRAIN; a slot is a detection iff sigmoid(presence_logit) >= g",
            "gates": fit["gates"],
            "provenance": _prov(a, {"n_windows": n_windows, "source_file": src.name, "source_md5": _md5(src),
                                    "train_precision_recall_at_gate": fit["train_precision_recall_at_gate"],
                                    "train_n_pos": fit["train_n_pos"],
                                    "caveat": "fitted on ONE checkpoint of ONE run; the planner consumes the raw presence "
                                              "probability as a soft scale and is NOT re-gated by these values"})}


def nms_file(fit: dict, a, src: Path, n_windows: int) -> dict:
    return {"schema": "tanitad.det_nms/1",
            "definition": "score-ordered greedy centre-distance NMS (all classes together) on slots with sigmoid(logit) >= "
                          "p_floor; radius_m = TRAIN-AP@2m argmax (tie -> smaller); gate = TRAIN P = R re-fitted AFTER the NMS "
                          "(detection_metrics.pr_equal_gate). Radius and gate are ONE operating point.",
            "heads": {hd: {"radius_m": f["radius_m"], "gate": f["gate"], "p_floor": f["p_floor"]} for hd, f in fit.items()},
            "provenance": _prov(a, {"n_windows": n_windows, "source_file": src.name, "source_md5": _md5(src),
                                    "train_grid": {hd: f["grid"] for hd, f in fit.items()},
                                    "caveat": "fitted on ONE checkpoint of ONE run; NEVER fed to the planner (raw presence "
                                              "is its input)"})}


def map_file(fit: dict, a, src: Path, n_episodes: int | None) -> dict:
    return {"schema": "tanitad.map_hires_class_thresholds/1", "classes": list(CLASS_KEYS),
            "score": "p_hat = softmax(z - ln w) over the 8 classes (w = the run's frozen class weights); rule: cell is class c "
                     "iff logit(p_hat_c) >= tau_phat_logit[c]  (one-vs-rest, MULTI-LABEL)",
            "tau_phat_logit": fit["tau_phat_logit"], "tau_phat_prob": fit["tau_phat_prob"],
            "class_weight_values": fit["class_weight_values"],
            "provenance": _prov(a, {"n_episodes": n_episodes, "source_file": src.name, "source_md5": _md5(src),
                                    "fit_objective": "per class, the one-vs-rest threshold on logit(p_hat_c) that maximises the "
                                                     "pooled TRAIN IoU (histogram sweep, 4000 bins over [-20, 20])",
                                    "train_iou_at_tau_phat": fit["train_iou_at_tau_phat"],
                                    "caveat": "fitted on ONE checkpoint of ONE run: valid for this model's final weights and "
                                              "the class weights above"})}


def _write(path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="run name for the provenance block")
    ap.add_argument("--checkpoint-step", type=int, required=True)
    ap.add_argument("--checkpoint-md5", required=True)
    ap.add_argument("--fit-set", required=True, help="one line: which TRAIN windows (e.g. TRAIN-DIAG 139 ep x 8 windows)")
    ap.add_argument("--box-train-packs", default=None)
    ap.add_argument("--out-gates", default=None)
    ap.add_argument("--nms", action="store_true", help="also fit the NMS radius + its gate (needs --out-nms)")
    ap.add_argument("--out-nms", default=None)
    ap.add_argument("--map-train-acc", default=None)
    ap.add_argument("--out-map", default=None)
    a = ap.parse_args(argv)
    if not (a.box_train_packs or a.map_train_acc):
        ap.error("nothing to fit: pass --box-train-packs and/or --map-train-acc")
    if a.box_train_packs and not (a.out_gates or (a.nms and a.out_nms)):
        ap.error("--box-train-packs needs --out-gates and/or (--nms and --out-nms)")
    if a.map_train_acc and not a.out_map:
        ap.error("--map-train-acc needs --out-map")
    if bool(a.nms) != bool(a.out_nms):
        ap.error("--nms and --out-nms go together")
    if a.box_train_packs:
        src = Path(a.box_train_packs)
        with open(src, "rb") as fh:
            packs = pickle.load(fh)
        n_w = len(packs["box3d"])
        from tanitad.eval import detection_metrics as det
        from tanitad.eval import detection_nms as nms_mod
        if a.out_gates:
            fit = refit_gates(packs)
            _write(a.out_gates, gates_file(fit, a, src, n_w))
            g, _ = det.load_head_gates(a.out_gates)                       # the artifact, read back by the stack's loader
            print("[refit] gates", {k: round(v, 6) for k, v in g.items()}, flush=True)
        if a.nms:
            fit = refit_nms(packs)
            _write(a.out_nms, nms_file(fit, a, src, n_w))
            c, _ = nms_mod.load_head_nms(a.out_nms)
            print("[refit] nms", {k: (v["radius_m"], round(v["gate"], 6)) for k, v in c.items()}, flush=True)
    if a.map_train_acc:
        import torch
        src = Path(a.map_train_acc)
        acc = torch.load(src, map_location="cpu", weights_only=False)
        fit = refit_map_thresholds(acc["hist"]["phat"].numpy(), acc["class_weight"].numpy())
        n_ep = len(acc["episodes"]) if acc.get("episodes") is not None else None
        _write(a.out_map, map_file(fit, a, src, n_ep))
        from tanitad.models import map_head_hires as mhr
        th, _st = mhr.load_class_thresholds(a.out_map, class_weight=acc["class_weight"])   # read back by the stack's loader
        print("[refit] map tau_phat_logit", [round(float(v), 4) for v in th.tolist()], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
