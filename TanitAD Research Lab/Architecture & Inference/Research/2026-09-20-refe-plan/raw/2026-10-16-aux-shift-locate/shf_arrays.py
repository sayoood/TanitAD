"""Per-frame arrays of one evaluation domain (held-out navtrain logs | navtest C3-1058): true mask/SDF on the head's 2 m grid, each arm's predicted mask/SDF,
and the per-frame attributes (map, speed, ...).  Built with the programme's OWN target definition (AUX.build_targets) and prediction loader (RO.load_head_pred)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shf_common as C

MAPNAME = {"us-nv-las-vegas-strip": "LV", "us-ma-boston": "BOS", "us-pa-pittsburgh-hazelwood": "PIT", "sg-one-north": "SG"}


def _attrs():
    return json.load(open(os.path.join(C.SCRATCH, "attributes.json"), encoding="utf-8"))


def domain(name: str, arms=("H1-s0", "H1-s1", "H1-s2")):
    """name in {'heldout', 'navtest'}. Rows are in each pred file's own order (identical across arms: asserted)."""
    A = _attrs()
    if name == "heldout":
        preds = {a: C.load_heldout_pred(a) for a in arms}
        src = C.open_train_labels()
        logmap = json.load(open(f"{C.AUXDIR}/log_map.json", encoding="utf-8"))["logs"]
    else:
        preds = {a: C.load_navtest_pred(a) for a in arms}
        src = C.open_navtest_labels()
        logmap = json.load(open(f"{C.AUXDIR}/label_store_navtest/inputs/log_map_navtest.json", encoding="utf-8"))["logs"]
    keys = preds[arms[0]]["keys"]
    for a in arms:
        assert preds[a]["keys"] == keys, f"{a}: key order differs"
    lab, valid = src.labels(keys)
    assert bool(valid.all()), "unlabelled frames"
    t = C.AUX.build_targets(lab, valid)
    out = dict(name=name, keys=keys, n=len(keys), logs=np.array([k[0] for k in keys]), tokens=np.array([k[1] for k in keys]),
               maps=np.array([MAPNAME[logmap[k[0]]["map"]] for k in keys]),
               TM=(t.mask_frac.numpy() >= 0.5), SDFT=t.sdf_m.numpy().astype(np.float32), arms=list(arms))
    out["PM"] = {a: preds[a]["logit"] > 0 for a in arms}
    out["PL"] = {a: preds[a]["logit"] for a in arms}
    out["PS"] = {a: preds[a]["sdf"] for a in arms}
    if name == "heldout":
        out["speed"] = np.array([A["train"][k[0] + "|" + k[1]][0] for k in keys], np.float32)
        out["scen"] = np.array([A["train"][k[0] + "|" + k[1]][1] for k in keys])
        out["stratum"] = np.array(["heldout"] * len(keys))
    else:
        c3 = C.c3()
        out["speed"] = np.array([A["navtest"][k[1]][0] for k in keys], np.float32)
        out["cmd"] = np.array([A["navtest"][k[1]][1] for k in keys])
        out["stratum"] = np.array([c3["stratum"][k[1]] for k in keys])
        out["scen"] = np.array(["?"] * len(keys))
    return out


BAND = None


def band(lo=0.0, hi=40.0):
    return C.RO.band_cells(lo, hi)


def frame_iou_parts(D, arm, lo=0.0, hi=40.0):
    """per-frame (inter, union, cells) of the arm's mask vs the true mask over the band, exactly RO.frame_stats' definition (known = all cells of a labelled frame)."""
    b = band(lo, hi)[None]
    tm, pm = D["TM"], D["PM"][arm]
    return (tm & pm & b).sum((1, 2)).astype(np.float64), ((tm | pm) & b).sum((1, 2)).astype(np.float64), np.full(D["n"], b.sum(), np.float64)


def pooled_iou(inter, union, w=None):
    w = np.ones_like(inter) if w is None else w
    return float((w * inter).sum() / (w * union).sum())
