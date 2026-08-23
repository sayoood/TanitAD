#!/usr/bin/env python3
"""B1 — THE TACTICAL LABEL, READ OVER THE BAND `TAC_BAND_S = (2.0, 6.0]` s.

⛔ WHY THIS EXISTS — C89 (`Project Steering/RETRACTION_LOG.md`). The 40-clip
review sheet and every tactical kappa quoted this session were read at **2.0 s**,
which is the **SEAM**, not the band. The binding spec is a one-line lookup
(`stack/tanitad/models/v6.py:136-140`):

    PLAN_STEPS = 60 · DT = 0.1 · HORIZON_S = 6.0
    OP_BAND_S  = (0.0, 2.0)   # operative
    TAC_BAND_S = (2.0, 6.0)   # TACTICAL

⭐ AND THE PROPOSED CORRECTION WAS ALSO WRONG — THIS IS THE SECOND DEFECT.
C89 says to restate the numbers at "6.0 s (band end)" using the banked sweep's
0.2331 / 0.4040. **Those are NOT band values.** `tac_a4_horizon_sweep.py:140-148`
anchors EVERY horizon at `t0`:

    k  = min(int(round(t0 + H * POSE_HZ)), poses.shape[0] - 1)
    dv = float(v[k] - v[t0])                      # <-- anchored at t0, always

so the row labelled `horizon_s: 6.0` measures the net change over **(0.0, 6.0]**
— the FULL horizon, operative band included — and the row labelled `2.0`
measures **(0.0, 2.0]**, which is precisely `OP_BAND_S`, the OPERATIVE band.
⇒ **Neither row in the banked sweep is the tactical band.** Reading 0.2331 as
"the band value" would repeat C89 one level down: a number that answers a
different question, adopted because it was already in the JSON.
⇒ This script computes the band quantity that did not previously exist:
**anchored at `t0+20` (2.0 s) and read across `t0+21 … t0+60`.**

──────────────────────────────────────────────────────────────────────────────
⭐ THE INTERVAL-vs-ENDPOINT DECISION, STATED BEFORE ANY KAPPA WAS COMPUTED
──────────────────────────────────────────────────────────────────────────────
"At 6.0 s" (one sample) and "over (2.0, 6.0]" (the interval the tactical layer
owns) are different quantities. Three principles decide it, and NONE of them is
"which scores better":

  P1 · BOTH ENDS INSIDE THE BAND. A tactical label may not be computed from
       data the OPERATIVE layer owns. This alone kills the `t0`-anchored family
       (the banked sweep's entire surface, at every horizon).
  P2 · THE STATISTIC READS THE SLICE, NOT TWO SAMPLES. The PI: *"tactical
       behavior is evolving until 6s horizon (this is the whole resulting
       trajectory)"*; `v6.py:...` §4b: *"bands are SLICES of one rollout"*.
       A two-point difference across a 4 s window is blind to everything
       between — it calls "brake hard, then recover" `maintain`.
  P3 · ROBUST TO A SINGLE BAD POSE SAMPLE. 40 samples at 10 Hz; a statistic
       that a single outlier can set is not a description of the interval.

⇒ **PRIMARY = `mean_band`** — the mean of (value(k) − value(t0+20)) over the 40
   in-band samples. It satisfies P1, P2 and P3.
⇒ **SENSITIVITIES, always reported beside it, never instead of it:**
     * `net_band` — v[t0+60] − v[t0+20]. Satisfies P1, fails P2 (two points).
     * `ext_band` — the largest-|·| signed deviation from the band start.
       Satisfies P1 and P2, fails P3 (one sample can set it).
⇒ **REFERENCE ROWS, carried so the shape of the C89 error stays legible:**
     * `seam_op`  — v[t0+20] − v[t0]  ≡ the sheet's "2.0 s" ≡ `OP_BAND_S`.
     * `full_h6`  — v[t0+60] − v[t0]  ≡ the sweep's "6.0 s" ≡ (0.0, 6.0].

⛔ THE THRESHOLD IS A SPEC LOOKUP, NOT AN ARGMAX (C89 rule 1). The headline uses
the PRODUCTION thresholds — `EGO_DV_MS = 1.0`, `EGO_DYAW_RAD = 0.15`
(`tac_a3_three_leg_agreement.py:120-121`, themselves the shipped ph1 values).
⚠️ The 2.0 s sheet used `DV 0.75 / DYAW 0.05`, described in its own source as
*"the thresholds that maximised kappa at this horizon (a4 sweep, best cell)"* —
i.e. the sheet argmax-selected its THRESHOLD as well as its HORIZON. C89 caught
only the horizon. The full threshold surface is emitted here so the headline can
be checked against it, and the argmax cell is reported LABELLED as the argmax.

Estimator: **episode-cluster bootstrap** over clips (`taniteval/ci.py`), paired
for band-vs-seam. ⛔ never `overlapping_holdout_se`. One clip = one cluster:
these are clip-level labels, one per clip.

Usage (CPU only, no GPU, no network):
  python tacrev_band_agreement.py --out <json>
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parents[1]
INC = _PKG.parent
_REPO = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(_REPO / "taniteval"))

from taniteval.ci import (episode_cluster_bootstrap,  # noqa: E402
                          paired_episode_cluster_bootstrap)

SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")

POSE_HZ = 10.0
T0_IDX = 80
#: the binding band — v6.py:139-140. NOT an argmax; a spec lookup.
OP_BAND_S = (0.0, 2.0)
TAC_BAND_S = (2.0, 6.0)
BAND_LO_IDX = T0_IDX + int(round(TAC_BAND_S[0] * POSE_HZ))   # 100  == 2.0 s
BAND_HI_IDX = T0_IDX + int(round(TAC_BAND_S[1] * POSE_HZ))   # 140  == 6.0 s

#: PRODUCTION thresholds (spec lookup — a3:120-121 / ph1_fuse)
PROD_DV_MS = 1.0
PROD_DYAW_RAD = 0.15
#: the 2.0 s sheet's argmax-selected thresholds, carried for the restatement
SEAM_SHEET_DV_MS = 0.75
SEAM_SHEET_DYAW_RAD = 0.05

DV_THRESH = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
DYAW_THRESH = (0.05, 0.10, 0.15, 0.25, 0.40)

ALP_LON3 = {"Maintain Speed": "maintain",
            "Gentle Deceleration": "decelerate",
            "Strong Deceleration": "decelerate", "Stop": "decelerate",
            "Gentle Acceleration": "accelerate",
            "Strong Acceleration": "accelerate", "Reverse": None}
ALP_LAT3 = {"Go Straight": "straight",
            "Steer Left": "left", "Sharp Steer Left": "left",
            "Slight Steer Left": "left", "Steer Right": "right",
            "Sharp Steer Right": "right", "Slight Steer Right": "right",
            "Reverse Left": None, "Reverse Right": None}

LON3 = ("decelerate", "maintain", "accelerate")
LAT3 = ("left", "straight", "right")


def wrap(d: float) -> float:
    """Signed yaw difference wrapped to (-pi, pi]."""
    return (d + math.pi) % (2 * math.pi) - math.pi


def band_stats(series: np.ndarray, *, is_yaw: bool) -> dict:
    """The five readouts, for one clip, for one channel (speed or yaw).

    ⚠️ Every in-band deviation is taken against the BAND START (`t0+20`), which
    is what makes these tactical quantities rather than horizon quantities.
    """
    a = float(series[BAND_LO_IDX])                     # band start == 2.0 s
    inb = series[BAND_LO_IDX + 1:BAND_HI_IDX + 1]      # (2.0, 6.0] -> 40 samples
    if is_yaw:
        dev = np.array([wrap(float(x) - a) for x in inb])
        net = wrap(float(series[BAND_HI_IDX]) - a)
        seam = wrap(a - float(series[T0_IDX]))
        full = wrap(float(series[BAND_HI_IDX]) - float(series[T0_IDX]))
    else:
        dev = inb.astype(float) - a
        net = float(series[BAND_HI_IDX]) - a
        seam = a - float(series[T0_IDX])
        full = float(series[BAND_HI_IDX]) - float(series[T0_IDX])
    ext = float(dev[int(np.argmax(np.abs(dev)))]) if dev.size else 0.0
    return {"mean_band": float(dev.mean()) if dev.size else 0.0,
            "net_band": net, "ext_band": ext,
            "seam_op": seam, "full_h6": full, "n_in_band": int(dev.size)}


def band3(x: float, th: float, axis: str) -> str:
    if axis == "lon":
        return "accelerate" if x > th else "decelerate" if x < -th else "maintain"
    return "left" if x > th else "right" if x < -th else "straight"


def kappa_from_pairs(pairs) -> dict:
    """Cohen's kappa + agreement. Missing excluded and counted, never imputed."""
    both = [(a, b) for a, b in pairs if a is not None and b is not None]
    n = len(both)
    if not n:
        return {"n": 0, "agreement": None, "kappa": None, "chance": None}
    agree = sum(1 for a, b in both if a == b)
    labs = sorted({a for a, _ in both} | {b for _, b in both})
    po = agree / n
    pe = sum((sum(1 for a, _ in both if a == c) / n) *
             (sum(1 for _, b in both if b == c) / n) for c in labs)
    return {"n": n, "agreement": round(po, 4),
            "kappa": round((po - pe) / (1 - pe), 4) if abs(1 - pe) > 1e-9 else None,
            "chance": round(pe, 4),
            "n_dropped_missing": len(pairs) - n}


# --------------------------------------------------------------------------- #
# kappa under the episode-cluster bootstrap.                                   #
# The estimator resamples 1-D float values; a (a_label, b_label) PAIR is        #
# encoded as one float so kappa can be recomputed inside every draw. This is an #
# ENCODING, not an approximation — decode is exact.                            #
# --------------------------------------------------------------------------- #
def encode_pairs(pairs, cls) -> tuple[np.ndarray, list]:
    idx = {c: i for i, c in enumerate(cls)}
    vals, keep = [], []
    for k, (a, b) in enumerate(pairs):
        if a is None or b is None:
            continue
        vals.append(float(idx[a] * len(cls) + idx[b]))
        keep.append(k)
    return np.asarray(vals, dtype=np.float64), keep


def _kappa_reducer(ncls: int):
    def _r(v: np.ndarray) -> float:
        if v.size == 0:
            return float("nan")
        a = (v // ncls).astype(int)
        b = (v % ncls).astype(int)
        n = v.size
        po = float((a == b).mean())
        pe = 0.0
        for c in range(ncls):
            pe += float((a == c).mean()) * float((b == c).mean())
        return (po - pe) / (1 - pe) if abs(1 - pe) > 1e-9 else float("nan")
    return _r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_PKG / "raw" / "b1_band_agreement.json"))
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    tl = INC / "2026-08-16-tactical-labels" / "raw"
    alp = {}
    with open(tl / "a1_alpamayo_taxonomy_per_clip.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                alp[r["clip_id"]] = r

    per: dict[str, dict] = {}
    n_short = 0
    for p in sorted(glob.glob(os.path.join(SP, "s2_ego", "aug120", "*.npz"))):
        cid = os.path.splitext(os.path.basename(p))[0]
        if cid not in alp:
            continue
        poses = np.load(p)["poses"].astype(float)
        # ⛔ NO CLAMPING. A clip that cannot reach the band END is DROPPED, not
        # silently shortened — a truncated band is a different quantity.
        if poses.shape[0] <= BAND_HI_IDX:
            n_short += 1
            continue
        per[cid] = {"lon": band_stats(poses[:, 3], is_yaw=False),
                    "lat": band_stats(poses[:, 2], is_yaw=True),
                    "v_t0": float(poses[T0_IDX, 3]),
                    "v_band_lo": float(poses[BAND_LO_IDX, 3]),
                    "v_band_hi": float(poses[BAND_HI_IDX, 3])}

    res: dict = {
        "_evidence_class": "MEASURED (ours; 201 local ego pose tracks, 10 Hz, "
                           "this directory)",
        "_horizon": {
            "band_s": list(TAC_BAND_S), "op_band_s": list(OP_BAND_S),
            "_spec_source": "stack/tanitad/models/v6.py:136-140 (TAC_BAND_S)",
            "band_lo_idx": BAND_LO_IDX, "band_hi_idx": BAND_HI_IDX,
            "t0_idx": T0_IDX, "pose_hz": POSE_HZ,
            "_anchor": ("every in-band deviation is taken against the BAND "
                        "START t0+20 (2.0 s), NOT t0. This is the difference "
                        "between a tactical quantity and a horizon quantity."),
        },
        "_primary_definition": {
            "name": "mean_band",
            "formula": "mean over k in (t0+20, t0+60] of (x[k] - x[t0+20])",
            "chosen_by": "P1 both ends in band; P2 reads the slice not two "
                         "samples; P3 robust to one bad pose sample. Stated "
                         "before any kappa was computed; NOT chosen by score.",
        },
        "n_clips": len(per),
        "n_clips_dropped_cannot_reach_band_end": n_short,
    }

    # ---------------- yaw sign convention: READ, never assumed --------------
    cal: dict[str, list] = collections.defaultdict(list)
    for cid, r in per.items():
        cal[alp[cid].get("lateral") or "?"].append(r["lat"]["mean_band"])
    res["_yaw_sign_calibration_on_the_BAND"] = {
        "_why": ("the sign convention was previously calibrated on the 2.0 s "
                 "SEAM quantity. It is re-read here on the BAND quantity, "
                 "because a convention verified for one readout is not "
                 "evidence for another."),
        "mean_band_dyaw_by_alpamayo_lateral": {
            k: {"n": len(v), "mean": round(float(np.mean(v)), 4)}
            for k, v in sorted(cal.items())},
    }

    # ---------------- the surface: definition x threshold -------------------
    defs = ("mean_band", "net_band", "ext_band", "seam_op", "full_h6")
    surface = {"lon": [], "lat": []}
    for d in defs:
        for th in DV_THRESH:
            pairs = [(ALP_LON3.get(alp[c]["longitudinal"]),
                      band3(per[c]["lon"][d], th, "lon")) for c in sorted(per)]
            surface["lon"].append({"definition": d, "dv_thresh_ms": th,
                                   **kappa_from_pairs(pairs)})
        for th in DYAW_THRESH:
            pairs = [(ALP_LAT3.get(alp[c]["lateral"]),
                      band3(per[c]["lat"][d], th, "lat")) for c in sorted(per)]
            surface["lat"].append({"definition": d, "dyaw_thresh_rad": th,
                                   **kappa_from_pairs(pairs)})
    res["surface"] = surface

    # ---------------- HEADLINE: primary definition, PRODUCTION threshold ----
    clips = sorted(per)
    eid = list(clips)                     # one clip = one cluster

    def boot(axis: str, d: str, th: float) -> dict:
        if axis == "lon":
            pairs = [(ALP_LON3.get(alp[c]["longitudinal"]),
                      band3(per[c]["lon"][d], th, "lon")) for c in clips]
            cls = LON3
        else:
            pairs = [(ALP_LAT3.get(alp[c]["lateral"]),
                      band3(per[c]["lat"][d], th, "lat")) for c in clips]
            cls = LAT3
        v, keep = encode_pairs(pairs, cls)
        out = episode_cluster_bootstrap(v, [eid[k] for k in keep],
                                        reduce=_kappa_reducer(len(cls)),
                                        n_boot=a.n_boot)
        out["reducer"] = "cohens_kappa"
        out.update(kappa_from_pairs(pairs))
        return out

    res["headline_BAND_primary"] = {
        "_definition": "mean_band", "_band_s": list(TAC_BAND_S),
        "_thresholds": {"dv_thresh_ms": PROD_DV_MS,
                        "dyaw_thresh_rad": PROD_DYAW_RAD,
                        "_source": "PRODUCTION values, a3:120-121 — spec "
                                   "lookup, not an argmax."},
        "LON": boot("lon", "mean_band", PROD_DV_MS),
        "LAT": boot("lat", "mean_band", PROD_DYAW_RAD),
    }
    res["headline_BAND_sensitivities"] = {
        d: {"LON": boot("lon", d, PROD_DV_MS),
            "LAT": boot("lat", d, PROD_DYAW_RAD)}
        for d in ("net_band", "ext_band")}
    res["reference_rows_NOT_the_band"] = {
        d: {"_what_it_is": ("the sheet's 2.0 s reading == OP_BAND_S, the "
                            "OPERATIVE band" if d == "seam_op" else
                            "the sweep's 6.0 s row == the FULL horizon "
                            "(0.0, 6.0], operative band included"),
            "LON": boot("lon", d, PROD_DV_MS),
            "LAT": boot("lat", d, PROD_DYAW_RAD)}
        for d in ("seam_op", "full_h6")}

    # ---- the exact restatement of the numbers quoted all session ----------
    res["restatement_of_the_quoted_numbers"] = {
        "_quoted_all_session": {"LON_kappa": 0.3655, "LAT_kappa": 0.4694,
                                "_what_they_actually_are": (
                                    "kappa of (0.0, 2.0] == OP_BAND_S, at the "
                                    "argmax threshold cell (dv 0.75 / dyaw "
                                    "0.05). SEAM/OPERATIVE values.")},
        "_c89_proposed_correction": {
            "LON_kappa": 0.2331, "LAT_kappa": 0.4040,
            "_what_they_actually_are": (
                "kappa of (0.0, 6.0] — the FULL horizon, still t0-anchored, "
                "still at the seam's argmax threshold. NOT the tactical band."),
        },
        "_at_the_seam_sheet_thresholds_for_a_like_for_like_read": {
            "seam_op": {"LON": boot("lon", "seam_op", SEAM_SHEET_DV_MS),
                        "LAT": boot("lat", "seam_op", SEAM_SHEET_DYAW_RAD)},
            "full_h6": {"LON": boot("lon", "full_h6", SEAM_SHEET_DV_MS),
                        "LAT": boot("lat", "full_h6", SEAM_SHEET_DYAW_RAD)},
            "mean_band": {"LON": boot("lon", "mean_band", SEAM_SHEET_DV_MS),
                          "LAT": boot("lat", "mean_band", SEAM_SHEET_DYAW_RAD)},
        },
    }

    # ---- PAIRED band-vs-seam, same clips, same draw -----------------------
    paired = {}
    for axis, cls, th in (("lon", LON3, PROD_DV_MS), ("lat", LAT3, PROD_DYAW_RAD)):
        key = "longitudinal" if axis == "lon" else "lateral"
        amap = ALP_LON3 if axis == "lon" else ALP_LAT3
        pa = [(amap.get(alp[c][key]), band3(per[c][axis]["mean_band"], th, axis))
              for c in clips]
        pb = [(amap.get(alp[c][key]), band3(per[c][axis]["seam_op"], th, axis))
              for c in clips]
        keep = [k for k, (x, y) in enumerate(zip(pa, pb))
                if x[0] is not None and y[0] is not None]
        va, _ = encode_pairs([pa[k] for k in keep], cls)
        vb, _ = encode_pairs([pb[k] for k in keep], cls)
        paired[axis.upper()] = paired_episode_cluster_bootstrap(
            va, vb, [eid[k] for k in keep], reduce=_kappa_reducer(len(cls)),
            n_boot=a.n_boot)
        paired[axis.upper()]["reducer"] = "cohens_kappa"
    res["paired_BAND_minus_SEAM"] = {
        "_what": ("kappa(mean_band over (2.0,6.0]) MINUS kappa(seam_op over "
                  "(0.0,2.0]) at the PRODUCTION thresholds, same clips, same "
                  "resampled draw. Negative = the correct horizon agrees WORSE."),
        "_estimator": "paired_episode_cluster_bootstrap (taniteval/ci.py). "
                      "⛔ never overlapping_holdout_se.",
        **paired,
    }

    # ---- the argmax cell, LABELLED as the argmax (C89 rule 2) -------------
    for axis in ("lon", "lat"):
        rows = [r for r in surface[axis]
                if r["definition"] == "mean_band" and r["kappa"] is not None]
        best = max(rows, key=lambda r: r["kappa"])
        res.setdefault("_argmax_cells_reported_second_and_labelled", {})[axis] = {
            "_warning": ("this is the ARGMAX over thresholds for the primary "
                         "definition. It is reported SECOND and is NOT the "
                         "headline — C89 rule 2."),
            **best}

    per_clip = a.out.replace(".json", "_per_clip.jsonl")
    with open(per_clip, "w", encoding="utf-8") as fh:
        for c in clips:
            fh.write(json.dumps({"clip_id": c,
                                 "alp_lat": alp[c].get("lateral"),
                                 "alp_lon": alp[c].get("longitudinal"),
                                 **{f"lon_{k}": v for k, v in per[c]["lon"].items()},
                                 **{f"lat_{k}": v for k, v in per[c]["lat"].items()}
                                 }) + "\n")
    res["_per_clip_path"] = per_clip

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1, ensure_ascii=False)
    slim = {k: v for k, v in res.items() if k != "surface"}
    print(json.dumps(slim, indent=1, ensure_ascii=False)[:6000])
    print(f"\n[out] {a.out}\n[out] {per_clip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
