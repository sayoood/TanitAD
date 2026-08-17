#!/usr/bin/env python3
"""Build the TACTICAL visual review sheet over the BAND `TAC_BAND_S = (2.0, 6.0]`.

⛔ WHY THIS EXISTS — C89 (`Project Steering/RETRACTION_LOG.md`). The sheet this
one replaces (`tacrev_visual_review.py`, `TACTICAL_VISUAL_REVIEW.html`) reads
its tactical labels at **2.0 s**, which is the **SEAM** between the operative
and tactical bands — chosen because it was the argmax of kappa, then justified
post-hoc as "the v6 tactical band". The binding spec
(`stack/tanitad/models/v6.py:136-140`) says otherwise:

    OP_BAND_S  = (0.0, 2.0)   # operative
    TAC_BAND_S = (2.0, 6.0)   # TACTICAL

⛔ THIS IS A HORIZON CHANGE, NOT A REDESIGN. Everything the PI has already seen
is preserved deliberately — real rendered frames (never a text-only card), LAT
and LON with SEPARATE verdict controls, three legs side by side with the
VLM/ego pair drawn as non-independent, Alpamayo's `cot` verbatim, labelled
strata, unreviewed rows exporting as `null`. A second design would make his two
review passes non-comparable, which is the one thing worse than a wrong horizon.

⭐ WHAT CHANGED, AND ONLY THIS:
  1. the read window: `t0+20 … t0+60` instead of `t0 … t0+20`;
  2. the ego leg's statistic: `mean_band` (below) instead of a 2-point Δ;
  3. the frames: they now SPAN the band (+2.0 s → +6.0 s) instead of ending
     where the band begins;
  4. the thresholds: the PRODUCTION values, not the argmax cell (see below);
  5. the localStorage key — ⛔ **`tacrev_band_v1_`, deliberately different**.
     Sharing the old key would silently pre-fill this sheet with the verdicts
     the PI gave at the WRONG horizon, and they would export as if he had
     judged these frames. Two horizons, two verdict sets.

──────────────────────────────────────────────────────────────────────────────
⭐ THE INTERVAL-vs-ENDPOINT DECISION (full reasoning: `tacrev_band_agreement.py`)
──────────────────────────────────────────────────────────────────────────────
"At 6.0 s" (one sample) and "over (2.0, 6.0]" (the interval the tactical layer
owns) are different quantities. The PI's words — *"tactical behavior is evolving
until 6s horizon (this is the whole resulting trajectory)"* — and `v6.py` §4b's
*"bands are SLICES of one rollout"* both read as the INTERVAL.

⇒ ego leg = **`mean_band`**: the mean over the 40 in-band samples of
   `x[k] − x[t0+20]`. Anchored at the BAND START, so no operative-band data
   enters a tactical label; reads every sample, so a manoeuvre that reverses
   inside the band is not erased; and robust to a single bad pose sample.
   `net_band` (2-point across the band) is printed beside it on every card so
   the PI can see when the two disagree.

⛔ THE THRESHOLD IS A SPEC LOOKUP, NOT AN ARGMAX. `DV 1.0 m/s` / `DYAW 0.15 rad`
are the PRODUCTION values (`tac_a3_three_leg_agreement.py:120-121`). ⚠️ The 2.0 s
sheet used `0.75 / 0.05`, described in its own source as *"the thresholds that
maximised κ at this horizon"* — it argmax-selected its threshold as well as its
horizon, and C89 caught only the horizon.

Usage (CPU only, no GPU, no network):
  python tacrev_visual_review_band.py --out <html> --meta <json> [--max-clips N]
"""
from __future__ import annotations

import argparse
import base64
import collections
import glob
import html
import io
import json
import math
import os
import subprocess
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
_STACK = Path(__file__).resolve().parents[6] / "stack"
sys.path.insert(0, str(_STACK))
sys.path.insert(0, str(_STACK / "scripts"))

import ph1_fuse  # noqa: E402

SP = (r"C:\Users\Admin\AppData\Local\Temp\claude"
      r"\G--Meine-Ablage-SayBouBase-raw-Projects-TanitAD"
      r"\8fc25020-a1d5-4e1b-a9e2-aeccf845c5a2\scratchpad")
FFMPEG = (r"C:\Users\Admin\AppData\Local\Microsoft\WinGet\Packages"
          r"\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe"
          r"\ffmpeg-8.1.1-full_build\bin\ffmpeg.exe")
HUB = r"C:\Users\Admin\.cache\huggingface\hub\datasets--Sayood--tanitad-ph0-aug120"
INC = _PKG.parent

T0_IDX = 80
FPS = 10.0
#: ⭐ THE BAND. A SPEC LOOKUP — v6.py:139-140. Never an argmax.
OP_BAND_S = (0.0, 2.0)
TAC_BAND_S = (2.0, 6.0)
BAND_LO_IDX = T0_IDX + int(round(TAC_BAND_S[0] * FPS))    # 100 == +2.0 s
BAND_HI_IDX = T0_IDX + int(round(TAC_BAND_S[1] * FPS))    # 140 == +6.0 s
#: PRODUCTION thresholds (a3:120-121). NOT the argmax cell.
DV_THRESH = 1.0           # m/s
DYAW_THRESH = 0.15        # rad
#: frames: 1 s of run-up, t0, then the WHOLE tactical band at 1 s spacing
FRAME_IDXS = [T0_IDX - 10, T0_IDX, BAND_LO_IDX, BAND_LO_IDX + 10,
              BAND_LO_IDX + 20, BAND_LO_IDX + 30, BAND_HI_IDX]

# --------------------------------------------------------------------------- #
# 3-BAND PROJECTIONS — FOR STRATIFICATION ONLY. (unchanged from the 2.0 s sheet)#
# ⚠️ They DISCARD severity, so any agreement figure computed on them is an      #
# UPPER BOUND. They decide which clips are shown, never what the card claims.   #
# --------------------------------------------------------------------------- #
ALP_LAT3 = {"go straight": "straight", "steer left": "left",
            "steer right": "right", "sharp steer left": "left",
            "sharp steer right": "right", "reverse left": "left",
            "reverse right": "right"}
ALP_LON3 = {"gentle deceleration": "decelerate", "strong deceleration":
            "decelerate", "stop": "decelerate", "maintain speed": "maintain",
            "gentle acceleration": "accelerate", "strong acceleration":
            "accelerate", "reverse": None}
V6_LAT3 = {"LANE_KEEP": "straight", "LANE_CHANGE_L": "left",
           "LANE_CHANGE_R": "right", "NUDGE_L": "left", "NUDGE_R": "right",
           "ABORT_LC": None}
V6_LON3 = {"BRAKE_TO": "decelerate", "CREEP": "decelerate",
           "HOLD": "decelerate", "YIELD_MERGE": "decelerate",
           "CRUISE": "maintain", "FOLLOW": "maintain"}


def b64img(data: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def wrap(d: float) -> float:
    return (d + math.pi) % (2 * math.pi) - math.pi


def find_mp4s() -> dict:
    """Every locally-present clip video. ⚠️ ABSENCE FOUND AT ONE LOCATION IS
    NOT ABSENCE — three roots are probed, and the HF cache is one of them."""
    out: dict[str, str] = {}
    for pat in (os.path.join(SP, "s2rev_assets", "mp4", "*.mp4"),
                os.path.join(SP, "sam3fix_assets", "videos", "*.mp4"),
                os.path.join(HUB, "snapshots", "*", "bridged_w120train_2400",
                             "videos", "*.mp4")):
        for p in glob.glob(pat):
            out.setdefault(os.path.basename(p)[:-4], p)
    return out


def frames_from_mp4(mp4: str, idxs: list[int]) -> list[bytes] | None:
    tmp = os.path.join(SP, "tacrev_band_tmp")
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))
    expr = "+".join(f"eq(n\\,{i})" for i in idxs)
    cmd = [FFMPEG, "-y", "-v", "error", "-i", mp4, "-vf", f"select='{expr}'",
           "-fps_mode", "vfr", "-q:v", "5", os.path.join(tmp, "f_%02d.jpg")]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except Exception:                                            # noqa: BLE001
        return None
    got = sorted(glob.glob(os.path.join(tmp, "f_*.jpg")))
    if len(got) != len(idxs):
        return None
    return [open(p, "rb").read() for p in got]


def bev_png(poses) -> bytes:
    """Metric BEV in the ego frame at t0, with the OPERATIVE and TACTICAL
    slices drawn DIFFERENTLY — the band this sheet judges is the bold one."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    x0, y0, yaw0 = poses[T0_IDX, 0], poses[T0_IDX, 1], poses[T0_IDX, 2]
    dx, dy = poses[:, 0] - x0, poses[:, 1] - y0
    fwd = dx * np.cos(yaw0) + dy * np.sin(yaw0)
    lft = -dx * np.sin(yaw0) + dy * np.cos(yaw0)
    lo, hi = max(0, T0_IDX - 40), min(len(poses) - 1, BAND_HI_IDX + 20)
    b_lo = min(BAND_LO_IDX, len(poses) - 1)
    b_hi = min(BAND_HI_IDX, len(poses) - 1)
    fig, ax = plt.subplots(figsize=(2.6, 3.2), dpi=105)
    ax.plot(-lft[lo:hi], fwd[lo:hi], color="#c9ccd1", lw=1.2, zorder=1)
    # operative slice 0 -> 2 s: thin, muted — NOT what this sheet judges
    ax.plot(-lft[T0_IDX:b_lo + 1], fwd[T0_IDX:b_lo + 1], color="#8c959f",
            lw=1.8, zorder=2, solid_capstyle="round")
    # ⭐ TACTICAL band 2 -> 6 s: bold — the quantity under review
    ax.plot(-lft[b_lo:b_hi + 1], fwd[b_lo:b_hi + 1], color="#1f6feb", lw=3.0,
            zorder=3, solid_capstyle="round")
    ax.scatter([0], [0], s=52, c="#111", marker="^", zorder=4)
    ax.scatter([-lft[b_lo]], [fwd[b_lo]], s=40, c="#8c959f", zorder=4)
    ax.scatter([-lft[b_hi]], [fwd[b_hi]], s=46, c="#1f6feb", zorder=4)
    ax.annotate("+2.0s", (-lft[b_lo], fwd[b_lo]), textcoords="offset points",
                xytext=(5, -8), fontsize=6.5, color="#57606a")
    ax.annotate("+6.0s", (-lft[b_hi], fwd[b_hi]), textcoords="offset points",
                xytext=(5, 2), fontsize=7, color="#1f6feb")
    ax.axhline(0, color="#e5e7eb", lw=0.8, zorder=0)
    ax.axvline(0, color="#e5e7eb", lw=0.8, zorder=0)
    span = max(12.0, float(np.nanmax(np.abs(fwd[T0_IDX:b_hi + 1]))) * 1.2)
    ax.set_xlim(-span * 0.6, span * 0.6)
    ax.set_ylim(-span * 0.18, span)
    ax.set_xlabel("← left    lateral m    right →", fontsize=6.5)
    ax.set_ylabel("forward m", fontsize=6.5)
    ax.tick_params(labelsize=6)
    ax.set_title("BEV · bold = TACTICAL band", fontsize=7)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(pad=0.3)
    bio = io.BytesIO()
    fig.savefig(bio, format="png")
    plt.close(fig)
    return bio.getvalue()


def ego_over_band(poses) -> dict:
    """The ego GEOMETRY leg, read OVER the band (2.0, 6.0] — never at a point.

    PRIMARY = `mean_band`: mean over the 40 in-band samples of the deviation
    from the BAND START (t0+20). `net_band` (the 2-point difference across the
    band) is carried so the card can show when the two disagree — that gap IS
    the interval-vs-endpoint question, made visible per clip.

    ⚠️ The yaw sign convention is MEASURED ON THE BAND QUANTITY, not inherited
    from the 2.0 s calibration: over the 201 clips, mean_band Δyaw is **+0.1660**
    for Alpamayo `Sharp Steer Left` and **−0.1565** for `Sharp Steer Right`
    (`Go Straight` +0.0065). ⇒ POSITIVE Δyaw = LEFT, on the band too.

    ⭐ THE SEAM→BAND CHANGE IS ASYMMETRIC BY TURN DIRECTION, and the sheet says so.
    MEASURED (net Δyaw, seam (0,2.0] -> band (2.0,6.0]):
      * `Steer Left`        n=24  +0.0161 -> +0.1082   (6.72x, GROWS)
      * `Sharp Steer Left`  n= 4  +0.1963 -> +0.3844   (1.96x, GROWS)
      * `Steer Right`       n=37  -0.0569 -> -0.0125   (0.22x, decays)
      * `Sharp Steer Right` n= 7  -0.5371 -> -0.2403   (0.45x, decays)
    ⛔ So "turns are already resolved by 2 s" — which an earlier draft of the
    analysis asserted — is TRUE ONLY FOR THE RIGHT-HAND CLASSES and false for the
    left-hand ones. At DYAW_THRESH 0.15 both ordinary `Steer *` CLASS MEANS fall
    under threshold; per clip it is 47/61 (77%) by `mean_band`, 36/61 by
    `net_band` — ⚠️ a class mean is NOT a per-clip fact and an earlier draft
    stated it as one ("all 61"). That is the LAT kappa mechanism, and it is NOT a
    single fixable window offset because the two directions move OPPOSITE ways.
    """
    import numpy as np

    b_lo = min(BAND_LO_IDX, len(poses) - 1)
    b_hi = min(BAND_HI_IDX, len(poses) - 1)
    v_a, yaw_a = float(poses[b_lo, 3]), float(poses[b_lo, 2])
    inb_v = poses[b_lo + 1:b_hi + 1, 3].astype(float)
    inb_y = poses[b_lo + 1:b_hi + 1, 2].astype(float)
    dv_dev = inb_v - v_a
    dyaw_dev = np.array([wrap(float(y) - yaw_a) for y in inb_y])
    dv_mean = float(dv_dev.mean()) if dv_dev.size else 0.0
    dyaw_mean = float(dyaw_dev.mean()) if dyaw_dev.size else 0.0
    dv_net = float(poses[b_hi, 3]) - v_a
    dyaw_net = wrap(float(poses[b_hi, 2]) - yaw_a)
    lat3 = ("left" if dyaw_mean > DYAW_THRESH else
            "right" if dyaw_mean < -DYAW_THRESH else "straight")
    lon3 = ("accelerate" if dv_mean > DV_THRESH else
            "decelerate" if dv_mean < -DV_THRESH else "maintain")
    lat3_net = ("left" if dyaw_net > DYAW_THRESH else
                "right" if dyaw_net < -DYAW_THRESH else "straight")
    lon3_net = ("accelerate" if dv_net > DV_THRESH else
                "decelerate" if dv_net < -DV_THRESH else "maintain")
    return {"dv_mean": dv_mean, "dyaw_mean": dyaw_mean,
            "dv_net": dv_net, "dyaw_net": dyaw_net,
            "lat3": lat3, "lon3": lon3,
            "lat3_net": lat3_net, "lon3_net": lon3_net,
            "v_t0": float(poses[T0_IDX, 3]), "v_band_lo": v_a,
            "v_band_hi": float(poses[b_hi, 3]),
            "n_in_band": int(dv_dev.size)}


def agreement(a, b, c) -> str:
    """Stratum of three opinions on one axis. `None` = the leg was SILENT and
    is counted separately — silence is never folded into disagreement."""
    spoke = [x for x in (a, b, c) if x]
    if len(spoke) < 2:
        return "insufficient"
    u = set(spoke)
    if len(u) == 1:
        return "unanimous" if len(spoke) == 3 else "agree2"
    if len(spoke) == 3 and len(u) == 3:
        return "three_way_split"
    return "two_of_three"


def main() -> int:
    import numpy as np

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(
        _PKG / "review" / "TACTICAL_VISUAL_REVIEW_BAND_2_6S.html"))
    ap.add_argument("--meta", default=str(
        _PKG / "raw" / "tacrev_selection_band.json"))
    ap.add_argument("--max-clips", type=int, default=44)
    a = ap.parse_args()

    tl = INC / "2026-08-16-tactical-labels" / "raw"
    a1 = {}
    for line in open(tl / "a1_alpamayo_taxonomy_per_clip.jsonl",
                     encoding="utf-8"):
        r = json.loads(line)
        a1[r["clip_id"]] = r
    v2raw = json.load(open(os.path.join(SP, "aug120", "merged",
                                        "ph0_v2.json"), encoding="utf-8"))
    v2raw = v2raw if isinstance(v2raw, list) else v2raw.get("clips", v2raw)
    v2_by = {r["clip_id"]: r for r in v2raw if isinstance(r, dict)}
    mp4s = find_mp4s()

    # ------------------------------------------------------------------ #
    # one row per candidate clip: three legs, both axes, OVER (2.0, 6.0]  #
    # ------------------------------------------------------------------ #
    rows, n_short = [], 0
    for cid, v2 in sorted(v2_by.items()):
        npz = os.path.join(SP, "s2_ego", "aug120", f"{cid}.npz")
        if cid not in mp4s or not os.path.exists(npz) or cid not in a1:
            continue
        poses = np.load(npz)["poses"]
        # ⛔ NO CLAMPING to a short track: a truncated band is a DIFFERENT
        # quantity, so the clip is dropped and counted, never silently shortened.
        if len(poses) <= BAND_HI_IDX:
            n_short += 1
            continue
        ego = ego_over_band(poses)
        alp = a1[cid]
        alp_lane = (alp.get("lane") or "").strip()
        alp_lat_raw = (alp.get("lateral") or "").strip()
        alp_lon_raw = (alp.get("longitudinal") or "").strip()
        # ---- Alpamayo through the POST-FIX explicit mapping -------------
        alp_lat_v6, alp_lon_v6, alp_notes = ph1_fuse.map_alpamayo_axes(
            {"lane": alp_lane or None, "longitudinal": alp_lon_raw or None})
        # ---- VLM through the POST-FIX explicit mapping ------------------
        vlm_acts, vlm_lat_v6, vlm_lon_v6 = [], None, None
        for act in ((v2.get("symbols") or {}).get("actions") or []):
            lat_t, lon_t, _n = ph1_fuse.map_vlm_action(act.get("verb"),
                                                       act.get("direction"))
            vlm_acts.append({"verb": act.get("verb"),
                             "direction": act.get("direction"),
                             "lat": lat_t, "lon": lon_t})
            vlm_lat_v6 = vlm_lat_v6 or lat_t
            vlm_lon_v6 = vlm_lon_v6 or lon_t
        egost = v2.get("ego_state") or {}
        rows.append({
            "clip_id": cid, "mp4": mp4s[cid], "npz": npz,
            "alp_lane": alp_lane, "alp_lat_raw": alp_lat_raw,
            "alp_lon_raw": alp_lon_raw, "cot": alp.get("cot") or "",
            "alp_lat_v6": alp_lat_v6, "alp_lon_v6": alp_lon_v6,
            "alp_notes": alp_notes,
            "alp_lat3": ALP_LAT3.get(alp_lat_raw.lower()),
            "alp_lon3": ALP_LON3.get(alp_lon_raw.lower()),
            "vlm_acts": vlm_acts, "vlm_lat_v6": vlm_lat_v6,
            "vlm_lon_v6": vlm_lon_v6,
            "vlm_lat3": V6_LAT3.get(vlm_lat_v6),
            "vlm_lon3": V6_LON3.get(vlm_lon_v6),
            "ego": ego,
            "ego_turning": egost.get("turning"),
            "ego_motion": egost.get("motion"),
            "ego_prompt_mode": v2.get("_ego_prompt_mode"),
            "scene": v2.get("scene") or {},
        })
    for r in rows:
        r["lat_state"] = agreement(r["alp_lat3"], r["vlm_lat3"],
                                   r["ego"]["lat3"])
        r["lon_state"] = agreement(r["alp_lon3"], r["vlm_lon3"],
                                   r["ego"]["lon3"])
        if r["alp_lane"] in ("Turn Left", "Turn Right"):
            r["stratum"] = "D_UNREPRESENTABLE"
        elif not r["alp_lane"] or not r["alp_lat_raw"]:
            r["stratum"] = "E_STOP_ROW_ONE_AXIS"
        elif "three_way_split" in (r["lat_state"], r["lon_state"]):
            r["stratum"] = "C_THREE_WAY_SPLIT"
        elif "two_of_three" in (r["lat_state"], r["lon_state"]):
            r["stratum"] = "B_TWO_OF_THREE"
        elif "unanimous" in (r["lat_state"], r["lon_state"]):
            r["stratum"] = "A_UNANIMOUS"
        else:
            r["stratum"] = "F_INSUFFICIENT"

    # ---- the Stop-class fact, RECOMPUTED ON THE BAND -------------------
    # ⚠️ the 2.0 s sheet stated this as "+2.44 m/s by the 2 s horizon". That is a
    # SEAM number; carrying it onto a band sheet would be the C89 error again.
    stop = [r for r in rows if r["alp_lon_raw"].lower() == "stop"]
    stop_fact = None
    if stop:
        stop_fact = {
            "n": len(stop),
            "v_t0": round(float(np.mean([r["ego"]["v_t0"] for r in stop])), 2),
            "v_band_lo": round(float(np.mean([r["ego"]["v_band_lo"]
                                              for r in stop])), 2),
            "v_band_hi": round(float(np.mean([r["ego"]["v_band_hi"]
                                              for r in stop])), 2),
            "dv_mean_band": round(float(np.mean([r["ego"]["dv_mean"]
                                                 for r in stop])), 2),
        }

    # ---- stratified, DETERMINISTIC selection (sorted, no RNG) ----------
    quota = {"D_UNREPRESENTABLE": 6, "E_STOP_ROW_ONE_AXIS": 4,
             "C_THREE_WAY_SPLIT": 12, "B_TWO_OF_THREE": 18,
             "A_UNANIMOUS": 12, "F_INSUFFICIENT": 2}
    by_s: dict[str, list] = collections.defaultdict(list)
    for r in sorted(rows, key=lambda r: r["clip_id"]):
        by_s[r["stratum"]].append(r)
    sel: list = []
    for s in ("D_UNREPRESENTABLE", "E_STOP_ROW_ONE_AXIS", "C_THREE_WAY_SPLIT",
              "B_TWO_OF_THREE", "A_UNANIMOUS", "F_INSUFFICIENT"):
        sel.extend(by_s[s][:quota[s]])
    sel = sel[:a.max_clips]

    # ---- render --------------------------------------------------------
    cards, ok, failed = [], 0, []
    for r in sel:
        frames = frames_from_mp4(r["mp4"], FRAME_IDXS)
        if not frames:
            failed.append(r["clip_id"])
            continue                      # ⛔ never emit a card without frames
        poses = np.load(r["npz"])["poses"]
        cards.append(render_card(r, frames, bev_png(poses)))
        ok += 1
        print(f"[tacrev-band] {ok}/{len(sel)} {r['clip_id']}", flush=True)

    counts = collections.Counter(x["stratum"] for x in sel)
    pool = collections.Counter(x["stratum"] for x in rows)
    hdr = header_html(len(rows), len(sel), ok, counts, pool, failed, stop_fact)
    doc = HTML_SHELL.replace("__BODY__", hdr + "\n".join(cards))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w", encoding="utf-8").write(doc)
    mb = os.path.getsize(a.out) / 1e6
    assert mb < 40, f"over the 40 MB budget ({mb:.1f})"
    meta = {
        "_evidence_class": "MEASURED (ours) for every leg value; the PI "
                           "verdicts this sheet collects are the first HUMAN "
                           "adjudication of any tactical label AT THE BAND.",
        "_supersedes": "TACTICAL_VISUAL_REVIEW.html (read at the 2.0 s SEAM; "
                       "C89). Kept beside this one, banner-marked.",
        "band_s": list(TAC_BAND_S), "op_band_s": list(OP_BAND_S),
        "_band_spec_source": "stack/tanitad/models/v6.py:136-140 (TAC_BAND_S)",
        "ego_statistic": "mean_band (mean in-band deviation from t0+20)",
        "t0_idx": T0_IDX, "band_lo_idx": BAND_LO_IDX,
        "band_hi_idx": BAND_HI_IDX, "fps": FPS,
        "dv_thresh_ms": DV_THRESH, "dyaw_thresh_rad": DYAW_THRESH,
        "_threshold_source": "PRODUCTION (a3:120-121) — spec lookup, NOT the "
                             "argmax cell the 2.0 s sheet used (0.75/0.05).",
        "frame_idxs": FRAME_IDXS,
        "localstorage_key": "tacrev_band_v1_",
        "_localstorage_note": "deliberately DIFFERENT from the 2.0 s sheet's "
                              "`tacrev_v1_`, so seam verdicts cannot pre-fill "
                              "or export as band verdicts.",
        "n_candidate_pool": len(rows),
        "n_dropped_track_shorter_than_band": n_short,
        "_pool_bound": "the 201 aug120 clips are ALL that have our w120 "
                       "video; of those the pool here is the subset with a "
                       "LOCAL mp4. The other 4,528 Alpamayo clips have NO "
                       "video anywhere local or on HF — a visual sheet for "
                       "them is impossible (TACTICAL_LABEL_VALIDATION.md §5.5).",
        "pool_by_stratum": dict(pool),
        "n_selected": len(sel), "n_rendered": ok,
        "selected_by_stratum": dict(counts),
        "render_failures": failed,
        "stop_class_on_the_band": stop_fact,
        "html_mb": round(mb, 2),
        "clips": [{k: v for k, v in r.items()
                   if k not in ("mp4", "npz")} for r in sel],
    }
    json.dump(meta, open(a.meta, "w", encoding="utf-8"), indent=1)
    print(f"TACREV_BAND_DONE rendered={ok} strata={dict(counts)} {mb:.2f} MB "
          f"-> {a.out}")
    return 0


# --------------------------------------------------------------------------- #
# rendering                                                                     #
# --------------------------------------------------------------------------- #
def _leg(name: str, native: str, v6: str | None, band: str | None,
         warn: str = "") -> str:
    e = html.escape
    tok = (f'<span class="tok">{e(v6)}</span>' if v6 else
           '<span class="tok none">— no token</span>')
    return (f'<div class="leg{" dep" if warn else ""}">'
            f'<div class="lname">{e(name)}{f" <sup>{e(warn)}</sup>" if warn else ""}</div>'
            f'<div class="lnat">{e(native or "silent")}</div>'
            f'{tok}<div class="lband">{e(band or "—")}</div></div>')


def render_card(r: dict, frames: list[bytes], bev: bytes) -> str:
    e = html.escape
    cid = r["clip_id"]
    times = [f"{(i - T0_IDX) / FPS:+.1f}s" for i in FRAME_IDXS]

    def cap(i: int) -> str:
        if i == T0_IDX:
            return " · t0"
        if i == BAND_LO_IDX:
            return " · BAND START"
        if i == BAND_HI_IDX:
            return " · BAND END"
        return " · in band" if BAND_LO_IDX < i < BAND_HI_IDX else ""
    imgs = "".join(
        f'<figure class="{"key" if BAND_LO_IDX <= i <= BAND_HI_IDX else ""}'
        f'{" t0f" if i == T0_IDX else ""}">'
        f'<img src="{b64img(b)}" alt="frame {t}">'
        f'<figcaption>{e(t)}{cap(i)}</figcaption></figure>'
        for b, i, t in zip(frames, FRAME_IDXS, times))
    ego = r["ego"]
    # ⭐ when mean and net disagree, SAY SO on the card — that gap is the
    # interval-vs-endpoint question made visible for this clip.
    lat_gap = ("" if ego["lat3"] == ego["lat3_net"] else
               f' <span class="gapw">⚠ endpoint says {e(ego["lat3_net"])}</span>')
    lon_gap = ("" if ego["lon3"] == ego["lon3_net"] else
               f' <span class="gapw">⚠ endpoint says {e(ego["lon3_net"])}</span>')
    lat = ('<div class="axis"><h4>LATERAL <span class="st">'
           f'{e(r["lat_state"])}</span></h4><div class="legs">'
           + _leg("Alpamayo", r["alp_lane"] or "—", r["alp_lat_v6"],
                  r["alp_lat3"])
           + _leg("VLM", ", ".join(f'{x["verb"]}/{x["direction"]}'
                                   for x in r["vlm_acts"]) or "—",
                  r["vlm_lat_v6"], r["vlm_lat3"], warn="not independent")
           + _leg("ego over 2–6 s",
                  f'mean Δyaw {ego["dyaw_mean"]:+.3f} rad · '
                  f'net {ego["dyaw_net"]:+.3f} (turning={r["ego_turning"]})',
                  None, (ego["lat3"] or "") + lat_gap, warn="not independent")
           + "</div></div>")
    lon = ('<div class="axis"><h4>LONGITUDINAL <span class="st">'
           f'{e(r["lon_state"])}</span></h4><div class="legs">'
           + _leg("Alpamayo", r["alp_lon_raw"] or "—", r["alp_lon_v6"],
                  r["alp_lon3"])
           + _leg("VLM", ", ".join(f'{x["verb"]}/{x["direction"]}'
                                   for x in r["vlm_acts"]) or "—",
                  r["vlm_lon_v6"], r["vlm_lon3"], warn="not independent")
           + _leg("ego over 2–6 s",
                  f'mean Δv {ego["dv_mean"]:+.2f} m/s · net {ego["dv_net"]:+.2f} '
                  f'({ego["v_band_lo"]:.1f}→{ego["v_band_hi"]:.1f})',
                  None, (ego["lon3"] or "") + lon_gap, warn="not independent")
           + "</div></div>")
    notes = r.get("alp_notes") or {}
    gaps = "".join(f'<li><b>{e(k)}</b>: {e(v)}</li>' for k, v in notes.items())
    sc = r["scene"]
    return f"""
<section class="clip" data-clip="{e(cid)}" data-stratum="{e(r['stratum'])}"
         data-lat="{e(r['alp_lane'] or '')}" data-lon="{e(r['alp_lon_raw'] or '')}"
         data-lattok="{e(r['alp_lat_v6'] or '')}"
         data-lontok="{e(r['alp_lon_v6'] or '')}">
  <header><h3>{e(cid)}</h3>
    <span class="badge b{e(r['stratum'][0])}">{e(r['stratum'])}</span>
    <span class="meta">{e(sc.get('illumination','?'))} ·
      {e(sc.get('weather','?'))} · {e(sc.get('road_type','?'))} ·
      v(t0)={ego['v_t0']:.1f} m/s · v(+2s)={ego['v_band_lo']:.1f} m/s</span></header>
  <div class="strip">{imgs}<figure class="bev">
    <img src="{b64img(bev,'image/png')}" alt="BEV">
    <figcaption>bold = 2→6 s band</figcaption></figure></div>
  <div class="cot"><b>Alpamayo reasoning (<code>cot</code>)</b>
    <q>{e(r['cot'])}</q></div>
  <div class="axes">{lat}{lon}</div>
  {f'<ul class="gaps">{gaps}</ul>' if gaps else ''}
  <div class="verdicts">
    <div class="vgroup"><span class="vlab">LATERAL label correct <i>for 2–6 s</i>?</span>
      {"".join(f'<label><input type="radio" name="lat_{cid}" value="{v}">{v}</label>' for v in ("correct","wrong","unsure"))}</div>
    <div class="vgroup"><span class="vlab">LONGITUDINAL label correct <i>for 2–6 s</i>?</span>
      {"".join(f'<label><input type="radio" name="lon_{cid}" value="{v}">{v}</label>' for v in ("correct","wrong","unsure"))}</div>
    <input class="note" id="n_{e(cid)}" placeholder="note — which leg is wrong, and why?">
  </div>
</section>"""


def header_html(pool_n, sel_n, ok, counts, pool, failed, stop_fact) -> str:
    e = html.escape
    strat = "".join(
        f"<tr><td><code>{e(k)}</code></td><td>{pool.get(k,0)}</td>"
        f"<td><b>{counts.get(k,0)}</b></td><td>{e(STRAT_WHY[k])}</td></tr>"
        for k in ("D_UNREPRESENTABLE", "E_STOP_ROW_ONE_AXIS",
                  "C_THREE_WAY_SPLIT", "B_TWO_OF_THREE", "A_UNANIMOUS",
                  "F_INSUFFICIENT"))
    fail = (f'<p class="warn">⚠️ {len(failed)} clip(s) dropped for missing '
            f'frames: {e(", ".join(failed))}</p>' if failed else "")
    stop_html = ""
    if stop_fact:
        stop_html = f"""
<p class="warn">⭐ <b>One thing to look for specifically — Alpamayo's
<code>Stop</code> class is a STATE, not an ACTION.</b> MEASURED on the
{stop_fact['n']} <code>Stop</code> clips in this pool: mean
<b>v(t0) = {stop_fact['v_t0']:.2f} m/s</b> — the ego is <i>already stopped</i> —
and it is at <b>{stop_fact['v_band_lo']:.2f} m/s</b> when the tactical band
opens, reaching <b>{stop_fact['v_band_hi']:.2f} m/s</b> by its end
(mean in-band Δv <b>{stop_fact['dv_mean_band']:+.2f} m/s</b>). Some
<code>cot</code> strings say verbatim <i>"Resume speed from stop since the
traffic light turns green"</i>: the label says <code>Stop</code> while its own
reasoning says the ego is <b>launching</b> — and across 2–6 s it has launched.
⇒ A naive <code>Stop → BRAKE_TO</code> mapping would be wrong on most of them.
This is why the longitudinal axis is left <code>REASON_REQUIRED</code> rather
than typed by magnitude. <b>Please adjudicate whether "Stop" is an acceptable
label for these frames.</b> ⚠️ These figures are RECOMPUTED on the band; the
2.0 s sheet quoted the seam version (+2.44 m/s), which does not describe this
window.</p>"""
    return f"""
<div class="hbanner">⭐ HORIZON = THE TACTICAL BAND <b>(2.0, 6.0] s</b> &nbsp;·&nbsp;
 <code>TAC_BAND_S</code>, <code>stack/tanitad/models/v6.py:136-140</code>
 &nbsp;·&nbsp; judge the behaviour <b>from +2.0 s to +6.0 s</b>, not at t0</div>
<h1>Tactical label review — LAT and LON, read OVER the (2.0, 6.0] s band</h1>
<p class="lede"><b>No human has ever reviewed a tactical label at the correct
horizon.</b> Every number in <code>TACTICAL_LABEL_VALIDATION.md</code> is
machine-vs-machine agreement, and the sheet you saw before this one was read at
<b>2.0 s</b> — the <i>seam</i>, not the band. Judge each axis <b>separately</b>
— a clip can be right laterally and wrong longitudinally, and one verdict
cannot say so.</p>
<div class="box err">
<h2>⛔ What was wrong with the previous sheet (C89), and what changed</h2>
<ul>
<li>The old sheet read every label at <b>2.0 s</b>. That is the <b>SEAM</b>
  where the operative band ends and the tactical band begins — the single least
  representative point in the plan for a tactical claim. It was picked because
  it was the <b>argmax of κ</b>, then described as "the v6 tactical band",
  which it is not.</li>
<li>⭐ <b>The obvious fix was also wrong.</b> The banked horizon sweep's
  "6.0 s" row is anchored at <code>t0</code>
  (<code>tac_a4_horizon_sweep.py:140-148</code>), so it measures
  <b>(0.0, 6.0]</b> — the FULL horizon, <i>including</i> the operative band.
  Neither row in that sweep is the tactical band. This sheet uses a quantity
  that did not previously exist: anchored at <b>t0+20 (2.0 s)</b> and read
  across <b>t0+21 … t0+60</b>.</li>
<li><b>Interval, not endpoint.</b> The ego leg is <code>mean_band</code> — the
  mean in-band deviation from the band start, over all 40 samples — because
  tactical behaviour <i>evolves</i> across the window. The 2-point
  <code>net</code> value is printed beside it on every card; where the two
  disagree the card says <span class="gapw">⚠ endpoint says …</span>.</li>
<li><b>Thresholds are now the PRODUCTION values</b> (Δv 1.0 m/s, Δyaw 0.15 rad),
  a spec lookup. The old sheet used 0.75 / 0.05 — the cell that
  <i>maximised κ</i>. It argmax-selected its threshold as well as its horizon.</li>
<li>⚠️ <b>Your earlier verdicts are NOT carried over</b> (separate storage key).
  They were given against different frames answering a different question.</li>
</ul>
</div>
<div class="box">
<h2>What you are looking at</h2>
<ul>
<li><b>Frames span the band.</b> <code>t0−1.0s</code> and <code>t0</code> are
  context; <b>+2.0 s → +6.0 s</b> are outlined as the window under judgement.
  The BEV draws the operative slice thin and grey and the <b>tactical band
  bold blue</b>.</li>
<li><b>Three legs, side by side</b>, each showing its NATIVE output, the v6
  token it maps to, and the 3-band projection used only for grouping.</li>
<li>⛔ <b>The VLM and ego legs are NOT independent</b> (marked
  <sup>not independent</sup>). MEASURED 201/201: the VLM's prompt contains the
  ego <code>motion</code>/<code>turning</code> fields the ego voter reads.
  VLM ↔ ego-<i>geometry</i> LAT κ <b>0.7608</b> (engine-A, whole-clip) vs
  Alpamayo ↔ VLM <b>0.1717</b>. If those two agree and Alpamayo differs, that is
  <i>one source agreeing with itself</i> — not two votes.</li>
<li><b>Alpamayo is a TEACHER, not ground truth.</b> PhysicalAI-AV is listed as
  its training data; the overlap is UNRESOLVED.</li>
<li>⚠️ <b>Alpamayo's meta-action is a decision AT t0</b>, and this sheet asks
  about <b>2–6 s</b>. Where the label looks wrong, the honest possibility is
  that the label is <i>fine for t0</i> and simply does not describe the band.
  A note saying which of the two you mean is the most useful thing you can
  leave.</li>
<li>⭐ <b>The seam→band change is ASYMMETRIC by turn direction — expect the two
  sides to fail differently.</b> MEASURED over the 201 clips (mean Δyaw,
  positive = left): <b>LEFT turns GROW</b> into the band
  (<code>Steer Left</code> +0.0161 → <b>+0.1082</b>, <b>6.72×</b>, n=24;
  <code>Sharp Steer Left</code> 1.96×, n=4) while <b>RIGHT turns DECAY</b>
  toward zero (<code>Steer Right</code> −0.0569 → −0.0125, <b>0.22×</b>, n=37;
  <code>Sharp Steer Right</code> 0.45×, n=7).
  ⇒ On a <b>right</b>-turn clip the manoeuvre may be <i>ending</i> as the band
  opens; on a <b>left</b>-turn clip it may only be <i>starting</i>.
  ⚠️ At the 0.15 rad threshold this pushes <b>47 of the 61 ordinary
  <code>Steer *</code> clips (77 %) into <code>straight</code></b> — so where the
  ego leg says <code>straight</code> on a clip you can plainly see turning, that
  is the expected failure, not a surprise. Even the <code>Sharp</code> classes
  lose about half their clips individually (2/4 and 4/7) although their class
  means clear the threshold. <i>(Two earlier drafts of this analysis were wrong
  and were corrected before you saw this sheet: one said turns are simply
  "already resolved" by 2 s — generalised from the right-hand classes and false
  for the left-hand ones; the other said <b>all</b> 61 collapse, which stated a
  class-mean as a per-clip fact.)</i></li>
<li>The VLM tokens here are the <b>POST-FIX</b> mapping. The banked corpus
  still holds the pre-fix ones — it has not been re-fused.</li>
</ul>
{stop_html}
<h2>The sample — {sel_n} selected of a {pool_n}-clip pool, {ok} rendered</h2>
<p>⚠️ A sheet of only clean cases would flatter the corpus; a sheet of only
conflicts would libel it. Both are here, labelled.</p>
<table class="strata"><tr><th>stratum</th><th>in pool</th><th>shown</th>
<th>why it is here</th></tr>{strat}</table>
<p class="warn">⚠️ <b>Pool bound:</b> only the <b>201</b> aug120 clips have our
w120 video at all, and this pool is the subset with a local copy. The other
<b>4,528</b> Alpamayo-labelled clips have no video — they cannot be reviewed
visually by any means available today.</p>{fail}
</div>
<div class="exportbar"><button onclick="doExport()">Export verdicts JSON</button>
<span id="prog"></span><textarea id="exportbox"></textarea></div>"""


STRAT_WHY = {
    "D_UNREPRESENTABLE": "Alpamayo says Turn Left/Right — ⛔ v6 has NO such "
                         "lateral action token. 186 clips corpus-wide have "
                         "nowhere to go. Shown so the gap is visible.",
    "E_STOP_ROW_ONE_AXIS": "a STOPPED vehicle emits ONE axis only; the missing "
                           "axis is NOT-APPLICABLE, never a value to impute.",
    "C_THREE_WAY_SPLIT": "all three legs speak and all three differ — the "
                         "hardest cases, and where a label is least trustworthy.",
    "B_TWO_OF_THREE": "exactly two agree. ⚠️ If the two are VLM+ego this is "
                      "the ECHO, not corroboration.",
    "A_UNANIMOUS": "all three agree. Included so the sheet cannot be accused "
                   "of showing only failures — but unanimity among "
                   "non-independent legs is weaker than it looks.",
    "F_INSUFFICIENT": "fewer than two legs spoke — an availability fact, not "
                      "a disagreement.",
}

HTML_SHELL = """<!doctype html><html><head><meta charset="utf-8">
<title>TanitAD — tactical label review (2.0–6.0 s BAND)</title>
<style>
:root{--ok:#1a7f37;--bad:#cf222e;--mid:#9a6700;--line:#d0d7de;--ink:#1f2328}
*{box-sizing:border-box}
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);
 margin:0;padding:22px;background:#f6f8fa;max-width:1500px}
h1{font-size:23px;margin:0 0 6px} h2{font-size:15px;margin:16px 0 6px}
.lede{font-size:15px;max-width:1000px}
.hbanner{background:#0b3d91;color:#fff;padding:10px 14px;border-radius:8px;
 font-size:14.5px;margin:0 0 14px;letter-spacing:.01em}
.hbanner code{background:rgba(255,255,255,.16);padding:1px 5px;border-radius:3px}
.box{background:#fff;border:1px solid var(--line);border-radius:8px;
 padding:14px 18px;margin:14px 0 22px}
.box.err{border-color:#cf222e33;background:#fff7f7}
.box ul{margin:6px 0;padding-left:20px} .box li{margin:4px 0}
.warn{color:#7d4e00;background:#fff8e5;border-left:3px solid #d4a72c;
 padding:7px 10px;margin:8px 0}
table.strata{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}
table.strata th,table.strata td{border:1px solid var(--line);padding:5px 8px;
 text-align:left;vertical-align:top}
table.strata th{background:#f6f8fa}
.exportbar{position:sticky;top:0;z-index:9;background:#f6f8fa;padding:8px 0;
 border-bottom:1px solid var(--line)}
.exportbar button{font-size:14px;padding:7px 14px;cursor:pointer;
 border:1px solid var(--line);border-radius:6px;background:#fff}
#exportbox{display:none;width:100%;height:170px;margin-top:8px;
 font-family:ui-monospace,Consolas,monospace;font-size:11px}
#prog{margin-left:12px;color:#57606a}
section.clip{background:#fff;border:1px solid var(--line);border-radius:8px;
 padding:14px;margin:0 0 20px}
section.clip header{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
section.clip h3{margin:0;font:600 13px ui-monospace,Consolas,monospace}
.badge{font-size:11px;padding:2px 7px;border-radius:10px;background:#eaeef2}
.badge.bD{background:#ffe9e9;color:#8b1a1a}.badge.bC{background:#fff1e0;color:#8a4b00}
.badge.bB{background:#fff8e5;color:#7d4e00}.badge.bA{background:#e7f5ec;color:#0f5132}
.meta{font-size:12px;color:#57606a}
.strip{display:flex;gap:6px;margin:10px 0;overflow-x:auto;padding-bottom:4px}
.strip figure{margin:0;flex:0 0 auto}
.strip img{height:132px;display:block;border-radius:4px;border:2px solid transparent}
.strip figure.key img{border-color:#1f6feb}
.strip figure.t0f img{border-color:#8c959f}
.strip figcaption{font-size:10.5px;color:#57606a;text-align:center;margin-top:2px}
.strip figure.bev img{height:132px;border:1px solid var(--line);background:#fff}
.cot{background:#f6f8fa;border-left:3px solid #1f6feb;padding:7px 11px;
 margin:6px 0;font-size:13px}
.cot q{display:block;margin-top:2px;font-style:italic}
.axes{display:flex;gap:14px;flex-wrap:wrap;margin-top:10px}
.axis{flex:1 1 400px;border:1px solid var(--line);border-radius:6px;padding:9px}
.axis h4{margin:0 0 7px;font-size:12.5px;letter-spacing:.04em}
.st{font-weight:400;color:#57606a;font-size:11.5px;margin-left:6px}
.legs{display:flex;gap:7px}
.leg{flex:1;background:#fafbfc;border:1px solid var(--line);border-radius:5px;
 padding:6px 8px;font-size:12px}
.leg.dep{border-style:dashed;border-color:#d4a72c}
.lname{font-weight:600;font-size:11px;text-transform:uppercase;color:#57606a}
.lname sup{font-weight:400;text-transform:none;color:#9a6700;font-size:9px}
.lnat{margin:2px 0;font-size:11.5px}
.tok{display:inline-block;font:600 11px ui-monospace,Consolas,monospace;
 background:#ddf4ff;padding:1px 5px;border-radius:3px}
.tok.none{background:#ffebe9;color:#8b1a1a;font-weight:400}
.lband{font-size:10.5px;color:#57606a;margin-top:3px}
.gapw{color:#8a4b00;background:#fff1e0;padding:0 4px;border-radius:3px}
.gaps{margin:8px 0 0;padding-left:20px;font-size:12px;color:#7d4e00}
.verdicts{display:flex;gap:16px;align-items:center;flex-wrap:wrap;
 margin-top:11px;padding-top:10px;border-top:1px dashed var(--line)}
.vlab{font-size:11.5px;font-weight:600;color:#57606a;margin-right:6px}
.vgroup label{margin-right:8px;font-size:12.5px;cursor:pointer;
 border:1px solid var(--line);border-radius:5px;padding:3px 8px}
.vgroup label:has(input[value=correct]:checked){background:#e7f5ec;border-color:var(--ok)}
.vgroup label:has(input[value=wrong]:checked){background:#ffebe9;border-color:var(--bad)}
.vgroup label:has(input[value=unsure]:checked){background:#fff8e5;border-color:var(--mid)}
.note{flex:1 1 300px;padding:5px 8px;border:1px solid var(--line);border-radius:5px}
</style></head><body>__BODY__
<script>
/* ⛔ DELIBERATELY different from the 2.0 s sheet's 'tacrev_v1_'. Sharing it
   would pre-fill this sheet with verdicts given at the WRONG horizon and
   export them as if they judged these frames. */
const KEY='tacrev_band_v1_';
function save(cid){
 const g=n=>{const el=document.querySelector(`input[name="${n}_${cid}"]:checked`);
  return el?el.value:null;};
 localStorage.setItem(KEY+cid,JSON.stringify({lat:g('lat'),lon:g('lon'),
  note:(document.getElementById('n_'+cid)||{}).value||''}));
 prog();}
function prog(){let d=0;const all=document.querySelectorAll('.clip[data-clip]');
 all.forEach(el=>{const s=localStorage.getItem(KEY+el.dataset.clip);
  if(s){const o=JSON.parse(s);if(o.lat||o.lon)d++;}});
 document.getElementById('prog').textContent=d+' / '+all.length+' clips judged';}
document.querySelectorAll('.clip[data-clip]').forEach(el=>{
 const cid=el.dataset.clip;
 const s=localStorage.getItem(KEY+cid);
 if(s){const o=JSON.parse(s);
  ['lat','lon'].forEach(ax=>{if(o[ax]){const r=el.querySelector(
   `input[name="${ax}_${cid}"][value="${o[ax]}"]`);if(r)r.checked=true;}});
  const n=document.getElementById('n_'+cid);if(n&&o.note)n.value=o.note;}
 el.querySelectorAll('input').forEach(i=>{
  i.addEventListener('change',()=>save(cid));
  i.addEventListener('input',()=>save(cid));});});
function doExport(){
 const out={_sheet:'TanitAD tactical label review (LAT/LON over the (2.0,6.0] s BAND)',
  _band_s:[2.0,6.0],_band_spec:'v6.py:136-140 TAC_BAND_S',
  _ego_statistic:'mean_band (mean in-band deviation from t0+20)',
  _supersedes:'the 2.0 s SEAM sheet (C89) — verdicts are NOT shared',
  _exported:new Date().toISOString(),
  _note:'lat and lon are judged SEPARATELY; null = UNREVIEWED, not agreement',
  verdicts:{}};
 document.querySelectorAll('.clip[data-clip]').forEach(el=>{
  const s=localStorage.getItem(KEY+el.dataset.clip);
  out.verdicts[el.dataset.clip]=Object.assign(
   {stratum:el.dataset.stratum,
    alp_lane_raw:el.dataset.lat||null,alp_lon_raw:el.dataset.lon||null,
    alp_lat_v6:el.dataset.lattok||null,alp_lon_v6:el.dataset.lontok||null},
   s?JSON.parse(s):{lat:null,lon:null,note:''});});
 const box=document.getElementById('exportbox');
 box.style.display='block';box.value=JSON.stringify(out,null,1);
 box.select();try{document.execCommand('copy');}catch(e){}}
prog();
</script></body></html>"""


if __name__ == "__main__":
    raise SystemExit(main())
