"""E-DETECT-1 — the ORACLE arm: an instrument-validity control.

⛔⛔ WITHOUT THIS ARM A NULL IS UNINTERPRETABLE. If every real arm lands at or
below the `prior` floor, there are two completely different explanations and the
measurement so far cannot separate them:

  (a) the representations carry no localisable vehicles;
  (b) the HEAD cannot learn this task from 5,617 frames / 130 clips at all,
      whatever it is fed — i.e. the instrument is data-starved.

The oracle settles it. It hands the head a token field that DOES contain the
vehicles, in the same shape a working encoder would produce, and asks whether the
head can then do the job. Fails -> (b), and no representation claim is admissible.
Passes -> (a) is on the table and the nulls mean something.

⭐ IT IS DELIBERATELY NOT EASY, and that is the whole design.

  * Vehicles are written into IMAGE space through the corpus's REAL cylindrical
    projection (`f_ref` 305.577 px, 640 x 256 -> +-60 deg, i.e. the 120-deg
    front-wide rig), on the SAME 16 x 40 token grid the encoders use.
  * ⛔ **RANGE IS WITHHELD.** A token says "a vehicle of this size and heading
    is HERE in the image" and nothing more. The head must recover depth from the
    token's ROW — the monocular elevation cue — exactly as it must for a real
    encoder. Handing over `log(range)` would have made the control a lookup
    table and would have proven nothing about the inversion.
  * The per-token payload is a FIXED RANDOM PROJECTION of (presence, sin/cos
    yaw, length, width) into 64 dims, so the head cannot read a raw field off a
    known channel.

⚠️ A PASS DOES NOT VALIDATE THE HEAD FOR ALL INPUTS. It shows the head can invert
this projection given clean per-token evidence. A real encoder's evidence is
noisier and entangled; the oracle is an UPPER bound on the head, not a promise.

TIER: T0-DIAGNOSTIC.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import e_detect_prep as P

#: the corpus's own rig, read off the episode cache's `frame` dict
F_REF = 305.5774907364391
IMG_W, IMG_H = 640, 256
#: ego-frame camera height (m). Only sets WHERE the horizon sits; the head has
#: to learn the row->range map regardless of the constant.
CAM_H = 1.5
D_ORACLE = 64
#: image-space spread of a vehicle's evidence, in TOKENS. A delta would let the
#: head memorise exact cells; real encoders smear evidence over neighbours.
SIGMA_TOK = 0.8


def project(cx: np.ndarray, cy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ego BEV -> cylindrical image (u, v). x forward, y LEFT."""
    az = np.arctan2(-cy, cx)                       # +az to the right in image
    u = IMG_W / 2.0 + F_REF * az
    rng = np.hypot(cx, cy)
    v = IMG_H / 2.0 + F_REF * CAM_H / np.maximum(rng, 1e-3)
    return u, v


def main() -> None:
    keys = [tuple(k) for k in
            json.loads((P.FEAT / "keys.json").read_text(encoding="utf-8"))]
    want = {k: i for i, k in enumerate(keys)}
    rng_proj = np.random.default_rng(0)
    # (presence, sin yaw, cos yaw, l, w) -> D_ORACLE. Fixed, shared by all rows.
    W = rng_proj.standard_normal((5, D_ORACLE)).astype(np.float32) / np.sqrt(5)

    out = np.zeros((len(keys), P.N_TOK, D_ORACLE), dtype=np.float32)
    gr, gc = np.meshgrid(np.arange(P.GRID_H), np.arange(P.GRID_W), indexing="ij")
    gr = gr.ravel().astype(np.float32)
    gc = gc.ravel().astype(np.float32)

    n_written = 0
    n_offscreen = 0
    for line in P.JOIN.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        i = want.get((r["clip_id"], int(r["frame_idx"])))
        if i is None:
            continue
        veh = [a for a in r.get("agents", ())
               if a.get("cls") in P.VEHICLE
               and P.cell_of(float(a["cx"]), float(a["cy"])) is not None]
        if not veh:
            continue
        cx = np.array([a["cx"] for a in veh], dtype=np.float32)
        cy = np.array([a["cy"] for a in veh], dtype=np.float32)
        yaw = np.array([a.get("yaw", 0.0) for a in veh], dtype=np.float32)
        ln = np.array([a.get("l", 4.5) for a in veh], dtype=np.float32)
        wd = np.array([a.get("w", 1.9) for a in veh], dtype=np.float32)
        u, v = project(cx, cy)
        tu, tv = u / P.PATCH, v / P.PATCH               # token coordinates
        on = (tu >= 0) & (tu < P.GRID_W) & (tv >= 0) & (tv < P.GRID_H)
        n_offscreen += int((~on).sum())
        if not on.any():
            continue
        # ⛔ NO RANGE CHANNEL — the head recovers depth from the row it lands on.
        code = np.stack([np.ones(on.sum(), np.float32), np.sin(yaw[on]),
                         np.cos(yaw[on]), ln[on] / 5.0, wd[on] / 2.0], 1) @ W
        d2 = ((gr[None, :] - tv[on, None]) ** 2
              + (gc[None, :] - tu[on, None]) ** 2)
        wgt = np.exp(-d2 / (2 * SIGMA_TOK ** 2)).astype(np.float32)
        out[i] = wgt.T @ code
        n_written += 1

    path = P.OUT / "oracle.npy"
    np.save(path, out)
    live = int((np.abs(out).reshape(len(out), -1).max(1) > 0).sum())
    stats = {
        "_evidence_class": "MEASURED (ours; synthetic control bank)",
        "purpose": "instrument-validity control: can the HEAD do this task at "
                   "all, given evidence that provably contains the answer?",
        "d": D_ORACLE, "n_tok": P.N_TOK, "grid": [P.GRID_H, P.GRID_W],
        # ⚠️ CYLINDRICAL, so u is LINEAR IN AZIMUTH: az_max = (W/2)/f_ref.
        # The pinhole formula 2*atan((W/2)/f) is wrong here and understates the
        # rig by ~27 deg (92.6 vs 120) — the same "right formula, wrong
        # projection" scope error the trap list keeps collecting.
        "f_ref_px": F_REF, "fov_deg": round(
            float(2 * np.degrees(IMG_W / 2 / F_REF)), 2),
        "cam_h_m": CAM_H, "sigma_tokens": SIGMA_TOK,
        "range_channel_withheld": True,
        "frames_written": n_written, "frames_nonzero": live,
        "vehicles_projected_offscreen": n_offscreen,
        "rows": len(keys),
    }
    (P.OUT / "oracle_stats.json").write_text(json.dumps(stats, indent=1),
                                             encoding="utf-8")
    print(json.dumps(stats, indent=1))
    if live < 0.9 * len(keys):
        raise SystemExit(f"[FATAL] only {live}/{len(keys)} rows carry evidence "
                         "— the projection or the class filter is wrong, and a "
                         "control that is mostly empty proves nothing")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
