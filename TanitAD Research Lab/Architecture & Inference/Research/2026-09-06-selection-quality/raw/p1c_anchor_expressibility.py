"""P1c - CAN ANY OF THE 117 CANDIDATES EXPRESS A LANE CHANGE?  (model-free)

The brief's explicit test.  Nothing here needs a forward pass: the candidate
fan is a deterministic function of (anchor_controls, v0) through the
programme's own integrator, so this is a SUPPLY/CEILING statement about the
vocabulary and may never be compared against an achievement.

CONTRACT READ FROM SOURCE BEFORE ANY ARITHMETIC
-----------------------------------------------
`refc.py::RefCDecoder` (`_anchor_bank`):
  ctrl[b, n, k, :] = (a_lon[n], kappa[b, n])  -- THE SAME CONTROL AT EVERY STEP
  kappa[b, n] = clamp(a_lat[n] / max(v0_b, alat_v_floor)^2, +-kappa_cap)
`kinematic.rollout_unicycle`: yaw_{k+1} = yaw_k + v_k * kappa * dt.
UNITS (refc.py, `anchor_controls` doc): channel 0 = accel m/s^2,
channel 1 = LATERAL ACCELERATION m/s^2 when control_units == "alat".

=> kappa is CONSTANT and its SIGN IS FIXED within a candidate, so yaw(t) is
MONOTONE for every one of the 117.  A lane change is by definition
non-monotone in heading: depart the lane heading, then return to it, ending
with ~one lane width of lateral offset and ~zero net heading change.

DEFINITIONS, DECLARED BEFORE ANY NUMBER (following the sibling turn-coverage
instrument's convention).  For a candidate path over 0..T in the ego frame at t0:
  lat_T   = y(T)                       signed lateral offset, metres
  dyaw_T  = yaw(T) - yaw(0)            net heading change, integrated exactly
  LANE-CHANGE BOX : |lat_T| in [2.5, 5.0) m  AND  |dyaw_T| <= 10 deg
  TURN BOX        : |dyaw_T| >= 30 deg                       (the CONTROL box;
                    it MUST read non-zero or the geometry code is broken)
  STRAIGHT        : |lat_T| < 0.25 m AND |dyaw_T| < 1 deg
"""
import json
import sys

import numpy as np
import torch

import _env  # noqa: F401
import load

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CKPT = r"C:\Users\Admin\refcv4b_final\ckpt_40284_FINAL.pt"
CFG = r"C:\Users\Admin\refcv4b_final\config.json"
SIB = (r"G:\Meine Ablage\SayBouBase\raw\Projects\TanitAD\TanitAD Research Lab"
       r"\Architecture & Inference\Research\2026-09-04-refcv4b-turn-coverage"
       r"\raw\anchors_live_refcv4b.pt")

LANE_LO, LANE_HI = 2.5, 5.0
LC_MAX_DYAW_DEG = 10.0
TURN_MIN_DYAW_DEG = 30.0


def controls_from_ckpt():
    """anchor_controls straight out of the refcv4b checkpoint (the authority)."""
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    for outer in ("model", "state_dict", "ema", None):
        d = sd.get(outer) if outer else sd
        if not isinstance(d, dict):
            continue
        hits = [k for k in d if k.endswith("anchor_controls")]
        if hits:
            return d[hits[0]].float(), f"{outer or '<root>'}:{hits[0]}"
    raise SystemExit("anchor_controls not found in the checkpoint")


def roll(ctrl_alat, a_lon, v0, kappa_cap, v_floor, steps=60, dt=0.1):
    """The programme's own integrator, vectorised over (window, candidate)."""
    from tanitad.models.kinematic import rollout_unicycle
    B, N = len(v0), len(a_lon)
    vv = np.maximum(v0, v_floor) ** 2
    kap = np.clip(ctrl_alat[None, :] / vv[:, None], -kappa_cap, kappa_cap)
    ctrl = np.stack([np.broadcast_to(a_lon[None, :], (B, N)), kap], -1)
    ctrl = torch.as_tensor(ctrl, dtype=torch.float32)[:, :, None, :]
    ctrl = ctrl.expand(B, N, steps, 2).reshape(-1, steps, 2)
    s0 = torch.zeros(B * N, 4, dtype=torch.float32)
    s0[:, 3] = torch.as_tensor(np.repeat(v0, N), dtype=torch.float32)
    st = rollout_unicycle(s0, ctrl, dt=dt)           # [B*N, steps, 4]
    return st.reshape(B, N, steps, 4)


def main():
    with open(CFG, encoding="utf-8") as fh:
        cfg = json.load(fh)
    argv = " ".join(cfg.get("argv", [])) if isinstance(cfg.get("argv"), list) \
        else str(cfg.get("argv"))
    ctrl, where = controls_from_ckpt()
    sib = torch.load(SIB, map_location="cpu", weights_only=False)["controls"]
    same = bool(torch.allclose(ctrl, sib.float(), atol=0, rtol=0))
    print(f"anchor_controls from the CHECKPOINT at {where}  shape "
          f"{tuple(ctrl.shape)}")
    print(f"  byte-identical to the sibling's banked anchors_live_refcv4b.pt "
          f"'controls': {same}")
    a_lon = ctrl[:, 0].numpy().astype(np.float64)
    a_lat = ctrl[:, 1].numpy().astype(np.float64)
    print(f"  a_lon grid  ({len(np.unique(a_lon))} distinct): "
          f"{np.round(np.unique(a_lon), 4).tolist()}")
    print(f"  a_lat grid  ({len(np.unique(a_lat))} distinct): "
          f"{np.round(np.unique(a_lat), 4).tolist()}")
    print(f"  => {len(np.unique(a_lon))} x {len(np.unique(a_lat))} = "
          f"{len(np.unique(a_lon)) * len(np.unique(a_lat))} candidates")
    for k in ("--anchor-control-units", "--anchor-kappa-cap",
              "--anchor-alat-v-floor", "--anchor-ref-speed",
              "--anchors-v0-conditioned"):
        i = argv.find(k)
        print(f"  run argv {k:<28} "
              f"{argv[i:i+len(k)+12] if i >= 0 else 'ABSENT'}")

    kappa_cap, v_floor = 0.12, 4.0
    D = load.load_all()
    v0 = D["v0"].astype(np.float64)
    print(f"\ncorpus v0 (the SAME 141-episode / 4823-window eval grid): "
          f"n={len(v0)}  median {np.median(v0):.2f} m/s  "
          f"p05 {np.percentile(v0,5):.2f}  p95 {np.percentile(v0,95):.2f}")

    st = roll(a_lat, a_lon, v0, kappa_cap, v_floor)          # [B,117,60,4]
    T6 = 59                                                   # 6.0 s
    lat6 = st[:, :, T6, 1].numpy().astype(np.float64)
    dyaw6 = np.degrees(st[:, :, T6, 2].numpy().astype(np.float64))
    T4 = 39                                                   # 4.0 s
    lat4 = st[:, :, T4, 1].numpy().astype(np.float64)
    dyaw4 = np.degrees(st[:, :, T4, 2].numpy().astype(np.float64))

    # --- controls that must read known values ------------------------------
    i0 = int(np.argmin(np.abs(a_lat)))
    print(f"\nCONTROLS")
    print(f"  K-straight  candidate #{i0} (a_lat={a_lat[i0]:+.3f}): "
          f"max|y(6s)| = {np.abs(lat6[:, i0]).max():.3e} m, "
          f"max|dyaw(6s)| = {np.abs(dyaw6[:, i0]).max():.3e} deg  "
          f"(must be EXACTLY 0)")
    # closed form for the extreme-curvature candidate at a fixed v0
    iX = int(np.lexsort((-np.abs(a_lat), np.abs(a_lon)))[0])
    vtest = 10.0
    kap = min(abs(a_lat[iX]) / max(vtest, v_floor) ** 2, kappa_cap)
    closed = np.degrees(vtest * kap * 6.0)
    stX = roll(np.array([a_lat[iX]]), np.array([0.0]), np.array([vtest]),
               kappa_cap, v_floor)
    print(f"  K-closedform candidate a_lat={a_lat[iX]:+.3f}, a_lon forced 0, "
          f"v0=10 m/s: closed form |dyaw(6s)| = {closed:.3f} deg, "
          f"integrated = {abs(np.degrees(stX[0,0,T6,2].item())):.3f} deg")
    mono = 0
    dy = np.degrees(st[:, :, :, 2].numpy())
    d1 = np.diff(dy, axis=2)
    mono = float(((d1 >= -1e-9).all(2) | (d1 <= 1e-9).all(2)).mean())
    print(f"  K-monotone   fraction of (window, candidate) pairs whose yaw is "
          f"MONOTONE over 0-6 s: {mono:.6f}  (source says it must be 1.0)")

    # --- the boxes ----------------------------------------------------------
    for tag, lat, dyaw in (("6.0 s", lat6, dyaw6), ("4.0 s", lat4, dyaw4)):
        lc = (np.abs(lat) >= LANE_LO) & (np.abs(lat) < LANE_HI) & \
             (np.abs(dyaw) <= LC_MAX_DYAW_DEG)
        turn = np.abs(dyaw) >= TURN_MIN_DYAW_DEG
        strt = (np.abs(lat) < 0.25) & (np.abs(dyaw) < 1.0)
        wide = (np.abs(lat) >= LANE_LO) & (np.abs(dyaw) <= LC_MAX_DYAW_DEG)
        print(f"\nHORIZON {tag}  ({lat.shape[0]} windows x {lat.shape[1]} "
              f"candidates = {lat.size} pairs)")
        print(f"  LANE-CHANGE BOX  |lat| in [{LANE_LO},{LANE_HI}) m AND "
              f"|dyaw| <= {LC_MAX_DYAW_DEG} deg : {int(lc.sum())} / {lat.size}"
              f"   windows with >=1 such candidate: "
              f"{int(lc.any(1).sum())} / {lat.shape[0]}")
        print(f"  (relaxed: |lat| >= {LANE_LO} m AND |dyaw| <= "
              f"{LC_MAX_DYAW_DEG} deg, NO upper bound on offset): "
              f"{int(wide.sum())} / {lat.size}")
        print(f"  TURN BOX  |dyaw| >= {TURN_MIN_DYAW_DEG} deg  "
              f"[non-zero CONTROL]        : {int(turn.sum())} / {lat.size}"
              f"   windows with >=1: {int(turn.any(1).sum())} / "
              f"{lat.shape[0]}")
        print(f"  STRAIGHT                                            "
              f"    : {int(strt.sum())} / {lat.size}")
        # what does a candidate that DOES reach a lane width look like?
        reach = np.abs(lat) >= LANE_LO
        if reach.any():
            print(f"  |dyaw| on the pairs that DO reach >= {LANE_LO} m of "
                  f"lateral offset: n={int(reach.sum())}  "
                  f"min {np.abs(dyaw)[reach].min():.2f} deg  "
                  f"p05 {np.percentile(np.abs(dyaw)[reach],5):.2f}  "
                  f"median {np.median(np.abs(dyaw)[reach]):.2f}")
    out = {
        "vocabulary": {"n": int(ctrl.shape[0]),
                       "a_lon_grid": np.unique(a_lon).round(6).tolist(),
                       "a_lat_grid": np.unique(a_lat).round(6).tolist(),
                       "source": where, "ckpt": CKPT,
                       "matches_sibling_banked_anchors": same,
                       "kappa_cap": kappa_cap, "alat_v_floor": v_floor},
        "controls": {"straight_max_abs_y_m": float(np.abs(lat6[:, i0]).max()),
                     "straight_max_abs_dyaw_deg":
                         float(np.abs(dyaw6[:, i0]).max()),
                     "monotone_yaw_fraction": mono},
        "boxes_6s": {
            "lane_change_pairs": int((((np.abs(lat6) >= LANE_LO) &
                                       (np.abs(lat6) < LANE_HI) &
                                       (np.abs(dyaw6) <= LC_MAX_DYAW_DEG))
                                      ).sum()),
            "lane_change_relaxed_pairs":
                int(((np.abs(lat6) >= LANE_LO) &
                     (np.abs(dyaw6) <= LC_MAX_DYAW_DEG)).sum()),
            "turn_pairs": int((np.abs(dyaw6) >= TURN_MIN_DYAW_DEG).sum()),
            "n_pairs": int(lat6.size), "n_windows": int(lat6.shape[0])},
    }
    with open("out_p1c_anchor_expressibility.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("\nwrote out_p1c_anchor_expressibility.json")


if __name__ == "__main__":
    main()
