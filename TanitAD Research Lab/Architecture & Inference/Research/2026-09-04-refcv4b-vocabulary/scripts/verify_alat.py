"""Verify the decoder's `alat` roll == the offline derivation, and re-measure
turn coverage + Kamm for the SHIPPED speed-clamped family."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

STACK = r"C:\Users\Admin\run_refcv4v\repo\stack"
sys.path.insert(0, STACK)
from tanitad.refs import refc_v3                                # noqa: E402
from tanitad.refs.refa_v1_plan import unicycle_paths            # noqa: E402
from tanitad.refs.refc_v3 import V3_HORIZONS                    # noqa: E402
import tanitad.refs.refc_select as sl                           # noqa: E402
import tanitad                                                  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DUMP = os.path.join(HERE, "vocab", "refcv3_40284_dump")
OUT = os.path.join(HERE, "build", "out")
DT, H = 0.1, max(V3_HORIZONS)
SLOTS = [k - 1 for k in V3_HORIZONS]
V_FLOOR, KAPPA_CAP, REF = 4.0, 0.12, 10.0

print("[tree]", tanitad.__file__)
f = torch.load(os.path.join(OUT, "refc_anchors_6s_v0cond_alat_117.pt"),
               map_location="cpu", weights_only=True)
A, C = f["anchors"], f["controls"]
N = A.shape[0]

cfg = refc_v3.refc_v3_smoke_config(hier=True)
cfg.core.anchors.n_anchors = N
cfg.core.anchors.v0_conditioned = True
cfg.core.anchors.control_units = "alat"
m = refc_v3.RefCV3Model(cfg).eval()
m.core.decoder.load_anchors(A, C)
print("[assert] torch.equal(decoder.anchors,          file) =",
      torch.equal(m.core.decoder.anchors, A))
print("[assert] torch.equal(decoder.anchor_controls,  file) =",
      torch.equal(m.core.decoder.anchor_controls, C))


def offline(v0):
    kap = np.clip(C[:, 1].numpy() / max(float(v0), V_FLOOR) ** 2,
                  -KAPPA_CAP, KAPPA_CAP)
    ct = torch.from_numpy(np.stack([C[:, 0].numpy(), kap], -1)).float()
    return unicycle_paths(ct[:, None, :].expand(-1, H, -1).contiguous(),
                          torch.tensor(float(v0)), DT,
                          action_units="kappa")[:, SLOTS]


print("\n=== decoder roll vs the offline derivation ===")
vs = torch.tensor([1.0, 4.0, 10.0, 18.0, 27.0, 36.0])
bank = m.core.decoder.roll_bank(vs, None, len(vs), torch.float32)
for i, v in enumerate(vs):
    d = (bank[i] - offline(v)).abs().max().item()
    print("  v0=%5.1f  max|decoder - offline| = %.3e" % (v, d))
    assert d < 1e-5, d
t = torch.tensor([k * DT for k in V3_HORIZONS])
zi = int(((C[:, 0] == 0) & (C[:, 1] == 0)).nonzero()[0, 0])
for i, v in enumerate(vs):
    cv = torch.stack([v * t, torch.zeros_like(t)], -1)
    print("  v0=%5.1f  {0,0} vs (v0*t, 0): %.3e m"
          % (v, (bank[i, zi] - cv).abs().max().item()))
print("  ref-speed roll vs decoder@%.1f: %.3e"
      % (REF, (bank[2] - A).abs().max().item()))

# ---- turn coverage + Kamm on the 4,823-window surface --------------------
eps = sorted(x for x in os.listdir(DUMP) if x.startswith("ep") and x.endswith(".npz"))
V0 = np.concatenate([np.load(os.path.join(DUMP, x))["v0"] for x in eps]).astype(np.float64)
NW = len(V0)
print("\n=== reach clamp + turn coverage, %d windows ===" % NW)


def kamm_peak(P):
    tt = np.array(V3_HORIZONS, np.float64) * DT
    v = np.gradient(P, tt, axis=1)
    a = np.gradient(v, tt, axis=1)
    sp = np.linalg.norm(v, axis=-1)
    lon = np.abs((v * a).sum(-1) / np.maximum(sp, 1e-6))
    lat = np.abs((v[..., 0] * a[..., 1] - v[..., 1] * a[..., 0])
                 / np.maximum(sp, 1e-6))
    return (np.sqrt(lon ** 2 + lat ** 2) / 9.81).max(1)


key = np.round(V0, 3)
cache = {k: offline(k).numpy().astype(np.float64) for k in np.unique(key)}
res = {}
for amax in (2.0, 2.5, 1.5):
    surv = np.empty(NW); t_eb = np.empty(NW); t_th = np.empty(NW); empty = 0
    for i in range(NW):
        P = cache[key[i]]
        msk = sl.anchor_reachability_mask(
            torch.from_numpy(P)[None],
            torch.tensor([V0[i]], dtype=torch.float64),
            accel_max=amax, horizon_s=6.0)[0].numpy()
        s = int(msk.sum()); surv[i] = s
        if s == 0:
            empty += 1; t_eb[i] = t_th[i] = 0; continue
        Q = P[msk]
        eb = np.degrees(np.abs(np.arctan2(Q[:, -1, 1], Q[:, -1, 0])))
        seg = Q[:, -1] - Q[:, -2]
        th = np.degrees(np.abs(np.arctan2(seg[:, 1], seg[:, 0])))
        t_eb[i] = int((eb > 30).sum()); t_th[i] = int((th > 30).sum())
    row = {"accel_max": amax,
           "killed_pct": float((N - surv.mean()) / N * 100.0),
           "empty_pct": float(empty / NW * 100.0),
           "survivors_per_window": float(surv.mean()),
           "turns_gt30_end_bearing_per_window": float(t_eb.mean()),
           "turns_gt30_terminal_heading_per_window": float(t_th.mean()),
           "windows_with_no_gt30_turn_pct": float((t_eb == 0).mean() * 100.0)}
    res["a_max_%.1f" % amax] = row
    print("  a_max %.1f  killed %6.2f%%  empty %.2f%%  surv/win %6.1f  "
          ">30deg %5.1f / %5.1f  NO-turn %.2f%%"
          % (amax, row["killed_pct"], row["empty_pct"],
             row["survivors_per_window"],
             row["turns_gt30_end_bearing_per_window"],
             row["turns_gt30_terminal_heading_per_window"],
             row["windows_with_no_gt30_turn_pct"]))

print("\n=== Kamm circle, SHIPPED speed-clamped family ===")
kam = {}
for q, lab in ((5, "p5"), (50, "median"), (95, "p95"), (100, "max")):
    v = float(np.percentile(V0, q))
    pk = kamm_peak(offline(v).numpy().astype(np.float64))
    kam[lab] = {"v0_ms": v, "over_mu_0.7": int((pk > 0.7).sum()),
                "over_mu_1.0": int((pk > 1.0).sum()), "peak_g": float(pk.max())}
    print("  v0 %-6s = %5.2f m/s : over mu=0.7 %3d/%d  over mu=1.0 %3d  peak %.2f g"
          % (lab, v, int((pk > 0.7).sum()), N, int((pk > 1.0).sum()), pk.max()))

json.dump({"family": "v0-conditioned, SPEED-CLAMPED (a_lon, a_lat), 13x9 = 117, "
                     "a_lat_max 3.0 m/s^2",
           "surface": {"windows": NW, "episodes": len(eps)},
           "clamp": res, "kamm": kam,
           "_incumbents_for_scale": {
               "refcv3 synthetic @ 2.5": {"killed_pct": 37.10,
                                          "no_turn_pct": 0.00,
                                          "kamm_over_mu_0.7": 43,
                                          "peak_g": 1.50},
               "refcv4 fixed-path @ 2.0 (ABORTED)": {"killed_pct": 26.60,
                                                     "no_turn_pct": 4.62,
                                                     "kamm_over_mu_0.7": 0,
                                                     "peak_g": 0.44},
               "refcv4b flat-kappa (rejected)": {"killed_pct": 12.61,
                                                 "no_turn_pct": 0.70,
                                                 "kamm_over_mu_0.7_at_27ms": 104,
                                                 "peak_g": 3.96}}},
          open(os.path.join(OUT, "COVERAGE6S_ALAT.json"), "w"), indent=1)
print("\nwrote " + os.path.join(OUT, "COVERAGE6S_ALAT.json"))
